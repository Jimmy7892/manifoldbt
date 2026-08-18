"""Turn a results JSON into the Markdown a human reads.

Prints to stdout and, when running under GitHub Actions, appends the same text
to the job summary so the numbers are visible without downloading an artifact.

    python report.py results.json

This is a run output, not an article: tables, and only the glue needed to read
them. Every "why" belongs in README.md, which is written once instead of being
reprinted underneath every single run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

METHOD_LINK = "benchmarks/vs_vectorbt/README.md"


def _ms(seconds: float) -> str:
    if seconds < 1.0:
        return "{:.1f} ms".format(seconds * 1e3)
    return "{:.2f} s".format(seconds)


def _header(payload: Dict[str, Any]) -> List[str]:
    env = payload["environment"]
    versions = env["versions"]
    cores = str(env["logical_cores"])
    if env.get("pinned_cores"):
        cores += " (pinned to {})".format(env["pinned_cores"])
    lines = [
        "# manifoldbt {} vs vectorbt {}".format(versions["manifoldbt"], versions["vectorbt"]),
        "",
        "`{os} {arch}` | {cpu} | {cores} cores | {ram} GB | python {py} | "
        "numpy {np} / numba {nb} / pandas {pd} | {reps} interleaved reps | {when}".format(
            os=env["os"], arch=env["arch"], cpu=env["cpu"], cores=cores, ram=env["ram_gb"],
            py=env["python"], np=versions["numpy"], nb=versions["numba"],
            pd=versions["pandas"], reps=payload["reps"], when=payload["generated_at"][:19],
        ),
    ]
    if env.get("run_url"):
        lines += ["", env["run_url"]]
    lines.append("")
    return lines


def _speed_table(rows: List[Dict[str, Any]], title: str) -> List[str]:
    if not rows:
        return []
    lines = [
        "## " + title,
        "",
        "| Workload | Bars | manifoldbt | vectorbt | Ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        timings = row["timings"]
        lines.append(
            "| {w} | {b:,} | {m} | {v} | **x{s:.1f}**{f} |".format(
                w=row["workload"], b=row["bars"],
                m=_ms(timings["manifoldbt"]["median_s"]),
                v=_ms(timings["vectorbt"]["median_s"]),
                s=row["speedup"]["median_of_ratios"],
                f=" ~" if row.get("noisy") else "",
            )
        )
    lines.append("")
    return lines


def _summary_cost(exact: List[Dict[str, Any]]) -> List[str]:
    plain = {r["bars"]: r for r in exact if r["workload"] == "sma_cross"}
    summarised = {r["bars"]: r for r in exact if r["workload"] == "sma_cross_metrics"}
    # Only subtract timings that were measured in the same interleaved loop.
    shared = sorted(
        b for b in set(plain) & set(summarised)
        if "sma_cross_metrics" in (plain[b].get("paired_with") or [])
    )
    if not shared:
        return []

    lines = [
        "## Cost of the performance summary",
        "",
        "Same simulation, with and without max drawdown / Sharpe / Sortino / volatility.",
        "",
        "| Bars | Engine | Without | With | Delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for bars in shared:
        for engine in ("manifoldbt", "vectorbt"):
            without = plain[bars]["timings"][engine]["median_s"]
            with_ = summarised[bars]["timings"][engine]["median_s"]
            lines.append(
                "| {b:,} | {e} | {a} | {c} | {d} |".format(
                    b=bars, e=engine, a=_ms(without), c=_ms(with_),
                    d=("+" + _ms(with_ - without)) if with_ > without else "none measurable",
                )
            )
    lines.append("")
    advisory = summarised[shared[-1]]["parity"]["diffs"].get("advisory_ratio_rel")
    if advisory:
        lines += [
            "Gated on total return, round-trips and max drawdown (exact). Sharpe, Sortino "
            "and volatility agree to {:.1e}, from daily bucketing.".format(max(advisory.values())),
            "",
        ]
    return lines


def _side_measures(payload: Dict[str, Any], results: List[Dict[str, Any]]) -> List[str]:
    cold = payload.get("cold_start")
    mem = payload.get("memory")
    threading_rows = [
        r for r in results
        if r.get("cpu_over_wall")
        and all(r["cpu_over_wall"].get(e) is not None for e in ("manifoldbt", "vectorbt"))
    ]
    if not (cold or mem or threading_rows):
        return []

    lines = ["## Cold start, memory, threads", "", "| | manifoldbt | vectorbt |", "|---|---:|---:|"]
    if cold:
        medians, share = cold["median_s"], cold["engine_share_s"]
        lines.append("| Fresh process to first backtest | {:.2f} s | {:.2f} s |".format(
            medians["manifoldbt"], medians["vectorbt"]))
        lines.append("| ... minus the {:.2f} s python baseline | {:.2f} s | {:.2f} s |".format(
            medians["baseline"], share["manifoldbt"], share["vectorbt"]))
    if mem:
        lines.append("| RAM added by the run, per 1M bars | {:.0f} MB | {:.0f} MB |".format(
            mem["manifoldbt"]["added_mb_per_million_bars"],
            mem["vectorbt"]["added_mb_per_million_bars"]))
    if threading_rows:
        biggest = max(threading_rows, key=lambda r: r["bars"])
        ratio = biggest["cpu_over_wall"]
        lines.append("| CPU over wall time at {:,} bars | {:.2f} | {:.2f} |".format(
            biggest["bars"], ratio["manifoldbt"], ratio["vectorbt"]))
    lines.append("")
    return lines


def render(payload: Dict[str, Any]) -> str:
    results = payload["results"]
    exact = [r for r in results if r["parity"]["status"] == "exact" and r.get("timings")]
    documented = [r for r in results if r["parity"]["status"] == "documented"]
    failed = [r for r in results if r["parity"]["status"] == "failed"]

    lines = _header(payload)
    lines += _speed_table(exact, "Same results, both engines")
    lines += _summary_cost(exact)
    lines += _side_measures(payload, results)

    timed = [r for r in documented if r.get("timings")]
    if timed:
        lines += _speed_table(timed, "Results differ, kept out of the headline")
        scales = [r for r in documented if r.get("divergence_scale")]
        if scales:
            row = scales[0]
            scale = row["divergence_scale"]
            lines += [
                "`{w}`: {n} of {t} round-trips re-enter on the exit bar ({p:.0%}) at {b:,} "
                "bars, from {sl} stop and {tp} target exits. Cause in {link}.".format(
                    w=row["workload"], n=scale["reentries_on_exit_bar"],
                    t=scale["round_trips"], p=scale["share_of_round_trips"],
                    b=row["bars"], sl=scale["sl_exits"], tp=scale["tp_exits"],
                    link=METHOD_LINK,
                ),
                "",
            ]

    if failed:
        lines += ["## Timing withheld", ""]
        for row in failed:
            diffs = row["parity"]["diffs"]
            lines.append(
                "- `{w}` at {b:,} bars: final equity differs by {d:.2e} of capital, "
                "round-trips by {t}.".format(
                    w=row["workload"], b=row["bars"],
                    d=diffs["final_equity_vs_capital"], t=diffs["round_trips_delta"],
                )
            )
        lines.append("")

    lines += [
        "---",
        "",
        "`~` = IQR above 15% of the median, indicative only. Ratios are medians of "
        "per-repetition ratios with the engines interleaved; data loading excluded, warmup "
        "discarded. Method and caveats: {}".format(METHOD_LINK),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="render a benchmark result")
    parser.add_argument("results", nargs="?", default="results.json")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        payload = json.load(fh)

    text = render(payload)
    sys.stdout.write(text + "\n")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
