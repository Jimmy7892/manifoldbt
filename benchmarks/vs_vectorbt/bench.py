"""The runner: interleaved A/B timing with a parity gate in front of it.

Design decisions that are the whole point of this harness
---------------------------------------------------------
*Parity first.* Every workload runs once per engine before any measurement, and
the results are compared. A workload the engines disagree on gets no published
timing (see ``parity.py``).

*Interleaved repetitions.* The engines alternate within each repetition rather
than running in two blocks. A cloud runner that slows down halfway through
penalises both engines equally instead of whichever one happened to be second.

*Ratios are the headline, milliseconds are context.* The per-repetition ratio is
computed from two measurements taken seconds apart on the same machine, so it
survives the noise that absolute timings on shared hardware do not.

*Dispersion is published.* Every point carries min, median, max and IQR. A point
whose IQR exceeds 15% of its median is flagged noisy, and a flagged point is not
headline material no matter how good it looks.

Usage
-----
    python bench.py --bars 10000 100000 1000000 --reps 7 --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod  # noqa: E402
import engine_mbt  # noqa: E402
import engine_vbt  # noqa: E402
import parity as parity_mod  # noqa: E402
from workloads import DEFAULT_KEYS, SCOPE_PAIR, WORKLOADS  # noqa: E402

SCHEMA_VERSION = 1
NOISE_THRESHOLD = 0.15


# --------------------------------------------------------------------------- #
# Environment: what a reader needs to know before trusting a number
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif sys.platform.startswith("win"):
            return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb():
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal"):
                        return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except Exception:
        pass
    return None


def _versions() -> Dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in ("manifoldbt", "vectorbt", "numpy", "numba", "pandas"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "absent"
    return out


def environment() -> Dict[str, Any]:
    run_id = os.environ.get("GITHUB_RUN_ID")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_url = None
    if run_id and repo:
        run_url = server + "/" + repo + "/actions/runs/" + run_id
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
        "versions": _versions(),
        "runner": os.environ.get("RUNNER_NAME") or "local",
        "ci": bool(run_id),
        "run_url": run_url,
        "commit": os.environ.get("GITHUB_SHA"),
    }


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _time_once(fn: Callable[[], Dict[str, Any]]):
    start = time.perf_counter()
    metrics = fn()
    return time.perf_counter() - start, metrics


def _cpu_seconds() -> float:
    """Process CPU time, user plus system. Returns 0.0 if psutil is missing."""
    try:
        import psutil

        times = psutil.Process().cpu_times()
        return times.user + times.system
    except Exception:
        return 0.0


class _Parallelism:
    """CPU time over wall time, accumulated across every repetition.

    This is the number that decides whether a result measured on a 20-core
    workstation transposes to a 4-vCPU cloud runner. Close to 1.0 means the
    engine used one thread and the core count barely matters; well above 1.0
    means the measurement is a function of the machine it ran on, and a
    reader on smaller hardware should expect less. Accumulating over all
    repetitions rather than timing each call keeps the coarse resolution of
    the OS accounting clocks from dominating short runs.

    Below `MIN_WALL_S` of accumulated work the answer is withheld rather than
    guessed. Process CPU accounting advances in scheduler ticks of roughly 15 ms,
    so a sub-millisecond run lands either on zero ticks or on one whole tick, and
    the ratio comes out as 0.00 or as 11.0 for work that is plainly single
    threaded. A benchmark that prints "this engine used 11 cores" once has
    spent its credibility on a rounding artefact.
    """

    MIN_WALL_S = 0.5

    def __init__(self) -> None:
        self.cpu = 0.0
        self.wall = 0.0

    def record(self, fn: Callable[[], Dict[str, Any]]):
        cpu_before = _cpu_seconds()
        elapsed, metrics = _time_once(fn)
        self.cpu += _cpu_seconds() - cpu_before
        self.wall += elapsed
        return elapsed, metrics

    @property
    def ratio(self):
        if self.wall < self.MIN_WALL_S:
            return None
        return self.cpu / self.wall


def _summarise(samples: List[float]) -> Dict[str, Any]:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    if len(ordered) >= 4:
        mid = len(ordered) // 2
        lower = statistics.median(ordered[:mid])
        upper = statistics.median(ordered[-mid:])
        iqr = upper - lower
    else:
        iqr = ordered[-1] - ordered[0]
    return {
        "samples_s": samples,
        "min_s": ordered[0],
        "median_s": median,
        "max_s": ordered[-1],
        "iqr_s": iqr,
        "iqr_over_median": iqr / median if median else 0.0,
    }


def measure_pair(keys: List[str], bars: int, reps: int, workdir: str) -> List[Dict[str, Any]]:
    """Measure two workloads inside ONE interleaved loop.

    The report puts these two side by side to show what a performance summary
    costs each engine. That subtraction is only legitimate if all four timings
    come from the same stretch of machine time: measured in separate blocks, the
    drift in absolute timings is larger than the difference being reported, and
    the table ends up claiming the version doing more work is the faster one.
    """
    frame = data_mod.make_ohlcv(bars)
    runners = {}
    entries = {}
    for key in keys:
        run_mbt = engine_mbt.prepare(key, frame, workdir)
        run_vbt = engine_vbt.prepare(key, frame, workdir)
        _, warm_mbt = _time_once(run_mbt)
        _, warm_vbt = _time_once(run_vbt)
        verdict = parity_mod.compare(warm_mbt, warm_vbt, key)
        runners[key] = (run_mbt, run_vbt)
        entries[key] = {
            "workload": key,
            "title": WORKLOADS[key].title,
            "bars": bars,
            "data_digest": data_mod.digest(frame),
            "parity": verdict,
            "paired_with": [k for k in keys if k != key],
        }

    samples: Dict[str, Dict[str, List[float]]] = {
        key: {"manifoldbt": [], "vectorbt": [], "ratio": []} for key in keys
    }
    threading_use = {key: {"manifoldbt": _Parallelism(), "vectorbt": _Parallelism()}
                     for key in keys}
    for _ in range(reps):
        for key in keys:
            run_mbt, run_vbt = runners[key]
            t_mbt, _ = threading_use[key]["manifoldbt"].record(run_mbt)
            t_vbt, _ = threading_use[key]["vectorbt"].record(run_vbt)
            samples[key]["manifoldbt"].append(t_mbt)
            samples[key]["vectorbt"].append(t_vbt)
            samples[key]["ratio"].append(t_vbt / t_mbt if t_mbt > 0 else float("nan"))

    out = []
    for key in keys:
        entry = entries[key]
        if entry["parity"]["status"] == "failed":
            entry["timings"] = None
            entry["note"] = "timing withheld: unexplained disagreement between engines"
            out.append(entry)
            continue
        mbt_stats = _summarise(samples[key]["manifoldbt"])
        vbt_stats = _summarise(samples[key]["vectorbt"])
        ratios = samples[key]["ratio"]
        entry["timings"] = {"manifoldbt": mbt_stats, "vectorbt": vbt_stats}
        entry["speedup"] = {
            "median_of_ratios": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        }
        entry["noisy"] = (
            mbt_stats["iqr_over_median"] > NOISE_THRESHOLD
            or vbt_stats["iqr_over_median"] > NOISE_THRESHOLD
        )
        entry["cpu_over_wall"] = {
            engine: threading_use[key][engine].ratio for engine in ("manifoldbt", "vectorbt")
        }
        out.append(entry)
    return out


def measure(key: str, bars: int, reps: int, workdir: str) -> Dict[str, Any]:
    frame = data_mod.make_ohlcv(bars)
    run_mbt = engine_mbt.prepare(key, frame, workdir)
    run_vbt = engine_vbt.prepare(key, frame, workdir)

    # Warmup, discarded: numba compiles on vectorbt's first call, and the engine
    # warms its own caches. Both are one-off costs, reported separately rather
    # than smeared across every repetition.
    _, warm_mbt = _time_once(run_mbt)
    _, warm_vbt = _time_once(run_vbt)

    verdict = parity_mod.compare(warm_mbt, warm_vbt, key)
    entry: Dict[str, Any] = {
        "workload": key,
        "title": WORKLOADS[key].title,
        "bars": bars,
        "data_digest": data_mod.digest(frame),
        "parity": verdict,
    }
    if verdict["status"] == "documented":
        entry["divergence_scale"] = engine_mbt.diagnose(key, frame, workdir)

    if verdict["status"] == "failed":
        entry["timings"] = None
        entry["note"] = "timing withheld: unexplained disagreement between engines"
        return entry

    mbt_samples: List[float] = []
    vbt_samples: List[float] = []
    ratios: List[float] = []
    threading_use = {"manifoldbt": _Parallelism(), "vectorbt": _Parallelism()}
    for _ in range(reps):
        t_mbt, _ = threading_use["manifoldbt"].record(run_mbt)
        t_vbt, _ = threading_use["vectorbt"].record(run_vbt)
        mbt_samples.append(t_mbt)
        vbt_samples.append(t_vbt)
        ratios.append(t_vbt / t_mbt if t_mbt > 0 else float("nan"))

    mbt_stats = _summarise(mbt_samples)
    vbt_stats = _summarise(vbt_samples)
    entry["timings"] = {"manifoldbt": mbt_stats, "vectorbt": vbt_stats}
    entry["speedup"] = {
        "median_of_ratios": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
    }
    entry["noisy"] = (
        mbt_stats["iqr_over_median"] > NOISE_THRESHOLD
        or vbt_stats["iqr_over_median"] > NOISE_THRESHOLD
    )
    entry["cpu_over_wall"] = {
        engine: threading_use[engine].ratio for engine in ("manifoldbt", "vectorbt")
    }
    return entry


# --------------------------------------------------------------------------- #
# Probes that need a fresh process (see probe_child.py)
# --------------------------------------------------------------------------- #
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_child.py")


def _probe(mode: str, engine: str, workload: str, bars: int) -> Dict[str, Any]:
    out = subprocess.run(
        [sys.executable, CHILD, mode, engine, workload, str(bars)],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("PROBE_RESULT "):
            return json.loads(line[len("PROBE_RESULT "):])
    raise RuntimeError("probe produced no result: " + out[-500:])


def cold_start(workload: str, bars: int, reps: int) -> Dict[str, Any]:
    """Time to a first backtest in a process that has never seen either engine.

    This is the cost the steady-state benchmark throws away as warmup, and the
    one a user actually waits through every time they open a notebook. The
    engines alternate here too, and the interpreter/numpy/pandas baseline is
    measured alongside so the engine's own share is visible.
    """
    samples: Dict[str, List[float]] = {"manifoldbt": [], "vectorbt": [], "baseline": []}
    for _ in range(reps):
        samples["manifoldbt"].append(_probe("coldstart", "mbt", workload, bars)["seconds"])
        samples["vectorbt"].append(_probe("coldstart", "vbt", workload, bars)["seconds"])
        samples["baseline"].append(_probe("baseline", "none", workload, bars)["seconds"])
    medians = {k: statistics.median(v) for k, v in samples.items()}
    base = medians["baseline"]
    return {
        "workload": workload,
        "bars": bars,
        "samples_s": samples,
        "median_s": medians,
        "engine_share_s": {
            "manifoldbt": medians["manifoldbt"] - base,
            "vectorbt": medians["vectorbt"] - base,
        },
        "ratio": medians["vectorbt"] / medians["manifoldbt"] if medians["manifoldbt"] else None,
    }


def memory(workload: str, bars: int) -> Dict[str, Any]:
    """Resident memory each engine adds while running one backtest.

    vectorbt materialises the simulation as arrays, so its footprint grows with
    the series; manifoldbt streams bars out of its store. The number reported is
    what the *run* adds, measured after a warmup, not the process total: the
    one-off cost of building each engine's data representation is a different
    question and is excluded on both sides.
    """
    mbt = _probe("memory", "mbt", workload, bars)
    vbt = _probe("memory", "vbt", workload, bars)
    return {
        "workload": workload,
        "bars": bars,
        "manifoldbt": mbt,
        "vectorbt": vbt,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="manifoldbt vs vectorbt")
    parser.add_argument("--bars", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--workloads", nargs="+", default=DEFAULT_KEYS, choices=DEFAULT_KEYS)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--workdir", default=None, help="where the engine store is built")
    parser.add_argument("--cold-start-reps", type=int, default=3,
                        help="0 disables the cold-start probe")
    parser.add_argument("--memory-bars", type=int, default=0,
                        help="bars for the memory probe; 0 disables it")
    parser.add_argument("--pin-cores", type=int, default=0,
                        help="restrict the process to N logical cores, so a big "
                             "workstation can reproduce what a small cloud runner "
                             "sees; 0 leaves the machine alone")
    args = parser.parse_args()

    pinned = None
    if args.pin_cores:
        try:
            import psutil

            everything = list(range(os.cpu_count() or 1))
            psutil.Process().cpu_affinity(everything[: args.pin_cores])
            pinned = args.pin_cores
        except Exception as exc:            # macOS has no affinity API
            print("could not pin to {} cores: {}".format(args.pin_cores, exc))

    workdir = args.workdir or tempfile.mkdtemp(prefix="mbt_vs_vbt_")
    env = environment()
    env["pinned_cores"] = pinned
    versions = env["versions"]
    print("manifoldbt " + versions["manifoldbt"] + " vs vectorbt " + versions["vectorbt"])
    print("{cpu} | {cores} logical cores | {ram} GB | {os} {arch}".format(
        cpu=env["cpu"], cores=env["logical_cores"], ram=env["ram_gb"],
        os=env["os"], arch=env["arch"]))
    if pinned:
        print("pinned to {} logical cores for this run".format(pinned))
    print(str(args.reps) + " interleaved repetitions per point\n")

    def announce(entry: Dict[str, Any]) -> bool:
        status = entry["parity"]["status"]
        print("  {k:18s} {b:>9,} bars ... ".format(k=entry["workload"], b=entry["bars"]),
              end="", flush=True)
        if entry.get("timings"):
            print("{s:10s} manifoldbt x{v:.1f}{m}".format(
                s=status, v=entry["speedup"]["median_of_ratios"],
                m=" NOISY" if entry.get("noisy") else ""))
            return False
        print("{s:10s} timing withheld".format(s=status))
        return True

    # The scope pair is measured together; everything else one workload at a time.
    paired = [k for k in SCOPE_PAIR if k in args.workloads]
    singles = [k for k in args.workloads if k not in paired]

    results = []
    failures = 0
    for bars in args.bars:
        if len(paired) > 1:
            for entry in measure_pair(paired, bars, args.reps, workdir):
                results.append(entry)
                failures += announce(entry)
    for key in singles + (paired if len(paired) == 1 else []):
        for bars in args.bars:
            entry = measure(key, bars, args.reps, workdir)
            results.append(entry)
            failures += announce(entry)

    cold = None
    if args.cold_start_reps:
        print("")
        print("  cold start ... ", end="", flush=True)
        cold = cold_start(args.workloads[0], 20_000, args.cold_start_reps)
        print("manifoldbt {:.2f} s vs vectorbt {:.2f} s (x{:.1f})".format(
            cold["median_s"]["manifoldbt"], cold["median_s"]["vectorbt"], cold["ratio"]))

    mem = None
    if args.memory_bars:
        print("  memory     ... ", end="", flush=True)
        mem = memory(args.workloads[0], args.memory_bars)
        print("manifoldbt +{:.0f} MB vs vectorbt +{:.0f} MB at {:,} bars".format(
            mem["manifoldbt"]["added_mb"], mem["vectorbt"]["added_mb"], args.memory_bars))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "reps": args.reps,
        "results": results,
        "cold_start": cold,
        "memory": mem,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote " + args.out)

    # A parity failure is a finding, and CI should go red on it.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
