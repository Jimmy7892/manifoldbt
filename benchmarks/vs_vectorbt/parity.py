"""The gate: no speed number is published for a workload the engines disagree on.

A benchmark between two backtesters is only a benchmark if both engines did the
same work. This module compares what each engine produced and classifies the
result, and ``bench.py`` refuses to report a timing for anything it classifies
as a failure.

Three verdicts:

``exact``
    Agreement down to float-reordering noise. The timing is publishable.
``documented``
    The engines disagree, the workload declared it in advance, and the reason is
    written down. The timing goes to the annex with the reason attached.
``failed``
    The engines disagree and nobody predicted it. That is a finding about the
    engines, not about their speed: the timing is withheld.
"""
from __future__ import annotations

from typing import Any, Dict

from workloads import CAPITAL, WORKLOADS

# Float reordering across two implementations of the same arithmetic lands
# around 1e-13 of the account on a million bars. Anything above this is a
# different decision somewhere, not a different summation order.
REL_TOL = 1e-9


def _rel(a: float, b: float) -> float:
    """Plain relative difference, for quantities that are not money."""
    return abs(a - b) / max(1e-12, abs(b))


def _vs_capital(a: float, b: float) -> float:
    """Difference as a fraction of the money at risk, not of the result itself.

    Anchoring on the result breaks exactly when the result is interesting: a
    strategy that ends near zero equity, or near zero return, turns a difference
    of a hundredth of a cent into a 1% relative error and fails a comparison the
    engines actually passed. The account size is the stable yardstick."""
    return abs(a - b) / CAPITAL


def compare(mbt: Dict[str, Any], vbt: Dict[str, Any], key: str) -> Dict[str, Any]:
    expected = WORKLOADS[key].parity

    diffs = {
        "final_equity_vs_capital": _vs_capital(mbt["final_equity"], vbt["final_equity"]),
        "round_trips_delta": mbt["round_trips"] - vbt["round_trips"],
        "total_fees_vs_capital": _vs_capital(mbt["total_fees"], vbt["total_fees"]),
    }
    agrees = (
        diffs["final_equity_vs_capital"] <= REL_TOL
        and diffs["round_trips_delta"] == 0
        and diffs["total_fees_vs_capital"] <= REL_TOL
    )

    # Workloads that also produce a performance summary are gated on the
    # drawdown, which both engines compute at full bar resolution and which must
    # match. The ratios are reported but not gated: manifoldbt buckets its daily
    # returns slightly differently, a difference worth stating rather than
    # hiding, and worth nothing at all as an argument about speed.
    if "max_drawdown" in mbt and "max_drawdown" in vbt:
        diffs["max_drawdown_rel"] = _rel(mbt["max_drawdown"], vbt["max_drawdown"])
        agrees = agrees and diffs["max_drawdown_rel"] <= REL_TOL
        diffs["advisory_ratio_rel"] = {
            name: _rel(mbt[name], vbt[name])
            for name in ("sharpe", "sortino", "volatility")
            if name in mbt and name in vbt
        }

    if agrees:
        status = "exact"
    elif expected == "documented":
        status = "documented"
    else:
        status = "failed"

    return {
        "status": status,
        "expected": expected,
        "publishable": status == "exact",
        "diffs": diffs,
        "metrics": {"manifoldbt": mbt, "vectorbt": vbt},
        "note": WORKLOADS[key].divergence if status == "documented" else "",
    }
