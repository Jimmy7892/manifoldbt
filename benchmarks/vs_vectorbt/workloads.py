"""Workload definitions: the numbers both engines read, in one place.

Nothing here is engine-specific. Each adapter (``engine_mbt``, ``engine_vbt``)
reads the same constants, so a workload cannot drift between the two sides by
someone editing one file and forgetting the other.

Sizing policy, and why it changes when fees are on
--------------------------------------------------
With ``FractionOfEquity`` sizing and a non-zero fee, the two engines size a
position differently: manifoldbt charges the fee on top of a full-equity
notional, vectorbt reserves it out of cash first. Both are defensible product
decisions, and comparing them would compare *policy*, not speed or correctness.
The fee workload therefore sizes in fixed units, which isolates the fee
arithmetic itself. This mirrors the choice already made in the cross-engine
parity suite shipped with the library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

CAPITAL = 100_000.0

# Bar interval of the generated data. Both engines are told the same thing:
# manifoldbt through `Interval.minutes(1)`, vectorbt through `freq="1min"`
# (annualisation only; it does not touch the simulation).
FREQ = "1min"


@dataclass(frozen=True)
class Workload:
    key: str
    title: str
    why: str
    params: Dict[str, Any] = field(default_factory=dict)
    # "exact": the engines must agree to float-reordering noise, or the timing is
    # not published. "documented": they are known to disagree for a reason
    # written down in `divergence`; the timing goes to the annex, never to the
    # headline table.
    parity: str = "exact"
    divergence: str = ""


WORKLOADS: Dict[str, Workload] = {
    w.key: w
    for w in (
        Workload(
            key="sma_cross",
            title="SMA 10/50 crossover, long-only, no cost",
            why="The canonical baseline. Unambiguous indicator, no fee policy, "
                "no stop semantics: if the engines disagree here, nothing else "
                "in the suite is worth reading.",
            params=dict(fast=10, slow=50, alloc=1.0),
        ),
        Workload(
            key="ema_rsi_fees",
            title="EMA 12/26 crossover + RSI(14) filter, 5 bps taker fee",
            why="A realistic signal stack with a real cost model, sized in fixed "
                "units so the fee arithmetic is comparable across engines. The "
                "unit count is small relative to capital on purpose: at 1-minute "
                "resolution this strategy turns over often enough that a larger "
                "size would spend the whole account on fees, and comparing two "
                "engines on a wiped-out account compares rounding noise.",
            params=dict(fast=12, slow=26, rsi_period=14, rsi_lo=30.0, rsi_hi=70.0,
                        units=5.0, fee_bps=5.0),
        ),
        Workload(
            key="sma_cross_metrics",
            title="SMA 10/50 crossover, with a performance summary",
            why="The same simulation as `sma_cross`, but both engines are asked "
                "for what a user actually reads: max drawdown, Sharpe, Sortino "
                "and volatility alongside the return. manifoldbt computes them "
                "inside run() whether you ask or not; vectorbt defers the equity "
                "curve until a risk metric needs it, and every one of them pays "
                "for materialising it. This is a scope difference, not a trick: "
                "`sma_cross` above is the same work without the summary, and the "
                "two are reported side by side so the reader can see what the "
                "summary costs each engine.",
            params=dict(fast=10, slow=50, alloc=1.0, metrics=True),
        ),
        Workload(
            key="bracket_sl_tp",
            title="SMA 10/50 entry with a 15 bps stop / 30 bps target bracket",
            why="Brackets are where two engines most easily disagree: the stop "
                "level, the fill on the triggering bar, and whether a re-entry "
                "is allowed on the bar after an exit.",
            params=dict(fast=10, slow=50, alloc=1.0, sl_pct=0.15, tp_pct=0.30),
            parity="documented",
            divergence=(
                "Re-entry on the exit bar. When a bracket fires intrabar and the "
                "entry condition still holds at that bar's close, manifoldbt books "
                "two orders on that bar (the stop or target exit, then a fresh "
                "entry at the close); vectorbt processes one order per bar and "
                "re-enters on the next bar instead. Neither is wrong, and on "
                "controlled bars the bracket fills themselves match exactly (see "
                "the cross-engine parity suite shipped with the library). The "
                "harness counts the affected round-trips so the size of the "
                "divergence is measured, not asserted."
            ),
        ),
    )
}

DEFAULT_KEYS = list(WORKLOADS)

# Two workloads that the report compares directly against each other, so they
# must be measured inside ONE interleaved loop rather than in two blocks minutes
# apart. Comparing medians across blocks is exactly the mistake the interleaving
# exists to prevent: absolute timings drift between blocks, and the drift once
# produced a table claiming the version doing MORE work was the faster one.
SCOPE_PAIR = ("sma_cross", "sma_cross_metrics")
