"""Convenience helpers for configuration.

Simplifies creating ``BacktestConfig`` by accepting human-readable dates,
named slippage models, and bar intervals.

Usage::

    from manifoldbt.helpers import date_to_ns, time_range, Slippage, Interval

    start, end = time_range("2022-01-01", "2024-01-01")
    config = bt.BacktestConfig(
        time_range_start=start,
        time_range_end=end,
        slippage=Slippage.fixed_bps(1.0),
        bar_interval=Interval.minutes(1),
    )
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

NANOS_PER_SECOND = 1_000_000_000


def date_to_ns(date_str: str) -> int:
    """Convert a date string to nanoseconds since Unix epoch (UTC).

    Accepted formats:
        - ``"2021-01-15"``
        - ``"2021-01-15 09:30:00"``
        - ``"2021-01-15T09:30:00"``
    """
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp()) * NANOS_PER_SECOND
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date '{date_str}'. "
        "Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
    )


def time_range(start: str, end: str) -> Tuple[int, int]:
    """Convert two date strings to a ``(start_ns, end_ns)`` tuple."""
    return date_to_ns(start), date_to_ns(end)


# ---------------------------------------------------------------------------
# Slippage model factories
# ---------------------------------------------------------------------------


class Slippage:
    """Factory for slippage configuration dicts."""

    @staticmethod
    def fixed_bps(bps: float) -> Dict[str, Any]:
        """Fixed basis-point slippage on every fill."""
        return {"FixedBps": {"bps": bps}}

    @staticmethod
    def volume_impact(impact_coeff: float, exponent: float = 1.5) -> Dict[str, Any]:
        """Volume-participation impact model.

        Cost = ``impact_coeff * participation_rate ^ exponent``.
        """
        return {"VolumeImpact": {"impact_coeff": impact_coeff, "exponent": exponent}}

    @staticmethod
    def spread_based(spread_fraction: float = 1.0) -> Dict[str, Any]:
        """Spread-based slippage (fraction of bid-ask spread)."""
        return {"SpreadBased": {"spread_fraction": spread_fraction}}

    @staticmethod
    def none() -> Dict[str, Any]:
        """No slippage."""
        return {"FixedBps": {"bps": 0.0}}


# ---------------------------------------------------------------------------
# Bar interval factories
# ---------------------------------------------------------------------------


class Interval:
    """Factory for bar interval configuration dicts."""

    @staticmethod
    def seconds(n: int = 1) -> Dict[str, int]:
        return {"Seconds": n}

    @staticmethod
    def minutes(n: int = 1) -> Dict[str, int]:
        return {"Minutes": n}

    @staticmethod
    def hours(n: int = 1) -> Dict[str, int]:
        return {"Hours": n}

    @staticmethod
    def days(n: int = 1) -> Dict[str, int]:
        return {"Days": n}


# ---------------------------------------------------------------------------
# Execution price constants
# ---------------------------------------------------------------------------


class ExecutionPrice:
    """Constants matching Rust ``ExecutionPrice`` enum variants."""

    NEXT_BAR_OPEN = "NextBarOpen"
    NEXT_BAR_CLOSE = "NextBarClose"
    NEXT_BAR_VWAP = "NextBarVwap"
    AT_CLOSE = "AtClose"
    AT_OPEN = "AtOpen"
    AT_VWAP = "AtVwap"
    MID_PRICE = "MidPrice"

    @staticmethod
    def custom(column: str) -> Dict[str, str]:
        """Fill at a named column from bar data."""
        return {"Custom": column}


# ---------------------------------------------------------------------------
# Fill model factories
# ---------------------------------------------------------------------------


class FillModel:
    """Factory for fill model configuration dicts."""

    @staticmethod
    def atomic() -> Dict[str, Any]:
        """Atomic fill — entire order at single price (default)."""
        return {"max_participation_rate": 0.0, "intra_bar_price": "SinglePoint"}

    @staticmethod
    def participation(
        rate: float, intra_bar_price: str = "SinglePoint"
    ) -> Dict[str, Any]:
        """Partial fill limited to a fraction of bar volume.

        Args:
            rate: Max fraction of bar volume per fill (e.g. 0.1 = 10%).
            intra_bar_price: "SinglePoint", "TypicalPrice", or "OhlcAverage".
        """
        return {"max_participation_rate": rate, "intra_bar_price": intra_bar_price}
