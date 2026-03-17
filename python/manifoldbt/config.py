"""BacktestConfig helpers matching Rust serde format."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class OrderConfig:
    """Order management configuration for limit entries, stop-loss, take-profit,
    and trailing stops. All fields are optional — when nothing is set the engine
    uses the legacy market-order path with zero overhead.

    Sub-config dicts:
      limit_entry: {"offset_bps": 10.0, "time_in_force": "GTC"}
        offset_bps: distance from close in bps (buy: close*(1-offset/10000))
        time_in_force: "GTC" (default), {"GTB": 5}, or "IOC"
      stop_loss:  {"stop_pct": 2.0}   — % from entry price
      take_profit: {"profit_pct": 5.0} — % from entry price
      trailing_stop: {"trail_pct": 3.0, "use_high": true}
    """

    limit_entry: Optional[dict] = None
    stop_loss: Optional[dict] = None
    take_profit: Optional[dict] = None
    trailing_stop: Optional[dict] = None

    @classmethod
    def bracket(cls, stop_pct: float, profit_pct: float) -> "OrderConfig":
        """Convenience: create a bracket order (SL + TP)."""
        return cls(
            stop_loss={"stop_pct": stop_pct},
            take_profit={"profit_pct": profit_pct},
        )

    @classmethod
    def stop_loss_only(cls, stop_pct: float) -> "OrderConfig":
        """Convenience: stop-loss only."""
        return cls(stop_loss={"stop_pct": stop_pct})

    @classmethod
    def trailing(cls, trail_pct: float, use_high: bool = True) -> "OrderConfig":
        """Convenience: trailing stop only."""
        return cls(trailing_stop={"trail_pct": trail_pct, "use_high": use_high})

    def to_json_dict(self) -> dict:
        d: dict = {}
        if self.limit_entry is not None:
            d["limit_entry"] = self.limit_entry
        if self.stop_loss is not None:
            d["stop_loss"] = self.stop_loss
        if self.take_profit is not None:
            d["take_profit"] = self.take_profit
        if self.trailing_stop is not None:
            d["trailing_stop"] = self.trailing_stop
        return d


@dataclass
class ExecutionConfig:
    signal_delay: int = 0
    execution_price: str = "AtClose"
    max_position_pct: float = 1.0
    allow_short: bool = True
    allow_fractional: bool = True
    skip_gap_bars: bool = False
    position_sizing_mode: str = "FractionOfEquity"
    """How position_sizing output is interpreted:
    - "FractionOfEquity": target 1.0 = 100% of equity (default, compounds)
    - "FractionOfInitialCapital": same but uses initial capital (no compounding)
    - "Units": target 1.0 = 1 unit (share/contract/coin)
    """
    pyramiding: bool = False
    """When True, the signal is treated as a delta to ADD to the current position
    each bar (pyramiding), instead of a target position. Works with any sizing mode.
    Signal: 0.0 = go flat, NaN/None = hold, nonzero = add to position."""
    fill_model: Optional[dict] = None
    """Fill model configuration. None = Rust defaults (atomic fill, single point).
    Example: {"max_participation_rate": 0.1, "intra_bar_price": "TypicalPrice"}
    intra_bar_price options: "SinglePoint", "TypicalPrice", "OhlcAverage"
    """
    orders: Optional[OrderConfig] = None
    """Order management: limit entries, stop-loss, take-profit, trailing stops.
    When None (default), the engine uses the legacy market-order path."""

    def to_json_dict(self) -> dict:
        d = {
            "signal_delay": self.signal_delay,
            "execution_price": self.execution_price,
            "max_position_pct": self.max_position_pct,
            "allow_short": self.allow_short,
            "allow_fractional": self.allow_fractional,
            "skip_gap_bars": self.skip_gap_bars,
            "position_sizing_mode": self.position_sizing_mode,
            "pyramiding": self.pyramiding,
        }
        if self.fill_model is not None:
            d["fill_model"] = self.fill_model
        if self.orders is not None:
            d["orders"] = self.orders.to_json_dict()
        return d


@dataclass
class FeeConfig:
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    funding_rate_column: Optional[str] = None
    borrow_rate_annual_bps: float = 0.0
    min_fee: float = 0.0
    default_fill_type: str = "Taker"
    """Default fill type for fee calculation: "Maker" or "Taker" (conservative)."""

    def to_json_dict(self) -> dict:
        return {
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "funding_rate_column": self.funding_rate_column,
            "borrow_rate_annual_bps": self.borrow_rate_annual_bps,
            "min_fee": self.min_fee,
            "default_fill_type": self.default_fill_type,
        }

    @classmethod
    def binance_perps(cls) -> "FeeConfig":
        """Binance USDM perpetual futures defaults (taker fees + funding)."""
        return cls(
            maker_fee_bps=2.0,
            taker_fee_bps=5.0,
            funding_rate_column="funding_rate",
            borrow_rate_annual_bps=0.0,
            min_fee=0.0,
            default_fill_type="Taker",
        )

    @classmethod
    def binance_spot(cls) -> "FeeConfig":
        """Binance spot defaults (taker fees, no funding)."""
        return cls(
            maker_fee_bps=10.0,
            taker_fee_bps=10.0,
            funding_rate_column=None,
            borrow_rate_annual_bps=0.0,
            min_fee=0.0,
            default_fill_type="Taker",
        )

    @classmethod
    def zero(cls) -> "FeeConfig":
        """No fees (for development/debugging)."""
        return cls()


@dataclass
class BacktestConfig:
    universe: List[int] = field(default_factory=lambda: [1])
    time_range_start: int = 0
    time_range_end: int = 4_000_000_000
    initial_capital: float = 1000.0
    currency: str = "USD"
    bar_interval: Any = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    slippage: Any = None
    data_version: Optional[str] = None
    rng_seed: Optional[int] = None
    trading_days_per_year: float = 365.25
    """Annualisation factor: 365.25 for crypto/futures, 252 for equities."""
    output_resolution: Any = None
    """Downsample output timeseries (equity, positions).
    None = auto (uses resample_to if set, else bar_interval; min 1h).
    Use ``{"Hours": 1}``, ``{"Days": 1}``, etc. for explicit control."""
    resample_to: Any = None
    """Resample raw bars to this interval before simulation.
    E.g. ``{"Minutes": 60}`` aggregates 1-min data into 60-min OHLCV bars.
    ``None`` (default) = use data as-is from the store."""
    feature_sets: List[str] = field(default_factory=list)
    """Feature sets to preload. Feature columns from
    ``features/{set_name}/{symbol_id}/`` are injected into signal env."""
    symbol_names: Dict[str, int] = field(default_factory=dict)
    """Mapping of human-readable symbol names to SymbolId integers.
    Required when using ``bt.asset()`` or ``symbol_ref()`` cross-asset references.
    Example: ``{"BTCUSDT": 1, "ETHUSDT": 2}``"""
    warmup_bars: int = 0
    """Number of bars to skip before acting on signals.
    Allows indicators (EMA, SMA, etc.) to stabilise. During warmup,
    equity tracking runs but no trades are generated.
    Set to at least the longest indicator window (e.g. 25 for EMA(25))."""
    accuracy: bool = False
    """When True, simulation runs on 1-minute bars regardless of bar_interval.
    Signals are still evaluated at bar_interval resolution (hybrid mode).
    Use for precise SL/TP fills and intraday drawdown tracking. Slower."""

    def to_json_dict(self) -> dict:
        d: dict = {
            "universe": self.universe,
            "time_range": {
                "start": self.time_range_start,
                "end": self.time_range_end,
            },
            "bar_interval": self.bar_interval or {"Seconds": 1},
            "initial_capital": self.initial_capital,
            "currency": self.currency,
            "execution": self.execution.to_json_dict(),
            "fees": self.fees.to_json_dict(),
            "slippage": self.slippage or {"FixedBps": {"bps": 0.0}},
            "data_version": self.data_version,
            "rng_seed": self.rng_seed,
            "trading_days_per_year": self.trading_days_per_year,
        }
        if self.output_resolution is not None:
            d["output_resolution"] = self.output_resolution
        if self.resample_to is not None:
            d["resample_to"] = self.resample_to
        if self.feature_sets:
            d["feature_sets"] = self.feature_sets
        if self.symbol_names:
            d["symbol_names"] = self.symbol_names
        if self.warmup_bars > 0:
            d["warmup_bars"] = self.warmup_bars
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict())


def resolve_universe(
    universe: List[Union[int, str]],
    store: Any,
) -> List[int]:
    """Resolve a mixed list of symbol IDs and ticker names to integer IDs.

    Args:
        universe: List of integer IDs or string ticker names.
        store: A ``DataStore`` instance (must have ``resolve_symbol()``).

    Returns:
        List of integer symbol IDs.

    Raises:
        ValueError: If a ticker name cannot be resolved.
        TypeError: If store is None and string tickers are present.
    """
    result = []
    for item in universe:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, str):
            if store is None:
                raise TypeError(
                    f"DataStore required to resolve symbol name {item!r}. "
                    f"Pass integer IDs or provide a store."
                )
            result.append(store.resolve_symbol(item))
        else:
            result.append(int(item))
    return result
