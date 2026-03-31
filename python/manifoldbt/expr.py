"""Expression AST builder — the core of the Python DSL.

Builds an expression tree that serializes to JSON matching the Rust
``bt_expr::Expr`` serde (externally-tagged) format.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from manifoldbt._serde import scalar_value_to_json

Numeric = Union[int, float, "Expr"]
Period = Union[int, "Expr"]
Span = Union[float, int, "Expr"]


# Global registry of param metadata encountered during expression construction.
# Populated by _resolve_period/_resolve_span, read by Strategy.to_json_dict().
_param_registry: dict = {}


def _resolve_period(value: Period) -> Any:
    """Convert a period argument for DynPeriod serialization.

    - int → int (serializes as JSON number → DynPeriod::Fixed)
    - param("name") Expr → "name" (serializes as JSON string → DynPeriod::Param)
    """
    if isinstance(value, Expr) and value._variant == "Parameter":
        if value._param_meta is not None:
            _param_registry[value._args[0]] = value._param_meta
        return value._args[0]
    if isinstance(value, Expr):
        raise TypeError("Only param() expressions can be used as indicator periods, not arbitrary expressions")
    return int(value)


def _resolve_span(value: Span) -> Any:
    """Convert a span/float argument for DynFloat serialization."""
    if isinstance(value, Expr) and value._variant == "Parameter":
        if value._param_meta is not None:
            _param_registry[value._args[0]] = value._param_meta
        return value._args[0]
    if isinstance(value, Expr):
        raise TypeError("Only param() expressions can be used as indicator spans, not arbitrary expressions")
    return float(value)

# Variants that wrap a single Box<Expr>
_UNARY_BOX = frozenset(
    [
        "Not", "CumSum", "CumProd", "Rank", "CrossSectionalMean", "CrossSectionalRank",
        "Hour", "Minute", "DayOfWeek", "Month", "DayOfMonth",
    ]
)

# Variants with two Box<Expr>
_BINARY_BOX = frozenset(["Add", "Sub", "Mul", "Div", "Gt", "Lt", "Eq", "And", "Or"])

# Variants with Box<Expr> + usize (or f64 for EwmMean)
_EXPR_SCALAR = frozenset(
    [
        "Lag",
        "Lead",
        "RollingMean",
        "RollingStd",
        "RollingSum",
        "RollingMin",
        "RollingMax",
        "EwmMean",
        "Diff",
        "PctChange",
        "ZScore",
        "Rsi",
        "LinRegSlope",
        "LinRegValue",
        "LinRegR2",
        # New indicators (Box<Expr>, usize)
        "Dema",
        "Tema",
        "Wma",
        "Hma",
        "Kama",
        "Roc",
        "RollingMedian",
    ]
)

# Box<Expr> + usize + usize
_EXPR_2SCALAR = frozenset(["Macd"])

# Box<Expr> + usize + usize + usize
_EXPR_3SCALAR = frozenset(["MacdSignal", "MacdHist"])

# Box<Expr> + usize + f64
_EXPR_SCALAR_F64 = frozenset(["BollingerUpper", "BollingerLower", "BollingerWidth"])

# 3×Box<Expr> (no extra scalar)
_HLC_NO_SCALAR = frozenset(["TrueRange"])

# 3×Box<Expr> + usize — same layout as Atr
_HLC_USIZE = frozenset(["StochK", "WilliamsR", "Cci", "Adx", "Natr"])

# 3×Box<Expr> + usize + f64
_HLC_USIZE_F64 = frozenset(["KeltnerUpper", "KeltnerLower", "SuperTrend"])

# 2×Box<Expr>
_BINARY_EXPR = frozenset(["Obv", "CrossAbove", "CrossBelow"])

# 4×Box<Expr>
_HLCV_NO_SCALAR = frozenset(["Vwap", "AdLine"])

# 4×Box<Expr> + usize
_HLCV_USIZE = frozenset(["Mfi"])


class Expr:
    """AST node representing a backtester expression."""

    __slots__ = ("_variant", "_args", "_param_meta")
    __hash__ = None  # not hashable (we override __eq__)

    def __init__(self, variant: str, *args: Any) -> None:
        self._variant = variant
        self._args = args
        self._param_meta = None

    # -- Serialization -------------------------------------------------------

    def to_json(self) -> Any:
        """Serialize to a dict/value matching Rust ``Expr`` serde format."""
        v = self._variant
        args = self._args

        if v in _UNARY_BOX:
            return {v: args[0].to_json()}

        if v in _BINARY_BOX:
            return {v: [args[0].to_json(), args[1].to_json()]}

        if v in _EXPR_SCALAR:
            return {v: [args[0].to_json(), args[1]]}

        if v in _EXPR_2SCALAR:
            # e.g. Macd(Box<Expr>, usize, usize)
            return {v: [args[0].to_json(), args[1], args[2]]}

        if v in _EXPR_3SCALAR:
            # e.g. MacdSignal(Box<Expr>, usize, usize, usize)
            return {v: [args[0].to_json(), args[1], args[2], args[3]]}

        if v in _EXPR_SCALAR_F64:
            # e.g. BollingerUpper(Box<Expr>, usize, f64)
            return {v: [args[0].to_json(), args[1], args[2]]}

        if v in _HLC_NO_SCALAR:
            # e.g. TrueRange(Box<Expr>, Box<Expr>, Box<Expr>)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json()]}

        if v == "Atr" or v in _HLC_USIZE:
            # Atr/StochK/WilliamsR/Cci/Adx/Natr(Box<Expr>, Box<Expr>, Box<Expr>, usize)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3]]}

        if v in _HLC_USIZE_F64:
            # KeltnerUpper/KeltnerLower/SuperTrend(Box<Expr>, Box<Expr>, Box<Expr>, usize, f64)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3], args[4]]}

        if v in _BINARY_EXPR:
            # Obv/CrossAbove/CrossBelow(Box<Expr>, Box<Expr>)
            return {v: [args[0].to_json(), args[1].to_json()]}

        if v in _HLCV_NO_SCALAR:
            # Vwap/AdLine(Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3].to_json()]}

        if v in _HLCV_USIZE:
            # Mfi(Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>, usize)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3].to_json(), args[4]]}

        if v == "ParabolicSar":
            # ParabolicSar(Box<Expr>, Box<Expr>, f64, f64)
            return {v: [args[0].to_json(), args[1].to_json(), args[2], args[3]]}

        if v == "IfElse":
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json()]}

        if v == "Column":
            return {"Column": args[0]}
        if v == "Literal":
            return {"Literal": scalar_value_to_json(args[0])}
        if v == "Parameter":
            return {"Parameter": args[0]}

        if v == "Function":
            return {"Function": [args[0], [a.to_json() for a in args[1]]]}

        if v == "SymbolRef":
            return {"SymbolRef": [args[0], args[1].to_json()]}

        if v == "Scan":
            state_names, init_exprs, update_names, update_exprs, output = args
            return {
                "Scan": {
                    "state_names": list(state_names),
                    "init_exprs": [e.to_json() for e in init_exprs],
                    "update_names": list(update_names),
                    "update_exprs": [e.to_json() for e in update_exprs],
                    "output": output,
                }
            }
        if v == "ScanPrev":
            return {"ScanPrev": args[0]}
        if v == "ScanVar":
            return {"ScanVar": args[0]}

        raise ValueError(f"Unknown Expr variant: {v}")

    # -- Arithmetic operators ------------------------------------------------

    def __add__(self, other: Numeric) -> Expr:
        return Expr("Add", self, _coerce(other))

    def __radd__(self, other: Numeric) -> Expr:
        return Expr("Add", _coerce(other), self)

    def __sub__(self, other: Numeric) -> Expr:
        return Expr("Sub", self, _coerce(other))

    def __rsub__(self, other: Numeric) -> Expr:
        return Expr("Sub", _coerce(other), self)

    def __mul__(self, other: Numeric) -> Expr:
        return Expr("Mul", self, _coerce(other))

    def __rmul__(self, other: Numeric) -> Expr:
        return Expr("Mul", _coerce(other), self)

    def __truediv__(self, other: Numeric) -> Expr:
        return Expr("Div", self, _coerce(other))

    def __rtruediv__(self, other: Numeric) -> Expr:
        return Expr("Div", _coerce(other), self)

    def __neg__(self) -> Expr:
        return Expr("Mul", Expr("Literal", -1.0), self)

    # -- Comparison operators ------------------------------------------------

    def __gt__(self, other: Numeric) -> Expr:
        return Expr("Gt", self, _coerce(other))

    def __lt__(self, other: Numeric) -> Expr:
        return Expr("Lt", self, _coerce(other))

    def __eq__(self, other: Numeric) -> Expr:  # type: ignore[override]
        return Expr("Eq", self, _coerce(other))

    def __ge__(self, other: Numeric) -> Expr:
        return (self > other) | (self == other)

    def __le__(self, other: Numeric) -> Expr:
        return (self < other) | (self == other)

    # -- Boolean operators ---------------------------------------------------

    def __and__(self, other: Expr) -> Expr:
        return Expr("And", self, other)

    def __or__(self, other: Expr) -> Expr:
        return Expr("Or", self, other)

    def __invert__(self) -> Expr:
        return Expr("Not", self)

    # -- Time-series methods -------------------------------------------------

    def lag(self, n: Period) -> Expr:
        return Expr("Lag", self, _resolve_period(n))

    def lead(self, n: Period) -> Expr:
        return Expr("Lead", self, _resolve_period(n))

    def diff(self, n: Period = 1) -> Expr:
        return Expr("Diff", self, _resolve_period(n))

    def pct_change(self, n: Period = 1) -> Expr:
        return Expr("PctChange", self, _resolve_period(n))

    def rolling_mean(self, window: Period) -> Expr:
        return Expr("RollingMean", self, _resolve_period(window))

    def rolling_std(self, window: Period) -> Expr:
        return Expr("RollingStd", self, _resolve_period(window))

    def rolling_sum(self, window: Period) -> Expr:
        return Expr("RollingSum", self, _resolve_period(window))

    def rolling_min(self, window: Period) -> Expr:
        return Expr("RollingMin", self, _resolve_period(window))

    def rolling_max(self, window: Period) -> Expr:
        return Expr("RollingMax", self, _resolve_period(window))

    def ewm_mean(self, span: Span) -> Expr:
        return Expr("EwmMean", self, _resolve_span(span))

    def zscore(self, window: Period) -> Expr:
        return Expr("ZScore", self, _resolve_period(window))

    def rsi(self, period: Period = 14) -> Expr:
        """Native Rust RSI (Wilder's smoothing, single-pass O(n))."""
        return Expr("Rsi", self, _resolve_period(period))

    def linreg_slope(self, window: Period) -> Expr:
        """Rolling linear regression slope (single-pass O(n))."""
        return Expr("LinRegSlope", self, _resolve_period(window))

    def linreg_value(self, window: Period) -> Expr:
        """Rolling linear regression predicted value at end of window."""
        return Expr("LinRegValue", self, _resolve_period(window))

    def linreg_r2(self, window: Period) -> Expr:
        """Rolling linear regression R-squared (single-pass O(n))."""
        return Expr("LinRegR2", self, _resolve_period(window))

    # -- New indicators (native Rust) ----------------------------------------

    def dema(self, period: Period) -> Expr:
        """Double Exponential Moving Average."""
        return Expr("Dema", self, _resolve_period(period))

    def tema(self, period: Period) -> Expr:
        """Triple Exponential Moving Average."""
        return Expr("Tema", self, _resolve_period(period))

    def wma(self, period: Period) -> Expr:
        """Weighted Moving Average."""
        return Expr("Wma", self, _resolve_period(period))

    def hma(self, period: Period) -> Expr:
        """Hull Moving Average."""
        return Expr("Hma", self, _resolve_period(period))

    def kama(self, period: Period) -> Expr:
        """Kaufman Adaptive Moving Average."""
        return Expr("Kama", self, _resolve_period(period))

    def roc(self, period: Period) -> Expr:
        """Rate of Change."""
        return Expr("Roc", self, _resolve_period(period))

    def rolling_median(self, window: Period) -> Expr:
        """Rolling median."""
        return Expr("RollingMedian", self, _resolve_period(window))

    def macd_line(self, fast: int = 12, slow: int = 26) -> Expr:
        """MACD line (fast EMA - slow EMA)."""
        return Expr("Macd", self, fast, slow)

    def macd_signal(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Expr:
        """MACD signal line."""
        return Expr("MacdSignal", self, fast, slow, signal)

    def macd_hist(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Expr:
        """MACD histogram."""
        return Expr("MacdHist", self, fast, slow, signal)

    def bollinger_upper(self, period: int = 20, num_std: float = 2.0) -> Expr:
        """Bollinger upper band."""
        return Expr("BollingerUpper", self, period, num_std)

    def bollinger_lower(self, period: int = 20, num_std: float = 2.0) -> Expr:
        """Bollinger lower band."""
        return Expr("BollingerLower", self, period, num_std)

    def bollinger_width(self, period: int = 20, num_std: float = 2.0) -> Expr:
        """Bollinger bandwidth."""
        return Expr("BollingerWidth", self, period, num_std)

    def cross_above(self, other: "Expr") -> Expr:
        """True when self crosses above other."""
        return Expr("CrossAbove", self, _coerce(other))

    def cross_below(self, other: "Expr") -> Expr:
        """True when self crosses below other."""
        return Expr("CrossBelow", self, _coerce(other))

    # -- Cumulative ----------------------------------------------------------

    def cumsum(self) -> Expr:
        return Expr("CumSum", self)

    def cumprod(self) -> Expr:
        return Expr("CumProd", self)

    def rank(self) -> Expr:
        return Expr("Rank", self)

    # -- Cross-sectional -----------------------------------------------------

    def cs_mean(self) -> Expr:
        return Expr("CrossSectionalMean", self)

    def cs_rank(self) -> Expr:
        return Expr("CrossSectionalRank", self)

    # -- Cross-asset reference -----------------------------------------------

    def of_symbol(self, symbol: str) -> Expr:
        """Reference this column from a specific symbol's data.

        Example::

            btc_close = col("close").of_symbol("BTCUSDT")
            signal = col("close") - btc_close  # ETH close minus BTC close
        """
        return Expr("SymbolRef", symbol, self)

    # -- Datetime extraction -------------------------------------------------

    def hour(self) -> Expr:
        """Extract hour (0-23) from a timestamp column (UTC)."""
        return Expr("Hour", self)

    def minute(self) -> Expr:
        """Extract minute (0-59) from a timestamp column (UTC)."""
        return Expr("Minute", self)

    def day_of_week(self) -> Expr:
        """Extract day of week from a timestamp column (0=Monday, 6=Sunday)."""
        return Expr("DayOfWeek", self)

    def month(self) -> Expr:
        """Extract month (1-12) from a timestamp column (UTC)."""
        return Expr("Month", self)

    def day_of_month(self) -> Expr:
        """Extract day of month (1-31) from a timestamp column (UTC)."""
        return Expr("DayOfMonth", self)

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        if self._variant in ("Column", "Parameter", "Literal"):
            return f"Expr.{self._variant}({self._args[0]!r})"
        return f"Expr.{self._variant}(...)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Expr:
    """Coerce a raw Python value into an Expr.Literal.

    int values are promoted to float so the Rust type-checker never sees
    Int64 vs Float64 mismatches in arithmetic/comparison expressions.
    """
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        return Expr("Literal", value)
    if isinstance(value, int):
        return Expr("Literal", float(value))
    if isinstance(value, (float, str)) or value is None:
        return Expr("Literal", value)
    raise TypeError(f"Cannot coerce {type(value).__name__} to Expr")


# ---------------------------------------------------------------------------
# Module-level factory functions (public API)
# ---------------------------------------------------------------------------


def col(name: str) -> Expr:
    """Reference a data column (e.g. ``'close'``, ``'volume'``)."""
    return Expr("Column", name)


def lit(value: Any) -> Expr:
    """Create a literal constant expression."""
    return Expr("Literal", value)


def hold() -> Expr:
    """Return NaN — tells the engine to hold the current position unchanged."""
    return Expr("Literal", float("nan"))


def param(
    name: str,
    *,
    default: Any = None,
    range: Any = None,
    description: str = "",
) -> Expr:
    """Create a parameter reference.

    The returned ``Expr`` serializes as ``Expr::Parameter(name)``.
    Metadata (default, range, description) is stored as ``_param_meta``
    and picked up by :class:`Strategy` when building the ``ParamSpec``.
    """
    expr = Expr("Parameter", name)
    expr._param_meta = {
        "name": name,
        "default": default,
        "range": range,
        "description": description,
    }
    return expr


def when(condition: Expr, true_value: Any = 1.0, false_value: Any = float("nan")) -> Expr:
    """Conditional expression (if/else).

    Omit true_value to default to 1.0 (full position, clamped by max_position_pct).
    Omit false_value to hold current position.
    """
    return Expr("IfElse", condition, _coerce(true_value), _coerce(false_value))


def exo(name: str, column: Optional[str] = None) -> Expr:
    """Reference an exogenous data column.

    Exogenous data is registered via ``bt.register_exo()`` and declared
    in ``BacktestConfig(exo_data=[...])``.

    Args:
        name: Exo series name (e.g. ``"hashrate"``).
        column: Column name within the exo series. If ``None``, defaults
                to ``name`` (convenient when the series has a single value column
                with the same name as the series).

    Returns:
        An ``Expr`` referencing ``col("exo.{name}.{column}")``.

    Example::

        # Single-column shorthand
        signal = rsi(exo("hashrate"), 14) > 70

        # Multi-column explicit
        signal = exo("onchain", "active_addresses") > 1_000_000
    """
    col_name = column if column is not None else name
    return col(f"exo.{name}.{col_name}")


def symbol_ref(symbol: str, column: str) -> Expr:
    """Reference a column from a specific symbol's data.

    Args:
        symbol: Symbol name (e.g., "BTCUSDT").
        column: Column or signal name to reference.

    Example::

        btc_momentum = symbol_ref("BTCUSDT", "momentum")
    """
    return Expr("SymbolRef", symbol, col(column))


class AssetRef:
    """Reference to a specific symbol for cross-asset column access."""

    __slots__ = ("_symbol",)

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol

    def col(self, name: str) -> Expr:
        """Reference a column from this symbol's data."""
        return Expr("SymbolRef", self._symbol, Expr("Column", name))

    def __repr__(self) -> str:
        return f"AssetRef({self._symbol!r})"


def asset(symbol: str) -> AssetRef:
    """Reference a specific symbol for cross-asset data access.

    Usage::

        btc_close = bt.asset("BTCUSDT").col("close")
        # Then define as a signal and use in downstream expressions:
        relative = bt.col("close") / bt.col("btc_close")
    """
    return AssetRef(symbol)


class TimeframeRef:
    """Reference columns from a higher timeframe.

    The columns are forward-filled: a completed 1h bar's value becomes
    available at the start of the *next* 1h bar and persists until that
    bar completes.  This avoids lookahead bias.

    Requires ``extra_timeframes`` in ``BacktestConfig``.
    """

    __slots__ = ("_tf",)

    def __init__(self, tf: str) -> None:
        self._tf = tf

    @property
    def open(self) -> Expr:
        return col(f"{self._tf}.open")

    @property
    def high(self) -> Expr:
        return col(f"{self._tf}.high")

    @property
    def low(self) -> Expr:
        return col(f"{self._tf}.low")

    @property
    def close(self) -> Expr:
        return col(f"{self._tf}.close")

    @property
    def volume(self) -> Expr:
        return col(f"{self._tf}.volume")

    def col(self, name: str) -> Expr:
        """Reference any column from this timeframe."""
        return col(f"{self._tf}.{name}")

    def __repr__(self) -> str:
        return f"TimeframeRef({self._tf!r})"


def tf(timeframe: str) -> TimeframeRef:
    """Reference a higher timeframe for multi-TF strategies.

    Usage::

        h1 = bt.tf("1h")
        trend = ema(h1.close, 20) > ema(h1.close, 50)

    Requires ``extra_timeframes={"1h": Interval.hours(1)}`` in config.
    """
    return TimeframeRef(timeframe)


# ---------------------------------------------------------------------------
# Scan (stateful fold) support
# ---------------------------------------------------------------------------


class _ScanState:
    """Helper to build ``ScanPrev`` / ``ScanVar`` references inside a scan.

    Usage::

        from manifoldbt.expr import s, scan

        kalman = scan(
            state={"x": col("close"), "p": lit(1.0)},
            update={
                "p_pred": s.prev("p") + param("q"),
                "k":      s.var("p_pred") / (s.var("p_pred") + param("r")),
                "x":      s.prev("x") + s.var("k") * (col("close") - s.prev("x")),
                "p":      (lit(1.0) - s.var("k")) * s.var("p_pred"),
            },
            output="x",
        )
    """

    __slots__ = ()

    def prev(self, name: str) -> Expr:
        """Reference a state variable's value at t-1."""
        return Expr("ScanPrev", name)

    def var(self, name: str) -> Expr:
        """Reference a variable computed earlier in the current scan step."""
        return Expr("ScanVar", name)


s = _ScanState()
"""Singleton for building scan state references: ``s.prev("x")``, ``s.var("k")``."""


def scan(
    state: "dict[str, Expr]",
    update: "dict[str, Expr]",
    output: str,
) -> Expr:
    """Create a stateful scan (fold) expression.

    The scan executes entirely in Rust as a flat register-based scalar VM —
    no Python callbacks, no Arrow overhead per row.

    Args:
        state: Initial state variables. Keys are names, values are ``Expr``
            objects whose first-row value seeds the state.
        update: Ordered dict of update expressions. Each expression can
            reference ``s.prev("name")`` for previous state and
            ``s.var("name")`` for variables computed earlier in the same step.
            If an update name matches a state name, it writes back to that state.
        output: Name of the update variable to emit as the scan output.

    Returns:
        An ``Expr`` that evaluates to a Float64 array.

    Example::

        # Exponential moving average via scan
        ema_scan = scan(
            state={"ema": col("close")},
            update={"ema": s.prev("ema") * lit(0.9) + col("close") * lit(0.1)},
            output="ema",
        )
    """
    state_names = list(state.keys())
    init_exprs = [_coerce(v) for v in state.values()]
    update_names = list(update.keys())
    update_exprs = [_coerce(v) for v in update.values()]
    return Expr("Scan", state_names, init_exprs, update_names, update_exprs, output)
