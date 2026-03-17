"""Technical indicators built on top of the Expr DSL.

All functions return ``Expr`` objects that compose into the expression graph
evaluated by the Rust engine. No data is touched at definition time.

Usage::

    from manifoldbt.indicators import sma, rsi, macd, bollinger_bands

    fast = sma(close, 20)
    slow = sma(close, 60)
    my_rsi = rsi(close, 14)
"""
from __future__ import annotations

from typing import Tuple

from manifoldbt.expr import Expr, col, lit, when, _coerce, s, scan

# ---------------------------------------------------------------------------
# Pre-built column references
# ---------------------------------------------------------------------------

open = col("open")
high = col("high")
low = col("low")
close = col("close")
volume = col("volume")
vwap = col("vwap")
timestamp = col("timestamp")

# ---------------------------------------------------------------------------
# Math helpers (wrapping Rust built-in functions)
# ---------------------------------------------------------------------------


def abs_val(x: Expr) -> Expr:
    """Absolute value (element-wise)."""
    return Expr("Function", "abs", [_coerce(x)])


def sqrt(x: Expr) -> Expr:
    """Square root (element-wise)."""
    return Expr("Function", "sqrt", [_coerce(x)])


def log(x: Expr) -> Expr:
    """Natural logarithm (element-wise)."""
    return Expr("Function", "log", [_coerce(x)])


def exp(x: Expr) -> Expr:
    """Exponential e^x (element-wise)."""
    return Expr("Function", "exp", [_coerce(x)])


def max_val(a: Expr, b: Expr) -> Expr:
    """Element-wise maximum of two expressions."""
    return Expr("Function", "max", [_coerce(a), _coerce(b)])


def min_val(a: Expr, b: Expr) -> Expr:
    """Element-wise minimum of two expressions."""
    return Expr("Function", "min", [_coerce(a), _coerce(b)])


# ---------------------------------------------------------------------------
# Trend / Moving averages
# ---------------------------------------------------------------------------


def sma(source: Expr, period) -> Expr:
    """Simple Moving Average. Period can be int or param()."""
    return source.rolling_mean(period)


def ema(source: Expr, span) -> Expr:
    """Exponential Moving Average (span-based). Span can be int or param()."""
    return source.ewm_mean(span)


def dema(source: Expr, period=14) -> Expr:
    """Double Exponential Moving Average. Period can be int or param()."""
    return source.dema(period)


def tema(source: Expr, period=14) -> Expr:
    """Triple Exponential Moving Average. Period can be int or param()."""
    return source.tema(period)


def wma(source: Expr, period=14) -> Expr:
    """Weighted Moving Average. Period can be int or param()."""
    return source.wma(period)


def hma(source: Expr, period=14) -> Expr:
    """Hull Moving Average. Period can be int or param()."""
    return source.hma(period)


def kama(source: Expr, period=10) -> Expr:
    """Kaufman Adaptive Moving Average. Period can be int or param()."""
    return source.kama(period)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def roc(source: Expr, period=1) -> Expr:
    """Rate of Change. Period can be int or param()."""
    return source.roc(period)


def momentum(source: Expr, period=1) -> Expr:
    """Momentum (raw price difference). Period can be int or param()."""
    return source.diff(period)


def rsi(source: Expr, period=14) -> Expr:
    """Relative Strength Index (native Rust, Wilder's smoothing, single-pass O(n)).

    Returns an expression in [0, 100]. Values below 30 are typically
    considered oversold, above 70 overbought.
    """
    return source.rsi(period)


def stoch_k(period: int = 14) -> Expr:
    """Stochastic %K oscillator (native Rust, uses high/low/close)."""
    return Expr("StochK", high, low, close, period)


def stochastic_k(period: int = 14, source: Expr = None) -> Expr:
    """Stochastic %K oscillator (DSL-based fallback).

    ``(close - lowest_low) / (highest_high - lowest_low) * 100``
    """
    c = source if source is not None else close
    lowest = c.rolling_min(period)
    highest = c.rolling_max(period)
    return (c - lowest) / (highest - lowest + lit(1e-12)) * lit(100.0)


def williams_r(period: int = 14) -> Expr:
    """Williams %R oscillator (native Rust, uses high/low/close)."""
    return Expr("WilliamsR", high, low, close, period)


def cci(period: int = 20) -> Expr:
    """Commodity Channel Index (native Rust, uses high/low/close)."""
    return Expr("Cci", high, low, close, period)


def adx(period: int = 14) -> Expr:
    """Average Directional Index (native Rust, uses high/low/close)."""
    return Expr("Adx", high, low, close, period)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def bollinger_bands(
    source: Expr, period: int = 20, num_std: float = 2.0
) -> Tuple[Expr, Expr, Expr]:
    """Bollinger Bands (native Rust).

    Returns:
        ``(upper, middle, lower)`` — three ``Expr`` objects.
    """
    upper = source.bollinger_upper(period, num_std)
    middle = source.rolling_mean(period)
    lower = source.bollinger_lower(period, num_std)
    return upper, middle, lower


def bollinger_width(source: Expr, period: int = 20, num_std: float = 2.0) -> Expr:
    """Bollinger Bandwidth (native Rust)."""
    return source.bollinger_width(period, num_std)


def atr(period: int = 14) -> Expr:
    """Average True Range (native Rust, Wilder's smoothing, single-pass O(n)).

    Uses ``high``, ``low``, ``close`` columns from the bar data.
    """
    return Expr("Atr", high, low, close, period)


def true_range() -> Expr:
    """True Range (native Rust, uses high/low/close)."""
    return Expr("TrueRange", high, low, close)


def natr(period: int = 14) -> Expr:
    """Normalized ATR (native Rust, uses high/low/close)."""
    return Expr("Natr", high, low, close, period)


def keltner_channels(period: int = 20, multiplier: float = 1.5) -> Tuple[Expr, Expr, Expr]:
    """Keltner Channels (native Rust, uses high/low/close).

    Returns:
        ``(upper, middle, lower)`` — three ``Expr`` objects.
    """
    upper = Expr("KeltnerUpper", high, low, close, period, multiplier)
    middle = close.ewm_mean(float(period))
    lower = Expr("KeltnerLower", high, low, close, period, multiplier)
    return upper, middle, lower


def supertrend(period: int = 10, multiplier: float = 3.0) -> Expr:
    """SuperTrend indicator (native Rust, uses high/low/close)."""
    return Expr("SuperTrend", high, low, close, period, multiplier)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    source: Expr,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[Expr, Expr, Expr]:
    """Moving Average Convergence Divergence (native Rust).

    Returns:
        ``(macd_line, signal_line, histogram)`` — three ``Expr`` objects.
    """
    macd_line = source.macd_line(fast_period, slow_period)
    signal_line = source.macd_signal(fast_period, slow_period, signal_period)
    histogram = source.macd_hist(fast_period, slow_period, signal_period)
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Crossover signals
# ---------------------------------------------------------------------------


def crossover(a: Expr, b: Expr) -> Expr:
    """True on bars where ``a`` crosses above ``b`` (native Rust)."""
    return a.cross_above(b)


def crossunder(a: Expr, b: Expr) -> Expr:
    """True on bars where ``a`` crosses below ``b`` (native Rust)."""
    return a.cross_below(b)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def obv(source: Expr = None, vol: Expr = None) -> Expr:
    """On-Balance Volume (native Rust).

    Args:
        source: Price series. Defaults to ``close``.
        vol: Volume series. Defaults to ``volume``.
    """
    return Expr("Obv", source if source is not None else close,
                vol if vol is not None else volume)


def vwap() -> Expr:
    """Volume Weighted Average Price (native Rust, uses high/low/close/volume)."""
    return Expr("Vwap", high, low, close, volume)


def ad_line() -> Expr:
    """Accumulation/Distribution Line (native Rust, uses high/low/close/volume)."""
    return Expr("AdLine", high, low, close, volume)


def mfi(period: int = 14) -> Expr:
    """Money Flow Index (native Rust, uses high/low/close/volume)."""
    return Expr("Mfi", high, low, close, volume, period)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def rolling_median(source: Expr, window: int) -> Expr:
    """Rolling median (native Rust)."""
    return source.rolling_median(window)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def parabolic_sar(af_start: float = 0.02, af_max: float = 0.2) -> Expr:
    """Parabolic SAR (native Rust, uses high/low)."""
    return Expr("ParabolicSar", high, low, af_start, af_max)


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


def linreg_slope(source: Expr, window: int) -> Expr:
    """Rolling linear regression slope (native Rust, single-pass O(n)).

    Fits y = a + b*x over a rolling window and returns the slope b.
    """
    return source.linreg_slope(window)


def linreg_value(source: Expr, window: int) -> Expr:
    """Rolling linear regression predicted value (native Rust, single-pass O(n)).

    Returns the predicted y at the last point of the rolling window.
    Equivalent to ``mean + slope * (window - 1) / 2``.
    """
    return source.linreg_value(window)


def linreg_r2(source: Expr, window: int) -> Expr:
    """Rolling linear regression R-squared (native Rust, single-pass O(n)).

    Returns the coefficient of determination in [0, 1].
    NaN when the series is constant within the window.
    """
    return source.linreg_r2(window)


# ---------------------------------------------------------------------------
# Datetime extraction
# ---------------------------------------------------------------------------


def hour(source: Expr = None) -> Expr:
    """Extract hour (0-23 UTC) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.

    Usage::

        # Trade only during US equity hours (14:30-21:00 UTC)
        us_hours = (hour() >= 14) & (hour() < 21)
    """
    return (source if source is not None else timestamp).hour()


def minute(source: Expr = None) -> Expr:
    """Extract minute (0-59) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.
    """
    return (source if source is not None else timestamp).minute()


def day_of_week(source: Expr = None) -> Expr:
    """Extract day of week from a timestamp column (0=Monday, 6=Sunday).

    Defaults to the ``timestamp`` bar column if no source given.

    Usage::

        # Only trade on weekdays
        is_weekday = day_of_week() < 5
    """
    return (source if source is not None else timestamp).day_of_week()


def month(source: Expr = None) -> Expr:
    """Extract month (1-12) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.

    Usage::

        # Seasonal filter: trade only Q4 (Oct-Dec)
        is_q4 = month() >= 10
    """
    return (source if source is not None else timestamp).month()


def day_of_month(source: Expr = None) -> Expr:
    """Extract day of month (1-31) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.
    """
    return (source if source is not None else timestamp).day_of_month()


# ---------------------------------------------------------------------------
# Scan-based indicators (arbitrary stateful computations)
# ---------------------------------------------------------------------------


def kalman(source: Expr = None, q: float = 1e-5, r: float = 1e-2) -> Expr:
    """Kalman filter (1-D constant-velocity model).

    Uses the ``scan`` primitive — runs entirely in Rust as a flat scalar VM.

    Args:
        source: Input price series. Defaults to ``close``.
        q: Process noise covariance (how much the true value can change per step).
        r: Measurement noise covariance (how noisy the observations are).

    Returns:
        Smoothed estimate ``Expr`` (Float64 array).
    """
    src = source if source is not None else close
    return scan(
        state={"x": src, "p": lit(1.0)},
        update={
            "p_pred": s.prev("p") + _coerce(q),
            "k": s.var("p_pred") / (s.var("p_pred") + _coerce(r)),
            "x": s.prev("x") + s.var("k") * (src - s.prev("x")),
            "p": (lit(1.0) - s.var("k")) * s.var("p_pred"),
        },
        output="x",
    )


def garch(source: Expr = None, omega: float = 1e-6, alpha: float = 0.1, beta: float = 0.85) -> Expr:
    """GARCH(1,1) conditional volatility estimator.

    Uses the ``scan`` primitive — runs entirely in Rust.

    Args:
        source: Return series. Defaults to ``close.pct_change(1)``.
        omega: Long-run variance weight.
        alpha: Weight on lagged squared return (ARCH term).
        beta: Weight on lagged conditional variance (GARCH term).

    Returns:
        Conditional standard deviation ``Expr`` (Float64 array).
    """
    src = source if source is not None else close.pct_change(1)
    return scan(
        state={"sigma2": lit(omega / (1.0 - alpha - beta)), "ret": src},
        update={
            "ret": src,
            "sigma2": _coerce(omega)
                + _coerce(alpha) * s.prev("ret") * s.prev("ret")
                + _coerce(beta) * s.prev("sigma2"),
            "sigma": Expr("Function", "sqrt", [s.var("sigma2")]),
        },
        output="sigma",
    )
