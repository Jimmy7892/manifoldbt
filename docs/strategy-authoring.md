# Strategy Authoring Guide

> **manifoldbt** — Python DSL for Declarative Strategy Definition

This guide describes how to define trading strategies using the manifoldbt Python DSL. Strategies are compiled into an optimized expression graph and executed by the Rust vectorized engine.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Indicators](#indicators)
3. [Signals & Sizing](#signals--sizing)
4. [Parameters & Sweeps](#parameters--sweeps)
5. [Backtest Configuration](#backtest-configuration)
6. [Execution Model](#execution-model)
7. [Fee & Slippage Models](#fee--slippage-models)
8. [Orders (SL/TP/Trailing)](#orders-sltp-trailing)
9. [Cross-Asset References](#cross-asset-references)
10. [Dataset Auto-Resolution](#dataset-auto-resolution)
11. [Diagnostics](#diagnostics)
12. [Profiling](#profiling)
13. [Complete Examples](#complete-examples)
14. [Indicator Reference](#indicator-reference)

---

## Quick Start

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Indicators
fast = ema(close, 12)
slow = ema(close, 50)

# -- Strategy
strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.5, 0.0))
    .stop_loss(pct=3.0)
)

# -- Config
start, end = time_range("2022-01-01", "2025-01-01")
config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,
)

# -- Run
store = mbt.DataStore(data_root="data", metadata_db="metadata/metadata.sqlite")
result = mbt.run(strategy, config, store)
print(result.summary())
```

---

## Indicators

All indicators are available from `manifoldbt.indicators`. They return `Expr` objects that compose into the expression graph — no data is touched at definition time.

```python
from manifoldbt.indicators import (
    close, open, high, low, volume,  # price columns
    ema, sma, dema, tema, wma, hma, kama,  # moving averages
    rsi, roc, momentum, macd,  # momentum
    bollinger_bands, atr, natr, keltner_channels,  # volatility
    stoch_k, williams_r, cci, adx,  # oscillators
    obv, vwap, mfi,  # volume
    kalman, garch,  # filters
)
```

### Usage

```python
fast = ema(close, 12)          # EMA with span 12
slow = sma(close, 50)          # SMA with window 50
strength = rsi(close, 14)      # RSI with period 14
upper, mid, lower = bollinger_bands(close, period=20, num_std=2.0)
```

### Method chaining

Column expressions (`close`, `high`, etc.) support method chaining:

```python
zscore = close.zscore(60)           # rolling z-score
slope = close.linreg_slope(20)      # linear regression slope
smoothed = close.ewm_mean(12)       # EMA
lagged = close.lag(5)               # 5-bar lag
ret = close.pct_change(1)           # 1-bar return
```

---

## Signals & Sizing

### Strategy builder

```python
strategy = (
    mbt.Strategy.create("my_strategy")
    .signal("fast", fast)           # named signal
    .signal("slow", slow)           # signals form a DAG
    .size(signal_expr)              # position sizing expression
    .describe("Strategy description")
)
```

### `mbt.when()` — conditional logic

```python
# Long when fast > slow, flat otherwise
signal = mbt.when(fast > slow, 0.5, 0.0)

# Nested: long / short / flat
signal = mbt.when(fast > slow, 0.25,
         mbt.when(fast < slow, -0.25, 0.0))

# Hold current position (omit 3rd arg or use NaN)
signal = mbt.when(rsi < 30, 1.0)  # buy oversold, hold otherwise
```

### Arithmetic on expressions

```python
trend = fast - slow
spread = close / (pair_close + mbt.lit(1e-12))   # mbt.lit() for constants in arithmetic
signal = -spread_z * mbt.lit(0.05)                # negation + scaling
```

> **Note:** `mbt.lit()` is needed for constants in arithmetic (`close + mbt.lit(1e-12)`). Numbers auto-coerce inside `mbt.when()`.

### Sizing modes

| Mode                         | Meaning                                              |
|------------------------------|------------------------------------------------------|
| `FractionOfEquity` (default) | `1.0` = allocate 100% of current equity              |
| `FractionOfInitialCapital`   | `1.0` = allocate 100% of initial capital (no compounding) |
| `Units`                      | `1.0` = hold exactly 1 unit (share/contract/coin)    |

```python
execution=mbt.ExecutionConfig(position_sizing_mode="FractionOfInitialCapital")
```

### Special values

| Value  | Behavior                                |
|--------|-----------------------------------------|
| `1.0`  | Full long position                      |
| `0.0`  | Flat (close position)                   |
| `-0.5` | Short 50% (requires `allow_short=True`) |
| `NaN`  | Hold current position unchanged         |

---

## Parameters & Sweeps

Use `mbt.param()` to define sweepable parameters in indicator periods:

```python
fast = ema(close, mbt.param("fast", default=12))
slow = ema(close, mbt.param("slow", default=50))

strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.25, -0.25))
)
```

Parameters are auto-collected from expressions — no `.param()` needed on the Strategy.

### Sweep execution

```python
# Full sweep (returns Result per combo)
sweep = mbt.run_sweep(strategy, {"fast": [5, 12, 20], "slow": [50, 100]}, config, store)
best = sweep.best("sharpe")

# Lite sweep (metrics only, much faster for large grids)
batch = mbt.run_sweep_lite(strategy, {"fast": range(5, 100), "slow": range(10, 500)}, config, store)
```

`run_sweep_lite` is optimized for large parameter grids (100k+ combos):
- Cartesian product expansion in Rust (no Python loop)
- Shared indicator cache (EMA(12) computed once, reused across combos)
- Pre-resampled bars (no per-combo resample overhead)
- Metrics only — no Arrow output

---

## Backtest Configuration

```python
config = mbt.BacktestConfig(
    universe=[1, 2],                       # symbol IDs
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(4),        # signal evaluation resolution
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(...),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,                        # bars to skip for indicator warmup
    accuracy=False,                        # True = simulate on 1-min bars
)
```

### Bar intervals

```python
Interval.minutes(1)    # 1-min
Interval.minutes(15)   # 15-min
Interval.hours(1)      # 1-hour
Interval.hours(4)      # 4-hour
Interval.hours(12)     # 12-hour
Interval.days(1)       # daily
```

### Accuracy mode

```python
config = mbt.BacktestConfig(
    bar_interval=Interval.hours(4),   # signals on 4h
    accuracy=True,                    # simulation on 1-min bars
    ...
)
```

When `accuracy=True`, the engine loads `bars_1m` and runs in hybrid mode: signals evaluated on `bar_interval`, simulation tick-by-tick on 1-min bars. Use for precise SL/TP fill detection. ~60x slower than normal mode.

---

## Execution Model

```python
mbt.ExecutionConfig(
    signal_delay=1,                    # bars between signal and execution
    execution_price="AtClose",         # fill price: AtClose, AtOpen, AtVwap, MidPrice
    max_position_pct=0.5,              # max position as fraction of equity
    allow_short=True,                  # allow short positions
    allow_fractional=True,             # allow fractional units
    position_sizing_mode="FractionOfEquity",
    pyramiding=False,                  # True = signal is delta, not target
)
```

### Signal delay

| Value | Behavior                                          |
|-------|---------------------------------------------------|
| `0`   | Execute same bar (look-ahead bias risk)           |
| `1`   | **Default.** Execute next bar (t+1)               |
| `2+`  | Execute N bars after signal                       |

---

## Fee & Slippage Models

### Fees

```python
mbt.FeeConfig.binance_perps()    # maker=2bps, taker=5bps, funding
mbt.FeeConfig.binance_spot()     # maker=10bps, taker=10bps
mbt.FeeConfig.zero()             # no fees (for development)

# Custom
mbt.FeeConfig(
    maker_fee_bps=2.0,
    taker_fee_bps=5.0,
    funding_rate_column="funding_rate",
    default_fill_type="Taker",
)
```

### Slippage

```python
Slippage.fixed_bps(2)     # 2 bps per trade (simplest)
Slippage.volume_impact(0.1, exponent=0.5)   # qty/volume model
Slippage.spread_based(0.5)                  # spread-based
```

---

## Orders (SL/TP/Trailing)

```python
strategy = (
    mbt.Strategy.create("my_strat")
    .signal(...)
    .size(...)
    .stop_loss(pct=3.0)           # 3% stop-loss
    .take_profit(pct=5.0)         # 5% take-profit
    .trailing_stop(pct=2.0)       # 2% trailing stop
)
```

---

## Cross-Asset References

Use `mbt.symbol_ref()` to reference another symbol's data in multi-asset strategies:

```python
pair_close = mbt.symbol_ref("ETHUSDT", "close")
ratio = close / (pair_close + mbt.lit(1e-12))
```

> **Important:** Expressions using `symbol_ref()` must be registered as named signals (`.signal("name", expr)`), not passed directly to `.size()`. The multi-pass evaluator needs named signals to route cross-asset data correctly.

```python
# Required: symbol_names mapping
config = mbt.BacktestConfig(
    universe=[1, 2, 5],
    symbol_names={"BTCUSDT": 1, "ETHUSDT": 2, "BNBUSDT": 5},
    ...
)
```

---

## Dataset Auto-Resolution

The engine automatically selects the best dataset based on `bar_interval`:

| bar_interval     | Dataset loaded  | Bars (5 years) |
|------------------|-----------------|----------------|
| 1 min            | `bars_1m`       | ~2.6M          |
| 15 min           | `bars_15m`      | ~175k          |
| 1h - 23h         | `bars_1h`       | ~44k           |
| >= 24h           | `bars_1d`       | ~1.8k          |

When `bar_interval` doesn't exactly match a dataset (e.g. `4h`), the engine loads the closest smaller dataset (`bars_1h`) and pre-resamples to `4h` before simulation.

Override with `accuracy=True` to always load `bars_1m` (precise SL/TP fills).

Override manually with `dataset=`:
```python
store = mbt.DataStore(data_root="data", metadata_db="...", dataset="bars_1m")
```

---

## Diagnostics

```python
# Look-ahead bias detection
lookahead = mbt.diagnostics.detect_lookahead(strategy, config, store)
print(lookahead)  # PASS or FAIL with details

# Exposure stability (position consistency across time windows)
stability = mbt.diagnostics.check_exposure_stability(strategy, config, store)

# Post-run risk check
result = mbt.run(strategy, config, store)
risk = mbt.diagnostics.risk_check(result)
```

---

## Profiling

Every result includes microsecond-precision timing:

```python
result = mbt.run(strategy, config, store)
print(result.profile)
# {'data_load_us': 45000, 'align_us': 1000, 'signal_eval_us': 28000,
#  'runtime_prep_us': 500, 'simulation_us': 16000, 'output_build_us': 8000,
#  'total_us': 110000}

print(result.profile_summary())
# Profile (total: 110.0ms)
# ----------------------------------------
#   Data loading      45.0ms   40.9%  ################
#   Signal eval       28.0ms   25.5%  ##########
#   Simulation        16.0ms   14.5%  #####
#   ...
```

---

## Complete Examples

### Trend Following — EMA Crossover

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

fast = ema(close, 12)
slow = ema(close, 50)

strategy = (
    mbt.Strategy.create("trend_following")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.5, 0.0))
    .stop_loss(pct=3.0)
)

start, end = time_range("2022-01-01", "2025-01-01")
config = mbt.BacktestConfig(
    universe=[1], time_range_start=start, time_range_end=end,
    bar_interval=Interval.hours(12), initial_capital=10_000,
    fees=mbt.FeeConfig.binance_perps(), slippage=Slippage.fixed_bps(2),
    warmup_bars=60,
)
store = mbt.DataStore(data_root="data", metadata_db="metadata/metadata.sqlite")
result = mbt.run(strategy, config, store)
print(result.summary())
```

### Parameter Sweep — 2D Heatmap

```python
fast = ema(close, mbt.param("fast", default=12))
slow = ema(close, mbt.param("slow", default=50))

strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.25, -0.25))
)

batch = mbt.run_sweep_lite(
    strategy,
    {"fast": list(range(5, 100)), "slow": list(range(10, 500))},
    config, store,
)

# Build metric grid and visualize
mbt.plot.heatmap_2d({...}, show=True)
mbt.plot.surface_3d({...}, show=True)
```

### Statistical Arbitrage — Cross-Asset

```python
pair_close = mbt.symbol_ref("ETHUSDT", "close")
ratio = close / (pair_close + mbt.lit(1e-12))
equilibrium = kalman(ratio, q=1e-4, r=1e-2)
spread_z = (ratio - equilibrium).zscore(28)

strategy = (
    mbt.Strategy.create("stat_arb")
    .signal("pair_close", pair_close)
    .signal("spread_z", spread_z)
    .signal("signal", -spread_z)
    .size(mbt.col("signal"))
)

config = mbt.BacktestConfig(
    universe=[1, 2, 5],
    symbol_names={"BTCUSDT": 1, "ETHUSDT": 2, "BNBUSDT": 5},
    ...
)
```

---

## Indicator Reference

### Moving Averages

| Function | Description |
|----------|-------------|
| `sma(source, period)` | Simple Moving Average |
| `ema(source, span)` | Exponential Moving Average |
| `dema(source, period)` | Double EMA |
| `tema(source, period)` | Triple EMA |
| `wma(source, period)` | Weighted MA |
| `hma(source, period)` | Hull MA |
| `kama(source, period)` | Kaufman Adaptive MA |

### Momentum

| Function | Description |
|----------|-------------|
| `rsi(source, period)` | Relative Strength Index [0-100] |
| `roc(source, period)` | Rate of Change |
| `momentum(source, period)` | Raw price difference |
| `macd(source, fast, slow)` | MACD line |
| `stoch_k(period)` | Stochastic %K |
| `williams_r(period)` | Williams %R |
| `cci(period)` | Commodity Channel Index |
| `adx(period)` | Average Directional Index |

### Volatility

| Function | Description |
|----------|-------------|
| `atr(period)` | Average True Range |
| `natr(period)` | Normalized ATR |
| `bollinger_bands(source, period, num_std)` | Returns (upper, middle, lower) |
| `keltner_channels(period, multiplier)` | Returns (upper, middle, lower) |

### Volume

| Function | Description |
|----------|-------------|
| `obv(source, vol)` | On-Balance Volume |
| `vwap()` | Volume-Weighted Average Price |
| `mfi(period)` | Money Flow Index |

### Filters

| Function | Description |
|----------|-------------|
| `kalman(source, q, r)` | Kalman filter |
| `garch(source, omega, alpha, beta)` | GARCH volatility |

### Statistics

| Function | Description |
|----------|-------------|
| `source.zscore(window)` | Rolling z-score |
| `source.linreg_slope(window)` | Linear regression slope |
| `source.linreg_value(window)` | Linear regression fitted value |
| `source.linreg_r2(window)` | Linear regression R-squared |
| `source.rolling_median(window)` | Rolling median |

### Time

| Function | Description |
|----------|-------------|
| `source.lag(n)` | Value n bars ago |
| `source.lead(n)` | Value n bars ahead |
| `source.diff(n)` | Difference over n bars |
| `source.pct_change(n)` | Percentage change over n bars |
| `source.rolling_mean(w)` | Rolling mean |
| `source.rolling_std(w)` | Rolling standard deviation |
| `source.cumsum()` | Cumulative sum |

> All period/window arguments accept `mbt.param("name", default)` for sweep grids.

---

## Metrics Reference

Every result includes these performance metrics:

| Metric | Description |
|--------|-------------|
| `total_return` | Total return |
| `cagr` | Compound Annual Growth Rate |
| `volatility` | Annualized volatility |
| `sharpe` | Sharpe ratio |
| `sortino` | Sortino ratio |
| `calmar` | Calmar ratio |
| `max_drawdown` | Maximum drawdown |
| `tstat_sharpe` | t-statistic of Sharpe (sharpe * sqrt(years)) |
| `alpha` | Annualized CAPM alpha vs buy-and-hold benchmark |
| `beta` | Beta to benchmark |
| `tstat_alpha` | t-statistic of alpha (OLS regression) |

---

## Best Practices

1. **Use `signal_delay=1`** (default). `signal_delay=0` introduces look-ahead bias.
2. **Set `warmup_bars`** to at least the longest indicator period.
3. **Use `mbt.when()` for sizing.** Keep signal logic readable and composable.
4. **Run diagnostics** (`detect_lookahead`, `check_exposure_stability`) on new strategies.
5. **Start with `bar_interval=hours(12)` or `days(1)`** for fast iteration, then refine with smaller intervals.
6. **Use `accuracy=True`** only for final validation with SL/TP — it's 60x slower.
7. **Sweep with `run_sweep_lite`** for large grids. Use `run_sweep` only when you need full Result objects.
