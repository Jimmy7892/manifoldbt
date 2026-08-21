<p align="center">
  <img src="https://raw.githubusercontent.com/manifoldbt/manifoldbt/master/assets/logo.png" width="110" alt="ManifoldBT logo">
</p>

<p align="center">
  <strong>ManifoldBT</strong><br>
  Rust-powered backtesting engine for quantitative research
</p>

<p align="center">
  <a href="https://discord.gg/bvU6Wjc72d"><img src="https://img.shields.io/badge/Discord-join%20the%20community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://www.manifoldbt.com">Website</a> &middot;
  <a href="https://www.manifoldbt.com/docs/documentation.html">Documentation</a> &middot;
  <a href="https://github.com/manifoldbt/manifoldbt/tree/master/examples">Examples</a>
</p>

---

ManifoldBT is a Python backtesting library with a Rust core. Strategies are written in a
fluent Python DSL, compiled to a vectorized Rust expression graph, then run through a
sequential fill simulation with realistic fees, slippage, funding and look-ahead protection.
**Vectorized speed with event-driven execution realism.**

## Why ManifoldBT

- **Fast**: 10M bars in 317 ms. 78x faster than vectorbt, 308x once you also want drawdown and Sharpe, ~3,500x faster than backtrader. [Measured in public CI](#performance), every run linked.
- **Expressive**: fluent DSL with 30+ indicators, conditional logic, cross-asset references
- **Rigorous**: Monte Carlo, walk-forward, parameter sweeps, lookahead detection, exposure diagnostics
- **Portable**: `pip install`, no Rust toolchain needed. Works on Python 3.9+.

## Installation

```bash
pip install manifoldbt              # engine only: backtests, sweeps, metrics
pip install manifoldbt[plot]        # + interactive charts and native windows (show=True)
pip install manifoldbt[all]         # everything: plots, windows, PNG export, pandas/polars
pip install manifoldbt[gpu]         # + NVIDIA runtime compiler, for device="cuda" (Pro)
```

The base install stays light (no browser, no GUI) for scripts, servers and CI.
`[plot]` adds plotly and a native window backend; `[all]` also pulls kaleido for
static PNG/SVG export (which bundles a headless Chromium).

The Linux and Windows x86_64 wheels already carry the CUDA kernels, so `[gpu]`
only adds the NVIDIA runtime compiler (~180 MB) that compiles them on your
machine. Skip it if you already have a CUDA toolkit installed. An NVIDIA driver
is required, and GPU acceleration is a Pro feature; everything else runs at full
speed on the CPU.

## Quick Start

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Interval, Slippage

fast = ema(close, 12)
slow = ema(close, 26)

strategy = (
    mbt.Strategy.create("ema_crossover")
    .signal("fast", fast)
    .signal("slow", slow)
    .signal("signal", mbt.when(fast > slow, mbt.lit(1.0), mbt.lit(-1.0)))
    .size(mbt.col("signal") * mbt.lit(0.25))
)

start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(allow_short=True, max_position_pct=0.5),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=30,
)

store = mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1,
                   start="2022-01-01T00:00:00Z", end="2025-01-01T00:00:00Z", interval="1h")
result = mbt.run(strategy, config, store)
print(result.summary())
```

## Loading data

Bring your own data, or pull it from a built-in connector. Both return a
`DataStore` ready for `mbt.run(...)`.

**CSV**, free on all tiers, auto-detects standard / MetaTrader 4 / MetaTrader 5:

```python
store = mbt.import_csv("EURUSD_1m.csv", symbol="EURUSD", symbol_id=1,
                       interval="1m", asset_class="forex")
```

**Exchange connectors**: Binance, Bybit, Hyperliquid, dYdX, Bitstamp (free); Databento, Massive (Pro):

```python
store = mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1,
                   start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z")
```

Or from the CLI:

```bash
manifoldbt import-csv data.csv --symbol EURUSD --symbol-id 1 --interval 1m
manifoldbt ingest --provider binance --symbol BTCUSDT --symbol-id 1 --start ... --end ...
```

## Examples

| # | Example | What it shows |
|---|---------|---------------|
| 00 | [Template](https://github.com/manifoldbt/manifoldbt/blob/master/examples/00_template.py) | Minimal starting point |
| 01 | [Trend Following](https://github.com/manifoldbt/manifoldbt/blob/master/examples/01_trend_following.py) | EMA crossover, volume filter, stop-loss |
| 02 | [Mean Reversion](https://github.com/manifoldbt/manifoldbt/blob/master/examples/02_mean_reversion.py) | EMA crossover with parameter sweep |
| 03 | [Multi-Asset Momentum](https://github.com/manifoldbt/manifoldbt/blob/master/examples/03_multi_asset_momentum.py) | Cross-asset signals |
| 04 | [Linear Regression](https://github.com/manifoldbt/manifoldbt/blob/master/examples/04_linear_regression.py) | Regression-based signal |
| 05 | [Statistical Arbitrage](https://github.com/manifoldbt/manifoldbt/blob/master/examples/05_stat_arb.py) | Pairs trading, spread z-score |
| 06 | [Full Visualization](https://github.com/manifoldbt/manifoldbt/blob/master/examples/06_full_visualization.py) | Tearsheet and charts |
| 07 | [Walk-Forward](https://github.com/manifoldbt/manifoldbt/blob/master/examples/07_walk_forward.py) | Out-of-sample validation |
| 08 | [2D Sweep](https://github.com/manifoldbt/manifoldbt/blob/master/examples/08_sweep_2d_heatmap.py) | Parameter grid heatmap |
| 09 | [3D Surface](https://github.com/manifoldbt/manifoldbt/blob/master/examples/09_surface_3d.py) | Parameter surface plot |
| 10 | [Monte Carlo](https://github.com/manifoldbt/manifoldbt/blob/master/examples/10_monte_carlo.py) | Permutation-based robustness |
| 11 | [Portfolio](https://github.com/manifoldbt/manifoldbt/blob/master/examples/11_portfolio.py) | Multi-strategy portfolio |
| 12 | [Diagnostics](https://github.com/manifoldbt/manifoldbt/blob/master/examples/12_diagnostics.py) | Lookahead & exposure safety checks |
| 13 | [Stochastic Simulation](https://github.com/manifoldbt/manifoldbt/blob/master/examples/13_stochastic_simulation.py) | SDE path simulation (GBM, Heston, …) |
| 14 | [Multi-Timeframe](https://github.com/manifoldbt/manifoldbt/blob/master/examples/14_multi_timeframe.py) | Combining signals across timeframes |
| 15 | [Cross-Exchange](https://github.com/manifoldbt/manifoldbt/blob/master/examples/15_cross_exchange.py) | Signal on one venue, execute on another |
| 16 | [Exogenous Data](https://github.com/manifoldbt/manifoldbt/blob/master/examples/16_hashrate_exogene.py) | External series (e.g. hashrate) as a signal |
| 17 | [Per-Venue Fees](https://github.com/manifoldbt/manifoldbt/blob/master/examples/17_per_venue_fees.py) | Per-venue funding & borrow costs |
| 18 | [CSV Import](https://github.com/manifoldbt/manifoldbt/blob/master/examples/18_csv_import.py) | Load OHLCV from CSV (standard / MT4 / MT5) |

## Performance

Every number below comes from a benchmark that runs in public CI on a standard
GitHub runner, and links back to the run that produced it. It installs each
engine from PyPI the way a user would, generates its own data, checks that the
engines produced the **same result**, and only then reports how long each took:
a workload they disagree on gets no published timing at all.

**Latest run: [#11](https://github.com/manifoldbt/manifoldbt/actions/runs/32396472073)**
ran on Linux x86_64, 4 vCPU, Python 3.12, manifoldbt 0.17.3 / vectorbt 0.28.4 /
raptorbt 0.9.0, 3 interleaved repetitions.

| Workload | Bars | ManifoldBT | vectorbt | raptorbt |
|---|---:|---:|---:|---:|
| SMA crossover | 10M | **317 ms** | 24.75 s (x78) | 878 ms (x2.8) |
| ...with drawdown, Sharpe, Sortino, volatility | 10M | **317 ms** | 97.46 s (**x308**) | 894 ms (x2.8) |
| ...with a 5 bps fee and 2 bps slippage | 10M | **316 ms** | 24.53 s (x78) | not supported |
| EMA + RSI filter, 5 bps fee | 1M | **52 ms** | 2.21 s (x41) | not supported |
| Five assets in one book | 1M | **140 ms** | 2.34 s (x17) | not supported |

The second row is the one worth reading twice. Asking for a performance summary
costs ManifoldBT nothing measurable, because it computes one during the run
whether you read it or not, and costs vectorbt 73 seconds, because it defers the
equity curve until a risk metric needs it and then has to build one.

The fifth row is the one where ManifoldBT does worst, and it is published for
that reason: broadcasting a column per asset is close to free for vectorbt,
while walking five books is not free for anything.

### Parameter sweeps

| Bars | Combinations | ManifoldBT | vectorbt | raptorbt |
|---:|---:|---:|---:|---:|
| 20,000 | 5,000 | **446 ms**, 40 MB | 5.84 s, 2.5 GB | 7.08 s |
| 200,000 | 10,000 | **9.96 s**, 79 MB | out of memory | 164.50 s |

Past a certain grid the question stops being speed. vectorbt materialises the
simulation per combination, 1.57 MB of it at 20,000 bars, so the second row
would ask a machine for tens of gigabytes. ManifoldBT runs it in ten seconds
inside 79 MB.

Reproduce any of it yourself: fork the repository and press **Run workflow** on
[the benchmark](https://github.com/manifoldbt/manifoldbt/actions/workflows/bench-vs-vectorbt.yml),
or run it locally from
[`benchmarks/vs_vectorbt/`](https://github.com/manifoldbt/manifoldbt/tree/master/benchmarks/vs_vectorbt).
The method, the parity gate and the known divergences are written up in
[its README](https://github.com/manifoldbt/manifoldbt/blob/master/benchmarks/vs_vectorbt/README.md).

### Against an event-driven engine

backtrader runs the same EMA(12/26) + RSI(14) strategy on 500K 1-minute bars in
**46,944 ms**, against **13 ms** for ManifoldBT: a factor of **3,556**. Measured
with `benchmarks/bench_vs_competitors.py`, median of 3 runs. It sits outside the
CI suite because its event-driven fills produce a different PnL, and the parity
gate publishes no timing for engines that did not do the same work.

### How it compares

| | ManifoldBT | vectorbt | backtrader | Nautilus |
|---|---|---|---|---|
| Engine | Rust (vectorized + sequential fills) | Numba/NumPy (vectorized) | Python (event-driven) | Rust/Python (event-driven) |
| Execution realism¹ | High | Basic | High | High |
| Focus | Backtesting + research | Backtesting at scale | Backtest + live | Backtest + live (production) |

¹ fees, slippage, funding, partial fills, look-ahead detection.

> On GPU (Pro), the Monte Carlo engine runs **~36x faster** than the all-core CPU path (SDE path simulation, RTX 3090, f32).

## Documentation

Full API reference, indicator list, configuration guide, and best practices:

**[www.manifoldbt.com/docs/documentation.html](https://www.manifoldbt.com/docs/documentation.html)**

## Community vs Pro

| | Community | Pro |
|---|---|---|
| Output resolution | Daily | 1m, 5m, 15m, 1h |
| Monte Carlo | 1K sims | Unlimited |
| Walk-Forward | - | Anchored + Rolling |
| Parameter Stability | - | Yes |
| Crypto connectors (Binance, Bybit, Hyperliquid) | Yes | Yes |
| Databento & Massive connectors | - | Yes |
| Safety checks (lookahead, exposure) | - | Yes |
| Tearsheets & export | - | Yes |

## License

Apache 2.0 with Commons Clause. The source is available, free to use,
modify and self-host. Reselling the software or offering it as a paid
hosted service is not permitted. See [LICENSE](https://github.com/manifoldbt/manifoldbt/blob/master/LICENSE) for the full text.
