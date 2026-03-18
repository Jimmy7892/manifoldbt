<p align="center">
  <strong>ManifoldBT</strong><br>
  Rust-powered backtesting engine for quantitative research
</p>

<p align="center">
  <a href="https://manifold-bt.com">Website</a> &middot;
  <a href="https://manifold-bt.com/docs/documentation.html">Documentation</a> &middot;
  <a href="examples/">Examples</a>
</p>

---

ManifoldBT compiles Python strategy definitions into an optimized Rust expression graph.
Write strategies in a fluent Python DSL — execute them on a vectorized Rust engine.

## Why ManifoldBT

- **Fast** — 500K bars in ~26ms. 161x faster than vectorbt, 1000x+ faster than backtrader.
- **Expressive** — fluent DSL with 30+ indicators, conditional logic, cross-asset references
- **Rigorous** — Monte Carlo, walk-forward, parameter sweeps, lookahead detection, exposure diagnostics
- **Portable** — `pip install`, no Rust toolchain needed. Works on Python 3.9+.

## Installation

```bash
pip install manifoldbt
```

With all extras (plotting, pandas, polars):

```bash
pip install manifoldbt[all]
```

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

store = mbt.DataStore(data_root="data", metadata_db="metadata/metadata.sqlite")
result = mbt.run(strategy, config, store)
print(result.summary())
```

## Examples

| # | Example | What it shows |
|---|---------|---------------|
| 00 | [Template](examples/00_template.py) | Minimal starting point |
| 01 | [Trend Following](examples/01_trend_following.py) | EMA crossover, volume filter, stop-loss |
| 02 | [Mean Reversion](examples/02_mean_reversion.py) | EMA crossover with parameter sweep |
| 03 | [Multi-Asset Momentum](examples/03_multi_asset_momentum.py) | Cross-asset signals |
| 04 | [Linear Regression](examples/04_linear_regression.py) | Regression-based signal |
| 05 | [Statistical Arbitrage](examples/05_stat_arb.py) | Pairs trading, spread z-score |
| 06 | [Full Visualization](examples/06_full_visualization.py) | Tearsheet and charts |
| 07 | [Walk-Forward](examples/07_walk_forward.py) | Out-of-sample validation |
| 08 | [2D Sweep](examples/08_sweep_2d_heatmap.py) | Parameter grid heatmap |
| 09 | [3D Surface](examples/09_surface_3d.py) | Parameter surface plot |
| 10 | [Monte Carlo](examples/10_monte_carlo.py) | Permutation-based robustness |
| 11 | [Portfolio](examples/11_portfolio.py) | Multi-strategy portfolio |

## Performance

EMA(12/26) + RSI(14) on 500K synthetic 1-min bars (median of 5 runs):

| Engine | Time | vs ManifoldBT |
|--------|------|---------------|
| **ManifoldBT** (Rust) | **26 ms** | 1x |
| vectorbt (NumPy) | 4,094 ms | 161x slower |
| backtrader (Python) | — | ~1000x slower |

Reproduce: `python benchmarks/bench_vs_competitors.py --rows 500000 --runs 5`

## Documentation

Full API reference, indicator list, configuration guide, and best practices:

**[manifold-bt.com/docs/documentation.html](https://manifold-bt.com/docs/documentation.html)**

## Community vs Pro

| | Community | Pro |
|---|---|---|
| Output resolution | Daily | 1m, 5m, 15m, 1h |
| Monte Carlo | 1K sims | Unlimited |
| Walk-Forward | - | Anchored + Rolling |
| Parameter Stability | - | Yes |
| Crypto connectors (Binance, Hyperliquid) | Yes | Yes |
| Databento & Massive connectors | - | Yes |
| Safety checks (lookahead, exposure) | - | Yes |
| Tearsheets & export | - | Yes |

## License

MIT
