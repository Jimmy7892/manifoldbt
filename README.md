# ManifoldBT

**Rust-powered backtesting engine for quantitative research.**

ManifoldBT is a high-performance backtesting framework with a Python DSL that compiles strategies into an optimized Rust expression graph. It is designed for speed, correctness, and ergonomics.

## Highlights

- **Rust core** — vectorized engine handles 1-minute resolution across years of data
- **Python DSL** — fluent strategy builder with indicators, signals, and sizing
- **Monte Carlo** — permutation-based simulation for robustness testing
- **Walk-Forward** — out-of-sample validation with rolling windows
- **Parameter Sweeps** — 2D heatmaps and 3D surface plots
- **Portfolio** — multi-strategy portfolio with risk rules and rebalancing

## Installation

```bash
pip install manifoldbt
```

With plotting support:

```bash
pip install manifoldbt[all]
```

## Quick Start

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Interval, Slippage

# Define indicators
fast = ema(close, 12)
slow = ema(close, 26)

# Build strategy
strategy = (
    mbt.Strategy.create("ema_crossover")
    .signal("fast", fast)
    .signal("slow", slow)
    .signal("signal", mbt.when(fast > slow, mbt.lit(1.0), mbt.lit(-1.0)))
    .size(mbt.col("signal") * mbt.lit(0.25))
)

# Configure backtest
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

# Run
store = mbt.DataStore(data_root="data", metadata_db="metadata/metadata.sqlite")
result = mbt.run(strategy, config, store)
print(result.summary())
```

## Examples

See the [examples/](examples/) directory for complete runnable strategies:

| # | Example | Description |
|---|---------|-------------|
| 00 | [Template](examples/00_template.py) | Minimal starting point |
| 01 | [Trend Following](examples/01_trend_following.py) | EMA crossover with stop-loss and volume filter |
| 02 | [Mean Reversion](examples/02_mean_reversion.py) | EMA crossover with parameter sweep |
| 03 | [Multi-Asset Momentum](examples/03_multi_asset_momentum.py) | Cross-asset momentum signals |
| 04 | [Linear Regression](examples/04_linear_regression.py) | Regression-based signal |
| 05 | [Statistical Arbitrage](examples/05_stat_arb.py) | Pairs trading with spread z-score |
| 06 | [Full Visualization](examples/06_full_visualization.py) | Complete tearsheet and charts |
| 07 | [Walk-Forward](examples/07_walk_forward.py) | Out-of-sample validation |
| 08 | [2D Sweep Heatmap](examples/08_sweep_2d_heatmap.py) | Parameter grid search |
| 09 | [3D Surface](examples/09_surface_3d.py) | 3D parameter surface plot |
| 10 | [Monte Carlo](examples/10_monte_carlo.py) | Permutation-based robustness |
| 11 | [Portfolio](examples/11_portfolio.py) | Multi-strategy portfolio |

## Documentation

- [Strategy Authoring Guide](docs/strategy-authoring.md) — full DSL reference

## Performance

ManifoldBT's Rust engine is orders of magnitude faster than pure-Python alternatives:

| Engine | 500K bars | 5M bars |
|--------|-----------|---------|
| **ManifoldBT** | ~0.02s | ~0.15s |
| vectorbt | ~0.8s | ~8s |
| backtrader | ~12s | ~120s+ |

Run `python benchmarks/bench_vs_competitors.py` to reproduce.

## License

MIT
