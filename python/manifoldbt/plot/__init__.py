"""Plotting module for manifoldbt (requires matplotlib).

Install with::

    pip install manifoldbt[plot]

Quick start::

    import manifoldbt as bt

    result = bt.run(strategy, config, store)
    bt.plot.tearsheet(result)              # full-page dashboard
    bt.plot.equity(result, show=True)      # single chart
"""
try:
    import matplotlib  # noqa: F401
except ImportError:
    raise ImportError(
        "matplotlib is required for the plotting module. "
        "Install it with: pip install manifoldbt[plot]"
    ) from None

# Backtest result charts
from manifoldbt.plot.backtest import (
    annual_returns,
    benchmark_equity,
    drawdown,
    equity,
    monthly_returns,
    returns_histogram,
    rolling_sharpe,
    rolling_volatility,
    summary,
    var_chart,
)

# Candlestick / indicator chart
from manifoldbt.plot.chart import chart

# Research charts
from manifoldbt.plot.research import (
    correlation_matrix,
    heatmap_2d,
    monte_carlo,
    stability,
    stochastic_paths,
    surface_3d,
    walk_forward,
)

# Composite layouts
from manifoldbt.plot.tearsheet import research_report, tearsheet

# Theme
from manifoldbt.plot._theme import THEME, apply_theme

__all__ = [
    # Backtest result plots
    "chart",
    "summary",
    "equity",
    "benchmark_equity",
    "drawdown",
    "monthly_returns",
    "annual_returns",
    "returns_histogram",
    "var_chart",
    "rolling_sharpe",
    "rolling_volatility",
    # Research plots
    "heatmap_2d",
    "surface_3d",
    "walk_forward",
    "stability",
    "correlation_matrix",
    "monte_carlo",
    "stochastic_paths",
    # Composites
    "tearsheet",
    "research_report",
    # Theme
    "THEME",
    "apply_theme",
]
