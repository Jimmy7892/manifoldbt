"""The lite sweep path must agree with `run()` on intraday bars.

`run_sweep_lite` is a separate transcription of the simulation, kept for speed
(roughly ten times the throughput of the full sweep). Its metrics are computed
from a *daily* equity curve, and that curve's first point is the equity at the
CLOSE of day one. Taking it as the growth base silently drops day one's profit
and loss from every metric measured against it, which shipped as an 8% error on
`total_return` for a fourteen-day intraday backtest.

The bug was invisible on daily bars: with a 60-period indicator the warmup
covers sixty days, so the close of day one still equals the initial capital and
the base is right by accident. It only appears when trading starts on day one,
which on 1-minute bars is the normal case. Hence this test runs intraday.

All fourteen metrics must be identical. `ulcer_index` used to be the exception:
it is accumulated over whichever curve it is handed, so the lite and GPU sweeps
measured it on daily points while `run()` measured it bar by bar, and the same
backtest carried two different values depending on the entry point. It now
follows the daily series on every path, like the Sharpe, Sortino and volatility
beside it, and like the published definition of the Ulcer Index. `max_drawdown`
deliberately stays full-resolution: a drawdown that opens and recovers inside a
day is a real one and belongs in the maximum.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, param, when  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.indicators import close as close_px, sma  # noqa: E402

CAPITAL = 100_000.0
FAST, SLOW = 10, 60

# Metrics that are pure functions of the equity path and its base, so the two
# code paths must agree to float-reordering noise.
MUST_MATCH = (
    "total_return",
    "cagr",
    "calmar",
    "tstat_sharpe",
    "sharpe",
    "sortino",
    "volatility",
    "max_drawdown",
    "avg_daily_return",
    "best_day",
    "worst_day",
    "pct_positive_days",
    "ulcer_index",
    "alpha",
    "beta",
)


def _intraday_bars(rows=8_000, seed=7):
    """Gap-free 1-minute random walk. Long enough to span several days, and
    volatile enough that the crossover trades inside the first day."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 3e-4, rows)))
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = rng.uniform(0.2, 1.8, rows) * 3e-4 * close
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=rows, freq="1min", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": np.full(rows, 1_000.0),
        }
    )


def _config(df):
    last_ns = int(df["timestamp"].iloc[-1].value)
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 86_400_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=0,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def test_lite_sweep_matches_run_on_intraday_bars(tmp_path):
    df = _intraday_bars()
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df,
        symbol="TEST",
        symbol_id=1,
        interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    config = _config(df)

    sized = when(col("fast") > col("slow"), lit(1.0), lit(0.0))
    fixed = (
        bt.Strategy.create("fixed")
        .signal("fast", sma(close_px, FAST))
        .signal("slow", sma(close_px, SLOW))
        .size(sized)
    )
    swept = (
        bt.Strategy.create("swept")
        .signal("fast", sma(close_px, param("fast")))
        .signal("slow", sma(close_px, param("slow")))
        .size(sized)
    )

    full = bt.run(fixed, config, store).metrics
    lite = bt.run_sweep_lite(
        swept, {"fast": [FAST], "slow": [SLOW]}, config, store
    )[0].metrics

    # The strategy must actually trade on day one, otherwise the base is right
    # by accident and the test proves nothing.
    assert full["total_return"] != 0.0

    for name in MUST_MATCH:
        expected, got = full[name], lite[name]
        assert abs(expected - got) <= 1e-9 * max(1.0, abs(expected)), (
            f"{name}: run()={expected!r} but run_sweep_lite()={got!r}. "
            "The lite path has drifted from the full simulation."
        )
