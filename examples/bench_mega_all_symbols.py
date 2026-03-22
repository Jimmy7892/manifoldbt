"""Benchmark: all symbols, Arrow IPC store (bars_1m + bars_1h).

Usage:
    python examples/bench_mega_all_symbols.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import ema, close
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Strategy -------------------------------------------------------------------
fast = ema(close, 12)
slow = ema(close, 200)
trend = fast - slow

strategy = (
    mbt.Strategy.create("ema_crossover_all")
    .signal("trend", trend)
    .size(mbt.when(trend > 0.0, 0.5, 0.0))
)

# -- Config: 21 crypto symbols, 3 years, 1h bars --------------------------------
# SOL (3) starts 2024 only — excluded
universe = [s for s in range(1, 23) if s != 3]
start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe=universe,
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.minutes(60),
    precise=True,
    initial_capital=100_000,
    execution=mbt.ExecutionConfig(
        allow_short=False,
        max_position_pct=0.05,
        position_sizing_mode="FractionOfInitialCapital",
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=30,
)

# -- Run -----------------------------------------------------------------------
root = os.path.join(os.path.dirname(__file__), "..")
data_root = os.path.abspath(os.path.join(root, "data"))
metadata_db = os.path.abspath(os.path.join(root, "metadata", "metadata.sqlite"))

store = mbt.DataStore(data_root=data_root, metadata_db=metadata_db, arrow_dir=os.path.join(data_root, "mega"))

t0 = time.perf_counter()
result = mbt.run(strategy, config, store)
elapsed = time.perf_counter() - t0

print(result.profile_summary())
print(f"\nWall clock: {elapsed:.3f}s")
print(f"Trades: {result.trade_count}")
print(f"Symbols: {len(universe)}")
