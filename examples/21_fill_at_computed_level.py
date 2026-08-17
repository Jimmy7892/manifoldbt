"""Filling at a computed level — ExecutionPrice.custom(<signal name>).

A mean-reversion band strategy on native 1-minute bars: short at the touch of
an upper band around an hourly SMA, cover at the lower band. The engine always
knew how to COMPUTE the band; this example shows the fill landing ON it.

The touch bar itself proves the level traded: it opens below the band and its
high crosses it, so the band price sits inside [open, high]. Yet with
``AtClose`` the only reachable fill is the bar's close — on a mean-reverting
touch, systematically on the wrong side of the level. The same run is done
both ways so the difference is visible in one place.

Self-contained: generates its own synthetic data in a temp store.

Usage:
    python examples/21_fill_at_computed_level.py
"""

import os
import tempfile

import numpy as np
import pandas as pd

import manifoldbt as mbt
from manifoldbt.indicators import close, high, low, open as open_px, sma
from manifoldbt.helpers import ExecutionPrice, Interval, Slippage

# -- Synthetic 1m data: a mean-reverting walk ---------------------------------
N = 30 * 1440  # 30 days of 1-minute bars
rng = np.random.default_rng(7)
steps = rng.normal(0.0, 0.0010, N)
level = np.cumsum(steps) * 0.85  # pull the walk back toward its mean
px = 100.0 * np.exp(level - np.linspace(0, level[-1], N))
o, c = px, np.roll(px, -1)
c[-1] = px[-1]
amp = np.abs(rng.normal(0.0, 0.0012, N))
ts = pd.date_range("2024-01-01", periods=N, freq="1min", tz="UTC")
frame = pd.DataFrame(
    {"timestamp": ts, "open": o,
     "high": np.maximum(o, c) * (1 + amp), "low": np.minimum(o, c) * (1 - amp),
     "close": c, "volume": rng.uniform(1_000, 5_000, N)}
)

# -- Bands around an hourly SMA, evaluated on 1m native bars ------------------
DEV_UP, DEV_DN = 0.004, 0.003

h1 = mbt.tf("1h")  # hourly columns, as of the last closed hour
band_up = sma(h1.close, 8) * (1 + DEV_UP)
band_dn = sma(h1.close, 8) * (1 - DEV_DN)

touch_up = high >= band_up   # entry: short at the touch of the upper band
touch_dn = low <= band_dn    # exit: cover at the touch of the lower band
target = mbt.when(touch_dn, 0.0, mbt.when(touch_up, -1.0))

# The level each fill should land on. The nesting mirrors the target's
# priority, and a bar that opens through a band fills at its open.
exec_level = mbt.when(
    touch_dn, mbt.when(open_px <= band_dn, open_px, band_dn),
    mbt.when(touch_up, mbt.when(open_px >= band_up, open_px, band_up), close),
)

strategy = (
    mbt.Strategy.create("band_touch_short")
    .signal("position", target)
    .signal("exec_level", exec_level)
    .size(target)
    .stop_loss(pct=25.0)
)

# -- Run the same strategy both ways ------------------------------------------
if __name__ == "__main__":
    root = tempfile.mkdtemp(prefix="mbt_example21_")
    store = mbt.import_dataframe(
        frame, symbol="SYNTH", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )

    def run(execution_price):
        config = mbt.BacktestConfig(
            universe=[1],
            time_range_start=0,
            time_range_end=int(ts[-1].value) + 86_400_000_000_000,
            bar_interval=Interval.minutes(1),
            initial_capital=10_000,
            execution=mbt.ExecutionConfig(
                signal_delay=0,
                execution_price=execution_price,
                max_position_pct=0.4,
                allow_short=True,
                position_sizing_mode="FractionOfEquity",
            ),
            fees=mbt.FeeConfig.binance_perps(),
            slippage=Slippage.fixed_bps(2),
            warmup_bars=60 * 10,
            extra_timeframes={"1h": Interval.hours(1)},
        )
        return mbt.run(strategy, config, store)

    print(f"{'execution price':<22} {'trades':>7} {'return':>9}   first entry fills")
    print("-" * 78)
    for label, price in (("AtClose", "AtClose"),
                         ("custom('exec_level')", ExecutionPrice.custom("exec_level"))):
        result = run(price)
        tr = result.trades_df()
        entries = tr[tr["fill_price"] > 0].head(3)["fill_price"].round(4).tolist()
        print(f"{label:<22} {len(tr):>7} {result.metrics['total_return']:>8.2%}   {entries}")

    print(
        "\nSame signals, same bars: only WHERE the order fills changed. The"
        "\ncustom fills land on the band level (inside the touch bar's range),"
        "\nnot on its close. A fill outside [low, high] would be warned about."
    )
