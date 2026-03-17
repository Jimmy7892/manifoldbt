"""Full Visualization Suite -- Bollinger Bands mean-reversion + all plots.

Strategy:
  - Long when price touches lower band (oversold)
  - Short when price touches upper band (overbought)
  - Size proportional to distance from middle band
  - Stop-loss 2%, take-profit 4%

Demonstrates every plotting function available in manifoldbt.

Usage:
    python examples/06_full_visualization.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, bollinger_bands, ema
from manifoldbt.helpers import time_range, Slippage, Interval

upper, middle, lower = bollinger_bands(close, period=20, num_std=2.0)
trend_ema = ema(close, 100)

# Z-score: how far price is from the mean, normalized by band width
band_width = upper - lower
zscore = (close - middle) / (band_width + mbt.lit(1e-12))

# Trend filter: EMA(100) above close = downtrend (no longs), below = uptrend (no shorts)
is_uptrend = close > trend_ema
is_downtrend = close < trend_ema

# -- Strategy -----------------------------------------------------------------
# Entry: touch lower band → long (only in uptrend), touch upper band → short (only in downtrend)
# Exit:  long exits at upper band, short exits at lower band
# Size flips to 0 at opposite band = exit

# Long signal: price near lower band + uptrend
long_entry = (zscore < -0.5) & is_uptrend
# Short signal: price near upper band + downtrend
short_entry = (zscore > 0.5) & is_downtrend

# Long exits at upper band (zscore > 0.5), short exits at lower band (zscore < -0.5)
# When neither entry nor in opposite-band exit zone → flat (0)
signal = mbt.when(
    long_entry, 1.0,                           # long
    mbt.when(short_entry, -1.0, 0.0),         # short / flat
)

strategy = (
    mbt.Strategy.create("Reversion_strategy")
    .signal("upper", upper)
    .signal("lower", lower)
    .signal("ema100", trend_ema)
    .signal("zscore", zscore)
    .size(signal * 0.25)
    .describe(
        "Bollinger Bands mean-reversion: long at lower band, short at upper band, "
        "exit at opposite band. EMA(100) trend filter — no shorts in uptrend, "
        "no longs in downtrend."
    )
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2026-01-01")

ALL_SYMBOLS = list(range(1, 23))  # 22 symbols: BTCUSDT to ARBUSDT

config = mbt.BacktestConfig(
    universe=ALL_SYMBOLS,
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.minutes(120),
    initial_capital=100_000,
    execution=mbt.ExecutionConfig(
        allow_short=True,
        max_position_pct=0.5,
        position_sizing_mode="FractionOfInitialCapital",
    ),
    fees=mbt.FeeConfig.zero(),
    slippage=Slippage.fixed_bps(0),
    warmup_bars=25,
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    os.makedirs(os.path.join(root, "output"), exist_ok=True)
    store = mbt.DataStore(
        data_root=os.path.abspath(os.path.join(root, "data")),
        metadata_db=os.path.abspath(os.path.join(root, "metadata", "metadata.sqlite")),
    )

    # -- 1. Single backtest --------------------------------------------------
    print("Running backtest...")
    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0
    print(result.summary())
    print(f"Elapsed: {elapsed:.3f}s\n")

    # -- 2. Tearsheet (3 figures: overview, returns, rolling) ---------------
    print("Generating tearsheet...")
    mbt.plot.tearsheet(
        result, show=True,
        save=os.path.join(root, "output", "tearsheet.png"),
    )

    # -- 3. Summary 3-panel ---------------------------------------------------
    mbt.plot.summary(result, show=True)

    # -- 4. Candlestick chart (symbol_id=1 matches universe) ----------------
    mbt.plot.chart(
        result, store, symbol_id=1,
        emas=[10, 25],
        smas=[50],
        n_bars=120,
        interactive=False,
        show=True,
    )

    # -- 5. Individual charts -------------------------------------------------
    mbt.plot.equity(result, show=True)
    mbt.plot.drawdown(result, show=True)
    mbt.plot.monthly_returns(result, show=True)
    mbt.plot.annual_returns(result, show=True)
    mbt.plot.returns_histogram(result, show=True)
    mbt.plot.var_chart(result, show=True)
    mbt.plot.rolling_sharpe(result, show=True)
    mbt.plot.rolling_volatility(result, show=True)

    # -- 6. Sweep heatmap 2D -------------------------------------------------
    # Sweep over BB period and num_std by rebuilding strategies
    print("\nRunning 2D sweep (BB period × num_std)...")
    t0 = time.perf_counter()

    periods = [10, 15, 20, 30]
    stds = [1.5, 2.0, 2.5, 3.0]
    sweep_strategies = []
    for p in periods:
        for ns in stds:
            u, m, l = bollinger_bands(close, period=p, num_std=ns)
            bw = u - l
            zs = (close - m) / (bw + mbt.lit(1e-12))
            up = close > trend_ema
            dn = close < trend_ema
            sig = mbt.when(
                (zs < -0.5) & up, 1.0,
                mbt.when((zs > 0.5) & dn, -1.0, 0.0),
            )
            s = (
                mbt.Strategy.create(f"bb_p{p}_s{ns}")
                .signal("zscore", zs)
                .size(sig * 0.25)
                .stop_loss(pct=2.0)
                .take_profit(pct=4.0)
            )
            sweep_strategies.append(s)

    batch_results = mbt.run_batch_lite(sweep_strategies, config, store)
    # Build a sweep_result dict compatible with heatmap_2d
    metric_grid = []
    idx = 0
    for _ in periods:
        row = []
        for _ in stds:
            r = batch_results[idx]
            row.append(r.metrics.get("sharpe", 0.0))
            idx += 1
        metric_grid.append(row)

    sweep_result = {
        "x_param": "num_std",
        "y_param": "period",
        "x_values": stds,
        "y_values": periods,
        "metric": "sharpe",
        "metric_grid": metric_grid,
    }
    print(f"Sweep done in {time.perf_counter() - t0:.1f}s")
    mbt.plot.heatmap_2d(sweep_result, show=True)

    # -- 7. Walk-forward validation -------------------------------------------
    # Manual walk-forward: split 2024 into 5 folds
    print("\nRunning walk-forward (manual folds)...")
    t0 = time.perf_counter()

    fold_months = [
        ("2024-01-01", "2024-07-01", "2024-07-01", "2024-09-01"),
        ("2024-01-01", "2024-08-01", "2024-08-01", "2024-10-01"),
        ("2024-01-01", "2024-09-01", "2024-09-01", "2024-11-01"),
        ("2024-01-01", "2024-10-01", "2024-10-01", "2024-12-01"),
        ("2024-01-01", "2024-11-01", "2024-11-01", "2025-01-01"),
    ]
    wf_folds = []
    for train_start, train_end, test_start, test_end in fold_months:
        ts, te = time_range(train_start, train_end)
        train_cfg = mbt.BacktestConfig(
            universe=ALL_SYMBOLS, time_range_start=ts, time_range_end=te,
            bar_interval=Interval.minutes(60), initial_capital=100_000,
            execution=config.execution, fees=config.fees,
            slippage=config.slippage, warmup_bars=25,
        )
        ts2, te2 = time_range(test_start, test_end)
        test_cfg = mbt.BacktestConfig(
            universe=ALL_SYMBOLS, time_range_start=ts2, time_range_end=te2,
            bar_interval=Interval.minutes(60), initial_capital=100_000,
            execution=config.execution, fees=config.fees,
            slippage=config.slippage, warmup_bars=25,
        )
        train_r = mbt.run(strategy, train_cfg, store)
        test_r = mbt.run(strategy, test_cfg, store)
        train_m = train_r.metrics
        test_m = test_r.metrics
        wf_folds.append({
            "train_metric": train_m.get("sharpe", 0.0),
            "test_metric": test_m.get("sharpe", 0.0),
        })

    wf_result = {
        "metric": "sharpe",
        "folds": wf_folds,
    }
    print(f"Walk-forward done in {time.perf_counter() - t0:.1f}s")
    mbt.plot.walk_forward(wf_result, show=True)

    # -- 8. Monte Carlo -------------------------------------------------------
    print("\nRunning Monte Carlo (1000 paths)...")
    mc_result = mbt.py_run_monte_carlo(result.raw, 1000, 42)
    mbt.plot.monte_carlo(mc_result, show=True)

    # -- 9. Parameter stability -----------------------------------------------
    print("\nRunning stability analysis (BB period)...")
    t0 = time.perf_counter()
    stability_periods = [10, 12, 15, 18, 20, 25, 30, 40]
    stability_metrics = []
    for p in stability_periods:
        u, m, l = bollinger_bands(close, period=p, num_std=2.0)
        bw = u - l
        zs = (close - m) / (bw + mbt.lit(1e-12))
        up = close > trend_ema
        dn = close < trend_ema
        sig = mbt.when(
            (zs < -0.5) & up, 1.0,
            mbt.when((zs > 0.5) & dn, -1.0, 0.0),
        )
        s = (
            mbt.Strategy.create(f"bb_stab_{p}")
            .signal("zscore", zs)
            .size(sig * 0.25)
            .stop_loss(pct=2.0)
            .take_profit(pct=4.0)
        )
        r = mbt.run(s, config, store)
        stability_metrics.append(r.metrics.get("sharpe", 0.0))

    import numpy as np
    mean_m = float(np.mean(stability_metrics))
    std_m = float(np.std(stability_metrics))
    stab_result = {
        "param_name": "period",
        "metric": "sharpe",
        "values": stability_periods,
        "metric_values": stability_metrics,
        "mean_metric": mean_m,
        "std_metric": std_m,
        "stability_score": 1.0 - (std_m / abs(mean_m)) if mean_m != 0 else 0.0,
    }
    print(f"Stability done in {time.perf_counter() - t0:.1f}s")
    mbt.plot.stability(stab_result, show=True)

    # -- 10. Research report (composite) --------------------------------------
    print("\nGenerating research report...")
    mbt.plot.research_report(
        sweep_result=sweep_result,
        wf_result=wf_result,
        stability_result=stab_result,
        show=True,
        save=os.path.join(root, "output", "research.png"),
    )

    print("\nDone — all visualizations generated.")
    print(f"PNGs saved to {os.path.join(root, 'output')}")
