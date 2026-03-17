"""Charts for BacktestResult visualization."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from manifoldbt.plot._theme import (
    ACCENT,
    ACCENT_ALT,
    DARK_GRAY,
    GRAY,
    GREEN,
    ORANGE,
    RED,
    WHITE,
    theme_context,
)
from manifoldbt.plot._convert import (
    daily_returns_array,
    equity_with_dates,
    positions_arrays,
    trades_arrays,
    _ts_to_int64,
)
from manifoldbt.plot._utils import finalize, format_pct, get_or_create_ax


# ── Summary (the essential chart) ────────────────────────────────────────────


def summary(
    result,
    *,
    figsize: Tuple[float, float] = (14, 8),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """The essential chart: TWR equity + buy-and-hold benchmark, trade activity.

    Top panel:  TWR-normalized equity curve vs buy-and-hold (close price).
    Bottom panel: daily trade count as a bar chart.
    Metrics displayed in a clean header line.
    """
    with theme_context():
        fig, (ax_eq, ax_trades, ax_margin) = plt.subplots(
            3, 1, figsize=figsize, height_ratios=[3, 1, 1],
            sharex=True, gridspec_kw={"hspace": 0.25},
        )

        dates, eq_vals = equity_with_dates(result)
        metrics = result.metrics if hasattr(result, "metrics") else {}

        # ── TWR equity (normalized to 100) ────────────────────────
        twr = eq_vals / eq_vals[0] * 100
        ax_eq.plot(dates, twr, color=ACCENT, linewidth=0.8, label="Strategy")
        ax_eq.fill_between(dates, twr, 100, where=(twr >= 100),
                           color=GREEN, alpha=0.04, interpolate=True)
        ax_eq.fill_between(dates, twr, 100, where=(twr < 100),
                           color=RED, alpha=0.04, interpolate=True)

        # ── Benchmark: buy-and-hold from close prices ─────────────
        positions = result.positions
        close_col = positions.column("close")
        close_raw = close_col.to_numpy(zero_copy_only=False) if hasattr(close_col, "to_numpy") else np.array(close_col.to_pylist())
        ts_ns = _ts_to_int64(positions.column("timestamp"))
        _, unique_idx = np.unique(ts_ns, return_index=True)
        unique_idx.sort()
        close_vals = close_raw[unique_idx].astype(np.float64)

        if len(close_vals) > 0 and close_vals[0] > 0:
            benchmark_raw = close_vals / close_vals[0] * 100

            # Vol-adjusted benchmark: scale to same volatility as strategy
            strat_rets = np.diff(twr) / twr[:-1]
            bench_rets = np.diff(benchmark_raw) / benchmark_raw[:-1]
            strat_vol = np.nanstd(strat_rets)
            bench_vol = np.nanstd(bench_rets)
            if bench_vol > 1e-12:
                adj_rets = bench_rets * (strat_vol / bench_vol)
                benchmark = np.empty_like(benchmark_raw)
                benchmark[0] = 100.0
                benchmark[1:] = 100.0 * np.cumprod(1.0 + adj_rets)
            else:
                benchmark = benchmark_raw

            ax_eq.plot(dates, benchmark, color=GRAY, linewidth=1.0,
                       label="Buy & Hold (vol-adj)", alpha=0.7)

        ax_eq.axhline(100, color=DARK_GRAY, linewidth=0.4)
        # Ensure y-axis zooms to strategy range with some padding
        twr_min, twr_max = float(np.nanmin(twr)), float(np.nanmax(twr))
        twr_range = max(twr_max - twr_min, 0.1)
        ax_eq.set_ylim(twr_min - twr_range * 0.15, twr_max + twr_range * 0.15)
        ax_eq.set_ylabel("TWR (base 100)", fontsize=9)
        ax_eq.legend(loc="upper left", framealpha=0.3, fontsize=8)

        # Header metrics
        ret = metrics.get("total_return", 0)
        sharpe = metrics.get("sharpe", 0)
        mdd = metrics.get("max_drawdown", 0)
        n_trades = metrics.get("total_trades", result.trade_count)
        title = (
            f"Return {ret * 100:+.1f}%"
            f"    Sharpe {sharpe:.2f}"
            f"    Max DD {mdd * 100:.1f}%"
            f"    Trades {n_trades:,}"
        )
        ax_eq.set_title(title, fontsize=10, loc="left", pad=10)

        # ── Adaptive smoothing window ──────────────────────────────
        # Scale window: min(7d, max(1d, 5% of total period))
        smooth_label = ""
        if len(dates) >= 2:
            bar_ns = int(dates[1]) - int(dates[0])
            total_ns = int(dates[-1]) - int(dates[0])
            day_ns = 24 * 3_600_000_000_000
            target_ns = min(7 * day_ns, max(day_ns, int(total_ns * 0.05)))
            smooth_window = max(1, target_ns // max(bar_ns, 1))
            smooth_window = min(smooth_window, len(dates))
            smooth_days = round(target_ns / day_ns)
            smooth_label = f" ({smooth_days}d)" if smooth_days >= 1 else ""
        else:
            smooth_window = 1

        # ── Trade activity (daily trade count) ─────────────────────
        try:
            ta = trades_arrays(result)
            trade_ts = ta.get("execution_timestamp", np.array([], dtype="datetime64[ns]"))
            if len(trade_ts) > 0 and len(dates) >= 2:
                # Bucket trades into calendar days
                trade_days = trade_ts.astype("datetime64[D]")
                unique_days, day_counts = np.unique(trade_days, return_counts=True)
                day_dates = unique_days.astype("datetime64[ns]")

                ax_trades.bar(day_dates, day_counts,
                              width=np.timedelta64(1, "D"),
                              color=ACCENT_ALT, alpha=0.4, edgecolor="none")

                # Rolling 7-day average overlay
                eq_days = dates.astype("datetime64[D]")
                unique_eq_days = np.unique(eq_days)
                daily_on_grid = np.zeros(len(unique_eq_days), dtype=np.float64)
                day_map = {d: c for d, c in zip(unique_days, day_counts)}
                for i, d in enumerate(unique_eq_days):
                    daily_on_grid[i] = day_map.get(d, 0)
                win = min(7, len(daily_on_grid))
                if win > 1:
                    kernel = np.ones(win) / win
                    smoothed = np.convolve(daily_on_grid, kernel, mode="same")
                    ax_trades.plot(unique_eq_days.astype("datetime64[ns]"), smoothed,
                                   color=ACCENT_ALT, linewidth=1.0, alpha=0.8)
            else:
                ax_trades.text(0.5, 0.5, "No trade data", transform=ax_trades.transAxes,
                               ha="center", va="center", color=DARK_GRAY, fontsize=9)
        except Exception:
            ax_trades.text(0.5, 0.5, "No trade data", transform=ax_trades.transAxes,
                           ha="center", va="center", color=DARK_GRAY, fontsize=9)

        ax_trades.set_ylabel("Trades/day", fontsize=8)

        # ── Used margin % (daily) ──────────────────────────────
        try:
            pa = positions_arrays(result)
            pos_ts = pa["timestamp"]
            pos_cap = pa["capital"]
            pos_eq = pa["equity"]

            unique_ts, first_idx = np.unique(pos_ts, return_index=True)
            first_idx.sort()
            cap = pos_cap[first_idx]
            eq_arr = pos_eq[first_idx]
            used = np.where(eq_arr > 0, (1.0 - cap / eq_arr) * 100, 0.0)
            used = np.clip(used, 0, None)
            used_dates = unique_ts.astype("datetime64[ns]")

            # Resample to daily (end-of-day snapshot)
            days = used_dates.astype("datetime64[D]")
            unique_days, _ = np.unique(days, return_index=True)
            # Use last value per day (not first) for end-of-day margin
            day_last = np.searchsorted(days, unique_days, side="right") - 1
            daily_used = used[day_last]
            daily_dates = unique_days.astype("datetime64[ns]")

            ax_margin.fill_between(daily_dates, 0, daily_used,
                                   color=GREEN, alpha=0.10, edgecolor="none")
            ax_margin.plot(daily_dates, daily_used,
                           color=GREEN, linewidth=0.7, alpha=0.8)
            ax_margin.axhline(0, color=DARK_GRAY, linewidth=0.4)
        except Exception:
            ax_margin.text(
                0.5, 0.5, "No position data",
                transform=ax_margin.transAxes,
                ha="center", va="center", color=DARK_GRAY, fontsize=9,
            )

        ax_margin.set_ylabel(f"Margin %{smooth_label}", fontsize=8)
        ax_margin.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_margin.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.align_ylabels([ax_eq, ax_trades, ax_margin])
        fig.autofmt_xdate(rotation=0, ha="center")

        return finalize(fig, show=show, save=save)


# ── Equity Curve ─────────────────────────────────────────────────────────────


def equity(
    result,
    *,
    ax: Optional[Axes] = None,
    color: str = ACCENT,
    title: str = "Equity Curve",
    figsize: Tuple[float, float] = (14, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot the portfolio equity curve over time."""
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)
        dates, values = equity_with_dates(result)
        ax_.plot(dates, values, color=color, linewidth=1.3)
        ax_.fill_between(dates, values, values.min(), color=color, alpha=0.05)
        ax_.set_title(title)
        ax_.set_ylabel("Equity", fontsize=9)
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=0, ha="center")
        return finalize(fig, show=show, save=save)


# ── Benchmark Overlay ────────────────────────────────────────────────────────


def benchmark_equity(
    result,
    benchmark: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    strategy_color: str = ACCENT,
    benchmark_color: str = DARK_GRAY,
    normalize: bool = True,
    labels: Tuple[str, str] = ("Strategy", "Buy & Hold"),
    title: str = "Strategy vs Benchmark",
    figsize: Tuple[float, float] = (14, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Overlay strategy equity and a benchmark, both normalized to 100."""
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)
        dates, strat_eq = equity_with_dates(result)
        bench = np.asarray(benchmark, dtype=np.float64)
        n = min(len(strat_eq), len(bench))
        strat_eq, bench, dates = strat_eq[:n], bench[:n], dates[:n]

        if normalize and strat_eq[0] != 0 and bench[0] != 0:
            strat_eq = strat_eq / strat_eq[0] * 100
            bench = bench / bench[0] * 100

        ax_.plot(dates, strat_eq, color=strategy_color, linewidth=1.3, label=labels[0])
        ax_.plot(dates, bench, color=benchmark_color, linewidth=1.0, label=labels[1])
        ax_.set_title(title)
        ax_.set_ylabel("Normalized" if normalize else "Equity")
        ax_.legend(loc="upper left", framealpha=0.5)
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=0, ha="center")
        return finalize(fig, show=show, save=save)


# ── Drawdown / Underwater ────────────────────────────────────────────────────


def drawdown(
    result,
    *,
    ax: Optional[Axes] = None,
    color: str = RED,
    title: str = "Drawdown",
    figsize: Tuple[float, float] = (14, 3),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot the drawdown as a filled area chart."""
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)
        dates, values = equity_with_dates(result)
        running_max = np.maximum.accumulate(values)
        dd = (values - running_max) / running_max

        ax_.fill_between(dates, dd, 0, color=color, alpha=0.25)
        ax_.plot(dates, dd, color=color, linewidth=0.8)
        ax_.set_title(title)
        ax_.set_ylabel("Drawdown")
        ax_.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax_.set_ylim(top=0)
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=0, ha="center")
        return finalize(fig, show=show, save=save)


# ── Monthly Returns Heatmap ──────────────────────────────────────────────────


def monthly_returns(
    result,
    *,
    ax: Optional[Axes] = None,
    annotate: bool = True,
    title: str = "Monthly Returns (%)",
    figsize: Tuple[float, float] = (12, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Monthly returns heatmap (year rows x month columns + annual)."""
    with theme_context():
        dates, values = equity_with_dates(result)
        ts = dates.astype("datetime64[M]")
        months = np.unique(ts)
        month_returns = {}
        for m in months:
            idx = np.nonzero(ts == m)[0]
            if len(idx) >= 2:
                month_returns[m] = values[idx[-1]] / values[idx[0]] - 1.0

        years = sorted({int(m.astype("datetime64[Y]").astype(int)) + 1970 for m in months})
        grid = np.full((len(years), 13), np.nan)

        for m, ret in month_returns.items():
            y = int(m.astype("datetime64[Y]").astype(int)) + 1970
            mo = int(m.astype("datetime64[M]").astype(int)) % 12
            grid[years.index(y), mo] = ret

        for yi in range(len(years)):
            row = grid[yi, :12]
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                grid[yi, 12] = np.prod(1.0 + valid) - 1.0

        fig, ax_ = get_or_create_ax(ax, figsize)
        abs_max = max(np.nanmax(np.abs(grid)), 0.01)
        cmap = plt.get_cmap("bt_diverging")
        im = ax_.imshow(grid, cmap=cmap, aspect="auto", vmin=-abs_max, vmax=abs_max)

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "YTD"]
        ax_.set_xticks(range(13))
        ax_.set_xticklabels(month_labels, fontsize=8)
        ax_.set_yticks(range(len(years)))
        ax_.set_yticklabels([str(y) for y in years], fontsize=9)

        if annotate:
            for yi in range(len(years)):
                for mi in range(13):
                    val = grid[yi, mi]
                    if np.isnan(val):
                        continue
                    txt = f"{val * 100:+.1f}"
                    brightness = abs(val) / abs_max
                    txt_color = WHITE if brightness > 0.4 else GRAY
                    ax_.text(mi, yi, txt, ha="center", va="center",
                             fontsize=7, color=txt_color, fontweight="medium")

        ax_.set_title(title)
        return finalize(fig, show=show, save=save)


# ── Annual Returns ───────────────────────────────────────────────────────────


def annual_returns(
    result,
    *,
    ax: Optional[Axes] = None,
    title: str = "Annual Returns",
    figsize: Tuple[float, float] = (10, 4),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Annual returns bar chart with green/red conditional coloring."""
    with theme_context():
        dates, values = equity_with_dates(result)
        years_arr = dates.astype("datetime64[Y]").astype(int) + 1970
        unique_years = sorted(set(years_arr))
        ann_rets = []
        for y in unique_years:
            idx = np.nonzero(years_arr == y)[0]
            ann_rets.append(values[idx[-1]] / values[idx[0]] - 1.0 if len(idx) >= 2 else 0.0)

        fig, ax_ = get_or_create_ax(ax, figsize)
        colors = [GREEN if r >= 0 else RED for r in ann_rets]
        bars = ax_.bar([str(y) for y in unique_years], ann_rets, color=colors,
                       width=0.5, alpha=0.85, edgecolor="none")
        ax_.axhline(0, color=DARK_GRAY, linewidth=0.5)
        ax_.set_title(title)
        ax_.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        for bar, ret in zip(bars, ann_rets):
            ax_.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     format_pct(ret), ha="center",
                     va="bottom" if ret >= 0 else "top",
                     fontsize=8, color=GRAY)
        return finalize(fig, show=show, save=save)


# ── Returns Histogram ────────────────────────────────────────────────────────


def returns_histogram(
    result,
    *,
    ax: Optional[Axes] = None,
    bins: int = 100,
    title: str = "Returns Distribution",
    figsize: Tuple[float, float] = (12, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Histogram of daily returns with green/red coloring by sign."""
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)
        rets = daily_returns_array(result)
        if len(rets) == 0:
            ax_.set_title(title + " (no data)")
            return finalize(fig, show=show, save=save)

        # Clip x-axis to P1-P99 range to avoid empty space from outliers
        p1, p99 = np.percentile(rets, [1, 99])
        margin = (p99 - p1) * 0.3
        xlim = (p1 - margin, p99 + margin)

        _, bin_edges, patches = ax_.hist(rets, bins=bins, edgecolor="none", alpha=0.7,
                                          range=xlim)
        for patch, left in zip(patches, bin_edges[:-1]):
            patch.set_facecolor(GREEN if left >= 0 else RED)

        ax_.axvline(0, color=DARK_GRAY, linewidth=0.8, linestyle="--")
        ax_.set_xlim(xlim)

        # Normal fit (pure numpy)
        mu, sigma = rets.mean(), rets.std()
        if sigma > 0:
            x = np.linspace(xlim[0], xlim[1], 200)
            bw = bin_edges[1] - bin_edges[0]
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax_.plot(x, pdf * len(rets) * bw, color=ACCENT, linewidth=1.0,
                     alpha=0.7, label="Normal")
            ax_.legend(loc="upper right", framealpha=0.3)

        ax_.set_title(title)
        ax_.set_xlabel("Daily Return")
        ax_.set_ylabel("Frequency")
        ax_.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        return finalize(fig, show=show, save=save)


# ── Value at Risk ────────────────────────────────────────────────────────────


def var_chart(
    result,
    *,
    ax: Optional[Axes] = None,
    confidence: float = 0.05,
    bins: int = 120,
    title: str = "Value at Risk",
    figsize: Tuple[float, float] = (12, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Returns histogram with VaR and CVaR lines at 5% and 1% levels."""
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)
        rets = daily_returns_array(result)
        if len(rets) == 0:
            ax_.set_title(title + " (no data)")
            return finalize(fig, show=show, save=save)

        rets_pct = rets * 100

        # Histogram
        n, bin_edges, patches = ax_.hist(
            rets_pct, bins=bins, color=ACCENT, alpha=0.5, edgecolor="none",
        )

        # VaR/CVaR at 5%
        var_5 = float(np.percentile(rets, 5))
        cvar_5 = float(rets[rets <= var_5].mean()) if np.any(rets <= var_5) else var_5

        # VaR/CVaR at 1%
        var_1 = float(np.percentile(rets, 1))
        cvar_1 = float(rets[rets <= var_1].mean()) if np.any(rets <= var_1) else var_1

        # Color tail bins
        for b, p in zip(bin_edges, patches):
            if b < var_1 * 100:
                p.set_facecolor(RED)
                p.set_alpha(0.5)
            elif b < var_5 * 100:
                p.set_facecolor(ORANGE)
                p.set_alpha(0.4)

        # VaR lines
        ax_.axvline(var_5 * 100, color=ORANGE, linewidth=0.8,
                    label=f"VaR 5%: {format_pct(var_5)}")
        ax_.axvline(cvar_5 * 100, color=ORANGE, linewidth=0.6, linestyle="--", alpha=0.5,
                    label=f"CVaR 5%: {format_pct(cvar_5)}")
        ax_.axvline(var_1 * 100, color=RED, linewidth=0.8,
                    label=f"VaR 1%: {format_pct(var_1)}")
        ax_.axvline(cvar_1 * 100, color=RED, linewidth=0.6, linestyle="--", alpha=0.5,
                    label=f"CVaR 1%: {format_pct(cvar_1)}")

        ax_.set_title(title)
        ax_.set_xlabel("Daily Return (%)")
        ax_.set_ylabel("Frequency")
        ax_.legend(loc="upper right", fontsize=8, framealpha=0.3)
        return finalize(fig, show=show, save=save)


# ── Rolling Sharpe ───────────────────────────────────────────────────────────


def rolling_sharpe(
    result,
    *,
    windows: Optional[List[int]] = None,
    ax: Optional[Axes] = None,
    title: str = "Rolling Sharpe",
    trading_days_per_year: float = 365.25,
    figsize: Tuple[float, float] = (14, 4),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Rolling annualized Sharpe ratio."""
    if windows is None:
        windows = [126, 252]
    colors = [ACCENT, ACCENT_ALT, GREEN, RED]

    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)
        rets = daily_returns_array(result)

        for i, w in enumerate(windows):
            if len(rets) < w:
                continue
            rm = _rolling(rets, w, np.mean)
            rs = _rolling(rets, w, np.std)
            with np.errstate(divide="ignore", invalid="ignore"):
                sharpe = np.where(rs > 0, rm / rs * np.sqrt(trading_days_per_year), 0.0)
            label = f"{w}d"
            ax_.plot(sharpe, color=colors[i % len(colors)], linewidth=1.0, label=label)

        ax_.axhline(0, color=DARK_GRAY, linewidth=0.5, linestyle="--")
        ax_.set_title(title)
        ax_.set_ylabel("Sharpe")
        ax_.legend(loc="upper left", framealpha=0.3)
        return finalize(fig, show=show, save=save)


# ── Rolling Volatility ──────────────────────────────────────────────────────


def rolling_volatility(
    result,
    *,
    windows: Optional[List[int]] = None,
    ax: Optional[Axes] = None,
    title: str = "Rolling Volatility",
    trading_days_per_year: float = 365.25,
    figsize: Tuple[float, float] = (14, 4),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Rolling annualized volatility."""
    if windows is None:
        windows = [126, 252]
    colors = [ACCENT, ACCENT_ALT, GREEN, RED]

    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)
        rets = daily_returns_array(result)

        for i, w in enumerate(windows):
            if len(rets) < w:
                continue
            rs = _rolling(rets, w, np.std)
            vol = rs * np.sqrt(trading_days_per_year)
            ax_.plot(vol, color=colors[i % len(colors)], linewidth=1.0, label=f"{w}d")

        ax_.set_title(title)
        ax_.set_ylabel("Volatility")
        ax_.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax_.legend(loc="upper left", framealpha=0.3)
        return finalize(fig, show=show, save=save)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _rolling(arr: np.ndarray, window: int, func) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(window - 1, len(arr)):
        out[i] = func(arr[i - window + 1 : i + 1])
    return out
