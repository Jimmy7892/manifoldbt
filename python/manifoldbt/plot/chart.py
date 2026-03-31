"""Candlestick chart with indicators and trade markers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from manifoldbt.plot._convert import trades_arrays
from manifoldbt.plot._theme import (
    ACCENT,
    ACCENT_ALT,
    BG_AXES,
    BG_FIGURE,
    BORDER,
    DARK_GRAY,
    GREEN,
    GRID_RGBA,
    GRAY,
    RED,
    WHITE,
)
from manifoldbt.plot._utils import finalize


# ---------------------------------------------------------------------------
# EMA helper (pure numpy)
# ---------------------------------------------------------------------------

def _ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average matching standard TradingView formula."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(close)
    out[0] = close[0]
    for i in range(1, len(close)):
        out[i] = alpha * close[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full_like(close, np.nan)
    cs = np.cumsum(close)
    out[period - 1 :] = (cs[period - 1 :] - np.concatenate([[0], cs[: -period]])) / period
    return out


# ---------------------------------------------------------------------------
# OHLC loading
# ---------------------------------------------------------------------------

def _load_bars(
    store,
    symbol_id: int,
    start_ns: int,
    end_ns: int,
    bar_interval_seconds: int,
) -> Dict[str, np.ndarray]:
    """Load OHLC bars from parquet and resample to target interval."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import datetime, timezone, timedelta

    data_root = Path(store.data_root())

    start_dt = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)

    # Try Arrow IPC file first (new layout), then Parquet partitions (legacy)
    arrow_dir = Path(store.data_root()) / "mega" if not str(data_root).endswith("mega") else data_root
    ipc_path = arrow_dir / "bars_1m" / f"{symbol_id}.arrow"
    if ipc_path.exists():
        table = pa.ipc.open_file(str(ipc_path)).read_all()
    else:
        tables = []
        day = start_dt.date()
        end_day = end_dt.date()
        while day <= end_day:
            path = (
                data_root
                / "bars_1m"
                / str(symbol_id)
                / str(day.year)
                / f"{day.month:02d}"
                / f"{day.day:02d}.parquet"
            )
            if path.exists():
                tables.append(pq.read_table(str(path)))
            day += timedelta(days=1)

        if not tables:
            return {}

        table = pa.concat_tables(tables)

    # Filter to time range
    ts_col = table.column("timestamp").cast(pa.int64()).to_numpy(zero_copy_only=False)
    mask = (ts_col >= start_ns) & (ts_col < end_ns)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return {}
    table = table.take(indices)

    ts = table.column("timestamp").cast(pa.int64()).to_numpy(zero_copy_only=False)
    o = table.column("open").to_numpy(zero_copy_only=False)
    h = table.column("high").to_numpy(zero_copy_only=False)
    l = table.column("low").to_numpy(zero_copy_only=False)
    c = table.column("close").to_numpy(zero_copy_only=False)
    v = table.column("volume").to_numpy(zero_copy_only=False)

    # Resample to target interval
    interval_ns = bar_interval_seconds * 1_000_000_000
    bucket = ts // interval_ns

    unique_buckets, first_idx = np.unique(bucket, return_index=True)
    n_bars = len(unique_buckets)

    ts_out = np.empty(n_bars, dtype=np.int64)
    o_out = np.empty(n_bars)
    h_out = np.empty(n_bars)
    l_out = np.empty(n_bars)
    c_out = np.empty(n_bars)
    v_out = np.empty(n_bars)

    boundaries = np.append(first_idx, len(ts))
    for i in range(n_bars):
        s, e = boundaries[i], boundaries[i + 1]
        ts_out[i] = ts[s]
        o_out[i] = o[s]
        h_out[i] = h[s:e].max()
        l_out[i] = l[s:e].min()
        c_out[i] = c[e - 1]
        v_out[i] = v[s:e].sum()

    return {
        "timestamp": ts_out,
        "open": o_out,
        "high": h_out,
        "low": l_out,
        "close": c_out,
        "volume": v_out,
    }


# ---------------------------------------------------------------------------
# Candlestick drawing
# ---------------------------------------------------------------------------

def _draw_candles(ax, dates, o, h, l, c, width_ratio=0.6):
    """Draw candlestick bodies and wicks on an axes."""
    n = len(dates)
    if n < 2:
        return

    # Width in date units
    delta = np.median(np.diff(dates)).astype("timedelta64[s]").astype(float)
    w = np.timedelta64(int(delta * width_ratio), "s")

    bull = c >= o
    bear = ~bull

    # Wicks (high-low lines)
    for i in range(n):
        color = GREEN if bull[i] else RED
        ax.plot([dates[i], dates[i]], [l[i], h[i]], color=color, linewidth=0.5, alpha=0.7)

    # Bodies
    for mask, color in [(bull, GREEN), (bear, RED)]:
        idx = np.where(mask)[0]
        for i in idx:
            bottom = min(o[i], c[i])
            height = abs(c[i] - o[i])
            if height < 1e-10:
                height = (h[i] - l[i]) * 0.01
            rect = __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (dates[i] - w / 2, bottom),
                w,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
                linewidth=0.5,
            )
            ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

INDICATOR_COLORS = [ACCENT, ACCENT_ALT, "#2dd4bf", "#f59e0b", "#f472b6"]


def _resolve_sym_name(store, symbol_id: int) -> str:
    """Get ticker name from metadata DB."""
    try:
        import sqlite3
        conn = sqlite3.connect(store.metadata_db())
        row = conn.execute(
            "SELECT ticker FROM symbols WHERE id = ?", (symbol_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else f"#{symbol_id}"
    except Exception:
        return f"#{symbol_id}"


def _prepare_chart_data(result, store, symbol_id, n_bars):
    """Load bars, compute trim offset, extract trades — shared by both renderers."""
    manifest = result.manifest
    cfg = manifest.get("config", {})
    tr = cfg.get("time_range", {})
    start_ns = tr["start"]
    end_ns = tr["end"]
    bi = cfg.get("bar_interval", {})
    bar_interval_s = _bar_interval_to_seconds(bi)

    bars = _load_bars(store, symbol_id, start_ns, end_ns, int(bar_interval_s))
    if not bars:
        raise ValueError(f"No bar data found for symbol {symbol_id}")

    total = len(bars["timestamp"])
    offset = max(0, total - n_bars)

    # Filter trades to visible window
    trades = trades_arrays(result)
    trade_ts = trades.get("execution_timestamp", np.array([], dtype="datetime64[ns]"))
    trade_sym = trades.get("symbol_id", np.array([], dtype=np.uint32))
    trade_side = trades.get("side", np.array([], dtype=np.uint8))
    trade_price = trades.get("fill_price", np.array([], dtype=np.float64))
    trade_qty = trades.get("quantity", np.array([], dtype=np.float64))

    ts = bars["timestamp"][offset:]
    sym_mask = trade_sym == symbol_id
    ts_int = trade_ts.view(np.int64)
    time_mask = (ts_int >= ts[0]) & (ts_int <= ts[-1])
    mask = sym_mask & time_mask

    return bars, offset, bar_interval_s, {
        "ts": trade_ts[mask],
        "side": trade_side[mask],
        "price": trade_price[mask],
        "qty": trade_qty[mask],
    }


# ---------------------------------------------------------------------------
# Interactive chart (plotly)
# ---------------------------------------------------------------------------

def _chart_interactive(result, store, symbol_id, *, emas, smas, n_bars, save):
    """Plotly-based interactive candlestick chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    bars, offset, bar_interval_s, filtered_trades = _prepare_chart_data(
        result, store, symbol_id, n_bars,
    )

    close_full = bars["close"]
    ts = bars["timestamp"][offset:]
    o = bars["open"][offset:]
    h = bars["high"][offset:]
    l = bars["low"][offset:]
    c = bars["close"][offset:]
    vol = bars["volume"][offset:]
    dates = ts.view("datetime64[ns]")

    sym_name = _resolve_sym_name(store, symbol_id)
    interval_label = _interval_label(bar_interval_s)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
    )

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=dates, open=o, high=h, low=l, close=c,
            increasing_line_color=GREEN, decreasing_line_color=RED,
            increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
            name="OHLC",
        ),
        row=1, col=1,
    )

    # Indicators
    color_idx = 0
    if emas:
        for period in emas:
            vals = _ema(close_full, period)[offset:]
            color = INDICATOR_COLORS[color_idx % len(INDICATOR_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=dates, y=vals, mode="lines",
                    name=f"EMA({period})",
                    line=dict(color=color, width=1.5),
                ),
                row=1, col=1,
            )
            color_idx += 1

    if smas:
        for period in smas:
            vals = _sma(close_full, period)[offset:]
            color = INDICATOR_COLORS[color_idx % len(INDICATOR_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=dates, y=vals, mode="lines",
                    name=f"SMA({period})",
                    line=dict(color=color, width=1.5, dash="dash"),
                ),
                row=1, col=1,
            )
            color_idx += 1

    # Trade markers
    t_ts = filtered_trades["ts"]
    t_side = filtered_trades["side"]
    t_price = filtered_trades["price"]
    t_qty = filtered_trades["qty"]

    buy_mask = t_side == 1
    sell_mask = t_side == 2

    if buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=t_ts[buy_mask], y=t_price[buy_mask],
                mode="markers",
                name="BUY",
                marker=dict(
                    symbol="triangle-up", size=12,
                    color=GREEN, line=dict(color="white", width=1),
                ),
                text=[f"BUY {q:.6f} @ {p:.2f}" for q, p in
                      zip(t_qty[buy_mask], t_price[buy_mask])],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    if sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=t_ts[sell_mask], y=t_price[sell_mask],
                mode="markers",
                name="SELL",
                marker=dict(
                    symbol="triangle-down", size=12,
                    color=RED, line=dict(color="white", width=1),
                ),
                text=[f"SELL {q:.6f} @ {p:.2f}" for q, p in
                      zip(t_qty[sell_mask], t_price[sell_mask])],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    # Volume bars
    vol_colors = [GREEN if c[i] >= o[i] else RED for i in range(len(c))]
    fig.add_trace(
        go.Bar(
            x=dates, y=vol, name="Volume",
            marker_color=vol_colors, opacity=0.5,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Layout — dark theme
    fig.update_layout(
        title=f"{sym_name}  {interval_label}",
        template="plotly_dark",
        paper_bgcolor=BG_FIGURE,
        plot_bgcolor=BG_AXES,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=700,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    if save:
        fig.write_html(str(save))

    fig.show()
    return fig


# ---------------------------------------------------------------------------
# Matplotlib (static) chart
# ---------------------------------------------------------------------------

def _chart_matplotlib(result, store, symbol_id, *, emas, smas, n_bars, figsize, show, save):
    """Matplotlib-based static candlestick chart."""
    import matplotlib.pyplot as plt
    from manifoldbt.plot._theme import theme_context

    bars, offset, bar_interval_s, filtered_trades = _prepare_chart_data(
        result, store, symbol_id, n_bars,
    )

    close_full = bars["close"]
    ts = bars["timestamp"][offset:]
    o = bars["open"][offset:]
    h = bars["high"][offset:]
    l = bars["low"][offset:]
    c = bars["close"][offset:]
    dates = ts.view("datetime64[ns]")

    sym_name = _resolve_sym_name(store, symbol_id)
    interval_label = _interval_label(bar_interval_s)

    with theme_context():
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[4, 1],
            sharex=True, gridspec_kw={"hspace": 0.05},
        )

        _draw_candles(ax_price, dates, o, h, l, c)

        color_idx = 0
        if emas:
            for period in emas:
                vals = _ema(close_full, period)[offset:]
                color = INDICATOR_COLORS[color_idx % len(INDICATOR_COLORS)]
                ax_price.plot(dates, vals, color=color, linewidth=1.2,
                              label=f"EMA({period})", alpha=0.9)
                color_idx += 1
        if smas:
            for period in smas:
                vals = _sma(close_full, period)[offset:]
                color = INDICATOR_COLORS[color_idx % len(INDICATOR_COLORS)]
                ax_price.plot(dates, vals, color=color, linewidth=1.2,
                              label=f"SMA({period})", linestyle="--", alpha=0.9)
                color_idx += 1

        # Trade markers
        t_ts = filtered_trades["ts"]
        t_side = filtered_trades["side"]
        t_price = filtered_trades["price"]
        buy_mask = t_side == 1
        sell_mask = t_side == 2

        if buy_mask.any():
            ax_price.scatter(
                t_ts[buy_mask], t_price[buy_mask],
                marker="^", color=GREEN, s=80, zorder=5,
                edgecolors=WHITE, linewidths=0.5, label="BUY",
            )
        if sell_mask.any():
            ax_price.scatter(
                t_ts[sell_mask], t_price[sell_mask],
                marker="v", color=RED, s=80, zorder=5,
                edgecolors=WHITE, linewidths=0.5, label="SELL",
            )

        ax_price.legend(loc="upper left", fontsize=8)
        ax_price.set_title(f"{sym_name}  {interval_label}", fontsize=11, loc="left")
        ax_price.set_ylabel("Price", fontsize=9)

        vol = bars["volume"][offset:]
        vol_colors = np.where(c >= o, GREEN, RED)
        ax_vol.bar(dates, vol, width=np.timedelta64(int(bar_interval_s * 0.6), "s"),
                   color=vol_colors, alpha=0.5)
        ax_vol.set_ylabel("Volume", fontsize=9)

        import matplotlib.dates as mdates
        if bar_interval_s < 86400:
            ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=30, ha="right")

    return finalize(fig, show=show, save=save)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chart(
    result,
    store,
    symbol_id: int,
    *,
    emas: Optional[List[int]] = None,
    smas: Optional[List[int]] = None,
    n_bars: int = 120,
    interactive: bool = True,
    figsize: Tuple[float, float] = (14, 7),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
):
    """Plot candlestick chart with indicators and trade markers.

    Args:
        result: BacktestResult.
        store: DataStore (to load OHLC bars).
        symbol_id: Which symbol to chart.
        emas: List of EMA periods to overlay (e.g. [10, 25]).
        smas: List of SMA periods to overlay.
        n_bars: Number of bars to display (last N).
        interactive: Use plotly (True) or matplotlib (False).
        figsize: Figure size (matplotlib only).
        show: Display the chart (matplotlib only; plotly always shows).
        save: Save path (.html for plotly, .png for matplotlib).
    """
    if interactive:
        return _chart_interactive(
            result, store, symbol_id,
            emas=emas, smas=smas, n_bars=n_bars, save=save,
        )
    return _chart_matplotlib(
        result, store, symbol_id,
        emas=emas, smas=smas, n_bars=n_bars,
        figsize=figsize, show=show, save=save,
    )


def _bar_interval_to_seconds(bi: dict) -> int:
    """Convert manifest bar_interval dict to seconds."""
    if "Seconds" in bi:
        return bi["Seconds"]
    if "Minutes" in bi:
        return bi["Minutes"] * 60
    if "Hours" in bi:
        return bi["Hours"] * 3600
    if "Days" in bi:
        return bi["Days"] * 86400
    return 3600


def _interval_label(seconds: float) -> str:
    """Human-readable interval label."""
    s = int(seconds)
    if s >= 86400:
        return f"{s // 86400}D"
    if s >= 3600:
        return f"{s // 3600}H"
    if s >= 60:
        return f"{s // 60}m"
    return f"{s}s"
