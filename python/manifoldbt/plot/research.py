"""Charts for research analysis results (sweep, walk-forward, stability)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
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
from manifoldbt.plot._convert import daily_returns_array, equity_with_dates
from manifoldbt.plot._utils import finalize, format_pct, get_or_create_ax


# ── 2D Parameter Sweep Heatmap ──────────────────────────────────────────────


def heatmap_2d(
    sweep_result: Dict[str, Any],
    *,
    ax: Optional[Axes] = None,
    annotate: bool = True,
    fmt: str = ".3f",
    highlight_best: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """2D parameter sweep heatmap from ``run_sweep_2d()`` result.

    Expected keys: metric_grid, x_values, y_values, x_param, y_param, metric.
    """
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)

        grid = np.array(sweep_result["metric_grid"], dtype=np.float64)
        x_vals_raw = sweep_result["x_values"]
        y_vals_raw = sweep_result["y_values"]
        x_param = sweep_result.get("x_param", "x")
        y_param = sweep_result.get("y_param", "y")
        metric = sweep_result.get("metric", "metric")

        # Extract numeric values from ScalarValue dicts like {'Float64': 1.23}
        def _extract_val(v):
            if isinstance(v, dict):
                for val in v.values():
                    return val
            return v

        x_vals = [_extract_val(v) for v in x_vals_raw]
        y_vals = [_extract_val(v) for v in y_vals_raw]

        cmap = plt.get_cmap("bt_sequential")
        im = ax_.imshow(
            grid, cmap=cmap, aspect="auto", interpolation="nearest",
            origin="lower",
        )

        # Adaptive tick labels: show max ~10 ticks per axis
        max_ticks = 10
        nx, ny = len(x_vals), len(y_vals)

        x_step = max(1, nx // max_ticks)
        x_tick_idx = list(range(0, nx, x_step))
        ax_.set_xticks(x_tick_idx)
        ax_.set_xticklabels([f"{x_vals[i]:.2f}" for i in x_tick_idx], rotation=45, ha="right", fontsize=9)

        y_step = max(1, ny // max_ticks)
        y_tick_idx = list(range(0, ny, y_step))
        ax_.set_yticks(y_tick_idx)
        ax_.set_yticklabels([f"{y_vals[i]:.2f}" for i in y_tick_idx], fontsize=9)

        ax_.set_xlabel(x_param, fontsize=10, labelpad=8)
        ax_.set_ylabel(y_param, fontsize=10, labelpad=8)

        # Only annotate if grid is small enough to be readable
        if annotate and nx * ny <= 100:
            for yi in range(grid.shape[0]):
                for xi in range(grid.shape[1]):
                    val = grid[yi, xi]
                    if np.isnan(val):
                        continue
                    norm = (val - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid) + 1e-12)
                    txt_color = "white" if norm > 0.6 or norm < 0.4 else "#1a1a1a"
                    ax_.text(
                        xi, yi, f"{val:{fmt}}",
                        ha="center", va="center", fontsize=8, color=txt_color,
                    )

        if highlight_best:
            from scipy.ndimage import gaussian_filter

            # Plateau-optimal: Gaussian blur finds the center of the best
            # stable region, not a lucky spike (overfit-resistant).
            # sigma = ~5% of each axis → favors broad plateaus.
            sigma_y = max(1.0, grid.shape[0] * 0.05)
            sigma_x = max(1.0, grid.shape[1] * 0.05)
            smoothed = gaussian_filter(
                np.nan_to_num(grid, nan=np.nanmin(grid)),
                sigma=(sigma_y, sigma_x),
            )
            best_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            best_val = grid[best_idx]
            best_x = x_vals[best_idx[1]]
            best_y = y_vals[best_idx[0]]

            rect = plt.Rectangle(
                (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                linewidth=2.5, edgecolor="white", facecolor="none",
            )
            ax_.add_patch(rect)
            best_label = f"best: {best_val:{fmt}} ({x_param}={best_x:.0f}, {y_param}={best_y:.0f})"
            ax_.text(
                best_idx[1], best_idx[0], f"{best_val:{fmt}}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.7, "edgecolor": "white"},
            )

        combos = nx * ny
        main_title = title or f"{metric} -- Parameter Sweep ({combos:,} combos)"
        if highlight_best:
            ax_.set_title(f"{main_title}\n{best_label}", fontsize=11)
        else:
            ax_.set_title(main_title)
        fig.colorbar(im, ax=ax_, shrink=0.7)
        return finalize(fig, show=show, save=save)


# ── 3D Surface Plot ─────────────────────────────────────────────────────────


def surface_3d(
    sweep_result: Dict[str, Any],
    *,
    highlight_best: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    elev: float = 30,
    azim: float = -45,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """3D surface plot from a 2D parameter sweep result.

    Same input format as ``heatmap_2d``:
    Expected keys: metric_grid, x_values, y_values, x_param, y_param, metric.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    with theme_context():
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        grid = np.array(sweep_result["metric_grid"], dtype=np.float64)
        x_vals_raw = sweep_result["x_values"]
        y_vals_raw = sweep_result["y_values"]
        x_param = sweep_result.get("x_param", "x")
        y_param = sweep_result.get("y_param", "y")
        metric = sweep_result.get("metric", "metric")

        def _extract_val(v):
            if isinstance(v, dict):
                for val in v.values():
                    return val
            return v

        x_vals = np.array([_extract_val(v) for v in x_vals_raw], dtype=np.float64)
        y_vals = np.array([_extract_val(v) for v in y_vals_raw], dtype=np.float64)

        X, Y = np.meshgrid(x_vals, y_vals)

        cmap = plt.get_cmap("bt_sequential")
        surf = ax.plot_surface(
            X, Y, grid,
            cmap=cmap, alpha=0.9, linewidth=0, antialiased=True,
            rstride=max(1, grid.shape[0] // 80),
            cstride=max(1, grid.shape[1] // 80),
        )

        if highlight_best:
            from scipy.ndimage import gaussian_filter

            sigma_y = max(1.0, grid.shape[0] * 0.05)
            sigma_x = max(1.0, grid.shape[1] * 0.05)
            smoothed = gaussian_filter(
                np.nan_to_num(grid, nan=np.nanmin(grid)),
                sigma=(sigma_y, sigma_x),
            )
            best_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            best_val = grid[best_idx]
            bx = x_vals[best_idx[1]]
            by = y_vals[best_idx[0]]
            ax.scatter([bx], [by], [best_val], color="white", s=80, zorder=5,
                       edgecolors="black", linewidths=1.5)
            best_label = f"best: {best_val:.3f} ({x_param}={bx:.0f}, {y_param}={by:.0f})"

        # Force dark panes (matplotlib 3D ignores rc theme)
        pane_color = (0.1, 0.1, 0.1, 0.9)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color("white")
            axis.set_tick_params(colors="white")
        ax.set_xlabel(x_param, fontsize=10, labelpad=10)
        ax.set_ylabel(y_param, fontsize=10, labelpad=10)
        ax.set_zlabel(metric, fontsize=10, labelpad=10)
        ax.view_init(elev=elev, azim=azim)

        combos = len(x_vals) * len(y_vals)
        main_title = title or f"{metric} -- Surface ({combos:,} combos)"
        if highlight_best:
            ax.set_title(f"{main_title}\n{best_label}", fontsize=11)
        else:
            ax.set_title(main_title)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

        return finalize(fig, show=show, save=save)


# ── Walk-Forward Analysis ────────────────────────────────────────────────────


def walk_forward(
    wf_result: Dict[str, Any],
    *,
    mode: str = "auto",
    full_result=None,
    ax: Optional[Axes] = None,
    is_color: str = ACCENT,
    oos_color: str = ORANGE,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Walk-forward analysis chart.

    Args:
        mode: ``"auto"`` (equity curves if available, bars otherwise),
              ``"equity"`` (force equity curves), ``"bars"`` (force bar chart),
              ``"stitched"`` (stitched OOS vs full backtest).
        full_result: BacktestResult from ``bt.run()`` on the full period
              (no WFO). Used by ``"stitched"`` mode as the baseline.
              If not provided, stitched mode only shows the OOS curve.
    """
    folds = wf_result["folds"]
    has_equity = any(len(f.get("is_equity", [])) > 0 for f in folds)

    if mode == "auto":
        mode = "equity" if has_equity else "bars"

    if mode == "equity":
        return _walk_forward_equity(wf_result, folds, ax=ax, is_color=is_color,
                                     oos_color=oos_color, title=title, figsize=figsize,
                                     show=show, save=save)
    elif mode == "stitched":
        return _walk_forward_stitched(wf_result, folds, full_result=full_result,
                                       ax=ax, is_color=is_color,
                                       oos_color=oos_color, title=title, figsize=figsize,
                                       show=show, save=save)
    else:
        return _walk_forward_bars(wf_result, folds, ax=ax, is_color=is_color,
                                   oos_color=oos_color, title=title, figsize=figsize,
                                   show=show, save=save)


def _walk_forward_equity(wf_result, folds, *, ax, is_color, oos_color, title, figsize, show, save):
    """Equity curve per fold: IS (blue) + OOS (orange) side by side."""
    from matplotlib.gridspec import GridSpec

    optimize_metric = wf_result.get("optimize_metric", "sharpe")
    n = len(folds)

    with theme_context():
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, n, figure=fig, wspace=0.08)
        fig.suptitle(title or f"Walk-Forward Analysis ({optimize_metric})", fontsize=10)

        for i, fold in enumerate(folds):
            ax_ = fig.add_subplot(gs[0, i])
            is_eq = fold.get("is_equity", [])
            oos_eq = fold.get("oos_equity", [])

            if is_eq:
                is_x = np.arange(len(is_eq))
                ax_.plot(is_x, is_eq, color=is_color, linewidth=1.2, alpha=0.8)

            if oos_eq:
                oos_x = np.arange(len(is_eq), len(is_eq) + len(oos_eq))
                ax_.plot(oos_x, oos_eq, color=oos_color, linewidth=1.2, alpha=0.8)

            if is_eq and oos_eq:
                ax_.axvline(x=len(is_eq), color=DARK_GRAY, linewidth=0.8, linestyle="--")

            # Extract metric values for labels
            def _get_metric(key):
                val = fold.get(key)
                if isinstance(val, dict):
                    return val.get(optimize_metric, val.get("sharpe", 0))
                return val if val is not None else 0

            is_m = _get_metric("is_metrics") or _get_metric("is_metric")
            oos_m = _get_metric("oos_metrics") or _get_metric("oos_metric")

            ax_.text(0.05, 0.92, f"IS: {is_m:.2f}", transform=ax_.transAxes,
                     fontsize=7, color=is_color, fontfamily="monospace")
            ax_.text(0.05, 0.82, f"OOS: {oos_m:.2f}", transform=ax_.transAxes,
                     fontsize=7, color=oos_color, fontfamily="monospace")

            fold_idx = fold.get("fold_index", fold.get("fold", i))
            ax_.set_title(f"Fold {fold_idx + 1}", fontsize=8)
            ax_.tick_params(labelsize=6)
            ax_.grid(True, alpha=0.08)
            if i > 0:
                ax_.set_yticklabels([])

        return finalize(fig, show=show, save=save)


def _walk_forward_bars(wf_result, folds, *, ax, is_color, oos_color, title, figsize, show, save):
    """Grouped bar chart: IS vs OOS metric per fold."""
    optimize_metric = wf_result.get("optimize_metric", "sharpe")
    n = len(folds)
    x = np.arange(n)
    width = 0.35

    def _extract(fold, key):
        val = fold.get(key)
        if isinstance(val, dict):
            return val.get(optimize_metric, val.get("sharpe", 0))
        return val if val is not None else 0

    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)

        is_vals = [_extract(f, "is_metrics") or _extract(f, "is_metric") for f in folds]
        oos_vals = [_extract(f, "oos_metrics") or _extract(f, "oos_metric") for f in folds]

        ax_.bar(x - width / 2, is_vals, width, label="In-Sample", color=is_color, alpha=0.65)
        ax_.bar(x + width / 2, oos_vals, width, label="Out-of-Sample", color=oos_color, alpha=0.65)

        for i, (is_v, oos_v) in enumerate(zip(is_vals, oos_vals)):
            if is_v != 0:
                ax_.text(i - width / 2, is_v, f"{is_v:.2f}", ha="center",
                         va="bottom" if is_v > 0 else "top", fontsize=7, color=is_color)
            if oos_v != 0:
                ax_.text(i + width / 2, oos_v, f"{oos_v:.2f}", ha="center",
                         va="bottom" if oos_v > 0 else "top", fontsize=7, color=oos_color)

        ax_.set_xticks(x)
        ax_.set_xticklabels([f"Fold {f.get('fold_index', f.get('fold', i)) + 1}" for i, f in enumerate(folds)])
        ax_.axhline(0, color=DARK_GRAY, linewidth=0.5, linestyle="--")
        ax_.set_title(title or f"Walk-Forward Analysis ({optimize_metric})")
        ax_.set_ylabel(optimize_metric.capitalize())
        ax_.legend(loc="upper right")
        return finalize(fig, show=show, save=save)


def _walk_forward_stitched(wf_result, folds, *, full_result=None, ax, is_color, oos_color, title, figsize, show, save):
    """Stitched OOS equity vs full backtest.

    - Orange: OOS segments from each fold, chained end-to-end.
      This is the TRUE out-of-sample performance of the WFO strategy.
    - Blue: full backtest with default params over the same period (no WFO).
      This is what you'd get without walk-forward optimization.

    If orange ~ blue → no overfitting, WFO adds little.
    If blue >> orange → full backtest is overfitted.
    If orange >> blue → WFO optimization adds real value.

    Args:
        full_result: BacktestResult from bt.run() on the full period.
    """
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)

        # 1. Stitch OOS segments: chain so each starts where previous ended
        stitched = []
        current_val = None
        fold_boundaries = []
        for fold in folds:
            oos_eq = fold.get("oos_equity", [])
            if not oos_eq:
                continue
            oos = np.array(oos_eq, dtype=float)
            if current_val is None:
                stitched.extend(oos.tolist())
                current_val = oos[-1]
            else:
                scale = current_val / oos[0] if oos[0] != 0 else 1.0
                scaled = oos * scale
                stitched.extend(scaled.tolist())
                current_val = scaled[-1]
            fold_boundaries.append(len(stitched))

        if not stitched:
            ax_.set_title("No OOS equity data available")
            return finalize(fig, show=show, save=save)

        stitched = np.array(stitched)
        x = np.arange(len(stitched))

        # 2. Full backtest equity (if provided)
        if full_result is not None:
            full_eq_raw = full_result.equity_curve
            full_eq = np.array(full_eq_raw)
            if len(full_eq) > 0:
                # Resample to match stitched length
                indices = np.linspace(0, len(full_eq) - 1, len(stitched), dtype=int)
                full_resampled = full_eq[indices].astype(float)
                # Normalize to start at same value as stitched
                if full_resampled[0] != 0:
                    full_resampled = full_resampled * (stitched[0] / full_resampled[0])
                ax_.plot(x, full_resampled, color=is_color, linewidth=0.8, alpha=0.4,
                         label="Full backtest (default params)")

                full_ret = (full_resampled[-1] / full_resampled[0] - 1) * 100

        # 3. Plot stitched OOS on top
        ax_.plot(x, stitched, color=oos_color, linewidth=0.9, alpha=0.85,
                 label="Walk-forward (stitched OOS)", zorder=3)

        # Fold boundaries
        for b in fold_boundaries[:-1]:
            ax_.axvline(x=b, color=DARK_GRAY, linewidth=0.5,
                        linestyle="--", alpha=0.3)

        # No floating text - returns are visible from the curves

        ax_.set_title(title or "Walk-Forward: Stitched OOS vs Full Backtest")
        ax_.set_xlabel("Bars")
        ax_.set_ylabel("Equity")
        ax_.legend(loc="upper left", fontsize=8)
        ax_.grid(True, alpha=0.08)
        return finalize(fig, show=show, save=save)


# ── Parameter Stability ─────────────────────────────────────────────────────


def stability(
    stability_result: Dict[str, Any],
    *,
    ax: Optional[Axes] = None,
    line_color: str = ACCENT,
    band_color: str = ACCENT,
    band_alpha: float = 0.15,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Parameter stability chart with mean +/- std shaded bands.

    Expected keys: values, metric_values, mean_metric, std_metric,
                   param_name, metric, stability_score.
    """
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)

        param_vals = np.array(stability_result["values"], dtype=np.float64)
        metric_vals = np.array(stability_result["metric_values"], dtype=np.float64)
        mean = stability_result["mean_metric"]
        std = stability_result["std_metric"]
        param_name = stability_result.get("param_name", "parameter")
        metric_name = stability_result.get("metric", "metric")
        score = stability_result.get("stability_score", None)

        ax_.plot(param_vals, metric_vals, color=line_color, linewidth=1.8, marker="o", markersize=4)
        ax_.axhline(mean, color=band_color, linewidth=1.0, linestyle="--", label=f"Mean: {mean:.3f}")
        ax_.fill_between(
            param_vals, mean - std, mean + std,
            color=band_color, alpha=band_alpha, label=f"\u00b11\u03c3: {std:.3f}",
        )

        ax_.set_xlabel(param_name)
        ax_.set_ylabel(metric_name)
        t = title or f"{metric_name} Stability"
        if score is not None:
            t += f"  (score: {score:.2f})"
        ax_.set_title(t)
        ax_.legend(loc="upper right")
        return finalize(fig, show=show, save=save)


# ── Correlation Matrix ───────────────────────────────────────────────────────


def correlation_matrix(
    symbols: List[str],
    matrix: List[List[float]],
    *,
    ax: Optional[Axes] = None,
    annotate: bool = True,
    title: str = "Correlation Matrix",
    figsize: Tuple[float, float] = (8, 7),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Symbol correlation matrix heatmap."""
    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)

        mat = np.array(matrix, dtype=np.float64)
        n = len(symbols)
        cmap = plt.get_cmap("bt_correlation")
        im = ax_.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="equal", interpolation="nearest")

        ax_.set_xticks(range(n))
        ax_.set_xticklabels(symbols, rotation=45, ha="right")
        ax_.set_yticks(range(n))
        ax_.set_yticklabels(symbols)

        if annotate:
            for yi in range(n):
                for xi in range(n):
                    val = mat[yi, xi]
                    txt_color = DARK_GRAY if yi == xi else ("white" if abs(val) > 0.5 else DARK_GRAY)
                    ax_.text(
                        xi, yi, f"{val:.2f}",
                        ha="center", va="center", fontsize=9, color=txt_color,
                    )

        ax_.set_title(title)
        fig.colorbar(im, ax=ax_, shrink=0.7)
        return finalize(fig, show=show, save=save)


# ── Monte Carlo Fan ──────────────────────────────────────────────────────────


def monte_carlo(
    result,
    *,
    n_simulations: int = 1000,
    method: str = "bootstrap",
    percentiles: Optional[List[int]] = None,
    n_sample_paths: int = 50,
    ax: Optional[Axes] = None,
    median_color: str = ACCENT,
    band_color: str = ACCENT,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    seed: Optional[int] = None,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> Figure:
    """Monte Carlo fan chart with percentile bands, sample paths, and risk stats.

    Args:
        result: BacktestResult from ``bt.run()``.
        n_simulations: Number of simulated paths.
        method: ``"bootstrap"`` (sample with replacement, default) for tail risk
            estimation, or ``"permutation"`` (shuffle without replacement) for
            path-dependency testing.
        percentiles: Percentile levels for bands. Default ``[5, 25, 50, 75, 95]``.
        n_sample_paths: Number of individual paths to draw (faded). 0 to disable.
        seed: Random seed for reproducibility.
    """
    # Cap to 1000 sims for Community
    try:
        from manifoldbt import _license_info, _warn_pro
        tier, _ = _license_info()
        if tier != "Pro" and n_simulations > 1000:
            _warn_pro(f"Monte Carlo capped to 1,000 sims (requested {n_simulations:,})")
            n_simulations = 1000
    except Exception:
        if n_simulations > 1000:
            n_simulations = 1000

    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    if title is None:
        method_label = "bootstrap" if method == "bootstrap" else "permutation"
        title = f"Monte Carlo - {n_simulations:,} paths ({method_label})"

    with theme_context():
        fig, ax_ = get_or_create_ax(ax, figsize)

        rets = daily_returns_array(result)
        _, orig_equity = equity_with_dates(result)

        if len(rets) < 2:
            ax_.set_title(title + " (insufficient data)")
            return finalize(fig, show=show, save=save)

        rng = np.random.default_rng(seed)
        initial = orig_equity[0] if len(orig_equity) > 0 else 1.0
        n_days = len(rets)

        # Generate simulated paths
        paths = np.zeros((n_simulations, n_days + 1))
        paths[:, 0] = initial
        for i in range(n_simulations):
            if method == "permutation":
                sampled = rng.permutation(rets)
            else:  # bootstrap (default)
                sampled = rng.choice(rets, size=n_days, replace=True)
            paths[i, 1:] = initial * np.cumprod(1.0 + sampled)

        # Compute percentile bands
        x = np.arange(n_days + 1)
        pct_lines = {pct: np.percentile(paths, pct, axis=0) for pct in percentiles}

        # Draw sample paths (faded)
        if n_sample_paths > 0:
            for i in range(min(n_sample_paths, n_simulations)):
                ax_.plot(x, paths[i], color=band_color, linewidth=0.3, alpha=0.06)

        # Fill between symmetric bands
        for lo, hi in [(0, -1), (1, -2)]:
            ax_.fill_between(
                x, pct_lines[percentiles[lo]], pct_lines[percentiles[hi]],
                color=band_color, alpha=0.08,
            )

        # Original equity (dashed) — resample to match MC daily resolution
        if len(orig_equity) > n_days * 2:
            indices = np.linspace(0, len(orig_equity) - 1, n_days + 1, dtype=int)
            orig_resampled = np.array(orig_equity)[indices]
        else:
            orig_resampled = np.array(orig_equity[:n_days + 1])
        orig_x = np.arange(len(orig_resampled))
        ax_.plot(orig_x, orig_resampled, color="#e8e9ed", linewidth=0.8,
                 alpha=0.4, linestyle="--", label="Original")

        if method == "bootstrap":
            # Bootstrap: percentile lines with final return %
            for pct in percentiles:
                ret_pct = (pct_lines[pct][-1] / initial - 1) * 100
                if pct == 50:
                    ax_.plot(x, pct_lines[pct], color=median_color, linewidth=2,
                             label=f"P{pct} (median): {ret_pct:+.1f}%", zorder=3)
                else:
                    ax_.plot(x, pct_lines[pct], color=band_color, linewidth=0.5,
                             alpha=0.4, label=f"P{pct}: {ret_pct:+.1f}%")

            # Drawdown stats
            running_peak = np.maximum.accumulate(paths, axis=1)
            drawdowns = (paths - running_peak) / running_peak
            max_dd_per_path = drawdowns.min(axis=1) * 100

            dd_p5 = np.percentile(max_dd_per_path, 5)
            dd_p50 = np.percentile(max_dd_per_path, 50)

            # P(ruin)
            p_ruin = np.mean((paths[:, -1] / initial - 1) < -0.5) * 100

            stats_text = f"P(ruin) = {p_ruin:.2f}%\nMax DD (P5): {dd_p5:.1f}%\nMax DD (median): {dd_p50:.1f}%"
            ax_.text(
                0.98, 0.95, stats_text,
                transform=ax_.transAxes, ha="right", va="top",
                color="#8a8a8a", fontsize=8, fontfamily="monospace",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "#111116",
                       "edgecolor": "#1e1e24", "alpha": 0.9},
            )

        else:
            # Permutation: all paths end at the same point.
            # Skill vs luck analysis: compare original drawdown to permuted distribution.
            ax_.plot(x, pct_lines[50], color=median_color, linewidth=2,
                     label="Median path", zorder=3)
            for pct in percentiles:
                if pct != 50:
                    ax_.plot(x, pct_lines[pct], color=band_color, linewidth=0.5, alpha=0.4)

            # Max drawdown per path
            running_peak = np.maximum.accumulate(paths, axis=1)
            drawdowns = (paths - running_peak) / running_peak
            max_dd_per_path = drawdowns.min(axis=1) * 100

            # Original strategy drawdown
            orig_eq = np.array(orig_resampled)
            orig_peak = np.maximum.accumulate(orig_eq)
            orig_max_dd = ((orig_eq - orig_peak) / orig_peak).min() * 100

            dd_p50 = np.percentile(max_dd_per_path, 50)

            dd_p5 = np.percentile(max_dd_per_path, 5)
            dd_p95 = np.percentile(max_dd_per_path, 95)
            dd_rank = np.mean(max_dd_per_path <= orig_max_dd) * 100

            stats_text = (
                f"Realized max DD:  {orig_max_dd:.1f}%\n"
                f"Permuted DD P5:   {dd_p5:.1f}%\n"
                f"Permuted DD P50:  {dd_p50:.1f}%\n"
                f"Permuted DD P95:  {dd_p95:.1f}%\n"
                f"DD rank:          {dd_rank:.0f}th percentile"
            )
            ax_.text(
                0.98, 0.95, stats_text,
                transform=ax_.transAxes, ha="right", va="top",
                color="#8a8a8a", fontsize=8, fontfamily="monospace",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "#111116",
                       "edgecolor": "#1e1e24", "alpha": 0.9},
            )

        ax_.margins(x=0.02)
        ax_.set_title(title)
        ax_.set_xlabel("Days")
        ax_.set_ylabel("Equity")
        ax_.legend(loc="upper left", fontsize=7, framealpha=0.3)
        return finalize(fig, show=show, save=save)
