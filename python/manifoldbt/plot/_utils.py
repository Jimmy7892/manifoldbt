"""Shared plotting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from manifoldbt.plot._theme import theme_context


def get_or_create_ax(
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> Tuple[Figure, Axes]:
    """Return (fig, ax). Creates a new themed figure if *ax* is None."""
    if ax is not None:
        return ax.figure, ax
    fig, new_ax = plt.subplots(figsize=figsize)
    return fig, new_ax


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal fraction as a percentage string."""
    return f"{value * 100:+.{decimals}f}%"


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a number as currency."""
    symbol = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3"}.get(currency, "")
    return f"{symbol}{value:,.2f}"


def finalize(
    fig: Figure,
    *,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Optionally save and/or display the figure, then return it."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass  # Skip when axes are incompatible (e.g. inside GridSpec)
    if save is not None:
        fig.savefig(str(save), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def auto_title(result, fallback: str) -> str:
    """Build a title from result manifest strategy_name, or use fallback."""
    try:
        manifest = result.manifest
        if isinstance(manifest, dict) and "strategy_name" in manifest:
            return manifest["strategy_name"]
    except Exception:
        pass
    return fallback
