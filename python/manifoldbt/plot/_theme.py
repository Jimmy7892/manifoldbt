"""Clean dark theme — modern, readable, quant-oriented."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Color palette — neutral dark, no decorative colors
# ---------------------------------------------------------------------------
WHITE = "#e8e6e3"
GRAY = "#8a8a8a"
DARK_GRAY = "#555555"
ACCENT = "#60a5fa"       # Neutral blue — primary data line
ACCENT_ALT = "#a78bfa"   # Subtle purple — secondary series
GREEN = "#22c55e"        # Positive only
RED = "#ef4444"          # Negative only
ORANGE = "#f59e0b"       # OOS / warning

BG_FIGURE = "#0c0c0f"
BG_AXES = "#111116"
BORDER = "#1e1e24"
GRID_RGBA = (1.0, 1.0, 1.0, 0.04)

SERIES_COLORS = [ACCENT, ACCENT_ALT, "#2dd4bf", ORANGE, RED, GREEN, "#f472b6", WHITE]

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
THEME: Dict[str, Any] = {
    "figure.facecolor": BG_FIGURE,
    "figure.edgecolor": BG_FIGURE,
    "figure.dpi": 120,
    "axes.facecolor": BG_AXES,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": GRAY,
    "axes.titlecolor": WHITE,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "axes.titlepad": 12,
    "axes.labelsize": 9,
    "axes.labelpad": 8,
    "axes.grid": True,
    "grid.color": GRID_RGBA,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": DARK_GRAY,
    "ytick.color": DARK_GRAY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": WHITE,
    "font.family": "monospace",
    "font.size": 9,
    "legend.facecolor": BG_AXES,
    "legend.edgecolor": BORDER,
    "legend.fontsize": 8,
    "legend.labelcolor": GRAY,
    "lines.linewidth": 1.3,
    "lines.antialiased": True,
    "savefig.facecolor": BG_FIGURE,
    "savefig.edgecolor": BG_FIGURE,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
}


def _build_theme() -> Dict[str, Any]:
    """Finalize THEME dict with cycler."""
    import matplotlib.pyplot as plt
    theme = dict(THEME)
    theme["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    return theme


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
def _register_colormaps() -> None:
    """Register custom colormaps (idempotent)."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl

    _cmaps = {
        "bt_diverging": [(0.0, "#b91c1c"), (0.5, "#262626"), (1.0, "#15803d")],
        "bt_sequential": [(0.0, "#b91c1c"), (0.5, "#d97706"), (1.0, "#15803d")],
        "bt_correlation": [(0.0, "#b91c1c"), (0.5, "#262626"), (1.0, "#1d4ed8")],
    }
    for name, stops in _cmaps.items():
        try:
            mpl.colormaps.get_cmap(name)
        except ValueError:
            positions = [s[0] for s in stops]
            colors = [s[1] for s in stops]
            cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, colors)), N=256)
            mpl.colormaps.register(cmap, name=name)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
def apply_theme() -> None:
    """Apply the dark theme globally."""
    import matplotlib.pyplot as plt
    _register_colormaps()
    plt.rcParams.update(_build_theme())


@contextmanager
def theme_context():
    """Context manager: apply theme temporarily."""
    import matplotlib.pyplot as plt
    _register_colormaps()
    with plt.rc_context(_build_theme()):
        yield
