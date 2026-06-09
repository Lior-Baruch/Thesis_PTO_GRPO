"""
figures.py — small, reusable plotting *helpers* (not full plots).

By design (v2) the actual seaborn/matplotlib code lives **inline in the notebooks** so every figure is
visible and editable. This module only holds the bits worth sharing: a consistent style, a stable arm
palette, the left-to-right model order, and a subplot-grid scaffold. Compose your plots in the
notebook on top of the tidy `scores_long` / `behavior_by_iter` frames.
"""

from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# PTO = purples, GRPO = blues; darker = higher look-ahead K.
_ARM_COLORS = {
    "PTO_LA0": "#7b4fb0", "PTO_LA5": "#c5b0d5",
    "GRPO_LA0": "#1f77b4", "GRPO_LA5": "#9ecae1",
}


def set_style():
    """Consistent, publication-grade global style for every Exp3 figure.

    Whitegrid theme + tight, vector-friendly save defaults so `exports.save_fig` produces clean
    PDF (editable text via fonttype 42) and 200-dpi PNG with no manual tweaking.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable/embeddable text in vector output
        "figure.autolayout": False,
    })


def arm_palette(labels: Sequence[str]) -> dict:
    """Stable ``{arm_label: color}`` (unknown arms get tab10 fallbacks)."""
    pal = {l: _ARM_COLORS.get(l) for l in labels}
    missing = [l for l in labels if pal[l] is None]
    for l, c in zip(missing, sns.color_palette("tab10", len(missing)).as_hex()):
        pal[l] = c
    return pal


def model_order(scores_long) -> List[str]:
    """Models left-to-right by (method, K, iteration) — for stable bar/x ordering."""
    meta = (scores_long[["model", "method", "K", "iteration"]]
            .drop_duplicates().sort_values(["method", "K", "iteration"]))
    return meta["model"].tolist()


def grid(n: int, ncols: int = 3, panel=(5.0, 3.2)):
    """A ready (fig, axes_flat) grid sized for *n* panels; trailing axes hidden.

    Usage in a notebook:
        fig, axes = figures.grid(len(METRICS), ncols=3)
        for ax, m in zip(axes, METRICS):
            sns.lineplot(..., ax=ax)
    """
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False)
    axes_flat = axes.flat
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)
