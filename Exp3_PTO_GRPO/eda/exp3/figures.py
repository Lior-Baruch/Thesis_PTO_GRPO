"""
figures.py — small, reusable plotting *helpers* (not full plots).

By design (v2) the actual seaborn/matplotlib code lives **inline in the notebooks** so every figure is
visible and editable. This module only holds the bits worth sharing: a consistent style, a stable arm
palette, the left-to-right model order, and a subplot-grid scaffold. Compose your plots in the
notebook on top of the tidy `scores_long` / `behavior_by_iter` frames.
"""

import re
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Okabe-Ito colourblind-safe palette. Grouped by TEMPERATURE so the method reads at a glance
# (PTO = cool / blues, GRPO = warm / orange-red), while the two within-method look-ahead arms stay
# clearly distinct. Base = neutral grey.
_ARM_COLORS = {
    "PTO_LA0": "#0072B2",   # blue
    "PTO_LA5": "#56B4E9",   # sky blue
    "GRPO_LA0": "#D55E00",  # vermillion
    "GRPO_LA5": "#E69F00",  # orange
    "Base": "#555555",      # the pooled descriptive base (see scores.collapse_base)
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
    """Models left-to-right by (method, K, iteration) — for stable bar/x ordering.

    The pooled descriptive ``Base`` (``scores.collapse_base``: method ``"Base"``, K ``-1``)
    always sorts first.
    """
    meta = (scores_long[["model", "method", "K", "iteration"]]
            .drop_duplicates().sort_values(["method", "K", "iteration"]))
    order = meta["model"].tolist()
    if "Base" in order:  # guarantee the pooled base leads regardless of sort keys
        order = ["Base"] + [m for m in order if m != "Base"]
    return order


_MODEL_RE = re.compile(r"^(PTO|GRPO)Exp3_LA(\d+)_(Base|I\d+)$")


def clean_label(model: str) -> str:
    """Tidy (full, no-abbreviation) axis label: ``PTOExp3_LA0_I3`` -> ``PTO_LA0_I3``.

    Only drops the redundant constant ``Exp3`` (every model is Exp3); keeps method, look-ahead K,
    and iteration spelled out. Pooled ``Base`` -> ``Base``. Unknown strings pass through unchanged.
    """
    if model == "Base":
        return "Base"
    m = _MODEL_RE.match(model)
    if not m:
        return model
    method, k, tail = m.groups()
    return f"{method}_LA{k}_{tail}"


def relabel_xticks(ax, *, rotation: int = 90, fontsize: int = 7):
    """Re-label a categorical x-axis with :func:`clean_label`, pinning ticks first.

    Pinning the existing tick positions before relabeling avoids matplotlib's FixedLocator
    warning (set_ticklabels without set_ticks) and any label/tick drift.
    """
    ticks = ax.get_xticks()
    labels = [clean_label(t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rotation, fontsize=fontsize)


def add_base_line(ax, base_value, *, annotate: bool = True):
    """Draw a dotted horizontal reference at the (pooled) base score on a bar/point panel.

    Lets the reader see at a glance which models sit above vs below base. No-op if
    ``base_value`` is None/NaN.
    """
    if base_value is None or (isinstance(base_value, float) and np.isnan(base_value)):
        return
    ax.axhline(base_value, ls=":", lw=1.1, color="#555555", zorder=0.5)
    if annotate:
        ax.text(0.995, base_value, " base", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color="#555555")


def figure_legend_from(ax, fig, *, title="arm", ncol: int = 4):
    """Lift ``ax``'s legend to a single figure-level legend ABOVE the grid (out of the data).

    Reads the handles/labels off ``ax``, removes every per-axis legend it can see, and draws one
    shared legend so multi-panel figures don't repeat a key inside a data area. No-op if ``ax``
    has nothing to key.
    """
    handles, labels = ax.get_legend_handles_labels()
    for a in fig.axes:
        if a.get_legend() is not None:
            a.legend_.remove()
    if handles:
        fig.legend(handles, labels, title=title, loc="upper center",
                   bbox_to_anchor=(0.5, 1.04), ncol=ncol, frameon=False, fontsize=8)


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
