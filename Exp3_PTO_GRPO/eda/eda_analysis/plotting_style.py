"""
plotting_style.py — the shared figure-STYLE layer (helpers only, no named figures).

Split out of ``plotting.py`` (2026-07-08) so the style/scaffold concern lives in its own ~200-line
file and ``plotting.py`` holds just the named recurring figures. Everything here is re-imported into
``plotting`` (``from .plotting_style import …``), so ``plotting``/``figures``/``plots`` still expose
these helpers unchanged — notebook calls like ``figures.set_style(...)`` / ``figures.grid(...)`` and
the figures' own ``figures.arm_palette(...)`` self-calls keep resolving.

Contents: the publication style (:func:`set_style`), the colourblind arm palette
(:func:`arm_palette`), the score-axis limiter (:func:`apply_score_axis`), left-to-right
:func:`model_order`, readable labels (:func:`clean_label`/:func:`relabel_xticks`/
:func:`relabel_legend`), a dotted base-line (:func:`add_base_line`), a shared figure legend
(:func:`figure_legend_from`), and the :func:`grid` subplot scaffold. The module-level ``_SCALE``
defaults are updated by ``set_style(cfg)`` so an ``EdaConfig``'s panel/ncols/score_ylim propagate.
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
    "Base": "#555555",      # the pooled descriptive base (see data.collapse_base)
}


# Plot-scale defaults read by grid() / plot functions when their args are omitted. Updated by
# set_style(cfg) so an EdaConfig's panel/ncols/score_ylim/share_y propagate everywhere. Defaults
# match the pre-refactor behaviour (grid ncols=3, panel=(5.0,3.2), no y-limit).
_SCALE = {"panel": (5.0, 3.2), "ncols": 3, "score_ylim": None, "share_y": False,
          "palette_overrides": {}}


def set_style(cfg=None):
    """Consistent, publication-grade global style for every Exp3 figure.

    Whitegrid theme + tight, vector-friendly save defaults so `exports.save_fig` produces clean
    PDF (editable text via fonttype 42). When an ``EdaConfig`` is passed, its ``context``,
    ``font_scale``, ``dpi``, ``savefig_dpi`` are applied and its ``panel``/``ncols``/``score_ylim``/
    ``share_y``/``palette_overrides`` become the module-level defaults used by :func:`grid`,
    :func:`apply_score_axis`, and :func:`arm_palette`.
    """
    context = getattr(cfg, "context", "notebook") or "notebook"
    font_scale = getattr(cfg, "font_scale", 1.0) or 1.0
    dpi = getattr(cfg, "dpi", 110) or 110
    savefig_dpi = getattr(cfg, "savefig_dpi", 200) or 200
    sns.set_theme(style="whitegrid", context=context, font_scale=font_scale)
    plt.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": savefig_dpi,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable/embeddable text in vector output
        "figure.autolayout": False,
    })
    if cfg is not None:
        # Only override when the cfg sets a value (None = inherit the pre-refactor default).
        if getattr(cfg, "panel", None) is not None:
            _SCALE["panel"] = tuple(cfg.panel)
        if getattr(cfg, "ncols", None) is not None:
            _SCALE["ncols"] = int(cfg.ncols)
        _SCALE["score_ylim"] = getattr(cfg, "score_ylim", None)
        _SCALE["share_y"] = bool(getattr(cfg, "share_y", False))
        _SCALE["palette_overrides"] = dict(getattr(cfg, "palette_overrides", {}) or {})


def arm_palette(labels: Sequence[str]) -> dict:
    """Stable ``{arm_label: color}`` (cfg overrides > known Okabe-Ito > tab10 fallback)."""
    pal = {l: _ARM_COLORS.get(l) for l in labels}
    missing = [l for l in labels if pal[l] is None]
    for l, c in zip(missing, sns.color_palette("tab10", len(missing)).as_hex()):
        pal[l] = c
    pal.update({l: c for l, c in _SCALE["palette_overrides"].items() if l in pal})
    return pal


def apply_score_axis(ax, *, ylim=None, metric: str = ""):
    """Apply the configured score y-limits to ``ax`` (no-op if neither cfg nor arg sets them).

    ``ylim`` (arg) wins over the module default from ``set_style(cfg)``. Skipped for proportion /
    rate metrics whose natural range differs from the 1–5 rubric scale.
    """
    lim = ylim if ylim is not None else _SCALE.get("score_ylim")
    if lim is None:
        return
    if metric in {"PCT", "MICI", "R:Q", "%CR", "%MICO"}:   # different natural scale
        return
    ax.set_ylim(*lim)


def model_order(scores_long) -> List[str]:
    """Models left-to-right by (method, K, iteration) — for stable bar/x ordering.

    The pooled descriptive ``Base`` (``data.collapse_base``: method ``"Base"``, K ``-1``) always
    sorts first.
    """
    meta = (scores_long[["model", "method", "K", "iteration"]]
            .drop_duplicates().sort_values(["method", "K", "iteration"]))
    order = meta["model"].tolist()
    if "Base" in order:  # guarantee the pooled base leads regardless of sort keys
        order = ["Base"] + [m for m in order if m != "Base"]
    return order


_MODEL_RE = re.compile(r"^(PTO|GRPO)Exp3_LA(\d+)_(Base|I\d+)$")


def clean_label(model: str) -> str:
    """Tidy, readable axis label: ``PTOExp3_LA0_I3`` -> ``PTO K=0 I3``.

    Drops the redundant constant ``Exp3`` (every model is Exp3) and spells the look-ahead as
    ``K=<k>``; keeps the iteration tag (``I3`` / ``Base``). Pooled ``Base`` -> ``Base``. Unknown
    strings pass through unchanged.
    """
    if model == "Base":
        return "Base"
    m = _MODEL_RE.match(model)
    if not m:
        return model
    method, k, tail = m.groups()
    return f"{method} K={k} {tail}"


def relabel_xticks(ax, *, rotation: int = 90, fontsize: int = 7):
    """Re-label a categorical x-axis with :func:`clean_label`, pinning ticks first.

    Pinning the existing tick positions before relabeling avoids matplotlib's FixedLocator
    warning (set_ticklabels without set_ticks) and any label/tick drift.
    """
    ticks = ax.get_xticks()
    labels = [clean_label(t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rotation, fontsize=fontsize)


def relabel_legend(ax, mapping=None):
    """Rewrite an axis legend's entry TEXTS (default: readable :func:`arm_label`).

    Label layer only: the hue column keeps its canonical values (so the palette mapping stays
    intact); only the visible legend text is swapped. No-op if ``ax`` has no legend.
    """
    from . import arm_label
    leg = ax.get_legend()
    if leg is None:
        return
    fn = (lambda t: mapping.get(t, t)) if mapping is not None else arm_label
    for txt in leg.get_texts():
        txt.set_text(fn(txt.get_text()))


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
    from . import arm_label
    handles, labels = ax.get_legend_handles_labels()
    labels = [arm_label(l) for l in labels]   # readable arm names (canonical keys unchanged)
    for a in fig.axes:
        if a.get_legend() is not None:
            a.legend_.remove()
    if handles:
        fig.legend(handles, labels, title=title, loc="upper center",
                   bbox_to_anchor=(0.5, 1.04), ncol=ncol, frameon=False, fontsize=8)


def grid(n: int, ncols: int = None, panel=None):
    """A ready (fig, axes_flat) grid sized for *n* panels; trailing axes hidden.

    ``ncols``/``panel`` default to the values set by ``set_style(cfg)`` (the EdaConfig scales),
    so a notebook that sets ``ncols=3, panel=(6,4)`` in cell 1 gets it everywhere.

    Usage in a notebook:
        fig, axes = figures.grid(len(METRICS))
        for ax, m in zip(axes, METRICS):
            sns.lineplot(..., ax=ax)
    """
    ncols = _SCALE["ncols"] if ncols is None else ncols
    panel = _SCALE["panel"] if panel is None else panel
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False)
    axes_flat = axes.flat
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)
