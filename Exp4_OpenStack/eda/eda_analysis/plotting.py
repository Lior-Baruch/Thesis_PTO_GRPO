"""plotting.py -- the figure layer: one style, one arm palette, four reusable figures.

Two rules shape this module.

**Figures are returned, never saved.** Nothing here touches the filesystem; ``exports.save_fig``
owns paths, formats and DPI. That boundary is not tidiness -- it is what lets the same figure be
rendered at screen DPI in a notebook and at print DPI into ``results/`` without a second code path,
and what keeps a family notebook from being able to write outside its own results folder. A builder
that saved would also have to know the judge, the family and the artifact name, i.e. it would have
to import the export layer, and the export layer already imports nothing.

**An arm is the same colour in every artifact.** :func:`arm_color` maps an arm *label* to a colour
deterministically, without consulting the set of labels in the current figure. Exp3's palette
assigned fallback colours by position in the caller's list, so an arm's colour depended on which
other arms happened to be in that panel -- and a reader comparing two figures side by side was
silently comparing two different colour schemes. Here ``GRPO_LA5`` is the same orange in the
trajectory panel, the distribution panel and the cost scatter, whether or not the other three arms
are present.

Everything is deliberately generic about column names. The families own the frame shapes; the
figure builders take ``x=`` / ``y=`` / ``arm_col=`` so a new family does not need a new figure. The
only package coupling is :data:`constants.BOOT_SEED` and :func:`stats.higher_is_better` -- the
latter so a forest plot can colour "better" correctly on ``MICI``, where better means lower.

Reproducibility: every seaborn call that draws a bootstrap error bar passes ``seed=BOOT_SEED``.
Seaborn's ``errorbar=("ci", 95)`` defaults to ``seed=None``, i.e. a fresh 1,000-sample bootstrap per
call; left unset, three renders of the same notebook on identical data differ by a few percent of
pixels and every tracked PNG churns in git.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib


def _ensure_headless_backend() -> None:
    """Select the non-interactive Agg backend when there is no display to draw on.

    ``tools/render_results.py`` executes the family notebooks headlessly, often over SSH or in a
    scheduled shell. An interactive backend there either fails at import or opens a window that
    nothing ever closes, and the failure arrives halfway through a render rather than at import.

    Windows and macOS always have a window server, so their default backends are left alone; on
    Linux the decision is ``DISPLAY``/``WAYLAND_DISPLAY``. Must run BEFORE ``matplotlib.pyplot`` is
    imported -- switching afterwards requires ``force=True`` and can strand already-created
    figures.
    """
    if matplotlib.get_backend().lower() == "agg":
        return
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg", force=True)


_ensure_headless_backend()

import matplotlib.pyplot as plt          # noqa: E402  (backend must be chosen first)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import seaborn as sns                    # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.figure import Figure     # noqa: E402
from matplotlib.patches import Patch     # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .constants import BOOT_SEED         # noqa: E402
from .stats import higher_is_better      # noqa: E402

__all__ = [
    "ARM_COLORS",
    "set_style",
    "arm_color",
    "arm_palette",
    "grid",
    "add_base_line",
    "legend_outside",
    "score_trajectory",
    "arm_distribution",
    "contrast_forest",
    "cost_benefit",
]


# ==============================================================================
#  Style
# ==============================================================================

#: Defaults read by :func:`grid` and the figure builders when their arguments are omitted.
#: :func:`set_style` overwrites them from an ``EdaConfig`` so a notebook's cell 1 propagates
#: everywhere without threading the config through every call.
_SCALE: Dict[str, Any] = {
    "panel": (5.2, 3.3),
    "ncols": 3,
    "score_ylim": None,
    "palette_overrides": {},
}


def set_style(cfg: Any = None) -> None:
    """Apply the publication style, and adopt *cfg*'s scales as the module defaults.

    Args:
        cfg: An ``EdaConfig`` (or anything with the same attribute names); ``None`` applies the
            style with built-in defaults. Read attributes: ``context``, ``font_scale``, ``dpi``,
            ``savefig_dpi``, ``panel``, ``ncols``, ``score_ylim``, ``palette_overrides``. Missing
            attributes keep their defaults, so a partial config object is fine.

    Notes:
        ``pdf.fonttype``/``ps.fonttype`` are 42 so vector output carries editable, embeddable text
        -- a thesis figure gets opened in a vector editor eventually, and type-3 outlines cannot be
        re-typeset.

        This is global matplotlib/seaborn state. Call it once, from ``notebook_setup``; calling it
        mid-notebook restyles only figures created afterwards, which is a good way to ship one
        panel that does not match its neighbours.
    """
    context = getattr(cfg, "context", None) or "notebook"
    font_scale = getattr(cfg, "font_scale", None) or 1.0
    dpi = getattr(cfg, "dpi", None) or 110
    savefig_dpi = getattr(cfg, "savefig_dpi", None) or 200

    sns.set_theme(style="whitegrid", context=context, font_scale=float(font_scale))
    plt.rcParams.update({
        "figure.dpi": float(dpi),
        "savefig.dpi": float(savefig_dpi),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.autolayout": False,
    })

    if cfg is None:
        return
    if getattr(cfg, "panel", None) is not None:
        _SCALE["panel"] = tuple(cfg.panel)
    if getattr(cfg, "ncols", None) is not None:
        _SCALE["ncols"] = int(cfg.ncols)
    _SCALE["score_ylim"] = getattr(cfg, "score_ylim", None)
    _SCALE["palette_overrides"] = dict(getattr(cfg, "palette_overrides", None) or {})


# ==============================================================================
#  Colour
# ==============================================================================

#: Pinned colours for the canonical grid. Okabe-Ito, grouped by TEMPERATURE so the method reads at
#: a glance (PTO cool, GRPO warm) while the two look-ahead arms within a method stay distinct.
#: ``Base`` is the untrained policy (``model_iter_0``) wherever a family pools it as its own series.
ARM_COLORS: Dict[str, str] = {
    "PTO_LA0": "#0072B2",    # blue
    "PTO_LA5": "#56B4E9",    # sky blue
    "GRPO_LA0": "#D55E00",   # vermillion
    "GRPO_LA5": "#E69F00",   # orange
    "Base": "#555555",       # neutral grey
}

_METHOD_BASE = {"PTO": "#0072B2", "GRPO": "#D55E00"}

#: Display-layer parse of an arm label as ``naming.ArmInfo.label`` spells it:
#: ``GRPO_LA5``, ``PTO_LA0_indep``, ``GRPO_LA0_Ogpt4m``. This duplicates no grammar that matters --
#: the label is already a lossy display key (it drops MCL, the branch width and the training
#: rubric), so it is never a data key and this regex never decides what is loaded.
_LABEL_RE = re.compile(r"^(GRPO|PTO)_LA(\d+)(?:_(.+))?$")

#: Colours for labels that are not arms at all (a judge, a rubric, a persona stratum used as a
#: series). Okabe-Ito remainder, then a few tab10 entries.
_FALLBACK_COLORS = ("#009E73", "#CC79A7", "#F0E442", "#8C564B", "#7F7F7F", "#17BECF")


def _lighten(color: str, amount: float) -> str:
    """Mix *color* toward white by *amount* in [0, 1]; 0 returns it unchanged."""
    r, g, b = mcolors.to_rgb(color)
    a = min(max(float(amount), 0.0), 1.0)
    return mcolors.to_hex((r + (1.0 - r) * a, g + (1.0 - g) * a, b + (1.0 - b) * a))


def _stable_index(text: str, n: int) -> int:
    """Deterministic index into an *n*-element table, stable across processes.

    ``hash()`` is salted per interpreter run (PYTHONHASHSEED), so using it here would give an arm a
    different colour on every render -- exactly the churn this module exists to avoid.
    """
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % max(int(n), 1)


def arm_color(label: str) -> str:
    """The colour for one arm label -- the same colour in every figure, forever.

    Resolution order:

    1. ``palette_overrides`` from the active ``EdaConfig`` (see :func:`set_style`);
    2. :data:`ARM_COLORS`, the pinned canonical grid;
    3. a derived shade for any other ``{METHOD}_LA{K}[_extra]`` label -- the method sets the hue,
       the look-ahead depth sets how far it is lightened, and a suffix (a preference-tree mode, a
       swapped role) shifts it further by a stable digest of the label;
    4. a fallback colour picked by digest for a label that is not an arm at all.

    Warning:
        Step 3's lightness is bounded, so two exotic labels at the same K CAN land close together;
        step 4 can collide outright. Both are display-layer approximations for labels outside the
        planned grid -- if an arm becomes a regular part of the experiment, pin it in
        :data:`ARM_COLORS` rather than relying on the derivation.
    """
    key = str(label)
    override = _SCALE.get("palette_overrides") or {}
    if key in override:
        return str(override[key])
    if key in ARM_COLORS:
        return ARM_COLORS[key]

    match = _LABEL_RE.match(key)
    if match:
        method, k_text, extra = match.groups()
        amount = min(0.11 * int(k_text), 0.55)
        if extra:
            amount = min(amount + 0.08 + 0.08 * _stable_index(key, 4), 0.72)
        return _lighten(_METHOD_BASE[method], amount)

    return _FALLBACK_COLORS[_stable_index(key, len(_FALLBACK_COLORS))]


def arm_palette(labels: Iterable[str]) -> Dict[str, str]:
    """``{label: colour}`` for *labels*, via :func:`arm_color`.

    The result depends only on the labels themselves, never on their order or on which other
    labels are present -- pass a subset and the colours are unchanged. Hand it straight to
    seaborn's ``palette=``.
    """
    return {str(l): arm_color(l) for l in labels}


# ==============================================================================
#  Scaffolding
# ==============================================================================


def grid(n: int, ncols: Optional[int] = None, panel: Optional[Tuple[float, float]] = None):
    """A ``(fig, axes)`` grid sized for *n* panels, with trailing axes hidden.

    Args:
        n: Number of panels that will be drawn.
        ncols: Columns; defaults to the value from :func:`set_style`.
        panel: ``(width, height)`` inches per panel; defaults likewise.

    Returns:
        ``(fig, axes)`` where ``axes`` is a flat list of length ``nrows * ncols``. Only the first
        *n* are visible, so ``zip(axes, items)`` is safe and the layout stays rectangular.
    """
    ncols = int(_SCALE["ncols"] if ncols is None else ncols)
    panel = tuple(_SCALE["panel"] if panel is None else panel)
    n = max(int(n), 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def add_base_line(ax, value: Optional[float], *, label: str = "base",
                  annotate: bool = True, color: str = "#555555"):
    """Dotted horizontal reference at the untrained policy's score.

    Every trajectory answers "did training help?", and that question needs the ``model_iter_0``
    level visible in the panel rather than in the caller's memory. No-op for ``None``/NaN, so a
    family can pass a base that may not have been scored yet.
    """
    if value is None:
        return ax
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ax
    if v != v:                                    # NaN
        return ax
    ax.axhline(v, ls=":", lw=1.1, color=color, zorder=0.5)
    if annotate and label:
        ax.text(0.995, v, f" {label}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color=color)
    return ax


def legend_outside(ax, *, title: str = "arm", loc: str = "upper left",
                   bbox: Tuple[float, float] = (1.01, 1.0)):
    """Move an axis legend outside the data area (no-op when there is nothing to key)."""
    if ax.get_legend() is None:
        return ax
    sns.move_legend(ax, loc, bbox_to_anchor=bbox, title=title, frameon=False)
    return ax


def _select(df: pd.DataFrame, *, metric: Optional[str], metric_col: str,
            arm_col: Optional[str], arms: Optional[Sequence[str]]) -> pd.DataFrame:
    """Apply the two filters every builder shares: one metric, an optional arm subset.

    Raises:
        KeyError: when a metric was requested but *metric_col* is not in the frame. Skipping the
            filter instead would plot every rubric pooled onto one axis under the requested
            metric's title -- a wrong figure that renders cleanly, which is worse than a stop.
    """
    out = df
    if metric is not None:
        if metric_col not in out.columns:
            raise KeyError(
                f"metric={metric!r} was requested but the frame has no {metric_col!r} column "
                f"(it has {list(out.columns)}). Pass metric_col= or pre-filter the frame."
            )
        out = out[out[metric_col] == metric]
    if arms is not None and arm_col and arm_col in out.columns:
        out = out[out[arm_col].isin(list(arms))]
    return out


def _num(value: Any) -> float:
    """Best-effort float; ``nan`` for None or anything unconvertible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _require(df: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    """Fail loudly on a missing column instead of drawing an empty panel.

    A figure builder that quietly plots nothing is the worst outcome available: the render
    succeeds, an empty PNG lands in ``results/``, and the index links to it.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: frame has no column(s) {missing}; it has {list(df.columns)}")


def _finish(fig: Figure) -> Figure:
    fig.tight_layout()
    return fig


# ==============================================================================
#  Figures
# ==============================================================================


def score_trajectory(scores: pd.DataFrame,
                     *,
                     metric: Optional[str] = None,
                     metric_col: str = "metric",
                     x: str = "iteration",
                     y: str = "score",
                     arm_col: str = "arm",
                     arms: Optional[Sequence[str]] = None,
                     palette: Optional[Mapping[str, str]] = None,
                     base_value: Optional[float] = None,
                     title: Optional[str] = None,
                     xlabel: str = "model state (iteration)",
                     ylabel: Optional[str] = None,
                     errorbar=("ci", 95),
                     ax=None,
                     figsize: Tuple[float, float] = (8.0, 4.5)) -> Figure:
    """Mean score per model state with a 95% CI band, arms overlaid -- the learning curve.

    Args:
        scores: Long frame with one row per (arm, model state, persona, metric).
        metric: Filter to this metric first; ``None`` assumes *scores* is already one metric.
        metric_col: Where the metric key lives (the score lake's ``metric=<M>`` level).
        x: Model-state column. ``model_iter_<N>`` is labelled by the GENERATING policy, so
            ``iteration`` 0 is the untrained base and the curve starts there.
        y: Score column.
        arm_col: Series column; each value gets its stable :func:`arm_color`.
        arms: Optional subset, in case a family wants the K contrast alone.
        palette: Override the arm palette entirely.
        base_value: Draw a dotted reference line (see :func:`add_base_line`).
        errorbar: Passed to seaborn; ``None`` suppresses the band.
        ax: Draw into an existing axis (for a grid); a new figure is made when omitted.

    Returns:
        The ``Figure``. Not saved -- hand it to ``exports.save_fig``.

    Notes:
        The CI band is a bootstrap over PERSONAS within a model state, seeded with
        :data:`constants.BOOT_SEED`. It describes the spread across the 96 clients, not the
        uncertainty of a *paired* arm difference -- two bands can overlap while the paired contrast
        between them is decisive, because pairing removes the persona variance the bands show. Read
        differences off :func:`contrast_forest`, never off the overlap of two bands.
    """
    data = _select(scores, metric=metric, metric_col=metric_col, arm_col=arm_col, arms=arms)
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    labels = sorted(data[arm_col].astype(str).unique()) if arm_col in data.columns else []
    pal = dict(palette) if palette is not None else arm_palette(labels)

    sns.lineplot(data=data, x=x, y=y, hue=arm_col if arm_col in data.columns else None,
                 palette=pal if labels else None, hue_order=labels or None,
                 marker="o", errorbar=errorbar, seed=BOOT_SEED, ax=ax)

    add_base_line(ax, base_value)
    if _SCALE.get("score_ylim") is not None:
        ax.set_ylim(*_SCALE["score_ylim"])
    # Model states are whole numbers; without this the auto-locator offers "1.5", which reads as a
    # checkpoint that does not exist.
    if x in data.columns and pd.api.types.is_numeric_dtype(data[x]):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or (metric or y))
    ax.set_title(title or (f"{metric} across model states" if metric else "Score across model states"))
    legend_outside(ax)
    return _finish(fig)


def arm_distribution(scores: pd.DataFrame,
                     *,
                     metric: Optional[str] = None,
                     metric_col: str = "metric",
                     value: str = "score",
                     arm_col: str = "arm",
                     arms: Optional[Sequence[str]] = None,
                     palette: Optional[Mapping[str, str]] = None,
                     kind: str = "box",
                     show_points: bool = True,
                     show_mean: bool = True,
                     title: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     ax=None,
                     figsize: Optional[Tuple[float, float]] = None) -> Figure:
    """Per-arm score distribution across personas -- the spread behind every reported mean.

    Args:
        scores: Long frame, usually filtered to one model state (an endpoint comparison).
        metric / metric_col: Optional metric filter, as in :func:`score_trajectory`.
        value: Score column.
        arm_col: Category column; arms are drawn in sorted order.
        kind: ``"box"`` or ``"violin"``.
        show_points: Overlay the individual persona scores (alpha-blended strip).
        show_mean: Overlay the mean with a bootstrap 95% CI (seeded with
            :data:`constants.BOOT_SEED`). The box shows the MEDIAN, which is not the statistic any
            table in this project reports -- drawing the mean on top keeps the figure and the table
            describing the same quantity.

    Returns:
        The ``Figure``.

    Warning:
        These distributions are NOT independent samples of different things: the same 96 personas
        appear in every box. Two boxes overlapping heavily is compatible with a large, consistent
        paired difference (every persona moved the same direction by a little). Use this figure to
        show spread and outliers; use a paired contrast to claim a difference.
    """
    data = _select(scores, metric=metric, metric_col=metric_col, arm_col=arm_col, arms=arms)
    _require(data, [arm_col, value], "arm_distribution")
    order = sorted(data[arm_col].astype(str).unique())
    pal = dict(palette) if palette is not None else arm_palette(order)

    if figsize is None:
        figsize = (max(4.5, 1.35 * max(len(order), 1) + 1.6), 4.2)
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    common = dict(data=data, x=arm_col, y=value, order=order, hue=arm_col, hue_order=order,
                  palette=pal, legend=False, ax=ax)
    if kind == "violin":
        sns.violinplot(cut=0, inner="quartile", **common)
    else:
        sns.boxplot(showfliers=not show_points, width=0.62, **common)

    if show_points:
        sns.stripplot(data=data, x=arm_col, y=value, order=order, color="#333333",
                      size=2.6, alpha=0.35, jitter=0.18, ax=ax)
    if show_mean:
        sns.pointplot(data=data, x=arm_col, y=value, order=order, color="#111111",
                      errorbar=("ci", 95), seed=BOOT_SEED, linestyle="none",
                      markers="D", markersize=4, capsize=0.12, ax=ax)

    if _SCALE.get("score_ylim") is not None:
        ax.set_ylim(*_SCALE["score_ylim"])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or (metric or value))
    ax.set_title(title or (f"{metric} by arm" if metric else "Score by arm"))
    return _finish(fig)


def contrast_forest(rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
                    *,
                    label_col: str = "label",
                    value_col: str = "mean_delta",
                    lo_col: str = "ci_lo",
                    hi_col: str = "ci_hi",
                    annot_col: Optional[str] = "dz",
                    metric_col: Optional[str] = None,
                    title: Optional[str] = None,
                    xlabel: str = "paired difference (95% bootstrap CI)",
                    figsize: Optional[Tuple[float, float]] = None) -> Optional[Figure]:
    """Forest plot of paired contrasts: one dot + CI whisker per row, zero line marked.

    Args:
        rows: :func:`stats.paired_contrast` output rows (usually via
            :func:`stats.summarize_contrasts`), each carrying a descriptive *label_col*.
        label_col: Row label, e.g. ``"PTO_LA5 - PTO_LA0 @ Q1Q2"``.
        value_col / lo_col / hi_col: The estimate and its interval.
        annot_col: Printed to the right of each whisker; ``None`` to suppress.
        metric_col: When given, each row's metric decides which DIRECTION is good, via
            :func:`stats.higher_is_better`. Without it every row is assumed higher-is-better.
        figsize: Defaults to a height that grows with the row count.

    Returns:
        The ``Figure``, or ``None`` for an empty input (so a caller can skip the export).

    Warning:
        On ``MICI`` a positive difference is a WORSE therapist. Pass ``metric_col`` whenever the
        frame can contain a lower-is-better metric -- otherwise the colouring says "improved" for
        exactly the rows that got worse. The x-axis always shows the RAW difference; only the
        colour is oriented, so the plotted number still matches the table.
    """
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df is None or df.empty:
        return None

    records = list(df.to_dict("records"))[::-1]        # first row at the top (y grows upward)
    if figsize is None:
        figsize = (7.6, max(2.6, 0.36 * len(records) + 1.4))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    saw_worse = False
    for i, r in enumerate(records):
        value = _num(r.get(value_col))
        lo, hi = _num(r.get(lo_col)), _num(r.get(hi_col))
        have_ci = (lo == lo) and (hi == hi)                 # both non-NaN
        sign = 1
        if metric_col and r.get(metric_col) is not None:
            sign = 1 if higher_is_better(str(r[metric_col]), default=True) else -1
        # An unresolved interval is drawn as "not distinguishable from zero" rather than as a
        # direction, so a missing CI can never be read as evidence.
        if not have_ci or lo <= 0.0 <= hi:
            color = "#9e9e9e"
        elif sign * value > 0:
            color = "#2ca02c"
        else:
            color = "#D55E00"
            saw_worse = True
        if have_ci:
            ax.plot([lo, hi], [i, i], color=color, lw=2.4, solid_capstyle="round", zorder=2)
        if value == value:
            ax.scatter([value], [i], color=color, s=44, zorder=3)
        annot = _num(r.get(annot_col)) if annot_col else float("nan")
        anchor = max(v for v in (hi, value) if v == v) if (have_ci or value == value) else None
        if annot == annot and anchor is not None:
            ax.text(anchor, i, f"  {annot_col}={annot:.2f}", va="center",
                    fontsize=6.5, color="#333333")

    ax.axvline(0.0, color="#555555", lw=1.0, ls="--")
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([str(r.get(label_col, "")) for r in records], fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title or "Paired contrasts (dashed = no difference)")
    handles = [Patch(color="#2ca02c", label="better (CI excludes 0)"),
               Patch(color="#9e9e9e", label="CI includes 0")]
    if saw_worse:
        handles.insert(1, Patch(color="#D55E00", label="worse (CI excludes 0)"))
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
    return _finish(fig)


def cost_benefit(points: pd.DataFrame,
                 *,
                 x: str = "gpu_hours",
                 y: str = "score",
                 arm_col: str = "arm",
                 label_col: Optional[str] = "iteration",
                 palette: Optional[Mapping[str, str]] = None,
                 connect: bool = True,
                 annotate: bool = True,
                 title: Optional[str] = None,
                 xlabel: str = "cumulative GPU-hours",
                 ylabel: Optional[str] = None,
                 ax=None,
                 figsize: Tuple[float, float] = (7.2, 4.6)) -> Figure:
    """Score against spend -- what each arm bought per GPU-hour.

    Args:
        points: One row per (arm, model state): its cumulative cost and its mean score.
            ``compute``-family input, built from ``core/timing.py``'s session logs.
        x: Cost column (cumulative GPU-hours; ``eda/`` reads them from
            ``iteration_N/timing_sessions.jsonl``, never from directory mtimes).
        y: Score column (a per-arm, per-state MEAN, not a per-conversation row).
        arm_col: Series column; stable :func:`arm_color` per arm.
        label_col: Annotate each marker with this (usually the iteration index); ``None`` to
            suppress.
        connect: Join each arm's points in cost order, so the trajectory reads as a path through
            the budget rather than a cloud.

    Returns:
        The ``Figure``.

    Notes:
        This is the only view in the EDA that is indexed by SPEND rather than by iteration. Every
        other contrast is per-iteration, and an iteration is not a fixed unit of cost: a K=5 step
        costs roughly twice a K=0 step, and PTO's preference build is a whole phase GRPO does not
        have. Two arms compared at the same iteration are therefore NOT compared at the same price,
        and a lever's sign can differ between the two readings.

    Warning:
        Read a lever off the whole curve, not off one crossing. Which arm is ahead is a function of
        budget: an arm can trail badly at a small budget and draw level later, so quoting the pair
        at a single spend is choosing the answer.
    """
    data = points
    _require(data, [arm_col, x, y], "cost_benefit")
    order = sorted(data[arm_col].astype(str).unique())
    pal = dict(palette) if palette is not None else arm_palette(order)

    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    for arm in order:
        g = data[data[arm_col].astype(str) == arm].sort_values(x, kind="mergesort")
        color = pal.get(arm, "#555555")
        if connect and len(g) > 1:
            ax.plot(g[x], g[y], color=color, lw=1.4, alpha=0.85, zorder=2)
        ax.scatter(g[x], g[y], color=color, s=46, label=arm, zorder=3)
        if annotate and label_col and label_col in g.columns:
            for _, row in g.iterrows():
                ax.annotate(str(row[label_col]), xy=(row[x], row[y]), xytext=(4, 4),
                            textcoords="offset points", fontsize=6.5, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or "Score vs compute spend")
    if order:
        ax.legend(title="arm", fontsize=8, frameon=False, loc="best")
    return _finish(fig)
