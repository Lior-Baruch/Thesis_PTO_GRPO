"""heterogeneity.py — persona-trait splits: per-arm grids, the all-metric overview, endpoint bars."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..constants import display_label, arm_label
from ..plotting_style import grid, arm_palette, relabel_legend, add_base_line
from ._shared import _metrics, _QUAL_COLORS

# Readable persona-trait category names (label layer; enum .name -> plain English).
_PERSONA_VALUE_LABELS = {
    "cooperation_level": {"Low": "Resistant", "High": "Cooperative",
                          "StartLowAndChangesToHigh": "Warms up"},
}


def _persona_cats(series, char):
    """Sorted distinct trait values + a {raw: readable} map for ``char``."""
    cats = sorted(x for x in series.dropna().unique())
    return cats, _PERSONA_VALUE_LABELS.get(char, {})


def heterogeneity_grid(scores_long, char: str, *, arms: Optional[Sequence[str]] = None,
                       metric: str = "Q1Q2", palette=None, ncols: int = 2):
    """ONE figure: *metric* across iterations split by persona ``char``, a panel per selected arm.

    Replaces the old ``2 × N`` ``heterogeneity_{char}_{arm}`` PNG explosion — pick the trait + the
    arms and get a single small-multiples grid. Trait categories use a colourblind qualitative
    palette + readable names (e.g. cooperation ``Low → Resistant``); one shared legend. Arms with <3
    scored iters (or missing ``char``) are skipped. Returns ``None`` if nothing is plottable.
    """
    if char not in scores_long.columns:
        return None
    d = scores_long[scores_long.questionnaire == metric]
    arm_list = [a for a in (arms if arms is not None else sorted(d.arm.unique()))
                if d[(d.arm == a)].iteration.nunique() >= 3 and d[d.arm == a][char].notna().any()]
    if not arm_list:
        return None
    cats, valmap = _persona_cats(d[char], char)
    hue_pal = {c: _QUAL_COLORS[i % len(_QUAL_COLORS)] for i, c in enumerate(cats)}
    fig, axes = grid(len(arm_list), ncols=ncols, panel=(6.4, 3.8))
    for ax, arm in zip(axes, arm_list):
        sns.lineplot(d[d.arm == arm], x="iteration", y="score", hue=char, hue_order=cats,
                     marker="o", palette=hue_pal, ax=ax)
        ax.set_title(arm_label(arm)); ax.set_ylabel(display_label(metric)); ax.set_xlabel("iteration")
        if ax is axes[0]:
            relabel_legend(ax, valmap)      # readable category names; palette keyed on raw
            if ax.get_legend():
                ax.legend_.set_title(char.replace("_", " "))
        elif ax.get_legend():
            ax.legend_.remove()
    fig.suptitle(f"{display_label(metric)} by {char.replace('_', ' ')} (true persona) — per arm",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def heterogeneity_overview_grid(scores_long, char: str, *, arms: Optional[Sequence[str]] = None,
                                metrics: Optional[Sequence[str]] = None, palette=None):
    """ALL metrics × ALL arms for ONE trait — rows = metric, cols = arm, hue = persona category.

    The combined "all-metrics overview per trait" sibling of the per-metric
    :func:`heterogeneity_grid` files (which fix a metric and panel over arms): here every rubric
    (rows) is shown across iterations split by persona ``char``, a column per arm, so the whole
    persona story for a trait reads at a glance — is the gap consistent across metrics, and do the
    arms diverge the same way? Same colourblind category palette + readable names as
    :func:`heterogeneity_grid`, one shared persona legend above the grid. ``palette`` is accepted
    for signature symmetry but unused (colour keys on persona, not arm). Arms with <3 scored iters
    (or missing ``char``) are dropped; returns ``None`` if nothing is plottable.
    """
    if char not in scores_long.columns:
        return None
    metrics = _metrics(scores_long["questionnaire"].unique(), metrics)
    arm_list = [a for a in (arms if arms is not None else sorted(scores_long.arm.unique()))
                if scores_long[scores_long.arm == a].iteration.nunique() >= 3
                and scores_long[scores_long.arm == a][char].notna().any()]
    if not arm_list or not metrics:
        return None
    cats, valmap = _persona_cats(scores_long[char], char)
    hue_pal = {c: _QUAL_COLORS[i % len(_QUAL_COLORS)] for i, c in enumerate(cats)}
    nrows, ncols = len(metrics), len(arm_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.7 * nrows),
                             squeeze=False, sharex=True)
    leg_ax = None
    for r, m in enumerate(metrics):
        dm = scores_long[scores_long.questionnaire == m]
        for c, arm in enumerate(arm_list):
            ax = axes[r][c]
            sns.lineplot(dm[dm.arm == arm], x="iteration", y="score", hue=char, hue_order=cats,
                         marker="o", palette=hue_pal, ax=ax)
            if r == 0:
                ax.set_title(arm_label(arm))
            ax.set_ylabel(display_label(m) if c == 0 else "")
            ax.set_xlabel("iteration" if r == nrows - 1 else "")
            if ax.get_legend():
                if leg_ax is None:
                    leg_ax = ax                 # keep the first to lift into a shared legend
                else:
                    ax.legend_.remove()
    if leg_ax is not None and leg_ax.get_legend():
        handles, labels = leg_ax.get_legend_handles_labels()
        leg_ax.legend_.remove()
        fig.legend(handles, [valmap.get(l, l) for l in labels], title=char.replace("_", " "),
                   loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=max(1, len(cats)),
                   frameon=False, fontsize=8)
    fig.suptitle(f"All metrics by {char.replace('_', ' ')} (true persona) — rows = metric, cols = arm",
                 y=1.05, fontweight="bold")
    fig.tight_layout()
    return fig


def subgroup_endpoint_bars(scores_long, char: str, *, arms: Optional[Sequence[str]] = None,
                           metric: str = "Q1Q2", palette=None):
    """Final-iteration *metric* per (persona ``char`` × arm) — 'where does an arm win / regress?'

    Grouped bars: x = persona category (readable), hue = arm, y = mean **final-iteration** score over
    that subgroup (95% CI), with a dotted pooled-base reference. The single-glance companion to
    :func:`heterogeneity_grid` — e.g. GRPO's late regression concentrated on the *Resistant*
    (Low-cooperation) personas. ``None`` if nothing is plottable.
    """
    if char not in scores_long.columns:
        return None
    d = scores_long[scores_long.questionnaire == metric]
    arm_list = [a for a in (arms if arms is not None else sorted(d.arm.unique()))
                if d[(d.arm == a)].iteration.nunique() >= 3 and d[d.arm == a][char].notna().any()]
    if not arm_list:
        return None
    parts = []
    for a in arm_list:
        da = d[d.arm == a]
        parts.append(da[da.iteration == int(da.iteration.max())])
    fin = pd.concat(parts, ignore_index=True)
    fin = fin[fin[char].notna()].copy()
    if fin.empty:
        return None
    cats, valmap = _persona_cats(d[char], char)
    fin["cat"] = fin[char].map(lambda v: valmap.get(v, v))
    fin["arm_disp"] = fin["arm"].map(arm_label)
    cat_order = [valmap.get(c, c) for c in cats]
    pal = palette or arm_palette(arm_list)
    pal_disp = {arm_label(a): pal.get(a, "#777777") for a in arm_list}
    fig, ax = plt.subplots(figsize=(1.7 * max(3, len(cat_order)) + 2, 4.4))
    sns.barplot(fin, x="cat", y="score", hue="arm_disp", order=cat_order,
                hue_order=[arm_label(a) for a in arm_list], palette=pal_disp,
                errorbar=("ci", 95), ax=ax)
    add_base_line(ax, float(d[d.is_base].score.mean()) if d.is_base.any() else None)
    ax.set_title(f"Final-iteration {display_label(metric)} by {char.replace('_', ' ')} "
                 f"(per arm; dotted = base)")
    ax.set_xlabel(""); ax.set_ylabel(display_label(metric))
    ax.legend(title="arm", fontsize=8)
    fig.tight_layout()
    return fig
