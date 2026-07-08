"""
plotting.py — the figure layer: the named recurring plot functions.

The style/scaffold helpers were split out into :mod:`eda_analysis.plotting_style` (2026-07-08); this
file holds just the figures that recur across notebooks (outcomes, trajectories, faithfulness,
behavior, training internals), defined ONCE and called from multiple notebooks. Genuinely one-off
exploration still lives inline in the notebooks.

Contract for every named-plot function: takes an already-built tidy frame (never touches disk),
returns a matplotlib ``fig`` (no ``plt.show()`` / ``save_fig`` — the notebook owns those), reuses the
``plotting_style`` helpers, and degrades gracefully on thin/absent arms (returns ``None`` or an empty
panel).

Both ``eda_analysis.figures`` and ``eda_analysis.plots`` are aliased to THIS module in
``__init__.py``, so existing ``figures.set_style(...)`` / ``plots.trajectory_grid(...)`` notebook
calls keep working — the style helpers are re-imported below, so ``figures.grid(...)`` etc. resolve
here too.
"""

import sys
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import (
    QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, ORTHOGONAL_METRICS, LOWER_IS_BETTER,
    display_label, short_label, arm_label,
)
# Style/scaffold helpers now live in plotting_style; re-import them so this module (and its
# ``figures``/``plots`` aliases) still exposes set_style/arm_palette/grid/... and the figures' own
# ``figures.grid(...)`` self-calls resolve here.
from .plotting_style import (  # noqa: F401
    set_style, arm_palette, apply_score_axis, model_order, clean_label,
    relabel_xticks, relabel_legend, add_base_line, figure_legend_from, grid,
)

# Self-alias: the named-plot functions below call ``figures.grid(...)`` / ``figures.arm_palette(...)``
# etc. — ``figures`` points at THIS module, where those helpers are now imported.
figures = sys.modules[__name__]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NAMED PLOTS — the figures that recur across notebooks                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _metrics(frame_metrics, metrics: Optional[Sequence[str]]) -> list:
    present = set(frame_metrics)
    return [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in present]


# ── Outcomes ─────────────────────────────────────────────────────────────────
def outcomes_by_model(scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                      order: Optional[Sequence[str]] = None, ncols: int = 2):
    """Grouped outcome bars per rubric over every model (left-to-right by method/K/iter)."""
    metrics = _metrics(scores_long["questionnaire"].unique(), metrics)
    order = list(order) if order is not None else figures.model_order(scores_long)
    fig, axes = figures.grid(len(metrics), ncols=ncols, panel=(7.6, 3.4))
    for ax, m in zip(axes, metrics):
        dm = scores_long[scores_long.questionnaire == m]
        sns.barplot(dm, x="model", y="score", hue="arm",
                    order=order, palette=palette, errorbar=("ci", 95), dodge=False, ax=ax)
        ax.set_title(display_label(m)); ax.set_xlabel("")
        figures.relabel_xticks(ax)
        figures.add_base_line(ax, float(dm[dm.is_base].score.mean()) if dm.is_base.any() else None)
    figures.figure_legend_from(axes[0], fig, title="arm")
    fig.suptitle("Outcome metrics by model — full-conversation eval (dotted line = base)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Effect forest (replaces the wide main-results table inline) ───────────────
_EFFECT_COLOR = {"negligible": "#bdbdbd", "small": "#9ecae1", "medium": "#4292c6", "large": "#08519c"}


def effect_forest(main_results_df, *, arms: Optional[Sequence[str]] = None,
                  metric_order: Optional[Sequence[str]] = None,
                  lower_is_better: Optional[Sequence[str]] = None,
                  caption: Optional[str] = None):
    """Forest/dot plot of each arm×rubric change vs base — the readable stand-in for the
    28-row main-results table.

    Takes :func:`stats.main_results_table` output. One row per (arm, rubric): the mean Δ vs base
    (dot) with its 95% bootstrap CI (whisker), dz annotated; dashed line at 0 = base (no change).
    Arms are blocked together, rubrics in canonical order.

    Coloring: higher-is-better rubrics use the effect-size ramp (darker = larger effect). Rubrics in
    ``lower_is_better`` (default = the package ``LOWER_IS_BETTER`` set, i.e. MICI) invert valence, so
    a positive Δ is *bad*; those rows are colored by DIRECTION (red = moved the wrong way, green =
    improved) and their label carries a ``↓``. ``caption`` prints an italic note under the axis.
    """
    if main_results_df is None or main_results_df.empty:
        return None
    lower = set(LOWER_IS_BETTER if lower_is_better is None else lower_is_better)
    df = main_results_df
    if arms is not None:
        df = df[df["arm"].isin(arms)]
    metric_order = [m for m in (metric_order or QUESTIONNAIRE_ORDER) if m in set(df["rubric"])]
    arm_order = sorted(df["arm"].unique())
    rows = [df[(df.arm == a) & (df.rubric == m)].iloc[0]
            for a in arm_order for m in metric_order
            if len(df[(df.arm == a) & (df.rubric == m)])]
    rows = rows[::-1]  # first arm/rubric at the top (matplotlib y grows upward)
    fig, ax = plt.subplots(figsize=(7.8, max(4.0, 0.34 * len(rows))))
    yticks, ylabels, has_lower = [], [], False
    for i, r in enumerate(rows):
        if r["rubric"] in lower:                       # valence-inverted (positive Δ = worse)
            has_lower = True
            color = "#D55E00" if r["delta"] > 0 else "#2ca02c"
        else:
            color = _EFFECT_COLOR.get(r["effect"], "#999999")
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, lw=2.4, solid_capstyle="round", zorder=2)
        ax.scatter([r["delta"]], [i], color=color, s=42, zorder=3)
        ax.text(r["ci_high"], i, f"  dz={r['dz']:.2f}", va="center", fontsize=6.5, color="#333333")
        yticks.append(i); ylabels.append(f"{arm_label(r['arm'])} · {display_label(r['rubric'])}")
    ax.axvline(0, color="#555555", lw=1.0, ls="--")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("change vs base (Δ mean score, 95% CI)")
    ax.set_title("Effect on full-conversation eval vs base (dashed = base; dz labelled)")
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for l, c in _EFFECT_COLOR.items()]
    if has_lower:
        handles += [Patch(color="#D55E00", label="↓metric: worse"),
                    Patch(color="#2ca02c", label="↓metric: better")]
    ax.legend(handles=handles, title="effect size", fontsize=7, loc="lower right", framealpha=0.9)
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.5,
                 style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def reliability_curve(agreement_df, *, palette=None):
    """Rank-agreement vs partial-conversation length — the Exp3 reward-reliability curve.

    Takes :func:`stats.rank_agreement_by_nturns` output (arm, n_turns, agreement, n_pairs). One
    line per arm; dashed line at 0.5 = chance. Comparing LA0 vs LA5 tests whether look-ahead makes
    the short training reward more faithful to the full-conversation eval.
    """
    if agreement_df is None or agreement_df.empty:
        return None
    pal = palette or figures.arm_palette(sorted(agreement_df.arm.unique()))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.lineplot(agreement_df, x="n_turns", y="agreement", hue="arm", marker="o", palette=pal, ax=ax)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.text(ax.get_xlim()[1], 0.5, " chance", ha="right", va="bottom", fontsize=7, color="grey")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Training-reward faithfulness: rank agreement vs conversation length")
    ax.set_xlabel("partial-conversation length when scored (n_turns)")
    ax.set_ylabel("sign-agreement with full-conv eval")
    sns.move_legend(ax, "lower right", title="arm", frameon=True)
    fig.tight_layout()
    return fig


def subscale_trajectory_grid(subscales_long, *, parents: Sequence[str] = ("WAI-SR", "MITI"),
                             min_iters: int = 3, arms: Optional[Sequence[str]] = None):
    """WAI-SR / MITI subscale *trajectories* — one panel per (parent, arm).

    Replaces the old 26-model × 3–4-subscale grouped-bar wall (unreadable; the subscales are
    near-equal within a model). Here each panel shows one colored line per subscale across
    ``iteration`` (±95% CI), so the reader can see how the components evolve and whether some
    (e.g. Bond / Empathy) climb faster than others (Goal / ChangeTalk). Each arm starts from its
    own iter-0 base. Arms with fewer than ``min_iters`` scored iterations are omitted (keeps a
    one-point arm like GRPO_LA5 out of the grid).
    """
    if subscales_long is None or subscales_long.empty:
        return None
    df = subscales_long
    iters_per_arm = df.groupby("arm")["iteration"].nunique()
    keep = [a for a in (arms if arms is not None else sorted(df.arm.unique()))
            if iters_per_arm.get(a, 0) >= min_iters]
    parents = [p for p in parents if p in set(df.parent)]
    if not keep or not parents:
        return None
    nrows, ncols = len(parents), len(keep)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows),
                             squeeze=False, sharex=True)
    for r, parent in enumerate(parents):
        for c, arm in enumerate(keep):
            ax = axes[r][c]
            d = df[(df.parent == parent) & (df.arm == arm)]
            sns.lineplot(d, x="iteration", y="score", hue="subscale", marker="o",
                         errorbar=("ci", 95), ax=ax)
            ax.set_title(f"{display_label(parent)} — {arm_label(arm)}")
            ax.set_xlabel("iteration" if r == nrows - 1 else "")
            ax.set_ylabel("score" if c == 0 else "")
            # keep one legend per parent (its subscale set) on the left-most panel
            if c == 0 and ax.get_legend():
                ax.legend(fontsize=7, title="")
            elif ax.get_legend():
                ax.legend_.remove()
    fig.suptitle("WAI-SR / MITI subscale trajectories across iterations",
                 y=1.01, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Trajectories ─────────────────────────────────────────────────────────────
_SHARED_FACTOR_CAVEAT = ("The 9 metrics share one dominant warmth/satisfaction factor (PC1≈55%; "
                         "technique + MI-inconsistency load a second factor, PC2≈16%), so a uniform "
                         "rise partly reflects one axis, not independent multi-skill gains.")


def trajectory_grid(scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                    arms: Optional[Sequence[str]] = None, iters: Optional[Sequence[int]] = None,
                    ncols: int = 3, caption: Optional[str] = _SHARED_FACTOR_CAVEAT):
    """Per-rubric mean ±95% CI across iterations, arms overlaid (one panel per rubric).

    ``arms``/``iters`` select which arms/iterations to show (None = all). A single shared arm
    legend sits above the grid; ``caption`` (default = the shared-factor caveat: PC1≈55% once the
    orthogonal axes are included) is printed under it so "all metrics up" isn't read as multi-skill
    evidence. Pass ``caption=None`` to suppress.
    """
    if arms is not None:
        scores_long = scores_long[scores_long.arm.isin(list(arms))]
    if iters is not None:
        scores_long = scores_long[scores_long.iteration.isin(list(iters))]
    metrics = _metrics(scores_long["questionnaire"].unique(), metrics)
    fig, axes = figures.grid(len(metrics), ncols=ncols, panel=(5.2, 3.2))
    for ax, m in zip(axes, metrics):
        sns.lineplot(scores_long[scores_long.questionnaire == m], x="iteration", y="score",
                     hue="arm", palette=palette, marker="o", errorbar=("ci", 95), ax=ax)
        ax.set_title(display_label(m))
    figures.figure_legend_from(axes[0], fig, title="arm")
    fig.suptitle("Outcome trajectories across iterations — full-conversation eval",
                 y=1.12, fontweight="bold")
    if caption:
        fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=7.5,
                 style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def single_metric_trajectory(scores_long, metric: str = "Q1Q2", *, palette,
                             arms: Optional[Sequence[str]] = None,
                             iters: Optional[Sequence[int]] = None,
                             oracle_noise: Optional[float] = 0.10,
                             baseline_arm: Optional[str] = None,
                             mark_peaks: bool = False):
    """One-metric learning curve (arms overlaid) with an oracle-noise band around base.

    ``arms``/``iters`` select which arms/iterations to overlay (None = all) — the lever for
    "show only PTO_LA0 vs GRPO_LA0" without a separate hardcoded figure. ``baseline_arm`` anchors
    the grey ±``oracle_noise`` band; if ``None`` the first arm with a base row is used.
    ``oracle_noise=None`` suppresses the band entirely — the ~0.10 reproducibility figure was
    measured on Q1Q2, so per-metric loops pass it only for Q1Q2.

    ``mark_peaks=True`` draws a dotted vline + label at each arm's peak iteration **only where the
    peak precedes the final iteration** (i.e. the arm regressed afterwards) — auto-surfacing e.g.
    GRPO's iter-8 peak-then-decline without hardcoding any arm/iteration.
    """
    d = scores_long[scores_long.questionnaire == metric]
    if arms is not None:
        d = d[d.arm.isin(list(arms))]
    if iters is not None:
        d = d[d.iteration.isin(list(iters))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(d, x="iteration", y="score", hue="arm", palette=palette, marker="o",
                 errorbar=("ci", 95), ax=ax)
    base = d[d.is_base]
    if baseline_arm is not None:
        base = base[base.arm == baseline_arm]
    if oracle_noise is not None and not base.empty:
        b0 = float(base.score.mean())
        ax.axhspan(b0 - oracle_noise, b0 + oracle_noise, color="grey", alpha=0.15)
        ax.text(0.02, b0 + oracle_noise, " ~oracle-noise band around base", fontsize=7,
                va="bottom", color="grey")
    if mark_peaks:
        for arm, g in d.groupby("arm"):
            per_iter = g.groupby("iteration")["score"].mean()
            if per_iter.empty:
                continue
            peak_it, final_it = int(per_iter.idxmax()), int(per_iter.index.max())
            if peak_it < final_it:   # regressed after the peak — worth flagging
                col = palette.get(arm, "#555555") if isinstance(palette, dict) else "#555555"
                ax.axvline(peak_it, color=col, lw=1.0, ls=":", alpha=0.85, zorder=1)
                ax.annotate(f"{arm_label(arm)} peak (it {peak_it}) → regresses",
                            xy=(peak_it, float(per_iter.max())), xytext=(4, 6),
                            textcoords="offset points", fontsize=7, color=col, va="bottom")
    ax.set_title(f"{display_label(metric)} across iterations — full-conversation eval")
    ax.set_xlabel("training iteration (model state)"); ax.set_ylabel(f"{display_label(metric)} (eval)")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), title="arm", frameon=False)
    figures.relabel_legend(ax)
    fig.tight_layout()
    return fig


def reward_hack_panel(scores_long, *, arms: Sequence[str], palette=None,
                      warmth_metric: str = "Q1Q2",
                      right_metrics: Sequence[str] = ("MICI", "PCT")):
    """The reward-hack in ONE frame: warmth climbs while the orthogonal axes reveal the cost.

    One panel per arm with twin y-axes. LEFT (1–5): the warmth reward proxy (``warmth_metric``,
    solid grey). RIGHT (0–1): each of ``right_metrics`` — ``MICI`` (MI-inconsistency, dashed, red =
    worse) and ``PCT`` (patient change-talk, dotted, green = the real MI goal). Reads at a glance:
    warmth **and** MI-inconsistency rise together while patient change-talk barely moves — so "all
    rubrics up" is NOT multi-skill. Per-iteration means (no CI band, to keep the twin axis legible).
    Returns ``None`` if no requested arm is present.
    """
    arms = [a for a in arms if a in set(scores_long.arm.unique())]
    rights = [m for m in right_metrics if m in set(scores_long.questionnaire.unique())]
    if not arms:
        return None
    warm_color = "#333333"
    right_style = {"MICI": ("--", "#D55E00", "s"), "PCT": (":", "#009E73", "^")}
    fig, axes = figures.grid(len(arms), ncols=min(2, len(arms)), panel=(6.2, 4.0))
    for ax, arm in zip(axes, arms):
        dw = (scores_long[(scores_long.arm == arm) & (scores_long.questionnaire == warmth_metric)]
              .groupby("iteration")["score"].mean())
        ax.plot(dw.index, dw.values, marker="o", color=warm_color, lw=2.0,
                label=display_label(warmth_metric))
        ax.set_ylabel(f"{display_label(warmth_metric)} (1–5)", color=warm_color)
        ax.tick_params(axis="y", labelcolor=warm_color)
        ax.set_xlabel("training iteration"); ax.set_title(arm_label(arm))
        axr = ax.twinx()
        for m in rights:
            ls, col, mk = right_style.get(m, ("-.", "#777777", "d"))
            dm = (scores_long[(scores_long.arm == arm) & (scores_long.questionnaire == m)]
                  .groupby("iteration")["score"].mean())
            axr.plot(dm.index, dm.values, marker=mk, ms=4, ls=ls, color=col, label=display_label(m))
        axr.set_ylabel("rate / proportion (0–1)")
        axr.set_ylim(0, None)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle("Reward-hacking: warmth ↑ and MI-inconsistency ↑ together, "
                 "patient change-talk ~flat", y=1.03, fontweight="bold")
    fig.text(0.5, -0.02, "Left axis = warmth reward proxy (higher = better). Right axis: "
             "MI-Inconsistency (red, higher = worse) · Patient Change-Talk (green, higher = better "
             "— the actual MI goal).", ha="center", va="top", fontsize=7.5, style="italic",
             color="#444444", wrap=True)
    fig.tight_layout()
    return fig


# Okabe-Ito qualitative colors for nominal persona-trait categories (not the arm palette).
_QUAL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
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
    fig, axes = figures.grid(len(arm_list), ncols=ncols, panel=(6.4, 3.8))
    for ax, arm in zip(axes, arm_list):
        sns.lineplot(d[d.arm == arm], x="iteration", y="score", hue=char, hue_order=cats,
                     marker="o", palette=hue_pal, ax=ax)
        ax.set_title(arm_label(arm)); ax.set_ylabel(display_label(metric)); ax.set_xlabel("iteration")
        if ax is axes[0]:
            figures.relabel_legend(ax, valmap)      # readable category names; palette keyed on raw
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
    pal = palette or figures.arm_palette(arm_list)
    pal_disp = {arm_label(a): pal.get(a, "#777777") for a in arm_list}
    fig, ax = plt.subplots(figsize=(1.7 * max(3, len(cat_order)) + 2, 4.4))
    sns.barplot(fin, x="cat", y="score", hue="arm_disp", order=cat_order,
                hue_order=[arm_label(a) for a in arm_list], palette=pal_disp,
                errorbar=("ci", 95), ax=ax)
    figures.add_base_line(ax, float(d[d.is_base].score.mean()) if d.is_base.any() else None)
    ax.set_title(f"Final-iteration {display_label(metric)} by {char.replace('_', ' ')} "
                 f"(per arm; dotted = base)")
    ax.set_xlabel(""); ax.set_ylabel(display_label(metric))
    ax.legend(title="arm", fontsize=8)
    fig.tight_layout()
    return fig


# ── Mechanism: faithfulness + rubric structure ───────────────────────────────
def faithfulness_proxy_vs_eval(scores_long, generations, *, metric: str = "Q1Q2",
                               palette=None):
    """Training proxy reward vs full-conversation eval, per (arm, iteration); dashed y=x.

    Joins on ``eval_iter = train_iter - 1`` (the proxy iteration N branches the policy that
    produced the ``model_iter_{N-1}`` eval convs). Used by ``4_Training_and_Reliability``.
    """
    if generations.empty:
        return None
    proxy = (generations.groupby(["arm", "eval_iter"])["score"].mean()
             .rename("proxy").reset_index().rename(columns={"eval_iter": "iteration"}))
    evalq = (scores_long[scores_long.questionnaire == metric]
             .groupby(["arm", "iteration"])["score"].mean().rename("eval").reset_index())
    faith = proxy.merge(evalq, on=["arm", "iteration"])
    if faith.empty:
        return None
    pal = palette or figures.arm_palette(sorted(faith.arm.unique()))
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.scatterplot(faith, x="proxy", y="eval", hue="arm", s=90, palette=pal, ax=ax)
    for _, r in faith.iterrows():
        ax.annotate(int(r.iteration), (r.proxy, r.eval), fontsize=7)
    lo = float(faith[["proxy", "eval"]].min().min())
    hi = float(faith[["proxy", "eval"]].max().max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
    ax.set_title(f"Training proxy reward vs full-conv eval ({metric})")
    ax.set_xlabel("proxy (training, K=0 branch)"); ax.set_ylabel(f"eval {metric}")
    fig.tight_layout()
    return fig


def rubric_correlation_heatmap(scores_long, *, metrics: Optional[Sequence[str]] = None,
                               corr_method: str = "spearman"):
    """Inter-rubric correlation heatmap (pooled over conversations).

    Diverging ``[-1, 1]`` colormap so a genuinely ORTHOGONAL or anti-correlated axis (e.g. the
    lower-is-better ``MICI``) shows as white/blue instead of being clipped to 0 — the whole point
    of adding the new axes is to see them NOT block-correlate with the warmth rubrics. Labels are
    sign-flagged via :func:`display_label` (lower-is-better metrics get a trailing ↓).
    """
    from . import stats
    corr = stats.rubric_correlation(scores_long, metrics=metrics, method=corr_method)
    cols = list(corr.columns)
    labels = [short_label(m) for m in cols]   # acronym-only ticks (a 10x10 matrix can't fit the gloss)
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="vlag",
                square=True, xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": f"{corr_method.title()} ρ"}, ax=ax)
    # Family divider: the warmth rubrics (top-left block) should intercorrelate while the orthogonal
    # axes sit apart — draw a heavy separator where warmth ends + name the two blocks so the
    # two-factor structure reads at a glance (metrics arrive warmth-first, see WARMTH+ORTHOGONAL order).
    n_warm = sum(1 for m in cols if m in set(WARMTH_RUBRICS))
    if 0 < n_warm < len(cols):
        for line in (ax.axhline, ax.axvline):
            line(n_warm, color="#222222", lw=2.0)
        ax.text(n_warm / 2, -0.35, "Warmth (one factor)", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#0072B2", clip_on=False)
        ax.text((n_warm + len(cols)) / 2, -0.35, "Orthogonal axes", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#D55E00", clip_on=False)
    ax.set_title(f"Inter-rubric correlation ({corr_method.title()}, pooled)", pad=34)
    fig.tight_layout()
    return fig


def factor_loadings_bars(scores_long, *, metrics: Optional[Sequence[str]] = None,
                         components: Sequence[str] = ("PC1",)):
    """How much each metric belongs to the dominant 'warmth' factor — a readable loadings bar chart.

    Standardized PCA over the (expanded) metric set; plots each metric's **loading** (correlation
    with the factor) on PC1 (and PC2 if requested) as a horizontal bar, blue for the 5 warmth rubrics
    and orange for the orthogonal axes. Reads in one glance: the warmth rubrics all load high on PC1
    (≈0.44 each — one shared factor), while R:Q/%CR/%MICO/PCT/MICI load ≈0 on PC1 (they are NOT on
    the warmth factor). Replaces the hard-to-read PC1×PC2 biplot. ``None`` if PCA can't be fit.
    """
    from . import stats
    fs = stats.rubric_factor_space(scores_long, metrics=metrics)
    if fs is None:
        return None
    load, evr, mets = fs["loadings"], fs["explained"], fs["metrics"]
    comp_idx = {"PC1": 0, "PC2": 1}
    comps = [c for c in components if comp_idx.get(c, 99) < len(evr)]
    order = sorted(mets, key=lambda m: load[m][0])   # ascending PC1 loading
    warm = set(WARMTH_RUBRICS)
    colors = ["#0072B2" if m in warm else "#D55E00" for m in order]
    fig, axes = plt.subplots(1, len(comps), figsize=(4.6 * len(comps), 0.42 * len(order) + 1.2),
                             squeeze=False)
    y = np.arange(len(order))
    for ax, comp in zip(axes.flat, comps):
        vals = [load[m][comp_idx[comp]] for m in order]
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y); ax.set_yticklabels([display_label(m) for m in order], fontsize=8.5)
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_xlabel(f"loading on {comp} ({evr[comp_idx[comp]]:.0%} of variance)")
        ax.set_title({"PC1": "Dominant 'warmth' factor", "PC2": "Second factor"}.get(comp, comp))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#0072B2", label="warmth rubrics (one factor)"),
                        Patch(color="#D55E00", label="orthogonal axes")],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def leaderboard_scorecard(scores_long, *, metrics: Optional[Sequence[str]] = None,
                          selection: str = "best"):
    """One-glance scorecard: each arm's headline score per metric, warmth beside orthogonal axes.

    ``selection="best"`` uses each arm's best iteration (by own oracle) + its base; ``"final"`` uses
    the last iteration. Returns a tidy DataFrame (arm × metric) with lower-is-better metrics flagged
    ``↓`` in the column name — drop straight into ``save_table``.
    """
    from . import best_per_experiment, all_models
    order = [m for m in (list(QUESTIONNAIRE_ORDER) + list(ORTHOGONAL_METRICS)) if m not in ("Q1", "Q2")]
    present = [m for m in (metrics or order) if m in set(scores_long.questionnaire.unique())]
    # "best" already filters to base + the own-oracle peak iteration per arm; both selections then
    # take the highest remaining non-base iteration as the headline row.
    sel = best_per_experiment(scores_long)[0] if selection == "best" else all_models(scores_long)
    rows = []
    for arm, g in sel.groupby("arm", sort=False):
        nb = g[~g.is_base]
        if nb.empty:
            continue
        pick = int(nb.iteration.max())
        gg = nb[nb.iteration == pick]
        row = {"arm": arm_label(arm), "iteration": pick}
        for m in present:
            v = gg[gg.questionnaire == m]["score"]
            row[display_label(m)] = round(float(v.mean()), 3) if len(v) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ── Behavior ─────────────────────────────────────────────────────────────────
# Per-therapist-turn rates for the length-scaling MITI counts (not raw B*_ counts), so the drift
# figure isn't inflated by longer late-iteration conversations. RtoQ is already a ratio; q_per_turn
# is already a rate. behavior_by_iter emits the `<m>_per_turn` columns.
_DEFAULT_BEHAVIOR_METRICS = ["B3_Q_per_turn", "B4_SR_per_turn", "B5_CR_per_turn", "B6_AF_per_turn",
                             "B2_Persuade_per_turn", "B1_GI_per_turn", "B7_Seek_per_turn",
                             "RtoQ", "Empathy", "loop", "q_per_turn"]


def behavior_trajectory_grid(behavior_by_iter, *, palette=None,
                             metrics: Optional[Sequence[str]] = None, ncols: int = 3,
                             title: str = "Behavior trajectories (MITI counts + text metrics)"):
    """Behavior metric trajectories across iterations, all arms (one panel per metric).

    Generic wide-frame → grid: reused for the MITI drift set (default ``metrics``) and for the
    MICI / PCT per-item detail frames (pass an explicit ``metrics`` list + a ``title``). Each
    panel is one arm-hued line per metric column; ``display_label`` names the axes/titles.
    """
    bm = [m for m in (metrics or _DEFAULT_BEHAVIOR_METRICS) if m in behavior_by_iter.columns]
    if not bm:
        return None
    pal = palette or figures.arm_palette(sorted(behavior_by_iter.arm.unique()))
    fig, axes = figures.grid(len(bm), ncols=ncols)
    for ax, m in zip(axes, bm):
        sns.lineplot(behavior_by_iter, x="iteration", y=m, hue="arm", palette=pal, marker="o", ax=ax)
        ax.set_title(display_label(m)); ax.set_ylabel(display_label(m))
        if ax is axes[0]:
            figures.relabel_legend(ax)
        elif ax.get_legend():
            ax.legend_.remove()
    fig.suptitle(title, y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def single_behavior_trajectory(behavior_by_iter, metric: str, *, palette=None):
    """One behavior metric across iterations, arms overlaid — the per-metric zoom of
    :func:`behavior_trajectory_grid` (for the ``3_mechanism/behavior/`` subfolder).

    Same data + palette as the combined grid; a full-size single panel so a reader can read one
    signal (e.g. B6_AF affirmations, or q_per_turn) closely. ``None`` if ``metric`` is absent.
    """
    if metric not in behavior_by_iter.columns:
        return None
    pal = palette or figures.arm_palette(sorted(behavior_by_iter.arm.unique()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(behavior_by_iter, x="iteration", y=metric, hue="arm", palette=pal, marker="o", ax=ax)
    ax.set_title(f"{display_label(metric)} across iterations")
    ax.set_xlabel("training iteration"); ax.set_ylabel(display_label(metric))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), title="arm", frameon=False)
    figures.relabel_legend(ax)
    fig.tight_layout()
    return fig


def question_rate_crosscheck(cross_df, *, palette=None):
    """Question rate: deterministic ``?``/turn (solid) vs oracle MITI ``B3_Q``/turn (dashed), per arm.

    Takes :func:`behavior.question_rate_crosscheck` (both columns already unit-harmonized to
    questions-per-therapist-turn). Overlays both measures per arm on ONE axis so the reader sees
    they track each other (cross-validation) and where they diverge (e.g. GRPO late). ``None`` if
    unscored/empty.
    """
    if cross_df is None or cross_df.empty:
        return None
    pal = palette or figures.arm_palette(sorted(cross_df.arm.unique()))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for arm, g in cross_df.groupby("arm"):
        g = g.sort_values("iteration")
        col = pal.get(arm, "#777777")
        ax.plot(g.iteration, g.q_per_turn, marker="o", color=col, ls="-",
                label=f"{arm_label(arm)} — regex ?/turn")
        ax.plot(g.iteration, g.q_per_turn_miti, marker="s", ms=4, color=col, ls="--",
                label=f"{arm_label(arm)} — oracle Q/turn")
    ax.set_title("Question rate: deterministic ?-count vs oracle MITI count (unit-harmonized cross-check)")
    ax.set_xlabel("training iteration"); ax.set_ylabel("questions per therapist turn")
    ax.legend(fontsize=7, ncol=2, frameon=True)
    fig.tight_layout()
    return fig


# ── Training internals (both methods, side-by-side) ──────────────────────────
def reward_distribution(reward_frame, *, ncols: int = 2):
    """Per-candidate training-reward distribution per iteration — one panel per arm.

    Takes the tidy ``(arm, method, train_iter, score)`` frame
    (:func:`training.reward_distribution_frame`) so PTO and GRPO sit side-by-side under matched
    axes — the symmetric replacement for the old per-arm DeepDive boxplot.
    """
    if reward_frame.empty:
        return None
    arms = sorted(reward_frame.arm.unique())
    pal = figures.arm_palette(arms)
    fig, axes = figures.grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
    for ax, arm in zip(axes, arms):
        g = reward_frame[reward_frame.arm == arm]
        sns.boxplot(g, x="train_iter", y="score", color=pal.get(arm, "#c5b0d5"), ax=ax)
        ax.set_title(arm_label(arm)); ax.set_xlabel("training iteration"); ax.set_ylabel("candidate reward")
    fig.suptitle("TRAINING signal — candidate reward distribution per iteration "
                 "(oracle on partial-conv branches, NOT the full-conv eval)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def advantage_signal_sidebyside(advantage_df, *, ncols: int = 2):
    """The training-advantage signal for BOTH methods in one figure (never gated by method).

    Takes :func:`training.advantage_signal_by_iter`. One panel per arm on the same oracle-score-gap
    y-axis. The PRIMARY (solid) line for BOTH methods is the UNFILTERED per-branch **best − worst
    candidate reward range** (``group_range``) — the true like-for-like decisiveness signal. Each
    method's native secondary is faint dashed: GRPO ``group_std`` (within-group spread); PTO the
    **τ-filtered chosen − rejected margin** (the actual DPO signal). The PTO margin sits slightly
    ABOVE its own range because τ keeps only large-gap branches — i.e. the filtered margin mildly
    overstates PTO's unfiltered decisiveness, which is why margin-vs-range comparisons (not
    range-vs-range) previously made PTO look more comparable to GRPO than it is. Colors follow the
    arm palette (PTO cool / GRPO warm). Arms with no on-disk training capture don't appear.
    """
    if advantage_df.empty:
        return None
    arms = sorted(advantage_df.arm.unique())
    pal = figures.arm_palette(arms)
    # Shared y-limit so every panel's range/margin is visually comparable (same units + same scale).
    gap_cols = [c for c in ("group_range", "margin") if c in advantage_df.columns]
    gmax = float(np.nanmax(advantage_df[gap_cols].to_numpy())) if gap_cols else 1.0
    ymax = (gmax * 1.15) if np.isfinite(gmax) and gmax > 0 else 1.0
    fig, axes = figures.grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
    for ax, arm in zip(axes, arms):
        g = advantage_df[advantage_df.arm == arm].sort_values("train_iter")
        method = g.method.iloc[0]
        color = pal.get(arm, "#555555")
        # PRIMARY (both methods): unfiltered best − worst candidate range = the like-for-like signal.
        if "group_range" in g and g["group_range"].notna().any():
            ax.plot(g.train_iter, g.group_range, marker="o", color=color,
                    label="best − worst candidate reward (unfiltered range)")
        # SECONDARY (method-native, faint dashed).
        if method == "GRPO" and "group_std" in g and g["group_std"].notna().any():
            ax.plot(g.train_iter, g.group_std, marker="s", ms=4, ls="--", color=color, alpha=0.5,
                    label="within-group std")
        elif method == "PTO" and "margin" in g and g["margin"].notna().any():
            ax.plot(g.train_iter, g.margin, marker="s", ms=4, ls="--", color=color, alpha=0.5,
                    label="chosen − rejected margin (τ-filtered pairs)")
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_ylabel("oracle-score gap")
        ax.set_title(arm_label(arm))
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("training iteration")
        ax.set_ylim(0, ymax)   # shared across panels → the decisiveness curves are comparable
    fig.suptitle("Training decisiveness (same oracle-score-gap scale): GRPO vs PTO best−worst candidate "
                 "range — unfiltered, like-for-like; PTO τ-filtered margin faint", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig
