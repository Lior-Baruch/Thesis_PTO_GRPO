"""outcomes.py — headline outcome figures: per-model bars, the vs-base effect forest, the scorecard."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..constants import (
    QUESTIONNAIRE_ORDER, ORTHOGONAL_METRICS, LOWER_IS_BETTER,
    display_label, arm_label,
)
from ..plotting_style import (
    grid, model_order, relabel_xticks, add_base_line, figure_legend_from,
)
from ._shared import _metrics


def outcomes_by_model(scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                      order: Optional[Sequence[str]] = None, ncols: int = 2,
                      title: Optional[str] = None):
    """Grouped outcome bars per rubric over the models present (left-to-right by method/K/iter).

    ``title`` overrides the default suptitle — the final-vs-best figure pair passes
    "…at the FINAL iteration" / "…at each arm's BEST iteration".
    """
    metrics = _metrics(scores_long["questionnaire"].unique(), metrics)
    order = list(order) if order is not None else model_order(scores_long)
    fig, axes = grid(len(metrics), ncols=ncols, panel=(7.6, 3.4))
    for ax, m in zip(axes, metrics):
        dm = scores_long[scores_long.questionnaire == m]
        sns.barplot(dm, x="model", y="score", hue="arm",
                    order=order, palette=palette, errorbar=("ci", 95), dodge=False, ax=ax)
        ax.set_title(display_label(m)); ax.set_xlabel("")
        relabel_xticks(ax)
        add_base_line(ax, float(dm[dm.is_base].score.mean()) if dm.is_base.any() else None)
    figure_legend_from(axes[0], fig, title="arm")
    fig.suptitle(title or "Outcome metrics by model — full-conversation eval (dotted line = base)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Effect forest (replaces the wide main-results table inline) ───────────────
_EFFECT_COLOR = {"negligible": "#bdbdbd", "small": "#9ecae1", "medium": "#4292c6", "large": "#08519c"}


def effect_forest(main_results_df, *, arms: Optional[Sequence[str]] = None,
                  metric_order: Optional[Sequence[str]] = None,
                  lower_is_better: Optional[Sequence[str]] = None,
                  caption: Optional[str] = None, title: Optional[str] = None):
    """Forest/dot plot of each arm×rubric change vs base — the readable stand-in for the
    28-row main-results table.

    Takes :func:`stats.main_results_table` output. One row per (arm, rubric): the mean Δ vs base
    (dot) with its 95% bootstrap CI (whisker), dz annotated; dashed line at 0 = base (no change).
    Arms are blocked together, rubrics in canonical order.

    Coloring: higher-is-better rubrics use the effect-size ramp (darker = larger effect). Rubrics in
    ``lower_is_better`` (default = the package ``LOWER_IS_BETTER`` set, i.e. MICI) invert valence, so
    a positive Δ is *bad*; those rows are colored by DIRECTION (red = moved the wrong way, green =
    improved) and their label carries a ``↓``. ``caption`` prints an italic note under the axis;
    ``title`` overrides the default axes title (the final-vs-best pair labels itself with it).
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
    ax.set_title(title or "Effect on full-conversation eval vs base (dashed = base; dz labelled)")
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


def leaderboard_scorecard(scores_long, *, metrics: Optional[Sequence[str]] = None,
                          selection: str = "best"):
    """One-glance scorecard: each arm's headline score per metric, global-eval rubrics beside orthogonal axes.

    ``selection`` ∈ {``"best"`` (each arm's best iteration by own oracle), ``"final"`` (each arm's
    last iteration)}. Returns a tidy DataFrame (arm × metric) with lower-is-better metrics flagged
    ``↓`` in the column name — drop straight into ``save_table`` (the notebook concatenates the two
    selections with a ``target`` column).
    """
    from ..data import best_per_experiment, all_models
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
