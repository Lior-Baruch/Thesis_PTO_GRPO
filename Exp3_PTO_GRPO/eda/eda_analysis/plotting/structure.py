"""structure.py — reward-faithfulness + rubric-structure figures: the reliability curve,
proxy-vs-eval scatter, the inter-rubric heatmap, and the factor-loadings bars."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..constants import WARMTH_RUBRICS, display_label, short_label
from ..plotting_style import arm_palette


def reliability_curve(agreement_df, *, palette=None):
    """Rank-agreement vs partial-conversation length — the Exp3 reward-reliability curve.

    Takes :func:`stats.rank_agreement_by_nturns` output (arm, n_turns, agreement, n_pairs). One
    line per arm; dashed line at 0.5 = chance. Comparing LA0 vs LA5 tests whether look-ahead makes
    the short training reward more faithful to the full-conversation eval.
    """
    if agreement_df is None or agreement_df.empty:
        return None
    pal = palette or arm_palette(sorted(agreement_df.arm.unique()))
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
    pal = palette or arm_palette(sorted(faith.arm.unique()))
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
    of adding the new axes is to see them NOT block-correlate with the global-eval (halo) rubrics.
    Labels are sign-flagged via :func:`display_label` (lower-is-better metrics get a trailing ↓).
    """
    from .. import stats
    corr = stats.rubric_correlation(scores_long, metrics=metrics, method=corr_method)
    cols = list(corr.columns)
    labels = [short_label(m) for m in cols]   # acronym-only ticks (a 10x10 matrix can't fit the gloss)
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="vlag",
                square=True, xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": f"{corr_method.title()} ρ"}, ax=ax)
    # Family divider: the global-eval (halo) rubrics (top-left block) should intercorrelate while the
    # orthogonal axes sit apart — draw a heavy separator where the halo block ends + name the two
    # blocks so the two-factor structure reads at a glance (metrics arrive halo-first, see
    # WARMTH+ORTHOGONAL order — WARMTH_RUBRICS is the historical code name for this cluster).
    n_warm = sum(1 for m in cols if m in set(WARMTH_RUBRICS))
    if 0 < n_warm < len(cols):
        for line in (ax.axhline, ax.axvline):
            line(n_warm, color="#222222", lw=2.0)
        ax.text(n_warm / 2, -0.35, "Global-eval halo (one factor)", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#0072B2", clip_on=False)
        ax.text((n_warm + len(cols)) / 2, -0.35, "Orthogonal axes", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#D55E00", clip_on=False)
    ax.set_title(f"Inter-rubric correlation ({corr_method.title()}, pooled)", pad=34)
    fig.tight_layout()
    return fig


def factor_loadings_bars(scores_long, *, metrics: Optional[Sequence[str]] = None,
                         components: Sequence[str] = ("PC1",)):
    """How much each metric belongs to the dominant global-eval (halo) factor — a loadings bar chart.

    Standardized PCA over the (expanded) metric set; plots each metric's **loading** (correlation
    with the factor) on PC1 (and PC2 if requested) as a horizontal bar, blue for the 5 global-eval
    rubrics and orange for the orthogonal axes. Reads in one glance: the global-eval rubrics all load
    high on PC1 (≈0.44 each — one shared halo factor), while R:Q/%CR/%MICO/MICI load ≈0 on PC1 (they
    are NOT on the halo factor). Replaces the hard-to-read PC1×PC2 biplot. ``None`` if PCA can't be fit.
    """
    from .. import stats
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
        ax.set_title({"PC1": "Dominant global-eval (halo) factor",
                      "PC2": "Second factor"}.get(comp, comp))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#0072B2", label="global-eval rubrics (halo factor)"),
                        Patch(color="#D55E00", label="orthogonal axes")],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
