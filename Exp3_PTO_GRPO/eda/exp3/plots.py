"""
plots.py — named plot functions for the figures that recur across notebooks.

The hybrid plotting layer (v3): the *reused* figures (outcomes, trajectories, faithfulness,
behavior, training internals) are defined ONCE here and called from multiple notebooks, instead
of being copy-pasted inline three times. Genuinely one-off exploration still lives inline in the
notebooks (that's the point of the hybrid split).

Contract for every function here:
- takes an already-built tidy frame (``scores_long`` / ``behavior_by_iter`` / a generations or
  reward frame) — never touches disk, so plotting stays fast and host-agnostic;
- returns a matplotlib ``fig`` and does NOT call ``plt.show()`` or ``save_fig`` — the notebook
  does that (keeps the export name + inline display under the notebook's control);
- reuses :mod:`figures` helpers (``set_style`` is applied globally; ``arm_palette``/``grid``);
- degrades gracefully on thin/absent arms (returns ``None`` or an empty-but-labelled panel).
"""

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import QUESTIONNAIRE_ORDER, figures


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
        sns.barplot(scores_long[scores_long.questionnaire == m], x="model", y="score", hue="arm",
                    order=order, palette=palette, errorbar=("ci", 95), dodge=False, ax=ax)
        ax.set_title(m); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=90, labelsize=6)
        if ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("Outcome metrics by model (Exp3)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def outcomes_headline_by_arm(sel_scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                             ncols: int = 3):
    """Headline outcome bars: one bar per arm (caller passes a best-per-arm-filtered frame)."""
    metrics = _metrics(sel_scores_long["questionnaire"].unique(), metrics)
    arm_order = sorted(sel_scores_long.arm.unique())
    fig, axes = figures.grid(len(metrics), ncols=ncols, panel=(5.0, 3.0))
    for ax, m in zip(axes, metrics):
        sns.barplot(sel_scores_long[sel_scores_long.questionnaire == m], x="arm", y="score",
                    hue="arm", order=arm_order, palette=palette, errorbar=("ci", 95), ax=ax)
        ax.set_title(m); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30, labelsize=8)
        if ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("Best-iteration outcomes by arm (each arm's peak by own oracle)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def subscales_by_model(subscales_long, *, order: Optional[Sequence[str]] = None,
                       parents: Sequence[str] = ("WAI-SR", "MITI")):
    """WAI-SR (Goal/Task/Bond) + MITI (4 globals) subscale bars per model."""
    fig, axes = figures.grid(len(parents), ncols=2, panel=(8, 3.6))
    for ax, parent in zip(axes, parents):
        d = subscales_long[subscales_long.parent == parent]
        o = [m for m in (order or sorted(d.model.unique())) if m in set(d.model)]
        sns.barplot(d, x="model", y="score", hue="subscale", order=o, errorbar=("ci", 95), ax=ax)
        ax.set_title(f"{parent} subscales"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=6); ax.legend(fontsize=7, title="")
    fig.tight_layout()
    return fig


# ── Trajectories ─────────────────────────────────────────────────────────────
def trajectory_grid(scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                    ncols: int = 3):
    """Per-rubric mean ±95% CI across iterations, arms overlaid (one panel per rubric)."""
    metrics = _metrics(scores_long["questionnaire"].unique(), metrics)
    fig, axes = figures.grid(len(metrics), ncols=ncols, panel=(5.2, 3.2))
    for ax, m in zip(axes, metrics):
        sns.lineplot(scores_long[scores_long.questionnaire == m], x="iteration", y="score",
                     hue="arm", palette=palette, marker="o", errorbar=("ci", 95), ax=ax)
        ax.set_title(m)
        if ax is not axes[0] and ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("Outcome trajectories across iterations", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def single_metric_trajectory(scores_long, metric: str = "Q1Q2", *, palette,
                             oracle_noise: float = 0.10, baseline_arm: Optional[str] = None):
    """One-metric learning curve (arms overlaid) with an oracle-noise band around base.

    ``baseline_arm`` anchors the grey ±``oracle_noise`` band; if ``None`` the first arm that has a
    base row for *metric* is used (so this never assumes ``PTO_LA0`` exists).
    """
    d = scores_long[scores_long.questionnaire == metric]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(d, x="iteration", y="score", hue="arm", palette=palette, marker="o",
                 errorbar=("ci", 95), ax=ax)
    base = d[d.is_base]
    if baseline_arm is not None:
        base = base[base.arm == baseline_arm]
    if not base.empty:
        b0 = float(base.score.mean())
        ax.axhspan(b0 - oracle_noise, b0 + oracle_noise, color="grey", alpha=0.15)
        ax.text(0.02, b0 + oracle_noise, " ~oracle-noise band around base", fontsize=7,
                va="bottom", color="grey")
    ax.set_title(f"{metric} across iterations (training reward)")
    ax.set_xlabel("iteration"); ax.set_ylabel(metric)
    fig.tight_layout()
    return fig


def method_contrast_overlay(scores_long, metric: str = "Q1Q2", *,
                            pair: Tuple[str, str] = ("PTO_LA0", "GRPO_LA0"), palette):
    """Overlay two arms' trajectories on one axis (the head-to-head contrast figure).

    Degrades to whatever arms in *pair* are present (one line if the partner is unscored).
    """
    d = scores_long[(scores_long.arm.isin(pair)) & (scores_long.questionnaire == metric)]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(d, x="iteration", y="score", hue="arm", hue_order=[p for p in pair if p in set(d.arm)],
                 palette=palette, marker="o", errorbar=("ci", 95), ax=ax)
    ax.set_title(f"{metric}: {pair[0]} vs {pair[1]} (matched)")
    ax.set_xlabel("iteration"); ax.set_ylabel(metric)
    fig.tight_layout()
    return fig


# ── Mechanism: faithfulness + rubric structure ───────────────────────────────
def faithfulness_proxy_vs_eval(scores_long, generations, *, metric: str = "Q1Q2",
                               palette=None):
    """Training proxy reward vs full-conversation eval, per (arm, iteration); dashed y=x.

    Joins on ``eval_iter = train_iter - 1`` (the proxy iteration N branches the policy that
    produced the ``model_iter_{N-1}`` eval convs). Deduped from ``00`` + ``02``.
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
    """Inter-rubric correlation heatmap (pooled over conversations)."""
    from . import stats
    corr = stats.rubric_correlation(scores_long, metrics=metrics, method=corr_method)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="rocket_r", square=True, ax=ax)
    ax.set_title(f"Inter-rubric correlation ({corr_method.title()}, pooled)")
    fig.tight_layout()
    return fig


# ── Behavior ─────────────────────────────────────────────────────────────────
_DEFAULT_BEHAVIOR_METRICS = ["B3_Q", "B4_SR", "B5_CR", "B6_AF", "B2_Persuade",
                             "RtoQ", "Empathy", "loop", "affirm_rate"]


def behavior_trajectory_grid(behavior_by_iter, *, palette=None,
                             metrics: Optional[Sequence[str]] = None, ncols: int = 3):
    """Behavior metric trajectories (MITI counts + text metrics) across iterations, all arms."""
    bm = [m for m in (metrics or _DEFAULT_BEHAVIOR_METRICS) if m in behavior_by_iter.columns]
    if not bm:
        return None
    pal = palette or figures.arm_palette(sorted(behavior_by_iter.arm.unique()))
    fig, axes = figures.grid(len(bm), ncols=ncols)
    for ax, m in zip(axes, bm):
        sns.lineplot(behavior_by_iter, x="iteration", y=m, hue="arm", palette=pal, marker="o", ax=ax)
        ax.set_title(m)
        if ax is not axes[0] and ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("Behavior trajectories (MITI counts + text metrics)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def score_violin_by_model(scores_long, metric: str, *, palette, order: Optional[Sequence[str]] = None):
    """Per-model score distribution (violin) for one rubric (the descriptive spread view)."""
    d = scores_long[scores_long.questionnaire == metric]
    o = list(order) if order is not None else figures.model_order(scores_long)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * d.model.nunique()), 4))
    sns.violinplot(d, x="model", y="score", order=o, hue="arm", palette=palette, dodge=False,
                   density_norm="width", cut=0, ax=ax)
    ax.set_title(f"{metric} distribution by model"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    if ax.get_legend():
        ax.legend_.remove()
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
    fig, axes = figures.grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
    for ax, arm in zip(axes, arms):
        g = reward_frame[reward_frame.arm == arm]
        method = g.method.iloc[0] if "method" in g and len(g) else ""
        sns.boxplot(g, x="train_iter", y="score", color="#c5b0d5", ax=ax)
        ax.set_title(f"{arm} ({method})"); ax.set_xlabel("training iteration"); ax.set_ylabel("candidate reward")
    fig.suptitle("Candidate reward distribution per training iteration", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def advantage_signal_sidebyside(advantage_df, *, ncols: int = 2):
    """The training-advantage signal for BOTH methods in one figure (never gated by method).

    Takes :func:`training.advantage_signal_by_iter`. One panel per arm: GRPO panels plot
    ``group_std`` (within-group reward spread), PTO panels plot the chosen−rejected ``margin``.
    Arms with no on-disk training capture (e.g. GRPO_LA5) simply don't appear.
    """
    if advantage_df.empty:
        return None
    arms = sorted(advantage_df.arm.unique())
    fig, axes = figures.grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
    for ax, arm in zip(axes, arms):
        g = advantage_df[advantage_df.arm == arm].sort_values("train_iter")
        method = g.method.iloc[0]
        if method == "GRPO":
            ax.plot(g.train_iter, g.group_std, marker="o", color="#1f77b4")
            ax.set_ylabel("mean group_std")
            ax.set_title(f"{arm} (GRPO advantage spread)")
        else:
            ax.plot(g.train_iter, g.margin, marker="o", color="#7b4fb0")
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_ylabel("chosen − rejected margin")
            ax.set_title(f"{arm} (PTO preference margin)")
        ax.set_xlabel("training iteration")
    fig.suptitle("Training advantage signal (method-native, side-by-side)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig
