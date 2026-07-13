"""trajectories.py — learning-curve figures: per-rubric grids, single-metric curves, subscales, the reward-hack panel."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

from ..constants import display_label, arm_label
from ..plotting_style import grid, figure_legend_from, relabel_legend
from ._shared import _metrics

_SHARED_FACTOR_CAVEAT = ("The 9 metrics share one dominant global-evaluation (halo) factor (PC1≈55%; "
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
    fig, axes = grid(len(metrics), ncols=ncols, panel=(5.2, 3.2))
    for ax, m in zip(axes, metrics):
        sns.lineplot(scores_long[scores_long.questionnaire == m], x="iteration", y="score",
                     hue="arm", palette=palette, marker="o", errorbar=("ci", 95), ax=ax)
        ax.set_title(display_label(m))
    figure_legend_from(axes[0], fig, title="arm")
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
    relabel_legend(ax)
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


def reward_hack_panel(scores_long, *, arms: Sequence[str], palette=None,
                      warmth_metric: str = "Q1Q2",
                      right_metrics: Sequence[str] = ("MICI", "PCT")):
    """The reward-hack in ONE frame: the reward proxy climbs while the orthogonal axes reveal the cost.

    One panel per arm with twin y-axes. LEFT (1–5): the global-eval reward proxy (``warmth_metric``
    — historical param name, kept for API stability; solid grey). RIGHT (0–1): each of
    ``right_metrics`` — ``MICI`` (MI-inconsistency, dashed, red = worse) and ``PCT`` (patient
    change-talk, dotted, green = the real MI goal). Reads at a glance: the reward proxy **and**
    MI-inconsistency rise together while patient change-talk barely moves — so "all rubrics up" is
    NOT multi-skill. Per-iteration means (no CI band, to keep the twin axis legible).
    Returns ``None`` if no requested arm is present.
    """
    arms = [a for a in arms if a in set(scores_long.arm.unique())]
    rights = [m for m in right_metrics if m in set(scores_long.questionnaire.unique())]
    if not arms:
        return None
    warm_color = "#333333"
    right_style = {"MICI": ("--", "#D55E00", "s"), "PCT": (":", "#009E73", "^")}
    fig, axes = grid(len(arms), ncols=min(2, len(arms)), panel=(6.2, 4.0))
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
    fig.suptitle("Reward-hacking: global-eval reward ↑ and MI-inconsistency ↑ together, "
                 "patient change-talk ~flat", y=1.03, fontweight="bold")
    fig.text(0.5, -0.02, "Left axis = the Q1+Q2 reward proxy (global evaluation; higher = better). "
             "Right axis: MI-Inconsistency (red, higher = worse) · Patient Change-Talk (green, higher "
             "= better — the actual MI goal).", ha="center", va="top", fontsize=7.5, style="italic",
             color="#444444", wrap=True)
    fig.tight_layout()
    return fig
