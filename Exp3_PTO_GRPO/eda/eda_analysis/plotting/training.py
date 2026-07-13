"""training.py — TRAINING-signal figures, both methods side-by-side: candidate-reward
distributions and the advantage/decisiveness comparison. (Data-side counterparts live in
:mod:`eda_analysis.training`.)"""

import numpy as np
import seaborn as sns

from ..constants import arm_label
from ..plotting_style import grid, arm_palette


def reward_distribution(reward_frame, *, ncols: int = 2):
    """Per-candidate training-reward distribution per iteration — one panel per arm.

    Takes the tidy ``(arm, method, train_iter, score)`` frame
    (:func:`training.reward_distribution_frame`) so PTO and GRPO sit side-by-side under matched
    axes — the symmetric replacement for the old per-arm DeepDive boxplot.
    """
    if reward_frame.empty:
        return None
    arms = sorted(reward_frame.arm.unique())
    pal = arm_palette(arms)
    fig, axes = grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
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
    pal = arm_palette(arms)
    # Shared y-limit so every panel's range/margin is visually comparable (same units + same scale).
    gap_cols = [c for c in ("group_range", "margin") if c in advantage_df.columns]
    gmax = float(np.nanmax(advantage_df[gap_cols].to_numpy())) if gap_cols else 1.0
    ymax = (gmax * 1.15) if np.isfinite(gmax) and gmax > 0 else 1.0
    fig, axes = grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
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
