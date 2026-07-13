"""behavior.py — behaviour-drift figures: MITI count/text-metric trajectories, the official
MITI 4.2.1 threshold panel/table, the question-rate cross-check, and the Q2 item-level
reward-composition figures. (Data-side counterparts live in :mod:`eda_analysis.behavior`.)"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..constants import (
    MITI_THRESHOLDS, Q2_ITEM_SHORT, Q2_ITEM_GROUP_OF, Q2_ITEM_GROUPS,
    display_label, arm_label,
)
from ..plotting_style import grid, arm_palette, relabel_legend
from ._shared import _QUAL_COLORS

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
    pal = palette or arm_palette(sorted(behavior_by_iter.arm.unique()))
    fig, axes = grid(len(bm), ncols=ncols)
    for ax, m in zip(axes, bm):
        sns.lineplot(behavior_by_iter, x="iteration", y=m, hue="arm", palette=pal, marker="o", ax=ax)
        ax.set_title(display_label(m)); ax.set_ylabel(display_label(m))
        if ax is axes[0]:
            relabel_legend(ax)
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
    pal = palette or arm_palette(sorted(behavior_by_iter.arm.unique()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(behavior_by_iter, x="iteration", y=metric, hue="arm", palette=pal, marker="o", ax=ax)
    ax.set_title(f"{display_label(metric)} across iterations")
    ax.set_xlabel("training iteration"); ax.set_ylabel(display_label(metric))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), title="arm", frameon=False)
    relabel_legend(ax)
    fig.tight_layout()
    return fig


# ── MITI 4.2.1 official competency thresholds ────────────────────────────────
_MITI_THRESHOLD_CAVEAT = (
    "Thresholds from the MITI 4.2.1 manual (Moyers, Manuel & Ernst 2014, rev. 2015) — expert "
    "opinion, not normatively validated, and defined for ~20-min human audio sessions (short "
    "text-chat sessions are out-of-domain).")


def miti_threshold_panel(prof_df, *, palette=None, ncols: int = 2,
                         caption: Optional[str] = _MITI_THRESHOLD_CAVEAT):
    """The 4 official MITI 4.2.1 summary scores vs the manual's competency thresholds.

    Takes :func:`behavior.miti_proficiency_by_iter` (per arm, iteration: ``R:Q``, ``%CR``,
    ``MITI_Technical``, ``MITI_Relational``). One panel per summary score, arms overlaid, with
    the manual's **fair** (dashed amber) and **good** (dashed green) thresholds drawn as
    reference lines — the absolute anchor for "is this therapist competent in official MITI
    terms", not just better than base. ``caption`` (default = the manual's own expert-opinion
    caveat + our domain caveat) prints under the grid. ``None`` if nothing is plottable.
    """
    if prof_df is None or prof_df.empty:
        return None
    mets = [m for m in MITI_THRESHOLDS if m in prof_df.columns]
    if not mets:
        return None
    pal = palette or arm_palette(sorted(prof_df.arm.unique()))
    fig, axes = grid(len(mets), ncols=ncols, panel=(6.0, 3.6))
    for ax, m in zip(axes, mets):
        sns.lineplot(prof_df, x="iteration", y=m, hue="arm", palette=pal, marker="o", ax=ax)
        fair, good = MITI_THRESHOLDS[m]
        ax.axhline(fair, color="#E69F00", lw=1.2, ls="--", zorder=1)
        ax.axhline(good, color="#009E73", lw=1.2, ls="--", zorder=1)
        x1 = ax.get_xlim()[1]
        ax.text(x1, fair, " fair", ha="left", va="center", fontsize=7, color="#E69F00")
        ax.text(x1, good, " good", ha="left", va="center", fontsize=7, color="#009E73")
        ax.set_title(display_label(m)); ax.set_ylabel(display_label(m))
        if m == "%CR":
            ax.set_ylim(0, 1)
        if ax is axes[0]:
            relabel_legend(ax)
        elif ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("MITI 4.2.1 summary scores vs the manual's competency thresholds "
                 "(dashed: fair / good)", y=1.03, fontweight="bold")
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.5,
                 style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def miti_threshold_table(prof_df) -> Optional[pd.DataFrame]:
    """Per (arm × {base, final iteration}): each MITI 4.2.1 summary score with its
    threshold verdict — ``✓good`` / ``✓fair`` / ``✗`` (below basic competence).

    Tidy companion to :func:`miti_threshold_panel`; drop straight into ``save_table``.
    """
    if prof_df is None or prof_df.empty:
        return None
    mets = [m for m in MITI_THRESHOLDS if m in prof_df.columns]

    def verdict(m, v):
        if v is None or pd.isna(v):
            return "—"
        fair, good = MITI_THRESHOLDS[m]
        flag = "✓good" if v >= good else ("✓fair" if v >= fair else "✗")
        return f"{v:.2f} {flag}"

    rows = []
    for arm, g in prof_df.groupby("arm", sort=True):
        for label, it in (("base", int(g.iteration.min())), ("final", int(g.iteration.max()))):
            gi = g[g.iteration == it]
            if gi.empty:
                continue
            row = {"arm": arm_label(arm), "state": f"{label} (iter {it})"}
            for m in mets:
                row[display_label(m)] = verdict(m, float(gi[m].iloc[0]))
            rows.append(row)
    return pd.DataFrame(rows)


def question_rate_crosscheck(cross_df, *, palette=None):
    """Question rate: deterministic ``?``/turn (solid) vs oracle MITI ``B3_Q``/turn (dashed), per arm.

    Takes :func:`behavior.question_rate_crosscheck` (both columns already unit-harmonized to
    questions-per-therapist-turn). Overlays both measures per arm on ONE axis so the reader sees
    they track each other (cross-validation) and where they diverge (e.g. GRPO late). ``None`` if
    unscored/empty.
    """
    if cross_df is None or cross_df.empty:
        return None
    pal = palette or arm_palette(sorted(cross_df.arm.unique()))
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


# ── Q2 item-level reward composition (which items does the optimizer exploit?) ──
# Group colors reuse the Okabe-Ito qualitative set, keyed on the face-content groups.
_Q2_GROUP_COLORS = None  # built lazily (constants import order) in _q2_group_colors()


def _q2_group_colors() -> dict:
    global _Q2_GROUP_COLORS
    if _Q2_GROUP_COLORS is None:
        _Q2_GROUP_COLORS = {g: _QUAL_COLORS[i % len(_QUAL_COLORS)]
                            for i, g in enumerate(Q2_ITEM_GROUPS)}
    return _Q2_GROUP_COLORS


def q2_item_delta_bars(q2_deltas, *, ncols: int = 2):
    """Which Q2 items drive the reward gain — endpoint Δ vs base per item, one panel per arm.

    Takes :func:`stats.q2_item_endpoint_deltas` (arm, item, short, group, base, final, delta).
    Horizontal bars (one per item, shared cross-arm order by pooled Δ so panels compare), colored
    by the face-content item group (OUR analytical grouping, not a validated subscale — see
    ``constants.Q2_ITEM_GROUPS``). The reward-composition view: if the self-disclosure /
    warmth-closeness items climb hardest while non-judgment stays flat, the Q1+Q2 training reward
    is directly incentivizing the emotive drift. ``None`` if empty.
    """
    if q2_deltas is None or q2_deltas.empty:
        return None
    colors = _q2_group_colors()
    order = (q2_deltas.groupby("item")["delta"].mean().sort_values().index.tolist())
    arms = sorted(q2_deltas.arm.unique())
    # Shared x-limits so the per-arm panels are visually comparable (same Δ scale).
    xlo = min(0.0, float(q2_deltas["delta"].min()) * 1.05)
    xhi = float(q2_deltas["delta"].max()) * 1.05
    fig, axes = grid(len(arms), ncols=min(ncols, len(arms)), panel=(5.6, 4.6))
    for ax, arm in zip(axes, arms):
        d = q2_deltas[q2_deltas.arm == arm].set_index("item").reindex(order)
        y = np.arange(len(d))
        ax.barh(y, d["delta"].values,
                color=[colors.get(g, "#777777") for g in d["group"]])
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i}. {Q2_ITEM_SHORT.get(i, i)}" for i in d.index], fontsize=7.5)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlim(xlo, xhi)
        ax.set_title(arm_label(arm))
        ax.set_xlabel("Δ item mean, final − base (1–5 scale)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=g) for g, c in colors.items()],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=8,
               title="face-content item group (analytical, not a validated subscale)")
    fig.suptitle("Q2 reward composition — which alliance items drive the gain?",
                 y=1.12, fontweight="bold")
    fig.tight_layout()
    return fig


def q2_item_group_trajectory(q2_long, *, ncols: int = 2):
    """Mean Q2 item-group score across iterations, one panel per arm (hue = item group).

    The trajectory companion to :func:`q2_item_delta_bars`: maps each of the 17 items to its
    face-content group (``constants.Q2_ITEM_GROUPS``) and plots the per-group mean (±95% CI over
    conversations × items) per iteration. Shows *when* the exploited components take off, not just
    the endpoint. ``None`` if empty.
    """
    if q2_long is None or q2_long.empty:
        return None
    d = q2_long.copy()
    d["group"] = d["item"].map(Q2_ITEM_GROUP_OF)
    colors = _q2_group_colors()
    arms = sorted(d.arm.unique())
    fig, axes = grid(len(arms), ncols=min(ncols, len(arms)), panel=(6.0, 3.8))
    for ax, arm in zip(axes, arms):
        sns.lineplot(d[d.arm == arm], x="iteration", y="score", hue="group",
                     hue_order=list(colors), palette=colors, marker="o",
                     errorbar=("ci", 95), ax=ax)
        ax.set_title(arm_label(arm)); ax.set_xlabel("iteration"); ax.set_ylabel("Q2 item mean (1–5)")
        if ax is axes[0]:
            ax.legend(fontsize=7, title="item group")
        elif ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("Q2 item-group trajectories (face-content groups — analytical, "
                 "not a validated subscale)", y=1.03, fontweight="bold")
    fig.tight_layout()
    return fig
