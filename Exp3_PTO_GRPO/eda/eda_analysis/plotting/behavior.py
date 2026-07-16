"""behavior.py — behaviour-count trajectory figures: the generic wide-frame detail grid (reused
by the MITI / MICI / PCT detail sections and the session-shape view), the per-metric zoom, the
official MITI 4.2.1 threshold panel/table, and the question-rate cross-check. (Data-side
counterparts live in :mod:`eda_analysis.behavior`; the Likert-item figures live in
:mod:`.questionnaires`.)"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..constants import MITI_THRESHOLDS, display_label, arm_label
from ..plotting_style import grid, arm_palette, relabel_legend

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
    :func:`behavior_trajectory_grid` (for the ``2_questionnaires/{miti,mici,pct}/`` subfolders).

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
