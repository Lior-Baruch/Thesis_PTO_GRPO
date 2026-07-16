"""questionnaires.py — per-questionnaire drill-down figures (family ``2_questionnaires``):
the uniform item trajectory grid + the "which items drive the change" delta bars, generic over
every Likert-item rubric (Q1 / Q2 / WAI-SR / CSQ-8 / MI-SAT), plus the Q2 face-content-group
specializations. (Data side: :func:`eda_analysis.data.load_items` /
:func:`eda_analysis.stats.item_endpoint_deltas`. The MITI/PCT/MICI detail grids reuse
:func:`eda_analysis.plotting.behavior.behavior_trajectory_grid` on their by-iter frames.)"""

from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns

from ..constants import Q2_ITEM_GROUP_OF, Q2_ITEM_GROUPS, arm_label
from ..plotting_style import grid, arm_palette, figure_legend_from
from ._shared import _QUAL_COLORS

# Neutral bar color when no group coloring is requested (panels are per-arm, so bars must NOT
# reuse the arm palette — that would read as a cross-arm encoding inside a single-arm panel).
_NEUTRAL_BAR = "#777777"


def _item_tick(item, short) -> str:
    """``"3. facilitated motivation"`` for numbered items, the short label alone otherwise."""
    return f"{int(item)}. {short}" if isinstance(item, (int, np.integer)) else str(short)


def item_trajectory_grid(items_long, *, palette=None, ncols: int = 4,
                         title: str = "Item trajectories", panel=(4.6, 3.0)):
    """One panel per questionnaire ITEM, arms overlaid — the item-level twin of
    :func:`~eda_analysis.plotting.trajectories.trajectory_grid`.

    Takes a :func:`~eda_analysis.data.load_items` frame (``item``/``short``/``score`` long).
    Panels follow item order (1..n); each shows the per-iteration item mean ±95% CI per arm.
    The uniform detail-grid style of family ``2_questionnaires``. ``None`` if empty.
    """
    if items_long is None or items_long.empty:
        return None
    pal = palette or arm_palette(sorted(items_long.arm.unique()))
    items = list(pd.unique(items_long["item"]))          # preserves item order 1..n
    shorts = items_long["short"] if "short" in items_long.columns else items_long["item"]
    short_of = dict(zip(items_long["item"], shorts))
    fig, axes = grid(len(items), ncols=ncols, panel=panel)
    for ax, it in zip(axes, items):
        sns.lineplot(items_long[items_long["item"] == it], x="iteration", y="score",
                     hue="arm", palette=pal, marker="o", errorbar=("ci", 95), ax=ax)
        ax.set_title(_item_tick(it, short_of.get(it, it)), fontsize=9)
        ax.set_xlabel("iteration"); ax.set_ylabel("item mean (1–5)")
    figure_legend_from(axes[0], fig, title="arm")
    fig.suptitle(title, y=1.06, fontweight="bold")
    fig.tight_layout()
    return fig


def item_delta_bars(deltas, *, ncols: int = 2, group_colors: Optional[dict] = None,
                    title: str = "Item-level change vs base",
                    xlabel: str = "Δ item mean vs base",
                    legend_title: Optional[str] = None):
    """Which items drive the change — Δ vs base per item, one panel per arm.

    Takes :func:`~eda_analysis.stats.item_endpoint_deltas` (``arm, item, short, group,
    target_iter, base, target, delta``). Horizontal bars, one per item, in a SHARED cross-arm
    order (pooled Δ) with shared x-limits so panels compare; each panel title carries the arm's
    target iteration (final or best). ``group_colors`` (group -> color) colors bars by item
    group and adds a legend; otherwise bars are a neutral grey. ``None`` if empty.
    """
    if deltas is None or deltas.empty:
        return None
    order = deltas.groupby("item", sort=False)["delta"].mean().sort_values().index.tolist()
    arms = sorted(deltas.arm.unique())
    xlo = min(0.0, float(deltas["delta"].min()) * 1.05)
    xhi = max(0.0, float(deltas["delta"].max()) * 1.05)
    fig, axes = grid(len(arms), ncols=min(ncols, len(arms)), panel=(5.6, 0.34 * len(order) + 1.6))
    for ax, arm in zip(axes, arms):
        d = deltas[deltas.arm == arm].set_index("item").reindex(order)
        y = np.arange(len(d))
        colors = ([(group_colors or {}).get(g, _NEUTRAL_BAR) for g in d["group"]]
                  if group_colors else _NEUTRAL_BAR)
        ax.barh(y, d["delta"].values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels([_item_tick(i, s) for i, s in zip(d.index, d["short"])], fontsize=7.5)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlim(xlo, xhi)
        tgt = d["target_iter"].dropna()
        ax.set_title(f"{arm_label(arm)} — iter {int(tgt.iloc[0])}" if len(tgt) else arm_label(arm))
        ax.set_xlabel(xlabel)
    if group_colors:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=c, label=g) for g, c in group_colors.items()],
                   loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False,
                   fontsize=8, title=legend_title)
    fig.suptitle(title, y=1.12 if group_colors else 1.04, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Q2 specializations (face-content groups — analytical, not a validated subscale) ──
_Q2_GROUP_COLORS = None  # built lazily (constants import order) in _q2_group_colors()


def _q2_group_colors() -> dict:
    global _Q2_GROUP_COLORS
    if _Q2_GROUP_COLORS is None:
        _Q2_GROUP_COLORS = {g: _QUAL_COLORS[i % len(_QUAL_COLORS)]
                            for i, g in enumerate(Q2_ITEM_GROUPS)}
    return _Q2_GROUP_COLORS


def q2_item_delta_bars(q2_deltas, *, ncols: int = 2):
    """Which Q2 items drive the reward gain — the Q2 specialization of :func:`item_delta_bars`.

    Takes :func:`stats.q2_item_endpoint_deltas`; bars colored by the face-content item group
    (OUR analytical grouping, not a validated subscale — see ``constants.Q2_ITEM_GROUPS``). The
    reward-composition view: if the self-disclosure / warmth-closeness items climb hardest while
    non-judgment stays flat, the Q1+Q2 training reward is directly incentivizing the emotive
    drift. ``None`` if empty.
    """
    return item_delta_bars(
        q2_deltas, ncols=ncols, group_colors=_q2_group_colors(),
        title="Q2 reward composition — which alliance items drive the gain?",
        xlabel="Δ item mean vs base (1–5 scale)",
        legend_title="face-content item group (analytical, not a validated subscale)")


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
