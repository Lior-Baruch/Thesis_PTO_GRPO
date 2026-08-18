"""plotting/replication.py — the two figures of the ICLR-claims replication family.

- :func:`shape_fig` — session shape by iteration, three panels (conversation length / therapist
  turn length / questions per turn), all four arms, mean +- SE over the 96 personas; deterministic
  text metrics, so grader-free. Takes the :func:`eda_analysis.replication.session_shape_levels`
  frame.
- :func:`sd_fig` — the ICLR "K=5 is more stable" claim: across-persona SD of the training reward
  (Q1Q2) by iteration under each grader (one panel per grader, never averaged) plus an SD-vs-mean
  scatter over the trained states (filled = primary oracle, open = held-out judge). Takes the
  :func:`eda_analysis.replication.sd_by_iter` frame.

Style: PTO cool / GRPO warm (:func:`eda_analysis.plotting_style.arm_palette`); K=0 solid line +
filled circle, K=5 dashed line + open square (:data:`K_STYLE`) so the K contrast survives greyscale.
Iteration 0 = the two independent base draws. GRPO_LA5 stops at iteration 5 (right-censored).

Contract as everywhere in ``plotting``: takes frames, never touches disk, returns a ``fig`` (the
notebook owns ``save_fig``). Promoted 2026-08-18 from
``papers/2026_lookahead_pto_grpo/analysis/session_shape_stability.py`` (fig_shape + fig_sd).
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from ..constants import arm_label, judge_dirname
from ..plotting_style import arm_palette

__all__ = ["K_STYLE", "shape_fig", "sd_fig"]

# K=0 solid + circle, K=5 dashed + square — survives greyscale printing; colour carries the arm.
K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}
_FOUR_ARMS = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]


def _k_of(arm: str) -> int:
    return 5 if arm.endswith("LA5") else 0


def _traj(ax, lvl: pd.DataFrame, metric: str, ylabel: str, title: str, arms, pal):
    for arm in arms:
        d = lvl[lvl["arm"] == arm].sort_values("iteration")
        if d.empty:
            continue
        k = _k_of(arm); st = K_STYLE[k]
        m, s = d[f"{metric}_mean"].to_numpy(), d[f"{metric}_sem"].to_numpy()
        ax.fill_between(d["iteration"], m - s, m + s, color=pal[arm], alpha=0.15, lw=0)
        ax.plot(d["iteration"], m, ls=st["ls"], marker=st["marker"], ms=5, lw=1.7, color=pal[arm],
                label=arm_label(arm), markerfacecolor=pal[arm] if k == 0 else "white", markeredgewidth=1.4)
    ax.set_xlabel("iteration"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.set_xticks(range(0, 11)); ax.grid(True, alpha=0.35)


def shape_fig(levels: pd.DataFrame, *, arms: Sequence[str] = tuple(_FOUR_ARMS),
              palette: Optional[dict] = None):
    """Session shape by iteration — deterministic text metrics (mean +- SE over 96 personas;
    grader-free): conversation length (utterances), therapist turn length (chars) and questions per
    therapist turn, one line per arm. ``levels`` = :func:`eda_analysis.replication.session_shape_levels`
    (per (arm, iteration) ``<metric>_mean`` / ``<metric>_sem``). Returns the ``fig``."""
    pal = palette or arm_palette(list(arms))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    _traj(axes[0], levels, "conv_len", "utterances / conversation", "Conversation length", arms, pal)
    _traj(axes[1], levels, "mean_turn_len", "chars / therapist turn", "Therapist turn length", arms, pal)
    _traj(axes[2], levels, "q_per_turn", "'?' / therapist turn", "Questions per turn", arms, pal)
    hnd, lab = axes[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
    axes[2].set_ylim(bottom=0)
    fig.suptitle("Session shape by iteration — deterministic text metrics (mean ± SE over 96 personas; grader-free)",
                 fontsize=9.5, y=1.09)
    fig.tight_layout()
    return fig


def sd_fig(sd: pd.DataFrame, summary: Optional[pd.DataFrame] = None, *, metric: str = "Q1Q2",
           judges: Optional[Sequence[str]] = None, arms: Sequence[str] = tuple(_FOUR_ARMS),
           palette: Optional[dict] = None):
    """Across-persona SD of the training reward by iteration under each grader + SD vs mean.

    ``sd`` = :func:`eda_analysis.replication.sd_by_iter` (columns ``judge, metric, arm, iteration,
    mean, sd``). ``summary`` (optional) = :func:`eda_analysis.replication.sd_summary`; when given,
    each grader's lowest-SD trained state (``min_sd_arm`` / ``min_sd_iteration``) is ringed in its
    panel. ``judges`` = the two ``judge`` labels to panel, primary first (default: the frame's order
    with the primary oracle's short label first if present); the first is drawn FILLED in the
    scatter, the second OPEN. Two SD panels share nothing (graders on different levels; never
    averaged); the third panel pools the trained states (iteration > 0) of both graders. Returns the
    ``fig``."""
    pal = palette or arm_palette(list(arms))
    present = list(dict.fromkeys(sd["judge"].tolist()))
    if judges is None:
        prim = judge_dirname("")
        judges = ([prim] if prim in present else []) + [j for j in present if j != prim]
    judges = list(judges)[:2]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1.1, 0.95]})
    for i, (ax, judge) in enumerate(zip(axes[:2], judges)):
        d0 = sd[(sd["judge"] == judge) & (sd["metric"] == metric)]
        for arm in arms:
            d = d0[d0["arm"] == arm].sort_values("iteration"); k = _k_of(arm); st = K_STYLE[k]
            ax.plot(d["iteration"], d["sd"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.7, color=pal[arm],
                    label=arm_label(arm), markerfacecolor=pal[arm] if k == 0 else "white", markeredgewidth=1.4)
        if summary is not None:
            srow = summary[(summary["judge"] == judge) & (summary["metric"] == metric)]
            if len(srow):
                ax.scatter([srow["min_sd_iteration"].iloc[0]], [srow["min_sd"].iloc[0]], s=110,
                           facecolors="none", edgecolors="#333333", lw=1.2, zorder=4, label="_min SD")
        ax.set_xlabel("iteration"); ax.set_ylabel(f"SD of {metric} over 96 personas" if i == 0 else "")
        ax.set_title(f"grader: {judge}", fontsize=10); ax.set_xticks(range(0, 11)); ax.grid(True, alpha=0.35)
    hnd, lab = axes[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
    ax = axes[2]
    for judge, mk_fill in zip(judges, (True, False)):
        d0 = sd[(sd["judge"] == judge) & (sd["metric"] == metric) & (sd["iteration"] > 0)]
        for arm in arms:
            d = d0[d0["arm"] == arm]; k = _k_of(arm); st = K_STYLE[k]
            ax.scatter(d["mean"], d["sd"], marker=st["marker"], s=22, color=pal[arm], lw=1.1,
                       facecolors=pal[arm] if mk_fill else "white", edgecolors=pal[arm], zorder=3)
    ax.set_xlabel(f"mean {metric} (model state)"); ax.set_ylabel(f"SD of {metric} over 96 personas")
    ax.set_title("SD vs mean (iters 1..N)", fontsize=10)
    if len(judges) == 2:
        ax.text(0.03, 0.05, f"filled = {judges[0]}\nopen = {judges[1]}", transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left", color="#333333")
    ax.grid(True, alpha=0.35)
    fig.suptitle(f"Across-persona SD of the training reward ({metric}) — the ICLR 'K=5 is more stable' claim",
                 fontsize=9.5, y=1.09)
    fig.tight_layout()
    return fig
