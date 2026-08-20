"""tails.py — figures for the look-ahead TAIL audit and the API-call cost axis.

Promoted 2026-08-18 from the look-ahead paper's ``analysis/tail_audit.py`` (its ``figures/
tail_audit_fig.png`` and ``tail_audit_fig_api.png`` are the reference renders). Frames come from
:mod:`eda_analysis.tails`; nothing here touches disk (the notebook owns ``save_fig``).

- :func:`tail_audit_fig` — 3 panels on the K>0 arms: (a) share of tails that ended early (< K
  turns) by training iteration, with the ``patient closed`` share dotted; (b) the within-group
  score deviation by realized tail length (pooled); (c) P(candidate is the group argmax) for
  ended-early vs full-tail candidates by iteration, with the 1/8 chance line.
- :func:`api_calls_fig` — 2 panels, log y, all arms: (a) oracle calls (training reward) and
  (b) patient-simulator calls per training iteration.

Encoding: project arm palette (PTO cool / GRPO warm — :func:`~eda_analysis.plotting_style.arm_palette`);
:data:`K_STYLE` = K=0 solid + circle, K=5 dashed + square (the same encoding as
``plotting/lookahead.py`` and ``plotting/compute.py``, so the K contrast survives greyscale).
Grader: the training oracle (gpt-4o-mini) — training-side figures, not judge-swappable.
Censoring (GRPO_LA5 stops at iteration 5) is derived from the frames and stated in the suptitle.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plotting_style import arm_palette

__all__ = ["K_STYLE", "tail_audit_fig", "api_calls_fig"]

# K=0 solid + circle, K=5 dashed + square. Other K fall back to dash-dot + triangle.
from ._shared import K_STYLE  # noqa: F401  (one definition; see _shared)
_K_FALLBACK = {"ls": "-.", "marker": "^"}
_METHOD_ORDER = {"PTO": 0, "GRPO": 1}


def _k_of(arm: str) -> int:
    try:
        return int(str(arm).split("_LA")[1].split("_")[0])
    except (IndexError, ValueError):
        return 0


def _k_style(k: int) -> dict:
    return K_STYLE.get(int(k), _K_FALLBACK)


def _arm_order(labels) -> list:
    """PTO before GRPO, K ascending — the paper's ``ARMS`` order for any label set."""
    return sorted(set(labels), key=lambda a: (_METHOD_ORDER.get(str(a).split("_")[0], 9), _k_of(a), a))


def _censor_note(frame: pd.DataFrame, iter_col: str = "train_iter") -> str:
    """'GRPO_LA5 censored at iteration 5' when the arms' last iterations differ (else '')."""
    d = frame[frame[iter_col] != "pooled"] if frame[iter_col].dtype == object else frame
    if d.empty:
        return ""
    last = d.assign(_it=pd.to_numeric(d[iter_col], errors="coerce")).groupby("arm")["_it"].max()
    if last.nunique() <= 1:
        return ""
    mx = last.max()
    return "; ".join(f"{a} censored at iteration {int(v)}" for a, v in last.items() if v < mx)


def tail_audit_fig(by_iter: pd.DataFrame, score_by_rt: pd.DataFrame, within: pd.DataFrame, *,
                   arms: Optional[Sequence[str]] = None, palette: Optional[dict] = None,
                   figsize=(7.2, 2.7), ylim_a=(0, 0.47), ylim_b=(-0.15, 0.06),
                   title: Optional[str] = None):
    """The 3-panel tail-audit figure (the paper's ``tail_audit_fig``).

    ``by_iter`` = :func:`eda_analysis.tails.tail_audit_by_iter`, ``score_by_rt`` =
    :func:`~eda_analysis.tails.score_by_realized_turns` (its ``pooled`` rows are drawn),
    ``within`` = :func:`~eda_analysis.tails.tail_within_group`. ``arms`` defaults to every arm in
    ``by_iter`` (PTO first). Panel (a) also dots the ``patient_closed_share`` under the ended-early
    curve, so the reader sees that almost all early endings are the patient closing the session.
    Returns the fig, or ``None`` when the frames are empty.
    """
    if by_iter is None or by_iter.empty:
        return None
    arm_list = list(arms) if arms is not None else _arm_order(by_iter["arm"])
    pal = dict(arm_palette(arm_list))
    if palette:
        pal.update({k: v for k, v in palette.items() if k in pal})
    it_all = pd.to_numeric(by_iter.loc[by_iter["train_iter"] != "pooled", "train_iter"], errors="coerce")
    it_max = int(np.nanmax(it_all)) if it_all.notna().any() else 10
    xticks = range(1, it_max + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # (a) ended-early rate by iteration
    ax = axes[0]
    for arm_l in arm_list:
        d = by_iter[(by_iter["arm"] == arm_l) & (by_iter["train_iter"] != "pooled")].copy()
        if d.empty:
            continue
        d["train_iter"] = d["train_iter"].astype(int)
        d = d.sort_values("train_iter")
        ks = _k_style(_k_of(arm_l))
        ax.plot(d["train_iter"], d["ended_early_rate"], color=pal[arm_l], ls=ks["ls"], marker=ks["marker"],
                lw=1.6, ms=5, label=arm_l)
        ax.fill_between(d["train_iter"], d["ended_early_ci_lo"], d["ended_early_ci_hi"], color=pal[arm_l],
                        alpha=0.15, lw=0)
        ax.plot(d["train_iter"], d["patient_closed_share"], color=pal[arm_l], ls=":", lw=1.2, alpha=0.9,
                label=f"{arm_l}: patient closed")
    ax.set_xlabel("training iteration"); ax.set_ylabel("share of K=5 tails")
    ax.set_title("(a) tail ended early (<5 turns)", fontsize=8.5)
    if ylim_a is not None:
        ax.set_ylim(*ylim_a)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=1); ax.set_xticks(list(xticks))
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)

    # (b) within-group score deviation by realized_turns (pooled)
    ax = axes[1]
    k_max = 5
    if score_by_rt is not None and not score_by_rt.empty:
        k_max = int(score_by_rt["realized_turns"].max())
        for arm_l in arm_list:
            d = score_by_rt[(score_by_rt["arm"] == arm_l) & (score_by_rt["train_iter"] == "pooled")]
            if d.empty:
                continue
            d = d.sort_values("realized_turns")
            ks = _k_style(_k_of(arm_l))
            ax.errorbar(d["realized_turns"], d["dev_mean"],
                        yerr=[d["dev_mean"] - d["dev_lo"], d["dev_hi"] - d["dev_mean"]],
                        color=pal[arm_l], ls=ks["ls"], marker=ks["marker"], lw=1.6, ms=5, capsize=2, label=arm_l)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("realized look-ahead turns in tail"); ax.set_ylabel("score − group mean (Q1Q2 pts)")
    ax.set_title("(b) within-group reward by ending", fontsize=8.5)
    ticks = list(range(0, k_max + 1))
    ax.set_xticks(ticks)
    lab = []
    for k in ticks:
        if k == 0:
            lab.append("0\nno tail")
        elif k == k_max:
            lab.append(f"{k}\nfull")
        elif k % 2 == 1:
            lab.append(f"{k}\npatient\nclosed")
        else:
            lab.append(f"{k}\nther.\nend")
    ax.set_xticklabels(lab, fontsize=6.5)
    if ylim_b is not None:
        ax.set_ylim(*ylim_b)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.tick_params(axis="y", labelsize=8); ax.grid(True, alpha=0.3)

    # (c) P(chosen | ended early) vs P(chosen | full) by iteration
    ax = axes[2]
    if within is not None and not within.empty:
        for arm_l in arm_list:
            d = within[(within["arm"] == arm_l) & (within["train_iter"] != "pooled")].copy()
            if d.empty:
                continue
            d["train_iter"] = d["train_iter"].astype(int)
            d = d.sort_values("train_iter")
            ks = _k_style(_k_of(arm_l))
            ax.plot(d["train_iter"], d["p_chosen_given_ee"], color=pal[arm_l], ls=ks["ls"], marker=ks["marker"],
                    lw=1.6, ms=5, label=f"{arm_l}: ended early")
            ax.plot(d["train_iter"], d["p_chosen_given_full"], color=pal[arm_l], ls="-", marker="o", mfc="white",
                    lw=1.2, ms=4.5, label=f"{arm_l}: full tail")
    ax.axhline(1 / 8, color="grey", lw=0.8, ls=":")
    ax.text(it_max + 0.4, 1 / 8, "1/8", fontsize=7, va="center", ha="right", color="grey")
    ax.set_xlabel("training iteration"); ax.set_ylabel("P(candidate is group argmax)")
    ax.set_title("(c) P(argmax) by tail ending", fontsize=8.5)
    ax.set_ylim(0, None); ax.set_xticks(list(xticks)); ax.legend(fontsize=6, frameon=False, ncol=1)
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)

    if title is None:
        note = _censor_note(by_iter)
        title = ("Look-ahead tails (K=5 arms) — grader: training oracle (gpt-4o-mini)"
                 + (f"; {note}" if note else ""))
    fig.suptitle(title, fontsize=8.5, y=1.02)
    fig.tight_layout()
    return fig


def api_calls_fig(api: pd.DataFrame, *, arms: Optional[Sequence[str]] = None,
                  palette: Optional[dict] = None, figsize=(7.2, 2.7), title: Optional[str] = None):
    """The 2-panel API-call figure (the paper's ``tail_audit_fig_api``): per training iteration,
    (a) oracle calls (training reward) and (b) patient-simulator calls, log y, one line per arm
    (``row_kind == 'iteration'`` rows only — the final eval pass is not a training iteration).

    ``api`` = :func:`eda_analysis.tails.api_calls`. Returns the fig, or ``None`` when empty.
    """
    if api is None or api.empty:
        return None
    arm_list = list(arms) if arms is not None else _arm_order(api["arm"])
    pal = dict(arm_palette(arm_list))
    if palette:
        pal.update({k: v for k, v in palette.items() if k in pal})
    it = api.loc[api["row_kind"] == "iteration", "train_iter"]
    it_max = int(it.max()) if len(it) else 10

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, col, ttl in ((axes[0], "oracle_calls_train", "(a) oracle calls (training reward)"),
                         (axes[1], "patient_calls_total", "(b) patient-simulator calls")):
        for arm_l in arm_list:
            d = api[(api["arm"] == arm_l) & (api["row_kind"] == "iteration")].sort_values("train_iter")
            if d.empty:
                continue
            st = _k_style(_k_of(arm_l))
            ax.plot(d["train_iter"], d[col], color=pal[arm_l], ls=st["ls"], marker=st["marker"], lw=1.6, ms=5,
                    label=arm_l)
        ax.set_yscale("log"); ax.set_xlabel("training iteration"); ax.set_ylabel("API calls per iteration (log)")
        ax.set_title(ttl, fontsize=9); ax.set_xticks(list(range(1, it_max + 1)))
        ax.tick_params(labelsize=8); ax.grid(True, which="both", alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=max(1, len(lab)), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    if title is None:
        note = _censor_note(api[api["row_kind"] == "iteration"])
        title = ("API calls per training iteration = eval convs of pi_{n-1} + training-time calls "
                 "(GRPO rescaled to its true step count)" + (f"; {note}" if note else ""))
    fig.suptitle(title, fontsize=8.5, y=1.02)
    fig.tight_layout()
    return fig
