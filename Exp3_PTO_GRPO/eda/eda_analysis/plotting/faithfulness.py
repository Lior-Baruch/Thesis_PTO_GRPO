"""plotting/faithfulness.py — the reward-faithfulness figure for :mod:`eda_analysis.faithfulness`.
Promoted 2026-08-18 from the look-ahead paper's ``analysis/reward_faithfulness.py``
(``figures/reward_faithfulness_fig.png`` + ``_fig_heldout.png``).

:func:`faithfulness_fig` — three panels, called ONCE PER EVAL GRADER (the proxy is always the
training oracle; only the eval side changes): (a) PTO and (b) GRPO agreement vs prefix length
n_turns, K=0 solid/circle vs K=5 dashed/square, ribbons = 95% cluster-bootstrap CI, pooled over
iterations (GRPO on the 1–5 series — GRPO_LA5 is right-censored at 5 — with full-support GRPO_LA0
as a faint dotted reference); (c) the matched-policy (train_iter 1, base policy) K0 − K5
difference per bin with CI — NEGATIVE = look-ahead more faithful.

Contract as everywhere in ``plotting``: takes the tidy frames, never touches disk, returns a
``fig`` (the notebook owns ``save_fig``).
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plotting_style import arm_palette

__all__ = ["K_STYLE", "faithfulness_fig"]

# Solid + circle = K=0, dashed + square = K=5 (survives greyscale; the palette carries the method).
K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}
_METHODS = ["PTO", "GRPO"]


def faithfulness_fig(curve: pd.DataFrame, matched: pd.DataFrame, judge_label: str, *,
                     grader_display: Optional[str] = None, methods=_METHODS,
                     main_iters: Optional[dict] = None, figsize=(7.2, 2.75)):
    """Panels a,b: agreement vs n_turns per method; panel c: matched-policy K0 − K5 per bin.

    ``curve`` = :func:`eda_analysis.faithfulness.faithfulness_curve` (long, all graders);
    ``matched`` = the per-bin frame from :func:`eda_analysis.faithfulness.matched_policy`;
    ``judge_label`` selects the eval-side grader's rows (e.g. ``"gpt-4o-mini"``,
    ``"claude-haiku-4-5"``); ``grader_display`` = the long label for the suptitle (default via
    :func:`eda_analysis.faithfulness.judge_display`). ``main_iters`` maps method → the ``iters``
    series label to draw (default PTO ``"1-10"``, GRPO ``"1-5"``)."""
    from ..faithfulness import judge_display
    gl = grader_display or judge_display(judge_label)
    main_iters = main_iters or {"PTO": "1-10", "GRPO": "1-5"}
    arms = [f"{m}_LA{k}" for m in methods for k in (0, 5)]
    pal = arm_palette(arms)
    cj = curve[curve["judge"] == judge_label]
    mj = matched[matched["judge"] == judge_label]
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    for ax, m in zip(axes[:2], methods):
        mi = main_iters.get(m, "1-10")
        for K in (0, 5):
            arm = f"{m}_LA{K}"
            d = cj[(cj["arm"] == arm) & (cj["iters"] == mi) & (cj["n_turns"].str.isdigit())].copy()
            if d.empty:
                continue
            d["nt"] = d["n_turns"].astype(int)
            d = d.sort_values("nt")
            st = K_STYLE[K]
            ax.fill_between(d["nt"], d["ci_lo"], d["ci_hi"], color=pal[arm], alpha=0.18, lw=0)
            ax.plot(d["nt"], d["agreement"], color=pal[arm], ls=st["ls"], marker=st["marker"], ms=4.5, lw=1.6,
                    label=f"{arm} (iters {mi})")
        # full-support K=0 as a faint reference wherever the main series is a censored subset
        full = cj[(cj["arm"] == f"{m}_LA0") & (cj["iters"] != mi) & (cj["n_turns"].str.isdigit())].copy()
        if not full.empty:
            full_lab = full["iters"].iloc[0]
            full = full[full["iters"] == full_lab]
            full["nt"] = full["n_turns"].astype(int)
            full = full.sort_values("nt")
            ax.plot(full["nt"], full["agreement"], color=pal[f"{m}_LA0"], ls=":", lw=1.2, alpha=0.8,
                    label=f"{m}_LA0 (iters {full_lab})")
        ax.set_title(f"{m}: proxy vs full-conv eval", fontsize=9)
        ax.set_xlabel("prefix length n_turns (utterances)", fontsize=8)
        ax.set_ylabel("pairwise sign-agreement (0.5 = chance)", fontsize=8)
        ax.set_ylim(0.62, 1.0)
        ax.set_xlim(10, 52)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=6.8, loc="lower left", frameon=True)
        ax.grid(True, alpha=0.35)
    ax = axes[2]
    for m, off in zip(methods, (-0.35, 0.35)):
        d = mj[(mj["method"] == m) & (mj["cut"] == "train_iter_1") & (mj["n_turns"].str.isdigit())].copy()
        if d.empty:
            continue
        d["nt"] = d["n_turns"].astype(int)
        d = d.sort_values("nt")
        col = pal[f"{m}_LA0"]
        yerr = np.vstack([d["delta_K0_minus_K5"] - d["d_lo"], d["d_hi"] - d["delta_K0_minus_K5"]])
        ax.errorbar(d["nt"] + off, d["delta_K0_minus_K5"], yerr=yerr, color=col, fmt="o-" if m == "PTO" else "s-",
                    ms=4, lw=1.3, elinewidth=0.8, capsize=1.2, label=f"{m} (train_iter 1, base policy)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("matched policy: K0 − K5", fontsize=9)
    ax.set_xlabel("prefix length n_turns (utterances)", fontsize=8)
    ax.set_ylabel("Δ agreement, K0 − K5 (− = K5 more faithful)", fontsize=8)
    ax.set_xlim(10, 52)
    ax.set_ylim(-0.3, 0.3)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.5, loc="upper left", frameon=True)
    ax.grid(True, alpha=0.35)
    fig.suptitle(f"Training-reward faithfulness — eval graded by the {gl}; proxy = training oracle",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    return fig
