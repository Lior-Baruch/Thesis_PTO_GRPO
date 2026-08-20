"""plotting/crossgen.py — the Exp1-under-two-graders figure (the ICLR replication link).

:func:`crossgen_fig` — Final = mean(Q1, Q2) by PTO iteration for Exp1's K=0 (solid, circles) and
K=5 (dashed, squares) arms, untrained Base as a dotted line, 95% percentile-bootstrap CI bands of
the mean; one panel per grader — GPT-3.5 (the original ICLR oracle) and gpt-4o-mini (the Exp3
oracle re-scoring the SAME transcripts). **Separate y-axes per panel**: the graders sit on
different levels (gpt-4o-mini reads ~0.19-0.43 higher) and are never averaged.
``layout="wide"`` = the two panels side by side (the paper's ``crossgen_exp1_fig``);
``layout="col"`` = the same two panels stacked for a single ACL column (``crossgen_exp1_fig_col``;
same data, narrow-width fonts).

Takes the per-conversation frames from :mod:`eda_analysis.crossgen` (``load_exp1_gpt35`` /
``load_crossgen``), NOT the levels table, because the CI bands need the 96 per-conversation
values. Bootstrap seeded with :data:`eda_analysis.constants.BOOT_SEED` (the paper generator used
seed 0 — the bands differ imperceptibly, the lines are identical). Colours: the PTO arm palette
(K=0 = ``PTO_LA0`` blue, K=5 = ``PTO_LA5`` sky blue) — Exp1 has only PTO.

Contract as everywhere in ``plotting``: takes frames, never touches disk, returns a ``fig`` (the
notebook owns ``save_fig``). Promoted 2026-08-18 from
``papers/2026_lookahead_pto_grpo/analysis/crossgen_exp1.py`` (§3 fig + fig_col).
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import BOOT_SEED
from ..plotting_style import arm_palette

__all__ = ["K_STYLE", "crossgen_fig"]

# K=0 solid + circle, K=5 dashed + square — survives greyscale printing; colour carries the arm.
from ._shared import K_STYLE  # noqa: F401  (one definition; see _shared)

_ITERS = list(range(1, 8))
_TITLES = {"gpt-3.5": "GPT-3.5 (original ICLR oracle)",
           "gpt-4o-mini": "gpt-4o-mini (Exp3 oracle, same transcripts)"}


def _series(df: pd.DataFrame, k: int, metric: str, iters: Sequence[int], n_boot: int, seed: int):
    ys, lo, hi = [], [], []
    for it in iters:
        v = df.loc[df["model"] == f"Exp1_LA{k}_I{it}", metric].dropna().to_numpy()
        if v.size == 0:
            ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        ys.append(v.mean())
        rng = np.random.default_rng(seed)
        b = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
        lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    return ys, lo, hi


def crossgen_fig(gpt35, crossgen: Optional[pd.DataFrame] = None, *, layout: str = "wide",
                 metric: str = "Final", iters: Sequence[int] = tuple(_ITERS),
                 n_boot: int = 2000, seed: int = BOOT_SEED, palette: Optional[dict] = None,
                 titles: Optional[dict] = None):
    """Exp1 (ICLR 2025) PTO models by iteration under two graders — see the module docstring.

    ``gpt35`` / ``crossgen`` = the per-conversation frames (``model, arm, iteration, conv_index,
    Q1, Q2, Final``) from :func:`eda_analysis.crossgen.load_exp1_gpt35` /
    :func:`eda_analysis.crossgen.load_crossgen` — or pass the :func:`eda_analysis.crossgen.crossgen_all`
    dict as the first argument. ``layout`` ``"wide"`` (1x2, 7.0x2.9 in) or ``"col"`` (2x1,
    3.4x4.0 in). Returns the ``fig``.
    """
    if isinstance(gpt35, dict) and crossgen is None:          # the crossgen_all() bundle
        gpt35, crossgen = gpt35["gpt35"], gpt35["crossgen"]
    if layout not in ("wide", "col"):
        raise ValueError(f"layout must be 'wide' or 'col', got {layout!r}")
    pal = palette or arm_palette(["PTO_LA0", "PTO_LA5"])
    col = {0: pal["PTO_LA0"], 5: pal["PTO_LA5"]}
    titles = {**_TITLES, **(titles or {})}
    panels = ((gpt35, titles["gpt-3.5"]), (crossgen, titles["gpt-4o-mini"]))
    wide = layout == "wide"
    if wide:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
        ms, lw, tfs, yfs, tick = 5.5, 1.7, 10, None, None
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True)
        ms, lw, tfs, yfs, tick = 4.5, 1.4, 9, 7.5, 7.5
    for ax, (df, title) in zip(axes, panels):
        for k in (0, 5):
            ys, lo, hi = _series(df, k, metric, iters, n_boot, seed)
            st = K_STYLE[k]
            ax.plot(list(iters), ys, ls=st["ls"], marker=st["marker"], ms=ms, lw=lw, color=col[k],
                    label=f"PTO K={k}")
            ax.fill_between(list(iters), lo, hi, color=col[k], alpha=0.15, lw=0)
        base = df.loc[df["model"] == "Exp1_Base", metric].mean()
        ax.axhline(base, ls=":", lw=1.5 if wide else 1.3, color="#555555", label="Base (Llama-2-7B)")
        ax.set_title(title, fontsize=tfs)
        ax.set_xticks(list(iters))
        ax.grid(True, alpha=0.3)
        ylab = f"{metric} = mean(Q1, Q2)  [1-5]" if metric == "Final" else f"{metric}  [1-5]"
        if yfs is None:
            ax.set_ylabel(ylab)
        else:
            ax.set_ylabel(ylab, fontsize=yfs)
            ax.tick_params(labelsize=tick)
    if wide:
        for ax in axes:
            ax.set_xlabel("PTO iteration")
    else:
        axes[1].set_xlabel("PTO iteration", fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    lab = [l + ("  (bands: 95% bootstrap CI)" if l.startswith("PTO K=5") else "") for l in lab]
    if wide:
        fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.10))
        fig.suptitle("Exp1 (ICLR 2025) conversations under two graders — K=0 solid, K=5 dashed",
                     fontsize=10, y=1.02)
    else:
        fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, -0.06), columnspacing=1.2, handlelength=2.2)
        fig.suptitle("Exp1 (ICLR 2025) conversations under two graders\nK=0 solid, K=5 dashed",
                     fontsize=8.5)
    fig.tight_layout()
    return fig
