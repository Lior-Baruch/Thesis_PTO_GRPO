"""compute.py — figures on the COMPUTE axis, the x-axis this project never had.

Every trajectory figure elsewhere plots score against **iteration**. That is a fair axis only if
an iteration costs the same everywhere, and it does not: a K=5 GRPO step costs ~1.9x a K=0 step,
and a whole PTO iteration costs a fraction of a GRPO one. Re-drawn against cumulative GPU-hours
the same curves tell a materially different story — arms that "stopped early" turn out to have
spent the same money, and the ordering of two arms can invert.

- :func:`compute_trajectory` — the headline: score vs cumulative GPU-h, one line per arm, with
  iteration numbers annotated so the reader can move between the two axes.
- :func:`budget_sweep_plot` — the lever's paired effect size as a function of budget, with the
  zero line and the crossover visible.
- :func:`cost_breakdown` — where each arm's hours actually go (generate / build / train), which
  is what explains the cost gap rather than asserting it.

Contract as everywhere in ``plotting``: takes already-built tidy frames, never touches disk,
returns a ``fig`` (the notebook owns ``save_fig``), returns ``None`` when the arms are absent.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import LOWER_IS_BETTER, arm_label, display_label
from ..plotting_style import arm_palette

# Solid = K=0, dashed = K=5 — same encoding as plotting/lookahead.py, so the two families read
# together and the K contrast survives greyscale printing.
_K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}


def _k_of(arm: str) -> int:
    return 5 if arm.endswith("LA5") else 0


def compute_trajectory(by_compute: pd.DataFrame, *, metric: str = "Q1Q2",
                       arms: Optional[Sequence[str]] = None,
                       annotate_iters: bool = True,
                       figsize=(9.0, 5.2)):
    """Score vs **cumulative GPU-hours**, one line per arm.

    ``by_compute`` is :func:`eda_analysis.compute.score_by_compute` output
    (``arm, iteration, cum_gpu_h, mean, sem``).

    Iteration numbers are annotated on the markers because the whole point of the figure is that
    the two axes disagree: a reader who knows the iteration-indexed figure needs to see *which*
    iteration each point is to register that e.g. one arm's 10 iterations and another's 5 land on
    the same x. Points are drawn at their own cost, so unequal spacing along x IS the finding.
    """
    if by_compute is None or by_compute.empty:
        return None
    d = by_compute[by_compute["metric"] == metric] if "metric" in by_compute.columns else by_compute
    if arms is not None:
        d = d[d.arm.isin(list(arms))]
    d = d.dropna(subset=["cum_gpu_h", "mean"])
    if d.empty:
        return None

    pal = arm_palette(sorted(d.arm.unique()))
    fig, ax = plt.subplots(figsize=figsize)
    for arm, g in d.groupby("arm"):
        g = g.sort_values("cum_gpu_h")
        st = _K_STYLE.get(_k_of(arm), _K_STYLE[0])
        ax.errorbar(g.cum_gpu_h, g["mean"], yerr=g.get("sem"),
                    label=arm_label(arm), color=pal.get(arm), lw=1.9, ms=5.5,
                    capsize=2.5, elinewidth=0.9, alpha=0.95, **st)
        if annotate_iters:
            for x, y, it in zip(g.cum_gpu_h, g["mean"], g.iteration):
                ax.annotate(str(int(it)), (x, y), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color=pal.get(arm), alpha=0.85)

    ax.set_xlabel("cumulative GPU-hours (generate + build + train, wall-clock on the training host)")
    lo = " — lower is better" if metric in LOWER_IS_BETTER else ""
    ax.set_ylabel(f"{display_label(metric)}{lo}")
    ax.set_title(f"{display_label(metric)} against COMPUTE, not iteration"
                 "\nmarker labels are iteration numbers", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def budget_sweep_plot(sweep: pd.DataFrame, *, label_a: str = "arm A", label_b: str = "arm B",
                      figsize=(8.0, 4.4)):
    """Paired effect size (``dz``) of A-over-B as a function of budget, with the crossover shown.

    ``sweep`` is :func:`eda_analysis.compute.budget_sweep` output. Points are coloured by whether
    they clear p < .05, because an unmarked sign flip across a non-significant middle would read
    as a cleaner crossover than the data supports.
    """
    if sweep is None or sweep.empty:
        return None
    d = sweep.sort_values("budget_gpu_h")
    fig, ax = plt.subplots(figsize=figsize)
    sig = d.p < 0.05
    ax.axhline(0, color="0.35", lw=1.0, ls="--", zorder=1)
    ax.plot(d.budget_gpu_h, d.dz, "-", color="0.45", lw=1.6, zorder=2)
    ax.scatter(d.budget_gpu_h[sig], d.dz[sig], s=58, color="#1b6ca8",
               zorder=3, label="p < .05")
    ax.scatter(d.budget_gpu_h[~sig], d.dz[~sig], s=58, facecolors="none",
               edgecolors="#1b6ca8", zorder=3, label="n.s.")
    for x, y, ia, ib in zip(d.budget_gpu_h, d.dz, d.best_iter_a, d.best_iter_b):
        ax.annotate(f"{int(ia)}v{int(ib)}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7, color="0.30")
    ax.set_xlabel("budget (cumulative GPU-hours)")
    ax.set_ylabel(f"paired dz — {label_a} minus {label_b}")
    ax.set_title(f"Is {label_a} worth it? Best checkpoint within budget, each arm"
                 "\nmarker labels are the iterations compared", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def cost_breakdown(summary: pd.DataFrame, *, figsize=(7.6, 4.0)):
    """Stacked GPU-hours per arm by phase — where the money actually goes.

    ``summary`` is :func:`eda_analysis.compute.compute_summary` output. The iteration count is
    printed on each bar, since "cheaper per iteration" and "cheaper overall" are different claims
    and this figure exists to keep them apart.
    """
    if summary is None or summary.empty:
        return None
    d = summary.sort_values("total_gpu_h")
    phases = [("gen_h", "generate", "#7fb3d5"), ("build_h", "build (PTO)", "#f5b041"),
              ("train_h", "train", "#5d6d7e")]
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(d))
    y = np.arange(len(d))
    for col, lab, colour in phases:
        vals = d[col].to_numpy() if col in d.columns else np.zeros(len(d))
        ax.barh(y, vals, left=left, label=lab, color=colour, height=0.62)
        left = left + vals
    ax.set_yticks(y)
    ax.set_yticklabels([arm_label(a) for a in d.arm])
    for i, (tot, n_it) in enumerate(zip(d.total_gpu_h, d.n_iters)):
        ax.annotate(f"{tot:.1f} h  ({int(n_it)} iters)", (tot, i),
                    textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8.5, color="0.25")
    ax.set_xlabel("GPU-hours (wall-clock on the training host)")
    ax.set_title("Cost per arm by phase", fontsize=11)
    ax.set_xlim(0, float(d.total_gpu_h.max()) * 1.22)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig
