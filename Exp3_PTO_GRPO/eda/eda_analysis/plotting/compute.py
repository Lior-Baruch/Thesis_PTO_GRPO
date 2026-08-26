"""compute.py — figures on the COMPUTE axis, the x-axis this project never had.

Every trajectory figure elsewhere plots score against **iteration**. That is a fair axis only if
an iteration costs the same everywhere, and it does not: a K=5 GRPO step costs ~1.9x a K=0 step,
and a whole PTO iteration costs a fraction of a GRPO one. Re-drawn against cumulative GPU-hours
the same curves tell a materially different story — arms that "stopped early" turn out to have
spent the same money, and the ordering of two arms can invert.

Single-judge figures (the tracked ``compute/cost`` set):

- :func:`compute_trajectory` — score vs cumulative GPU-h, one line per arm, iteration numbers
  annotated so the reader can move between the two axes.
- :func:`budget_sweep_plot` — one contrast's paired effect size as a function of budget.
- :func:`cost_breakdown_by_arm` — stacked GPU-h per arm by phase (generate / build / train).

Both-graders figures (promoted 2026-08-18 from the look-ahead paper's ``compute_axis.py``
generator, ``papers/2026_lookahead_pto_grpo/analysis/``; the paper's PNGs are the visual
fixture):

- :func:`budget_sweep_grid` — ``fig_budget_sweep``: 2x2, rows = grader, cols = method; y = the
  paired Q1Q2 delta **K5 - K0** between best-within-budget checkpoints (above zero = look-ahead
  ahead — the tracked-EDA sign, NOT the paper's ``+ => K=0 higher`` table convention), bootstrap
  CI bars, hollow markers = Holm p >= .05, labels ``I<K5>/I<K0>`` = the selected iterations.
- :func:`trajectory_by_compute` — ``fig_trajectory`` (two grader panels side by side) and its
  ``layout="col"`` variant ``fig_trajectory_col`` (the same two panels stacked for a single ACL
  column): Q1Q2 mean ± SEM over the 96 personas vs cumulative GPU-h, four arms, iteration 0 at
  0 h, last point labelled with its iteration.
- :func:`cost_breakdown_by_iteration` — ``fig_breakdown``: stacked generate / build / train
  GPU-h per iteration, one panel per arm, with PTO_LA5's known mtime caveat annotated (its
  generation of I1-I5 lands in I6 because its conversation mtimes were batch-flushed) and any
  short arm's right-censoring annotated at ITS OWN last iteration, read off the frame.
- :func:`cost_breakdown` dispatches on its input: an :func:`~eda_analysis.compute.iteration_compute`
  frame (has ``iteration``) -> :func:`cost_breakdown_by_iteration`; a
  :func:`~eda_analysis.compute.compute_summary` frame -> :func:`cost_breakdown_by_arm` (the
  pre-2026-08-18 behaviour, so old call sites keep working).

K encoding everywhere: K=0 solid + circle, K=5 dashed + square (:data:`K_STYLE`) — the same
encoding as ``plotting/lookahead.py``, so the families read together and the K contrast survives
greyscale printing. Colours = :func:`~eda_analysis.plotting_style.arm_palette` (PTO cool / GRPO
warm, Okabe-Ito).

Contract as everywhere in ``plotting``: takes already-built tidy frames, never touches disk,
returns a ``fig`` (the notebook owns ``save_fig``), returns ``None`` when the arms are absent.
"""

import re
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ..constants import LOWER_IS_BETTER, arm_label, display_label
from ..plotting_style import arm_palette

__all__ = [
    "K_STYLE", "BREAKDOWN_NOTES", "compute_trajectory", "budget_sweep_plot", "cost_breakdown",
    "cost_breakdown_by_arm", "cost_breakdown_by_iteration",
    "budget_sweep_grid", "trajectory_by_compute",
]

# Solid = K=0, dashed = K=5 — the one definition lives in ._shared. (This used to try/except an
# import of ``plotting_style.K_STYLE``, a name that module never defined, so the fallback was the
# only live branch; _shared is where the deferral actually landed.)
from ._shared import K_STYLE  # noqa: F401
from ..constants import k_of as _k_of_canonical  # noqa: E402


def _k_of(arm: str) -> int:
    """Re-export of :func:`eda_analysis.constants.k_of` (THE canonical arm parse)."""
    return _k_of_canonical(arm)


def _kstyle(arm: str) -> dict:
    return K_STYLE.get(_k_of(arm), K_STYLE[0])


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
        st = _kstyle(arm)
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


def cost_breakdown_by_arm(summary: pd.DataFrame, *, figsize=(7.6, 4.0)):
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


#: The known per-iteration cost ARTEFACTS, annotated on the breakdown panels (the paper's
#: ``fig_breakdown``). Keyed by arm; ``(x, y_frac_of_ymax, text)``. Left as data so a future run
#: without the artefact simply drops the entry.
#:
#: ⚠ Censoring is NOT in here. It used to be - ``"right-censored (stopped at I5)"`` pinned at
#: ``x=7.9`` - and it went stale four iterations before anyone noticed, because where an arm stops
#: is a property of the data, not of a caption. :func:`cost_breakdown_by_iteration` now DERIVES
#: that annotation for any arm ending before the shared x-axis, at that arm's own last iteration.
BREAKDOWN_NOTES = {
    "PTO_LA5": (3.2, 0.53, "gen of I1–I5 lands in I6\n(flushed conv mtimes)"),
}


def cost_breakdown_by_iteration(comp: pd.DataFrame, *, arms: Optional[Sequence[str]] = None,
                                figsize=(7.2, 2.9), notes: Optional[dict] = None,
                                max_iter: Optional[int] = None):
    """Stacked generate / build / train GPU-h **per iteration**, one panel per arm (the paper's
    ``fig_breakdown``).

    ``comp`` is :func:`eda_analysis.compute.iteration_compute` output. Phase encoding: train =
    solid arm colour; build = arm colour, light; generate = white + arm-colour hatch. Each panel's
    title carries the arm's total (``Σ h``). ``notes`` (default :data:`BREAKDOWN_NOTES`)
    annotates the known artefacts — today just PTO_LA5's I1-I5 generation landing in I6
    (batch-flushed conv mtimes). All panels share the x-ticks ``1..max_iter`` (default = the
    largest trained iteration across the arms) so the censoring is VISIBLE as empty slots rather
    than a narrower axis, and any arm that stops short is labelled "right-censored (stops at
    I<n>)" over its own empty slots — ``n`` read off ``comp``, never hard-coded.
    """
    if comp is None or comp.empty:
        return None
    d_all = comp[comp.iteration > 0]
    order = list(arms) if arms is not None else [a for a in ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5")
                                                    if a in set(d_all.arm)]
    order += [a for a in sorted(d_all.arm.unique()) if a not in order] if arms is None else []
    order = [a for a in order if a in set(d_all.arm)]
    if not order:
        return None
    notes = BREAKDOWN_NOTES if notes is None else notes
    pal = arm_palette(order)
    n_it = int(max_iter or d_all.iteration.max())
    fig, axes = plt.subplots(1, len(order), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    ymax = float(d_all.gpu_h.max())
    for ax, arm in zip(axes, order):
        d = d_all[d_all.arm == arm].sort_values("iteration")
        bottom = np.zeros(len(d))
        ax.bar(d["iteration"], d["gen_h"], bottom=bottom, facecolor="white", edgecolor=pal[arm],
               hatch="////", linewidth=0.5, width=0.8)
        bottom += d["gen_h"].values
        ax.bar(d["iteration"], d["build_h"], bottom=bottom, color=pal[arm], alpha=0.45,
               edgecolor="white", linewidth=0.5, width=0.8)
        bottom += d["build_h"].values
        ax.bar(d["iteration"], d["train_h"], bottom=bottom, color=pal[arm], edgecolor="white",
               linewidth=0.5, width=0.8)
        ax.set_title(f"{arm}  (Σ {d['gpu_h'].sum():.1f} h)", fontsize=9)
        ax.set_xlabel("iteration", fontsize=9)
        ax.set_xticks(range(1, n_it + 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if arm in notes:
            x, yf, txt = notes[arm]
            ax.text(x, yf * ymax, txt, ha="center", va="center", fontsize=6.5, color="0.3")
        # Censoring is DERIVED: an arm whose last iteration falls short of the shared axis gets the
        # label centred over its own empty slots. Never a hard-coded arm or iteration (see
        # BREAKDOWN_NOTES) - that is exactly the assertion that went stale.
        last_it = int(d["iteration"].max()) if len(d) else 0
        if last_it < n_it:
            ax.text((last_it + n_it) / 2 + 0.5, 0.38 * ymax,
                    f"right-censored\n(stops at I{last_it})", ha="center", va="center",
                    fontsize=6.5, color="0.3")
    axes[0].set_ylabel("GPU-hours per iteration", fontsize=9)
    handles = [Patch(facecolor="white", edgecolor="0.4", hatch="////", label="generate"),
               Patch(facecolor="0.4", alpha=0.45, label="build (PTO only)"),
               Patch(facecolor="0.4", label="train")]
    axes[-1].legend(handles=handles, fontsize=7, loc="upper right", frameon=True, title="phase",
                    title_fontsize=7)
    fig.tight_layout()
    return fig


def cost_breakdown(frame: pd.DataFrame, **kw):
    """Where the GPU-hours go. Dispatches on the frame: an ``iteration_compute`` frame (has an
    ``iteration`` column) -> :func:`cost_breakdown_by_iteration` (the paper's per-iteration
    panels); a ``compute_summary`` frame -> :func:`cost_breakdown_by_arm` (stacked per-arm bars,
    the pre-2026-08-18 behaviour). ``**kw`` are forwarded."""
    if frame is None or frame.empty:
        return None
    if "iteration" in frame.columns:
        return cost_breakdown_by_iteration(frame, **kw)
    return cost_breakdown_by_arm(frame, **kw)


def _auto_xlim(x: np.ndarray, pad: float = 0.1):
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    r = max(hi - lo, 1.0)
    return (max(0.0, lo - pad * r), hi + pad * r)


def budget_sweep_grid(sweeps: Dict[tuple, pd.DataFrame], *,
                      methods: Sequence[str] = ("PTO", "GRPO"),
                      judges: Optional[Sequence[str]] = None,
                      select_metric: str = "Q1Q2", eval_metric: str = "Q1Q2",
                      alpha: float = 0.05, figsize=(7.0, 5.2),
                      xlims: Optional[Dict[str, tuple]] = None):
    """The look-ahead budget sweep, both graders x both methods (the paper's ``fig_budget_sweep``).

    ``sweeps`` is :func:`eda_analysis.compute.all_budget_sweeps` output
    (``{(contrast_tag, judge_label): frame}``); the ``<METHOD>_K`` contrasts are drawn. Rows =
    graders (``judges``, default: every judge present, in first-seen order), cols = ``methods``.
    x = the K=5 arm's cumulative GPU-h; y = paired ``eval_metric`` delta **K5 - K0** between the
    best-within-budget checkpoints (``mean_delta`` as tabled — the tracked-EDA sign; above zero =
    look-ahead ahead), bootstrap 95% CI bars; hollow markers = Holm p >= ``alpha``; labels
    ``I<K5>/I<K0>`` name the selected iterations (a repeated checkpoint pair is labelled once).
    ``xlims`` = ``{method: (lo, hi)}`` overrides the per-column auto range.
    """
    if not sweeps:
        return None
    tags = {m: f"{m}_K" for m in methods}
    if judges is None:
        judges = []
        for (tag, jl) in sweeps:
            if tag in tags.values() and jl not in judges:
                judges.append(jl)
    judges = list(judges)
    if not judges:
        return None
    fig, axes = plt.subplots(len(judges), len(methods), figsize=figsize, sharex="col", squeeze=False)
    xs_by_col: Dict[int, list] = {}
    for ci_, method in enumerate(methods):
        tag = tags[method]
        for ri, jl in enumerate(judges):
            ax = axes[ri, ci_]
            d = sweeps.get((tag, jl))
            if d is None or d.empty:
                ax.set_title(f"{method} — {jl} (no sweep)", fontsize=10)
                ax.axis("off")
                continue
            d = d[(d.select_metric == select_metric) & (d.eval_metric == eval_metric)]
            if d.empty:
                ax.axis("off")
                continue
            a, b = str(d["arm_a"].iloc[0]), str(d["arm_b"].iloc[0])
            col = arm_palette([a])[a]
            x = d["budget_gpu_h"].values; y = d["mean_delta"].values
            yerr = np.vstack([y - d["ci_lo"].values, d["ci_hi"].values - y])
            ax.axhline(0, color="0.35", lw=0.9, zorder=1)
            ax.errorbar(x, y, yerr=yerr, color=col, lw=1.7, capsize=2.5, elinewidth=1.0,
                        label=f"{a} vs {b}", zorder=3, ms=5.5, **_kstyle(a))
            ns = d["p_holm"].values >= alpha
            if ns.any():
                ax.plot(x[ns], y[ns], ls="none", marker=_kstyle(a)["marker"], ms=5.5, mfc="white",
                        mec=col, mew=1.3, zorder=4, label="not sig. (Holm)")
            prev = None
            for xi, yi, ia, ib in zip(x, y, d["best_iter_a"], d["best_iter_b"]):
                if (ia, ib) == prev:
                    continue
                prev = (ia, ib)
                ax.annotate(f"I{int(ia)}/I{int(ib)}", (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=6.5, color="0.25")
            ax.set_title(f"{method} — {jl}", fontsize=10)
            if ci_ == 0:
                ax.set_ylabel(f"{display_label(eval_metric)} Δ  (K=5 − K=0)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower right", frameon=True)
            xs_by_col.setdefault(ci_, []).extend(list(x))
    for ci_, method in enumerate(methods):
        lim = (xlims or {}).get(method)
        if lim is None and xs_by_col.get(ci_):
            lim = _auto_xlim(np.asarray(xs_by_col[ci_]))
        if lim is not None:
            for ax in axes[:, ci_]:
                ax.set_xlim(*lim)
    for ax in axes[-1, :]:
        ax.set_xlabel("cumulative GPU-hours (K=5 arm's budget)", fontsize=9)
    fig.suptitle(f"Look-ahead vs budget: paired {display_label(eval_metric)} delta between "
                 "best-within-budget checkpoints (labels I_K5/I_K0 = selected iterations)", fontsize=9)
    fig.tight_layout()
    return fig


def trajectory_by_compute(scores_by_judge: Dict[str, pd.DataFrame], comp: pd.DataFrame, *,
                          metric: str = "Q1Q2", arms: Optional[Sequence[str]] = None,
                          layout: str = "wide", figsize=None):
    """``metric`` mean ± SEM vs cumulative GPU-h, four arms, one panel per grader.

    ``scores_by_judge`` = ``{judge_label: scores_long}`` (primary first by convention); ``comp`` =
    :func:`~eda_analysis.compute.iteration_compute`. Each panel is
    :func:`~eda_analysis.compute.score_by_compute` drawn per arm (iteration 0 at 0 h; K=0
    solid/circle, K=5 dashed/square; last point labelled with its iteration — a censored arm simply ends first).

    ``layout="wide"`` = the paper's ``fig_trajectory`` (panels side by side, shared y);
    ``layout="col"`` = ``fig_trajectory_col`` (the SAME panels stacked, shared x and y, sized for
    a 3.4-in single column with fonts scaled for the narrow width). Same data, same style;
    the layout is the only difference, so a numbers ledger stays layout-agnostic.
    """
    from ..compute import score_by_compute
    if not scores_by_judge or comp is None or comp.empty:
        return None
    judges = list(scores_by_judge)
    col = layout == "col"
    if figsize is None:
        figsize = (3.4, 2.0 * len(judges)) if col else (3.5 * len(judges), 3.3)
    if col:
        fig, axes = plt.subplots(len(judges), 1, figsize=figsize, sharex=True, sharey=True, squeeze=False)
        axes = axes[:, 0]
    else:
        fig, axes = plt.subplots(1, len(judges), figsize=figsize, sharey=True, squeeze=False)
        axes = axes[0]
    ms, lw, cap, elw, fs_ann = (4, 1.4, 1.5, 0.8, 6.5) if col else (5, 1.7, 2, 0.9, 7)
    drew = False
    for ax, jl in zip(axes, judges):
        sbc = score_by_compute(scores_by_judge[jl], comp, metric=metric)
        if sbc is None or sbc.empty:
            ax.set_title(f"{display_label(metric)} vs compute — {jl} (no scores)", fontsize=9)
            continue
        order = list(arms) if arms is not None else \
            [a for a in ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5") if a in set(sbc.arm)] + \
            [a for a in sorted(sbc.arm.unique()) if a not in ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5")]
        pal = arm_palette(order)
        for arm in order:
            d = sbc[sbc.arm == arm].sort_values("iteration").dropna(subset=["cum_gpu_h", "mean"])
            if d.empty:
                continue
            ks = _kstyle(arm)
            ax.errorbar(d["cum_gpu_h"], d["mean"], yerr=d["sem"], color=pal[arm], ls=ks["ls"],
                        marker=ks["marker"], ms=ms, lw=lw, capsize=cap, elinewidth=elw, label=arm)
            last = d.iloc[-1]
            ax.annotate(f"I{int(last.iteration)}", (last.cum_gpu_h, last["mean"]),
                        textcoords="offset points",
                        xytext=((3, -3 if _k_of(arm) == 0 else 3) if col else (4, -3 if _k_of(arm) == 0 else 4)),
                        fontsize=fs_ann, color=pal[arm])
            drew = True
        ax.set_title(f"{display_label(metric)} vs compute — {jl}", fontsize=9 if col else 10)
        ax.tick_params(labelsize=7.5 if col else 8)
        ax.grid(True, alpha=0.3)
        if col:
            ax.set_ylabel(f"{display_label(metric)} (mean ± SEM, 96 personas)", fontsize=7.5)
        else:
            ax.set_xlabel("cumulative GPU-hours", fontsize=9)
    if not drew:
        plt.close(fig)
        return None
    if col:
        axes[-1].set_xlabel("cumulative GPU-hours", fontsize=8)
        axes[0].legend(fontsize=6.5, loc="lower right", frameon=True, handlelength=2.2)
    else:
        axes[0].set_ylabel(f"{display_label(metric)} (mean ± SEM, 96 personas)", fontsize=9)
        axes[0].legend(fontsize=7.5, loc="lower right", frameon=True)
    fig.tight_layout()
    return fig
