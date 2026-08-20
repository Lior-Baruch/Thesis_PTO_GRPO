"""instruments.py — figures for the held-out-instrument read of the K knob (RQ-i, behaviour family).

Companion of :mod:`eda_analysis.instruments`; promoted (2026-08-18) from the look-ahead paper's
generator ``papers/2026_lookahead_pto_grpo/analysis/held_out_instruments.py`` (figures
``held_out_instruments_fig_wai`` / ``held_out_instruments_fig_hetero`` — the frozen PNGs are the
visual fixture). Both figures put the two graders side by side (one panel column per grader,
never averaged) and read persona-paired contrasts with 95% percentile-bootstrap CIs.

- :func:`wai_fig` — WAI-SR subscale gain over own base (Task / Goal / Bond) at each arm's
  endpoint, PLUS the K=0 arm of a censored method at the matched iteration (GRPO_LA0 @ 5 next to
  GRPO_LA5 @ 5, drawn faded). K=5 arms are hatched; the arm palette carries the method.
- :func:`hetero_fig` — the K0−K5 contrast (``+ => K=0 higher``) within patient cooperation
  level, a 2×2 grid: rows = metrics (Q1Q2 then MICI, where LOWER is better so a positive bar is
  K=0 WORSE), columns = graders; PTO vs GRPO bars per stratum with a ``*`` above the CI whisker
  where the Holm-adjusted p (3 strata) < 0.05.

Contract as everywhere in ``plotting``: takes already-built tidy frames (:func:`wai_fig_data` /
:func:`hetero_kcontrast` output), never touches disk, returns a ``fig`` (the notebook owns
``save_fig``), returns ``None`` when the frame is empty.
"""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plotting_style import arm_palette

__all__ = ["K_STYLE", "wai_fig", "hetero_fig"]

# K=0 solid + circle (+ plain bars), K=5 dashed + square (+ hatched bars) — the same encoding as
# plotting/lookahead.py and plotting/compute.py, so the K contrast survives greyscale printing
# while the arm palette (PTO cool / GRPO warm) still carries the method.
from ._shared import K_STYLE_HATCHED as K_STYLE  # noqa: F401  (one definition; see _shared)
from ..constants import k_of as _k_of_canonical  # noqa: E402

_ARM_ORDER = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]


def _k_of(arm: str) -> int:
    """Re-export of :func:`eda_analysis.constants.k_of` (THE canonical arm parse)."""
    return _k_of_canonical(arm)


def _judges(df: pd.DataFrame) -> list:
    return list(dict.fromkeys(df["judge"]))


def _series_order(fig_data: pd.DataFrame) -> list:
    """(arm, iteration) pairs in display order: arm order, each arm's endpoint (max iteration)
    first, then any extra (matched) iteration."""
    arms = [a for a in _ARM_ORDER if a in set(fig_data["arm"])] + \
           sorted(a for a in set(fig_data["arm"]) if a not in _ARM_ORDER)
    out = []
    for arm in arms:
        its = sorted(set(int(i) for i in fig_data.loc[fig_data["arm"] == arm, "iteration"]), reverse=True)
        out += [(arm, it) for it in its]
    return out


def wai_fig(fig_data: pd.DataFrame, *, palette: Optional[Dict[str, str]] = None,
            subscales: Sequence[str] = ("Task", "Goal", "Bond"), figsize=(7.2, 3.1),
            title: Optional[str] = "WAI-SR subscale gain at the endpoint (persona-paired vs own base, 95% bootstrap CI)"):
    """WAI-SR subscale gain over own base by arm, one panel per grader.

    ``fig_data`` = :func:`eda_analysis.instruments.wai_fig_data` output (``judge, arm, iteration,
    subscale, gain, ci_lo, ci_hi, n``). Bars = mean persona-paired gain, whiskers = 95% bootstrap
    CI. Each arm's endpoint series is drawn at full alpha; a second (matched, non-endpoint)
    iteration of the same arm — GRPO_LA0 @ 5 today — is drawn faded (alpha 0.45) so it reads as
    context for the censored K=5 arm beside it. K=5 bars are hatched (:data:`K_STYLE`).
    """
    if fig_data is None or fig_data.empty:
        return None
    judges = _judges(fig_data)
    series = _series_order(fig_data)
    pal = palette or arm_palette([a for a, _ in series])
    end_iter = {arm: max(it for a, it in series if a == arm) for arm, _ in series}
    fig, axes = plt.subplots(1, len(judges), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(subscales)); w = 0.8 / max(len(series), 1)
    off0 = (len(series) - 1) / 2.0
    for ax, j in zip(axes, judges):
        d = fig_data[fig_data["judge"] == j]
        for si, (arm, it) in enumerate(series):
            t = d[(d["arm"] == arm) & (d["iteration"] == it)].set_index("subscale").reindex(list(subscales))
            vals = t["gain"].to_numpy(float)
            los = vals - t["ci_lo"].to_numpy(float); his = t["ci_hi"].to_numpy(float) - vals
            alpha = 1.0 if it == end_iter[arm] else 0.45
            ax.bar(x + (si - off0) * w, vals, w, color=pal.get(arm, "#777777"), alpha=alpha,
                   hatch=K_STYLE.get(_k_of(arm), K_STYLE[0])["hatch"], edgecolor="white", linewidth=0.6,
                   yerr=[np.nan_to_num(los), np.nan_to_num(his)],
                   error_kw=dict(elinewidth=1.0, capsize=2, ecolor="#333333"),
                   label=f"{arm} @ iter {it}")
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(list(subscales))
        ax.set_title(f"grader: {j}", fontsize=10)
        ax.set_xlabel("WAI-SR subscale")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.08)
    axes[0].set_ylabel("gain over own base\n(WAI-SR points, 1–5 scale)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, loc="lower center", ncol=min(len(series), 5), frameon=False,
               bbox_to_anchor=(0.5, -0.10), handlelength=1.6, columnspacing=1.0)
    if title:
        fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    return fig


_YLABEL = {"Q1Q2": "K0 − K5, Q1Q2 (1–5)",
           "MICI": "K0 − K5, MICI rate\n(per therapist turn; ↓ better)",
           "PCT": "K0 − K5, PCT change-talk\nproportion (↑ better)"}


def hetero_fig(hetero: pd.DataFrame, *, palette: Optional[Dict[str, str]] = None,
               metrics: Sequence[str] = ("Q1Q2", "MICI"), target: str = "matched_final",
               methods: Sequence[str] = ("PTO", "GRPO"),
               coop_order: Sequence[str] = ("Cooperative", "Warms up", "Resistant"),
               figsize=(7.2, 4.8),
               title: Optional[str] = "Look-ahead contrast by patient cooperation (persona-paired; + = K=0 higher)"):
    """K0−K5 delta by cooperation stratum: rows = ``metrics``, columns = graders; PTO vs GRPO bars.

    ``hetero`` = :func:`eda_analysis.instruments.hetero_kcontrast` output; only ``target`` rows and
    the named strata are drawn (the ``All`` reference row is not). Bar colour = the method's K=0
    arm colour (the bar is a within-method contrast, so the arm palette carries the method).
    Whiskers = 95% bootstrap CI; ``*`` above the whisker (or above 0) where ``p_holm`` < 0.05
    (Holm across the 3 strata). Iteration labels in the legend come from the frame's
    ``iter_K0``/``iter_K5``.
    """
    if hetero is None or hetero.empty:
        return None
    d = hetero[(hetero["target"] == target) & (hetero["cooperation"].isin(list(coop_order)))]
    if d.empty:
        return None
    judges = _judges(d)
    pal = palette or arm_palette([f"{m}_LA0" for m in methods])
    fig, axes = plt.subplots(len(metrics), len(judges), figsize=figsize, sharex=True, sharey="row",
                             squeeze=False)
    xs = np.arange(len(coop_order)); w = 0.68 / max(len(methods), 1)
    off0 = (len(methods) - 1) / 2.0
    for ci, j in enumerate(judges):
        for ri, metric in enumerate(metrics):
            ax = axes[ri, ci]
            for mi, method in enumerate(methods):
                t = d[(d["judge"] == j) & (d["method"] == method) & (d["metric"] == metric)]
                if t.empty:
                    continue
                t = t.set_index("cooperation").reindex(list(coop_order))
                vals = t["mean_delta"].to_numpy(float)
                yerr = [np.nan_to_num(vals - t["ci_lo"].to_numpy(float)),
                        np.nan_to_num(t["ci_hi"].to_numpy(float) - vals)]
                it0 = int(t["iter_K0"].dropna().iloc[0]); it5 = int(t["iter_K5"].dropna().iloc[0])
                ax.bar(xs + (mi - off0) * w, vals, w, color=pal.get(f"{method}_LA0", "#777777"),
                       edgecolor="white", yerr=yerr,
                       error_kw=dict(elinewidth=1.0, capsize=2, ecolor="#333333"),
                       label=f"{method} (iter {it0} vs {it5})")
                for xi, (_, r) in zip(xs + (mi - off0) * w, t.iterrows()):
                    if pd.notna(r["p_holm"]) and r["p_holm"] < 0.05:
                        ax.annotate("*", (xi, max(float(r["ci_hi"]), 0.0)), ha="center", va="bottom", fontsize=11)
            ax.axhline(0, color="#555555", lw=0.8)
            ax.set_xticks(xs); ax.set_xticklabels(list(coop_order))
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.18)
            if ri == 0:
                ax.set_title(f"grader: {j}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(_YLABEL.get(metric, f"K0 − K5, {metric}"))
            if ri == len(metrics) - 1:
                ax.set_xlabel(f"patient cooperation level ({_n_per_stratum(d)} personas each)")
    axes[0, 0].legend(fontsize=7.5, loc="upper left", title="* Holm p<0.05 (3 strata)", title_fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    return fig


def _n_per_stratum(d: pd.DataFrame) -> str:
    ns = sorted(set(int(n) for n in d["n"].dropna()))
    return str(ns[0]) if len(ns) == 1 else f"{ns[0]}–{ns[-1]}"
