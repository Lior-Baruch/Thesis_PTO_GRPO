"""reliability.py — MEASUREMENT-VALIDITY figures: oracle repeatability (ICC) and second-judge
agreement / contrast preservation. (Data-side counterpart: :mod:`eda_analysis.reliability`.)

These read the ``data/eval_scores_by_judge/`` re-scoring tree via that module — no API calls, so they render
inside ``render_views.py`` while the paid re-scoring stays behind ``Judge_Reliability.ipynb``.
"""

import colorsys
import re

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from ..plotting_style import grid, arm_palette, clean_label

_MODEL_RE = re.compile(r"^(PTO|GRPO)Exp3_LA(\d+)_(Base|I\d+)$")

# Koo & Li (2016) test-retest reliability guideline cuts.
_ICC_CUTS = ((0.90, "excellent"), (0.75, "good"))


def _arm_of(model: str) -> str:
    """``PTOExp3_LA0_I10`` -> ``PTO_LA0`` so the shared arm palette applies (unknown -> ``Base``)."""
    m = _MODEL_RE.match(str(model))
    return f"{m.group(1)}_LA{m.group(2)}" if m else "Base"


def _iter_of(model: str) -> int:
    """Iteration index for within-arm ordering (``Base`` sorts first)."""
    m = _MODEL_RE.match(str(model))
    if not m:
        return -1
    tail = m.group(3)
    return -1 if tail == "Base" else int(tail[1:])


def _shade(color, f: float):
    """``color`` lightened toward white as ``f`` -> 0 (``f=1`` returns the colour unchanged)."""
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return colorsys.hls_to_rgb(h, l + (1 - f) * (0.86 - l), s)


def _model_palette(models):
    """``{model: colour}`` — arm HUE (matching every other EDA figure) with within-arm LIGHTNESS
    by iteration, so two models of the same arm (e.g. GRPO I8 vs I10) stay distinguishable.
    The latest iteration keeps the canonical arm colour; earlier states are lighter.
    """
    models = list(dict.fromkeys(models))
    pal = arm_palette(sorted({_arm_of(m) for m in models}))
    out = {}
    for arm in {_arm_of(m) for m in models}:
        members = sorted([m for m in models if _arm_of(m) == arm], key=_iter_of)
        fracs = np.linspace(0.45, 1.0, len(members)) if len(members) > 1 else [1.0]
        for m, f in zip(members, fracs):
            out[m] = _shade(pal[arm], float(f))
    return out


def _ordered(values, order=None):
    """Unique ``values`` in ``order`` when given (extras appended), else first-seen order."""
    seen = list(dict.fromkeys(values))
    if not order:
        return seen
    return [v for v in order if v in seen] + [v for v in seen if v not in set(order)]


def oracle_repeatability_bars(rep_tab, *, metrics=None, ncols: int = 3):
    """ICC(2,1) per model, one panel per metric, with the good/excellent guideline cuts.

    The instrument-noise figure: how much of the per-conversation score variance is real signal
    rather than re-scoring noise. Takes :func:`reliability.repeatability`. ``metrics`` fixes the
    panel order (default: whatever order the frame is in).
    """
    if rep_tab is None or rep_tab.empty:
        return None
    metrics = _ordered(rep_tab.metric, metrics)
    pal = _model_palette(rep_tab.model.unique())
    fig, axes = grid(len(metrics), ncols=ncols, panel=(4.6, 3.2))
    for ax, metric in zip(axes, metrics):
        g = rep_tab[rep_tab.metric == metric].sort_values("model")
        ax.bar(range(len(g)), g.icc_2_1, color=[pal[m] for m in g.model], width=0.62)
        for cut, _name in _ICC_CUTS:   # guideline cuts are named in the suptitle, not in-axes
            ax.axhline(cut, ls=":", lw=1.0, color="#555555", zorder=0.5)
        for i, (v, d) in enumerate(zip(g.icc_2_1, g.mean_abs_diff)):
            ax.text(i, v + 0.008, f"{v:.3f}\n|Δ|{d:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([clean_label(m) for m in g.model], rotation=90, fontsize=7)
        ax.set_ylim(0.5, 1.06)
        ax.set_title(metric)
        ax.set_ylabel("ICC(2,1) across reps")
    fig.suptitle("[EVAL] Oracle repeatability — same conversations re-scored "
                 f"{int(rep_tab.n_reps.max())}× (seeds differ, nothing else); "
                 "dotted = Koo & Li good (0.75) / excellent (0.90)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def judge_agreement_scatter(judge_long, primary_long, *, agr_tab=None, metrics=None,
                            judge_name: str = "second judge", ncols: int = 3,
                            jitter: float = 0.045, seed: int = 0):
    """Per-conversation second judge (y) vs primary oracle (x), one panel per metric.

    The dashed identity line makes the LEVEL bias visible (a harsher judge sits below it); the
    thesis claims live in the *spread around* a trend, not the offset. Panel annotation reports
    the across-model r range and the attenuation ceiling — agreement can never reach 1.0 because
    both raters are noisy (see :mod:`eda_analysis.reliability`).

    Rubric means land on a coarse grid (a 5-item mean moves in 0.2 steps), so points are jittered
    by ``±jitter`` to show density; pass ``jitter=0`` for exact positions. The correlations
    themselves are computed on UN-jittered values in :func:`reliability.agreement`.
    """
    if judge_long is None or judge_long.empty or primary_long is None or primary_long.empty:
        return None
    j = judge_long.groupby(["metric", "model", "file_index"])["value"].mean().rename("judge")
    p = primary_long.groupby(["metric", "model", "file_index"])["value"].mean().rename("primary")
    merged = pd.concat([j, p], axis=1).dropna().reset_index()
    if merged.empty:
        return None
    metrics = _ordered(merged.metric, metrics)
    pal = _model_palette(merged.model.unique())
    if jitter:
        rng = np.random.default_rng(seed)
        for col in ("primary", "judge"):
            merged[col] = merged[col] + rng.uniform(-jitter, jitter, len(merged))
    fig, axes = grid(len(metrics), ncols=ncols, panel=(4.6, 4.0))
    for ax, metric in zip(axes, metrics):
        g = merged[merged.metric == metric]
        sns.scatterplot(g, x="primary", y="judge", hue="model", palette=pal,
                        s=20, alpha=0.65, edgecolor="none", ax=ax, legend=(ax is axes[0]))
        lo = float(min(g.primary.min(), g.judge.min()))
        hi = float(max(g.primary.max(), g.judge.max()))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#555555", zorder=0.5)
        ax.text(0.02, 0.98, "dashed = identity", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5, color="#555555")
        note = ""
        if agr_tab is not None and not agr_tab.empty:
            a = agr_tab[agr_tab.metric == metric]
            if not a.empty:
                note = f"r {a.pearson_r.min():.2f}–{a.pearson_r.max():.2f}"
                if "ceiling" in a and a.ceiling.notna().any():
                    note += f"  (ceiling ≈ {a.ceiling.mean():.2f})"
        ax.set_title(f"{metric}   {note}".strip())
        ax.set_xlabel("primary oracle (gpt-4o-mini)")
        ax.set_ylabel(judge_name)
    if axes[0].get_legend() is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend_.remove()
        fig.legend(handles, [clean_label(l) for l in labels], title="model", loc="upper center",
                   bbox_to_anchor=(0.5, 1.05), ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(f"[EVAL] Per-conversation agreement — {judge_name} vs the primary oracle "
                 "(n=96 per model; offset from identity = level bias, not rank disagreement"
                 + (f"; points jittered ±{jitter:g})" if jitter else ")"),
                 y=1.13, fontweight="bold")
    fig.tight_layout()
    return fig


def judge_contrast_bars(contrast_df, *, metrics=None, judge_name: str = "second judge",
                        ncols: int = 2):
    """THE defense figure: each endpoint contrast under both judges, one panel per model pair.

    Same-sign bars = the claim is not an artifact of the patient/oracle sharing a model. Takes
    :func:`reliability.contrasts`. Bars are paired deltas (a − b) over the 96 matched personas;
    for MICI lower is better, so a negative bar on the PTO−GRPO pair favours PTO.
    """
    if contrast_df is None or contrast_df.empty:
        return None
    df = contrast_df.copy()
    if "contrast" not in df:
        df["contrast"] = df.model_a + " − " + df.model_b
    long = df.melt(id_vars=["contrast", "metric", "same_sign"],
                   value_vars=["primary_delta", "judge_delta"],
                   var_name="judge", value_name="delta")
    long["judge"] = long.judge.map({"primary_delta": "primary oracle (gpt-4o-mini)",
                                    "judge_delta": judge_name})
    pairs = list(dict.fromkeys(df.contrast))
    order = _ordered(long.metric, metrics)
    fig, axes = grid(len(pairs), ncols=ncols, panel=(5.4, 3.6))
    for ax, pair in zip(axes, pairs):
        g = long[long.contrast == pair]
        sns.barplot(g, x="metric", y="delta", hue="judge", order=order,
                    palette=["#0072B2", "#D55E00"], ax=ax, legend=(ax is axes[0]))
        ax.axhline(0, lw=1.0, color="#333333", zorder=2)
        lo, hi = ax.get_ylim()                      # headroom so the verdict never sits on a bar
        pad = 0.20 * (hi - lo)
        ax.set_ylim(lo, hi + pad)
        for i, metric in enumerate(order):
            row = df[(df.contrast == pair) & (df.metric == metric)]
            if row.empty:
                continue
            ok = bool(row.same_sign.iloc[0])
            # ASCII only — the check-mark glyph is missing from the default Arial face.
            ax.text(i, hi + pad * 0.45, "same sign" if ok else "SIGN FLIP",
                    ha="center", va="center", fontsize=7.5,
                    color="#2c7a2c" if ok else "#b22222", fontweight="bold")
        ax.set_title(pair)
        ax.set_ylabel("paired Δ (a − b)")
        ax.set_xlabel("")
    if axes[0].get_legend() is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend_.remove()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
                   ncol=2, frameon=False, fontsize=8)
    fig.suptitle("[EVAL] Contrast preservation — does the result survive a judge that never "
                 "played the patient?  (MICI: lower is better)", y=1.13, fontweight="bold")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-JUDGE VIEWS — level vs ordering, variance sources, transfer, resolution
# ══════════════════════════════════════════════════════════════════════════════
#
# Data side: :mod:`eda_analysis.reliability` (``variance_components_arm``, ``gain_retention``,
# ``concordance_by_effect_size``). Shared two-judge colours below; ``judge_agreement_scatter``
# above uses the same pair.

_PRIMARY_C, _JUDGE_C = "#0072B2", "#D55E00"


def judge_dumbbell(judge_long, primary_long, *, metrics=None, judge_name: str = "second judge",
                   ncols: int = 3):
    """Arm means under BOTH judges, one dumbbell per model — the honest multi-judge summary.

    Deliberately does NOT average the two judges. Level offset is large (~1.2-1.7 points on Q1/Q2)
    and MODEL-DEPENDENT, so a mean of the two sits on neither judge's rubric anchors and silently
    applies a model-dependent shrinkage to every effect. Showing both endpoints keeps the level
    difference visible (dumbbell LENGTH) while letting the reader check the thing that actually
    carries the thesis: whether both judges order the arms the same way (dumbbell ORDER).
    """
    if judge_long is None or judge_long.empty or primary_long is None or primary_long.empty:
        return None
    j = judge_long.groupby(["metric", "model"])["value"].mean().rename("judge")
    p = primary_long.groupby(["metric", "model"])["value"].mean().rename("primary")
    m = pd.concat([p, j], axis=1).dropna().reset_index()
    if m.empty:
        return None
    metrics = _ordered(m.metric, metrics)
    fig, axes = grid(len(metrics), ncols=ncols, panel=(4.6, 3.4))
    for ax, metric in zip(axes, metrics):
        g = m[m.metric == metric].copy()
        g = g.sort_values("primary").reset_index(drop=True)   # rank by the reported (primary) judge
        y = np.arange(len(g))
        ax.hlines(y, g.primary, g.judge, color="#999999", lw=1.6, zorder=1)
        ax.scatter(g.primary, y, s=42, color=_PRIMARY_C, zorder=3,
                   label="primary oracle (gpt-4o-mini)")
        ax.scatter(g.judge, y, s=42, color=_JUDGE_C, zorder=3, label=judge_name)
        ax.set_yticks(y)
        ax.set_yticklabels([clean_label(v) for v in g.model], fontsize=7)
        ax.set_title(f"{metric}   (mean |offset| {np.abs(g.judge - g.primary).mean():.2f})")
        ax.set_xlabel("arm mean score")
        ax.grid(axis="x", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2,
               frameon=False, fontsize=8)
    fig.suptitle("[EVAL] Arm means under both judges — bar LENGTH is level bias (cancels in every "
                 "contrast); bar ORDER is what the thesis claims. Never averaged across judges.",
                 y=1.13, fontweight="bold")
    fig.tight_layout()
    return fig


def variance_decomposition_bars(var_arm, *, metrics=None):
    """Where arm-mean variance comes from: real arm differences vs judge level vs arm x judge.

    The one figure that answers "how much of what I measured is real?". Stacked to 100% per metric:

    - **arm** — genuine between-policy differences. The signal.
    - **judge level** — the two judges' overall offset. Large and harmless: it cancels in contrasts.
    - **arm x judge** — arm ordering that depends on who is grading. The only component that
      threatens a claim, and the one a reward-hacked policy inflates.

    Takes :func:`reliability.variance_components_arm`. The annotation reports the dependability of
    an arm mean measured with one judge — the number to quote when asked how much a single-judge
    ranking can be trusted.
    """
    if var_arm is None or var_arm.empty:
        return None
    g = var_arm.copy()
    if metrics:
        g = g[g.metric.isin(metrics)]
        g["metric"] = pd.Categorical(g.metric, categories=[m for m in metrics if m in set(g.metric)],
                                     ordered=True)
        g = g.sort_values("metric")
    fig, axes = grid(1, ncols=1, panel=(max(5.2, 1.5 * len(g)), 3.8))
    ax = axes[0]
    x = np.arange(len(g))
    parts = [("pct_arm", "arm (signal)", "#2c7a2c"),
             ("pct_judge", "judge level (cancels in contrasts)", "#9ecae1"),
             ("pct_arm_x_judge", "arm x judge (the risk)", "#D55E00")]
    bottom = np.zeros(len(g))
    for col, label, colour in parts:
        vals = g[col].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=colour, label=label, width=0.62)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 4:                      # only label slices wide enough to read
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if colour != "#9ecae1" else "#222222")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(g.metric, fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("share of arm-mean variance (%)")
    for xi, row in enumerate(g.itertuples()):
        ax.text(xi, 102, f"G(1 judge)={row.dependability_k1:.2f}", ha="center", va="bottom",
                fontsize=6.8, color="#333333")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle("[EVAL] Sources of variance in the numbers the thesis reports "
                 "(two-way random effects over arms x judges)", y=1.14, fontweight="bold")
    fig.tight_layout()
    return fig


def gain_retention_bars(ret, *, metrics=None, ncols: int = 3, judge_name: str = "second judge"):
    """Fraction of each arm's gain over the reference that survives the judge swap, with CI.

    The reward-hacking test, read as a picture. Because the primary judge WAS the training reward
    and the second judge is held out, this is a train/test generalization ratio:

    - bars near **1.0** across all arms — the metric just compresses; nothing to see.
    - one arm's bar **collapsing toward 0** while the others hold — that arm's gain lived in the
      grader it optimized, not in behaviour a fresh judge can see.

    Takes :func:`reliability.gain_retention`. Whiskers are the persona bootstrap; compare arms by
    interval overlap, not by bar height.
    """
    if ret is None or ret.empty:
        return None
    g = ret.dropna(subset=["retention"]).copy()
    if g.empty:
        return None
    metrics = _ordered(g.metric, metrics)
    pal = _model_palette(g.model.unique())
    fig, axes = grid(len(metrics), ncols=ncols, panel=(4.6, 3.4))
    for ax, metric in zip(axes, metrics):
        s = g[g.metric == metric].sort_values("retention", ascending=False).reset_index(drop=True)
        x = np.arange(len(s))
        ax.bar(x, s.retention, color=[pal[m] for m in s.model], width=0.6)
        has_ci = s.retention_ci_lo.notna() & s.retention_ci_hi.notna()
        if has_ci.any():
            ax.errorbar(x[has_ci.to_numpy()], s.retention[has_ci],
                        yerr=[(s.retention - s.retention_ci_lo)[has_ci],
                              (s.retention_ci_hi - s.retention)[has_ci]],
                        fmt="none", ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)
        ax.axhline(1.0, ls="--", lw=1.0, color="#2c7a2c", zorder=0.5)
        ax.axhline(0.0, lw=1.0, color="#333333", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([clean_label(m) for m in s.model], rotation=90, fontsize=7)
        ax.set_ylabel("Δ(second judge) / Δ(primary)")
        ax.set_title(metric)
    ref = str(g.reference.iloc[0])
    fig.suptitle(f"[EVAL] Does the improvement transfer? Gain over {clean_label(ref)} as seen by "
                 f"{judge_name}, relative to the primary oracle it was trained against "
                 "(dashed = full transfer)", y=1.05, fontweight="bold")
    fig.tight_layout()
    return fig


def retention_trajectory(ret, *, metrics=None, ncols: int = 3, judge_name: str = "second judge",
                         palette=None):
    """Gain retention plotted ACROSS ITERATIONS, one line per arm — the full-grid payoff figure.

    :func:`gain_retention_bars` shows retention at a handful of model states. Once every iteration
    of every arm has been scored by both judges, the same quantity becomes a *trajectory*, and that
    is a sharper instrument: reward hacking is a process, so the question is not only "did this
    endpoint transfer?" but **"at which iteration did the gains stop transferring?"**

    Read a line that stays near 1.0 as gains a held-out judge also sees. A line that *declines with
    training* is the signature of a policy progressively fitting its grader — and the iteration
    where it turns is an estimate of when hacking set in, which no single-endpoint comparison can
    give you. Takes :func:`reliability.gain_retention` over many model states.
    """
    if ret is None or ret.empty:
        return None
    g = ret.dropna(subset=["retention"]).copy()
    if g.empty:
        return None
    g["arm"] = [_arm_of(m) for m in g.model]
    g["iteration"] = [_iter_of(m) for m in g.model]
    g = g[g.iteration >= 0]
    if g.empty:
        return None
    metrics = _ordered(g.metric, metrics)
    pal = palette or arm_palette(sorted(g.arm.unique()))
    fig, axes = grid(len(metrics), ncols=ncols, panel=(4.8, 3.4))
    for ax, metric in zip(axes, metrics):
        s = g[g.metric == metric]
        for arm, a in s.groupby("arm"):
            a = a.sort_values("iteration")
            colour = pal.get(arm, "#666666")
            ax.plot(a.iteration, a.retention, marker="o", ms=4, lw=1.7, color=colour, label=arm)
            if {"retention_ci_lo", "retention_ci_hi"} <= set(a.columns) and a.retention_ci_lo.notna().any():
                ax.fill_between(a.iteration, a.retention_ci_lo, a.retention_ci_hi,
                                color=colour, alpha=0.16, linewidth=0)
        ax.axhline(1.0, ls="--", lw=1.0, color="#2c7a2c", zorder=0.5)
        ax.axhline(0.0, lw=1.0, color="#333333", zorder=2)
        ax.set_xlabel("iteration")
        ax.set_ylabel("Δ(second judge) / Δ(primary)")
        ax.set_title(metric)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"[EVAL] When do the gains stop transferring? Retention vs iteration under "
                 f"{judge_name} (dashed = full transfer; a declining line = progressively fitting "
                 "the trained-against grader)", y=1.05, fontweight="bold")
    fig.tight_layout()
    return fig


def concordance_curve(conc, *, ncols: int = 3, judge_name: str = "second judge"):
    """Cross-judge ordering agreement as a function of the gap the primary judge reports.

    Answers "is a gap of THIS size trustworthy?" — which a single r or rho cannot, because level
    bias dominates Pearson and rank statistics discard the magnitude that decides whether a gap
    matters. Takes one or more :func:`reliability.concordance_by_effect_size` frames (concatenated;
    ``scope`` becomes the line style).

    ⚠ Each point is a pair of SINGLE CONVERSATIONS. The thesis compares 96-conversation means,
    which resolve ~10x better — do not read a bin height as confidence in an arm-level claim. The
    curve's job is to show how per-conversation resolution grows with effect size, i.e. why 96
    conversations per arm are needed.
    """
    if conc is None or (hasattr(conc, "empty") and conc.empty):
        return None
    df = pd.concat(conc, ignore_index=True) if isinstance(conc, (list, tuple)) else conc.copy()
    if df.empty:
        return None
    metrics = _ordered(df.metric)
    styles = {"cross_model": ("-", "o"), "within_model": ("--", "s"), "all": (":", "^")}
    fig, axes = grid(len(metrics), ncols=ncols, panel=(4.6, 3.4))
    for ax, metric in zip(axes, metrics):
        g = df[df.metric == metric]
        for scope, s in g.groupby("scope"):
            ls, mk = styles.get(str(scope), ("-", "o"))
            s = s.sort_values("mean_abs_delta_primary")
            ax.plot(s.mean_abs_delta_primary, s.concordance, ls=ls, marker=mk, ms=4.5,
                    lw=1.6, color=_JUDGE_C if scope == "cross_model" else "#888888",
                    label=str(scope).replace("_", " "))
        ax.axhline(0.5, ls=":", lw=1.0, color="#555555", zorder=0.5)
        ax.text(0.02, 0.52, "chance", transform=ax.get_yaxis_transform(), fontsize=6.5,
                color="#555555", va="bottom")
        ax.set_ylim(0.35, 1.03)
        ax.set_xlabel("|Δ| reported by the primary oracle")
        ax.set_ylabel(f"P({judge_name} agrees on order)")
        ax.set_title(metric)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("[EVAL] Ordering agreement vs effect size — per CONVERSATION PAIR, not per arm "
                 "(exact primary-judge ties excluded: they state no order to reproduce)",
                 y=1.05, fontweight="bold")
    fig.tight_layout()
    return fig
