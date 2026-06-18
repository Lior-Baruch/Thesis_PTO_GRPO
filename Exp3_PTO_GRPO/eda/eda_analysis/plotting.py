"""
plotting.py — the figure layer: shared style helpers + named recurring plot functions.

Merged from the two former plotting modules so there is ONE file to open when you tweak a figure:

- **helpers** (the former ``figures.py``) — a consistent publication style (:func:`set_style`), the
  colourblind arm palette (:func:`arm_palette`), left-to-right :func:`model_order`, clean labels, a
  shared-legend helper, a dotted base-line helper, and the :func:`grid` subplot scaffold.
- **named plots** (the former ``plots.py``) — the figures that recur across notebooks (outcomes,
  trajectories, faithfulness, behavior, training internals), defined ONCE and called from multiple
  notebooks. Genuinely one-off exploration still lives inline in the notebooks.

Contract for every named-plot function: takes an already-built tidy frame (never touches disk),
returns a matplotlib ``fig`` (no ``plt.show()`` / ``save_fig`` — the notebook owns those), reuses the
helpers above, and degrades gracefully on thin/absent arms (returns ``None`` or an empty panel).

Both ``eda_analysis.figures`` and ``eda_analysis.plots`` are aliased to THIS module in
``__init__.py``, so existing ``figures.set_style(...)`` / ``plots.overlay_trajectory(...)`` notebook
calls keep working unchanged.
"""

import re
import sys
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import QUESTIONNAIRE_ORDER

# Self-alias: the named-plot functions below call ``figures.grid(...)`` / ``figures.arm_palette(...)``
# etc.; after the merge those helpers live in THIS module, so point ``figures`` at ourselves.
# (Attribute access happens at call time, by which point every helper below is defined.)
figures = sys.modules[__name__]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS — style, palette, model order, grid scaffold                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Okabe-Ito colourblind-safe palette. Grouped by TEMPERATURE so the method reads at a glance
# (PTO = cool / blues, GRPO = warm / orange-red), while the two within-method look-ahead arms stay
# clearly distinct. Base = neutral grey.
_ARM_COLORS = {
    "PTO_LA0": "#0072B2",   # blue
    "PTO_LA5": "#56B4E9",   # sky blue
    "GRPO_LA0": "#D55E00",  # vermillion
    "GRPO_LA5": "#E69F00",  # orange
    "Base": "#555555",      # the pooled descriptive base (see data.collapse_base)
}


# Plot-scale defaults read by grid() / plot functions when their args are omitted. Updated by
# set_style(cfg) so an EdaConfig's panel/ncols/score_ylim/share_y propagate everywhere. Defaults
# match the pre-refactor behaviour (grid ncols=3, panel=(5.0,3.2), no y-limit).
_SCALE = {"panel": (5.0, 3.2), "ncols": 3, "score_ylim": None, "share_y": False,
          "palette_overrides": {}}


def set_style(cfg=None):
    """Consistent, publication-grade global style for every Exp3 figure.

    Whitegrid theme + tight, vector-friendly save defaults so `exports.save_fig` produces clean
    PDF (editable text via fonttype 42). When an ``EdaConfig`` is passed, its ``context``,
    ``font_scale``, ``dpi``, ``savefig_dpi`` are applied and its ``panel``/``ncols``/``score_ylim``/
    ``share_y``/``palette_overrides`` become the module-level defaults used by :func:`grid`,
    :func:`apply_score_axis`, and :func:`arm_palette`.
    """
    context = getattr(cfg, "context", "notebook") or "notebook"
    font_scale = getattr(cfg, "font_scale", 1.0) or 1.0
    dpi = getattr(cfg, "dpi", 110) or 110
    savefig_dpi = getattr(cfg, "savefig_dpi", 200) or 200
    sns.set_theme(style="whitegrid", context=context, font_scale=font_scale)
    plt.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": savefig_dpi,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable/embeddable text in vector output
        "figure.autolayout": False,
    })
    if cfg is not None:
        # Only override when the cfg sets a value (None = inherit the pre-refactor default).
        if getattr(cfg, "panel", None) is not None:
            _SCALE["panel"] = tuple(cfg.panel)
        if getattr(cfg, "ncols", None) is not None:
            _SCALE["ncols"] = int(cfg.ncols)
        _SCALE["score_ylim"] = getattr(cfg, "score_ylim", None)
        _SCALE["share_y"] = bool(getattr(cfg, "share_y", False))
        _SCALE["palette_overrides"] = dict(getattr(cfg, "palette_overrides", {}) or {})


def arm_palette(labels: Sequence[str]) -> dict:
    """Stable ``{arm_label: color}`` (cfg overrides > known Okabe-Ito > tab10 fallback)."""
    pal = {l: _ARM_COLORS.get(l) for l in labels}
    missing = [l for l in labels if pal[l] is None]
    for l, c in zip(missing, sns.color_palette("tab10", len(missing)).as_hex()):
        pal[l] = c
    pal.update({l: c for l, c in _SCALE["palette_overrides"].items() if l in pal})
    return pal


def apply_score_axis(ax, *, ylim=None, metric: str = ""):
    """Apply the configured score y-limits to ``ax`` (no-op if neither cfg nor arg sets them).

    ``ylim`` (arg) wins over the module default from ``set_style(cfg)``. Skipped for proportion /
    rate metrics whose natural range differs from the 1–5 rubric scale.
    """
    lim = ylim if ylim is not None else _SCALE.get("score_ylim")
    if lim is None:
        return
    if metric in {"PCT", "MICI", "R:Q", "%CR", "%MICO"}:   # different natural scale
        return
    ax.set_ylim(*lim)


def model_order(scores_long) -> List[str]:
    """Models left-to-right by (method, K, iteration) — for stable bar/x ordering.

    The pooled descriptive ``Base`` (``data.collapse_base``: method ``"Base"``, K ``-1``) always
    sorts first.
    """
    meta = (scores_long[["model", "method", "K", "iteration"]]
            .drop_duplicates().sort_values(["method", "K", "iteration"]))
    order = meta["model"].tolist()
    if "Base" in order:  # guarantee the pooled base leads regardless of sort keys
        order = ["Base"] + [m for m in order if m != "Base"]
    return order


_MODEL_RE = re.compile(r"^(PTO|GRPO)Exp3_LA(\d+)_(Base|I\d+)$")


def clean_label(model: str) -> str:
    """Tidy (full, no-abbreviation) axis label: ``PTOExp3_LA0_I3`` -> ``PTO_LA0_I3``.

    Only drops the redundant constant ``Exp3`` (every model is Exp3); keeps method, look-ahead K,
    and iteration spelled out. Pooled ``Base`` -> ``Base``. Unknown strings pass through unchanged.
    """
    if model == "Base":
        return "Base"
    m = _MODEL_RE.match(model)
    if not m:
        return model
    method, k, tail = m.groups()
    return f"{method}_LA{k}_{tail}"


def relabel_xticks(ax, *, rotation: int = 90, fontsize: int = 7):
    """Re-label a categorical x-axis with :func:`clean_label`, pinning ticks first.

    Pinning the existing tick positions before relabeling avoids matplotlib's FixedLocator
    warning (set_ticklabels without set_ticks) and any label/tick drift.
    """
    ticks = ax.get_xticks()
    labels = [clean_label(t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rotation, fontsize=fontsize)


def add_base_line(ax, base_value, *, annotate: bool = True):
    """Draw a dotted horizontal reference at the (pooled) base score on a bar/point panel.

    Lets the reader see at a glance which models sit above vs below base. No-op if
    ``base_value`` is None/NaN.
    """
    if base_value is None or (isinstance(base_value, float) and np.isnan(base_value)):
        return
    ax.axhline(base_value, ls=":", lw=1.1, color="#555555", zorder=0.5)
    if annotate:
        ax.text(0.995, base_value, " base", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color="#555555")


def figure_legend_from(ax, fig, *, title="arm", ncol: int = 4):
    """Lift ``ax``'s legend to a single figure-level legend ABOVE the grid (out of the data).

    Reads the handles/labels off ``ax``, removes every per-axis legend it can see, and draws one
    shared legend so multi-panel figures don't repeat a key inside a data area. No-op if ``ax``
    has nothing to key.
    """
    handles, labels = ax.get_legend_handles_labels()
    for a in fig.axes:
        if a.get_legend() is not None:
            a.legend_.remove()
    if handles:
        fig.legend(handles, labels, title=title, loc="upper center",
                   bbox_to_anchor=(0.5, 1.04), ncol=ncol, frameon=False, fontsize=8)


def grid(n: int, ncols: int = None, panel=None):
    """A ready (fig, axes_flat) grid sized for *n* panels; trailing axes hidden.

    ``ncols``/``panel`` default to the values set by ``set_style(cfg)`` (the EdaConfig scales),
    so a notebook that sets ``ncols=3, panel=(6,4)`` in cell 1 gets it everywhere.

    Usage in a notebook:
        fig, axes = figures.grid(len(METRICS))
        for ax, m in zip(axes, METRICS):
            sns.lineplot(..., ax=ax)
    """
    ncols = _SCALE["ncols"] if ncols is None else ncols
    panel = _SCALE["panel"] if panel is None else panel
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False)
    axes_flat = axes.flat
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NAMED PLOTS — the figures that recur across notebooks                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _metrics(frame_metrics, metrics: Optional[Sequence[str]]) -> list:
    present = set(frame_metrics)
    return [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in present]


# ── Outcomes ─────────────────────────────────────────────────────────────────
def outcomes_by_model(scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                      order: Optional[Sequence[str]] = None, ncols: int = 2):
    """Grouped outcome bars per rubric over every model (left-to-right by method/K/iter)."""
    metrics = _metrics(scores_long["questionnaire"].unique(), metrics)
    order = list(order) if order is not None else figures.model_order(scores_long)
    fig, axes = figures.grid(len(metrics), ncols=ncols, panel=(7.6, 3.4))
    for ax, m in zip(axes, metrics):
        dm = scores_long[scores_long.questionnaire == m]
        sns.barplot(dm, x="model", y="score", hue="arm",
                    order=order, palette=palette, errorbar=("ci", 95), dodge=False, ax=ax)
        ax.set_title(m); ax.set_xlabel("")
        figures.relabel_xticks(ax)
        figures.add_base_line(ax, float(dm[dm.is_base].score.mean()) if dm.is_base.any() else None)
    figures.figure_legend_from(axes[0], fig, title="arm")
    fig.suptitle("Outcome metrics by model — full-conversation eval (dotted line = base)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def outcomes_headline_by_arm(sel_scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                             ncols: int = 3):
    """Headline outcome bars: a pooled ``Base`` bar + one bar per arm's best iteration.

    Caller passes a best-per-arm frame run through :func:`data.collapse_base` (so the arm bases
    pool into a single ``Base`` column and each arm column is its peak iteration). A dotted base
    reference line is drawn per panel so above/below-base reads instantly.
    """
    metrics = _metrics(sel_scores_long["questionnaire"].unique(), metrics)
    arms = sorted(sel_scores_long.arm.unique())
    arm_order = (["Base"] if "Base" in arms else []) + [a for a in arms if a != "Base"]
    fig, axes = figures.grid(len(metrics), ncols=ncols, panel=(5.0, 3.0))
    for ax, m in zip(axes, metrics):
        dm = sel_scores_long[sel_scores_long.questionnaire == m]
        sns.barplot(dm, x="arm", y="score", hue="arm", order=arm_order, palette=palette,
                    errorbar=("ci", 95), ax=ax)
        ax.set_title(m); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30, labelsize=8)
        figures.add_base_line(ax, float(dm[dm.is_base].score.mean()) if dm.is_base.any() else None)
        if ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("Best-iteration outcomes by arm vs pooled Base — full-conversation eval",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Effect forest (replaces the wide main-results table inline) ───────────────
_EFFECT_COLOR = {"negligible": "#bdbdbd", "small": "#9ecae1", "medium": "#4292c6", "large": "#08519c"}


def effect_forest(main_results_df, *, arms: Optional[Sequence[str]] = None,
                  metric_order: Optional[Sequence[str]] = None):
    """Forest/dot plot of each arm×rubric improvement vs base — the readable stand-in for the
    28-row main-results table.

    Takes :func:`stats.main_results_table` output. One row per (arm, rubric): the mean Δ vs base
    (dot) with its 95% bootstrap CI (whisker), colored by effect-size label, dz annotated; dashed
    line at 0 = base (no change). Arms are blocked together, rubrics in canonical order.
    """
    if main_results_df is None or main_results_df.empty:
        return None
    df = main_results_df
    if arms is not None:
        df = df[df["arm"].isin(arms)]
    metric_order = [m for m in (metric_order or QUESTIONNAIRE_ORDER) if m in set(df["rubric"])]
    arm_order = sorted(df["arm"].unique())
    rows = [df[(df.arm == a) & (df.rubric == m)].iloc[0]
            for a in arm_order for m in metric_order
            if len(df[(df.arm == a) & (df.rubric == m)])]
    rows = rows[::-1]  # first arm/rubric at the top (matplotlib y grows upward)
    fig, ax = plt.subplots(figsize=(7.6, max(4.0, 0.34 * len(rows))))
    yticks, ylabels = [], []
    for i, r in enumerate(rows):
        color = _EFFECT_COLOR.get(r["effect"], "#999999")
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, lw=2.4, solid_capstyle="round", zorder=2)
        ax.scatter([r["delta"]], [i], color=color, s=42, zorder=3)
        ax.text(r["ci_high"], i, f"  dz={r['dz']:.2f}", va="center", fontsize=6.5, color="#333333")
        yticks.append(i); ylabels.append(f"{r['arm']} · {r['rubric']}")
    ax.axvline(0, color="#555555", lw=1.0, ls="--")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("improvement vs base (Δ mean score, 95% CI)")
    ax.set_title("Effect on full-conversation eval vs base (dashed = base; dz labelled)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in _EFFECT_COLOR.items()],
              title="effect size", fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def reliability_curve(agreement_df, *, palette=None):
    """Rank-agreement vs partial-conversation length — the Exp3 reward-reliability curve.

    Takes :func:`stats.rank_agreement_by_nturns` output (arm, n_turns, agreement, n_pairs). One
    line per arm; dashed line at 0.5 = chance. Comparing LA0 vs LA5 tests whether look-ahead makes
    the short training reward more faithful to the full-conversation eval.
    """
    if agreement_df is None or agreement_df.empty:
        return None
    pal = palette or figures.arm_palette(sorted(agreement_df.arm.unique()))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.lineplot(agreement_df, x="n_turns", y="agreement", hue="arm", marker="o", palette=pal, ax=ax)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.text(ax.get_xlim()[1], 0.5, " chance", ha="right", va="bottom", fontsize=7, color="grey")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Training-reward faithfulness: rank agreement vs conversation length")
    ax.set_xlabel("partial-conversation length when scored (n_turns)")
    ax.set_ylabel("sign-agreement with full-conv eval")
    sns.move_legend(ax, "lower right", title="arm", frameon=True)
    fig.tight_layout()
    return fig


def subscale_trajectory_grid(subscales_long, *, parents: Sequence[str] = ("WAI-SR", "MITI"),
                             min_iters: int = 3, arms: Optional[Sequence[str]] = None):
    """WAI-SR / MITI subscale *trajectories* — one panel per (parent, arm).

    Replaces the old 26-model × 3–4-subscale grouped-bar wall (unreadable; the subscales are
    near-equal within a model). Here each panel shows one colored line per subscale across
    ``iteration`` (±95% CI), so the reader can see how the components evolve and whether some
    (e.g. Bond / Empathy) climb faster than others (Goal / ChangeTalk). Each arm starts from its
    own iter-0 base. Arms with fewer than ``min_iters`` scored iterations are omitted (keeps a
    one-point arm like GRPO_LA5 out of the grid).
    """
    if subscales_long is None or subscales_long.empty:
        return None
    df = subscales_long
    iters_per_arm = df.groupby("arm")["iteration"].nunique()
    keep = [a for a in (arms if arms is not None else sorted(df.arm.unique()))
            if iters_per_arm.get(a, 0) >= min_iters]
    parents = [p for p in parents if p in set(df.parent)]
    if not keep or not parents:
        return None
    nrows, ncols = len(parents), len(keep)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows),
                             squeeze=False, sharex=True)
    for r, parent in enumerate(parents):
        for c, arm in enumerate(keep):
            ax = axes[r][c]
            d = df[(df.parent == parent) & (df.arm == arm)]
            sns.lineplot(d, x="iteration", y="score", hue="subscale", marker="o",
                         errorbar=("ci", 95), ax=ax)
            ax.set_title(f"{parent} — {arm}")
            ax.set_xlabel("iteration" if r == nrows - 1 else "")
            ax.set_ylabel("score" if c == 0 else "")
            # keep one legend per parent (its subscale set) on the left-most panel
            if c == 0 and ax.get_legend():
                ax.legend(fontsize=7, title="")
            elif ax.get_legend():
                ax.legend_.remove()
    fig.suptitle("WAI-SR / MITI subscale trajectories across iterations",
                 y=1.01, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Trajectories ─────────────────────────────────────────────────────────────
_SHARED_FACTOR_CAVEAT = ("The 6 rubrics load on ~one factor (PC1≈91%), so a uniform rise reflects "
                         "one warmth/satisfaction axis, not independent multi-skill gains.")


def trajectory_grid(scores_long, *, palette, metrics: Optional[Sequence[str]] = None,
                    arms: Optional[Sequence[str]] = None, iters: Optional[Sequence[int]] = None,
                    ncols: int = 3, caption: Optional[str] = _SHARED_FACTOR_CAVEAT):
    """Per-rubric mean ±95% CI across iterations, arms overlaid (one panel per rubric).

    ``arms``/``iters`` select which arms/iterations to show (None = all). A single shared arm
    legend sits above the grid; ``caption`` (default = the PC1≈91% shared-factor caveat) is printed
    under it so "all rubrics up" isn't read as multi-skill evidence. Pass ``caption=None`` to suppress.
    """
    if arms is not None:
        scores_long = scores_long[scores_long.arm.isin(list(arms))]
    if iters is not None:
        scores_long = scores_long[scores_long.iteration.isin(list(iters))]
    metrics = _metrics(scores_long["questionnaire"].unique(), metrics)
    fig, axes = figures.grid(len(metrics), ncols=ncols, panel=(5.2, 3.2))
    for ax, m in zip(axes, metrics):
        sns.lineplot(scores_long[scores_long.questionnaire == m], x="iteration", y="score",
                     hue="arm", palette=palette, marker="o", errorbar=("ci", 95), ax=ax)
        ax.set_title(m)
    figures.figure_legend_from(axes[0], fig, title="arm")
    fig.suptitle("Outcome trajectories across iterations — full-conversation eval",
                 y=1.12, fontweight="bold")
    if caption:
        fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=7.5,
                 style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def single_metric_trajectory(scores_long, metric: str = "Q1Q2", *, palette,
                             arms: Optional[Sequence[str]] = None,
                             iters: Optional[Sequence[int]] = None,
                             oracle_noise: float = 0.10, baseline_arm: Optional[str] = None):
    """One-metric learning curve (arms overlaid) with an oracle-noise band around base.

    ``arms``/``iters`` select which arms/iterations to overlay (None = all) — the lever for
    "show only PTO_LA0 vs GRPO_LA0" without a separate hardcoded figure. ``baseline_arm`` anchors
    the grey ±``oracle_noise`` band; if ``None`` the first arm with a base row is used.
    """
    d = scores_long[scores_long.questionnaire == metric]
    if arms is not None:
        d = d[d.arm.isin(list(arms))]
    if iters is not None:
        d = d[d.iteration.isin(list(iters))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(d, x="iteration", y="score", hue="arm", palette=palette, marker="o",
                 errorbar=("ci", 95), ax=ax)
    base = d[d.is_base]
    if baseline_arm is not None:
        base = base[base.arm == baseline_arm]
    if not base.empty:
        b0 = float(base.score.mean())
        ax.axhspan(b0 - oracle_noise, b0 + oracle_noise, color="grey", alpha=0.15)
        ax.text(0.02, b0 + oracle_noise, " ~oracle-noise band around base", fontsize=7,
                va="bottom", color="grey")
    ax.set_title(f"{metric} across iterations — full-conversation eval")
    ax.set_xlabel("training iteration (model state)"); ax.set_ylabel(f"{metric} (eval)")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), title="arm", frameon=False)
    fig.tight_layout()
    return fig


def overlay_trajectory(scores_long, metric: str = "Q1Q2", *, arms: Sequence[str], palette,
                       title: Optional[str] = None):
    """Overlay ANY chosen arms' trajectories of *metric* on one axis — the configurable contrast.

    One reusable figure for "PTO vs GRPO at K=0", "PTO K0 vs K5", or any arm set you pass — replaces
    the old per-K / per-method contrast loops. Degrades to whichever of ``arms`` are present;
    returns ``None`` if none are.
    """
    arms = list(arms)
    d = scores_long[(scores_long.arm.isin(arms)) & (scores_long.questionnaire == metric)]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(d, x="iteration", y="score", hue="arm", hue_order=[a for a in arms if a in set(d.arm)],
                 palette=palette, marker="o", errorbar=("ci", 95), ax=ax)
    ax.set_title(title or f"{metric}: {' vs '.join(a for a in arms if a in set(d.arm))}")
    ax.set_xlabel("iteration"); ax.set_ylabel(metric)
    fig.tight_layout()
    return fig


def method_contrast_overlay(scores_long, metric: str = "Q1Q2", *,
                            pair: Tuple[str, str] = ("PTO_LA0", "GRPO_LA0"), palette):
    """Back-compat thin wrapper around :func:`overlay_trajectory` for a two-arm ``pair``."""
    return overlay_trajectory(scores_long, metric, arms=list(pair), palette=palette,
                              title=f"{metric}: {pair[0]} vs {pair[1]} (matched)")


def heterogeneity_grid(scores_long, char: str, *, arms: Optional[Sequence[str]] = None,
                       metric: str = "Q1Q2", palette: str = "viridis", ncols: int = 2):
    """ONE figure: *metric* across iterations split by persona ``char``, a panel per selected arm.

    Replaces the old ``2 × N`` ``heterogeneity_{char}_{arm}`` PNG explosion — pick the trait + the
    arms and get a single small-multiples grid. Arms with <3 scored iters (or missing ``char``) are
    skipped. Returns ``None`` if nothing is plottable.
    """
    if char not in scores_long.columns:
        return None
    d = scores_long[scores_long.questionnaire == metric]
    arm_list = [a for a in (arms if arms is not None else sorted(d.arm.unique()))
                if d[(d.arm == a)].iteration.nunique() >= 3 and d[d.arm == a][char].notna().any()]
    if not arm_list:
        return None
    fig, axes = figures.grid(len(arm_list), ncols=ncols, panel=(6.4, 3.8))
    for ax, arm in zip(axes, arm_list):
        sns.lineplot(d[d.arm == arm], x="iteration", y="score", hue=char, marker="o",
                     palette=palette, ax=ax)
        ax.set_title(f"{arm}"); ax.set_ylabel(metric)
    fig.suptitle(f"{metric} by {char} (true persona) — per arm", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Mechanism: faithfulness + rubric structure ───────────────────────────────
def faithfulness_proxy_vs_eval(scores_long, generations, *, metric: str = "Q1Q2",
                               palette=None):
    """Training proxy reward vs full-conversation eval, per (arm, iteration); dashed y=x.

    Joins on ``eval_iter = train_iter - 1`` (the proxy iteration N branches the policy that
    produced the ``model_iter_{N-1}`` eval convs). Deduped from ``00`` + ``02``.
    """
    if generations.empty:
        return None
    proxy = (generations.groupby(["arm", "eval_iter"])["score"].mean()
             .rename("proxy").reset_index().rename(columns={"eval_iter": "iteration"}))
    evalq = (scores_long[scores_long.questionnaire == metric]
             .groupby(["arm", "iteration"])["score"].mean().rename("eval").reset_index())
    faith = proxy.merge(evalq, on=["arm", "iteration"])
    if faith.empty:
        return None
    pal = palette or figures.arm_palette(sorted(faith.arm.unique()))
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.scatterplot(faith, x="proxy", y="eval", hue="arm", s=90, palette=pal, ax=ax)
    for _, r in faith.iterrows():
        ax.annotate(int(r.iteration), (r.proxy, r.eval), fontsize=7)
    lo = float(faith[["proxy", "eval"]].min().min())
    hi = float(faith[["proxy", "eval"]].max().max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
    ax.set_title(f"Training proxy reward vs full-conv eval ({metric})")
    ax.set_xlabel("proxy (training, K=0 branch)"); ax.set_ylabel(f"eval {metric}")
    fig.tight_layout()
    return fig


def rubric_correlation_heatmap(scores_long, *, metrics: Optional[Sequence[str]] = None,
                               corr_method: str = "spearman"):
    """Inter-rubric correlation heatmap (pooled over conversations).

    Diverging ``[-1, 1]`` colormap so a genuinely ORTHOGONAL or anti-correlated axis (e.g. the
    lower-is-better ``MICI``) shows as white/blue instead of being clipped to 0 — the whole point
    of adding the new axes is to see them NOT block-correlate with the warmth rubrics. Labels are
    sign-flagged via :func:`display_label` (lower-is-better metrics get a trailing ↓).
    """
    from . import stats, display_label
    corr = stats.rubric_correlation(scores_long, metrics=metrics, method=corr_method)
    labels = [display_label(m) for m in corr.columns]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="vlag",
                square=True, xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": f"{corr_method.title()} ρ"}, ax=ax)
    ax.set_title(f"Inter-rubric correlation ({corr_method.title()}, pooled)")
    fig.tight_layout()
    return fig


def factor_loadings_bars(scores_long, *, metrics: Optional[Sequence[str]] = None,
                         components: Sequence[str] = ("PC1",)):
    """How much each metric belongs to the dominant 'warmth' factor — a readable loadings bar chart.

    Standardized PCA over the (expanded) metric set; plots each metric's **loading** (correlation
    with the factor) on PC1 (and PC2 if requested) as a horizontal bar, blue for the 5 warmth rubrics
    and orange for the orthogonal axes. Reads in one glance: the warmth rubrics all load high on PC1
    (≈0.44 each — one shared factor), while R:Q/%CR/%MICO/PCT/MICI load ≈0 on PC1 (they are NOT on
    the warmth factor). Replaces the hard-to-read PC1×PC2 biplot. ``None`` if PCA can't be fit.
    """
    from . import stats, display_label
    fs = stats.rubric_factor_space(scores_long, metrics=metrics)
    if fs is None:
        return None
    load, evr, mets = fs["loadings"], fs["explained"], fs["metrics"]
    comp_idx = {"PC1": 0, "PC2": 1}
    comps = [c for c in components if comp_idx.get(c, 99) < len(evr)]
    order = sorted(mets, key=lambda m: load[m][0])   # ascending PC1 loading
    colors = ["#0072B2" if m in {"Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI"} else "#D55E00"
              for m in order]
    fig, axes = plt.subplots(1, len(comps), figsize=(4.6 * len(comps), 0.42 * len(order) + 1.2),
                             squeeze=False)
    y = np.arange(len(order))
    for ax, comp in zip(axes.flat, comps):
        vals = [load[m][comp_idx[comp]] for m in order]
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y); ax.set_yticklabels([display_label(m) for m in order], fontsize=8.5)
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_xlabel(f"loading on {comp} ({evr[comp_idx[comp]]:.0%} of variance)")
        ax.set_title({"PC1": "Dominant 'warmth' factor", "PC2": "Second factor"}.get(comp, comp))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#0072B2", label="warmth rubrics (one factor)"),
                        Patch(color="#D55E00", label="orthogonal axes")],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def leaderboard_scorecard(scores_long, *, metrics: Optional[Sequence[str]] = None,
                          selection: str = "best"):
    """One-glance scorecard: each arm's headline score per metric, warmth beside orthogonal axes.

    ``selection="best"`` uses each arm's best iteration (by own oracle) + its base; ``"final"`` uses
    the last iteration. Returns a tidy DataFrame (arm × metric) with lower-is-better metrics flagged
    ``↓`` in the column name — drop straight into ``save_table``.
    """
    from . import best_per_experiment, all_models, display_label, ORTHOGONAL_METRICS
    order = [m for m in (list(QUESTIONNAIRE_ORDER) + list(ORTHOGONAL_METRICS)) if m not in ("Q1", "Q2")]
    present = [m for m in (metrics or order) if m in set(scores_long.questionnaire.unique())]
    # "best" already filters to base + the own-oracle peak iteration per arm; both selections then
    # take the highest remaining non-base iteration as the headline row.
    sel = best_per_experiment(scores_long)[0] if selection == "best" else all_models(scores_long)
    rows = []
    for arm, g in sel.groupby("arm", sort=False):
        nb = g[~g.is_base]
        if nb.empty:
            continue
        pick = int(nb.iteration.max())
        gg = nb[nb.iteration == pick]
        row = {"arm": arm, "iteration": pick}
        for m in present:
            v = gg[gg.questionnaire == m]["score"]
            row[display_label(m)] = round(float(v.mean()), 3) if len(v) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ── Behavior ─────────────────────────────────────────────────────────────────
_DEFAULT_BEHAVIOR_METRICS = ["B3_Q", "B4_SR", "B5_CR", "B6_AF", "B2_Persuade",
                             "RtoQ", "Empathy", "loop", "q_per_turn"]


def behavior_trajectory_grid(behavior_by_iter, *, palette=None,
                             metrics: Optional[Sequence[str]] = None, ncols: int = 3):
    """Behavior metric trajectories (MITI counts + text metrics) across iterations, all arms."""
    bm = [m for m in (metrics or _DEFAULT_BEHAVIOR_METRICS) if m in behavior_by_iter.columns]
    if not bm:
        return None
    pal = palette or figures.arm_palette(sorted(behavior_by_iter.arm.unique()))
    fig, axes = figures.grid(len(bm), ncols=ncols)
    for ax, m in zip(axes, bm):
        sns.lineplot(behavior_by_iter, x="iteration", y=m, hue="arm", palette=pal, marker="o", ax=ax)
        ax.set_title(m)
        if ax is not axes[0] and ax.get_legend():
            ax.legend_.remove()
    fig.suptitle("Behavior trajectories (MITI counts + text metrics)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Training internals (both methods, side-by-side) ──────────────────────────
def reward_distribution(reward_frame, *, ncols: int = 2):
    """Per-candidate training-reward distribution per iteration — one panel per arm.

    Takes the tidy ``(arm, method, train_iter, score)`` frame
    (:func:`training.reward_distribution_frame`) so PTO and GRPO sit side-by-side under matched
    axes — the symmetric replacement for the old per-arm DeepDive boxplot.
    """
    if reward_frame.empty:
        return None
    arms = sorted(reward_frame.arm.unique())
    fig, axes = figures.grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
    for ax, arm in zip(axes, arms):
        g = reward_frame[reward_frame.arm == arm]
        method = g.method.iloc[0] if "method" in g and len(g) else ""
        sns.boxplot(g, x="train_iter", y="score", color="#c5b0d5", ax=ax)
        ax.set_title(f"{arm} ({method})"); ax.set_xlabel("training iteration"); ax.set_ylabel("candidate reward")
    fig.suptitle("TRAINING signal — candidate reward distribution per iteration "
                 "(oracle on partial-conv branches, NOT the full-conv eval)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def advantage_signal_sidebyside(advantage_df, *, ncols: int = 2):
    """The training-advantage signal for BOTH methods in one figure (never gated by method).

    Takes :func:`training.advantage_signal_by_iter`. One panel per arm: GRPO panels plot
    ``group_std`` (within-group reward spread), PTO panels plot the chosen−rejected ``margin``.
    Arms with no on-disk training capture (e.g. GRPO_LA5) simply don't appear.
    """
    if advantage_df.empty:
        return None
    arms = sorted(advantage_df.arm.unique())
    fig, axes = figures.grid(len(arms), ncols=ncols, panel=(6.0, 3.4))
    for ax, arm in zip(axes, arms):
        g = advantage_df[advantage_df.arm == arm].sort_values("train_iter")
        method = g.method.iloc[0]
        if method == "GRPO":
            ax.plot(g.train_iter, g.group_std, marker="o", color="#1f77b4")
            ax.set_ylabel("mean group_std")
            ax.set_title(f"{arm} (GRPO advantage spread)")
        else:
            ax.plot(g.train_iter, g.margin, marker="o", color="#7b4fb0")
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_ylabel("chosen − rejected margin")
            ax.set_title(f"{arm} (PTO preference margin)")
        ax.set_xlabel("training iteration")
    fig.suptitle("Training advantage signal (method-native, side-by-side)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig
