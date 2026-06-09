"""
analysis.py — Statistics and plotting for the EDA notebook.

Split into two clearly-labelled sections:
- **Stats** — descriptive stats, effect sizes, FDR, omnibus and pairwise tests
  on ``test_cases`` (see :mod:`data`).
- **Plots** — barplots, subscale grids, correlation heatmaps, and session-end
  countplots, all stylable via a :class:`PlotContext`.
"""

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import f_oneway, kruskal, levene, mannwhitneyu, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from .config import ALPHA, EXPERIMENT_PALETTE, FIG_WIDE
from .data import parse_model_metadata


# ═══════════════════════════════════════════════════════════════════════════════
# ║                                  STATS                                     ║
# ═══════════════════════════════════════════════════════════════════════════════


def format_pvalue(p: float) -> str:
    """Pretty-print a p-value with ``*`` / ``**`` / ``***`` significance stars."""
    if p < 0.001: return f"{p:.2e} ***"
    if p < 0.01:  return f"{p:.4f} **"
    if p < 0.05:  return f"{p:.4f} *"
    return f"{p:.4f}"


def cohens_d(g1: Sequence[float], g2: Sequence[float]) -> float:
    """Pooled-SD Cohen's d. Positive means ``g1 > g2``. ``0.0`` if SD is zero."""
    g1 = np.asarray(g1, float)
    g2 = np.asarray(g2, float)
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
    return float((g1.mean() - g2.mean()) / pooled) if pooled > 0 else 0.0


def interpret_effect_size(d: float) -> str:
    """Map absolute Cohen's d to negligible / small / medium / large."""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def fdr_correct(p_values):
    """Benjamini-Hochberg FDR correction. Returns ``(reject: bool[], q_values: float[])``."""
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    reject, q, _, _ = multipletests(p_values, method="fdr_bh")
    return reject, q


def iter_metric_cases(cases: Iterable):
    """Yield only ``(name, df, col)`` tuples where df is non-empty and col is present."""
    for name, df, col in cases:
        if df is not None and len(df) > 0 and col in df.columns:
            yield name, df, col


def _check_normality(df: pd.DataFrame, metric: str, groupby: str = "ModelGroup", alpha: float = ALPHA) -> pd.DataFrame:
    """Shapiro-Wilk normality test for each group. p > alpha → assume normal."""
    rows = []
    for name, sub in df.groupby(groupby, observed=True):
        vals = sub[metric].dropna()
        if len(vals) >= 3:
            w, p = shapiro(vals)
            rows.append({"Group": name, "N": len(vals), "Shapiro_W": w, "p_value": p, "Normal": p > alpha})
    return pd.DataFrame(rows)


def _check_homogeneity(df: pd.DataFrame, metric: str, groupby: str = "ModelGroup") -> dict:
    """Levene's test for equality of variances. p > ALPHA → assume homogeneous."""
    groups = [sub[metric].dropna().values for _, sub in df.groupby(groupby, observed=True)]
    w, p = levene(*groups)
    return {"Levene_W": w, "p_value": p, "Homogeneous": p > ALPHA}


def _run_anova_with_posthoc(df: pd.DataFrame, metric: str, groupby: str = "ModelGroup", alpha: float = ALPHA):
    """One-way ANOVA + eta-squared + Tukey HSD post-hoc."""
    groups = [sub[metric].dropna().values for _, sub in df.groupby(groupby, observed=True)]
    f_stat, p = f_oneway(*groups)
    grand = df[metric].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = sum((df[metric] - grand) ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    anova = {"F_statistic": f_stat, "p_value": p, "eta_squared": eta_sq, "significant": p < alpha}
    tukey = pairwise_tukeyhsd(df[metric].dropna(), df[groupby].dropna(), alpha=alpha)
    posthoc = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    return anova, posthoc


def _run_kruskal_with_posthoc(df: pd.DataFrame, metric: str, groupby: str = "ModelGroup"):
    """Kruskal-Wallis + pairwise Mann-Whitney U (Bonferroni-corrected)."""
    grouped = {name: sub[metric].dropna().values for name, sub in df.groupby(groupby, observed=True)}
    names = list(grouped.keys())
    h_stat, p = kruskal(*grouped.values())
    n_comp = len(names) * (len(names) - 1) // 2
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            u, p_raw = mannwhitneyu(grouped[a], grouped[b], alternative="two-sided")
            p_corr = min(p_raw * n_comp, 1.0)
            pairs.append({
                "group1": a, "group2": b, "U_statistic": u,
                "p_value": p_raw, "p_corrected": p_corr,
                "significant": p_corr < ALPHA,
            })
    kw = {"H_statistic": h_stat, "p_value": p, "significant": p < ALPHA}
    return kw, pd.DataFrame(pairs)


def run_full_stats_battery(test_cases, groupby_levels=None, baseline: str = "Base") -> dict:
    """Run descriptive + assumption checks + omnibus tests for each metric × groupby."""
    groupby_levels = groupby_levels or ["ExperimentGroup", "Model"]
    out = {}
    for name, df, col in iter_metric_cases(test_cases):
        for level in groupby_levels:
            r = {}
            r["descriptive"] = df.groupby(level, observed=True)[col].agg(
                ["count", "mean", "std", "median", "min", "max"]
            ).round(3)
            r["normality"] = _check_normality(df, col, groupby=level)
            r["homogeneity"] = _check_homogeneity(df, col, groupby=level)
            if df[level].nunique() >= 2:
                r["anova"], r["posthoc_tukey"] = _run_anova_with_posthoc(df, col, groupby=level)
                r["kruskal"], r["posthoc_mannwhitney"] = _run_kruskal_with_posthoc(df, col, groupby=level)
            else:
                r["anova"], r["kruskal"] = None, None
            out[f"{name}_{level}"] = r
    return out


def compare_all_vs_baseline(test_cases, baseline: str = "Base") -> pd.DataFrame:
    """Mann-Whitney U per model vs baseline; BH-FDR within each metric."""
    all_rows = []
    for name, df, col in iter_metric_cases(test_cases):
        base = df.loc[df["Model"].astype(str) == baseline, col].dropna().values
        if len(base) < 2:
            continue
        non_base = sorted(m for m in df["Model"].astype(str).unique() if m != baseline)
        rows, p_raws = [], []
        for m in non_base:
            vals = df.loc[df["Model"].astype(str) == m, col].dropna().values
            if len(vals) < 2:
                continue
            u, p = mannwhitneyu(vals, base, alternative="two-sided")
            d = cohens_d(vals, base)
            rows.append({
                "Metric": name, "Model": m, "N": len(vals),
                "Mean": round(float(vals.mean()), 4),
                "Baseline_Mean": round(float(base.mean()), 4),
                "Delta_Mean": round(float(vals.mean() - base.mean()), 4),
                "U": round(u, 1), "p_raw": p,
                "Cohens_d": round(d, 4), "Effect_Size": interpret_effect_size(d),
            })
            p_raws.append(p)
        if not rows:
            continue
        reject, q_vals = fdr_correct(p_raws)
        for r, q, sig in zip(rows, q_vals, reject):
            r["q_fdr"] = q
            r["Significant"] = bool(sig)
        all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    cols = ["Metric", "Model", "N", "Mean", "Baseline_Mean", "Delta_Mean",
            "U", "p_raw", "q_fdr", "Significant", "Cohens_d", "Effect_Size"]
    return pd.DataFrame(all_rows)[cols].sort_values(["Metric", "q_fdr"]).reset_index(drop=True)


def compare_all_pairwise(test_cases) -> pd.DataFrame:
    """All-pairs Mann-Whitney U per metric; BH-FDR; returns only significant pairs."""
    all_rows = []
    for name, df, col in iter_metric_cases(test_cases):
        models = sorted(df["Model"].astype(str).dropna().unique().tolist())
        if len(models) < 2:
            continue
        vals_by_model = {
            m: df.loc[df["Model"].astype(str) == m, col].dropna().values
            for m in models
        }
        rows, p_raws = [], []
        for i, a in enumerate(models):
            if len(vals_by_model[a]) < 2:
                continue
            for b in models[i + 1:]:
                if len(vals_by_model[b]) < 2:
                    continue
                u, p = mannwhitneyu(vals_by_model[a], vals_by_model[b], alternative="two-sided")
                d = cohens_d(vals_by_model[a], vals_by_model[b])
                rows.append({
                    "Metric": name, "Model_A": a, "Model_B": b,
                    "Mean_A": round(float(vals_by_model[a].mean()), 4),
                    "Mean_B": round(float(vals_by_model[b].mean()), 4),
                    "Delta": round(float(vals_by_model[a].mean() - vals_by_model[b].mean()), 4),
                    "U": round(u, 1), "p_raw": p,
                    "Cohens_d": round(d, 4), "Effect_Size": interpret_effect_size(d),
                })
                p_raws.append(p)
        if not rows:
            continue
        reject, q_vals = fdr_correct(p_raws)
        for r, q in zip(rows, q_vals):
            r["q_fdr"] = q
        all_rows.extend(r for r, sig in zip(rows, reject) if sig)
    if not all_rows:
        return pd.DataFrame()
    cols = ["Metric", "Model_A", "Model_B", "Mean_A", "Mean_B", "Delta",
            "U", "p_raw", "q_fdr", "Cohens_d", "Effect_Size"]
    return pd.DataFrame(all_rows)[cols].sort_values(["Metric", "q_fdr"]).reset_index(drop=True)


def compare_lookahead(test_cases) -> dict:
    """L0 vs L5 Mann-Whitney U: overall and per OracleGroup."""
    out = {}
    for name, df, col in iter_metric_cases(test_cases):
        df = df.copy()
        df["LookAhead"] = df["Model"].apply(lambda m: parse_model_metadata(str(m))["LookAhead"])
        sub = df[df["LookAhead"].isin([0, 5])]
        if len(sub) == 0:
            continue
        results = {}
        l0 = sub[sub["LookAhead"] == 0][col].dropna()
        l5 = sub[sub["LookAhead"] == 5][col].dropna()
        if len(l0) > 1 and len(l5) > 1:
            u, p = mannwhitneyu(l0.values, l5.values, alternative="two-sided")
            d = cohens_d(l0.values, l5.values)
            results["overall"] = {
                "metric": name, "group1_name": "L0", "group2_name": "L5",
                "group1_mean": float(l0.mean()), "group1_n": len(l0),
                "group2_mean": float(l5.mean()), "group2_n": len(l5),
                "U_statistic": u, "p_value": p,
                "cohens_d": d, "effect_size": interpret_effect_size(d),
                "significant": p < ALPHA,
            }
        for og in sorted(g for g in sub["OracleGroup"].unique() if g != "Base"):
            l0g = sub[(sub["LookAhead"] == 0) & (sub["OracleGroup"] == og)][col].dropna()
            l5g = sub[(sub["LookAhead"] == 5) & (sub["OracleGroup"] == og)][col].dropna()
            if len(l0g) > 1 and len(l5g) > 1:
                u, p = mannwhitneyu(l0g.values, l5g.values, alternative="two-sided")
                d = cohens_d(l0g.values, l5g.values)
                results[og] = {
                    "metric": name, "group1_name": f"L0_{og}", "group2_name": f"L5_{og}",
                    "group1_mean": float(l0g.mean()), "group1_n": len(l0g),
                    "group2_mean": float(l5g.mean()), "group2_n": len(l5g),
                    "U_statistic": u, "p_value": p,
                    "cohens_d": d, "effect_size": interpret_effect_size(d),
                    "significant": p < ALPHA,
                }
        out[name] = results
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ║                                  PLOTS                                     ║
# ═══════════════════════════════════════════════════════════════════════════════


def _add_baseline_ref(ax, data: pd.DataFrame, metric: str, baseline: str = "Base") -> None:
    if "Model" not in data.columns or metric not in data.columns:
        return
    base_scores = data[data["Model"] == baseline][metric]
    if len(base_scores) > 0:
        m = base_scores.mean()
        ax.axhline(y=m, color="black", linestyle=":", linewidth=2, label=f"{baseline} mean ({m:.2f})")


def _resolve_ylim(ax, ylim, data_values=None, floor: float = 0.0, padding: float = 0.1) -> None:
    """``None`` keeps mpl default; ``(lo, hi)`` sets fixed; ``"auto"`` tight + padding."""
    if ylim is None:
        return
    if isinstance(ylim, (tuple, list)):
        ax.set_ylim(ylim)
        return
    if ylim == "auto":
        if data_values is not None:
            values = np.asarray(data_values, float)
            values = values[np.isfinite(values)]
        else:
            values = np.array([])
            for patch in ax.patches:
                h = patch.get_height()
                if h > 0:
                    values = np.append(values, h)
            for line in ax.lines:
                y = np.asarray(line.get_ydata(), float)
                values = np.append(values, y[np.isfinite(y)])
        if len(values) == 0:
            return
        lo, hi = float(values.min()), float(values.max())
        pad = (hi - lo) * padding if hi > lo else 0.5
        lower = max(floor, lo - pad) if floor is not None else lo - pad
        upper = hi + pad
        ax.set_ylim(lower, upper)


def _resolve_ctx(ctx, palette, hue_col, model_order, baseline_model, show, default_hue: str = "ModelGroup"):
    """Merge explicit kwargs with a :class:`PlotContext`'s defaults."""
    if ctx is None:
        return (palette or {}, hue_col, model_order, baseline_model, True if show is None else show)
    return (
        palette if palette is not None else (ctx.model_palette or ctx.experiment_palette or {}),
        hue_col if hue_col not in (None, default_hue) else ctx.hue_col,
        model_order if model_order is not None else ctx.model_order,
        baseline_model if baseline_model is not None else ctx.baseline_model,
        show if show is not None else ctx.show,
    )


def _experiment_group_legend(
    ax_or_fig,
    experiment_palette: dict,
    loc: str = "upper left",
    bbox=(1.02, 1),
    title: str = "ExperimentGroup",
    fontsize: int = 8,
    fig_level: bool = False,
) -> None:
    handles = [Patch(facecolor=c, label=g) for g, c in experiment_palette.items()]
    target = ax_or_fig
    if fig_level:
        if hasattr(target, "axes"):
            for axis in target.axes:
                if axis.get_legend():
                    axis.get_legend().remove()
        target.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, title=title, fontsize=fontsize)
    else:
        if target.get_legend():
            target.get_legend().remove()
        target.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, title=title, fontsize=fontsize)


def _get_model_hue_order(model_order, df: pd.DataFrame):
    if model_order:
        return [m for m in model_order if m in df["Model"].unique()]
    return df["Model"].astype(str).unique().tolist()


def plot_metric_by_model(
    df, metric, title=None, ylim=None, padding=0.1, figsize=(15, 6),
    add_baseline=True, palette=None, show=None, model_order=None,
    baseline_model="Base", ctx=None,
):
    """Single-metric bar chart colored by model; dotted baseline reference line."""
    palette, _, model_order, baseline_model, show = _resolve_ctx(
        ctx, palette, "Model", model_order, baseline_model, show, default_hue="Model",
    )
    hue_order = _get_model_hue_order(model_order, df)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(
        x="Model", y=metric, data=df,
        hue="Model", order=hue_order, hue_order=hue_order,
        palette=palette, ax=ax, dodge=False, legend=False,
    )
    ax.set_title(title or f"{metric} by Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _resolve_ylim(ax, ylim, df[metric].values, padding=padding)
    if add_baseline:
        _add_baseline_ref(ax, df, metric, baseline_model)
    exp_palette = ctx.experiment_palette if ctx else EXPERIMENT_PALETTE
    _experiment_group_legend(ax, exp_palette)
    plt.tight_layout()
    if show: plt.show()
    return fig, ax


def plot_subscales(
    df, subscale_cols, subscale_titles, figsize=(18, 5), ylim=None, padding=0.1,
    add_baseline=True, palette=None, show=None, suptitle=None,
    model_order=None, baseline_model="Base", ctx=None,
):
    """Side-by-side bar charts for subscale columns (e.g. WAI-SR or MITI)."""
    palette, _, model_order, baseline_model, show = _resolve_ctx(
        ctx, palette, "Model", model_order, baseline_model, show, default_hue="Model",
    )
    hue_order = _get_model_hue_order(model_order, df)
    n = len(subscale_cols)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    for i, (col, title) in enumerate(zip(subscale_cols, subscale_titles)):
        sns.barplot(
            x="Model", y=col, data=df, ax=axes[i],
            hue="Model", order=hue_order, hue_order=hue_order,
            palette=palette, dodge=False, legend=False,
        )
        axes[i].set_title(title)
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right")
        _resolve_ylim(axes[i], ylim, padding=padding)
        if add_baseline:
            _add_baseline_ref(axes[i], df, col, baseline_model)
    exp_palette = ctx.experiment_palette if ctx else EXPERIMENT_PALETTE
    _experiment_group_legend(axes[-1], exp_palette)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if show: plt.show()
    return fig, axes


def plot_all_metrics_grid(
    dataframes, figsize=(16, 12), palette=None, show=None, group_by="Model",
    model_order=None, baseline_model="Base", ylim=None, padding=0.1, ctx=None,
):
    """Two-column grid of bar charts for every metric in a test_cases list."""
    palette, _, model_order, baseline_model, show = _resolve_ctx(
        ctx, palette, "Model", model_order, baseline_model, show, default_hue="Model",
    )
    n = len(dataframes)
    n_rows = (n + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    for i, (name, df, col) in enumerate(dataframes):
        if group_by == "Model":
            order = [m for m in model_order if m in df["Model"].unique()] if model_order else None
            sns.barplot(
                x="Model", y=col, data=df,
                hue="Model", order=order, hue_order=order,
                palette=palette, ax=axes[i], errorbar=("ci", 95),
                dodge=False, legend=False,
            )
        else:
            exp_palette = ctx.experiment_palette if ctx else palette
            sns.barplot(
                x=group_by, y=col, data=df,
                hue=group_by, palette=exp_palette, ax=axes[i],
                errorbar=("ci", 95), legend=False,
            )
        axes[i].set_title(name, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        plt.setp(
            axes[i].get_xticklabels(),
            rotation=45, ha="right",
            fontsize=8 if group_by == "Model" else 10,
        )
        if "Model" in df.columns and baseline_model in df["Model"].values:
            m = df[df["Model"] == baseline_model][col].mean()
            axes[i].axhline(y=m, color="black", linestyle=":", linewidth=1.5, alpha=0.7)
        _resolve_ylim(axes[i], ylim, padding=padding)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    if n > 0:
        exp_palette = ctx.experiment_palette if ctx else EXPERIMENT_PALETTE
        _experiment_group_legend(fig, exp_palette, loc="upper left", bbox=(1.02, 1), fontsize=8, fig_level=True)
    suffix = "(by Model)" if group_by == "Model" else "(by Group)"
    fig.suptitle(f"Main Outcome Metrics {suffix}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if show: plt.show()
    return fig


def plot_metrics_by_patient_characteristic(
    dfs, metrics, characteristic, group_by: str = "ExperimentGroup",
    palette=None, model_palette=None, figsize=(18, 12), show=None, ylim=None, ctx=None,
):
    """Per-characteristic bar charts across metrics."""
    exp_palette = palette or (ctx.experiment_palette if ctx else EXPERIMENT_PALETTE)
    model_color_palette = model_palette or (ctx.model_palette if ctx else {})
    _, _, _, _, show = _resolve_ctx(ctx, palette, "ExperimentGroup", None, None, show, default_hue="ExperimentGroup")
    n_panels = len(dfs)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols)) if n_panels > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()
    for i, (display_name, df) in enumerate(dfs.items()):
        if i >= len(axes_flat):
            break
        metric_col = metrics[display_name]
        if characteristic not in df.columns:
            continue
        plot_df = df.dropna(subset=[metric_col, characteristic]).copy()
        if plot_df.empty:
            continue
        if isinstance(plot_df["Model"].dtype, pd.CategoricalDtype):
            plot_df["Model"] = plot_df["Model"].cat.remove_unused_categories()
        sns.barplot(
            data=plot_df, x=characteristic, y=metric_col, hue="Model",
            palette=model_color_palette, errorbar=("ci", 95), ax=axes_flat[i],
        )
        axes_flat[i].set_title(f"{display_name} by {characteristic}", fontsize=12, fontweight="bold")
        plt.setp(axes_flat[i].get_xticklabels(), rotation=30, ha="right")
        _resolve_ylim(axes_flat[i], ylim)
        _experiment_group_legend(axes_flat[i], exp_palette, loc="upper left", bbox=(1.02, 1), fontsize=8)
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"Metrics by {characteristic}", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    if show: plt.show()
    return fig, axes


def plot_correlation_heatmaps(merged_df, title: str = "Cross-Metric Correlation", figsize=(12, 6), show: Optional[bool] = True, ctx=None):
    """Side-by-side Pearson / Spearman correlation heatmaps."""
    show = ctx.show if ctx and show is None else (True if show is None else show)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(merged_df.corr(method="pearson"), annot=True, cmap="coolwarm", ax=axes[0], fmt=".3f")
    axes[0].set_title("Pearson")
    sns.heatmap(merged_df.corr(method="spearman"), annot=True, cmap="coolwarm", ax=axes[1], fmt=".3f")
    axes[1].set_title("Spearman")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if show: plt.show()
    return fig, axes


def plot_session_ending(data, palette=None, figsize=None, show=None, model_order=None, ctx=None):
    """Count-plot of how each model's sessions ended."""
    palette, _, _, _, show = _resolve_ctx(ctx, palette, "ModelGroup", None, None, show)
    fig, ax = plt.subplots(1, 1, figsize=figsize or FIG_WIDE)
    sns.countplot(x="Model", hue="session_ended_by", data=data, order=model_order, ax=ax)
    ax.set_title("Session Ending by Model", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Ended by", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if show: plt.show()
    return fig, ax
