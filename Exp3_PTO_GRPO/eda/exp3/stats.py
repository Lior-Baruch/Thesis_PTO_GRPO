"""
stats.py — statistics that exploit the matched-persona (repeated-measures) design.

The same 96 personas recur across every iteration and both methods, so comparisons
are **paired by persona** (Wilcoxon signed-rank, paired bootstrap CIs, Cohen's dz),
which is far stronger than the old independent-group Mann-Whitney. Caveat: pairing
controls persona difficulty, NOT patient-simulator stochasticity (the patient api
seed and the policy both differ across models) — so treat it as matched-subjects,
not a deterministic re-run.

Also: model ranking across rubrics; iteration-trajectory tests; multiplicity
correction; inter-rubric correlation + PCA (to quantify why "all 6 metrics up" is
weak evidence of multi-dimensional skill gain).
"""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import QUESTIONNAIRE_ORDER, to_wide

_BOOT_SEED = 12345  # fixed for reproducibility


# ── Ranking across rubrics ───────────────────────────────────────────────────
def model_means(scores_long: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean per questionnaire (index = model, + arm/iteration meta)."""
    means = (scores_long.groupby(["model", "questionnaire"], observed=True)["score"]
             .mean().unstack())
    meta = (scores_long[["model", "arm", "method", "K", "iteration", "is_base"]]
            .drop_duplicates("model").set_index("model"))
    return meta.join(means)


def rank_table(scores_long: pd.DataFrame, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-questionnaire rank (1 = best) + average rank across rubrics, per model."""
    mm = model_means(scores_long)
    metrics = [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in mm.columns]
    ranks = mm[metrics].rank(ascending=False, method="min")
    ranks.columns = [f"rank_{c}" for c in metrics]
    out = mm[["arm", "iteration", "is_base"] + metrics].join(ranks)
    out["AvgRank"] = ranks.mean(axis=1)
    return out.sort_values("AvgRank")


# ── Paired comparisons (by persona) ──────────────────────────────────────────
def _paired_deltas(wide: pd.DataFrame, metric: str, model_a: str, model_b: str,
                   key: str = "persona_id") -> np.ndarray:
    a = wide[wide["model"] == model_a][[key, metric]].dropna()
    b = wide[wide["model"] == model_b][[key, metric]].dropna()
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    return (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI for the mean of paired deltas."""
    d = np.asarray(deltas, float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(_BOOT_SEED)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    return tuple(np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def paired_compare(wide: pd.DataFrame, metric: str, model_a: str, model_b: str,
                   key: str = "persona_id") -> dict:
    """Paired comparison model_a − model_b on *metric*: Wilcoxon + dz + bootstrap CI.

    Positive ``mean_delta`` ⇒ model_a scores higher. Returns NaNs gracefully if
    fewer than ~3 paired observations or all deltas are zero.
    """
    d = _paired_deltas(wide, metric, model_a, model_b, key)
    d = d[~np.isnan(d)]
    out = {"metric": metric, "model_a": model_a, "model_b": model_b, "n": int(d.size),
           "mean_delta": float(np.mean(d)) if d.size else np.nan,
           "median_delta": float(np.median(d)) if d.size else np.nan,
           "dz": np.nan, "W": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if d.size >= 3 and np.any(d != 0):
        sd = d.std(ddof=1)
        out["dz"] = float(d.mean() / sd) if sd > 0 else np.nan
        try:
            w = stats.wilcoxon(d, zero_method="wilcox", correction=False)
            out["W"], out["p"] = float(w.statistic), float(w.pvalue)
        except Exception:
            pass
        out["ci_low"], out["ci_high"] = bootstrap_ci(d)
    return out


def paired_vs_base(scores_long: pd.DataFrame, arm: str, metric: str) -> pd.DataFrame:
    """Each iteration of *arm* vs that arm's own base, paired by persona (+Holm p)."""
    wide = to_wide(scores_long[scores_long["arm"] == arm])
    base = wide[wide["is_base"]]
    if base.empty:
        return pd.DataFrame()
    base_model = base["model"].iloc[0]
    rows = []
    for it in sorted(wide.loc[~wide["is_base"], "iteration"].unique()):
        model = wide[wide["iteration"] == it]["model"].iloc[0]
        rows.append({"arm": arm, "iteration": int(it), **paired_compare(wide, metric, model, base_model)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"].to_numpy())
    return out


def compare_two_models(scores_long: pd.DataFrame, model_a: str, model_b: str,
                       metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Paired model_a − model_b across rubrics (e.g. PTO vs GRPO at a matched iter)."""
    wide = to_wide(scores_long)
    metrics = [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in wide.columns]
    out = pd.DataFrame([paired_compare(wide, m, model_a, model_b) for m in metrics])
    if not out.empty:
        out["p_holm"] = holm(out["p"].to_numpy())
    return out


# ── Trajectory ───────────────────────────────────────────────────────────────
def trajectory_test(scores_long: pd.DataFrame, arm: str, metric: str) -> dict:
    """Is *metric* climbing over iterations for *arm*? Spearman + OLS slope on raw rows."""
    g = scores_long[(scores_long["arm"] == arm) & (scores_long["questionnaire"] == metric)]
    if g["iteration"].nunique() < 3:
        return {"arm": arm, "metric": metric, "spearman_rho": np.nan, "p": np.nan,
                "ols_slope": np.nan, "peak_iter": np.nan, "final_iter": np.nan}
    x = g["iteration"].to_numpy(float)
    y = g["score"].to_numpy(float)
    rho, p = stats.spearmanr(x, y)
    lin = stats.linregress(x, y)
    per_iter = g.groupby("iteration")["score"].mean()
    return {"arm": arm, "metric": metric, "spearman_rho": float(rho), "p": float(p),
            "ols_slope": float(lin.slope), "peak_iter": int(per_iter.idxmax()),
            "final_iter": int(per_iter.index.max())}


# ── Inter-rubric structure ───────────────────────────────────────────────────
def rubric_correlation(scores_long_or_wide, metrics: Optional[Sequence[str]] = None,
                       method: str = "spearman") -> pd.DataFrame:
    """Correlation matrix among rubric scores (per-conversation, pooled)."""
    wide = scores_long_or_wide if "Q1Q2" in scores_long_or_wide.columns else to_wide(scores_long_or_wide)
    metrics = [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in wide.columns]
    return wide[metrics].corr(method=method)


def rubric_pca(scores_long_or_wide, metrics: Optional[Sequence[str]] = None) -> dict:
    """PCA on standardized rubric scores → explained variance + PC1 loadings.

    A dominant PC1 means the rubrics largely measure one latent 'good therapist'
    factor — i.e. improving all of them together is weak evidence of multi-skill gain.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    wide = scores_long_or_wide if "Q1Q2" in scores_long_or_wide.columns else to_wide(scores_long_or_wide)
    metrics = [m for m in (metrics or ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI"]) if m in wide.columns]
    X = wide[metrics].dropna()
    if len(X) < 3 or len(metrics) < 2:
        return {"metrics": metrics, "explained_variance_ratio": [], "pc1_loadings": {}}
    Z = StandardScaler().fit_transform(X)
    pca = PCA().fit(Z)
    return {"metrics": metrics,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pc1_loadings": dict(zip(metrics, pca.components_[0].round(3).tolist()))}


# ── Multiplicity ─────────────────────────────────────────────────────────────
def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni step-down adjusted p-values (NaNs preserved)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = idx.size
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        prev = max(prev, adj)
        out[i] = prev
    return out


def fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values (NaNs preserved)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = idx.size
    prev = 1.0
    for rank in range(m - 1, -1, -1):          # largest p first
        i = order[rank]
        adj = min(prev, p[i] * m / (rank + 1))
        prev = adj
        out[i] = adj
    return out


# ── Familiar Exp2-style battery (independent-group) ──────────────────────────
def omnibus(scores_long: pd.DataFrame, metric: str, group: str = "model") -> dict:
    """Kruskal–Wallis across *group* for *metric* + epsilon-squared effect size.

    The non-parametric one-way 'is anything different at all?' test (the old
    `run_full_stats_battery` omnibus), pooled over personas.
    """
    g = scores_long[scores_long["questionnaire"] == metric]
    samples = [s["score"].to_numpy() for _, s in g.groupby(group, observed=True)]
    samples = [s for s in samples if len(s) > 0]
    k, n = len(samples), sum(len(s) for s in samples)
    if k < 2:
        return {"metric": metric, "H": np.nan, "p": np.nan, "eps_sq": np.nan, "k": k, "n": n}
    H, p = stats.kruskal(*samples)
    eps_sq = (H - k + 1) / (n - k) if n > k else np.nan       # epsilon-squared
    return {"metric": metric, "H": float(H), "p": float(p),
            "eps_sq": float(eps_sq), "k": k, "n": n}


def mannwhitney_vs_base(scores_long: pd.DataFrame, arm: str, metric: str) -> pd.DataFrame:
    """Each iteration of *arm* vs its own base, **independent-group** Mann–Whitney U + FDR.

    The familiar Exp2 treatment (treats the 96 personas as independent samples;
    contrast with the persona-paired :func:`paired_vs_base`). Effect size = Cliff's δ.
    """
    g = scores_long[(scores_long["arm"] == arm) & (scores_long["questionnaire"] == metric)]
    base = g[g["is_base"]]["score"].to_numpy()
    if base.size == 0:
        return pd.DataFrame()
    rows = []
    for it in sorted(g.loc[~g["is_base"], "iteration"].unique()):
        x = g[g["iteration"] == it]["score"].to_numpy()
        if x.size == 0:
            continue
        try:
            U, p = stats.mannwhitneyu(x, base, alternative="two-sided")
            cliffs = 2 * U / (x.size * base.size) - 1
        except ValueError:
            U, p, cliffs = np.nan, np.nan, np.nan
        rows.append({"arm": arm, "iteration": int(it), "n_iter": x.size, "n_base": base.size,
                     "median_delta": float(np.median(x) - np.median(base)),
                     "cliffs_delta": float(cliffs) if cliffs == cliffs else np.nan,
                     "U": float(U) if U == U else np.nan, "p": float(p) if p == p else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = fdr(out["p"].to_numpy())
    return out


# ── Research-grade rigor: repeated-measures + bootstrap + a main-results table ───
def effect_label(d: float) -> str:
    """Cohen's-style magnitude label for |effect| (dz / Cliff's delta share thresholds)."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return ""
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def friedman_trajectory(scores_long: pd.DataFrame, arm: str, metric: str) -> dict:
    """Repeated-measures omnibus across iterations (the matched-persona-correct test).

    Pivots persona_id × iteration (complete after persona recovery) and runs Friedman χ² + Kendall's W
    effect size. Preferred over the independent-group :func:`omnibus` for this design.
    """
    g = scores_long[(scores_long["arm"] == arm) & (scores_long["questionnaire"] == metric)]
    if "persona_id" not in g.columns:
        return {"arm": arm, "metric": metric, "chi2": np.nan, "p": np.nan, "kendall_w": np.nan,
                "k_iters": np.nan, "n_personas": np.nan}
    piv = g.pivot_table(index="persona_id", columns="iteration", values="score").dropna()
    k, n = piv.shape[1], piv.shape[0]
    if k < 3 or n < 3:
        return {"arm": arm, "metric": metric, "chi2": np.nan, "p": np.nan, "kendall_w": np.nan,
                "k_iters": k, "n_personas": n}
    chi2, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    return {"arm": arm, "metric": metric, "chi2": float(chi2), "p": float(p),
            "kendall_w": float(chi2 / (n * (k - 1))), "k_iters": int(k), "n_personas": int(n)}


def mean_ci_by_iter(scores_long: pd.DataFrame, arm: str, metric: str, n_boot: int = 2000) -> pd.DataFrame:
    """Per-iteration mean + bootstrap 95% CI (clean trajectory table for the thesis)."""
    g = scores_long[(scores_long["arm"] == arm) & (scores_long["questionnaire"] == metric)]
    rows = []
    for it in sorted(g["iteration"].unique()):
        v = g[g["iteration"] == it]["score"].to_numpy()
        lo, hi = bootstrap_ci(v, n_boot=n_boot)
        rows.append({"arm": arm, "metric": metric, "iteration": int(it), "n": v.size,
                     "mean": float(np.mean(v)), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main_results_table(scores_long: pd.DataFrame, target: str = "final",
                       metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The single authoritative thesis table: per (arm × rubric), target vs base.

    ``target`` ∈ {"final" (max iteration), "best" (best iteration by own oracle, via
    :func:`~exp3.select.best_per_experiment`)}. Columns: base/target means, Δ, paired Cohen's ``dz`` +
    label, Wilcoxon ``p`` (Holm-corrected across rubrics within arm), bootstrap CI, trajectory Spearman
    ρ + OLS slope. Paired by persona throughout.
    """
    from . import QUESTIONNAIRE_ORDER, to_wide
    metrics = [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in set(scores_long["questionnaire"])]
    # resolve the target model per arm
    target_model = {}
    if target == "best":
        from .select import best_per_experiment
        _, summ = best_per_experiment(scores_long)
        target_model = dict(zip(summ.get("arm", []), summ.get("best_model", [])))
    rows = []
    for arm in sorted(scores_long["arm"].unique()):
        a = scores_long[scores_long["arm"] == arm]
        base_rows = a[a["is_base"]]
        if base_rows.empty:
            continue
        base_model = base_rows["model"].iloc[0]
        if target == "best":
            tgt_model = target_model.get(arm)
            if not tgt_model:
                continue
            tgt_iter = int(a[a["model"] == tgt_model]["iteration"].iloc[0])
        else:  # final
            tgt_iter = int(a.loc[~a["is_base"], "iteration"].max()) if (~a["is_base"]).any() else None
            if tgt_iter is None:
                continue
            tgt_model = a[a["iteration"] == tgt_iter]["model"].iloc[0]
        wide = to_wide(a)
        arm_rows = []
        for m in metrics:
            if m not in wide.columns:
                continue
            cmp = paired_compare(wide, m, tgt_model, base_model)
            base_mean = float(a[(a["questionnaire"] == m) & (a["is_base"])]["score"].mean())
            tgt_mean = float(a[(a["questionnaire"] == m) & (a["iteration"] == tgt_iter)]["score"].mean())
            traj = trajectory_test(scores_long, arm, m)
            arm_rows.append({
                "arm": arm, "rubric": m, "base": round(base_mean, 3),
                "target_iter": tgt_iter, "target": round(tgt_mean, 3),
                "delta": round(cmp["mean_delta"], 3), "dz": round(cmp["dz"], 3),
                "effect": effect_label(cmp["dz"]), "wilcoxon_p": cmp["p"],
                "ci_low": round(cmp["ci_low"], 3), "ci_high": round(cmp["ci_high"], 3),
                "traj_rho": round(traj["spearman_rho"], 3), "traj_slope": round(traj["ols_slope"], 4),
            })
        if arm_rows:
            ps = holm(np.array([r["wilcoxon_p"] for r in arm_rows], dtype=float))
            for r, ph in zip(arm_rows, ps):
                r["wilcoxon_p"] = round(r["wilcoxon_p"], 4) if r["wilcoxon_p"] == r["wilcoxon_p"] else np.nan
                r["p_holm"] = round(float(ph), 4) if ph == ph else np.nan
            rows.extend(arm_rows)
    return pd.DataFrame(rows)
