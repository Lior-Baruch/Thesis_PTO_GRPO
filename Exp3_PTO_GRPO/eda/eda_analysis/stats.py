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

from .constants import QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, ORTHOGONAL_METRICS
from .data import to_wide

_BOOT_SEED = 12345  # fixed for reproducibility


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


# ── Controlled cross-arm comparisons (matched iteration, paired by persona) ──────
def _common_iters(scores_long: pd.DataFrame, arm_a: str, arm_b: str) -> List[int]:
    """Iterations scored for BOTH arms (the matched-budget comparison points)."""
    ia = set(scores_long.loc[scores_long["arm"] == arm_a, "iteration"])
    ib = set(scores_long.loc[scores_long["arm"] == arm_b, "iteration"])
    return sorted(int(i) for i in (ia & ib))


def _model_at(scores_long: pd.DataFrame, arm: str, it: int) -> Optional[str]:
    s = scores_long[(scores_long["arm"] == arm) & (scores_long["iteration"] == it)]
    return s["model"].iloc[0] if len(s) else None


def _paired_arm_comparison(scores_long: pd.DataFrame, arm_a: str, arm_b: str,
                           metrics: Optional[Sequence[str]] = None, **assign) -> pd.DataFrame:
    """arm_a − arm_b at every common iteration, paired by persona across rubrics.

    Reuses :func:`compare_two_models` per iteration. Returns an EMPTY frame when the two
    arms share no scored iteration (graceful for thin/unscored arms). ``+ => arm_a higher``.
    Extra ``assign`` kwargs are added as constant columns (e.g. ``K=0`` / ``method="PTO"``).

    ⚠ Holm SCOPE: each iteration's ``p_holm`` is corrected across the RUBRICS at that one
    matched-budget point (the family = the rubric set within a single iteration). Concatenating
    iterations does NOT re-pool the correction — ``p_holm`` is per-(iteration) across rubrics, not
    a whole-table correction over iteration×rubric. State this wherever the merged frame is shown.
    """
    rows = []
    for it in _common_iters(scores_long, arm_a, arm_b):
        ma, mb = _model_at(scores_long, arm_a, it), _model_at(scores_long, arm_b, it)
        if ma and mb:
            rows.append(compare_two_models(scores_long, ma, mb, metrics).assign(iteration=it))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for k, v in assign.items():
        out[k] = v
    return out


def paired_method_comparison(scores_long: pd.DataFrame, method_a: str = "PTO",
                             method_b: str = "GRPO", K: int = 0,
                             metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """PTO vs GRPO at MATCHED look-ahead K, paired by persona at every common iteration.

    The thesis's core method comparison as a first-class tidy frame (columns: iteration,
    metric, n, mean_delta, dz, p, p_holm, K). ``+ => method_a higher``. Empty if the two
    arms (``{method}_LA{K}``) share no scored iteration.
    """
    return _paired_arm_comparison(scores_long, f"{method_a}_LA{K}", f"{method_b}_LA{K}",
                                  metrics, K=K)


def paired_k_comparison(scores_long: pd.DataFrame, method: str = "PTO",
                        K_lo: int = 0, K_hi: int = 5,
                        metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """K_lo vs K_hi within ONE method (the look-ahead lever), paired by persona.

    ``+ => K_lo higher``. Empty if the two arms (``{method}_LA{K}``) share no scored iteration
    (the LA5 arms are thin → expect empty/short until the K=5 sweep lands).
    """
    return _paired_arm_comparison(scores_long, f"{method}_LA{K_lo}", f"{method}_LA{K_hi}",
                                  metrics, method=method)


def paired_best_method_comparison(scores_long: pd.DataFrame, method_a: str = "PTO",
                                  method_b: str = "GRPO", K: int = 0,
                                  metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """method_a at its own-oracle BEST iteration vs method_b at ITS best, paired by persona.

    The model-selection twin of :func:`paired_method_comparison` (which walks matched
    iterations): each arm's best checkpoint comes from
    :func:`~eda_analysis.data.best_per_experiment`, so the two sides may sit at DIFFERENT
    iterations (e.g. PTO@10 vs GRPO@8). Persona pairing stays valid across iterations — every
    iteration reshuffles the SAME 96 personas. Columns: metric, n, mean_delta, dz, p, p_holm,
    iter_a, iter_b, K. ``+ => method_a higher``. Empty if either arm has no non-base iteration.
    """
    from .data import best_per_experiment  # deferred: data imports stats-free modules only
    arm_a, arm_b = f"{method_a}_LA{K}", f"{method_b}_LA{K}"
    sub = scores_long[scores_long["arm"].isin([arm_a, arm_b])]
    if sub.empty:
        return pd.DataFrame()
    _, summ = best_per_experiment(sub)
    info = {r["arm"]: r for _, r in summ.iterrows()} if not summ.empty else {}
    if arm_a not in info or arm_b not in info:
        return pd.DataFrame()
    out = compare_two_models(scores_long, info[arm_a]["best_model"],
                             info[arm_b]["best_model"], metrics)
    if out.empty:
        return out
    out["iter_a"] = int(info[arm_a]["best_iteration"])
    out["iter_b"] = int(info[arm_b]["best_iteration"])
    out["K"] = K
    return out


# ── Trajectory ───────────────────────────────────────────────────────────────
def trajectory_test(scores_long: pd.DataFrame, arm: str, metric: str) -> dict:
    """Is *metric* climbing over iterations for *arm*? Spearman + OLS slope on raw rows.

    ⚠ The ``p`` (Spearman) and ``ols_slope`` pool every persona×iteration row and treat them as
    independent — but personas repeat across iterations, so the p-value is DESCRIPTIVE only. Use
    :func:`friedman_trajectory` for the repeated-measures-correct omnibus (that is why the thesis
    ``slope_by_arm`` table reports ρ/slope but NOT this p).
    """
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
    default = WARMTH_RUBRICS + ORTHOGONAL_METRICS   # the canonical 10-metric factor space
    metrics = [m for m in (metrics or default) if m in wide.columns]
    X = wide[metrics].dropna()
    if len(X) < 3 or len(metrics) < 2:
        return {"metrics": metrics, "explained_variance_ratio": [], "pc1_loadings": {}}
    Z = StandardScaler().fit_transform(X)
    pca = PCA().fit(Z)
    return {"metrics": metrics,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pc1_loadings": dict(zip(metrics, pca.components_[0].round(3).tolist()))}


def rubric_factor_space(scores_long_or_wide, metrics: Optional[Sequence[str]] = None) -> Optional[dict]:
    """2-component PCA for the PC1×PC2 factor-space scatter (the halo-breaking figure).

    Returns ``{points (n×2), loadings {metric:(x,y)}, explained (2,), metrics}`` or ``None`` if a
    2-component PCA can't be fit. ``points`` are the standardized rows projected onto PC1/PC2;
    ``loadings`` are the per-metric component weights (the arrow directions).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    wide = scores_long_or_wide if "Q1Q2" in scores_long_or_wide.columns else to_wide(scores_long_or_wide)
    default = WARMTH_RUBRICS + ORTHOGONAL_METRICS   # the canonical 10-metric factor space
    metrics = [m for m in (metrics or default) if m in wide.columns]
    X = wide[metrics].dropna()
    if len(X) < 3 or len(metrics) < 2:
        return None
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Z)
    pts = pca.transform(Z)
    comps = pca.components_
    return {"points": pts, "metrics": metrics,
            "explained": pca.explained_variance_ratio_[:2].tolist(),
            "loadings": {m: (float(comps[0, i]), float(comps[1, i])) for i, m in enumerate(metrics)}}


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


# ── Research-grade rigor: repeated-measures + bootstrap + a main-results table ───
def effect_label(d: float) -> str:
    """Cohen's-style magnitude label for |effect| (dz / Cliff's delta share thresholds)."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return ""
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


# ── Thin-arm hygiene (kills NaN rows from under-scored arms) ─────────────────────
def thin_arms(scores_long: pd.DataFrame, min_iters: int = 3) -> List[str]:
    """Arms with fewer than ``min_iters`` scored iterations (their Friedman/slope rows are NaN)."""
    n = scores_long.groupby("arm")["iteration"].nunique()
    return sorted(n[n < min_iters].index)


def filter_thin_arms(df: pd.DataFrame, scores_long: pd.DataFrame, *, min_iters: int = 3,
                     arm_col: str = "arm", verbose: bool = True) -> pd.DataFrame:
    """Drop rows of *df* belonging to thin arms (so stat tables don't carry NaN rows)."""
    thin = set(thin_arms(scores_long, min_iters))
    if not thin or arm_col not in df.columns:
        return df
    if verbose:
        print(f"  [stats] dropping thin arms (<{min_iters} scored iters): {sorted(thin)}")
    return df[~df[arm_col].isin(thin)].reset_index(drop=True)


# ── Partial-conversation reward reliability (Exp3, from generations.jsonl) ───────
def rank_agreement_by_nturns(branch_reliability: pd.DataFrame, scores_long: pd.DataFrame, *,
                             metric: str = "Q1Q2", min_pairs: int = 20) -> pd.DataFrame:
    """Does the partial-conv training reward rank conversations like the full-conv eval does?

    The Exp2 ``Partial_Conv_Oracle_EDA`` statistic rebuilt on Exp3 data: join each branch's proxy
    score (:func:`training.load_branch_reliability`) to the full-conversation eval of the same
    conversation (``eval_iter`` ↔ ``scores_long.iteration``, ``conversation_id`` ↔ ``file_index``),
    then per ``n_turns`` measure the fraction of conversation pairs whose proxy-difference sign
    matches the eval-difference sign. 0.5 = chance; rises toward 1 as the cut lengthens. Pairs are
    formed WITHIN (arm, eval_iter, n_turns) so both scores share a model state, then pooled per
    (arm, n_turns). Returns ``(arm, n_turns, agreement, n_pairs)`` (bins with < ``min_pairs`` dropped).
    """
    from itertools import combinations
    if branch_reliability is None or branch_reliability.empty:
        return pd.DataFrame(columns=["arm", "n_turns", "agreement", "n_pairs"])
    ev = (scores_long[scores_long["questionnaire"] == metric]
          .rename(columns={"iteration": "eval_iter", "file_index": "conversation_id"})
          .groupby(["arm", "eval_iter", "conversation_id"])["score"].mean()
          .rename("eval_score").reset_index())
    df = branch_reliability.merge(ev, on=["arm", "eval_iter", "conversation_id"], how="inner")
    if df.empty:
        return pd.DataFrame(columns=["arm", "n_turns", "agreement", "n_pairs"])
    agg = {}  # (arm, n_turns) -> [n_correct, n_total]
    for (arm, _ei, nt), g in df.groupby(["arm", "eval_iter", "n_turns"], observed=True):
        gg = g.groupby("conversation_id").agg(proxy=("proxy_score", "mean"),
                                              evl=("eval_score", "first"))
        if len(gg) < 2:
            continue
        p = gg["proxy"].to_numpy(); e = gg["evl"].to_numpy()
        for i, j in combinations(range(len(gg)), 2):
            dp, de = p[i] - p[j], e[i] - e[j]
            if dp == 0 or de == 0:
                continue
            c, t = agg.get((arm, int(nt)), (0, 0))
            agg[(arm, int(nt))] = (c + int((dp > 0) == (de > 0)), t + 1)
    rows = [{"arm": a, "n_turns": nt, "agreement": c / t, "n_pairs": t}
            for (a, nt), (c, t) in agg.items() if t >= min_pairs]
    return pd.DataFrame(rows).sort_values(["arm", "n_turns"]).reset_index(drop=True)


def item_endpoint_deltas(items_long: pd.DataFrame, *,
                         target_iter_by_arm: Optional[dict] = None,
                         short=None, group_of=None) -> pd.DataFrame:
    """Per (arm, item): base mean, target-iteration mean, and Δ — the generic
    "which items drive the change" table behind every questionnaire's delta bars.

    ``items_long`` is any long frame with ``(arm, iteration, is_base, item, score)`` — the
    per-conversation item frames from :func:`~eda_analysis.data.load_items` OR a melted per-iter
    detail frame (MITI/PCT/MICI; a mean over one row is the value itself, so the same groupby
    serves both). Target iteration per arm = ``target_iter_by_arm[arm]`` (e.g. the
    :func:`~eda_analysis.data.best_iteration_by_arm` map) with fallback to that arm's FINAL
    iteration when the map is ``None``/missing the arm.

    ``short`` labels items: a mapping (``Q2_ITEM_SHORT``), a callable (``display_label``), or
    ``None`` (use the frame's ``short`` column if present, else ``str(item)``). ``group_of``
    optionally maps item -> face-content group (colored bars). Output columns:
    ``arm, item, short, group, target_iter, base, target, delta``.
    """
    if items_long is None or items_long.empty:
        return pd.DataFrame(columns=["arm", "item", "short", "group",
                                     "target_iter", "base", "target", "delta"])

    frame_short = ("short" in items_long.columns)

    def _short(i, g):
        if callable(short):
            return short(i)
        if short is not None:
            return short.get(i, str(i))
        if frame_short:
            s = g.loc[g["item"] == i, "short"]
            if len(s):
                return s.iloc[0]
        return str(i)

    rows = []
    for arm, g in items_long.groupby("arm", sort=True):
        nb = g[~g["is_base"]]
        if nb.empty:
            continue
        tgt = (target_iter_by_arm or {}).get(arm)
        tgt = int(tgt) if tgt is not None else int(nb["iteration"].max())
        base = g[g["is_base"]].groupby("item")["score"].mean()
        target = g[g["iteration"] == tgt].groupby("item")["score"].mean()
        for i in pd.unique(g["item"]):                     # preserve item order (1..n / metric order)
            b, t = base.get(i, np.nan), target.get(i, np.nan)
            rows.append({"arm": arm, "item": i, "short": _short(i, g),
                         "group": (group_of or {}).get(i, ""), "target_iter": tgt,
                         "base": round(float(b), 3), "target": round(float(t), 3),
                         "delta": round(float(t - b), 3)})
    return pd.DataFrame(rows)


def q2_item_endpoint_deltas(q2_long: pd.DataFrame,
                            target_iter_by_arm: Optional[dict] = None) -> pd.DataFrame:
    """Per (arm, Q2 item): base/target means + Δ — the Q2 reward-composition table.

    Thin wrapper over :func:`item_endpoint_deltas` with the Q2 labels/groups from
    ``constants.Q2_ITEM_SHORT`` / ``Q2_ITEM_GROUP_OF`` (face-content grouping — analytical, not a
    validated subscale). Feeds :func:`plotting.q2_item_delta_bars`; drop into ``save_table``.
    """
    from .constants import Q2_ITEM_SHORT, Q2_ITEM_GROUP_OF
    return item_endpoint_deltas(q2_long, target_iter_by_arm=target_iter_by_arm,
                                short=Q2_ITEM_SHORT, group_of=Q2_ITEM_GROUP_OF)


def friedman_trajectory(scores_long: pd.DataFrame, arm: str, metric: str) -> dict:
    """Repeated-measures omnibus across iterations (the matched-persona-correct test).

    Pivots persona_id × iteration (complete after persona recovery) and runs Friedman χ² + Kendall's W
    effect size. Preferred over an independent-group Kruskal–Wallis omnibus for this design.
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


def main_results_table(scores_long: pd.DataFrame, target: str = "final",
                       metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The single authoritative thesis table: per (arm × rubric), target vs base.

    ``target`` ∈ {"final" (max iteration), "best" (best iteration by own oracle, via
    :func:`~eda_analysis.data.best_per_experiment`)}. Columns: base/target means, Δ, paired Cohen's ``dz`` +
    label, Wilcoxon ``p`` (Holm-corrected across rubrics within arm), bootstrap CI, trajectory Spearman
    ρ + OLS slope. Paired by persona throughout.
    """
    metrics = [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in set(scores_long["questionnaire"])]
    # resolve the target model per arm
    target_model = {}
    if target == "best":
        from .data import best_per_experiment  # select merged into data.py
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
