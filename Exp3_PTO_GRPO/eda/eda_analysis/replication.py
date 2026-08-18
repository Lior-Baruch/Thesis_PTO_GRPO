"""replication.py — three ICLR look-ahead-paper claims re-tested on Exp3, plus the session-shape K
contrast and the selection-level length push.

**What it computes** (all four arms — PTO_LA0 / PTO_LA5 / GRPO_LA0 / GRPO_LA5 — on one axis):

1. **Session shape** (deterministic text metrics from the transcripts, judge-invariant): ``conv_len``,
   ``n_th_turns``, ``mean_turn_len``, ``q_per_turn``, ``loop`` — persona-paired K0 - K5 within each
   method at every matched iteration (:func:`session_shape_paired`), the per-arm mean +- SE
   trajectories (:func:`session_shape_levels`), the base -> final length endpoints
   (:func:`length_endpoints`) and the ICLR "K=5 gives shorter conversations" claim at the endpoints
   (:func:`length_kcontrast`).
2. **ICLR "stability" claim** (K=5 has the lowest SD): per arm x iteration SD / IQR / ceiling shares
   of Q1, Q2, Q1Q2 under BOTH graders (:func:`sd_by_iter`); Brown-Forsythe (independent-groups) and
   Pitman-Morgan (persona-paired) variance tests K0 vs K5 per matched iteration (:func:`sd_tests`);
   the per-(judge, method, rubric) tally (:func:`sd_tally`); where the lowest SD sits and how SD
   tracks the mean (:func:`sd_summary`); ceiling-compression check = share of conversations at
   Q1Q2 >= 4.5 / == 5 by cooperation level (:func:`ceiling`).
3. **Selection-level length push + praise weights** (NO recompute): the tracked preference tables
   ``update_lexical_push`` + ``generation_pool_means`` joined into one compact table
   (:func:`selection_table`).
4. :func:`replication_numbers` — every quotable number as ``{dotted.key: {value, source, note}}``
   for ``exports.save_numbers``.

Figures live in :mod:`eda_analysis.plotting.replication` (:func:`shape_fig`, :func:`sd_fig`).

**Sign convention:** + => K=0 higher (``K0 - K5``), everywhere. Pitman-Morgan ``pm_r > 0`` => K=0
MORE dispersed (the same sign applied to variance). ``sd_ratio_K5_over_K0 < 1`` => the K=5 arm is
LESS dispersed. :func:`length_kcontrast` additionally reports ``K5_minus_K0`` (positive = K=5
LONGER) because that is how the ICLR claim is phrased.

**Pairing unit:** ``persona_id`` (the per-iteration file shuffle replayed by
:func:`eda_analysis.data.attach_personas`; NEVER ``file_index``). Group-level tables (selection)
have no persona pairing. Brown-Forsythe treats the two arms as independent groups (pairing not used).

**Censoring:** GRPO_LA5 is right-censored at iteration 5 (its K=0 sibling runs to 10), so matched
iterations are 0..10 for PTO and 0..5 for GRPO. **Iteration 0** = two INDEPENDENT base draws
(same base policy) per method — a free noise-floor row for every K contrast; the SD tally and the
lowest-SD summary exclude it (trained iterations 1..N only).

**Caveats kept from the paper generator.**
- Length metrics are unvalenced (longer is not better).
- On a bounded 1-5 scale a higher mean mechanically compresses SD: read SD next to the mean, and
  read the ceiling table — a low arm SD that comes with a high ceiling share is scale compression,
  not stability. The held-out judge (Claude Haiku 4.5) never awards >= 4.5 on Q1Q2 (its max is
  4.25), so its ceiling shares are 0 by construction.
- ``p_holm`` in the shape table is Holm-corrected WITHIN each (method, metric) family ACROSS
  iterations; the tracked ``k_paired_channels`` sheet corrects across channels within an iteration
  instead — same delta / dz / p, different ``p_holm`` scope.
- The selection table is COPIED from the tracked preference tables, not recomputed; it is the
  primary training oracle by construction (``generations.jsonl`` records the training oracle's own
  selection). ``train_iter n`` samples from the iter-start policy, i.e. the eval set's
  ``model_iter_{n-1}`` (= ``policy_iteration``).
- Two graders are always side by side, never averaged (the primary WAS the training reward; the
  second judge is held out).
- Bootstrap CIs use :data:`eda_analysis.constants.BOOT_SEED` (the package seed) where the paper
  generator used seed 0, so CI bounds may differ from the frozen fixture in the third decimal;
  means / dz / p / SDs / test statistics are identical.

**Provenance.** Promoted 2026-08-18 from
``papers/2026_lookahead_pto_grpo/analysis/session_shape_stability.py`` (the paper's generator; its
frozen ``tables/session_shape_stability_*.csv`` + ``analysis/out/session_shape_stability.json`` are
the fixture this module reproduces). Contract as everywhere in the analysis layer: functions take
frames (``scores_long`` per judge, arms, the text-metrics frame) and return tidy frames / dicts, NO
disk writes — the notebooks (``lookahead/replication.ipynb`` for §2, ``lookahead/behaviour.ipynb``
for §1 + §3) own ``exports.*``.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sps

from .constants import judge_dirname
from .stats import holm, paired_arrays

__all__ = [
    "FOUR_ARMS", "METHODS", "SHAPE_METRICS", "SHAPE_UNITS", "STAB_METRICS", "COOP_LABEL",
    "SIGN", "PAIR", "CENSOR", "ITER0",
    "k_of", "method_of", "brown_forsythe", "pitman_morgan", "read_md_table",
    "shape_text_metrics", "session_shape_levels", "session_shape_paired",
    "length_endpoints", "length_kcontrast",
    "sd_by_iter", "sd_tests", "sd_tally", "sd_summary", "ceiling",
    "selection_table", "default_selection_dirs", "replication_numbers", "CAPTIONS",
]

FOUR_ARMS = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]
METHODS = ["PTO", "GRPO"]
SHAPE_METRICS = ["conv_len", "n_th_turns", "mean_turn_len", "q_per_turn", "loop"]
SHAPE_UNITS = {"conv_len": "utterances / conversation", "n_th_turns": "therapist turns / conversation",
               "mean_turn_len": "chars / therapist turn", "q_per_turn": "'?' / therapist turn",
               "loop": "share of conversations with a verbatim-repeated therapist turn"}
LENGTH_METRICS = ["conv_len", "n_th_turns", "mean_turn_len"]
STAB_METRICS = ["Q1", "Q2", "Q1Q2"]
COOP_LABEL = {"Low": "Resistant", "High": "Cooperative", "StartLowAndChangesToHigh": "WarmsUp"}
COOP_ORDER = ["Resistant", "WarmsUp", "Cooperative"]

# Caption fragments (verbatim from the paper generator) — the notebook composes captions from them.
SIGN = "Sign: + => K=0 higher (K0 - K5)."
PAIR = "Pairing unit: persona_id (the per-iteration file shuffle replayed; never file_index)."
CENSOR = "GRPO_LA5 is right-censored at iteration 5 (its K=0 sibling runs to 10)."
ITER0 = "Iteration 0 = two INDEPENDENT base draws (same base policy) — a free noise-floor row."


def k_of(arm: str) -> int:
    return int(arm.split("_LA")[1])


def method_of(arm: str) -> str:
    return arm.split("_")[0]


# ── helpers ──────────────────────────────────────────────────────────────────

def _pair_wide(df: pd.DataFrame, value: str, arm_a: str, arm_b: str, it: int):
    """Persona-aligned (a, b) arrays of *value* for two arms at one iteration."""
    a = df[(df["arm"] == arm_a) & (df["iteration"] == it)][["persona_id", value]].dropna()
    b = df[(df["arm"] == arm_b) & (df["iteration"] == it)][["persona_id", value]].dropna()
    m = a.merge(b, on="persona_id", suffixes=("_a", "_b"))
    return m[f"{value}_a"].to_numpy(float), m[f"{value}_b"].to_numpy(float)


def _common_iters(df, arm_a, arm_b) -> List[int]:
    ia = set(df.loc[df["arm"] == arm_a, "iteration"]); ib = set(df.loc[df["arm"] == arm_b, "iteration"])
    return sorted(int(i) for i in ia & ib)


def _iqr(x) -> float:
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if x.size else np.nan


def brown_forsythe(a, b) -> dict:
    """Brown-Forsythe = Levene on |x - group median|; two INDEPENDENT groups (pairing not used).
    Returns ``bf_W, bf_p``."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size < 3 or b.size < 3:
        return dict(bf_W=np.nan, bf_p=np.nan)
    r = sps.levene(a, b, center="median")
    return dict(bf_W=float(r.statistic), bf_p=float(r.pvalue))


def pitman_morgan(a, b) -> dict:
    """Pitman-Morgan test for equal variances of PAIRED samples: corr(a+b, a-b) = 0 <=> var(a)=var(b).
    ``r > 0 => var(a) > var(b)`` (here a = K0, b = K5, so r > 0 => K=0 more dispersed). Returns
    ``pm_r, pm_p``."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if a.size < 4:
        return dict(pm_r=np.nan, pm_p=np.nan)
    s, d = a + b, a - b
    if s.std() == 0 or d.std() == 0:
        return dict(pm_r=np.nan, pm_p=np.nan)
    r = sps.pearsonr(s, d)
    return dict(pm_r=float(r[0]), pm_p=float(r[1]))


def read_md_table(path) -> pd.DataFrame:
    """Parse a pipe-delimited markdown table (as written by ``exports.save_table``) into a frame,
    numeric columns coerced."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    lines = [ln for ln in lines if not re.match(r"^\|\s*:?-{2,}", ln)]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    return df


def _judge_key(key: str) -> str:
    """Dict key -> the short judge label used in the ``judge`` column ('gpt-4o-mini', ...). Accepts a
    full ``judge=`` tag, a short label, or the paper's 'primary'/'heldout' aliases."""
    if key in ("primary", "", None):
        return judge_dirname("")
    if key == "heldout":
        return "claude-haiku-4-5"
    return judge_dirname(key)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Session shape (judge-invariant text metrics)
# ═════════════════════════════════════════════════════════════════════════════

def shape_text_metrics(arms) -> pd.DataFrame:
    """The per-conversation text-metrics frame behind every shape table: ``behavior.text_metrics``
    over ``arms`` (e.g. ``eda_analysis.cross_k_arms(cfg)``), restricted to :data:`FOUR_ARMS`, with
    ``persona_id`` attached and ``loop`` as float. Asserts 96 conversations and 96 distinct personas
    per (arm, iteration) — the pairing invariant. Reads the transcripts (slow-ish; parquet-cached
    upstream is NOT used here because ``text_metrics`` returns per-conversation rows)."""
    from .behavior import text_metrics
    tm = text_metrics(arms, attach_persona=True)
    tm = tm[tm["arm"].isin(FOUR_ARMS)].copy()
    tm["loop"] = tm["loop"].astype(float)
    n = tm.groupby(["arm", "iteration"]).size()
    if not (n == 96).all():
        raise AssertionError(f"expected 96 conversations per arm x iteration; got\n{n[n != 96]}")
    if not tm.groupby(["arm", "iteration"])["persona_id"].nunique().eq(96).all():
        raise AssertionError("persona recovery not 1:1 (some arm x iteration has < 96 distinct persona_id)")
    return tm


def _tm(tm_or_arms) -> pd.DataFrame:
    return tm_or_arms if isinstance(tm_or_arms, pd.DataFrame) else shape_text_metrics(tm_or_arms)


def session_shape_levels(tm_or_arms) -> pd.DataFrame:
    """Per (arm, method, K, iteration): ``<metric>_mean`` + ``<metric>_sem`` over the 96 personas for
    every :data:`SHAPE_METRICS` — the frame behind :func:`plotting.replication.shape_fig` and
    :func:`length_endpoints`. Takes the text-metrics frame (or an arms list, then builds it)."""
    tm = _tm(tm_or_arms)
    lvl = (tm.groupby(["arm", "method", "K", "iteration"])[SHAPE_METRICS]
           .agg(["mean", "sem"]).reset_index())
    lvl.columns = ["_".join(c).rstrip("_") for c in lvl.columns]
    return lvl


def session_shape_paired(tm_or_arms) -> pd.DataFrame:
    """Session shape, persona-paired K0 - K5 by matched iteration, within each method.

    One row per (method, metric, iteration): ``mean_K0`` / ``mean_K5`` (arm means over the same 96
    personas), ``n``, ``mean_delta`` (K0 - K5, + => K=0 higher), ``dz``, bootstrap 95% CI, Wilcoxon
    ``p``, ``p_holm`` (Holm WITHIN each (method, metric) family ACROSS iterations), ``metric_unit``.
    Includes iteration 0 (the two independent base draws — the noise floor). Reproduces
    ``session_shape_stability_shape``."""
    tm = _tm(tm_or_arms)
    rows = []
    for method in METHODS:
        a0, a5 = f"{method}_LA0", f"{method}_LA5"
        for metric in SHAPE_METRICS:
            fam = []
            for it in _common_iters(tm, a0, a5):
                x, y = _pair_wide(tm, metric, a0, a5, it)
                r = paired_arrays(x, y)
                fam.append({"method": method, "metric": metric, "iteration": it,
                            "mean_K0": float(np.nanmean(x)), "mean_K5": float(np.nanmean(y)),
                            "n": r["n"], "mean_delta": r["mean_delta"], "dz": r["dz"],
                            "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"]})
            f = pd.DataFrame(fam)
            f["p_holm"] = holm(f["p"].to_numpy())
            rows.append(f)
    shape = pd.concat(rows, ignore_index=True)
    shape["metric_unit"] = shape["metric"].map(SHAPE_UNITS)
    return shape


def length_endpoints(levels_or_tm, *, metrics: Sequence[str] = tuple(LENGTH_METRICS),
                     arms: Sequence[str] = tuple(FOUR_ARMS)) -> pd.DataFrame:
    """Base -> final session length per arm (arm means over 96 personas; base = the arm's own
    iteration-0 draw): ``<metric>_base``, ``<metric>_final``, ``<metric>_change`` (= final - base)
    for ``metrics`` (default conv_len, n_th_turns, mean_turn_len; pass ``SHAPE_METRICS`` for all
    five), plus ``final_iteration`` (10 for PTO_*, GRPO_LA0; 5 for the censored GRPO_LA5). Takes the
    :func:`session_shape_levels` frame (or the text-metrics frame). Reproduces
    ``session_shape_stability_length_endpoints``."""
    lvl = levels_or_tm
    if "conv_len_mean" not in getattr(lvl, "columns", []):
        lvl = session_shape_levels(lvl)
    rows = []
    for arm in arms:
        d = lvl[lvl["arm"] == arm].sort_values("iteration")
        if d.empty:
            continue
        it_last = int(d["iteration"].max())
        row = {"arm": arm, "final_iteration": it_last}
        for m in metrics:
            b = float(d.loc[d["iteration"] == 0, f"{m}_mean"].iloc[0])
            e = float(d.loc[d["iteration"] == it_last, f"{m}_mean"].iloc[0])
            row[f"{m}_base"] = b; row[f"{m}_final"] = e; row[f"{m}_change"] = e - b
        rows.append(row)
    return pd.DataFrame(rows)


def length_kcontrast(shape: pd.DataFrame, *, endpoints: Optional[Sequence[Tuple[str, int]]] = None,
                     metrics: Sequence[str] = tuple(LENGTH_METRICS)) -> pd.DataFrame:
    """The ICLR "K=5 gives shorter conversations" claim at the endpoints: the persona-paired K
    contrast on session length at each method's LAST MATCHED iteration (default: derived from
    ``shape`` — today PTO iter 10 and GRPO iter 5, where GRPO_LA5 is right-censored). Rows are
    pulled from :func:`session_shape_paired`; ``K5_minus_K0`` = the K=5 arm's mean minus the K=0
    arm's mean (positive = K=5 LONGER); ``mean_delta_K0_minus_K5`` keeps the package convention.
    Reproduces ``session_shape_stability_length_kcontrast``."""
    if endpoints is None:
        endpoints = [(m, int(shape.loc[shape["method"] == m, "iteration"].max())) for m in METHODS
                     if (shape["method"] == m).any()]
    rows = []
    for method, it in endpoints:
        for metric in metrics:
            sel = shape[(shape["method"] == method) & (shape["metric"] == metric) & (shape["iteration"] == it)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            rows.append({"contrast": f"{method} iter {it}", "method": method, "iteration": it, "metric": metric,
                         "mean_K0": r["mean_K0"], "mean_K5": r["mean_K5"], "K5_minus_K0": -r["mean_delta"],
                         "mean_delta_K0_minus_K5": r["mean_delta"], "dz": r["dz"], "ci_lo": r["ci_lo"],
                         "ci_hi": r["ci_hi"], "p": r["p"], "p_holm": r["p_holm"], "n": r["n"]})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 2. ICLR "stability": SD / IQR per arm x iteration under both graders + variance tests
# ═════════════════════════════════════════════════════════════════════════════

def _iter_judges(scores_by_judge: Dict[str, pd.DataFrame]):
    for key, sc in scores_by_judge.items():
        yield _judge_key(key), sc[sc["arm"].isin(FOUR_ARMS)]


def sd_by_iter(scores_by_judge: Dict[str, pd.DataFrame], *,
               metrics: Sequence[str] = tuple(STAB_METRICS)) -> pd.DataFrame:
    """Dispersion per arm x iteration under each grader (the ICLR 'K=5 is more stable / has the
    lowest SD' claim), graders side by side, never averaged. For each metric (default Q1, Q2, Q1Q2):
    ``n``, ``mean``, ``median``, ``sd`` (ddof=1), ``iqr`` (Q75 - Q25), and the ceiling shares
    ``share_ge4`` / ``share_ge45`` / ``share_eq5`` over the 96 personas of that model state. No
    pairing (within-arm descriptives). ``scores_by_judge`` = ``{judge_label_or_tag: scores_long}``
    (``eda_analysis.scores_by_judge(...)``); the ``judge`` column carries the short label. Sorted by
    (judge, metric, arm, iteration). Reproduces ``session_shape_stability_sd``."""
    rows = []
    for judge, sc in _iter_judges(scores_by_judge):
        for metric in metrics:
            d = sc[sc["questionnaire"] == metric]
            for (arm, it), s in d.groupby(["arm", "iteration"])["score"]:
                s = s.to_numpy(float)
                rows.append({"judge": judge, "metric": metric, "arm": arm, "method": method_of(arm),
                             "K": k_of(arm), "iteration": int(it), "n": int(s.size), "mean": float(s.mean()),
                             "median": float(np.median(s)), "sd": float(s.std(ddof=1)), "iqr": _iqr(s),
                             "share_ge4": float((s >= 4.0).mean()), "share_ge45": float((s >= 4.5).mean()),
                             "share_eq5": float((s == 5.0).mean())})
    return pd.DataFrame(rows).sort_values(["judge", "metric", "arm", "iteration"]).reset_index(drop=True)


def sd_tests(scores_by_judge: Dict[str, pd.DataFrame], *,
             metrics: Sequence[str] = tuple(STAB_METRICS)) -> pd.DataFrame:
    """Variance contrast K0 vs K5 per matched iteration, both graders: ``sd_K0`` / ``sd_K5``,
    ``sd_ratio_K5_over_K0`` (< 1 => the K=5 arm is LESS dispersed), ``iqr_K0`` / ``iqr_K5`` over the
    same 96 personas; ``bf_W`` / ``bf_p`` = Brown-Forsythe (independent groups, pairing not used);
    ``pm_r`` / ``pm_p`` = Pitman-Morgan paired variance test (r > 0 => K=0 more dispersed);
    ``bf_p_holm`` / ``pm_p_holm`` Holm-corrected within each (judge, method, metric) family across
    iterations. Rows follow the dict order of the graders (paper: primary first). Reproduces
    ``session_shape_stability_sd_bf``."""
    out = []
    for judge, sc in _iter_judges(scores_by_judge):
        for metric in metrics:
            d = sc[sc["questionnaire"] == metric]
            for method in METHODS:
                a0, a5 = f"{method}_LA0", f"{method}_LA5"
                fam = []
                for it in _common_iters(d, a0, a5):
                    x, y = _pair_wide(d, "score", a0, a5, it)
                    fam.append({"judge": judge, "method": method, "metric": metric, "iteration": it, "n": int(x.size),
                                "mean_K0": float(x.mean()), "mean_K5": float(y.mean()),
                                "sd_K0": float(x.std(ddof=1)), "sd_K5": float(y.std(ddof=1)),
                                "sd_ratio_K5_over_K0": float(y.std(ddof=1) / x.std(ddof=1)),
                                "iqr_K0": _iqr(x), "iqr_K5": _iqr(y), **brown_forsythe(x, y), **pitman_morgan(x, y)})
                if not fam:
                    continue
                f = pd.DataFrame(fam)
                f["bf_p_holm"] = holm(f["bf_p"].to_numpy())
                f["pm_p_holm"] = holm(f["pm_p"].to_numpy())
                out.append(f)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sd_tally(bf: pd.DataFrame) -> pd.DataFrame:
    """Tally of the K0-vs-K5 dispersion contrast over the trained matched iterations (1..N), per
    grader x method x rubric: ``n_K5_lower_sd`` / ``n_K5_lower_iqr`` = iterations at which the K=5
    arm's SD / IQR is smaller; ``median_sd_ratio_K5_over_K0`` (< 1 => K=5 typically less
    dispersed); ``n_pm_holm_sig_K5_lower`` / ``_K0_lower`` = Pitman-Morgan Holm-significant
    iterations with K=5 resp. K=0 less dispersed; ``n_bf_holm_sig`` = Brown-Forsythe Holm-significant
    (either direction); ``iter0_sd_K0`` / ``iter0_sd_K5`` = the two independent base draws (noise
    floor for an SD difference). Takes the :func:`sd_tests` frame. Reproduces
    ``session_shape_stability_sd_tally``."""
    tally = []
    for (judge, method, metric), d in bf.groupby(["judge", "method", "metric"], sort=False):
        dt = d[d["iteration"] > 0]
        i0 = d[d["iteration"] == 0]
        tally.append({"judge": judge, "method": method, "metric": metric, "n_iters": int(len(dt)),
                      "n_K5_lower_sd": int((dt["sd_K5"] < dt["sd_K0"]).sum()),
                      "n_K5_lower_iqr": int((dt["iqr_K5"] < dt["iqr_K0"]).sum()),
                      "median_sd_ratio_K5_over_K0": float(dt["sd_ratio_K5_over_K0"].median()),
                      "n_pm_holm_sig_K5_lower": int(((dt["pm_p_holm"] < 0.05) & (dt["pm_r"] > 0)).sum()),
                      "n_pm_holm_sig_K0_lower": int(((dt["pm_p_holm"] < 0.05) & (dt["pm_r"] < 0)).sum()),
                      "n_bf_holm_sig": int((dt["bf_p_holm"] < 0.05).sum()),
                      "iter0_sd_K0": float(i0["sd_K0"].iloc[0]) if len(i0) else np.nan,
                      "iter0_sd_K5": float(i0["sd_K5"].iloc[0]) if len(i0) else np.nan})
    return pd.DataFrame(tally)


def sd_summary(sd: pd.DataFrame, *, arms: Sequence[str] = tuple(FOUR_ARMS)) -> pd.DataFrame:
    """Where the lowest SD sits (trained iterations 1..N only), per grader and rubric: the model
    state with the smallest across-persona SD and its mean (``min_sd_*``), the state with the highest
    mean and its SD (``max_mean_*``), ``spearman_mean_vs_sd`` across all trained states (strongly
    negative = dispersion tracks the ceiling, not the optimizer), ``n_states``, and each arm's own
    lowest SD + where it occurs (``min_sd_<arm>`` / ``min_sd_iter_<arm>``). Takes the
    :func:`sd_by_iter` frame. Reproduces ``session_shape_stability_sd_summary``."""
    summ = []
    for (judge, metric), d in sd.groupby(["judge", "metric"]):
        dt = d[d["iteration"] > 0]
        imin = dt["sd"].idxmin()
        imax_mean = dt["mean"].idxmax()
        rho = sps.spearmanr(dt["mean"], dt["sd"])
        per_arm_min = dt.loc[dt.groupby("arm")["sd"].idxmin(), ["arm", "iteration", "sd", "mean"]]
        row = {"judge": judge, "metric": metric,
               "min_sd_arm": dt.loc[imin, "arm"], "min_sd_iteration": int(dt.loc[imin, "iteration"]),
               "min_sd": float(dt.loc[imin, "sd"]), "min_sd_mean": float(dt.loc[imin, "mean"]),
               "max_mean_arm": dt.loc[imax_mean, "arm"], "max_mean_iteration": int(dt.loc[imax_mean, "iteration"]),
               "max_mean": float(dt.loc[imax_mean, "mean"]), "max_mean_sd": float(dt.loc[imax_mean, "sd"]),
               "spearman_mean_vs_sd": float(rho.statistic), "spearman_p": float(rho.pvalue),
               "n_states": int(len(dt))}
        for a in arms:
            sel = per_arm_min[per_arm_min["arm"] == a]
            row[f"min_sd_{a}"] = float(sel["sd"].iloc[0]) if len(sel) else np.nan
        for a in arms:
            sel = per_arm_min[per_arm_min["arm"] == a]
            row[f"min_sd_iter_{a}"] = int(sel["iteration"].iloc[0]) if len(sel) else -1
        summ.append(row)
    return pd.DataFrame(summ)


def ceiling(scores_by_judge: Dict[str, pd.DataFrame], *, metric: str = "Q1Q2") -> pd.DataFrame:
    """Ceiling-compression check on Q1Q2 by patient cooperation level (persona trait: Resistant =
    cooperation_level Low, WarmsUp = StartLowAndChangesToHigh, Cooperative = High; 32 personas
    each), per arm x iteration, both graders: ``share_ge45_*`` = share of conversations scoring
    >= 4.5, ``share_eq5_*`` = exactly 5.0 (Q1Q2 == 5 requires every Q1 and Q2 item at 5); ``mean_*``
    / ``sd_*`` per subgroup (+ ``_all``). Requires the ``cooperation_level`` persona column (attached
    by ``attach_persona=True``). Reproduces ``session_shape_stability_ceiling``."""
    rows = []
    for judge, sc in _iter_judges(scores_by_judge):
        d = sc[sc["questionnaire"] == metric].copy()
        d["coop"] = d["cooperation_level"].map(COOP_LABEL).fillna(d["cooperation_level"])
        for (arm, it), s in d.groupby(["arm", "iteration"]):
            row = {"judge": judge, "arm": arm, "iteration": int(it), "n": int(len(s)),
                   "mean_all": float(s["score"].mean()), "sd_all": float(s["score"].std(ddof=1)),
                   "share_ge45_all": float((s["score"] >= 4.5).mean()), "share_eq5_all": float((s["score"] == 5).mean())}
            for cl in COOP_ORDER:
                v = s.loc[s["coop"] == cl, "score"].to_numpy(float)
                row[f"n_{cl}"] = int(v.size)
                row[f"mean_{cl}"] = float(v.mean()) if v.size else np.nan
                row[f"sd_{cl}"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
                row[f"share_ge45_{cl}"] = float((v >= 4.5).mean()) if v.size else np.nan
                row[f"share_eq5_{cl}"] = float((v == 5).mean()) if v.size else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["judge", "arm", "iteration"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Selection-level length push + praise weights (tracked preference tables; NO recompute)
# ═════════════════════════════════════════════════════════════════════════════

def _results_dir() -> str:
    from .exports import RESULTS_DIR
    return RESULTS_DIR


def default_selection_dirs() -> List[str]:
    """Where the tracked ``update_lexical_push.md`` + ``generation_pool_means.md`` live: the single
    ``results/arms/preference/tables/gpt-4o-mini/`` folder written by ``arms/preference.ipynb``
    (all four arms in one table; the training-side rows exist only under the primary oracle).

    Before the 2026-08-18 reorg these were two per-K files under ``results/{L0,L5}/tables/6_preference/``
    (git keeps that tree at ``b09eb6f``); the notebook that calls this falls back to any directory
    list you pass explicitly, so a re-render of ``arms/preference`` is the only prerequisite.
    """
    R = _results_dir()
    return [os.path.join(R, "arms", "preference", "tables", "gpt-4o-mini")]


_SEL_COLS = ["arm", "method", "K", "train_iter", "policy_iteration", "n_groups", "n_candidates",
             "w_len", "w_len_se", "w_len_over_se", "w_overpraise", "w_overpraise_se", "w_affirm", "w_affirm_se",
             "w_question", "w_question_se", "pool_len", "pool_len_se", "pool_overpraise", "pool_affirm",
             "pool_question"]


def selection_table(source_dir: Union[None, str, Sequence[str]] = None) -> pd.DataFrame:
    """Selection-level lexical push and generation-pool means per arm x training iteration —
    COPIED, not recomputed, from the tracked preference tables ``update_lexical_push.md`` and
    ``generation_pool_means.md`` found in ``source_dir`` (one directory or several; default
    :func:`default_selection_dirs` — the old ``results/{L0,L5}/tables/6_preference/gpt-4o-mini/``
    pair; TODO flip the default to ``results/arms/preference/tables/gpt-4o-mini/`` after Phase C).

    ``w_<feature>`` = the lexical contrast the update pushes for, Sum(w * feature) per group +/- SE
    over groups, on a shared scale for both methods (DPO's +/-1 pair; GRPO's standardized advantages
    rescaled to match); 0 = indifferent; ``w_len`` in characters, ``w_question`` in '?' per
    completion, ``w_affirm`` / ``w_overpraise`` in marker-rate units. ``pool_<feature>`` = the mean
    over ALL candidates the policy generated (what it GENERATES vs what the update SELECTS for).
    ``w_len_over_se`` = w_len / SE (a z-like ratio); ``policy_iteration = train_iter - 1``. Primary
    training oracle by construction; no persona pairing (group-level). Duplicate (arm, train_iter)
    rows across directories are dropped (first wins). Reproduces
    ``session_shape_stability_selection``."""
    if source_dir is None:
        dirs = default_selection_dirs()
    elif isinstance(source_dir, (str, os.PathLike)):
        dirs = [str(source_dir)]
    else:
        dirs = [str(d) for d in source_dir]
    push, pool = [], []
    for d in dirs:
        p1, p2 = os.path.join(d, "update_lexical_push.md"), os.path.join(d, "generation_pool_means.md")
        if os.path.exists(p1):
            push.append(read_md_table(p1))
        if os.path.exists(p2):
            pool.append(read_md_table(p2))
    if not push or not pool:
        raise FileNotFoundError(f"update_lexical_push.md / generation_pool_means.md not found under {dirs}")
    push = pd.concat(push, ignore_index=True).drop_duplicates(["arm", "train_iter"])
    pool = pd.concat(pool, ignore_index=True).drop_duplicates(["arm", "train_iter"])
    sel = push.merge(pool[["arm", "train_iter", "n_candidates", "pool_len", "pool_len_se", "pool_overpraise",
                           "pool_overpraise_se", "pool_affirm", "pool_affirm_se", "pool_question"]],
                     on=["arm", "train_iter"], how="left", suffixes=("", "_pool"))
    sel["w_len_over_se"] = sel["w_len"] / sel["w_len_se"]
    sel["policy_iteration"] = sel["train_iter"] - 1
    if "n_candidates_pool" in sel.columns:
        sel = sel.drop(columns=["n_candidates"], errors="ignore").rename(columns={"n_candidates_pool": "n_candidates"})
    sel = sel[_SEL_COLS]
    return sel.sort_values(["method", "K", "train_iter"], ascending=[False, True, True]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 4. The numbers ledger
# ═════════════════════════════════════════════════════════════════════════════

def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _entry(value, source: str = "", note: str = "") -> dict:
    return {"value": _clean(value), "source": source, "note": note}


def replication_numbers(*, shape: Optional[pd.DataFrame] = None, levels: Optional[pd.DataFrame] = None,
                        endpoints: Optional[pd.DataFrame] = None, kcontrast: Optional[pd.DataFrame] = None,
                        sd: Optional[pd.DataFrame] = None, bf: Optional[pd.DataFrame] = None,
                        tally: Optional[pd.DataFrame] = None, summary: Optional[pd.DataFrame] = None,
                        ceil: Optional[pd.DataFrame] = None, selection: Optional[pd.DataFrame] = None,
                        table_prefix: str = "tables/") -> dict:
    """Every quotable number of the family as ``{dotted.key: {value, source, note}}`` (the shape
    ``exports.save_numbers`` writes). Pass whichever frames were computed; absent ones are skipped.
    Keys mirror the paper ledger ``analysis/out/session_shape_stability.json``:
    ``shape.<method>.<metric>.iter<n>``, ``levels.<arm>``, ``length_endpoint.<method>.iter<n>.<metric>``,
    ``sd.<judge>.<metric>.<arm>.iter<n>``, ``sd_bf.<judge>.<method>.<metric>.iter<n>``,
    ``sd_tally.<judge>.<method>.<metric>``, ``sd_summary.<judge>.<metric>``,
    ``ceiling.<judge>.<arm>.iter<n>``, ``selection.<arm>.train_iter<n>`` + the selection endpoints.
    ``levels`` is the :func:`length_endpoints` frame computed with ``metrics=SHAPE_METRICS`` (all
    five) — the paper ledgered every shape metric there; the three-metric table is fine too. The
    paper's two ``_crosscheck.*`` keys (the PTO Q1Q2 iter-6 anchor, the ``k_paired_channels`` sheet
    agreement) are NOT produced here — the anchor lives in ``_selfcheck`` (paper-fixture check) and
    the sheet check belongs to the notebook that renders both tables."""
    N = {}
    T = table_prefix
    if shape is not None:
        for _, r in shape.iterrows():
            N[f"shape.{r.method}.{r.metric}.iter{int(r.iteration)}"] = _entry(
                {k: r[k] for k in ["n", "mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]},
                source=f"{T}shape.md row method={r.method} metric={r.metric} iteration={int(r.iteration)}")
    if levels is not None:
        for _, r in levels.iterrows():
            N[f"levels.{r['arm']}"] = _entry({k: r[k] for k in levels.columns},
                                             source=f"{T}length_endpoints.md row arm={r['arm']}")
    if kcontrast is not None:
        for _, r in kcontrast.iterrows():
            N[f"length_endpoint.{r['method']}.iter{int(r['iteration'])}.{r['metric']}"] = _entry(
                {k: r[k] for k in kcontrast.columns},
                source=f"{T}length_kcontrast.md row contrast='{r['contrast']}' metric={r['metric']}")
    if bf is not None:
        for _, r in bf.iterrows():
            N[f"sd_bf.{r.judge}.{r.method}.{r.metric}.iter{int(r.iteration)}"] = _entry(
                {k: r[k] for k in ["n", "sd_K0", "sd_K5", "sd_ratio_K5_over_K0", "iqr_K0", "iqr_K5", "bf_W", "bf_p",
                                   "bf_p_holm", "pm_r", "pm_p", "pm_p_holm", "mean_K0", "mean_K5"]},
                source=f"{T}sd_bf.md row judge={r.judge} method={r.method} metric={r.metric} iteration={int(r.iteration)}")
    if sd is not None:
        for _, r in sd.iterrows():
            N[f"sd.{r.judge}.{r.metric}.{r.arm}.iter{int(r.iteration)}"] = _entry(
                {k: r[k] for k in ["n", "mean", "median", "sd", "iqr", "share_ge4", "share_ge45", "share_eq5"]},
                source=f"{T}sd.md row judge={r.judge} metric={r.metric} arm={r.arm} iteration={int(r.iteration)}")
    if ceil is not None:
        for _, r in ceil.iterrows():
            N[f"ceiling.{r.judge}.{r.arm}.iter{int(r.iteration)}"] = _entry(
                {k: r[k] for k in ceil.columns if k not in ("judge", "arm", "iteration")},
                source=f"{T}ceiling.md row judge={r.judge} arm={r.arm} iteration={int(r.iteration)}")
    if tally is not None:
        for _, r in tally.iterrows():
            N[f"sd_tally.{r.judge}.{r.method}.{r.metric}"] = _entry(
                {k: r[k] for k in tally.columns if k not in ("judge", "method", "metric")},
                source=f"{T}sd_tally.md row judge={r.judge} method={r.method} metric={r.metric}")
    if summary is not None:
        for _, r in summary.iterrows():
            N[f"sd_summary.{r.judge}.{r.metric}"] = _entry(
                {k: r[k] for k in summary.columns if k not in ("judge", "metric")},
                source=f"{T}sd_summary.md row judge={r.judge} metric={r.metric}")
    if selection is not None:
        SEL = selection
        by = {a: SEL[SEL["arm"] == a].sort_values("train_iter") for a in FOUR_ARMS}
        pto5, pto0, grpo0, grpo5 = by["PTO_LA5"], by["PTO_LA0"], by["GRPO_LA0"], by["GRPO_LA5"]
        if len(pto5):
            N["selection.PTO_LA5.w_len_series"] = _entry(
                {"train_iter": pto5["train_iter"].tolist(), "w_len": pto5["w_len"].tolist(),
                 "w_len_se": pto5["w_len_se"].tolist(),
                 "n_positive": int((pto5["w_len"] > 0).sum()), "n_iters": int(len(pto5)),
                 "min_w_len_over_se": float(pto5["w_len_over_se"].min())},
                source=f"{T}selection.md rows arm=PTO_LA5 (from update_lexical_push.md)")
        if len(pto0):
            i = pto0["w_len_over_se"].abs().idxmax()
            N["selection.PTO_LA0.max_abs_w_len_over_se"] = _entry(
                {"max_abs_w_len_over_se": float(pto0["w_len_over_se"].abs().max()),
                 "at_train_iter": int(pto0.loc[i, "train_iter"]), "w_len_at": float(pto0.loc[i, "w_len"]),
                 "w_len_series": pto0["w_len"].tolist(), "w_len_se_series": pto0["w_len_se"].tolist()},
                source=f"{T}selection.md rows arm=PTO_LA0 (from update_lexical_push.md)")
        for arm, d in by.items():
            if not len(d):
                continue
            N[f"selection.{arm}.pool_len_endpoints"] = _entry(
                {"train_iter_first": int(d["train_iter"].iloc[0]), "pool_len_first": float(d["pool_len"].iloc[0]),
                 "train_iter_last": int(d["train_iter"].iloc[-1]), "pool_len_last": float(d["pool_len"].iloc[-1]),
                 "pool_overpraise_last": float(d["pool_overpraise"].iloc[-1]),
                 "pool_affirm_last": float(d["pool_affirm"].iloc[-1])},
                source=f"{T}selection.md rows arm={arm} (from generation_pool_means.md)")
            for _, r in d.iterrows():
                N[f"selection.{arm}.train_iter{int(r.train_iter)}"] = _entry(
                    {k: r[k] for k in ["w_len", "w_len_se", "w_overpraise", "w_overpraise_se", "w_affirm", "w_affirm_se",
                                       "w_question", "w_question_se", "pool_len", "pool_overpraise", "pool_affirm",
                                       "n_groups"]},
                    source=f"{T}selection.md row arm={arm} train_iter={int(r.train_iter)}")
        if len(pto5) and len(pto0) and (grpo5["train_iter"] == 5).any() and (grpo0["train_iter"] == 5).any():
            N["selection.pool_len_contrast"] = _entry(
                {"PTO_iter10_LA5_vs_LA0": [float(pto5["pool_len"].iloc[-1]), float(pto0["pool_len"].iloc[-1])],
                 "GRPO_iter5_LA5_vs_LA0": [float(grpo5.loc[grpo5["train_iter"] == 5, "pool_len"].iloc[0]),
                                           float(grpo0.loc[grpo0["train_iter"] == 5, "pool_len"].iloc[0])]},
                source=f"{T}selection.md",
                note="pool_len at PTO train_iter 10 (K5 vs K0) and GRPO train_iter 5 (K5 vs K0)")
    return N


# ── captions (verbatim from the paper generator; the notebook passes them to save_table) ──
CAPTIONS = {
    "shape": (
        "**Session shape, persona-paired K0 - K5 by matched iteration.** Deterministic text metrics computed "
        "from the eval transcripts (`eda_analysis.behavior.text_metrics`; judge-invariant): conv_len = utterances "
        "per conversation (therapist + patient), n_th_turns = therapist turns, mean_turn_len = characters per "
        "therapist turn, q_per_turn = literal '?' per therapist turn, loop = share of conversations with a "
        f"verbatim-repeated therapist turn (degeneracy). {SIGN} {PAIR} mean_K0/mean_K5 are the arm means over "
        "the same 96 personas; mean_delta/dz/bootstrap 95% CI/Wilcoxon p are on the paired deltas; p_holm is "
        "Holm-corrected WITHIN each (method, metric) family ACROSS iterations (the tracked "
        "k_paired_channels corrects across channels within an iteration instead — same delta/dz/p, different "
        f"p_holm scope). {ITER0} {CENSOR} Length metrics are unvalenced (longer is not better)."),
    "sd": (
        "**Dispersion per arm x iteration (the ICLR 'K=5 is more stable / has the lowest SD' claim), both graders "
        "side by side (never averaged).** For Q1, Q2 and Q1Q2 (= mean of the Q1 and Q2 means, the training "
        "reward): n conversations, mean, median, SD (ddof=1), IQR (Q75 - Q25), and the ceiling shares "
        "(score >= 4, >= 4.5, == 5) over the 96 personas of that model state. No pairing here (within-arm "
        f"descriptives). Grader named in `judge`. {ITER0} {CENSOR} Read SD next to the mean: on a bounded 1-5 "
        "scale a higher mean mechanically compresses SD (see the ceiling table)."),
    "sd_bf": (
        "**Variance contrast K0 vs K5 per matched iteration, both graders.** sd_K0/sd_K5 and iqr_K0/iqr_K5 over "
        "the same 96 personas; sd_ratio_K5_over_K0 < 1 => the K=5 arm is LESS dispersed. bf_W/bf_p = "
        "Brown-Forsythe (Levene on |x - median|; treats the two arms as independent groups, persona pairing not "
        "used). pm_r/pm_p = Pitman-Morgan paired variance test = Pearson r between (K0 + K5) and (K0 - K5) over "
        f"personas; r > 0 => K=0 more dispersed (sign matches {SIGN[:-1]} applied to variance). {PAIR} p_holm "
        f"columns are Holm-corrected within each (judge, method, metric) family across iterations. {ITER0} {CENSOR}"),
    "ceiling": (
        "**Ceiling-compression check on Q1Q2 by patient cooperation level (persona trait: Resistant = "
        "cooperation_level Low, WarmsUp = StartLowAndChangesToHigh, Cooperative = High; 32 personas each), per "
        "arm x iteration, both graders.** share_ge45 = share of conversations scoring >= 4.5, share_eq5 = exactly "
        "5.0 (Q1Q2 == 5 requires every Q1 and Q2 item at 5); mean/sd per subgroup. A low arm SD that comes with a "
        "high ceiling share is scale compression, not stability. The held-out judge (Claude Haiku 4.5) never "
        f"awards >= 4.5 on Q1Q2 (its max is 4.25), so its ceiling shares are 0 by construction. {ITER0} {CENSOR}"),
    "sd_tally": (
        "**Tally of the K0-vs-K5 dispersion contrast over the trained matched iterations (1..N), per grader x "
        "method x rubric.** n_K5_lower_sd / n_K5_lower_iqr = iterations at which the K=5 arm's SD / IQR is smaller "
        "than K=0's; median_sd_ratio = median of sd_K5 / sd_K0 (< 1 => K=5 typically less dispersed); "
        "n_pm_holm_sig_K5_lower / _K0_lower = iterations where the persona-paired Pitman-Morgan test is "
        "Holm-significant (within judge x method x rubric across iterations) with K=5 resp. K=0 less dispersed; "
        "n_bf_holm_sig = Brown-Forsythe Holm-significant iterations (either direction). iter0_sd_* = the two "
        f"independent base draws (noise floor for an SD difference). {CENSOR} PTO: N=10 iterations; GRPO: N=5."),
    "sd_summary": (
        "**Where the lowest SD sits (trained iterations 1..N only; iteration 0 excluded), per grader and rubric.** "
        "min_sd_* = the model state with the smallest across-persona SD and its mean; max_mean_* = the state with "
        "the highest mean and its SD; spearman_mean_vs_sd = rank correlation between arm-iteration mean and SD "
        "across all trained states (n_states) — strongly negative = dispersion tracks the ceiling, not the "
        f"optimizer; min_sd_<arm> / min_sd_iter_<arm> = each arm's own lowest SD and where it occurs. {CENSOR}"),
    "length_endpoints": (
        "**Base -> final session length per arm (arm means over 96 personas; base = the arm's own iteration-0 "
        "draw).** conv_len in utterances, n_th_turns in therapist turns, mean_turn_len in characters per therapist "
        f"turn; change = final - base. {CENSOR} The K contrast at the endpoints (PTO iter 10, GRPO iter 5) is in "
        "the length_kcontrast table."),
    "length_kcontrast": (
        "**The ICLR 'K=5 gives shorter conversations' claim at the endpoints: persona-paired K contrast on session "
        "length at PTO iteration 10 and GRPO iteration 5 (the last matched GRPO iteration; GRPO_LA5 is "
        "right-censored there).** K5_minus_K0 is the K=5 arm's mean minus the K=0 arm's mean (positive = K=5 "
        f"LONGER); mean_delta_K0_minus_K5 keeps the paper's convention ({SIGN}). dz / bootstrap 95% CI / Wilcoxon "
        f"p on the paired deltas; p_holm within (method, metric) across iterations. {PAIR} Judge-free."),
    "selection": (
        "**Selection-level lexical push and generation-pool means per arm x training iteration (copied, not "
        "recomputed, from the tracked preference tables `update_lexical_push.md` and `generation_pool_means.md`).** "
        "w_<feature> = the lexical contrast the update pushes for, Sum(w * feature) per group +/- SE "
        "over groups, on a shared scale for both methods (DPO's +/-1 pair; GRPO's standardized advantages rescaled "
        "to match); 0 = the update is indifferent to that feature; w_len in characters, w_question in '?' per "
        "completion, w_affirm / w_overpraise in marker-rate units. pool_<feature> = the mean of that feature over "
        "ALL candidates the policy generated (what it GENERATES vs what the update SELECTS for). train_iter n "
        "samples from the iter-start policy, i.e. the eval set's model_iter_{n-1} (= policy_iteration). "
        "w_len_over_se = w_len / SE (a z-like ratio). Primary training oracle by construction (generations.jsonl "
        f"records the training oracle's own selection); no persona pairing (group-level). {CENSOR}"),
    "fig_shape": "Session shape by iteration — deterministic text metrics (mean ± SE over 96 personas; grader-free); "
                 "K=0 solid filled, K=5 dashed open; PTO cool / GRPO warm.",
    "fig_sd": "Across-persona SD of the training reward (Q1Q2) by iteration under each grader (two panels) plus SD vs "
              "mean over the trained states (filled = gpt-4o-mini, open = claude-haiku-4-5) — the ICLR 'K=5 is more "
              "stable' claim.",
}
