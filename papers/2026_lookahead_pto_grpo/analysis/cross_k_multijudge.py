"""cross_k_multijudge.py — does the look-ahead (K) contrast itself survive the held-out judge?

The tracked EDA (``8_Measurement_Validity``) judge-tests contrasts WITHIN a K view only (its
``all_pairs_contrasts`` enumerates the 17 K=5 model states, or the 22 K=0 ones — never a K=0 vs
K=5 pair), and its gain-retention table uses one shared PTO base per view. This generator asks the
cross-K questions the paper needs:

1. **Cross-K contrasts under both graders** — for each method, ``LA0_In − LA5_In`` at every matched
   iteration (iteration 0 = the two independent base draws, a free noise-floor row), on all 9
   rubrics: primary and held-out delta / dz / bootstrap CI / Wilcoxon p / Holm p, ``same_sign``,
   and a sign-preservation ladder by |Δ primary| rung (mirrors ``reliability.sign_preservation``).
2. **Gain retention by K** — ``retention = Δ held-out / Δ primary`` vs each arm's OWN base, and
   again vs a SHARED reference (the method's LA0 base, its LA5 base, and — for the GRPO arms — the
   tracked EDA's PTO_LA{K}_Base) so the base-reference caveat is visible.
3. **K × method interaction** — persona-level difference-in-differences
   ``(PTO_LA0 − GRPO_LA0) − (PTO_LA5 − GRPO_LA5)`` at matched iterations 0..5 (GRPO_LA5 stops at 5,
   so 6..10 is not estimable), plus the simple PTO − GRPO gap at each K per iteration.
4. **Held-out endpoint contrasts** the paper quotes (PTO_LA5_I10 vs GRPO_LA5_I5, etc.).

Sign conventions: cross-K contrast ``+ => K=0 higher``; method gap ``+ => PTO higher``;
DiD ``+ => PTO's lead is larger at K=0 than at K=5``. MICI is lower-is-better, so on MICI a positive
K-contrast favours K=5 — every table carries an explicit ``favours`` column. Everything is paired on
``persona_id`` (never ``file_index``); the two graders' raw scores are never averaged.

Run::  .venv/Scripts/python.exe papers/2026_lookahead_pto_grpo/analysis/cross_k_multijudge.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _common as C  # noqa: E702

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eda_analysis import reliability as R

NAME = "cross_k_multijudge"
L = C.Ledger(NAME)
RUBRICS = list(C.RUBRICS)                       # Q1Q2, Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI, PCT, MICI
LAST_ITER = {"PTO_LA0": 10, "PTO_LA5": 10, "GRPO_LA0": 10, "GRPO_LA5": 5}
JUDGE_COL = {"primary": "primary", "heldout": "judge"}   # column prefixes mirror reliability.all_pairs_contrasts
JUDGE_NAME = {"primary": C.JUDGE_SHORT[C.PRIMARY], "heldout": C.JUDGE_SHORT[C.HELDOUT]}
CENSOR = "GRPO_LA5 is right-censored at iteration 5 (its full budget); PTO arms and GRPO_LA0 run to 10."
PAIRING = "Paired on persona_id (the trainer reshuffles the 96 personas every iteration; file_index is not a pairing key)."


def model_name(arm: str, it: int) -> str:
    return f"{C.method_of(arm)}Exp3_LA{C.k_of(arm)}_{'Base' if it == 0 else f'I{it}'}"


def favours(metric: str, delta: float, plus_label: str, minus_label: str) -> str:
    """Readable direction label; flips for lower-is-better metrics (MICI)."""
    if delta is None or (isinstance(delta, float) and np.isnan(delta)) or delta == 0:
        return ""
    hi = plus_label if delta > 0 else minus_label
    if metric in C.LOWER_IS_BETTER:
        hi = minus_label if delta > 0 else plus_label
    return hi


# ═══════════════════════════════════════════════════════════════════════════════
# 0. load once, build persona × model matrices per grader + reliability-shaped long frames
# ═══════════════════════════════════════════════════════════════════════════════
S = C.load_scores_both()
S = {j: df[df["questionnaire"].isin(RUBRICS)].copy() for j, df in S.items()}
W = {j: {m: C.wide(df, m) for m in RUBRICS} for j, df in S.items()}
MODELS = sorted(S["primary"]["model"].unique())
assert len(MODELS) == 39, MODELS
for j in S:
    n_cell = S[j].groupby(["model", "questionnaire"]).size()
    assert (n_cell == 96).all(), f"{j}: incomplete cells\n{n_cell[n_cell != 96]}"


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """scores_long → the (metric, model, file_index, value) shape reliability.* expects."""
    return (df.rename(columns={"questionnaire": "metric", "score": "value"})
              [["metric", "model", "file_index", "value"]].reset_index(drop=True))


JL, PL = to_long(S["heldout"]), to_long(S["primary"])   # judge_long / primary_long


def contrast(judge: str, metric: str, a: str, b: str) -> dict:
    """a − b under one grader, persona-paired (C.paired: mean, dz, bootstrap CI, Wilcoxon p, n)."""
    w = W[judge][metric]
    return C.paired(w[a].to_numpy(), w[b].to_numpy())


def both_graders(metric: str, a: str, b: str) -> dict:
    """One row with primary_* and judge_* columns (names mirror reliability.all_pairs_contrasts)."""
    rec = {"metric": metric, "model_a": a, "model_b": b}
    for jkey, pref in JUDGE_COL.items():
        r = contrast(jkey, metric, a, b)
        rec.update({f"{pref}_n": r["n"], f"{pref}_delta": r["mean_delta"], f"{pref}_dz": r["dz"],
                    f"{pref}_ci_lo": r["ci_lo"], f"{pref}_ci_hi": r["ci_hi"], f"{pref}_p": r["p"]})
    rec["same_sign"] = bool(np.sign(rec["judge_delta"]) == np.sign(rec["primary_delta"]))
    rec["judge_ci_excl0"] = bool(rec["judge_ci_lo"] > 0 or rec["judge_ci_hi"] < 0)
    rec["primary_ci_excl0"] = bool(rec["primary_ci_lo"] > 0 or rec["primary_ci_hi"] < 0)
    return rec


def holm_within(df: pd.DataFrame, by: list, pcol: str, out: str) -> pd.DataFrame:
    df = df.copy()
    df[out] = np.nan
    for _, g in df.groupby(by, sort=False):
        df.loc[g.index, out] = C.holm(g[pcol].to_numpy())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. cross-K all-pairs contrasts (LA0_In − LA5_In) under both graders + sign-preservation ladder
# ═══════════════════════════════════════════════════════════════════════════════
rows = []
for method in ("PTO", "GRPO"):
    a_arm, b_arm = f"{method}_LA0", f"{method}_LA5"
    for it in range(0, min(LAST_ITER[a_arm], LAST_ITER[b_arm]) + 1):
        for m in RUBRICS:
            rec = both_graders(m, model_name(a_arm, it), model_name(b_arm, it))
            rec.update(method=method, iteration=it,
                       contrast=f"{method}_LA0_{'Base' if it == 0 else f'I{it}'} − {method}_LA5_{'Base' if it == 0 else f'I{it}'}")
            rows.append(rec)
pairs = pd.DataFrame(rows)
# Holm across ITERATIONS within (grader, method, metric)
pairs = holm_within(pairs, ["method", "metric"], "primary_p", "primary_p_holm")
pairs = holm_within(pairs, ["method", "metric"], "judge_p", "judge_p_holm")
pairs["favours_primary"] = [favours(m, d, "K0", "K5") for m, d in zip(pairs.metric, pairs.primary_delta)]
pairs["favours_judge"] = [favours(m, d, "K0", "K5") for m, d in zip(pairs.metric, pairs.judge_delta)]
pairs_cols = ["method", "iteration", "metric", "contrast", "primary_n",
              "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi", "primary_p", "primary_p_holm",
              "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p", "judge_p_holm",
              "same_sign", "judge_ci_excl0", "favours_primary", "favours_judge"]
pairs = pairs[pairs_cols].sort_values(["method", "metric", "iteration"]).reset_index(drop=True)
pairs["metric"] = pd.Categorical(pairs["metric"], RUBRICS, ordered=True)
pairs = pairs.sort_values(["method", "metric", "iteration"]).reset_index(drop=True)
pairs["metric"] = pairs["metric"].astype(str)

# ── cross-check against the tracked EDA (7_stats/<judge>/k_paired_by_method.md) and reliability.all_pairs_contrasts
chk = pairs[(pairs.method == "PTO") & (pairs.metric == "Q1Q2") & (pairs.iteration == 6)].iloc[0]
print(f"[check] PTO Q1Q2 iter 6 primary K0−K5: delta {chk.primary_delta:.3f} dz {chk.primary_dz:.3f} "
      f"(tracked k_paired_by_method.md: +0.257, dz 0.417)")
assert abs(chk.primary_delta - 0.257) < 0.002 and abs(chk.primary_dz - 0.417) < 0.002
apc = R.all_pairs_contrasts(JL, PL, metrics=["Q1Q2"], models=["PTOExp3_LA0_I6", "PTOExp3_LA5_I6"], n_boot=200)
print(f"[check] reliability.all_pairs_contrasts same pair: primary {apc.primary_delta.iloc[0]:.3f} dz "
      f"{apc.primary_dz.iloc[0]:.3f}; judge {apc.judge_delta.iloc[0]:.3f} dz {apc.judge_dz.iloc[0]:.3f} "
      f"(mine: judge {chk.judge_delta:.3f} dz {chk.judge_dz:.3f})")
assert abs(apc.judge_delta.iloc[0] - chk.judge_delta) < 0.002 and abs(apc.judge_dz.iloc[0] - chk.judge_dz) < 0.002

C.save_table(pairs, f"{NAME}_pairs", caption=(
    "**Cross-K contrasts under both graders.** For each method, `LA0_In − LA5_In` at every matched iteration "
    "(iteration 0 = the two arms' INDEPENDENT base draws, a noise-floor row) on all 9 rubrics. Sign: "
    "**+ => K=0 higher**; MICI is lower-is-better, so on MICI + favours K=5 — read the `favours_*` columns. "
    f"{PAIRING} `primary_*` = training oracle (gpt-4o-mini); `judge_*` = held-out judge (Claude Haiku 4.5). "
    "dz = mean/SD of persona deltas; CI = 2,000-draw percentile bootstrap over personas; p = Wilcoxon; "
    "`*_p_holm` = Holm across ITERATIONS within (grader, method, metric). `same_sign` = the two graders "
    f"agree on direction; `judge_ci_excl0` = the held-out CI excludes 0. {CENSOR} "
    "Cross-check: PTO Q1Q2 iteration 6 primary = +0.257, dz 0.417 (tracked 7_stats/k_paired_by_method.md)."))

# ── sign-preservation ladder (mirrors reliability.sign_preservation, + Holm-significance rungs)
def extra_rungs(df: pd.DataFrame) -> pd.DataFrame:
    subs = [("primary p_holm < 0.05", df[df.primary_p_holm < 0.05]),
            ("judge p_holm < 0.05", df[df.judge_p_holm < 0.05]),
            ("both graders p_holm < 0.05", df[(df.primary_p_holm < 0.05) & (df.judge_p_holm < 0.05)]),
            ("iteration >= 1 (base row excluded)", df[df.iteration >= 1])]
    out = []
    for label, sub in subs:
        n = len(sub)
        out.append({"subset": label, "n_contrasts": n,
                    "n_same_sign": int(sub.same_sign.sum()) if n else 0,
                    "pct_same_sign": round(100 * sub.same_sign.mean(), 1) if n else np.nan})
    return pd.DataFrame(out)


groups = [("all cross-K contrasts", pairs)] + [(f"method={m}", pairs[pairs.method == m]) for m in ("PTO", "GRPO")]          + [(f"metric={m}", pairs[pairs.metric == m]) for m in RUBRICS]
ladder = pd.concat([pd.concat([R.sign_preservation(sub), extra_rungs(sub)], ignore_index=True).assign(group=label)
                    for label, sub in groups], ignore_index=True)
ladder = ladder[["group", "subset", "n_contrasts", "n_same_sign", "pct_same_sign"]]
C.save_table(ladder, f"{NAME}_ladder", nd=1, caption=(
    "**Sign preservation of the cross-K contrast under the held-out judge**, as a ladder over the gap the "
    "primary oracle reports (mirrors `reliability.sign_preservation`; thresholds are ABSOLUTE, so per-metric "
    "rows compare down a rubric, never across — PCT/MICI live on 0–1). Rows are the `LA0_In − LA5_In` "
    "contrasts of `cross_k_multijudge_pairs` (17 iteration pairs × 9 rubrics = 153; PTO 11×9, GRPO 6×9). "
    "Extra rungs restrict to contrasts each grader calls Holm-significant, and to iteration ≥ 1 (dropping the "
    f"base-vs-base noise-floor rows). {PAIRING} {CENSOR}"))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. gain retention by K — own base vs shared references
# ═══════════════════════════════════════════════════════════════════════════════
def _sub(long: pd.DataFrame, models) -> pd.DataFrame:
    return long[long.model.isin(set(models))]


RATE_METRICS = ["PCT", "MICI"]
SCALE_METRICS = [m for m in RUBRICS if m not in RATE_METRICS]
ret_rows = []
for arm in C.ARMS:
    method, K = C.method_of(arm), C.k_of(arm)
    arm_models = [model_name(arm, i) for i in range(0, LAST_ITER[arm] + 1)]
    refs = [("own_base", model_name(arm, 0)),
            (f"method_LA0_base", model_name(f"{method}_LA0", 0)),
            (f"method_LA5_base", model_name(f"{method}_LA5", 0))]
    if method == "GRPO":
        refs.append((f"eda_view_PTO_LA{K}_base", model_name(f"PTO_LA{K}", 0)))
    for kind, ref in refs:
        models = sorted(set(arm_models) | {ref})
        # |Δ primary| floor below which the ratio is suppressed: 0.15 on the 1–5 / 1–7 rubrics (the
        # eda_analysis default) and 0.05 on the 0–1 RATE metrics (PCT, MICI), whose deltas are ~3× smaller.
        parts = []
        for mets, floor in ((SCALE_METRICS, 0.15), (RATE_METRICS, 0.05)):
            rt = R.gain_retention(_sub(JL, models), _sub(PL, models), reference_model=ref, metrics=mets,
                                  min_primary_delta=floor)
            rt["min_primary_delta"] = floor
            parts.append(rt)
        rt = pd.concat(parts, ignore_index=True)
        rt = rt[rt.model.isin(arm_models)].copy()
        rt["arm"], rt["K"], rt["method"] = arm, K, method
        rt["iteration"] = rt["model"].map(R.model_iteration)
        rt["ref_kind"] = kind
        rt["ref_is_own_base"] = (ref == model_name(arm, 0))
        ret_rows.append(rt)
ret = pd.concat(ret_rows, ignore_index=True)
ret = ret[["arm", "method", "K", "iteration", "metric", "ref_kind", "reference", "ref_is_own_base", "n",
           "delta_primary", "delta_judge", "retention", "retention_ci_lo", "retention_ci_hi", "same_sign",
           "min_primary_delta"]]
ret["metric"] = pd.Categorical(ret["metric"], RUBRICS, ordered=True)
ret = ret.sort_values(["method", "K", "ref_kind", "metric", "iteration"]).reset_index(drop=True)
ret["metric"] = ret["metric"].astype(str)
C.save_table(ret, f"{NAME}_retention", caption=(
    "**Gain retention by look-ahead K.** `retention = Δ held-out / Δ primary` of each model state over a "
    "reference base — the train/test generalisation ratio (~1 = the gain is real to a judge that never played "
    "the patient; ~0 = it existed only in the optimised grader). `ref_kind`: `own_base` = the arm's OWN base "
    "draw; `method_LA0_base` / `method_LA5_base` = the method's K=0 / K=5 base draw as a SHARED reference for "
    "both K arms (for a PTO_LA0 row, `own_base` and `method_LA0_base` are the same reference and duplicate "
    "each other by design); `eda_view_PTO_LA{K}_base` = the tracked EDA's convention (PTO's base of the same "
    "view), given for the GRPO arms so the tracked 8_measurement/multijudge_gain_retention.md numbers can be "
    "matched. Iteration-0 rows under a shared reference are two INDEPENDENT base draws (noise floor). "
    "`retention` and its CI are suppressed (blank) where |Δ primary| < `min_primary_delta` — 0.15 on the 1–5 / 1–7 "
    "rubrics (the `reliability.gain_retention` default, whose persona-bootstrap CI this is) and 0.05 on the 0–1 rate "
    f"metrics PCT/MICI (their deltas are ~3× smaller; a 0.15 floor blanks almost every MICI row). {PAIRING} Direction-agnostic on MICI "
    f"(both deltas flip together). {CENSOR} Cross-check: GRPO_LA5 Q1 iteration 5 vs eda_view_PTO_LA5_base = "
    "1.082 [0.937, 1.274]; PTO_LA5 Q2 iteration 10 = 0.562 vs PTO_LA0 0.849 (tracked L5/L0 tables)."))

# ── compact endpoint summary (own-base) for the paper body
def ret_at(arm, it, m, kind="own_base"):
    r = ret[(ret.arm == arm) & (ret.iteration == it) & (ret.metric == m) & (ret.ref_kind == kind)]
    return r.iloc[0] if len(r) else None


sum_rows = []
for method in ("PTO", "GRPO"):
    for it in sorted({5, LAST_ITER[f"{method}_LA5"]}):
        for m in ["Q1Q2", "Q1", "Q2", "MITI", "MICI"]:
            r0, r5 = ret_at(f"{method}_LA0", it, m), ret_at(f"{method}_LA5", it, m)
            disjoint = (not np.isnan(r0.retention_ci_lo) and not np.isnan(r5.retention_ci_lo)
                        and (r0.retention_ci_hi < r5.retention_ci_lo or r5.retention_ci_hi < r0.retention_ci_lo))
            sum_rows.append({"method": method, "iteration": it, "metric": m,
                             "K0_delta_primary": r0.delta_primary, "K0_delta_judge": r0.delta_judge,
                             "K0_retention": r0.retention, "K0_ci_lo": r0.retention_ci_lo, "K0_ci_hi": r0.retention_ci_hi,
                             "K5_delta_primary": r5.delta_primary, "K5_delta_judge": r5.delta_judge,
                             "K5_retention": r5.retention, "K5_ci_lo": r5.retention_ci_lo, "K5_ci_hi": r5.retention_ci_hi,
                             "cis_disjoint": disjoint})
ret_sum = pd.DataFrame(sum_rows)
C.save_table(ret_sum, f"{NAME}_retention_summary", caption=(
    "**Gain retention, K=0 vs K=5 side by side (own-base reference)** at iteration 5 (the last iteration all "
    "four arms share) and at each K=5 arm's endpoint (PTO 10, GRPO 5). retention = Δ held-out (Claude Haiku "
    "4.5) / Δ primary (gpt-4o-mini) over the arm's own base; CI = persona bootstrap; `cis_disjoint` = the two "
    f"K arms' retention intervals do not overlap. {PAIRING} {CENSOR} Full table: `cross_k_multijudge_retention`."))

# ── cross-checks vs tracked tables
r = ret_at("GRPO_LA5", 5, "Q1", "eda_view_PTO_LA5_base")
print(f"[check] GRPO_LA5 Q1 I5 retention vs PTO_LA5_Base: {r.retention:.3f} [{r.retention_ci_lo:.3f}, {r.retention_ci_hi:.3f}] (tracked 1.082 [0.937, 1.274])")
assert abs(r.retention - 1.082) < 0.002
r = ret_at("PTO_LA5", 10, "Q2"); r0 = ret_at("PTO_LA0", 10, "Q2")
print(f"[check] PTO_LA5 Q2 I10 own-base retention {r.retention:.3f} (tracked 0.562); PTO_LA0 {r0.retention:.3f} (tracked 0.849)")
assert abs(r.retention - 0.562) < 0.002 and abs(r0.retention - 0.849) < 0.002

# ═══════════════════════════════════════════════════════════════════════════════
# 3. K × method interaction — DiD per persona at matched iterations 0..5, + method gap at each K
# ═══════════════════════════════════════════════════════════════════════════════
did_rows, gap_rows = [], []
for jkey in ("primary", "heldout"):
    for m in RUBRICS:
        w = W[jkey][m]
        # simple method gap PTO − GRPO at each K, every matched iteration of that K
        for K in (0, 5):
            for it in range(0, min(LAST_ITER[f"PTO_LA{K}"], LAST_ITER[f"GRPO_LA{K}"]) + 1):
                a, b = model_name(f"PTO_LA{K}", it), model_name(f"GRPO_LA{K}", it)
                r = contrast(jkey, m, a, b)
                gap_rows.append({"judge": JUDGE_NAME[jkey], "K": K, "iteration": it, "metric": m,
                                 "contrast": f"{a.replace('Exp3', '')} − {b.replace('Exp3', '')}", **r,
                                 "favours": favours(m, r["mean_delta"], "PTO", "GRPO")})
        # DiD
        for it in range(0, LAST_ITER["GRPO_LA5"] + 1):
            g0 = (w[model_name("PTO_LA0", it)] - w[model_name("GRPO_LA0", it)])
            g5 = (w[model_name("PTO_LA5", it)] - w[model_name("GRPO_LA5", it)])
            r = C.paired(g0.to_numpy(), g5.to_numpy())
            did_rows.append({"judge": JUDGE_NAME[jkey], "iteration": it, "metric": m,
                             "gap_K0": float(np.nanmean(g0)), "gap_K5": float(np.nanmean(g5)),
                             "did_mean": r["mean_delta"], "did_dz": r["dz"], "did_ci_lo": r["ci_lo"],
                             "did_ci_hi": r["ci_hi"], "p": r["p"], "n": r["n"]})
gap = pd.DataFrame(gap_rows).rename(columns={"mean_delta": "delta"})
gap = holm_within(gap, ["judge", "K", "metric"], "p", "p_holm")           # across iterations
gap = holm_within(gap, ["judge", "K", "iteration"], "p", "p_holm_rubrics")  # across the 9 rubrics (EDA convention)
gap["metric"] = pd.Categorical(gap["metric"], RUBRICS, ordered=True)
gap = gap.sort_values(["judge", "K", "metric", "iteration"]).reset_index(drop=True)
gap["metric"] = gap["metric"].astype(str)
gap = gap[["judge", "K", "iteration", "metric", "contrast", "n", "delta", "dz", "ci_lo", "ci_hi", "p", "p_holm",
           "p_holm_rubrics", "favours"]]
C.save_table(gap, f"{NAME}_method_gap", caption=(
    "**The method gap at each look-ahead K under both graders.** `PTO_LA{K}_In − GRPO_LA{K}_In` at every matched "
    "iteration (iteration 0 = two independent base draws). Sign: **+ => PTO higher**; on MICI (lower-is-better) "
    f"+ favours GRPO — read `favours`. {PAIRING} `judge` names the grader (gpt-4o-mini = training oracle; "
    "claude-haiku-4-5 = held-out). CI = persona bootstrap; p = Wilcoxon; `p_holm` = Holm across ITERATIONS "
    "within (grader, K, metric); `p_holm_rubrics` = Holm across the 9 rubrics within (grader, K, iteration) "
    f"(the tracked EDA's `method_paired_by_K` convention). {CENSOR}"))

did = pd.DataFrame(did_rows)
did = holm_within(did, ["judge", "metric"], "p", "p_holm")               # across iterations 0..5
did = holm_within(did, ["judge", "iteration"], "p", "p_holm_rubrics")    # across rubrics
did["metric"] = pd.Categorical(did["metric"], RUBRICS, ordered=True)
did = did.sort_values(["judge", "metric", "iteration"]).reset_index(drop=True)
did["metric"] = did["metric"].astype(str)
did = did[["judge", "iteration", "metric", "n", "gap_K0", "gap_K5", "did_mean", "did_dz", "did_ci_lo", "did_ci_hi",
           "p", "p_holm", "p_holm_rubrics"]]
C.save_table(did, f"{NAME}_did", caption=(
    "**K × method interaction (difference-in-differences) per persona**, iterations 0..5 (the only iterations "
    "all four arms share — GRPO_LA5 stops at 5, so the interaction is NOT estimable at 6..10; iteration 0 = four "
    "independent base draws, a noise-floor row). `gap_K0` = mean(PTO_LA0 − GRPO_LA0), `gap_K5` = mean(PTO_LA5 − "
    "GRPO_LA5) (+ => PTO higher); `did = gap_K0 − gap_K5` computed persona by persona, so **+ => PTO's lead over "
    "GRPO is LARGER at K=0 than at K=5** (equivalently, look-ahead helps GRPO more than PTO). On MICI the sign "
    f"reads the other way round (lower is better). {PAIRING} `judge` names the grader. dz = mean/SD of the "
    "per-persona DiD; CI = persona bootstrap; p = Wilcoxon; `p_holm` = Holm across iterations 0..5 within "
    "(grader, metric); `p_holm_rubrics` = Holm across the 9 rubrics within (grader, iteration). "
    "Cross-check: held-out Q1Q2 iteration 5 dz ≈ 0.525 (STATUS.md / L5 SUMMARY §4)."))

chk = did[(did.judge == JUDGE_NAME["heldout"]) & (did.metric == "Q1Q2") & (did.iteration == 5)].iloc[0]
print(f"[check] DiD iter 5 held-out Q1Q2: mean {chk.did_mean:.3f} dz {chk.did_dz:.3f} p_holm(iters) {chk.p_holm:.4f} "
      f"p_holm(rubrics) {chk.p_holm_rubrics:.4f} (STATUS: dz 0.525, p_holm .0001)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. held-out endpoint contrasts the paper quotes
# ═══════════════════════════════════════════════════════════════════════════════
def best_iter(arm: str, jkey: str, metric: str = "Q1Q2") -> int:
    means = {it: float(W[jkey][metric][model_name(arm, it)].mean()) for it in range(1, LAST_ITER[arm] + 1)}
    return max(means, key=means.get)


best_grpo0 = {j: best_iter("GRPO_LA0", j) for j in ("primary", "heldout")}
print(f"[info] GRPO_LA0 best Q1Q2 iteration: primary I{best_grpo0['primary']}, held-out I{best_grpo0['heldout']}")
end_pairs = [
    ("PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline)", "PTOExp3_LA0_I10", "GRPOExp3_LA0_I10"),
    ("PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints)", "PTOExp3_LA5_I10", "GRPOExp3_LA5_I5"),
    ("PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint)", "PTOExp3_LA5_I10", "PTOExp3_LA0_I10"),
    ("GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter)", "GRPOExp3_LA5_I5", "GRPOExp3_LA0_I5"),
    ("GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint)", "GRPOExp3_LA5_I5", "GRPOExp3_LA0_I10"),
    (f"GRPO_LA5_I5 − GRPO_LA0_I{best_grpo0['primary']} (K=0 best by primary Q1Q2)", "GRPOExp3_LA5_I5",
     f"GRPOExp3_LA0_I{best_grpo0['primary']}"),
]
if best_grpo0["heldout"] != best_grpo0["primary"]:
    end_pairs.append((f"GRPO_LA5_I5 − GRPO_LA0_I{best_grpo0['heldout']} (K=0 best by held-out Q1Q2)", "GRPOExp3_LA5_I5",
                      f"GRPOExp3_LA0_I{best_grpo0['heldout']}"))
end_rows = []
for label, a, b in end_pairs:
    for m in RUBRICS:
        rec = both_graders(m, a, b)
        rec["pair"] = label
        end_rows.append(rec)
end = pd.DataFrame(end_rows)
end = holm_within(end, ["pair"], "primary_p", "primary_p_holm")   # across the 9 rubrics within a pair
end = holm_within(end, ["pair"], "judge_p", "judge_p_holm")
plus_minus = lambda lab: (lab.split(" − ")[0].split("_")[0], lab.split(" − ")[1].split(" ")[0].split("_")[0])
end["favours_primary"] = [favours(m, d, "A", "B") for m, d in zip(end.metric, end.primary_delta)]
end["favours_judge"] = [favours(m, d, "A", "B") for m, d in zip(end.metric, end.judge_delta)]
end = end[["pair", "metric", "primary_n", "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi",
           "primary_p", "primary_p_holm", "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p",
           "judge_p_holm", "same_sign", "judge_ci_excl0", "favours_primary", "favours_judge"]]
C.save_table(end, f"{NAME}_endpoints", caption=(
    "**Endpoint contrasts under both graders** (`A − B` as named in `pair`; + => A higher; on MICI, lower is "
    "better, so + favours B — read `favours_*`, where A/B are the pair's left/right model). `primary_*` = training "
    f"oracle gpt-4o-mini; `judge_*` = held-out Claude Haiku 4.5. {PAIRING} CI = persona bootstrap; p = Wilcoxon; "
    "`*_p_holm` = Holm across the 9 rubrics within a pair (the tracked EDA's `compare_two_models` convention). "
    f"GRPO_LA0's best iteration is chosen by mean Q1Q2 under each grader (primary I{best_grpo0['primary']}, "
    f"held-out I{best_grpo0['heldout']}). {CENSOR}"))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. figures
# ═══════════════════════════════════════════════════════════════════════════════
C.style()
PAL = C.palette()
plt.rcParams.update({"axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5})
GRADER_LABEL = {"primary": "training oracle (gpt-4o-mini)", "heldout": "held-out judge (Claude Haiku 4.5)"}
GRADER_TITLE = {"primary": "gpt-4o-mini (training oracle)", "heldout": "Claude Haiku 4.5 (held-out)"}


def _unit(metric):
    return "rate" if metric in ("PCT", "MICI") else "score points"


# ── fig A: the K contrast under both graders (rows Q1Q2, MICI; cols PTO, GRPO)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True)
for i, m in enumerate(["Q1Q2", "MICI"]):
    for j, method in enumerate(["PTO", "GRPO"]):
        ax = axes[i, j]
        sub = pairs[(pairs.method == method) & (pairs.metric == m)].sort_values("iteration")
        col = PAL[f"{method}_LA0"]
        # primary: solid line + filled circle + CI ribbon; held-out: dotted line + open circle + CI bars
        ax.fill_between(sub.iteration, sub.primary_ci_lo, sub.primary_ci_hi, color=col, alpha=0.13, lw=0)
        ax.plot(sub.iteration, sub.primary_delta, ls="-", marker="o", color=col, lw=1.6, ms=5.5)
        ax.errorbar(sub.iteration, sub.judge_delta, yerr=[sub.judge_delta - sub.judge_ci_lo, sub.judge_ci_hi - sub.judge_delta],
                    fmt="none", ecolor=col, elinewidth=0.9, capsize=2, alpha=0.8)
        ax.plot(sub.iteration, sub.judge_delta, ls=":", marker="o", mfc="white", mec=col, color=col, lw=1.6, ms=5.5)
        for pref in ("primary", "judge"):
            sig = sub[sub[f"{pref}_p_holm"] < 0.05]
            if len(sig):
                ax.scatter(sig.iteration, sig[f"{pref}_delta"], marker="*", s=75, color=col, zorder=5,
                           edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="0.35", lw=0.8)
        ax.set_title(f"{method}: {m}, K=0 − K=5, both graders" + (" (LA5 ends at 5)" if method == "GRPO" else ""))
        ax.set_ylabel(f"{m} Δ (K=0 − K=5), {_unit(m)}" if j == 0 else "")
        ax.set_xticks(range(0, 11))
        if i == 1:
            ax.set_xlabel("iteration (0 = base: two independent draws)")
        ax.grid(True, alpha=0.35)
h = [Line2D([], [], color="0.2", ls="-", marker="o", ms=5.5, label=GRADER_LABEL["primary"] + ", CI ribbon"),
     Line2D([], [], color="0.2", ls=":", marker="o", mfc="white", ms=5.5, label=GRADER_LABEL["heldout"] + ", CI bars"),
     Line2D([], [], color="0.2", ls="none", marker="*", ms=9, mec="black", mew=0.5, label="Holm p < .05 (across iterations)")]
lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8,
                title="sign: + => K=0 higher.  MICI is lower-is-better, so on MICI + favours K=5.  Paired on persona (n = 96).",
                title_fontsize=7.5)
lg.get_title().set_color("0.3")
figA = C.save_fig(fig, f"{NAME}_fig_kcontrast")

# ── fig B: gain retention by K (rows Q1, Q2, MICI; cols PTO, GRPO), own-base reference
fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.6), sharex=True)
for i, m in enumerate(["Q1", "Q2", "MICI"]):
    for j, method in enumerate(["PTO", "GRPO"]):
        ax = axes[i, j]
        for K in (0, 5):
            arm = f"{method}_LA{K}"
            sub = ret[(ret.arm == arm) & (ret.metric == m) & (ret.ref_kind == "own_base")].sort_values("iteration")
            col, st = PAL[arm], C.K_STYLE[K]
            ax.plot(sub.iteration, sub.retention, ls=st["ls"], marker=st["marker"], color=col, lw=1.7, ms=5.5,
                    label=f"{arm.replace('_LA', ' K=')}")
            ax.fill_between(sub.iteration, sub.retention_ci_lo, sub.retention_ci_hi, color=col, alpha=0.15, lw=0)
        ax.axhline(1.0, color="0.25", lw=0.9, ls="-.")
        ax.axhline(0.0, color="0.6", lw=0.7)
        ax.set_title(f"{method}: {m} gain retention" + (" (harm channel)" if m == "MICI" else "")
                     + (" (LA5 ends at 5)" if method == "GRPO" else ""))
        ax.set_ylabel("retention = Δ held-out / Δ primary" if j == 0 else "")
        ax.set_xticks(range(0, 11))
        ax.set_ylim((-0.5, 6.0) if m == "MICI" else (-0.4, 2.0))
        ax.grid(True, alpha=0.35)
        if i == 2:
            ax.set_xlabel("iteration (gain over the arm's OWN base)")
h = [Line2D([], [], color=PAL[a], ls=C.K_STYLE[C.k_of(a)]["ls"], marker=C.K_STYLE[C.k_of(a)]["marker"], ms=5.5, lw=1.7,
            label=a.replace("_LA", " K=")) for a in C.ARMS]
h.append(Line2D([], [], color="0.25", ls="-.", lw=0.9, label="retention = 1 (gain fully real to the held-out judge)"))
lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=8,
                title="Δ held-out = Claude Haiku 4.5 gain over base;  Δ primary = gpt-4o-mini (training oracle) gain over base;  "
                      "ribbons = persona-bootstrap 95% CI;  blank where |Δ primary| < 0.15 (Q1/Q2) or < 0.05 (MICI)",
                title_fontsize=7.2)
lg.get_title().set_color("0.3")
figB = C.save_fig(fig, f"{NAME}_fig_retention")

# ── fig C: method gap by iteration at K=0 vs K=5, two graders as columns; row 2 = the DiD (Q1Q2)
GAP_COL = {0: "#111111", 5: "#009E73"}
DID_COL = "#CC79A7"
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
for j, jkey in enumerate(["primary", "heldout"]):
    jn = JUDGE_NAME[jkey]
    ax = axes[0, j]
    for K in (0, 5):
        sub = gap[(gap.judge == jn) & (gap.K == K) & (gap.metric == "Q1Q2")].sort_values("iteration")
        st = C.K_STYLE[K]
        ax.fill_between(sub.iteration, sub.ci_lo, sub.ci_hi, color=GAP_COL[K], alpha=0.13, lw=0)
        ax.plot(sub.iteration, sub.delta, ls=st["ls"], marker=st["marker"], color=GAP_COL[K], lw=1.7, ms=5.5)
        sig = sub[sub.p_holm < 0.05]
        ax.scatter(sig.iteration, sig.delta, marker="*", s=75, color=GAP_COL[K], zorder=5, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title(f"Q1Q2 gap PTO − GRPO: {GRADER_TITLE[jkey]}")
    ax.set_ylabel("Q1Q2 Δ (PTO − GRPO), score points" if j == 0 else "")
    ax.grid(True, alpha=0.35)
    ax = axes[1, j]
    sub = did[(did.judge == jn) & (did.metric == "Q1Q2")].sort_values("iteration")
    ax.fill_between(sub.iteration, sub.did_ci_lo, sub.did_ci_hi, color=DID_COL, alpha=0.15, lw=0)
    ax.plot(sub.iteration, sub.did_mean, ls="-", marker="D", color=DID_COL, lw=1.7, ms=5.5)
    sig = sub[sub.p_holm < 0.05]
    ax.scatter(sig.iteration, sig.did_mean, marker="*", s=85, color=DID_COL, zorder=5, edgecolor="black", linewidth=0.5)
    for _, r in sub.iterrows():
        ax.annotate(f"dz {r.did_dz:.2f}", (r.iteration, r.did_ci_hi), textcoords="offset points", xytext=(0, 3),
                    ha="left" if r.iteration == 0 else "center", fontsize=6.5, color="0.25")
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title(f"Q1Q2 DiD: {GRADER_TITLE[jkey]}")
    ax.set_ylabel("Q1Q2 DiD, score points" if j == 0 else "")
    ax.set_xlabel("iteration (0 = base draws; DiD estimable only to 5)")
    ax.set_xticks(range(0, 11))
    ax.grid(True, alpha=0.35)
for row in (0, 1):
    ylo = min(a.get_ylim()[0] for a in axes[row]); yhi = max(a.get_ylim()[1] for a in axes[row])
    for a in axes[row]:
        a.set_ylim(ylo, yhi + (0.12 if row == 1 else 0))
h = [Line2D([], [], color=GAP_COL[0], ls="-", marker="o", ms=5.5, lw=1.7, label="K=0: PTO_LA0 − GRPO_LA0"),
     Line2D([], [], color=GAP_COL[5], ls="--", marker="s", ms=5.5, lw=1.7, label="K=5: PTO_LA5 − GRPO_LA5 (to iter 5)"),
     Line2D([], [], color=DID_COL, ls="-", marker="D", ms=5.5, lw=1.7, label="DiD = gap(K=0) − gap(K=5)"),
     Line2D([], [], color="0.2", ls="none", marker="*", ms=9, mec="black", mew=0.5, label="Holm p < .05 (across iterations)")]
lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8,
                title="signs: gap + => PTO higher;  DiD + => PTO's lead over GRPO is larger at K=0 than at K=5.  "
                      "Ribbons = persona-bootstrap 95% CI, n = 96 personas.", title_fontsize=7.5)
lg.get_title().set_color("0.3")
figC = C.save_fig(fig, f"{NAME}_fig_did")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ledger
# ═══════════════════════════════════════════════════════════════════════════════
def _row(r, cols):
    return {c: (None if isinstance(r[c], float) and np.isnan(r[c]) else r[c]) for c in cols}


T = f"tables/{NAME}"
for _, r in pairs.iterrows():
    L.put(f"kcontrast.{r.method}.{r.metric}.iter{r.iteration}",
          _row(r, ["primary_n", "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi", "primary_p", "primary_p_holm",
                   "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p", "judge_p_holm", "same_sign",
                   "judge_ci_excl0", "favours_primary", "favours_judge"]),
          source=f"{T}_pairs.md row method={r.method} metric={r.metric} iteration={r.iteration}",
          note="+ => K=0 higher; MICI lower-better")
for _, r in ladder.iterrows():
    L.put(f"ladder.{r.group}.{r.subset}", _row(r, ["n_contrasts", "n_same_sign", "pct_same_sign"]),
          source=f"{T}_ladder.md row group={r.group} subset={r.subset}")
for _, r in ret.iterrows():
    L.put(f"retention.{r.arm}.{r.metric}.iter{r.iteration}.{r.ref_kind}",
          _row(r, ["reference", "n", "delta_primary", "delta_judge", "retention", "retention_ci_lo", "retention_ci_hi", "same_sign"]),
          source=f"{T}_retention.md row arm={r.arm} metric={r.metric} iteration={r.iteration} ref_kind={r.ref_kind}")
for _, r in ret_sum.iterrows():
    L.put(f"retention_summary.{r.method}.{r.metric}.iter{r.iteration}",
          _row(r, [c for c in ret_sum.columns if c not in ("method", "metric", "iteration")]),
          source=f"{T}_retention_summary.md row method={r.method} metric={r.metric} iteration={r.iteration}")
for _, r in gap.iterrows():
    L.put(f"method_gap.K{r.K}.{r.metric}.iter{r.iteration}.{r.judge}",
          _row(r, ["n", "delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "p_holm_rubrics", "favours"]),
          source=f"{T}_method_gap.md row judge={r.judge} K={r.K} metric={r.metric} iteration={r.iteration}",
          note="+ => PTO higher")
for _, r in did.iterrows():
    L.put(f"did.{r.metric}.iter{r.iteration}.{r.judge}",
          _row(r, ["n", "gap_K0", "gap_K5", "did_mean", "did_dz", "did_ci_lo", "did_ci_hi", "p", "p_holm", "p_holm_rubrics"]),
          source=f"{T}_did.md row judge={r.judge} metric={r.metric} iteration={r.iteration}",
          note="did = (PTO_LA0-GRPO_LA0)-(PTO_LA5-GRPO_LA5) per persona; + => PTO lead larger at K=0")
for _, r in end.iterrows():
    L.put(f"endpoint.{r.pair.split(' (')[0]}.{r.metric}",
          _row(r, ["primary_n", "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi", "primary_p", "primary_p_holm",
                   "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p", "judge_p_holm", "same_sign",
                   "judge_ci_excl0"]),
          source=f"{T}_endpoints.md row pair='{r.pair}' metric={r.metric}", note="+ => A (left model) higher")
L.put("grpo_la0_best_iter_by_q1q2", best_grpo0, source=f"{T}_endpoints.md caption")
L.put("figures", {"kcontrast": str(figA.name), "retention": str(figB.name), "did": str(figC.name)}, source="figures/")
p = L.save()
print("ledger ->", p)
print("done.")
