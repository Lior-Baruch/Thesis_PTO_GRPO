"""session_shape_stability.py — three ICLR look-ahead-paper claims re-tested on Exp3, plus the
session-shape K contrast and the selection-level length push.

1. Session shape (deterministic text metrics from the transcripts, judge-invariant): conv_len,
   n_th_turns, mean_turn_len, q_per_turn, loop — persona-paired K0 - K5 within each method at every
   matched iteration (+ => K=0 higher).                       -> *_shape.md, *_fig_shape.png
2. ICLR "stability" claim (K=5 has the lowest SD): per arm x iteration SD / IQR of Q1, Q2, Q1Q2 under
   BOTH graders; Brown-Forsythe (independent-groups) and Pitman-Morgan (persona-paired) variance tests
   K0 vs K5 per matched iteration; ceiling-compression check = share of conversations at Q1Q2 >= 4.5
   / == 5 by cooperation level.                                -> *_sd.md, *_sd_bf.md, *_ceiling.md, *_fig_sd.png
3. ICLR "shorter conversations" claim: base -> final conv_len per arm + the K contrast at PTO iter 10
   and GRPO iter 5, cross-checked against the tracked 7_stats k_paired_channels sheet.
                                                              -> *_length_endpoints.md
4. Selection-level length push + praise weights (NO recompute): the tracked 6_preference tables
   update_lexical_push / generation_pool_means from BOTH views, joined into one compact table.
                                                              -> *_selection.md
5. Everything ledgered into out/session_shape_stability.json.

Run:  .venv/Scripts/python.exe papers/2026_lookahead_pto_grpo/analysis/session_shape_stability.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E702
import _common as C  # noqa: E402

import re  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import eda_analysis  # noqa: E402
from eda_analysis import behavior  # noqa: E402
from eda_analysis.constants import arm_label  # noqa: E402

SCRIPT = "session_shape_stability"
L = C.Ledger(SCRIPT)
SIGN = "Sign: + => K=0 higher (K0 - K5)."
PAIR = "Pairing unit: persona_id (the per-iteration file shuffle replayed; never file_index)."
CENSOR = "GRPO_LA5 is right-censored at iteration 5 (its K=0 sibling runs to 10)."
ITER0 = "Iteration 0 = two INDEPENDENT base draws (same base policy) — a free noise-floor row."

SHAPE_METRICS = ["conv_len", "n_th_turns", "mean_turn_len", "q_per_turn", "loop"]
SHAPE_UNITS = {"conv_len": "utterances / conversation", "n_th_turns": "therapist turns / conversation",
               "mean_turn_len": "chars / therapist turn", "q_per_turn": "'?' / therapist turn",
               "loop": "share of conversations with a verbatim-repeated therapist turn"}
STAB_METRICS = ["Q1", "Q2", "Q1Q2"]
COOP_LABEL = {"Low": "Resistant", "High": "Cooperative", "StartLowAndChangesToHigh": "WarmsUp"}
METHODS = ["PTO", "GRPO"]


# ── helpers ──────────────────────────────────────────────────────────────────
def _pair_wide(df: pd.DataFrame, value: str, arm_a: str, arm_b: str, it: int):
    """Persona-aligned (a, b) arrays of *value* for two arms at one iteration."""
    a = df[(df["arm"] == arm_a) & (df["iteration"] == it)][["persona_id", value]].dropna()
    b = df[(df["arm"] == arm_b) & (df["iteration"] == it)][["persona_id", value]].dropna()
    m = a.merge(b, on="persona_id", suffixes=("_a", "_b"))
    return m[f"{value}_a"].to_numpy(float), m[f"{value}_b"].to_numpy(float)


def _common_iters(df, arm_a, arm_b):
    ia = set(df.loc[df["arm"] == arm_a, "iteration"]); ib = set(df.loc[df["arm"] == arm_b, "iteration"])
    return sorted(int(i) for i in ia & ib)


def _iqr(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if x.size else np.nan


def brown_forsythe(a, b) -> dict:
    """Brown-Forsythe = Levene on |x - group median|; two INDEPENDENT groups (pairing not used)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size < 3 or b.size < 3:
        return dict(bf_W=np.nan, bf_p=np.nan)
    r = sps.levene(a, b, center="median")
    return dict(bf_W=float(r.statistic), bf_p=float(r.pvalue))


def pitman_morgan(a, b) -> dict:
    """Pitman-Morgan test for equal variances of PAIRED samples: corr(a+b, a-b) = 0 <=> var(a)=var(b).
    r > 0 => var(a) > var(b) (here a = K0, b = K5, so r > 0 => K=0 more dispersed)."""
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
    """Parse a pipe-delimited markdown table (as written by the EDA exports) into a DataFrame."""
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip().startswith("|")]
    lines = [ln for ln in lines if not re.match(r"^\|\s*:?-{2,}", ln)]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 0. Load — scores under both graders ONCE, then the transcripts' text metrics
# ═════════════════════════════════════════════════════════════════════════════
print("[load] scores under both graders ...")
SC = C.load_scores_both()                       # {'primary', 'heldout'}
JUDGE_OF = {"primary": C.JUDGE_SHORT[C.PRIMARY], "heldout": C.JUDGE_SHORT[C.HELDOUT]}

print("[load] text metrics from the transcripts (judge-invariant) ...")
ARMS = eda_analysis.cross_k_arms(eda_analysis.EdaConfig(view="L5", verbose=False))
TM = behavior.text_metrics(ARMS, attach_persona=True)
TM = TM[TM["arm"].isin(C.ARMS)].copy()
TM["loop"] = TM["loop"].astype(float)
assert (TM.groupby(["arm", "iteration"]).size() == 96).all(), "expected 96 conversations per arm x iteration"
assert TM.groupby(["arm", "iteration"])["persona_id"].nunique().eq(96).all(), "persona recovery not 1:1"

# self-check of the loader/pairing against the tracked EDA (k_paired_by_method: PTO Q1Q2 iter 6 primary)
w = C.wide(SC["primary"], "Q1Q2")
chk = C.paired(w["PTOExp3_LA0_I6"].to_numpy(), w["PTOExp3_LA5_I6"].to_numpy())
print(f"[check] PTO Q1Q2 iter 6 primary K0-K5: delta={chk['mean_delta']:+.3f} dz={chk['dz']:.3f} "
      f"(tracked k_paired_by_method: +0.257, 0.417)")
assert abs(chk["mean_delta"] - 0.257) < 0.002 and abs(chk["dz"] - 0.417) < 0.002
L.put("_crosscheck.pto_q1q2_iter6_primary", {"mean_delta": chk["mean_delta"], "dz": chk["dz"], "n": chk["n"]},
      source="reproduces results/L5/tables/7_stats/gpt-4o-mini/k_paired_by_method.md row PTO/6/Q1Q2")

# ═════════════════════════════════════════════════════════════════════════════
# 1. Session shape — persona-paired K0 - K5 by matched iteration
# ═════════════════════════════════════════════════════════════════════════════
lvl = (TM.groupby(["arm", "method", "K", "iteration"])[SHAPE_METRICS]
       .agg(["mean", "sem"]).reset_index())
lvl.columns = ["_".join(c).rstrip("_") for c in lvl.columns]

rows = []
for method in METHODS:
    a0, a5 = f"{method}_LA0", f"{method}_LA5"
    for metric in SHAPE_METRICS:
        fam = []
        for it in _common_iters(TM, a0, a5):
            x, y = _pair_wide(TM, metric, a0, a5, it)
            r = C.paired(x, y)
            fam.append({"method": method, "metric": metric, "iteration": it,
                        "mean_K0": float(np.nanmean(x)), "mean_K5": float(np.nanmean(y)),
                        "n": r["n"], "mean_delta": r["mean_delta"], "dz": r["dz"],
                        "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"]})
        f = pd.DataFrame(fam)
        f["p_holm"] = C.holm(f["p"].to_numpy())          # family = (method, metric) across iterations
        rows.append(f)
SHAPE = pd.concat(rows, ignore_index=True)
SHAPE["metric_unit"] = SHAPE["metric"].map(SHAPE_UNITS)

C.save_table(SHAPE, f"{SCRIPT}_shape", caption=(
    "**Session shape, persona-paired K0 - K5 by matched iteration.** Deterministic text metrics computed "
    "from the eval transcripts (`eda_analysis.behavior.text_metrics`; judge-invariant): conv_len = utterances "
    "per conversation (therapist + patient), n_th_turns = therapist turns, mean_turn_len = characters per "
    f"therapist turn, q_per_turn = literal '?' per therapist turn, loop = share of conversations with a "
    f"verbatim-repeated therapist turn (degeneracy). {SIGN} {PAIR} mean_K0/mean_K5 are the arm means over "
    "the same 96 personas; mean_delta/dz/bootstrap 95% CI/Wilcoxon p are on the paired deltas; p_holm is "
    "Holm-corrected WITHIN each (method, metric) family ACROSS iterations (the tracked 7_stats "
    "k_paired_channels corrects across channels within an iteration instead — same delta/dz/p, different "
    f"p_holm scope). {ITER0} {CENSOR} Length metrics are unvalenced (longer is not better)."))

for _, r in SHAPE.iterrows():
    L.put(f"shape.{r.method}.{r.metric}.iter{int(r.iteration)}",
          {k: r[k] for k in ["n", "mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]},
          source=f"tables/{SCRIPT}_shape.md row method={r.method} metric={r.metric} iteration={int(r.iteration)}")

# per-arm levels (base -> final) for the shape metrics
arm_levels = []
for arm in C.ARMS:
    d = lvl[lvl["arm"] == arm].sort_values("iteration")
    it_last = int(d["iteration"].max())
    row = {"arm": arm, "final_iteration": it_last}
    for m in SHAPE_METRICS:
        b = float(d.loc[d["iteration"] == 0, f"{m}_mean"].iloc[0]); e = float(d.loc[d["iteration"] == it_last, f"{m}_mean"].iloc[0])
        row[f"{m}_base"] = b; row[f"{m}_final"] = e; row[f"{m}_change"] = e - b
    arm_levels.append(row)
    L.put(f"levels.{arm}", row, source=f"tables/{SCRIPT}_length_endpoints.md row arm={arm}")
LEVELS = pd.DataFrame(arm_levels)

# ── Figure 1: session shape trajectories ─────────────────────────────────────
C.style()
PAL = C.palette()


def _traj(ax, metric, ylabel, title):
    for arm in C.ARMS:
        d = lvl[lvl["arm"] == arm].sort_values("iteration")
        k = C.k_of(arm); st = C.K_STYLE[k]
        m, s = d[f"{metric}_mean"].to_numpy(), d[f"{metric}_sem"].to_numpy()
        ax.fill_between(d["iteration"], m - s, m + s, color=PAL[arm], alpha=0.15, lw=0)
        ax.plot(d["iteration"], m, ls=st["ls"], marker=st["marker"], ms=5, lw=1.7, color=PAL[arm],
                label=arm_label(arm), markerfacecolor=PAL[arm] if k == 0 else "white", markeredgewidth=1.4)
    ax.set_xlabel("iteration"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.set_xticks(range(0, 11)); ax.grid(True, alpha=0.35)


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
_traj(axes[0], "conv_len", "utterances / conversation", "Conversation length")
_traj(axes[1], "mean_turn_len", "chars / therapist turn", "Therapist turn length")
_traj(axes[2], "q_per_turn", "'?' / therapist turn", "Questions per turn")
hnd, lab = axes[0].get_legend_handles_labels()
fig.legend(hnd, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
axes[2].set_ylim(bottom=0)
fig.suptitle("Session shape by iteration — deterministic text metrics (mean ± SE over 96 personas; grader-free)",
             fontsize=9.5, y=1.09)
C.save_fig(fig, f"{SCRIPT}_fig_shape")

# ═════════════════════════════════════════════════════════════════════════════
# 2. ICLR "stability": SD / IQR per arm x iteration under both graders + variance tests
# ═════════════════════════════════════════════════════════════════════════════
sd_rows, bf_rows, ceil_rows = [], [], []
for jkey, sc in SC.items():
    judge = JUDGE_OF[jkey]
    sc = sc[sc["arm"].isin(C.ARMS)]
    for metric in STAB_METRICS:
        d = sc[sc["questionnaire"] == metric]
        g = d.groupby(["arm", "iteration"])["score"]
        for (arm, it), s in g:
            s = s.to_numpy(float)
            sd_rows.append({"judge": judge, "metric": metric, "arm": arm, "method": C.method_of(arm),
                            "K": C.k_of(arm), "iteration": int(it), "n": int(s.size), "mean": float(s.mean()),
                            "median": float(np.median(s)), "sd": float(s.std(ddof=1)), "iqr": _iqr(s),
                            "share_ge4": float((s >= 4.0).mean()), "share_ge45": float((s >= 4.5).mean()),
                            "share_eq5": float((s == 5.0).mean())})
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
            f = pd.DataFrame(fam)
            f["bf_p_holm"] = C.holm(f["bf_p"].to_numpy())
            f["pm_p_holm"] = C.holm(f["pm_p"].to_numpy())
            bf_rows.append(f)
    # ceiling compression on Q1Q2 by cooperation level (+ pooled)
    d = sc[sc["questionnaire"] == "Q1Q2"].copy()
    d["coop"] = d["cooperation_level"].map(COOP_LABEL).fillna(d["cooperation_level"])
    for (arm, it), s in d.groupby(["arm", "iteration"]):
        row = {"judge": judge, "arm": arm, "iteration": int(it), "n": int(len(s)),
               "mean_all": float(s["score"].mean()), "sd_all": float(s["score"].std(ddof=1)),
               "share_ge45_all": float((s["score"] >= 4.5).mean()), "share_eq5_all": float((s["score"] == 5).mean())}
        for cl in ["Resistant", "WarmsUp", "Cooperative"]:
            v = s.loc[s["coop"] == cl, "score"].to_numpy(float)
            row[f"n_{cl}"] = int(v.size)
            row[f"mean_{cl}"] = float(v.mean()) if v.size else np.nan
            row[f"sd_{cl}"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
            row[f"share_ge45_{cl}"] = float((v >= 4.5).mean()) if v.size else np.nan
            row[f"share_eq5_{cl}"] = float((v == 5).mean()) if v.size else np.nan
        ceil_rows.append(row)

SD = pd.DataFrame(sd_rows).sort_values(["judge", "metric", "arm", "iteration"]).reset_index(drop=True)
BF = pd.concat(bf_rows, ignore_index=True)
CEIL = pd.DataFrame(ceil_rows).sort_values(["judge", "arm", "iteration"]).reset_index(drop=True)

C.save_table(SD, f"{SCRIPT}_sd", caption=(
    "**Dispersion per arm x iteration (the ICLR 'K=5 is more stable / has the lowest SD' claim), both graders "
    "side by side (never averaged).** For Q1, Q2 and Q1Q2 (= mean of the Q1 and Q2 means, the training "
    "reward): n conversations, mean, median, SD (ddof=1), IQR (Q75 - Q25), and the ceiling shares "
    "(score >= 4, >= 4.5, == 5) over the 96 personas of that model state. No pairing here (within-arm "
    f"descriptives). Grader named in `judge`. {ITER0} {CENSOR} Read SD next to the mean: on a bounded 1-5 "
    "scale a higher mean mechanically compresses SD (see the ceiling table)."))
C.save_table(BF, f"{SCRIPT}_sd_bf", caption=(
    "**Variance contrast K0 vs K5 per matched iteration, both graders.** sd_K0/sd_K5 and iqr_K0/iqr_K5 over "
    "the same 96 personas; sd_ratio_K5_over_K0 < 1 => the K=5 arm is LESS dispersed. bf_W/bf_p = "
    "Brown-Forsythe (Levene on |x - median|; treats the two arms as independent groups, persona pairing not "
    "used). pm_r/pm_p = Pitman-Morgan paired variance test = Pearson r between (K0 + K5) and (K0 - K5) over "
    f"personas; r > 0 => K=0 more dispersed (sign matches {SIGN[:-1]} applied to variance). {PAIR} p_holm "
    f"columns are Holm-corrected within each (judge, method, metric) family across iterations. {ITER0} {CENSOR}"))
C.save_table(CEIL, f"{SCRIPT}_ceiling", caption=(
    "**Ceiling-compression check on Q1Q2 by patient cooperation level (persona trait: Resistant = "
    "cooperation_level Low, WarmsUp = StartLowAndChangesToHigh, Cooperative = High; 32 personas each), per "
    "arm x iteration, both graders.** share_ge45 = share of conversations scoring >= 4.5, share_eq5 = exactly "
    "5.0 (Q1Q2 == 5 requires every Q1 and Q2 item at 5); mean/sd per subgroup. A low arm SD that comes with a "
    "high ceiling share is scale compression, not stability. The held-out judge (Claude Haiku 4.5) never "
    f"awards >= 4.5 on Q1Q2 (its max is 4.25), so its ceiling shares are 0 by construction. {ITER0} {CENSOR}"))

for _, r in BF.iterrows():
    L.put(f"sd_bf.{r.judge}.{r.method}.{r.metric}.iter{int(r.iteration)}",
          {k: r[k] for k in ["n", "sd_K0", "sd_K5", "sd_ratio_K5_over_K0", "iqr_K0", "iqr_K5", "bf_W", "bf_p",
                             "bf_p_holm", "pm_r", "pm_p", "pm_p_holm", "mean_K0", "mean_K5"]},
          source=f"tables/{SCRIPT}_sd_bf.md row judge={r.judge} method={r.method} metric={r.metric} iteration={int(r.iteration)}")
for _, r in SD.iterrows():
    L.put(f"sd.{r.judge}.{r.metric}.{r.arm}.iter{int(r.iteration)}",
          {k: r[k] for k in ["n", "mean", "median", "sd", "iqr", "share_ge4", "share_ge45", "share_eq5"]},
          source=f"tables/{SCRIPT}_sd.md row judge={r.judge} metric={r.metric} arm={r.arm} iteration={int(r.iteration)}")
for _, r in CEIL.iterrows():
    L.put(f"ceiling.{r.judge}.{r.arm}.iter{int(r.iteration)}",
          {k: r[k] for k in CEIL.columns if k not in ("judge", "arm", "iteration")},
          source=f"tables/{SCRIPT}_ceiling.md row judge={r.judge} arm={r.arm} iteration={int(r.iteration)}")

# tally of the variance contrast per (judge, method, metric): how often is the K=5 arm LESS dispersed?
tally = []
for (judge, method, metric), d in BF.groupby(["judge", "method", "metric"], sort=False):
    dt = d[d["iteration"] > 0]
    tally.append({"judge": judge, "method": method, "metric": metric, "n_iters": int(len(dt)),
                  "n_K5_lower_sd": int((dt["sd_K5"] < dt["sd_K0"]).sum()),
                  "n_K5_lower_iqr": int((dt["iqr_K5"] < dt["iqr_K0"]).sum()),
                  "median_sd_ratio_K5_over_K0": float(dt["sd_ratio_K5_over_K0"].median()),
                  "n_pm_holm_sig_K5_lower": int(((dt["pm_p_holm"] < 0.05) & (dt["pm_r"] > 0)).sum()),
                  "n_pm_holm_sig_K0_lower": int(((dt["pm_p_holm"] < 0.05) & (dt["pm_r"] < 0)).sum()),
                  "n_bf_holm_sig": int((dt["bf_p_holm"] < 0.05).sum()),
                  "iter0_sd_K0": float(d.loc[d["iteration"] == 0, "sd_K0"].iloc[0]),
                  "iter0_sd_K5": float(d.loc[d["iteration"] == 0, "sd_K5"].iloc[0])})
TALLY = pd.DataFrame(tally)
C.save_table(TALLY, f"{SCRIPT}_sd_tally", caption=(
    "**Tally of the K0-vs-K5 dispersion contrast over the trained matched iterations (1..N), per grader x "
    "method x rubric.** n_K5_lower_sd / n_K5_lower_iqr = iterations at which the K=5 arm's SD / IQR is smaller "
    "than K=0's; median_sd_ratio = median of sd_K5 / sd_K0 (< 1 => K=5 typically less dispersed); "
    "n_pm_holm_sig_K5_lower / _K0_lower = iterations where the persona-paired Pitman-Morgan test is "
    "Holm-significant (within judge x method x rubric across iterations) with K=5 resp. K=0 less dispersed; "
    "n_bf_holm_sig = Brown-Forsythe Holm-significant iterations (either direction). iter0_sd_* = the two "
    f"independent base draws (noise floor for an SD difference). {CENSOR} PTO: N=10 iterations; GRPO: N=5."))
for _, r in TALLY.iterrows():
    L.put(f"sd_tally.{r.judge}.{r.method}.{r.metric}", {k: r[k] for k in TALLY.columns if k not in ("judge", "method", "metric")},
          source=f"tables/{SCRIPT}_sd_tally.md row judge={r.judge} method={r.method} metric={r.metric}")

# summary of the "lowest SD" claim: which model state has the min SD per (judge, metric); trained iters only
summ = []
for (judge, metric), d in SD.groupby(["judge", "metric"]):
    dt = d[d["iteration"] > 0]
    imin = dt["sd"].idxmin()
    imax_mean = dt["mean"].idxmax()
    rho = sps.spearmanr(dt["mean"], dt["sd"])
    per_arm_min = dt.loc[dt.groupby("arm")["sd"].idxmin(), ["arm", "iteration", "sd", "mean"]]
    summ.append({"judge": judge, "metric": metric,
                 "min_sd_arm": dt.loc[imin, "arm"], "min_sd_iteration": int(dt.loc[imin, "iteration"]),
                 "min_sd": float(dt.loc[imin, "sd"]), "min_sd_mean": float(dt.loc[imin, "mean"]),
                 "max_mean_arm": dt.loc[imax_mean, "arm"], "max_mean_iteration": int(dt.loc[imax_mean, "iteration"]),
                 "max_mean": float(dt.loc[imax_mean, "mean"]), "max_mean_sd": float(dt.loc[imax_mean, "sd"]),
                 "spearman_mean_vs_sd": float(rho.statistic), "spearman_p": float(rho.pvalue),
                 "n_states": int(len(dt)),
                 **{f"min_sd_{a}": float(per_arm_min.loc[per_arm_min['arm'] == a, 'sd'].iloc[0]) for a in C.ARMS},
                 **{f"min_sd_iter_{a}": int(per_arm_min.loc[per_arm_min['arm'] == a, 'iteration'].iloc[0]) for a in C.ARMS}})
SUMM = pd.DataFrame(summ)
C.save_table(SUMM, f"{SCRIPT}_sd_summary", caption=(
    "**Where the lowest SD sits (trained iterations 1..N only; iteration 0 excluded), per grader and rubric.** "
    "min_sd_* = the model state with the smallest across-persona SD and its mean; max_mean_* = the state with "
    "the highest mean and its SD; spearman_mean_vs_sd = rank correlation between arm-iteration mean and SD "
    "across all trained states (n_states) — strongly negative = dispersion tracks the ceiling, not the "
    f"optimizer; min_sd_<arm> / min_sd_iter_<arm> = each arm's own lowest SD and where it occurs. {CENSOR}"))
for _, r in SUMM.iterrows():
    L.put(f"sd_summary.{r.judge}.{r.metric}", {k: r[k] for k in SUMM.columns if k not in ("judge", "metric")},
          source=f"tables/{SCRIPT}_sd_summary.md row judge={r.judge} metric={r.metric}")

# ── Figure 2: SD of Q1Q2 by iteration, two grader panels + SD-vs-mean scatter ─
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1.1, 0.95]})
for ax, jkey in zip(axes[:2], ["primary", "heldout"]):
    judge = JUDGE_OF[jkey]
    d0 = SD[(SD["judge"] == judge) & (SD["metric"] == "Q1Q2")]
    for arm in C.ARMS:
        d = d0[d0["arm"] == arm].sort_values("iteration"); k = C.k_of(arm); st = C.K_STYLE[k]
        ax.plot(d["iteration"], d["sd"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.7, color=PAL[arm],
                label=arm_label(arm), markerfacecolor=PAL[arm] if k == 0 else "white", markeredgewidth=1.4)
    ax.set_xlabel("iteration"); ax.set_ylabel("SD of Q1Q2 over 96 personas" if jkey == "primary" else "")
    ax.set_title(f"grader: {judge}", fontsize=10); ax.set_xticks(range(0, 11)); ax.grid(True, alpha=0.35)
hnd, lab = axes[0].get_legend_handles_labels()
fig.legend(hnd, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
ax = axes[2]
for jkey, mk_fill in (("primary", True), ("heldout", False)):
    judge = JUDGE_OF[jkey]
    d0 = SD[(SD["judge"] == judge) & (SD["metric"] == "Q1Q2") & (SD["iteration"] > 0)]
    for arm in C.ARMS:
        d = d0[d0["arm"] == arm]; k = C.k_of(arm); st = C.K_STYLE[k]
        ax.scatter(d["mean"], d["sd"], marker=st["marker"], s=22, color=PAL[arm], lw=1.1,
                   facecolors=PAL[arm] if mk_fill else "white", edgecolors=PAL[arm], zorder=3)
ax.set_xlabel("mean Q1Q2 (model state)"); ax.set_ylabel("SD of Q1Q2 over 96 personas")
ax.set_title("SD vs mean (iters 1..N)", fontsize=10)
ax.text(0.03, 0.05, "filled = gpt-4o-mini\nopen = claude-haiku-4-5", transform=ax.transAxes, fontsize=7,
        va="bottom", ha="left", color="#333333")
ax.grid(True, alpha=0.35)
fig.suptitle("Across-persona SD of the training reward (Q1Q2) — the ICLR 'K=5 is more stable' claim",
             fontsize=9.5, y=1.09)
C.save_fig(fig, f"{SCRIPT}_fig_sd")

# ═════════════════════════════════════════════════════════════════════════════
# 3. ICLR "shorter conversations": base -> final conv_len + the K contrast at the endpoints
# ═════════════════════════════════════════════════════════════════════════════
def _shape_row(method, metric, it):
    return SHAPE[(SHAPE["method"] == method) & (SHAPE["metric"] == metric) & (SHAPE["iteration"] == it)].iloc[0]


ep = []
for method, it in (("PTO", 10), ("GRPO", 5)):
    for metric in ["conv_len", "n_th_turns", "mean_turn_len"]:
        r = _shape_row(method, metric, it)
        ep.append({"contrast": f"{method} iter {it}", "method": method, "iteration": it, "metric": metric,
                   "mean_K0": r["mean_K0"], "mean_K5": r["mean_K5"], "K5_minus_K0": -r["mean_delta"],
                   "mean_delta_K0_minus_K5": r["mean_delta"], "dz": r["dz"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                   "p": r["p"], "p_holm": r["p_holm"], "n": r["n"]})
        L.put(f"length_endpoint.{method}.iter{it}.{metric}", ep[-1],
              source=f"tables/{SCRIPT}_length_endpoints.md row contrast='{method} iter {it}' metric={metric}")
EP = pd.DataFrame(ep)
LEV_OUT = LEVELS[["arm", "final_iteration"] + [f"{m}_{s}" for m in ["conv_len", "n_th_turns", "mean_turn_len"]
                                                for s in ["base", "final", "change"]]]
C.save_table(LEV_OUT, f"{SCRIPT}_length_endpoints", caption=(
    "**Base -> final session length per arm (arm means over 96 personas; base = the arm's own iteration-0 "
    "draw).** conv_len in utterances, n_th_turns in therapist turns, mean_turn_len in characters per therapist "
    f"turn; change = final - base. {CENSOR} The K contrast at the endpoints (PTO iter 10, GRPO iter 5) is in "
    f"`{SCRIPT}_length_kcontrast.md`."))
C.save_table(EP, f"{SCRIPT}_length_kcontrast", caption=(
    "**The ICLR 'K=5 gives shorter conversations' claim at the endpoints: persona-paired K contrast on session "
    "length at PTO iteration 10 and GRPO iteration 5 (the last matched GRPO iteration; GRPO_LA5 is "
    "right-censored there).** K5_minus_K0 is the K=5 arm's mean minus the K=0 arm's mean (positive = K=5 "
    f"LONGER); mean_delta_K0_minus_K5 keeps the paper's convention ({SIGN}). dz / bootstrap 95% CI / Wilcoxon "
    f"p on the paired deltas; p_holm within (method, metric) across iterations. {PAIR} Judge-free."))

# cross-check against the tracked 7_stats workbook (k_paired_channels sheet)
xlsx = C.RESULTS / "L5" / "tables" / "7_stats" / "gpt-4o-mini" / "7_stats.xlsx"
if xlsx.exists():
    kc = pd.read_excel(xlsx, sheet_name="k_paired_channels")
    kc = kc[kc["family"].astype(str).str.contains("session", case=False)]
    for method, it, metric in (("PTO", 10, "conv_len"), ("GRPO", 5, "conv_len"), ("PTO", 10, "mean_turn_len"),
                               ("GRPO", 5, "mean_turn_len")):
        t = kc[(kc["method"] == method) & (kc["iteration"] == it) & (kc["metric"] == metric)].iloc[0]
        r = _shape_row(method, metric, it)
        print(f"[check] {method} iter {it} {metric}: mine delta={r['mean_delta']:+.4f} dz={r['dz']:+.4f} | "
              f"tracked delta={t['mean_delta']:+.4f} dz={t['dz']:+.4f}")
        assert abs(r["mean_delta"] - t["mean_delta"]) < 1e-3 and abs(r["dz"] - t["dz"]) < 2e-3, (method, it, metric)
    L.put("_crosscheck.k_paired_channels_sheet", "conv_len + mean_turn_len at PTO iter 10 / GRPO iter 5 agree "
          "with results/L5/tables/7_stats/gpt-4o-mini/7_stats.xlsx sheet k_paired_channels to 1e-3",
          source="7_stats.xlsx sheet k_paired_channels")
else:
    print("[check] 7_stats.xlsx not found — skipped the k_paired_channels cross-check")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Selection-level length push + praise weights (tracked 6_preference tables; NO recompute)
# ═════════════════════════════════════════════════════════════════════════════
PREF = {v: C.RESULTS / v / "tables" / "6_preference" / "gpt-4o-mini" for v in ("L0", "L5")}
push = pd.concat([read_md_table(PREF[v] / "update_lexical_push.md").assign(view=v) for v in ("L0", "L5")],
                 ignore_index=True)
pool = pd.concat([read_md_table(PREF[v] / "generation_pool_means.md").assign(view=v) for v in ("L0", "L5")],
                 ignore_index=True)
SEL = push.merge(pool[["arm", "train_iter", "n_candidates", "pool_len", "pool_len_se", "pool_overpraise",
                       "pool_overpraise_se", "pool_affirm", "pool_affirm_se", "pool_question"]],
                 on=["arm", "train_iter"], how="left", suffixes=("", "_pool"))
SEL["w_len_over_se"] = SEL["w_len"] / SEL["w_len_se"]
SEL["policy_iteration"] = SEL["train_iter"] - 1
SEL = SEL[["arm", "method", "K", "train_iter", "policy_iteration", "n_groups", "n_candidates_pool",
           "w_len", "w_len_se", "w_len_over_se", "w_overpraise", "w_overpraise_se", "w_affirm", "w_affirm_se",
           "w_question", "w_question_se", "pool_len", "pool_len_se", "pool_overpraise", "pool_affirm",
           "pool_question"]].rename(columns={"n_candidates_pool": "n_candidates"})
SEL = SEL.sort_values(["method", "K", "train_iter"], ascending=[False, True, True]).reset_index(drop=True)
C.save_table(SEL, f"{SCRIPT}_selection", caption=(
    "**Selection-level lexical push and generation-pool means per arm x training iteration (copied, not "
    "recomputed, from the tracked EDA: `Exp3_PTO_GRPO/eda/results/{L0,L5}/tables/6_preference/gpt-4o-mini/"
    "update_lexical_push.md` and `generation_pool_means.md`; the LA0 arms come from the L0 view, the LA5 arms "
    "from L5).** w_<feature> = the lexical contrast the update pushes for, Sum(w * feature) per group +/- SE "
    "over groups, on a shared scale for both methods (DPO's +/-1 pair; GRPO's standardized advantages rescaled "
    "to match); 0 = the update is indifferent to that feature; w_len in characters, w_question in '?' per "
    "completion, w_affirm / w_overpraise in marker-rate units. pool_<feature> = the mean of that feature over "
    "ALL candidates the policy generated (what it GENERATES vs what the update SELECTS for). train_iter n "
    "samples from the iter-start policy, i.e. the eval set's model_iter_{n-1} (= policy_iteration). "
    "w_len_over_se = w_len / SE (a z-like ratio). Primary training oracle by construction (generations.jsonl "
    f"records the training oracle's own selection); no persona pairing (group-level). {CENSOR}"))

pto5 = SEL[SEL["arm"] == "PTO_LA5"].sort_values("train_iter")
pto0 = SEL[SEL["arm"] == "PTO_LA0"].sort_values("train_iter")
grpo0 = SEL[SEL["arm"] == "GRPO_LA0"].sort_values("train_iter")
grpo5 = SEL[SEL["arm"] == "GRPO_LA5"].sort_values("train_iter")
L.put("selection.PTO_LA5.w_len_series", {"train_iter": pto5["train_iter"].tolist(), "w_len": pto5["w_len"].tolist(),
                                          "w_len_se": pto5["w_len_se"].tolist(),
                                          "n_positive": int((pto5["w_len"] > 0).sum()), "n_iters": int(len(pto5)),
                                          "min_w_len_over_se": float(pto5["w_len_over_se"].min())},
      source=f"tables/{SCRIPT}_selection.md rows arm=PTO_LA5 (from L5 update_lexical_push.md)")
L.put("selection.PTO_LA0.max_abs_w_len_over_se", {"max_abs_w_len_over_se": float(pto0["w_len_over_se"].abs().max()),
                                                   "at_train_iter": int(pto0.loc[pto0["w_len_over_se"].abs().idxmax(), "train_iter"]),
                                                   "w_len_at": float(pto0.loc[pto0["w_len_over_se"].abs().idxmax(), "w_len"]),
                                                   "w_len_series": pto0["w_len"].tolist(), "w_len_se_series": pto0["w_len_se"].tolist()},
      source=f"tables/{SCRIPT}_selection.md rows arm=PTO_LA0 (from L0 update_lexical_push.md)")
for arm, d in (("PTO_LA0", pto0), ("PTO_LA5", pto5), ("GRPO_LA0", grpo0), ("GRPO_LA5", grpo5)):
    L.put(f"selection.{arm}.pool_len_endpoints", {"train_iter_first": int(d["train_iter"].iloc[0]),
                                                  "pool_len_first": float(d["pool_len"].iloc[0]),
                                                  "train_iter_last": int(d["train_iter"].iloc[-1]),
                                                  "pool_len_last": float(d["pool_len"].iloc[-1]),
                                                  "pool_overpraise_last": float(d["pool_overpraise"].iloc[-1]),
                                                  "pool_affirm_last": float(d["pool_affirm"].iloc[-1])},
          source=f"tables/{SCRIPT}_selection.md rows arm={arm} (from generation_pool_means.md)")
    for _, r in d.iterrows():
        L.put(f"selection.{arm}.train_iter{int(r.train_iter)}",
              {k: r[k] for k in ["w_len", "w_len_se", "w_overpraise", "w_overpraise_se", "w_affirm", "w_affirm_se",
                                 "w_question", "w_question_se", "pool_len", "pool_overpraise", "pool_affirm", "n_groups"]},
              source=f"tables/{SCRIPT}_selection.md row arm={arm} train_iter={int(r.train_iter)}")
L.put("selection.pool_len_contrast", {
    "PTO_iter10_LA5_vs_LA0": [float(pto5["pool_len"].iloc[-1]), float(pto0["pool_len"].iloc[-1])],
    "GRPO_iter5_LA5_vs_LA0": [float(grpo5.loc[grpo5["train_iter"] == 5, "pool_len"].iloc[0]),
                              float(grpo0.loc[grpo0["train_iter"] == 5, "pool_len"].iloc[0])]},
    source=f"tables/{SCRIPT}_selection.md", note="pool_len at PTO train_iter 10 (K5 vs K0) and GRPO train_iter 5 (K5 vs K0)")

# ═════════════════════════════════════════════════════════════════════════════
# 5. Console digest + ledger
# ═════════════════════════════════════════════════════════════════════════════
print("\n== conv_len K contrast (K0 - K5) at the endpoints ==")
print(EP[["contrast", "metric", "mean_K0", "mean_K5", "K5_minus_K0", "dz", "p_holm"]].to_string(index=False))
print("\n== SD of Q1Q2, both graders, at each arm's last iteration ==")
last = SD[(SD["metric"] == "Q1Q2")].sort_values("iteration").groupby(["judge", "arm"]).tail(1)
print(last[["judge", "arm", "iteration", "mean", "sd", "iqr", "share_ge45"]].to_string(index=False))
print("\n== lowest-SD summary ==")
print(SUMM[["judge", "metric", "min_sd_arm", "min_sd_iteration", "min_sd", "min_sd_mean", "spearman_mean_vs_sd"]].to_string(index=False))
print("\n== Brown-Forsythe / Pitman-Morgan Q1Q2, Holm-significant rows ==")
print(BF[(BF["metric"] == "Q1Q2") & ((BF["bf_p_holm"] < 0.05) | (BF["pm_p_holm"] < 0.05))]
      [["judge", "method", "iteration", "sd_K0", "sd_K5", "bf_p_holm", "pm_r", "pm_p_holm"]].to_string(index=False))
p = L.save()
print(f"\n[ledger] {p}")
