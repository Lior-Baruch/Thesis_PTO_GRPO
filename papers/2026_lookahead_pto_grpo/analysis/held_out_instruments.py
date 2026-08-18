"""held_out_instruments.py — K-turn look-ahead effects on instruments OUTSIDE the training reward.

Four questions, each read under BOTH graders (training oracle gpt-4o-mini; held-out Claude Haiku 4.5),
paired on ``persona_id`` (never ``file_index``), K-contrast sign ``+ => K=0 higher``:

1. **WAI-SR subscale composition** — Task (items 1,2,10,12) / Goal (4,6,8,11) / Bond (3,5,7,9),
   the WAI-SR standard AND the package's ``WAI_{Goal,Task,Bond}_Mean`` columns (verified identical
   here). Per arm x iteration levels + gain over own base; persona-paired K0-K5 contrast on the
   *bond excess* = Bond - mean(Goal, Task) at every matched iteration.
2. **PCT (patient change talk)** — the lake's ``PCT`` metric is ``PCT_ChangeProp``; the components
   (globals + utterance counts) come from ``behavior.load_pct_behavior``. Paired K0-K5 by iteration.
3. **Q2 item profile** — per-item endpoint gain over own base for every arm + the per-item K0-K5
   contrast at the matched endpoint (PTO iter 10, GRPO iter 5). Items 1/2/3/10 = the
   "self-disclosure" face-content group; 3 and 10 = emotional self-disclosure.
4. **Heterogeneity** — the K0-K5 contrast on Q1Q2 / MICI / PCT WITHIN cooperation level
   (High -> Cooperative, StartLowAndChangesToHigh -> Warms up, Low -> Resistant; 32 personas each)
   at the matched endpoint and at each arm's own-oracle best iteration.

Outputs (all prefixed ``held_out_instruments_``): tables ``wai``, ``wai_kcontrast``, ``fig_wai_data``,
``pct``, ``q2items``, ``q2items_long``, ``q2items_kcontrast``, ``hetero``; figures ``fig_wai``, ``fig_hetero``; ledger
``out/held_out_instruments.json``.

Run:  .venv/Scripts/python.exe papers/2026_lookahead_pto_grpo/analysis/held_out_instruments.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import eda_analysis  # noqa: E402
from eda_analysis import data as D, behavior as B  # noqa: E402
from eda_analysis.constants import (set_active_judge, Q2_ITEM_SHORT, Q2_ITEM_GROUP_OF)  # noqa: E402

SCRIPT = "held_out_instruments"
L = C.Ledger(SCRIPT)
JUDGE_KEYS = ["primary", "heldout"]
JNAME = {"primary": C.JUDGE_SHORT[C.PRIMARY], "heldout": C.JUDGE_SHORT[C.HELDOUT]}
JTAG = {"primary": "", "heldout": C.HELDOUT}
SIGN = "K-contrast sign: + => K=0 higher (K0 - K5)."
PAIR = "Paired on persona_id (the recovered patient persona), never file_index."
CENSOR = "GRPO_LA5 is right-censored at iteration 5 (PTO arms and GRPO_LA0 run to 10)."

# WAI-SR standard subscale map (Hatcher & Gillaspy 2006) — identical to code/questionnaires.py
# ``WAI_Goal / WAI_Task / WAI_Bond`` (the columns the package's wai_subscales figure reads).
WAI_SUBSCALES = {"Task": [1, 2, 10, 12], "Goal": [4, 6, 8, 11], "Bond": [3, 5, 7, 9]}
COOP_LABEL = {"High": "Cooperative", "StartLowAndChangesToHigh": "Warms up", "Low": "Resistant"}
COOP_ORDER = ["Cooperative", "Warms up", "Resistant"]
PCT_METRICS = ["PCT_ChangeProp", "PCT_GlobalMean", "PCT_Importance", "PCT_Confidence",
               "PCT_Readiness", "PCT_ChangeTalk", "PCT_SustainTalk", "PCT_Neutral",
               "PCT_BehaviorTotal"]
PCT_LABEL = {"PCT_ChangeProp": "ChangeProp = CT/(CT+ST) [= lake 'PCT']",
             "PCT_GlobalMean": "GlobalMean (Importance/Confidence/Readiness, 1-5)",
             "PCT_Importance": "Importance (1-5)", "PCT_Confidence": "Confidence (1-5)",
             "PCT_Readiness": "Readiness (1-5)", "PCT_ChangeTalk": "change-talk utterances (count)",
             "PCT_SustainTalk": "sustain-talk utterances (count)",
             "PCT_Neutral": "neutral utterances (count)",
             "PCT_BehaviorTotal": "patient utterances (count)"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _attach(df: pd.DataFrame, arms) -> pd.DataFrame:
    """persona_id + characteristics per arm (each arm's own seed; all 42 today)."""
    seed_by_arm = {a.label: a.seed for a in arms}
    parts = [D.attach_personas(g, seed_by_arm.get(lab, 42))
             for lab, g in df.groupby("arm", sort=False)]
    return pd.concat(parts, ignore_index=True)


def _pair(df: pd.DataFrame, value: str, model_a: str, model_b: str, *, key="persona_id") -> dict:
    """persona-aligned paired contrast model_a - model_b on ``value`` (C.paired conventions)."""
    a = df[df["model"] == model_a][[key, value]].dropna().groupby(key)[value].mean()
    b = df[df["model"] == model_b][[key, value]].dropna().groupby(key)[value].mean()
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    out = C.paired(m["a"].to_numpy(), m["b"].to_numpy())
    out["mean_a"] = float(m["a"].mean()) if len(m) else np.nan
    out["mean_b"] = float(m["b"].mean()) if len(m) else np.nan
    return out


def _model(method: str, K: int, it: int) -> str:
    return f"{method}Exp3_LA{K}_{'Base' if it == 0 else f'I{it}'}"


def _iters_both(df: pd.DataFrame, method: str) -> list:
    i0 = set(df.loc[df["arm"] == f"{method}_LA0", "iteration"])
    i5 = set(df.loc[df["arm"] == f"{method}_LA5", "iteration"])
    return sorted(int(i) for i in (i0 & i5))


def _k_contrast_by_iter(df: pd.DataFrame, value: str, method: str) -> pd.DataFrame:
    """K0 - K5 at every matched iteration (iteration 0 = two independent base draws)."""
    rows = []
    for it in _iters_both(df, method):
        r = _pair(df, value, _model(method, 0, it), _model(method, 5, it))
        rows.append({"method": method, "iteration": it, "mean_K0": r["mean_a"],
                     "mean_K5": r["mean_b"], "n": r["n"], "mean_delta": r["mean_delta"],
                     "dz": r["dz"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = C.holm(out["p"].to_numpy())   # family = iterations within (judge, method, metric)
    return out


def _endpoints(df: pd.DataFrame) -> dict:
    return {arm: int(df.loc[df["arm"] == arm, "iteration"].max()) for arm in C.ARMS
            if (df["arm"] == arm).any()}


def _fmt3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD (primary first, then held-out; never interleave)
# ═══════════════════════════════════════════════════════════════════════════════
print("[held_out_instruments] loading scores (both graders) ...")
SC = C.load_scores_both()                       # {'primary','heldout'} scores_long, persona attached
ARMS = eda_analysis.cross_k_arms(C.EdaConfig(view="L5", verbose=False))
ARM_LABELS = [a.label for a in ARMS]
assert set(ARM_LABELS) == set(C.ARMS), ARM_LABELS

ITEMS = {}   # judge -> {"wai": items_long, "wai_sub": subscales_long, "q2": items_long, "pct": pct_per_conv}
for jk in JUDGE_KEYS:
    set_active_judge(JTAG[jk], 0)
    print(f"[held_out_instruments] loading item frames under {JNAME[jk]} ...")
    wai = _attach(D.load_items("WAI-SR", ARMS), ARMS)
    sub = _attach(D.load_subscales(ARMS), ARMS)
    sub = sub[sub["parent"] == "WAI-SR"].copy()
    q2 = _attach(D.load_items("Q2", ARMS), ARMS)
    pct = B.load_pct_behavior(ARMS, attach_persona=True)
    ITEMS[jk] = {"wai": wai, "wai_sub": sub, "q2": q2, "pct": pct}
set_active_judge("", 0)

END = _endpoints(SC["primary"])                 # {'PTO_LA0': 10, 'PTO_LA5': 10, 'GRPO_LA0': 10, 'GRPO_LA5': 5}
assert END == _endpoints(SC["heldout"]), (END, _endpoints(SC["heldout"]))
MATCHED_END = {"PTO": min(END["PTO_LA0"], END["PTO_LA5"]), "GRPO": min(END["GRPO_LA0"], END["GRPO_LA5"])}
# own-oracle best iteration per arm — selected on the TRAINING ORACLE's Q1Q2 (the primary grader),
# and reused for the held-out grader so both graders judge the same checkpoints.
BEST = D.best_iteration_by_arm(SC["primary"])
L.put("endpoints", END, source="data: max scored iteration per arm (both graders agree)")
L.put("best_iteration_by_arm", BEST, source="eda_analysis.data.best_iteration_by_arm on primary Q1Q2")
L.put("matched_endpoint", MATCHED_END, source="min(END[LA0], END[LA5]) per method")
print("endpoints", END, "| best", BEST)

# check the item-derived vs lake WAI subscale means agree (both graders)
for jk in JUDGE_KEYS:
    it = ITEMS[jk]["wai"]
    keys = ["arm", "model", "iteration", "file_index", "persona_id"]
    piv = it.pivot_table(index=keys, columns="item", values="score").reset_index()
    for name, ids in WAI_SUBSCALES.items():
        piv[name] = piv[[i for i in ids]].mean(axis=1)
    piv["bond_excess"] = piv["Bond"] - (piv["Goal"] + piv["Task"]) / 2
    piv["WAI_total_items"] = piv[list(range(1, 13))].mean(axis=1)
    ITEMS[jk]["wai_conv"] = piv
    lake = ITEMS[jk]["wai_sub"].pivot_table(index=["arm", "model", "file_index"],
                                            columns="subscale", values="score").reset_index()
    m = piv.merge(lake, on=["arm", "model", "file_index"], suffixes=("", "_lake"))
    maxdiff = max(float((m[s] - m[f"{s}_lake"]).abs().max()) for s in WAI_SUBSCALES)
    L.put(f"wai.subscale_map_check.{jk}", {"n_convs": int(len(m)), "max_abs_diff_items_vs_lake": maxdiff},
          source="in-script check: WAI-SR standard map (Task 1,2,10,12 / Goal 4,6,8,11 / Bond 3,5,7,9) "
                 "vs eval_scores WAI_{Task,Goal,Bond}_Mean (code/questionnaires.py map)")
    print(f"  WAI subscale map check [{JNAME[jk]}]: {len(m)} convs, max |diff| = {maxdiff:.2e}")
    assert maxdiff < 1e-9, "WAI-SR standard map disagrees with the package's subscale columns"

C.style()
PAL = C.palette()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. WAI-SR SUBSCALES
# ═══════════════════════════════════════════════════════════════════════════════
wai_rows, waik_rows = [], []
for jk in JUDGE_KEYS:
    piv = ITEMS[jk]["wai_conv"]
    for arm in C.ARMS:
        g = piv[piv["arm"] == arm]
        base = g[g["iteration"] == 0].set_index("persona_id")
        for it in sorted(g["iteration"].unique()):
            gi = g[g["iteration"] == it]
            row = {"judge": JNAME[jk], "arm": arm, "iteration": int(it), "n": int(len(gi))}
            for s in ["Task", "Goal", "Bond", "bond_excess", "WAI_total_items"]:
                row[f"{s}"] = float(gi[s].mean())
                # gain over OWN base, persona-paired (iteration 0 gain = 0 by construction)
                mm = gi.set_index("persona_id")[s].to_frame("a").join(base[s].rename("b"), how="inner")
                row[f"{s}_gain"] = float((mm["a"] - mm["b"]).mean())
            wai_rows.append(row)
    for method in ["PTO", "GRPO"]:
        for val in ["bond_excess", "Bond", "Goal", "Task", "WAI_total_items"]:
            t = _k_contrast_by_iter(piv, val, method)
            t.insert(0, "judge", JNAME[jk]); t.insert(2, "measure", val)
            waik_rows.append(t)
WAI = pd.DataFrame(wai_rows).rename(columns={"WAI_total_items": "WAI_total", "WAI_total_items_gain": "WAI_total_gain"})
WAIK = pd.concat(waik_rows, ignore_index=True)
WAIK["measure"] = WAIK["measure"].replace({"WAI_total_items": "WAI_total"})

C.save_table(WAI, f"{SCRIPT}_wai", caption=(
    "**WAI-SR subscale levels + gain over own base**, per grader x arm x iteration. Subscales use the WAI-SR "
    "standard map (Task = items 1,2,10,12; Goal = 4,6,8,11; Bond = 3,5,7,9), which is identical to the score "
    "lake's WAI_{Task,Goal,Bond}_Mean columns that the EDA's wai_subscales figure reads (verified in-script, "
    "max |diff| < 1e-9). `bond_excess` = Bond - mean(Goal, Task) per conversation, then averaged. `*_gain` = "
    "mean persona-paired difference vs the arm's OWN iteration-0 base (1-5 Likert points). Iteration 0 rows are "
    "the base draws (gain 0 by construction). n = conversations (personas). " + CENSOR + " Rows for both "
    "graders (judge column); never average them."))
C.save_table(WAIK, f"{SCRIPT}_wai_kcontrast", caption=(
    "**Persona-paired K0-K5 contrast on WAI-SR subscales**, per grader x method x measure x matched iteration. "
    + SIGN + " " + PAIR + " `bond_excess` = Bond - mean(Goal,Task) — a positive delta means the K=0 arm's "
    "alliance gain is MORE bond-weighted (relational) relative to its task/goal component than the K=5 arm's. "
    "mean_K0/mean_K5 = arm means on the paired personas; dz = mean/sd of paired deltas; 95% CI = percentile "
    "bootstrap (2000 draws); p = Wilcoxon signed-rank; p_holm = Holm within (judge, method, measure) across "
    "iterations. Iteration 0 = two independent base draws (noise floor). " + CENSOR + " Graders side by side, "
    "never averaged."))

# ledger: endpoint numbers per arm + matched-endpoint contrasts
for jk in JUDGE_KEYS:
    for arm in C.ARMS:
        r = WAI[(WAI["judge"] == JNAME[jk]) & (WAI["arm"] == arm) & (WAI["iteration"] == END[arm])].iloc[0]
        L.put(f"wai.endpoint.{jk}.{arm}", {c: _fmt3(r[c]) for c in
              ["Task", "Goal", "Bond", "bond_excess", "WAI_total", "Task_gain", "Goal_gain", "Bond_gain",
               "bond_excess_gain", "WAI_total_gain"]} | {"iteration": int(r["iteration"]), "n": int(r["n"])},
              source=f"tables/{SCRIPT}_wai.md row judge={JNAME[jk]} arm={arm} iteration={END[arm]}")
    for method in ["PTO", "GRPO"]:
        for val in ["bond_excess", "Bond", "Goal", "Task", "WAI_total"]:
            t = WAIK[(WAIK["judge"] == JNAME[jk]) & (WAIK["method"] == method) & (WAIK["measure"] == val)]
            r = t[t["iteration"] == MATCHED_END[method]].iloc[0]
            L.put(f"wai.kcontrast.{jk}.{method}.{val}.iter{MATCHED_END[method]}",
                  {c: _fmt3(r[c]) for c in ["mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]}
                  | {"n": int(r["n"])},
                  source=f"tables/{SCRIPT}_wai_kcontrast.md row judge={JNAME[jk]} method={method} "
                         f"measure={val} iteration={MATCHED_END[method]}")
            sig = t[(t["iteration"] > 0) & (t["p_holm"] < 0.05)]
            L.put(f"wai.kcontrast.{jk}.{method}.{val}.n_iters_holm_sig",
                  {"n_sig": int(len(sig)), "n_iters": int((t["iteration"] > 0).sum()),
                   "iters_sig": [int(i) for i in sig["iteration"]],
                   "sign_of_sig": [("K0>K5" if d > 0 else "K5>K0") for d in sig["mean_delta"]]},
                  source=f"tables/{SCRIPT}_wai_kcontrast.md (judge={JNAME[jk]}, method={method}, measure={val})")

# figure: endpoint gain over own base by subscale, 4 arms (+ GRPO_LA0 @ the matched iter 5), two grader panels
def _gain_ci(piv: pd.DataFrame, arm: str, it: int, s: str) -> dict:
    g = piv[piv["arm"] == arm]
    return _pair(g, s, _model(C.method_of(arm), C.k_of(arm), it), _model(C.method_of(arm), C.k_of(arm), 0))

series = [("PTO_LA0", END["PTO_LA0"], PAL["PTO_LA0"], 1.0, ""),
          ("PTO_LA5", END["PTO_LA5"], PAL["PTO_LA5"], 1.0, "//"),
          ("GRPO_LA0", END["GRPO_LA0"], PAL["GRPO_LA0"], 1.0, ""),
          ("GRPO_LA0", MATCHED_END["GRPO"], PAL["GRPO_LA0"], 0.45, ""),
          ("GRPO_LA5", END["GRPO_LA5"], PAL["GRPO_LA5"], 1.0, "//")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
subs = ["Task", "Goal", "Bond"]
x = np.arange(len(subs)); w = 0.16
fig_rows = []
for ax, jk in zip(axes, JUDGE_KEYS):
    piv = ITEMS[jk]["wai_conv"]
    for si, (arm, it, col, alpha, hatch) in enumerate(series):
        vals, los, his = [], [], []
        for s in subs:
            r = _gain_ci(piv, arm, it, s)
            vals.append(r["mean_delta"]); los.append(r["mean_delta"] - r["ci_lo"]); his.append(r["ci_hi"] - r["mean_delta"])
            fig_rows.append({"judge": JNAME[jk], "arm": arm, "iteration": it, "subscale": s,
                             "gain": r["mean_delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"]})
        ax.bar(x + (si - 2) * w, vals, w, color=col, alpha=alpha, hatch=hatch, edgecolor="white",
               linewidth=0.6, yerr=[los, his], error_kw=dict(elinewidth=1.0, capsize=2, ecolor="#333333"),
               label=f"{arm} @ iter {it}")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(subs)
    ax.set_title(f"grader: {JNAME[jk]}", fontsize=10)
    ax.set_xlabel("WAI-SR subscale")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("gain over own base\n(WAI-SR points, 1–5 scale)")
for ax in axes:
    ax.margins(y=0.08)
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, fontsize=7.5, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.10), handlelength=1.6, columnspacing=1.0)
fig.suptitle("WAI-SR subscale gain at the endpoint (persona-paired vs own base, 95% bootstrap CI)",
             fontsize=10, y=1.0)
C.save_fig(fig, f"{SCRIPT}_fig_wai")
FIGWAI = pd.DataFrame(fig_rows)
C.save_table(FIGWAI, f"{SCRIPT}_fig_wai_data", caption=(
    "Data behind fig_wai: persona-paired gain over own base per WAI-SR subscale at each arm's endpoint "
    "(GRPO_LA0 also at iteration 5, the iteration matched to the right-censored GRPO_LA5), both graders. "
    "95% percentile-bootstrap CI over paired deltas. " + PAIR))
for _, r in FIGWAI.iterrows():
    L.put(f"wai.fig_gain.{r.judge}.{r.arm}.iter{int(r.iteration)}.{r.subscale}",
          {"gain": _fmt3(r.gain), "ci_lo": _fmt3(r.ci_lo), "ci_hi": _fmt3(r.ci_hi), "n": int(r.n)},
          source=f"tables/{SCRIPT}_fig_wai_data.md / figures/{SCRIPT}_fig_wai.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PCT — paired K0-K5 by iteration on the lake metric + components
# ═══════════════════════════════════════════════════════════════════════════════
pct_rows = []
for jk in JUDGE_KEYS:
    pct = ITEMS[jk]["pct"]
    for method in ["PTO", "GRPO"]:
        for m in PCT_METRICS:
            if m not in pct.columns or pct[m].notna().sum() == 0:
                continue
            t = _k_contrast_by_iter(pct, m, method)
            t.insert(0, "judge", JNAME[jk]); t.insert(2, "metric", m)
            pct_rows.append(t)
PCT = pd.concat(pct_rows, ignore_index=True)
C.save_table(PCT, f"{SCRIPT}_pct", caption=(
    "**Persona-paired K0-K5 contrast on PCT (patient change talk) and its components**, per grader x method x "
    "metric x matched iteration. `PCT_ChangeProp` = CT/(CT+ST) is the score lake's `PCT` metric (higher = more "
    "change talk); `PCT_GlobalMean` = mean of the three 1-5 patient globals; the three utterance counts sum to "
    "`PCT_BehaviorTotal` (patient utterances coded). " + SIGN + " " + PAIR + " mean_K0/mean_K5 = arm means on "
    "the paired personas; dz = mean/sd of paired deltas; 95% percentile-bootstrap CI; p = Wilcoxon signed-rank; "
    "p_holm = Holm within (judge, method, metric) across iterations. Iteration 0 = two independent base draws "
    "(noise floor). " + CENSOR + " Graders side by side, never averaged. Source: behavior.load_pct_behavior "
    "under each grader's partition of the score lake."))
for jk in JUDGE_KEYS:
    for method in ["PTO", "GRPO"]:
        for m in PCT_METRICS:
            t = PCT[(PCT["judge"] == JNAME[jk]) & (PCT["method"] == method) & (PCT["metric"] == m)]
            if t.empty:
                continue
            r = t[t["iteration"] == MATCHED_END[method]].iloc[0]
            L.put(f"pct.kcontrast.{jk}.{method}.{m}.iter{MATCHED_END[method]}",
                  {c: _fmt3(r[c]) for c in ["mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]}
                  | {"n": int(r["n"])},
                  source=f"tables/{SCRIPT}_pct.md row judge={JNAME[jk]} method={method} metric={m} "
                         f"iteration={MATCHED_END[method]}")
            sig = t[(t["iteration"] > 0) & (t["p_holm"] < 0.05)]
            L.put(f"pct.kcontrast.{jk}.{method}.{m}.holm_sig_iters",
                  {"iters_sig": [int(i) for i in sig["iteration"]],
                   "sign_of_sig": [("K0>K5" if d > 0 else "K5>K0") for d in sig["mean_delta"]],
                   "n_iters": int((t["iteration"] > 0).sum())},
                  source=f"tables/{SCRIPT}_pct.md (judge={JNAME[jk]}, method={method}, metric={m})")
            # the largest |dz| iteration too
            tt = t[t["iteration"] > 0]
            if not tt.empty:
                rr = tt.loc[tt["dz"].abs().idxmax()]
                L.put(f"pct.kcontrast.{jk}.{method}.{m}.max_abs_dz",
                      {"iteration": int(rr["iteration"]), "mean_delta": _fmt3(rr["mean_delta"]),
                       "dz": _fmt3(rr["dz"]), "p": _fmt3(rr["p"]), "p_holm": _fmt3(rr["p_holm"])},
                      source=f"tables/{SCRIPT}_pct.md (judge={JNAME[jk]}, method={method}, metric={m})")

# cross-check vs the tracked EDA (results/L5/tables/7_stats/gpt-4o-mini/k_paired_by_method.md):
# PCT (=ChangeProp) primary: PTO iter 6 mean_delta 0.006 dz 0.037; GRPO iter 4 -0.062 dz -0.321.
_pto6 = PCT[(PCT.judge == JNAME["primary"]) & (PCT.method == "PTO") & (PCT.metric == "PCT_ChangeProp") & (PCT.iteration == 6)].iloc[0]
_grpo4 = PCT[(PCT.judge == JNAME["primary"]) & (PCT.method == "GRPO") & (PCT.metric == "PCT_ChangeProp") & (PCT.iteration == 4)].iloc[0]
# and Q1Q2 from the scores frame: PTO iter 6 primary +0.257 dz 0.417
_q = SC["primary"]
_q1q2 = _pair(_q[_q["questionnaire"] == "Q1Q2"], "score", _model("PTO", 0, 6), _model("PTO", 5, 6))
L.put("crosscheck.tracked_k_paired_by_method", {
    "pct_pto_iter6": {"mine": {"mean_delta": _fmt3(_pto6.mean_delta), "dz": _fmt3(_pto6.dz)}, "tracked": {"mean_delta": 0.006, "dz": 0.037}},
    "pct_grpo_iter4": {"mine": {"mean_delta": _fmt3(_grpo4.mean_delta), "dz": _fmt3(_grpo4.dz)}, "tracked": {"mean_delta": -0.062, "dz": -0.321}},
    "q1q2_pto_iter6": {"mine": {"mean_delta": _fmt3(_q1q2["mean_delta"]), "dz": _fmt3(_q1q2["dz"])}, "tracked": {"mean_delta": 0.257, "dz": 0.417}},
}, source="Exp3_PTO_GRPO/eda/results/L5/tables/7_stats/gpt-4o-mini/k_paired_by_method.md")
print("crosscheck PCT PTO@6", round(_pto6.mean_delta, 3), round(_pto6.dz, 3), "| GRPO@4", round(_grpo4.mean_delta, 3), round(_grpo4.dz, 3),
      "| Q1Q2 PTO@6", round(_q1q2["mean_delta"], 3), round(_q1q2["dz"], 3))
assert abs(_pto6.mean_delta - 0.006) < 0.0015 and abs(_grpo4.mean_delta + 0.062) < 0.0015 and abs(_q1q2["mean_delta"] - 0.257) < 0.0015

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Q2 ITEM PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
q2_long_rows = []
for jk in JUDGE_KEYS:
    q2 = ITEMS[jk]["q2"]
    for arm in C.ARMS:
        g = q2[q2["arm"] == arm]
        meth, K = C.method_of(arm), C.k_of(arm)
        for item in range(1, 18):
            gi = g[g["item"] == item]
            r = _pair(gi, "score", _model(meth, K, END[arm]), _model(meth, K, 0))
            q2_long_rows.append({"judge": JNAME[jk], "arm": arm, "item": item, "short": Q2_ITEM_SHORT[item],
                                 "group": Q2_ITEM_GROUP_OF[item], "target_iter": END[arm], "n": r["n"],
                                 "base": r["mean_b"], "target": r["mean_a"], "gain": r["mean_delta"],
                                 "gain_ci_lo": r["ci_lo"], "gain_ci_hi": r["ci_hi"], "gain_dz": r["dz"], "gain_p": r["p"]})
Q2L = pd.DataFrame(q2_long_rows)
# per-item K contrast at the matched endpoint (+ Holm across the 17 items within judge x method)
q2k_rows = []
for jk in JUDGE_KEYS:
    q2 = ITEMS[jk]["q2"]
    for method in ["PTO", "GRPO"]:
        it = MATCHED_END[method]
        rows = []
        for item in range(1, 18):
            gi = q2[q2["item"] == item]
            r = _pair(gi, "score", _model(method, 0, it), _model(method, 5, it))
            rows.append({"judge": JNAME[jk], "method": method, "iteration": it, "item": item,
                         "short": Q2_ITEM_SHORT[item], "group": Q2_ITEM_GROUP_OF[item], "n": r["n"],
                         "mean_K0": r["mean_a"], "mean_K5": r["mean_b"], "k_delta": r["mean_delta"],
                         "k_dz": r["dz"], "k_ci_lo": r["ci_lo"], "k_ci_hi": r["ci_hi"], "k_p": r["p"]})
        t = pd.DataFrame(rows); t["k_p_holm"] = C.holm(t["k_p"].to_numpy())
        q2k_rows.append(t)
Q2K = pd.concat(q2k_rows, ignore_index=True)

# wide, paper-facing table: one row per (judge, item)
wide_rows = []
for jk in JUDGE_KEYS:
    for item in range(1, 18):
        row = {"judge": JNAME[jk], "item": item, "short": Q2_ITEM_SHORT[item], "group": Q2_ITEM_GROUP_OF[item]}
        bases = []
        for arm in C.ARMS:
            r = Q2L[(Q2L.judge == JNAME[jk]) & (Q2L.arm == arm) & (Q2L.item == item)].iloc[0]
            bases.append(r["base"]); row[f"{arm}_gain@{END[arm]}"] = r["gain"]
        row["base_mean(4 arms)"] = float(np.mean(bases))
        for method in ["PTO", "GRPO"]:
            r = Q2K[(Q2K.judge == JNAME[jk]) & (Q2K.method == method) & (Q2K.item == item)].iloc[0]
            row[f"{method}_K0-K5@{MATCHED_END[method]}"] = r["k_delta"]
            row[f"{method}_dz"] = r["k_dz"]; row[f"{method}_p_holm"] = r["k_p_holm"]
        wide_rows.append(row)
Q2W = pd.DataFrame(wide_rows)
cols = ["judge", "item", "short", "group", "base_mean(4 arms)"] + [f"{a}_gain@{END[a]}" for a in C.ARMS] + \
       [c for m in ["PTO", "GRPO"] for c in (f"{m}_K0-K5@{MATCHED_END[m]}", f"{m}_dz", f"{m}_p_holm")]
Q2W = Q2W[cols]
C.save_table(Q2W, f"{SCRIPT}_q2items", caption=(
    "**Q2 item profile at the endpoint** (17 items of the Q2 relational-communication rubric, 1-5), per grader. "
    "`<arm>_gain@N` = persona-paired mean gain over the arm's OWN base at its final iteration N (PTO arms and "
    "GRPO_LA0 at 10; GRPO_LA5 right-censored at 5); `base_mean(4 arms)` = mean of the four arms' independent "
    "base draws (descriptive). `<method>_K0-K5@N` = persona-paired K0-K5 contrast at the matched endpoint "
    "(PTO iter 10, GRPO iter 5), " + SIGN + " dz = mean/sd of paired deltas; p_holm = Holm across the 17 items "
    "within (judge, method). Groups are the face-content reading of constants.Q2_ITEM_GROUPS (analytical, not "
    "a validated subscale); items 3 and 10 = emotional self-disclosure ('shared his feelings', 'said when "
    "happy/sad'), items 1,2,3,10 = the self-disclosure group. " + PAIR + " Graders side by side, never averaged."))
C.save_table(Q2L, f"{SCRIPT}_q2items_long", caption=(
    "Long companion of q2items: per (grader, arm, item) the persona-paired gain over the arm's own base at its "
    "endpoint (`target_iter`), with 95% percentile-bootstrap CI, dz and Wilcoxon p; `base`/`target` = arm means "
    "on the paired personas. " + PAIR + " " + CENSOR))
C.save_table(Q2K, f"{SCRIPT}_q2items_kcontrast", caption=(
    "Per-item persona-paired K0-K5 contrast on the 17 Q2 items at the matched endpoint (PTO iter 10, GRPO iter 5; "
    + CENSOR + ") per grader. " + SIGN + " " + PAIR + " k_p_holm = Holm across the 17 items within (judge, method)."))

for jk in JUDGE_KEYS:
    for item in [1, 2, 3, 10]:
        for arm in C.ARMS:
            r = Q2L[(Q2L.judge == JNAME[jk]) & (Q2L.arm == arm) & (Q2L.item == item)].iloc[0]
            L.put(f"q2.item{item}.gain.{jk}.{arm}", {"base": _fmt3(r.base), "target": _fmt3(r.target),
                  "gain": _fmt3(r.gain), "ci_lo": _fmt3(r.gain_ci_lo), "ci_hi": _fmt3(r.gain_ci_hi),
                  "dz": _fmt3(r.gain_dz), "target_iter": int(r.target_iter), "n": int(r.n)},
                  source=f"tables/{SCRIPT}_q2items_long.md row judge={JNAME[jk]} arm={arm} item={item}")
        for method in ["PTO", "GRPO"]:
            r = Q2K[(Q2K.judge == JNAME[jk]) & (Q2K.method == method) & (Q2K.item == item)].iloc[0]
            L.put(f"q2.item{item}.kcontrast.{jk}.{method}.iter{int(r.iteration)}",
                  {"mean_K0": _fmt3(r.mean_K0), "mean_K5": _fmt3(r.mean_K5), "k_delta": _fmt3(r.k_delta),
                   "dz": _fmt3(r.k_dz), "ci_lo": _fmt3(r.k_ci_lo), "ci_hi": _fmt3(r.k_ci_hi), "p": _fmt3(r.k_p),
                   "p_holm": _fmt3(r.k_p_holm), "n": int(r.n)},
                  source=f"tables/{SCRIPT}_q2items.md row judge={JNAME[jk]} item={item} ({method} columns)")
    # group-level summary: mean gain of self-disclosure group vs the other 13 items, per arm
    for arm in C.ARMS:
        t = Q2L[(Q2L.judge == JNAME[jk]) & (Q2L.arm == arm)]
        sd = t[t["item"].isin([1, 2, 3, 10])]["gain"].mean(); rest = t[~t["item"].isin([1, 2, 3, 10])]["gain"].mean()
        emo = t[t["item"].isin([3, 10])]["gain"].mean()
        rank = t.sort_values("gain", ascending=False)["item"].tolist()
        L.put(f"q2.groups.{jk}.{arm}", {"self_disclosure_1_2_3_10_mean_gain": _fmt3(sd),
              "emotional_3_10_mean_gain": _fmt3(emo), "other_13_items_mean_gain": _fmt3(rest),
              "all_17_mean_gain": _fmt3(t["gain"].mean()),
              "rank_of_item3": int(rank.index(3) + 1), "rank_of_item10": int(rank.index(10) + 1),
              "top3_items": rank[:3], "bottom3_items": rank[-3:]},
              source=f"tables/{SCRIPT}_q2items.md (judge={JNAME[jk]}, column {arm}_gain@{END[arm]})")
    for method in ["PTO", "GRPO"]:
        t = Q2K[(Q2K.judge == JNAME[jk]) & (Q2K.method == method)]
        sig = t[t["k_p_holm"] < 0.05]
        L.put(f"q2.kcontrast_summary.{jk}.{method}", {
            "iteration": int(t["iteration"].iloc[0]), "n_items_holm_sig": int(len(sig)),
            "items_sig": [int(i) for i in sig["item"]], "sign_of_sig": [("K0>K5" if d > 0 else "K5>K0") for d in sig["k_delta"]],
            "mean_k_delta_all17": _fmt3(t["k_delta"].mean()),
            "mean_k_delta_selfdisc": _fmt3(t[t["item"].isin([1, 2, 3, 10])]["k_delta"].mean()),
            "mean_k_delta_emotional_3_10": _fmt3(t[t["item"].isin([3, 10])]["k_delta"].mean()),
            "mean_k_delta_other13": _fmt3(t[~t["item"].isin([1, 2, 3, 10])]["k_delta"].mean())},
            source=f"tables/{SCRIPT}_q2items.md (judge={JNAME[jk]}, {method} columns)")

# cross-check vs tracked q2_item_deltas (unpaired base/target means -> identical gains when both cells are complete)
_chk = Q2L[(Q2L.judge == JNAME["primary"]) & (Q2L.arm == "PTO_LA5") & (Q2L.item == 3)].iloc[0]
L.put("crosscheck.tracked_q2_item_deltas.PTO_LA5_item3_final_primary",
      {"mine": {"base": _fmt3(_chk.base), "gain": _fmt3(_chk.gain)}, "tracked": {"base": 2.135, "delta": 1.135}},
      source="Exp3_PTO_GRPO/eda/results/L5/tables/2_questionnaires/gpt-4o-mini/q2_item_deltas.md")
assert abs(_chk.base - 2.135) < 0.0015 and abs(_chk.gain - 1.135) < 0.0015, (_chk.base, _chk.gain)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. HETEROGENEITY by cooperation level
# ═══════════════════════════════════════════════════════════════════════════════
het_rows = []
for jk in JUDGE_KEYS:
    sc = SC[jk].copy()
    sc["coop"] = sc["cooperation_level"].map(COOP_LABEL)
    for method in ["PTO", "GRPO"]:
        targets = [("matched_final", MATCHED_END[method], MATCHED_END[method]),
                   ("own_best", BEST[f"{method}_LA0"], BEST[f"{method}_LA5"])]
        for metric in ["Q1Q2", "MICI", "PCT"]:
            d = sc[sc["questionnaire"] == metric]
            for tname, it0, it5 in targets:
                rows = []
                for coop in COOP_ORDER + ["All"]:
                    dd = d if coop == "All" else d[d["coop"] == coop]
                    r = _pair(dd, "score", _model(method, 0, it0), _model(method, 5, it5))
                    # ceiling info: share of K0 endpoint conversations at >= 4.5 (Q1Q2 1-5 scale only)
                    k0 = dd[dd["model"] == _model(method, 0, it0)]["score"]
                    k5 = dd[dd["model"] == _model(method, 5, it5)]["score"]
                    rows.append({"judge": JNAME[jk], "method": method, "metric": metric, "target": tname,
                                 "iter_K0": it0, "iter_K5": it5, "cooperation": coop, "n": r["n"],
                                 "mean_K0": r["mean_a"], "mean_K5": r["mean_b"], "mean_delta": r["mean_delta"],
                                 "dz": r["dz"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"],
                                 "share_K0_ge_4.5": float((k0 >= 4.5).mean()) if metric == "Q1Q2" and len(k0) else np.nan,
                                 "share_K5_ge_4.5": float((k5 >= 4.5).mean()) if metric == "Q1Q2" and len(k5) else np.nan})
                t = pd.DataFrame(rows)
                # Holm across the 3 cooperation subgroups (the 'All' row is a reference, not in the family)
                mask = t["cooperation"] != "All"
                ph = np.full(len(t), np.nan); ph[mask.to_numpy()] = C.holm(t.loc[mask, "p"].to_numpy())
                t["p_holm"] = ph
                het_rows.append(t)
HET = pd.concat(het_rows, ignore_index=True)
HET = HET[["judge", "method", "metric", "target", "iter_K0", "iter_K5", "cooperation", "n", "mean_K0", "mean_K5",
           "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "share_K0_ge_4.5", "share_K5_ge_4.5"]]
C.save_table(HET, f"{SCRIPT}_hetero", caption=(
    "**K0-K5 contrast WITHIN patient cooperation level** (persona trait from the patient system prompt: "
    "High -> Cooperative, StartLowAndChangesToHigh -> Warms up, Low -> Resistant; 32 personas each), on Q1Q2 "
    "(the training reward, 1-5), MICI (MI-inconsistent behaviours per therapist turn; LOWER = better, so a "
    "positive delta means K=0 is WORSE) and PCT (change-talk proportion; higher = better). " + SIGN + " " + PAIR +
    " target=matched_final: PTO iter 10 vs 10, GRPO iter 5 vs 5 (GRPO_LA5 right-censored at 5); target=own_best: "
    "each arm at its own-oracle best iteration (selected on the training oracle's Q1Q2 mean; iter_K0/iter_K5 "
    "columns). mean_K0/mean_K5 = subgroup arm means on the paired personas; dz = mean/sd of paired deltas; 95% "
    "percentile-bootstrap CI; p = Wilcoxon signed-rank; p_holm = Holm across the three cooperation subgroups "
    "within (judge, method, metric, target) — the 'All' row (all 96 personas) is a reference, outside the "
    "family. `share_*_ge_4.5` (Q1Q2 only) = fraction of that arm's subgroup conversations scoring >= 4.5, the "
    "ceiling diagnostic for the Cooperative stratum. Graders side by side, never averaged."))

for jk in JUDGE_KEYS:
    for method in ["PTO", "GRPO"]:
        for metric in ["Q1Q2", "MICI", "PCT"]:
            for tname in ["matched_final", "own_best"]:
                for coop in COOP_ORDER + ["All"]:
                    r = HET[(HET.judge == JNAME[jk]) & (HET.method == method) & (HET.metric == metric)
                            & (HET.target == tname) & (HET.cooperation == coop)].iloc[0]
                    L.put(f"hetero.{jk}.{method}.{metric}.{tname}.{coop.replace(' ', '_')}",
                          {"iter_K0": int(r.iter_K0), "iter_K5": int(r.iter_K5), "n": int(r.n),
                           "mean_K0": _fmt3(r.mean_K0), "mean_K5": _fmt3(r.mean_K5), "mean_delta": _fmt3(r.mean_delta),
                           "dz": _fmt3(r.dz), "ci_lo": _fmt3(r.ci_lo), "ci_hi": _fmt3(r.ci_hi), "p": _fmt3(r.p),
                           "p_holm": _fmt3(r.p_holm), "share_K0_ge_4.5": _fmt3(r["share_K0_ge_4.5"]),
                           "share_K5_ge_4.5": _fmt3(r["share_K5_ge_4.5"])},
                          source=f"tables/{SCRIPT}_hetero.md row judge={JNAME[jk]} method={method} metric={metric} "
                                 f"target={tname} cooperation={coop}")
# ceiling note: Cooperative stratum Q1Q2 levels (base + matched endpoint) per grader
for jk in JUDGE_KEYS:
    sc = SC[jk]; sc = sc[sc["questionnaire"] == "Q1Q2"].copy(); sc["coop"] = sc["cooperation_level"].map(COOP_LABEL)
    out = {}
    for arm in C.ARMS:
        for it in [0, END[arm]]:
            g = sc[(sc["arm"] == arm) & (sc["iteration"] == it)]
            out[f"{arm}.iter{it}"] = {c: {"mean": _fmt3(g[g.coop == c]["score"].mean()),
                                          "share_ge_4.5": _fmt3((g[g.coop == c]["score"] >= 4.5).mean()),
                                          "n": int((g.coop == c).sum())} for c in COOP_ORDER}
    L.put(f"hetero.ceiling.{jk}", out, source="in-script from scores_long (Q1Q2 by cooperation stratum); "
          f"see tables/{SCRIPT}_hetero.md share_K0_ge_4.5 columns")

# figure: K0-K5 delta by cooperation, Q1Q2 (top) + MICI (bottom), grader panels; PTO@10 vs GRPO@5
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True, sharey="row")
xs = np.arange(len(COOP_ORDER)); w = 0.34
for ci, jk in enumerate(JUDGE_KEYS):
    for ri, metric in enumerate(["Q1Q2", "MICI"]):
        ax = axes[ri, ci]
        for mi, method in enumerate(["PTO", "GRPO"]):
            t = HET[(HET.judge == JNAME[jk]) & (HET.method == method) & (HET.metric == metric)
                    & (HET.target == "matched_final") & (HET.cooperation.isin(COOP_ORDER))].set_index("cooperation").loc[COOP_ORDER]
            yerr = [t["mean_delta"] - t["ci_lo"], t["ci_hi"] - t["mean_delta"]]
            ax.bar(xs + (mi - 0.5) * w, t["mean_delta"], w, color=PAL[f"{method}_LA0"], edgecolor="white",
                   yerr=yerr, error_kw=dict(elinewidth=1.0, capsize=2, ecolor="#333333"),
                   label=f"{method} (iter {MATCHED_END[method]} vs {MATCHED_END[method]})")
            for xi, (_, r) in zip(xs + (mi - 0.5) * w, t.iterrows()):
                if r["p_holm"] < 0.05:      # star just above the top of the CI whisker (or above 0)
                    ax.annotate("*", (xi, max(r["ci_hi"], 0.0)), ha="center", va="bottom", fontsize=11)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(COOP_ORDER)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.18)
        if ri == 0:
            ax.set_title(f"grader: {JNAME[jk]}", fontsize=10)
        if ci == 0:
            ax.set_ylabel("K0 − K5, Q1Q2 (1–5)" if metric == "Q1Q2"
                          else "K0 − K5, MICI rate\n(per therapist turn; ↓ better)")
        if ri == 1:
            ax.set_xlabel("patient cooperation level (32 personas each)")
axes[0, 0].legend(fontsize=7.5, loc="upper left", title="* Holm p<0.05 (3 strata)", title_fontsize=7)
fig.suptitle("Look-ahead contrast by patient cooperation (persona-paired; + = K=0 higher)",
             fontsize=10, y=1.0)
C.save_fig(fig, f"{SCRIPT}_fig_hetero")

L.save()
print("[held_out_instruments] done ->", C.OUT / f"{SCRIPT}.json")
