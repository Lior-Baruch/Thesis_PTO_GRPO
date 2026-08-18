"""compute_axis.py — the COST of look-ahead, and every contrast re-read at matched GPU-hours.

Every other table in this paper is indexed by iteration, which is not a unit of spend: a K=5
GRPO optimizer step costs ~1.9x a K=0 step and a whole PTO iteration costs a fraction of a GRPO
one. This generator reconstructs GPU-hours per (arm, iteration) from artifact mtimes
(``eda_analysis.compute``) and re-reads the four contrasts — PTO K, GRPO K, PTO-vs-GRPO at K=0,
PTO-vs-GRPO at K=5 — as a function of BUDGET, under both graders, plus the K contrast at matched
compute on the behaviour channels.

Outputs (all prefixed ``compute_axis_``):
  tables/   by_arm, by_iteration, step_multiplier,
            budget_sweep_<contrast>_<judge> (x8), budget_sweep_crossjudge (+ _verdicts),
            iso_channels (+ _selected)
  figures/  fig_budget_sweep, fig_trajectory (+ fig_trajectory_col, the same two panels stacked
            for a single ACL column), fig_breakdown
  out/compute_axis.json  the ledger

Sign conventions (also stated in every caption):
  * K contrasts are computed as arm_a = LA5, arm_b = LA0 to reproduce the tracked EDA table
    (``results/L5/tables/7_stats/<judge>/budget_sweep.md``); the column ``mean_delta`` is
    K5 - K0.  The paper's convention (+ => K=0 higher) is carried in ``delta_K0_minus_K5`` /
    ``dz_K0_minus_K5`` beside it. The budget-sweep FIGURE plots K5 - K0 (above zero = look-ahead
    ahead), as its axis label says.
  * Method contrasts: arm_a = PTO, arm_b = GRPO; ``+ mean_delta => PTO higher``.
  * MICI and every MICI_* channel are LOWER-is-better; count/length channels have no valence.
  * Everything is paired on ``persona_id`` (never file_index): matched-budget iterations differ
    across arms and the persona shuffle is per-iteration.

Run:  .venv/Scripts/python.exe papers/2026_lookahead_pto_grpo/analysis/compute_axis.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import json  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import eda_analysis  # noqa: E402
from eda_analysis import EdaConfig, compute, behavior  # noqa: E402
from eda_analysis.constants import set_active_judge, LOWER_IS_BETTER as _LIB, DISPLAY_NAMES  # noqa: E402

SCRIPT = "compute_axis"
L = C.Ledger(SCRIPT)
C.style()
PAL = C.palette()

# ── contrasts ─────────────────────────────────────────────────────────────────
#: (tag, arm_a, arm_b, human label). K contrasts put LA5 first to reproduce the tracked EDA table.
CONTRASTS = [
    ("PTO_K",     "PTO_LA5",  "PTO_LA0",  "PTO: K=5 vs K=0"),
    ("GRPO_K",    "GRPO_LA5", "GRPO_LA0", "GRPO: K=5 vs K=0"),
    ("method_K0", "PTO_LA0",  "GRPO_LA0", "K=0: PTO vs GRPO"),
    ("method_K5", "PTO_LA5",  "GRPO_LA5", "K=5: PTO vs GRPO"),
]
K_CONTRASTS = [c for c in CONTRASTS if c[0].endswith("_K")]

#: Behaviour channels for the matched-compute K contrast (one Holm family per budget).
CHANNELS = [
    "MICI_OverPraise", "MICI_OverPraise_rate",
    "MICI_AdviseNoPermission", "MICI_AdviseNoPermission_rate",
    "MICI_BehaviorTotal", "B6_AF", "B6_AF_per_turn",
    "conv_len", "mean_turn_len",
]
TEXT_CHANNELS = {"conv_len", "mean_turn_len"}      # deterministic, grader-independent
CENSOR = "GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h)."
SIGN_K = ("K contrast: mean_delta = K5 - K0 (arm_a=LA5, as the tracked EDA table); "
          "delta_K0_minus_K5 = -mean_delta is the paper's convention (+ => K=0 higher).")
SIGN_M = "Method contrast: mean_delta = PTO - GRPO (+ => PTO higher)."


def direction(metric: str) -> str:
    if metric in _LIB or metric in C.LOWER_IS_BETTER:
        return "lower=better"
    if metric in TEXT_CHANNELS or metric.startswith("B") or metric.endswith("_per_turn"):
        return "count (no valence)" if metric in TEXT_CHANNELS else "higher=more MI-consistent"
    return "higher=better"


def sign_note(tag: str) -> str:
    return SIGN_K if tag.endswith("_K") else SIGN_M


def label_of(metric: str) -> str:
    return DISPLAY_NAMES.get(metric, metric)


# ═════════════════════════════════════════════════════════════════════════════
# 0. Load everything once (primary first, then held-out; never interleave)
# ═════════════════════════════════════════════════════════════════════════════
SC = C.load_scores_both()                       # {'primary': ..., 'heldout': ...}; leaves primary active
ARMS = [a for a in eda_analysis.cross_k_arms(EdaConfig(view="L5")) if a.label in C.ARMS]
assert {a.label for a in ARMS} == set(C.ARMS), [a.label for a in ARMS]

# behaviour channels under each grader (MICI_* and B6_AF are grader-coded; text channels are not)
CH = {"primary": behavior.channel_scores_long(ARMS)}
set_active_judge(C.HELDOUT, 0)
CH["heldout"] = behavior.channel_scores_long(ARMS)
set_active_judge("", 0)
for j in CH:
    CH[j] = CH[j][CH[j]["questionnaire"].isin(CHANNELS)].copy()
    CH[j]["judge"] = C.JUDGE_SHORT[C.JUDGES[j]]

JSHORT = {j: C.JUDGE_SHORT[C.JUDGES[j]] for j in ("primary", "heldout")}
JLABEL = {j: C.JUDGE_LABEL[C.JUDGES[j]] for j in ("primary", "heldout")}

# ═════════════════════════════════════════════════════════════════════════════
# 1. The compute frame + summary + step multiplier
# ═════════════════════════════════════════════════════════════════════════════
comp = compute.iteration_compute(ARMS)
summ = compute.compute_summary(comp)


def _meta_generation_h(arm, it: int) -> float:
    """iteration_metadata.json generation_time_s (per-PROCESS: a resumed/reloaded pass records
    only seconds) — an informational FLOOR beside the mtime-based gen_h, never the headline."""
    fp = os.path.join(arm.runs_dir, f"iteration_{it}", "iteration_metadata.json")
    try:
        with open(fp, encoding="utf-8") as fh:
            v = json.load(fh).get("generation_time_s")
        return float(v) / 3600.0 if v is not None else np.nan
    except Exception:
        return np.nan


arm_by_label = {a.label: a for a in ARMS}
comp["gen_h_meta"] = [
    _meta_generation_h(arm_by_label[r.arm], int(r.iteration)) if r.iteration > 0 else 0.0
    for r in comp.itertuples(index=False)]
# generation FLOOR: the mtime span of model_iter_{k-1}/*.csv starts at the FIRST conversation write,
# so with CONVERSATION_BATCH_SIZE=64 it misses the whole first batch (and collapses to ~0 when all
# CSVs flush together — PTO_LA5 iters 1-5). The recorded per-process generation_time_s is a
# lower bound whenever the pass ran once. gen_h_floor = max(mtime span, recorded) is therefore a
# floor on the true generation time; headline gen_h/gpu_h/cum_gpu_h stay the tracked EDA numbers.
comp["gen_h_floor"] = comp[["gen_h", "gen_h_meta"]].max(axis=1)
comp["gpu_h_floor"] = comp["gen_h_floor"] + comp["build_h"] + comp["train_h"]
comp["cum_gpu_h_floor"] = comp.groupby("arm")["gpu_h_floor"].cumsum()

by_iter_cols = ["arm", "method", "K", "iteration", "n_steps", "median_step_s", "n_imputed",
                "gen_h", "build_h", "train_h", "gpu_h", "cum_gpu_h", "train_source",
                "gen_h_meta", "gen_h_floor", "cum_gpu_h_floor"]
C.save_table(comp[by_iter_cols], f"{SCRIPT}_by_iteration", caption=(
    "GPU-hours per (arm, iteration), reconstructed from artifact mtimes by "
    "eda_analysis.compute.iteration_compute (gap_cutoff 3600 s; deltas outside (0, 3600 s) imputed at "
    "the phase median, n_imputed counts them). gen = rollout pass that produced model_iter_{k-1}; "
    "build = PTO pref-tree branching + oracle (PTO only); train = optimizer loop (GRPO: completions "
    "parquet mtimes; PTO: TensorBoard wall_time). Iteration 0 = the base policy (0 h by construction). "
    "cum_gpu_h = cost of having produced <Arm>_I{k} (headline = the tracked EDA numbers). "
    "gen_h_meta = iteration_metadata.json generation_time_s/3600 (per-PROCESS: a reloaded/resumed pass "
    "records seconds); gen_h_floor = max(gen_h, gen_h_meta) is a FLOOR on generation time, because the "
    "mtime span starts at the first conversation write and so misses the first batch of 64 (~0.1 h) and "
    "collapses to ~0 when all CSVs flush together (PTO_LA5 iters 1-5, whose time lands in iter 6). "
    f"cum_gpu_h_floor re-cumulates with it. {CENSOR}"))

# per-arm summary + ratios with arithmetic
summ = summ.copy()
tot = summ.set_index("arm")["total_gpu_h"]
per = summ.set_index("arm")["gpu_h_per_iter"]
ff = comp[comp.iteration > 0].groupby("arm")["gpu_h_floor"].sum()
summ["total_gpu_h_floor"] = summ["arm"].map(ff)
summ["build_share"] = summ["build_h"] / summ["total_gpu_h"]
summ["train_share"] = summ["train_h"] / summ["total_gpu_h"]
C.save_table(summ, f"{SCRIPT}_by_arm", caption=(
    "One row per arm: iterations trained, phase GPU-hours and total, cost per iteration "
    "(eda_analysis.compute.compute_summary). build_share/train_share = phase / total. "
    "total_gpu_h_floor uses gen_h_floor = max(mtime span, recorded generation_time_s) per iteration "
    "(see by_iteration: the mtime span misses the first batch, ~0.1 h/iter, and is ~0 for PTO_LA5 "
    f"iters 1-5); the headline total_gpu_h is the tracked EDA number. {CENSOR}"))

ratios = {
    "GRPO_per_iter_K5_over_K0": (per["GRPO_LA5"], per["GRPO_LA0"]),
    "PTO_per_iter_K5_over_K0": (per["PTO_LA5"], per["PTO_LA0"]),
    "GRPO_over_PTO_per_iter_K0": (per["GRPO_LA0"], per["PTO_LA0"]),
    "GRPO_over_PTO_per_iter_K5": (per["GRPO_LA5"], per["PTO_LA5"]),
    "GRPO_LA0_total_over_PTO_LA0_total": (tot["GRPO_LA0"], tot["PTO_LA0"]),
    "GRPO_LA5_total_over_PTO_LA5_total": (tot["GRPO_LA5"], tot["PTO_LA5"]),
    "PTO_LA5_total_over_PTO_LA0_total": (tot["PTO_LA5"], tot["PTO_LA0"]),
    "GRPO_LA5_5iters_over_GRPO_LA0_10iters": (tot["GRPO_LA5"], tot["GRPO_LA0"]),
}
for k, (a, b) in ratios.items():
    L.put(f"ratio.{k}", {"num": round(float(a), 3), "den": round(float(b), 3),
                         "ratio": round(float(a / b), 3),
                         "arithmetic": f"{a:.3f}/{b:.3f}={a / b:.2f}"},
          source=f"tables/{SCRIPT}_by_arm.md")
totf = summ.set_index("arm")["total_gpu_h_floor"]
for k, (a, b) in {"GRPO_LA0_total_over_PTO_LA0_total": ("GRPO_LA0", "PTO_LA0"),
                  "PTO_LA5_total_over_PTO_LA0_total": ("PTO_LA5", "PTO_LA0"),
                  "GRPO_LA5_total_over_PTO_LA5_total": ("GRPO_LA5", "PTO_LA5")}.items():
    L.put(f"ratio_floor.{k}", {"num": round(float(totf[a]), 3), "den": round(float(totf[b]), 3),
                               "ratio": round(float(totf[a] / totf[b]), 3),
                               "arithmetic": f"{totf[a]:.3f}/{totf[b]:.3f}={totf[a] / totf[b]:.2f}"},
          source=f"tables/{SCRIPT}_by_arm.md (total_gpu_h_floor)",
          note="generation floor applied (max of mtime span and recorded generation_time_s); headline ratios are ratio.*")
for r in summ.itertuples(index=False):
    L.put(f"by_arm.{r.arm}", {c: (round(float(getattr(r, c)), 3) if isinstance(getattr(r, c), (float, np.floating)) else getattr(r, c))
                              for c in ["last_iter", "n_iters", "gen_h", "build_h", "train_h",
                                        "total_gpu_h", "gpu_h_per_iter", "median_step_s",
                                        "n_imputed", "build_share", "train_share",
                                        "total_gpu_h_floor"]},
          source=f"tables/{SCRIPT}_by_arm.md")
for r in comp[comp.iteration > 0].itertuples(index=False):
    L.put(f"by_iteration.{r.arm}.I{int(r.iteration)}",
          {"gen_h": round(float(r.gen_h), 3), "build_h": round(float(r.build_h), 3),
           "train_h": round(float(r.train_h), 3), "gpu_h": round(float(r.gpu_h), 3),
           "cum_gpu_h": round(float(r.cum_gpu_h), 3), "n_steps": int(r.n_steps),
           "median_step_s": round(float(r.median_step_s), 3), "n_imputed": int(r.n_imputed),
           "gen_h_meta": round(float(r.gen_h_meta), 3),
           "gen_h_floor": round(float(r.gen_h_floor), 3),
           "cum_gpu_h_floor": round(float(r.cum_gpu_h_floor), 3)},
          source=f"tables/{SCRIPT}_by_iteration.md")

# step multiplier: GRPO per-step ratio (the price of look-ahead inside the training loop) beside
# PTO's build-phase ratio (where PTO's look-ahead cost actually lands — its DPO step has no K).
sm = compute.step_multiplier(comp, "GRPO").rename(columns={
    "median_s_K0": "GRPO_median_step_s_K0", "median_s_K5": "GRPO_median_step_s_K5",
    "ratio_median": "GRPO_step_ratio_K5_over_K0"})
pto = comp[(comp.method == "PTO") & (comp.iteration > 0)].pivot_table(
    index="iteration", columns="arm", values=["build_h", "gpu_h", "median_step_s"])
sm = sm.set_index("iteration")
sm["PTO_dpo_median_step_s_K0"] = pto[("median_step_s", "PTO_LA0")]
sm["PTO_dpo_median_step_s_K5"] = pto[("median_step_s", "PTO_LA5")]
sm["PTO_dpo_step_ratio"] = sm["PTO_dpo_median_step_s_K5"] / sm["PTO_dpo_median_step_s_K0"]
sm["PTO_build_h_K0"] = pto[("build_h", "PTO_LA0")]
sm["PTO_build_h_K5"] = pto[("build_h", "PTO_LA5")]
sm["PTO_build_ratio_K5_over_K0"] = sm["PTO_build_h_K5"] / sm["PTO_build_h_K0"]
sm["PTO_iter_gpu_h_K0"] = pto[("gpu_h", "PTO_LA0")]
sm["PTO_iter_gpu_h_K5"] = pto[("gpu_h", "PTO_LA5")]
sm["PTO_iter_ratio_K5_over_K0"] = sm["PTO_iter_gpu_h_K5"] / sm["PTO_iter_gpu_h_K0"]
sm = sm.reset_index()
sm["iteration"] = sm["iteration"].astype(int)
C.save_table(sm, f"{SCRIPT}_step_multiplier", caption=(
    "The per-iteration price of look-ahead. GRPO: median optimizer-step seconds K=0 vs K=5 and their "
    "ratio (eda_analysis.compute.step_multiplier; the K=5 reward computation runs 5 extra simulated turns "
    "per candidate INSIDE the training loop). GRPO_LA5 has no rows past iteration 5 (right-censored). "
    "PTO: the DPO step carries no look-ahead (ratio ~1); PTO's look-ahead cost lands in the pref-tree "
    "BUILD phase, so its build_h ratio and whole-iteration gpu_h ratio are shown instead. Iteration 1 "
    "of GRPO_LA5 ran at LOOKAHEAD_SUB_BATCH_SIZE=64 with a fat API-latency tail (ratio 2.41), so quote "
    "the settled iterations 3-5 (~1.9x)."))
for r in sm.itertuples(index=False):
    L.put(f"step_multiplier.I{int(r.iteration)}",
          {"GRPO_median_step_s_K0": round(float(r.GRPO_median_step_s_K0), 3),
           "GRPO_median_step_s_K5": (round(float(r.GRPO_median_step_s_K5), 3) if pd.notna(r.GRPO_median_step_s_K5) else None),
           "GRPO_step_ratio": (round(float(r.GRPO_step_ratio_K5_over_K0), 3) if pd.notna(r.GRPO_step_ratio_K5_over_K0) else None),
           "PTO_build_ratio": round(float(r.PTO_build_ratio_K5_over_K0), 3),
           "PTO_iter_ratio": round(float(r.PTO_iter_ratio_K5_over_K0), 3),
           "PTO_dpo_step_ratio": round(float(r.PTO_dpo_step_ratio), 3)},
          source=f"tables/{SCRIPT}_step_multiplier.md")
_settled = sm[sm.iteration.isin([3, 4, 5])]["GRPO_step_ratio_K5_over_K0"]
L.put("step_multiplier.GRPO_settled_iters_3_5", {"median_ratio": round(float(_settled.median()), 3),
                                                  "values": [round(float(x), 3) for x in _settled]},
      source=f"tables/{SCRIPT}_step_multiplier.md")
_pb = sm["PTO_build_ratio_K5_over_K0"]
L.put("step_multiplier.PTO_build_ratio_all_iters", {"median": round(float(_pb.median()), 3),
                                                     "min": round(float(_pb.min()), 3), "max": round(float(_pb.max()), 3)},
      source=f"tables/{SCRIPT}_step_multiplier.md")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Budget sweeps — best checkpoint within budget, both graders, four contrasts
# ═════════════════════════════════════════════════════════════════════════════
def _means_names(scores: pd.DataFrame, metric: str):
    sub = scores[scores["questionnaire"] == metric]
    means = sub.groupby(["arm", "iteration"])["score"].mean()
    names = {(a, int(i)): m for a, i, m in
             sub[["arm", "iteration", "model"]].drop_duplicates().itertuples(index=False)}
    return means, names


def _best_within(arm: str, budget: float, cum: pd.Series, means: pd.Series, lower_better: bool):
    elig = [int(i) for i, c in cum.items() if c <= budget + 1e-9 and (arm, int(i)) in means.index]
    if not elig:
        return None
    key = (lambda i: -means.loc[(arm, i)]) if lower_better else (lambda i: means.loc[(arm, i)])
    return max(elig, key=key)


def sweep(eval_scores: pd.DataFrame, select_scores: pd.DataFrame, arm_a: str, arm_b: str, *,
          select_metric: str, eval_metric: str, budgets: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Mirror of eda_analysis.compute.budget_sweep with three additions: (i) the checkpoint is
    SELECTED on ``select_scores``/``select_metric`` and the contrast SCORED on
    ``eval_scores``/``eval_metric`` (same frame+metric = the tracked sweep); (ii) LOWER_IS_BETTER
    metrics select the minimum; (iii) bootstrap CI + the selected model names are returned.
    ``+ mean_delta => arm_a higher``. Pairs on persona_id."""
    A = comp[(comp.arm == arm_a) & (comp.iteration > 0)].set_index("iteration")["cum_gpu_h"]
    B = comp[(comp.arm == arm_b) & (comp.iteration > 0)].set_index("iteration")["cum_gpu_h"]
    sel_means, _ = _means_names(select_scores, select_metric)
    ev_means, ev_names = _means_names(eval_scores, eval_metric)
    lower = select_metric in C.LOWER_IS_BETTER or select_metric in _LIB
    W = C.wide(eval_scores, eval_metric)
    rows = []
    for budget in (sorted(A.values) if budgets is None else budgets):
        ia = _best_within(arm_a, budget, A, sel_means, lower)
        ib = _best_within(arm_b, budget, B, sel_means, lower)
        if ia is None or ib is None:
            continue
        ma, mb = ev_names.get((arm_a, ia)), ev_names.get((arm_b, ib))
        if ma is None or mb is None or ma not in W.columns or mb not in W.columns:
            continue
        st = C.paired(W[ma].to_numpy(), W[mb].to_numpy())
        rows.append({"budget_gpu_h": round(float(budget), 2),
                     "select_metric": select_metric, "eval_metric": eval_metric,
                     "best_iter_a": ia, "best_iter_b": ib,
                     "cum_gpu_h_a": round(float(A.loc[ia]), 2), "cum_gpu_h_b": round(float(B.loc[ib]), 2),
                     "model_a": ma, "model_b": mb,
                     "mean_a": float(ev_means.loc[(arm_a, ia)]), "mean_b": float(ev_means.loc[(arm_b, ib)]),
                     "n": st["n"], "mean_delta": st["mean_delta"], "dz": st["dz"],
                     "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"], "p": st["p"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Holm within this (judge, contrast, select->eval) family over the UNIQUE checkpoint pairs
    # (repeated budgets re-use one pair; correcting per row would over-count the same test).
    key = df["best_iter_a"].astype(str) + "/" + df["best_iter_b"].astype(str)
    uniq = df.drop_duplicates(subset=["best_iter_a", "best_iter_b"])
    ph = dict(zip(uniq["best_iter_a"].astype(str) + "/" + uniq["best_iter_b"].astype(str),
                  C.holm(uniq["p"].values)))
    df["p_holm"] = key.map(ph)
    df["n_unique_pairs"] = len(uniq)
    df["arm_a"], df["arm_b"] = arm_a, arm_b
    return df


def add_k_convention(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if tag.endswith("_K") and not df.empty:
        df = df.copy()
        df["delta_K0_minus_K5"] = -df["mean_delta"]
        df["dz_K0_minus_K5"] = -df["dz"]
    return df


# -- 2a. same-judge sweeps (the tracked table + MICI + Q1Q2-selected MICI) -----------------------
SWEEPS = {}          # (judge, tag) -> DataFrame (all metric variants)
for j in ("primary", "heldout"):
    for tag, a, b, lab in CONTRASTS:
        parts = [sweep(SC[j], SC[j], a, b, select_metric="Q1Q2", eval_metric="Q1Q2"),
                 sweep(SC[j], SC[j], a, b, select_metric="MICI", eval_metric="MICI"),
                 sweep(SC[j], SC[j], a, b, select_metric="Q1Q2", eval_metric="MICI")]
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        df["judge"] = JSHORT[j]
        df = add_k_convention(df, tag)
        SWEEPS[(j, tag)] = df
        # cross-check against eda_analysis.compute.budget_sweep on Q1Q2 (same-judge selection)
        ref = compute.budget_sweep(SC[j], comp, a, b, metric="Q1Q2")
        mine = df[(df.select_metric == "Q1Q2") & (df.eval_metric == "Q1Q2")].reset_index(drop=True)
        assert len(ref) == len(mine), (j, tag, len(ref), len(mine))
        assert np.allclose(ref["mean_delta"].values, mine["mean_delta"].values, atol=1e-9)
        assert np.allclose(ref["dz"].values, mine["dz"].values, atol=1e-9)
        assert (ref["best_iter_a"].values == mine["best_iter_a"].values).all()
        assert (ref["best_iter_b"].values == mine["best_iter_b"].values).all()
        cols = ["judge", "budget_gpu_h", "select_metric", "eval_metric", "best_iter_a", "best_iter_b",
                "model_a", "model_b", "cum_gpu_h_a", "cum_gpu_h_b", "mean_a", "mean_b", "n",
                "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "n_unique_pairs"]
        if tag.endswith("_K"):
            cols += ["delta_K0_minus_K5", "dz_K0_minus_K5"]
        C.save_table(df[cols], f"{SCRIPT}_budget_sweep_{tag}_{JSHORT[j]}", caption=(
            f"Budget sweep, {lab} ({a} = arm_a vs {b} = arm_b), grader = {JLABEL[j]}. At each of arm_a's "
            "cumulative GPU-h budgets both arms are represented by the best checkpoint they could have "
            "reached for that money (best on select_metric under this grader; MICI selects the LOWEST), "
            "and the contrast is scored on eval_metric paired on persona_id (n personas; bootstrap 95% CI; "
            "Wilcoxon p; Holm within this table's (select_metric, eval_metric) family over the unique "
            f"checkpoint pairs). {sign_note(tag)} MICI is lower-is-better. Rows select_metric=Q1Q2 -> "
            "eval_metric=MICI score the Q1Q2-selected checkpoints on MICI (does the reward-selected "
            f"policy carry the hack?). Mirrors eda_analysis.compute.budget_sweep row-for-row on Q1Q2. {CENSOR}"))
        for r in df.itertuples(index=False):
            L.put(f"sweep.{JSHORT[j]}.{tag}.{r.select_metric}_to_{r.eval_metric}.budget_{r.budget_gpu_h:g}h",
                  {"best_iter_a": int(r.best_iter_a), "best_iter_b": int(r.best_iter_b),
                   "model_a": r.model_a, "model_b": r.model_b,
                   "mean_a": round(float(r.mean_a), 3), "mean_b": round(float(r.mean_b), 3),
                   "n": int(r.n), "mean_delta": round(float(r.mean_delta), 3), "dz": round(float(r.dz), 3),
                   "ci": [round(float(r.ci_lo), 3), round(float(r.ci_hi), 3)],
                   "p": round(float(r.p), 4), "p_holm": round(float(r.p_holm), 4),
                   **({"delta_K0_minus_K5": round(float(-r.mean_delta), 3)} if tag.endswith("_K") else {})},
                  source=f"tables/{SCRIPT}_budget_sweep_{tag}_{JSHORT[j]}.md",
                  note=("mean_delta = arm_a - arm_b; " + sign_note(tag)))

# top-of-sweep verdicts (same-judge)
top_rows = []
for (j, tag), df in SWEEPS.items():
    d = df[(df.select_metric == "Q1Q2") & (df.eval_metric == "Q1Q2")]
    if d.empty:
        continue
    r = d.iloc[-1]
    top_rows.append({"judge": JSHORT[j], "contrast": tag, "arm_a": r.arm_a, "arm_b": r.arm_b,
                     "budget_gpu_h": r.budget_gpu_h, "best_iter_a": r.best_iter_a, "best_iter_b": r.best_iter_b,
                     "mean_delta": r.mean_delta, "dz": r.dz, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                     "p": r.p, "p_holm": r.p_holm,
                     "sign_flips_within_sweep": bool((np.sign(d["mean_delta"]) != np.sign(r.mean_delta)).any())})
    L.put(f"sweep_top.{JSHORT[j]}.{tag}",
          {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
           for k, v in top_rows[-1].items()},
          source=f"tables/{SCRIPT}_budget_sweep_{tag}_{JSHORT[j]}.md (last row, Q1Q2->Q1Q2)")

# -- 2b. cross-judge selection sweep (Q1Q2) ------------------------------------------------------
xj_rows = []
for tag, a, b, lab in CONTRASTS:
    A_budgets = np.sort(comp[(comp.arm == a) & (comp.iteration > 0)]["cum_gpu_h"].values)
    for sj in ("primary", "heldout"):
        for ej in ("primary", "heldout"):
            d = sweep(SC[ej], SC[sj], a, b, select_metric="Q1Q2", eval_metric="Q1Q2", budgets=A_budgets)
            if d.empty:
                continue
            d.insert(0, "eval_judge", JSHORT[ej])
            d.insert(0, "select_judge", JSHORT[sj])
            d.insert(0, "contrast", tag)
            d["honest_selection"] = sj != ej
            xj_rows.append(add_k_convention(d, tag))
XJ = pd.concat(xj_rows, ignore_index=True)
xj_cols = ["contrast", "arm_a", "arm_b", "select_judge", "eval_judge", "honest_selection", "budget_gpu_h",
           "best_iter_a", "best_iter_b", "model_a", "model_b", "mean_a", "mean_b", "n",
           "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "delta_K0_minus_K5", "dz_K0_minus_K5"]
C.save_table(XJ[xj_cols], f"{SCRIPT}_budget_sweep_crossjudge", caption=(
    "Cross-judge selection sweep on Q1Q2. Each arm's best-within-budget checkpoint is SELECTED on "
    "select_judge's Q1Q2 means and the paired contrast is SCORED on eval_judge's Q1Q2 (persona_id "
    "pairing; bootstrap 95% CI; Wilcoxon p; Holm within each (contrast, select_judge, eval_judge) "
    "family over unique checkpoint pairs). honest_selection = the grader that picked the checkpoint "
    "is not the grader that scores it (the same-judge rows reproduce the per-judge sweep tables). "
    f"{SIGN_K} {SIGN_M} delta_K0_minus_K5 is blank for method contrasts. {CENSOR}"))

# verdicts at the top budget of each contrast, all four (select, eval) combinations
verd = (XJ.sort_values("budget_gpu_h").groupby(["contrast", "select_judge", "eval_judge"], sort=False)
        .tail(1).sort_values(["contrast", "select_judge", "eval_judge"]))
verd = verd[["contrast", "arm_a", "arm_b", "select_judge", "eval_judge", "honest_selection", "budget_gpu_h",
             "best_iter_a", "best_iter_b", "n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]].copy()
verd["verdict"] = np.where(verd["p_holm"] < 0.05,
                           np.where(verd["mean_delta"] > 0, "arm_a > arm_b", "arm_a < arm_b"), "no sig. difference")
C.save_table(verd, f"{SCRIPT}_budget_sweep_crossjudge_verdicts", caption=(
    "Top-of-sweep verdicts (each contrast at arm_a's LAST cumulative budget) under every "
    "(select_judge, eval_judge) combination, from budget_sweep_crossjudge. A verdict that holds only "
    "when the same grader selects and scores is a selection artefact; the honest_selection rows are "
    f"the ones to quote. + mean_delta => arm_a higher ({SIGN_K} {SIGN_M}) Paired on persona_id; "
    f"p_holm within the (contrast, select_judge, eval_judge) family. {CENSOR}"))
for r in verd.itertuples(index=False):
    L.put(f"crossjudge_verdict.{r.contrast}.select_{r.select_judge}.eval_{r.eval_judge}",
          {"budget_gpu_h": float(r.budget_gpu_h), "best_iter_a": int(r.best_iter_a), "best_iter_b": int(r.best_iter_b),
           "mean_delta": round(float(r.mean_delta), 3), "dz": round(float(r.dz), 3),
           "ci": [round(float(r.ci_lo), 3), round(float(r.ci_hi), 3)],
           "p": round(float(r.p), 4), "p_holm": round(float(r.p_holm), 4), "verdict": r.verdict,
           "honest_selection": bool(r.honest_selection)},
          source=f"tables/{SCRIPT}_budget_sweep_crossjudge_verdicts.md")
# every honest cross-judge row into the ledger too
for r in XJ[XJ.honest_selection].itertuples(index=False):
    L.put(f"crossjudge.{r.contrast}.select_{r.select_judge}.eval_{r.eval_judge}.budget_{r.budget_gpu_h:g}h",
          {"best_iter_a": int(r.best_iter_a), "best_iter_b": int(r.best_iter_b),
           "mean_delta": round(float(r.mean_delta), 3), "dz": round(float(r.dz), 3),
           "ci": [round(float(r.ci_lo), 3), round(float(r.ci_hi), 3)],
           "p": round(float(r.p), 4), "p_holm": round(float(r.p_holm), 4)},
          source=f"tables/{SCRIPT}_budget_sweep_crossjudge.md")

# ═════════════════════════════════════════════════════════════════════════════
# 3. K at matched compute on the behaviour channels
# ═════════════════════════════════════════════════════════════════════════════
iso_parts = []
for j in ("primary", "heldout"):
    for tag, a, b, lab in K_CONTRASTS:
        d = compute.iso_compute_contrast(CH[j], comp, a, b, metrics=CHANNELS)
        if d.empty:
            continue
        # text channels are grader-independent: keep them once (under the primary row set)
        if j == "heldout":
            d = d[~d["metric"].isin(TEXT_CHANNELS)]
        d.insert(0, "judge", np.where(d["metric"].isin(TEXT_CHANNELS), "text (grader-independent)", JSHORT[j]))
        d.insert(0, "contrast", tag)
        # bootstrap CI (iso_compute_contrast reports mean/dz/p only)
        W = {m: C.wide(CH[j], m) for m in d["metric"].unique()}
        ci = [C.paired(W[r.metric][r.model_a].to_numpy(), W[r.metric][r.model_b].to_numpy())
              for r in d.itertuples(index=False)]
        d["ci_lo"] = [c["ci_lo"] for c in ci]
        d["ci_hi"] = [c["ci_hi"] for c in ci]
        iso_parts.append(d)
ISO = pd.concat(iso_parts, ignore_index=True)
ISO["channel"] = ISO["metric"].map(label_of)
ISO["direction"] = ISO["metric"].map(direction)
ISO["delta_K0_minus_K5"] = -ISO["mean_delta"]
ISO["dz_K0_minus_K5"] = -ISO["dz"]
ISO["iso_ok"] = ISO["budget_ratio"].between(0.9, 1.1)      # only these rows are honestly iso-compute
iso_cols = ["contrast", "judge", "metric", "channel", "direction", "iter_a", "iter_b", "cum_gpu_h_a", "cum_gpu_h_b",
            "budget_ratio", "iso_ok", "model_a", "model_b", "n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm",
            "delta_K0_minus_K5", "dz_K0_minus_K5"]
C.save_table(ISO[iso_cols], f"{SCRIPT}_iso_channels", caption=(
    "Look-ahead at MATCHED compute on the behaviour channels (eda_analysis.compute.iso_compute_contrast on "
    "behavior.channel_scores_long). For every trained iteration of the K=5 arm (arm_a) the K=0 iteration "
    "of the same method with the closest cumulative GPU-h is paired on persona_id (budget_ratio = b/a; "
    "iso_ok flags 0.9-1.1; PTO_LA0 tops out at 8.12 GPU-h so PTO_LA5 iters >= 5 have no iso partner and are "
    "flagged False). mean_delta = K5 - K0 on the channel's own unit; "
    "delta_K0_minus_K5 = -mean_delta is the paper's convention (+ => K=0 higher). direction says how to "
    "read the sign: MICI_* channels are lower-is-better (over-praise / advise-without-permission per "
    "session and per therapist turn, MI-inconsistent acts per session); B6_AF = MITI-coded affirmations "
    "per session / per turn; conv_len (utterances) and mean_turn_len (chars) are deterministic text "
    "measures with no valence and are grader-independent (reported once). Holm within the 9-channel "
    "family at one budget pair. MICI/B6_AF rows are shown under both graders side by side; never "
    f"averaged. {CENSOR}"))
for r in ISO.itertuples(index=False):
    L.put(f"iso_channels.{r.contrast}.{r.judge.split(' ')[0]}.{r.metric}.iterA{int(r.iter_a)}_iterB{int(r.iter_b)}",
          {"cum_gpu_h_a": float(r.cum_gpu_h_a), "cum_gpu_h_b": float(r.cum_gpu_h_b), "budget_ratio": float(r.budget_ratio),
           "n": int(r.n), "mean_delta_K5_minus_K0": round(float(r.mean_delta), 3), "dz": round(float(r.dz), 3),
           "ci": [round(float(r.ci_lo), 3), round(float(r.ci_hi), 3)],
           "p": round(float(r.p), 4), "p_holm": round(float(r.p_holm), 4), "direction": r.direction,
           "iso_ok": bool(r.iso_ok)},
          source=f"tables/{SCRIPT}_iso_channels.md")

# 3b. the channels at the Q1Q2-SELECTED checkpoints of the top budget (honest: what an operator who
#     picked the best-within-budget reward checkpoint would actually deploy)
sel_rows = []
for j in ("primary", "heldout"):
    for tag, a, b, lab in K_CONTRASTS:
        d = SWEEPS[(j, tag)]
        d = d[(d.select_metric == "Q1Q2") & (d.eval_metric == "Q1Q2")]
        top = d.iloc[-1]
        ma, mb = top.model_a, top.model_b
        for m in CHANNELS:
            if j == "heldout" and m in TEXT_CHANNELS:
                continue
            W = C.wide(CH[j], m)
            if ma not in W.columns or mb not in W.columns:
                continue
            st = C.paired(W[ma].to_numpy(), W[mb].to_numpy())
            sel_rows.append({"contrast": tag, "judge": ("text (grader-independent)" if m in TEXT_CHANNELS else JSHORT[j]),
                             "selected_on": f"Q1Q2 ({JSHORT[j]}) best-within-budget", "budget_gpu_h": top.budget_gpu_h,
                             "metric": m, "channel": label_of(m), "direction": direction(m),
                             "iter_a": int(top.best_iter_a), "iter_b": int(top.best_iter_b), "model_a": ma, "model_b": mb,
                             "mean_a": float(np.nanmean(W[ma])), "mean_b": float(np.nanmean(W[mb])), **st})
SEL = pd.DataFrame(sel_rows)
SEL["p_holm"] = np.nan
for (tag, j), idx in SEL.groupby(["contrast", "judge"]).groups.items():
    SEL.loc[idx, "p_holm"] = C.holm(SEL.loc[idx, "p"].values)
SEL["delta_K0_minus_K5"] = -SEL["mean_delta"]
SEL["dz_K0_minus_K5"] = -SEL["dz"]
C.save_table(SEL[["contrast", "judge", "selected_on", "budget_gpu_h", "metric", "channel", "direction", "iter_a", "iter_b",
                  "model_a", "model_b", "mean_a", "mean_b", "n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm",
                  "delta_K0_minus_K5", "dz_K0_minus_K5"]],
             f"{SCRIPT}_iso_channels_selected", caption=(
    "Behaviour channels at the checkpoints an operator would actually deploy: for each method, the K=5 "
    "(arm_a) and K=0 (arm_b) checkpoints selected as best-within-budget on Q1Q2 under the named grader at "
    "the TOP budget of the K sweep (the last row of budget_sweep_<method>_K_<judge>), contrasted on each "
    "channel paired on persona_id (bootstrap 95% CI, Wilcoxon p, Holm within the channel family per "
    "(contrast, judge)). mean_delta = K5 - K0 in the channel's unit; delta_K0_minus_K5 is the paper's "
    "convention (+ => K=0 higher); direction gives the valence (MICI_* lower=better; text channels none, "
    f"grader-independent, reported once). {CENSOR}"))
for r in SEL.itertuples(index=False):
    L.put(f"iso_channels_selected.{r.contrast}.{r.judge.split(' ')[0]}.{r.metric}",
          {"iter_a": int(r.iter_a), "iter_b": int(r.iter_b), "budget_gpu_h": float(r.budget_gpu_h),
           "mean_a_K5": round(float(r.mean_a), 3), "mean_b_K0": round(float(r.mean_b), 3), "n": int(r.n),
           "mean_delta_K5_minus_K0": round(float(r.mean_delta), 3), "dz": round(float(r.dz), 3),
           "ci": [round(float(r.ci_lo), 3), round(float(r.ci_hi), 3)],
           "p": round(float(r.p), 4), "p_holm": round(float(r.p_holm), 4), "direction": r.direction},
          source=f"tables/{SCRIPT}_iso_channels_selected.md")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Figures
# ═════════════════════════════════════════════════════════════════════════════
def _kstyle(arm):
    return C.K_STYLE[C.k_of(arm)]


# 4a. budget sweep: 2x2 rows = grader, cols = method; y = K5 - K0 Q1Q2 with CI
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex="col")
for ci_, method in enumerate(["PTO", "GRPO"]):
    tag, a, b, lab = next(c for c in K_CONTRASTS if c[0].startswith(method))
    for ri, j in enumerate(["primary", "heldout"]):
        ax = axes[ri, ci_]
        d = SWEEPS[(j, tag)]
        d = d[(d.select_metric == "Q1Q2") & (d.eval_metric == "Q1Q2")]
        x = d["budget_gpu_h"].values; y = d["mean_delta"].values
        yerr = np.vstack([y - d["ci_lo"].values, d["ci_hi"].values - y])
        col = PAL[a]
        ax.axhline(0, color="0.35", lw=0.9, zorder=1)
        ax.errorbar(x, y, yerr=yerr, color=col, ls="--", marker="s", ms=5.5, lw=1.7, capsize=2.5,
                    elinewidth=1.0, label=f"{a} vs {b}", zorder=3)
        # significant points filled, non-significant hollow
        ns = d["p_holm"].values >= 0.05
        if ns.any():
            ax.plot(x[ns], y[ns], ls="none", marker="s", ms=5.5, mfc="white", mec=col, mew=1.3, zorder=4,
                    label="not sig. (Holm)")
        prev = None
        for xi, yi, ia, ib in zip(x, y, d["best_iter_a"], d["best_iter_b"]):
            if (ia, ib) == prev:          # a repeated checkpoint pair — label it once
                continue
            prev = (ia, ib)
            ax.annotate(f"I{ia}/I{ib}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.5, color="0.25")
        ax.set_title(f"{method} — {JSHORT[j]}", fontsize=10)
        if ci_ == 0:
            ax.set_ylabel("Q1Q2 Δ  (K=5 − K=0)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right", frameon=True)
for ax in axes[:, 0]:
    ax.set_xlim(0, 21)
for ax in axes[:, 1]:
    ax.set_xlim(5, 29)
for ax in axes[1, :]:
    ax.set_xlabel("cumulative GPU-hours (K=5 arm's budget)", fontsize=9)
fig.suptitle("Look-ahead vs budget: paired Q1Q2 delta between best-within-budget checkpoints "
             "(labels I_K5/I_K0 = selected iterations)", fontsize=9)
p = C.save_fig(fig, f"{SCRIPT}_fig_budget_sweep")
L.put("figures.budget_sweep", str(p.name), source=(
    "Budget sweep 2x2 (rows = grader, cols = method): x = K=5 arm's cumulative GPU-h; y = paired Q1Q2 "
    "delta K5 - K0 between the best-within-budget checkpoints (bootstrap 95% CI; hollow = Holm p>=0.05); "
    "labels I_K5/I_K0 name the selected iterations; persona_id pairing; GRPO_LA5 right-censored at I5."))

# 4b. trajectory: Q1Q2 vs cumulative GPU-h, four arms, two grader panels
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
for ax, j in zip(axes, ["primary", "heldout"]):
    sbc = compute.score_by_compute(SC[j], comp, metric="Q1Q2")
    for arm in C.ARMS:
        d = sbc[sbc.arm == arm].sort_values("iteration")
        ks = _kstyle(arm)
        ax.errorbar(d["cum_gpu_h"], d["mean"], yerr=d["sem"], color=PAL[arm], ls=ks["ls"], marker=ks["marker"],
                    ms=5, lw=1.7, capsize=2, elinewidth=0.9, label=arm)
        last = d.iloc[-1]
        ax.annotate(f"I{int(last.iteration)}", (last.cum_gpu_h, last["mean"]), textcoords="offset points",
                    xytext=(4, -3 if arm.endswith("LA0") else 4), fontsize=7, color=PAL[arm])
    ax.set_title(f"Q1Q2 vs compute — {JSHORT[j]}", fontsize=10)
    ax.set_xlabel("cumulative GPU-hours", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Q1Q2 (mean ± SEM, 96 personas)", fontsize=9)
axes[0].legend(fontsize=7.5, loc="lower right", frameon=True)
p = C.save_fig(fig, f"{SCRIPT}_fig_trajectory")
L.put("figures.trajectory", str(p.name), source=(
    "Q1Q2 mean +- SEM (96 personas) vs cumulative GPU-h per arm, one panel per grader; iteration 0 at 0 h; "
    "K=0 solid/circle, K=5 dashed/square; last point labelled with its iteration; GRPO_LA5 ends at I5."))

# 4b'. single-column variant of 4b for the paper body: the SAME two panels stacked (shared x and
#      y), sized for a 3.4-in ACL column. Same data, same style, fonts scaled for the narrow width.
#      Deliberately NOT a ledger entry — the ledger (out/compute_axis.json) is layout-agnostic and
#      must stay byte-identical whichever variant the .tex includes.
fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True, sharey=True)
for ax, j in zip(axes, ["primary", "heldout"]):
    sbc = compute.score_by_compute(SC[j], comp, metric="Q1Q2")
    for arm in C.ARMS:
        d = sbc[sbc.arm == arm].sort_values("iteration")
        ks = _kstyle(arm)
        ax.errorbar(d["cum_gpu_h"], d["mean"], yerr=d["sem"], color=PAL[arm], ls=ks["ls"], marker=ks["marker"],
                    ms=4, lw=1.4, capsize=1.5, elinewidth=0.8, label=arm)
        last = d.iloc[-1]
        ax.annotate(f"I{int(last.iteration)}", (last.cum_gpu_h, last["mean"]), textcoords="offset points",
                    xytext=(3, -3 if arm.endswith("LA0") else 3), fontsize=6.5, color=PAL[arm])
    ax.set_title(f"Q1Q2 vs compute — {JSHORT[j]}", fontsize=9)
    ax.set_ylabel("Q1Q2 (mean ± SEM, 96 personas)", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.3)
axes[1].set_xlabel("cumulative GPU-hours", fontsize=8)
axes[0].legend(fontsize=6.5, loc="lower right", frameon=True, handlelength=2.2)
C.save_fig(fig, f"{SCRIPT}_fig_trajectory_col")

# 4c. breakdown: stacked gen/build/train per iteration per arm
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
# phase encoding: train = solid arm colour; build = arm colour, light; generate = white + arm-colour hatch
for ax, arm in zip(axes, C.ARMS):
    d = comp[(comp.arm == arm) & (comp.iteration > 0)]
    bottom = np.zeros(len(d))
    ax.bar(d["iteration"], d["gen_h"], bottom=bottom, facecolor="white", edgecolor=PAL[arm],
           hatch="////", linewidth=0.5, width=0.8)
    bottom += d["gen_h"].values
    ax.bar(d["iteration"], d["build_h"], bottom=bottom, color=PAL[arm], alpha=0.45,
           edgecolor="white", linewidth=0.5, width=0.8)
    bottom += d["build_h"].values
    ax.bar(d["iteration"], d["train_h"], bottom=bottom, color=PAL[arm], edgecolor="white",
           linewidth=0.5, width=0.8)
    ax.set_title(f"{arm}  (Σ {d['gpu_h'].sum():.1f} h)", fontsize=9)
    ax.set_xlabel("iteration", fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    if arm == "GRPO_LA5":
        ax.text(7.9, 3.0, "right-censored\n(stopped at I5)", ha="center", va="center", fontsize=7, color="0.3")
    if arm == "PTO_LA5":
        ax.text(3.2, 4.2, "gen of I1–I5 lands in I6\n(flushed conv mtimes)", ha="center", va="center",
                fontsize=6.5, color="0.3")
axes[0].set_ylabel("GPU-hours per iteration", fontsize=9)
handles = [Patch(facecolor="white", edgecolor="0.4", hatch="////", label="generate"),
           Patch(facecolor="0.4", alpha=0.45, label="build (PTO only)"),
           Patch(facecolor="0.4", label="train")]
axes[3].legend(handles=handles, fontsize=7, loc="upper right", frameon=True, title="phase", title_fontsize=7)
p = C.save_fig(fig, f"{SCRIPT}_fig_breakdown")
L.put("figures.breakdown", str(p.name), source=(
    "Stacked GPU-h per iteration per arm (generate / build / train, mtime-reconstructed); PTO_LA5 gen "
    "for iters 1-5 lands in iter 6 (batch-flushed conv mtimes); GRPO_LA5 right-censored at I5."))

# ═════════════════════════════════════════════════════════════════════════════
# 5. Ledger extras + save
# ═════════════════════════════════════════════════════════════════════════════
L.put("caveats", [
    "All GPU-hours are mtime-reconstructed (eda_analysis.compute); never quote iteration_metadata.json timings.",
    "PTO_LA5 iters 1-5 show gen_h ~0.000 because their conversation CSV mtimes were batch-flushed; the time "
    "lands in iter 6 (0.967 h). Cumulative totals are right; per-iteration gen splits are not, for that arm. "
    "gen_h is a systematic UNDER-estimate (~0.1 h/iter: the mtime span misses the first batch of 64); gen_h_floor / "
    "cum_gpu_h_floor / total_gpu_h_floor use max(mtime, recorded generation_time_s) without changing the headline; "
    "under the floor PTO_LA0 8.12->9.22 h, PTO_LA5 19.68->21.08 h, GRPO_LA0 27.91->28.77 h, GRPO_LA5 27.08->27.42 h.",
    "GRPO_LA5 is right-censored at iteration 5 (27.08 GPU-h); its sweep never reaches GRPO_LA0's later checkpoints.",
    "Iso-compute pairs different iterations across arms; pairing is on persona_id.",
    "Quote budget_sweep rows, not a single iso-compute row: the sign of the K lever depends on budget.",
    "K-contrast tables carry mean_delta = K5 - K0 (tracked-EDA convention) AND delta_K0_minus_K5 (paper convention).",
    "Same-judge best-within-budget selection is optimistic for the selecting grader; read the crossjudge tables.",
], source="compute_axis.py")
out = L.save()
print("wrote", out)
