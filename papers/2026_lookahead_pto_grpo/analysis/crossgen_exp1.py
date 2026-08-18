"""crossgen_exp1.py — the replication link to the ICLR 2025 paper.

The ICLR poster (Exp1: Llama-2-7B therapist, GPT-3.5 patient + oracle, PTO K in {0,5}, 7
iterations, 96 patient permutations) found K=5 clearly ahead of K=0. Exp3 (Llama-3.2-1B,
gpt-4o-mini) finds K=5 never leading. Both the model AND the grader changed between the two, so
the reversal is confounded. ``Exp3_PTO_GRPO/eda/tools/score_crossgen.py`` re-scored the very same
Exp1 conversations with the Exp3 oracle (gpt-4o-mini-2024-07-18 @ T=0.1, V5 JSON-schema Q1 + Q2)
into ``data/eval_scores/_crossgen/``. This script is the first analysis of that re-score:

  1. levels    — per Exp1 model state, Q1 / Q2 / Final under gpt-4o-mini beside the ICLR Table 1
                 GPT-3.5 means (transcribed) AND the on-disk GPT-3.5 per-conversation scores.
  2. kcontrast — K=0 minus K=5 per iteration, paired on conversation index (= patient permutation:
                 Exp1 did not shuffle personas), under BOTH graders; best-vs-best; the "every L5
                 above every L0" ordering claim; Spearman rank agreement between the graders' 15
                 model means; pooled-arm contrast; each model vs Base.
  3. fig       — two panels (GPT-3.5 left, gpt-4o-mini right), Final by iteration, K=0 solid /
                 K=5 dashed, Base dotted, 95% bootstrap CIs; ``fig_col`` = the same two panels
                 stacked for a single ACL column (same data, narrow-width fonts).
  4. la3       — the K=3 sweep on disk (4 iterations, DIFFERENT hyper-parameters) is not scored by
                 the tool; report what scoring it would cost + its GPT-3.5 means. NO API calls.

Sign convention everywhere: ``delta = K0 - K5``; + => K=0 higher (mirrors eda_analysis).
Pairing unit: Exp1 conversation index i == patient permutation i (verified in-script).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402

SCRIPT = "crossgen_exp1"
L = C.Ledger(SCRIPT)

EXP1 = C.REPO / "Exp1_ICLR2025" / "data" / "conversations_eval"
CROSSGEN = (C.EXP3 / "data" / "eval_scores" / "_crossgen"
            / f"judge={C.PRIMARY}" / "rep=0")
ORACLE_TOKEN = "Q1Q2"
N_PERSONAS = 96
ITERS = list(range(1, 8))

# ── the Exp1 model manifest (mirrors score_crossgen.exp1_models exactly) ─────
_ARM_DIR = "TTree1.4_TT0.9_TP0.7_TE0.2_V{}"


def exp1_manifest() -> dict[str, tuple[str, int, Path]]:
    """model -> (arm label 'Base'/'L0'/'L5', iteration, conversation dir)."""
    m = {"Exp1_Base": ("Base", 0, EXP1 / "Base" / "Basic_50_TT0.9_TP0.7_TE0.2_V2")}
    for k in (0, 5):
        for i in ITERS:
            m[f"Exp1_LA{k}_I{i}"] = (f"L{k}", i, EXP1 / f"LookAhead_{k}" / _ARM_DIR.format(i))
    return m


MANIFEST = exp1_manifest()
LA3_DIRS = {f"Exp1_LA3_I{i}": EXP1 / "LookAhead_3"
            / f"FullEval_TTree1.4_TT0.7_TP0.7_TE0.2_Filter0.2_V{i}.0" for i in range(1, 5)}


# ── loaders ──────────────────────────────────────────────────────────────────

def load_crossgen() -> pd.DataFrame:
    """Long frame of the gpt-4o-mini re-score: model, arm, iteration, conv_index, Q1, Q2, Final.

    One CSV per (metric, model, conversation): ``metric=Q1/.../<model>/<i>.csv`` holds the item
    scores + ``Q1_Mean``; ``metric=Q2`` likewise with ``Q2_Mean``. ``i`` is the Exp1 conversation
    index (``conversation_i.csv``), i.e. patient permutation i. Final = mean(Q1_Mean, Q2_Mean),
    exactly the lake's ``Q1Q2`` composite and the ICLR paper's "Final Score".
    """
    rows = []
    for model, (arm, it, _) in MANIFEST.items():
        per_metric = {}
        for metric in ("Q1", "Q2"):
            d = CROSSGEN / f"metric={metric}" / f"oracle={ORACLE_TOKEN}" / model
            vals = {}
            for f in sorted(d.glob("*.csv")):
                i = int(f.stem)
                s = pd.read_csv(f)
                vals[i] = float(s[f"{metric}_Mean"].iloc[0])
            per_metric[metric] = vals
        idx = sorted(set(per_metric["Q1"]) & set(per_metric["Q2"]))
        for i in idx:
            q1, q2 = per_metric["Q1"][i], per_metric["Q2"][i]
            rows.append(dict(model=model, arm=arm, iteration=it, conv_index=i,
                             Q1=q1, Q2=q2, Final=(q1 + q2) / 2.0))
    return pd.DataFrame(rows)


def load_gpt35(manifest: dict) -> pd.DataFrame:
    """The ORIGINAL GPT-3.5 per-conversation oracle scores that Exp1 saved beside each
    conversation (``scores_i.csv``: scores1_avg = Q1 mean, scores2_avg = Q2 mean,
    scores_avg = Final). These are the numbers ICLR Table 1 averaged."""
    rows = []
    for model, (arm, it, d) in manifest.items():
        for i in range(N_PERSONAS):
            p = d / f"scores_{i}.csv"
            if not p.exists():
                continue
            s = pd.read_csv(p)
            rows.append(dict(model=model, arm=arm, iteration=it, conv_index=i,
                             Q1=float(s["scores1_avg"].iloc[0]),
                             Q2=float(s["scores2_avg"].iloc[0]),
                             Final=float(s["scores_avg"].iloc[0])))
    return pd.DataFrame(rows)


def _persona_signature(text: str) -> tuple:
    """(age, problem-keyword) parsed from the patient's first utterance — a cheap fingerprint of
    the patient permutation, used only to VERIFY that index i is the same persona across models."""
    age = re.search(r"(\d\d)[- ]years?[- ]old", text)
    low = text.lower()
    # Exp1's grid has two problems (smoking / obesity); collapse the wording to that class.
    if any(k in low for k in ("smok", "cigarette")):
        prob = "smoking"
    elif any(k in low for k in ("weight", "obes", "diet", "exercis")):
        prob = "obesity"
    else:
        prob = None
    return (age.group(1) if age else None, prob)


def persona_alignment_check(manifest: dict) -> dict:
    """Exp1 generated conversation_i from ``permutations[i]`` of a deterministic nested-loop
    generator (system_prompts_builder.generate_all_permutations) with NO shuffle, so index i is
    persona i in every model dir. Confirm empirically: for each i, the (age, problem) fingerprint
    of the patient's opening line must agree across all model states (where parseable)."""
    sig = {}
    for model, (_, _, d) in manifest.items():
        for i in range(N_PERSONAS):
            p = d / f"conversation_{i}.csv"
            if p.exists():
                df = pd.read_csv(p)
                if len(df) > 1:
                    sig.setdefault(i, []).append(_persona_signature(str(df["conversation"].iloc[1])))
    n_ok = n_conf = 0
    for i, sigs in sig.items():
        ages = {a for a, _ in sigs if a is not None}
        probs = {p for _, p in sigs if p is not None}
        # "conflict" = two DIFFERENT parsed values at the same index (missing parses are ignored)
        if len(ages) > 1 or len(probs) > 1:
            n_conf += 1
        else:
            n_ok += 1
    return {"n_indices": len(sig), "n_consistent": n_ok, "n_conflicting": n_conf}


# ── statistics ───────────────────────────────────────────────────────────────

def _wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(index="conv_index", columns="model", values=metric, aggfunc="mean")


def paired_models(df: pd.DataFrame, metric: str, model_a: str, model_b: str) -> dict:
    """model_a − model_b on *metric*, paired on conv_index. Mirrors C.paired (Wilcoxon, dz,
    percentile-bootstrap CI). ``+ => model_a higher``."""
    w = _wide(df, metric)
    if model_a not in w or model_b not in w:
        return dict(n=0, mean_delta=np.nan, dz=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
                    p_t=np.nan, n_a_higher=0, n_b_higher=0)
    return _paired_arrays(w[model_a].to_numpy(), w[model_b].to_numpy())


def _paired_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    """C.paired (Wilcoxon p, dz, bootstrap CI) + a paired-t p and the sign split of the deltas.
    The extras exist because GPT-3.5's deltas are heavy-tailed (a few 2-3 point collapses), so
    the Wilcoxon p and the bootstrap CI can disagree; the sign split says which read is fair."""
    out = C.paired(a, b)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    out["p_t"] = float(sps.ttest_rel(a[ok], b[ok]).pvalue) if d.size >= 3 else np.nan
    out["n_a_higher"] = int((d > 0).sum())
    out["n_b_higher"] = int((d < 0).sum())
    return out


def unpaired_delta(df: pd.DataFrame, metric: str, model_a: str, model_b: str) -> dict:
    """Difference of means + Welch t (the UNPAIRED reading, reported beside the paired one)."""
    a = df.loc[df["model"] == model_a, metric].dropna().to_numpy()
    b = df.loc[df["model"] == model_b, metric].dropna().to_numpy()
    if len(a) < 3 or len(b) < 3:
        return dict(delta_unpaired=np.nan, p_welch=np.nan)
    t = sps.ttest_ind(a, b, equal_var=False)
    return dict(delta_unpaired=float(a.mean() - b.mean()), p_welch=float(t.pvalue))


def k_contrast_table(df: pd.DataFrame, grader: str) -> pd.DataFrame:
    """K0 − K5 at every iteration for Q1, Q2, Final; Holm within (grader, metric) across the 7
    iterations. Paired on conv_index. ``+ => K=0 higher``."""
    rows = []
    for metric in ("Final", "Q1", "Q2"):
        for it in ITERS:
            a, b = f"Exp1_LA0_I{it}", f"Exp1_LA5_I{it}"
            r = paired_models(df, metric, a, b)
            r.update(unpaired_delta(df, metric, a, b))
            rows.append(dict(grader=grader, metric=metric, iteration=it, model_K0=a, model_K5=b, **r))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for metric, g in out.groupby("metric"):
        out.loc[g.index, "p_holm"] = C.holm(g["p"].to_numpy())
    return out


def vs_base_table(df: pd.DataFrame, grader: str) -> pd.DataFrame:
    """Every trained model − Base (Q1, Q2, Final), paired on conv_index; Holm within
    (grader, arm, metric) across the 7 iterations. ``+ => trained model higher``."""
    rows = []
    for metric in ("Final", "Q1", "Q2"):
        for arm, k in (("L0", 0), ("L5", 5)):
            for it in ITERS:
                m = f"Exp1_LA{k}_I{it}"
                r = paired_models(df, metric, m, "Exp1_Base")
                rows.append(dict(grader=grader, arm=arm, metric=metric, iteration=it, model=m, **r))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for (arm, metric), g in out.groupby(["arm", "metric"]):
        out.loc[g.index, "p_holm"] = C.holm(g["p"].to_numpy())
    return out


def pooled_arm_contrast(df: pd.DataFrame, grader: str) -> pd.DataFrame:
    """Per-persona mean over the 7 iterations of each arm, then K0 − K5 paired on conv_index —
    the arm-level reading of "L5 models score higher than L0 models". ``+ => K=0 higher``."""
    rows = []
    for metric in ("Final", "Q1", "Q2"):
        per = (df[df["arm"].isin(["L0", "L5"])]
               .groupby(["arm", "conv_index"])[metric].mean().unstack("arm"))
        r = _paired_arrays(per["L0"].to_numpy(), per["L5"].to_numpy())
        rows.append(dict(grader=grader, metric=metric, contrast="mean over iters 1-7, K0 - K5", **r))
    return pd.DataFrame(rows)


def la3_cost_estimate() -> dict:
    """Mirror score_crossgen.dry_run for the 4 LookAhead_3 dirs WITHOUT importing the tool or
    calling any API: prefix caches at 50%; transcripts billed at full rate; ~180 output tokens.
    Prices = the tool's gpt-4o-mini list rates (USD / 1M tok)."""
    PRICE_IN, PRICE_IN_CACHED, PRICE_OUT = 0.150, 0.075, 0.600
    PREFIX_TOK = 1084          # measured rubric-first fixed prefix (CLAUDE.md, gotchas)
    n_conv = 0
    chars = 0
    for d in LA3_DIRS.values():
        for i in range(N_PERSONAS):
            p = d / f"conversation_{i}.csv"
            if p.exists():
                utts = pd.read_csv(p)["conversation"].astype(str).tolist()
                if not utts:
                    continue
                n_conv += 1
                chars += sum(len(f"[{'THERAPIST' if j % 2 == 0 else 'PATIENT'}]: {u}\n")
                             for j, u in enumerate(utts))
    n_calls = n_conv * 2                     # Q1 + Q2
    body_tok = chars / 4.0 * 2               # both metrics see the transcript
    cached_tok = PREFIX_TOK * n_calls
    out_tok = n_calls * 180
    cost = (body_tok * PRICE_IN + cached_tok * PRICE_IN_CACHED + out_tok * PRICE_OUT) / 1e6
    uncached = ((body_tok + cached_tok) * PRICE_IN + out_tok * PRICE_OUT) / 1e6
    return dict(n_conversations=n_conv, n_calls=n_calls,
                transcript_tok_per_call=body_tok / max(n_calls, 1),
                est_cost_usd=cost, est_cost_usd_if_cache_misses=uncached)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    C.style()

    # 0. pairing-unit check + loaders ------------------------------------------------
    align = persona_alignment_check(MANIFEST)
    print("persona alignment by conversation index:", align)
    L.put("pairing.persona_index_alignment", align,
          source="in-script check on Exp1 conversation_i.csv openings",
          note="Exp1 wrote conversation_i from permutations[i] of a deterministic nested-loop "
               "generator (no shuffle); index i = patient permutation i in every model dir. "
               "n_conflicting counts indices whose parsed (age, problem) disagree across models.")

    cg = load_crossgen()
    g35 = load_gpt35(MANIFEST)
    t1 = C.load_iclr_table1()
    print(f"crossgen rows: {len(cg)}  ({cg['model'].nunique()} models)   gpt-3.5 rows: {len(g35)}")

    # Cross-check: on-disk GPT-3.5 per-conversation means must reproduce ICLR Table 1 -------
    g35_means = g35.groupby(["arm", "iteration"])[["Q1", "Q2", "Final"]].mean().reset_index()
    chk = t1.merge(g35_means, on=["arm", "iteration"], suffixes=("_tab1", "_disk"))
    max_abs = max(float((chk[f"{m}_tab1"] - chk[f"{m}_disk"]).abs().max()) for m in ("Q1", "Q2", "Final"))
    print(f"ICLR Table 1 vs on-disk GPT-3.5 means: max |diff| = {max_abs:.4f} over 15 models x 3 cols")
    assert max_abs < 0.0015, "on-disk GPT-3.5 scores do not reproduce ICLR Table 1"
    L.put("crosscheck.iclr_table1_vs_disk_max_abs_diff", max_abs,
          source="C.load_iclr_table1() vs Exp1 scores_i.csv means (tables/crossgen_exp1_levels.md)",
          note="15 model states x {Q1,Q2,Final}; the transcription and the on-disk scores agree to 3 dp")

    # 1. levels ------------------------------------------------------------------------
    def _lv(df, tag):
        g = df.groupby("model")
        out = pd.DataFrame({
            f"n_{tag}": g.size(),
            f"Q1_{tag}": g["Q1"].mean(), f"Q2_{tag}": g["Q2"].mean(),
            f"Final_{tag}": g["Final"].mean(), f"Final_sd_{tag}": g["Final"].std(ddof=1),
        })
        return out
    lv = _lv(cg, "gpt4omini").join(_lv(g35, "gpt35"), how="outer")
    lv["arm"] = [MANIFEST[m][0] for m in lv.index]
    lv["iteration"] = [MANIFEST[m][1] for m in lv.index]
    lv["K"] = lv["arm"].map({"Base": "", "L0": "0", "L5": "5"})
    lv = lv.reset_index().rename(columns={"index": "model"})
    t1x = t1.rename(columns={"Q1": "Q1_iclr_tab1", "Q2": "Q2_iclr_tab1", "Final": "Final_iclr_tab1"})
    lv = lv.merge(t1x, on=["arm", "iteration"], how="left")
    lv["Final_gap_gpt4omini_minus_gpt35"] = lv["Final_gpt4omini"] - lv["Final_gpt35"]
    lv = lv.sort_values(["K", "iteration"])
    cols = ["model", "arm", "K", "iteration",
            "n_gpt4omini", "Q1_gpt4omini", "Q2_gpt4omini", "Final_gpt4omini", "Final_sd_gpt4omini",
            "n_gpt35", "Q1_gpt35", "Q2_gpt35", "Final_gpt35", "Final_sd_gpt35",
            "Q1_iclr_tab1", "Q2_iclr_tab1", "Final_iclr_tab1", "Final_gap_gpt4omini_minus_gpt35"]
    lv = lv[cols]
    C.save_table(lv, f"{SCRIPT}_levels", caption=(
        "Exp1 (ICLR 2025: Llama-2-7B therapist, GPT-3.5 patient) model states, 96 conversations each "
        "(one per patient permutation), scored by TWO graders side by side (never averaged): "
        "`*_gpt4omini` = the SAME conversations re-scored by the Exp3 oracle (gpt-4o-mini-2024-07-18, "
        "T=0.1, V5 JSON-schema Q1+Q2; data/eval_scores/_crossgen); `*_gpt35` = the original GPT-3.5 "
        "oracle scores Exp1 saved beside each conversation (scores_i.csv); `*_iclr_tab1` = ICLR Table 1 "
        "as printed (reproduced by `*_gpt35` to 3 dp). Final = mean(Q1, Q2) per conversation, then "
        "averaged (= the lake's Q1Q2 composite). Iteration 0 = the untrained Llama-2-7B base (a "
        "single draw; both arms share it). Q1 = session satisfaction (5 items), Q2 = working "
        "alliance (17 items), both 1-5. Rows: Base, then K=0 iters 1-7, then K=5 iters 1-7."))
    for _, r in lv.iterrows():
        L.put(f"levels.{r['model']}", {k: r[k] for k in cols[4:]},
              source=f"tables/{SCRIPT}_levels.md row model={r['model']}")

    # 2. K contrast under both graders --------------------------------------------------
    kc = pd.concat([k_contrast_table(cg, "gpt-4o-mini"), k_contrast_table(g35, "gpt-3.5")],
                   ignore_index=True)
    kc = kc.rename(columns={"n_a_higher": "n_K0_higher", "n_b_higher": "n_K5_higher"})
    kc = kc[["grader", "metric", "iteration", "model_K0", "model_K5", "n", "mean_delta", "dz",
             "ci_lo", "ci_hi", "p", "p_holm", "p_t", "n_K0_higher", "n_K5_higher",
             "delta_unpaired", "p_welch"]]
    kc = kc.sort_values(["grader", "metric", "iteration"]).reset_index(drop=True)
    C.save_table(kc, f"{SCRIPT}_kcontrast", caption=(
        "Look-ahead contrast on Exp1's conversations at matched iteration: `mean_delta` = "
        "score(K=0 model) - score(K=5 model); **+ => K=0 higher, - => K=5 higher**. Paired on "
        "conversation index (= patient permutation; Exp1 did not shuffle personas), n = 96 pairs; "
        "dz = mean/sd of the paired deltas; ci = 95% percentile bootstrap (2000 draws); p = Wilcoxon "
        "signed-rank; p_holm = Holm within each (grader, metric) family across the 7 iterations; "
        "p_t = paired t (uncorrected); n_K0_higher / n_K5_higher = sign split of the 96 deltas (ties "
        "excluded). GPT-3.5's deltas are heavy-tailed (a few 2-3 point collapses of one side), so "
        "its Wilcoxon p and bootstrap CI can disagree — read the sign split. "
        "`delta_unpaired`/`p_welch` = the unpaired difference of means + Welch t, for reference. "
        "Two graders side by side, never averaged: gpt-4o-mini = the Exp3 oracle re-scoring the same "
        "transcripts (Q1+Q2, V5 rubric); gpt-3.5 = the original ICLR oracle's per-conversation scores. "
        "Metric Final = mean(Q1,Q2) = the lake's Q1Q2. Both arms run 7 iterations (no censoring)."))
    for _, r in kc.iterrows():
        L.put(f"kcontrast.{r['grader']}.{r['metric']}.iter{int(r['iteration'])}",
              {k: r[k] for k in ("n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "p_t",
                                 "n_K0_higher", "n_K5_higher", "delta_unpaired", "p_welch")},
              source=f"tables/{SCRIPT}_kcontrast.md row grader={r['grader']} metric={r['metric']} iteration={int(r['iteration'])}",
              note="delta = K0 - K5; + => K=0 higher")

    # summary rows: best-vs-best (ICLR's L0_M4 vs L5_M7), own-best per grader, pooled arms,
    # 'every L5 > every L0', Spearman between the graders' 15 model means.
    summ = []
    for grader, df in (("gpt-4o-mini", cg), ("gpt-3.5", g35)):
        means = df.groupby("model")["Final"].mean()
        # (a) ICLR's pick: L0_M4 vs L5_M7
        for metric in ("Final", "Q1", "Q2"):
            r = paired_models(df, metric, "Exp1_LA0_I4", "Exp1_LA5_I7")
            summ.append(dict(grader=grader, contrast="ICLR best-vs-best: L0_I4 - L5_I7", metric=metric, **r))
        # (b) each grader's own best (by that grader's Final mean)
        b0 = max((f"Exp1_LA0_I{i}" for i in ITERS), key=lambda m: means[m])
        b5 = max((f"Exp1_LA5_I{i}" for i in ITERS), key=lambda m: means[m])
        for metric in ("Final", "Q1", "Q2"):
            r = paired_models(df, metric, b0, b5)
            summ.append(dict(grader=grader, contrast=f"own best-vs-best: {b0.split('_', 1)[1]} - {b5.split('_', 1)[1]}",
                             metric=metric, **r))
        L.put(f"best.{grader}", {"best_K0": b0, "best_K5": b5,
                                 "best_K0_final": float(means[b0]), "best_K5_final": float(means[b5])},
              source=f"tables/{SCRIPT}_levels.md (max Final_* over iters 1-7 per arm)")
        # (c) pooled arms
        for _, r in pooled_arm_contrast(df, grader).iterrows():
            summ.append(dict(grader=grader, contrast=r["contrast"], metric=r["metric"],
                             **{k: r[k] for k in ("n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_t",
                                                  "n_a_higher", "n_b_higher")}))
        # (d) ordering claim: min over L5 model means > max over L0 model means?
        l0 = means[[f"Exp1_LA0_I{i}" for i in ITERS]]
        l5 = means[[f"Exp1_LA5_I{i}" for i in ITERS]]
        n_l5_above_all_l0 = int((l5 > l0.max()).sum())
        n_pairs_l5_gt_l0 = int(sum(v5 > v0 for v5 in l5 for v0 in l0))
        n_iter_l5_gt_l0 = int(sum(l5.iloc[i] > l0.iloc[i] for i in range(7)))
        order = dict(min_L5_final=float(l5.min()), max_L0_final=float(l0.max()),
                     every_L5_above_every_L0=bool(l5.min() > l0.max()),
                     n_L5_models_above_max_L0=n_l5_above_all_l0,
                     n_of_49_L5xL0_pairs_with_L5_higher=n_pairs_l5_gt_l0,
                     n_of_7_iterations_L5_higher=n_iter_l5_gt_l0,
                     n_of_14_trained_models_above_base=int((means.drop("Exp1_Base") > means["Exp1_Base"]).sum()))
        L.put(f"ordering.{grader}", order, source=f"tables/{SCRIPT}_levels.md (Final_* columns)",
              note="the ICLR text's group claim, checked on model MEANS")
        summ.append(dict(grader=grader, contrast="ordering: min(L5 mean) - max(L0 mean)", metric="Final",
                         n=7, mean_delta=float(l5.min() - l0.max()), dz=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                         p=np.nan, p_t=np.nan, n_a_higher=49 - n_pairs_l5_gt_l0, n_b_higher=n_pairs_l5_gt_l0))
    summ = pd.DataFrame(summ).rename(columns={"n_a_higher": "n_K0_higher", "n_b_higher": "n_K5_higher"})
    summ = summ[["grader", "contrast", "metric", "n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_t",
                 "n_K0_higher", "n_K5_higher"]]
    summ["note"] = np.where(summ["contrast"].str.startswith("ordering"),
                            "L5-minus-L0 of MEANS: + => every K=5 model above every K=0 model",
                            "+ => first-named (K=0) higher")

    # (e) Spearman between the two graders' 15 model means (+ per-conversation agreement)
    corr_rows = []
    lv_idx = lv.set_index("model")
    for metric in ("Final", "Q1", "Q2"):
        a = lv_idx[f"{metric}_gpt4omini"]; b = lv_idx[f"{metric}_gpt35"]
        rho, p = sps.spearmanr(a, b)
        pr, pp = sps.pearsonr(a, b)
        # trained-only (14) so the base does not anchor the rank
        tr = [m for m in lv_idx.index if m != "Exp1_Base"]
        rho14, p14 = sps.spearmanr(a[tr], b[tr])
        # per-conversation agreement, pooled over the 15 x 96 conversations
        m = cg.merge(g35, on=["model", "conv_index"], suffixes=("_4m", "_35"))
        rc, pc = sps.spearmanr(m[f"{metric}_4m"], m[f"{metric}_35"])
        corr_rows.append(dict(metric=metric, level="15 model means", spearman_rho=float(rho), p=float(p),
                              pearson_r=float(pr), n=15))
        corr_rows.append(dict(metric=metric, level="14 trained model means", spearman_rho=float(rho14),
                              p=float(p14), pearson_r=float(sps.pearsonr(a[tr], b[tr])[0]), n=14))
        corr_rows.append(dict(metric=metric, level="per conversation (pooled)", spearman_rho=float(rc),
                              p=float(pc), pearson_r=float(sps.pearsonr(m[f"{metric}_4m"], m[f"{metric}_35"])[0]),
                              n=int(len(m))))
    corr = pd.DataFrame(corr_rows)
    for _, r in corr.iterrows():
        L.put(f"grader_agreement.{r['metric']}.{r['level'].replace(' ', '_')}",
              {k: r[k] for k in ("spearman_rho", "p", "pearson_r", "n")},
              source=f"tables/{SCRIPT}_kcontrast_summary.md (grader agreement block)")

    # write summary + agreement as one md (two csvs)
    C.save_table(summ, f"{SCRIPT}_kcontrast_summary", caption=(
        "Summary contrasts on Exp1's conversations under each grader (never averaged across graders). "
        "Best-vs-best rows: `mean_delta` = first-named model minus second (K=0 minus K=5); "
        "**+ => K=0 higher**; paired on conversation index (= persona), n = 96, dz / 95% bootstrap CI / "
        "Wilcoxon p (uncorrected: one planned contrast per row). 'ICLR best-vs-best' repeats the poster's "
        "L0_M4 vs L5_M7 comparison; 'own best-vs-best' picks each arm's best iteration under THAT grader. "
        "p_t = paired t; n_K0_higher / n_K5_higher = sign split of the 96 paired deltas (ties excluded); "
        "for the 'ordering' row they instead count, over the 49 (K=0 model, K=5 model) pairs of MEANS, "
        "how many have the K=0 / the K=5 model higher. "
        "'mean over iters 1-7' averages each persona's score over an arm's 7 iterations first, then "
        "contrasts the arms (the arm-level 'K=5 models score higher' claim). The 'ordering' row is "
        "min(K=5 model mean) - max(K=0 model mean): + means every K=5 model outscores every K=0 model."))
    C.save_table(corr, f"{SCRIPT}_grader_agreement", caption=(
        "Agreement between the two graders on Exp1's conversations: Spearman rho (and Pearson r) between "
        "the gpt-4o-mini re-score and the original GPT-3.5 oracle, at the level of the 15 model-state "
        "means (Base + 7 K=0 + 7 K=5), the 14 trained-model means, and per conversation (15 x 96 pooled). "
        "Metric Final = mean(Q1,Q2)."))
    for _, r in summ.iterrows():
        key = re.sub(r"[^A-Za-z0-9]+", "_", f"{r['contrast']}").strip("_")
        L.put(f"summary.{r['grader']}.{r['metric']}.{key}",
              {k: r[k] for k in ("n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_t", "n_K0_higher", "n_K5_higher")},
              source=f"tables/{SCRIPT}_kcontrast_summary.md row grader={r['grader']} contrast='{r['contrast']}' metric={r['metric']}",
              note=str(r["note"]))

    # (f) vs base under both graders (the ICLR "every PTO model beats the baseline" claim)
    vb = pd.concat([vs_base_table(cg, "gpt-4o-mini"), vs_base_table(g35, "gpt-3.5")], ignore_index=True)
    vb = vb.rename(columns={"n_a_higher": "n_model_higher", "n_b_higher": "n_base_higher"})
    vb = vb[["grader", "arm", "metric", "iteration", "model", "n", "mean_delta", "dz", "ci_lo", "ci_hi",
             "p", "p_holm", "p_t", "n_model_higher", "n_base_higher"]]
    C.save_table(vb, f"{SCRIPT}_vsbase", caption=(
        "Each trained Exp1 model minus the untrained Llama-2-7B Base, under each grader (side by side, "
        "never averaged): `mean_delta` = model - Base; **+ => trained model higher**; paired on conversation "
        "index (= persona), n = 96; dz, 95% bootstrap CI, Wilcoxon p, Holm within each (grader, arm, metric) "
        "family across the 7 iterations; p_t = paired t; n_model_higher / n_base_higher = sign split. gpt-4o-mini = the Exp3 oracle re-score; gpt-3.5 = the original "
        "ICLR oracle. Metric Final = mean(Q1,Q2)."))
    for grader in ("gpt-4o-mini", "gpt-3.5"):
        sub = vb[(vb["grader"] == grader) & (vb["metric"] == "Final")]
        L.put(f"vsbase.{grader}.Final",
              {"n_models_positive": int((sub["mean_delta"] > 0).sum()),
               "n_models_p_holm_lt_05": int((sub["p_holm"] < 0.05).sum()),
               "min_delta": float(sub["mean_delta"].min()), "max_delta": float(sub["mean_delta"].max()),
               "per_model": {r["model"]: {"mean_delta": r["mean_delta"], "dz": r["dz"], "p_holm": r["p_holm"]}
                             for _, r in sub.iterrows()}},
              source=f"tables/{SCRIPT}_vsbase.md rows grader={grader} metric=Final",
              note="14 trained models; + => above Base")

    # 3. figure ---------------------------------------------------------------------------
    pal = C.palette(["PTO_LA0", "PTO_LA5"])
    col = {"L0": pal["PTO_LA0"], "L5": pal["PTO_LA5"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, (df, title) in zip(axes, ((g35, f"GPT-3.5 (original ICLR oracle)"),
                                      (cg, "gpt-4o-mini (Exp3 oracle, same transcripts)"))):
        for arm, k in (("L0", 0), ("L5", 5)):
            ys, lo, hi = [], [], []
            for it in ITERS:
                v = df.loc[df["model"] == f"Exp1_LA{k}_I{it}", "Final"].dropna().to_numpy()
                ys.append(v.mean())
                rng = np.random.default_rng(0)
                b = rng.choice(v, size=(2000, v.size), replace=True).mean(axis=1)
                lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
            st = C.K_STYLE[k]
            ax.plot(ITERS, ys, ls=st["ls"], marker=st["marker"], ms=5.5, lw=1.7, color=col[arm],
                    label=f"PTO K={k}")
            ax.fill_between(ITERS, lo, hi, color=col[arm], alpha=0.15, lw=0)
        base = df.loc[df["model"] == "Exp1_Base", "Final"].mean()
        ax.axhline(base, ls=":", lw=1.5, color="#555555", label="Base (Llama-2-7B)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PTO iteration")
        ax.set_xticks(ITERS)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Final = mean(Q1, Q2)  [1-5]")
    axes[1].set_ylabel("Final = mean(Q1, Q2)  [1-5]")
    h, lab = axes[0].get_legend_handles_labels()
    lab = [l + ("  (bands: 95% bootstrap CI)" if l.startswith("PTO K=5") else "") for l in lab]
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Exp1 (ICLR 2025) conversations under two graders — K=0 solid, K=5 dashed",
                 fontsize=10, y=1.02)
    p = C.save_fig(fig, f"{SCRIPT}_fig")
    print("figure ->", p)

    # 3b. single-column variant for the paper body: the SAME two panels stacked (shared x), sized
    #     for a 3.4-in ACL column; separate y-axes as above (the graders sit on different levels).
    #     Deliberately NOT a ledger entry — out/crossgen_exp1.json is layout-agnostic and must stay
    #     byte-identical whichever variant the .tex includes.
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True)
    for ax, (df, title) in zip(axes, ((g35, "GPT-3.5 (original ICLR oracle)"),
                                      (cg, "gpt-4o-mini (Exp3 oracle, same transcripts)"))):
        for arm, k in (("L0", 0), ("L5", 5)):
            ys, lo, hi = [], [], []
            for it in ITERS:
                v = df.loc[df["model"] == f"Exp1_LA{k}_I{it}", "Final"].dropna().to_numpy()
                ys.append(v.mean())
                rng = np.random.default_rng(0)
                b = rng.choice(v, size=(2000, v.size), replace=True).mean(axis=1)
                lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
            st = C.K_STYLE[k]
            ax.plot(ITERS, ys, ls=st["ls"], marker=st["marker"], ms=4.5, lw=1.4, color=col[arm],
                    label=f"PTO K={k}")
            ax.fill_between(ITERS, lo, hi, color=col[arm], alpha=0.15, lw=0)
        base = df.loc[df["model"] == "Exp1_Base", "Final"].mean()
        ax.axhline(base, ls=":", lw=1.3, color="#555555", label="Base (Llama-2-7B)")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(ITERS)
        ax.set_ylabel("Final = mean(Q1, Q2)  [1-5]", fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("PTO iteration", fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    lab = [l + ("  (bands: 95% bootstrap CI)" if l.startswith("PTO K=5") else "") for l in lab]
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.06), columnspacing=1.2, handlelength=2.2)
    fig.suptitle("Exp1 (ICLR 2025) conversations under two graders\nK=0 solid, K=5 dashed",
                 fontsize=8.5)
    p_col = C.save_fig(fig, f"{SCRIPT}_fig_col")
    print("figure ->", p_col)
    L.put("fig.caption", (
        "Exp1 (ICLR 2025; Llama-2-7B therapist, GPT-3.5 patient) PTO models by iteration, Final = mean(Q1,Q2) "
        "averaged over the 96 conversations (one per patient permutation), K=0 solid circles vs K=5 dashed "
        "squares, untrained Base as the dotted line, bands = 95% percentile-bootstrap CI of the mean. Left: the "
        "original GPT-3.5 oracle (ICLR Table 1); right: the SAME transcripts re-scored by the Exp3 oracle "
        "(gpt-4o-mini-2024-07-18, V5 Q1+Q2). Separate y-axes per panel because the graders sit on different "
        "levels (gpt-4o-mini reads ~0.19-0.43 higher; never averaged). Paired statistics for the gap are in "
        "tables/crossgen_exp1_kcontrast.md (delta = K0 - K5, + => K=0 higher, paired on conversation index). "
        "Both arms complete 7 iterations (no censoring)."),
        source=f"figures/{SCRIPT}_fig.png")

    # 4. LookAhead_3 (K=3): on disk, unscored by the tool; report what it would take -------
    la3 = la3_cost_estimate()
    la3_g35 = load_gpt35({m: ("L3", i + 1, d) for i, (m, d) in enumerate(LA3_DIRS.items())})
    la3_lv = (la3_g35.groupby(["model", "iteration"])[["Q1", "Q2", "Final"]].mean()
              .reset_index())
    la3_lv["n_gpt35"] = la3_g35.groupby("model").size().values
    la3_lv["scored_by_gpt4omini"] = False
    la3_lv["hyperparams"] = "TT0.7 / Filter(tau)0.2 / 'FullEval'  (K=0,5 sweep: TT0.9 / tau 0.1)"
    C.save_table(la3_lv, f"{SCRIPT}_la3_gpt35", caption=(
        "Exp1's K=3 sweep (LookAhead_3, 4 iterations, 96 conversations each) — the only look-ahead 'dose' "
        "data on disk. NOT re-scored by gpt-4o-mini: score_crossgen.py deliberately excludes it (its "
        "manifest is the paper's K=0/K=5 pair only, and K=3 ran with different hyper-parameters — therapist "
        "temperature 0.7 and filter tau 0.2 vs 0.9 / 0.1 — so it is not a matched dose arm). Shown here "
        "are its ORIGINAL GPT-3.5 oracle means from the on-disk scores_i.csv (Q1, Q2, Final = mean). "
        "Not comparable to the K=0/K=5 rows of crossgen_exp1_levels.md without that caveat."))
    L.put("la3.status", {"scored_by_gpt4omini": False, "n_iterations": 4,
                         "tool_supports_it": False,
                         "reason": "score_crossgen.py --gen has choices {all,exp1,exp2}; exp1_models() "
                                   "excludes LookAhead_3 (different hyperparameters: TT0.7, tau=0.2)",
                         "what_it_would_take": "add the 4 FullEval_TTree1.4_TT0.7_TP0.7_TE0.2_Filter0.2_V{1..4}.0 "
                                               "dirs to exp1_models() as Exp1_LA3_I{1..4}, run --dry-run then score; "
                                               "estimated cost below (no API call was made here)",
                         **la3},
          source=f"tables/{SCRIPT}_la3_gpt35.md + in-script estimate mirroring score_crossgen.dry_run",
          note="cost = list-price gpt-4o-mini, 50% cached 1084-tok prefix, ~180 output tok/call")
    for _, r in la3_lv.iterrows():
        L.put(f"la3.gpt35.{r['model']}", {k: r[k] for k in ("Q1", "Q2", "Final", "n_gpt35")},
              source=f"tables/{SCRIPT}_la3_gpt35.md row model={r['model']}")
    print(f"LA3 estimate: {la3}")

    # 5. headline verdicts into the ledger --------------------------------------------------
    for grader in ("gpt-4o-mini", "gpt-3.5"):
        f = kc[(kc["grader"] == grader) & (kc["metric"] == "Final")]
        L.put(f"verdict.{grader}.Final",
              {"n_iters_K5_higher": int((f["mean_delta"] < 0).sum()),
               "n_iters_K5_higher_p_holm_lt_05": int(((f["mean_delta"] < 0) & (f["p_holm"] < 0.05)).sum()),
               "n_iters_K0_higher_p_holm_lt_05": int(((f["mean_delta"] > 0) & (f["p_holm"] < 0.05)).sum()),
               "mean_of_7_deltas": float(f["mean_delta"].mean()),
               "median_dz": float(f["dz"].median()),
               "iter_deltas": {int(r["iteration"]): r["mean_delta"] for _, r in f.iterrows()}},
              source=f"tables/{SCRIPT}_kcontrast.md rows grader={grader} metric=Final",
              note="delta = K0 - K5; negative => K=5 higher")
    L.save()
    print("ledger ->", C.OUT / f"{SCRIPT}.json")


if __name__ == "__main__":
    main()
