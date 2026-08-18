"""crossgen.py — the replication link to the ICLR 2025 look-ahead paper (Exp1 re-scored by the Exp3 oracle).

**What it computes.** The ICLR poster (Exp1: Llama-2-7B therapist, GPT-3.5 patient + oracle, PTO
K in {0, 5}, 7 iterations, 96 patient permutations) found K=5 clearly ahead of K=0. Exp3
(Llama-3.2-1B, gpt-4o-mini) finds K=5 never leading. Both the *model* AND the *grader* changed
between the two, so the reversal is confounded. ``eda/tools/score_crossgen.py`` re-scored the very
same Exp1 conversations with the Exp3 oracle (gpt-4o-mini-2024-07-18 @ T=0.1, V5 JSON-schema
Q1 + Q2) into ``data/eval_scores/_crossgen/`` (kept OUT of the Exp3 score lake proper). This
module is the analysis of that re-score, one grader beside the other, never averaged:

1. :func:`levels` — per Exp1 model state, Q1 / Q2 / Final under gpt-4o-mini beside the ICLR
   Table 1 GPT-3.5 means (transcribed, :data:`ICLR_TABLE1`) AND the on-disk GPT-3.5
   per-conversation scores (:func:`load_exp1_gpt35`).
2. :func:`k_contrast` — K=0 minus K=5 per iteration under BOTH graders; :func:`k_summary` —
   best-vs-best (ICLR's pick and each grader's own), the pooled-arm contrast, the "every L5 above
   every L0" ordering claim (:func:`ordering_claims`); :func:`grader_agreement` — Spearman /
   Pearson between the graders' 15 model means and per conversation; :func:`vs_base` — each
   trained model vs Base under both graders.
3. :func:`la3_gpt35` / :func:`la3_cost_estimate` — the K=3 sweep on disk (4 iterations,
   DIFFERENT hyper-parameters) is not scored by the tool; report its GPT-3.5 means + what scoring
   it would cost. NO API calls.
4. :func:`crossgen_numbers` — every quotable number, ``{dotted.key: {value, source, note}}``,
   for ``exports.save_numbers``.

Figures live in :mod:`eda_analysis.plotting.crossgen` (``crossgen_fig(levels, layout=...)``).

**Sign convention** everywhere: ``delta = K0 - K5``; **+ => K=0 higher** (mirrors
:func:`eda_analysis.stats.paired_k_comparison`). vs-Base rows: **+ => trained model higher**.

**Pairing unit:** Exp1 conversation index ``i`` == patient permutation ``i``. Exp1 wrote
``conversation_i.csv`` from ``permutations[i]`` of a deterministic nested-loop generator
(``system_prompts_builder.generate_all_permutations``) with NO shuffle, so index ``i`` is persona
``i`` in every model dir — unlike Exp3, where the 96 personas are reshuffled per iteration.
:func:`persona_alignment_check` verifies this empirically from the patients' opening lines. n = 96
pairs per contrast.

**Censoring:** none — both Exp1 arms complete 7 iterations. Iteration 0 = the untrained
Llama-2-7B base, a SINGLE draw shared by both arms (Exp3, by contrast, has two independent base
draws per method).

**Caveats kept from the paper generator.**
- GPT-3.5's paired deltas are heavy-tailed (a few 2-3 point collapses of one side), so its Wilcoxon
  ``p`` and the bootstrap CI can disagree — read the sign split (``n_K0_higher`` / ``n_K5_higher``);
  the paired-t ``p_t`` and the unpaired Welch reading are reported beside them for that reason.
- The two graders sit on different levels (gpt-4o-mini reads ~0.19-0.43 higher on Final); they are
  never averaged, and the figure gives each its own y-axis.
- ``Final`` = mean(Q1_Mean, Q2_Mean) per conversation then averaged — exactly the lake's ``Q1Q2``
  composite and the ICLR paper's "Final Score". The on-disk GPT-3.5 per-conversation scores
  reproduce ICLR Table 1 to 3 dp (:func:`table1_crosscheck`, max |diff| < 0.0015).
- The K=3 sweep (``LookAhead_3``) ran at therapist temperature 0.7 / filter tau 0.2 (the K=0/5
  pair: 0.9 / 0.1) — not a matched dose arm; its rows are NOT comparable to :func:`levels`.
- Bootstrap CIs use :data:`eda_analysis.constants.BOOT_SEED` (the package seed) where the paper
  generator used seed 0, so CI bounds may differ from the frozen fixture in the third decimal;
  means / dz / p are identical.

**Provenance.** Promoted 2026-08-18 from
``papers/2026_lookahead_pto_grpo/analysis/crossgen_exp1.py`` (the paper's generator; its frozen
outputs ``tables/crossgen_exp1_*.csv`` + ``analysis/out/crossgen_exp1.json`` are the fixture this
module reproduces). Contract as everywhere in the analysis layer: return tidy frames / dicts, NO
disk writes — the notebook (``lookahead/replication.ipynb``) owns ``exports.*``.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from .constants import EVAL_SCORES, PRIMARY_JUDGE_TAG, WORKSPACE_ROOT
from .stats import holm, paired_arrays

__all__ = [
    "ICLR_TABLE1", "ITERS", "N_PERSONAS", "EXP1_DIR", "CROSSGEN_ROOT", "GRADER_GPT4OMINI",
    "GRADER_GPT35",
    "exp1_manifest", "la3_manifest", "load_crossgen", "load_exp1_gpt35",
    "persona_alignment_check", "table1_crosscheck",
    "paired_models", "unpaired_delta", "k_contrast_table", "vs_base_table", "pooled_arm_contrast",
    "levels", "k_contrast", "k_summary", "ordering_claims", "grader_agreement", "vs_base",
    "la3_gpt35", "la3_cost_estimate", "crossgen_all", "crossgen_numbers",
]

# ── locations ────────────────────────────────────────────────────────────────
# Exp1 lives beside Exp3 under the repo root (WORKSPACE_ROOT = Exp3_PTO_GRPO/).
_REPO = os.path.dirname(WORKSPACE_ROOT)
EXP1_DIR = os.path.join(_REPO, "Exp1_ICLR2025", "data", "conversations_eval")
CROSSGEN_ROOT = os.path.join(EVAL_SCORES, "_crossgen")   # judge=<tag>/rep=0/metric=<M>/oracle=<O>/<Model>/<i>.csv
ORACLE_TOKEN = "Q1Q2"
N_PERSONAS = 96
ITERS = list(range(1, 8))
GRADER_GPT4OMINI = "gpt-4o-mini"      # the Exp3 oracle re-scoring Exp1's transcripts
GRADER_GPT35 = "gpt-3.5"              # the original ICLR oracle (on-disk per-conversation scores)
_ARM_DIR = "TTree1.4_TT0.9_TP0.7_TE0.2_V{}"
_METRICS = ("Final", "Q1", "Q2")

# The ICLR SSI-FM poster's Table 1 (Llama-2-7B, GPT-3.5 patient+oracle; 7 iterations), transcribed
# from papers/2025_iclr_pto_lookahead/submitted/paper.pdf. Mean scores only. Columns: arm
# (Base/L0/L5), iteration (0 = Base), Q1, Q2, Final (= mean of the Q1 and Q2 means).
ICLR_TABLE1 = pd.DataFrame([
    ("Base", 0, 3.521, 3.385, 3.453),
    ("L0", 1, 3.863, 3.452, 3.657), ("L0", 2, 3.750, 3.435, 3.593), ("L0", 3, 3.796, 3.567, 3.682),
    ("L0", 4, 3.969, 3.585, 3.777), ("L0", 5, 3.744, 3.478, 3.611), ("L0", 6, 3.794, 3.494, 3.644),
    ("L0", 7, 3.677, 3.452, 3.565),
    ("L5", 1, 3.898, 3.523, 3.710), ("L5", 2, 3.969, 3.618, 3.794), ("L5", 3, 4.050, 3.683, 3.866),
    ("L5", 4, 3.981, 3.605, 3.793), ("L5", 5, 4.225, 3.660, 3.942), ("L5", 6, 4.112, 3.656, 3.884),
    ("L5", 7, 4.190, 3.775, 3.982),
], columns=["arm", "iteration", "Q1", "Q2", "Final"])


# ── the Exp1 model manifest (mirrors tools/score_crossgen.exp1_models exactly) ───────────

def exp1_manifest() -> Dict[str, Tuple[str, int, str]]:
    """``model -> (arm label 'Base'/'L0'/'L5', iteration, conversation dir)``.

    The paper's Base is ``Basic_50_TT0.9_TP0.7_TE0.2_V2`` (Final 3.453, Table 1). The
    ``Q2_``-prefixed, ``_OLD`` and ``_FAIL_Q2`` directories are separate/abandoned sweeps and are
    deliberately excluded, as is ``LookAhead_3`` (see :func:`la3_manifest`).
    """
    m = {"Exp1_Base": ("Base", 0, os.path.join(EXP1_DIR, "Base", "Basic_50_TT0.9_TP0.7_TE0.2_V2"))}
    for k in (0, 5):
        for i in ITERS:
            m[f"Exp1_LA{k}_I{i}"] = (f"L{k}", i, os.path.join(EXP1_DIR, f"LookAhead_{k}", _ARM_DIR.format(i)))
    return m


def la3_manifest() -> Dict[str, Tuple[str, int, str]]:
    """The K=3 sweep: 4 iterations at DIFFERENT hyper-parameters (TT0.7 / tau 0.2 / 'FullEval')."""
    return {f"Exp1_LA3_I{i}": ("L3", i, os.path.join(
        EXP1_DIR, "LookAhead_3", f"FullEval_TTree1.4_TT0.7_TP0.7_TE0.2_Filter0.2_V{i}.0"))
        for i in range(1, 5)}


# ── loaders ──────────────────────────────────────────────────────────────────

def load_crossgen(judge_tag: str = PRIMARY_JUDGE_TAG, *, oracle_token: str = ORACLE_TOKEN,
                  manifest: Optional[dict] = None, rep: int = 0) -> pd.DataFrame:
    """Long frame of the Exp3-oracle re-score of Exp1: ``model, arm, iteration, conv_index, Q1, Q2, Final``.

    Reads ``data/eval_scores/_crossgen/judge=<judge_tag>/rep=<rep>/metric=Q1|Q2/oracle=<O>/<model>/<i>.csv``
    — one CSV per (metric, model, conversation) holding the item scores + ``Q1_Mean`` / ``Q2_Mean``.
    ``i`` is the Exp1 conversation index (``conversation_i.csv``), i.e. patient permutation ``i``.
    ``Final = mean(Q1_Mean, Q2_Mean)`` — exactly the lake's ``Q1Q2`` composite and the ICLR paper's
    "Final Score". Only conversations scored on BOTH metrics are kept.
    """
    manifest = manifest or exp1_manifest()
    root = os.path.join(CROSSGEN_ROOT, f"judge={judge_tag or PRIMARY_JUDGE_TAG}", f"rep={int(rep)}")
    rows = []
    for model, (arm, it, _) in manifest.items():
        per_metric = {}
        for metric in ("Q1", "Q2"):
            d = os.path.join(root, f"metric={metric}", f"oracle={oracle_token}", model)
            vals = {}
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    if not fn.endswith(".csv"):
                        continue
                    i = int(os.path.splitext(fn)[0])
                    s = pd.read_csv(os.path.join(d, fn))
                    vals[i] = float(s[f"{metric}_Mean"].iloc[0])
            per_metric[metric] = vals
        for i in sorted(set(per_metric["Q1"]) & set(per_metric["Q2"])):
            q1, q2 = per_metric["Q1"][i], per_metric["Q2"][i]
            rows.append(dict(model=model, arm=arm, iteration=it, conv_index=i,
                             Q1=q1, Q2=q2, Final=(q1 + q2) / 2.0))
    return pd.DataFrame(rows, columns=["model", "arm", "iteration", "conv_index", "Q1", "Q2", "Final"])


def load_exp1_gpt35(manifest: Optional[dict] = None) -> pd.DataFrame:
    """The ORIGINAL GPT-3.5 per-conversation oracle scores Exp1 saved beside each conversation.

    ``scores_i.csv``: ``scores1_avg`` = Q1 mean, ``scores2_avg`` = Q2 mean, ``scores_avg`` = Final.
    These are the numbers ICLR Table 1 averaged (see :func:`table1_crosscheck`). Same columns as
    :func:`load_crossgen` so the two graders can be handled symmetrically.
    """
    manifest = manifest or exp1_manifest()
    rows = []
    for model, (arm, it, d) in manifest.items():
        for i in range(N_PERSONAS):
            p = os.path.join(d, f"scores_{i}.csv")
            if not os.path.exists(p):
                continue
            s = pd.read_csv(p)
            rows.append(dict(model=model, arm=arm, iteration=it, conv_index=i,
                             Q1=float(s["scores1_avg"].iloc[0]),
                             Q2=float(s["scores2_avg"].iloc[0]),
                             Final=float(s["scores_avg"].iloc[0])))
    return pd.DataFrame(rows, columns=["model", "arm", "iteration", "conv_index", "Q1", "Q2", "Final"])


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


def persona_alignment_check(manifest: Optional[dict] = None) -> dict:
    """Verify the pairing unit: for each conversation index ``i``, the (age, problem) fingerprint of
    the patient's opening line must agree across all model states (where parseable).

    Exp1 generated ``conversation_i`` from ``permutations[i]`` of a deterministic nested-loop
    generator with NO shuffle, so index ``i`` is persona ``i`` in every model dir. Returns
    ``{n_indices, n_consistent, n_conflicting}``; "conflict" = two DIFFERENT parsed values at the
    same index (missing parses are ignored). Reads 15 x 96 conversation CSVs — slow-ish.
    """
    manifest = manifest or exp1_manifest()
    sig: Dict[int, list] = {}
    for _, (_, _, d) in manifest.items():
        for i in range(N_PERSONAS):
            p = os.path.join(d, f"conversation_{i}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                if len(df) > 1:
                    sig.setdefault(i, []).append(_persona_signature(str(df["conversation"].iloc[1])))
    n_ok = n_conf = 0
    for _, sigs in sig.items():
        ages = {a for a, _ in sigs if a is not None}
        probs = {p for _, p in sigs if p is not None}
        if len(ages) > 1 or len(probs) > 1:
            n_conf += 1
        else:
            n_ok += 1
    return {"n_indices": len(sig), "n_consistent": n_ok, "n_conflicting": n_conf}


def table1_crosscheck(gpt35: pd.DataFrame, table1: pd.DataFrame = ICLR_TABLE1) -> float:
    """max |ICLR Table 1 - on-disk GPT-3.5 mean| over 15 model states x {Q1, Q2, Final}.

    The paper generator asserted ``< 0.0015`` (the transcription and the on-disk scores agree to
    3 dp). Returned, not asserted, so the notebook can print/ledger it and decide.
    """
    means = gpt35.groupby(["arm", "iteration"])[["Q1", "Q2", "Final"]].mean().reset_index()
    chk = table1.merge(means, on=["arm", "iteration"], suffixes=("_tab1", "_disk"))
    return max(float((chk[f"{m}_tab1"] - chk[f"{m}_disk"]).abs().max()) for m in ("Q1", "Q2", "Final"))


# ── statistics ───────────────────────────────────────────────────────────────

def _wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """conv_index x model matrix of one metric (NaN where unscored)."""
    return df.pivot_table(index="conv_index", columns="model", values=metric, aggfunc="mean")


def _paired_arrays_ext(a, b) -> dict:
    """:func:`stats.paired_arrays` (Wilcoxon p, dz, bootstrap CI) + a paired-t p and the sign split
    of the deltas. The extras exist because GPT-3.5's deltas are heavy-tailed (a few 2-3 point
    collapses), so the Wilcoxon p and the bootstrap CI can disagree; the sign split says which read
    is fair."""
    out = paired_arrays(a, b)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    out["p_t"] = float(sps.ttest_rel(a[ok], b[ok]).pvalue) if d.size >= 3 else np.nan
    out["n_a_higher"] = int((d > 0).sum())
    out["n_b_higher"] = int((d < 0).sum())
    return out


def paired_models(df: pd.DataFrame, metric: str, model_a: str, model_b: str) -> dict:
    """``model_a - model_b`` on *metric*, paired on ``conv_index``. ``+ => model_a higher``.
    Returns ``n, mean_delta, dz, ci_lo, ci_hi, p, p_t, n_a_higher, n_b_higher``."""
    w = _wide(df, metric)
    if model_a not in w or model_b not in w:
        return dict(n=0, mean_delta=np.nan, dz=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
                    p_t=np.nan, n_a_higher=0, n_b_higher=0)
    return _paired_arrays_ext(w[model_a].to_numpy(), w[model_b].to_numpy())


def unpaired_delta(df: pd.DataFrame, metric: str, model_a: str, model_b: str) -> dict:
    """Difference of means + Welch t (the UNPAIRED reading, reported beside the paired one)."""
    a = df.loc[df["model"] == model_a, metric].dropna().to_numpy()
    b = df.loc[df["model"] == model_b, metric].dropna().to_numpy()
    if len(a) < 3 or len(b) < 3:
        return dict(delta_unpaired=np.nan, p_welch=np.nan)
    t = sps.ttest_ind(a, b, equal_var=False)
    return dict(delta_unpaired=float(a.mean() - b.mean()), p_welch=float(t.pvalue))


def k_contrast_table(df: pd.DataFrame, grader: str) -> pd.DataFrame:
    """K0 - K5 at every iteration for Final, Q1, Q2 under ONE grader; Holm within (grader, metric)
    across the 7 iterations. Paired on ``conv_index``. ``+ => K=0 higher``."""
    rows = []
    for metric in _METRICS:
        for it in ITERS:
            a, b = f"Exp1_LA0_I{it}", f"Exp1_LA5_I{it}"
            r = paired_models(df, metric, a, b)
            r.update(unpaired_delta(df, metric, a, b))
            rows.append(dict(grader=grader, metric=metric, iteration=it, model_K0=a, model_K5=b, **r))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, g in out.groupby("metric"):
        out.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    return out


def vs_base_table(df: pd.DataFrame, grader: str) -> pd.DataFrame:
    """Every trained model - Base (Final, Q1, Q2) under ONE grader, paired on ``conv_index``; Holm
    within (grader, arm, metric) across the 7 iterations. ``+ => trained model higher``."""
    rows = []
    for metric in _METRICS:
        for arm, k in (("L0", 0), ("L5", 5)):
            for it in ITERS:
                m = f"Exp1_LA{k}_I{it}"
                r = paired_models(df, metric, m, "Exp1_Base")
                rows.append(dict(grader=grader, arm=arm, metric=metric, iteration=it, model=m, **r))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, g in out.groupby(["arm", "metric"]):
        out.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    return out


def pooled_arm_contrast(df: pd.DataFrame, grader: str) -> pd.DataFrame:
    """Per-persona mean over the 7 iterations of each arm, then K0 - K5 paired on ``conv_index`` —
    the arm-level reading of "L5 models score higher than L0 models". ``+ => K=0 higher``."""
    rows = []
    for metric in _METRICS:
        per = (df[df["arm"].isin(["L0", "L5"])]
               .groupby(["arm", "conv_index"])[metric].mean().unstack("arm"))
        r = _paired_arrays_ext(per["L0"].to_numpy(), per["L5"].to_numpy())
        rows.append(dict(grader=grader, metric=metric, contrast="mean over iters 1-7, K0 - K5", **r))
    return pd.DataFrame(rows)


# ── the paper tables ─────────────────────────────────────────────────────────

_LEVELS_COLS = ["model", "arm", "K", "iteration",
                "n_gpt4omini", "Q1_gpt4omini", "Q2_gpt4omini", "Final_gpt4omini", "Final_sd_gpt4omini",
                "n_gpt35", "Q1_gpt35", "Q2_gpt35", "Final_gpt35", "Final_sd_gpt35",
                "Q1_iclr_tab1", "Q2_iclr_tab1", "Final_iclr_tab1", "Final_gap_gpt4omini_minus_gpt35"]


def levels(crossgen: pd.DataFrame, gpt35: pd.DataFrame, *, table1: pd.DataFrame = ICLR_TABLE1,
           manifest: Optional[dict] = None) -> pd.DataFrame:
    """Per Exp1 model state (Base, K=0 iters 1-7, K=5 iters 1-7): n / Q1 / Q2 / Final / Final SD
    under gpt-4o-mini (``*_gpt4omini``) and under the original GPT-3.5 oracle (``*_gpt35``), plus
    ICLR Table 1 as printed (``*_iclr_tab1``) and ``Final_gap_gpt4omini_minus_gpt35``. Two graders
    side by side, never averaged. Reproduces ``crossgen_exp1_levels``."""
    manifest = manifest or exp1_manifest()

    def _lv(df, tag):
        g = df.groupby("model")
        return pd.DataFrame({
            f"n_{tag}": g.size(),
            f"Q1_{tag}": g["Q1"].mean(), f"Q2_{tag}": g["Q2"].mean(),
            f"Final_{tag}": g["Final"].mean(), f"Final_sd_{tag}": g["Final"].std(ddof=1),
        })
    lv = _lv(crossgen, "gpt4omini").join(_lv(gpt35, "gpt35"), how="outer")
    lv["arm"] = [manifest[m][0] for m in lv.index]
    lv["iteration"] = [manifest[m][1] for m in lv.index]
    lv["K"] = lv["arm"].map({"Base": "", "L0": "0", "L5": "5"})
    lv = lv.reset_index().rename(columns={"index": "model"})
    t1x = table1.rename(columns={"Q1": "Q1_iclr_tab1", "Q2": "Q2_iclr_tab1", "Final": "Final_iclr_tab1"})
    lv = lv.merge(t1x, on=["arm", "iteration"], how="left")
    lv["Final_gap_gpt4omini_minus_gpt35"] = lv["Final_gpt4omini"] - lv["Final_gpt35"]
    lv = lv.sort_values(["K", "iteration"])
    return lv[_LEVELS_COLS].reset_index(drop=True)


def k_contrast(crossgen: pd.DataFrame, gpt35: pd.DataFrame) -> pd.DataFrame:
    """The look-ahead contrast on Exp1's conversations at matched iteration under BOTH graders:
    ``mean_delta = score(K=0 model) - score(K=5 model)``, **+ => K=0 higher**. Paired on
    conversation index (n = 96); dz; 95% percentile-bootstrap CI; Wilcoxon ``p``; ``p_holm`` within
    each (grader, metric) family across the 7 iterations; ``p_t`` = paired t (uncorrected);
    ``n_K0_higher`` / ``n_K5_higher`` = sign split of the 96 deltas (ties excluded);
    ``delta_unpaired`` / ``p_welch`` = the unpaired reading. Reproduces ``crossgen_exp1_kcontrast``."""
    kc = pd.concat([k_contrast_table(crossgen, GRADER_GPT4OMINI), k_contrast_table(gpt35, GRADER_GPT35)],
                   ignore_index=True)
    kc = kc.rename(columns={"n_a_higher": "n_K0_higher", "n_b_higher": "n_K5_higher"})
    kc = kc[["grader", "metric", "iteration", "model_K0", "model_K5", "n", "mean_delta", "dz",
             "ci_lo", "ci_hi", "p", "p_holm", "p_t", "n_K0_higher", "n_K5_higher",
             "delta_unpaired", "p_welch"]]
    return kc.sort_values(["grader", "metric", "iteration"]).reset_index(drop=True)


def ordering_claims(crossgen: pd.DataFrame, gpt35: pd.DataFrame) -> pd.DataFrame:
    """One row per grader: each grader's own best K=0 / K=5 model (by that grader's Final mean) and
    the ICLR text's group claim checked on model MEANS — ``every_L5_above_every_L0`` =
    min(L5 means) > max(L0 means), the count of L5 models above the best L0, how many of the 49
    (L5, L0) mean pairs have L5 higher, how many of the 7 matched iterations, and how many of the 14
    trained models sit above Base."""
    rows = []
    for grader, df in ((GRADER_GPT4OMINI, crossgen), (GRADER_GPT35, gpt35)):
        means = df.groupby("model")["Final"].mean()
        b0 = max((f"Exp1_LA0_I{i}" for i in ITERS), key=lambda m: means[m])
        b5 = max((f"Exp1_LA5_I{i}" for i in ITERS), key=lambda m: means[m])
        l0 = means[[f"Exp1_LA0_I{i}" for i in ITERS]]
        l5 = means[[f"Exp1_LA5_I{i}" for i in ITERS]]
        rows.append(dict(
            grader=grader, best_K0=b0, best_K5=b5,
            best_K0_final=float(means[b0]), best_K5_final=float(means[b5]),
            min_L5_final=float(l5.min()), max_L0_final=float(l0.max()),
            every_L5_above_every_L0=bool(l5.min() > l0.max()),
            n_L5_models_above_max_L0=int((l5 > l0.max()).sum()),
            n_of_49_L5xL0_pairs_with_L5_higher=int(sum(v5 > v0 for v5 in l5 for v0 in l0)),
            n_of_7_iterations_L5_higher=int(sum(l5.iloc[i] > l0.iloc[i] for i in range(7))),
            n_of_14_trained_models_above_base=int((means.drop("Exp1_Base") > means["Exp1_Base"]).sum()),
        ))
    return pd.DataFrame(rows)


def k_summary(crossgen: pd.DataFrame, gpt35: pd.DataFrame) -> pd.DataFrame:
    """Summary contrasts under each grader (never averaged across graders): the ICLR best-vs-best
    (L0_I4 - L5_I7), each grader's own best-vs-best, the pooled-arm contrast (mean over iters 1-7
    per persona, then K0 - K5) and the ordering row (min(L5 mean) - max(L0 mean); its
    ``n_K0_higher`` / ``n_K5_higher`` count the 49 (K=0, K=5) MEAN pairs instead of paired
    deltas). **+ => first-named (K=0) higher.** Reproduces ``crossgen_exp1_kcontrast_summary``."""
    summ = []
    for grader, df in ((GRADER_GPT4OMINI, crossgen), (GRADER_GPT35, gpt35)):
        means = df.groupby("model")["Final"].mean()
        for metric in _METRICS:                                    # (a) ICLR's pick
            r = paired_models(df, metric, "Exp1_LA0_I4", "Exp1_LA5_I7")
            summ.append(dict(grader=grader, contrast="ICLR best-vs-best: L0_I4 - L5_I7", metric=metric, **r))
        b0 = max((f"Exp1_LA0_I{i}" for i in ITERS), key=lambda m: means[m])   # (b) own best
        b5 = max((f"Exp1_LA5_I{i}" for i in ITERS), key=lambda m: means[m])
        for metric in _METRICS:
            r = paired_models(df, metric, b0, b5)
            summ.append(dict(grader=grader, metric=metric,
                             contrast=f"own best-vs-best: {b0.split('_', 1)[1]} - {b5.split('_', 1)[1]}", **r))
        for _, r in pooled_arm_contrast(df, grader).iterrows():   # (c) pooled arms
            summ.append(dict(grader=grader, contrast=r["contrast"], metric=r["metric"],
                             **{k: r[k] for k in ("n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_t",
                                                  "n_a_higher", "n_b_higher")}))
        l0 = means[[f"Exp1_LA0_I{i}" for i in ITERS]]             # (d) ordering claim
        l5 = means[[f"Exp1_LA5_I{i}" for i in ITERS]]
        n_pairs_l5_gt_l0 = int(sum(v5 > v0 for v5 in l5 for v0 in l0))
        summ.append(dict(grader=grader, contrast="ordering: min(L5 mean) - max(L0 mean)", metric="Final",
                         n=7, mean_delta=float(l5.min() - l0.max()), dz=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                         p=np.nan, p_t=np.nan, n_a_higher=49 - n_pairs_l5_gt_l0, n_b_higher=n_pairs_l5_gt_l0))
    summ = pd.DataFrame(summ).rename(columns={"n_a_higher": "n_K0_higher", "n_b_higher": "n_K5_higher"})
    summ = summ[["grader", "contrast", "metric", "n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_t",
                 "n_K0_higher", "n_K5_higher"]]
    summ["note"] = np.where(summ["contrast"].str.startswith("ordering"),
                            "L5-minus-L0 of MEANS: + => every K=5 model above every K=0 model",
                            "+ => first-named (K=0) higher")
    return summ.reset_index(drop=True)


def grader_agreement(levels_df: pd.DataFrame, crossgen: pd.DataFrame, gpt35: pd.DataFrame) -> pd.DataFrame:
    """Agreement between the two graders on Exp1's conversations: Spearman rho (and Pearson r)
    between the gpt-4o-mini re-score and the original GPT-3.5 oracle at the level of the 15
    model-state means (Base + 7 + 7), the 14 trained-model means (so the base does not anchor the
    rank), and per conversation (15 x 96 pooled). Reproduces ``crossgen_exp1_grader_agreement``."""
    rows = []
    lv_idx = levels_df.set_index("model")
    tr = [m for m in lv_idx.index if m != "Exp1_Base"]
    m = crossgen.merge(gpt35, on=["model", "conv_index"], suffixes=("_4m", "_35"))
    for metric in _METRICS:
        a = lv_idx[f"{metric}_gpt4omini"]; b = lv_idx[f"{metric}_gpt35"]
        rho, p = sps.spearmanr(a, b)
        pr, _ = sps.pearsonr(a, b)
        rho14, p14 = sps.spearmanr(a[tr], b[tr])
        rc, pc = sps.spearmanr(m[f"{metric}_4m"], m[f"{metric}_35"])
        rows.append(dict(metric=metric, level="15 model means", spearman_rho=float(rho), p=float(p),
                         pearson_r=float(pr), n=15))
        rows.append(dict(metric=metric, level="14 trained model means", spearman_rho=float(rho14),
                         p=float(p14), pearson_r=float(sps.pearsonr(a[tr], b[tr])[0]), n=14))
        rows.append(dict(metric=metric, level="per conversation (pooled)", spearman_rho=float(rc),
                         p=float(pc), pearson_r=float(sps.pearsonr(m[f"{metric}_4m"], m[f"{metric}_35"])[0]),
                         n=int(len(m))))
    return pd.DataFrame(rows)


def vs_base(crossgen: pd.DataFrame, gpt35: pd.DataFrame) -> pd.DataFrame:
    """Each trained Exp1 model minus the untrained Llama-2-7B Base under each grader (side by side,
    never averaged): ``mean_delta = model - Base``, **+ => trained model higher**; paired on
    conversation index (n = 96); dz, bootstrap CI, Wilcoxon p, Holm within each (grader, arm,
    metric) family across the 7 iterations; ``p_t`` = paired t; ``n_model_higher`` /
    ``n_base_higher`` = sign split. Reproduces ``crossgen_exp1_vsbase``."""
    vb = pd.concat([vs_base_table(crossgen, GRADER_GPT4OMINI), vs_base_table(gpt35, GRADER_GPT35)],
                   ignore_index=True)
    vb = vb.rename(columns={"n_a_higher": "n_model_higher", "n_b_higher": "n_base_higher"})
    return vb[["grader", "arm", "metric", "iteration", "model", "n", "mean_delta", "dz", "ci_lo", "ci_hi",
               "p", "p_holm", "p_t", "n_model_higher", "n_base_higher"]]


# ── K=3 (LookAhead_3): on disk, unscored by the tool ─────────────────────────

def la3_gpt35() -> pd.DataFrame:
    """Exp1's K=3 sweep (4 iterations, 96 conversations each) — the only look-ahead 'dose' data on
    disk — as its ORIGINAL GPT-3.5 oracle means (Q1, Q2, Final) from the on-disk ``scores_i.csv``.
    NOT re-scored by gpt-4o-mini: ``score_crossgen.py`` deliberately excludes it (its manifest is
    the paper's K=0/K=5 pair only, and K=3 ran with different hyper-parameters — therapist
    temperature 0.7 and filter tau 0.2 vs 0.9 / 0.1 — so it is not a matched dose arm). Not
    comparable to :func:`levels` without that caveat. Reproduces ``crossgen_exp1_la3_gpt35``."""
    g = load_exp1_gpt35(la3_manifest())
    lv = g.groupby(["model", "iteration"])[["Q1", "Q2", "Final"]].mean().reset_index()
    lv["n_gpt35"] = g.groupby("model").size().values
    lv["scored_by_gpt4omini"] = False
    lv["hyperparams"] = "TT0.7 / Filter(tau)0.2 / 'FullEval'  (K=0,5 sweep: TT0.9 / tau 0.1)"
    return lv


def la3_cost_estimate() -> dict:
    """What re-scoring the 4 LookAhead_3 dirs would cost — mirrors ``score_crossgen.dry_run``
    WITHOUT importing the tool or calling any API: the 1,084-token rubric-first prefix caches at
    50%; transcripts billed at full rate; ~180 output tokens per call; 2 calls (Q1 + Q2) per
    conversation. Prices = the tool's gpt-4o-mini list rates (USD / 1M tok: 0.150 in, 0.075 cached,
    0.600 out) — verify against the billing dashboard before quoting."""
    PRICE_IN, PRICE_IN_CACHED, PRICE_OUT = 0.150, 0.075, 0.600
    PREFIX_TOK = 1084          # measured rubric-first fixed prefix (CLAUDE.md, gotchas)
    n_conv = 0
    chars = 0
    for _, (_, _, d) in la3_manifest().items():
        for i in range(N_PERSONAS):
            p = os.path.join(d, f"conversation_{i}.csv")
            if os.path.exists(p):
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


# ── one-call bundle + the numbers ledger ─────────────────────────────────────

def crossgen_all(judge_tag: str = PRIMARY_JUDGE_TAG, *, alignment: bool = True) -> dict:
    """Load both graders' frames and compute every table of the family in one call.

    Returns ``{crossgen, gpt35, alignment, table1_max_abs_diff, levels, kcontrast, summary,
    ordering, agreement, vsbase, la3, la3_cost}``. ``alignment=False`` skips the 15x96-CSV
    persona-alignment read (the pairing-unit check) when only the tables are wanted.
    """
    cg = load_crossgen(judge_tag)
    g35 = load_exp1_gpt35()
    lv = levels(cg, g35)
    return dict(
        crossgen=cg, gpt35=g35,
        alignment=persona_alignment_check() if alignment else None,
        table1_max_abs_diff=table1_crosscheck(g35),
        levels=lv, kcontrast=k_contrast(cg, g35), summary=k_summary(cg, g35),
        ordering=ordering_claims(cg, g35), agreement=grader_agreement(lv, cg, g35),
        vsbase=vs_base(cg, g35), la3=la3_gpt35(), la3_cost=la3_cost_estimate(),
    )


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


def crossgen_numbers(F: dict, *, table_prefix: str = "tables/crossgen_") -> dict:
    """Every quotable number of the family as ``{dotted.key: {value, source, note}}`` — the shape
    ``exports.save_numbers`` writes and the paper's ``NUMBERS.md`` cites. ``F`` is the dict from
    :func:`crossgen_all` (or the same keys assembled by hand). Keys mirror the paper ledger
    ``analysis/out/crossgen_exp1.json``: ``pairing.*``, ``crosscheck.*``, ``levels.<model>``,
    ``kcontrast.<grader>.<metric>.iter<n>``, ``best.<grader>``, ``ordering.<grader>``,
    ``summary.<grader>.<metric>.<contrast>``, ``grader_agreement.<metric>.<level>``,
    ``vsbase.<grader>.Final``, ``la3.*``, ``verdict.<grader>.Final``. ``table_prefix`` names the
    tables the ``source`` strings point at (``<prefix><name>.md``)."""
    N = {}
    T = table_prefix
    if F.get("alignment") is not None:
        N["pairing.persona_index_alignment"] = _entry(
            F["alignment"], source="persona_alignment_check() on Exp1 conversation_i.csv openings",
            note="Exp1 wrote conversation_i from permutations[i] of a deterministic nested-loop generator "
                 "(no shuffle); index i = patient permutation i in every model dir. n_conflicting counts "
                 "indices whose parsed (age, problem) disagree across models.")
    N["crosscheck.iclr_table1_vs_disk_max_abs_diff"] = _entry(
        F["table1_max_abs_diff"], source=f"ICLR_TABLE1 vs Exp1 scores_i.csv means ({T}levels.md)",
        note="15 model states x {Q1,Q2,Final}; the transcription and the on-disk scores agree to 3 dp")
    lv = F["levels"]
    for _, r in lv.iterrows():
        N[f"levels.{r['model']}"] = _entry({k: r[k] for k in _LEVELS_COLS[4:]},
                                           source=f"{T}levels.md row model={r['model']}")
    kc = F["kcontrast"]
    for _, r in kc.iterrows():
        N[f"kcontrast.{r['grader']}.{r['metric']}.iter{int(r['iteration'])}"] = _entry(
            {k: r[k] for k in ("n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "p_t",
                               "n_K0_higher", "n_K5_higher", "delta_unpaired", "p_welch")},
            source=f"{T}kcontrast.md row grader={r['grader']} metric={r['metric']} iteration={int(r['iteration'])}",
            note="delta = K0 - K5; + => K=0 higher")
    for _, r in F["ordering"].iterrows():
        N[f"best.{r['grader']}"] = _entry(
            {k: r[k] for k in ("best_K0", "best_K5", "best_K0_final", "best_K5_final")},
            source=f"{T}levels.md (max Final_* over iters 1-7 per arm)")
        N[f"ordering.{r['grader']}"] = _entry(
            {k: r[k] for k in ("min_L5_final", "max_L0_final", "every_L5_above_every_L0",
                               "n_L5_models_above_max_L0", "n_of_49_L5xL0_pairs_with_L5_higher",
                               "n_of_7_iterations_L5_higher", "n_of_14_trained_models_above_base")},
            source=f"{T}levels.md (Final_* columns)", note="the ICLR text's group claim, checked on model MEANS")
    for _, r in F["summary"].iterrows():
        key = re.sub(r"[^A-Za-z0-9]+", "_", f"{r['contrast']}").strip("_")
        N[f"summary.{r['grader']}.{r['metric']}.{key}"] = _entry(
            {k: r[k] for k in ("n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_t", "n_K0_higher", "n_K5_higher")},
            source=f"{T}kcontrast_summary.md row grader={r['grader']} contrast='{r['contrast']}' metric={r['metric']}",
            note=str(r["note"]))
    for _, r in F["agreement"].iterrows():
        N[f"grader_agreement.{r['metric']}.{r['level'].replace(' ', '_')}"] = _entry(
            {k: r[k] for k in ("spearman_rho", "p", "pearson_r", "n")},
            source=f"{T}grader_agreement.md")
    vb = F["vsbase"]
    for grader in (GRADER_GPT4OMINI, GRADER_GPT35):
        sub = vb[(vb["grader"] == grader) & (vb["metric"] == "Final")]
        N[f"vsbase.{grader}.Final"] = _entry(
            {"n_models_positive": int((sub["mean_delta"] > 0).sum()),
             "n_models_p_holm_lt_05": int((sub["p_holm"] < 0.05).sum()),
             "min_delta": float(sub["mean_delta"].min()), "max_delta": float(sub["mean_delta"].max()),
             "per_model": {r["model"]: {"mean_delta": r["mean_delta"], "dz": r["dz"], "p_holm": r["p_holm"]}
                           for _, r in sub.iterrows()}},
            source=f"{T}vsbase.md rows grader={grader} metric=Final", note="14 trained models; + => above Base")
    la3 = F.get("la3_cost") or {}
    N["la3.status"] = _entry(
        {"scored_by_gpt4omini": False, "n_iterations": 4, "tool_supports_it": False,
         "reason": "score_crossgen.py --gen has choices {all,exp1,exp2}; exp1_models() excludes LookAhead_3 "
                   "(different hyperparameters: TT0.7, tau=0.2)",
         "what_it_would_take": "add the 4 FullEval_TTree1.4_TT0.7_TP0.7_TE0.2_Filter0.2_V{1..4}.0 dirs to "
                               "exp1_models() as Exp1_LA3_I{1..4}, run --dry-run then score; estimated cost "
                               "below (no API call was made here)", **la3},
        source=f"{T}la3_gpt35.md + la3_cost_estimate() mirroring score_crossgen.dry_run",
        note="cost = list-price gpt-4o-mini, 50% cached 1084-tok prefix, ~180 output tok/call")
    if F.get("la3") is not None:
        for _, r in F["la3"].iterrows():
            N[f"la3.gpt35.{r['model']}"] = _entry({k: r[k] for k in ("Q1", "Q2", "Final", "n_gpt35")},
                                                  source=f"{T}la3_gpt35.md row model={r['model']}")
    for grader in (GRADER_GPT4OMINI, GRADER_GPT35):
        f = kc[(kc["grader"] == grader) & (kc["metric"] == "Final")]
        N[f"verdict.{grader}.Final"] = _entry(
            {"n_iters_K5_higher": int((f["mean_delta"] < 0).sum()),
             "n_iters_K5_higher_p_holm_lt_05": int(((f["mean_delta"] < 0) & (f["p_holm"] < 0.05)).sum()),
             "n_iters_K0_higher_p_holm_lt_05": int(((f["mean_delta"] > 0) & (f["p_holm"] < 0.05)).sum()),
             "mean_of_7_deltas": float(f["mean_delta"].mean()),
             "median_dz": float(f["dz"].median()),
             "iter_deltas": {int(r["iteration"]): r["mean_delta"] for _, r in f.iterrows()}},
            source=f"{T}kcontrast.md rows grader={grader} metric=Final",
            note="delta = K0 - K5; negative => K=5 higher")
    return N


# ── captions (kept verbatim from the paper generator so the notebook can save them) ─────
CAPTIONS = {
    "levels": (
        "Exp1 (ICLR 2025: Llama-2-7B therapist, GPT-3.5 patient) model states, 96 conversations each "
        "(one per patient permutation), scored by TWO graders side by side (never averaged): "
        "`*_gpt4omini` = the SAME conversations re-scored by the Exp3 oracle (gpt-4o-mini-2024-07-18, "
        "T=0.1, V5 JSON-schema Q1+Q2; data/eval_scores/_crossgen); `*_gpt35` = the original GPT-3.5 "
        "oracle scores Exp1 saved beside each conversation (scores_i.csv); `*_iclr_tab1` = ICLR Table 1 "
        "as printed (reproduced by `*_gpt35` to 3 dp). Final = mean(Q1, Q2) per conversation, then "
        "averaged (= the lake's Q1Q2 composite). Iteration 0 = the untrained Llama-2-7B base (a "
        "single draw; both arms share it). Q1 = session satisfaction (5 items), Q2 = working "
        "alliance (17 items), both 1-5. Rows: Base, then K=0 iters 1-7, then K=5 iters 1-7."),
    "kcontrast": (
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
        "Metric Final = mean(Q1,Q2) = the lake's Q1Q2. Both arms run 7 iterations (no censoring)."),
    "kcontrast_summary": (
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
        "min(K=5 model mean) - max(K=0 model mean): + means every K=5 model outscores every K=0 model."),
    "grader_agreement": (
        "Agreement between the two graders on Exp1's conversations: Spearman rho (and Pearson r) between "
        "the gpt-4o-mini re-score and the original GPT-3.5 oracle, at the level of the 15 model-state "
        "means (Base + 7 K=0 + 7 K=5), the 14 trained-model means, and per conversation (15 x 96 pooled). "
        "Metric Final = mean(Q1,Q2)."),
    "vsbase": (
        "Each trained Exp1 model minus the untrained Llama-2-7B Base, under each grader (side by side, "
        "never averaged): `mean_delta` = model - Base; **+ => trained model higher**; paired on conversation "
        "index (= persona), n = 96; dz, 95% bootstrap CI, Wilcoxon p, Holm within each (grader, arm, metric) "
        "family across the 7 iterations; p_t = paired t; n_model_higher / n_base_higher = sign split. "
        "gpt-4o-mini = the Exp3 oracle re-score; gpt-3.5 = the original ICLR oracle. Metric Final = mean(Q1,Q2)."),
    "la3_gpt35": (
        "Exp1's K=3 sweep (LookAhead_3, 4 iterations, 96 conversations each) — the only look-ahead 'dose' "
        "data on disk. NOT re-scored by gpt-4o-mini: score_crossgen.py deliberately excludes it (its "
        "manifest is the paper's K=0/K=5 pair only, and K=3 ran with different hyper-parameters — therapist "
        "temperature 0.7 and filter tau 0.2 vs 0.9 / 0.1 — so it is not a matched dose arm). Shown here "
        "are its ORIGINAL GPT-3.5 oracle means from the on-disk scores_i.csv (Q1, Q2, Final = mean). "
        "Not comparable to the K=0/K=5 rows of the levels table without that caveat."),
    "fig": (
        "Exp1 (ICLR 2025; Llama-2-7B therapist, GPT-3.5 patient) PTO models by iteration, Final = mean(Q1,Q2) "
        "averaged over the 96 conversations (one per patient permutation), K=0 solid circles vs K=5 dashed "
        "squares, untrained Base as the dotted line, bands = 95% percentile-bootstrap CI of the mean. Left: the "
        "original GPT-3.5 oracle (ICLR Table 1); right: the SAME transcripts re-scored by the Exp3 oracle "
        "(gpt-4o-mini-2024-07-18, V5 Q1+Q2). Separate y-axes per panel because the graders sit on different "
        "levels (gpt-4o-mini reads ~0.19-0.43 higher; never averaged). Paired statistics for the gap are in "
        "the kcontrast table (delta = K0 - K5, + => K=0 higher, paired on conversation index). "
        "Both arms complete 7 iterations (no censoring)."),
}
__all__.append("CAPTIONS")
