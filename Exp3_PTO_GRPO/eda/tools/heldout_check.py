"""Held-out personas: does the K=5 vs K=0 endpoint hold on patients the training never saw?

Added 2026-10-08. The held-out set (``score_heldout.py``) is 48 personas with a problem the training
grid never had (alcohol), every other persona factor as in training, simulated by the untrained
Base, K=0 at iterations 8 (its best checkpoint on the training oracle's Q1+Q2) and 10, and K=5 at
10, each scored by both graders. Reads the score lake directly (the held-out models live OUTSIDE
``discover_arms`` by design) and writes one markdown report,
``results/lookahead/heldout_personas.md`` -- beside the family's ``SUMMARY.md``, outside the family
leaves, so a re-render never touches it.

Pairing: the held-out persona list is NOT shuffled, so ``conversation_<i>`` is held-out persona i
in every state, and id-pairing is persona-pairing (n <= 48).

Two blocks, both graders, sign A - B:
  instruments  the nine rows of Table 1 (Q1+Q2, Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI, PCT, MICI);
               Holm across the nine rows per (contrast, grader), as in Table 1
  process      the utterance coder's measures quoted in the body (praise / complex reflection /
               persuasion shares, MI-consistent and MI-inconsistent shares, reflection after change
               talk, praise / persuasion after sustain talk, change-talk persistence, change talk
               after sustain talk); Holm across the process rows per (contrast, grader), as Table 3.
Contrasts: K=5@10 - K=0@8, K=5@10 - K=0@10, and each trained state - Base (levels context).

    python tools/heldout_check.py     # after score_heldout.py reports full coverage on both graders
"""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

_p = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _p not in sys.path:
    sys.path.insert(0, _p)

from eda_analysis import stats  # noqa: E402
from eda_analysis.constants import LOWER_IS_BETTER, REPORTED_SUBDIR, reported_value  # noqa: E402
from eda_analysis.process import conversation_metrics  # noqa: E402
from eda_analysis.scoring import EVAL_QUESTIONNAIRE_DIRS, eval_csv_dir, eval_scores_root  # noqa: E402
from eda_analysis.scoring.judge import JUDGE_METRIC_COLS  # noqa: E402

RESULTS = os.path.join(_p, "results")
OUT_MD = os.path.join(RESULTS, "lookahead", "heldout_personas.md")
N = 48

JUDGES = [("training oracle", ""), ("held-out judge", "anthropic_claude-haiku-4-5")]
METRICS = ["Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI"]

BASE, K0_8, K0_10, K5_10 = ("GRPOExp3_LA0_alc_Base", "GRPOExp3_LA0_alc_I8",
                            "GRPOExp3_LA0_alc_I10", "GRPOExp3_LA5_alc_I10")
STATES = [("Base", BASE), ("K=0 at 8", K0_8), ("K=0 at 10", K0_10), ("K=5 at 10", K5_10)]
ORACLE = {BASE: "none", K0_8: "Q1Q2", K0_10: "Q1Q2", K5_10: "Q1Q2"}

# (label, A, B)
CONTRASTS = [
    ("K=5 at 10 - K=0 at 8 (best)", K5_10, K0_8),
    ("K=5 at 10 - K=0 at 10 (last)", K5_10, K0_10),
    ("K=5 at 10 - Base", K5_10, BASE),
    ("K=0 at 10 - Base", K0_10, BASE),
    ("K=0 at 8 - Base", K0_8, BASE),
]

# process measures: (column, label, lower_is_better)
PROCESS = [
    ("th_PRA_rate", "non-specific praise, share of therapist turns", False),
    ("th_CR_rate", "complex reflection, share of therapist turns", False),
    ("th_PERS_rate", "persuasion, share of therapist turns", True),
    ("mi_adherent_rate", "MI-consistent share of therapist turns", False),
    ("mi_incons_rate", "MI-inconsistent share of therapist turns", True),
    ("refl_after_ct", "reflection after change talk", False),
    ("pra_after_st", "praise after sustain talk", True),
    ("pers_after_st", "persuasion after sustain talk", True),
    ("ct_persist", "change talk after change talk (persistence)", False),
    ("st_to_ct", "change talk after sustain talk", False),
]


def _csvs(d: str):
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        if fn.endswith(".csv"):
            try:
                yield int(os.path.splitext(fn)[0]), pd.read_csv(os.path.join(d, fn)).iloc[0]
            except Exception:
                continue


def load_instruments(judge_tag: str, model: str) -> pd.DataFrame:
    """id x metric frame (the reported values: PCT from the utterance coder)."""
    root = eval_scores_root(judge_tag, 0)
    cols = {}
    for qname, subdir in EVAL_QUESTIONNAIRE_DIRS.items():
        if qname == "MIPROC":
            continue
        d = eval_csv_dir(root, ORACLE[model], REPORTED_SUBDIR.get(qname, subdir), model)
        vcol = JUDGE_METRIC_COLS[qname][1]
        vals = {}
        for i, row in _csvs(d):
            v = reported_value(qname, row, vcol)
            if v == v:
                vals[i] = v
        cols[qname] = pd.Series(vals, dtype=float)
    df = pd.DataFrame(cols).sort_index()
    if {"Q1", "Q2"} <= set(df.columns):
        df["Q1Q2"] = df[["Q1", "Q2"]].mean(axis=1)
    return df


def load_process(judge_tag: str, model: str) -> pd.DataFrame:
    """id x process-measure frame from the MIPROC codes (opener excluded, as in the paper)."""
    root = eval_scores_root(judge_tag, 0)
    d = eval_csv_dir(root, ORACLE[model], EVAL_QUESTIONNAIRE_DIRS["MIPROC"], model)
    rows = {}
    for i, r in _csvs(d):
        th = str(r.get("MIPROC_ThCodes", "-")); pt = str(r.get("MIPROC_PtCodes", "-"))
        th = [] if th in ("-", "nan", "") else th.split("|")
        pt = [] if pt in ("-", "nan", "") else pt.split("|")
        m = conversation_metrics(th, pt)
        # persistence as process.persistence_metrics: (patient #(i-1), patient #i), i >= 1
        pairs = [(pt[j - 1], pt[j]) for j in range(1, min(len(th), len(pt)))]
        a_ct = [b for a, b in pairs if a == "CT"]; a_st = [b for a, b in pairs if a == "ST"]
        m["ct_persist"] = sum(b == "CT" for b in a_ct) / len(a_ct) if a_ct else np.nan
        m["st_to_ct"] = sum(b == "CT" for b in a_st) / len(a_st) if a_st else np.nan
        rows[i] = m
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def paired_block(fa: pd.DataFrame, fb: pd.DataFrame, cols):
    out, ps = [], []
    for c in cols:
        if c not in fa.columns or c not in fb.columns:
            out.append((c, None)); ps.append(np.nan); continue
        j = pd.concat({"a": fa[c], "b": fb[c]}, axis=1).dropna()
        if len(j) < 5:
            out.append((c, None)); ps.append(np.nan); continue
        r = stats.paired_arrays(j["a"].to_numpy(), j["b"].to_numpy())
        out.append((c, r)); ps.append(r["p"])
    return out, stats.holm(ps)


def main() -> int:
    inst, proc, short = {}, {}, []
    for jl, jt in JUDGES:
        for _, m in STATES:
            fi = load_instruments(jt, m); fp = load_process(jt, m)
            inst[(jl, m)], proc[(jl, m)] = fi, fp
            miss = [x for x in METRICS if x not in fi.columns or fi[x].notna().sum() < N - 2]
            if fi.shape[0] < N or fp.shape[0] < N or miss:
                short.append(f"{jl}/{m}: {fi.shape[0]} instrument convs, {fp.shape[0]} coded"
                             + (f", short on {miss}" if miss else ""))
    if short:
        print("Incomplete:\n  " + "\n  ".join(short))
        print("Run score_heldout.py --primary --judge until coverage is full, then re-run.")
        return 1

    L = [
        "# Held-out personas: alcohol, a problem the training grid never had",
        "",
        f"*Generated {date.today().isoformat()} by `tools/heldout_check.py` (rerunnable; a re-render "
        "never touches this file). 48 personas (2 gender x 3 cooperation x 2 duration x 2 prior "
        "attempts x 2 ages, problem = alcohol; `system_prompts_builder.generate_heldout_permutations`), "
        "simulated by four states with one patient seed (`HELDOUT_SEED`), scored by both graders on "
        "the eight instruments and the utterance coder. Persona-paired on conversation id (the list "
        "is not shuffled). Sign A - B. Holm across the nine instrument rows, and separately across "
        "the process rows, per (contrast, grader). MICI and the rows marked (lower) are lower-is-better.*",
        "",
        "## Levels (means over conversations)",
        "",
    ]
    for jl, _ in JUDGES:
        L += [f"### {jl}", "", "| measure | " + " | ".join(s for s, _ in STATES) + " |",
              "|---|" + "---:|" * len(STATES)]
        for met in METRICS:
            L.append(f"| {met}{' (lower)' if met in LOWER_IS_BETTER else ''} | "
                     + " | ".join(f"{inst[(jl, m)][met].mean():.3f}" for _, m in STATES) + " |")
        for col, lab, low in PROCESS:
            L.append(f"| {lab}{' (lower)' if low else ''} | "
                     + " | ".join(f"{proc[(jl, m)][col].mean():.3f}" for _, m in STATES) + " |")
        L.append("")

    for title, frames, cols, labels in (
            ("Instruments", inst, METRICS, {m: m for m in METRICS}),
            ("Process (utterance coder)", proc, [c for c, _, _ in PROCESS], {c: l for c, l, _ in PROCESS})):
        L += [f"## {title}: persona-paired contrasts", ""]
        for clab, a, b in CONTRASTS:
            L += [f"### {clab}", "", "| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |",
                  "|---|---|---:|---:|---:|---|---:|---:|"]
            for jl, _ in JUDGES:
                rows, ph = paired_block(frames[(jl, a)], frames[(jl, b)], cols)
                for (c, r), h in zip(rows, ph):
                    if r is None:
                        L.append(f"| {labels[c]} | {jl} | - | - | - | - | - | - |"); continue
                    L.append(f"| {labels[c]} | {jl} | {r['n']} | {r['mean_delta']:+.3f} | {r['dz']:+.3f} | "
                             f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p']:.2e} | {h:.4f} |")
            L.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}")

    for jl, _ in JUDGES:
        for clab, a, b in CONTRASTS[:2]:
            rows, ph = paired_block(inst[(jl, a)], inst[(jl, b)], METRICS)
            r = dict(rows)["Q1Q2"]
            n_pos = sum(1 for (c, x), h in zip(rows, ph) if x is not None and h < .05 and
                        ((x["mean_delta"] < 0) if c in LOWER_IS_BETTER else (x["mean_delta"] > 0)))
            print(f"{jl:16s} {clab:30s} Q1+Q2 {r['mean_delta']:+.3f} dz {r['dz']:+.2f}; "
                  f"K=5 significantly better on {n_pos}/9 rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
