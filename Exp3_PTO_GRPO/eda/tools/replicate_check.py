"""Does the endpoint survive a second draw? — the replicate-draw analysis (STATUS.md § replicate).

Reads the score lake directly (the replicate models live OUTSIDE ``discover_arms``, by design —
see ``score_replicate.py``), pairs every contrast on conversation id, and writes one markdown
report to ``results/measurement/replicate_draw.md`` (beside ``SUMMARY.md``, outside the family
leaves, so a re-render never touches it).

Pairing note: every state here is an iteration-10 draw, and the persona shuffle at
``model_iter_10`` is ``random.Random(seed + 11)`` with ``seed = 42`` in EVERY arm and every draw
(verified per arm by ``generate_eval_convs.py --verify-seeds`` and package-wide by
``_selfcheck``'s persona-permutation check) — so ``conversation_<id>`` is the SAME persona in
every column being paired, and id-pairing IS persona-pairing for this specific set. It is NOT
valid across different iterations; do not reuse this script's join for those.

Contrasts (sign: A − B), each on both graders, Holm across the 9 metric rows within
(contrast, grader):

  same-policy noise floor        GRPO_LA5 I10 draw2 − draw1        expect ~0
                                 PTO_LA0  I10 draw2 − draw1        expect ~0
  K lever @10 (GRPO), replicate  GRPO_LA5_rep1_I10 − GRPO_LA0_I10  original: +0.765 P / +0.616 H
  method @K0, replicate          PTO_LA0_rep1_I10 − GRPO_LA0_I10   original: +0.507 P / +0.609 H
  method @K5, replicate          PTO_LA5_I10 − GRPO_LA5_rep1_I10   original: +0.210 P / +0.206 H
  held-out top pair, paired      GRPO_LA5_rep1 − PTO_LA0_rep1      original: 0.007 H, unpaired

Run AFTER ``score_replicate.py`` reports full coverage on both graders:
    python tools/replicate_check.py
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
from eda_analysis.constants import LOWER_IS_BETTER  # noqa: E402
from eda_analysis.scoring import (  # noqa: E402
    EVAL_QUESTIONNAIRE_DIRS, eval_csv_dir, eval_scores_root,
)
from eda_analysis.scoring.judge import JUDGE_METRIC_COLS  # noqa: E402

RESULTS = os.path.join(_p, "results")
OUT_MD = os.path.join(RESULTS, "measurement", "replicate_draw.md")

JUDGES = [("gpt-4o-mini", ""), ("claude-haiku-4-5", "anthropic_claude-haiku-4-5")]
METRICS = ["Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI"]

# model -> training-oracle label in the lake path (all trained states here are Q1Q2-trained).
MODELS = {
    "GRPOExp3_LA5_I10": "Q1Q2", "GRPOExp3_LA5_rep1_I10": "Q1Q2",
    "PTOExp3_LA0_I10": "Q1Q2", "PTOExp3_LA0_rep1_I10": "Q1Q2",
    "GRPOExp3_LA0_I10": "Q1Q2", "PTOExp3_LA5_I10": "Q1Q2",
}

CONTRASTS = [
    ("same-policy: GRPO_LA5@10 draw2-draw1", "GRPOExp3_LA5_rep1_I10", "GRPOExp3_LA5_I10"),
    ("same-policy: PTO_LA0@10 draw2-draw1", "PTOExp3_LA0_rep1_I10", "PTOExp3_LA0_I10"),
    ("K lever @10 (GRPO, replicate K5)", "GRPOExp3_LA5_rep1_I10", "GRPOExp3_LA0_I10"),
    ("method @K0 (replicate PTO)", "PTOExp3_LA0_rep1_I10", "GRPOExp3_LA0_I10"),
    ("method @K5 (replicate GRPO)", "PTOExp3_LA5_I10", "GRPOExp3_LA5_rep1_I10"),
    ("held-out top pair (both replicates)", "GRPOExp3_LA5_rep1_I10", "PTOExp3_LA0_rep1_I10"),
]


def load_model_scores(judge_tag: str, model: str) -> pd.DataFrame:
    """id x metric frame for one model under one grader, straight off the lake CSVs."""
    root = eval_scores_root(judge_tag, 0)
    cols = {}
    for qname, subdir in EVAL_QUESTIONNAIRE_DIRS.items():
        d = eval_csv_dir(root, MODELS[model], subdir, model)
        if not os.path.isdir(d):
            continue
        vcol = JUDGE_METRIC_COLS[qname][1]
        vals = {}
        for fn in os.listdir(d):
            if not fn.endswith(".csv"):
                continue
            try:
                row = pd.read_csv(os.path.join(d, fn))
                vals[int(os.path.splitext(fn)[0])] = float(row[vcol].iloc[0])
            except Exception:
                pass
        cols[qname] = pd.Series(vals)
    df = pd.DataFrame(cols)
    if {"Q1", "Q2"} <= set(df.columns):
        df["Q1Q2"] = df[["Q1", "Q2"]].mean(axis=1)
    return df


def main() -> int:
    frames = {}   # (judge_label, model) -> id x metric frame
    for jlabel, jtag in JUDGES:
        for m in MODELS:
            f = load_model_scores(jtag, m)
            frames[(jlabel, m)] = f
            n = f.dropna().shape[0]
            print(f"  [{jlabel}] {m}: {f.shape[0]} convs, {f.shape[1]} metrics, {n} complete rows")
            if f.shape[0] < 96 or f.shape[1] < 9:
                print(f"    ! incomplete — scoring not finished for this state?")

    lines = [
        "# Replicate draw — does the endpoint survive a second sample?",
        "",
        f"*Generated {date.today().isoformat()} by `tools/replicate_check.py` (rerunnable; a "
        "re-render never touches this file). Second independent 96-conversation draws of "
        "`GRPO_LA5@10` and `PTO_LA0@10` (unseeded decoding, same adapters, same personas, same "
        "seed-53 shuffle), scored on all 8 instruments by both graders. Pairing is on "
        "conversation id, which IS persona-pairing here — every column is an iteration-10 draw "
        "under the same shuffle. Sign: A − B as named. Holm across the 9 metric rows within "
        "(contrast, grader). MICI is lower-is-better.*",
        "",
    ]
    for cname, ma, mb in CONTRASTS:
        lines += [f"## {cname}", "",
                  f"`A = {ma}` · `B = {mb}`", "",
                  "| metric | grader | n | Δ (A−B) | dz | 95% CI | p | p_holm |",
                  "|---|---|---:|---:|---:|---|---:|---:|"]
        for jlabel, _ in JUDGES:
            fa, fb = frames[(jlabel, ma)], frames[(jlabel, mb)]
            rows, ps = [], []
            for met in METRICS:
                if met not in fa.columns or met not in fb.columns:
                    rows.append((met, None)); ps.append(np.nan); continue
                j = pd.concat({"a": fa[met], "b": fb[met]}, axis=1).dropna()
                r = stats.paired_arrays(j["a"].to_numpy(), j["b"].to_numpy())
                rows.append((met, r)); ps.append(r["p"])
            ph = stats.holm(ps)
            for (met, r), h in zip(rows, ph):
                if r is None:
                    lines.append(f"| {met} | {jlabel} | – | – | – | – | – | – |")
                    continue
                arrow = " ↓" if met in LOWER_IS_BETTER else ""
                lines.append(
                    f"| {met}{arrow} | {jlabel} | {r['n']} | {r['mean_delta']:+.3f} | "
                    f"{r['dz']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                    f"{r['p']:.2e} | {h:.4f} |")
        lines.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}")

    # Console headline: Q1Q2 per contrast per grader.
    print("\nQ1+Q2 headline (Δ = A − B):")
    for cname, ma, mb in CONTRASTS:
        for jlabel, _ in JUDGES:
            fa, fb = frames[(jlabel, ma)], frames[(jlabel, mb)]
            if "Q1Q2" not in fa.columns or "Q1Q2" not in fb.columns:
                continue
            j = pd.concat({"a": fa["Q1Q2"], "b": fb["Q1Q2"]}, axis=1).dropna()
            r = stats.paired_arrays(j["a"].to_numpy(), j["b"].to_numpy())
            print(f"  {cname:44s} {jlabel:16s} Δ {r['mean_delta']:+.3f}  dz {r['dz']:+.3f}  p {r['p']:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
