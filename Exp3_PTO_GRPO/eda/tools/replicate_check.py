"""Does the endpoint survive a second draw? — the replicate-draw analysis (STATUS.md § replicate).

Reads the score lake directly (the replicate models live OUTSIDE ``discover_arms``, by design —
see ``score_replicate.py``) and writes one markdown report to
``results/measurement/replicate_draw.md`` — beside ``SUMMARY.md``, outside the family leaves, so a
re-render never touches it.

**The design: every headline contrast is computed TWICE** — once with the original
96-conversation draw (reproducing the published table) and once with the replicate substituted —
and the two are printed adjacent. That is the comparison the paper's §4 todo actually asks for:
not "what is the replicate's number" but "does the claim survive a second sample". A row pair
that keeps its sign, its significance and roughly its effect size is a claim that replicates.

Contrasts (sign: A − B), each on both graders, Holm across the 9 metric rows within
(contrast, draw, grader):

  noise floor    GRPO_LA5 @10 draw2 − draw1        same policy, two draws → expect ~0
                 PTO_LA0  @10 draw2 − draw1        same policy, two draws → expect ~0
  K lever @10    GRPO_LA5 − GRPO_LA0               published +0.765 P / +0.616 H
  method @K0     PTO_LA0 − GRPO_LA0                published +0.507 P / +0.609 H
  method @K5     PTO_LA5 − GRPO_LA5                published −0.210 P / −0.206 H
  top pair (H)   GRPO_LA5 − PTO_LA0                published 0.007 H, UNPAIRED and with no p;
                                                   the replicate makes it a real paired test

Pairing note: every state here is an iteration-10 draw, and the persona shuffle at
``model_iter_10`` is ``random.Random(seed + 11)`` with ``seed = 42`` in EVERY arm and every draw
(verified per arm by ``generate_eval_convs.py --verify-seeds`` and package-wide by
``_selfcheck``'s persona-permutation check) — so ``conversation_<id>`` is the SAME persona in
every column being paired, and id-pairing IS persona-pairing for this specific set. It is NOT
valid across different iterations; do not reuse this join for those.

Validated 2026-08-26 before the replicate existed: the machinery below reproduces
``k_endpoints.md``'s published K-lever row exactly on both graders (+0.765 / dz 0.905 primary,
+0.616 / dz 1.030 held-out).

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

JUDGES = [("primary", ""), ("held-out", "anthropic_claude-haiku-4-5")]
METRICS = ["Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI"]

ORIG_G5, REP_G5 = "GRPOExp3_LA5_I10", "GRPOExp3_LA5_rep1_I10"
ORIG_P0, REP_P0 = "PTOExp3_LA0_I10", "PTOExp3_LA0_rep1_I10"
G0, P5 = "GRPOExp3_LA0_I10", "PTOExp3_LA5_I10"

# model -> training-oracle label in the lake path (every trained state here is Q1Q2-trained).
MODELS = {m: "Q1Q2" for m in (ORIG_G5, REP_G5, ORIG_P0, REP_P0, G0, P5)}

# (section, contrast label, draw label, model A, model B, published Q1Q2 note)
CONTRASTS = [
    ("noise floor", "GRPO_LA5 @10, draw 2 - draw 1", "same policy", REP_G5, ORIG_G5,
     "expect ~0 (same adapter, same personas, unseeded decoding)"),
    ("noise floor", "PTO_LA0 @10, draw 2 - draw 1", "same policy", REP_P0, ORIG_P0,
     "expect ~0 (same adapter, same personas, unseeded decoding)"),

    ("survival", "K lever @10 (GRPO K5 - K0)", "original", ORIG_G5, G0,
     "published +0.765 primary / +0.616 held-out"),
    ("survival", "K lever @10 (GRPO K5 - K0)", "replicate", REP_G5, G0, ""),

    ("survival", "method @K0 (PTO - GRPO)", "original", ORIG_P0, G0,
     "published +0.507 primary / +0.609 held-out"),
    ("survival", "method @K0 (PTO - GRPO)", "replicate", REP_P0, G0, ""),

    ("survival", "method @K5 (PTO - GRPO)", "original", P5, ORIG_G5,
     "published -0.210 primary / -0.206 held-out"),
    ("survival", "method @K5 (PTO - GRPO)", "replicate", P5, REP_G5, ""),

    ("survival", "top pair (GRPO_LA5 - PTO_LA0)", "original", ORIG_G5, ORIG_P0,
     "published held-out gap 0.007, UNPAIRED with no p; this makes it a paired test"),
    ("survival", "top pair (GRPO_LA5 - PTO_LA0)", "replicate", REP_G5, REP_P0, ""),
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


def paired_rows(fa: pd.DataFrame, fb: pd.DataFrame):
    """One result dict per metric (None where a metric is absent), Holm-corrected across them."""
    rows, ps = [], []
    for met in METRICS:
        if met not in fa.columns or met not in fb.columns:
            rows.append((met, None)); ps.append(np.nan); continue
        j = pd.concat({"a": fa[met], "b": fb[met]}, axis=1).dropna()
        if j.empty:
            rows.append((met, None)); ps.append(np.nan); continue
        r = stats.paired_arrays(j["a"].to_numpy(), j["b"].to_numpy())
        rows.append((met, r)); ps.append(r["p"])
    return rows, stats.holm(ps)


def main() -> int:
    needed = sorted({m for c in CONTRASTS for m in (c[3], c[4])})
    frames = {}
    incomplete = []
    for jlabel, jtag in JUDGES:
        for m in needed:
            f = load_model_scores(jtag, m)
            frames[(jlabel, m)] = f
            n_complete = f.dropna().shape[0] if not f.empty else 0
            flag = "" if (f.shape[0] >= 96 and f.shape[1] >= 9) else "   <-- INCOMPLETE"
            if flag:
                incomplete.append(f"{jlabel}/{m}")
            print(f"  [{jlabel:8s}] {m:24s} {f.shape[0]:3d} convs x {f.shape[1]} metrics"
                  f"  ({n_complete} complete){flag}")
    if incomplete:
        print("\nIncomplete states: " + ", ".join(incomplete))
        print("Run score_replicate.py --primary --judge until coverage is full, then re-run.")
        return 1

    lines = [
        "# Replicate draw — does the endpoint survive a second sample?",
        "",
        f"*Generated {date.today().isoformat()} by `tools/replicate_check.py` (rerunnable; a "
        "re-render never touches this file). Second independent 96-conversation draws of "
        "`GRPO_LA5@10` and `PTO_LA0@10` — same adapters, same 96 personas, same seed-53 shuffle, "
        "unseeded decoding — scored on all 8 instruments by both graders.*",
        "",
        "**Every headline contrast is computed twice**: once with the original draw (reproducing "
        "the published table) and once with the replicate substituted. A claim replicates when "
        "the pair keeps its sign, its significance, and roughly its effect size.",
        "",
        "*Pairing is on conversation id, which IS persona-pairing here — every column is an "
        "iteration-10 draw under the same shuffle. Sign: A − B as named. Holm across the 9 metric "
        "rows within (contrast, draw, grader). MICI is lower-is-better (↓).*",
        "",
        "---",
        "",
        "## Q1+Q2 at a glance",
        "",
        "| section | contrast | draw | grader | Δ | dz | 95% CI | p_holm |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    cache = {}
    for section, label, draw, ma, mb, note in CONTRASTS:
        for jlabel, _ in JUDGES:
            rows, ph = paired_rows(frames[(jlabel, ma)], frames[(jlabel, mb)])
            cache[(label, draw, jlabel)] = (rows, ph, ma, mb, note)
            r = dict(rows)["Q1Q2"]
            h = ph[METRICS.index("Q1Q2")]
            lines.append(f"| {section} | {label} | {draw} | {jlabel} | {r['mean_delta']:+.3f} | "
                         f"{r['dz']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {h:.4f} |")

    lines += ["", "---", "", "## Every instrument, per contrast", ""]
    seen = set()
    for section, label, draw, ma, mb, note in CONTRASTS:
        if (label, draw) in seen:
            continue
        seen.add((label, draw))
        lines += [f"### {label} — {draw}", "",
                  f"`A = {ma}` · `B = {mb}`" + (f"  \n{note}" if note else ""), "",
                  "| metric | grader | n | Δ (A−B) | dz | 95% CI | p | p_holm |",
                  "|---|---|---:|---:|---:|---|---:|---:|"]
        for jlabel, _ in JUDGES:
            rows, ph, _, _, _ = cache[(label, draw, jlabel)]
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

    # Console: the original-vs-replicate comparison, Q1+Q2 only.
    print("\n" + "=" * 78)
    print("Q1+Q2 — original vs replicate (Δ = A - B, dz, p_holm)")
    print("=" * 78)
    printed = set()
    for section, label, draw, ma, mb, note in CONTRASTS:
        if label in printed:
            continue
        printed.add(label)
        print(f"\n{label}")
        for d in ("same policy", "original", "replicate"):
            for jlabel, _ in JUDGES:
                key = (label, d, jlabel)
                if key not in cache:
                    continue
                rows, ph, _, _, _ = cache[key]
                r = dict(rows)["Q1Q2"]
                h = ph[METRICS.index("Q1Q2")]
                sig = "*" if h < 0.05 else " "
                print(f"    {d:11s} {jlabel:9s} {r['mean_delta']:+.3f}  dz {r['dz']:+.3f}  "
                      f"p_holm {h:.4f}{sig}")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
