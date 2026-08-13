"""score_crossgen.py — put Exp1 and Exp2 conversations on the Exp3 measurement axis.

WHY
---
The look-ahead (K) result changes sign across the three experiment generations:
Exp1 (ICLR 2025) finds K=5 clearly ahead, Exp2 finds a null, Exp3 finds K=5 never
leading. But each generation was graded by a *different* oracle (Exp1: GPT-3.5 with a
regex-parsed V1 rubric; Exp2/Exp3: gpt-4o-mini with the V5 JSON-schema rubric), so the
comparison is confounded: "the grader changed" and "the model/setup changed" are not
separated.

This script re-scores the ALREADY-GENERATED Exp1 and Exp2 conversations with the *Exp3*
oracle (gpt-4o-mini + V5 questionnaires), on Q1 and Q2. No GPU, no patient simulation —
oracle calls only. If the modern grader still sees K=5 ahead on Exp1's conversations,
the reversal is a property of the model/task, not of the judge.

WHERE IT WRITES
---------------
``data/eval_scores/_crossgen/judge=<tag>/rep=0/metric=<M>/oracle=<O>/<Model>/<id>.csv``

The ``_crossgen`` prefix keeps this OUT of the Exp3 score lake proper: nothing in the
analysis layer globs it (arm discovery reads conversations under ``data/{grpo,pto}_Exp3``,
and judge partitions are resolved by explicit tag), so the tracked Exp3 results cannot
move because of it. It still sits inside the Drive-backed ``eval_scores`` symlink, so the
spend is backed up — following the existing ``_parquet`` / ``_batches`` precedent.

Model names are generation-prefixed (``Exp1_LA5_I3``, ``Exp2_WAI_LA0_I2``) so they can
never collide with Exp3's ``PTOExp3_*`` / ``GRPOExp3_*``.

USAGE
-----
    python tools/score_crossgen.py --dry-run     # token + cost estimate, no API calls
    python tools/score_crossgen.py               # score (resume-safe: skips existing CSVs)
    python tools/score_crossgen.py --gen exp1    # one generation only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_EDA = os.path.dirname(_HERE)
if _EDA not in sys.path:
    sys.path.insert(0, _EDA)

# constants is the package leaf: importing it prepends code/ to sys.path so the canonical
# `questionnaires` (V5) resolves. Must happen before the scoring layer is imported.
from eda_analysis.constants import EVAL_SCORES, WORKSPACE_ROOT  # noqa: E402
from eda_analysis.scoring import conversations as convmod       # noqa: E402
from eda_analysis.scoring import pipeline as pipe               # noqa: E402
from eda_analysis.scoring.registry import (                     # noqa: E402
    EVAL_MODEL, EVAL_QUESTIONNAIRE_DIRS, EVAL_TEMPERATURE, eval_csv_dir,
)

# The two experiments live beside Exp3 under the repo root.
_REPO = os.path.dirname(WORKSPACE_ROOT)
EXP1 = os.path.join(_REPO, "Exp1_ICLR2025", "data", "conversations_eval")
EXP2 = os.path.join(_REPO, "Exp2_PTO", "data", "conversations_eval")

CROSSGEN_ROOT = os.path.join(EVAL_SCORES, "_crossgen")

# gpt-4o-mini pricing, USD per 1M tokens (verify against the billing dashboard before quoting).
PRICE_IN, PRICE_IN_CACHED, PRICE_OUT = 0.150, 0.075, 0.600


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          THE MODEL MANIFEST                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def exp1_models() -> dict[str, str]:
    """Exp1 (ICLR 2025): Llama-2-7B / GPT-3.5. The paper's main sweep only.

    The paper's Base is ``Basic_50_TT0.9_TP0.7_TE0.2_V2`` (Final 3.453, Table 1). The
    ``Q2_``-prefixed, ``_OLD`` and ``_FAIL_Q2`` directories are separate/abandoned sweeps
    and are deliberately excluded, as is LookAhead_3 (different hyperparameters —
    Filter0.2/TT0.7 — so it is not matched to the K=0/K=5 pair).
    """
    arm = "TTree1.4_TT0.9_TP0.7_TE0.2_V{}"
    m = {"Exp1_Base": os.path.join(EXP1, "Base", "Basic_50_TT0.9_TP0.7_TE0.2_V2")}
    for k in (0, 5):
        for i in range(1, 8):
            m[f"Exp1_LA{k}_I{i}"] = os.path.join(EXP1, f"LookAhead_{k}", arm.format(i))
    return m


def exp2_models() -> dict[str, str]:
    """Exp2: Llama-3.2-1B (4-bit NF4) / gpt-4o-mini, three training oracles.

    Iteration coverage is asymmetric on disk, so the manifest lists what exists:
    Q1Q2 has K=0 V1-V6 and K=5 V1-V10; WAI and CSQ-8 have V1-V5 on both arms. CTRL has
    no K=5 arm at all and so cannot enter the K contrast. ``_OLD`` dirs are excluded.
    """
    # (oracle token, on-disk oracle dir, arm-dir template, K=0 iters, K=5 iters)
    specs = [
        ("Q1Q2", "Q1Q2",  "TTree1.2_TT0.9_TP0.7_TE0.2_V{}", range(1, 7), range(1, 11)),
        ("WAI",  "WAI",   "TTree1.2_TT0.9_TP0.7_TE0.1_V{}", range(1, 6), range(1, 6)),
        ("CSQ8", "CSQ-8", "TTree1.2_TT0.9_TP0.7_TE0.1_V{}", range(1, 6), range(1, 6)),
    ]
    m = {"Exp2_Base": os.path.join(EXP2, "Base", "Good_50_TT0.9_TP0.7_TE0.1")}
    for tok, d, arm, it0, it5 in specs:
        for k, iters in ((0, it0), (5, it5)):
            for i in iters:
                m[f"Exp2_{tok}_LA{k}_I{i}"] = os.path.join(EXP2, d, f"LookAhead_{k}", arm.format(i))
    return m


def build_manifest(gen: str) -> dict[str, str]:
    m: dict[str, str] = {}
    if gen in ("all", "exp1"):
        m.update(exp1_models())
    if gen in ("all", "exp2"):
        m.update(exp2_models())
    missing = {k: v for k, v in m.items() if not os.path.isdir(v)}
    if missing:
        print(f"  ! {len(missing)} manifest entries have no directory on disk:")
        for k, v in list(missing.items())[:10]:
            print(f"      {k}  ->  {v}")
    return {k: v for k, v in m.items() if k not in missing}


def oracle_token(model: str) -> str:
    """The TRAINING oracle for a model — the ``oracle=<O>`` partition level.

    Exp1 trained on Q1+Q2 only; Exp2 encodes its training oracle in the model name.
    The base models were not trained at all, so they carry the reward composition the
    generation reports against.
    """
    if model.startswith("Exp1"):
        return "Q1Q2"
    parts = model.split("_")
    return parts[1] if len(parts) > 2 else "Q1Q2"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              LOAD + ESTIMATE                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_all(manifest: dict[str, str]) -> pd.DataFrame:
    names = list(manifest)
    sets = convmod.load_data([manifest[n] for n in names])
    combined = convmod.combine_data(sets, names)
    return combined[combined["conversation"].map(len) > 0].reset_index(drop=True)


def dry_run(combined: pd.DataFrame, metrics: list[str]) -> None:
    """Estimate spend from real prompt lengths — no API calls.

    The oracle prompt is rubric-first, so the fixed ~1,084-token prefix caches at 50%
    off; only the transcript is billed at full rate. Estimated separately here rather
    than assuming a flat per-call price, because Exp1's conversations are much longer
    than Exp3's and the transcript dominates.
    """
    from questionnaires import QuestionnaireID, get_prompt_eval_questionnaire

    qmap = {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2}
    print(f"\n  conversations: {len(combined):,}   metrics: {metrics}   "
          f"calls: {len(combined) * len(metrics):,}")

    total_in = total_cached = 0
    for m in metrics:
        # Prefix length is transcript-independent; measure it once on an empty transcript.
        prefix = len(get_prompt_eval_questionnaire(questionnaire=qmap[m], conversation="")["prompt"])
        chars = combined["conversation"].map(
            lambda u: len(convmod.reconstruct_conversation_text(u))).sum()
        cached_tok = (prefix / 4.0) * len(combined)
        body_tok = chars / 4.0
        total_cached += cached_tok
        total_in += body_tok
        print(f"    {m}: cacheable prefix ~{prefix/4:,.0f} tok/call, "
              f"transcripts ~{body_tok/len(combined):,.0f} tok/call avg")

    out_tok = len(combined) * len(metrics) * 180  # schema-constrained JSON, ~180 tok
    cost = (total_in * PRICE_IN + total_cached * PRICE_IN_CACHED + out_tok * PRICE_OUT) / 1e6
    uncached = (total_in + total_cached) * PRICE_IN / 1e6 + out_tok * PRICE_OUT / 1e6
    print(f"\n  input  ~{total_in/1e6:.2f}M uncached + {total_cached/1e6:.2f}M cacheable")
    print(f"  output ~{out_tok/1e6:.2f}M")
    print(f"\n  ESTIMATED COST: ${cost:.2f}  (${uncached:.2f} if the prefix cache misses)")
    print("  Prices are gpt-4o-mini list rates — verify against the billing dashboard.\n")


def already_done(manifest: dict[str, str], metrics: list[str], judge_tag: str) -> int:
    root = os.path.join(CROSSGEN_ROOT, f"judge={judge_tag}", "rep=0")
    n = 0
    for model in manifest:
        for m in metrics:
            d = eval_csv_dir(root, oracle_token(model), EVAL_QUESTIONNAIRE_DIRS[m], model)
            if os.path.isdir(d):
                n += len([f for f in os.listdir(d) if f.endswith(".csv")])
    return n


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                   MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", choices=["all", "exp1", "exp2"], default="all")
    ap.add_argument("--metrics", nargs="+", default=["Q1", "Q2"],
                    choices=sorted(EVAL_QUESTIONNAIRE_DIRS))
    ap.add_argument("--dry-run", action="store_true", help="estimate cost, make no API calls")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    print(f"cross-generation re-scoring — grader {EVAL_MODEL} @ T={EVAL_TEMPERATURE}")
    manifest = build_manifest(args.gen)
    print(f"  {len(manifest)} model states")

    combined = load_all(manifest)
    per_model = combined.groupby("Model").size()
    if (per_model != 96).any():
        odd = per_model[per_model != 96]
        print(f"  ! {len(odd)} model(s) do not have 96 conversations: {dict(odd.head(10))}")

    if args.dry_run:
        dry_run(combined, args.metrics)
        return 0

    judge_tag = "openai_" + EVAL_MODEL
    root = os.path.join(CROSSGEN_ROOT, f"judge={judge_tag}", "rep=0")
    done = already_done(manifest, args.metrics, judge_tag)
    todo = len(combined) * len(args.metrics) - done
    print(f"  {done:,} cells already scored; {todo:,} to go -> {root}")
    if todo <= 0:
        print("  nothing to do.")
        return 0

    from openai import AsyncOpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        with open(os.path.join(WORKSPACE_ROOT, "openai_key.txt")) as f:
            key = f.read().strip()
    client = AsyncOpenAI(api_key=key)

    from questionnaires import QuestionnaireID

    # Display names carry hyphens ("CSQ-8", "MI-SAT") that enum attributes cannot, so map
    # through the folder basename, which is already the enum-safe spelling.
    def qid_for(display: str):
        return getattr(QuestionnaireID, EVAL_QUESTIONNAIRE_DIRS[display].replace("-", "_"))

    configs = [
        {"name": m, "id": qid_for(m), "q_subdir": EVAL_QUESTIONNAIRE_DIRS[m],
         "model": EVAL_MODEL, "eval_temperature": EVAL_TEMPERATURE}
        for m in args.metrics
    ]
    layout = {name: {"root": root, "oracle": oracle_token(name)} for name in manifest}

    asyncio.run(pipe.run_all_evaluations_async(
        client, combined, configs, layout, concurrency=args.concurrency))
    print(f"\ndone -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
