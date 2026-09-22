"""Score the Exp4 lake with the held-out gpt-4o-mini judge, locally, headless.

The command-line twin of ``notebooks/scoring/Run_Eval.ipynb`` at ``JUDGE_PRESET = "gpt"``. It runs
the SAME functions in the SAME order -- sanity gate, plan, run, verify, prompt-length gate -- so
this file is a driver, not a second implementation. It exists because the vendor judge needs no
GPU and no vLLM: it is pure HTTP, so it belongs in a terminal that can run for an hour rather than
in a notebook kernel.

⚠ THIS BILLS. Every call is a real OpenAI request. ``--dry-run`` prints the plan and its cost and
makes no call; ``--sanity-only`` runs just the ~24-call gate. A partition already on disk is
skipped, so an interrupted run resumes for free.

    python tools/score_gpt_local.py --dry-run
    python tools/score_gpt_local.py --sanity-only
    python tools/score_gpt_local.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_EDA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDA_DIR not in sys.path:
    sys.path.insert(0, _EDA_DIR)

import pandas as pd

from eda_analysis import data, scoring
from eda_analysis.constants import N_PERSONAS

from core.runtime import authenticate                               # noqa: E402
from tools import oracle_sanity                                     # noqa: E402

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_PROVIDER = "openai"
REP = 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="plan + cost only; no calls")
    ap.add_argument("--sanity-only", action="store_true", help="run the gate, then stop")
    ap.add_argument("--skip-sanity", action="store_true", help="the gate already passed today")
    ap.add_argument("--concurrency", type=int, default=scoring.DEFAULT_CONCURRENCY)
    ap.add_argument("--metrics", nargs="*", default=None, help="default: all eight stored")
    args = ap.parse_args(argv)

    max_tokens = scoring.JUDGE_MAX_TOKENS
    max_retries = scoring.JUDGE_MAX_RETRIES
    request_timeout = scoring.JUDGE_REQUEST_TIMEOUT

    if not args.dry_run:
        authenticate(hf=False, openai=True)
    judge = scoring.judge_binding_for(
        JUDGE_MODEL, provider=JUDGE_PROVIDER,
        request_timeout=request_timeout, max_retries=max_retries,
    )
    judge_tag = scoring.judge_tag(judge)
    print(f"judge      : {judge.provider}:{judge.model}  ->  judge={judge_tag}, rep={REP}")
    print(f"call policy: concurrency={args.concurrency} max_tokens={max_tokens} "
          f"max_retries={max_retries} timeout={request_timeout:.0f}s")
    print("*** VENDOR JUDGE -- every call below is BILLED ***\n")

    arms = data.discover_arms()
    metric_keys = list(scoring.STORED_METRICS) if args.metrics is None else list(args.metrics)
    print(f"arms       : {len(arms)}  [{', '.join(a.label for a in arms)}]")
    print(f"states     : {sum(len(a.iters) for a in arms)}")
    print(f"metrics    : {', '.join(metric_keys)}\n")

    # -- 1. the sanity gate -------------------------------------------------------------------
    if not args.dry_run and not args.skip_sanity:
        print("--- SANITY GATE (12 fixture transcripts x Q1,Q2 ~ 24 calls) ---")
        report = scoring.run_async(oracle_sanity.run_sanity(
            judge, questionnaire_ids=(1, 2), quick=False, concurrency=args.concurrency,
            max_tokens=max_tokens, max_retries=max_retries,
            request_timeout=request_timeout, progress=True,
        ))
        print(oracle_sanity.format_report(report))
        passed, reasons = oracle_sanity.check_gates(report)
        if not passed:
            print("\nORACLE SANITY FAILED -- refusing to score:\n  " + "\n  ".join(reasons))
            return 1
        print(f"\nSANITY GATE PASSED -- {judge.provider}:{judge.model} honours the schema and "
              f"separates the fixture.\n")
        if args.sanity_only:
            return 0
    elif args.sanity_only:
        print("--sanity-only with --dry-run/--skip-sanity: nothing to do.")
        return 0

    # -- 2. the plan --------------------------------------------------------------------------
    plan = scoring.discover_scorable(arms, judge=judge_tag, rep=REP, metrics=metric_keys)
    estimate = scoring.estimate_calls(plan, binding=judge, max_retries=max_retries)
    print(f"partitions : {estimate['n_partitions']}")
    print(f"calls      : {estimate['n_calls']:,}  "
          f"(worst case with retries {estimate['n_calls_worst_case']:,})")
    print(f"cost       : {estimate['cost']}")
    for note in estimate["notes"]:
        print(f"  note: {note}")
    if plan.empty:
        print("\nNothing to score -- every partition already exists. Skipping to verify.")
    if args.dry_run:
        print("\n--dry-run: stopping before any billed call.")
        return 0

    # -- 3. the run ---------------------------------------------------------------------------
    if not plan.empty:
        print(f"\n--- SCORING {len(plan)} partitions ---", flush=True)
        started = time.time()
        results = scoring.run_async(scoring.run_scoring(
            plan, binding=judge, rep=REP, concurrency=args.concurrency, dry_run=False,
            progress=True, max_tokens=max_tokens, max_retries=max_retries,
            request_timeout=request_timeout, min_success_ratio=scoring.MIN_SUCCESS_RATIO,
        ))
        print("\n" + results["status"].value_counts().to_string())
        written = results[results["status"] == "written"]
        if not written.empty:
            print(f"\nwrote {len(written)} partition(s), {int(written['n_rows'].sum()):,} rows, "
                  f"wall-clock {(time.time() - started) / 60.0:.1f} min")
        errors = results[results["status"] == "error"]
        if not errors.empty:
            print(f"\n{len(errors)} partition(s) FAILED (they stay in the plan; re-run to retry "
                  f"only these):")
            print(errors[["arm_label", "model_state", "metric", "error"]].to_string())

    # -- 4. verify ----------------------------------------------------------------------------
    remaining = scoring.discover_scorable(arms, judge=judge_tag, rep=REP, metrics=metric_keys)
    if not remaining.empty:
        print(f"\n{len(remaining)} partition(s) still missing:")
        print(remaining[["arm_label", "model_state", "metric",
                         "n_conversations", "n_rows_existing"]].to_string())
        return 1
    print(f"\nPLAN IS EMPTY: every partition exists for judge={judge_tag}, rep={REP}. "
          f"A re-run would make zero calls.")

    scores = data.load_scores_long(arms, metric_keys, judge=judge_tag, rep=REP,
                                   attach_persona=False, cache=False)
    frame = scores.copy()
    frame["graded"] = (frame["oracle_success"].fillna(False).astype(bool)
                       if "oracle_success" in frame.columns else frame["score"].notna())
    coverage = (frame.groupby(["arm_label", "model_state", "metric"], as_index=False)
                     .agg(rows=("persona_id", "size"), personas=("persona_id", "nunique"),
                          graded=("graded", "sum"), mean_score=("score", "mean")))
    coverage["ungraded"] = coverage["rows"] - coverage["graded"]
    per_arm = (coverage.groupby("arm_label", as_index=False)
                       .agg(states=("model_state", "nunique"), partitions=("metric", "size"),
                            min_personas=("personas", "min"), max_personas=("personas", "max"),
                            ungraded_rows=("ungraded", "sum")))
    print(f"\nPer-arm coverage (judge={judge_tag}, rep={REP}; complete = "
          f"{len(metric_keys)} metrics x {N_PERSONAS} personas):")
    print(per_arm.to_string(index=False))
    ungraded = int(coverage["ungraded"].sum())
    print(f"\nungraded rows across the lake: {ungraded}"
          + ("  (visible NaN holes, not absences)" if ungraded else "  -- none"))

    # -- 5. the prompt-length gate ------------------------------------------------------------
    transcripts = scoring.gather_transcripts(arms)
    length_report = scoring.prompt_length_gate(
        transcripts, max_model_len=16384, questionnaire_ids=(1, 2), model=judge.model,
        base_url=None,
    )
    fmt = getattr(oracle_sanity, "format_prompt_length_report", None)
    print("\n--- PROMPT-LENGTH REPORT (vendor judge: measured against the 16384 literal) ---")
    print(fmt(length_report) if fmt else length_report)
    ok, messages = scoring.check_prompt_length_gate(length_report)
    for m in messages:
        print("  " + m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
