"""score_miproc.py — the MIPROC (utterance-level MI process coder) sweep, both graders.

Scores every conversation of the arms matching ``MIPROC_ARM_RE`` (default: the GRPO grid,
2 arms x 11 states x 96 = 2,112 conversations) with questionnaire id 10, ``rep=0``. Resume-safe:
both paths skip cells already on disk.

    python tools/score_miproc.py plan              # FREE: what would be scored (per grader coverage)
    python tools/score_miproc.py primary           # PAID (~$1.3 for the GRPO grid): live gpt-4o-mini via the Run_Eval pipeline
    python tools/score_miproc.py submit            # FREE dry run of the Anthropic Message Batches request set
    python tools/score_miproc.py submit --go       # PAID (~$4.6 at the 50 % batch rate): submit for claude-haiku-4-5
    python tools/score_miproc.py wait [seconds]    # poll until every submitted batch has ended (default every 90 s)
    python tools/score_miproc.py collect           # write the ended batches' rows into the lake
    python tools/score_miproc.py poll              # one status line per batch

Then ``python tools/consolidate_scores.py build`` and ``python tools/render_results.py --family
lookahead/process``. Run from ``eda/`` with the repo venv. Widen the arm set with e.g.
``MIPROC_ARM_RE="^(GRPO|PTO)Exp3_LA[05]_(Base|I(?:[1-9]|10))$"`` for all four arms (double the cost).
Keys: ``openai_key.txt`` / ``anthropic_key.txt`` at the experiment root (or the env vars).
"""
import asyncio
import os
import re
import sys
import time

EDA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EDA)
sys.path.insert(0, EDA)

from eda_analysis.scoring import (resolve_paths, get_model_eval_layout, load_data, combine_data,   # noqa: E402
                                  ScoringConfig, EXPERIMENTS)
from eda_analysis.scoring import pipeline as pipe, judge as jc, judge_batch as jb, judge_plan as jp  # noqa: E402
from eda_analysis.constants import WORKSPACE_ROOT  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "plan"
ARM_RE = re.compile(os.environ.get("MIPROC_ARM_RE", r"^GRPOExp3_LA[05]_(Base|I(?:[1-9]|10))$"))
METRICS = ["MIPROC"]
HELD_OUT = jc.JudgeSpec(provider="anthropic", model="claude-haiku-4-5")


def data():
    exps = [e for e in EXPERIMENTS if ARM_RE.match(e.model_name)]
    names = [e.model_name for e in exps]
    comb = combine_data(load_data(resolve_paths(exps)), names)
    comb["id"] = comb["id"].astype(int)
    print(f"{len(exps)} model states, {len(comb):,} conversations: {sorted(names)}")
    return exps, comb


def main() -> None:
    if MODE == "plan":
        exps, comb = data()
        print(jp.check_rubric_parity(METRICS).to_string(index=False))
        print(jp.plan_sweep(HELD_OUT, comb, METRICS, get_model_eval_layout(exps), rep=0))
        primary = get_model_eval_layout(exps, judge_tag="", rep=0)
        from eda_analysis.scoring.registry import eval_csv_dir, EVAL_QUESTIONNAIRE_DIRS
        n_done = sum(len([f for f in os.listdir(d) if f.endswith(".csv")]) if os.path.isdir(d) else 0
                     for m, e in primary.items()
                     for d in [eval_csv_dir(e["root"], e["oracle"], EVAL_QUESTIONNAIRE_DIRS["MIPROC"], m)])
        print(f"primary grader: {n_done:,} / {len(comb):,} MIPROC cells on disk")

    elif MODE == "primary":
        from openai import AsyncOpenAI
        exps, comb = data()
        layout = get_model_eval_layout(exps, judge_tag="", rep=0)
        cfgs = [c for c in pipe.build_default_eval_configs(ScoringConfig()) if c["name"] == "MIPROC"]
        assert len(cfgs) == 1, cfgs
        key = os.environ.get("OPENAI_API_KEY") or open(os.path.join(WORKSPACE_ROOT, "openai_key.txt"),
                                                        encoding="utf-8").read().strip()
        t0 = time.time()
        res = asyncio.run(pipe.run_all_evaluations_async(AsyncOpenAI(api_key=key), comb, cfgs, layout,
                                                         concurrency=int(os.environ.get("MIPROC_CONC", "64"))))
        print(f"done in {time.time() - t0:.0f}s:", res)

    elif MODE == "submit":
        exps, comb = data()
        go = "--go" in sys.argv
        ids = jb.submit_sweep(HELD_OUT, comb, METRICS, get_model_eval_layout(exps), rep=0, dry_run=not go)
        print("submitted batch ids:" if go else "DRY RUN — would submit:", ids)

    elif MODE == "poll":
        print(jb.poll_batches(HELD_OUT, rep=0).to_string())

    elif MODE == "wait":
        every = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        while True:
            st = jb.poll_batches(HELD_OUT, rep=0)
            live = st[~st["collected"].astype(bool)] if "collected" in st.columns else st
            line = "; ".join(f"{r.batch_id[-8:]}:{r.status} ok={getattr(r, 'succeeded', '?')} "
                             f"err={getattr(r, 'errored', '?')} proc={getattr(r, 'processing', '?')}"
                             for r in live.itertuples())
            print(time.strftime("%H:%M:%S"), line or "no live batches", flush=True)
            if live.empty or (live["status"] == "ended").all():
                print("ALL_ENDED", flush=True)
                break
            time.sleep(every)

    elif MODE == "collect":
        print(jb.collect_batches(HELD_OUT, rep=0))

    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
