"""score_partial.py — the training oracle's Q1/Q2 on every PREFIX of the shared Base conversations.

WHY
---
The GRPO paper's faithfulness figure asks how well the oracle's score of a SHORT conversation
prefix (what a training reward sees) agrees with its score of the whole session (what the
evaluation sees). Both GRPO arms trained at ``MCL=12``, so the training reward never scored a
prefix shorter than 12 utterances; those scores do not exist anywhere on disk. This script makes
them, on one consistent grid: every prefix of the paper's shared Base, the untrained model's
``model_iter_0`` draws of BOTH GRPO arms pooled (2 x 96 = 192 conversations, two per persona).

A prefix is ``utterances[:n]`` for every even ``n = 2, 4, 6, ...`` up to the conversation length,
so it always ends on a patient turn (utterance 0 is the therapist; checked against each file's own
``role`` column before anything is counted). When a conversation ends on a patient turn its last
prefix IS the whole session, and it is kept: it re-asks the oracle a question the score lake
already holds, which gives the oracle's agreement with its own full-session score.

THE CALL
--------
``pipeline.evaluate_conversation(client, utterances[:n], QuestionnaireID.Q1 | Q2)`` with
``registry.EVAL_MODEL`` / ``EVAL_TEMPERATURE``, the same function ``Run_Eval``'s writer used to
fill the lake's Q1/Q2 (gpt-4o-mini-2024-07-18, T=0.1, seed=42, strict json_schema). The proxy is
``mean(Q1_Mean, Q2_Mean)`` at the same ``(Model, conversation_id, n_turns)``. Same rule as the lake
writer: a row with any NaN is not written, so the next run retries it.

WHERE IT WRITES
---------------
``data/eval_scores/_partial/judge=openai_gpt-4o-mini-2024-07-18/rep=0/metric=<Q1|Q2>/oracle=none/
<Model>/<conversation_id>_t<n>.csv`` (one row per call: the lake's Q1/Q2 columns + ``n_turns``).

The ``_partial`` prefix keeps it OUT of the score lake proper, as ``_crossgen`` does: nothing in
the analysis layer globs it, so no tracked result can move because of it, while it still sits in
the Drive-backed ``eval_scores`` symlink and so is backed up. ``conversation_id`` is the FILE
INDEX (``conversation_<id>.csv``), not the persona (each iteration's 96 are shuffled; pair to
personas through ``eda_analysis.data``). Ids 0-95 repeat across the two Base draws, and the
``<Model>`` folder (``GRPOExp3_LA0_Base`` / ``GRPOExp3_LA5_Base``) keeps them apart.

Resume-safe: a CSV that exists is skipped. Each CSV is written to ``.tmp`` and renamed, so an
interrupted run never leaves a half-written file that a resume would count as done.

USAGE
-----
    python tools/score_partial.py                    # DRY RUN (the default): counts, tokens, cost, resume state; no API
    python tools/score_partial.py --limit 5          # dry run of a 5-prefix (10-call) test
    python tools/score_partial.py --run --limit 5    # PAID: that test (the first 5 prefixes in arm / id / n order)
    python tools/score_partial.py --run              # PAID: everything not yet on disk

Run with the repo venv (``.venv/Scripts/python.exe``); the working directory does not matter.
The OpenAI client is created only under ``--run`` (key: env ``OPENAI_API_KEY`` or
``openai_key.txt`` at the experiment root).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_EDA = os.path.dirname(_HERE)
if _EDA not in sys.path:
    sys.path.insert(0, _EDA)

# constants is the package leaf: importing it prepends code/ to sys.path so the canonical
# `questionnaires` resolves. Must happen before the scoring layer is imported.
from eda_analysis.constants import EVAL_SCORES, PRIMARY_JUDGE_TAG, WORKSPACE_ROOT  # noqa: E402
from eda_analysis.data import discover_arms                                       # noqa: E402
from eda_analysis.scoring import conversations as convmod                          # noqa: E402
from eda_analysis.scoring import pipeline as pipe                                  # noqa: E402
from eda_analysis.scoring.judge_plan import pricing_for                            # noqa: E402
from eda_analysis.scoring.registry import (                                        # noqa: E402
    EVAL_MODEL, EVAL_QUESTIONNAIRE_DIRS, EVAL_TEMPERATURE, eval_csv_dir, eval_scores_root,
)
from questionnaires import QuestionnaireID, get_prompt_eval_questionnaire         # noqa: E402

JUDGE_TAG = "openai_" + EVAL_MODEL
assert JUDGE_TAG == PRIMARY_JUDGE_TAG, (JUDGE_TAG, PRIMARY_JUDGE_TAG)

PARTIAL_ROOT = os.path.join(EVAL_SCORES, "_partial", f"judge={JUDGE_TAG}", "rep=0")
LAKE_ROOT = eval_scores_root("", 0)            # the primary grader's full-session scores
ORACLE = "none"                                # the Base is untrained: the lake's oracle label
METRICS = ("Q1", "Q2")
QIDS = {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2}
DEFAULT_ARMS = ("GRPO_LA0", "GRPO_LA5")        # the GRPO paper's shared Base = both iter-0 draws

# Token model for the estimate (gpt-4o-mini tokenizes with o200k_base). Per call: the real prompt's
# tokens + FRAMING_TOK of chat/schema framing; output is the schema-closed JSON score array.
FRAMING_TOK = 95
OUT_TOK = {"Q1": 19, "Q2": 43}
CACHE_STEP = 128                               # OpenAI caches in 128-token steps above the minimum

_REPO = os.path.dirname(WORKSPACE_ROOT)
_VENV_PY = os.path.join(_REPO, ".venv", "Scripts", "python.exe")


@dataclass(frozen=True)
class Prefix:
    arm: str              # arm label, e.g. "GRPO_LA0"
    model: str            # score-lake model name, e.g. "GRPOExp3_LA0_Base"
    conv_id: int          # file index of conversation_<id>.csv (NOT the persona)
    n: int                # utterances in the prefix; even, so it ends on a patient turn
    length: int           # utterances in the whole conversation
    utterances: Tuple[str, ...]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                   LOAD                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _check_speaker_parity(conv_dir: str, ids: Sequence[int]) -> None:
    """``reconstruct_conversation_text`` labels utterance i THERAPIST when i is even, so an
    even-length prefix ends on a patient turn only if each file's own ``role`` column agrees."""
    bad = []
    for i in ids:
        roles = pd.read_csv(os.path.join(conv_dir, f"conversation_{i}.csv"),
                            usecols=["role"])["role"].tolist()
        if roles != ["therapist" if j % 2 == 0 else "patient" for j in range(len(roles))]:
            bad.append(i)
    if bad:
        raise SystemExit(f"  ! {conv_dir}: role column breaks therapist/patient alternation in "
                         f"conversations {bad[:10]} — prefix parity would be wrong; aborting.")


def load_prefixes(labels: Sequence[str]) -> Tuple[Dict[str, dict], List[Prefix]]:
    found = {a.label: a for a in discover_arms()}
    missing = [lab for lab in labels if lab not in found]
    if missing:
        raise SystemExit(f"arm(s) not on disk: {missing}; discovered: {sorted(found)}")

    info: Dict[str, dict] = {}
    prefixes: List[Prefix] = []
    for lab in labels:
        arm = found[lab]
        d = arm.conv_dir(0)
        if d is None:
            raise SystemExit(f"{lab}: no model_iter_0 conversations on disk")
        convs = convmod.load_data([d])[0].sort_values("id").reset_index(drop=True)
        _check_speaker_parity(d, convs["id"].astype(int).tolist())
        non_str = int(convs["conversation"].map(
            lambda u: sum(not isinstance(x, str) for x in u)).sum())
        model = arm.model_name(0)
        for _, row in convs.iterrows():
            utts = tuple(row["conversation"])
            for n in range(2, len(utts) + 1, 2):
                prefixes.append(Prefix(lab, model, int(row["id"]), n, len(utts), utts[:n]))
        info[lab] = {"arm": arm, "model": model, "dir": d, "n_convs": len(convs),
                     "non_str": non_str}
    return info, prefixes


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              PATHS + RESUME                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def out_dir(model: str, metric: str) -> str:
    return eval_csv_dir(PARTIAL_ROOT, ORACLE, EVAL_QUESTIONNAIRE_DIRS[metric], model)


def out_name(p: Prefix) -> str:
    return f"{p.conv_id}_t{p.n}.csv"


def _csvs_in(d: str) -> set:
    return {f for f in os.listdir(d) if f.endswith(".csv")} if os.path.isdir(d) else set()


def existing(prefixes: Sequence[Prefix]) -> Dict[Tuple[str, str], set]:
    """``{(model, metric): {csv names on disk}}`` — one listdir per folder, not one stat per
    file (the folders sit behind the Drive symlink)."""
    return {(m, q): _csvs_in(out_dir(m, q))
            for m in sorted({p.model for p in prefixes}) for q in METRICS}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                 ESTIMATE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _prompt(metric: str, utterances: Sequence[str]) -> str:
    conv_str = convmod.reconstruct_conversation_text(list(utterances))
    return get_prompt_eval_questionnaire(questionnaire=QIDS[metric], conversation=conv_str)["prompt"]


def fixed_prefix_tokens(enc, metric: str) -> int:
    """Tokens of the transcript-INDEPENDENT leading prompt text: the common string prefix of the
    prompt built on two different transcripts. That span is what a prefix-match cache can serve."""
    a = _prompt(metric, ["Hello.", "Hi."])
    b = _prompt(metric, ["Good morning, what brings you in today?", "My doctor sent me."])
    return len(enc.encode(os.path.commonprefix([a, b])))


def cached_tokens(prefix_tok: int, min_cache: int) -> int:
    """Tokens OpenAI would serve from cache for a shared prefix of ``prefix_tok`` tokens:
    nothing below the minimum, then whole 128-token steps from it."""
    if prefix_tok < min_cache:
        return 0
    return min_cache + CACHE_STEP * ((prefix_tok - min_cache) // CACHE_STEP)


def estimate(cells: Sequence[Tuple[Prefix, str]], tok: Dict[Tuple[str, int, int, str], int],
             cache_tok: Dict[str, int]) -> dict:
    pr = pricing_for(EVAL_MODEL)
    r = {"calls": Counter(), "in": Counter(), "out": Counter(), "cached": Counter()}
    for p, q in cells:
        r["calls"][q] += 1
        r["in"][q] += tok[(p.model, p.conv_id, p.n, q)] + FRAMING_TOK
        r["out"][q] += OUT_TOK[q]
        r["cached"][q] += cache_tok[q]
    r["list_usd"] = {q: (r["in"][q] * pr.input_usd_per_mtok
                         + r["out"][q] * pr.output_usd_per_mtok) / 1e6 for q in METRICS}
    r["cached_usd"] = {q: r["list_usd"][q] - r["cached"][q] * pr.input_usd_per_mtok
                       * (1.0 - pr.cache_read_mult) / 1e6 for q in METRICS}
    return r


def _usd(x: float) -> str:
    """Dollars to the cent, or to 1/100 cent for a --limit test that would round to $0.00."""
    return f"{x:.2f}" if x >= 0.10 else f"{x:.4f}"


def run_command(args) -> str:
    """The billed command line, carrying every non-default flag of this invocation."""
    cmd = [_VENV_PY.replace("\\", "/"), os.path.abspath(__file__).replace("\\", "/"), "--run"]
    if tuple(args.arms) != DEFAULT_ARMS:
        cmd += ["--arms", *args.arms]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.concurrency != 32:
        cmd += ["--concurrency", str(args.concurrency)]
    return " ".join(cmd)


def dry_run(args, info: Dict[str, dict], prefixes: List[Prefix],
            cells: List[Tuple[Prefix, str]], todo: List[Tuple[Prefix, str]],
            have: Dict[Tuple[str, str], set]) -> None:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    pr = pricing_for(EVAL_MODEL)

    # ── what is there ────────────────────────────────────────────────────────
    print("\n  conversations (model_iter_0) per arm:")
    for lab, d in info.items():
        print(f"    {lab:<9} {d['model']:<18} {d['n_convs']:>3} conversations   <- {d['dir']}")
        if d["n_convs"] != 96:
            print(f"      ! expected 96")
        if d["non_str"]:
            print(f"      ! {d['non_str']} non-string utterances (rendered as 'nan', as in the lake)")

    lake_full = {lab: _csvs_in(eval_csv_dir(LAKE_ROOT, ORACLE, "Q1", d["model"]))
                 for lab, d in info.items()}
    sel = [p for p, q in cells if q == METRICS[0]]          # the selected prefixes, once each
    print(f"\n  prefixes (n_turns = 2, 4, ... up to the conversation length; every one ends on a "
          f"patient turn):")
    print(f"    {'arm':<9} {'prefixes':>8} {'< MCL':>6} {'>= MCL':>7} {'whole-session':>14} "
          f"{'n range':>8}")
    tot = Counter()
    for lab, d in info.items():
        ps = [p for p in sel if p.arm == lab]
        mcl = d["arm"].mcl
        short = sum(p.n < mcl for p in ps)
        whole = [p for p in ps if p.n == p.length]
        whole_in_lake = sum(f"{p.conv_id}.csv" in lake_full[lab] for p in whole)
        rng = f"{min(p.n for p in ps)}-{max(p.n for p in ps)}" if ps else "-"
        print(f"    {lab:<9} {len(ps):>8,} {short:>6,} {len(ps) - short:>7,} {len(whole):>14,} "
              f"{rng:>8}   (MCL={mcl}; {whole_in_lake} whole sessions have a lake Q1 score)")
        tot.update(prefixes=len(ps), short=short, whole=len(whole))
    print(f"    {'total':<9} {tot['prefixes']:>8,} {tot['short']:>6,} "
          f"{tot['prefixes'] - tot['short']:>7,} {tot['whole']:>14,}")
    if args.limit is not None:
        print(f"    (--limit {args.limit}: the first {len(sel):,} of {len(prefixes):,} prefixes "
              f"in arm / conversation id / n order)")
    print(f"\n  calls: {len(sel):,} prefixes x {len(METRICS)} rubrics ({' + '.join(METRICS)}) "
          f"= {len(cells):,}")

    # ── tokens + dollars ─────────────────────────────────────────────────────
    t0 = time.time()
    tok = {(p.model, p.conv_id, p.n, q): len(enc.encode(_prompt(q, p.utterances)))
           for p, q in cells}
    fixed = {q: fixed_prefix_tokens(enc, q) for q in METRICS}
    cache_tok = {q: cached_tokens(fixed[q], pr.min_cache_tokens) for q in METRICS}
    print(f"\n  tokens (tiktoken o200k_base on the real prompts, {time.time() - t0:.1f}s; "
          f"+{FRAMING_TOK} framing/call; output Q1 {OUT_TOK['Q1']}, Q2 {OUT_TOK['Q2']}/call):")
    for q in METRICS:
        print(f"    {q}: transcript-independent prefix {fixed[q]:,} tok -> "
              + (f"caches: {cache_tok[q]:,} tok/call at {pr.cache_read_mult:.0%} price "
                 f"(whole {CACHE_STEP}-tok steps from {pr.min_cache_tokens:,})" if cache_tok[q]
                 else f"does NOT cache (< {pr.min_cache_tokens:,}-tok minimum)"))

    full = estimate(cells, tok, cache_tok)
    print(f"\n    {'rubric':<6} {'calls':>6} {'input tok':>12} {'mean/call':>10} "
          f"{'output tok':>11} {'cached tok':>11} {'$ list':>8} {'$ cached':>9}")
    for q in METRICS:
        c = full["calls"][q]
        print(f"    {q:<6} {c:>6,} {full['in'][q]:>12,} {full['in'][q] / max(c, 1):>10,.0f} "
              f"{full['out'][q]:>11,} {full['cached'][q]:>11,} {_usd(full['list_usd'][q]):>8} "
              f"{_usd(full['cached_usd'][q]):>9}")
    tin, tout, tc = (sum(full[k].values()) for k in ("in", "out", "cached"))
    tl, tcu = sum(full["list_usd"].values()), sum(full["cached_usd"].values())
    print(f"    {'total':<6} {sum(full['calls'].values()):>6,} {tin:>12,} "
          f"{tin / max(len(cells), 1):>10,.0f} {tout:>11,} {tc:>11,} {_usd(tl):>8} {_usd(tcu):>9}")
    cached_q = [q for q in METRICS if cache_tok[q]]
    print(f"\n  ESTIMATED COST: ${_usd(tl)} at list price; ${_usd(tcu)} with the "
          f"{'+'.join(cached_q) or 'no'} prefix cached")
    print(f"  (gpt-4o-mini {pr.input_usd_per_mtok:.2f} in / {pr.output_usd_per_mtok:.2f} out "
          f"USD per 1M tok, judge_plan.JUDGE_PRICING; verify against the billing dashboard)")

    # ── resume ───────────────────────────────────────────────────────────────
    print(f"\n  already scored (resume) under {PARTIAL_ROOT}:")
    for q in METRICS:
        n_sel = sum(1 for _, m in cells if m == q)
        n_have = sum(1 for p, m in cells if m == q and out_name(p) in have[(p.model, q)])
        n_any = sum(len(have[(mo, q)]) for mo in {p.model for p in prefixes})
        print(f"    {q}: {n_have:,} / {n_sel:,} selected cells on disk"
              + (f"  ({n_any:,} CSVs in the folders overall)" if n_any != n_have else ""))
    left = estimate(todo, tok, cache_tok)
    print(f"    left to score: {len(todo):,} calls  (~${_usd(sum(left['list_usd'].values()))} list, "
          f"~${_usd(sum(left['cached_usd'].values()))} cached)")

    leaked = [m for m in ("openai", "anthropic") if m in sys.modules]
    assert not leaked, f"dry run imported an API client module: {leaked}"
    print("\n  DRY RUN — no API calls made; no openai/anthropic module imported.")
    print("  To score for real:\n")
    print(f"    {run_command(args)}\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                               SCORE (PAID)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def score(todo: List[Tuple[Prefix, str]], concurrency: int) -> Counter:
    """BILLED. Score each (prefix, rubric) cell with the lake's own call; skip files on disk."""
    from openai import AsyncOpenAI     # the only API import in this file; --run path only

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        with open(os.path.join(WORKSPACE_ROOT, "openai_key.txt"), encoding="utf-8") as f:
            key = f.read().strip()
    client = AsyncOpenAI(api_key=key)

    for d in {out_dir(p.model, q) for p, q in todo}:
        os.makedirs(d, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    stats: Counter = Counter()
    t0 = time.time()

    async def one(p: Prefix, q: str) -> None:
        fp = os.path.join(out_dir(p.model, q), out_name(p))
        if os.path.exists(fp):                         # landed since the plan was made
            stats["existing"] += 1
            return
        try:
            async with sem:
                rdf = await pipe.evaluate_conversation(
                    client, conversation=list(p.utterances), questionnaire_id=QIDS[q],
                    model=EVAL_MODEL, eval_temperature=EVAL_TEMPERATURE)
            if rdf is None or rdf.isnull().values.any():
                stats["incomplete"] += 1                # not written -> retried next run
                return
            rdf["n_turns"] = p.n
            tmp = fp + ".tmp"
            rdf.to_csv(tmp, index=False)
            os.replace(tmp, fp)
            stats["written"] += 1
        except Exception as e:                          # one bad cell must not sink the run
            print(f"  error {q} {p.model}/{out_name(p)}: {e}")
            stats["errors"] += 1
        done = sum(stats.values())
        if done % 200 == 0:
            print(f"  {done:,}/{len(todo):,}  ({dict(stats)})  {time.time() - t0:.0f}s")

    try:
        await asyncio.gather(*(one(p, q) for p, q in todo))
    finally:
        await client.close()
    return stats


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                   MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="count, estimate tokens + cost, report resume state; no API (DEFAULT)")
    mode.add_argument("--run", action="store_true", help="BILLED: score with gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=32, help="in-flight oracle calls (32)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N prefixes (arm / conversation id / n order); 2N calls")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                    help="arm labels whose model_iter_0 to use (default: GRPO_LA0 GRPO_LA5)")
    args = ap.parse_args()

    seed = re.search(r"seed=(\d+)", inspect.getsource(pipe.call_openai_json))
    print(f"partial-conversation oracle scoring — {EVAL_MODEL} @ T={EVAL_TEMPERATURE}, "
          f"seed={seed.group(1) if seed else '?'}, strict json_schema (pipeline.evaluate_conversation)")
    print(f"  mode: {'RUN (billed)' if args.run else 'DRY RUN (no API calls)'}   "
          f"arms: {' '.join(args.arms)}   concurrency: {args.concurrency}")

    info, prefixes = load_prefixes(args.arms)
    selected = prefixes if args.limit is None else prefixes[:max(args.limit, 0)]
    cells = [(p, q) for p in selected for q in METRICS]
    have = existing(prefixes)
    todo = [(p, q) for p, q in cells if out_name(p) not in have[(p.model, q)]]

    if not args.run:
        dry_run(args, info, prefixes, cells, todo, have)
        return 0

    print(f"  {len(cells) - len(todo):,} of {len(cells):,} cells already on disk; "
          f"{len(todo):,} to score -> {PARTIAL_ROOT}")
    if not todo:
        print("  nothing to do.")
        return 0
    t0 = time.time()
    stats = asyncio.run(score(todo, args.concurrency))
    print(f"\ndone in {time.time() - t0:.0f}s: {dict(stats)} -> {PARTIAL_ROOT}")
    if stats["incomplete"] or stats["errors"]:
        print("  re-run the same command to retry the cells that did not land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
