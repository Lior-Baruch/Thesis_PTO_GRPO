"""judge_batch.py — the PAID full-sweep path: score every conversation with a second judge via
the **Anthropic Message Batches API** (50% off list price) instead of live streaming calls.

Why batch, and why only Anthropic:

- The second-judge sweep is 22,272 calls per rep and has **no latency requirement whatsoever** —
  it feeds an offline EDA. Trading 24h turnaround (usually <1h) for 50% off is free money, and it
  is the single largest cost lever available now that prompt caching is off the table for this
  workload (see ``judge_plan.prefix_report``: 6 of 8 rubrics have a sub-1,024-token prefix, and
  Haiku 4.5's cache minimum is 4,096 — the second judge never caches).
- The PRIMARY judge already has a complete rep on disk (all 22,272 cells), and the analysis in
  ``reliability.variance_components`` shows extra reps buy essentially nothing at the arm-mean
  level (rep noise contributes ~0.01 to an arm mean vs ~0.09 from persona sampling). The handful
  of extra reps worth buying are MICI-only and cheap, so they can stay on the existing live async
  path in :mod:`judge`. No OpenAI batch implementation is needed; adding one would be dead code.

**Three-phase, resume-safe by construction.** Submission and collection are separate calls so a
dropped connection, a closed laptop, or a new session never loses work:

    plan  = judge_plan.plan_sweep(...)          # free: what still needs scoring
    ids   = submit_sweep(...)                   # PAID: creates batches, writes manifests
    ...                                          # (walk away; hours later, new kernel is fine)
    poll_batches(judge)                          # free: status table
    collect_batches(judge)                       # free: writes the CSVs

State lives beside the scores, never in memory:

    data/eval_scores_by_judge/_batches/<judge_tag>/rep=<r>/<batch_id>.json    # manifest + status

``custom_id`` is a bare index (``c000123``) into that manifest rather than an encoded path — model
names plus metric plus oracle overflow the 64-char ``custom_id`` limit, and an opaque index cannot
be silently truncated into a collision that writes a score to the wrong model's folder.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import judge as _judge
from . import judge_plan as _plan
from . import pipeline as _pipeline
from . import registry as _registry
from .conversations import reconstruct_conversation_text

BATCH_STATE_ROOT = os.path.join(_judge.JUDGE_CHECK_ROOT, "_batches")

# Anthropic caps a batch at 100,000 requests / 256 MB. We chunk far below both so that progress is
# visible, a single rejected batch costs little, and each manifest stays comfortably loadable.
MAX_REQUESTS_PER_BATCH = 5_000


def _name_to_qid() -> dict:
    from questionnaires import QuestionnaireID
    return {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2,
            "WAI-SR": QuestionnaireID.WAI_SR, "CSQ-8": QuestionnaireID.CSQ8,
            "MI-SAT": QuestionnaireID.MI_SAT, "MITI": QuestionnaireID.MITI,
            "PCT": QuestionnaireID.PCT, "MICI": QuestionnaireID.MICI}


def _state_dir(judge_tag: str, rep: int) -> str:
    d = os.path.join(BATCH_STATE_ROOT, judge_tag, f"rep={rep}")
    os.makedirs(d, exist_ok=True)
    return d


# ── request construction ──────────────────────────────────────────────────────

@dataclass
class _Cell:
    custom_id: str
    out_path: str
    metric: str
    model: str
    oracle: str
    file_index: int
    n_therapist_utt: int     # MICI's rate denominator — stored so collection needs no conv data
    params: dict


def build_requests(judge: "_judge.JudgeSpec", combined_data: pd.DataFrame,
                   questionnaire_names: Sequence[str],
                   model_layout: Dict[str, Dict[str, str]], *, rep: int = 0,
                   subset_n: Optional[int] = None) -> List[_Cell]:
    """Build one batch request per (model, metric, conversation) cell still missing from disk.

    Skips cells whose CSV already exists — the same resume rule the live runner uses, applied
    before a single token is bought.
    """
    if not _pipeline.EVAL_CODE_AVAILABLE:
        raise RuntimeError("questionnaires module not importable — run from eda/ with code/ on sys.path")
    if judge.provider != "anthropic":
        raise ValueError(f"Batch path is Anthropic-only; got provider={judge.provider!r}. "
                         "Use judge.run_judge_scoring for the OpenAI live path.")
    qid_of = _name_to_qid()
    cells: List[_Cell] = []
    idx = 0
    for model in combined_data["Model"].unique():
        entry = model_layout.get(str(model))
        oracle = entry["oracle"] if entry else "none"
        sub = combined_data[combined_data["Model"] == model]
        if subset_n is not None:
            sub = sub.iloc[:subset_n]
        for qname in questionnaire_names:
            out_dir = _judge.judge_out_dir(judge.tag, rep,
                                           _registry.EVAL_QUESTIONNAIRE_DIRS[qname], oracle, str(model))
            os.makedirs(out_dir, exist_ok=True)
            for _, row in sub.iterrows():
                out_fp = os.path.join(out_dir, f"{row['id']}.csv")
                if os.path.exists(out_fp):
                    continue
                conv_str = reconstruct_conversation_text(row["conversation"])
                ed = _pipeline.get_prompt_eval_questionnaire(questionnaire=qid_of[qname],
                                                             conversation=conv_str)
                params = {
                    "model": judge.model,
                    "max_tokens": judge.max_tokens,
                    "messages": [{"role": "user", "content": ed["prompt"]}],
                    "output_config": {"format": {
                        "type": "json_schema",
                        "schema": _judge._strip_unsupported_constraints(ed["schema"])}},
                }
                if judge.thinking is not None:
                    params["thinking"] = judge.thinking
                cells.append(_Cell(custom_id=f"c{idx:06d}", out_path=out_fp, metric=qname,
                                   model=str(model), oracle=oracle, file_index=int(row["id"]),
                                   n_therapist_utt=_pipeline._count_therapist_utterances(conv_str),
                                   params=params))
                idx += 1
    return cells


# ── phase 1: submit ───────────────────────────────────────────────────────────

def submit_sweep(judge: "_judge.JudgeSpec", combined_data: pd.DataFrame,
                 questionnaire_names: Sequence[str], model_layout: Dict[str, Dict[str, str]],
                 *, rep: int = 0, subset_n: Optional[int] = None,
                 max_per_batch: int = MAX_REQUESTS_PER_BATCH,
                 dry_run: bool = True) -> List[str]:
    """**PAID.** Create Message Batches for every cell still missing on disk.

    ``dry_run=True`` (the default, deliberately) builds every request and reports what WOULD be
    submitted without calling the API — run it once and read the count before flipping the switch.
    Returns the created batch ids; state is persisted per batch so :func:`collect_batches` can run
    from a fresh kernel.
    """
    cells = build_requests(judge, combined_data, questionnaire_names, model_layout,
                           rep=rep, subset_n=subset_n)
    if not cells:
        print(f"[judge_batch] {judge.tag} rep={rep}: nothing to do — all cells already on disk.")
        return []
    n_batches = (len(cells) + max_per_batch - 1) // max_per_batch
    print(f"[judge_batch] {judge.tag} rep={rep}: {len(cells):,} requests -> {n_batches} batch(es) "
          f"of <= {max_per_batch:,}")
    if dry_run:
        print("[judge_batch] DRY RUN — nothing submitted. Pass dry_run=False to spend.")
        return []

    client = _judge.init_judge_client(judge)
    # The async client is fine for batch submission, but the calls are one-shot and sequential;
    # use the sync client to keep this callable from a plain notebook cell.
    import anthropic
    key = (os.environ.get("ANTHROPIC_API_KEY") or _judge._read_key_file("anthropic_key.txt"))
    sync_client = anthropic.Anthropic(api_key=key)

    batch_ids = []
    for b in range(n_batches):
        chunk = cells[b * max_per_batch:(b + 1) * max_per_batch]
        requests = [{"custom_id": c.custom_id, "params": c.params} for c in chunk]
        batch = sync_client.messages.batches.create(requests=requests)
        state = {
            "batch_id": batch.id, "judge_tag": judge.tag, "judge_model": judge.model, "rep": rep,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_requests": len(chunk),
            "collected": False,
            "manifest": [{"custom_id": c.custom_id, "out_path": os.path.relpath(
                              c.out_path, _judge.JUDGE_CHECK_ROOT),
                          "metric": c.metric, "model": c.model, "oracle": c.oracle,
                          "file_index": c.file_index, "n_therapist_utt": c.n_therapist_utt}
                         for c in chunk],
        }
        fp = os.path.join(_state_dir(judge.tag, rep), f"{batch.id}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        batch_ids.append(batch.id)
        print(f"  submitted {batch.id}  ({len(chunk):,} requests)  -> {fp}")
    return batch_ids


# ── phase 2: poll ─────────────────────────────────────────────────────────────

def _load_states(judge_tag: str, rep: Optional[int] = None) -> List[dict]:
    root = os.path.join(BATCH_STATE_ROOT, judge_tag)
    if not os.path.isdir(root):
        return []
    out = []
    for rep_dir in sorted(os.listdir(root)):
        if not rep_dir.startswith("rep="):
            continue
        r = int(rep_dir.split("=")[1])
        if rep is not None and r != rep:
            continue
        d = os.path.join(root, rep_dir)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    st = json.load(f)
                st["_state_path"] = os.path.join(d, fn)
                out.append(st)
    return out


def poll_batches(judge: "_judge.JudgeSpec", *, rep: Optional[int] = None) -> pd.DataFrame:
    """Status of every submitted batch for this judge. Free (metadata reads only)."""
    states = _load_states(judge.tag, rep)
    if not states:
        return pd.DataFrame()
    import anthropic
    key = (os.environ.get("ANTHROPIC_API_KEY") or _judge._read_key_file("anthropic_key.txt"))
    client = anthropic.Anthropic(api_key=key)
    rows = []
    for st in states:
        try:
            b = client.messages.batches.retrieve(st["batch_id"])
            rc = b.request_counts
            rows.append({"batch_id": st["batch_id"], "rep": st["rep"],
                         "status": b.processing_status, "n_requests": st["n_requests"],
                         "succeeded": rc.succeeded, "errored": rc.errored,
                         "processing": rc.processing, "canceled": rc.canceled,
                         "expired": rc.expired, "collected": st.get("collected", False)})
        except Exception as e:
            rows.append({"batch_id": st["batch_id"], "rep": st["rep"], "status": f"ERROR: {e}",
                         "n_requests": st["n_requests"], "collected": st.get("collected", False)})
    return pd.DataFrame(rows)


def wait_for_batches(judge: "_judge.JudgeSpec", *, rep: Optional[int] = None,
                     poll_seconds: int = 300, max_hours: float = 26.0) -> pd.DataFrame:
    """Block until every batch has ended (or ``max_hours`` elapses). Optional convenience — the
    three-phase design means you can just as well close the notebook and call
    :func:`collect_batches` later."""
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        tab = poll_batches(judge, rep=rep)
        if tab.empty or (tab["status"] == "ended").all():
            return tab
        pending = int((tab["status"] != "ended").sum())
        print(f"[judge_batch] {pending} batch(es) still running; sleeping {poll_seconds}s")
        time.sleep(poll_seconds)
    return poll_batches(judge, rep=rep)


# ── phase 3: collect ──────────────────────────────────────────────────────────

def collect_batches(judge: "_judge.JudgeSpec", *, rep: Optional[int] = None,
                    overwrite: bool = False) -> dict:
    """Retrieve ended batches and write per-conversation CSVs in the standard by-judge layout.

    Parsing goes through the SAME ``parse_json_response`` + ``_build_row`` path as the live runner,
    so a batch-scored CSV is byte-identical in shape to a live-scored one and every downstream
    loader is unchanged. Results whose JSON fails validation are counted as errors and left absent
    rather than written as partial rows — a missing cell is visible to the coverage planner and
    gets retried on the next submit; a malformed one would silently poison an arm mean.
    """
    states = _load_states(judge.tag, rep)
    if not states:
        print(f"[judge_batch] no submitted batches found for {judge.tag}")
        return {}
    import anthropic
    key = (os.environ.get("ANTHROPIC_API_KEY") or _judge._read_key_file("anthropic_key.txt"))
    client = anthropic.Anthropic(api_key=key)
    qid_of = _name_to_qid()
    totals = {"written": 0, "skipped_existing": 0, "errors": 0, "batches": 0}
    err_sample: Dict[str, str] = {}    # first message seen per error kind, across ALL batches

    for st in states:
        if st.get("collected") and not overwrite:
            continue
        try:
            b = client.messages.batches.retrieve(st["batch_id"])
        except Exception as e:
            print(f"  [{st['batch_id']}] retrieve failed: {e}")
            continue
        if b.processing_status != "ended":
            print(f"  [{st['batch_id']}] not ended yet ({b.processing_status}) — skipping")
            continue

        by_id = {m["custom_id"]: m for m in st["manifest"]}
        n_w = n_e = n_s = 0
        # API-level failures are TALLIED BY TYPE, not just counted. A bare count ("12,090 errors")
        # says nothing about whether the cause is transient (rate limit -> just resubmit),
        # structural (invalid_request -> fix the payload), or external (an exhausted credit
        # balance, which looks like a code bug until you read the message).
        err_kinds: Dict[str, int] = {}
        for result in client.messages.batches.results(st["batch_id"]):
            meta = by_id.get(result.custom_id)
            if meta is None:
                continue
            # REBUILD the destination from the manifest's COMPONENTS, never from its stored
            # relative path. A manifest can outlive a layout change (submit today, collect after a
            # tree migration), and a stale relative path then writes a valid-looking CSV into a
            # sibling directory the loaders don't read — silent, and invisible until a coverage
            # count comes back short. Components + `judge_out_dir` always resolve to the CURRENT
            # layout. `out_path` is retained in the manifest for forensics only.
            out_fp = os.path.join(
                _judge.judge_out_dir(judge.tag, st["rep"],
                                     _registry.EVAL_QUESTIONNAIRE_DIRS[meta["metric"]],
                                     meta["oracle"], meta["model"]),
                f"{meta['file_index']}.csv")
            if os.path.exists(out_fp) and not overwrite:
                n_s += 1
                continue
            if result.result.type != "succeeded":
                n_e += 1
                err = getattr(result.result, "error", None)
                inner = getattr(err, "error", None)
                kind = (getattr(inner, "type", None) or getattr(err, "type", None)
                        or str(result.result.type))
                key = f"{result.result.type}/{kind}"
                err_kinds[key] = err_kinds.get(key, 0) + 1
                if key not in err_sample:
                    err_sample[key] = str(getattr(inner, "message", None) or err)[:200]
                continue
            msg = result.result.message
            try:
                if msg.stop_reason == "refusal":
                    raise ValueError("stop_reason=refusal")
                text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                payload = json.loads(text)
                qid = qid_of[meta["metric"]]
                ed = _pipeline.get_prompt_eval_questionnaire(questionnaire=qid, conversation="")
                parsed = _pipeline.parse_json_response(response_content=payload,
                                                       questionnaire_id=qid, labels=ed["labels"])
                # MICI's rate needs the therapist-turn denominator; it was captured at submit time
                # so collection never has to re-read conversation data.
                if meta["metric"] == "MICI":
                    rdf = pd.DataFrame([_pipeline._build_mici_row(parsed["scores_dict"],
                                                                 meta["n_therapist_utt"])])
                else:
                    rdf = _pipeline._build_row(qid, parsed["scores_dict"], "")
                if rdf is None or rdf.isnull().values.any():
                    raise ValueError("null values in parsed row")
            except Exception as e:
                n_e += 1
                if n_e <= 3:
                    print(f"    parse error ({meta['metric']}/{meta['model']}/"
                          f"{meta['file_index']}): {e}")
                continue
            os.makedirs(os.path.dirname(out_fp), exist_ok=True)
            rdf.to_csv(out_fp, index=False)
            n_w += 1

        st["collected"] = True
        st["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        st["n_written"], st["n_errors"] = n_w, n_e
        st["error_kinds"] = err_kinds
        sp = st.pop("_state_path")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        totals["written"] += n_w
        totals["skipped_existing"] += n_s
        totals["errors"] += n_e
        totals["batches"] += 1
        for k, v in err_kinds.items():
            totals.setdefault("error_kinds", {})
            totals["error_kinds"][k] = totals["error_kinds"].get(k, 0) + v
        print(f"  [{st['batch_id']}] wrote {n_w:,}, skipped {n_s:,}, errors {n_e:,}"
              + (f"  [{', '.join(f'{k} x{v}' for k, v in err_kinds.items())}]" if err_kinds else ""))
    print(f"[judge_batch] {judge.tag}: {totals['written']:,} CSVs written across "
          f"{totals['batches']} batch(es); {totals['errors']:,} errors")
    for k, v in (totals.get("error_kinds") or {}).items():
        print(f"    {v:,} x {k}: {err_sample.get(k, '')}")
    return totals


# ── metered usage probe (for the cost model) ──────────────────────────────────

async def probe_usage(judge: "_judge.JudgeSpec", combined_data: pd.DataFrame,
                      questionnaire_names: Sequence[str], *, n_per_metric: int = 2
                      ) -> "_plan.UsageProfile":
    """**PAID but trivially cheap** (``n_per_metric x len(metrics)`` calls, a few cents).

    Runs real scoring calls and reads the API's OWN ``usage`` numbers, then averages them into a
    :class:`judge_plan.UsageProfile`. This replaces guessing at tokenizer behaviour, output length,
    and cache hits with three measured facts — the difference between a cost plan you can defend
    and one you hope is right.

    **Sampling is stratified at quantile MIDPOINTS**, i.e. the ``(i + 0.5) / n`` quantiles of the
    transcript-length distribution. Input cost is transcript-dominated and conversation length is
    right-skewed, so the obvious alternatives both bias the estimate badly: taking the first N rows
    samples whatever order the loader produced, and spreading endpoint-to-endpoint (``i / (n-1)``)
    puts the shortest AND longest conversation in the sample — at ``n=2`` that is literally
    ``(min + max) / 2``, which on this data overestimates the mean by ~2x. Midpoint quantiles give
    each stratum equal weight and converge to the distribution mean.
    """
    client = _judge.init_judge_client(judge)
    qid_of = _name_to_qid()
    convs = combined_data["conversation"].tolist()
    texts = [reconstruct_conversation_text(c) for c in convs]
    texts.sort(key=len)
    n_pick = max(int(n_per_metric), 1)
    picks = [texts[min(int((i + 0.5) / n_pick * len(texts)), len(texts) - 1)]
             for i in range(n_pick)]

    tot_in = tot_out = tot_cached = 0.0
    n = 0
    for qname in questionnaire_names:
        for conv_str in picks:
            ed = _pipeline.get_prompt_eval_questionnaire(questionnaire=qid_of[qname],
                                                         conversation=conv_str)
            extra = {"thinking": judge.thinking} if judge.thinking is not None else {}
            resp = await client.messages.create(
                model=judge.model, max_tokens=judge.max_tokens,
                output_config={"format": {"type": "json_schema",
                                          "schema": _judge._strip_unsupported_constraints(ed["schema"])}},
                messages=[{"role": "user", "content": ed["prompt"]}], **extra)
            u = resp.usage
            tot_in += u.input_tokens
            tot_out += u.output_tokens
            tot_cached += getattr(u, "cache_read_input_tokens", 0) or 0
            n += 1
    return _plan.UsageProfile(input_tokens=tot_in / n, output_tokens=tot_out / n,
                              cached_input_tokens=tot_cached / n,
                              source=f"measured ({judge.model})", n_samples=n)


def probe_usage_sync(judge, combined_data, questionnaire_names, *, n_per_metric: int = 2):
    """Notebook-friendly wrapper around :func:`probe_usage`."""
    return asyncio.get_event_loop().run_until_complete(
        probe_usage(judge, combined_data, questionnaire_names, n_per_metric=n_per_metric))
