"""judge_plan.py — PRE-FLIGHT for a multi-judge sweep: rubric parity, coverage planning, and
cost estimation. **Nothing here calls a paid endpoint**, so it is free to re-run.

The full dual-judge eval is the largest single API purchase left in the thesis (22,272 calls per
judge per rep). Three things have to be true before that money is spent, and each has a function
here:

1. **Parity** (:func:`check_rubric_parity`) — the second judge must answer the SAME rubric. Claude's
   ``json_schema`` rejects ``minimum``/``maximum``/``minItems``/``maxItems``, so
   ``judge._strip_unsupported_constraints`` folds them into ``description``. That keeps the
   *instruction* but drops the *enforcement*: an array-shaped rubric (Q1/Q2/WAI/CSQ8/MI-SAT) whose
   ``scores`` comes back the wrong length fails ``parse_json_response``, the exception is swallowed,
   and the conversation is silently dropped — biased missingness on the headline metric. This
   function is the gate: it verifies every dropped constraint was restated in prose and that the
   two encodings are otherwise structurally identical.
2. **Coverage** (:func:`plan_sweep`) — what actually needs scoring, after skipping what is already
   on disk. Resume-safe planning, so a re-run costs nothing for work already done.
3. **Cost** (:func:`estimate_cost`) — from MEASURED token usage where possible (see
   :func:`probe_usage` in ``judge_batch``), from transcript character counts otherwise. Never from
   a guess: the sweep is priced per call, and per-call cost is dominated by the transcript, which
   varies ~4x across conversations.

Prompt-caching thresholds matter here and differ by judge (see :data:`JUDGE_PRICING`). The oracle
prompt's fixed prefix is ~1.1k tokens — rubric-first by design so OpenAI's 1,024-token minimum is
cleared (see the caching gotcha in ``Exp3_PTO_GRPO/CLAUDE.md``). Claude Haiku 4.5's minimum is
4,096, so **the same prompt does not cache on Haiku**: the second judge pays full input price on
every call while the primary gets ~50% off. :func:`prefix_report` measures the real prefix so this
is a checked fact rather than an assumption.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..constants import WORKSPACE_ROOT
from . import judge as _judge
from . import registry as _registry
from . import pipeline as _pipeline

# ── pricing ───────────────────────────────────────────────────────────────────
# USD per 1M tokens. `cache_read` is the multiplier applied to input tokens served from the prompt
# cache; `min_cache_tokens` is the shortest prefix that will cache at all (shorter prefixes cache
# SILENTLY NOT AT ALL — no error, `cache_creation_input_tokens` just stays 0).
#
# Anthropic rows: Claude API pricing as documented 2026-06-24. OpenAI row: gpt-4o-mini standard
# pricing with the automatic 50% cached-input discount. VERIFY BOTH against your billing dashboard
# before quoting a number in the thesis — vendor pricing changes without touching this file.


@dataclass(frozen=True)
class Pricing:
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_mult: float
    min_cache_tokens: int
    batch_discount: float = 0.5   # both vendors: 50% off for async batch


JUDGE_PRICING: Dict[str, Pricing] = {
    "gpt-4o-mini-2024-07-18": Pricing(0.15, 0.60, 0.50, 1024),
    "claude-haiku-4-5":       Pricing(1.00, 5.00, 0.10, 4096),
    "claude-sonnet-5":        Pricing(2.00, 10.00, 0.10, 1024),   # intro pricing thru 2026-08-31
    "claude-opus-4-8":        Pricing(5.00, 25.00, 0.10, 1024),
    "claude-opus-5":          Pricing(5.00, 25.00, 0.10, 512),
}


def pricing_for(model: str) -> Optional[Pricing]:
    return JUDGE_PRICING.get(model)


# ── 1. rubric parity ──────────────────────────────────────────────────────────

_DROPPED_KEYS = ("minimum", "maximum", "minItems", "maxItems", "multipleOf")

# A short transcript is enough to build a schema — the schema is transcript-independent.
_DUMMY_CONV = "[THERAPIST] Hello.\n[PATIENT] Hi.\n[THERAPIST] How are you?\n[PATIENT] Fine.\n"


def _walk(node, path="$"):
    """Yield (path, dict-node) for every dict in a JSON-schema tree."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _shape_signature(schema: dict) -> str:
    """Structure-only fingerprint: property names, types, required sets, nesting — everything
    EXCEPT the constraint keys we knowingly move into prose. Two encodings of the same rubric must
    produce an identical signature, or the judges are not answering the same question."""
    def norm(node):
        if isinstance(node, dict):
            return {k: norm(v) for k, v in sorted(node.items())
                    if k not in _DROPPED_KEYS and k != "description"}
        if isinstance(node, list):
            return [norm(v) for v in node]
        return node
    return json.dumps(norm(schema), sort_keys=True)


def check_rubric_parity(questionnaire_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """THE GATE before a second-judge sweep. Free — builds prompts locally, calls nothing.

    For every questionnaire, compares the schema the primary oracle receives against the one a
    Claude judge receives (post ``_strip_unsupported_constraints``) and checks:

    - ``n_items`` matches the rubric's own ``questions_count`` / label count;
    - every array whose length was pinned by ``minItems == maxItems`` has that exact count RESTATED
      in its ``description`` (otherwise the one-score-per-item guarantee is gone and short arrays
      become silent missingness);
    - every numeric field whose ``minimum``/``maximum`` was dropped has the range restated;
    - no constraint key Claude rejects survives anywhere (a leftover would 400 the whole rubric);
    - the structural signature is byte-identical between encodings.

    Returns one row per questionnaire with ``parity_ok``. **Do not launch a sweep with any row
    False** — that rubric would be measuring something different for the two judges.
    """
    names = list(questionnaire_names or _judge.JUDGE_METRIC_COLS.keys())
    from questionnaires import QuestionnaireID  # canonical enum (code/ on sys.path)
    name_to_qid = {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2,
                   "WAI-SR": QuestionnaireID.WAI_SR, "CSQ-8": QuestionnaireID.CSQ8,
                   "MI-SAT": QuestionnaireID.MI_SAT, "MITI": QuestionnaireID.MITI,
                   "PCT": QuestionnaireID.PCT, "MICI": QuestionnaireID.MICI}

    rows = []
    for name in names:
        qid = name_to_qid[name]
        ed = _pipeline.get_prompt_eval_questionnaire(questionnaire=qid, conversation=_DUMMY_CONV)
        orig, claude = ed["schema"], _judge._strip_unsupported_constraints(ed["schema"])
        n_labels = len(ed.get("labels") or [])

        problems: List[str] = []

        # (a) every pinned-length array must restate its count in prose
        pinned = [(p, n.get("minItems")) for p, n in _walk(orig)
                  if n.get("type") == "array" and n.get("minItems") is not None
                  and n.get("minItems") == n.get("maxItems")]
        claude_nodes = dict(_walk(claude))
        for path, n_items in pinned:
            desc = (claude_nodes.get(path, {}) or {}).get("description", "")
            if str(int(n_items)) not in desc:
                problems.append(f"array {path}: length {n_items} not restated in description")

        # (b) every dropped numeric bound must restate its range in prose
        bounded = [(p, n.get("minimum"), n.get("maximum")) for p, n in _walk(orig)
                   if n.get("minimum") is not None or n.get("maximum") is not None]
        for path, lo, hi in bounded:
            desc = (claude_nodes.get(path, {}) or {}).get("description", "")
            missing = [str(v) for v in (lo, hi) if v is not None and str(v) not in desc]
            if missing:
                problems.append(f"numeric {path}: bound(s) {'/'.join(missing)} not restated")

        # (c) nothing Claude rejects may survive
        leftovers = [f"{p}.{k}" for p, n in _walk(claude) for k in _DROPPED_KEYS if k in n]
        if leftovers:
            problems.append(f"unsupported keys survive: {', '.join(leftovers[:4])}")

        # (d) structure must be identical
        same_shape = _shape_signature(orig) == _shape_signature(claude)
        if not same_shape:
            problems.append("structural signature differs between encodings")

        rows.append({
            "metric": name,
            "n_items": int(ed.get("questions_count") or n_labels or 0),
            "n_labels": n_labels,
            "arrays_pinned": len(pinned),
            "numeric_bounded": len(bounded),
            "shape_identical": same_shape,
            "parity_ok": not problems,
            "problems": "; ".join(problems),
        })
    return pd.DataFrame(rows)


# ── 2. prompt prefix / caching ────────────────────────────────────────────────

def prefix_report(questionnaire_names: Optional[Sequence[str]] = None,
                  chars_per_token: float = 3.9) -> pd.DataFrame:
    """Measure the transcript-INDEPENDENT prompt prefix per rubric — the part prompt caching can
    reuse — and flag which judges will actually cache it. Free (no API, no tokenizer).

    Method: build each prompt twice with different transcripts and take the common leading
    substring. That is exactly the span a prefix-match cache can serve, measured rather than
    assumed. ``chars_per_token`` converts to an approximate token count; it is deliberately a
    single documented constant rather than a real tokenizer, because the only decision it feeds is
    a threshold comparison with wide margins (1,024 / 4,096).

    A rubric whose prefix falls below a judge's ``min_cache_tokens`` caches SILENTLY NOT AT ALL.
    """
    names = list(questionnaire_names or _judge.JUDGE_METRIC_COLS.keys())
    from questionnaires import QuestionnaireID
    name_to_qid = {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2,
                   "WAI-SR": QuestionnaireID.WAI_SR, "CSQ-8": QuestionnaireID.CSQ8,
                   "MI-SAT": QuestionnaireID.MI_SAT, "MITI": QuestionnaireID.MITI,
                   "PCT": QuestionnaireID.PCT, "MICI": QuestionnaireID.MICI}
    conv_a = _DUMMY_CONV
    conv_b = "[THERAPIST] Good morning, what brings you in today?\n[PATIENT] My doctor sent me.\n"

    rows = []
    for name in names:
        qid = name_to_qid[name]
        pa = _pipeline.get_prompt_eval_questionnaire(questionnaire=qid, conversation=conv_a)["prompt"]
        pb = _pipeline.get_prompt_eval_questionnaire(questionnaire=qid, conversation=conv_b)["prompt"]
        n = 0
        for ca, cb in zip(pa, pb):
            if ca != cb:
                break
            n += 1
        approx_tok = n / chars_per_token
        # The CACHEABLE prefix (above) and the FIXED prompt (below) are different quantities and
        # must not be confused: for MITI/PCT/MICI a per-conversation utterance count sits early in
        # the instructions, so the cacheable prefix collapses to 138-206 tokens while the fixed
        # instruction+rubric text is still ~1,000. Costing off the cacheable prefix underestimates
        # input by ~25% overall. Use `fixed_prompt_tokens_approx` for cost, `prefix_*` for caching.
        fixed_chars = len(pa) - len(conv_a)
        row = {"metric": name, "prefix_chars": n, "prefix_tokens_approx": round(approx_tok),
               "fixed_prompt_chars": fixed_chars,
               "fixed_prompt_tokens_approx": round(fixed_chars / chars_per_token),
               "prompt_chars_total": len(pa)}
        for model, pr in JUDGE_PRICING.items():
            row[f"caches_on_{model}"] = approx_tok >= pr.min_cache_tokens
        rows.append(row)
    return pd.DataFrame(rows)


# ── 3. coverage planning ──────────────────────────────────────────────────────

@dataclass
class SweepPlan:
    """What a sweep would actually score, after subtracting what is already on disk."""
    judge_tag: str
    rep: int
    cells: pd.DataFrame                     # model x metric x n_todo
    n_calls: int
    n_existing: int
    models: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"<SweepPlan {self.judge_tag} rep={self.rep}: {self.n_calls:,} calls to make, "
                f"{self.n_existing:,} already on disk, "
                f"{len(self.models)} models x {len(self.metrics)} metrics>")


def plan_sweep(judge: "_judge.JudgeSpec", combined_data: pd.DataFrame,
               questionnaire_names: Sequence[str],
               model_layout: Dict[str, Dict[str, str]], *, rep: int = 0,
               subset_n: Optional[int] = None) -> SweepPlan:
    """Enumerate the (model, metric, conversation) cells a sweep would score, skipping existing
    CSVs exactly the way the runners do. Costs nothing and is the input to :func:`estimate_cost`.
    """
    rows, n_todo, n_have = [], 0, 0
    models = [str(m) for m in combined_data["Model"].unique()]
    for model in models:
        entry = model_layout.get(model)
        oracle = entry["oracle"] if entry else "none"
        sub = combined_data[combined_data["Model"].astype(str) == model]
        if subset_n is not None:
            sub = sub.iloc[:subset_n]
        ids = [int(i) for i in sub["id"]]
        for qname in questionnaire_names:
            out_dir = _judge.judge_out_dir(judge.tag, rep,
                                           _registry.EVAL_QUESTIONNAIRE_DIRS[qname], oracle, model)
            have = 0
            if os.path.isdir(out_dir):
                present = {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".csv")}
                have = sum(1 for i in ids if str(i) in present)
            todo = len(ids) - have
            n_todo += todo
            n_have += have
            rows.append({"model": model, "metric": qname, "oracle": oracle,
                         "n_total": len(ids), "n_existing": have, "n_todo": todo})
    return SweepPlan(judge_tag=judge.tag, rep=rep, cells=pd.DataFrame(rows), n_calls=n_todo,
                     n_existing=n_have, models=models, metrics=list(questionnaire_names))


# ── 4. cost ───────────────────────────────────────────────────────────────────

@dataclass
class UsageProfile:
    """Per-call token usage for one judge on this workload. Prefer MEASURED (from
    ``judge_batch.probe_usage``) over estimated — per-call cost is dominated by the transcript,
    which varies ~4x across conversations, so a single-point guess is not good enough to plan a
    $50+ purchase."""
    input_tokens: float
    output_tokens: float
    cached_input_tokens: float = 0.0    # subset of input_tokens served from cache
    source: str = "estimated"
    n_samples: int = 0


def usage_from_chars(combined_data: pd.DataFrame, questionnaire_names: Sequence[str],
                     *, model_name: str, chars_per_token: float = 3.9) -> UsageProfile:
    """Free, approximate usage profile from transcript lengths + measured prompt prefixes.

    Cross-check for the measured profile, and the only option before any key exists. Output tokens
    are estimated from the rubric's item count (~4 tokens per integer score + JSON overhead), which
    is an upper-ish bound: the schema is closed, so the model cannot ramble.
    """
    from .conversations import reconstruct_conversation_text
    pref = prefix_report(questionnaire_names, chars_per_token=chars_per_token).set_index("metric")
    par = check_rubric_parity(questionnaire_names).set_index("metric")

    conv_chars = []
    for conv in combined_data["conversation"].head(200):
        try:
            conv_chars.append(len(reconstruct_conversation_text(conv)))
        except Exception:
            continue
    mean_conv_tok = (sum(conv_chars) / max(len(conv_chars), 1)) / chars_per_token

    pr = pricing_for(model_name)
    in_tok, out_tok, cached = [], [], []
    for name in questionnaire_names:
        # Cost is driven by the WHOLE fixed prompt; only the cacheable PREFIX earns the discount.
        fixed_tok = float(pref.loc[name, "fixed_prompt_tokens_approx"])
        prefix_tok = float(pref.loc[name, "prefix_tokens_approx"])
        in_tok.append(fixed_tok + mean_conv_tok)
        n_items = float(par.loc[name, "n_items"]) or 8.0
        out_tok.append(n_items * 4.0 + 20.0)
        cacheable = pr is not None and prefix_tok >= pr.min_cache_tokens
        cached.append(prefix_tok if cacheable else 0.0)
    n = len(questionnaire_names)
    return UsageProfile(input_tokens=sum(in_tok) / n, output_tokens=sum(out_tok) / n,
                        cached_input_tokens=sum(cached) / n, source="estimated-from-chars",
                        n_samples=len(conv_chars))


def estimate_cost(plan: SweepPlan, usage: UsageProfile, *, model_name: str,
                  batch: bool = True) -> dict:
    """Project the USD cost of a :class:`SweepPlan` under a :class:`UsageProfile`.

    ``batch=True`` applies the vendor async-batch discount (50% on both). For an offline scoring
    job with no latency requirement this is free money and should be the default.
    """
    pr = pricing_for(model_name)
    if pr is None:
        raise ValueError(f"No pricing row for {model_name!r} — add one to JUDGE_PRICING.")
    uncached_in = max(usage.input_tokens - usage.cached_input_tokens, 0.0)
    per_call = (
        uncached_in / 1e6 * pr.input_usd_per_mtok
        + usage.cached_input_tokens / 1e6 * pr.input_usd_per_mtok * pr.cache_read_mult
        + usage.output_tokens / 1e6 * pr.output_usd_per_mtok
    )
    if batch:
        per_call *= (1.0 - pr.batch_discount)
    return {
        "model": model_name, "judge_tag": plan.judge_tag, "rep": plan.rep, "batch": batch,
        "n_calls": plan.n_calls, "usd_per_call": round(per_call, 6),
        "usd_total": round(per_call * plan.n_calls, 2),
        "input_tokens_per_call": round(usage.input_tokens),
        "cached_input_tokens_per_call": round(usage.cached_input_tokens),
        "output_tokens_per_call": round(usage.output_tokens),
        "usage_source": usage.source, "usage_n_samples": usage.n_samples,
        "caches": usage.cached_input_tokens > 0,
    }


def calibrate_from_receipt(observed_usd: float, observed_calls: int, plan: SweepPlan,
                           *, batch: bool = True, batch_discount: float = 0.5) -> dict:
    """Independent, vendor-agnostic cost estimate anchored on a bill you actually paid.

    Token arithmetic has many places to be wrong (tokenizer, cache hits, retries, output length);
    a receipt has none. Use this as the primary number and :func:`estimate_cost` as the cross-check
    — if the two disagree by more than ~2x, something in the token model is wrong and the sweep
    should not launch until it is understood.
    """
    per_call = observed_usd / max(observed_calls, 1)
    if batch:
        per_call *= (1.0 - batch_discount)
    return {"basis": f"receipt ${observed_usd:.2f} over {observed_calls:,} calls",
            "usd_per_call": round(per_call, 6), "n_calls": plan.n_calls, "batch": batch,
            "usd_total": round(per_call * plan.n_calls, 2)}


def sweep_report(plan: SweepPlan, usage: UsageProfile, *, model_name: str,
                 receipt: Optional[Tuple[float, int]] = None) -> pd.DataFrame:
    """One table with both cost estimates, batched and not — the thing to read before spending."""
    rows = []
    for batch in (True, False):
        e = estimate_cost(plan, usage, model_name=model_name, batch=batch)
        rows.append({"basis": f"tokens ({usage.source})", "batch": batch,
                     "usd_per_call": e["usd_per_call"], "usd_total": e["usd_total"],
                     "n_calls": e["n_calls"]})
        if receipt is not None:
            c = calibrate_from_receipt(receipt[0], receipt[1], plan, batch=batch)
            rows.append({"basis": c["basis"], "batch": batch, "usd_per_call": c["usd_per_call"],
                         "usd_total": c["usd_total"], "n_calls": c["n_calls"]})
    return pd.DataFrame(rows)
