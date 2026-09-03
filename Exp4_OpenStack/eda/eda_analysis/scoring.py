"""scoring.py -- the EVAL side: grade a model state's conversations and write the score lake.

This is the only module in ``eda_analysis`` that TALKS TO A MODEL. Everything else reads finished
artifacts; this one produces them. It is deliberately absent from the package's lazy attribute map
(see ``eda_analysis/__init__.py``), so ``import eda_analysis`` can never reach it by accident --
``Run_Eval.ipynb`` imports it explicitly, and that import IS the moment someone chooses to spend.

On the default Exp4 stack there is nothing to spend: the judge is the same local Gemma behind vLLM
that plays the oracle and the patient, so a full grid costs **$0 in API**. That is the headline
property of the experiment, and :func:`estimate_calls` is written to state it plainly rather than
to imply a cost that does not exist.

What one run produces
---------------------
One parquet per (judge, rep, metric, arm, model state), 96 rows -- one per persona::

    data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/<EXPERIMENT_NAME>/model_iter_<N>.parquet

Exp3 wrote ~50k single-row CSVs into that same conceptual space and needed a parquet fold cache
plus a content-signature manifest to read them back. One file per model state needs none of that,
and it is also what makes resume trivial: **a state is done iff its file exists and has a row for
every conversation on disk.** :func:`discover_scorable` applies exactly that test, which is what
makes a second ``Run_Eval`` run cost zero calls, and it is why writes go through a temp file and
``os.replace`` -- an interrupted write must never leave a half-file that the next run counts as
finished.

Three invariants
----------------
1. **Never write into another judge's partition.** The partition tag is derived INSIDE
   :func:`score_model_state` from the binding that is about to do the grading
   (``constants.judge_tag(binding)``), never from a caller-supplied string, so the folder cannot
   disagree with the grader that filled it. :func:`run_scoring` additionally refuses a plan whose
   ``judge`` column does not match the binding it was handed. A tag is a DIRECTORY NAME: changing
   how it is built orphans every parquet already written, and the loader then reports an unscored
   arm while the files sit on disk.
2. **``persona_id`` is in every row.** It is the only valid pairing key. Exp4 names conversation
   files by the stable persona id, so unlike Exp3 there is no shuffle to replay -- but a directory
   listing is still not the persona order once one conversation is missing. Join on
   ``persona_id``; never on row order.
3. **The judge path and the TRAINING path are the same code.** Grading goes through
   ``core.oracle.get_evaluation_json`` -- the same prompt builder, the same schema shim, the same
   validation ladder, the same retry policy the trainer's reward uses. If they diverged,
   ``code/tools/oracle_sanity.py`` (the gate that proves an open grader is not degenerate) would
   be testing something other than what trains, which is the one failure it cannot catch.

Why the per-rubric call rather than ``score_conversation``
---------------------------------------------------------
``core.oracle.score_conversation`` reduces a conversation to ONE number across a rubric SET and
short-circuits on the first rubric that fails. That is exactly right for a training reward and
exactly wrong here: the lake partitions per metric and each parquet stores the per-ITEM scores, so
this module calls :func:`~core.oracle.get_evaluation_json` once per (conversation, rubric) and
keeps the whole response. Both functions sit on the same call and the same validation ladder, so
"the same code" is preserved where it matters.

Usage (``Run_Eval.ipynb``)::

    from eda_analysis import scoring
    from core.concurrency import run_async

    binding = scoring.judge_binding_for(roles.DEFAULT_JUDGE_MODEL,      # google/gemma-4-E4B-it
                                        base_url="http://127.0.0.1:8000/v1")
    plan = scoring.discover_scorable(judge=scoring.judge_tag(binding))
    print(scoring.estimate_calls(plan, binding=binding))       # look BEFORE you spend
    done = run_async(scoring.run_scoring(plan, binding=binding))
    report = scoring.prompt_length_gate(scoring.gather_transcripts(),
                                        model=binding.model, base_url=binding.base_url)
    ok, messages = scoring.check_prompt_length_gate(report)    # the Phase 2 measurement
    # (max_model_len= is the OFFLINE fallback only: with a base_url the SERVED cap wins)

Where this runs: **on the GPU that serves the judge** -- Colab (A100 80 GB) or a GPU server over
SSH. The local 12 GB card cannot host the E4B judge at all (14.89 GiB of weights alone), so it is
for smoke tests only; ``Run_Eval.ipynb`` carries the same Drive-mount preamble as the trainer
notebooks for exactly that reason.

Import weight: pandas plus the stdlib-only canonical modules. No torch, no ``core.policy``, no
``core.lookahead``. The provider SDK is imported lazily by ``roles.make_client``, so this module
imports fine on a machine with no ``openai`` package as long as nothing is actually scored.
"""

from __future__ import annotations

import asyncio
import glob
import inspect
import math
import os
import time
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

# The package leaf FIRST: importing ``constants`` is what prepends ``Exp4_OpenStack/code`` to
# ``sys.path``, so every import below it resolves to the CANONICAL trainer-side module rather than
# to a copy. Do not reorder these blocks.
from .constants import (ALL_METRICS, DATA_DIR, DEFAULT_JUDGE_TAG, METRIC_ORDER, N_PERSONAS,
                        judge_tag)
from .constants import metric as metric_registry
from .data import Arm, discover_arms

from core.concurrency import AsyncPrimitives, run_async            # noqa: E402
from core.conversations import (CONV_FILE_PREFIX, ConversationState,  # noqa: E402
                                format_conversation_for_oracle, load_conversations_dir)
from core.oracle import (OPENAI_SHAPED_PROVIDERS, OracleConfig,    # noqa: E402
                         get_evaluation_json, make_oracle_client)
from naming import parse_experiment_name                           # noqa: E402
from questionnaires import (MICI_BEHAVIOR_LABELS, MICI_GLOBAL_LABELS,  # noqa: E402
                            MITI_BEHAVIOR_LABELS, MITI_GLOBAL_LABELS,
                            PCT_BEHAVIOR_LABELS, PCT_GLOBAL_LABELS,
                            WAI_SR_SUBSCALES, parse_json_response)
# Private on purpose: this is the SAME counter ``questionnaires`` interpolates into the MITI / PCT /
# MICI prompts ("therapist_utterance_count = N"), and MICI_Rate's denominator has to be the number
# the grader was shown. Counting turns off the ConversationState instead would drift from it the
# moment an empty utterance is dropped by the transcript formatter.
from questionnaires import _count_therapist_utterances             # noqa: E402
from roles import RoleBinding, make_binding                        # noqa: E402

__all__ = [
    # Contracts
    "PLAN_COLUMNS",
    "RESULT_COLUMNS",
    "STORED_METRICS",
    # Knobs
    "DEFAULT_CONCURRENCY",
    "JUDGE_MAX_TOKENS",
    "JUDGE_MAX_RETRIES",
    "JUDGE_REQUEST_TIMEOUT",
    "MIN_SUCCESS_RATIO",
    # Pricing (API judges only)
    "JudgePricing",
    "TokenProfile",
    "JUDGE_PRICING",
    "VENDOR_JUDGE_TAGS",
    "DEFAULT_TOKEN_PROFILE",
    # API
    "judge_binding_for",
    "discover_scorable",
    "score_model_state",
    "run_scoring",
    "estimate_calls",
    # The Phase 2 measurement (oracle prompt length vs the server's max_model_len)
    "PROMPT_LENGTH_REPORT_SIGNATURE",
    "TRANSCRIPT_COLUMNS",
    "gather_transcripts",
    "prompt_length_gate",
    "check_prompt_length_gate",
    # Re-export, so a notebook can drive the coroutines without a second import
    "judge_tag",
    "run_async",
]


# ==============================================================================
#  1. KNOBS
# ==============================================================================

#: In-flight grading calls per model state. One state is 96 conversations, so this is the real
#: bound on what the local server is asked to hold at once. It is deliberately lower than the
#: trainer's oracle concurrency: the trainer owns the GPU, while a scoring pass may be sharing it
#: with a live run.
DEFAULT_CONCURRENCY = 32

#: Response budget for the JSON, per call.
#:
#: Larger than ``core.oracle.OracleConfig``'s 256 because the eval side scores rubrics the trainer
#: never does. Q2 returns 17 integers under short keys; MITI returns 11 values under keys like
#: ``MITI1_CultivatingChangeTalk``, and PCT/MICI carry two nested objects. A clipped response is
#: not an error -- it is unparseable content, a retry, and then a HOLE, which is the biased
#: missingness this whole module is arranged to avoid. Raise it before suspecting the schema.
JUDGE_MAX_TOKENS = 512

#: Attempts per (conversation, rubric) call, including the first.
JUDGE_MAX_RETRIES = 3

#: Per-ATTEMPT ceiling in seconds, enforced by ``asyncio.wait_for`` inside ``get_evaluation_json``.
#: Generous compared with a patient turn: a judge call emits a whole rubric and is never on the
#: critical path of a lock-step simulation, so there is nothing for it to stall.
JUDGE_REQUEST_TIMEOUT = 180.0

#: Floor on the fraction of a model state's conversations that must grade successfully before the
#: parquet is written at all.
#:
#: Below it :func:`score_model_state` RAISES and writes nothing. That is the deliberate choice: a
#: written file is "done" forever as far as :func:`discover_scorable` is concerned, so a file that
#: is mostly holes would freeze a broken grader's output into the lake and every downstream table
#: would silently be computed over the conversations that happened to be easy to score.
MIN_SUCCESS_RATIO = 0.5

#: The eight stored instruments, in plot order. Composites (``Q1Q2``) are computed by the loader
#: from their components and are never scored or stored -- see :func:`_resolve_metrics`.
STORED_METRICS: Tuple[str, ...] = tuple(
    key for key in METRIC_ORDER if not ALL_METRICS[key].is_composite
)

_CONV_GLOB = f"{CONV_FILE_PREFIX}*.csv"


def _log(message: str) -> None:
    print(f"  [scoring] {message}")


# ==============================================================================
#  2. FRAME CONTRACTS
# ==============================================================================
#
# Both frames keep their columns (and dtypes) when empty. That is not politeness: ``Run_Eval`` is
# expected to be re-runnable at any time, and the SECOND run's plan is legitimately empty. A bare
# ``pd.DataFrame()`` would make the notebook die on the first ``plan["metric"]`` instead, which
# reads like a broken notebook rather than like "nothing left to do".

#: :func:`discover_scorable` -- one row per (arm, model state, metric) still MISSING from the lake.
#:
#: ``state_index`` is the MODEL STATE index -- the ``n`` of :func:`score_model_state`, where 0 is
#: the untrained base policy. It is NOT a training iteration (iteration ``n`` generates model state
#: ``n-1``); the plan never refers to training iterations at all.
PLAN_COLUMNS: Tuple[str, ...] = (
    "experiment_name", "arm_label", "arm", "method", "k", "mcl", "mode", "qtag",
    "judge", "rep", "metric", "questionnaire_id",
    "state_index", "model_state", "n_conversations", "n_rows_existing",
    "path", "conv_dir", "data_root",
)

#: :func:`run_scoring` -- the plan plus what happened to each row.
RESULT_COLUMNS: Tuple[str, ...] = PLAN_COLUMNS + (
    "status", "n_rows", "n_failed", "n_calls", "elapsed_s", "error",
)

_COLUMN_DTYPES: Dict[str, str] = {
    "experiment_name": "object", "arm_label": "object", "arm": "object", "method": "object",
    "mode": "object", "qtag": "object", "judge": "object", "metric": "object",
    "model_state": "object", "path": "object", "conv_dir": "object", "data_root": "object",
    "status": "object", "error": "object",
    "k": "int64", "mcl": "int64", "rep": "int64", "questionnaire_id": "int64",
    "state_index": "int64", "n_conversations": "int64", "n_rows_existing": "int64",
    "n_rows": "int64", "n_failed": "int64", "n_calls": "int64",
    "elapsed_s": "float64",
    # gather_transcripts
    "persona_id": "int64", "n_utterances": "int64", "transcript": "object",
}


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """A zero-row frame with the contract's columns AND their dtypes."""
    return pd.DataFrame(
        {c: pd.Series(dtype=_COLUMN_DTYPES.get(c, "object")) for c in columns}
    )


def _order_columns(df: pd.DataFrame, leading: Sequence[str]) -> pd.DataFrame:
    """Contract columns first, in order; every extra column after them."""
    lead = [c for c in leading if c in df.columns]
    rest = [c for c in df.columns if c not in set(lead)]
    return df[lead + rest]


# ==============================================================================
#  3. BINDINGS
# ==============================================================================


def judge_binding_for(model: str,
                      *,
                      provider: str = "openai_compat",
                      base_url: Optional[str] = None,
                      **kw) -> RoleBinding:
    """Build the :class:`roles.RoleBinding` for an eval judge.

    Args:
        model: Full model id as the provider spells it (``roles.DEFAULT_JUDGE_MODEL`` =
            ``google/gemma-4-E4B-it``, ``gpt-4o-mini``). Required and never defaulted: this id
            decides the ``judge=<tag>`` directory every score is written under, and an implicit
            grader is precisely the error the partition scheme exists to make impossible.
        provider: ``openai_compat`` (default -- any OpenAI-compatible server: vLLM, llama.cpp,
            TGI) or ``openai``. See the raise below for why ``anthropic`` is refused.
        base_url: Endpoint for ``openai_compat``. Normally the value
            ``tools.vllm_serve.serve_roles`` filled in; pass it explicitly when driving the EDA
            against a server someone else started.
        **kw: Any other :class:`roles.RoleBinding` field. ``request_timeout`` and ``max_retries``
            default to this module's judge-side values rather than to ``RoleBinding``'s
            patient-side ones.

    Returns:
        A frozen binding, with Gemma's thinking mode switched off for ``openai_compat``
        (``roles.make_binding(disable_thinking=True)``) -- a reasoning preamble in front of a
        schema-constrained response is a good way to lose the schema.

    Raises:
        ValueError: for ``anthropic``. Its Messages API rejects ``minimum``/``maximum``/
            ``minItems``/``maxItems``, so a Claude grader needs those constraints folded into
            ``description`` text (Exp3 carried a whole shim for it,
            ``scoring/judge.py::_strip_unsupported_constraints``). Exp4 has no such shim, and
            silently DROPPING the constraints is the worst option available: a wrong-length
            ``scores`` array then parses, averages to a plausible number, and the conversations the
            grader found hardest are the ones that go missing. Refusing is the honest failure.
        ValueError: for any provider ``core.oracle`` cannot speak (see
            :data:`~core.oracle.OPENAI_SHAPED_PROVIDERS`).

    Warning:
        An ``openai_compat`` binding with **no** ``base_url`` builds an OpenAI SDK client pointed
        at ``api.openai.com``. That does not fail -- it BILLS, on a run that exists to cost $0, and
        grades with a vendor model while the lake records the local tag. This function warns;
        :func:`score_model_state` refuses outright.
    """
    if provider not in OPENAI_SHAPED_PROVIDERS:
        raise ValueError(
            f"judge_binding_for: provider {provider!r} cannot grade in Exp4; expected one of "
            f"{OPENAI_SHAPED_PROVIDERS}. Anthropic in particular needs the constraint-stripping "
            f"shim Exp3 carried (its Messages API rejects minimum/maximum/minItems/maxItems), and "
            f"dropping those constraints instead is how a wrong-length scores array turns into "
            f"biased missingness that nothing reports."
        )
    kw.setdefault("request_timeout", JUDGE_REQUEST_TIMEOUT)
    kw.setdefault("max_retries", JUDGE_MAX_RETRIES)
    binding = make_binding(provider, model, base_url=base_url, **kw)
    if binding.is_local and not binding.base_url:
        _log(
            f"WARNING: {model!r} is bound as 'openai_compat' with no base_url. The OpenAI SDK "
            f"would default to api.openai.com -- billing a run that is supposed to cost $0 and "
            f"grading with the wrong model. Fill base_url in before scoring."
        )
    return binding


def _assert_scoreable(binding: RoleBinding) -> None:
    """Refuse a binding that would grade with a model other than the one it names.

    The ``base_url`` check is not pedantry -- see the warning on :func:`judge_binding_for`. It is
    the same guard ``code/tools/oracle_sanity.py`` applies before the sanity gate, and for the same
    reason: a passing report about the wrong grader is worse than no report.
    """
    if not isinstance(binding, RoleBinding):
        raise TypeError(
            f"binding must be a roles.RoleBinding, got {type(binding).__name__}. Build one with "
            f"scoring.judge_binding_for(...), or take it from tools.vllm_serve.serve_roles()."
        )
    if binding.provider not in OPENAI_SHAPED_PROVIDERS:
        raise ValueError(
            f"judge provider {binding.provider!r} is not supported; expected one of "
            f"{OPENAI_SHAPED_PROVIDERS}. See judge_binding_for for why anthropic is refused."
        )
    if binding.is_local and not binding.base_url:
        raise ValueError(
            "the judge binding is 'openai_compat' but carries no base_url, so the OpenAI SDK "
            "would send these calls to api.openai.com -- a vendor bill on a $0 experiment, and "
            "scores filed under the LOCAL model's judge tag. Pass base_url=, or run the binding "
            "through tools.vllm_serve.serve_roles first."
        )


# ==============================================================================
#  4. METRIC -> ROW
# ==============================================================================
#
# The parquet's schema per metric. Each row carries the per-item scores exactly as the rubric names
# them, plus the derived columns -- one of which is the metric's ``score_column``, THE number every
# downstream figure plots (``constants.METRICS``). The derived definitions are inherited verbatim
# from Exp3's row builders so the two experiments' tables have the same shape; the SCORES are of
# course not comparable across experiments (different grader, different axis).

#: Sum column emitted alongside the mean for the flat, equal-item rubrics.
_TOTAL_COLUMN: Dict[str, str] = {
    "Q1": "Q1_Total", "Q2": "Q2_Total", "CSQ8": "CSQ8_Total", "MI_SAT": "MI_Total",
}

#: ``metric -> (global rating labels, behaviour count labels)`` for the nested rubrics. Their
#: behaviour counts are UNBOUNDED above and are not on the 1-5 rating scale, so they are never
#: averaged into the globals mean -- that is why each of the three needs its own reduction.
_NESTED_SPLIT: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "MITI": (tuple(MITI_GLOBAL_LABELS), tuple(MITI_BEHAVIOR_LABELS)),
    "PCT": (tuple(PCT_GLOBAL_LABELS), tuple(PCT_BEHAVIOR_LABELS)),
    "MICI": (tuple(MICI_GLOBAL_LABELS), tuple(MICI_BEHAVIOR_LABELS)),
}


def _is_number(value: Any) -> bool:
    """True for a real, finite number that is not a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _mean(values: Iterable[Any]) -> float:
    """Mean over the numeric entries; ``nan`` when there are none."""
    nums = [float(v) for v in values if _is_number(v)]
    return float(fmean(nums)) if nums else math.nan


def _total(values: Iterable[Any]) -> float:
    """Sum over the numeric entries; ``nan`` when there are none."""
    nums = [float(v) for v in values if _is_number(v)]
    return float(sum(nums)) if nums else math.nan


def _ratio(numerator: Any, denominator: Any) -> float:
    """``numerator / denominator``, or ``nan`` unless both are numbers and the divisor is > 0."""
    if not (_is_number(numerator) and _is_number(denominator)) or float(denominator) <= 0:
        return math.nan
    return float(numerator) / float(denominator)


def _build_row(metric_key: str,
               scores: Dict[str, Any],
               *,
               n_therapist_utterances: int) -> Dict[str, float]:
    """One conversation's score columns for one metric: the items, then the derived numbers.

    Args:
        metric_key: A registered STORED metric (``constants.METRICS``).
        scores: ``{item label: value}`` as ``questionnaires.parse_json_response`` returns it. An
            **empty dict is the documented failure shape** -- every column then comes back ``nan``
            with the right names, which is how a failed conversation stays a visible row instead of
            becoming an invisible absence.
        n_therapist_utterances: MICI's rate denominator; ignored by every other metric.

    Returns:
        ``{column: value}``, item columns first (in rubric order), derived columns after.

    Raises:
        KeyError: if the metric's registered ``score_column`` was not produced. That means a metric
            was added to ``constants.METRICS`` without a branch here -- which would otherwise
            surface as ``data.load_scores_long`` skipping the partition with a "lacks <col>"
            warning, weeks later.
    """
    m = metric_registry(metric_key)
    items = list(m.item_columns)
    row: Dict[str, float] = {
        label: (float(scores[label]) if _is_number(scores.get(label)) else math.nan)
        for label in items
    }

    if metric_key in _NESTED_SPLIT:
        global_labels, behavior_labels = _NESTED_SPLIT[metric_key]
        global_mean = _mean(row.get(label) for label in global_labels)
        behavior_total = _total(row.get(label) for label in behavior_labels)

        if metric_key == "MITI":
            row["MITI_GlobalMean"] = global_mean
            row["MITI_BehaviorTotal"] = behavior_total
        elif metric_key == "PCT":
            row["PCT_GlobalMean"] = global_mean
            row["PCT_BehaviorTotal"] = behavior_total
            change, sustain = row.get("PCT_ChangeTalk"), row.get("PCT_SustainTalk")
            denom = (float(change) if _is_number(change) else 0.0) + \
                    (float(sustain) if _is_number(sustain) else 0.0)
            row["PCT_ChangeProp"] = _ratio(change, denom)
        else:  # MICI -- the one lower-is-better instrument
            row["MICI_BehaviorTotal"] = behavior_total
            row["MICI_Rate"] = _ratio(behavior_total, n_therapist_utterances)
            row["MICI_OverPraiseRate"] = _ratio(row.get("MICI_OverPraise"), n_therapist_utterances)
    elif metric_key == "WAI_SR":
        for subscale, sub_items in WAI_SR_SUBSCALES.items():
            row[f"{subscale}_Mean"] = _mean(row.get(label) for label in sub_items)
        row["WAI_TotalMean"] = _mean(row.get(label) for label in items)
        row["WAI_TotalSum"] = _total(row.get(label) for label in items)
    else:
        row[m.score_column] = _mean(row.get(label) for label in items)
        total_column = _TOTAL_COLUMN.get(metric_key)
        if total_column:
            row[total_column] = _total(row.get(label) for label in items)

    if m.score_column not in row:
        raise KeyError(
            f"_build_row({metric_key!r}) produced no {m.score_column!r} column. That column is "
            f"what every figure plots and what data.load_scores_long reads as 'score'; a metric "
            f"registered in constants.METRICS needs a reduction branch here."
        )
    return row


# ==============================================================================
#  5. ONE MODEL STATE
# ==============================================================================


def _resolve_metrics(metrics: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Normalise the ``metrics`` argument to registered STORED metric keys, de-duplicated.

    Raises:
        KeyError: naming every registered key, for a typo.
        ValueError: for a COMPOSITE (``Q1Q2``). It is computed by the loader from Q1 and Q2 after
            reading and has no partition of its own, so asking to score it means asking to write a
            file no reader will ever look for.
    """
    if metrics is None:
        return list(STORED_METRICS)
    if isinstance(metrics, str):
        metrics = [metrics]
    out: List[str] = []
    for name in metrics:
        key = str(name)
        m = metric_registry(key)                 # raises KeyError with the registered list
        if m.is_composite:
            raise ValueError(
                f"metric {key!r} is a composite of {list(m.composite_of)}; it is computed after "
                f"loading and is never stored. Score its components instead "
                f"({', '.join(m.composite_of)})."
            )
        if key not in out:
            out.append(key)
    return out


def _count_conversations(conv_dir: str) -> int:
    """How many ``pers<PID>.csv`` files a model-state folder holds. Cheap -- no parsing.

    Warning:
        On the Google Drive symlink, "the directory reads as empty" is NOT proof the conversations
        are missing: the mount can wedge on a single folder and report zero entries while every
        file is present in Drive. A whole arm suddenly appearing in the plan after it was fully
        scored is that failure, not a regression -- check the cloud before re-scoring it.
    """
    if not os.path.isdir(conv_dir):
        return 0
    return len(glob.glob(os.path.join(conv_dir, _CONV_GLOB)))


def _parquet_rowcount(path: str) -> Optional[int]:
    """Rows in an existing score parquet, or ``None`` if it is absent or unreadable.

    Reads the footer metadata rather than the data where pyarrow is available, so testing tens of
    thousands of partitions for completeness stays a metadata scan.

    An unreadable file returns ``None`` -- i.e. it is treated as MISSING and will be re-scored.
    That is the safe direction: the alternative is to count a torn file as done and leave a
    permanent hole in the lake.
    """
    if not os.path.isfile(path):
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:                            # noqa: BLE001 -- no pyarrow, or a torn footer
        pass
    try:
        return int(len(pd.read_parquet(path)))
    except Exception as exc:                     # noqa: BLE001 -- any engine/IO failure
        _log(f"WARNING: unreadable score partition {path}: {type(exc).__name__}: {exc} "
             f"-- treated as missing, so it will be re-scored")
        return None


async def _score_one_conversation(client,
                                  cfg: OracleConfig,
                                  primitives: AsyncPrimitives,
                                  state: ConversationState,
                                  *,
                                  metric_key: str,
                                  questionnaire_id: int,
                                  item_labels: Sequence[str],
                                  binding: RoleBinding,
                                  scored_at: str) -> Dict[str, Any]:
    """Grade one conversation on one rubric and return its parquet row.

    Never raises for a grading failure: the row comes back with ``oracle_success=False`` and NaN
    scores. A visible NaN row is strictly better than an absent one -- an absent row makes the
    partition permanently incomplete (so every re-run re-scores the whole state), and it hides the
    fact that a conversation could not be graded at all.
    """
    transcript = format_conversation_for_oracle(state.turns)
    n_therapist = _count_therapist_utterances(transcript)

    scores: Dict[str, Any] = {}
    attempts = 0
    success = False

    if transcript.strip():
        data, _n_questions, attempts = await get_evaluation_json(
            client, cfg, primitives, transcript, questionnaire_id
        )
        if data is not None:
            try:
                parsed = parse_json_response(data, questionnaire_id, list(item_labels))
                scores = parsed["scores_dict"]
                success = True
            except Exception as exc:             # noqa: BLE001 -- a shape the ladder let through
                _log(f"WARNING: {state.conversation_id} {metric_key}: response validated but did "
                     f"not unpack ({type(exc).__name__}: {exc}) -- row written as NaN")
                scores = {}

    row: Dict[str, Any] = {
        "persona_id": int(state.persona_id),
        "conversation_id": state.conversation_id,
    }
    row.update(_build_row(metric_key, scores, n_therapist_utterances=n_therapist))
    row.update({
        "n_utterances": int(state.n_utterances),
        "n_therapist_utterances": int(n_therapist),
        "n_attempts": int(attempts),
        "oracle_success": bool(success),
        "judge_model": binding.model,
        "judge_provider": binding.provider,
        "scored_at": scored_at,
    })
    return row


def _write_parquet(frame: pd.DataFrame, path: str) -> str:
    """Write *frame* to *path* atomically. Returns *path*.

    Temp file plus ``os.replace`` (same directory, so the rename is atomic on NTFS as well as on
    POSIX). This is load-bearing rather than tidy: :func:`discover_scorable` decides "already
    scored" from a file's existence and row count, so a process killed mid-write must leave either
    the previous complete file or no file -- never a short one that the next run counts as done.

    Notes:
        The temp name is deliberately terse. A score-lake path is already long -- data root plus
        four partition levels plus a ~55-character arm name -- and the suffix is spent against
        Windows' 260-character limit, where an over-long path fails as ``FileNotFoundError`` on the
        TEMP file while the final path is perfectly legal. The real tree leaves ~60 characters of
        headroom; a scratch ``data_root`` buried under a deep temp directory can run out.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


async def score_model_state(arm: Arm,
                            n: int,
                            metric: str,
                            *,
                            binding: RoleBinding,
                            rep: int = 0,
                            concurrency: int = DEFAULT_CONCURRENCY,
                            max_tokens: int = JUDGE_MAX_TOKENS,
                            max_retries: int = JUDGE_MAX_RETRIES,
                            request_timeout: float = JUDGE_REQUEST_TIMEOUT,
                            min_success_ratio: float = MIN_SUCCESS_RATIO,
                            progress: bool = True) -> str:
    """Grade one model state's conversations on ONE rubric and write its parquet.

    Args:
        arm: The arm to score.
        n: The MODEL STATE index (``model_iter_<n>``); ``0`` is the untrained base policy.
        metric: A registered stored metric key (``"Q1"``, ``"WAI_SR"``, ``"MICI"``, ...).
        binding: The grader. Its model decides the ``judge=<tag>`` partition, so this argument --
            not any caller-supplied string -- is what the scores are filed under.
        rep: Repeat draw. ``0`` is the full-grid draw every family reads; ``>= 1`` are
            repeatability re-draws that must never overwrite it.
        concurrency: In-flight grading calls.
        max_tokens: See :data:`JUDGE_MAX_TOKENS`.
        max_retries: Attempts per call, including the first.
        request_timeout: Per-ATTEMPT ceiling, seconds.
        min_success_ratio: See :data:`MIN_SUCCESS_RATIO`.
        progress: Print one line per state.

    Returns:
        The path of the parquet written.

    Raises:
        ValueError: for a composite metric, an unusable binding, or a model state with no
            conversations on disk.
        RuntimeError: if fewer than *min_success_ratio* of the conversations graded successfully.
            **Nothing is written in that case** -- see :data:`MIN_SUCCESS_RATIO`.

    Notes:
        **This function does not consult the lake.** It always scores and always overwrites.
        :func:`discover_scorable` is the resume layer; calling this directly on an already-scored
        state costs a full re-score and produces an equivalent file.

        Every conversation present on disk becomes a row, including one whose grading failed (NaN
        scores, ``oracle_success=False``). The row count is therefore the conversation count -- 96
        on a healthy state -- which is exactly the test :func:`discover_scorable` applies.

        Grading is one call per conversation through ``core.oracle.get_evaluation_json``: the same
        prompt builder, schema shim, validation ladder and retry policy the trainer's reward uses.
        The rubric-first prompt layout that makes vLLM's prefix cache hit is a property of
        ``questionnaires.py`` and must not be "optimized" here.
    """
    _assert_scoreable(binding)

    m = metric_registry(metric)
    if m.is_composite:
        raise ValueError(
            f"metric {metric!r} is a composite of {list(m.composite_of)} and has no partition; "
            f"score its components instead."
        )
    qid = int(m.questionnaire_id)
    tag = judge_tag(binding)

    conv_dir = arm.conv_dir(n)
    states = load_conversations_dir(conv_dir)
    if not states:
        raise ValueError(
            f"score_model_state: no conversations under {conv_dir}. Nothing to grade. (If the arm "
            f"is on the Google Drive symlink, a wedged mount reports an empty folder while every "
            f"file is present in Drive -- check the cloud before regenerating.)"
        )

    cfg = OracleConfig(
        binding=binding,
        questionnaire_ids=(qid,),
        eval_temperature=0.0,
        max_tokens=int(max_tokens),
        max_retries=int(max_retries),
        request_timeout=float(request_timeout),
        max_concurrency=int(concurrency),
        min_success_ratio=float(min_success_ratio),
    )
    client = make_oracle_client(cfg)
    # patient_concurrency is unused here -- judging makes no patient calls -- but must be >= 1.
    primitives = AsyncPrimitives(oracle_concurrency=int(concurrency), patient_concurrency=1)

    scored_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    persona_ids = sorted(states)
    started = time.time()

    rows = await asyncio.gather(*(
        _score_one_conversation(
            client, cfg, primitives, states[pid],
            metric_key=metric, questionnaire_id=qid, item_labels=m.item_columns,
            binding=binding, scored_at=scored_at,
        )
        for pid in persona_ids
    ))

    frame = pd.DataFrame(list(rows)).sort_values("persona_id", kind="mergesort")
    frame = frame.reset_index(drop=True)

    n_ok = int(frame["oracle_success"].sum())
    ratio = n_ok / len(frame) if len(frame) else 1.0
    if ratio < float(min_success_ratio):
        raise RuntimeError(
            f"score_model_state({arm.experiment_name}, {n}, {metric!r}): only {n_ok}/{len(frame)} "
            f"conversations graded successfully ({ratio:.0%} < {min_success_ratio:.0%}). Nothing "
            f"was written -- a mostly-empty partition would be counted as DONE by every later run "
            f"and would silently reduce this metric to the conversations that were easy to score. "
            f"Check the server is up and that max_tokens ({max_tokens}) is not clipping the JSON."
        )
    if n_ok < len(frame):
        _log(f"WARNING: {arm.experiment_name} {arm.model_state(n)} {metric}: "
             f"{len(frame) - n_ok}/{len(frame)} conversations did not grade -- written as NaN rows "
             f"so the hole is visible rather than absent")
    if len(frame) != N_PERSONAS and progress:
        _log(f"NOTE: {arm.experiment_name} {arm.model_state(n)} holds {len(frame)} conversations, "
             f"not {N_PERSONAS}. The partition is complete for what is on disk; if conversations "
             f"are added later the state re-enters the plan and is fully re-scored.")

    path = arm.score_path(n, metric, judge=tag, rep=int(rep))
    _write_parquet(frame, path)

    if progress:
        elapsed = time.time() - started
        _log(f"{arm.experiment_name} {arm.model_state(n)} {metric:<7} judge={tag} rep={rep}: "
             f"{len(frame)} rows ({len(frame) - n_ok} failed) in {elapsed:.1f}s -> "
             f"{os.path.basename(path)}")
    return path


# ==============================================================================
#  6. DISCOVERY -- what is still missing
# ==============================================================================


def discover_scorable(arms: Optional[Sequence[Arm]] = None,
                      *,
                      judge: str = "",
                      rep: int = 0,
                      metrics: Optional[Union[str, Sequence[str]]] = None,
                      verbose: bool = True) -> pd.DataFrame:
    """The work list: one row per (arm, model state, metric) still MISSING from the score lake.

    Args:
        arms: Arms to consider. ``None`` discovers every arm under ``data/conversations/``.
        judge: Grader tag -- the ``judge=<tag>`` partition to check. ``""`` resolves to
            ``constants.DEFAULT_JUDGE_TAG``. Build it with ``constants.judge_tag(binding)`` so it
            cannot drift from the binding that will do the grading.
        rep: Repeat draw; ``0`` is the full-grid draw.
        metrics: Metric keys, or ``None`` for all eight stored instruments
            (:data:`STORED_METRICS`).
        verbose: Print the "N to score, M already complete" summary.

    Returns:
        A frame with :data:`PLAN_COLUMNS`, ordered (experiment_name, state_index, metric). **Empty
        and correctly typed** when everything is scored, when no arm exists, or when ``data/`` is
        not mounted -- none of those is an error, and the second one is the normal state of a fresh
        checkout.

    Notes:
        **This function is what makes a re-run cost zero.** A partition is complete when its
        parquet exists AND holds at least one row per conversation currently on disk for that state
        (96 on a healthy state). Nothing below file granularity is resumable: a partially-scored
        state is re-scored whole, which is the price of never having to reconcile a half-written
        file.

        A state whose conversation count LATER grows -- a repair pass that regenerates two failed
        personas -- re-enters the plan and is re-scored in full, all 96 calls, not two. On the
        local stack that is free; on a vendor judge, check :func:`estimate_calls` first.

        An unreadable parquet counts as missing (with a warning) rather than as done, because a
        torn file counted as done is a permanent hole.
    """
    arm_list = list(discover_arms() if arms is None else arms)
    tag = (judge or "").strip().strip("/\\") or DEFAULT_JUDGE_TAG
    r = int(rep)
    metric_keys = _resolve_metrics(metrics)

    rows: List[Dict[str, Any]] = []
    n_complete = 0
    for arm in arm_list:
        key = arm.key()
        for state in arm.iters:
            conv_dir = arm.conv_dir(state)
            n_convs = _count_conversations(conv_dir)
            if n_convs == 0:
                if verbose:
                    _log(f"{arm.experiment_name} {arm.model_state(state)}: no conversations on "
                         f"disk -- skipped")
                continue
            for metric_key in metric_keys:
                path = arm.score_path(state, metric_key, judge=tag, rep=r)
                existing = _parquet_rowcount(path)
                if existing is not None and existing >= n_convs:
                    n_complete += 1
                    continue
                rows.append({
                    **key,
                    "judge": tag,
                    "rep": r,
                    "metric": metric_key,
                    "questionnaire_id": int(metric_registry(metric_key).questionnaire_id),
                    "state_index": int(state),
                    "model_state": arm.model_state(state),
                    "n_conversations": int(n_convs),
                    "n_rows_existing": int(existing or 0),
                    "path": path,
                    "conv_dir": conv_dir,
                    "data_root": arm.data_root,
                })

    if not rows:
        if verbose:
            if n_complete:
                _log(f"nothing to score for judge={tag!r} rep={r}: all {n_complete} partitions "
                     f"are complete")
            elif not arm_list:
                _log(f"no arms found under {os.path.join(DATA_DIR, 'conversations')} -- nothing "
                     f"to score (this is the normal state before the first run)")
            else:
                _log(f"no scorable model states for judge={tag!r} rep={r}")
        return _empty_frame(PLAN_COLUMNS)

    frame = pd.DataFrame(rows).sort_values(
        ["experiment_name", "state_index", "metric"], kind="mergesort"
    ).reset_index(drop=True)
    if verbose:
        n_calls = int(frame["n_conversations"].sum())
        _log(f"judge={tag!r} rep={r}: {len(frame)} partitions to score "
             f"({n_calls} grading calls), {n_complete} already complete")
    return _order_columns(frame, PLAN_COLUMNS)


# ==============================================================================
#  7. RUNNING A PLAN
# ==============================================================================


def _arm_for(experiment_name: str, data_root: str, cache: Dict[Tuple[str, str], Arm]) -> Arm:
    """Rebuild an :class:`~eda_analysis.data.Arm` from a plan row, memoized.

    A plan is a frame, so it carries the arm's NAME rather than the object. The name is the
    identity (``naming.parse_experiment_name`` is the one parser), so this reconstruction is
    lossless for everything :func:`score_model_state` needs -- it uses ``conv_dir`` and
    ``score_path``, both of which depend only on the name and the data root.
    """
    ck = (experiment_name, data_root)
    if ck not in cache:
        cache[ck] = Arm(
            experiment_name=experiment_name,
            info=parse_experiment_name(experiment_name),
            iters=(),
            data_root=data_root or DATA_DIR,
        )
    return cache[ck]


async def run_scoring(plan: pd.DataFrame,
                      *,
                      binding: RoleBinding,
                      rep: int = 0,
                      concurrency: int = DEFAULT_CONCURRENCY,
                      dry_run: bool = False,
                      progress: bool = True,
                      **score_kwargs) -> pd.DataFrame:
    """Score every row of *plan*, one model state at a time. Returns what happened to each.

    Args:
        plan: A :func:`discover_scorable` frame (or a subset of one -- filter it freely; the row
            order is preserved).
        binding: The grader. **Must be the one the plan was discovered for** -- see the raise
            below.
        rep: Repeat draw; must equal the plan's ``rep``.
        concurrency: In-flight grading calls WITHIN one model state.
        dry_run: Validate and report without making a single call. Every row comes back with
            ``status="planned"``.
        progress: Print one line per state.
        **score_kwargs: Forwarded to :func:`score_model_state` (``max_tokens``, ``max_retries``,
            ``request_timeout``, ``min_success_ratio``).

    Returns:
        A frame with :data:`RESULT_COLUMNS`: the plan plus ``status``
        (``written`` | ``planned`` | ``error``), ``n_rows``, ``n_failed``, ``n_calls``,
        ``elapsed_s`` and ``error``. Empty and correctly typed for an empty plan -- which is the
        normal result of a second ``Run_Eval`` run and must not look like a failure.

    Raises:
        ValueError: if the plan's ``judge`` column disagrees with ``judge_tag(binding)``, or its
            ``rep`` with *rep*. This is the guard that stops a grader from writing into another
            grader's partition. It cannot happen through :func:`score_model_state` (which derives
            the tag from the binding), but a plan discovered for one judge and then handed to
            another binding is a natural mistake, and it would file one model's scores under
            another's name -- unrecoverable without knowing it happened.
        ValueError: for an unusable binding (see :func:`judge_binding_for`).

    Notes:
        **Model states run SEQUENTIALLY.** Each already fans out to 96 concurrent calls, so running
        two states at once would double the in-flight load without touching the bound that was
        actually configured. Sequential also makes the resume granularity exactly one file: an
        interrupted run leaves every completed partition on disk and re-plans the rest.

        A state that fails is recorded with ``status="error"`` and the run CONTINUES. One state
        that could not be graded should not cost the twenty that could -- and the failure is in the
        returned frame, so it is reportable rather than silent.
    """
    _assert_scoreable(binding)
    tag = judge_tag(binding)

    if plan is None or not isinstance(plan, pd.DataFrame):
        raise TypeError(
            f"run_scoring: plan must be a DataFrame from discover_scorable, got "
            f"{type(plan).__name__}"
        )
    missing = [c for c in ("experiment_name", "judge", "rep", "metric", "state_index")
               if c not in plan.columns]
    if missing:
        raise ValueError(
            f"run_scoring: plan is missing {missing}; it must come from discover_scorable "
            f"(columns {list(PLAN_COLUMNS)})"
        )
    if plan.empty:
        if progress:
            _log("plan is empty -- nothing to score (a re-run costs zero calls by design)")
        return _empty_frame(RESULT_COLUMNS)

    plan_judges = sorted(set(str(j) for j in plan["judge"]))
    if plan_judges != [tag]:
        raise ValueError(
            f"run_scoring: this plan was discovered for judge(s) {plan_judges} but the binding "
            f"grades as {tag!r} ({binding.provider}:{binding.model}). Running it would file one "
            f"grader's scores under another's judge= partition, which no downstream reader can "
            f"detect. Re-run discover_scorable(judge=judge_tag(binding))."
        )
    plan_reps = sorted(set(int(v) for v in plan["rep"]))
    if plan_reps != [int(rep)]:
        raise ValueError(
            f"run_scoring: plan covers rep(s) {plan_reps} but rep={int(rep)} was passed. rep=0 is "
            f"the full-grid draw every family reads; a mismatch would overwrite it with a "
            f"repeatability re-draw."
        )

    arm_cache: Dict[Tuple[str, str], Arm] = {}
    results: List[Dict[str, Any]] = []
    started_all = time.time()

    for _, row in plan.iterrows():
        record: Dict[str, Any] = {c: row[c] for c in PLAN_COLUMNS if c in plan.columns}
        n_calls = int(row.get("n_conversations", 0) or 0)
        record.update({"status": "planned", "n_rows": 0, "n_failed": 0,
                       "n_calls": 0 if dry_run else n_calls, "elapsed_s": 0.0, "error": ""})
        if dry_run:
            record["n_calls"] = n_calls
            results.append(record)
            continue

        arm = _arm_for(str(row["experiment_name"]), str(row.get("data_root") or DATA_DIR),
                       arm_cache)
        started = time.time()
        try:
            path = await score_model_state(
                arm, int(row["state_index"]), str(row["metric"]),
                binding=binding, rep=int(rep), concurrency=int(concurrency),
                progress=progress, **score_kwargs,
            )
            written = _parquet_rowcount(path)
            record.update({
                "status": "written",
                "path": path,
                "n_rows": int(written or 0),
                "n_failed": max(n_calls - int(written or 0), 0),
                "elapsed_s": round(time.time() - started, 3),
            })
        except Exception as exc:                 # noqa: BLE001 -- one state must not kill the run
            record.update({
                "status": "error",
                "elapsed_s": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })
            _log(f"ERROR {row['experiment_name']} {row.get('model_state')} {row['metric']}: "
                 f"{type(exc).__name__}: {exc}")
        results.append(record)

    frame = _order_columns(pd.DataFrame(results), RESULT_COLUMNS)
    if progress:
        counts = frame["status"].value_counts().to_dict()
        _log(f"done in {time.time() - started_all:.1f}s: "
             + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return frame


# ==============================================================================
#  8. WHAT IT COSTS
# ==============================================================================
#
# The default Exp4 judge is a local model. It costs GPU-hours and nothing else, and this section
# exists mostly to say so without ambiguity: a cost line that implies a dollar figure for the
# default path would be actively misleading about the one property that distinguishes this
# experiment from Exp3.


@dataclass(frozen=True)
class JudgePricing:
    """USD per 1M tokens for a VENDOR judge.

    Attributes:
        input_usd_per_mtok: Uncached input.
        output_usd_per_mtok: Output.
        cache_read_mult: Multiplier applied to input tokens served from the vendor's prompt cache.
        min_cache_tokens: Shortest prefix that caches at all. A shorter prefix caches SILENTLY NOT
            AT ALL -- no error, the discount simply never appears on the bill.

    Warning:
        Vendor pricing changes without touching this file. **Verify against the billing dashboard
        before quoting any number**, and prefer a figure anchored on a receipt you actually paid.
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_mult: float = 1.0
    min_cache_tokens: int = 1024


@dataclass(frozen=True)
class TokenProfile:
    """Tokens per grading call, for a cost projection.

    Attributes:
        input_tokens: Total prompt tokens (rubric + transcript).
        cached_input_tokens: The part served from the prompt cache. Nonzero only when the fixed
            rubric-first prefix clears the vendor's ``min_cache_tokens``.
        output_tokens: The JSON response.
        source: Where the numbers came from. Print it whenever the estimate is quoted.
    """

    input_tokens: float
    cached_input_tokens: float
    output_tokens: float
    source: str = "assumed"


#: Pricing rows keyed by JUDGE TAG (``constants.judge_tag``), because that is what a plan carries.
#: Values as documented for Exp3 on 2026-06-24; see the warning on :class:`JudgePricing`.
JUDGE_PRICING: Dict[str, JudgePricing] = {
    "gpt4m": JudgePricing(0.15, 0.60, 0.50, 1024),
    "gpt4o": JudgePricing(2.50, 10.00, 0.50, 1024),
    "haiku45": JudgePricing(1.00, 5.00, 0.10, 4096),
}

#: Tags known to be VENDOR-served. Used only when no binding is supplied, to decide whether "$0
#: (local)" is the honest answer. ``binding.provider`` is the authority -- pass the binding.
VENDOR_JUDGE_TAGS = frozenset(JUDGE_PRICING) | {"gpt4omini", "sonnet5", "opus5"}

#: Central token figures for one grading call, from the prompt-length measurement recorded in
#: ``Exp4_OpenStack/CLAUDE.md`` (192 real Exp3 PTO_LA0 transcripts, o200k tokenizer: full Q2
#: prompts median 3,794 / p95 6,524 / max 10,042; the cacheable rubric-first prefix is ~1.1k, and
#: only Q1/Q2 clear OpenAI's 1,024-token minimum at all).
#:
#: Warning:
#:     This is a CENTRAL figure across eight rubrics of very different lengths, not a measurement
#:     of the plan in front of you, and it says nothing about retries. Treat any dollar figure
#:     derived from it as an order of magnitude and override it with ``usd_per_call=`` once a real
#:     receipt exists.
DEFAULT_TOKEN_PROFILE = TokenProfile(
    input_tokens=4000.0,
    cached_input_tokens=1100.0,
    output_tokens=90.0,
    source="CLAUDE.md prompt-length measurement (central across the 8 rubrics); NOT this plan",
)


def estimate_calls(plan: pd.DataFrame,
                   *,
                   binding: Optional[RoleBinding] = None,
                   max_retries: int = JUDGE_MAX_RETRIES,
                   usd_per_call: Optional[float] = None,
                   token_profile: TokenProfile = DEFAULT_TOKEN_PROFILE) -> dict:
    """How many grading calls *plan* implies -- and dollars ONLY if the judge is a vendor API.

    Args:
        plan: A :func:`discover_scorable` frame.
        binding: The grader that will run it. **Pass it.** Without it the local/vendor question is
            answered by guessing from the judge tag, and the guess defaults to LOCAL -- which is
            right for every Exp4 arm and wrong, in the expensive direction, for an unlisted vendor
            model.
        max_retries: Used only for the worst-case call count.
        usd_per_call: Override the token model with a figure anchored on a real receipt. Token
            arithmetic has many places to be wrong (tokenizer, cache hits, retries, response
            length); a receipt has none.
        token_profile: See :data:`DEFAULT_TOKEN_PROFILE`.

    Returns:
        ``{"n_partitions", "n_calls", "n_calls_worst_case", "by_metric", "judge", "provider",
        "is_local", "usd_total", "cost", "notes"}``.

        ``cost`` is a string meant to be printed as-is. For the default open stack it is exactly
        ``"$0 (local)"`` -- the local judge costs GPU-hours, and nothing this module does can be
        billed. For a vendor judge with a pricing row it is a rough projection labelled as such;
        for a vendor judge with no pricing row it says ``unknown``, never ``$0``.

    Notes:
        One call per (conversation, metric): each parquet is one rubric, and each rubric is one
        schema-constrained request. ``n_calls`` counts first attempts only; ``n_calls_worst_case``
        multiplies by *max_retries*, which is the ceiling ``core.oracle`` will spend on a grader
        that keeps returning invalid JSON.
    """
    empty_plan = plan is None or not isinstance(plan, pd.DataFrame) or plan.empty
    if empty_plan:
        tag = judge_tag(binding) if binding is not None else DEFAULT_JUDGE_TAG
        is_local = bool(binding.is_local) if binding is not None else tag not in VENDOR_JUDGE_TAGS
        return {
            "n_partitions": 0, "n_calls": 0, "n_calls_worst_case": 0, "by_metric": {},
            "judge": tag, "provider": (binding.provider if binding is not None else "unknown"),
            "is_local": is_local, "usd_total": 0.0,
            "cost": "$0 (local)" if is_local else "$0.00 (nothing to score)",
            "notes": ["empty plan -- everything is already scored"],
        }

    tags = sorted(set(str(j) for j in plan["judge"]))
    tag = judge_tag(binding) if binding is not None else (tags[0] if tags else DEFAULT_JUDGE_TAG)
    notes: List[str] = []
    if binding is not None and tags != [tag]:
        notes.append(
            f"MISMATCH: plan judge(s) {tags} vs binding tag {tag!r} -- run_scoring will refuse this"
        )
    elif len(tags) > 1:
        notes.append(f"plan mixes judge tags {tags}; costed as {tag!r}")

    n_calls = int(plan["n_conversations"].sum())
    by_metric = {
        str(k): int(v) for k, v in plan.groupby("metric")["n_conversations"].sum().items()
    }

    if binding is not None:
        is_local = bool(binding.is_local)
        provider = binding.provider
    else:
        is_local = tag not in VENDOR_JUDGE_TAGS
        provider = "openai_compat (inferred)" if is_local else "vendor (inferred)"
        notes.append(
            "provider inferred from the judge tag because no binding was passed; pass "
            "binding= for the authoritative answer"
        )

    out: Dict[str, Any] = {
        "n_partitions": int(len(plan)),
        "n_calls": n_calls,
        "n_calls_worst_case": n_calls * int(max_retries),
        "by_metric": by_metric,
        "judge": tag,
        "provider": provider,
        "is_local": is_local,
        "notes": notes,
    }

    if is_local:
        out["usd_total"] = 0.0
        out["cost"] = "$0 (local)"
        out["notes"].append(
            "the judge is served locally: this plan costs GPU-hours and no API money"
        )
        return out

    if usd_per_call is None:
        pricing = JUDGE_PRICING.get(tag)
        if pricing is None:
            out["usd_total"] = None
            out["cost"] = (f"unknown -- no pricing row for judge tag {tag!r} "
                           f"(add one to scoring.JUDGE_PRICING)")
            return out
        uncached = max(token_profile.input_tokens - token_profile.cached_input_tokens, 0.0)
        cached = token_profile.cached_input_tokens
        if cached and token_profile.cached_input_tokens < pricing.min_cache_tokens:
            # The minimum is on the CACHEABLE PREFIX, not on the whole prompt: a 6k-token call
            # whose fixed rubric prefix is 1.1k caches nothing at Anthropic's 4,096 floor. Testing
            # input_tokens here would hand a long-transcript call a discount the vendor never
            # gives, and suppress the note that says so.
            uncached, cached = token_profile.input_tokens, 0.0
            out["notes"].append(
                f"prefix below {tag}'s {pricing.min_cache_tokens}-token cache minimum: "
                f"full input price on every call"
            )
        usd_per_call = (
            uncached / 1e6 * pricing.input_usd_per_mtok
            + cached / 1e6 * pricing.input_usd_per_mtok * pricing.cache_read_mult
            + token_profile.output_tokens / 1e6 * pricing.output_usd_per_mtok
        )
        out["notes"].append(f"token profile: {token_profile.source}")

    total = float(usd_per_call) * n_calls
    out["usd_per_call"] = round(float(usd_per_call), 6)
    out["usd_total"] = round(total, 2)
    out["cost"] = (f"${total:,.2f} (ESTIMATE -- verify pricing against the billing dashboard; "
                   f"retries could take it to ${total * max_retries:,.2f})")
    return out


# ==============================================================================
#  9. PROMPT-LENGTH GATE -- the Phase 2 measurement
# ==============================================================================
#
# The server's ``max_model_len`` (16384) was sized on Exp3 transcripts written by a gpt-4o-mini
# patient, counted with proxy tokenizers (o200k: Q2 max 10,042; Llama-3.2: 10,279). Exp4's Gemma
# patient may write longer turns, and the prompts that would overflow the cap are the LONGEST
# conversations -- whose length varies by arm and by K (CLAUDE.md § VRAM budget) -- so an overflow
# is arm-dependent biased missingness on the headline metric, and it is silent: an unscoreable
# conversation is simply absent, not an error. The spec therefore asks for the measurement to be
# REPEATED on real Exp4 conversations at the Phase 2 gate. This section is where that lives in the
# scoring path: gather every conversation on disk exactly as the oracle reads it, hand the list to
# ``tools.oracle_sanity.prompt_length_report`` (the counter -- it owns the rubric prompt builder
# and the tokenizer choice, so the number is the judge's own token count, not a proxy), and reduce
# the report to a pass/fail the notebook can raise on.

#: The call this module makes, spelled out so a signature drift is NAMED in the error rather than
#: guessed at. ``tools.oracle_sanity.prompt_length_report`` belongs to the trainer-side tools
#: layer (see CLAUDE.md § Module contract, ``code/tools/oracle_sanity.py``); the keys listed are
#: the ones :func:`check_prompt_length_gate` reads.
PROMPT_LENGTH_REPORT_SIGNATURE = (
    "tools.oracle_sanity.prompt_length_report(source: str | Sequence[str], "
    "questionnaire_ids: Sequence[int] = (1, 2), *, base_url: Optional[str] = None, "
    "tokenizer=None, model: Optional[str] = None, max_model_len: Optional[int] = None) "
    "-> PromptLengthReport (.to_dict())   "
    "# dict keys read here: n_transcripts, max_model_len, per_questionnaire {qid: {n, median, "
    "p95, max, n_over}}, max_tokens (longest prompt), n_over, headroom (= max_model_len / max_tokens)"
)

#: :func:`gather_transcripts` -- one row per conversation on disk.
TRANSCRIPT_COLUMNS: Tuple[str, ...] = (
    "experiment_name", "arm_label", "state_index", "model_state", "persona_id",
    "n_utterances", "transcript",
)


def gather_transcripts(arms: Optional[Sequence[Arm]] = None,
                       *,
                       states: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Every conversation on disk for *arms*, formatted exactly as the oracle reads it.

    Args:
        arms: Arms to walk; ``None`` discovers every arm under ``data/conversations/``.
        states: Model-state indices to keep (``None`` = every state the arm has on disk).

    Returns:
        :data:`TRANSCRIPT_COLUMNS`; ``transcript`` is
        ``core.conversations.format_conversation_for_oracle(turns)`` -- the SAME string
        :func:`score_model_state` grades, so a length measured on it is a length the judge sees.
        Empty and correctly typed when nothing is on disk.

    Notes:
        Reads every CSV, so it is not free on a large lake (a full four-arm grid is
        ``4 x 11 x 96 = 4,224`` files); it runs once, after scoring, and the Drive symlink's
        "empty directory" failure applies here as everywhere (check the cloud before believing a
        zero).
    """
    arm_list = list(discover_arms() if arms is None else arms)
    wanted = None if states is None else {int(s) for s in states}
    rows: List[Dict[str, Any]] = []
    for arm in arm_list:
        key = arm.key()
        for state in arm.iters:
            if wanted is not None and int(state) not in wanted:
                continue
            for persona_id, conv in sorted(load_conversations_dir(arm.conv_dir(state)).items()):
                rows.append({
                    "experiment_name": key["experiment_name"],
                    "arm_label": key["arm_label"],
                    "state_index": int(state),
                    "model_state": arm.model_state(state),
                    "persona_id": int(persona_id),
                    "n_utterances": len(conv.turns),
                    "transcript": format_conversation_for_oracle(conv.turns),
                })
    if not rows:
        return _empty_frame(TRANSCRIPT_COLUMNS)
    return _order_columns(pd.DataFrame(rows), TRANSCRIPT_COLUMNS)


def _accepted_keywords(fn: Any) -> Optional[set]:
    """Keyword names *fn* accepts, or ``None`` when it takes ``**kwargs`` (or cannot be inspected)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}


def prompt_length_gate(transcripts: Union[pd.DataFrame, Sequence[str]],
                       *,
                       max_model_len: Optional[int] = None,
                       questionnaire_ids: Optional[Sequence[int]] = (1, 2),
                       model: Optional[str] = None,
                       base_url: Optional[str] = None) -> Dict[str, Any]:
    """Measure the realised oracle-prompt lengths against the SERVED ``max_model_len``.

    Args:
        transcripts: :func:`gather_transcripts` output, or a plain list of oracle-formatted
            transcripts.
        max_model_len: The OFFLINE fallback cap only. With a *base_url* the cap is what the
            server itself reports through ``/tokenize`` (or ``/v1/models``) and this argument is
            deliberately NOT forwarded -- an adopted server keeps whatever ``--max-model-len`` it
            was launched with, and a notebook literal (``SERVE_MAX_MODEL_LEN``) must not be
            allowed to override the number the server actually compares prompts against. If both
            are given and the served cap differs, the report carries a ``note`` saying so.
            Without a *base_url* (no server: a vendor judge, or an offline HF-tokenizer / chars
            estimate) it is forwarded as the cap to check against; ``None`` there lets the tools
            layer fall back to its assumed ``ServeSpec.max_model_len`` (labelled ``assumed``).
        questionnaire_ids: Rubrics whose prompt to build around each transcript. ``None`` means
            every stored instrument (:data:`STORED_METRICS`); the default ``(1, 2)`` is the
            training reward, and Q2 carries the longest rubric.
        model: The judge binding's model id -- sent with the server's ``/tokenize`` request and
            the fallback HF tokenizer lookup, so the count is the served model's own.
        base_url: The judge binding's endpoint. With it the report counts through vLLM's
            ``/tokenize`` (THE measurement -- the number the server compares against its cap)
            AND takes the cap from the same answer; without it the report falls back to a local
            HF tokenizer or a labelled chars/3.5 estimate, and says so in its ``method``.

    Returns:
        The ``to_dict()`` of the report ``tools.oracle_sanity.prompt_length_report`` produced
        (see :data:`PROMPT_LENGTH_REPORT_SIGNATURE`), with ``n_transcripts`` (and, on the
        offline path, ``max_model_len``) filled in if the report omitted them. Reduce it with
        :func:`check_prompt_length_gate`; render it with
        ``tools.oracle_sanity.format_prompt_length_report`` (which accepts the dict). With zero
        transcripts nothing is measured and the dict says so in the formatter's own vocabulary
        (``gate.passed`` True with no failures, ``method`` ``"none"``, ``measured`` False) so a
        vacuous pass renders as one, never as ``VERDICT: FAIL`` followed by ``GATE PASSED``.

    Raises:
        ImportError: when the tools layer has no ``prompt_length_report`` -- the measurement is
            then simply not available in this checkout, and the error says which function is
            missing rather than letting the gate silently pass.
        TypeError: when the report is not a mapping.

    Notes:
        The call is keyword-filtered against the function's real signature: a keyword the tools
        layer does not accept is dropped with a printed WARNING naming the documented call, so a
        signature drift degrades to a visible message instead of a ``TypeError`` in the notebook.
    """
    try:
        from tools import oracle_sanity as _sanity  # canonical copy; code/ is on sys.path
    except ImportError as exc:
        raise ImportError(
            f"cannot import tools.oracle_sanity ({exc}); the prompt-length gate needs "
            f"{PROMPT_LENGTH_REPORT_SIGNATURE}"
        ) from exc
    fn = getattr(_sanity, "prompt_length_report", None)
    if fn is None:
        raise ImportError(
            "tools.oracle_sanity has no prompt_length_report, so the Phase 2 prompt-length "
            "measurement is NOT available in this checkout. The gate refuses to pass on nothing "
            f"measured. Expected: {PROMPT_LENGTH_REPORT_SIGNATURE}"
        )

    if isinstance(transcripts, pd.DataFrame):
        texts = [str(t) for t in transcripts["transcript"].tolist()] if not transcripts.empty else []
    else:
        texts = [str(t) for t in transcripts]
    if not texts:
        # The vacuous case, in the formatter's own shape: an empty gate block is a PASS with no
        # failures (the notebook reads gate.passed / n_over, the formatter reads gate + method +
        # measured), and nothing claims a tokenizer or a cap that was never consulted.
        return {
            "conv_dir": "<0 transcripts in memory>",
            "n_transcripts": 0, "n_conversations": 0,
            "questionnaire_ids": [],
            "max_model_len": None if max_model_len is None else int(max_model_len),
            "max_model_len_source": "argument" if max_model_len is not None else "none",
            "method": "none", "measured": False,
            "grader": {"model": model, "base_url": base_url or None},
            "per_rubric": {}, "per_questionnaire": {},
            "n_over": 0, "max_tokens": 0, "headroom": None,
            "gate": {"passed": True, "failures": []},
            "notes": ["no conversations on disk -- nothing was measured (vacuous pass)"],
        }

    if questionnaire_ids is None:
        qids: Tuple[int, ...] = tuple(
            int(metric_registry(key).questionnaire_id) for key in STORED_METRICS
        )
    else:
        qids = tuple(int(q) for q in questionnaire_ids)
    kwargs: Dict[str, Any] = {"questionnaire_ids": qids}
    if model is not None:
        kwargs["model"] = model
    if base_url:
        # The SERVED cap wins: prompt_length_report reads it back from /tokenize (or /v1/models)
        # when no explicit cap is passed, so the argument is withheld here on purpose.
        kwargs["base_url"] = base_url
    elif max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    accepted = _accepted_keywords(fn)
    dropped = [k for k in kwargs if accepted is not None and k not in accepted]
    if dropped:
        _log(f"WARNING: prompt_length_report does not accept {dropped}; calling without them. "
             f"Documented call: {PROMPT_LENGTH_REPORT_SIGNATURE}")
    report = fn(texts, **{k: v for k, v in kwargs.items() if k not in dropped})
    if not isinstance(report, Mapping) and callable(getattr(report, "to_dict", None)):
        report = report.to_dict()       # the tools layer's dataclass -> its archived shape
    if not isinstance(report, Mapping):
        raise TypeError(
            f"prompt_length_report returned {type(report).__name__}, expected a dict "
            f"({PROMPT_LENGTH_REPORT_SIGNATURE})"
        )
    out: Dict[str, Any] = dict(report)
    out.setdefault("n_transcripts", len(texts))
    if max_model_len is not None:
        if not base_url:
            out.setdefault("max_model_len", int(max_model_len))
        else:
            served = out.get("max_model_len")
            if served is not None and int(served) != int(max_model_len):
                notes = list(out.get("notes") or ())
                notes.append(
                    f"max_model_len={int(max_model_len)} was passed but NOT used: the served cap "
                    f"({served}, {out.get('max_model_len_source', 'server')}) is what the server "
                    f"compares prompts against. Keep the notebook literal in step with the "
                    f"launch flag, or restart the server -- an adopted one keeps its old cap."
                )
                out["notes"] = notes
    return out


def check_prompt_length_gate(report: Mapping[str, Any],
                             *,
                             min_headroom: float = 1.25) -> Tuple[bool, List[str]]:
    """Reduce a prompt-length report to ``(passed, messages)``.

    Passes iff **no** measured prompt exceeds ``max_model_len``. Headroom below *min_headroom*
    (``max_model_len / longest prompt``) is reported as a note, never a failure -- the spec's own
    margin was ~1.6x over Exp3's longest prompt, and eating into it is a reason to raise the cap
    before the next arm, not to refuse the scores already on disk.

    Args:
        report: :func:`prompt_length_gate` output.
        min_headroom: Ratio below which a "raise the cap" note is emitted.

    Returns:
        ``(passed, messages)``. With zero transcripts measured the gate is vacuous: it passes and
        says so, because a fresh checkout has nothing to measure and that is not a failure.
        A report that carries neither ``n_over`` nor a per-prompt token list FAILS with a message
        naming the documented shape -- an undecidable gate must not read as a green one.
    """
    max_model_len = report.get("max_model_len")
    n_transcripts = int(report.get("n_transcripts") or 0)
    if n_transcripts == 0:
        return True, ["nothing measured: no conversations on disk (the gate is vacuous)"]

    n_over = report.get("n_over")
    if n_over is None:
        counts = report.get("prompt_tokens") or report.get("token_counts")
        if counts is None or max_model_len is None:
            return False, [
                "the prompt-length report carries neither 'n_over' nor a per-prompt token list "
                f"('prompt_tokens'/'token_counts'), so the gate cannot decide. Keys present: "
                f"{sorted(report)}. Documented shape: {PROMPT_LENGTH_REPORT_SIGNATURE}"
            ]
        n_over = sum(1 for c in counts if float(c) > float(max_model_len))
    n_over = int(n_over)

    messages: List[str] = []
    max_tokens = report.get("max_tokens")
    if n_over > 0:
        messages.append(
            f"{n_over} of {n_transcripts} conversations produce an oracle prompt over "
            f"max_model_len={max_model_len}"
            + (f" (longest {int(max_tokens):,} tokens)" if max_tokens else "")
            + ". Those conversations CANNOT be graded, they are the longest ones, and session "
              "length varies by arm and by K -- raise SERVE_MAX_MODEL_LEN here AND "
              "VLLM_MAX_MODEL_LEN in both trainer notebooks above the longest prompt, then "
              "re-score the affected partitions. Never lower the cap to save memory."
        )
    if max_tokens and max_model_len:
        headroom = float(max_model_len) / float(max_tokens)
        if headroom < float(min_headroom):
            messages.append(
                f"note: headroom is only {headroom:.2f}x (longest prompt {int(max_tokens):,} vs "
                f"cap {max_model_len}); the spec sized the cap at ~1.6x Exp3's longest prompt. "
                f"Raise the cap before the next arm rather than after the first overflow."
            )
    return n_over == 0, messages
