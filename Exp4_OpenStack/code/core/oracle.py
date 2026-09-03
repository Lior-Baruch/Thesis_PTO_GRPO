"""oracle.py -- schema-constrained grading of one conversation, and the scalar training reward.

Everything the policy learns comes through this module. GRPO's advantages and PTO's
(chosen, rejected) split are both functions of one number per candidate, and this is where that
number is produced: build the rubric prompt, ask the grader for JSON that matches a schema,
**refuse to believe the answer until it has been checked**, average the item scores, then average
across the configured questionnaires.

Why the checking is the interesting part
----------------------------------------
Exp3 graded with ``gpt-4o-mini`` under OpenAI's strict structured outputs, where a response that
does not match the schema is not returned at all. Exp4's default grader is an open model behind
vLLM, and guided decoding there is a best-effort constraint engine, not a contract: some builds
honour ``minItems``/``maxItems``, some quietly ignore them, and some reject the ``strict`` key
outright. Exp3's eval-side Claude judge already taught the lesson in its own way -- the Anthropic
Messages API rejects numeric bounds and array-length constraints, so
``scoring/judge.py::_strip_unsupported_constraints`` had to fold them into ``description`` text
rather than simply drop them. The reason it could not simply drop them is the failure mode:

    when a backend silently ignores ``minItems``/``maxItems``, a wrong-length ``scores`` array is
    not an error. It parses. The exception it raises downstream is swallowed, that conversation is
    dropped, and the result is **biased missingness on the headline metric** -- the conversations
    the grader found hardest to score are exactly the ones that disappear.

That is why the validation ladder below is not optional and is not defensive boilerplate. It is
the thing that makes an open grader safe to train against. It runs client-side, after every call,
regardless of what the schema was supposed to guarantee.

A grader can also fail the other way -- honour the schema perfectly and return a 4 for every item
of every conversation. Nothing in this module can see that, because each response is individually
valid; it is caught upstream by ``tools/oracle_sanity.py``, which checks pooled variance against a
frozen fixture before a run is allowed to start. This module's job is the loud half.

What is NOT here
----------------
No torch, no trainer imports, no flooring of degenerate completions, no look-ahead. The EDA's
scoring path imports this module to score conversations after the fact with a held-out judge, and
it must not drag CUDA in to do so. :data:`REWARD_FLOOR` is *defined* here (it has to sit below the
rubric scale minimum, and this is the module that knows the scales) but it is *applied* in
``core/reward.py``, which is the only place that knows which completions were degenerate.

Changes from Exp3's ``_shared/reward.py``
-----------------------------------------
* :attr:`OracleConfig.binding` is **mandatory**. Exp3 allowed ``binding=None`` meaning "grade with
  whatever client the caller happened to pass", an identity path kept so that runs predating role
  bindings stayed byte-reproducible. Exp4 has no such history: every role is bound explicitly, and
  the arm name carries the oracle tag, so a config that does not say who graded it is a bug.
* ``max_tokens`` is a config field (Exp3 hardcoded 256).
* ``request_timeout`` wraps each attempt in ``asyncio.wait_for``, and the backoff between attempts
  is capped -- see :data:`MAX_BACKOFF_SECONDS`.
* :func:`response_format_for` is new: the single place a provider difference is allowed to live.

Prompt caching -- do not "optimize" the prompt
----------------------------------------------
``questionnaires.get_prompt_eval_questionnaire`` puts the fixed instructions and the rubric FIRST
and the variable transcript LAST. That ordering is load-bearing: on the OpenAI API it hits the
automatic prefix cache (~50% input discount), and on vLLM it hits prefix caching, which is a large
part of why the open stack is affordable at ~10k oracle calls per iteration. Moving the transcript
ahead of the rubric, or trimming the instructions below OpenAI's 1,024-token minimum, silently
stops the reuse -- silently, because nothing fails; the run just gets slower and (on an API arm)
more expensive. This module is where someone would be tempted to shorten the prompt. Do not.

Known exception, inherited from Exp3: MITI/PCT/MICI interpolate a per-conversation utterance count
into the instructions *ahead of* the rubric, so their cacheable prefix is short. That count is the
denominator of their rate metrics, so it cannot be moved without breaking comparability with every
conversation already scored.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import APIStatusError

from core.concurrency import AsyncPrimitives
from questionnaires import (
    QuestionnaireID,
    get_prompt_eval_questionnaire,
    parse_json_response,
)
from roles import RoleBinding, make_client

__all__ = [
    "REWARD_FLOOR",
    "MAX_BACKOFF_SECONDS",
    "NON_RETRYABLE",
    "RETRYABLE_4XX",
    "is_non_retryable_http_error",
    "NESTED_QUESTIONNAIRE_IDS",
    "OPENAI_SHAPED_PROVIDERS",
    "OracleConfig",
    "openai_compat_strict",
    "set_openai_compat_strict",
    "response_format_for",
    "make_oracle_client",
    "get_evaluation_json",
    "score_conversation",
    "score_conversations_batch",
    "batch_success_ratio",
]


# ==============================================================================
#                              CONSTANTS
# ==============================================================================

#: Reward assigned to a degenerate completion (one that cleans to the empty string because the
#: policy self-played a ChatML marker and produced no real turn). Defined here because it must sit
#: strictly BELOW every rubric's ``scale_min`` -- all of them start at 1 -- so a degenerate sibling
#: is unambiguously the worst member of its GRPO group. **Applied in ``core/reward.py``**, which is
#: the only caller that knows which completions were degenerate; this module never floors anything.
REWARD_FLOOR = 0.0

#: Ceiling on the exponential backoff between oracle attempts, in seconds.
#:
#: Exp3 slept an uncapped ``2 ** attempt`` and got away with it because ``max_retries`` was 3
#: (1 s + 2 s). Exp4's whole retry philosophy is "short per-attempt timeout times MANY attempts"
#: (see ``RoleBinding.request_timeout``), and uncapped doubling turns that into precisely the long
#: total budget the philosophy exists to avoid: at ``max_retries=10`` the last sleep alone is 512 s.
#: The cap keeps the worst case linear in the attempt count.
MAX_BACKOFF_SECONDS = 30.0

#: Exceptions that mean "the caller has a bug", not "the network hiccuped". Retrying a programming
#: error just burns ``max_retries`` calls per candidate before failing anyway, so these
#: short-circuit immediately.
#:
#: WARNING: a ``KeyError`` raised while inspecting a *model response* is NOT a programming error --
#: it is a malformed response, which is exactly what retries are for. The nested-rubric validator
#: below therefore converts response-derived ``KeyError``/``TypeError`` into ``ValueError`` so this
#: classification stays honest.
NON_RETRYABLE = (KeyError, TypeError)

#: The two 4xx statuses that DO mean "try again": 408 Request Timeout and 429 Too Many Requests.
#: Every other 4xx is the server rejecting THIS request as sent -- a prompt over
#: ``--max-model-len`` (400), a ``json_schema`` key the pinned vLLM does not accept (400/422), a
#: wrong model id (404), a bad key (401/403) -- and resending the identical request
#: ``max_retries`` times with backoff in between cannot change the answer. It only delays the
#: failure by the whole retry budget per candidate, which at G=8 x 16 prompts x 2 rubrics is a
#: long, silent stall before ``min_success_ratio`` finally fires. 5xx and connection errors are
#: still retried: those are the server, not the request.
RETRYABLE_4XX = frozenset({408, 429})


def is_non_retryable_http_error(exc: BaseException) -> bool:
    """Is *exc* an HTTP status error that a retry cannot fix (4xx other than 408/429)?

    Checked against ``openai.APIStatusError`` (the pinned ``openai==2.36.0`` raises one subclass
    per status -- ``BadRequestError``, ``NotFoundError``, ``UnprocessableEntityError``, ... -- all
    carrying ``status_code``). ``RateLimitError`` (429) and a 408 are retryable and return False;
    so does everything that is not an ``APIStatusError`` (timeouts, connection errors, 5xx,
    validation ``ValueError``s).
    """
    if not isinstance(exc, APIStatusError):
        return False
    status = int(getattr(exc, "status_code", 0) or 0)
    return 400 <= status < 500 and status not in RETRYABLE_4XX


#: Questionnaires whose response is ``{globals: {...}, behaviors: {...}}`` rather than a flat
#: ``scores`` array. They need their own validation branch (see :func:`_validate_nested`).
NESTED_QUESTIONNAIRE_IDS = frozenset({
    QuestionnaireID.MITI.value,
    QuestionnaireID.PCT.value,
    QuestionnaireID.MICI.value,
})

#: Providers whose call shape this module speaks: ``chat.completions`` plus
#: ``response_format={"type": "json_schema"}``. Anthropic's Messages API is deliberately excluded
#: -- it needs the constraint-stripping shim, and a Claude grader belongs on the eval side, never
#: as a training oracle.
OPENAI_SHAPED_PROVIDERS = ("openai", "openai_compat")


# ==============================================================================
#                         PROVIDER QUIRK: strict json_schema
# ==============================================================================

# The ONE provider difference this module tolerates. OpenAI requires `strict: true` for structured
# outputs to actually be enforced; vLLM's OpenAI-compatible layer accepts it on recent builds but
# some versions 400 on the unknown key. Module-level rather than per-binding on purpose: it is a
# property of the pinned SERVER build, not of a role, and the default stack has one server behind
# every local binding, so a per-binding switch would just be three copies of one fact.
_OPENAI_COMPAT_STRICT = True


def openai_compat_strict() -> bool:
    """Whether ``strict: true`` is currently sent to ``openai_compat`` backends."""
    return _OPENAI_COMPAT_STRICT


def set_openai_compat_strict(enabled: bool) -> bool:
    """Turn the ``strict`` key on/off for ``openai_compat`` requests; returns the previous value.

    Args:
        enabled: True to send ``"strict": True`` inside ``json_schema`` (the default), False to
            omit the key entirely.

    Returns:
        The previous setting, so a caller can restore it.

    Notes:
        **The signal to call this with False** is a ``400`` from the local server whose message
        mentions ``strict`` or ``json_schema`` -- e.g. "unknown field", "extra fields not
        permitted", or a pydantic validation error naming ``json_schema``. That is a pinned-vLLM
        incompatibility, not a bad schema; flip the flag once at notebook start (before any
        scoring) rather than editing call sites.

        Omitting ``strict`` does not merely relax a formality -- on some builds it is what turns
        guided decoding from "enforced" into "suggested". Everything still works because the
        validation ladder below re-checks every field client-side, but expect the retry count to
        rise, and treat a sudden jump in ``attempts`` as evidence the flag is off when it should
        be on. Never disable it for the ``openai`` provider: this flag does not apply there, and
        OpenAI without ``strict`` is genuinely unvalidated.

        Not thread-safe and not per-run state; set it once, early.
    """
    global _OPENAI_COMPAT_STRICT
    previous = _OPENAI_COMPAT_STRICT
    _OPENAI_COMPAT_STRICT = bool(enabled)
    return previous


def response_format_for(binding: RoleBinding, schema: dict, name: str) -> dict:
    """Build the ``response_format`` body for *binding*, honouring its provider's quirks.

    Args:
        binding: The oracle's role binding. Must be an OpenAI-shaped provider.
        schema: A JSON Schema dict, as produced by ``questionnaires.make_eval_schema`` and its
            nested siblings.
        name: Schema name sent to the provider. Must match ``^[A-Za-z0-9_-]+$``.

    Returns:
        ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}``
        for ``openai``; the same shape for ``openai_compat``, with ``strict`` present only when
        :func:`openai_compat_strict` is True.

    Raises:
        ValueError: for a provider this module cannot speak to (notably ``anthropic``).

    Notes:
        **This is the only place in Exp4 where a provider difference may live.** Everything else --
        the prompt, the retry loop, the validation ladder, the aggregation -- is provider-blind on
        purpose, so that flipping the oracle from the local Gemma to the OpenAI API is a one-line
        config change and not a second code path that can drift.

        OpenAI's strict mode imposes structural requirements on the schema itself: every object
        must carry ``"additionalProperties": false`` and list ALL of its properties in
        ``"required"``. Every builder in ``questionnaires.py`` already satisfies this. A new rubric
        that does not will be rejected at request time with a schema error, which is the good
        failure -- it happens on the first call, not after an iteration of quietly unconstrained
        output.

        Anthropic is refused rather than shimmed. Its Messages API rejects ``minimum``/``maximum``/
        ``minItems``/``maxItems``, so a Claude grader needs those constraints folded into
        ``description`` text (Exp3's ``scoring/judge.py::_strip_unsupported_constraints``). That
        belongs on the eval side, where a re-score is cheap and re-runnable; a training oracle
        whose length constraint is advisory is how biased missingness gets into the reward itself.
    """
    if binding.provider not in OPENAI_SHAPED_PROVIDERS:
        raise ValueError(
            f"Oracle provider {binding.provider!r} is not supported as a TRAINING oracle: this "
            f"module needs chat.completions + response_format json_schema. Use one of "
            f"{OPENAI_SHAPED_PROVIDERS}. For a Claude grader, bind it as an eval JUDGE instead -- "
            f"the Messages API needs the constraint-stripping shim, and a training reward is the "
            f"wrong place for an advisory schema."
        )

    inner: Dict[str, Any] = {"name": name, "schema": schema}
    if binding.provider == "openai" or _OPENAI_COMPAT_STRICT:
        inner["strict"] = True
    return {"type": "json_schema", "json_schema": inner}


# ==============================================================================
#                                 CONFIG
# ==============================================================================


@dataclass(frozen=True)
class OracleConfig:
    """Which grader scores the training reward, on what rubric, and how hard it tries.

    Frozen so it can be stashed in other frozen configs and hashed; ``questionnaire_ids`` is
    coerced to a tuple in ``__post_init__`` so a notebook global written as a list stays hashable.

    Attributes:
        binding: **Mandatory.** The grader's provider/model/endpoint and its per-attempt timeout
            and retry budget. Exp3's ``binding=None`` fallback ("score with whatever client the
            caller passed") does not exist here: in Exp4 the oracle tag is part of every arm name,
            so an unbound oracle would be an arm whose identity does not describe what trained it.
        questionnaire_ids: Rubrics to average over. ``(1, 2)`` -- Q1+Q2 -- matches the ICLR
            look-ahead paper and every Exp3 arm.
        eval_temperature: 0.0. A grader filling a fixed rubric has nothing to sample.
        max_tokens: Response budget for the JSON. Q2 returns 17 integers, so 256 is ample for an
            API model that emits nothing else. A local model that opens with whitespace, a code
            fence or a one-line preamble can clip the JSON at this ceiling -- which surfaces as
            unparseable content, retries, and (if it persists) exactly the biased missingness the
            sanity gate exists to catch. If ``attempts`` is high on long transcripts, raise this
            before suspecting the schema.
        max_retries: Attempts per (conversation, questionnaire) call, including the first.
        request_timeout: **Per attempt**, in seconds, enforced by ``asyncio.wait_for``. Distinct
            from ``binding.request_timeout``, which the SDK applies to the socket: the oracle can
            afford a longer ceiling than a patient turn because it is never on the critical path
            of a lock-step simulation.
        max_concurrency: Advisory record of the bound the caller's ``AsyncPrimitives`` was built
            with. This module does not create semaphores; it uses the one it is handed.
        min_success_ratio: Floor below which ``core/reward.py`` aborts training. Read here, acted
            on there -- see :func:`batch_success_ratio`. It is the LAST line of defence, not the
            only one: an individual failure is repaired per-group by
            ``core.reward.rewards_for_trl`` (the candidate takes its siblings' mean, so it carries
            ~zero advantage). What this floor still catches is the case that repair cannot fix --
            a grader failing often enough that the *surviving* scores are a biased subset of the
            candidates, which no substitution can undo.
    """

    binding: RoleBinding
    questionnaire_ids: Tuple[int, ...] = (1, 2)
    eval_temperature: float = 0.0
    max_tokens: int = 256
    max_retries: int = 3
    request_timeout: float = 120.0
    max_concurrency: int = 64
    min_success_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.binding, RoleBinding):
            raise TypeError(
                "OracleConfig.binding is mandatory and must be a RoleBinding "
                f"(got {type(self.binding).__name__}). Exp3's None-fallback is gone: bind the "
                "oracle explicitly so the arm name records who graded it."
            )
        if self.binding.provider not in OPENAI_SHAPED_PROVIDERS:
            raise ValueError(
                f"Oracle binding provider {self.binding.provider!r} cannot serve as a training "
                f"oracle; expected one of {OPENAI_SHAPED_PROVIDERS}."
            )

        ids = tuple(int(q) for q in self.questionnaire_ids)
        if not ids:
            raise ValueError("OracleConfig.questionnaire_ids must name at least one rubric.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"OracleConfig.questionnaire_ids has duplicates: {ids}")
        object.__setattr__(self, "questionnaire_ids", ids)

        if self.max_retries < 1:
            raise ValueError("OracleConfig.max_retries must be >= 1 (the first attempt counts).")
        if self.max_tokens < 1:
            raise ValueError("OracleConfig.max_tokens must be >= 1.")
        if self.request_timeout <= 0:
            raise ValueError("OracleConfig.request_timeout must be > 0 seconds.")
        if self.max_concurrency < 1:
            raise ValueError("OracleConfig.max_concurrency must be >= 1.")
        if not 0.0 <= self.min_success_ratio <= 1.0:
            raise ValueError("OracleConfig.min_success_ratio must lie in [0, 1].")

    @property
    def model(self) -> str:
        """The grader's model id, as the provider spells it."""
        return self.binding.model

    @property
    def tag(self) -> str:
        """The grader's short tag, as it appears in ``EXPERIMENT_NAME``."""
        return self.binding.tag


def make_oracle_client(cfg: OracleConfig, *, api_key: Optional[str] = None):
    """The async client for *cfg*'s binding (cached by ``roles.make_client``).

    Provided so callers never have to reach past the config to build one, and so there is a single
    obvious answer to "which client do I pass to :func:`score_conversation`?". Passing a client
    built from a *different* binding is not detectable here -- the model id travels with the
    request, but the endpoint travels with the client -- so build it from the config.
    """
    return make_client(cfg.binding, api_key=api_key)


# ==============================================================================
#                          THE VALIDATION LADDER
# ==============================================================================


def _validate_scores_array(data: dict,
                           questionnaire_id: int,
                           n_questions: int,
                           scale_min: int,
                           scale_max: int) -> float:
    """Check a flat ``{questionnaire_id, scores: [...]}`` response and return its mean.

    Three rungs, in order, all raising ``ValueError`` (retryable) on failure:

    1. ``questionnaire_id`` ECHOES the requested id. Catches a grader answering the previous
       prompt from its own context, and catches a mis-routed batch.
    2. ``len(scores)`` equals the rubric's item count. This is the rung that cannot be skipped:
       ``minItems``/``maxItems`` are exactly the constraints a backend is most likely to ignore,
       and a short array averages to a number that looks perfectly reasonable.
    3. Every element is an ``int`` inside ``[scale_min, scale_max]``. ``isinstance(s, int)``
       rejects floats and strings; note it also accepts ``bool``, which is a Python quirk and
       harmless here -- ``True`` would still have to fall inside the scale, and 1 is a legal score.

    Returns:
        The unweighted mean of the item scores.
    """
    if data.get("questionnaire_id") != questionnaire_id:
        raise ValueError(
            f"Wrong questionnaire_id: expected {questionnaire_id}, got {data.get('questionnaire_id')}"
        )

    scores = data.get("scores", [])
    if not isinstance(scores, list):
        raise ValueError(f"'scores' must be a list, got {type(scores).__name__}")
    if len(scores) != n_questions:
        raise ValueError(f"Expected {n_questions} scores, got {len(scores)}")
    if any(not isinstance(s, int) or s < scale_min or s > scale_max for s in scores):
        raise ValueError(f"Invalid score values (expected integers {scale_min}-{scale_max}): {scores}")

    return float(fmean(scores))


def _validate_nested(data: dict,
                     questionnaire_id: int,
                     labels: Sequence[str],
                     scale_min: int,
                     scale_max: int) -> float:
    """Check a ``{globals, behaviors}`` response (MITI / PCT / MICI) and return its globals mean.

    Delegates the structural check to ``questionnaires.parse_json_response`` so the canonical label
    lists live in exactly one place, then adds the type/range rung that function does not do.

    Returns:
        The mean of the GLOBAL ratings only -- the convention ``parse_json_response`` uses and the
        one the EDA's MITI metric reports. The behaviour COUNTS are unbounded above and are not
        on the rubric's 1-5 scale, so averaging them into the reward would silently rescale it.

    Notes:
        Exp3's training oracle never took this branch: it only ever trained on array-shaped
        rubrics. Exp4's arm-name grammar admits ``QTAG=MITI``, so without this branch selecting
        MITI as the training oracle would fail 100% of calls after burning ``max_retries`` each.

        ``parse_json_response`` indexes its label lists directly, so a response missing a key
        raises ``KeyError``. That is a malformed RESPONSE, not a programming error, so it is
        re-raised as ``ValueError`` -- otherwise :data:`NON_RETRYABLE` would short-circuit a
        perfectly retryable hiccup.
    """
    try:
        parsed = parse_json_response(data, questionnaire_id, list(labels))
    except (KeyError, TypeError) as e:
        raise ValueError(f"Malformed nested response for qid={questionnaire_id}: {e!r}") from e

    globals_dict = parsed.get("globals") or {}
    behaviors_dict = parsed.get("behaviors") or {}

    bad_globals = [
        (k, v) for k, v in globals_dict.items()
        if not isinstance(v, int) or v < scale_min or v > scale_max
    ]
    if bad_globals:
        raise ValueError(
            f"Invalid global ratings (expected integers {scale_min}-{scale_max}): {bad_globals}"
        )
    bad_behaviors = [(k, v) for k, v in behaviors_dict.items() if not isinstance(v, int) or v < 0]
    if bad_behaviors:
        raise ValueError(f"Invalid behavior counts (expected integers >= 0): {bad_behaviors}")

    return float(parsed["mean_score"])


# ==============================================================================
#                              ONE ORACLE CALL
# ==============================================================================


async def get_evaluation_json(client,
                              cfg: OracleConfig,
                              primitives: AsyncPrimitives,
                              conversation_text: str,
                              questionnaire_id: int) -> Tuple[Optional[dict], int, int]:
    """Score one (conversation, questionnaire) pair, retrying until the answer validates.

    Args:
        client: An async OpenAI-shaped client, built from ``cfg.binding`` (see
            :func:`make_oracle_client`).
        cfg: The oracle configuration.
        primitives: Supplies ``oracle_sem()``, the bound on concurrent grading calls.
        conversation_text: The COMPLETE transcript to grade, in the
            ``"[THERAPIST]: ...\\n\\n[PATIENT]: ..."`` format. Unlike Exp3 this takes the finished
            text: the caller (look-ahead or not) already knows what it wants graded, and splitting
            it into transcript + completion here only created a second way to build the same
            string.
        questionnaire_id: Which rubric to score.

    Returns:
        ``(data, n_questions, attempts)``. On success ``data`` is the parsed response augmented
        with a ``"mean_score"`` float; on failure it is ``None``. ``attempts`` counts oracle HTTP
        calls made, including retries, and is surfaced so the EDA can see grading cost per
        candidate.

    Notes:
        **The semaphore wraps only the request, never the backoff sleep.** A retrying call that
        held its permit while sleeping would occupy a concurrency slot doing nothing, and with a
        capped-but-real backoff that is how a batch stalls behind a handful of unlucky calls.

        **``asyncio.wait_for`` is a second, independent bound** on top of the timeout the SDK
        client already carries. They are set to the same intent but protect different things: the
        SDK's guards the socket, this one guards the coroutine, and a client that hangs *after* the
        response (or before the request leaves) is only caught by the latter.

        ``max_retries`` is a per-call budget, so a batch of B conversations over Q questionnaires
        can in the worst case make ``B * Q * max_retries`` calls. That is the intended shape --
        short timeout, many attempts -- but it is why :data:`MAX_BACKOFF_SECONDS` exists.

        **Two things short-circuit the budget**: a programming error (:data:`NON_RETRYABLE`) and
        an HTTP 4xx other than 408/429 (:func:`is_non_retryable_http_error`) -- both are answers
        that resending the same request cannot change. A response that parses but fails the
        validation ladder is neither: it is retried, because a grader that produced a wrong-length
        array once may well produce the right one next time.
    """
    qid = int(questionnaire_id)
    eval_dict = get_prompt_eval_questionnaire(questionnaire=qid, conversation=conversation_text)
    eval_prompt = eval_dict["prompt"]
    n_questions = int(eval_dict["questions_count"])
    schema = eval_dict["schema"]
    scale_min = int(eval_dict["scale_min"])
    scale_max = int(eval_dict["scale_max"])
    labels = eval_dict["labels"]

    response_format = response_format_for(
        cfg.binding, schema, f"questionnaire_{qid}_evaluation"
    )
    extra_body = cfg.binding.extra_body

    for attempt in range(cfg.max_retries):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": cfg.binding.model,
                "messages": [{"role": "user", "content": eval_prompt}],
                "temperature": cfg.eval_temperature,
                "max_tokens": cfg.max_tokens,
                "response_format": response_format,
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            async with primitives.oracle_sem():
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**request_kwargs),
                    timeout=cfg.request_timeout,
                )

            content = resp.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty oracle response")

            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError(f"Oracle response is not a JSON object: {type(data).__name__}")

            if qid in NESTED_QUESTIONNAIRE_IDS:
                mean_score = _validate_nested(data, qid, labels, scale_min, scale_max)
            else:
                mean_score = _validate_scores_array(data, qid, n_questions, scale_min, scale_max)

            data["mean_score"] = mean_score
            return data, n_questions, attempt + 1

        except NON_RETRYABLE as e:
            # A bug in this file or its callers, not a flaky grader. Retrying it would burn the
            # whole budget to arrive at the same exception.
            print(f"  [oracle] non-retryable error (qid={qid}): {e!r}")
            return None, n_questions, attempt + 1

        except Exception as e:
            if is_non_retryable_http_error(e):
                # The server rejected the request as sent (4xx other than 408/429); the same
                # bytes will be rejected again. The JSON-validation ladder's ValueErrors are
                # NOT in this branch -- those are grader failures, and retrying them is the
                # point. Still a failure (score None), so it still counts against
                # min_success_ratio.
                print(
                    f"  [oracle] HTTP {getattr(e, 'status_code', '?')} is not retryable "
                    f"(qid={qid}): {e!r}"
                )
                return None, n_questions, attempt + 1
            if attempt >= cfg.max_retries - 1:
                print(f"  [oracle] failed after {cfg.max_retries} attempts (qid={qid}): {e!r}")
                return None, n_questions, cfg.max_retries
            await asyncio.sleep(min(2.0 ** attempt, MAX_BACKOFF_SECONDS))

    # Unreachable while max_retries >= 1 (enforced in __post_init__); kept so the function has a
    # total return type rather than falling off the end as None.
    return None, n_questions, 0


# ==============================================================================
#                        AGGREGATION: ONE CONVERSATION
# ==============================================================================


async def score_conversation(client,
                             cfg: OracleConfig,
                             primitives: AsyncPrimitives,
                             conversation_text: str) -> dict:
    """Grade one conversation on every configured rubric and reduce it to one reward.

    Args:
        client: Accepted for API stability but re-resolved internally -- the effective client is
            ``make_client(cfg.binding)`` on the RUNNING loop, because a client object that crossed
            event loops poisons its first calls (see ``roles.make_client``).
        cfg: The oracle configuration; ``cfg.questionnaire_ids`` selects the rubrics.
        primitives: Supplies the oracle semaphore.
        conversation_text: The complete transcript to grade.

    Returns:
        ``{"score": float|None, "sub_scores": {qid_str: mean}|None, "success": bool,
        "attempts": int}``.

        * ``score`` -- the **unweighted mean across questionnaires**, or ``None`` if any single
          rubric failed.
        * ``sub_scores`` -- per-rubric means, keyed by the questionnaire id as a **string** (the
          shape ``core/recorder.py`` writes to ``generations.jsonl``). Partial results are kept on
          failure for diagnosis, so a non-``None`` ``sub_scores`` with ``success=False`` is normal.
        * ``attempts`` -- total oracle HTTP calls made for this conversation, retries included.
          On a clean pass this equals ``len(cfg.questionnaire_ids)``; the excess over that is the
          retry count.

    Notes:
        **Equal weight per questionnaire is deliberate.** Under the default ``(1, 2)``, Q1's 5
        items and Q2's 17 items each contribute half the reward, so one Q1 item is worth 3.4 Q2
        items. That is Exp1's convention and it is preserved on purpose: the alternative
        (pooling all 22 items) is a different reward, and changing it would silently move the
        score axis out from under every comparison. If a future arm wants item-pooled scoring it
        needs a new rubric tag, not a quiet edit here.

        **One failed rubric zeroes the whole candidate -- it does not average the survivors.**
        Falling back to "score with whatever came back" would mean the reward's *definition*
        changes based on which calls happened to fail, and it fails soft: a grader that struggles
        with exactly one rubric produces a run whose reward is quietly a different quantity.
        ``None`` is the loud option: it means "not graded", never "graded badly". ⚠ It must NOT
        reach TRL -- the pinned trl 1.4.0 maps ``None`` to NaN and then ``nansum``s it to 0.0,
        i.e. optimises it as the worst possible completion. ``core.reward.rewards_for_trl``
        substitutes the candidate's group mean before the vector leaves the reward fn, and a
        batch with too many failures still trips ``min_success_ratio`` in ``core/reward.py``.

        Rubrics are scored **sequentially** for one conversation, matching Exp3. Concurrency comes
        from scoring many conversations at once (:func:`score_conversations_batch`), which keeps
        the oracle semaphore the single honest bound on in-flight calls.
    """
    # Re-resolve on the RUNNING loop (make_client is loop-keyed): a client built on another
    # loop -- the notebook's setup cell, a previous run_async pass, TRL's own reward loop --
    # carries keep-alive connections that poison calls here with APIConnectionError.
    client = make_client(cfg.binding)

    rewards: List[float] = []
    sub_scores: Dict[str, float] = {}
    attempts_total = 0

    for qid in cfg.questionnaire_ids:
        data, _n_questions, attempts = await get_evaluation_json(
            client, cfg, primitives, conversation_text, int(qid)
        )
        attempts_total += attempts
        if data is None:
            return {
                "score": None,
                "sub_scores": (sub_scores or None),
                "success": False,
                "attempts": attempts_total,
            }
        mean_score = float(data["mean_score"])
        sub_scores[str(int(qid))] = mean_score
        rewards.append(mean_score)

    return {
        "score": float(fmean(rewards)),
        "sub_scores": sub_scores,
        "success": True,
        "attempts": attempts_total,
    }


# ==============================================================================
#                          AGGREGATION: A BATCH
# ==============================================================================


async def score_conversations_batch(client,
                                    cfg: OracleConfig,
                                    primitives: AsyncPrimitives,
                                    texts: Sequence[str],
                                    *,
                                    progress_every: int = 0) -> List[dict]:
    """Grade many conversations concurrently; one result dict per input, in input order.

    Args:
        client: Async client built from ``cfg.binding``.
        cfg: The oracle configuration.
        primitives: Supplies the oracle semaphore that bounds in-flight calls.
        texts: Complete transcripts to grade.
        progress_every: Print a line every N completed conversations; 0 (default) is silent.

    Returns:
        A list of :func:`score_conversation` results, index-aligned with *texts*. Order is
        guaranteed by ``asyncio.gather``, which is what lets the reward function zip results back
        onto TRL's completions without carrying an id.

    Notes:
        **Do NOT wrap this gather in the oracle semaphore.** The bound already lives one level
        down, around each individual request in :func:`get_evaluation_json`. Adding an outer
        acquire of the SAME semaphore is a genuine deadlock, not merely redundant: once
        ``max_concurrency`` conversations each hold a permit and then await a permit for their
        first questionnaire call, no permit can ever be released and the batch hangs forever with
        no error. This is the single most tempting "obvious fix" in the module.

        ``return_exceptions`` is deliberately off. :func:`score_conversation` already converts
        every grading failure into ``success=False``, so anything that escapes is a real bug and
        should abort the step loudly rather than be smuggled into the results list as an exception
        object that later fails an arithmetic comparison somewhere unrelated.
    """
    if not texts:
        return []

    done = {"n": 0}
    step = int(progress_every)

    async def _one(text: str) -> dict:
        result = await score_conversation(client, cfg, primitives, text)
        done["n"] += 1
        if step > 0 and done["n"] % step == 0:
            print(f"    [oracle] scored {done['n']}/{len(texts)}")
        return result

    return list(await asyncio.gather(*(_one(t) for t in texts)))


def batch_success_ratio(results: Sequence[dict]) -> float:
    """Fraction of *results* whose grading succeeded, in ``[0, 1]``.

    Args:
        results: What :func:`score_conversations_batch` returned.

    Returns:
        ``successes / len(results)``, and **1.0 for an empty batch**.

    Notes:
        The empty case returns 1.0, not 0.0, because the caller compares this against
        ``cfg.min_success_ratio`` and aborts training below it: an empty batch carries no evidence
        of failure, and returning 0.0 would kill a run over a no-op call. (Exp3 returned 0.0 here
        and guarded every comparison with a separate ``total > 0`` check -- one guard that had to
        be remembered at each call site. Folding it into the helper removes the chance of
        forgetting it.)

        A candidate whose completion was degenerate never reaches the oracle, so it is not counted
        as a failure here. That is correct: it is a policy failure, handled by
        :data:`REWARD_FLOOR` in ``core/reward.py``, and counting it against the GRADER would let a
        badly-behaved policy trip the abort threshold and stop a run that is working exactly as
        designed.
    """
    if not results:
        return 1.0
    n_ok = sum(1 for r in results if r.get("success"))
    return n_ok / len(results)
