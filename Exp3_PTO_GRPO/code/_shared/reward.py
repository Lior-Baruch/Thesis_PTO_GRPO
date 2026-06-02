"""
reward.py — Look-ahead simulation + oracle-based reward function.

These were two files (``lookahead.py`` + ``oracle_reward.py``) sharing one
concern: scoring a candidate therapist completion via either (a) immediate
oracle eval, or (b) ``K`` simulated extra turns followed by oracle eval. They
move together and share dataclasses, so they share a module now.

Layout (top → bottom):
- Look-ahead transcript parsing (round-trip with :func:`convs.format_conversation_for_oracle`).
- Single-sample therapist GPU generation, serialized via an ``asyncio.Lock``.
- :func:`simulate_lookahead_single` / :func:`simulate_lookahead_batch` — K extra turns after a completion.
- :class:`OracleConfig` / :class:`LookaheadConfig` / :class:`OracleAsyncPrimitives` — knobs and loop-local async primitives.
- :func:`get_evaluation_json` — one oracle call (one questionnaire) with retry.
- :func:`make_reward_fn` — the per-iteration reward closure handed to TRL.
"""

import re
import gc
import time
import asyncio
import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import torch

from questionnaires import get_prompt_eval_questionnaire
from .convs import (
    handle_session_end,
    generate_patient_response_async,
    generate_patient_responses_batch,
    generate_therapist_responses_batch,
)


# Pattern: [PATIENT]: or [THERAPIST]: at the start of a transcript segment.
_TRANSCRIPT_LINE_RE = re.compile(r"^\[(PATIENT|THERAPIST)\]:\s*(.*)$", re.DOTALL)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     TRANSCRIPT PARSING                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _parse_transcript_to_messages(
    transcript: str,
    system_prompt_therapist: str,
    system_prompt_patient: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Parse a plain-text transcript back into message lists for both assistants.

    Reverse of :func:`convs.format_conversation_for_oracle`.

    The transcript uses ``\\n\\n`` between serialized turns and
    ``[PATIENT]:`` / ``[THERAPIST]:`` labels. Since content can also contain
    ``\\n\\n``, unlabeled split fragments are treated as continuations of the
    previous labeled turn.

    Returns:
        ``(messages_Therapist_assist, messages_Patient_assist)`` —
        therapist=assistant / patient=user in the first list,
        patient=assistant / therapist=user in the second.
    """
    messages_therapist = [{"role": "system", "content": system_prompt_therapist}]
    messages_patient = [{"role": "system", "content": system_prompt_patient}]

    segments = [s.strip() for s in transcript.split("\n\n") if s.strip()]

    current_label: Optional[str] = None
    current_fragments: List[str] = []

    def _flush_current_turn() -> None:
        nonlocal current_label, current_fragments
        if current_label is None:
            return

        content = "\n\n".join(fragment for fragment in current_fragments if fragment).strip()
        if not content:
            current_label = None
            current_fragments = []
            return

        if current_label == "THERAPIST":
            messages_therapist.append({"role": "assistant", "content": content})
            messages_patient.append({"role": "user", "content": content})
        else:  # PATIENT
            messages_therapist.append({"role": "user", "content": content})
            messages_patient.append({"role": "assistant", "content": content})

        current_label = None
        current_fragments = []

    for segment in segments:
        m = _TRANSCRIPT_LINE_RE.match(segment)
        if m is not None:
            _flush_current_turn()
            current_label = m.group(1)  # "PATIENT" or "THERAPIST"
            current_fragments = [m.group(2).strip()]
            continue

        if current_label is None:
            raise ValueError(
                "Cannot parse transcript segment without a preceding role label: "
                f"{segment!r}"
            )

        # Continuation fragment from embedded "\n\n" inside a turn.
        current_fragments.append(segment)

    _flush_current_turn()
    return messages_therapist, messages_patient


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  THERAPIST SINGLE-SAMPLE GENERATION                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def _generate_therapist_single_async(
    messages_Therapist_assist: List[Dict],
    therapist_model,
    therapist_tokenizer,
    max_tokens: int,
    temperature: float,
    max_input_tokens: int,
    stop_strings: List[str],
    gpu_lock: asyncio.Lock,
) -> Optional[str]:
    """Generate a single therapist response on GPU, serialized via *gpu_lock*.

    Uses ``torch.inference_mode()`` and temporarily enables KV-cache so that
    generation works even when the model is in training mode. The caller's
    ``model.train()`` / ``use_cache`` state is restored in a ``finally`` block.

    LEGACY (batch-of-1): the live look-ahead path now uses the lock-step batched
    rollout in :func:`simulate_lookahead_batch` (which generates all active sims
    in one padded ``model.generate``). This function + :func:`simulate_lookahead_single`
    are kept only as the ground-truth semantics reference and the comparison
    oracle for the batched-vs-serial equivalence check. Not on the hot path.
    """

    def _sync_generate() -> Optional[str]:
        prompt = therapist_tokenizer.apply_chat_template(
            messages_Therapist_assist, tokenize=False, add_generation_prompt=True,
        )
        encoded = therapist_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=max_input_tokens,
        )
        input_ids = encoded["input_ids"].to(therapist_model.device)
        attention_mask = encoded["attention_mask"].to(therapist_model.device)

        old_use_cache = therapist_model.config.use_cache
        was_training = therapist_model.training
        therapist_model.config.use_cache = True
        therapist_model.eval()
        try:
            with torch.inference_mode():
                outputs = therapist_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=max_tokens,
                    pad_token_id=therapist_tokenizer.eos_token_id,
                    eos_token_id=therapist_tokenizer.eos_token_id,
                    temperature=temperature,
                    num_return_sequences=1,
                    stop_strings=stop_strings,
                    tokenizer=therapist_tokenizer,
                )
        finally:
            therapist_model.config.use_cache = old_use_cache
            if was_training:
                therapist_model.train()

        new_tokens = outputs[0][input_ids.shape[1]:]
        decoded = therapist_tokenizer.decode(new_tokens, skip_special_tokens=True)
        cleaned = decoded.split("<|im_end|>")[0].strip()

        del encoded, outputs, input_ids, attention_mask
        return cleaned or None

    try:
        async with gpu_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _sync_generate)
    except torch.OutOfMemoryError as e:
        print(f"  Look-ahead OOM: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            print(f"  Look-ahead runtime OOM: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            return None
        raise


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              BATCHED THERAPIST GENERATION (OOM-resilient)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _therapist_generate_chunked(
    therapist_model,
    therapist_tokenizer,
    batch_messages: List[List[Dict]],
    max_tokens: int,
    temperature: float,
    max_input_tokens: int,
    stop_strings: Optional[List[str]],
    start_sub_batch: int,
) -> Tuple[List[Optional[str]], int, int]:
    """Generate one therapist response per message-list, with OOM-driven halving.

    Sync helper (call via ``run_in_executor`` under ``gpu_lock``). Wraps
    :func:`convs.generate_therapist_responses_batch`, which returns
    ``(responses|None, error_type)`` and never raises on OOM (it cleans the CUDA
    cache and returns ``"oom"``). Inputs are processed in chunks of ``sb``
    (starting at ``start_sub_batch``):

    - success → place responses at their indices.
    - ``"oom"`` → if ``sb == 1`` freeze that single item (``None``) and advance;
      else halve ``sb`` and retry the same chunk. The reduced ``sb`` is **sticky**
      (returned so the caller reuses it for later steps — no re-paying OOM cost).
    - ``"runtime_error"`` (non-OOM) → halving won't help, so freeze the chunk's
      items (``None``) and advance. (Diverges from ``conversation_loop_batch``,
      which aborts the whole batch; for a reward computation, scoring a shorter
      transcript beats killing a GRPO step on a transient hiccup.)

    Even a sub-batch=1 OOM is non-fatal: that item returns ``None`` and the sim
    freezes on its current transcript.

    Returns ``(responses, final_sub_batch, n_generate_calls)`` with
    ``responses`` order-aligned to ``batch_messages``.
    """
    n = len(batch_messages)
    responses: List[Optional[str]] = [None] * n
    sb = max(1, start_sub_batch)
    n_calls = 0

    i = 0
    while i < n:
        chunk = batch_messages[i:i + sb]
        resp, error_type = generate_therapist_responses_batch(
            therapist_model, therapist_tokenizer, chunk,
            max_tokens, temperature,
            max_input_tokens=max_input_tokens, stop_strings=stop_strings,
        )
        n_calls += 1

        if error_type is None:
            for j, r in enumerate(resp):
                responses[i + j] = r
            i += len(chunk)
        elif error_type == "oom":
            if sb == 1:
                # A single sequence still OOMs — freeze it and move on.
                responses[i] = None
                i += 1
            else:
                sb = max(1, sb // 2)  # sticky: smaller sb persists for later chunks
                # don't advance i — retry the same start at the smaller sb
        else:  # "runtime_error" — freeze this chunk, advance.
            for j in range(len(chunk)):
                responses[i + j] = None
            i += len(chunk)

    return responses, sb, n_calls


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   LOOK-AHEAD SIMULATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class _LookaheadSim:
    """Mutable per-completion state for the lock-step batched look-ahead rollout."""
    msgs_therapist: List[Dict]
    msgs_patient: List[Dict]
    extended_transcript: str
    active: bool = True


async def simulate_lookahead_single(
    transcript: str,
    completion: str,
    system_prompt_therapist: str,
    system_prompt_patient: str,
    therapist_model,
    therapist_tokenizer,
    client,
    patient_model_id: str,
    lookahead_k: int,
    temperature_therapist: float,
    temperature_patient: float,
    max_tokens: int,
    max_input_tokens: int,
    stop_strings: List[str],
    patient_sem: asyncio.Semaphore,
    gpu_lock: asyncio.Lock,
    patient_max_retries: int = 3,
    patient_backoff_seconds: float = 1.0,
) -> str:
    """Simulate *lookahead_k* additional conversation turns after *completion*.

    Alternates patient (OpenAI API) and therapist (local GPU) turns.
    Returns the full extended transcript (original + completion + look-ahead
    turns) as plain text suitable for oracle evaluation.

    LEGACY (serial, batch-of-1): kept as the ground-truth semantics reference and
    the comparison oracle for the batched-vs-serial equivalence check. The live
    path is :func:`simulate_lookahead_batch` (lock-step batched). Not on the hot
    path — see the module/notebook verification cell.
    """
    try:
        msgs_therapist, msgs_patient = _parse_transcript_to_messages(
            transcript, system_prompt_therapist, system_prompt_patient,
        )
    except ValueError as e:
        preview = transcript[:280] + ("..." if len(transcript) > 280 else "")
        raise ValueError(
            "Failed to parse transcript for look-ahead simulation "
            f"(length={len(transcript)} chars). Preview: {preview!r}"
        ) from e

    # Append the completion (therapist response) to both lists
    msgs_therapist.append({"role": "assistant", "content": completion})
    msgs_patient.append({"role": "user", "content": completion})

    extended_transcript = f"{transcript}\n\n[THERAPIST]: {completion}"

    # After the completion (therapist), next speaker is patient
    speaker_role = "patient"

    for _ in range(lookahead_k):
        if speaker_role == "patient":
            try:
                response = await generate_patient_response_async(
                    client, patient_model_id, msgs_patient, max_tokens,
                    temperature_patient, patient_sem,
                    max_retries=patient_max_retries,
                    backoff_seconds=patient_backoff_seconds,
                )
            except RuntimeError:
                break  # API failure — return what we have

            if response is None:
                break

            if "SESSION ENDED" in response.upper():
                try:
                    _, _, cleaned = handle_session_end(response, "patient")
                    if cleaned:
                        extended_transcript += f"\n\n[PATIENT]: {cleaned}"
                except ValueError:
                    pass
                break

            msgs_therapist.append({"role": "user", "content": response})
            msgs_patient.append({"role": "assistant", "content": response})
            extended_transcript += f"\n\n[PATIENT]: {response}"
            speaker_role = "therapist"

        else:
            response = await _generate_therapist_single_async(
                msgs_therapist, therapist_model, therapist_tokenizer,
                max_tokens, temperature_therapist, max_input_tokens,
                stop_strings, gpu_lock,
            )

            if response is None:
                break

            if "SESSION ENDED" in response.upper():
                try:
                    _, _, cleaned = handle_session_end(response, "therapist")
                    if cleaned:
                        extended_transcript += f"\n\n[THERAPIST]: {cleaned}"
                except ValueError:
                    pass
                break

            msgs_therapist.append({"role": "assistant", "content": response})
            msgs_patient.append({"role": "user", "content": response})
            extended_transcript += f"\n\n[THERAPIST]: {response}"
            speaker_role = "patient"

    return extended_transcript


async def simulate_lookahead_batch(
    transcripts: List[str],
    completions: List[str],
    system_prompt_therapist: str,
    system_prompts_patient: List[str],
    therapist_model,
    therapist_tokenizer,
    client,
    patient_model_id: str,
    lookahead_k: int,
    temperature_therapist: float,
    temperature_patient: float,
    max_tokens: int,
    max_input_tokens: int,
    stop_strings: List[str],
    patient_sem: asyncio.Semaphore,
    gpu_lock: asyncio.Lock,
    patient_max_retries: int = 3,
    patient_backoff_seconds: float = 1.0,
    sub_batch_size: Optional[int] = None,
    telemetry: Optional[dict] = None,
) -> List[str]:
    """Lock-step batched look-ahead for a batch of (transcript, completion) pairs.

    All sims advance in unison (patient → therapist → patient → …), so each
    therapist turn is **one padded batched** ``model.generate`` over the active
    sims rather than B serial batch-of-1 calls. Collapses ~B·K serial
    generations into ~K batched ones. Semantics match the legacy serial
    :func:`simulate_lookahead_single` (statistically equivalent, not
    bit-identical — sampling RNG differs).

    Speaker is a pure function of the step index (even = patient, odd =
    therapist): every sim is constructed here at the same phase, and the only
    way to leave the cadence is to go inactive (dropped from the active set), so
    — unlike :func:`convs.conversation_loop_batch` — there is no speaker-desync
    case to recover from.

    Safety: patient API calls run concurrently (bounded by *patient_sem*).
    Therapist generation holds *gpu_lock* per step (never across the patient
    ``await``) and toggles ``eval()`` / ``use_cache=True`` for the duration,
    restoring the caller's ``train()`` / ``use_cache`` in a ``finally`` — this
    runs during a live GRPO step with the policy in ``train()``. OOM is handled
    by sub-batch halving in :func:`_therapist_generate_chunked`.

    A sim is **frozen** (kept at its current transcript, removed from later
    steps) when its patient call fails, either side emits ``SESSION ENDED``, or
    therapist generation returns ``None`` (OOM/runtime error). A transcript that
    fails to parse is also frozen on its seed text rather than aborting the whole
    call — deliberately more robust than the serial path, which propagates the
    ``ValueError``.

    Args:
        sub_batch_size: cap on the therapist generate batch (None = all active
            sims at once). Halved automatically on OOM and kept sticky.
        telemetry: if a dict is passed, it is populated with ``gpu_calls`` and
            ``sub_batch`` (the final, possibly-halved size) for logging. Callers
            that don't need it (e.g. PTO branch scoring) omit it to stay quiet.

    Returns:
        ``List[str]`` of extended transcripts, in input order.
    """
    # Build per-sim state. A transcript that can't be parsed freezes that sim on
    # its seed transcript rather than aborting the whole reward call.
    sims: List[_LookaheadSim] = []
    for transcript, completion, sp_patient in zip(
        transcripts, completions, system_prompts_patient
    ):
        seed_transcript = f"{transcript}\n\n[THERAPIST]: {completion}"
        try:
            msgs_therapist, msgs_patient = _parse_transcript_to_messages(
                transcript, system_prompt_therapist, sp_patient,
            )
            msgs_therapist.append({"role": "assistant", "content": completion})
            msgs_patient.append({"role": "user", "content": completion})
            sims.append(_LookaheadSim(msgs_therapist, msgs_patient, seed_transcript, True))
        except ValueError as e:
            preview = transcript[:280] + ("..." if len(transcript) > 280 else "")
            print(
                f"  ⚠ Look-ahead transcript parse failed (length={len(transcript)} "
                f"chars); freezing this sim on its seed transcript. Preview: "
                f"{preview!r} ({e})"
            )
            sims.append(_LookaheadSim([], [], seed_transcript, False))

    loop = asyncio.get_event_loop()
    sb = sub_batch_size  # mutable; halving in the chunked helper persists across steps
    total_gpu_calls = 0

    # After the completion (a therapist turn), the next speaker is the patient.
    for step in range(lookahead_k):
        active = [s for s in sims if s.active]
        if not active:
            break

        speaker_role = "patient" if step % 2 == 0 else "therapist"

        if speaker_role == "patient":
            responses = await generate_patient_responses_batch(
                client, patient_model_id,
                [s.msgs_patient for s in active],
                max_tokens, temperature_patient, patient_sem,
                patient_max_retries, patient_backoff_seconds,
            )
            for sim, resp in zip(active, responses):
                if isinstance(resp, BaseException) or resp is None:
                    sim.active = False
                    continue
                if "SESSION ENDED" in resp.upper():
                    try:
                        _, _, cleaned = handle_session_end(resp, "patient")
                        if cleaned:
                            sim.extended_transcript += f"\n\n[PATIENT]: {cleaned}"
                    except ValueError:
                        pass
                    sim.active = False
                    continue
                sim.msgs_therapist.append({"role": "user", "content": resp})
                sim.msgs_patient.append({"role": "assistant", "content": resp})
                sim.extended_transcript += f"\n\n[PATIENT]: {resp}"
        else:  # therapist turn — one batched GPU generate under gpu_lock + eval toggle
            start_sb = sb if sb is not None else len(active)
            async with gpu_lock:
                old_use_cache = therapist_model.config.use_cache
                was_training = therapist_model.training
                therapist_model.config.use_cache = True
                therapist_model.eval()
                try:
                    responses, sb, n_calls = await loop.run_in_executor(
                        None, _therapist_generate_chunked,
                        therapist_model, therapist_tokenizer,
                        [s.msgs_therapist for s in active],
                        max_tokens, temperature_therapist, max_input_tokens,
                        stop_strings, start_sb,
                    )
                finally:
                    therapist_model.config.use_cache = old_use_cache
                    if was_training:
                        therapist_model.train()
            total_gpu_calls += n_calls

            for sim, resp in zip(active, responses):
                if resp is None:
                    sim.active = False
                    continue
                if "SESSION ENDED" in resp.upper():
                    try:
                        _, _, cleaned = handle_session_end(resp, "therapist")
                        if cleaned:
                            sim.extended_transcript += f"\n\n[THERAPIST]: {cleaned}"
                    except ValueError:
                        pass
                    sim.active = False
                    continue
                sim.msgs_therapist.append({"role": "assistant", "content": resp})
                sim.msgs_patient.append({"role": "user", "content": resp})
                sim.extended_transcript += f"\n\n[THERAPIST]: {resp}"

    if telemetry is not None:
        telemetry["gpu_calls"] = total_gpu_calls
        telemetry["sub_batch"] = sb

    return [s.extended_transcript for s in sims]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIG DATACLASSES                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class OracleConfig:
    """Oracle (questionnaire-grading LLM) settings."""
    model_id: str
    request_timeout: float
    max_retries: int
    eval_temperature: float
    max_concurrency: int
    min_success_ratio: float


@dataclass(frozen=True)
class LookaheadConfig:
    """Settings for look-ahead reward (K extra simulated turns before scoring)."""
    k: int                           # 0 disables look-ahead
    temperature_therapist: float
    temperature_patient: float
    max_tokens: int
    max_input_tokens: int
    patient_model_id: str
    patient_api_concurrency: int
    patient_api_max_retries: int
    patient_api_backoff_seconds: float
    stop_strings: Optional[List[str]] = None
    # Sub-batch size for the batched look-ahead therapist generation. None =
    # generate all active sims in one padded ``model.generate`` call (largest
    # batch, fastest). Set an int to cap GPU memory; the batched rollout halves
    # it automatically on OOM. See :func:`simulate_lookahead_batch`.
    lookahead_sub_batch_size: Optional[int] = None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  LOOP-LOCAL ASYNC PRIMITIVES                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class OracleAsyncPrimitives:
    """Lazy semaphores/lock keyed by the currently-running event loop.

    Python 3.10+ raises ``RuntimeError`` if you reuse an asyncio primitive
    across event loops. TRL may launch its own loop separate from the one the
    notebook owns, so we create primitives the first time they're requested
    inside a given loop and discard stale entries.
    """

    def __init__(self, oracle_concurrency: int, lookahead_patient_concurrency: int):
        self._oracle_concurrency = oracle_concurrency
        self._lookahead_patient_concurrency = lookahead_patient_concurrency
        self._cache: dict = {}

    def _get(self, name: str, factory: Callable):
        loop = asyncio.get_running_loop()
        key = (name, id(loop))
        if key not in self._cache:
            stale = [k for k in self._cache if k[0] == name and k[1] != id(loop)]
            for k in stale:
                del self._cache[k]
            self._cache[key] = factory()
        return self._cache[key]

    def oracle_sem(self) -> asyncio.Semaphore:
        return self._get("oracle_sem", lambda: asyncio.Semaphore(self._oracle_concurrency))

    def gpu_lock(self) -> asyncio.Lock:
        return self._get("gpu_lock", asyncio.Lock)

    def lookahead_patient_sem(self) -> asyncio.Semaphore:
        return self._get(
            "lookahead_patient_sem",
            lambda: asyncio.Semaphore(self._lookahead_patient_concurrency),
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       ORACLE SCORING                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# Errors that aren't worth retrying (programming errors, not transient).
_NON_RETRYABLE = (KeyError, TypeError)


async def get_evaluation_json(
    client,
    oracle_cfg: OracleConfig,
    primitives: OracleAsyncPrimitives,
    transcript: str,
    completion: str,
    questionnaire_id: int,
    full_conversation_override: Optional[str] = None,
):
    """Score one (conversation, questionnaire) with the oracle.

    Returns ``(data, n_questions)`` where ``data`` is the parsed JSON response
    augmented with ``mean_score``, or ``(None, n_questions)`` on failure.

    Args:
        full_conversation_override: If provided, use this as the full
            conversation text instead of building it from ``transcript`` +
            ``completion``. Used by look-ahead reward to pass the extended
            transcript directly.
    """
    full_conversation = (
        full_conversation_override
        if full_conversation_override is not None
        else f"{transcript}\n\n[THERAPIST]: {completion}"
    )
    eval_dict = get_prompt_eval_questionnaire(
        questionnaire=questionnaire_id,
        conversation=full_conversation,
    )
    eval_prompt = eval_dict["prompt"]
    n_questions = int(eval_dict["questions_count"])
    schema = eval_dict["schema"]
    scale_min = eval_dict["scale_min"]
    scale_max = eval_dict["scale_max"]

    for attempt in range(oracle_cfg.max_retries):
        try:
            async with primitives.oracle_sem():
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=oracle_cfg.model_id,
                        messages=[{"role": "user", "content": eval_prompt}],
                        temperature=oracle_cfg.eval_temperature,
                        max_tokens=256,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": f"questionnaire_{questionnaire_id}_evaluation",
                                "schema": schema,
                                "strict": True,
                            },
                        },
                    ),
                    timeout=oracle_cfg.request_timeout,
                )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty oracle response")

            data = json.loads(content)

            if data.get("questionnaire_id") != questionnaire_id:
                raise ValueError(f"Wrong questionnaire_id: {data.get('questionnaire_id')}")

            scores = data.get("scores", [])
            if len(scores) != n_questions:
                raise ValueError(f"Expected {n_questions} scores, got {len(scores)}")
            if any(not isinstance(s, int) or s < scale_min or s > scale_max for s in scores):
                raise ValueError(f"Invalid score values (expected {scale_min}-{scale_max})")

            data["mean_score"] = float(np.mean(scores))
            return data, n_questions

        except _NON_RETRYABLE as e:
            print(f"  ⚠ Oracle non-retryable error (qid={questionnaire_id}): {e}")
            return None, n_questions

        except Exception as e:
            if attempt >= oracle_cfg.max_retries - 1:
                print(f"  ⚠ Oracle failed after {oracle_cfg.max_retries} attempts (qid={questionnaire_id}): {e}")
                return None, n_questions
            await asyncio.sleep(2 ** attempt)

    return None, n_questions


async def _process_single_sample(
    client,
    oracle_cfg: OracleConfig,
    primitives: OracleAsyncPrimitives,
    questionnaire_ids: Sequence[int],
    stats: dict,
    idx: int,
    transcript: str,
    completion: str,
    full_conversation_override: Optional[str] = None,
) -> Optional[float]:
    """Compute reward for one sample by averaging scores across questionnaires.

    Returns None if any oracle call fails (TRL converts None → NaN and skips).
    Tracks per-batch success/fail in ``stats``.
    """
    if full_conversation_override is None and not completion.strip():
        stats["fail"] += 1
        return None

    rewards: List[float] = []
    for qid in questionnaire_ids:
        data, _ = await get_evaluation_json(
            client, oracle_cfg, primitives,
            transcript=transcript,
            completion=completion,
            questionnaire_id=int(qid),
            full_conversation_override=full_conversation_override,
        )
        if data is None:
            stats["fail"] += 1
            return None
        rewards.append(float(data["mean_score"]))

    if (idx + 1) % 10 == 0:
        print(f"    Evaluated sample {idx + 1}")

    stats["success"] += 1
    return float(np.mean(rewards))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       REWARD FN FACTORY                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def make_reward_fn(
    *,
    client,
    policy,
    tokenizer,
    therapist_system_prompt: str,
    questionnaire_ids: Sequence[int],
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
) -> Callable:
    """Build the async reward function for TRL v0.28+.

    Call this once per training iteration with the *current* policy. TRL
    awaits the returned coroutine natively (no sync wrapper needed).

    The reward fn closes over ``policy`` so look-ahead simulation uses the
    iteration's current weights.
    """
    stats = {"success": 0, "fail": 0}

    async def reward_fn(prompts, completions, transcript, **kwargs):
        stats["success"] = 0
        stats["fail"] = 0

        if lookahead_cfg.k > 0:
            patient_system_prompt = kwargs.get("patient_system_prompt", [""] * len(prompts))
            # Guard: look-ahead simulates patient turns conditioned on this prompt.
            # If it's missing/empty the patient is degenerate — warn loudly rather
            # than silently scoring against an empty system prompt.
            if all(not (sp or "").strip() for sp in patient_system_prompt):
                print(
                    "    ⚠ Look-ahead k>0 but all patient_system_prompt are empty — "
                    "simulating patient turns against an empty system prompt "
                    "(degenerate patient). Check that the dataset carries "
                    "'patient_system_prompt'."
                )

            la_start = time.time()
            la_telemetry: dict = {}
            extended_transcripts = await simulate_lookahead_batch(
                transcripts=list(transcript),
                completions=list(completions),
                system_prompt_therapist=therapist_system_prompt,
                system_prompts_patient=list(patient_system_prompt),
                therapist_model=policy,
                therapist_tokenizer=tokenizer,
                client=client,
                patient_model_id=lookahead_cfg.patient_model_id,
                lookahead_k=lookahead_cfg.k,
                temperature_therapist=lookahead_cfg.temperature_therapist,
                temperature_patient=lookahead_cfg.temperature_patient,
                max_tokens=lookahead_cfg.max_tokens,
                max_input_tokens=lookahead_cfg.max_input_tokens,
                stop_strings=lookahead_cfg.stop_strings,
                patient_sem=primitives.lookahead_patient_sem(),
                gpu_lock=primitives.gpu_lock(),
                patient_max_retries=lookahead_cfg.patient_api_max_retries,
                patient_backoff_seconds=lookahead_cfg.patient_api_backoff_seconds,
                sub_batch_size=lookahead_cfg.lookahead_sub_batch_size,
                telemetry=la_telemetry,
            )
            la_time = time.time() - la_start

            # Telemetry: realized look-ahead turns per sim (approximate, by counting
            # role labels added beyond the original transcript + the completion). A
            # sim that ran fewer than K turns ended early (SESSION ENDED / API fail).
            # This is the serial-path baseline for the deferred batched rewrite.
            def _count_labels(s: str) -> int:
                return s.count("[PATIENT]:") + s.count("[THERAPIST]:")

            la_turns = [
                max(0, _count_labels(ext) - _count_labels(orig) - 1)
                for ext, orig in zip(extended_transcripts, list(transcript))
            ]
            avg_la = float(np.mean(la_turns)) if la_turns else 0.0
            n_early = sum(1 for n in la_turns if n < lookahead_cfg.k)
            print(
                f"    Look-ahead: {len(extended_transcripts)} sims × K={lookahead_cfg.k} "
                f"in {la_time:.1f}s (avg {avg_la:.1f} turns realized, "
                f"{n_early} ended early; batched, "
                f"{la_telemetry.get('gpu_calls', '?')} GPU calls, "
                f"sub_batch={la_telemetry.get('sub_batch', '?')})"
            )

            tasks = [
                _process_single_sample(
                    client, oracle_cfg, primitives, questionnaire_ids, stats,
                    idx, "", "", full_conversation_override=ext_t,
                )
                for idx, ext_t in enumerate(extended_transcripts)
            ]
        else:
            tasks = [
                _process_single_sample(
                    client, oracle_cfg, primitives, questionnaire_ids, stats,
                    idx, t, c,
                )
                for idx, (t, c) in enumerate(zip(transcript, completions))
            ]

        results = await asyncio.gather(*tasks)

        total = stats["success"] + stats["fail"]
        success_rate = stats["success"] / total if total > 0 else 0.0
        n_none = sum(1 for r in results if r is None)
        print(
            f"    Oracle batch: {stats['success']}/{total} succeeded "
            f"({success_rate:.0%}), {n_none} rewards → None"
        )

        if total > 0 and success_rate < oracle_cfg.min_success_ratio:
            raise RuntimeError(
                f"Oracle success rate {success_rate:.1%} below threshold "
                f"{oracle_cfg.min_success_ratio:.0%} ({stats['fail']}/{total} failed). "
                f"Aborting to prevent training on biased subset."
            )

        return results

    return reward_fn
