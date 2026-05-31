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
import asyncio
import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import torch

from questionnaires import get_prompt_eval_questionnaire
from .convs import handle_session_end, generate_patient_response_async


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
# ║                   LOOK-AHEAD SIMULATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


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
) -> List[str]:
    """Run look-ahead simulations for a batch of (transcript, completion) pairs.

    Patient API calls run concurrently (bounded by *patient_sem*); GPU calls
    are serialized by *gpu_lock*.
    """
    tasks = [
        simulate_lookahead_single(
            transcript=t, completion=c,
            system_prompt_therapist=system_prompt_therapist,
            system_prompt_patient=sp,
            therapist_model=therapist_model,
            therapist_tokenizer=therapist_tokenizer,
            client=client,
            patient_model_id=patient_model_id,
            lookahead_k=lookahead_k,
            temperature_therapist=temperature_therapist,
            temperature_patient=temperature_patient,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            stop_strings=stop_strings,
            patient_sem=patient_sem,
            gpu_lock=gpu_lock,
            patient_max_retries=patient_max_retries,
            patient_backoff_seconds=patient_backoff_seconds,
        )
        for t, c, sp in zip(transcripts, completions, system_prompts_patient)
    ]
    return await asyncio.gather(*tasks)


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
