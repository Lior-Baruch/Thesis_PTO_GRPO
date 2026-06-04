"""
convs.py — Conversation lifecycle: state, generation, prompt extraction.

This module owns the full "conversation" mental model end-to-end. The three
historical sub-concerns (state/IO, generation, prompt extraction) share enough
machinery that splitting them across files was generating noise:

- **State + I/O** (``ConversationState``, CSV round-trip, init/update/session-end).
- **Generation** (async patient via OpenAI API, batched therapist on GPU,
  lock-step conv loop, batch retry, OOM recovery, resume-from-disk).
- **Prompt extraction** (per-turn training samples, token-budget truncation,
  MCL filter — drop slices shorter than ``min_conv_length`` total utterances).

The top-level entry points are :func:`generate_all_conversations` (produces
``ConversationState`` records) and :func:`extract_prompts_from_conversations`
(turns those records into per-turn GRPO/PTO training prompts).
"""

import os
import gc
import time
import asyncio
import textwrap
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pandas as pd
import torch


# Default stop-strings for therapist generation (ChatML end-of-turn marker).
_DEFAULT_STOP_STRINGS = ["<|im_end|>"]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            DATA CLASSES                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class ConversationState:
    """Mutable state for one permutation during batched synthesis.

    The object keeps two parallel representations:
    - ``conversation``: backward-compatible flat utterance list.
    - ``turns``: explicit role-tagged turns used by prompt extraction.

    ``messages_*_assist`` are per-assistant chat histories in each model's
    role convention and are cleared once a conversation ends to reduce memory.
    """

    permutation_index: int
    conversation: list  # Backward-compatible format: flat list of utterance strings
    messages_Patient_assist: list  # [{"role": "system"|"user"|"assistant", "content": str}]
    messages_Therapist_assist: list  # [{"role": "system"|"user"|"assistant", "content": str}]
    turns: list = field(default_factory=list)  # [{"role": "therapist"|"patient", "content": str}]
    next_speaker: str = "patient"  # "therapist" or "patient" — who speaks next
    session_ended_by: Optional[str] = None
    session_ended_explanation: Optional[str] = None
    is_active: bool = True
    failed: bool = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            CSV I/O                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _extract_unique_col(df: pd.DataFrame, col_name: str) -> Optional[str]:
    """Extract a single unique non-null value from a DataFrame column, or None."""
    if col_name not in df.columns:
        return None
    vals = df[col_name].dropna().unique()
    return str(vals[0]) if len(vals) > 0 else None


def load_conversation_from_csv(csv_path: str, permutation_index: int) -> ConversationState:
    """Load a single conversation from a CSV file.

    If a ``role`` column exists, roles are read explicitly.
    Otherwise roles are inferred by alternating turns (therapist first).
    """
    df = pd.read_csv(csv_path, keep_default_na=False)
    conversation = df["conversation"].tolist()

    if "role" in df.columns:
        turns = [
            {"role": str(r), "content": str(c)}
            for r, c in zip(df["role"], df["conversation"])
        ]
    else:
        turns = [
            {"role": "therapist" if j % 2 == 0 else "patient", "content": str(c)}
            for j, c in enumerate(conversation)
        ]

    return ConversationState(
        permutation_index=permutation_index,
        conversation=conversation,
        messages_Patient_assist=[],
        messages_Therapist_assist=[],
        turns=turns,
        session_ended_by=_extract_unique_col(df, "session_ended_by"),
        session_ended_explanation=_extract_unique_col(df, "session_ended_explanation"),
        is_active=False,
    )


def save_conversation_csv(state: ConversationState, save_dir: str) -> str:
    """Write one ConversationState to ``save_dir/conversation_<idx>.csv``.

    Roles are taken from ``state.turns`` when present, else inferred by
    alternating (therapist first). Returns the path written.
    """
    save_path = os.path.join(save_dir, f"conversation_{state.permutation_index}.csv")
    roles = (
        [t["role"] for t in state.turns]
        if state.turns
        else ["therapist" if j % 2 == 0 else "patient" for j in range(len(state.conversation))]
    )
    pd.DataFrame({
        "role": roles,
        "conversation": state.conversation,
        "session_ended_by": state.session_ended_by,
        "session_ended_explanation": state.session_ended_explanation,
    }).to_csv(save_path, index=False)
    return save_path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        TEXT UTILITIES                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def reconstruct_conversation_text(turns_or_utterances) -> str:
    """Convert role-tagged turns or utterance lists to a labeled transcript.

    Accepts:
        - List[Dict] with 'role' and 'content' keys (new format)
        - List[str] utterances (therapist-first alternating)
    """
    if turns_or_utterances and isinstance(turns_or_utterances[0], dict):
        labels = {"therapist": "THERAPIST", "patient": "PATIENT"}
        return "\n".join(
            f"[{labels[t['role']]}]: {t['content']}" for t in turns_or_utterances
        )
    return "\n".join(
        f"[{'THERAPIST' if i % 2 == 0 else 'PATIENT'}]: {utt}"
        for i, utt in enumerate(turns_or_utterances)
    )


def print_conversation(turns_or_utterances, max_width: int = 80) -> None:
    """Print the conversation with roles labeled as [THERAPIST] and [PATIENT].

    Accepts role-tagged turn dicts or utterance strings.
    """
    for i, item in enumerate(turns_or_utterances):
        if isinstance(item, dict):
            label = "[THERAPIST]" if item["role"] == "therapist" else "[PATIENT]"
            content = item["content"]
        else:
            label = "[THERAPIST]" if i % 2 == 0 else "[PATIENT]"
            content = item
        print(f"{label}: \n{textwrap.fill(content, width=max_width)} \n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     CONVERSATION INITIALIZATION                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def initialize_conversation(
    system_prompt_therapist: str,
    system_prompt_patient: str,
    therapist_init_utterance: str,
    patient_init_utterance: str,
    include_empty_init_user_message: bool = False,
):
    """Initialize conversation and message lists for both assistants.

    Args:
        include_empty_init_user_message: If True, include the initial user
            message (even if empty) before the therapist's first assistant
            message. Required by chat templates that expect
            system→user→assistant ordering. Set to False if the model template
            supports system→assistant directly.
    """
    conversation = [therapist_init_utterance]

    messages_Patient_assist = [
        {"role": "system", "content": system_prompt_patient},
        {"role": "user", "content": therapist_init_utterance},
    ]

    if include_empty_init_user_message or patient_init_utterance:
        messages_Therapist_assist = [
            {"role": "system", "content": system_prompt_therapist},
            {"role": "user", "content": patient_init_utterance},
            {"role": "assistant", "content": therapist_init_utterance},
        ]
    else:
        messages_Therapist_assist = [
            {"role": "system", "content": system_prompt_therapist},
            {"role": "assistant", "content": therapist_init_utterance},
        ]

    return conversation, messages_Patient_assist, messages_Therapist_assist


def update_conversation(
    conversation: list,
    messages_Patient_assist: list,
    messages_Therapist_assist: list,
    role_Patient: str,
    role_Therapist: str,
    response_content: str,
    turns: Optional[list] = None,
    speaker_role: Optional[str] = None,
) -> None:
    """Append a new response to conversation and message lists.

    Args:
        turns: If provided, also append a role-tagged turn dict.
        speaker_role: "therapist" or "patient" — required when ``turns`` is provided.
    """
    conversation.append(response_content)
    messages_Patient_assist.append({"role": role_Patient, "content": response_content})
    messages_Therapist_assist.append({"role": role_Therapist, "content": response_content})
    if turns is not None and speaker_role is not None:
        turns.append({"role": speaker_role, "content": response_content})


def handle_session_end(response_content: str, speaker_role: str):
    """Handle 'SESSION ENDED' keyword in response content.

    Returns ``(ended_by, ended_explanation, cleaned_response_content)``.
    Raises ``ValueError`` if the keyword is absent.
    """
    session_ended_keyword = "SESSION ENDED"
    idx = response_content.upper().find(session_ended_keyword)
    if idx == -1:
        raise ValueError("SESSION ENDED keyword not found in response content")

    ended_explanation = response_content[idx + len(session_ended_keyword):]
    cleaned = response_content[:idx]
    return speaker_role, ended_explanation, cleaned


def _process_session_response(
    state: ConversationState,
    response_content: str,
    speaker_role: str,
    role_Patient: str,
    role_Therapist: str,
) -> bool:
    """Apply one generated response to state and handle terminal marker.

    If ``SESSION ENDED`` appears, the conversation is marked inactive and the
    trailing explanation is stored. Any text before the marker is still kept as
    a valid turn. Returns ``True`` when generation should continue.
    """
    if "SESSION ENDED" in response_content.upper():
        try:
            ended_by, ended_expl, response_content = handle_session_end(
                response_content, speaker_role
            )
            state.session_ended_by = ended_by
            state.session_ended_explanation = ended_expl
            state.is_active = False
            if response_content:
                update_conversation(
                    state.conversation, state.messages_Patient_assist,
                    state.messages_Therapist_assist, role_Patient,
                    role_Therapist, response_content,
                    turns=state.turns, speaker_role=speaker_role,
                )
        except ValueError as e:
            print(f"  Warning: SESSION ENDED handling failed for conversation {state.permutation_index}: {e}")
            state.is_active = False
        return False

    update_conversation(
        state.conversation, state.messages_Patient_assist,
        state.messages_Therapist_assist, role_Patient,
        role_Therapist, response_content,
        turns=state.turns, speaker_role=speaker_role,
    )
    next_map = {"patient": "therapist", "therapist": "patient"}
    state.next_speaker = next_map[speaker_role]
    return True


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   ASYNC PATIENT GENERATION                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def generate_patient_response_async(
    client,
    model_id: str,
    messages_Patient_assist: list,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    seed: Optional[int] = None,
) -> str:
    """Generate a single patient response using the async OpenAI API with retry/backoff."""
    last_error: Optional[BaseException] = None
    api_kwargs = dict(
        model=model_id,
        messages=messages_Patient_assist,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if seed is not None:
        api_kwargs["seed"] = seed
    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(**api_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            sleep_s = backoff_seconds * (2 ** (attempt - 1))
            print(f"  Patient API attempt {attempt}/{max_retries} failed: {e}. Retrying in {sleep_s:.1f}s...")
            await asyncio.sleep(sleep_s)

    raise RuntimeError(f"Patient API failed after {max_retries} retries: {last_error}")


async def generate_patient_responses_batch(
    client,
    model_id: str,
    batch_messages: List[list],
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    seed: Optional[int] = None,
) -> list:
    """Generate patient responses for a batch of conversations concurrently."""
    tasks = [
        generate_patient_response_async(
            client, model_id, messages, max_tokens, temperature,
            semaphore, max_retries, backoff_seconds, seed=seed,
        )
        for messages in batch_messages
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   BATCHED THERAPIST GENERATION                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def generate_therapist_responses_batch(
    therapist_model,
    therapist_tokenizer,
    batch_messages: List[list],
    max_tokens: int,
    temperature: float,
    max_input_tokens: int = 4096,
    stop_strings: Optional[List[str]] = None,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Generate one therapist response per conversation using batched GPU inference.

    Returns:
        (responses, error_type):
            - responses is List[str] on success, else None.
            - error_type is one of: None, "oom", "runtime_error".
    """
    if stop_strings is None:
        stop_strings = _DEFAULT_STOP_STRINGS

    prompts = [
        therapist_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in batch_messages
    ]

    encoded = None
    outputs = None
    try:
        encoded = therapist_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        ).to(therapist_model.device)

        with torch.inference_mode():
            outputs = therapist_model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                do_sample=True,
                max_new_tokens=max_tokens,
                pad_token_id=therapist_tokenizer.eos_token_id,
                eos_token_id=therapist_tokenizer.eos_token_id,
                temperature=temperature,
                num_return_sequences=1,
                stop_strings=stop_strings,
                tokenizer=therapist_tokenizer,
            )
    except torch.OutOfMemoryError as e:
        print(f"  CUDA OOM during therapist generation: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None, "oom"
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            print(f"  Runtime CUDA memory failure during therapist generation: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            return None, "oom"
        print(f"  Runtime error during therapist generation: {e}")
        return None, "runtime_error"

    padded_input_length = encoded.input_ids.shape[1]

    responses = []
    for i in range(len(batch_messages)):
        new_tokens = outputs[i][padded_input_length:]
        decoded = therapist_tokenizer.decode(new_tokens, skip_special_tokens=True)
        cleaned = decoded.split("<|im_end|>")[0].strip()
        responses.append(cleaned)

    del encoded, outputs

    return responses, None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     CONVERSATION LOOP (BATCHED)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def conversation_loop_batch(
    batch_states: List[ConversationState],
    therapist_model,
    therapist_tokenizer,
    client,
    semaphore: asyncio.Semaphore,
    num_utterances: int = 49,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    temperature_patient: float = 0.7,
    temperature_therapist: float = 0.9,
    patient_api_max_retries: int = 3,
    patient_api_backoff_seconds: float = 1.0,
    patient_api_seed: Optional[int] = None,
    therapist_max_input_tokens: int = 2048,
    stop_strings: Optional[List[str]] = None,
    verbose_detailed: bool = False,
):
    """Run a lockstep batched conversation loop over active states.

    Each step processes only active conversations and expects the same
    ``next_speaker`` across them. If desynchronization occurs, only desynced
    conversations are ended gracefully while valid partial data is preserved.

    Returns:
        (batch_states, error_type, desynced_indices):
            - batch_states: list of ConversationState (updated in-place).
            - error_type: None on success, else an error category string.
            - desynced_indices: list of permutation indices that were ended
              due to speaker desync (valid partial data is kept).
    """
    desynced_indices: List[int] = []
    for turn_num in range(num_utterances):
        active_states = [s for s in batch_states if s.is_active]
        if not active_states:
            break

        # Determine current speaker from state (all active states advance in lockstep)
        speaker_role = active_states[0].next_speaker
        desynced = [s for s in active_states if s.next_speaker != speaker_role]
        if desynced:
            ids = [(s.permutation_index, s.next_speaker) for s in desynced]
            print(
                f"  Warning: Speaker desync detected at turn {turn_num}. "
                f"Expected '{speaker_role}', desynced: {ids}. "
                f"Ending desynced conversations gracefully (data up to this point is kept)."
            )
            for s in desynced:
                s.is_active = False
                # Don't set failed=True — turns collected so far are valid for prompt extraction
                desynced_indices.append(s.permutation_index)
            active_states = [s for s in active_states if s.is_active]
            if not active_states:
                break

        if speaker_role == "patient":
            role_Patient = "assistant"
            role_Therapist = "user"

            batch_patient_messages = [s.messages_Patient_assist for s in active_states]
            responses = await generate_patient_responses_batch(
                client, patient_model_id, batch_patient_messages,
                max_tokens_per_response, temperature_patient,
                semaphore, patient_api_max_retries, patient_api_backoff_seconds,
                seed=patient_api_seed,
            )

            for state, response_content in zip(active_states, responses):
                # Handle per-conversation API failures gracefully
                if isinstance(response_content, BaseException):
                    print(f"  Patient API failed for conversation {state.permutation_index}: {response_content}")
                    state.is_active = False
                    state.failed = True
                    continue

                _process_session_response(state, response_content, speaker_role, role_Patient, role_Therapist)
        else:  # speaker_role == "therapist"
            role_Patient = "user"
            role_Therapist = "assistant"

            batch_therapist_messages = [s.messages_Therapist_assist for s in active_states]
            responses, error_type = generate_therapist_responses_batch(
                therapist_model, therapist_tokenizer, batch_therapist_messages,
                max_tokens_per_response, temperature_therapist,
                max_input_tokens=therapist_max_input_tokens,
                stop_strings=stop_strings,
            )

            if responses is None:
                for state in active_states:
                    state.is_active = False
                    state.failed = True
                return batch_states, (error_type or "therapist_generation_failed"), desynced_indices

            for state, response_content in zip(active_states, responses):
                _process_session_response(state, response_content, speaker_role, role_Patient, role_Therapist)

        if verbose_detailed:
            indent = "  " if speaker_role == "patient" else ""
            truncated = [r[:100] + "..." if isinstance(r, str) and len(r) > 100 else r for r in responses]
            print(f"{speaker_role.upper()} RESPONSES ({len(responses)}):{indent} {truncated}")

        # Cleanup ended conversations (CPU memory only)
        newly_ended = [s for s in active_states if not s.is_active]
        if newly_ended:
            for s in newly_ended:
                s.messages_Patient_assist = []
                s.messages_Therapist_assist = []

    return batch_states, None, desynced_indices


async def synthesize_conversations_batch(
    permutations_batch,
    system_prompt_therapist: str,
    therapist_init_utterance: str,
    therapist_model,
    therapist_tokenizer,
    client,
    patient_api_concurrency: int = 16,
    include_empty_init_user_message: bool = False,
    num_utterances: int = 49,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    temperature_patient: float = 0.7,
    temperature_therapist: float = 0.9,
    patient_api_max_retries: int = 3,
    patient_api_backoff_seconds: float = 1.0,
    patient_api_seed: Optional[int] = None,
    therapist_max_input_tokens: int = 2048,
    stop_strings: Optional[List[str]] = None,
    verbose_detailed: bool = False,
):
    """Synthesize a batch of conversations in parallel."""
    semaphore = asyncio.Semaphore(patient_api_concurrency)
    batch_states: List[ConversationState] = []
    for perm_idx, perm in permutations_batch:
        # patient_init_utterance is always empty: the patient speaks only after
        # seeing the therapist's greeting. When include_empty_init_user_message
        # is True, an empty user message is inserted before the therapist's
        # first assistant message to satisfy chat templates that require
        # system → user → assistant ordering.
        conversation, messages_Patient, messages_Therapist = initialize_conversation(
            system_prompt_therapist,
            perm["patient_system_prompt"],
            therapist_init_utterance,
            "",
            include_empty_init_user_message=include_empty_init_user_message,
        )
        batch_states.append(
            ConversationState(
                permutation_index=perm_idx,
                conversation=conversation,
                messages_Patient_assist=messages_Patient,
                messages_Therapist_assist=messages_Therapist,
                turns=[{"role": "therapist", "content": therapist_init_utterance}],
                next_speaker="patient",
            )
        )

    batch_states, error_type, desynced_indices = await conversation_loop_batch(
        batch_states, therapist_model, therapist_tokenizer, client, semaphore,
        num_utterances=num_utterances,
        patient_model_id=patient_model_id,
        max_tokens_per_response=max_tokens_per_response,
        temperature_patient=temperature_patient,
        temperature_therapist=temperature_therapist,
        patient_api_max_retries=patient_api_max_retries,
        patient_api_backoff_seconds=patient_api_backoff_seconds,
        patient_api_seed=patient_api_seed,
        therapist_max_input_tokens=therapist_max_input_tokens,
        stop_strings=stop_strings,
        verbose_detailed=verbose_detailed,
    )
    return batch_states, error_type, desynced_indices


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     BATCH EXECUTION HELPERS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _run_async(coro):
    """Run a coroutine, compatible with notebooks and plain scripts.

    When no event loop is running (plain scripts), uses ``asyncio.run()``.
    When called from inside an already-running loop (e.g. Jupyter notebooks),
    runs the coroutine in a *separate thread* with its own event loop to avoid
    the need for ``nest_asyncio`` (broken on Python >= 3.13 due to stricter
    ``contextvars`` re-entry rules).

    NOTE: The spawned thread shares the process's CUDA context. PyTorch GPU
    operations called from coroutines in this thread work correctly because
    PyTorch's CUDA subsystem is per-process, not per-thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_future: Future = Future()

    def _thread_target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                result_future.set_result(result)
            except BaseException as exc:
                result_future.set_exception(exc)
            finally:
                loop.close()
        except BaseException as exc:
            if not result_future.done():
                result_future.set_exception(exc)

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join()
    return result_future.result()


def run_synthesis_batch(
    permutations_batch,
    therapist_model,
    therapist_tokenizer,
    client,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    patient_api_concurrency: int = 16,
    include_empty_init_user_message: bool = False,
    num_utterances: int = 49,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    temperature_patient: float = 0.7,
    temperature_therapist: float = 0.9,
    patient_api_max_retries: int = 3,
    patient_api_backoff_seconds: float = 1.0,
    patient_api_seed: Optional[int] = None,
    therapist_max_input_tokens: int = 2048,
    stop_strings: Optional[List[str]] = None,
    verbose_detailed: bool = False,
) -> dict:
    """Execute one batch synthesis call with normalized error handling.

    Returns:
        dict with keys:
            final_states (list[ConversationState] | None),
            error_type (None | str): one of
                {"oom", "runtime_error", "batch_exception"} on failure,
            desynced_indices (list[int])  — permutation indices ended by desync.
    """
    try:
        final_states, error_type, desynced_indices = _run_async(
            synthesize_conversations_batch(
                permutations_batch=permutations_batch,
                system_prompt_therapist=therapist_system_prompt,
                therapist_init_utterance=therapist_init_utterance,
                therapist_model=therapist_model,
                therapist_tokenizer=therapist_tokenizer,
                client=client,
                patient_api_concurrency=patient_api_concurrency,
                include_empty_init_user_message=include_empty_init_user_message,
                num_utterances=num_utterances,
                patient_model_id=patient_model_id,
                max_tokens_per_response=max_tokens_per_response,
                temperature_patient=temperature_patient,
                temperature_therapist=temperature_therapist,
                patient_api_max_retries=patient_api_max_retries,
                patient_api_backoff_seconds=patient_api_backoff_seconds,
                patient_api_seed=patient_api_seed,
                therapist_max_input_tokens=therapist_max_input_tokens,
                stop_strings=stop_strings,
                verbose_detailed=verbose_detailed,
            )
        )
        return {"final_states": final_states, "error_type": error_type, "desynced_indices": desynced_indices}
    except torch.OutOfMemoryError as e:
        print(f"  ERROR: CUDA OOM in batch generation: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return {"final_states": None, "error_type": "oom", "desynced_indices": []}
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            print(f"  ERROR: Runtime CUDA memory failure in batch generation: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            return {"final_states": None, "error_type": "oom", "desynced_indices": []}
        print(f"  ERROR: Batch generation runtime error: {e}")
        return {"final_states": None, "error_type": "runtime_error", "desynced_indices": []}
    except Exception as e:
        print(f"  ERROR: Batch generation failed: {e}")
        return {"final_states": None, "error_type": "batch_exception", "desynced_indices": []}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  FULL CONVERSATION GENERATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class _Counters:
    session_ended: int = 0
    completed: int = 0
    failed: int = 0
    desynced: List[int] = field(default_factory=list)


def _resume_from_disk(save_dir: str, total: int, verbose: bool = False) -> Dict[int, ConversationState]:
    """Load previously saved conversation_*.csv files. Returns ``{idx: state}``."""
    states: Dict[int, ConversationState] = {}
    for i in range(total):
        csv_path = os.path.join(save_dir, f"conversation_{i}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            states[i] = load_conversation_from_csv(csv_path, i)
        except Exception as e:
            print(f"  Warning: Could not load conversation_{i}.csv: {e}")
    if states and verbose:
        print(f"  Resumed {len(states)}/{total} conversations from {save_dir}")
    return states


def _record_completed_state(
    state: ConversationState,
    save_dir: Optional[str],
    counters: _Counters,
    detailed: bool,
) -> bool:
    """Update counters and (optionally) save CSV for one finished state.

    Returns True if the state should be kept; False if skipped as failed/empty.
    In both cases, large message-history fields are released.
    """
    if state.failed or not state.conversation or len(state.conversation) <= 1:
        if detailed:
            print(f"      Skipping conversation {state.permutation_index} (failed or empty)")
        counters.failed += 1
        state.turns = []
        state.conversation = []
        state.messages_Patient_assist = []
        state.messages_Therapist_assist = []
        return False

    if state.session_ended_by is not None:
        counters.session_ended += 1
    else:
        counters.completed += 1

    if save_dir:
        save_conversation_csv(state, save_dir)
        if detailed:
            print(
                f"      Saved conversation_{state.permutation_index}.csv "
                f"({len(state.conversation)} utterances)"
            )

    # Free message history after saving (turns/conversation kept for prompt extraction)
    state.messages_Patient_assist = []
    state.messages_Therapist_assist = []
    return True


def _run_one_pass(
    remaining: List[int],
    permutations: List[Dict],
    completed: Dict[int, ConversationState],
    counters: _Counters,
    *,
    therapist_model,
    therapist_tokenizer,
    client,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    save_dir: Optional[str],
    batch_size: int,
    batch_cooldown_seconds: float,
    total: int,
    start_time: float,
    verbose: bool,
    detailed: bool,
    synthesis_kwargs: dict,
) -> bool:
    """Run one full pass over remaining indices.

    Mutates ``completed`` and ``counters`` in place. Returns ``True`` if any
    new conversations were added (used for the retry-without-progress check).
    """
    num_batches = (len(remaining) + batch_size - 1) // batch_size
    progress_made = False

    for batch_num, batch_start in enumerate(range(0, len(remaining), batch_size), 1):
        batch_indices = remaining[batch_start: batch_start + batch_size]
        permutations_batch = [(idx, permutations[idx]) for idx in batch_indices]
        batch_time_start = time.time()

        if detailed:
            print(f"    Batch {batch_num}/{num_batches}: indices {batch_indices}")

        batch_result = run_synthesis_batch(
            permutations_batch=permutations_batch,
            therapist_model=therapist_model,
            therapist_tokenizer=therapist_tokenizer,
            client=client,
            therapist_system_prompt=therapist_system_prompt,
            therapist_init_utterance=therapist_init_utterance,
            **synthesis_kwargs,
        )

        final_states = batch_result["final_states"]
        counters.desynced.extend(batch_result["desynced_indices"])
        batch_elapsed = time.time() - batch_time_start

        if final_states is None:
            if verbose:
                print(f"    Batch {batch_num}/{num_batches} FAILED ({batch_result['error_type']}) — {batch_elapsed:.1f}s")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(batch_cooldown_seconds)
            continue

        batch_saved = 0
        for state in final_states:
            if _record_completed_state(state, save_dir, counters, detailed):
                completed[state.permutation_index] = state
                progress_made = True
                batch_saved += 1

        if verbose:
            total_elapsed = time.time() - start_time
            print(
                f"    Batch {batch_num}/{num_batches}: "
                f"{batch_saved}/{len(batch_indices)} saved — "
                f"{len(completed)}/{total} total "
                f"({len(completed)/total*100:.0f}%) — "
                f"batch {batch_elapsed:.1f}s, total {total_elapsed:.1f}s"
            )

    return progress_made


def _summarize(
    completed: Dict[int, ConversationState],
    counters: _Counters,
    total: int,
    elapsed: float,
    detailed: bool,
) -> None:
    """Print the final per-run outcome summary."""
    saved_total = len(completed)
    missing = total - saved_total
    print(
        f"\n  Generation summary ({elapsed:.1f}s):\n"
        f"    Total requested:    {total}\n"
        f"    Saved (usable):     {saved_total}\n"
        f"      - Session ended:  {counters.session_ended}\n"
        f"      - Completed:      {counters.completed}\n"
        f"    Desync (graceful):  {len(counters.desynced)}\n"
        f"    Failed / empty:     {counters.failed}\n"
        f"    Missing:            {missing}"
    )
    if counters.desynced and detailed:
        print(f"    Desynced indices:   {counters.desynced}")


def generate_all_conversations(
    therapist_model,
    therapist_tokenizer,
    client,
    permutations: List[Dict],
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    save_dir: Optional[str] = None,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    num_utterances: int = 49,
    temperature_therapist: float = 0.9,
    temperature_patient: float = 0.7,
    batch_size: int = 8,
    therapist_max_input_tokens: int = 2048,
    patient_api_concurrency: int = 16,
    patient_api_max_retries: int = 3,
    patient_api_seed: Optional[int] = None,
    patient_api_backoff_seconds: float = 1.0,
    batch_cooldown_seconds: float = 1.0,
    max_retries_without_progress: int = 3,
    stop_strings: Optional[List[str]] = None,
    include_empty_init_user_message: bool = False,
    verbose: bool = False,
    verbose_detailed: bool = False,
) -> List[ConversationState]:
    """Generate conversations for all permutations with retry and OOM recovery.

    Args:
        save_dir: If provided, save individual conversation CSVs here (enables resume).
    """
    total = len(permutations)
    completed: Dict[int, ConversationState] = {}
    counters = _Counters()
    start_time = time.time()
    detailed = verbose_detailed

    # Bundle per-synthesis knobs once to keep _run_one_pass's signature readable.
    synthesis_kwargs = dict(
        patient_api_concurrency=patient_api_concurrency,
        include_empty_init_user_message=include_empty_init_user_message,
        num_utterances=num_utterances,
        patient_model_id=patient_model_id,
        max_tokens_per_response=max_tokens_per_response,
        temperature_patient=temperature_patient,
        temperature_therapist=temperature_therapist,
        patient_api_max_retries=patient_api_max_retries,
        patient_api_backoff_seconds=patient_api_backoff_seconds,
        patient_api_seed=patient_api_seed,
        therapist_max_input_tokens=therapist_max_input_tokens,
        stop_strings=stop_strings,
        verbose_detailed=verbose_detailed,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        completed = _resume_from_disk(save_dir, total, verbose=verbose)

    retries_without_progress = 0
    while True:
        remaining = [i for i in range(total) if i not in completed]

        if not remaining:
            if verbose:
                print(f"  All {total} conversations completed! ({time.time() - start_time:.1f}s)")
            break

        if retries_without_progress >= max_retries_without_progress:
            print(
                f"  WARNING: Reached retry limit ({max_retries_without_progress}). "
                f"Stopping with {len(remaining)} conversations missing."
            )
            break

        if verbose:
            num_batches = (len(remaining) + batch_size - 1) // batch_size
            elapsed = time.time() - start_time
            print(
                f"\n  Remaining: {len(remaining)}/{total} conversations "
                f"({len(completed)}/{total} done, {num_batches} batches) "
                f"[{elapsed:.1f}s elapsed]"
            )

        progress = _run_one_pass(
            remaining, permutations, completed, counters,
            therapist_model=therapist_model,
            therapist_tokenizer=therapist_tokenizer,
            client=client,
            therapist_system_prompt=therapist_system_prompt,
            therapist_init_utterance=therapist_init_utterance,
            save_dir=save_dir,
            batch_size=batch_size,
            batch_cooldown_seconds=batch_cooldown_seconds,
            total=total,
            start_time=start_time,
            verbose=verbose,
            detailed=detailed,
            synthesis_kwargs=synthesis_kwargs,
        )

        if progress:
            retries_without_progress = 0
        else:
            retries_without_progress += 1
            if verbose:
                print(
                    f"  No progress this pass. "
                    f"Retry {retries_without_progress}/{max_retries_without_progress}"
                )

    _summarize(completed, counters, total, time.time() - start_time, detailed)
    return list(completed.values())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  PROMPT EXTRACTION (CONV → TRAINING SAMPLES)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# For each conversation, emit one training sample after every patient turn (the
# point right before therapist generation). Each sample contains a chat-template-
# formatted prompt and a plain-text transcript for oracle eval.
#
# MCL filter: ``min_conv_length`` drops slices whose conversation-so-far is
# shorter than ``min_conv_length`` total utterances (therapist + patient
# combined; same ``n_turns`` unit as Partial_Conv_Oracle_EDA).
#
# Token budgeting:
# - ``_estimate_turn_token_costs`` + ``_compute_system_overhead`` give O(1)
#   running totals so most prompts skip the truncation path.
# - When over budget, ``_truncate_by_dropping_turns`` drops the oldest turns
#   and re-renders the chat template, preserving template structure.
# - ``_truncate_by_token_tail`` is the legacy fallback — keeps the last N tokens
#   of the rendered prompt; may slice through template control tokens.


def format_conversation_for_oracle(messages: List[Dict]) -> str:
    """Convert a list of message dicts into a plain text transcript using
    ``[PATIENT]`` and ``[THERAPIST]`` labels (user=patient, assistant=therapist).
    Removes the system prompt.

    Note:
        Message content may include embedded newlines (including ``"\\n\\n"``).
        :func:`reward._parse_transcript_to_messages` is the inverse parser
        and handles continuation fragments to reconstruct the original turn.
    """
    transcript = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            transcript.append(f"[PATIENT]: {content}")
        elif role == "assistant":
            transcript.append(f"[THERAPIST]: {content}")
        # Explicitly ignore 'system' role
    return "\n\n".join(transcript)


def turns_to_messages(turns: List[Dict], system_prompt: str) -> List[Dict]:
    """Convert role-tagged turns into message dicts with system prompt.

    Args:
        turns: List of {"role": "therapist"|"patient", "content": str} dicts.
        system_prompt: System prompt for the therapist.

    Returns:
        List of message dicts with role in {'system','assistant','user'}.
    """
    messages = [{"role": "system", "content": str(system_prompt)}]
    role_map = {"therapist": "assistant", "patient": "user"}
    for turn in turns:
        messages.append({"role": role_map[turn["role"]], "content": str(turn["content"])})
    return messages


def turns_to_patient_messages(turns: List[Dict], system_prompt: str) -> List[Dict]:
    """Patient-perspective view of role-tagged turns (therapist→user, patient→assistant).

    Mirror of :func:`turns_to_messages` with the roles flipped and the patient's
    system prompt — matches the ``messages_Patient_assist`` shape built by
    :func:`initialize_conversation`.

    Args:
        turns: List of {"role": "therapist"|"patient", "content": str} dicts.
        system_prompt: System prompt for the patient.

    Returns:
        List of message dicts with role in {'system','assistant','user'}.
    """
    messages = [{"role": "system", "content": str(system_prompt)}]
    role_map = {"therapist": "user", "patient": "assistant"}
    for turn in turns:
        messages.append({"role": role_map[turn["role"]], "content": str(turn["content"])})
    return messages


def _estimate_turn_token_costs(turns: List[Dict], tokenizer) -> List[int]:
    """Estimate per-turn token costs by encoding each turn's ChatML wrapper.

    Estimates may differ slightly from a full template render due to token
    boundary effects at message joins, but are accurate enough for budget
    decisions.
    """
    role_map = {"therapist": "assistant", "patient": "user"}
    costs = []
    for turn in turns:
        role = role_map.get(turn["role"], turn["role"])
        single_msg = f"<|im_start|>{role}\n{turn['content']}<|im_end|>\n"
        costs.append(len(tokenizer.encode(single_msg, add_special_tokens=False)))
    return costs


def _compute_system_overhead(system_prompt: str, tokenizer) -> int:
    """Compute token overhead for system message + generation prompt suffix."""
    sys_msg = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    overhead = len(tokenizer.encode(sys_msg, add_special_tokens=False))
    gen_suffix = "<|im_start|>assistant\n"
    overhead += len(tokenizer.encode(gen_suffix, add_special_tokens=False))
    return overhead


def _truncate_by_dropping_turns(
    partial_turns: List[Dict],
    system_prompt: str,
    tokenizer,
    max_prompt_tokens: int,
    turn_token_costs: Optional[List[int]] = None,
    system_overhead: Optional[int] = None,
) -> Optional[str]:
    """Drop oldest turns until the rendered prompt fits the token budget.

    Uses pre-computed per-turn token costs (if provided) to find the drop
    point in O(1), then verifies with a single accurate render. Falls back
    to iterative render if the estimate is slightly off.
    """
    if turn_token_costs is None:
        turn_token_costs = _estimate_turn_token_costs(partial_turns, tokenizer)
    if system_overhead is None:
        system_overhead = _compute_system_overhead(system_prompt, tokenizer)

    total = system_overhead + sum(turn_token_costs)

    drop = 0
    while total > max_prompt_tokens and drop < len(partial_turns) - 1:
        total -= turn_token_costs[drop]
        drop += 1

    if drop >= len(partial_turns):
        return None

    remaining = partial_turns[drop:]
    messages = turns_to_messages(remaining, system_prompt)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
        return prompt

    # Estimate was slightly off — fall back to iterative drop from here
    for _ in range(1, len(remaining)):
        remaining = remaining[1:]
        if not remaining:
            return None
        messages = turns_to_messages(remaining, system_prompt)
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
            return prompt

    return None


def _truncate_by_token_tail(prompt: str, tokenizer, max_prompt_tokens: int) -> str:
    """Keep the last ``max_prompt_tokens`` tokens of the rendered prompt.

    Warning: May cut through template control tokens or special token structure.
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.decode(tokens[-max_prompt_tokens:], skip_special_tokens=False)


def extract_prompts_from_conversations(
    completed_states: List[ConversationState],
    system_prompt: str,
    tokenizer,
    min_conv_length: int = 2,
    max_prompt_tokens: int = 4096,
    truncation_mode: str = "drop_oldest",
    permutations: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Extract therapist-next-turn training prompts from completed conversations.

    For each conversation, extracts a sample after each patient turn (i.e.,
    right before therapist generation). Each sample contains:

    - ``prompt``: Chat-template-formatted partial conversation for model input.
    - ``transcript``: Plain text [PATIENT]/[THERAPIST] transcript for oracle eval.
    - ``patient_system_prompt``: The patient's unique system prompt (if
      ``permutations`` provided — required for look-ahead reward).

    Args:
        min_conv_length: Minimum number of utterances (therapist + patient
            combined, same unit as Partial_Conv_Oracle_EDA's ``n_turns``)
            in the conversation-so-far before extracting a prompt. Default
            2 = skip the very first exchange (too little context). Larger
            values filter slices the partial-conv oracle proxy can't score
            faithfully — see EDA: rank-agreement is ~0.66/0.73 at n_turns=2,
            clears 0.8 at ~10, 0.9 at ~30.
        truncation_mode:
            ``"drop_oldest"`` (default): iteratively drop oldest turns and
            re-render the chat template, preserving template integrity.
            ``"legacy"``: raw token-tail truncation (may break template).
    """
    all_prompts = []

    # Pre-compute system overhead once (shared across all conversations)
    sys_overhead = _compute_system_overhead(system_prompt, tokenizer)

    for state in completed_states:
        # Prefer explicit role-tagged turns; fall back to utterance list
        turns = state.turns
        if not turns and state.conversation:
            turns = [
                {"role": "therapist" if j % 2 == 0 else "patient", "content": utt}
                for j, utt in enumerate(state.conversation)
            ]

        if not turns or len(turns) < min_conv_length:
            continue

        patient_sys_prompt = None
        if permutations is not None:
            patient_sys_prompt = permutations[state.permutation_index]["patient_system_prompt"]

        # Pre-compute per-turn token costs once per conversation (O(n) encodes)
        turn_costs = _estimate_turn_token_costs(turns, tokenizer)

        # Extract a prompt after each patient turn (therapist speaks next)
        running_tokens = sys_overhead
        for i, turn in enumerate(turns):
            running_tokens += turn_costs[i]

            if turn["role"] != "patient":
                continue
            if (i + 1) < min_conv_length:
                continue

            partial_turns = turns[: i + 1]

            if running_tokens <= max_prompt_tokens:
                # Under budget — render template (needed for exact prompt text)
                messages = turns_to_messages(partial_turns, system_prompt)
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                # Over budget — truncate
                if truncation_mode == "drop_oldest":
                    prompt = _truncate_by_dropping_turns(
                        partial_turns, system_prompt, tokenizer, max_prompt_tokens,
                        turn_token_costs=turn_costs[: i + 1],
                        system_overhead=sys_overhead,
                    )
                    if prompt is None:
                        continue  # Skip — even a single turn exceeds token budget
                else:  # "legacy" — raw token-tail truncation
                    messages = turns_to_messages(partial_turns, system_prompt)
                    prompt = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    prompt = _truncate_by_token_tail(prompt, tokenizer, max_prompt_tokens)

            # Build transcript (plain text for oracle)
            messages = turns_to_messages(partial_turns, system_prompt)
            transcript = format_conversation_for_oracle(messages)

            entry = {
                "prompt": prompt,
                "transcript": transcript,
                "conversation_id": state.permutation_index,
            }
            if patient_sys_prompt is not None:
                entry["patient_system_prompt"] = patient_sys_prompt

            all_prompts.append(entry)

    return all_prompts


def extract_prompts_from_saved_conversations(
    conversation_dir: str,
    system_prompt: str,
    tokenizer,
    num_files: int = 96,
    min_conv_length: int = 2,
    max_prompt_tokens: int = 4096,
    truncation_mode: str = "drop_oldest",
    permutations: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Extract GRPO training prompts from previously saved conversation CSV files.

    Alternative to :func:`extract_prompts_from_conversations` when conversations
    were saved to disk (e.g., from a previous run).
    """
    states = []
    for i in range(num_files):
        csv_path = os.path.join(conversation_dir, f"conversation_{i}.csv")
        if not os.path.exists(csv_path):
            continue
        states.append(load_conversation_from_csv(csv_path, i))

    print(f"  Loaded {len(states)} conversations from {conversation_dir}")

    return extract_prompts_from_conversations(
        completed_states=states,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        min_conv_length=min_conv_length,
        max_prompt_tokens=max_prompt_tokens,
        truncation_mode=truncation_mode,
        permutations=permutations,
    )
