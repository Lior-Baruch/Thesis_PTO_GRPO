"""pto_trainer.py -- preference trees, the tau filter, and the DPO update.

GRPO keeps every completion it samples and turns the whole group into one gradient step. PTO
throws almost all of them away: at each branch point it samples ``M`` candidate therapist turns,
scores them through the same look-ahead + oracle path GRPO's reward uses, and keeps exactly two --
the best and the worst -- as a ``(chosen, rejected)`` preference pair for DPO. Everything in this
module exists to make that reduction faithful, resumable and auditable.

Two modes, and the default is the one that makes it a TREE
----------------------------------------------------------
``greedy`` (default, and true PTO) slices an ``MCL``-length prefix off each step-1 conversation as
a trunk seed, then grows ONE trunk per persona: branch ``M``, score, **append the best completion
to the trunk**, let the patient reply, repeat. That append is the whole point -- the choice made at
depth ``d`` is the context every branch at depth ``d+2`` is sampled from, so the pairs describe a
trajectory the policy actually walks rather than ``M`` independent draws off a fixed script.

``independent`` branches every eligible patient turn of the PRE-RECORDED step-1 conversation and
never feeds the winner back. It is kept as an alternate arm (``_PTindep`` in the arm name) and is
not run in v1. It is also the cheaper control: if greedy and independent produce the same pairs,
the feedback bought nothing.

The grower runs LOCK-STEP across all trunks -- one batched branch-sample and one batched
score call per depth, for every live trunk at once -- mirroring ``core.lookahead``. Growing one
trunk at a time would issue ~96x more GPU calls and leave the oracle semaphore idle.

Three things that are easy to get wrong and silent when you do
--------------------------------------------------------------
**1. TRL 1.4.0 removed ``max_prompt_length``.** ``DPOConfig`` now caps prompt+completion with a
single ``max_length`` under ``truncation_mode='keep_start'``, which drops the END of the sequence --
i.e. it would slice the RESPONSE off and leave a pair whose chosen and rejected are both empty.
Every prompt is therefore PRE-capped here with
``core.conversations.build_truncated_training_prompt`` (drop-oldest), and ``max_length`` is set to
``max_prompt_tokens + max_completion_length`` so ``keep_start`` can never bite. When the truncator
returns ``None`` (even one most-recent turn exceeds the budget) the pair is skipped -- but in
``greedy`` mode the trunk still advances, because freezing a trunk over an unrenderable prompt
would silently shorten every later branch point on it.

**2. ``pairs.csv`` is both the audit trail AND the completion marker.** Its presence makes a
resumed iteration reload the pairs and skip the (dominant) build -- and an EMPTY one is a trap:
reload 0 pairs, skip the build, then hit the zero-pairs guard. That guard sits at function-body
indentation AFTER the reload/build branch precisely so BOTH paths reach it before any adapter is
written. The fix for an empty marker is to DELETE it and rebuild. It is never to lower
``PREF_FILTER_TAU``: tau is not encoded in ``EXPERIMENT_NAME``, so changing it mid-arm writes two
different configurations into one folder with nothing on disk able to tell them apart.

**3. Message lists are freed when a conversation finishes.** ``core.conversations`` calls
``release_messages()`` on every completed state, so a step-1 conversation reaching the greedy
grower has ``turns`` but empty ``messages_therapist`` / ``messages_patient``. They are rebuilt from
``turns`` via ``turns_to_messages`` / ``turns_to_patient_messages`` before growth starts; forgetting
that yields trunks with no system prompt and no history, which generates plausible text and grades
it against nothing.

Resume
------
``_progress.json`` is an atomic per-step snapshot (written after every lock-step depth in greedy,
after every conversation in independent) guarded by a config fingerprint that includes
``pref_filter_tau``, ``num_branches_per_turn``, ``min_conv_length``, ``num_utterances_for_data``,
``greedy_trunk_target_len`` and ``seed``. A snapshot whose fingerprint differs is DISCARDED, never
merged: half a build at one tau plus half at another is a folder nobody can interpret.

Timing
------
The preference build is PTO's DOMINANT phase (5.7 of 8.1 h in the Exp3 measurement), so it is
timed separately and logged through ``core.timing.log_session``. A resumed iteration that reloaded
``pairs.csv`` legitimately logs ``pref_pair_s = 0.0`` -- the earlier session already recorded the
build -- which is exactly why the append-only per-session log, rather than one field, is what makes
the total recoverable.

The per-iteration orchestration LOOP lives in ``train_pto.ipynb``; this module supplies the phases
it composes.
"""

from __future__ import annotations

# trl BEFORE torch. On the local Blackwell card (sm_120) importing trl after torch segfaults at
# CUDA init -- exit 139, no traceback, nothing to catch. core.runtime.assert_import_order turns
# that into a readable error; this import position is what keeps it from ever firing.
from trl import DPOConfig, DPOTrainer  # isort: skip

import asyncio
import contextlib
import functools
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# datasets SECOND, still ahead of torch: pyarrow and torch each initialise native
# runtimes and whichever goes first wins. MEASURED locally -- `import torch, datasets`
# is an access violation (exit 139) inside pyarrow.dataset. This module used to be safe
# only because core.* pulled pandas in first; that was luck, so it is now explicit.
import pandas as pd
from datasets import Dataset

import torch
from peft import PeftModel

from core.concurrency import AsyncPrimitives, run_async
from core.config import GenConfig, PTOTrainingConfig, RunPaths, config_to_metadata
from core.config import write_run_metadata as _write_metadata_payload
from core.conversations import (
    SESSION_END_KEYWORD,
    ConversationState,
    build_truncated_training_prompt,
    format_conversation_for_oracle,
    generate_all_conversations,
    generate_patient_batch,
    handle_session_end,
    turns_to_messages,
    turns_to_patient_messages,
)
from core.lookahead import LookaheadConfig, LookaheadState
from core.oracle import OracleConfig
from core.policy import generate_therapist_batch, list_hf_checkpoints, patch_generate
from core.recorder import (
    PHASE_INDEPENDENT,
    PHASE_TREE,
    EDARecorder,
    build_branch_record,
    to_jsonable,
)
from core.reward import CandidateScore, score_pref_candidates
from core.tb import patch_trainer_tensorboard_callback, setup_tensorboard_logging
from core.timing import log_session, metadata_fields

__all__ = [
    # Constants
    "ROLE_THERAPIST",
    "ROLE_PATIENT",
    "PAIR_COLUMNS_COMMON",
    "BRANCH_COLUMN_GREEDY",
    "BRANCH_COLUMN_INDEPENDENT",
    # Results
    "IterationResult",
    # Preference-pair construction
    "slice_trunk_seeds",
    "grow_preference_trees_batch",
    "build_pref_pairs_for_conversation",
    "build_pref_pairs_independent",
    "build_pref_pairs",
    "build_pref_pairs_async",
    # Persistence / resume
    "pref_config_fingerprint",
    "write_pairs_csv",
    "reload_pairs_csv",
    # DPO
    "build_lora_config",
    "build_dpo_dataset",
    "build_dpo_config",
    # Phases
    "run_generation_phase",
    "run_training_phase",
    "run_one_iteration",
    "run_final_eval",
    # Artifacts
    "save_iteration_checkpoint",
    "write_run_metadata",
    "push_adapter_to_hub",
]


# =============================================================================
# CONSTANTS
# =============================================================================

#: Turn roles as ``core.conversations.ConversationState.turns`` spells them. Mirrored here rather
#: than imported because the module keeps them private; they are part of the documented shape of
#: ``turns`` (``[{"role": "therapist"|"patient", "content": str}, ...]``), so this is a copy of a
#: contract, not of an implementation detail.
ROLE_THERAPIST = "therapist"
ROLE_PATIENT = "patient"

_NEXT_SPEAKER = {ROLE_THERAPIST: ROLE_PATIENT, ROLE_PATIENT: ROLE_THERAPIST}

#: Columns every pair carries, in ``pairs.csv`` order.
PAIR_COLUMNS_COMMON = (
    "prompt", "chosen", "rejected", "chosen_score", "rejected_score",
    "conversation_id", "persona_id",
)

#: The branch-position column, which DIFFERS BY MODE and is the one thing a reader of ``pairs.csv``
#: must branch on. ``greedy`` records trunk DEPTH (utterances of the trunk when the branch point
#: was reached); ``independent`` records the index of the patient turn in the pre-recorded
#: conversation. Neither is unique on its own -- key on ``(conversation_id, <branch column>)``.
BRANCH_COLUMN_GREEDY = "branch_depth"
BRANCH_COLUMN_INDEPENDENT = "branch_turn_index"

#: Mode tokens as ``PTOTrainingConfig.pref_tree_mode`` spells them (the arm name uses ``indep``).
_MODE_GREEDY = "greedy"
_MODE_INDEPENDENT = "independent"

_LOG = "  "


# =============================================================================
# RESULT
# =============================================================================


@dataclass(frozen=True)
class IterationResult:
    """What one PTO iteration produced. Returned by :func:`run_one_iteration`.

    Attributes:
        policy: the UPDATED policy (TRL's trainer hands back a possibly re-wrapped model, so the
            orchestration loop must rebind its own variable from this field -- reusing the object
            it passed in would train iteration ``n+1`` from a stale wrapper).
        iteration: the iteration number, echoed so a log line needs no extra bookkeeping.
        step_delta: optimizer steps this iteration contributed, for the cumulative TB x-axis.
        n_conversations: usable conversations generated in step 1.
        n_pref_pairs: preference pairs the build produced (or reloaded).
        generation_s / pref_pair_s / training_s: THIS process's seconds per phase. ``pref_pair_s``
            is 0.0 when the build was reloaded from ``pairs.csv`` -- see the module docstring.
        adapter_dir: where the iteration's adapter was written.
        pref_pairs_reloaded: True when ``pairs.csv`` already existed and the build was skipped.
        lookahead_sub_batch: the sub-batch the look-ahead rollout ENDED the iteration at. It is
            not in ``EXPERIMENT_NAME``, so an OOM halving leaves no other trace and per-iteration
            wall-clock silently stops being comparable without it.
    """

    policy: Any
    iteration: int
    step_delta: int
    n_conversations: int
    n_pref_pairs: int
    generation_s: float
    pref_pair_s: float
    training_s: float
    adapter_dir: str
    pref_pairs_reloaded: bool
    lookahead_sub_batch: Optional[int]


# =============================================================================
# SMALL SHARED HELPERS
# =============================================================================


@contextlib.contextmanager
def _inference_policy(model):
    """Put *model* in eval + KV-cache mode for a generate, and restore both afterwards.

    The preference build runs between generation and training, so the policy is normally already
    in eval mode -- but ``run_one_iteration`` may be resumed into at any point and a leaked
    ``use_cache=True`` silently disables gradient checkpointing's assumptions in the DPO step that
    follows. Restoring in a ``finally`` costs nothing and removes the ordering dependency.
    """
    was_training = bool(getattr(model, "training", False))
    old_use_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    try:
        yield
    finally:
        model.config.use_cache = old_use_cache
        model.train(was_training)


async def _sample_completions_batch(
    model,
    tokenizer,
    primitives: AsyncPrimitives,
    batch_messages: Sequence[Sequence[Dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
    max_input_tokens: int,
    stop_strings: Optional[Sequence[str]],
    chunk_size: int,
) -> List[str]:
    """Sample one therapist completion per message-list, in GPU-bounded chunks.

    Args:
        batch_messages: therapist-perspective histories. The branch fan-out is expressed by
            REPEATING a trunk's history ``M`` times -- ``do_sample=True`` makes each draw
            independent, and one padded batch is far cheaper than ``M`` calls.
        chunk_size: cap on one ``generate`` batch. ``trunks x M`` is 768 on the default grid;
            issuing that as a single padded generate is a VRAM spike with no throughput benefit.

    Returns:
        One string per input, in order. ``""`` marks a completion that could not be produced --
        either the model returned nothing usable (``clean_completion`` cut it to empty) or the
        whole chunk failed. Both are treated identically downstream: the candidate is floored, is
        excluded from the ranking, and is still recorded.

    Notes:
        The GPU lock is held across the generate and released between chunks, so an oracle call
        that is still in flight from an earlier depth is never blocked by it.
    """
    n = len(batch_messages)
    out: List[str] = [""] * n
    if n == 0:
        return out

    loop = asyncio.get_running_loop()
    gpu_lock = primitives.gpu_lock()
    stops = list(stop_strings) if stop_strings else None
    chunk = max(1, int(chunk_size))

    for start in range(0, n, chunk):
        sub = list(batch_messages[start:start + chunk])
        async with gpu_lock:
            with _inference_policy(model):
                responses, error = await loop.run_in_executor(
                    None,
                    functools.partial(
                        generate_therapist_batch,
                        model,
                        tokenizer,
                        sub,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_input_tokens=max_input_tokens,
                        stop_strings=stops,
                    ),
                )
        if error is not None or responses is None:
            # The chunk's candidates stay "" -> floored, excluded from the ranking, recorded.
            # Dropping the whole depth instead would freeze every trunk in the chunk over a
            # transient generate failure.
            print(f"{_LOG}  WARNING: branch sampling failed for a chunk of {len(sub)} ({error}); "
                  f"those candidates are treated as degenerate")
            continue
        for j, resp in enumerate(responses):
            out[start + j] = resp or ""
    return out


def _eligible(candidates: Sequence[CandidateScore]) -> List[Tuple[float, int]]:
    """``[(score, index), ...]`` for candidates that can take part in a preference pair.

    Excludes both degenerate completions and oracle failures:

    * a DEGENERATE candidate cleaned to the empty string, so using it as ``rejected`` would ask
      ``DPOTrainer`` for the log-probability of an empty completion. GRPO can floor such a
      candidate because it has a real reward channel; a preference pair has only text, so the
      exclusion is not a policy choice but a shape requirement.
    * an oracle FAILURE has ``score is None``. Treating it as a low score would manufacture a
      preference out of a network error.

    Returned unsorted; callers take ``max``/``min``.
    """
    return [
        (float(c.score), i)
        for i, c in enumerate(candidates)
        if (not c.degenerate) and (c.score is not None)
    ]


def _roles_for(best_idx: Optional[int], worst_idx: Optional[int], emitted: bool, idx: int) -> str:
    """Per-candidate PTO role for the EDA row.

    ``"rejected"`` is what ``EDARecorder.aggregate`` counts as "a preference pair was emitted at
    this branch point", so it is set ONLY when the tau filter actually passed. ``"chosen"`` is set
    for the top-ranked candidate whether or not a pair was emitted, because in greedy mode that
    candidate was appended to the trunk regardless -- it is a real decision, not a hypothetical.
    """
    if best_idx is not None and idx == best_idx:
        return "chosen"
    if emitted and worst_idx is not None and idx == worst_idx:
        return "rejected"
    return "neither"


def _record_branch(
    recorder: Optional[EDARecorder],
    *,
    phase: str,
    iteration: int,
    conversation_id: Any,
    persona_id: Optional[int],
    branch_id: int,
    prefix: str,
    candidates: Sequence[CandidateScore],
    best_idx: Optional[int],
    worst_idx: Optional[int],
    emitted: bool,
) -> None:
    """Buffer ONE branch row: the prefix once, all ``M`` candidates nested under it.

    Every candidate is recorded -- degenerate ones, oracle failures and middle ranks included.
    Those are the rows the EDA needs to answer "how often did the tau filter reject", "does
    look-ahead change the ranking" and "how degenerate is the policy at iteration n"; they exist
    nowhere else once the build is over.

    No-op when the recorder is absent or disabled.
    """
    if recorder is None or not getattr(recorder, "enabled", False):
        return
    recorder.append(build_branch_record(
        phase=phase,
        iteration=iteration,
        conversation_id=conversation_id,
        persona_id=persona_id,
        branch_id=branch_id,
        prefix=prefix,
        candidates=[
            c.to_record(i, role=_roles_for(best_idx, worst_idx, emitted, i))
            for i, c in enumerate(candidates)
        ],
        chosen_idx=best_idx,
    ))


def _patient_prompt_for(permutations: Sequence[Dict[str, str]], persona_id: int) -> str:
    """The persona's patient system prompt, or ``""`` with a warning.

    An empty prompt is not fatal but it IS a silent science failure: the look-ahead rollout would
    continue against a degenerate patient while every score still looks perfectly ordinary. The
    warning is the only signal.
    """
    if 0 <= int(persona_id) < len(permutations):
        prompt = str(permutations[int(persona_id)].get("patient_system_prompt") or "")
        if prompt:
            return prompt
    print(f"{_LOG}  WARNING: no patient_system_prompt for persona {persona_id}; look-ahead would "
          f"roll out against a degenerate patient. Pass the FULL generate_all_permutations() list, "
          f"indexed by persona_id.")
    return ""


# =============================================================================
# STEP-2 PERSISTENCE: pairs.csv (marker) + _progress.json (mid-build snapshot)
# =============================================================================


def pref_config_fingerprint(train_cfg: PTOTrainingConfig, gen_cfg: GenConfig) -> Dict[str, Any]:
    """The knobs a resumed ``_progress.json`` must match before it may be reused.

    ``EXPERIMENT_NAME`` encodes the mode, ``M``, ``MCL`` and ``K`` -- but NOT ``pref_filter_tau``,
    NOT ``num_utterances_for_data`` and NOT ``greedy_trunk_target_len``. A snapshot built under a
    different tau therefore lives in the SAME folder as the current run and would merge silently,
    producing an iteration that is half one configuration and half another with nothing on disk
    recording the split. Guarding on the full set is what makes that impossible.

    Returns:
        A small JSON-serialisable dict. Compared by equality, so key order is irrelevant.
    """
    return {
        "mode": str(train_cfg.pref_tree_mode).strip().lower(),
        "pref_filter_tau": float(train_cfg.pref_filter_tau),
        "num_branches_per_turn": int(train_cfg.num_branches_per_turn),
        "min_conv_length": int(gen_cfg.min_conv_length),
        "num_utterances_for_data": int(gen_cfg.num_utterances_for_data),
        "greedy_trunk_target_len": (
            None if train_cfg.greedy_trunk_target_len is None
            else int(train_cfg.greedy_trunk_target_len)
        ),
        "seed": int(train_cfg.seed),
    }


def _atomic_write_text(path: str, text: str) -> None:
    """Write *text* to *path* atomically (temp file + ``os.replace``).

    The run tree is a Google Drive FUSE mount on Colab, where a killed process mid-write leaves a
    truncated file that ``json.load`` accepts up to the tear. Replacing an already-complete temp
    file makes a partial snapshot impossible.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_pairs_csv(pairs: Sequence[Dict[str, Any]], path: str) -> str:
    """Write the preference pairs to *path* atomically. Returns the path.

    Warning:
        This file is the Step-2 COMPLETION MARKER as well as the audit trail. Writing it declares
        the build finished, so it must be written only after the build actually completed -- never
        as a checkpoint. ``_progress.json`` is the checkpoint.

        An empty ``pairs`` still writes a file, on purpose: the marker means "the build ran", and
        the zero-pairs guard in :func:`run_one_iteration` is what turns that into a loud failure.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    pd.DataFrame(list(pairs)).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def reload_pairs_csv(path: str) -> List[Dict[str, Any]]:
    """Reconstruct the pair list from a completed ``pairs.csv``.

    Returns:
        One dict per row. ``[]`` for an empty or unparseable file, which then trips the zero-pairs
        guard exactly as a fresh build producing nothing would.

    Notes:
        ``keep_default_na=False`` is required, not stylistic: without it pandas turns an empty
        ``chosen``/``rejected`` string into ``NaN`` (a float), and DPO would be handed a float
        where it expects text. The two score columns and ``persona_id`` are coerced back to
        numbers; ``conversation_id`` stays a string (``"pers07"``).
    """
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return []

    records = df.to_dict("records")
    for rec in records:
        for key in ("chosen_score", "rejected_score"):
            if rec.get(key, "") != "":
                try:
                    rec[key] = float(rec[key])
                except (TypeError, ValueError):
                    pass
        for key in ("persona_id", BRANCH_COLUMN_GREEDY, BRANCH_COLUMN_INDEPENDENT):
            if rec.get(key, "") != "":
                try:
                    rec[key] = int(float(rec[key]))
                except (TypeError, ValueError):
                    pass
        if "conversation_id" in rec:
            rec["conversation_id"] = str(rec["conversation_id"])
    return records


def _write_progress(
    progress_path: Optional[str],
    *,
    mode: str,
    iteration: int,
    fingerprint: Dict[str, Any],
    persona_ids: Sequence[int],
    pairs: Sequence[Dict[str, Any]],
    eda_records: Sequence[Dict[str, Any]],
    depth: int = 0,
    processed_persona_ids: Optional[Sequence[int]] = None,
    trunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Snapshot in-build Step-2 state so an interrupted build resumes where it stopped.

    Written after every lock-step depth (greedy) or every conversation (independent). Cheap
    relative to the phase it protects -- the build is hours, the snapshot is trunk text plus the
    pairs so far -- and atomic, so a crash during the write cannot corrupt the previous one.

    Notes:
        A snapshot failure is logged and swallowed. Losing the ability to resume costs a rebuild;
        aborting the build to report a failed telemetry write costs the same rebuild plus the work
        already done.

        The snapshot carries the EDA records, which on a ``K=5`` arm are dominated by the per-
        candidate look-ahead tails. ``SAVE_LOOKAHEAD_TRANSCRIPTS=False`` drops those tails at
        APPEND time, so it shrinks this file as well as ``generations.jsonl`` -- it is the size
        lever for both.
    """
    if not progress_path:
        return
    snapshot = {
        "mode": mode,
        "iteration": int(iteration),
        "config_key": fingerprint,
        "persona_ids": sorted(int(p) for p in persona_ids),
        "depth": int(depth),
        "processed_persona_ids": sorted(int(p) for p in (processed_persona_ids or [])),
        "trunks": list(trunks or []),
        "pairs": list(pairs),
        "eda_records": list(eda_records or []),
    }
    try:
        _atomic_write_text(progress_path, json.dumps(to_jsonable(snapshot), ensure_ascii=False))
    except OSError as exc:
        print(f"{_LOG}  WARNING: could not write {os.path.basename(progress_path)} ({exc}); "
              f"an interrupted build will restart from scratch")


def _load_progress(
    progress_path: Optional[str],
    *,
    mode: str,
    iteration: int,
    fingerprint: Dict[str, Any],
    persona_ids: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """Load a Step-2 snapshot iff it belongs to THIS build; otherwise ``None`` (rebuild).

    Guards on mode, iteration, the config fingerprint (see :func:`pref_config_fingerprint`) and
    the persona set. Any mismatch, any read error and any parse error all mean "rebuild": a
    partially-applicable snapshot is worth strictly less than the hours it would corrupt.
    """
    if not progress_path or not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"{_LOG}  WARNING: {os.path.basename(progress_path)} unreadable "
              f"({type(exc).__name__}: {exc}); rebuilding Step 2 from scratch")
        return None
    if not isinstance(snapshot, dict):
        return None

    mismatches: List[str] = []
    if snapshot.get("mode") != mode:
        mismatches.append("mode")
    if int(snapshot.get("iteration", -1)) != int(iteration):
        mismatches.append("iteration")
    if snapshot.get("config_key") != fingerprint:
        mismatches.append("config (tau / M / MCL / target length / seed)")
    if snapshot.get("persona_ids") != sorted(int(p) for p in persona_ids):
        mismatches.append("persona set")
    if mismatches:
        print(f"{_LOG}  WARNING: stale _progress.json ({', '.join(mismatches)}); discarding it and "
              f"rebuilding Step 2 from scratch. Mixing a checkpoint from a different tau into this "
              f"build would put two configurations in one folder.")
        return None
    return snapshot


# =============================================================================
# GREEDY MODE -- the true preference tree, grown lock-step
# =============================================================================


@dataclass
class _Trunk:
    """One live trunk during greedy growth.

    ``conv`` is a real :class:`~core.conversations.ConversationState`, so ``append_turn`` keeps
    ``turns`` and BOTH message-perspective lists in sync as the trunk grows. Re-implementing that
    bookkeeping here is how a patient ends up not seeing what the therapist just said -- a failure
    that reads as a bad model rather than as a bug.
    """

    conv: ConversationState
    patient_system_prompt: str
    pairs: List[Dict[str, Any]] = field(default_factory=list)


def _advance(state: ConversationState, content: str, speaker: str) -> bool:
    """Append one utterance to a trunk. Returns whether the trunk is still growing.

    Mirrors what the conversation loop does to a live conversation: an empty utterance ends the
    trunk rather than padding it with a blank turn, ``SESSION ENDED`` records who ended it and
    keeps whatever text preceded the keyword, and anything else is appended with the speaker
    flipped.
    """
    if not content or not content.strip():
        state.active = False
        return False

    if SESSION_END_KEYWORD in content.upper():
        ended_by, explanation, cleaned = handle_session_end(content, speaker)
        state.session_ended_by = ended_by
        state.session_ended_explanation = explanation.strip()
        state.active = False
        if cleaned and cleaned.strip():
            state.append_turn(speaker, cleaned.strip())
        return False

    state.append_turn(speaker, content)
    state.next_speaker = _NEXT_SPEAKER[speaker]
    return True


def slice_trunk_seeds(
    states: Sequence[ConversationState],
    min_conv_length: int,
    *,
    verbose: bool = True,
) -> List[ConversationState]:
    """Slice the first ``min_conv_length`` utterances off each step-1 conversation as trunk seeds.

    There is no separate prefix-generation pass: the seeds reuse the eval conversations' openings
    and then diverge the moment the first best-of-``M`` is appended. That is a deliberate saving
    (a prefix pass would be a second full generate) and a deliberate control -- every trunk starts
    from an opening the current policy actually produced.

    Args:
        states: finished step-1 conversations.
        min_conv_length: ``MCL``. Must be EVEN, because the sliced prefix has to end on a PATIENT
            turn so the trunk's first branch point is a therapist turn.
        verbose: print a line per skipped conversation.

    Returns:
        Fresh :class:`~core.conversations.ConversationState` seeds with ``turns`` copied,
        ``next_speaker="therapist"`` and EMPTY message lists -- the grower rebuilds those from
        ``turns`` (see the module docstring).

    Raises:
        ValueError: *min_conv_length* is odd. ``core.config.validate_config`` already refuses that
            combination; this is the second gate, because an odd MCL here does not fail -- it
            silently seeds every trunk one turn out of phase.

    Notes:
        A conversation is skipped when it did not grow PAST ``min_conv_length``: its prefix would
        end exactly where the conversation ended, so the trunk would have nowhere to grow. The
        seeds are deep copies, so the step-1 conversations stay untouched -- they are the frozen
        eval data for ``model_iter_{n-1}``.
    """
    mcl = int(min_conv_length)
    if mcl % 2 != 0:
        raise ValueError(
            f"slice_trunk_seeds: min_conv_length={mcl} is odd. The greedy trunk seed is sliced off "
            f"the step-1 conversation and must end on a PATIENT turn, so MCL must be even."
        )

    seeds: List[ConversationState] = []
    n_short = 0
    for state in states:
        if state.failed or len(state.turns) <= mcl:
            n_short += 1
            continue
        prefix = state.turns[:mcl]
        if prefix[-1]["role"] != ROLE_PATIENT:
            if verbose:
                print(f"{_LOG}  WARNING: persona {state.persona_id}: the MCL prefix ends on "
                      f"{prefix[-1]['role']!r}, not a patient turn; skipped as a seed")
            continue
        seeds.append(ConversationState(
            persona_id=int(state.persona_id),
            turns=[dict(t) for t in prefix],
            messages_therapist=[],
            messages_patient=[],
            active=True,
            next_speaker=ROLE_THERAPIST,
        ))

    if verbose:
        print(f"{_LOG}Sliced {len(seeds)} trunk seeds of {mcl} utterances from {len(states)} "
              f"conversations ({n_short} failed or too short to grow past MCL)")
    return seeds


async def _grow_therapist_depth(
    active: Sequence[_Trunk],
    *,
    model,
    tokenizer,
    client,
    sp_therapist: str,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    recorder: Optional[EDARecorder],
    iteration: int,
    lookahead_state: Optional[LookaheadState],
) -> None:
    """One branching (therapist) depth across every live trunk, in lock-step.

    Branch ``M`` completions per trunk, score all ``len(active) * M`` of them in ONE call through
    the shared reward path, then per trunk: append the best completion (always, so the trunk
    advances) and emit a ``(chosen, rejected)`` pair when the gap clears ``pref_filter_tau``.

    Notes:
        **The winner is appended even when no pair is emitted.** A tie within tau is a statement
        about the two candidates, not about the trunk; freezing the trunk there would make trunk
        length a function of tau and quietly shorten the context of every arm with a larger one.

        The prompt is snapshotted BEFORE the winner is appended -- it is the context the chosen
        completion was sampled from, so taking it afterwards would train the model to produce a
        turn it has already seen.

        Trunks with no scorable candidate at all (every completion degenerate, or every oracle
        call failed) are frozen: there is nothing to append, and appending an empty turn would
        corrupt the trunk for every later depth.
    """
    M = int(train_cfg.num_branches_per_turn)

    # Prefix per trunk, computed once and reused for all M candidates. `turns` rather than
    # `messages_therapist`: turns carry their own speaker names and cannot be mislabelled.
    transcripts = [format_conversation_for_oracle(t.conv.turns) for t in active]
    flat_messages = [t.conv.messages_therapist for t in active for _ in range(M)]

    completions = await _sample_completions_batch(
        model,
        tokenizer,
        primitives,
        flat_messages,
        temperature=train_cfg.branch_sample_temperature,
        max_tokens=train_cfg.branch_max_tokens,
        max_input_tokens=gen_cfg.therapist_max_input_tokens,
        stop_strings=gen_cfg.stop_strings,
        chunk_size=gen_cfg.conversation_batch_size,
    )

    # One scoring call for the whole depth. Degenerate ("") candidates cost nothing here: the
    # shared path excludes them from both the rollout and the oracle batch and floors them.
    scored = await score_pref_candidates(
        model,
        tokenizer,
        client,
        oracle_cfg,
        la_cfg,
        primitives,
        transcripts=[transcripts[i // M] for i in range(len(flat_messages))],
        completions=completions,
        sp_therapist=sp_therapist,
        sp_patient_list=[active[i // M].patient_system_prompt for i in range(len(flat_messages))],
        enforce_success_ratio=True,
        lookahead_state=lookahead_state,
    )

    for ti, trunk in enumerate(active):
        block = scored[ti * M:(ti + 1) * M]
        ranked = _eligible(block)

        if not ranked:
            # Nothing scorable: every completion was degenerate or every oracle call failed.
            # Record the branch anyway -- an all-degenerate branch point is itself a finding --
            # then freeze, because appending an empty turn would corrupt every later depth.
            _record_branch(
                recorder,
                phase=PHASE_TREE, iteration=iteration,
                conversation_id=trunk.conv.conversation_id,
                persona_id=trunk.conv.persona_id,
                branch_id=trunk.conv.n_utterances,
                prefix=transcripts[ti], candidates=block,
                best_idx=None, worst_idx=None, emitted=False,
            )
            trunk.conv.active = False
            continue

        best_score, best_idx = max(ranked)
        worst_score, worst_idx = min(ranked)

        prompt = build_truncated_training_prompt(
            trunk.conv.turns, sp_therapist, tokenizer, gen_cfg.max_prompt_tokens,
        )
        branch_depth = trunk.conv.n_utterances
        emitted = (
            len(ranked) >= 2
            and (best_score - worst_score) > float(train_cfg.pref_filter_tau)
            and prompt is not None
        )

        _record_branch(
            recorder,
            phase=PHASE_TREE,
            iteration=iteration,
            conversation_id=trunk.conv.conversation_id,
            persona_id=trunk.conv.persona_id,
            branch_id=branch_depth,
            prefix=transcripts[ti],
            candidates=block,
            best_idx=best_idx,
            worst_idx=worst_idx,
            emitted=emitted,
        )

        if emitted:
            trunk.pairs.append({
                "prompt": prompt,
                "chosen": block[best_idx].completion,
                "rejected": block[worst_idx].completion,
                "chosen_score": best_score,
                "rejected_score": worst_score,
                "conversation_id": trunk.conv.conversation_id,
                "persona_id": int(trunk.conv.persona_id),
                BRANCH_COLUMN_GREEDY: int(branch_depth),
            })

        # The greedy feedback: the chosen completion becomes the context of the NEXT branch point.
        _advance(trunk.conv, block[best_idx].completion, ROLE_THERAPIST)


async def _grow_patient_depth(
    active: Sequence[_Trunk],
    *,
    client,
    patient_binding,
    primitives: AsyncPrimitives,
    gen_cfg: GenConfig,
    patient_seed: Optional[int] = None,
) -> None:
    """One patient depth across every live trunk -- a single batched round of API calls.

    A trunk whose patient call exhausted its retries is frozen and marked ``failed``: it is
    missing an utterance it should have had, so the pairs already collected from it stay valid
    while nothing further is grown on a conversation with a hole in it.
    """
    responses = await generate_patient_batch(
        client,
        patient_binding,
        [t.conv.messages_patient for t in active],
        primitives.patient_sem(),
        max_tokens=gen_cfg.max_tokens_per_response,
        temperature=gen_cfg.temperature_patient,
        seed=patient_seed,
    )
    for trunk, response in zip(active, responses):
        if isinstance(response, BaseException) or not response:
            print(f"{_LOG}  Patient call failed for persona {trunk.conv.persona_id} "
                  f"({response}); freezing that trunk")
            trunk.conv.active = False
            trunk.conv.failed = True
            continue
        _advance(trunk.conv, response, ROLE_PATIENT)


async def grow_preference_trees_batch(
    seed_states: Sequence[ConversationState],
    permutations: Sequence[Dict[str, str]],
    *,
    model,
    tokenizer,
    client,
    sp_therapist: str,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    patient_binding=None,
    recorder: Optional[EDARecorder] = None,
    iteration: int = 0,
    progress_path: Optional[str] = None,
    lookahead_state: Optional[LookaheadState] = None,
    patient_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Grow every trunk lock-step and return the preference pairs they emitted (greedy mode).

    Args:
        seed_states: trunk seeds from :func:`slice_trunk_seeds` -- MCL-length prefixes ending on a
            patient turn.
        permutations: the FULL ``generate_all_permutations()`` list, indexed by ``persona_id``.
            Passing a shuffled subset silently pairs each trunk with another patient's persona.
        model: the CURRENT policy. Both the branch samples and the look-ahead rollout come from it,
            which is what makes the pairs a statement about where this policy's openings lead.
        sp_therapist: the therapist system prompt, rendered into every prompt and every rollout.
        patient_binding: which model plays the patient during trunk growth. Defaults to
            ``la_cfg.patient_binding``, which is the binding the conversations were generated with.
        recorder: per-iteration :class:`~core.recorder.EDARecorder`, or ``None``.
        progress_path: ``iteration_N/pref_pairs/_progress.json``. When given, the build snapshots
            after every depth and resumes from a matching snapshot.
        lookahead_state: ONE :class:`~core.lookahead.LookaheadState` for the whole build. It is what
            makes the look-ahead's OOM sub-batch halving sticky across depths -- omit it and the
            OOM is re-paid at every one of the ~20 depths an iteration runs.
        patient_seed: forwarded to the patient calls; servers that ignore it are unaffected.

    Returns:
        A flat list of pair dicts (the same shape the independent path returns, except for the
        branch-position column -- see :data:`BRANCH_COLUMN_GREEDY`).

    Raises:
        RuntimeError: propagated from the shared scoring path when the oracle success rate falls
            below ``oracle_cfg.min_success_ratio``. The last depth's ``_progress.json`` survives,
            so fixing the grader and re-running resumes rather than restarting.

    Notes:
        **The message lists are rebuilt from ``turns`` here.** Seeds arrive with empty ones (see
        the module docstring); without this the trunks would grow with no system prompt and no
        history.

        Growth stops when every trunk is frozen or has reached the target length --
        ``gen_cfg.num_utterances_for_data``, lowered by ``train_cfg.greedy_trunk_target_len`` when
        that is set. Lowering it is a speed lever AND a science change (shallower trunks mean
        shallower context at every branch point) and it is not encoded in ``EXPERIMENT_NAME``, so
        it only survives in ``run_metadata.json``.
    """
    binding = patient_binding if patient_binding is not None else la_cfg.patient_binding

    trunks: List[_Trunk] = []
    for seed in seed_states:
        if seed.failed or len(seed.turns) < 2 or seed.next_speaker != ROLE_THERAPIST:
            continue
        patient_sp = _patient_prompt_for(permutations, seed.persona_id)
        # Rebuilt here, not by the caller: generation frees these on every completed state, so a
        # seed sliced off a step-1 conversation always arrives with empty message lists.
        seed.messages_therapist = turns_to_messages(seed.turns, sp_therapist)
        seed.messages_patient = turns_to_patient_messages(seed.turns, patient_sp)
        trunks.append(_Trunk(conv=seed, patient_system_prompt=patient_sp))

    if not trunks:
        print(f"{_LOG}WARNING: no usable trunk seeds; this iteration will produce 0 pref pairs")
        return []

    fingerprint = pref_config_fingerprint(train_cfg, gen_cfg)
    persona_ids = sorted(int(t.conv.persona_id) for t in trunks)
    carried: List[Dict[str, Any]] = []
    depth = 0

    snapshot = _load_progress(
        progress_path, mode=_MODE_GREEDY, iteration=iteration,
        fingerprint=fingerprint, persona_ids=persona_ids,
    )
    if snapshot is not None:
        saved = {int(t["persona_id"]): t for t in snapshot.get("trunks", [])}
        n_restored = 0
        for trunk in trunks:
            state = saved.get(int(trunk.conv.persona_id))
            if state is None:
                continue
            trunk.conv.turns = [dict(t) for t in state["turns"]]
            trunk.conv.next_speaker = state["next_speaker"]
            trunk.conv.active = bool(state["active"])
            trunk.conv.session_ended_by = state.get("session_ended_by", "")
            trunk.conv.session_ended_explanation = state.get("session_ended_explanation", "")
            trunk.conv.messages_therapist = turns_to_messages(trunk.conv.turns, sp_therapist)
            trunk.conv.messages_patient = turns_to_patient_messages(
                trunk.conv.turns, trunk.patient_system_prompt)
            n_restored += 1
        carried = list(snapshot.get("pairs", []))
        depth = int(snapshot.get("depth", 0))
        if recorder is not None and getattr(recorder, "enabled", False):
            recorder.records = list(snapshot.get("eda_records", []))
        print(f"{_LOG}[resume] greedy: restored {n_restored} trunks at depth {depth}, "
              f"{len(carried)} pairs carried")

    target_len = int(gen_cfg.num_utterances_for_data)
    if train_cfg.greedy_trunk_target_len is not None:
        target_len = min(target_len, int(train_cfg.greedy_trunk_target_len))

    while True:
        active = [t for t in trunks if t.conv.active and t.conv.n_utterances < target_len]
        if not active:
            break

        speaker = active[0].conv.next_speaker
        off_beat = [t for t in active if t.conv.next_speaker != speaker]
        if off_beat:
            # Retire the stragglers rather than aborting: their pairs so far are valid training
            # data, and one trunk drifting off cadence must not cost the other 95.
            print(f"{_LOG}  WARNING: trunk desync at depth {depth} (expected {speaker!r}); "
                  f"freezing personas {[t.conv.persona_id for t in off_beat]}")
            for trunk in off_beat:
                trunk.conv.active = False
            active = [t for t in active if t.conv.active]
            if not active:
                break

        if speaker == ROLE_THERAPIST:
            await _grow_therapist_depth(
                active,
                model=model, tokenizer=tokenizer, client=client,
                sp_therapist=sp_therapist,
                oracle_cfg=oracle_cfg, la_cfg=la_cfg, primitives=primitives,
                train_cfg=train_cfg, gen_cfg=gen_cfg,
                recorder=recorder, iteration=iteration,
                lookahead_state=lookahead_state,
            )
        else:
            await _grow_patient_depth(
                active,
                client=client, patient_binding=binding, primitives=primitives,
                gen_cfg=gen_cfg, patient_seed=patient_seed,
            )

        depth += 1
        all_pairs = carried + [p for t in trunks for p in t.pairs]
        _write_progress(
            progress_path,
            mode=_MODE_GREEDY, iteration=iteration, fingerprint=fingerprint,
            persona_ids=persona_ids, pairs=all_pairs,
            eda_records=(recorder.records if (recorder is not None
                                              and getattr(recorder, "enabled", False)) else []),
            depth=depth,
            trunks=[{
                "persona_id": int(t.conv.persona_id),
                "turns": t.conv.turns,
                "next_speaker": t.conv.next_speaker,
                "active": bool(t.conv.active),
                "session_ended_by": t.conv.session_ended_by,
                "session_ended_explanation": t.conv.session_ended_explanation,
            } for t in trunks],
        )

        if gen_cfg.verbose:
            print(f"{_LOG}[tree] depth {depth} ({speaker}): {len(active)} active trunks, "
                  f"{len(all_pairs)} pairs so far")

    return carried + [p for t in trunks for p in t.pairs]


# =============================================================================
# INDEPENDENT MODE -- branch a pre-recorded conversation, no feedback
# =============================================================================


async def build_pref_pairs_for_conversation(
    state: ConversationState,
    permutations: Sequence[Dict[str, str]],
    *,
    model,
    tokenizer,
    client,
    sp_therapist: str,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    recorder: Optional[EDARecorder] = None,
    iteration: int = 0,
    lookahead_state: Optional[LookaheadState] = None,
) -> List[Dict[str, Any]]:
    """Every preference pair one PRE-RECORDED conversation yields (independent mode).

    Branch at each patient turn whose conversation-so-far is at least ``MCL`` utterances and which
    is not the conversation's final turn: sample ``M`` completions from that fixed prefix, score
    them through the shared look-ahead + oracle path, and emit a pair when the best-worst gap
    clears ``pref_filter_tau``.

    Args:
        state: a finished step-1 conversation. It is READ, never mutated -- unlike greedy, nothing
            is fed back, so the conversation stays exactly the eval data it also serves as.
        permutations: the FULL permutation list, indexed by ``state.persona_id``.
        lookahead_state: the iteration's :class:`~core.lookahead.LookaheadState`. Pass the same one
            for every conversation so an OOM sub-batch halving is paid once, not once per
            conversation.

    Returns:
        Pair dicts carrying :data:`BRANCH_COLUMN_INDEPENDENT` (the patient turn's index) instead of
        greedy's trunk depth.

    Notes:
        **The final patient turn is skipped.** It has no following therapist turn in the recorded
        conversation, so there is no branch point there to anchor a pair -- only an artificial
        continuation past the end of a conversation that already finished.

        The prompt is built with the SAME truncator greedy uses, so a pair from either mode enters
        ``DPOTrainer`` with an identical context budget. A branch point whose prompt will not fit
        is skipped BEFORE the expensive sampling and scoring, not after.
    """
    turns = state.turns
    if not turns:
        return []

    patient_sp = _patient_prompt_for(permutations, state.persona_id)
    M = int(train_cfg.num_branches_per_turn)
    mcl = int(gen_cfg.min_conv_length)
    pairs: List[Dict[str, Any]] = []

    for i, turn in enumerate(turns):
        if turn["role"] != ROLE_PATIENT:
            continue
        if (i + 1) < mcl:                       # i+1 == utterances in the conversation-so-far
            continue
        if (i + 1) >= len(turns):               # final turn: no branch point follows it
            continue

        partial = turns[:i + 1]
        prompt = build_truncated_training_prompt(
            partial, sp_therapist, tokenizer, gen_cfg.max_prompt_tokens,
        )
        if prompt is None:
            continue

        prefix_messages = turns_to_messages(partial, sp_therapist)
        transcript = format_conversation_for_oracle(partial)

        completions = await _sample_completions_batch(
            model, tokenizer, primitives, [prefix_messages] * M,
            temperature=train_cfg.branch_sample_temperature,
            max_tokens=train_cfg.branch_max_tokens,
            max_input_tokens=gen_cfg.therapist_max_input_tokens,
            stop_strings=gen_cfg.stop_strings,
            chunk_size=gen_cfg.conversation_batch_size,
        )

        scored = await score_pref_candidates(
            model, tokenizer, client, oracle_cfg, la_cfg, primitives,
            transcripts=[transcript] * M,
            completions=completions,
            sp_therapist=sp_therapist,
            sp_patient_list=[patient_sp] * M,
            enforce_success_ratio=True,
            lookahead_state=lookahead_state,
        )

        ranked = _eligible(scored)
        best_score, best_idx = (max(ranked) if ranked else (0.0, None))
        worst_score, worst_idx = (min(ranked) if ranked else (0.0, None))
        emitted = len(ranked) >= 2 and (best_score - worst_score) > float(train_cfg.pref_filter_tau)

        _record_branch(
            recorder,
            phase=PHASE_INDEPENDENT,
            iteration=iteration,
            conversation_id=state.conversation_id,
            persona_id=state.persona_id,
            branch_id=i,
            prefix=transcript,
            candidates=scored,
            best_idx=best_idx,
            worst_idx=worst_idx,
            emitted=emitted,
        )

        if emitted:
            pairs.append({
                "prompt": prompt,
                "chosen": scored[best_idx].completion,
                "rejected": scored[worst_idx].completion,
                "chosen_score": best_score,
                "rejected_score": worst_score,
                "conversation_id": state.conversation_id,
                "persona_id": int(state.persona_id),
                BRANCH_COLUMN_INDEPENDENT: int(i),
            })

    return pairs


async def build_pref_pairs_independent(
    states: Sequence[ConversationState],
    permutations: Sequence[Dict[str, str]],
    *,
    model,
    tokenizer,
    client,
    sp_therapist: str,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    recorder: Optional[EDARecorder] = None,
    iteration: int = 0,
    progress_path: Optional[str] = None,
    lookahead_state: Optional[LookaheadState] = None,
) -> List[Dict[str, Any]]:
    """Run :func:`build_pref_pairs_for_conversation` over every usable conversation.

    Conversations are processed sequentially -- the branch sampling is GPU-bound, so overlapping
    them would only contend for the same lock -- while the scoring inside each one is batched.

    Resume-aware: with *progress_path* set, a snapshot is written after every conversation and a
    matching one lets a restart skip the conversations already done and carry their pairs forward.
    """
    usable = [s for s in states if not (s.failed or len(s.turns) <= 1)]
    persona_ids = sorted(int(s.persona_id) for s in usable)
    fingerprint = pref_config_fingerprint(train_cfg, gen_cfg)

    all_pairs: List[Dict[str, Any]] = []
    processed: set = set()

    snapshot = _load_progress(
        progress_path, mode=_MODE_INDEPENDENT, iteration=iteration,
        fingerprint=fingerprint, persona_ids=persona_ids,
    )
    if snapshot is not None:
        all_pairs = list(snapshot.get("pairs", []))
        processed = {int(p) for p in snapshot.get("processed_persona_ids", [])}
        if recorder is not None and getattr(recorder, "enabled", False):
            recorder.records = list(snapshot.get("eda_records", []))
        print(f"{_LOG}[resume] independent: {len(processed)} conversations already done, "
              f"{len(all_pairs)} pairs carried")

    for state in usable:
        if int(state.persona_id) in processed:
            continue
        pairs = await build_pref_pairs_for_conversation(
            state, permutations,
            model=model, tokenizer=tokenizer, client=client,
            sp_therapist=sp_therapist,
            oracle_cfg=oracle_cfg, la_cfg=la_cfg, primitives=primitives,
            train_cfg=train_cfg, gen_cfg=gen_cfg,
            recorder=recorder, iteration=iteration,
            lookahead_state=lookahead_state,
        )
        all_pairs.extend(pairs)
        processed.add(int(state.persona_id))

        _write_progress(
            progress_path,
            mode=_MODE_INDEPENDENT, iteration=iteration, fingerprint=fingerprint,
            persona_ids=persona_ids, pairs=all_pairs,
            eda_records=(recorder.records if (recorder is not None
                                              and getattr(recorder, "enabled", False)) else []),
            processed_persona_ids=sorted(processed),
        )
        if gen_cfg.verbose:
            print(f"{_LOG}[indep] {state.conversation_id}: {len(pairs)} pair(s) "
                  f"({len(all_pairs)} total)")

    return all_pairs


# =============================================================================
# THE STEP-2 DISPATCHER
# =============================================================================


async def build_pref_pairs_async(
    states: Sequence[ConversationState],
    permutations: Sequence[Dict[str, str]],
    *,
    model,
    tokenizer,
    client,
    sp_therapist: str,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    patient_binding=None,
    recorder: Optional[EDARecorder] = None,
    iteration: int = 0,
    progress_path: Optional[str] = None,
    lookahead_state: Optional[LookaheadState] = None,
    patient_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build this iteration's preference pairs, dispatching on ``train_cfg.pref_tree_mode``.

    ``greedy`` slices trunk seeds off *states* and grows them; ``independent`` branches *states*
    themselves. The async body; :func:`build_pref_pairs` is the synchronous entry point.

    Raises:
        ValueError: unknown ``pref_tree_mode``. ``core.config.validate_config`` catches this at
            config time; the second check is here because a mode typo that silently fell through
            to greedy would produce an arm whose name says ``_PTindep``.
    """
    mode = str(train_cfg.pref_tree_mode).strip().lower()

    if mode == _MODE_GREEDY:
        seeds = slice_trunk_seeds(states, gen_cfg.min_conv_length, verbose=gen_cfg.verbose)
        gc.collect()
        torch.cuda.empty_cache()
        return await grow_preference_trees_batch(
            seeds, permutations,
            model=model, tokenizer=tokenizer, client=client,
            sp_therapist=sp_therapist,
            oracle_cfg=oracle_cfg, la_cfg=la_cfg, primitives=primitives,
            train_cfg=train_cfg, gen_cfg=gen_cfg, patient_binding=patient_binding,
            recorder=recorder, iteration=iteration, progress_path=progress_path,
            lookahead_state=lookahead_state, patient_seed=patient_seed,
        )

    if mode in (_MODE_INDEPENDENT, "indep"):
        return await build_pref_pairs_independent(
            states, permutations,
            model=model, tokenizer=tokenizer, client=client,
            sp_therapist=sp_therapist,
            oracle_cfg=oracle_cfg, la_cfg=la_cfg, primitives=primitives,
            train_cfg=train_cfg, gen_cfg=gen_cfg,
            recorder=recorder, iteration=iteration, progress_path=progress_path,
            lookahead_state=lookahead_state,
        )

    raise ValueError(
        f"build_pref_pairs: pref_tree_mode={train_cfg.pref_tree_mode!r} is not "
        f"{_MODE_GREEDY!r} or {_MODE_INDEPENDENT!r}"
    )


def build_pref_pairs(*args, **kwargs) -> List[Dict[str, Any]]:
    """Synchronous entry point for :func:`build_pref_pairs_async`.

    Same arguments, same return value. Runs the whole build on ONE event loop via
    ``core.concurrency.run_async``, which works from a plain script and from inside a live Jupyter
    loop alike -- so the notebook's orchestration cell can call it directly.

    Notes:
        Code already inside a coroutine must await :func:`build_pref_pairs_async` instead: wrapping
        it in ``run_async`` from there would spin up a nested loop on a worker thread for nothing.
    """
    return run_async(build_pref_pairs_async(*args, **kwargs))


# =============================================================================
# DPO: DATASET + CONFIG + THE TRAINING PHASE
# =============================================================================


def build_lora_config(train_cfg: PTOTrainingConfig):
    """A ``peft.LoraConfig`` from the frozen training config.

    Needed because ``core.policy.resolve_start_state`` returns a BARE base model on a fresh start
    (case A) -- there is no adapter to load yet, so the LoRA has to be attached by whoever trains
    first. Hand the result to :func:`run_training_phase` as ``lora_config=``; it is ignored when
    the policy is already PEFT-wrapped (every resume, and every iteration after the first).
    """
    from peft import LoraConfig

    return LoraConfig(
        r=int(train_cfg.lora_r),
        lora_alpha=int(train_cfg.lora_alpha),
        lora_dropout=float(train_cfg.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(train_cfg.lora_target_modules),
    )


def build_dpo_dataset(
    pairs: Sequence[Dict[str, Any]],
    *,
    eval_split_ratio: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Split the pairs by CONVERSATION and return ``(train_dataset, eval_dataset)``.

    Args:
        pairs: pair dicts from the build (or reloaded from ``pairs.csv``). Only ``prompt`` /
            ``chosen`` / ``rejected`` / ``conversation_id`` are read.
        eval_split_ratio: fraction of CONVERSATIONS (not pairs) held out.
        seed: shuffles the conversation order, so the split is reproducible across a resume.

    Returns:
        Two HuggingFace ``Dataset`` objects with exactly the columns ``DPOTrainer`` consumes:
        ``prompt``, ``chosen``, ``rejected``.

    Raises:
        ValueError: *pairs* is empty, or every conversation landed in eval.

    Notes:
        **The split is conversation-level, not pair-level.** Greedy emits many pairs from one
        trunk, sharing a prefix that grows monotonically, so a pair-level split would put a
        prompt's own prefix in train and its continuation in eval -- an eval loss measuring
        memorisation. At least one conversation is always kept in train, so a tiny smoke run
        cannot route its only conversation to eval and leave nothing to train on.
    """
    if not pairs:
        raise ValueError(
            "build_dpo_dataset received 0 preference pairs. Every branch point either tied within "
            "PREF_FILTER_TAU or was filtered out by MIN_CONV_LENGTH."
        )

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for pair in pairs:
        groups.setdefault(pair.get("conversation_id"), []).append(pair)

    conv_ids = sorted(groups, key=str)
    rng = random.Random(int(seed))
    rng.shuffle(conv_ids)

    n_eval = (
        min(max(1, int(len(conv_ids) * float(eval_split_ratio))), len(conv_ids) - 1)
        if len(conv_ids) >= 2 else 0
    )
    eval_ids = conv_ids[:n_eval]
    train_ids = conv_ids[n_eval:]

    train_pairs = [p for cid in train_ids for p in groups[cid]]
    eval_pairs = [p for cid in eval_ids for p in groups[cid]]
    if not train_pairs:
        raise ValueError(
            f"All {len(pairs)} preference pairs landed in the eval split "
            f"({len(conv_ids)} conversations, {n_eval} held out), leaving nothing to train on. "
            f"Raise NUM_CONVERSATIONS_PER_ITER or lower EVAL_SPLIT_RATIO."
        )

    def _to_dataset(rows: Sequence[Dict[str, Any]]) -> Dataset:
        return Dataset.from_dict({
            "prompt": [str(r["prompt"]) for r in rows],
            "chosen": [str(r["chosen"]) for r in rows],
            "rejected": [str(r["rejected"]) for r in rows],
        })

    if verbose:
        print(f"{_LOG}Conversations: {len(train_ids)} train / {len(eval_ids)} eval "
              f"(grouped split, no prefix leakage)")
        print(f"{_LOG}Preference pairs: {len(train_pairs)} train / {len(eval_pairs)} eval")
    return _to_dataset(train_pairs), _to_dataset(eval_pairs)


def build_dpo_config(
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    *,
    output_dir: str,
    num_train_pairs: int,
    hub_model_id: str = "",
    has_eval: bool = True,
) -> DPOConfig:
    """Assemble the ``DPOConfig`` for one iteration.

    Args:
        output_dir: the HF Trainer ``output_dir`` -- ``iteration_N/training``.
        num_train_pairs: size of the train split, used only to turn ``warmup_steps_ratio`` into a
            step count.
        hub_model_id: repo id to stamp into the config; pushing is done explicitly by
            :func:`push_adapter_to_hub`, never by the Trainer.
        has_eval: False disables per-epoch evaluation. An empty eval dataset with
            ``eval_strategy="epoch"`` raises inside the Trainer at the END of the first epoch --
            after the expensive part.

    Returns:
        A ``trl.DPOConfig``.

    Warning:
        **``max_length`` is the only length cap TRL 1.4.0 has.** ``max_prompt_length`` and
        ``max_completion_length`` were removed, and the single remaining cap truncates with
        ``truncation_mode='keep_start'``, which drops the END -- the response. It is set here to
        ``max_prompt_tokens + max_completion_length`` and every prompt is pre-capped to
        ``max_prompt_tokens`` by ``build_truncated_training_prompt``, so the two together make
        ``keep_start`` unreachable. Changing either half re-arms the failure, and it is silent:
        the pairs still train, on empty completions.

        **``per_device_train_batch_size`` is the memory lever, and 2 is not a placeholder.** DPO
        materialises full-SEQUENCE LM-head logits over a 128k vocab for four forward passes
        (chosen/rejected x policy/reference), so the tensor scales with
        ``batch x seq_len x 128k``. Raise ``gradient_accumulation_steps`` instead: 2 x 8 is 16
        pairs per optimizer step, matched to GRPO's 16 prompts per step.

        ``precompute_ref_log_probs`` computes the reference log-probs in a no-grad pre-pass, which
        is semantically identical for DPO (the reference is frozen anyway) and frees the reference
        model's memory during the step. ``gradient_checkpointing`` trades ~30% step time for the
        activation memory; it does NOT shrink the logits tensor, so the two levers are
        complementary rather than alternatives.
    """
    effective_batch = max(1, int(train_cfg.train_batch_size) * int(train_cfg.gradient_accumulation_steps))
    steps_per_epoch = max(1, math.ceil(max(1, int(num_train_pairs)) / effective_batch))
    total_steps = max(1, steps_per_epoch * int(train_cfg.epochs_per_iteration))
    warmup_steps = max(0, math.ceil(total_steps * float(train_cfg.warmup_steps_ratio)))

    tb_dir = os.path.join(output_dir, "tb_logs")
    os.makedirs(tb_dir, exist_ok=True)

    print(f"{_LOG}DPO: {num_train_pairs} pairs, {effective_batch}/step -> ~{total_steps} steps "
          f"({warmup_steps} warmup)")

    return DPOConfig(
        output_dir=output_dir,
        logging_dir=tb_dir,
        hub_model_id=(hub_model_id or None),
        run_name=(hub_model_id or train_cfg.experiment_name),
        per_device_train_batch_size=int(train_cfg.train_batch_size),
        per_device_eval_batch_size=int(train_cfg.eval_batch_size),
        gradient_accumulation_steps=int(train_cfg.gradient_accumulation_steps),
        learning_rate=float(train_cfg.learning_rate),
        num_train_epochs=int(train_cfg.epochs_per_iteration),
        # See the Warning above: this is the ONLY length cap in TRL 1.4.0.
        max_length=int(gen_cfg.max_prompt_tokens) + int(train_cfg.max_completion_length),
        beta=float(train_cfg.dpo_beta),
        loss_type=str(train_cfg.dpo_loss_type),
        bf16=True,
        precompute_ref_log_probs=bool(train_cfg.precompute_ref_log_probs),
        gradient_checkpointing=bool(train_cfg.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=int(train_cfg.seed),
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=int(train_cfg.logging_steps),
        report_to=list(train_cfg.report_to),
        save_strategy=str(train_cfg.save_strategy),
        save_steps=int(train_cfg.save_steps),
        save_total_limit=train_cfg.save_total_limit,
        push_to_hub=False,
        eval_strategy=("epoch" if has_eval else "no"),
    )


def run_training_phase(
    *,
    policy,
    tokenizer,
    dpo_args: DPOConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    train_cfg: PTOTrainingConfig,
    iteration: int,
    start_iteration: int,
    resume_checkpoint: Optional[str] = None,
    lora_config=None,
    tensorboard_log_dir: Optional[str] = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> Tuple[Any, int, float]:
    """Run one iteration's DPO update. Returns ``(updated_policy, step_delta, seconds)``.

    Args:
        policy: the current policy. PEFT-wrapped from iteration 2 onward and on every resume; bare
            on a fresh start, in which case *lora_config* is what attaches the adapter.
        dpo_args: from :func:`build_dpo_config`.
        eval_dataset: may be empty; ``build_dpo_config(has_eval=False)`` must match.
        resume_checkpoint: an HF ``checkpoint-*`` path from ``core.policy.resolve_start_state``.
            **Applied only when ``iteration == start_iteration``** -- it describes the FIRST
            iteration this process runs, and handing it to the next one would resume iteration
            ``n+1`` from iteration ``n``'s optimizer state.
        tensorboard_log_dir: normally ``iteration_N/training/tb_logs``.
        callbacks: extra ``TrainerCallback``s.

    Returns:
        ``updated_policy`` is what TRL hands back -- possibly a NEW wrapper. The caller must rebind
        its own policy variable from it.

    Notes:
        ``patch_generate`` is re-applied on BOTH sides of ``train()``: the trainer installs a fresh
        ``generate`` when it builds its wrapper, and ``train()`` may rebuild it again. Without it,
        ``stop_strings`` is silently inert at the next generation phase and the policy self-plays
        ChatML into the saved conversations.

        Unlike GRPO, PTO's EDA rows are all produced BEFORE training starts (during the preference
        build), so there is no recorder snapshot to keep aligned with a mid-training checkpoint --
        the HF fast-forward-on-resume hazard does not apply here.
    """
    started = time.time()
    policy.config.use_cache = False
    policy.train()

    already_peft = isinstance(policy, PeftModel) or hasattr(policy, "peft_config")
    peft_cfg = None if already_peft else lora_config
    if peft_cfg is None and not already_peft:
        print(f"{_LOG}WARNING: the policy is not PEFT-wrapped and no lora_config was given; "
              f"DPOTrainer will full-finetune the base model.")

    if tensorboard_log_dir and "tensorboard" in tuple(train_cfg.report_to):
        # Must precede the Trainer: TensorBoardCallback reads the env var in __init__.
        setup_tensorboard_logging(tensorboard_log_dir)

    trainer = DPOTrainer(
        model=policy,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=(eval_dataset if (eval_dataset is not None and len(eval_dataset) > 0) else None),
        peft_config=peft_cfg,
        callbacks=list(callbacks) if callbacks else None,
    )

    if tensorboard_log_dir and "tensorboard" in tuple(train_cfg.report_to):
        patch_trainer_tensorboard_callback(trainer, tensorboard_log_dir)

    patch_generate(trainer.model, tokenizer)

    resume = resume_checkpoint if (iteration == start_iteration and resume_checkpoint) else None
    if resume:
        print(f"{_LOG}Resuming from HF checkpoint: {os.path.basename(resume)}")
    trainer.train(resume_from_checkpoint=resume)

    patch_generate(trainer.model, tokenizer)

    updated = trainer.model
    resumed_steps = int(os.path.basename(resume).split("-")[-1]) if resume else 0
    step_delta = max(0, int(trainer.state.global_step) - resumed_steps)
    elapsed = time.time() - started

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"{_LOG}Training complete in {elapsed:.1f}s ({step_delta} optimizer steps)")
    return updated, step_delta, elapsed


# =============================================================================
# GENERATION PHASE
# =============================================================================


def run_generation_phase(
    *,
    policy,
    tokenizer,
    client,
    patient_binding,
    primitives: AsyncPrimitives,
    permutations: Sequence[Dict[str, str]],
    persona_ids: Sequence[int],
    sp_therapist: str,
    therapist_init_utterance: str,
    gen_cfg: GenConfig,
    save_dir: Optional[str],
    patient_seed: Optional[int] = None,
    num_utterances: Optional[int] = None,
) -> Tuple[List[ConversationState], float, float]:
    """Simulate one conversation per requested persona. Returns ``(states, seconds, avg_length)``.

    Args:
        permutations: the FULL permutation list -- it is indexed by persona id, never iterated.
        persona_ids: which personas to run, in processing order. This is where the per-iteration
            shuffle belongs; it never reaches a filename (files are ``pers<PID>.csv``).
        save_dir: ``conversations/<EXP_NAME>/model_iter_<N>``. Personas already on disk there are
            skipped, which is what makes a killed generation phase resume for free.
        num_utterances: overrides ``gen_cfg.num_utterances_for_data``.

    Notes:
        The policy is put in eval + KV-cache mode for the pass. Generation without a KV cache is
        the training setting and is roughly an order of magnitude slower.
    """
    started = time.time()
    policy.eval()
    policy.config.use_cache = True

    states = generate_all_conversations(
        policy,
        tokenizer,
        client,
        patient_binding,
        primitives,
        permutations,
        sp_therapist,
        therapist_init_utterance,
        persona_ids=list(persona_ids),
        save_dir=save_dir,
        num_utterances=(gen_cfg.num_utterances_for_data if num_utterances is None
                        else int(num_utterances)),
        max_tokens=gen_cfg.max_tokens_per_response,
        temperature_therapist=gen_cfg.temperature_therapist,
        temperature_patient=gen_cfg.temperature_patient,
        therapist_max_input_tokens=gen_cfg.therapist_max_input_tokens,
        stop_strings=list(gen_cfg.stop_strings),
        patient_seed=patient_seed,
        batch_size=gen_cfg.conversation_batch_size,
        max_retries_without_progress=gen_cfg.max_retries_without_progress,
        verbose=gen_cfg.verbose,
        verbose_detailed=gen_cfg.verbose_detailed,
    )

    elapsed = time.time() - started
    avg_len = (sum(s.n_utterances for s in states) / len(states)) if states else 0.0
    print(f"{_LOG}Generated {len(states)} conversations in {elapsed:.1f}s "
          f"(avg {avg_len:.1f} utterances)")
    return states, elapsed, float(avg_len)


# =============================================================================
# ARTIFACTS
# =============================================================================


def write_run_metadata(*cfgs: Any) -> str:
    """Serialise the config bundle to ``run_metadata.json`` + ``run_metadata_history.jsonl``.

    Args:
        *cfgs: the whole bundle ``core.config.build_pto_config`` returned, splatted in any order --
            training, roles, generation, oracle, look-ahead AND paths::

                write_run_metadata(train_cfg, roles_cfg, gen_cfg, oracle_cfg, la_cfg, paths)

            The :class:`~core.config.RunPaths` among them says where to write.

    Returns:
        The path of the current-metadata file.

    Raises:
        ValueError: no ``RunPaths`` in *cfgs*.
        RuntimeError: from ``config_to_metadata``, when a complete bundle produced a payload
            missing one of the silently-mutable knobs.

    Notes:
        A thin composition of ``core.config.config_to_metadata`` and
        ``core.config.write_run_metadata`` -- the payload shape and the completeness assertion over
        the knobs that are NOT encoded in ``EXPERIMENT_NAME`` both live there. It is wrapped here
        so a trainer notebook makes one call rather than two.

        Call it once per process, BEFORE iteration 1. Every process appends a line to the history
        log, so a resume under changed knobs no longer erases what the earlier iterations ran
        under -- the Exp3 defect this layout exists to fix.
    """
    paths = next((c for c in cfgs if isinstance(c, RunPaths)), None)
    if paths is None:
        raise ValueError(
            "write_run_metadata needs the run's RunPaths among its arguments -- splat the whole "
            "bundle from build_pto_config(globals()), paths included."
        )
    payload = config_to_metadata(*cfgs)
    path = _write_metadata_payload(payload, paths)
    print(f"{_LOG}Run metadata: {path}")
    return path


def save_iteration_checkpoint(
    *,
    policy,
    tokenizer,
    paths: RunPaths,
    iteration: int,
    iter_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Write the iteration's adapter, tokenizer and metadata. Returns the adapter directory.

    Warning:
        **``selected_adapters=["default"]`` is passed whenever TRL's PEFT-DPO ``"ref"`` adapter is
        present.** ``DPOTrainer`` adds a frozen copy of the iteration-start weights as a second
        adapter to serve as the reference. Saving without the filter persists that copy into every
        checkpoint (and every Hub push) as a redundant ``ref/`` subfolder. Resume is unaffected
        either way: ``PeftModel.from_pretrained`` loads the root ``"default"`` adapter, and TRL
        recreates ``"ref"`` at the start of the next iteration.

    Notes:
        The existence of ``iteration_<N>/adapter/`` is the ONLY definition of "iteration done" --
        ``core.policy.resolve_start_state``, the EDA and the eval-generation tool all key on it. So
        this is the last thing an iteration does, after the metadata that describes it.
    """
    adapter_dir = paths.adapter_dir(iteration)
    print(f"\n{_LOG}Saving iteration_{iteration} checkpoint")

    adapters = getattr(policy, "peft_config", None) or {}
    selected = ["default"] if (isinstance(policy, PeftModel) and "ref" in adapters) else None
    policy.save_pretrained(adapter_dir, selected_adapters=selected)
    tokenizer.save_pretrained(adapter_dir)
    print(f"{_LOG}  Adapter saved: {adapter_dir}")

    checkpoints = list_hf_checkpoints(paths.training_dir(iteration))
    if checkpoints:
        print(f"{_LOG}  Sub-epoch checkpoints: {[os.path.basename(c) for c in checkpoints]}")

    if iter_metadata is not None:
        payload = dict(iter_metadata)
        payload["epoch_checkpoints"] = [os.path.basename(c) for c in checkpoints]
        _atomic_write_text(
            paths.iteration_metadata_path(iteration),
            json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
        )
    return adapter_dir


def push_adapter_to_hub(
    policy,
    repo_id: str,
    *,
    iteration: Optional[int] = None,
    num_iterations: Optional[int] = None,
) -> bool:
    """Push the adapter to the Hub. Returns whether it succeeded.

    Never raises: the local adapter is already on disk when this runs, so a Hub outage, a missing
    token or a rate limit must not destroy an iteration that completed. The failure is printed and
    the loop continues.
    """
    if not repo_id:
        return False
    message = (
        f"iteration_{iteration}/{num_iterations}" if iteration is not None else "adapter update"
    )
    try:
        policy.push_to_hub(repo_id=repo_id, commit_message=message)
    except Exception as exc:  # noqa: BLE001 - a push failure must never lose a finished iteration
        print(f"{_LOG}  WARNING: Hub push failed ({type(exc).__name__}: {exc}). "
              f"The local adapter is intact.")
        return False
    print(f"{_LOG}  Adapter pushed to Hub: {repo_id} ({message})")
    return True


# =============================================================================
# THE ITERATION
# =============================================================================


def _persona_order(n_personas: int, n_wanted: int, seed: int, iteration: int) -> List[int]:
    """Which personas run this iteration, in processing order.

    Shuffled per iteration so a subset run is not always the same personas, and so batch
    composition varies. With the full 96 requested this only permutes the processing order -- and
    it never reaches a filename, because conversations are named by the STABLE persona id.
    """
    order = list(range(int(n_personas)))
    random.Random(int(seed) + int(iteration)).shuffle(order)
    return order[:max(0, int(n_wanted))]


def run_one_iteration(
    *,
    iteration: int,
    start_iteration: int,
    resume_checkpoint: Optional[str],
    cumulative_step_offset: int,
    policy,
    tokenizer,
    client,
    permutations: Sequence[Dict[str, str]],
    sp_therapist: str,
    therapist_init_utterance: str,
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    paths: RunPaths,
    primitives: AsyncPrimitives,
    patient_binding=None,
    lora_config=None,
    tb_logger=None,
    lookahead_state: Optional[LookaheadState] = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> IterationResult:
    """One full PTO iteration: generate -> build preference pairs -> DPO -> save.

    Args:
        iteration: 1-based. It generates ``model_iter_{iteration - 1}`` (the output of the policy
            it STARTS with) and produces ``iteration_{iteration}/adapter``.
        start_iteration: the first iteration THIS process runs, from
            ``core.policy.resolve_start_state``. Only that one may consume *resume_checkpoint*.
        cumulative_step_offset: from ``core.policy.compute_cumulative_step_offset``; used to place
            this iteration's aggregates on the continuous TensorBoard x-axis.
        policy: the current policy; the returned :class:`IterationResult` carries its successor.
        permutations: the FULL ``generate_all_permutations()`` list.
        patient_binding: defaults to ``la_cfg.patient_binding`` -- the binding the look-ahead
            rollout uses, which must be the one the conversations were generated with.
        lora_config: see :func:`build_lora_config`; only used on a fresh start.
        tb_logger: an optional ``core.tb.RunTBLogger`` for the continuous run-level view.
        lookahead_state: one per ARM ideally, one per iteration at minimum. It carries the
            sub-batch the OOM halving arrived at, which is otherwise re-paid at every depth.

    Returns:
        :class:`IterationResult`. **Rebind the orchestration loop's policy from
        ``result.policy``** -- TRL may hand back a new wrapper.

    Raises:
        ValueError: the iteration produced (or reloaded) ZERO preference pairs. Both the build path
            and the reload path reach that check before any adapter is written -- see the module
            docstring, and delete the empty ``pairs.csv`` to force a clean rebuild.
        RuntimeError: the oracle success rate fell below ``oracle_cfg.min_success_ratio`` during
            the build. ``_progress.json`` survives, so the build resumes.

    Notes:
        The EDA recorder is flushed after the BUILD, not after training: every PTO row is produced
        during the build, and flushing on the reload path would clobber the ``generations.jsonl``
        the earlier session already wrote with an empty buffer.
    """
    iter_started = time.time()
    binding = patient_binding if patient_binding is not None else la_cfg.patient_binding
    la_state = lookahead_state if lookahead_state is not None else LookaheadState()

    iter_dir = paths.ensure_iteration_dir(iteration)
    conv_dir = paths.ensure_conv_dir(iteration - 1)
    pairs_csv = paths.pairs_csv_path(iteration)
    progress_path = paths.pref_progress_path(iteration)

    recorder = EDARecorder(
        paths.generations_path(iteration),
        enabled=bool(train_cfg.save_eda_generations),
        save_transcripts=bool(train_cfg.save_lookahead_transcripts),
    )

    print("\n" + "=" * 78)
    print(f"PTO ITERATION {iteration}/{train_cfg.num_iterations}   "
          f"[{train_cfg.experiment_name}]")
    print(f"  conversations from model_iter_{iteration - 1} -> "
          f"iteration_{iteration}/adapter")
    print("=" * 78)

    # -- Step 1: generate this iteration's conversations (they are also the eval set) ------
    persona_ids = _persona_order(
        len(permutations), gen_cfg.num_conversations_per_iter, train_cfg.seed, iteration)
    print(f"\n{_LOG}Step 1: generating {len(persona_ids)} conversations")
    states, generation_s, avg_len = run_generation_phase(
        policy=policy, tokenizer=tokenizer, client=client,
        patient_binding=binding, primitives=primitives,
        permutations=permutations, persona_ids=persona_ids,
        sp_therapist=sp_therapist, therapist_init_utterance=therapist_init_utterance,
        gen_cfg=gen_cfg, save_dir=conv_dir,
        patient_seed=int(train_cfg.seed) + int(iteration),
    )

    gc.collect()
    torch.cuda.empty_cache()

    # -- Step 2: preference pairs (the dominant phase; resume-aware) -----------------------
    print(f"\n{_LOG}Step 2: building preference pairs "
          f"[mode={train_cfg.pref_tree_mode}, M={train_cfg.num_branches_per_turn}, "
          f"tau={train_cfg.pref_filter_tau}, MCL={gen_cfg.min_conv_length}, K={la_cfg.k}]")
    build_started = time.time()

    if os.path.exists(pairs_csv):
        # The completion marker exists: reload and skip the build entirely. Do NOT flush the
        # recorder here -- generations.jsonl was written by the session that built these pairs,
        # and this recorder is empty.
        pref_pairs = reload_pairs_csv(pairs_csv)
        pref_pairs_reloaded = True
        pref_pair_s = 0.0            # this process did not build; the earlier session logged it
        print(f"{_LOG}  Found pairs.csv -- reloaded {len(pref_pairs)} pairs, skipping the build "
              f"({time.time() - build_started:.1f}s)")
    else:
        pref_pairs = build_pref_pairs(
            states, permutations,
            model=policy, tokenizer=tokenizer, client=client,
            sp_therapist=sp_therapist,
            oracle_cfg=oracle_cfg, la_cfg=la_cfg, primitives=primitives,
            train_cfg=train_cfg, gen_cfg=gen_cfg, patient_binding=binding,
            recorder=recorder, iteration=iteration, progress_path=progress_path,
            lookahead_state=la_state,
            patient_seed=int(train_cfg.seed) + int(iteration),
        )
        pref_pairs_reloaded = False
        pref_pair_s = time.time() - build_started
        print(f"{_LOG}  Built {len(pref_pairs)} preference pairs in {pref_pair_s:.1f}s "
              f"from {len(states)} conversations")

        write_pairs_csv(pref_pairs, pairs_csv)
        print(f"{_LOG}  Preference pairs saved: {pairs_csv}")

        flushed = recorder.flush()
        if flushed:
            print(f"{_LOG}  EDA generations saved: {flushed} ({len(recorder)} branch rows)")

        # The build is complete; pairs.csv is now the marker, so the in-build snapshot is dead.
        if os.path.exists(progress_path):
            try:
                os.remove(progress_path)
            except OSError:
                pass

    # Function-body indentation on purpose: BOTH the reload path and the build path reach this,
    # so an empty pairs.csv fails here rather than producing an adapter trained on nothing.
    if not pref_pairs:
        raise ValueError(
            f"Iteration {iteration} has 0 preference pairs from {len(states)} conversations "
            f"(avg {avg_len:.1f} utterances). Either every branch point tied within "
            f"PREF_FILTER_TAU={train_cfg.pref_filter_tau}, or MIN_CONV_LENGTH="
            f"{gen_cfg.min_conv_length} filtered every branch point, or a previous run left an "
            f"EMPTY completion marker.\n"
            f"  Fix: DELETE {pairs_csv} to force a clean rebuild.\n"
            f"  Do NOT lower PREF_FILTER_TAU to get past this. Tau is not encoded in "
            f"EXPERIMENT_NAME, so changing it mid-arm writes two different configurations into "
            f"this one folder with nothing on disk able to tell them apart."
        )

    gc.collect()
    torch.cuda.empty_cache()

    # -- Step 3: datasets -----------------------------------------------------------------
    train_dataset, eval_dataset = build_dpo_dataset(
        pref_pairs,
        eval_split_ratio=train_cfg.eval_split_ratio,
        seed=train_cfg.seed,
        verbose=gen_cfg.verbose,
    )

    # -- Step 4: the DPO update -----------------------------------------------------------
    print(f"\n{_LOG}Step 4: DPO for {train_cfg.epochs_per_iteration} epoch(s)")
    training_dir = paths.training_dir(iteration)
    tb_dir = os.path.join(training_dir, "tb_logs")
    dpo_args = build_dpo_config(
        train_cfg, gen_cfg,
        output_dir=training_dir,
        num_train_pairs=len(train_dataset),
        hub_model_id=train_cfg.adapter_repo,
        has_eval=len(eval_dataset) > 0,
    )

    new_policy, step_delta, training_s = run_training_phase(
        policy=policy, tokenizer=tokenizer,
        dpo_args=dpo_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        train_cfg=train_cfg,
        iteration=iteration, start_iteration=start_iteration,
        resume_checkpoint=resume_checkpoint,
        lora_config=lora_config,
        tensorboard_log_dir=tb_dir,
        callbacks=callbacks,
    )

    # -- Run-level live view (opt-in): the aggregates TRL does not know about --------------
    if tb_logger is not None and len(recorder):
        scalars, scores = recorder.aggregate()
        scalars["iteration/num_pref_pairs"] = float(len(pref_pairs))
        scalars["iteration/num_conversations"] = float(len(states))
        scalars["iteration/avg_conversation_length"] = float(avg_len)
        end_step = int(cumulative_step_offset) + int(step_delta)
        tb_logger.log_scalars(scalars, step=end_step, iteration=iteration)
        if scores:
            tb_logger.log_histogram(
                "eda/candidate_reward_hist", scores, step=end_step, iteration=iteration)
        samples = recorder.sample_for_display(int(train_cfg.tb_sample_completions_n))
        if samples:
            tb_logger.log_sample_completions(samples, step=end_step, iteration=iteration)

    # -- Step 5: timing, metadata, adapter ------------------------------------------------
    # Logged BEFORE the metadata is assembled so metadata_fields() sees this session. A reloaded
    # build logs pref_pair_s=0.0 on purpose: the session that actually built the pairs recorded it,
    # and the cumulative total is the sum over sessions.
    log_session(
        iter_dir,
        generation_s=generation_s,
        pref_pair_s=pref_pair_s,
        training_s=training_s,
        started_at=iter_started,
        note=("reloaded pairs.csv" if pref_pairs_reloaded else ""),
    )

    iter_metadata: Dict[str, Any] = {
        "experiment_name": train_cfg.experiment_name,
        "method": "PTO",
        "iteration": int(iteration),
        "model_state_generated": f"model_iter_{iteration - 1}",
        "base_model": train_cfg.base_model_id,
        "oracle_model": oracle_cfg.binding.model,
        "patient_model": binding.model,
        "questionnaire_ids": list(train_cfg.questionnaire_ids),
        "lookahead_k": int(la_cfg.k),
        # Not in EXPERIMENT_NAME and auto-halved on OOM: without this field a wall-clock
        # comparison between two iterations of the same arm is meaningless.
        "lookahead_sub_batch_final": la_state.sub_batch,
        "lookahead_oom_events": int(la_state.oom_events),
        "min_conv_length": int(gen_cfg.min_conv_length),
        "pref_tree_mode": train_cfg.pref_tree_mode,
        "num_branches_per_turn": int(train_cfg.num_branches_per_turn),
        "pref_filter_tau": float(train_cfg.pref_filter_tau),
        "greedy_trunk_target_len": train_cfg.greedy_trunk_target_len,
        "dpo_beta": float(train_cfg.dpo_beta),
        "dpo_loss_type": train_cfg.dpo_loss_type,
        "learning_rate": float(train_cfg.learning_rate),
        "num_conversations": len(states),
        "num_pref_pairs": len(pref_pairs),
        "num_train_pairs": len(train_dataset),
        "num_eval_pairs": len(eval_dataset),
        "avg_conversation_length": float(avg_len),
        "epochs_per_iteration": int(train_cfg.epochs_per_iteration),
        "pref_pairs_reloaded": bool(pref_pairs_reloaded),
        # Per-PROCESS seconds. Read the cumulative_* fields (and n_timing_sessions) instead
        # whenever the iteration was resumed -- these describe this session only.
        "generation_time_s": round(generation_s, 3),
        "pref_pair_time_s": round(pref_pair_s, 3),
        "training_time_s": round(training_s, 3),
        **metadata_fields(iter_dir),
    }

    adapter_dir = save_iteration_checkpoint(
        policy=new_policy, tokenizer=tokenizer,
        paths=paths, iteration=iteration, iter_metadata=iter_metadata,
    )
    if train_cfg.push_to_hub:
        push_adapter_to_hub(
            new_policy, train_cfg.adapter_repo,
            iteration=iteration, num_iterations=train_cfg.num_iterations,
        )

    print(f"\n{_LOG}Iteration {iteration} complete in {time.time() - iter_started:.1f}s "
          f"({len(states)} conversations, {len(pref_pairs)} pairs, {step_delta} steps)")
    print("=" * 78)

    return IterationResult(
        policy=new_policy,
        iteration=int(iteration),
        step_delta=int(step_delta),
        n_conversations=len(states),
        n_pref_pairs=len(pref_pairs),
        generation_s=float(generation_s),
        pref_pair_s=float(pref_pair_s),
        training_s=float(training_s),
        adapter_dir=adapter_dir,
        pref_pairs_reloaded=bool(pref_pairs_reloaded),
        lookahead_sub_batch=la_state.sub_batch,
    )


def run_final_eval(
    *,
    policy,
    tokenizer,
    client,
    permutations: Sequence[Dict[str, str]],
    sp_therapist: str,
    therapist_init_utterance: str,
    train_cfg: PTOTrainingConfig,
    gen_cfg: GenConfig,
    paths: RunPaths,
    primitives: AsyncPrimitives,
    patient_binding=None,
    la_cfg: Optional[LookaheadConfig] = None,
) -> str:
    """Generate ``model_iter_{NUM_ITERATIONS}`` with the FINAL policy. Returns the directory.

    Every trained iteration's conversations are generated by the policy it starts with, so the
    last adapter would otherwise have no eval data at all: ``N`` iterations produce conversation
    folders ``model_iter_0 .. model_iter_{N-1}``. This pass produces the ``N+1``-th.

    Args:
        patient_binding: the patient simulator. Defaults to ``la_cfg.patient_binding`` when
            *la_cfg* is given; otherwise it is REQUIRED, because there is no other way to know
            which model the arm was trained against.

    Raises:
        ValueError: no patient binding could be resolved.

    Notes:
        Timed as ``eval_gen_s`` against the LAST iteration's directory, so the arm's total cost is
        the sum over ``iteration_*/timing_sessions.jsonl`` with nothing left outside it. The
        conversations resume per-persona from disk, so a killed pass costs only what it had not
        yet written.
    """
    binding = patient_binding
    if binding is None and la_cfg is not None:
        binding = la_cfg.patient_binding
    if binding is None:
        raise ValueError(
            "run_final_eval needs a patient binding: pass patient_binding= (or la_cfg=, whose "
            "patient_binding is used). Defaulting it would silently evaluate the arm against a "
            "different simulator than it was trained on."
        )

    final_state = int(train_cfg.num_iterations)
    conv_dir = paths.ensure_conv_dir(final_state)
    started = time.time()

    print("\n" + "=" * 78)
    print(f"FINAL EVAL GENERATION -- model_iter_{final_state}   [{train_cfg.experiment_name}]")
    print("=" * 78)

    # A distinct seed so the final pass is not a replay of the last training iteration's order.
    seed = int(train_cfg.seed) + final_state + 1
    persona_ids = _persona_order(
        len(permutations), gen_cfg.num_conversations_per_iter, train_cfg.seed, final_state + 1)

    states, elapsed, avg_len = run_generation_phase(
        policy=policy, tokenizer=tokenizer, client=client,
        patient_binding=binding, primitives=primitives,
        permutations=permutations, persona_ids=persona_ids,
        sp_therapist=sp_therapist, therapist_init_utterance=therapist_init_utterance,
        gen_cfg=gen_cfg, save_dir=conv_dir, patient_seed=seed,
    )

    log_session(
        paths.iteration_dir(final_state),
        eval_gen_s=elapsed,
        started_at=started,
        note=f"final eval generate -> model_iter_{final_state}",
    )
    print(f"{_LOG}Final eval: {len(states)} conversations, avg {avg_len:.1f} utterances")
    print(f"{_LOG}Saved to: {conv_dir}")

    del states
    gc.collect()
    torch.cuda.empty_cache()
    return conv_dir
