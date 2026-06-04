"""
pto_trainer.py — PTO_Exp3 orchestration helpers + pref-tree construction + DPO loop.

PTO_Exp3 mirrors GRPO_Exp3's iterative structure but uses preference-tree
construction (per-turn branching, K-turn look-ahead, oracle scoring, τ-filtered
best/worst pair extraction) → DPO loss instead of GRPO loss.

Each iteration:
1. Generate ``num_conversations_per_iter`` conversations with the current policy
   (same machinery as GRPO).
2. For each therapist turn whose conversation-so-far has ≥ ``min_conv_length``
   utterances, sample ``num_branches_per_turn`` therapist completions, run K-turn
   look-ahead on each, oracle-score, and emit a (chosen, rejected) pair if the
   score gap exceeds ``pref_filter_tau``.
3. Train ``DPOTrainer`` on the collected pref pairs for ``epochs_per_iteration``.
4. Save adapter + iteration metadata.

The notebook keeps the per-iteration loop visible — :func:`run_iterative_training`
exists as a convenience wrapper but the recommended pattern is to compose
:func:`run_one_iteration` directly from the notebook.

Shared with GRPO_Exp3/ via ``code/_shared/``:
- conversation generation, oracle scoring, look-ahead simulation, model loading,
  iteration-resume helpers, TB callbacks + logging lifecycle.
"""

import os
import gc
import json
import time
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import asyncio
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from _shared import (
    # convs
    ConversationState,
    generate_all_conversations,
    generate_therapist_responses_batch,
    generate_patient_responses_batch,
    turns_to_messages,
    turns_to_patient_messages,
    format_conversation_for_oracle,
    # reward
    OracleConfig, LookaheadConfig, OracleAsyncPrimitives,
    get_evaluation_json, simulate_lookahead_batch,
    # model + resume
    list_iteration_checkpoints, list_hf_checkpoints,
    resolve_start_state, compute_cumulative_step_offset,
    patch_generate, ADAPTER_SUBDIR, ITER_PREFIX,
    # logging lifecycle + TB callbacks
    CheckpointMetadataCallback, CumulativeStepCallback,
    init_iteration_logging, finish_iteration_logging,
    setup_tensorboard_logging, patch_trainer_tensorboard_callback,
)
# Private helper (not in _shared public API): applies one generated response to a
# ConversationState — appends the turn, keeps both message-perspective lists synced,
# flips next_speaker, and handles the SESSION ENDED marker. The greedy grower drives
# trunk growth through it (same machinery conversation_loop_batch uses).
from _shared.convs import _process_session_response


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIG DATACLASS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class PTOConfig:
    """Loop knobs for PTO_Exp3 iterative training.

    Field-name parity with GRPO_Exp3's TrainingConfig for generation-related
    knobs so shared helpers (e.g. ``generate_all_conversations``) consume the
    same cfg shape. PTO-specific additions: ``pref_filter_tau``,
    ``num_branches_per_turn``, ``dpo_beta``, ``dpo_loss_type``.
    """
    # Identity
    experiment_name: str
    mode_tag: str
    base_model_id: str
    oracle_model_id: str
    current_adapter_repo: str
    seed: int

    # Loop
    num_iterations: int
    num_conversations_per_iter: int
    epochs_per_iteration: int
    total_effective_epochs: float

    # Paths
    local_outdir: str
    conv_outdir: str

    # Generation
    patient_model_id: str
    temperature_therapist_gen: float
    temperature_patient: float
    num_utterances_for_data: int
    max_tokens_per_response: int
    therapist_max_input_tokens: int
    conversation_batch_size: int
    patient_api_concurrency: int
    patient_api_max_retries: int
    patient_api_backoff_seconds: float
    max_gen_retries_without_progress: int
    min_conv_length: int
    max_allowed_prompt_length: int
    stop_strings: Optional[List[str]]

    # Pref tree construction (PTO-specific)
    pref_tree_mode: str                 # "greedy" (true PTO: grow trunk from best-of-M) | "independent" (branch pre-recorded conv)
    num_branches_per_turn: int          # M candidate completions per branch point
    pref_filter_tau: float              # drop pairs with chosen_score - rejected_score <= tau
    branch_sample_temperature: float    # temperature for sampling M completions
    branch_max_tokens: int              # max tokens per completion

    # DPO training
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_completion_length: int          # max tokens used by DPOTrainer for chosen/rejected
    dpo_beta: float
    dpo_loss_type: str                  # "sigmoid" | "ipo" | "kto_pair" | etc.
    warmup_steps_ratio: float
    logging_steps: int
    save_strategy: str
    save_total_limit: Optional[int]
    eval_split_ratio: float
    lora_r: int
    lora_alpha: int

    # Reporting / hub
    report_to: List[str]
    push_to_hub: bool

    # Misc
    questionnaire_ids: Sequence[int]

    # Verbosity
    gen_verbose: bool
    gen_verbose_detailed: bool


@dataclass
class PTOSummary:
    """Returned by :func:`run_iterative_training`."""
    iterations_run: List[int]
    iterations_saved: List[int]
    final_conv_dir: Optional[str]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  RUN METADATA + W&B CONTEXT                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def write_run_metadata(cfg: PTOConfig) -> str:
    """Snapshot the full PTOConfig to ``<local_outdir>/run_metadata.json``.

    Mirror of GRPO_Exp3's helper — kept locally for trainer self-containedness.
    """
    os.makedirs(cfg.local_outdir, exist_ok=True)
    payload = {
        "experiment_name": cfg.experiment_name,
        "mode_tag": cfg.mode_tag,
        "method": "PTO_Exp3",
        "started_at": pd.Timestamp.now().isoformat(),
        "config": asdict(cfg),
    }
    cfg_dict = payload["config"]
    if isinstance(cfg_dict.get("questionnaire_ids"), tuple):
        cfg_dict["questionnaire_ids"] = list(cfg_dict["questionnaire_ids"])

    path = os.path.join(cfg.local_outdir, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓ Run metadata: {path}")
    return path


def build_wandb_ctx(cfg: PTOConfig) -> Optional[dict]:
    """Build the wandb context dict from cfg, or None if not reporting to wandb."""
    if "wandb" not in cfg.report_to:
        return None
    if wandb.run is not None:
        wandb.finish()
    project = f"PTO_Iterative_Experiments_{cfg.mode_tag.upper()}"
    run_id = f"{cfg.experiment_name}_{cfg.mode_tag}"
    ctx = {
        "run_id": run_id,
        "project": project,
        "config": {
            "experiment_name": cfg.experiment_name,
            "method": "PTO_Exp3",
            "base_model": cfg.base_model_id,
            "mode_tag": cfg.mode_tag,
            "adapter_repo": cfg.current_adapter_repo,
            "oracle_model": cfg.oracle_model_id,
            "num_iterations": cfg.num_iterations,
            "epochs_per_iteration": cfg.epochs_per_iteration,
            "total_effective_epochs": cfg.total_effective_epochs,
            "num_conversations_per_iter": cfg.num_conversations_per_iter,
            "num_utterances_for_data": cfg.num_utterances_for_data,
            "min_conv_length": cfg.min_conv_length,
            "num_branches_per_turn": cfg.num_branches_per_turn,
            "pref_filter_tau": cfg.pref_filter_tau,
            "learning_rate": cfg.learning_rate,
            "dpo_beta": cfg.dpo_beta,
            "dpo_loss_type": cfg.dpo_loss_type,
            "train_batch_size": cfg.train_batch_size,
            "max_completion_length": cfg.max_completion_length,
            "questionnaire_ids": cfg.questionnaire_ids,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
        },
    }
    print(f"✓ W&B config ready: project={project}, run={run_id}")
    return ctx


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    GENERATION PHASE (reuses _shared)                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _conv_dir_for_iter(cfg: PTOConfig, model_iter_label: int) -> str:
    """``CONV_OUTDIR/model_iter_<label>_TT<temp_t>_TP<temp_p>/``."""
    return os.path.join(
        cfg.conv_outdir,
        f"model_iter_{model_iter_label}"
        f"_TT{cfg.temperature_therapist_gen}_TP{cfg.temperature_patient}",
    )


def run_generation_only(
    *,
    policy, tokenizer, client,
    active_permutations,
    therapist_system_prompt, therapist_init_utterance,
    conv_dir: Optional[str],
    cfg: PTOConfig,
    patient_api_seed: Optional[int] = None,
    num_utterances: Optional[int] = None,
):
    """Generate conversations with the current policy. Mirrors GRPO_Exp3's lean
    generation path — produces ConversationState objects, no prompt extraction.

    ``conv_dir=None`` skips disk persistence (used for throwaway greedy-mode tree
    prefixes). ``num_utterances`` overrides ``cfg.num_utterances_for_data`` (e.g. a
    short ``min_conv_length-1`` prefix pass).
    """
    gen_start = time.time()
    policy.eval()
    policy.config.use_cache = True

    completed_states = generate_all_conversations(
        therapist_model=policy,
        therapist_tokenizer=tokenizer,
        client=client,
        permutations=active_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        save_dir=conv_dir,
        patient_model_id=cfg.patient_model_id,
        max_tokens_per_response=cfg.max_tokens_per_response,
        num_utterances=(num_utterances if num_utterances is not None else cfg.num_utterances_for_data),
        temperature_therapist=cfg.temperature_therapist_gen,
        temperature_patient=cfg.temperature_patient,
        batch_size=cfg.conversation_batch_size,
        therapist_max_input_tokens=cfg.therapist_max_input_tokens,
        patient_api_concurrency=cfg.patient_api_concurrency,
        patient_api_max_retries=cfg.patient_api_max_retries,
        patient_api_seed=patient_api_seed,
        patient_api_backoff_seconds=cfg.patient_api_backoff_seconds,
        batch_cooldown_seconds=1.0,
        max_retries_without_progress=cfg.max_gen_retries_without_progress,
        stop_strings=cfg.stop_strings,
        verbose=cfg.gen_verbose,
        verbose_detailed=cfg.gen_verbose_detailed,
    )

    gen_time = time.time() - gen_start
    avg_conv_len = (
        float(np.mean([len(s.conversation) for s in completed_states]))
        if completed_states else 0.0
    )
    print(f"  ✓ Generated {len(completed_states)} conversations in {gen_time:.1f}s")
    print(f"  Average conversation length: {avg_conv_len:.1f} utterances")
    return completed_states, gen_time, avg_conv_len


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    PREF-TREE CONSTRUCTION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _sample_M_completions(
    policy, tokenizer,
    prefix_messages: List[Dict],
    num_branches: int,
    temperature: float,
    max_tokens: int,
    max_input_tokens: int,
    stop_strings: Optional[List[str]],
) -> List[str]:
    """Sample ``num_branches`` therapist completions from the same prefix.

    Implemented by feeding ``generate_therapist_responses_batch`` the same prefix
    repeated ``num_branches`` times — ``do_sample=True`` ensures each draw is
    independent.
    """
    batch_messages = [prefix_messages] * num_branches
    responses, error = generate_therapist_responses_batch(
        therapist_model=policy,
        therapist_tokenizer=tokenizer,
        batch_messages=batch_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        max_input_tokens=max_input_tokens,
        stop_strings=stop_strings,
    )
    if error is not None or responses is None:
        print(f"    ⚠ Branch sampling failed ({error}); skipping this branch point")
        return []
    # Filter out empty / whitespace-only completions
    return [r for r in responses if r and r.strip()]


async def _oracle_score_extended(
    client,
    oracle_cfg: OracleConfig,
    primitives: OracleAsyncPrimitives,
    full_conversation: str,
    questionnaire_ids: Sequence[int],
) -> Optional[float]:
    """Mean oracle reward over ``questionnaire_ids`` for one already-extended
    transcript. Returns ``None`` if any questionnaire fails."""
    scores: List[float] = []
    for qid in questionnaire_ids:
        data, _ = await get_evaluation_json(
            client, oracle_cfg, primitives,
            transcript="", completion="",
            questionnaire_id=int(qid),
            full_conversation_override=full_conversation,
        )
        if data is None:
            return None
        scores.append(float(data["mean_score"]))
    return float(np.mean(scores))


async def _score_completions_batch(
    client,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    *,
    policy, tokenizer,
    therapist_system_prompt: str,
    transcripts: List[str],
    completions: List[str],
    patient_system_prompts: List[str],
    questionnaire_ids: Sequence[int],
) -> List[Optional[float]]:
    """Score a parallel batch of (transcript, completion, patient_prompt) triples via
    K-turn look-ahead (when ``k>0``) + oracle. Shared by the independent-branch scorer
    (:func:`_score_branches`) and the greedy grower (:func:`grow_preference_trees_batch`).
    Returns one mean reward per item (``None`` on oracle failure)."""
    if lookahead_cfg.k > 0:
        full_convs = await simulate_lookahead_batch(
            transcripts=transcripts,
            completions=completions,
            system_prompt_therapist=therapist_system_prompt,
            system_prompts_patient=patient_system_prompts,
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
        )
    else:
        full_convs = [
            f"{t}\n\n[THERAPIST]: {c}" for t, c in zip(transcripts, completions)
        ]

    return await asyncio.gather(*[
        _oracle_score_extended(client, oracle_cfg, primitives, fc, questionnaire_ids)
        for fc in full_convs
    ])


async def _score_branches(
    client,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    *,
    policy, tokenizer,
    therapist_system_prompt: str,
    patient_system_prompt: str,
    transcript: str,
    completions: List[str],
    questionnaire_ids: Sequence[int],
) -> List[Optional[float]]:
    """Score ``len(completions)`` candidates from ONE prefix (independent-branch path).

    Thin wrapper over :func:`_score_completions_batch` with the single transcript +
    patient prompt broadcast across the completions.
    """
    n = len(completions)
    return await _score_completions_batch(
        client, oracle_cfg, lookahead_cfg, primitives,
        policy=policy, tokenizer=tokenizer,
        therapist_system_prompt=therapist_system_prompt,
        transcripts=[transcript] * n,
        completions=completions,
        patient_system_prompts=[patient_system_prompt] * n,
        questionnaire_ids=questionnaire_ids,
    )


async def build_pref_pairs_for_conversation(
    state,
    permutation,
    *,
    policy, tokenizer, client,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTOConfig,
) -> List[Dict]:
    """Build all (prompt, chosen, rejected) DPO pairs from one conversation.

    For each therapist turn position where the conversation-so-far has at least
    ``cfg.min_conv_length`` utterances, sample ``cfg.num_branches_per_turn``
    candidates, score them, and emit a pair if the score gap exceeds
    ``cfg.pref_filter_tau``.
    """
    turns = state.turns
    if not turns:
        return []

    patient_system_prompt = permutation["patient_system_prompt"]
    pairs: List[Dict] = []

    # Branch points: positions where the NEXT turn would be a therapist turn.
    # I.e., the previous utterance was a patient. We exclude position 0 (no
    # conversation history) and require min_conv_length total utterances so far.
    for i, turn in enumerate(turns):
        if turn["role"] != "patient":
            continue
        if (i + 1) < cfg.min_conv_length:
            continue
        # The next therapist turn must actually exist in the recorded conv
        # — i.e., i is not the final patient turn — otherwise there's no
        # branching point to anchor a pref pair.
        if i + 1 >= len(turns):
            continue

        partial_turns = turns[: i + 1]  # ends on patient turn
        prefix_messages = turns_to_messages(partial_turns, therapist_system_prompt)
        transcript = format_conversation_for_oracle(prefix_messages)

        # Render the prompt the same way DPOTrainer's tokenizer will see it.
        prompt = tokenizer.apply_chat_template(
            prefix_messages, add_generation_prompt=True, tokenize=False,
        )

        # ── Step 1: sample M branches ──
        completions = _sample_M_completions(
            policy, tokenizer,
            prefix_messages=prefix_messages,
            num_branches=cfg.num_branches_per_turn,
            temperature=cfg.branch_sample_temperature,
            max_tokens=cfg.branch_max_tokens,
            max_input_tokens=cfg.therapist_max_input_tokens,
            stop_strings=cfg.stop_strings,
        )
        if len(completions) < 2:
            continue  # need at least 2 to form a pair

        # ── Step 2: score each branch (lookahead + oracle) ──
        scores = await _score_branches(
            client, oracle_cfg, lookahead_cfg, primitives,
            policy=policy, tokenizer=tokenizer,
            therapist_system_prompt=therapist_system_prompt,
            patient_system_prompt=patient_system_prompt,
            transcript=transcript,
            completions=completions,
            questionnaire_ids=cfg.questionnaire_ids,
        )

        # ── Step 3: best/worst → pref pair with τ filter ──
        scored = [(s, c) for s, c in zip(scores, completions) if s is not None]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda sc: sc[0])
        worst_score, worst_text = scored[0]
        best_score, best_text = scored[-1]
        if (best_score - worst_score) <= cfg.pref_filter_tau:
            continue
        pairs.append({
            "prompt": prompt,
            "chosen": best_text,
            "rejected": worst_text,
            "chosen_score": best_score,
            "rejected_score": worst_score,
            "conversation_id": state.permutation_index,
            "branch_turn_index": i,
        })

    return pairs


async def extract_pref_pairs_from_conversations(
    completed_states,
    permutations,
    *,
    policy, tokenizer, client,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTOConfig,
) -> List[Dict]:
    """Build pref pairs across all conversations. Sequential per-conversation
    (branch sampling is GPU-bound) but the scoring inside each conversation is
    async-batched.
    """
    all_pairs: List[Dict] = []
    for state in completed_states:
        if state.failed or not state.conversation or len(state.conversation) <= 1:
            continue
        perm = permutations[state.permutation_index]
        pairs = await build_pref_pairs_for_conversation(
            state, perm,
            policy=policy, tokenizer=tokenizer, client=client,
            therapist_system_prompt=therapist_system_prompt,
            oracle_cfg=oracle_cfg, lookahead_cfg=lookahead_cfg,
            primitives=primitives,
            cfg=cfg,
        )
        all_pairs.extend(pairs)
        if cfg.gen_verbose:
            print(
                f"    conv {state.permutation_index}: emitted {len(pairs)} pref pair(s) "
                f"(min_conv_length={cfg.min_conv_length}, tau={cfg.pref_filter_tau})"
            )
    return all_pairs


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              PREF-TREE CONSTRUCTION — GREEDY MODE (lock-step)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# True PTO: start each conversation from a short prefix (``min_conv_length`` utts,
# ending on a patient turn) and grow ONE trunk greedily — at each therapist turn,
# branch M completions, look-ahead + oracle-score them, append the BEST to the trunk
# (so the chosen completion feeds the next branch point), let the patient continue,
# repeat. Contrast with the independent path above, which branches each patient turn
# of a PRE-RECORDED conversation and never feeds the winner back.


@dataclass
class _TreeState:
    """One live trunk during greedy growth.

    ``conv`` is a plain :class:`ConversationState` — we reuse all its turn/message
    bookkeeping so the patient- and therapist-perspective message lists stay synced
    as the trunk grows (via ``_process_session_response``).
    """
    conv: ConversationState
    patient_system_prompt: str
    pairs: List[Dict] = field(default_factory=list)


def _slice_prefix_seeds(completed_states, min_conv_length: int) -> List[ConversationState]:
    """Slice the first ``min_conv_length`` utterances off each Step-1 conversation to
    seed greedy tree growth — no separate prefix-generation pass needed.

    A conv qualifies only if it extends *past* ``min_conv_length`` (so the MCL-length
    prefix never ends on a session-terminating patient turn). ``min_conv_length`` is even
    (notebook ``_validate_config``), so ``turns[min_conv_length - 1]`` is a patient turn
    and the prefix's next speaker is the therapist.

    Containers are copied (``turns`` dict-by-dict, fresh ``conversation``/message lists)
    so the grower — which appends turns and rehydrates ``messages_*`` via
    :func:`grow_preference_trees_batch` — never mutates the Step-1 conv (it is the frozen
    eval data + metadata source). Message lists are left empty; the grower rebuilds them
    from ``turns``. Mirrors the :func:`load_conversation_from_csv` construction.
    """
    seeds: List[ConversationState] = []
    for s in completed_states:
        if s.failed or len(s.turns) <= min_conv_length:
            continue
        prefix_turns = s.turns[:min_conv_length]
        if prefix_turns[-1]["role"] != "patient":
            print(f"    ⚠ Conv {s.permutation_index}: MCL prefix doesn't end on a patient "
                  f"turn (last role={prefix_turns[-1]['role']!r}); skipped as a seed")
            continue
        seeds.append(ConversationState(
            permutation_index=s.permutation_index,
            conversation=list(s.conversation[:min_conv_length]),
            messages_Patient_assist=[],
            messages_Therapist_assist=[],
            turns=[dict(t) for t in prefix_turns],
            next_speaker="therapist",
        ))
    return seeds


async def _grow_therapist_depth(
    active: List["_TreeState"],
    *,
    client, policy, tokenizer,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTOConfig,
    M: int,
) -> None:
    """One therapist (branching) depth across all active trunks.

    Branch ``M`` completions per trunk, look-ahead + oracle-score them all in one
    batch, then per trunk append the best completion (always — to advance the trunk)
    and emit a (chosen, rejected) pair when the score gap exceeds ``pref_filter_tau``.
    """
    # 1. Flatten (trunk × M): repeat each trunk's therapist-view messages M times;
    #    keep the per-trunk oracle transcript (computed once) for scoring.
    transcripts = [
        format_conversation_for_oracle(t.conv.messages_Therapist_assist) for t in active
    ]
    flat_messages = [t.conv.messages_Therapist_assist for t in active for _ in range(M)]

    # 2. Branch-sample, chunked at conversation_batch_size to bound VRAM (active×M
    #    can be large; simulate_lookahead_batch sub-batches its own therapist turns).
    completions: List[Optional[str]] = [None] * len(flat_messages)
    chunk = max(1, cfg.conversation_batch_size)
    gpu_lock = primitives.gpu_lock()
    for start in range(0, len(flat_messages), chunk):
        sub = flat_messages[start:start + chunk]
        async with gpu_lock:
            responses, error = generate_therapist_responses_batch(
                therapist_model=policy,
                therapist_tokenizer=tokenizer,
                batch_messages=sub,
                max_tokens=cfg.branch_max_tokens,
                temperature=cfg.branch_sample_temperature,
                max_input_tokens=cfg.therapist_max_input_tokens,
                stop_strings=cfg.stop_strings,
            )
        if error is not None or responses is None:
            print(f"    ⚠ Branch sampling failed for a chunk ({error}); those branches dropped")
            continue
        for j, r in enumerate(responses):
            completions[start + j] = r

    # 3+4. Look-ahead + oracle-score every non-empty completion in ONE batch.
    score_idx = [i for i, c in enumerate(completions) if c and c.strip()]
    if score_idx:
        packed = await _score_completions_batch(
            client, oracle_cfg, lookahead_cfg, primitives,
            policy=policy, tokenizer=tokenizer,
            therapist_system_prompt=therapist_system_prompt,
            transcripts=[transcripts[i // M] for i in score_idx],
            completions=[completions[i] for i in score_idx],
            patient_system_prompts=[active[i // M].patient_system_prompt for i in score_idx],
            questionnaire_ids=cfg.questionnaire_ids,
        )
        score_by_flat = {i: packed[k] for k, i in enumerate(score_idx)}
    else:
        score_by_flat = {}

    # 5. Per trunk: pick best/worst, append winner, emit pair if gap > τ.
    for ti, t in enumerate(active):
        scored = []
        for m in range(M):
            flat_i = ti * M + m
            c = completions[flat_i]
            s = score_by_flat.get(flat_i)
            if c and c.strip() and s is not None:
                scored.append((s, c))
        if not scored:
            t.conv.is_active = False  # freeze: no valid branch to advance the trunk
            continue
        scored.sort(key=lambda sc: sc[0])
        worst_score, worst_text = scored[0]
        best_score, best_text = scored[-1]

        # Snapshot the prompt from the trunk BEFORE appending the winner.
        prompt = tokenizer.apply_chat_template(
            turns_to_messages(t.conv.turns, therapist_system_prompt),
            add_generation_prompt=True, tokenize=False,
        )
        branch_depth = len(t.conv.conversation)

        # Always append the winner so the trunk advances (the greedy feedback the
        # independent path lacks). Flips next_speaker -> patient; handles SESSION
        # ENDED inside the winner (freezes the trunk, keeps cleaned text).
        _process_session_response(t.conv, best_text, "therapist", "user", "assistant")

        if len(scored) >= 2 and (best_score - worst_score) > cfg.pref_filter_tau:
            t.pairs.append({
                "prompt": prompt,
                "chosen": best_text,
                "rejected": worst_text,
                "chosen_score": best_score,
                "rejected_score": worst_score,
                "conversation_id": t.conv.permutation_index,
                "branch_depth": branch_depth,
            })


async def _grow_patient_depth(
    active: List["_TreeState"],
    *,
    client,
    cfg: PTOConfig,
    primitives: OracleAsyncPrimitives,
) -> None:
    """One patient depth across all active trunks (fresh API call per trunk)."""
    batch_messages = [t.conv.messages_Patient_assist for t in active]
    responses = await generate_patient_responses_batch(
        client, cfg.patient_model_id, batch_messages,
        cfg.max_tokens_per_response, cfg.temperature_patient,
        primitives.lookahead_patient_sem(),
        cfg.patient_api_max_retries, cfg.patient_api_backoff_seconds,
        seed=None,
    )
    for t, resp in zip(active, responses):
        if isinstance(resp, BaseException) or resp is None:
            t.conv.is_active = False
            t.conv.failed = True
            continue
        _process_session_response(t.conv, resp, "patient", "assistant", "user")


async def grow_preference_trees_batch(
    seed_states,
    permutations,
    *,
    policy, tokenizer, client,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTOConfig,
) -> List[Dict]:
    """Greedy preference-tree growth (true PTO), lock-step across all seed trunks.

    Each seed is a short prefix (``min_conv_length`` utts ending on a patient turn).
    All active trunks advance in unison: a therapist depth branches ``M`` per trunk,
    look-ahead + oracle-scores them, and appends the best to each trunk (emitting a
    (chosen, rejected) pair when the gap exceeds ``pref_filter_tau``); a patient depth
    continues each trunk. Trunks grow until they reach ``num_utterances_for_data``
    utterances or freeze (SESSION ENDED / API failure / no valid branch score).

    Returns the same flat ``List[Dict]`` pref-pair shape as the independent path.
    """
    policy.eval()
    policy.config.use_cache = True

    trees: List[_TreeState] = []
    for s in seed_states:
        # A usable seed ends on a patient turn (next speaker = therapist) with real
        # content; skip prefixes cut short by an early SESSION ENDED.
        if s.failed or len(s.conversation) < 2 or s.next_speaker != "therapist":
            continue
        patient_sys = permutations[s.permutation_index]["patient_system_prompt"]
        # run_generation_only freed these for memory (_record_completed_state); rebuild
        # from the intact `turns` so the therapist/patient depths and
        # _process_session_response have full system+history context, not stale [].
        s.messages_Therapist_assist = turns_to_messages(s.turns, therapist_system_prompt)
        s.messages_Patient_assist = turns_to_patient_messages(s.turns, patient_sys)
        trees.append(_TreeState(conv=s, patient_system_prompt=patient_sys))
    if not trees:
        return []

    M = cfg.num_branches_per_turn
    target_len = cfg.num_utterances_for_data
    depth = 0

    while True:
        active = [
            t for t in trees
            if t.conv.is_active and len(t.conv.conversation) < target_len
        ]
        if not active:
            break

        speaker = active[0].conv.next_speaker
        # Defensive desync guard (mirror conversation_loop_batch): freeze any trunk
        # that drifted off-cadence, keeping its data.
        desynced = [t for t in active if t.conv.next_speaker != speaker]
        if desynced:
            for t in desynced:
                t.conv.is_active = False
            active = [t for t in active if t.conv.is_active]
            if not active:
                break

        if speaker == "therapist":
            await _grow_therapist_depth(
                active, client=client, policy=policy, tokenizer=tokenizer,
                therapist_system_prompt=therapist_system_prompt,
                oracle_cfg=oracle_cfg, lookahead_cfg=lookahead_cfg,
                primitives=primitives, cfg=cfg, M=M,
            )
        else:
            await _grow_patient_depth(
                active, client=client, cfg=cfg, primitives=primitives,
            )

        depth += 1
        if cfg.gen_verbose:
            n_pairs = sum(len(t.pairs) for t in trees)
            print(f"    [tree] depth {depth} ({speaker}): {len(active)} active, {n_pairs} pairs so far")

    return [p for t in trees for p in t.pairs]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  DATASET BUILDING + DPO TRAINING                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def build_iteration_datasets_dpo(
    pref_pairs: List[Dict],
    eval_split_ratio: float,
    seed: int,
):
    """Build train/eval HuggingFace datasets from pref pairs.

    Conversation-level grouping (no leakage between train/eval).
    Returns: ``(train_dataset, eval_dataset)``.
    """
    if not pref_pairs:
        raise ValueError(
            "build_iteration_datasets_dpo received 0 pref pairs. All branches tied "
            "within PREF_FILTER_TAU, or MIN_CONV_LENGTH filtered every branch point — "
            "lower PREF_FILTER_TAU / MIN_CONV_LENGTH or raise NUM_BRANCHES_PER_TURN."
        )

    conv_groups = defaultdict(list)
    for p in pref_pairs:
        conv_groups[p["conversation_id"]].append(p)

    conv_ids = sorted(conv_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    # Keep >=1 conv in train even when very few convs yield pairs (else a tiny
    # quicktest can route the only conv to eval and leave train empty).
    n_eval_convs = (
        min(max(1, int(len(conv_ids) * eval_split_ratio)), len(conv_ids) - 1)
        if len(conv_ids) >= 2 else 0
    )
    eval_conv_ids = set(conv_ids[:n_eval_convs])
    train_conv_ids = set(conv_ids[n_eval_convs:])

    train_pairs = [p for cid in train_conv_ids for p in conv_groups[cid]]
    eval_pairs = [p for cid in eval_conv_ids for p in conv_groups[cid]]
    if not train_pairs:
        raise ValueError(
            f"All {len(pref_pairs)} pref pairs landed in eval (train empty): "
            f"conv_ids={len(conv_ids)}, n_eval_convs={n_eval_convs}. "
            "Raise NUM_CONVERSATIONS_PER_ITER or lower EVAL_SPLIT_RATIO."
        )

    def _to_dataset(pairs):
        return Dataset.from_dict({
            "prompt": [p["prompt"] for p in pairs],
            "chosen": [p["chosen"] for p in pairs],
            "rejected": [p["rejected"] for p in pairs],
        })

    print(f"  Conversations: {len(train_conv_ids)} train, {len(eval_conv_ids)} eval (grouped split)")
    print(f"  Pref pairs: Train {len(train_pairs)} | Eval {len(eval_pairs)}")
    return _to_dataset(train_pairs), _to_dataset(eval_pairs)


def _build_dpo_args(cfg: PTOConfig, inner_outdir: str, num_train_pairs: int) -> DPOConfig:
    """Assemble DPOConfig for one iteration."""
    effective_batch_size = cfg.train_batch_size * cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, int(np.ceil(num_train_pairs / effective_batch_size)))
    total_train_steps = max(1, steps_per_epoch * cfg.epochs_per_iteration)
    warmup_steps = max(0, int(np.ceil(total_train_steps * cfg.warmup_steps_ratio)))
    print(f"  Warmup: {warmup_steps} steps (total_train_steps={total_train_steps})")

    tb_logging_dir = os.path.join(inner_outdir, "tb_logs")
    os.makedirs(tb_logging_dir, exist_ok=True)

    return DPOConfig(
        output_dir=inner_outdir,
        hub_model_id=cfg.current_adapter_repo,
        run_name=cfg.current_adapter_repo,  # WandbCallback names artifacts model-{run_name}; align with hub_model_id
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs_per_iteration,
        # TRL 1.4.0 DPOConfig caps the FULL tokenized prompt+completion via a single
        # max_length (the separate max_prompt_length / max_completion_length args were
        # removed). Size it as prompt cap + completion cap so the full-transcript greedy
        # prompts aren't truncated from the start (truncation_mode is 'keep_start').
        max_length=cfg.max_allowed_prompt_length + cfg.max_completion_length,
        beta=cfg.dpo_beta,
        loss_type=cfg.dpo_loss_type,
        seed=cfg.seed,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        push_to_hub=False,
        eval_strategy="epoch",
    )


def run_training_phase(
    *,
    policy, tokenizer,
    dpo_args, lora_config,
    train_dataset, eval_dataset,
    iteration: int, start_iteration: int,
    resume_checkpoint: Optional[str],
    cumulative_step_offset: int,
    iter_metadata_base: dict,
    wandb_ctx, report_to,
    tensorboard_log_dir: Optional[str] = None,
):
    """Wrap DPOTrainer with the same logging-lifecycle scaffolding as GRPO_Exp3.

    Returns: ``(updated_policy, global_step_delta, train_time)``.
    """
    train_start = time.time()
    policy.config.use_cache = False
    policy.train()

    is_already_peft = isinstance(policy, PeftModel) or hasattr(policy, "peft_config")
    peft_cfg = None if is_already_peft else lora_config

    setup_tensorboard_logging(report_to, tensorboard_log_dir)
    init_iteration_logging(report_to, iteration, cumulative_step_offset, wandb_ctx)

    trainer = DPOTrainer(
        model=policy,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_cfg,
        callbacks=[
            CheckpointMetadataCallback(iteration=iteration, metadata=iter_metadata_base),
            CumulativeStepCallback(step_offset=cumulative_step_offset, report_to=report_to),
        ],
    )

    if "tensorboard" in report_to:
        patch_trainer_tensorboard_callback(trainer, tensorboard_log_dir)

    patch_generate(trainer.model, tokenizer)

    _resume = resume_checkpoint if (iteration == start_iteration and resume_checkpoint) else None
    if _resume:
        print(f"  Resuming from HF checkpoint: {os.path.basename(_resume)}")
    trainer.train(resume_from_checkpoint=_resume)

    patch_generate(trainer.model, tokenizer)

    updated_policy = trainer.model
    resumed_steps = int(os.path.basename(_resume).split("-")[-1]) if _resume else 0
    step_delta = max(0, trainer.state.global_step - resumed_steps)
    train_time = time.time() - train_start

    del trainer
    print(f"  ✓ Training complete in {train_time:.1f}s")
    return updated_policy, step_delta, train_time


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  CHECKPOINT SAVE + ITERATION DRIVER                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def save_iteration_checkpoint(
    *,
    policy, tokenizer,
    iter_dir: str, inner_outdir: str, iteration: int,
    num_iterations: int, num_conversations: int, num_pref_pairs: int,
    avg_conv_len: float, gen_time: float, train_time: float,
    iter_metadata: dict, report_to,
    push_to_hub: bool = False, hub_repo_id: Optional[str] = None,
) -> None:
    """Save adapter + tokenizer + metadata, optionally push to Hub.

    Mirrors GRPO_Exp3's helper.
    """
    print(f"\n── Saving iteration_{iteration} checkpoint ──")

    adapter_save_path = os.path.join(iter_dir, ADAPTER_SUBDIR)
    # Save ONLY the policy adapter — not the transient frozen "ref" adapter that TRL's
    # DPOTrainer adds for PEFT DPO (a copy of the iter-start weights, used only as the
    # reference during training). Without this, iter 2+ would persist a redundant ref/
    # subfolder into every checkpoint (and push it to the Hub). Resume is unaffected:
    # from_pretrained loads the root "default" adapter, and TRL recreates "ref" next iter.
    save_adapters = (
        ["default"]
        if (isinstance(policy, PeftModel) and "ref" in getattr(policy, "peft_config", {}))
        else None
    )
    policy.save_pretrained(adapter_save_path, selected_adapters=save_adapters)
    tokenizer.save_pretrained(adapter_save_path)
    print(f"  ✓ Adapter saved: {adapter_save_path}")

    saved_ckpts = list_hf_checkpoints(inner_outdir)
    print(f"  ✓ Epoch checkpoints saved: {[os.path.basename(c) for c in saved_ckpts]}")

    iter_metadata["epoch_checkpoints"] = [os.path.basename(c) for c in saved_ckpts]
    with open(os.path.join(iter_dir, "iteration_metadata.json"), "w") as f:
        json.dump(iter_metadata, f, indent=2)

    if "tensorboard" in report_to:
        tb_dir = os.path.join(inner_outdir, "tb_logs")
        event_files = list(Path(tb_dir).rglob("events.out.tfevents.*")) if os.path.isdir(tb_dir) else []
        if event_files:
            print(f"  ✓ TensorBoard: {len(event_files)} event file(s) confirmed")
        else:
            print(f"  ⚠ TensorBoard: no event files found in {tb_dir}")

    if push_to_hub and hub_repo_id:
        try:
            policy.push_to_hub(
                repo_id=hub_repo_id,
                commit_message=f"iteration_{iteration}/{num_iterations}",
            )
            print(f"  ✓ Adapter pushed to Hub: {hub_repo_id} (iter {iteration}/{num_iterations})")
        except Exception as e:
            print(f"  ⚠ Hub push failed for iteration_{iteration} ({type(e).__name__}: {e}). "
                  f"Local copy intact at {adapter_save_path}.")

    finish_iteration_logging(report_to, iteration, {
        "iteration/num": iteration,
        "iteration/num_conversations": num_conversations,
        "iteration/num_pref_pairs": num_pref_pairs,
        "iteration/avg_conv_length": avg_conv_len,
        "iteration/generation_time_s": gen_time,
        "iteration/training_time_s": train_time,
    })


def _run_async(coro):
    """Run an async coroutine, compatible with notebooks via a fresh-loop thread.

    Mirror of the helper in ``_shared.convs._run_async`` — duplicated here so
    the trainer doesn't need to import a private function.
    """
    import threading
    from concurrent.futures import Future

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


def run_one_iteration(
    *,
    iteration: int,
    start_iteration: int,
    resume_checkpoint: Optional[str],
    cumulative_step_offset: int,
    policy, tokenizer, client,
    all_permutations,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    lora_config,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    wandb_ctx,
    cfg: PTOConfig,
):
    """One full PTO iteration: generate → build pref pairs → DPO train → save.

    Returns ``(new_policy, step_delta, completed_count, pref_pair_count)``.
    """
    iter_start_time = time.time()
    iter_dir = os.path.join(cfg.local_outdir, f"{ITER_PREFIX}{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    conv_dir = _conv_dir_for_iter(cfg, iteration - 1)

    iter_rng = random.Random(cfg.seed + iteration)
    shuffled = list(all_permutations)
    iter_rng.shuffle(shuffled)
    active_permutations = shuffled[: cfg.num_conversations_per_iter]

    print("\n" + "=" * 70)
    print(f"PTO ITERATION {iteration}/{cfg.num_iterations}  (conv from model_iter_{iteration - 1})")
    print("=" * 70)

    # ── Step 1: Generate conversations ──
    print(f"\n── Step 1: Generating {len(active_permutations)} conversations ──")
    completed_states, gen_time, avg_conv_len = run_generation_only(
        policy=policy, tokenizer=tokenizer, client=client,
        active_permutations=active_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=conv_dir, cfg=cfg,
        patient_api_seed=cfg.seed + iteration,
    )

    # Free KV caches before pref-tree construction (it uses different batching)
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 2: Build pref pairs (mode-dependent) ──
    print(f"\n── Step 2: Building pref pairs [mode={cfg.pref_tree_mode}] "
          f"(num_branches={cfg.num_branches_per_turn}, tau={cfg.pref_filter_tau}, "
          f"min_conv_length={cfg.min_conv_length}) ──")
    pref_start = time.time()
    if cfg.pref_tree_mode == "greedy":
        # Greedy true-PTO: SLICE the first MCL utts off each Step-1 conv (ending on a
        # patient turn) → grow each trunk by appending best-of-M. No separate prefix
        # generation pass — the seeds reuse the Step-1 openings (then diverge); the
        # Step-1 convs themselves stay the frozen model_iter_* eval data.
        seed_states = _slice_prefix_seeds(completed_states, cfg.min_conv_length)
        print(f"  Sliced {len(seed_states)} tree prefixes ({cfg.min_conv_length} utts, "
              f"ending on a patient turn) from {len(completed_states)} Step-1 convs")
        gc.collect()
        torch.cuda.empty_cache()
        pref_pairs = _run_async(grow_preference_trees_batch(
            seed_states=seed_states,
            permutations=active_permutations,
            policy=policy, tokenizer=tokenizer, client=client,
            therapist_system_prompt=therapist_system_prompt,
            oracle_cfg=oracle_cfg, lookahead_cfg=lookahead_cfg,
            primitives=primitives, cfg=cfg,
        ))
    else:  # "independent" — branch each patient turn of the (pre-recorded) eval convs
        pref_pairs = _run_async(extract_pref_pairs_from_conversations(
            completed_states=completed_states,
            permutations=active_permutations,
            policy=policy, tokenizer=tokenizer, client=client,
            therapist_system_prompt=therapist_system_prompt,
            oracle_cfg=oracle_cfg, lookahead_cfg=lookahead_cfg,
            primitives=primitives,
            cfg=cfg,
        ))
    pref_time = time.time() - pref_start
    print(f"  ✓ Built {len(pref_pairs)} pref pairs in {pref_time:.1f}s "
          f"(from {len(completed_states)} conversations)")

    # Persist the per-iter pref pairs to disk so the run is auditable.
    pref_dir = os.path.join(iter_dir, "pref_pairs")
    os.makedirs(pref_dir, exist_ok=True)
    pd.DataFrame(pref_pairs).to_csv(
        os.path.join(pref_dir, "pairs.csv"), index=False
    )
    print(f"  ✓ Pref pairs saved: {pref_dir}/pairs.csv")

    # Fail fast (with guidance) if this iteration produced no trainable signal.
    # The (empty) pairs.csv above is kept as evidence.
    if not pref_pairs:
        raise ValueError(
            f"Iteration {iteration} produced 0 pref pairs from {len(completed_states)} "
            f"conversations (avg len {avg_conv_len:.1f} utts). Every branch tied within "
            f"PREF_FILTER_TAU={cfg.pref_filter_tau}, or MIN_CONV_LENGTH={cfg.min_conv_length} "
            f"filtered every branch point. Lower PREF_FILTER_TAU / MIN_CONV_LENGTH, or raise "
            f"NUM_BRANCHES_PER_TURN (currently {cfg.num_branches_per_turn})."
        )

    # Free KV caches before training setup
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 3: Datasets ──
    train_dataset, eval_dataset = build_iteration_datasets_dpo(
        pref_pairs=pref_pairs,
        eval_split_ratio=cfg.eval_split_ratio,
        seed=cfg.seed,
    )

    # ── Step 4: Train DPO ──
    print(f"\n── Step 4: Training DPO for {cfg.epochs_per_iteration} epochs ──")
    inner_outdir = os.path.join(iter_dir, "training")
    dpo_args = _build_dpo_args(cfg, inner_outdir, num_train_pairs=len(train_dataset))
    tb_logging_dir = os.path.join(inner_outdir, "tb_logs")

    iter_metadata_base = {
        "experiment_name": cfg.experiment_name,
        "method": "PTO_Exp3",
        "iteration": iteration,
        "base_model": cfg.base_model_id,
        "oracle_model": cfg.oracle_model_id,
        "questionnaire_ids": list(cfg.questionnaire_ids),
        "min_conv_length": cfg.min_conv_length,
        "num_branches_per_turn": cfg.num_branches_per_turn,
        "pref_filter_tau": cfg.pref_filter_tau,
        "dpo_beta": cfg.dpo_beta,
        "dpo_loss_type": cfg.dpo_loss_type,
        "learning_rate": cfg.learning_rate,
        "lora_r": cfg.lora_r,
    }

    new_policy, step_delta, train_time = run_training_phase(
        policy=policy, tokenizer=tokenizer,
        dpo_args=dpo_args, lora_config=lora_config,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        iteration=iteration, start_iteration=start_iteration,
        resume_checkpoint=resume_checkpoint,
        cumulative_step_offset=cumulative_step_offset,
        iter_metadata_base=iter_metadata_base,
        wandb_ctx=wandb_ctx, report_to=cfg.report_to,
        tensorboard_log_dir=tb_logging_dir,
    )

    # ── Step 5: Save ──
    iter_metadata = {
        **iter_metadata_base,
        "num_conversations": len(completed_states),
        "num_pref_pairs": len(pref_pairs),
        "avg_conversation_length": float(avg_conv_len),
        "epochs_per_iteration": cfg.epochs_per_iteration,
        "generation_time_s": gen_time,
        "pref_pair_time_s": pref_time,
        "training_time_s": train_time,
    }
    save_iteration_checkpoint(
        policy=new_policy, tokenizer=tokenizer,
        iter_dir=iter_dir, inner_outdir=inner_outdir, iteration=iteration,
        num_iterations=cfg.num_iterations,
        num_conversations=len(completed_states), num_pref_pairs=len(pref_pairs),
        avg_conv_len=avg_conv_len, gen_time=gen_time, train_time=train_time,
        iter_metadata=iter_metadata, report_to=cfg.report_to,
        push_to_hub=cfg.push_to_hub,
        hub_repo_id=cfg.current_adapter_repo,
    )

    iter_time = time.time() - iter_start_time
    print(f"\n  ✓ Iteration {iteration} complete in {iter_time:.1f}s")
    print(f"    Conversations: {len(completed_states)} | Pref pairs: {len(pref_pairs)}")
    print("=" * 70)

    return new_policy, step_delta, len(completed_states), len(pref_pairs)


def run_final_eval(
    *,
    policy, tokenizer, client,
    all_permutations,
    therapist_system_prompt, therapist_init_utterance,
    cfg: PTOConfig,
) -> str:
    """Generate one more conversation set with the final policy."""
    final_label = cfg.num_iterations
    final_conv_dir = _conv_dir_for_iter(cfg, final_label)

    print("\n" + "=" * 70)
    print(f"FINAL ADAPTER EVAL — model_iter_{final_label}")
    print("=" * 70)

    final_seed = cfg.seed + cfg.num_iterations + 1
    final_rng = random.Random(final_seed)
    final_shuffled = list(all_permutations)
    final_rng.shuffle(final_shuffled)
    final_active = final_shuffled[: cfg.num_conversations_per_iter]

    final_states, final_gen_time, final_avg_len = run_generation_only(
        policy=policy, tokenizer=tokenizer, client=client,
        active_permutations=final_active,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=final_conv_dir,
        cfg=cfg,
        patient_api_seed=final_seed,
    )
    print(
        f"  ✓ Final eval done in {final_gen_time:.1f}s — "
        f"{len(final_states)} conv, avg len {final_avg_len:.1f}"
    )
    print(f"    Saved to: {final_conv_dir}")

    del final_states
    gc.collect()
    torch.cuda.empty_cache()
    return final_conv_dir
