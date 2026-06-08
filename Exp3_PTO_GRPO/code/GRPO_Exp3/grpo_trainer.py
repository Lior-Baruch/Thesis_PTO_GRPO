"""
grpo_trainer.py — GRPO_Exp3 orchestration helpers and convenience entry point.

The user-facing pattern is to compose the helpers visibly from the notebook:

    cfg = TrainingConfig(...)
    write_run_metadata(cfg)
    start_iter, policy, resume_ckpt = resolve_start_state(cfg.local_outdir, base_policy, tokenizer)
    cumulative = compute_cumulative_step_offset(cfg.local_outdir)
    for iteration in range(start_iter, cfg.num_iterations + 1):
        policy, step_delta, *_ = run_one_iteration(
            iteration=iteration, start_iteration=start_iter,
            resume_checkpoint=resume_ckpt, cumulative_step_offset=cumulative,
            policy=policy, tokenizer=tokenizer, client=client,
            all_permutations=all_permutations,
            therapist_system_prompt=therapist_system_prompt,
            therapist_init_utterance=therapist_init_utterance,
            lora_config=lora_config, reward_factory=reward_factory,
            wandb_ctx=wandb_ctx, cfg=cfg,
        )
        cumulative += step_delta
        resume_ckpt = None  # one-shot
    run_final_eval(policy, tokenizer, client, all_permutations, ..., cfg)

For a one-call convenience the legacy :func:`run_iterative_training` wraps
the loop. New code should prefer the explicit loop in the notebook so the
"control" lives where the user can see it.

One iteration is:
1. Generate ``num_conversations_per_iter`` conversations with the current policy
   and extract per-turn training prompts.
2. Build train/eval HuggingFace datasets (conversation-level split, no leakage).
3. Train GRPO for ``epochs_per_iteration`` epochs.
4. Save adapter + metadata + finish logging.

After the loop, an extra "final adapter eval" pass generates one more
conversation set so the final adapter has matched eval data — without paying
for the prompt-extraction step (its output is discarded for the final eval).
"""

import os
import gc
import json
import time
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from _shared import (
    # checkpoint discovery + iteration resume
    list_iteration_checkpoints, list_hf_checkpoints,
    resolve_start_state, compute_cumulative_step_offset,
    patch_generate, ADAPTER_SUBDIR, ITER_PREFIX,
    # convs
    generate_all_conversations, extract_prompts_from_conversations,
    # logging lifecycle + TB callbacks
    CheckpointMetadataCallback, CumulativeStepCallback,
    init_iteration_logging, finish_iteration_logging,
    setup_tensorboard_logging, patch_trainer_tensorboard_callback,
    # EDA capture
    EDARecorder,
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIG DATACLASS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class TrainingConfig:
    """Loop knobs for :func:`run_iterative_training`.

    Bundles the notebook's flat config globals into one transport object so the
    function signature stays manageable. Build it in a single notebook cell:

        cfg = TrainingConfig(
            num_iterations=NUM_ITERATIONS, seed=SEED, ...,
        )

    All fields are required — the notebook is the source of truth for every
    knob, so we don't carry defaults here. Missing a field is a config error,
    not a "use the default" signal.
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

    # Training
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    grpo_inner_iterations: int
    num_generations: int
    max_completion_length: int
    grpo_beta: float
    grpo_temperature: float
    grpo_loss_type: str
    warmup_steps_ratio: float
    logging_steps: int
    save_strategy: str
    save_steps: int          # checkpoint cadence when save_strategy="steps"
    save_total_limit: Optional[int]
    eval_split_ratio: float
    log_completions: bool
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

    # EDA capture + live TensorBoard. Exception to the "all fields required" rule
    # above: these are OPTIONAL observability features, defaulted off so they never
    # silently change a training run. The notebook still sets them explicitly.
    # save_eda_generations: write iteration_N/eda/generations.jsonl with every
    #   GRPO candidate (all G per prompt-group) + scores + sub-scores + look-ahead.
    # save_lookahead_transcripts: keep the heavy per-candidate K-turn transcript.
    # tb_live_logging: run-level continuous SummaryWriter (smoothable TB web UI).
    # tb_sample_completions_n: how many spread-sampled completions to log as TB text.
    save_eda_generations: bool = False
    save_lookahead_transcripts: bool = True
    tb_live_logging: bool = False
    tb_sample_completions_n: int = 8


@dataclass
class TrainingSummary:
    """Returned by :func:`run_iterative_training`."""
    iterations_run: List[int]
    iterations_saved: List[int]
    final_conv_dir: Optional[str]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  W&B CONTEXT (GRPO-specific config fields)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def build_wandb_ctx(cfg: TrainingConfig) -> Optional[dict]:
    """Build the wandb context dict from cfg, or None if not reporting to wandb.

    Closes any pre-existing wandb run so a clean run starts.
    """
    if "wandb" not in cfg.report_to:
        return None
    if wandb.run is not None:
        wandb.finish()
    project = f"GRPO_Iterative_Experiments_{cfg.mode_tag.upper()}"
    run_id = f"{cfg.experiment_name}_{cfg.mode_tag}"
    ctx = {
        "run_id": run_id,
        "project": project,
        "config": {
            "experiment_name": cfg.experiment_name,
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
            "learning_rate": cfg.learning_rate,
            "num_generations": cfg.num_generations,
            "grpo_beta": cfg.grpo_beta,
            "grpo_temperature": cfg.grpo_temperature,
            "train_batch_size": cfg.train_batch_size,
            "max_completion_length": cfg.max_completion_length,
            "questionnaire_ids": cfg.questionnaire_ids,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
        },
    }
    print(f"✓ W&B config ready: project={project}, run={run_id}")
    return ctx


def write_run_metadata(cfg: TrainingConfig) -> str:
    """Write a top-level ``run_metadata.json`` describing the entire sweep.

    Snapshots every TrainingConfig field plus a timestamp. Idempotent: a run
    that resumes overwrites the previous file with the same (or updated)
    config. Returns the path written.
    """
    os.makedirs(cfg.local_outdir, exist_ok=True)
    payload = {
        "experiment_name": cfg.experiment_name,
        "mode_tag": cfg.mode_tag,
        "started_at": pd.Timestamp.now().isoformat(),
        "config": asdict(cfg),
    }
    # Serialize tuples as lists for JSON friendliness
    cfg_dict = payload["config"]
    if isinstance(cfg_dict.get("questionnaire_ids"), tuple):
        cfg_dict["questionnaire_ids"] = list(cfg_dict["questionnaire_ids"])

    path = os.path.join(cfg.local_outdir, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓ Run metadata: {path}")
    return path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    PER-ITERATION HELPERS                                   ║
# ║ (resume helpers + TB callbacks now live in _shared/{model,tb_plots}.py)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def run_generation_only(
    *,
    policy, tokenizer, client,
    active_permutations,
    therapist_system_prompt, therapist_init_utterance,
    conv_dir: str,
    cfg: TrainingConfig,
    patient_api_seed: Optional[int] = None,
):
    """Generate conversations (eval mode) — no prompt extraction.

    Switches policy to eval mode. Caller is responsible for switching back
    to train mode before training. Used by :func:`run_final_eval`, which
    doesn't need training prompts.

    Returns: ``(completed_states, gen_time, avg_conv_len)``.
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
        num_utterances=cfg.num_utterances_for_data,
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


def run_generation_phase(
    *,
    policy, tokenizer, client,
    active_permutations,
    therapist_system_prompt, therapist_init_utterance,
    conv_dir: str,
    cfg: TrainingConfig,
    patient_api_seed: Optional[int] = None,
):
    """Generate + extract per-turn training prompts. Used by per-iteration training.

    Composes :func:`run_generation_only` with prompt extraction. The final-eval
    path skips this and calls :func:`run_generation_only` directly.

    Returns: ``(completed_states, all_prompts, gen_time, avg_conv_len)``.
    """
    completed_states, gen_time, avg_conv_len = run_generation_only(
        policy=policy, tokenizer=tokenizer, client=client,
        active_permutations=active_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=conv_dir, cfg=cfg,
        patient_api_seed=patient_api_seed,
    )

    print("\n── Step 2: Extracting prompts ──")
    all_prompts = extract_prompts_from_conversations(
        completed_states=completed_states,
        system_prompt=therapist_system_prompt,
        tokenizer=tokenizer,
        min_conv_length=cfg.min_conv_length,
        max_prompt_tokens=cfg.max_allowed_prompt_length,
        permutations=active_permutations,
    )
    print(
        f"  ✓ Extracted {len(all_prompts)} training prompts "
        f"(min_conv_length={cfg.min_conv_length})"
    )

    return completed_states, all_prompts, gen_time, avg_conv_len


def build_iteration_datasets(
    all_prompts: list,
    eval_split_ratio: float,
    seed: int,
    conv_dir: str,
):
    """Build train/eval HuggingFace datasets from extracted prompts.

    Uses conversation-level grouping so all prompts from one conversation end
    up in either train OR eval — never both. Prevents leakage from shared
    conversational context.

    Returns: ``(train_dataset, eval_dataset, full_dataset)``.
    """
    assert len(all_prompts) > 0, f"No prompts extracted! Check conversations in {conv_dir}"

    conv_groups = defaultdict(list)
    for p in all_prompts:
        conv_groups[p["conversation_id"]].append(p)

    conv_ids = sorted(conv_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    n_eval_convs = max(1, int(len(conv_ids) * eval_split_ratio))
    eval_conv_ids = set(conv_ids[:n_eval_convs])
    train_conv_ids = set(conv_ids[n_eval_convs:])

    overlap = train_conv_ids & eval_conv_ids
    assert len(overlap) == 0, f"Train/eval conversation overlap detected: {overlap}"

    train_prompts = [p for cid in train_conv_ids for p in conv_groups[cid]]
    eval_prompts = [p for cid in eval_conv_ids for p in conv_groups[cid]]

    def _to_dataset(prompts):
        d = {
            "prompt": [p["prompt"] for p in prompts],
            "transcript": [p["transcript"] for p in prompts],
            "conversation_id": [p["conversation_id"] for p in prompts],
        }
        if prompts and "patient_system_prompt" in prompts[0]:
            d["patient_system_prompt"] = [p["patient_system_prompt"] for p in prompts]
        return Dataset.from_dict(d)

    full_dataset = _to_dataset(all_prompts)
    train_dataset = _to_dataset(train_prompts)
    eval_dataset = _to_dataset(eval_prompts)

    print(f"  Conversations: {len(train_conv_ids)} train, {len(eval_conv_ids)} eval (grouped split, 0 overlap)")
    print(f"  Prompts: Train {len(train_dataset)} | Eval {len(eval_dataset)}")
    return train_dataset, eval_dataset, full_dataset


def run_training_phase(
    *,
    policy,
    tokenizer,
    grpo_args,
    lora_config,
    reward_funcs,
    train_dataset,
    eval_dataset,
    iteration: int,
    start_iteration: int,
    resume_checkpoint: Optional[str],
    cumulative_step_offset: int,
    iter_metadata_base: dict,
    wandb_ctx,
    report_to,
    tensorboard_log_dir: Optional[str] = None,
    recorder=None,
):
    """Create GRPOTrainer, train, and return updated policy.

    Handles unified logging init, LoRA adapter continuity (skip ``peft_config``
    if ``policy`` is already a PeftModel), one-shot resume checkpoint
    consumption, and policy handoff from trainer → caller.

    Returns: ``(updated_policy, global_step_delta, train_time)``.
    """
    train_start = time.time()

    policy.config.use_cache = False
    policy.train()

    # Robust PEFT detection: don't add a second adapter stack.
    is_already_peft = isinstance(policy, PeftModel) or hasattr(policy, "peft_config")
    peft_cfg = None if is_already_peft else lora_config

    setup_tensorboard_logging(report_to, tensorboard_log_dir)
    init_iteration_logging(report_to, iteration, cumulative_step_offset, wandb_ctx)

    trainer = GRPOTrainer(
        model=policy,
        args=grpo_args,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_cfg,
        callbacks=[
            # recorder → also snapshots the EDA buffer into each checkpoint at
            # on_save, so a mid-iteration crash + resume can reload the pre-crash
            # candidates (HF fast-forwards skipped steps without re-recording).
            CheckpointMetadataCallback(
                iteration=iteration, metadata=iter_metadata_base, recorder=recorder
            ),
        ],
    )

    if "tensorboard" in report_to:
        patch_trainer_tensorboard_callback(trainer, tensorboard_log_dir)

    patch_generate(trainer.model, tokenizer)

    # Resume only on the first iteration after a crash.
    _resume = resume_checkpoint if (iteration == start_iteration and resume_checkpoint) else None
    if _resume:
        print(f"  Resuming from HF checkpoint: {os.path.basename(_resume)}")
    trainer.train(resume_from_checkpoint=_resume)

    patch_generate(trainer.model, tokenizer)

    updated_policy = trainer.model
    # On resume, global_step includes previously completed steps. Return only
    # newly executed steps to avoid double-counting cumulative offsets.
    resumed_steps = int(os.path.basename(_resume).split("-")[-1]) if _resume else 0
    step_delta = max(0, trainer.state.global_step - resumed_steps)
    train_time = time.time() - train_start

    del trainer  # model stays alive via updated_policy

    print(f"  ✓ Training complete in {train_time:.1f}s")
    return updated_policy, step_delta, train_time


def save_iteration_checkpoint(
    *,
    policy,
    tokenizer,
    iter_dir: str,
    inner_outdir: str,
    iteration: int,
    num_iterations: int,
    num_conversations: int,
    num_prompts: int,
    avg_conv_len: float,
    gen_time: float,
    train_time: float,
    iter_metadata: dict,
    report_to,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
) -> None:
    """Save adapter + tokenizer, write metadata JSON, finalize logging backend.

    When ``push_to_hub`` is True, also pushes the adapter to ``hub_repo_id``
    as a per-iteration safety net. Push failures are logged but do NOT abort
    training — the local copy is still on disk.
    """
    print(f"\n── Step 4: Saving iteration_{iteration} checkpoint ──")

    adapter_save_path = os.path.join(iter_dir, ADAPTER_SUBDIR)
    policy.save_pretrained(adapter_save_path)
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

    # ── Hub push (per-iter safety net) ──
    if push_to_hub and hub_repo_id:
        try:
            policy.push_to_hub(
                repo_id=hub_repo_id,
                commit_message=f"iteration_{iteration}/{num_iterations}",
            )
            print(f"  ✓ Adapter pushed to Hub: {hub_repo_id} (iter {iteration}/{num_iterations})")
        except Exception as e:
            print(f"  ⚠ Hub push failed for iteration_{iteration} ({type(e).__name__}: {e}).")
            print("    Continuing — local copy is intact at {}".format(adapter_save_path))

    finish_iteration_logging(report_to, iteration, {
        "iteration/num": iteration,
        "iteration/num_conversations": num_conversations,
        "iteration/num_prompts": num_prompts,
        "iteration/avg_conv_length": avg_conv_len,
        "iteration/generation_time_s": gen_time,
        "iteration/training_time_s": train_time,
    })


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  TOP-LEVEL ORCHESTRATOR                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _build_grpo_args(cfg: TrainingConfig, inner_outdir: str, num_train_prompts: int) -> GRPOConfig:
    """Compute warmup_steps and assemble GRPOConfig for one iteration."""
    # GRPO's per_device_train_batch_size counts *completions*, not prompts: TRL's
    # RepeatSampler emits num_generations completions per prompt, so the unique
    # prompts consumed per optimizer step is
    #   (train_batch_size / num_generations) * gradient_accumulation_steps.
    # Dividing num_train_prompts by the raw train_batch_size*grad_accum under-counts
    # steps by num_generations× (e.g. printed 21 vs the real ~150 at G=8), which only
    # under-sized the warmup print/value — the HF Trainer recomputes the cosine
    # horizon from the real dataloader length, so the LR schedule itself was fine.
    prompts_per_device = max(1, cfg.train_batch_size // max(1, cfg.num_generations))
    prompts_per_step = max(1, prompts_per_device * cfg.gradient_accumulation_steps)
    steps_per_epoch = max(1, int(np.ceil(num_train_prompts / prompts_per_step)))
    total_train_steps = max(1, steps_per_epoch * cfg.epochs_per_iteration)
    warmup_steps = max(0, int(np.ceil(total_train_steps * cfg.warmup_steps_ratio)))
    print(f"  Warmup: {warmup_steps} steps (total_train_steps={total_train_steps}, ratio={cfg.warmup_steps_ratio})")

    # Explicit TensorBoard path; consumed by Trainer callbacks via TENSORBOARD_LOGGING_DIR.
    # Avoids default_logdir() producing illegal Windows folder names (WinError 123).
    tb_logging_dir = os.path.join(inner_outdir, "tb_logs")
    os.makedirs(tb_logging_dir, exist_ok=True)

    return GRPOConfig(
        output_dir=inner_outdir,
        hub_model_id=cfg.current_adapter_repo,
        run_name=cfg.current_adapter_repo,  # WandbCallback names artifacts model-{run_name}; align with hub_model_id
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs_per_iteration,
        num_iterations=cfg.grpo_inner_iterations,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        beta=cfg.grpo_beta,
        temperature=cfg.grpo_temperature,
        scale_rewards="group",  # Original GRPO group normalization
        # Stop TRL's in-loop sampling at the turn terminator. <|im_end|> is template
        # text (not the base tokenizer's eos), so without this the 1B policy samples
        # straight through it to max_completion_length — fabricating the patient's
        # reply + the next turn (self-play). That polluted the oracle-scored transcript
        # AND, because GRPO credits every sampled token, trained the policy toward
        # 200-token rambles (length/entropy collapse). patch_generate() injects the
        # tokenizer into model.generate so stop_strings works (it's the same patched
        # path look-ahead already relies on during the step).
        generation_kwargs=({"stop_strings": list(cfg.stop_strings)} if cfg.stop_strings else None),
        loss_type=cfg.grpo_loss_type,
        bf16=True,  # bf16 autocast for LoRA params (correct for both bf16 + 4-bit-bf16-compute bases; A100 supports it)
        seed=cfg.seed,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        log_completions=cfg.log_completions,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,  # honored only when save_strategy="steps"
        save_total_limit=cfg.save_total_limit,
        push_to_hub=False,
        eval_strategy="epoch",
    )


def _conv_dir_for_iter(cfg: TrainingConfig, model_iter_label: int) -> str:
    """``CONV_OUTDIR/model_iter_<label>_TT<temp_t>_TP<temp_p>/`` — folder is
    labeled by the *generating model*, not the consuming training iteration."""
    return os.path.join(
        cfg.conv_outdir,
        f"model_iter_{model_iter_label}"
        f"_TT{cfg.temperature_therapist_gen}_TP{cfg.temperature_patient}",
    )


def run_one_iteration(
    *,
    iteration: int,
    start_iteration: int,
    resume_checkpoint: Optional[str],
    cumulative_step_offset: int,
    policy,
    tokenizer,
    client,
    all_permutations,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    lora_config,
    reward_factory: Callable,
    wandb_ctx,
    tb_logger=None,
    cfg: TrainingConfig,
):
    """One full iteration: generate → split → train → save. Returns
    ``(new_policy, step_delta, completed_count, prompt_count)``."""
    iter_start_time = time.time()
    iter_dir = os.path.join(cfg.local_outdir, f"{ITER_PREFIX}{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    conv_dir = _conv_dir_for_iter(cfg, iteration - 1)

    iter_rng = random.Random(cfg.seed + iteration)
    shuffled = list(all_permutations)
    iter_rng.shuffle(shuffled)
    active_permutations = shuffled[: cfg.num_conversations_per_iter]

    print("\n" + "=" * 70)
    print(f"ITERATION {iteration}/{cfg.num_iterations}  (conv from model_iter_{iteration - 1})")
    print("=" * 70)

    # ── Steps 1+2: Generate conversations & extract prompts ──
    print(f"\n── Step 1: Generating {len(active_permutations)} conversations ──")
    completed_states, all_prompts, gen_time, avg_conv_len = run_generation_phase(
        policy=policy, tokenizer=tokenizer, client=client,
        active_permutations=active_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=conv_dir,
        cfg=cfg,
        patient_api_seed=cfg.seed + iteration,
    )

    # Free KV caches before training setup
    gc.collect()
    torch.cuda.empty_cache()

    # ── Datasets ──
    train_dataset, eval_dataset, _ = build_iteration_datasets(
        all_prompts=all_prompts,
        eval_split_ratio=cfg.eval_split_ratio,
        seed=cfg.seed,
        conv_dir=conv_dir,
    )

    # ── Step 3: Train ──
    print(f"\n── Step 3: Training GRPO for {cfg.epochs_per_iteration} epochs ──")
    inner_outdir = os.path.join(iter_dir, "training")
    grpo_args = _build_grpo_args(cfg, inner_outdir, num_train_prompts=len(train_dataset))
    tb_logging_dir = os.path.join(inner_outdir, "tb_logs")

    # The full sweep-arm identity (K, MCL, G) is encoded in ``experiment_name``
    # itself, so iteration_metadata.json is self-describing without needing
    # a separate K marker.
    iter_metadata_base = {
        "experiment_name": cfg.experiment_name,
        "iteration": iteration,
        "base_model": cfg.base_model_id,
        "oracle_model": cfg.oracle_model_id,
        "questionnaire_ids": list(cfg.questionnaire_ids),
        "min_conv_length": cfg.min_conv_length,
        "grpo_beta": cfg.grpo_beta,
        "learning_rate": cfg.learning_rate,
        "lora_r": cfg.lora_r,
    }

    # Per-iteration EDA recorder. The reward fn appends one record per GRPO
    # candidate (all G per prompt-group) in-memory; we flush once after training.
    recorder = EDARecorder(
        os.path.join(iter_dir, "eda", "generations.jsonl"),
        enabled=getattr(cfg, "save_eda_generations", False),
        save_transcripts=getattr(cfg, "save_lookahead_transcripts", True),
    )

    # Build the reward function *with the current policy* so look-ahead sees
    # the iteration's current weights. See ``_shared.reward.make_reward_fn``.
    reward_fn = reward_factory(policy, recorder=recorder, iteration=iteration)

    # Mid-iteration resume: reload the EDA snapshot saved alongside the checkpoint
    # we're resuming from, so the post-resume flush keeps the pre-crash candidates
    # (HF fast-forwards skipped steps without re-invoking the reward fn). One-shot,
    # matching run_training_phase's resume consumption; no-op if the snapshot is
    # absent (e.g. checkpoints written before this feature).
    if resume_checkpoint and iteration == start_iteration and recorder.enabled:
        n_loaded = recorder.load_from(os.path.join(resume_checkpoint, "eda_snapshot.jsonl"))
        if n_loaded:
            print(f"  ✓ Reloaded {n_loaded} EDA candidate rows from {os.path.basename(resume_checkpoint)} snapshot")

    new_policy, step_delta, train_time = run_training_phase(
        policy=policy, tokenizer=tokenizer,
        grpo_args=grpo_args, lora_config=lora_config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        iteration=iteration, start_iteration=start_iteration,
        resume_checkpoint=resume_checkpoint,
        cumulative_step_offset=cumulative_step_offset,
        iter_metadata_base=iter_metadata_base,
        wandb_ctx=wandb_ctx, report_to=cfg.report_to,
        tensorboard_log_dir=tb_logging_dir,
        recorder=recorder,
    )

    # Persist the full per-candidate EDA records (all G per group, train + eval).
    recorder.flush()
    if recorder.enabled:
        print(f"  ✓ EDA generations saved: {recorder.out_path} ({len(recorder.records)} candidate rows)")

    # ── Live TensorBoard / W&B: per-iteration EDA aggregates at the iteration's
    #    end-of-training cumulative step (continuous, smoothable run-level view). ──
    if tb_logger is not None and recorder.records:
        scalars, scores = recorder.aggregate()
        scalars["iteration/num_prompts"] = float(len(all_prompts))
        end_step = cumulative_step_offset + step_delta
        tb_logger.log_scalars(scalars, step=end_step, iteration=iteration)
        if scores:
            tb_logger.log_histogram("eda/candidate_reward_hist", scores, step=end_step, iteration=iteration)
        samples = recorder.sample_for_display(getattr(cfg, "tb_sample_completions_n", 0))
        if samples:
            tb_logger.log_sample_completions(samples, step=end_step, iteration=iteration)

    # ── Step 4: Save ──
    iter_metadata = {
        **iter_metadata_base,
        "num_conversations": len(completed_states),
        "num_prompts": len(all_prompts),
        "avg_conversation_length": float(avg_conv_len),
        "epochs_per_iteration": cfg.epochs_per_iteration,
        "generation_time_s": gen_time,
        "training_time_s": train_time,
    }
    save_iteration_checkpoint(
        policy=new_policy, tokenizer=tokenizer,
        iter_dir=iter_dir, inner_outdir=inner_outdir, iteration=iteration,
        num_iterations=cfg.num_iterations,
        num_conversations=len(completed_states), num_prompts=len(all_prompts),
        avg_conv_len=avg_conv_len, gen_time=gen_time, train_time=train_time,
        iter_metadata=iter_metadata, report_to=cfg.report_to,
        push_to_hub=cfg.push_to_hub,
        hub_repo_id=cfg.current_adapter_repo,
    )

    iter_time = time.time() - iter_start_time
    print(f"\n  ✓ Iteration {iteration} complete in {iter_time:.1f}s")
    print(f"    Conversations: {len(completed_states)} | Prompts: {len(all_prompts)}")
    print("=" * 70)

    return new_policy, step_delta, len(completed_states), len(all_prompts)


def run_final_eval(
    *,
    policy, tokenizer, client,
    all_permutations,
    therapist_system_prompt, therapist_init_utterance,
    cfg: TrainingConfig,
) -> str:
    """Generate one more conversation set with the final policy so the final
    adapter has a matched-shape eval folder ``model_iter_<NUM_ITERATIONS>/``.

    Skips prompt extraction — the final eval doesn't need training prompts.
    ``generate_all_conversations`` skips conversations whose CSVs already exist,
    so re-running this is a no-op after a successful run.
    """
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


def run_iterative_training(
    *,
    base_policy,
    tokenizer,
    client,
    all_permutations,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    reward_factory: Callable,
    lora_config,
    cfg: TrainingConfig,
) -> TrainingSummary:
    """Top-level orchestrator: resume → loop → final eval → optional Hub push.

    Args:
        reward_factory: Callable ``policy -> async reward_fn``. Called once per
            iteration with the iteration's current policy. See
            :func:`oracle_reward.make_reward_fn`.
    """
    wandb_ctx = build_wandb_ctx(cfg)
    print(f"✓ Logging backends: {sorted(set(cfg.report_to))}")

    # Snapshot the full TrainingConfig once per run start. Resuming overwrites
    # with current state — that's fine; the per-iter iteration_metadata.json
    # files are the authoritative per-iteration record.
    write_run_metadata(cfg)

    # ── Resolve start state ──
    start_iteration, policy, resume_checkpoint = resolve_start_state(
        local_outdir=cfg.local_outdir,
        base_policy=base_policy,
        tokenizer=tokenizer,
    )
    print(f"  Starting from iteration {start_iteration}")
    print("=" * 60)

    cumulative_step_offset = compute_cumulative_step_offset(cfg.local_outdir)

    # NOTE: Each iteration creates a new GRPOTrainer with a fresh optimizer.
    # LoRA adapter weights carry over, but Adam momentum/variance reset.
    # Effectively warm-restart training.

    iterations_run: List[int] = []
    for iteration in range(start_iteration, cfg.num_iterations + 1):
        policy, step_delta, _ncompleted, _nprompts = run_one_iteration(
            iteration=iteration,
            start_iteration=start_iteration,
            resume_checkpoint=resume_checkpoint,
            cumulative_step_offset=cumulative_step_offset,
            policy=policy, tokenizer=tokenizer, client=client,
            all_permutations=all_permutations,
            therapist_system_prompt=therapist_system_prompt,
            therapist_init_utterance=therapist_init_utterance,
            lora_config=lora_config,
            reward_factory=reward_factory,
            wandb_ctx=wandb_ctx,
            cfg=cfg,
        )
        cumulative_step_offset += step_delta
        resume_checkpoint = None  # consumed; subsequent iterations start fresh
        iterations_run.append(iteration)

        gc.collect()
        torch.cuda.empty_cache()

    # ── Final adapter eval ──
    final_conv_dir = run_final_eval(
        policy=policy, tokenizer=tokenizer, client=client,
        all_permutations=all_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        cfg=cfg,
    )

    # ── Final Hub marker push ──
    # Each iteration already pushes its adapter inside save_iteration_checkpoint
    # (per-iter safety net). This final push just stamps a clearer "sweep
    # complete" commit message on top so the latest commit is easy to identify.
    if cfg.push_to_hub:
        print("\nPushing final 'sweep complete' commit to Hub...")
        try:
            policy.push_to_hub(
                repo_id=cfg.current_adapter_repo,
                commit_message=f"Sweep complete — {cfg.num_iterations} iterations",
            )
            print(f"  ✓ Pushed to: {cfg.current_adapter_repo}")
        except Exception as e:
            print(f"  ⚠ Final Hub push failed ({type(e).__name__}: {e}). "
                  f"Per-iter pushes already covered each iteration.")

    if "wandb" in cfg.report_to and wandb.run is not None:
        wandb.finish()
        print("✓ W&B run finalized")

    # ── Summary ──
    iterations_saved = [n for n, _ in list_iteration_checkpoints(cfg.local_outdir)]
    print("\n" + "=" * 70)
    print("ITERATIVE TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Experiment:  {cfg.experiment_name}")
    print(f"  Iterations:  {cfg.num_iterations}")
    print(f"  Total effective epochs: {cfg.total_effective_epochs}")
    print(f"  Hub repo:    {cfg.current_adapter_repo}")
    print(f"  Local dir:   {cfg.local_outdir}")
    print(f"  Checkpoints: {[f'iteration_{it}' for it in iterations_saved]}")
    print(f"  Conv folders under: {cfg.conv_outdir}")
    print(f"    Expected: model_iter_0 .. model_iter_{cfg.num_iterations}")
    print("=" * 70)

    return TrainingSummary(
        iterations_run=iterations_run,
        iterations_saved=iterations_saved,
        final_conv_dir=final_conv_dir,
    )
