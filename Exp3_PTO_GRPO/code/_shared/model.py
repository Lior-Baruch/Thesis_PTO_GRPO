"""
model.py — Tokenizer/quant/LoRA + checkpoint discovery + iteration-resume helpers.

Covers everything trainers need to (a) load the base model with quantization
and LoRA adapters, (b) discover what's already been checkpointed on disk, and
(c) resume an interrupted multi-iteration training run.

Shared between GRPO_Exp3 and PTO_Exp3 trainers.
"""

import os
import types
from typing import List, Tuple, Dict, Optional, Callable


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            CONSTANTS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Directory naming conventions used across the experiment layout.
ITER_PREFIX = "iteration_"           # e.g. iteration_3/
HF_CKPT_PREFIX = "checkpoint-"       # e.g. checkpoint-500/
ADAPTER_SUBDIR = "adapter"           # iteration_N/adapter/
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
HF_TRAINER_FILES = ADAPTER_FILES + ("trainer_state.json",)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CHAT TEMPLATE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CHATML_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         TOKENIZER SETUP                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def setup_tokenizer(tokenizer_id: str, padding_side: str = "left"):
    """Load tokenizer with ChatML template, padding, and truncation configured.

    Args:
        tokenizer_id: HuggingFace model/tokenizer ID.
        padding_side: Which side to pad ("left" for generation).

    Returns:
        Configured AutoTokenizer instance.

    Notes:
        The configured template assumes messages use ChatML tags and that
        left-padding is preferred for causal generation in batched inference.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.chat_template = CHATML_TEMPLATE
    return tokenizer


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       QUANTIZATION CONFIG                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def build_quantization_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    compute_dtype=None,
):
    """Build a BitsAndBytesConfig with bf16 auto-detection.

    Args:
        load_in_4bit: Whether to load in 4-bit quantization.
        quant_type: Quantization type (default: "nf4").
        compute_dtype: Override compute dtype. If None, auto-detects
            bf16 support and falls back to fp16.

    Returns:
        Configured BitsAndBytesConfig.
    """
    import torch
    from transformers import BitsAndBytesConfig
    if compute_dtype is None:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       CHECKPOINT UTILITIES                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _list_numbered_dirs(
    root: str,
    prefix: str,
    separator: str = "_",
) -> List[Tuple[int, str]]:
    """List subdirs of ``root`` named ``{prefix}{N}`` sorted ascending by N.

    Returns ``[(n, full_path), ...]``. Skips entries whose suffix isn't a
    parseable int. Returns ``[]`` if ``root`` doesn't exist.
    """
    if not os.path.isdir(root):
        return []
    entries: List[Tuple[int, str]] = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not (name.startswith(prefix) and os.path.isdir(full)):
            continue
        try:
            n = int(name[len(prefix):].split(separator, 1)[0])
        except ValueError:
            continue
        entries.append((n, full))
    entries.sort(key=lambda x: x[0])
    return entries


def list_iteration_checkpoints(output_dir: str) -> List[Tuple[int, str]]:
    """List ``iteration_N/adapter`` directories sorted by iteration number.

    Returns:
        Sorted list of ``(iteration_number, adapter_path)`` tuples.
    """
    result: List[Tuple[int, str]] = []
    for n, iter_dir in _list_numbered_dirs(output_dir, ITER_PREFIX):
        adapter_path = os.path.join(iter_dir, ADAPTER_SUBDIR)
        if os.path.isdir(adapter_path):
            result.append((n, adapter_path))
    return result


def get_latest_iteration(output_dir: str) -> int:
    """Highest completed iteration number, or 0 (= base model, no adapter)."""
    checkpoints = list_iteration_checkpoints(output_dir)
    return checkpoints[-1][0] if checkpoints else 0


def validate_iteration_checkpoint(iteration_dir: str) -> bool:
    """True iff ``<iteration_dir>/adapter/`` contains the required adapter files."""
    adapter_path = os.path.join(iteration_dir, ADAPTER_SUBDIR)
    return all(os.path.exists(os.path.join(adapter_path, f)) for f in ADAPTER_FILES)


def list_hf_checkpoints(training_dir: str) -> List[str]:
    """List HuggingFace ``checkpoint-N`` directories sorted ascending by step.

    Args:
        training_dir: Directory containing ``checkpoint-N/`` subdirs (the
            ``GRPOTrainer`` output_dir, typically ``iteration_N/training/``).
    """
    return [path for _, path in _list_numbered_dirs(training_dir, HF_CKPT_PREFIX, separator="-")]


def get_latest_hf_checkpoint(training_dir: str) -> Optional[str]:
    """Highest-step ``checkpoint-N`` path in *training_dir*, or None."""
    checkpoints = list_hf_checkpoints(training_dir)
    return checkpoints[-1] if checkpoints else None


def get_latest_valid_hf_checkpoint(training_dir: str) -> Optional[str]:
    """Highest-step ``checkpoint-N`` that passes :func:`validate_hf_checkpoint`, or None.

    Walks the checkpoints newest→oldest and returns the first complete one. Unlike
    :func:`get_latest_hf_checkpoint` (which returns the highest-step dir even if it's a
    half-written crash artifact), this skips corrupt checkpoints — so a crash *during* a
    checkpoint write falls back to the previous good one instead of discarding the whole
    iteration. Matters once ``save_strategy="steps"`` makes checkpoint writes frequent
    (use ``save_total_limit >= 2`` so a fallback actually exists on disk).
    """
    for path in reversed(list_hf_checkpoints(training_dir)):
        if validate_hf_checkpoint(path)[0]:
            return path
    return None


def validate_hf_checkpoint(checkpoint_path: str) -> Tuple[bool, List[str]]:
    """Validate a HuggingFace checkpoint for ``resume_from_checkpoint``.

    Returns:
        ``(is_valid, missing_files)``.
    """
    missing = [f for f in HF_TRAINER_FILES if not os.path.exists(os.path.join(checkpoint_path, f))]
    return len(missing) == 0, missing


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         MODEL LOADING                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def load_base_model(
    model_id: str,
    quantization_config=None,
    for_training: bool = False,
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = "sdpa",
    dtype=None,
):
    """Load a base causal LM, optionally 4-bit quantized.

    Two paths:
      * ``quantization_config`` given → 4-bit (bnb) load + ``prepare_model_for_kbit_training``.
      * ``quantization_config is None`` → full-precision (bf16) load. On a GPU with
        ample VRAM (e.g. an A100 vs a 1B model) this skips the per-matmul dequant
        overhead, speeding up generation. LoRA then trains under bf16 autocast
        (set ``bf16=True`` in the trainer args).

    Args:
        model_id: HuggingFace model ID.
        quantization_config: BitsAndBytesConfig, or None for a bf16 load.
        for_training: If True, sets use_cache=False (and, when quantized, runs
            ``prepare_model_for_kbit_training``). If False, inference mode.
        trust_remote_code: Pass to from_pretrained.
        attn_implementation: Attention implementation (e.g., "sdpa"). Set None to omit.
        dtype: Override compute dtype for the bf16 path. If None, auto-detects
            bf16 support and falls back to fp16. Ignored when quantized (the
            bnb compute dtype is used instead).

    Returns:
        Loaded model on GPU.
    """
    import torch

    kwargs = dict(
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        if for_training:
            kwargs["dtype"] = quantization_config.bnb_4bit_compute_dtype
    else:
        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        kwargs["dtype"] = dtype

    from transformers import AutoModelForCausalLM
    _quant_tag = "4bit" if quantization_config is not None else "bf16"
    print(f"  Loading base model: {model_id} (for_training={for_training}, quant={_quant_tag})")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = not for_training

    if for_training and quantization_config is not None:
        from peft import prepare_model_for_kbit_training
        prepare_model_for_kbit_training(model)

    return model


def sync_pad_token(model, tokenizer):
    """Synchronize pad/eos/bos token IDs between model config and tokenizer."""
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id


def patch_generate(model, tokenizer):
    """Patch model.generate() to auto-inject tokenizer (required for stop_strings).

    No-op if already patched (checks for `_generate_patched` attribute).

    Why re-patching is needed: When PeftModel.from_pretrained() wraps a base
    model, the resulting PeftModel gets a fresh `generate` method from the
    PeftModel class. This new method does NOT have the `_generate_patched` flag,
    so our previous patch on the base model is effectively lost. Similarly,
    GRPOTrainer may replace the model's generate method. Call this function
    after any model wrapping operation.

    Args:
        model: The model whose generate() method to patch.
        tokenizer: Tokenizer to inject into generate() calls.
    """
    if hasattr(model, "_generate_patched"):
        return

    if not hasattr(model, "_original_generate"):
        model._original_generate = model.generate

    _tokenizer = tokenizer  # capture in closure

    def generate_with_tokenizer(self, *args, **kwargs):
        if "tokenizer" not in kwargs:
            kwargs["tokenizer"] = _tokenizer

        # Transformers warns when generation_config is provided together with
        # generation kwargs like disable_compile. Move disable_compile onto the
        # provided generation_config to keep behavior while avoiding the warning.
        if "generation_config" in kwargs and "disable_compile" in kwargs:
            gen_cfg = kwargs.get("generation_config")
            if gen_cfg is not None:
                setattr(gen_cfg, "disable_compile", kwargs.pop("disable_compile"))
        return self._original_generate(*args, **kwargs)

    model.generate = types.MethodType(generate_with_tokenizer, model)
    model._generate_patched = True
    print(f"  ✓ Patched generate() for {type(model).__name__}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   MULTI-ADAPTER MODEL BUILDING                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def build_multi_adapter_model_iterative(
    base_model,
    run_dir: str,
    iterations: List[int],
    post_load_fn: Optional[Callable] = None,
) -> Tuple:
    """Build a multi-adapter PEFT model from selected training iterations.

    Args:
        base_model: Base model (without adapters).
        run_dir: Root experiment directory with iteration_N/adapter/ subdirs.
        iterations: List of iteration numbers. ``0`` means base model only;
            ``1+`` correspond to ``iteration_N/adapter`` checkpoints.
        post_load_fn: Optional callable(model) invoked after all adapters are
            loaded (e.g., to call patch_generate).

    Returns:
        (model, labels_by_iter): model is the PeftModel (or base model when no
            adapters requested), labels_by_iter maps iteration number to the
            adapter name registered on the model.
    """
    labels: Dict[int, str] = {}
    non_base = [i for i in iterations if i > 0]

    for it in iterations:
        labels[it] = "base" if it == 0 else f"iter_{it}"

    if not non_base:
        base_model.eval()
        return base_model, labels

    # Resolve adapter paths using list_iteration_checkpoints
    available = list_iteration_checkpoints(run_dir)
    iter_to_path: Dict[int, str] = {}
    for it in non_base:
        match = [path for num, path in available if num == it]
        if not match:
            raise FileNotFoundError(
                f"Iteration {it} not found! Available: {[n for n, _ in available]}"
            )
        iter_to_path[it] = match[0]

    from peft import PeftModel
    # First adapter: create PeftModel
    first_it = non_base[0]
    first_label = labels[first_it]
    print(f"  Creating PeftModel from iteration {first_it} (adapter: '{first_label}')")
    multi_model = PeftModel.from_pretrained(
        base_model, iter_to_path[first_it],
        adapter_name=first_label, is_trainable=False,
    )

    # Load remaining adapters
    for it in non_base[1:]:
        label = labels[it]
        print(f"  Loading adapter for iteration {it} (adapter: '{label}')")
        multi_model.load_adapter(iter_to_path[it], adapter_name=label, is_trainable=False)

    if post_load_fn is not None:
        post_load_fn(multi_model)

    multi_model.eval()
    print(f"  ✓ Adapters loaded: {list(multi_model.peft_config.keys())}")
    return multi_model, labels


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      PARAMETER COUNTING                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def get_adapter_param_count(model) -> dict:
    """Get trainable vs total parameter counts for a model.

    Returns:
        Dict with keys: trainable, total, trainable_pct, trainable_M, total_M.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
        "trainable_M": trainable / 1e6,
        "total_M": total / 1e6,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  PATIENT PERMUTATION SETUP                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def setup_permutations(only_expert_therapist: bool = True) -> Tuple:
    """Generate patient permutations and initialize therapist persona prompts.

    Args:
        only_expert_therapist: If True, only use expert-level therapist permutations.

    Returns:
        ``(permutations, therapist_system_prompt, therapist_init_utterance)``

    Notes:
        Uses a randomly chosen therapist name at ``Good`` personality level to
        produce a stable-quality therapist prompt/initial utterance pair.
    """
    from system_prompts_builder import generate_all_permutations, CounselorPersonality

    permutations = generate_all_permutations(only_expert_therapist=only_expert_therapist)

    good_level = CounselorPersonality.PersonalityLevel.Good
    therapist = CounselorPersonality.choose_random_therapist_name()
    therapist_system_prompt = CounselorPersonality.build_system_prompt(
        personality_level=good_level, name=therapist["name"]
    )
    therapist_init_utterance = CounselorPersonality.get_init_utterance(
        personality_level=good_level, name=therapist["name"]
    )

    return permutations, therapist_system_prompt, therapist_init_utterance


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  ITERATION RESUME (iteration_N/ discovery)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def resolve_start_state(local_outdir: str, base_policy, tokenizer):
    """Determine starting iteration, load appropriate policy, detect HF resume checkpoint.

    Iteration convention: ``0`` = base model (no adapter), ``1+`` = trained.

    Resume logic:

    A. **Fresh start** — no checkpoints anywhere → start at iteration 1.
    B. **Mid-iteration crash** — ``iteration_N/training/checkpoint-*/`` exists
       but ``iteration_N/adapter/`` does not (training didn't finish).
    C. **Between iterations** — ``iteration_N/adapter/`` exists, start at N+1.

    Shared between GRPO_Exp3 and PTO_Exp3 — same on-disk layout in both methods.

    Returns:
        ``(start_iteration, policy, resume_checkpoint)``.
    """
    from peft import PeftModel

    latest_iteration = get_latest_iteration(local_outdir)

    # Always attach adapters to a plain base model (avoid nested PEFT wrappers).
    base_for_adapter = (
        base_policy.get_base_model()
        if hasattr(base_policy, "peft_config") and hasattr(base_policy, "get_base_model")
        else base_policy
    )

    candidate_training_dir = os.path.join(
        local_outdir, f"{ITER_PREFIX}{latest_iteration + 1}", "training"
    )
    all_ckpts = list_hf_checkpoints(candidate_training_dir)

    # ── Case B: incomplete iteration ──
    if all_ckpts:
        candidate_iter = latest_iteration + 1
        # Walk back to the newest *complete* checkpoint: a crash mid-write can leave the
        # highest-step dir half-written, and with save_strategy="steps" that's frequent
        # enough to matter. With save_total_limit >= 2 a good fallback exists on disk.
        valid_ckpt = get_latest_valid_hf_checkpoint(candidate_training_dir)
        if valid_ckpt is not None:
            newest = os.path.basename(all_ckpts[-1])
            if os.path.basename(valid_ckpt) != newest:
                print(f"  ⚠ Newest checkpoint {newest} invalid; falling back to "
                      f"{os.path.basename(valid_ckpt)}")
            print(f"  Resuming iteration_{candidate_iter} from {os.path.basename(valid_ckpt)}")
            policy = PeftModel.from_pretrained(base_for_adapter, valid_ckpt, is_trainable=True)
            patch_generate(policy, tokenizer)
            return candidate_iter, policy, valid_ckpt
        print(f"  ⚠ No valid checkpoint in iteration_{candidate_iter}/training "
              f"({len(all_ckpts)} found, all incomplete)")
        print(f"    Starting iteration_{candidate_iter} from scratch")
        if latest_iteration == 0:
            policy = base_for_adapter
        else:
            adapter_path = os.path.join(
                local_outdir, f"{ITER_PREFIX}{latest_iteration}", ADAPTER_SUBDIR
            )
            policy = PeftModel.from_pretrained(base_for_adapter, adapter_path, is_trainable=True)
        patch_generate(policy, tokenizer)
        return candidate_iter, policy, None

    # ── Case C: normal resume ──
    if latest_iteration > 0:
        adapter_path = os.path.join(
            local_outdir, f"{ITER_PREFIX}{latest_iteration}", ADAPTER_SUBDIR
        )
        print(f"  Resuming: loading adapter from iteration_{latest_iteration}")
        policy = PeftModel.from_pretrained(base_for_adapter, adapter_path, is_trainable=True)
        patch_generate(policy, tokenizer)
        return latest_iteration + 1, policy, None

    # ── Case A: fresh start ──
    return 1, base_for_adapter, None


def compute_cumulative_step_offset(local_outdir: str) -> int:
    """Sum global_steps from completed and in-progress iterations.

    Each iteration's trainer (``GRPOTrainer`` or ``DPOTrainer``) resets
    ``global_step`` to 0; this tracks the total so the W&B x-axis is continuous
    across iterations. Also counts steps from in-progress iterations
    (mid-crash recovery) to prevent overlap.
    """
    offset = 0
    completed = list_iteration_checkpoints(local_outdir)
    for it_num, _ in completed:
        training_dir = os.path.join(local_outdir, f"{ITER_PREFIX}{it_num}", "training")
        ckpts = list_hf_checkpoints(training_dir)
        if ckpts:
            offset += int(os.path.basename(ckpts[-1]).split("-")[-1])

    # In-progress iteration (training started but adapter/ not saved yet)
    latest_completed = completed[-1][0] if completed else 0
    in_progress_training_dir = os.path.join(
        local_outdir, f"{ITER_PREFIX}{latest_completed + 1}", "training"
    )
    # Walk-back to match the checkpoint resolve_start_state actually resumes from
    # (a half-written newest checkpoint is skipped there, so skip it here too).
    in_progress_ckpt = get_latest_valid_hf_checkpoint(in_progress_training_dir)
    if in_progress_ckpt is not None:
        partial_steps = int(os.path.basename(in_progress_ckpt).split("-")[-1])
        offset += partial_steps
        print(f"  Including {partial_steps} steps from in-progress iteration_{latest_completed + 1}")

    if offset > 0:
        print(f"  Cumulative step offset: {offset}")
    return offset
