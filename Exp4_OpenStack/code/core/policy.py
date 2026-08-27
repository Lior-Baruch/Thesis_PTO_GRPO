"""policy.py -- the local therapist policy: template, weights, decoding, resume.

Every other Exp4 module talks to a *remote* model over an OpenAI-compatible socket
(patient, oracle, judge). This module owns the one model that lives in this process:
the Llama-3.2-1B therapist policy that is actually being optimized. It covers

- the hand-written ChatML template (the base checkpoint ships none),
- base-weight load (bf16) and LoRA attachment,
- ``patch_generate`` -- the bind that makes ``stop_strings`` work at all,
- the anti-degeneracy stack (stop strings + ``clean_completion``),
- batched therapist decoding that reports OOM instead of raising, and
- checkpoint discovery + multi-iteration resume.

Two of those exist for reasons that are easy to mistake for boilerplate:

**The therapist is a BASE model, not an instruct model.** It has no chat template
and ``<|im_start|>`` / ``<|im_end|>`` are ordinary BPE pieces, NOT special tokens.
Early in training the policy therefore self-plays: it emits ``<|im_start|>`` and
writes the *patient's* next turn as literal text. Unchecked, that leaked text lands
in the saved conversation, in the transcript the oracle grades, and in the DPO pair
-- and since a 200-token ramble scores like anything else, the run trains toward
rambling. The stack that prevents it is ``STOP_STRINGS`` at every decode site +
``clean_completion`` on every decoded string + ``patch_generate`` (without which the
stop strings are silently ignored) + the caller flooring empty completions to
``REWARD_FLOOR``. All four are load-bearing; removing any one restores the failure.

**Resume is not optional.** An arm is 6-8 iterations of multi-hour work on a
preemptible Colab GPU. ``resolve_start_state`` is what turns a killed session into a
few lost steps instead of a lost arm, and ``get_latest_valid_hf_checkpoint`` is what
stops a checkpoint that was half-written when the process died from taking the whole
iteration down with it.

torch/transformers are imported at module scope here on purpose: this module is
inherently GPU-side and the read-only EDA never imports it.
"""

from __future__ import annotations

import gc
import os
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    # Constants
    "CHATML_TEMPLATE",
    "CHATML_MARKERS",
    "LLAMA3_END_MARKERS",
    "CHAT_TEMPLATE_DATE",
    "STOP_STRINGS",
    "ITER_PREFIX",
    "HF_CKPT_PREFIX",
    "ADAPTER_SUBDIR",
    "ADAPTER_FILES",
    "HF_TRAINER_FILES",
    # Model / tokenizer setup
    "setup_tokenizer",
    "setup_base_model",
    "attach_lora",
    "sync_pad_token",
    "patch_generate",
    "therapist_stop_token_ids",
    "get_adapter_param_count",
    # Generation
    "clean_completion",
    "generate_therapist_batch",
    "vram_report",
    # Checkpoint discovery
    "list_iteration_checkpoints",
    "get_latest_iteration",
    "validate_iteration_checkpoint",
    "list_hf_checkpoints",
    "get_latest_hf_checkpoint",
    "validate_hf_checkpoint",
    "get_latest_valid_hf_checkpoint",
    # Resume
    "resolve_start_state",
    "compute_cumulative_step_offset",
]


# =============================================================================
# CONSTANTS
# =============================================================================

# Directory grammar of a run dir. Mirrors the layout documented in CLAUDE.md:
#   runs/<EXP_NAME>/iteration_<N>/{adapter,training/checkpoint-<STEP>}
# The presence of iteration_<N>/adapter/ is the ONLY definition of "iteration done".
ITER_PREFIX = "iteration_"
HF_CKPT_PREFIX = "checkpoint-"
ADAPTER_SUBDIR = "adapter"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")

# What `resume_from_checkpoint` actually needs. Deliberately EXACTLY three files:
# the project's own sidecars (iteration_metadata.json, generations.jsonl, ...) are
# not required, so a checkpoint missing only those still resumes.
HF_TRAINER_FILES = ADAPTER_FILES + ("trainer_state.json",)

# The base Llama-3.2-1B checkpoint has no chat template at all, so one is written by
# hand and installed on the tokenizer. Both trainers, the look-ahead simulator and the
# eval-generation tool must render prompts with this exact template or the policy sees
# a different prompt distribution at train time and at eval time.
CHATML_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# ChatML control markers. `<|im_end|>` closes a turn; `<|im_start|>` opens one.
# Generation stops at the FIRST of either and `clean_completion` cuts at the FIRST of
# either, so the two use one definition and can never disagree about where a turn ends.
CHATML_MARKERS = ("<|im_end|>", "<|im_start|>")

# Default stop strings for every therapist decode site ON THE BASE THERAPIST. `<|im_start|>`
# is in the list so a self-play attempt halts the instant the model opens a fake turn -- that
# is what prevents both `<|im_start|>`-spam degenerate turns and the role-swap derailment where
# a leaked first-person "user" line flips the patient simulator into counselor mode for the
# rest of the conversation.
#
# The INSTRUCT therapist needs none of this: its native Llama-3 template ends every assistant
# turn with `<|eot_id|>`, a real single special token the model is trained to emit, so stopping
# is token-id-exact and free. Instruct arms therefore run with EMPTY stop strings (the config
# builder resolves STOP_STRINGS="auto" to () for them) and rely on the eos-id list from
# :func:`therapist_stop_token_ids` instead.
STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]

# Llama-3 turn-terminating special tokens. `<|eot_id|>` closes a normal turn (and IS the
# Instruct tokenizer's eos), `<|eom_id|>` closes a tool-call turn, and `<|start_header_id|>`
# opens the NEXT turn's role header -- stopping on it halts a self-play attempt at the first
# token, the exact analogue of `<|im_start|>` in the ChatML list above. All three exist in the
# BASE tokenizer's vocab too (harmless there: the base model was never trained to emit them),
# so :func:`therapist_stop_token_ids` needs no model-variant branch.
LLAMA3_END_MARKERS = ("<|eot_id|>", "<|eom_id|>", "<|start_header_id|>")

# Pinned `date_string` for every chat-template render. The Llama-3.2 Instruct template
# interpolates "Today Date: <date_string>" into its system header and defaults the variable to
# strftime_now(...) -- i.e. the REAL current date, so an arm resumed on a different day would
# train and generate under a slightly different prompt than its first half, and no render would
# be reproducible after the fact. Pinning it removes the nondeterminism; the value is the
# template's own documented fallback date. Templates that never read the variable (the ChatML
# one above) simply ignore it -- an unused Jinja variable is not an error.
CHAT_TEMPLATE_DATE = "26 Jul 2024"

# torch >= 2.5 exposes torch.OutOfMemoryError; the alias is kept for older wheels.
_OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

_GIB = 1024 ** 3


# =============================================================================
# TOKENIZER / MODEL SETUP
# =============================================================================


def setup_tokenizer(tokenizer_id: str, padding_side: str = "left"):
    """Load the therapist tokenizer with template and padding configured.

    Args:
        tokenizer_id: HuggingFace tokenizer/model id (the therapist model).
        padding_side: "left" for causal generation (a right-padded batch would decode
            pad tokens as the start of the completion).

    Returns:
        A configured ``AutoTokenizer``.

    Notes:
        Four settings, all load-bearing:

        - ``pad_token = eos_token``: neither therapist checkpoint defines a pad token, and
          batched generation needs one. The pad id is also what
          :func:`generate_therapist_batch` passes as ``pad_token_id``. (On the Instruct
          tokenizer eos is ``<|eot_id|>``, so that is also the pad -- standard for Llama-3
          Instruct.)
        - ``padding_side = left``: see above.
        - ``truncation_side = left``: an over-long conversation must lose its OLDEST
          turns, never the most recent patient utterance the therapist is replying to.
        - ``chat_template``: the checkpoint's NATIVE template is kept when it ships one
          (the Instruct therapist -- the official Llama-3 format it was trained on);
          :data:`CHATML_TEMPLATE` is installed only when the checkpoint ships none (the
          base therapist, which has no template at all -- ``apply_chat_template`` raises
          without one). The therapist variant is encoded in the arm name (``_Th<tag>``),
          so which template a run used is recoverable from the folder.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHATML_TEMPLATE
        print(f"  tokenizer {tokenizer_id}: no chat template shipped -- installed hand-written ChatML")
    else:
        print(f"  tokenizer {tokenizer_id}: keeping the checkpoint's native chat template")
    return tokenizer


def setup_base_model(base_model_id: str, *, use_4bit: bool = False):
    """Load the therapist base weights (bf16 by default) ready for LoRA training.

    Args:
        base_model_id: HuggingFace model id, e.g. ``meta-llama/Llama-3.2-1B``.
        use_4bit: Load NF4 4-bit via bitsandbytes instead of bf16. **Exp4 never runs
            this.** The toggle exists only so the path stays reachable for a
            memory-starved smoke test.

    Returns:
        The loaded model, on GPU, with ``use_cache=False``.

    Notes:
        **Why 4-bit is off.** Exp2 generated its conversations in 4-bit NF4 and Exp3
        in bf16 on the *same* base model, and 4-bit induced roughly 30x more
        phrase-loop degeneration (~9.5% vs ~0.3% of therapist turns running to the
        token cap as repeated spam). The oracle floors that, which moved the whole
        score axis. A 1B model in bf16 is ~2.5 GB, so there is nothing to buy here and
        a comparability hazard to lose.

        **``use_cache`` is left False** (the training setting; it is also what
        gradient checkpointing requires). Generation without a KV cache is very slow,
        so the trainer flips ``policy.config.use_cache = True`` around every
        generation phase and back to False before ``trainer.train()``. The look-ahead
        simulator does the same toggle itself and restores the previous value in a
        ``finally``, because it runs mid-optimizer-step.

        The pad/eos/bos ids are NOT synced here (no tokenizer in scope) -- call
        :func:`sync_pad_token` right after.
    """
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "dtype": dtype,
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )

    quant_tag = "4bit-nf4" if use_4bit else "bf16"
    print(f"  Loading base policy: {base_model_id} (precision={quant_tag})")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
    model.config.use_cache = False

    if use_4bit:
        from peft import prepare_model_for_kbit_training

        prepare_model_for_kbit_training(model)

    return model


def attach_lora(
    model,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
):
    """Wrap *model* in a fresh LoRA adapter and return the ``PeftModel``.

    Args:
        model: A base causal LM from :func:`setup_base_model` (NOT an already-wrapped
            ``PeftModel`` -- nesting PEFT wrappers breaks adapter save/load).
        r: LoRA rank.
        alpha: LoRA alpha (Exp3 used ``r == alpha == 16``).
        dropout: LoRA dropout.
        target_modules: Module name suffixes to adapt, e.g.
            ``["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]``.

    Returns:
        A ``PeftModel`` with one trainable adapter named ``default``.

    Notes:
        **Only for the path that builds the adapter explicitly.** When a TRL trainer is
        handed a ``peft_config`` it attaches LoRA itself; calling this first would give
        the trainer an already-wrapped model to wrap again. Pick one.

        **Call :func:`patch_generate` on the result.** Wrapping installs a new
        ``generate`` on the wrapper.

        **Colab install gotcha:** peft raises inside ``get_peft_model``'s LoRA
        dispatcher (``dispatch_torchao``) against Colab's pre-baked torchao < 0.16.0.
        Uninstall torchao in the install cell; the failure looks like an unrelated
        ImportError deep in peft.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    peft_model = get_peft_model(model, lora_config)
    counts = get_adapter_param_count(peft_model)
    print(
        f"  LoRA attached: r={r}, alpha={alpha}, dropout={dropout} "
        f"({counts['trainable_M']:.2f}M trainable / {counts['total_M']:.1f}M total)"
    )
    return peft_model


def therapist_stop_token_ids(tokenizer) -> List[int]:
    """Token ids that terminate a therapist generation: eos plus the Llama-3 end markers.

    Returns:
        ``[tokenizer.eos_token_id, ...]`` deduplicated, order-stable. On the Instruct
        tokenizer eos IS ``<|eot_id|>``, so the list is ``[eot, eom, start_header]`` -- the
        turn terminator plus the two ways a self-play attempt can open a fake turn. On the
        base tokenizer it is ``[end_of_text, eot, eom, start_header]``; the extra ids are
        inert there (the base model was never trained to emit them) and stopping would be
        the right response if it ever did.

    Notes:
        Token-id stopping is EXACT for the Llama-3 markers because they are single special
        tokens. It is exactly wrong for the ChatML markers (6-token ordinary BPE sequences --
        see :func:`generate_therapist_batch`'s stop-string notes), which is why base arms keep
        string stopping ON TOP of this list, while Instruct arms need only this list.
    """
    ids: List[int] = []
    for candidate in (tokenizer.eos_token_id,
                      *(tokenizer.convert_tokens_to_ids(t) for t in LLAMA3_END_MARKERS)):
        if candidate is None or candidate < 0:
            continue
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is not None and candidate == unk:
            continue
        if candidate not in ids:
            ids.append(int(candidate))
    return ids


def sync_pad_token(model, tokenizer) -> None:
    """Copy pad/eos/bos ids from *tokenizer* onto the model config and generation config.

    Without this the model config keeps the checkpoint's own (often ``None``) pad id
    while the tokenizer pads with eos, and batched generation either warns on every
    call or masks the wrong positions.

    The generation config's ``eos_token_id`` is set to the full
    :func:`therapist_stop_token_ids` LIST (the model config keeps the scalar): any default
    ``generate`` call then stops at a turn terminator or a fake-turn opener without needing
    string criteria. Explicit per-call ``eos_token_id`` arguments (both decode paths pass one)
    override it either way; this is the belt-and-braces layer for calls that pass none.
    """
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
        gen_cfg.eos_token_id = therapist_stop_token_ids(tokenizer)
        gen_cfg.bos_token_id = tokenizer.bos_token_id


def patch_generate(model, tokenizer) -> None:
    """Bind *tokenizer* into ``model.generate`` so ``stop_strings`` works. Idempotent.

    HuggingFace's ``stop_strings`` argument is silently inert unless a ``tokenizer=``
    is passed to the same ``generate`` call -- it needs the vocab to build its matching
    table. TRL's internal generation calls do not pass one, so without this patch every
    stop string in the GRPO rollout path is ignored and the anti-degeneracy stack loses
    its first line of defence. The failure is silent: generation simply runs to
    ``max_new_tokens`` and the leaked ChatML shows up downstream.

    Args:
        model: Any model whose ``generate`` should carry the tokenizer.
        tokenizer: Captured in the closure and injected when the caller passes none.

    Notes:
        **Re-apply after EVERY re-wrap.** ``PeftModel.from_pretrained``,
        ``get_peft_model`` and both TRL trainers hand back an object with a fresh
        ``generate``. The three mandatory call sites are: right after the base load,
        inside :func:`resolve_start_state` (done for you), and on BOTH sides of
        ``trainer.train()`` -- before, because the trainer just built its wrapper, and
        after, because ``train()`` may have rebuilt it again.

        **The idempotence guard sees through PEFT's attribute forwarding.** A
        ``PeftModel`` forwards unknown attribute lookups to the wrapped model, so
        ``hasattr(wrapper, "_generate_patched")`` is True as soon as the *inner* model
        was patched, and this function returns early on the wrapper. That is deliberate
        and is why Exp3 worked: ``PeftModel.generate`` delegates down to the inner
        model's instance-level patched ``generate``, so the injection still happens.
        Do not "fix" the guard by inspecting ``__dict__`` instead -- patching the
        wrapper would capture the *inner* ``_original_generate`` through the same
        forwarding and thereby bypass ``PeftModel.generate`` entirely.

        The ``disable_compile`` shuffle exists because transformers warns when a
        ``generation_config`` is passed alongside loose generation kwargs; moving the
        flag onto the config keeps the behaviour and drops the per-call warning.
    """
    if hasattr(model, "_generate_patched"):
        return

    if not hasattr(model, "_original_generate"):
        model._original_generate = model.generate

    _tokenizer = tokenizer  # captured in the closure

    def generate_with_tokenizer(self, *args, **kwargs):
        if "tokenizer" not in kwargs:
            kwargs["tokenizer"] = _tokenizer

        if "generation_config" in kwargs and "disable_compile" in kwargs:
            gen_cfg = kwargs.get("generation_config")
            if gen_cfg is not None:
                setattr(gen_cfg, "disable_compile", kwargs.pop("disable_compile"))
        return self._original_generate(*args, **kwargs)

    model.generate = types.MethodType(generate_with_tokenizer, model)
    model._generate_patched = True
    print(f"  OK patched generate() for {type(model).__name__}")


def get_adapter_param_count(model) -> Dict[str, float]:
    """Trainable vs total parameter counts.

    Returns:
        ``{"trainable", "total", "trainable_pct", "trainable_M", "total_M"}``. Worth
        stamping into ``run_metadata.json``: a LoRA config that silently matched no
        module names shows up here as 0 trainable and nowhere else until the loss
        refuses to move.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": float(trainable),
        "total": float(total),
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
        "trainable_M": trainable / 1e6,
        "total_M": total / 1e6,
    }


# =============================================================================
# GENERATION
# =============================================================================


def clean_completion(text: Optional[str]) -> str:
    """Cut a raw therapist completion at the first ChatML marker and strip it.

    Args:
        text: A decoded completion, possibly ``None``.

    Returns:
        Everything before the earliest ``<|im_start|>`` / ``<|im_end|>``, stripped.
        **An empty string means the turn was degenerate** -- the model produced only a
        marker, only whitespace, or nothing usable.

    Notes:
        With ``<|im_start|>`` in the stop strings this is usually a no-op, but it
        salvages any completion that overran (a stop string only fires on a decoded
        boundary, and the look-ahead path decodes differently). Because the stop
        strings and this cut share :data:`CHATML_MARKERS`, the saved conversation, the
        transcript sent to the oracle and any DPO pair are all free of leaked role
        headers by the same rule.

        Callers must handle the empty return explicitly: the conversation loop ends the
        conversation, and the reward function floors the candidate to ``REWARD_FLOOR``.
        Passing "" on to the oracle would ask it to grade a turn that does not exist.
    """
    if not text:
        return ""
    cut = len(text)
    for marker in CHATML_MARKERS:
        i = text.find(marker)
        if i != -1 and i < cut:
            cut = i
    return text[:cut].strip()


def generate_therapist_batch(
    model,
    tokenizer,
    batch_messages: List[List[Dict[str, str]]],
    *,
    max_tokens: int,
    temperature: float,
    max_input_tokens: int = 2048,
    stop_strings: Optional[Sequence[str]] = None,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Generate one therapist reply per conversation in a single padded batch.

    Args:
        model: The therapist policy (base or PEFT-wrapped), already patched by
            :func:`patch_generate`.
        tokenizer: From :func:`setup_tokenizer` (left padding, left truncation).
        batch_messages: One chat-message list per conversation, in the therapist's
            role convention (``system``/``user``/``assistant``).
        max_tokens: ``max_new_tokens`` per completion.
        temperature: Sampling temperature; sampling is always on (``do_sample=True``).
        max_input_tokens: Prompt truncation budget. Truncation is LEFT, so the oldest
            turns are dropped and the patient's latest utterance always survives.
        stop_strings: ``None`` means the :data:`STOP_STRINGS` default (base-therapist
            ChatML markers). An EMPTY sequence means "no string stopping" and is the
            correct value for Instruct arms -- generation then stops on the
            :func:`therapist_stop_token_ids` eos list alone, with no per-call
            ``StopStringCriteria`` table build.

    Returns:
        ``(responses, None)`` on success, where ``responses[i]`` corresponds to
        ``batch_messages[i]`` and has already been through :func:`clean_completion`
        (so ``""`` marks a degenerate turn). On failure:
        ``(None, "oom")`` or ``(None, "runtime_error")``.

    Notes:
        **This never raises on OOM.** It returns ``(None, "oom")`` after
        ``gc.collect()`` + ``torch.cuda.empty_cache()`` so the caller can halve its
        batch and retry rather than lose the whole phase. RuntimeErrors whose text
        smells of CUDA memory are classified as OOM too, because not every allocator
        failure surfaces as ``torch.OutOfMemoryError``.

        **This cannot protect you on the local 12 GB card.** An over-budget VRAM
        *request* there is a driver fault that reboots the machine with no Python
        exception at all, so batch size is a safety setting, not a throughput knob.
        Do the arithmetic before raising it (see ``tools/smoke.py``).

        **The padded input length is sliced off, not the per-row prompt length.** With
        left padding every row starts its completion at the same column, which is the
        whole reason padding_side must stay "left".

        **Stop-string cost is per CALL, not per step** (measured in Exp3): HF's
        ``StopStringCriteria`` rebuilds a 128k-vocab table on every ``generate()``
        because ``get_vocab()`` iteration order is unstable, so its internal cache key
        never hits -- 0.8-1.7 s per build, worth ~3-6% of an iteration. A memoised
        criteria object would be bit-identical, but the object has to be threaded
        through ``generate`` (and through TRL's own generate calls) to help, so it is
        recorded here rather than done. Do NOT "fix" it by switching to token-id or
        ``eos_token_id`` stopping: the markers are 6-token BPE sequences
        (``<|im_end|>`` -> ``['<','|','im','_end','|','>']``) and are not special
        tokens, so token-id stopping matches a strict SUBSET of what string stopping
        matches. That is a science change, and a K-asymmetric one.
    """
    if not batch_messages:
        return [], None

    effective_stops = list(STOP_STRINGS if stop_strings is None else stop_strings)

    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      date_string=CHAT_TEMPLATE_DATE)
        for messages in batch_messages
    ]

    # String criteria only when there are strings: an empty stop_strings=[] still pays the
    # per-call 128k-vocab criteria table build, so Instruct arms (empty stops) omit the kwargs
    # entirely and stop on the eos-id list alone.
    stop_kwargs = (
        {"stop_strings": effective_stops, "tokenizer": tokenizer} if effective_stops else {}
    )

    encoded = None
    outputs = None
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,  # the chat template already supplies the framing
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                do_sample=True,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=therapist_stop_token_ids(tokenizer),
                temperature=temperature,
                num_return_sequences=1,
                **stop_kwargs,
            )
    except _OOM_ERROR as exc:
        print(f"  CUDA OOM during therapist generation: {exc}")
        gc.collect()
        torch.cuda.empty_cache()
        return None, "oom"
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            print(f"  Runtime CUDA memory failure during therapist generation: {exc}")
            gc.collect()
            torch.cuda.empty_cache()
            return None, "oom"
        print(f"  Runtime error during therapist generation: {exc}")
        return None, "runtime_error"

    padded_input_length = encoded.input_ids.shape[1]

    responses: List[str] = []
    for i in range(len(batch_messages)):
        new_tokens = outputs[i][padded_input_length:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(clean_completion(decoded))

    del encoded, outputs

    return responses, None


def vram_report() -> Dict[str, float]:
    """Current CUDA memory, in GiB. Safe to call with no CUDA present.

    Returns:
        ``{"reserved_gib", "allocated_gib", "total_gib"}``; all zeros when CUDA is
        unavailable or the query fails.

    Notes:
        ``reserved_gib`` (the caching allocator's high-water footprint) is the number
        worth printing per batch as ``f"vram {r:.1f}G"``. Flat across batches means the
        inter-batch ``gc.collect()`` + ``empty_cache()`` is doing its job; climbing
        across batches means it has regressed -- consecutive batches have different max
        sequence lengths, so freed blocks are the wrong size to reuse and the allocator
        keeps asking the driver for more. On the 12 GB local card that ends in a reboot,
        not an exception, and a single-batch smoke test cannot detect it (needs >= 2).
    """
    zero = {"reserved_gib": 0.0, "allocated_gib": 0.0, "total_gib": 0.0}
    try:
        if not torch.cuda.is_available():
            return zero
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "reserved_gib": torch.cuda.memory_reserved() / _GIB,
            "allocated_gib": torch.cuda.memory_allocated() / _GIB,
            "total_gib": props.total_memory / _GIB,
        }
    except Exception:
        return zero


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================


def _list_numbered_dirs(
    root: str,
    prefix: str,
    separator: str = "_",
) -> List[Tuple[int, str]]:
    """List subdirs of *root* named ``{prefix}{N}``, sorted ascending by N.

    Returns ``[(n, full_path), ...]``; entries whose suffix is not a parseable int are
    skipped, and a missing *root* yields ``[]`` (a run dir that does not exist yet is a
    normal state, not an error).
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


def list_iteration_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    """List completed iterations as ``[(iteration_number, adapter_path), ...]``, ascending.

    An iteration counts as completed exactly when ``iteration_<N>/adapter/`` holds both
    files in :data:`ADAPTER_FILES` -- directory presence alone is NOT enough. A process
    killed inside the end-of-iteration ``save_pretrained`` (a multi-second window on the
    Drive mount) leaves ``adapter_config.json`` without ``adapter_model.safetensors``;
    counting that torn save as "done" would send every subsequent resume into
    ``PeftModel.from_pretrained`` on it, crashing the notebook until a human deletes the
    directory by hand. Treating it as incomplete instead routes ``resolve_start_state``
    to case B, which resumes from the latest valid HF checkpoint sitting right beside it.
    This is the single definition of "done" used by the resume logic, the EDA and the
    eval-generation tool.
    """
    result: List[Tuple[int, str]] = []
    for n, iter_dir in _list_numbered_dirs(run_dir, ITER_PREFIX):
        adapter_path = os.path.join(iter_dir, ADAPTER_SUBDIR)
        if not os.path.isdir(adapter_path):
            continue
        if not validate_iteration_checkpoint(iter_dir):
            print(
                f"  WARNING: {os.path.basename(iter_dir)}/adapter/ exists but is missing "
                f"adapter files (torn save?) -- treating the iteration as INCOMPLETE"
            )
            continue
        result.append((n, adapter_path))
    return result


def get_latest_iteration(run_dir: str) -> int:
    """Highest completed iteration number, or ``0`` for "base model, no adapter yet"."""
    checkpoints = list_iteration_checkpoints(run_dir)
    return checkpoints[-1][0] if checkpoints else 0


def validate_iteration_checkpoint(iteration_dir: str) -> bool:
    """True iff ``<iteration_dir>/adapter/`` holds both files in :data:`ADAPTER_FILES`."""
    adapter_path = os.path.join(iteration_dir, ADAPTER_SUBDIR)
    return all(os.path.exists(os.path.join(adapter_path, f)) for f in ADAPTER_FILES)


def list_hf_checkpoints(training_dir: str) -> List[str]:
    """List ``checkpoint-<STEP>`` dirs under *training_dir*, sorted ascending by step.

    Args:
        training_dir: An HF Trainer ``output_dir`` -- ``iteration_<N>/training/``.
    """
    return [path for _, path in _list_numbered_dirs(training_dir, HF_CKPT_PREFIX, separator="-")]


def get_latest_hf_checkpoint(training_dir: str) -> Optional[str]:
    """Highest-step ``checkpoint-<STEP>`` path, or ``None``. Does NOT validate it."""
    checkpoints = list_hf_checkpoints(training_dir)
    return checkpoints[-1] if checkpoints else None


def validate_hf_checkpoint(checkpoint_path: str) -> Tuple[bool, List[str]]:
    """Check a checkpoint against :data:`HF_TRAINER_FILES`.

    Returns:
        ``(is_valid, missing_files)``.
    """
    missing = [f for f in HF_TRAINER_FILES if not os.path.exists(os.path.join(checkpoint_path, f))]
    return len(missing) == 0, missing


def get_latest_valid_hf_checkpoint(training_dir: str) -> Optional[str]:
    """Highest-step checkpoint that passes :func:`validate_hf_checkpoint`, else ``None``.

    Walks newest -> oldest and returns the first complete one. Unlike
    :func:`get_latest_hf_checkpoint`, this skips a half-written crash artifact -- a
    process killed *during* a checkpoint write leaves the highest-step dir incomplete,
    and pointing ``resume_from_checkpoint`` at it throws away the entire iteration
    rather than the last few steps. Keep ``save_total_limit >= 2`` so a fallback
    actually exists on disk.
    """
    for path in reversed(list_hf_checkpoints(training_dir)):
        if validate_hf_checkpoint(path)[0]:
            return path
    return None


# =============================================================================
# ITERATION RESUME
# =============================================================================


def resolve_start_state(
    run_dir: str,
    base_policy,
    tokenizer,
) -> Tuple[int, Any, Optional[str]]:
    """Decide which iteration to start at and hand back the policy to start it with.

    Iteration convention: ``0`` = base model (no adapter), ``1+`` = trained.

    Three cases:

    A. **Fresh start** -- nothing on disk. Returns ``(1, base, None)``.
    B. **Mid-iteration crash** -- ``iteration_<N>/training/checkpoint-*/`` exists but
       ``iteration_<N>/adapter/`` does not (or is a torn save). Returns the
       **iteration-start** policy (previous iteration's adapter, or the bare base for
       an iteration-1 resume) together with the latest VALID HF checkpoint path: the
       TRL trainers snapshot the handed-in policy as their frozen reference at
       construction, so the policy must carry iteration-start weights, while
       ``train(resume_from_checkpoint=...)`` restores the mid-training weights
       afterwards. If every checkpoint is half-written, restarts iteration ``N`` from
       the previous iteration's adapter with no resume checkpoint.
    C. **Between iterations** -- ``iteration_<N>/adapter/`` exists. Loads it and starts
       at ``N+1``.

    Args:
        run_dir: ``data/runs/<EXP_NAME>/``.
        base_policy: A freshly loaded base model. If it is already PEFT-wrapped, the
            wrapper is stripped with ``get_base_model()`` first -- attaching an adapter
            to an already-wrapped model produces nested PEFT, whose saved
            ``adapter_config.json`` no longer describes the model you think it does.
        tokenizer: Passed straight to :func:`patch_generate`.

    Returns:
        ``(start_iteration, policy, resume_checkpoint)``. ``resume_checkpoint`` is a
        path to hand to ``trainer.train(resume_from_checkpoint=...)`` and is non-None
        only in case B.

    Notes:
        :func:`patch_generate` is re-applied on every return path, because
        ``PeftModel.from_pretrained`` installs a fresh ``generate``.

        ``resume_checkpoint`` applies to the FIRST iteration of this process only. The
        caller must not pass it to iteration ``start_iteration + 1``.
    """
    from peft import PeftModel

    latest_iteration = get_latest_iteration(run_dir)

    # Always attach adapters to a plain base model (never nest PEFT wrappers).
    base_for_adapter = (
        base_policy.get_base_model()
        if hasattr(base_policy, "peft_config") and hasattr(base_policy, "get_base_model")
        else base_policy
    )

    candidate_training_dir = os.path.join(
        run_dir, f"{ITER_PREFIX}{latest_iteration + 1}", "training"
    )
    all_ckpts = list_hf_checkpoints(candidate_training_dir)

    # --- Case B: an iteration started but never wrote its adapter ---
    if all_ckpts:
        candidate_iter = latest_iteration + 1
        valid_ckpt = get_latest_valid_hf_checkpoint(candidate_training_dir)
        if valid_ckpt is not None:
            newest = os.path.basename(all_ckpts[-1])
            if os.path.basename(valid_ckpt) != newest:
                print(
                    f"  WARNING: newest checkpoint {newest} is incomplete; falling back to "
                    f"{os.path.basename(valid_ckpt)}"
                )
            print(f"  Resuming iteration_{candidate_iter} from {os.path.basename(valid_ckpt)}")
            # The returned policy holds the ITERATION-START weights, not the crash
            # checkpoint. Both TRL trainers snapshot the policy they are HANDED into
            # their frozen reference at construction (DPO copies the current "default"
            # adapter into "ref" and precomputes ref log-probs inside __init__; GRPO
            # snapshots default->ref the same way) -- all BEFORE
            # train(resume_from_checkpoint=...) restores the mid-training weights into
            # the "default" adapter. Loading the checkpoint here would therefore
            # silently re-anchor the KL/DPO reference to the crash point, so a resumed
            # iteration would train the objective of a different experiment.
            if latest_iteration == 0:
                # Iteration-1 resume: iteration-start == the bare base. The trainer
                # re-attaches a fresh LoRA (identity at step 0, exactly the reference
                # the crashed process used) and _load_from_checkpoint restores the
                # trained weights at train() time.
                policy = base_for_adapter
            else:
                iter_start_adapter = os.path.join(
                    run_dir, f"{ITER_PREFIX}{latest_iteration}", ADAPTER_SUBDIR
                )
                policy = PeftModel.from_pretrained(
                    base_for_adapter, iter_start_adapter, is_trainable=True
                )
            patch_generate(policy, tokenizer)
            return candidate_iter, policy, valid_ckpt

        print(
            f"  WARNING: no valid checkpoint in iteration_{candidate_iter}/training "
            f"({len(all_ckpts)} found, all incomplete)"
        )
        print(f"    Restarting iteration_{candidate_iter} from scratch")
        if latest_iteration == 0:
            policy = base_for_adapter
        else:
            adapter_path = os.path.join(
                run_dir, f"{ITER_PREFIX}{latest_iteration}", ADAPTER_SUBDIR
            )
            policy = PeftModel.from_pretrained(base_for_adapter, adapter_path, is_trainable=True)
        patch_generate(policy, tokenizer)
        return candidate_iter, policy, None

    # --- Case C: clean boundary between iterations ---
    if latest_iteration > 0:
        adapter_path = os.path.join(run_dir, f"{ITER_PREFIX}{latest_iteration}", ADAPTER_SUBDIR)
        print(f"  Resuming: loading adapter from iteration_{latest_iteration}")
        policy = PeftModel.from_pretrained(base_for_adapter, adapter_path, is_trainable=True)
        patch_generate(policy, tokenizer)
        return latest_iteration + 1, policy, None

    # --- Case A: fresh start ---
    patch_generate(base_for_adapter, tokenizer)
    return 1, base_for_adapter, None


def compute_cumulative_step_offset(run_dir: str) -> int:
    """Sum ``global_step`` across completed and in-progress iterations.

    Every iteration builds a new TRL trainer, so ``global_step`` restarts at 0 each
    time. Adding this offset when logging keeps the TensorBoard x-axis continuous
    across an arm instead of drawing every iteration on top of iteration 1.

    Notes:
        The in-progress iteration is counted from the latest *valid* checkpoint -- the
        same one :func:`resolve_start_state` will actually resume from -- so the offset
        and the resume point cannot disagree and produce overlapping step ranges.
    """
    offset = 0
    completed = list_iteration_checkpoints(run_dir)
    for it_num, _ in completed:
        training_dir = os.path.join(run_dir, f"{ITER_PREFIX}{it_num}", "training")
        ckpts = list_hf_checkpoints(training_dir)
        if ckpts:
            offset += int(os.path.basename(ckpts[-1]).split("-")[-1])

    latest_completed = completed[-1][0] if completed else 0
    in_progress_training_dir = os.path.join(
        run_dir, f"{ITER_PREFIX}{latest_completed + 1}", "training"
    )
    in_progress_ckpt = get_latest_valid_hf_checkpoint(in_progress_training_dir)
    if in_progress_ckpt is not None:
        partial_steps = int(os.path.basename(in_progress_ckpt).split("-")[-1])
        offset += partial_steps
        print(
            f"  Including {partial_steps} steps from in-progress "
            f"iteration_{latest_completed + 1}"
        )

    if offset > 0:
        print(f"  Cumulative step offset: {offset}")
    return offset
