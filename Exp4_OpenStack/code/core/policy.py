"""policy.py -- the local therapist policy: template, weights, decoding, resume.

Every other Exp4 module talks to a *remote* model over an OpenAI-compatible socket
(patient, oracle, judge). This module owns the one model that lives in this process:
the Llama-3.2-1B therapist policy that is actually being optimized. It covers

- the hand-written ChatML template (the base checkpoint ships none),
- base-weight load (bf16) and LoRA attachment,
- ``patch_generate`` -- the bind that makes ``stop_strings`` work at all,
- the anti-degeneracy stack (stop strings + ``clean_completion``),
- **the prompt rule** -- how a message list becomes the exact token ids the policy
  conditions on (:func:`render_prompt`, :func:`prompt_token_ids`,
  :func:`truncate_messages_drop_oldest`, :func:`build_prompt`), shared by serving AND
  training so the two can never see different prompts for the same conversation,
- batched therapist decoding that reports OOM instead of raising, and
- checkpoint discovery + multi-iteration resume.

**The prompt rule, in one line: rendered TEXT never carries a BOS; every TOKENIZATION
adds exactly one.** The two therapist variants disagree about BOS at the text level --
the Instruct checkpoint's native Llama-3 template writes a literal ``<|begin_of_text|>``
at the start of every render, the hand-written ChatML template writes none -- and the
consumers disagree about tokenization: this module's decode path and TRL's
``GRPOTrainer`` (``processing_class(text=prompts)``, i.e. ``add_special_tokens=True``)
both tokenize the prompt STRING. Left alone that gives the Instruct policy a double BOS
at training time and none of the two paths a BOS the base policy was pretrained with.
So every render goes through :func:`strip_leading_bos`, and every tokenization of a
prompt string goes through :func:`prompt_token_ids` (``add_special_tokens=True`` when the
tokenizer's post-processor adds a BOS, which is what TRL's call does too). The result on
BOTH variants: exactly one BOS, at serving and at training. For the BASE therapist this
ADDS a BOS at serving relative to the pre-rule code -- deliberate: Llama-3.2 base was
pretrained with one, and Exp4 has no data generated under the old rule.

**Generation never token-truncates.** An over-long conversation is shortened by dropping
its OLDEST turns whole while keeping the system message and the most recent turns
(:func:`truncate_messages_drop_oldest`) -- the same rule the training prompts use
(``core.conversations.build_truncated_training_prompt`` calls the same function), so a
PTO branch is sampled from byte-identical text to what its DPO pair trains on. This is a
science change relative to Exp3, whose serve-time prompts were LEFT-truncated at the
token level (once a conversation outgrew ``max_input_tokens`` every later therapist turn
was generated from a prompt that started mid-utterance with no system prompt, while its
training prompts kept the system message). It applies equally to both methods and to
both K arms.

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
from dataclasses import dataclass
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
    # The prompt rule (text <-> ids)
    "strip_leading_bos",
    "tokenizer_adds_bos",
    "render_prompt",
    "prompt_token_ids",
    "count_prompt_tokens",
    "message_overheads",
    "estimate_message_costs",
    "system_overhead",
    "truncate_messages_drop_oldest",
    "build_prompt",
    "TruncationCounter",
    "TRUNCATION_COUNTER",
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
        - ``truncation_side = left``: belt-and-braces for any HF/TRL path that token-
          truncates on its own -- if something ever does, it must lose the OLDEST tokens,
          never the patient utterance being answered. Exp4's own paths never rely on it:
          :func:`generate_therapist_batch` and the training-prompt builders shorten at
          the MESSAGE level via :func:`truncate_messages_drop_oldest` (system message and
          most recent turns kept whole) and tokenize with no ``truncation``.
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


# =============================================================================
# THE PROMPT RULE -- message list -> text -> token ids, one way, everywhere
# =============================================================================
#
# See the module docstring. Rendered text never carries a BOS; every tokenization of a
# prompt string adds exactly one (when the tokenizer has a BOS post-processor at all).
# Budgets (`max_input_tokens`, `max_prompt_tokens`) count the ids `prompt_token_ids`
# returns -- i.e. INCLUDING that BOS -- so a budget is exactly the model's input length.

# Per-tokenizer facts, keyed by id(). The entry keeps a strong reference to the tokenizer so
# its id cannot be recycled by a later object and read back a stale answer. A process holds
# one or two tokenizers, so the cache never grows.
_TOKENIZER_FACTS: Dict[int, Dict[str, Any]] = {}

# Content used to MEASURE per-role template overhead: one token on every BPE vocabulary
# that matters here, so the wrapper cost is the difference of two renders.
_TURN_PROBE = "XQZPROBE"


def _facts(tokenizer) -> Dict[str, Any]:
    entry = _TOKENIZER_FACTS.get(id(tokenizer))
    if entry is None:
        entry = {"tokenizer": tokenizer}
        _TOKENIZER_FACTS[id(tokenizer)] = entry
    return entry


def tokenizer_adds_bos(tokenizer) -> bool:
    """True iff ``tokenizer(text)`` prepends the BOS id (probed once, cached per tokenizer).

    The probe is the behaviour itself -- ``tokenizer("x")["input_ids"][0] == bos_token_id`` --
    not a config flag, because it is the post-processor's action that TRL's
    ``processing_class(text=...)`` call inherits. Both Llama-3.2 tokenizers answer True. A
    tokenizer with no ``bos_token_id`` (the offline stub in ``tools/smoke.py``) answers False
    without being called.
    """
    facts = _facts(tokenizer)
    if "adds_bos" not in facts:
        bos_id = getattr(tokenizer, "bos_token_id", None)
        adds = False
        if bos_id is not None and int(bos_id) >= 0:
            try:
                ids = tokenizer("x")["input_ids"]
            except TypeError:  # not callable -- fall back to the encode() API
                ids = tokenizer.encode("x", add_special_tokens=True)
            adds = bool(len(ids)) and int(ids[0]) == int(bos_id)
        facts["adds_bos"] = adds
    return bool(facts["adds_bos"])


def strip_leading_bos(text: str, tokenizer) -> str:
    """Remove ONE leading ``tokenizer.bos_token`` from *text*; no-op without a BOS token.

    The Llama-3 Instruct template writes ``<|begin_of_text|>`` into its rendered string;
    the ChatML template writes nothing. After this call both are BOS-free, and whichever
    consumer tokenizes the string next (this module's decode path, TRL's trainers) adds the
    single BOS itself. Exactly one is removed on purpose: a second one would be content.
    """
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def render_prompt(messages: Sequence[Dict[str, str]], tokenizer) -> str:
    """Chat-template text for *messages* with the generation prompt appended, BOS-stripped.

    Pins ``date_string=CHAT_TEMPLATE_DATE`` (the Llama-3.2 template interpolates the current
    date otherwise) and applies :func:`strip_leading_bos`. This is THE render: the decode
    path, the GRPO prompt extraction and the PTO pair builder all produce their prompt text
    here, so the text the policy generated from and the text it later trains on are
    byte-identical for the same messages.
    """
    text = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True,
        date_string=CHAT_TEMPLATE_DATE,
    )
    return strip_leading_bos(text, tokenizer)


def prompt_token_ids(text: str, tokenizer) -> List[int]:
    """The ids the policy conditions on for prompt *text*: exactly one BOS, then the text.

    ``add_special_tokens=True`` when the tokenizer adds a BOS -- the same call TRL's
    ``GRPOTrainer`` makes on the prompt string, so the two agree by construction -- and
    ``add_special_tokens=False`` otherwise (a tokenizer with no BOS post-processor would add
    nothing useful and some add an EOS). Strips a leading BOS from *text* first, so passing
    an un-stripped render cannot double it.
    """
    text = strip_leading_bos(text, tokenizer)
    if tokenizer_adds_bos(tokenizer):
        return list(tokenizer(text)["input_ids"])
    # Not coerced to int: a stub tokenizer (tools/smoke.py) returns words, and only len() matters.
    return list(tokenizer.encode(text, add_special_tokens=False))


def count_prompt_tokens(text: str, tokenizer) -> int:
    """``len(prompt_token_ids(text, tokenizer))`` -- the number every budget is compared to."""
    return len(prompt_token_ids(text, tokenizer))


def message_overheads(tokenizer) -> Dict[str, int]:
    """Per-role template wrapper cost in tokens, MEASURED on the tokenizer's live template.

    Renders a three-message probe and differences the counts, so the estimate tracks
    whichever template :func:`setup_tokenizer` left on the tokenizer -- the hand-written
    ChatML wrapper on the base therapist (~6 BPE pieces per marker), the native Llama-3
    header/footer on the Instruct one (single special tokens). Hardcoding the ChatML strings
    would over-bill every Instruct turn ~2x. Approximate at message joins (a real render can
    merge tokens across a boundary), which is fine: the estimate only picks a candidate drop
    point and the exact render that follows decides. Cached per tokenizer.
    """
    facts = _facts(tokenizer)
    if "overheads" not in facts:
        def _ntok(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        def _render(messages: List[Dict[str, str]]) -> str:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                date_string=CHAT_TEMPLATE_DATE,
            )

        sys_msgs = [{"role": "system", "content": _TURN_PROBE}]
        user_msgs = sys_msgs + [{"role": "user", "content": _TURN_PROBE}]
        both_msgs = user_msgs + [{"role": "assistant", "content": _TURN_PROBE}]
        n_probe = _ntok(_TURN_PROBE)
        n_sys = _ntok(_render(sys_msgs))
        n_user = _ntok(_render(user_msgs))
        n_both = _ntok(_render(both_msgs))
        facts["overheads"] = {
            "user": max(0, n_user - n_sys - n_probe),
            "assistant": max(0, n_both - n_user - n_probe),
        }
    return dict(facts["overheads"])


def estimate_message_costs(messages: Sequence[Dict[str, str]], tokenizer) -> List[int]:
    """Estimated token cost of every NON-system message: content + its role's wrapper overhead.

    Order-aligned with the non-system messages of *messages*. Feed the result to
    :func:`truncate_messages_drop_oldest` / :func:`build_prompt` when many prefixes of the
    same conversation are built (the prompt extraction pass), so the per-turn encodes happen
    once per conversation rather than once per slice.
    """
    overheads = message_overheads(tokenizer)
    costs: List[int] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        if role == "system":
            continue
        content_tokens = len(tokenizer.encode(str(msg.get("content", "")), add_special_tokens=False))
        costs.append(content_tokens + overheads.get(role, overheads["user"]))
    return costs


def _fixed_overhead(head: Sequence[Dict[str, str]], tokenizer) -> int:
    """Exact cost of the part that never gets dropped: system message + generation prompt + BOS.

    An EMPTY head (a message list with no leading system message -- ``generate_therapist_batch``
    accepts those, with ``system_overhead=None``) contributes 0. Rendering ``[]`` through the chat
    template is not "the fixed framing alone": Llama's template raises on an empty conversation,
    and ChatML's renders just the generation prompt. The 0 is only the starting point of the
    drop-oldest ESTIMATE in :func:`_fit_messages`; the exact search that follows renders the
    real candidate prompts and corrects it, so the framing is still counted to the token.
    """
    if not head:
        return 0
    return count_prompt_tokens(render_prompt(list(head), tokenizer), tokenizer)


def system_overhead(system_prompt: str, tokenizer) -> int:
    """Exact token cost of a system-only prompt: BOS + system message + generation suffix.

    A real render through the live template, so it counts whatever framing that template adds
    (the Llama-3 date header, the ChatML system turn) plus the BOS the tokenization adds.
    Compute once per pass and hand it to :func:`build_prompt` for every slice.
    """
    return _fixed_overhead([{"role": "system", "content": str(system_prompt)}], tokenizer)


def _fit_messages(
    messages: Sequence[Dict[str, str]],
    tokenizer,
    max_tokens: int,
    *,
    message_token_costs: Optional[Sequence[int]] = None,
    system_overhead: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str], int]:
    """The one implementation behind :func:`truncate_messages_drop_oldest` / :func:`build_prompt`.

    Returns ``(kept_messages, rendered_text, n_dropped)``; ``(None, None, n_body)`` when even
    the newest non-system message alone does not fit.

    The answer is CANONICAL -- the longest suffix of the non-system messages that fits, with
    the head -- and does not depend on the cost estimates: they only choose where the exact
    search starts, and the search then moves in both directions until it sits on the boundary.
    So the same messages and budget give the same prompt whether or not precomputed costs
    were supplied, and the budget is exact to the token.

    A list WITHOUT a leading system message is legal (``head`` is then empty, and its fixed
    overhead counts as 0 for the estimate -- see :func:`_fixed_overhead`); the drop-oldest search
    then keeps the newest turns that fit, or returns ``None`` when even the newest alone does
    not. It never renders an empty conversation. An empty *messages* has no turn to keep and
    returns ``(None, None, 0)``.
    """
    messages = list(messages)
    head = messages[:1] if messages and str(messages[0].get("role", "")) == "system" else []
    body = messages[len(head):]
    if not messages:
        return None, None, 0

    # Fast path -- and the exact check every prompt gets: one render, one count.
    text = render_prompt(messages, tokenizer)
    if count_prompt_tokens(text, tokenizer) <= max_tokens:
        return messages, text, 0
    if len(body) <= 1:
        # Nothing to drop: the newest (only) turn with the head is what was just measured
        # over budget. Never fall through to a head-only prompt -- that is a prompt with no
        # turn in it, which `None` exists to prevent.
        return None, None, len(body)

    # Estimate a drop point in one pass, then verify with real renders and keep dropping if
    # the estimate was optimistic. The head (system message) and the newest turn are what
    # survive: the therapist is replying to the patient's latest utterance, so that must
    # never be the thing that gets cut.
    costs = (list(message_token_costs) if message_token_costs is not None
             else estimate_message_costs(body, tokenizer))
    if len(costs) != len(body):
        raise ValueError(
            f"_fit_messages: message_token_costs has {len(costs)} entries for {len(body)} "
            f"non-system messages -- the costs must be order-aligned with the messages"
        )
    fixed = int(system_overhead) if system_overhead is not None else _fixed_overhead(head, tokenizer)

    drop = 1  # the full list was already measured over budget above
    total = fixed + sum(costs[drop:])
    while total > max_tokens and drop < len(body) - 1:
        total -= costs[drop]
        drop += 1
    # `drop` is now the estimate's first candidate; the exact search below corrects it either way.

    def _fits(d: int) -> Optional[str]:
        t = render_prompt(head + body[d:], tokenizer)
        return t if count_prompt_tokens(t, tokenizer) <= max_tokens else None

    # Estimate optimistic: keep dropping until it fits (or nothing fits).
    text = _fits(drop)
    while text is None and drop < len(body) - 1:
        drop += 1
        text = _fits(drop)
    if text is None:
        return None, None, len(body)
    # Estimate pessimistic: add turns back while they still fit, so the boundary is exact.
    while drop > 1:
        more = _fits(drop - 1)
        if more is None:
            break
        drop -= 1
        text = more
    return head + body[drop:], text, drop


def truncate_messages_drop_oldest(
    messages: Sequence[Dict[str, str]],
    tokenizer,
    max_tokens: int,
    *,
    message_token_costs: Optional[Sequence[int]] = None,
    system_overhead: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, str]]], int]:
    """Drop the OLDEST non-system messages until the rendered prompt fits *max_tokens*.

    Args:
        messages: Therapist-perspective chat messages; a leading ``system`` message is kept
            unconditionally.
        tokenizer: From :func:`setup_tokenizer` (carries the therapist's chat template).
        max_tokens: Budget for :func:`prompt_token_ids` of the rendered prompt -- BOS included.
        message_token_costs: Optional per-non-system-message estimates from
            :func:`estimate_message_costs` (same order), to skip re-encoding each turn.
        system_overhead: Optional exact cost from :func:`system_overhead`.

    Returns:
        ``(kept_messages, n_dropped)``. ``kept_messages`` is ``None`` when even the newest
        non-system message alone (with the system message) exceeds the budget -- the caller
        must treat that as a failure, never as "generate anyway". ``n_dropped`` counts the
        messages removed (``len(body)`` in the ``None`` case).

    Notes:
        Measured on the live template with the pinned ``date_string``; the result is what
        :func:`render_prompt` renders. Turns are dropped WHOLE: the prompt always starts on a
        message boundary, so the policy never sees a mid-utterance start or loses its system
        prompt, whichever therapist template is installed.
    """
    kept, _, n_dropped = _fit_messages(
        messages, tokenizer, max_tokens,
        message_token_costs=message_token_costs, system_overhead=system_overhead,
    )
    return kept, n_dropped


def build_prompt(
    messages: Sequence[Dict[str, str]],
    tokenizer,
    max_tokens: int,
    *,
    message_token_costs: Optional[Sequence[int]] = None,
    system_overhead: Optional[int] = None,
) -> Tuple[Optional[str], int]:
    """:func:`truncate_messages_drop_oldest` + :func:`render_prompt` in one call.

    Returns ``(prompt_text, n_dropped)``; ``prompt_text`` is ``None`` exactly when the
    truncation would be. The decode path and the training-prompt builders both call this,
    which is what makes their texts byte-identical for the same messages and budget.
    """
    _, text, n_dropped = _fit_messages(
        messages, tokenizer, max_tokens,
        message_token_costs=message_token_costs, system_overhead=system_overhead,
    )
    return text, n_dropped


@dataclass
class TruncationCounter:
    """Running totals of what :func:`generate_therapist_batch` did to its prompts.

    Attributes:
        prompts: Therapist prompts built (one per message list handed in).
        truncated: Prompts that lost at least one turn to fit ``max_input_tokens``.
        dropped_turns: Turns dropped, summed over all prompts.
        overflow: Prompts that could not be built at all (newest turn alone over budget);
            those items came back as ``None``.

    The conversation loop prints the per-batch delta as ``trunc <truncated>/<prompts>``;
    the trainers read :meth:`snapshot` at phase boundaries and log the delta. The rate is
    a science-relevant number -- a truncated prompt is a policy that no longer sees the
    start of the session -- so it is worth recording next to the per-iteration metadata.
    """

    prompts: int = 0
    truncated: int = 0
    dropped_turns: int = 0
    overflow: int = 0

    def snapshot(self) -> Dict[str, int]:
        return {"prompts": self.prompts, "truncated": self.truncated,
                "dropped_turns": self.dropped_turns, "overflow": self.overflow}

    def delta_since(self, snapshot: Dict[str, int]) -> Dict[str, int]:
        """The counts accumulated since *snapshot* (from :meth:`snapshot`)."""
        now = self.snapshot()
        return {k: now[k] - int(snapshot.get(k, 0)) for k in now}

    def reset(self) -> None:
        self.prompts = self.truncated = self.dropped_turns = self.overflow = 0


#: The process-wide counter :func:`generate_therapist_batch` updates. Therapist generates are
#: serialised by the GPU lock, so plain integer updates are safe.
TRUNCATION_COUNTER = TruncationCounter()


def generate_therapist_batch(
    model,
    tokenizer,
    batch_messages: List[List[Dict[str, str]]],
    *,
    max_tokens: int,
    temperature: float,
    max_input_tokens: int = 2048,
    stop_strings: Optional[Sequence[str]] = None,
) -> Tuple[Optional[List[Optional[str]]], Optional[str]]:
    """Generate one therapist reply per conversation in a single padded batch.

    Args:
        model: The therapist policy (base or PEFT-wrapped), already patched by
            :func:`patch_generate`.
        tokenizer: From :func:`setup_tokenizer` (left padding).
        batch_messages: One chat-message list per conversation, in the therapist's
            role convention (``system``/``user``/``assistant``).
        max_tokens: ``max_new_tokens`` per completion.
        temperature: Sampling temperature; sampling is always on (``do_sample=True``).
        max_input_tokens: Prompt budget in tokens, BOS included (the length of
            :func:`prompt_token_ids`). An over-budget conversation drops its OLDEST turns
            whole via :func:`build_prompt` -- the system message and the most recent
            turns survive, and the prompt is never token-truncated.
        stop_strings: ``None`` means the :data:`STOP_STRINGS` default (base-therapist
            ChatML markers). An EMPTY sequence means "no string stopping" and is the
            correct value for Instruct arms -- generation then stops on the
            :func:`therapist_stop_token_ids` eos list alone, with no per-call
            ``StopStringCriteria`` table build.

    Returns:
        ``(responses, None)`` on success, where ``responses[i]`` corresponds to
        ``batch_messages[i]`` and is one of three things:

        - a cleaned completion (:func:`clean_completion` applied),
        - ``""`` -- a DEGENERATE turn (the model produced nothing usable), or
        - ``None`` -- **no prompt could be built**: even the newest turn alone, with the
          system message, exceeds ``max_input_tokens``. Nothing was generated for that
          item; the others were. Callers must handle ``None`` as a failure of that item
          (the conversation loop marks the conversation failed; look-ahead freezes the
          sim) and never as an empty utterance.

        On a batch-level failure: ``(None, "oom")`` or ``(None, "runtime_error")``.

    Notes:
        **The prompt rule (module docstring).** Each message list is truncated at the
        message level, rendered BOS-free, and tokenized with ``add_special_tokens=True``
        when the tokenizer adds a BOS -- exactly one BOS on both therapist variants, and
        the same ids TRL's ``GRPOTrainer`` produces from the prompt string. What was done
        to the prompts is accumulated in :data:`TRUNCATION_COUNTER` (prompts / truncated /
        dropped turns / overflow) for the batch line and the trainers' logs.

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

    # Build every prompt under the rule: message-level drop-oldest, BOS-free text. The system
    # overhead is exact and identical for every list sharing a system prompt, so it is
    # measured once per distinct system prompt in the batch rather than once per item.
    overhead_by_system: Dict[str, int] = {}
    prompts: List[Optional[str]] = []
    n_truncated = 0
    n_dropped = 0
    for messages in batch_messages:
        sys_text = (str(messages[0].get("content", ""))
                    if messages and str(messages[0].get("role", "")) == "system" else None)
        if sys_text is not None and sys_text not in overhead_by_system:
            overhead_by_system[sys_text] = system_overhead(sys_text, tokenizer)
        text, dropped = build_prompt(
            messages, tokenizer, max_input_tokens,
            system_overhead=overhead_by_system.get(sys_text) if sys_text is not None else None,
        )
        prompts.append(text)
        if text is None:
            continue  # overflow: counted below, nothing was dropped INTO a prompt
        if dropped:
            n_truncated += 1
            n_dropped += dropped
    live_idx = [i for i, p in enumerate(prompts) if p is not None]
    n_overflow = len(prompts) - len(live_idx)

    TRUNCATION_COUNTER.prompts += len(prompts)
    TRUNCATION_COUNTER.truncated += n_truncated
    TRUNCATION_COUNTER.dropped_turns += n_dropped
    TRUNCATION_COUNTER.overflow += n_overflow
    if n_overflow:
        print(
            f"  WARNING: {n_overflow}/{len(prompts)} therapist prompts could not be built -- "
            f"the newest turn alone exceeds max_input_tokens={max_input_tokens}; those items "
            f"return None"
        )
    if not live_idx:
        return [None] * len(prompts), None

    # String criteria only when there are strings: an empty stop_strings=[] still pays the
    # per-call 128k-vocab criteria table build, so Instruct arms (empty stops) omit the kwargs
    # entirely and stop on the eos-id list alone.
    stop_kwargs = (
        {"stop_strings": effective_stops, "tokenizer": tokenizer} if effective_stops else {}
    )

    encoded = None
    outputs = None
    try:
        # No `truncation=`: the prompts already fit by construction (message-level
        # drop-oldest above), and token truncation would cut the system prompt and start
        # mid-utterance. add_special_tokens follows the tokenizer's BOS behaviour, so the
        # BOS-free text gets exactly one BOS -- the same ids TRL produces from the string.
        encoded = tokenizer(
            [prompts[i] for i in live_idx],
            return_tensors="pt",
            padding=True,
            add_special_tokens=tokenizer_adds_bos(tokenizer),
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

    responses: List[Optional[str]] = [None] * len(batch_messages)
    for row, i in enumerate(live_idx):
        new_tokens = outputs[row][padded_input_length:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses[i] = clean_completion(decoded)

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
