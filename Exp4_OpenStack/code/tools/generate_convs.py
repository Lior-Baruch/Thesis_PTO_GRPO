"""generate_convs.py -- one model state, N conversations, no training and no oracle.

Simulate ``--n-convs`` conversations with ONE therapist policy against the patient simulator and
write them to a conversations folder. That is the whole job: no branching, no look-ahead, no
preference pairs, no oracle calls, no gradient step. Three separate needs converge on it.

**1. REPAIR.** ``model_iter_k`` is normally produced as step 1 of iteration ``k+1``. A run that
dies between "iteration k's adapter was saved" and "iteration k+1's step 1 finished" leaves an
adapter with NO conversations -- a model state that can never be scored, and therefore a hole in
every contrast table that endpoint appears in. Exp3 hit exactly this (PTO LA5 had adapters for
iterations 1..5 and an empty ``model_iter_5``), and the alternative to a tool like this one is
re-running hours of training to recover a pass that costs ~50 minutes of generation.

**2. REPLICATE.** Every contested endpoint in Exp3 was a SINGLE draw of 96 conversations, so no
number in it has a noise floor: two arms differing by 0.1 could not be told from one arm drawn
twice. Therapist decoding is unseeded, so a second independent draw of the same model state costs
only GPU time here -- and Exp4's zero-API-cost stack makes scoring it free as well. What was
unaffordable in Exp3 is a housekeeping task in Exp4.

**3. LOCAL SMOKE.** This is the only GPU workload in the project that fits on the 12 GB local card,
which makes it the natural end-to-end check of the whole generation stack (ChatML template ->
batched decode -> stop strings -> patient socket -> session-end protocol -> CSV round trip) without
booking a Colab session.

Nothing here re-implements generation. The pass is
:func:`core.conversations.generate_all_conversations` with the policy from :mod:`core.policy`, so a
conversation produced by this tool is produced by the same code path -- and under the same knobs,
because the generation config is rebuilt from the arm's own ``run_metadata.json`` when one exists.

Two hazards this file is mostly about
-------------------------------------
**VRAM arithmetic comes before any CUDA allocation.** On the local RTX 5070 Ti an over-budget
request is a driver fault that REBOOTS the machine: no ``OutOfMemoryError``, no traceback, nothing
to catch, so ``--batch-size`` is a safety setting rather than a throughput knob. Measured in Exp3:
weights ~2.6 GiB plus ~1.1 GiB per concurrent conversation (batch 4 = 7.1 GiB observed; batch 32
would be ~38 GiB and did take a machine down). This tool does that arithmetic, prints it, and
refuses to start on a local host when it does not fit -- ``--force`` is the only way past. Do NOT
reason "it is inference only, so it is safe": the crash is about the size of the memory REQUEST,
not about a backward pass.

**A replicate must not be written into the primary conversations tree.** Arm discovery globs
``data/conversations/<EXP_NAME>/model_iter_<N>/pers*.csv``; a second draw landing there does not
produce two draws, it produces one folder in which whichever conversation was written last silently
wins. ``--conv-dir`` is how a replicate stays separate, and a replicate-marked directory such as
``data/conversations/<EXP_NAME>__rep1/model_iter_3`` keeps it recognisable months later. Because
Exp4 names files by the stable persona id, the collision is guaranteed rather than probabilistic:
``pers07.csv`` is persona 7 in every draw.

Usage
-----
::

    # Phase 2 gate -- 96 base-policy conversations against the Gemma patient
    python tools/generate_convs.py --exp-name GRPO4_Q1Q2_LA0_MCL12_G8_Ogemma4E2B_Patgemma4E2B \\
        --model-iter 0 --dry-run
    python tools/generate_convs.py --exp-name GRPO4_Q1Q2_LA0_MCL12_G8_Ogemma4E2B_Patgemma4E2B \\
        --model-iter 0 --batch-size 4

    # REPAIR -- refill a model state whose conversations were lost
    python tools/generate_convs.py --exp-name PTO4_... --model-iter 5 \\
        --adapter ../../data/runs/PTO4_.../iteration_5/adapter

    # REPLICATE -- a second independent draw, kept OUT of the primary tree
    python tools/generate_convs.py --exp-name GRPO4_... --model-iter 3 \\
        --adapter .../iteration_3/adapter \\
        --conv-dir ../../data/conversations/GRPO4_..._rep1/model_iter_3

Run it from ``Exp4_OpenStack/code``. Works unchanged on Colab (mounts Drive) and locally.
"""

from __future__ import annotations

# The local Blackwell card (sm_120) segfaults at CUDA init -- exit 139, no traceback -- when trl is
# imported AFTER torch. This script is the one that actually runs there, so trl goes first, ahead of
# every import that could pull torch in. A missing trl is not an error: nothing below imports it
# again, so with trl absent there is no ordering to violate in the first place.
try:  # noqa: SIM105 - the fallback needs a comment, so contextlib.suppress would hide the reason
    import trl  # noqa: F401  (imported for its side effect on native init order)
except ImportError:
    trl = None  # type: ignore[assignment]

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

__all__ = [
    "WEIGHTS_GIB",
    "PER_CONV_GIB",
    "SAFE_VRAM_FRACTION",
    "DEFAULT_BATCH_SIZE",
    "PATIENT_PROVIDERS",
    "VramBudget",
    "estimate_batch_vram_gib",
    "plan_vram_budget",
    "enforce_vram_budget",
    "load_run_defaults",
    "build_gen_config",
    "resolve_patient_binding",
    "resolve_patient_endpoint",
    "select_personas",
    "therapist_prompt_pair",
    "check_adapter_path",
    "load_policy",
    "build_parser",
    "main",
]


# =============================================================================
#  Constants
# =============================================================================

#: Llama-3.2-1B bf16 weights plus this process's CUDA context, in GiB. Measured in Exp3 on the
#: local card; it is the fixed part of the budget, paid once regardless of batch size.
WEIGHTS_GIB = 2.6

#: Marginal VRAM per CONCURRENT conversation, in GiB (KV cache + activations for one padded row of
#: the batch). Measured in Exp3 on the 12 GB card: batch 4 = 7.1 GiB observed against 2.6 + 4 x 1.1
#: = 7.0 predicted.
#:
#: WARNING: this constant is calibrated on the local card and is a conservative UPPER bound at large
#: batch -- the observed batch-6 figure (~8.0 GiB) is already below the 9.2 GiB it predicts, because
#: rows share padding rather than each paying a full-length cache. Treat it as a safety margin, not
#: as a forecast, and see :func:`enforce_vram_budget` for why that asymmetry decides where the guard
#: refuses and where it only warns.
PER_CONV_GIB = 1.1

#: Fraction of FREE VRAM the estimate may claim. The remainder absorbs allocator fragmentation
#: between batches (consecutive batches reach different maximum sequence lengths, so freed blocks
#: are the wrong size to reuse) and anything else that shares the card.
SAFE_VRAM_FRACTION = 0.85

#: Default conversations in flight, per host. The local value is a SAFETY setting; the Colab value
#: is a throughput one -- a big batch on an A100 amortizes the patient round-trip across all
#: conversations, which is the whole reason the loop is batched.
DEFAULT_BATCH_SIZE: Dict[str, int] = {"local": 6, "colab": 64}

#: Providers this tool can bind the patient to. Deliberately NOT all of ``roles.PROVIDERS``:
#: :func:`core.conversations.generate_patient_response` speaks the OpenAI chat-completions shape, so
#: an ``anthropic`` patient would need a different call path and would fail on the first turn.
PATIENT_PROVIDERS = ("openai_compat", "openai")

#: Seconds to wait for the patient endpoint to answer ``GET {base_url}/models`` before giving up.
#: Short on purpose: this is a reachability probe, not a server launch. Starting a vLLM server from
#: here would claim its pre-allocation on the same card the therapist is about to load onto.
SERVER_PROBE_SECONDS = 30.0

#: The oracle semaphore is never used on this path (no scoring happens), but ``AsyncPrimitives``
#: requires a positive bound for both.
_ORACLE_CONCURRENCY_UNUSED = 1

_GIB_PER_MIB = 1.0 / 1024.0


# =============================================================================
#  Bootstrap
# =============================================================================


def _bootstrap_sys_path() -> str:
    """Put ``Exp4_OpenStack/code`` on ``sys.path`` and return it.

    Returns:
        Absolute path of the ``code/`` directory this file lives under.

    Notes:
        The trainers prepend the same directory, so ``core``, ``roles``, ``naming``,
        ``questionnaires``, ``system_prompts_builder`` and ``tools`` resolve identically whether the
        caller is a notebook or this script. Derived from ``__file__`` rather than from the cwd:
        the tool is documented as "run it from ``code/``", but a wrong cwd should produce a working
        run rather than an ImportError three frames deep.
    """
    here = os.path.dirname(os.path.abspath(__file__))          # <root>/code/tools
    code_dir = os.path.abspath(os.path.join(here, os.pardir))  # <root>/code
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    return code_dir


# =============================================================================
#  VRAM budget -- arithmetic BEFORE any CUDA allocation
# =============================================================================


@dataclass(frozen=True)
class VramBudget:
    """What one generation pass will ask the card for, against what the card has free.

    Attributes:
        batch_size: Conversations in flight at once -- the only term that scales.
        estimate_gib: :data:`WEIGHTS_GIB` + ``batch_size`` x :data:`PER_CONV_GIB`.
        free_gib: Free VRAM as ``nvidia-smi`` reports it, or ``None`` if it could not be asked.
            FREE rather than total on purpose: a vLLM server serving the patient on the same card
            has already taken its pre-allocation, and that reservation never comes back.
        total_gib: Total VRAM, for the report only.
        budget_gib: ``free_gib`` x :data:`SAFE_VRAM_FRACTION`, or ``None``.
        fits: ``estimate_gib <= budget_gib``, or ``None`` when nothing could be measured.
    """

    batch_size: int
    estimate_gib: float
    free_gib: Optional[float]
    total_gib: Optional[float]
    budget_gib: Optional[float]
    fits: Optional[bool]

    def arithmetic(self) -> str:
        """The estimate spelled out as a sum, so a reader can audit every term.

        A composite number quoted as one figure never gets checked; quoted as
        ``2.6 + 6 x 1.1 = 9.2`` it does.
        """
        line = (f"    {WEIGHTS_GIB:.1f} GiB weights + {self.batch_size} x {PER_CONV_GIB:.1f} GiB "
                f"per conversation = {self.estimate_gib:.1f} GiB requested")
        if self.free_gib is None:
            return line + "\n    free VRAM: UNKNOWN (nvidia-smi unavailable)"
        return (line + f"\n    {self.free_gib:.1f} GiB free of {self.total_gib:.1f} GiB total"
                f" x {SAFE_VRAM_FRACTION:.2f} safety = {self.budget_gib:.1f} GiB budget"
                f"  ->  {'FITS' if self.fits else 'OVER BUDGET'}")


def _nvidia_smi_memory_gib(device: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """``(free_gib, total_gib)`` for *device* via ``nvidia-smi``, or ``(None, None)``.

    Notes:
        ``nvidia-smi`` rather than torch, and no torch fallback at all. Reading the figure through
        torch initialises a CUDA context in this process, and the entire point of the caller is to
        decide whether there is room BEFORE claiming any -- a guard that allocates in order to run
        is not a guard.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if proc.returncode != 0:
        return None, None

    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not (0 <= device < len(lines)):
        return None, None
    try:
        free_mib, total_mib = (float(part) for part in lines[device].split(",")[:2])
    except (TypeError, ValueError):
        return None, None
    return free_mib * _GIB_PER_MIB, total_mib * _GIB_PER_MIB


def estimate_batch_vram_gib(batch_size: int) -> float:
    """Peak VRAM a pass at *batch_size* is expected to request, in GiB.

    Args:
        batch_size: Conversations simulated concurrently.

    Returns:
        ``WEIGHTS_GIB + batch_size * PER_CONV_GIB``.

    Notes:
        Linear because the measured points are: the weights are paid once, and every additional
        concurrent conversation adds its own KV cache and activations. See :data:`PER_CONV_GIB` for
        why the linear term is an upper bound rather than a forecast.
    """
    return WEIGHTS_GIB + max(0, int(batch_size)) * PER_CONV_GIB


def plan_vram_budget(batch_size: int, *, device: int = 0) -> VramBudget:
    """Measure the card and compute the budget. Allocates nothing, raises nothing."""
    free_gib, total_gib = _nvidia_smi_memory_gib(device)
    estimate = estimate_batch_vram_gib(batch_size)
    budget = None if free_gib is None else free_gib * SAFE_VRAM_FRACTION
    fits = None if budget is None else estimate <= budget
    return VramBudget(batch_size=int(batch_size), estimate_gib=estimate, free_gib=free_gib,
                      total_gib=total_gib, budget_gib=budget, fits=fits)


def enforce_vram_budget(budget: VramBudget, *, host: str, force: bool) -> None:
    """Stop the run when the estimate does not fit, on the host where not stopping costs a reboot.

    Args:
        budget: From :func:`plan_vram_budget`.
        host: ``"local"`` or ``"colab"`` (:func:`core.runtime.detect_host`).
        force: Downgrade every refusal to a warning. The caller typed ``--force``; they own it.

    Raises:
        SystemExit: on a local host when the estimate exceeds the budget, or when the card could
            not be measured at all.

    Notes:
        **The two hosts fail differently, and the guard follows the failure, not the arithmetic.**
        On the local RTX 5070 Ti an over-budget request is a driver fault that takes the OS down
        with it -- there is no exception, so nothing downstream can recover and the guard must fail
        CLOSED, including when the card cannot be measured. On Colab the same request raises
        ``torch.OutOfMemoryError``, which :func:`core.policy.generate_therapist_batch` already
        catches and reports as ``(None, "oom")`` so the caller can retry smaller; refusing there
        would also block the documented A100 default of ``--batch-size 64``, which the linear model
        over-predicts at 73 GiB and which Exp3 ran routinely on a 40 GB card. So Colab gets the same
        arithmetic printed and a warning, and keeps going.

        This is deliberately NOT the reasoning "generation is inference-only, so it is safe". The
        hazard is the size of the memory request; what differs between the hosts is only whether
        the failure is catchable.
    """
    if budget.fits is True:
        return

    # The caller prints budget.arithmetic() in its plan block; restating the two numbers inside each
    # message below keeps this function readable on its own without duplicating that block.
    if budget.fits is None:
        problem = ("Could not determine free VRAM (nvidia-smi is unavailable), so the batch-size "
                   "guard cannot be evaluated.")
    else:
        problem = (f"Batch size {budget.batch_size} is expected to request "
                   f"{budget.estimate_gib:.1f} GiB against a {budget.budget_gib:.1f} GiB budget.")

    if force:
        print(f"  !! {problem}\n  !! --force given: starting anyway.")
        return

    if host != "local":
        print(f"  !! {problem}\n"
              f"  !! Host is {host!r}, where an over-budget request raises a catchable "
              f"OutOfMemoryError and the conversation loop retries at a smaller batch, so this is "
              f"a warning rather than a refusal. Note that the per-conversation constant is "
              f"calibrated on the 12 GB local card and OVER-predicts on an A100.")
        return

    largest = max(1, int((budget.budget_gib - WEIGHTS_GIB) // PER_CONV_GIB)) if budget.budget_gib \
        else 1
    raise SystemExit(
        f"REFUSING TO START.\n"
        f"  {problem}\n"
        f"  On this machine an over-budget VRAM request is a GPU/driver fault that REBOOTS the "
        f"host: it does not raise OutOfMemoryError, there is no traceback, and nothing can catch "
        f"it. --batch-size is a safety setting here, not a throughput knob.\n"
        f"  Fix: pass --batch-size {largest} or lower"
        + ("" if budget.free_gib is None else
           f" (budget {budget.budget_gib:.1f} GiB = {budget.free_gib:.1f} GiB free x "
           f"{SAFE_VRAM_FRACTION:.2f}, minus {WEIGHTS_GIB:.1f} GiB of weights)")
        + ", free the card (a resident vLLM server holds its pre-allocation for its whole life), "
          "or pass --force if you have done the arithmetic yourself."
    )


# =============================================================================
#  Configuration -- rebuilt from the arm's own run_metadata.json where one exists
# =============================================================================


def load_run_defaults(metadata_path: str) -> Dict[str, Any]:
    """Read an arm's ``run_metadata.json`` config sections. ``{}`` when there is no usable file.

    Args:
        metadata_path: ``data/runs/<EXP_NAME>/run_metadata.json``.

    Returns:
        The payload's ``config`` mapping (``training`` / ``generation`` / ``roles`` / ...), or
        ``{}``.

    Notes:
        **This is the fidelity mechanism.** A repaired model state must be generated the way its
        siblings were -- same conversation length, same temperatures, same token caps, same patient
        model -- or the "missing" state becomes a state generated under different rules, which is
        worse than leaving the hole. ``run_metadata.json`` is ``asdict`` of the configs the run
        actually froze, so it round-trips exactly and cannot drift from a notebook global someone
        has since edited.

        An absent file is a normal state, not an error: the Phase 2 base pass runs before any arm
        has trained. The caller then falls back to the defaults in :class:`core.config.GenConfig`,
        which ARE the matched grid.

        WARNING: ``run_metadata.json`` is overwritten by every process that works on the arm. If
        that arm was resumed under changed knobs, this file describes the LAST process, not the one
        that generated the sibling states -- ``run_metadata_history.jsonl`` next to it holds the
        superseded payloads, one line per process. Check it before trusting a repair whose arm has
        more than one line there.
    """
    if not os.path.isfile(metadata_path):
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  !! could not read {metadata_path} ({exc}); falling back to defaults")
        return {}
    config = payload.get("config")
    return dict(config) if isinstance(config, Mapping) else {}


def _filter_to_fields(cls, values: Any) -> Dict[str, Any]:
    """Keep only the keys that are fields of the dataclass *cls*.

    Metadata carries extras a dataclass never had (``roles`` entries gain ``tag`` and ``is_local``),
    and a future schema may add more. Dropping unknown keys means an older tool keeps working
    against a newer record instead of dying on a keyword argument.
    """
    if not isinstance(values, Mapping):
        return {}
    known = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in values.items() if k in known}


def build_gen_config(config: Dict[str, Any], *, batch_size: int,
                     n_convs: Optional[int], max_utterances: Optional[int]):
    """Assemble the :class:`core.config.GenConfig` this pass runs under.

    Args:
        config: The ``config`` mapping from :func:`load_run_defaults` (may be empty).
        batch_size: Resolved ``--batch-size``.
        n_convs: ``--n-convs``, or ``None`` to keep the recorded/default value.
        max_utterances: ``--max-utterances``, or ``None``.

    Returns:
        A frozen ``GenConfig``.

    Notes:
        Precedence is CLI > recorded > default, and the two optional arguments are ``None`` rather
        than pre-filled with the default for exactly that reason: a CLI default is
        indistinguishable from a typed value, so ``--n-convs`` defaulting to 96 would silently
        override an arm that recorded 48.

        Only these three knobs are exposed on the command line. The rest (temperatures, token caps,
        MCL) define what a conversation IS, and a conversation generated under different ones is not
        a draw from the same distribution as its siblings -- which is the whole point of rebuilding
        from the arm's own record. Change them in cell 1 and re-record, never here.
    """
    from core.config import GenConfig

    gen = GenConfig(**_filter_to_fields(GenConfig, config.get("generation")))
    overrides: Dict[str, Any] = {"conversation_batch_size": int(batch_size)}
    if n_convs is not None:
        overrides["num_conversations_per_iter"] = int(n_convs)
    if max_utterances is not None:
        overrides["num_utterances_for_data"] = int(max_utterances)
    return dataclasses.replace(gen, **overrides)


def resolve_patient_binding(config: Dict[str, Any], *, model: Optional[str],
                            provider: Optional[str], base_url: Optional[str]):
    """The patient :class:`roles.RoleBinding` for this pass.

    Args:
        config: The ``config`` mapping from :func:`load_run_defaults`.
        model: ``--patient-model`` or ``None``.
        provider: ``--patient-provider`` or ``None``.
        base_url: ``--base-url`` or ``None``.

    Returns:
        A frozen ``RoleBinding``.

    Raises:
        SystemExit: if the resolved provider is not in :data:`PATIENT_PROVIDERS`.

    Notes:
        With no model/provider override the recorded binding is reconstructed WHOLE -- including
        ``request_timeout``, ``max_retries`` and ``extra_body_json``. That last one is the
        thinking-off switch, and losing it is invisible: the run keeps working while every patient
        turn quietly pays for reasoning tokens (or comes back as ``content=None``, which the retry
        loop reports as a probable thinking-off failure).

        Overriding the model or the provider rebuilds through ``roles.make_binding``, which
        re-attaches the thinking-off body for ``openai_compat`` and correctly omits it for a vendor
        API, where an unknown body key is a 400.
    """
    from roles import DEFAULT_PATIENT_MODEL, RoleBinding, make_binding

    recorded = (config.get("roles") or {}).get("patient") if isinstance(
        config.get("roles"), Mapping) else None

    if recorded and model is None and provider is None:
        binding = RoleBinding(**_filter_to_fields(RoleBinding, recorded))
    else:
        recorded = recorded if isinstance(recorded, Mapping) else {}
        binding = make_binding(
            provider or str(recorded.get("provider") or "openai_compat"),
            model or str(recorded.get("model") or DEFAULT_PATIENT_MODEL),
        )

    if base_url:
        binding = dataclasses.replace(binding, base_url=base_url)

    if binding.provider not in PATIENT_PROVIDERS:
        raise SystemExit(
            f"--patient-provider {binding.provider!r} is not supported for the patient role "
            f"(supported: {list(PATIENT_PROVIDERS)}). core.conversations.generate_patient_response "
            f"issues chat.completions calls, so a provider with a different request shape would "
            f"fail on the first patient turn of every conversation."
        )
    return binding


def resolve_patient_endpoint(binding, *, fatal: bool):
    """Fill in (and verify) ``base_url`` for a locally-served patient.

    Args:
        binding: The patient binding.
        fatal: Raise ``SystemExit`` when the endpoint cannot be reached. ``False`` during
            ``--dry-run``, where an unreachable server is worth reporting but is not a reason to
            fail a plan that spends nothing.

    Returns:
        The binding, with ``base_url`` set when it could be resolved.

    Raises:
        SystemExit: when *fatal* and no reachable endpoint was found. ``RuntimeError`` propagates
            when a loopback port answers with a DIFFERENT model -- that one is never downgraded to
            a warning, not even in a dry run.

    Notes:
        This deliberately ADOPTS a server and never starts one. ``vllm serve``'s
        ``gpu_memory_utilization`` is a pre-allocation held for the process's whole life, so
        launching one from a tool that is about to load the therapist onto the same card would
        claim the memory the VRAM guard just budgeted -- silently, after the guard ran.

        Adoption goes through ``tools.vllm_serve.adopt_if_running``, which raises when the port
        serves a different model. That check matters more here than anywhere else: a patient swap
        produces a complete, valid-looking folder of conversations that are simply not the arm's
        conversations, and nothing downstream can tell. It is therefore applied to an explicit
        loopback ``--base-url`` too, not only to the port :func:`roles.plan_servers` would pick --
        typing the port by hand is exactly when the wrong server is easiest to hit.

        A NON-loopback ``--base-url`` (a server on another machine) gets a reachability probe only:
        ``adopt_if_running`` addresses a port on ``127.0.0.1``, so it cannot speak for a remote
        host, and inventing a second probe path here would be a second thing to keep in sync.
    """
    from roles import plan_servers
    from tools.vllm_serve import adopt_if_running, wait_until_ready

    if not binding.is_local:
        return binding

    spec = _loopback_spec(binding) if binding.base_url else plan_servers({"patient": binding})[0]

    if spec is None:                                  # remote endpoint: reachability only
        try:
            wait_until_ready(binding.base_url, timeout=SERVER_PROBE_SECONDS, poll_seconds=2.0)
        except (TimeoutError, RuntimeError) as exc:
            _endpoint_failure(
                f"patient endpoint {binding.base_url} did not answer within "
                f"{SERVER_PROBE_SECONDS:.0f}s ({exc})", fatal=fatal,
                fix="Point --base-url at a live server.")
        return binding

    handle = adopt_if_running(spec)                   # raises if the port serves another model
    if handle is None:
        _endpoint_failure(
            f"no server answering on port {spec.port} for {binding.model!r}", fatal=fatal,
            fix=(f"Start one (`vllm serve {binding.model} --port {spec.port} "
                 f"--gpu-memory-utilization 0.25`), or pass --base-url for a server elsewhere."))
        return binding

    print(f"  adopted patient server: {handle.model} @ {handle.base_url}")
    return dataclasses.replace(binding, base_url=handle.base_url)


def _loopback_spec(binding):
    """A :class:`roles.ServeSpec` addressing ``binding.base_url``, or ``None`` if it is not local.

    Exists so an explicit ``--base-url`` on this machine gets the same served-model check as an
    auto-adopted one. ``ServeSpec``'s other fields are irrelevant here -- ``adopt_if_running`` reads
    only ``model`` and ``port``, because a running server's memory settings cannot be read back over
    HTTP anyway.
    """
    from roles import ServeSpec

    parts = urlsplit(binding.base_url or "")
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    try:
        port = parts.port
    except ValueError:                                # a malformed port in the URL
        return None
    return None if not port else ServeSpec(model=binding.model, port=int(port))


def _endpoint_failure(problem: str, *, fatal: bool, fix: str) -> None:
    """Refuse (or warn) about an unreachable patient endpoint.

    Fatal on a real pass: without a patient, every conversation would exhaust its retry budget one
    at a time and the pass would burn the full no-progress allowance before reporting nothing.
    Non-fatal under ``--dry-run``, which is allowed to describe a plan for a server nobody has
    started yet.
    """
    if fatal:
        raise SystemExit(f"REFUSING TO START. {problem}. {fix}")
    print(f"  !! {problem}. {fix}")


# =============================================================================
#  Personas and prompts
# =============================================================================


def select_personas(n_permutations: int, n_convs: int, seed: int) -> List[int]:
    """Which personas run, in processing order.

    Args:
        n_permutations: Size of the canonical permutation list (96).
        n_convs: How many conversations to produce.
        seed: Shuffle seed.

    Returns:
        Persona ids, shuffled, truncated to *n_convs*.

    Notes:
        **The seed is far less load-bearing in Exp4 than it was in Exp3.** Exp3 named conversation
        files by the shuffled processing index, so ``conversation_3.csv`` meant a different patient
        every iteration and the seed convention decided what every file MEANT -- which is why this
        tool's Exp3 predecessor carried a whole ``--verify-seeds`` ceremony (replaying the shuffle,
        matching the age the patient states, plus two decoy offsets) before it was allowed to spend
        anything. Exp4 names files by the stable persona id, so the shuffle can no longer change
        what a file means. It decides only the order conversations are processed in and, when
        ``n_convs < n_permutations``, which subset runs.

        A full pass therefore makes the seed irrelevant to membership; keep the shuffle anyway so a
        truncated pass is a spread over the persona grid rather than the first cell of every nested
        loop (all Male, all Smoking, all FewMonths...).
    """
    order = list(range(int(n_permutations)))
    random.Random(int(seed)).shuffle(order)
    return order[: max(0, int(n_convs))]


def therapist_prompt_pair(permutations: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    """``(therapist_system_prompt, therapist_init_utterance)`` for the whole pass.

    Args:
        permutations: ``generate_all_permutations(only_expert_therapist=True)``.

    Returns:
        The single therapist system prompt and its scripted opening line.

    Raises:
        SystemExit: if the permutations disagree about either.

    Notes:
        Under the expert filter every permutation carries the same counselor persona (Good level,
        one name), so the therapist side is a constant and the 96 permutations vary only the
        patient. The check exists because that invariant is an argument, not a guarantee: drop the
        filter and the therapist prompt would silently vary per conversation, so the folder would
        no longer hold one policy under one prompt and every per-arm mean would mix three therapist
        personas.
    """
    if not permutations:
        raise SystemExit("generate_all_permutations() returned nothing -- cannot build a prompt.")
    prompts = {str(p.get("counselor_system_prompt") or "") for p in permutations}
    openings = {str(p.get("counselor_init_utterance") or "") for p in permutations}
    if len(prompts) != 1 or len(openings) != 1:
        raise SystemExit(
            f"the permutation set carries {len(prompts)} therapist system prompt(s) and "
            f"{len(openings)} opening utterance(s); exactly one of each is required. This tool "
            f"builds them with only_expert_therapist=True, under which the therapist side is "
            f"constant -- a set that varies it would mix therapist personas inside one model state."
        )
    return prompts.pop(), openings.pop()


# =============================================================================
#  Policy
# =============================================================================


def check_adapter_path(adapter: Optional[str]) -> None:
    """Refuse an ``--adapter`` that looks like a path but is not one. No-op for a Hub id.

    Args:
        adapter: The ``--adapter`` value, or ``None``.

    Raises:
        SystemExit: when *adapter* contains a path separator (or is absolute) and no such directory
            exists.

    Notes:
        Called during the plan, so ``--dry-run`` catches the typo, and again inside
        :func:`load_policy`, which cannot assume its caller checked. An unchecked path reaches
        ``PeftModel.from_pretrained`` and comes back as a Hub 404 for a repository nobody meant to
        fetch -- several minutes and one confusing traceback later.

        A value with no separator is left alone: it is a Hub id, and only the Hub can say whether
        it exists.
    """
    if not adapter:
        return
    looks_like_path = (os.sep in adapter or (os.altsep or "") in adapter
                       or os.path.isabs(adapter))
    if looks_like_path and not os.path.isdir(adapter):
        raise SystemExit(
            f"REFUSING TO START. --adapter {adapter!r} looks like a path but is not a directory. "
            f"A completed iteration writes iteration_<N>/adapter/; if that folder is missing, the "
            f"iteration never finished training and there is no policy to generate with."
        )


def load_policy(base_model_id: str, tokenizer_id: str, adapter: Optional[str], *,
                use_4bit: bool = False):
    """Load the therapist policy and its tokenizer, ready to generate.

    Args:
        base_model_id: Therapist base weights (``meta-llama/Llama-3.2-1B``).
        tokenizer_id: Tokenizer id; falls back to *base_model_id* when empty.
        adapter: LoRA adapter directory or Hub id, or ``None`` for the BASE policy.
        use_4bit: Pass-through to :func:`core.policy.setup_base_model`. Exp4 never runs this.

    Returns:
        ``(policy, tokenizer)``.

    Raises:
        SystemExit: via :func:`check_adapter_path`, when *adapter* looks like a filesystem path but
            does not exist.

    Notes:
        Three settings that are easy to lose and expensive to notice:

        * ``patch_generate`` is re-applied AFTER the PEFT wrap. ``PeftModel.from_pretrained``
          installs a fresh ``generate``, and without the tokenizer bound into it ``stop_strings`` is
          silently inert -- generation then runs to the token cap and the leaked ChatML self-play
          lands in the saved conversation.
        * ``use_cache`` is flipped back ON. :func:`core.policy.setup_base_model` leaves it False
          because that is the training setting; generating without a KV cache is correct but
          hours slower.
        * ``eval()`` disables LoRA dropout, so a generate-only pass is not sampling from a
          stochastically-thinned adapter.
    """
    from core.policy import patch_generate, setup_base_model, setup_tokenizer, sync_pad_token

    check_adapter_path(adapter)
    tokenizer = setup_tokenizer(tokenizer_id or base_model_id)
    policy = setup_base_model(base_model_id, use_4bit=use_4bit)
    sync_pad_token(policy, tokenizer)

    if adapter:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, adapter, is_trainable=False)
        print(f"  Loaded adapter: {adapter}")

    patch_generate(policy, tokenizer)                 # after every re-wrap, not just at base load
    policy.config.use_cache = True                    # generation setting; the loader set it False
    policy.eval()
    return policy, tokenizer


# =============================================================================
#  CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface. See the module docstring for worked examples."""
    parser = argparse.ArgumentParser(
        prog="generate_convs.py",
        description=("Generate-only pass: simulate N conversations with ONE model state and write "
                     "them to the conversations tree. No training, no oracle, no preference "
                     "building."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Batch size is a SAFETY setting on the 12 GB local card: an over-budget VRAM "
                "request reboots the machine instead of raising. The arithmetic is printed before "
                "anything touches CUDA."),
    )
    parser.add_argument("--exp-name", required=True,
                        help="EXPERIMENT_NAME of the arm these conversations belong to")
    parser.add_argument("--model-iter", type=int, default=0,
                        help="MODEL STATE being generated (default 0 = the untrained base policy). "
                             "State N is the policy from iteration_N/adapter.")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter directory or Hub id; OMIT for the base policy")
    parser.add_argument("--n-convs", type=int, default=None,
                        help="conversations to generate (default: the arm's recorded value, "
                             "else 96 = one per persona)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="conversations in flight (default: 6 local / 64 Colab). SAFETY knob "
                             "locally -- see the VRAM arithmetic printed at startup.")
    parser.add_argument("--conv-dir", default=None,
                        help="write conversations HERE instead of "
                             "data/conversations/<EXP_NAME>/model_iter_<N>/. REQUIRED for a "
                             "replicate draw, which must never share a folder with the primary.")
    parser.add_argument("--seed", type=int, default=None,
                        help="persona-shuffle and patient-request seed "
                             "(default: the arm's recorded training seed, else 42)")
    parser.add_argument("--patient-model", default=None,
                        help="override the patient model id")
    parser.add_argument("--patient-provider", default=None, choices=list(PATIENT_PROVIDERS),
                        help="override the patient provider "
                             "(default: recorded, else openai_compat)")
    parser.add_argument("--base-url", default=None,
                        help="patient endpoint; omit to adopt the local server on the planned port")
    parser.add_argument("--max-utterances", type=int, default=None,
                        help="cap conversation length in utterances (therapist + patient)")
    parser.add_argument("--data-root", default=None,
                        help="override the data/ directory (default: resolved from the workspace)")
    parser.add_argument("--force", action="store_true",
                        help="proceed despite the VRAM guard. You own the arithmetic.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit before loading the model or spending "
                             "anything")
    return parser


# =============================================================================
#  Guards on where the output lands
# =============================================================================


def _check_state_matches_adapter(model_iter: int, adapter: Optional[str], *,
                                 canonical: bool) -> None:
    """Refuse to write a policy's conversations into another policy's model-state folder.

    ``model_iter_0`` is by definition the untrained base and ``model_iter_N`` (N >= 1) is by
    definition the iteration-N adapter, so ``--model-iter 5`` with no ``--adapter`` would file base
    conversations as iteration 5's output. Nothing downstream can detect that: the folder is
    complete, the CSVs are valid, and the arm's trajectory simply flattens.

    Only the CANONICAL tree is protected. With ``--conv-dir`` the caller has already said the output
    is not the arm's own state folder, so the mismatch is reported and allowed.
    """
    if model_iter > 0 and not adapter:
        message = (f"--model-iter {model_iter} names a TRAINED state but no --adapter was given, "
                   f"so this would generate with the BASE policy and file the result as iteration "
                   f"{model_iter}'s output.")
        fix = f"Pass --adapter <run_dir>/iteration_{model_iter}/adapter, or use --model-iter 0."
    elif model_iter == 0 and adapter:
        message = ("--model-iter 0 names the UNTRAINED base state but an --adapter was given, so "
                   "this would file trained conversations as the base policy's output.")
        fix = "Pass the --model-iter that matches the adapter, or drop --adapter."
    else:
        return

    if canonical:
        raise SystemExit(f"REFUSING TO START. {message}\n  {fix}\n  (Or pass --conv-dir to write "
                         f"outside the arm's own conversations tree, where this is your call.)")
    print(f"  !! {message}\n  !! Allowed because --conv-dir puts the output outside the arm's "
          f"tree. {fix}")


def _check_patient_matches_arm(binding, arm: Optional[Any], *, canonical: bool) -> None:
    """Refuse to file conversations produced against a DIFFERENT patient as the arm's own.

    The arm name encodes the patient as ``Pat<tag>``, and every conversation under
    ``data/conversations/<EXP_NAME>/`` is read back as that arm's data -- scored by ``Run_Eval``,
    plotted as a point on the arm's trajectory, differenced against its other model states. A
    ``--patient-model`` / ``--patient-provider`` override that is not reflected in the folder name
    therefore measures one model state against a different environment than every other state of
    the same arm, and nothing downstream can detect it: the folder is complete, the CSVs are valid,
    and the trajectory simply moves.

    This is the patient-side twin of :func:`_check_state_matches_adapter`, and it is enforced under
    the same rule: only the CANONICAL tree is protected, because ``--conv-dir`` is the caller
    saying the output is not the arm's own state folder.

    Note that the grammar encodes the patient MODEL, not the provider, so a same-model swap across
    providers (local vLLM vs the vendor API) passes this check -- the arm name cannot express it.
    Record it in the pass's note if it matters.
    """
    if arm is None or binding.tag == arm.patient_tag:
        return

    message = (f"--exp-name names patient {arm.patient_tag!r} but this pass would generate against "
               f"{binding.model!r} [{binding.provider}] (tag {binding.tag!r}), so the "
               f"conversations would be filed, scored and plotted as the arm's own data.")
    fix = ("Drop the --patient-model/--patient-provider override, or use the --exp-name whose "
           "Pat<tag> matches the patient you want.")

    if canonical:
        raise SystemExit(f"REFUSING TO START. {message}\n  {fix}\n  (Or pass --conv-dir to write "
                         f"outside the arm's own conversations tree, where this is your call.)")
    print(f"  !! {message}\n  !! Allowed because --conv-dir puts the output outside the arm's "
          f"tree. {fix}")


def _check_arm_name(exp_name: str, *, canonical: bool) -> Optional[Any]:
    """Decode *exp_name*; a non-arm name is fatal only when writing into the canonical tree.

    A folder under ``data/conversations/`` whose name does not parse is invisible to arm discovery,
    so the conversations exist and are never scored. Outside that tree the name is just a label.
    """
    from naming import parse_experiment_name

    try:
        return parse_experiment_name(exp_name)
    except ValueError as exc:
        if canonical:
            raise SystemExit(f"REFUSING TO START. {exc}\n  A conversations folder whose name does "
                             f"not parse is invisible to arm discovery: the pass would succeed and "
                             f"the data would never be scored.") from exc
        print(f"  !! --exp-name does not parse as an arm name ({exc}). Allowed because --conv-dir "
              f"puts the output outside the discovered tree.")
        return None


# =============================================================================
#  main
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one generate-only pass. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    _bootstrap_sys_path()

    from core.runtime import assert_import_order, authenticate, detect_host, mount_drive_if_colab

    host = detect_host()
    if host == "colab":
        mount_drive_if_colab()
    assert_import_order()

    # Everything up to the VRAM guard stays torch-free. core.config / naming / roles /
    # core.conversations are stdlib-side (conversations imports torch lazily, inside the functions
    # that generate), so a --dry-run and every refusal happen without a CUDA-capable import having
    # run at all. core.policy is imported below, after the guard has passed.
    from core.config import RunPaths
    from naming import model_state_label
    from system_prompts_builder import generate_all_permutations

    canonical = args.conv_dir is None
    label = model_state_label(args.model_iter)
    rule = "=" * 78
    print(rule)
    print(f"GENERATE-ONLY PASS  {args.exp_name}  {label}")
    print(rule)

    # The guards run before anything is measured or resolved, so their warnings sit under the
    # banner rather than ahead of it, and a refusal costs nothing but the argument parse.
    arm = _check_arm_name(args.exp_name, canonical=canonical)
    _check_state_matches_adapter(args.model_iter, args.adapter, canonical=canonical)
    check_adapter_path(args.adapter)

    paths = RunPaths.from_workspace(args.exp_name, data_root=args.data_root)
    config = load_run_defaults(paths.run_metadata_path)
    training = config.get("training") if isinstance(config.get("training"), Mapping) else {}

    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE[host]
    seed = args.seed if args.seed is not None else int(training.get("seed", 42))
    gen = build_gen_config(config, batch_size=batch_size, n_convs=args.n_convs,
                           max_utterances=args.max_utterances)

    conv_dir = args.conv_dir or paths.conv_dir_for(args.model_iter)

    permutations = generate_all_permutations(only_expert_therapist=True)
    therapist_system_prompt, therapist_init_utterance = therapist_prompt_pair(permutations)
    n_requested = min(gen.num_conversations_per_iter, len(permutations))
    if gen.num_conversations_per_iter > len(permutations):
        print(f"  !! --n-convs {gen.num_conversations_per_iter} exceeds the {len(permutations)} "
              f"personas (one conversation each); generating {n_requested}.")
    persona_ids = select_personas(len(permutations), n_requested, seed)

    existing = _existing_persona_ids(conv_dir)
    todo = [pid for pid in persona_ids if pid not in existing]
    n_batches = math.ceil(len(todo) / batch_size) if todo else 0

    binding = resolve_patient_binding(config, model=args.patient_model,
                                      provider=args.patient_provider, base_url=args.base_url)
    _check_patient_matches_arm(binding, arm, canonical=canonical)
    binding = resolve_patient_endpoint(binding, fatal=not args.dry_run)

    budget = plan_vram_budget(batch_size)

    # ---- the plan ------------------------------------------------------------
    print(f"  arm          {arm.label if arm is not None else '(unparsed)'}"
          f"    host {host}"
          f"    metadata {'yes' if config else 'no (using defaults)'}")
    print(f"  policy       {training.get('base_model_id') or _default_base_model_id()}"
          f"  + {args.adapter or 'NO ADAPTER (base policy)'}")
    print(f"  patient      {binding.model}  [{binding.provider}]  "
          f"{binding.base_url or '(unresolved)'}")
    print(f"  output       {conv_dir}"
          f"{'' if canonical else '   [--conv-dir: OUTSIDE the arm tree]'}")
    print(f"  convs        {n_requested} x up to {gen.num_utterances_for_data} utterances"
          f"   T_ther {gen.temperature_therapist} / T_pat {gen.temperature_patient}"
          f"   max {gen.max_tokens_per_response} tok")
    print(f"  batching     {batch_size} in flight -> {n_batches} batch(es)"
          f"   seed {seed}   stop {list(gen.stop_strings)}")
    print(f"  resume       {len(existing)} conversation(s) already on disk"
          f"{' -> skipped' if existing else ''};  {len(todo)} to generate")
    print("  VRAM")
    print(budget.arithmetic())
    print("  No oracle calls, no branching, no look-ahead, no training.")

    if n_batches == 1:
        print("  !! ONE BATCH ONLY. The per-batch `vram <N>G` field is how an inter-batch "
              "empty_cache() regression is caught -- flat is healthy, climbing is not -- and a "
              "single batch cannot show a trend. Use >= 2 batches for a smoke test.")

    enforce_vram_budget(budget, host=host, force=args.force)

    if args.dry_run:
        print("\n  [dry-run] stopping before the model load. Nothing generated, nothing spent.")
        print(rule)
        return 0

    if not todo:
        print(f"\n  OK already complete: {len(existing)} conversation(s) in {conv_dir}")
        print(rule)
        return 0

    # ---- the pass ------------------------------------------------------------
    authenticate(hf=True, openai=(binding.provider == "openai"))

    from core.concurrency import AsyncPrimitives
    from core.conversations import generate_all_conversations
    from core.policy import vram_report
    from roles import make_client

    policy, tokenizer = load_policy(
        str(training.get("base_model_id") or _default_base_model_id()),
        str(training.get("tokenizer_id") or ""),
        args.adapter,
        use_4bit=bool(training.get("use_4bit", False)),
    )
    print(f"  policy resident: vram {vram_report()['reserved_gib']:.1f}G reserved "
          f"(estimate assumed {WEIGHTS_GIB:.1f}G for weights)")

    # At most `batch_size` patient calls are ever in flight -- the loop advances one lock-step batch
    # at a time -- so a larger bound would not bound anything, and a smaller one would throttle the
    # batch it is meant to serve.
    primitives = AsyncPrimitives(
        oracle_concurrency=_ORACLE_CONCURRENCY_UNUSED,
        patient_concurrency=max(1, min(gen.patient_concurrency, batch_size)),
    )

    started = time.time()
    states = generate_all_conversations(
        policy, tokenizer, make_client(binding), binding, primitives,
        permutations, therapist_system_prompt, therapist_init_utterance,
        persona_ids=persona_ids,
        save_dir=conv_dir,
        num_utterances=gen.num_utterances_for_data,
        max_tokens=gen.max_tokens_per_response,
        temperature_therapist=gen.temperature_therapist,
        temperature_patient=gen.temperature_patient,
        therapist_max_input_tokens=gen.therapist_max_input_tokens,
        stop_strings=list(gen.stop_strings) or None,
        patient_seed=seed,
        batch_size=batch_size,
        max_retries_without_progress=gen.max_retries_without_progress,
        verbose=gen.verbose,
        verbose_detailed=gen.verbose_detailed,
    )
    elapsed = time.time() - started

    _log_timing(paths, args.model_iter, elapsed, started, canonical=canonical, label=label)
    _report(states, conv_dir, elapsed, n_requested, canonical=canonical, rule=rule)
    return 0 if len(states) >= n_requested else 1


def _default_base_model_id() -> str:
    """The therapist base model, from :mod:`core.config` so there is one spelling of it."""
    from core.config import DEFAULT_BASE_MODEL_ID

    return DEFAULT_BASE_MODEL_ID


def _existing_persona_ids(conv_dir: str) -> set:
    """Persona ids already written to *conv_dir*.

    Read through ``core.conversations`` rather than by globbing here, so "what counts as a finished
    conversation on disk" has one definition -- the same one the resume inside
    ``generate_all_conversations`` uses. A file this cannot parse is not counted, and is therefore
    regenerated.
    """
    from core.conversations import load_conversations_dir

    return set(load_conversations_dir(conv_dir).keys())


def _log_timing(paths, model_iter: int, elapsed: float, started: float, *,
                canonical: bool, label: str) -> None:
    """Append this pass to the iteration's append-only timing log, where that is meaningful.

    Recorded as ``eval_gen_s`` against ``iteration_<model_iter>`` -- the iteration that produced the
    adapter this pass generated with. Skipped for the base state (there is no iteration_0) and for a
    ``--conv-dir`` run (a replicate is not part of the arm's cost). The iteration directory is
    never created here: if it does not exist, this pass is not repairing anything that arm recorded.

    Appending rather than overwriting is the point -- a repair adds its cost to the sessions the
    original attempt already logged, so the sum stays the true cost of the state.
    """
    if not canonical or model_iter < 1:
        return
    iter_dir = paths.iteration_dir(model_iter)
    if not os.path.isdir(iter_dir):
        return
    from core.timing import log_session

    log_session(iter_dir, eval_gen_s=elapsed, started_at=started,
                note=f"tools/generate_convs.py -> {label}")


def _report(states: Sequence[Any], conv_dir: str, elapsed: float, n_requested: int, *,
            canonical: bool, rule: str) -> None:
    """Print what the pass produced, including the session-end rate.

    The session-end count is not decoration. :data:`core.conversations.SESSION_END_KEYWORD` is the
    only early-termination channel in the experiment and both system prompts ask for it in PROSE, so
    a patient model that is not steered to the protocol simply never emits it: every conversation
    runs to the utterance cap, nothing raises, and the only visible symptom is this number being
    zero. Exp4 swaps the patient off gpt-4o-mini, which is exactly when that would first appear.
    """
    ended = sum(1 for s in states if getattr(s, "session_ended_by", ""))
    lengths = [s.n_utterances for s in states] or [0]
    print("\n" + rule)
    print(f"  Done in {elapsed / 60:.1f} min -- {len(states)}/{n_requested} conversation(s), "
          f"mean {sum(lengths) / len(lengths):.1f} utterances "
          f"(min {min(lengths)}, max {max(lengths)})")
    print(f"  Ended by SESSION ENDED: {ended}/{len(states)}; the rest reached the utterance cap.")
    if states and ended == 0:
        print("  !! NO conversation ended early. That is the signature of a patient model that "
              "ignores the SESSION ENDED protocol (it is requested in prose, not enforced by a "
              "stop token), not necessarily of long conversations. Check a transcript before "
              "scoring this state.")
    if len(states) < n_requested:
        print(f"  !! {n_requested - len(states)} conversation(s) missing. Re-run: generation "
              f"resumes per persona, so only the gaps are regenerated.")
    print(f"  {conv_dir}")
    if canonical:
        print("  Next: score this model state, then render the EDA families.")
    else:
        print("  This is an ISOLATED draw (--conv-dir): it is not in the arm's conversations tree, "
              "so arm discovery will not see it. Point the scorer at it explicitly.")
    print(rule)


if __name__ == "__main__":
    raise SystemExit(main())
