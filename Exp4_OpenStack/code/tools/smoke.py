"""smoke.py -- the offline gate between a config mistake and a wasted Colab session.

Everything expensive in Exp4 happens far from the mistake that caused it. A misspelled
questionnaire id becomes a folder nobody can parse eight hours later; a role binding whose
``base_url`` was never filled in becomes a 40-minute generation pass that dies on its first
oracle call; a batch size that is one conversation too large does not raise on the local
card, it **reboots the machine**. This file is where those are supposed to be caught, in
seconds, with no GPU and no network.

It is also the Phase 0-4 gate table in CLAUDE.md, made executable::

    python tools/smoke.py naming     # Phase 0: the arm-name grammar round-trips
    python tools/smoke.py config     # cell-1 globals freeze, and every validator fires
    python tools/smoke.py convs      # the transcript wire protocol and the MCL filter
    python tools/smoke.py vram       # the arithmetic, printed, before anything allocates
    python tools/smoke.py serve      # a real vLLM server comes up and answers
    python tools/smoke.py roles      # Phase 1: schema + NO THINKING TOKENS + kill/restart
    python tools/smoke.py stopgen    # GPU: stop_strings actually binds
    python tools/smoke.py dpo        # GPU: one DPO step, prompt capped, no OOM
    python tools/smoke.py grpo       # GPU: one GRPO step with a stub reward
    python tools/smoke.py all        # every part this host can run, one subprocess each

Three rules this file exists to enforce
---------------------------------------
**1. The VRAM guard is a safety feature, not a convenience.** On the local RTX 5070 Ti an
over-budget request is a GPU/driver fault that takes the OS down with it: no
``OutOfMemoryError``, no traceback, nothing to catch. So every subcommand that will allocate
does the arithmetic first, prints the sum it computed, and REFUSES rather than trying. A
refusal is reported as SKIP, never as a failure -- nothing was learned, but nothing was
risked either. ``--force`` downgrades the refusal to a warning and hands the arithmetic back
to the caller.

**2. ``trl`` is imported before ``torch``, by construction.** On the same local card
importing trl *after* torch segfaults at CUDA init -- exit 139, no traceback, easily mistaken
for OOM. The import sits at the very top of this module, ahead of every project import,
because several of those (``core.policy``, ``core.lookahead``, and ``core.config`` through its
lazy ``LookaheadConfig`` builder) pull torch in transitively. Every GPU subcommand also calls
:func:`core.runtime.assert_import_order`, which is the check that still means something on a
host where trl is not installed at all.

**3. SKIP is never a failure.** A missing GPU, a missing vLLM, a base model that is not in
the local HF cache, a VRAM refusal -- all of those are "this host cannot answer that
question", and the runner's exit code must not conflate them with "the answer was wrong".
Exit codes are ``0`` PASS, ``1`` FAIL, ``3`` SKIP, and ``all`` treats 3 as success.

Nothing here needs a paid API key. That is the whole premise of Exp4, and a smoke test that
quietly required one would be testing a configuration nobody runs.

Note on "NO GPU": ``config`` and ``convs`` do import torch (transitively, through
``core.lookahead`` / ``core.policy``). They never touch CUDA, never allocate, and never open a
socket. Importing torch is not the hazard; requesting memory is.
"""

from __future__ import annotations

import os
import sys

# huggingface_hub freezes HF_HUB_OFFLINE into a module constant at ITS import time, and that
# import happens inside the `import trl` below. A smoke test must never silently download 2.5 GB
# of gated therapist weights, so the flag is set from a pre-scan of argv -- the only point early
# enough to matter. `--allow-download` opts out, and the serve/roles parts are deliberately NOT
# covered: their vLLM subprocess inherits this environment and does need to fetch its model.
if sys.argv[1:2] and sys.argv[1] in ("stopgen", "dpo", "grpo") \
        and "--allow-download" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# trl FIRST -- ahead of every other project import in this file. See rule 2 in the module
# docstring. A missing trl is not an error: only the GPU subcommands need it, and they say so.
try:  # noqa: SIM105 - the fallback needs a comment, so contextlib.suppress would hide the reason
    import trl  # noqa: F401  (imported for its side effect on native init order)
except ImportError:
    trl = None  # type: ignore[assignment]

# datasets SECOND -- also ahead of torch, and for the same class of reason. MEASURED on the local
# RTX 5070 Ti (sm_120): `import torch, datasets` and `import trl, torch, datasets` both die with a
# Windows access violation (exit 139) inside `pyarrow.dataset`, while `import datasets, torch` and
# `import trl, datasets, torch` are fine. pyarrow and torch each load native runtimes, and the
# survivor is whichever initialises first.
#
# The trainers happen to be safe already -- they pull pandas (and therefore pyarrow) in through
# core.* before their own `import torch`. That is luck, not design: it survives only as long as
# nobody reorders those imports. This file has no such accident, which is exactly why `smoke.py
# dpo` segfaulted before reaching a single check. Importing it here makes the ordering explicit
# and gives `assert_import_order` something true to assert.
try:  # noqa: SIM105
    import datasets  # noqa: F401  (imported for its side effect on native init order)
except ImportError:
    datasets = None  # type: ignore[assignment]

import argparse
import json
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# `python tools/smoke.py` puts tools/ on sys.path, NOT code/, so the project imports below would
# not resolve when this file runs as a script. The trainer notebooks already prepend code/, where
# this is a no-op. It has to happen before the imports, hence the noqa markers.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from core.runtime import assert_import_order, detect_host                     # noqa: E402
from naming import (                                                          # noqa: E402
    GRAMMAR,
    PTO_MODES,
    QTAG_BY_IDS,
    ArmInfo,
    build_experiment_name,
    model_state_label,
    parse_experiment_name,
    parse_model_state_label,
    qtag_for,
)
from roles import (                                                           # noqa: E402
    DEFAULT_JUDGE_MODEL,
    DEFAULT_ORACLE_MODEL,
    DEFAULT_PATIENT_MODEL,
    RoleBinding,
    ServeSpec,
    make_binding,
    model_tag,
)

__all__ = [
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_SKIP",
    "EXIT_PASS",
    "EXIT_FAIL",
    "EXIT_SKIP",
    "PARTS",
    "GPU_PARTS",
    "SERVER_PARTS",
    "THINKING_MARKERS",
    "SMOKE_MODEL",
    "Section",
    "StubTokenizer",
    "VramPlan",
    "cmd_naming",
    "cmd_config",
    "cmd_convs",
    "cmd_vram",
    "cmd_serve",
    "cmd_roles",
    "cmd_stopgen",
    "cmd_dpo",
    "cmd_grpo",
    "run_part",
    "build_parser",
    "main",
]


# ==============================================================================
#                                 CONSTANTS
# ==============================================================================

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"

#: Process exit codes. 3 is a THIRD outcome on purpose: a caller that only tests "nonzero"
#: would otherwise treat "this host has no GPU" as "the DPO step is broken".
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 3

#: Subcommand order for ``all`` -- cheapest and most diagnostic first, so a broken grammar is
#: reported before a GPU part spends two minutes loading weights to fail for the same reason.
PARTS: Tuple[str, ...] = (
    "naming", "config", "convs", "vram", "serve", "roles", "stopgen", "dpo", "grpo",
)

#: Parts that allocate VRAM. Each one guards itself; this tuple is for the ``all`` summary.
GPU_PARTS: Tuple[str, ...] = ("stopgen", "dpo", "grpo")

#: Parts that need a server (their own, or one already listening).
SERVER_PARTS: Tuple[str, ...] = ("serve", "roles")

#: Substrings that betray a reasoning preamble leaking into ``message.content``.
#:
#: WARNING: this list is the ONLY thing standing between "the thinking-off switch works" and
#: "the thinking-off switch is a no-op". ``roles.thinking_off_extra_body`` sends
#: ``chat_template_kwargs={"enable_thinking": false}``, and vLLM passes that straight into the
#: Jinja render, where an unrecognised NAME is simply an unused variable -- the request
#: succeeds, the schema is usually still honoured, and every one of the ~10k oracle calls per
#: iteration quietly pays for reasoning tokens. Nothing else in the stack can see that. If a
#: model spells its thinking block some other way, add it here rather than trusting the flag.
THINKING_MARKERS: Tuple[str, ...] = (
    "<think>", "</think>", "<thinking>", "</thinking>",
    "<reasoning>", "</reasoning>", "<|channel|>", "<|thinking|>",
    "<tool_think>", "[THINK]", "[/THINK]",
)

#: Default model for ``serve``: small, ungated, and downloads in under a minute. Deliberately
#: NOT the Gemma the real stack serves -- this subcommand answers "does the launch/readiness/
#: adopt path work", which is model-independent, and pulling 3 GB to learn that is a waste.
SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

#: Port for ``serve``. Off the default 8000 so a smoke test can never be adopted as -- or
#: collide with -- the server a real arm is running on.
SMOKE_PORT = 8011

#: Therapist/patient system prompts for the offline text checks. Short stand-ins: nothing here
#: measures prompt quality, only that the wire formats round-trip.
_SYS_THERAPIST = (
    "You are a motivational interviewing counselor named David. You are empathetic and help "
    "the patient explore ambivalence about change."
)
_SYS_PATIENT = "You are a patient who is ambivalent about quitting smoking."

#: Smoke-scale VRAM estimates, in GiB, for the three GPU parts. These are NOT the arm's budget
#: (see CLAUDE.md's VRAM table for that) -- they are what a 1B bf16 policy plus a
#: two-row LoRA step is expected to request at the tiny hyperparameters below.
_GPU_NEED_GIB: Dict[str, float] = {"stopgen": 3.6, "dpo": 6.0, "grpo": 6.0}

# Tiny training shapes. Small enough that the step is about mechanics, not throughput.
_MAX_PROMPT_TOKENS = 128
_MAX_RESPONSE_TOKENS = 32


# ==============================================================================
#                          REPORTING: PASS / FAIL / SKIP
# ==============================================================================


class _Skip(Exception):
    """Raised inside a subcommand to end it as SKIP with a reason.

    Used for "this host cannot answer the question" (no CUDA, no vLLM, base model absent) and
    for a VRAM refusal. Never for a wrong answer -- that is a failed check.
    """


@dataclass
class Section:
    """One subcommand's running result: what was checked, what failed, what was skipped.

    Checks are printed as they run so a subcommand that dies mid-way still shows how far it
    got, and are also accumulated so the closing verdict can count them.

    Attributes:
        name: The subcommand name, echoed in the verdict line.
        passed: Number of checks that held.
        failures: One string per failed check, used in the verdict line.
        skipped: Reason string when the whole section was skipped, else ``None``.
    """

    name: str
    passed: int = 0
    failures: List[str] = field(default_factory=list)
    skipped: Optional[str] = None

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        """Record and print one check. Returns *ok*, so callers can branch on it.

        Args:
            ok: Whether the property held.
            label: What was checked, in the present tense ("every name round-trips").
            detail: Evidence -- the counts, the offending value, the computed number. Printed
                for a pass as well as a failure: a check that only shows its numbers when it
                breaks is a check nobody can sanity-read.
        """
        suffix = f"  {detail}" if detail else ""
        if ok:
            self.passed += 1
            print(f"  [ ok ] {label}{suffix}")
        else:
            self.failures.append(f"{label}{(' -- ' + detail) if detail else ''}")
            print(f"  [FAIL] {label}{suffix}")
        return ok

    def note(self, text: str) -> None:
        """Print an informational line that is not a check (a computed budget, a sample name)."""
        print(f"         {text}")

    def hard_fail(self, text: str) -> None:
        """Record an exception that ended the section, as one failure.

        The whole traceback is printed once; only its last line goes into the failure list, so
        the closing verdict stays readable when several parts died the same way.
        """
        print(f"  [FAIL] {self.name} raised:\n{text}")
        self.failures.append(text.strip().splitlines()[-1] if text.strip() else "raised")

    @property
    def status(self) -> str:
        """``PASS`` / ``FAIL`` / ``SKIP``.

        A skip is never a failure, but it never MASKS one either: a section that recorded a
        failed check and then hit a skip (a server that could not be restarted, say) reports
        FAIL. The alternative silently converts a real defect into "this host cannot run it".
        """
        if self.failures:
            return STATUS_FAIL
        return STATUS_SKIP if self.skipped is not None else STATUS_PASS

    @property
    def exit_code(self) -> int:
        """The process exit code this section implies."""
        return {STATUS_PASS: EXIT_PASS, STATUS_FAIL: EXIT_FAIL, STATUS_SKIP: EXIT_SKIP}[self.status]

    def verdict(self) -> str:
        """The closing line, e.g. ``PASS naming (7 checks)``."""
        if self.status == STATUS_SKIP:
            return f"{STATUS_SKIP} {self.name}: {self.skipped}"
        if self.status == STATUS_FAIL and self.skipped is not None:
            return (f"{STATUS_FAIL} {self.name} (skipped part-way: {self.skipped})\n"
                    + "\n".join(f"       - {f}" for f in self.failures))
        if self.status == STATUS_FAIL:
            head = f"{STATUS_FAIL} {self.name} ({len(self.failures)} of "
            head += f"{len(self.failures) + self.passed} checks failed)"
            return "\n".join([head] + [f"       - {f}" for f in self.failures])
        return f"{STATUS_PASS} {self.name} ({self.passed} checks)"


def run_part(name: str, fn: Callable[[Section, argparse.Namespace], None],
             args: argparse.Namespace) -> Section:
    """Run one subcommand body, converting :class:`_Skip` and any exception into a status.

    Args:
        name: Subcommand name.
        fn: The subcommand body; it takes ``(section, args)`` and records checks on the section.
        args: Parsed command-line arguments.

    Returns:
        The completed :class:`Section`.

    Notes:
        An unexpected exception is a FAIL with its traceback, never a crash: the runner has to
        keep its exit-code contract even when a part explodes, and a bare traceback on stderr
        would be indistinguishable from the harness itself being broken.
    """
    print(f"=== smoke: {name} ===")
    section = Section(name)
    try:
        fn(section, args)
    except _Skip as exc:
        section.skipped = str(exc)
    except Exception:  # noqa: BLE001 - a part must never take the runner down
        section.hard_fail(traceback.format_exc().rstrip())
    print(section.verdict())
    return section


def _expect_error(sec: Section, label: str, expected: str,
                  fn: Callable[[], Any], *,
                  exc_types: Tuple[type, ...] = (ValueError, TypeError)) -> None:
    """Check that *fn* raises, and that its message mentions *expected*.

    The message text is part of the check on purpose. Every validator in ``core.config`` and
    ``naming`` exists to tell a human what to change in cell 1, so an error that fires with the
    wrong explanation has only half worked -- and a test that accepts any exception would pass
    just as happily when the failure moved somewhere unrelated.
    """
    try:
        fn()
    except exc_types as exc:
        text = str(exc)
        sec.check(expected in text, label,
                  f"raised, message mentions {expected!r}" if expected in text
                  else f"raised, but message was: {text[:200]!r}")
    except Exception as exc:  # noqa: BLE001
        sec.check(False, label, f"raised {type(exc).__name__} (expected one of "
                                f"{[t.__name__ for t in exc_types]}): {exc}")
    else:
        sec.check(False, label, "did NOT raise")


# ==============================================================================
#                                   naming
# ==============================================================================
#
# Phase 0. The arm name is the only channel between the trainer and the EDA (see naming.py):
# the folder name IS the identity, so a name that does not round-trip is a run whose scores
# cannot be attributed, and two arms that render to the SAME name share a conversations folder
# and a score partition, which a resume-by-skipping-existing scorer reports as "already scored"
# against the other arm's numbers.

#: Every questionnaire set the grammar can name, derived from the table rather than listed, so
#: a new rubric token is covered by this gate the moment it becomes buildable.
_QUESTIONNAIRE_SETS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(sorted(ids)) for ids in sorted(QTAG_BY_IDS, key=lambda s: (len(s), sorted(s)))
)

_K_VALUES: Tuple[int, ...] = (0, 5)

#: (oracle, patient) model pairs: the all-open default, plus each role flipped to a vendor API,
#: plus a two-open-model pair. The point is that the role tags are ALWAYS encoded (unlike Exp3,
#: where a suffix appeared only for a non-default binding), so a flipped role must widen the
#: name rather than silently reusing the default arm's folder.
_ROLE_PAIRS: Tuple[Tuple[str, str], ...] = (
    (DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL),
    ("gpt-4o-mini-2024-07-18", DEFAULT_PATIENT_MODEL),
    (DEFAULT_ORACLE_MODEL, "gpt-4o"),
    ("google/gemma-4-E4B-it", DEFAULT_PATIENT_MODEL),
)

_NAME_CHARSET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")

#: The three worked examples in CLAUDE.md and naming.py's docstring. If the grammar drifts, the
#: documentation and the code disagree, and the documentation is what someone will grep for.
_DOCUMENTED_NAMES: Tuple[str, ...] = (
    "GRPO4_Q1Q2_LA5_MCL12_G8_Ogemma4E2B_Patgemma4E2B",
    "PTO4_Q1Q2_LA0_MCL12_M8_PTgreedy_Ogemma4E2B_Patgemma4E2B",
    "GRPO4_WAI_LA0_MCL12_G8_Ogpt4m_Patgemma4E2B",
)


def cmd_naming(sec: Section, args: argparse.Namespace) -> None:
    """Build the full arm grid, round-trip every name, and prove the negatives fire.

    No GPU, no network, no filesystem. This is the Phase 0 gate.
    """
    sec.note(f"grammar: {GRAMMAR}")

    built: Dict[str, ArmInfo] = {}
    n_built = 0
    bad_chars: List[str] = []
    roundtrip_failures: List[str] = []
    idempotence_failures: List[str] = []
    collisions: List[str] = []

    for method in ("GRPO", "PTO"):
        modes: Tuple[Optional[str], ...] = (None,) if method == "GRPO" else PTO_MODES
        for ids in _QUESTIONNAIRE_SETS:
            for k in _K_VALUES:
                for oracle_model, patient_model in _ROLE_PAIRS:
                    for mode in modes:
                        name = build_experiment_name(
                            method, ids, k, 12,
                            g=8 if method == "GRPO" else None,
                            m=None if method == "GRPO" else 8,
                            mode=mode,
                            oracle_model=oracle_model,
                            patient_model=patient_model,
                        )
                        n_built += 1

                        if set(name) - _NAME_CHARSET:
                            bad_chars.append(name)

                        try:
                            arm = parse_experiment_name(name)
                        except ValueError as exc:
                            roundtrip_failures.append(f"{name}: {exc}")
                            continue

                        expected = (method, qtag_for(ids), k, 12,
                                    model_tag(oracle_model), model_tag(patient_model))
                        actual = (arm.method, arm.qtag, arm.k, arm.mcl,
                                  arm.oracle_tag, arm.patient_tag)
                        if actual != expected:
                            roundtrip_failures.append(f"{name}: {actual} != {expected}")
                        if method == "GRPO" and (arm.g != 8 or arm.m is not None
                                                 or arm.mode is not None):
                            roundtrip_failures.append(f"{name}: GRPO fields wrong ({arm})")
                        if method == "PTO" and (arm.m != 8 or arm.g is not None
                                                or arm.mode != mode):
                            roundtrip_failures.append(f"{name}: PTO fields wrong ({arm})")

                        if arm.experiment_name != name:
                            idempotence_failures.append(f"{name} -> {arm.experiment_name}")
                        if name in built:
                            collisions.append(name)
                        built[name] = arm

    sec.note(f"grid: 2 methods x {len(_QUESTIONNAIRE_SETS)} rubric sets x {len(_K_VALUES)} K "
             f"x {len(_ROLE_PAIRS)} role pairs (+{len(PTO_MODES)} PTO modes) = {n_built} names")
    for sample in list(built)[:2] + list(built)[-1:]:
        sec.note(f"  {sample}")

    sec.check(not bad_chars, "every name is [A-Za-z0-9_] only (legal NTFS + TensorBoard dir)",
              f"{n_built} names" if not bad_chars else f"offenders: {bad_chars[:3]}")
    sec.check(not roundtrip_failures,
              "parse_experiment_name recovers every field it was built from",
              f"{n_built} names" if not roundtrip_failures else f"{roundtrip_failures[:2]}")
    sec.check(not idempotence_failures, "ArmInfo.experiment_name reproduces the name it parsed",
              f"{n_built} names" if not idempotence_failures else f"{idempotence_failures[:2]}")
    sec.check(not collisions and len(built) == n_built,
              "no two arms in the grid render to the same folder name",
              f"{len(built)} distinct of {n_built}")

    # The 'independent' spelling is what the trainer's PREF_TREE_MODE global says; 'indep' is
    # what a folder is called. They must be the same arm, not two.
    alias = build_experiment_name("PTO", (1, 2), 5, 12, m=8, mode="independent",
                                  oracle_model=DEFAULT_ORACLE_MODEL,
                                  patient_model=DEFAULT_PATIENT_MODEL)
    short = build_experiment_name("PTO", (1, 2), 5, 12, m=8, mode="indep",
                                  oracle_model=DEFAULT_ORACLE_MODEL,
                                  patient_model=DEFAULT_PATIENT_MODEL)
    sec.check(alias == short, "PREF_TREE_MODE='independent' and 'indep' name the same arm", alias)

    for documented in _DOCUMENTED_NAMES:
        arm = parse_experiment_name(documented)
        sec.check(arm.experiment_name == documented,
                  "documented example round-trips", documented)

    # The many-to-one tag map is deliberate (a tag identifies a model FAMILY; the exact snapshot
    # lives in run_metadata.json). Checked so that "deliberate" stays visible.
    sec.check(model_tag("gpt-4o-mini") == model_tag("gpt-4o-mini-2024-07-18") == "gpt4m",
              "model_tag is many-to-one by design (family, not snapshot)", "both -> gpt4m")
    sec.check(all(set(model_tag(m)) <= _NAME_CHARSET - {"_"}
                  for m in ("google/gemma-4-E2B-it", "meta-llama/Llama-3.2-1B",
                            "some.vendor/weird_model.v2-it")),
              "model_tag never emits '_' or '.' (the field delimiter survives)")

    sec.check(model_state_label(0) == "model_iter_0" and model_state_label(7) == "model_iter_7",
              "model_state_label spells the conversations folder")
    sec.check(parse_model_state_label(model_state_label(4)) == 4,
              "parse_model_state_label inverts it")

    # --- negatives: every one of these was a real way to lose an arm -----------------------
    _expect_error(sec, "a truncated name is rejected, not silently skipped", "not an Exp4 arm name",
                  lambda: parse_experiment_name("GRPO4_Q1Q2_LA5_MCL12_G8"))
    # The regex is shape-only (it would match a PTO name carrying G8); the cross-field rule that
    # rejects it lives in ArmInfo, which every construction path goes through.
    _expect_error(sec, "a PTO name carrying a GRPO group size is rejected",
                  "PTO arms need m",
                  lambda: parse_experiment_name(
                      "PTO4_Q1Q2_LA5_MCL12_G8_Ogemma4E2B_Patgemma4E2B"))
    _expect_error(sec, "a GRPO name carrying a preference tree is rejected",
                  "GRPO arms need g",
                  lambda: parse_experiment_name(
                      "GRPO4_Q1Q2_LA5_MCL12_M8_PTgreedy_Ogemma4E2B_Patgemma4E2B"))
    _expect_error(sec, "building GRPO with PTO branch args and no G is rejected",
                  "GRPO arms need g",
                  lambda: build_experiment_name("GRPO", (1, 2), 5, 12, m=8, mode="greedy",
                                                oracle_model=DEFAULT_ORACLE_MODEL,
                                                patient_model=DEFAULT_PATIENT_MODEL))
    _expect_error(sec, "GRPO cannot carry a preference tree even alongside a valid G",
                  "carry no preference tree",
                  lambda: build_experiment_name("GRPO", (1, 2), 5, 12, g=8, m=8, mode="greedy",
                                                oracle_model=DEFAULT_ORACLE_MODEL,
                                                patient_model=DEFAULT_PATIENT_MODEL))
    _expect_error(sec, "PTO cannot carry a group size even alongside a valid M",
                  "carry no group size",
                  lambda: build_experiment_name("PTO", (1, 2), 5, 12, g=8, m=8, mode="greedy",
                                                oracle_model=DEFAULT_ORACLE_MODEL,
                                                patient_model=DEFAULT_PATIENT_MODEL))
    _expect_error(sec, "an unmapped questionnaire set cannot be written to disk",
                  "no name token for questionnaire set",
                  lambda: qtag_for([1, 2, 3]))
    _expect_error(sec, "an empty questionnaire set is rejected", "at least one rubric",
                  lambda: qtag_for([]))
    _expect_error(sec, "a negative look-ahead K is rejected", "non-negative",
                  lambda: build_experiment_name("GRPO", (1, 2), -1, 12, g=8,
                                                oracle_model=DEFAULT_ORACLE_MODEL,
                                                patient_model=DEFAULT_PATIENT_MODEL))
    _expect_error(sec, "k=True is a caller mistake, not a depth", "non-negative",
                  lambda: ArmInfo(method="GRPO", qtag="Q1Q2", k=True, mcl=12, g=8, m=None,
                                  mode=None, oracle_tag="gemma4E2B", patient_tag="gemma4E2B"))
    _expect_error(sec, "an Exp3-style temperature suffix on a model state is rejected",
                  "is not a model-state folder",
                  lambda: parse_model_state_label("model_iter_0_TT0.9_TP0.7"))
    _expect_error(sec, "a negative model state is rejected", "must be >= 0",
                  lambda: model_state_label(-1))


# ==============================================================================
#                                   config
# ==============================================================================
#
# Cell 1 is the only place a human types a number, and EXPERIMENT_NAME is COMPUTED from those
# numbers rather than typed alongside them. This section proves both halves: that a realistic
# globals dict freezes into the name CLAUDE.md documents, and that each validator fires on the
# mistake it was written for. The broken dicts are the point -- a validator nobody has seen
# fire is a validator nobody knows is still wired up.


def _smoke_bindings(*, oracle_url: Optional[str] = "http://127.0.0.1:8000/v1",
                    patient_url: Optional[str] = "http://127.0.0.1:8000/v1",
                    judge_url: Optional[str] = "http://127.0.0.1:8000/v1",
                    oracle: Optional[RoleBinding] = None) -> Dict[str, RoleBinding]:
    """The ``{role: RoleBinding}`` table ``serve_roles`` would have returned.

    ``base_url`` is filled in, because that is the state cell 1 is in by the time a config is
    built (notebook cell 3 runs first). Passing ``oracle_url=None`` simulates the mistake of
    building the config BEFORE the serve cell, which is one of the broken cases below.
    """
    return {
        "oracle": oracle or make_binding("openai_compat", DEFAULT_ORACLE_MODEL,
                                         base_url=oracle_url, request_timeout=120.0,
                                         max_retries=3),
        "patient": make_binding("openai_compat", DEFAULT_PATIENT_MODEL, base_url=patient_url,
                                request_timeout=90.0, max_retries=8),
        "judge": make_binding("openai_compat", DEFAULT_JUDGE_MODEL, base_url=judge_url,
                              request_timeout=120.0, max_retries=3),
    }


def _grpo_globals(data_root: str, **overrides: Any) -> Dict[str, Any]:
    """A realistic GRPO cell-1 namespace. The defaults ARE the matched grid from CLAUDE.md."""
    values: Dict[str, Any] = {
        "ROLE_BINDINGS": _smoke_bindings(),
        "QUESTIONNAIRE_IDS": [1, 2],
        "LOOKAHEAD_K": 5,
        "LOOKAHEAD_SUB_BATCH_SIZE": 64,
        "MIN_CONV_LENGTH": 12,
        "NUM_CONVERSATIONS_PER_ITER": 96,
        "NUM_UTTERANCES_FOR_DATA": 49,
        "CONVERSATION_BATCH_SIZE": 64,
        "NUM_ITERATIONS": 6,
        "EPOCHS_PER_ITERATION": 2,
        "NUM_GENERATIONS": 8,
        "TRAIN_BATCH_SIZE": 64,
        "EVAL_BATCH_SIZE": 64,
        "GRADIENT_ACCUMULATION_STEPS": 2,
        "LEARNING_RATE": 1e-5,
        "SEED": 42,
        "DATA_ROOT": data_root,
    }
    values.update(overrides)
    return values


def _pto_globals(data_root: str, **overrides: Any) -> Dict[str, Any]:
    """A realistic PTO cell-1 namespace (K=0 arm, greedy trees)."""
    values: Dict[str, Any] = {
        "ROLE_BINDINGS": _smoke_bindings(),
        "QUESTIONNAIRE_IDS": [1, 2],
        "LOOKAHEAD_K": 0,
        "MIN_CONV_LENGTH": 12,
        "NUM_CONVERSATIONS_PER_ITER": 96,
        "NUM_UTTERANCES_FOR_DATA": 49,
        "CONVERSATION_BATCH_SIZE": 64,
        "NUM_ITERATIONS": 8,
        "EPOCHS_PER_ITERATION": 2,
        "PREF_TREE_MODE": "greedy",
        "NUM_BRANCHES_PER_TURN": 8,
        "PREF_FILTER_TAU": 0.1,
        "BRANCH_SAMPLE_TEMPERATURE": 1.2,
        "DPO_BETA": 0.1,
        "TRAIN_BATCH_SIZE": 2,
        "EVAL_BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 8,
        "LEARNING_RATE": 1e-5,
        "SEED": 42,
        "DATA_ROOT": data_root,
    }
    values.update(overrides)
    return values


def cmd_config(sec: Section, args: argparse.Namespace) -> None:
    """Freeze both methods' globals, print the computed names, and fire every validator.

    No GPU (it imports torch transitively, through ``core.lookahead``), no network. All
    filesystem work happens under a temporary ``DATA_ROOT``.
    """
    from core.config import (
        GenConfig,
        GRPOTrainingConfig,
        RolesConfig,
        RunPaths,
        build_grpo_config,
        build_pto_config,
        config_to_metadata,
        validate_config,
        write_run_metadata,
    )

    with tempfile.TemporaryDirectory(prefix="exp4_smoke_cfg_") as tmp:
        data_root = os.path.join(tmp, "data")

        # --- the happy path -----------------------------------------------------------
        grpo = build_grpo_config(_grpo_globals(data_root), verbose=False)
        g_train, g_roles, g_gen, g_oracle, g_la, g_paths = grpo
        sec.note(f"GRPO EXPERIMENT_NAME = {g_train.experiment_name}")
        sec.check(g_train.experiment_name == _DOCUMENTED_NAMES[0],
                  "GRPO cell-1 globals compute the documented arm name",
                  g_train.experiment_name)
        sec.check(g_train.prompts_per_step == 16,
                  "GRPO batch arithmetic gives 16 unique prompts/step (the Phase 3 gate number)",
                  f"{g_train.train_batch_size} x {g_train.gradient_accumulation_steps} = "
                  f"{g_train.generation_batch_size} completions / G={g_train.num_generations}")

        pto = build_pto_config(_pto_globals(data_root), verbose=False)
        p_train, p_roles, p_gen, p_oracle, p_la, p_paths = pto
        sec.note(f"PTO  EXPERIMENT_NAME = {p_train.experiment_name}")
        sec.check(p_train.experiment_name == _DOCUMENTED_NAMES[1],
                  "PTO cell-1 globals compute the documented arm name", p_train.experiment_name)
        sec.check(p_train.pairs_per_step == 16,
                  "PTO pairs/step matches GRPO's prompts/step (the matched-grid claim)",
                  f"{p_train.train_batch_size} x {p_train.gradient_accumulation_steps} = 16")

        sec.check(g_paths.experiment_name == g_train.experiment_name
                  and g_paths.conv_dir_for(0).endswith(os.path.join(g_train.experiment_name,
                                                                    "model_iter_0")),
                  "RunPaths derives every path from the computed name",
                  g_paths.conversation_csv_path(0, 7))

        # The metadata file is the ONLY record of knobs the folder name does not encode, so a
        # section that stops being serialised has to be loud. config_to_metadata asserts that
        # itself; this proves the assertion is reachable and that the file lands.
        payload = config_to_metadata(*grpo)
        knobs = payload["config"]
        sec.check(knobs["lookahead"]["sub_batch_size"] == 64
                  and knobs["lookahead"]["k"] == 5
                  and knobs["training"]["num_iterations"] == 6,
                  "run_metadata carries the silently-mutable knobs (K, sub-batch, iterations)",
                  f"{len(payload['silently_mutable_knobs'])} knob paths asserted")
        written = write_run_metadata(payload, g_paths)
        write_run_metadata(payload, g_paths)
        with open(g_paths.run_metadata_history_path, encoding="utf-8") as fh:
            history_lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        sec.check(os.path.exists(written) and len(history_lines) == 2,
                  "run_metadata.json is overwritten while the history log APPENDS (Exp3 fix #5)",
                  f"{len(history_lines)} history lines after 2 processes")
        sec.check(json.loads(history_lines[0])["experiment_name"] == g_train.experiment_name,
                  "each history line is a complete, parseable payload")

        # --- broken cell-1 dicts: each validator, fired -------------------------------
        broken_grpo: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
            ("config built BEFORE serve_roles (local role, no base_url)",
             {"ROLE_BINDINGS": _smoke_bindings(oracle_url=None)}, "base_url is unset"),
            ("train_batch_size not divisible by num_generations",
             {"TRAIN_BATCH_SIZE": 60}, "must be divisible by"),
            ("num_generations = 1 (a group of one has no advantage)",
             {"NUM_GENERATIONS": 1}, "num_generations must be >= 2"),
            ("MCL longer than the conversations (every slice filtered out)",
             {"MIN_CONV_LENGTH": 60}, "exceeds num_utterances_for_data"),
            ("eval_split_ratio outside (0, 1)",
             {"EVAL_SPLIT_RATIO": 1.5}, "eval_split_ratio"),
            ("an unmapped questionnaire set",
             {"QUESTIONNAIRE_IDS": [1, 2, 3]}, "no name token for questionnaire set"),
            ("a Claude grader bound as the TRAINING oracle",
             {"ROLE_BINDINGS": _smoke_bindings(
                 oracle=make_binding("anthropic", "claude-haiku-4-5"))},
             "cannot serve as a training oracle"),
        )
        for label, override, expected in broken_grpo:
            _expect_error(sec, f"GRPO: {label}", expected,
                          lambda o=override: build_grpo_config(_grpo_globals(data_root, **o),
                                                               verbose=False))

        broken_pto: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
            ("greedy trees with an ODD MCL (the seed must end on a patient turn)",
             {"MIN_CONV_LENGTH": 11}, "EVEN min_conv_length"),
            ("a negative preference filter tau",
             {"PREF_FILTER_TAU": -0.5}, "pref_filter_tau"),
            ("a misspelled PREF_TREE_MODE",
             {"PREF_TREE_MODE": "greddy"}, "must be one of"),
            ("M = 1 (a pair needs a best AND a worst)",
             {"NUM_BRANCHES_PER_TURN": 1}, "must be >= 2"),
            ("a greedy trunk target that never grows past MCL",
             {"GREEDY_TRUNK_TARGET_LEN": 10}, "must exceed min_conv_length"),
        )
        for label, override, expected in broken_pto:
            _expect_error(sec, f"PTO: {label}", expected,
                          lambda o=override: build_pto_config(_pto_globals(data_root, **o),
                                                              verbose=False))

        # --- hand-assembled bundles: the two rules the builders cannot break ----------
        wrong_paths = RunPaths(data_root=data_root, experiment_name=_DOCUMENTED_NAMES[2])
        _expect_error(sec, "artifacts pointed at another arm's folder are refused",
                      "another arm's folder",
                      lambda: validate_config(g_train, g_roles, g_gen, g_oracle, g_la,
                                              wrong_paths))

        import dataclasses as _dc

        other_oracle = _dc.replace(
            g_oracle, binding=make_binding("openai", "gpt-4o-mini-2024-07-18"))
        _expect_error(sec, "an OracleConfig whose grader is not the arm's is refused",
                      "would not be the one that scored",
                      lambda: validate_config(g_train, g_roles, g_gen, other_oracle, g_la,
                                              g_paths))

        mismatched_ids = _dc.replace(g_train, questionnaire_ids=(3,))
        _expect_error(sec, "training and oracle rubric sets must agree",
                      "would describe different rubrics",
                      lambda: validate_config(mismatched_ids, g_roles, g_gen, g_oracle, g_la,
                                              g_paths))

        # A WARNING is not an error: a collapsed gas is a legitimate (bad) choice somebody may
        # make deliberately, and it does not change EXPERIMENT_NAME -- which is exactly why the
        # metadata file above has to carry it.
        warned = build_grpo_config(_grpo_globals(data_root, GRADIENT_ACCUMULATION_STEPS=1),
                                   verbose=False)
        sec.check(warned[0].experiment_name == g_train.experiment_name,
                  "a silently-mutable knob warns but still builds, under the SAME arm name",
                  "gradient_accumulation_steps 2 -> 1")

        summary_ok = isinstance(RolesConfig.from_bindings(_smoke_bindings()), RolesConfig)
        sec.check(summary_ok, "RolesConfig.from_bindings accepts the serve_roles table")
        sec.check(isinstance(g_gen, GenConfig) and isinstance(g_train, GRPOTrainingConfig),
                  "the builder returns the contracted bundle types")


# ==============================================================================
#                                    convs
# ==============================================================================
#
# The transcript is a WIRE PROTOCOL, not a pretty-printer: look-ahead recovers its own tail by
# `extended[len(seed):]`, so a changed label or joiner does not raise -- it silently produces
# empty or misaligned tails and every recorded look-ahead becomes wrong while every score stays
# plausible. These checks are the only thing that notices.


class StubTokenizer:
    """A whitespace tokenizer with a ChatML-shaped template, for offline prompt checks.

    The real therapist tokenizer is ``meta-llama/Llama-3.2-1B``'s, which is gated and may not be
    in the local cache -- and downloading it would make the Phase 0 gate need a network. Only
    two properties of a tokenizer matter to ``core.conversations``' prompt path: ``encode``
    returns something whose ``len`` is a token count, and ``apply_chat_template`` renders
    messages to a string. Budgets are therefore counted in WORDS here, which is fine: the checks
    are about which turns survive truncation, not about a real token budget.
    """

    def __init__(self, start: str = "<|im_start|>", end: str = "<|im_end|>") -> None:
        self.start = start
        self.end = end

    def encode(self, text: str, add_special_tokens: bool = False) -> List[str]:
        """Whitespace 'tokens'. Only ``len()`` of the result is ever used."""
        return str(text).split()

    def apply_chat_template(self, messages: Sequence[Dict[str, str]], *,
                            add_generation_prompt: bool = False,
                            tokenize: bool = False) -> str:
        """Render messages the way ``core.policy.CHATML_TEMPLATE`` does."""
        parts = [f"{self.start}{m['role']}\n{m['content']}{self.end}\n" for m in messages]
        if add_generation_prompt:
            parts.append(f"{self.start}assistant\n")
        return "".join(parts)


def _demo_turns(n: int) -> List[Dict[str, str]]:
    """``n`` alternating turns starting with the therapist (who always speaks first)."""
    return [
        {"role": "therapist" if i % 2 == 0 else "patient",
         "content": f"Turn {i:02d} about what change would mean for you."}
        for i in range(n)
    ]


def cmd_convs(sec: Section, args: argparse.Namespace) -> None:
    """Transcript round-trip, CSV round-trip, SESSION ENDED, and the MCL filter.

    No GPU, no network. It imports torch transitively (``core.policy`` owns the ChatML markers
    and the stop strings) but never touches CUDA.
    """
    import pandas as pd

    from core.config import DEFAULT_STOP_STRINGS
    from core.conversations import (
        SESSION_END_KEYWORD,
        ConversationState,
        build_truncated_training_prompt,
        conversation_filename,
        extract_prompts_from_conversations,
        format_conversation_for_oracle,
        handle_session_end,
        load_conversation_csv,
        parse_transcript_to_messages,
        turns_to_messages,
        turns_to_patient_messages,
        save_conversation_csv,
    )
    from core.lookahead import check_transcript_format_agreement, seed_transcript
    from core.policy import CHATML_MARKERS, STOP_STRINGS, clean_completion

    # --- the transcript, including the continuation case ------------------------------
    multiline = "That sounds hard.\n\nWhat would be different if you did cut down?"
    turns = _demo_turns(4)
    turns[2] = {"role": "therapist", "content": multiline}

    transcript = format_conversation_for_oracle(turns)
    msgs_ther, msgs_pat = parse_transcript_to_messages(transcript, _SYS_THERAPIST, _SYS_PATIENT)

    sec.check(msgs_ther == turns_to_messages(turns, _SYS_THERAPIST),
              "transcript -> parse reproduces the therapist-perspective messages exactly",
              f"{len(msgs_ther)} messages, one turn containing a blank line")
    sec.check(msgs_pat == turns_to_patient_messages(turns, _SYS_PATIENT),
              "and the patient-perspective messages, with the roles flipped")
    sec.check(msgs_ther[3]["content"] == multiline,
              "an UNLABELLED segment is reattached as a continuation, not a new turn",
              repr(multiline[:32] + "..."))
    sec.check(format_conversation_for_oracle(msgs_ther) == transcript,
              "format(parse(t)) == t (the system message is dropped, not relabelled)")

    _expect_error(sec, "a transcript opening with an unlabelled segment is refused",
                  "no preceding role label",
                  lambda: parse_transcript_to_messages("who said this?\n\n[PATIENT]: hi",
                                                       _SYS_THERAPIST, _SYS_PATIENT))

    # --- the exact-slicing contract look-ahead depends on -----------------------------
    seed = seed_transcript(transcript, "What would be a first small step?")
    tail = "\n\n[PATIENT]: Maybe cutting the morning one."
    sec.check((seed + tail)[len(seed):] == tail,
              "look-ahead recovers its tail by exact string slicing", f"{len(tail)} chars")
    ok_fmt, detail = check_transcript_format_agreement()
    sec.check(ok_fmt, "core.lookahead and core.conversations agree on the transcript grammar",
              detail.splitlines()[0])

    # --- the anti-degeneracy stack ----------------------------------------------------
    sec.check(tuple(DEFAULT_STOP_STRINGS) == tuple(STOP_STRINGS),
              "core.config's stop-string copy still equals core.policy's original",
              f"{list(STOP_STRINGS)}")
    sec.check(clean_completion("Tell me more.<|im_end|><|im_start|>user\nI quit") == "Tell me more."
              and clean_completion("<|im_start|>") == "",
              "clean_completion cuts at the first ChatML marker; '' marks a degenerate turn",
              f"markers {list(CHATML_MARKERS)}")

    # --- SESSION ENDED ----------------------------------------------------------------
    ended_by, explanation, cleaned = handle_session_end(
        f"Thanks for coming in today. {SESSION_END_KEYWORD} the patient set a goal.", "therapist")
    sec.check(ended_by == "therapist" and cleaned.strip() == "Thanks for coming in today."
              and explanation.strip() == "the patient set a goal.",
              "handle_session_end keeps the utterance and splits off the explanation")
    lower_by, _, lower_cleaned = handle_session_end("Take care. session ended", "patient")
    sec.check(lower_by == "patient" and lower_cleaned.strip() == "Take care.",
              "the keyword match is case-insensitive but preserves the model's own casing")
    _expect_error(sec, "handle_session_end on a non-terminal utterance is a caller bug",
                  "not found",
                  lambda: handle_session_end("Tell me more.", "therapist"))

    # --- the CSV: named by the STABLE persona id (Exp3 fix #2) ------------------------
    with tempfile.TemporaryDirectory(prefix="exp4_smoke_convs_") as tmp:
        state = ConversationState(persona_id=7, turns=list(turns),
                                  session_ended_by="therapist",
                                  session_ended_explanation="goal set")
        path = save_conversation_csv(state, tmp)
        sec.check(os.path.basename(path) == conversation_filename(7) == "pers07.csv",
                  "a conversation is filed under its persona id, not a shuffled index",
                  os.path.basename(path))

        reloaded = load_conversation_csv(path)
        sec.check(reloaded.turns == state.turns and reloaded.persona_id == 7,
                  "CSV round-trip preserves every turn, in order, with its speaker")
        sec.check(reloaded.session_ended_by == "therapist"
                  and reloaded.session_ended_explanation == "goal set",
                  "and the conversation-level session-end scalars (as strings, never NaN)")
        sec.check(reloaded.conversation_id == "pers07",
                  "conversation_id points at the artifact it came from", reloaded.conversation_id)

        stripped_dir = os.path.join(tmp, "no_column")
        os.makedirs(stripped_dir, exist_ok=True)
        frame = pd.read_csv(path, keep_default_na=False)
        frame.drop(columns=["persona_id"]).to_csv(
            os.path.join(stripped_dir, "pers07.csv"), index=False)
        fallback = load_conversation_csv(os.path.join(stripped_dir, "pers07.csv"))
        sec.check(fallback.persona_id == 7,
                  "a CSV missing the persona_id column falls back to the file name")

    # --- prompt extraction under the MCL filter ---------------------------------------
    tokenizer = StubTokenizer()
    permutations = [{"patient_system_prompt": f"persona {i}"} for i in range(8)]
    states = [
        ConversationState(persona_id=1, turns=_demo_turns(12)),
        ConversationState(persona_id=2, turns=_demo_turns(12)),
    ]

    open_filter = extract_prompts_from_conversations(
        states, _SYS_THERAPIST, tokenizer, min_conv_length=2, max_prompt_tokens=10_000,
        permutations=permutations)
    sec.check(len(open_filter) == 12,
              "MCL=2 is a no-op: one sample after every patient turn",
              f"2 conversations x 6 patient turns = {len(open_filter)} samples")

    mcl_12 = extract_prompts_from_conversations(
        states, _SYS_THERAPIST, tokenizer, min_conv_length=12, max_prompt_tokens=10_000,
        permutations=permutations)
    sec.check(len(mcl_12) == 2,
              "MCL=12 keeps only slices whose conversation-so-far is >= 12 utterances",
              f"{len(mcl_12)} samples (the final patient turn of each conversation)")

    mcl_13 = extract_prompts_from_conversations(
        states, _SYS_THERAPIST, tokenizer, min_conv_length=13, max_prompt_tokens=10_000,
        permutations=permutations)
    sec.check(mcl_13 == [],
              "an MCL above the conversation length yields NO training rows (it does not clamp)")

    sample = mcl_12[0]
    sec.check(set(sample) == {"prompt", "transcript", "conversation_id", "persona_id",
                              "patient_system_prompt"},
              "each sample carries the five contracted keys", sorted(sample))
    sec.check(sample["transcript"].rstrip().split("\n")[-1].startswith("[PATIENT]:"),
              "every slice ends on a patient turn -- the point where the therapist speaks next")
    sec.check(sample["patient_system_prompt"] == "persona 1"
              and sample["conversation_id"] == "pers01",
              "the persona's system prompt travels with the sample (look-ahead needs it)")

    # --- the DPO prompt cap -----------------------------------------------------------
    long_turns = _demo_turns(30)
    full = tokenizer.apply_chat_template(turns_to_messages(long_turns, _SYS_THERAPIST),
                                         add_generation_prompt=True, tokenize=False)
    capped = build_truncated_training_prompt(long_turns, _SYS_THERAPIST, tokenizer,
                                             _MAX_PROMPT_TOKENS)
    n_full = len(tokenizer.encode(full))
    n_capped = len(tokenizer.encode(capped or ""))
    sec.check(capped is not None and n_capped <= _MAX_PROMPT_TOKENS < n_full,
              "an over-long trunk is capped by dropping the OLDEST turns",
              f"{n_full} -> {n_capped} 'tokens' (budget {_MAX_PROMPT_TOKENS})")
    sec.check(capped is not None and "counselor named David" in capped
              and long_turns[-1]["content"] in capped,
              "the system prompt and the most recent turn always survive truncation")
    sec.check(build_truncated_training_prompt(long_turns, _SYS_THERAPIST, tokenizer, 3) is None,
              "an impossible budget returns None (SKIP the pair) rather than a mangled prompt")
    _expect_error(sec, "Exp3's token-tail truncation mode is not available", "not supported",
                  lambda: build_truncated_training_prompt(long_turns, _SYS_THERAPIST, tokenizer,
                                                          _MAX_PROMPT_TOKENS,
                                                          truncation_mode="legacy"))


# ==============================================================================
#                                    vram
# ==============================================================================
#
# On the local card an over-budget request REBOOTS the machine: no OutOfMemoryError, no
# traceback, nothing to catch. So the arithmetic is printed before anything allocates, and the
# constants come from tools/generate_convs.py rather than being restated here -- a second copy
# of a measured safety constant is exactly how one of them ends up stale.


@dataclass(frozen=True)
class VramPlan:
    """One candidate configuration and the terms of its VRAM request.

    Attributes:
        label: What the plan is.
        parts: ``(term, gib)`` pairs. Kept as terms, not a total, so the printed line shows the
            arithmetic -- a composite quoted as one number never gets audited.
    """

    label: str
    parts: Tuple[Tuple[str, float], ...]

    @property
    def total_gib(self) -> float:
        """Sum of the terms."""
        return sum(gib for _, gib in self.parts)

    def arithmetic(self) -> str:
        """``2.6 GiB weights + 6 x 1.1 GiB per conversation = 9.2 GiB``."""
        terms = " + ".join(f"{gib:.1f} GiB {term}" for term, gib in self.parts)
        return f"{terms} = {self.total_gib:.1f} GiB requested"


def _candidate_plans(total_gib: Optional[float], weights: float, per_conv: float,
                     server_util: float) -> List[VramPlan]:
    """The plans worth printing for a card of *total_gib* (server plans need the card size)."""
    plans: List[VramPlan] = []
    if total_gib is not None:
        for util in (server_util, 0.20):
            plans.append(VramPlan(
                f"vLLM server only (gpu-memory-utilization {util:.2f})",
                ((f"pre-allocation ({util:.2f} x {total_gib:.1f} GiB card)", util * total_gib),)))
    for batch in (2, 4, 6, 32, 64):
        plans.append(VramPlan(
            f"conversation generation, batch {batch}, no server resident",
            (("therapist weights + CUDA context", weights),
             (f"per conversation ({batch} x {per_conv:.1f})", batch * per_conv))))
    if total_gib is not None:
        for batch in (2, 6):
            plans.append(VramPlan(
                f"vLLM server (util {server_util:.2f}) + generation batch {batch}",
                ((f"vLLM pre-allocation ({server_util:.2f} x {total_gib:.1f})",
                  server_util * total_gib),
                 ("therapist weights + CUDA context", weights),
                 (f"per conversation ({batch} x {per_conv:.1f})", batch * per_conv))))
    return plans


def cmd_vram(sec: Section, args: argparse.Namespace) -> None:
    """Print the VRAM arithmetic for a set of candidate plans and mark each SAFE or REFUSED.

    Allocates nothing and imports no CUDA context: the card is measured with ``nvidia-smi``,
    because a guard that has to allocate in order to run is not a guard.
    """
    from tools.generate_convs import (
        PER_CONV_GIB,
        SAFE_VRAM_FRACTION,
        WEIGHTS_GIB,
        estimate_batch_vram_gib,
        plan_vram_budget,
    )
    from tools.vllm_serve import estimate_vram_gib

    host = detect_host()
    probe = plan_vram_budget(0)                      # batch 0: measures the card, nothing else
    free_gib, total_gib = probe.free_gib, probe.total_gib
    budget = None if free_gib is None else free_gib * SAFE_VRAM_FRACTION

    if free_gib is None:
        sec.note("card: UNKNOWN (nvidia-smi unavailable) -- every plan below is REFUSED, "
                 "because a guard that cannot measure must fail CLOSED")
    else:
        sec.note(f"card: {total_gib:.1f} GiB total, {free_gib:.1f} GiB free "
                 f"(free, not total: a resident vLLM server never gives its reservation back)")
        sec.note(f"budget: {free_gib:.1f} x {SAFE_VRAM_FRACTION:.2f} safety = {budget:.1f} GiB")
    sec.note(f"host: {host} -- " + ("an over-budget request REBOOTS this machine, so a refusal "
                                    "here is a hard stop" if host == "local" else
                                    "an over-budget request raises a catchable OutOfMemoryError "
                                    "and the loop retries smaller"))

    plans = _candidate_plans(total_gib, WEIGHTS_GIB, PER_CONV_GIB, 0.25)
    n_safe = 0
    for plan in plans:
        safe = budget is not None and plan.total_gib <= budget
        n_safe += int(safe)
        verdict = "SAFE   " if safe else "REFUSED"
        print(f"  [{verdict}] {plan.label}")
        print(f"            {plan.arithmetic()}"
              + ("" if budget is None else
                 f"  {'<=' if safe else '>'} {budget:.1f} GiB budget"))
    sec.note(f"{n_safe} of {len(plans)} candidate plans are within budget on this card")

    # --- the arithmetic itself, which must hold on any host --------------------------
    sec.check(abs(estimate_batch_vram_gib(0) - WEIGHTS_GIB) < 1e-9,
              "batch 0 costs exactly the weights", f"{WEIGHTS_GIB:.1f} GiB")
    sizes = [estimate_batch_vram_gib(b) for b in (1, 2, 4, 8, 16, 32)]
    sec.check(all(b > a for a, b in zip(sizes, sizes[1:])),
              "the estimate is strictly increasing in batch size",
              " -> ".join(f"{s:.1f}" for s in sizes))
    sec.check(abs(estimate_batch_vram_gib(32) - (WEIGHTS_GIB + 32 * PER_CONV_GIB)) < 1e-9,
              "batch 32 is the measured reboot case, and the model says so",
              f"{WEIGHTS_GIB:.1f} + 32 x {PER_CONV_GIB:.1f} = "
              f"{estimate_batch_vram_gib(32):.1f} GiB")

    if budget is not None and total_gib is not None and total_gib < 16.0:
        sec.check(estimate_batch_vram_gib(32) > budget,
                  "on a 12 GiB card batch 32 is REFUSED (it rebooted this machine once)",
                  f"{estimate_batch_vram_gib(32):.1f} GiB > {budget:.1f} GiB budget")

    # Server pre-allocation, checked against an explicit card size so the result does not
    # depend on which machine runs the gate.
    spec = ServeSpec(model=DEFAULT_ORACLE_MODEL, gpu_memory_utilization=0.25)
    a100 = estimate_vram_gib(spec, total_gib=40.0)
    sec.check(abs(a100 - 10.0) < 1e-6,
              "gpu_memory_utilization is a PRE-ALLOCATION: 0.25 x 40 GiB = 10.0 GiB", f"{a100}")
    sec.check(spec.max_model_len == 16384,
              "the served context stays at 16384 (8192 drops 2.1% of Q2 prompts, and the "
              "longest conversations are the ones that would vanish)",
              f"max_model_len={spec.max_model_len}")
    trainer_gib = 29.0
    sec.note(f"reference (CLAUDE.md, Colab A100 40 GiB): {a100:.1f} server + "
             f"{trainer_gib:.1f} trainer = {a100 + trainer_gib:.1f} GiB, "
             f"headroom {40.0 - a100 - trainer_gib:.1f} GiB for two CUDA contexts")
    sec.check(a100 + trainer_gib < 40.0,
              "the documented Colab budget still leaves headroom")


# ==============================================================================
#                                    serve
# ==============================================================================


def _vram_refusal(sec: Section, *, need_gib: float, what: str, force: bool) -> None:
    """Print the arithmetic and REFUSE (as a SKIP) when *need_gib* does not fit.

    Args:
        need_gib: What the caller is about to request.
        what: Human name for the request, used in the refusal.
        force: The caller typed ``--force``; downgrade every refusal to a warning. They own it.

    Raises:
        _Skip: the request does not fit, or the card could not be measured at all.

    Notes:
        A refusal is a SKIP, not a FAIL. Nothing was learned, but nothing was risked -- and on
        the local card "risked" means the OS goes down without a traceback, so the guard also
        fails closed when ``nvidia-smi`` cannot answer. This is deliberately not the reasoning
        "it is only inference, so it is safe": the hazard is the size of the request.
    """
    from tools.generate_convs import SAFE_VRAM_FRACTION, plan_vram_budget

    probe = plan_vram_budget(0)
    free_gib = probe.free_gib
    budget = None if free_gib is None else free_gib * SAFE_VRAM_FRACTION

    if budget is None:
        sec.note(f"{what}: needs ~{need_gib:.1f} GiB; free VRAM UNKNOWN (no nvidia-smi)")
    else:
        sec.note(f"{what}: needs ~{need_gib:.1f} GiB against a {budget:.1f} GiB budget "
                 f"({free_gib:.1f} GiB free x {SAFE_VRAM_FRACTION:.2f})")

    if budget is not None and need_gib <= budget:
        return
    if force:
        sec.note("--force given: proceeding anyway, on the caller's arithmetic")
        return
    raise _Skip(
        f"REFUSED before allocating: {what} needs ~{need_gib:.1f} GiB but the budget is "
        + ("unmeasurable (nvidia-smi unavailable)" if budget is None else f"{budget:.1f} GiB")
        + ". On this machine an over-budget request is a driver fault that reboots the host, "
          "not an exception. Free the card (a resident vLLM server holds its pre-allocation "
          "for its whole life) or pass --force if you have done the arithmetic yourself."
    )


def cmd_serve(sec: Section, args: argparse.Namespace) -> None:
    """Start a small vLLM server, wait for readiness, hit ``/v1/models``, stop it.

    SKIPs when vLLM is not installed -- the serving path is a Colab/GPU concern and the rest of
    the gate must still run on a laptop without it.
    """
    import shutil

    from tools.vllm_serve import (
        adopt_if_running,
        base_url_for_port,
        detect_total_vram_gib,
        report_weights_gib,
        start_server,
    )

    if shutil.which(args.executable) is None:
        raise _Skip(f"{args.executable!r} is not on PATH (pip install vllm). The serving path "
                    f"only runs where a GPU does.")

    # A tiny model and a short context: this subcommand answers "does launch/readiness/adopt
    # work", which is model-independent. The oracle's 16384 floor does not apply -- nothing
    # here scores a rubric prompt.
    spec = ServeSpec(model=args.model or SMOKE_MODEL, port=args.port or SMOKE_PORT,
                     gpu_memory_utilization=args.gpu_memory_utilization or 0.15,
                     max_model_len=args.max_model_len or 4096)
    sec.note(f"spec: {spec.model} @ {spec.base_url} (util {spec.gpu_memory_utilization}, "
             f"max_model_len {spec.max_model_len})")

    # Adoption is checked BEFORE the VRAM arithmetic: a server already on this port has already
    # taken its reservation, so refusing on free VRAM would report the wrong problem.
    try:
        already = adopt_if_running(spec)
    except RuntimeError as exc:
        raise _Skip(f"port {spec.port} is serving a different model: {exc}") from exc
    if already is not None:
        sec.check(already.is_alive(), "a server already on the port was ADOPTED, not duplicated",
                  already.base_url)
        sec.check(already.process is None,
                  "an adopted handle owns no process, so stop() cannot kill someone else's server")
        raise _Skip(f"port {spec.port} already serves {already.model}; the launch path was not "
                    f"exercised. Stop that server or pass --port to test a launch.")

    total = detect_total_vram_gib()
    if total is not None:
        _vram_refusal(sec, need_gib=spec.gpu_memory_utilization * total,
                      what=f"vLLM pre-allocation for {spec.model}", force=args.force)

    handle = start_server(spec, log_dir=tempfile.gettempdir(), timeout=args.timeout,
                          executable=args.executable)
    try:
        sec.check(handle.is_alive(), "the server answers GET /v1/models", handle.base_url)
        sec.check(handle.base_url == base_url_for_port(spec.port),
                  "the endpoint is the one the plan promised", handle.base_url)

        adopted = adopt_if_running(spec)
        sec.check(adopted is not None and adopted.process is None,
                  "a second serve_roles call would adopt this server instead of binding again")

        weights = report_weights_gib(handle)
        sec.check(weights is not None,
                  "the real weight figure is parseable from the startup log",
                  f"{weights} GiB" if weights is not None
                  else "not found -- vLLM may have reworded the line")
    finally:
        handle.stop()
        sec.note("server stopped")
    sec.check(not handle.is_alive(), "stop() actually terminated the process we started")


# ==============================================================================
#                                    roles
# ==============================================================================
#
# THE PHASE 1 GATE. Three properties, and the third is the one nothing else can see:
#
#   1. a plain chat completion works for every role binding;
#   2. a json_schema-constrained completion comes back and VALIDATES against the rubric schema;
#   3. NO THINKING TOKENS reach message.content.
#
# (3) is what proves the enable_thinking kwarg is actually right. vLLM passes
# chat_template_kwargs straight into the Jinja render, so a WRONG KEY is an unused variable:
# the request succeeds, the schema is usually still honoured, and every subsequent call quietly
# burns reasoning tokens. There is no error to catch -- only this assertion.


def _thinking_leak(content: Optional[str]) -> Optional[str]:
    """The first thinking marker present in *content*, or ``None``. ``None`` content leaks too."""
    if content is None:
        return "message.content is None (a reasoning block probably ate max_tokens)"
    lowered = content.lower()
    for marker in THINKING_MARKERS:
        if marker.lower() in lowered:
            return marker
    return None


def _sample_transcript() -> str:
    """A short, real-shaped MI transcript for the schema-constrained call."""
    from core.conversations import format_conversation_for_oracle

    return format_conversation_for_oracle([
        {"role": "therapist", "content": "What brings you in today?"},
        {"role": "patient", "content": "My doctor says I should cut down on drinking."},
        {"role": "therapist", "content": "What do you make of that advice?"},
        {"role": "patient", "content": "Part of me agrees. Part of me thinks it is fine."},
        {"role": "therapist", "content": "You can see both sides. What would change look like?"},
    ])


def cmd_roles(sec: Section, args: argparse.Namespace) -> None:
    """Phase 1: chat + schema + no-thinking per role binding, then kill/restart the server.

    Needs a reachable OpenAI-compatible endpoint. If none is listening and vLLM is not
    installed, this SKIPs -- the gate belongs on the GPU host.
    """
    import shutil
    from dataclasses import replace as _replace

    from core.concurrency import AsyncPrimitives, run_async
    from core.oracle import OracleConfig, get_evaluation_json, response_format_for
    from questionnaires import get_prompt_eval_questionnaire
    from roles import default_bindings, make_client, reset_client_cache
    from tools.vllm_serve import ensure_alive, report_weights_gib, serve_roles

    model = args.model or DEFAULT_ORACLE_MODEL
    port = args.port or 8000
    # The REAL default bindings, with only the model and endpoint overridden -- so what is under
    # test is the thinking-off extra_body and retry policy a run would actually send, not a
    # hand-built binding that happens to look like them.
    bindings = {role: _replace(binding, model=model, base_url=args.base_url)
                for role, binding in default_bindings().items()}
    sec.note(f"roles bound to {model} (one server serves all three: they differ only in "
             f"per-request sampling params, so every role is probed separately -- three "
             f"identical-looking bindings can still differ in extra_body or timeout)")

    if args.base_url:
        # An explicit endpoint means "do not plan a server". serve_roles would still plan one for
        # a local binding and REPLACE this base_url with the port it chose, which is the opposite
        # of what the flag says.
        wired: Dict[str, RoleBinding] = dict(bindings)
        handles: Dict[str, Any] = {}
        sec.note(f"--base-url given: talking to {args.base_url}; no server planned, and the "
                 f"kill/restart check is skipped (this process owns no process)")
    else:
        from tools.vllm_serve import adopt_if_running

        if shutil.which(args.executable) is None and \
                adopt_if_running(ServeSpec(model=model, port=port)) is None:
            raise _Skip(f"nothing serving {model} on port {port} and {args.executable!r} is not "
                        f"on PATH. Run this on the GPU host, or pass --base-url.")
        try:
            wired, handles = serve_roles(
                bindings, base_port=port, log_dir=tempfile.gettempdir(), timeout=args.timeout,
                executable=args.executable,
                gpu_memory_utilization=args.gpu_memory_utilization or 0.25)
        except RuntimeError as exc:
            raise _Skip(f"could not reach or start a server: {exc}") from exc

    handle = handles.get(model)
    transcript = _sample_transcript()
    eval_dict = get_prompt_eval_questionnaire(questionnaire=1, conversation=transcript)
    primitives = AsyncPrimitives(oracle_concurrency=4, patient_concurrency=4)

    async def _raw(binding: RoleBinding, messages: List[Dict[str, str]],
                   *, response_format: Optional[dict], max_tokens: int) -> Any:
        client = make_client(binding)
        kwargs: Dict[str, Any] = {
            "model": binding.model, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        extra = binding.extra_body
        if extra:
            kwargs["extra_body"] = extra
        return await client.chat.completions.create(**kwargs)

    for role, binding in wired.items():
        sec.note(f"--- role {role}: {binding.model} @ {binding.base_url} "
                 f"(extra_body={json.dumps(binding.extra_body)})")

        plain = run_async(_raw(binding, [{"role": "user",
                                          "content": "Reply with exactly: READY"}],
                               response_format=None, max_tokens=32))
        content = plain.choices[0].message.content
        sec.check(bool(content and content.strip()), f"{role}: a plain chat completion answers",
                  repr((content or "")[:60]))
        leak = _thinking_leak(content)
        sec.check(leak is None, f"{role}: NO thinking tokens in the plain completion",
                  "clean" if leak is None else f"LEAKED {leak!r} -- the enable_thinking key is "
                                               f"being ignored by the chat template")

        schema_format = response_format_for(binding, eval_dict["schema"],
                                            "questionnaire_1_evaluation")
        constrained = run_async(_raw(binding,
                                     [{"role": "user", "content": eval_dict["prompt"]}],
                                     response_format=schema_format, max_tokens=256))
        raw_json = constrained.choices[0].message.content
        leak = _thinking_leak(raw_json)
        sec.check(leak is None, f"{role}: NO thinking tokens in the json_schema completion",
                  "clean" if leak is None else f"LEAKED {leak!r}")

        # The authoritative validation is the oracle's own ladder (id echo, item count, type and
        # range), not a second copy of it here: a copy would drift from what actually grades.
        # 120 s, not args.timeout: that flag bounds SERVER READINESS (weight load plus CUDA
        # graph capture, minutes), while this bounds one scoring coroutine.
        cfg = OracleConfig(binding=binding, questionnaire_ids=(1,), max_tokens=256,
                           max_retries=2, request_timeout=120.0)
        data, n_questions, attempts = run_async(
            get_evaluation_json(make_client(binding), cfg, primitives, transcript, 1))
        sec.check(data is not None,
                  f"{role}: the json_schema response passes the oracle validation ladder",
                  f"{attempts} attempt(s)" if data is not None
                  else f"FAILED after {attempts} attempts -- see the [oracle] line above")
        if data is not None:
            scores = data.get("scores", [])
            sec.check(len(scores) == n_questions and all(isinstance(s, int) for s in scores),
                      f"{role}: exactly {n_questions} integer item scores came back",
                      f"scores={scores} mean={data.get('mean_score')}")

    # --- the server must survive its own death ---------------------------------------
    if handle is None:
        sec.note("no handle for this model (an explicit --base-url was used); "
                 "skipping the restart check")
    elif handle.process is None:
        sec.note("the server was ADOPTED (this process does not own it); "
                 "skipping the kill/restart check rather than killing someone else's server")
    else:
        weights = report_weights_gib(handle)
        sec.check(weights is not None, "real model-weight memory read from the startup log",
                  f"{weights} GiB" if weights is not None else "line not found in the log")
        handle.stop()
        reset_client_cache()          # the old pool points at a process that no longer exists
        sec.check(not handle.is_alive(), "the server is down after stop()")
        revived = ensure_alive(handle, timeout=args.timeout)
        reset_client_cache()          # again: the restart replaced the process behind the URL
        sec.check(revived.is_alive() and revived.restarts >= 1,
                  "ensure_alive relaunched the SAME spec after the server died",
                  f"restarts={revived.restarts}")
        weights_after = report_weights_gib(revived)
        sec.check(weights_after is not None,
                  "the relaunched server reports its weights too",
                  f"{weights_after} GiB" if weights_after is not None else "line not found")
        if not args.keep:
            revived.stop()
            sec.note("server stopped (pass --keep to leave it up)")


# ==============================================================================
#                          GPU PARTS: stopgen / dpo / grpo
# ==============================================================================


def _require_gpu(sec: Section, part: str, args: argparse.Namespace):
    """Common preamble for a GPU part: import order, CUDA presence, VRAM refusal, offline HF.

    Returns:
        The imported ``torch`` module.

    Raises:
        _Skip: no trl, no CUDA, or the VRAM arithmetic refuses.

    Notes:
        ``assert_import_order`` is called even though this module imports trl at the top: the
        top-level import is wrapped in ``try/except ImportError``, so on a host without trl the
        order guard is the thing that still says why a GPU part cannot run.
    """
    if trl is None:
        raise _Skip("trl is not installed; the GPU parts train with TRL trainers")
    assert_import_order()

    import torch

    if not torch.cuda.is_available():
        raise _Skip("no CUDA device visible")
    sec.note(f"device: {torch.cuda.get_device_name(0)}")
    _vram_refusal(sec, need_gib=_GPU_NEED_GIB[part], what=f"the {part} smoke", force=args.force)

    # Belt and braces for a programmatic caller (``main(["dpo"])`` from another script), where
    # the argv pre-scan at the top of this module never saw the subcommand. Effective only if
    # huggingface_hub has not been imported yet -- which is why the pre-scan exists.
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return torch


def _load_smoke_policy(sec: Section, args: argparse.Namespace):
    """Load the therapist base + tokenizer for a GPU part. Returns ``(tokenizer, model)``.

    Raises:
        _Skip: the weights are not in the local HF cache and downloading was not allowed. That
            is "this host cannot answer the question", not a defect in the code under test.
    """
    from core.config import DEFAULT_BASE_MODEL_ID

    from core.policy import patch_generate, setup_base_model, setup_tokenizer, sync_pad_token

    model_id = args.model or DEFAULT_BASE_MODEL_ID
    sec.note(f"policy: {model_id} "
             f"({'downloads allowed' if args.allow_download else 'local HF cache only'})")
    try:
        tokenizer = setup_tokenizer(model_id)
        model = setup_base_model(model_id)
    except OSError as exc:   # huggingface_hub's offline/gated errors all subclass OSError
        raise _Skip(f"{model_id} is not in the local HF cache and HF_HUB_OFFLINE is set "
                    f"({type(exc).__name__}). Pass --allow-download to fetch it.") from exc
    sync_pad_token(model, tokenizer)
    patch_generate(model, tokenizer)
    return tokenizer, model


def _lora_config():
    """The LoRA config the GPU parts train, taken from the project defaults (not restated)."""
    from peft import LoraConfig

    from core.config import DEFAULT_LORA_TARGET_MODULES

    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=list(DEFAULT_LORA_TARGET_MODULES))


def _filtered_trainer_kwargs(config_cls, kwargs: Dict[str, Any], critical: Sequence[str],
                             sec: Section) -> Dict[str, Any]:
    """Drop kwargs the pinned TRL does not know; FAIL if a critical one is among them.

    TRL renames fields between minor versions. A cosmetic rename should not break a smoke test,
    but a renamed CRITICAL field (the prompt cap, the group size, the stop strings) would let
    the step run while testing something else entirely -- so those are reported, loudly.
    """
    import dataclasses

    known = {f.name for f in dataclasses.fields(config_cls)}
    unknown = sorted(set(kwargs) - known)
    missing_critical = [name for name in unknown if name in critical]
    sec.check(not missing_critical,
              f"the pinned TRL's {config_cls.__name__} still accepts every critical field",
              f"critical {list(critical)} present"
              + (f"; dropped cosmetic {unknown}" if unknown and not missing_critical else "")
              if not missing_critical
              else f"CRITICAL fields absent from the pinned TRL: {missing_critical} -- the step "
                   f"would run while testing something else")
    return {k: v for k, v in kwargs.items() if k in known}


def cmd_stopgen(sec: Section, args: argparse.Namespace) -> None:
    """GPU: prove ``stop_strings`` binds through ``patch_generate`` and nothing leaks.

    ``stop_strings`` is silently inert unless a ``tokenizer=`` reaches the same ``generate``
    call, and TRL's internal generation does not pass one. The failure mode is not an error:
    generation runs to ``max_new_tokens``, the policy's self-played ChatML lands in the saved
    conversation, and the run trains toward rambling.
    """
    torch = _require_gpu(sec, "stopgen", args)

    from transformers import GenerationConfig

    from core.policy import CHATML_MARKERS, STOP_STRINGS, clean_completion, vram_report

    tokenizer, model = _load_smoke_policy(sec, args)
    model.eval()

    messages = [{"role": "system", "content": _SYS_THERAPIST},
                {"role": "user", "content": "Hi, I want to talk about my smoking."}]
    encoded = tokenizer(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
        return_tensors="pt", add_special_tokens=False).to(model.device)

    def _generate(stops: Optional[Sequence[str]]) -> str:
        cfg = GenerationConfig(max_new_tokens=80, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id,
                               **({"stop_strings": list(stops)} if stops else {}))
        with torch.no_grad():
            out = model.generate(**encoded, generation_config=cfg)   # patch injects tokenizer=
        return tokenizer.decode(out[0, encoded["input_ids"].shape[1]:], skip_special_tokens=False)

    unstopped = _generate(None)
    stopped = _generate(["."])
    n_unstopped = len(tokenizer.encode(unstopped, add_special_tokens=False))
    n_stopped = len(tokenizer.encode(stopped, add_special_tokens=False))
    # HF's StopStringCriteria is evaluated AFTER a token is appended, so generation stops on the
    # token FOLLOWING the one that completed the stop string: with stop=["."] the decode comes back
    # as "...my smoking.<", not "...my smoking.". Asserting endswith(".") therefore fails on a
    # working bind. What the bind actually guarantees is that the stop string was REACHED and
    # generation was cut there -- so require the marker to be present, the output to be shorter,
    # and the overshoot past the marker to be at most one token's worth of characters.
    tail_after_stop = stopped.split(".", 1)[1] if "." in stopped else stopped
    sec.check(n_stopped < n_unstopped and "." in stopped and len(tail_after_stop.strip()) <= 8,
              "stop_strings reaches generate() through patch_generate (the bind works)",
              f"{n_unstopped} tokens unstopped -> {n_stopped} with stop=['.'], "
              f"overshoot past the marker: {tail_after_stop.strip()!r}")

    chatml = _generate(STOP_STRINGS)
    cleaned = clean_completion(chatml)
    sec.check(not any(marker in cleaned for marker in CHATML_MARKERS),
              "no ChatML marker survives STOP_STRINGS + clean_completion",
              repr(cleaned[:80]))
    sec.check(cleaned == cleaned.strip(),
              "the cleaned completion is a usable utterance (stripped)")
    sec.note(f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, "
             f"reserved {vram_report()['reserved_gib']:.2f} GiB")


def cmd_dpo(sec: Section, args: argparse.Namespace) -> None:
    """GPU: one tiny DPO step, proving the prompt cap holds and the first step does not OOM.

    PTO must pre-cap its prompt: TRL 1.4.0's ``DPOConfig`` dropped ``max_prompt_length`` and caps
    prompt+completion with a single ``max_length`` under ``truncation_mode='keep_start'`` --
    which slices the RESPONSE off, leaving a pair whose chosen and rejected are both empty.
    """
    torch = _require_gpu(sec, "dpo", args)

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    from core.conversations import build_truncated_training_prompt, turns_to_messages
    from pto.pto_trainer import DPO_FRAMING_HEADROOM_TOKENS

    tokenizer, model = _load_smoke_policy(sec, args)

    turns = _demo_turns(30)
    full = tokenizer.apply_chat_template(turns_to_messages(turns, _SYS_THERAPIST),
                                         add_generation_prompt=True, tokenize=False)
    capped = build_truncated_training_prompt(turns, _SYS_THERAPIST, tokenizer,
                                             _MAX_PROMPT_TOKENS)
    n_full = len(tokenizer.encode(full, add_special_tokens=False))
    n_capped = len(tokenizer.encode(capped or "", add_special_tokens=False))
    sec.check(capped is not None and n_capped <= _MAX_PROMPT_TOKENS,
              "the DPO prompt is capped BEFORE the trainer sees it",
              f"{n_full} -> {n_capped} tokens (budget {_MAX_PROMPT_TOKENS})")
    sec.check(capped is not None and "counselor named David" in capped
              and turns[-1]["content"][:12] in capped,
              "the system prompt and the newest turn survived the cap")

    dataset = Dataset.from_list([
        {"prompt": capped,
         "chosen": f"What makes change matter to you now? ({i})",
         "rejected": f"You should just quit. ({i})"}
        for i in range(8)
    ])
    kwargs = _filtered_trainer_kwargs(DPOConfig, {
        "output_dir": os.path.join(tempfile.gettempdir(), "exp4_smoke_dpo"),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1,
        "learning_rate": 1e-4,
        "beta": 0.1,
        # The shipped formula, headroom included: TRL prepends BOS to the prompt and appends EOS
        # to chosen/rejected AFTER the pre-cap, so the two configured halves alone are 2 short.
        "max_length": _MAX_PROMPT_TOKENS + _MAX_RESPONSE_TOKENS + DPO_FRAMING_HEADROOM_TOKENS,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "precompute_ref_log_probs": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "seed": 42,
    }, critical=("max_length", "beta", "precompute_ref_log_probs", "gradient_checkpointing"),
        sec=sec)

    trainer = DPOTrainer(model=model, args=DPOConfig(**kwargs), processing_class=tokenizer,
                        train_dataset=dataset, peft_config=_lora_config())
    trainer.train()
    sec.check(trainer.state.global_step > 0,
              "DPO trains with grad-checkpointing + precomputed reference log-probs, no OOM",
              f"global_step={trainer.state.global_step}, "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def cmd_grpo(sec: Section, args: argparse.Namespace) -> None:
    """GPU: one tiny GRPO step with a stub reward, proving the stop strings reach TRL's generate.

    The reward is a stub on purpose -- the real one calls the oracle, and this part must run
    with no server and no key. What is under test is that TRL accepts the group arithmetic and
    that ``generation_kwargs={"stop_strings": ...}`` survives the trainer's own unwrap path.
    """
    torch = _require_gpu(sec, "grpo", args)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from core.policy import STOP_STRINGS, patch_generate

    tokenizer, model = _load_smoke_policy(sec, args)

    questions = [
        "I keep putting off quitting.",
        "I'm not sure I can change.",
        "My doctor told me to cut down.",
        "I feel stuck about my habit.",
    ]
    dataset = Dataset.from_list([
        {"prompt": tokenizer.apply_chat_template(
            [{"role": "system", "content": _SYS_THERAPIST}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False)}
        for q in questions
    ])

    def stub_reward(prompts: Sequence[str], completions: Sequence[str], **_: Any) -> List[float]:
        """Length in characters, scaled. Deterministic, free, and never None."""
        return [float(len(c)) / 50.0 for c in completions]

    kwargs = _filtered_trainer_kwargs(GRPOConfig, {
        "output_dir": os.path.join(tempfile.gettempdir(), "exp4_smoke_grpo"),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "num_train_epochs": 1,
        "learning_rate": 1e-4,
        "beta": 0.0,
        "max_completion_length": _MAX_RESPONSE_TOKENS,
        "temperature": 1.2,
        "bf16": True,
        "generation_kwargs": {"stop_strings": list(STOP_STRINGS)},
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "seed": 42,
    }, critical=("num_generations", "generation_kwargs", "max_completion_length", "beta"),
        sec=sec)

    trainer = GRPOTrainer(model=model, args=GRPOConfig(**kwargs), processing_class=tokenizer,
                         reward_funcs=stub_reward, train_dataset=dataset,
                         peft_config=_lora_config())
    patch_generate(trainer.model, tokenizer)   # the trainer just built a fresh wrapper
    trainer.train()

    terminated = [h["completions/mean_terminated_length"] for h in trainer.state.log_history
                  if "completions/mean_terminated_length" in h]
    sec.check(trainer.state.global_step > 0,
              "a GRPO step completes with generation_kwargs stop_strings",
              f"global_step={trainer.state.global_step}, "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    sec.check(bool(terminated),
              "TRL reported completion lengths (the group actually generated)",
              f"terminated_length={terminated} (cap {_MAX_RESPONSE_TOKENS})")


# ==============================================================================
#                                     CLI
# ==============================================================================


_COMMANDS: Dict[str, Callable[[Section, argparse.Namespace], None]] = {
    "naming": cmd_naming,
    "config": cmd_config,
    "convs": cmd_convs,
    "vram": cmd_vram,
    "serve": cmd_serve,
    "roles": cmd_roles,
    "stopgen": cmd_stopgen,
    "dpo": cmd_dpo,
    "grpo": cmd_grpo,
}


def build_parser() -> argparse.ArgumentParser:
    """The command line. The subcommand must be the FIRST token (``all`` forwards the rest)."""
    parser = argparse.ArgumentParser(
        prog="smoke.py",
        description="Exp4 offline/local smoke gate. Phase 0-4 of the CLAUDE.md gate table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("parts: " + ", ".join(PARTS) + ", all\n"
                "exit codes: 0 PASS, 1 FAIL, 3 SKIP (never a failure)"),
    )
    parser.add_argument("command", choices=(*PARTS, "all"),
                        help="which part to run; must be the first argument")
    parser.add_argument("--force", action="store_true",
                        help="downgrade a VRAM refusal to a warning (you own the arithmetic)")
    parser.add_argument("--allow-download", action="store_true",
                        help="let the GPU parts fetch model weights instead of running offline")
    parser.add_argument("--model", default=None,
                        help="model override: the served model (serve/roles) or the therapist "
                             "base (stopgen/dpo/grpo)")
    parser.add_argument("--base-url", default=None,
                        help="roles: talk to this endpoint instead of starting/adopting a server")
    parser.add_argument("--port", type=int, default=None,
                        help=f"serve: default {SMOKE_PORT}; roles: default 8000")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="vLLM pre-allocation fraction (serve default 0.15, roles 0.25)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="serve: served context length (default 4096 for the smoke model)")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="server readiness timeout, seconds")
    parser.add_argument("--executable", default="vllm",
                        help="server binary for serve/roles")
    parser.add_argument("--keep", action="store_true",
                        help="roles: leave the server running when the gate finishes")
    return parser


def _run_all(forwarded: Sequence[str]) -> int:
    """Run every part in its own subprocess and summarise. Returns the aggregate exit code.

    One process per part on purpose: each frees its VRAM on exit, and a part that takes the
    interpreter down (or the machine, on the local card) does not cost the other eight.
    """
    results: List[Tuple[str, int]] = []
    for part in PARTS:
        print(f"\n########## {part} ##########", flush=True)
        completed = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), part, *forwarded], check=False)
        results.append((part, completed.returncode))

    label = {EXIT_PASS: STATUS_PASS, EXIT_FAIL: STATUS_FAIL, EXIT_SKIP: STATUS_SKIP}
    print("\n" + "=" * 78)
    print("smoke summary")
    print("-" * 78)
    for part, code in results:
        print(f"  {label.get(code, f'EXIT {code}'):5}  {part}")
    print("=" * 78)

    failed = [part for part, code in results if code not in (EXIT_PASS, EXIT_SKIP)]
    if failed:
        print(f"{STATUS_FAIL}: {', '.join(failed)}")
        return EXIT_FAIL
    if all(code == EXIT_SKIP for _, code in results):
        return EXIT_SKIP
    return EXIT_PASS


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one part (or all of them) and return the process exit code.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``. The subcommand must come first.

    Returns:
        ``0`` PASS, ``1`` FAIL, ``3`` SKIP. ``all`` returns 1 if any part failed, 3 if every
        part skipped, else 0 -- so a caller that tests "nonzero" never mistakes "this host has
        no GPU" for "the DPO step is broken".
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    forwarded = raw[1:]          # everything after the subcommand, for `all` to pass through

    if args.command == "all":
        return _run_all(forwarded)

    print(f"smoke.py {args.command} | host={detect_host()} | "
          f"python={sys.version.split()[0]} | trl={'yes' if trl is not None else 'absent'}")
    section = run_part(args.command, _COMMANDS[args.command], args)
    return section.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
