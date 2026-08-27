"""config.py -- the frozen knobs, the computed arm name, and every path shape.

A training run has exactly one place a human types a number: cell 1 of a trainer notebook. Every
other module is handed an already-decided value. This module is the boundary between those two
worlds -- it reads the flat cell-1 globals, freezes them into typed dataclasses, computes the arm
name from them, derives every filesystem path from that name, and serialises the whole thing into
``run_metadata.json``. Nothing downstream re-reads a global, and nothing downstream joins a path.

Three failure modes from Exp3 are designed out here, and each one cost real work:

**1. A hand-typed ``EXPERIMENT_NAME``.** Exp3's cell 1 assigned the name as an f-string that
happened to mention ``LA{K}`` and ``MCL{N}``, but *not* the oracle or patient model. Change
``ORACLE_MODEL_ID`` and the run writes a differently-rewarded policy into the default arm's folder,
where the resume-by-skipping-existing scorer reports "already scored" against the other arm's
numbers. Exp3 could only defend that with a runtime assertion (``assert_name_matches_roles``).
Here :func:`build_grpo_config` / :func:`build_pto_config` **compute** the name via
:func:`naming.build_experiment_name` from the very values that are about to be frozen into the
config, and the name is never read from the globals dict at all. The assertion has nothing left to
guard.

**2. Knobs that change a run without forking its folder.** ``EXPERIMENT_NAME`` encodes the arm's
identity axes (method, rubric, K, MCL, branch width/mode, roles) and *deliberately* nothing else --
a folder per learning rate would be unusable. So every other knob is silently mutable: change
``PREF_FILTER_TAU``, ``LOOKAHEAD_SUB_BATCH_SIZE`` or a sampling temperature mid-arm and the folder
name is identical while the science is not. ``run_metadata.json`` is the only record that can
distinguish them, which makes serialisation completeness a correctness property, not a nicety.
:func:`config_to_metadata` therefore **asserts** that a documented list of silently-mutable knobs
(:data:`SILENTLY_MUTABLE_KNOBS`) is present in the payload it produces, so a future field cannot be
added to a config and quietly left out of the record.

Exp3 had to mirror ``lookahead_k`` and ``lookahead_sub_batch_size`` onto its ``TrainingConfig`` as
audit-only duplicate fields, purely because its ``LookaheadConfig`` was never serialised -- so
before that mirror existed, no run on disk said which sub-batch it ran. **Exp4 serialises
``LookaheadConfig`` itself** (``core.lookahead``'s dataclass goes into the payload's ``lookahead``
section verbatim), so there is no mirror, and no way for the mirror to drift from the live value.

**3. Paths built by string concatenation in five modules.** Every path shape in CLAUDE.md's "Data
layout" section is a method on :class:`RunPaths`. If a layout ever changes, it changes here.

Import weight
-------------
**No torch.** The read-only EDA imports this module to resolve paths and read metadata, so heavy
imports are lazy: ``core.oracle`` and ``core.lookahead`` are imported *inside* the builders (the
latter is torch-side by nature). ``roles``, ``naming``, ``core.timing`` and ``core.recorder`` are
stdlib-only and imported normally.

Usage (trainer notebook, after ``serve_roles`` has filled in the base URLs)::

    from core.config import build_grpo_config
    train_cfg, roles_cfg, gen_cfg, oracle_cfg, la_cfg, paths = build_grpo_config(globals())
    write_run_metadata(config_to_metadata(train_cfg, roles_cfg, gen_cfg,
                                          oracle_cfg, la_cfg, paths), paths)
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import os
import platform
import re
import socket
import time
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from naming import (
    GRAMMAR_VERSION,
    build_experiment_name,
    model_state_label,
    parse_experiment_name,
    parse_model_state_label,
    qtag_for,
)
from roles import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_ORACLE_MODEL,
    DEFAULT_PATIENT_MODEL,
    PROVIDERS,
    RoleBinding,
    make_binding,
)
from core.recorder import to_jsonable
from core.timing import sessions_path

if TYPE_CHECKING:  # pragma: no cover - annotations only, never imported at runtime
    from core.lookahead import LookaheadConfig
    from core.oracle import OracleConfig

__all__ = [
    # Constants
    "METADATA_SCHEMA",
    "DEFAULT_BASE_MODEL_ID",
    "DEFAULT_LORA_TARGET_MODULES",
    "DEFAULT_STOP_STRINGS",
    "resolve_stop_strings",
    "ITERATION_PREFIX",
    "ADAPTER_SUBDIR",
    "TRAINING_SUBDIR",
    "EDA_SUBDIR",
    "PREF_PAIRS_SUBDIR",
    "GENERATIONS_FILENAME",
    "PAIRS_FILENAME",
    "PROGRESS_FILENAME",
    "ITERATION_METADATA_FILENAME",
    "RUN_METADATA_FILENAME",
    "RUN_METADATA_HISTORY_FILENAME",
    "SILENTLY_MUTABLE_KNOBS",
    # Dataclasses
    "RunPaths",
    "RolesConfig",
    "GenConfig",
    "TrainingConfigBase",
    "GRPOTrainingConfig",
    "PTOTrainingConfig",
    # Builders + checks
    "build_grpo_config",
    "build_pto_config",
    "validate_config",
    "config_to_metadata",
    "write_run_metadata",
    "format_summary",
]


# ==============================================================================
#  Constants
# ==============================================================================

#: Bumped when the ``run_metadata.json`` payload SHAPE changes (a reader keys on this).
METADATA_SCHEMA = "exp4-run-metadata/1"

#: The therapist policy default. Selectable per arm since 2026-08-27 and ENCODED in the arm name
#: (the ``_Th<tag>`` field -- see naming.py), so the two variants can never share a folder:
#:   meta-llama/Llama-3.2-1B-Instruct  (_ThL1Bi, default) -- ships the official Llama-3 chat
#:       template; stops on the single special token <|eot_id|>, so no string-stopping and no
#:       ChatML self-play class. STOP_STRINGS="auto" resolves to () for it.
#:   meta-llama/Llama-3.2-1B           (_ThL1B) -- ships NO template; the hand-written ChatML
#:       template is installed and the ChatML stop strings are required.
#: ``run_metadata.json`` still records the exact snapshot id (the tag names a family).
DEFAULT_BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

DEFAULT_LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",
)

#: The anti-degeneracy stop strings FOR THE BASE THERAPIST, duplicated from
#: ``core.policy.STOP_STRINGS``.
#:
#: WARNING: this is a deliberate duplicate. ``core.policy`` imports torch at module level, and
#: this module must stay importable by the read-only EDA, so it cannot import the constant it is
#: copying. The two MUST stay equal -- ``tools/smoke.py`` is the right place to assert that. The
#: markers are ordinary BPE pieces, not special tokens, so the base Llama happily writes both
#: speakers without them; the whole anti-degeneracy stack (stop strings + clean_completion +
#: patch_generate + REWARD_FLOOR) is load-bearing THERE. The Instruct therapist needs none of
#: it: its native template terminates every turn with the single special token <|eot_id|>, so
#: ``STOP_STRINGS = "auto"`` resolves to an EMPTY tuple for it (see
#: :func:`resolve_stop_strings`) and generation stops on the eos-id list instead.
DEFAULT_STOP_STRINGS: Tuple[str, ...] = ("<|im_end|>", "<|im_start|>")


def _therapist_has_native_template(base_model_id: str) -> bool:
    """True when the therapist checkpoint ships its own chat template (the Instruct variants).

    Decided from the model id, not the tokenizer -- the config is frozen before any tokenizer
    is loaded, and this module must stay importable without transformers. ``"instruct"`` in the
    id is exact for both supported therapists; a future therapist family that spells it
    differently needs this predicate extended, and ``core.policy.setup_tokenizer`` prints which
    template it actually used, so a mismatch is visible at model-load time.
    """
    return "instruct" in str(base_model_id).lower()


def resolve_stop_strings(raw: Any, base_model_id: str) -> Tuple[str, ...]:
    """Resolve the cell-1 ``STOP_STRINGS`` value against the therapist variant.

    ``"auto"`` (the notebooks' default) resolves to :data:`DEFAULT_STOP_STRINGS` for the
    template-less base therapist and to ``()`` for an Instruct therapist -- string stopping on
    Instruct is pure cost (the markers never occur; its turns end on the special <|eot_id|>).
    Anything else is taken verbatim: an explicit tuple pins the behavior, an explicit ``None``
    or ``()`` means "no string stopping" regardless of variant (validated downstream).
    """
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return () if _therapist_has_native_template(base_model_id) else DEFAULT_STOP_STRINGS
    if raw is None:
        return ()
    return _as_str_tuple(raw, "STOP_STRINGS")

# Layout tokens. Same duplication caveat as DEFAULT_STOP_STRINGS: ``core.policy`` defines
# ITER_PREFIX / ADAPTER_SUBDIR for its checkpoint walk and cannot be imported from here.
ITERATION_PREFIX = "iteration_"
ADAPTER_SUBDIR = "adapter"
TRAINING_SUBDIR = "training"
EDA_SUBDIR = "eda"
PREF_PAIRS_SUBDIR = "pref_pairs"

GENERATIONS_FILENAME = "generations.jsonl"
PAIRS_FILENAME = "pairs.csv"
PROGRESS_FILENAME = "_progress.json"
ITERATION_METADATA_FILENAME = "iteration_metadata.json"
RUN_METADATA_FILENAME = "run_metadata.json"
RUN_METADATA_HISTORY_FILENAME = "run_metadata_history.jsonl"

_DATA_SUBDIR = "data"
_RUNS_SUBDIR = "runs"
_CONVERSATIONS_SUBDIR = "conversations"
_EVAL_SCORES_SUBDIR = "eval_scores"

_PTO_MODES = ("greedy", "independent")
_PTO_MODE_TOKEN = {"greedy": "greedy", "independent": "indep", "indep": "indep"}

_SAVE_STRATEGIES = ("steps", "epoch", "no")
_REPORT_TARGETS = ("tensorboard", "wandb", "none")

# Windows reserved device names: a directory called "CON" or "NUL" cannot be created on NTFS, and
# the failure surfaces as a confusing permission error rather than as "bad name".
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                 *(f"LPT{i}" for i in range(1, 10))}
_ILLEGAL_SEGMENT_CHARS = '<>:"/\\|?*'
_SAFE_PARTITION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ==============================================================================
#  Silently-mutable knobs -- the completeness contract for run_metadata.json
# ==============================================================================
#
# Dotted paths into the ``config`` section of the payload built by ``config_to_metadata``. Every
# one of these changes results WITHOUT changing EXPERIMENT_NAME, so the metadata file is the only
# thing that can tell two runs of the "same" arm apart. ``config_to_metadata`` asserts each path
# resolves, which turns "we intended to serialise it" into "it is serialised or the run stops".
#
# Adding a knob to a config dataclass and forgetting it here is not caught (the assert only checks
# the listed paths exist, not that the list is exhaustive) -- but every field of every dataclass is
# serialised wholesale by ``dataclasses.asdict``, so the only way to lose one is to stop putting a
# whole section into the payload, which is exactly what this list detects.

_SILENT_COMMON: Tuple[str, ...] = (
    "training.num_iterations",
    "training.epochs_per_iteration",
    "training.learning_rate",
    "training.train_batch_size",
    "training.eval_batch_size",
    "training.gradient_accumulation_steps",
    "training.max_completion_length",
    "training.warmup_steps_ratio",
    "training.eval_split_ratio",
    "training.gradient_checkpointing",
    "training.lora_r",
    "training.lora_alpha",
    "training.lora_dropout",
    "training.lora_target_modules",
    "training.use_4bit",
    "training.seed",
    "training.save_strategy",
    "training.save_steps",
    "training.save_total_limit",
    "generation.num_conversations_per_iter",
    "generation.num_utterances_for_data",
    "generation.conversation_batch_size",
    "generation.temperature_therapist",
    "generation.temperature_patient",
    "generation.max_tokens_per_response",
    "generation.max_prompt_tokens",
    "generation.therapist_max_input_tokens",
    "generation.patient_concurrency",
    "generation.stop_strings",
    "lookahead.k",
    "lookahead.temperature_therapist",
    "lookahead.temperature_patient",
    "lookahead.max_tokens",
    "lookahead.sub_batch_size",
    "oracle.eval_temperature",
    "oracle.max_tokens",
    "oracle.max_retries",
    "oracle.request_timeout",
    "oracle.max_concurrency",
    "oracle.min_success_ratio",
)

_SILENT_GRPO: Tuple[str, ...] = (
    "training.num_generations",
    "training.grpo_beta",
    "training.grpo_temperature",
    "training.grpo_loss_type",
    "training.grpo_inner_iterations",
)

_SILENT_PTO: Tuple[str, ...] = (
    "training.num_branches_per_turn",
    "training.pref_filter_tau",
    "training.branch_sample_temperature",
    "training.branch_max_tokens",
    "training.dpo_beta",
    "training.dpo_loss_type",
    "training.precompute_ref_log_probs",
    "training.greedy_trunk_target_len",
)

#: Public view of the above, for documentation and for a test that wants to enumerate them.
SILENTLY_MUTABLE_KNOBS: Dict[str, Tuple[str, ...]] = {
    "common": _SILENT_COMMON,
    "GRPO": _SILENT_GRPO,
    "PTO": _SILENT_PTO,
}


def _warn(message: str) -> None:
    """One-line warning channel. Printed, not ``warnings.warn``: a notebook shows print output
    inline and swallows warnings into a filter the user never set."""
    print(f"[config] WARNING: {message}")


# ==============================================================================
#  RunPaths
# ==============================================================================


@dataclass(frozen=True)
class RunPaths:
    """Every path shape in CLAUDE.md's "Data layout", derived from one arm name.

    No other module joins a path. That is the whole point: Exp3 had five modules building
    ``os.path.join(data, "grpo_Exp3", "conversations", mode_tag, name, f"model_iter_{n}_TT{...}")``
    from their own local variables, and the temperature suffix meant every reader needed a glob.

    Attributes:
        data_root: The ``data/`` directory -- normally ``<workspace>/data``, which is three Google
            Drive symlinks. Overridable so a Colab session can point straight at the mounted Drive
            path and a smoke test can point at a scratch dir.
        experiment_name: The computed arm name. Validated as a legal single path segment here;
            :func:`naming.parse_experiment_name` is the authority on whether it is a legal *arm*.

    Notes:
        Two Exp3 path levels are deliberately GONE:

        * the ``_TT0.9_TP0.7`` sampling-temperature suffix on the conversations folder -- the
          temperatures live in ``run_metadata.json``, and the suffix only made the folder name a
          second, unparsed config record that every reader had to glob past;
        * the ``full|quicktest`` MODE_TAG level -- a quicktest normally differs in K, G or MCL and
          therefore already resolves to a different arm name.

          WARNING: "normally" is not "always". A quicktest that keeps the same rubric, K, MCL,
          branch width and role models as a real arm resolves to the SAME folder and will pollute
          it. When smoke-testing an arm you intend to run for real, change something the name
          encodes (K, or the branch width) or point ``data_root`` at a scratch directory.
    """

    data_root: str
    experiment_name: str

    def __post_init__(self) -> None:
        _assert_path_segment(self.experiment_name, "experiment_name")
        object.__setattr__(self, "data_root", os.path.abspath(os.path.expanduser(self.data_root)))

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_workspace(cls,
                       experiment_name: str,
                       *,
                       workspace_root: Optional[str] = None,
                       data_root: Optional[str] = None) -> "RunPaths":
        """Build from the workspace root, resolving it if not given.

        Args:
            experiment_name: The computed arm name.
            workspace_root: ``Exp4_OpenStack/``. Defaults to
                :func:`core.runtime.resolve_workspace_root`, which walks up from the cwd and
                falls back to this file's own location.
            data_root: Explicit ``data/`` directory; wins over *workspace_root*. Use it on Colab
                to point at mounted Drive without pretending the code lives there.

        Notes:
            Resolution is deliberately NOT cached -- on Colab the answer legitimately changes the
            moment Drive mounts.
        """
        if data_root:
            return cls(data_root=data_root, experiment_name=experiment_name)
        if not workspace_root:
            from core.runtime import resolve_workspace_root  # local: keeps the import graph flat
            workspace_root = resolve_workspace_root()
        return cls(data_root=os.path.join(workspace_root, _DATA_SUBDIR),
                   experiment_name=experiment_name)

    # -- roots -----------------------------------------------------------------

    @property
    def runs_root(self) -> str:
        """``data/runs`` -- one Drive symlink."""
        return os.path.join(self.data_root, _RUNS_SUBDIR)

    @property
    def conversations_root(self) -> str:
        """``data/conversations`` -- one Drive symlink."""
        return os.path.join(self.data_root, _CONVERSATIONS_SUBDIR)

    @property
    def eval_scores_root(self) -> str:
        """``data/eval_scores`` -- one Drive symlink; the score lake."""
        return os.path.join(self.data_root, _EVAL_SCORES_SUBDIR)

    # -- run level -------------------------------------------------------------

    @property
    def run_dir(self) -> str:
        """``data/runs/<EXP_NAME>`` -- adapters, checkpoints, per-iteration artifacts."""
        return os.path.join(self.runs_root, self.experiment_name)

    @property
    def run_metadata_path(self) -> str:
        """``run_metadata.json`` -- the CURRENT config. Overwritten on every process."""
        return os.path.join(self.run_dir, RUN_METADATA_FILENAME)

    @property
    def run_metadata_history_path(self) -> str:
        """``run_metadata_history.jsonl`` -- append-only, one line per process.

        Exp3 fix #5: its ``run_metadata.json`` was overwritten in place, so a resume under changed
        knobs restamped the whole arm, earlier iterations included, and the previous values were
        simply gone. Here the overwrite still happens (a reader wants one current file) but the
        superseded payload survives in this log.
        """
        return os.path.join(self.run_dir, RUN_METADATA_HISTORY_FILENAME)

    @property
    def conv_root(self) -> str:
        """``data/conversations/<EXP_NAME>`` -- parent of every ``model_iter_<N>``."""
        return os.path.join(self.conversations_root, self.experiment_name)

    # -- iteration level -------------------------------------------------------

    def iteration_dir(self, n: int) -> str:
        """``data/runs/<EXP_NAME>/iteration_<N>``."""
        return os.path.join(self.run_dir, f"{ITERATION_PREFIX}{_positive_index(n, 'iteration')}")

    def adapter_dir(self, n: int) -> str:
        """``iteration_<N>/adapter`` -- its EXISTENCE is the definition of "iteration done"."""
        return os.path.join(self.iteration_dir(n), ADAPTER_SUBDIR)

    def training_dir(self, n: int) -> str:
        """``iteration_<N>/training`` -- the HF Trainer ``output_dir`` (``checkpoint-*``, tb logs)."""
        return os.path.join(self.iteration_dir(n), TRAINING_SUBDIR)

    def eda_dir(self, n: int) -> str:
        """``iteration_<N>/eda``."""
        return os.path.join(self.iteration_dir(n), EDA_SUBDIR)

    def generations_path(self, n: int) -> str:
        """``iteration_<N>/eda/generations.jsonl`` -- the per-branch capture."""
        return os.path.join(self.eda_dir(n), GENERATIONS_FILENAME)

    def pref_pairs_dir(self, n: int) -> str:
        """``iteration_<N>/pref_pairs`` -- PTO only."""
        return os.path.join(self.iteration_dir(n), PREF_PAIRS_SUBDIR)

    def pairs_csv_path(self, n: int) -> str:
        """``iteration_<N>/pref_pairs/pairs.csv``.

        Both the DPO audit trail AND the Step-2 completion marker: its presence makes a resumed
        iteration reload the pairs and skip the (dominant) preference build. An EMPTY file is
        therefore a trap -- reload 0 pairs, skip the build, then fail the "0 pref pairs" guard.
        Delete it to force a clean rebuild; do NOT lower tau, which is a science change mid-arm
        that the folder name cannot record.
        """
        return os.path.join(self.pref_pairs_dir(n), PAIRS_FILENAME)

    def pref_progress_path(self, n: int) -> str:
        """``iteration_<N>/pref_pairs/_progress.json`` -- the mid-build resume snapshot."""
        return os.path.join(self.pref_pairs_dir(n), PROGRESS_FILENAME)

    def iteration_metadata_path(self, n: int) -> str:
        """``iteration_<N>/iteration_metadata.json``."""
        return os.path.join(self.iteration_dir(n), ITERATION_METADATA_FILENAME)

    def timing_sessions_path(self, n: int) -> str:
        """``iteration_<N>/timing_sessions.jsonl`` -- delegated to :mod:`core.timing`."""
        return sessions_path(self.iteration_dir(n))

    # -- conversations ---------------------------------------------------------

    def conv_dir_for(self, iter_label: Union[int, str]) -> str:
        """``data/conversations/<EXP_NAME>/model_iter_<N>``.

        Args:
            iter_label: The MODEL STATE, either as an int or as a ``"model_iter_<N>"`` string.

        Notes:
            The label names the policy that GENERATED the conversations, not the iteration that
            consumed them: iteration ``n`` generates with the iter-(``n``-1) adapter and writes
            ``model_iter_{n-1}``, so ``N`` iterations yield ``N+1`` folders and ``model_iter_0``
            is always the untrained base. Passing the loop counter here is the classic off-by-one.
        """
        if isinstance(iter_label, str):
            parse_model_state_label(iter_label)          # raises on anything off-contract
            label = iter_label
        else:
            label = model_state_label(_positive_index(iter_label, "model state"))
        return os.path.join(self.conv_root, label)

    def conversation_csv_path(self, iter_label: Union[int, str], persona_id: int) -> str:
        """``.../model_iter_<N>/pers<PID>.csv`` -- named by the STABLE persona id (Exp3 fix #2).

        ``pers07.csv`` is persona 7 in every iteration, forever. Exp3 named conversation files by
        the shuffled processing index, so ``conversation_3.csv`` was a different persona each
        iteration and every EDA module had to replay ``Random(seed + k + 1)`` to pair anything.
        """
        pid = int(persona_id)
        if pid < 0:
            raise ValueError(f"persona_id={persona_id!r} must be >= 0")
        return os.path.join(self.conv_dir_for(iter_label), f"pers{pid:02d}.csv")

    # -- score lake ------------------------------------------------------------

    def score_partition_dir(self, judge: str, rep: int, metric: str) -> str:
        """``data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/<EXP_NAME>``.

        There is no ``oracle=<O>`` level (Exp3 had one): the training oracle is inside
        ``<EXP_NAME>`` because Exp4 encodes role tags unconditionally.
        """
        _assert_partition_token(judge, "judge")
        _assert_partition_token(metric, "metric")
        r = int(rep)
        if r < 0:
            raise ValueError(f"rep={rep!r} must be >= 0 (rep=0 is the full-grid draw)")
        return os.path.join(self.eval_scores_root, f"judge={judge}", f"rep={r}",
                            f"metric={metric}", self.experiment_name)

    def score_parquet_path(self, judge: str, rep: int, metric: str, n: Union[int, str]) -> str:
        """One parquet per (judge, rep, metric, arm, model state) -- 96 rows (Exp3 fix #4).

        Exp3 wrote ~50k single-row CSVs and needed a parquet fold cache plus a content-signature
        manifest to read them at all. One file per model state needs no fold, no cache and no
        manifest.
        """
        label = n if isinstance(n, str) else model_state_label(_positive_index(n, "model state"))
        if isinstance(n, str):
            parse_model_state_label(label)
        return os.path.join(self.score_partition_dir(judge, rep, metric), f"{label}.parquet")

    # -- side effects ----------------------------------------------------------

    def ensure_run_dir(self) -> str:
        """Create ``run_dir`` (and parents) and return it."""
        os.makedirs(self.run_dir, exist_ok=True)
        return self.run_dir

    def ensure_iteration_dir(self, n: int) -> str:
        """Create ``iteration_<N>`` (and parents) and return it."""
        path = self.iteration_dir(n)
        os.makedirs(path, exist_ok=True)
        return path

    def ensure_conv_dir(self, iter_label: Union[int, str]) -> str:
        """Create the model-state conversations folder and return it."""
        path = self.conv_dir_for(iter_label)
        os.makedirs(path, exist_ok=True)
        return path

    def describe(self) -> Dict[str, str]:
        """The path set, for ``run_metadata.json``. Absolute, so a reader can tell hosts apart."""
        return {
            "data_root": self.data_root,
            "run_dir": self.run_dir,
            "conv_root": self.conv_root,
            "eval_scores_root": self.eval_scores_root,
            "run_metadata": self.run_metadata_path,
            "run_metadata_history": self.run_metadata_history_path,
        }


def _positive_index(n: Any, what: str) -> int:
    """Coerce an index to a non-negative int, or say which one was wrong."""
    try:
        idx = int(n)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{what} index {n!r} is not an integer") from ex
    if idx < 0:
        raise ValueError(f"{what} index {n!r} must be >= 0")
    return idx


def _assert_path_segment(value: str, what: str) -> None:
    """Reject anything that cannot be one NTFS directory name.

    A bad arm name does not fail at ``os.makedirs`` in a readable way -- on Windows a trailing dot
    is silently stripped and a reserved device name raises a permission error, so the run lands in
    a folder nobody can find again.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    bad = sorted({ch for ch in value if ch in _ILLEGAL_SEGMENT_CHARS})
    if bad:
        raise ValueError(f"{what}={value!r} contains path-illegal characters {bad}")
    if value != value.strip() or value.endswith("."):
        raise ValueError(f"{what}={value!r} may not start/end with whitespace or end with '.'")
    if value.split(".")[0].upper() in _WIN_RESERVED:
        raise ValueError(f"{what}={value!r} is a reserved Windows device name")


def _assert_partition_token(value: str, what: str) -> None:
    """Score-lake partition tokens go into ``judge=<tag>`` / ``metric=<M>`` folder names."""
    if not isinstance(value, str) or not _SAFE_PARTITION_RE.match(value):
        raise ValueError(
            f"{what}={value!r} must match [A-Za-z0-9][A-Za-z0-9._-]* -- it becomes a path segment"
        )


# ==============================================================================
#  RolesConfig
# ==============================================================================


@dataclass(frozen=True)
class RolesConfig:
    """Which model plays each LLM role, with its endpoint and call policy.

    The patient defines the TASK and the oracle defines the TRAINING TARGET, so swapping either
    makes arms incomparable -- which is why both tags are encoded in ``EXPERIMENT_NAME``. The
    judge grades after the fact and is re-runnable, so it partitions the score lake instead
    (``judge=<tag>/``) and does not appear in the arm name.

    Notes:
        For an ``openai_compat`` role, ``base_url`` is filled in by
        ``tools.vllm_serve.serve_roles``. Building a config before that cell has run is an error,
        not a default -- see :func:`validate_config`.
    """

    oracle: RoleBinding
    patient: RoleBinding
    judge: RoleBinding

    def as_dict(self) -> Dict[str, RoleBinding]:
        """``{role: binding}`` -- the shape ``plan_servers`` / ``serve_roles`` take."""
        return {"oracle": self.oracle, "patient": self.patient, "judge": self.judge}

    @classmethod
    def from_bindings(cls, bindings: Mapping[str, RoleBinding]) -> "RolesConfig":
        """Build from the dict ``serve_roles`` returns.

        Raises:
            KeyError: if a role is missing. Deliberately not defaulted: a missing patient binding
                would silently fall back to the default model and mislabel the arm.
        """
        return cls(oracle=bindings["oracle"], patient=bindings["patient"], judge=bindings["judge"])

    @property
    def tags(self) -> Dict[str, str]:
        """``{role: model_tag}`` -- the tags that appear in the arm name."""
        return {role: b.tag for role, b in self.as_dict().items()}

    def to_metadata(self) -> Dict[str, Any]:
        """Serialisable role table, with the tag alongside the exact model id.

        The tag is many-to-one (``gpt-4o-mini`` and ``gpt-4o-mini-2024-07-18`` both tag ``gpt4m``),
        so the folder name identifies a model FAMILY and this record identifies the snapshot.
        """
        out: Dict[str, Any] = {}
        for role, b in self.as_dict().items():
            entry = asdict(b)
            entry["tag"] = b.tag
            entry["is_local"] = b.is_local
            out[role] = entry
        return out


# ==============================================================================
#  GenConfig
# ==============================================================================


@dataclass(frozen=True)
class GenConfig:
    """Conversation-generation and data-shaping knobs.

    These govern the rollouts both methods share: how many conversations per iteration, how long,
    at what temperatures, and which slices of them are eligible to become training context.

    Attributes:
        num_conversations_per_iter: One per patient persona (96 = the full V3 permutation set).
        num_utterances_for_data: Target conversation length in utterances (therapist + patient).
        min_conv_length: ``MCL``. Slices/branches whose conversation-so-far is shorter are
            dropped. This is the response to the partial-conversation reward-faithfulness finding:
            pairwise rank agreement between a short cut and the full-conversation score is barely
            above chance at 2 utterances and only clears 0.8 at ~10. For PTO ``greedy`` it is also
            where the trunk STARTS, which is why it must be even there -- the sliced seed has to
            end on a patient turn.
        conversation_batch_size: Concurrent simulations per batch. On the local 12 GB card this is
            a SAFETY setting, not a throughput knob: weights ~2.6 GB + ~1.1 GB per concurrent
            conversation, and an over-budget request REBOOTS the machine rather than raising.
        max_prompt_tokens: ``MAX_ALLOWED_PROMPT_LENGTH`` -- the cap
            ``build_truncated_training_prompt`` enforces by dropping oldest turns. PTO needs it
            because TRL 1.4.0's ``DPOConfig`` dropped ``max_prompt_length`` and caps
            prompt+completion with one ``max_length`` under ``truncation_mode='keep_start'``,
            which slices the RESPONSE off.
        patient_concurrency: Bound on in-flight patient calls, shared by the conversation loop and
            the look-ahead rollout (one local server, one honest bound).
        stop_strings: See :data:`DEFAULT_STOP_STRINGS`.

    Notes:
        There is no ``seed`` here: the seed lives once, on the training config, and is passed down.
    """

    num_conversations_per_iter: int = 96
    num_utterances_for_data: int = 49
    min_conv_length: int = 12
    conversation_batch_size: int = 64
    temperature_therapist: float = 0.9
    temperature_patient: float = 0.7
    max_tokens_per_response: int = 200
    therapist_max_input_tokens: int = 2048
    max_prompt_tokens: int = 2048
    patient_concurrency: int = 96
    max_retries_without_progress: int = 3
    stop_strings: Tuple[str, ...] = DEFAULT_STOP_STRINGS
    verbose: bool = True
    verbose_detailed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_strings", _as_str_tuple(self.stop_strings, "stop_strings"))


# ==============================================================================
#  Training configs
# ==============================================================================


@dataclass(frozen=True)
class TrainingConfigBase:
    """Knobs both trainers share: identity, loop, optimizer, LoRA, checkpointing, capture.

    Every field carries a default, and the defaults ARE the matched grid from CLAUDE.md -- the
    builders fall back to them, so the grid is written down exactly once, here, rather than in two
    notebooks that have to agree.

    Warning:
        Frozen, but construction does NOT validate. :func:`validate_config` is the single gate and
        the builders call it; a hand-built bundle must call it too.
    """

    # -- identity ---------------------------------------------------------------
    experiment_name: str = ""
    method: str = ""
    base_model_id: str = DEFAULT_BASE_MODEL_ID
    tokenizer_id: str = ""            # "" -> base_model_id, resolved in __post_init__
    use_4bit: bool = False            # bf16 by default: no per-matmul dequant on the hot path
    seed: int = 42
    run_mode: str = "full"            # audit only -- Exp4 has NO mode_tag path level
    questionnaire_ids: Tuple[int, ...] = (1, 2)   # echo of OracleConfig; equality is enforced

    # -- loop -------------------------------------------------------------------
    num_iterations: int = 10
    # 1, matched across both methods: a GRPO "epoch" re-samples G fresh completions per prompt
    # and re-grades them, while a DPO epoch re-treads the SAME fixed pairs -- so epochs=1 is the
    # only value at which "one pass over data produced by this iteration's policy" holds for
    # both. Raise num_iterations, not this, for more updates.
    epochs_per_iteration: int = 1

    # -- optimizer --------------------------------------------------------------
    learning_rate: float = 1e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_completion_length: int = 200
    warmup_steps_ratio: float = 0.01
    eval_split_ratio: float = 0.05
    gradient_checkpointing: bool = False

    # -- LoRA -------------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES

    # -- checkpointing / logging ------------------------------------------------
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 10
    save_total_limit: Optional[int] = 2       # >= 2 so resume can walk back over a torn write
    report_to: Tuple[str, ...] = ("tensorboard",)
    push_to_hub: bool = False
    hub_entity: str = ""

    # -- capture ----------------------------------------------------------------
    save_eda_generations: bool = True
    save_lookahead_transcripts: bool = True
    tb_live_logging: bool = False
    tb_sample_completions_n: int = 8

    def __post_init__(self) -> None:
        if not self.tokenizer_id:
            object.__setattr__(self, "tokenizer_id", self.base_model_id)
        object.__setattr__(self, "questionnaire_ids",
                           _as_int_tuple(self.questionnaire_ids, "questionnaire_ids"))
        object.__setattr__(self, "lora_target_modules",
                           _as_str_tuple(self.lora_target_modules, "lora_target_modules"))
        object.__setattr__(self, "report_to", _as_str_tuple(self.report_to, "report_to"))

    @property
    def total_effective_epochs(self) -> float:
        """``num_iterations x epochs_per_iteration`` -- the headline "how much training"."""
        return float(self.num_iterations * self.epochs_per_iteration)

    @property
    def adapter_repo(self) -> str:
        """``<hub_entity>/<EXPERIMENT_NAME>``, or ``""`` when not pushing."""
        if not (self.push_to_hub and self.hub_entity):
            return ""
        return f"{self.hub_entity}/{self.experiment_name}"


@dataclass(frozen=True)
class GRPOTrainingConfig(TrainingConfigBase):
    """GRPO-specific knobs on top of :class:`TrainingConfigBase`.

    Warning:
        ``train_batch_size`` counts **completions**, not prompts, and
        ``gradient_accumulation_steps=2`` exists for the DESIGN MATCH (128 completions -> 16
        unique prompts per optimizer step, mirroring PTO's 16 pairs), not for gradient scale.
        On the pinned trl 1.4.0, ``gas`` changes are gradient-scale-neutral: trl bypasses
        transformers' ``training_step`` scaling with a non-None ``compute_loss_func`` sentinel
        and divides the loss exactly once by ``current_gradient_accumulation_steps``. (The old
        "1/gas^2, halving gas doubles the gradient" claim was Exp3's earlier stack; re-verify on
        any trl bump.) Collapsing ``gas`` also buys nothing: TRL issues ONE ``generate()`` per
        optimizer step over the whole ``generation_batch_size``, so 64x2 and 128x1 emit the same
        single call.
    """

    method: str = "GRPO"
    train_batch_size: int = 64            # completions per device
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2  # -> 128 completions -> 16 unique prompts per step
    num_generations: int = 8              # G, matched to PTO's M
    grpo_beta: float = 0.01               # KL against the iteration's reference adapter
    grpo_temperature: float = 1.2
    grpo_loss_type: str = "grpo"
    grpo_inner_iterations: int = 1
    log_completions: bool = True

    @property
    def generation_batch_size(self) -> int:
        """Completions generated per optimizer step (``per_device x gas``)."""
        return int(self.train_batch_size * self.gradient_accumulation_steps)

    @property
    def prompts_per_step(self) -> int:
        """Unique prompts per optimizer step -- ``(64/8) x 2 = 16``, matched to PTO's 16 pairs.

        This is the Phase 3 gate's number; read it from here rather than recomputing it.
        """
        return int(self.generation_batch_size // max(1, self.num_generations))


@dataclass(frozen=True)
class PTOTrainingConfig(TrainingConfigBase):
    """PTO-specific knobs on top of :class:`TrainingConfigBase`.

    Attributes:
        pref_tree_mode: ``greedy`` (true PTO -- slice an MCL-length prefix off the step-1
            conversation and grow ONE trunk by appending the best-of-M completion, so the choice
            feeds the next branch point) or ``independent`` (branch a pre-recorded conversation;
            the winner is never fed back).
        num_branches_per_turn: ``M``, matched to GRPO's ``G``.
        pref_filter_tau: Emit a pair only when ``chosen - rejected > tau``. NOT encoded in the arm
            name -- lowering it mid-arm silently mixes two configurations into one folder, so if a
            build yields zero pairs the fix is to delete the empty ``pairs.csv`` marker, never to
            drop tau.
        greedy_trunk_target_len: Trunk-length cap for ``greedy``; ``None`` means
            ``GenConfig.num_utterances_for_data``. Lowering it is a speed lever AND a science
            change (shallower trunks = shallower context), and it is not in the arm name.
        precompute_ref_log_probs: Compute reference log-probs in a no-grad pre-pass. Semantically
            identical for DPO (the reference is frozen anyway) and frees reference VRAM during the
            step.

    Warning:
        ``train_batch_size`` sizes the full-sequence LM-head logits tensor over a 128k vocab -- it
        is the memory lever, and 2 is not a placeholder. Raise ``gradient_accumulation_steps``
        instead. ``gradient_checkpointing`` defaults True here for the same reason.
    """

    method: str = "PTO"
    train_batch_size: int = 2             # pairs per device; sizes the 128k-vocab logits tensor
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # -> 16 pairs per optimizer step
    gradient_checkpointing: bool = True
    pref_tree_mode: str = "greedy"
    num_branches_per_turn: int = 8        # M
    pref_filter_tau: float = 0.1
    branch_sample_temperature: float = 1.2
    branch_max_tokens: int = 200
    dpo_beta: float = 0.1                 # DPO loss temperature, NOT GRPO's KL beta
    dpo_loss_type: str = "sigmoid"
    precompute_ref_log_probs: bool = True
    greedy_trunk_target_len: Optional[int] = None

    @property
    def mode_token(self) -> str:
        """``greedy`` | ``indep`` -- the spelling that goes into the arm name."""
        return _PTO_MODE_TOKEN.get(str(self.pref_tree_mode).strip().lower(), "")

    @property
    def pairs_per_step(self) -> int:
        """Preference pairs per optimizer step (``per_device x gas``) -- 16, matched to GRPO."""
        return int(self.train_batch_size * self.gradient_accumulation_steps)


# ==============================================================================
#  Cell-1 globals reader
# ==============================================================================


_MISSING = object()
_TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
_FALSE_STRINGS = {"false", "0", "no", "n", "off"}
_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


class _Cell1:
    """Typed, typo-aware reader over a notebook's flat globals.

    Every lookup records the name it asked for, whether or not it was present. After the config is
    built, :meth:`report_unrecognized` compares the ALL-CAPS globals that were never asked for
    against that record and warns about near-misses -- because a misspelled cell-1 global
    (``LOOKAHED_K = 5``) is otherwise completely silent: the builder uses the default, the arm name
    says ``LA0``, and the run is simply a different experiment than the one that was intended.
    """

    def __init__(self, globals_dict: Optional[Mapping[str, Any]]) -> None:
        self._d: Dict[str, Any] = dict(globals_dict or {})
        self.known: set = set()

    def _raw(self, name: str, aliases: Sequence[str] = ()) -> Any:
        self.known.add(name)
        self.known.update(aliases)
        for key in (name, *aliases):
            if key in self._d and self._d[key] is not None:
                return self._d[key]
            if key in self._d:            # present but None -- an explicit "no value"
                return None
        return _MISSING

    def has(self, name: str) -> bool:
        self.known.add(name)
        return name in self._d

    def raw(self, name: str, default: Any = None, aliases: Sequence[str] = ()) -> Any:
        value = self._raw(name, aliases)
        return default if value is _MISSING else value

    def int_(self, name: str, default: int, aliases: Sequence[str] = ()) -> int:
        value = self._raw(name, aliases)
        return default if value is _MISSING or value is None else _cast_int(name, value)

    def float_(self, name: str, default: float, aliases: Sequence[str] = ()) -> float:
        value = self._raw(name, aliases)
        return default if value is _MISSING or value is None else _cast_float(name, value)

    def bool_(self, name: str, default: bool, aliases: Sequence[str] = ()) -> bool:
        value = self._raw(name, aliases)
        return default if value is _MISSING or value is None else _cast_bool(name, value)

    def str_(self, name: str, default: str, aliases: Sequence[str] = ()) -> str:
        value = self._raw(name, aliases)
        if value is _MISSING or value is None:
            return default
        if not isinstance(value, str):
            raise ValueError(f"cell-1 global {name}={value!r} must be a string")
        return value

    def opt_int(self, name: str, default: Optional[int],
                aliases: Sequence[str] = ()) -> Optional[int]:
        """An int knob whose ``None`` is meaningful (sub-batch, save_total_limit, trunk cap)."""
        value = self._raw(name, aliases)
        if value is _MISSING:
            return default
        return None if value is None else _cast_int(name, value)

    def str_tuple(self, name: str, default: Tuple[str, ...],
                  aliases: Sequence[str] = ()) -> Tuple[str, ...]:
        value = self._raw(name, aliases)
        return default if value is _MISSING or value is None else _as_str_tuple(value, name)

    def int_tuple(self, name: str, default: Tuple[int, ...],
                  aliases: Sequence[str] = ()) -> Tuple[int, ...]:
        value = self._raw(name, aliases)
        return default if value is _MISSING or value is None else _as_int_tuple(value, name)

    def binding(self, name: str) -> Optional[RoleBinding]:
        value = self._raw(name)
        if value is _MISSING or value is None:
            return None
        if not isinstance(value, RoleBinding):
            raise ValueError(
                f"cell-1 global {name} must be a roles.RoleBinding, got {type(value).__name__}"
            )
        return value

    def report_unrecognized(self) -> List[str]:
        """Warn about ALL-CAPS globals that look like a typo of a knob we read. Returns them."""
        suspects: List[str] = []
        for key, value in self._d.items():
            if key in self.known or not _CAPS_RE.match(key):
                continue
            if not isinstance(value, (int, float, str, bool, list, tuple, type(None))):
                continue
            close = difflib.get_close_matches(key, sorted(self.known), n=1, cutoff=0.85)
            if close:
                suspects.append(key)
                _warn(
                    f"cell-1 global {key}={value!r} was never read -- did you mean {close[0]}? "
                    f"An unread knob is silent: the default is used and the run is a different "
                    f"experiment than the one you configured."
                )
        return suspects


def _cast_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"cell-1 global {name}={value!r} must be an int, not a bool")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"cell-1 global {name}={value!r} is not an int") from ex
    if as_float != int(as_float):
        raise ValueError(f"cell-1 global {name}={value!r} must be a whole number")
    return int(as_float)


def _cast_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"cell-1 global {name}={value!r} must be a number, not a bool")
    try:
        return float(value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"cell-1 global {name}={value!r} is not a number") from ex


def _cast_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
    raise ValueError(f"cell-1 global {name}={value!r} is not a boolean")


def _as_str_tuple(value: Any, what: str) -> Tuple[str, ...]:
    """Coerce a scalar or sequence to a tuple of str. Tuples keep the dataclass hashable."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(v) for v in value)
    raise ValueError(f"{what}={value!r} must be a string or a sequence of strings")


def _as_int_tuple(value: Any, what: str) -> Tuple[int, ...]:
    """Coerce a scalar, sequence or set to a tuple of int; Enum members contribute ``.value``."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{what}={value!r} must be an int or a sequence of ints")
    items = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    out: List[int] = []
    for item in items:
        raw = getattr(item, "value", item)
        if isinstance(raw, bool):
            raise ValueError(f"{what} contains a bool ({item!r}); expected ints")
        try:
            out.append(int(raw))
        except (TypeError, ValueError) as ex:
            raise ValueError(f"{what} contains {item!r}, which is not an int") from ex
    return tuple(out)


# ==============================================================================
#  Builders
# ==============================================================================


def _lazy(module_name: str, symbol: str):
    """Import ``symbol`` from ``module_name`` at call time, with a message that names the contract.

    Both ``core.oracle`` and ``core.lookahead`` are imported this way: the latter is torch-side by
    nature, and this module must stay importable by the read-only EDA.
    """
    try:
        module = __import__(module_name, fromlist=[symbol])
        return getattr(module, symbol)
    except (ImportError, AttributeError) as ex:
        raise ImportError(
            f"{module_name}.{symbol} is required to build a config bundle but could not be "
            f"imported ({ex}). It is part of the Module contract in Exp4_OpenStack/CLAUDE.md; "
            f"if it moved, this builder and that contract disagree."
        ) from ex


def _construct(cls, what: str, **kwargs):
    """Construct a dataclass owned by another module, reporting field mismatches precisely.

    A plain ``TypeError: unexpected keyword argument`` from deep inside a builder is a bad way to
    learn that a contracted field was renamed, so the field names are checked first and the error
    names both sides.
    """
    if dataclasses.is_dataclass(cls):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise TypeError(
                f"{what} does not accept {unknown} (it has {sorted(known)}). "
                f"core/config.py builds it from the Module contract in CLAUDE.md; one of the two "
                f"is out of date."
            )
    return cls(**kwargs)


def _binding_for_role(cell: _Cell1,
                      role: str,
                      *,
                      default_model: str,
                      default_timeout: float,
                      default_retries: int) -> RoleBinding:
    """Resolve one role's binding from the globals, in precedence order.

    1. ``ROLE_BINDINGS[role]`` -- the dict ``serve_roles`` returns, with ``base_url`` filled in.
       This is the intended path and the only one that can already know the port.
    2. ``<ROLE>_BINDING`` -- a hand-built :class:`roles.RoleBinding`.
    3. ``<ROLE>_MODEL_ID`` / ``<ROLE>_PROVIDER`` / ``<ROLE>_BASE_URL`` / ... -- constructed here.

    Notes:
        Path 3 leaves ``base_url`` unset for an ``openai_compat`` role unless the notebook typed
        one, and :func:`validate_config` then refuses the config with "run the serve cell". That
        refusal is the point: an unserved local role fails at config time, not 40 minutes into
        generation.
    """
    upper = role.upper()
    table = cell.raw("ROLE_BINDINGS", None)
    if isinstance(table, Mapping) and role in table:
        binding = table[role]
        if not isinstance(binding, RoleBinding):
            raise ValueError(
                f"ROLE_BINDINGS[{role!r}] must be a roles.RoleBinding, "
                f"got {type(binding).__name__}"
            )
        return binding

    explicit = cell.binding(f"{upper}_BINDING")
    if explicit is not None:
        return explicit

    provider = cell.str_(f"{upper}_PROVIDER", "openai_compat")
    if provider not in PROVIDERS:
        raise ValueError(f"{upper}_PROVIDER={provider!r} is not one of {PROVIDERS}")
    return make_binding(
        provider,
        cell.str_(f"{upper}_MODEL_ID", default_model),
        base_url=cell.raw(f"{upper}_BASE_URL", None),
        disable_thinking=cell.bool_(f"{upper}_DISABLE_THINKING", True),
        api_key_env=cell.raw(f"{upper}_API_KEY_ENV", None),
        request_timeout=cell.float_(f"{upper}_REQUEST_TIMEOUT", default_timeout),
        max_retries=cell.int_(f"{upper}_MAX_RETRIES", default_retries),
    )


def _roles_from_globals(cell: _Cell1) -> RolesConfig:
    """The three role bindings. Oracle and judge get longer timeouts than the patient: a scoring
    call emits a whole JSON rubric, a patient turn emits one utterance."""
    return RolesConfig(
        oracle=_binding_for_role(cell, "oracle", default_model=DEFAULT_ORACLE_MODEL,
                                 default_timeout=120.0, default_retries=3),
        patient=_binding_for_role(cell, "patient", default_model=DEFAULT_PATIENT_MODEL,
                                  default_timeout=90.0, default_retries=8),
        judge=_binding_for_role(cell, "judge", default_model=DEFAULT_JUDGE_MODEL,
                                default_timeout=120.0, default_retries=3),
    )


def _gen_from_globals(cell: _Cell1, base_model_id: str) -> GenConfig:
    """Generation knobs, defaulting to the matched grid on :class:`GenConfig`.

    *base_model_id* steers the ``STOP_STRINGS="auto"`` resolution: ChatML markers for the
    template-less base therapist, empty for an Instruct therapist (see
    :func:`resolve_stop_strings`).
    """
    d = GenConfig
    return GenConfig(
        num_conversations_per_iter=cell.int_("NUM_CONVERSATIONS_PER_ITER",
                                             d.num_conversations_per_iter),
        num_utterances_for_data=cell.int_("NUM_UTTERANCES_FOR_DATA", d.num_utterances_for_data),
        min_conv_length=cell.int_("MIN_CONV_LENGTH", d.min_conv_length),
        conversation_batch_size=cell.int_("CONVERSATION_BATCH_SIZE", d.conversation_batch_size),
        temperature_therapist=cell.float_("TEMPERATURE_THERAPIST_GEN", d.temperature_therapist,
                                          aliases=("TEMPERATURE_THERAPIST",)),
        temperature_patient=cell.float_("TEMPERATURE_PATIENT", d.temperature_patient),
        max_tokens_per_response=cell.int_("MAX_TOKENS_PER_RESPONSE", d.max_tokens_per_response),
        therapist_max_input_tokens=cell.int_("THERAPIST_MAX_INPUT_TOKENS",
                                             d.therapist_max_input_tokens),
        max_prompt_tokens=cell.int_("MAX_ALLOWED_PROMPT_LENGTH", d.max_prompt_tokens,
                                    aliases=("MAX_PROMPT_TOKENS",)),
        patient_concurrency=cell.int_("PATIENT_CONCURRENCY", d.patient_concurrency,
                                      aliases=("PATIENT_API_CONCURRENCY",)),
        max_retries_without_progress=cell.int_("MAX_GEN_RETRIES_WITHOUT_PROGRESS",
                                               d.max_retries_without_progress),
        stop_strings=resolve_stop_strings(cell.raw("STOP_STRINGS", "auto"), base_model_id),
        verbose=cell.bool_("GEN_VERBOSE", d.verbose),
        verbose_detailed=cell.bool_("GEN_VERBOSE_DETAILED", d.verbose_detailed),
    )


def _oracle_from_globals(cell: _Cell1, binding: RoleBinding, questionnaire_ids: Tuple[int, ...]):
    """Build ``core.oracle.OracleConfig``.

    Notes:
        ``OracleConfig.request_timeout`` bounds the scoring COROUTINE while
        ``binding.request_timeout`` bounds the socket. Both default from
        ``ORACLE_REQUEST_TIMEOUT``; keep them equal unless you have a reason, since a coroutine
        bound below the socket bound just makes retries the only real budget.
    """
    oracle_cls = _lazy("core.oracle", "OracleConfig")
    return _construct(
        oracle_cls, "core.oracle.OracleConfig",
        binding=binding,
        questionnaire_ids=questionnaire_ids,
        eval_temperature=cell.float_("EVAL_TEMPERATURE", 0.0),
        max_tokens=cell.int_("ORACLE_MAX_TOKENS", 256),
        max_retries=cell.int_("ORACLE_MAX_RETRIES", 3),
        request_timeout=cell.float_("ORACLE_REQUEST_TIMEOUT", 120.0),
        max_concurrency=cell.int_("ORACLE_MAX_CONCURRENCY", 64),
        min_success_ratio=cell.float_("ORACLE_MIN_SUCCESS_RATIO", 0.5),
    )


def _lookahead_from_globals(cell: _Cell1, patient: RoleBinding, gen: GenConfig):
    """Build ``core.lookahead.LookaheadConfig``.

    This object is serialised into ``run_metadata.json`` verbatim, which is the point: Exp3's
    ``LookaheadConfig`` was never written down, so ``lookahead_k`` and ``lookahead_sub_batch_size``
    had to be mirrored onto its ``TrainingConfig`` as audit-only duplicates -- and a mirror that is
    not kept in sync records a value the run did not use. There is no mirror here.

    ``sub_batch_size`` is not in the arm name and auto-halves (stickily) on OOM, so per-iteration
    wall-clock is only comparable between iterations that ran at the same value. That is exactly
    why it must be in the metadata.
    """
    la_cls = _lazy("core.lookahead", "LookaheadConfig")
    return _construct(
        la_cls, "core.lookahead.LookaheadConfig",
        k=cell.int_("LOOKAHEAD_K", 0),
        temperature_therapist=cell.float_("LOOKAHEAD_TEMP_THERAPIST", 0.9),
        temperature_patient=cell.float_("LOOKAHEAD_TEMP_PATIENT", 0.7),
        max_tokens=cell.int_("LOOKAHEAD_MAX_TOKENS", 200),
        max_input_tokens=cell.int_("LOOKAHEAD_MAX_INPUT_TOKENS", gen.therapist_max_input_tokens),
        patient_binding=patient,
        stop_strings=gen.stop_strings,
        sub_batch_size=cell.opt_int("LOOKAHEAD_SUB_BATCH_SIZE", 64),
    )


def _common_training_kwargs(cell: _Cell1, questionnaire_ids: Tuple[int, ...],
                            defaults: type) -> Dict[str, Any]:
    """Every :class:`TrainingConfigBase` field, read once for both methods.

    ``defaults`` is the concrete subclass, so each method's own defaults (batch sizes, grad
    checkpointing) are the fallbacks rather than the base's placeholders.
    """
    return {
        "base_model_id": cell.str_("BASE_MODEL_ID", defaults.base_model_id),
        "tokenizer_id": cell.str_("TOKENIZER_ID", ""),
        "use_4bit": cell.bool_("USE_4BIT", defaults.use_4bit),
        "seed": cell.int_("SEED", defaults.seed),
        "run_mode": cell.str_("RUN_MODE", defaults.run_mode),
        "questionnaire_ids": questionnaire_ids,
        "num_iterations": cell.int_("NUM_ITERATIONS", defaults.num_iterations),
        "epochs_per_iteration": cell.int_("EPOCHS_PER_ITERATION", defaults.epochs_per_iteration),
        "learning_rate": cell.float_("LEARNING_RATE", defaults.learning_rate),
        "train_batch_size": cell.int_("TRAIN_BATCH_SIZE", defaults.train_batch_size),
        "eval_batch_size": cell.int_("EVAL_BATCH_SIZE", defaults.eval_batch_size),
        "gradient_accumulation_steps": cell.int_("GRADIENT_ACCUMULATION_STEPS",
                                                 defaults.gradient_accumulation_steps),
        "max_completion_length": cell.int_("MAX_COMPLETION_LENGTH",
                                           defaults.max_completion_length),
        "warmup_steps_ratio": cell.float_("WARMUP_STEPS_RATIO", defaults.warmup_steps_ratio),
        "eval_split_ratio": cell.float_("EVAL_SPLIT_RATIO", defaults.eval_split_ratio),
        "gradient_checkpointing": cell.bool_("GRADIENT_CHECKPOINTING",
                                             defaults.gradient_checkpointing,
                                             aliases=("DPO_GRADIENT_CHECKPOINTING",)),
        "lora_r": cell.int_("LORA_R", defaults.lora_r),
        "lora_alpha": cell.int_("LORA_ALPHA", defaults.lora_alpha),
        "lora_dropout": cell.float_("LORA_DROPOUT", defaults.lora_dropout),
        "lora_target_modules": cell.str_tuple("LORA_TARGET_MODULES", defaults.lora_target_modules),
        "logging_steps": cell.int_("LOGGING_STEPS", defaults.logging_steps),
        "save_strategy": cell.str_("SAVE_STRATEGY", defaults.save_strategy),
        "save_steps": cell.int_("SAVE_STEPS", defaults.save_steps),
        "save_total_limit": cell.opt_int("SAVE_TOTAL_LIMIT", defaults.save_total_limit),
        "report_to": cell.str_tuple("REPORT_TO", defaults.report_to),
        "push_to_hub": cell.bool_("PUSH_TO_HUB", defaults.push_to_hub),
        "hub_entity": cell.str_("HUB_ENTITY", defaults.hub_entity),
        "save_eda_generations": cell.bool_("SAVE_EDA_GENERATIONS", defaults.save_eda_generations),
        "save_lookahead_transcripts": cell.bool_("SAVE_LOOKAHEAD_TRANSCRIPTS",
                                                 defaults.save_lookahead_transcripts),
        "tb_live_logging": cell.bool_("TB_LIVE_LOGGING", defaults.tb_live_logging),
        "tb_sample_completions_n": cell.int_("TB_SAMPLE_COMPLETIONS_N",
                                             defaults.tb_sample_completions_n),
    }


def _paths_from_globals(cell: _Cell1, experiment_name: str) -> RunPaths:
    """``RunPaths`` from ``DATA_ROOT`` / ``WORKSPACE_ROOT`` if given, else by resolving the root."""
    return RunPaths.from_workspace(
        experiment_name,
        workspace_root=cell.raw("WORKSPACE_ROOT", None),
        data_root=cell.raw("DATA_ROOT", None),
    )


def _check_name_not_typed(cell: _Cell1, computed: str) -> None:
    """Warn if the globals carry an ``EXPERIMENT_NAME`` that disagrees with the computed one.

    The typed value is never used. A notebook that assigns ``EXPERIMENT_NAME = cfg.experiment_name``
    after building (so later cells can print it) is fine and silent; a stale hand-typed one gets
    called out, because in Exp3 that exact string is what a run was filed under.
    """
    typed = cell.raw("EXPERIMENT_NAME", None)
    if isinstance(typed, str) and typed and typed != computed:
        _warn(
            f"cell-1 defines EXPERIMENT_NAME={typed!r}, which is IGNORED. The name is computed "
            f"from the config: {computed!r}. Delete the assignment (or let the notebook set it "
            f"from the returned config) so the two cannot disagree."
        )


def build_grpo_config(globals_dict: dict, *, verbose: bool = True) -> Tuple[
        GRPOTrainingConfig, RolesConfig, GenConfig, "OracleConfig", "LookaheadConfig", RunPaths]:
    """Freeze a GRPO notebook's cell-1 globals into the config bundle.

    Args:
        globals_dict: Usually ``globals()``. Only recognised ALL-CAPS names are read; unread
            near-misses are reported (see :class:`_Cell1`).
        verbose: Print the configuration summary. Additive to the contracted signature --
            ``build_grpo_config(globals())`` behaves exactly as specified.

    Returns:
        ``(training, roles, generation, oracle, lookahead, paths)``.

    Raises:
        ValueError: on an unparseable knob, an unmapped questionnaire set, or any
            :func:`validate_config` failure.
        ImportError: if ``core.oracle`` / ``core.lookahead`` are unavailable.

    Notes:
        ``EXPERIMENT_NAME`` is COMPUTED here from the questionnaire set, K, MCL, G and the
        oracle, patient and therapist model ids. It is never read from *globals_dict*. That is the structural fix for
        Exp3's "changed ORACLE_MODEL_ID but left EXPERIMENT_NAME alone" failure, which Exp3 could
        only guard with a runtime assertion.

        Call this AFTER ``serve_roles()`` (notebook cell 3). Local roles have no ``base_url``
        before it runs, and this refuses to build a config that cannot reach its own oracle.
    """
    cell = _Cell1(globals_dict)
    roles = _roles_from_globals(cell)
    base_model_id = cell.str_("BASE_MODEL_ID", GRPOTrainingConfig.base_model_id)
    gen = _gen_from_globals(cell, base_model_id)
    qids = cell.int_tuple("QUESTIONNAIRE_IDS", TrainingConfigBase.questionnaire_ids)

    oracle_cfg = _oracle_from_globals(cell, roles.oracle, qids)
    la_cfg = _lookahead_from_globals(cell, roles.patient, gen)

    num_generations = cell.int_("NUM_GENERATIONS", GRPOTrainingConfig.num_generations)
    experiment_name = build_experiment_name(
        "GRPO", qids, la_cfg.k, gen.min_conv_length,
        g=num_generations,
        oracle_model=roles.oracle.model,
        patient_model=roles.patient.model,
        therapist_model=base_model_id,
    )
    _check_name_not_typed(cell, experiment_name)

    train = GRPOTrainingConfig(
        experiment_name=experiment_name,
        num_generations=num_generations,
        grpo_beta=cell.float_("GRPO_BETA", GRPOTrainingConfig.grpo_beta),
        grpo_temperature=cell.float_("GRPO_TEMPERATURE", GRPOTrainingConfig.grpo_temperature),
        grpo_loss_type=cell.str_("GRPO_LOSS_TYPE", GRPOTrainingConfig.grpo_loss_type),
        grpo_inner_iterations=cell.int_("GRPO_INNER_ITERATIONS",
                                        GRPOTrainingConfig.grpo_inner_iterations),
        log_completions=cell.bool_("LOG_COMPLETIONS", GRPOTrainingConfig.log_completions),
        **_common_training_kwargs(cell, qids, GRPOTrainingConfig),
    )

    paths = _paths_from_globals(cell, experiment_name)
    cell.report_unrecognized()
    validate_config(train, roles, gen, oracle_cfg, la_cfg, paths)
    if verbose:
        print(format_summary(train, roles, gen, oracle_cfg, la_cfg, paths))
    return train, roles, gen, oracle_cfg, la_cfg, paths


def build_pto_config(globals_dict: dict, *, verbose: bool = True) -> Tuple[
        PTOTrainingConfig, RolesConfig, GenConfig, "OracleConfig", "LookaheadConfig", RunPaths]:
    """Freeze a PTO notebook's cell-1 globals into the config bundle.

    Same contract as :func:`build_grpo_config`; the extra knobs are ``PREF_TREE_MODE``,
    ``NUM_BRANCHES_PER_TURN``, ``PREF_FILTER_TAU``, ``BRANCH_SAMPLE_TEMPERATURE``,
    ``BRANCH_MAX_TOKENS``, ``DPO_BETA``, ``DPO_LOSS_TYPE``, ``DPO_PRECOMPUTE_REF_LOGPS`` and
    ``GREEDY_TRUNK_TARGET_LEN``.

    Notes:
        ``greedy`` mode requires an EVEN ``MIN_CONV_LENGTH`` -- the trunk seed is sliced off the
        step-1 conversation and must end on a patient turn. :func:`validate_config` enforces it;
        :func:`naming.build_experiment_name` deliberately does not, so that a name this builder
        refuses to create is still parseable if it exists on disk.
    """
    cell = _Cell1(globals_dict)
    roles = _roles_from_globals(cell)
    base_model_id = cell.str_("BASE_MODEL_ID", PTOTrainingConfig.base_model_id)
    gen = _gen_from_globals(cell, base_model_id)
    qids = cell.int_tuple("QUESTIONNAIRE_IDS", TrainingConfigBase.questionnaire_ids)

    oracle_cfg = _oracle_from_globals(cell, roles.oracle, qids)
    la_cfg = _lookahead_from_globals(cell, roles.patient, gen)

    num_branches = cell.int_("NUM_BRANCHES_PER_TURN", PTOTrainingConfig.num_branches_per_turn)
    mode = cell.str_("PREF_TREE_MODE", PTOTrainingConfig.pref_tree_mode).strip().lower()
    if mode not in _PTO_MODE_TOKEN:
        raise ValueError(f"PREF_TREE_MODE={mode!r} must be one of {_PTO_MODES}")

    experiment_name = build_experiment_name(
        "PTO", qids, la_cfg.k, gen.min_conv_length,
        m=num_branches, mode=mode,
        oracle_model=roles.oracle.model,
        patient_model=roles.patient.model,
        therapist_model=base_model_id,
    )
    _check_name_not_typed(cell, experiment_name)

    train = PTOTrainingConfig(
        experiment_name=experiment_name,
        pref_tree_mode=mode,
        num_branches_per_turn=num_branches,
        pref_filter_tau=cell.float_("PREF_FILTER_TAU", PTOTrainingConfig.pref_filter_tau),
        branch_sample_temperature=cell.float_("BRANCH_SAMPLE_TEMPERATURE",
                                              PTOTrainingConfig.branch_sample_temperature),
        branch_max_tokens=cell.int_("BRANCH_MAX_TOKENS", PTOTrainingConfig.branch_max_tokens),
        dpo_beta=cell.float_("DPO_BETA", PTOTrainingConfig.dpo_beta),
        dpo_loss_type=cell.str_("DPO_LOSS_TYPE", PTOTrainingConfig.dpo_loss_type),
        precompute_ref_log_probs=cell.bool_("DPO_PRECOMPUTE_REF_LOGPS",
                                            PTOTrainingConfig.precompute_ref_log_probs),
        greedy_trunk_target_len=cell.opt_int("GREEDY_TRUNK_TARGET_LEN",
                                             PTOTrainingConfig.greedy_trunk_target_len),
        **_common_training_kwargs(cell, qids, PTOTrainingConfig),
    )

    paths = _paths_from_globals(cell, experiment_name)
    cell.report_unrecognized()
    validate_config(train, roles, gen, oracle_cfg, la_cfg, paths)
    if verbose:
        print(format_summary(train, roles, gen, oracle_cfg, la_cfg, paths))
    return train, roles, gen, oracle_cfg, la_cfg, paths


# ==============================================================================
#  Validation
# ==============================================================================


def _bundle(cfgs: Sequence[Any]) -> Dict[str, Any]:
    """Index a loose bundle of config objects by kind.

    Dispatch is on ``type(cfg).__name__``, not ``isinstance``. Importing ``core.lookahead`` just to
    type-check an argument would drag torch into a read-only process, which is exactly what this
    module refuses to do.
    """
    kinds = {
        "GRPOTrainingConfig": "training",
        "PTOTrainingConfig": "training",
        "TrainingConfigBase": "training",
        "RolesConfig": "roles",
        "GenConfig": "generation",
        "OracleConfig": "oracle",
        "LookaheadConfig": "lookahead",
        "RunPaths": "paths",
    }
    out: Dict[str, Any] = {}
    for cfg in cfgs:
        if cfg is None:
            continue
        kind = kinds.get(type(cfg).__name__)
        if kind is None:
            raise TypeError(
                f"validate_config/config_to_metadata got a {type(cfg).__name__}, which is not an "
                f"Exp4 config object (expected one of {sorted(set(kinds))})"
            )
        out[kind] = cfg
    return out


def _binding_errors(binding: RoleBinding, role: str,
                    seen: Optional[set] = None) -> List[str]:
    """Per-binding checks, including the one that stops a run before it starts.

    Args:
        seen: Bindings already reported. ``OracleConfig`` and ``LookaheadConfig`` each hold a
            binding that is normally the very same frozen object as one in :class:`RolesConfig`,
            and reporting one unserved server five times buries the other four errors. Passing the
            set makes each distinct binding report once.
    """
    if seen is not None:
        if binding in seen:
            return []
        seen.add(binding)
    errs: List[str] = []
    if not getattr(binding, "model", ""):
        errs.append(f"{role} binding has no model id")
    if binding.provider not in PROVIDERS:
        errs.append(f"{role} provider {binding.provider!r} is not one of {PROVIDERS}")
    if binding.is_local and not binding.base_url:
        errs.append(
            f"{role} is bound to {binding.model!r} via openai_compat but base_url is unset -- "
            f"the vLLM server has not been started. Run the serve_roles() cell (notebook cell 3, "
            f"BEFORE any torch import) and build the config from the bindings it returns."
        )
    if float(binding.request_timeout) <= 0:
        errs.append(f"{role} request_timeout must be > 0 (it is PER ATTEMPT)")
    if int(binding.max_retries) < 1:
        errs.append(f"{role} max_retries must be >= 1")
    try:
        binding.extra_body  # noqa: B018 - property raises on malformed JSON, which is the check
    except ValueError as ex:
        errs.append(f"{role} extra_body_json is malformed: {ex}")
    if not binding.is_local and binding.extra_body_json:
        _warn(
            f"{role} binding is provider={binding.provider!r} and carries an extra body; vendor "
            f"APIs return 400 on unknown body keys, and that arrives on the first real call."
        )
    return errs


def _roles_errors(roles: RolesConfig, seen: Optional[set] = None) -> List[str]:
    errs: List[str] = []
    for role, binding in roles.as_dict().items():
        errs.extend(_binding_errors(binding, role, seen))

    # Both trainers thread exactly ONE async client through run_one_iteration and use it for the
    # oracle AND every patient call (conversation turns and look-ahead alike). A split stack is
    # therefore not expressible: whichever endpoint the client was built from would receive both
    # roles' requests, so one of them asks a server for a model it does not serve -- 404 per turn,
    # retried max_retries times, and only after the base model has been downloaded and loaded.
    # The arm-name grammar CAN spell a split stack; the v1 trainers cannot run one. Refuse it at
    # config time. (The judge is exempt: it is never called by a trainer -- the EDA builds its own
    # client from RolesConfig.judge.)
    if (roles.oracle.provider, roles.oracle.base_url) != (roles.patient.provider,
                                                          roles.patient.base_url):
        errs.append(
            f"oracle ({roles.oracle.provider} @ {roles.oracle.base_url}) and patient "
            f"({roles.patient.provider} @ {roles.patient.base_url}) resolve to different "
            f"endpoints. run_one_iteration takes ONE client and uses it for both roles, so one of "
            f"them would be sent to the wrong server. Bind both to the same endpoint."
        )
    return errs


def _gen_errors(gen: GenConfig) -> List[str]:
    errs: List[str] = []
    if gen.min_conv_length < 2:
        errs.append(f"min_conv_length ({gen.min_conv_length}) must be >= 2 (smallest viable slice)")
    if gen.num_conversations_per_iter <= 0:
        errs.append("num_conversations_per_iter must be > 0")
    if gen.num_utterances_for_data <= 0:
        errs.append("num_utterances_for_data must be > 0")
    if gen.conversation_batch_size <= 0:
        errs.append("conversation_batch_size must be > 0")
    if gen.max_tokens_per_response <= 0:
        errs.append("max_tokens_per_response must be > 0")
    if gen.therapist_max_input_tokens <= 0 or gen.max_prompt_tokens <= 0:
        errs.append("therapist_max_input_tokens and max_prompt_tokens must be > 0")
    if gen.patient_concurrency <= 0:
        errs.append("patient_concurrency must be > 0")
    if gen.max_retries_without_progress < 0:
        errs.append("max_retries_without_progress must be >= 0")
    for name in ("temperature_therapist", "temperature_patient"):
        value = float(getattr(gen, name))
        if value < 0:
            errs.append(f"{name} ({value}) must be >= 0")
    if gen.min_conv_length > gen.num_utterances_for_data:
        errs.append(
            f"min_conv_length ({gen.min_conv_length}) exceeds num_utterances_for_data "
            f"({gen.num_utterances_for_data}) -- every slice would be filtered out and the "
            f"iteration would produce zero training rows"
        )
    # Empty stop_strings is judged in _cross_errors, where the therapist variant is known:
    # required for the template-less base, correct (and free) for Instruct.
    return errs


def _oracle_errors(oracle: Any, seen: Optional[set] = None) -> List[str]:
    errs: List[str] = []
    errs.extend(_binding_errors(oracle.binding, "oracle(OracleConfig)", seen))
    try:
        qtag_for(oracle.questionnaire_ids)
    except ValueError as ex:
        errs.append(str(ex))
    if not (0.0 < float(oracle.min_success_ratio) <= 1.0):
        errs.append(f"oracle min_success_ratio ({oracle.min_success_ratio}) must be in (0, 1]")
    if int(oracle.max_concurrency) <= 0:
        errs.append("oracle max_concurrency must be > 0")
    if int(oracle.max_retries) < 1:
        errs.append("oracle max_retries must be >= 1")
    if float(oracle.request_timeout) <= 0:
        errs.append("oracle request_timeout must be > 0")
    if int(oracle.max_tokens) <= 0:
        errs.append("oracle max_tokens must be > 0")
    if float(oracle.eval_temperature) < 0:
        errs.append("oracle eval_temperature must be >= 0")
    elif float(oracle.eval_temperature) > 0:
        _warn(
            f"oracle eval_temperature={oracle.eval_temperature} > 0: the grader becomes a random "
            f"variable, so a re-score of the same transcript will not reproduce."
        )
    return errs


def _lookahead_errors(la: Any, seen: Optional[set] = None) -> List[str]:
    errs: List[str] = []
    if int(la.k) < 0:
        errs.append("lookahead k must be >= 0 (0 disables look-ahead)")
    if int(la.max_tokens) <= 0 or int(la.max_input_tokens) <= 0:
        errs.append("lookahead max_tokens and max_input_tokens must be > 0")
    for name in ("temperature_therapist", "temperature_patient"):
        if float(getattr(la, name)) < 0:
            errs.append(f"lookahead {name} must be >= 0")
    sub = getattr(la, "sub_batch_size", None)
    if sub is not None and int(sub) <= 0:
        errs.append("lookahead sub_batch_size must be > 0 or None (None = one padded generate)")
    if int(la.k) > 0:
        binding = getattr(la, "patient_binding", None)
        if binding is None:
            errs.append("lookahead k > 0 but patient_binding is None -- nothing can reply")
        else:
            errs.extend(_binding_errors(binding, "patient(LookaheadConfig)", seen))
    return errs


def _training_errors(train: TrainingConfigBase) -> List[str]:
    errs: List[str] = []
    if not train.experiment_name:
        errs.append("experiment_name is empty -- it must be computed by naming.build_experiment_name")
    else:
        try:
            parse_experiment_name(train.experiment_name)
        except ValueError as ex:
            errs.append(str(ex))
    if train.num_iterations <= 0 or train.epochs_per_iteration <= 0:
        errs.append("num_iterations and epochs_per_iteration must be > 0")
    if train.learning_rate <= 0:
        errs.append("learning_rate must be > 0")
    if train.train_batch_size <= 0 or train.eval_batch_size <= 0:
        errs.append("train_batch_size and eval_batch_size must be > 0")
    if train.gradient_accumulation_steps < 1:
        errs.append("gradient_accumulation_steps must be >= 1")
    if train.max_completion_length <= 0:
        errs.append("max_completion_length must be > 0")
    if not (0.0 < train.eval_split_ratio < 1.0):
        errs.append(f"eval_split_ratio ({train.eval_split_ratio}) must be in (0, 1)")
    if not (0.0 <= train.warmup_steps_ratio < 1.0):
        errs.append(f"warmup_steps_ratio ({train.warmup_steps_ratio}) must be in [0, 1)")
    if train.save_strategy not in _SAVE_STRATEGIES:
        errs.append(f"save_strategy ({train.save_strategy!r}) must be one of {_SAVE_STRATEGIES}")
    if train.save_strategy == "steps" and train.save_steps <= 0:
        errs.append(f"save_steps ({train.save_steps}) must be > 0 when save_strategy='steps'")
    if train.logging_steps <= 0:
        errs.append("logging_steps must be > 0")
    if train.save_total_limit is not None and train.save_total_limit < 1:
        errs.append("save_total_limit must be >= 1 or None")
    if train.lora_r <= 0 or train.lora_alpha <= 0:
        errs.append("lora_r and lora_alpha must be > 0")
    if not (0.0 <= train.lora_dropout < 1.0):
        errs.append("lora_dropout must be in [0, 1)")
    if not train.lora_target_modules:
        errs.append("lora_target_modules is empty -- nothing would train")
    if train.seed < 0:
        errs.append("seed must be >= 0")
    bad_targets = [t for t in train.report_to if t not in _REPORT_TARGETS]
    if bad_targets:
        errs.append(f"report_to contains {bad_targets}; expected a subset of {_REPORT_TARGETS}")
    if train.push_to_hub and not train.hub_entity:
        errs.append("push_to_hub is True but hub_entity is empty -- there is nowhere to push")

    if train.save_total_limit == 1:
        _warn(
            "save_total_limit=1: a process killed DURING a checkpoint write leaves the only "
            "checkpoint half-written, and resume then loses the whole iteration. Keep >= 2 so "
            "get_latest_valid_hf_checkpoint has something to walk back to."
        )
    if "wandb" in train.report_to:
        _warn(
            "report_to includes 'wandb', but Exp4 logs to TensorBoard only (see CLAUDE.md) -- "
            "nothing here creates or groups a W&B run."
        )
    return errs


def _grpo_errors(train: GRPOTrainingConfig) -> List[str]:
    errs: List[str] = []
    if train.num_generations < 2:
        errs.append("num_generations must be >= 2 (a group of one has no group-relative advantage)")
    if train.num_generations > 0:
        if train.train_batch_size % train.num_generations != 0:
            errs.append(
                f"train_batch_size ({train.train_batch_size}) must be divisible by "
                f"num_generations ({train.num_generations}) -- it counts COMPLETIONS, not prompts"
            )
        if train.eval_batch_size % train.num_generations != 0:
            errs.append(
                f"eval_batch_size ({train.eval_batch_size}) must be divisible by "
                f"num_generations ({train.num_generations})"
            )
    if train.grpo_beta < 0:
        errs.append("grpo_beta (KL coefficient) must be >= 0")
    if train.grpo_temperature <= 0:
        errs.append("grpo_temperature must be > 0")
    if train.grpo_inner_iterations < 1:
        errs.append("grpo_inner_iterations must be >= 1")

    if train.gradient_accumulation_steps == 1:
        _warn(
            "GRPO gradient_accumulation_steps=1. On the pinned trl 1.4.0 this is gradient-scale-"
            "neutral (trl bypasses transformers' scaling and divides once itself), but it halves "
            "the unique prompts per optimizer step -- 64x2 keeps prompts/step=16, matched to "
            "PTO's 16 pairs -- and buys no throughput: TRL emits ONE generate() per optimizer "
            "step either way. Use per_device=64 x gas=2 unless you mean to change the match."
        )
    return errs


def _pto_errors(train: PTOTrainingConfig) -> List[str]:
    errs: List[str] = []
    if str(train.pref_tree_mode).strip().lower() not in _PTO_MODE_TOKEN:
        errs.append(f"pref_tree_mode ({train.pref_tree_mode!r}) must be one of {_PTO_MODES}")
    if train.num_branches_per_turn < 2:
        errs.append(
            f"num_branches_per_turn ({train.num_branches_per_turn}) must be >= 2 -- a preference "
            f"pair needs a best AND a worst"
        )
    if train.pref_filter_tau < 0:
        errs.append(f"pref_filter_tau ({train.pref_filter_tau}) must be >= 0")
    if train.branch_sample_temperature <= 0:
        errs.append("branch_sample_temperature must be > 0 (M identical candidates yield no pair)")
    if train.branch_max_tokens <= 0:
        errs.append("branch_max_tokens must be > 0")
    if train.dpo_beta <= 0:
        errs.append("dpo_beta must be > 0")
    if not train.dpo_loss_type:
        errs.append("dpo_loss_type is empty")
    if train.greedy_trunk_target_len is not None and train.greedy_trunk_target_len <= 0:
        errs.append("greedy_trunk_target_len must be > 0 or None")
    return errs


def _cross_errors(b: Dict[str, Any]) -> List[str]:
    """Rules that need two configs at once. Skipped when only one was supplied."""
    errs: List[str] = []
    train = b.get("training")
    gen = b.get("generation")
    roles = b.get("roles")
    oracle = b.get("oracle")
    la = b.get("lookahead")

    if train is not None and gen is not None and getattr(train, "method", "") == "PTO":
        mode = str(train.pref_tree_mode).strip().lower()
        if mode == "greedy" and gen.min_conv_length % 2 != 0:
            errs.append(
                f"pref_tree_mode='greedy' needs an EVEN min_conv_length (got "
                f"{gen.min_conv_length}) -- the trunk seed is sliced off the step-1 conversation "
                f"and must end on a patient turn"
            )
        target = train.greedy_trunk_target_len or gen.num_utterances_for_data
        if mode == "greedy" and target <= gen.min_conv_length:
            errs.append(
                f"greedy trunk target ({target}) must exceed min_conv_length "
                f"({gen.min_conv_length}) -- the trunk starts at MCL and has to grow"
            )

    if train is not None and gen is not None:
        native = _therapist_has_native_template(train.base_model_id)
        if not native and not gen.stop_strings:
            errs.append(
                f"stop_strings is empty but the therapist ({train.base_model_id!r}) is a BASE "
                f"model on the hand-written ChatML template: the markers are ordinary BPE "
                f"pieces, not special tokens, so nothing would cut a self-played completion. "
                f"Set STOP_STRINGS='auto' (or the ChatML pair) for this therapist."
            )
        if native and gen.stop_strings:
            _warn(
                f"stop_strings={list(gen.stop_strings)} on the Instruct therapist "
                f"({train.base_model_id!r}). Its native template ends turns with the special "
                f"<|eot_id|>, so string stopping buys nothing and costs a criteria-table build "
                f"per generate() call. STOP_STRINGS='auto' resolves to () here."
            )

    if train is not None and oracle is not None:
        if tuple(train.questionnaire_ids) != tuple(oracle.questionnaire_ids):
            errs.append(
                f"training.questionnaire_ids {tuple(train.questionnaire_ids)} != "
                f"oracle.questionnaire_ids {tuple(oracle.questionnaire_ids)}; the training reward "
                f"and the arm name would describe different rubrics"
            )

    if roles is not None and oracle is not None and oracle.binding.model != roles.oracle.model:
        errs.append(
            f"OracleConfig.binding is {oracle.binding.model!r} but RolesConfig.oracle is "
            f"{roles.oracle.model!r}; the arm name encodes the latter, so the recorded grader "
            f"would not be the one that scored"
        )

    if roles is not None and la is not None and int(getattr(la, "k", 0)) > 0:
        binding = getattr(la, "patient_binding", None)
        if binding is not None and binding.model != roles.patient.model:
            _warn(
                f"look-ahead patient is {binding.model!r} but the conversation patient is "
                f"{roles.patient.model!r}: the rollout would be scored against a different "
                f"simulator than the one the policy is trained to talk to."
            )

    # The name is computed, so this can only fire for a hand-assembled bundle -- but it is the one
    # check that proves the folder on disk describes the config that wrote it.
    if all(x is not None for x in (train, gen, roles, oracle, la)):
        try:
            expected = build_experiment_name(
                train.method, oracle.questionnaire_ids, int(la.k), gen.min_conv_length,
                g=getattr(train, "num_generations", None) if train.method == "GRPO" else None,
                m=getattr(train, "num_branches_per_turn", None) if train.method == "PTO" else None,
                mode=getattr(train, "pref_tree_mode", None) if train.method == "PTO" else None,
                oracle_model=roles.oracle.model,
                patient_model=roles.patient.model,
                therapist_model=train.base_model_id,
            )
        except ValueError as ex:
            errs.append(f"could not recompute the arm name from this bundle: {ex}")
        else:
            if expected != train.experiment_name:
                errs.append(
                    f"experiment_name {train.experiment_name!r} does not match the config it "
                    f"claims to describe (recomputed: {expected!r}). The name must be COMPUTED by "
                    f"naming.build_experiment_name, never typed."
                )

    paths = b.get("paths")
    if paths is not None and train is not None and paths.experiment_name != train.experiment_name:
        errs.append(
            f"RunPaths.experiment_name {paths.experiment_name!r} != training config "
            f"{train.experiment_name!r}; artifacts would land in another arm's folder"
        )
    return errs


def validate_config(cfg: Any, *more: Any) -> None:
    """Check one config object, or a whole bundle, and raise once with every problem found.

    Args:
        cfg: Any Exp4 config object (:class:`GRPOTrainingConfig`, :class:`PTOTrainingConfig`,
            :class:`RolesConfig`, :class:`GenConfig`, ``OracleConfig``, ``LookaheadConfig``,
            :class:`RunPaths`).
        *more: Additional config objects from the same bundle.

    Raises:
        ValueError: listing every failed check. Errors are collected rather than raised one at a
            time so a cell-1 edit fixes all of them in one pass.
        TypeError: if an argument is not an Exp4 config object.

    Notes:
        **Cross-config rules run only when both halves are supplied.** The PTO greedy/even-MCL
        rule needs the training config AND the generation config; the arm-name recomputation needs
        five. The builders always pass the whole bundle, which is why they are the intended entry
        point -- validating a single object is a partial check by construction.

        Some findings are WARNINGS, not errors: a collapsed GRPO ``gradient_accumulation_steps``,
        ``save_total_limit=1``, a non-zero oracle temperature, a W&B report target. They are
        printed, and the config is still returned, because each of them is a legitimate choice
        somebody might make deliberately.
    """
    b = _bundle([cfg, *more])
    errors: List[str] = []
    seen_bindings: set = set()

    train = b.get("training")
    if train is not None:
        errors.extend(_training_errors(train))
        if type(train).__name__ == "GRPOTrainingConfig":
            errors.extend(_grpo_errors(train))
        elif type(train).__name__ == "PTOTrainingConfig":
            errors.extend(_pto_errors(train))
    # Roles first, so a binding shared with OracleConfig/LookaheadConfig reports under its role
    # name rather than under whichever config happened to be checked first.
    if "roles" in b:
        errors.extend(_roles_errors(b["roles"], seen_bindings))
    if "generation" in b:
        errors.extend(_gen_errors(b["generation"]))
    if "oracle" in b:
        errors.extend(_oracle_errors(b["oracle"], seen_bindings))
    if "lookahead" in b:
        errors.extend(_lookahead_errors(b["lookahead"], seen_bindings))
    errors.extend(_cross_errors(b))

    unique = list(dict.fromkeys(errors))          # order-preserving dedupe
    if unique:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(unique))


# ==============================================================================
#  Metadata
# ==============================================================================


def _resolve_path(payload: Mapping[str, Any], dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def config_to_metadata(*cfgs: Any) -> Dict[str, Any]:
    """Build the ``run_metadata.json`` payload from a config bundle.

    Args:
        *cfgs: The bundle returned by a builder, in any order.

    Returns:
        A JSON-serialisable dict with ``config`` sections ``training`` / ``generation`` /
        ``roles`` / ``oracle`` / ``lookahead`` / ``paths``, plus the decoded arm, the derived
        batch arithmetic, and provenance (host, pid, time).

    Raises:
        RuntimeError: if a complete bundle produced a payload that is missing one of
            :data:`SILENTLY_MUTABLE_KNOBS`.

    Notes:
        **This file is the only record of every knob NOT encoded in the arm name.** Change
        ``num_iterations``, ``pref_filter_tau``, ``lookahead.sub_batch_size``, a temperature, the
        learning rate, the LoRA settings, a batch size, ``epochs_per_iteration`` or
        ``greedy_trunk_target_len`` and the folder name is byte-identical. Hence the completeness
        assertion: it runs whenever all six sections are present, so a section that stops being
        serialised fails loudly instead of silently losing the run's provenance.

        ``LookaheadConfig`` is serialised directly. Exp3 mirrored ``lookahead_k`` and
        ``lookahead_sub_batch_size`` onto its ``TrainingConfig`` because its ``LookaheadConfig``
        never reached disk; there is no mirror here, and therefore nothing to drift.

        Values are passed through ``recorder.to_jsonable``, which is total -- an unrecognised
        object degrades to ``str`` rather than taking down a run several GPU-hours in.
    """
    b = _bundle(list(cfgs))
    train = b.get("training")
    paths = b.get("paths")

    config: Dict[str, Any] = {}
    if train is not None:
        config["training"] = asdict(train)
    if "generation" in b:
        config["generation"] = asdict(b["generation"])
    if "roles" in b:
        config["roles"] = b["roles"].to_metadata()
    if "oracle" in b:
        config["oracle"] = asdict(b["oracle"])
        # A module-level flag, not a dataclass field -- record it here or a flipped strictness
        # (set_openai_compat_strict) leaves no trace in the run's provenance.
        from core.oracle import openai_compat_strict
        config["oracle"]["openai_compat_strict"] = bool(openai_compat_strict())
    if "lookahead" in b:
        config["lookahead"] = asdict(b["lookahead"])
    if paths is not None:
        config["paths"] = paths.describe()

    experiment_name = getattr(train, "experiment_name", None) or getattr(
        paths, "experiment_name", "")
    method = getattr(train, "method", "")

    arm: Optional[Dict[str, Any]] = None
    arm_error: Optional[str] = None
    if experiment_name:
        try:
            info = parse_experiment_name(experiment_name)
            arm = asdict(info)
            arm["label"] = info.label
            arm["branches"] = info.branches
        except ValueError as ex:
            arm_error = str(ex)

    derived: Dict[str, Any] = {}
    if train is not None:
        derived["total_effective_epochs"] = train.total_effective_epochs
        derived["adapter_repo"] = train.adapter_repo
        if method == "GRPO":
            derived["generation_batch_size"] = train.generation_batch_size
            derived["prompts_per_step"] = train.prompts_per_step
        elif method == "PTO":
            derived["pairs_per_step"] = train.pairs_per_step
            derived["mode_token"] = train.mode_token

    payload: Dict[str, Any] = {
        "schema": METADATA_SCHEMA,
        "grammar_version": GRAMMAR_VERSION,
        "experiment_name": experiment_name,
        "method": method,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "arm": arm,
        "config": config,
        "derived": derived,
        "silently_mutable_knobs": list(_silent_knobs_for(method)),
    }
    if arm_error:
        payload["arm_parse_error"] = arm_error

    payload = to_jsonable(payload)
    _assert_knob_coverage(payload, method, complete=set(config) >= {
        "training", "generation", "roles", "oracle", "lookahead", "paths"})
    return payload


def _silent_knobs_for(method: str) -> Tuple[str, ...]:
    extra = _SILENT_GRPO if method == "GRPO" else (_SILENT_PTO if method == "PTO" else ())
    return _SILENT_COMMON + extra


def _assert_knob_coverage(payload: Mapping[str, Any], method: str, *, complete: bool) -> None:
    """Fail loudly if a documented silently-mutable knob did not reach the payload."""
    if not complete:
        return
    missing = [path for path in _silent_knobs_for(method)
               if _resolve_path(payload.get("config", {}), path) is _MISSING]
    if missing:
        raise RuntimeError(
            "run_metadata payload is missing knobs that are NOT encoded in EXPERIMENT_NAME: "
            f"{missing}. These are the only record distinguishing two runs of the same arm, so "
            "the payload must carry them. Either a config field was renamed or a section stopped "
            "being serialised; fix core/config.py's SILENTLY_MUTABLE_KNOBS and the section that "
            "should contain them."
        )


def write_run_metadata(payload: Mapping[str, Any], paths: RunPaths) -> str:
    """Write ``run_metadata.json`` and append the same payload to ``run_metadata_history.jsonl``.

    Args:
        payload: The dict from :func:`config_to_metadata`.
        paths: The run's :class:`RunPaths`.

    Returns:
        The path of the current-metadata file.

    Notes:
        Exp3 fix #5. The current file is overwritten (a reader wants exactly one "what is this arm
        configured as?"), but the superseded payload is never lost: every process appends its own
        line to the history log first. Exp3 overwrote in place with no history, so a resume under
        changed knobs restamped the whole arm, earlier iterations included, and the values those
        iterations actually ran under were gone.

        The current file is written to a temp name and moved into place, so a crash mid-write
        cannot leave an unparseable ``run_metadata.json`` behind.
    """
    paths.ensure_run_dir()
    line = json.dumps(payload, allow_nan=False)
    try:
        with open(paths.run_metadata_history_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as ex:                       # history is provenance, not control flow
        _warn(f"could not append run_metadata history ({ex})")

    tmp = paths.run_metadata_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    os.replace(tmp, paths.run_metadata_path)
    return paths.run_metadata_path


# ==============================================================================
#  Summary
# ==============================================================================


def format_summary(*cfgs: Any) -> str:
    """Human-readable configuration summary -- what the builders print.

    Shows the numbers that are easy to get wrong and expensive to discover late: the computed arm
    name, the batch arithmetic (completions -> prompts/step), the endpoints each role resolved to,
    and where the artifacts will land.
    """
    b = _bundle(list(cfgs))
    train = b.get("training")
    gen = b.get("generation")
    roles = b.get("roles")
    oracle = b.get("oracle")
    la = b.get("lookahead")
    paths = b.get("paths")

    name = getattr(train, "experiment_name", None) or getattr(paths, "experiment_name", "?")
    rule = "=" * 78
    out: List[str] = [rule, f"EXP4 CONFIG  {name}", rule]

    if train is not None:
        out.append(f"  method       {train.method}"
                   f"    K {getattr(la, 'k', '?')}"
                   f"    MCL {getattr(gen, 'min_conv_length', '?')}"
                   f"    run_mode {train.run_mode}")
        try:
            tag = qtag_for(train.questionnaire_ids)
        except ValueError:
            tag = "?"
        out.append(f"  rubric       {tag} (ids {list(train.questionnaire_ids)})")
        out.append(f"  policy       {train.base_model_id}"
                   f"  ({'4-bit NF4' if train.use_4bit else 'bf16'})"
                   f"  LoRA r={train.lora_r} a={train.lora_alpha} drop={train.lora_dropout}")
        out.append(f"  loop         {train.num_iterations} iters x "
                   f"{train.epochs_per_iteration} epochs = "
                   f"{train.total_effective_epochs:g} effective")
        if train.method == "GRPO":
            out.append(f"  batch        per_device {train.train_batch_size} x gas "
                       f"{train.gradient_accumulation_steps} = "
                       f"{train.generation_batch_size} completions -> "
                       f"{train.prompts_per_step} prompts/step (G={train.num_generations})")
        else:
            out.append(f"  batch        per_device {train.train_batch_size} x gas "
                       f"{train.gradient_accumulation_steps} = {train.pairs_per_step} pairs/step "
                       f"(M={train.num_branches_per_turn}, tau={train.pref_filter_tau}, "
                       f"mode={train.pref_tree_mode})")
        out.append(f"  checkpoint   {train.save_strategy}" +
                   (f" every {train.save_steps}" if train.save_strategy == "steps" else "") +
                   f" (keep {train.save_total_limit})   report_to {list(train.report_to)}")

    if gen is not None:
        out.append(f"  generation   {gen.num_conversations_per_iter} convs/iter x "
                   f"{gen.num_utterances_for_data} utts   batch {gen.conversation_batch_size}   "
                   f"T_ther {gen.temperature_therapist} / T_pat {gen.temperature_patient}")
    if roles is not None:
        for role, binding in roles.as_dict().items():
            out.append(f"  {role:<12} {binding.model}  [{binding.provider}]  "
                       f"{binding.base_url or '(no base_url)'}")
    if oracle is not None and la is not None:
        sub = getattr(la, "sub_batch_size", None)
        out.append(f"  concurrency  oracle {oracle.max_concurrency}   "
                   f"patient {getattr(gen, 'patient_concurrency', '?')}   "
                   f"lookahead sub-batch {sub if sub is not None else 'all'}")
    if paths is not None:
        out.append(f"  run dir      {paths.run_dir}")
        out.append(f"  conv dir     {paths.conv_root}")
    out.append(rule)
    return "\n".join(out)
