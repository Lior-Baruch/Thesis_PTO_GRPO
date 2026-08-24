"""data.py -- where every number in the EDA comes from: discovery, the lake, the readers, a cache.

Exactly one module knows how to turn a directory tree into a DataFrame. Every family notebook
starts by calling something here and then only reshapes what it got, so a change to the on-disk
layout is a change to this file and to nothing else.

Five readers, one per artifact the trainers leave behind:

===========================  ==============================================================
:func:`load_scores_long`     ``data/eval_scores/...`` -- the eval scores (one parquet per
                             model state), tidied to one row per (arm, state, persona, metric)
:func:`load_conversations`   ``data/conversations/<ARM>/model_iter_<N>/pers<PID>.csv``
:func:`load_generations`     ``runs/<ARM>/iteration_<N>/eda/generations.jsonl``, candidates
                             exploded to one row each
:func:`load_timing`          ``runs/<ARM>/iteration_<N>/timing_sessions.jsonl`` -- per-phase
                             seconds, summed over sessions
:func:`load_pref_pairs`      ``runs/<ARM>/iteration_<N>/pref_pairs/pairs.csv`` (PTO only)
===========================  ==============================================================

Why this is one lean module and Exp3's was 896 lines plus a 500-line parquet fold
--------------------------------------------------------------------------------
Four Exp3 problems are fixed UPSTREAM in Exp4 and have no code here at all:

1. **No persona-shuffle replay.** Exp3 saved ``conversation_{shuffled_index}.csv``, so every
   module had to re-derive ``Random(seed + k + 1)`` before it could pair a conversation across
   iterations. Exp4 names files by the STABLE persona id and stores ``persona_id`` as a column.
   **Pair on ``persona_id``, always -- never on file order, directory order or row order.**
2. **No fold cache.** The lake is one 96-row parquet per model state, not ~50k single-row CSVs,
   so there is no manifest and no content-signature fold to keep in sync with disk.
3. **No mtime forensics.** The trainer logs per-phase wall-clock to ``timing_sessions.jsonl``,
   so :func:`load_timing` reads a log instead of reconstructing GPU-hours from artifact mtimes.
4. **No ``oracle=`` path level and no name regex.** The training oracle is inside the arm name,
   and :func:`naming.parse_experiment_name` is the ONE parser -- this module never writes a
   second regex for a folder name.

Two index conventions -- read this before joining two frames
------------------------------------------------------------
``model_iter_<N>`` names the policy that GENERATED conversations; ``iteration_<N>`` names the
training pass that CONSUMED them. Iteration ``n`` generates with the iter-(``n``-1) adapter, so
they are off by one and both are called "iteration" in conversation:

* **model-state frames** (:func:`load_scores_long`, :func:`load_conversations`) carry
  ``iteration`` = the model-state index (base = 0) plus ``model_state`` = ``"model_iter_<N>"``.
* **training-side frames** (:func:`load_timing`, :func:`load_generations`,
  :func:`load_pref_pairs`) carry ``iteration`` = the TRAINING iteration (1-based) plus
  ``state_index`` = ``iteration - 1`` and ``model_state`` = the state that iteration trained
  FROM. Join a training-side frame to a score frame on ``state_index``, never on ``iteration``.

Import weight: no torch, ever. This module imports ``naming``, ``core.config``, ``core.timing``,
``core.recorder`` and ``core.conversations`` -- all stdlib-only by contract -- plus pandas.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

# The package leaf. Importing it FIRST is load-bearing: ``constants`` is what prepends
# ``Exp4_OpenStack/code`` to ``sys.path``, so every import below it resolves to the CANONICAL
# trainer-side module rather than to a copy. Do not reorder these two blocks.
from .constants import (COMPOSITES, CONV_DIR, DATA_DIR, DEFAULT_JUDGE_TAG, METRIC_ORDER,
                        N_PERSONAS, PERSONA_COLS, QUESTIONNAIRES, available_judge_tags)

from core.config import RunPaths                        # noqa: E402  (follows the path insert)
from core.conversations import conversation_id_for, load_conversations_dir  # noqa: E402
from core.recorder import iter_jsonl                    # noqa: E402
from core.timing import PHASE_KEYS, cumulative_seconds  # noqa: E402
from naming import (ArmInfo, model_state_label, parse_experiment_name,  # noqa: E402
                    parse_model_state_label)

__all__ = [
    # Frame contracts
    "ARM_KEY_COLUMNS",
    "SCORE_KEY_COLUMNS",
    "SCORE_COLUMNS",
    "CONVERSATION_COLUMNS",
    "GENERATION_COLUMNS",
    "TIMING_COLUMNS",
    "PREF_PAIR_COLUMNS",
    # Discovery
    "Arm",
    "discover_arms",
    "filter_arms",
    "judge_tags",
    # Readers
    "load_scores_long",
    "scores_by_judge",
    "load_conversations",
    "load_generations",
    "load_timing",
    "load_pref_pairs",
    "load_run_metadata",
    "canonical_personas",
    # Cache
    "load_cached",
    "cache_enabled",
    "set_cache",
    "reset_cache",
]


# ==============================================================================
#  What this module does NOT define
# ==============================================================================
#
# ``COMPOSITES``, ``QUESTIONNAIRES``, ``DEFAULT_JUDGE_TAG`` and the three data roots
# (``CONV_DIR`` / ``EVAL_SCORES_DIR`` / ``RUNS_DIR``) are imported from ``constants``. Per-ARM
# paths come from ``core.config.RunPaths``, the writer's own path builder, so a layout change
# cannot half-land.
#
# This module used to define ``COMPOSITES`` and ``DEFAULT_JUDGE_TAG`` itself, and its judge tag
# disagreed with the one in ``constants`` (``gemma4E2B`` vs ``local_gemma-4-E2B-it``). A judge tag
# is a DIRECTORY NAME in the score lake, so the writer and the reader were one rename apart from
# looking in different folders -- and that surfaces as an EDA rendering empty tables, not as an
# error. One definition each, in ``constants``; do not reintroduce a local copy.

_MODEL_STATE_GLOB = "model_iter_*"
_CONV_GLOB = "pers*.csv"
_ITERATION_GLOB = "iteration_*"


def _verbose(flag: Optional[bool] = None) -> bool:
    """Whether to narrate skipped/undiscovered things. ``EDA_VERBOSE`` in the environment."""
    if flag is not None:
        return bool(flag)
    return bool(os.environ.get("EDA_VERBOSE"))


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(f"  [data] {message}")


# ==============================================================================
#  Frame contracts -- the columns an EMPTY frame still has
# ==============================================================================
#
# Every reader returns a correctly-typed empty frame when nothing is on disk. That is not
# politeness: a family notebook run before any arm has trained must render empty artifacts, and
# a bare ``pd.DataFrame()`` makes it die on the first ``df["arm_label"]`` instead -- which reads
# like a broken notebook rather than like "no data yet".

#: Identity carried by every row of every frame in this module.
#:
#: ``arm`` is a DUPLICATE of ``arm_label`` -- same string, two names. ``config.py`` keys on
#: ``arm_label`` (``_ARM_COLUMN``) while ``plotting.py``'s figure builders default to
#: ``arm_col="arm"``, and a frame that satisfies only one of them fails the other with a KeyError
#: in the middle of a render. Emitting both costs one column and removes the guess; drop it once
#: the two agree on a spelling. Written in ONE place, :meth:`Arm.key` -- never set one alone.
ARM_KEY_COLUMNS: Tuple[str, ...] = (
    "arm_label", "arm", "experiment_name", "method", "k", "mcl", "mode", "qtag",
)

#: Model-state identity: ``iteration`` here is the MODEL STATE (base = 0). See the module docstring.
SCORE_KEY_COLUMNS: Tuple[str, ...] = ARM_KEY_COLUMNS + (
    "judge", "rep", "iteration", "model_state", "is_base", "persona_id", "conversation_id",
)

#: :func:`load_scores_long`. Per-item columns from the parquet are appended after these.
SCORE_COLUMNS: Tuple[str, ...] = SCORE_KEY_COLUMNS + ("metric", "score")

#: :func:`load_conversations` -- one row per utterance.
CONVERSATION_COLUMNS: Tuple[str, ...] = ARM_KEY_COLUMNS + (
    "iteration", "model_state", "is_base", "persona_id", "conversation_id",
    "turn_index", "role", "content", "n_utterances",
    "session_ended_by", "session_ended_explanation", "ended_early",
)

#: :func:`load_generations` -- one row per CANDIDATE. ``sub_score_<qid>`` columns are appended.
GENERATION_COLUMNS: Tuple[str, ...] = ARM_KEY_COLUMNS + (
    "iteration", "state_index", "model_state", "phase", "epoch",
    "conversation_id", "persona_id", "branch_id",
    "group_mean", "group_std", "chosen_idx",
    "candidate_idx", "is_chosen", "candidate_role", "score",
    "oracle_success", "oracle_attempts",
    "lookahead_tail", "lookahead_realized_turns", "lookahead_ended_early",
    "completion", "prefix",
)

#: :func:`load_timing`. Phase columns are derived from ``core.timing.PHASE_KEYS``, so a new
#: trainer phase appears here the moment it is added there.
TIMING_COLUMNS: Tuple[str, ...] = ARM_KEY_COLUMNS + (
    "iteration", "state_index", "model_state",
) + tuple(PHASE_KEYS) + ("total_s", "n_sessions", "resumed")

#: :func:`load_pref_pairs`. Beyond the identity columns this is the shape CLAUDE.md documents
#: for ``pairs.csv``; the reader passes through whatever columns the file actually has, so a
#: trainer that adds a column needs no change here.
PREF_PAIR_COLUMNS: Tuple[str, ...] = ARM_KEY_COLUMNS + (
    "iteration", "state_index", "model_state", "pair_index",
    "conversation_id", "persona_id", "branch_id",
    "prompt", "chosen", "rejected", "chosen_score", "rejected_score", "margin",
)

_COLUMN_DTYPES: Dict[str, str] = {
    "arm_label": "object", "arm": "object", "experiment_name": "object", "method": "object",
    "mode": "object", "qtag": "object", "judge": "object",
    "k": "int64", "mcl": "int64", "rep": "int64",
    "iteration": "int64", "state_index": "int64", "model_state": "object", "is_base": "bool",
    "persona_id": "int64", "conversation_id": "object",
    "metric": "object", "score": "float64",
    "turn_index": "int64", "role": "object", "content": "object", "n_utterances": "int64",
    "session_ended_by": "object", "session_ended_explanation": "object", "ended_early": "bool",
    "phase": "object", "epoch": "float64", "branch_id": "int64",
    "group_mean": "float64", "group_std": "float64", "chosen_idx": "float64",
    "candidate_idx": "int64", "is_chosen": "bool", "candidate_role": "object",
    "oracle_success": "object", "oracle_attempts": "float64",
    "lookahead_tail": "object", "lookahead_realized_turns": "float64",
    "lookahead_ended_early": "object", "completion": "object", "prefix": "object",
    "total_s": "float64", "n_sessions": "int64", "resumed": "bool",
    "pair_index": "int64", "prompt": "object", "chosen": "object", "rejected": "object",
    "chosen_score": "float64", "rejected_score": "float64", "margin": "float64",
}


def _dtype_for(column: str) -> str:
    """Dtype for an empty frame's column; anything ending in ``_s`` is seconds (float)."""
    if column in _COLUMN_DTYPES:
        return _COLUMN_DTYPES[column]
    return "float64" if column.endswith("_s") else "object"


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """A zero-row frame with the right columns AND the right dtypes.

    Dtypes matter as much as names: an ``object``-typed empty ``score`` column turns the first
    ``.mean()`` into a TypeError instead of NaN, and an ``object`` ``iteration`` breaks a merge
    against a populated frame.
    """
    return pd.DataFrame({c: pd.Series(dtype=_dtype_for(c)) for c in columns})


def _order_columns(df: pd.DataFrame, leading: Sequence[str]) -> pd.DataFrame:
    """Put the contract columns first, in order, and keep every extra column after them."""
    lead = [c for c in leading if c in df.columns]
    rest = [c for c in df.columns if c not in set(lead)]
    return df[lead + rest]


# ==============================================================================
#  Arm
# ==============================================================================


@dataclass(frozen=True)
class Arm:
    """One training arm found on disk: its identity, its model states, and its paths.

    Frozen and hashable -- an arm is a key (it names a folder), not a workspace. Every path is
    delegated to :class:`core.config.RunPaths`, so this class knows the arm's identity and
    nothing about the layout.

    Attributes:
        experiment_name: The folder name, e.g. ``GRPO4_Q1Q2_LA5_MCL12_G8_Ogemma4E2B_Patgemma4E2B``.
        info: The decoded identity from :func:`naming.parse_experiment_name`. Everything the name
            encodes -- rubric, K, MCL, branch width, preference-tree mode, role tags -- is here.
        iters: MODEL-STATE indices present on disk (``model_iter_<N>`` folders that actually hold
            conversations), ascending. A **tuple**, not a list, so the dataclass stays hashable;
            iterate it exactly as you would a list.
        data_root: The ``data/`` directory. Overridable so a test can point at a scratch tree.

    Notes:
        ``iters`` are MODEL STATES, not training iterations. ``0`` is the untrained base policy,
        and ``N`` iterations produce ``N+1`` states. Training-side artifacts are indexed by
        ITERATION -- see :meth:`iterations_on_disk` and the module docstring.
    """

    experiment_name: str
    info: ArmInfo
    iters: Tuple[int, ...] = ()
    data_root: str = DATA_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "iters", tuple(sorted(int(i) for i in self.iters)))
        if self.info.experiment_name != self.experiment_name:
            raise ValueError(
                f"Arm identity mismatch: folder {self.experiment_name!r} but info renders as "
                f"{self.info.experiment_name!r}. The folder name is the identity -- an Arm whose "
                f"info disagrees would read one arm's scores under another arm's label."
            )

    # -- identity ---------------------------------------------------------------

    @property
    def label(self) -> str:
        """Short display label (``"GRPO_LA5"``). See :attr:`naming.ArmInfo.label`.

        Warning:
            A DISPLAY key. It drops the rubric, MCL and branch width, so two arms differing only
            in MCL share a label and would merge in a groupby. Key on :attr:`experiment_name`
            for anything that reads or writes data.
        """
        return self.info.label

    @property
    def method(self) -> str:
        """``"GRPO"`` or ``"PTO"``."""
        return self.info.method

    @property
    def k(self) -> int:
        """Look-ahead depth; 0 means look-ahead is off."""
        return self.info.k

    @property
    def mode(self) -> str:
        """Preference-tree mode, or ``""`` for GRPO, which has no preference tree.

        ``""`` rather than ``None`` because this value becomes a frame column and
        ``groupby`` silently DROPS null keys -- a ``None`` here would make every GRPO row
        vanish from a ``groupby("mode")`` without an error. :attr:`info.mode` keeps the raw
        ``None``.
        """
        return self.info.mode or ""

    @property
    def mcl(self) -> int:
        """``MIN_CONV_LENGTH`` in utterances (therapist + patient combined)."""
        return self.info.mcl

    @property
    def qtag(self) -> str:
        """Training-rubric token from the ARM NAME (``Q1Q2``, ``WAI``, ``MISAT``, ...).

        Warning:
            Not a metric-registry key. ``naming`` spells MI-SAT ``MISAT`` because an arm-name
            field may not contain an underscore, while the score lake's partition and the
            registry key are ``MI_SAT``. Do not use this to index ``constants.METRICS``.
        """
        return self.info.qtag

    @property
    def paths(self) -> RunPaths:
        """The path builder for this arm. THE only place ``data/`` layout is spelled."""
        return RunPaths(data_root=self.data_root, experiment_name=self.experiment_name)

    def key(self) -> Dict[str, Any]:
        """The identity columns every frame in this module carries.

        ``arm`` duplicates ``arm_label`` on purpose -- see :data:`ARM_KEY_COLUMNS`.
        """
        return {
            "arm_label": self.label,
            "arm": self.label,
            "experiment_name": self.experiment_name,
            "method": self.method,
            "k": self.k,
            "mcl": self.mcl,
            "mode": self.mode,
            "qtag": self.qtag,
        }

    # -- paths ------------------------------------------------------------------

    def model_state(self, n: int) -> str:
        """``"model_iter_<n>"`` -- the folder naming the policy that GENERATED state *n*."""
        return model_state_label(n)

    def conv_dir(self, n: int) -> str:
        """``data/conversations/<EXP_NAME>/model_iter_<n>``.

        Returned whether or not it exists, so a caller can report the path it looked in.
        """
        return self.paths.conv_dir_for(n)

    def run_dir(self) -> str:
        """``data/runs/<EXP_NAME>`` -- adapters, checkpoints, per-iteration artifacts."""
        return self.paths.run_dir

    def iteration_dir(self, n: int) -> str:
        """``data/runs/<EXP_NAME>/iteration_<n>`` -- *n* is a TRAINING iteration (1-based)."""
        return self.paths.iteration_dir(n)

    def score_dir(self, metric: str, *, judge: str = "", rep: int = 0) -> str:
        """``eval_scores/judge=<tag>/rep=<r>/metric=<M>/<EXP_NAME>`` for this arm.

        Args:
            metric: A registered metric key (resolved to its lake partition through
                ``constants.QUESTIONNAIRES``) or a raw partition token.
            judge: Grader tag; ``""`` resolves to ``constants.DEFAULT_JUDGE_TAG``.
            rep: Repeat draw. ``0`` is the full-grid draw every family reads.
        """
        return self.paths.score_partition_dir(_resolve_judge(judge), rep, _metric_token(metric))

    def score_path(self, n: int, metric: str, *, judge: str = "", rep: int = 0) -> str:
        """The one parquet holding this arm's state-*n* scores on *metric* (96 rows).

        Raises:
            ValueError: if *metric* is a COMPOSITE (:data:`COMPOSITES`). A composite is computed
                from its components after loading and has no file of its own; asking for its
                path is a caller bug that would otherwise surface as "no scores found".
        """
        return self.paths.score_parquet_path(
            _resolve_judge(judge), rep, _metric_token(metric), n
        )

    # -- disk queries -----------------------------------------------------------

    def iterations_on_disk(self) -> List[int]:
        """TRAINING iteration indices (1-based) with an ``iteration_<N>/`` folder, ascending.

        Distinct from :attr:`iters`, which are MODEL STATES. An iteration folder exists as soon
        as the iteration starts, so this includes an in-flight iteration; presence of
        ``iteration_<N>/adapter/`` is what means "finished".
        """
        run_dir = self.run_dir()
        if not os.path.isdir(run_dir):
            return []
        found: List[int] = []
        for path in glob.glob(os.path.join(run_dir, _ITERATION_GLOB)):
            if not os.path.isdir(path):
                continue
            tail = os.path.basename(path).split("_")[-1]
            if tail.isdigit():
                found.append(int(tail))
        return sorted(found)


def _resolve_judge(judge: Optional[str]) -> str:
    """``""``/``None`` -> :data:`DEFAULT_JUDGE_TAG`; anything else passes through stripped."""
    tag = (judge or "").strip().strip("/\\")
    return tag or DEFAULT_JUDGE_TAG


def _metric_spec(metric: str) -> Tuple[Optional[str], str]:
    """``(lake partition token, per-conversation score column)`` for a metric key.

    Reads :data:`~eda_analysis.constants.QUESTIONNAIRES`, the flat projection of the metric
    registry built for exactly this: the two facts a loader needs per metric -- which
    ``metric=<M>`` directory to open and which column carries the number. A ``None`` token marks
    a COMPOSITE, which has no parquet of its own and is assembled from
    :data:`~eda_analysis.constants.COMPOSITES` after loading.

    Raises:
        KeyError: naming the registered keys. A typo'd metric would otherwise filter the frame to
            zero rows and render an empty figure that looks like missing data.
    """
    try:
        token, score_col = QUESTIONNAIRES[metric]
    except KeyError:
        raise KeyError(
            f"unknown metric {metric!r}; registered metrics are {list(METRIC_ORDER)}. Register it "
            f"in constants.METRICS rather than hardcoding a path -- the lake partition name and "
            f"the score column both come from that table."
        ) from None
    return (None if token is None else str(token)), str(score_col)


def _metric_token(metric: str) -> str:
    """The ``metric=<M>`` path token for a registered metric key.

    An unregistered string passes straight through, so a caller can address a partition the
    registry does not know about (a one-off re-score, say) without editing ``constants``.

    Raises:
        ValueError: for a composite, which is computed after loading and is never stored.
    """
    if metric not in QUESTIONNAIRES:
        return str(metric)
    token, _score_col = _metric_spec(metric)
    if token is None:
        components = " + ".join(COMPOSITES.get(metric, ())) or "its components"
        raise ValueError(
            f"metric {metric!r} is a composite ({components}); it is computed after loading and "
            f"has no parquet. Request its components instead."
        )
    return token


# ==============================================================================
#  Discovery
# ==============================================================================


def discover_arms(*,
                  include_incomplete: bool = False,
                  data_root: Optional[str] = None,
                  verbose: Optional[bool] = None) -> List[Arm]:
    """Every arm present under ``data/conversations/``, ordered (method, K, name).

    An arm is discovered from its CONVERSATIONS, not from its runs directory: conversations are
    what the eval scores are computed over, so this makes an arm analysable the moment its data
    lands and not before. A model-state folder counts only when it actually holds a
    ``pers<PID>.csv`` -- an empty in-flight ``model_iter_<N>/`` must not create a phantom state
    that every downstream reader then reports as "missing scores".

    Args:
        include_incomplete: Also record model-state folders that exist but hold no conversation
            yet, and keep an arm that has no usable state at all. Use it to see what is
            in flight; never for analysis, because those states have nothing to score.
        data_root: ``data/`` directory. Defaults to ``constants.DATA_DIR``.
        verbose: Narrate skipped entries. Defaults to the ``EDA_VERBOSE`` environment variable.

    Returns:
        A list of :class:`Arm`, possibly empty. An absent or unmounted ``data/`` is an empty
        list, not an error -- the notebooks must render before any run exists.

    Notes:
        **A folder that does not parse as an arm name is SKIPPED SILENTLY** (logged only under
        *verbose*). A scratch folder next to the arms is not an error, and raising here would
        make one stray directory take down every notebook.

        WARNING: on the Google Drive symlink, "the directory reads as empty" is not proof the
        conversations are missing -- the mount can wedge on a single folder and report zero
        files while every conversation is present in Drive. Check the cloud before concluding
        an arm is unfinished.
    """
    loud = _verbose(verbose)
    base = data_root or DATA_DIR
    root = os.path.join(data_root, "conversations") if data_root else CONV_DIR
    if not os.path.isdir(root):
        _log(f"no conversations root at {root}", loud)
        return []

    arms: List[Arm] = []
    for name in sorted(os.listdir(root)):
        exp_path = os.path.join(root, name)
        if not os.path.isdir(exp_path):
            continue
        try:
            info = parse_experiment_name(name)
        except ValueError as exc:
            _log(f"skipping {name!r}: {exc.__class__.__name__}: {exc}", loud)
            continue

        states: List[int] = []
        for path in sorted(glob.glob(os.path.join(exp_path, _MODEL_STATE_GLOB))):
            if not os.path.isdir(path):
                continue
            try:
                index = parse_model_state_label(os.path.basename(path))
            except ValueError as exc:
                _log(f"{name}: skipping {os.path.basename(path)!r}: {exc}", loud)
                continue
            if glob.glob(os.path.join(path, _CONV_GLOB)):
                states.append(index)
            elif include_incomplete:
                states.append(index)
            else:
                _log(f"{name}: model_iter_{index} holds no {_CONV_GLOB} -- in flight, skipped", loud)

        if not states and not include_incomplete:
            _log(f"{name}: no model state holds conversations -- arm skipped", loud)
            continue
        arms.append(Arm(experiment_name=name, info=info, iters=tuple(states), data_root=base))

    arms.sort(key=lambda a: (a.method, a.k, a.experiment_name))
    return arms


def filter_arms(arms: Sequence[Arm],
                *,
                methods: Optional[Iterable[str]] = None,
                ks: Optional[Iterable[int]] = None,
                modes: Optional[Iterable[str]] = None,
                arm_labels: Optional[Iterable[str]] = None) -> List[Arm]:
    """Subset a discovered arm list. Each criterion left ``None`` is not applied.

    Args:
        methods: ``["PTO"]``, ``["GRPO"]``, or both.
        ks: Look-ahead depths to keep, e.g. ``[0]``.
        modes: Preference-tree modes. ``""`` selects GRPO arms (see :attr:`Arm.mode`).
        arm_labels: Whitelist on :attr:`Arm.label` -- a DISPLAY key, so it can match more than
            one arm when two differ only in something the label elides (MCL, branch width).

    Returns:
        A new list, order preserved.
    """
    method_set = set(methods) if methods else None
    k_set = set(int(k) for k in ks) if ks is not None else None
    mode_set = set(modes) if modes is not None else None
    label_set = set(arm_labels) if arm_labels else None

    def keep(arm: Arm) -> bool:
        if method_set is not None and arm.method not in method_set:
            return False
        if k_set is not None and arm.k not in k_set:
            return False
        if mode_set is not None and arm.mode not in mode_set:
            return False
        if label_set is not None and arm.label not in label_set:
            return False
        return True

    return [a for a in arms if keep(a)]


def judge_tags(*, data_root: Optional[str] = None) -> List[str]:
    """Grader tags actually present in the score lake, :data:`DEFAULT_JUDGE_TAG` first.

    Delegates to ``constants.available_judge_tags`` for the normal case, so "which graders exist"
    has one implementation; the *data_root* branch is for a scratch tree in a test. Reports what
    was SCORED, not what is configured, and returns ``[]`` when the lake is absent -- which is
    what makes :func:`scores_by_judge` a no-op rather than a crash before the first scoring pass.
    """
    if data_root is None:
        return available_judge_tags()
    root = os.path.join(data_root, "eval_scores")
    if not os.path.isdir(root):
        return []
    tags = sorted(
        name.split("=", 1)[1]
        for name in os.listdir(root)
        if name.startswith("judge=") and os.path.isdir(os.path.join(root, name))
        and len(name.split("=", 1)) == 2 and name.split("=", 1)[1]
    )
    if DEFAULT_JUDGE_TAG in tags:
        tags.remove(DEFAULT_JUDGE_TAG)
        tags.insert(0, DEFAULT_JUDGE_TAG)
    return tags


# ==============================================================================
#  Personas
# ==============================================================================


@lru_cache(maxsize=4)
def canonical_personas(n: int = N_PERSONAS) -> pd.DataFrame:
    """The traits in ``constants.PERSONA_COLS``, indexed by the STABLE ``persona_id`` 0..n-1.

    Read from ``system_prompts_builder`` -- the canonical copy in ``code/`` that generated the
    patients -- so the trait table cannot drift from the personas the conversations ran against.

    Notes:
        In Exp3 recovering a conversation's persona meant replaying ``Random(seed + k + 1)``. In
        Exp4 ``persona_id`` is the file name AND a column, so this is a plain lookup table and
        pairing across iterations is a join.
    """
    from system_prompts_builder import get_patient_permutation_characteristics

    rows: List[Dict[str, Any]] = []
    for pid in range(int(n)):
        try:
            traits = get_patient_permutation_characteristics(pid) or {}
        except IndexError:
            break
        rows.append({"persona_id": pid, **{c: traits.get(c) for c in PERSONA_COLS}})
    if not rows:
        return pd.DataFrame(
            {"persona_id": pd.Series(dtype="int64"),
             **{c: pd.Series(dtype="object") for c in PERSONA_COLS}}
        ).set_index("persona_id")
    return pd.DataFrame(rows).set_index("persona_id")


def _attach_personas(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the persona traits onto a frame carrying ``persona_id``."""
    if df.empty or "persona_id" not in df.columns:
        return df
    traits = canonical_personas()
    if traits.empty:
        return df
    overlap = [c for c in traits.columns if c in df.columns]
    if overlap:
        traits = traits.drop(columns=overlap)
    return df.merge(traits, left_on="persona_id", right_index=True, how="left")


# ==============================================================================
#  The score lake
# ==============================================================================


def _read_parquet(path: str) -> Optional[pd.DataFrame]:
    """One score partition, or ``None`` (with a warning) if it cannot be read."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:                     # noqa: BLE001 -- any engine/IO failure
        print(f"  [data] WARNING: unreadable score partition {path}: "
              f"{type(exc).__name__}: {exc}")
        return None


def _disambiguate(df: pd.DataFrame, reserved: Iterable[str]) -> pd.DataFrame:
    """Rename incoming columns that collide with the frame contract, never drop them.

    A parquet that happened to carry its own ``score`` or ``metric`` column would otherwise
    overwrite the tidy one silently. The renamed copy keeps the value visible instead.
    """
    clash = {c: f"{c}_src" for c in df.columns if c in set(reserved)}
    return df.rename(columns=clash) if clash else df


def load_scores_long(arms: Optional[Sequence[Arm]] = None,
                     metrics: Optional[Union[str, Sequence[str]]] = None,
                     *,
                     judge: str = "",
                     rep: int = 0,
                     attach_persona: bool = True,
                     include_items: bool = True,
                     cache: Optional[bool] = None) -> pd.DataFrame:
    """The tidy score backbone: one row per (arm, model state, persona, metric).

    Args:
        arms: Arms to read. ``None`` discovers every arm on disk.
        metrics: Metric keys (``"Q1Q2"``, ``"WAI_SR"``, ...). ``None`` reads every registered
            metric, in ``constants.METRIC_ORDER``. A composite is assembled from its components
            after loading, so requesting only ``"Q1Q2"`` reads Q1 and Q2 and returns just Q1Q2.
        judge: Grader tag; ``""`` resolves to :data:`DEFAULT_JUDGE_TAG`.
        rep: Repeat draw; ``0`` is the full-grid draw every family reports.
        attach_persona: Join the persona characteristics on ``persona_id``.
        include_items: Carry the per-item columns from each parquet through (``Q1_1``,
            ``WAI3_TherapistLikesMe``, ...). They are naturally sparse across metrics -- a Q1 row
            has no WAI columns.
        cache: Override the parquet memo for this call (see :func:`load_cached`).

    Returns:
        Columns :data:`SCORE_COLUMNS` plus the per-item columns, sorted by
        (experiment_name, iteration, metric, persona_id). Empty and correctly typed when nothing
        is on disk.

    Notes:
        **Pair on ``persona_id``.** Every downstream contrast that compares two model states must
        join on ``persona_id`` -- the same patient, the same opening. Never on row order or on a
        file index: means survive a wrong pairing, but paired effect sizes and CIs do not, and
        nothing raises.

        A missing partition is skipped, so a partially-scored arm contributes whatever exists.
        An UNREADABLE partition warns loudly, because a silent skip is how biased missingness
        gets into a headline number: the states a grader struggled with are exactly the ones
        that would disappear.

        The per-conversation mean column (``Q1_Mean``) is dropped from the carried columns -- it
        IS ``score``, and keeping both invites two names for one number.
    """
    arms = list(discover_arms() if arms is None else arms)
    tag, r = _resolve_judge(judge), int(rep)
    requested = _requested_metrics(metrics)
    stored, composites = _split_metrics(requested)

    roots = [a.score_dir(m, judge=tag, rep=r) for a in arms for m in stored]
    return load_cached(
        "scores_long",
        lambda: _load_scores_long_impl(arms, stored, composites, requested, tag, r,
                                       attach_persona=attach_persona,
                                       include_items=include_items),
        roots=roots,
        arms=arms,
        judge=tag,
        rep=r,
        params={"metrics": tuple(requested), "attach_persona": attach_persona,
                "include_items": include_items},
        cache=cache,
    )


def _requested_metrics(metrics: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Normalise the ``metrics`` argument to a de-duplicated list of registered metric keys."""
    if metrics is None:
        return [k for k in METRIC_ORDER if k in QUESTIONNAIRES]
    if isinstance(metrics, str):
        metrics = [metrics]
    seen: List[str] = []
    for name in metrics:
        key = str(name)
        if key not in seen:
            _metric_spec(key)                    # raises on a typo, before any disk read
            seen.append(key)
    return seen


def _split_metrics(requested: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``(stored keys to read, composite keys to build)``.

    A composite's components are added to the read list even when the caller did not ask for
    them -- otherwise requesting ``["Q1Q2"]`` alone would read nothing and return an empty frame
    that looks exactly like an unscored arm. The extra component rows are dropped again once the
    composite has been built.
    """
    stored: List[str] = []
    composites: List[str] = []
    for key in requested:
        token, _score_col = _metric_spec(key)
        if token is None:
            composites.append(key)
            for component in COMPOSITES.get(key, ()):
                if component not in stored:
                    stored.append(component)
        elif key not in stored:
            stored.append(key)
    return stored, composites


def _load_scores_long_impl(arms: Sequence[Arm],
                           base_metrics: Sequence[str],
                           composites: Sequence[str],
                           requested: Sequence[str],
                           judge: str,
                           rep: int,
                           *,
                           attach_persona: bool,
                           include_items: bool) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    n_expected = n_missing = n_unreadable = 0

    for arm in arms:
        key = arm.key()
        for state in arm.iters:
            for metric in base_metrics:
                _token, mean_col = _metric_spec(metric)
                path = arm.score_path(state, metric, judge=judge, rep=rep)
                n_expected += 1
                if not os.path.isfile(path):
                    n_missing += 1
                    continue
                raw = _read_parquet(path)
                if raw is None:
                    n_unreadable += 1
                    continue
                if "persona_id" not in raw.columns or mean_col not in raw.columns:
                    n_unreadable += 1
                    print(f"  [data] WARNING: {path} lacks "
                          f"{'persona_id' if 'persona_id' not in raw.columns else mean_col!r} "
                          f"-- partition skipped")
                    continue

                personas = raw["persona_id"].astype("int64")
                if personas.duplicated().any():
                    print(f"  [data] WARNING: {path} repeats persona ids "
                          f"{sorted(personas[personas.duplicated()].unique())} -- every pairing "
                          f"downstream keys on persona_id, so this partition is ambiguous")

                out = pd.DataFrame(index=raw.index)
                for column, value in key.items():
                    out[column] = value
                out["judge"] = judge
                out["rep"] = int(rep)
                out["iteration"] = int(state)
                out["model_state"] = arm.model_state(state)
                out["is_base"] = (state == 0)
                out["persona_id"] = personas.to_numpy()
                out["conversation_id"] = [conversation_id_for(p) for p in personas]
                out["metric"] = metric
                out["score"] = pd.to_numeric(raw[mean_col], errors="coerce").astype("float64")

                if include_items:
                    drop = [c for c in ("persona_id", "conversation_id", mean_col)
                            if c in raw.columns]
                    extra = _disambiguate(raw.drop(columns=drop), SCORE_COLUMNS)
                    out = pd.concat([out, extra], axis=1)
                frames.append(out)

    if n_unreadable:
        print(f"  [data] {n_unreadable}/{n_expected} score partitions were unreadable "
              f"(judge={judge}, rep={rep}). Do NOT read the resulting table as a finding.")
    if not frames:
        if n_expected:
            print(f"  [data] no scores found for judge={judge!r} rep={rep} "
                  f"({n_missing}/{n_expected} partitions absent). "
                  f"Judges present in the lake: {judge_tags() or 'none'}")
        return _empty_frame(SCORE_COLUMNS)

    long = pd.concat(frames, ignore_index=True, sort=False)
    long = _add_composites(long, composites)
    # A component loaded only to build a composite is not something the caller asked for.
    long = long[long["metric"].isin(set(requested))]
    if long.empty:
        return _empty_frame(SCORE_COLUMNS)
    if attach_persona:
        long = _attach_personas(long)
    long = long.sort_values(["experiment_name", "iteration", "metric", "persona_id"],
                            kind="mergesort").reset_index(drop=True)
    return _order_columns(long, SCORE_COLUMNS)


def _add_composites(long: pd.DataFrame, composites: Sequence[str]) -> pd.DataFrame:
    """Append composite-metric rows (``Q1Q2`` = mean of the Q1 and Q2 per-conversation means).

    Composites are built per (arm, model state, persona) and only where EVERY component is
    present, so a conversation whose Q2 call failed contributes no Q1Q2 row rather than a
    half-composite that silently means something else.

    The average is UNWEIGHTED across rubrics, matching ``core/oracle.py::score_conversation``:
    Q1's 5 items and Q2's 17 items each carry half of the number the policy was trained on.
    Pooling all 22 items would be a different reward on a different axis.
    """
    if long.empty or not composites:
        return long
    key = ["experiment_name", "iteration", "persona_id"]
    identity = [c for c in SCORE_KEY_COLUMNS if c in long.columns]
    parts: List[pd.DataFrame] = [long]

    for name in composites:
        components = list(COMPOSITES.get(name, ()))
        if not components:
            continue
        source = long[long["metric"].isin(components)]
        if source.empty or source["metric"].nunique() < len(components):
            continue
        wide = source.pivot_table(index=key, columns="metric", values="score")
        if not set(components).issubset(wide.columns):
            continue
        wide = wide.dropna(subset=components)
        if wide.empty:
            continue
        composite = wide[components].mean(axis=1).rename("score").reset_index()
        composite["metric"] = name
        skeleton = source[identity].drop_duplicates(subset=key)
        merged = composite.merge(skeleton, on=key, how="left", suffixes=("", "_dup"))
        parts.append(merged[[c for c in identity if c in merged.columns] + ["metric", "score"]])

    return pd.concat(parts, ignore_index=True, sort=False)


def scores_by_judge(arms: Optional[Sequence[Arm]] = None,
                    metrics: Optional[Union[str, Sequence[str]]] = None,
                    *,
                    rep: int = 0,
                    judges: Optional[Sequence[str]] = None,
                    **kwargs) -> pd.DataFrame:
    """Every grader's scores in one frame, distinguished by the ``judge`` column.

    Args:
        arms: Arms to read; ``None`` discovers.
        metrics: As :func:`load_scores_long`.
        rep: Repeat draw.
        judges: Explicit grader tags; ``None`` reads every tag in the lake (:func:`judge_tags`).
        **kwargs: Passed through to :func:`load_scores_long`.

    Returns:
        The concatenation of one :func:`load_scores_long` frame per judge. Empty and correctly
        typed when the lake holds no judge partition.

    Warning:
        **NEVER AVERAGE RAW SCORES ACROSS JUDGES.** One grader was the TRAINING oracle -- the
        thing the policy was optimized against -- and any other is held out. That is
        train-vs-test, not two raters of one construct, and the two do not share a scale: in
        Exp3 the level offset between graders was 1.2-1.7 points AND model-dependent, so a mean
        over judges applies a silent, model-dependent shrinkage to every effect it touches.

        Combine only CONTRASTS (a difference between two model states under ONE judge) or
        standardized quantities. A finding worth reporting is one that survives in each judge's
        own column, shown side by side -- which is what this frame is for.
    """
    tags = list(judges) if judges is not None else judge_tags()
    if not tags:
        return _empty_frame(SCORE_COLUMNS)
    frames = [load_scores_long(arms, metrics, judge=tag, rep=rep, **kwargs) for tag in tags]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_frame(SCORE_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)


# ==============================================================================
#  Conversations
# ==============================================================================


def load_conversations(arm: Arm, n: int) -> pd.DataFrame:
    """One model state's conversations, one row per utterance.

    Args:
        arm: The arm to read.
        n: The MODEL STATE index (``model_iter_<n>``); ``0`` is the untrained base.

    Returns:
        Columns :data:`CONVERSATION_COLUMNS`, sorted by (persona_id, turn_index). Empty and
        correctly typed when the folder is absent or holds nothing.

    Notes:
        Parsing is delegated to ``core.conversations.load_conversations_dir``, the same reader
        the trainer resumes with, so the CSV contract has exactly one implementation. One
        corrupt file costs one conversation (it warns and continues) rather than the state.

        ``ended_early`` is True when a speaker emitted ``SESSION ENDED``; False means the
        conversation ran to the utterance cap. WARNING: if that fraction is ZERO for every
        persona, the patient backend is not honouring the session-end protocol -- the run is not
        broken in any way that raises, it just never terminates early.
    """
    states = load_conversations_dir(arm.conv_dir(n))
    if not states:
        return _empty_frame(CONVERSATION_COLUMNS)

    key = arm.key()
    rows: List[Dict[str, Any]] = []
    for persona_id in sorted(states):
        state = states[persona_id]
        for turn_index, turn in enumerate(state.turns):
            rows.append({
                **key,
                "iteration": int(n),
                "model_state": arm.model_state(n),
                "is_base": (int(n) == 0),
                "persona_id": int(persona_id),
                "conversation_id": state.conversation_id,
                "turn_index": turn_index,
                "role": str(turn.get("role", "")),
                "content": str(turn.get("content", "")),
                "n_utterances": state.n_utterances,
                "session_ended_by": state.session_ended_by,
                "session_ended_explanation": state.session_ended_explanation,
                "ended_early": bool(state.session_ended_by),
            })
    if not rows:
        return _empty_frame(CONVERSATION_COLUMNS)
    return _order_columns(pd.DataFrame(rows), CONVERSATION_COLUMNS)


# ==============================================================================
#  Per-generation capture
# ==============================================================================


def load_generations(arm: Arm, n: int, *, include_prefix: bool = True) -> pd.DataFrame:
    """One training iteration's branch capture, exploded to one row per CANDIDATE.

    Args:
        arm: The arm to read.
        n: The TRAINING iteration (1-based) -- ``runs/<ARM>/iteration_<n>/eda/generations.jsonl``.
            Its policy is the state ``n-1``, which is what ``state_index`` / ``model_state``
            record.
        include_prefix: Keep the branch prefix. It is repeated on each of the branch's candidates
            and can be several thousand characters, so pass False for a lean frame.

    Returns:
        Columns :data:`GENERATION_COLUMNS` plus one ``sub_score_<qid>`` column per rubric the
        oracle returned. Empty and correctly typed when the file is absent.

    Notes:
        **Key every per-branch aggregation on ``(conversation_id, branch_id)``.** For PTO
        ``branch_id`` is trunk DEPTH, not a unique id: it restarts at 0 for every trunk and
        therefore repeats across conversations within one iteration. Grouping on ``branch_id``
        alone silently pools unrelated conversations. Both columns are returned so that pairing
        is available; GRPO's counter happens to be unique, but keying both methods the same way
        is what keeps a shared analysis honest.

        The exact text the oracle scored is
        ``prefix + "\\n\\n[THERAPIST]: " + completion + (lookahead_tail or "")``.

        ``score`` is the reward the trainer USED -- a degenerate completion appears as the reward
        floor with its ``sub_score_*`` columns empty, which is how a floored row is identified.
    """
    path = arm.paths.generations_path(n)
    if not os.path.isfile(path):
        return _empty_frame(GENERATION_COLUMNS)

    key = arm.key()
    state_index = max(int(n) - 1, 0)
    rows: List[Dict[str, Any]] = []
    for record in iter_jsonl(path):
        chosen_idx = record.get("chosen_idx")
        branch = {
            **key,
            "iteration": int(record.get("iteration", n) or n),
            "state_index": state_index,
            "model_state": arm.model_state(state_index),
            "phase": record.get("phase"),
            "epoch": record.get("epoch"),
            "conversation_id": record.get("conversation_id"),
            "persona_id": record.get("persona_id"),
            "branch_id": record.get("branch_id"),
            "group_mean": record.get("group_mean"),
            "group_std": record.get("group_std"),
            "chosen_idx": chosen_idx,
        }
        if include_prefix:
            branch["prefix"] = record.get("prefix")

        candidates = record.get("candidates") or []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            idx = candidate.get("idx", position)
            oracle = candidate.get("oracle") or {}
            lookahead = candidate.get("lookahead") or {}
            row = {
                **branch,
                "candidate_idx": int(idx),
                "is_chosen": (chosen_idx is not None and int(idx) == int(chosen_idx)),
                "candidate_role": candidate.get("role"),
                "score": candidate.get("score"),
                "oracle_success": oracle.get("success"),
                "oracle_attempts": oracle.get("attempts"),
                "lookahead_tail": lookahead.get("tail"),
                "lookahead_realized_turns": lookahead.get("realized_turns"),
                "lookahead_ended_early": lookahead.get("ended_early"),
                "completion": candidate.get("completion"),
            }
            for qid, value in (candidate.get("sub_scores") or {}).items():
                row[f"sub_score_{qid}"] = value
            rows.append(row)

    if not rows:
        return _empty_frame(GENERATION_COLUMNS)
    return _order_columns(pd.DataFrame(rows), GENERATION_COLUMNS)


# ==============================================================================
#  Timing
# ==============================================================================


def load_timing(arms: Optional[Sequence[Arm]] = None) -> pd.DataFrame:
    """Per-(arm, training iteration) wall-clock, summed over every session that worked on it.

    Args:
        arms: Arms to read; ``None`` discovers every arm on disk.

    Returns:
        Columns :data:`TIMING_COLUMNS` -- one row per iteration folder, with one column per
        phase in ``core.timing.PHASE_KEYS`` plus ``total_s``, ``n_sessions`` and ``resumed``.
        Empty and correctly typed when no timing log exists.

    Notes:
        **``n_sessions > 1`` means the iteration was RESUMED** (``resumed`` is the same fact as
        a bool). That flag is the one to surface in any cost table: for such an iteration every
        per-PROCESS field in ``iteration_metadata.json`` is an UNDERCOUNT, because it records
        only the last session. This is the Exp3 defect that made a 7.7 h iteration report
        14,501 s and cost 1,336 lines of mtime archaeology to undo. The append-only log summed
        here is the correct number; do not mix the two sources in one table.

        An iteration that recorded nothing (a phase that never called ``log_session``) shows
        zeros, not NaN -- ``cumulative_seconds`` sums an empty log to 0.0. A row of all-zero
        phases with ``n_sessions == 0`` therefore means "nothing was logged", which is a
        trainer-side omission, not a free iteration.
    """
    arms = list(discover_arms() if arms is None else arms)
    rows: List[Dict[str, Any]] = []
    for arm in arms:
        key = arm.key()
        for iteration in arm.iterations_on_disk():
            totals = cumulative_seconds(arm.iteration_dir(iteration))
            n_sessions = int(totals.get("n_sessions", 0) or 0)
            state_index = max(iteration - 1, 0)
            rows.append({
                **key,
                "iteration": int(iteration),
                "state_index": state_index,
                "model_state": arm.model_state(state_index),
                **{phase: float(totals.get(phase, 0.0) or 0.0) for phase in PHASE_KEYS},
                "total_s": float(totals.get("total_s", 0.0) or 0.0),
                "n_sessions": n_sessions,
                "resumed": n_sessions > 1,
            })
    if not rows:
        return _empty_frame(TIMING_COLUMNS)
    frame = pd.DataFrame(rows).sort_values(["experiment_name", "iteration"], kind="mergesort")
    return _order_columns(frame.reset_index(drop=True), TIMING_COLUMNS)


# ==============================================================================
#  PTO preference pairs
# ==============================================================================


def load_pref_pairs(arm: Arm, n: int) -> pd.DataFrame:
    """One PTO iteration's ``(prompt, chosen, rejected)`` audit trail.

    Args:
        arm: A PTO arm. GRPO has no preference data -- only prompts -- so a GRPO arm returns an
            empty frame with a note rather than raising.
        n: The TRAINING iteration (1-based) -- ``iteration_<n>/pref_pairs/pairs.csv``.

    Returns:
        The identity columns followed by every column the CSV actually has. When the file is
        absent the frame is empty with :data:`PREF_PAIR_COLUMNS`, the shape CLAUDE.md documents.

    Notes:
        ``pairs.csv`` is both the DPO audit trail AND the Step-2 completion marker: its presence
        makes a resumed iteration reload it and skip the (dominant) preference build. Reading it
        here is harmless, but never delete or rewrite it from the analysis side.

        A pair exists only where the score gap cleared ``PREF_FILTER_TAU``, so the row count is
        NOT the number of branch points -- a tie emits no pair while still advancing the trunk.
        Use :func:`load_generations` for the full candidate population.
    """
    if arm.method != "PTO":
        print(f"  [data] {arm.label} is a {arm.method} arm: GRPO has no preference data, only "
              f"prompts. Returning an empty frame.")
        return _empty_frame(PREF_PAIR_COLUMNS)

    path = arm.paths.pairs_csv_path(n)
    if not os.path.isfile(path):
        return _empty_frame(PREF_PAIR_COLUMNS)
    try:
        raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except Exception as exc:                     # noqa: BLE001 -- a torn CSV is not fatal here
        print(f"  [data] WARNING: could not read {path}: {type(exc).__name__}: {exc}")
        return _empty_frame(PREF_PAIR_COLUMNS)
    if raw.empty:
        print(f"  [data] WARNING: {path} is empty. An empty marker makes a RESUMED iteration "
              f"reload 0 pairs and skip the build, which then fails the '0 pref pairs' guard. "
              f"Delete the file to force a clean rebuild -- do not lower tau, which is a science "
              f"change the folder name cannot record.")
        return _empty_frame(PREF_PAIR_COLUMNS)

    state_index = max(int(n) - 1, 0)
    out = pd.DataFrame(index=raw.index)
    for column, value in arm.key().items():
        out[column] = value
    out["iteration"] = int(n)
    out["state_index"] = state_index
    out["model_state"] = arm.model_state(state_index)
    out["pair_index"] = range(len(raw))
    out = pd.concat([out, _disambiguate(raw, out.columns)], axis=1)
    return _order_columns(out, PREF_PAIR_COLUMNS)


def load_run_metadata(arm: Arm) -> dict:
    """``runs/<EXP_NAME>/run_metadata.json`` -- the arm's CURRENT config, or ``{}``.

    Notes:
        This file is overwritten by every process that works on the arm, so it describes the
        LAST configuration, not necessarily the one an early iteration ran under. The superseded
        payloads survive in the sibling ``run_metadata_history.jsonl`` (one line per process) --
        read that when a knob that is not in the arm name (tau, a sampling temperature, the
        look-ahead sub-batch) matters to the claim being made.

        Returns ``{}`` for a missing or corrupt file: an arm can legitimately have conversations
        and no metadata (a generate-only pass writes none), and no analysis should fail on it.
    """
    path = arm.paths.run_metadata_path
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"  [data] WARNING: could not read {path}: {type(exc).__name__}: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


# ==============================================================================
#  Cache
# ==============================================================================
#
# The readers above are pure functions of files on disk, so a built frame can be memoized to
# ``eda/.eda_cache/*.parquet``. Invalidation is by CONTENT -- the key hashes every input file's
# (name, size, mtime_ns) -- so a re-score or a re-generation invalidates automatically and the
# cache can never serve a stale number. That property is worth more than the speed: a cache that
# needs manual clearing is a cache that eventually reports last week's result as this week's.

_EDA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_EDA_DIR, ".eda_cache")
_CACHE_ENABLED: Optional[bool] = None


def set_cache(enabled: Optional[bool]) -> None:
    """Enable/disable the memo process-wide. ``None`` restores the default (on)."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = None if enabled is None else bool(enabled)


def cache_enabled() -> bool:
    """True unless ``EDA_NO_CACHE`` is set in the environment or :func:`set_cache` turned it off."""
    if os.environ.get("EDA_NO_CACHE"):
        return False
    return True if _CACHE_ENABLED is None else _CACHE_ENABLED


def reset_cache() -> int:
    """Delete every memoized frame. Returns how many files were removed."""
    removed = 0
    for path in glob.glob(os.path.join(_CACHE_DIR, "*.parquet")):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def _content_signature(roots: Iterable[str]) -> str:
    """blake2b over ``(name, size, mtime_ns)`` of every file in *roots* (non-recursive).

    A path that is itself a file is hashed directly; a missing path contributes a ``missing``
    marker, so "the data appeared" is as much a cache miss as "the data changed".

    Unlike Exp3 this watches EVERY file in a root rather than a caller-supplied extension list.
    The extension filter was a trap -- a loader that read ``generations.jsonl`` while watching
    ``*.csv`` keyed its frame on files it never read and went stale without a miss -- and Exp4's
    directories are small enough (<= ~96 files) that filtering buys nothing.
    """
    digest = hashlib.blake2b(digest_size=16)
    for root in sorted({r for r in roots if r}):
        digest.update(root.encode("utf-8"))
        digest.update(b"|")
        if os.path.isfile(root):
            try:
                stat = os.stat(root)
                digest.update(f"{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
            except OSError:
                digest.update(b"err\n")
            continue
        if not os.path.isdir(root):
            digest.update(b"missing\n")
            continue
        try:
            entries = sorted((e for e in os.scandir(root) if e.is_file()), key=lambda e: e.name)
        except OSError:
            digest.update(b"err\n")
            continue
        for entry in entries:
            try:
                stat = entry.stat()
            except OSError:
                continue
            digest.update(f"{entry.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def load_cached(name: str,
                builder: Callable[[], pd.DataFrame],
                *,
                roots: Iterable[str],
                params: Optional[dict] = None,
                arms: Optional[Sequence[Arm]] = None,
                judge: str = "",
                rep: int = 0,
                cache: Optional[bool] = None) -> pd.DataFrame:
    """Return ``builder()``, memoized to ``eda/.eda_cache/<name>__<params>__<content>.parquet``.

    Args:
        name: Frame name; the cache-file prefix.
        builder: Zero-argument callable returning the frame. Called on a miss, and on every call
            when caching is off.
        roots: Directories (or files) whose contents define the frame. Their
            ``(name, size, mtime_ns)`` become the content half of the key, so any rewrite of an
            input invalidates the entry.
        params: Anything else that changes the RESULT -- metric list, flags. Folded into the key
            by ``repr`` of its sorted items.
        arms: The arms the frame covers. Their names and model-state lists go into the key, so a
            two-arm frame and a four-arm frame coexist instead of overwriting each other.
        judge: Grader tag. **Part of the key on purpose:** two graders hold same-named files for
            the same arms, so the content signature alone would not separate them and a judge
            swap could be served the other grader's frame.
        rep: Repeat draw; keyed for the same reason.
        cache: Override :func:`cache_enabled` for this call.

    Returns:
        The built (or restored) frame.

    Notes:
        The write is atomic (temp file + ``os.replace``), so an interrupted render cannot leave a
        torn parquet behind. **Every parquet failure degrades to an uncached build** and never
        raises: a frame with a column pyarrow cannot serialise is a reason not to cache it, not
        a reason to fail the notebook. Set ``EDA_CACHE_VERBOSE`` to see those.
    """
    use_cache = cache_enabled() if cache is None else bool(cache)
    if not use_cache:
        return builder()

    arm_signature = "|".join(
        f"{a.experiment_name}:{','.join(str(i) for i in a.iters)}"
        for a in sorted(arms or (), key=lambda a: a.experiment_name)
    )
    param_signature = (f"{arm_signature}||judge={_resolve_judge(judge)}:{int(rep)}"
                       f"||{repr(sorted((params or {}).items()))}")
    param_key = hashlib.blake2b(param_signature.encode("utf-8"), digest_size=8).hexdigest()
    content_key = _content_signature(roots)
    path = os.path.join(_CACHE_DIR, f"{name}__{param_key}__{content_key}.parquet")

    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:                        # noqa: BLE001 -- corrupt/partial: rebuild
            pass

    frame = builder()
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        for stale in glob.glob(os.path.join(_CACHE_DIR, f"{name}__{param_key}__*.parquet")):
            if stale != path:
                try:
                    os.remove(stale)
                except OSError:
                    pass
        tmp = f"{path}.{os.getpid()}.tmp"
        frame.reset_index(drop=True).to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception as exc:                     # noqa: BLE001 -- never fail over a cache write
        if os.environ.get("EDA_CACHE_VERBOSE"):
            print(f"  [data] {name}: not cached ({type(exc).__name__}: {exc})")
    return frame
