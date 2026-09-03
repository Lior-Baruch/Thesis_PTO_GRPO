"""constants.py -- the package LEAF: workspace paths, the metric registry, judge tags, labels.

Every other module in ``eda_analysis`` does a plain top-level ``from .constants import ...``. That
is only safe because this module imports nothing from the package, so there is no import order and
no cycle to reason about. Keep it that way: a helper that needs pandas, matplotlib or another
``eda_analysis`` module does not belong here.

What this module owns
---------------------
1. **Where the data is.** :func:`resolve_workspace_root` walks up to the ``Exp4_OpenStack`` root and
   puts ``<root>/code`` on ``sys.path``, which is what makes ``import questionnaires`` /
   ``import naming`` / ``import roles`` resolve to the SINGLE canonical copies the trainers use.
   Every path constant below hangs off that root.
2. **What a metric IS.** The eight instruments, each with its questionnaire id, its score-lake
   partition name, the per-conversation column that carries its headline number, its per-item
   columns, its display name, and -- the one that bites -- whether higher is better.
3. **Which grader produced a number.** :func:`judge_tag` builds the ``judge=<tag>`` partition name;
   :func:`judge_dirname` shortens it for display.
4. **Label and colour keys**, so a figure and a table name the same arm the same way.

The metric registry is DERIVED, not retyped
-------------------------------------------
Item labels, item counts and rating scales come from ``questionnaires.py`` at import time, not from
a hand-maintained copy. ``questionnaires.py`` is a verbatim copy of Exp3's and is stdlib-only, so
importing it here is cheap and it is the instrument itself -- transcribing its 17 Q2 labels into
this file would create exactly the kind of second source that drifts. (Exp3's constants.py did
transcribe them, to keep its leaf import-free; Exp4 gets the same leafness for free because the
canonical module has no dependencies.)

Warning:
    A judge tag and a metric partition name are DIRECTORY NAMES. They appear in the path of every
    score already written (``data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/...``). Renaming one
    does not migrate anything -- it orphans every parquet under the old name, and the loader will
    quietly report "no scores on disk" for a grader whose scores are sitting right there. Treat
    both as append-only vocabularies.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

__all__ = [
    # Workspace
    "resolve_workspace_root",
    "WORKSPACE_ROOT",
    "CODE_DIR",
    "DATA_DIR",
    "RUNS_DIR",
    "CONV_DIR",
    "EVAL_SCORES_DIR",
    "RESULTS_DIR",
    "NOTEBOOKS_DIR",
    "run_paths",
    # Metrics
    "Metric",
    "METRICS",
    "COMPOSITE_METRICS",
    "ALL_METRICS",
    "COMPOSITES",
    "QUESTIONNAIRES",
    "METRIC_ORDER",
    "TRAINING_REWARD_METRIC",
    "LOWER_IS_BETTER",
    "metric",
    "metric_for_qid",
    "is_lower_better",
    "sign_of",
    "score_column",
    "metric_partition",
    # Reproducibility
    "BOOT_SEED",
    # Judges
    "JUDGE_PARTITION",
    "REP_PARTITION",
    "METRIC_PARTITION",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_TAG",
    "judge_tag",
    "judge_dirname",
    "judge_dir",
    "available_judge_tags",
    "available_judge_reps",
    # Labels + palettes
    "BASE_ARM",
    "ARM_DISPLAY",
    "ARM_COLORS",
    "METRIC_COLORS",
    "FALLBACK_COLORS",
    "K_LINESTYLE",
    "display_label",
    "short_label",
    "arm_label",
    "k_of",
    "method_of",
    # Personas
    "N_PERSONAS",
    "PERSONA_COLS",
    "COOP_LABEL",
    "COOP_ORDER",
]


# ==============================================================================
#  1. THE WORKSPACE
# ==============================================================================

#: Files that, together, identify the ``Exp4_OpenStack`` root and nothing else. Structural rather
#: than nominal: Exp3's root has ``code/questionnaires.py`` too, but no ``code/naming.py`` (its arm
#: grammar was scattered) and no ``code/roles.py``, so all three together cannot match it. Exp3
#: keyed on ``HF_key.txt`` + ``openai_key.txt`` instead; Exp4 costs $0 in API and can be run with no
#: key files present at all, so a key file is the wrong marker here.
_ROOT_MARKERS: Tuple[str, ...] = (
    os.path.join("code", "naming.py"),
    os.path.join("code", "roles.py"),
    os.path.join("code", "questionnaires.py"),
)


def resolve_workspace_root(*starts: str,
                           max_steps: int = 12,
                           install_path: bool = True) -> str:
    """Locate the ``Exp4_OpenStack`` root and put its ``code/`` directory on ``sys.path``.

    Args:
        starts: Directories to walk up from. Defaults to this module's own directory followed by
            ``os.getcwd()`` -- the module's location wins so that importing the package from an
            unrelated working directory still resolves correctly, while the cwd fallback covers a
            copied/zipped checkout.
        max_steps: How many parent directories to try per start.
        install_path: Insert ``<root>/code`` at the FRONT of ``sys.path`` (idempotent). This is the
            whole reason the function has a side effect: it is what makes ``import questionnaires``,
            ``import naming``, ``import roles`` and ``import system_prompts_builder`` resolve to the
            same files the trainers ran, rather than to a second copy.

    Returns:
        Absolute path to the experiment root.

    Raises:
        RuntimeError: naming every directory that was searched. A wrong root is not a recoverable
            condition -- it would silently point the whole EDA at an empty or foreign ``data/``
            tree and every family would render "no arms found".

    Warning:
        ``sys.path`` is process-global. If an Exp3 notebook in the same kernel already imported its
        own ``questionnaires``, that module object is in ``sys.modules`` and a later ``import
        questionnaires`` returns EXP3's copy no matter what this function prepends. The two copies
        are byte-identical by contract, so the import below prints a loud warning rather than
        raising -- but if Exp3's copy is ever edited, that warning is the only thing standing
        between you and scores computed against a different instrument.
    """
    if not starts:
        starts = (os.path.dirname(os.path.abspath(__file__)), os.getcwd())

    searched: List[str] = []
    for start in starts:
        current = os.path.abspath(start)
        for _ in range(max_steps):
            searched.append(current)
            if all(os.path.exists(os.path.join(current, marker)) for marker in _ROOT_MARKERS):
                if install_path:
                    code_dir = os.path.join(current, "code")
                    if code_dir not in sys.path:
                        sys.path.insert(0, code_dir)
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    raise RuntimeError(
        "eda_analysis.constants: could not locate the Exp4_OpenStack root. Looked for all of "
        f"{list(_ROOT_MARKERS)} while walking up from {list(starts)}; searched "
        f"{sorted(set(searched))}. The root is the directory that contains code/ and eda/."
    )


#: The experiment root (``.../Exp4_OpenStack``). Resolved once, at import, with the ``code/``
#: sys.path insert as a side effect -- so every import below this line sees the canonical modules.
WORKSPACE_ROOT: str = resolve_workspace_root()

CODE_DIR: str = os.path.join(WORKSPACE_ROOT, "code")

#: ``data/`` and its three Google Drive directory symlinks. Gitignored; CLAUDE.md's "Data layout"
#: section is the only record of their shape.
DATA_DIR: str = os.path.join(WORKSPACE_ROOT, "data")
RUNS_DIR: str = os.path.join(DATA_DIR, "runs")
CONV_DIR: str = os.path.join(DATA_DIR, "conversations")
EVAL_SCORES_DIR: str = os.path.join(DATA_DIR, "eval_scores")

#: The analysis side. ``results/`` is written by the family notebooks (via ``exports``);
#: ``notebooks/`` holds one notebook per family.
_EDA_DIR: str = os.path.join(WORKSPACE_ROOT, "eda")
RESULTS_DIR: str = os.path.join(_EDA_DIR, "results")
NOTEBOOKS_DIR: str = os.path.join(_EDA_DIR, "notebooks")


# -- the canonical modules, imported through the path insert above -------------------------------
from questionnaires import (  # noqa: E402  (import must follow the sys.path insert)
    QuestionnaireID,
    get_questionnaire,
)
from roles import DEFAULT_JUDGE_MODEL, model_tag  # noqa: E402

for _canonical in (QuestionnaireID.__module__, model_tag.__module__):
    _module_file = getattr(sys.modules.get(_canonical), "__file__", "") or ""
    if _module_file and os.path.dirname(os.path.abspath(_module_file)) != CODE_DIR:
        print(
            f"  [eda_analysis] WARNING: module {_canonical!r} was imported from {_module_file!r}, "
            f"not from this experiment's {CODE_DIR!r}. Another experiment's copy is already in "
            f"sys.modules for this kernel. The copies are supposed to be identical; if they are "
            f"not, every number below was computed against a different instrument. Restart the "
            f"kernel and import eda_analysis first."
        )


def run_paths(experiment_name: str):
    """A ``core.config.RunPaths`` for *experiment_name*, rooted at this workspace's ``data/``.

    The WRITER owns every per-arm path shape (``core/config.py``), so the EDA borrows it instead of
    re-joining strings: an arm's run directory, its conversation folders and its score-lake
    partitions are then defined in exactly one place, and a layout change cannot half-land.

    Imported lazily because it drags in ``core.config`` -- which is stdlib-only and torch-free, but
    is still trainer code that a plain ``import eda_analysis`` has no reason to load.

    Raises:
        ValueError: via ``RunPaths.__post_init__`` if *experiment_name* is not a legal single path
            segment. It does NOT check that the name is a legal ARM name; use
            ``naming.parse_experiment_name`` for that.
    """
    from core.config import RunPaths  # local import: keeps this module a leaf

    return RunPaths(data_root=DATA_DIR, experiment_name=experiment_name)


# ==============================================================================
#  2. THE METRIC REGISTRY
# ==============================================================================
#
# One entry per instrument. The score lake stores ONE PARQUET PER (judge, rep, metric, arm, model
# state) -- 96 rows, one per persona -- under ``metric=<partition>/``, and each row carries the
# per-item columns plus the single ``score_column`` that every headline figure plots.

#: Rubrics whose response is ``{globals: {...}, behaviors: {...}}`` rather than a flat ``scores``
#: array (see ``core/oracle.py``). Their behaviour counts are unbounded, so their headline number is
#: a rate or a proportion rather than a mean over the rating scale.
_NESTED_IDS: FrozenSet[int] = frozenset({
    QuestionnaireID.MITI.value, QuestionnaireID.PCT.value, QuestionnaireID.MICI.value,
})


@dataclass(frozen=True)
class Metric:
    """One evaluation instrument, as the EDA sees it.

    Attributes:
        key: The registry key and the value of the ``metric`` column in every long score frame.
        questionnaire_id: ``QuestionnaireID`` value, or ``None`` for a composite.
        partition: The ``metric=<M>`` directory in the score lake, or ``None`` for a composite
            (which is computed from other metrics and is never stored).
        score_column: The per-conversation column that carries this metric's headline number.
            THE column every trajectory, contrast and ranking uses.
        item_columns: The per-item / per-behaviour columns stored alongside it, in rubric order.
        display: Human-readable name for figure titles and table headers.
        higher_is_better: See the warning below.
        scale: ``(min, max)`` of the rating scale for the items that HAVE one. ``None`` where the
            headline number is a rate or a proportion rather than a rating.
        shape: ``"array"`` (flat ``scores`` list) | ``"nested"`` (globals + behaviours) |
            ``"composite"`` (derived from other metrics).
        composite_of: For a composite, the keys it averages.

    Warning:
        ``higher_is_better`` is False for exactly one instrument, **MICI** -- it counts
        MI-INCONSISTENT therapist behaviour, so a bigger number is a worse therapist. Every
        ordering, ranking, "best checkpoint" selection, effect-size sign and sequential colour scale
        must consult it. Multiply by :func:`sign_of` before an ``argmax``/``sort`` rather than
        remembering the exception at each call site; that is the whole reason the flag exists on the
        registry instead of living in a comment.
    """

    key: str
    questionnaire_id: Optional[int]
    partition: Optional[str]
    score_column: str
    item_columns: Tuple[str, ...]
    display: str
    higher_is_better: bool = True
    scale: Optional[Tuple[int, int]] = None
    shape: str = "array"
    composite_of: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_composite(self) -> bool:
        """True when this metric is derived rather than stored (``partition is None``)."""
        return self.partition is None

    @property
    def sign(self) -> int:
        """``+1`` when higher is better, ``-1`` when lower is. See :func:`sign_of`."""
        return 1 if self.higher_is_better else -1


def _labels_and_scale(qid: int) -> Tuple[Tuple[str, ...], Tuple[int, int], int]:
    """``(item labels, (scale_min, scale_max), item count)`` straight from ``questionnaires.py``.

    The nested rubrics' builders interpolate an utterance count into their prompt and therefore take
    ``conversation_text``; an empty string is fine here because only the labels and the scale are
    wanted, and neither depends on the transcript.
    """
    q = get_questionnaire(qid, conversation_text="")
    return tuple(q.labels), (int(q.scale_min), int(q.scale_max)), int(q.questions_count)


def _metric(key: str,
            qid: int,
            partition: str,
            score_column: str,
            display: str,
            *,
            higher_is_better: bool = True) -> Metric:
    """Build one stored instrument, deriving its items and scale from the canonical rubric."""
    labels, scale, _count = _labels_and_scale(qid)
    nested = qid in _NESTED_IDS
    return Metric(
        key=key,
        questionnaire_id=qid,
        partition=partition,
        score_column=score_column,
        item_columns=labels,
        display=display,
        higher_is_better=higher_is_better,
        # A nested rubric's SCALE describes its global ratings only -- its behaviour counts are
        # unbounded above, which is exactly why none of them is the score_column.
        scale=scale,
        shape="nested" if nested else "array",
    )


#: The eight instruments, in questionnaire-id order. Keys double as the ``metric=<M>`` partition
#: names (``WAI_SR``, ``MI_SAT`` keep their underscore -- a path segment may contain one; only an
#: ARM NAME token may not, which is why ``naming.py`` spells the same rubric ``MISAT``).
METRICS: Dict[str, Metric] = {
    "Q1": _metric("Q1", QuestionnaireID.Q1.value, "Q1", "Q1_Mean",
                  "Q1 (Session Satisfaction)"),
    "Q2": _metric("Q2", QuestionnaireID.Q2.value, "Q2", "Q2_Mean",
                  "Q2 (Relational Communication)"),
    "WAI_SR": _metric("WAI_SR", QuestionnaireID.WAI_SR.value, "WAI_SR", "WAI_TotalMean",
                      "WAI-SR (Working Alliance)"),
    "CSQ8": _metric("CSQ8", QuestionnaireID.CSQ8.value, "CSQ8", "CSQ8_Mean",
                    "CSQ-8 (Client Satisfaction)"),
    "MI_SAT": _metric("MI_SAT", QuestionnaireID.MI_SAT.value, "MI_SAT", "MI_Mean",
                      "MI-SAT (MI Satisfaction)"),
    "MITI": _metric("MITI", QuestionnaireID.MITI.value, "MITI", "MITI_GlobalMean",
                    "MITI (MI Integrity)"),
    "PCT": _metric("PCT", QuestionnaireID.PCT.value, "PCT", "PCT_ChangeProp",
                   "PCT (Patient Change-Talk)"),
    # The one lower-is-better instrument. MICI_Rate is MI-inconsistent acts per therapist turn.
    "MICI": _metric("MICI", QuestionnaireID.MICI.value, "MICI", "MICI_Rate",
                    "MICI (MI-Inconsistency)", higher_is_better=False),
}

#: Derived metrics: computed from stored ones, never written to the score lake (``partition=None``).
#:
#: ``Q1Q2`` is the TRAINING REWARD axis -- ``core/oracle.py`` averages the two rubrics unweighted,
#: so Q1's 5 items and Q2's 17 items each contribute half. Recomputing it here the same way keeps
#: the headline eval number on the same definition as the number the policy optimized. Do NOT pool
#: all 22 items instead: that is a different reward and would silently move the axis.
COMPOSITE_METRICS: Dict[str, Metric] = {
    "Q1Q2": Metric(
        key="Q1Q2",
        questionnaire_id=None,
        partition=None,
        score_column="Q1Q2_Mean",
        item_columns=(),
        display="Q1+Q2 (training reward)",
        higher_is_better=True,
        scale=(1, 5),
        shape="composite",
        composite_of=("Q1", "Q2"),
    ),
}

#: Everything the ``metric`` column of a long score frame may contain.
ALL_METRICS: Dict[str, Metric] = {**METRICS, **COMPOSITE_METRICS}

#: Left-to-right plot order: the training-reward composite first, then its two components, then the
#: held-out instruments. Anything not present in the data is dropped by the caller, not here.
METRIC_ORDER: Tuple[str, ...] = (
    "Q1Q2", "Q1", "Q2", "WAI_SR", "CSQ8", "MI_SAT", "MITI", "PCT", "MICI",
)

#: ``composite -> the metrics it is assembled from``, for the loader that assembles it after
#: reading. Derived from the registry, so the two cannot disagree about what Q1Q2 means.
COMPOSITES: Dict[str, Tuple[str, ...]] = {
    key: m.composite_of for key, m in COMPOSITE_METRICS.items()
}

#: ``metric -> (lake token, per-conversation mean column)``, in :data:`METRIC_ORDER`. A ``None``
#: token marks a composite, which has no parquet of its own.
#:
#: The flat projection of :data:`ALL_METRICS` that the data layer reads: it needs exactly two
#: facts per metric -- which ``metric=<M>`` directory to open and which column carries the number --
#: and pinning it to a two-tuple keeps a path resolver from having to know what a
#: :class:`Metric` is. Derived, never retyped.
QUESTIONNAIRES: Dict[str, Tuple[Optional[str], str]] = {
    key: (ALL_METRICS[key].partition, ALL_METRICS[key].score_column) for key in METRIC_ORDER
}

#: The metric the reward optimized, and therefore the default focus of every contrast.
TRAINING_REWARD_METRIC: str = "Q1Q2"

#: Metric keys AND detail columns where a smaller number is better. The metric keys are derived
#: from the registry so the two cannot disagree; the extra column names are the per-behaviour
#: detail that shares MICI's valence plus the patient's sustain-talk share, none of which is a
#: registry entry of its own.
LOWER_IS_BETTER: FrozenSet[str] = frozenset(
    {k for k, m in ALL_METRICS.items() if not m.higher_is_better}
    | {"MICI_Severity", "MICI_Rate", "MICI_BehaviorTotal"}
    | set(METRICS["MICI"].item_columns)
    | {"PCT_SustainTalk", "PCT_SustainTalk_prop"}
)


def metric(key: str) -> Metric:
    """Registry lookup by key.

    Raises:
        KeyError: naming every valid key. A typo'd metric name would otherwise filter a frame down
            to zero rows and render an empty figure that looks like missing data.
    """
    try:
        return ALL_METRICS[key]
    except KeyError:
        raise KeyError(
            f"unknown metric {key!r}; known metrics: {list(METRIC_ORDER)}"
        ) from None


def metric_for_qid(questionnaire_id: int) -> Metric:
    """The stored instrument for a ``QuestionnaireID`` value (accepts the enum member too).

    Raises:
        KeyError: if no stored instrument has that id (composites have none).
    """
    qid = int(getattr(questionnaire_id, "value", questionnaire_id))
    for m in METRICS.values():
        if m.questionnaire_id == qid:
            return m
    raise KeyError(
        f"no stored instrument for questionnaire_id={qid}; "
        f"ids on record: {sorted(m.questionnaire_id for m in METRICS.values())}"
    )


def is_lower_better(name: str) -> bool:
    """True for a metric key or detail column where a SMALLER value is the better result."""
    return name in LOWER_IS_BETTER


def sign_of(name: str) -> int:
    """``+1`` if higher is better for *name*, ``-1`` if lower is.

    Multiply a value (or a delta) by this before ranking, taking an ``argmax``, picking a "best"
    checkpoint, or choosing a diverging colour direction. Unknown names are treated as
    higher-is-better, which is the majority case and matches every rubric in the registry except
    MICI -- so a metric added without a registry entry fails safe for eight of nine instruments and
    wrong for the one that matters. Register new metrics; do not rely on the fallback.
    """
    return -1 if is_lower_better(name) else 1


def score_column(key: str) -> str:
    """The per-conversation column carrying *key*'s headline number."""
    return metric(key).score_column


def metric_partition(key: str) -> str:
    """The ``metric=<M>`` score-lake directory name for *key*.

    Raises:
        ValueError: for a composite -- it is computed, so asking for its partition means a caller
            is about to look for a directory that will never exist.
    """
    m = metric(key)
    if m.partition is None:
        raise ValueError(
            f"metric {key!r} is a composite of {list(m.composite_of)} and is never stored; "
            f"it has no metric= partition. Load its components and combine them."
        )
    return m.partition


# ==============================================================================
#  3. REPRODUCIBILITY
# ==============================================================================

#: THE resampling seed. Every bootstrap in the package AND every seaborn ``errorbar=`` callsite must
#: pass it.
#:
#: WARNING: seaborn's ``errorbar=("ci", 95)`` defaults to ``seed=None``, i.e. a fresh 1,000-sample
#: bootstrap on each call. Left unset, a re-render of an unchanged notebook on unchanged data
#: produces visibly different PNGs -- measured in Exp3 at ~6% of pixels across three consecutive
#: renders -- so every tracked figure churns in git on every render and a real change becomes
#: impossible to spot in a diff. Pass ``seed=BOOT_SEED`` at every seaborn callsite that draws a
#: bootstrap error bar, and to every resampler in ``stats``.
BOOT_SEED: int = 12345


# ==============================================================================
#  4. JUDGES
# ==============================================================================
#
# The score lake partitions on the GRADER:
#
#     data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/<EXP_NAME>/model_iter_<N>.parquet
#
# A judge grades after the fact, so a judge swap is re-runnable and cheap -- which is why it is a
# partition key rather than part of the arm name (the oracle and the patient, which are not
# re-runnable, ARE in the arm name). rep=0 is the full draw; rep>=1 are repeatability re-draws.

JUDGE_PARTITION = "judge="
REP_PARTITION = "rep="
METRIC_PARTITION = "metric="

# A partition token becomes a directory name and must satisfy core/config.py's
# `_assert_partition_token`: [A-Za-z0-9][A-Za-z0-9._-]* .
_TAG_VALID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_JUDGE_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def judge_tag(binding_or_model) -> str:
    """The ``judge=<tag>`` partition name for a grader.

    Args:
        binding_or_model: A ``roles.RoleBinding`` (its ``model`` is used) or a bare model id.

    Returns:
        ``roles.model_tag`` of the model: ``gemma4E4B``, ``gpt4m``, ``haiku45``.

    Warning:
        **This string is a directory name in the path of every score already written**
        (``data/eval_scores/judge=<tag>/...``). Changing how it is built does not migrate anything
        -- it orphans every parquet at once, and the loader then reports an unscored arm while the
        files sit on disk. If the scheme ever has to change, move the directories in the same
        commit.

        ``model_tag`` is deliberately many-to-one: ``gpt-4o-mini`` and ``gpt-4o-mini-2024-07-18``
        both tag ``gpt4m``, so a tag identifies a model FAMILY, not a snapshot. That is the right
        granularity for the arm name, and it is reused here so that the grader in ``judge=gpt4m``
        and the training oracle in ``...Ogpt4m`` are spelled identically and can be compared
        without a translation table. The cost is that re-judging with a NEWER snapshot of the same
        family writes into the same partition and silently interleaves two graders. If that
        becomes a real scenario, give the snapshot its own curated entry in ``roles._MODEL_TAGS``
        BEFORE scoring -- not afterwards.
    """
    model = getattr(binding_or_model, "model", binding_or_model)
    tag = model_tag(str(model))
    if not _TAG_VALID_RE.match(tag or ""):
        raise ValueError(
            f"judge_tag({binding_or_model!r}) produced {tag!r}, which is not a legal score-lake "
            f"partition token ([A-Za-z0-9][A-Za-z0-9._-]*). It would become a directory name."
        )
    return tag


#: The default grader: the same local Gemma that plays the oracle and the patient. Resolved from
#: ``roles.DEFAULT_JUDGE_MODEL`` so there is one place the open stack's model id is named.
DEFAULT_JUDGE_TAG: str = judge_tag(DEFAULT_JUDGE_MODEL)


def judge_dirname(tag: str = "") -> str:
    """Short display label for a judge tag. ``""`` resolves to :data:`DEFAULT_JUDGE_TAG`.

    Under the :func:`judge_tag` scheme a tag is already short (``gemma4E4B``) and this is the
    identity. It still trims a provider prefix and a trailing ISO release date, so a full-form tag
    -- one hand-written into the lake, or one carried over from Exp3's
    ``openai_gpt-4o-mini-2024-07-18`` -- renders as ``gpt-4o-mini`` rather than as a path fragment.

    Notes:
        This is a LABEL, not a path. Exp4's results tree has **no ``<judge>/`` level** -- every
        family puts the graders side by side inside one table or figure, so the label ends up in a
        column header or a legend entry, never in a directory. (Exp3 nested a ``<judge>/`` folder
        under its per-arm families and used this function to name it; that is the part Exp4 does not
        reproduce. See ``config.py``.)
    """
    t = (tag or DEFAULT_JUDGE_TAG).strip().strip("/\\")
    t = t.split("_", 1)[-1]
    return _JUDGE_DATE_SUFFIX_RE.sub("", t)


def judge_dir(tag: str = "", rep: Optional[int] = None) -> str:
    """``data/eval_scores/judge=<tag>[/rep=<r>]``. ``""`` resolves to :data:`DEFAULT_JUDGE_TAG`."""
    path = os.path.join(EVAL_SCORES_DIR, f"{JUDGE_PARTITION}{tag or DEFAULT_JUDGE_TAG}")
    return path if rep is None else os.path.join(path, f"{REP_PARTITION}{int(rep)}")


def available_judge_tags() -> List[str]:
    """Judge tags that actually have a partition on disk, default first, then sorted.

    An empty list means the score lake has not been written yet (or the Drive symlink is not
    mounted) -- not that scoring failed.

    Warning:
        On the Google Drive symlink, "the directory reads as empty" is NOT proof the scores are
        missing: the mount can wedge on a single folder and report zero entries while every file is
        present in Drive. Check the cloud before concluding an arm is unscored.
    """
    try:
        entries = os.listdir(EVAL_SCORES_DIR)
    except OSError:
        return []
    tags = sorted(
        name[len(JUDGE_PARTITION):]
        for name in entries
        if name.startswith(JUDGE_PARTITION) and os.path.isdir(os.path.join(EVAL_SCORES_DIR, name))
    )
    if DEFAULT_JUDGE_TAG in tags:  # the training-stack grader reads first everywhere else too
        tags.remove(DEFAULT_JUDGE_TAG)
        tags.insert(0, DEFAULT_JUDGE_TAG)
    return tags


def available_judge_reps(tag: str = "") -> List[int]:
    """Repetition indices present for one judge, ascending. ``rep=0`` is the full draw."""
    try:
        entries = os.listdir(judge_dir(tag))
    except OSError:
        return []
    reps: List[int] = []
    for name in entries:
        if not name.startswith(REP_PARTITION):
            continue
        try:
            reps.append(int(name[len(REP_PARTITION):]))
        except ValueError:
            continue
    return sorted(reps)


# ==============================================================================
#  5. LABELS AND PALETTE KEYS
# ==============================================================================
#
# Display layer ONLY. These never rename a data key: `arm`, `questionnaire` and every column name
# stay exactly as the loader produced them, because they are join and filter keys throughout the
# package. Only the rendered text and the colour lookup go through here.

#: The untrained policy (``model_iter_0``). Not an arm name -- it is what every arm starts from, so
#: it appears in every figure and needs a reserved label and colour.
BASE_ARM = "Base"

# `naming.ArmInfo.label` is the canonical arm key: "GRPO_LA5", "PTO_LA0", plus a suffix when
# something non-default was swapped ("PTO_LA5_indep", "GRPO_LA0_Ogpt4m").
_ARM_RE = re.compile(r"^(?P<method>PTO|GRPO)_LA(?P<k>\d+)(?:_(?P<extra>.+))?$")

#: Readable names for the four arms of the main grid. Anything else is parsed by :func:`arm_label`.
ARM_DISPLAY: Dict[str, str] = {
    "GRPO_LA0": "GRPO (K=0)",
    "GRPO_LA5": "GRPO (K=5)",
    "PTO_LA0": "PTO (K=0)",
    "PTO_LA5": "PTO (K=5)",
    BASE_ARM: "Base (untrained)",
}

#: Canonical arm colours: method chooses the hue, K chooses the shade. Okabe-Ito blue/vermillion
#: rather than red/green, so the METHOD contrast -- RQ-ii, the point of the experiment -- survives
#: the common colour-vision deficiencies and a greyscale print.
#:
#: Warning:
#:     ``plotting.py`` currently defines its own identical copy of this map. These two must stay
#:     byte-equal, and the fix is for ``plotting`` to import this one rather than for the values to
#:     be kept in step by hand: a figure and a legend that disagree about which hue is PTO is the
#:     kind of error nobody reads off a plot.
ARM_COLORS: Dict[str, str] = {
    "PTO_LA0": "#0072B2",     # blue
    "PTO_LA5": "#56B4E9",     # sky blue
    "GRPO_LA0": "#D55E00",    # vermillion
    "GRPO_LA5": "#E69F00",    # orange
    BASE_ARM: "#555555",      # neutral grey
}

#: Colours for metric-keyed figures (one line/bar per instrument).
METRIC_COLORS: Dict[str, str] = {
    "Q1Q2": "#2B2B2B",
    "Q1": "#6C9BD2",
    "Q2": "#1F4E93",
    "WAI_SR": "#4C9F70",
    "CSQ8": "#8C6BB1",
    "MI_SAT": "#D9A441",
    "MITI": "#4EA3A3",
    "PCT": "#C4739B",
    "MICI": "#B5462F",
}

#: Assigned in order to any key with no entry above (a new arm, a derived channel). Distinct from
#: every colour in :data:`ARM_COLORS`, so an unregistered arm is visibly unregistered.
FALLBACK_COLORS: Tuple[str, ...] = (
    "#556B2F", "#8B4A62", "#3F6F8F", "#9A6A3A", "#5E5E9E", "#7A7A2E",
)

#: Line style by look-ahead depth, so K reads off a monochrome print as well as off colour.
K_LINESTYLE: Dict[int, str] = {0: "-", 5: "--"}


def display_label(name: str) -> str:
    """Readable label for a metric key or a column, flagging lower-is-better.

    Registered metrics render their :attr:`Metric.display`; anything else falls through unchanged.
    A lower-is-better name gains a trailing ``" [lower better]"`` so MICI can never be read on the
    same "up is good" convention as the eight instruments around it. ASCII only -- these strings end
    up in Markdown tables, Excel headers and matplotlib text, and a unicode arrow renders as a box
    in at least one of the three.
    """
    m = ALL_METRICS.get(name)
    label = m.display if m is not None else name
    return f"{label} [lower better]" if is_lower_better(name) else label


def short_label(name: str) -> str:
    """Acronym-only label for dense axes (heatmap ticks, packed grids).

    :func:`display_label` renders ``"ACRONYM (gloss)"``, which overflows a 9x9 correlation-matrix
    tick. This returns the key itself -- already the instrument's acronym -- still flagged for
    lower-is-better. Put the gloss in the caption instead.
    """
    base = "Q1+Q2" if name == "Q1Q2" else name.replace("_", "-")
    return f"{base} [lower better]" if is_lower_better(name) else base


def arm_label(arm: str) -> str:
    """Readable arm label: ``"PTO_LA5"`` -> ``"PTO (K=5)"``, ``"PTO_LA5_indep"`` -> ``"PTO (K=5,
    indep)"``.

    Unknown strings pass through unchanged. Display layer only: the canonical ``arm`` key
    (``naming.ArmInfo.label``) is what every figure hues and every table groups on.
    """
    if arm in ARM_DISPLAY:
        return ARM_DISPLAY[arm]
    match = _ARM_RE.match(arm or "")
    if match is None:
        return arm
    extra = match.group("extra")
    suffix = f", {extra.replace('_', ', ')}" if extra else ""
    return f"{match.group('method')} (K={match.group('k')}{suffix})"


def k_of(arm: str) -> int:
    """Look-ahead depth parsed out of an arm key; ``0`` for anything unparseable, incl. ``Base``.

    THE canonical parse. In Exp3 this lived in ten modules with five mutually inconsistent bodies
    (``endswith("LA5")`` reads a hypothetical K=3 arm as K=0; ``int(arm.split("_LA")[1])`` raises on
    ``"Base"``), and because K is both a STYLE key and a GROUPING key, the disagreement mis-styled a
    new arm in some figures and mis-grouped it in others -- silently.
    """
    match = _ARM_RE.match(arm or "")
    return int(match.group("k")) if match else 0


def method_of(arm: str) -> str:
    """``"PTO"`` / ``"GRPO"`` parsed out of an arm key; ``""`` for anything else."""
    match = _ARM_RE.match(arm or "")
    return match.group("method") if match else ""


# ==============================================================================
#  6. PERSONAS
# ==============================================================================

#: The V3 patient permutation set, copied verbatim from Exp3 so the task is identical.
N_PERSONAS = 96

#: Persona traits recoverable per conversation, as ``system_prompts_builder`` names them.
#:
#: Pairing across iterations is on ``persona_id`` and nothing else. Exp4 names every conversation
#: file by the STABLE persona id (``pers07.csv`` is persona 7 in every iteration of every arm), so
#: unlike Exp3 there is no shuffle to replay -- but the file ORDER in a directory listing is still
#: not the persona order once a conversation is missing. Join on ``persona_id``.
PERSONA_COLS: Tuple[str, ...] = (
    "gender", "age_value", "problem", "problem_time", "tried_to_solve", "cooperation_level",
)

#: Display names for the ``cooperation_level`` trait (32 personas per level) and their plot order.
#: ONE copy on purpose: in Exp3 this map was duplicated into four modules and had already drifted
#: into two spellings of the same stratum, both of which shipped in the same results index.
COOP_LABEL: Dict[str, str] = {
    "High": "Cooperative",
    "StartLowAndChangesToHigh": "Warms up",
    "Low": "Resistant",
}
COOP_ORDER: Tuple[str, ...] = ("Cooperative", "Warms up", "Resistant")
