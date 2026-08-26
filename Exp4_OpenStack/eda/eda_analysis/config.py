"""config.py -- the EDA control surface: ``FAMILIES``, ``EdaConfig``, ``notebook_setup``.

Every analysis notebook has the same cell 1. It names the family it owns, builds one
:class:`EdaConfig`, and hands it to :func:`notebook_setup`, which returns everything the rest of
the notebook needs. That is the entire contract::

    import os, eda_analysis
    cfg = eda_analysis.EdaConfig(family="arms/outcomes")
    S   = eda_analysis.notebook_setup(cfg)

The point of routing it through one object is that a rendered result is reproducible from a file
rather than from a sequence of hand-edited cells: the config is in the notebook, the notebook is in
git, and ``tools/render_results.py`` can re-run every family non-interactively by setting the same
fields.

The FAMILY knob
---------------
``results/`` is organised by RESEARCH QUESTION, not by arm subset. ``family`` is a ``"<top>/<sub>"``
path validated against :data:`FAMILIES`, and it maps 1:1 onto both a notebook and an output
directory::

    arms/outcomes       per-arm descriptives -- all four arms on one axis
    lookahead/reward    RQ-i:  K=0 vs K=5 within each optimizer
    method/contrast     RQ-ii: PTO vs GRPO at matched K
    compute/cost        the spend axis (GPU-hours from timing_sessions.jsonl, API calls)

    notebooks/<top>/<sub>.ipynb   <->   results/<top>/<sub>/{figures,tables}/

Four families is the whole of v1. Adding one means adding an entry here AND the matching notebook;
``tools/render_results.py`` iterates this dict, so an entry with no notebook renders nothing and a
notebook with no entry is never rendered.

The JUDGE knob, and why there is no ``<judge>/`` directory
---------------------------------------------------------
Exp4 has one judge dimension: the score lake partitions on ``judge=<tag>``, and a family reads
whichever grader it wants. It has **no per-judge results leaf**. Exp3 nested one
(``results/arms/<sub>/figures/<judge>/...``) and rendered those families once per grader; Exp4 does
not, because a table that puts two graders side by side -- which is the interesting table, since the
default judge shares a model with the training oracle and a second judge is genuinely held out --
cannot honestly live under a directory named after one of them. The path would assert something
false about its own contents.

So ``judge`` here selects **which grader's scores this notebook loads by default**. A family that
wants both loads both explicitly and labels the columns with
:func:`~eda_analysis.constants.judge_dirname`; the export path is the same either way.

Warning:
    Only EVAL-side numbers are judge-swappable. Anything read off the training side -- candidate
    rewards in ``generations.jsonl``, PTO's ``pairs.csv``, TensorBoard curves -- was produced by the
    TRAINING oracle while the run happened and cannot be re-graded after the fact. Re-rendering
    those under a second judge would emit identical numbers under a different grader's label,
    implying a measurement that never took place.

The sibling contract
--------------------
:func:`notebook_setup` is the only place the package's modules are wired together, so it is the one
place their interfaces have to agree. It calls exactly these, and nothing else::

    plotting.set_style(cfg)                                          -> None
    plotting.arm_palette(labels)                                     -> {label: colour}
    exports.set_family(family)                                       -> None   ("" disables saving)
    exports.set_formats(*, fig_formats=None, table_formats=None)     -> None
    exports.family_root()                                            -> str
    exports.save_provenance(cfg, scores)                             -> str
    data.set_cache(enabled)                                          -> None
    data.discover_arms()                                             -> [Arm]
    data.filter_arms(arms, *, methods, ks, modes, arm_labels)        -> [Arm]
    data.load_scores_long(arms, *, judge, rep, attach_persona)       -> DataFrame

``load_scores_long`` takes the judge as an ARGUMENT rather than reading a module-global "active
judge" (Exp3's shape). Two reasons: a family that renders both graders would otherwise have to
set and restore global state around each load, and ``render_results.py`` runs families in parallel,
where a process-global grader selection is a race.

Of the long frame it returns, this module reads only two columns -- ``arm_label`` (the display key
every figure hues on) and ``metric`` -- so a change anywhere else in ``SCORE_COLUMNS`` does not
reach here.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from .constants import (
    ALL_METRICS,
    BOOT_SEED,
    DEFAULT_JUDGE_TAG,
    METRIC_ORDER,
    RESULTS_DIR,
    available_judge_reps,
    available_judge_tags,
    judge_dirname,
)

if TYPE_CHECKING:  # pragma: no cover -- annotations only; the package never imports pandas eagerly
    import pandas as pd

__all__ = [
    "FAMILIES",
    "all_families",
    "split_family",
    "EdaConfig",
    "Setup",
    "notebook_setup",
]


# ==============================================================================
#  THE FAMILY MAP
# ==============================================================================

#: ``top -> [sub, ...]``. Each entry is ONE notebook (``notebooks/<top>/<sub>.ipynb``) writing to
#: ONE directory (``results/<top>/<sub>/``). Order matters: ``tools/render_results.py`` renders in
#: this order, and ``arms`` first means the descriptive tables exist before anything that a reader
#: would want to check a contrast against.
#:
#: Note:
#:     Exp3 needed a ``FAMILY_READS`` map because one family re-read another's RENDERED tables,
#:     which made the render order load-bearing and raced when the driver parallelised. Exp4's
#:     families are self-contained: each recomputes what it needs from the score lake. Keep it that
#:     way -- the fix for "I need that number" is to compute it, not to read a sibling's Markdown.
FAMILIES: Dict[str, List[str]] = {
    "arms": ["outcomes"],
    "lookahead": ["reward"],
    "method": ["contrast"],
    "compute": ["cost"],
}


def all_families() -> List[str]:
    """Every ``"<top>/<sub>"`` family, in :data:`FAMILIES` order."""
    return [f"{top}/{sub}" for top, subs in FAMILIES.items() for sub in subs]


def split_family(family: str) -> Tuple[str, str]:
    """Validate ``"<top>/<sub>"`` against :data:`FAMILIES` and return ``(top, sub)``.

    Accepts either separator (a Windows-style ``arms\\outcomes`` normalises) and strips surrounding
    slashes.

    Raises:
        ValueError: listing every valid family. A typo would otherwise create a phantom results
            directory that no index, no summary and no render driver ever points at -- the notebook
            appears to succeed and its output is invisible.
    """
    fam = (family or "").strip().strip("/\\").replace("\\", "/")
    if not fam or fam.count("/") != 1:
        raise ValueError(
            f"family must be '<top>/<sub>', got {family!r}; valid families: {all_families()}"
        )
    top, sub = fam.split("/")
    if top not in FAMILIES or sub not in FAMILIES[top]:
        raise ValueError(f"unknown family {fam!r}; valid families: {all_families()}")
    return top, sub


# ==============================================================================
#  THE CONFIG
# ==============================================================================


@dataclass(frozen=True)
class EdaConfig:
    """Every user-facing EDA knob, in one frozen object.

    Frozen so that a cell halfway down a notebook cannot quietly re-point the run: use
    :meth:`with_` to derive a variant, which makes the change visible at the call site and leaves
    the original config intact for the provenance record.

    Attributes:
        family: ``"<top>/<sub>"`` from :data:`FAMILIES` -- the results directory this notebook owns.
            ``""`` means interactive exploration with exports disabled.
        judge: Which grader's scores to load. ``""`` resolves to
            :data:`~eda_analysis.constants.DEFAULT_JUDGE_TAG` (the local Gemma). Validated against
            the tags actually present in the score lake.
        judge_rep: Repetition index. ``0`` is the full draw; ``>=1`` are repeatability re-draws and
            typically cover a subset, so a non-zero rep yields a mostly-empty frame.
        methods, ks, modes, arm_labels, experiment_names: Arm filters. ``None`` on an axis means
            "no filter", and the default is therefore every arm on disk -- the four-arm grid
            shares one axis in every family, so narrowing is the exception.
            ``experiment_names`` matches the FULL folder identity, for pinning exactly one arm
            (e.g. excluding a quicktest sibling that shares the short label's axes).
        metrics: Metric keys to report, in :data:`~eda_analysis.constants.METRIC_ORDER` order.
            ``None`` means every metric present in the data.
        focus_metric: The single metric a one-panel figure defaults to. ``Q1Q2`` is the training
            reward, so it is the axis the policy actually optimized.
        attach_persona: Join persona traits onto every score row. Needed by anything that pairs on
            or stratifies by persona.
        context ... share_y: Plot style, applied by ``plotting.set_style``.
        fig_formats, table_formats: Export formats. Markdown is readable in a diff; xlsx is what
            gets opened in a meeting.
        cache: Let the data layer memoise loaded frames for this session.
        verbose: Print the setup banner.
        note: Free text recorded in the provenance record -- what this render was for.

    Notes:
        The mutable-container fields (``metrics``, ``palette_overrides``, ...) make instances
        unhashable in practice despite ``frozen=True``. They are configuration, not keys; nothing
        in the package hashes one.
    """

    # -- what this notebook owns ------------------------------------------------
    family: str = ""

    # -- which grader ------------------------------------------------------------
    judge: str = ""
    judge_rep: int = 0

    # -- arm selection (None = no filter on that axis) ---------------------------
    methods: Optional[Sequence[str]] = None        # ["PTO"] | ["PTO", "GRPO"]
    ks: Optional[Sequence[int]] = None             # [0] | [0, 5]
    modes: Optional[Sequence[str]] = None          # ["greedy"] | ["indep"]  (PTO only)
    arm_labels: Optional[Sequence[str]] = None     # explicit whitelist, e.g. ["PTO_LA0"]
    experiment_names: Optional[Sequence[str]] = None  # whitelist on the FULL folder identity

    # -- metric selection --------------------------------------------------------
    metrics: Optional[Sequence[str]] = None
    focus_metric: str = "Q1Q2"
    attach_persona: bool = True

    # -- style -------------------------------------------------------------------
    context: str = "notebook"                      # seaborn context: paper|notebook|talk|poster
    font_scale: float = 1.0
    dpi: int = 110                                 # inline preview
    savefig_dpi: int = 200                         # exported raster
    panel: Optional[Tuple[float, float]] = None    # (width, height) per grid panel
    ncols: Optional[int] = None
    score_ylim: Optional[Tuple[float, float]] = None
    share_y: bool = False
    palette_overrides: Dict[str, str] = field(default_factory=dict)

    # -- exports -----------------------------------------------------------------
    fig_formats: Tuple[str, ...] = ("png",)
    table_formats: Tuple[str, ...] = ("md", "xlsx")

    # -- misc --------------------------------------------------------------------
    boot_seed: int = BOOT_SEED
    cache: bool = True
    verbose: bool = True
    note: str = ""

    def with_(self, **overrides) -> "EdaConfig":
        """A copy with *overrides* applied, e.g. ``cfg.with_(ks=[5], note='K=5 only')``."""
        return replace(self, **overrides)

    def as_dict(self) -> Dict[str, Any]:
        """Plain JSON-able dict for the provenance record and the setup banner.

        Sequences are copied to lists and tuples to lists so the result survives ``json.dumps``
        unchanged -- the provenance file is meant to be diffable, and a tuple that serialises one
        way today and another tomorrow makes every record look modified.
        """
        def _list(value):
            return list(value) if value is not None else None

        return {
            "family": self.family,
            "judge": self.judge, "judge_rep": self.judge_rep,
            "methods": _list(self.methods), "ks": _list(self.ks),
            "modes": _list(self.modes), "arm_labels": _list(self.arm_labels),
            "experiment_names": _list(self.experiment_names),
            "metrics": _list(self.metrics), "focus_metric": self.focus_metric,
            "attach_persona": self.attach_persona,
            "context": self.context, "font_scale": self.font_scale,
            "dpi": self.dpi, "savefig_dpi": self.savefig_dpi,
            "panel": _list(self.panel), "ncols": self.ncols,
            "score_ylim": _list(self.score_ylim), "share_y": self.share_y,
            "palette_overrides": dict(self.palette_overrides),
            "fig_formats": list(self.fig_formats), "table_formats": list(self.table_formats),
            "boot_seed": self.boot_seed, "cache": self.cache, "note": self.note,
        }


@dataclass(frozen=True)
class Setup:
    """What a notebook gets back from :func:`notebook_setup`.

    Attributes:
        ARMS: Discovered arms after filtering. Each carries at least ``.label`` and ``.iters``.
        SCORES: The long score frame for :attr:`JUDGE`. Possibly EMPTY -- an arm that has trained
            but not been scored yet is a normal state, and every family must render something
            sensible rather than raise.
        PALETTE: ``{arm label: colour}``, covering every arm in :attr:`SCORES`.
        METRICS: Metric keys present in the data, in canonical plot order.
        RESULTS_DIR: ``results/<top>/<sub>/`` for this family (the results root if none was set).
        FAMILY: The validated ``"<top>/<sub>"``, or ``""``.
        JUDGE: The RESOLVED judge tag -- never ``""``, so a figure caption can always name the
            grader. Use ``judge_dirname`` for the short label.
        CFG: The config this setup came from, overrides already applied. Read ``S.CFG`` rather than
            the notebook's own ``cfg`` variable; they differ whenever ``notebook_setup`` was passed
            keyword overrides.
    """

    ARMS: List[Any]
    SCORES: "pd.DataFrame"
    PALETTE: Dict[str, str]
    METRICS: List[str]
    RESULTS_DIR: str
    FAMILY: str
    JUDGE: str
    CFG: EdaConfig


# ==============================================================================
#  WIRING
# ==============================================================================


#: The two columns of ``data.load_scores_long``'s frame that this module reads. ``arm_label`` is the
#: display key (``naming.ArmInfo.label``) every figure hues and every table groups on; ``metric`` is
#: a key of :data:`~eda_analysis.constants.ALL_METRICS`.
_ARM_COLUMN = "arm_label"
_METRIC_COLUMN = "metric"


def _sibling(module: str, symbol: str):
    """Fetch ``eda_analysis.<module>.<symbol>``, failing with the contract if it is not there.

    Resolved at call time, not at import: this module must stay importable on its own (so
    ``import eda_analysis`` costs nothing and pulls in no pandas), and a plain ``AttributeError``
    from three frames down is a bad way to learn that a contracted function was renamed.
    """
    try:
        mod = importlib.import_module(f".{module}", __package__)
    except ImportError as ex:
        raise ImportError(
            f"eda_analysis.{module} is required by notebook_setup but could not be imported "
            f"({ex}). See the sibling contract in eda_analysis/config.py."
        ) from ex
    try:
        return getattr(mod, symbol)
    except AttributeError as ex:
        raise ImportError(
            f"eda_analysis.{module}.{symbol} is part of the sibling contract in "
            f"eda_analysis/config.py but does not exist. One of the two is out of date."
        ) from ex


def _resolve_judge(judge: str, rep: int, *, verbose: bool = True) -> str:
    """Normalise a judge tag and check it against the score lake.

    ``""`` resolves to :data:`~eda_analysis.constants.DEFAULT_JUDGE_TAG`. An empty lake is accepted
    with a note -- "nothing has been scored yet" is a normal early state, and refusing to set up a
    notebook over it would make the EDA unusable exactly when someone is checking whether scoring
    worked.

    An EXPLICIT tag that is not on disk raises; the IMPLICIT default falls back to the first
    grader present. Every notebook's cell 1 passes ``""``, and every family loads through
    ``scores_by_judge`` (all graders, no judge level), so a lake whose only grader is not the
    default is a perfectly analysable lake -- raising there would fail all four notebooks and
    leave ``render_results.py`` with nothing rendered over data that is fine.

    Raises:
        ValueError: when a judge was named explicitly and the lake has partitions but not that
            one. That is a typo, and the failure it prevents is silent: the loader would find no
            files and every family would render an empty frame that looks like an unscored arm.
    """
    requested = (judge or "").strip().strip("/\\")
    tag = requested or DEFAULT_JUDGE_TAG
    on_disk = available_judge_tags()

    if not on_disk:
        if verbose:
            print(
                f"  [notebook_setup] NOTE: no judge partitions on disk yet -- accepting "
                f"judge={tag!r} unvalidated. (Score lake: {os.path.join('data', 'eval_scores')}. "
                f"If a sweep HAS run, check the Drive mount: a wedged folder reads as empty.)"
            )
        return tag

    if tag not in on_disk:
        if requested:
            raise ValueError(
                f"unknown judge {tag!r}; graders present in the score lake: {on_disk}. "
                f"(A judge tag is a directory name -- see constants.judge_tag.)"
            )
        # Nobody asked for this tag -- it is just the default. Use what is actually there.
        tag = on_disk[0]
        if verbose:
            print(
                f"  [notebook_setup] NOTE: the default grader {DEFAULT_JUDGE_TAG!r} has no "
                f"partition in the score lake; using judge={tag!r} instead (present: {on_disk}). "
                f"Only the arms/* leaf name depends on this -- every other family reads all "
                f"graders."
            )

    reps = available_judge_reps(tag)
    if reps and int(rep) not in reps and verbose:
        print(
            f"  [notebook_setup] WARNING: judge={tag!r} has reps {reps} on disk but rep={rep} was "
            f"requested; the score frame will be empty. rep=0 is the full draw."
        )
    return tag


def _select_metrics(scores, requested: Optional[Sequence[str]]) -> List[str]:
    """Metric keys to report: those present in *scores*, in canonical order.

    An explicitly requested metric that is absent from the DATA is dropped rather than raising --
    an arm scored on six of nine instruments is a normal in-flight state. An unregistered key is a
    typo and raises, because it would otherwise filter the frame to nothing and render an empty
    figure that looks like missing data.
    """
    # Validated BEFORE the empty-frame shortcut: a typo must be reported even on the first run
    # against an unscored lake, which is exactly when a notebook is being written.
    if requested:
        for key in requested:
            if key not in ALL_METRICS:
                raise ValueError(
                    f"unknown metric {key!r} in EdaConfig.metrics; known: {list(METRIC_ORDER)}"
                )
        wanted = [k for k in METRIC_ORDER if k in set(requested)]
        wanted += [k for k in requested if k not in METRIC_ORDER]
    else:
        wanted = list(METRIC_ORDER)

    if scores is None or getattr(scores, "empty", True):
        return []
    present = set(scores[_METRIC_COLUMN].unique())
    return [k for k in wanted if k in present]


def notebook_setup(cfg: Optional[EdaConfig] = None, **overrides) -> Setup:
    """Resolve the family and the judge, load the arms and their scores, route exports, report.

    Args:
        cfg: The notebook's :class:`EdaConfig`. Defaults to every arm, every metric, no family
            (exports disabled).
        **overrides: Patch individual fields for a one-off, e.g.
            ``notebook_setup(cfg, ks=[5], verbose=False)``. Applied via :meth:`EdaConfig.with_`, so
            ``S.CFG`` records what actually ran.

    Returns:
        A :class:`Setup`.

    Raises:
        ValueError: on an unknown family, an unknown judge, or an unregistered metric key.
        ImportError: if a sibling module does not satisfy the contract in this module's docstring.

    Notes:
        The order below is not arbitrary. The family is validated FIRST, before any disk work, so a
        typo costs a millisecond rather than a full score load. Export routing is configured BEFORE
        anything is computed, so a notebook cannot compute a figure it then has nowhere to put.
        Provenance is written LAST, from the frame that was actually loaded, so the record describes
        the render rather than the intent.

        An empty ``S.SCORES`` is returned, not raised on: arms exist on disk before they are
        scored, and a family that renders "no data yet" is more useful than one that cannot open.
    """
    cfg = cfg or EdaConfig()
    if overrides:
        cfg = cfg.with_(**overrides)

    # 1. FAMILY -- validated before any I/O.
    family = (cfg.family or "").strip().strip("/\\").replace("\\", "/")
    if family:
        split_family(family)

    # 2. JUDGE -- resolved against what is actually in the score lake.
    judge = _resolve_judge(cfg.judge, cfg.judge_rep, verbose=cfg.verbose)

    # 3. Style, then export routing -- both before anything is computed.
    _sibling("plotting", "set_style")(cfg)
    _sibling("exports", "set_family")(family)
    _sibling("exports", "set_formats")(
        fig_formats=cfg.fig_formats, table_formats=cfg.table_formats
    )
    _sibling("data", "set_cache")(cfg.cache)

    # 4. Arms.
    arms = _sibling("data", "discover_arms")()
    arms = _sibling("data", "filter_arms")(
        arms, methods=cfg.methods, ks=cfg.ks, modes=cfg.modes, arm_labels=cfg.arm_labels,
        experiment_names=cfg.experiment_names,
    )

    # 5. Scores for THIS judge. The judge travels as an argument, never as global state -- families
    #    that render both graders call this again with the other tag, and render_results.py runs
    #    families concurrently.
    scores = _sibling("data", "load_scores_long")(
        arms, judge=judge, rep=cfg.judge_rep, attach_persona=cfg.attach_persona
    )

    # 6. Palette + metric list. Overrides are applied HERE rather than inside arm_palette, so the
    #    plotting layer keeps one colour rule and the config keeps one escape hatch from it.
    if scores is None or getattr(scores, "empty", True):
        palette: Dict[str, str] = {}
    else:
        palette = dict(_sibling("plotting", "arm_palette")(sorted(scores[_ARM_COLUMN].unique())))
    palette.update(cfg.palette_overrides or {})
    metrics = _select_metrics(scores, cfg.metrics)

    # 7. Provenance, from the frame that was actually loaded.
    if family:
        _sibling("exports", "save_provenance")(cfg, scores)

    results_dir = _sibling("exports", "family_root")() if family else RESULTS_DIR

    if cfg.verbose:
        print(f"FAMILY = {family or '(none -- exports disabled)'}"
              f"  |  JUDGE = {judge_dirname(judge)}  [{judge}] rep={cfg.judge_rep}")
        print(f"arms on disk (after filter): "
              f"{[(a.label, len(getattr(a, 'iters', ()) or ())) for a in arms]}")
        if scores is None or getattr(scores, "empty", True):
            print("scores: EMPTY -- these arms have no scores in this judge's partition yet.")
        else:
            print(f"scores: {scores.shape} | "
                  f"arms scored: {sorted(scores[_ARM_COLUMN].unique())}")
            print(f"metrics: {metrics} | focus: {cfg.focus_metric}")
        if family:
            print(f"exports -> {os.path.relpath(results_dir, RESULTS_DIR).replace(os.sep, '/')}/"
                  f"  (figures/ + tables/)")

    return Setup(
        ARMS=arms,
        SCORES=scores,
        PALETTE=palette,
        METRICS=metrics,
        RESULTS_DIR=results_dir,
        FAMILY=family,
        JUDGE=judge,
        CFG=cfg,
    )
