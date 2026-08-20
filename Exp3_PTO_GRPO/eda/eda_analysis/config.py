"""
config.py — the single EDA control surface: ``EdaConfig`` + ``notebook_setup`` (the "cell 1" kernel).

Every analysis notebook's cell 1 is flat globals bundled into one ``EdaConfig`` that is passed to
:func:`notebook_setup`. One place to choose the **FAMILY** (which results folder this notebook
owns), the **JUDGE** (which grader's scores are read), the arm/metric filters, the selection mode,
plot scales, and the export formats — reproducible and git-diffable (the run's config is in the
file, not in scattered cell hand-edits).

**The FAMILY knob (2026-08-18 reorg).** ``results/`` is organised by *research question*, not by
arm subset. ``family`` is a ``"<top>/<sub>"`` path validated against :data:`FAMILIES`::

    arms/        outcomes questionnaires validity heterogeneity training preference stats
                 (per-arm descriptives, ALL FOUR ARMS on one axis; rendered ONCE PER JUDGE →
                  results/arms/<sub>/{figures,tables}/<judge>/…)
    lookahead/   reward transfer behaviour mechanism replication   (RQ-i: K=0 vs K=5, both graders inside)
    method/      contrast                                          (RQ-ii: PTO vs GRPO at each K)
    compute/     cost                                              (GPU-h + API axis, budget sweeps)
    measurement/ validity                                          (judge validity, multi-judge)

Only the tops in :data:`PER_JUDGE_TOPS` (``arms``) get a ``<judge>/`` segment; every other family
is **judge-invariant** — its notebook loads both graders explicitly (:func:`scores_by_judge`) and
puts them side by side, so a path naming one grader would be false. Those families ignore the
``judge`` knob (a note is printed if one is passed).

**Arm filter default = every arm** (``ks=None``): the retired ``VIEW`` knob used to bind the arm
subset to the output folder (``L0`` / ``L5``); now the folder is the *question* and the four arms
share one axis. ``ks`` / ``methods`` / ``modes`` / ``arm_labels`` remain explicit filters.

Usage (notebook cell 1 — the contract every notebook follows)::

    import os, eda_analysis
    cfg = eda_analysis.EdaConfig(family="arms/outcomes", judge=os.environ.get("EDA_JUDGE", ""))
    S = eda_analysis.notebook_setup(cfg)          # S.RESULTS_DIR = results/arms/outcomes/
"""

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ── The results tree: top → subfamilies. Each entry is ONE notebook, notebooks/<top>/<sub>.ipynb,
# writing to results/<top>/<sub>/{figures,tables}/[<judge>/]. Add a family here + its notebook;
# `_selfcheck`'s `family map` asserts the two stay 1:1 and `tools/render_results.py` iterates this
# dict in order.
FAMILIES: Dict[str, List[str]] = {
    "arms":        ["outcomes", "questionnaires", "validity", "heterogeneity",
                    "training", "preference", "stats"],
    "lookahead":   ["reward", "transfer", "behaviour", "mechanism", "replication"],
    "method":      ["contrast"],
    "compute":     ["cost"],
    "measurement": ["validity"],
}
# Tops whose artifacts are PRODUCED BY one grader and therefore carry a <judge>/ segment (and are
# rendered once per judge on disk). Everything else is judge-invariant: cross-grader by design.
PER_JUDGE_TOPS = frozenset({"arms"})

#: Families that read ANOTHER family's **rendered artifacts** (not just the score lake), as
#: ``reader -> (producer, ...)``.
#:
#: ⚠ This is a render-ORDER constraint, and without it a from-clean run races. ``tools/
#: render_results.py`` schedules units of ``(top, judge)`` in a thread pool, so ``lookahead`` and
#: ``arms`` run concurrently — and ``lookahead/behaviour`` copies the tracked preference tables
#: (``update_lexical_push`` / ``generation_pool_means``) out of ``arms/preference``. When the pool
#: happens to reach the reader first the notebook dies on a missing file; when it does not, the
#: render passes. Observed both ways on the same machine minutes apart.
#:
#: Keep this map tiny. A family here is a family that is NOT self-contained; the better fix for a
#: new entry is usually to compute the value rather than re-read a rendered table.
FAMILY_READS: Dict[str, Tuple[str, ...]] = {
    "lookahead/behaviour": ("arms/preference",),
}


def producer_tops(families: List[str]) -> set:
    """Tops that must finish BEFORE the given families render (see :data:`FAMILY_READS`)."""
    out = set()
    for fam in families:
        for producer in FAMILY_READS.get(fam, ()):
            top = producer.split("/")[0]
            if top != fam.split("/")[0]:          # same-top order is the unit's own sequence
                out.add(top)
    return out


def all_families() -> List[str]:
    """Every ``"<top>/<sub>"`` family in :data:`FAMILIES` order."""
    return [f"{top}/{sub}" for top, subs in FAMILIES.items() for sub in subs]


def split_family(family: str) -> Tuple[str, str]:
    """Validate ``"<top>/<sub>"`` against :data:`FAMILIES` and return ``(top, sub)``.

    Raises ``ValueError`` with the full list of valid families on any mismatch — a typo here would
    otherwise create a phantom results folder that no index, summary or paper ledger points at.
    """
    fam = (family or "").strip().strip("/\\").replace("\\", "/")
    if not fam or fam.count("/") != 1:
        raise ValueError(f"family must be '<top>/<sub>', got {family!r}; "
                         f"valid: {all_families()}")
    top, sub = fam.split("/")
    if top not in FAMILIES or sub not in FAMILIES[top]:
        raise ValueError(f"unknown family {fam!r}; valid: {all_families()}")
    return top, sub


def is_per_judge(family_or_top: str) -> bool:
    """True if this family/top nests a ``<judge>/`` level (its top is in :data:`PER_JUDGE_TOPS`)."""
    top = (family_or_top or "").strip().strip("/\\").replace("\\", "/").split("/", 1)[0]
    return top in PER_JUDGE_TOPS


@dataclass
class EdaConfig:
    """All user-facing EDA knobs in one object (see module docstring)."""

    # ── THE knob: family = which results folder this notebook owns ────────────
    # "<top>/<sub>" from FAMILIES, e.g. "arms/outcomes", "lookahead/reward". "" = no export
    # routing (interactive exploration only: save_* raises until a family is set).
    family: str = ""

    # ── The OTHER axis: judge = which grader's scores to read ─────────────────
    # "" = the primary oracle. Any tag (e.g. "anthropic_claude-haiku-4-5") reads that grader's
    # partition of the score lake, data/eval_scores/judge=<tag>/rep=<judge_rep>/, and (for a
    # PER_JUDGE_TOPS family) routes exports to results/<family>/{figures,tables}/<judge>/.
    # Judge-invariant families ignore it (they load every grader via scores_by_judge).
    # Training-side analyses are NOT judge-swappable — see the note in constants.py.
    judge: str = ""
    # rep 0 = the full-grid draw every judge reports; >=1 are repeatability draws on the anchor
    # subset only, so a non-zero rep yields a mostly-empty frame outside those cells.
    judge_rep: int = 0

    # ── Arm selection (None = no filter on that axis; default = EVERY arm) ────
    methods: Optional[Sequence[str]] = None        # e.g. ["PTO"] | ["PTO","GRPO"]
    ks: Optional[Sequence[int]] = None             # e.g. [0] | [0, 5]; None = both K arms
    modes: Optional[Sequence[str]] = None          # e.g. ["greedy"] (PTO) / ["group"] (GRPO)
    arm_labels: Optional[Sequence[str]] = None     # explicit whitelist, e.g. ["PTO_LA0"]
    include_archived: bool = False

    # ── Metric selection ─────────────────────────────────────────────────────
    metrics: Optional[Sequence[str]] = None        # None = auto (present in data, canonical order)
    add_derived_mitiprof: bool = True              # append R:Q / %CR / %MICO rows (free, no rescore)
    warmth_only: bool = False                      # restrict default metric views to WARMTH_RUBRICS

    # ── Cross-model selection mode ───────────────────────────────────────────
    selection: str = "all"                         # "all" | "best" (best iter per arm by own oracle)
    focus_arms: Optional[Sequence[str]] = None     # default arm subset for overlay/trajectory figures
    focus_metric: str = "Q1Q2"                     # default metric for single-metric / contrast figures

    # ── Plot scales / style (None = inherit the module default / per-plot value) ──
    context: str = "notebook"                      # seaborn context: paper|notebook|talk|poster
    font_scale: float = 1.0
    dpi: int = 110                                  # inline preview dpi
    savefig_dpi: int = 200                         # exported raster dpi
    panel: Optional[Tuple[float, float]] = None    # (width, height) in per grid panel; None = inherit
    ncols: Optional[int] = None                    # default grid columns; None = inherit
    score_ylim: Optional[Tuple[float, float]] = None   # e.g. (1, 5); None = autoscale
    share_y: bool = False                          # share y-limits across grid panels
    palette_overrides: Dict[str, str] = field(default_factory=dict)

    # ── Exports ──────────────────────────────────────────────────────────────
    fig_formats: Tuple[str, ...] = ("png",)         # PNG images by default; ("png","pdf") for vector too
    table_formats: Tuple[str, ...] = ("md", "xlsx") # readable Markdown + sortable Excel workbook

    # ── Misc ─────────────────────────────────────────────────────────────────
    oracle_noise: float = 0.10                     # reproducibility band (|Δ| from partial-conv EDA)
    attach_persona: bool = True
    cache: bool = True                             # parquet-memoize scores_long/behavior (content-keyed)
    verbose: bool = True
    note: str = ""                                 # free-text, recorded in the provenance banner

    def with_(self, **overrides) -> "EdaConfig":
        """Return a copy with ``overrides`` applied (e.g. ``cfg.with_(selection='best')``)."""
        return replace(self, **overrides)

    def as_dict(self) -> dict:
        """Plain dict for the provenance banner / logging."""
        return {
            "family": self.family,
            "judge": self.judge, "judge_rep": self.judge_rep,
            "methods": list(self.methods) if self.methods else None,
            "ks": list(self.ks) if self.ks else None,
            "modes": list(self.modes) if self.modes else None,
            "arm_labels": list(self.arm_labels) if self.arm_labels else None,
            "include_archived": self.include_archived,
            "metrics": list(self.metrics) if self.metrics else None,
            "add_derived_mitiprof": self.add_derived_mitiprof,
            "warmth_only": self.warmth_only,
            "selection": self.selection,
            "focus_arms": list(self.focus_arms) if self.focus_arms else None,
            "focus_metric": self.focus_metric,
            "context": self.context, "font_scale": self.font_scale,
            "dpi": self.dpi, "savefig_dpi": self.savefig_dpi,
            "panel": list(self.panel) if self.panel else None, "ncols": self.ncols,
            "score_ylim": list(self.score_ylim) if self.score_ylim else None,
            "share_y": self.share_y,
            "palette_overrides": dict(self.palette_overrides),
            "fig_formats": list(self.fig_formats), "table_formats": list(self.table_formats),
            "oracle_noise": self.oracle_noise, "attach_persona": self.attach_persona,
            "cache": self.cache,
            "note": self.note,
        }


@dataclass
class Setup:
    """The shared notebook context (built by :func:`notebook_setup`)."""
    ARMS: list
    SCORES: pd.DataFrame
    PALETTE: dict
    METRICS: List[str]
    ORACLE_NOISE: float
    RESULTS_DIR: str        # results/<family>/ (the results root if no family was set)
    FAMILY: str             # "<top>/<sub>" or "" (no export routing)
    JUDGE: str              # active judge tag ("" = primary; always "" for judge-invariant families)
    CFG: EdaConfig


def _warn_partial_judge_coverage(scores, judge: str, n_expected: int = 96) -> None:
    """Loud warning when a second judge has NOT scored every conversation of every arm.

    A partially-landed sweep is the dangerous case: the loader silently returns whatever exists, so
    an arm scored on 41 of 96 conversations produces a mean that LOOKS like the primary judge's but
    rests on a different (smaller, noisier) sample — and persona-paired contrasts between two such
    arms overlap on only a fraction of personas. Better to shout than to publish quietly.
    """
    cov = (scores.groupby(["model", "questionnaire"])["file_index"].nunique()
           .rename("n").reset_index())
    partial = cov[cov.n < n_expected]
    if partial.empty:
        print(f"  [notebook_setup] judge coverage COMPLETE: every (model, metric) cell has "
              f"{n_expected} conversations.")
        return
    print(f"  [notebook_setup] ⚠ JUDGE COVERAGE INCOMPLETE for {judge}: "
          f"{len(partial)}/{len(cov)} (model, metric) cells below {n_expected} conversations "
          f"(n {int(partial.n.min())}–{int(partial.n.max())}). Arm means rest on unequal samples "
          f"and persona-paired contrasts lose power — finish the sweep "
          f"(Judge_Reliability.ipynb §3) before citing these numbers.")


def _resolve_judge(judge: str) -> str:
    """Normalise + validate a judge tag against the score lake (``""`` = primary)."""
    from . import reliability as _rel
    judge = (judge or "").strip().strip("/\\")
    if judge:
        known = _rel.judge_tags()
        if judge not in known:
            raise ValueError(f"unknown judge {judge!r}; scored judges on disk: {known or '(none)'}")
    return judge


def notebook_setup(cfg: Optional[EdaConfig] = None, **overrides) -> Setup:
    """Resolve the FAMILY + JUDGE, discover+filter arms, build ``scores_long`` + palette + metrics,
    route exports to ``results/<family>/``, write a provenance banner, and return a :class:`Setup`.

    ``cfg`` is an :class:`EdaConfig` (default = every arm / all present metrics / no family).
    ``**overrides`` patch individual fields for a quick tweak, e.g.
    ``notebook_setup(cfg, selection="best")`` or ``notebook_setup(cfg, ks=[0])``.

    Prints the family, the judge and the arm list. A judge-invariant family (top not in
    :data:`PER_JUDGE_TOPS`) ignores ``cfg.judge`` — such notebooks load every grader themselves via
    :func:`scores_by_judge` and export with no ``<judge>/`` level; the frame in ``S.SCORES`` is
    always the primary oracle's there.
    """
    from . import (discover_arms, load_scores_long, add_derived_mitiprof_rows,
                   QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, plotting, exports)
    from .data import filter_arms, set_cache
    from .constants import set_active_judge, judge_dirname

    cfg = cfg or EdaConfig()
    if overrides:
        cfg = cfg.with_(**overrides)

    set_cache(cfg.cache)                            # parquet memoization on/off for this session

    # ── Resolve the FAMILY: results root + whether a <judge>/ level applies ───
    family = (cfg.family or "").strip().strip("/\\").replace("\\", "/")
    if family:
        split_family(family)                        # raises on an unknown family
        per_judge = is_per_judge(family)
    else:
        per_judge = True                            # no family: judge still selects the score source

    # ── Resolve the JUDGE: score source (+ export leaf for per-judge families) ─
    judge = _resolve_judge(cfg.judge)
    if judge and not per_judge:
        print(f"  [notebook_setup] NOTE: family {family!r} is judge-INVARIANT — judge={judge!r} "
              f"ignored (its notebook loads every grader via scores_by_judge and exports with no "
              f"<judge>/ level). S.SCORES is the primary oracle's frame.")
        judge = ""
    set_active_judge(judge, cfg.judge_rep)

    plotting.set_style(cfg)
    exports.set_family(family, per_judge=per_judge)      # results/<family>/ (or none)
    exports.set_formats(cfg.fig_formats, cfg.table_formats)

    arms = discover_arms(include_archived=cfg.include_archived)
    arms = filter_arms(arms, methods=cfg.methods, ks=cfg.ks, modes=cfg.modes,
                       arm_labels=cfg.arm_labels)

    scores = load_scores_long(arms, attach_persona=cfg.attach_persona)
    if judge and not scores.empty:
        _warn_partial_judge_coverage(scores, judge)
    if cfg.add_derived_mitiprof and not scores.empty:
        scores = add_derived_mitiprof_rows(scores, arms)

    if scores.empty:
        palette, metrics = {}, []
    else:
        palette = plotting.arm_palette(sorted(scores.arm.unique()))
        present = set(scores.questionnaire.unique())
        if cfg.metrics:
            metrics = [m for m in cfg.metrics if m in present]
        else:
            base = WARMTH_RUBRICS if cfg.warmth_only else QUESTIONNAIRE_ORDER
            metrics = [m for m in base if m in present]

    # Provenance banner (printed + exported) so every regenerated figure set is traceable.
    if family and not scores.empty:
        exports.save_provenance(cfg, scores)

    # Ask the exports router rather than re-deriving the path: it is the single place that knows
    # the family root (and, for per-judge families, the judge leaf below it).
    results_dir = exports.family_root() if family else exports.RESULTS_DIR

    if cfg.verbose:
        print(f"FAMILY = {family or '(none — exports disabled)'}  |  JUDGE = "
              f"{judge_dirname(judge)}{'' if judge else ' (primary)'}"
              f"{'' if per_judge or not family else '  [judge-invariant family]'}")
        print("arms on disk (after filter):", [(a.label, len(a.iters)) for a in arms])
        if scores.empty:
            print("scores_long: EMPTY — no eval scores found on disk for these arms yet.")
        else:
            print("scores_long:", scores.shape, "| arms scored:", sorted(scores.arm.unique()))
            print("metrics:", metrics, "| selection:", cfg.selection)
        if family:
            leaf = os.path.relpath(exports._fig_dir(None), exports.RESULTS_DIR).replace(os.sep, "/")
            print(f"exports -> results/{leaf}/  (+ tables/)")

    return Setup(ARMS=arms, SCORES=scores, PALETTE=palette, METRICS=metrics,
                 ORACLE_NOISE=cfg.oracle_noise, RESULTS_DIR=results_dir,
                 FAMILY=family, JUDGE=judge, CFG=cfg)


def cross_k_arms(source) -> list:
    """The ARMS behind :func:`cross_k_scores` — both K arms of every method the config allows.

    Same filters as :func:`notebook_setup` minus ``ks``. With the post-reorg default (``ks=None`` =
    every arm) this is normally the SAME list as ``S.ARMS``; it only differs when a config set an
    explicit ``ks``. Kept because the RQ-i contrast is not only about rubric scores: the
    behaviour-channel frame (:func:`~eda_analysis.behavior.channel_scores_long`), the training-side
    pref frames (:mod:`~eda_analysis.pref`) and anything else arm-driven want the same cross-K arm
    list, and re-deriving it at each call site is how the filters drift apart.

    Read-only w.r.t. routing: the active judge and the export root are untouched.
    """
    from . import discover_arms
    from .data import filter_arms

    cfg = source.CFG if isinstance(source, Setup) else source
    return filter_arms(discover_arms(include_archived=cfg.include_archived),
                       methods=cfg.methods, ks=None, modes=cfg.modes, arm_labels=cfg.arm_labels)


def cross_k_scores(source) -> pd.DataFrame:
    """Scores for BOTH look-ahead arms of every method — an explicit ``ks`` filter dropped.

    Rebuilds ``scores_long`` with ``ks=None`` and *everything else* — method/mode/label filters,
    judge + rep, persona attachment, derived MITI-proficiency rows — taken from the active config.
    Since the 2026-08-18 reorg the default arm filter is already every arm, so with a default
    config this returns the SAME frame as ``S.SCORES``; it widens only when the config narrowed
    ``ks``. (Before the reorg it was the escape hatch that let a K-specific ``L0``/``L5`` view
    compute the K contrast at all.)

    ``source`` is the :class:`Setup` returned by :func:`notebook_setup` (or a bare
    :class:`EdaConfig`). **Read-only w.r.t. routing**: the active judge and the export root are left
    exactly as :func:`notebook_setup` set them. An explicit ``cfg.arm_labels`` whitelist is still
    honoured; ``ks`` is the only filter dropped.
    """
    from . import load_scores_long, add_derived_mitiprof_rows

    cfg = source.CFG if isinstance(source, Setup) else source
    arms = cross_k_arms(source)
    scores = load_scores_long(arms, attach_persona=cfg.attach_persona)
    if cfg.add_derived_mitiprof and not scores.empty:
        scores = add_derived_mitiprof_rows(scores, arms)
    return scores


def scores_by_judge(source, judges: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """``{judge_label: scores_long}`` — the same arm/metric filters under EACH grader on disk.

    The read path for judge-INVARIANT families (``lookahead/``, ``method/``, ``compute/``,
    ``measurement/``): they put the training oracle and the held-out judge side by side in one
    table/figure, so they need every grader's frame at once rather than the single active one.

    ``judges`` — judge TAGS (``""`` or :data:`~eda_analysis.constants.PRIMARY_JUDGE_TAG` = the
    primary); default = the primary + every second judge in the score lake
    (:func:`~eda_analysis.reliability.second_judge_tags`). Keys are the short display labels
    (:func:`~eda_analysis.constants.judge_dirname`: ``"gpt-4o-mini"``, ``"claude-haiku-4-5"``),
    in the order given, primary first by default.

    Filters come from ``source`` (a :class:`Setup` or :class:`EdaConfig`): methods / ks / modes /
    arm_labels / persona attachment / derived rows — exactly what :func:`notebook_setup` applied.
    The active judge is switched per load and **restored afterwards** (rep too), so calling this
    mid-notebook never re-points the caller's subsequent loads. Read-only w.r.t. export routing.

    ⚠ Never average the frames across judges — the primary WAS the training reward and the second
    judge is held out; combine contrasts or standardized quantities only (see reliability.py).
    """
    from . import discover_arms, load_scores_long, add_derived_mitiprof_rows
    from . import reliability as _rel
    from .constants import (set_active_judge, active_judge, active_judge_rep, judge_dirname,
                            PRIMARY_JUDGE_TAG)
    from .data import filter_arms

    cfg = source.CFG if isinstance(source, Setup) else source
    if judges is None:
        judges = [""] + list(_rel.second_judge_tags())
    tags = []
    for j in judges:
        t = "" if j in ("", None, PRIMARY_JUDGE_TAG) else _resolve_judge(j)
        if t not in tags:
            tags.append(t)

    arms = filter_arms(discover_arms(include_archived=cfg.include_archived),
                       methods=cfg.methods, ks=cfg.ks, modes=cfg.modes, arm_labels=cfg.arm_labels)
    prev_tag, prev_rep = active_judge(), active_judge_rep()
    out: Dict[str, pd.DataFrame] = {}
    try:
        for t in tags:
            set_active_judge(t, cfg.judge_rep)
            s = load_scores_long(arms, attach_persona=cfg.attach_persona)
            if cfg.add_derived_mitiprof and not s.empty:
                s = add_derived_mitiprof_rows(s, arms)
            out[judge_dirname(t)] = s
    finally:
        set_active_judge(prev_tag, prev_rep)
    return out
