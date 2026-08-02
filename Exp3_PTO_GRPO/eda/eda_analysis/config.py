"""
config.py — the single EDA control surface: ``EdaConfig`` + ``notebook_setup`` (the "cell 1" kernel).

Every analysis notebook's cell 1 is flat globals bundled into one ``EdaConfig`` that is passed to
:func:`notebook_setup`. One place to choose the **VIEW** (which look-ahead arms + which results
subfolder), the metrics, the selection mode, plot scales, and where artifacts are saved —
reproducible and git-diffable (the run's config is in the file, not in scattered cell hand-edits).

**The VIEW knob.** ``view`` is the one control that matters day-to-day. It sets BOTH:
  - the arm filter — ``"all"`` = every arm, ``"L0"`` = K=0 arms (PTO_LA0/GRPO_LA0),
    ``"L5"`` = K=5 arms (PTO_LA5/GRPO_LA5); and
  - the results root — artifacts land under ``results/<view>/figures|tables/<group>/``.
So ``results/`` ends up with three parallel trees (``all/``, ``L0/``, ``L5/``). An explicit
``ks=[...]`` still overrides the view's arm filter (the view is a convenience default).

All fields have safe defaults, so ``EdaConfig()`` = the ``all`` view, all present metrics,
all-models selection, the old plot style.

Usage (notebook cell 1)::

    import eda_analysis
    cfg = eda_analysis.EdaConfig(view="L0", export_group="eval")   # K=0 arms -> results/L0/.../eval/
    S = eda_analysis.notebook_setup(cfg)
"""

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# VIEW -> ks arm filter. ``all`` = no K filter; ``L0`` = K=0 only; ``L5`` = K=5 only.
# To add a new K view (e.g. K=2 data lands): add '"L2": [2]' here + '"l2": "L2"' to the aliases
# + "L2" to render_views.VIEWS + the set asserted in _selfcheck._c_view_map.
_VIEW_KS: Dict[str, Optional[List[int]]] = {"all": None, "L0": [0], "L5": [5]}
# Case-insensitive input -> canonical view name (so "l0"/"L0" both work; folder stays "L0").
_VIEW_ALIASES: Dict[str, str] = {"all": "all", "l0": "L0", "l5": "L5"}

# The view that OWNS the K0-vs-K5 (RQ-i) artifacts. RQ-i is the one contrast no K-specific view can
# serve from its own ``S.SCORES`` — ``L0`` holds the K=0 arms and ``L5`` the K=5 arms, so a paired
# K comparison is empty in either. ``cross_k_scores`` supplies the frame; this constant decides WHERE
# the resulting tables/figures land, so there is exactly one copy to keep in sync (the look-ahead
# view, whose SUMMARY already narrates RQ-i). Other views print a pointer instead of a second copy.
RQ_I_VIEW = "L5"


@dataclass
class EdaConfig:
    """All user-facing EDA knobs in one object (see module docstring)."""

    # ── THE knob: view = which arms + which results subfolder ─────────────────
    view: str = "all"                              # "all" | "L0" | "L5" (arm filter + results/<view>/)

    # ── The OTHER axis: judge = which grader's scores to read ─────────────────
    # "" = the primary oracle. Any tag (e.g. "anthropic_claude-haiku-4-5") reads that grader's
    # partition of the score lake, data/eval_scores/judge=<tag>/rep=<judge_rep>/, and routes
    # exports to results/<view>/figures/<family>/<judge>/. Orthogonal to `view`: view filters
    # ARMS, judge selects the SCORE SOURCE. Training-side analyses are NOT judge-swappable — see
    # the note in constants.py and the warning notebook_setup emits.
    judge: str = ""
    # rep 0 = the full-grid draw every judge reports; >=1 are repeatability draws on the anchor
    # subset only, so a non-zero rep yields a mostly-empty frame outside those cells.
    judge_rep: int = 0

    # ── Arm selection (None = no filter on that axis; ks overrides the view) ───
    methods: Optional[Sequence[str]] = None        # e.g. ["PTO"] | ["PTO","GRPO"]
    ks: Optional[Sequence[int]] = None             # e.g. [0] | [0, 5]  — set = overrides view's K filter
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
    export_group: str = ""                         # results/<view>/<figures|tables>/<group>/ ; "" = flat
    fig_formats: Tuple[str, ...] = ("png",)         # PNG images by default; ("png","pdf") for vector too
    table_formats: Tuple[str, ...] = ("md", "xlsx") # readable Markdown + sortable Excel workbook
    results_subdirs: bool = True                   # route into per-group subfolders

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
            "view": self.view,
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
            "export_group": self.export_group,
            "fig_formats": list(self.fig_formats), "table_formats": list(self.table_formats),
            "results_subdirs": self.results_subdirs,
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
    RESULTS_DIR: str        # the VIEW-specific results dir (results/<view>/)
    VIEW: str
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


def notebook_setup(cfg: Optional[EdaConfig] = None, **overrides) -> Setup:
    """Discover+filter arms (by the VIEW), build ``scores_long`` + palette + metrics, set the
    view-aware export root, write a provenance banner, and return a :class:`Setup`.

    ``cfg`` is an :class:`EdaConfig` (default = the ``all`` view / all present metrics).
    ``**overrides`` patch individual fields for a quick tweak, e.g.
    ``notebook_setup(cfg, view="L0")`` or ``notebook_setup(cfg, selection="best")``.
    """
    from . import (discover_arms, load_scores_long, add_derived_mitiprof_rows,
                   QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, plotting, exports)
    from .data import filter_arms, set_cache

    cfg = cfg or EdaConfig()
    if overrides:
        cfg = cfg.with_(**overrides)

    set_cache(cfg.cache)                            # parquet memoization on/off for this session

    # ── Resolve the VIEW: arm filter (ks) + results root ──────────────────────
    view = _VIEW_ALIASES.get((cfg.view or "all").strip().lower())
    if view is None:
        raise ValueError(f"unknown view {cfg.view!r} (expected one of {list(_VIEW_KS)})")
    if cfg.ks is not None:
        effective_ks = cfg.ks                      # explicit ks wins over the view default
        if view != "all" and set(cfg.ks) != set(_VIEW_KS[view] or []):
            print(f"  [notebook_setup] NOTE: explicit ks={list(cfg.ks)} overrides view={view!r} "
                  f"(arms filtered by ks, results still under results/{view}/).")
    else:
        effective_ks = _VIEW_KS[view]

    # ── Resolve the JUDGE: score source + results/judges/<tag>/ prefix ────────
    from .constants import set_active_judge, judge_label
    from . import reliability as _rel
    judge = (cfg.judge or "").strip().strip("/\\")
    if judge:
        known = _rel.judge_tags()
        if judge not in known:
            raise ValueError(f"unknown judge {judge!r}; scored judges on disk: {known or '(none)'}")
    set_active_judge(judge, cfg.judge_rep)
    if judge:
        print(f"  [notebook_setup] JUDGE={judge_label(judge)} — reading "
              f"eval_scores/judge={judge}/rep={cfg.judge_rep}/, exporting to "
              f"results/{view}/figures/<family>/{judge_label(judge)}/")

    plotting.set_style(cfg)
    exports.set_view(view)                                                   # results/<view>/...
    exports.set_export_group(cfg.export_group if cfg.results_subdirs else "")
    exports.set_formats(cfg.fig_formats, cfg.table_formats)

    arms = discover_arms(include_archived=cfg.include_archived)
    arms = filter_arms(arms, methods=cfg.methods, ks=effective_ks, modes=cfg.modes,
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
    if not scores.empty:
        exports.save_provenance(cfg, scores)

    # Ask the exports router rather than re-deriving the path: it is the single place that knows
    # about BOTH the view and the active judge (results/judges/<tag>/<view>/). String-joining
    # RESULTS_DIR + view here silently pointed Setup.RESULTS_DIR (and anything using it, e.g.
    # build_index) at the primary tree while the actual saves went to the judge's.
    results_dir = exports._results_root()

    if cfg.verbose:
        print(f"VIEW = {view}  (ks={effective_ks if effective_ks is not None else 'all'})")
        print("arms on disk (after view filter):", [(a.label, len(a.iters)) for a in arms])
        if scores.empty:
            print("scores_long: EMPTY — no eval scores found on disk for this view yet.")
        else:
            print("scores_long:", scores.shape, "| arms scored:", sorted(scores.arm.unique()))
            print("metrics:", metrics, "| selection:", cfg.selection)
        grp = cfg.export_group or "(flat)"
        print(f"exports -> {results_dir}  [group: {grp}]")

    return Setup(ARMS=arms, SCORES=scores, PALETTE=palette, METRICS=metrics,
                 ORACLE_NOISE=cfg.oracle_noise, RESULTS_DIR=results_dir, VIEW=view, CFG=cfg)


def cross_k_scores(source) -> pd.DataFrame:
    """Scores for BOTH look-ahead arms of every method — the view's K filter dropped.

    The escape hatch RQ-i needs. A K-specific view is exactly the wrong frame for the look-ahead
    question: ``L0`` sees only the K=0 arms and ``L5`` only the K=5 arms, so
    :func:`~eda_analysis.stats.paired_k_comparison` comes back empty in both and the contrast used
    to exist only in the pooled ``all`` view (retired 2026-07-27). This rebuilds ``scores_long``
    with ``ks=None`` and *everything else* — method/mode/label filters, judge + rep, persona
    attachment, derived MITI-proficiency rows — taken from the active config, so the K contrast can
    be computed and exported from inside a view.

    ``source`` is the :class:`Setup` returned by :func:`notebook_setup` (or a bare
    :class:`EdaConfig`). **Read-only w.r.t. routing**: the active judge and the export root are left
    exactly as :func:`notebook_setup` set them, so artifacts still land under
    ``results/<view>/`` — see :data:`RQ_I_VIEW` for which view should be the one to save them.

    Note an explicit ``cfg.arm_labels`` whitelist is still honoured, so a config that named a single
    arm returns a single arm (the caller asked for that); ``ks`` is the only filter dropped.
    """
    from . import discover_arms, load_scores_long, add_derived_mitiprof_rows
    from .data import filter_arms

    cfg = source.CFG if isinstance(source, Setup) else source
    arms = filter_arms(discover_arms(include_archived=cfg.include_archived),
                       methods=cfg.methods, ks=None, modes=cfg.modes, arm_labels=cfg.arm_labels)
    scores = load_scores_long(arms, attach_persona=cfg.attach_persona)
    if cfg.add_derived_mitiprof and not scores.empty:
        scores = add_derived_mitiprof_rows(scores, arms)
    return scores
