"""
config.py — the single EDA control surface: ``EdaConfig`` + ``notebook_setup`` (the "cell 1" kernel).

Every analysis notebook's cell 1 is flat globals bundled into one ``EdaConfig`` that is passed to
:func:`notebook_setup`. One place to choose the **VIEW** (which look-ahead arms + which results
subfolder), the metrics, the selection mode, plot scales, and where artifacts are saved —
reproducible and git-diffable (the run's config is in the file, not in scattered cell hand-edits).

**The VIEW knob (new).** ``view`` is the one control that matters day-to-day. It sets BOTH:
  - the arm filter — ``"all"`` = every arm, ``"L0"`` = K=0 arms (PTO_LA0/GRPO_LA0),
    ``"L2"`` = K=2 arms (PTO_LA2/GRPO_LA2), ``"L5"`` = K=5 arms (PTO_LA5/GRPO_LA5); and
  - the results root — artifacts land under ``results/<view>/figures|tables/<group>/``.
So ``results/`` ends up with four parallel trees (``all/``, ``L0/``, ``L2/``, ``L5/``). An explicit
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

# VIEW -> ks arm filter. ``all`` = no K filter; ``L0`` = K=0 only; ``L2`` = K=2 only; ``L5`` = K=5 only.
_VIEW_KS: Dict[str, Optional[List[int]]] = {"all": None, "L0": [0], "L2": [2], "L5": [5]}
# Case-insensitive input -> canonical view name (so "l0"/"L0" both work; folder stays "L0").
_VIEW_ALIASES: Dict[str, str] = {"all": "all", "l0": "L0", "l2": "L2", "l5": "L5"}


@dataclass
class EdaConfig:
    """All user-facing EDA knobs in one object (see module docstring)."""

    # ── THE knob: view = which arms + which results subfolder ─────────────────
    view: str = "all"                              # "all" | "L0" | "L5" (arm filter + results/<view>/)

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


def notebook_setup(cfg: Optional[EdaConfig] = None, **overrides) -> Setup:
    """Discover+filter arms (by the VIEW), build ``scores_long`` + palette + metrics, set the
    view-aware export root, write a provenance banner, and return a :class:`Setup`.

    ``cfg`` is an :class:`EdaConfig` (default = the ``all`` view / all present metrics).
    ``**overrides`` patch individual fields for a quick tweak, e.g.
    ``notebook_setup(cfg, view="L0")`` or ``notebook_setup(cfg, selection="best")``.
    """
    from . import (discover_arms, load_scores_long, add_derived_mitiprof_rows,
                   QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, plotting, exports)
    from .data import filter_arms

    cfg = cfg or EdaConfig()
    if overrides:
        cfg = cfg.with_(**overrides)

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

    plotting.set_style(cfg)
    exports.set_view(view)                                                   # results/<view>/...
    exports.set_export_group(cfg.export_group if cfg.results_subdirs else "")
    exports.set_formats(cfg.fig_formats, cfg.table_formats)

    arms = discover_arms(include_archived=cfg.include_archived)
    arms = filter_arms(arms, methods=cfg.methods, ks=effective_ks, modes=cfg.modes,
                       arm_labels=cfg.arm_labels)

    scores = load_scores_long(arms, attach_persona=cfg.attach_persona)
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

    results_dir = os.path.join(exports.RESULTS_DIR, view)

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
