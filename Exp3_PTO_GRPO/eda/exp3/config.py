"""
config.py — the single EDA control surface (`EdaConfig`).

Every analysis notebook's cell 1 is now flat globals bundled into one ``EdaConfig`` that is
passed to :func:`exp3.notebook_setup`. This mirrors the trainer notebooks' "cell 1 = flat
globals" pattern: one place to choose arms, metrics, selection mode, plot scales, and where
artifacts are saved — reproducible and git-diffable (the run's config is in the file, not in
hand-edits scattered across cells).

All fields have safe defaults, so ``EdaConfig()`` reproduces the pre-refactor behaviour
(all arms, all present metrics, all-models selection, the old plot style, flat-ish exports).

Usage (notebook cell 1)::

    import exp3
    cfg = exp3.EdaConfig(
        methods=["PTO"], ks=[0],            # arm filter (None = all)
        selection="best",                   # cross-model default view
        metrics=None, add_derived_mitiprof=True,
        panel=(6.0, 4.0), ncols=2, score_ylim=(1, 5), share_y=True,
        export_group="eval",                # results/figures/eval/ ...
    )
    S = exp3.notebook_setup(cfg)
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class EdaConfig:
    """All user-facing EDA knobs in one object (see module docstring)."""

    # ── Arm selection (None = no filter on that axis) ────────────────────────
    methods: Optional[Sequence[str]] = None        # e.g. ["PTO"] | ["PTO","GRPO"]
    ks: Optional[Sequence[int]] = None             # e.g. [0] | [0, 5]
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
    export_group: str = ""                         # results/<figures|tables>/<group>/ ; "" = flat
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
