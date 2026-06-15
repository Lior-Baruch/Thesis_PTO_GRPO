"""
notebook.py — one-call setup that every analysis notebook shares.

Cell 1 of each notebook builds an :class:`eda_analysis.EdaConfig` from flat globals and passes it here.
``notebook_setup`` applies the publication style + plot scales, filters + discovers the arms,
builds the tidy ``scores_long`` backbone (optionally with the free derived MITI-proficiency
ratios), the stable arm palette, and the present-metric list, sets the export group, writes a
provenance banner, and returns a small :class:`Setup` namespace so downstream cells reference
``S.SCORES`` / ``S.PALETTE`` / ``S.CFG`` etc.

Usage (notebook cell 1)::

    import sys, os; sys.path.insert(0, os.path.abspath("."))
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
    import eda_analysis
    from eda_analysis import stats, behavior, training, pref, figures, plots
    cfg = eda_analysis.EdaConfig(export_group="eval")    # flat globals -> one config
    S = eda_analysis.notebook_setup(cfg)

``notebook_setup()`` with no args still works (default config = all arms, all present metrics).
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import EdaConfig


@dataclass
class Setup:
    """The shared notebook context (built by :func:`notebook_setup`)."""
    ARMS: list
    SCORES: pd.DataFrame
    PALETTE: dict
    METRICS: List[str]
    ORACLE_NOISE: float
    RESULTS_DIR: str
    CFG: EdaConfig


def notebook_setup(cfg: Optional[EdaConfig] = None, **overrides) -> Setup:
    """Discover+filter arms, build ``scores_long`` + palette + metrics, return a :class:`Setup`.

    ``cfg`` is an :class:`EdaConfig` (default = all arms / all present metrics). ``**overrides``
    patch individual fields for a quick tweak, e.g. ``notebook_setup(cfg, selection="best")``.
    """
    from . import (discover_arms, load_scores_long, add_derived_mitiprof_rows,
                   QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, RESULTS_DIR, figures, exports)
    from .discovery import filter_arms

    cfg = cfg or EdaConfig()
    if overrides:
        cfg = cfg.with_(**overrides)

    figures.set_style(cfg)
    exports.set_export_group(cfg.export_group if cfg.results_subdirs else "")
    exports.set_formats(cfg.fig_formats, cfg.table_formats)

    arms = discover_arms(include_archived=cfg.include_archived)
    arms = filter_arms(arms, methods=cfg.methods, ks=cfg.ks, modes=cfg.modes,
                       arm_labels=cfg.arm_labels)

    scores = load_scores_long(arms, attach_persona=cfg.attach_persona)
    if cfg.add_derived_mitiprof and not scores.empty:
        scores = add_derived_mitiprof_rows(scores, arms)

    if scores.empty:
        palette, metrics = {}, []
    else:
        palette = figures.arm_palette(sorted(scores.arm.unique()))
        present = set(scores.questionnaire.unique())
        if cfg.metrics:
            metrics = [m for m in cfg.metrics if m in present]
        else:
            base = WARMTH_RUBRICS if cfg.warmth_only else QUESTIONNAIRE_ORDER
            metrics = [m for m in base if m in present]

    # Provenance banner (printed + exported) so every regenerated figure set is traceable.
    if not scores.empty:
        exports.save_provenance(cfg, scores)

    if cfg.verbose:
        print("arms on disk:", [(a.label, len(a.iters)) for a in arms])
        if scores.empty:
            print("scores_long: EMPTY — no eval scores found on disk yet.")
        else:
            print("scores_long:", scores.shape, "| arms scored:", sorted(scores.arm.unique()))
            print("metrics:", metrics, "| selection:", cfg.selection)
        grp = cfg.export_group or "(flat)"
        print(f"exports -> {RESULTS_DIR}  [group: {grp}]")

    return Setup(ARMS=arms, SCORES=scores, PALETTE=palette, METRICS=metrics,
                 ORACLE_NOISE=cfg.oracle_noise, RESULTS_DIR=RESULTS_DIR, CFG=cfg)
