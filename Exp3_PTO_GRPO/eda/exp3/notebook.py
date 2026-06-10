"""
notebook.py — one-call setup that every analysis notebook shares.

Replaces the ~15-line boilerplate that used to be copy-pasted (byte-identical) into the
top of ``00``/``01``/``02``/``03``: apply the publication style, discover the arms on disk,
build the tidy ``scores_long`` backbone + the stable arm palette + the present-metrics list,
and print the same one-glance summary. Returns a small :class:`Setup` namespace so downstream
cells reference ``S.SCORES`` / ``S.PALETTE`` / ``S.METRICS`` etc.

Usage (notebook cell 1)::

    import sys, os; sys.path.insert(0, os.path.abspath("."))
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
    import exp3
    from exp3 import stats, behavior, training, pref, figures, plots
    S = exp3.notebook_setup()
"""

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Setup:
    """The shared notebook context (built by :func:`notebook_setup`)."""
    ARMS: list
    SCORES: pd.DataFrame
    PALETTE: dict
    METRICS: List[str]
    ORACLE_NOISE: float
    RESULTS_DIR: str


def notebook_setup(*, oracle_noise: float = 0.10, attach_persona: bool = True,
                   verbose: bool = True) -> Setup:
    """Discover arms, build ``scores_long`` + palette + metrics, and return a :class:`Setup`.

    ``oracle_noise`` is the oracle reproducibility band (~0.07-0.10 |Δ| from the partial-conv
    EDA): differences smaller than this are at oracle-noise scale. ``attach_persona`` recovers
    the true per-conversation persona (the matched-persona design). Set ``verbose=False`` to
    suppress the summary print.
    """
    # Lazy sibling imports (keep this module load-order-safe under the __init__ re-exports).
    from . import (discover_arms, load_scores_long, QUESTIONNAIRE_ORDER,
                   RESULTS_DIR, figures)
    figures.set_style()
    arms = discover_arms()
    scores = load_scores_long(arms, attach_persona=attach_persona)
    if scores.empty:
        palette, metrics = {}, []
    else:
        palette = figures.arm_palette(sorted(scores.arm.unique()))
        metrics = [m for m in QUESTIONNAIRE_ORDER if m in set(scores.questionnaire.unique())]
    if verbose:
        print("arms on disk:", [(a.label, len(a.iters)) for a in arms])
        if scores.empty:
            print("scores_long: EMPTY — no eval scores found on disk yet.")
        else:
            print("scores_long:", scores.shape, "| arms scored:", sorted(scores.arm.unique()))
        print("exports ->", RESULTS_DIR)
    return Setup(ARMS=arms, SCORES=scores, PALETTE=palette, METRICS=metrics,
                 ORACLE_NOISE=oracle_noise, RESULTS_DIR=RESULTS_DIR)
