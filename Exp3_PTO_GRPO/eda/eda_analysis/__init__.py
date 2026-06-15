"""
eda_analysis — brand-new EDA package for Exp3_PTO_GRPO (PTO_Exp3 vs GRPO_Exp3).

Designed from the data + the thesis's research questions, NOT ported from the
legacy ``oracle_scoring/`` package (which stays only for ``Run_Eval.ipynb`` oracle
scoring). Read-only, disk-discovery-driven.

Why this package exists / what it gets right that the old EDA didn't:
- **Persona recovery.** Each ``model_iter_k`` is a *seeded reshuffle* of the same
  96 patient personas (trainer: ``random.Random(cfg.seed + iteration)`` →
  ``model_iter_{iteration-1}``; final pass ``seed + num_iterations + 1`` →
  ``model_iter_{N}``; uniform formula ``seed + k + 1``). Conversations are saved
  under their *shuffled position* (``convs.py`` ``conversation_{permutation_index}.csv``),
  so ``conversation_{i}.csv`` is a DIFFERENT persona each iteration. The old EDA's
  ``add_patient_characteristics(patient_id=file_index)`` therefore joined the wrong
  persona for Exp3 runs. ``personas.py`` reconstructs the true map by replaying the
  shuffle — which also unlocks a matched-persona repeated-measures design.

Public API is re-exported at the bottom so notebooks can ``from eda_analysis import ...``.
"""

import os
import sys

# ── Resolve the experiment root (the Exp3 folder: HF_key.txt + openai_key.txt) ──
_KEY_FILES = ("HF_key.txt", "openai_key.txt")


def _resolve_workspace_root(*starts, max_steps: int = 10):
    for start in starts:
        cur = os.path.abspath(start)
        for _ in range(max_steps):
            if all(os.path.exists(os.path.join(cur, kf)) for kf in _KEY_FILES):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return None


WORKSPACE_ROOT = _resolve_workspace_root(os.path.dirname(__file__), os.getcwd())
if WORKSPACE_ROOT is None:
    raise RuntimeError(
        f"eda_analysis: could not locate experiment root containing {_KEY_FILES} by "
        f"walking up from {os.path.dirname(__file__)!r} or {os.getcwd()!r}"
    )

# Make the per-experiment helpers importable (system_prompts_builder, questionnaires).
_CODE_DIR = os.path.join(WORKSPACE_ROOT, "code")
for _p in (WORKSPACE_ROOT, _CODE_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")

# Display name -> (eval_scores metric subdir, the per-conv mean column).
QUESTIONNAIRES = {
    "Q1":     ("Q1",     "Q1_Mean"),
    "Q2":     ("Q2",     "Q2_Mean"),
    "Q1Q2":   (None,     "Q1Q2_Mean"),   # composite: mean(Q1_Mean, Q2_Mean)
    "WAI-SR": ("WAI_SR", "WAI_TotalMean"),
    "CSQ-8":  ("CSQ8",   "CSQ8_Mean"),
    "MI-SAT": ("MI_SAT", "MI_Mean"),
    "MITI":   ("MITI",   "MITI_GlobalMean"),
    # Orthogonal axes (added 2026-06-14 to break the PC1≈91% warmth halo):
    "PCT":    ("PCT",    "PCT_ChangeProp"),   # patient change-talk proportion CT/(CT+ST); higher = better
    "MICI":   ("MICI",   "MICI_Rate"),        # MI-inconsistent behaviors per therapist turn; LOWER = better
}
# Left-to-right plot order for the warmth rubrics (+ Q1/Q2 components) then the orthogonal axes.
QUESTIONNAIRE_ORDER = ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI", "Q1", "Q2"]

# The 5 warmth/satisfaction rubrics that share the dominant PC1 factor (the redundancy set).
WARMTH_RUBRICS = ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI"]
# Orthogonal axes intended to load OFF PC1 (incl. the free derived MITI-proficiency ratios).
ORTHOGONAL_METRICS = ["PCT", "MICI", "R:Q", "%CR", "%MICO"]
# Metrics where a LOWER value is better (must not be pooled into warmth composites / collapse_base).
LOWER_IS_BETTER = {"MICI"}


def display_label(metric: str) -> str:
    """Human label for a metric, flagging lower-is-better with a trailing '↓'.

    Used by leaderboards / forest plots so e.g. ``MICI`` reads ``MICI ↓`` and is never mistaken
    for a higher-is-better rubric.
    """
    return f"{metric} ↓" if metric in LOWER_IS_BETTER else metric


# Patient-characteristic columns recovered per persona.
PERSONA_COLS = ["gender", "age_value", "problem", "problem_time",
                "tried_to_solve", "cooperation_level"]


# ── Public API re-exports (submodules import `from . import WORKSPACE_ROOT, ...`) ──
from .config import EdaConfig  # noqa: E402
from .discovery import Arm, discover_arms, parse_experiment_name, filter_arms  # noqa: E402
from .personas import (  # noqa: E402
    canonical_personas, persona_order, file_to_persona,
    attach_personas, validate_recovery,
)
from .scores import (  # noqa: E402
    load_scores_long, load_subscales, to_wide, collapse_base, MEAN_COLS,
    add_derived_mitiprof_rows, select_scores,
)
from .select import all_models, best_per_experiment  # noqa: E402
from .exports import (  # noqa: E402
    save_fig, save_table, save_provenance, build_index, reset_results, set_export_group,
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
)

# One-call notebook setup + the new cross-method / training-internal / plotting helpers.
from .notebook import notebook_setup, Setup  # noqa: E402
from .stats import (  # noqa: E402
    paired_method_comparison, paired_k_comparison,
    rank_agreement_by_nturns, filter_thin_arms, thin_arms,
)
from .training import (  # noqa: E402
    advantage_signal_by_iter, reward_distribution_frame,
    load_branch_reliability, tb_curves, parse_run_tb,
)
from .pref import (  # noqa: E402
    pref_word_ranking, pref_word_drift_heatmap, plot_category_drift, top_words_by_iter,
    preference_direction_drift, plot_direction_drift, learn_unlearn_words, plot_learn_unlearn,
)
from . import figures, plots, stats, behavior, training, pref  # noqa: E402,F401

__all__ = [
    "WORKSPACE_ROOT", "DATA_DIR", "QUESTIONNAIRES", "QUESTIONNAIRE_ORDER", "PERSONA_COLS",
    "WARMTH_RUBRICS", "ORTHOGONAL_METRICS", "LOWER_IS_BETTER", "display_label",
    "EdaConfig",
    "Arm", "discover_arms", "parse_experiment_name", "filter_arms",
    "canonical_personas", "persona_order", "file_to_persona",
    "attach_personas", "validate_recovery",
    "load_scores_long", "load_subscales", "to_wide", "collapse_base", "MEAN_COLS",
    "add_derived_mitiprof_rows", "select_scores",
    "all_models", "best_per_experiment",
    "save_fig", "save_table", "save_provenance", "build_index", "reset_results", "set_export_group",
    "RESULTS_DIR", "FIGURES_DIR", "TABLES_DIR",
    "notebook_setup", "Setup",
    "paired_method_comparison", "paired_k_comparison",
    "rank_agreement_by_nturns", "filter_thin_arms", "thin_arms",
    "advantage_signal_by_iter", "reward_distribution_frame",
    "load_branch_reliability", "tb_curves", "parse_run_tb",
    "pref_word_ranking", "pref_word_drift_heatmap", "plot_category_drift", "top_words_by_iter",
    "preference_direction_drift", "plot_direction_drift", "learn_unlearn_words", "plot_learn_unlearn",
    "figures", "plots", "stats", "behavior", "training", "pref",
]
