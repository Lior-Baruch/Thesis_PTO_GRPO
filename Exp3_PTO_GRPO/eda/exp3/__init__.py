"""
exp3 — brand-new EDA package for Exp3_PTO_GRPO (PTO_Exp3 vs GRPO_Exp3).

Designed from the data + the thesis's research questions, NOT ported from the
Exp2-era ``lib/`` (which stays only for ``Run_Eval.ipynb`` scoring + the archived
notebooks). Read-only, disk-discovery-driven.

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

Public API is re-exported at the bottom so notebooks can ``from exp3 import ...``.
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
        f"exp3: could not locate experiment root containing {_KEY_FILES} by "
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
}
# Left-to-right plot order for the 6 headline rubrics (+ Q1/Q2 components).
QUESTIONNAIRE_ORDER = ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "Q1", "Q2"]

# Patient-characteristic columns recovered per persona.
PERSONA_COLS = ["gender", "age_value", "problem", "problem_time",
                "tried_to_solve", "cooperation_level"]


# ── Public API re-exports (submodules import `from . import WORKSPACE_ROOT, ...`) ──
from .discovery import Arm, discover_arms, parse_experiment_name  # noqa: E402
from .personas import (  # noqa: E402
    canonical_personas, persona_order, file_to_persona,
    attach_personas, validate_recovery,
)
from .scores import load_scores_long, load_subscales, to_wide, MEAN_COLS  # noqa: E402
from .select import all_models, best_per_experiment  # noqa: E402
from .exports import save_fig, save_table, RESULTS_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402

# One-call notebook setup + the new cross-method / training-internal / plotting helpers.
from .notebook import notebook_setup, Setup  # noqa: E402
from .stats import paired_method_comparison, paired_k_comparison  # noqa: E402
from .training import advantage_signal_by_iter, reward_distribution_frame  # noqa: E402
from .pref import pref_word_ranking  # noqa: E402
from . import figures, plots, stats, behavior, training, pref  # noqa: E402,F401

__all__ = [
    "WORKSPACE_ROOT", "DATA_DIR", "QUESTIONNAIRES", "QUESTIONNAIRE_ORDER", "PERSONA_COLS",
    "Arm", "discover_arms", "parse_experiment_name",
    "canonical_personas", "persona_order", "file_to_persona",
    "attach_personas", "validate_recovery",
    "load_scores_long", "load_subscales", "to_wide", "MEAN_COLS",
    "all_models", "best_per_experiment",
    "save_fig", "save_table", "RESULTS_DIR", "FIGURES_DIR", "TABLES_DIR",
    "notebook_setup", "Setup",
    "paired_method_comparison", "paired_k_comparison",
    "advantage_signal_by_iter", "reward_distribution_frame", "pref_word_ranking",
    "figures", "plots", "stats", "behavior", "training", "pref",
]
