"""
eda_analysis — brand-new EDA package for Exp3_PTO_GRPO (PTO_Exp3 vs GRPO_Exp3).

Designed from the data + the thesis's research questions, NOT ported from the
legacy Exp1/Exp2 EDA library. The analysis layer (this package's top level) is
read-only + disk-discovery-driven; the oracle-scoring layer lives in the
:mod:`eda_analysis.scoring` subpackage (imported explicitly by ``Run_Eval.ipynb``
and ``Judge_Reliability.ipynb`` — NOT re-exported here, because building its
registry scans the data dirs, which the analysis notebooks never need).

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

import sys

# ── Layer-0 core (paths, metric registries, label helpers) lives in the LEAF module
# ``constants.py`` (stdlib-only, imports nothing from the package), so submodules import it
# directly (``from .constants import ...``) with no circular-import / ordering risk. Re-exported
# here so the public surface (``eda_analysis.QUESTIONNAIRES`` etc.) is unchanged.
from .constants import (  # noqa: E402,F401
    WORKSPACE_ROOT, DATA_DIR,
    QUESTIONNAIRES, QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, ORTHOGONAL_METRICS, LOWER_IS_BETTER,
    MITI_THRESHOLDS, Q2_ITEM_SHORT, Q2_ITEM_GROUPS,
    DISPLAY_NAMES, ARM_LABELS, PERSONA_COLS,
    display_label, short_label, arm_label,
)


# ── Public API re-exports ──────────────────────────────────────────────────────
# Control surface (EdaConfig + the one-call notebook_setup) — config.py absorbed the old notebook.py.
from .config import EdaConfig, notebook_setup, Setup  # noqa: E402
# Data layer (discovery + personas + scores + select all merged into data.py).
from .data import (  # noqa: E402
    Arm, discover_arms, parse_experiment_name, filter_arms,
    canonical_personas, persona_order, attach_personas,
    load_scores_long, load_subscales, load_q2_items, to_wide, collapse_base,
    add_derived_mitiprof_rows,
    all_models, best_per_experiment,
    load_cached, set_cache, cache_enabled, reset_cache,
)
from .exports import (  # noqa: E402
    save_fig, save_table, save_provenance, build_index, reset_results,
    set_export_group, set_view, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
)
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

# Submodules + backward-compat aliases. ``figures``/``plots`` -> ``plotting`` are KEPT (heavily used
# across the notebooks and inside plotting itself). The data-module aliases from the 14->9 merge
# (personas/scores/discovery/select -> data) are RETIRED — their only live call sites now use the
# canonical top-level exports (e.g. ``from eda_analysis import persona_order`` /
# ``eda_analysis.data.best_per_experiment``).
from . import plotting, data, stats, behavior, training, pref, exports  # noqa: E402,F401
figures = plots = plotting              # notebooks: figures.set_style / plots.trajectory_grid
# Register the plotting aliases as importable submodules too, so ``from eda_analysis.figures import X``
# resolves — not only attribute access.
for _alias, _mod in (("figures", plotting), ("plots", plotting)):
    sys.modules[f"{__name__}.{_alias}"] = _mod

__all__ = [
    "WORKSPACE_ROOT", "DATA_DIR", "QUESTIONNAIRES", "QUESTIONNAIRE_ORDER", "PERSONA_COLS",
    "WARMTH_RUBRICS", "ORTHOGONAL_METRICS", "LOWER_IS_BETTER", "display_label", "short_label",
    "MITI_THRESHOLDS", "Q2_ITEM_SHORT", "Q2_ITEM_GROUPS",
    "DISPLAY_NAMES", "ARM_LABELS", "arm_label",
    "EdaConfig", "notebook_setup", "Setup",
    "Arm", "discover_arms", "parse_experiment_name", "filter_arms",
    "canonical_personas", "persona_order", "attach_personas",
    "load_scores_long", "load_subscales", "load_q2_items", "to_wide", "collapse_base",
    "add_derived_mitiprof_rows",
    "all_models", "best_per_experiment",
    "load_cached", "set_cache", "cache_enabled", "reset_cache",
    "save_fig", "save_table", "save_provenance", "build_index", "reset_results",
    "set_export_group", "set_view", "RESULTS_DIR", "FIGURES_DIR", "TABLES_DIR",
    "paired_method_comparison", "paired_k_comparison",
    "rank_agreement_by_nturns", "filter_thin_arms", "thin_arms",
    "advantage_signal_by_iter", "reward_distribution_frame",
    "load_branch_reliability", "tb_curves", "parse_run_tb",
    "pref_word_ranking", "pref_word_drift_heatmap", "plot_category_drift", "top_words_by_iter",
    "preference_direction_drift", "plot_direction_drift", "learn_unlearn_words", "plot_learn_unlearn",
    "plotting", "data", "figures", "plots", "stats", "behavior", "training", "pref",
]
