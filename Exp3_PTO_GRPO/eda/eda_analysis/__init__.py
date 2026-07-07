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
import re
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
# The "MICI" questionnaire aggregate + every per-item MICI detail column (severity, per-turn rates)
# are higher = worse, as is patient sustain-talk. Display layer only (drives the trailing ' ↓').
LOWER_IS_BETTER = {
    "MICI", "MICI_Severity", "MICI_Rate", "MICI_Confront_rate", "MICI_AdviseNoPermission_rate",
    "MICI_Warn_rate", "MICI_Direct_rate", "MICI_Judge_rate", "MICI_OverPraise_rate",
    "PCT_SustainTalk_prop",
}


# ── Human-readable display names (LABEL LAYER ONLY) ──────────────────────────────
# Applied when drawing figures / writing tables so a supervisor never has to decode `B3_Q` or
# `%MICO`. These NEVER rename the underlying `questionnaire` / `arm` / column KEYS (those are used
# as join+filter keys throughout the package). Any code not in the map falls through unchanged.
DISPLAY_NAMES = {
    # LABEL CONVENTION (keeps figures consistent — enforce this when adding a metric):
    #   • Every oracle-MITI-coded metric carries a trailing "(MITI)" so the source instrument is
    #     unambiguous and identical across panels (fixes the old behavior_drift asymmetry where only
    #     Questions/Empathy were tagged while Reflections/Affirmations/Persuasion read bare).
    #   • Deterministic (non-LLM) text metrics carry their OWN source tag instead ("regex ?",
    #     "Degeneration %") — never "(MITI)".
    #   • Standalone questionnaires keep their validated-instrument acronym up-front + a gloss.
    # Display layer only — these NEVER rename the underlying data keys.
    #
    # Warmth / satisfaction / alliance rubrics. Q1Q2/Q1/Q2 stay as their plain codes
    # (Lior: no "Satisfaction …" prefix) — Q1/Q2 simply fall through display_label unchanged.
    "Q1Q2": "Q1+Q2",
    # Original validated-instrument acronym KEPT up-front (Lior), descriptive gloss in parens.
    "WAI-SR": "WAI-SR (Working Alliance)", "CSQ-8": "CSQ-8 (Client Satisfaction)",
    "MI-SAT": "MI-SAT (MI Satisfaction)", "MITI": "MITI (MI Integrity)",
    # Standalone orthogonal questionnaires (their own instruments, NOT MITI-derived).
    "PCT": "PCT (Patient Change-Talk)", "MICI": "MICI (MI-Inconsistency)",
    # Derived MITI-proficiency ratios (computed FROM the MITI behavior counts → tagged "(MITI)").
    "R:Q": "Reflection:Question (MITI)", "%CR": "% Complex Reflections (MITI)",
    "%MICO": "% MI-Consistent (MITI)",
    # MITI behavior counts (per conversation). "Questions" is a per-conv COUNT of question-FUNCTION
    # utterances (oracle) — kept distinct from the regex "? / turn" RATE below to avoid misreading a
    # count against a rate (they are different constructs: function vs literal-? syntax).
    "B3_Q": "Questions / conv (MITI)", "B6_AF": "Affirmations (MITI)", "B4_SR": "Simple Reflections (MITI)",
    "B5_CR": "Complex Reflections (MITI)", "B2_Persuade": "Persuasion (MITI)", "B1_GI": "Giving Information (MITI)",
    "B7_Seek": "Seeking Collaboration (MITI)", "RtoQ": "Reflection:Question (MITI)", "Empathy": "Empathy (MITI)",
    # Per-therapist-turn rate versions of the MITI counts (length-normalized; the drift figure plots these).
    "B3_Q_per_turn": "Questions / turn (MITI)", "B6_AF_per_turn": "Affirmations / turn (MITI)",
    "B4_SR_per_turn": "Simple Reflections / turn (MITI)", "B5_CR_per_turn": "Complex Reflections / turn (MITI)",
    "B2_Persuade_per_turn": "Persuasion / turn (MITI)",
    "B1_GI_per_turn": "Giving Information / turn (MITI)", "B7_Seek_per_turn": "Seeking Collaboration / turn (MITI)",
    # MICI (MI-INCONSISTENT) detail — severity global (1-5) + per-therapist-turn behavior rates.
    # Every MICI column is higher = worse (↓-flagged via LOWER_IS_BETTER).
    "MICI_Severity": "MI-Incon. Severity (MICI)", "MICI_Rate": "MI-Incon. total / turn (MICI)",
    "MICI_Confront_rate": "Confront / turn (MICI)", "MICI_AdviseNoPermission_rate": "Advise w/o permission / turn (MICI)",
    "MICI_Warn_rate": "Warn / turn (MICI)", "MICI_Direct_rate": "Direct/order / turn (MICI)",
    "MICI_Judge_rate": "Judge/label / turn (MICI)", "MICI_OverPraise_rate": "Over-praise / turn (MICI)",
    # PCT (PATIENT change-talk) detail — patient-perspective globals (1-5) + utterance proportions.
    "PCT_Importance": "Importance (PCT)", "PCT_Confidence": "Confidence (PCT)",
    "PCT_Readiness": "Readiness (PCT)", "PCT_GlobalMean": "PCT global mean",
    "PCT_ChangeProp": "Change-Talk proportion (PCT)", "PCT_ChangeTalk_prop": "% Change Talk (PCT)",
    "PCT_SustainTalk_prop": "% Sustain Talk (PCT)", "PCT_Neutral_prop": "% Neutral (PCT)",
    # Deterministic text metrics — NOT MITI; each carries its own source tag.
    "q_per_turn": "Questions / turn (regex ?)", "q_per_turn_miti": "Questions / turn (MITI)",
    "mean_turn_len": "Turn length (chars)", "loop": "Degeneration %",
    "conv_len": "Conversation length", "n_th_turns": "Therapist turns",
}

# Readable arm labels: canonical key -> "<method> (K=<k>)".
ARM_LABELS = {"PTO_LA0": "PTO (K=0)", "PTO_LA5": "PTO (K=5)",
              "GRPO_LA0": "GRPO (K=0)", "GRPO_LA5": "GRPO (K=5)", "Base": "Base"}
_ARM_RE = re.compile(r"^(PTO|GRPO)_LA(\d+)$")


def display_label(metric: str) -> str:
    """Readable label for a metric / behavior code, flagging lower-is-better with a trailing '↓'.

    Consults :data:`DISPLAY_NAMES` (falls through to the raw code if absent), then appends ' ↓' for
    :data:`LOWER_IS_BETTER` metrics so e.g. ``MICI`` reads ``MI-Inconsistency ↓`` and is never
    mistaken for a higher-is-better rubric. Label layer only — never used as a data key.
    """
    name = DISPLAY_NAMES.get(metric, metric)
    return f"{name} ↓" if metric in LOWER_IS_BETTER else name


_SHORT_LABEL = {"Q1Q2": "Q1+Q2"}   # the rest of the keys already ARE their acronym (WAI-SR, CSQ-8, R:Q…)


def short_label(metric: str) -> str:
    """Compact acronym-only label for DENSE figures (correlation matrices, packed axes).

    The full :func:`display_label` is ``"ACRONYM (descriptive gloss)"`` — great for panel titles and
    tables, but it overflows a 10×10 heatmap tick. This returns just the acronym (the metric key, which
    already is the instrument acronym; ``Q1Q2→"Q1+Q2"``), still ↓-flagged for lower-is-better. The
    descriptive gloss lives in the surrounding caption/legend instead.
    """
    base = _SHORT_LABEL.get(metric, metric)
    return f"{base} ↓" if metric in LOWER_IS_BETTER else base


def arm_label(arm: str) -> str:
    """Readable arm label: ``"PTO_LA0"`` -> ``"PTO (K=0)"`` (auto-parses any ``LA<k>``).

    Unknown labels pass through unchanged. Label layer only — the canonical ``arm`` key is what
    every figure hues/filters on, so only the *displayed* text is swapped.
    """
    if arm in ARM_LABELS:
        return ARM_LABELS[arm]
    m = _ARM_RE.match(arm or "")
    return f"{m.group(1)} (K={m.group(2)})" if m else arm


# Patient-characteristic columns recovered per persona.
PERSONA_COLS = ["gender", "age_value", "problem", "problem_time",
                "tried_to_solve", "cooperation_level"]


# ── Public API re-exports (submodules import `from . import WORKSPACE_ROOT, ...`) ──
# Control surface (EdaConfig + the one-call notebook_setup) — config.py absorbed the old notebook.py.
from .config import EdaConfig, notebook_setup, Setup  # noqa: E402
# Data layer (discovery + personas + scores + select all merged into data.py).
from .data import (  # noqa: E402
    Arm, discover_arms, parse_experiment_name, filter_arms,
    canonical_personas, persona_order, file_to_persona, attach_personas, validate_recovery,
    load_scores_long, load_subscales, to_wide, collapse_base, MEAN_COLS,
    add_derived_mitiprof_rows, select_scores,
    all_models, best_per_experiment,
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

# Submodules + backward-compat ALIASES so every submodule-qualified notebook call keeps resolving
# after the plumbing merge: figures/plots -> plotting; discovery/personas/scores/select -> data.
from . import plotting, data, stats, behavior, training, pref, exports  # noqa: E402,F401
figures = plots = plotting              # notebooks: figures.set_style / plots.overlay_trajectory
personas = scores = discovery = select = data   # notebooks: eda_analysis.personas.canonical_personas
# Register the aliases as importable submodules too, so `from eda_analysis.personas import X`
# (the form used in 1_Eval_and_Behavior) resolves — not only attribute access.
for _alias, _mod in (("figures", plotting), ("plots", plotting), ("personas", data),
                     ("scores", data), ("discovery", data), ("select", data)):
    sys.modules[f"{__name__}.{_alias}"] = _mod

__all__ = [
    "WORKSPACE_ROOT", "DATA_DIR", "QUESTIONNAIRES", "QUESTIONNAIRE_ORDER", "PERSONA_COLS",
    "WARMTH_RUBRICS", "ORTHOGONAL_METRICS", "LOWER_IS_BETTER", "display_label", "short_label",
    "DISPLAY_NAMES", "ARM_LABELS", "arm_label",
    "EdaConfig", "notebook_setup", "Setup",
    "Arm", "discover_arms", "parse_experiment_name", "filter_arms",
    "canonical_personas", "persona_order", "file_to_persona",
    "attach_personas", "validate_recovery",
    "load_scores_long", "load_subscales", "to_wide", "collapse_base", "MEAN_COLS",
    "add_derived_mitiprof_rows", "select_scores",
    "all_models", "best_per_experiment",
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
