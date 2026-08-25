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
    QUESTIONNAIRES, QUESTIONNAIRE_ORDER, WARMTH_RUBRICS, EXTRA_METRICS, LOWER_IS_BETTER,
    MITI_THRESHOLDS, Q1_ITEM_SHORT, Q2_ITEM_SHORT, Q2_ITEM_GROUPS, ITEM_QUESTIONNAIRES,
    DISPLAY_NAMES, ARM_LABELS, PERSONA_COLS,
    display_label, short_label, arm_label, item_short_label,
    last_iterations, support_note,
)


# ── Public API re-exports ──────────────────────────────────────────────────────
# Control surface (EdaConfig + the one-call notebook_setup) — config.py absorbed the old notebook.py.
from .config import (EdaConfig, notebook_setup, Setup, cross_k_arms,  # noqa: E402
                     scores_by_judge, FAMILIES, PER_JUDGE_TOPS)
# Data layer (discovery + personas + scores + select all merged into data.py).
from .data import (  # noqa: E402
    Arm, discover_arms, parse_experiment_name, filter_arms,
    canonical_personas, persona_order, attach_personas,
    load_scores_long, load_subscales, load_items, load_q2_items, to_wide, collapse_base,
    add_derived_mitiprof_rows,
    all_models, best_per_experiment, final_per_experiment, best_iteration_by_arm,
    load_cached, set_cache, cache_enabled, reset_cache,
)
from .exports import (  # noqa: E402
    save_fig, save_table, save_numbers, save_provenance, build_index, reset_results,
    set_family, RESULTS_DIR,
)
from .stats import (  # noqa: E402
    paired_method_comparison, paired_k_comparison, paired_best_method_comparison,
    k_means_by_iter, rank_agreement_by_nturns, filter_thin_arms, thin_arms,
)
from .training import (  # noqa: E402
    advantage_signal_by_iter, reward_distribution_frame,
    load_branch_reliability, tb_curves, parse_run_tb,
)
from .pref import (  # noqa: E402
    pref_word_ranking, pref_word_drift_heatmap, plot_category_drift, top_words_by_iter,
    preference_direction_drift, plot_direction_drift, learn_unlearn_words, plot_learn_unlearn,
    # the method-agnostic update-weighted probe + the training-signal -> eval-move link
    load_weighted_candidates, sample_groups, embed_candidates, direction_by_iter,
    direction_by_arm, direction_quality, pooled_direction_quality, direction_cosine,
    pooled_direction_cosines, weighted_lexical_contrast,
    preference_features_by_iter, link_to_outcomes, outcome_correlations,
    plot_pref_outcome, plot_category_compare, plot_lexical_push,
    # loss-vs-data decomposition · generation-vs-selection · yield · exhibits
    reweight, weighting_decomposition, rule_reconstruction_check,
    pool_mean_by_iter, pair_yield_by_iter, pref_examples,
    plot_selection_vs_generation, plot_pair_yield,
)

# Submodules + backward-compat aliases. ``figures``/``plots`` -> ``plotting`` are KEPT (heavily used
# across the notebooks and inside plotting itself). The data-module aliases from the 14->9 merge
# (personas/scores/discovery/select -> data) are RETIRED — their only live call sites now use the
# canonical top-level exports (e.g. ``from eda_analysis import persona_order`` /
# ``eda_analysis.data.best_per_experiment``).
from .compute import (  # noqa: E402
    iteration_compute, compute_summary, step_multiplier, iso_compute_pairs,
    iso_compute_contrast, budget_sweep, score_by_compute,
)

# ── 2026-08-18 reorg: the paper generators promoted into the package (REORG_2026-08-18.md, Phase
# C1). Each module's ``__all__`` is re-exported here EXCEPT the names that more than one module
# defines — those stay module-qualified so ``eda_analysis.<name>`` never silently means one of
# several: ARMS, METHODS, GROUP_KEYS, TEXT_CHANNELS, COOP_LABEL, COOP_ORDER, SIGN_NOTE,
# CENSOR_NOTE, CAPTIONS (crossgen + replication), k_of / method_of (lookahead + replication,
# same body), k_summary (lookahead / faithfulness / crossgen — three DIFFERENT tables). Use
# ``eda_analysis.lookahead.k_summary`` etc. The modules themselves are exported below.
# compute axis, promoted paper generators (floor columns, cost ratios, sweeps x graders, iso-channels)
from .compute import (  # noqa: E402
    GAP_CUTOFF_S, clear_memo, CONTRASTS, K_CONTRASTS, METHOD_CONTRASTS, CHANNELS, SIGN_K,
    SIGN_M, sign_note, channel_direction, compute_by_iteration_with_floor,
    compute_by_arm_with_floor, cost_ratios, step_multiplier_table, budget_sweep_ci,
    add_k_convention, all_budget_sweeps, budget_sweep_top, budget_sweep_crossjudge,
    crossjudge_verdicts, iso_channels, iso_channels_selected, compute_numbers,
)
# RQ-i K contrast: paired K frames, levels, table 1, channels, DiD, method gap, endpoints
from .lookahead import (  # noqa: E402
    RUBRICS, FIVE_POINT, RATE_METRICS, FIG_CHANNELS, LOWER_BETTER, TEXT_JUDGE_LABEL, HOLM_NOTE,
    model_name, stars, favours, wide_by_persona, holm_within, paired_k_frames, k_levels,
    k_table1, channel_k_frames, did_by_iter, method_gap_by_iter, endpoint_contrasts,
    best_iteration, lookahead_numbers,
)
# RQ-i transfer to the held-out grader: cross-K pairs, sign ladder, retention by K
from .transfer import (  # noqa: E402
    DEFAULT_REFERENCE_KINDS, SCALE_FLOOR, RATE_FLOOR, to_reliability_long, cross_k_pairs,
    sign_ladder, retention_by_k, transfer_numbers,
)
# look-ahead TAIL audit + API-call accounting (generations.jsonl)
from .tails import (  # noqa: E402
    TailAudit, tail_audit_frames, tail_audit_by_iter, tail_cues_by_iter, tail_within_group,
    score_by_realized_turns, api_calls, api_ratio, eval_conv_stats, grpo_steps,
    stream_record_stats, tails_numbers, clear_tails_memo, parse_tail, end_reason, tail_features,
    SCOUT_EXPECTED, N_Q, N_EVAL_RUBRICS, GRPO_GROUPS_PER_STEP, GRPO_CANDS_PER_STEP,
)
# within-group reward dispersion by K (margin / SD / winner-z, tau sensitivity)
from .dispersion import (  # noqa: E402
    TAU_TRAINER, TAUS, M_BRANCHES, GRADER, group_frame, load_group_frame, shuffle_null,
    iid_expectation, dispersion_by_iter, dispersion_ratios, tau_sensitivity, dispersion_numbers,
)
# reward faithfulness (partial-conv rank agreement) by K, grader, cooperation
from .faithfulness import (  # noqa: E402
    METRIC, COARSE, CUTS, SERIES, PRIMARY_LABEL, UNIT_NOTE, GRADER_NOTE, CUT_NOTE, CAVEATS,
    judge_display, fmt_ci, eval_frame, AgreementBoot, delta_ci, FaithfulnessData,
    faithfulness_data, check_against_rank_agreement, faithfulness_curve, curve_wide,
    faithfulness_by_iter, by_iter_display, k_faithfulness_by_iter, k_by_iter_display,
    matched_policy, matched_policy_display, k_summary_display, by_cooperation,
    by_cooperation_display, proxy_levels, faithfulness_numbers,
)
# Exp1 (ICLR) re-scored under gpt-4o-mini: cross-generation replication
from .crossgen import (  # noqa: E402
    ICLR_TABLE1, ITERS, N_PERSONAS, EXP1_DIR, CROSSGEN_ROOT, GRADER_GPT4OMINI, GRADER_GPT35,
    exp1_manifest, la3_manifest, load_crossgen, load_exp1_gpt35, persona_alignment_check,
    table1_crosscheck, paired_models, unpaired_delta, k_contrast_table, vs_base_table,
    pooled_arm_contrast, levels, k_contrast, ordering_claims, grader_agreement, vs_base,
    la3_gpt35, la3_cost_estimate, crossgen_all, crossgen_numbers,
)
# session-shape + score-dispersion stability of the K contrast (+ selection table)
from .replication import (  # noqa: E402
    FOUR_ARMS, SHAPE_METRICS, SHAPE_UNITS, STAB_METRICS, SIGN, PAIR, CENSOR, ITER0,
    brown_forsythe, pitman_morgan, read_md_table, shape_text_metrics, session_shape_levels,
    session_shape_paired, length_endpoints, length_kcontrast, sd_by_iter, sd_tests, sd_tally,
    sd_summary, ceiling, selection_table, default_selection_dirs, replication_numbers,
)
# held-out instruments under K: WAI-SR subscales, PCT, Q2 items, heterogeneity
from .instruments import (  # noqa: E402
    ARM_ORDER, WAI_SUBSCALES, WAI_MEASURES, PCT_METRICS, PCT_LABEL, Q2_SELF_DISCLOSURE,
    Q2_EMOTIONAL, HETERO_METRICS, PAIR_NOTE, instrument_frames_by_judge, endpoints,
    matched_endpoints, wai_conversation_frame, wai_subscale_parity, wai_subscales,
    wai_kcontrast, wai_fig_data, pct_kcontrast, q2_items, hetero_kcontrast, hetero_ceiling,
    instruments_numbers,
)

from . import (plotting, data, stats, behavior, training, pref, exports, reliability,
               compute, lookahead, transfer, tails, dispersion, faithfulness, crossgen,
               replication, instruments)  # noqa: E402,F401
figures = plots = plotting              # notebooks: figures.set_style / plots.trajectory_grid
# Register the plotting aliases as importable submodules too, so ``from eda_analysis.figures import X``
# resolves — not only attribute access.
for _alias, _mod in (("figures", plotting), ("plots", plotting)):
    sys.modules[f"{__name__}.{_alias}"] = _mod

__all__ = [
    "WORKSPACE_ROOT", "DATA_DIR", "QUESTIONNAIRES", "QUESTIONNAIRE_ORDER", "PERSONA_COLS",
    "WARMTH_RUBRICS", "EXTRA_METRICS", "LOWER_IS_BETTER", "display_label", "short_label",
    "MITI_THRESHOLDS", "Q1_ITEM_SHORT", "Q2_ITEM_SHORT", "Q2_ITEM_GROUPS", "ITEM_QUESTIONNAIRES",
    "DISPLAY_NAMES", "ARM_LABELS", "arm_label", "item_short_label",
    "last_iterations", "support_note",
    "EdaConfig", "notebook_setup", "Setup", "cross_k_arms",
    "scores_by_judge", "FAMILIES", "PER_JUDGE_TOPS",
    "Arm", "discover_arms", "parse_experiment_name", "filter_arms",
    "canonical_personas", "persona_order", "attach_personas",
    "load_scores_long", "load_subscales", "load_items", "load_q2_items", "to_wide", "collapse_base",
    "add_derived_mitiprof_rows",
    "all_models", "best_per_experiment", "final_per_experiment", "best_iteration_by_arm",
    "load_cached", "set_cache", "cache_enabled", "reset_cache",
    "save_fig", "save_table", "save_numbers", "save_provenance", "build_index", "reset_results",
    "set_family", "RESULTS_DIR",
    "paired_method_comparison", "paired_k_comparison", "paired_best_method_comparison",
    "k_means_by_iter", "rank_agreement_by_nturns", "filter_thin_arms", "thin_arms",
    "advantage_signal_by_iter", "reward_distribution_frame",
    "load_branch_reliability", "tb_curves", "parse_run_tb",
    "pref_word_ranking", "pref_word_drift_heatmap", "plot_category_drift", "top_words_by_iter",
    "preference_direction_drift", "plot_direction_drift", "learn_unlearn_words", "plot_learn_unlearn",
    "load_weighted_candidates", "sample_groups", "embed_candidates", "direction_by_iter",
    "direction_by_arm", "direction_quality", "pooled_direction_quality", "direction_cosine",
    "pooled_direction_cosines", "weighted_lexical_contrast",
    "preference_features_by_iter", "link_to_outcomes", "outcome_correlations",
    "plot_pref_outcome", "plot_category_compare", "plot_lexical_push",
    "reweight", "weighting_decomposition", "rule_reconstruction_check",
    "pool_mean_by_iter", "pair_yield_by_iter", "pref_examples",
    "plot_selection_vs_generation", "plot_pair_yield",
    # compute axis (GPU-hours per iteration; iso-compute + budget-sweep contrasts)
    "iteration_compute", "compute_summary", "step_multiplier", "iso_compute_pairs",
    "iso_compute_contrast", "budget_sweep", "score_by_compute",
    # compute — compute axis, promoted paper generators (floor columns, cost ratios, sweeps x graders, iso-channels)
    "GAP_CUTOFF_S", "clear_memo", "CONTRASTS", "K_CONTRASTS", "METHOD_CONTRASTS", "CHANNELS",
    "SIGN_K", "SIGN_M", "sign_note", "channel_direction", "compute_by_iteration_with_floor",
    "compute_by_arm_with_floor", "cost_ratios", "step_multiplier_table", "budget_sweep_ci",
    "add_k_convention", "all_budget_sweeps", "budget_sweep_top", "budget_sweep_crossjudge",
    "crossjudge_verdicts", "iso_channels", "iso_channels_selected", "compute_numbers",
    # lookahead — RQ-i K contrast: paired K frames, levels, table 1, channels, DiD, method gap, endpoints
    "RUBRICS", "FIVE_POINT", "RATE_METRICS", "FIG_CHANNELS", "LOWER_BETTER", "TEXT_JUDGE_LABEL",
    "HOLM_NOTE", "model_name", "stars", "favours", "wide_by_persona", "holm_within",
    "paired_k_frames", "k_levels", "k_table1", "channel_k_frames", "did_by_iter",
    "method_gap_by_iter", "endpoint_contrasts", "best_iteration", "lookahead_numbers",
    # transfer — RQ-i transfer to the held-out grader: cross-K pairs, sign ladder, retention by K
    "DEFAULT_REFERENCE_KINDS", "SCALE_FLOOR", "RATE_FLOOR", "to_reliability_long",
    "cross_k_pairs", "sign_ladder", "retention_by_k", "transfer_numbers",
    # tails — look-ahead TAIL audit + API-call accounting (generations.jsonl)
    "TailAudit", "tail_audit_frames", "tail_audit_by_iter", "tail_cues_by_iter",
    "tail_within_group", "score_by_realized_turns", "api_calls", "api_ratio", "eval_conv_stats",
    "grpo_steps", "stream_record_stats", "tails_numbers", "clear_tails_memo", "parse_tail",
    "end_reason", "tail_features", "SCOUT_EXPECTED", "N_Q", "N_EVAL_RUBRICS",
    "GRPO_GROUPS_PER_STEP", "GRPO_CANDS_PER_STEP",
    # dispersion — within-group reward dispersion by K (margin / SD / winner-z, tau sensitivity)
    "TAU_TRAINER", "TAUS", "M_BRANCHES", "GRADER", "group_frame", "load_group_frame",
    "shuffle_null", "iid_expectation", "dispersion_by_iter", "dispersion_ratios",
    "tau_sensitivity", "dispersion_numbers",
    # faithfulness — reward faithfulness (partial-conv rank agreement) by K, grader, cooperation
    "METRIC", "COARSE", "CUTS", "SERIES", "PRIMARY_LABEL", "UNIT_NOTE", "GRADER_NOTE",
    "CUT_NOTE", "CAVEATS", "judge_display", "fmt_ci", "eval_frame", "AgreementBoot", "delta_ci",
    "FaithfulnessData", "faithfulness_data", "check_against_rank_agreement",
    "faithfulness_curve", "curve_wide", "faithfulness_by_iter", "by_iter_display",
    "k_faithfulness_by_iter", "k_by_iter_display", "matched_policy", "matched_policy_display",
    "k_summary_display", "by_cooperation", "by_cooperation_display", "proxy_levels",
    "faithfulness_numbers",
    # crossgen — Exp1 (ICLR) re-scored under gpt-4o-mini: cross-generation replication
    "ICLR_TABLE1", "ITERS", "N_PERSONAS", "EXP1_DIR", "CROSSGEN_ROOT", "GRADER_GPT4OMINI",
    "GRADER_GPT35", "exp1_manifest", "la3_manifest", "load_crossgen", "load_exp1_gpt35",
    "persona_alignment_check", "table1_crosscheck", "paired_models", "unpaired_delta",
    "k_contrast_table", "vs_base_table", "pooled_arm_contrast", "levels", "k_contrast",
    "ordering_claims", "grader_agreement", "vs_base", "la3_gpt35", "la3_cost_estimate",
    "crossgen_all", "crossgen_numbers",
    # replication — session-shape + score-dispersion stability of the K contrast (+ selection table)
    "FOUR_ARMS", "SHAPE_METRICS", "SHAPE_UNITS", "STAB_METRICS", "SIGN", "PAIR", "CENSOR",
    "ITER0", "brown_forsythe", "pitman_morgan", "read_md_table", "shape_text_metrics",
    "session_shape_levels", "session_shape_paired", "length_endpoints", "length_kcontrast",
    "sd_by_iter", "sd_tests", "sd_tally", "sd_summary", "ceiling", "selection_table",
    "default_selection_dirs", "replication_numbers",
    # instruments — held-out instruments under K: WAI-SR subscales, PCT, Q2 items, heterogeneity
    "ARM_ORDER", "WAI_SUBSCALES", "WAI_MEASURES", "PCT_METRICS", "PCT_LABEL",
    "Q2_SELF_DISCLOSURE", "Q2_EMOTIONAL", "HETERO_METRICS", "PAIR_NOTE",
    "instrument_frames_by_judge", "endpoints", "matched_endpoints", "wai_conversation_frame",
    "wai_subscale_parity", "wai_subscales", "wai_kcontrast", "wai_fig_data", "pct_kcontrast",
    "q2_items", "hetero_kcontrast", "hetero_ceiling", "instruments_numbers",
    "plotting", "data", "figures", "plots", "stats", "behavior", "training", "pref",
    "reliability", "compute",
    "lookahead", "transfer", "tails", "dispersion", "faithfulness", "crossgen", "replication",
    "instruments",
]
