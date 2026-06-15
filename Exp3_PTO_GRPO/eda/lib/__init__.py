"""
EDA library for ``Exp3_PTO_GRPO/eda/``.

Importing this package locates the **experiment root** (the Exp3 folder itself —
identified by ``HF_key.txt`` + ``openai_key.txt`` at top level) and prepends
both that root and its ``code/`` directory to ``sys.path`` so the per-experiment
helpers ``system_prompts_builder`` and ``questionnaires`` resolve regardless of
where the notebook was launched.

After import, ``WORKSPACE_ROOT`` is the absolute path of the experiment root,
and every name in the public API below is reachable as ``from lib import ...``.
"""

import os
import sys


_KEY_FILES = ("HF_key.txt", "openai_key.txt")


def _resolve_workspace_root(start: str, max_steps: int = 8):
    cur = os.path.abspath(start)
    for _ in range(max_steps):
        if all(os.path.exists(os.path.join(cur, kf)) for kf in _KEY_FILES):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


WORKSPACE_ROOT = _resolve_workspace_root(os.getcwd())
if WORKSPACE_ROOT is None:
    raise RuntimeError(
        f"Could not locate experiment root containing {_KEY_FILES} "
        f"by walking up from {os.getcwd()!r}"
    )

_EXPERIMENT_CODE = os.path.join(WORKSPACE_ROOT, "code")
for _p in (WORKSPACE_ROOT, _EXPERIMENT_CODE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


# ── Public API re-exports ───────────────────────────────────────────────
from .config import (  # noqa: E402
    # Constants
    FIG_WIDE, FIG_SINGLE, ALPHA, EFFECT_SIZE_THRESHOLDS,
    ORACLE_ORDER, GROUP_ORDER, DPO_GROUP_ORDER,
    EXPERIMENT_PALETTE, ORACLE_METRIC_MAP,
    ORACLE_TOKEN_ALIASES, COMPOSITE_METRICS,
    EVAL_MODEL, EVAL_TEMPERATURE, MAX_RETRIES, DEFAULT_CONCURRENCY,
    DATA_DIR, METHOD_DATA_DIR, EVAL_QUESTIONNAIRE_DIRS,
    eval_scores_root_for_method, eval_csv_dir,
    # Dataclasses + helpers
    EDAConfig, PlotContext, set_plot_style,
    # Experiment registry
    Experiment, EXPERIMENTS, get_data_paths, get_model_names,
    get_model_eval_layout, resolve_paths,
)

from .data import (  # noqa: E402
    # Conversations
    load_data, combine_data, reconstruct_conversation_text,
    add_patient_characteristics,
    # Model metadata + ordering
    parse_model_metadata, add_model_metadata_columns,
    compute_model_order, apply_model_order,
    build_experiment_palette,
    # Eval loading
    load_all_eval_results, merge_q1_q2_results, build_test_cases,
    # Selection / filtering
    select_best_models_by_own_oracle, filter_to_models, build_merged_metrics,
)

# NOTE: the legacy analysis/plots + cross-iteration training-reward EDA modules
# (lib/analysis.py, lib/iterations.py) were removed 2026-06-15 — they only served
# the now-deleted Exp2 archive notebooks. The current Exp3 analysis lives in the
# `exp3/` package; `lib/` survives ONLY to power Run_Eval.ipynb's oracle scoring.

# Eval pipeline (oracle scoring) — re-exported behind a flag because the
# questionnaires module needs to be reachable on sys.path. The path prepend
# above usually makes it work, but rare envs without it shouldn't break the
# whole package import.
from . import eval as _eval_mod  # noqa: E402
EVAL_CODE_AVAILABLE = _eval_mod.EVAL_CODE_AVAILABLE
if EVAL_CODE_AVAILABLE:
    from .eval import (  # noqa: E402
        call_openai_json,
        evaluate_conversation,
        build_default_eval_configs,
        run_all_evaluations_async,
    )


__all__ = [
    "WORKSPACE_ROOT",
    # config
    "FIG_WIDE", "FIG_SINGLE", "ALPHA", "EFFECT_SIZE_THRESHOLDS",
    "ORACLE_ORDER", "GROUP_ORDER", "DPO_GROUP_ORDER",
    "EXPERIMENT_PALETTE", "ORACLE_METRIC_MAP",
    "ORACLE_TOKEN_ALIASES", "COMPOSITE_METRICS",
    "EVAL_MODEL", "EVAL_TEMPERATURE", "MAX_RETRIES", "DEFAULT_CONCURRENCY",
    "DATA_DIR", "METHOD_DATA_DIR", "EVAL_QUESTIONNAIRE_DIRS",
    "eval_scores_root_for_method", "eval_csv_dir",
    "EDAConfig", "PlotContext", "set_plot_style",
    "Experiment", "EXPERIMENTS", "get_data_paths", "get_model_names",
    "get_model_eval_layout", "resolve_paths",
    # data
    "load_data", "combine_data", "reconstruct_conversation_text",
    "add_patient_characteristics",
    "parse_model_metadata", "add_model_metadata_columns",
    "compute_model_order", "apply_model_order", "build_experiment_palette",
    "load_all_eval_results", "merge_q1_q2_results", "build_test_cases",
    "select_best_models_by_own_oracle", "filter_to_models", "build_merged_metrics",
    # eval (gated)
    "EVAL_CODE_AVAILABLE",
]
if EVAL_CODE_AVAILABLE:
    __all__ += [
        "call_openai_json", "evaluate_conversation",
        "build_default_eval_configs", "run_all_evaluations_async",
    ]
