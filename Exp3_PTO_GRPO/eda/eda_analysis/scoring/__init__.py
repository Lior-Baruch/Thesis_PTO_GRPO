"""
eda_analysis.scoring — the oracle-scoring layer: the pipeline behind ``Run_Eval.ipynb``
and ``Judge_Reliability.ipynb``.

Folded in from the legacy ``oracle_scoring/`` package (2026-07-13) so the EDA is ONE package.
Modules are named by purpose (the old ``config``/``data``/``eval`` names collided with the
analysis layer's and shadowed the ``eval`` builtin):

- :mod:`.registry`      — eval settings + the ``EXPERIMENTS`` registry (auto-generated at import
  from :func:`eda_analysis.data.discover_arms`) + the ``eval_scores/`` layout helpers.
- :mod:`.conversations` — conversation loading for scoring + model-name metadata parsing.
- :mod:`.pipeline`      — the async oracle pipeline (OpenAI JSON-schema calls, row builders,
  resume-safe batch runners).
- :mod:`.judge`         — measurement-validity re-scoring: oracle repeatability (ICC) +
  pluggable second judge (OpenAI/Anthropic) + contrast preservation.

This subpackage is NOT imported by ``eda_analysis/__init__`` on purpose: building the registry
scans the data dirs on disk, which the analysis notebooks (``1_Outcomes`` … ``6_Stats``) never
need. Import it explicitly (``from eda_analysis import scoring`` /
``from eda_analysis.scoring import ...``) — only the two scoring notebooks do.

Importing it (via the package's ``constants`` leaf) resolves ``WORKSPACE_ROOT`` and prepends
``code/`` to ``sys.path`` so ``system_prompts_builder`` and ``questionnaires`` import from their
single canonical copies.
"""

from ..constants import WORKSPACE_ROOT  # noqa: F401  (re-export; also triggers the sys.path bootstrap)

# ── Public API re-exports ───────────────────────────────────────────────
from .registry import (  # noqa: E402
    # Constants
    ORACLE_TOKEN_ALIASES, COMPOSITE_METRICS,
    EVAL_MODEL, EVAL_TEMPERATURE, MAX_RETRIES, DEFAULT_CONCURRENCY,
    DATA_DIR, METHOD_DATA_DIR, EVAL_QUESTIONNAIRE_DIRS,
    eval_scores_root_for_method, eval_csv_dir,
    # Dataclasses + helpers
    ScoringConfig,
    # Experiment registry (auto-generated from eda_analysis.data.discover_arms)
    Experiment, EXPERIMENTS, build_experiments_from_disk, get_data_paths,
    get_model_names, get_model_eval_layout, resolve_paths,
)

from .conversations import (  # noqa: E402
    # Conversations (loaded so Run_Eval can score them)
    load_data, combine_data, reconstruct_conversation_text,
    # Model metadata
    parse_model_metadata, add_model_metadata_columns,
)

# Oracle pipeline — re-exported behind a flag because the questionnaires module
# needs to be reachable on sys.path. The constants-leaf path prepend above
# usually makes it work, but rare envs without it shouldn't break the whole
# package import.
from . import pipeline  # noqa: E402
EVAL_CODE_AVAILABLE = pipeline.EVAL_CODE_AVAILABLE
if EVAL_CODE_AVAILABLE:
    from .pipeline import (  # noqa: E402
        call_openai_json,
        evaluate_conversation,
        build_default_eval_configs,
        run_all_evaluations_async,
    )

# judge is NOT imported eagerly — it's only needed by Judge_Reliability.ipynb
# (``from eda_analysis.scoring import judge``) and pulls in provider SDK checks.

__all__ = [
    "WORKSPACE_ROOT",
    # registry
    "ORACLE_TOKEN_ALIASES", "COMPOSITE_METRICS",
    "EVAL_MODEL", "EVAL_TEMPERATURE", "MAX_RETRIES", "DEFAULT_CONCURRENCY",
    "DATA_DIR", "METHOD_DATA_DIR", "EVAL_QUESTIONNAIRE_DIRS",
    "eval_scores_root_for_method", "eval_csv_dir",
    "ScoringConfig",
    "Experiment", "EXPERIMENTS", "build_experiments_from_disk", "get_data_paths",
    "get_model_names", "get_model_eval_layout", "resolve_paths",
    # conversations
    "load_data", "combine_data", "reconstruct_conversation_text",
    "parse_model_metadata", "add_model_metadata_columns",
    # pipeline (gated)
    "pipeline", "EVAL_CODE_AVAILABLE",
]
if EVAL_CODE_AVAILABLE:
    __all__ += [
        "call_openai_json", "evaluate_conversation",
        "build_default_eval_configs", "run_all_evaluations_async",
    ]
