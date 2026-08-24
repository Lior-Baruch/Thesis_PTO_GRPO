"""eda_analysis -- the read-only analysis layer for Exp4_OpenStack.

Four families, one score lake, no registries: arms are discovered from disk, so a run becomes
analysable the moment its conversations and scores land. A notebook's whole preamble is::

    import os, eda_analysis
    cfg = eda_analysis.EdaConfig(family="arms/outcomes")
    S   = eda_analysis.notebook_setup(cfg)

Layering
--------
``constants`` is the leaf: paths, the metric registry, judge tags, label and colour keys, and the
``sys.path`` insert that makes ``questionnaires`` / ``naming`` / ``roles`` resolve to the SINGLE
canonical copies under ``code/``. Everything else imports it and nothing else imports upward.
``config`` owns the control surface. ``data`` / ``exports`` / ``stats`` / ``plotting`` do the work.

Import weight
-------------
``import eda_analysis`` loads only ``constants`` and ``config`` -- stdlib plus the two stdlib-only
canonical modules. Everything heavier (pandas, matplotlib, seaborn, pyarrow) arrives with the first
attribute that needs it, resolved through :func:`__getattr__`. Three things follow:

* a tool that just wants ``WORKSPACE_ROOT`` or ``FAMILIES`` pays nothing for it;
* :mod:`eda_analysis.scoring` -- the PAID side, which builds a client and talks to a grader -- is
  **never** reached implicitly. It is deliberately absent from the lazy map below; ``Run_Eval.ipynb``
  imports it explicitly (``from eda_analysis import scoring``), which is the point at which someone
  is choosing to spend;
* the analysis modules can be authored and imported in any order without a cycle.

The EDA must never import torch, ``core.policy`` or ``core.lookahead``. It reads finished artifacts;
if something here needs a GPU, it is in the wrong package.
"""

from __future__ import annotations

import importlib
from typing import Any, List, Tuple

# ---------------------------------------------------------------------------
# Eager: the leaf and the control surface. Both are cheap and both are what a
# non-notebook caller (render_results.py, _selfcheck, a one-off script) needs.
# ---------------------------------------------------------------------------
from .constants import (  # noqa: F401
    # workspace
    WORKSPACE_ROOT, CODE_DIR, DATA_DIR, RUNS_DIR, CONV_DIR, EVAL_SCORES_DIR, RESULTS_DIR,
    resolve_workspace_root, run_paths,
    # metrics
    Metric, METRICS, COMPOSITE_METRICS, ALL_METRICS, COMPOSITES, QUESTIONNAIRES,
    METRIC_ORDER, TRAINING_REWARD_METRIC, LOWER_IS_BETTER,
    metric, metric_for_qid, is_lower_better, sign_of, score_column, metric_partition,
    # reproducibility
    BOOT_SEED,
    # judges
    DEFAULT_JUDGE_TAG, judge_tag, judge_dirname, judge_dir,
    available_judge_tags, available_judge_reps,
    # labels + palette keys
    BASE_ARM, ARM_DISPLAY, ARM_COLORS, METRIC_COLORS, FALLBACK_COLORS, K_LINESTYLE,
    display_label, short_label, arm_label, k_of, method_of,
    # personas
    N_PERSONAS, PERSONA_COLS, COOP_LABEL, COOP_ORDER,
)
from .config import (  # noqa: F401
    EdaConfig, Setup, notebook_setup, FAMILIES, all_families, split_family,
)


# ---------------------------------------------------------------------------
# Lazy: the analysis modules and their entry points.
# ---------------------------------------------------------------------------

#: Submodules reachable as ``eda_analysis.<name>``, in the order :func:`__getattr__` searches them
#: when resolving a bare symbol. ``scoring`` is deliberately absent -- see the module docstring.
_SUBMODULES: Tuple[str, ...] = ("data", "exports", "stats", "plotting")

#: Entry points re-exported at package level, as ``name -> submodule``. This is the surface a
#: notebook may use unqualified; everything else stays module-qualified
#: (``eda_analysis.stats.<fn>``), which keeps a name from silently meaning one of several modules.
_ENTRY_POINTS = {
    "Arm": "data",
    "discover_arms": "data",
    "filter_arms": "data",
    "load_scores_long": "data",
    "scores_by_judge": "data",
    "judge_tags": "data",
    "save_fig": "exports",
    "save_table": "exports",
    "save_numbers": "exports",
    "save_provenance": "exports",
    "build_index": "exports",
    "reset_results": "exports",
    "set_family": "exports",
    "set_style": "plotting",
    "arm_palette": "plotting",
}

_LAZY_CACHE: dict = {}


def __getattr__(name: str) -> Any:
    """Resolve a submodule or a lazily re-exported entry point (PEP 562).

    A mapped entry point is fetched from its own module. An unmapped name is searched across
    :data:`_SUBMODULES` in order, so a function a sibling adds later is reachable without editing
    this file -- at the cost of importing those modules to look. The error names both the symbol and
    everywhere that was searched, because the usual cause is a contract drift between two modules
    rather than a caller's typo.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        _LAZY_CACHE[name] = module
        return module

    if name in _ENTRY_POINTS:
        module = importlib.import_module(f".{_ENTRY_POINTS[name]}", __name__)
        try:
            value = getattr(module, name)
        except AttributeError:
            raise AttributeError(
                f"eda_analysis.{name} is declared in _ENTRY_POINTS as coming from "
                f"eda_analysis.{_ENTRY_POINTS[name]}, but that module does not define it. "
                f"The two are out of date with each other."
            ) from None
        _LAZY_CACHE[name] = value
        return value

    for mod_name in _SUBMODULES:
        try:
            module = importlib.import_module(f".{mod_name}", __name__)
        except ImportError:
            continue
        if name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            _LAZY_CACHE[name] = value
            return value

    raise AttributeError(
        f"module 'eda_analysis' has no attribute {name!r} (searched this package's own names, "
        f"the entry-point map, and the __all__ of {list(_SUBMODULES)}). Note that "
        f"eda_analysis.scoring is never resolved implicitly -- import it explicitly."
    )


def __dir__() -> List[str]:
    """``dir()`` including the lazy names, so tab-completion sees the full surface."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # workspace
    "WORKSPACE_ROOT", "CODE_DIR", "DATA_DIR", "RUNS_DIR", "CONV_DIR", "EVAL_SCORES_DIR",
    "RESULTS_DIR", "resolve_workspace_root", "run_paths",
    # metrics
    "Metric", "METRICS", "COMPOSITE_METRICS", "ALL_METRICS", "COMPOSITES", "QUESTIONNAIRES",
    "METRIC_ORDER", "TRAINING_REWARD_METRIC", "LOWER_IS_BETTER",
    "metric", "metric_for_qid", "is_lower_better", "sign_of", "score_column", "metric_partition",
    # reproducibility
    "BOOT_SEED",
    # judges
    "DEFAULT_JUDGE_TAG", "judge_tag", "judge_dirname", "judge_dir",
    "available_judge_tags", "available_judge_reps",
    # labels + palette keys
    "BASE_ARM", "ARM_DISPLAY", "ARM_COLORS", "METRIC_COLORS", "FALLBACK_COLORS", "K_LINESTYLE",
    "display_label", "short_label", "arm_label", "k_of", "method_of",
    # personas
    "N_PERSONAS", "PERSONA_COLS", "COOP_LABEL", "COOP_ORDER",
    # control surface
    "EdaConfig", "Setup", "notebook_setup", "FAMILIES", "all_families", "split_family",
    # lazy submodules
    "data", "exports", "stats", "plotting",
    # lazy entry points
    "Arm", "discover_arms", "filter_arms", "load_scores_long", "scores_by_judge", "judge_tags",
    "save_fig", "save_table", "save_numbers", "save_provenance", "build_index", "reset_results",
    "set_family", "set_style", "arm_palette",
]
