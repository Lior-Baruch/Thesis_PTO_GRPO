"""
config.py — Constants, palettes, dataclasses, and the experiment registry.

Constants + dataclasses are pure data; the :data:`EXPERIMENTS` registry is
**auto-generated at import** from ``eda_analysis.data.discover_arms()`` (one disk
scan — EDA roadmap #7, 2026-07-11), so a freshly-landed run is scoreable by
Run_Eval with no registry edit.

All paths in :data:`EXPERIMENTS` are stored **relative to the experiment root**
(the Exp3 folder, where the API keys live). :func:`resolve_paths` joins each
entry with :data:`WORKSPACE_ROOT` to produce absolute paths.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


from . import WORKSPACE_ROOT


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONSTANTS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Alias map for oracle tokens parsed from model names. ``data._normalize_oracle_token``
# uppercases + replaces ``-`` with ``_`` before lookup, so keys must be uppercase
# canonical forms. Add new aliases here, not in data.py. Unknown tokens fall
# through to ``"Other"`` unless ``strict=True`` is passed.
ORACLE_TOKEN_ALIASES = {
    "CSQ":          "CSQ8",
    "CSQ8":         "CSQ8",
    "CSQ_8":        "CSQ8",
    "Q1Q2":         "Q1Q2",
    "Q1_Q2":        "Q1Q2",
    "WAI":          "WAI",
    "WAI_SR":       "WAI",
    "MI":           "MI_SAT",
    "MISAT":        "MI_SAT",
    "MI_SAT":       "MI_SAT",
    "MITI":         "MITI",
    "MITI_GLOBALS": "MITI",
    "BASE":         "Base",
}


# Composite metrics built by merging per-questionnaire eval scores.
# ``output_col`` is the column added to the merged DataFrame; ``sources``
# is the list of input columns to mean-average. ``aggregator`` is the
# reduction (currently only "mean"). Extend here when adding new composites
# (e.g. MITI_GlobalMean from the 4 MITI globals).
COMPOSITE_METRICS = {
    "Q1Q2_Mean": {
        "sources": ["Q1_Mean", "Q2_Mean"],
        "aggregator": "mean",
    },
}

# Eval-model settings (oracle scoring)
EVAL_MODEL = "gpt-4o-mini-2024-07-18"
EVAL_TEMPERATURE = 0.1
MAX_RETRIES = 3
DEFAULT_CONCURRENCY = 32


# Each training method owns its eval_scores/ inside its own data dir, so a
# model's gradings live next to the conversations they grade and method
# namespaces never collide. The score path is resolved per-model from the
# experiment's ``method`` + training ``oracle`` (see ``eval_scores_root_for_method``,
# ``get_model_eval_layout``, ``eval_csv_dir``).
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")

# Training method -> the data/ subdir that owns its eval_scores/.
METHOD_DATA_DIR = {
    "GRPO_Exp3": "grpo_Exp3",
    "PTO_Exp3":  "pto_Exp3",
}

# Questionnaire display name -> on-disk folder basename under <method>/eval_scores/.
EVAL_QUESTIONNAIRE_DIRS = {
    "CSQ-8":  "CSQ8",
    "WAI-SR": "WAI_SR",
    "MITI":   "MITI",
    "MI-SAT": "MI_SAT",
    "Q1":     "Q1",
    "Q2":     "Q2",
    "PCT":    "PCT",     # Patient Change Talk (orthogonal MI mechanism/outcome)
    "MICI":   "MICI",    # MI-Inconsistent therapist behaviors (negative-valence)
}


def eval_scores_root_for_method(method: str) -> str:
    """Absolute ``data/<method_dir>/eval_scores`` for a training method.

    Raises ``KeyError`` on an unknown method (the Exp2 ``pto_Exp2`` fallback was
    removed 2026-06-15 — see ``METHOD_DATA_DIR``).
    """
    try:
        return os.path.join(DATA_DIR, METHOD_DATA_DIR[method], "eval_scores")
    except KeyError:
        raise KeyError(
            f"Unknown training method {method!r}; expected one of "
            f"{sorted(METHOD_DATA_DIR)}"
        )


def eval_csv_dir(root: str, oracle: str, metric_subdir: str, model: str) -> str:
    """Folder holding a model's per-patient eval CSVs for one metric.

    Layout: ``<root>/metric=<metric>/oracle=<oracle>/<model>/``. The two labelled
    levels make the *scoring metric* (what graded it) and the *training oracle*
    (what it was trained on) explicit and unambiguous — e.g.
    ``…/eval_scores/metric=WAI_SR/oracle=Q1Q2/L5_Q1Q2_V10/``.
    """
    return os.path.join(root, f"metric={metric_subdir}", f"oracle={oracle}", model)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                             DATACLASSES                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class EDAConfig:
    """Runtime knobs for eval and analysis.

    Eval scores are co-located per method and labelled by metric + training
    oracle: ``data/<method>/eval_scores/metric=<M>/oracle=<O>/<model>/``.
    ``method`` selects the method root (``eval_base_dir``); cross-method work
    (Run_Eval / Conv_EDA) resolves each model's root + oracle via
    :func:`get_model_eval_layout` and builds paths with :func:`eval_csv_dir`.
    """
    method: str = "GRPO_Exp3"
    eval_model: str = EVAL_MODEL
    eval_temp: float = EVAL_TEMPERATURE
    async_concurrency: int = DEFAULT_CONCURRENCY
    eval_base_dir: Optional[str] = None

    def __post_init__(self):
        if self.eval_base_dir is None:
            self.eval_base_dir = eval_scores_root_for_method(self.method)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         EXPERIMENT REGISTRY                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class Experiment:
    """One (method × oracle × look-ahead × version) data location."""
    oracle: str
    lookahead: Optional[int]
    version: Optional[int]
    path: str
    method: str = "GRPO_Exp3"
    epoch: Optional[int] = None

    @property
    def model_name(self) -> str:
        # Exp3 methods carry their look-ahead K in the name (and so in the per-model
        # eval_scores/.../<model>/ folder) so LA0 and LA5 arms of the same method
        # never collide. ``lookahead`` is required for Exp3 entries.
        if self.method == "GRPO_Exp3":
            tail = "Base" if self.epoch == 0 else f"I{self.epoch}"
            return f"GRPOExp3_LA{self.lookahead}_{tail}"
        if self.method == "PTO_Exp3":
            tail = "Base" if self.epoch == 0 else f"I{self.epoch}"
            return f"PTOExp3_LA{self.lookahead}_{tail}"
        if self.oracle == "Base":
            return "Base"
        return f"L{self.lookahead}_{self.oracle}_V{self.version}"

    @property
    def oracle_label(self) -> str:
        """Training-oracle folder label; 'none' for the untrained Base rollout."""
        return "none" if self.oracle in (None, "Base") else self.oracle


def build_experiments_from_disk() -> List[Experiment]:
    """Auto-generate the registry from ``eda_analysis.data.discover_arms()``.

    One :class:`Experiment` per ``model_iter_N`` conv dir actually on disk (with
    conversations — empty in-flight dirs are skipped by discovery), per discovered
    arm. epoch=0 = the base-model rollout (``model_iter_0`` → ``<METHOD>Exp3_LA{K}_Base``),
    epoch=N = the iter-N policy (→ ``…_I{N}``). ``lookahead`` = the arm's K, so LA0/LA5
    (and any future LA2) arms get distinct model names + eval_scores folders; the arm's
    training-oracle token (Q1Q2/WAI/…) comes from its EXPERIMENT_NAME. Paths are stored
    experiment-root-relative, matching :func:`resolve_paths`.

    This replaced the hand-maintained list (2026-07-11, EDA roadmap #7): a new run is
    picked up by Run_Eval as soon as its conversations land — no registry edit. If the
    Drive symlinks are offline, discovery finds nothing and the registry is empty
    (a warning is printed at import).
    """
    import sys
    _eda_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _eda_dir not in sys.path:
        sys.path.insert(0, _eda_dir)
    from eda_analysis.data import discover_arms  # sibling package under eda/

    exps: List[Experiment] = []
    for arm in sorted(discover_arms(), key=lambda a: (a.method, a.K)):
        method = f"{arm.method}_Exp3"
        for k in arm.iters:
            exps.append(Experiment(
                oracle="Base" if k == 0 else arm.oracle,
                lookahead=arm.K,
                version=None,
                path=os.path.relpath(arm.conv_dirs[k], WORKSPACE_ROOT),
                method=method,
                epoch=k,
            ))
    return exps


EXPERIMENTS: List[Experiment] = build_experiments_from_disk()
if not EXPERIMENTS:
    print("[oracle_scoring.config] WARNING: discover_arms() found no runs on disk — "
          "EXPERIMENTS is empty (are the data/ Drive symlinks mounted?)")


def get_data_paths(experiments: Optional[List[Experiment]] = None) -> List[str]:
    """``[e.path for e in experiments]`` (defaults to :data:`EXPERIMENTS`)."""
    experiments = experiments or EXPERIMENTS
    return [e.path for e in experiments]


def get_model_names(experiments: Optional[List[Experiment]] = None) -> List[str]:
    """``[e.model_name for e in experiments]`` (defaults to :data:`EXPERIMENTS`)."""
    experiments = experiments or EXPERIMENTS
    return [e.model_name for e in experiments]


def get_model_eval_layout(experiments: Optional[List[Experiment]] = None) -> Dict[str, Dict[str, str]]:
    """``{model_name: {'root': <eval_scores root>, 'oracle': <oracle label>}}``.

    The single source of truth for *where a model's gradings live* (``root``,
    per method) and *which training oracle produced it* (``oracle``). Drives the
    ``<root>/metric=<M>/oracle=<O>/<model>/`` layout — build paths with
    :func:`eval_csv_dir`. The writer (``eval.run_all_evaluations_async``) and
    reader (``data.load_all_eval_results``) both take this map.
    """
    experiments = experiments or EXPERIMENTS
    return {
        e.model_name: {"root": eval_scores_root_for_method(e.method), "oracle": e.oracle_label}
        for e in experiments
    }


def resolve_paths(experiments: Optional[List[Experiment]] = None) -> List[str]:
    """Absolute paths produced by joining each ``e.path`` with :data:`WORKSPACE_ROOT`.

    Paths that don't exist on disk are returned unchanged — :func:`data.load_data`
    skips missing experiments so the EDA still runs on whatever IS available.
    """
    return [os.path.join(WORKSPACE_ROOT, p) for p in get_data_paths(experiments)]
