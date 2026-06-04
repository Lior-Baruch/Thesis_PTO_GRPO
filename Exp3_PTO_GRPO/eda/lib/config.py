"""
config.py — Constants, palettes, dataclasses, and the experiment registry.

Everything in here is pure data + small helpers — no I/O. Imported by every
notebook cell that needs styling, paths, or the list of experiments.

All paths in :data:`EXPERIMENTS` are stored **relative to the experiment root**
(the Exp3 folder, where the API keys live). :func:`resolve_paths` joins each
entry with :data:`lib.WORKSPACE_ROOT` to produce absolute paths.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from . import WORKSPACE_ROOT


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONSTANTS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Figure sizes
FIG_WIDE = (16, 5)
FIG_SINGLE = (12, 5)

# Statistical thresholds
ALPHA = 0.05
EFFECT_SIZE_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}

# Ordering (left-to-right in every plot)
ORACLE_ORDER = ["WAI", "CSQ8", "Q1Q2", "MI_SAT", "MITI"]
GROUP_ORDER = {"Base": 0, "GRPO_Exp3": 4, "PTO_Exp3": 5}
DPO_GROUP_ORDER = 1  # All DPO (L0/L5) variants share this rank

EXPERIMENT_PALETTE = {
    "Base":    "orange",
    "L0_Q1Q2": "#1f77b4", "L5_Q1Q2": "#6baed6",
    "L0_CSQ8": "#2ca02c", "L5_CSQ8": "#74c476",
    "L0_WAI":  "#d62728", "L5_WAI":  "#fb6a4a",
    "GRPO_Exp3": "#aec7e8",
    "PTO_Exp3":  "#c5b0d5",
}

# Oracle key -> (display name for plots, DataFrame column)
ORACLE_METRIC_MAP = {
    "WAI":    ("WAI-SR",  "WAI_TotalMean"),
    "CSQ8":   ("CSQ-8",   "CSQ8_Mean"),
    "Q1Q2":   ("Q1+Q2",   "Q1Q2_Mean"),
    "MI_SAT": ("MI-SAT",  "MI_Mean"),
    "MITI":   ("MITI",    "MITI_GlobalMean"),
    "Q1":     ("Q1",      "Q1_Mean"),
    "Q2":     ("Q2",      "Q2_Mean"),
}


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
    "DPO":       "pto_Exp2",   # Exp2 PTO baselines + Base (frozen reference)
    "GRPO_Exp3": "grpo_Exp3",
    "PTO_Exp3":  "pto_Exp3",
}

# Questionnaire display name -> on-disk folder basename under <method>/eval_scores/.
EVAL_QUESTIONNAIRE_DIRS = {
    "CSQ-8":  "CSQ8",
    "WAI-SR": "WAI_SR",
    "MI-SAT": "MI_SAT",
    "MITI":   "MITI",
    "Q1":     "Q1",
    "Q2":     "Q2",
}


def eval_scores_root_for_method(method: str) -> str:
    """Absolute ``data/<method_dir>/eval_scores`` for a training method.

    Unknown methods fall back to the Exp2 reference dir (``pto_Exp2``).
    """
    return os.path.join(DATA_DIR, METHOD_DATA_DIR.get(method, "pto_Exp2"), "eval_scores")


def eval_csv_dir(root: str, oracle: str, metric_subdir: str, model: str) -> str:
    """Folder holding a model's per-patient eval CSVs for one metric.

    Layout: ``<root>/metric=<metric>/oracle=<oracle>/<model>/``. The two labelled
    levels make the *scoring metric* (what graded it) and the *training oracle*
    (what it was trained on) explicit and unambiguous — e.g.
    ``…/eval_scores/metric=WAI_SR/oracle=Q1Q2/L5_Q1Q2_V10/``.
    """
    return os.path.join(root, f"metric={metric_subdir}", f"oracle={oracle}", model)


def set_plot_style() -> None:
    """Apply consistent global matplotlib + seaborn styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150


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
    method: str = "DPO"
    eval_model: str = EVAL_MODEL
    eval_temp: float = EVAL_TEMPERATURE
    async_concurrency: int = DEFAULT_CONCURRENCY
    eval_base_dir: Optional[str] = None

    def __post_init__(self):
        if self.eval_base_dir is None:
            self.eval_base_dir = eval_scores_root_for_method(self.method)


@dataclass
class PlotContext:
    """Bundle of plot defaults (palette, ordering, baseline) used across cells."""
    model_palette: dict = field(default_factory=dict)
    experiment_palette: dict = field(default_factory=dict)
    hue_col: str = "ExperimentGroup"
    model_order: Optional[list] = None
    baseline_model: str = "Base"
    show: bool = True


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
    method: str = "DPO"
    epoch: Optional[int] = None

    @property
    def model_name(self) -> str:
        if self.method == "GRPO_Exp3":
            return "GRPOExp3_Base" if self.epoch == 0 else f"GRPOExp3_I{self.epoch}"
        if self.method == "PTO_Exp3":
            return "PTOExp3_Base" if self.epoch == 0 else f"PTOExp3_I{self.epoch}"
        if self.oracle == "Base":
            return "Base"
        return f"L{self.lookahead}_{self.oracle}_V{self.version}"

    @property
    def oracle_label(self) -> str:
        """Training-oracle folder label; 'none' for the untrained Base rollout."""
        return "none" if self.oracle in (None, "Base") else self.oracle


# Path roots are experiment-root-relative (the experiment root is the Exp3 dir).
_PTO_CONV = "data/pto_Exp2/eval_conversations"
_GRPO_CONV = "data/grpo_Exp3/conversations"
_PTO_EXP3_CONV = "data/pto_Exp3/conversations"

EXPERIMENTS: List[Experiment] = [
    # Base
    Experiment("Base", None, None, f"{_PTO_CONV}/Base/Good_50_TT0.9_TP0.7_TE0.1"),

    # L0 WAI
    *[
        Experiment("WAI", 0, v, f"{_PTO_CONV}/WAI/LookAhead_0/TTree1.2_TT0.9_TP0.7_TE0.1_V{v}")
        for v in range(1, 6)
    ],
    # L0 CSQ8
    *[
        Experiment("CSQ8", 0, v, f"{_PTO_CONV}/CSQ-8/LookAhead_0/TTree1.2_TT0.9_TP0.7_TE0.1_V{v}")
        for v in range(1, 6)
    ],
    # L0 Q1Q2
    *[
        Experiment("Q1Q2", 0, v, f"{_PTO_CONV}/Q1Q2/LookAhead_0/TTree1.2_TT0.9_TP0.7_TE0.2_V{v}")
        for v in range(1, 6)
    ],
    # L5 WAI
    *[
        Experiment("WAI", 5, v, f"{_PTO_CONV}/WAI/LookAhead_5/TTree1.2_TT0.9_TP0.7_TE0.1_V{v}")
        for v in range(1, 6)
    ],
    # L5 CSQ8
    *[
        Experiment("CSQ8", 5, v, f"{_PTO_CONV}/CSQ-8/LookAhead_5/TTree1.2_TT0.9_TP0.7_TE0.1_V{v}")
        for v in range(1, 6)
    ],
    # L5 Q1Q2
    *[
        Experiment("Q1Q2", 5, v, f"{_PTO_CONV}/Q1Q2/LookAhead_5/TTree1.2_TT0.9_TP0.7_TE0.2_V{v}")
        for v in range(1, 11)
    ],

    # GRPO_Exp3 — uncomment + edit per real run.
    # epoch=0 → GRPOExp3_Base (base-model run, model_iter_0), epoch=N → GRPOExp3_IN.
    # Conv subdir pattern: ``model_iter_{N}_TT{temp_t}_TP{temp_p}``.
    # _GRPOExp3_EXP = "GRPO_Iterative_Q1Q2_Llama32-1B_LA5_MCL10_G4"  # _<ORACLE> token (Q1Q2/WAI/CSQ8/MI_SAT/MITI) must match Experiment(oracle=...)
    # Experiment("Base", None, None, f"{_GRPO_CONV}/full/{_GRPOExp3_EXP}/model_iter_0_TT0.9_TP0.7", method="GRPO_Exp3", epoch=0),
    # Experiment("Q1Q2", None, None, f"{_GRPO_CONV}/full/{_GRPOExp3_EXP}/model_iter_1_TT0.9_TP0.7", method="GRPO_Exp3", epoch=1),
    # Experiment("Q1Q2", None, None, f"{_GRPO_CONV}/full/{_GRPOExp3_EXP}/model_iter_2_TT0.9_TP0.7", method="GRPO_Exp3", epoch=2),

    # PTO_Exp3 — uncomment + edit per real run. Same shape as GRPO_Exp3, written
    # under data/pto_Exp3/ (its scores co-locate at data/pto_Exp3/eval_scores/).
    # method="PTO_Exp3" → model_name PTOExp3_Base / PTOExp3_I{epoch}.
    # _PTO_EXP3_NAME = "PTO_Iterative_Q1Q2_Llama32-1B_LA5_MCL10_M4_PTgreedy"  # _<ORACLE> token must match Experiment(oracle=...)
    # Experiment("Base", None, None, f"{_PTO_EXP3_CONV}/full/{_PTO_EXP3_NAME}/model_iter_0_TT0.9_TP0.7", method="PTO_Exp3", epoch=0),
    # Experiment("Q1Q2", None, None, f"{_PTO_EXP3_CONV}/full/{_PTO_EXP3_NAME}/model_iter_1_TT0.9_TP0.7", method="PTO_Exp3", epoch=1),
    # Experiment("Q1Q2", None, None, f"{_PTO_EXP3_CONV}/full/{_PTO_EXP3_NAME}/model_iter_2_TT0.9_TP0.7", method="PTO_Exp3", epoch=2),
]


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
