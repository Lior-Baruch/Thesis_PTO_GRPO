"""
discovery.py — find Exp3 runs on disk and describe them (no hand-maintained registry).

Globs ``data/{pto,grpo}_Exp3/conversations/full/<EXP_NAME>/model_iter_<k>_TT*_TP*``
and reads the sibling ``runs/full/<EXP_NAME>/run_metadata.json`` for the seed +
training config. Produces one :class:`Arm` per run, each knowing where its
conversations and eval scores live and how to name its per-iteration eval models.

Experiment-name schemes (see Exp3 CLAUDE.md):
- GRPO: ``GRPO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_G{G}``
- PTO:  ``PTO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_M{M}_PT{greedy|indep}``
"""

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import DATA_DIR

_METHOD_DIRS = {"PTO": "pto_Exp3", "GRPO": "grpo_Exp3"}
_MODEL_PREFIX = {"PTO": "PTOExp3", "GRPO": "GRPOExp3"}
_DEFAULT_SEED = 42  # all current runs; only used if run_metadata.json is missing
_ITER_RE = re.compile(r"model_iter_(\d+)_")

_EXP_RE = re.compile(
    r"^(?P<method>PTO|GRPO)_Iterative_(?P<oracle>[A-Za-z0-9]+)_Llama32-1B_"
    r"LA(?P<K>\d+)_MCL(?P<mcl>\d+)_(?:G(?P<g>\d+)|M(?P<m>\d+)_PT(?P<mode>greedy|indep))$"
)


def parse_experiment_name(exp_name: str) -> Optional[dict]:
    """Parse an EXPERIMENT_NAME folder into its fields, or ``None`` if it doesn't match."""
    m = _EXP_RE.match(exp_name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "method": d["method"],
        "oracle": d["oracle"],
        "K": int(d["K"]),
        "mcl": int(d["mcl"]),
        "mode": d["mode"] or ("group" if d["method"] == "GRPO" else None),
        "branches": int(d["g"]) if d["g"] else (int(d["m"]) if d["m"] else None),
    }


@dataclass
class Arm:
    """One discovered training run (a method × K × MCL × mode arm)."""
    method: str                 # "PTO" | "GRPO"
    exp_name: str
    K: int
    mcl: int
    mode: Optional[str]
    oracle: str                 # training-oracle token, e.g. "Q1Q2"
    seed: int
    n_personas: int
    conv_dirs: Dict[int, str]   # model_iter k -> abs conversation dir
    runs_dir: str
    eval_root: str
    config: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.method}_LA{self.K}"

    @property
    def iters(self) -> List[int]:
        return sorted(self.conv_dirs)

    def model_name(self, k: int) -> str:
        prefix = _MODEL_PREFIX[self.method]
        tail = "Base" if k == 0 else f"I{k}"
        return f"{prefix}_LA{self.K}_{tail}"

    def eval_oracle_label(self, k: int) -> str:
        return "none" if k == 0 else self.oracle

    def eval_dir(self, k: int, metric_subdir: str) -> str:
        return os.path.join(
            self.eval_root, f"metric={metric_subdir}",
            f"oracle={self.eval_oracle_label(k)}", self.model_name(k),
        )

    def conv_dir(self, k: int) -> Optional[str]:
        return self.conv_dirs.get(k)


def discover_arms(data_dir: str = DATA_DIR, *, include_archived: bool = False) -> List[Arm]:
    """Discover all Exp3 arms present on disk, newest-data-first within method."""
    arms: List[Arm] = []
    for method, mdir in _METHOD_DIRS.items():
        conv_root = os.path.join(data_dir, mdir, "conversations", "full")
        if not os.path.isdir(conv_root):
            continue
        for exp_name in sorted(os.listdir(conv_root)):
            if not include_archived and "Archive" in exp_name:
                continue
            exp_path = os.path.join(conv_root, exp_name)
            if not os.path.isdir(exp_path):
                continue
            parsed = parse_experiment_name(exp_name)
            if parsed is None:
                continue
            # iter dirs present
            conv_dirs: Dict[int, str] = {}
            for d in glob.glob(os.path.join(exp_path, "model_iter_*")):
                m = _ITER_RE.search(os.path.basename(d) + "_")
                if m and os.path.isdir(d):
                    conv_dirs[int(m.group(1))] = d
            if not conv_dirs:
                continue
            runs_dir = os.path.join(data_dir, mdir, "runs", "full", exp_name)
            seed, cfg = _read_seed_config(runs_dir)
            arms.append(Arm(
                method=parsed["method"], exp_name=exp_name, K=parsed["K"],
                mcl=parsed["mcl"], mode=parsed["mode"], oracle=parsed["oracle"],
                seed=seed, n_personas=int(cfg.get("num_conversations_per_iter", 96)),
                conv_dirs=conv_dirs, runs_dir=runs_dir,
                eval_root=os.path.join(data_dir, mdir, "eval_scores"), config=cfg,
            ))
    return arms


def filter_arms(arms: List[Arm], *, methods=None, ks=None, modes=None,
                arm_labels=None) -> List[Arm]:
    """Filter a discovered arm list by method / K / mode / explicit label (each None = no filter).

    Used by ``notebook_setup`` to honour ``EdaConfig`` arm selection. ``arm_labels`` is an explicit
    whitelist on ``Arm.label`` (e.g. ``["PTO_LA0"]``), applied alongside the field filters.
    """
    def keep(a: Arm) -> bool:
        if methods and a.method not in set(methods):
            return False
        if ks is not None and a.K not in set(ks):
            return False
        if modes and a.mode not in set(modes):
            return False
        if arm_labels and a.label not in set(arm_labels):
            return False
        return True
    return [a for a in arms if keep(a)]


def _read_seed_config(runs_dir: str):
    """Return ``(seed, config_dict)`` from ``run_metadata.json`` (defaults if absent)."""
    meta = os.path.join(runs_dir, "run_metadata.json")
    if os.path.exists(meta):
        try:
            cfg = json.load(open(meta)).get("config", {})
            return int(cfg.get("seed", _DEFAULT_SEED)), cfg
        except Exception:
            pass
    print(f"  [discovery] WARNING: no run_metadata.json under {runs_dir} — "
          f"assuming seed={_DEFAULT_SEED}")
    return _DEFAULT_SEED, {}
