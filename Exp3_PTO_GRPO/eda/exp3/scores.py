"""
scores.py — the tidy long backbone every analysis derives from.

``load_scores_long`` reads each arm's per-conversation eval CSVs
(``eval_scores/metric=<M>/oracle=<O>/<model>/<file>.csv``), recovers the true
persona per conversation (via :mod:`personas`), and returns one row per
``(arm, iteration, persona, questionnaire) -> score``.

Composite: ``Q1Q2 = mean(Q1_Mean, Q2_Mean)`` — matches the project's existing
``COMPOSITE_METRICS`` convention and the headline chart's axis (a [1,5] mean, not
the [2,10] sum).
"""

import os
from typing import List, Optional

import pandas as pd

from . import QUESTIONNAIRES, discover_arms
from .personas import attach_personas

# Display name -> per-conv mean column (the non-composite rubrics).
MEAN_COLS = {disp: meancol for disp, (sub, meancol) in QUESTIONNAIRES.items() if sub is not None}

_KEY = ["method", "arm", "K", "mcl", "mode", "oracle", "model", "iteration", "is_base", "file_index"]


def load_scores_long(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """Tidy long eval scores across all discovered arms.

    Columns: ``method, arm, K, mcl, mode, oracle, model, iteration, is_base,
    file_index, questionnaire, score`` (+ ``persona_id`` & characteristics if
    ``attach_persona``). Includes the ``Q1Q2`` composite. Missing eval folders are
    skipped, so partially-scored arms contribute whatever exists.
    """
    arms = discover_arms() if arms is None else arms
    rows = []
    for arm in arms:
        for k in arm.iters:
            base_meta = {
                "method": arm.method, "arm": arm.label, "K": arm.K, "mcl": arm.mcl,
                "mode": arm.mode, "oracle": arm.oracle, "model": arm.model_name(k),
                "iteration": k, "is_base": (k == 0),
            }
            for disp, (sub, meancol) in QUESTIONNAIRES.items():
                if sub is None:  # composite — built after the raw load
                    continue
                ddir = arm.eval_dir(k, sub)
                if not os.path.isdir(ddir):
                    continue
                for fn in os.listdir(ddir):
                    stem, ext = os.path.splitext(fn)
                    if ext != ".csv" or not stem.isdigit():
                        continue
                    try:
                        r = pd.read_csv(os.path.join(ddir, fn))
                    except Exception:
                        continue
                    if len(r) == 0 or meancol not in r.columns:
                        continue
                    rows.append({**base_meta, "file_index": int(stem),
                                 "questionnaire": disp, "score": float(r.iloc[0][meancol])})
    long = pd.DataFrame(rows)
    if long.empty:
        return long

    long = _add_q1q2_composite(long)
    if attach_persona:
        # seed is constant per arm; current runs all share seed — attach per arm.
        seed_by_arm = {a.label: a.seed for a in arms}
        parts = []
        for arm_label, g in long.groupby("arm", sort=False):
            parts.append(attach_personas(g, seed_by_arm.get(arm_label, 42)))
        long = pd.concat(parts, ignore_index=True)
    return long


def _add_q1q2_composite(long: pd.DataFrame) -> pd.DataFrame:
    """Append the ``Q1Q2`` = mean(Q1, Q2) composite rows (where both components exist)."""
    comp_src = long[long["questionnaire"].isin(["Q1", "Q2"])]
    if comp_src.empty:
        return long
    wide = comp_src.pivot_table(index=_KEY, columns="questionnaire", values="score")
    if not {"Q1", "Q2"}.issubset(wide.columns):
        return long
    wide = wide.dropna(subset=["Q1", "Q2"])
    comp = wide.reset_index()
    comp["questionnaire"] = "Q1Q2"
    comp["score"] = comp[["Q1", "Q2"]].mean(axis=1)
    comp = comp[_KEY + ["questionnaire", "score"]]
    return pd.concat([long, comp], ignore_index=True)


_SUBSCALES = {
    "WAI-SR": ("WAI_SR", {"WAI_Goal_Mean": "Goal", "WAI_Task_Mean": "Task", "WAI_Bond_Mean": "Bond"}),
    "MITI": ("MITI", {"MITI1_CultivatingChangeTalk": "ChangeTalk", "MITI2_SofteningSustainTalk": "SoftenSustain",
                       "MITI3_Partnership": "Partnership", "MITI4_Empathy": "Empathy"}),
}


def load_subscales(arms: Optional[List] = None) -> pd.DataFrame:
    """Tidy long frame of WAI (Goal/Task/Bond) + MITI (4 globals) subscales.

    One row per (arm, iteration, file_index, parent questionnaire, subscale) -> score.
    Used by the familiar 'subscales' view; complements the headline-mean `scores_long`.
    """
    arms = discover_arms() if arms is None else arms
    rows = []
    for arm in arms:
        for k in arm.iters:
            for parent, (sub, cols) in _SUBSCALES.items():
                ddir = arm.eval_dir(k, sub)
                if not os.path.isdir(ddir):
                    continue
                for fn in os.listdir(ddir):
                    stem, ext = os.path.splitext(fn)
                    if ext != ".csv" or not stem.isdigit():
                        continue
                    try:
                        r = pd.read_csv(os.path.join(ddir, fn)).iloc[0]
                    except Exception:
                        continue
                    for src, name in cols.items():
                        if src in r.index and pd.notna(r[src]):
                            rows.append({"arm": arm.label, "method": arm.method, "K": arm.K,
                                         "model": arm.model_name(k), "iteration": k,
                                         "is_base": (k == 0), "file_index": int(stem),
                                         "parent": parent, "subscale": name, "score": float(r[src])})
    return pd.DataFrame(rows)


def to_wide(scores_long: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    """Pivot to one row per (arm, iteration, persona) with a column per questionnaire.

    Convenient for paired stats + inter-rubric correlation. Persona characteristics
    are carried through if present.
    """
    from . import PERSONA_COLS
    idx = ["method", "arm", "K", "oracle", "model", "iteration", "is_base", "file_index"]
    if "persona_id" in scores_long.columns:
        idx.append("persona_id")
    wide = scores_long.pivot_table(index=idx, columns="questionnaire", values=value).reset_index()
    if "persona_id" in scores_long.columns:
        chars = (scores_long[["persona_id"] + [c for c in PERSONA_COLS if c in scores_long.columns]]
                 .drop_duplicates("persona_id"))
        wide = wide.merge(chars, on="persona_id", how="left")
    wide.columns.name = None
    return wide
