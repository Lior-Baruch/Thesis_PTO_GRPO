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


def select_scores(scores_long: pd.DataFrame, *, arms: Optional[List] = None,
                  iters: Optional[List] = None, metrics: Optional[List] = None) -> pd.DataFrame:
    """Slice ``scores_long`` to chosen arms / iterations / metrics (each None = keep all).

    The one selection helper every figure cell uses, so a notebook can point a plot at a subset
    (e.g. ``select_scores(S.SCORES, arms=["PTO_LA0","GRPO_LA0"])``) instead of looping per arm.
    """
    d = scores_long
    if arms is not None:
        d = d[d["arm"].isin(list(arms))]
    if iters is not None:
        d = d[d["iteration"].isin(list(iters))]
    if metrics is not None:
        d = d[d["questionnaire"].isin(list(metrics))]
    return d


def collapse_base(scores_long: pd.DataFrame, *, label: str = "Base") -> pd.DataFrame:
    """Pool every arm's iter-0 base into ONE descriptive model row block.

    All arms share the same base policy (frozen Llama-3.2-1B) on the same iter-0 persona
    order (shuffle ``seed+1`` for every arm), so the per-arm ``*_Base`` rows are near-replicates.
    For cross-model *descriptive* views (bars / subscales / violins) this relabels them to a
    single pooled model — decluttering the axis and giving a higher-N base reference.

    Relabel: ``model=label``, ``arm=label``, ``method="Base"``, ``K=-1`` (so
    :func:`figures.model_order` sorts it first). Non-base rows pass through untouched.

    NOTE: descriptive only — do **not** feed this to the persona-paired / vs-base ``stats.*``
    helpers, which must keep pairing each arm against its OWN base.
    """
    if scores_long.empty or "is_base" not in scores_long.columns:
        return scores_long
    out = scores_long.copy()
    base = out["is_base"]
    out.loc[base, "model"] = label
    out.loc[base, "arm"] = label
    if "method" in out.columns:
        out.loc[base, "method"] = "Base"
    if "K" in out.columns:
        out.loc[base, "K"] = -1
    return out


def add_derived_mitiprof_rows(scores_long: pd.DataFrame,
                              arms: Optional[List] = None) -> pd.DataFrame:
    """Append the **objective MITI-proficiency ratios** as extra ``questionnaire`` rows.

    Derived for FREE from the already-scored MITI behavior counts (no oracle re-run):

    - ``R:Q``   = (SR + CR) / Q                      — reflection-to-question ratio
    - ``%CR``   = CR / (SR + CR)                      — proportion complex reflections
    - ``%MICO`` = (SR+CR+AF+Seek) / (SR+CR+AF+Seek+Persuade)  — MI-consistent proportion

    These ratios are technique metrics (not warmth halos), so they belong in the inter-rubric
    correlation/PCA as candidate *orthogonal* axes. Rows are aligned to the existing
    ``scores_long`` conversation identities by (arm, iteration, file_index), inheriting the full
    key + persona columns, so they pivot onto the same rows in :func:`to_wide`.
    Returns ``scores_long`` unchanged if MITI behavior data is unavailable.
    """
    from .behavior import load_miti_behavior
    if scores_long.empty:
        return scores_long
    # Idempotent: notebook_setup already appends these, so a notebook re-calling is a no-op.
    if "R:Q" in set(scores_long["questionnaire"].unique()):
        return scores_long
    miti = load_miti_behavior(arms, attach_persona=False)
    if miti.empty:
        return scores_long

    def _ratio(num, den):
        return num / den if (den is not None and den > 0) else None

    recs = []
    for _, r in miti.iterrows():
        sr, cr = r.get("B4_SR") or 0, r.get("B5_CR") or 0
        q, af, seek = r.get("B3_Q") or 0, r.get("B6_AF") or 0, r.get("B7_Seek") or 0
        pers = r.get("B2_Persuade") or 0
        mico = sr + cr + af + seek
        recs.append({"arm": r["arm"], "iteration": r["iteration"], "file_index": r["file_index"],
                     "R:Q": _ratio(sr + cr, q), "%CR": _ratio(cr, sr + cr),
                     "%MICO": _ratio(mico, mico + pers)})
    deriv = pd.DataFrame(recs)
    if deriv.empty:
        return scores_long

    # Skeleton of conversation identities (full key + persona) from scores_long.
    id_cols = [c for c in scores_long.columns if c not in ("questionnaire", "score")]
    skel = scores_long[id_cols].drop_duplicates(["arm", "iteration", "file_index"])
    merged = skel.merge(deriv, on=["arm", "iteration", "file_index"], how="inner")
    if merged.empty:
        return scores_long
    long_new = merged.melt(id_vars=id_cols, value_vars=["R:Q", "%CR", "%MICO"],
                           var_name="questionnaire", value_name="score").dropna(subset=["score"])
    return pd.concat([scores_long, long_new], ignore_index=True)


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
