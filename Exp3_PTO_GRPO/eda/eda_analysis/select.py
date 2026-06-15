"""
select.py — the model-selection toggle for cross-model views.

Two modes (the user's requirement):
- ``all_models``         — every iteration of every arm.
- ``best_per_experiment`` — the peak iteration per arm judged by *its own training
  oracle* (Q1Q2 for the current arms; generalizes to WAI-SR/CSQ-8/… for the oracle
  sweep). Each arm's base (iter 0) is always retained as the reference.

Both return a filtered copy of ``scores_long`` so downstream plots/stats are
selection-agnostic.
"""

from typing import Optional, Tuple

import pandas as pd

# Training-oracle token -> the questionnaire display name that judges it.
_OWN_ORACLE = {
    "Q1Q2": "Q1Q2", "WAI": "WAI-SR", "CSQ8": "CSQ-8", "MI_SAT": "MI-SAT", "MITI": "MITI",
}


def all_models(scores_long: pd.DataFrame) -> pd.DataFrame:
    """Identity passthrough (every iteration of every arm)."""
    return scores_long


def best_per_experiment(
    scores_long: pd.DataFrame,
    by: str = "own_oracle",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep each arm's base + its best-scoring iteration on its own oracle.

    Returns ``(filtered_scores_long, summary)`` where ``summary`` has one row per
    arm: the selected best iteration, its own-oracle mean, and n personas. Ties
    break to the earliest iteration.
    """
    if by != "own_oracle":
        raise ValueError(f"unsupported selection mode {by!r} (only 'own_oracle')")
    if scores_long.empty:
        return scores_long, pd.DataFrame()

    keep_models, summary_rows = [], []
    for (arm, oracle), g in scores_long.groupby(["arm", "oracle"], sort=False):
        judge = _OWN_ORACLE.get(oracle)
        sub = g[g["questionnaire"] == judge] if judge else g.iloc[0:0]
        # always keep the base
        base_models = g.loc[g["is_base"], "model"].unique().tolist()
        keep_models += base_models
        if sub.empty:
            continue
        per_iter = (sub[~sub["is_base"]]
                    .groupby(["iteration", "model"], observed=True)["score"]
                    .mean().reset_index()
                    .sort_values(["score", "iteration"], ascending=[False, True]))
        if per_iter.empty:
            continue
        best = per_iter.iloc[0]
        keep_models.append(best["model"])
        summary_rows.append({
            "arm": arm, "oracle": oracle, "judged_by": judge,
            "best_iteration": int(best["iteration"]), "best_model": best["model"],
            "own_oracle_mean": round(float(best["score"]), 4),
            "n": int((sub["iteration"] == best["iteration"]).sum()),
        })

    filtered = scores_long[scores_long["model"].isin(set(keep_models))].copy()
    summary = pd.DataFrame(summary_rows).sort_values("arm").reset_index(drop=True)
    return filtered, summary
