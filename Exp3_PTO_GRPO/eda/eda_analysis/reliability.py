"""reliability.py — MEASUREMENT-VALIDITY data layer: reads the re-scoring tree written by
``Judge_Reliability.ipynb`` (``data/judge_check/``) and builds the tables the EDA displays.

Analysis-layer counterpart to :mod:`eda_analysis.scoring.judge`, which OWNS the paid API path that
*writes* that tree. **Nothing here calls an API** — this module is disk-only and free to re-run,
like every other ``eda_analysis`` module, so ``5_Training_and_Reliability`` can render these
tables/figures inside ``render_views.py`` while the money stays behind the ``RUN_*`` switches in
``Judge_Reliability.ipynb``. Same split as ``Run_Eval`` (paid, manual) → notebooks 1–7 (free, auto).

Two questions, from the re-scoring subset (anchor models × Q1/Q2/MICI × 96 convs):

1. **Repeatability** — the primary oracle re-scoring the SAME conversations N times (seeds differ,
   nothing else) → ICC(2,1) + mean |Δ|. This is the measurement error of the instrument.
2. **Second-judge agreement** — a different-family judge (Claude Haiku 4.5) scoring the same cells
   once → r / ρ / bias vs the primary oracle, and the defense-critical check: does the PTO−GRPO
   endpoint contrast keep its SIGN under a judge that never played the patient?

Read agreement against the **attenuation ceiling**, never against 1.0: two noisy raters cannot
correlate above ``sqrt(ICC_a × ICC_b)``. The subset scores the second judge once, so ``ICC_judge``
is unmeasured; :func:`agreement` therefore reports the ceiling under the assumption that the second
judge is as self-consistent as the primary (which makes the ceiling simply ``ICC_primary``) and
flags it as an upper bound — if Haiku is noisier, the true ceiling is lower and the observed
agreement is correspondingly better than it looks.

Figures: :mod:`eda_analysis.plotting.reliability`. Metric definitions: ``METRICS_REFERENCE.md`` §7.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .scoring import judge as _judge
from .scoring import registry as _registry

JUDGE_CHECK_ROOT = _judge.JUDGE_CHECK_ROOT
PRIMARY_TAG = _judge.PRIMARY_JUDGE.tag

# The endpoint contrasts worth checking for judge-independence. (a − b); for MICI lower = better,
# so a NEGATIVE delta on the first pair is the "PTO is less MI-inconsistent" direction.
DEFAULT_CONTRAST_PAIRS: List[Tuple[str, str]] = [
    ("PTOExp3_LA0_I10", "GRPOExp3_LA0_I10"),   # THE headline: matched 10-iter endpoint
    ("PTOExp3_LA0_I10", "PTOExp3_LA0_Base"),   # did PTO improve on base at all?
]


# ── discovery / guards ────────────────────────────────────────────────────────
def judge_tags() -> List[str]:
    """Judge folders present under ``data/judge_check/`` (``summary/`` excluded)."""
    if not os.path.isdir(JUDGE_CHECK_ROOT):
        return []
    return sorted(d for d in os.listdir(JUDGE_CHECK_ROOT)
                  if d != "summary" and os.path.isdir(os.path.join(JUDGE_CHECK_ROOT, d)))


def second_judge_tags() -> List[str]:
    """Judge tags other than the primary oracle (i.e. the decoupled second judges)."""
    return [t for t in judge_tags() if t != PRIMARY_TAG]


def available() -> bool:
    """True when any re-scoring data exists — notebooks guard their section on this."""
    return bool(judge_tags())


# Readable names for figure titles / captions (keys are the MODEL half of a judge tag).
_JUDGE_DISPLAY = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-4-8": "Claude Opus 4.8",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
}


def judge_display(tag: str) -> str:
    """``anthropic_claude-haiku-4-5`` -> ``Claude Haiku 4.5`` (unknown models pass through)."""
    model = str(tag).split("_", 1)[-1]
    return _JUDGE_DISPLAY.get(model, model)


# ── loading ───────────────────────────────────────────────────────────────────
def load_judge_long(tag: str, *, reps: Optional[List[int]] = None) -> pd.DataFrame:
    """Tidy ``(judge, rep, metric, oracle, model, file_index, value)`` for one judge tag."""
    return _judge.load_judge_scores(tag, reps=reps)


def load_primary_long(models: Sequence[str], metrics: Sequence[str],
                      layout: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """The PRODUCTION scores for the same cells, read from the real ``eval_scores/`` tree.

    This is the comparison baseline for the second judge — the numbers the thesis actually
    reports — not a re-score. ``layout`` defaults to the auto-discovered registry layout.
    """
    if layout is None:
        exps = [e for e in _registry.EXPERIMENTS if e.model_name in set(models)]
        layout = _registry.get_model_eval_layout(exps)
    rows = []
    for model in models:
        entry = layout.get(model)
        if entry is None:
            continue
        for name in metrics:
            subdir, col = _judge.JUDGE_METRIC_COLS[name]
            ddir = _registry.eval_csv_dir(entry["root"], entry["oracle"], subdir, model)
            if not os.path.isdir(ddir):
                continue
            for fn in os.listdir(ddir):
                stem, ext = os.path.splitext(fn)
                if ext != ".csv" or not stem.isdigit():
                    continue
                try:
                    df = pd.read_csv(os.path.join(ddir, fn))
                except Exception:
                    continue
                if len(df) and col in df.columns:
                    rows.append({"metric": name, "model": model, "file_index": int(stem),
                                 "value": float(df[col].iloc[0])})
    return pd.DataFrame(rows)


# ── tables ────────────────────────────────────────────────────────────────────
def repeatability(judge_long: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per (metric, model): ICC(2,1) across reps, mean |Δ| between rep pairs, n.

    Koo & Li (2016) reading: ICC ≥ 0.75 good, ≥ 0.90 excellent. ``mean_abs_diff`` is the typical
    per-conversation swing between two scorings — the number to quote as "oracle noise".
    Returns an empty frame when fewer than 2 reps exist.
    """
    if judge_long is None:
        judge_long = load_judge_long(PRIMARY_TAG)
    if judge_long.empty or judge_long.rep.nunique() < 2:
        return pd.DataFrame()
    return _judge.repeatability_table(judge_long)


def repeatability_by_metric(rep_tab: pd.DataFrame) -> pd.DataFrame:
    """Collapse :func:`repeatability` to one row per metric (mean over models) — the citable line."""
    if rep_tab is None or rep_tab.empty:
        return pd.DataFrame()
    return (rep_tab.groupby("metric")
            .agg(n_models=("model", "nunique"), n_convs=("n_convs", "sum"),
                 icc_2_1=("icc_2_1", "mean"), mean_abs_diff=("mean_abs_diff", "mean"))
            .round(3).reset_index())


def agreement(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
              rep_tab: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Second judge vs primary oracle per (metric, model): r, ρ, bias, + the attenuation ceiling.

    ``bias_judge_minus_primary`` is the LEVEL offset (a harsher judge scores everything lower);
    it is irrelevant to the thesis claims, which are all *contrasts* — see :func:`contrasts`.
    ``ceiling`` is the max r two raters of this reliability could reach (see the module docstring:
    an upper bound, since the second judge's own ICC is unmeasured); ``r_pct_of_ceiling`` is the
    observed agreement as a share of it.
    """
    tab = _judge.agreement_table(judge_long, primary_long)
    if tab.empty:
        return tab
    if rep_tab is None:
        rep_tab = repeatability()
    if rep_tab is not None and not rep_tab.empty:
        icc = rep_tab.set_index(["metric", "model"])["icc_2_1"]
        tab["icc_primary"] = [icc.get((m, mo), np.nan) for m, mo in zip(tab.metric, tab.model)]
        # sqrt(ICC_p * ICC_j) with ICC_j assumed == ICC_p  ->  the ceiling collapses to ICC_p.
        tab["ceiling"] = tab["icc_primary"].round(3)
        tab["r_pct_of_ceiling"] = (100 * tab["pearson_r"] / tab["ceiling"]).round(1)
    return tab


def contrasts(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
              metrics: Sequence[str], pairs: Sequence[Tuple[str, str]] = None) -> pd.DataFrame:
    """THE defense check: does each endpoint contrast keep its SIGN under the second judge?

    One row per (pair, metric) with both judges' paired deltas and ``same_sign``. Pairing is by
    ``file_index``, which is persona-valid only at matched iterations (same ``seed+k+1`` shuffle) —
    :data:`DEFAULT_CONTRAST_PAIRS` respects that.
    """
    pairs = DEFAULT_CONTRAST_PAIRS if pairs is None else pairs
    rows = [_judge.contrast_preservation(judge_long, primary_long, a, b, m)
            for a, b in pairs for m in metrics]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["contrast"] = df.model_a.str.replace("Exp3", "", regex=False) + " − " \
            + df.model_b.str.replace("Exp3", "", regex=False)
    return df


def arm_means_by_judge(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                       judge_name: str = "second judge") -> pd.DataFrame:
    """Wide ``model × (metric, judge)`` means — the level-shift view behind the bias column."""
    if judge_long.empty:
        return pd.DataFrame()
    both = pd.concat([
        judge_long.assign(judge=judge_name)[["judge", "metric", "model", "value"]],
        primary_long.assign(judge="primary")[["judge", "metric", "model", "value"]],
    ])
    return both.pivot_table(index="model", columns=["metric", "judge"], values="value").round(3)


def summary_line(rep_tab: pd.DataFrame, agr_tab: pd.DataFrame, con_tab: pd.DataFrame) -> str:
    """One-sentence verdict for the notebook/SUMMARY — safe on empty inputs."""
    bits = []
    if rep_tab is not None and not rep_tab.empty:
        bits.append(f"oracle ICC(2,1) {rep_tab.icc_2_1.min():.2f}–{rep_tab.icc_2_1.max():.2f} "
                    f"(mean |Δ| {rep_tab.mean_abs_diff.mean():.2f})")
    if agr_tab is not None and not agr_tab.empty:
        q = agr_tab[agr_tab.metric.isin(["Q1", "Q2"])]
        if not q.empty:
            bits.append(f"Q1/Q2 cross-judge r {q.pearson_r.min():.2f}–{q.pearson_r.max():.2f}")
    if con_tab is not None and not con_tab.empty:
        bits.append(f"{int(con_tab.same_sign.sum())}/{len(con_tab)} contrasts keep their sign")
    return "; ".join(bits) if bits else "no re-scoring data on disk"
