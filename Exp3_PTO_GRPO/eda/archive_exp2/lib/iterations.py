"""
iterations.py — cross-iteration training-reward EDA for the Exp3 runs.

The rest of ``lib`` is single-snapshot (final-conversation quality across models).
This module fills the one gap that matters before resuming a run: **is the policy
actually climbing its own training reward, iteration over iteration?** It reads the
per-generation capture the trainers already write —
``runs/<MODE>/<EXP>/iteration_N/eda/generations.jsonl`` (one branch record per line,
candidates nested; schema in ``code/_shared/eda_recorder.py``) — and needs **no
oracle calls**: every candidate's Q1+Q2 ``score`` + per-questionnaire ``sub_scores``
were cached by the oracle at training time.

Four functions:
- :func:`load_generations` — explode the JSONL into one tidy row per candidate.
- :func:`aggregate_reward_by_iter` — per-iteration (optionally per-epoch) reward
  summary, including the "kept/trunk" reward (PTO chosen / GRPO group mean).
- :func:`scan_degeneracy` — per-iteration counts of ChatML leak / empty / floored
  completions (did the 2026-06-07 fixes hold in the real runs?).
- :func:`plot_reward_trajectory` — overlay the per-method reward curves.

Caveat baked into the interpretation: the completed iterations are all K=0 (LA0),
so ``score`` is the *short-cut* training reward with no look-ahead tail — exactly
the proxy ``Partial_Conv_Oracle_EDA`` flags as weak at short prefixes. This answers
"is the policy improving on its own training signal", NOT "is full-conversation MI
quality rising" (that's the Run_Eval / Conv_EDA path).
"""

import glob
import json
import os
import re
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EXPERIMENT_PALETTE, FIG_SINGLE, set_plot_style

# GRPO floors degenerate (self-played / empty) completions to this reward so they
# get a strongly negative group-relative advantage. Mirror REWARD_FLOOR in
# code/_shared/reward.py — a candidate score exactly here means "flagged degenerate".
REWARD_FLOOR = 0.0

_ITER_DIR_RE = re.compile(r"iteration_(\d+)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              LOADING                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _iter_from_path(path: str) -> Optional[int]:
    """Pull N out of ``.../iteration_N/eda/generations.jsonl`` (fallback only)."""
    m = _ITER_DIR_RE.search(path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def _explode_branch(rec: dict, run_label: str, fallback_iter: Optional[int]) -> List[dict]:
    """Flatten one branch record into one row per nested candidate."""
    base = {
        "run_label": run_label,
        "method": rec.get("method"),
        "iteration": rec.get("iteration", fallback_iter),
        "phase": rec.get("phase"),
        "conversation_id": rec.get("conversation_id"),
        "branch_id": rec.get("branch_id"),
        "epoch": rec.get("epoch"),            # GRPO only; None for PTO
        "group_mean": rec.get("group_mean"),  # GRPO only
        "group_std": rec.get("group_std"),    # GRPO only
        "chosen_idx": rec.get("chosen_idx"),
        "prefix": rec.get("prefix"),
    }
    rows = []
    for cand in rec.get("candidates", []) or []:
        oracle = cand.get("oracle") or {}
        la = cand.get("lookahead") or {}
        row = dict(base)
        row.update({
            "cand_idx": cand.get("idx"),
            "role": cand.get("role"),          # PTO only: chosen/rejected/neither
            "score": cand.get("score"),        # Q1+Q2 mean (None on oracle fail)
            "completion": cand.get("completion"),
            "oracle_success": bool(oracle.get("success")) if oracle else None,
            "oracle_retries": oracle.get("retries"),
            "lookahead_k": la.get("k"),
            "lookahead_realized_turns": la.get("realized_turns"),
            "lookahead_ended_early": la.get("ended_early"),
        })
        for qid, val in (cand.get("sub_scores") or {}).items():
            row[f"sub_score_{qid}"] = val
        rows.append(row)
    return rows


def load_generations(run_dir: str, *, label: Optional[str] = None) -> pd.DataFrame:
    """Load every ``iteration_*/eda/generations.jsonl`` under *run_dir*.

    Returns one row per candidate (branch ``prefix`` repeated across its candidates).
    Robust to the partial/empty JSONL a stopped run leaves behind (e.g. iter-4's
    0-byte file): blank lines and empty files contribute zero rows, not an error.

    Args:
        run_dir: a run root, i.e. ``data/<method>_Exp3/runs/<MODE>/<EXP_NAME>``.
        label: short name for this run in plots/tables (default: the run dir name).
    """
    label = label or os.path.basename(os.path.normpath(run_dir))
    pattern = os.path.join(run_dir, "iteration_*", "eda", "generations.jsonl")
    exploded: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        fallback_iter = _iter_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                exploded.extend(_explode_branch(json.loads(line), label, fallback_iter))
    if not exploded:
        return pd.DataFrame()
    df = pd.DataFrame(exploded)
    # epoch_idx = the discrete epoch (≈ per-checkpoint, since save_strategy="epoch").
    # GRPO's raw `epoch` is the continuous trainer_state.epoch (one value per optimizer
    # step); flooring collapses it to the 0,1,… epochs the "checkpoint" view wants.
    # PTO has no epoch (NaN) → stays NaN → falls back to iteration-level downstream.
    df["epoch_idx"] = np.floor(pd.to_numeric(df["epoch"], errors="coerce")).astype("Int64")
    return df.sort_values(["iteration", "branch_id", "cand_idx"]).reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            AGGREGATION                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def aggregate_reward_by_iter(df: pd.DataFrame, granularity: str = "iteration") -> pd.DataFrame:
    """Per-iteration reward summary, one row per ``(run_label, method, iteration)``.

    Columns: ``n_candidates``, ``n_scored``, ``mean_score``, ``std_score``,
    ``oracle_success_rate``, ``mean_group_std`` (GRPO), and ``mean_chosen_score`` —
    the "kept/best" reward: the mean over branches of the **argmax** (``chosen_idx``)
    candidate's score. For PTO that argmax is exactly the best-of-M the greedy trunk
    followed (``role=="chosen"``); for GRPO it's the best-of-G in each group (the
    natural analog — the all-candidate mean already equals ``group_mean``). It sits
    at/above ``mean_score`` by construction.

    Args:
        granularity: ``"iteration"`` (default) or ``"epoch"``. With ``"epoch"`` an
            ``epoch_idx`` group level is added (the discrete, floored epoch ≈
            per-checkpoint). Only GRPO records carry an epoch; PTO logs once at
            branch-build so its ``epoch_idx`` is NA and it collapses back to
            iteration-level (a note is printed).
    """
    if df.empty:
        return df
    if granularity not in ("iteration", "epoch"):
        raise ValueError(f"granularity must be 'iteration' or 'epoch', got {granularity!r}")

    keys = ["run_label", "method", "iteration"]
    if granularity == "epoch":
        for method, sub in df.groupby("method"):
            if sub["epoch_idx"].isna().all():
                print(f"[note] {method}: no per-epoch reward (epoch is null — logged once "
                      f"at branch-build); falling back to iteration-level for this method.")
        keys = ["run_label", "method", "iteration", "epoch_idx"]

    out_rows = []
    for key_vals, g in df.groupby(keys, dropna=False, sort=True):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        rec = dict(zip(keys, key_vals))
        scores = g["score"].dropna()
        rec["n_candidates"] = len(g)
        rec["n_scored"] = len(scores)
        rec["mean_score"] = scores.mean()
        rec["std_score"] = scores.std(ddof=0)
        rec["oracle_success_rate"] = (
            g["oracle_success"].mean() if g["oracle_success"].notna().any() else np.nan
        )
        gs = g["group_std"].dropna()
        rec["mean_group_std"] = gs.mean() if len(gs) else np.nan
        # kept/best reward = the argmax (chosen_idx) candidate per branch, averaged.
        chosen = g.loc[g["cand_idx"] == g["chosen_idx"], "score"].dropna()
        rec["mean_chosen_score"] = chosen.mean() if len(chosen) else np.nan
        out_rows.append(rec)

    return pd.DataFrame(out_rows).sort_values(keys).reset_index(drop=True)


def scan_degeneracy(df: pd.DataFrame, reward_floor: float = REWARD_FLOOR) -> pd.DataFrame:
    """Per-iteration counts of degenerate completions — did the leak fixes hold?

    Flags per candidate: residual ``<|im_start|>`` / ``<|im_end|>`` substrings (the
    ChatML self-play leak), empty/whitespace completion, and ``score == reward_floor``
    (GRPO's degenerate-completion floor). Healthy real runs should show ~0 of each.
    """
    if df.empty:
        return df
    comp = df["completion"].fillna("").astype(str)
    flags = pd.DataFrame({
        "run_label": df["run_label"].values,
        "method": df["method"].values,
        "iteration": df["iteration"].values,
        "im_start": comp.str.contains("<|im_start|>", regex=False).values,
        "im_end": comp.str.contains("<|im_end|>", regex=False).values,
        "empty": comp.str.strip().eq("").values,
        "floored": df["score"].eq(reward_floor).values,
    })
    g = (flags.groupby(["run_label", "method", "iteration"], sort=True)
              .agg(n_candidates=("im_start", "size"),
                   n_im_start=("im_start", "sum"),
                   n_im_end=("im_end", "sum"),
                   n_empty=("empty", "sum"),
                   n_floored=("floored", "sum"))
              .reset_index())
    return g


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              PLOTTING                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _trajectory_x(agg: pd.DataFrame) -> pd.Series:
    """X position per row: integer iteration, or iteration + within-iter epoch slot."""
    if "epoch_idx" not in agg.columns or agg["epoch_idx"].isna().all():
        return agg["iteration"].astype(float)
    x = agg["iteration"].astype(float).copy()
    # place per-epoch points just inside the iteration tick, ordered by epoch_idx.
    for _, sub in agg.groupby(["run_label", "method"]):
        ep = pd.to_numeric(sub["epoch_idx"], errors="coerce")
        ranks = ep.rank(method="dense")
        denom = (ranks.max() + 1) if pd.notna(ranks.max()) and ranks.max() > 1 else 2.0
        # NA epoch (PTO fallback) stays on the integer tick.
        offset = (ranks.fillna(0) / denom).where(ep.notna(), 0.0)
        x.loc[sub.index] = sub["iteration"].astype(float) + offset.to_numpy()
    return x


def plot_reward_trajectory(agg: pd.DataFrame, *, ax=None, value: str = "mean_score",
                           show_chosen: bool = True, title: Optional[str] = None):
    """Overlay per-run reward curves (x=iteration, y=mean reward, ±1σ band).

    The dashed line per run is ``mean_chosen_score`` (the kept/trunk reward) — for
    PTO it should sit at/above the all-candidate ``mean_score`` (the trunk follows
    the best-of-M). Pass an existing ``ax`` to compose into a larger figure.
    """
    if agg.empty:
        print("[plot_reward_trajectory] nothing to plot (empty aggregate).")
        return ax
    set_plot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=FIG_SINGLE)

    agg = agg.copy()
    agg["_x"] = _trajectory_x(agg)
    for (run_label, method), sub in agg.groupby(["run_label", "method"], sort=True):
        sub = sub.sort_values("_x")
        color = EXPERIMENT_PALETTE.get(method)
        ax.plot(sub["_x"], sub[value], marker="o", color=color, label=f"{run_label} ({method})")
        if "std_score" in sub:
            lo = sub[value] - sub["std_score"].fillna(0)
            hi = sub[value] + sub["std_score"].fillna(0)
            ax.fill_between(sub["_x"], lo, hi, color=color, alpha=0.15)
        if show_chosen and "mean_chosen_score" in sub and sub["mean_chosen_score"].notna().any():
            ax.plot(sub["_x"], sub["mean_chosen_score"], marker="x", linestyle="--",
                    color=color, alpha=0.7)

    # integer iteration ticks
    its = sorted(agg["iteration"].dropna().unique())
    if its:
        ax.set_xticks(its)
    ax.set_xlabel("iteration")
    ax.set_ylabel("oracle training reward (Q1+Q2 mean)")
    ax.set_title(title or "Training-reward trajectory per iteration")
    ax.legend(title="run", fontsize=9)
    note = "solid = mean over all candidates · dashed (×) = best-of-branch reward (argmax: PTO trunk-chosen / GRPO best-of-G)"
    ax.text(0.0, -0.18, note, transform=ax.transAxes, fontsize=8, color="0.4")
    return ax
