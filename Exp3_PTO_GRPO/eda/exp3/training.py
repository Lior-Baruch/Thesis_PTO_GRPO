"""
training.py — training-time signal: proxy reward + degeneration QC + PTO pref pairs.

Reads the per-generation capture ``runs/.../iteration_N/eda/generations.jsonl`` (one
branch row, candidates nested — schema in ``code/_shared/eda_recorder.py``) and the
PTO ``iteration_N/pref_pairs/pairs.csv``. NO oracle calls — every candidate's score
was cached at training time.

Iteration alignment (important for faithfulness): training ``iteration_N`` is policy
π_N's branching, the SAME policy that produced the ``model_iter_{N-1}`` eval convs.
So ``eval_iter = train_iter - 1`` joins proxy reward to full-conversation eval.
"""

import glob
import json
import os
from typing import List, Optional

import pandas as pd

REWARD_FLOOR = 0.0  # GRPO floors degenerate completions here (mirror reward.py)
_LEAK = "<|im_start|>"
_END = "<|im_end|>"


def _arm_runs(arms):
    from . import discover_arms
    return discover_arms() if arms is None else arms


def load_generations(arms: Optional[List] = None, *, keep_tail: bool = False) -> pd.DataFrame:
    """One tidy row per candidate across all arms' ``generations.jsonl``.

    Columns: arm, method, K, train_iter, eval_iter, phase, conversation_id, branch_id,
    epoch, group_mean, group_std, chosen_idx, cand_idx, role, score, q1, q2,
    realized_turns, ended_early, len_chars, is_chosen, leak/end/empty/floored flags,
    completion (+ tail if keep_tail).
    """
    rows = []
    for arm in _arm_runs(arms):
        for fp in sorted(glob.glob(os.path.join(arm.runs_dir, "iteration_*", "eda", "generations.jsonl"))):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ti = rec.get("iteration")
                    base = {
                        "arm": arm.label, "method": arm.method, "K": arm.K,
                        "train_iter": ti, "eval_iter": (ti - 1) if ti is not None else None,
                        "phase": rec.get("phase"), "conversation_id": rec.get("conversation_id"),
                        "branch_id": rec.get("branch_id"), "epoch": rec.get("epoch"),
                        "group_mean": rec.get("group_mean"), "group_std": rec.get("group_std"),
                        "chosen_idx": rec.get("chosen_idx"),
                    }
                    for c in rec.get("candidates", []):
                        comp = c.get("completion") or ""
                        sub = c.get("sub_scores") or {}
                        la = c.get("lookahead") or {}
                        score = c.get("score")
                        row = {
                            **base, "cand_idx": c.get("idx"), "role": c.get("role"),
                            "score": score,
                            "q1": _num(sub.get("1")), "q2": _num(sub.get("2")),
                            "realized_turns": la.get("realized_turns"),
                            "ended_early": la.get("ended_early"),
                            "len_chars": len(comp),
                            "is_chosen": (c.get("idx") == rec.get("chosen_idx")),
                            "leak": _LEAK in comp, "has_end": _END in comp,
                            "empty": (len(comp.strip()) == 0),
                            "floored": (score is not None and float(score) <= REWARD_FLOOR),
                            "completion": comp,
                        }
                        if keep_tail:
                            row["tail"] = la.get("tail")
                        rows.append(row)
    return pd.DataFrame(rows)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def scan_degeneracy(gens: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, train_iter): candidate counts + degeneration rates (leak/empty/floored).

    Confirms the 2026-06-07 ChatML-leak + stop-string fixes held in the real runs.
    """
    if gens.empty:
        return gens
    g = gens.groupby(["arm", "train_iter"], observed=True)
    out = g.agg(
        n_candidates=("score", "size"),
        n_leak=("leak", "sum"), n_empty=("empty", "sum"), n_floored=("floored", "sum"),
        mean_score=("score", "mean"), mean_len=("len_chars", "mean"),
    ).reset_index()
    for c in ("leak", "empty", "floored"):
        out[f"pct_{c}"] = (100 * out[f"n_{c}"] / out["n_candidates"]).round(2)
    return out


def load_pref_pairs(arms: Optional[List] = None) -> pd.DataFrame:
    """PTO ``pref_pairs/pairs.csv`` across iterations (one row per emitted pair).

    Adds ``arm``, ``train_iter``, ``eval_iter``, and ``margin`` = chosen−rejected score.
    Returns empty for GRPO arms (no preference data). ``branch_depth`` is the depth in
    the greedy trunk where the pair was emitted.
    """
    rows = []
    for arm in _arm_runs(arms):
        if arm.method != "PTO":
            continue
        for fp in sorted(glob.glob(os.path.join(arm.runs_dir, "iteration_*", "pref_pairs", "pairs.csv"))):
            ti = _iter_from(fp)
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            df["arm"] = arm.label
            df["train_iter"] = ti
            df["eval_iter"] = (ti - 1) if ti is not None else None
            if {"chosen_score", "rejected_score"}.issubset(df.columns):
                df["margin"] = df["chosen_score"] - df["rejected_score"]
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _iter_from(path: str) -> Optional[int]:
    import re
    m = re.search(r"iteration_(\d+)", path.replace("\\", "/"))
    return int(m.group(1)) if m else None
