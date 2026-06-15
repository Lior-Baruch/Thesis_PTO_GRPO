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
import re
from typing import List, Optional

import numpy as np
import pandas as pd

REWARD_FLOOR = 0.0  # GRPO floors degenerate completions here (mirror reward.py)
_LEAK = "<|im_start|>"
_END = "<|im_end|>"
_ROLE_RE = re.compile(r"\[(?:THERAPIST|PATIENT)\]:")  # oracle-transcript turn markers


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


def load_branch_reliability(arms: Optional[List] = None, *, which: str = "chosen") -> pd.DataFrame:
    """Per-branch proxy score + partial-conversation length — the data ``load_generations`` drops.

    Re-reads ``generations.jsonl`` keeping each branch's ``prefix`` (the oracle-format transcript of
    the conversation-so-far) and counts its turns (``[THERAPIST]:`` / ``[PATIENT]:`` markers). This
    is what lets us rebuild the Exp2 partial-conv reliability curve for Exp3 — **no new oracle calls**.

    ``which`` selects the per-branch proxy: ``"chosen"`` (the candidate the policy kept; the trajectory
    it actually took), ``"max"``, or ``"mean"`` over candidates.
    Columns: ``arm, method, K, train_iter, eval_iter, conversation_id, n_turns, proxy_score``.
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
                    prefix = rec.get("prefix") or ""
                    n_turns = len(_ROLE_RE.findall(prefix))
                    cands = rec.get("candidates", []) or []
                    scored = [(c.get("idx"), _num(c.get("score"))) for c in cands]
                    scored = [(i, s) for i, s in scored if s is not None]
                    if not scored or n_turns == 0:
                        continue
                    if which == "chosen":
                        ci = rec.get("chosen_idx")
                        proxy = next((s for i, s in scored if i == ci), max(s for _, s in scored))
                    elif which == "max":
                        proxy = max(s for _, s in scored)
                    else:
                        proxy = float(np.mean([s for _, s in scored]))
                    ti = rec.get("iteration")
                    rows.append({"arm": arm.label, "method": arm.method, "K": arm.K,
                                 "train_iter": ti, "eval_iter": (ti - 1) if ti is not None else None,
                                 "conversation_id": rec.get("conversation_id"),
                                 "n_turns": int(n_turns), "proxy_score": float(proxy)})
    return pd.DataFrame(rows)


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


# ── Symmetric training-internals (both methods, one frame) ───────────────────────
def reward_distribution_frame(arms: Optional[List] = None) -> pd.DataFrame:
    """Per-candidate training reward with ``(arm, method, train_iter, score)`` for ALL arms.

    The tidy backbone for a side-by-side PTO-vs-GRPO reward-distribution plot — both methods
    log every candidate's score in ``generations.jsonl``, so this just selects those columns.
    """
    g = load_generations(arms)
    if g.empty:
        return pd.DataFrame(columns=["arm", "method", "train_iter", "score"])
    return g[["arm", "method", "train_iter", "score"]].dropna(subset=["score"])


def advantage_signal_by_iter(arms: Optional[List] = None) -> pd.DataFrame:
    """Unified per-(arm, train_iter) training advantage signal for BOTH methods.

    One tidy frame so a single plot can render both methods without an ``if`` — the two
    methods populate different (method-native) columns, NaN elsewhere:

    - **GRPO** (from ``generations.jsonl``, one group per branch): ``group_std`` (mean within-
      group reward spread = the implicit advantage signal) + ``frac_zero_std`` (fraction of
      near-collapsed groups, a degeneracy red flag).
    - **PTO** (from ``pref_pairs/pairs.csv``): ``margin`` (mean chosen−rejected oracle-score
      gap = how decisive the τ-filtered pairs are), ``margin_median``, ``n_pairs``.

    Columns: ``arm, method, train_iter, group_std, frac_zero_std, margin, margin_median, n_pairs``.
    Empty for arms with no training capture on disk (e.g. GRPO_LA5 has no generations.jsonl).
    """
    rows = []
    gens = load_generations(arms)
    if not gens.empty:
        grp = gens[gens["method"] == "GRPO"]
        if not grp.empty:
            # one group_std per branch (it's repeated across the group's candidates).
            per_branch = grp.dropna(subset=["group_std"]).drop_duplicates(
                ["arm", "train_iter", "branch_id"])
            for (arm, ti), g in per_branch.groupby(["arm", "train_iter"], observed=True):
                rows.append({"arm": arm, "method": "GRPO", "train_iter": int(ti),
                             "group_std": float(g["group_std"].mean()),
                             "frac_zero_std": float((g["group_std"] < 1e-6).mean()),
                             "margin": None, "margin_median": None, "n_pairs": None})
    pairs = load_pref_pairs(arms)
    if not pairs.empty and "margin" in pairs.columns:
        for (arm, ti), g in pairs.groupby(["arm", "train_iter"], observed=True):
            rows.append({"arm": arm, "method": "PTO", "train_iter": int(ti),
                         "group_std": None, "frac_zero_std": None,
                         "margin": float(g["margin"].mean()),
                         "margin_median": float(g["margin"].median()),
                         "n_pairs": int(len(g))})
    cols = ["arm", "method", "train_iter", "group_std", "frac_zero_std",
            "margin", "margin_median", "n_pairs"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["arm", "train_iter"]).reset_index(drop=True)


# ── TensorBoard training curves (the wandb-style graphs) ─────────────────────────
# Self-contained tensorboard parse (no torch/trl/wandb import) so the EDA stays
# host-agnostic and dodges the local trl-before-torch segfault that importing the
# trainers' _shared.tb_plots would trigger.
_ITER_PATH_RE = re.compile(r"iteration_(\d+)")
_TB_PRIORITY = [   # plotted if present, in this order (GRPO + DPO tags)
    "train/loss", "train/reward", "train/reward_std", "train/rewards/margins",
    "train/rewards/accuracies", "train/kl", "train/entropy",
    "train/completions/mean_length", "train/learning_rate",
]


def parse_run_tb(run_dir: str):
    """Parse a run's per-iteration TB event files → ``({tag: DataFrame[step,value]}, boundaries)``.

    Steps are chained across iterations (each trainer restarts at step 0) so curves are continuous;
    ``boundaries`` = ``[(iter, cumulative_step_end), ...]`` for drawing iteration separators.
    Returns ``({}, [])`` if tensorboard isn't installed or no event files are found.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except Exception as e:
        print(f"  [tb] tensorboard not available ({e}) — skipping training curves")
        return {}, []
    files = glob.glob(os.path.join(run_dir, "iteration_*", "**", "events.out.tfevents.*"), recursive=True)
    by_iter = {}
    for fp in files:
        m = _ITER_PATH_RE.search(fp.replace("\\", "/"))
        if m:
            by_iter.setdefault(int(m.group(1)), []).append(fp)
    series, boundaries, offset = {}, [], 0
    for it in sorted(by_iter):
        it_max = 0
        for fp in sorted(by_iter[it]):
            acc = ea.EventAccumulator(fp, size_guidance={ea.SCALARS: 0})
            try:
                acc.Reload()
            except Exception:
                continue
            for tag in acc.Tags().get("scalars", []):
                for e in acc.Scalars(tag):
                    series.setdefault(tag, []).append((e.step + offset, e.value))
                    it_max = max(it_max, e.step)
        offset += it_max
        boundaries.append((it, offset))
    out = {}
    for tag, pts in series.items():
        d = dict(pts)  # last value wins per (chained) step
        xs = sorted(d)
        out[tag] = pd.DataFrame({"step": xs, "value": [d[x] for x in xs]})
    return out, boundaries


def tb_curves(arm, *, tags: Optional[List[str]] = None, smooth: int = 1):
    """Plot one arm's TensorBoard training curves (the graphs seen on wandb/TB), chained across iters.

    Curated salient tags by default (loss, reward/reward_std or rewards/margins+accuracies, KL,
    entropy, completion length, lr — whichever the method emitted). Dotted vlines = iteration
    boundaries. Returns a fig, or ``None`` if no logs/tensorboard (degrades cleanly).
    """
    import matplotlib.pyplot as plt
    series, bounds = parse_run_tb(arm.runs_dir)
    if not series:
        print(f"  [tb] no event files for {arm.label} under {arm.runs_dir}")
        return None
    want = [t for t in (tags or _TB_PRIORITY) if t in series] or sorted(series)[:9]
    ncols = 3
    nrows = int(np.ceil(len(want) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.0 * nrows), squeeze=False)
    axflat = list(axes.flat)
    vlines = [b for (_it, b) in bounds[:-1]]
    for ax, tag in zip(axflat, want):
        df = series[tag]
        y = df["value"].rolling(smooth, min_periods=1).mean() if smooth > 1 else df["value"]
        ax.plot(df["step"], y, lw=1.3)
        for b in vlines:
            ax.axvline(b, color="grey", lw=0.5, ls=":")
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("step (chained across iters)")
    for ax in axflat[len(want):]:
        ax.set_visible(False)
    fig.suptitle(f"{arm.label} — training curves (TensorBoard)", y=1.0, fontweight="bold")
    fig.tight_layout()
    return fig
