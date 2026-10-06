"""
training.py — training-time signal: proxy reward + degeneration QC + PTO pref pairs.

Reads the per-generation capture ``runs/.../iteration_N/eda/generations.jsonl`` (one
branch row, candidates nested — schema in ``code/_shared/eda_recorder.py``) and the
PTO ``iteration_N/pref_pairs/pairs.csv``. NO oracle calls — every candidate's score
was cached at training time.

Iteration alignment (important for faithfulness): training ``iteration_N`` is policy
π_N's branching, the SAME policy that produced the ``model_iter_{N-1}`` eval convs.
So ``eval_iter = train_iter - 1`` joins proxy reward to full-conversation eval.

Trainer diagnostics (``trainer_log_steps`` → ``trainer_diagnostics_*``) read what the TRL trainer
itself logged per optimizer step (KL to the iteration-start policy, gradient norm, entropy, cap
hits, …) from each iteration's LATEST ``checkpoint-*/trainer_state.json`` — never the TensorBoard
events, which duplicate resumed steps and lost the first 30 steps of three iterations.
"""

import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REWARD_FLOOR = 0.0  # GRPO floors degenerate completions here (mirror reward.py)
_LEAK = "<|im_start|>"
_END = "<|im_end|>"
_ROLE_RE = re.compile(r"\[(?:THERAPIST|PATIENT)\]:")  # oracle-transcript turn markers


def _arm_runs(arms):
    from . import discover_arms
    return discover_arms() if arms is None else arms


# ── generations memo ───────────────────────────────────────────────────────────
# `generations.jsonl` is 557 MB across 29 files and EVERY training-side quantity starts from it:
# reward_distribution_frame, scan_degeneracy, load_branch_reliability, pref.load_weighted_candidates
# (twice, for the two drop_zero_weight settings) and the notebooks' own direct calls. Measured
# 2026-08-13 that is 5+ full re-parses of the same bytes per render — ~7.4 s each, ~30 s per unit.
#
# A process-level memo, not `.eda_cache`: the frame is 130 MB with free-text completions, so a
# parquet round-trip would cost more than the parse it saves, and the lifetime that matters is one
# notebook run. Under pandas Copy-on-Write (3.0.3 here) the returned `.copy()` is O(1) and mutation
# by one caller cannot reach another's frame, so handing out copies is both safe and free.
_GENERATIONS_MEMO: dict = {}


def _arm_memo_key(arms) -> str:
    """Stable key for an arm subset — mirrors ``data.load_cached``'s arm signature."""
    return "|".join(f"{a.exp_name}:{','.join(map(str, a.iters))}"
                    for a in sorted(_arm_runs(arms), key=lambda a: a.exp_name))


def clear_generations_memo() -> None:
    """Drop the in-process ``generations.jsonl`` memo (after a training run writes new rows)."""
    _GENERATIONS_MEMO.clear()


def load_generations(arms: Optional[List] = None, *, keep_tail: bool = False) -> pd.DataFrame:
    """One tidy row per candidate across all arms' ``generations.jsonl``.

    Columns: arm, method, K, train_iter, eval_iter, phase, conversation_id, branch_id,
    epoch, group_mean, group_std, chosen_idx, cand_idx, role, score, q1, q2,
    realized_turns, ended_early, len_chars, is_chosen, leak/end/empty/floored flags,
    completion (+ tail if keep_tail).

    Memoized per (arm subset, keep_tail) for the life of the process — see the memo note above.
    """
    memo_key = (_arm_memo_key(arms), keep_tail)
    hit = _GENERATIONS_MEMO.get(memo_key)
    if hit is not None:
        return hit.copy()

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
    df = pd.DataFrame(rows)
    _GENERATIONS_MEMO[memo_key] = df
    return df.copy()


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

    Parquet-cached on the RUN artifacts (``generations.jsonl`` et al., so ``exts`` must be
    ``RUN_SIGNATURE_EXTS`` — the default ``.csv`` would watch files this never reads and go stale
    without a miss). The frame is a few thousand rows; the parse behind it is ~17 s of an 879 MB
    tree, paid once per process before this and now once per content change.
    """
    from .data import load_cached, runs_input_roots, RUN_SIGNATURE_EXTS
    arms_l = _arm_runs(arms)
    return load_cached("branch_reliability", arms_l,
                       lambda: _load_branch_reliability_impl(arms_l, which=which),
                       input_roots=runs_input_roots(arms_l), params={"which": which},
                       exts=RUN_SIGNATURE_EXTS)


def _load_branch_reliability_impl(arms, *, which: str = "chosen") -> pd.DataFrame:
    rows = []
    for arm in arms:
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

    - **GRPO** (from ``generations.jsonl``, one group per branch): ``group_range`` (mean per-group
      **best − worst** reward = the direct analog to PTO's chosen−rejected margin, on the SAME 0–5
      oracle-score scale), ``group_std`` (mean within-group spread = the implicit advantage magnitude),
      and ``frac_zero_std`` (fraction of near-collapsed groups, a degeneracy red flag).
    - **PTO** (from ``pref_pairs/pairs.csv``): ``margin`` (mean chosen−rejected oracle-score
      gap = how decisive the τ-filtered pairs are), ``margin_median``, ``n_pairs``.

    ``group_range`` (GRPO best−worst) and ``margin`` (PTO chosen−rejected) are both oracle-score gaps,
    so they give the two methods a COMPARABLE decisiveness signal (unlike ``group_std``, a different
    scale). Columns: ``arm, method, train_iter, group_std, group_range, frac_zero_std, margin,
    margin_median, n_pairs``. Empty for arms with no training capture on disk (e.g. GRPO_LA5).
    """
    rows = []
    gens = load_generations(arms)
    pto_range: dict = {}   # (arm, train_iter) -> mean per-branch best−worst range (UNFILTERED)
    if not gens.empty:
        grp = gens[gens["method"] == "GRPO"].dropna(subset=["score"])
        if not grp.empty:
            # Per branch: the recorded group_std (repeated across candidates → first) AND the
            # best−worst reward RANGE computed from the group's own candidate scores (the margin analog).
            # KEY includes conversation_id: GRPO's branch_id is globally unique per iteration so this is
            # a no-op for GRPO, but it is REQUIRED for PTO below (whose branch_id is the trunk depth and
            # collides across conversations — see pto_trainer.py grow_preference_trees_batch).
            per_branch = grp.groupby(["arm", "train_iter", "conversation_id", "branch_id"], observed=True).agg(
                group_std=("group_std", "first"),
                group_range=("score", lambda s: float(s.max() - s.min())),
            ).reset_index()
            for (arm, ti), g in per_branch.groupby(["arm", "train_iter"], observed=True):
                std = g["group_std"].dropna()
                rows.append({"arm": arm, "method": "GRPO", "train_iter": int(ti),
                             "group_std": float(std.mean()) if len(std) else None,
                             "group_range": float(g["group_range"].mean()),
                             "frac_zero_std": float((std < 1e-6).mean()) if len(std) else None,
                             "margin": None, "margin_median": None, "n_pairs": None})
        # PTO: the UNFILTERED per-branch best−worst range over the branch's own M candidate scores —
        # the true like-for-like analog to GRPO group_range. NOTE this differs from `margin` below,
        # which is the DPO chosen−rejected gap read off τ-filtered pairs.csv (left-truncated at τ);
        # the range is over ALL candidates the oracle scored at that branch point, τ-free.
        # CRITICAL: group by conversation_id TOO — PTO branch_id is the trunk depth and repeats across
        # conversations, so omitting conversation_id would pool candidates from different conversations
        # into one "branch" and report the cross-conversation spread (huge), not the within-branch spread.
        pto = gens[gens["method"] == "PTO"].dropna(subset=["score"])
        if not pto.empty:
            pb = (pto.groupby(["arm", "train_iter", "conversation_id", "branch_id"], observed=True)["score"]
                  .agg(lambda s: float(s.max() - s.min())).reset_index(name="rng"))
            for (arm, ti), g in pb.groupby(["arm", "train_iter"], observed=True):
                pto_range[(arm, int(ti))] = float(g["rng"].mean())
    pairs = load_pref_pairs(arms)
    if not pairs.empty and "margin" in pairs.columns:
        for (arm, ti), g in pairs.groupby(["arm", "train_iter"], observed=True):
            rows.append({"arm": arm, "method": "PTO", "train_iter": int(ti),
                         "group_std": None, "group_range": pto_range.get((arm, int(ti))),
                         "frac_zero_std": None,
                         "margin": float(g["margin"].mean()),
                         "margin_median": float(g["margin"].median()),
                         "n_pairs": int(len(g))})
    cols = ["arm", "method", "train_iter", "group_std", "group_range", "frac_zero_std",
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


# ── Trainer diagnostics: what the TRL trainer logged per optimizer step ─────────────────────────────────
# Source = the LATEST checkpoint's trainer_state.json of each iteration (iteration_N/training/checkpoint-*/),
# NOT the TensorBoard events. Measured 2026-10-06 on the two GRPO runs: trainer_state holds exactly one row
# per optimizer step (1,128 = 1,128 for K=0, 1,070 = 1,070 for K=5), and on the 2,108 steps TB holds
# (2,198 - 3 x 30) all nine compared scalars (kl, grad_norm, entropy, frac_reward_zero_std, clipped_ratio,
# mean_length, reward, reward_std, learning_rate) equal TB's LAST-written value for that step to float32
# precision (relative difference <= 6e-8). TB itself holds duplicate rows in the five resumed iterations whose crash
# came after a checkpoint (K=0 it 2; K=5 it 1, 7, 9, 10: the steps between the checkpoint and the crash,
# first attempt + redo) and has NO rows for steps 1-30 in three (K=0 it 6, 8; K=5 it 2), whose pre-crash
# event file holds no scalars. A TB mean is therefore biased in exactly the resumed iterations; this reader
# never touches TB.
_CKPT_RE = re.compile(r"checkpoint-(\d+)$")

TRAINER_DIAG_KEYS: Dict[str, str] = {   # short column -> TRL 1.4 log_history key (GRPOTrainer)
    "kl": "kl",
    "grad_norm": "grad_norm",
    "entropy": "entropy",
    "frac_reward_zero_std": "frac_reward_zero_std",
    "clipped_ratio": "completions/clipped_ratio",
    "mean_length": "completions/mean_length",
    "reward": "reward",
    "reward_std": "reward_std",
    "learning_rate": "learning_rate",
    "clip_region": "clip_ratio/region_mean",
    "loss": "loss",
    "lookahead_turns": "lookahead/realized_turns_mean",
    "oracle_success": "oracle/success_rate",
}
# The K-contrast Holm family: logged quantities defined identically in both runs. reward / reward_std are
# NOT in it (the K=5 reward scores the completion plus a 5-turn rollout, so the two runs' rewards are
# different quantities), nor learning_rate (the same deterministic schedule in both).
TRAINER_DIAG_FAMILY = ["kl", "grad_norm", "entropy", "frac_reward_zero_std", "clipped_ratio", "mean_length"]


def _latest_trainer_state(iter_dir: str):
    """``(checkpoint_step, all_checkpoint_steps, state_dict)`` of the latest checkpoint with a trainer_state.json."""
    cks = []
    for c in glob.glob(os.path.join(iter_dir, "training", "checkpoint-*")):
        m = _CKPT_RE.search(c.replace("\\", "/"))
        if m and os.path.exists(os.path.join(c, "trainer_state.json")):
            cks.append((int(m.group(1)), c))
    if not cks:
        return None, [], None
    step, path = max(cks)
    with open(os.path.join(path, "trainer_state.json"), encoding="utf-8") as fh:
        return step, sorted(s for s, _ in cks), json.load(fh)


def _trainer_logs(arms):
    steps, cov = [], []
    for a in _arm_runs(arms):
        for it_dir in sorted(glob.glob(os.path.join(a.runs_dir, "iteration_*"))):
            m = re.search(r"iteration_(\d+)$", it_dir.replace("\\", "/"))
            if not m:
                continue
            it = int(m.group(1))
            ck, all_ck, st = _latest_trainer_state(it_dir)
            base = {"arm": a.label, "method": a.method, "K": a.K, "iteration": it}
            if st is None:
                cov.append({**base, "checkpoint": None, "max_steps": np.nan, "n_rows": 0, "n_steps": 0,
                            "n_duplicate_rows": 0, "n_missing_steps": np.nan, "coverage": 0.0})
                continue
            max_steps = int(st.get("max_steps") or 0)
            # train rows carry "loss"; eval rows carry eval_* keys; the end-of-training summary carries train_runtime
            rows = [r for r in st.get("log_history", []) if "loss" in r and "train_runtime" not in r]
            df = pd.DataFrame(rows)
            n_rows = len(df)
            if n_rows:
                df = df.drop_duplicates("step", keep="last")   # defensive: one row per optimizer step
            uniq = set(df["step"].astype(int)) if n_rows else set()
            cov.append({**base, "checkpoint": ck, "checkpoints_on_disk": ",".join(map(str, all_ck)),
                        "max_steps": max_steps, "global_step": st.get("global_step"), "n_rows": n_rows,
                        "n_steps": len(uniq), "n_duplicate_rows": n_rows - len(uniq),
                        "n_missing_steps": len(set(range(1, max_steps + 1)) - uniq),
                        "coverage": len(uniq) / max_steps if max_steps else np.nan})
            if n_rows:
                df = df.rename(columns={v: k for k, v in TRAINER_DIAG_KEYS.items()})
                df.insert(0, "max_steps", max_steps)
                for k, v in reversed(list(base.items())):
                    df.insert(0, k, v)
                steps.append(df)
    s = pd.concat(steps, ignore_index=True) if steps else pd.DataFrame()
    if not s.empty:
        s["step"] = s["step"].astype(int)
        s["frac_of_iteration"] = s["step"] / s["max_steps"]
        s = s.sort_values(["arm", "iteration", "step"]).reset_index(drop=True)
    c = pd.DataFrame(cov)
    if not c.empty:
        c = c.sort_values(["arm", "iteration"]).reset_index(drop=True)
    return s, c


def trainer_log_steps(arms: Optional[List] = None) -> pd.DataFrame:
    """One row per (arm, iteration, optimizer step): every scalar the trainer logged, from ``trainer_state.json``.

    Reads ``runs/<arm>/iteration_N/training/checkpoint-<latest>/trainer_state.json`` → ``log_history`` and keeps
    the training rows (the ones with ``loss``; eval rows at epoch ends and the run summary are dropped). TRL's
    keys are renamed to the short names of :data:`TRAINER_DIAG_KEYS` (``kl``, ``grad_norm``, ``entropy``,
    ``frac_reward_zero_std``, ``clipped_ratio``, ``mean_length``, ``reward``, ``reward_std``, ``learning_rate``,
    ``clip_region``, ``loss``, ``lookahead_turns``, ``oracle_success``); any other key is kept under its TRL name.
    Added columns: ``arm, method, K, iteration, max_steps, frac_of_iteration`` (= step / max_steps).

    Why trainer_state and not TensorBoard: on a resume, HF restores ``log_history`` from the checkpoint it
    resumes from, so the latest checkpoint holds exactly one row per step (a step redone after a crash keeps
    the redo, the values the adapter was actually trained on). The TB event files keep both attempts and, in
    three iterations here, lost the pre-crash file's scalars altogether — see the module note above. Rows are
    deduplicated on ``step`` (keep last) regardless; :func:`trainer_log_coverage` reports what that did.

    What the GRPO columns are (TRL 1.4.0, ``loss_type="grpo"``, ``beta=0.01``, ``num_iterations=1``):

    - ``kl`` — the per-token k3 estimate ``exp(r) - r - 1`` with ``r = log pi_ref - log pi_theta`` on the
      sampled completion tokens, masked mean over the step's 128 completions. The reference is the
      ITERATION-START policy (iteration 1: adapters disabled = the base, whose fresh LoRA is zero; iteration
      >= 2: TRL's frozen ``"ref"`` copy of the adapter as loaded). Both log-probs are of temperature-scaled
      logits (T = 1.2, the GRPO sampling temperature). ⚠ It has a **noise floor**: LoRA dropout (0.05) is
      active and TRL's ``disable_dropout`` defaults to False, so the policy and reference passes use
      different dropout masks. At step 1 of every iteration >= 2 the policy still equals its reference (the
      update has not happened yet) and ``kl`` reads 0.0005-0.0008, rising slowly with the adapter's size;
      at step 1 of iteration 1 (fresh LoRA, B = 0, dropout has nothing to perturb) it reads exactly 0.
      :func:`trainer_diagnostics_by_iter` reports that step-1 value as ``kl_step1`` and ``kl_net`` = mean
      minus it.
      ⚠ **One reference reset: GRPO_LA5 iteration 1, steps 55-108.** That iteration crashed and resumed from
      checkpoint-54. ``resolve_start_state`` (code/_shared/model.py, case B) loads the resumed policy as a
      PeftModel FROM the checkpoint; TRL then copies the loaded weights into a new ``"ref"`` adapter, and HF's
      resume has no ``ref/`` sub-adapter to restore, because iteration 1 had none (its reference was the
      disabled adapter). The ``ref/`` saved in checkpoint-108 is bit-identical to checkpoint-54's policy, so
      for its second epoch that iteration's KL (logged AND penalised) is to the step-54 policy, not the base;
      the logged kl drops from 0.0015 to 0.0006 at step 55. Every resume in iterations >= 2 restores the
      saved ``ref/`` (bit-identical to ``iteration_{N-1}/adapter`` in all seven resumed iterations: K=0 2, 6,
      8; K=5 2, 7, 9, 10), and the logged kl is continuous across those resume points.
    - ``grad_norm`` — global L2 norm of the LoRA gradient BEFORE clipping (``max_grad_norm`` = 1.0, so a
      value > 1 means the step was clipped).
    - ``entropy`` — mean per-token entropy of the temperature-scaled (T = 1.2) policy over the completion
      tokens, dropout active.
    - ``frac_reward_zero_std`` — share of the step's 16 groups whose 8 rewards are all equal (zero
      advantage, so the group contributes only the KL term).
    - ``clipped_ratio`` — share of completions whose last token is neither EOS nor pad, i.e. that ran to
      ``max_completion_length`` = 200 tokens. Stop-string endings are padded with EOS (pad = eos here) and
      do not count; every logged step has ``completions/max_length`` = 200, so no completion that ended on a
      stop string is the batch's longest and none is miscounted.
    - ``reward`` / ``reward_std`` — mean and SD of the step's 128 rewards ACROSS groups (between-prompt +
      within-group variance; the within-group spread the advantage sees is ``advantage_signal_by_iter``'s
      ``group_std``). Under K=5 the reward scores the completion plus a 5-turn rollout — not the K=0 quantity.
    - ``clip_region`` — PPO-clip activity; identically 0 here because one policy update per sampled batch
      makes the importance ratio exactly 1.
    """
    return _trainer_logs(arms)[0]


def trainer_log_coverage(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration): which checkpoint was read and whether its ``log_history`` covers the iteration.

    Columns: ``checkpoint`` (the latest step on disk, the one read), ``checkpoints_on_disk``, ``max_steps``
    (the iteration's planned optimizer steps = 2 epochs), ``global_step``, ``n_rows`` (train rows in
    ``log_history``), ``n_steps`` (distinct steps), ``n_duplicate_rows``, ``n_missing_steps`` (of 1..max_steps),
    ``coverage`` = n_steps / max_steps. Summing ``max_steps`` over iterations gives the run's optimizer-step
    count (the paper's configuration table). An iteration with no checkpoint gets ``coverage`` 0.
    """
    return _trainer_logs(arms)[1]


def trainer_diagnostics_by_iter(steps: pd.DataFrame, *, last_frac: float = 0.25,
                                metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per (arm, iteration) summary of :func:`trainer_log_steps`: mean and last-quarter mean of each metric.

    For each metric present: ``<m>`` = mean over the iteration's steps, ``<m>_lq`` = mean over the last
    ``last_frac`` of them (``frac_of_iteration > 1 - last_frac``; the policy furthest from the iteration-start
    reference), ``<m>_max``. Extra columns: ``kl_step1`` (the KL noise floor, see :func:`trainer_log_steps`),
    ``kl_net`` = ``kl - kl_step1``, ``n_grad_clipped`` (steps with ``grad_norm`` > 1.0), ``lr_peak``,
    ``n_steps``, ``max_steps``. The iteration is the unit: steps inside one iteration are serially dependent
    (one optimizer, one sampled pool), so a step-level test would overstate n.
    """
    metrics = list(metrics) if metrics is not None else [m for m in TRAINER_DIAG_KEYS if m in steps.columns]
    rows = []
    for (arm, it), g in steps.groupby(["arm", "iteration"], sort=True):
        g = g.sort_values("step")
        lq = g[g["frac_of_iteration"] > 1 - last_frac]
        r = {"arm": arm, "method": g["method"].iloc[0], "K": int(g["K"].iloc[0]), "iteration": int(it),
             "n_steps": int(len(g)), "max_steps": int(g["max_steps"].iloc[0])}
        for m in metrics:
            if m not in g.columns or g[m].isna().all():
                continue
            r[m] = float(g[m].mean())
            r[f"{m}_lq"] = float(lq[m].mean())
            r[f"{m}_max"] = float(g[m].max())
        if "kl" in g.columns and g["kl"].notna().any():
            r["kl_step1"] = float(g["kl"].iloc[0])
            r["kl_net"] = r["kl"] - r["kl_step1"]
        if "grad_norm" in g.columns:
            r["n_grad_clipped"] = int((g["grad_norm"] > 1.0).sum())
        if "learning_rate" in g.columns:
            r["lr_peak"] = float(g["learning_rate"].max())
        rows.append(r)
    return pd.DataFrame(rows)


def trainer_diagnostics_table(by_iter: pd.DataFrame, arm_a: str = "GRPO_LA0", arm_b: str = "GRPO_LA5", *,
                              metrics: Sequence[str] = ("kl", "kl_lq", "grad_norm", "entropy",
                                                        "frac_reward_zero_std", "clipped_ratio", "mean_length"),
                              ) -> pd.DataFrame:
    """The appendix shape: one row per iteration, ``<metric> · <arm>`` columns side by side (``arm_a`` first),
    plus a final ``median`` row (the median of the per-iteration values; NOT a step-weighted mean)."""
    w = []
    for m in metrics:
        for arm in (arm_a, arm_b):
            s = by_iter.loc[by_iter["arm"] == arm].set_index("iteration")[m] if m in by_iter else None
            if s is not None:
                w.append(s.rename(f"{m} · {arm}"))
    t = pd.concat(w, axis=1).sort_index()
    t.loc["median"] = t.median()
    t.index = t.index.map(str)
    return t.reset_index().rename(columns={"index": "iteration"})


def trainer_diagnostics_k_contrast(by_iter: pd.DataFrame, arm_a: str = "GRPO_LA5", arm_b: str = "GRPO_LA0", *,
                                   metrics: Sequence[str] = tuple(TRAINER_DIAG_FAMILY) + ("kl_net", "reward",
                                                                                          "reward_std"),
                                   stats_: Sequence[str] = ("mean", "lq")) -> pd.DataFrame:
    """``arm_a - arm_b`` per metric, paired on the training ITERATION (n = 10), for the iteration mean and the
    last-quarter mean. Sign convention: + => ``arm_a`` (default K=5) higher.

    Columns: ``metric, stat, n, median_a, median_b, ratio_of_medians, n_a_higher, n_a_lower, mean_delta, dz,
    ci_lo, ci_hi, p`` (``stats.paired_arrays``: percentile bootstrap over iterations seeded with
    ``constants.BOOT_SEED``, Wilcoxon signed-rank — exact at n = 10, smallest attainable two-sided p = 0.002),
    ``p_holm`` (Holm within each ``stat`` over :data:`TRAINER_DIAG_FAMILY` only; NaN for the other rows) and
    ``comparable``. ``reward`` / ``reward_std`` rows are ``comparable = False``: the K=5 reward scores a
    different object, so a K difference there is not a difference in one quantity. ``kl_net`` (KL minus its
    step-1 noise floor) is a robustness row, not a separate test. The pairing is iteration-to-iteration of
    two diverging policies, and consecutive iterations of one run are serially dependent (each starts from
    the last), so read ``n_a_higher`` / ``n_a_lower`` first and treat ``p`` as descriptive.
    """
    from .stats import paired_arrays, holm
    rows = []
    for stat in stats_:
        for m in metrics:
            col = m if stat == "mean" else f"{m}_{stat}"
            if col not in by_iter.columns:
                continue
            a = by_iter.loc[by_iter["arm"] == arm_a].set_index("iteration")[col]
            b = by_iter.loc[by_iter["arm"] == arm_b].set_index("iteration")[col]
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            r = paired_arrays(j["a"].values, j["b"].values)
            d = j["a"] - j["b"]
            rows.append({"metric": m, "stat": stat, "n": r["n"], "median_a": float(j["a"].median()),
                         "median_b": float(j["b"].median()),
                         "ratio_of_medians": float(j["a"].median() / j["b"].median()) if j["b"].median() else np.nan,
                         "n_a_higher": int((d > 0).sum()), "n_a_lower": int((d < 0).sum()),
                         "mean_delta": r["mean_delta"], "dz": r["dz"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "p": r["p"], "comparable": m not in ("reward", "reward_std")})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for stat, g in out.groupby("stat"):
        fam = g[g["metric"].isin(TRAINER_DIAG_FAMILY)]
        out.loc[fam.index, "p_holm"] = holm(fam["p"].values)
    out.insert(2, "contrast", f"{arm_a} - {arm_b}")
    return out


def trainer_diagnostics_trends(by_iter: pd.DataFrame, *,
                               metrics: Sequence[str] = tuple(TRAINER_DIAG_FAMILY) + ("reward", "reward_std")
                               ) -> pd.DataFrame:
    """Per (arm, metric): Spearman rho of the per-iteration mean against the iteration index, with its p, plus the
    first / last / min / max iteration means and which iterations hold the extremes — so a quoted start-to-end
    change can be read beside the trend (an endpoint that is the series' extremum makes a two-point ratio
    unstable). Descriptive; n = the arm's iterations."""
    from scipy import stats as _ss
    rows = []
    for arm, g in by_iter.sort_values("iteration").groupby("arm"):
        for m in metrics:
            if m not in g.columns:
                continue
            x = g[["iteration", m]].dropna()
            rho, p = _ss.spearmanr(x["iteration"], x[m]) if len(x) >= 3 else (np.nan, np.nan)
            rows.append({"arm": arm, "metric": m, "n_iter": int(len(x)), "rho": float(rho), "p": float(p),
                         "first": float(x[m].iloc[0]), "last": float(x[m].iloc[-1]),
                         "min": float(x[m].min()), "argmin_iter": int(x.loc[x[m].idxmin(), "iteration"]),
                         "max": float(x[m].max()), "argmax_iter": int(x.loc[x[m].idxmax(), "iteration"])})
    return pd.DataFrame(rows)


def trainer_diagnostics_by_length(steps: pd.DataFrame,
                                  bins: Sequence[float] = (0, 70, 80, 90, 110, 140, 170, 190, 201)) -> pd.DataFrame:
    """Step-level means of ``kl``, ``grad_norm``, ``entropy`` and ``clipped_ratio`` per arm inside bins of the
    step's mean completion length (tokens): does a K difference survive at matched completion length? Both
    runs' completions grow toward the 200-token cap over training, the K=5 run's earlier, so an
    iteration-level contrast could be a length contrast. Descriptive (steps are serially dependent)."""
    s = steps.copy()
    s["length_bin"] = pd.cut(s["mean_length"], list(bins))           # ordered categorical
    cols = [c for c in ("kl", "grad_norm", "entropy", "clipped_ratio") if c in s.columns]
    g = s.groupby(["length_bin", "arm"], observed=True, sort=True)
    out = g[cols].mean()
    out.insert(0, "n_steps", g.size())
    out = out.reset_index()
    out["length_bin"] = out["length_bin"].astype(str)
    return out


def trainer_diagnostics_numbers(by_iter: pd.DataFrame, contrast: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    """Ledger mapping for ``exports.save_numbers`` (keyed like ``dispersion_numbers``): ``coverage.<arm>`` (optimizer
    steps, logged steps, duplicate rows), ``by_iter.<arm>.iter<N>`` (a dict of the iteration's means, ``_lq``,
    ``kl_step1``, ``kl_net``, ``n_grad_clipped``) and ``k_contrast.<stat>.<metric>`` (a dict of the contrast row)."""
    out = {}
    for arm, g in coverage.groupby("arm"):
        out[f"coverage.{arm}"] = {
            "value": {"optimizer_steps": int(g["max_steps"].sum()), "logged_steps": int(g["n_steps"].sum()),
                      "duplicate_rows": int(g["n_duplicate_rows"].sum()),
                      "missing_steps": int(g["n_missing_steps"].sum()), "iterations": int(len(g))},
            "source": "tables/trainer_log_coverage.md",
            "note": "optimizer_steps = sum of max_steps; logged = distinct steps in each latest checkpoint's log_history"}
    keep = ["n_steps", "kl", "kl_lq", "kl_step1", "kl_net", "kl_max", "grad_norm", "grad_norm_lq", "n_grad_clipped",
            "entropy", "entropy_lq", "frac_reward_zero_std", "clipped_ratio", "mean_length", "reward", "reward_std",
            "lookahead_turns", "lr_peak"]
    for r in by_iter.to_dict("records"):
        out[f"by_iter.{r['arm']}.iter{r['iteration']}"] = {
            "value": {k: (int(r[k]) if k in ("n_steps", "n_grad_clipped") else float(r[k]))
                      for k in keep if k in r and pd.notna(r[k])},
            "source": f"tables/trainer_diagnostics_by_iter.md row arm={r['arm']} iteration={r['iteration']}",
            "note": "iteration means over optimizer steps; _lq = last quarter"}
    for r in contrast.to_dict("records"):
        out[f"k_contrast.{r['stat']}.{r['metric']}"] = {
            "value": {k: float(r[k]) for k in ("n", "median_a", "median_b", "ratio_of_medians", "n_a_higher",
                                                "n_a_lower", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm")
                      if pd.notna(r[k])},
            "source": f"tables/trainer_diagnostics_k_contrast.md row metric={r['metric']} stat={r['stat']}",
            "note": f"{r['contrast']} (+ => K=5 higher); paired on training iteration; comparable={r['comparable']}"}
    return out
