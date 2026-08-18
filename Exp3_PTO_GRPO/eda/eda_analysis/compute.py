"""compute.py — the COMPUTE axis: GPU-hours per (arm, iteration).

Every other contrast in this project is indexed by **iteration**. That is not a neutral choice.
A K=5 optimizer step costs ~1.9x a K=0 step, and a GRPO iteration costs several times a PTO one,
so a matched-*iteration* contrast silently hands one arm far more compute per cell than the
other. Read the same arms at matched *budget* and some contrasts change sign.

This module recovers that missing x-axis from artifacts already on disk — no new runs, no API.
It is the free precondition for any claim of the form "method/lever X is worth it".

The cost model
--------------
An iteration is billed as the phases that actually had to run to produce its adapter::

    generate   rollouts from the current policy      (both methods)
    build      preference-tree branching + oracle    (PTO only — its dominant phase)
    train      the optimizer loop                    (both methods)

``cum_gpu_h`` at iteration ``k`` is the sum of those over iterations ``1..k`` — the cost of
having produced the policy the score lake calls ``<Arm>_I{k}``. Iteration ``j``'s generation is
the rollout its own update trains on, so it belongs to ``j``. The base policy (``I0``) is 0 by
construction.

Why not the recorded timings
----------------------------
``iteration_metadata.json``'s ``training_time_s`` / ``generation_time_s`` / ``pref_pair_time_s``
are **per-PROCESS**: an iteration that crashed and resumed records only its last session. The
damage is not subtle — GRPO_LA5 iteration 1 logs 14,501 s for work spanning 7.7 h, and PTO
iteration 1 logs ``pref_pair_time_s = 3.2 s`` for a ~30 min build, because the build was reloaded
from ``pairs.csv`` rather than re-run. Every phase here is therefore timed from **artifact
mtimes**, which record when work actually landed:

===========  ===============================================================
phase        source
===========  ===============================================================
generate     mtime span of that iteration's ``model_iter_{k-1}/*.csv``
build (PTO)  last conversation mtime -> ``iteration_k/pref_pairs/pairs.csv``
train        GRPO: ``training/completions/*.parquet`` (one per optimizer step)
             PTO:  TensorBoard scalar ``wall_time`` (DPOTrainer writes no
                   completions), read via the same event files ``tb_curves`` uses
===========  ===============================================================

**Runs from here on skip all of that.** ``code/_shared/timing.py`` appends one line per process to
``iteration_N/timing_sessions.jsonl``, so summing it survives resumes by construction. When that log
is present it is used directly (``train_source = "timing_sessions"``) and the mtime path is not
consulted — it measures the work rather than the interval around it. No arm in this thesis has one;
the reconstruction below is what they all use, and it has to keep working.

Two artifacts corrupt raw mtime deltas and are handled identically everywhere:

* **Resume gaps** — a crashed-and-resumed phase leaves one enormous delta of idle wall-clock.
* **Drive mtime rewrites** — ``data/`` is a Google Drive Desktop symlink, and a re-synced file
  can return with a rewritten (even out-of-order) mtime.

Any delta outside ``(0, gap_cutoff_s)`` is replaced by that phase's own median, so the work still
counts **once** instead of being dropped (which undercounts a resumed phase) or summed (which
would bill days of idle time). ``n_imputed`` reports how often this fired; read it before
trusting a row. Phases measured as a single span (generate, build) fall back to the arm's median
for that phase when the span itself is a resume gap.

⚠ **Iso-compute pairs DIFFERENT iterations across arms**, so ``file_index`` pairing is invalid
there — the 96 personas are reshuffled every iteration (``seed + k + 1``). Everything in this
module pairs on ``persona_id``. See :func:`iso_compute_contrast`.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import QUESTIONNAIRE_ORDER

_STEP_RE = re.compile(r"(\d+)")
_ITER_DIR_RE = re.compile(r"iteration_(\d+)$")

#: A gap this large inside a phase is a crash/resume boundary or a re-synced Drive mtime, never
#: real work. One hour is ~20x the slowest observed optimizer step (K=5: 179.5 s median, 578 s
#: p90) and ~1.2x the longest observed single phase, so the separation is not marginal.
GAP_CUTOFF_S = 3600.0

_MEMO: Dict[str, pd.DataFrame] = {}


def clear_memo() -> None:
    """Drop the in-process cache (call after a run writes new steps)."""
    _MEMO.clear()


# ── phase timing primitives ───────────────────────────────────────────────────
def _clean_span(times: Sequence[float], *, gap_cutoff_s: float) -> Tuple[float, int, float]:
    """(seconds, n_imputed, median_delta) over a sorted mtime series, gaps imputed at the median.

    Returns ``(0.0, 0, nan)`` when there is nothing to measure. A series of n timestamps bounds
    n-1 intervals; imputing keeps the count at n-1 rather than silently shortening it.
    """
    t = np.sort(np.asarray([x for x in times if x], dtype=float))
    if t.size < 2:
        return 0.0, 0, float("nan")
    d = np.diff(t)
    ok = (d > 0) & (d < gap_cutoff_s)
    if not ok.any():
        return 0.0, int((~ok).sum()), float("nan")
    med = float(np.median(d[ok]))
    return float(np.where(ok, d, med).sum()), int((~ok).sum()), med


def _mtimes(pattern: str) -> List[float]:
    out = []
    for fp in glob.glob(pattern):
        try:
            out.append(os.path.getmtime(fp))
        except OSError:
            continue
    return out


def _iter_dirs(runs_dir: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not os.path.isdir(runs_dir):
        return out
    for d in sorted(glob.glob(os.path.join(runs_dir, "iteration_*"))):
        m = _ITER_DIR_RE.search(os.path.basename(d))
        if m and os.path.isdir(d):
            out[int(m.group(1))] = d
    return out


def _grpo_step_times(iter_dir: str) -> List[float]:
    """One mtime per optimizer step, ordered by step index (GRPO writes a parquet per step)."""
    d = os.path.join(iter_dir, "training", "completions")
    if not os.path.isdir(d):
        return []
    rows = []
    for fn in os.listdir(d):
        if not fn.endswith(".parquet"):
            continue
        m = _STEP_RE.search(fn)
        try:
            rows.append((int(m.group(1)) if m else 0, os.path.getmtime(os.path.join(d, fn))))
        except OSError:
            continue
    rows.sort()
    return [t for _, t in rows]


def _tb_step_times(iter_dir: str) -> List[float]:
    """Per-step wall_time from the iteration's TensorBoard scalars (the DPO/PTO path).

    DPOTrainer writes no per-step artifact, but the TB event stream stamps every scalar with a
    wall clock. Uses the densest scalar tag so the series is one point per optimizer step.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except Exception:
        return []
    best: List[float] = []
    for fp in glob.glob(os.path.join(iter_dir, "**", "events.out.tfevents.*"), recursive=True):
        try:
            acc = ea.EventAccumulator(fp, size_guidance={ea.SCALARS: 0})
            acc.Reload()
        except Exception:
            continue
        for tag in acc.Tags().get("scalars", []):
            try:
                ts = [e.wall_time for e in acc.Scalars(tag)]
            except Exception:
                continue
            if len(ts) > len(best):
                best = ts
    return best


def _meta(iter_dir: str) -> dict:
    try:
        with open(os.path.join(iter_dir, "iteration_metadata.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _recorded_phases(iter_dir: str) -> Optional[Dict[str, float]]:
    """Phase seconds from the trainer's own **cumulative** log, if this run recorded one.

    ``_shared/timing.py`` (added after every run currently on disk) appends one line per process to
    ``iteration_N/timing_sessions.jsonl``, so summing it survives resumes — which the per-process
    ``iteration_metadata.json`` fields do not. When that log exists it is strictly better than mtime
    reconstruction: it measures the work rather than the interval around it.

    Returns ``None`` when there is no log, which is the case for every arm in this thesis — they
    fall through to the mtime path below. Never raises; a malformed log is treated as absent.
    """
    fp = os.path.join(iter_dir, "timing_sessions.jsonl")
    if not os.path.isfile(fp):
        return None
    tot = {"generation_s": 0.0, "pref_pair_s": 0.0, "training_s": 0.0}
    n = 0
    try:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k in tot:
                    try:
                        tot[k] += float(rec.get(k) or 0.0)
                    except (TypeError, ValueError):
                        pass
                n += 1
    except OSError:
        return None
    if not n or sum(tot.values()) <= 0:
        return None
    tot["n_sessions"] = float(n)
    return tot


def iteration_compute(arms: Optional[List] = None, *,
                      gap_cutoff_s: float = GAP_CUTOFF_S) -> pd.DataFrame:
    """GPU-hours per (arm, iteration), reconstructed from artifact mtimes.

    Columns: ``arm, method, K, exp_name, iteration, n_steps, median_step_s, n_imputed,
    gen_h, build_h, train_h, gpu_h, cum_gpu_h, train_source``.

    An ``iteration = 0`` row is emitted per arm with ``cum_gpu_h = 0`` so the frame joins
    directly onto the score lake's base state. An iteration whose training never started
    (fewer than 3 timed steps — e.g. a run stopped moments into its next iteration) is
    EXCLUDED: no adapter and therefore no scored model state depends on it.

    Single-span phases (generate, build) whose only interval is a resume gap are back-filled
    with that arm's median for the phase, so one interrupted iteration does not silently bill
    as zero. Rows where that happened are counted in ``n_imputed``.
    """
    from .data import discover_arms
    arms = discover_arms() if arms is None else arms

    key = "|".join(sorted(f"{a.label}:{a.exp_name}" for a in arms)) + f"@{gap_cutoff_s}"
    if key in _MEMO:
        return _MEMO[key].copy()

    rows = []
    for arm in arms:
        rows.append({"arm": arm.label, "method": arm.method, "K": arm.K,
                     "exp_name": arm.exp_name, "iteration": 0, "n_steps": 0,
                     "median_step_s": np.nan, "n_imputed": 0,
                     "gen_h": 0.0, "build_h": 0.0, "train_h": 0.0, "gpu_h": 0.0,
                     "train_source": "-"})
        for it, d in sorted(_iter_dirs(arm.runs_dir).items()):
            # -- the trainer's own cumulative log wins when it exists (post-dates every run here) --
            rec = _recorded_phases(d)
            if rec is not None:
                rows.append({
                    "arm": arm.label, "method": arm.method, "K": arm.K, "exp_name": arm.exp_name,
                    "iteration": it, "n_steps": 0, "median_step_s": np.nan, "n_imputed": 0,
                    "gen_h": rec["generation_s"] / 3600.0,
                    "build_h": rec["pref_pair_s"] / 3600.0,
                    "train_h": rec["training_s"] / 3600.0,
                    "gpu_h": sum(rec[k] for k in ("generation_s", "pref_pair_s", "training_s")) / 3600.0,
                    "train_source": "timing_sessions",
                })
                continue

            # -- train: GRPO writes a parquet per step; DPO only writes TB scalars ----------
            steps, src = _grpo_step_times(d), "completions"
            if len(steps) < 3:
                steps, src = _tb_step_times(d), "tb_wall_time"
            if len(steps) < 3:
                continue
            train_s, n_imp, med = _clean_span(steps, gap_cutoff_s=gap_cutoff_s)
            if not train_s:
                continue

            # -- generate: the rollout pass that produced model_iter_{it-1} -----------------
            conv_dir = arm.conv_dirs.get(it - 1)
            gen_s, gi = 0.0, 0
            conv_end = None
            if conv_dir:
                ct = _mtimes(os.path.join(conv_dir, "conversation_*.csv"))
                gen_s, gi, _ = _clean_span(ct, gap_cutoff_s=gap_cutoff_s)
                conv_end = max(ct) if ct else None

            # -- build (PTO): last conversation -> pairs.csv --------------------------------
            build_s, bi = 0.0, 0
            pairs_csv = os.path.join(d, "pref_pairs", "pairs.csv")
            if conv_end is not None and os.path.exists(pairs_csv):
                try:
                    delta = os.path.getmtime(pairs_csv) - conv_end
                except OSError:
                    delta = 0.0
                if 0 < delta < gap_cutoff_s * 4:      # a build legitimately runs ~0.4-0.9 h
                    build_s = float(delta)
                else:
                    bi = 1                            # resume/rewrite — back-filled below

            rows.append({
                "arm": arm.label, "method": arm.method, "K": arm.K, "exp_name": arm.exp_name,
                "iteration": it, "n_steps": len(steps), "median_step_s": med,
                "n_imputed": n_imp + gi + bi,
                "gen_h": gen_s / 3600.0, "build_h": build_s / 3600.0,
                "train_h": train_s / 3600.0,
                "gpu_h": (gen_s + build_s + train_s) / 3600.0,
                "train_source": src,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Back-fill single-span phases that were lost to a resume, using the arm's own median.
    for phase in ("gen_h", "build_h"):
        for arm_label, grp in df[df.iteration > 0].groupby("arm"):
            pos = grp[grp[phase] > 0][phase]
            if pos.empty:
                continue
            med = float(pos.median())
            hit = grp.index[(grp[phase] <= 0) & (grp["n_imputed"] > 0)]
            df.loc[hit, phase] = med
    df["gpu_h"] = df[["gen_h", "build_h", "train_h"]].sum(axis=1)

    df = df.sort_values(["arm", "iteration"]).reset_index(drop=True)
    df["cum_gpu_h"] = df.groupby("arm")["gpu_h"].cumsum()
    _MEMO[key] = df.copy()
    return df


def compute_summary(comp: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: iterations trained, total GPU-h, and cost per iteration.

    The table that makes "arm X only reached iteration 5" checkable against "arm X cost the
    same as arm Y's ten".
    """
    d = comp[comp.iteration > 0]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("arm").agg(
        method=("method", "first"), K=("K", "first"),
        last_iter=("iteration", "max"), n_iters=("iteration", "count"),
        gen_h=("gen_h", "sum"), build_h=("build_h", "sum"), train_h=("train_h", "sum"),
        total_gpu_h=("gpu_h", "sum"), median_step_s=("median_step_s", "median"),
        n_imputed=("n_imputed", "sum"), train_source=("train_source", "first"),
    ).reset_index()
    g["gpu_h_per_iter"] = g["total_gpu_h"] / g["n_iters"]
    return g.sort_values(["method", "K"]).reset_index(drop=True)


def step_multiplier(comp: pd.DataFrame, method: str = "GRPO",
                    K_lo: int = 0, K_hi: int = 5) -> pd.DataFrame:
    """Per-step cost ratio K_hi / K_lo, by iteration — the price of look-ahead.

    Reported per iteration rather than pooled because it is NOT stable: an arm's earliest
    iterations can carry a different ``LOOKAHEAD_SUB_BATCH_SIZE`` and a fatter API-latency
    tail, both of which inflate the ratio without being intrinsic to K.
    """
    lo, hi = f"{method}_LA{K_lo}", f"{method}_LA{K_hi}"
    d = comp[comp.arm.isin([lo, hi]) & (comp.iteration > 0)]
    if d.empty:
        return pd.DataFrame()
    med = d.pivot_table(index="iteration", columns="arm", values="median_step_s")
    out = pd.DataFrame(index=med.index)
    for arm, K in ((lo, K_lo), (hi, K_hi)):
        out[f"median_s_K{K}"] = med[arm] if arm in med.columns else np.nan
    out["ratio_median"] = out[f"median_s_K{K_hi}"] / out[f"median_s_K{K_lo}"]
    return out.reset_index()


def iso_compute_pairs(comp: pd.DataFrame, arm_a: str, arm_b: str) -> pd.DataFrame:
    """For every trained iteration of ``arm_a``, the closest-BUDGET iteration of ``arm_b``.

    Returns ``iter_a, cum_gpu_h_a, iter_b, cum_gpu_h_b, budget_ratio, budget_gap_h``.
    Nothing is dropped for matching poorly — ``budget_ratio`` (b/a) is reported so the reader
    can see how well the budgets line up. A ratio outside ~0.9-1.1 should not be quoted as
    iso-compute.
    """
    A = comp[comp.arm == arm_a].set_index("iteration")["cum_gpu_h"]
    B = comp[comp.arm == arm_b].set_index("iteration")["cum_gpu_h"]
    if A.empty or B.empty:
        return pd.DataFrame()
    B = B[B.index > 0]
    if B.empty:
        return pd.DataFrame()
    rows = []
    for ia, ca in A.items():
        if ia == 0:
            continue
        ib = int((B - ca).abs().idxmin())
        cb = float(B.loc[ib])
        rows.append({"iter_a": int(ia), "cum_gpu_h_a": float(ca),
                     "iter_b": ib, "cum_gpu_h_b": cb,
                     "budget_ratio": (cb / ca) if ca else np.nan,
                     "budget_gap_h": cb - float(ca)})
    return pd.DataFrame(rows)


def _paired_on_persona(scores_long: pd.DataFrame, model_a: str, model_b: str,
                       metric: str) -> Optional[np.ndarray]:
    """Deltas a-b over the personas both states scored. ``None`` if unusable."""
    d = scores_long[(scores_long.questionnaire == metric)
                    & (scores_long.model.isin([model_a, model_b]))]
    if d.empty or "persona_id" not in d.columns:
        return None
    w = d.pivot_table(index="persona_id", columns="model", values="score")
    if model_a not in w.columns or model_b not in w.columns:
        return None
    w = w[[model_a, model_b]].dropna()
    if len(w) < 10:
        return None
    return (w[model_a] - w[model_b]).to_numpy()


def _model_names(scores_long: pd.DataFrame, arm: str) -> Dict[int, str]:
    return {int(i): m for i, m in
            scores_long[scores_long.arm == arm][["iteration", "model"]]
            .drop_duplicates().itertuples(index=False)}


def iso_compute_contrast(scores_long: pd.DataFrame, comp: pd.DataFrame,
                         arm_a: str, arm_b: str, *,
                         metrics: Optional[Sequence[str]] = None,
                         iters_a: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """``arm_a`` vs ``arm_b`` at matched cumulative GPU-hours, paired on persona.

    ``+ mean_delta => arm_a higher`` (so on a LOWER_IS_BETTER metric a positive delta means
    ``arm_a`` is WORSE — the convention the rest of ``stats.py`` uses).

    Pairing is on ``persona_id``, NOT ``file_index``: matched-budget iterations differ across
    the two arms and the persona shuffle is ``seed + k + 1``, so a file_index join across
    unmatched iterations pairs unrelated conversations. Means survive that mistake; ``dz`` and
    CIs do not.
    """
    from scipy import stats as sps
    from .stats import holm

    pairs = iso_compute_pairs(comp, arm_a, arm_b)
    if pairs.empty:
        return pd.DataFrame()
    if iters_a is not None:
        pairs = pairs[pairs.iter_a.isin(list(iters_a))]

    present = set(scores_long["questionnaire"].unique())
    metrics = [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in present]
    name_a, name_b = _model_names(scores_long, arm_a), _model_names(scores_long, arm_b)

    blocks = []
    for p in pairs.itertuples(index=False):
        ma, mb = name_a.get(p.iter_a), name_b.get(p.iter_b)
        if ma is None or mb is None:
            continue
        block = []
        for m in metrics:
            d = _paired_on_persona(scores_long, ma, mb, m)
            if d is None or np.std(d, ddof=1) == 0:
                continue
            try:
                _, pv = sps.wilcoxon(d)
            except ValueError:
                continue
            block.append({
                "arm_a": arm_a, "arm_b": arm_b, "metric": m,
                "iter_a": p.iter_a, "iter_b": p.iter_b,
                "cum_gpu_h_a": round(p.cum_gpu_h_a, 2), "cum_gpu_h_b": round(p.cum_gpu_h_b, 2),
                "budget_ratio": round(p.budget_ratio, 3),
                "model_a": ma, "model_b": mb, "n": len(d),
                "mean_delta": float(np.mean(d)), "dz": float(np.mean(d) / np.std(d, ddof=1)),
                "p": float(pv),
            })
        if block:
            b = pd.DataFrame(block)
            b["p_holm"] = holm(b["p"].values)      # family = the rubrics at one budget
            blocks.append(b)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def budget_sweep(scores_long: pd.DataFrame, comp: pd.DataFrame,
                 arm_a: str, arm_b: str, *, metric: str = "Q1Q2") -> pd.DataFrame:
    """The lever's sign as a function of budget — best-checkpoint-within-budget.

    At each of ``arm_a``'s cumulative budgets both arms are represented by the best iteration
    they could have *reached* for that money (best on ``metric`` under the active judge), and
    the two are contrasted paired-on-persona. This is the honest "was it worth it?" curve: each
    arm gets to spend its budget as well as it can, rather than one being frozen at a fixed
    iteration count.

    ``+ mean_delta => arm_a higher``.
    """
    from scipy import stats as sps

    A = comp[(comp.arm == arm_a) & (comp.iteration > 0)].set_index("iteration")["cum_gpu_h"]
    B = comp[(comp.arm == arm_b) & (comp.iteration > 0)].set_index("iteration")["cum_gpu_h"]
    if A.empty or B.empty:
        return pd.DataFrame()

    sub = scores_long[scores_long.questionnaire == metric]
    means = sub.groupby(["arm", "iteration"])["score"].mean()
    names = {(a, int(i)): m for a, i, m in
             sub[["arm", "iteration", "model"]].drop_duplicates().itertuples(index=False)}

    def best_within(arm, budget, cum):
        elig = [int(i) for i, c in cum.items()
                if c <= budget + 1e-9 and (arm, int(i)) in means.index]
        return max(elig, key=lambda i: means.loc[(arm, i)]) if elig else None

    rows = []
    for budget in sorted(A.values):
        ia, ib = best_within(arm_a, budget, A), best_within(arm_b, budget, B)
        if ia is None or ib is None:
            continue
        ma, mb = names.get((arm_a, ia)), names.get((arm_b, ib))
        d = _paired_on_persona(scores_long, ma, mb, metric) if ma and mb else None
        if d is None or np.std(d, ddof=1) == 0:
            continue
        try:
            _, pv = sps.wilcoxon(d)
        except ValueError:
            continue
        rows.append({"budget_gpu_h": round(float(budget), 2), "metric": metric,
                     "best_iter_a": ia, "best_iter_b": ib,
                     "mean_a": float(means.loc[(arm_a, ia)]),
                     "mean_b": float(means.loc[(arm_b, ib)]),
                     "n": len(d), "mean_delta": float(np.mean(d)),
                     "dz": float(np.mean(d) / np.std(d, ddof=1)), "p": float(pv)})
    return pd.DataFrame(rows)


def score_by_compute(scores_long: pd.DataFrame, comp: pd.DataFrame, *,
                     metric: str = "Q1Q2") -> pd.DataFrame:
    """Long frame of ``(arm, iteration, cum_gpu_h, mean, sem)`` — the plotting backbone.

    This is the frame every trajectory figure in the project *should* be able to draw against
    but currently cannot, because nothing else carries ``cum_gpu_h``.
    """
    sub = scores_long[scores_long.questionnaire == metric]
    if sub.empty:
        return pd.DataFrame()
    g = (sub.groupby(["arm", "iteration"])["score"]
         .agg(mean="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)), n="size")
         .reset_index())
    c = comp[["arm", "iteration", "cum_gpu_h", "gpu_h"]]
    out = g.merge(c, on=["arm", "iteration"], how="left")
    out["metric"] = metric
    return out.sort_values(["arm", "iteration"]).reset_index(drop=True)
