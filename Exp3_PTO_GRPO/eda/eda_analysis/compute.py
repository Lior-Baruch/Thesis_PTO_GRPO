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

The paper layer (promoted 2026-08-18)
-------------------------------------
The second half of this module is the look-ahead paper's ``compute_axis.py`` generator
(``papers/2026_lookahead_pto_grpo/analysis/compute_axis.py``, frozen 2026-08-17; its
``tables/compute_axis_*.csv`` + ``analysis/out/compute_axis.json`` are kept as the FIXTURE the
self-check compares against). It re-reads the four contrasts — PTO K, GRPO K, PTO-vs-GRPO at
K=0, PTO-vs-GRPO at K=5 — as a function of BUDGET under BOTH graders, adds a cross-judge
selection sweep, the K contrast at matched compute on the behaviour channels, a per-iteration
step-multiplier table and the generation-time FLOOR columns. Functions take frames per judge
(``{judge_label: scores_long}``) and return tidy DataFrames; nothing here writes to disk
(the ``compute/cost`` notebook owns ``exports.*``) and :func:`compute_numbers` returns the
quotable-numbers ledger for ``exports.save_numbers``.

Sign conventions (state them in every caption):

* **K contrasts** are computed as ``arm_a = LA5, arm_b = LA0`` to reproduce the tracked EDA
  table (``mean_delta = K5 - K0``); the paper's convention (``+ => K=0 higher``) is carried
  beside it in ``delta_K0_minus_K5`` / ``dz_K0_minus_K5``. :func:`~eda_analysis.plotting.compute.budget_sweep_grid`
  plots K5 - K0 (above zero = look-ahead ahead), as its axis label says.
* **Method contrasts**: ``arm_a = PTO, arm_b = GRPO``; ``+ mean_delta => PTO higher``.
* MICI and every ``MICI_*`` channel are LOWER-is-better; count/length channels have no valence.
* Everything is paired on ``persona_id`` (never ``file_index``).
* **Budget ceilings:** each arm's ceiling is its OWN last ``cum_gpu_h`` in
  ``compute_by_iteration`` (read it there; the number moves whenever a run advances), so a sweep
  reaches only the checkpoints that arm's spend bought — every sweep/iso frame
  inherits that.
* Bootstrap CIs use :func:`eda_analysis.stats.paired_arrays` (seeded with
  :data:`constants.BOOT_SEED`); the paper's generator seeded its own helper with 0, so CI bounds
  may differ from the frozen fixture in the third decimal while ``mean_delta``/``dz``/``p``/``n``
  match exactly.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (QUESTIONNAIRE_ORDER, LOWER_IS_BETTER, DISPLAY_NAMES,
                        PRIMARY_JUDGE_TAG, judge_dirname)
# The persona pivot is THE pairing primitive (see the file_index gotcha in CLAUDE.md); one
# definition so a fix to the pairing cannot land in one copy and not the other. (lookahead
# imports only constants/stats/ledger, so this is cycle-free.)
from .lookahead import wide_by_persona as _wide

__all__ = [
    # the mtime-reconstructed cost frame + the contrasts that need it (pre-2026-08-18 surface)
    "GAP_CUTOFF_S", "clear_memo", "iteration_compute", "compute_summary", "step_multiplier",
    "iso_compute_pairs", "iso_compute_contrast", "budget_sweep", "score_by_compute",
    # the paper layer (promoted from papers/2026_lookahead_pto_grpo/analysis/compute_axis.py)
    "CONTRASTS", "K_CONTRASTS", "METHOD_CONTRASTS", "CHANNELS", "TEXT_CHANNELS",
    "CENSOR_NOTE", "SIGN_K", "SIGN_M", "sign_note", "channel_direction",
    "compute_by_iteration_with_floor", "compute_by_arm_with_floor", "cost_ratios",
    "step_multiplier_table", "budget_sweep_ci", "add_k_convention", "all_budget_sweeps",
    "budget_sweep_top", "budget_sweep_crossjudge", "crossjudge_verdicts",
    "iso_channels", "iso_channels_selected", "compute_numbers",
]

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
    # Process-local memo above, cross-process parquet cache below: render_results.py runs six
    # separate kernels, so the memo alone re-paid this ~5.5 s Drive walk + TensorBoard reload in
    # every one of them.
    from .data import load_cached, runs_input_roots, RUN_SIGNATURE_EXTS
    cached = load_cached("iteration_compute", list(arms),
                         lambda: _iteration_compute_impl(arms, gap_cutoff_s=gap_cutoff_s),
                         input_roots=runs_input_roots(arms),
                         params={"gap_cutoff_s": gap_cutoff_s}, exts=RUN_SIGNATURE_EXTS)
    _MEMO[key] = cached.copy()
    return cached.copy()


def _iteration_compute_impl(arms, *, gap_cutoff_s: float = GAP_CUTOFF_S) -> pd.DataFrame:

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
    return df                     # memo + parquet cache are the caller's (iteration_compute)


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


# ═════════════════════════════════════════════════════════════════════════════
# The paper layer — promoted from papers/2026_lookahead_pto_grpo/analysis/compute_axis.py
# (generator frozen 2026-08-17; promoted 2026-08-18). Every function below takes frames and
# returns frames; nothing writes to disk.
# ═════════════════════════════════════════════════════════════════════════════

#: ``(tag, arm_a, arm_b, human label)``. K contrasts put LA5 FIRST so ``mean_delta`` reproduces
#: the tracked EDA sweep row-for-row (``mean_delta = K5 - K0``); the paper's ``+ => K=0 higher``
#: reading is added as ``delta_K0_minus_K5`` by :func:`add_k_convention`.
CONTRASTS: List[Tuple[str, str, str, str]] = [
    ("PTO_K",     "PTO_LA5",  "PTO_LA0",  "PTO: K=5 vs K=0"),
    ("GRPO_K",    "GRPO_LA5", "GRPO_LA0", "GRPO: K=5 vs K=0"),
    ("method_K0", "PTO_LA0",  "GRPO_LA0", "K=0: PTO vs GRPO"),
    ("method_K5", "PTO_LA5",  "GRPO_LA5", "K=5: PTO vs GRPO"),
]
K_CONTRASTS = [c for c in CONTRASTS if c[0].endswith("_K")]
METHOD_CONTRASTS = [c for c in CONTRASTS if not c[0].endswith("_K")]

#: Behaviour channels for the matched-compute K contrast (one Holm family per budget pair).
CHANNELS: List[str] = [
    "MICI_OverPraise", "MICI_OverPraise_rate",
    "MICI_AdviseNoPermission", "MICI_AdviseNoPermission_rate",
    "MICI_BehaviorTotal", "B6_AF", "B6_AF_per_turn",
    "conv_len", "mean_turn_len",
]
#: Deterministic text measures — grader-independent, so reported ONCE (under the primary's rows).
TEXT_CHANNELS = {"conv_len", "mean_turn_len"}
TEXT_JUDGE_LABEL = "text (grader-independent)"

# A LEGEND, not an assertion that any arm is short. It said "GRPO_LA5 is right-censored ..." until
# 2026-08-25 and kept saying it after that arm finished at iteration 10, shipping a false claim into
# every caption that interpolated it. `constants.support_note` derives the real sentence and returns
# "" when nothing is short.
CENSOR_NOTE = ("Each arm's budget ceiling is its own last cum_gpu_h in compute_by_iteration (read it "
               "there - it moves as a run advances), so a sweep only reaches the checkpoints that "
               "arm's spend actually bought.")
SIGN_K = ("K contrast: mean_delta = K5 - K0 (arm_a=LA5, as the tracked EDA table); "
          "delta_K0_minus_K5 = -mean_delta is the paper's convention (+ => K=0 higher).")
SIGN_M = "Method contrast: mean_delta = PTO - GRPO (+ => PTO higher)."


def sign_note(tag: str) -> str:
    """The sign sentence for a contrast tag (``*_K`` -> :data:`SIGN_K`, else :data:`SIGN_M`)."""
    return SIGN_K if tag.endswith("_K") else SIGN_M


def channel_direction(metric: str) -> str:
    """How to read the sign of a channel contrast (``lower=better`` / ``higher=better`` /
    ``higher=more MI-consistent`` for the MITI-coded B-codes and per-turn rates / ``count (no
    valence)`` for the text channels)."""
    if metric in LOWER_IS_BETTER:
        return "lower=better"
    if metric in TEXT_CHANNELS:
        return "count (no valence)"
    if metric.startswith("B") or metric.endswith("_per_turn"):
        return "higher=more MI-consistent"
    return "higher=better"


def _label_of(metric: str) -> str:
    return DISPLAY_NAMES.get(metric, metric)


def _primary_label(judges: Sequence[str]) -> str:
    """Which key of a ``{judge_label: frame}`` dict is the training oracle: the one matching
    :func:`constants.judge_dirname` of :data:`PRIMARY_JUDGE_TAG` if present, else the FIRST key
    (callers load the primary first by convention)."""
    judges = list(judges)
    prim = judge_dirname(PRIMARY_JUDGE_TAG)
    return prim if prim in judges else judges[0]


def _round_or_none(x, nd: int = 3):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ── 1. cost frame with the generation FLOOR + per-arm summary with shares ────────────────────
def _meta_generation_h(runs_dir: str, it: int) -> float:
    """``iteration_metadata.json`` ``generation_time_s`` / 3600 — per-PROCESS (a resumed or
    reloaded pass records only seconds), so an informational FLOOR beside the mtime ``gen_h``,
    never the headline."""
    fp = os.path.join(runs_dir, f"iteration_{it}", "iteration_metadata.json")
    try:
        with open(fp, encoding="utf-8") as fh:
            v = json.load(fh).get("generation_time_s")
        return float(v) / 3600.0 if v is not None else np.nan
    except Exception:
        return np.nan


def compute_by_iteration_with_floor(comp: pd.DataFrame, arms: Optional[List] = None) -> pd.DataFrame:
    """:func:`iteration_compute` output plus the generation-time FLOOR columns.

    Adds ``gen_h_meta`` (``iteration_metadata.json`` ``generation_time_s``/3600, per-PROCESS),
    ``gen_h_floor = max(gen_h, gen_h_meta)``, ``gpu_h_floor`` and ``cum_gpu_h_floor``.

    Why a floor exists at all: the headline ``gen_h`` is the mtime span of
    ``model_iter_{k-1}/*.csv``, which starts at the FIRST conversation write — so with
    ``CONVERSATION_BATCH_SIZE=64`` it misses the whole first batch (~0.1 h/iter) and collapses to
    ~0 when all CSVs flush together (PTO_LA5 iters 1-5, whose time then lands in iter 6). The
    recorded per-process ``generation_time_s`` is a lower bound whenever the pass ran once, so
    ``max(mtime span, recorded)`` is a floor on the true generation time. The headline
    ``gen_h``/``gpu_h``/``cum_gpu_h`` stay the tracked EDA numbers; the floor is a sensitivity
    column (under it PTO_LA0 8.12->9.22 h, PTO_LA5 19.68->21.08 h, GRPO_LA0 27.91->28.77 h,
    GRPO_LA5 27.08->27.42 h at the 2026-08-17 fixture). Iteration 0 = 0 by construction.

    ``arms`` supplies each arm's ``runs_dir`` (default: :func:`~eda_analysis.data.discover_arms`,
    matched by ``label``). Reproduces ``compute_axis_by_iteration.csv``.
    """
    from .data import discover_arms
    if comp is None or comp.empty:
        return comp
    arms = discover_arms() if arms is None else arms
    runs_by_label = {a.label: a.runs_dir for a in arms}
    out = comp.copy()
    out["gen_h_meta"] = [
        _meta_generation_h(runs_by_label[r.arm], int(r.iteration))
        if (r.iteration > 0 and r.arm in runs_by_label) else (0.0 if r.iteration == 0 else np.nan)
        for r in out.itertuples(index=False)]
    out["gen_h_floor"] = out[["gen_h", "gen_h_meta"]].max(axis=1)
    out["gpu_h_floor"] = out["gen_h_floor"] + out["build_h"] + out["train_h"]
    out = out.sort_values(["arm", "iteration"]).reset_index(drop=True)
    out["cum_gpu_h_floor"] = out.groupby("arm")["gpu_h_floor"].cumsum()
    return out


def compute_by_arm_with_floor(comp_floor: pd.DataFrame) -> pd.DataFrame:
    """:func:`compute_summary` plus ``total_gpu_h_floor``, ``build_share``, ``train_share``.

    ``comp_floor`` is :func:`compute_by_iteration_with_floor` output (a plain
    :func:`iteration_compute` frame also works — the floor column then equals the headline).
    Reproduces ``compute_axis_by_arm.csv``.
    """
    summ = compute_summary(comp_floor)
    if summ.empty:
        return summ
    summ = summ.copy()
    col = "gpu_h_floor" if "gpu_h_floor" in comp_floor.columns else "gpu_h"
    ff = comp_floor[comp_floor.iteration > 0].groupby("arm")[col].sum()
    summ["total_gpu_h_floor"] = summ["arm"].map(ff)
    summ["build_share"] = summ["build_h"] / summ["total_gpu_h"]
    summ["train_share"] = summ["train_h"] / summ["total_gpu_h"]
    return summ


#: ``name -> (numerator arm, denominator arm, column)`` for :func:`cost_ratios`.
_RATIOS = {
    "GRPO_per_iter_K5_over_K0": ("GRPO_LA5", "GRPO_LA0", "gpu_h_per_iter"),
    "PTO_per_iter_K5_over_K0": ("PTO_LA5", "PTO_LA0", "gpu_h_per_iter"),
    "GRPO_over_PTO_per_iter_K0": ("GRPO_LA0", "PTO_LA0", "gpu_h_per_iter"),
    "GRPO_over_PTO_per_iter_K5": ("GRPO_LA5", "PTO_LA5", "gpu_h_per_iter"),
    "GRPO_LA0_total_over_PTO_LA0_total": ("GRPO_LA0", "PTO_LA0", "total_gpu_h"),
    "GRPO_LA5_total_over_PTO_LA5_total": ("GRPO_LA5", "PTO_LA5", "total_gpu_h"),
    "PTO_LA5_total_over_PTO_LA0_total": ("PTO_LA5", "PTO_LA0", "total_gpu_h"),
    # Renamed 2026-08-25. The key was "GRPO_LA5_5iters_over_GRPO_LA0_10iters", which asserted an
    # iteration count IN THE LEDGER KEY — and kept asserting it after GRPO_LA5 finished at 10, so
    # the rendered compute_numbers.json named "5iters" over a value where BOTH sides are
    # 10-iteration totals. Key names must not carry facts that can go stale; the iteration counts
    # are columns of compute_by_arm.
    "GRPO_LA5_total_over_GRPO_LA0_total": ("GRPO_LA5", "GRPO_LA0", "total_gpu_h"),
}
#: The three totals-ratios re-read under the generation floor (same keys as the paper ledger's
#: ``ratio_floor.*``).
_RATIOS_FLOOR = {
    "GRPO_LA0_total_over_PTO_LA0_total": ("GRPO_LA0", "PTO_LA0", "total_gpu_h_floor"),
    "PTO_LA5_total_over_PTO_LA0_total": ("PTO_LA5", "PTO_LA0", "total_gpu_h_floor"),
    "GRPO_LA5_total_over_PTO_LA5_total": ("GRPO_LA5", "PTO_LA5", "total_gpu_h_floor"),
}


def cost_ratios(summ: pd.DataFrame) -> pd.DataFrame:
    """The cost ratios the paper quotes, WITH their arithmetic (rule: a composite number shows
    its arithmetic wherever it is quoted). One row per ratio: ``name, kind (headline|floor),
    num_arm, den_arm, column, num, den, ratio, arithmetic``. Missing arms yield no row."""
    s = summ.set_index("arm")
    rows = []
    for kind, table in (("headline", _RATIOS), ("floor", _RATIOS_FLOOR)):
        for name, (na, nb, col) in table.items():
            if col not in s.columns or na not in s.index or nb not in s.index:
                continue
            a, b = float(s.loc[na, col]), float(s.loc[nb, col])
            rows.append({"name": name, "kind": kind, "num_arm": na, "den_arm": nb, "column": col,
                         "num": a, "den": b, "ratio": a / b if b else np.nan,
                         "arithmetic": f"{a:.3f}/{b:.3f}={a / b:.2f}" if b else ""})
    return pd.DataFrame(rows)


# ── 2. the per-iteration price of look-ahead ─────────────────────────────────────────────────
def step_multiplier_table(comp: pd.DataFrame) -> pd.DataFrame:
    """GRPO's per-step K5/K0 ratio beside PTO's build-phase and whole-iteration ratios.

    GRPO's look-ahead cost lands INSIDE the optimizer step (5 extra simulated turns per
    candidate), so :func:`step_multiplier` on the median step is the right unit there. PTO's DPO
    step carries no look-ahead (ratio ~1); its look-ahead cost lands in the pref-tree BUILD
    phase, so the build_h ratio and the whole-iteration gpu_h ratio are shown instead. Columns::

        iteration, GRPO_median_step_s_K0, GRPO_median_step_s_K5, GRPO_step_ratio_K5_over_K0,
        PTO_dpo_median_step_s_K0, PTO_dpo_median_step_s_K5, PTO_dpo_step_ratio,
        PTO_build_h_K0, PTO_build_h_K5, PTO_build_ratio_K5_over_K0,
        PTO_iter_gpu_h_K0, PTO_iter_gpu_h_K5, PTO_iter_ratio_K5_over_K0

    ⚠ Iteration 1 of GRPO_LA5 ran at ``LOOKAHEAD_SUB_BATCH_SIZE=64`` with a fat API-latency tail
    (ratio 2.41), so quote the settled later iterations (~1.9x); GRPO_LA5 has no rows past its
    last trained iteration (right-censored — the frame's ``iteration`` column says where).
    Reproduces ``compute_axis_step_multiplier.csv``.
    """
    sm = step_multiplier(comp, "GRPO")
    if sm.empty:
        return sm
    sm = sm.rename(columns={"median_s_K0": "GRPO_median_step_s_K0",
                            "median_s_K5": "GRPO_median_step_s_K5",
                            "ratio_median": "GRPO_step_ratio_K5_over_K0"}).set_index("iteration")
    pto = comp[(comp.method == "PTO") & (comp.iteration > 0)]
    if not pto.empty:
        pv = pto.pivot_table(index="iteration", columns="arm",
                             values=["build_h", "gpu_h", "median_step_s"])

        def col(val, arm):
            return pv[(val, arm)] if (val, arm) in pv.columns else np.nan

        sm["PTO_dpo_median_step_s_K0"] = col("median_step_s", "PTO_LA0")
        sm["PTO_dpo_median_step_s_K5"] = col("median_step_s", "PTO_LA5")
        sm["PTO_dpo_step_ratio"] = sm["PTO_dpo_median_step_s_K5"] / sm["PTO_dpo_median_step_s_K0"]
        sm["PTO_build_h_K0"] = col("build_h", "PTO_LA0")
        sm["PTO_build_h_K5"] = col("build_h", "PTO_LA5")
        sm["PTO_build_ratio_K5_over_K0"] = sm["PTO_build_h_K5"] / sm["PTO_build_h_K0"]
        sm["PTO_iter_gpu_h_K0"] = col("gpu_h", "PTO_LA0")
        sm["PTO_iter_gpu_h_K5"] = col("gpu_h", "PTO_LA5")
        sm["PTO_iter_ratio_K5_over_K0"] = sm["PTO_iter_gpu_h_K5"] / sm["PTO_iter_gpu_h_K0"]
    sm = sm.reset_index()
    sm["iteration"] = sm["iteration"].astype(int)
    return sm


# ── 3. budget sweeps ─────────────────────────────────────────────────────────────────────────
def _means_names(scores: pd.DataFrame, metric: str):
    sub = scores[scores["questionnaire"] == metric]
    means = sub.groupby(["arm", "iteration"])["score"].mean()
    names = {(a, int(i)): m for a, i, m in
             sub[["arm", "iteration", "model"]].drop_duplicates().itertuples(index=False)}
    return means, names


def _best_within(arm: str, budget: float, cum: pd.Series, means: pd.Series, lower_better: bool):
    elig = [int(i) for i, c in cum.items() if c <= budget + 1e-9 and (arm, int(i)) in means.index]
    if not elig:
        return None
    key = (lambda i: -means.loc[(arm, i)]) if lower_better else (lambda i: means.loc[(arm, i)])
    return max(elig, key=key)


def budget_sweep_ci(eval_scores: pd.DataFrame, comp: pd.DataFrame, arm_a: str, arm_b: str, *,
                    select_scores: Optional[pd.DataFrame] = None,
                    select_metric: str = "Q1Q2", eval_metric: str = "Q1Q2",
                    budgets: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """:func:`budget_sweep` with three additions (the paper's ``sweep``):

    (i) the checkpoint is SELECTED on ``select_scores``/``select_metric`` and the contrast SCORED
    on ``eval_scores``/``eval_metric`` — same frame + metric = the tracked sweep row-for-row;
    a different frame = a cross-judge (honest) selection; a different metric = "does the
    reward-selected policy carry the hack?" (``Q1Q2 -> MICI``);
    (ii) LOWER_IS_BETTER select metrics pick the MINIMUM;
    (iii) a bootstrap 95% CI (:func:`~eda_analysis.stats.paired_arrays`), the selected model
    names, ``cum_gpu_h_a/b`` and a Holm ``p_holm`` over the UNIQUE checkpoint pairs of this
    table (repeated budgets re-use one pair; correcting per row would over-count one test) are
    returned, with ``n_unique_pairs``.

    ``+ mean_delta => arm_a higher``. Pairs on ``persona_id``. ``budgets`` defaults to arm_a's
    cumulative budgets (every trained iteration). ``select_scores=None`` = ``eval_scores``.
    """
    from .stats import paired_arrays, holm
    if select_scores is None:
        select_scores = eval_scores
    A = comp[(comp.arm == arm_a) & (comp.iteration > 0)].set_index("iteration")["cum_gpu_h"]
    B = comp[(comp.arm == arm_b) & (comp.iteration > 0)].set_index("iteration")["cum_gpu_h"]
    if A.empty or B.empty:
        return pd.DataFrame()
    sel_means, _ = _means_names(select_scores, select_metric)
    ev_means, ev_names = _means_names(eval_scores, eval_metric)
    lower = select_metric in LOWER_IS_BETTER
    W = _wide(eval_scores, eval_metric)
    rows = []
    for budget in (sorted(A.values) if budgets is None else budgets):
        ia = _best_within(arm_a, budget, A, sel_means, lower)
        ib = _best_within(arm_b, budget, B, sel_means, lower)
        if ia is None or ib is None:
            continue
        ma, mb = ev_names.get((arm_a, ia)), ev_names.get((arm_b, ib))
        if ma is None or mb is None or ma not in W.columns or mb not in W.columns:
            continue
        st = paired_arrays(W[ma].to_numpy(), W[mb].to_numpy())
        rows.append({"budget_gpu_h": round(float(budget), 2),
                     "select_metric": select_metric, "eval_metric": eval_metric,
                     "best_iter_a": ia, "best_iter_b": ib,
                     "cum_gpu_h_a": round(float(A.loc[ia]), 2), "cum_gpu_h_b": round(float(B.loc[ib]), 2),
                     "model_a": ma, "model_b": mb,
                     "mean_a": float(ev_means.loc[(arm_a, ia)]), "mean_b": float(ev_means.loc[(arm_b, ib)]),
                     "n": st["n"], "mean_delta": st["mean_delta"], "dz": st["dz"],
                     "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"], "p": st["p"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    key = df["best_iter_a"].astype(str) + "/" + df["best_iter_b"].astype(str)
    uniq = df.drop_duplicates(subset=["best_iter_a", "best_iter_b"])
    ph = dict(zip(uniq["best_iter_a"].astype(str) + "/" + uniq["best_iter_b"].astype(str),
                  holm(uniq["p"].values)))
    df["p_holm"] = key.map(ph)
    df["n_unique_pairs"] = len(uniq)
    df["arm_a"], df["arm_b"] = arm_a, arm_b
    return df


def add_k_convention(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """For a ``*_K`` contrast add ``delta_K0_minus_K5 = -mean_delta`` and ``dz_K0_minus_K5 = -dz``
    (the paper's ``+ => K=0 higher`` reading beside the tracked ``K5 - K0``). No-op otherwise."""
    if tag.endswith("_K") and df is not None and not df.empty:
        df = df.copy()
        df["delta_K0_minus_K5"] = -df["mean_delta"]
        df["dz_K0_minus_K5"] = -df["dz"]
    return df


#: ``(select_metric, eval_metric)`` variants stacked in each same-judge sweep table.
SWEEP_VARIANTS = (("Q1Q2", "Q1Q2"), ("MICI", "MICI"), ("Q1Q2", "MICI"))

_SWEEP_COLS = ["judge", "budget_gpu_h", "select_metric", "eval_metric", "best_iter_a", "best_iter_b",
               "model_a", "model_b", "cum_gpu_h_a", "cum_gpu_h_b", "mean_a", "mean_b", "n",
               "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "n_unique_pairs"]


def all_budget_sweeps(scores_by_judge: Dict[str, pd.DataFrame], comp: pd.DataFrame,
                      contrasts: Sequence[Tuple[str, str, str, str]] = CONTRASTS, *,
                      variants: Sequence[Tuple[str, str]] = SWEEP_VARIANTS,
                      check_against_tracked: bool = True
                      ) -> Dict[Tuple[str, str], pd.DataFrame]:
    """The same-judge budget-sweep tables: ``{(contrast_tag, judge_label): DataFrame}``.

    One table per (contrast, judge) — 4 contrasts x 2 graders = the paper's 8
    ``compute_axis_budget_sweep_<contrast>_<judge>`` tables — each stacking the ``variants``
    (Q1Q2->Q1Q2 = the tracked sweep; MICI->MICI selects the LOWEST; Q1Q2->MICI scores the
    reward-selected checkpoints on MICI). Columns: :data:`_SWEEP_COLS` (+ ``arm_a``, ``arm_b``;
    + ``delta_K0_minus_K5``/``dz_K0_minus_K5`` for K contrasts).

    ``check_against_tracked`` asserts the Q1Q2->Q1Q2 rows equal :func:`budget_sweep` (the
    tracked EDA table) on ``mean_delta``/``dz``/selected iterations — the guard that the two
    code paths never drift.
    """
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for jl, SC in scores_by_judge.items():
        for tag, a, b, _lab in contrasts:
            if not {a, b} <= set(SC["arm"].unique()):
                continue
            parts = [budget_sweep_ci(SC, comp, a, b, select_metric=sm, eval_metric=em)
                     for sm, em in variants]
            parts = [p for p in parts if not p.empty]
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            df.insert(0, "judge", jl)
            df = add_k_convention(df, tag)
            if check_against_tracked and ("Q1Q2", "Q1Q2") in tuple(variants):
                ref = budget_sweep(SC, comp, a, b, metric="Q1Q2")
                mine = df[(df.select_metric == "Q1Q2") & (df.eval_metric == "Q1Q2")].reset_index(drop=True)
                assert len(ref) == len(mine), (jl, tag, len(ref), len(mine))
                assert np.allclose(ref["mean_delta"].values, mine["mean_delta"].values, atol=1e-9)
                assert np.allclose(ref["dz"].values, mine["dz"].values, atol=1e-9)
                assert (ref["best_iter_a"].values == mine["best_iter_a"].values).all()
                assert (ref["best_iter_b"].values == mine["best_iter_b"].values).all()
            cols = _SWEEP_COLS + (["delta_K0_minus_K5", "dz_K0_minus_K5"] if tag.endswith("_K") else [])
            out[(tag, jl)] = df[cols + ["arm_a", "arm_b"]].reset_index(drop=True)
    return out


def budget_sweep_top(sweeps: Dict[Tuple[str, str], pd.DataFrame], *,
                     select_metric: str = "Q1Q2", eval_metric: str = "Q1Q2") -> pd.DataFrame:
    """Top-of-sweep verdict per (contrast, judge): the LAST row of the ``select->eval`` variant
    (arm_a's largest budget), plus ``sign_flips_within_sweep`` — whether ``mean_delta`` changed
    sign anywhere along the curve (quote the curve, not the endpoint, when it did)."""
    rows = []
    for (tag, jl), df in sweeps.items():
        d = df[(df.select_metric == select_metric) & (df.eval_metric == eval_metric)]
        if d.empty:
            continue
        r = d.iloc[-1]
        rows.append({"judge": jl, "contrast": tag, "arm_a": r.arm_a, "arm_b": r.arm_b,
                     "budget_gpu_h": r.budget_gpu_h, "best_iter_a": r.best_iter_a, "best_iter_b": r.best_iter_b,
                     "mean_delta": r.mean_delta, "dz": r.dz, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                     "p": r.p, "p_holm": r.p_holm,
                     "sign_flips_within_sweep": bool((np.sign(d["mean_delta"]) != np.sign(r.mean_delta)).any())})
    return pd.DataFrame(rows)


_XJ_COLS = ["contrast", "arm_a", "arm_b", "select_judge", "eval_judge", "honest_selection", "budget_gpu_h",
            "best_iter_a", "best_iter_b", "model_a", "model_b", "mean_a", "mean_b", "n",
            "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "delta_K0_minus_K5", "dz_K0_minus_K5"]


def budget_sweep_crossjudge(scores_by_judge: Dict[str, pd.DataFrame], comp: pd.DataFrame,
                            contrasts: Sequence[Tuple[str, str, str, str]] = CONTRASTS, *,
                            metric: str = "Q1Q2") -> pd.DataFrame:
    """Cross-judge selection sweep on ``metric``: every (select_judge, eval_judge) combination.

    Each arm's best-within-budget checkpoint is SELECTED on ``select_judge``'s means and the
    paired contrast is SCORED on ``eval_judge``'s scores. ``honest_selection = select != eval``
    — the grader that picked the checkpoint is not the grader that scores it; the same-judge
    rows reproduce the per-judge sweep tables and are optimistic for the selecting grader.
    Holm within each (contrast, select_judge, eval_judge) family over unique checkpoint pairs.
    ``delta_K0_minus_K5``/``dz_K0_minus_K5`` are blank (NaN) for method contrasts.
    Reproduces ``compute_axis_budget_sweep_crossjudge.csv``.
    """
    rows = []
    for tag, a, b, _lab in contrasts:
        A_budgets = np.sort(comp[(comp.arm == a) & (comp.iteration > 0)]["cum_gpu_h"].values)
        for sj, SS in scores_by_judge.items():
            for ej, ES in scores_by_judge.items():
                if not ({a, b} <= set(SS["arm"].unique()) and {a, b} <= set(ES["arm"].unique())):
                    continue
                d = budget_sweep_ci(ES, comp, a, b, select_scores=SS, select_metric=metric,
                                    eval_metric=metric, budgets=A_budgets)
                if d.empty:
                    continue
                d.insert(0, "eval_judge", ej)
                d.insert(0, "select_judge", sj)
                d.insert(0, "contrast", tag)
                d["honest_selection"] = sj != ej
                rows.append(add_k_convention(d, tag))
    if not rows:
        return pd.DataFrame()
    xj = pd.concat(rows, ignore_index=True)
    for c in ("delta_K0_minus_K5", "dz_K0_minus_K5"):
        if c not in xj.columns:
            xj[c] = np.nan
    return xj[_XJ_COLS + [c for c in xj.columns if c not in _XJ_COLS]]


def crossjudge_verdicts(xj: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Top-of-sweep verdicts (each contrast at arm_a's LAST budget) under every
    (select_judge, eval_judge) combination of :func:`budget_sweep_crossjudge`. A verdict that
    holds only when the same grader selects and scores is a selection artefact; the
    ``honest_selection`` rows are the ones to quote. ``verdict`` uses ``p_holm < alpha``.
    Reproduces ``compute_axis_budget_sweep_crossjudge_verdicts.csv``."""
    if xj is None or xj.empty:
        return pd.DataFrame()
    verd = (xj.sort_values("budget_gpu_h").groupby(["contrast", "select_judge", "eval_judge"], sort=False)
            .tail(1).sort_values(["contrast", "select_judge", "eval_judge"]))
    verd = verd[["contrast", "arm_a", "arm_b", "select_judge", "eval_judge", "honest_selection", "budget_gpu_h",
                 "best_iter_a", "best_iter_b", "n", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]].copy()
    verd["verdict"] = np.where(verd["p_holm"] < alpha,
                               np.where(verd["mean_delta"] > 0, "arm_a > arm_b", "arm_a < arm_b"),
                               "no sig. difference")
    return verd.reset_index(drop=True)


# ── 4. K at matched compute on the behaviour channels ────────────────────────────────────────
_ISO_COLS = ["contrast", "judge", "metric", "channel", "direction", "iter_a", "iter_b", "cum_gpu_h_a",
             "cum_gpu_h_b", "budget_ratio", "iso_ok", "model_a", "model_b", "n", "mean_delta", "dz",
             "ci_lo", "ci_hi", "p", "p_holm", "delta_K0_minus_K5", "dz_K0_minus_K5"]


def iso_channels(channels_by_judge: Dict[str, pd.DataFrame], comp: pd.DataFrame,
                 contrasts: Sequence[Tuple[str, str, str, str]] = K_CONTRASTS, *,
                 channels: Sequence[str] = CHANNELS,
                 iso_band: Tuple[float, float] = (0.9, 1.1)) -> pd.DataFrame:
    """Look-ahead at MATCHED compute on the behaviour channels, both graders side by side.

    ``channels_by_judge`` = ``{judge_label: behavior.channel_scores_long(arms)}`` (MICI_* and
    B6_AF are grader-coded; the text channels are not). For every trained iteration of the K=5
    arm (arm_a) the K=0 iteration of the same method with the closest cumulative GPU-h is paired
    on ``persona_id`` via :func:`iso_compute_contrast` (``budget_ratio = b/a``; ``iso_ok`` flags
    the ``iso_band`` — PTO_LA0 tops out at ~8.12 GPU-h so PTO_LA5 iters >= 5 have no iso partner
    and are flagged False). Bootstrap CIs are added from :func:`~eda_analysis.stats.paired_arrays`.
    ``mean_delta = K5 - K0`` on the channel's own unit; ``delta_K0_minus_K5 = -mean_delta``.
    ``direction`` says how to read the sign (:func:`channel_direction`). Text channels are
    grader-independent and reported ONCE, under ``judge = "text (grader-independent)"`` (taken
    from the primary's frame; dropped from every other judge). Holm within the channel family at
    one budget pair. Never averages the two graders. Reproduces ``compute_axis_iso_channels.csv``.
    """
    from .stats import paired_arrays
    prim = _primary_label(channels_by_judge.keys())
    parts = []
    for jl, CH in channels_by_judge.items():
        CH = CH[CH["questionnaire"].isin(list(channels))]
        for tag, a, b, _lab in contrasts:
            d = iso_compute_contrast(CH, comp, a, b, metrics=list(channels))
            if d.empty:
                continue
            if jl != prim:
                d = d[~d["metric"].isin(TEXT_CHANNELS)]
            if d.empty:
                continue
            d = d.copy()
            d.insert(0, "judge", np.where(d["metric"].isin(TEXT_CHANNELS), TEXT_JUDGE_LABEL, jl))
            d.insert(0, "contrast", tag)
            W = {m: _wide(CH, m) for m in d["metric"].unique()}
            ci = [paired_arrays(W[r.metric][r.model_a].to_numpy(), W[r.metric][r.model_b].to_numpy())
                  for r in d.itertuples(index=False)]
            d["ci_lo"] = [c["ci_lo"] for c in ci]
            d["ci_hi"] = [c["ci_hi"] for c in ci]
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    iso = pd.concat(parts, ignore_index=True)
    iso["channel"] = iso["metric"].map(_label_of)
    iso["direction"] = iso["metric"].map(channel_direction)
    iso["delta_K0_minus_K5"] = -iso["mean_delta"]
    iso["dz_K0_minus_K5"] = -iso["dz"]
    iso["iso_ok"] = iso["budget_ratio"].between(*iso_band)
    return iso[_ISO_COLS].reset_index(drop=True)


_SEL_COLS = ["contrast", "judge", "selected_on", "budget_gpu_h", "metric", "channel", "direction", "iter_a",
             "iter_b", "model_a", "model_b", "mean_a", "mean_b", "n", "mean_delta", "dz", "ci_lo", "ci_hi",
             "p", "p_holm", "delta_K0_minus_K5", "dz_K0_minus_K5"]


def iso_channels_selected(channels_by_judge: Dict[str, pd.DataFrame],
                          sweeps: Dict[Tuple[str, str], pd.DataFrame],
                          contrasts: Sequence[Tuple[str, str, str, str]] = K_CONTRASTS, *,
                          channels: Sequence[str] = CHANNELS,
                          select_metric: str = "Q1Q2", eval_metric: str = "Q1Q2") -> pd.DataFrame:
    """The channels at the checkpoints an operator would actually DEPLOY.

    For each method, the K=5 (arm_a) and K=0 (arm_b) checkpoints selected as best-within-budget
    on ``select_metric`` under the named grader at the TOP budget of the K sweep (the last
    ``select->eval`` row of ``sweeps[(tag, judge)]`` from :func:`all_budget_sweeps`), contrasted
    on each channel paired on ``persona_id`` (bootstrap 95% CI, Wilcoxon p, Holm within the
    channel family per (contrast, judge)). ``mean_delta = K5 - K0``; ``delta_K0_minus_K5`` is
    the paper's convention; text channels reported once (primary's frame).
    Reproduces ``compute_axis_iso_channels_selected.csv``.
    """
    from .stats import paired_arrays, holm
    prim = _primary_label(channels_by_judge.keys())
    rows = []
    for jl, CH in channels_by_judge.items():
        for tag, a, b, _lab in contrasts:
            d = sweeps.get((tag, jl))
            if d is None or d.empty:
                continue
            d = d[(d.select_metric == select_metric) & (d.eval_metric == eval_metric)]
            if d.empty:
                continue
            top = d.iloc[-1]
            ma, mb = top.model_a, top.model_b
            for m in channels:
                if jl != prim and m in TEXT_CHANNELS:
                    continue
                W = _wide(CH, m)
                if ma not in W.columns or mb not in W.columns:
                    continue
                st = paired_arrays(W[ma].to_numpy(), W[mb].to_numpy())
                rows.append({"contrast": tag,
                             "judge": (TEXT_JUDGE_LABEL if m in TEXT_CHANNELS else jl),
                             "selected_on": f"{select_metric} ({jl}) best-within-budget",
                             "budget_gpu_h": top.budget_gpu_h,
                             "metric": m, "channel": _label_of(m), "direction": channel_direction(m),
                             "iter_a": int(top.best_iter_a), "iter_b": int(top.best_iter_b),
                             "model_a": ma, "model_b": mb,
                             "mean_a": float(np.nanmean(W[ma])), "mean_b": float(np.nanmean(W[mb])), **st})
    if not rows:
        return pd.DataFrame()
    sel = pd.DataFrame(rows)
    sel["p_holm"] = np.nan
    for _, idx in sel.groupby(["contrast", "judge"]).groups.items():
        sel.loc[idx, "p_holm"] = holm(sel.loc[idx, "p"].values)
    sel["delta_K0_minus_K5"] = -sel["mean_delta"]
    sel["dz_K0_minus_K5"] = -sel["dz"]
    return sel[_SEL_COLS].reset_index(drop=True)


# ── 5. the quotable-numbers ledger ───────────────────────────────────────────────────────────
CAVEATS = [
    "All GPU-hours are mtime-reconstructed (eda_analysis.compute); never quote iteration_metadata.json timings.",
    "PTO_LA5 iters 1-5 show gen_h ~0.000 because their conversation CSV mtimes were batch-flushed; the time "
    "lands in iter 6 (0.967 h). Cumulative totals are right; per-iteration gen splits are not, for that arm. "
    "gen_h is a systematic UNDER-estimate (~0.1 h/iter: the mtime span misses the first batch of 64); gen_h_floor / "
    "cum_gpu_h_floor / total_gpu_h_floor use max(mtime, recorded generation_time_s) without changing the headline; "
    "read each arm's headline-vs-floor totals off the *_floor columns themselves, never from prose.",
    "Each arm's budget ceiling is its own last cum_gpu_h in compute_by_iteration, so a sweep reaches only the "
    "checkpoints that arm's spend bought - read the ceiling off that column, never from prose.",
    "Iso-compute pairs different iterations across arms; pairing is on persona_id.",
    "Quote budget_sweep rows, not a single iso-compute row: the sign of the K lever depends on budget.",
    "K-contrast tables carry mean_delta = K5 - K0 (tracked-EDA convention) AND delta_K0_minus_K5 (paper convention).",
    "Same-judge best-within-budget selection is optimistic for the selecting grader; read the crossjudge tables.",
]


def compute_numbers(comp_floor: pd.DataFrame, summ: pd.DataFrame, sm: pd.DataFrame,
                    sweeps: Dict[Tuple[str, str], pd.DataFrame], xj: pd.DataFrame,
                    verd: pd.DataFrame, iso: pd.DataFrame, sel: pd.DataFrame, *,
                    source_prefix: str = "tables/", name_prefix: str = "",
                    settled_iters: Sequence[int] = (3, 4, 5)) -> Dict[str, dict]:
    """Every number the write-up may quote, as ``{dotted_key: {"value","source","note"}}`` —
    the shape ``exports.save_numbers`` writes and the paper's ``out/compute_axis.json`` ledger
    used (same keys, so the frozen ledger diffs against it key-for-key; ``figures.*`` are the
    notebook's to add). Sources are ``f"{source_prefix}{name_prefix}<table>.md"`` — pass
    ``name_prefix="compute_axis_"`` to reproduce the paper's paths.

    Inputs are the frames of this module: :func:`compute_by_iteration_with_floor`,
    :func:`compute_by_arm_with_floor`, :func:`step_multiplier_table`, :func:`all_budget_sweeps`,
    :func:`budget_sweep_crossjudge`, :func:`crossjudge_verdicts`, :func:`iso_channels`,
    :func:`iso_channels_selected`. Any may be empty/None — its keys are then simply absent.
    """
    L: Dict[str, dict] = {}

    def put(key, value, *, source="", note=""):
        L[key] = {"value": value, "source": source, "note": note}

    def src(name):
        return f"{source_prefix}{name_prefix}{name}.md"

    def rf(x, nd=3):
        return _round_or_none(x, nd)

    # -- ratios with arithmetic + per-arm + per-iteration -------------------------------------
    if summ is not None and not summ.empty:
        for r in cost_ratios(summ).itertuples(index=False):
            key = ("ratio." if r.kind == "headline" else "ratio_floor.") + r.name
            put(key, {"num": rf(r.num), "den": rf(r.den), "ratio": rf(r.ratio), "arithmetic": r.arithmetic},
                source=src("by_arm") + (" (total_gpu_h_floor)" if r.kind == "floor" else ""),
                note=("" if r.kind == "headline" else
                      "generation floor applied (max of mtime span and recorded generation_time_s); "
                      "headline ratios are ratio.*"))
        by_arm_cols = ["last_iter", "n_iters", "gen_h", "build_h", "train_h", "total_gpu_h",
                       "gpu_h_per_iter", "median_step_s", "n_imputed", "build_share", "train_share",
                       "total_gpu_h_floor"]
        for r in summ.itertuples(index=False):
            put(f"by_arm.{r.arm}",
                {c: (rf(getattr(r, c)) if isinstance(getattr(r, c), (float, np.floating)) else _round_or_none(getattr(r, c)))
                 for c in by_arm_cols if hasattr(r, c)},
                source=src("by_arm"))
    if comp_floor is not None and not comp_floor.empty:
        for r in comp_floor[comp_floor.iteration > 0].itertuples(index=False):
            put(f"by_iteration.{r.arm}.I{int(r.iteration)}",
                {"gen_h": rf(r.gen_h), "build_h": rf(r.build_h), "train_h": rf(r.train_h),
                 "gpu_h": rf(r.gpu_h), "cum_gpu_h": rf(r.cum_gpu_h), "n_steps": int(r.n_steps),
                 "median_step_s": rf(r.median_step_s), "n_imputed": int(r.n_imputed),
                 "gen_h_meta": rf(getattr(r, "gen_h_meta", np.nan)),
                 "gen_h_floor": rf(getattr(r, "gen_h_floor", np.nan)),
                 "cum_gpu_h_floor": rf(getattr(r, "cum_gpu_h_floor", np.nan))},
                source=src("by_iteration"))

    # -- step multiplier ---------------------------------------------------------------------
    if sm is not None and not sm.empty:
        for r in sm.itertuples(index=False):
            put(f"step_multiplier.I{int(r.iteration)}",
                {"GRPO_median_step_s_K0": rf(r.GRPO_median_step_s_K0),
                 "GRPO_median_step_s_K5": rf(r.GRPO_median_step_s_K5),
                 "GRPO_step_ratio": rf(r.GRPO_step_ratio_K5_over_K0),
                 "PTO_build_ratio": rf(getattr(r, "PTO_build_ratio_K5_over_K0", np.nan)),
                 "PTO_iter_ratio": rf(getattr(r, "PTO_iter_ratio_K5_over_K0", np.nan)),
                 "PTO_dpo_step_ratio": rf(getattr(r, "PTO_dpo_step_ratio", np.nan))},
                source=src("step_multiplier"))
        settled = sm[sm.iteration.isin(list(settled_iters))]["GRPO_step_ratio_K5_over_K0"].dropna()
        if not settled.empty:
            put(f"step_multiplier.GRPO_settled_iters_{settled_iters[0]}_{settled_iters[-1]}",
                {"median_ratio": rf(settled.median()), "values": [rf(x) for x in settled]},
                source=src("step_multiplier"))
        if "PTO_build_ratio_K5_over_K0" in sm.columns:
            pb = sm["PTO_build_ratio_K5_over_K0"].dropna()
            if not pb.empty:
                put("step_multiplier.PTO_build_ratio_all_iters",
                    {"median": rf(pb.median()), "min": rf(pb.min()), "max": rf(pb.max())},
                    source=src("step_multiplier"))

    # -- same-judge sweeps + top verdicts ----------------------------------------------------
    if sweeps:
        for (tag, jl), df in sweeps.items():
            for r in df.itertuples(index=False):
                put(f"sweep.{jl}.{tag}.{r.select_metric}_to_{r.eval_metric}.budget_{r.budget_gpu_h:g}h",
                    {"best_iter_a": int(r.best_iter_a), "best_iter_b": int(r.best_iter_b),
                     "model_a": r.model_a, "model_b": r.model_b,
                     "mean_a": rf(r.mean_a), "mean_b": rf(r.mean_b),
                     "n": int(r.n), "mean_delta": rf(r.mean_delta), "dz": rf(r.dz),
                     "ci": [rf(r.ci_lo), rf(r.ci_hi)],
                     "p": rf(r.p, 4), "p_holm": rf(r.p_holm, 4),
                     **({"delta_K0_minus_K5": rf(-r.mean_delta)} if tag.endswith("_K") else {})},
                    source=src(f"budget_sweep_{tag}_{jl}"),
                    note=("mean_delta = arm_a - arm_b; " + sign_note(tag)))
        top = budget_sweep_top(sweeps)
        for r in top.itertuples(index=False):
            put(f"sweep_top.{r.judge}.{r.contrast}",
                {k: (rf(v) if isinstance(v, (float, np.floating)) else
                     (int(v) if isinstance(v, (np.integer,)) else v))
                 for k, v in r._asdict().items()},
                source=src(f"budget_sweep_{r.contrast}_{r.judge}") + " (last row, Q1Q2->Q1Q2)")

    # -- cross-judge --------------------------------------------------------------------------
    if verd is not None and not verd.empty:
        for r in verd.itertuples(index=False):
            put(f"crossjudge_verdict.{r.contrast}.select_{r.select_judge}.eval_{r.eval_judge}",
                {"budget_gpu_h": float(r.budget_gpu_h), "best_iter_a": int(r.best_iter_a),
                 "best_iter_b": int(r.best_iter_b), "mean_delta": rf(r.mean_delta), "dz": rf(r.dz),
                 "ci": [rf(r.ci_lo), rf(r.ci_hi)], "p": rf(r.p, 4), "p_holm": rf(r.p_holm, 4),
                 "verdict": r.verdict, "honest_selection": bool(r.honest_selection)},
                source=src("budget_sweep_crossjudge_verdicts"))
    if xj is not None and not xj.empty:
        for r in xj[xj.honest_selection].itertuples(index=False):
            put(f"crossjudge.{r.contrast}.select_{r.select_judge}.eval_{r.eval_judge}.budget_{r.budget_gpu_h:g}h",
                {"best_iter_a": int(r.best_iter_a), "best_iter_b": int(r.best_iter_b),
                 "mean_delta": rf(r.mean_delta), "dz": rf(r.dz), "ci": [rf(r.ci_lo), rf(r.ci_hi)],
                 "p": rf(r.p, 4), "p_holm": rf(r.p_holm, 4)},
                source=src("budget_sweep_crossjudge"))

    # -- channels at matched compute / at the deployed checkpoints ----------------------------
    if iso is not None and not iso.empty:
        for r in iso.itertuples(index=False):
            put(f"iso_channels.{r.contrast}.{r.judge.split(' ')[0]}.{r.metric}.iterA{int(r.iter_a)}_iterB{int(r.iter_b)}",
                {"cum_gpu_h_a": float(r.cum_gpu_h_a), "cum_gpu_h_b": float(r.cum_gpu_h_b),
                 "budget_ratio": float(r.budget_ratio), "n": int(r.n),
                 "mean_delta_K5_minus_K0": rf(r.mean_delta), "dz": rf(r.dz),
                 "ci": [rf(r.ci_lo), rf(r.ci_hi)], "p": rf(r.p, 4), "p_holm": rf(r.p_holm, 4),
                 "direction": r.direction, "iso_ok": bool(r.iso_ok)},
                source=src("iso_channels"))
    if sel is not None and not sel.empty:
        for r in sel.itertuples(index=False):
            put(f"iso_channels_selected.{r.contrast}.{r.judge.split(' ')[0]}.{r.metric}",
                {"iter_a": int(r.iter_a), "iter_b": int(r.iter_b), "budget_gpu_h": float(r.budget_gpu_h),
                 "mean_a_K5": rf(r.mean_a), "mean_b_K0": rf(r.mean_b), "n": int(r.n),
                 "mean_delta_K5_minus_K0": rf(r.mean_delta), "dz": rf(r.dz),
                 "ci": [rf(r.ci_lo), rf(r.ci_hi)], "p": rf(r.p, 4), "p_holm": rf(r.p_holm, 4),
                 "direction": r.direction},
                source=src("iso_channels_selected"))

    put("caveats", list(CAVEATS), source="eda_analysis.compute (promoted from compute_axis.py)")
    return L
