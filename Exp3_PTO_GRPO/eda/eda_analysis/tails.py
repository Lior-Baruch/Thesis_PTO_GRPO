"""tails.py — what the K-step look-ahead reward is actually scoring, plus the API-call axis.

Promoted 2026-08-18 from the look-ahead paper's generator
``papers/2026_lookahead_pto_grpo/analysis/tail_audit.py`` (its ``tables/tail_audit_*.csv`` and
``analysis/out/tail_audit.json`` are the frozen fixture these functions reproduce). Read by
``notebooks/lookahead/mechanism.ipynb`` (the tail audit) and ``notebooks/compute/cost.ipynb``
(the API-call tables); the figures live in :mod:`eda_analysis.plotting.tails`.

What it computes
----------------
Two things, both from training-time artifacts (no oracle calls):

1. **The tail audit** (K>0 arms only — they are the only ones with tails). Every candidate the
   K=5 arms scored during training carries the K simulated turns that were appended before the
   oracle saw it (``lookahead.tail`` in ``iteration_N/eda/generations.jsonl``, read through
   :func:`eda_analysis.training.load_generations` with ``keep_tail=True``, ONE ARM AT A TIME with
   the memo cleared between arms to bound memory). The audit asks how those tails terminate
   (:func:`tail_audit_by_iter`), what the therapist says inside them vs in the candidate itself
   (:func:`tail_cues_by_iter`), whether the tail's ending predicts the reward *within the group
   the update sees* (:func:`tail_within_group`) and the within-group reward by realized tail
   length (:func:`score_by_realized_turns`).
2. **API-call accounting** for ALL arms (:func:`api_calls`, :func:`api_ratio`): oracle calls,
   patient-simulator calls and a char-count token proxy per (arm, training iteration), read from
   ``generations.jsonl`` (a light second streaming pass that keeps the record-level fields the
   loader drops: prefix chars, oracle retries) plus the ``model_iter_*`` conversation CSVs.

Conventions
-----------
* **Pairing unit = the group**: GRPO's ``G=8`` siblings of one prompt (keyed by
  ``(train_iter, conversation_id, branch_id, epoch)`` — GRPO logs one record per group per epoch);
  PTO's ``M=8`` branches at one branch point (same key, ``epoch`` = NaN → -1). ⚠ PTO's
  ``branch_id`` is the trunk DEPTH, so ``conversation_id`` MUST be in the key.
* **Sign convention** of :func:`tail_within_group`'s ``delta_ee_minus_full``:
  ``+`` ⇒ the ended-early candidates score HIGHER than the full-tail candidates of the same group.
  (The package-wide K-contrast convention ``+`` ⇒ K=0 higher does not apply here — these are
  within-arm, within-group contrasts.)
* **Censoring**: GRPO_LA5 was stopped before the other arms — every per-iteration frame carries
  each arm as it is, and the ``pooled`` rows pool whatever iterations exist. State the censoring
  in captions, DERIVED off the frame
  (:func:`eda_analysis.plotting.tails._censor_note` / :func:`eda_analysis.constants.support_note`)
  — never as a written-down iteration.
* **Bootstraps** use :data:`eda_analysis.constants.BOOT_SEED` (the paper generator used seed 0
  with the same 1,000 draws, so CI *bounds* may differ from the frozen fixture in the third
  decimal; every mean / count / dz / p / rho reproduces exactly).
* Grader: the TRAINING oracle (gpt-4o-mini) — these are training-side numbers and are NOT
  judge-swappable.

Facts established while inspecting the rows (keep these; they are load-bearing)
------------------------------------------------------------------------------
* ``tail`` is the ORACLE-FORMAT transcript slice appended after the completion —
  ``"\\n\\n[PATIENT]: ...\\n\\n[THERAPIST]: ..."`` — starting with the patient's reply.
* ``realized_turns`` == number of ``[ROLE]:`` labels in the tail (0..K); ``ended_early`` ==
  ``realized_turns < K``. Odd counts end on a PATIENT turn, even (>0) on a THERAPIST turn.
* The literal ``"SESSION ENDED"`` marker never appears in a tail: ``reward.simulate_lookahead_batch``
  runs ``convs.handle_session_end`` which STRIPS the marker and keeps only the text before it.
  What survives is a **fingerprint** — the kept text ends in whitespace (the marker followed) —
  which in the eval CSVs separates patient-closed turns from ordinary ones at ~100% vs ~4%.
  :func:`tail_audit_by_iter` VALIDATES the fingerprint per row: ``wrapup_cue_rate_patient_closed``
  (share of fingerprinted tails whose last patient turn carries a wrap-up phrase) vs
  ``wrapup_cue_rate_full_open`` (the same share among full tails without the fingerprint).
* PTO candidates whose completion is EMPTY carry ``lookahead=None`` and ``score=None`` (never
  simulated, never scored) — dropped from the tail analysis (``n_not_simulated_dropped``) and they
  cost no calls. PTO_LA5 iteration 5 also has candidates WITH a tail but no score (an oracle-API
  incident) — ``n_unscored_dropped``.
* GRPO records also carry a ``phase == "eval"`` block (TRL's eval loop) — those candidates were
  scored (real calls) but produced no gradient; they are dropped from the training-signal
  analysis and KEPT in the API accounting (``n_candidates_eval_phase``).
* **Log coverage (GRPO only).** The EDA capture is per-PROCESS on the older iterations — a
  crashed+resumed iteration logged only its post-resume steps (GRPO_LA5 iters 1–2, GRPO_LA0 iters
  2/6/8). The ground truth for how many optimizer steps ran is one
  ``training/completions/*.parquet`` per step (:func:`grpo_steps`); ``log_coverage`` = logged
  groups / (steps × 16 groups per step), and the API-call counts are RESCALED by it. The tail-audit
  rows for a partial-log iteration describe the later part of that iteration (they are not
  rescaled — they are rates). PTO logs are complete (coverage 1.0).
* GRPO's ``chosen`` candidate = ``chosen_idx`` = the group argmax; PTO's = ``role == "chosen"`` —
  identical to argmax (checked).

Nothing here writes to disk: the notebook saves the frames / figures and the
:func:`tails_numbers` ledger (``exports.save_numbers``).
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .constants import BOOT_SEED, RE_AFFIRM, RE_EFFUSIVE
from .ledger import json_scalar, ledger_entry, round3  # noqa: E402,F401

__all__ = [
    "TailAudit", "tail_audit_frames", "tail_audit_by_iter", "tail_cues_by_iter",
    "tail_within_group", "score_by_realized_turns", "api_calls", "api_ratio",
    "eval_conv_stats", "grpo_steps", "stream_record_stats", "tails_numbers", "clear_tails_memo",
    "parse_tail", "end_reason", "tail_features",
    "SCOUT_EXPECTED", "GROUP_KEYS", "N_Q", "N_EVAL_RUBRICS", "GRPO_GROUPS_PER_STEP",
    "GRPO_CANDS_PER_STEP",
]

# ── constants ─────────────────────────────────────────────────────────────────
N_Q = 2                     # Q1 + Q2 → two oracle calls per scored candidate (training reward)
N_EVAL_RUBRICS = 8          # Run_Eval scores every model state on 8 instruments (per grader)
N_BOOT = 1000               # the paper generator's draw count (kept; seed = BOOT_SEED)
GROUP_KEYS = ["train_iter", "conversation_id", "branch_id", "epoch"]   # + arm (one arm at a time)
# GRPO: generation_batch = 128 completions per optimizer step / G = 8 siblings → 16 groups per step.
GRPO_GROUPS_PER_STEP = 16
GRPO_CANDS_PER_STEP = 128
_ROLE_RE = re.compile(r"\[(THERAPIST|PATIENT)\]: ")
_LEAK = "<|im_start|>"
# wrap-up cues in a PATIENT turn — used only to VALIDATE the whitespace fingerprint of a stripped
# SESSION ENDED (never as a metric of its own).
_WRAPUP_RE = re.compile(r"wrap|for now|next time|next session|enough for today|i think i'?m good|call it|"
                        r"end (the|this) session|good place to stop|thank", re.I)
_METHOD_ORDER = {"PTO": 0, "GRPO": 1}

# The scout cross-check the paper anchored on: GRPO_LA5 iteration 5, first 300 logged groups
# (2,400 candidates) — realized-turn histogram + ended-early count read by hand from
# generations.jsonl before the generator existed. `tails_numbers` recomputes it; the self-check
# compares against these constants.
SCOUT_EXPECTED = {"arm": "GRPO_LA5", "train_iter": 5, "n_groups": 300, "n": 2400,
                  "realized_turns": {5: 1743, 1: 374, 3: 233, 0: 45, 4: 3, 2: 2},
                  "ended_early": 657, "ended_early_rate": 657 / 2400}

# Where the notebooks save these frames (the `source` strings of the ledger point there).
_TBL_MECH = "lookahead/mechanism/tables"
_TBL_COST = "compute/cost/tables"


# ── helpers ───────────────────────────────────────────────────────────────────

def _arm_runs(arms):
    from . import discover_arms
    return list(discover_arms()) if arms is None else list(arms)


def _tail_arms(arms) -> list:
    """The arms that carry look-ahead tails (K > 0), PTO before GRPO (the paper's order)."""
    return sorted([a for a in _arm_runs(arms) if int(a.K) > 0],
                  key=lambda a: (_METHOD_ORDER.get(a.method, 9), a.K, a.label))


def _all_arms(arms) -> list:
    return sorted(_arm_runs(arms), key=lambda a: (_METHOD_ORDER.get(a.method, 9), a.K, a.label))


def _memo_key(arm_list) -> str:
    return "|".join(f"{a.exp_name}:{','.join(map(str, a.iters))}"
                    for a in sorted(arm_list, key=lambda a: a.exp_name))


def parse_tail(tail: str):
    """Split an oracle-format tail into ``[(role, text), ...]`` in order."""
    if not tail:
        return []
    ms = list(_ROLE_RE.finditer(tail))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(tail)
        out.append((m.group(1), tail[m.end():end]))
    return out


def end_reason(rt, tail: str, k: int = 5) -> str:
    """Classify how a K-turn look-ahead tail terminated (see the module docstring's fingerprint).

    ``full`` (realized_turns >= K), ``no_tail`` (0 turns: the patient call failed / a bare marker
    as the first reply), ``patient_closed`` (ends on a PATIENT turn AND the kept text ends in
    whitespace = a stripped SESSION ENDED), ``therapist_stalled`` (ends on a patient turn without
    the fingerprint: an empty/degenerate therapist turn froze the sim), ``after_therapist`` (ends
    on a THERAPIST turn: therapist marker / patient failure), ``not_simulated`` (``rt`` is None).
    """
    if rt is None:
        return "not_simulated"
    rt = int(rt)
    if rt >= k:
        return "full"
    if rt == 0:
        return "no_tail"
    ws = bool(tail) and (tail != tail.rstrip())
    if rt % 2 == 1:                           # ends on a PATIENT turn
        return "patient_closed" if ws else "therapist_stalled"
    return "after_therapist"                  # ends on a THERAPIST turn


def tail_features(row, k: int = 5) -> dict:
    """Per-candidate tail structure + lexical cue features (tail therapist turns vs the candidate).

    ``row`` needs ``tail``, ``completion``, ``realized_turns``. ``tail_loop`` = a therapist turn in
    the tail verbatim-repeats the candidate or another tail turn; ``last_patient_wrapup_cue`` = the
    tail ends on a patient turn that carries a wrap-up phrase (the fingerprint validator).
    """
    tail = row["tail"] if isinstance(row["tail"], str) else ""
    turns = parse_tail(tail)
    pat = [t for r, t in turns if r == "PATIENT"]
    th = [t for r, t in turns if r == "THERAPIST"]
    comp = row["completion"] if isinstance(row["completion"], str) else ""
    th_strip = [t.strip() for t in th if t.strip()]
    pool = [comp.strip()] + th_strip if comp.strip() else th_strip
    cnt = Counter(pool)
    max_rep = max(cnt.values()) if cnt else 0
    n_th = len(th)
    return {
        "tail_chars": len(tail),
        "tail_patient_turns": len(pat),
        "tail_therapist_turns": n_th,
        "tail_patient_chars": sum(len(t) for t in pat),
        "tail_th_chars": sum(len(t) for t in th),
        "tail_th_turn_len": (sum(len(t) for t in th) / n_th) if n_th else np.nan,
        "tail_th_q_per_turn": (sum(t.count("?") for t in th) / n_th) if n_th else np.nan,
        "tail_th_affirm_rate": (sum(bool(RE_AFFIRM.search(t)) for t in th) / n_th) if n_th else np.nan,
        "tail_th_effusive_rate": (sum(bool(RE_EFFUSIVE.search(t)) for t in th) / n_th) if n_th else np.nan,
        "tail_loop": bool(max_rep >= 2),
        "tail_leak": _LEAK in tail,
        "tail_ws_end": bool(tail) and (tail != tail.rstrip()),
        "last_patient_wrapup_cue": bool(pat) and bool(_WRAPUP_RE.search(pat[-1])) and (turns[-1][0] == "PATIENT"),
        "cand_q": comp.count("?"),
        "cand_affirm": bool(RE_AFFIRM.search(comp)),
        "cand_effusive": bool(RE_EFFUSIVE.search(comp)),
        "end_reason": end_reason(row["realized_turns"], tail, k),
    }


def _boot_ci(values, stat=np.mean, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Percentile bootstrap 95% CI of ``stat`` over ``values`` (NaNs dropped)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    b = stat(v[idx], axis=1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def grpo_steps(arm) -> Dict[int, int]:
    """GRPO ground truth for how many optimizer steps each iteration ran.

    One ``training/completions/*.parquet`` per step (== the last checkpoint number), for every
    iteration that also has a ``generations.jsonl``. ⚠ The EDA capture is per-PROCESS on the older
    iterations — a crashed+resumed iteration logged only its post-resume steps (GRPO_LA5 iters 1–2,
    GRPO_LA0 iters 2/6/8) — so logged counts are scaled by ``logged_groups / (steps × 16)``.
    Empty for PTO (whose Step-2 checkpoint carries the records; its logs are complete).
    """
    out: Dict[int, int] = {}
    if arm.method != "GRPO":
        return out
    for d in glob.glob(os.path.join(arm.runs_dir, "iteration_*")):
        m = re.search(r"iteration_(\d+)$", d.replace("\\", "/"))
        if not m:
            continue
        n = len(glob.glob(os.path.join(d, "training", "completions", "*.parquet")))
        if n and os.path.exists(os.path.join(d, "eda", "generations.jsonl")):
            out[int(m.group(1))] = n
    return out


def stream_record_stats(arm) -> pd.DataFrame:
    """Parquet-cached wrapper — see :func:`_stream_record_stats_impl`.

    Cached on the RUN artifacts (``exts=RUN_SIGNATURE_EXTS``: this reads ``generations.jsonl``,
    which the default ``.csv`` signature would not watch). ~40 output rows from a multi-hundred-MB
    parse, and ``render_results.py`` runs six kernels, so the in-process memo alone re-paid it.
    """
    from .data import load_cached, runs_input_roots, RUN_SIGNATURE_EXTS
    return load_cached("stream_record_stats", [arm], lambda: _stream_record_stats_impl(arm),
                       input_roots=runs_input_roots([arm]), exts=RUN_SIGNATURE_EXTS)


def _stream_record_stats_impl(arm) -> pd.DataFrame:
    """Second (light) pass over an arm's ``generations.jsonl``: record-level fields
    :func:`~eda_analysis.training.load_generations` drops — prefix chars + oracle retries + per-
    candidate char sizes — aggregated per ``(train_iter, phase)``. No text is retained.

    Per scored candidate: ``oracle_calls`` = ``N_Q`` (+ recorded retries; ``len(sub_scores)`` when the
    score is missing, 1 when nothing came back), ``oracle_input_chars`` = (prefix + role label +
    completion + tail) × calls, ``patient_calls_tail`` = ``ceil(realized_turns / 2)`` (1 for a
    zero-turn tail whose first patient call was made, i.e. ``k > 0``),
    ``patient_input_chars_tail_ub`` = an upper bound (each call sees ≤ the full doc),
    ``therapist_gens_tail`` = ``realized_turns // 2`` (GPU, not API). PTO empty completions
    (``lookahead=None``) count under ``n_not_scored`` and cost nothing.
    """
    rows = []
    for fp in sorted(glob.glob(os.path.join(arm.runs_dir, "iteration_*", "eda", "generations.jsonl"))):
        acc: dict = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = (rec.get("iteration"), rec.get("phase"))
                a = acc.setdefault(key, Counter())
                pre = len(rec.get("prefix") or "")
                a["n_records"] += 1
                for c in rec.get("candidates", []) or []:
                    comp = c.get("completion") or ""
                    la = c.get("lookahead") or None
                    orc = c.get("oracle") or {}
                    a["n_cands"] += 1
                    if la is None:                        # PTO empty completion: not simulated, not scored
                        a["n_not_scored"] += 1
                        continue
                    sub = c.get("sub_scores") or {}
                    n_calls = len(sub) if c.get("score") is None else N_Q
                    if c.get("score") is None and not sub:
                        n_calls = 1                       # first oracle call failed outright
                    a["n_scored"] += int(c.get("score") is not None)
                    a["oracle_calls"] += n_calls + int(orc.get("retries") or 0)
                    a["oracle_retries"] += int(orc.get("retries") or 0)
                    tail = la.get("tail") or ""
                    doc = pre + len("\n\n[THERAPIST]: ") + len(comp) + len(tail)
                    a["oracle_input_chars"] += doc * n_calls
                    rt = int(la.get("realized_turns") or 0)
                    n_pat = int(np.ceil(rt / 2)) if rt > 0 else (1 if (la.get("k") or 0) > 0 else 0)
                    a["patient_calls_tail"] += n_pat
                    a["patient_input_chars_tail_ub"] += n_pat * doc
                    a["therapist_gens_tail"] += rt // 2
        for (it, ph), a in acc.items():
            rows.append({"arm": arm.label, "method": arm.method, "K": arm.K, "train_iter": it, "phase": ph, **a})
    cols = ["arm", "method", "K", "train_iter", "phase", "n_records", "n_cands", "n_not_scored", "n_scored",
            "oracle_calls", "oracle_retries", "oracle_input_chars", "patient_calls_tail",
            "patient_input_chars_tail_ub", "therapist_gens_tail"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = 0 if c not in ("arm", "method", "phase") else None
    return df[cols] if len(df) else df


def eval_conv_stats(arm) -> pd.DataFrame:
    """Parquet-cached wrapper — see :func:`_eval_conv_stats_impl`."""
    from .data import load_cached, conv_input_roots
    return load_cached("eval_conv_stats", [arm], lambda: _eval_conv_stats_impl(arm),
                       input_roots=conv_input_roots([arm]))


def _eval_conv_stats_impl(arm) -> pd.DataFrame:
    """Per ``model_iter`` of one arm: patient/therapist turn + char counts, mean conversation length
    and the session-end reasons (``session_ended_by``: patient / therapist / none), read from the
    eval conversation CSVs. The eval convs generated at the start of iteration ``n`` are
    ``model_iter_{n-1}`` (policy π_{n-1}). Pooled over model_iters this cross-checks the tracked
    ``session_end_reasons`` table.
    """
    rows = []
    for k in arm.iters:
        cdir = arm.conv_dir(k)
        if not cdir or not os.path.isdir(cdir):
            continue
        n_conv = 0; pat = 0; th = 0; pchars = 0; tchars = 0
        ended: Counter = Counter(); conv_len = []
        for fn in os.listdir(cdir):
            if not re.match(r"conversation_\d+\.csv$", fn):
                continue
            try:
                cdf = pd.read_csv(os.path.join(cdir, fn))
            except Exception:
                continue
            n_conv += 1
            r = cdf["role"].astype(str)
            txt = cdf["conversation"].fillna("").astype(str)
            pat += int((r == "patient").sum()); th += int((r == "therapist").sum())
            pchars += int(txt[r == "patient"].str.len().sum()); tchars += int(txt[r == "therapist"].str.len().sum())
            conv_len.append(len(cdf))
            eb = cdf["session_ended_by"].dropna() if "session_ended_by" in cdf.columns else pd.Series(dtype=object)
            ended[str(eb.iloc[0]) if len(eb) else "none"] += 1
        rows.append({"arm": arm.label, "method": arm.method, "K": arm.K, "model_iter": k, "n_convs": n_conv,
                     "eval_patient_turns": pat, "eval_therapist_turns": th,
                     "eval_patient_chars": pchars, "eval_therapist_chars": tchars,
                     "eval_conv_len_mean": float(np.mean(conv_len)) if conv_len else np.nan,
                     "eval_ended_by_patient": ended.get("patient", 0),
                     "eval_ended_by_therapist": ended.get("therapist", 0),
                     "eval_ended_none": ended.get("none", 0)})
    return pd.DataFrame(rows)


# ── the tail audit (one pass per K>0 arm; four frames out) ────────────────────

@dataclass
class TailAudit:
    """The four tail-audit frames + the row-filter bookkeeping, computed in ONE pass per arm.

    ``by_iter`` / ``cues`` / ``within_group`` / ``score_by_realized_turns`` are the frames the
    notebook saves; ``rows`` = per-arm ``{all, eval_phase_dropped, not_simulated_dropped,
    unscored_dropped, analysed}`` row counts; ``scout_check`` = the recomputed anchor
    (:data:`SCOUT_EXPECTED`) — empty when GRPO_LA5 iteration 5 is not among the arms.
    """
    by_iter: pd.DataFrame
    cues: pd.DataFrame
    within_group: pd.DataFrame
    score_by_realized_turns: pd.DataFrame
    rows: Dict[str, dict] = field(default_factory=dict)
    scout_check: dict = field(default_factory=dict)


_TAILS_MEMO: Dict[str, TailAudit] = {}
_API_MEMO: Dict[str, pd.DataFrame] = {}
_EV_MEMO: Dict[str, pd.DataFrame] = {}


def clear_tails_memo() -> None:
    """Drop the in-process tail-audit / api-calls / eval-conv memos (after a training run writes new rows)."""
    _TAILS_MEMO.clear()
    _API_MEMO.clear()
    _EV_MEMO.clear()


def _eval_conv_stats_cached(arm) -> pd.DataFrame:
    """Memoized :func:`eval_conv_stats` (the eval CSVs live behind a Drive symlink; read once per arm)."""
    key = _memo_key([arm])
    if key not in _EV_MEMO:
        _EV_MEMO[key] = eval_conv_stats(arm)
    return _EV_MEMO[key].copy()


def _audit_one_arm(arm, verbose: bool = True):
    """The per-arm body of the audit — returns (by_iter_rows, cues_rows, wg_rows, rt_rows, rows_info, scout)."""
    from . import training as T
    from .stats import paired_arrays

    k_la = int(arm.K)
    if verbose:
        print(f"  [tails] {arm.label}: load_generations(keep_tail=True) …")
    g = T.load_generations([arm], keep_tail=True)
    # Bound memory to one arm's tails — but drop ONLY the entry this call added. The old
    # `clear_generations_memo()` emptied the process-wide memo, so any later `load_generations`
    # in the same kernel (render_results.py shares one kernel across a whole top) paid a fresh
    # ~17 s full-tree parse. Harmless in today's cell order; a landmine for the next cell.
    T._GENERATIONS_MEMO.pop((T._arm_memo_key([arm]), True), None)
    n_all = len(g)
    n_eval_phase = int((g["phase"] == "eval").sum()) if len(g) else 0
    g = g[(g["phase"] != "eval")].copy() if len(g) else g
    n_train_phase = len(g)
    if not len(g):
        return [], [], [], [], {"all": 0, "eval_phase_dropped": 0, "not_simulated_dropped": 0,
                                "unscored_dropped": 0, "analysed": 0}, {}
    n_not_sim = int(g["realized_turns"].isna().sum())         # PTO empty completions (lookahead=None)
    pre_counts = (g.assign(unscored=g["realized_turns"].notna() & g["score"].isna(),
                           not_sim=g["realized_turns"].isna())
                  .groupby("train_iter")[["unscored", "not_sim"]].sum())
    steps = grpo_steps(arm)
    g = g[g["realized_turns"].notna() & g["score"].notna()].copy()
    g["realized_turns"] = g["realized_turns"].astype(int)
    g["ended_early"] = g["ended_early"].astype(bool)
    g["epoch"] = g["epoch"].fillna(-1.0)
    rows_info = {"all": n_all, "eval_phase_dropped": n_eval_phase, "not_simulated_dropped": n_not_sim,
                 "unscored_dropped": n_train_phase - n_not_sim - len(g), "analysed": len(g)}
    if verbose:
        print(f"    rows: all={n_all} eval_phase={n_eval_phase} train_phase={n_train_phase} "
              f"not_simulated={n_not_sim} analysed={len(g)}")

    # scout cross-check: GRPO_LA5 iteration 5, first 300 groups (2,400 candidates)
    scout: dict = {}
    if arm.label == SCOUT_EXPECTED["arm"] and (g["train_iter"] == SCOUT_EXPECTED["train_iter"]).any():
        g5 = g[g["train_iter"] == SCOUT_EXPECTED["train_iter"]]
        first_groups = g5.drop_duplicates(GROUP_KEYS)[GROUP_KEYS].head(SCOUT_EXPECTED["n_groups"])
        sub = g5.merge(first_groups, on=GROUP_KEYS)
        scout = {"n": int(len(sub)),
                 "realized_turns": {int(k): int(v) for k, v in sub["realized_turns"].value_counts().items()},
                 "ended_early": int(sub["ended_early"].sum()),
                 "ended_early_rate": float(sub["ended_early"].mean())}
        if verbose:
            print("    scout cross-check (GRPO_LA5 iter 5, first 300 groups):", scout)

    feats = pd.DataFrame([tail_features(r, k_la) for r in g[["tail", "completion", "realized_turns"]].to_dict("records")],
                         index=g.index)
    g = pd.concat([g.drop(columns=["tail"]), feats], axis=1)

    # within-group quantities: deviation from group mean, z (= GRPO's advantage), group size
    grp = g.groupby(GROUP_KEYS, observed=True)["score"]
    g["group_n"] = grp.transform("size")
    g["dev"] = g["score"] - grp.transform("mean")
    sd = grp.transform("std").replace(0.0, np.nan)
    g["z"] = g["dev"] / sd
    g["chosen"] = g["is_chosen"].astype(bool)     # GRPO chosen_idx; PTO role=='chosen' — identical (checked)

    by_iter_rows, cues_rows, wg_rows, rt_rows = [], [], [], []
    iters = sorted(g["train_iter"].unique())
    for it in iters + ["pooled"]:
        d = g if it == "pooled" else g[g["train_iter"] == it]
        n = len(d)
        n_groups = d.drop_duplicates(GROUP_KEYS).shape[0]
        rt_counts = d["realized_turns"].value_counts()
        ee = d["ended_early"].astype(float)
        ee_lo, ee_hi = _boot_ci(ee.values)
        er = d["end_reason"].value_counts(normalize=True)
        if it == "pooled":
            cov = (sum(d.drop_duplicates(GROUP_KEYS).groupby("train_iter").size().get(i, 0) for i in steps)
                   / (GRPO_GROUPS_PER_STEP * sum(steps.values()))) if steps else 1.0
            n_unsc, n_nsim = int(pre_counts["unscored"].sum()), int(pre_counts["not_sim"].sum())
        else:
            cov = (n_groups / (GRPO_GROUPS_PER_STEP * steps[it])) if (steps and it in steps) else 1.0
            n_unsc = int(pre_counts["unscored"].get(it, 0)); n_nsim = int(pre_counts["not_sim"].get(it, 0))
        pc_mask = d["end_reason"] == "patient_closed"
        by_iter_rows.append({
            "arm": arm.label, "train_iter": it, "n_groups": n_groups, "n_candidates": n,
            "log_coverage": cov, "n_unscored_dropped": n_unsc, "n_not_simulated_dropped": n_nsim,
            "mean_score": d["score"].mean(),
            "realized_turns_mean": d["realized_turns"].mean(),
            **{f"rt{k}_share": float(rt_counts.get(k, 0)) / n for k in range(k_la + 1)},
            "ended_early_rate": ee.mean(), "ended_early_ci_lo": ee_lo, "ended_early_ci_hi": ee_hi,
            "patient_closed_share": float(er.get("patient_closed", 0.0)),
            "therapist_stalled_share": float(er.get("therapist_stalled", 0.0)),
            "after_therapist_share": float(er.get("after_therapist", 0.0)),
            "no_tail_share": float(er.get("no_tail", 0.0)),
            "full_share": float(er.get("full", 0.0)),
            "full_closed_at_turn5_share": float((d["tail_ws_end"] & (d["realized_turns"] == k_la)).mean()),
            "wrapup_cue_rate_patient_closed": float(d.loc[pc_mask, "last_patient_wrapup_cue"].mean()) if pc_mask.any() else np.nan,
            "wrapup_cue_rate_full_open": float(d.loc[(d["realized_turns"] == k_la) & ~d["tail_ws_end"], "last_patient_wrapup_cue"].mean()),
            "tail_chars_mean": d["tail_chars"].mean(),
            "tail_patient_turns_mean": d["tail_patient_turns"].mean(),
            "tail_therapist_turns_mean": d["tail_therapist_turns"].mean(),
            "tail_loop_rate": d["tail_loop"].astype(float).mean(),
            "tail_leak_rate": d["tail_leak"].astype(float).mean(),
            "cand_floored_rate": d["floored"].astype(float).mean(),
            "cand_empty_rate": d["empty"].astype(float).mean(),
            "cand_leak_rate": d["leak"].astype(float).mean()})
        cues_rows.append({
            "arm": arm.label, "train_iter": it, "n_candidates": n,
            "n_tail_th_turns": int(d["tail_therapist_turns"].sum()),
            "cand_len_chars": d["len_chars"].mean(),
            "tail_th_turn_len_chars": d["tail_th_turn_len"].mean(),
            "cand_q_per_turn": d["cand_q"].mean(),
            "tail_th_q_per_turn": d["tail_th_q_per_turn"].mean(),
            "cand_affirm_rate": d["cand_affirm"].astype(float).mean(),
            "tail_th_affirm_rate": d["tail_th_affirm_rate"].mean(),
            "cand_effusive_rate": d["cand_effusive"].astype(float).mean(),
            "tail_th_effusive_rate": d["tail_th_effusive_rate"].mean(),
            "tail_loop_rate": d["tail_loop"].astype(float).mean(),
            "tail_patient_turn_len_chars": (d["tail_patient_chars"].sum() / max(1, d["tail_patient_turns"].sum()))})

        # ── within-group: does the tail predict the reward? ──
        gk = d.groupby(GROUP_KEYS, observed=True)
        # (i) pooled group-demeaned Spearman: score dev vs realized_turns / tail chars (groups with variance)
        rt_dev = d["realized_turns"] - gk["realized_turns"].transform("mean")
        has_var = gk["realized_turns"].transform("nunique") > 1
        rho_rt = sps.spearmanr(d.loc[has_var, "dev"], rt_dev[has_var]).statistic if has_var.sum() > 10 else np.nan
        tc_dev = d["tail_chars"] - gk["tail_chars"].transform("mean")
        rho_tc = sps.spearmanr(d["dev"], tc_dev).statistic if len(d) > 10 else np.nan
        # (ii) paired: mean(score | ended_early) − mean(score | full) within groups holding both
        dd = d.assign(s_ee=d["score"].where(d["ended_early"]), s_full=d["score"].where(~d["ended_early"]),
                      z_ee=d["z"].where(d["ended_early"]), z_full=d["z"].where(~d["ended_early"]),
                      n_ee=d["ended_early"].astype(int), n_full=(~d["ended_early"]).astype(int),
                      ch_ee=(d["chosen"] & d["ended_early"]).astype(int),
                      ch_full=(d["chosen"] & ~d["ended_early"]).astype(int))
        per = dd.groupby(GROUP_KEYS, observed=True).agg(
            m_ee=("s_ee", "mean"), m_full=("s_full", "mean"), z_ee=("z_ee", "mean"), z_full=("z_full", "mean"),
            n_ee=("n_ee", "sum"), n_full=("n_full", "sum"), ch_ee=("ch_ee", "sum"), ch_full=("ch_full", "sum")).reset_index()
        both = per[(per["n_ee"] > 0) & (per["n_full"] > 0)]
        pr = paired_arrays(both["m_ee"].values, both["m_full"].values, n_boot=N_BOOT)
        # (iii) P(chosen | ended_early) vs P(chosen | full) per candidate; P(argmax ended early) vs base
        #       rate per group — both with a bootstrap over GROUPS (the sampling unit)
        A = per[["n_ee", "n_full", "ch_ee", "ch_full"]].to_numpy(float)

        def _stats(M):
            n_ee_, n_full_, ch_ee_, ch_full_ = M[..., 0], M[..., 1], M[..., 2], M[..., 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                p_ee = ch_ee_ / n_ee_; p_full = ch_full_ / n_full_
                rr_ = p_ee / p_full
                diff_ = ch_ee_ / (ch_ee_ + ch_full_) - n_ee_ / (n_ee_ + n_full_)
            return p_ee, p_full, rr_, diff_

        p_ch_ee, p_ch_full, rr, diff = _stats(A.sum(axis=0))
        rng = np.random.default_rng(BOOT_SEED)
        ng = A.shape[0]
        rr_b, diff_b = [], []
        for _ in range(N_BOOT // 100):            # chunked group bootstrap (memory)
            idx = rng.integers(0, ng, size=(100, ng))
            _, _, r_b, d_b = _stats(A[idx].sum(axis=1))
            rr_b.append(r_b); diff_b.append(d_b)
        rr_b = np.concatenate(rr_b); diff_b = np.concatenate(diff_b)
        p_ch_ee, p_ch_full, rr = float(p_ch_ee), float(p_ch_full), float(rr)
        rr_lo, rr_hi = np.nanpercentile(rr_b, [2.5, 97.5])
        df_lo, df_hi = np.nanpercentile(diff_b, [2.5, 97.5])
        p_chosen_is_ee = float(per["ch_ee"].sum() / max(1, (per["ch_ee"] + per["ch_full"]).sum()))
        base_ee = float(d["ended_early"].mean())
        n_groups_rt_var = int(d.loc[has_var].drop_duplicates(GROUP_KEYS).shape[0]) if has_var.any() else 0
        wg_rows.append({
            "arm": arm.label, "train_iter": it, "n_groups": n_groups, "n_candidates": n, "log_coverage": cov,
            "rho_dev_vs_realized_turns": rho_rt, "n_groups_rt_var": n_groups_rt_var,
            "rho_dev_vs_tail_chars": rho_tc,
            "n_groups_both": pr["n"], "delta_ee_minus_full": pr["mean_delta"], "dz": pr["dz"],
            "ci_lo": pr["ci_lo"], "ci_hi": pr["ci_hi"], "p": pr["p"],
            "z_ee_mean": both["z_ee"].mean() if len(both) else np.nan,
            "z_full_mean": both["z_full"].mean() if len(both) else np.nan,
            "base_ee_rate": base_ee, "p_chosen_is_ee": p_chosen_is_ee,
            "chosen_minus_base": p_chosen_is_ee - base_ee, "chosen_minus_base_ci_lo": df_lo, "chosen_minus_base_ci_hi": df_hi,
            "p_chosen_given_ee": p_ch_ee, "p_chosen_given_full": p_ch_full,
            "rr_chosen_ee_vs_full": rr, "rr_ci_lo": rr_lo, "rr_ci_hi": rr_hi})
        # curve for the figure's panel (b): mean within-group deviation by realized_turns
        for k, s in d.groupby("realized_turns")["dev"]:
            if len(s) >= 5:
                lo, hi = _boot_ci(s.values, n_boot=500)
                rt_rows.append({"arm": arm.label, "train_iter": it, "realized_turns": int(k), "n": len(s),
                                "dev_mean": s.mean(), "dev_lo": lo, "dev_hi": hi,
                                "score_mean": d.loc[s.index, "score"].mean(),
                                "p_chosen": d.loc[s.index, "chosen"].mean()})
    del g, feats
    return by_iter_rows, cues_rows, wg_rows, rt_rows, rows_info, scout


def tail_audit_frames(arms=None, *, refresh: bool = False, verbose: bool = True) -> TailAudit:
    """Run the tail audit over every K>0 arm in ``arms`` (default: every arm on disk) and return
    the four frames as one :class:`TailAudit`. Memoized per arm subset for the life of the process
    (~2 min per LA5 arm on a warm Drive; ``refresh=True`` recomputes).

    ``within_group['p_holm']`` = Holm within arm across iterations (the ``pooled`` row is not part
    of the family). This is the single expensive pass; :func:`tail_audit_by_iter`,
    :func:`tail_cues_by_iter`, :func:`tail_within_group` and :func:`score_by_realized_turns` are
    views on its result.
    """
    from .stats import holm
    la = _tail_arms(arms)
    key = _memo_key(la)
    if not refresh and key in _TAILS_MEMO:
        return _TAILS_MEMO[key]
    by_iter_rows, cues_rows, wg_rows, rt_rows = [], [], [], []
    rows_info: Dict[str, dict] = {}
    scout: dict = {}
    for arm in la:
        b, c, w, r, info, sc = _audit_one_arm(arm, verbose=verbose)
        by_iter_rows += b; cues_rows += c; wg_rows += w; rt_rows += r
        rows_info[arm.label] = info
        if sc:
            scout = sc
    by_iter = pd.DataFrame(by_iter_rows)
    cues = pd.DataFrame(cues_rows)
    wg = pd.DataFrame(wg_rows)
    rt = pd.DataFrame(rt_rows)
    if len(wg):
        wg["p_holm"] = np.nan
        for _arm, idx in wg[wg["train_iter"] != "pooled"].groupby("arm").groups.items():
            wg.loc[idx, "p_holm"] = holm(wg.loc[idx, "p"].values)
    out = TailAudit(by_iter=by_iter, cues=cues, within_group=wg, score_by_realized_turns=rt,
                    rows=rows_info, scout_check=scout)
    _TAILS_MEMO[key] = out
    return out


def tail_audit_by_iter(arms=None, **kw) -> pd.DataFrame:
    """Look-ahead TAIL structure per K>0 arm × training iteration (+ a ``pooled`` row per arm).

    ``train_iter`` n = the branching done by policy π_n, whose eval convs are ``model_iter_{n-1}``;
    there is no tail at iteration 0 because look-ahead only runs during training. One row per
    candidate scored under K=5 (GRPO ``eval``-phase groups and PTO empty completions excluded).
    Columns: ``n_groups``, ``n_candidates``, ``log_coverage`` (GRPO: logged groups / (steps × 16);
    PTO 1.0), ``n_unscored_dropped``, ``n_not_simulated_dropped``, ``mean_score``,
    ``realized_turns_mean``, ``rt{k}_share`` (share of candidates with k realized turns), the
    end-reason shares (``patient_closed`` = the patient wrote SESSION ENDED — stripped, identified
    by the trailing-whitespace fingerprint; ``therapist_stalled``; ``after_therapist``;
    ``no_tail``; ``full``), ``full_closed_at_turn5_share`` (the fingerprint on a full tail),
    ``wrapup_cue_rate_patient_closed`` vs ``wrapup_cue_rate_full_open`` (the fingerprint's
    validation), ``ended_early_rate`` + bootstrap 95% CI over candidates, ``tail_chars_mean``,
    ``tail_patient/therapist_turns_mean``, ``tail_loop_rate`` (a tail therapist turn verbatim-
    repeats the candidate or another tail turn), ``tail_leak_rate``, ``cand_floored/empty/leak_rate``.
    Reproduces the paper's ``tail_audit_by_iter``.
    """
    return tail_audit_frames(arms, **kw).by_iter.copy()


def tail_cues_by_iter(arms=None, **kw) -> pd.DataFrame:
    """Simple lexical cues in the tail's THERAPIST turns versus in the scored candidate itself, per
    K>0 arm × ``train_iter`` (+ ``pooled``): chars per turn, question marks per turn, the
    ``RE_AFFIRM`` / ``RE_EFFUSIVE`` regexes from :mod:`eda_analysis.constants` (directional sanity
    cues, not primary metrics), ``tail_loop_rate``, patient turn length. Reproduces the paper's
    ``tail_audit_cues_by_iter``.
    """
    return tail_audit_frames(arms, **kw).cues.copy()


def tail_within_group(arms=None, **kw) -> pd.DataFrame:
    """Does the tail predict the reward WITHIN a group (the unit the update sees: GRPO's G=8
    siblings, PTO's M=8 branches)?

    ``rho_dev_vs_realized_turns`` / ``rho_dev_vs_tail_chars`` = Spearman of the group-demeaned score
    against the group-demeaned realized_turns / tail chars (pooled over groups with within-group
    variance; ``n_groups_rt_var``). ``delta_ee_minus_full`` = mean(score | ended early) − mean(score
    | full tail) paired within groups holding both kinds (``n_groups_both``; SIGN: ``+`` ⇒ the
    ended-early candidates score HIGHER); ``dz``, bootstrap 95% CI, Wilcoxon ``p``, ``p_holm`` within
    arm across iterations. ``z_ee_mean`` / ``z_full_mean`` = the group-standardised score (GRPO's
    advantage). ``p_chosen_is_ee`` = share of groups whose argmax candidate ended early vs
    ``base_ee_rate`` = share of all candidates that ended early (``chosen_minus_base`` + group-
    bootstrap CI); ``p_chosen_given_ee`` / ``_full`` = per-candidate P(argmax) by tail kind (1/8 =
    chance) and their ratio ``rr_chosen_ee_vs_full`` with a group-bootstrap 95% CI. Reproduces the
    paper's ``tail_audit_within_group``.
    """
    return tail_audit_frames(arms, **kw).within_group.copy()


def score_by_realized_turns(arms=None, **kw) -> pd.DataFrame:
    """Mean within-group score deviation (score − group mean; Q1Q2 points) and P(argmax) by
    ``realized_turns``, per K>0 arm and ``train_iter`` (+ ``pooled``); bootstrap 95% CI over
    candidates (500 draws), cells with ≥ 5 candidates only. Odd realized_turns = the tail ends on a
    patient turn (the patient closed the session), even = ends on a therapist turn (0 = no tail).
    Reproduces the paper's ``tail_audit_score_by_realized_turns``.
    """
    return tail_audit_frames(arms, **kw).score_by_realized_turns.copy()


# ── API-call accounting (all arms) ─────────────────────────────────────────────

def api_calls(arms=None, *, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """API-call accounting per arm × training iteration ``n`` for EVERY arm in ``arms``.

    A row = the cost of iteration ``n``: the 96 eval convs generated at its start by policy π_{n-1}
    (``eval_convs_of = model_iter_{n-1}``) PLUS the training-time calls; the last row per arm,
    ``row_kind = 'final eval pass'``, is the post-loop generate-only pass with no training (dropped
    when its convs are missing). Columns:

    * ``oracle_calls_train`` = Q1 + Q2 calls per scored candidate (2 per candidate, + recorded
      retries; GRPO's TRL eval-phase groups included — ``n_candidates_eval_phase``), read from
      ``generations.jsonl`` and, for GRPO, RESCALED to the ground-truth step count (``n_steps`` =
      ``training/completions/*.parquet`` files, 128 candidates = 16 groups × G=8 per step;
      ``log_coverage`` = logged / expected groups, < 1 where a crashed iteration's pre-resume
      records were lost: GRPO_LA5 iters 1–2, GRPO_LA0 iters 2, 6, 8).
    * ``oracle_input_Mchars`` = the chars the oracle read (prefix + completion + tail, × 2 rubrics),
      a token proxy. ``oracle_retries`` as recorded.
    * ``eval_scoring_calls_run_eval`` = 96 × 8 instruments per model state (Run_Eval; identical for
      every arm; per grader).
    * ``patient_calls_eval_convs`` = patient turns in the ``model_iter_{n-1}`` CSVs;
      ``patient_calls_trunk`` = PTO greedy trunk replies (≤ 1 per branch point — the last branch
      point of a trunk that reaches its length cap needs none — an upper bound within 96 per
      iteration; 0 for GRPO); ``patient_calls_tail`` = realized patient turns inside K=5 tails
      (``ceil(realized_turns/2)``; 1 for a zero-turn tail whose first patient call was made);
      ``patient_calls_total`` = the sum of the three.
    * ``therapist_gens_tail`` = therapist turns generated inside tails (GPU, not API).
    * ``eval_conv_len_mean``, ``eval_ended_by_patient``, ``n_eval_convs`` from the eval CSVs.

    Memoized per arm subset. Reproduces the paper's ``tail_audit_api_calls``.
    """
    all_arms = _all_arms(arms)
    key = _memo_key(all_arms)
    if not refresh and key in _API_MEMO:
        return _API_MEMO[key].copy()
    if verbose:
        print(f"  [tails] api_calls: streaming generations.jsonl + eval CSVs for {[a.label for a in all_arms]} …")
    rec_stats = pd.concat([stream_record_stats(a) for a in all_arms], ignore_index=True)
    ev = pd.concat([_eval_conv_stats_cached(a) for a in all_arms], ignore_index=True)
    api_rows = []
    for a in all_arms:
        steps = grpo_steps(a)
        rs = rec_stats[rec_stats["arm"] == a.label] if len(rec_stats) else rec_stats
        e = ev[ev["arm"] == a.label].set_index("model_iter") if len(ev) else pd.DataFrame()
        n_iter = int(rs["train_iter"].max()) if len(rs) else 0
        for n in range(1, n_iter + 2):
            r_train = rs[(rs["train_iter"] == n) & (rs["phase"] != "eval")]
            r_evalph = rs[(rs["train_iter"] == n) & (rs["phase"] == "eval")]
            k = n - 1                                     # eval convs generated at the start of iteration n
            erow = e.loc[k] if k in e.index else None
            final_pass = (n == n_iter + 1)
            n_rec_logged = int(r_train["n_records"].sum()); n_c_logged = int(r_train["n_cands"].sum())
            n_steps = steps.get(n, np.nan) if a.method == "GRPO" else np.nan
            cov = (n_rec_logged / (GRPO_GROUPS_PER_STEP * n_steps)) if (a.method == "GRPO" and n_steps and not final_pass) else 1.0
            cov = cov if cov > 0 else 1.0
            n_rec = int(round(n_rec_logged / cov)); n_c = int(round(n_c_logged / cov))
            n_c_eval = int(round(r_evalph["n_cands"].sum() / cov))
            oc_train = int(round((r_train["oracle_calls"].sum() + r_evalph["oracle_calls"].sum()) / cov))
            pat_tail = int(round((r_train["patient_calls_tail"].sum() + r_evalph["patient_calls_tail"].sum()) / cov))
            pat_trunk = n_rec if a.method == "PTO" else 0
            pat_eval = int(erow["eval_patient_turns"]) if erow is not None else 0
            api_rows.append({
                "arm": a.label, "method": a.method, "K": a.K, "train_iter": n,
                "row_kind": "final eval pass" if final_pass else "iteration",
                "eval_convs_of": f"model_iter_{k}",
                "n_eval_convs": int(erow["n_convs"]) if erow is not None else 0,
                "eval_conv_len_mean": float(erow["eval_conv_len_mean"]) if erow is not None else np.nan,
                "eval_ended_by_patient": int(erow["eval_ended_by_patient"]) if erow is not None else 0,
                "n_steps": n_steps, "n_groups_logged": n_rec_logged, "log_coverage": cov,
                "n_groups": n_rec, "n_candidates": n_c, "n_candidates_eval_phase": n_c_eval,
                "oracle_calls_train": oc_train,
                "oracle_retries": int(round((r_train["oracle_retries"].sum() + r_evalph["oracle_retries"].sum()) / cov)),
                "oracle_input_Mchars": (r_train["oracle_input_chars"].sum() + r_evalph["oracle_input_chars"].sum()) / cov / 1e6,
                "eval_scoring_calls_run_eval": (int(erow["n_convs"]) * N_EVAL_RUBRICS) if erow is not None else 0,
                "patient_calls_eval_convs": pat_eval,
                "patient_calls_trunk": pat_trunk,
                "patient_calls_tail": pat_tail,
                "patient_calls_total": pat_eval + pat_trunk + pat_tail,
                "therapist_gens_tail": int(round((r_train["therapist_gens_tail"].sum() + r_evalph["therapist_gens_tail"].sum()) / cov))})
    api = pd.DataFrame(api_rows)
    if len(api):
        api = api[~((api["row_kind"] == "final eval pass") & (api["n_eval_convs"] == 0))].reset_index(drop=True)
    _API_MEMO[key] = api
    return api.copy()


def api_ratio(api_df: pd.DataFrame) -> pd.DataFrame:
    """K=5 / K=0 API-call ratios per method, summed over matched training iterations.

    Two windows per method: ``iters 1-5 (matched)`` (the iterations both K arms of the method
    ran, capped at 5 — a FIXED early window, kept as-is so its numbers stay comparable with the
    frozen fixture; it was the censored arm's full support when it was named, which it no longer
    is) and ``all matched iters`` (every iteration both K arms have, derived); quantities
    ``oracle_calls_train``, ``oracle_input_Mchars``, ``patient_calls_total``, ``patient_calls_tail``,
    ``n_candidates``, ``total_api_calls`` (= oracle_calls_train + patient_calls_total), each with
    ``K0_sum``, ``K5_sum``, ``K5_over_K0`` and the ``arithmetic`` string. A final
    ``patient_calls_tail_per_candidate`` row per method (the physics: 3 patient calls per full tail).
    ⚠ The oracle ratio is NOT 1 even though calls per candidate are matched: the number of
    candidates per iteration differs between arms (GRPO's prompt count follows the eval-conv length
    of the current policy; PTO's branch points follow how far its trunks grow before the patient
    closes the session). Reproduces the paper's ``tail_audit_api_ratio``.
    """
    ratio_rows = []
    for m in [m for m in ("PTO", "GRPO") if m in set(api_df["method"])]:
        a0 = api_df[(api_df["method"] == m) & (api_df["K"] == 0) & (api_df["row_kind"] == "iteration")]
        a5 = api_df[(api_df["method"] == m) & (api_df["K"] > 0) & (api_df["row_kind"] == "iteration")]
        common = sorted(set(a0["train_iter"]) & set(a5["train_iter"]))
        for label, its in (("iters 1-5 (matched)", [i for i in common if i <= 5]), ("all matched iters", common)):
            s0 = a0[a0["train_iter"].isin(its)]; s5 = a5[a5["train_iter"].isin(its)]
            for col in ("oracle_calls_train", "oracle_input_Mchars", "patient_calls_total", "patient_calls_tail",
                        "n_candidates", "total_api_calls"):
                if col == "total_api_calls":
                    v0 = float(s0["oracle_calls_train"].sum() + s0["patient_calls_total"].sum())
                    v5 = float(s5["oracle_calls_train"].sum() + s5["patient_calls_total"].sum())
                else:
                    v0 = float(s0[col].sum()); v5 = float(s5[col].sum())
                ratio_rows.append({"method": m, "window": label, "iters": ",".join(map(str, its)), "quantity": col,
                                   "K0_sum": v0, "K5_sum": v5, "K5_over_K0": (v5 / v0) if v0 else np.nan,
                                   "arithmetic": f"{v5:,.0f} / {v0:,.0f}" if col != "oracle_input_Mchars" else f"{v5:,.1f} / {v0:,.1f}"})
        n5 = float(a5["n_candidates"].sum()); pt5 = float(a5["patient_calls_tail"].sum())
        ratio_rows.append({"method": m, "window": "all K5 iters", "iters": ",".join(map(str, sorted(a5["train_iter"]))),
                           "quantity": "patient_calls_tail_per_candidate", "K0_sum": 0.0, "K5_sum": pt5 / n5 if n5 else np.nan,
                           "K5_over_K0": np.nan, "arithmetic": f"{pt5:,.0f} / {n5:,.0f} candidates"})
    return pd.DataFrame(ratio_rows)


# ── ledger ─────────────────────────────────────────────────────────────────────

_clean = json_scalar          # one definition — see eda_analysis/ledger.py


def tails_numbers(arms=None, *, audit: Optional[TailAudit] = None, api: Optional[pd.DataFrame] = None,
                  ratio: Optional[pd.DataFrame] = None, verbose: bool = True) -> Dict[str, dict]:
    """Every number the write-up may quote from this module, as ``{dotted.key: {value, source, note}}``
    (the shape ``exports.save_numbers`` writes; mirrors the paper's ``out/tail_audit.json``).

    Keys: ``rows.<arm>`` (row-filter counts), ``scout_check.GRPO_LA5.iter5.first300groups`` (the
    anchor: :data:`SCOUT_EXPECTED` — expected rt {5:1743,1:374,3:233,0:45,4:3,2:2}, ended_early
    657/2400), ``by_iter.<arm>.<iter>``, ``within_group.<arm>.<iter>``, ``cues.<arm>.<iter>``,
    ``score_by_realized_turns.<arm>.pooled.rt<k>``, ``api.<arm>.iter<n>``,
    ``api_ratio.<method>.<window>.<quantity>``, ``api_totals.<arm>`` (column sums incl. the final
    eval pass) and ``eval_session_end_reasons_pooled`` (cross-checks the tracked
    ``session_end_reasons`` table). Anything not passed in is computed (memoized) from ``arms``.
    """
    audit = audit if audit is not None else tail_audit_frames(arms, verbose=verbose)
    api = api if api is not None else api_calls(arms, verbose=verbose)
    ratio = ratio if ratio is not None else api_ratio(api)
    N: Dict[str, dict] = {}

    def put(key, value, source="", note=""):
        N[key] = {"value": _clean(value), "source": source, "note": note}

    for arm_l, info in audit.rows.items():
        put(f"rows.{arm_l}", info, source="tails.tail_audit_frames row filter over generations.jsonl")
    put("scout_check.GRPO_LA5.iter5.first300groups", audit.scout_check,
        source="tails.tail_audit_frames (recomputed from generations.jsonl)",
        note=(f"scout expected rt {SCOUT_EXPECTED['realized_turns']}, ended_early "
              f"{SCOUT_EXPECTED['ended_early']}/{SCOUT_EXPECTED['n']}={100 * SCOUT_EXPECTED['ended_early_rate']:.1f}%"))
    for _, r in audit.by_iter.iterrows():
        put(f"by_iter.{r['arm']}.{r['train_iter']}",
            {k: r[k] for k in ("n_groups", "n_candidates", "log_coverage", "n_unscored_dropped", "ended_early_rate",
                               "ended_early_ci_lo", "ended_early_ci_hi", "realized_turns_mean", "patient_closed_share",
                               "therapist_stalled_share", "no_tail_share", "full_share", "full_closed_at_turn5_share",
                               "wrapup_cue_rate_patient_closed", "wrapup_cue_rate_full_open", "tail_chars_mean",
                               "tail_loop_rate", "cand_floored_rate")},
            source=f"{_TBL_MECH}/tail_audit_by_iter.md row arm={r['arm']} train_iter={r['train_iter']}")
    for _, r in audit.within_group.iterrows():
        put(f"within_group.{r['arm']}.{r['train_iter']}",
            {k: r[k] for k in ("n_groups", "n_groups_both", "delta_ee_minus_full", "dz", "ci_lo", "ci_hi", "p", "p_holm",
                               "rho_dev_vs_realized_turns", "rho_dev_vs_tail_chars", "z_ee_mean", "z_full_mean",
                               "base_ee_rate", "p_chosen_is_ee", "chosen_minus_base", "chosen_minus_base_ci_lo",
                               "chosen_minus_base_ci_hi", "p_chosen_given_ee", "p_chosen_given_full",
                               "rr_chosen_ee_vs_full", "rr_ci_lo", "rr_ci_hi")},
            source=f"{_TBL_MECH}/tail_audit_within_group.md row arm={r['arm']} train_iter={r['train_iter']}")
    for _, r in audit.cues.iterrows():
        put(f"cues.{r['arm']}.{r['train_iter']}",
            {k: r[k] for k in ("cand_q_per_turn", "tail_th_q_per_turn", "cand_affirm_rate", "tail_th_affirm_rate",
                               "cand_effusive_rate", "tail_th_effusive_rate", "cand_len_chars", "tail_th_turn_len_chars",
                               "tail_loop_rate")},
            source=f"{_TBL_MECH}/tail_audit_cues_by_iter.md row arm={r['arm']} train_iter={r['train_iter']}")
    rt = audit.score_by_realized_turns
    if len(rt):
        for _, r in rt[rt["train_iter"] == "pooled"].iterrows():
            put(f"score_by_realized_turns.{r['arm']}.pooled.rt{r['realized_turns']}",
                {k: r[k] for k in ("n", "dev_mean", "dev_lo", "dev_hi", "score_mean", "p_chosen")},
                source=(f"{_TBL_MECH}/tail_audit_score_by_realized_turns.md row arm={r['arm']} "
                        f"train_iter=pooled realized_turns={r['realized_turns']}"))
    for _, r in api.iterrows():
        put(f"api.{r['arm']}.iter{r['train_iter']}",
            {k: r[k] for k in ("row_kind", "eval_convs_of", "n_steps", "n_groups_logged", "log_coverage", "n_groups",
                               "n_candidates", "n_candidates_eval_phase", "oracle_calls_train", "oracle_retries",
                               "oracle_input_Mchars", "patient_calls_eval_convs", "patient_calls_trunk",
                               "patient_calls_tail", "patient_calls_total", "therapist_gens_tail",
                               "eval_conv_len_mean", "eval_ended_by_patient")},
            source=f"{_TBL_COST}/tail_audit_api_calls.md row arm={r['arm']} train_iter={r['train_iter']}")
    for _, r in ratio.iterrows():
        put(f"api_ratio.{r['method']}.{r['window'].replace(' ', '_')}.{r['quantity']}",
            {k: r[k] for k in ("iters", "K0_sum", "K5_sum", "K5_over_K0", "arithmetic")},
            source=(f"{_TBL_COST}/tail_audit_api_ratio.md row method={r['method']} window={r['window']} "
                    f"quantity={r['quantity']}"))
    for arm_l in list(dict.fromkeys(api["arm"])) if len(api) else []:
        d = api[api["arm"] == arm_l]
        put(f"api_totals.{arm_l}",
            {"iters": int((d["row_kind"] == "iteration").sum()),
             "oracle_calls_train": int(d["oracle_calls_train"].sum()),
             "oracle_input_Mchars": float(d["oracle_input_Mchars"].sum()),
             "patient_calls_total": int(d["patient_calls_total"].sum()),
             "n_candidates": int(d["n_candidates"].sum()) + int(d["n_candidates_eval_phase"].sum()),
             "arithmetic": "column sums over all rows of the arm incl. the final eval pass"},
            source=f"{_TBL_COST}/tail_audit_api_calls.md column sums for arm={arm_l}")
    # pooled eval-conv session-end reasons (from the eval CSVs; cross-checks arms/validity session_end_reasons)
    ev = pd.concat([_eval_conv_stats_cached(a) for a in _all_arms(arms)], ignore_index=True)
    if len(ev):
        ser = ev.groupby("arm")[["eval_ended_by_patient", "eval_ended_by_therapist", "eval_ended_none"]].sum()
        put("eval_session_end_reasons_pooled", {a: {c: int(ser.loc[a, c]) for c in ser.columns} for a in ser.index},
            source="tails.eval_conv_stats pooled over model_iters; cross-checks arms/validity/tables/<judge>/session_end_reasons.md")
    return N
