"""timing.py -- resume-proof per-iteration timing.

Exp3 stamped an iteration's cost into ``iteration_metadata.json`` as ``generation_time_s`` /
``training_time_s`` / ``pref_pair_time_s``. Those fields are **per-PROCESS**: they record only the
session that happened to write the file. A crashed-and-resumed iteration therefore reports a
fraction of what it cost, and the error is not small -- GRPO_LA5 iteration 1 logged 14,501 s for
work that actually spanned 7.7 h, and PTO logged ``pref_pair_time_s = 3.2 s`` for a ~30 min
preference build it had reloaded from ``pairs.csv``. Recovering the real numbers afterwards took
1,336 LOC of archaeology over artifact mtimes, and that reconstruction is guesswork by
construction: it cannot see a phase that leaves no file behind, and it cannot tell a resume gap
from an idle Drive re-sync.

**In Exp4 this module is the only timing record, and it is written from the first iteration of the
first arm.** There is no mtime reconstruction here and none is planned. What is not logged through
``log_session`` is simply not recorded -- so every trainer phase (conversation generation,
preference-pair building, training, the post-loop eval generate) must log itself.

**Design: an append-only session log, not a running total.** Each process that works on an
iteration appends a line to ``iteration_N/timing_sessions.jsonl`` **as each phase completes** --
one line after generation, one after the preference build (PTO), one after training. Cumulative
cost is the sum over lines; every line carries a per-process token, so process-level counters
(``n_sessions``, ``n_sessions_production``) count distinct PROCESSES, not lines. Nothing is ever
rewritten, so:

* a process that dies mid-iteration still leaves its **finished phases** recorded -- this is why
  logging happens per phase, not once at iteration end: a Colab preemption during training must
  not erase the 50-minute generation phase that already happened, or the compute/cost axis
  silently undercounts exactly the preempted arms (the Exp3 defect this module exists to fix);
* two processes cannot race a read-modify-write and lose a session;
* the log is auditable -- you can see the resume boundaries rather than inferring them.

Wall-clock spans are recorded alongside the phase durations, so a reader can separate compute time
from idle time without guessing.

**The training phase is also logged INCREMENTALLY, at every checkpoint save.** "One line after
training" protects the finished phases, but the training phase itself is the longest one in GRPO
and it ends with ``trainer.train()`` returning -- a Colab preemption at step 90 of 135 would lose
the whole phase, and the resumed process would log only its own 45 steps. That is Exp3's
per-process undercount moved one phase to the right. So the trainers' ``on_save`` callback calls
:func:`log_training_progress` with the elapsed time of THIS process's training phase, and the
module appends only the INCREMENT since the last partial line it wrote (tracked per process, per
iteration directory). When ``trainer.train()`` returns, :func:`finalize_training` logs the
remaining delta. The sum over all of a process's training lines is therefore exactly the phase's
elapsed time, with no double counting, and a process killed between two saves loses at most one
save interval. Because every line carries the per-process token, the extra lines do not inflate
``n_sessions`` / ``n_sessions_production``: those count distinct tokens.

**The per-process ledger is per training ATTEMPT, not per process lifetime.** The increment
bookkeeping above remembers, per iteration directory, how much ``training_s`` this process has
already written. That memory must be reset when the same process starts the training phase of the
same iteration AGAIN -- the documented in-kernel resume (fix the cause, re-run the loop cell) does
exactly that, with a fresh ``train_started_at`` clock that restarts at 0 while the ledger still
holds the crashed attempt's total. Without the reset every partial of the second attempt reads as a
negative delta (nothing is written) and :func:`finalize_training` clamps to a zero line with a
warning, so 120 s of real training records only the 70 s of the first attempt. The trainers
therefore call :func:`begin_training_phase` at the start of EVERY training attempt, immediately
before the attempt's training clock starts; it pops this process's ledger entry for that iteration
and touches nothing on disk (the earlier attempt's lines stay on record, as they should).

Usage (both trainers, after EACH phase)::

    from core.timing import (begin_training_phase, log_session, log_training_progress,
                             finalize_training)

    log_session(iter_dir, generation_s=gen_time)         # right after the generation phase
    log_session(iter_dir, pref_pair_s=pref_time)         # PTO only, right after the build

    begin_training_phase(iter_dir)                       # EVERY training attempt, then start the clock
    train_started_at = time.time()
    # inside the TrainerCallback.on_save, with the phase's start time captured once:
    log_training_progress(iter_dir, elapsed_s=time.time() - train_started_at,
                          note=f"checkpoint-{state.global_step}")
    # right after trainer.train() returns -- INSTEAD of log_session(training_s=...):
    finalize_training(iter_dir, time.time() - train_started_at, started_at=iter_start,
                      note=f"training, iteration {n}")
    totals = cumulative_seconds(iter_dir)                # {'generation_s': ..., 'total_s': ...}

⚠ A trainer that wires ``on_save`` MUST end the phase with :func:`finalize_training`, never with
``log_session(training_s=total)`` -- the latter would re-log everything the partial lines already
recorded.

Reading it back costs nothing and never raises: a missing or corrupt log returns zeros, so the EDA
can read an arm that is still training, and a telemetry failure can never fail a run.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Dict, List, Optional

__all__ = [
    "SESSIONS_FILENAME",
    "PHASE_KEYS",
    "PRODUCTION_PHASE_KEYS",
    "sessions_path",
    "log_session",
    "begin_training_phase",
    "log_training_progress",
    "finalize_training",
    "read_sessions",
    "cumulative_seconds",
    "metadata_fields",
]

#: One line per completed phase. Append-only by contract.
SESSIONS_FILENAME = "timing_sessions.jsonl"

#: Stamped into every record so the reader can group lines by the process that wrote them.
#: PID alone is not enough -- Windows reuses PIDs across days, and an arm spans days.
_PROCESS_TOKEN = f"{socket.gethostname()}:{os.getpid()}:{int(time.time())}"

#: The phases a session may report. Unknown keys are preserved in the record but not summed into
#: ``total_s``, so adding a phase here is what makes it count -- a silent typo in a caller cannot
#: inflate a total. ``metadata_fields`` derives its output names from this tuple, so one edit here
#: is enough to carry a new phase all the way through to ``iteration_metadata.json``.
PHASE_KEYS = ("generation_s", "pref_pair_s", "training_s", "eval_gen_s")

#: The subset of :data:`PHASE_KEYS` that is a cost of PRODUCING a policy, as opposed to a cost of
#: MEASURING one. ``eval_gen_s`` is the odd one out and is deliberately excluded here.
#:
#: A state's conversations are generated at the START of the next iteration, so state ``j``'s
#: measurement is billed to iteration ``j+1`` and falls outside the ``1..j`` cumulative sum -- for
#: every state except the last. The post-loop final-eval pass has no iteration ``N+1`` to be billed
#: to and logs itself against ``iteration_N``, so summing ``total_s`` would price exactly one point
#: per arm -- the endpoint every budget sweep is read at -- under a different rule from all the
#: others, and shift it right by a whole generation pass. Sum ``production_s`` for cost, and read
#: ``eval_gen_s`` separately as the measurement axis.
#:
#: One known ambiguity: a ``tools/generate_convs.py`` repair on an arm that later RESUMES bills
#: its pass as ``eval_gen_s`` against iteration ``k``, and iteration ``k+1`` then reloads those
#: same conversations with a near-zero ``generation_s`` -- the pass's cost sits on the measurement
#: axis instead of ``production_s``. The repair session's ``note`` names the tool, so a cost
#: analysis can reclassify it when it matters.
PRODUCTION_PHASE_KEYS = ("generation_s", "pref_pair_s", "training_s")


def sessions_path(iter_dir: str) -> str:
    """Path of the append-only session log for one iteration directory."""
    return os.path.join(iter_dir, SESSIONS_FILENAME)


def log_session(iter_dir: str, *, generation_s: float = 0.0, training_s: float = 0.0,
                pref_pair_s: float = 0.0, eval_gen_s: float = 0.0,
                started_at: Optional[float] = None, note: str = "") -> dict:
    """Append this process's contribution to the iteration's timing log.

    Args:
        iter_dir: ``iteration_N/`` directory; created if missing.
        generation_s: seconds spent simulating this iteration's training conversations.
        training_s: seconds spent inside the TRL trainer.
        pref_pair_s: seconds spent building preference pairs (PTO only; 0.0 for GRPO).
        eval_gen_s: seconds spent in a generate-only eval pass attributed to this iteration.
        started_at: ``time.time()`` at the start of this process's work on the iteration. Supplied
            so the record can distinguish compute time from wall-clock span.
        note: free text, e.g. ``"resumed from checkpoint-40"``.

    Returns:
        The record written, or an empty dict if the write failed.

    Notes:
        Call this **as each phase completes**, not once at iteration end -- a process killed
        during training must still leave its finished generation phase on record. Multiple calls
        from one process are grouped by the per-process token, so per-phase logging does not
        inflate the process counters.

        Pass only what *this* process did. A resumed iteration that reloaded ``pairs.csv`` instead
        of rebuilding must log ``pref_pair_s=0.0`` (or skip the call) -- the earlier session
        already recorded the build, and double-counting it is exactly the failure this file exists
        to prevent.

        Timing must never be able to fail a training run, so every error here is swallowed after a
        warning; the caller gets ``{}`` and should not branch on it.

        ⚠ For the TRAINING phase, a trainer whose ``on_save`` callback calls
        :func:`log_training_progress` must close the phase with :func:`finalize_training`, not
        with ``training_s=`` here -- the partial lines already hold most of the phase.
    """
    return _append_record(
        iter_dir,
        generation_s=generation_s,
        pref_pair_s=pref_pair_s,
        training_s=training_s,
        eval_gen_s=eval_gen_s,
        started_at=started_at,
        note=note,
    )


def _append_record(iter_dir: str, *, generation_s: float = 0.0, training_s: float = 0.0,
                   pref_pair_s: float = 0.0, eval_gen_s: float = 0.0,
                   started_at: Optional[float] = None, note: str = "",
                   partial: bool = False) -> dict:
    """Build one session line and append it. The single writer behind every public logger.

    ``partial`` marks a mid-phase increment written by :func:`log_training_progress`; readers
    sum :data:`PHASE_KEYS` regardless, so the flag is purely for audit (it lets a human see the
    save cadence and the resume boundary in the raw log).
    """
    now = time.time()
    rec = {
        "generation_s": float(generation_s or 0.0),
        "pref_pair_s": float(pref_pair_s or 0.0),
        "training_s": float(training_s or 0.0),
        "eval_gen_s": float(eval_gen_s or 0.0),
        "partial": bool(partial),
        "wall_start": float(started_at) if started_at else None,
        "wall_end": now,
        "wall_span_s": (now - float(started_at)) if started_at else None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process": _PROCESS_TOKEN,
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "note": note,
    }
    try:
        os.makedirs(iter_dir, exist_ok=True)
        with open(sessions_path(iter_dir), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as e:                                   # never break a run over telemetry
        print(f"  [timing] WARNING: could not append session log ({e})")
        return {}
    return rec


# ------------------------------------------------------------------------------
#  Incremental training-phase logging (checkpoint-aligned)
# ------------------------------------------------------------------------------

#: ``training_s`` already written by THIS process, per iteration directory (normalised absolute
#: path), for the CURRENT training attempt. Process-local by construction: a resumed process
#: starts from zero, and its own partial lines are the only ones it must not repeat. The value is
#: the attempt's cumulative elapsed training time at the last line written, so an increment is
#: ``elapsed_s - value``. :func:`begin_training_phase` pops the entry so a second attempt in the
#: same process (the in-kernel resume) starts from zero as well.
_TRAINING_LOGGED: Dict[str, float] = {}


def _iter_key(iter_dir: str) -> str:
    return os.path.normcase(os.path.abspath(iter_dir))


def begin_training_phase(iter_dir: str) -> None:
    """Start a training ATTEMPT: forget what this process logged for *iter_dir* so far.

    Call it at the start of every training attempt, immediately before the attempt's
    ``train_started_at`` clock is taken -- i.e. before the first :func:`log_training_progress`
    that will be measured on that clock. Nothing on disk is touched: the lines an earlier attempt
    wrote stay on record and still sum into ``cumulative_seconds``; only the per-process "already
    logged" ledger is reset, so the new attempt's partials are measured against zero.

    Args:
        iter_dir: ``iteration_N/`` directory.

    Notes:
        Why this is needed: the ledger is keyed by iteration directory only, and the trainers'
        documented in-kernel resume (fix the cause, re-run the loop cell) runs a SECOND training
        attempt for the same iteration in the SAME process with a fresh clock. Without the reset
        the ledger still holds the crashed attempt's total, every new partial reads as a negative
        delta (``{}``, nothing written) and :func:`finalize_training` clamps to a zero line -- the
        second attempt's whole training time vanishes from the cost record. The negative-delta
        warning in :func:`finalize_training` is kept for what it was meant for: two calls within
        one attempt fed different clocks.

        A process that only ever trains an iteration once (the normal Colab run, and a resume in a
        fresh kernel) sees no difference: there is nothing to pop.
    """
    _TRAINING_LOGGED.pop(_iter_key(iter_dir), None)


def log_training_progress(iter_dir: str, *, elapsed_s: float, note: str = "") -> dict:
    """Append the training time accrued since this process's previous partial line.

    Call it from the trainer's ``on_save`` callback (every ``save_steps`` optimizer steps) with
    ``elapsed_s = time.time() - train_started_at``, where ``train_started_at`` is the clock this
    process started its training phase on -- the SAME clock the final ``training_s`` is read off,
    so the partial lines and the closing delta from :func:`finalize_training` sum to exactly the
    phase's elapsed time. The attempt must have been opened with :func:`begin_training_phase`
    (a second attempt on the same iteration in the same process otherwise measures against the
    previous attempt's total).

    Args:
        iter_dir: ``iteration_N/`` directory.
        elapsed_s: seconds since this process's training phase began (cumulative, not a delta --
            the delta is computed here against the last line written for *iter_dir*).
        note: free text, e.g. ``"checkpoint-40"``.

    Returns:
        The record written, ``{}`` when nothing new had accrued (``elapsed_s`` at or below the
        last logged value) or when the write failed.

    Notes:
        Each line carries the per-process token, so however many partial lines a process writes,
        ``cumulative_seconds`` counts it as ONE production session. A process killed between two
        saves loses at most one save interval of ``training_s`` -- and its finished partials stay
        on record, which is the whole point.

        A ``training_s`` sum built from these lines is then a lower bound on the phase's true
        cost by at most one save interval per preemption, instead of Exp3's per-process figure
        that could be off by the entire pre-crash session.
    """
    key = _iter_key(iter_dir)
    last = _TRAINING_LOGGED.get(key, 0.0)
    increment = float(elapsed_s) - last
    if increment <= 0.0:
        return {}
    rec = _append_record(
        iter_dir,
        training_s=increment,
        note=f"training partial: {note}" if note else "training partial",
        partial=True,
    )
    if rec:
        _TRAINING_LOGGED[key] = float(elapsed_s)
    return rec


def finalize_training(iter_dir: str, total_s: float, *, started_at: Optional[float] = None,
                      note: str = "") -> dict:
    """Close this process's training phase: log only what the partial lines have not.

    Args:
        iter_dir: ``iteration_N/`` directory.
        total_s: this process's whole training-phase elapsed time, on the same clock the
            ``on_save`` partials were measured on.
        started_at: ``time.time()`` at the start of this process's work on the iteration, for
            the wall-clock span (as in :func:`log_session`).
        note: free text, e.g. ``"training, iteration 3"``.

    Returns:
        The record written (``training_s`` = the remaining delta, possibly 0.0), or ``{}`` if the
        write failed.

    Notes:
        A line is written even when the delta is zero, so the phase's completion is on record
        with its note; a zero line changes no sum and, because the earlier partials from this
        process already carry its token, no counter. Called with no preceding partials (a trainer
        that never wired ``on_save``) this is exactly ``log_session(training_s=total_s)``.

        Idempotent per process and training attempt: the logged total is remembered, so a second
        call with the same ``total_s`` writes a zero delta rather than re-logging the phase.
        ``total_s`` below the partials already logged is clamped to a zero delta with a warning
        -- within one attempt it means the two calls were fed different clocks, which is a caller
        bug worth seeing. (A NEW attempt on the same iteration is not a clock mismatch: it must
        open with :func:`begin_training_phase`, which is what stops the warning from firing --
        and the attempt's time from being lost -- on the in-kernel resume.)
    """
    key = _iter_key(iter_dir)
    last = _TRAINING_LOGGED.get(key, 0.0)
    delta = float(total_s) - last
    if delta < 0.0:
        print(
            f"  [timing] WARNING: finalize_training total_s={float(total_s):.1f} is below the "
            f"{last:.1f}s already logged by partial lines for {iter_dir}; logging a zero delta. "
            f"log_training_progress and finalize_training must share one clock."
        )
        delta = 0.0
    rec = _append_record(iter_dir, training_s=delta, started_at=started_at, note=note)
    if rec:
        _TRAINING_LOGGED[key] = max(last, float(total_s))
    return rec


def read_sessions(iter_dir: str) -> List[dict]:
    """Every session recorded for this iteration; ``[]`` if absent. Skips unparseable lines.

    Notes:
        A process killed mid-write leaves a torn final line. That is a normal state for an
        append-only log being read while a run is live, not an error -- the line is dropped and the
        completed sessions before it are returned.
    """
    fp = sessions_path(iter_dir)
    if not os.path.isfile(fp):
        return []
    out: List[dict] = []
    try:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue                                 # a torn final line, not a reason to fail
    except OSError:
        return []
    return out


def cumulative_seconds(iter_dir: str) -> Dict[str, float]:
    """Phase totals summed over every session, plus the four derived counters below.

    Returns:
        One key per :data:`PHASE_KEYS`, plus

        * ``total_s`` -- every phase, measurement included;
        * ``production_s`` -- :data:`PRODUCTION_PHASE_KEYS` only, i.e. what it cost to PRODUCE
          this iteration's policy. **This is the cost axis**; see the constant's note;
        * ``n_sessions`` -- how many distinct PROCESSES appended a line at all (lines are grouped
          by the per-process token, since one process logs each phase separately);
        * ``n_sessions_production`` -- how many of those processes did production work.

    Notes:
        ⚠ **Resume is ``n_sessions_production > 1``, not ``n_sessions > 1``.** The post-loop
        final-eval pass appends a second session to the LAST training iteration of every arm, and
        every ``tools/generate_convs.py`` repair appends another -- so the plain session count
        reports the last iteration of a perfectly healthy arm as resumed, which then tells a
        reader that its per-process numbers are undercounts when they are not.

        A genuinely resumed iteration is still the flag any cost analysis wants: its wall-clock
        span includes time nobody was computing, so a span read off directory mtimes would be an
        overestimate and any single process's own numbers an underestimate.
    """
    sessions = read_sessions(iter_dir)
    totals = {k: 0.0 for k in PHASE_KEYS}
    all_procs: set = set()
    production_procs: set = set()

    def _proc_key(rec: dict, idx: int):
        # Lines without a token fall back to (host, pid); a line missing both counts alone.
        if rec.get("process"):
            return rec["process"]
        if rec.get("pid") is not None:
            return (rec.get("host"), rec.get("pid"))
        return ("line", idx)

    for i, s in enumerate(sessions):
        key = _proc_key(s, i)
        all_procs.add(key)
        for k in PHASE_KEYS:
            try:
                totals[k] += float(s.get(k) or 0.0)
            except (TypeError, ValueError):
                continue
        for k in PRODUCTION_PHASE_KEYS:
            try:
                if float(s.get(k) or 0.0) > 0.0:
                    production_procs.add(key)
                    break
            except (TypeError, ValueError):
                continue
    totals["total_s"] = sum(totals[k] for k in PHASE_KEYS)
    totals["production_s"] = sum(totals[k] for k in PRODUCTION_PHASE_KEYS)
    totals["n_sessions"] = float(len(all_procs))
    totals["n_sessions_production"] = float(len(production_procs))
    return totals


def metadata_fields(iter_dir: str) -> Dict[str, float]:
    """Cumulative fields to merge into ``iteration_metadata.json``.

    Field names are derived from :data:`PHASE_KEYS` (``generation_s`` -> ``cumulative_generation_time_s``)
    and namespaced with ``cumulative_`` so they can never be confused with a per-process figure.

    Returns ``{}`` when there is no log, so a caller can splat it unconditionally.
    """
    totals = cumulative_seconds(iter_dir)
    if not totals.get("n_sessions"):
        return {}
    out: Dict[str, float] = {}
    for k in PHASE_KEYS:
        out[f"cumulative_{k[:-2]}_time_s"] = round(totals[k], 3)
    out["cumulative_total_time_s"] = round(totals["total_s"], 3)
    out["n_timing_sessions"] = int(totals["n_sessions"])
    return out
