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
iteration appends ONE line to ``iteration_N/timing_sessions.jsonl`` describing what *it* did.
Cumulative cost is the sum over lines. Nothing is ever rewritten, so:

* a process that dies mid-iteration still leaves its finished phases recorded;
* two processes cannot race a read-modify-write and lose a session;
* the log is auditable -- you can see the resume boundaries rather than inferring them.

Wall-clock spans are recorded alongside the phase durations, so a reader can separate compute time
from idle time without guessing.

Usage (both trainers, at the end of an iteration)::

    from core.timing import log_session, cumulative_seconds

    log_session(iter_dir, generation_s=gen_time, training_s=train_time,
                pref_pair_s=pref_time)                   # PTO passes the third; GRPO omits it
    totals = cumulative_seconds(iter_dir)                # {'generation_s': ..., 'total_s': ...}

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
    "read_sessions",
    "cumulative_seconds",
    "metadata_fields",
]

#: One line per process that worked on the iteration. Append-only by contract.
SESSIONS_FILENAME = "timing_sessions.jsonl"

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
        Pass only what *this* process did. A resumed iteration that reloaded ``pairs.csv`` instead
        of rebuilding must log ``pref_pair_s=0.0`` -- the earlier session already recorded the
        build, and double-counting it is exactly the failure this file exists to prevent.

        Timing must never be able to fail a training run, so every error here is swallowed after a
        warning; the caller gets ``{}`` and should not branch on it.
    """
    now = time.time()
    rec = {
        "generation_s": float(generation_s or 0.0),
        "pref_pair_s": float(pref_pair_s or 0.0),
        "training_s": float(training_s or 0.0),
        "eval_gen_s": float(eval_gen_s or 0.0),
        "wall_start": float(started_at) if started_at else None,
        "wall_end": now,
        "wall_span_s": (now - float(started_at)) if started_at else None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
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
        * ``n_sessions`` -- how many processes appended a line at all;
        * ``n_sessions_production`` -- how many of them did production work.

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
    n_production = 0
    for s in sessions:
        for k in PHASE_KEYS:
            try:
                totals[k] += float(s.get(k) or 0.0)
            except (TypeError, ValueError):
                continue
        for k in PRODUCTION_PHASE_KEYS:
            try:
                if float(s.get(k) or 0.0) > 0.0:
                    n_production += 1
                    break
            except (TypeError, ValueError):
                continue
    totals["total_s"] = sum(totals[k] for k in PHASE_KEYS)
    totals["production_s"] = sum(totals[k] for k in PRODUCTION_PHASE_KEYS)
    totals["n_sessions"] = float(len(sessions))
    totals["n_sessions_production"] = float(n_production)
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
