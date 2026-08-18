"""timing.py — resume-proof per-iteration timing.

`iteration_metadata.json`'s `generation_time_s` / `training_time_s` / `pref_pair_time_s` are
**per-PROCESS**: they record only the session that happened to write the file. A crashed-and-resumed
iteration therefore reports a fraction of what it cost, and the error is not small — GRPO_LA5
iteration 1 logs 14,501 s for work that actually spanned 7.7 h, and PTO logs
``pref_pair_time_s = 3.2 s`` for a ~30 min preference build it reloaded from ``pairs.csv``.

That made the true cost of a run unrecoverable except by archaeology on artifact mtimes (which is
what ``eda_analysis/compute.py`` does, and has to keep doing for every run already on disk). This
module stops the bleeding for runs from here on.

**Design: an append-only session log, not a running total.** Each process that works on an iteration
appends ONE line to ``iteration_N/timing_sessions.jsonl`` describing what *it* did. Cumulative cost
is the sum over lines. Nothing is ever rewritten, so:

* a process that dies mid-iteration still leaves its work recorded;
* two processes cannot race a read-modify-write and lose a session;
* the log is auditable — you can see the resume boundaries rather than inferring them.

Wall-clock spans are recorded alongside the phase durations, so a reader can tell compute time from
idle time without guessing.

Usage (both trainers, at the end of an iteration)::

    from _shared.timing import log_session, cumulative_seconds

    log_session(iter_dir, generation_s=gen_time, training_s=train_time,
                pref_pair_s=pref_time)                       # PTO passes the third; GRPO omits it
    totals = cumulative_seconds(iter_dir)                    # {'generation_s': ..., 'total_s': ...}

Reading it back costs nothing and never raises: a missing or corrupt log returns zeros, so callers
can always fall back to the mtime reconstruction.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Dict, List, Optional

#: One line per process that worked on the iteration. Append-only by contract.
SESSIONS_FILENAME = "timing_sessions.jsonl"

#: The phases a session may report. Unknown keys are preserved but not summed into ``total_s``,
#: so adding a phase here is what makes it count — a silent typo in a caller cannot inflate a total.
PHASE_KEYS = ("generation_s", "pref_pair_s", "training_s")


def sessions_path(iter_dir: str) -> str:
    return os.path.join(iter_dir, SESSIONS_FILENAME)


def log_session(iter_dir: str, *, generation_s: float = 0.0, training_s: float = 0.0,
                pref_pair_s: float = 0.0, started_at: Optional[float] = None,
                note: str = "") -> dict:
    """Append this process's contribution to the iteration's timing log.

    Returns the record written (or an empty dict if the write failed — timing must never be able to
    fail a training run, so every error here is swallowed after a warning).
    """
    now = time.time()
    rec = {
        "generation_s": float(generation_s or 0.0),
        "pref_pair_s": float(pref_pair_s or 0.0),
        "training_s": float(training_s or 0.0),
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
    """Every session recorded for this iteration; ``[]`` if absent. Skips unparseable lines."""
    fp = sessions_path(iter_dir)
    if not os.path.isfile(fp):
        return []
    out = []
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
    """Phase totals summed over every session, plus ``total_s`` and ``n_sessions``.

    ``n_sessions > 1`` means the iteration was resumed — which is exactly the case where
    ``iteration_metadata.json``'s own timings are wrong, so it doubles as the flag for "do not
    trust the per-process number here".
    """
    sessions = read_sessions(iter_dir)
    totals = {k: 0.0 for k in PHASE_KEYS}
    for s in sessions:
        for k in PHASE_KEYS:
            try:
                totals[k] += float(s.get(k) or 0.0)
            except (TypeError, ValueError):
                continue
    totals["total_s"] = sum(totals[k] for k in PHASE_KEYS)
    totals["n_sessions"] = float(len(sessions))
    return totals


def metadata_fields(iter_dir: str) -> Dict[str, float]:
    """Cumulative fields to merge into ``iteration_metadata.json``, namespaced so they cannot be
    confused with the per-process ones already there.

    Returns ``{}`` when there is no log, so a caller can splat it unconditionally.
    """
    totals = cumulative_seconds(iter_dir)
    if not totals.get("n_sessions"):
        return {}
    return {
        "cumulative_generation_time_s": round(totals["generation_s"], 3),
        "cumulative_pref_pair_time_s": round(totals["pref_pair_s"], 3),
        "cumulative_training_time_s": round(totals["training_s"], 3),
        "cumulative_total_time_s": round(totals["total_s"], 3),
        "n_timing_sessions": int(totals["n_sessions"]),
    }
