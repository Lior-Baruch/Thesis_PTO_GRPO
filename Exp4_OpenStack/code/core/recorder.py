"""recorder.py -- per-generation EDA capture for both Exp4 trainers.

Both optimizers throw almost everything away. GRPO keeps a gradient step and forgets the G
completions it computed rewards for; PTO keeps one (chosen, rejected) pair and forgets the other
M-2 branches. Every question the EDA actually asks -- how faithful is a short-cut training reward,
what does look-ahead change about the ranking, how often does a group collapse to zero variance --
needs the candidates that were discarded, together with the score the oracle gave them and the
K-turn tail the oracle actually read. Without this module those artifacts do not exist anywhere:
re-deriving them means paying for the whole iteration again.

``EDARecorder`` is a buffer, not a logger. The hot path calls :meth:`append` (a list append -- safe
inside the async reward fn: it never blocks the event loop and never touches the Colab Drive-FUSE
mount). GRPO writes the buffer exactly once per iteration, atomically, at the end
(:meth:`EDARecorder.flush`); PTO appends the rows of each trunk depth as it completes
(:meth:`EDARecorder.append_to_disk`, so a preempted build keeps its rows on disk) and normalises the
file on resume (:meth:`EDARecorder.rewrite`). Both paths produce the same line format.

One branch-centric row serves all three producers (GRPO groups, PTO greedy trunks, PTO independent
branches). The prefix is stored ONCE per row and the candidates are nested under it, because the G
(or M) rows of a group share a prefix that can be several thousand tokens -- flattening it would
multiply the file by G::

    {"phase": "group"|"tree"|"independent", "iteration": 3,
     "conversation_id": 17, "persona_id": 7, "branch_id": 0,
     "epoch": 1.0|None, "prefix": "[PATIENT]: ...\\n\\n[THERAPIST]: ...",
     "group_mean": 2.41|None, "group_std": 0.52|None,     # GRPO only
     "chosen_idx": 2|None,
     "candidates": [
       {"idx": 0, "completion": "...", "score": 3.4|None,
        "sub_scores": {"1": 3.0, "2": 3.8}|None,
        "reward_used": 3.3,                               # only when != score (GRPO substitution)
        "role": "chosen"|"rejected"|"neither"|None,       # PTO only
        "oracle": {"success": true, "attempts": 1},
        "ended_by_candidate": false,                      # EVERY candidate (written by core.reward)
        "not_graded_reason": "oracle_failed"|"patient_error"|"gpu_error"|"prompt_overflow"
                             |"parse_error",             # ONLY when score is null
        "lookahead": {"tail": "..."|None, "realized_turns": 5, "ended_early": false,
                      "stop_reason": ""|"session_ended"|"degenerate"|"patient_error"
                                     |"gpu_error"|"prompt_overflow"|"parse_error"}|None}, ...]}

Two candidate keys are written by ``core.reward.CandidateScore.to_record`` ON TOP of
:func:`build_candidate`'s dict (the recorder writes candidates as given):

* ``ended_by_candidate`` (every candidate) -- True when the completion itself contained
  ``SESSION ENDED``: the oracle graded only the text BEFORE the keyword, no rollout was run
  (at K>0 the ``lookahead`` dict is a zero-turn record: ``tail ""``, ``realized_turns 0``,
  ``ended_early true``, ``stop_reason "session_ended"``), and ``completion`` still carries the
  keyword plus the model's explanation -- the row records what the policy emitted, and PTO's
  trunk advance reads the keyword to freeze the trunk. What trained on it differs by method: a
  DPO pair uses the SPLIT text (``pto_trainer._pair_text``, exactly what the oracle graded),
  while GRPO trains on the completion ids TRL sampled (this string) with its reward computed on
  the split text.
* ``not_graded_reason`` (only when ``score`` is null) -- ``"oracle_failed"`` = the grader was
  asked and could not answer; ``"patient_error"`` / ``"gpu_error"`` / ``"prompt_overflow"`` /
  ``"parse_error"`` = the K-turn look-ahead froze before the oracle was ever called
  (``core.lookahead``'s ``NOT_GRADED_STOP_REASONS``), so the candidate was left ungraded rather
  than scored on a truncated future. Both kinds count as failures against ``min_success_ratio``.
  On PTO rows ``"gpu_error"`` / ``"prompt_overflow"`` can also come from the BRANCH SAMPLER
  (``pto_trainer._apply_sampling_failures``: the candidate was never generated -- the sampler
  failed at chunk size 1, or the newest turn alone exceeded ``therapist_max_input_tokens``);
  those rows carry ``score null``, ``degenerate false``, ``oracle {success: false, attempts: 0}``
  and ``lookahead null``, which is how they are told apart from a look-ahead failure.

``lookahead.stop_reason`` (K>0 only) is the simulator's own verdict; ``""`` means the rollout ran
to K turns.

Reconstruct the exact text the oracle scored as
``prefix + "\\n\\n[THERAPIST]: " + completion + (tail or "")`` -- which is why the transcript labels
and the ``"\\n\\n"`` joiner in ``core/conversations.py`` are load-bearing here too. ⚠ For an
``ended_by_candidate`` row split ``completion`` at the keyword first
(``core.lookahead.split_session_end``): the graded text stops there, and ``tail`` is empty.

``persona_id`` is an Exp4 addition (Exp3 rows carried only ``conversation_id``, and its EDA had to
replay the per-iteration persona shuffle to recover which patient a branch belonged to). Record it
and cross-iteration pairing is a join, not archaeology.

WARNING -- ``branch_id`` is trunk DEPTH for PTO, not a unique id. It restarts at 0 for every trunk,
so it repeats across conversations within one iteration; any per-branch aggregation must key on
``(conversation_id, branch_id)``. For GRPO it is a running group counter and is unique within the
iteration, but downstream code should key both methods the same way regardless.

Pure stdlib on purpose: the EDA imports this module read-side, and it must not drag torch or numpy
into a notebook that only wants to parse JSONL. Numpy/torch scalars arriving from the trainer are
coerced by duck-typing (:func:`to_jsonable`), never by importing numpy.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PHASE_GROUP",
    "PHASE_TREE",
    "PHASE_INDEPENDENT",
    "GRPO_PHASES",
    "PTO_PHASES",
    "NOT_GRADED_STOP_REASONS",
    "EDARecorder",
    "to_jsonable",
    "build_candidate",
    "build_branch_record",
    "iter_jsonl",
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASES -- what produced the row                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# The phase is the ONLY method discriminator in the schema (Exp3 also carried a
# redundant "method" string). aggregate() uses it to decide which method-specific
# scalars a buffer can support, so a GRPO run never emits pto/* zeros and vice versa.

PHASE_GROUP = "group"              # GRPO: one row per prompt-group, per epoch
PHASE_TREE = "tree"                # PTO greedy: one row per branch point on a trunk
PHASE_INDEPENDENT = "independent"  # PTO independent: one row per branch point on a fixed conv

GRPO_PHASES = (PHASE_GROUP,)
PTO_PHASES = (PHASE_TREE, PHASE_INDEPENDENT)

#: ``lookahead.stop_reason`` values for which the candidate was left UNGRADED (the simulator
#: froze before the oracle ran). Mirrors ``core.lookahead.NOT_GRADED_STOP_REASONS`` -- duplicated
#: rather than imported because that module pulls torch in and this one must stay stdlib-only for
#: the read-side EDA. Keep the two in step.
NOT_GRADED_STOP_REASONS = frozenset({"patient_error", "gpu_error", "prompt_overflow", "parse_error"})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  JSON COERCION                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def to_jsonable(obj: Any) -> Any:
    """Coerce *obj* into something :func:`json.dumps` can write with ``allow_nan=False``.

    Total by construction -- anything unrecognised degrades to ``str(obj)`` -- because a record
    that fails to serialize would take down a training run several GPU-hours in, and losing the
    EDA row is always the cheaper failure.

    Handles, in order: NaN/Inf floats -> ``None`` (JSON has no literal for them, and Python's
    ``json`` would otherwise emit bare ``NaN``, which is invalid JSON that pandas/`json.loads`
    accept but strict parsers reject); dicts (keys coerced too, since a numpy key raises); lists,
    tuples and sets; then duck-typed ``.tolist()`` / ``.item()`` so numpy and torch scalars and
    arrays convert WITHOUT importing numpy or torch into a read-only EDA process.

    Notes:
        ``numpy.float64`` is a subclass of ``float``, so NaN from numpy is caught by the float
        branch; ``numpy.int64`` is not a subclass of ``int`` and is caught by the duck-typed one.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, (str, int, float, bool)) or k is None else str(to_jsonable(k))
            if isinstance(key, float) and not math.isfinite(key):
                key = str(key)
            out[key] = to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    for attr in ("tolist", "item"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return to_jsonable(fn())
            except (TypeError, ValueError, RuntimeError):
                pass
    return str(obj)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ROW BUILDERS (optional convenience -- the schema is the contract)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_candidate(
    idx: int,
    completion: Optional[str],
    *,
    score: Optional[float] = None,
    sub_scores: Optional[Dict[Any, float]] = None,
    role: Optional[str] = None,
    lookahead: Optional[Dict[str, Any]] = None,
    oracle_success: Optional[bool] = None,
    oracle_attempts: Optional[int] = None,
    reward_used: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one nested candidate dict in the schema above.

    Args:
        score: the RAW oracle score, except a degenerate completion floored to
            ``oracle.REWARD_FLOOR``, and ``None`` when the grader failed. Keep ``sub_scores``
            raw-oracle, so a floored row is identifiable as ``score == floor`` with
            per-questionnaire scores absent.
        reward_used: the number the trainer actually optimised, when it differs from *score*.
            GRPO only, and in practice only for a failed grader call whose group mean
            ``core.reward.rewards_for_trl`` substituted -- trl 1.4.0 has no way to drop a sample,
            so ``None`` would be optimised as 0.0. Absent when it equals *score*; a consumer
            wanting "what was optimised" reads ``reward_used`` falling back to ``score``.
        role: PTO only -- "chosen" | "rejected" | "neither". ``"rejected"`` is what
            :meth:`EDARecorder.aggregate` counts as "a preference pair was emitted at this branch",
            so set it only when the tau filter actually passed.
        lookahead: ``{"tail": str|None, "realized_turns": int, "ended_early": bool}`` (``"k"``
            and ``"stop_reason"`` keys are accepted and preserved). Pass ``None`` at K=0 -- an
            absent dict is what marks a no-look-ahead run, and the look-ahead scalars are then
            simply not emitted.

    Notes:
        ``core.reward.CandidateScore.to_record`` adds two keys this builder does not know about:
        ``ended_by_candidate`` (always) and ``not_graded_reason`` (only when *score* is None) --
        see the module docstring. The recorder writes whatever dict it is handed, so a producer
        that builds candidates here and wants those keys sets them on the result.
    """
    cand: Dict[str, Any] = {
        "idx": int(idx),
        "completion": completion,
        "score": score,
        "sub_scores": sub_scores,
    }
    if reward_used is not None:
        cand["reward_used"] = float(reward_used)
    if role is not None:
        cand["role"] = role
    if oracle_success is not None or oracle_attempts is not None:
        cand["oracle"] = {
            "success": bool(oracle_success),
            "attempts": int(oracle_attempts or 0),
        }
    cand["lookahead"] = lookahead
    return cand


def build_branch_record(
    *,
    phase: str,
    iteration: int,
    conversation_id: Any,
    persona_id: Optional[int],
    branch_id: int,
    prefix: Optional[str],
    candidates: Sequence[Dict[str, Any]],
    chosen_idx: Optional[int] = None,
    epoch: Optional[float] = None,
    group_mean: Optional[float] = None,
    group_std: Optional[float] = None,
    eval_pass: bool = False,
) -> Dict[str, Any]:
    """Build one branch row (prefix once, candidates nested).

    Args:
        phase: one of :data:`PHASE_GROUP`, :data:`PHASE_TREE`, :data:`PHASE_INDEPENDENT`. This is
            the method discriminator -- a wrong phase silently moves a row into the other method's
            aggregate bucket.
        persona_id: the STABLE 0..95 persona index, not the shuffled processing index.
        branch_id: trunk DEPTH for PTO (repeats across conversations), running group counter for
            GRPO. Never assume it is unique on its own.
        group_mean / group_std: GRPO only -- the group statistics TRL's advantage was computed
            from. Leaving them ``None`` on a PTO row is what keeps the GRPO scalars honest.
        eval_pass: True for a group scored during TRL's ``evaluate()`` -- held-out prompts, policy
            in eval mode, no gradient. Written on EVERY row (never omitted) so a reader can filter
            on it without having to know that absence means False: an aggregate that pools the two
            reports a blend of on-policy and held-out candidates under a training-only name.
    """
    return {
        "phase": phase,
        "iteration": int(iteration),
        "conversation_id": conversation_id,
        "persona_id": (None if persona_id is None else int(persona_id)),
        "branch_id": int(branch_id),
        "epoch": (None if epoch is None else float(epoch)),
        "eval_pass": bool(eval_pass),
        "prefix": prefix,
        "group_mean": group_mean,
        "group_std": group_std,
        "chosen_idx": (None if chosen_idx is None else int(chosen_idx)),
        "candidates": list(candidates),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CLASSIFIERS + SMALL STATS (stdlib only)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _is_grpo_branch(rec: Dict[str, Any]) -> bool:
    """A row is GRPO's if its phase says so, or (legacy/defensive) it carries group stats."""
    phase = rec.get("phase")
    if phase in GRPO_PHASES:
        return True
    if phase in PTO_PHASES:
        return False
    return rec.get("group_mean") is not None or rec.get("group_std") is not None


def _is_pto_branch(rec: Dict[str, Any]) -> bool:
    """A row is PTO's if its phase says so, or a candidate carries a preference ``role``."""
    phase = rec.get("phase")
    if phase in PTO_PHASES:
        return True
    if phase in GRPO_PHASES:
        return False
    return any(c.get("role") is not None for c in rec.get("candidates") or [])


def _pair_emitted(rec: Dict[str, Any]) -> bool:
    """Did this PTO branch point produce a (chosen, rejected) pair, or did tau filter it out?

    Three encodings are accepted so the PTO trainer can use whichever is natural at its call site:
    an explicit ``pair_emitted`` flag, a non-None ``rejected_idx``, or a candidate whose ``role``
    is ``"rejected"``.
    """
    if rec.get("pair_emitted") is not None:
        return bool(rec["pair_emitted"])
    if rec.get("rejected_idx") is not None:
        return True
    return any(c.get("role") == "rejected" for c in rec.get("candidates") or [])


def _candidate_success(cand: Dict[str, Any]) -> bool:
    """Did the oracle return a usable score for this candidate?

    Prefers the explicit ``oracle.success`` flag, then a flat ``success`` key, and falls back to
    "a score is present" -- so a producer that only records scores still yields a meaningful
    success rate rather than a constant 0.
    """
    oracle = cand.get("oracle")
    if isinstance(oracle, dict) and "success" in oracle:
        return bool(oracle["success"])
    if "success" in cand:
        return bool(cand["success"])
    return cand.get("score") is not None


def _lookahead_of(cand: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the candidate's look-ahead dict when look-ahead actually ran, else None.

    At K=0 the producer attaches no dict at all; a dict carrying an explicit ``k == 0`` is also
    treated as "did not run", so a K=0 arm never reports look-ahead scalars.
    """
    la = cand.get("lookahead")
    if not isinstance(la, dict):
        return None
    if "k" in la and not (la.get("k") or 0) > 0:
        return None
    return la


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float, returning *default* for anything non-numeric or non-finite.

    Trainer-side numbers can arrive as numpy/torch scalars or as NaN (a failed oracle call). This
    keeps :meth:`EDARecorder.aggregate` total -- a summary statistic must never raise at the end of
    a multi-hour iteration, and a NaN silently poisoning a logged mean is just as bad.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        coerced = to_jsonable(value)
        if isinstance(coerced, bool) or not isinstance(coerced, (int, float)):
            return default
        out = float(coerced)
    return out if math.isfinite(out) else default


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _pstdev(values: Sequence[float]) -> float:
    """Population standard deviation (ddof=0), matching what ``numpy.std`` reported in Exp3."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  THE RECORDER                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class EDARecorder:
    """In-memory per-iteration buffer of branch records; GRPO flushes it once, PTO appends per depth.

    Args:
        out_path: where :meth:`flush` / :meth:`append_to_disk` write the JSONL -- normally
            ``iteration_N/eda/generations.jsonl``.
        enabled: when False every method is a no-op and nothing is written (the zero-overhead
            off-switch; ``SAVE_EDA_GENERATIONS=False``).
        save_transcripts: when False, each candidate's ``lookahead.tail`` is dropped AT APPEND TIME
            while the scalar look-ahead fields (``realized_turns``, ``ended_early``) are kept. The
            tails dominate file size on a K=5 arm -- five simulated turns per candidate, G or M
            candidates per branch -- so this is the size lever, not a correctness switch. Dropping
            at append time (not at write time) also caps peak memory, which matters because the
            buffer lives for a whole iteration.

    Notes:
        The buffer is a plain list; :meth:`append` does no I/O. Every whole-file write is atomic
        (temp file + ``os.replace``), so a crash mid-write can never leave a half-written JSONL
        that the EDA would silently parse as a short iteration. :meth:`append_to_disk` is the one
        non-atomic write (an O_APPEND of the new rows); its caller (PTO's per-depth flush) records
        the count only AFTER the append returned and normalises the file with :meth:`rewrite` on
        resume, so a torn trailing line is dropped rather than parsed.
    """

    def __init__(self, out_path: str, *, enabled: bool = True, save_transcripts: bool = True):
        self.out_path = out_path
        self.enabled = bool(enabled)
        self.save_transcripts = bool(save_transcripts)
        self.records: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.records)

    # ── capture ──────────────────────────────────────────────────────────────

    def append(self, record: Dict[str, Any]) -> None:
        """Buffer one BRANCH record (prefix once, ``candidates`` nested).

        Cheap and non-blocking -- callable from inside the async reward fn. No-op when disabled.
        When ``save_transcripts=False`` the per-candidate ``lookahead.tail`` is set to None here,
        so the tails never accumulate in memory either.

        Notes:
            The record is stored by reference and mutated in place for the tail drop; do not reuse
            the dict you passed in.
        """
        if not self.enabled:
            return
        if not self.save_transcripts:
            for cand in record.get("candidates") or []:
                la = cand.get("lookahead")
                if isinstance(la, dict) and la.get("tail") is not None:
                    la["tail"] = None
        self.records.append(record)

    def clear(self) -> None:
        """Drop the buffer (start-of-iteration reset). Does not touch anything on disk."""
        self.records = []

    # ── persistence ──────────────────────────────────────────────────────────

    def _write_jsonl(self, path: str) -> str:
        """Write the buffer to *path* atomically (temp file + ``os.replace``).

        Shared by :meth:`flush` and :meth:`snapshot_to`. An empty buffer still writes an empty
        file, so the file's presence is a reliable "this iteration ran" signal rather than an
        ambiguous absence. Every value goes through :func:`to_jsonable` first and the dump runs
        with ``allow_nan=False``, so the output is strict JSON on every line.
        """
        self._replace_with(path, self.records)
        return path

    @staticmethod
    def jsonl_line(record: Dict[str, Any]) -> str:
        """One JSONL line for *record*: strict JSON (``allow_nan=False``), ``to_jsonable`` first.

        THE line format -- :meth:`flush`, :meth:`snapshot_to`, :meth:`append_to_disk` and
        :meth:`rewrite` all go through it, so a file written by any of them reads back through
        :func:`iter_jsonl` identically.
        """
        return json.dumps(to_jsonable(record), ensure_ascii=False, allow_nan=False, default=str) + "\n"

    @staticmethod
    def _replace_with(path: str, rows: Sequence[Dict[str, Any]]) -> None:
        """Atomically make *path* hold exactly *rows* (temp file + ``os.replace``)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for rec in rows:
                f.write(EDARecorder.jsonl_line(rec))
        os.replace(tmp, path)

    def flush(self) -> Optional[str]:
        """Write the buffer to ``out_path``. Returns the path, or None when disabled.

        One write per iteration is deliberate for GRPO: the Colab output dir is a Drive-FUSE mount
        where many small writes are both slow and failure-prone. Idempotent -- calling it twice
        rewrites the same file.
        """
        if not self.enabled:
            return None
        return self._write_jsonl(self.out_path)

    def append_to_disk(self, n_already: int) -> int:
        """APPEND ``records[n_already:]`` to ``out_path``; return the new on-disk count.

        The incremental write PTO uses after every trunk depth (its build runs for ~an hour, and
        a preemption must not lose the rows of the depths that finished). *n_already* is the
        count the caller knows to be on disk; rows before it are never rewritten. An append of
        zero rows still creates the file, so its presence keeps meaning "this build ran".

        Returns ``len(self.records)`` on success, or 0 when disabled. Raises ``OSError`` /
        ``ValueError`` on a failed write -- the caller decides what to count as flushed (see
        ``pto_trainer._flush_eda_rows``, which then calls :meth:`rewrite` with the rows it knows
        reached the disk, so a torn trailing line cannot survive into the next append).
        """
        if not self.enabled:
            return 0
        n_already = max(0, min(int(n_already), len(self.records)))
        parent = os.path.dirname(self.out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.out_path, "a", encoding="utf-8", newline="\n") as f:
            for rec in self.records[n_already:]:
                f.write(self.jsonl_line(rec))
        return len(self.records)

    def rewrite(self, rows: Sequence[Dict[str, Any]]) -> None:
        """Atomically make ``out_path`` hold exactly *rows*. Does NOT touch the buffer.

        The resume-side complement of :meth:`append_to_disk`: after a kill mid-append the file
        may end in a torn line or hold rows the progress snapshot never counted, and this puts
        it back on a clean line boundary before the next append. No-op when disabled.
        """
        if not self.enabled:
            return
        self._replace_with(self.out_path, rows)

    def snapshot_to(self, path: str) -> Optional[str]:
        """Write the current buffer beside an HF checkpoint (crash-recovery copy).

        WHY THIS EXISTS: HF's ``Trainer`` resumes by FAST-FORWARDING through the batches that were
        already consumed, and fast-forwarded batches never re-invoke the reward function. Without a
        snapshot, a resumed iteration's ``generations.jsonl`` would silently contain only the rows
        produced AFTER the resume point -- an iteration that looks complete but is missing its
        first half, with nothing anywhere flagging the loss. Call this from the checkpoint
        ``on_save`` callback with a path inside the checkpoint dir, so the snapshot stays aligned
        with whichever checkpoint :func:`policy.get_latest_valid_hf_checkpoint` walks back to.

        Returns the path written, or None when disabled.
        """
        if not self.enabled:
            return None
        return self._write_jsonl(path)

    def load_from(self, path: str) -> int:
        """REPLACE the buffer with the records in the JSONL snapshot at *path*.

        Inverse of :meth:`snapshot_to`; call it on resume BEFORE training restarts, so the
        end-of-iteration :meth:`flush` writes pre-crash rows followed by post-resume rows.

        Returns the number of records loaded -- 0, as a guarded no-op, when disabled, when *path*
        is empty/missing, or when the file is unreadable or corrupt. A checkpoint written before
        snapshots existed therefore resumes exactly as it always did.

        Notes:
            This REPLACES rather than extends. Loading after appending post-resume rows would
            discard them; the call belongs immediately after the checkpoint is resolved.
            Corrupt trailing lines abort the whole load (returning 0) rather than yielding a
            silently truncated buffer -- losing the snapshot is recoverable, a half-loaded one
            that then gets flushed over the real file is not.
        """
        if not self.enabled or not path or not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                recs = [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            return 0
        self.records = recs
        return len(recs)

    def next_group_branch_id(self) -> int:
        """The first GRPO ``branch_id`` that does not collide with anything already buffered.

        GRPO's ``branch_id`` is a running prompt-group counter that must be unique within the
        iteration -- the EDA keys per-branch aggregation on ``(conversation_id, branch_id)``, so a
        repeat silently pools two unrelated prompt-groups into one. The counter lives in the
        reward closure, which is rebuilt from scratch on a mid-iteration resume while
        :meth:`load_from` restores the pre-crash rows; without seeding it from those rows the
        post-resume groups restart at 0 and collide with them.

        Returns:
            ``max(branch_id over the buffered GRPO rows) + 1``, or 0 when there are none. PTO rows
            are ignored: their ``branch_id`` is trunk DEPTH, not a counter, and a GRPO iteration
            never buffers any.
        """
        best = -1
        for rec in self.records:
            if not _is_grpo_branch(rec):
                continue
            bid = _as_float(rec.get("branch_id"))
            if bid is not None:
                best = max(best, int(bid))
        return best + 1

    # ── aggregates (per-iteration TensorBoard scalars) ───────────────────────

    def aggregate(self) -> Tuple[Dict[str, float], List[float]]:
        """Summarize the buffer into ``(scalars, scores)`` for per-iteration logging.

        ``scalars`` is a flat ``{tag: float}`` a TB writer can splat; ``scores`` is the flat list
        of usable candidate rewards for a histogram -- None and NaN (a failed oracle call) are
        excluded, so the mean is over what the grader actually returned rather than over the
        group-mean stand-in ``core.reward.rewards_for_trl`` hands TRL for such a candidate.

        Keys are emitted ONLY when the recorded rows support them, never as zeros:

        * always -- ``eda/n_branches``, ``eda/n_candidates``
        * when any candidate scored -- ``eda/mean_candidate_reward``, ``eda/reward_std``
        * when any candidate exists -- ``eda/oracle_success_rate``,
          ``eda/ended_by_candidate_frac`` (completions that closed the session themselves)
        * when look-ahead ran (K>0) -- ``eda/lookahead_realized_turns_mean``,
          ``eda/lookahead_ended_early_frac``, ``eda/lookahead_not_graded_frac`` (rollouts the
          simulator froze -- ``stop_reason`` in :data:`NOT_GRADED_STOP_REASONS`, left ungraded)
        * PTO rows only -- ``pto/branch_points``, ``pto/pref_pair_count``, ``pto/tau_filter_rate``
        * GRPO rows only -- ``grpo/num_groups``, and when group stats are present
          ``grpo/group_reward_std_mean``, ``grpo/frac_zero_std``
        * when TRL's ``evaluate()`` also scored groups -- ``eda/eval_n_branches``,
          ``eda/eval_n_candidates`` and, when they scored, ``eda/eval_mean_candidate_reward``,
          ``eda/eval_reward_std``, ``eda/eval_oracle_success_rate``

        A GRPO run emits no ``pto/*`` key at all (and vice versa): a zero would be indistinguishable
        from "the tau filter rejected everything", which is a real and alarming state.

        Notes:
            **Every non-``eval_*`` key, and the returned ``scores``, cover the GRADIENT-BEARING
            rows only** -- rows whose ``eval_pass`` is False. With an eval split, TRL calls the
            reward function during ``evaluate()`` too (once per eval prompt per epoch), and those
            groups never produced a gradient; pooling them would report a blend of on-policy and
            held-out candidates under names every reader takes to mean "what the optimizer saw".
            The eval half is kept, under the ``eda/eval_*`` prefix.

            ``eda/reward_std`` is the population SD (ddof=0) pooled across ALL training candidates
            in the iteration -- it is NOT the mean within-group SD that drives GRPO's advantages;
            that is ``grpo/group_reward_std_mean``. ``grpo/frac_zero_std`` is the fraction of groups
            whose eight siblings all scored identically, i.e. groups that contributed no gradient
            signal.
        """
        branches = [b for b in self.records if not b.get("eval_pass")]
        eval_branches = [b for b in self.records if b.get("eval_pass")]
        cands = [c for b in branches for c in (b.get("candidates") or [])]
        n_cands = len(cands)

        scores = [
            v for v in (_as_float(c.get("score")) for c in cands) if v is not None
        ]
        scalars: Dict[str, float] = {
            "eda/n_branches": float(len(branches)),
            "eda/n_candidates": float(n_cands),
        }
        if scores:
            scalars["eda/mean_candidate_reward"] = _mean(scores)
            scalars["eda/reward_std"] = _pstdev(scores)
        if n_cands:
            scalars["eda/oracle_success_rate"] = _mean(
                [1.0 if _candidate_success(c) else 0.0 for c in cands]
            )
            scalars["eda/ended_by_candidate_frac"] = _mean(
                [1.0 if c.get("ended_by_candidate") else 0.0 for c in cands]
            )

        las = [la for la in (_lookahead_of(c) for c in cands) if la is not None]
        if las:
            scalars["eda/lookahead_realized_turns_mean"] = _mean(
                [(_as_float(la.get("realized_turns"), 0.0) or 0.0) for la in las]
            )
            scalars["eda/lookahead_ended_early_frac"] = _mean(
                [1.0 if la.get("ended_early") else 0.0 for la in las]
            )
            # Simulator failures as a rendered scalar rather than a log line: a rising curve is
            # a saturating patient server, hours before the min_success_ratio gate fires.
            scalars["eda/lookahead_not_graded_frac"] = _mean(
                [1.0 if la.get("stop_reason") in NOT_GRADED_STOP_REASONS else 0.0 for la in las]
            )

        # PTO: one row per branch point; a pair exists iff tau passed there.
        pto_branches = [b for b in branches if _is_pto_branch(b)]
        if pto_branches:
            n_bp = len(pto_branches)
            n_pairs = sum(1 for b in pto_branches if _pair_emitted(b))
            scalars["pto/branch_points"] = float(n_bp)
            scalars["pto/pref_pair_count"] = float(n_pairs)
            scalars["pto/tau_filter_rate"] = float(1.0 - n_pairs / n_bp)

        # GRPO: one row per prompt-group per epoch; group stats live on the row.
        grpo_branches = [b for b in branches if _is_grpo_branch(b)]
        if grpo_branches:
            scalars["grpo/num_groups"] = float(len(grpo_branches))
            stds = [
                v for v in (_as_float(b.get("group_std")) for b in grpo_branches)
                if v is not None
            ]
            if stds:
                scalars["grpo/group_reward_std_mean"] = _mean(stds)
                scalars["grpo/frac_zero_std"] = _mean([1.0 if v == 0.0 else 0.0 for v in stds])

        # The held-out half, reported separately rather than pooled or dropped. Emitted only when
        # evaluate() actually ran, so an arm with no eval split carries no dead key.
        if eval_branches:
            eval_cands = [c for b in eval_branches for c in (b.get("candidates") or [])]
            scalars["eda/eval_n_branches"] = float(len(eval_branches))
            scalars["eda/eval_n_candidates"] = float(len(eval_cands))
            eval_scores = [
                v for v in (_as_float(c.get("score")) for c in eval_cands) if v is not None
            ]
            if eval_scores:
                scalars["eda/eval_mean_candidate_reward"] = _mean(eval_scores)
                scalars["eda/eval_reward_std"] = _pstdev(eval_scores)
            if eval_cands:
                scalars["eda/eval_oracle_success_rate"] = _mean(
                    [1.0 if _candidate_success(c) else 0.0 for c in eval_cands]
                )

        return scalars, scores

    def sample_for_display(self, n: int) -> List[Dict[str, Any]]:
        """Pick up to *n* candidates spread evenly across the score range (worst -> best).

        Flattens the nested candidates, re-attaching each one's branch context, shaped for a TB
        text/table logger. Spread, not top-n, on purpose: a page of eight near-identical winners
        says nothing about what the policy is doing, while worst/median/best side by side shows
        whether the oracle is discriminating at all -- the exact failure the oracle-sanity gate
        exists to catch.

        Returns an empty list when *n* <= 0. Unscored candidates are used only as a fallback when
        nothing scored (an all-failed batch is itself worth eyeballing).
        """
        if n <= 0:
            return []
        flat: List[Dict[str, Any]] = []
        for b in self.records:
            for c in b.get("candidates") or []:
                flat.append({
                    "prompt": b.get("prefix"),
                    "completion": c.get("completion"),
                    "score": c.get("score"),
                    "sub_scores": c.get("sub_scores"),
                    "lookahead": c.get("lookahead"),
                    "persona_id": b.get("persona_id"),
                    "conversation_id": b.get("conversation_id"),
                    "branch_id": b.get("branch_id"),
                    "pto": ({"role": c.get("role")} if c.get("role") is not None else None),
                    "grpo": (
                        {"group_mean": b.get("group_mean"), "group_std": b.get("group_std")}
                        if b.get("group_mean") is not None else None
                    ),
                })
        scored = [r for r in flat if r.get("score") is not None]
        if not scored:
            return flat[:n]
        scored.sort(key=lambda r: r["score"])
        if n >= len(scored):
            return scored
        if n == 1:
            return [scored[len(scored) // 2]]
        idxs = sorted({int(round(i * (len(scored) - 1) / (n - 1))) for i in range(n)})
        return [scored[i] for i in idxs]


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield branch records from a ``generations.jsonl``, skipping blank/corrupt lines.

    The read side used by the EDA. Tolerant by design: a run killed between the temp write and
    ``os.replace`` cannot produce a partial file, but a Drive sync can still surface one mid-copy,
    and a whole iteration should not be unreadable because of a single bad line.
    """
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue
