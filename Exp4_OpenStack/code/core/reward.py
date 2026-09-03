"""reward.py -- the one place look-ahead and the oracle are composed, for BOTH methods.

Scoring a candidate therapist turn is three steps that must always happen in the same order:
clean the completion, optionally simulate ``K`` more turns, then ask the oracle to grade the
resulting transcript. GRPO consumes the result as a reward (TRL calls the closure built by
:func:`make_reward_fn` once per optimizer step) and PTO consumes it as a ranking signal (the
trunk grower calls :func:`score_pref_candidates` at every branch point). In Exp3 those were two
separate implementations -- ``_shared/reward.py`` for GRPO and ``_score_completions_batch_detailed``
in ``pto_trainer.py`` for PTO -- which is a slow-motion hazard: the K knob is the whole point of
the experiment, and any drift between the two copies (a different seed string, a different
realized-turn count, a different degenerate-completion rule) silently makes "K=5 helps PTO but not
GRPO" a statement about the two code paths rather than about look-ahead. Here both methods enter
the same private :func:`_score_candidates`, so the K in {0, 5} contrast means the same thing on
both sides by construction.

Three rules in this module are load-bearing rather than defensive:

**A completion that cleans to empty is floored, never graded.** The therapist is a base model
with a hand-written ChatML template, so early in training it self-plays: it emits ``<|im_start|>``
and writes the patient's next line. After ``clean_completion`` that turn is the empty string. If
the oracle were handed ``transcript + "[THERAPIST]: "`` it would grade the surviving *real* turns
and return a perfectly ordinary score -- rewarding the policy for producing nothing. Every such
candidate is assigned :data:`~core.oracle.REWARD_FLOOR` (0.0, below the oracle's ~1-5 range) and
is never sent to the look-ahead simulator or the oracle at all, which is also why a degenerate
batch costs no GPU time and no tokens.

**A batch whose oracle success rate falls below ``min_success_ratio`` raises.** Failed calls are
not missing at random: an over-long transcript, a schema the server started rejecting, a server
that died mid-iteration -- each drops a *biased* subset of candidates, and in GRPO's
group-relative advantage a systematically-missing sibling shifts both the mean and the std of its
group of 8. Training on that is worse than stopping, so this raises ``RuntimeError`` and lets the
resume machinery restart the iteration from its last checkpoint.

**The seed string has exactly one definition.** ``core.lookahead.seed_transcript`` builds
``f"{transcript}\\n\\n[THERAPIST]: {completion}"``, and that same string is what the oracle reads
at K=0, what the rollout extends at K>0, and what the EDA slices tails off. This module calls it
rather than re-deriving it: a second copy of the joiner here is how the K=0 and K=5 arms would end
up grading subtly different objects, and nothing downstream would say so.

Two more rules keep the training reward the same object as the eval measurement:

**A candidate that closes the session is graded the way the eval path would save it.** The
conversation loop strips ``SESSION ENDED`` from a closing turn and discards the explanation the
model wrote after it; the look-ahead does the same for every SIMULATED turn. The candidate itself
must get the same treatment, or the oracle grades a therapist line containing the literal keyword
plus text no saved conversation would carry, and at K>0 the patient is then asked to continue a
session the therapist just ended. So the candidate is split with the same helper
(``core.lookahead.split_session_end``), the text before the keyword is what the oracle reads, and
a closing candidate is graded on its seed alone -- no rollout, and at K>0 it is recorded as a
complete rollout of zero turns (``ended_early=True``, ``stop_reason="session_ended"``). A candidate
that is ONLY the keyword has no utterance and is floored like any other degenerate turn. The
candidate's :attr:`~CandidateScore.completion` keeps the keyword for two readers only: PTO's trunk
advance, which reads it to freeze an ended trunk, and the EDA row, where ``ended_by_candidate``
says the graded text stops before it. Neither optimizer trains on that string as-is. A DPO pair
uses the SPLIT text (``pto_trainer._pair_text``) -- exactly what the oracle graded, so the pair's
score describes the text it trains on; GRPO trains on the completion ids TRL sampled (the policy's
own emission, keyword included) while its reward is computed on the split text, since the reward
callable never rewrites TRL's completions.

**A look-ahead the simulator failed to run is not graded.** ``simulate_lookahead_batch`` freezes a
sim on a patient-call failure, a non-OOM generate error or an unparseable seed, and keeps the
transcript it reached. Grading that transcript would score one sibling at K=0 (or K=1, K=2, ...)
beside seven scored at K=5 -- a within-group K asymmetry that ``scale_rewards="group"`` amplifies
into the advantages of all eight, and under server saturation the freezes correlate across a
whole optimizer step while nothing raises, because the oracle still succeeds on the short text.
Such a candidate gets ``score=None`` with a machine-readable ``not_graded_reason`` (its
``stop_reason``), GRPO substitutes its group mean exactly as for an oracle failure, PTO excludes it
from the tau comparison, and it counts as a FAILURE against ``min_success_ratio`` -- so a
saturated patient server stops the run instead of quietly training the K=5 arm on K=0 rewards.
A session that closes during the rollout, or a simulated turn that cleans to nothing, is a
complete rollout and stays graded.

TRL interface facts this module depends on (get them wrong and a run mistrains silently):

* completions come back as **G-consecutive blocks per prompt** (``RepeatSampler`` with
  ``mini_repeat_count=num_generations``), so groups are recovered by reshaping to ``(-1, G)``;
* the extra dataset columns (``transcript``, ``conversation_id``, ``persona_id``,
  ``patient_system_prompt``) arrive as ``**kwargs`` lists parallel to ``completions``, and ONLY
  because the trainer sets ``remove_unused_columns=False``. Without that flag TRL drops every
  column it does not recognise and the reward fn raises on a missing ``transcript``;
* the closure is awaited on TRL's own event loop, which is not the notebook's -- which is why
  every semaphore comes from :class:`~core.concurrency.AsyncPrimitives` (loop-keyed) instead of
  being cached in this module.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.concurrency import AsyncPrimitives
from core.lookahead import (
    NOT_GRADED_STOP_REASONS,
    LookaheadConfig,
    LookaheadResult,
    LookaheadState,
    seed_transcript,
    simulate_lookahead_batch,
    split_session_end,
)
from core.oracle import (
    REWARD_FLOOR,
    OracleConfig,
    batch_success_ratio,
    score_conversations_batch,
)
from core.policy import clean_completion
from core.recorder import PHASE_GROUP, build_branch_record, build_candidate

__all__ = [
    "ORACLE_FAILED_REASON",
    "CandidateScore",
    "make_reward_fn",
    "rewards_for_trl",
    "score_pref_candidates",
]

#: ``CandidateScore.not_graded_reason`` for a candidate the oracle was asked about and could not
#: grade (every retry failed on at least one rubric). The other values are the look-ahead's
#: :data:`~core.lookahead.NOT_GRADED_STOP_REASONS`, for candidates the oracle never saw. Together
#: they let the EDA separate grader failures from simulator failures on a ``score == null`` row.
ORACLE_FAILED_REASON = "oracle_failed"


# The text the oracle grades for a candidate at K=0 comes from
# ``core.lookahead.seed_transcript`` and is NOT redefined here. That function is also what the
# rollout extends at K>0 and what its exact tail slice depends on, so a second copy of the
# ``"\n\n[THERAPIST]: "`` joiner in this module would be a place for the two to drift apart --
# which is the one failure that would silently make the K=0 and K=5 arms score different objects.

# Printed indent for the batch lines, matching the trainers' per-phase logging depth.
_LOG = "    "


# =============================================================================
# PER-CANDIDATE RESULT
# =============================================================================


@dataclass(frozen=True)
class CandidateScore:
    """Everything known about one scored candidate completion.

    GRPO needs only :attr:`score`; PTO needs the rest -- the per-questionnaire breakdown for the
    ``pairs.csv`` audit trail, and the look-ahead tail so the EDA can see what the oracle actually
    read when it preferred one branch over another.

    Attributes:
        completion: the CLEANED completion (``""`` marks a degenerate turn). ⚠ When
            :attr:`ended_by_candidate` is set this still CONTAINS the ``SESSION ENDED`` keyword
            and whatever the model wrote after it -- kept for PTO's trunk advance, which reads
            the keyword to freeze an ended trunk, and for the EDA row. It is NOT what a DPO pair
            trains on: the pair uses the split text (``pto_trainer._pair_text``), exactly what
            the oracle graded. GRPO trains on the completion ids TRL sampled and only its reward
            is computed on the split text. The oracle graded only the text BEFORE the keyword;
            see :attr:`scored_text`.
        score: the RAW result -- the oracle's unweighted mean across questionnaires, or
            :data:`~core.oracle.REWARD_FLOOR` when :attr:`degenerate`, or ``None`` when the
            candidate was NOT graded (:attr:`not_graded_reason` says why). ``None`` means "not
            graded", never "graded badly", and PTO must not rank on a fabricated number -- it
            excludes such a candidate from the tau comparison.
            ⚠ GRPO cannot pass ``None`` to TRL: the pinned trl 1.4.0 turns it into NaN and then
            ``nansum``s it to **0.0**, i.e. trains on it as the worst possible completion. The
            reward vector handed to TRL is therefore built by :func:`rewards_for_trl`, which
            substitutes the group mean; :attr:`score` itself stays raw so the EDA can still tell
            a failure from a floor.
        sub_scores: ``{questionnaire id (str): mean}``, or ``None``. Kept RAW-oracle, so a floored
            row is identifiable as ``score == REWARD_FLOOR`` with ``sub_scores is None``.
        success: did the oracle return a usable score for every requested questionnaire.
            **False for a degenerate candidate and for an ungraded look-ahead failure too** --
            no call was made, so there is no usable score. That means the recorder's
            ``eda/oracle_success_rate`` mixes grader failures with policy degeneracy and
            simulator failures; the grader-health metric is the one logged as
            ``oracle/success_rate``, whose denominator is the candidates actually sent to the
            grader, and the GATE quantity is ``success_rate`` in the telemetry (graded over
            gradable).
        attempts: total oracle calls made for this candidate (retries included); 0 when degenerate
            or when the look-ahead failed before the oracle was asked.
        degenerate: the completion cleaned to ``""`` (or to nothing but the ``SESSION ENDED``
            keyword); no oracle call and no rollout were made.
        scored_text: the exact string the oracle graded (or would have, when degenerate); for an
            ungraded look-ahead failure, the transcript the sim had reached when it froze.
        lookahead: ``{"k", "tail", "realized_turns", "ended_early", "stop_reason"}`` when a
            rollout ran OR the candidate itself closed the session at K>0 (a complete rollout of
            zero turns: ``realized_turns=0``, ``ended_early=True``,
            ``stop_reason="session_ended"``), else ``None``. ``None`` at K=0 AND for degenerate
            candidates at K>0 -- no simulation happened there, and the recorder's look-ahead
            scalars are averages over candidates that had a future to score.
        not_graded_reason: ``None`` when :attr:`score` is a number. Otherwise
            :data:`ORACLE_FAILED_REASON` (the grader failed) or the look-ahead ``stop_reason``
            from :data:`~core.lookahead.NOT_GRADED_STOP_REASONS` (the simulator failed and the
            oracle was never asked). Written to the EDA candidate so the two failure classes
            can be separated without inspecting ``oracle.attempts``.
        ended_by_candidate: the candidate contained ``SESSION ENDED``; it was graded on its seed
            alone (no rollout) and, when it also carried an utterance, that utterance is what
            the oracle read.
    """

    completion: str
    score: Optional[float]
    sub_scores: Optional[Dict[str, float]]
    success: bool
    attempts: int
    degenerate: bool
    scored_text: str
    lookahead: Optional[Dict[str, Any]]
    not_graded_reason: Optional[str] = None
    ended_by_candidate: bool = False

    def to_record(
        self,
        idx: int,
        *,
        role: Optional[str] = None,
        reward_used: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Render this candidate as a nested :mod:`core.recorder` candidate dict.

        Args:
            idx: position within the branch's candidate list (0..G-1 or 0..M-1).
            role: PTO only -- ``"chosen"`` / ``"rejected"`` / ``"neither"``. Set ``"rejected"``
                only when the tau filter actually passed, because that is what the recorder
                counts as "a preference pair was emitted here".
            reward_used: GRPO only -- the number actually handed to TRL, when it differs from
                :attr:`score` (i.e. this candidate was not graded and :func:`rewards_for_trl`
                substituted its group's mean). Leave ``None`` when the two agree; the key is
                then absent and the EDA falls back to ``score``.

        Notes:
            Both methods build their EDA rows through this, so a GRPO candidate and a PTO
            candidate are the same shape and the EDA needs no per-method branch.

            Two keys ride on top of ``core.recorder.build_candidate``'s schema (the recorder
            writes candidates as given): ``ended_by_candidate`` on EVERY candidate (a bool that
            is always present, like ``eval_pass`` on the row, so nobody has to know that absence
            means False), and ``not_graded_reason`` ONLY when :attr:`score` is ``None``.
        """
        cand = build_candidate(
            idx,
            self.completion,
            score=self.score,
            sub_scores=self.sub_scores,
            role=role,
            lookahead=self.lookahead,
            oracle_success=self.success,
            oracle_attempts=self.attempts,
            reward_used=reward_used,
        )
        cand["ended_by_candidate"] = bool(self.ended_by_candidate)
        if self.score is None and self.not_graded_reason:
            cand["not_graded_reason"] = str(self.not_graded_reason)
        return cand


# =============================================================================
# THE SHARED SCORING PATH (both methods enter here)
# =============================================================================


def _at(seq: Optional[Sequence[Any]], i: int, default: Any = "") -> Any:
    """``seq[i]`` with a default for a short/absent list -- TRL kwargs are not guaranteed."""
    if seq is None:
        return default
    try:
        return seq[i]
    except (IndexError, KeyError, TypeError):
        return default


def _coerce_int(value: Any) -> Any:
    """Best-effort ``int()`` for ids arriving as numpy scalars or strings; pass through on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


async def _score_candidates(
    model,
    tokenizer,
    client,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    *,
    transcripts: Sequence[str],
    completions: Sequence[str],
    sp_therapist: Optional[str],
    sp_patient_list: Optional[Sequence[str]],
    label: str,
    enforce_success_ratio: bool,
    state: Optional[LookaheadState],
) -> Tuple[List[CandidateScore], Dict[str, float]]:
    """Clean -> (look-ahead) -> oracle, for a flat parallel batch of candidates.

    This is the single implementation behind :func:`make_reward_fn` and
    :func:`score_pref_candidates`. Both pass the same shapes: ``transcripts[i]`` is the
    conversation prefix candidate ``i`` continues (PTO repeats one trunk prefix M times),
    ``completions[i]`` is the raw sampled turn, ``sp_patient_list[i]`` is the persona system
    prompt the simulated patient must answer with.

    Returns:
        ``(candidates, telemetry)``. ``telemetry`` carries, all as floats:

        * ``n``, ``n_degenerate``, ``n_ended_by_candidate`` (closed the session; seed graded);
        * ``n_rollout`` (sims handed to the look-ahead), ``n_lookahead_failed`` (of those, frozen
          by a simulator failure and NOT graded) and its breakdown ``n_la_patient_error`` /
          ``n_la_gpu_error`` / ``n_la_prompt_overflow`` / ``n_la_parse_error``;
        * ``n_oracle`` (candidates actually sent to the grader), ``n_success``,
          ``oracle_success_rate`` (= ``n_success / n_oracle``; grader health) and
          ``success_rate`` (= ``n_success / (n_oracle + n_lookahead_failed)``; **the gate**:
          graded over gradable, so a saturated patient server fails it too);
        * ``lookahead_seconds``, ``realized_turns_mean``, ``n_ended_early`` -- over GRADED
          candidates that had a future to score (a session-closing candidate at K>0 contributes
          zero realized turns and one early end).

    Raises:
        ValueError: ``transcripts`` and ``completions`` differ in length, or look-ahead / the
            oracle returned a result list of the wrong length. A length mismatch would pair a
            score with the wrong candidate, which is unrecoverable and invisible downstream, so
            it is fatal rather than warned about.
        RuntimeError: ``success_rate`` is below ``oracle_cfg.min_success_ratio`` and
            *enforce_success_ratio* is set -- whether the failures came from the grader or from
            the look-ahead simulator.

    Notes:
        **Degenerate candidates are excluded from both the rollout and the oracle batch**, not
        scored and then overridden. That keeps the success-rate gate a measure of grader/server
        health rather than of policy degeneracy (an all-degenerate batch makes zero oracle calls
        and cannot trip the gate), and it is free -- the reward is the floor either way.

        **Session-closing candidates are split before anything else** (see the module docstring):
        the oracle reads the text before the keyword, no rollout is run, and a keyword-only
        completion is degenerate.

        **Look-ahead failures are excluded from the oracle batch and reported as ungraded**,
        never scored on the short transcript they reached. They DO count against the gate.

        *state* must be ONE :class:`~core.lookahead.LookaheadState` per iteration. It carries the
        sub-batch that OOM halving arrived at, and that halving is only sticky across optimizer
        steps because the same object comes back on the next call.
    """
    n = len(completions)
    if len(transcripts) != n:
        raise ValueError(
            f"transcripts/completions length mismatch ({len(transcripts)} vs {n}); "
            f"they must be parallel lists over candidates"
        )

    k = int(la_cfg.k)

    # -- 0. Clean, then split a closing candidate exactly as the eval path would save it ------
    # `cleaned` is the candidate as the policy emitted it (ChatML leak cut off) -- what PTO
    # advances the trunk with and what the EDA row records. `utterance` is the text the oracle
    # reads: the same string unless the candidate contains SESSION ENDED, in which case it is
    # the text before the keyword (the explanation after it is dropped, as in a saved
    # conversation) -- and that split text is also what a DPO pair trains on
    # (pto_trainer._pair_text). A keyword-only candidate has no utterance and is degenerate.
    cleaned: List[str] = []
    utterance: List[str] = []
    ended_by_candidate: List[bool] = []
    for raw in completions:
        c = clean_completion(raw)
        content, ended = split_session_end(c) if c else ("", False)
        cleaned.append(c)
        utterance.append(content)
        ended_by_candidate.append(ended)
    degenerate: List[bool] = [not u for u in utterance]
    active: List[int] = [i for i in range(n) if not degenerate[i]]
    # A candidate that closed the session has no future to simulate: seed graded, no rollout.
    rollout: List[int] = [i for i in active if not ended_by_candidate[i]]

    telemetry: Dict[str, float] = {
        "n": float(n),
        "n_degenerate": float(n - len(active)),
        "n_ended_by_candidate": float(sum(1 for i in range(n) if ended_by_candidate[i])),
        "n_rollout": 0.0,
        "n_lookahead_failed": 0.0,
        "n_la_patient_error": 0.0,
        "n_la_gpu_error": 0.0,
        "n_la_prompt_overflow": 0.0,
        "n_la_parse_error": 0.0,
        "n_oracle": 0.0,
        "n_success": 0.0,
        "oracle_success_rate": 1.0,
        "success_rate": 1.0,
        "lookahead_seconds": 0.0,
        "realized_turns_mean": 0.0,
        "n_ended_early": 0.0,
    }

    # -- 1. Build the text the oracle will read, per active candidate ---------
    scored_text: Dict[int, str] = {}
    la_records: Dict[int, Dict[str, Any]] = {}
    not_graded: Dict[int, str] = {}          # index -> look-ahead stop_reason

    if k > 0 and rollout:
        # simulate_lookahead_batch already warns when every patient system prompt is empty and
        # raises on a length mismatch of its own three inputs; not duplicated here.
        telemetry["n_rollout"] = float(len(rollout))
        la_start = time.time()
        results = await simulate_lookahead_batch(
            model,
            tokenizer,
            client,
            la_cfg,
            primitives,
            [transcripts[i] for i in rollout],
            [utterance[i] for i in rollout],
            sp_therapist or "",
            [str(_at(sp_patient_list, i, "")) for i in rollout],
            state=state,
        )
        telemetry["lookahead_seconds"] = time.time() - la_start
        if len(results) != len(rollout):
            raise ValueError(
                f"simulate_lookahead_batch returned {len(results)} results for "
                f"{len(rollout)} sims; refusing to pair scores with the wrong candidates"
            )

        for slot, i in enumerate(rollout):
            res = results[slot]
            # LookaheadResult owns the recorder dict's shape; building it here by hand is how the
            # two would drift.
            la_records[i] = res.to_record()
            # The transcript the sim reached is kept for the EDA either way; whether the oracle
            # sees it is decided by stop_reason (module docstring: a simulator failure is a
            # K=0 snapshot masquerading as a K=5 rollout, and is NOT graded).
            scored_text[i] = res.extended_transcript
            if not res.graded:
                not_graded[i] = res.stop_reason
                telemetry["n_lookahead_failed"] += 1.0
                telemetry[f"n_la_{res.stop_reason}"] = telemetry.get(f"n_la_{res.stop_reason}", 0.0) + 1.0

    if k > 0:
        for i in active:
            if ended_by_candidate[i]:
                # The candidate closed the session itself: a COMPLETE rollout of zero turns, in
                # the same record shape as a session that closes during the rollout.
                scored_text[i] = seed_transcript(transcripts[i], utterance[i])
                la_records[i] = LookaheadResult(
                    extended_transcript=scored_text[i],
                    tail="",
                    realized_turns=0,
                    ended_early=True,
                    stop_reason="session_ended",
                    k=k,
                ).to_record()
    else:
        for i in active:
            scored_text[i] = seed_transcript(transcripts[i], utterance[i])

    # Look-ahead science scalars, over GRADED candidates that had a future to score.
    realized: List[float] = []
    for i in active:
        rec = la_records.get(i)
        if rec is None or i in not_graded:
            continue
        realized.append(float(rec["realized_turns"]))
        if rec["ended_early"]:
            telemetry["n_ended_early"] += 1.0
    telemetry["realized_turns_mean"] = statistics.fmean(realized) if realized else 0.0

    # -- 2. Oracle, over the gradable candidates only --------------------------
    to_grade: List[int] = [i for i in active if i not in not_graded]
    telemetry["n_oracle"] = float(len(to_grade))
    details: List[Dict[str, Any]] = []
    if to_grade:
        details = list(
            await score_conversations_batch(
                client, oracle_cfg, primitives, [scored_text[i] for i in to_grade]
            )
        )
        if len(details) != len(to_grade):
            raise ValueError(
                f"score_conversations_batch returned {len(details)} results for "
                f"{len(to_grade)} conversations; refusing to pair scores with the wrong candidates"
            )

    n_success = sum(1 for d in details if d.get("success"))
    n_la_failed = len(not_graded)
    telemetry["n_success"] = float(n_success)
    # batch_success_ratio owns the empty-batch convention (1.0, not 0.0): an all-degenerate batch
    # makes no oracle call, carries no evidence about the grader, and must not kill the run.
    telemetry["oracle_success_rate"] = batch_success_ratio(details)
    # The GATE: graded over gradable. A look-ahead the simulator failed to run is a failure of
    # the reward pipeline exactly as an oracle failure is -- both leave a non-random hole.
    n_gradable = len(to_grade) + n_la_failed
    telemetry["success_rate"] = (n_success / n_gradable) if n_gradable else 1.0

    line = (
        f"{_LOG}{label}: {n} candidates "
        f"({int(telemetry['n_degenerate'])} degenerate -> floor"
    )
    if telemetry["n_ended_by_candidate"]:
        line += f", {int(telemetry['n_ended_by_candidate'])} closed the session -> seed graded"
    line += "), "
    if active:
        if to_grade:
            line += (
                f"oracle {n_success}/{len(to_grade)} ok "
                f"({telemetry['oracle_success_rate']:.0%})"
            )
        else:
            line += "oracle not called"
        if k > 0:
            line += (
                f", look-ahead {telemetry['lookahead_seconds']:.1f}s "
                f"(avg {telemetry['realized_turns_mean']:.1f}/{k} turns realized, "
                f"{int(telemetry['n_ended_early'])} ended early"
            )
            if n_la_failed:
                breakdown = ", ".join(
                    f"{reason}={int(telemetry.get(f'n_la_{reason}', 0.0))}"
                    for reason in sorted(NOT_GRADED_STOP_REASONS)
                    if telemetry.get(f"n_la_{reason}", 0.0)
                )
                line += f", {n_la_failed} NOT graded: {breakdown}"
            line += ")"
    else:
        line += "no oracle calls (every completion was degenerate)"
    print(line)

    # -- 3. The gate. Missingness here is not random -- see the module docstring.
    if enforce_success_ratio and active and telemetry["success_rate"] < oracle_cfg.min_success_ratio:
        raise RuntimeError(
            f"Reward success rate {telemetry['success_rate']:.1%} is below "
            f"min_success_ratio={oracle_cfg.min_success_ratio:.0%} "
            f"({n_gradable - n_success}/{n_gradable} gradable candidates were not graded: "
            f"{len(to_grade) - n_success} oracle failures, {n_la_failed} look-ahead simulator "
            f"failures). Aborting rather than training on a biased subset -- failures are not "
            f"missing at random, and in GRPO a systematically-absent sibling shifts both the mean "
            f"and the std of its group. Likely causes: the vLLM server died, is out of memory or "
            f"is saturated (check its log and tools/vllm_serve.ensure_alive), the grader started "
            f"rejecting the json_schema response_format, transcripts have outgrown "
            f"--max-model-len, the look-ahead patient calls are timing out, or an API-backed role "
            f"hit a spend cap / rate limit. The iteration resumes from its last checkpoint once "
            f"the cause is fixed."
        )

    # -- 4. Assemble ----------------------------------------------------------
    by_index = {i: details[slot] for slot, i in enumerate(to_grade)}
    out: List[CandidateScore] = []
    for i in range(n):
        if degenerate[i]:
            out.append(
                CandidateScore(
                    completion="",
                    score=REWARD_FLOOR,
                    sub_scores=None,
                    success=False,
                    attempts=0,
                    degenerate=True,
                    scored_text=seed_transcript(transcripts[i], ""),
                    lookahead=None,
                    ended_by_candidate=ended_by_candidate[i],
                )
            )
            continue
        if i in not_graded:
            out.append(
                CandidateScore(
                    completion=cleaned[i],
                    score=None,
                    sub_scores=None,
                    success=False,
                    attempts=0,
                    degenerate=False,
                    scored_text=scored_text[i],
                    lookahead=la_records.get(i),
                    not_graded_reason=not_graded[i],
                    ended_by_candidate=ended_by_candidate[i],
                )
            )
            continue
        d = by_index[i]
        score = d.get("score")
        out.append(
            CandidateScore(
                completion=cleaned[i],
                score=(None if score is None else float(score)),
                sub_scores=d.get("sub_scores") or None,
                success=bool(d.get("success")),
                attempts=int(d.get("attempts") or 0),
                degenerate=False,
                scored_text=scored_text[i],
                lookahead=la_records.get(i),
                not_graded_reason=(ORACLE_FAILED_REASON if score is None else None),
                ended_by_candidate=ended_by_candidate[i],
            )
        )
    return out, telemetry


# =============================================================================
# GRPO: THE TRL-FACING CALLABLE
# =============================================================================


def rewards_for_trl(
    candidates: Sequence[CandidateScore],
    num_generations: int,
) -> List[Optional[float]]:
    """The reward vector to hand TRL: a failed candidate takes its group's mean.

    ``None`` cannot be forwarded to the pinned ``trl==1.4.0``. Its ``GRPOTrainer`` maps ``None``
    to ``torch.nan`` (``grpo_trainer.py:1259``) and then reduces the per-function rewards with
    ``nansum`` (``:2145``) -- and ``nansum`` of an all-NaN row is **0.0**, not NaN. A candidate
    nobody graded therefore enters the group as the worst possible completion: with G=8 and real
    scores around 3.3, one failed sibling drops the group mean to 2.89 and inflates the group SD
    from ~0.20 to ~1.18, which is enough to flip the sign of a genuinely-worst sibling's
    advantage. The failure is silent -- ``min_success_ratio`` is a batch-level floor and one
    hiccup in 128 never trips it.

    Substituting the group mean is the minimal-perturbation repair: the failed candidate gets
    advantage ~0 (it is trained neither toward nor away from), the group mean is unchanged, and
    the group SD moves only by the ddof shrinkage of one duplicated point.

    "Failed" here is any ``score is None`` -- an oracle failure and a look-ahead the simulator
    could not run (:attr:`CandidateScore.not_graded_reason` tells them apart) are repaired the
    same way, because both are holes the policy did not earn.

    Args:
        candidates: the scored candidates, in TRL's order (G-consecutive blocks per prompt).
        num_generations: TRL's ``G``.

    Returns:
        One reward per candidate, in order. ``None`` survives only where an ENTIRE group failed
        (there is no sibling mean to borrow); TRL then floors all G to 0.0, which is a constant
        group and so contributes no gradient, and it logs its own all-None warning. The list is
        also returned unchanged when the batch is not divisible by ``G`` -- the same refusal to
        guess at groups that :func:`_record_grpo_groups` makes.
    """
    out: List[Optional[float]] = [c.score for c in candidates]
    G = max(1, int(num_generations))
    n = len(out)
    if n == 0 or n % G != 0:
        if n:
            print(
                f"{_LOG}WARNING: {n} rewards not divisible by G={G}; cannot recover groups, so a "
                f"failed candidate cannot borrow its siblings' mean and TRL will floor it to 0.0"
            )
        return out

    n_sub = 0
    for start in range(0, n, G):
        block = out[start:start + G]
        good = [v for v in block if v is not None]
        if not good or len(good) == G:
            continue
        fill = statistics.fmean(good)
        for j in range(G):
            if block[j] is None:
                out[start + j] = fill
                n_sub += 1
    if n_sub:
        print(
            f"{_LOG}WARNING: {n_sub}/{n} candidates had no oracle score; each took its group's "
            f"mean so it carries ~zero advantage (trl 1.4.0 would otherwise reward it as 0.0). "
            f"The substitution is recorded per candidate as `reward_used` in generations.jsonl"
        )
    return out


def _record_grpo_groups(
    recorder,
    *,
    iteration: int,
    num_generations: int,
    group_base: Dict[str, int],
    epoch: Optional[float],
    was_training: bool,
    transcripts: Sequence[str],
    candidates: Sequence[CandidateScore],
    rewards: Sequence[Optional[float]],
    conversation_ids: Optional[Sequence[Any]],
    persona_ids: Optional[Sequence[Any]],
) -> None:
    """Buffer one EDA branch row per prompt-group, all G candidates nested.

    TRL hands the reward fn G-consecutive completions per prompt, so group ``g`` occupies
    ``[g*G, (g+1)*G)``. The prefix is identical across a group, hence stored once on the row;
    ``group_mean`` / ``group_std`` are the statistics the group-relative advantage was computed
    from, so ``sign(reward_used - group_mean)`` is recoverable from the EDA alone.

    Args:
        group_base: mutable ``{"n": int}`` carried across every call within an iteration, so
            ``branch_id`` is unique over the iteration's many reward-fn invocations. Seed it from
            the reloaded snapshot on a mid-iteration resume, or the ids restart at 0 and collide.
        rewards: the vector actually returned to TRL (:func:`rewards_for_trl` output), parallel
            to *candidates*.

    Notes:
        **A batch that is not divisible by G is skipped with a warning, never regrouped.** A
        mis-grouped record would produce plausible-looking group statistics for candidates that
        were never siblings -- silently wrong EDA is worse than a missing row, and losing EDA rows
        must never take down a multi-hour training run.

        **Group statistics are TRL's, not a prettier version of them.** They are computed over
        the whole block of G with any surviving ``None`` at 0.0 (trl 1.4.0's ``nansum``, see
        :func:`rewards_for_trl`) and with the SAMPLE SD (ddof=1), because that is what
        ``torch.Tensor.std`` defaults to at ``grpo_trainer.py:2151``. Using the population SD
        over only the graded siblings -- the obvious-looking choice -- overstates every
        reconstructed advantage by ``sqrt(G/(G-1))`` = 6.9% at G=8, and gets the SIGN wrong on any
        group that contained a substitution.
    """
    G = max(1, int(num_generations))
    n = len(candidates)
    if n == 0:
        return
    if n % G != 0:
        print(
            f"{_LOG}WARNING: EDA: {n} completions not divisible by G={G}; skipping the group "
            f"records for this call rather than risking a mis-grouped row"
        )
        return
    if len(rewards) != n:
        print(
            f"{_LOG}WARNING: EDA: {len(rewards)} rewards for {n} candidates; skipping the group "
            f"records for this call rather than pairing a score with the wrong candidate"
        )
        return

    base = group_base["n"]
    for grp in range(n // G):
        start = grp * G
        block = list(candidates[start:start + G])
        block_rewards = list(rewards[start:start + G])
        # Exactly the vector TRL reduces: a None that no sibling could fill is floored to 0.0
        # by its nansum, so the recorded statistics say so instead of hiding it.
        used = [0.0 if r is None else float(r) for r in block_rewards]
        valid = [(c.score, j) for j, c in enumerate(block) if c.score is not None]
        record = build_branch_record(
            phase=PHASE_GROUP,
            iteration=iteration,
            conversation_id=_coerce_int(_at(conversation_ids, start, None)),
            persona_id=(
                None
                if _at(persona_ids, start, None) is None
                else _coerce_int(_at(persona_ids, start, None))
            ),
            branch_id=base + grp,
            prefix=_at(transcripts, start, None),
            candidates=[
                c.to_record(
                    j,
                    reward_used=(
                        block_rewards[j] if c.score is None and block_rewards[j] is not None
                        else None
                    ),
                )
                for j, c in enumerate(block)
            ],
            chosen_idx=(max(valid)[1] if valid else None),
            epoch=epoch,
            group_mean=statistics.fmean(used),
            group_std=(statistics.stdev(used) if len(used) > 1 else 0.0),
            # TRL puts the policy in eval mode for evaluate(); those groups never produced a
            # gradient. Written on EVERY row, True or False -- an absent key is a key nobody
            # filters on, and pooling the two halves silently blends held-out candidates into
            # every per-iteration aggregate (see EDARecorder.aggregate).
            eval_pass=not was_training,
        )
        recorder.append(record)
    group_base["n"] = base + n // G


def make_reward_fn(
    model,
    tokenizer,
    client,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    *,
    recorder=None,
    sp_therapist: Optional[str] = None,
    iteration: int = 0,
    num_generations: int = 1,
    lookahead_state: Optional[LookaheadState] = None,
    branch_id_start: Optional[int] = None,
) -> Callable:
    """Build the async reward callable TRL's ``GRPOTrainer`` calls once per optimizer step.

    Call this once per iteration, with that iteration's CURRENT policy: the closure captures
    *model*, so the look-ahead rollout continues each candidate under the weights being trained
    rather than under some earlier snapshot.

    Args:
        model: the therapist policy (already ``patch_generate``-ed). Only look-ahead touches it.
        tokenizer: the therapist tokenizer, for the look-ahead rollout.
        client: an OpenAI-compatible async client for the patient and oracle roles.
        oracle_cfg: grader binding, questionnaire ids, concurrency, ``min_success_ratio``.
        la_cfg: ``k`` and the rollout's sampling knobs. ``k == 0`` disables look-ahead entirely.
        primitives: loop-keyed semaphores + GPU lock. MUST be the shared instance -- TRL awaits
            this closure on its own event loop, and a semaphore cached anywhere else would raise
            "attached to a different loop" partway into training.
        recorder: an :class:`~core.recorder.EDARecorder`, or ``None`` to record nothing.
        sp_therapist: the therapist system prompt, required when ``la_cfg.k > 0``.
        iteration: stamped onto every EDA row.
        num_generations: TRL's ``G``. Used ONLY to recover prompt-groups for the EDA; the reward
            values themselves are per-candidate and G-independent.
        lookahead_state: the iteration's :class:`~core.lookahead.LookaheadState`. One is created
            if omitted; pass your own only when the same iteration also runs look-ahead outside
            this closure.
        branch_id_start: first EDA ``branch_id`` this closure may use. Leave ``None`` on a fresh
            iteration (it then continues from whatever *recorder* already holds, i.e. 0). ⚠ On a
            MID-ITERATION RESUME this must be past the reloaded snapshot's ids, or the
            post-resume groups restart at 0 and collide with the pre-crash rows in the same
            ``generations.jsonl`` -- and ``(conversation_id, branch_id)``, the key the EDA
            prescribes for every per-branch aggregation, would then pool two unrelated
            prompt-groups. :meth:`~core.recorder.EDARecorder.next_group_branch_id` computes it,
            and is the default when a *recorder* is given.

    Returns:
        ``async def reward_fn(prompts, completions, transcript, **kwargs) -> List[Optional[float]]``,
        one reward per completion, in order. An ungraded candidate does NOT come back as ``None``:
        :func:`rewards_for_trl` gives it its group's mean first, because trl 1.4.0 floors a
        ``None`` to 0.0 rather than skipping it. ``None`` survives only when an entire group
        failed, where TRL's flooring is harmless (a constant group has no advantage) and its own
        all-None warning fires. The state object is attached to the returned
        callable as ``reward_fn.lookahead_state``, so after training the trainer can read
        ``.sub_batch`` (and ``.oom_events``) into ``run_metadata.json`` -- the sub-batch is in no
        ``EXPERIMENT_NAME``, so a halving leaves no other trace and per-iteration wall-clock
        silently stops being comparable.

    Raises:
        ValueError: ``la_cfg.k > 0`` with no *sp_therapist*. Checked here, at factory time,
            because the alternative is discovering it inside the first reward call after the
            iteration has already paid for its rollouts.

    Notes:
        **``transcript`` is a required parameter of the returned callable on purpose.** TRL
        forwards every non-prompt/completion dataset column as a keyword argument, and only when
        the trainer is configured with ``remove_unused_columns=False``. If that flag is lost, the
        call fails immediately with a missing-argument error instead of quietly scoring against
        something else.

        **Do not wrap the result in ``run_async``.** TRL 1.x awaits an async reward function
        natively; a sync wrapper would try to start a second loop inside a running one. Sync
        callers (smoke tests) are the exception and should use
        :func:`core.concurrency.run_async` themselves.

        ``kwargs`` is also where the optional ``log_metric`` callable and ``trainer_state``
        arrive. Both are used opportunistically and inside ``try``: a TRL logging change must
        never abort a training run.
    """
    if la_cfg.k > 0 and not (sp_therapist or "").strip():
        raise ValueError(
            "make_reward_fn(la_cfg.k > 0) requires sp_therapist: the look-ahead rollout "
            "generates therapist turns from the policy and needs its system prompt, or the "
            "simulated continuation is drawn from a different prompt distribution than training."
        )

    # Running prompt-group offset, so branch_id stays unique across the iteration's many calls
    # AND across a mid-iteration resume, where this closure is rebuilt but the recorder is
    # reloaded from the checkpoint snapshot. Seeded past whatever the recorder already holds.
    if branch_id_start is None:
        _next = getattr(recorder, "next_group_branch_id", None)
        branch_id_start = int(_next()) if callable(_next) else 0
    group_base: Dict[str, int] = {"n": max(0, int(branch_id_start))}
    # ONE state for the whole iteration: this is what makes the look-ahead's OOM sub-batch
    # halving sticky across optimizer steps. A per-call state would re-pay the OOM on every one
    # of the iteration's ~135 steps.
    la_state = lookahead_state if lookahead_state is not None else LookaheadState()

    async def reward_fn(prompts, completions, transcript, **kwargs) -> List[Optional[float]]:
        # Captured BEFORE look-ahead: the rollout toggles model.eval() and restores it, so
        # reading `training` afterwards would report train mode even during evaluate().
        was_training = bool(getattr(model, "training", True))

        candidates, telemetry = await _score_candidates(
            model,
            tokenizer,
            client,
            oracle_cfg,
            la_cfg,
            primitives,
            transcripts=list(transcript),
            completions=list(completions),
            sp_therapist=sp_therapist,
            sp_patient_list=kwargs.get("patient_system_prompt"),
            label="grpo/reward",
            enforce_success_ratio=True,
            state=la_state,
        )

        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            try:
                if telemetry["n_oracle"] > 0:
                    # Grader health: successes over candidates actually sent to the oracle.
                    log_metric("oracle/success_rate", telemetry["oracle_success_rate"])
                if telemetry["n"] > 0:
                    log_metric(
                        "reward/degenerate_frac", telemetry["n_degenerate"] / telemetry["n"]
                    )
                    log_metric(
                        "reward/graded_frac",
                        (telemetry["n_success"] + telemetry["n_degenerate"]) / telemetry["n"],
                    )
                if la_cfg.k > 0 and telemetry["n_oracle"] > 0:
                    log_metric(
                        "lookahead/realized_turns_mean", telemetry["realized_turns_mean"]
                    )
                if la_cfg.k > 0 and telemetry["n_rollout"] > 0:
                    # Simulator failures left ungraded (K-asymmetry avoided, gate-counted). A
                    # rising curve here is a saturating patient server, hours before the gate.
                    log_metric(
                        "lookahead/not_graded_frac",
                        telemetry["n_lookahead_failed"] / telemetry["n_rollout"],
                    )
            except Exception as exc:  # logging must never take a run down
                print(f"{_LOG}WARNING: log_metric failed (non-fatal): {exc}")

        # In-memory only. The caller flushes once per iteration -- the Colab output dir is a
        # Drive-FUSE mount and this is the hot path.
        # Built BEFORE the recorder call: the EDA row must carry the group statistics TRL will
        # actually reduce, not a tidier set computed from the raw oracle scores.
        rewards = rewards_for_trl(candidates, num_generations)

        if recorder is not None and getattr(recorder, "enabled", False):
            trainer_state = kwargs.get("trainer_state")
            epoch = getattr(trainer_state, "epoch", None) if trainer_state is not None else None
            _record_grpo_groups(
                recorder,
                iteration=iteration,
                num_generations=num_generations,
                group_base=group_base,
                epoch=epoch,
                was_training=was_training,
                transcripts=list(transcript),
                candidates=candidates,
                rewards=rewards,
                conversation_ids=kwargs.get("conversation_id"),
                persona_ids=kwargs.get("persona_id"),
            )

        return rewards

    # Exposed so the trainer can stamp the realized sub-batch into run_metadata.json.
    reward_fn.lookahead_state = la_state
    return reward_fn


# =============================================================================
# PTO: THE PREFERENCE-BRANCH SCORER
# =============================================================================


async def score_pref_candidates(
    model,
    tokenizer,
    client,
    oracle_cfg: OracleConfig,
    la_cfg: LookaheadConfig,
    primitives: AsyncPrimitives,
    *,
    transcripts: Sequence[str],
    completions: Sequence[str],
    sp_therapist: Optional[str] = None,
    sp_patient_list: Optional[Sequence[str]] = None,
    enforce_success_ratio: bool = True,
    lookahead_state: Optional[LookaheadState] = None,
) -> List[CandidateScore]:
    """Score PTO branch candidates through the SAME path GRPO's reward uses.

    PTO uses the scores only to pick a (chosen, rejected) pair, so it needs the detail GRPO
    discards: the per-questionnaire breakdown for ``pref_pairs/pairs.csv``, and the look-ahead
    tail for the EDA row. This returns :class:`CandidateScore` objects instead of bare floats;
    everything up to that point is byte-identical to the reward path, which is what keeps the
    K in {0, 5} contrast meaningful on both methods.

    Args:
        transcripts: parallel to *completions* -- the trunk prefix each candidate continues.
            The greedy grower runs lock-step across trunks, so one call typically carries
            ``n_trunks * M`` candidates and each trunk's prefix repeats M times.
        completions: raw sampled turns; cleaned here, so callers need not pre-clean.
        sp_therapist: the therapist system prompt; required when ``la_cfg.k > 0``.
        sp_patient_list: parallel persona system prompts for the simulated patient.
        enforce_success_ratio: raise when the oracle success rate is below
            ``oracle_cfg.min_success_ratio``. Leave it on: the tau filter would otherwise
            absorb a dead grader as "no branch beat any other", and the iteration would fail
            much later with the far less informative "produced 0 pref pairs".
        lookahead_state: pass ONE :class:`~core.lookahead.LookaheadState` per iteration and reuse
            it for every branch point. Without it the look-ahead's OOM sub-batch halving restarts
            at ``la_cfg.sub_batch_size`` on every call, re-paying the OOM at each depth of every
            trunk, and the realized sub-batch is lost before it can be recorded.

    Returns:
        One :class:`CandidateScore` per completion, in order. A candidate whose completion
        cleaned to empty carries ``score == REWARD_FLOOR`` and ``degenerate=True``; one whose
        oracle call failed, or whose look-ahead the simulator could not run, carries
        ``score is None`` (with ``not_graded_reason``) -- exclude those from the tau comparison
        rather than treating them as a low score. A candidate that closed the session keeps the
        keyword in ``completion`` (so the trunk advance freezes the trunk, unchanged) but was
        graded on the text before it, with no rollout; ``ended_by_candidate`` says so, and the
        DPO pair must then be built from the split text (``pto_trainer._pair_text``), never
        from ``completion``.

    Raises:
        ValueError: parallel lists disagree in length, or ``la_cfg.k > 0`` with no
            *sp_therapist*.
        RuntimeError: graded-over-gradable rate (oracle AND look-ahead failures) below the
            configured floor.

    Notes:
        Feed the results straight to the recorder with
        :meth:`CandidateScore.to_record`, tagging ``role="chosen"`` / ``"rejected"`` only when
        the tau filter actually emitted a pair -- the recorder counts a ``"rejected"`` role as
        proof that a preference pair exists at that branch point.
    """
    if la_cfg.k > 0 and not (sp_therapist or "").strip():
        raise ValueError(
            "score_pref_candidates(la_cfg.k > 0) requires sp_therapist: the look-ahead rollout "
            "generates therapist turns from the policy and needs its system prompt."
        )

    candidates, _ = await _score_candidates(
        model,
        tokenizer,
        client,
        oracle_cfg,
        la_cfg,
        primitives,
        transcripts=transcripts,
        completions=completions,
        sp_therapist=sp_therapist,
        sp_patient_list=sp_patient_list,
        label="pto/pref",
        enforce_success_ratio=enforce_success_ratio,
        state=lookahead_state,
    )
    return candidates
