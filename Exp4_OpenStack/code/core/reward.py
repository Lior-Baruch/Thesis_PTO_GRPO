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
    LookaheadConfig,
    LookaheadState,
    seed_transcript,
    simulate_lookahead_batch,
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
    "CandidateScore",
    "make_reward_fn",
    "rewards_for_trl",
    "score_pref_candidates",
]


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
        completion: the CLEANED completion (``""`` marks a degenerate turn).
        score: the RAW result -- the oracle's unweighted mean across questionnaires, or
            :data:`~core.oracle.REWARD_FLOOR` when :attr:`degenerate`, or ``None`` when the
            oracle failed. ``None`` means "not graded", never "graded badly", and PTO must not
            rank on a fabricated number -- it excludes such a candidate from the tau comparison.
            ⚠ GRPO cannot pass ``None`` to TRL: the pinned trl 1.4.0 turns it into NaN and then
            ``nansum``s it to **0.0**, i.e. trains on it as the worst possible completion. The
            reward vector handed to TRL is therefore built by :func:`rewards_for_trl`, which
            substitutes the group mean; :attr:`score` itself stays raw so the EDA can still tell
            a failure from a floor.
        sub_scores: ``{questionnaire id (str): mean}``, or ``None``. Kept RAW-oracle, so a floored
            row is identifiable as ``score == REWARD_FLOOR`` with ``sub_scores is None``.
        success: did the oracle return a usable score for every requested questionnaire.
            **False for a degenerate candidate too** -- no call was made, so there is no usable
            score. That means the recorder's ``eda/oracle_success_rate`` mixes grader failures
            with policy degeneracy; the gate metric is the one logged as ``oracle/success_rate``,
            whose denominator is the candidates actually sent to the grader.
        attempts: total oracle calls made for this candidate (retries included); 0 when degenerate.
        degenerate: the completion cleaned to ``""``; no oracle call and no rollout were made.
        scored_text: the exact string the oracle graded (or would have, when degenerate).
        lookahead: ``{"k", "tail", "realized_turns", "ended_early"}`` when a rollout ran, else
            ``None``. ``None`` at K=0 AND for degenerate candidates at K>0 -- in both cases no
            simulation happened, and the recorder's look-ahead scalars are averages over sims
            that actually ran.
    """

    completion: str
    score: Optional[float]
    sub_scores: Optional[Dict[str, float]]
    success: bool
    attempts: int
    degenerate: bool
    scored_text: str
    lookahead: Optional[Dict[str, Any]]

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
                :attr:`score` (i.e. this candidate's oracle call failed and
                :func:`rewards_for_trl` substituted its group's mean). Leave ``None`` when the
                two agree; the key is then absent and the EDA falls back to ``score``.

        Notes:
            Both methods build their EDA rows through this, so a GRPO candidate and a PTO
            candidate are the same shape and the EDA needs no per-method branch.
        """
        return build_candidate(
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
        ``(candidates, telemetry)``. ``telemetry`` carries ``n``, ``n_degenerate``, ``n_oracle``,
        ``n_success``, ``success_rate``, ``lookahead_seconds``, ``realized_turns_mean``,
        ``n_ended_early`` -- enough for the batch log line and for ``log_metric``.

    Raises:
        ValueError: ``transcripts`` and ``completions`` differ in length, or look-ahead / the
            oracle returned a result list of the wrong length. A length mismatch would pair a
            score with the wrong candidate, which is unrecoverable and invisible downstream, so
            it is fatal rather than warned about.
        RuntimeError: the oracle success rate is below ``oracle_cfg.min_success_ratio`` and
            *enforce_success_ratio* is set.

    Notes:
        **Degenerate candidates are excluded from both the rollout and the oracle batch**, not
        scored and then overridden. That keeps the success-rate gate a measure of grader/server
        health rather than of policy degeneracy (an all-degenerate batch makes zero oracle calls
        and cannot trip the gate), and it is free -- the reward is the floor either way.

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

    cleaned: List[str] = [clean_completion(c) for c in completions]
    degenerate: List[bool] = [not c for c in cleaned]
    active: List[int] = [i for i in range(n) if not degenerate[i]]

    telemetry: Dict[str, float] = {
        "n": float(n),
        "n_degenerate": float(n - len(active)),
        "n_oracle": float(len(active)),
        "n_success": 0.0,
        "success_rate": 1.0,
        "lookahead_seconds": 0.0,
        "realized_turns_mean": 0.0,
        "n_ended_early": 0.0,
    }

    # -- 1. Build the text the oracle will read, per active candidate ---------
    scored_text: Dict[int, str] = {}
    la_records: Dict[int, Dict[str, Any]] = {}

    if la_cfg.k > 0 and active:
        # simulate_lookahead_batch already warns when every patient system prompt is empty and
        # raises on a length mismatch of its own three inputs; not duplicated here.
        la_start = time.time()
        results = await simulate_lookahead_batch(
            model,
            tokenizer,
            client,
            la_cfg,
            primitives,
            [transcripts[i] for i in active],
            [cleaned[i] for i in active],
            sp_therapist or "",
            [str(_at(sp_patient_list, i, "")) for i in active],
            state=state,
        )
        telemetry["lookahead_seconds"] = time.time() - la_start
        if len(results) != len(active):
            raise ValueError(
                f"simulate_lookahead_batch returned {len(results)} results for "
                f"{len(active)} sims; refusing to pair scores with the wrong candidates"
            )

        realized: List[float] = []
        for slot, i in enumerate(active):
            res = results[slot]
            scored_text[i] = res.extended_transcript
            # LookaheadResult owns the recorder dict's shape; building it here by hand is how the
            # two would drift.
            la_records[i] = res.to_record()
            realized.append(float(res.realized_turns))
            if res.ended_early:
                telemetry["n_ended_early"] += 1.0
        telemetry["realized_turns_mean"] = statistics.fmean(realized) if realized else 0.0
    else:
        for i in active:
            scored_text[i] = seed_transcript(transcripts[i], cleaned[i])

    # -- 2. Oracle ------------------------------------------------------------
    details: List[Dict[str, Any]] = []
    if active:
        details = list(
            await score_conversations_batch(
                client, oracle_cfg, primitives, [scored_text[i] for i in active]
            )
        )
        if len(details) != len(active):
            raise ValueError(
                f"score_conversations_batch returned {len(details)} results for "
                f"{len(active)} conversations; refusing to pair scores with the wrong candidates"
            )

    n_success = sum(1 for d in details if d.get("success"))
    telemetry["n_success"] = float(n_success)
    # batch_success_ratio owns the empty-batch convention (1.0, not 0.0): an all-degenerate batch
    # makes no oracle call, carries no evidence about the grader, and must not kill the run.
    telemetry["success_rate"] = batch_success_ratio(details)

    line = (
        f"{_LOG}{label}: {n} candidates "
        f"({int(telemetry['n_degenerate'])} degenerate -> floor), "
    )
    if active:
        line += f"oracle {n_success}/{len(active)} ok ({telemetry['success_rate']:.0%})"
        if la_cfg.k > 0:
            line += (
                f", look-ahead {telemetry['lookahead_seconds']:.1f}s "
                f"(avg {telemetry['realized_turns_mean']:.1f}/{la_cfg.k} turns realized, "
                f"{int(telemetry['n_ended_early'])} ended early)"
            )
    else:
        line += "no oracle calls (every completion was degenerate)"
    print(line)

    # -- 3. The gate. Missingness here is not random -- see the module docstring.
    if enforce_success_ratio and active and telemetry["success_rate"] < oracle_cfg.min_success_ratio:
        raise RuntimeError(
            f"Oracle success rate {telemetry['success_rate']:.1%} is below "
            f"min_success_ratio={oracle_cfg.min_success_ratio:.0%} "
            f"({len(active) - n_success}/{len(active)} candidates failed). Aborting rather than "
            f"training on a biased subset -- failures are not missing at random, and in GRPO a "
            f"systematically-absent sibling shifts both the mean and the std of its group. "
            f"Likely causes: the vLLM server died or is out of memory (check its log and "
            f"tools/vllm_serve.ensure_alive), the grader started rejecting the json_schema "
            f"response_format, transcripts have outgrown --max-model-len, or an API-backed "
            f"oracle hit a spend cap / rate limit. The iteration resumes from its last "
            f"checkpoint once the cause is fixed."
        )

    # -- 4. Assemble ----------------------------------------------------------
    by_index = {i: details[slot] for slot, i in enumerate(active)}
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
                    log_metric("oracle/success_rate", telemetry["success_rate"])
                if telemetry["n"] > 0:
                    log_metric(
                        "reward/degenerate_frac", telemetry["n_degenerate"] / telemetry["n"]
                    )
                if la_cfg.k > 0 and telemetry["n_oracle"] > 0:
                    log_metric(
                        "lookahead/realized_turns_mean", telemetry["realized_turns_mean"]
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
        oracle call failed carries ``score is None`` -- exclude those from the tau comparison
        rather than treating them as a low score.

    Raises:
        ValueError: parallel lists disagree in length, or ``la_cfg.k > 0`` with no
            *sp_therapist*.
        RuntimeError: oracle success rate below the configured floor.

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
