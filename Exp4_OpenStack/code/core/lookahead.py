"""lookahead.py -- K extra simulated turns between a candidate and the oracle.

This module is the thesis's central experimental lever, and it is small on purpose.

Scoring a candidate therapist turn ``t`` on the prefix it answers -- ``oracle(c + t)`` --
rewards openings that *look good in isolation*: a warm reflection, a tidy open question,
graded on the two sentences immediately around it. Scoring ``oracle(c + t + K more turns
simulated by the CURRENT policy against the patient simulator)`` rewards openings that
*lead somewhere good*, because the K turns are drawn from the same policy that produced
``t`` and therefore expose what that opening actually unwinds into. That is the whole
claim of the look-ahead paper, and ``K`` is the only thing this module varies.

Look-ahead touches nothing else in either optimizer. It changes **what text the oracle
reads**, never the loss: GRPO still takes the returned score as its group-relative reward
and PTO still uses it only to rank the ``M`` branches. That is precisely why the
``K in {0, 5}`` contrast is meaningful on *both* methods -- it isolates the look-ahead
lever from the loss family.

Three invariants this file exists to hold
-----------------------------------------

**1. Lock-step, not per-sim.** All ``B`` simulations advance together: one padded batched
``model.generate`` per simulated therapist turn, then one batched patient round. Not a
loop over sims. With ``B = 128`` completions per optimizer step and ``K = 5`` the serial
shape is ~320 generate calls; the lock-step shape is ~2. Exp3 ran the batched path against
its serial reference on GPU and found them statistically equivalent (``|delta mean| =
0.024``) at 1.5x the throughput -- they are not bit-identical, because the sampling RNG is
consumed differently by a padded batch. Exp4 keeps only the batched path; that equivalence
claim is inherited from Exp3's check, not re-established here.

Speaker phase is a pure function of the step index (even = patient, odd = therapist)
because every sim is constructed here at the same phase and the only way to leave the
cadence is to go inactive. There is no speaker-desync case to recover from, unlike the
conversation loop.

**2. The GPU lock is held across the therapist ``generate`` and NOTHING else.** The
patient round runs under ``primitives.patient_sem()`` with the lock released. Holding the
lock across a patient ``await`` would serialize the entire batch behind one network
round-trip and delete the reason the rollout is batched at all. This is the single most
important line-level invariant in the module.

**3. Nothing here may take a training step down.** This runs *inside* a live optimizer
step, with the policy in ``train()`` and TRL waiting on the reward. So:

- ``eval()`` + ``use_cache = True`` are toggled around generation and restored in a
  ``finally``. Leaking either corrupts the very step that called us.
- An OOM halves the sub-batch and retries the chunk; the halving is **sticky**, so a step
  pays the OOM cost once rather than every turn (and, with a :class:`LookaheadState`
  handed in, once per *arm* rather than once per step). At sub-batch 1 an OOM freezes that
  one sim.
- A non-OOM runtime error freezes the chunk and advances -- deliberately unlike the
  conversation loop, which aborts. A conversation that dies can be regenerated; an
  optimizer step that dies takes the iteration with it, and the siblings that worked still
  deserve rewards.
- A transcript that will not parse freezes that sim on its seed instead of raising.

A **frozen** sim is not an error: it keeps the transcript it has reached, drops out of
later steps, and is still scored. It simply got a shorter look-ahead than ``K``, which
:attr:`LookaheadResult.realized_turns` and :attr:`LookaheadResult.ended_early` record.
Those two fields are science, not telemetry -- Exp3 measured 19-23% of ``K=5`` tails
ending early, and early-ending siblings both score lower and are ~23% less likely to be
their group's argmax.

The tail is recovered by EXACT string slicing
---------------------------------------------
``seed = transcript + "\\n\\n[THERAPIST]: " + completion`` and every simulated turn is
*appended* to that string. Nothing in this path re-serializes the transcript, so
``extended[len(seed):]`` is exactly the K simulated turns and nothing else. Reformatting
the transcript anywhere upstream -- changing a label, changing the ``"\\n\\n"`` joiner --
would break the slice **silently**, which is why :func:`check_transcript_format_agreement`
exists and why the smoke test calls it.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from roles import PATIENT_DEFAULT, RoleBinding, make_client

from core.conversations import (
    SESSION_END_KEYWORD,
    format_conversation_for_oracle,
    generate_patient_batch,
    parse_transcript_to_messages,
)
from core.policy import STOP_STRINGS, generate_therapist_batch

__all__ = [
    "TURN_JOINER",
    "THERAPIST_LABEL",
    "PATIENT_LABEL",
    "LookaheadConfig",
    "LookaheadResult",
    "LookaheadState",
    "seed_transcript",
    "simulate_lookahead_batch",
    "check_transcript_format_agreement",
]


# =============================================================================
# TRANSCRIPT FORMAT
# =============================================================================

# The transcript grammar, spelled here because the seed slice depends on it byte for byte.
# `core.conversations.format_conversation_for_oracle` owns the same grammar on the write
# side; `check_transcript_format_agreement` is what proves the two still agree.
TURN_JOINER = "\n\n"
THERAPIST_LABEL = "[THERAPIST]:"
PATIENT_LABEL = "[PATIENT]:"

_LABEL_BY_SPEAKER = {"therapist": THERAPIST_LABEL, "patient": PATIENT_LABEL}


def seed_transcript(transcript: str, completion: str) -> str:
    """The text the oracle would grade at ``K = 0``: prefix plus the candidate turn.

    Args:
        transcript: The conversation-so-far, already in oracle format and ending on a
            patient turn.
        completion: The candidate therapist turn, already cleaned.

    Returns:
        ``f"{transcript}\\n\\n[THERAPIST]: {completion}"``.

    Notes:
        Every look-ahead turn is appended to this exact string, so
        ``extended[len(seed):]`` recovers the tail with no parsing. The EDA reconstructs a
        scored text the same way (``prefix + "\\n\\n[THERAPIST]: " + completion + tail``),
        which is why this one-line concatenation is a public function rather than an inline
        f-string in three places that can drift apart.
    """
    return f"{transcript}{TURN_JOINER}{THERAPIST_LABEL} {completion}"


def _append_turn(extended: str, speaker: str, content: str) -> str:
    """Append one labelled utterance to a transcript. Pure concatenation, by contract."""
    return f"{extended}{TURN_JOINER}{_LABEL_BY_SPEAKER[speaker]} {content}"


def _split_session_end(text: str) -> Tuple[str, bool]:
    """Split a response at :data:`SESSION_END_KEYWORD`.

    Returns:
        ``(content_before_the_marker, marker_was_present)``. The content is stripped and may
        be empty -- a speaker that emits only the marker contributes no utterance.

    Notes:
        Local rather than imported: Exp4's ``core.conversations`` contract exposes the
        keyword but no splitter. If it ever grows one, delete this and use it -- two
        definitions of "where does a closing turn end" is exactly the kind of drift that
        makes an ended-early rate un-auditable.
    """
    idx = text.upper().find(SESSION_END_KEYWORD)
    if idx == -1:
        return text, False
    return text[:idx].strip(), True


# =============================================================================
# CONFIG / RESULT / STATE
# =============================================================================


@dataclass(frozen=True)
class LookaheadConfig:
    """Knobs for the K-turn rollout. Frozen: it is stashed in the trainer's config tree.

    Attributes:
        k: Extra utterances simulated after the candidate. ``0`` disables look-ahead
            entirely -- and ``K = 0`` is an ARM of the experiment, not a switched-off
            feature, so that path short-circuits to zero API calls and zero generates
            rather than running an empty loop.
        temperature_therapist: Sampling temperature for simulated therapist turns. Match
            the temperature the candidates themselves were drawn at, or the rollout is
            measuring a different policy from the one being scored.
        temperature_patient: Sampling temperature for simulated patient turns.
        max_tokens: ``max_new_tokens`` per simulated turn, both sides.
        max_input_tokens: Prompt budget for the therapist generate. Truncation is LEFT (see
            ``core.policy.setup_tokenizer``), so a long rollout drops its oldest turns and
            always keeps the patient utterance being answered.
        patient_binding: Which model plays the patient here. **It must be the same binding
            the conversations were generated with** -- the look-ahead patient defines the
            future the candidate is graded on, so swapping it changes the reward.
        stop_strings: ChatML markers that terminate a therapist turn. Defaults to
            ``core.policy.STOP_STRINGS``; the base policy self-plays without them.
        sub_batch_size: Cap on the therapist generate batch. ``None`` means "one padded
            generate over all active sims", which is fastest and is what the A100 budget
            assumes. Halved automatically on OOM, and the halving is sticky (see
            :class:`LookaheadState`).

    Notes:
        ``sub_batch_size`` is NOT part of ``EXPERIMENT_NAME`` -- it is a memory knob, not a
        science knob. But it changes per-iteration wall-clock, and wall-clock is a reported
        number in the look-ahead paper's cost argument, so it must be mirrored into
        ``run_metadata.json``. A halving that happens at run time and is never recorded is
        the failure mode: two arms then look like they cost different amounts for a reason
        nobody wrote down.
    """

    k: int = 0
    temperature_therapist: float = 0.9
    temperature_patient: float = 0.7
    max_tokens: int = 200
    max_input_tokens: int = 2048
    patient_binding: RoleBinding = PATIENT_DEFAULT
    stop_strings: Tuple[str, ...] = tuple(STOP_STRINGS)
    sub_batch_size: Optional[int] = None


@dataclass(frozen=True)
class LookaheadResult:
    """One simulation's outcome, in input order.

    Attributes:
        extended_transcript: What the oracle should grade -- seed plus the simulated turns.
            At ``k == 0`` this is exactly the seed.
        tail: Only the simulated turns (``extended_transcript`` minus the seed), recovered
            by exact slicing. ``""`` when nothing was simulated. Stored instead of the full
            extension because the prefix is already on the branch record, once, and
            duplicating it per candidate multiplies ``generations.jsonl`` by ``G``.
        realized_turns: Utterances actually appended. Counted as they are appended, not by
            counting role labels afterwards -- a turn whose *content* contains the string
            ``"[PATIENT]:"`` would inflate a label count.
        ended_early: ``realized_turns < k``. This is the quantity Exp3 reported (19-23% of
            ``K=5`` tails), so it keeps that definition even though a session that closes
            exactly on the K-th turn reads as ``False``; :attr:`stop_reason` is where that
            case is visible.
        stop_reason: ``""`` when the rollout ran to ``k`` (or ``k == 0``), else one of
            ``"session_ended"``, ``"patient_error"``, ``"gpu_error"``, ``"degenerate"``,
            ``"parse_error"``.
        k: The ``k`` this result was produced under, carried so a record is self-describing.
    """

    extended_transcript: str
    tail: str
    realized_turns: int
    ended_early: bool
    stop_reason: str = ""
    k: int = 0

    def to_record(self) -> Dict[str, Any]:
        """The nested ``lookahead`` dict ``core.recorder.build_candidate`` expects.

        Notes:
            The recorder wants ``None`` (not this dict) at ``K = 0``: an ABSENT dict is what
            marks a no-look-ahead run, and the look-ahead scalars are then simply not
            emitted. Caller decides; this method always returns a dict.
        """
        return {
            "k": int(self.k),
            "tail": self.tail,
            "realized_turns": int(self.realized_turns),
            "ended_early": bool(self.ended_early),
        }


@dataclass
class LookaheadState:
    """Mutable, caller-owned rollout state: the sticky sub-batch plus call counters.

    Create ONE per iteration and pass it to every
    :func:`simulate_lookahead_batch` call. That is what makes the OOM halving sticky across
    optimizer steps: without it each call restarts at ``cfg.sub_batch_size``, re-pays the
    OOM, and re-halves -- an OOM costs one wasted generate, so paying it 135 times an
    iteration is real money on a preemptible GPU.

    Attributes:
        sub_batch: Current therapist generate cap. ``None`` = "all active sims"; set once
            from ``cfg.sub_batch_size`` on first use, then only ever lowered.
        gpu_calls: Total ``model.generate`` calls issued (chunks, not turns).
        oom_events: OOM returns seen. Non-zero means the VRAM budget is at its edge.
        runtime_errors: Non-OOM generate failures. Non-zero means chunks were frozen and
            their sims scored on short transcripts -- worth surfacing, not silently
            averaging away.

    Notes:
        Read ``sub_batch`` after the iteration and write it into ``run_metadata.json``:
        it is not in ``EXPERIMENT_NAME``, so a halving leaves no other trace, and
        per-iteration wall-clock stops being comparable without it.
    """

    sub_batch: Optional[int] = None
    gpu_calls: int = 0
    oom_events: int = 0
    runtime_errors: int = 0

    def summary(self) -> str:
        """One-line render for a training log."""
        sb = "all" if self.sub_batch is None else str(self.sub_batch)
        return (
            f"sub_batch={sb} gpu_calls={self.gpu_calls} "
            f"oom={self.oom_events} runtime_err={self.runtime_errors}"
        )


@dataclass
class _Sim:
    """Per-completion mutable state for the lock-step rollout. Private."""

    msgs_therapist: List[Dict[str, str]]
    msgs_patient: List[Dict[str, str]]
    seed: str
    extended: str
    active: bool = True
    realized_turns: int = 0
    stop_reason: str = ""

    def freeze(self, reason: str) -> None:
        """Drop this sim from later steps, keeping the transcript it has reached."""
        self.active = False
        if not self.stop_reason:
            self.stop_reason = reason

    def append(self, speaker: str, content: str) -> None:
        """Append one simulated utterance and count it as realized."""
        self.extended = _append_turn(self.extended, speaker, content)
        self.realized_turns += 1

    def to_result(self, k: int) -> LookaheadResult:
        """Freeze into the public result, slicing the tail off the seed exactly."""
        ext = self.extended
        # startswith() always holds -- the extension is pure concatenation onto the seed.
        # The guard is here so that a future refactor which reformats the transcript
        # degrades to "tail == the whole thing" instead of silently slicing mid-word.
        tail = ext[len(self.seed):] if ext.startswith(self.seed) else ext
        return LookaheadResult(
            extended_transcript=ext,
            tail=tail,
            realized_turns=self.realized_turns,
            ended_early=self.realized_turns < k,
            stop_reason=self.stop_reason,
            k=k,
        )


# =============================================================================
# THERAPIST GENERATION (OOM-resilient, sync -- runs in an executor)
# =============================================================================


def _therapist_generate_chunked(
    model: Any,
    tokenizer: Any,
    batch_messages: List[List[Dict[str, str]]],
    *,
    max_tokens: int,
    temperature: float,
    max_input_tokens: int,
    stop_strings: Optional[Sequence[str]],
    start_sub_batch: int,
) -> Tuple[List[Optional[str]], int, int, int, int]:
    """One therapist reply per message-list, in chunks, with OOM-driven halving.

    Synchronous: call it through ``run_in_executor`` while holding the GPU lock.
    ``core.policy.generate_therapist_batch`` never raises on OOM -- it cleans the CUDA cache
    and returns ``(None, "oom")`` -- so the whole policy lives in the three branches here:

    - success: place the responses at their indices and advance.
    - ``"oom"``: at ``sb == 1`` freeze that single item and advance (a batch of one that
      still will not fit is not going to fit later either); otherwise halve ``sb`` and retry
      the SAME chunk without advancing. The reduced ``sb`` is returned so it can be made
      sticky.
    - ``"runtime_error"``: halving cannot help, so freeze the chunk's items and advance.
      This is the deliberate divergence from ``conversation_loop_batch``, which aborts:
      killing a GRPO step over one transient generate failure discards the rewards of every
      sibling that worked.

    Returns:
        ``(responses, final_sub_batch, n_generate_calls, n_oom, n_runtime_errors)`` with
        ``responses`` order-aligned to *batch_messages*. ``None`` marks a frozen item;
        ``""`` marks a degenerate one (the caller must distinguish -- see
        :func:`simulate_lookahead_batch`).
    """
    n = len(batch_messages)
    responses: List[Optional[str]] = [None] * n
    sb = max(1, int(start_sub_batch))
    n_calls = 0
    n_oom = 0
    n_runtime = 0

    i = 0
    while i < n:
        chunk = batch_messages[i:i + sb]
        resp, error_type = generate_therapist_batch(
            model,
            tokenizer,
            chunk,
            max_tokens=max_tokens,
            temperature=temperature,
            max_input_tokens=max_input_tokens,
            stop_strings=stop_strings,
        )
        n_calls += 1

        if error_type is None:
            for j, r in enumerate(resp or []):
                responses[i + j] = r
            i += len(chunk)
            continue

        if error_type == "oom":
            n_oom += 1
            if sb == 1:
                responses[i] = None
                i += 1
            else:
                new_sb = max(1, sb // 2)
                print(f"  Look-ahead OOM: sub-batch {sb} -> {new_sb} (sticky), retrying chunk")
                sb = new_sb
            continue

        n_runtime += 1
        for j in range(len(chunk)):
            responses[i + j] = None
        i += len(chunk)

    return responses, sb, n_calls, n_oom, n_runtime


# =============================================================================
# THE ROLLOUT
# =============================================================================


async def simulate_lookahead_batch(
    model: Any,
    tokenizer: Any,
    client: Any,
    cfg: LookaheadConfig,
    primitives: Any,
    transcripts: Sequence[str],
    completions: Sequence[str],
    sp_therapist: str,
    sp_patient_list: Sequence[str],
    *,
    state: Optional[LookaheadState] = None,
) -> List[LookaheadResult]:
    """Simulate ``cfg.k`` extra turns after each candidate, all sims in lock-step.

    Args:
        model: The therapist policy, patched by ``core.policy.patch_generate``. This is the
            CURRENT policy on purpose -- the look-ahead measures where an opening leads
            *under the policy being trained*, so a stale reference model would grade
            candidates against a future that will not happen.
        tokenizer: The therapist tokenizer (left padding, ChatML template).
        client: Async client for ``cfg.patient_binding``. ``None`` is allowed and builds one
            via ``roles.make_client``; pass the shared cached client in the hot path so the
            connection pool is reused.
        cfg: :class:`LookaheadConfig`.
        primitives: ``core.concurrency.AsyncPrimitives`` -- only ``gpu_lock()`` and
            ``patient_sem()`` are used.
        transcripts: Conversation prefixes in oracle format, each ending on a patient turn.
        completions: One candidate therapist turn per transcript, ALREADY cleaned.
        sp_therapist: Therapist system prompt for the simulated therapist turns.
        sp_patient_list: Per-sim patient system prompt. These are the 96 V3 personas and
            they are NOT interchangeable -- passing the wrong one simulates a different
            patient from the one the prefix was generated against.
        state: Caller-owned :class:`LookaheadState`. Pass ONE per iteration so the OOM
            sub-batch halving is sticky across optimizer steps and the call counters
            accumulate. Omitted, an ephemeral one is used and its final sub-batch is lost.

    Returns:
        One :class:`LookaheadResult` per input pair, in input order. Never raises for a
        model, API or parse failure -- those freeze the affected sim.

    Raises:
        ValueError: if the three input sequences differ in length. That is a caller bug and
            silent truncation would drop whole candidates from a GRPO group, shifting the
            group mean AND std for every sibling.

    Notes:
        ``k == 0`` short-circuits: each result is the bare seed with an empty tail, and
        neither the GPU nor the patient server is touched. ``K = 0`` is an arm.

        **The GPU lock is held across the therapist generate only.** The patient round runs
        with it released, bounded by ``primitives.patient_sem()``.

        ``model.eval()`` and ``model.config.use_cache = True`` are set for the duration of
        each generate and restored in a ``finally``: this runs mid-optimizer-step with the
        policy in ``train()`` and ``use_cache = False``, and leaking either setting corrupts
        the step that called us.

        Per-attempt timeout and retry policy for the patient come from
        ``cfg.patient_binding``, not from here. The shape must stay "short per-attempt
        timeout x many retries": a long total budget freezes a sim, and under
        ``scale_rewards="group"`` one frozen sim moves the mean and the std of its group.
    """
    n = len(transcripts)
    if len(completions) != n or len(sp_patient_list) != n:
        raise ValueError(
            "simulate_lookahead_batch inputs must be the same length: "
            f"transcripts={n}, completions={len(completions)}, "
            f"sp_patient_list={len(sp_patient_list)}"
        )

    seeds = [seed_transcript(t, c) for t, c in zip(transcripts, completions)]
    k = int(cfg.k)

    if k <= 0:
        # K=0 arm: the oracle grades exactly (prefix + candidate). No generate, no API call.
        return [
            LookaheadResult(
                extended_transcript=seed,
                tail="",
                realized_turns=0,
                ended_early=False,
                stop_reason="",
                k=0,
            )
            for seed in seeds
        ]

    if state is None:
        state = LookaheadState()
    if state.sub_batch is None and cfg.sub_batch_size is not None:
        state.sub_batch = max(1, int(cfg.sub_batch_size))

    if all(not (sp or "").strip() for sp in sp_patient_list):
        # Not fatal, but the rollout is then conditioned on an empty persona, i.e. a
        # degenerate patient -- and the resulting scores look perfectly normal.
        print(
            "  WARNING: look-ahead k>0 but every patient system prompt is empty; "
            "simulating against a degenerate patient. Check that the dataset carries "
            "'patient_system_prompt'."
        )

    # Always re-resolve on the RUNNING loop (make_client is loop-keyed), even when a client was
    # passed: a handle built on another loop carries keep-alive connections that poison the
    # first patient calls here with APIConnectionError. Per-loop reuse is a dict hit.
    client = make_client(cfg.patient_binding)

    sims: List[_Sim] = []
    for transcript, completion, sp_patient, seed in zip(
        transcripts, completions, sp_patient_list, seeds
    ):
        try:
            msgs_therapist, msgs_patient = parse_transcript_to_messages(
                transcript, sp_therapist, sp_patient
            )
        except ValueError as exc:
            preview = transcript[:280] + ("..." if len(transcript) > 280 else "")
            print(
                f"  WARNING: look-ahead transcript parse failed (length={len(transcript)} "
                f"chars); freezing this sim on its seed. Preview: {preview!r} ({exc})"
            )
            sims.append(_Sim([], [], seed, seed, active=False, stop_reason="parse_error"))
            continue

        # The candidate is the last therapist turn of the conversation the rollout extends.
        msgs_therapist.append({"role": "assistant", "content": completion})
        msgs_patient.append({"role": "user", "content": completion})
        sims.append(_Sim(msgs_therapist, msgs_patient, seed, seed))

    loop = asyncio.get_running_loop()
    stop_strings = list(cfg.stop_strings) if cfg.stop_strings else None

    # The candidate was a therapist turn, so the patient speaks first: even step = patient.
    for step in range(k):
        active = [s for s in sims if s.active]
        if not active:
            break

        if step % 2 == 0:
            responses = await generate_patient_batch(
                client,
                cfg.patient_binding,
                [s.msgs_patient for s in active],
                primitives.patient_sem(),
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature_patient,
            )
            for sim, resp in zip(active, responses):
                if isinstance(resp, BaseException) or not resp:
                    sim.freeze("patient_error")
                    continue
                content, ended = _split_session_end(resp)
                if ended:
                    if content:
                        sim.append("patient", content)
                    sim.freeze("session_ended")
                    continue
                sim.msgs_therapist.append({"role": "user", "content": content})
                sim.msgs_patient.append({"role": "assistant", "content": content})
                sim.append("patient", content)
            continue

        # Therapist turn: one padded batched generate, under the GPU lock, eval/use_cache
        # toggled and restored. Nothing is awaited inside the lock except the executor
        # handoff that IS the generate.
        start_sb = state.sub_batch if state.sub_batch is not None else len(active)
        async with primitives.gpu_lock():
            was_training = bool(getattr(model, "training", False))
            old_use_cache = model.config.use_cache
            model.config.use_cache = True
            model.eval()
            try:
                responses, final_sb, n_calls, n_oom, n_runtime = await loop.run_in_executor(
                    None,
                    functools.partial(
                        _therapist_generate_chunked,
                        model,
                        tokenizer,
                        [s.msgs_therapist for s in active],
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature_therapist,
                        max_input_tokens=cfg.max_input_tokens,
                        stop_strings=stop_strings,
                        start_sub_batch=start_sb,
                    ),
                )
            finally:
                model.config.use_cache = old_use_cache
                model.train(was_training)

        state.gpu_calls += n_calls
        state.oom_events += n_oom
        state.runtime_errors += n_runtime
        # Only persist a REDUCED cap. Storing `len(active)` when nothing went wrong would
        # pin every later call to the size of the first one, which shrinks as sims freeze.
        if final_sb < start_sb:
            state.sub_batch = final_sb

        for sim, resp in zip(active, responses):
            if resp is None:
                # OOM at sub-batch 1, or a non-OOM generate failure for this chunk.
                sim.freeze("gpu_error")
                continue
            if not resp.strip():
                # Cleaned to nothing: the policy emitted only a ChatML marker (self-play).
                # Scoring an empty turn asks the oracle to grade something that is not there.
                sim.freeze("degenerate")
                continue
            content, ended = _split_session_end(resp)
            if ended:
                if content:
                    sim.append("therapist", content)
                sim.freeze("session_ended")
                continue
            sim.msgs_therapist.append({"role": "assistant", "content": content})
            sim.msgs_patient.append({"role": "user", "content": content})
            sim.append("therapist", content)

    return [s.to_result(k) for s in sims]


# =============================================================================
# FORMAT DRIFT GUARD
# =============================================================================


def check_transcript_format_agreement() -> Tuple[bool, str]:
    """Verify this module's transcript grammar still matches ``core.conversations``.

    The tail slice and the EDA's reconstruction both assume the labels and the ``"\\n\\n"``
    joiner used by :func:`seed_transcript` are byte-identical to what
    ``format_conversation_for_oracle`` writes. If they diverge, nothing raises: the seed
    simply stops being a prefix of the extension, ``tail`` silently becomes the entire
    extended transcript, and every recorded look-ahead tail is wrong while every score stays
    plausible. This turns that into a loud, free, offline check.

    Returns:
        ``(ok, detail)``. ``detail`` shows both renderings when they disagree.

    Notes:
        Returns a flag rather than raising so ``tools/smoke.py`` can report it alongside the
        other Phase 0 checks. Called with no arguments and no I/O -- safe anywhere.
    """
    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "assistant", "content": "How have things been since we last spoke?"},
        {"role": "user", "content": "Rough.\n\nI stopped going to the gym."},
    ]
    expected = (
        f"{THERAPIST_LABEL} {messages[1]['content']}"
        f"{TURN_JOINER}{PATIENT_LABEL} {messages[2]['content']}"
    )
    actual = format_conversation_for_oracle(messages)
    if actual == expected:
        return True, "transcript grammar agrees with core.conversations"
    return False, (
        "transcript grammar DRIFTED -- look-ahead tail slicing is now silently wrong.\n"
        f"  core.conversations: {actual!r}\n"
        f"  core.lookahead:     {expected!r}"
    )
