"""conversations.py -- the conversation: its state, its file, its text, its prompts.

A "conversation" is the unit everything in Exp4 is built out of. It is what the policy
generates, what the oracle grades, what the eval scores are computed over, and what the
per-turn training prompts are sliced from. Four concerns are kept together here because
they are one object seen from four sides, and splitting them across files means the four
sides drift:

- **State + persistence** -- :class:`ConversationState`, the ``pers<PID>.csv`` round trip.
- **Generation** -- the patient call (async, over an OpenAI-compatible socket), the
  therapist call (local GPU, via :mod:`core.policy`), and the lock-step batched loop that
  alternates them until someone says SESSION ENDED.
- **Text formats** -- the ``[THERAPIST]:`` / ``[PATIENT]:`` transcript and its exact
  inverse. This is the interchange format between the conversation, the oracle and the
  look-ahead simulator.
- **Prompt extraction** -- turning finished conversations into per-turn training samples
  under the MCL filter and a token budget.

Two things in here are load-bearing in ways that are not obvious from the code.

**The persona id is the file name (Exp3 fix #2).** Exp3 wrote
``conversation_{permutation_index}.csv`` where ``permutation_index`` was the index into the
per-iteration SHUFFLED subset -- so ``conversation_3.csv`` was a different patient in every
iteration, and every EDA module that wanted to pair a conversation across iterations had to
re-derive ``Random(seed + k + 1)`` and replay the shuffle. In Exp4
:attr:`ConversationState.persona_id` is the STABLE index into
``system_prompts_builder.generate_all_permutations()`` order (0..95), the file is
``pers07.csv`` for persona 7 in every iteration forever, and ``persona_id`` is also a CSV
column. The per-iteration shuffle still exists -- it decides WHICH personas run when a
subset is used, and in what order they are processed -- but it never reaches a filename.
**If you came from Exp3, this is the thing that changed.** Pair on ``persona_id``; never on
a file's position in a directory listing.

**The transcript format is a wire protocol, not a pretty-printer.** The look-ahead simulator
reconstructs message lists from a transcript string and then recovers the piece it appended
by EXACT string slicing::

    seed = f"{transcript}\\n\\n[THERAPIST]: {completion}"
    tail = extended[len(seed):]

so changing the labels, the ``": "`` separator or the ``"\\n\\n"`` joiner does not break
anything loudly -- it silently produces empty or misaligned tails, and the EDA's
look-ahead analysis quietly becomes wrong. :func:`format_conversation_for_oracle` and
:func:`parse_transcript_to_messages` are exact inverses and must stay that way.

Heavy imports are lazy on purpose: the read-only EDA imports this module to read a
conversation CSV and to reason about transcripts, and it must not pay for torch to do it.
``core.policy`` (torch) and ``pandas`` are imported inside the functions that need them.
"""

from __future__ import annotations

import asyncio
import gc
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.concurrency import AsyncPrimitives, run_async
from roles import RoleBinding

# The provider SDK's status-error class, for the 4xx short-circuit in generate_patient_response.
# Optional at import so the read-only EDA (which never calls a patient) can load this module on
# a host without the SDK; without it every failure simply stays retryable.
try:
    from openai import APIStatusError as _APIStatusError
except ImportError:  # pragma: no cover - EDA host without the openai SDK
    _APIStatusError = None  # type: ignore[assignment,misc]

__all__ = [
    # Constants
    "SESSION_END_KEYWORD",
    "THERAPIST_LABEL",
    "PATIENT_LABEL",
    "TRANSCRIPT_JOINER",
    "CONV_FILE_PREFIX",
    "CONV_CSV_COLUMNS",
    # State + persistence
    "ConversationState",
    "conversation_filename",
    "conversation_id_for",
    "save_conversation_csv",
    "load_conversation_csv",
    "load_conversations_dir",
    "new_conversation_state",
    # Text formats
    "format_conversation_for_oracle",
    "parse_transcript_to_messages",
    "turns_to_messages",
    "turns_to_patient_messages",
    "has_session_end",
    "handle_session_end",
    # Generation
    "generate_patient_response",
    "generate_patient_batch",
    "conversation_loop_batch",
    "generate_all_conversations",
    "generate_all_conversations_async",
    # Prompt extraction
    "build_truncated_training_prompt",
    "extract_prompts_from_conversations",
]


# ==============================================================================
#  CONSTANTS -- the protocol
# ==============================================================================

#: The literal both system prompts instruct their model to emit when the session is over.
#:
#: WARNING: this is the ONLY early-termination channel. Neither speaker has a stop token for
#: "we are done" -- the therapist and the patient are told, in prose, to write these two words.
#: Exp4 swaps the patient from ``gpt-4o-mini`` to Gemma, and an open model that is not steered
#: to the same protocol simply never emits it: conversations then always run to the
#: ``num_utterances`` cap, session-end statistics go to zero, and nothing raises. Verify at the
#: Phase 2 gate that some fraction of the 96 base conversations end early.
SESSION_END_KEYWORD = "SESSION ENDED"

# The ONE matcher for the keyword: case-insensitive, run on the ORIGINAL text so the match
# offsets index that text. (An earlier revision took the index on ``.upper()`` of the string --
# for some Unicode ``upper()`` changes the length, so the split could land mid-character.)
# Presence is tested with :func:`has_session_end` -- the SAME regex -- everywhere: a presence
# test written as ``KEYWORD in text.upper()`` disagrees with this matcher on Unicode case
# folding (``"seßion ended".upper()`` is ``"SESSION ENDED"``, which the regex does not match),
# and :func:`handle_session_end` then RAISES on text the caller had just declared terminal.
_SESSION_END_RE = re.compile(re.escape(SESSION_END_KEYWORD), re.IGNORECASE)

#: Transcript labels and joiner. See the module docstring: look-ahead slices on these EXACT
#: strings, so they are constants rather than literals scattered through format/parse.
THERAPIST_LABEL = "THERAPIST"
PATIENT_LABEL = "PATIENT"
TRANSCRIPT_JOINER = "\n\n"

#: ``pers07.csv`` -- two digits because the persona space is 0..95. A wider persona set needs a
#: width bump here AND a re-read of every folder already on disk, so it is a constant.
CONV_FILE_PREFIX = "pers"
_CONV_FILE_WIDTH = 2
_CONV_FILE_RE = re.compile(r"^" + CONV_FILE_PREFIX + r"(?P<pid>\d+)\.csv$")

#: One row per utterance. ``session_ended_by`` / ``session_ended_explanation`` are conversation
#: -level scalars broadcast to every row -- denormalized on purpose, so a single row of the CSV
#: is self-describing and no reader needs a sidecar to know how the conversation finished.
CONV_CSV_COLUMNS = (
    "persona_id",
    "role",
    "conversation",
    "session_ended_by",
    "session_ended_explanation",
)

_ROLE_THERAPIST = "therapist"
_ROLE_PATIENT = "patient"
_NEXT_SPEAKER = {_ROLE_THERAPIST: _ROLE_PATIENT, _ROLE_PATIENT: _ROLE_THERAPIST}

# Transcript-label matcher. DOTALL so a multi-line turn is captured whole by group(2).
_TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(" + PATIENT_LABEL + r"|" + THERAPIST_LABEL + r")\]:\s*(.*)$", re.DOTALL
)

# Both role conventions map onto the two transcript labels. See the warning on
# format_conversation_for_oracle: this table is only valid for THERAPIST-perspective messages.
_ROLE_TO_LABEL = {
    "assistant": THERAPIST_LABEL,
    _ROLE_THERAPIST: THERAPIST_LABEL,
    "user": PATIENT_LABEL,
    _ROLE_PATIENT: PATIENT_LABEL,
}


# NOTE: this module renders NO chat template of its own. Every prompt string comes from
# ``core.policy.render_prompt`` / ``build_prompt`` (imported lazily -- policy is torch-side),
# which pin ``date_string=CHAT_TEMPLATE_DATE`` and strip the BOS. That is what keeps the training
# prompts byte-identical to the decode path's, and it is why there is no date helper here.


# ==============================================================================
#  STATE
# ==============================================================================


@dataclass
class ConversationState:
    """One conversation, mid-flight or finished.

    Attributes:
        persona_id: **Stable** index into ``generate_all_permutations()`` order (0..95). Not a
            batch position, not a shuffled processing index -- the same integer identifies the
            same patient in every iteration of every arm. It names the file and is a CSV column.
        turns: ``[{"role": "therapist"|"patient", "content": str}, ...]`` in order. The single
            source of truth for the conversation's content; the two message lists are views.
        messages_therapist: Therapist-perspective chat history (therapist=assistant,
            patient=user), ready for the ChatML template.
        messages_patient: Patient-perspective chat history (patient=assistant,
            therapist=user), ready for the patient API.
        session_ended_by: ``""`` while running or when the conversation hit the utterance cap;
            otherwise the role that emitted :data:`SESSION_END_KEYWORD`.
        session_ended_explanation: Whatever the model wrote after the keyword. ``""`` if none.
        failed: The conversation is unusable -- an API call exhausted its retries, or therapist
            generation died. Failed states are dropped, never saved.
        active: Runtime only. False once the conversation has ended for any reason.
        next_speaker: Runtime only. Who speaks next; the batched loop advances all active
            conversations in lock-step and uses this to detect desync.

    Notes:
        The two message lists are cleared by :meth:`release_messages` once a conversation ends.
        They are the memory-heavy part (every turn is stored three times) and nothing downstream
        needs them -- prompt extraction rebuilds what it wants from ``turns``.

        ``session_ended_*`` are ``""``-not-``None`` so they survive a CSV round trip as strings
        under ``keep_default_na=False``. Exp3 used ``None`` and every reader had to handle both
        ``None`` and the string ``"nan"``.
    """

    persona_id: int
    turns: List[Dict[str, str]] = field(default_factory=list)
    messages_therapist: List[Dict[str, str]] = field(default_factory=list)
    messages_patient: List[Dict[str, str]] = field(default_factory=list)
    session_ended_by: str = ""
    session_ended_explanation: str = ""
    failed: bool = False
    active: bool = True
    next_speaker: str = _ROLE_PATIENT

    # -- derived views ---------------------------------------------------------

    @property
    def n_utterances(self) -> int:
        """Total utterances, therapist + patient combined -- the MCL unit."""
        return len(self.turns)

    @property
    def utterances(self) -> List[str]:
        """Just the text, in order."""
        return [t["content"] for t in self.turns]

    @property
    def conversation_id(self) -> str:
        """Stable string id (``"pers07"``) -- the CSV stem, so it points at the artifact."""
        return conversation_id_for(self.persona_id)

    # -- mutation --------------------------------------------------------------

    def append_turn(self, role: str, content: str) -> None:
        """Append one utterance to ``turns`` and to BOTH message views.

        Keeping the three in one method is what stops them from drifting: a turn appended to
        ``turns`` but not to ``messages_patient`` produces a patient that never sees what the
        therapist just said, which looks like a model failure rather than a bookkeeping bug.
        """
        self.turns.append({"role": role, "content": content})
        if role == _ROLE_THERAPIST:
            self.messages_therapist.append({"role": "assistant", "content": content})
            self.messages_patient.append({"role": "user", "content": content})
        else:
            self.messages_therapist.append({"role": "user", "content": content})
            self.messages_patient.append({"role": "assistant", "content": content})

    def release_messages(self) -> None:
        """Drop both message lists (CPU memory only). ``turns`` is untouched."""
        self.messages_therapist = []
        self.messages_patient = []


def conversation_filename(persona_id: int) -> str:
    """``pers07.csv`` for persona 7. The one place the file-name grammar is spelled."""
    return f"{CONV_FILE_PREFIX}{int(persona_id):0{_CONV_FILE_WIDTH}d}.csv"


def conversation_id_for(persona_id: int) -> str:
    """``pers07`` -- the CSV stem, used as ``conversation_id`` in training samples and EDA rows."""
    return f"{CONV_FILE_PREFIX}{int(persona_id):0{_CONV_FILE_WIDTH}d}"


def new_conversation_state(
    persona_id: int,
    *,
    therapist_system_prompt: str,
    patient_system_prompt: str,
    therapist_init_utterance: str,
) -> ConversationState:
    """Seed a conversation with the therapist's scripted opening line.

    The therapist always speaks first and its first utterance is scripted (it comes from the
    persona permutation), so no model is called to produce it and the patient's first real
    generation already has something to react to.

    Notes:
        Exp3 carried an ``include_empty_init_user_message`` flag to satisfy chat templates that
        demand system -> user -> assistant ordering. Exp4's hand-written ChatML template
        (``core.policy.CHATML_TEMPLATE``) renders any role in any order, so the flag -- and the
        empty user message it inserted, which the oracle transcript then had to drop -- is gone.
    """
    state = ConversationState(
        persona_id=int(persona_id),
        messages_therapist=[{"role": "system", "content": str(therapist_system_prompt)}],
        messages_patient=[{"role": "system", "content": str(patient_system_prompt)}],
        next_speaker=_ROLE_PATIENT,
    )
    state.append_turn(_ROLE_THERAPIST, therapist_init_utterance)
    return state


# ==============================================================================
#  PERSISTENCE
# ==============================================================================


def _extract_unique_col(df, col_name: str) -> str:
    """The single distinct value of a broadcast scalar column, or ``""``.

    ``session_ended_by`` / ``session_ended_explanation`` are conversation-level scalars written
    to every row. Reading them back means "take the one value", not "take row 0" -- which also
    means a file whose rows disagree returns the first non-empty one rather than raising, since
    losing a whole conversation over an inconsistent metadata column is the worse failure.
    """
    if col_name not in df.columns:
        return ""
    for value in df[col_name].tolist():
        text = "" if value is None else str(value)
        if text:
            return text
    return ""


def save_conversation_csv(state: ConversationState, save_dir: str) -> str:
    """Write one conversation to ``<save_dir>/pers<PID>.csv``. Returns the path.

    Args:
        state: The conversation to write. ``persona_id`` decides the file name.
        save_dir: Created if missing.

    Notes:
        The file name is derived from the STABLE persona id, never from a batch position or a
        shuffled processing index (Exp3 fix #2 -- see the module docstring). ``persona_id`` is
        also written as a column, so a file that gets renamed or copied is still self-describing.

        Writing is atomic: the frame goes to a temp file in the same directory, then
        ``os.replace`` onto the final name. "A half-written CSV fails to parse on the next
        resume" is FALSE for a truncation that lands on a row boundary -- pandas parses it as a
        valid, shorter conversation, the resume path marks the persona complete, and the
        truncated transcript permanently enters training extraction and the eval set. Atomic
        replace makes the file either absent (regenerated, cost one conversation) or whole.
    """
    os.makedirs(save_dir, exist_ok=True)
    import pandas as pd  # lazy: the transcript helpers must import without pandas present

    n = len(state.turns)
    path = os.path.join(save_dir, conversation_filename(state.persona_id))
    frame = pd.DataFrame({
        "persona_id": [int(state.persona_id)] * n,
        "role": [t["role"] for t in state.turns],
        "conversation": [t["content"] for t in state.turns],
        "session_ended_by": [state.session_ended_by or ""] * n,
        "session_ended_explanation": [state.session_ended_explanation or ""] * n,
    }, columns=list(CONV_CSV_COLUMNS))
    tmp_path = path + ".tmp"
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    return path


def load_conversation_csv(path: str) -> ConversationState:
    """Read a ``pers<PID>.csv`` back into a finished :class:`ConversationState`.

    Args:
        path: Full path to the CSV.

    Returns:
        A state with ``turns`` populated, ``active=False`` and both message lists empty --
        message histories are not persisted and are not needed downstream.

    Raises:
        ValueError: if ``persona_id`` can be recovered neither from the column nor from the file
            name. An unidentifiable conversation is worse than a missing one: it would be
            silently paired against the wrong patient.

    Notes:
        ``keep_default_na=False`` + ``dtype=str`` are required, not stylistic. Without them
        pandas turns an empty ``session_ended_by`` into ``NaN`` (a float) and infers a numeric
        dtype for a column of digit-only utterances, and every consumer then has to handle
        ``None``, ``float('nan')``, the string ``"nan"`` and non-string content. With them every
        cell is a string and "" stays "" (any residual NaN is normalised to "" as well).
    """
    import pandas as pd  # lazy -- see save_conversation_csv

    df = pd.read_csv(path, keep_default_na=False, dtype=str).fillna("")

    persona_id: Optional[int] = None
    if "persona_id" in df.columns and len(df) > 0:
        raw = _extract_unique_col(df, "persona_id")
        if raw:
            try:
                persona_id = int(float(raw))
            except ValueError:
                persona_id = None
    if persona_id is None:
        match = _CONV_FILE_RE.match(os.path.basename(path))
        if match is None:
            raise ValueError(
                f"load_conversation_csv: cannot determine persona_id for {path!r} -- no usable "
                f"'persona_id' column and the file name is not '{CONV_FILE_PREFIX}<N>.csv'. "
                f"A conversation with no stable persona id cannot be paired across iterations."
            )
        persona_id = int(match.group("pid"))

    contents = [str(c) for c in df["conversation"].tolist()]
    if "role" in df.columns:
        roles = [str(r) for r in df["role"].tolist()]
    else:
        # Pre-``role``-column fallback: the therapist always speaks first and speakers alternate.
        roles = [_ROLE_THERAPIST if j % 2 == 0 else _ROLE_PATIENT for j in range(len(contents))]

    return ConversationState(
        persona_id=persona_id,
        turns=[{"role": r, "content": c} for r, c in zip(roles, contents)],
        session_ended_by=_extract_unique_col(df, "session_ended_by"),
        session_ended_explanation=_extract_unique_col(df, "session_ended_explanation"),
        active=False,
    )


def load_conversations_dir(save_dir: str, *, verbose: bool = False) -> Dict[int, ConversationState]:
    """Load every ``pers*.csv`` in *save_dir*. Returns ``{persona_id: state}``.

    A missing directory yields ``{}`` -- "this model state has not been generated yet" is a
    normal state, not an error. An individual unreadable file warns and is skipped, so one
    corrupt CSV costs one conversation rather than the whole resume.

    Notes:
        This is also the read side the EDA uses. WARNING: on the Colab/Drive symlink, "the directory
        reads as empty" is NOT proof the conversations are missing -- the mount can wedge on a
        single folder and report zero files while every conversation is present in Drive. Check
        the cloud before concluding an arm is unfinished.
    """
    states: Dict[int, ConversationState] = {}
    if not os.path.isdir(save_dir):
        return states
    for name in sorted(os.listdir(save_dir)):
        if _CONV_FILE_RE.match(name) is None:
            continue
        path = os.path.join(save_dir, name)
        try:
            state = load_conversation_csv(path)
        except Exception as exc:
            print(f"  Warning: could not load {name}: {exc}")
            continue
        states[state.persona_id] = state
    if states and verbose:
        print(f"  Resumed {len(states)} conversations from {save_dir}")
    return states


# ==============================================================================
#  TEXT FORMATS -- the wire protocol between conversation, oracle and look-ahead
# ==============================================================================


def format_conversation_for_oracle(messages_or_turns: Sequence[Dict[str, str]]) -> str:
    """Render a conversation as the labelled plain-text transcript the oracle grades.

    Args:
        messages_or_turns: Either therapist-perspective message dicts (``system`` / ``user`` =
            patient / ``assistant`` = therapist) or role-tagged turns (``therapist`` /
            ``patient``). ``system`` entries are dropped; empty content is skipped.

    Returns:
        ``"[THERAPIST]: ...\\n\\n[PATIENT]: ...\\n\\n[THERAPIST]: ..."``.

    Notes:
        **WARNING -- therapist-perspective only.** In ``messages_patient`` the roles are FLIPPED
        (assistant = patient), so passing that list produces a transcript with every speaker
        mislabelled -- which parses fine, grades fine, and is wrong. Pass ``state.turns`` when
        in any doubt; turns carry their own speaker names and cannot be misread.

        **WARNING -- this is an exact-slicing protocol, not formatting.** Look-ahead builds
        ``f"{transcript}\\n\\n[THERAPIST]: {completion}"`` and recovers its own tail by
        ``extended[len(seed):]``. Change the labels, the ``": "`` or the ``"\\n\\n"`` joiner and
        the tails come back empty or offset, with nothing raising.
        :func:`parse_transcript_to_messages` is the exact inverse and must be changed with it.
    """
    parts: List[str] = []
    for msg in messages_or_turns:
        label = _ROLE_TO_LABEL.get(str(msg.get("role", "")))
        if label is None:  # 'system' and anything unrecognised
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"[{label}]: {content}")
    return TRANSCRIPT_JOINER.join(parts)


def parse_transcript_to_messages(
    transcript: str,
    system_prompt_therapist: str,
    system_prompt_patient: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Exact inverse of :func:`format_conversation_for_oracle`.

    Args:
        transcript: A labelled transcript.
        system_prompt_therapist: Prepended as the therapist list's system message.
        system_prompt_patient: Prepended as the patient list's system message.

    Returns:
        ``(messages_therapist, messages_patient)`` -- the first with therapist=assistant /
        patient=user, the second with the roles flipped, both ready to hand straight to their
        respective generator.

    Raises:
        ValueError: if the transcript opens with an unlabelled segment. There is no previous turn
            for it to continue, so the only alternatives are guessing a speaker or dropping text
            the oracle already read -- both silently corrupt the look-ahead context.

    Notes:
        **An unlabelled segment is a CONTINUATION of the previous labelled turn.** A turn's own
        content routinely contains a blank line, and the transcript joins turns with the same
        ``"\\n\\n"``, so splitting on the joiner over-segments. Reattaching unlabelled fragments
        to the turn above is what makes the round trip exact; without it a multi-paragraph
        therapist turn re-enters the model as several turns and the speaker alternation desyncs
        from there on.

        **Known limitation -- a label INSIDE a turn.** The inverse is exact only while no turn's
        content itself contains a segment that starts with ``[PATIENT]: `` or ``[THERAPIST]: ``
        right after a blank line (a model quoting the transcript format back). Such a segment is
        indistinguishable from a real turn boundary and is parsed as one. Neither speaker is
        prompted with the labels, so it has not been observed; it is documented rather than
        guarded because any escaping scheme would change the transcript the oracle grades.
    """
    messages_therapist: List[Dict[str, str]] = [
        {"role": "system", "content": str(system_prompt_therapist)}
    ]
    messages_patient: List[Dict[str, str]] = [
        {"role": "system", "content": str(system_prompt_patient)}
    ]

    label: Optional[str] = None
    fragments: List[str] = []

    def _flush() -> None:
        nonlocal label, fragments
        if label is None:
            return
        content = TRANSCRIPT_JOINER.join(f for f in fragments if f).strip()
        if content:
            if label == THERAPIST_LABEL:
                messages_therapist.append({"role": "assistant", "content": content})
                messages_patient.append({"role": "user", "content": content})
            else:
                messages_therapist.append({"role": "user", "content": content})
                messages_patient.append({"role": "assistant", "content": content})
        label = None
        fragments = []

    for segment in (s.strip() for s in str(transcript).split(TRANSCRIPT_JOINER)):
        if not segment:
            continue
        match = _TRANSCRIPT_LINE_RE.match(segment)
        if match is not None:
            _flush()
            label = match.group(1)
            fragments = [match.group(2).strip()]
            continue
        if label is None:
            raise ValueError(
                f"parse_transcript_to_messages: transcript segment has no preceding role label: "
                f"{segment[:120]!r}"
            )
        fragments.append(segment)

    _flush()
    return messages_therapist, messages_patient


def turns_to_messages(turns: Sequence[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
    """Therapist-perspective messages: therapist -> assistant, patient -> user."""
    messages = [{"role": "system", "content": str(system_prompt)}]
    for turn in turns:
        role = "assistant" if turn["role"] == _ROLE_THERAPIST else "user"
        messages.append({"role": role, "content": str(turn["content"])})
    return messages


def turns_to_patient_messages(
    turns: Sequence[Dict[str, str]], system_prompt: str
) -> List[Dict[str, str]]:
    """Patient-perspective messages: patient -> assistant, therapist -> user.

    Mirror of :func:`turns_to_messages`. This is the shape the patient API expects and the shape
    :func:`new_conversation_state` builds, so a conversation reconstructed from ``turns`` can be
    resumed against the patient exactly where it left off.
    """
    messages = [{"role": "system", "content": str(system_prompt)}]
    for turn in turns:
        role = "assistant" if turn["role"] == _ROLE_PATIENT else "user"
        messages.append({"role": role, "content": str(turn["content"])})
    return messages


def has_session_end(text: str) -> bool:
    """Does *text* contain :data:`SESSION_END_KEYWORD`, by the ONE matcher the split uses?

    This is the presence test every caller must use before :func:`handle_session_end` -- the
    conversation loop, the look-ahead simulator (``core.lookahead.split_session_end``) and,
    through it, the reward path and PTO's trunk advance -- so "is this utterance terminal" and
    "where does it end" agree by construction.

    Notes:
        The obvious alternative, ``SESSION_END_KEYWORD in text.upper()``, is NOT equivalent:
        ``str.upper`` applies full Unicode case mapping, so ``"seßion ended".upper()`` is
        ``"SESSION ENDED"`` (``ß`` -> ``SS``) and the substring test says True, while the
        ``re.IGNORECASE`` regex -- which matches character by character on the original string --
        says False. A caller that tested presence with ``.upper()`` and then split with the regex
        would raise ``ValueError`` from :func:`handle_session_end` on such a turn; inside a live
        GRPO reward call that takes the whole optimizer step down. ``None`` is treated as absent.
    """
    if not text:
        return False
    return _SESSION_END_RE.search(text) is not None


def handle_session_end(response_content: str, speaker_role: str) -> Tuple[str, str, str]:
    """Split a terminal utterance at :data:`SESSION_END_KEYWORD`.

    Args:
        response_content: The raw utterance, which contains the keyword.
        speaker_role: ``"therapist"`` or ``"patient"`` -- who ended the session.

    Returns:
        ``(ended_by, ended_explanation, cleaned_content)``. ``cleaned_content`` is everything
        BEFORE the keyword and is still a valid utterance worth keeping; ``ended_explanation`` is
        everything after it.

    Raises:
        ValueError: if the keyword is absent -- callers check first with :func:`has_session_end`
            (the same regex), so its absence here means the caller tested presence some other way
            and disagrees with this function about what a terminal utterance is.

    Notes:
        The match is case-insensitive (``re.IGNORECASE`` on the escaped keyword) and is run on the
        ORIGINAL string, so the split offsets index that string and the returned pieces preserve
        the model's own casing and spacing. Splitting on ``.upper()`` would not: for some Unicode
        ``upper()`` changes the string's length and the index lands mid-character (and a presence
        test on ``.upper()`` accepts ``"seßion ended"``, which this matcher does not -- see
        :func:`has_session_end`).

        WARNING: this keyword is the only early-termination channel in the experiment and both system
        prompts ask for it in prose. A patient backend that is not steered to the same protocol
        never ends a session, and every conversation silently runs to the utterance cap.
    """
    match = _SESSION_END_RE.search(response_content)
    if match is None:
        raise ValueError(f"handle_session_end: {SESSION_END_KEYWORD!r} not found in response")
    return (
        speaker_role,
        response_content[match.end():],
        response_content[:match.start()],
    )


def _apply_response(state: ConversationState, content: str, speaker_role: str) -> bool:
    """Fold one generated utterance into *state*. Returns True if the conversation continues.

    Three outcomes: a degenerate (empty after cleaning) utterance ENDS the conversation rather
    than padding it with an empty turn -- the therapist decoder cuts self-played ChatML leaks and
    can legitimately leave nothing usable; a terminal utterance records who ended it and keeps
    any text before the keyword; anything else is appended and the speaker flips.
    """
    if not content or not content.strip():
        state.active = False
        return False

    if has_session_end(content):
        ended_by, explanation, cleaned = handle_session_end(content, speaker_role)
        state.session_ended_by = ended_by
        state.session_ended_explanation = explanation.strip()
        state.active = False
        if cleaned and cleaned.strip():
            state.append_turn(speaker_role, cleaned.strip())
        return False

    state.append_turn(speaker_role, content)
    state.next_speaker = _NEXT_SPEAKER[speaker_role]
    return True


# ==============================================================================
#  PATIENT GENERATION
# ==============================================================================


def _describe_error(exc: Optional[BaseException]) -> str:
    """``"TypeName: message"``, or just the type when the exception carries no message.

    ``asyncio.TimeoutError`` stringifies to ``""``, so the naive f-string renders the single most
    important retry cause as ``"TimeoutError: "`` -- a log line that reads like truncated output
    rather than like the timeout it is.
    """
    if exc is None:
        return "unknown error"
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# 4xx statuses that ARE worth retrying: the request itself was fine, the server was busy.
_RETRYABLE_4XX = frozenset({408, 429})


def _non_retryable_status(exc: BaseException) -> Optional[int]:
    """The HTTP status of *exc* when it is a 4xx the server will reject identically on retry.

    ``None`` for everything else: timeouts, connection errors, 5xx, 408/429, and any exception
    that is not the SDK's ``APIStatusError`` (including its 'no SDK installed' stand-in).
    """
    if _APIStatusError is None or not isinstance(exc, _APIStatusError):
        return None
    status = getattr(exc, "status_code", None)
    if status is None:
        return None
    status = int(status)
    if 400 <= status < 500 and status not in _RETRYABLE_4XX:
        return status
    return None


def _body_excerpt(exc: BaseException, limit: int = 300) -> str:
    """The response body the SDK attached to *exc*, cut to *limit* characters."""
    body = getattr(exc, "body", None)
    text = str(body) if body is not None else ""
    return text[:limit] + ("..." if len(text) > limit else "") if text else "<none>"


async def generate_patient_response(
    client,
    binding: RoleBinding,
    messages: Sequence[Dict[str, str]],
    sem: asyncio.Semaphore,
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int] = None,
    backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 30.0,
) -> str:
    """One patient utterance, with a per-attempt timeout and bounded exponential backoff.

    Args:
        client: An ``AsyncOpenAI`` from ``roles.make_client``.
        binding: Supplies the model, the PER-ATTEMPT ``request_timeout``, ``max_retries`` and
            ``extra_body`` (which is what turns Gemma's thinking mode off).
        messages: Patient-perspective chat history.
        sem: The shared patient semaphore from ``AsyncPrimitives.patient_sem()``.
        max_tokens: Cap on the reply.
        temperature: Sampling temperature.
        seed: Passed through when not None; ignored by servers that do not honour it.
        backoff_seconds: First backoff; doubles per attempt.
        max_backoff_seconds: Cap on that doubling, so attempt 8 does not sleep two minutes.

    Returns:
        The reply text.

    Raises:
        RuntimeError: after ``binding.max_retries`` failed attempts, naming the last error --
            OR immediately, without any retry, on a NON-RETRYABLE HTTP status (see below).

    Notes:
        **4xx short-circuit.** An ``openai.APIStatusError`` with a 4xx status other than 408
        (request timeout) and 429 (rate limit) is a request the server will reject the same way
        eight times: a wrong model id (404), a prompt over ``max_model_len`` (400), a schema the
        server cannot honour (422), a bad key (401). Retrying those with backoff costs up to
        ``max_retries x request_timeout`` PER TURN PER CONVERSATION before anything surfaces, so
        they raise at once with the status and a body excerpt. Timeouts, 408, 429, 5xx and
        connection errors stay retryable. (Only the OpenAI SDK's class is recognised -- this
        function speaks ``chat.completions``, so that is the only client that reaches it.)

        **An EMPTY reply is a failure, not an utterance.** ``""`` / whitespace-only content is
        retried like ``None`` and, if it persists, raises like any other exhausted call. Before
        this, an empty patient reply ended the conversation as if it had reached the utterance
        cap and the truncated conversation was SAVED -- a patient-infrastructure fault recorded
        as a finished session. (The therapist's empty completion still ends a conversation
        gracefully: that is policy behaviour, which the run is allowed to observe.)
        **Why a short per-attempt timeout times MANY retries, and never a long total budget**
        (this is Exp3 fix #1 -- Exp3's patient call had no timeout at all, so it inherited the
        SDK's 600 s default). Exhausting the budget does not merely lose one utterance: the
        simulation FREEZES at that turn, the oracle then grades a truncated transcript as if it
        were a finished one, and under GRPO's ``scale_rewards="group"`` a single frozen sim
        shifts both the mean AND the std of its group of 8. The damage is not confined to the
        sample that failed -- it re-scales the advantages of seven healthy siblings. A short
        attempt bound with many retries drives the freeze probability below what a long budget
        gives you, at the same worst-case wall clock.

        **The backoff sleeps OUTSIDE the semaphore.** The ``async with`` closes before the sleep,
        so a retrying call gives its concurrency slot back instead of holding the server's queue
        idle for the length of its own backoff.

        ``asyncio.wait_for`` duplicates the timeout the SDK client was built with. That is
        deliberate: the SDK bound covers a hung socket, ``wait_for`` covers a hung coroutine
        (a stalled generator, a retry loop inside a middleware), and only both together bound
        the attempt.

        A ``None`` message content is treated as a retryable failure. The most likely cause is a
        model that answered with a reasoning block and no content -- i.e. the thinking-off
        ``extra_body`` key did not take. The final error says so, because that failure is
        otherwise invisible.
    """
    kwargs: Dict[str, Any] = {
        "model": binding.model,
        "messages": list(messages),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if seed is not None:
        kwargs["seed"] = seed
    extra_body = binding.extra_body
    if extra_body:
        kwargs["extra_body"] = extra_body

    attempts = max(1, int(binding.max_retries))
    last_error: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        try:
            async with sem:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=float(binding.request_timeout),
                )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("patient returned message.content=None")
            if not content.strip():
                raise RuntimeError("patient returned an EMPTY message.content")
            return content
        except asyncio.CancelledError:
            # Cancellation is not a provider failure: retrying it would ignore a shutdown.
            raise
        except Exception as exc:  # noqa: BLE001 -- classified below; most are retryable
            status = _non_retryable_status(exc)
            if status is not None:
                raise RuntimeError(
                    f"Patient call REJECTED with HTTP {status} by {binding.provider}:"
                    f"{binding.model} on attempt {attempt} -- not retried (a 4xx other than "
                    f"408/429 fails the same way every time). {_describe_error(exc)}. "
                    f"Body: {_body_excerpt(exc)}"
                ) from exc
            last_error = exc
            if attempt >= attempts:
                break
            sleep_s = min(backoff_seconds * (2 ** (attempt - 1)), max_backoff_seconds)
            print(
                f"  Patient call attempt {attempt}/{attempts} failed "
                f"({_describe_error(exc)}); retrying in {sleep_s:.1f}s"
            )
            await asyncio.sleep(sleep_s)  # outside the semaphore -- slot released

    raise RuntimeError(
        f"Patient call failed after {attempts} attempts against {binding.provider}:"
        f"{binding.model} (per-attempt timeout {binding.request_timeout}s). "
        f"Last error: {_describe_error(last_error)}. "
        f"If this is 'message.content=None', the thinking-off extra_body key is probably not "
        f"taking effect for this model -- check roles.thinking_off_extra_body()."
    )


async def generate_patient_batch(
    client,
    binding: RoleBinding,
    batch_messages: Sequence[Sequence[Dict[str, str]]],
    sem: asyncio.Semaphore,
    **kw,
) -> List[Any]:
    """Fan out :func:`generate_patient_response` over a batch.

    Returns:
        One entry per input, in order: the reply string, or the ``BaseException`` that call ended
        with. ``return_exceptions=True`` is the point -- one patient that exhausted its retries
        must fail ONE conversation, not abort the whole batch and lose the 95 healthy ones.
        The caller is responsible for checking each entry's type.
    """
    return await asyncio.gather(
        *(
            generate_patient_response(client, binding, messages, sem, **kw)
            for messages in batch_messages
        ),
        return_exceptions=True,
    )


# ==============================================================================
#  THE CONVERSATION LOOP
# ==============================================================================


async def conversation_loop_batch(
    states: List[ConversationState],
    model,
    tokenizer,
    client,
    patient_binding: RoleBinding,
    primitives: AsyncPrimitives,
    *,
    num_utterances: int = 49,
    max_tokens: int = 200,
    temperature_therapist: float = 0.9,
    temperature_patient: float = 0.7,
    therapist_max_input_tokens: int = 2048,
    stop_strings: Optional[Sequence[str]] = None,
    patient_seed: Optional[int] = None,
    verbose_detailed: bool = False,
) -> Tuple[List[ConversationState], Optional[str], List[int]]:
    """Advance a batch of conversations in lock-step until they end or hit the cap.

    Every step, all still-active conversations take the same speaker's turn together: one padded
    GPU batch for the therapist, one ``asyncio.gather`` for the patient. That is what amortizes
    the patient round-trip across the whole batch -- which is why a big batch wins on an A100
    even though the loop is GPU-bound overall.

    Args:
        states: Seeded states from :func:`new_conversation_state`. Mutated in place.
        model, tokenizer: The therapist policy and its tokenizer (patched -- see
            ``core.policy.patch_generate``).
        client: Patient client.
        patient_binding: Patient role binding (model, timeout, retries, extra_body).
        primitives: Supplies ``patient_sem()`` and ``gpu_lock()``.
        num_utterances: Hard cap on ADDITIONAL utterances generated AFTER the scripted therapist
            opener that :func:`new_conversation_state` seeds -- one loop step is one utterance,
            and the seed already holds one. A conversation therefore has **at most
            ``num_utterances + 1`` utterances in total: 50 at the default 49** (the value cell 1
            passes as ``NUM_UTTERANCES_FOR_DATA``). The loop also stops when nothing is active.
        therapist_max_input_tokens: Therapist prompt budget, BOS included; see
            ``core.policy.generate_therapist_batch`` -- over-budget conversations drop their
            OLDEST turns whole (system message kept), never token-truncate.
        stop_strings: Defaults to ``core.policy.STOP_STRINGS``.

    Returns:
        ``(states, error_type, desynced_persona_ids)``. ``error_type`` is ``None`` on a clean
        pass, else ``"oom"`` / ``"runtime_error"`` from therapist generation -- in which case
        every still-active state in the batch is marked ``failed`` and the batch is retried by
        the caller at a smaller size.

    Notes:
        **The GPU lock is held across ``generate`` and nothing else.** The patient round is
        awaited with the lock released, which is the invariant ``core.concurrency`` documents.

        **A therapist prompt that cannot be built fails THAT conversation.**
        ``generate_therapist_batch`` returns ``None`` (not ``""``) for an item whose newest turn
        alone exceeds ``therapist_max_input_tokens``; such a state is marked ``failed`` -- it is
        a budget misconfiguration, not policy behaviour, and saving it would record a
        conversation the policy never actually continued. Truncation totals are accumulated in
        ``core.policy.TRUNCATION_COUNTER``; the caller prints the per-batch delta.

        **Desync ends conversations gracefully, with ``failed=False``.** If one conversation's
        ``next_speaker`` disagrees with the batch's, it is retired where it stands and its turns
        so far are still saved and still yield training prompts. Marking it failed would throw
        away a perfectly good partial conversation over a bookkeeping disagreement; aborting the
        batch would throw away the other 95.

        **A per-conversation patient failure sets ``failed=True``.** Unlike a desync, an
        exhausted retry budget means the conversation is missing an utterance it should have had,
        and the caller must regenerate it rather than train on it.
    """
    from core.policy import generate_therapist_batch

    patient_sem = primitives.patient_sem()
    gpu_lock = primitives.gpu_lock()
    desynced: List[int] = []

    for turn_num in range(num_utterances):
        active = [s for s in states if s.active]
        if not active:
            break

        speaker = active[0].next_speaker
        off_beat = [s for s in active if s.next_speaker != speaker]
        if off_beat:
            print(
                f"  Warning: speaker desync at turn {turn_num} (expected {speaker!r}); "
                f"retiring {[(s.persona_id, s.next_speaker) for s in off_beat]} gracefully -- "
                f"their turns so far are kept."
            )
            for s in off_beat:
                s.active = False  # NOT failed: partial turns remain valid training data
                desynced.append(s.persona_id)
            active = [s for s in active if s.active]
            if not active:
                break

        if speaker == _ROLE_PATIENT:
            responses = await generate_patient_batch(
                client,
                patient_binding,
                [s.messages_patient for s in active],
                patient_sem,
                max_tokens=max_tokens,
                temperature=temperature_patient,
                seed=patient_seed,
            )
            for state, response in zip(active, responses):
                if isinstance(response, BaseException):
                    print(f"  Patient call failed for persona {state.persona_id}: {response}")
                    state.active = False
                    state.failed = True
                    continue
                _apply_response(state, response, _ROLE_PATIENT)
        else:
            async with gpu_lock:  # held across generate() only -- never across an await on I/O
                responses, error_type = generate_therapist_batch(
                    model,
                    tokenizer,
                    [s.messages_therapist for s in active],
                    max_tokens=max_tokens,
                    temperature=temperature_therapist,
                    max_input_tokens=therapist_max_input_tokens,
                    stop_strings=stop_strings,
                )
            if responses is None:
                for state in active:
                    state.active = False
                    state.failed = True
                return states, (error_type or "therapist_generation_failed"), desynced
            for state, response in zip(active, responses):
                if response is None:
                    # No prompt could be built within the budget (see the Notes). Not a
                    # degenerate turn: nothing was generated, so the conversation is unusable.
                    print(
                        f"  Therapist prompt overflow for persona {state.persona_id} at "
                        f"{state.n_utterances} utterances (budget {therapist_max_input_tokens} "
                        f"tokens) -- marking the conversation failed"
                    )
                    state.active = False
                    state.failed = True
                    continue
                _apply_response(state, response, _ROLE_THERAPIST)

        if verbose_detailed:
            preview = [
                (r[:100] + "...") if isinstance(r, str) and len(r) > 100 else r for r in responses
            ]
            print(f"  {speaker.upper()} ({len(preview)}): {preview}")

        for state in active:
            if not state.active:
                state.release_messages()

    return states, None, desynced


# ==============================================================================
#  FULL GENERATION PASS
# ==============================================================================


async def _run_batch(
    persona_ids: Sequence[int],
    permutations: Sequence[Dict[str, str]],
    *,
    model,
    tokenizer,
    client,
    patient_binding: RoleBinding,
    primitives: AsyncPrimitives,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    loop_kwargs: Dict[str, Any],
) -> Tuple[Optional[List[ConversationState]], Optional[str], List[int]]:
    """Seed and run one batch, normalising every failure mode into an ``error_type`` string."""
    states = [
        new_conversation_state(
            pid,
            therapist_system_prompt=therapist_system_prompt,
            patient_system_prompt=permutations[pid]["patient_system_prompt"],
            therapist_init_utterance=therapist_init_utterance,
        )
        for pid in persona_ids
    ]
    try:
        states, error_type, desynced = await conversation_loop_batch(
            states, model, tokenizer, client, patient_binding, primitives, **loop_kwargs
        )
        return states, error_type, desynced
    except Exception as exc:  # noqa: BLE001 -- a batch must never take the pass down
        print(f"  ERROR: batch generation failed: {type(exc).__name__}: {exc}")
        return None, "batch_exception", []


def _keep_state(state: ConversationState, save_dir: Optional[str], detailed: bool) -> bool:
    """Save a finished conversation and say whether to keep it.

    Dropped: ``failed=True`` (an utterance is missing), and ``<= 1`` utterance (only the scripted
    opening line, so there is nothing the policy actually produced). Message histories are
    released either way.
    """
    if state.failed or state.n_utterances <= 1:
        if detailed:
            print(f"      Skipping persona {state.persona_id} (failed or empty)")
        state.release_messages()
        return False
    if save_dir:
        save_conversation_csv(state, save_dir)
        if detailed:
            print(
                f"      Saved {conversation_filename(state.persona_id)} "
                f"({state.n_utterances} utterances)"
            )
    state.release_messages()
    return True


async def generate_all_conversations_async(
    model,
    tokenizer,
    client,
    patient_binding: RoleBinding,
    primitives: AsyncPrimitives,
    permutations: Sequence[Dict[str, str]],
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    *,
    persona_ids: Optional[Sequence[int]] = None,
    save_dir: Optional[str] = None,
    num_utterances: int = 49,
    max_tokens: int = 200,
    temperature_therapist: float = 0.9,
    temperature_patient: float = 0.7,
    therapist_max_input_tokens: int = 2048,
    stop_strings: Optional[Sequence[str]] = None,
    patient_seed: Optional[int] = None,
    batch_size: int = 8,
    batch_cooldown_seconds: float = 1.0,
    max_retries_without_progress: int = 3,
    allow_partial: bool = False,
    verbose: bool = True,
    verbose_detailed: bool = False,
) -> List[ConversationState]:
    """Generate one conversation per requested persona. The async body; see the sync wrapper.

    Args:
        permutations: The full ``generate_all_permutations()`` list. Indexed by persona id, so it
            must be the WHOLE list in its canonical order, never a shuffled subset.
        persona_ids: Which personas to run, in processing order. Defaults to every index of
            *permutations*. **This is where a per-iteration shuffle belongs**: it decides which
            personas run and in what order they are processed, and it never reaches a file name
            (see the module docstring).
        save_dir: When set, each finished conversation is written to
            ``<save_dir>/pers<PID>.csv`` and personas already on disk there are skipped.
        num_utterances: ADDITIONAL utterances after the scripted therapist opener, so a
            conversation holds at most ``num_utterances + 1`` -- **50 at the default 49**. See
            :func:`conversation_loop_batch`.
        batch_size: Conversations in flight at once. WARNING: a safety setting on the local 12 GB card
            (~1.1 GB per concurrent conversation on top of the weights, and an over-budget VRAM
            request there REBOOTS the machine instead of raising). Do the arithmetic first.

    Returns:
        Usable conversations, **sorted by persona id** -- so the return order does not depend on
        the shuffle and two iterations' outputs line up without a sort at the call site.

    Notes:
        **``gc.collect()`` + ``torch.cuda.empty_cache()`` run BETWEEN batches, and that is not
        cosmetic.** Consecutive batches reach different maximum sequence lengths, so blocks freed
        by one batch are the wrong size for the next and the caching allocator keeps asking the
        driver for more. Measured on the 12 GB local card (96 conversations, batch 6): batch 1
        peaked at 8.0 GB and batch 2 reached 11.9 GB. ``empty_cache`` frees only UNUSED cached
        blocks, so results stay bit-identical, and the cost is one re-allocation per batch. Each
        batch line prints ``vram <N>G`` (the allocator's reserved high-water mark): **flat across
        batches is healthy, climbing means this was removed.** A single-batch smoke test cannot
        detect it -- it needs at least two.

        **The batch line also prints ``trunc <n>/<B>``**: of the ``B`` therapist prompts built for
        this batch (one per therapist turn per conversation), ``n`` lost at least one oldest turn
        to ``therapist_max_input_tokens`` -- read off ``core.policy.TRUNCATION_COUNTER`` as a
        per-batch delta (plus ``overflow <k>`` when any prompt could not be built at all). A
        rising rate late in a pass is expected (conversations grow); a non-zero rate at the FIRST
        therapist turn means the budget is smaller than the system prompt plus one turn.

        **Bounded no-progress retries.** A pass that adds no conversation at all increments a
        counter; after ``max_retries_without_progress`` such passes the function stops rather
        than looping forever against a server that is down. A pass that saves even one
        conversation resets the counter.

        **OOM halves the batch, stickily.** When a batch comes back with ``error_type="oom"``,
        the working batch size is halved (floor 1) for the REST of the pass and the remaining
        personas are re-sliced at the new size -- retrying at the same size would replay the
        same allocation and fail the same personas every pass until the no-progress bound fires.
        Mirrors the look-ahead's sticky halving.

        **Partial coverage raises by default.** The personas that fail are not a random sample
        (they correlate with conversation length and difficulty), so a short set silently feeding
        training/eval is a biased-missingness hazard on the headline metric. When personas are
        still missing after the retry bound, this raises ``RuntimeError`` unless
        ``allow_partial=True`` -- pass that only where a partial set is explicitly acceptable
        (e.g. a repair tool that reports what it could not fill).
    """
    import torch  # lazy: only the generation path needs it

    from core.policy import TRUNCATION_COUNTER, vram_report
    from roles import make_client

    # Re-resolve the client on THIS loop (make_client is loop-keyed): run_async gives every
    # pass a fresh loop, and a client object from an earlier loop poisons its first calls
    # with APIConnectionError. Within this pass the client only serves patient calls, so the
    # patient binding's per-attempt timeout is the right socket bound.
    client = make_client(patient_binding)

    if persona_ids is None:
        persona_ids = list(range(len(permutations)))
    requested = [int(p) for p in persona_ids]
    total = len(requested)
    start_time = time.time()

    completed: Dict[int, ConversationState] = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for pid, state in load_conversations_dir(save_dir, verbose=verbose).items():
            if pid in requested:
                completed[pid] = state

    loop_kwargs: Dict[str, Any] = dict(
        num_utterances=num_utterances,
        max_tokens=max_tokens,
        temperature_therapist=temperature_therapist,
        temperature_patient=temperature_patient,
        therapist_max_input_tokens=therapist_max_input_tokens,
        stop_strings=stop_strings,
        patient_seed=patient_seed,
        verbose_detailed=verbose_detailed,
    )

    n_session_ended = 0
    n_capped = 0
    n_failed = 0
    desynced_all: List[int] = []
    retries_without_progress = 0

    while True:
        remaining = [pid for pid in requested if pid not in completed]
        if not remaining:
            break
        if retries_without_progress >= max_retries_without_progress:
            print(
                f"  WARNING: no progress in {max_retries_without_progress} passes; stopping with "
                f"{len(remaining)} conversations missing: {remaining}"
            )
            break

        n_batches = (len(remaining) + batch_size - 1) // batch_size
        if verbose:
            print(
                f"\n  Remaining {len(remaining)}/{total} ({n_batches} batches) "
                f"[{time.time() - start_time:.1f}s elapsed]"
            )

        progress = False
        oom_this_pass = False
        for batch_num, offset in enumerate(range(0, len(remaining), batch_size), 1):
            batch_ids = remaining[offset: offset + batch_size]
            batch_start = time.time()
            trunc_before = TRUNCATION_COUNTER.snapshot()

            states, error_type, desynced = await _run_batch(
                batch_ids,
                permutations,
                model=model,
                tokenizer=tokenizer,
                client=client,
                patient_binding=patient_binding,
                primitives=primitives,
                therapist_system_prompt=therapist_system_prompt,
                therapist_init_utterance=therapist_init_utterance,
                loop_kwargs=loop_kwargs,
            )
            desynced_all.extend(desynced)

            if states is None:
                if verbose:
                    print(
                        f"    Batch {batch_num}/{n_batches} FAILED ({error_type}) -- "
                        f"{time.time() - batch_start:.1f}s"
                    )
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(batch_cooldown_seconds)
                continue

            saved = 0
            for state in states:
                if _keep_state(state, save_dir, verbose_detailed):
                    completed[state.persona_id] = state
                    saved += 1
                    progress = True
                    if state.session_ended_by:
                        n_session_ended += 1
                    else:
                        n_capped += 1
                else:
                    n_failed += 1

            # A returned batch can still carry a therapist-side error (the loop marks the
            # still-active states failed and returns what it has). OOM at this batch size will
            # OOM at this batch size again -- halve stickily and re-slice the remaining
            # personas at the new width instead of replaying the same allocation.
            if error_type == "oom" and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                oom_this_pass = True
                print(
                    f"    OOM in batch {batch_num}: halving conversation batch size to "
                    f"{batch_size} (sticky); re-slicing the remaining personas"
                )
                del states
                gc.collect()
                torch.cuda.empty_cache()
                break  # the offsets of this pass were strided at the old size

            # Release this batch's KV cache and activations before the next batch allocates its
            # own. See the Notes above: without this the allocator's high-water mark grows every
            # batch, and on a 12 GB card that ends in a reboot rather than an exception.
            del states
            gc.collect()
            torch.cuda.empty_cache()

            if verbose:
                mem = vram_report()
                trunc = TRUNCATION_COUNTER.delta_since(trunc_before)
                overflow = f", overflow {trunc['overflow']}" if trunc["overflow"] else ""
                print(
                    f"    Batch {batch_num}/{n_batches}: {saved}/{len(batch_ids)} saved -- "
                    f"{len(completed)}/{total} total ({len(completed) / total * 100:.0f}%) -- "
                    f"batch {time.time() - batch_start:.1f}s, "
                    f"total {time.time() - start_time:.1f}s, "
                    f"vram {mem['reserved_gib']:.1f}G, "
                    f"trunc {trunc['truncated']}/{trunc['prompts']}{overflow}"
                )

        if progress or oom_this_pass:
            # A pass that only halved the batch made progress of a kind: the next pass runs a
            # genuinely different (smaller) allocation. Halving is log2-bounded, so this cannot
            # loop forever -- at batch_size 1 an OOM no longer resets the counter.
            retries_without_progress = 0
        else:
            retries_without_progress += 1
            if verbose:
                print(
                    f"  No progress this pass "
                    f"({retries_without_progress}/{max_retries_without_progress})"
                )

    missing = [pid for pid in requested if pid not in completed]
    if verbose:
        print(
            f"\n  Generation summary ({time.time() - start_time:.1f}s):\n"
            f"    Requested         : {total}\n"
            f"    Usable            : {len(completed)}\n"
            f"      session ended   : {n_session_ended}\n"
            f"      reached cap     : {n_capped}\n"
            f"    Desync (graceful) : {len(desynced_all)}\n"
            f"    Failed / empty    : {n_failed}\n"
            f"    Missing           : {len(missing)}"
        )

    if missing and not allow_partial:
        raise RuntimeError(
            f"conversation generation is INCOMPLETE: {len(missing)}/{total} personas missing "
            f"after the retry bound ({sorted(missing)}). The failures are not a random sample "
            f"(they correlate with length and difficulty), so training or evaluating on the "
            f"survivors is a biased subset. Fix the cause (server down? OOM at batch 1? patient "
            f"timeouts?) and re-run -- conversations already on disk are reloaded, so only the "
            f"missing personas are regenerated. Pass allow_partial=True only where a partial "
            f"set is explicitly acceptable."
        )

    return [completed[pid] for pid in sorted(completed)]


def generate_all_conversations(*args, **kwargs) -> List[ConversationState]:
    """Synchronous entry point for :func:`generate_all_conversations_async`.

    Takes exactly the same arguments and returns the same list. Runs the whole pass on ONE event
    loop via ``core.concurrency.run_async``, which works from a plain script and from inside a
    live Jupyter loop alike.

    Notes:
        One loop for the entire pass, not one per batch: ``AsyncPrimitives`` keys its semaphores
        by loop id and evicts stale loops, so a per-batch loop would rebuild the patient semaphore
        every batch. Correct either way, but the bound would be re-created rather than shared.

        Call this from the trainers' orchestration cells. Code that is ALREADY inside a coroutine
        must await :func:`generate_all_conversations_async` directly -- wrapping it in
        :func:`run_async` from there would run a nested loop on a worker thread for no reason.
    """
    return run_async(generate_all_conversations_async(*args, **kwargs))


# ==============================================================================
#  PROMPT EXTRACTION (conversations -> per-turn training samples)
# ==============================================================================
#
# One training sample after EVERY patient turn -- the exact point where the therapist is about to
# speak. Two filters apply:
#
#   MCL     drop a slice whose conversation-so-far has fewer than `min_conv_length` total
#           utterances (therapist + patient combined). Short cuts are a weak proxy for the full
#           conversation the thesis actually evaluates: rank agreement with the final-conversation
#           score is barely above chance at 2 utterances and only clears 0.8 around 10.
#   BUDGET  cap the prompt at `max_prompt_tokens` by dropping the OLDEST turns whole -- the SAME
#           `core.policy.build_prompt` the therapist decode path uses, so a training prompt is
#           byte-identical to the text the policy generated from for the same turns and budget.
#           Exp3's alternative "legacy" mode kept the last N tokens of the rendered string, which
#           slices through the template's control tokens; it is not ported.
#
# THE BOS RULE (core.policy module docstring): every `prompt` string returned from here is
# BOS-FREE. TRL's GRPOTrainer tokenizes it with `processing_class(text=prompts)` -- i.e.
# `add_special_tokens=True` -- which adds the one BOS back, matching the decode path's ids
# exactly. (DPO's tokenization path is the PTO trainer's to verify; `strip_leading_bos` is the
# tool.) Budgets count that BOS: `max_prompt_tokens` is the length of
# `core.policy.prompt_token_ids(prompt)`, the model's actual input length.


def _build_prompt_for_turns(
    turns: Sequence[Dict[str, str]],
    system_prompt: str,
    tokenizer,
    max_prompt_tokens: int,
    *,
    turn_token_costs: Optional[Sequence[int]] = None,
    system_overhead: Optional[int] = None,
) -> Optional[str]:
    """``core.policy.build_prompt`` over ``turns_to_messages(turns, system_prompt)``.

    Returns the BOS-free rendered prompt, or ``None`` when even the newest turn alone exceeds
    the budget. The optional estimates (from ``core.policy.estimate_message_costs`` /
    ``system_overhead``) let the extraction pass avoid re-encoding every turn per slice.
    """
    from core.policy import build_prompt  # lazy: torch-side module

    if not turns:
        return None
    text, _ = build_prompt(
        turns_to_messages(turns, system_prompt), tokenizer, int(max_prompt_tokens),
        message_token_costs=turn_token_costs, system_overhead=system_overhead,
    )
    return text


def build_truncated_training_prompt(
    turns: Sequence[Dict[str, str]],
    system_prompt: str,
    tokenizer,
    max_prompt_tokens: int,
    truncation_mode: str = "drop_oldest",
) -> Optional[str]:
    """Render *turns* as a therapist prompt, capped at *max_prompt_tokens*.

    Args:
        turns: Conversation-so-far, ending on the patient turn the therapist will answer.
        system_prompt: The therapist system prompt.
        tokenizer: From ``core.policy.setup_tokenizer`` (carries the therapist's chat template).
        max_prompt_tokens: Budget, in tokens, for the prompt -- the length of
            ``core.policy.prompt_token_ids(prompt)``, i.e. INCLUDING the one BOS the tokenization
            adds. Give it the same value as the decode path's ``max_input_tokens`` and the two
            produce byte-identical text.
        truncation_mode: Only ``"drop_oldest"`` is supported.

    Returns:
        The rendered prompt -- **BOS-FREE**: a leading ``<|begin_of_text|>`` written by the
        Instruct template is stripped, so whoever tokenizes the string adds exactly one BOS
        (TRL's ``processing_class(text=...)`` does; so does ``core.policy.prompt_token_ids``).
        Or ``None`` when even a single most-recent turn exceeds the budget. **A ``None`` means
        SKIP this sample.** Training on a prompt that was silently mangled to fit is worse than
        training on one fewer pair.

    Raises:
        ValueError: on any other ``truncation_mode``. Exp3 had a ``"legacy"`` mode that kept the
            last N tokens of the rendered string; it can slice through the template's control
            tokens and produce a prompt whose role framing is broken, so it is deliberately not
            ported rather than left as a footgun with a default.

    Notes:
        This is ``core.policy.build_prompt`` -- the exact function
        ``core.policy.generate_therapist_batch`` builds its prompts with -- applied to
        ``turns_to_messages(turns, system_prompt)``. PTO samples its branch candidates through
        that decode path and trains on THIS output, so the two being one function is what makes
        the DPO prompt byte-identical to the text the candidates were sampled from (same turns,
        same budget). :func:`extract_prompts_from_conversations` applies the same rule to every
        GRPO prompt. Two further reasons the cap matters. (1) It matches the therapist's
        inference-time context window, so the policy is not trained on a context it never sees
        at serve time. (2) It stops a full grown trunk (~2.4k tokens, up to ~6k) from reaching
        ``DPOTrainer`` verbatim: TRL 1.4.0's ``DPOConfig`` has no ``max_prompt_length`` and caps
        prompt+completion with one ``max_length`` under ``truncation_mode='keep_start'`` -- which
        slices the RESPONSE off the end, leaving a pair whose chosen and rejected are both
        truncated to nothing.
    """
    if truncation_mode != "drop_oldest":
        raise ValueError(
            f"build_truncated_training_prompt: truncation_mode={truncation_mode!r} is not "
            f"supported; only 'drop_oldest' exists in Exp4. (Exp3's 'legacy' token-tail mode "
            f"could cut through chat-template control tokens and was not ported.)"
        )
    return _build_prompt_for_turns(turns, system_prompt, tokenizer, max_prompt_tokens)


def extract_prompts_from_conversations(
    states: Sequence[ConversationState],
    system_prompt: str,
    tokenizer,
    *,
    min_conv_length: int = 2,
    max_prompt_tokens: int = 4096,
    permutations: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Slice finished conversations into per-turn therapist training samples.

    Args:
        states: Finished conversations (from generation or from disk).
        system_prompt: The therapist system prompt, rendered into every prompt.
        tokenizer: Carries the therapist's chat template.
        min_conv_length: MCL -- the minimum number of utterances (therapist + patient combined)
            in the conversation-so-far for a slice to be eligible. ``2`` is a no-op. The knob
            exists because the training reward grades these partial cuts while the thesis
            evaluates whole conversations: rank agreement with the final-conversation score is
            barely above chance at 2 utterances, clears 0.8 near 10 and 0.9 near 30.
        max_prompt_tokens: Token budget per prompt, BOS included (see
            :func:`build_truncated_training_prompt`); over-budget slices drop their oldest turns
            whole, and a slice that cannot fit at all is skipped.
        permutations: The full ``generate_all_permutations()`` list, indexed by ``persona_id``.

    Returns:
        One dict per slice with keys ``prompt`` (chat-template text for the model -- **BOS-FREE**,
        exactly what :func:`build_truncated_training_prompt` returns; TRL's ``GRPOTrainer``
        tokenizes it with ``add_special_tokens=True`` and thereby adds the single BOS the decode
        path also has), ``transcript`` (labelled plain text for the oracle), ``conversation_id``
        (``"pers07"``), ``persona_id`` (int) and ``patient_system_prompt``.

    Notes:
        **WARNING -- ``patient_system_prompt`` is REQUIRED for look-ahead.** K-turn simulation has to
        continue the conversation against the SAME patient, and it can only do that if this field
        arrives. Passing ``permutations=None``, or a list that does not cover the persona ids in
        *states*, makes every value ``""`` -- and look-ahead would then roll out against a
        default patient, silently changing what the reward measures. This function prints a loud
        warning when that happens; do not train through it.

        ``conversation_id`` is the file stem, so a sample can be traced back to the CSV it came
        from without a lookup, and ``persona_id`` is the stable int to join on.

        Every slice pays one exact render + count (the fit check), and only an over-budget slice
        pays the drop-point estimate -- fed from per-turn costs measured ONCE per conversation
        (``core.policy.estimate_message_costs``) rather than once per slice, ~2,300 slices per
        iteration.
    """
    from core.policy import estimate_message_costs, system_overhead as _system_overhead

    sys_overhead = _system_overhead(system_prompt, tokenizer)
    samples: List[Dict[str, Any]] = []
    n_missing_patient_prompt = 0

    for state in states:
        turns = state.turns
        if not turns or len(turns) < min_conv_length:
            continue

        patient_system_prompt = ""
        if permutations is not None and 0 <= state.persona_id < len(permutations):
            patient_system_prompt = str(
                permutations[state.persona_id].get("patient_system_prompt") or ""
            )
        if not patient_system_prompt:
            n_missing_patient_prompt += 1

        # Order-aligned with `turns` (turns_to_messages adds only the system message in front).
        turn_costs = estimate_message_costs(turns_to_messages(turns, system_prompt), tokenizer)

        for i, turn in enumerate(turns):
            if turn["role"] != _ROLE_PATIENT:
                continue
            if (i + 1) < min_conv_length:  # i+1 == utterances in the conversation-so-far
                continue

            partial = turns[: i + 1]
            prompt = _build_prompt_for_turns(
                partial, system_prompt, tokenizer, max_prompt_tokens,
                turn_token_costs=turn_costs[: i + 1], system_overhead=sys_overhead,
            )
            if prompt is None:
                continue  # even one most-recent turn exceeds the budget -- skip the sample

            samples.append({
                "prompt": prompt,
                "transcript": format_conversation_for_oracle(partial),
                "conversation_id": state.conversation_id,
                "persona_id": state.persona_id,
                "patient_system_prompt": patient_system_prompt,
            })

    if states and n_missing_patient_prompt:
        scope = "EVERY" if n_missing_patient_prompt == len(states) else "some"
        print(
            f"  WARNING: {scope} conversation ({n_missing_patient_prompt}/{len(states)}) produced "
            f"an EMPTY patient_system_prompt. K-turn look-ahead cannot roll out against the right "
            f"patient without it and would silently simulate a different one. Pass the full "
            f"generate_all_permutations() list as permutations=, indexed by persona_id."
        )

    return samples
