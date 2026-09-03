"""oracle_sanity.py -- is this grader a measuring instrument? The gate before any training spend.

Exp4 replaces a vendor API grader with an open model behind a local vLLM server. That buys a $0
run and costs the one guarantee the vendor was providing for free: that the thing scoring the
conversations can actually tell them apart. Every number this experiment produces -- GRPO's
advantages, PTO's (chosen, rejected) split, every eval table, every contrast between arms -- is a
function of what the oracle returns. If the oracle is not measuring, nothing downstream fails; it
just produces confident numbers about nothing.

So this module is run FIRST, against a committed fixture of real Exp3 transcripts whose quality
range is known, and it blocks the run when the grader is unfit.

The two ways an open grader fails
---------------------------------
**1. LOUD -- it ignores the schema, or returns the wrong number of item scores.** This is caught by
``core.oracle``'s validation ladder (the ``questionnaire_id`` echo, the item-count check, the
type/range check), which is exactly why that ladder exists: vLLM's guided decoding is a best-effort
constraint engine, not a contract, and some builds quietly ignore ``minItems``/``maxItems``. But
"caught" does not mean "visible". The scoring path converts a failure into a skipped candidate, so
what a run actually sees is a rising retry count and then **biased missingness on the headline
metric** -- the conversations the grader found hardest to score are precisely the ones that
disappear from the mean. Here that failure is counted per rubric and blocks the run at the first
occurrence: ``schema_valid_rate`` must be 1.000, not 0.98.

**2. SILENT, AND WORSE -- it honours the schema perfectly and returns degenerate scores.** Every
item a 4; near-zero variance across conversations that a validated grader placed four points apart.
That response parses, passes every rung of the validation ladder, writes valid parquet, and yields
a grader that cannot distinguish any two arms. **Nothing downstream flags it.** The contrast tables
come back at ~0, the plots look flat, and it reads like a finding: "look-ahead makes no
difference", "PTO and GRPO are equivalent". There is no error, no warning, and no artifact that
says otherwise -- which is why the spread check is a HARD gate here rather than a note in a report.
A grader with no variance is not a weak grader; it is not a grader.

Hard gates vs soft signals -- and why the offset is meaningless
--------------------------------------------------------------
* **HARD (blocks the run, nonzero exit):** ``schema_valid_rate == 1.0`` on every requested rubric,
  and a per-conversation SD at or above :data:`MIN_SCORE_SD` on every rubric AND on the pooled
  training reward.
* **SOFT (reported, never blocks):** Spearman rank agreement with the reference, and the level
  offset.

The reference scores in the fixture are what ``gpt-4o-mini`` gave those same transcripts in Exp3.
**Exp4 and Exp3 are not on the same score axis** -- a different grader with a different prior over
the rubric sits systematically higher or lower, and the offset carries no information about
fitness. Blocking on it would reject a perfectly good grader for the crime of being a different
one. What survives the axis change is the ORDERING (does this grader rank the conversations the way
a validated grader did?) and the SPREAD (does it separate them at all?). Spread is hard because a
grader without it cannot measure anything; ordering is soft because rho over twelve conversations
is weak evidence, and because the reference is itself one draw from a stochastic judge.

Why it calls the training code path
-----------------------------------
Scoring goes through ``core.oracle.get_evaluation_json`` -- the same prompt builder, the same
``response_format`` shim, the same retry loop, the same validation ladder the trainer uses. A
re-implementation here would be a *second* code path, and the one class of bug this gate exists to
catch is precisely a bug in the path that trains. A green report against a private scorer proves
nothing about the run.

``get_evaluation_json`` rather than ``score_conversation`` for one reason: ``score_conversation``
short-circuits the moment any rubric fails, so a grader that cannot do Q2 would hide Q1's result.
Per-rubric counts need each call to be independent. The pooled row is then reduced exactly the way
``score_conversation`` documents it -- the unweighted mean across rubrics, counted only where every
rubric validated.

The third gate: does the oracle prompt FIT?
------------------------------------------
Unrelated to the grader's judgement and just as silent: the full oracle prompt (rubric + the
whole transcript) has to fit the server's ``max_model_len``, and a conversation whose prompt does
not fit is simply absent from the score -- no error, just a missing row. The prompts that
overflow are the LONGEST conversations, and session length varies by arm and by K, so the
dropout would be arm-dependent: a bias on the very contrast the experiment measures. The 16384
cap was sized on Exp3 transcripts written by ``gpt-4o-mini``; Exp4's Gemma patient may write
longer turns. :func:`prompt_length_report` measures the REALISED distribution on real Exp4
conversations (the Phase 2 gate), counting tokens through the served model's own tokenizer
(vLLM's ``/tokenize``) whenever a server is reachable, and :func:`check_prompt_lengths` fails
loudly when any prompt exceeds the served context length. ``tools/generate_convs.py`` runs it
at the end of every pass.

Usage::

    # CLI, from Exp4_OpenStack/code/
    python tools/oracle_sanity.py --quick
    python tools/oracle_sanity.py --model google/gemma-4-E4B-it --out ../data/runs/<ARM>/
    python tools/oracle_sanity.py --conv-dir ../data/conversations/<ARM>/model_iter_0   # lengths only

    # notebook, before iteration 1
    report = run_async(run_sanity(bindings["oracle"], quick=True))
    print(format_report(report))
    ok, reasons = check_gates(report)
    if not ok:
        raise RuntimeError("oracle unfit: " + "; ".join(reasons))

    # Phase 2: the realised oracle-prompt lengths on real Exp4 conversations -- a model-state
    # folder, or a list of oracle-formatted transcripts already in memory (Run_Eval § 8)
    lengths = prompt_length_report(conv_dir, (1, 2), base_url=bindings["oracle"].base_url,
                                   model=bindings["oracle"].model)
    print(format_prompt_length_report(lengths))
    ok, reasons = check_prompt_lengths(lengths)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import fmean, median, stdev
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# `python tools/oracle_sanity.py` puts tools/ on sys.path, NOT code/, so the project imports below
# would not resolve when this file runs as a script. The trainer notebooks already prepend code/,
# where this is a no-op. It has to happen before the imports, hence the noqa markers.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from core.concurrency import AsyncPrimitives, run_async            # noqa: E402
from core.oracle import (                                          # noqa: E402
    OracleConfig,
    get_evaluation_json,
    make_oracle_client,
    openai_compat_strict,
    set_openai_compat_strict,
)
from naming import qtag_for                                        # noqa: E402
from questionnaires import QuestionnaireID, get_prompt_eval_questionnaire   # noqa: E402
from roles import (                                                # noqa: E402
    DEFAULT_ORACLE_MODEL,
    PROVIDERS,
    RoleBinding,
    ServeSpec,
    make_binding,
)
from tools.vllm_serve import (                                     # noqa: E402
    _get_json,
    _model_ids_match,
    _models_url,
    _served_model_ids,
    served_max_model_len,
    wait_until_ready,
)

__all__ = [
    "SCHEMA_VERSION",
    "FIXTURE_PATH",
    "MIN_SCORE_SD",
    "REQUIRED_SCHEMA_VALID_RATE",
    "MIN_ITEMS_FOR_SPEARMAN",
    "QUICK_N_ITEMS",
    "REPORT_FILENAME",
    "CHARS_PER_TOKEN_ESTIMATE",
    "PROMPT_LENGTH_REPORT_FILENAME",
    "FixtureItem",
    "Fixture",
    "Observation",
    "MetricReport",
    "SanityReport",
    "PromptLengthRow",
    "PromptLengthReport",
    "load_fixture",
    "select_spanning",
    "metric_label",
    "combined_label",
    "spearman",
    "run_sanity",
    "check_gates",
    "format_report",
    "write_report",
    "prompt_length_report",
    "check_prompt_lengths",
    "format_prompt_length_report",
    "write_prompt_length_report",
    "main",
]


# ==============================================================================
#                               CONSTANTS
# ==============================================================================

#: Payload shape of the archived JSON report. A reader keys on this.
SCHEMA_VERSION = "exp4-oracle-sanity/1"

#: The committed fixture: 12 real Exp3 conversations spanning the quality range, each carrying the
#: ``gpt-4o-mini`` scores it actually received in Exp3. Rebuilt by
#: ``tools/fixtures/build_fixture.py`` -- but rebuilding changes the yardstick, so do not do it to
#: make a report look better.
FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fixtures", "sanity", "transcripts.json")

#: **THE degenerate-grader threshold.** Sample SD (ddof=1) of the per-conversation scores, below
#: which the grader is judged to be answering from a template rather than reading.
#:
#: 0.25 is chosen against the fixture's own yardstick: the reference SD on the pooled Q1+Q2 reward
#: is ~1.41 across these 12 conversations, whose reference scores run the full 1.00-5.00 range. A
#: grader that compresses a four-point spread into less than a quarter of a point has thrown away
#: ~98% of the variance the reference found, and no downstream contrast -- which is a difference of
#: MEANS between arms, measured against exactly this within-arm spread -- could clear noise. The
#: threshold is deliberately far below any plausible real grader (Exp3's own graders sit near 1.0
#: on this fixture) so that tripping it is unambiguous evidence of a template, not a judgement call
#: about a grader that is merely conservative.
#:
#: WARNING: this is a floor on being a measuring instrument at all, NOT a target. Passing at 0.30 is
#: a bad sign worth investigating (thinking mode still on, a truncated ``max_tokens`` clipping the
#: JSON, the prompt hitting a refusal template); passing is not the same as being good.
MIN_SCORE_SD = 0.25

#: Valid scores a rubric needs before the spread gate is allowed to BLOCK a run.
#:
#: Below this the gate has no power and false-fails honest graders. Measured: in ``--quick`` mode
#: (2 transcripts) a healthy grader returned the same Q1 mean for both conversations, sd 0.000,
#: and was refused -- two conversations scoring alike is ordinary, not evidence of a template.
#:
#: The cost of getting this wrong is asymmetric and not symmetric-looking. A gate that blocks real
#: work teaches whoever hits it to pass ``--force`` or delete the call, and then it is not
#: protecting the full pre-flight run either, which is the one that matters. So below this size the
#: spread check is REPORTED and not enforced, while schema compliance stays hard at any size (a
#: malformed response is malformed whether you saw two of them or two hundred).
MIN_N_FOR_SPREAD_GATE = 5

#: Schema compliance demanded of every requested rubric. 1.0, not 0.99: a partial rate is not a
#: slightly worse grader, it is missingness correlated with conversation difficulty (see the module
#: docstring), and a mean over the survivors is a biased number that no error will ever mention.
REQUIRED_SCHEMA_VALID_RATE = 1.0

#: Below this many paired (observed, reference) points, rank agreement is reported as ``None``.
#: Spearman on two points is +/-1 by construction and says nothing.
MIN_ITEMS_FOR_SPEARMAN = 3

#: ``--quick`` fixture size: the extremes of the reference range (see :func:`select_spanning`).
QUICK_N_ITEMS = 2

#: Filename used when ``--out`` names a directory -- e.g. an arm's run folder, next to
#: ``run_metadata.json``.
REPORT_FILENAME = "oracle_sanity.json"

#: Fixture key holding the frozen reference scores, and the sort key used to order the fixture by
#: quality. ``Q1Q2`` is the pooled Exp1/Exp3 training reward.
_REFERENCE_SORT_KEY = "Q1Q2"

#: Characters per token for the LAST-RESORT prompt-length estimate (no server, no tokenizer).
#: English prose on the Gemma / o200k / Llama-3 tokenizers runs ~3.5-4.2 chars per token, so
#: dividing by 3.5 OVER-counts -- deliberately: an estimate that has to stand in for a gate
#: should err toward reporting an overflow, never toward hiding one. A report built on it says
#: so in its ``method`` column and must not be quoted as a measurement.
CHARS_PER_TOKEN_ESTIMATE = 3.5

#: Filename used when ``--out`` names a directory for the prompt-length report.
PROMPT_LENGTH_REPORT_FILENAME = "oracle_prompt_lengths.json"

#: Per-call ceiling for the ``/tokenize`` probe. Tokenizing a 10k-token prompt is milliseconds
#: server-side; anything slower is a queue, and 96 x 2 calls behind a busy server are still
#: bounded by this.
_TOKENIZE_TIMEOUT_S = 30.0


# ==============================================================================
#                                THE FIXTURE
# ==============================================================================


@dataclass(frozen=True)
class FixtureItem:
    """One reference conversation: real Exp3 text plus the scores a validated grader gave it.

    Attributes:
        id: Stable identifier, e.g. ``PTOExp3_LA0_Base_c34``.
        source_model_state: Which Exp3 policy produced it. Three states are represented on purpose
            -- an untrained base (the bottom of the range), a late PTO iteration (the top), and a
            late GRPO iteration that regressed into sycophancy. The last is a DIFFERENT failure
            shape, so discrimination is tested against more than one distribution of bad text.
        n_utterances: Therapist + patient turns.
        n_chars: Transcript length, for spotting a grader whose failures track input length.
        transcript: The text to grade, in the ``[THERAPIST]: ... \\n\\n[PATIENT]: ...`` format
            ``core.conversations.format_conversation_for_oracle`` produces.
        reference: ``{"Q1": float, "Q2": float, "Q1Q2": float}`` from the Exp3 score lake.
            **A different grader's axis, not ground truth** -- see :meth:`reference_for`.
    """

    id: str
    source_model_state: str
    n_utterances: int
    n_chars: int
    transcript: str
    reference: Dict[str, float]

    def reference_for(self, label: str) -> Optional[float]:
        """The reference score for rubric *label*, or ``None`` if the fixture has none.

        Warning:
            The fixture only carries ``Q1``, ``Q2`` and their pooled mean ``Q1Q2``. Running the
            gate on WAI-SR, CSQ-8, MI-SAT, MITI, PCT or MICI is legitimate -- the hard gates
            (schema compliance and spread) need no reference at all -- but every SOFT number
            for those rubrics comes back ``None``, and a report full of ``n/a`` in the rho column
            means "no reference exists", never "the grader failed".
        """
        value = self.reference.get(label)
        return None if value is None else float(value)

    @property
    def sort_value(self) -> float:
        """Quality rank key: the pooled reference reward, or the mean of whatever references exist.

        Used only to order the fixture so :func:`select_spanning` can pick from the ends of the
        quality range. Missing references sort to 0.0, which puts them at the bottom -- acceptable
        for a selection heuristic and never used in any reported statistic.
        """
        pooled = self.reference.get(_REFERENCE_SORT_KEY)
        if pooled is not None:
            return float(pooled)
        values = [float(v) for v in self.reference.values() if v is not None]
        return fmean(values) if values else 0.0


@dataclass(frozen=True)
class Fixture:
    """The loaded fixture file: its provenance metadata plus the conversations."""

    path: str
    schema_version: int
    built_at: str
    reference_judge: str
    reference_source: str
    note: str
    items: Tuple[FixtureItem, ...]


def load_fixture(path: Optional[str] = None) -> Fixture:
    """Load the committed sanity fixture.

    Args:
        path: Fixture JSON to read. Defaults to :data:`FIXTURE_PATH`.

    Returns:
        A :class:`Fixture` with at least one item.

    Raises:
        FileNotFoundError: the fixture is missing. It is committed to the repo, so this means the
            checkout is incomplete -- not that it should be regenerated to make the gate run.
        ValueError: the file parses but does not carry the expected shape.

    Notes:
        The fixture cost nothing to build (the reference scores were lifted from the Exp3 score
        lake, not re-graded) and it must stay frozen. Regenerating it against different source
        conversations silently changes the yardstick every past report was measured with, so a
        stored report would no longer be comparable to a new one.
    """
    fixture_path = os.path.abspath(path or FIXTURE_PATH)
    if not os.path.isfile(fixture_path):
        raise FileNotFoundError(
            f"Oracle-sanity fixture not found at {fixture_path}. It is committed to the repo "
            f"(code/tools/fixtures/sanity/transcripts.json); a missing file means an incomplete "
            f"checkout. Rebuilding it with tools/fixtures/build_fixture.py needs the Exp3 score "
            f"lake and CHANGES THE YARDSTICK -- do not do that to get past this error."
        )

    with open(fixture_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(
            f"{fixture_path} is not a sanity fixture (expected an object with 'items')."
        )

    items: List[FixtureItem] = []
    for raw in payload["items"]:
        try:
            items.append(FixtureItem(
                id=str(raw["id"]),
                source_model_state=str(raw.get("source_model_state", "")),
                n_utterances=int(raw.get("n_utterances", 0)),
                n_chars=int(raw.get("n_chars", len(str(raw.get("transcript", ""))))),
                transcript=str(raw["transcript"]),
                reference={str(k): float(v) for k, v in (raw.get("reference") or {}).items()},
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{fixture_path}: malformed fixture item {raw!r} ({exc})") from exc

    if not items:
        raise ValueError(f"{fixture_path} holds no items; there is nothing to gate on.")

    # Results are keyed by (item id, rubric), so a duplicate id would silently collapse two
    # conversations into one -- shrinking every count and every SD without any error.
    duplicates = sorted({it.id for it in items if [x.id for x in items].count(it.id) > 1})
    if duplicates:
        raise ValueError(f"{fixture_path}: duplicate item ids {duplicates}; ids must be unique.")

    return Fixture(
        path=fixture_path,
        schema_version=int(payload.get("schema_version", 0)),
        built_at=str(payload.get("built_at", "")),
        reference_judge=str(payload.get("reference_judge", "")),
        reference_source=str(payload.get("reference_source", "")),
        note=str(payload.get("note", "")),
        items=tuple(items),
    )


def select_spanning(items: Sequence[FixtureItem], n: int) -> Tuple[FixtureItem, ...]:
    """Pick *n* items spread across the reference quality range, endpoints included.

    Args:
        items: Fixture items to choose from.
        n: How many to keep. ``n >= len(items)`` returns everything.

    Returns:
        Items ordered worst-to-best by :attr:`FixtureItem.sort_value`.

    Notes:
        **The endpoints are the point.** ``--quick`` scores two conversations, and two conversations
        drawn arbitrarily could easily be near-neighbours in quality -- against which a template
        grader returning one constant looks exactly like a careful grader. Taking the bottom and the
        top of the reference range (1.00 and 5.00 on the shipped fixture) means even the two-item
        spread check asks a real question: does this grader separate the worst conversation in the
        set from the best?

        The quick spread gate is still weaker than the full one -- with n=2 the sample SD is
        ``|a-b| / sqrt(2)``, so :data:`MIN_SCORE_SD` fires only when the grader puts those two
        within ~0.35 points of each other. That is the intended sensitivity for a pre-flight: it
        catches a template, not a merely compressed scale. Run the full report before a real arm.
    """
    ordered = sorted(items, key=lambda it: it.sort_value)
    count = len(ordered)
    if n <= 0 or n >= count:
        return tuple(ordered)
    if n == 1:
        return (ordered[count // 2],)

    picked: List[FixtureItem] = []
    seen: set = set()
    for i in range(n):
        idx = round(i * (count - 1) / (n - 1))
        if idx not in seen:
            seen.add(idx)
            picked.append(ordered[idx])
    return tuple(picked)


# ==============================================================================
#                            RUBRIC LABELS
# ==============================================================================


def metric_label(questionnaire_id: int) -> str:
    """Display/reference label for one rubric: ``1 -> "Q1"``, ``3 -> "WAI"``, ``7 -> "MITI"``.

    Notes:
        Delegates to ``naming.qtag_for`` so the labels in this report are the same tokens that
        appear in an arm name and in the fixture's ``reference`` keys -- one spelling, not three.
        PCT (8) and MICI (9) have no arm-name token (they are eval-only instruments), so they fall
        back to their ``QuestionnaireID`` names.
    """
    qid = int(questionnaire_id)
    try:
        return qtag_for([qid])
    except ValueError:
        pass
    try:
        return QuestionnaireID(qid).name
    except ValueError:
        return f"qid{qid}"


def combined_label(questionnaire_ids: Sequence[int]) -> str:
    """Label for the pooled reward across *questionnaire_ids*: ``(1, 2) -> "Q1Q2"``.

    Falls back to ``"Q1+WAI"``-style concatenation for a set with no arm-name token. The pooled
    row is the training reward itself, which is why it gets a label at all.
    """
    ids = [int(q) for q in questionnaire_ids]
    try:
        return qtag_for(ids)
    except ValueError:
        return "+".join(metric_label(q) for q in ids)


def _validate_questionnaire_ids(questionnaire_ids: Sequence[int]) -> Tuple[int, ...]:
    """Coerce to a de-duplicated tuple of valid rubric ids, preserving order. Raises on unknowns."""
    if not questionnaire_ids:
        raise ValueError("run_sanity: questionnaire_ids is empty; there is nothing to check.")
    valid = {member.value for member in QuestionnaireID}
    out: List[int] = []
    for raw in questionnaire_ids:
        qid = int(getattr(raw, "value", raw))
        if qid not in valid:
            raise ValueError(
                f"run_sanity: questionnaire id {qid} is not a QuestionnaireID "
                f"(known: {sorted(valid)})."
            )
        if qid not in out:
            out.append(qid)
    return tuple(out)


# ==============================================================================
#                     RANK AGREEMENT (no scipy, no numpy)
# ==============================================================================
#
# Deliberately stdlib-only. This tool runs on the trainer host before anything else is installed,
# and on Colab before the notebook's heavy imports; a scipy dependency here would be a reason for
# the gate not to run, and a gate that does not run is worse than no gate.


def _average_ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks of *values*, ties sharing the mean of the ranks they span.

    Tie handling is not cosmetic here: a partly-degenerate grader returns many identical scores,
    and ranking those arbitrarily would manufacture agreement (or disagreement) out of ordering
    noise. Averaged ties make an all-constant vector rank perfectly flat, which is what makes
    :func:`spearman` return ``None`` for it rather than an invented number.
    """
    count = len(values)
    order = sorted(range(count), key=lambda i: values[i])
    ranks = [0.0] * count
    i = 0
    while i < count:
        j = i
        while j + 1 < count and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or ``None`` when either vector has zero variance."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = fmean(xs), fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(d * d for d in dx) * sum(d * d for d in dy))
    if denom <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    """Spearman rank correlation of two equal-length vectors, pairwise-complete.

    Args:
        xs: Observed values; ``None`` entries drop the pair.
        ys: Reference values; ``None`` entries drop the pair.

    Returns:
        rho in ``[-1, 1]``, or ``None`` when fewer than :data:`MIN_ITEMS_FOR_SPEARMAN` complete
        pairs remain or when either ranked vector is constant.

    Notes:
        Implemented as Pearson on average ranks -- the definition, valid with ties, and identical
        to ``scipy.stats.spearmanr``'s statistic. The rank-difference shortcut
        (``1 - 6*sum(d^2)/(n(n^2-1))``) is NOT used because it is only correct when there are no
        ties, and ties are exactly the case a partly-degenerate grader produces.

        ``None`` for a constant vector is deliberate and is the interesting case: a grader that
        returns one score for everything has no ordering to agree with, and reporting rho = 0
        would present "undefined" as "no relationship" -- which reads as a soft finding rather
        than as the hard spread failure it actually is.
    """
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < MIN_ITEMS_FOR_SPEARMAN:
        return None
    rank_x = _average_ranks([p[0] for p in pairs])
    rank_y = _average_ranks([p[1] for p in pairs])
    return _pearson(rank_x, rank_y)


# ==============================================================================
#                              REPORT SHAPES
# ==============================================================================


@dataclass(frozen=True)
class Observation:
    """One (conversation, rubric) outcome.

    Attributes:
        item_id: Fixture item id.
        score: Mean item score the grader returned, or ``None`` when the call never validated.
        reference: The frozen Exp3 score for this rubric, or ``None`` when the fixture has none.
        attempts: Oracle HTTP calls spent, retries included. 1 is a clean pass.
    """

    item_id: str
    score: Optional[float]
    reference: Optional[float]
    attempts: int

    def to_dict(self) -> Dict[str, Any]:
        return {"item_id": self.item_id, "score": self.score,
                "reference": self.reference, "attempts": self.attempts}


@dataclass(frozen=True)
class MetricReport:
    """Everything measured for one rubric (or for the pooled reward) across the fixture.

    Attributes:
        label: ``Q1`` / ``Q2`` / ``Q1Q2`` / ...
        questionnaire_id: The rubric's id, or ``None`` for the pooled row.
        is_pooled: True for the pooled training-reward row.
        n_total: Conversations attempted.
        n_ok: Conversations that produced a valid score.
        n_fail: ``n_total - n_ok``.
        schema_valid_rate: ``n_ok / n_total``. For the pooled row this is the fraction of
            conversations that would have yielded a usable REWARD, i.e. where every rubric
            validated -- a stricter and more training-relevant number than any single rubric's.
        mean, sd: Sample mean and sample SD (ddof=1) over the valid scores. ``sd`` is ``None``
            below two valid scores.
        reference_mean, reference_sd: The same statistics over the frozen reference, for the
            items that carry one. Printed beside the observed pair as the yardstick that
            :data:`MIN_SCORE_SD` was chosen against -- NOT as a target.
        level_offset: Mean signed ``observed - reference``. Expected to be nonzero; see
            :attr:`mean_abs_offset`.
        mean_abs_offset: Mean ``|observed - reference|``. SOFT. Two graders on different axes
            differ by a constant plus noise, and this number cannot separate the two.
        spearman: Rank agreement with the reference. SOFT.
        n_paired: Complete (observed, reference) pairs behind the soft numbers.
        attempts_total: Oracle calls spent on this row, retries included. **The pooled row sums
            the same calls the rubric rows already counted**, so it must not be added to them.
        calls_per_conversation: Calls one clean conversation costs on this row -- 1 for a rubric,
            one per rubric for the pooled row. Only there to make :attr:`n_retries` correct on
            both.
        degenerate: ``sd`` is missing or below :data:`MIN_SCORE_SD`.
        observations: The per-conversation rows.
    """

    label: str
    questionnaire_id: Optional[int]
    is_pooled: bool
    n_total: int
    n_ok: int
    n_fail: int
    schema_valid_rate: float
    mean: Optional[float]
    sd: Optional[float]
    reference_mean: Optional[float]
    reference_sd: Optional[float]
    level_offset: Optional[float]
    mean_abs_offset: Optional[float]
    spearman: Optional[float]
    n_paired: int
    attempts_total: int
    calls_per_conversation: int
    degenerate: bool
    observations: Tuple[Observation, ...]

    @property
    def n_retries(self) -> int:
        """Oracle calls beyond the one-per-conversation-per-rubric floor.

        A nonzero count on an otherwise passing report is the early warning for failure mode 1: the
        grader is drifting off-schema and the validation ladder is catching it. Watch it -- retries
        are the only thing that moves before missingness starts.
        """
        return max(0, self.attempts_total - self.n_total * max(1, self.calls_per_conversation))

    @classmethod
    def build(cls,
              label: str,
              questionnaire_id: Optional[int],
              observations: Sequence[Observation],
              *,
              is_pooled: bool = False,
              calls_per_conversation: int = 1) -> "MetricReport":
        """Reduce per-conversation observations to one rubric row."""
        obs = tuple(observations)
        n_total = len(obs)
        valid = [o for o in obs if o.score is not None]
        scores = [float(o.score) for o in valid]

        refs = [float(o.reference) for o in obs if o.reference is not None]
        paired = [(float(o.score), float(o.reference))
                  for o in obs if o.score is not None and o.reference is not None]

        sd = stdev(scores) if len(scores) >= 2 else None
        return cls(
            label=label,
            questionnaire_id=questionnaire_id,
            is_pooled=is_pooled,
            n_total=n_total,
            n_ok=len(valid),
            n_fail=n_total - len(valid),
            schema_valid_rate=(len(valid) / n_total) if n_total else 0.0,
            mean=fmean(scores) if scores else None,
            sd=sd,
            reference_mean=fmean(refs) if refs else None,
            reference_sd=stdev(refs) if len(refs) >= 2 else None,
            level_offset=fmean([s - r for s, r in paired]) if paired else None,
            mean_abs_offset=fmean([abs(s - r) for s, r in paired]) if paired else None,
            spearman=spearman([o.score for o in obs], [o.reference for o in obs]),
            n_paired=len(paired),
            attempts_total=sum(int(o.attempts) for o in obs),
            calls_per_conversation=max(1, int(calls_per_conversation)),
            degenerate=(sd is None or sd < MIN_SCORE_SD),
            observations=obs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "questionnaire_id": self.questionnaire_id,
            "is_pooled": self.is_pooled,
            "n_total": self.n_total,
            "n_ok": self.n_ok,
            "n_fail": self.n_fail,
            "schema_valid_rate": self.schema_valid_rate,
            "mean": self.mean,
            "sd": self.sd,
            "reference_mean": self.reference_mean,
            "reference_sd": self.reference_sd,
            "level_offset": self.level_offset,
            "mean_abs_offset": self.mean_abs_offset,
            "spearman": self.spearman,
            "n_paired": self.n_paired,
            "attempts_total": self.attempts_total,
            "calls_per_conversation": self.calls_per_conversation,
            "n_retries": self.n_retries,
            "degenerate": self.degenerate,
            "observations": [o.to_dict() for o in self.observations],
        }


@dataclass(frozen=True)
class SanityReport:
    """The full result: what was asked, what came back, and whether it is fit to train against.

    Attributes:
        model / provider / base_url: The grader that was probed. Recorded because an archived
            report next to ``run_metadata.json`` must say what it tested, not just the verdict.
        questionnaire_ids: Rubrics requested.
        quick: Whether this was the two-conversation pre-flight (weaker spread gate).
        metrics: One :class:`MetricReport` per rubric, plus the pooled row last when more than one
            rubric was requested.
        min_score_sd: The threshold in force when this report was produced, stored so an archived
            report explains its own verdict even after the constant changes.
        strict_json_schema: Whether ``strict: true`` was sent (see
            ``core.oracle.set_openai_compat_strict``). A grader that only passes with strict off is
            being constrained by the SERVER, not following the rubric.
    """

    schema_version: str
    started_at: str
    elapsed_s: float
    model: str
    provider: str
    base_url: Optional[str]
    questionnaire_ids: Tuple[int, ...]
    quick: bool
    fixture_path: str
    fixture_built_at: str
    reference_judge: str
    n_items: int
    max_tokens: int
    max_retries: int
    request_timeout: float
    strict_json_schema: bool
    min_score_sd: float
    metrics: Tuple[MetricReport, ...]
    items: Tuple[FixtureItem, ...]

    @property
    def per_rubric(self) -> Tuple[MetricReport, ...]:
        """The rubric rows, excluding the pooled reward."""
        return tuple(m for m in self.metrics if not m.is_pooled)

    @property
    def pooled(self) -> Optional[MetricReport]:
        """The pooled training-reward row, or ``None`` when a single rubric was requested."""
        for m in self.metrics:
            if m.is_pooled:
                return m
        return None

    @property
    def primary(self) -> Optional[MetricReport]:
        """The row a human should read first: the pooled reward, else the only rubric."""
        return self.pooled or (self.metrics[0] if self.metrics else None)

    @property
    def total_calls(self) -> int:
        """Oracle HTTP calls spent, retries included.

        Summed over the rubric rows only -- the pooled row re-counts the same calls."""
        return sum(m.attempts_total for m in self.per_rubric)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable payload, gate verdict included, for ``--out``."""
        passed, reasons = check_gates(self)
        return {
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s, 2),
            "grader": {"model": self.model, "provider": self.provider, "base_url": self.base_url},
            "questionnaire_ids": list(self.questionnaire_ids),
            "quick": self.quick,
            "fixture": {"path": self.fixture_path, "built_at": self.fixture_built_at,
                        "reference_judge": self.reference_judge, "n_items": self.n_items,
                        "item_ids": [it.id for it in self.items]},
            "oracle": {"max_tokens": self.max_tokens, "max_retries": self.max_retries,
                       "request_timeout": self.request_timeout,
                       "strict_json_schema": self.strict_json_schema},
            "gates": {"passed": passed, "failures": reasons,
                      "min_score_sd": self.min_score_sd,
                      "required_schema_valid_rate": REQUIRED_SCHEMA_VALID_RATE},
            "metrics": [m.to_dict() for m in self.metrics],
            "total_calls": self.total_calls,
        }


# ==============================================================================
#                              THE PROBE
# ==============================================================================


def _preflight(binding: RoleBinding, timeout: float) -> None:
    """Fail fast when a local endpoint is not answering -- or is serving ANOTHER model.

    Raises:
        RuntimeError: the endpoint did not answer within *timeout*, or it answered ``/models``
            with entries none of which is ``binding.model``.

    Notes:
        Without the readiness half, an unreachable server produces a full sweep of connection
        failures that look exactly like schema failures -- ``n_fail == n_total`` on every rubric
        -- and the report would read as "this grader cannot follow the schema" when the truth is
        "nothing was listening". The gate cannot tell those apart after the fact, because
        ``core.oracle.get_evaluation_json`` deliberately converts every transport error into a
        failed call.

        The model half exists because the archived report is titled with ``binding.model``. A
        vLLM server serving a different model 404s every request (it only knows its own id), so
        again every call fails -- but an OpenAI-compatible server that routes any ``model`` to
        whatever it has loaded would grade the fixture with the WRONG grader and produce a
        clean report about a model the server never ran. ``tools.vllm_serve._model_ids_match``
        is the same equality ``adopt_if_running`` uses, so this refuses exactly what that
        refuses. A server that lists no models at all is left to the sweep (nothing to compare).

        Synchronous and blocking inside the async caller on purpose: it runs before any task is
        scheduled, so there is nothing for it to block.
    """
    if not (binding.is_local and binding.base_url):
        return
    if timeout > 0:
        try:
            wait_until_ready(binding.base_url, timeout=timeout, poll_seconds=2.0)
        except Exception as exc:
            raise RuntimeError(
                f"Oracle endpoint {binding.base_url} did not answer within {timeout:.0f}s ({exc}). "
                f"Start the server first (tools/vllm_serve.py: serve_roles/start_server) and "
                f"re-run. Without this check the sweep would report 100% schema failure for a "
                f"server that is simply not there."
            ) from exc

    served = _served_model_ids(_get_json(_models_url(binding.base_url)))
    if served and not any(_model_ids_match(binding.model, sid) for sid in served):
        raise RuntimeError(
            f"Oracle endpoint {binding.base_url} serves {served!r}, not {binding.model!r}. "
            f"Refusing to score: the report would be filed under a model the server never ran. "
            f"Pass --model with the served id (or --base-url with the right server)."
        )


async def run_sanity(binding: RoleBinding,
                     *,
                     questionnaire_ids: Sequence[int] = (1, 2),
                     items: Optional[Sequence[FixtureItem]] = None,
                     quick: bool = False,
                     fixture_path: Optional[str] = None,
                     concurrency: int = 8,
                     max_tokens: int = 256,
                     max_retries: int = 3,
                     request_timeout: float = 120.0,
                     preflight_timeout: float = 20.0,
                     progress: bool = True) -> SanityReport:
    """Score the fixture with *binding*'s grader and reduce it to a fitness report.

    Args:
        binding: The grader to test. **Bind it exactly as the trainer will** -- same model, same
            endpoint, same provider. Testing a different binding proves nothing about the run.
        questionnaire_ids: Rubrics to check. ``(1, 2)`` is Q1+Q2, the default training reward.
        items: Fixture items to use. Defaults to the committed fixture (see :func:`load_fixture`).
        quick: Score only :data:`QUICK_N_ITEMS` conversations, taken from the ends of the reference
            quality range. This is the pre-flight the trainer notebooks run before iteration 1; the
            spread gate is genuinely weaker at n=2 (see :func:`select_spanning`).
        fixture_path: Override the fixture file. Ignored when *items* is given.
        concurrency: In-flight oracle calls. The bound is honest -- it is the semaphore
            ``core.oracle`` itself acquires -- so keep it near what the trainer will use rather
            than maximal, or a green report may hide a server that falls over under real load.
        max_tokens: Oracle response budget. **Keep this equal to the trainer's setting.** A gate
            run at a larger budget than the run tests a different configuration: a local model that
            opens with a preamble can fit the JSON at 512 and clip it at 256, which surfaces only as
            retries and missingness during training.
        max_retries: Attempts per (conversation, rubric) call, including the first.
        request_timeout: Per-attempt ceiling in seconds, enforced by ``asyncio.wait_for``.
        preflight_timeout: Seconds to wait for a local endpoint to answer before scoring; 0
            disables the check.
        progress: Print one line per completed call.

    Returns:
        A :class:`SanityReport`. **It does not raise on a failing grader** -- an unfit grader is a
        result, not an exception. Call :func:`check_gates` (or :func:`main`) to act on it.

    Raises:
        ValueError: an ``openai_compat`` binding with no ``base_url``, an unknown rubric id, or an
            empty item list.
        RuntimeError: a local endpoint that does not answer (see :func:`_preflight`).

    Notes:
        **The missing-base_url guard is not pedantry.** ``RoleBinding.base_url`` is normally filled
        in by ``tools.vllm_serve.serve_roles`` once a port exists; an ``openai_compat`` binding that
        still has ``None`` builds an OpenAI SDK client pointed at ``api.openai.com``. The gate would
        then quietly grade the fixture with a VENDOR model -- billing a run that exists to cost $0,
        and returning a glowing report about a grader that is not the one the trainer will use.

        Scoring goes through ``core.oracle.get_evaluation_json`` one (conversation, rubric) call at
        a time, so each rubric's compliance is measured independently;
        ``core.oracle.score_conversation`` would short-circuit on the first failing rubric and hide
        the rest. The pooled row is then reduced exactly as that function documents -- the
        unweighted mean across rubrics, counted only where every rubric validated.
    """
    qids = _validate_questionnaire_ids(questionnaire_ids)

    if binding.is_local and not binding.base_url:
        raise ValueError(
            "run_sanity: the oracle binding is 'openai_compat' but carries no base_url. The OpenAI "
            "SDK would default to api.openai.com, so this gate would grade the fixture with a "
            "vendor model instead of the local server -- a passing report about the wrong grader. "
            "Pass base_url=, or run the binding through tools.vllm_serve.serve_roles first."
        )

    fixture: Optional[Fixture] = None
    if items is None:
        fixture = load_fixture(fixture_path)
        selected = tuple(sorted(fixture.items, key=lambda it: it.sort_value))
    else:
        selected = tuple(items)
    if quick:
        selected = select_spanning(selected, QUICK_N_ITEMS)
    if not selected:
        raise ValueError("run_sanity: no fixture items to score.")

    cfg = OracleConfig(
        binding=binding,
        questionnaire_ids=qids,
        eval_temperature=0.0,
        max_tokens=int(max_tokens),
        max_retries=int(max_retries),
        request_timeout=float(request_timeout),
        max_concurrency=int(concurrency),
    )
    _preflight(binding, preflight_timeout)

    client = make_oracle_client(cfg)
    # patient_concurrency is unused here (the gate makes no patient calls) but must be positive.
    primitives = AsyncPrimitives(oracle_concurrency=int(concurrency), patient_concurrency=1)

    tasks_meta: List[Tuple[FixtureItem, int]] = [(it, q) for it in selected for q in qids]
    total = len(tasks_meta)
    done = {"n": 0}
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()

    if progress:
        mode = "quick" if quick else "full"
        print(f"[sanity] {mode}: {len(selected)} transcripts x {len(qids)} rubric(s) = {total} "
              f"oracle calls against {binding.provider}:{binding.model}")

    async def _one(item: FixtureItem, qid: int) -> Tuple[str, int, Optional[float], int]:
        data, _n_questions, attempts = await get_evaluation_json(
            client, cfg, primitives, item.transcript, qid
        )
        score = None if data is None else float(data["mean_score"])
        done["n"] += 1
        if progress:
            label = metric_label(qid)
            ref = item.reference_for(label)
            print(f"  [sanity] {done['n']:>3}/{total} {label:<6} {item.id:<26} "
                  f"score {'  n/a' if score is None else f'{score:5.2f}'} "
                  f"ref {'  n/a' if ref is None else f'{ref:5.2f}'} attempts {attempts}")
        return item.id, qid, score, attempts

    results = await asyncio.gather(*(_one(item, qid) for item, qid in tasks_meta))
    elapsed = time.time() - t0

    by_key: Dict[Tuple[str, int], Tuple[Optional[float], int]] = {
        (item_id, qid): (score, attempts) for item_id, qid, score, attempts in results
    }

    metrics: List[MetricReport] = []
    for qid in qids:
        label = metric_label(qid)
        obs = []
        for item in selected:
            score, attempts = by_key.get((item.id, qid), (None, 0))
            obs.append(Observation(item_id=item.id, score=score,
                                   reference=item.reference_for(label), attempts=attempts))
        metrics.append(MetricReport.build(label, qid, obs))

    if len(qids) > 1:
        pooled_label = combined_label(qids)
        pooled_obs = []
        for item in selected:
            per_rubric = [by_key.get((item.id, q), (None, 0)) for q in qids]
            scores = [s for s, _ in per_rubric]
            attempts = sum(a for _, a in per_rubric)
            # Mirrors core.oracle.score_conversation: one failed rubric voids the candidate, it
            # does not average the survivors -- otherwise the reward's DEFINITION would depend on
            # which calls happened to fail.
            complete = all(s is not None for s in scores)
            pooled = fmean([float(s) for s in scores]) if complete else None
            pooled_obs.append(Observation(item_id=item.id, score=pooled,
                                          reference=item.reference_for(pooled_label),
                                          attempts=attempts))
        metrics.append(MetricReport.build(pooled_label, None, pooled_obs, is_pooled=True,
                                          calls_per_conversation=len(qids)))

    fixture_label = fixture.path if fixture is not None else (fixture_path or "(caller-supplied)")
    return SanityReport(
        schema_version=SCHEMA_VERSION,
        started_at=started_at,
        elapsed_s=elapsed,
        model=binding.model,
        provider=binding.provider,
        base_url=binding.base_url,
        questionnaire_ids=qids,
        quick=bool(quick),
        fixture_path=fixture_label,
        fixture_built_at=(fixture.built_at if fixture is not None else ""),
        reference_judge=(fixture.reference_judge if fixture is not None else ""),
        n_items=len(selected),
        max_tokens=int(max_tokens),
        max_retries=int(max_retries),
        request_timeout=float(request_timeout),
        # OpenAI always gets strict:true; openai_compat only when the module-level flag is on.
        strict_json_schema=bool(binding.provider == "openai" or openai_compat_strict()),
        min_score_sd=MIN_SCORE_SD,
        metrics=tuple(metrics),
        items=selected,
    )


# ==============================================================================
#                                 GATES
# ==============================================================================


def check_gates(report: SanityReport) -> Tuple[bool, List[str]]:
    """Apply the HARD gates. Returns ``(passed, reasons)``; ``reasons`` is empty when passed.

    Args:
        report: What :func:`run_sanity` returned.

    Returns:
        ``passed`` is True only when every hard gate holds. ``reasons`` holds one human-readable
        line per failure, ordered rubric by rubric, each naming the number that failed and the
        threshold it failed against.

    Notes:
        Two gates, and only two:

        1. **Schema compliance must be perfect** on every requested rubric
           (:data:`REQUIRED_SCHEMA_VALID_RATE`). Checked on the rubric rows only -- the pooled row's
           failures are implied by them, and reporting both would print one problem twice.
        2. **Per-conversation spread must clear :data:`MIN_SCORE_SD`** on every rubric AND on the
           pooled reward. The pooled row is checked separately because it is the quantity the policy
           actually optimizes: two rubrics can each vary while their mean does not.

        Spearman and the level offset are deliberately NOT gates. Exp3 and Exp4 are not on the same
        score axis, so an offset is expected and uninformative, and rho over a dozen conversations
        is too weak to block a run on. They belong in the report, where a human weighs them.

        An empty report fails: a gate with nothing to check has not passed, it has not run.
    """
    reasons: List[str] = []

    if not report.metrics or report.n_items == 0:
        return False, ["[empty] no rubric was scored -- the gate did not run, so it did not pass."]

    for metric in report.per_rubric:
        if metric.n_total == 0:
            reasons.append(f"[schema] {metric.label}: no conversations were scored.")
            continue
        if metric.n_fail > 0:
            reasons.append(
                f"[schema] {metric.label}: {metric.n_fail}/{metric.n_total} conversations never "
                f"returned a valid response (schema_valid_rate "
                f"{metric.schema_valid_rate:.3f} < {REQUIRED_SCHEMA_VALID_RATE:.3f}). Partial "
                f"compliance is missingness correlated with conversation difficulty, not a "
                f"slightly noisier grader. Check thinking mode is off, then max_tokens, then "
                f"whether the server accepts strict json_schema."
            )

    for metric in report.metrics:
        scope = "pooled training reward" if metric.is_pooled else "rubric"
        if metric.sd is None:
            reasons.append(
                f"[spread] {metric.label} ({scope}): fewer than 2 valid scores, so spread cannot "
                f"be evaluated. A grader that cannot be checked for degeneracy is not cleared."
            )
        elif metric.n_ok < MIN_N_FOR_SPREAD_GATE:
            # Reported by format_report, deliberately NOT blocking: see MIN_N_FOR_SPREAD_GATE.
            continue
        elif metric.sd < MIN_SCORE_SD:
            ref_sd = "n/a" if metric.reference_sd is None else f"{metric.reference_sd:.2f}"
            reasons.append(
                f"[spread] {metric.label} ({scope}): sd {metric.sd:.3f} < {MIN_SCORE_SD:.2f} over "
                f"{metric.n_ok} conversations the reference spread at sd {ref_sd}. This grader is "
                f"answering from a template, not reading -- it would produce valid parquet, "
                f"contrast tables at ~0, and no error anywhere."
            )

    return (not reasons), reasons


# ==============================================================================
#                                REPORTING
# ==============================================================================


def _fmt(value: Optional[float], precision: int = 2) -> str:
    """Number, or ``n/a`` -- never a fabricated 0.0 for something that was not measured."""
    return "n/a" if value is None else f"{value:.{precision}f}"


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Fixed-width ASCII table: first column left-aligned, the rest right-aligned."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: Sequence[str]) -> str:
        out = [cells[0].ljust(widths[0])]
        out += [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
        return "  ".join(out).rstrip()

    lines = [_line(headers), "  ".join("-" * w for w in widths)]
    lines += [_line(r) for r in rows]
    return lines


def format_report(report: SanityReport) -> str:
    """Render *report* as an ASCII block for a notebook cell or a terminal.

    Args:
        report: What :func:`run_sanity` returned.

    Returns:
        A multi-line ASCII string: provenance header, per-rubric table, per-conversation table on
        the reward axis, the hard-gate verdict, and the soft numbers with the paragraph explaining
        why they are soft.

    Notes:
        The soft/hard split is spelled out in the output itself, not just in this file. The report
        is archived next to ``run_metadata.json`` and read months later by someone deciding whether
        an arm is trustworthy, and the single most likely misreading is treating the level offset
        as a defect -- it is the expected consequence of a different grader on a different axis.
    """
    passed, reasons = check_gates(report)
    lines: List[str] = []
    rule = "=" * 88

    lines.append(rule)
    lines.append(f"Exp4 oracle sanity -- {report.provider}:{report.model}")
    lines.append("-" * 88)
    shape = f"{report.n_items} transcripts x {len(report.questionnaire_ids)} rubric(s)"
    mode = (f"quick -- {shape}, spread gate is weaker at this size" if report.quick
            else f"full -- {shape}")
    header_rows = [
        ("endpoint", report.base_url or "(provider default)"),
        ("rubrics", ", ".join(f"{metric_label(q)} (id {q})" for q in report.questionnaire_ids)),
        ("mode", mode),
        ("fixture", f"{report.fixture_path} (built {report.fixture_built_at or 'unknown'})"),
        ("reference", f"{report.reference_judge or 'unknown'} -- a DIFFERENT axis, see below"),
        ("oracle call", (f"max_tokens={report.max_tokens} max_retries={report.max_retries} "
                         f"timeout={report.request_timeout:.0f}s "
                         f"strict_json_schema={report.strict_json_schema}")),
        ("calls", f"{report.total_calls} in {report.elapsed_s:.1f}s (started {report.started_at})"),
    ]
    width = max(len(label) for label, _ in header_rows)
    lines += [f"{label.ljust(width)} : {value}" for label, value in header_rows]

    lines.append("")
    lines.append("PER RUBRIC")
    headers = ["metric", "n_ok", "n_fail", "valid%", "mean", "sd", "ref_mean", "ref_sd",
               "offset", "|offset|", "rho", "calls"]
    rows: List[List[str]] = []
    for metric in report.metrics:
        rows.append([
            metric.label + ("*" if metric.is_pooled else ""),
            str(metric.n_ok),
            str(metric.n_fail),
            f"{100.0 * metric.schema_valid_rate:.1f}",
            _fmt(metric.mean),
            _fmt(metric.sd),
            _fmt(metric.reference_mean),
            _fmt(metric.reference_sd),
            _fmt(metric.level_offset),
            _fmt(metric.mean_abs_offset),
            _fmt(metric.spearman),
            str(metric.attempts_total),
        ])
    lines += _render_table(headers, rows)
    if report.pooled is not None:
        lines += [
            "* the pooled TRAINING REWARD: unweighted mean across rubrics per conversation,",
            "  counted only where every rubric validated (core.oracle.score_conversation).",
            "  Its calls column re-counts the rubric rows above; total spend is on the",
            "  'calls' header line, not the sum of this column.",
        ]

    primary = report.primary
    if primary is not None:
        by_id = {it.id: it for it in report.items}
        lines.append("")
        lines.append(f"PER CONVERSATION (axis: {primary.label})")
        conv_headers = ["id", "source", "utt", "ref", "obs", "delta"]
        conv_rows: List[List[str]] = []
        def _quality(obs: Observation) -> float:
            item = by_id.get(obs.item_id)
            return item.sort_value if item is not None else 0.0

        for obs in sorted(primary.observations, key=_quality):
            item = by_id.get(obs.item_id)
            delta = (None if obs.score is None or obs.reference is None
                     else obs.score - obs.reference)
            conv_rows.append([
                obs.item_id,
                item.source_model_state if item else "",
                str(item.n_utterances) if item else "",
                _fmt(obs.reference),
                _fmt(obs.score),
                ("n/a" if delta is None else f"{delta:+.2f}"),
            ])
        lines += _render_table(conv_headers, conv_rows)

    lines.append("")
    lines.append("HARD GATES (block the run)")

    # Rubrics whose spread was measured but deliberately not enforced, because at this sample size
    # the check cannot tell a template apart from two conversations that simply scored alike.
    unenforced = [m for m in report.metrics
                  if m.sd is not None and m.n_ok < MIN_N_FOR_SPREAD_GATE]

    if passed:
        pooled_note = " and on the pooled reward" if report.pooled is not None else ""
        lines.append(f"  [PASS] schema_valid_rate == {REQUIRED_SCHEMA_VALID_RATE:.3f} "
                     f"on every rubric")
        if unenforced:
            lines.append(f"  [n/a ] per-conversation sd NOT enforced: only {unenforced[0].n_ok} "
                         f"scored conversation(s), below the {MIN_N_FOR_SPREAD_GATE} this check "
                         f"needs to mean anything")
        else:
            lines.append(f"  [PASS] per-conversation sd >= {MIN_SCORE_SD:.2f} on every rubric"
                         f"{pooled_note}")
    else:
        for reason in reasons:
            lines.append(f"  [FAIL] {reason}")

    if unenforced:
        low = ", ".join(f"{m.label} sd {m.sd:.3f}" for m in unenforced)
        lines.append(f"         spread observed but not enforced ({low}). A degenerate grader can "
                     f"pass at this size -- run the FULL gate (drop --quick) before any real arm.")

    lines.append("")
    lines.append("SOFT (reported, never blocking)")
    for metric in report.metrics:
        lines.append(f"  {metric.label:<8} rho {_fmt(metric.spearman)} over {metric.n_paired} "
                     f"paired conversations; level offset {_fmt(metric.level_offset)}; "
                     f"{metric.n_retries} retry call(s)")
    lines += [
        "  Why soft: the reference is what gpt-4o-mini gave these same transcripts in",
        "  Exp3, and Exp4 is NOT on that axis. A different grader with a different prior",
        "  over the rubric sits systematically higher or lower, so an offset of a point or",
        "  more is EXPECTED and says nothing about fitness. What survives the axis change",
        "  is the ORDERING (rho) and the SPREAD (sd). Spread is hard because a grader",
        "  without it cannot measure; ordering is soft because rho over a dozen",
        "  conversations is weak evidence, and the reference is itself one draw from a",
        "  stochastic judge.",
    ]

    verdict = "PASS -- fit to train against" if passed else "FAIL -- do not spend on this grader"
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append(rule)
    return "\n".join(lines)


def write_report(report: SanityReport, path: str) -> str:
    """Write *report* as JSON and return the path written.

    Args:
        report: What :func:`run_sanity` returned.
        path: Destination. **A path that does not end in ``.json`` is treated as a DIRECTORY** and
            gets :data:`REPORT_FILENAME` written inside it (creating it if needed). The directory
            form is the intended one: point it at an arm's run folder so the report lands beside
            ``run_metadata.json`` and travels with the run.

    Returns:
        The absolute path written.

    Notes:
        The suffix rule exists because the run folder may not exist yet when the pre-flight runs,
        so ``os.path.isdir`` cannot be the test -- and silently writing an extension-less FILE
        called ``GRPO4_Q1Q2_LA0_...`` where a folder was meant is the kind of thing nobody notices
        until they look for the report and find a directory that is a file.

        The payload carries the gate verdict, the thresholds in force, every per-conversation
        observation and the fixture's provenance, so it is readable without this module and stays
        interpretable if :data:`MIN_SCORE_SD` is ever changed.
    """
    target = os.path.abspath(path)
    if os.path.isdir(target) or not os.path.basename(target).lower().endswith(".json"):
        target = os.path.join(target, REPORT_FILENAME)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
    return target


# ==============================================================================
#                 THE ORACLE PROMPT LENGTH -- Phase 2: does it FIT?
# ==============================================================================
#
# Measured on REAL conversations, never on the fixture: the question is what the Exp4 patient
# makes the therapist's sessions look like, and the fixture is Exp3 text. Token counts come from
# the served model's own tokenizer through vLLM's /tokenize endpoint whenever a server is
# reachable, because the number that matters is the one the SERVER computes when it decides
# whether the request fits. The fallbacks (a local HF tokenizer, then chars/3.5) are labelled in
# the report so nobody quotes an estimate as a measurement.


@dataclass(frozen=True)
class PromptLengthRow:
    """One (conversation, rubric) prompt and how long it is.

    Attributes:
        conversation_id: ``pers07`` -- the CSV stem.
        persona_id: The stable persona index.
        questionnaire_id: The rubric whose full prompt was measured.
        label: ``Q1`` / ``Q2`` / ...
        n_utterances: Conversation length (therapist + patient), to see length driving overflow.
        n_chars: Characters in the full prompt (rubric + transcript).
        n_tokens: Tokens in the full prompt, by ``method``.
        method: How ``n_tokens`` was obtained -- see :class:`PromptLengthReport`.
    """

    conversation_id: str
    persona_id: int
    questionnaire_id: int
    label: str
    n_utterances: int
    n_chars: int
    n_tokens: int
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {"conversation_id": self.conversation_id, "persona_id": self.persona_id,
                "questionnaire_id": self.questionnaire_id, "label": self.label,
                "n_utterances": self.n_utterances, "n_chars": self.n_chars,
                "n_tokens": self.n_tokens, "method": self.method}


@dataclass(frozen=True)
class PromptLengthReport:
    """The realised oracle-prompt length distribution over one conversations folder.

    Attributes:
        conv_dir: The folder measured (one model state), or a ``<N transcript(s) in memory>``
            label when :func:`prompt_length_report` was handed transcripts instead of a folder.
        questionnaire_ids: Rubrics whose full prompts were built.
        n_conversations: Conversations measured (also exported as ``n_transcripts``).
        max_model_len: The context length the prompts are checked against.
        max_model_len_source: Where that number came from -- ``"argument"``, ``"server:/tokenize"``,
            ``"server:/v1/models"`` or ``"assumed:ServeSpec.max_model_len"``. An ASSUMED cap is
            the spec's 16384, not the server's, and the report says so.
        method: How tokens were counted, the same for every row: ``"vllm:/tokenize"`` (the served
            tokenizer -- the measurement), ``"tokenizer:<id>"`` (a local HF tokenizer; the same
            numbers iff it is the served model's) or ``"estimate:chars/3.5"`` (a labelled
            over-count -- see :data:`CHARS_PER_TOKEN_ESTIMATE`).
        base_url / model: The endpoint and model the count was resolved against, for provenance.
        rows: One :class:`PromptLengthRow` per (conversation, rubric).
        notes: Anything the resolver had to fall back over (an unreachable server, a rejected
            ``/tokenize`` form), so the archived JSON explains its own ``method``.
    """

    conv_dir: str
    questionnaire_ids: Tuple[int, ...]
    n_conversations: int
    max_model_len: int
    max_model_len_source: str
    method: str
    base_url: Optional[str]
    model: Optional[str]
    rows: Tuple[PromptLengthRow, ...]
    notes: Tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        """True when the counts are the served tokenizer's, not a stand-in."""
        return self.method.startswith("vllm:")

    def rows_for(self, questionnaire_id: int) -> Tuple[PromptLengthRow, ...]:
        return tuple(r for r in self.rows if r.questionnaire_id == int(questionnaire_id))

    def summary(self, questionnaire_id: int) -> Dict[str, Any]:
        """``{n, median, p95, max, n_over, frac_over, headroom}`` for one rubric."""
        tokens = sorted(r.n_tokens for r in self.rows_for(questionnaire_id))
        if not tokens:
            return {"n": 0, "median": None, "p95": None, "max": None, "n_over": 0,
                    "frac_over": 0.0, "headroom": None}
        p95 = tokens[max(0, math.ceil(0.95 * len(tokens)) - 1)]
        n_over = sum(1 for t in tokens if t > self.max_model_len)
        return {
            "n": len(tokens),
            "median": median(tokens),
            "p95": p95,
            "max": tokens[-1],
            "n_over": n_over,
            "frac_over": n_over / len(tokens),
            # Fraction of the cap the longest prompt leaves unused; negative = overflow.
            "headroom": (self.max_model_len - tokens[-1]) / float(self.max_model_len),
        }

    @property
    def n_over(self) -> int:
        """Prompts, over every rubric, that exceed ``max_model_len``."""
        return sum(1 for r in self.rows if r.n_tokens > self.max_model_len)

    @property
    def max_tokens(self) -> int:
        """The longest measured prompt over every rubric (0 when nothing was measured)."""
        return max((r.n_tokens for r in self.rows), default=0)

    def to_dict(self) -> Dict[str, Any]:
        """The archived / gate-readable shape.

        Two spellings of the same facts are carried on purpose: ``per_rubric`` (keyed by metric
        label, per-rubric ``headroom`` = fraction of the cap the longest prompt leaves unused --
        what :func:`format_prompt_length_report` renders) and the flat keys
        ``eda_analysis.scoring.check_prompt_length_gate`` reads -- ``n_transcripts``,
        ``per_questionnaire`` (keyed by rubric id), ``n_over``, ``max_tokens`` (the longest
        prompt) and top-level ``headroom`` = ``max_model_len / max_tokens`` (a RATIO; None
        when nothing was measured). Renaming any of the flat keys makes that gate FAIL loudly.
        """
        ok, reasons = check_prompt_lengths(self)
        longest = self.max_tokens
        return {
            "schema_version": "exp4-oracle-prompt-lengths/1",
            "conv_dir": self.conv_dir,
            "questionnaire_ids": list(self.questionnaire_ids),
            "n_conversations": self.n_conversations,
            "n_transcripts": self.n_conversations,
            "max_model_len": self.max_model_len,
            "max_model_len_source": self.max_model_len_source,
            "method": self.method,
            "measured": self.measured,
            "grader": {"base_url": self.base_url, "model": self.model},
            "per_rubric": {metric_label(q): self.summary(q) for q in self.questionnaire_ids},
            "per_questionnaire": {str(int(q)): self.summary(q) for q in self.questionnaire_ids},
            "n_over": self.n_over,
            "max_tokens": longest,
            "headroom": (self.max_model_len / float(longest)) if longest else None,
            "gate": {"passed": ok, "failures": reasons},
            "notes": list(self.notes),
            "rows": [r.to_dict() for r in self.rows],
        }


def _tokenize_root(base_url: str) -> str:
    """vLLM mounts ``/tokenize`` at the server ROOT, beside ``/v1``, not under it."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/tokenize"


def _post_json(url: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    """POST JSON, return the parsed JSON body. Raises ``urllib.error.URLError``/``ValueError``."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read().decode("utf-8", errors="replace"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{url} answered a non-object body")
    return parsed


def _vllm_tokenize(base_url: str, model: Optional[str], text: str) -> Tuple[int, Optional[int]]:
    """``(count, max_model_len)`` from vLLM's ``/tokenize`` for the oracle's exact request shape.

    The oracle sends ``messages=[{"role": "user", "content": prompt}]``, and the server applies
    the model's chat template on top -- so the ``messages`` form is tried first (that is the
    count the server will compare against ``max_model_len``), and the raw ``prompt`` form only if
    the server rejects it. Raises on transport failure or an unusable body.
    """
    url = _tokenize_root(base_url)
    payloads: List[Dict[str, Any]] = [
        {"messages": [{"role": "user", "content": text}], "add_generation_prompt": True},
        {"prompt": text},
    ]
    last: Optional[BaseException] = None
    for payload in payloads:
        if model:
            payload["model"] = model
        try:
            parsed = _post_json(url, payload, timeout=_TOKENIZE_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            last = exc
            if 400 <= exc.code < 500:
                continue        # this form is not accepted; try the other
            raise
        count = parsed.get("count")
        if count is None and isinstance(parsed.get("tokens"), list):
            count = len(parsed["tokens"])
        if count is None:
            raise ValueError(f"{url} answered without a token count: {list(parsed)[:6]}")
        raw_len = parsed.get("max_model_len")
        try:
            served_len = int(raw_len) if raw_len is not None else None
        except (TypeError, ValueError):
            served_len = None
        return int(count), served_len
    raise RuntimeError(f"{url} rejected both request forms ({last})")


def _resolve_counter(base_url: Optional[str], model: Optional[str], tokenizer: Any,
                     notes: List[str]) -> Tuple[Callable[[str], int], str, Optional[int]]:
    """Pick the token counter: served ``/tokenize`` > HF tokenizer > chars/3.5 estimate.

    Returns ``(count_fn, method_label, served_max_model_len_or_None)``. Every fallback is
    recorded in *notes*, so the report can say why its method is not the measurement.
    """
    if base_url:
        try:
            _count, served_len = _vllm_tokenize(base_url, model, "probe")
            return (lambda text: _vllm_tokenize(base_url, model, text)[0],
                    "vllm:/tokenize", served_len)
        except Exception as exc:  # noqa: BLE001 - every failure means "fall back, and say so"
            notes.append(f"/tokenize at {base_url} unusable ({type(exc).__name__}: "
                         f"{str(exc)[:120]}); falling back")

    if tokenizer is None and model:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415 - optional, heavy

            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            notes.append(f"using the locally cached HF tokenizer for {model}")
        except Exception as exc:  # noqa: BLE001 - not cached / not installed: estimate instead
            notes.append(f"no local HF tokenizer for {model!r} ({type(exc).__name__}); "
                         f"estimating at {CHARS_PER_TOKEN_ESTIMATE} chars/token")
            tokenizer = None

    if tokenizer is not None:
        label = f"tokenizer:{getattr(tokenizer, 'name_or_path', None) or model or 'given'}"

        def _count_hf(text: str) -> int:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(tokenizer.encode(text))

        return _count_hf, label, None

    return (lambda text: int(math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)),
            f"estimate:chars/{CHARS_PER_TOKEN_ESTIMATE}", None)


#: One ``[THERAPIST]:`` / ``[PATIENT]:`` label at a line start = one utterance of an in-memory
#: transcript (``format_conversation_for_oracle`` output). Unlabelled ``\n\n`` segments are
#: continuations of the previous turn (``core.conversations.parse_transcript_to_messages``).
_TURN_LABEL_RE = re.compile(r"(?m)^\[(?:THERAPIST|PATIENT)\]: ")


def _count_labelled_turns(transcript: str) -> int:
    return len(_TURN_LABEL_RE.findall(transcript))


def prompt_length_report(source: Union[str, Sequence[str]],
                         questionnaire_ids: Sequence[int] = (1, 2),
                         *,
                         base_url: Optional[str] = None,
                         tokenizer: Any = None,
                         model: Optional[str] = None,
                         max_model_len: Optional[int] = None,
                         progress: bool = False) -> PromptLengthReport:
    """Measure the full oracle prompt for every conversation in *source*, per rubric.

    Args:
        source: EITHER a model-state folder of ``pers*.csv`` conversations
            (``core.conversations`` layout; read with ``load_conversations_dir``, so an
            unreadable file is skipped and counted as absent, never as short) OR a sequence of
            oracle-formatted transcripts already in memory (``format_conversation_for_oracle``
            output -- what ``eda_analysis.scoring.gather_transcripts`` yields for
            ``Run_Eval`` § 8, the Phase 2 gate over EVERY arm on disk). In-memory transcripts
            are labelled ``transcript_<i>`` with ``persona_id -1``, and their ``n_utterances``
            is the count of turn labels.
        questionnaire_ids: Rubrics to build prompts for. ``(1, 2)`` is the training reward.
        base_url: A local OpenAI-compatible endpoint. When given, tokens are counted by the
            server's own tokenizer via vLLM's ``/tokenize`` -- the measurement -- and the served
            ``max_model_len`` is read from the same answer (or from ``/v1/models``).
        tokenizer: A HF tokenizer to count with instead (used only if ``/tokenize`` is not
            available). Counts agree with the server's iff it is the served model's tokenizer.
        model: The served model id, sent with ``/tokenize`` and used to look for a cached HF
            tokenizer when neither *base_url* nor *tokenizer* yields a count.
        max_model_len: The cap the prompts are checked against when no server advertises one.
            When a served cap IS read (``/tokenize`` or ``/v1/models``) and differs from this,
            the SMALLER of the two is used and the report notes both -- the server decides
            what fits, so a larger argument must not override the real cap, and a smaller one
            is a deliberate tighter budget. With neither, ``ServeSpec.max_model_len`` (16384),
            labelled ``assumed`` in the report because it is the spec's number, not the
            server's.
        progress: Print one line per conversation.

    Returns:
        A :class:`PromptLengthReport`. **It does not raise on overflow** -- call
        :func:`check_prompt_lengths` to gate on it. It does raise when there is nothing to
        measure.

    Raises:
        ValueError: *source* holds no readable conversations / no transcripts, or a rubric id
            is unknown.

    Notes:
        The prompt measured is exactly what ``core.oracle.get_evaluation_json`` sends:
        ``get_prompt_eval_questionnaire(qid, transcript)["prompt"]`` over
        ``format_conversation_for_oracle(state.turns)``. There is no second prompt builder here.

        Cheap: one ``/tokenize`` call per (conversation, rubric), 192 for a full state -- a
        few seconds against an idle server, and zero model compute.
    """
    from core.conversations import format_conversation_for_oracle, load_conversations_dir

    qids = _validate_questionnaire_ids(questionnaire_ids)
    # (conversation_id, persona_id, n_utterances, transcript) per conversation, either source.
    items: List[Tuple[str, int, int, str]]
    if isinstance(source, str):
        states = load_conversations_dir(source)
        if not states:
            raise ValueError(f"prompt_length_report: no readable conversations in {source!r}")
        items = [(state.conversation_id, int(state.persona_id), int(state.n_utterances),
                  format_conversation_for_oracle(state.turns))
                 for state in (states[pid] for pid in sorted(states))]
        label = os.path.abspath(source)
    else:
        texts = [str(t) for t in source]
        if not texts:
            raise ValueError("prompt_length_report: no transcripts to measure")
        items = [(f"transcript_{i:04d}", -1, _count_labelled_turns(t), t)
                 for i, t in enumerate(texts)]
        label = f"<{len(texts)} transcript(s) in memory>"

    notes: List[str] = []
    count, method, served_len = _resolve_counter(base_url, model, tokenizer, notes)

    # The served cap, from whichever surface answered: /tokenize carries it beside the count,
    # /v1/models (vLLM's model card) is the fallback. None when no server advertised one.
    if served_len is not None:
        served_cap, served_source = int(served_len), "server:/tokenize"
    else:
        advertised = served_max_model_len(base_url, model=model) if base_url else None
        served_cap = int(advertised) if advertised is not None else None
        served_source = "server:/v1/models"

    if max_model_len is not None and served_cap is not None and int(max_model_len) != served_cap:
        # Both an explicit cap (the notebook's VLLM_MAX_MODEL_LEN, say) and a served one exist
        # and disagree. The SERVER decides whether a request fits, so the smaller is the one a
        # prompt has to clear -- and a caller's larger number must not silently override the
        # real cap (the gate would pass prompts the server rejects), nor a caller's smaller one
        # be discarded (it may be a deliberate tighter budget).
        cap = min(int(max_model_len), served_cap)
        cap_source = f"min(argument {int(max_model_len)}, {served_source} {served_cap})"
        notes.append(f"explicit max_model_len={int(max_model_len)} differs from the served cap "
                     f"{served_cap} ({served_source}); checking against the SMALLER, {cap}")
    elif max_model_len is not None:
        cap, cap_source = int(max_model_len), "argument"
    elif served_cap is not None:
        cap, cap_source = served_cap, served_source
    else:
        cap, cap_source = int(ServeSpec(model=model or DEFAULT_ORACLE_MODEL).max_model_len), \
            "assumed:ServeSpec.max_model_len"
        notes.append("max_model_len ASSUMED from ServeSpec (no server advertised one); "
                     "the served cap may differ")

    rows: List[PromptLengthRow] = []
    for i, (conv_id, persona_id, n_utterances, transcript) in enumerate(items, start=1):
        for qid in qids:
            prompt = get_prompt_eval_questionnaire(questionnaire=qid, conversation=transcript)["prompt"]
            n_tokens = int(count(prompt))
            rows.append(PromptLengthRow(
                conversation_id=conv_id, persona_id=persona_id,
                questionnaire_id=int(qid), label=metric_label(qid),
                n_utterances=n_utterances, n_chars=len(prompt),
                n_tokens=n_tokens, method=method,
            ))
        if progress:
            longest = max(r.n_tokens for r in rows[-len(qids):])
            print(f"  [lengths] {i:>3}/{len(items)} {conv_id:<15} "
                  f"{n_utterances:>3} utt  longest prompt {longest:>6} tok "
                  f"{'OVER' if longest > cap else 'ok':<4} (cap {cap})")

    return PromptLengthReport(
        conv_dir=label, questionnaire_ids=qids, n_conversations=len(items),
        max_model_len=cap, max_model_len_source=cap_source, method=method,
        base_url=base_url, model=model, rows=tuple(rows), notes=tuple(notes),
    )


def check_prompt_lengths(report: PromptLengthReport) -> Tuple[bool, List[str]]:
    """The HARD gate on the length report: no prompt may exceed the served context length.

    Returns ``(passed, reasons)``. One reason per rubric with any overflow, naming the count,
    the longest prompt and the cap; an empty report fails. A report whose method is an
    ESTIMATE can still fail -- the estimate over-counts on purpose -- and its reason says so,
    because the fix is different (measure against the real server before acting).
    """
    reasons: List[str] = []
    if not report.rows:
        return False, ["[length] nothing was measured -- the gate did not run, so it did not pass."]
    for qid in report.questionnaire_ids:
        s = report.summary(qid)
        if s["n_over"]:
            how = "" if report.measured else f" [{report.method} -- NOT a measurement; verify against the server]"
            reasons.append(
                f"[length] {metric_label(qid)}: {s['n_over']}/{s['n']} prompts exceed "
                f"max_model_len={report.max_model_len} ({report.max_model_len_source}); longest "
                f"{s['max']} tokens. Those conversations cannot be scored at all, and they are "
                f"the LONGEST ones -- an arm-dependent hole in the headline metric, not noise. "
                f"Raise VLLM_MAX_MODEL_LEN (never lower it) before scoring this state.{how}"
            )
    return (not reasons), reasons


def format_prompt_length_report(report: Union[PromptLengthReport, Mapping[str, Any]]) -> str:
    """Render the length report as an ASCII block: per-rubric table, cap provenance, verdict.

    Accepts the :class:`PromptLengthReport` or its :meth:`~PromptLengthReport.to_dict` form (what
    ``eda_analysis.scoring.prompt_length_gate`` hands ``Run_Eval``), rendered identically.
    """
    d: Mapping[str, Any] = report if isinstance(report, Mapping) else report.to_dict()
    gate = d.get("gate") or {}
    passed, reasons = bool(gate.get("passed", False)), list(gate.get("failures") or [])
    grader = d.get("grader") or {}
    qids = [int(q) for q in (d.get("questionnaire_ids") or ())]
    per_rubric = d.get("per_rubric") or {}
    rule = "=" * 88
    lines = [rule, f"Exp4 oracle prompt lengths -- {d.get('conv_dir')}", "-" * 88,
             f"conversations : {d.get('n_conversations', d.get('n_transcripts'))}",
             f"rubrics       : {', '.join(f'{metric_label(q)} (id {q})' for q in qids)}",
             f"token count   : {d.get('method')}"
             + ("" if d.get("measured") else "   <-- NOT the served tokenizer; a stand-in"),
             f"max_model_len : {d.get('max_model_len')} ({d.get('max_model_len_source')})",
             f"grader        : {grader.get('model') or '?'} @ {grader.get('base_url') or '(no server)'}"]
    for note in d.get("notes") or ():
        lines.append(f"note          : {note}")
    lines.append("")
    headers = ["metric", "n", "median", "p95", "max", "over", "headroom"]
    rows: List[List[str]] = []
    for qid in qids:
        s = per_rubric.get(metric_label(qid)) or {}
        n_over, frac_over = int(s.get("n_over", 0)), float(s.get("frac_over", 0.0))
        rows.append([
            metric_label(qid), str(s.get("n", 0)), _fmt(s.get("median"), 0), _fmt(s.get("p95"), 0),
            _fmt(s.get("max"), 0), f"{n_over} ({100.0 * frac_over:.1f}%)",
            "n/a" if s.get("headroom") is None else f"{100.0 * s['headroom']:+.0f}%",
        ])
    lines += _render_table(headers, rows)
    lines += ["  headroom = (max_model_len - longest prompt) / max_model_len; negative means",
              "  the longest conversation of this state could not be scored."]
    lines.append("")
    lines.append("HARD GATE (blocks scoring)")
    # Nothing measured is neither a pass nor an overflow: there was no prompt to check. The
    # gate function still refuses such a report (an undecidable gate must not read as green),
    # but rendering it as "FAIL -- prompts overflow" would name a failure that did not happen.
    n_measured = d.get("n_transcripts", d.get("n_conversations"))
    if n_measured is None:
        n_measured = len(d.get("rows") or ())
    if not n_measured:
        lines.append("  [n/a ] nothing measured (vacuous pass): no transcript was checked, so "
                     "no prompt overflowed and none was proven to fit")
        lines.append("")
        lines.append("VERDICT: nothing measured (vacuous pass) -- no transcripts; run the gate "
                     "on a populated model state before scoring it")
        lines.append(rule)
        return "\n".join(lines)
    if passed:
        lines.append(f"  [PASS] every prompt fits max_model_len={d.get('max_model_len')}")
    else:
        lines += [f"  [FAIL] {r}" for r in reasons]
    lines.append("")
    lines.append("VERDICT: " + ("PASS -- every conversation of this state is scoreable" if passed
                                else "FAIL -- prompts overflow the served context; do not score yet"))
    lines.append(rule)
    return "\n".join(lines)


def write_prompt_length_report(report: PromptLengthReport, path: str) -> str:
    """Write the length report as JSON; a non-``.json`` *path* is a directory (see :func:`write_report`)."""
    target = os.path.abspath(path)
    if os.path.isdir(target) or not os.path.basename(target).lower().endswith(".json"):
        target = os.path.join(target, PROMPT_LENGTH_REPORT_FILENAME)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
    return target


# ==============================================================================
#                                   CLI
# ==============================================================================


def _build_token_map() -> Dict[str, int]:
    """Every accepted spelling of a rubric for ``--questionnaires``, lowercased.

    Ids, ``QuestionnaireID`` names (``MI_SAT``), their underscore-free forms (``MISAT``) and the
    arm-name tokens from ``naming.qtag_for`` (``WAI``) all map to the same id, so the flag accepts
    whatever spelling the caller has in front of them.
    """
    tokens: Dict[str, int] = {}
    for member in QuestionnaireID:
        tokens[str(member.value)] = member.value
        tokens[member.name.lower()] = member.value
        tokens[member.name.lower().replace("_", "")] = member.value
        try:
            tokens[qtag_for([member.value]).lower()] = member.value
        except ValueError:
            pass  # eval-only instruments (PCT, MICI) have no arm-name token
    return tokens


def _parse_questionnaires(raw: str) -> Tuple[int, ...]:
    """``"1,2"`` / ``"Q1,Q2"`` / ``"wai"`` -> a tuple of ids.

    Raises:
        argparse.ArgumentTypeError: on an unknown or empty spelling.
    """
    tokens = _build_token_map()
    out: List[int] = []
    for piece in str(raw).replace(";", ",").replace(" ", ",").split(","):
        piece = piece.strip().lower()
        if not piece:
            continue
        if piece not in tokens:
            raise argparse.ArgumentTypeError(
                f"unknown questionnaire {piece!r}; expected ids or names such as "
                f"{', '.join(sorted({str(m.value) + '/' + m.name for m in QuestionnaireID}))}"
            )
        if tokens[piece] not in out:
            out.append(tokens[piece])
    if not out:
        raise argparse.ArgumentTypeError("--questionnaires is empty")
    return tuple(out)


def _build_parser() -> argparse.ArgumentParser:
    """The CLI. Defaults describe the all-open stack on its default port."""
    parser = argparse.ArgumentParser(
        prog="oracle_sanity",
        description=(
            "Check whether a grader is fit to be a measuring instrument, BEFORE any training "
            "spend. Scores a committed fixture of 12 real Exp3 conversations spanning the quality "
            "range and blocks (nonzero exit) on two hard gates: perfect schema compliance, and "
            "non-degenerate spread. Rank agreement and level offset are reported but never block "
            "-- Exp4 and Exp3 are not on the same score axis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_ORACLE_MODEL,
                        help="grader model id, as the provider spells it")
    parser.add_argument("--provider", default="openai_compat", choices=list(PROVIDERS),
                        help="openai_compat = a local vLLM server; anthropic is refused by "
                             "core.oracle (a training oracle needs an enforced schema)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint; defaults to the local vLLM port for "
                             "openai_compat, and to the vendor default otherwise")
    parser.add_argument("--questionnaires", default="1,2", type=_parse_questionnaires,
                        help="rubrics to check: ids or names, e.g. '1,2', 'Q1,Q2', 'wai'")
    parser.add_argument("--quick", action="store_true",
                        help=f"score only {QUICK_N_ITEMS} transcripts (the ends of the reference "
                             f"range) -- the pre-flight the notebooks run before iteration 1")
    parser.add_argument("--out", default=None,
                        help="write the JSON report here; anything not ending in .json is treated "
                             f"as a directory and gets {REPORT_FILENAME} (point it at the arm's "
                             "run folder, next to run_metadata.json)")
    parser.add_argument("--fixture", default=None,
                        help="override the fixture file (default: the committed one)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="in-flight oracle calls")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="oracle response budget; keep equal to the trainer's setting")
    parser.add_argument("--retries", type=int, default=3,
                        help="attempts per (conversation, rubric) call, including the first")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="per-attempt ceiling in seconds")
    parser.add_argument("--preflight-timeout", type=float, default=20.0,
                        help="seconds to wait for a local endpoint before scoring; 0 disables")
    parser.add_argument("--no-strict", action="store_true",
                        help="omit strict:true from json_schema -- for a pinned vLLM that 400s on "
                             "the key. A grader that only passes with this set is being held to "
                             "the schema by the SERVER, not following the rubric; record it.")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-call progress lines (the report still prints)")
    parser.add_argument("--conv-dir", default=None,
                        help="LENGTH MODE: measure the full oracle prompt of every conversation "
                             "in this model-state folder against the served max_model_len "
                             "(the Phase 2 gate) instead of scoring the fixture. Tokens are "
                             "counted by the server at --base-url when it answers /tokenize.")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="length mode: override the served context length to check against")
    return parser


def _main_lengths(args: argparse.Namespace, base_url: Optional[str]) -> int:
    """``--conv-dir``: the prompt-length report. 0 fits, 1 overflow, 2 nothing measured."""
    try:
        report = prompt_length_report(
            args.conv_dir, args.questionnaires,
            base_url=base_url if args.provider == "openai_compat" else None,
            model=args.model, max_model_len=args.max_model_len, progress=not args.quiet,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"oracle_sanity: could not measure prompt lengths -- {exc}", file=sys.stderr)
        return 2
    print(format_prompt_length_report(report))
    if args.out:
        print(f"report written to {write_prompt_length_report(report, args.out)}")
    passed, _reasons = check_prompt_lengths(report)
    return 0 if passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the gate from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: ``0`` every hard gate passed, ``1`` a HARD GATE FAILED (do not train against
        this grader -- or, with ``--conv-dir``, do not score this state: a prompt overflows the
        served context), ``2`` the check could not be run at all (missing fixture, unreachable
        endpoint, bad binding, empty conversations folder).

    Notes:
        1 and 2 are kept apart on purpose. A CI step or a notebook that only checks "nonzero" is
        correct either way, but the two demand different responses: 1 means the grader is unfit and
        the model/prompt/serving config has to change; 2 means nothing was learned and the check
        still has to be run.
    """
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    base_url = args.base_url
    if base_url is None and args.provider == "openai_compat":
        # One source of truth for the default port: the same ServeSpec plan_servers builds.
        base_url = ServeSpec(model=args.model).base_url

    if args.conv_dir:
        return _main_lengths(args, base_url)

    if args.no_strict:
        set_openai_compat_strict(False)

    try:
        binding = make_binding(
            args.provider,
            args.model,
            base_url=base_url,
            request_timeout=float(args.timeout),
            max_retries=int(args.retries),
        )
        report = run_async(run_sanity(
            binding,
            questionnaire_ids=args.questionnaires,
            quick=bool(args.quick),
            fixture_path=args.fixture,
            concurrency=int(args.concurrency),
            max_tokens=int(args.max_tokens),
            max_retries=int(args.retries),
            request_timeout=float(args.timeout),
            preflight_timeout=float(args.preflight_timeout),
            progress=not args.quiet,
        ))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"oracle_sanity: could not run the check -- {exc}", file=sys.stderr)
        return 2

    print(format_report(report))

    if args.out:
        written = write_report(report, args.out)
        print(f"report written to {written}")

    passed, _reasons = check_gates(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
