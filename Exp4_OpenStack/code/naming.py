"""naming.py -- the arm identity grammar: one regex, both sides of the experiment.

An "arm" is one training run: a method (PTO/GRPO) at one look-ahead K, one MCL, one branch
width, one training questionnaire, graded by one oracle model against one patient model. Its
identity has to survive a round trip through the filesystem, because that is the only channel
between the two halves of this project:

    trainer  --writes-->  data/runs/<EXP_NAME>/
                          data/conversations/<EXP_NAME>/model_iter_<N>/
                          data/eval_scores/judge=.../metric=.../<EXP_NAME>/model_iter_<N>.parquet
    EDA      --reads-->   the same three, and must recover every field from the folder name

Nothing else carries that information. ``run_metadata.json`` is a convenience, not the source of
truth: it is overwritten on resume, it is absent for a conversations-only pass, and it lives one
directory away from the scores. **The folder name is where the data physically is**, so the folder
name is the identity.

That is why the grammar lives in exactly one module, and why the writer and the reader are the same
regex rather than two implementations that agree today. The failure this design prevents is not
hypothetical: if two different arms ever render to the same ``<EXP_NAME>``, they share a
conversations folder and a score partition, and a resume-by-skipping-existing scorer will report
"already scored" against the *other* arm's numbers -- silently, with no error, producing a contrast
table that is a blend of two policies. Exp3 hit the near-miss version of this when the role
dimension was added late; Exp4 encodes the role tags unconditionally so it cannot recur.

The grammar
-----------
::

    {GRPO|PTO}4_{QTAG}_LA{K}_MCL{N}_{G{G} | M{M}_PT{greedy|indep}}_O{otag}_Pat{ptag}_Th{ttag}

    GRPO4_Q1Q2_LA5_MCL12_G8_Ogemma4E4B_Patgemma4E4B_ThL1Bi
    PTO4_Q1Q2_LA0_MCL12_M8_PTgreedy_Ogemma4E4B_Patgemma4E4B_ThL1Bi
    GRPO4_Q1Q2_LA0_MCL12_G8_Ogemma4E4B_Patgemma4E4B_ThL1B   # therapist flipped to the BASE model

Fields are ``_``-delimited, so **no token may contain an underscore**: ``MI_SAT`` is spelled
``MISAT``, and model tags come from :func:`roles.model_tag`, which is restricted to
``[A-Za-z0-9]``. A name is therefore also a legal Windows path segment and a legal TensorBoard
logdir, which is the other reason for the restriction.

The therapist policy IS encoded (the ``Th`` field), since 2026-08-27: Exp4 runs the same
Llama-3.2-1B family in two variants -- ``L1Bi`` (Instruct, the default: ships the official
Llama-3 chat template) and ``L1B`` (the template-less base with the hand-written ChatML
template) -- and two different policies must never share a folder. The field was added while the
grammar had produced zero folders on disk, which is the only moment a mandatory field can be
added without a version bump (there is no such thing as an optional field -- see
:data:`GRAMMAR_VERSION`).

``EXPERIMENT_NAME`` is **computed, never typed.** Both trainer notebooks call
:func:`build_experiment_name` from their cell-1 globals, so the name cannot drift from the config
that produced it -- which is why Exp3's ``assert_name_matches_roles`` guard has no equivalent here.

Stdlib only (plus :mod:`roles`, which is itself stdlib-only): the read-only EDA imports this module
and must never pull in torch.
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence

from roles import (DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL, DEFAULT_THERAPIST_MODEL,
                   model_tag)

__all__ = [
    "GRAMMAR",
    "GRAMMAR_VERSION",
    "METHODS",
    "PTO_MODES",
    "DEFAULT_PTO_MODE",
    "QTAG_BY_IDS",
    "IDS_BY_QTAG",
    "ARM_RE",
    "MODEL_STATE_PREFIX",
    "MODEL_STATE_RE",
    "ArmInfo",
    "qtag_for",
    "build_experiment_name",
    "parse_experiment_name",
    "model_state_label",
    "parse_model_state_label",
]


# ==============================================================================
#  Tokens
# ==============================================================================

#: Human-readable grammar, quoted verbatim in every error message. Not an f-string anywhere --
#: the braces are part of the notation.
GRAMMAR = "{GRPO|PTO}4_{QTAG}_LA{K}_MCL{N}_{G{G}|M{M}_PT{greedy|indep}}_O{otag}_Pat{ptag}_Th{ttag}"

#: Grammar version. Bumped only if the *meaning* of an existing field changes; a new optional
#: field would orphan every folder already on disk, so there is no such thing as an optional field.
GRAMMAR_VERSION = 4

METHODS = ("GRPO", "PTO")

#: PTO preference-tree modes as they appear in a NAME. The trainer's ``PREF_TREE_MODE`` global
#: spells the second one ``independent``; :func:`build_experiment_name` accepts either and emits
#: the short token, because ``independent`` is not what any folder on disk is called.
PTO_MODES = ("greedy", "indep")
DEFAULT_PTO_MODE = "greedy"

_MODE_ALIASES = {"greedy": "greedy", "indep": "indep", "independent": "indep"}

#: Training-questionnaire set -> its name token. The KEY is a frozenset because the reward is the
#: unweighted mean over the set, so ``[1, 2]`` and ``[2, 1]`` are the same experiment and must not
#: produce two folders.
#:
#: PCT (8) and MICI (9) are deliberately absent: they are eval-only instruments. PCT grades the
#: PATIENT, and MICI is lower-is-better, so neither is a reward a policy can maximize without a
#: sign convention this experiment does not define. Adding one means defining that first.
QTAG_BY_IDS: Dict[FrozenSet[int], str] = {
    frozenset({1, 2}): "Q1Q2",
    frozenset({1}): "Q1",
    frozenset({2}): "Q2",
    frozenset({3}): "WAI",
    frozenset({4}): "CSQ8",
    frozenset({6}): "MISAT",   # MI_SAT without the underscore -- see module docstring
    frozenset({7}): "MITI",
}

#: Reader-side inverse: given a parsed arm, which rubric WAS the training reward. Built from the
#: forward table so the two cannot disagree.
IDS_BY_QTAG: Dict[str, FrozenSet[int]] = {tag: ids for ids, tag in QTAG_BY_IDS.items()}


# ==============================================================================
#  The regex
# ==============================================================================
#
# Built from QTAG_BY_IDS rather than hand-listed, so a new questionnaire token becomes parseable
# the moment it becomes buildable. Longest-first ordering keeps "Q1Q2" from being consumed as
# "Q1" (the trailing "_" would force a backtrack anyway, but relying on that is a trap for the
# next token pair that shares a prefix).
_QTAG_ALT = "|".join(sorted(set(QTAG_BY_IDS.values()), key=lambda t: (-len(t), t)))

#: THE arm-name regex. Shape only -- it accepts ``PTO4_..._G8_...`` and ``GRPO4_..._M8_PTgreedy_...``
#: because encoding the method-to-branch dependency in a single pattern needs duplicate group names.
#: That cross-field rule is enforced by :meth:`ArmInfo.__post_init__`, which every construction path
#: goes through, so no malformed name survives :func:`parse_experiment_name`.
ARM_RE = re.compile(
    r"^(?P<method>GRPO|PTO)4"
    r"_(?P<qtag>" + _QTAG_ALT + r")"
    r"_LA(?P<k>\d+)"
    r"_MCL(?P<mcl>\d+)"
    r"_(?:G(?P<g>\d+)|M(?P<m>\d+)_PT(?P<mode>greedy|indep))"
    r"_O(?P<otag>[A-Za-z0-9]+)"
    r"_Pat(?P<ptag>[A-Za-z0-9]+)"
    r"_Th(?P<ttag>[A-Za-z0-9]+)$"
)

#: Model-state folder inside ``conversations/<EXP_NAME>/``. Exp3 appended ``_TT*_TP*`` sampling
#: temperatures here and every reader needed a glob; Exp4 records temperatures in the metadata and
#: keeps the folder name exact, so a state can be addressed by string equality.
MODEL_STATE_PREFIX = "model_iter_"
MODEL_STATE_RE = re.compile(r"^" + MODEL_STATE_PREFIX + r"(?P<n>\d+)$")

_TAG_RE = re.compile(r"^[A-Za-z0-9]+$")

# Computed once: the tags an arm carries when nothing was swapped. Used ONLY by
# :attr:`ArmInfo.label` -- the NAME always spells every tag out.
_DEFAULT_ORACLE_TAG = model_tag(DEFAULT_ORACLE_MODEL)
_DEFAULT_PATIENT_TAG = model_tag(DEFAULT_PATIENT_MODEL)
_DEFAULT_THERAPIST_TAG = model_tag(DEFAULT_THERAPIST_MODEL)


# ==============================================================================
#  ArmInfo
# ==============================================================================


@dataclass(frozen=True)
class ArmInfo:
    """One arm's identity, fully decoded. Frozen: it is a key, not a workspace.

    Constructing an inconsistent arm raises -- GRPO carries ``g`` and no mode, PTO carries ``m``
    plus a mode and no ``g``. That check lives here rather than in the callers so that every
    path into an :class:`ArmInfo` (built, parsed, or hand-written in a notebook) is validated by
    the same code.

    Note:
        ``mode`` is the short token (``greedy`` / ``indep``), never the trainer's spelling
        ``independent``. :func:`build_experiment_name` normalizes; this class does not.
    """

    method: str          # "GRPO" | "PTO"
    qtag: str            # "Q1Q2" | "Q1" | "Q2" | "WAI" | "CSQ8" | "MISAT" | "MITI"
    k: int               # look-ahead depth; 0 disables look-ahead
    mcl: int             # MIN_CONV_LENGTH, in utterances (therapist + patient combined)
    g: Optional[int]     # GRPO group size          (None for PTO)
    m: Optional[int]     # PTO branches per turn    (None for GRPO)
    mode: Optional[str]  # PTO preference-tree mode (None for GRPO)
    oracle_tag: str      # model_tag of the TRAINING grader
    patient_tag: str     # model_tag of the patient simulator
    therapist_tag: str   # model_tag of the therapist POLICY (L1Bi = Instruct, L1B = base)

    def __post_init__(self) -> None:
        _validate_fields(self.method, self.qtag, self.k, self.mcl,
                         self.g, self.m, self.mode, self.oracle_tag, self.patient_tag,
                         self.therapist_tag)

    @property
    def experiment_name(self) -> str:
        """The folder name. The single formatter -- :func:`build_experiment_name` delegates here."""
        branch = f"G{self.g}" if self.method == "GRPO" else f"M{self.m}_PT{self.mode}"
        return (f"{self.method}{GRAMMAR_VERSION}_{self.qtag}_LA{self.k}_MCL{self.mcl}"
                f"_{branch}_O{self.oracle_tag}_Pat{self.patient_tag}_Th{self.therapist_tag}")

    @property
    def label(self) -> str:
        """Short display label -- the plot legend entry and table row key.

        Everything on its default is elided, so the common grid reads ``GRPO_LA5`` / ``PTO_LA0``
        while a swapped role, a non-default preference-tree mode or the base therapist still
        separates: ``GRPO_LA0_Ogpt4m``, ``PTO_LA5_indep``, ``GRPO_LA0_ThL1B``.

        Warning:
            This is a DISPLAY key, not an identity. It deliberately drops ``qtag``, ``mcl`` and
            the branch width, so two arms that differ only in MCL (or only in G/M, or only in the
            training questionnaire) share a label and would silently merge in a groupby. Key on
            :attr:`experiment_name` for anything that reads or writes data; use ``label`` only
            where a human is looking at it.
        """
        parts = [f"{self.method}_LA{self.k}"]
        if self.mode is not None and self.mode != DEFAULT_PTO_MODE:
            parts.append(self.mode)
        if self.oracle_tag != _DEFAULT_ORACLE_TAG:
            parts.append("O" + self.oracle_tag)
        if self.patient_tag != _DEFAULT_PATIENT_TAG:
            parts.append("Pat" + self.patient_tag)
        if self.therapist_tag != _DEFAULT_THERAPIST_TAG:
            parts.append("Th" + self.therapist_tag)
        return "_".join(parts)

    @property
    def branches(self) -> int:
        """Candidates scored per branch point: GRPO's ``G`` or PTO's ``M``.

        The two are matched by design (both 8), which is what makes the per-prompt oracle spend
        comparable across methods; reading them through one property keeps a cost table from
        having to branch on the method.
        """
        return int(self.g if self.method == "GRPO" else self.m)

    @property
    def questionnaire_ids(self) -> FrozenSet[int]:
        """Which questionnaire ids were the training reward."""
        return IDS_BY_QTAG[self.qtag]

    def model_state(self, n: int) -> str:
        """Conversations subfolder for the policy at state *n*. See :func:`model_state_label`."""
        return model_state_label(n)


# ==============================================================================
#  Build
# ==============================================================================


def qtag_for(questionnaire_ids: Sequence[int]) -> str:
    """Name token for a training-questionnaire SET.

    Args:
        questionnaire_ids: ids from ``questionnaires.QuestionnaireID`` -- ints, or enum members
            (their ``.value`` is taken). Order and duplicates are irrelevant: the reward is the
            unweighted mean over the set, so ``[2, 1]`` and ``[1, 2]`` are one experiment and get
            one token.

    Returns:
        One of the values of :data:`QTAG_BY_IDS`.

    Raises:
        ValueError: if the set has no token. This is a hard stop on purpose -- inventing a token
            on the fly (say ``Q1Q2WAI``) would produce a folder no released parser can read, and
            the scores would be unreachable. Add the entry to :data:`QTAG_BY_IDS` instead, which
            also makes it parseable, because :data:`ARM_RE` is built from that table.
    """
    if questionnaire_ids is None:
        raise ValueError("qtag_for: questionnaire_ids is None; expected a non-empty sequence of ids")
    try:
        ids = frozenset(int(getattr(q, "value", q)) for q in questionnaire_ids)
    except (TypeError, ValueError) as ex:
        raise ValueError(
            f"qtag_for: questionnaire_ids={questionnaire_ids!r} is not a sequence of ints ({ex})"
        ) from ex
    if not ids:
        raise ValueError("qtag_for: questionnaire_ids is empty; the training reward needs at least one rubric")
    tag = QTAG_BY_IDS.get(ids)
    if tag is None:
        known = ", ".join(
            f"{sorted(k)}->{v}" for k, v in sorted(QTAG_BY_IDS.items(), key=lambda kv: sorted(kv[0]))
        )
        raise ValueError(
            f"qtag_for: no name token for questionnaire set {sorted(ids)}. "
            f"Known sets: {known}. Add an entry to naming.QTAG_BY_IDS (which also extends ARM_RE) "
            f"before running this arm -- an unnamed set cannot be written to disk."
        )
    return tag


def build_experiment_name(method: str,
                          questionnaire_ids: Sequence[int],
                          k: int,
                          mcl: int,
                          *,
                          g: Optional[int] = None,
                          m: Optional[int] = None,
                          mode: Optional[str] = None,
                          oracle_model: str,
                          patient_model: str,
                          therapist_model: str = DEFAULT_THERAPIST_MODEL) -> str:
    """Compose ``EXPERIMENT_NAME`` from a trainer's cell-1 globals.

    Args:
        method: ``"GRPO"`` or ``"PTO"`` (case-insensitive).
        questionnaire_ids: the TRAINING rubric set -- see :func:`qtag_for`.
        k: ``LOOKAHEAD_K``. 0 disables look-ahead.
        mcl: ``MIN_CONV_LENGTH`` in utterances.
        g: GRPO's ``NUM_GENERATIONS``. Required for GRPO, forbidden for PTO.
        m: PTO's ``NUM_BRANCHES_PER_TURN``. Required for PTO, forbidden for GRPO.
        mode: PTO's ``PREF_TREE_MODE`` -- ``greedy`` | ``indep`` | ``independent``
            (the last is accepted and written as ``indep``). Required for PTO, forbidden for GRPO.
        oracle_model: model id of the TRAINING grader, e.g. ``google/gemma-4-E2B-it``.
        patient_model: model id of the patient simulator.
        therapist_model: model id of the therapist POLICY. Defaults to
            :data:`roles.DEFAULT_THERAPIST_MODEL` (the Instruct variant) -- the one tag with a
            true project-wide default. Unlike Exp3's optional role suffixes this cannot cause a
            collision: the tag is encoded in the name whether or not the caller passed it, so a
            forgotten argument mislabels nothing that ``run_metadata.json`` does not correct.
            The trainer config builders always pass it explicitly from ``BASE_MODEL_ID``.

    Returns:
        A name matching :data:`GRAMMAR`, made only of ``[A-Za-z0-9_]``.

    Raises:
        ValueError: on an unmapped questionnaire set, an unknown method or mode, a
            method/branch-argument mismatch (GRPO with ``m``, PTO with ``g``), a non-positive
            width, a negative ``k``, or a role tag that is not ``[A-Za-z0-9]+``.

    Notes:
        The role models are keyword-only and mandatory. They were optional in Exp3 -- which is
        exactly how an arm ends up unlabelled and colliding with the default-graded one.

        MCL is NOT checked for evenness here. PTO's ``greedy`` mode requires an even MCL (the
        sliced trunk seed has to end on a patient turn), but that is a config-validity rule owned
        by ``core/config.py``; enforcing it here would make :func:`build_experiment_name` reject
        names that :func:`parse_experiment_name` still accepts, and the asymmetry is worse than
        the duplication.
    """
    meth = str(method).strip().upper()
    if meth not in METHODS:
        raise ValueError(f"build_experiment_name: method={method!r} is not one of {METHODS}")

    qtag = qtag_for(questionnaire_ids)

    norm_mode: Optional[str] = None
    if mode is not None:
        norm_mode = _MODE_ALIASES.get(str(mode).strip().lower())
        if norm_mode is None:
            raise ValueError(
                f"build_experiment_name: mode={mode!r} is not a preference-tree mode; "
                f"expected one of {PTO_MODES} (or 'independent' for 'indep')"
            )

    otag = model_tag(oracle_model)
    ptag = model_tag(patient_model)
    ttag = model_tag(therapist_model)

    # roles.model_tag promises [A-Za-z0-9]; check it anyway. A tag with an underscore or a dot
    # would not raise here -- it would produce a name that parses back into DIFFERENT fields, and
    # the corruption would only surface as a mislabelled row in an EDA table months later.
    for role, model_id, tag in (("oracle", oracle_model, otag), ("patient", patient_model, ptag),
                                ("therapist", therapist_model, ttag)):
        if not _TAG_RE.match(tag or ""):
            raise ValueError(
                f"build_experiment_name: {role} model {model_id!r} maps to tag {tag!r}, which is "
                f"not [A-Za-z0-9]+. Name fields are '_'-delimited, so such a tag would corrupt the "
                f"grammar. Add a curated entry to roles._MODEL_TAGS."
            )

    try:
        arm = ArmInfo(method=meth, qtag=qtag, k=int(k), mcl=int(mcl),
                      g=None if g is None else int(g),
                      m=None if m is None else int(m),
                      mode=norm_mode, oracle_tag=otag, patient_tag=ptag, therapist_tag=ttag)
    except ValueError as ex:
        # Almost always a cell-1 typo, so name the entry point and the method being built.
        raise ValueError(f"build_experiment_name({meth}): {ex}") from ex
    return arm.experiment_name


# ==============================================================================
#  Parse
# ==============================================================================


def parse_experiment_name(name: str) -> ArmInfo:
    """Decode a folder name into an :class:`ArmInfo`.

    The reader-side half of the contract:
    ``parse_experiment_name(build_experiment_name(...))`` reproduces the arm exactly.

    Args:
        name: an ``EXPERIMENT_NAME`` folder segment (no path, no trailing separator).

    Returns:
        The decoded arm.

    Raises:
        ValueError: if *name* does not match :data:`GRAMMAR`, or matches structurally but is
            internally inconsistent (a PTO name carrying ``G8``, say).

    Notes:
        This RAISES rather than returning ``None`` -- Exp3's version returned ``None`` and every
        discovery loop quietly ``continue``d on it, so a typo'd folder became an invisible missing
        arm instead of an error. If you are scanning a directory that may legitimately contain
        non-arm entries, catch the ValueError at the call site where you can say so out loud.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"parse_experiment_name: expected a non-empty str, got {name!r}")
    match = ARM_RE.match(name)
    if match is None:
        raise ValueError(
            f"parse_experiment_name: {name!r} is not an Exp4 arm name.\n"
            f"  expected: {GRAMMAR}\n"
            f"  example : GRPO4_Q1Q2_LA5_MCL12_G8_Ogemma4E4B_Patgemma4E4B_ThL1Bi\n"
            f"  example : PTO4_Q1Q2_LA0_MCL12_M8_PTgreedy_Ogemma4E4B_Patgemma4E4B_ThL1Bi\n"
            f"  QTAG in {sorted(set(QTAG_BY_IDS.values()))}; model tags are [A-Za-z0-9]+ "
            f"(no '_' inside a token)."
        )
    d = match.groupdict()
    try:
        return ArmInfo(
            method=d["method"],
            qtag=d["qtag"],
            k=int(d["k"]),
            mcl=int(d["mcl"]),
            g=int(d["g"]) if d["g"] is not None else None,
            m=int(d["m"]) if d["m"] is not None else None,
            mode=d["mode"],
            oracle_tag=d["otag"],
            patient_tag=d["ptag"],
            therapist_tag=d["ttag"],
        )
    except ValueError as ex:
        raise ValueError(f"parse_experiment_name: {name!r} is malformed -- {ex}") from ex


def model_state_label(n: int) -> str:
    """Conversations-folder name for the policy that GENERATED them.

    Iteration ``n`` generates with the iter-(``n``-1) adapter and saves to ``model_iter_{n-1}``,
    then produces ``iteration_n/adapter/``; a post-loop generate-only pass writes ``model_iter_N``.
    So ``N`` iterations yield ``N+1`` conversation folders, and ``model_iter_0`` is always the
    untrained base policy.

    Raises:
        ValueError: on a negative or non-integer *n* -- a stringified ``None`` here would create
            a folder that no reader ever finds.
    """
    idx = int(n)
    if idx < 0:
        raise ValueError(f"model_state_label: n={n!r} must be >= 0 (model_iter_0 is the base policy)")
    return f"{MODEL_STATE_PREFIX}{idx}"


def parse_model_state_label(label: str) -> int:
    """Inverse of :func:`model_state_label`.

    Raises:
        ValueError: if *label* is not exactly ``model_iter_<digits>``. Exp3's states carried a
            ``_TT0.9_TP0.7`` temperature suffix and needed a glob; Exp4 does not, so an unexpected
            suffix means something wrote a folder off-contract and should be reported, not matched.
    """
    match = MODEL_STATE_RE.match(label or "")
    if match is None:
        raise ValueError(
            f"parse_model_state_label: {label!r} is not a model-state folder "
            f"(expected '{MODEL_STATE_PREFIX}<N>')"
        )
    return int(match.group("n"))


# ==============================================================================
#  Shared validation
# ==============================================================================


def _is_int(value: object) -> bool:
    """True for a whole-number scalar that is not a bool.

    ``numbers.Integral`` rather than ``int`` because the EDA reconstructs arms from pandas
    frames, where an integer column yields ``numpy.int64`` -- which is not an ``int`` and would
    fail an ``isinstance`` check for no reason. ``bool`` is excluded because ``True`` is an
    ``Integral`` equal to 1, and ``k=True`` is a caller mistake, not a look-ahead depth.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _validate_fields(method: str, qtag: str, k: int, mcl: int,
                     g: Optional[int], m: Optional[int], mode: Optional[str],
                     oracle_tag: str, patient_tag: str, therapist_tag: str) -> None:
    """Every consistency rule, in one place, run by both build and parse.

    The method/branch rules are the load-bearing ones: they are what stops a GRPO arm from being
    recorded with a preference-tree mode it never had, and a PTO arm from claiming a group size.
    """
    if method not in METHODS:
        raise ValueError(f"method={method!r} is not one of {METHODS}")
    if qtag not in IDS_BY_QTAG:
        raise ValueError(f"qtag={qtag!r} is not one of {sorted(IDS_BY_QTAG)}")
    if not _is_int(k) or k < 0:
        raise ValueError(f"k={k!r} must be a non-negative int (0 disables look-ahead)")
    if not _is_int(mcl) or mcl < 1:
        raise ValueError(f"mcl={mcl!r} must be a positive int (utterances of conversation-so-far)")

    if method == "GRPO":
        if g is None:
            raise ValueError("GRPO arms need g (NUM_GENERATIONS); got g=None")
        if not _is_int(g) or g < 1:
            raise ValueError(f"g={g!r} must be a positive int")
        if m is not None or mode is not None:
            raise ValueError(
                f"GRPO arms carry no preference tree: got m={m!r}, mode={mode!r}. "
                f"GRPO has no preference data -- only prompts."
            )
    else:  # PTO
        if m is None:
            raise ValueError("PTO arms need m (NUM_BRANCHES_PER_TURN); got m=None")
        if not _is_int(m) or m < 1:
            raise ValueError(f"m={m!r} must be a positive int")
        if mode not in PTO_MODES:
            raise ValueError(f"PTO arms need mode in {PTO_MODES}; got mode={mode!r}")
        if g is not None:
            raise ValueError(f"PTO arms carry no group size: got g={g!r} (did you mean m={g!r}?)")

    for role, tag in (("oracle_tag", oracle_tag), ("patient_tag", patient_tag),
                      ("therapist_tag", therapist_tag)):
        if not isinstance(tag, str) or not _TAG_RE.match(tag):
            raise ValueError(
                f"{role}={tag!r} must be [A-Za-z0-9]+ -- name fields are '_'-delimited"
            )
