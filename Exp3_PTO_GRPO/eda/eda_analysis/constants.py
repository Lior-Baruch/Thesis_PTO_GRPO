"""
constants.py — the package's LEAF module: workspace paths, metric registries, and label helpers.

Imports NOTHING from the package (stdlib only), so every submodule can do a plain top-level
``from .constants import ...`` with no circular-import risk. This is what lets ``data``/``stats``/
``plotting``/... keep their imports at the top of the file instead of deferring them inside
functions. ``__init__.py`` re-exports everything here, so the public surface
(``eda_analysis.QUESTIONNAIRES`` etc.) is unchanged.
"""

import os
import re
import sys

# ── Resolve the experiment root (the Exp3 folder: HF_key.txt + openai_key.txt) ──
_KEY_FILES = ("HF_key.txt", "openai_key.txt")


def _resolve_workspace_root(*starts, max_steps: int = 10):
    for start in starts:
        cur = os.path.abspath(start)
        for _ in range(max_steps):
            if all(os.path.exists(os.path.join(cur, kf)) for kf in _KEY_FILES):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return None


WORKSPACE_ROOT = _resolve_workspace_root(os.path.dirname(__file__), os.getcwd())
if WORKSPACE_ROOT is None:
    raise RuntimeError(
        f"eda_analysis: could not locate experiment root containing {_KEY_FILES} by "
        f"walking up from {os.path.dirname(__file__)!r} or {os.getcwd()!r}"
    )

# Make the per-experiment helpers importable (system_prompts_builder, questionnaires).
_CODE_DIR = os.path.join(WORKSPACE_ROOT, "code")
for _p in (WORKSPACE_ROOT, _CODE_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")

# Display name -> (eval_scores metric subdir, the per-conv mean column).
QUESTIONNAIRES = {
    "Q1":     ("Q1",     "Q1_Mean"),
    "Q2":     ("Q2",     "Q2_Mean"),
    "Q1Q2":   (None,     "Q1Q2_Mean"),   # composite: mean(Q1_Mean, Q2_Mean)
    "WAI-SR": ("WAI_SR", "WAI_TotalMean"),
    "CSQ-8":  ("CSQ8",   "CSQ8_Mean"),
    "MI-SAT": ("MI_SAT", "MI_Mean"),
    "MITI":   ("MITI",   "MITI_GlobalMean"),
    # Orthogonal axes (added 2026-06-14 to break the PC1≈91% warmth halo):
    "PCT":    ("PCT",    "PCT_ChangeProp"),   # patient change-talk proportion CT/(CT+ST); higher = better
    "MICI":   ("MICI",   "MICI_Rate"),        # MI-inconsistent behaviors per therapist turn; LOWER = better
}
# Left-to-right plot order for the warmth rubrics (+ Q1/Q2 components) then the orthogonal axes.
QUESTIONNAIRE_ORDER = ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI", "Q1", "Q2"]

# The 5 global-evaluation rubrics that share the dominant PC1 factor (the empirical halo /
# redundancy set — NOT one official construct). "WARMTH_RUBRICS" is the historical code name,
# kept for API stability; prose should say "global-evaluation (halo) cluster".
WARMTH_RUBRICS = ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI"]
# Orthogonal axes intended to load OFF PC1 (incl. the free derived MITI-proficiency ratios).
ORTHOGONAL_METRICS = ["PCT", "MICI", "R:Q", "%CR", "%MICO"]

# ── Official MITI 4.2.1 clinician thresholds — (fair, good) per summary score ────
# Source: MITI 4.2.1 manual (Moyers, Manuel & Ernst 2014; manual rev. June 2015), §I
# "Clinician basic competence and proficiency thresholds" + §H summary-score formulas.
# Caveats the manual itself states (repeat them wherever these lines are drawn):
#   • thresholds are EXPERT OPINION — no normative/validity data support them yet;
#   • Total MIA / MINA thresholds are intentionally unspecified;
# plus ours: the MITI was designed for ~20-min human audio sessions, not short text chats.
# Formulas: Technical = (CultivatingChangeTalk + SofteningSustainTalk)/2;
#           Relational = (Partnership + Empathy)/2; %CR = CR/(SR+CR); R:Q = reflections/questions.
MITI_THRESHOLDS = {
    "R:Q":             (1.0, 2.0),
    "%CR":             (0.40, 0.50),
    "MITI_Technical":  (3.0, 4.0),
    "MITI_Relational": (3.5, 4.0),
}

# ── Q2 per-item labels + face-content groups (for the item-level reward-composition EDA) ──
# Q2 = the 17-item Working Alliance / Relational Communication LLM-evaluator prompt from the
# lab's CLPsych 2024 paper (Yosef et al.) — see METRICS_REFERENCE.md §1. Short labels paraphrase
# each item for axis ticks. The GROUPS are OUR face-content reading of the items (an analytical
# grouping for attribution figures), NOT a validated subscale structure — label figures accordingly.
# Note items 1/2/3/10 reward therapist SELF-DISCLOSURE — behavior MI does not prescribe — which is
# why the item-level view matters: training on Q1+Q2 may directly incentivize the emotive drift.
Q2_ITEM_SHORT = {
    1: "sense of who he was", 2: "revealed his thinking", 3: "shared his feelings",
    4: "knew how I was feeling", 5: "understood me", 6: "put himself in my shoes",
    7: "comfortable talking", 8: "relaxed and secure", 9: "took charge",
    10: "said when happy/sad", 11: "no difficulty w/ words", 12: "expressed himself",
    13: "a 'warm' partner", 14: "did not judge me", 15: "treated me as equal",
    16: "made me feel cared for", 17: "made me feel close",
}
Q2_ITEM_GROUPS = {
    "Self-disclosure":       [1, 2, 3, 10],
    "Empathy/understanding": [4, 5, 6],
    "Fluency/ease":          [7, 8, 11, 12],
    "Direction/control":     [9],
    "Warmth/closeness":      [13, 16, 17],
    "Non-judgment/equality": [14, 15],
}
# item number -> group name (the lookup figures actually use).
Q2_ITEM_GROUP_OF = {i: g for g, items in Q2_ITEM_GROUPS.items() for i in items}

# ── Q1 per-item labels (same convention as Q2_ITEM_SHORT) ────────────────────────
# Q1 = the 5-item Session Satisfaction LLM-evaluator prompt (CLPsych 2024; see
# METRICS_REFERENCE.md §1). Short labels paraphrase code/questionnaires.py::get_questionnaire_1.
Q1_ITEM_SHORT = {
    1: "overall chat satisfaction", 2: "content satisfaction", 3: "facilitated motivation",
    4: "learned something new", 5: "learning relevant to daily life",
}

# ── Per-item column layout of every Likert-item questionnaire in eval_scores/ ────
# Display name -> (eval_scores metric subdir, ordered per-item column list). Source of truth for
# the item TEXT is code/questionnaires.py (single canonical copy) — hardcoded here so this module
# stays a leaf (imports nothing). MITI/PCT/MICI are NOT here: their detail is behavior counts /
# rates (see behavior.py), not rating-scale items.
ITEM_QUESTIONNAIRES = {
    "Q1": ("Q1", [f"Q1_{i}" for i in range(1, 6)]),
    "Q2": ("Q2", [f"Q2_{i}" for i in range(1, 18)]),
    "WAI-SR": ("WAI_SR", [
        "WAI1_ClearChange", "WAI2_NewWays", "WAI3_TherapistLikesMe", "WAI4_CollaborateGoals",
        "WAI5_MutualRespect", "WAI6_WorkingTowardGoals", "WAI7_AppreciatesMe",
        "WAI8_AgreeImportantWork", "WAI9_CaresDespiteDisapproval", "WAI10_TasksHelpChange",
        "WAI11_UnderstandGoodChanges", "WAI12_WayOfWorkingCorrect",
    ]),
    "CSQ-8": ("CSQ8", [
        "CSQ1_Quality", "CSQ2_ServiceFit", "CSQ3_NeedsMet", "CSQ4_Recommend",
        "CSQ5_AmountOfHelp", "CSQ6_Effectiveness", "CSQ7_OverallSatisfaction",
        "CSQ8_ReturnIntention",
    ]),
    "MI-SAT": ("MI_SAT", [
        "MI1_Helpful", "MI2_Enjoyable", "MI3_Interesting", "MI4_EasyToUse",
        "MI5_WorthTime", "MI6_LikelyChange",
    ]),
}

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ITEM_PREFIX_RE = re.compile(r"^[A-Za-z]+\d+_")


def item_short_label(questionnaire: str, item, item_key: str = "") -> str:
    """Short per-item label for axis ticks: Q1/Q2 via the explicit maps, the rest parsed
    from the semantic column-name tail (``WAI9_CaresDespiteDisapproval`` -> ``"Cares despite
    disapproval"``). ``item`` is the 1-based item number; ``item_key`` the raw column name."""
    if questionnaire == "Q1":
        return Q1_ITEM_SHORT.get(int(item), str(item))
    if questionnaire == "Q2":
        return Q2_ITEM_SHORT.get(int(item), str(item))
    tail = _ITEM_PREFIX_RE.sub("", item_key or "")
    if not tail:
        return str(item)
    words = _CAMEL_RE.sub(" ", tail).split()
    return " ".join([words[0]] + [w.lower() for w in words[1:]]) if words else str(item)
# Metrics where a LOWER value is better (must not be pooled into warmth composites / collapse_base).
# The "MICI" questionnaire aggregate + every per-item MICI detail column (severity, per-turn rates)
# are higher = worse, as is patient sustain-talk. Display layer only (drives the trailing ' ↓').
LOWER_IS_BETTER = {
    "MICI", "MICI_Severity", "MICI_Rate", "MICI_Confront_rate", "MICI_AdviseNoPermission_rate",
    "MICI_Warn_rate", "MICI_Direct_rate", "MICI_Judge_rate", "MICI_OverPraise_rate",
    "PCT_SustainTalk_prop",
}


# ── Human-readable display names (LABEL LAYER ONLY) ──────────────────────────────
# Applied when drawing figures / writing tables so a supervisor never has to decode `B3_Q` or
# `%MICO`. These NEVER rename the underlying `questionnaire` / `arm` / column KEYS (those are used
# as join+filter keys throughout the package). Any code not in the map falls through unchanged.
DISPLAY_NAMES = {
    # LABEL CONVENTION (keeps figures consistent — enforce this when adding a metric):
    #   • Every oracle-MITI-coded metric carries a trailing "(MITI)" so the source instrument is
    #     unambiguous and identical across panels (fixes the old behavior_drift asymmetry where only
    #     Questions/Empathy were tagged while Reflections/Affirmations/Persuasion read bare).
    #   • Deterministic (non-LLM) text metrics carry their OWN source tag instead ("regex ?",
    #     "Degeneration %") — never "(MITI)".
    #   • Standalone questionnaires keep their validated-instrument acronym up-front + a gloss.
    # Display layer only — these NEVER rename the underlying data keys.
    #
    # Global-evaluation (halo) rubrics. Q1Q2/Q1/Q2 stay as their plain codes
    # (Lior: no "Satisfaction …" prefix) — Q1/Q2 simply fall through display_label unchanged.
    "Q1Q2": "Q1+Q2",
    # Original validated-instrument acronym KEPT up-front (Lior), descriptive gloss in parens.
    "WAI-SR": "WAI-SR (Working Alliance)", "CSQ-8": "CSQ-8 (Client Satisfaction)",
    "MI-SAT": "MI-SAT (MI Satisfaction)", "MITI": "MITI (MI Integrity)",
    # Standalone orthogonal questionnaires (their own instruments, NOT MITI-derived).
    "PCT": "PCT (Patient Change-Talk)", "MICI": "MICI (MI-Inconsistency)",
    # Derived MITI-proficiency ratios (computed FROM the MITI behavior counts → tagged "(MITI)").
    "R:Q": "Reflection:Question (MITI)", "%CR": "% Complex Reflections (MITI)",
    "%MICO": "% MI-Consistent (MITI)",
    # Official MITI 4.2.1 summary globals (manual §H) — the threshold panel plots these.
    "MITI_Technical": "Technical global (MITI)", "MITI_Relational": "Relational global (MITI)",
    "SoftenSustain": "Softening Sustain Talk (MITI)",
    # MITI global ratings (1-5) as they appear in the behavior/detail frames.
    "ChangeTalk": "Cultivating Change Talk (MITI)", "Partnership": "Partnership (MITI)",
    # MITI behavior counts (per conversation). "Questions" is a per-conv COUNT of question-FUNCTION
    # utterances (oracle) — kept distinct from the regex "? / turn" RATE below to avoid misreading a
    # count against a rate (they are different constructs: function vs literal-? syntax).
    "B3_Q": "Questions / conv (MITI)", "B6_AF": "Affirmations (MITI)", "B4_SR": "Simple Reflections (MITI)",
    "B5_CR": "Complex Reflections (MITI)", "B2_Persuade": "Persuasion (MITI)", "B1_GI": "Giving Information (MITI)",
    "B7_Seek": "Seeking Collaboration (MITI)", "RtoQ": "Reflection:Question (MITI)", "Empathy": "Empathy (MITI)",
    # Per-therapist-turn rate versions of the MITI counts (length-normalized; the drift figure plots these).
    "B3_Q_per_turn": "Questions / turn (MITI)", "B6_AF_per_turn": "Affirmations / turn (MITI)",
    "B4_SR_per_turn": "Simple Reflections / turn (MITI)", "B5_CR_per_turn": "Complex Reflections / turn (MITI)",
    "B2_Persuade_per_turn": "Persuasion / turn (MITI)",
    "B1_GI_per_turn": "Giving Information / turn (MITI)", "B7_Seek_per_turn": "Seeking Collaboration / turn (MITI)",
    # MICI (MI-INCONSISTENT) detail — severity global (1-5) + per-therapist-turn behavior rates.
    # Every MICI column is higher = worse (↓-flagged via LOWER_IS_BETTER).
    "MICI_Severity": "MI-Incon. Severity (MICI)", "MICI_Rate": "MI-Incon. total / turn (MICI)",
    "MICI_Confront_rate": "Confront / turn (MICI)", "MICI_AdviseNoPermission_rate": "Advise w/o permission / turn (MICI)",
    "MICI_Warn_rate": "Warn / turn (MICI)", "MICI_Direct_rate": "Direct/order / turn (MICI)",
    "MICI_Judge_rate": "Judge/label / turn (MICI)", "MICI_OverPraise_rate": "Over-praise / turn (MICI)",
    # PCT (PATIENT change-talk) detail — patient-perspective globals (1-5) + utterance proportions.
    "PCT_Importance": "Importance (PCT)", "PCT_Confidence": "Confidence (PCT)",
    "PCT_Readiness": "Readiness (PCT)", "PCT_GlobalMean": "PCT global mean",
    "PCT_ChangeProp": "Change-Talk proportion (PCT)", "PCT_ChangeTalk_prop": "% Change Talk (PCT)",
    "PCT_SustainTalk_prop": "% Sustain Talk (PCT)", "PCT_Neutral_prop": "% Neutral (PCT)",
    # Deterministic text metrics — NOT MITI; each carries its own source tag.
    "q_per_turn": "Questions / turn (regex ?)", "q_per_turn_miti": "Questions / turn (MITI)",
    "mean_turn_len": "Turn length (chars)", "loop": "Degeneration %",
    "conv_len": "Conversation length", "n_th_turns": "Therapist turns",
}

# Readable arm labels: canonical key -> "<method> (K=<k>)".
ARM_LABELS = {"PTO_LA0": "PTO (K=0)", "PTO_LA5": "PTO (K=5)",
              "GRPO_LA0": "GRPO (K=0)", "GRPO_LA5": "GRPO (K=5)", "Base": "Base"}
_ARM_RE = re.compile(r"^(PTO|GRPO)_LA(\d+)$")


def display_label(metric: str) -> str:
    """Readable label for a metric / behavior code, flagging lower-is-better with a trailing '↓'.

    Consults :data:`DISPLAY_NAMES` (falls through to the raw code if absent), then appends ' ↓' for
    :data:`LOWER_IS_BETTER` metrics so e.g. ``MICI`` reads ``MI-Inconsistency ↓`` and is never
    mistaken for a higher-is-better rubric. Label layer only — never used as a data key.
    """
    name = DISPLAY_NAMES.get(metric, metric)
    return f"{name} ↓" if metric in LOWER_IS_BETTER else name


_SHORT_LABEL = {"Q1Q2": "Q1+Q2"}   # the rest of the keys already ARE their acronym (WAI-SR, CSQ-8, R:Q…)


def short_label(metric: str) -> str:
    """Compact acronym-only label for DENSE figures (correlation matrices, packed axes).

    The full :func:`display_label` is ``"ACRONYM (descriptive gloss)"`` — great for panel titles and
    tables, but it overflows a 10×10 heatmap tick. This returns just the acronym (the metric key, which
    already is the instrument acronym; ``Q1Q2→"Q1+Q2"``), still ↓-flagged for lower-is-better. The
    descriptive gloss lives in the surrounding caption/legend instead.
    """
    base = _SHORT_LABEL.get(metric, metric)
    return f"{base} ↓" if metric in LOWER_IS_BETTER else base


def arm_label(arm: str) -> str:
    """Readable arm label: ``"PTO_LA0"`` -> ``"PTO (K=0)"`` (auto-parses any ``LA<k>``).

    Unknown labels pass through unchanged. Label layer only — the canonical ``arm`` key is what
    every figure hues/filters on, so only the *displayed* text is swapped.
    """
    if arm in ARM_LABELS:
        return ARM_LABELS[arm]
    m = _ARM_RE.match(arm or "")
    return f"{m.group(1)} (K={m.group(2)})" if m else arm


# Patient-characteristic columns recovered per persona.
PERSONA_COLS = ["gender", "age_value", "problem", "problem_time",
                "tried_to_solve", "cooperation_level"]


# Lexical affirmation cue (case-insensitive, per therapist turn / completion). A DIRECTIONAL
# sanity-check on the oracle's affirmation counts, NOT a primary metric — shared by
# ``behavior`` (lex_affirm_marker_rate) and ``pref`` (chosen/rejected text features).
RE_AFFIRM = re.compile(r"\byou are\b|\byou're (worthy|enough|strong|powerful|brave|amazing|a )", re.I)
