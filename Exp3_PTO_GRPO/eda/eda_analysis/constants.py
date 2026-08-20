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
    # Added 2026-06-14 alongside the 5 global-eval rubrics (see EXTRA_METRICS below):
    "PCT":    ("PCT",    "PCT_ChangeProp"),   # patient change-talk proportion CT/(CT+ST); higher = better
    "MICI":   ("MICI",   "MICI_Rate"),        # MI-inconsistent behaviors per therapist turn; LOWER = better
}
# Left-to-right plot order: the global-eval rubrics (+ Q1/Q2 components) then the added metrics.
QUESTIONNAIRE_ORDER = ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI", "Q1", "Q2"]

# The 5 global-evaluation rubrics that share the dominant PC1 factor (the empirical halo /
# redundancy set — NOT one official construct). "WARMTH_RUBRICS" is the historical code name,
# kept for API stability; prose should say "global-evaluation (halo) cluster".
WARMTH_RUBRICS = ["Q1Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI"]
# The metrics added on top of the 5 global-eval rubrics (incl. the free derived MITI-proficiency
# ratios). Membership list only — plot order + the factor space. It makes NO independence claim:
# empirically PCT loads WITH the global-eval rubrics (Spearman ~0.79–0.94), and only MICI + the
# MITI ratios sit off PC1. Do NOT call these "orthogonal axes" in code or prose — they are simply
# additional evaluation metrics, reported flat alongside the rubrics (MICI is lower-is-better).
EXTRA_METRICS = ["PCT", "MICI", "R:Q", "%CR", "%MICO"]

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
    # The raw per-session COUNTS of the same behaviours carry the same valence as their rates.
    # Omitting them would silently colour a count bar as unvalenced in any valence-aware figure.
    "MICI_BehaviorTotal", "MICI_OverPraise", "MICI_AdviseNoPermission", "MICI_Confront",
    "MICI_Warn", "MICI_Direct", "MICI_Judge",
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
    # Standalone questionnaires of their own (NOT MITI-derived).
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
    # The same MICI behaviours as RAW PER-SESSION COUNTS (the denominator control — see
    # behavior.MICI_COUNT_CHANNELS). Labelled "/ session" so a count is never read as a rate.
    "MICI_BehaviorTotal": "MI-Incon. acts / session (MICI)",
    "MICI_OverPraise": "Over-praise / session (MICI)",
    "MICI_AdviseNoPermission": "Advise w/o permission / session (MICI)",
    "MICI_Confront": "Confront / session (MICI)", "MICI_Warn": "Warn / session (MICI)",
    "MICI_Direct": "Direct/order / session (MICI)", "MICI_Judge": "Judge/label / session (MICI)",
    # PCT (PATIENT change-talk) detail — patient-perspective globals (1-5) + utterance proportions.
    "PCT_Importance": "Importance (PCT)", "PCT_Confidence": "Confidence (PCT)",
    "PCT_Readiness": "Readiness (PCT)", "PCT_GlobalMean": "PCT global mean",
    "PCT_ChangeProp": "Change-Talk proportion (PCT)", "PCT_ChangeTalk_prop": "% Change Talk (PCT)",
    "PCT_SustainTalk_prop": "% Sustain Talk (PCT)", "PCT_Neutral_prop": "% Neutral (PCT)",
    # Deterministic text metrics — NOT MITI; each carries its own source tag.
    "q_per_turn": "Questions / turn (regex ?)", "q_per_turn_miti": "Questions / turn (MITI)",
    "mean_turn_len": "Turn length (chars)", "loop": "Degeneration %",
    "conv_len": "Conversation length", "n_th_turns": "Therapist turns",
    # TRAINING-side lexical features (pref.py). Two families that must never be read as one:
    # `w_*` = what the update SELECTS FOR within a group (chosen-rejected units, can be negative);
    # `pool_*` = what the policy GENERATES over every candidate (a level, never negative). The
    # mechanism panel stacks them, so the labels have to distinguish them at a glance.
    "w_len": "Selected for: turn length (chars)", "w_question": "Selected for: questions / turn",
    "w_affirm": "Selected for: affirmation markers", "w_overpraise": "Selected for: over-praise markers",
    "pool_len": "Generated: turn length (chars)", "pool_question": "Generated: questions / turn",
    "pool_affirm": "Generated: affirmation markers", "pool_overpraise": "Generated: over-praise markers",
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


def k_of(arm: str) -> int:
    """Look-ahead K parsed out of an arm label; ``0`` for anything unparseable (incl. ``"Base"``).

    THE canonical parse. It lived in ten places across the analysis and plotting modules with five
    mutually inconsistent bodies — ``endswith("LA5")`` (reads a hypothetical K=3 arm as K=0),
    ``int(arm.split("_LA")[1])`` (raises on ``"Base"``), and three regex/split variants. Since K is
    both a *style* key (solid vs dashed) and a *grouping* key, disagreeing parses meant a new K arm
    would be mis-styled in some figures and mis-grouped in others, silently rather than loudly.
    """
    m = _ARM_RE.match(arm or "")
    return int(m.group(2)) if m else 0


def method_of(arm: str) -> str:
    """``"PTO"`` / ``"GRPO"`` parsed out of an arm label; ``""`` when it is neither."""
    m = _ARM_RE.match(arm or "")
    return m.group(1) if m else ""


# Patient-characteristic columns recovered per persona.
PERSONA_COLS = ["gender", "age_value", "problem", "problem_time",
                "tried_to_solve", "cooperation_level"]

# Display labels + plot order for the `cooperation_level` persona trait (32 personas per level).
# ⚠ ONE canonical copy on purpose. This map was duplicated into four modules and had already
# DRIFTED: `replication.py` rendered StartLowAndChangesToHigh as "WarmsUp" with a Resistant-first
# order while `faithfulness`/`instruments`/`plotting.heterogeneity` used "Warms up" with the
# reverse — so the same persona stratum shipped under two spellings, both of them inside
# `results/lookahead/INDEX.md`, and any join keyed on the label broke across families.
COOP_LABEL = {"High": "Cooperative", "StartLowAndChangesToHigh": "Warms up", "Low": "Resistant"}
COOP_ORDER = ["Cooperative", "Warms up", "Resistant"]

# Column-name slug for the same three strata. ⚠ An IDENTIFIER must not be a display string —
# conflating the two is exactly how the drift above happened, because `replication.py` built
# column names like ``share_ge45_{label}`` and so could not change its label without renaming
# columns. The slug is the historical no-space form, so ``n_WarmsUp`` / ``share_ge45_WarmsUp``
# stay byte-comparable with the frozen paper table
# (papers/2026_lookahead_pto_grpo/tables/session_shape_stability_ceiling.md) while the *displayed*
# label converges with every other family.
COOP_SLUG = {"Cooperative": "Cooperative", "Warms up": "WarmsUp", "Resistant": "Resistant"}



# THE resampling seed for every bootstrap in the EDA — `stats.bootstrap_ci` AND the CI bands
# seaborn draws inside the figures. It lives in this leaf so both sides share one value.
#
# ⚠ Seaborn's `errorbar=("ci", 95)` defaults to `seed=None`, i.e. a fresh 1,000-sample bootstrap on
# every call. Left unset, the tracked figures were NOT reproducible: three consecutive renders of
# the same notebook on identical data differed by ~6% of pixels, and every `results/` PNG churned
# in git on each render (found 2026-07-28). Pass `seed=BOOT_SEED` at every seaborn callsite that
# draws a bootstrap errorbar.
BOOT_SEED = 12345

# Lexical affirmation cue (case-insensitive, per therapist turn / completion). A DIRECTIONAL
# sanity-check on the oracle's affirmation counts, NOT a primary metric — shared by
# ``behavior`` (lex_affirm_marker_rate) and ``pref`` (chosen/rejected text features).
RE_AFFIRM = re.compile(r"\byou are\b|\byou're (worthy|enough|strong|powerful|brave|amazing|a )", re.I)
# The over-praise cue (effusive, "you are a beacon" register). Lives here rather than in behavior.py
# because BOTH the conversation side (behavior.text_metrics) and the training side
# (pref.weighted_lexical_contrast) test the same drift — one regex, or the two sides stop agreeing.
RE_EFFUSIVE = re.compile(
    r"\bi'?m so proud|proud of you|inspiration to me|you got this|beautiful|beacon|"
    r"shining|warrior|hero of your|you are a (light|beacon)", re.I)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACTIVE JUDGE — which grader's scores the whole EDA reads                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Orthogonal to the VIEW (which filters ARMS). The judge selects the SCORE SOURCE:
#   ""                              -> the primary oracle (resolves to PRIMARY_JUDGE_TAG), default
#   "anthropic_claude-haiku-4-5"    -> a second judge
# Either way the path is data/eval_scores/judge=<tag>/rep=<r>/ — see EVAL_SCORES below.
#
# State lives here, in the leaf, so BOTH `data` (which resolves per-metric score directories) and
# `exports` (which routes results/) can read it without an import cycle. Set it once per session
# via `EdaConfig.judge` -> `notebook_setup`; never poke these globals directly from a notebook.
#
# ⚠ ONLY EVAL-SCORE-DERIVED ANALYSES ARE JUDGE-SWAPPABLE. Anything reading the training side —
# `generations.jsonl` candidate rewards, PTO preference pairs, TensorBoard curves — was produced by
# the TRAINING oracle during the run and cannot be re-graded after the fact. Re-rendering those
# under a second judge would emit byte-identical figures into that judge's folder, implying a
# measurement that never happened. `notebook_setup` warns when a training-side notebook runs with a
# non-primary judge; see eda/README.md § "Judge dimension".

# ── THE SCORE LAKE ────────────────────────────────────────────────────────────
# ONE tree holds every score any grader ever produced:
#
#   data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<file_index>.csv
#
# `judge` is an ordinary partition key alongside metric/oracle/rep — no grader is privileged by
# layout, and there is a single resolver (`Arm.eval_dir`) rather than a primary-vs-other branch.
# `<Model>` already encodes the method (GRPOExp3_* / PTOExp3_*), so the tree is method-flat.
#
# **rep=0 is the full-grid draw for EVERY judge** (every scored model state × 8 rubrics × 96 convs;
# 39 × 8 × 96 = 29,952 cells at the time of writing — read the live count off
# results/*/tables/8_measurement/multijudge_coverage.md, never from this comment) and is
# what the EDA reads by default; reps ≥1 are the repeatability draws on the anchor subset. Reps of
# one judge differ only by scoring seed (`1000+rep` on the OpenAI path; the Anthropic path has no
# seed and varies by inherent API nondeterminism), so they are exchangeable — which is what makes
# ICC(2,1) across them meaningful.
#
# The lake is a Google Drive symlink, so it is backed up and reachable from Colab. Before
# 2026-07-28 the primary's production draw lived co-located per method
# (data/<method>_Exp3/eval_scores/) while every other grader lived in a local-only
# data/eval_scores_by_judge/ tree: the primary was split across two roots under two different
# partition schemes, and the second judge's $50 of scores were backed up nowhere.
EVAL_SCORES = os.path.join(DATA_DIR, "eval_scores")
JUDGE_PARTITION = "judge="            # key=value partition level, matching metric=/oracle=/rep=

# The grader that produced the primary scores — and, because it was also the training reward, the
# one every other judge is held out against. `EdaConfig.judge=""` resolves here, so "no judge set"
# and "the primary judge" name the same partition instead of the same special case.
PRIMARY_JUDGE_TAG = "openai_gpt-4o-mini-2024-07-18"
_JUDGE_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def judge_partition_dir(tag: str = "") -> str:
    """``data/eval_scores/judge=<tag>`` — the root of one judge's scores.

    An empty tag resolves to the primary oracle, matching ``EdaConfig.judge=""``.
    """
    return os.path.join(EVAL_SCORES, f"{JUDGE_PARTITION}{tag or PRIMARY_JUDGE_TAG}")

_ACTIVE_JUDGE = ""
_ACTIVE_JUDGE_REP = 0


def set_active_judge(tag: str = "", rep: int = 0) -> None:
    """Select the score source for every subsequent load. ``""`` = the primary eval_scores tree."""
    global _ACTIVE_JUDGE, _ACTIVE_JUDGE_REP
    _ACTIVE_JUDGE = (tag or "").strip().strip("/\\")
    _ACTIVE_JUDGE_REP = int(rep)


def active_judge() -> str:
    """Active judge tag, or ``""`` for the primary oracle (see :data:`PRIMARY_JUDGE_TAG`)."""
    return _ACTIVE_JUDGE


def active_judge_rep() -> int:
    return _ACTIVE_JUDGE_REP


def judge_dirname(tag: str = None) -> str:
    """Short model label used as the export-path segment for a judge.

    Drops the provider prefix and any trailing ISO release date, so
    ``openai_gpt-4o-mini-2024-07-18 -> gpt-4o-mini`` and
    ``anthropic_claude-haiku-4-5 -> claude-haiku-4-5``. The DATA tree keeps the full tag
    (``eval_scores/judge=<tag>/``) because that is a stable partition key; the RESULTS tree uses the
    short label because a human reads those paths. ``tag=None`` resolves the active judge, and the
    primary oracle resolves to its own label rather than to a flat path — every grader gets a
    folder, so no tree is privileged by layout.
    """
    t = (active_judge() if tag is None else tag) or PRIMARY_JUDGE_TAG
    t = t.split("_", 1)[-1]                       # provider prefix
    return _JUDGE_DATE_SUFFIX.sub("", t)          # trailing -YYYY-MM-DD release date


# Back-compat: `judge_label` was the pre-2026-07-28 name, when the primary rendered as "primary".
judge_label = judge_dirname
