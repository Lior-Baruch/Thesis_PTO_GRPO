# Exp3 metrics & EDA-check reference

A single cheat-sheet for **what every number in the Exp3 EDA is, what it measures, and how it's
computed**. Everything is scored on the same object: a therapist(Llama-3.2-1B)↔patient(gpt-4o-mini)
conversation. Two data sources feed the EDA:

- **Oracle (gpt-4o-mini, JSON-schema output)** — grades a full transcript against MI questionnaires.
  Defined in [code/questionnaires.py](../code/questionnaires.py); scored by [Run_Eval.ipynb](Run_Eval.ipynb) → `data/<method>_Exp3/eval_scores/`.
- **Deterministic text metrics** — cheap regex/counting over the raw transcript, no LLM. Defined in
  [eda_analysis/behavior.py](eda_analysis/behavior.py). These cross-check the oracle.

> **Two valences.** Almost everything is *higher = better*. The one exception is **MICI** (and its
> sub-counts): *lower = better*. The EDA flags these with a trailing `↓` (`display_label`) and the
> package set `LOWER_IS_BETTER` = `{MICI, MICI_Severity, MICI_Rate, the 6 MICI_*_rate detail columns,
> PCT_SustainTalk_prop}`.

---

## 1 · Questionnaires (the oracle instruments)

Each is a validated (or validated-style) MI questionnaire the oracle fills in from the patient's or a
coder's perspective. `ID` = the `questionnaire_id`; `Scale` = per-item Likert range; `Perspective` =
whose point of view the oracle adopts.

| Name | ID | Items | Scale | Perspective | What it measures | The per-conv number the EDA uses |
|---|---|---|---|---|---|---|
| **Q1** | 1 | 5 | 1–5 | Patient | Overall satisfaction, motivation, learning, real-life relevance | `Q1_Mean` (mean of 5 items) |
| **Q2** | 2 | 17 | 1–5 | Patient | Relational qualities: warmth, empathy, understanding, non-judgment, connection | `Q2_Mean` (mean of 17 items) |
| **Q1+Q2** | — | 22 | 1–5 | Patient | **The TRAINING reward** (composite, matches the ICLR paper) | `Q1Q2_Mean` = mean(`Q1_Mean`, `Q2_Mean`) |
| **WAI-SR** | 3 | 12 | 1–5 | Patient | Working alliance = **Goal + Task + Bond** subscales | `WAI_TotalMean` |
| **CSQ-8** | 4 | 8 | 1–4 | Patient | Client satisfaction with the "service" (quality, needs met, would-recommend) | `CSQ8_Mean` |
| **MI-SAT** | 6 | 6 | 1–5 | Patient | Satisfaction with the MI intervention (helpful, enjoyable, worth the time) | `MI_Mean` |
| **MITI** | 7 | 4 globals + 7 counts | globals 1–5 | MI coder (therapist) | MI Treatment Integrity: how technically MI-consistent the therapist is | `MITI_GlobalMean` (mean of 4 globals) + behavior counts (§3) |
| **PCT** | 8 | 3 globals + 3 counts | globals 1–5 | MI coder (patient) | **Patient** change-talk: did the *client* express motivation? | `PCT_ChangeProp` = CT / (CT + ST) |
| **MICI** ↓ | 9 | 1 global + 6 counts | global 1–5 | MI coder (therapist) | **MI-INCONSISTENT** therapist moves (confront, unsolicited advice, over-praise/sycophancy). **Lower = better** | `MICI_Rate` = inconsistent behaviors / therapist turn |

**Groupings the EDA relies on** (from `eda_analysis/__init__.py`):

- **Warmth rubrics** (`WARMTH_RUBRICS`) = `Q1+Q2, WAI-SR, CSQ-8, MI-SAT, MITI`. These 5 are the
  subjective "does the patient feel good" cluster — they collapse onto **one PC1 factor** (~91% of
  variance before the orthogonal axes were added). Moving them all up together is *not* proof of
  multi-skill improvement.
- **Orthogonal axes** (`ORTHOGONAL_METRICS`) = `PCT, MICI↓, R:Q, %CR, %MICO` (§2). Added specifically
  to break the warmth halo. Adding them drops PC1 from ≈91% → ≈55% — warmth is one factor, technique +
  MI-inconsistency form a genuine second.

**MITI globals** (part of ID 7, each 1–5): `MITI1_CultivatingChangeTalk`, `MITI2_SofteningSustainTalk`,
`MITI3_Partnership`, `MITI4_Empathy`. **PCT globals**: `PCT_Importance`, `PCT_Confidence`,
`PCT_Readiness`. **MICI global**: `MICI_Severity`.

**Per-item detail plots (2026-07-07).** Every global + behaviour of the three count-based instruments
now has its own trajectory, so the aggregates (`MITI_GlobalMean` / `MICI_Rate` / `PCT_ChangeProp`) are no
longer black boxes: the 4 MITI globals are the §2 subscale grid; all 7 MITI behaviours (incl. the
previously-omitted `B1_GI`/`B7_Seek`) are the §1 drift grid; `MICI_Severity` + the 6 MI-inconsistent
behaviours (per therapist turn) are **§4d** `mici_behavior_detail`; the 3 PCT globals + change/sustain/
neutral proportions are **§4e** `pct_patient_detail`. Loaders: `behavior.mici_behavior_by_iter` /
`behavior.pct_behavior_by_iter` / `behavior.load_pct_behavior`.

---

## 2 · Derived MI-proficiency ratios (free, no oracle re-run)

Computed from the MITI behavior counts in `data.py::add_derived_mitiprof_rows`. These are **objective
technique** metrics (not warmth), so they're treated as candidate orthogonal axes. All *higher = better*.

| Metric | Formula | Reads as |
|---|---|---|
| **R:Q** (Reflection:Question) | `(SR + CR) / Q` | Reflective listening vs interrogating. Good MI is reflection-heavy. |
| **%CR** (% Complex Reflections) | `CR / (SR + CR)` | Depth of reflection — complex reflections add meaning, not just mirror. |
| **%MICO** (% MI-Consistent) | `(SR + CR + AF + Seek) / (SR + CR + AF + Seek + Persuade)` | Share of "good MI" moves vs the one MI-inconsistent behavior MITI counts (persuade). |

(`SR`=simple reflections, `CR`=complex reflections, `Q`=questions, `AF`=affirmations, `Seek`=seeking
collaboration, `Persuade`=persuasion — all from §3.)

**Per-therapist-turn rates (2026-07-07).** `behavior_by_iter` also emits each length-scaling MITI count as
a rate — `B3_Q_per_turn`, `B4_SR_per_turn`, `B5_CR_per_turn`, `B6_AF_per_turn`, `B2_Persuade_per_turn`,
`B1_GI_per_turn`, `B7_Seek_per_turn` (= count ÷ therapist turns, mean-of-ratios) — and the behaviour-drift
figure plots the **rates**, not the raw counts, so a longer late-iteration conversation doesn't
mechanically inflate them. The MICI detail (§4d) uses the same per-therapist-turn convention
(`MICI_*_rate`); the PCT detail (§4e) uses proportions of patient utterances (`PCT_*_prop`, ÷
`PCT_BehaviorTotal`).

---

## 3 · Behavior metrics (what the therapist actually does)

Two cross-validating sources. **Oracle MITI counts** are the professional coder's tally;
**deterministic text metrics** are cheap regex counts that confirm the direction and catch things the
oracle misses (degeneration loops). Trajectory backbone: `behavior.py::behavior_by_iter`.

### 3a · Oracle MITI behavior counts (`load_miti_behavior`)
Per-conversation counts of each coded therapist move (one code per therapist utterance; counts sum to
the number of therapist turns).

| Code | Name | Valence | Meaning |
|---|---|---|---|
| `B1_GI` | Giving Information | neutral | Education / feedback / info provision |
| `B2_Persuade` | Persuasion | ✗ MI-inconsistent | Trying to influence/advise toward change (incl. with permission) |
| `B3_Q` | Questions | ✓ | All therapist questions (open + closed) |
| `B4_SR` | Simple Reflections | ✓ | Mirroring client content |
| `B5_CR` | Complex Reflections | ✓✓ | Paraphrase / metaphor / added meaning |
| `B6_AF` | Affirmations | ✓ (but watch drift) | Recognizing genuine strength/effort. **Runaway B6_AF = the over-praise reward-hack** |
| `B7_Seek` | Seeking Collaboration | ✓ | Inviting the client's input/choice |
| `Empathy`, `ChangeTalk`, `Partnership` | MITI globals | ✓ | The 1–5 global ratings (see §1) |
| `RtoQ` | Reflection:Question ratio | ✓ | `(SR + CR) / Q`, per conversation (= R:Q at conv level) |

### 3b · Deterministic text metrics (`text_metrics`)
Regex/counting over the transcript — no LLM, fully reproducible.

| Metric | Definition | Why it's here |
|---|---|---|
| `n_th_turns` | # therapist turns | Denominator for the rates |
| `mean_turn_len` | Mean chars per therapist turn | Length-hacking / verbosity signal |
| `max_repeat` | Max count of any verbatim-identical therapist turn | Raw degeneration signal |
| `loop` | `max_repeat ≥ 2` (bool) → **degeneration %** when averaged | Catches phrase-loop collapse the oracle floors but doesn't itemize (0.49→0 over training) |
| `q_per_turn` | `?`-count per therapist turn | Deterministic question rate (see §4) |
| `conv_len` | # utterances in the conversation | Session shape |

### 3c · Lexical marker rates — **sanity-check ONLY** (`lex_*`)
`lex_affirm_marker_rate`, `lex_overpraise_marker_rate` — brittle keyword regexes ("you're amazing",
"I'm so proud", "beacon"…). **Deliberately excluded from the headline behavior metrics.** They exist
only to validate the *direction* of the oracle's `B6_AF` / `MICI_OverPraise`. For the real
affirmation/over-praise story, always use the oracle-coded counts, never these.

---

## 4 · Question rate (and its cross-check)

Two ways to measure "how much is the therapist asking questions", intentionally unit-harmonized to
**questions per therapist turn**:

| Metric | Source | Definition |
|---|---|---|
| `q_per_turn` | Deterministic (text) | Literal `?` count / therapist turns |
| `q_per_turn_miti` | Oracle | MITI `B3_Q` / therapist turns |

**`behavior.question_rate_crosscheck`** puts them side by side per (arm, iteration); the figure
`plotting.question_rate_crosscheck` overlays them per arm. They should track each other
(cross-validation). Their **late divergence is itself the finding**: in GRPO's late iterations
`B3_Q ≈ 4.1` but `q_per_turn ≈ 0.15` — praise-heavy turns still register as "question-function"
utterances to the coder but no longer carry a literal `?`. (Audited 2026-07-03: NOT a bug — the merge
is conv-aligned 96/96 with harmonized denominators; it's a real question-**syntax** vs question-**function**
gap: late affirmation/advice turns carry question-function without a `?`.)

---

## 5 · Reward-hacking checks (the "is warmth genuine?" battery)

The core RQ-ii worry: both methods can raise the warmth reward by **over-praising / sycophancy** rather
than doing real MI. These figures/checks are how the EDA exposes it.

| Check / figure | Where | What it shows |
|---|---|---|
| **`reward_hack_panel`** | `3_Mechanism` | The hack in one frame: per arm, twin y-axis — warmth (`Q1+Q2`, left) **climbs** while `MICI↓` (MI-inconsistency, right) **climbs with it** and `PCT` (real patient change-talk) barely moves. "All rubrics up" ≠ multi-skill. |
| **Peak-then-regress marking** | `single_metric_trajectory(mark_peaks=True)`, `1_Outcomes` | Auto-draws a vline at any arm's peak iteration *only if it regressed after* — surfaces **GRPO's iter-8 peak (4.08) → iter-10 (3.75)** without hardcoding. PTO climbs stably. |
| **Affirmation drift** | `behavior_by_iter` / behavior trajectories, `3_Mechanism` | `B6_AF` rises (PTO 0.42→1.64; GRPO 0.52→**1.98** by iter 10) while `B3_Q` falls — confirmed in **both** methods, worse in late GRPO. |
| **`overpraise_crosscheck`** | `behavior.py` + `3_Mechanism` | Lexical over-praise marker rate beside the oracle's `MICI_OverPraiseRate` — validates the sycophancy direction. |
| **`MICI_Rate` trajectory** | `2`/`3` | MI-inconsistent behavior per turn rises with warmth: ~2.3× PTO / ~4× GRPO at the iter-10 endpoint (base 0.21 → 0.49 PTO / 0.84 GRPO; GRPO's iter-8 peak was 0.54). |
| **`subgroup_endpoint_bars`** | `2_Heterogeneity` | Final-iteration score per persona × arm — shows GRPO's late regression **concentrates on Resistant (low-cooperation) personas**. |
| **`effect_forest`** | `1_Outcomes` | Each arm×rubric Δ-vs-base with 95% CI + `dz`; MICI is direction-colored (a positive Δ is *bad*). Readable stand-in for the 28-row table. |
| **PCA / `factor_loadings_bars`** | `3_Mechanism` / `6_Stats` | PC1 share drops ≈91% → ≈55% once orthogonal axes are added → warmth is one factor, technique+MICI a genuine second. |
| **`question_rate_crosscheck`** | `3_Mechanism` | (§4) — questions collapsing while warmth rises is part of the same drift. |

---

## 6 · Reward-faithfulness (why MIN_CONV_LENGTH exists)

Separate but related: the **training** reward scores *partial* conversations, but the thesis evaluates
*full* ones. `4_Training_and_Reliability` rebuilds the partial-conv reliability curve on Exp3 data
(`stats.rank_agreement_by_nturns`, from `generations.jsonl`): short cuts (`n_turns=2`) agree with the
final-conv ranking only ~0.66–0.73 (barely above chance), clearing 0.8 at ~10 turns, 0.9 at ~30.
Motivates the `MIN_CONV_LENGTH` knob (drop training slices shorter than N utterances).

---

### Quick map: figure family → notebook
`1_Outcomes` (trajectories, effect forest, scorecard) · `2_Heterogeneity` (persona splits, endpoint
bars) · `3_Mechanism` (behavior drift, reward_hack_panel, question/over-praise cross-checks, factor
structure) · `4_Training_and_Reliability` (TB curves, reward dist, reliability curve) · `5_Preference`
(PTO preference probe) · `6_Stats` (all heavy tables + PCA).
