# Exp3 metrics & EDA-check reference

A single cheat-sheet for **what every number in the Exp3 EDA is, what it measures, and how it's
computed**. Everything is scored on the same object: a therapist(Llama-3.2-1B)↔patient(gpt-4o-mini)
conversation. Two data sources feed the EDA:

- **Oracle (gpt-4o-mini, JSON-schema output)** — grades a full transcript against MI questionnaires.
  Defined in [code/questionnaires.py](../../code/questionnaires.py); scored by
  [Run_Eval.ipynb](../notebooks/scoring/Run_Eval.ipynb) into the score lake,
  `data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv`. A second
  judge writes the identical shape under its own `judge=` partition.
- **Deterministic text metrics** — cheap regex/counting over the raw transcript, no LLM. Defined in
  [eda_analysis/behavior.py](../eda_analysis/behavior.py). These cross-check the oracle.

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
| **Q1** | 1 | 5 | 1–5 | Patient | Session satisfaction: overall satisfaction, motivation, learning, real-life relevance | `Q1_Mean` (mean of 5 items) |
| **Q2** | 2 | 17 | 1–5 | Patient | Working alliance / relational communication: warmth, empathy, understanding, non-judgment, connection | `Q2_Mean` (mean of 17 items) |
| **Q1+Q2** | — | 22 | 1–5 | Patient | **The TRAINING reward** (composite, matches the ICLR paper) | `Q1Q2_Mean` = mean(`Q1_Mean`, `Q2_Mean`) |
| **WAI-SR** | 3 | 12 | 1–5 | Patient | Working alliance = **Goal + Task + Bond** subscales | `WAI_TotalMean` |
| **CSQ-8** | 4 | 8 | 1–4 | Patient | Client satisfaction with the "service" (quality, needs met, would-recommend) | `CSQ8_Mean` |
| **MI-SAT** | 6 | 6 | 1–5 | Patient | Satisfaction with the MI intervention (helpful, enjoyable, worth the time) | `MI_Mean` |
| **MITI** | 7 | 4 globals + 7 counts | globals 1–5 | MI coder (therapist) | MI Treatment Integrity: how technically MI-consistent the therapist is | `MITI_GlobalMean` (mean of 4 globals) + behavior counts (§3) |
| **PCT** | 8 | 3 globals + 3 counts | globals 1–5 | MI coder (patient) | **Patient** change-talk: did the *client* express motivation? | `PCT_ChangeProp` = CT / (CT + ST) |
| **MICI** ↓ | 9 | 1 global + 6 counts | global 1–5 | MI coder (therapist) | **MI-INCONSISTENT** therapist moves (confront, unsolicited advice, over-praise/sycophancy). **Lower = better** | `MICI_Rate` = inconsistent behaviors / therapist turn |

**Instrument provenance** (what "validated" means per instrument — cite accordingly in the thesis):

- **Q1 (Session Satisfaction)** and **Q2 (Working Alliance / Relational Communication)** are the
  published LLM-evaluator prompts from the lab's CLPsych 2024 paper: *Yosef, Zisquit, Cohen,
  Brunstein Klomek, Bar & Friedman (2024), "Assessing Motivational Interviewing Sessions with
  AI-Generated Patient Simulations", Proc. CLPsych @ EACL 2024*
  ([ACL Anthology 2024.clpsych-1.1](https://aclanthology.org/2024.clpsych-1.1/)). That paper
  validates them **as LLM evaluators** (ratings statistically reliable; distinguish three levels
  of therapist expertise) — the relevant validation basis for this LLM-graded pipeline. Do NOT
  present Q2 as the WAI itself.
- **WAI-SR** (Hatcher & Gillaspy 2006) and **CSQ-8** (Larsen/Attkisson et al. 1979) are classically
  validated human-report scales, here completed by the oracle in the patient's voice.
- **MI-SAT** is an adapted MI-intervention satisfaction survey (validated-style, not canonical).
- **MITI 4.2** (Moyers et al.) is the official MI treatment-integrity coding system; **PCT** and
  **MICI** are custom MITI-style coders (change-talk / MI-inconsistent behavior) built for Exp3.

**Groupings the EDA relies on** (from `eda_analysis/__init__.py`):

- **Global-evaluation / halo cluster** (`WARMTH_RUBRICS` — historical code name, kept for
  stability) = `Q1+Q2, WAI-SR, CSQ-8, MI-SAT, MITI`. An **empirical redundancy set, not an
  official construct**: these 5 subjective/global ratings collapse onto **one PC1 factor**
  (~91% of variance before the further metrics were added) — the single-oracle halo. Their
  constructs even overlap by design (Q2 and WAI-SR are both alliance measures). Moving them
  all up together is *not* proof of multi-skill improvement.
- **Added metrics** (`EXTRA_METRICS` — a membership list for plot order + the factor space, **no
  independence implied**) = `PCT, MICI↓, R:Q, %CR, %MICO` (§2). Added to test whether anything
  measures outside the halo. Adding them drops PC1 from ≈91% → ≈55%, but the split is **not** the
  one intended: `PCT` loads WITH the 5 rubrics (ρ≈0.79–0.94); only `MICI↓` + the MITI ratios define
  the second factor. Report all of them flat as evaluation metrics — do **not** call them
  "orthogonal axes" (see [LIMITATIONS.md](LIMITATIONS.md) §4).

**MITI globals** (part of ID 7, each 1–5): `MITI1_CultivatingChangeTalk`, `MITI2_SofteningSustainTalk`,
`MITI3_Partnership`, `MITI4_Empathy`. **PCT globals**: `PCT_Importance`, `PCT_Confidence`,
`PCT_Readiness`. **MICI global**: `MICI_Severity`.

**Per-item / per-component detail plots (2026-07-07; reorganized into `arms/questionnaires.ipynb`
2026-07-16).** Every rubric now has a uniform drill-down grid, so no aggregate is a black box:
the 4 MITI globals + all 7 MITI behaviours (incl. the previously-omitted `B1_GI`/`B7_Seek`) + the
proficiency ratios are the §6 `miti_detail_grid`; `MICI_Severity` + the 6 MI-inconsistent behaviours
(per therapist turn) are the §8 `mici_detail_grid`; the 3 PCT globals + change/sustain/neutral
proportions are the §7 `pct_detail_grid`; and the Likert-item rubrics (Q1/Q2/WAI-SR/CSQ-8/MI-SAT)
get per-item grids (`<slug>_detail_grid`) + "which items drive the change" delta bars at final AND
best (`<slug>_item_deltas_*`). Loaders: `data.load_items` (generic, over
`constants.ITEM_QUESTIONNAIRES`) / `behavior.miti_detail_by_iter` / `behavior.mici_behavior_by_iter` /
`behavior.pct_behavior_by_iter`; deltas: `stats.item_endpoint_deltas`.

---

## 2 · Derived MI-proficiency ratios (free, no oracle re-run)

Computed from the MITI behavior counts in `data.py::add_derived_mitiprof_rows`. These are **objective
technique** metrics (not warmth), so they are reported alongside the questionnaire rubrics. All *higher = better*.

| Metric | Formula | Reads as |
|---|---|---|
| **R:Q** (Reflection:Question) | `(SR + CR) / Q` | Reflective listening vs interrogating. Good MI is reflection-heavy. |
| **%CR** (% Complex Reflections) | `CR / (SR + CR)` | Depth of reflection — complex reflections add meaning, not just mirror. |
| **%MICO** (% MI-Consistent) | `(SR + CR + AF + Seek) / (SR + CR + AF + Seek + Persuade)` | Share of "good MI" moves vs the one MI-inconsistent behavior MITI counts (persuade). |

(`SR`=simple reflections, `CR`=complex reflections, `Q`=questions, `AF`=affirmations, `Seek`=seeking
collaboration, `Persuade`=persuasion — all from §3.)

### 2b · Official MITI 4.2.1 summary scores + competency thresholds

The MITI 4.2.1 manual (Moyers, Manuel & Ernst 2014; manual rev. June 2015, §H–I) defines four
summary scores with suggested **basic competence ("fair") / proficiency ("good")** thresholds —
computed for free from the stored MITI globals + counts (`behavior.miti_proficiency_by_iter`;
constants in `eda_analysis.MITI_THRESHOLDS`; figure/table in `arms/questionnaires.ipynb` §6b):

| Summary score | Formula | Fair | Good |
|---|---|---|---|
| **R:Q** | total reflections / total questions | 1:1 | 2:1 |
| **%CR** | CR / (SR + CR) | 40% | 50% |
| **Technical global** | (CultivatingChangeTalk + SofteningSustainTalk) / 2 | 3.0 | 4.0 |
| **Relational global** | (Partnership + Empathy) / 2 | 3.5 | 4.0 |

⚠ **Caveats** (state them wherever the thresholds are drawn): the manual itself flags the
thresholds as *expert opinion without normative validation* (MIA/MINA thresholds intentionally
unspecified); the MITI is designed for ~20-min human audio sessions, so short text chats are
out-of-domain — use as an anchor, not a certification. Note Technical/Relational are the manual's
2-global splits, **not** our 4-global `MITI_GlobalMean`. Also note R:Q can improve via the
pathological route (fewer questions shrinking the denominator — GRPO's iter 10): read it against
`B3_Q_per_turn`.

**Per-therapist-turn rates (2026-07-07).** `behavior_by_iter` also emits each length-scaling MITI count as
a rate — `B3_Q_per_turn`, `B4_SR_per_turn`, `B5_CR_per_turn`, `B6_AF_per_turn`, `B2_Persuade_per_turn`,
`B1_GI_per_turn`, `B7_Seek_per_turn` (= count ÷ therapist turns, mean-of-ratios) — and the behaviour-drift
figure plots the **rates**, not the raw counts, so a longer late-iteration conversation doesn't
mechanically inflate them. The MICI detail (`arms/questionnaires.ipynb` §8) uses the same
per-therapist-turn convention (`MICI_*_rate`); the PCT detail (§7 there) uses proportions of patient
utterances (`PCT_*_prop`, ÷ `PCT_BehaviorTotal`).

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
(cross-validation). Their **late divergence is itself the finding**: in an affirmation-drifted arm
the oracle's `q_per_turn_miti` stays well above the literal-`?` rate — praise-heavy turns still
register as "question-function" utterances to the coder but no longer carry a `?`. (Audited
2026-07-03: NOT a bug — the merge is conv-aligned 96/96 with harmonized denominators; it's a real
question-**syntax** vs question-**function** gap: late affirmation/advice turns carry
question-function without a `?`.)

---

## 5 · Reward-hacking checks (the "is warmth genuine?" battery)

The core RQ-ii worry: both methods can raise the warmth reward by **over-praising / sycophancy** rather
than doing real MI. These figures/checks are how the EDA exposes it.

*(Definitions + directionality only — the current values live in `results/<top>/SUMMARY.md`, not
here.)*

| Check / figure | Where | What it shows |
|---|---|---|
| **`reward_hack_panel`** | `arms/validity.ipynb` | The hack in one frame: per arm, twin y-axis — warmth (`Q1+Q2`, left) **climbs** while `MICI↓` (MI-inconsistency, right) **climbs with it** and `PCT` (real patient change-talk) barely moves. "All rubrics up" ≠ multi-skill. |
| **Peak-then-regress marking** | `single_metric_trajectory(mark_peaks=True)`, `arms/outcomes.ipynb` | Auto-draws a vline at any arm's peak iteration *only if it regressed after* — surfaces a peak-then-regression arm (e.g. late GRPO) without hardcoding. |
| **Affirmation drift** | `behavior_by_iter` / behavior trajectories, `arms/validity.ipynb` | `B6_AF` rising while `B3_Q` falls over iterations — the over-praise drift signature. |
| **`overpraise_crosscheck`** | `behavior.py` + `arms/validity.ipynb` | Lexical over-praise marker rate beside the oracle's `MICI_OverPraiseRate` — validates the sycophancy direction. |
| **`MICI_Rate` trajectory** | `2`/`3` | MI-inconsistent behavior per therapist turn across iterations — does it rise with warmth? |
| **`subgroup_endpoint_bars`** | `arms/heterogeneity.ipynb` | Score per persona × arm at each arm's final AND best iteration (`subgroup_endpoint_<trait>_{final,best}`) — where does a late regression concentrate? |
| **`effect_forest`** | `arms/outcomes.ipynb` | Each arm×rubric Δ-vs-base with 95% CI + `dz`; MICI is direction-colored (a positive Δ is *bad*). Readable stand-in for the 28-row table. |
| **PCA / `factor_loadings_bars`** | `arms/validity.ipynb` / `arms/stats.ipynb` | PC1 share once the further metrics are added → is the global-eval halo one factor and technique+MICI a second? |
| **`question_rate_crosscheck`** | `arms/validity.ipynb` | (§4) — questions collapsing while the halo scores rise is part of the same drift. |
| **`q2_item_deltas_{final,best}` / `q2_item_group_trajectories`** | `arms/questionnaires.ipynb` §2 | The **reward-composition** view: per-item Δ vs base for Q2's 17 items (per-item scores already stored — no oracle re-run), colored by face-content group (`Q2_ITEM_GROUPS` — OUR analytical grouping, not a validated subscale). Q2 items 1/2/3/10 reward therapist *self-disclosure*, which MI does not prescribe — if those top the Δ ranking, the Q1+Q2 reward itself incentivizes the emotive drift. Loader `data.load_q2_items`; deltas `stats.q2_item_endpoint_deltas`. |
| **`miti_proficiency_thresholds` / `miti_threshold_verdicts`** | `arms/questionnaires.ipynb` §6b | The absolute anchor (this doc §2b): official-threshold verdicts per arm — did training reach basic MI competence in the manual's own terms? |

---

## 6 · Reward-faithfulness (why MIN_CONV_LENGTH exists)

Separate but related: the **training** reward scores *partial* conversations, but the thesis evaluates
*full* ones.

⚠ **The ~0.66–0.73-at-`n_turns=2` figures are Exp2's**, from the original `Partial_Conv_Oracle_EDA`
that motivated the knob (clearing 0.8 at ~10 turns and 0.9 at ~30 — the source of the recommended
MCL values). They are NOT outputs of the Exp3 rebuild and must not be cited as such.

`arms/training.ipynb` rebuilds the curve **on Exp3 data** (`stats.rank_agreement_by_nturns`, from
`generations.jsonl`), but MCL=12 means Exp3 has **no slices below 12 turns at all**, so its curve
starts where Exp2's had already recovered: agreement ≈0.86–0.89 at `n_turns=12`, and from there it
edges *up* for GRPO (≈0.90 at 50) and *down* for PTO (≈0.76 at 48). Exp3 therefore cannot confirm
or refute the short-cut finding — it is evidence that the knob is doing its job, not a replication.
Read the live numbers off `figures/5_training/<judge>/reward_reliability_curve.png` and its
`_provenance.md` (which arms were drawn), never from prose.

## 6b · The update-weighted probe (what the training signal pushes toward)

§6 asks whether the training reward *predicts* the eval score. This asks what the update actually
rewards — for **both** methods, which is possible because "preference" is not the essential thing:
each method weights the candidates of a group and steps along the weighted sum. `pref` gives every
candidate its method's weight (`arms/preference.ipynb` §3):

| method | weight `w` |
|---|---|
| PTO / DPO | `+1` on the logged `chosen`, `−1` on `rejected`, 0 otherwise — the literal τ-filtered pair |
| GRPO | `(r_g − mean_g) / std_g`, the standardized advantage that scales each completion's gradient |

Weights are rescaled **within each group to `Σ|w| = 2`** (DPO's natural ±1 size), which is what makes
the two comparable — without it GRPO's weights carry the group's reward spread and every
cross-method contrast is a scale artifact. Groups lacking either sign are dropped: for PTO that is a
branch point the τ filter left unpaired (logged `chosen`, no `rejected`), which never trained.

- **`w_len` / `w_question` / `w_affirm` / `w_overpraise`** (`weighted_lexical_contrast`) —
  `Σ w·feature` per group, averaged over groups, ± SE. Units are chosen-minus-rejected, so
  `w_len = +40` means "the update pushes toward ~40-character-longer completions" and `0` means
  indifferent. Exact, no embeddings, every group. `w_affirm`/`w_overpraise` use the same
  `RE_AFFIRM`/`RE_EFFUSIVE` cues as the conversation-side lexical checks in §5, deliberately: the
  two sides must be able to agree.
- **Update direction** (`direction_by_iter` / `direction_by_arm`) — `normalize(Σ w·embedding)`, the
  Mass-Mean-Probe generalized to any weighting. Word/MI-concept projections read out on it.
- **Probe audit** (`direction_quality`, `pooled_direction_quality`) — `wins_correct` is IN-sample
  and optimistic; **`wins_holdout`** (each half scored by the other half's direction) is the honest
  one; **`split_half_cos`** says whether the direction is estimated at all. Measured: a
  per-iteration PTO direction is only ~0.19 reliable and wins 0.55 held out (vs 0.68 in-sample), so
  per-iteration PTO projections are mostly noise; pooled over iterations it reaches ~0.60, and GRPO
  ~0.91 (8 candidates per group instead of 2, and more groups).
- **`cosine_corrected`** (`pooled_direction_cosines`) — cross-arm direction cosine divided by the
  attenuation ceiling `sqrt(r_a·r_b)`, each `r` a Spearman-Brown-corrected split-half. Same
  correction as the cross-judge agreement in §7, for the same reason: two noisy estimates cannot
  correlate to 1 even when identical.
- **`rho_partial_iter`** (`outcome_correlations`) — the training-signal → eval-move link
  (`arms/preference.ipynb` §4). **Always read the partial, never the raw ρ**: most features rise
  monotonically over training and most eval deltas trend too, so a raw correlation is confounded
  with iteration index by construction. n ≤ 10 iterations per arm, uncorrected — descriptive.
- **`pool_len` / `pool_question` / `pool_affirm` / `pool_overpraise`** (`pool_mean_by_iter`) — the
  unweighted mean of the same features over **all** candidates: what the policy *generates*, as
  opposed to what the update *selects for*. The pair is the point — a selection contrast of ~0.05
  next to a pool that moves 0.02 → 0.54 says the drift is a compounding on-policy loop, not one
  hard pull. ⚠ **Indexing:** `train_iter n` samples from the iter-start policy, so its pool row
  describes the policy the eval set calls `model_iter_{n-1}` — off by one from the selection
  contrast on the same row, deliberately.
- **`weighting_decomposition`** — the loss-vs-data test. `reweight` applies one method's weighting
  rule to the other's groups, so the as-trained cosine can be split into a **rule** effect (same
  groups, swapped rule) and a **data** effect (same rule, each method's own groups). ⚠ Read the
  `read` column: the attenuation ceiling assumes independent estimation errors, which is true
  across arms and false for the same-groups rows (both directions share their noise, so the
  correction over-corrects — a corrected value above 1.0 is the tell).
- **`yield_rate`** (`pair_yield_by_iter`) — `groups_trained / groups_built` per iteration. GRPO
  trains on every prompt group it builds; PTO emits a pair only where the best and worst branch
  differ by more than τ, so its yield (and its absolute pair count) can fall as branches converge.
  A shrinking training set is a candidate explanation for a flattening curve that no outcome figure
  can see.

## 6c · Compute (GPU-hours) — the cost axis every other metric is missing

**Module:** `eda_analysis/compute.py`. **Rendered by:** `compute/cost.ipynb` into `results/compute/cost/`.

Every other metric in this file is indexed by **iteration**. An iteration is not a fixed unit of
spend, so a matched-iteration contrast is not a matched-budget one.

**Definition.** `gpu_h` for one (arm, iteration) = `gen_h + build_h + train_h`, wall-clock on the
training host (it includes the API latency the GPU sits through, which is what an A100 rental
actually bills). `cum_gpu_h` at iteration *k* = the sum over iterations 1..*k* — the cost of having
produced the iter-*k* adapter, i.e. the policy the score lake calls `<Arm>_I{k}`. The base state is
0 by construction.

| phase | what it is | timed from |
|---|---|---|
| `gen_h` | the rollout pass producing `model_iter_{k-1}` | mtime span of that dir's `conversation_*.csv` |
| `build_h` | **PTO only** — preference-tree branching + oracle scoring | last conversation mtime → `iteration_k/pref_pairs/pairs.csv` |
| `train_h` | the optimizer loop | GRPO: `training/completions/*.parquet` (one per step) · PTO: TensorBoard scalar `wall_time` |

**Why not the recorded timings.** `iteration_metadata.json`'s `training_time_s`,
`generation_time_s` and `pref_pair_time_s` are **per-PROCESS**: a resumed iteration records only its
last session. GRPO_LA5 iteration 1 logs 14,501 s for work spanning 7.7 h; PTO logs
`pref_pair_time_s = 3.2 s` for a ~30 min build it reloaded from `pairs.csv`. **Never quote them.**

**Gap handling.** Any mtime delta outside `(0, 3600 s)` is a crash/resume boundary or a re-synced
Google Drive mtime (one observed interval was 1,649 h, another was negative). Each is replaced by
that phase's own median, so the interval **counts once** — dropping it would undercount a resumed
phase by exactly one step, summing it would bill days of idle time. `n_imputed` reports the count;
read it before trusting a single row.

**Derived quantities.**
- `step_multiplier` — per-step cost ratio K=5 / K=0, **per iteration and deliberately not pooled**:
  an arm's first iteration can carry a different `LOOKAHEAD_SUB_BATCH_SIZE` and a fatter API tail,
  which inflates the ratio without being intrinsic to K.
- `iso_compute_contrast` — two arms at matched `cum_gpu_h`. ⚠ This reads a **different iteration**
  from each arm, so it pairs on `persona_id`, never `file_index` (personas reshuffle `seed + k + 1`).
  `budget_ratio` = arm_b's spend / arm_a's; outside ~0.9–1.1 it is not an iso-compute comparison.
- `budget_sweep` — the same question as a *curve*: at each budget both arms are represented by the
  best checkpoint reachable within it. ⚠ **Quote the curve, not a point** — on this data the K
  lever's sign changes with budget.

**Sign convention** matches `stats.py`: `+ mean_delta ⇒ arm_a higher`, so on a `LOWER_IS_BETTER`
metric a positive delta means arm_a is *worse*.

## 7 · Judge reliability (is the measuring instrument trustworthy?)

§6 asks whether the *training* reward predicts the *eval* score. This asks whether the eval score
itself is reproducible and grader-independent. Both come from `data/eval_scores/`, written by
`Judge_Reliability.ipynb` (paid, manual) and read by `measurement/validity.ipynb` §1 through
`eda_analysis.reliability` (free, disk-only). Subset: 4 anchor models × {Q1, Q2, MICI} × 96 convs.

| Metric | Definition | Read |
|---|---|---|
| `icc_2_1` | ICC(2,1) — two-way random effects, absolute agreement, single rater (Shrout & Fleiss), computed across N re-scorings of the SAME conversations by the SAME oracle with only the API `seed` differing | Share of per-conversation variance that is signal rather than re-scoring noise. Koo & Li (2016): ≥0.75 good, ≥0.90 excellent |
| `mean_abs_diff` | Mean \|Δ\| between two reps of the same conversation | The citable "oracle noise" figure, in rubric points |
| `pearson_r` / `spearman_rho` | Second judge vs primary oracle, per conversation, within (metric, model) | Rank agreement across grader families. **Compare to `ceiling`, never to 1.0** |
| `ceiling` / `ceiling_basis` | Attenuation ceiling = `sqrt(ICC_primary × ICC_judge)`. **Both terms measured since 2026-07-28** (the second judge has 3 reps on the anchor subset); `ceiling_basis` says whether a cell used measured values or fell back to the old `ICC_judge == ICC_primary` assumption | Upper bound on achievable r — **compare `pearson_r` to this, never to 1.0**. The assumption it replaced flattered MICI: Haiku's MICI ICC is 0.53–0.93, so the true ceiling there is 0.70–0.94, not 0.93 |
| `icc_judge` | The second judge's own ICC(2,1), same construction as `icc_2_1` but across *its* reps. `NaN` where that judge has <2 full reps | Whether weak agreement is the second judge's noise or genuine disagreement — unanswerable without it |
| `bias_judge_minus_primary` | Mean level offset between judges | Expected and harmless — a harsher grader shifts every score. The thesis reports *contrasts*, which cancel it |
| `same_sign` | Whether an endpoint contrast (paired Δ over the 96 matched personas) has the same sign under both judges | **The load-bearing number.** Answers `LIMITATIONS.md` §2: is the result an artifact of the patient simulator and the grader being the same model? |

*(Definitions only — as everywhere else in this file, the measured values live elsewhere. Both
judges' ICC tables, the observed-r-vs-ceiling table and what they imply are owned by
[LIMITATIONS.md](LIMITATIONS.md) §1–§2.)*

## 7b · Multi-judge (what to do once two judges have scored the same conversations)

Read by `measurement/validity.ipynb` §2 from the same tree. **The two judges are not
interchangeable raters.** The primary oracle *was the training reward*; the second judge never
touched training. That makes every comparison below an **optimization-target vs held-out-test**
comparison, so nothing here averages raw scores across judges — level bias is 1.2–1.7 points and is
*model-dependent*, i.e. comparable in size to the headline effect.

| Metric | Definition | Read |
|---|---|---|
| `var_arm` / `var_judge` / `var_arm_x_judge` | Two-way random-effects variance components on the ARM MEANS (targets = model states, raters = judges), via expected mean squares; negative estimates clamped to 0 | `var_judge` is the level offset — large and **harmless**, it cancels in every contrast. `var_arm_x_judge` is the only component that threatens a claim: arm ordering that depends on who is grading |
| `var_arm_x_judge_adj` | The same, minus the sampling error implied by the conversation-level decomposition (`var_resid / n_convs`) | At one observation per cell, interaction and error are confounded; this subtracts the part 96-conversation sampling alone would produce |
| `dependability_k1` / `k2` | `G(k) = var_arm / (var_arm + var_arm_x_judge/k + var_resid/(n_convs·k))` | Generalizability of an arm mean read off **one** judge vs **both** averaged. The k1→k2 gain is the honest answer to "would a second judge make my ranking more trustworthy?" |
| `retention` = `delta_judge / delta_primary` | Each arm's gain over a reference (default `PTOExp3_LA0_Base`), as seen by the held-out judge, relative to the trained-against judge. Persona-bootstrap CI | **The reward-hacking test.** ~1.0 = a real behaviour change both judges see; ~0 = a gain that existed only in the optimized grader. Uniform retention across arms is scale compression; retention that *differs by arm on one metric while flat on another* is the hacking signature |
| `concordance` (by `|Δ|` bin) | P(second judge agrees on the direction) as a function of the gap the primary judge reports, over conversation PAIRS. Exact primary-judge ties excluded | "Is a gap of *this size* trustworthy?" — which a scalar r cannot answer (level bias dominates Pearson; rank statistics discard magnitude). ⚠ **Per conversation pair, not per arm.** Arm means over 96 conversations resolve ~10× better; do not read a bin height as confidence in an arm-level claim |
| `same_sign` (all pairs) | As §7, but over **every** model pair, paired on the recovered `persona_id` rather than `file_index` | `file_index` is reshuffled every iteration, so a `file_index` join across unmatched iterations pairs unrelated conversations. Means survive that; `dz` and CIs do not |
| `pct_same_sign` (`sign_preservation`) | The **rate** of `same_sign` over the all-pairs table, laddered by `\|delta_primary\|` (≥0.10 / 0.25 / 0.50) and, in the `_by_metric` variant, per rubric | The rate is meaningless without an effect size: pooled it counts contrasts too small to claim. Read the row at the gap you are claiming. Per rubric it re-detects a judge-dependent rubric from a completely different direction than `dependability_k1` — when both flag the same rubric, believe it |

⚠ **One reading rule that belongs with the definitions, not the results.** The sign-preservation
ladder's thresholds are **absolute**, so read a ladder *down its own rubric*, never across rubrics:
`PCT` is a genuine 0–1 proportion (`PCT_ChangeProp`) and never reaches |Δ|≥0.25 at all — its
ladder stops at the ≥0.10 rung. ⚠ **`MICI` is NOT on that scale and does reach the upper rungs:**
`MICI_Rate` is an unbounded acts-per-therapist-turn rate (observed 0.00–1.71 per conversation), and
in `L0` it has **48 of 231** contrasts at |Δ|≥0.25 and **15** at ≥0.50, so read its ladder normally.
The global ratings Q1/Q2/WAI-SR/MI-SAT/MITI are 1–5 and CSQ-8 is 1–4; nothing here is 1–7. Only the
pooled `all contrasts` row is cross-rubric comparable.

*(The measured values — sign-preservation rates, the variance components, gain retention — are
owned by [`measurement/SUMMARY.md`](measurement/SUMMARY.md), with the caveats they imply in
[LIMITATIONS.md](LIMITATIONS.md) §2–§3.)*

---

### Quick map: figure family → notebook
`arms/outcomes.ipynb` (trajectories, effect forest, scorecard) · `arms/heterogeneity.ipynb` (persona splits, endpoint
bars) · `arms/validity.ipynb` (behavior drift, reward_hack_panel, question/over-praise cross-checks, factor
structure) · `arms/training.ipynb` (TB curves, reward dist, reliability curve) · `arms/preference.ipynb`
(PTO preference probe; the both-methods update-weighted probe + the signal→outcome link) ·
`arms/stats.ipynb` + `method/contrast.ipynb` + `lookahead/*.ipynb` + `compute/cost.ipynb` (the heavy tables) · `measurement/validity.ipynb`
(judge ICC, second-judge agreement, multi-judge variance + gain retention).
