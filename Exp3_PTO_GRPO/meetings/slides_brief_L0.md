# Slide brief — Exp3 supervisor meeting (view L0, K=0)

**How to use this file.** This is a *content brief* for Claude Design (or any slide tool): one `##`
section per slide, each with **Takeaway** (the one line the slide must land), **Content** (the bullets),
**Visual** (a figure path to drop in, or a diagram spec to draw), and **Notes** (what to say out loud).
Build a clean, professional deck — plenty of whitespace, one idea per slide, consistent color per method
(**PTO = cool/blue, GRPO = warm/orange, Base = grey**), sans-serif, readable from the back of a room.

**Figures.** Every `Visual:` path is relative to the project root and the PNG already exists on disk —
insert the real image (do not redraw the EDA charts). All results are the **L0 view** = PTO vs GRPO at
**matched look-ahead K=0**, both trained to 10 iterations, scored on the full metric battery, paired over
the same 96 patient personas.

**Audience.** Assume the viewer has *not* seen this project. Explain every term and every metric the first
time it appears. Numbers below are the current L0 results; the source tables are named so they can be
double-checked.

---

## Slide 1 — Title

**Takeaway:** One thesis question, framed in one line.

**Content:**
- **Looking Ahead in Goal-Oriented Dialogue** — Comparing Preference-Tree (PTO) and Group-Relative (GRPO) optimization of small language models for Motivational Interviewing.
- Lior Baruch · Reichman University · MSc thesis · (building on our ICLR 2025 workshop paper)
- Subtitle: *Can a 1B-parameter model be trained to counsel well — and which training method does it better?*

**Visual:** Title slide; optional faint background of a trajectory line going up-and-right.

**Notes:** "This chapter is a controlled head-to-head between two RL-style training methods on the same task, same model, same grader."

---

## Slide 2 — What is Motivational Interviewing (MI), and why is it hard?

**Takeaway:** MI is a specific, coachable counseling *style* — which makes it a clean target for training and grading.

**Content:**
- MI = an evidence-based counseling style for behavior change (smoking, weight, etc.). The counselor **asks open questions, reflects, affirms, and elicits the patient's own "change talk"** — and **avoids** lecturing, warning, or unsolicited advice.
- It's hard because "being warm" and "being a *good* MI counselor" are not the same thing — a therapist can sound supportive while doing MI *badly* (empty praise, advice without permission).
- That gap is exactly what we measure (later slides).

**Visual:** Simple 2-column contrast: **MI-consistent** (open question, reflection, affirmation, ask permission) vs **MI-inconsistent** (confront, warn, advise unasked, over-praise).

**Notes:** Set up the tension early: the reward can be "gamed" by warmth. Foreshadows the reward-hack finding.

---

## Slide 3 — The setup: three LLMs in a loop

**Takeaway:** A small therapist model learns by being graded by a bigger "oracle" against simulated patients.

**Content:**
- **Therapist** (the model we train): **Llama-3.2-1B** — small, fast, runs locally.
- **Patient** (simulated): **gpt-4o-mini**, given one of **96 distinct personas** (age, problem, cooperativeness…). Same 96 personas every iteration, so comparisons are *paired*.
- **Oracle** (the grader): **gpt-4o-mini** with a strict JSON rubric — scores each finished conversation on validated MI questionnaires. The oracle's score *is* the reward.

**Visual (diagram spec — draw this):**
- Box **Therapist (Llama-3.2-1B)** ⇄ Box **Patient (gpt-4o-mini, persona p)** — two arrows between them labeled "conversation turns."
- The finished conversation → arrow → Box **Oracle (gpt-4o-mini + MI rubric)** → arrow "score (reward)" → back to Therapist (training signal).
- Caption: "The therapist improves by chasing the oracle's score."

**Notes:** Stress that patient + oracle are the *same* model family — flag this now; it becomes the key caveat later.

---

## Slide 4 — The research questions (and what this deck covers)

**Takeaway:** Three controlled comparisons; today is #2 at K=0.

**Content:**
- **RQ-i — Look-ahead depth K:** does anticipating a few future turns before scoring help? (K=0 vs K=5)
- **RQ-ii — PTO vs GRPO:** under *matched* settings, does the simpler iterative GRPO compete with preference-tree PTO? ← **this deck**
- **RQ-iii — Which oracle questionnaire to train on:** held for later.
- **This deck = the L0 view: RQ-ii at K=0**, the arm that is fully trained and scored.

**Visual:** Three-row list, RQ-ii highlighted; small tag "L0 = K=0" in the corner.

**Notes:** Be explicit that RQ-i (look-ahead) is *not* answered here — those arms are paused (last section).

---

## Slide 5 — How training works: an iterative self-improvement loop

**Takeaway:** Both methods share one loop — generate from the current model, score, update, repeat.

**Content:**
- Each **iteration**: the current therapist policy `π_n` generates fresh conversations → those are scored → the model is updated → `π_{n+1}` → repeat (10 iterations here).
- The conversations a model generates **double as its evaluation set** — no separate eval run.
- Everything below differs between the two methods only in **(a)** how conversations become training signal and **(b)** which optimizer/loss is used.

**Visual (diagram spec):** a simple cycle: **π_n → generate convs → oracle scores → update → π_{n+1} →** (loop back). Label "×10 iterations."

**Notes:** This shared skeleton is why the comparison is fair.

---

## Slide 6 — The shared lever: K-turn look-ahead

**Takeaway:** Look-ahead changes *what the oracle sees* before it scores — not the loss.

**Content:**
- Normally we score a candidate reply on the conversation *so far*. That rewards openings that "look good in isolation."
- **Look-ahead (K>0):** before scoring, simulate **K more turns** (patient replies, the policy replies, …) and let the oracle score that *extended* transcript — rewarding replies that "**lead somewhere good.**"
- K is the RQ-i lever. **In this deck K=0** (no look-ahead), so it's the same for both methods.

**Visual (diagram spec):** two mini-timelines. Top: `[…conversation…] + [candidate reply] → score` labeled "K=0". Bottom: `[…] + [reply] + [P][π][P][π] → score` labeled "K=5 look-ahead" with the extra turns highlighted.

**Notes:** One sentence: "Today both arms are K=0, so this lever is held fixed."

---

## Slide 7 — Method A: GRPO (online reward — oracle *inside* training)

**Takeaway:** Build a prefix dataset first (no oracle), then train with the oracle called live *inside* the GRPO loop.

**Content (diagram spec — TWO phases, then a loop):**
**Phase 1 — build the prefix dataset (no oracle yet):**
1. **π_n** (current policy) → simulate **96 conversations** vs patient.
2. From each conversation, **cut every prefix that is ≥ MCL utterances long** → a **prefix dataset (≈ 96 × ~15 ≈ 1,400 prompts).**

**Phase 2 — GRPO training (the oracle is called INSIDE the training loop):**
3. For each prompt: **sample G completions** from π_n.
4. *(dashed, only if K>0)* **K-turn look-ahead** appended to each completion.
5. **Oracle scores each completion → reward** — this happens *during* the gradient step (the oracle *is* GRPO's reward function).
6. **Group-relative advantage:** Aᵍ = (rᵍ − mean)/std over the G siblings.
7. **PPO-clipped update + KL penalty** toward π_n → **π_{n+1}** → repeat from Phase 1.

- **Key idea:** the reward is computed **online** — the oracle is queried on freshly sampled replies at *every* training step.

**Visual:** box-and-arrow diagram with a clear **Phase 1 (grey, "no oracle") → Phase 2 (highlighted, "oracle inside training")** split; dashed look-ahead box greyed "(K=0: skipped)".

**Notes:** "The oracle lives *inside* GRPO training — every step samples new replies and scores them live. That's the opposite of PTO, coming up next."

---

## Slide 8 — Method B: PTO (offline reward — oracle *before* training)

**Takeaway:** Score everything up front to build the *entire* preference dataset, then train DPO on that static set — the oracle is used *before* training, never during it.

**Content (diagram spec — TWO phases, then a loop):**
**Phase 1 — build the FULL preference dataset (the oracle is used HERE):**
1. **π_n** → simulate **96 conversations** (also the eval set).
2. For each conversation, **grow a trunk**; at each therapist turn **branch — sample M candidate replies** (≈ **96 × ~15 ≈ 1,400 branch points**).
3. *(dashed, only if K>0)* **K-turn look-ahead** on each candidate.
4. **Oracle scores each → keep best & worst;** if (best − worst) > τ, emit a **(chosen, rejected) pair**, and append the best reply to advance the trunk.
5. → a **static preference dataset** of up to ~1,400 (chosen, rejected) pairs.

**Phase 2 — DPO training (NO oracle here):**
6. **DPO** trains on the finished preference dataset: push toward *chosen*, away from *rejected* (anchored to π_n) → **π_{n+1}** → repeat from Phase 1.

- **Key idea:** the reward is computed **offline** — the oracle builds a fixed preference dataset *before* DPO; DPO itself never calls the oracle.

**Visual:** box-and-arrow diagram — the **mirror image** of GRPO's: **Phase 1 (highlighted, "oracle here") → Phase 2 (grey, "no oracle")**; dashed look-ahead greyed "(K=0: skipped)".

**Notes:** "PTO is the framework; DPO is the loss. The oracle *builds the data*, then DPO trains on it — the opposite placement from GRPO, which calls the oracle inside training."

---

## Slide 9 — PTO vs GRPO at a glance

**Takeaway:** Same loop, one decisive difference — *when* the oracle is called: online (GRPO) vs offline (PTO).

**Content (small table):**
| | **GRPO** | **PTO** |
|---|---|---|
| **When is the oracle called?** | **online — inside training** (it *is* the reward fn) | **offline — before training** (it *builds* the pref set) |
| Data the trainer sees | prefix dataset (~1,400 prompts) | preference dataset (~1,400 chosen/rejected pairs) |
| Candidates per step | G, **all kept** | M, **best + worst kept** |
| Signal | reward → group-relative advantage | reward → *pick* a (chosen, rejected) pair |
| Loss | PPO-clip + KL | DPO sigmoid (implicit KL) |
| Yields no training row? | never | yes, if best≈worst (within τ) |

- Everything else (base model, patients, oracle, K=0, 10 iterations, MCL) is **matched.**

**Visual:** the table, method colors on the headers; bold the top "oracle called" row.

**Notes:** "The headline difference is oracle placement — GRPO scores live during training, PTO scores up front to build a fixed dataset. Everything else is matched, so differences are the method, not the setup."

---

## Slide 10 — The metrics, part 1: warmth & alliance rubrics

**Takeaway:** Five validated questionnaires, all oracle-graded 1–5; one is the training reward.

**Content (each = what it is / where it comes from):**
- **Q1 + Q2** — two core MI-quality questions (a 22-item MI rubric); **mean(Q1,Q2) is the training reward.** Oracle-graded.
- **WAI-SR** — Working Alliance Inventory (short) → therapeutic *alliance/bond*. Oracle-graded.
- **CSQ-8** — Client Satisfaction Questionnaire → *patient satisfaction*. Oracle-graded.
- **MI-SAT** — MI-specific satisfaction. Oracle-graded.
- **MITI** — MI Treatment Integrity → MI *fidelity* (global rating + behavior counts). Oracle-graded.
- All higher = better.

**Visual:** a labeled legend/table of the five; note "training reward = Q1+Q2" with a star.

**Notes:** "These five are all 'is the patient feeling well-counseled?' — they turn out to move together (slide 16)."

---

## Slide 11 — The metrics, part 2: orthogonal axes (added to catch reward-hacking)

**Takeaway:** We added axes that *don't* just reward warmth — to detect gaming.

**Content:**
- **PCT — Patient Change-Talk:** does the *patient* move toward change (vs sustain-talk)? Oracle-graded. Higher = better.
- **MICI — MI-*Inconsistency*:** counts MI-*violating* therapist moves (over-praise, advise-without-permission, confront, warn). Oracle-graded. **Lower = better** (shown with a ↓).
- **Derived MITI ratios (free, from the MITI coding):** **R:Q** (reflections per question), **%CR** (% complex reflections), **%MICO** (% MI-consistent behaviors).
- **Behavior rates (per therapist turn, from MITI coding):** questions, simple/complex reflections, affirmations, persuasion — all tagged **(MITI)** on the charts.
- **Deterministic text checks (no LLM):** **Degeneration %** (repetition loops), **Questions/turn (regex ?)** — reward-independent sanity checks.

**Visual:** two groups — "oracle-graded" vs "deterministic text" — with the ↓ marker on MICI.

**Notes:** "MICI and the text metrics are our honesty checks — they can't be gamed by sounding warm."

---

## Slide 12 — Result 1: it works — both methods improve a lot

**Takeaway:** Every metric climbs far above the untrained baseline.

**Content:**
- All warmth/alliance rubrics rise steeply from base; effects are **large** (Cohen's dz up to ~1.4), Holm-corrected p ≈ 0.
- Both PTO and GRPO clear the oracle's noise band (~0.10) — the gains are structural, not measurement wobble.

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/1_outcomes/trajectories_all_metrics.png`
— *(MUST SHOW)* one small panel per metric, mean vs iteration (x = 0–10), one line per arm (PTO blue, GRPO orange, base grey). **How to read:** up-and-to-the-right = improving. Pair with `.../1_outcomes/effect_vs_base_forest.png` (dot = effect size vs base; right of 0 = better).

**Notes:** "First-order answer: yes, a 1B model learns to counsel much better. Now — which method, and is it *real* skill?"

---

## Slide 13 — Result 2 (the core): PTO wins at the matched endpoint

**Takeaway:** At the fair 10-iteration endpoint, PTO beats GRPO — because GRPO peaks then regresses.

**Content:**
- **Q1+Q2 at iter 10: PTO 4.26 vs GRPO 3.75** (paired PTO−GRPO **+0.51**, dz +0.73, Holm p<0.001).
- **GRPO peaks at iter 8 (4.08) then regresses** (→3.81→3.75); **PTO keeps climbing** (4.22→4.26).
- Even GRPO's *best* (4.08) is below PTO's best (4.26). With GRPO you'd need early-stopping.

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/1_outcomes/trajectories/trajectory_Q1Q2.png`
— two lines with peak markers. **How to read:** watch GRPO turn over after iter 8 while PTO stays flat-to-up. Source table: `.../results/L0/tables/6_stats/method_paired_by_K.md`.

**Notes:** "The earlier 'it's a tie at iter 8' was a snapshot artifact — at the matched endpoint PTO is ahead, and the *why* is instability."

---

## Slide 14 — Result 3: stability & climb rate

**Takeaway:** PTO climbs faster and doesn't turn over.

**Content:**
- OLS slope: **PTO 0.120/iter** (peak = final iter 10) vs **GRPO 0.072/iter** (peak iter 8).
- GRPO shows a simultaneous **iter-9 dip across most metrics**, partly recovering at 10 (quantified in `grpo_iter9_check.md`).
- Core answer (RQ-ii): **GRPO is competitive up to its peak, then overshoots and degrades; PTO sustains gains.**

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/1_outcomes/outcomes_by_model.png` (per-model bars) or reuse the Q1+Q2 trajectory. Tables: `.../6_stats/slope_by_arm.md`, `.../6_stats/grpo_iter9_check.md`.

**Notes:** Keep it short — this is the "why PTO wins" mechanism at the outcome level; the *behavioral* mechanism is next.

---

## Slide 15 — Result 4: the catch — the gains are partly a reward-hack

**Takeaway:** As scores rise, the therapist over-praises and stops asking questions — GRPO worse.

**Content:**
- **Affirmation drift in BOTH arms; GRPO is the worse offender at iter 10:** GRPO affirmations 0.52→**1.98**/turn, questions collapse to **0.15/turn** (PTO holds **0.55**). GRPO emits ~**3.5× more praise** than PTO.
- **MI-inconsistency (MICI) rises ~2.3×** with warmth (base 0.21 → **0.49 PTO / 0.84 GRPO**; GRPO effect dz 1.72, large).
- The iter-10 GRPO regression **is** this over-praise, which the full-conversation oracle penalizes.

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/3_mechanism/behavior_drift.png`
— *(MUST SHOW)* per-turn MI behaviors vs iteration (panels now consistently labeled "(MITI)"). Pair with `.../3_mechanism/reward_hack_panel.png` (warmth↑ alongside MI-inconsistency↑) and `.../1_outcomes/trajectories/trajectory_MICI.png` (MICI↓ is better — watch it rise = worse). **How to read behavior_drift:** questions falling + affirmations rising = drifting into empty praise.

**Notes:** "This slide is reward-independent — it's counting behaviors, not asking the oracle — so the reward-hack read isn't circular. This is a *finding*, not a bug."

---

## Slide 16 — Result 5: is it multi-skill, or one warmth factor?

**Takeaway:** The five warmth rubrics are basically one factor; the orthogonal axes reveal a genuine second.

**Content:**
- The five warmth rubrics co-move — one principal component (**PC1**) explained ≈**91%** of variance.
- **Adding the orthogonal axes drops PC1 to ≈55%:** warmth is one factor; technique (R:Q/%CR/%MICO) + MI-inconsistency form a second. So "all rubrics went up" is **not** proof of multi-skill mastery.
- Caveat to state: **PCT (change-talk) empirically co-loads with warmth** (ρ≈0.79–0.94), so it's less orthogonal than intended.

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/3_mechanism/factor_loadings.png` (each metric's loading on PC1/PC2) and `.../3_mechanism/rubric_correlation.png` (correlation heatmap; a warmth block + a separate technique/MICI block). **How to read:** bars high on PC1 = "warmth cluster"; near-0 on PC1 = the independent second axis.

**Notes:** "This is why we added the extra axes — without them we'd have overclaimed general skill."

---

## Slide 17 — Result 6: who does the regression hurt? (heterogeneity)

**Takeaway:** GRPO's late collapse concentrates on the hard (resistant) patients.

**Content:**
- Splitting outcomes by persona trait (cooperativeness, problem type): improvements are broad, but **GRPO's endpoint drop concentrates on the *Resistant* personas** — exactly the patients where empty praise fails and concrete help is demanded.
- PTO holds up better on the same hard subgroup.

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/2_heterogeneity/subgroup_endpoint_cooperation_level.png` (final-iteration bars by cooperativeness) and `.../2_heterogeneity/cooperation_level_all_metrics.png` (trajectories split by trait).

**Notes:** "The averages hide it; the resistant subgroup is where the methods actually separate."

---

## Slide 18 — Result 7: is the training reward even trustworthy?

**Takeaway:** We score partial conversations but evaluate full ones — we checked the proxy holds up.

**Content:**
- The reward grades *partial* conversations (cheaper); the thesis judges *full* conversations. Do they agree?
- Rank-agreement rises with conversation length; our **MCL=12** floor keeps training out of the unreliable short-cut zone (in an earlier experiment agreement fell to ~0.66 at 2 turns — near chance).
- Interesting split: GRPO's proxy grows *more* faithful with length (≈0.86→0.94), PTO's grows *less* (≈0.87→0.76).

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/4_training/reward_reliability_curve.png` (agreement vs conversation length). Optional: `.../4_training/advantage_signal_sidebyside.png` (how decisive each method's reward gaps are).

**Notes:** "This justifies the MCL knob and pre-empts 'but your reward is a short-cut.'"

---

## Slide 19 — Result 8 (optional depth): PTO's preference signal is real

**Takeaway:** The chosen-vs-rejected direction genuinely separates good replies — and drifts toward affirmation.

**Content:**
- A probe on the (chosen − rejected) direction: `wins_correct` **0.65 → 0.71** over iterations (>0.5 = the learned direction separates held-out pairs), strengthening late.
- What it learns to prefer drifts toward affirmation/achievement language — the latent-space echo of the behavior drift on slide 15.
- (GRPO has no preference pairs, so this analysis is PTO-only by construction.)

**Visual:** `Exp3_PTO_GRPO/eda/results/L0/figures/5_preference/PTO_LA0_pref_word_ranking.png` and `.../5_preference/PTO_LA0_pref_direction_drift.png`.

**Notes:** Include only if there's time; it's supporting evidence, not headline.

---

## Slide 20 — Where we stand

**Takeaway:** RQ-ii (K=0) is answered; RQ-i (look-ahead) is paused on cost.

**Content:**
- **Done & scored:** PTO and GRPO at **K=0**, base → iter 10, full battery, 96 paired personas (this deck).
- **Paused:** both **K=5 (look-ahead)** arms — PTO K=5 got 4 iterations, GRPO K=5 base only.
- **Why paused:** OpenAI API spend hit **~$300** and is the binding constraint. Cost is dominated by oracle scoring + (at K=5) look-ahead patient calls, ∝ candidate-count × iterations. So **RQ-i is still open.**

**Visual:** a small status matrix: rows = {PTO, GRPO}, cols = {K=0, K=5}; K=0 cells ✅ "iter 10 scored", K=5 cells ⏸ "paused". Add a "~$300 spent" cost flag.

**Notes:** Be honest that look-ahead is unanswered; frame cost as a design constraint, not a failure.

---

## Slide 21 — How to continue (options + trade-offs)

**Takeaway:** Several viable paths; each trades cost against which question it answers.

**Content (option / what it buys / cost):**
- **A. Resume the K=5 arms** → directly answers RQ-i (does look-ahead help). *Cost: highest* (look-ahead patient calls). Mitigate: fewer iterations (~5–6; gains plateau by ~iter 4) and M/G 8→4.
- **B. Add a cheaper middle point: K=2 (an L2 view)** → a K-gradient {0,2} at a fraction of K=5's cost — a partial RQ-i answer.
- **C. Switch the therapist to an *instruct* model (Llama-3.2-1B-Instruct)** → the base model self-plays chat tokens (a known confound we patched); an instruct model handles turns natively → cleaner, likely stronger baseline. *Note: changes absolute scores, but the PTO-vs-GRPO comparison still holds within the new base.*
- **D. Oracle sweep (RQ-iii) / more questionnaires** → train on WAI-SR / CSQ-8 / MI-SAT / MITI instead of Q1+Q2; does the choice of grader change the winner? (All six are already scored for *eval*.)
- **E. Deeper EDA / stats** → per-persona paired trajectories, formal tests on the MICI/affirmation drift, ablate the MCL knob.

**Visual:** a 2-column "option → what it answers" list; color-code by cost (green/amber/red).

**Notes:** Ask the supervisor to pick a lane — the constraint is budget, so we can't do all.

---

## Slide 22 — Additional suggestions (my recommendations)

**Takeaway:** A few high-leverage moves beyond the obvious next runs.

**Content:**
- **Break the shared-oracle confound:** patient + oracle are the same model family, and reward = outcome. Add a **second, different judge model** (or a small human rating) on a sample to show the gains aren't just gaming gpt-4o-mini. *(Highest-value credibility move.)*
- **Report GRPO fairly with peak-selection / early-stopping**, not just the final iteration — its instability is the finding, but the comparison should show both.
- **Test-time generalization:** evaluate on **held-out personas** (or a harder patient set) to check the gains transfer beyond the 96 training personas.
- **Elevate the reward-hack to a contribution:** "both methods reward-hack (over-praise); PTO less, and the orthogonal axes catch it" is a genuine result, not a caveat to bury.
- **Cost-first experiment design:** cap iterations ~5–6, drop M/G 8→4, keep **K** and the **gpt-4o-mini oracle** fixed — this makes RQ-i affordable.

**Visual:** 5 punchy bullets; star the first.

**Notes:** These directly answer "any more suggestions?" — lead with the second-judge idea; it's what a committee will ask about.

---

## Slide 23 — Summary / takeaways

**Takeaway:** PTO wins at the fair endpoint; both reward-hack; cost gates the next question.

**Content:**
- **A 1B model learns MI well** — large gains over base for both methods.
- **PTO > GRPO at the matched 10-iteration endpoint (4.26 vs 3.75)** — GRPO peaks then regresses; PTO is stable.
- **The gains are partly a reward-hack** (over-praise, fewer questions; GRPO worse) — visible in reward-independent behavior counts and the MICI axis.
- **"All rubrics up" ≠ multi-skill** — the orthogonal axes drop shared variance from ≈91% to ≈55%.
- **Look-ahead (RQ-i) is unanswered — paused on ~$300 cost.** Clear, affordable paths to finish it.

**Visual:** the five takeaways as a clean list; method colors on the PTO/GRPO line.

**Notes:** Close by naming the one decision you want from the meeting (which continuation lane to fund).

---

### Appendix — figure & table index (for building the deck)

All paths under the project root; view = `L0`.

**Figures** (`Exp3_PTO_GRPO/eda/results/L0/figures/`):
- `1_outcomes/trajectories_all_metrics.png` · `1_outcomes/effect_vs_base_forest.png` · `1_outcomes/outcomes_by_model.png`
- `1_outcomes/trajectories/trajectory_Q1Q2.png` (+ per-metric siblings: `trajectory_MICI.png`, `trajectory_PCT.png`, `trajectory_MITI.png`, …)
- `2_heterogeneity/subgroup_endpoint_cooperation_level.png` · `2_heterogeneity/cooperation_level_all_metrics.png` · `2_heterogeneity/problem_all_metrics.png`
- `3_mechanism/behavior_drift.png` · `3_mechanism/reward_hack_panel.png` · `3_mechanism/factor_loadings.png` · `3_mechanism/rubric_correlation.png`
- `4_training/reward_reliability_curve.png` · `4_training/advantage_signal_sidebyside.png`
- `5_preference/PTO_LA0_pref_word_ranking.png` · `5_preference/PTO_LA0_pref_direction_drift.png`

**Tables** (`Exp3_PTO_GRPO/eda/results/L0/tables/`): `6_stats/main_results.md` · `6_stats/method_paired_by_K.md` · `6_stats/slope_by_arm.md` · `6_stats/grpo_iter9_check.md` · `1_outcomes/leaderboard_scorecard.md` · `3_mechanism/behavior_by_iter.md`

**Narrative source of truth:** `Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md` · **metric definitions:** `Exp3_PTO_GRPO/eda/METRICS_REFERENCE.md` · **limitations:** `Exp3_PTO_GRPO/eda/LIMITATIONS.md`
