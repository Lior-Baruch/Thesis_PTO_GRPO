# Exp3 EDA Summary — `arms/` (per-arm descriptives, all four arms on one axis)

*Ported from `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` on 2026-08-18 (reorg by research
question); numbers unchanged, paths rewritten.*

> ⚠ **This file is INTERPRETATION, not evidence.** It is a hand-authored reading of the tables under
> [`*/tables/`](INDEX.md) — written in past sessions, largely by Claude. The tables are the evidence;
> this is a claim *about* them. Two consequences:
>
> - **Do not brainstorm framings from this file.** Read the tables cold first, write your own
>   candidates down, and only then read this and diff. Otherwise the section headers below become
>   the option space. See [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs".
> - **Do not quote a number from here into a paper.** Open the table it cites, and check it cites
>   the right one — §4 has quoted the regex question rate while pointing at the oracle-coded table.

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`<sub>/{figures,tables}/<judge>/…`); numbers are full-conversation eval (the held-out
outcome), persona-paired over the 96 shared personas, on the primary oracle unless a held-out
column is named. The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

## What this top covers

`arms/` holds the **per-arm descriptives**: what each of the four arms (`PTO_LA0`, `PTO_LA5`,
`GRPO_LA0`, `GRPO_LA5`) scores, how it moves across iterations, what its therapist actually does,
whether its training signal was usable and faithful — every artifact rendered **once per grader**
into a `<judge>/` leaf. The retired `L0` / `L5` summaries read the two K subsets separately (K=0
arms in `L0`, K=5 arms in `L5`); **`arms/*` now shows all four arms on one axis**, so a figure that
used to hold two lines holds four. The *contrasts* live elsewhere: PTO vs GRPO in
[`../method/SUMMARY.md`](../method/SUMMARY.md), K=0 vs K=5 in
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md), spend in
[`../compute/SUMMARY.md`](../compute/SUMMARY.md), whether the ruler is trustworthy in
[`../measurement/SUMMARY.md`](../measurement/SUMMARY.md). The prose below was written arm-pair by
arm-pair (K=0 first, from `L0`), so it still names the arms it was reading — treat "the K=0 arms"
as a scope note, not a filter.

| Arm | K | Iters scored | GPU-h | Notes |
|---|---|---|---|---|
| `PTO_LA0`  | 0 | 0–10 | **8.1** | pref-tree → DPO, MCL=12, oracle = Q1+Q2 |
| `PTO_LA5`  | 5 | 0–10 | **19.7** | |
| `GRPO_LA0` | 0 | 0–10 | **27.9** | group-relative, MCL=12, oracle = Q1+Q2 |
| `GRPO_LA5` | 5 | 0–5  | **27.1** | stopped ~2 min into iteration 6 (one step, no adapter) |

Eleven matched points for PTO (0–10) and six for GRPO (0–5), on two independent graders. GPU-h from
[`../compute/cost/tables/compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md).

- **Metrics:** the 5 **global-evaluation rubrics** (Q1Q2, WAI-SR, CSQ-8, MI-SAT, MITI) — an
  *empirical* halo cluster (they co-load on one PC1 factor), not one official construct; per-
  instrument provenance incl. the CLPsych-2024 Q1/Q2 source is in
  [`METRICS_REFERENCE.md`](../METRICS_REFERENCE.md) §1 — **plus** three further evaluation
  metrics added to test whether anything measures outside that halo: `PCT` (patient change-talk),
  `MICI` (MI-inconsistent therapist behaviour, **lower = better**), and the free derived
  MITI-proficiency ratios `R:Q` / `%CR` / `%MICO`. They are reported flat alongside the rubrics —
  **not** as an "orthogonal" family: `PCT` turned out to co-move with the halo (§3).

---

## 2. Both arms improve a lot — PTO is stronger *and* more stable (K=0 arms)
See [`outcomes/figures/gpt-4o-mini/outcomes_by_model_final.png`](outcomes/figures/gpt-4o-mini/outcomes_by_model_final.png),
[`outcomes/figures/gpt-4o-mini/effect_vs_base_forest_final.png`](outcomes/figures/gpt-4o-mini/effect_vs_base_forest_final.png),
[`outcomes/figures/gpt-4o-mini/trajectories/trajectory_Q1Q2.png`](outcomes/figures/gpt-4o-mini/trajectories/trajectory_Q1Q2.png), and
[`stats/tables/gpt-4o-mini/main_results.md`](stats/tables/gpt-4o-mini/main_results.md).

- **Each arm vs base — large global-evaluation gains.** PTO_LA0 Q1+Q2 **3.00 → 4.26** (dz 1.43, *large*,
  Holm p≈0, Friedman W=0.45). GRPO_LA0 Q1+Q2 **3.07 → 4.08 at its iter-8 peak**, falling to **3.75
  by iter 10** (final dz 0.72 *medium*, best dz 1.22). Every global-evaluation rubric is a *large* effect for
  PTO; Holm p≈0 everywhere.
- **PTO ahead at the matched 10-iter endpoint** — the paired PTO−GRPO contrast (**Q1+Q2 +0.51**,
  dz +0.73, Holm p<0.001) is the method question and lives in
  [`../method/SUMMARY.md`](../method/SUMMARY.md) §2 /
  [`../method/contrast/tables/method_paired_by_K.md`](../method/contrast/tables/method_paired_by_K.md).
  What the per-arm curves add: the earlier "near-tie at iter 8" was a snapshot — **GRPO peaks at
  iter 8 then regresses** (4.08 → 3.81 → 3.75) while PTO keeps climbing (4.22 → 4.26).
- **Climb rate.** OLS Q1+Q2 slope PTO **0.120/iter** (peak = final iter 10) vs GRPO **0.072/iter**
  (peak iter 8) — [`stats/tables/gpt-4o-mini/slope_by_arm.md`](stats/tables/gpt-4o-mini/slope_by_arm.md). With
  GRPO, peak-iter selection / early stopping matters; even so its best (4.08) is below PTO's (4.26).
- **Per-metric learning curves** (every metric, peaks auto-flagged) live in
  [`outcomes/figures/gpt-4o-mini/trajectories/`](outcomes/figures/gpt-4o-mini/trajectories/); the persona splits
  (every metric × cooperation/problem) in [`heterogeneity/figures/gpt-4o-mini/`](heterogeneity/figures/gpt-4o-mini/) —
  GRPO's endpoint collapse concentrates on the *Resistant* personas.
- **Iter-9 caveat:** GRPO_LA0 dips at iter 9 across most metrics simultaneously then partially
  recovers at 10 — [`validity/tables/gpt-4o-mini/grpo_iter9_check.md`](validity/tables/gpt-4o-mini/grpo_iter9_check.md)
  quantifies it (a paired one-iteration dip on top of the monotonic Q1+Q2 decline).

The K=5 arms' levels per iteration are in
[`../lookahead/reward/tables/k_means_by_iter.md`](../lookahead/reward/tables/k_means_by_iter.md)
(and now on the same axis in every `outcomes/` figure); the K contrast itself is
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) §3.

## 3. The gains come *with* a measurable reward-hack — that's why the added metrics matter
See [`validity/figures/gpt-4o-mini/factor_loadings.png`](validity/figures/gpt-4o-mini/factor_loadings.png),
[`validity/figures/gpt-4o-mini/rubric_correlation.png`](validity/figures/gpt-4o-mini/rubric_correlation.png), and
[`outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md`](outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md).

- **MI-inconsistent behaviour rises ~2.3× (PTO) / ~4× (GRPO)** as the global-evaluation scores climb
  (MICI base 0.21 → 0.49 PTO / 0.84 GRPO at iter 10; GRPO's MICI effect is dz 1.72, *large*). The
  gains are partly over-praise/advice in **both** methods, **worse in GRPO**.
- **Adding the further metrics drops PC1 from ≈91% → ≈55%** (per-arm PC1 ≈55–56%). Global
  evaluation (the halo) is one factor; technique (R:Q/%CR/%MICO) + MI-inconsistency form a second —
  so "all rubrics up" is *not* multi-skill. **The second factor is NOT PCT:** change-talk co-moves
  with the halo rubrics (Spearman ρ≈0.79–0.94 with them; PC1 loading ~0.39), so it does not isolate
  MI technique. Report all eight metrics flat; don't describe them as orthogonal families.
- **Patient change-talk (PCT) rises modestly**, more for PTO (0.49 → 0.63, *medium*) than GRPO
  (0.49 → 0.57, *small*).

## 4. Mechanism — what the therapist actually does
See [`questionnaires/figures/gpt-4o-mini/miti_detail_grid.png`](questionnaires/figures/gpt-4o-mini/miti_detail_grid.png) and the merged
behaviour table [`questionnaires/tables/gpt-4o-mini/miti_detail_by_iter.md`](questionnaires/tables/gpt-4o-mini/miti_detail_by_iter.md).

- **Affirmation drift is confirmed in BOTH arms, and at iter 10 GRPO is the worse offender:**
  GRPO B6_AF 0.52 → **1.98**, questions B3_Q 6.4 → **4.1**, q/turn 0.83 → **0.15**, R:Q → **1.44**.
  PTO's drift is milder and plateaus (iter-10 B6_AF 1.64, q/turn 0.55).
- **Across all 96 iter-10 conversations:** GRPO collapses to **0.15 questions/turn** vs PTO's
  **0.55** — ⚠ that is the **regex `"?"` count** from
  [`validity/tables/gpt-4o-mini/session_shape_by_iter.md`](validity/tables/gpt-4o-mini/session_shape_by_iter.md),
  **not** the oracle-coded rate. The oracle-coded rate separates the arms far less:
  `B3_Q_per_turn` **0.32 (GRPO) vs 0.41 (PTO)** in
  [`questionnaires/tables/gpt-4o-mini/miti_detail_by_iter.md`](questionnaires/tables/gpt-4o-mini/miti_detail_by_iter.md).
  Quote whichever you mean and say which. The oracle codes GRPO as far more
  MI-inconsistent (**MICI 0.84 vs 0.49**). A lexical praise-word count (the demoted sanity-check)
  puts GRPO at **~3.5× PTO's praise rate**. The iter-10 eval regression *is* this over-praise
  reward-hack, which the full-conversation oracle penalises; GRPO falls into it harder.
- **Absolute anchor — official MITI 4.2.1 competency thresholds**
  ([`questionnaires/figures/gpt-4o-mini/miti_proficiency_thresholds.png`](questionnaires/figures/gpt-4o-mini/miti_proficiency_thresholds.png),
  [`questionnaires/tables/gpt-4o-mini/miti_threshold_verdicts.md`](questionnaires/tables/gpt-4o-mini/miti_threshold_verdicts.md)):
  training takes both arms from *below basic competence* to **fair-to-good on the global ratings**
  (Relational crosses "good": PTO 4.61, GRPO 4.20) — but **neither arm reaches "good" on the
  technique ratios** (%CR PTO 0.36✗ / GRPO 0.41 fair; R:Q PTO 0.75✗), and GRPO's iter-10 R:Q
  1.43 "fair" is the *pathological* route (the question collapse shrinks the denominator).
  Thresholds are the manual's expert opinion and defined for 20-min human sessions — an anchor,
  not a certification.
- **Reward composition — which Q2 items the optimizer exploits**
  ([`questionnaires/figures/gpt-4o-mini/q2_item_deltas_final.png`](questionnaires/figures/gpt-4o-mini/q2_item_deltas_final.png),
  [`questionnaires/tables/gpt-4o-mini/q2_item_deltas.md`](questionnaires/tables/gpt-4o-mini/q2_item_deltas.md)): the top
  endpoint-Δ Q2 item differs by arm: in **GRPO** it is *"revealed what he was thinking"*
  (self-disclosure, 1.07), but in **PTO** it is *"put himself in my shoes"* (1.54), with
  *"made me feel cared for"* (1.53) ahead of self-disclosure and *"took charge"* (1.48 each) —
  so self-disclosure tops only the GRPO arm — the Q1+Q2 reward's own composition
  (items 1/2/3/10 reward therapist self-disclosure, which MI does not prescribe) incentivizes the
  emotive drift, i.e. the hack traces to specific reward components, not only to the optimizer.
- Both arms kill the early degeneration loops (loop% 0.49 → 0); the leak/empty health gate stays
  clean (see [`training/figures/gpt-4o-mini/`](training/figures/gpt-4o-mini/)).

## 5. Is the training reward faithful?
See [`training/figures/gpt-4o-mini/reward_reliability_curve.png`](training/figures/gpt-4o-mini/reward_reliability_curve.png).
At MCL=12, GRPO's short proxy reward stays roughly flat and edges *up* with conversation length
(rank agreement ≈0.86 → 0.90) while PTO's grows *less* faithful (≈0.86 → 0.76). ⚠ The ≈0.94 this
line used to quote is `GRPO_LA5`'s (0.935), an arm the old `L0` view excluded by construction — the
figure now draws all four arms; check its `_provenance.md` for which arms it actually drew. The
MCL=12 floor keeps both out of the unreliable short-cut regime (Exp2 saw agreement as low as 0.66 at
n_turns=2). The K=0 vs K=5 faithfulness contrast *at a matched policy* is
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) § "Cross-K findings" /
[`../lookahead/mechanism/`](../lookahead/mechanism/).

## 6. What the training signal pushes toward — now measured for BOTH methods
See [`preference/figures/gpt-4o-mini/`](preference/figures/gpt-4o-mini/) and
[`preference/tables/gpt-4o-mini/`](preference/tables/gpt-4o-mini/). Since 2026-08-02 this is not
a PTO-only section: both methods weight the candidates of a group and step along the weighted sum
(DPO ±1 on chosen/rejected, GRPO the standardized advantage), so rescaling to a common per-group
size puts them on one probe.

- **The affirmation push is real, and it grows in both methods** (`update_lexical_push`, exact, every
  group): GRPO −0.006 → **+0.086 ± 0.008**, PTO 0.008 → **0.103 ± 0.029** (iter 8). §3–§4's
  reward-hack now has a *training-side* measurement, not only an outcome-side inference. GRPO's
  series dips negative at **iter 9** — the same iteration the outcome grid dips across nearly every
  metric, from a completely independent source.
- **The two losses do not want the same thing.** Pooled update-direction cosine PTO vs GRPO is 0.267
  raw; against the attenuation ceiling of 0.844 (how well each direction is estimated) that is
  **0.317 corrected** — under a third of the achievable agreement, at matched K and a shared oracle.
- **The push predicts the MICI move in GRPO, not in PTO** (`pref_outcome_correlations`, with
  `train_iter` partialled out of both sides — the raw ρ is confounded with iteration by
  construction): GRPO's ΔMICI tracks its affirmation push **ρ 0.647 (p .043)**, its length push
  0.706 (p .023) and its over-praise push 0.617 (p .057); PTO's does not (−0.492, ns). Same
  direction as the endpoint MICI gap (0.84 vs 0.49), reached from the training data. n ≤ 10
  iterations per arm and uncorrected — a mechanism consistent with the curves, not a cause.

**And three things the aggregate curves could not say:**

- **The PTO-vs-GRPO gap is about the DATA, not the loss.** Swapping the weighting rule on the *same*
  groups barely moves the direction (0.908 on PTO's groups, 0.988 on GRPO's); holding the rule
  fixed across the two methods' *own* groups leaves them as far apart as ever (0.397 / 0.324
  corrected, vs 0.317 as trained). At matched K and a shared oracle the two losses extract nearly
  the same direction from the same eight completions — so "PTO vs GRPO" is a statement about
  **exploration**, not about DPO vs group-relative weighting. *(⚠ Framing note, 2026-08-18: STATUS.md
  now words this as the **state distribution** the two methods train on, not "exploration" —
  candidate sampling is matched by construction, temp 1.2, M=G=8. The numbers are unchanged.)*
- **The reward-hack is a compounding loop, not a hard pull.** Per-iteration *selection* pressure on
  affirmation is ≈0.01 → 0.10, while what the policy *generates* goes **0.02 → 0.54** (GRPO) and
  **0.04 → 0.57** (PTO); over-praise reaches **0.74** of GRPO's candidates and questions collapse
  from 0.71 to **0.06** per completion. Small, persistent, same-signed pressure, applied each
  iteration to an already-more-effusive policy. By the end the update is choosing between two
  effusive completions — which is why the selection contrast understates the drift.
- **PTO's training signal shrinks by two-thirds.** Branch points built fall 949 → 410 and the τ
  yield falls 0.82 → 0.69, so groups that actually trained fall **782 → 281**, with the best−worst
  margin decaying 0.274 → 0.196. GRPO trains on 94–98% of its groups throughout. A flattening PTO
  curve may partly be a data-starvation curve.

> ⚠️ **Correction to what this section used to claim.** It previously reported `wins_correct`
> 0.65 → 0.71 as evidence that "the DPO signal is real and its latent target drifts toward
> affirmation". That number is **in-sample** — the direction was scored on the very pairs it was
> fitted on. Held out, the same per-iteration PTO direction wins only **0.47–0.59**, and its
> split-half reliability is **0.15–0.32**: two halves of one iteration's pairs point almost
> independently. The per-iteration latent-drift artifacts are therefore mostly estimation noise.
> What survives is the *pooled* direction (split-half 0.597 PTO / 0.911 GRPO) and the exact lexical
> contrasts above, which need no embedding at all.

## 7. Does the result survive a different judge?
Yes — and the reward-hack gets *sharper* under the held-out grader. The judge-validity evidence
(variance decomposition, sign preservation over all 1,848 pairwise contrasts, gain retention, the
MITI exception) is [`../measurement/SUMMARY.md`](../measurement/SUMMARY.md); the per-arm figures
here render once per grader (`<sub>/{figures,tables}/claude-haiku-4-5/`), so any arm-level reading
above can be re-checked on the held-out leaf directly. **Never average the two graders**, only
compare contrasts.

## 8. Caveats
- Oracle reproducibility is **measured**, not assumed: ICC(2,1) **0.86–0.99**, mean |Δ| 0.04–0.09
  across four draws (Q1/Q2 0.96–0.99; only MICI falls below 0.90, floor 0.864 at PTO@10) — the project's informal "≈0.10 noise" figure is a conservative upper bound, and it
  shrinks by ~√96 at the arm-mean level this summary reports.
- Absolute scores are **Exp3-internal only** — not comparable to Exp2 (4-bit vs bf16 generation),
  and **never comparable across judges** (Haiku's level offset is 1.2–1.7 points).
- MITI arm differences are **provisional** — see the MITI warning in
  [`../measurement/SUMMARY.md`](../measurement/SUMMARY.md).
- ⚠ `GRPO_LA5` runs to iteration **5**, not 10, so matched-*iteration* rows hand it ~2× the compute
  per cell; on the compute axis it is budget-matched to `GRPO_LA0` within 3%
  ([`../compute/SUMMARY.md`](../compute/SUMMARY.md)). Every four-arm figure here is censored at 5
  for that arm.
- **Every endpoint is a single 96-conversation draw.** The only measured noise floor is at the base
  (54 same-policy contrasts, 0 significant, max |dz| 0.128 / 0.147). Therapist decoding is
  unseeded, so no conversation set is reproducible.
- **All 96 personas are used for both training and eval**, so everything is in-sample with respect
  to the patient distribution.
