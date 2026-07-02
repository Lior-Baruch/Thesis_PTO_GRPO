# Exp3 EDA Summary — view `L0` (no look-ahead, K=0)

*Hand-authored narrative. Preserved across reruns / `reset_results`. Artifacts are referenced by
relative path; numbers are full-conversation eval (the held-out outcome), persona-paired over the
96 shared personas. The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

This is the **primary comparison view**: PTO vs GRPO at matched look-ahead K=0, both run to 10
iterations and fully scored on the expanded battery. (The K=0 vs K=5 question lives in the `L5`
view, which is still thin.)

---

## 1. What this view covers
- **Arms:** `PTO_LA0` (pref-tree → DPO) and `GRPO_LA0` (group-relative), both K=0, MCL=12, oracle =
  Q1+Q2, 10 training iterations + base.
- **Metrics:** the 5 warmth/alliance rubrics (Q1Q2, WAI-SR, CSQ-8, MI-SAT, MITI) **plus** the
  orthogonal axes added to break the warmth halo — `PCT` (patient change-talk), `MICI` (MI-
  inconsistent therapist behaviour, **lower = better**), and the free derived MITI-proficiency
  ratios `R:Q` / `%CR` / `%MICO`.

## 2. Headline — both arms improve a lot, but PTO is stronger *and* more stable
See [`figures/1_outcomes/outcomes_by_model.png`](figures/1_outcomes/outcomes_by_model.png),
[`figures/1_outcomes/effect_vs_base_forest.png`](figures/1_outcomes/effect_vs_base_forest.png),
[`figures/1_outcomes/trajectories/trajectory_Q1Q2.png`](figures/1_outcomes/trajectories/trajectory_Q1Q2.png), and
[`tables/6_stats/main_results.md`](tables/6_stats/main_results.md).

- **Each arm vs base — large warmth gains.** PTO_LA0 Q1+Q2 **3.00 → 4.26** (dz 1.43, *large*,
  Holm p≈0, Friedman W=0.45). GRPO_LA0 Q1+Q2 **3.07 → 4.08 at its iter-8 peak**, falling to **3.75
  by iter 10** (final dz 0.72 *medium*, best dz 1.22). Every warmth rubric is a *large* effect for
  PTO; Holm p≈0 everywhere.
- **PTO ahead at the matched 10-iter endpoint.** Paired PTO−GRPO at iter 10: **Q1+Q2 +0.51**
  (dz +0.73, Holm p<0.001), with MITI, MI-SAT, PCT and the Q1/Q2 components all favouring PTO — see
  [`tables/6_stats/method_paired_by_K.md`](tables/6_stats/method_paired_by_K.md). The earlier
  "near-tie at iter 8" was a snapshot: **GRPO peaks at iter 8 then regresses** (4.08 → 3.81 → 3.75)
  while PTO keeps climbing (4.22 → 4.26).
- **Climb rate.** OLS Q1+Q2 slope PTO **0.120/iter** (peak = final iter 10) vs GRPO **0.072/iter**
  (peak iter 8) — [`tables/6_stats/slope_by_arm.md`](tables/6_stats/slope_by_arm.md). With
  GRPO, peak-iter selection / early stopping matters; even so its best (4.08) is below PTO's (4.26).
- **Per-metric learning curves** (every metric, peaks auto-flagged) live in
  [`figures/1_outcomes/trajectories/`](figures/1_outcomes/trajectories/); the persona splits
  (every metric × cooperation/problem) in [`figures/2_heterogeneity/`](figures/2_heterogeneity/) —
  GRPO's endpoint collapse concentrates on the *Resistant* personas.
- **Iter-9 caveat:** GRPO_LA0 dips at iter 9 across most metrics simultaneously then partially
  recovers at 10 — [`tables/6_stats/grpo_iter9_check.md`](tables/6_stats/grpo_iter9_check.md)
  quantifies it (a paired one-iteration dip on top of the monotonic Q1+Q2 decline).

**Revised core answer (RQ-ii):** GRPO is competitive *up to its peak* but overshoots into
reward-hacking and degrades; **PTO sustains gains across all 10 iterations.**

## 3. The gains come *with* a measurable reward-hack — that's why the orthogonal axes matter
See [`figures/3_mechanism/factor_loadings.png`](figures/3_mechanism/factor_loadings.png),
[`figures/3_mechanism/rubric_correlation.png`](figures/3_mechanism/rubric_correlation.png), and
[`tables/1_outcomes/leaderboard_scorecard.md`](tables/1_outcomes/leaderboard_scorecard.md).

- **MI-inconsistent behaviour rises ~2.3–2.5×** as warmth climbs (MICI base 0.21 → 0.49 PTO /
  0.84 GRPO at iter 10; GRPO's MICI effect is dz 1.72, *large*). The warmth gains are partly
  over-praise/advice in **both** methods, **worse in GRPO**.
- **Adding the orthogonal axes drops PC1 from ≈91% → ≈55%** (per-arm PC1 ≈55–56%). Warmth is one
  factor; technique (R:Q/%CR/%MICO) + MI-inconsistency form a second — so "all rubrics up" is *not*
  multi-skill. PCT partly co-moves with warmth (loads ~0.39 on PC1).
- **Patient change-talk (PCT) rises modestly**, more for PTO (0.49 → 0.63, *medium*) than GRPO
  (0.49 → 0.57, *small*).

## 4. Mechanism — what the therapist actually does
See [`figures/3_mechanism/behavior_drift.png`](figures/3_mechanism/behavior_drift.png) and the merged
behaviour table [`tables/3_mechanism/behavior_by_iter.md`](tables/3_mechanism/behavior_by_iter.md).

- **Affirmation drift is confirmed in BOTH arms, and at iter 10 GRPO is the worse offender:**
  GRPO B6_AF 0.52 → **1.98**, questions B3_Q 6.4 → **4.1**, q/turn 0.83 → **0.15**, R:Q → **1.44**.
  PTO's drift is milder and plateaus (iter-10 B6_AF 1.64, q/turn 0.55).
- **Across all 96 iter-10 conversations** (`tables/3_mechanism/behavior_by_iter.md`): GRPO
  collapses to **0.15 questions/turn** vs PTO's **0.55**, and the oracle codes GRPO as far more
  MI-inconsistent (**MICI 0.84 vs 0.49**). A lexical praise-word count (the demoted sanity-check)
  puts GRPO at **~3.5× PTO's praise rate**. The iter-10 eval regression *is* this over-praise
  reward-hack, which the full-conversation oracle penalises; GRPO falls into it harder.
- Both arms kill the early degeneration loops (loop% 0.49 → 0); the leak/empty health gate stays
  clean (see [`figures/4_training/`](figures/4_training/)).

## 5. Is the training reward faithful?
See [`figures/4_training/reward_reliability_curve.png`](figures/4_training/reward_reliability_curve.png).
At MCL=12, GRPO's short proxy reward grows *more* faithful with conversation length (rank agreement
≈0.86 → 0.94) while PTO's grows *less* (≈0.87 → 0.76). The MCL=12 floor keeps both out of the
unreliable short-cut regime (Exp2 saw agreement as low as 0.66 at n_turns=2).

## 6. PTO preference probe (PTO_LA0 only)
See [`figures/5_preference/`](figures/5_preference/). The mean(chosen − rejected) direction genuinely
separates the pairs (`wins_correct` 0.65 → 0.71 over iters, strengthening late) — the DPO signal is
real, and its latent target drifts toward affirmation/achievement language over training (the
latent-space echo of the §4 behaviour drift).

## 7. Caveats
- Oracle reproducibility noise ≈ 0.10 mean |Δ|; effects above that band are structural.
- Absolute scores are **Exp3-internal only** — not comparable to Exp2 (4-bit vs bf16 generation).
- This view is K=0 only; the look-ahead (K=5) comparison is in the `L5` view (currently thin,
  paused for OpenAI-API budget).
