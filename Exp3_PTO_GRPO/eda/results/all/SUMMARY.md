# Exp3 EDA Summary — view `all` (every arm)

*Hand-authored narrative. Preserved across reruns / `reset_results`. Artifacts are referenced by
relative path; numbers are full-conversation eval, persona-paired over the 96 shared personas. The
auto-generated artifact map is [`INDEX.md`](INDEX.md).*

This view pools all four arms — `PTO_LA0`, `GRPO_LA0` (both K=0, 10 iters, fully scored) and
`PTO_LA5`, `GRPO_LA5` (K=5, **thin and paused for budget**). For the clean head-to-head read the
[`L0` view](../L0/SUMMARY.md); for the look-ahead question read the [`L5` view](../L5/SUMMARY.md).
This page is the cross-cutting overview.

---

## 1. What this view covers
| Arm | K | Iters scored | Status |
|---|---|---|---|
| `PTO_LA0`  | 0 | 0–10 | complete |
| `GRPO_LA0` | 0 | 0–10 | complete (FINISHED) |
| `PTO_LA5`  | 5 | 0–4  | partial (paused) |
| `GRPO_LA5` | 5 | 0–1 | thin — dropped from per-arm batteries (<3 iters) |

Metrics: 5 warmth rubrics (Q1Q2, WAI-SR, CSQ-8, MI-SAT, MITI) + orthogonal axes (PCT, MICI↓,
R:Q/%CR/%MICO). See [`tables/1_outcomes/leaderboard_scorecard.md`](tables/1_outcomes/leaderboard_scorecard.md)
for the one-glance arm × metric scorecard.

## 2. Headline (RQ-ii, K=0) — PTO ahead at the matched endpoint
See [`figures/1_outcomes/`](figures/1_outcomes/) and
[`tables/6_stats/main_results.md`](tables/6_stats/main_results.md).

- **PTO_LA0 Q1+Q2 3.00 → 4.26** (dz 1.43, *large*); **GRPO_LA0 peaks 4.08 @ iter 8 then regresses
  to 3.75 @ iter 10** (final dz 0.72). At the matched 10-iter endpoint **PTO beats GRPO 4.26 vs
  3.75** (paired +0.51, dz +0.73, Holm p<0.001) —
  [`tables/6_stats/method_paired_by_K.md`](tables/6_stats/method_paired_by_K.md).
- OLS slope PTO 0.120/iter vs GRPO 0.072/iter
  ([`tables/6_stats/slope_by_arm.md`](tables/6_stats/slope_by_arm.md)).
- **Answer:** GRPO is competitive up to its peak but overshoots into reward-hacking and degrades;
  PTO sustains gains. With GRPO, early stopping matters.

## 3. Reward-hacking is real and multi-skill is not free
See [`figures/3_mechanism/factor_loadings.png`](figures/3_mechanism/factor_loadings.png) and
[`figures/3_mechanism/behavior_drift.png`](figures/3_mechanism/behavior_drift.png).

- **MI-inconsistent behaviour rises ~2.3× (PTO) / ~4× (GRPO)** as warmth climbs (MICI base 0.21 →
  0.49 PTO / 0.84 GRPO @ iter 10). **Affirmation drift is in BOTH methods; GRPO is worse late** (B6_AF → 1.98,
  q/turn → 0.15, R:Q → 1.44 by iter 10 vs PTO B6_AF 1.64, q/turn 0.55).
- **Adding the orthogonal axes drops PC1 ≈91% → ≈55%** — warmth is one factor, technique + MI-
  inconsistency a second. "All rubrics up" ≠ multi-skill.
- Patient change-talk rises modestly, more for PTO (PCT 0.49 → 0.63) than GRPO (0.49 → 0.57).

## 4. Look-ahead (RQ-i, K=0 vs K=5) — preliminary
`PTO_LA5` reaches 3.00 → 3.89 in 4 iters (dz 0.88); `GRPO_LA5` has only 1 scored iter. No significant
K0-vs-K5 difference yet at matched early iters
([`tables/6_stats/k_paired_by_method.md`](tables/6_stats/k_paired_by_method.md)). On reward
faithfulness, look-ahead helps slightly (LA5 ≥ LA0). **Both LA5 arms are paused for OpenAI-API
budget** — RQ-i is on hold. Full detail: [`L5` view](../L5/SUMMARY.md).

## 5. Training health & reward faithfulness
- Degeneration gate clean across arms (leak ≈ 0, loops eliminated) —
  [`figures/4_training/`](figures/4_training/).
- Short-reward faithfulness at MCL=12: GRPO grows more faithful with length (≈0.86 → 0.94), PTO less
  (≈0.87 → 0.76) — [`figures/4_training/`](figures/4_training/).
- PTO preference probe is real (`wins_correct` 0.65 → 0.71) — [`figures/5_preference/`](figures/5_preference/).

## 6. Caveats
- Oracle noise ≈ 0.10 mean |Δ|; thin arms (<3 iters, i.e. `GRPO_LA5`) are dropped from per-arm
  stats to avoid NaN rows.
- Absolute scores are Exp3-internal only (bf16); not comparable to Exp2 (4-bit).
- The K=5 arms are partial — treat all RQ-i statements as preliminary.
