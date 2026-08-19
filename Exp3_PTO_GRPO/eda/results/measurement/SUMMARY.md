# Exp3 EDA Summary — `measurement/` (is the ruler trustworthy? judge validity + multi-judge)

*Ported from `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` on 2026-08-18 (reorg by research
question); numbers unchanged, paths rewritten.*

> ⚠ **This file is INTERPRETATION, not evidence** — a hand-authored reading of the tables under
> [`validity/tables/`](validity/tables/), written in past sessions, largely by Claude. Brainstorm
> from the tables cold, not from here; quote numbers from the tables, not from here. See
> [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs".

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`validity/{figures,tables}/…` — no `<judge>/` level: every artifact here contains EVERY
grader). The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

## What this top covers

`measurement/` asks whether the measuring instrument can be trusted: §1 single-judge validity
(oracle repeatability ICC, second-judge agreement, contrast preservation) and §2 the multi-judge
layer (variance decomposition of the arm means, gain retention, every pairwise arm×metric contrast
under both graders, the sign-preservation ladder, concordance vs effect size) — read from the score
lake by `eda_analysis/reliability.py`, no API calls (the paid scoring is
`notebooks/scoring/Judge_Reliability.ipynb`). It is judge-*invariant* by construction: rendered
once, no `<judge>/` leaf. In the retired tree this was `8_measurement/` under each view — `L0`
(22 model states) carried the narrative below and `L5`'s retention table was empty until 2026-08-18;
now every table spans **all 39 model states** of the four arms. The K-specific transfer question
(does the K contrast keep its sign under the held-out judge; retention by K) is
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) §6–§6b / [`../lookahead/transfer/`](../lookahead/transfer/).
Full treatment + what is still uncovered: [`LIMITATIONS.md`](../LIMITATIONS.md) §1–§3; the metric
definitions: [`METRICS_REFERENCE.md`](../METRICS_REFERENCE.md) §7–§7b.

---

## 7. Does the result survive a different judge? Yes — and the reward-hack gets *sharper*
See [`validity/figures/multijudge_variance_decomposition.png`](validity/figures/multijudge_variance_decomposition.png),
[`validity/figures/multijudge_gain_retention.png`](validity/figures/multijudge_gain_retention.png),
[`validity/figures/multijudge_retention_trajectory.png`](validity/figures/multijudge_retention_trajectory.png),
and the tables in [`validity/tables/`](validity/tables/).

Every conversation of the K=0 arms was **re-scored by Claude Haiku 4.5** — a different model family
that never played the patient and never touched training — and so, since, was every conversation
of the K=5 arms. The full project grid is 39 model states × 8 rubrics × 96 convs =
**29,952 / 29,952 cells**, of which the K=0 arms' 22 model states account for 22 × 8 × 96 =
**16,896** (the counts below were computed on those 22 states; the live count is
[`validity/tables/multijudge_coverage.md`](validity/tables/multijudge_coverage.md)). The primary
oracle *was* the training reward, so this is a held-out grader, not a second rater: **never average
the two**, only compare contrasts.

- **The rankings survive.** Across **all 1,848** pairwise arm×metric contrasts
  ([`validity/tables/multijudge_all_pairs_contrasts.md`](validity/tables/multijudge_all_pairs_contrasts.md)),
  **88.3%** keep their sign under the held-out judge — rising to **94.1%** at |Δ|≥0.10, **97.0%** at
  ≥0.25 and **98.9%** at ≥0.50
  ([`validity/tables/multijudge_sign_preservation.md`](validity/tables/multijudge_sign_preservation.md)).
  The judges disagree only about differences too small to claim. The six
  contrasts the K=0 summary leans on hardest — including the best-vs-best steelman (PTO@10 − GRPO@8)
  and the regression claim (GRPO@8 − GRPO@10) — are **18/18** preserved, with bootstrap CIs
  excluding zero. Haiku *widens* the headline PTO−GRPO Q1 gap (+0.77 vs the primary's +0.53).
- **They disagree about level, not order.** A two-way decomposition of the arm means
  ([`validity/tables/multijudge_variance_components.md`](validity/tables/multijudge_variance_components.md))
  puts only **1–7%** of variance in the arm×judge interaction — the only component that could
  invalidate a claim. Haiku is systematically harsher (Q1 −1.25 to −1.74), which cancels in every
  contrast. Dependability of an arm mean from one judge is **0.88–0.95** on seven rubrics.
- **⚠ MITI is the exception.** Only **3.6%** of MITI's arm-mean variance is between-arm signal
  (94.5% is grader level); dependability **0.65**, and it preserves its sign on only **77.5%** of
  contrasts — the worst of the eight
  ([`validity/tables/multijudge_sign_preservation_by_metric.md`](validity/tables/multijudge_sign_preservation_by_metric.md)).
  Worse, it is the only rubric that still disagrees at *claimable* effect sizes: every other rubric
  reaches 95.5–100% by |Δ|≥0.25, MITI only 88.2%. **Treat the MITI numbers in
  [`../arms/SUMMARY.md`](../arms/SUMMARY.md) §4 as provisional unless both judges agree on the
  direction** — in particular the "neither arm reaches good on the technique ratios" verdict.
  Q1/Q2/PCT/MICI are unaffected.
- **The strongest new evidence for the reward-hack — gain retention.** `Δ(held-out) /
  Δ(trained-against)` is a train/test generalization ratio per arm
  ([`validity/tables/multijudge_gain_retention.md`](validity/tables/multijudge_gain_retention.md)).
  At iter 10, **Q1 retention is PTO 0.80 [0.68, 0.93] vs GRPO 0.28 [0.06, 0.43] — non-overlapping**,
  while every Q2 interval overlaps (0.80–0.85, i.e. uninteresting scale compression). **Under a
  judge that never graded during training, GRPO's net 10-iteration Q1 gain is ≈0.19 points, not the
  ≈0.68 the primary credits it.**
- **And it is an *onset* curve, not an endpoint accident.** Q1 retention by iteration:

  | iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | PTO_LA0 | 0.97 | 0.84 | 0.89 | 0.94 | 0.98 | 0.97 | 0.94 | 0.89 | 0.88 | **0.80** |
  | GRPO_LA0 | 1.13 | 0.79 | 0.89 | 0.79 | 0.73 | 0.57 | 0.70 | 0.64 | 0.03 | **0.28** |

  The arms are indistinguishable for three iterations, then separate: PTO holds 0.80–0.98 for the
  whole run while GRPO decays monotonically in trend to 0.28. The held-out grader stops crediting
  GRPO's gains *progressively* — which is what a policy drifting onto grader-specific features looks
  like, and a cleaner signal than the MICI rate (whose cross-judge agreement is weak, r 0.20–0.55).
  Retention for the K=5 arms is method-dependent in the same direction as the reward (GRPO K=5
  retains its full Q1 gain, PTO K=5 the same or less) — [`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) §6b.

**Net:** the PTO-vs-GRPO conclusion is not an artifact of the patient and the grader sharing a
model. Full treatment + what is still uncovered: [`LIMITATIONS.md`](../LIMITATIONS.md) §1–§3.

## 8. Second-judge scope on the K contrast

Every K contrast is reported on both graders because the primary oracle **was** the training reward.
The held-out judge has the full grid for every arm. Where the two disagree it is stated inline in
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md), and the disagreements are not decorative: the
K×method interaction is significant on one grader and null on the other, and the aggregate
MI-inconsistency claim flips with it. Never average the two — that is train-vs-test, not two
raters. Combine contrasts only.

## 9. Caveats
- Oracle reproducibility is **measured**, not assumed: ICC(2,1) **0.86–0.99**, mean |Δ| 0.04–0.09
  across four draws (Q1/Q2 0.96–0.99; only MICI falls below 0.90, floor 0.864 at PTO@10) — the
  project's informal "≈0.10 noise" figure is a conservative upper bound, and it shrinks by ~√96 at
  the arm-mean level the summaries report.
- Absolute scores are **never comparable across judges** (Haiku's level offset is 1.2–1.7 points)
  and **Exp3-internal only** (not comparable to Exp2: 4-bit vs bf16 generation).
- **`MITI` dependability is 0.553 and `MICI` 0.628** on the four-arm grid, and those two
  instruments carry the behaviour-channel claims. There is **no channel-level ICC at all**, and
  **no repeatability rep for any K=5 state**.
- **Every endpoint is a single 96-conversation draw.** The only measured noise floor is at the base
  (54 same-policy contrasts, 0 significant, max |dz| 0.128 / 0.147). Therapist decoding is
  unseeded, so no conversation set is reproducible.
- **All 96 personas are used for both training and eval**, so everything is in-sample with respect
  to the patient distribution.
