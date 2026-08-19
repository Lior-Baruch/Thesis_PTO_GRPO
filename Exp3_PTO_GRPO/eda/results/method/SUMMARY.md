# Exp3 EDA Summary — `method/` (RQ-ii: PTO vs GRPO at each K)

*Ported from `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` on 2026-08-18 (reorg by research
question); numbers unchanged, paths rewritten.*

> ⚠ **This file is INTERPRETATION, not evidence** — a hand-authored reading of the tables under
> [`contrast/tables/`](contrast/tables/), written in past sessions, largely by Claude. Brainstorm
> from the tables cold, not from here; quote numbers from the tables, not from here. See
> [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs".

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`contrast/{figures,tables}/…` — no `<judge>/` level: the tables carry BOTH graders).
Numbers are full-conversation eval (the held-out outcome), persona-paired over the 96 shared
personas. The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

## What this top covers

`method/` owns the **method contrast**: PTO vs GRPO at matched K (K=0 and K=5), matched MCL=12,
matched oracle (Q1+Q2), matched candidate count (M = G = 8), persona-paired at each iteration and
best-vs-best — [`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md)
(every iteration × metric × K, both graders) and
[`contrast/tables/method_paired_best.md`](contrast/tables/method_paired_best.md) (each arm at its
own peak iteration on its training oracle — the steelman); figure
[`contrast/figures/method_gap.png`](contrast/figures/method_gap.png). In the retired tree the K=0
half of this lived in `L0/SUMMARY.md` §2 and the K=5 half in `L5`; here both K levels are one
table. The per-arm curves the contrast is read against are [`../arms/SUMMARY.md`](../arms/SUMMARY.md)
§2; how look-ahead *changes* the winner (the K×method interaction) is
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) §4; the same contrast at matched **budget**
rather than matched iteration is [`../compute/SUMMARY.md`](../compute/SUMMARY.md) — the two axes
disagree, and every claim below names its axis.

---

## 2. Headline (K=0) — both arms improve a lot, but PTO is stronger *and* more stable
See [`../arms/outcomes/figures/gpt-4o-mini/outcomes_by_model_final.png`](../arms/outcomes/figures/gpt-4o-mini/outcomes_by_model_final.png),
[`../arms/outcomes/figures/gpt-4o-mini/effect_vs_base_forest_final.png`](../arms/outcomes/figures/gpt-4o-mini/effect_vs_base_forest_final.png),
[`../arms/outcomes/figures/gpt-4o-mini/trajectories/trajectory_Q1Q2.png`](../arms/outcomes/figures/gpt-4o-mini/trajectories/trajectory_Q1Q2.png), and
[`../arms/stats/tables/gpt-4o-mini/main_results.md`](../arms/stats/tables/gpt-4o-mini/main_results.md).

- **Each arm vs base — large global-evaluation gains.** PTO_LA0 Q1+Q2 **3.00 → 4.26** (dz 1.43, *large*,
  Holm p≈0, Friedman W=0.45). GRPO_LA0 Q1+Q2 **3.07 → 4.08 at its iter-8 peak**, falling to **3.75
  by iter 10** (final dz 0.72 *medium*, best dz 1.22). Every global-evaluation rubric is a *large* effect for
  PTO; Holm p≈0 everywhere.
- **PTO ahead at the matched 10-iter endpoint.** Paired PTO−GRPO at iter 10: **Q1+Q2 +0.51**
  (dz +0.73, Holm p<0.001), with MITI, MI-SAT, PCT and the Q1/Q2 components all favouring PTO — see
  [`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md). The earlier
  "near-tie at iter 8" was a snapshot: **GRPO peaks at iter 8 then regresses** (4.08 → 3.81 → 3.75)
  while PTO keeps climbing (4.22 → 4.26).
- **Climb rate.** OLS Q1+Q2 slope PTO **0.120/iter** (peak = final iter 10) vs GRPO **0.072/iter**
  (peak iter 8) — [`../arms/stats/tables/gpt-4o-mini/slope_by_arm.md`](../arms/stats/tables/gpt-4o-mini/slope_by_arm.md). With
  GRPO, peak-iter selection / early stopping matters; even so its best (4.08) is below PTO's (4.26).
  The best-vs-best steelman (PTO@10 − GRPO@8) is
  [`contrast/tables/method_paired_best.md`](contrast/tables/method_paired_best.md); it is one of the
  six contrasts the held-out judge preserves 18/18 with bootstrap CIs excluding zero
  ([`../measurement/SUMMARY.md`](../measurement/SUMMARY.md)), and Haiku *widens* the headline
  PTO−GRPO Q1 gap (+0.77 vs the primary's +0.53).
- **Iter-9 caveat:** GRPO_LA0 dips at iter 9 across most metrics simultaneously then partially
  recovers at 10 — [`../arms/validity/tables/gpt-4o-mini/grpo_iter9_check.md`](../arms/validity/tables/gpt-4o-mini/grpo_iter9_check.md)
  quantifies it (a paired one-iteration dip on top of the monotonic Q1+Q2 decline).

**Revised core answer (RQ-ii):** GRPO is competitive *up to its peak* but overshoots into
reward-hacking and degrades; **PTO sustains gains across all 10 iterations.** The gains come *with*
a measurable reward-hack in both arms, worse in GRPO (MICI base 0.21 → 0.49 PTO / 0.84 GRPO at
iter 10) — [`../arms/SUMMARY.md`](../arms/SUMMARY.md) §3–§4; the training-side reading of *why*
the two methods diverge (the gap is about the data the two methods train on, not the loss —
swapping the weighting rule on the same groups barely moves the update direction, 0.908 / 0.988)
is [`../arms/SUMMARY.md`](../arms/SUMMARY.md) §6.

## 2b. PTO is also 3.4× cheaper — the comparison on the COMPUTE axis
Source: [`../compute/cost/tables/compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md)
and [`../compute/cost/tables/iso_compute_contrast.md`](../compute/cost/tables/iso_compute_contrast.md)
(the compute axis is owned by [`../compute/`](../compute/SUMMARY.md); backing module
`eda_analysis/compute.py`).

Every contrast above is indexed by **iteration**, which is not a fixed unit of spend. Reconstructed
from artifact mtimes:

| arm | iterations | GPU-h | h / iteration |
|---|---|---|---|
| `PTO_LA0`  | 10 | **8.1** | 0.81 |
| `GRPO_LA0` | 10 | **27.9** | 2.79 |

**PTO reaches the same iteration 10 for 27.91 / 8.12 = 3.4× less compute, and scores higher.** On
the compute axis PTO dominates GRPO outright — a strictly stronger claim than the matched-iteration
one, and one that does not depend on the grader. The reason is structural rather than incidental:
PTO's preference-tree **build** (5.7 of its 8.1 h) runs *once per iteration*, whereas GRPO
recomputes its reward *inside* the training loop on every optimizer step.

⚠ **At matched BUDGET the picture is split, and worth stating honestly.** At 8.1 GPU-h — PTO's
entire run — GRPO has only reached iteration 3 (8.21 h, budget ratio 1.011). `PTO_LA0 @10` vs
`GRPO_LA0 @3`, persona-paired:

| metric | primary Δ / dz / p_holm | held-out Δ / dz / p_holm |
|---|---|---|
| Q1+Q2 | **+0.266 / 0.529 / <.001** | **+0.230 / 0.456 / .0002** |
| MICI (lower better) | **+0.261 / 0.904 / <.001** | **+0.418 / 1.280 / <.001** |
| MITI | +0.336 / 0.602 / <.001 | −0.031 / ns |

PTO wins the reward at equal spend but is **markedly worse on MI-inconsistency** there — because at
equal spend it has trained ten iterations to GRPO's three, and MI-inconsistency accumulates with
training depth in both methods ([`../arms/SUMMARY.md`](../arms/SUMMARY.md) §3). **The honest
summary is: PTO buys more reward per GPU-hour, and more reward-hacking per GPU-hour with it.**
Neither half should be quoted without the other. The method budget sweeps are
[`../compute/cost/tables/budget_sweep_method_K0_gpt-4o-mini.md`](../compute/cost/tables/budget_sweep_method_K0_gpt-4o-mini.md)
(+ `_claude-haiku-4-5`, and the `method_K5` pair) — quote the sweep, not one row.

## 3. At K=5 the winner flips — see `lookahead/`

At iteration 5, with both K arms of both methods scored on the same 96 personas, PTO leads under
K=0 (held-out +0.265, dz 0.355, p_holm .014) and GRPO leads under K=5 (−0.219, dz 0.377, p_holm
.005); the difference-in-differences is dz 0.525 (p_holm .0001) on the held-out judge and null on
the primary. The primary oracle does see GRPO > PTO under K=5 one iteration earlier (iter 4 Q1Q2
−0.232, dz −0.351, p_holm .024; MITI dz −0.411) — the K=5 rows of
[`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md). Full treatment,
scope (GRPO_LA5 stops at 5) and the second-judge reading:
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) §4.

## 4. Caveats
- **Every claim needs its axis named.** Matched iteration and matched budget give different winners
  on MICI, and both are correct answers to different questions ([`../compute/SUMMARY.md`](../compute/SUMMARY.md)).
- ⚠ `GRPO_LA5` runs to iteration **5**, not 10, so its matched-*iteration* rows hand it ~2× the
  compute per cell; on the compute axis it is budget-matched to `GRPO_LA0` within 3%.
- Absolute scores are **Exp3-internal only** — not comparable to Exp2 (4-bit vs bf16 generation),
  and **never comparable across judges** (Haiku's level offset is 1.2–1.7 points). Contrasts, not
  levels, are what the two graders share.
- MITI arm differences are **provisional** — see the MITI warning in
  [`../measurement/SUMMARY.md`](../measurement/SUMMARY.md).
- **Every endpoint is a single 96-conversation draw**; therapist decoding is unseeded. **All 96
  personas are used for both training and eval**, so everything is in-sample with respect to the
  patient distribution.
