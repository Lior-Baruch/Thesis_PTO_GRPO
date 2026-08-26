# Exp3 EDA Summary — `measurement/` (is the ruler trustworthy? judge validity + multi-judge)

> ⚠ **This file is INTERPRETATION, not evidence** — a hand-authored reading of the tables under
> [`validity/tables/`](validity/tables/), written in past sessions, largely by Claude. Brainstorm
> from the tables cold, not from here; quote numbers from the tables, not from here. See
> [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs".

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`validity/{figures,tables}/…` — no `<judge>/` level: every artifact here contains EVERY
grader). The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

*Rewritten 2026-08-25 on the completed grid, replacing the reading written while GRPO_LA5 was
right-censored. Every number below was re-read off a table for this rewrite; retracted claims are
marked inline in the `(Corrected 2026-08-25: …)` style rather than silently overwritten. The
pre-completion narrative — and the `results/L0/` + `results/L5/` files it was ported from on
2026-08-18 — are recoverable from git history.*

## What this top covers

`measurement/` asks whether the measuring instrument can be trusted — everything else in
`results/` is a claim *made with* this instrument. One subfamily, `validity`, in six parts: the
grid it is measured on (§1), whether the primary oracle repeats itself (§2), whether a grader from
a different model family sees the same ordering (§3), the one place where it emphatically does
**not** (§4), gain retention as a train/test generalization ratio (§5), and the caveats that
constrain how every other summary may be written (§6). Everything is read from the score lake by
`eda_analysis/reliability.py` — no API calls; the paid scoring is
`notebooks/scoring/Judge_Reliability.ipynb`.

The family is **judge-invariant by construction**: rendered once, with both graders inside every
artifact and no `<judge>/` leaf, because comparing the two graders *is* the analysis.

Neighbours, so this file does not duplicate them: the K-specific transfer question (does the K
contrast keep its sign under the held-out judge; retention read as a K contrast) is
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) and
[`../lookahead/transfer/`](../lookahead/transfer/). The measurement-*quality* evidence and the
caveats that follow from it — both graders' ICCs, agreement against the attenuation ceiling, sweep
provenance — are owned by [`LIMITATIONS.md`](../LIMITATIONS.md) §1–§3, which cites this file for the
findings rather than restating them. Metric definitions, no values:
[`METRICS_REFERENCE.md`](../METRICS_REFERENCE.md) §7–§7b. Current run status:
[STATUS.md](../../../../STATUS.md).

✅ *(Resolved 2026-08-25 — kept so the warning's history is visible.)* This file warned that some
auto-generated captions here still read "GRPO K=5 is right-censored and ends first." The tree-wide
caption purge later the same day (see STATUS.md) removed every such string — support is now derived
per render. Verified by grep on 2026-08-26: no caption in this top asserts censoring.

---

## 1 · The grid: what the ruler was measured on

Two graders scored the same conversations, and they are **not** two interchangeable raters:

- **primary** = `gpt-4o-mini` = **the training oracle**. It *was* the reward. Its scores are the
  optimization target.
- **held-out** = `claude-haiku-4-5`, a different model family that never played the patient and
  never touched training.

Every comparison in this file is therefore **optimization-target vs held-out-test**, not inter-rater
reliability. ⚠ **Never average the two graders' raw scores** — see §6.

Coverage is now the complete grid.
[`validity/tables/multijudge_coverage.md`](validity/tables/multijudge_coverage.md) is
**44 × 8 = 352** rows — `4 × 11 = 44` model states (base + iterations 1–10 in each of the four
arms) × 8 rubrics — and every one of them reads `96 / 96`, `complete = True`. At the conversation
level that is **44 × 8 × 96 = 33,792**
cells *per grader*, with nothing quarantined: `filter_complete_cells` drops zero rows, so every
multi-judge statistic below runs at full n = 96 per cell.

*(Corrected 2026-08-25: this section read "39 model states × 8 rubrics × 96 convs =
**29,952 / 29,952 cells**", of which "the K=0 arms' 22 model states account for 22 × 8 × 96 =
**16,896** (the counts below were computed on those 22 states)". Both described the censored grid;
everything below is now computed on all 44 states.)*

## 2 · Does the primary oracle repeat itself? Yes — and the noise floor is small

See [`validity/figures/oracle_repeatability_icc.png`](validity/figures/oracle_repeatability_icc.png),
[`validity/tables/oracle_repeatability_by_metric.md`](validity/tables/oracle_repeatability_by_metric.md)
and the per-model
[`validity/tables/oracle_repeatability_icc.md`](validity/tables/oracle_repeatability_icc.md).

Repeatability is **measured, not assumed** — but on a deliberately narrow anchor subset:
**3 × 4 × 96 = 1,152** conversation-level cells per rep (3 metrics × 4 model states × 96
conversations), with `n_reps = 4` draws per cell (three seeded reps plus the draw the thesis
actually reports). Pooled per metric over `4 × 96 = 384` conversations:

| metric | ICC(2,1) | mean \|Δ\| between reps | per-model span (4 states) |
|---|---|---|---|
| Q1 | 0.990 | 0.056 | 0.982 – 0.994 |
| Q2 | 0.976 | 0.081 | 0.955 – 0.992 |
| MICI | 0.916 | 0.054 | 0.864 – 0.943 |

Q1 and Q2 are "excellent" on the Koo & Li (2016) ≥0.90 guideline; MICI's weakest cell
(`PTOExp3_LA0_I10`, 0.864) is "good" rather than excellent. The project's informal "oracle noise
≈ 0.10" figure is a conservative upper bound on all three, and since every arm-level claim is a mean
over 96 conversations, this per-conversation noise shrinks by ~√96 at the level the summaries
actually report.

⚠ **Two structural gaps, both live.** First, **all four anchor states are K=0** (`PTOExp3_LA0_Base`,
`PTOExp3_LA0_I10`, `GRPOExp3_LA0_I8`, `GRPOExp3_LA0_I10` — read the `model` column). **No K=5 state
has a repeatability rep on either grader**, so there is **no attenuation ceiling for the look-ahead
arms at all** — which is exactly where §4's problem lives. Second, re-seeding at `temperature=0.1`
probes *sampling* noise only; it says nothing about sensitivity to rubric wording or item order.
Both are elaborated in [`LIMITATIONS.md`](../LIMITATIONS.md) §1.

## 3 · Does a held-out grader see the same thing? At the arm level, yes

See [`validity/figures/judge_agreement_scatter.png`](validity/figures/judge_agreement_scatter.png),
[`validity/figures/judge_contrast_preservation.png`](validity/figures/judge_contrast_preservation.png),
[`validity/figures/multijudge_variance_decomposition.png`](validity/figures/multijudge_variance_decomposition.png)
and [`validity/figures/multijudge_arm_means_dumbbell.png`](validity/figures/multijudge_arm_means_dumbbell.png).

**They disagree about level, not about order.** A two-way random-effects decomposition of the arm
means the thesis reports
([`validity/tables/multijudge_variance_components.md`](validity/tables/multijudge_variance_components.md),
`n_arms = 44` model states × 2 graders, every cell n = 96) puts only **1.1–7.0%** of arm-mean
variance in the **arm × judge** interaction — the only component that could invalidate a claim. The
judge-level term is large (85.4% on Q1, 93.9% on MITI) and harmless: it cancels in every contrast.
`dependability_k1`, the generalizability of an arm mean read off *one* grader, is **0.914–0.974** on
**six of the eight** rubrics — **MITI (0.624) and MICI (0.812) are the exceptions**, and both
reappear in §4 and §6; averaging both graders lifts Q1 only 0.928 → 0.963 and Q2 only
0.914 → 0.955, which is the quantitative reason the design bought **breadth** (all states × both
graders) over **depth** (more reps of a few cells).
*(Corrected 2026-08-25: this read "**seven** of the eight rubrics", which silently counted MICI's
0.812 as inside a 0.914–0.974 band and contradicted §6 of this same file, where MICI is stated as
0.812. Six. The eight values are WAI-SR 0.948, CSQ-8 0.945, MI-SAT 0.955, MITI 0.624, PCT 0.974,
MICI 0.812, Q1 0.928, Q2 0.914.)*

**Rank agreement per conversation, read against its ceiling.** Per-`(metric, model)` Pearson r lives
in [`validity/tables/second_judge_agreement.md`](validity/tables/second_judge_agreement.md) (the
`.md` is a 60-row excerpt; all 352 rows are on sheet `second_judge_agreement` of
[`validity/tables/validity.xlsx`](validity/tables/validity.xlsx)). Median of each rubric's 44 rows:

| rubric | PCT | MI-SAT | WAI-SR | CSQ-8 | **Q1** | **Q2** | MITI | MICI |
|---|---|---|---|---|---|---|---|---|
| median r | 0.954 | 0.930 | 0.922 | 0.903 | **0.855** | **0.784** | 0.658 | 0.518 |

⚠ **Compare r to the attenuation ceiling, never to 1.0.** On the four anchor cells where both
graders' ICCs are measured, `r_pct_of_ceiling` runs **85.8–90.8%** on Q1 and **83.2–88.2%** on Q2 —
Q1/Q2 recover ~85–90% of the agreement two raters of this reliability could possibly reach. MICI
recovers only **29.3–59.0%** of its (corrected, much lower) ceiling. The ceiling exists on those four
cells only; every other row reads `ceiling_basis = "no ICC measured for this cell"`.

**The hand-picked endpoint contrasts all survive the swap — 16/16.**
[`validity/tables/second_judge_contrasts.md`](validity/tables/second_judge_contrasts.md) holds
2 endpoint contrasts (both at K=0) × 8 rubrics = 16 rows, every one `same_sign = True`. On
`PTO_LA0_I10 − GRPO_LA0_I10` the held-out grader *widens* the headline Q1 gap rather than shrinking
it (judge +0.773 vs primary +0.533), and flips nothing.
*(Corrected 2026-08-25: this said "**18/18** preserved" and named among them "the best-vs-best
steelman (PTO@10 − GRPO@8)" and "the regression claim (GRPO@8 − GRPO@10)". Neither pair is in that
table — it holds `PTO_LA0_I10 − GRPO_LA0_I10` and `PTO_LA0_I10 − PTO_LA0_Base` and nothing else, so
16 rows is the whole table. Any other contrast must be read off
[`validity/tables/multijudge_all_pairs_contrasts.md`](validity/tables/multijudge_all_pairs_contrasts.md).)*

**And so does the whole grid, wherever the gap is big enough to claim.** `all_pairs_contrasts`
enumerates *every* unordered model-state pair × rubric — **44 × 43 / 2 = 946** pairs,
**8 × 946 = 7,568** contrasts, persona-paired with a seeded bootstrap CI. The sign-preservation
ladder
([`validity/tables/multijudge_sign_preservation.md`](validity/tables/multijudge_sign_preservation.md)):

| subset | n contrasts | same sign | % |
|---|---|---|---|
| all contrasts | 7,568 | 6,693 | **88.4** |
| \|Δ primary\| ≥ 0.10 | 5,071 | 4,793 | 94.5 |
| \|Δ primary\| ≥ 0.25 | 3,236 | 3,146 | 97.2 |
| \|Δ primary\| ≥ 0.50 | 1,692 | 1,681 | **99.3** |
| judge CI excludes 0 | 5,361 | 5,144 | 96.0 |

The graders disagree **only about differences too small to claim**, which is the pattern a
trustworthy instrument should show. Read the row at the effect size you are actually claiming; the
pooled row is dragged down by contrasts nobody would report.

**The pooled rate's stability is itself the evidence.** Across the last three renders of this table
the pooled `all contrasts` figure has been **88.4% → 88.5% → 88.4%**, at
`39 × 38 / 2 × 8 = 5,928`, `40 × 39 / 2 × 8 = 6,240` and `44 × 43 / 2 × 8 = 7,568` contrasts
respectively (git history of `multijudge_sign_preservation.md`). The grid grew by `44 − 40 = 4`
whole model states — including the late GRPO_LA5 states §4 is about — and the rate did not move. Two graders
whose agreement rate is invariant to which states you throw at them are measuring the same
**arm-level** construct. That invariance is a claim about arms; §4 is what it does *not* license.
*(Corrected 2026-08-25: the ladder read 88.3% / 94.1% / 97.0% / 98.9% over "**1,848** pairwise
arm×metric contrasts" — that count is the 22 K=0 states of the retired `L0` view, not the project
grid.)*

**Per rubric, the ladder re-detects the same weak instrument that dependability does.**
[`validity/tables/multijudge_sign_preservation_by_metric.md`](validity/tables/multijudge_sign_preservation_by_metric.md):
pooled, MITI is the worst of the eight at **79.8%** (MICI 82.9%, Q1 87.4%, WAI-SR 89.1%, Q2 89.6%,
CSQ-8 92.1%, MI-SAT 92.7%, PCT 93.9%). The sharper point is *where* MITI still fails: every other
rubric that has contrasts that large reaches **96.2–100%** by |Δ|≥0.25, MITI only **89.6%**, needing
|Δ|≥0.50 to reach 97.9%. **A MITI difference large enough to report can still flip sign under a
different grader** — an independent confirmation, from a completely different statistic, of the
`dependability_k1 = 0.624` warning in §6.
⚠ Ladder thresholds are **absolute**, so read a ladder *down its own rubric*, never across rubrics:
`PCT` is a 0–1 proportion with **zero** contrasts at |Δ|≥0.25, so its ladder stops at the ≥0.10 rung,
while `MICI_Rate` is an unbounded per-turn rate that does reach the upper rungs (99 contrasts at
≥0.25, 36 at ≥0.50). Only the pooled `all contrasts` row is cross-rubric comparable.
*(Corrected 2026-08-25: MITI's pooled rate read 77.5%, and "MITI only 88.2%" was attached to the
|Δ|≥0.25 rung — 88.2% is that rubric's `judge CI excludes 0` row, a different question. The ≥0.25
value is 89.6%.)*

**How much resolving power does a *single* conversation carry?**
[`validity/tables/multijudge_concordance_by_effect_size.md`](validity/tables/multijudge_concordance_by_effect_size.md)
and [`validity/figures/multijudge_concordance_curve.png`](validity/figures/multijudge_concordance_curve.png)
answer that per conversation **PAIR**: on Q1, cross-model concordance runs 0.484 at
|Δ| ∈ [0.1, 0.25), 0.607 at [0.25, 0.5), 0.719 at [0.5, 1), 0.895 at [1, 2) and 0.985 at ≥2.
⚠ **This is not a confidence in any arm-level claim.** A 0.5-point gap between two *conversations*
is close to a coin flip; a 0.5-point gap between two *arm means over 96 conversations* is the ≥0.50
rung of the ladder above, at 99.3%. The curve is the argument for n = 96, not a discount on the
results.

## 4 · Where the ruler breaks: cross-grader agreement collapses on GRPO_LA5's last iterations — on four of the eight rubrics, not only the rewarded pair

**This is the most important new finding in this top, and it is the one thing §3 does not cover.**
Everything in §3 is an **arm-level** statistic. Per conversation, the instrument has one severe
local failure — one arm's late states, on four of its eight rubrics — and it sits on the
experiment's best-scoring checkpoint.

Read sheet `second_judge_agreement` of
[`validity/tables/validity.xlsx`](validity/tables/validity.xlsx) (the `.md` excerpt does not reach
these rows). GRPO_LA5's Q1 agreement along its own trajectory:

| GRPO_LA5, Q1 | I5 | I6 | I7 | I8 | I9 | I10 |
|---|---|---|---|---|---|---|
| `pearson_r` | 0.941 | 0.877 | 0.842 | 0.769 | **0.487** | **0.544** |
| `spearman_rho` | 0.902 | 0.854 | 0.748 | 0.706 | 0.409 | 0.512 |

Against a Q1 column whose 44 rows have a **median of 0.855** and a range of 0.487–0.941, **I9 and
I10 are the two lowest-agreeing states of the entire 44-state grid on Q1** — I9 *is* the column
minimum. The decline from I5 is monotonic apart from the final tick.

**It is NOT confined to the rewarded rubrics — it splits the eight instruments into two groups of
four, and "was it the reward?" is not what carves them.** At `GRPOExp3_LA5_I10` **all eight** rubrics
sit below their own column median, but by magnitudes an order of magnitude apart. Four barely move:
MI-SAT **0.906** (column median 0.930), WAI-SR **0.898** (0.922), PCT **0.928** (0.954), CSQ-8
**0.851** (0.903) — drops of `0.930 - 0.906 = 0.024` to `0.903 - 0.851 = 0.052`. Four fall away:
Q2 **0.590** (median 0.784), MICI **0.287** (0.518), Q1 **0.544** (0.855) and **MITI 0.333** (0.658)
— drops of `0.784 - 0.590 = 0.194` to `0.658 - 0.333 = 0.325`.

Only two of those four were the training reward. **MITI never was**, and it is the *largest* drop of
the eight: 0.333 is the **minimum of the entire 44-state MITI column**, reached on exactly this
cell, at the end of a near-monotone decline along this arm's own trajectory (base 0.834 → I5 0.733
→ I8 0.484 → I9 0.398 → I10 0.333) that tracks Q1 and Q2 step for step (Q2 declines across I4–I10;
Q1 across I5–I10, table above). MICI is the weakest read of the four — it is noisy across the whole
grid (r 0.198–0.675), its own base state already reads 0.514, and it is 4th-lowest rather than
extremal — so treat MICI as consistent with the pattern, not as evidence for it. Grid-wide these
four are also the four rubrics with the *lowest* median agreement to begin with (Q1 0.855, Q2 0.784,
MITI 0.658, MICI 0.518, against 0.903–0.954 for the other four): the collapse lands on the already-weaker
half of the instrument set. **No verified mechanism explains that exact membership** — it is not
rewarded-vs-unrewarded (MITI and MICI were never rewarded), and it is not rater perspective either
(Q1/Q2 are patient-perspective rubrics like WAI-SR/CSQ-8/MI-SAT, and PCT is an MI-coder rubric like
MITI/MICI — [`METRICS_REFERENCE.md`](../METRICS_REFERENCE.md) §1). State the membership; do not name
a cause.
*(Corrected 2026-08-25: this paragraph asserted "**It is confined to the rewarded rubrics**" and
dismissed MITI and MICI as instruments that "are weak everywhere … so they add little". That is
retracted. MITI's 0.333 at this cell is the 44-state column **minimum** for that rubric and MITI was
never part of the training reward, so the collapse cannot be attributed to the reward. This
section's heading said "on the rewarded rubric only" for the same reason and has been changed. What
survives is the *contrast* the old sentence was reaching for: Q1/Q2 fall by 0.19–0.31 where CSQ-8 /
MI-SAT / WAI-SR / PCT fall by 0.02–0.05.)*

### The mechanism: one-sided saturation of the *training* grader, not homogenised outputs

The tempting reading — "the arm's outputs became uniform, so there is nothing left to correlate" —
is wrong, and the same-conversation dispersion trajectories refute it. From
[`../lookahead/replication/tables/sd_by_iter.md`](../lookahead/replication/tables/sd_by_iter.md),
GRPO_LA5 on Q1, base → I10, **on the identical 96 conversations at each state**:

- **primary (the training oracle): SD 1.336 → 0.701, and it falls MONOTONICALLY** (Spearman
  SD-vs-iteration −0.86, p = .001). Variance ratio `0.701^2 / 1.336^2 = 0.275` — roughly three
  quarters of its spread is gone, and the ratio survives re-anchoring (0.285 against iteration 1,
  0.544 against the mean of iterations 1–10).
- **held-out judge: SD does NOT move.** Spearman +0.44, p = 0.18; the series wanders between 0.763
  and 1.001 with no trend.

> ⚠ **(Corrected 2026-08-25, second pass.)** This read "held-out judge: SD 0.763 → 0.906, variance
> ratio `0.906^2 / 0.763^2 = 1.410` — it *gained* spread". **Iteration 0 is that series' minimum**,
> so the ratio is an anchoring artifact: against iteration 1 it is `0.906^2 / 0.879^2 = 1.062` and
> against the mean of iterations 1–10, 1.034 — with a null trend test. The held-out spread is
> **flat**. Do not reinstate the growth claim; the inference does not need it (below).

Had the policy homogenised its outputs, **both** graders would compress. Only the grader that was
optimized against does — that asymmetry is the whole argument, and "one falls, one is flat"
carries it exactly as well as "one falls, one rises" would have. The primary's Q1 mean rises 2.910 → 4.465 on a 1–5 scale, with 84.4% of
sessions at ≥4, 58.3% at ≥4.5 and 39.6% at exactly 5.0 — the highest saturation of any arm — while
the held-out grader still places only 7.3% of those same sessions at ≥4.0. ⚠ **That is not a level
comparison across graders** (forbidden — see §6): it is each grader's own dispersion and own
ceiling behaviour, measured separately on one shared set of conversations. The composite view is
[`../lookahead/replication/tables/ceiling.md`](../lookahead/replication/tables/ceiling.md), where
GRPO_LA5's endpoint Q1+Q2 `share_ge45_all = 0.594` and its `Cooperative` persona third reads
`mean 4.939, sd 0.047, share_ge45 1.000` — a grader with no headroom left on a third of the sample.

**The attenuation arithmetic closes the loop.** The per-conversation decomposition
([`validity/tables/multijudge_variance_components_per_conversation.md`](validity/tables/multijudge_variance_components_per_conversation.md))
splits each cell into shared conversation variance, judge level, and residual per-conversation
disagreement. On GRPO_LA5 / Q1, `var_resid` barely moves — 0.0614 at I5, 0.3035 at I9, 0.3106 at
I10, the last two indistinguishable from the *base* states (`GRPOExp3_LA5_Base` 0.2909,
`PTOExp3_LA0_Base` 0.3040). What collapses is `var_conversation`: 0.9686 → 0.2815 → 0.3451. The
implied cross-grader correlation `var_conversation / (var_conversation + var_resid)` reproduces the
observed r almost exactly:

- I5: `0.9686 / (0.9686 + 0.0614) = 0.940` — observed r 0.941
- I9: `0.2815 / (0.2815 + 0.3035) = 0.481` — observed r 0.487
- I10: `0.3451 / (0.3451 + 0.3106) = 0.526` — observed r 0.544

**The graders do not disagree more at I10 than they did at the base. There is simply almost nothing
left for them to agree about.** The shared signal was squeezed out of Q1.

**MITI shows the same per-conversation signature — and it was never rewarded.** In the same
decomposition, GRPO_LA5 / MITI runs `var_conversation` 0.5113 at base → 0.1072 at I10 while
`var_resid` moves only 0.1700 → 0.2204, so the implied correlation
`0.1072 / (0.1072 + 0.2204) = 0.327` again reproduces the observed r of 0.333. ⚠ **Only the
attenuation half of the mechanism is demonstrated for MITI.** `sd_by_iter.md` carries Q1, Q2 and
Q1+Q2 and nothing else, so there is no per-grader dispersion trajectory on MITI to test the
one-sided-saturation half against — do not assert it there.

### What this costs, stated sharply

- **Sign preservation is an ARM-LEVEL statistic and does not license a per-conversation claim.**
  §3's 88.4% pooled / 99.3% at |Δ|≥0.50 remains true, and remains the right defence of every
  *arm-mean* contrast in the thesis. It says nothing about whether the two graders rank two
  *conversations* from `GRPOExp3_LA5_I10` the same way — and on Q1 there, they largely do not.
- **The experiment's best-scoring checkpoint is also among its worst-agreeing — on Q1 *and* on
  MITI.** `GRPOExp3_LA5_I10` is the highest Q1+Q2 endpoint on the primary grader in
  [`../lookahead/replication/tables/ceiling.md`](../lookahead/replication/tables/ceiling.md) (4.517,
  above PTO_LA5 4.307, PTO_LA0 4.260, GRPO_LA0 3.753) and simultaneously the second-lowest Q1
  agreement cell of 44 and the **lowest MITI** agreement cell of 44. Any sentence of the form "GRPO
  at K=5 reaches the highest Q1+Q2" should carry this in the same paragraph.
  *(Corrected 2026-08-25: this bullet said "its worst-agreeing one on the rewarded rubric", which
  both overstated Q1 — I9, not I10, is the Q1 column minimum — and missed that the cell IS the
  column minimum on MITI, a rubric the reward never touched.)*
- **It cannot be decomposed, because §2's gap bites exactly here.** No K=5 state has a repeatability
  rep on either grader, so `ceiling_basis` reads `"no ICC measured for this cell"` for every row in
  the table above. We therefore **cannot** say how much of r = 0.487 is genuine construct
  disagreement and how much is either grader's own noise on a saturated, low-variance cell. It is
  worse for MITI, which has **no measured ICC on any cell of the grid** — the anchor subset is
  {Q1, Q2, MICI} only (§2). Three seeded reps on `GRPOExp3_LA5_{I9,I10}` × {Q1, Q2, MITI} would
  settle it, and it is the cheapest open measurement question in the project.
- **It does not, by itself, retract anything.** No contrast involving these states flips at a
  claimable effect size and the arm-mean ordering is intact. What it retracts is *confidence in Q1,
  Q2, MITI and MICI as per-conversation instruments on this arm's late states* — the states where
  the primary grader has run out of headroom on Q1/Q2.

## 5 · Gain retention: whose gains survive a grader that never graded during training?

See [`validity/figures/multijudge_gain_retention.png`](validity/figures/multijudge_gain_retention.png),
[`validity/figures/multijudge_retention_trajectory.png`](validity/figures/multijudge_retention_trajectory.png)
and the one-column
[`validity/figures/multijudge_retention_trajectory_Q1.png`](validity/figures/multijudge_retention_trajectory_Q1.png).

`retention = Δ(held-out) / Δ(trained-against)` is a **train/test generalization ratio** per model
state: ~1.0 = a behaviour change both graders see; ~0 = a gain that existed only in the optimized
grader. Persona-paired, persona-bootstrap CI, suppressed to `nan` where |Δ primary| < 0.15.
[`validity/tables/multijudge_gain_retention.md`](validity/tables/multijudge_gain_retention.md).

**Q1 retention at the matched iteration-10 endpoint, all four arms:**

| arm | retention | CI |
|---|---|---|
| PTO_LA0 | 0.795 | [0.674, 0.928] |
| PTO_LA5 | 0.715 | [0.606, 0.836] |
| GRPO_LA5 | 0.686 | [0.578, 0.809] |
| **GRPO_LA0** | **0.284** | **[0.055, 0.428]** |

**The retention gap is a K=0 phenomenon.** At K=0 the PTO and GRPO intervals are disjoint
(0.674 > 0.428) — under a grader that never graded during training, GRPO_LA0's net 10-iteration Q1
gain is ≈0.19 points where the primary credits it ≈0.68. At K=5 the two methods land on top of each
other (0.715 vs 0.686, fully overlapping): **look-ahead largely repairs GRPO's retention.** ⚠ Never
write the retention verdict without naming K — the ordering it implies is not the same at K=0 and
K=5.

**And it is an *onset* curve, not an endpoint accident** (Q1, from the same table):

| iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| PTO_LA0 | 0.967 | 0.841 | 0.890 | 0.942 | 0.975 | 0.970 | 0.937 | 0.892 | 0.875 | **0.795** |
| PTO_LA5 | 1.046 | 0.687 | 0.813 | 0.879 | 0.819 | 0.818 | 0.940 | 0.843 | 0.828 | **0.715** |
| GRPO_LA5 | 0.808 | 0.681 | 0.962 | 0.970 | 1.068 | 0.959 | 0.932 | 0.794 | 0.705 | **0.686** |
| GRPO_LA0 | 1.131 | 0.786 | 0.893 | 0.793 | 0.727 | 0.570 | 0.701 | 0.644 | 0.034 | **0.284** |

GRPO_LA0 decays monotonically in trend from ~0.89 (I3) to 0.284, with I9 the visible floor at 0.034
— that is also the arm's global dip in the primary eval, so read it as the extreme of the trend
rather than a separate event. GRPO_LA5 holds ~0.96–1.07 through I3–I6 and then decays over I8–I10 —
**the same window and the same direction as §4's agreement collapse**. Two independent statistics
point at one process: from roughly iteration 7 the K=5 arm's gains are increasingly visible to the
grader it was trained against and less so to the one it was not.
*(Corrected 2026-08-25: the previous reading said "GRPO K=5 retains its full Q1 gain". That was read
at that arm's then-endpoint of iteration 5, where retention is 1.068. By iteration 10 it is 0.686
[0.578, 0.809] — not full, and trending down.)*

**Q2 retention is no longer a uniform, uninteresting band.** At iteration 10: GRPO_LA0 0.805
[0.686, 0.962], PTO_LA0 0.849 [0.745, 0.977], GRPO_LA5 0.689 [0.598, 0.796], **PTO_LA5 0.567
[0.485, 0.665]** — the last interval disjoint from PTO_LA0's.
*(Corrected 2026-08-25: this file said "every Q2 interval overlaps (0.80–0.85, i.e. uninteresting
scale compression)". True of the K=0 arms, **false of PTO at K=5**, whose Q2 gain is the
least-retained of the four.)*

**MITI retention is the lowest of every rubric, in every arm** — 0.275 (PTO_LA5) to 0.450 (PTO_LA0)
at iteration 10 — but this is the one rubric where retention is confounded with the instrument
itself: MITI's `dependability_k1` is 0.624 and 93.9% of its arm-mean variance is grader level, so a
low MITI retention cannot be cleanly attributed to the policy rather than to the ruler. Q1 carries
the retention claim; MITI corroborates at best.

⚠ **Two reading traps on this table.**
1. **Reference base.** Every row here is referenced to the *shared* `PTOExp3_LA0_Base` draw, so all
   four arms sit on one scale. The `lookahead/transfer` family's
   [`k_retention_summary.md`](../lookahead/transfer/tables/k_retention_summary.md) uses each arm's
   **own** base, so the same arm and iteration legitimately reads a different number there (GRPO
   K=0, Q1, iteration 10: 0.295 own-base vs 0.284 shared-base). They do not contradict; they answer
   slightly different questions. Name the reference whenever quoting either.
2. **Cost axis.** This whole table is indexed by **iteration**, and an iteration is not a unit of
   spend. Whole-arm GPU-hours are PTO_LA0 8.119, PTO_LA5 19.681, GRPO_LA0 27.906, GRPO_LA5 51.205
   ([`../compute/cost/tables/compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md)) — so
   `51.205 / 27.906 = 1.835` and `19.681 / 8.119 = 2.424`: the K=5 arms above bought their retention
   at roughly twice the compute of their K=0 siblings. **A matched-*budget* reading of retention
   would pair different iterations across arms and is not rendered anywhere**, so do not read the
   iteration-10 row as a statement about equal spend. For any lever whose verdict depends on budget,
   quote the sweep, never one row —
   [`../compute/cost/tables/budget_sweep_crossjudge_verdicts.md`](../compute/cost/tables/budget_sweep_crossjudge_verdicts.md).

## 6 · Standing caveats — the ones that constrain how every other summary may be written

- ⚠ **Never average the two graders' raw scores.** The primary *was* the training reward and the
  held-out grader never touched training: averaging them averages train and test. It is also unsound
  numerically — the level offset is large **and model-dependent** (on Q1 it runs from −1.177 at
  `GRPOExp3_LA5_I5` to −1.835 at `GRPOExp3_LA0_I9`, column `bias_judge_minus_primary`), so a mean
  lands on neither grader's rubric anchors and applies a silent, model-dependent shrinkage to every
  effect. Combine **contrasts** or **standardized** quantities only. For the same reason **absolute
  scores are never comparable across graders**, and (separately) never comparable to Exp2, which
  generated its conversations in 4-bit rather than bf16.
- ⚠ **Pair on `persona_id`, never `file_index`.** The trainer reshuffles the 96 personas every
  iteration, so a `file_index` join across unmatched iterations pairs unrelated conversations.
  `all_pairs_contrasts` and `gain_retention` are persona-paired.
  [`second_judge_contrasts.md`](validity/tables/second_judge_contrasts.md) is `file_index`-paired —
  safe as published, because a paired *mean* delta equals the difference of means either way and
  that table reports no `dz` and no CI — but do not extend it with either.
- ⚠ **MITI is the least dependable instrument.** `dependability_k1 = 0.624` (both graders averaged:
  0.768); only 3.8% of its arm-mean variance is between-state signal against 93.9% grader level; and
  it preserves its sign on 79.8% of contrasts — worst on both statistics, which agree from
  completely different directions. §4 adds a third, independent flag from a third direction: MITI's
  **lowest** per-conversation agreement cell in the whole 44-state grid (r 0.333) is
  `GRPOExp3_LA5_I10`, the highest-scoring checkpoint in the experiment. **Treat MITI differences as
  provisional unless both graders agree on the direction**, including the "neither arm reaches
  *good* on the technique ratios" verdict in [`../arms/SUMMARY.md`](../arms/SUMMARY.md).
  *(Corrected 2026-08-25: this file carried MITI `dependability_k1` as 0.65 in one place and 0.553
  in another, arm share 3.6%, judge share 94.5%, sign preservation 77.5% — all read on the censored
  grid. It also carried MICI `dependability_k1 = 0.628`; on the 44-state grid MICI is **0.812**.)*
- ⚠ **MICI cross-grader agreement is weak and mostly *construct* disagreement.** Per-conversation r
  runs 0.198–0.675 across the 44 states (ρ 0.214–0.681). Part of that is the held-out grader's own
  MICI noise — its ICC falls as the MI-inconsistency rate rises, to 0.525 on the highest-MICI anchor
  state — but measured against the corrected ceiling MICI still recovers only ~29–59% of achievable
  agreement where Q1/Q2 recover ~85%. **Every MI-consistency / reward-hacking claim must name its
  axis (per therapist turn, per session, or share of coded acts) and its grader**, and should lean
  on the judge-free deterministic text channels
  ([`../lookahead/behaviour/tables/k_channels_text_grpo.md`](../lookahead/behaviour/tables/k_channels_text_grpo.md)
  and [`..._pto.md`](../lookahead/behaviour/tables/k_channels_text_pto.md) — literal `?` per
  therapist turn, characters per turn, verbatim-loop share, all computed from transcripts with no
  grader involved) rather than on a per-conversation MICI rate. Full treatment:
  [`LIMITATIONS.md`](../LIMITATIONS.md) §2.
- ⚠ **Never state a PTO-vs-GRPO verdict without naming K.** The sign flips: on Q1+Q2 at the matched
  iteration-10 endpoint, PTO − GRPO is **+0.507** (dz 0.729) at K=0 and **−0.210** (dz −0.356) at
  K=5 under the primary, **+0.609** (dz 1.265) / **−0.206** (dz −0.313) under the held-out grader —
  **both graders flip together**
  ([`../method/contrast/tables/method_paired_by_K.md`](../method/contrast/tables/method_paired_by_K.md)).
  That they flip together is a *measurement* result and belongs to this top: the interaction is not
  an artifact of the patient and the grader sharing a model. Whether it survives a matched **budget**
  is a different question — [`../method/SUMMARY.md`](../method/SUMMARY.md) and
  [`../compute/`](../compute/).
- **Every endpoint is a single 96-conversation draw**, and therapist decoding is unseeded, so no
  conversation set is reproducible. The only measured noise floor is at the base.
- **All 96 personas are used for both training and eval**, so everything is in-sample with respect
  to the patient distribution.
- **There is still exactly one alternative grader, and no human MI/MITI-coder validation.** The
  second judge decouples the **grader**, not the **generator**: `gpt-4o-mini` plays the patient in
  every conversation of every arm. A fully decoupled replication would mean regenerating every
  conversation, not just re-scoring it. An oracle can be perfectly repeatable and consistently wrong.

---

**Net.** The instrument is trustworthy for what the thesis actually claims — arm-mean contrasts at
reportable effect sizes, on rubrics other than MITI — and completing the grid strengthened rather
than weakened that: the sign-preservation rate did not budge as `44 − 40 = 4` more model states and
`7,568 − 6,240 = 1,328` more contrasts arrived. What completing the grid *added* is a bounded,
locatable failure: on four of the eight rubrics — the two rewarded ones **plus MITI and MICI**, which
were never rewarded — at the top of the primary grader's range, the two graders stop agreeing per
conversation, and the state where that is worst is the state with the highest score.

---

**Artifact note (2026-08-26).** §3b of the validity notebook now recomputes the full-grid
statistics on the 22 GRPO states for the GRPO-scoped paper: `multijudge_sign_preservation_grpo`
(1,640/1,848 = 88.7% pooled; 97.0% at |Δ|≥0.25) and `judge_saturation_grpo` (+`_data`; Q1 median
0.8415 over 22 states — printed 0.841 in the figure, displayed 0.842 in the md — with per-rubric
medians and ranks /22 in its panel-c rows). Recomputed, not re-scoped: the 44-state artifacts are
unchanged and remain the four-arm paper's sources.
