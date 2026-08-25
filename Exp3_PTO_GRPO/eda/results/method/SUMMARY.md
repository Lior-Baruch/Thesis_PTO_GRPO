# Exp3 EDA Summary — `method/` (RQ-ii: PTO vs GRPO at each K)

*Rewritten 2026-08-25 on the complete four-arm grid (4 arms × 11 states, both graders). The
previous narrative — ported from `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` on 2026-08-18 —
was written while `GRPO_LA5` was right-censored and is superseded; the claims it got **wrong**
(not merely out of date) are flagged inline as `(Corrected 2026-08-25: …)` so a future reader knows
they were retracted rather than quietly updated.*

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
matched training oracle (Q1+Q2), matched candidate count (M = G = 8), matched iteration count
(all four arms reach **iteration 10**), persona-paired on the same 96 personas. Four artifacts:

| artifact | what it answers |
|---|---|
| [`contrast/tables/headline_grid.md`](contrast/tables/headline_grid.md) + [`figures/headline_grid.png`](contrast/figures/headline_grid.png) | **where all four arms land** at their endpoint, on every rubric, under both graders, each arm anchored to **its own** base, with bootstrap CIs |
| [`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md) | the PTO − GRPO gap at **every** iteration × rubric × K, both graders — the main table |
| [`contrast/tables/method_paired_best.md`](contrast/tables/method_paired_best.md) | the **model-selection steelman**: each arm at its own peak iteration *under the grader that scores it* |
| [`contrast/figures/method_gap.png`](contrast/figures/method_gap.png) | the same gap as a trajectory, one panel per grader, K=0 vs K=5 overlaid |

The number ledger behind the prose is
[`contrast/tables/method_contrast.json`](contrast/tables/method_contrast.json).

Everything below names its **cost axis**. Sections 1–6 are matched-**ITERATION**; section 7 is
matched-**BUDGET** (GPU-hours), which lives in [`../compute/`](../compute/SUMMARY.md). The two
axes give different winners at K=5, and that disagreement is itself a finding.

The per-arm curves this contrast is read against are [`../arms/SUMMARY.md`](../arms/SUMMARY.md);
how look-ahead *changes* the winner — the same interaction, read as an RQ-i question — is
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md).

---

## 1. Headline — the PTO-vs-GRPO verdict is an INTERACTION with K

**Never state a PTO-vs-GRPO verdict without naming K. The sign flips.** At the matched
iteration-10 endpoint on Q1+Q2, persona-paired over the same 96 personas, sign `+` = PTO higher
([`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md)):

| K | primary (gpt-4o-mini, the training oracle) | held-out (claude-haiku-4-5) | verdict |
|---|---|---|---|
| **0** | **+0.507** (dz 0.729, p_holm < .001) | **+0.609** (dz 1.265, p_holm < .001) | **PTO wins** |
| **5** | **−0.210** (dz −0.356, p_holm .001) | **−0.206** (dz −0.313, p_holm .034) | **GRPO wins** |

Both graders agree on both signs, and both cells clear Holm within their rubric family. This is
not a near-tie being read two ways: the K=0 gap is *medium-to-large* and the K=5 gap is *small but
significant on both graders*, in the opposite direction.

The formal test of "the sign flips" is the difference-in-differences in
[`../lookahead/reward/tables/k_did.md`](../lookahead/reward/tables/k_did.md): at iteration 10 on
Q1+Q2, `did_mean` = **0.718** (dz 0.793, p_holm ≈ 0) on the primary grader and **0.815**
(dz 0.972, p_holm ≈ 0) on the held-out judge — i.e. the method gap is ~0.7–0.8 Q1+Q2 points
*larger at K=0 than at K=5*, on both rulers. The interaction is present from iteration 6 onward on
both graders (and from iteration 4 on the held-out judge) and is the single most robust
cross-arm structure in the experiment.

The noise floor for the same statistic is the iteration-0 row of that table — two independent
96-conversation draws of the *same* untrained model: gap = −0.066 (K=0) and +0.040 (K=5) on the
primary grader, −0.031 and −0.001 on the held-out judge. The endpoint gaps are an order of
magnitude above it.

> **(Corrected 2026-08-25: the previous headline read "PTO ahead at the matched 10-iter endpoint"
> and "PTO sustains gains while GRPO overshoots and degrades", stated unconditionally. That is
> only the K=0 half. It was written when `GRPO_LA5` stopped at iteration 5, so the K=5 half of the
> grid could not be read at a matched endpoint. Stated without naming K it is now false.)**

## 2. Where the four arms actually land (matched iteration)

[`contrast/tables/headline_grid.md`](contrast/tables/headline_grid.md) is the artifact this
section exists for — it was added on 2026-08-25 because no single table showed the four arms
together. Q1+Q2 at iteration 10, each arm against **its own** base (iteration 0 of that same arm),
persona-paired, `delta` = endpoint − own base:

| arm | GPU-h | primary base → end (Δ, dz) | held-out base → end (Δ, dz) |
|---|---|---|---|
| `GRPO_LA5` | 51.205 | 2.963 → **4.517** (1.554, dz 1.518) | 1.835 → **2.873** (1.038, dz 1.539) |
| `PTO_LA5`  | 19.681 | 3.003 → **4.307** (1.304, dz 1.353) | 1.834 → **2.667** (0.833, dz 1.124) |
| `PTO_LA0`  | 8.119  | 3.000 → **4.260** (1.259, dz 1.429) | 1.830 → **2.866** (1.036, dz 1.653) |
| `GRPO_LA0` | 27.906 | 3.067 → **3.753** (0.686, dz 0.721) | 1.861 → **2.257** (0.396, dz 0.658) |

(GPU-h from [`../compute/cost/tables/compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md);
every Q1+Q2 Δ above has a bootstrap CI excluding zero and p_holm ≈ 0 in the grid table.)

Three readings, all worth keeping:

- **On the primary grader the ordering is unambiguous** and spans
  4.517 − 3.753 = 0.764 Q1+Q2 points between the best and worst arm.
- **On the held-out judge the top is a tie**: `GRPO_LA5` 2.873 vs `PTO_LA0` 2.866, i.e.
  2.873 − 2.866 = 0.007 — while the primary grader separates the same two by
  4.517 − 4.260 = 0.257. ⚠ Those two differences are arithmetic on two cells of one table; they
  are **not** persona-paired contrasts and carry no CI. `GRPO_LA5` vs `PTO_LA0` matches neither K
  nor method, so it is a description of where the arms landed, never a controlled comparison. The
  only controlled cross-method contrasts in this top are the matched-K ones in §1.
- ⚠ **Never compare a level across the two grader blocks.** The held-out judge's offset is
  1.2–1.7 points and model-dependent; only Δ and dz travel between graders.

The rubric-by-rubric version of the same endpoint (including the free MITI ratios) is
[`../arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md`](../arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md).

## 3. K=0 — PTO climbs to the end; GRPO peaks at iteration 8 and stays down

Per-iteration Q1+Q2 means ([`../lookahead/reward/tables/k_means_by_iter.md`](../lookahead/reward/tables/k_means_by_iter.md),
primary grader): `PTO_LA0` runs 3.000 → 3.815 (it 3) → 4.154 (it 6) → 4.221 (it 8) → **4.260**
(it 10), peak = final. `GRPO_LA0` runs 3.067 → 3.993 (it 3) → 4.082 (it 8, its peak) → 3.808
(it 9) → **3.753** (it 10). The OLS Q1+Q2 slope is 0.120/iter for `PTO_LA0` (peak_iter 10) vs
0.072/iter for `GRPO_LA0` (peak_iter 8) —
[`../arms/stats/tables/gpt-4o-mini/slope_by_arm.md`](../arms/stats/tables/gpt-4o-mini/slope_by_arm.md).

The gap therefore grows late. On the primary grader it is small and inconsistent through
iterations 1–7 — briefly *negative* at iteration 3 (−0.179, dz −0.328, p_holm .014) and otherwise
indistinguishable from zero (|dz| ≤ 0.33, p_holm mostly 1.000) — then turns positive and grows:
+0.138 at 8, +0.431 at 9, +0.507 at 10. On the held-out judge it is significantly positive from
iteration 5 onward (+0.265, dz 0.355, p_holm .014) and reaches +0.919 at iteration 9 — the K=0 rows of
[`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md) and the black line
of [`contrast/figures/method_gap.png`](contrast/figures/method_gap.png).

**The regression is sustained, not a dip.**
[`../arms/validity/tables/gpt-4o-mini/grpo_iter9_check.md`](../arms/validity/tables/gpt-4o-mini/grpo_iter9_check.md)
(primary grader, persona-paired): Q1+Q2 it9 − it8 = −0.275 (dz −0.406, p_holm < .001), it10 − it9 =
−0.055 (dz −0.071, **ns**), it10 − it8 = −0.330 (dz −0.426, p_holm < .001). MITI behaves the same
way (−0.297 / −0.016 ns / −0.312). Only WAI-SR recovers (it10 − it9 = +0.350, dz 0.656).

> **(Corrected 2026-08-25: the previous text called iteration 9 "a paired one-iteration dip …
> then partially recovers at 10". On Q1+Q2 and MITI there is no recovery — the it10 − it9 step is
> null and the arm ends 0.330 Q1+Q2 points below its own iteration-8 peak. "Partial recovery" was
> true only of WAI-SR.)**

**The held-out judge tells a harsher version of the same story.** Under `claude-haiku-4-5`,
`GRPO_LA0`'s Q1+Q2 peaks at **iteration 3** (2.637) and ends at 2.257 — below where it was seven
iterations earlier — which is why the steelman table picks iteration 3 for it (§5). And every one
of the 8 rubrics in the iteration-10 K=0 contrast keeps its sign under the held-out judge
([`../measurement/validity/tables/second_judge_contrasts.md`](../measurement/validity/tables/second_judge_contrasts.md),
8/8 `same_sign` True), with the held-out judge *widening* the Q1 gap: +0.773 vs the primary's
+0.533.

## 4. K=5 — GRPO overtakes from iteration 3 and never gives it back

The K=5 rows of [`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md)
are negative (GRPO higher) at **every** iteration from 3 to 10 on both graders. Primary Q1+Q2:
−0.188 (it 3), −0.232 (it 4, dz −0.351, p_holm .024), −0.025 (it 5, ns), −0.332 (it 6, dz −0.437),
−0.185, −0.111, −0.257, **−0.210** (it 10). Held-out Q1+Q2: −0.236, −0.226, −0.219, −0.397
(it 6, dz −0.599), −0.177, −0.066, −0.125, **−0.206** (it 10). MITI is negative at every K=5
iteration under both graders and carries the endpoint's largest primary-grader gap (−0.279; −0.227
held-out, both p_holm ≤ .007).

Per-iteration levels: `GRPO_LA5` climbs 2.963 → 4.120 (it 4) → 4.270 (it 7) → 4.455 (it 9) →
**4.517** (it 10) on the primary grader, OLS slope 0.144/iter with peak_iter 10 — the steepest and
latest-peaking arm in the grid ([`slope_by_arm.md`](../arms/stats/tables/gpt-4o-mini/slope_by_arm.md)).
`PTO_LA5` climbs to 4.307, slope 0.127/iter, also peaking at 10. **Neither K=5 arm shows the
post-peak regression that defines `GRPO_LA0`** — which is the cleanest way to see that "GRPO
overshoots" was never a property of GRPO, only of GRPO *without look-ahead*.

> **(Corrected 2026-08-25: the previous §3 read the K=5 flip at iteration 5 and flagged
> "GRPO_LA5 runs to iteration 5, not 10" as a scope limit that handed GRPO ~2× the compute per
> cell. `GRPO_LA5` finished at iteration 10 for 51.205 GPU-h; the K=5 contrast now runs the full
> length and `method_contrast.json`'s `meta.censoring` key reads "no arm is censored in this
> frame". The old iteration-4/5 numbers were right for their iteration and are simply no longer
> the endpoint.)**

## 5. Model selection does not rescue GRPO at K=0 — and the graders disagree about where its peak is

[`contrast/tables/method_paired_best.md`](contrast/tables/method_paired_best.md) credits each arm
at its own best iteration **under the grader that then scores it** (`iter_a` = PTO, `iter_b` =
GRPO):

| grader | K | PTO @ | GRPO @ | Q1+Q2 gap | dz | p_holm |
|---|---|---|---|---|---|---|
| primary | 0 | 10 | **8** | +0.177 | 0.296 | .010 |
| primary | 5 | 10 | 10 | −0.210 | −0.356 | .001 |
| held-out | 0 | 9 | **3** | +0.284 | 0.568 | < .001 |
| held-out | 5 | 7 | 7 | −0.177 | −0.270 | **.107 (ns)** |

Three things to take from it:

- **At K=0 the steelman shrinks PTO's win but does not erase it** (+0.507 → +0.177 primary;
  +0.609 → +0.284 held-out), and it stays significant on both graders. Early stopping is worth
  0.507 − 0.177 = 0.330 Q1+Q2 points to GRPO at K=0 (the same size as the it10 − it8 drop
  in §3) and little to PTO, whose best iteration is its last on the primary grader and iteration 9
  on the held-out judge.
- **At K=5 both arms peak at the same iteration under each grader**, so best-vs-best is the same
  contrast as matched-endpoint on the primary grader (10 vs 10, identical rows) and a 7-vs-7
  contrast on the held-out judge.
- ⚠ **The one cell where GRPO's K=5 win does not clear Holm is the held-out best-vs-best**
  (p_holm .107). State the K=5 verdict as "GRPO higher at the matched endpoint on both graders,
  and at best-vs-best on the primary grader; the held-out best-vs-best cell is directionally the
  same but not significant" — never as "GRPO wins at K=5 under every selection rule".

## 6. MI-inconsistency — at K=0 PTO wins the reward *and* the hack; at K=5 the two methods are level

`MICI` = **MI-inconsistent therapist acts per therapist turn** (`MICI_Rate`), oracle-coded by the
named grader, **lower = better** ([`../METRICS_REFERENCE.md`](../METRICS_REFERENCE.md) §1). In
`method_paired_by_K` a *positive* MICI delta means PTO is worse; no sign is flipped.

At iteration 10 ([`contrast/tables/method_paired_by_K.md`](contrast/tables/method_paired_by_K.md)):

| K | MICI gap, primary | MICI gap, held-out |
|---|---|---|
| 0 | **−0.346** (dz −0.989, p_holm < .001) → PTO **better** | **−0.225** (dz −0.667, p_holm < .001) → PTO **better** |
| 5 | +0.053 (dz 0.224, p_holm .180, **ns**) | −0.047 (dz −0.140, p_holm .286, **ns**) |

The endpoint levels behind it ([`headline_grid.md`](contrast/tables/headline_grid.md), acts per
therapist turn): primary grader `GRPO_LA0` 0.211 → **0.838** vs `PTO_LA0` 0.213 → **0.491**
(0.838 / 0.491 = 1.707× as many); held-out `GRPO_LA0` 0.384 → **1.050** vs `PTO_LA0` 0.364 →
**0.825** (1.050 / 0.825 = 1.273×). Both K=5 arms sit far below both: `PTO_LA5` 0.264 and
`GRPO_LA5` 0.210 on the primary grader, the latter statistically **indistinguishable from its own
base** (delta 0.002, dz 0.006, p .711).

**A judge-free cross-check agrees.** The deterministic lexical over-praise marker — the share of
therapist turns containing an effusive-praise regex hit, computed off the transcripts with no
oracle involved — at iteration 10 reads `GRPO_LA0` **0.671**, `PTO_LA0` **0.210**, `PTO_LA5`
0.045, `GRPO_LA5` 0.064
([`../arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md`](../arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md);
that file sits under a `<judge>/` leaf only because the oracle's `MICI_OverPraiseRate` column
beside it is grader-specific). GRPO's K=0 endpoint over-praises on
0.671 / 0.210 = 3.195× as large a share of its turns as PTO's, and both K=5 arms are an order of
magnitude below both K=0 arms. ⚠ `behavior.py` flags this regex as brittle and explicitly **not**
a primary behaviour metric — use it to confirm the *direction* of the oracle-coded MICI, never as
the effect size.

> **(Corrected 2026-08-25 — this box's own first version, written earlier the same day, was wrong
> twice and is retracted. (a) It claimed the previous file's *only* MI-inconsistency statement
> about the two methods was budget-matched. It was not: that file's §2 (K=0, primary grader,
> matched **iteration** 10) already read "a measurable reward-hack in both arms, worse in GRPO
> (MICI base 0.21 → 0.49 PTO / 0.84 GRPO at iter 10)" — the same levels as the primary block above.
> Nothing about the matched-iteration direction was retracted; what that statement lacked was a
> named grader, and it is restated per-grader here. (b) It then claimed "PTO ends with roughly half
> of GRPO's MI-inconsistency rate on both graders". "Roughly half" is a **primary-grader** figure
> only — 0.491 / 0.838 = 0.586 there, against 0.825 / 1.050 = 0.786 on the held-out judge, where
> PTO's rate is about four fifths of GRPO's, not half. What holds on **both** graders is the sign
> and the significance of the paired K=0 contrast (−0.346 primary, −0.225 held-out, both
> p_holm < .001), and its size in dz (−0.989 / −0.667); the ratio of levels is a within-grader
> quantity and must be quoted with its grader. The budget-axis statement in §7 — "PTO buys more
> reward per GPU-hour, and more reward-hacking per GPU-hour with it" — is a separate claim, still
> true, and is about training **depth** rather than about the method.)**

## 7. The other axis — at matched BUDGET, PTO wins at BOTH K levels

Every contrast above is indexed by **iteration**, which is not a fixed unit of spend.
Reconstructed from artifact mtimes
([`../compute/cost/tables/compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md); the
compute axis is owned by [`../compute/`](../compute/SUMMARY.md)):

| arm | iters | gen h | build h | train h | **total GPU-h** | h / iter |
|---|---|---|---|---|---|---|
| `PTO_LA0`  | 10 | 1.323 | 5.669 | 1.127 | **8.119** | 0.812 |
| `PTO_LA5`  | 10 | 1.370 | 16.797 | 1.514 | **19.681** | 1.968 |
| `GRPO_LA0` | 10 | 1.214 | 0.000 | 26.692 | **27.906** | 2.791 |
| `GRPO_LA5` | 10 | 0.915 | 0.000 | 50.290 | **51.205** | 5.120 |

- **At K=0, PTO reaches the same iteration 10 for 27.906 / 8.119 = 3.437× less GPU time** — and
  scores higher. On this axis PTO dominates GRPO at K=0 outright.
- **At K=5, PTO reaches iteration 10 for 51.205 / 19.681 = 2.602× less GPU time** — and scores
  0.210 Q1+Q2 points *lower*. That is the whole K=5 trade in one line.
- Across both K levels GRPO spent 27.906 + 51.205 = 79.111 GPU-h against PTO's
  8.119 + 19.681 = 27.800, i.e. 79.111 / 27.800 = 2.846× more GPU time.
- The reason is structural, not incidental. PTO's preference-tree **build** runs once per
  iteration and dominates its cost (5.669 / 8.119 = 0.698 of `PTO_LA0`;
  16.797 / 19.681 = 0.853 of `PTO_LA5`), whereas GRPO has no build phase at all and recomputes its
  reward *inside* the training loop (train_share 0.957 and 0.982). Look-ahead therefore lands in
  different phases: GRPO pays ~1.9× per optimizer step (median step-second ratios 1.965 / 1.962 /
  1.911 at iterations 3 / 4 / 5,
  [`../compute/cost/tables/step_multiplier.md`](../compute/cost/tables/step_multiplier.md)),
  while PTO's DPO step is unchanged (ratio ~1.0) and the cost appears in `build_h`.
- The **API** axis moves the same way and is judge-invariant. Summed over iterations 1–10,
  `total_api_calls` is 117,209 (PTO K=0) vs 316,795 (GRPO K=0) — 316,795 / 117,209 = 2.703× — and
  301,440 (PTO K=5) vs 696,548 (GRPO K=5) — 696,548 / 301,440 = 2.311×
  ([`../compute/cost/tables/api_ratio.md`](../compute/cost/tables/api_ratio.md)).

**Quote the sweep, not one iso-compute row** — the lever's sign is a function of budget. Across the
whole of `PTO_LA0`'s budget range PTO leads at every step
([`../compute/cost/tables/budget_sweep_method_K0_gpt-4o-mini.md`](../compute/cost/tables/budget_sweep_method_K0_gpt-4o-mini.md)
and [`_claude-haiku-4-5`](../compute/cost/tables/budget_sweep_method_K0_claude-haiku-4-5.md),
Q1Q2→Q1Q2 rows, `+` = PTO higher):

| budget (GPU-h) | primary Δ (dz) | held-out Δ (dz) |
|---|---|---|
| 2.800 | +0.546 (0.544) | +0.406 (0.576) |
| 5.370 | +0.795 (0.909) | +0.743 (1.144) |
| 8.120 | **+0.900 (1.086)** | **+0.814 (1.394)** |

At K=5 the same thing happens — and **this is where the two axes disagree**
([`budget_sweep_method_K5_gpt-4o-mini.md`](../compute/cost/tables/budget_sweep_method_K5_gpt-4o-mini.md)
and [`_claude-haiku-4-5`](../compute/cost/tables/budget_sweep_method_K5_claude-haiku-4-5.md)):
every row from 8.940 to 19.680 GPU-h favours PTO — primary +0.616, +0.745, +0.745, +0.650, +0.709,
+0.762, **+0.445**; held-out +0.511, +0.533, +0.533, +0.594, +0.594, +0.594, **+0.149**. The
mechanism is plain in the table's own columns: at `PTO_LA5`'s full 19.680 GPU-h, `GRPO_LA5` has
only reached **iteration 3** (18.310 h). GRPO's K=5 advantage at matched iteration is bought with
compute PTO never spent.

The honest-selection view (the grader that picks the checkpoint ≠ the grader that scores it) is
[`../compute/cost/tables/budget_sweep_crossjudge_verdicts.md`](../compute/cost/tables/budget_sweep_crossjudge_verdicts.md).
At the top of the sweep, `method_K0` is `arm_a > arm_b` (PTO) in **4 of 4** (select, eval) cells and
`method_K5` in **3 of 4**, the exception being select = primary / eval = held-out (+0.081,
dz 0.132, p_holm .075, "no sig. difference"). So the budget-axis verdict is "PTO at both K levels",
with one non-significant cell at K=5.

⚠ **The budget axis also carries an MI-inconsistency penalty the matched-iteration axis does not.**
In the same K=0 sweep at 8.120 GPU-h, the Q1Q2-selected checkpoints scored on MICI give +0.322
(dz 1.095) primary and +0.438 (dz 1.172) held-out — PTO **worse**. That is not a method property:
at equal spend PTO has trained ten iterations (nine under the held-out judge's selection) to
GRPO's two, and MI-inconsistency accumulates with
training depth in both methods ([`../arms/SUMMARY.md`](../arms/SUMMARY.md)). Neither half of "more
reward per GPU-hour, and more MI-inconsistency per GPU-hour with it" should be quoted without the
other, and both must be labelled matched-**budget**.

> **(Corrected 2026-08-25: the previous §2b's "3.4× cheaper" was arithmetically right — it used the
> same two totals rounded to 27.91 and 8.12 — but was presented as "on the compute axis PTO
> dominates GRPO outright" with no K named, and the
> K=5 arms' costs were missing because `GRPO_LA5` had not finished. The K=5 pair is
> 51.205 / 19.681 = 2.602, and at K=5 "cheaper" and "better" come apart — PTO is cheaper and
> scores lower at matched iteration.)**

## 8. Mechanism — the gap is in the DATA, and look-ahead makes the two data streams converge

The most useful cross-method result is not an outcome at all. Pooling every gradient group and
fitting the lexical direction each update pushes for
([`../arms/preference/tables/gpt-4o-mini/weighting_decomposition.md`](../arms/preference/tables/gpt-4o-mini/weighting_decomposition.md),
training oracle only):

| K | comparison | cosine | corrected |
|---|---|---|---|
| 0 | as trained (rule AND data differ) | 0.267 | 0.317 |
| 0 | same data, **rule** swapped (PTO's groups) | 0.908 | — |
| 0 | same data, **rule** swapped (GRPO's groups) | 0.988 | — |
| 0 | same rule, **data** differs | 0.356 / 0.266 | 0.397 / 0.324 |
| 5 | as trained (rule AND data differ) | 0.669 | 0.756 |
| 5 | same data, **rule** swapped (PTO / GRPO groups) | 0.961 / 0.978 | — |
| 5 | same rule, **data** differs | 0.673 / 0.636 | 0.740 / 0.730 |

Read down the K=0 block: swapping DPO's best-vs-worst rule for GRPO's group-relative weighting
**on the same groups** leaves the update direction essentially unchanged (0.908, 0.988), while
holding the rule fixed and swapping the *groups* collapses the cosine to 0.266–0.356. **The method
gap at K=0 is about the data the two methods train on, not about the loss family.** (The
same-groups rows share their estimation noise, so their attenuation-corrected values exceed 1.0 and
the raw cosine is the one to read — the table's `read` column says so.)

Then read across to K=5: the as-trained cosine rises from 0.267 to 0.669 (corrected 0.317 → 0.756),
and the same-rule-different-data rows rise from ~0.27–0.36 to ~0.64–0.67. **Look-ahead makes the
two methods' training data — and hence their updates — much more alike.** That is consistent with a
smaller method gap at K=5; it does *not* by itself explain why the residual gap points the other
way, and nothing in this top tests that. The pairwise version across all six arm pairs is
[`../arms/preference/tables/gpt-4o-mini/update_direction_cosines.md`](../arms/preference/tables/gpt-4o-mini/update_direction_cosines.md)
(`GRPO_LA0`–`PTO_LA0` 0.267 vs `GRPO_LA5`–`PTO_LA5` 0.669).

The other structural difference is how much of the generated data each method keeps
([`../arms/preference/tables/gpt-4o-mini/training_signal_yield.md`](../arms/preference/tables/gpt-4o-mini/training_signal_yield.md)):
GRPO trains on essentially every group (`yield_rate` 0.938–0.984 at K=0), while PTO's τ filter
drops the branch points whose branches tie (0.685–0.828 at K=0, falling to 0.685 by iteration 10).
PTO trains on fewer, higher-contrast decisions per unit of generation — the same fact that shows up
as its much smaller `train_h` in §7.

## 9. Caveats

- **Every claim needs its cost axis named.** Matched iteration says GRPO wins at K=5; matched
  budget says PTO wins at K=5. Both are correct answers to different questions
  ([`../compute/SUMMARY.md`](../compute/SUMMARY.md)).
- **Every claim needs its K named.** See §1 — there is no unconditional PTO-vs-GRPO verdict in
  this experiment.
- **Never average the two graders, and never compare levels across them.** The primary oracle WAS
  the training reward and the held-out judge never touched training — that is train-vs-test, not
  two raters. Only Δ, dz and standardized quantities travel.
- **Sign preservation across graders is high but not perfect**: 6,693 of 7,568 contrasts keep
  their sign (88.4%), rising to 99.3% once |Δ primary| ≥ 0.50
  ([`../measurement/validity/tables/multijudge_sign_preservation.md`](../measurement/validity/tables/multijudge_sign_preservation.md)).
  *(Corrected 2026-08-25: the previous file claimed the held-out judge "preserves 18/18 with
  bootstrap CIs excluding zero" across six named contrasts. No artifact in the current tree carries
  that statistic in that shape; the two supported statements are the 88.4% table above and the 8/8
  same-sign rows of `second_judge_contrasts.md` for the K=0 endpoint contrast.)*
- ⚠ **MITI arm differences are provisional.** Only 3.8% of MITI's arm-mean variance is between-arm
  signal (93.9% is judge level), `dependability_k1` = 0.624, and its cross-judge sign preservation
  is 79.8% overall / 89.6% at |Δ| ≥ 0.25 — against 100% for WAI-SR, CSQ-8 and MI-SAT at that
  threshold
  ([`multijudge_variance_components.md`](../measurement/validity/tables/multijudge_variance_components.md),
  [`multijudge_sign_preservation_by_metric.md`](../measurement/validity/tables/multijudge_sign_preservation_by_metric.md)).
  Lead with Q1+Q2.
- **Pairing is on `persona_id`, never `file_index`** — the 96 personas reshuffle every iteration.
  Means survive a `file_index` join; dz and CIs do not.
- **Every endpoint is a single 96-conversation draw** and therapist decoding is unseeded. **All 96
  personas are used for both training and eval**, so everything is in-sample with respect to the
  patient distribution ([`../LIMITATIONS.md`](../LIMITATIONS.md)).
- **Absolute scores are Exp3-internal only** — not comparable to Exp2 (4-bit vs bf16 generation),
  and never comparable across graders.
- **GPU-hours are reconstructed from artifact mtimes**, not from `iteration_metadata.json`, whose
  timings are per-process and undercount every resumed iteration; `n_imputed` in
  [`compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md) reports how often a resume gap
  had to be imputed at the phase median (1–6 intervals per arm).
