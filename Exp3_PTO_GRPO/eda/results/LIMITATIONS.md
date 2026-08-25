# Exp3 — measurement & inference limitations (for the thesis write-up)

Deliberately-scoped limitations of the Exp3 evaluation. These are **documented, not fixed**
(what *was* fixed is in [../../history/CHANGELOG_EDA.md](../../history/CHANGELOG_EDA.md)). Each
names where in the notebooks the reader meets it.

> **Scope of this file.** It owns the *measurement-quality* evidence — both judges' ICCs, agreement
> against the attenuation ceiling, coverage and sweep provenance — and the caveats that follow from
> them. The multi-judge **findings** (sign-preservation rates, the variance decomposition, gain
> retention) are owned by [`measurement/SUMMARY.md`](measurement/SUMMARY.md) and are cited
> here rather than restated, so the two cannot drift apart.
>
> **Grid status (2026-08-25).** All four arms (PTO_LA0, PTO_LA5, GRPO_LA0, GRPO_LA5) are trained to
> iteration 10, and all 4 arms × 11 states = **44** model states are scored under **both** graders.
> Every count and every range below is read off that complete grid. Where a figure here was computed
> on the earlier, right-censored grid, the correction is marked inline rather than silently
> overwritten.

## 1 · Judge reliability — MEASURED on a subset (2026-07-26); no human validation
Every conversation in the main eval is scored **once** by the oracle (`temperature=0.1, seed=42`),
which *freezes* the judge's bias for reproducibility but does not by itself measure it. It has now
been measured on the anchor-model subset (4 models × {Q1, Q2, MICI} × 96 convs, re-scored 3× with
per-rep seeds and compared against the reported draw — `Judge_Reliability.ipynb` Part 1, displayed
in `measurement/validity.ipynb` §1):

| metric | ICC(2,1) | mean \|Δ\| between reps |
|---|---|---|
| Q1 | 0.982–0.994 | 0.047–0.070 |
| Q2 | 0.955–0.992 | 0.076–0.089 |
| MICI | 0.864–0.943 | 0.037–0.069 |

Q1 and Q2 are "excellent" by the Koo & Li (2016) ≥0.90 guideline; MICI is 0.86–0.94, i.e. "good"
rather than excellent on its weakest cell (PTO@10). The mean |Δ| **confirms** the project's
informal "oracle noise ≈ 0.10" figure as a conservative upper bound. Since arm-level claims are
means over 96 conversations, this per-conversation noise shrinks by ~√96 at the level the thesis
actually reports.

> **Basis (changed 2026-07-28).** The ICC now spans **four** draws — the three seeded reps *plus
> the draw the thesis actually reports*, which the score-lake migration made addressable as
> `rep=0`. That is the more honest quantity: the question is how reproducible the reported number
> is, so the reported number belongs in the estimate, and it is how the second judge's ICC was
> already computed. It costs nothing (the draw existed) and moves only MICI, whose floor goes
> 0.895 → 0.864; Q1/Q2 shift by ≤0.007.

**What this still does not cover.** (a) Re-seeding at `temperature=0.1` probes **sampling** noise
only — it is a *floor* on reliability and says nothing about systematic sensitivity to rubric
wording, item order, or transcript position; a paraphrased-prompt rep would be needed for that.
(b) **This ICC table is anchor-subset only** — 3 metrics × 4 model states. That is a deliberate
choice, not a gap: see the rep argument below. The *second-judge* half is no longer subset-limited —
`Judge_Reliability.ipynb` §3 completed the full grid and has been kept complete since as new
model states landed — currently **44 × 8 × 96 = 33,792 / 33,792 cells**, so
`reliability.filter_complete_cells` drops nothing and every multi-judge number below is computed at
full n=96 per cell. Coverage is recorded in
[`measurement/validity/tables/multijudge_coverage.md`](measurement/validity/tables/multijudge_coverage.md),
which is 44 × 8 = 352 rows, every one of them **96 / 96 complete**.
*(Corrected 2026-08-25: this read "39 × 8 × 96 = 29,952" and "312 of 312". The grid stood at 39
model states while GRPO_LA5 was right-censored; it finished at iteration 10, so the grid is now
4 arms × 11 states = 44.)* ⚠ Quote the count from that table, not from here — it has been stale
three times now.

> **Provenance of the sweep** (kept because it explains the guard that is still in the code). The
> first submission — 9 batches, 21,120 requests — landed only 43%: the Anthropic credit balance ran
> out mid-sweep and 12,090 requests returned `invalid_request_error` (never billed). Partial cells
> were **quarantined rather than used**, because a partial cell's mean is unbiased but not
> *comparable* to a complete one: precision differs (SE scales as √(96/n), and n ran 24–96), and
> persona-PAIRED statistics collapse outright — two arms each covering a random ~43% of personas
> overlap on only ~0.43² ≈ 18% of them, gutting the pairing that `all_pairs_contrasts` and
> `gain_retention` depend on. After a top-up the remainder was resubmitted (5 batches) and completed,
> plus 13 stragglers filled via the live path (`Grammar compilation timed out`, transient).
> `filter_complete_cells` stays in the pipeline as a guard for any future partial sweep.
(c) There is still **no human MI/MITI-coder validation** — an oracle can be perfectly repeatable
and consistently wrong. That remains the strongest further addition (costs Lior-time, not API
budget), and no ICC substitutes for it.

**Why breadth was bought before depth.** Quantified in `measurement/validity.ipynb` §2: oracle
noise contributes ≈0.01 to a 96-conversation arm mean, against ≈0.09 from persona sampling — an
order of magnitude less. A second rep therefore cannot move any arm-level conclusion (Q1 per-
conversation reliability goes 0.98 → 0.99 from 1 → 3 reps). At equal cost, **breadth** (all models ×
both judges) strictly dominates **depth** (more reps of a few cells), which is why the full grid was
scored at one rep per judge. Depth was bought only where it answered a question breadth could not —
the second judge's own repeatability, whose absence made the MICI diagnosis ambiguous (next
paragraph).

**Second-judge repeatability — measured (2026-07-28).** The attenuation ceiling reported in §2 is
`sqrt(ICC_primary × ICC_judge)`. Until now `ICC_judge` was unmeasured and assumed equal to the
primary's, collapsing the ceiling to `ICC_primary`. Two further Haiku reps on the anchor subset
(3 reps total, 4 model states × {Q1, Q2, MICI} × 96 conversations) make both terms measured:

| metric | ICC(2,1) primary | ICC(2,1) Haiku |
|---|---|---|
| Q1 | 0.982–0.994 | 0.951–0.978 |
| Q2 | 0.955–0.992 | 0.938–0.963 |
| MICI | 0.864–0.943 | **0.525–0.929** |

*(Primary column = the same four-draw estimate as the table at the top of this section; source
`oracle_repeatability_icc.md` + `second_judge_agreement.md`.)*

On Q1 and Q2 the assumption was sound: Haiku is nearly as repeatable as the primary and the ceiling
moves by <0.03. On MICI it was not. Haiku's MICI repeatability falls as the MI-inconsistency rate
rises — PTO Base (MICI 0.21) 0.929, PTO@10 (0.49) 0.815, GRPO@8 (0.54) 0.749, GRPO@10 (0.84)
**0.525** — so it is least reliable on the arms the sycophancy claim concerns, where the achievable
ceiling is **0.70**, not the 0.93 previously assumed.

Correcting the ceiling therefore raises MICI's agreement-as-a-share-of-achievable, but not to the
level of the other rubrics:

| metric | observed r | ceiling (measured) | r as % of ceiling |
|---|---|---|---|
| Q1 | 0.84–0.88 | 0.97–0.99 | 86–91% |
| Q2 | 0.80–0.86 | 0.96–0.98 | 83–88% |
| MICI | 0.20–0.55 | 0.70–0.93 | **29–59%** |

Weak MICI agreement is thus **partly** the second judge's own noise and **mostly** construct
disagreement: against a ceiling corrected for that noise, MICI still recovers only ~39% of
achievable agreement where Q1/Q2 recover ~85%. The §2 MICI caveat therefore stands. `reliability.agreement`
now computes the ceiling from measured values on both sides and records which basis applied in
`ceiling_basis`.

> ⚠ **Consequence for the multi-judge analysis** (`measurement/validity.ipynb` §2). It reads Haiku **rep 0 only**, and a single-rep
> Haiku MICI score on GRPO@10 has ICC 0.525 — barely half its variance is signal. Treat one-rep
> MICI on the high-MICI arms as indicative; averaging the three anchor reps now on disk would
> resolve it for those four model states.

## 2 · Shared-model (patient = oracle) coupling
The simulated patient **and** the grading oracle are the **same** model
(`gpt-4o-mini-2024-07-18`). Several instruments (WAI-SR, CSQ-8, MI-SAT, PCT) rate the session
"from the patient's perspective," so the generator and the evaluator are coupled — this can
inflate patient-perspective alliance/satisfaction. The reward-hacking argument in
`arms/validity.ipynb` §2 is built to survive this: its load-bearing evidence is the **deterministic
text metrics** (turn length, loop %, question rate) that use no oracle at all, with the
un-rewarded oracle axes (MICI, PCT, MITI ratios) as corroboration.

**Empirically bounded (2026-07-26).** The same subset was re-scored by **Claude Haiku 4.5** — a
different model family that never played the patient (`Judge_Reliability.ipynb` Part 2, displayed in
`measurement/validity.ipynb` §1). Three findings:

1. **Every endpoint contrast keeps its sign — 16/16**
   ([`measurement/validity/tables/second_judge_contrasts.md`](measurement/validity/tables/second_judge_contrasts.md):
   2 endpoint contrasts × 8 rubrics = 16 rows, all `same_sign = True`; was 6/6 when only two
   hand-picked pairs were checked). At K=0, PTO@10 − GRPO@10: Q1 +0.77 (primary +0.53), Q2 +0.45
   (+0.48), MICI −0.22 (−0.35, lower = better). PTO@10 − PTO Base: same sign on all eight under both
   judges. The decoupled judge *widens* the headline Q1 gap rather than shrinking it.

   **But the method verdict is a function of K, and that must be said in the same breath.** The
   matched-iteration head-to-head now lives in
   [`method/contrast/tables/method_paired_by_K.md`](method/contrast/tables/method_paired_by_K.md)
   (+ = PTO higher; persona-paired n = 96; Holm within each judge × K). On Q1+Q2 at iteration 10,
   PTO − GRPO is **+0.507** (dz 0.729, p_holm .000) at K=0 and **−0.210** (dz −0.356, p_holm .001)
   at K=5 under the primary, and **+0.609** (dz 1.265, p_holm .000) / **−0.206** (dz −0.313,
   p_holm .034) under the held-out judge. **The sign flips with K, and both graders flip with it.**
   So: the PTO-vs-GRPO result is *not* an artifact of the patient and the grader sharing a model —
   and it is *not* a verdict about PTO vs GRPO unless K is named. **Never write one without K.**

   **Beyond the endpoint contrasts, the whole grid agrees where it matters.**
   `all_pairs_contrasts` enumerates *every* model-state pair × rubric on the full grid —
   8 rubrics × 946 state pairs = 7,568 contrasts (44 × 43 / 2 = 946 pairs) — and sign
   preservation rises monotonically with effect size, from **88.4%** pooled to **99.3%** at
   |Δ|≥0.50. The two judges therefore disagree **only about differences too small to claim**, which
   is the pattern a trustworthy instrument should show. *(Full ladder + the per-rubric breakdown:
   [`measurement/SUMMARY.md`](measurement/SUMMARY.md), tracked tables
   `multijudge_sign_preservation{,_by_metric}.md`. Corrected 2026-08-25: the pooled/≥0.50 pair read
   88.3% / 98.9% over "1,848 contrasts in `L0`" — a count from the retired L0 view on the
   39-state grid.)*

   **What matters here is where it fails: MITI.** It is the worst rubric pooled (**79.8%**), and —
   the sharper point — the only one that still disagrees at a **claimable** gap: every other rubric
   with contrasts that large reaches 96.2–100% by |Δ|≥0.25, MITI only 89.6%, needing |Δ|≥0.50 to
   reach 97.9%. So MITI is not merely noisier; a difference large enough to report can still flip
   sign under a different grader.
   This is an independent confirmation of the dependability warning below, arrived at from a
   completely different statistic. ⚠ Ladder thresholds are *absolute* — read one down its own
   rubric, never across rubrics (`METRICS_REFERENCE.md` §7b).
2. **Rank agreement on Q1/Q2 is high**: r 0.84–0.88 (Q1) and 0.80–0.86 (Q2) against a **measured**
   attenuation ceiling of 0.97–0.99 / 0.96–0.98 — i.e. 86–91% (Q1) and 83–88% (Q2) of the agreement
   two raters of this reliability could reach (§1; both ICC terms measured since 2026-07-28).
3. **Haiku is systematically harsher** (on the completed 44-state grid: Q1 −1.18 to −1.84, Q2
   −1.03 to −1.78) and flags *more*
   MI-inconsistent behaviour (MICI +0.12 to +0.42). This is a **level** shift, which cancels in
   every contrast the thesis reports; absolute Q1/Q2 values are grader-specific and should never be
   compared across judges.

**The MICI caveat.** Per-conversation cross-judge agreement on MICI is weak (r 0.20–0.55, ρ
0.21–0.47) even though the primary oracle is reasonably self-consistent on it (ICC 0.86–0.94). Since
2026-07-28 the three contributing causes are separated rather than conflated (§1). The second
judge's *own* MICI noise is real — Haiku's ICC is 0.525–0.929 and falls as the MI-inconsistency
rate rises, lowering the achievable ceiling to 0.70–0.93 — and statistical attenuation contributes
too, since `MICI_Rate` is a low, zero-inflated count-per-turn whose restricted range depresses
correlation. But neither accounts for most of the gap: measured against the corrected ceiling, MICI
recovers only ~39% of achievable agreement where Q1/Q2 recover ~85%. **Genuine construct
disagreement remains the dominant term** — the two model families do not count MI-inconsistent
behaviours the same way.

So the **sycophancy claim should be stated at the contrast level, with its axis, its K and its
grader named** — **not as a precise per-conversation rate**. At **K=0**, on per-therapist-turn
`MICI_Rate`, both judges agree GRPO@10 is more MI-inconsistent than PTO@10 (0.838 vs 0.491 primary;
1.050 vs 0.825 held-out) and that MICI rises from base in both arms. At **K=5** that ordering does
not hold and is itself grader-dependent (0.210 vs 0.264 primary — GRPO *lower*; 0.628 vs 0.581
held-out — GRPO marginally higher; both leaderboard scorecards cited in §5c). The load-bearing
evidence should remain gain retention (§3) and the deterministic text metrics, not MICI. A
human-coded MICI sample is the fix (see §1).

**Variance decomposition (2026-07-27).** The level shift in (3) is not merely *assumed* to be
harmless — it is now separated from the part that would matter. A two-way random-effects
decomposition of the arm means the thesis reports (`measurement/validity.ipynb` §2b) splits their
variance into arm (signal), judge level, and **arm × judge** (an ordering that depends on who is
grading — the only component that can invalidate a claim):

Across the eight rubrics the judge term is large and the interaction is small — arm×judge is
**1.1–7.0%** of arm-mean variance, so **the two judges disagree about the level, not about the
ordering of arms**. Averaging both judges raises dependability only 0.928 → 0.963 on Q1 and
0.914 → 0.955 on Q2, which is the quantitative reason the design spent on breadth (all states ×
both judges) rather than on more repetitions of a few cells. *(Narrative in
[`measurement/SUMMARY.md`](measurement/SUMMARY.md); the full 8-rubric table is the tracked
[`measurement/validity/tables/multijudge_variance_components.md`](measurement/validity/tables/multijudge_variance_components.md)
— `n_arms = 44` model states × 2 judges, every cell n=96. Corrected 2026-08-25: this said "22 arms"
"on the `L0` view", both of which describe the retired view on the pre-completion grid. The pooled
29-arm figures quoted here before 2026-07-28 came from the `all` view, since retired to gitignored
scratch; same story to within a point or two, but no longer reproducible from a tracked artifact.)*

> ⚠ **MITI is the exception and it is a limitation, not a footnote.** Only **3.8%** of MITI's
> arm-mean variance is genuine between-state signal; **93.9%** is grader level. A single-judge MITI
> ranking is therefore only **0.624** dependable — well below the 0.81–0.97 spanned by the other
> seven rubrics, and below any conventional "good" threshold. Averaging both judges lifts it to
> **0.768**, still the weakest. **Treat MITI differences as provisional unless both judges agree on
> the direction.** The all-pairs enumeration in finding 1 reaches the same verdict independently:
> MITI preserves its sign on only **79.8%** of contrasts, the lowest of the eight rubrics.
> *(Corrected 2026-08-25 on the 44-state grid: 3.6% → 3.8%, 94.5% → 93.9%, `dependability_k1`
> 0.65 → 0.624, `dependability_k2` 0.79 → 0.768, sign preservation 77.5% → 79.8%. §5d quotes the
> same two `dependability_k1` cells and is now consistent with this box.)*
> This is consistent with MITI being the rubric with the most judge-dependent construct (behaviour
> counts and technique ratios rather than a Likert impression) and with the MICI caveat below.

**Coverage:** the second judge covers **all 8 rubrics × all 44 model states × 96 conversations**
(44 × 8 × 96 = 33,792, all 33,792 present). What remains: still only **one** alternative judge, and the
patient simulator is `gpt-4o-mini` in every conversation — this decouples the **grader**, not the
**generator**. A fully decoupled replication would need the patient re-simulated by another family,
which means regenerating every conversation (GPU + API), not just re-scoring.

> **Never average the two judges' raw scores.** The primary oracle *was the training reward*; the
> second judge never touched training. This is an optimization-target vs held-out-test comparison,
> not two interchangeable raters — averaging them is averaging train and test accuracy. It is also
> unsound numerically: the level offset is 1.0–1.8 points *and model-dependent* (on the completed
> grid the Q1 bias runs from −1.177 at `GRPOExp3_LA5_I5` to −1.835 at `GRPOExp3_LA0_I9`;
> `validity.xlsx`, sheet `second_judge_agreement`, column `bias_judge_minus_primary`), so a mean
> lands on neither judge's rubric anchors and applies a
> silent, model-dependent shrinkage to every effect. `eda_analysis.reliability` reports the two
> side by side and only ever combines *contrasts* or *standardized* quantities.

## 3 · Training reward = outcome metric (circularity)
Q1+Q2 is **both** the training reward **and** a headline eval metric. "Q1+Q2 improved" is
therefore partly circular and cannot by itself demonstrate MI-skill gain. Q1+Q2 is best framed
as a **satisfaction/alliance proxy** (Q1 = session satisfaction, Q2 = working alliance /
relational communication — the lab's CLPsych-2024 LLM-evaluator prompts, see
`METRICS_REFERENCE.md` §1; 22 subjective items with endpoint-only Likert anchors — itself a
plausible *cause* of the observed reward-hacking, not only the optimiser). The honest
outcome axes are the ones **outside** the reward: `PCT`, `MICI`, the MITI technique ratios, and
the deterministic text metrics. See the confirmatory/exploratory split in `arms/stats.ipynb` §0.

**Partially quantified (2026-07-27) — gain retention under a held-out judge.** The second judge
gives a direct handle on this circularity rather than only a caveat. Because the primary oracle
*was* the training reward and Claude Haiku 4.5 never touched training, `Δ(held-out) / Δ(trained-
against)` is a **train/test generalization ratio** for each arm's gain over Base
(`measurement/validity.ipynb` §2c, persona-bootstrap CIs):

| metric | PTO@10 | GRPO@8 | GRPO@10 |
|---|---|---|---|
| **Q1** | **0.80** [0.67, 0.93] | 0.64 [0.53, 0.78] | **0.28** [0.06, 0.43] |
| Q2 | 0.85 [0.75, 0.98] | 0.80 [0.68, 0.95] | 0.81 [0.69, 0.96] |

*(Source: [`measurement/validity/tables/multijudge_gain_retention.md`](measurement/validity/tables/multijudge_gain_retention.md);
CI bounds re-read on the 44-state render, which moved **four** of them by one unit in the second
decimal — Q1 PTO@10 lo, Q1 GRPO@10 lo, Q2 GRPO@8 hi and Q2 GRPO@10 hi. Every point estimate is
unchanged.)*

Read it as two facts. (a) **Q2 retention is flat** at 0.80–0.85 with fully overlapping intervals —
that is scale compression between graders, and it is uninteresting. (b) **Q1 retention is
arm-specific and collapses**: PTO@10 and GRPO@10 do not overlap. Under a judge that never graded
during training, GRPO's net Q1 gain over 10 iterations is ≈0.19 points — it ends close to where it
started — while the primary judge credits it ≈0.68.

**The per-iteration trajectory makes it an onset curve, not just an endpoint fact**
(`multijudge_retention_trajectory.png`; the same table, every iteration — printed in
[`measurement/SUMMARY.md`](measurement/SUMMARY.md)). The two arms are indistinguishable for
the first three iterations and then separate: PTO holds 0.80–0.98 for the whole run while GRPO
**decays monotonically in trend** from ~0.89 (I3) to 0.28 (I10). So the divergence is not a
property of the endpoint that happened to be measured — the held-out grader stops crediting GRPO's
gains progressively, exactly as an optimiser drifting onto grader-specific features would predict.
(Iteration 9 is the visible floor at 0.03; it is also the arm's global dip in the primary eval, so
read it as the extreme of the trend rather than a separate event.)

This is the standard reward-hacking signature (a policy that overfits its grader does not
transfer), and it is **stronger evidence for the sycophancy claim than the MICI rate**, which
carries a known weak-agreement caveat (§2). It does not dissolve the circularity — Q1+Q2 remains
the reward — but it converts "partly circular, treat with caution" into a measured statement about
*which* arms' gains survive an independent grader, by how much, and *from which iteration on*.

**The three columns above are the K=0 arms, and reading (a) is a K=0 statement.** Both K=5 arms now
reach iteration 10 in the same table. Their Q1 retention lands between PTO@10 and GRPO@10
(PTO_LA5 0.715 [0.61, 0.84]; GRPO_LA5 0.686 [0.58, 0.81]), but their **Q2** retention does *not*
join the flat 0.80–0.85 band — PTO_LA5 is 0.567 [0.49, 0.67], an interval disjoint from PTO_LA0's
[0.75, 0.98]. "Q2 retention is uninteresting scale compression" is therefore true of K=0 and false
of PTO at K=5.

Two honest limits remain. (a) Retention is a ratio of two estimated deltas, hence the CIs — compare
arms by interval overlap, not by point estimate; where the denominator delta is near zero the ratio
is undefined and the table reports `nan` rather than a large spurious number. (b) All arms are
referenced to the **shared `PTOExp3_LA0_Base`**, not to their own base, so that the columns are
on one scale; the bases differ by Q1 0.10 (primary) / 0.02 (judge), far too little to drive the
PTO-vs-GRPO gap. The coverage limit is gone: this is measured on every scored iteration of all four
arms, and the K=0 Q1 **point estimates** are unchanged from the 4-anchor version, which is itself
reassuring. (Its intervals did move by one unit in the second decimal on this render — see the
source note above; the reassurance is about the estimates, not the bounds.)

> ⚠ **Retention is metric-dependent AND iteration-dependent — name both, and never generalise from
> one iteration.** Read as a K contrast in
> [`lookahead/transfer/tables/k_retention_summary.md`](lookahead/transfer/tables/k_retention_summary.md)
> (which reports iterations **5 and 10 only**), GRPO's Q1+Q2 retention intervals at iteration 10
> **overlap**: K=0 0.578 [0.457, 0.723] vs K=5 0.668 [0.587, 0.765], `cis_disjoint = False`. Only
> **Q1** stays disjoint there (K=0 0.295 [0.054, 0.488] vs K=5 0.676 [0.578, 0.795]).
> An earlier reading generalised **iteration 9**, where the same two Q1+Q2 intervals *were* disjoint
> (K=0 0.191 [−0.064, 0.382] vs K=5 0.686 [0.599, 0.793]) — but that row is not in the `.md` above;
> it lives on the `k_retention` sheet of
> [`lookahead/transfer/tables/transfer.xlsx`](lookahead/transfer/tables/transfer.xlsx)
> (`ref_kind = own_base`). Iteration 9 is an outlier, not the trend. A retention verdict quoted
> without its rubric and its iteration is not a claim about a table — and one quoted from the `.md`
> for an iteration the `.md` does not contain is not a claim about *that* table.

## 4 · PCT is not independent of the global-eval rubrics
Empirically `PCT` (patient change-talk proportion) loads **with** the global-evaluation (halo)
family (ρ≈0.79–0.94; high PC1 loading), so it does not isolate MI *technique*. The genuine second
factor is `MICI ↓` + the MITI ratios (`R:Q`/`%CR`/`%MICO`). Reported as a finding in
`arms/validity.ipynb` §1 rather than hidden.

## 5 · Look-ahead (K=0 vs K=5) — no longer thin, but AXIS-dependent
Both K=5 arms are now trained to iteration 10 and fully scored, as are both K=0 arms, so the K
contrast is inferential rather than merely descriptive. What replaced thinness as the live caveat
is that **the answer depends on which axis you match on**:

- At matched **iteration**, K=5 buys ~1.9x the compute per cell: the GRPO per-optimizer-step ratio
  of K=5 over K=0 runs 1.83–2.41 across iterations 1–10 and sits at 1.874 at the endpoint
  ([`compute/cost/tables/step_multiplier.md`](compute/cost/tables/step_multiplier.md)). A
  matched-iteration comparison is therefore **not** budget-neutral.
- Over whole arms the gap is of the same size within GRPO, and larger within PTO. GRPO K=5 cost
  **51.205** GPU-hours against K=0's **27.906**, i.e. 51.205 / 27.906 = 1.835x — slightly *below*
  the endpoint per-step ratio above, because generation and scoring outside the optimizer loop are
  shared. Within PTO the whole-arm gap is bigger: 19.681 / 8.119 = 2.424x
  ([`compute/cost/tables/compute_by_arm.md`](compute/cost/tables/compute_by_arm.md); PTO_LA0 8.119,
  PTO_LA5 19.681). *(Corrected 2026-08-25: this bullet said "the two GRPO arms turn out to have cost
  the same total GPU-hours (27.1 vs 27.9)". That was an artifact of GRPO_LA5 stopping at iteration 5.
  It ran to iteration 10; the two GRPO arms are nowhere near iso-compute, and any prose built on
  "look-ahead was free here" is void.)*
- The lever's sign is a **function of budget**, not a constant — and that part survives intact. On
  Q1+Q2 under the primary
  ([`compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md`](compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md)),
  GRPO K=5 is **worse** at every rung up to 18.31 GPU-h (−0.569 at 13.27, p_holm .000; −0.143 at
  18.31, p_holm .053), **level** at 23.21–27.08 (+0.038, p 0.789), and **better** from 30.53 on —
  first significant at 35.29
  (+0.188, p_holm .020) and reaching +0.435 (dz 0.743, p_holm .000) at 51.20. The held-out twin
  ([`compute/cost/tables/budget_sweep_GRPO_K_claude-haiku-4-5.md`](compute/cost/tables/budget_sweep_GRPO_K_claude-haiku-4-5.md))
  crosses earlier: level at 18.31 (−0.051, n.s.), already significantly positive at 23.21 (+0.147,
  p_holm .012).
- At the **top** rung the verdict is unanimous:
  [`compute/cost/tables/budget_sweep_crossjudge_verdicts.md`](compute/cost/tables/budget_sweep_crossjudge_verdicts.md)
  puts GRPO_LA5 > GRPO_LA0 on **all four** select-judge × eval-judge combinations at 51.200 GPU-h
  (mean_delta 0.256–0.435, every p_holm .000). So "look-ahead does not pay for GRPO" is a
  **low-budget** statement, not a general one.
- The **MI-consistency** reading at matched budget is selection- *and* grader-dependent and must
  never be quoted bare. At 51.200 GPU-h, selecting on Q1+Q2 and evaluating per-therapist-turn
  `MICI_Rate`, GRPO K=5 comes out at **−0.325** (dz −1.129, better) under the primary but **+0.172**
  (dz 0.626, worse) under the held-out judge — not because the graders disagree about a state, but
  because they select different checkpoints (I10 vs I8 primary; I7 vs I3 held-out). Selecting on
  MICI itself flips it a third time (+0.027, p 0.110, n.s., primary).

**No sentence about look-ahead is complete without naming its axis** — and, on MI-consistency, its
selection rule and its grader as well. See `eda_analysis/compute.py` and the `compute/cost/`
artifacts.

## 5a · K has exactly two levels, {0, 5} — there is no dose-response
`LOOKAHEAD_K` was run at **two** values by design: 0 and 5. Nothing in Exp3 interpolates between
them and nothing goes past 5. ⚠ **One K=3 artifact does exist in the tree and it is NOT a dose
point:** [`lookahead/replication/tables/crossgen_la3_gpt35.md`](lookahead/replication/tables/crossgen_la3_gpt35.md)
carries **Exp1's** K=3 sweep (4 iterations, 96 conversations each). It is a different experiment —
different base model, patient prompts, hyperparameters, and it was never re-scored by the Exp3
grader — so it cannot be read as the midpoint of this experiment's K contrast. Its own caption says
so.
Every look-ahead statement in this EDA is therefore a statement about **one contrast**, not about a
monotone relationship in K. "Look-ahead helps GRPO" means "K=5 beat K=0 for GRPO, at this budget,
on this grader" — it licenses nothing about K=2, K=10, or the shape between the two measured
points, where diminishing returns, an interior optimum and an outright reversal are all equally
consistent with the evidence.

This is invisible from the results tree, because the `k_*` tables look exactly like a sweep would:
they are indexed by iteration, arm and grader, and the K column simply has two values. A reader who
meets them without this paragraph can reasonably infer K was swept. **State the two levels
explicitly wherever a K claim is made.** Adding a level is not a re-analysis but a new training run
per method — 2 methods × 1 extra level = 2 whole arms — and a K>5 arm costs more per step again than
K=5 does (§5, first bullet).

## 5b · Three denominators disagree, and only one of them is denominator-free
Behaviour claims can be stated per therapist TURN, per SESSION, or per unit of therapist LANGUAGE,
and on this data the three **disagree in direction** for the K contrast — because both the turn
count and the turn length move under K, and at the completed endpoint the **length** move is the
method-dependent one. At the matched iteration-10
endpoint ([`lookahead/behaviour/tables/session_shape.md`](lookahead/behaviour/tables/session_shape.md),
means in [`lookahead/behaviour/tables/length_endpoints.md`](lookahead/behaviour/tables/length_endpoints.md)):

| at iteration 10 | therapist turns, K=0 → K=5 | chars per therapist turn, K=0 → K=5 |
|---|---|---|
| PTO | 10.23 → 14.39 (+4.16, dz 0.55) | 686.2 → 810.9, i.e. 810.875 / 686.202 = 1.18x |
| GRPO | 12.75 → 15.97 (+3.22, dz 0.42) | 895.7 → 849.3, i.e. 849.274 / 895.711 = 0.95x |

So at the completed endpoint K=5 takes **more** therapist turns in *both* optimizers, and the
method-dependence has moved into turn **length**: K=5 writes longer turns under PTO and slightly
shorter ones under GRPO. Either denominator still moves, so a per-turn or per-session rate still
confounds the behaviour with the session shape. *(Corrected 2026-08-25: this said "at GRPO's
endpoint it takes ~4.0 *fewer* [turns], while writing ~1.7x longer turns". Both were read at
GRPO_LA5's then-endpoint of iteration 5, where K=0 15.34 vs K=5 11.31 — the turn-count sign
reverses by iteration 10. Corrected 2026-08-18: the PTO figure once read "~2.4 more", which is the
over-praise per-session Δ transposed onto the turn-count row — see
`lookahead/behaviour/tables/k_paired_channels.md`.)*

Per-1,000-character normalisation is not a fix either, for the same reason: the character
denominator is exactly the quantity that moves in the table above, so a change in MI-inconsistency
*density* under a K=5 arm can be dilution rather than skill. ⚠ No artifact in `results/` actually
computes a per-1,000-character rate — that sentence is an argument from the length tables, not a
measured result, and it should not be quoted as one. The only measure with no moving denominator is
the **share** of coded acts
([`lookahead/behaviour/tables/k_mici_composition.md`](lookahead/behaviour/tables/k_mici_composition.md)'s
`*_share` columns) — prefer it for any substitution claim, and label every rate as a rate.

## 5c · No replicate draw for any trained checkpoint; decoding has no per-call seed
Every dz in this EDA treats **one** 96-conversation draw as the model. The only noise floor that
exists is at the base: four independent draws of the *identical* untrained policy give
6 pairs x 9 metrics = 54 same-policy contrasts with **0 reaching even uncorrected p < .05**
(max |dz| 0.128 primary / 0.147 held-out). That floor is measured where sessions are short and
homogeneous; for trained checkpoints, where mean therapist turn length runs from 266–301 characters
at base to 686–896 at iteration 10
([`lookahead/behaviour/tables/length_endpoints.md`](lookahead/behaviour/tables/length_endpoints.md)),
**there is no replicate at all**.

**Which endpoints are contested is itself grader-dependent.** Under the primary only GRPO_LA0
disagrees with itself — best @8 = 4.082 vs final @10 = 3.753 on Q1+Q2
([`arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md`](arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md)).
Under the held-out judge **all four** do — none of its four best iterations is 10: PTO_LA0 @9,
PTO_LA5 @7, GRPO_LA5 @7, and **GRPO_LA0 @3 (2.637) vs final @10 (2.257), the largest best-vs-final
gap of the four**
([`arms/outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md`](arms/outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md)).
Every one of those states is a single draw, and GRPO_LA0's iteration-9 dip has the shape of a bad
draw with nothing to distinguish it from a real regression. *(Corrected 2026-08-25: this line named
"GRPO_LA0 @8 vs @10, GRPO_LA5 @4 vs @5"; the second pair was an artifact of that arm stopping at
iteration 5, and it now runs to 10.)*

Related and unrecorded elsewhere: therapist decoding has **no per-call seed**. Both trainer
notebooks do seed globally once at start-up (`SEED = 42` → `random.seed` / `np.random.seed` /
`torch.manual_seed` / `torch.cuda.manual_seed_all` in cell 1), but nothing in `_shared/` and nothing
in [`code/tools/generate_eval_convs.py`](../../code/tools/generate_eval_convs.py) seeds torch at
all — only the persona shuffle and the patient API seed are seeded there — and within a run the
global stream is consumed by everything that executes before a given `model.generate`. So a
stand-alone generate-only pass is unseeded and a resumed or re-ordered run does not reproduce the
same conversations: a replicate draw needs **no code change**, and an exact re-run cannot be used as
a check. *(Corrected 2026-08-25: this said there is "no `torch.manual_seed` / `set_seed`" in
`_shared/`, `generate_eval_convs.py` **or the trainers**. The first two are right; the trainer
notebooks do seed globally at cell 1. The conclusion is unchanged, because that seeding is
once-per-process rather than per-generation.)*

**This is a limitation about the EVAL draw only.** It holds the weights fixed and asks how much the
96 conversations would move if re-simulated. The separate — and entirely unmeasured — question of
how much the *weights themselves* would move under a different training seed is §5g.

## 5d · No channel-level reliability, and the weakest instruments carry the channel claims
`dependability_k1` is **0.624 for MITI** and **0.812 for MICI**
([`measurement/validity/tables/multijudge_variance_components.md`](measurement/validity/tables/multijudge_variance_components.md),
`n_arms = 44`) — the two least dependable instruments in the battery, the next weakest being Q2 at
0.914 — and they are the ones carrying the substitution/sycophancy results.
*(Corrected 2026-08-25: this read 0.553 / 0.628, values from an earlier and smaller grid, and it
disagreed with the figure §2 quoted for the same MITI cell. Both sections now read the same table.)*
Worse, the MICI ICC is computed on `MICI_Rate` only: there is **no reliability estimate for any
individual channel** (`MICI_OverPraise`, `MICI_Direct`, `MICI_AdviseNoPermission`), which is exactly
the granularity the channel claims live at. Repeatability reps exist for **4 model states, all
K=0, on 3 metrics** — 3 metrics × 4 model states = 12 rows in
[`measurement/validity/tables/oracle_repeatability_icc.md`](measurement/validity/tables/oracle_repeatability_icc.md),
and **no K=5 state has an ICC**, so cross-judge agreement on either look-ahead arm cannot be
benchmarked against a ceiling. Completing the grid did not change this: the anchor subset is still
the four K=0 states it always was.

## 5e · The patient distribution is saturated, so every number is in-sample
`generate_all_permutations(only_expert_therapist=True)` returns exactly
2 gender x 2 problem x 2 problem_time x 2 tried_to_solve x 3 cooperation_level x 2 age = **96**
permutations, all of them used. The self-loop makes those same 96 the training rollouts *and* the
eval set at every iteration. Generalisation to unseen patients is therefore **unmeasured and not
measurable without authoring new persona prompts** — state it, do not try to estimate it.

## 5f · Questions no artifact currently covers (from the 2026-08-18 and 2026-08-25 cold table audits)
The tables only exist for questions someone asked; these were never asked, so their absence is
invisible from the results tree:

- **Score by session-end reason.** PTO(K=0) learns to shorten sessions (28.4 → 20.4 utterances)
  while ~17–25% of conversations hit the turn cap and the therapist almost never ends one (3–7 per
  arm out of 11 states × 96 personas = 1,056 conversations,
  [`arms/validity/tables/gpt-4o-mini/session_end_reasons.md`](arms/validity/tables/gpt-4o-mini/session_end_reasons.md)).
  That table *counts* end reasons; no table anywhere **conditions a score** on how the session
  ended, so "shorter because the patient is satisfied" vs "shorter is a smaller surface to be marked
  down on" is unmeasured — and this interacts with the length confound every instrument carries.
- **Where in the session the over-praise sits.** All MICI channels are per-conversation counts;
  whether the praise concentrates in openings, closings, or uniformly (and whether K=5 moves its
  *position* as well as its volume) has no artifact.
- **Run-to-run training variance** — see §5g, which is large enough to be its own section.
- **Anything between or beyond K ∈ {0, 5}** — see §5a.

*(Retired 2026-08-25. A fourth entry stood here: "the K=5 endpoint under the held-out judge flips
the method ordering (GRPO_LA5@5 2.798 > PTO_LA5@10 2.667 on Q1+Q2) — visible in the L5 haiku
scorecard but tested nowhere, since it crosses unequal iterations". It is no longer uncovered. With
GRPO_LA5 trained to iteration 10 the contrast is tested at a **matched** iteration on **both**
graders in
[`method/contrast/tables/method_paired_by_K.md`](method/contrast/tables/method_paired_by_K.md):
PTO − GRPO at K=5, iteration 10 is −0.210 on Q1+Q2 (dz −0.356, p_holm .001) under the primary and
−0.206 (dz −0.313, p_holm .034) under the held-out judge. GRPO wins at K=5 on both graders. That is
now a headline result rather than a gap — and it is why §2 insists no PTO-vs-GRPO verdict may be
written without naming K.)*

## 5g · ONE training run per arm — run-to-run training variance is entirely unmeasured
Each of the four arms is a **single training run**. Every effect size in this EDA — every `dz`,
every bootstrap CI, every Holm-corrected p — is computed across the **96 personas inside that one
run**. Its uncertainty is therefore sampling uncertainty over *patients*, and nothing else. No arm
was retrained from the same configuration under a different seed, so the run-to-run spread of "what
PTO at K=0 converges to" has never been observed once, and **no artifact in `results/` can bound
it**. A grid of 4 arms × 1 run = 4 runs supports statements about these four runs; it does not, on
its own, support statements about the four *methods*.

**This is not §5c, and §5c's noise floor cannot stand in for it.** §5c is about the **eval** draw: a
*fixed* checkpoint re-simulated against the 96 personas with unseeded decoding. Its measured floor
(four independent draws of the identical base policy, 54 same-policy contrasts, 0 reaching
uncorrected p < .05) holds the weights constant *by construction* — that is precisely what makes it
a clean floor, and precisely why it says nothing about how far the *weights* would have moved under
a different training seed. The two replicates are different experiments with different price tags:
§5c's is one generate-only pass per checkpoint (`gen_h` runs 0.091–0.106 per iteration at the
endpoint, [`compute/cost/tables/compute_by_iteration.md`](compute/cost/tables/compute_by_iteration.md)),
§5g's is a whole arm
(8.119–51.205 GPU-hours each,
[`compute/cost/tables/compute_by_arm.md`](compute/cost/tables/compute_by_arm.md)).

**Why the variance is unlikely to be small.** Both trainers regenerate their training data from the
current policy at every iteration, and that generation goes through the same unseeded decode path
§5c describes. Training-seed variance here is therefore not merely optimizer noise on a fixed
dataset: a different draw changes the rollouts, which changes the pref pairs (PTO) or the group
advantages (GRPO), which changes the next policy — the arm's whole trajectory is downstream of one
sequence of samples. The observed iteration-to-iteration swings inside a single arm (GRPO_LA0's
iteration-9 dip, §5c) are a lower bound on how unstable that trajectory is, not an upper one.

**What it costs the claims.** Every arm-vs-arm and K-vs-K verdict in this EDA — the K interaction,
the method flip, the budget sweeps — is a comparison of **two single runs**, so its confidence
interval covers patient sampling but not training replication. State them as "in this run", and
treat a small effect between two arms as weaker evidence than its p-value suggests. The large ones
(dz > 0.7 at matched iteration, consistent across two graders and across a budget ladder) are the
ones least exposed to this, which is a reason to lead with them.

## 6 · Multiplicity is corrected within families, not across
Holm/BH corrections apply **within** each family (rubrics within one matched contrast, or
iterations within one arm-vs-base sweep) and are **not** pooled across the dozens of families
in the EDA. The confirmatory/exploratory split (`arms/stats.ipynb` §0) is what keeps this honest: treat
only the small pre-registered confirmatory set as tested claims; the rest are descriptive.
