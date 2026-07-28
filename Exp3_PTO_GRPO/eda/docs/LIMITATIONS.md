# Exp3 — measurement & inference limitations (for the thesis write-up)

Deliberately-scoped limitations of the Exp3 evaluation. These are **documented, not fixed**
(what *was* fixed is in [../history/CHANGELOG.md](../history/CHANGELOG.md)). Each names where in
the notebooks the reader meets it.

## 1 · Judge reliability — MEASURED on a subset (2026-07-26); no human validation
Every conversation in the main eval is scored **once** by the oracle (`temperature=0.1, seed=42`),
which *freezes* the judge's bias for reproducibility but does not by itself measure it. It has now
been measured on the anchor-model subset (4 models × {Q1, Q2, MICI} × 96 convs, re-scored 3× with
per-rep seeds — `Judge_Reliability.ipynb` Part 1, displayed in `5_Training_and_Reliability` §7):

| metric | ICC(2,1) | mean \|Δ\| between reps |
|---|---|---|
| Q1 | 0.981–0.994 | 0.04–0.08 |
| Q2 | 0.962–0.992 | 0.07–0.09 |
| MICI | 0.895–0.958 | 0.03–0.07 |

All are "excellent" by the Koo & Li (2016) ≥0.90 guideline, and the mean |Δ| **confirms** the
project's informal "oracle noise ≈ 0.10" figure as a conservative upper bound. Since arm-level
claims are means over 96 conversations, this per-conversation noise shrinks by ~√96 at the level
the thesis actually reports.

**What this still does not cover.** (a) Re-seeding at `temperature=0.1` probes **sampling** noise
only — it is a *floor* on reliability and says nothing about systematic sensitivity to rubric
wording, item order, or transcript position; a paraphrased-prompt rep would be needed for that.
(b) **This ICC table is anchor-subset only** — 3 metrics × 4 model states. That is a deliberate
choice, not a gap: see the rep argument below. The *second-judge* half is no longer subset-limited —
`Judge_Reliability.ipynb` §3 completed the full 8-rubric × 29-model grid on 2026-07-27
(**22,272 / 22,272 cells**, $42 batched), so `reliability.filter_complete_cells` now drops nothing
and every §8 number below is computed at full n=96 per cell. Coverage is recorded in
`multijudge_coverage.md` (232/232 cells).

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

**Why breadth was bought before depth.** Quantified in `5_Training_and_Reliability` §8: oracle
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
| Q1 | 0.981–0.994 | 0.951–0.978 |
| Q2 | 0.962–0.992 | 0.938–0.963 |
| MICI | 0.895–0.958 | **0.525–0.929** |

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
| MICI | 0.20–0.55 | 0.70–0.94 | **29–59%** |

Weak MICI agreement is thus **partly** the second judge's own noise and **mostly** construct
disagreement: against a ceiling corrected for that noise, MICI still recovers only ~39% of
achievable agreement where Q1/Q2 recover ~85%. The §2 MICI caveat therefore stands. `reliability.agreement`
now computes the ceiling from measured values on both sides and records which basis applied in
`ceiling_basis`.

> ⚠ **Consequence for §8.** The multi-judge analysis reads Haiku **rep 0 only**, and a single-rep
> Haiku MICI score on GRPO@10 has ICC 0.525 — barely half its variance is signal. Treat one-rep
> MICI on the high-MICI arms as indicative; averaging the three anchor reps now on disk would
> resolve it for those four model states.

## 2 · Shared-model (patient = oracle) coupling
The simulated patient **and** the grading oracle are the **same** model
(`gpt-4o-mini-2024-07-18`). Several instruments (WAI-SR, CSQ-8, MI-SAT, PCT) rate the session
"from the patient's perspective," so the generator and the evaluator are coupled — this can
inflate patient-perspective alliance/satisfaction. The reward-hacking argument in
`3_Validity_and_Hacking` §2 is built to survive this: its load-bearing evidence is the **deterministic
text metrics** (turn length, loop %, question rate) that use no oracle at all, with the
un-rewarded oracle axes (MICI, PCT, MITI ratios) as corroboration.

**Empirically bounded (2026-07-26).** The same subset was re-scored by **Claude Haiku 4.5** — a
different model family that never played the patient (`Judge_Reliability.ipynb` Part 2, displayed in
`5_Training_and_Reliability` §7). Three findings:

1. **Every contrast keeps its sign — 18/18** (2026-07-27; was 6/6 when only two hand-picked pairs
   were checked). PTO@10 − GRPO@10: Q1 +0.77 (primary +0.53), Q2 +0.45 (+0.48), MICI −0.22 (−0.35,
   lower = better). PTO@10 − PTO Base: all positive under both judges. The decoupled judge *widens*
   the headline Q1 gap rather than shrinking it. Critically, the enumeration added the two contrasts
   the thesis leans on hardest and that were never tested: the **best-vs-best steelman**
   (PTO@10 − GRPO@8: Q1 +0.32 judge vs +0.20 primary) and the **regression claim**
   (GRPO@8 − GRPO@10: Q1 +0.46 vs +0.33) — both preserved, both with bootstrap CIs excluding zero.
   **The PTO-vs-GRPO result is not an artifact of the patient and the grader sharing a model.**

   **Beyond the six thesis-critical contrasts, the whole grid agrees where it matters.**
   `all_pairs_contrasts` enumerates *every* arm pair × rubric in the view — **1,848** contrasts in
   `L0` (`multijudge_all_pairs_contrasts.md`) — and sign preservation rises monotonically with
   effect size (`multijudge_sign_preservation.md`):

   | primary-judge \|Δ\| | contrasts | same sign |
   |---|---|---|
   | any | 1,848 | **88.3%** |
   | ≥ 0.10 | 1,201 | 94.1% |
   | ≥ 0.25 | 736 | 97.0% |
   | ≥ 0.50 | 352 | **98.9%** |

   So the two judges disagree **only about differences too small to claim**, which is the pattern a
   trustworthy instrument should show. Restricting to contrasts whose judge-side bootstrap CI
   excludes zero (1,299 of 1,848) gives 94.7% agreement. Per rubric
   (`multijudge_sign_preservation_by_metric.md`), **MITI is the worst at 77.5%** and Q1 is 86.6%,
   against PCT 93.5% and MICI 92.2% — an independent confirmation of the MITI dependability warning
   below, arrived at from a completely different statistic.

   **MITI fails the ladder as well as the pooled rate, and that is the sharper finding.** Every
   other rubric reaches **95.5–100%** by |Δ|≥0.25 — i.e. once a gap is large enough to claim, the
   judges simply agree. MITI reaches only **88.2%** there and needs |Δ|≥0.50 to hit 97.6%. So MITI
   is not merely noisier: it is the one rubric where a **claimable-size** difference can still flip
   sign under a different grader. ⚠ The thresholds are *absolute*, so read a ladder down its own
   rubric, never across rubrics — PCT and MICI sit on a 0–1 scale and never reach 0.25 at all, while
   Q1/Q2/WAI-SR/MITI are 1–5 or 1–7. Only the pooled `all contrasts` row is cross-rubric comparable.
2. **Rank agreement on Q1/Q2 is high**: r 0.84–0.88 (Q1) and 0.80–0.86 (Q2) against an attenuation
   ceiling of ~0.98 — i.e. 82–89% of the agreement two raters of this reliability could reach.
3. **Haiku is systematically harsher** (Q1 −1.25 to −1.74, Q2 −1.09 to −1.32) and flags *more*
   MI-inconsistent behaviour (MICI +0.15 to +0.36). This is a **level** shift, which cancels in
   every contrast the thesis reports; absolute Q1/Q2 values are grader-specific and should never be
   compared across judges.

**The MICI caveat.** Per-conversation cross-judge agreement on MICI is weak (r 0.20–0.55, ρ
0.21–0.47) even though the primary oracle is self-consistent on it (ICC 0.90–0.96). Since
2026-07-28 the three contributing causes are separated rather than conflated (§1). The second
judge's *own* MICI noise is real — Haiku's ICC is 0.525–0.929 and falls as the MI-inconsistency
rate rises, lowering the achievable ceiling to 0.70–0.94 — and statistical attenuation contributes
too, since `MICI_Rate` is a low, zero-inflated count-per-turn whose restricted range depresses
correlation. But neither accounts for most of the gap: measured against the corrected ceiling, MICI
recovers only ~39% of achievable agreement where Q1/Q2 recover ~85%. **Genuine construct
disagreement remains the dominant term** — the two model families do not count MI-inconsistent
behaviours the same way.

So the **sycophancy claim should be stated at the contrast level** (both judges agree GRPO@10 is
more MI-inconsistent than PTO@10, and that MICI rises from base in both arms), **not as a precise
per-conversation rate** — and the load-bearing evidence should remain gain retention (§3) and the
deterministic text metrics, not MICI. A human-coded MICI sample is the fix (see §1).

**Variance decomposition (2026-07-27).** The level shift in (3) is not merely *assumed* to be
harmless — it is now separated from the part that would matter. A two-way random-effects
decomposition of the arm means the thesis reports (`5_Training_and_Reliability` §8b) splits their
variance into arm (signal), judge level, and **arm × judge** (an ordering that depends on who is
grading — the only component that can invalidate a claim):

Numbers below are the **`L0` view** (22 arms × 2 judges, every cell at n=96) — the tracked primary
deliverable, `multijudge_variance_components.md`. The pooled 29-arm figures quoted before
2026-07-28 came from the `all` view, which has since been retired to gitignored scratch; they told
the same story to within a percentage point or two, but are no longer reproducible from a tracked
artifact, so `L0` is now the cited source throughout this document (as §3's retention table already
was).

| metric | arm | judge level | **arm × judge** | dependability, 1 judge | 2 judges |
|---|---|---|---|---|---|
| PCT | 72.0% | 24.5% | **3.5%** | 0.95 | 0.98 |
| CSQ-8 | 70.6% | 22.5% | **6.9%** | 0.91 | 0.95 |
| MICI | 45.5% | 48.3% | **6.2%** | 0.88 | 0.94 |
| MI-SAT | 29.6% | 67.0% | **3.4%** | 0.90 | 0.95 |
| WAI-SR | 22.8% | 75.2% | **1.9%** | 0.92 | 0.96 |
| Q2 | 13.2% | 85.5% | **1.3%** | 0.91 | 0.95 |
| Q1 | 10.9% | 87.8% | **1.2%** | 0.90 | 0.95 |
| **MITI** | **3.6%** | **94.5%** | **1.9%** | **0.65** | 0.79 |

The judge term is large and the interaction is small everywhere: **the two judges disagree about
the level, not about the ordering of arms.** Averaging both judges raises dependability only
~0.91→0.95 on Q1/Q2, which is the quantitative reason the design spent on breadth (all arms × both
judges) rather than on more repetitions of a few cells.

> ⚠ **MITI is the exception and it is a limitation, not a footnote.** Only **3.6%** of MITI's
> arm-mean variance is genuine between-arm signal; **94.5%** is grader level. A single-judge MITI arm
> ranking is therefore only **0.65** dependable — well below the ~0.90 of every other rubric, and
> below any conventional "good" threshold. Averaging both judges lifts it to 0.79, still the
> weakest. **Treat MITI arm differences as provisional unless both judges agree on the direction.**
> The all-pairs enumeration in finding 1 reaches the same verdict independently: MITI preserves its
> sign on only **77.5%** of contrasts, the lowest of the eight rubrics.
> This is consistent with MITI being the rubric with the most judge-dependent construct (behaviour
> counts and technique ratios rather than a Likert impression) and with the MICI caveat below.

**Coverage as of 2026-07-27:** the second judge now covers **all 8 rubrics × all 29 model states ×
96 conversations** (22,272/22,272). What remains: still only **one** alternative judge, and the
patient simulator is `gpt-4o-mini` in every conversation — this decouples the **grader**, not the
**generator**. A fully decoupled replication would need the patient re-simulated by another family,
which means regenerating every conversation (GPU + API), not just re-scoring.

> **Never average the two judges' raw scores.** The primary oracle *was the training reward*; the
> second judge never touched training. This is an optimization-target vs held-out-test comparison,
> not two interchangeable raters — averaging them is averaging train and test accuracy. It is also
> unsound numerically: the level offset is 1.2–1.7 points *and model-dependent* (Q1 bias runs −1.25
> at Base to −1.74 at GRPO@10), so a mean lands on neither judge's rubric anchors and applies a
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
the deterministic text metrics. See the confirmatory/exploratory split in `7_Stats` §0.

**Partially quantified (2026-07-27) — gain retention under a held-out judge.** The second judge
gives a direct handle on this circularity rather than only a caveat. Because the primary oracle
*was* the training reward and Claude Haiku 4.5 never touched training, `Δ(held-out) / Δ(trained-
against)` is a **train/test generalization ratio** for each arm's gain over Base
(`5_Training_and_Reliability` §8c, persona-bootstrap CIs):

| metric | PTO@10 | GRPO@8 | GRPO@10 |
|---|---|---|---|
| **Q1** | **0.80** [0.68, 0.93] | 0.64 [0.53, 0.78] | **0.28** [0.05, 0.43] |
| Q2 | 0.85 [0.75, 0.98] | 0.80 [0.68, 0.96] | 0.81 [0.69, 0.95] |

Read it as two facts. (a) **Q2 retention is flat** at 0.80–0.85 with fully overlapping intervals —
that is scale compression between graders, and it is uninteresting. (b) **Q1 retention is
arm-specific and collapses**: PTO@10 and GRPO@10 do not overlap. Under a judge that never graded
during training, GRPO's net Q1 gain over 10 iterations is ≈0.19 points — it ends close to where it
started — while the primary judge credits it ≈0.68.

**The per-iteration trajectory makes it an onset curve, not just an endpoint fact**
(`multijudge_retention_trajectory.png`; the same table, every iteration). Q1 retention against the
shared PTO base:

| iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| PTO_LA0 | 0.97 | 0.84 | 0.89 | 0.94 | 0.98 | 0.97 | 0.94 | 0.89 | 0.88 | **0.80** |
| GRPO_LA0 | 1.13 | 0.79 | 0.89 | 0.79 | 0.73 | 0.57 | 0.70 | 0.64 | 0.03 | **0.28** |

The two arms are indistinguishable for the first three iterations and then separate: PTO holds
0.80–0.98 for the whole run while GRPO **decays monotonically in trend** from ~0.89 to 0.28. So the
divergence is not a property of the endpoint that happened to be measured — the held-out grader
stops crediting GRPO's gains progressively, exactly as an optimiser drifting onto grader-specific
features would predict. (Iteration 9 is the visible floor at 0.03; it is also the arm's global dip
in the primary eval, so read it as the extreme of the trend rather than a separate event.)

This is the standard reward-hacking signature (a policy that overfits its grader does not
transfer), and it is **stronger evidence for the sycophancy claim than the MICI rate**, which
carries a known weak-agreement caveat (§2). It does not dissolve the circularity — Q1+Q2 remains
the reward — but it converts "partly circular, treat with caution" into a measured statement about
*which* arms' gains survive an independent grader, by how much, and *from which iteration on*.

Two honest limits remain. (a) Retention is a ratio of two estimated deltas, hence the CIs — compare
arms by interval overlap, not by point estimate; where the denominator delta is near zero the ratio
is undefined and the table reports `nan` rather than a large spurious number. (b) Both arms are
referenced to the **shared `PTOExp3_LA0_Base`**, not to their own base, so that the two columns are
on one scale; the two bases differ by Q1 0.10 (primary) / 0.02 (judge), far too little to drive the
PTO-vs-GRPO gap. The coverage limit is gone: this is measured on every scored iteration of both K=0
arms, and the Q1 point estimates and intervals are unchanged from the 4-anchor version, which is
itself reassuring.

## 4 · PCT is not independent of the global-eval rubrics
Empirically `PCT` (patient change-talk proportion) loads **with** the global-evaluation (halo)
family (ρ≈0.79–0.94; high PC1 loading), so it does not isolate MI *technique*. The genuine second
factor is `MICI ↓` + the MITI ratios (`R:Q`/`%CR`/`%MICO`). Reported as a finding in
`3_Validity_and_Hacking` §1 rather than hidden.

## 5 · Look-ahead (K=0 vs K=5) is descriptive only
The LA5 arms are thin (PTO_LA5 = 4 scored iters, GRPO_LA5 = 1), so every K contrast
(`5_Training_and_Reliability` §4, `6_Preference` §2, `7_Stats` §4) is **hypothesis-generating, not inferential**
— banners mark these in-notebook. The confirmatory PTO-vs-GRPO result is at K=0 and is
unaffected.

## 6 · Multiplicity is corrected within families, not across
Holm/BH corrections apply **within** each family (rubrics within one matched contrast, or
iterations within one arm-vs-base sweep) and are **not** pooled across the dozens of families
in the EDA. The confirmatory/exploratory split (`7_Stats` §0) is what keeps this honest: treat
only the small pre-registered confirmatory set as tested claims; the rest are descriptive.
