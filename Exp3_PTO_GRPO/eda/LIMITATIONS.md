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
(b) The measurement is on 3 metrics and 4 model states, not the full 8-rubric × 29-model grid.
(c) There is still **no human MI/MITI-coder validation** — an oracle can be perfectly repeatable
and consistently wrong. That remains the strongest further addition (costs Lior-time, not API
budget), and no ICC substitutes for it.

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

1. **Every contrast keeps its sign — 6/6.** PTO@10 − GRPO@10: Q1 +0.77 (primary +0.53), Q2 +0.45
   (+0.48), MICI −0.22 (−0.35, lower = better). PTO@10 − PTO Base: all positive under both judges.
   The decoupled judge *widens* the headline Q1 gap rather than shrinking it. **The PTO-vs-GRPO
   result is not an artifact of the patient and the grader sharing a model.**
2. **Rank agreement on Q1/Q2 is high**: r 0.84–0.88 (Q1) and 0.80–0.86 (Q2) against an attenuation
   ceiling of ~0.98 — i.e. 82–89% of the agreement two raters of this reliability could reach.
3. **Haiku is systematically harsher** (Q1 −1.25 to −1.74, Q2 −1.09 to −1.32) and flags *more*
   MI-inconsistent behaviour (MICI +0.15 to +0.36). This is a **level** shift, which cancels in
   every contrast the thesis reports; absolute Q1/Q2 values are grader-specific and should never be
   compared across judges.

**The MICI caveat.** Per-conversation cross-judge agreement on MICI is weak (r 0.20–0.55, ρ
0.21–0.47) even though the primary oracle is self-consistent on it (ICC 0.90–0.96). Part of this is
statistical — `MICI_Rate` is a low, zero-inflated count-per-turn, and restriction of range
attenuates correlation — but the two families clearly do not count MI-inconsistent behaviours the
same way. So the **sycophancy claim should be stated at the contrast level** (both judges agree
GRPO@10 is more MI-inconsistent than PTO@10, and that MICI rises from base in both arms), **not as
a precise per-conversation rate**. A human-coded MICI sample is the fix (see §1).

Coverage limits unchanged: one alternative judge, 3 of 8 rubrics, 4 of 29 model states, and the
patient simulator is still `gpt-4o-mini` in every conversation — this decouples the *grader*, not
the *generator*.

## 3 · Training reward = outcome metric (circularity)
Q1+Q2 is **both** the training reward **and** a headline eval metric. "Q1+Q2 improved" is
therefore partly circular and cannot by itself demonstrate MI-skill gain. Q1+Q2 is best framed
as a **satisfaction/alliance proxy** (Q1 = session satisfaction, Q2 = working alliance /
relational communication — the lab's CLPsych-2024 LLM-evaluator prompts, see
`METRICS_REFERENCE.md` §1; 22 subjective items with endpoint-only Likert anchors — itself a
plausible *cause* of the observed reward-hacking, not only the optimiser). The honest
outcome axes are the ones **outside** the reward: `PCT`, `MICI`, the MITI technique ratios, and
the deterministic text metrics. See the confirmatory/exploratory split in `7_Stats` §0.

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
