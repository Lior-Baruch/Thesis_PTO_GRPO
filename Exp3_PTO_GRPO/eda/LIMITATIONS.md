# Exp3 — measurement & inference limitations (for the thesis write-up)

Deliberately-scoped limitations of the Exp3 evaluation. These are **documented, not fixed**
(see the CLAUDE.md backlog for what was fixed). Each names where in the notebooks the reader
meets it.

## 1 · Judge reliability is not measured (single scoring)
Every conversation is scored **once** by the oracle, with `temperature=0.1, seed=42`. That
setup *freezes* the judge's bias for reproducibility but does **not measure** it — there is no
per-item variance, ICC, or human-vs-oracle κ. So the questionnaire scores should be read as a
single consistent instrument, not as estimates with a known measurement error. A held-out
subset re-scored 3–5× (variance/ICC) and a human MI/MITI-coder validation on a small sample
would be the strongest future addition to the measurement-validity section; both cost oracle
budget and are deferred (OpenAI spend is a binding constraint, ~$300).

## 2 · Shared-model (patient = oracle) coupling
The simulated patient **and** the grading oracle are the **same** model
(`gpt-4o-mini-2024-07-18`). Several instruments (WAI-SR, CSQ-8, MI-SAT, PCT) rate the session
"from the patient's perspective," so the generator and the evaluator are coupled — this can
inflate patient-perspective alliance/satisfaction. The reward-hacking argument in
`3_Mechanism` §4 is built to survive this: its load-bearing evidence is the **deterministic
text metrics** (turn length, loop %, question rate) that use no oracle at all, with the
un-rewarded oracle axes (MICI, PCT, MITI ratios) as corroboration. A decoupled second judge
(different family/size) on a subset would let us report two-judge agreement; deferred with #1.

## 3 · Training reward = outcome metric (circularity)
Q1+Q2 is **both** the training reward **and** a headline eval metric. "Q1+Q2 improved" is
therefore partly circular and cannot by itself demonstrate MI-skill gain. Q1+Q2 is best framed
as a **warmth/satisfaction proxy** (a 22-item warmth halo with endpoint-only Likert anchors —
itself a plausible *cause* of the observed reward-hacking, not only the optimiser). The honest
outcome axes are the ones **outside** the reward: `PCT`, `MICI`, the MITI technique ratios, and
the deterministic text metrics. See the confirmatory/exploratory split in `6_Stats` §0.

## 4 · PCT is not the clean orthogonal axis intended
Empirically `PCT` (patient change-talk proportion) loads **with** the warmth family
(ρ≈0.79–0.94; high PC1 loading), so it does not isolate MI *technique*. The genuine second
factor is `MICI ↓` + the MITI ratios (`R:Q`/`%CR`/`%MICO`). Reported as a finding in
`3_Mechanism` §3 rather than hidden.

## 5 · Look-ahead (K=0 vs K=5) is descriptive only
The LA5 arms are thin (PTO_LA5 = 4 scored iters, GRPO_LA5 = 1), so every K contrast
(`4_Training` §4, `5_Preference` §2, `6_Stats` §4) is **hypothesis-generating, not inferential**
— banners mark these in-notebook. The confirmatory PTO-vs-GRPO result is at K=0 and is
unaffected.

## 6 · Multiplicity is corrected within families, not across
Holm/BH corrections apply **within** each family (rubrics within one matched contrast, or
iterations within one arm-vs-base sweep) and are **not** pooled across the dozens of families
in the EDA. The confirmatory/exploratory split (`6_Stats` §0) is what keeps this honest: treat
only the small pre-registered confirmatory set as tested claims; the rest are descriptive.
