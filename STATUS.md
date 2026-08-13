# Status — where the thesis stands

**THE single live copy of run status, headline numbers, and the cost constraint.** Every other doc
points here (see the Doc map in [CLAUDE.md](CLAUDE.md)). Keep this file short: it answers *where
things stand*, not *how they got here*. When an entry stops being current, move it to
[Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md](Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md) rather than
appending a new dated paragraph beneath the old one.

**Last updated 2026-08-13.**

## Run status

| Arm | Trained | Scored (both graders) | Notes |
|---|---|---|---|
| **PTO LA0** | iters 1–10 | ✅ 0–10 | complete |
| **GRPO LA0** | iters 1–10 | ✅ 0–10 | complete, re-scored |
| **PTO LA5** | iters 1–8 | ✅ 0–8 | **complete at its configured `NUM_ITERATIONS=8` endpoint** |
| **GRPO LA5** | iter 1 | ✅ 0–1 | **thin — this is the gap** |

**33 scored model states.** Raising `NUM_ITERATIONS` back to 10 later is safe: `model_iter_k` seeds
are `seed+k+1` from either the loop or the post-loop pass, so the persona shuffle matches and
per-CSV resume skips what already exists.

## Headline results

- **PTO beats GRPO** at the matched 10-iter endpoint — Q1+Q2 **4.26 vs 3.75**, paired **+0.51,
  dz 0.73**. GRPO peaks at iter 8 (4.08) then regresses into sycophancy (MICI endpoint 0.84 vs PTO
  0.49); PTO climbs stably. Narrative + tables:
  [eda/results/L0/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md).
- **Look-ahead never leads (RQ-i).** K=5 does not lead at **any** of 8 matched iterations, under
  **either** grader. Endpoint levels: primary 4.221 (K=0) vs 4.144 (K=5); held-out judge 2.895 vs
  2.710, Holm-significant. ⚠ **Within PTO only** — GRPO LA5 has just iter 1, so this is not yet a
  K×method comparison. Narrative: [eda/results/L5/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L5/SUMMARY.md);
  full mechanistic evidence: the `project-lookahead-negative-result` memory.
- **The PTO/GRPO gap is exploration, not the loss.** Swapping the weighting rule on the *same*
  groups barely moves the update direction (0.908 / 0.988); holding the rule fixed across each
  method's *own* groups leaves them as far apart as ever (0.397 / 0.324 corrected, vs 0.317 as
  trained). **Frame the thesis comparison as exploration, not DPO-vs-group-relative.**
  [L0/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md) §6.
- **The reward-hack is a compounding loop, not a hard pull.** Per-iteration *selection* pressure on
  affirmation is ≈0.01 → 0.10, while what the policy *generates* moves 0.02 → 0.54 (GRPO) /
  0.04 → 0.57 (PTO); questions collapse 0.71 → 0.06.

## Measurement validity

The instrument is measured, not assumed. Oracle **ICC(2,1) 0.86–0.99**; a decoupled second judge
(**Claude Haiku 4.5**, different family, never played the patient) reproduces **18/18** anchor
contrasts with the same sign and **88.3%** of all 1,848 arm×metric contrasts (98.9% at |Δ|≥0.50).
**Gain retention** is the load-bearing reward-hacking evidence (Q1 retention PTO@10 0.80 vs
GRPO@10 0.28, non-overlapping).

⚠ **Two standing caveats:** **MITI** dependability is 0.65 off one judge — treat MITI arm
differences as provisional. **MICI** cross-judge agreement is weak (r 0.20–0.55), so the sycophancy
claim holds at the *contrast* level, not as a precise rate. Both are thesis limitations, not
footnotes — see [eda/docs/LIMITATIONS.md](Exp3_PTO_GRPO/eda/docs/LIMITATIONS.md) §1–§3.

## Cost constraint (binding)

OpenAI + Anthropic spend is ~**$310** and is the limiting factor on every remaining decision.

- Cost is dominated by oracle scoring + (at K=5) look-ahead patient calls, both ∝ candidate count
  (`prompts×G` / `branch-points×M`) × iterations.
- Prompt caching is already maxed (~50% off the oracle's fixed prefix), so **the only lever is call
  COUNT**: cap `NUM_ITERATIONS` ~5–6 (curves plateau by iter ~4), drop `M`/`G` 8→4, (PTO) lower
  `GREEDY_TRUNK_TARGET_LEN`. Keep **K** (the RQ-i variable) and the **gpt-4o-mini oracle** (the
  measurement instrument) fixed.
- ⚠ **Price a Haiku sweep off `judge_plan.sweep_report(..., receipt=(42.0, 22272))`.** The
  receipt-calibrated basis put iter 8 at $0.72 where the char estimator said $1.33 and a pro-rata
  guess said $1.87. Never quote judge cost from memory — see the `project-openai-cost-constraint`
  memory.

## Next step

**Score GRPO LA5 beyond iter 1.** PTO LA5 is done at its endpoint, so more PTO K=5 iterations buy
nothing — eight matched points already say the same thing eight times. Without GRPO LA5 the
look-ahead result is *within PTO* and cannot be stated as a K×method comparison. Spend there before
extending PTO LA5 to 10.

## Write-up decisions already made

- Make the look-ahead claim **about the lever, never about convergence** — the old "arms tie at
  iter 5" reading was a primary-oracle artifact that did not survive iter 6, and the held-out judge
  never saw a tie.
- **Drop "look-ahead costs MI-consistency"** from the write-up. The MICI tilt reverses sign at
  iters 7–8; a claim that flips sign across the run is a claim about *when*, not about the lever.
- Report all 8 instruments flat. The "orthogonal axes" framing is retired — PCT correlates ρ≈0.79–0.94
  with the rubrics.
