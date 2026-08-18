# Status — where the thesis stands

**THE single live copy of run status, headline numbers, and the cost constraint.** Every other doc
points here (see the Doc map in [CLAUDE.md](CLAUDE.md)). Keep this file short: it answers *where
things stand*, not *how they got here*. When an entry stops being current, move it to
[Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md](Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md) rather than
appending a new dated paragraph beneath the old one.

**Last updated 2026-08-18.**

## Run status — all four arms trained and fully scored

| Arm | Iterations | Scored (both graders) | GPU-h |
|---|---|---|---|
| **PTO LA0** | 1–10 | ✅ 0–10 | **8.1** |
| **PTO LA5** | 1–10 | ✅ 0–10 | **19.7** |
| **GRPO LA0** | 1–10 | ✅ 0–10 | **27.9** |
| **GRPO LA5** | 1–5 (stopped) | ✅ 0–5 | **27.1** |

**39 scored model states** (11 + 6 + 11 + 11), and the score lake is a **full grid on both graders** —
39 × 8 rubrics × 96 personas = **29,952 cells each**. No arm is thin any more; RQ-i is a real
K×method comparison.

GRPO LA5 was stopped **~2 minutes into iteration 6**: `iteration_6/` holds one optimizer step
(`completions_00001.parquet`) and tb_logs, no adapter and no checkpoint. Nothing depends on it —
`model_iter_5` is the last complete policy — but a resume restarts iteration 6 from step 1.

## ⚠ Read every contrast on the COMPUTE axis, not just the iteration axis

`eda_analysis/compute.py` reconstructs GPU-hours per iteration from artifact
mtimes, and it reframes the whole comparison. **Owner: `results/L5/tables/7_stats/*/compute_by_arm.md`,
`iso_compute_contrast.md`, `budget_sweep.md`; figure `compute_trajectory.png`.**

- **The two GRPO arms are budget-matched to within 3%** — 27.08 vs 27.91 GPU-h — despite one
  running twice the iterations. "GRPO LA5 only reached iteration 5" is a statement about iteration
  count, **not about spend**, and every matched-*iteration* table hands K=5 ~2× the compute per cell.
- **PTO LA0 reaches iteration 10 for 8.1 GPU-h — 27.91 / 8.12 = 3.4× cheaper than GRPO LA0's ten.**
  It also scores higher. On the compute axis PTO dominates GRPO outright; that is a stronger claim
  than the matched-iteration one and it is not grader-dependent.
- **Look-ahead costs ~1.9× per optimizer step**, not the 2.4–3.0× previously recorded (median
  ratios 1.96 / 1.96 / 1.91 at iterations 3 / 4 / 5). The old figure came from iteration 1 alone,
  which ran at `LOOKAHEAD_SUB_BATCH_SIZE=64` and carried 12 API-tail steps > 500 s.
- ⚠ **The lever's sign is a function of budget, and the two budget framings disagree on the
  primary oracle.** `budget_sweep` (each arm at its *best checkpoint within budget*) vs
  `iso_compute_contrast` (each arm frozen at a fixed endpoint) are different questions:

  | budget | `budget_sweep` primary | `budget_sweep` held-out |
  |---|---|---|
  | 7.8 h | −0.09, p .41 (ns) | −0.08, p .42 (ns) |
  | 13.3 h | **−0.74**, p <.001 | **−0.78**, p <.001 |
  | 18.3 h | −0.28, p .013 | −0.11, p .35 (ns) |
  | 23–27 h | +0.07, p **.79 (ns)** | **+0.31**, p .007 |

  So K=5 is *clearly worse* only around **13 GPU-h**; it is null at 7.8 and (held-out) 18.3.
  ⚠ **At the top budget the primary oracle says NULL under `budget_sweep` (dz 0.07) while
  `iso_compute_contrast` says +0.289 (p_holm .018)** — because the fixed endpoint freezes
  `GRPO_LA0` at iteration 10, *after* its 4.082 → 3.753 regression (−0.33, larger than the whole
  +0.289). Only the **held-out judge** puts K=5 ahead under both framings. Quote the framing you
  mean, and prefer `budget_sweep` when the comparator arm is past its peak.

## Headline results

**Method (RQ-ii).** PTO beats GRPO at the matched 10-iteration endpoint — Q1+Q2 **4.26 vs 3.75**,
paired **+0.51, dz 0.73**. GRPO peaks at iter 8 (4.08) then regresses into sycophancy (MICI endpoint
0.84 vs PTO 0.49); PTO climbs stably. At **matched budget (8.1 GPU-h)** PTO LA0 @10 beats GRPO LA0 @3
on Q1+Q2 (+0.266 dz 0.529 primary, +0.230 dz 0.456 held-out, both p_holm ≤ .0002) — ⚠ but is **worse**
on MICI there (+0.261 dz 0.90 / +0.418 dz 1.28), because at equal spend PTO has trained ten
iterations to GRPO's three and is further along the reward-hacking curve. Narrative:
[eda/results/L0/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md).

**Look-ahead (RQ-i) — the answer is method-dependent.**

- **On PTO, K=5 never leads on the reward** across 11 matched iterations, under either grader.
- **On GRPO, K=5 does lead**, on both graders: Q1+Q2 at iteration 4 (Δ 0.115 dz 0.248 p_holm .037
  primary; Δ 0.233 dz 0.374 p_holm .005 held-out) and at iteration 5 under the held-out judge
  (Δ 0.311 dz 0.429 p_holm .006; primary Δ 0.070 ns).
- **At matched budget the GRPO gain is larger and MICI reverses sign**: Q1+Q2 +0.289 dz 0.359
  p_holm **.018** (primary) / +0.540 dz **0.838** p_holm **<.001** (held-out); MICI −0.497
  dz **−1.339** / −0.403 dz **−1.228**, both p_holm <.001 — K=5 is far less MI-inconsistent
  **per therapist turn** at equal spend. ⚠ `MICI` is `MICI_Rate` (acts ÷ therapist turns) and the
  denominators differ here (11.31 vs 12.75 turns), so name the unit: the per-SESSION counts point
  the same way and are ~13× larger — `MICI_BehaviorTotal` **3.45 vs 9.87** acts (primary),
  **7.14 vs 13.00** (held-out). (Holm across the 9 rubrics at that budget, per the rendered
  `iso_compute_contrast.md`.)

- **Retention by K (NEW 2026-08-18 — the L5 retention table existed but was EMPTY until then).**
  Gain retention (Δ held-out / Δ primary vs base) is **method-dependent in the same direction as
  the reward**: GRPO K=5 retains its FULL Q1 gain (1.08 [0.94, 1.27] at iter 5 vs K=0's 0.73
  [0.57, 0.92] — disjoint), while PTO K=5 retains the same or less (Q1 0.72 vs 0.80 overlapping;
  **Q2 0.56 vs 0.85 disjoint, K=5 worse**). Owner:
  `results/L5/tables/8_measurement/multijudge_gain_retention.md` + L5/SUMMARY §6b.

**Look-ahead flips which method wins — but only the held-out grader can see it.** At iteration 5,
K=0 → PTO leads (+0.265 dz 0.355 p_holm .014); K=5 → GRPO leads (−0.219 dz 0.377 p_holm .005).
Difference-in-differences on the same 96 personas: Q1Q2 dz **0.525** p_holm **.0001**, Q1 0.473,
Q2 0.474, MITI 0.441 (all p_holm ≤ .0005). ⚠ **On the primary oracle the same interaction is null**
(largest dz 0.211, nothing survives Holm) — the grader that *was* the training reward cannot see it.

**The reward gain is verbosity-shaped.** Coded MI acts per 1,000 therapist characters roughly halve
under K=5 at iteration 5 (4.08 → 2.22 primary, dz 0.717; 4.97 → 2.59 held-out, dz 0.650) — fewer,
longer turns with less MI content per word. Verbosity is a **training-depth** channel, not a
look-ahead one: chars/therapist-turn go GRPO LA0 @5 **394** → @10 **905**, vs GRPO LA5 @5 **678**.

**Substitution replicates on a second method, denominator-free.** Over-praise as a *share of
MI-inconsistent acts*: 0.178 → 0.086 (primary, dz 0.344, p_holm .0045) and 0.182 → 0.063 (held-out,
dz 0.722, p_holm <.0001), while the overall MI-inconsistent share is flat or slightly worse. ⚠ The
per-turn rate, the per-session count and the per-1k-character measure **disagree in direction** on
GRPO; only the share measure has no moving denominator (turns −26%, chars/turn +72%).

**The PTO/GRPO gap is the STATE DISTRIBUTION, not the loss.** Swapping the weighting rule on the
*same* groups barely moves the update direction (0.908 / 0.988); holding the rule fixed across each
method's *own* groups leaves them as far apart as ever (0.397 / 0.324 corrected, vs 0.317 as
trained). Frame it as the state distribution the two methods train on — GRPO slices an on-policy
rollout, PTO grows a best-of-M reranked trunk (closer to expert iteration) — **not** as
DPO-vs-group-relative, and **not** as "exploration" (candidate sampling is matched by construction,
temp 1.2, M=G=8). [L0/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md) §6.

**The reward-hack is a compounding loop, not a hard pull.** Per-iteration *selection* pressure on
affirmation is ≈0.01 → 0.10, while what the policy *generates* moves 0.02 → 0.54 (GRPO) / 0.04 →
0.57 (PTO); questions collapse 0.71 → 0.06.

## Measurement validity

Oracle **ICC(2,1) 0.86–0.99**; the decoupled second judge (**Claude Haiku 4.5**, different family,
never played the patient) reproduces **18/18** anchor contrasts with the same sign and **88.4%** of
all 5,928 arm×metric contrasts. **Gain retention** is the load-bearing reward-hacking evidence
(Q1 retention PTO@10 0.80 vs GRPO@10 0.28, non-overlapping). ⚠ The **L5 view's retention table was
a silent 0-byte file** until 2026-08-18 (hardcoded L0 reference base absent from the K=5 view);
fixed, re-rendered, and `save_table` now writes an explicit marker for any empty frame so this
failure class is visible in the render log.

⚠ **Standing caveats** — see [eda/docs/LIMITATIONS.md](Exp3_PTO_GRPO/eda/docs/LIMITATIONS.md):
**MITI** dependability is 0.55 and **MICI** 0.63 off one judge, and those two instruments carry the
channel results; there is **no channel-level ICC at all**, **no repeatability rep for any K=5
state**, and **no replicate draw for any trained checkpoint** (therapist decoding is unseeded, so
no conversation set here is reproducible). All 96 personas are used for both training and eval at
every iteration, so every number is in-sample w.r.t. the patient distribution.

## Cost constraint (binding)

OpenAI + Anthropic spend is ~**$317** (~$312 + $4.50 to score the four new GRPO LA5 states on both
graders: 4 × 8 × 96 = 3,072 calls per grader, 0 errors).

- Cost is dominated by oracle scoring + (at K=5) look-ahead patient calls, both ∝ candidate count
  (`prompts×G` / `branch-points×M`) × iterations.
- Prompt caching is already maxed (~50% off the oracle's fixed prefix), so **the only lever is call
  COUNT**: cap `NUM_ITERATIONS` ~5–6, drop `M`/`G` 8→4, (PTO) lower `GREEDY_TRUNK_TARGET_LEN`.
  Keep **K** (the RQ-i variable) and the **gpt-4o-mini oracle** (the measurement instrument) fixed.
- ⚠ **Price a Haiku sweep off `judge_plan.sweep_report(..., receipt=(42.0, 22272))`.** The
  receipt-calibrated basis put the last sweep at $2.90 where the char estimator said $5.33.
  Never quote judge cost from memory.

## Next step

**A second independent 96-conversation draw from 5 adapters** — `GRPO_LA0_I3/I8/I10`,
`GRPO_LA5_I4/I5`. Every contested endpoint in the thesis is currently a **single draw**, and the
only noise floor that exists is at the base (4 independent draws of the identical base policy:
6 pairs × 9 metrics = 54 same-policy contrasts, **0 reaching even uncorrected p < .05, max |dz|
0.128 primary / 0.147 held-out**). No code change
is needed — therapist decoding is unseeded, and `generate_eval_convs.py --conv-dir` keeps the
replicate out of the primary partition.

Cost: ~**$11.4** (6,000 patient calls ≈ $1.6 + 5 × 96 × 8 = 3,840 scoring calls per grader ≈ $1.2
primary + $8.7 Haiku, both batched) and **1.06 A100-hours**, or 4.2 free local hours at
`--batch-size 6`. Two decisive outcomes: replicates within |dz| ≲ 0.15 retire the
endpoint-fragility objection thesis-wide, or GRPO LA0's I9/I10 collapse fails to reproduce and the
headline PTO-vs-GRPO conclusion changes.

**Not recommended: extending GRPO LA5 to iteration 10** (~$118 + 23–34 A100-h). The compute axis
shows it would push that arm to ~50 GPU-h against LA0's 28, making the arms *less* comparable.

## Write-up decisions already made

- **Report the method comparison on BOTH axes.** Matched-iteration and matched-budget answer
  different questions and PTO wins both, but for different reasons — say which axis a number is on.
- Make the look-ahead claim **about the lever, never about convergence**.
- **State the look-ahead MI-consistency result at the CHANNEL level, not as a total**, and prefer
  the *share* of MI-inconsistent acts to any per-turn or per-session figure — the arms differ in
  both turn count and turn length, in method-dependent directions.
- Report all 8 instruments flat. The "orthogonal axes" framing is retired — PCT correlates ρ≈0.79–0.94
  with the rubrics.
- **Report the head-to-head both final-vs-final AND best-vs-best.** At best-vs-best PTO still leads
  on Q1+Q2 (+0.18, dz 0.30, p .010) but the **MITI and MICI gaps stop being significant** — so the
  sycophancy separation is a property of GRPO's post-peak run-off, not of its best state.
- **Gain-retention disjointness is metric-dependent, not "iterations 9–10".** Disjoint PTO/GRPO
  CIs occur at Q1 {6, 9, 10}, MITI {5, 6, 7, 8, 9}, Q2 {9}, MI-SAT {9}, MICI {10}, and **never**
  for CSQ-8 or WAI-SR — no metric gives {9, 10}. The defensible statement remains the ordering
  (PTO above GRPO at every iteration from 4 on); name the metric before naming an iteration.
- **Any claim that look-ahead reduces MI-inconsistency must name its axis and its grader.** It is
  false at matched iteration on the per-turn rate, true at matched budget on both graders, and the
  aggregate-vs-channel distinction survives while the aggregate does not.
