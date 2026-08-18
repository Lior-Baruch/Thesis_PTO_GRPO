# Status — where the thesis stands

**THE single live copy of run status, headline numbers, and the cost constraint.** Every other doc
points here (see the Doc map in [CLAUDE.md](CLAUDE.md)). Keep this file short: it answers *where
things stand*, not *how they got here*. When an entry stops being current, move it to
[Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md](Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md) rather than
appending a new dated paragraph beneath the old one.

**Last updated 2026-08-19.**

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

## Where the artifacts live

The EDA is organised **by research question** (2026-08-18 reorg):
`Exp3_PTO_GRPO/eda/results/<top>/<sub>/{figures,tables}/[<judge>/]` with tops **`arms/`** (per-arm
descriptives, all four arms on one axis, one leaf per grader), **`lookahead/`** (RQ-i: reward ·
transfer · behaviour · mechanism · replication), **`method/contrast`**, **`compute/cost`**,
**`measurement/validity`** — the four contrast tops carry both graders side by side and have no
`<judge>/` level. Each top has a hand-authored `SUMMARY.md`; `results/INDEX.md` maps every family to
its notebook. Regenerate with `python eda/tools/render_results.py` (see
[eda/README.md](Exp3_PTO_GRPO/eda/README.md), whose Migration table maps every retired
`results/L0|L5/...` path). Owner paths quoted below are in that tree.

## ⚠ Read every contrast on the COMPUTE axis, not just the iteration axis

`eda_analysis/compute.py` reconstructs GPU-hours per iteration from artifact
mtimes, and it reframes the whole comparison. **Owner: `eda/results/compute/cost/tables/{compute_by_arm,iso_compute_contrast,budget_sweep_<contrast>_<judge>}.md`; figure `compute/cost/figures/compute_trajectory.png`.**

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
[eda/results/method/SUMMARY.md](Exp3_PTO_GRPO/eda/results/method/SUMMARY.md) (+ [arms/SUMMARY.md](Exp3_PTO_GRPO/eda/results/arms/SUMMARY.md)).

**Look-ahead (RQ-i) — the answer is method-dependent.**

- **On PTO, K=5 never *significantly* leads on the reward** across 11 matched iterations, under
  either grader — and **K=0 leads significantly** at iteration 6 (primary, +0.257 dz 0.42) and at
  5/6/8 under the held-out judge (dz 0.33–0.51), the edge carried by **Q2** (the ICLR poster's
  own Q2-only K finding, reversed). Endpoint null on both graders (4.307 vs 4.260 primary).
  Owner: `eda/results/lookahead/reward/tables/k_table1.md` (persona-paired, both graders in one
  frame — first built for the paper on 2026-08-18, promoted into the EDA the same day).
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
  `eda/results/lookahead/transfer/tables/k_retention_summary.md` (own-base + shared-reference kinds) + lookahead/SUMMARY.md.

**Look-ahead flips which method wins — at iteration 5 only the held-out grader can see it** (the
primary does see GRPO > PTO under K=5 one iteration earlier: iter 4 Q1Q2 dz −0.351 p_holm .024).
At iteration 5,
K=0 → PTO leads (+0.265 dz 0.355 p_holm .014); K=5 → GRPO leads (−0.219 dz 0.377 p_holm .005).
Difference-in-differences on the same 96 personas: Q1Q2 dz **0.525** p_holm **.0001**, Q1 0.473,
Q2 0.474, MITI 0.441 (all p_holm ≤ .0005). ⚠ **On the primary oracle the same interaction is null**
(largest dz 0.211, nothing survives Holm) — the grader that *was* the training reward cannot see it.

**Cross-K findings (2026-08-18; first built for the paper, now EDA-owned under `eda/results/lookahead/` + `compute/`).**
- **The ICLR ordering reproduces on its own transcripts under the modern grader.** Re-scoring the
  poster's 1,440 Exp1 conversations (`eval_scores/_crossgen/`) with gpt-4o-mini keeps K=5 above K=0
  at 7/7 iterations (arm-level dz −0.54 vs −0.61 under GPT-3.5; Spearman 0.84 between the graders'
  15 model means) — the Exp3 null is a property of the **regime** (1B therapist, V3 patients,
  MCL=12, iterative regeneration, bf16), not of the judge.
- **Look-ahead rescales the training signal, it does not sharpen it.** Best–worst margin and
  within-group SD rise by the same ~1.4–1.8× (ratio-of-ratios 1.01–1.03); margin/SD sits at the
  8-draw expectation in every arm; ~half of PTO K=5's higher τ-yield at the base policy is that
  rescaling. At a **matched policy** (train_iter 1) look-ahead adds **no** reward faithfulness
  (K0−K5 +0.004 [−0.067, 0.074] PTO; +0.015 [−0.023, 0.057] GRPO).
- **What the K-step reward sees:** 19–23% of K=5 tails end early, almost always because the
  simulated patient closes; ended-early siblings score lower within group (dz −0.24/−0.26) and
  are ~23% less likely to be the argmax (RR 0.77/0.79), a pressure that grows over PTO training.
- **Session shape reverses by optimizer:** PTO K=5 sessions +8.3 utterances *longer* at iter 10
  (dz 0.55), GRPO K=5 −8.1 *shorter* at iter 5 (dz 0.53); both K=5 arms write longer turns; PTO_LA5
  is the only arm whose update pushes for length at every iteration (`w_len` +49.5 … +7.0).
- **The ICLR 'lowest SD = more stable' claim fails**: PTO K=5 is *more* dispersed than K=0 at 10/10
  iterations on the primary (Pitman–Morgan sig at 4); SD is a ceiling artefact (Spearman(mean, SD)
  −0.87 over 35 states, the cooperative third saturating ≥ 4.5) and absent under the held-out judge.
- **WAI-SR composition** shifts from Bond (K=0) to Goal/Task (K=5) on both graders (bond-excess
  K0−K5 +0.22/+0.27, dz ≈ 0.44); the held-out judge puts K=0's late Q2 gain on the two emotional
  self-disclosure items (+1.1 over K=5, dz > 1); PCT change-talk rises under K=5 in Warms-up personas.

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
temp 1.2, M=G=8). [arms/SUMMARY.md](Exp3_PTO_GRPO/eda/results/arms/SUMMARY.md) (training-signal section).

**The reward-hack is a compounding loop, not a hard pull.** Per-iteration *selection* pressure on
affirmation is ≈0.01 → 0.10, while what the policy *generates* moves 0.02 → 0.54 (GRPO) / 0.04 →
0.57 (PTO); questions collapse 0.71 → 0.06.

## Measurement validity

Oracle **ICC(2,1) 0.86–0.99**; the decoupled second judge (**Claude Haiku 4.5**, different family,
never played the patient) reproduces **18/18** anchor contrasts with the same sign and **88.4%** of
all 5,928 arm×metric contrasts. **Gain retention** is the load-bearing reward-hacking evidence
(Q1 retention PTO@10 0.80 vs GRPO@10 0.28, non-overlapping). ⚠ The retired **L5 view's retention table was
a silent 0-byte file** until 2026-08-18 (hardcoded L0 reference base absent from the K=5 view);
fixed, re-rendered (now `lookahead/transfer/tables/k_retention.md`, every reference kind), and `save_table` writes an explicit marker for any empty frame so this
failure class is visible in the render log.

⚠ **Standing caveats** — see [eda/results/LIMITATIONS.md](Exp3_PTO_GRPO/eda/results/LIMITATIONS.md):
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
is needed — therapist decoding is unseeded, and `code/tools/generate_eval_convs.py --conv-dir` keeps the
replicate out of the primary partition.

Cost: ~**$11.4** (6,000 patient calls ≈ $1.6 + 5 × 96 × 8 = 3,840 scoring calls per grader ≈ $1.2
primary + $8.7 Haiku, both batched) and **1.06 A100-hours**, or 4.2 free local hours at
`--batch-size 6`. Two decisive outcomes: replicates within |dz| ≲ 0.15 retire the
endpoint-fragility objection thesis-wide, or GRPO LA0's I9/I10 collapse fails to reproduce and the
headline PTO-vs-GRPO conclusion changes.

**Not recommended: extending GRPO LA5 to iteration 10** (~$118 + 23–34 A100-h). The compute axis
shows it would push that arm to ~50 GPU-h against LA0's 28, making the arms *less* comparable.

## Write-up decisions already made

- **One live paper: [`papers/2026_lookahead_pto_grpo/`](papers/2026_lookahead_pto_grpo/)** ("Same
  Lever, Different Optimizer") — both optimizers × both K × both graders × both cost axes, with the
  ICLR claims re-tested and the poster's transcripts re-scored. The two earlier drafts were **retired
  to `papers/archive/`** on 2026-08-18; their behavioural finding is the live draft's §6 and their
  `NUMBERS.md` traps still bind. Body is ~10 pages in ACL review mode; the README lists what to
  demote for an 8-page venue.
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
