# Status — where the thesis stands

**THE single live copy of run status, headline numbers, and the cost constraint.** Every other doc
points here (see the Doc map in [CLAUDE.md](CLAUDE.md)). Keep this file short: it answers *where
things stand*, not *how they got here*. When an entry stops being current, move it to
[Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md](Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md) rather than
appending a new dated paragraph beneath the old one.

**Last updated 2026-08-21.**

## Run status — three arms complete; GRPO LA5 STOPPED mid-iteration-7

| Arm | Adapters | Scored (both graders) | GPU-h |
|---|---|---|---|
| **PTO LA0** | 1–10 | ✅ Base + I1–I10 = 11 | **8.119** |
| **PTO LA5** | 1–10 | ✅ Base + I1–I10 = 11 | **19.681** |
| **GRPO LA0** | 1–10 | ✅ Base + I1–I10 = 11 | **27.906** |
| **GRPO LA5** | **1–6; iteration 7 stalled** | ✅ Base + I1–I6 = 7 | **30.528** through I6 · **32.424** billed |

**40 scored model states** (11 + 11 + 11 + 7) — a **full grid on both graders**,
40 × 8 rubrics × 96 personas = **30,720 cells each**. `_selfcheck`'s `score coverage (disk vs lake)`
reports **40/40 states complete**. Nothing is waiting to be scored.

⚠ **GRPO LA5 is not training. It stopped, and it has not written anything since 2026-08-20.**
It is *not* "in flight" — that reading is retired. Iteration 7 needs **106** optimizer steps
(`iteration_7/training/checkpoint-40/trainer_state.json`: `max_steps = 106`, `global_step = 40`), so
**40 / 106 = 37.7%** of it is on disk and no `iteration_7/adapter/` exists — which is how
`resolve_start_state` defines "not done".

### Why it stopped — two different failures, four Colab sessions

Reconstructed from the four TensorBoard event files in `iteration_7/training/tb_logs/` (one per
Colab VM) against the artifacts that actually landed, plus the W&B session logs under
`G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\code\GRPO_Exp3\wandb\run-*_iter7\files\output.log`.

| # | started (UTC) | host | steps trained | steps persisted | outcome |
|---|---|---|---|---|---|
| 1 | 08-19 14:22 | `252f11bc05b1` | **1–103** | 1–30 | writes stopped at `checkpoint-30`; **73 steps lost** |
| 2 | 08-20 08:38 | `38c8e9447bf8` | 0 | 0 | **OpenAI spend limit** |
| 3 | 08-20 10:39 | `7393ee7109e2` | 0 | 0 | **OpenAI spend limit** |
| 4 | 08-20 11:42 | `60b484945ba8` | **31–99** | 31–40 | writes stopped at `checkpoint-40`; **59 steps lost** |

- **Failure A — the OpenAI organization spend cap.** Sessions 2 and 3 died at their first optimizer
  step. **384 of 395 log lines** in each are
  `Error code: 429 … 'code': 'organization_spend_limit_exceeded'`, on *both* patient and oracle
  calls, ending `Oracle batch: 0/128 succeeded (0%), 128 rewards → None`. The cap cleared by ~11:42.
  **This is the cost constraint stopping being theoretical** — see § Cost constraint.
- **Failure B — Drive stopped accepting NEW files while appends kept working.** Sessions 1 and 4
  both had `Oracle batch: 128/128 succeeded (100%)` — no API problem. Their TensorBoard streams kept
  flushing scalars for **3 h 28 m** and **2 h 47 m** after the last parquet/checkpoint was created.
  Combined: 73 + 59 = **132 optimizer steps ≈ 6.25 h of K=5 GRPO training computed and discarded**.
  Session 1 reached step 103 of 106 — **3 steps short** — and banked none of it.
- ⚠ **Nothing on disk names the cause of Failure B.** drivefs wedge, quota, VM preemption and a
  swallowed save exception are all still open. Realized persistence yield per session is
  30 / 0 / 0 / 10 steps.

⚠ **Read GRPO LA5's cost as 30.528, not 32.424.** `compute.py` bills any iteration with training
artifacts and has **no adapter gate** (it excludes only at `< 3` timed steps), so
`compute_by_arm.md` reports `last_iter 7, n_iters 7` against **six** adapters and
`gpu_h_per_iter` 32.424/7 = 4.632 instead of 30.528/6 = 5.088 — a 9.0% understated denominator.
Both figures are in `compute/cost/tables/compute_by_iteration.md`; take them from there.

## Where the artifacts live

The EDA is organised **by research question**:
`Exp3_PTO_GRPO/eda/results/<top>/<sub>/{figures,tables}/[<judge>/]` with tops **`arms/`** (per-arm
descriptives, one leaf per grader), **`lookahead/`** (RQ-i: reward · transfer · behaviour ·
mechanism · replication), **`method/contrast`**, **`compute/cost`**, **`measurement/validity`** —
the four contrast tops carry both graders side by side and have no `<judge>/` level. Each top has a
hand-authored `SUMMARY.md`; `results/INDEX.md` maps every family to its notebook. Regenerate with
`python tools/render_results.py` from `Exp3_PTO_GRPO/eda/`.

✅ **The tree is CURRENT as of 2026-08-21** — a full re-render (6 units / 21 notebook executions,
no failures, 2,460 s) ran today and every family now includes `GRPOExp3_LA5_I6`. The standing
"re-render before reading any number" warning is retired until the next scoring pass.

⚠ **But 33 freshly rendered files still assert "GRPO_LA5 is right-censored at iteration 5 (its
budget stops at 27.08 GPU-h)"** — it is a hardcoded `CENSOR_NOTE` string in eight EDA modules
(`compute.py:636,1190`, `faithfulness.py:130`, `instruments.py:122`, `lookahead.py:84`,
`replication.py:108`, `transfer.py:75`, `plotting/tails.py:54`). The **data** is right; the prose
rendered beside it is not. `compute_by_arm.md` now says `n_iters 7 / 32.424` next to a `CAPTIONS.md`
that says 27.08. **Patch the constants before quoting any caption.**

⚠ **One live wrong VALUE, not a caption.** `faithfulness.py:110`'s `SERIES` pins
`("GRPO_LA0", "1-5", frozenset({0,1,2,3,4}))` but leaves `("GRPO_LA5", "1-5", None)` = full support.
Now that `iteration_6/eda/generations.jsonl` exists (1,172 rows), the column **labelled** `iters 1-5`
pools 1–6 for GRPO_LA5 only: 141,487 pairs against GRPO_LA0's genuine 128,176. The like-for-like
comparison in `lookahead/mechanism/tables/faithfulness_curve*.md` is broken until that is fixed.

## ⚠ Read every contrast on the COMPUTE axis, not just the iteration axis

**Owner: `eda/results/compute/cost/tables/{compute_by_arm,compute_by_iteration,iso_compute_contrast,budget_sweep_<contrast>_<judge>}.md`; figure `compute/cost/figures/compute_trajectory.png`.**

- **The "two GRPO arms are budget-matched to within 3%" framing is DEAD under every option.** It was
  true at iteration 5 (27.078 vs 27.906 = 0.970). At the last completed adapter it is
  30.528 / 27.906 = **1.094 → +9.4%**; including the stalled partial, 32.424 / 27.906 = **1.162 →
  +16.2%**. Any iso-compute sentence written before 2026-08-20 assumed the matched budget.
- **PTO LA0 reaches iteration 10 for 8.119 GPU-h — 27.906 / 8.119 = 3.44× cheaper than GRPO LA0's
  ten.** It also scores higher. On the compute axis PTO dominates GRPO outright; that is a stronger
  claim than the matched-iteration one and it is not grader-dependent.
- **Look-ahead costs ~1.9× per optimizer step** (median ratios 1.965 / 1.962 / 1.911 at iterations
  3 / 4 / 5; `step_multiplier.md`).
- ⚠ **Quote `budget_sweep` (each arm at its best checkpoint within budget), not a single
  `iso_compute_contrast` row (each arm frozen at a fixed endpoint)** — they answer different
  questions and disagree at the top budget, because the fixed endpoint freezes `GRPO_LA0` at
  iteration 10, *after* its 4.082 → 3.753 regression. Prefer `budget_sweep` when the comparator arm
  is past its peak. Both need re-reading now that the top of GRPO LA5's budget moved from 27.078 to
  30.528.

## Headline results

**Method (RQ-ii).** PTO beats GRPO at the matched 10-iteration endpoint — Q1+Q2 **4.260 vs 3.753**.
GRPO K=0 peaks at iteration 8 (4.082) then regresses into sycophancy (MICI endpoint 0.838 vs PTO
0.491); PTO climbs stably. ⚠ **But GRPO K=5 @6 is now the strongest GRPO state on both graders** —
primary **4.229** (vs GRPO K=0's best 4.082), held-out **2.903**, which is the **highest "final" row
of all four arms on the held-out judge** (PTO K=0 2.866, PTO K=5 2.667, GRPO K=0 2.257). Its MICI is
0.281 primary. The PTO-over-GRPO headline is a statement about *matched iteration 10*, and GRPO K=5
has no iteration 10. Owner: `arms/outcomes/tables/<judge>/leaderboard_scorecard.md`.

**Look-ahead (RQ-i) — the answer is method-dependent, and iteration 6 makes it unambiguous.**
Sign convention: **+ = K=0 higher**. Owner: `lookahead/reward/tables/k_table1.md` (persona-paired,
both graders in one frame).

| iter | PTO · primary | PTO · held-out | GRPO · primary | GRPO · held-out |
|---:|---|---|---|---|
| 4 | +0.120 (0.20) | +0.123 (0.21) | −0.115 (−0.25)* | −0.233 (−0.37)** |
| 5 | −0.002 (−0.00) | +0.173 (0.33)* | −0.070 (−0.13) | −0.311 (−0.43)** |
| **6** | **+0.257 (0.42)\*\*\*** | **+0.343 (0.51)\*\*\*** | **−0.263 (−0.42)\*\*\*** | **−0.533 (−0.55)\*\*\*** |

At iteration 6 the lever is **significant on both graders in both optimizers, with opposite signs**:
K=0 wins on PTO, K=5 wins on GRPO. Across all 11 PTO iterations K=5 never significantly leads on
either grader; the PTO endpoint is null (4.307 vs 4.260 primary).

⚠ **RETIRED: "the primary oracle cannot see the K×method interaction."** That was an
iteration-≤5 statement. At iteration 6 the difference-in-differences on the same 96 personas is
**significant on the primary too** — Q1Q2 gap_K0 0.188 − gap_K5 (−0.332) = **+0.520, dz 0.605,
p_holm .000** (`lookahead/reward/tables/k_did.md`). The held-out judge remains **1.68× larger**
(0.876 / 0.520 = 1.68, dz 0.754) — so *"the second judge sees it more sharply"* survives;
*"the training grader is blind to it"* does not. **Re-scope that argument, do not retract it.**

**Retention by K.** Gain retention (Δ held-out / Δ primary vs base) is method-dependent in the same
direction as the reward: GRPO K=5 retains more of its Q1 gain than K=0, PTO K=5 the same or less
(Q2 disjoint, K=5 worse). Owner: `lookahead/transfer/tables/k_retention_summary.md`.
⚠ **Quote the reference kind.** `k_retention_summary.md` is own-base only; the shared-reference
figures live in `transfer.xlsx` sheet `k_retention`, and disjointness holds under only **2 of 5**
reference conventions. The pair "1.08 [0.94, 1.27] vs 0.73 [0.57, 0.92]" this file used to quote
matched no artifact — the cited table says **1.048 [0.913, 1.223] vs 0.786 [0.587, 1.003]**,
`cis_disjoint = False`.

**Cross-K findings.**
- **The ICLR ordering reproduces on its own transcripts under the modern grader.** Re-scoring the
  poster's 1,440 Exp1 conversations with gpt-4o-mini keeps K=5 above K=0 at 7/7 iterations — the
  Exp3 result is a property of the **regime** (1B therapist, V3 patients, MCL=12, iterative
  regeneration, bf16), not of the judge.
- **Look-ahead rescales the training signal, it does not sharpen it.** Best–worst margin and
  within-group SD rise by the same ~1.4–1.8× (ratio-of-ratios 1.01–1.03). At a **matched policy**
  (train_iter 1) look-ahead adds **no** reward faithfulness.
- **What the K-step reward sees:** 19–23% of K=5 tails end early, almost always because the
  simulated patient closes; ended-early siblings score lower within group and are ~23% less likely
  to be the argmax.
- **Session shape reverses by optimizer:** PTO K=5 sessions longer at iteration 10, GRPO K=5 shorter
  at 5; both K=5 arms write longer turns.
- **The ICLR 'lowest SD = more stable' claim fails**: SD is a ceiling artefact and absent under the
  held-out judge.

**The PTO/GRPO gap is the STATE DISTRIBUTION, not the loss.** Swapping the weighting rule on the
*same* groups barely moves the update direction (0.908 / 0.988); holding the rule fixed across each
method's *own* groups leaves them as far apart as ever. Frame it as the state distribution the two
methods train on — GRPO slices an on-policy rollout, PTO grows a best-of-M reranked trunk — **not**
as DPO-vs-group-relative, and **not** as "exploration" (candidate sampling is matched by
construction, temp 1.2, M = G = 8).

**The reward-hack is a compounding loop, not a hard pull.** Per-iteration *selection* pressure on
affirmation is ≈0.01 → 0.10, while what the policy *generates* moves 0.02 → 0.54 (GRPO) / 0.04 →
0.57 (PTO); questions collapse 0.71 → 0.06.

⚠ **Two statistics this file used to quote have NO owning table** and must be rendered or dropped
before they go in the write-up: coded MI acts **per 1,000 therapist characters** (no code computes
it — 0 hits for `per_1k|per1k|per_1000`), and the over-praise share as a **per-conversation mean**
(the one rendered table, `lookahead/behaviour/tables/k_mici_composition.md`, uses ratio-of-means:
0.193 → 0.088 primary, 0.187 → 0.061 held-out). Both reproduce; neither is auditable today.

## Measurement validity

Oracle **ICC(2,1) 0.86–0.99**; the decoupled second judge (**Claude Haiku 4.5**, different family,
never played the patient) reproduces **18/18** anchor contrasts with the same sign and **88.5%** of
all **6,240** arm×metric contrasts (= 8 metrics × C(40,2) = 8 × 780; ladder 88.5 / 94.5 / 97.4 /
99.5 as |Δ| rises). Owner: `measurement/validity/tables/multijudge_sign_preservation.md`.

⚠ **Dependability corrected.** At n_arms = 40, `multijudge_variance_components.md` gives
`dependability_k1` **MITI 0.622** and **MICI 0.845** — not the 0.55 / 0.63 this file and
`LIMITATIONS.md:319` carried, which match no artifact. MITI is still the weakest instrument and
still carries channel results; MICI is substantially better than advertised.

⚠ **Standing caveats** — see [eda/results/LIMITATIONS.md](Exp3_PTO_GRPO/eda/results/LIMITATIONS.md):
there is **no channel-level ICC at all**, **no repeatability rep for any K=5 state**, and **no
replicate draw for any trained checkpoint** (therapist decoding is unseeded, so no conversation set
here is reproducible). All 96 personas are used for both training and eval at every iteration, so
every number is in-sample w.r.t. the patient distribution.

## Cost constraint (binding — and it BOUND on 2026-08-20)

⚠ **The OpenAI organization spend limit was actually hit**, killing two Colab sessions outright
(see § Run status). This is no longer a projected constraint.

⚠ **The "~$317 spent" tally is INTERPRETATION with no artifact behind it, and is probably a
large understatement.** No billing export, invoice, or receipts file exists anywhere in the repo;
`compute.py` has zero dollar arithmetic. `meetings/build/build_meeting_deck.py` says
"≈$300 OpenAI + ≈$51 Anthropic" = **$351** from the same commit. An independent re-derivation off
`compute/cost/tables/api_calls.md` puts the gpt-4o-mini **training-oracle line alone** near **$400**,
before eval scoring, patient calls, and the entire Anthropic side. **Reconcile against the actual
vendor dashboard before quoting any figure.**

- Cost is dominated by oracle scoring + (at K=5) look-ahead patient calls, both ∝ candidate count
  (`prompts×G` / `branch-points×M`) × iterations.
- Prompt caching is already maxed (~50% off the oracle's fixed prefix), so **the only lever is call
  COUNT**: cap `NUM_ITERATIONS`, drop `M`/`G` 8→4, (PTO) lower `GREEDY_TRUNK_TARGET_LEN`.
  Keep **K** (the RQ-i variable) and the **gpt-4o-mini oracle** (the measurement instrument) fixed.
- **Scoring one state on both graders = $2.08** (768 calls per grader: ~$0.63 primary live +
  768 × $0.001886 = $1.45 Haiku batched). ⚠ The **$4.50** this file used to quote for four states
  was the double-discounted row: 4 × $2.08 = **$8.32**.
- ⚠ **Price a Haiku sweep with `judge_plan.sweep_report(..., receipt=(42.0, 22272))` and read the
  `batch=False` row.** `calibrate_from_receipt` divides the receipt by its call count then
  multiplies by `(1 − batch_discount)` — i.e. it assumes list price. The $42 receipt was itself a
  Message Batches sweep, so `batch=True` halves an already-discounted rate.
- ⚠ **There is no batched primary path.** `judge_batch.py` raises "Batch path is Anthropic-only",
  so any "$1.2 primary, batched" estimate is unachievable — price the primary live.

## Next step

**1. Decide GRPO LA5 — this is the blocking call.** Three options, all costed:

| | cost | consequence |
|---|---|---|
| **(a) Stop at 6, censor at 6** | $0 | I6 is the arm's best state on both graders and the top held-out "final" row overall. The K contrast and the DiD are both significant at 6 on both graders. Persistence to 10 stays unobservable; iteration 7's 1.896 billed GPU-h stays orphaned. |
| **(b) Finish iteration 7 only** | 66 steps × 160.6 s = **2.95 GPU-h** + ~$23–33 API + $2.08 scoring | One more matched point; clears the `n_iters 7` distortion. Arm lands ≈ 35.3 GPU-h ≈ 1.27× LA0. |
| **(c) Resume to 10** | ≈ **16–18 GPU-h**; arm ≈ 48–50 GPU-h; ~$77–120 + $8.32 scoring | Full 11-point GRPO K contrast; settles whether the K=5 reversal persists. Arm lands ≈1.75× LA0 — the asymmetry the 2026-08-18 deck declined on the record. |

⚠ **(b) and (c) both need two preconditions first**: raise the OpenAI spend limit, and fix or accept
Failure B. At the measured yield (30 / 0 / 0 / 10 steps banked per session), (c) is ~38 session
starts, not 4. The cheap mitigation is to write checkpoints and parquets to **local Colab disk** and
copy to Drive once at iteration end.

**2. Patch the two EDA defects before the next render** — the hardcoded `CENSOR_NOTE` constants
(33 rendered files assert a censoring that no longer holds) and `faithfulness.py:110`'s asymmetric
`SERIES` (a wrong value, not a caption). Neither is fixed by re-rendering.

**3. A second independent 96-conversation draw.** Every contested endpoint is a **single draw**; the
only noise floor is at the base (4 independent draws of the identical base policy: 6 pairs ×
9 metrics = 54 same-policy contrasts, **0 reaching even uncorrected p < .05**, max |dz| 0.128
primary / 0.147 held-out). No code change needed — therapist decoding is unseeded, and
`code/tools/generate_eval_convs.py --conv-dir` keeps the replicate out of the primary partition.

⚠ **Pick the adapters AFTER decision 1** — which endpoints are contested changes with it. At
5 adapters: 5 × 96 × 8 = **3,840 scoring calls per grader** (≈ $2.50–3.15 primary live + $7.24 Haiku
batched) plus 5,987 patient calls ≈ $1.6, and **1.06 A100-hours** (or 4.2 free local hours at
`--batch-size 6`).

**Isolation for the replicate:** write it to `conversations/replicate/<EXP_NAME>/` via `--conv-dir`
(`discover_arms` only scans `conversations/full`), and name its lake folder with the draw marker as
an infix *before* the `_I{k}` tail — `GRPOExp3_LA0_rep1_I10`. ⚠ **Never** write it into
`conversations/full/`: `model_iter_10` matches inside `model_iter_10_rep1_TT…`, so the replicate and
the primary collide on one `conv_dirs` key and one silently wins by glob order.

## Write-up decisions already made

- **One live paper: [`papers/2026_lookahead_pto_grpo/`](papers/2026_lookahead_pto_grpo/)** ("Same
  Lever, Different Optimizer") — both optimizers × both K × both graders × both cost axes, with the
  ICLR claims re-tested and the poster's transcripts re-scored. ⚠ Its **39-state / iteration-5
  censoring premise is now stale in 13 of 15 `.tex` files**; the frozen fixture itself has *not*
  drifted, but the world it describes has. Decide whether to re-cut at I6 or keep the freeze and
  restate the reason as a choice to discard available data.
- **Report the method comparison on BOTH axes.** Matched-iteration and matched-budget answer
  different questions — say which axis a number is on.
- Make the look-ahead claim **about the lever, never about convergence**.
- **State the look-ahead MI-consistency result at the CHANNEL level, not as a total**, and prefer
  the *share* of MI-inconsistent acts to any per-turn or per-session figure — the arms differ in
  both turn count and turn length, in method-dependent directions.
- Report all 8 instruments flat. The "orthogonal axes" framing is retired — PCT correlates ρ≈0.79–0.94
  with the rubrics.
- **Report the head-to-head both final-vs-final AND best-vs-best.**
- **Gain-retention disjointness is metric-dependent, not "iterations 9–10".** Name the metric before
  naming an iteration; the defensible statement is the ordering (PTO above GRPO from iteration 4 on).
- **Any claim that look-ahead reduces MI-inconsistency must name its axis and its grader.**
