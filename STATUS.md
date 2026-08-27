# Status — where the thesis stands

**THE single live copy of run status, headline numbers, and the cost constraint.** Every other doc
points here (see the Doc map in [CLAUDE.md](CLAUDE.md)). Keep this file short: it answers *where
things stand*, not *how they got here*. When an entry stops being current, move it to
[Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md](Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md) rather than
appending a new dated paragraph beneath the old one.

**Last updated 2026-08-26.**

## Run status — ALL FOUR ARMS COMPLETE at iteration 10

| Arm | Adapters | Scored (both graders) | GPU-h |
|---|---|---|---|
| **PTO LA0** | 1–10 | ✅ Base + I1–I10 = 11 | **8.119** |
| **PTO LA5** | 1–10 | ✅ Base + I1–I10 = 11 | **19.681** |
| **GRPO LA0** | 1–10 | ✅ Base + I1–I10 = 11 | **27.906** |
| **GRPO LA5** | **1–10** | ✅ Base + I1–I10 = 11 | **51.205** |

**44 scored model states** (11 × 4) — a **full grid on both graders**,
44 × 8 rubrics × 96 personas = **33,792 cells** each. Nothing is waiting to be scored, and **the
2×2 design is now complete**: the matched iteration-10 endpoint that every headline needed exists
for the first time.

**The 2026-08-21 "stopped at iteration 6" reading is retired twice over.** GRPO LA5 trained
iterations 7, 8, 9 and 10 to completion (106/106, 110/110, 130/130, **136/136** optimizer steps;
adapter + `eda/generations.jsonl` for each; iteration 10's adapter written 2026-08-25 05:14), and
the post-loop generate-only pass produced `model_iter_10` (96 conversations).

- **Failure B (Drive taking appends but not new files) did not recur.** Iteration 7 remains the only
  severe case (132 steps discarded across two sessions); iteration 9 lost 6 steps to a 17-min
  overhang; iterations 8 and 10 lost none.
- **The OpenAI failure mode escalated** from `organization_spend_limit_exceeded` (transient, cleared
  in ~3 h) to `insufficient_quota` / `credit_balance_exhausted` (balance at zero). Three sessions
  died on the harder one.
- **The resume did not silently change the science.** `run_metadata.json` differs from
  `run_metadata_pre_resume_iter1.json` only in `started_at` and the two audit-mirror fields
  (`lookahead_k`, `lookahead_sub_batch_size`) added after the arm began. Every knob is unchanged.

✅ **The compute mis-billing is RESOLVED, and the latent half is now CLOSED (2026-08-26).**
`compute.py` bills GRPO from `iteration_N/training/completions/*.parquet` mtimes; the historical
bug was that it had no adapter check, so an in-flight iteration with ≥3 timed steps counted as a
whole one (GRPO_LA5's stalled iteration 7 billed as `n_iters 7` against six adapters). It now
carries an **adapter gate**: only iterations with `iteration_N/adapter/` on disk — the same
completion marker the trainers' resume uses — are billed; an in-flight iteration is excluded and
announced, and its spend is billed once, when it completes. Verified behavior-preserving on the
current data: a cache-bypassed re-render of `compute/cost` produced byte-identical tables
(8.119 / 19.681 / 27.906 / 51.205 GPU-h, `n_iters 10` everywhere).

## Where the artifacts live

The EDA is organised **by research question**:
`Exp3_PTO_GRPO/eda/results/<top>/<sub>/{figures,tables}/[<judge>/]` with tops **`arms/`** (per-arm
descriptives, one leaf per grader), **`lookahead/`** (RQ-i: reward · transfer · behaviour ·
mechanism · replication), **`method/contrast`**, **`compute/cost`**, **`measurement/validity`** —
the four contrast tops carry both graders side by side and have no `<judge>/` level. **Navigation
starts at the hand-authored `results/README.md`** (added 2026-08-26: each research question → its
headline artifacts + the reading rules). Each top has a hand-authored `SUMMARY.md`;
`results/INDEX.md` maps every family to its notebook (workbooks now listed too). Regenerate with
`python tools/render_results.py` from `Exp3_PTO_GRPO/eda/`.

✅ **EDA refactor pass 2026-08-26** (details: `history/CHANGELOG_EDA.md`): the duplicate `headline/`
presentation copies are retired (curation lives in `results/README.md`); `lookahead/behaviour`'s
ledger is renamed `shape_numbers.json` (was `replication_numbers.json`, colliding with the
different-content ledger of the same name in `lookahead/replication` — the archived *Same Lever*
`NUMBERS.md` cites the old name); the package surface was trimmed (dead exports + dead code out,
`_selfcheck` green); a full re-render followed.

✅ **The tree is CURRENT as of 2026-08-25** — a full re-render (6 units / 21 notebook executions,
**1,086 s, no failures**) ran on the complete 44-state grid after iteration 10 landed. The parquet
fold was rebuilt first (31 files, 73,344 rows).

⚠ **The 2026-08-21 "censoring is gone from all 16 modules and 15 notebooks" claim was WRONG, and
was corrected on 2026-08-25.** What that pass actually removed was the hardcoded *iteration
numbers*; the **assertions survived**. An audit on 2026-08-25 found **81 occurrences across 13
notebooks** plus **six module-level `CENSOR`/`CENSOR_NOTE` constants** (`compute`, `faithfulness`,
`instruments`, `replication`, `transfer`, `lookahead`) still stating "GRPO_LA5 is right-censored"
— and because those constants are interpolated into `caption=` arguments, they were shipping the
false claim into ~20 rendered `CAPTIONS.md` entries, telling every reader of the results tree that
the main arm was truncated. Two further falsehoods travelled with them: captions asserting the two
GRPO arms were "budget-matched to within ~3%" (now 51.205 / 27.906 = 1.835), and
`plotting.lookahead.k_headline_fourarm` drawing a **"GRPO K=5 ends" arrow unconditionally** — so
the headline figure annotated iteration 10, where all four arms end, as a censoring point.

✅ **All of that is now fixed.** Every `CENSOR*` constant is a neutral legend rather than an
assertion, the annotation only draws when the arm genuinely stops early, and the notebooks state
support in derived terms. Censoring is derived from the frame in hand via
`constants.support_note()` / `constants.last_iterations()` — it returns `""` when every arm reaches
the same iteration, and is derived *per grader*, which matters because an arm's scored support can
differ between them. The only surviving `27.08` strings are budget-sweep **rung labels**
(`budget_27.08h`), which are real values on the compute axis, not the retired claim.
**The lesson worth keeping: de-specifying a stale claim is not the same as retracting it.**
- `faithfulness.py`'s asymmetric `SERIES` is fixed, and the fix proved itself: with every arm now
  reaching iteration 10, the derivation emits **four** series all labelled `(iters 1-10)` and drops
  the like-for-like subset row entirely — because nothing is censored any more. The hardcoded
  version would still be printing `iters 1-5`.
- ⚠ **Ledger key names moved with the data** and anything citing them needs re-pointing: the
  faithfulness keys are now `curve.<grader>.<arm>.iters1-10.*` (were `iters1-5`), and
  `dispersion_numbers.json`'s `grpo_la5_censored_at_train_iter: 5` → `last_train_iter_by_arm: {…}`.

✅ **All five `results/<top>/SUMMARY.md` were rewritten on 2026-08-25** against the completed grid,
staleness banners removed, each one audited number-by-number against its own tables by a second
pass. `compute/SUMMARY.md`'s "the two GRPO arms cost the same within 3%" is retracted in place
(51.205 / 27.906 = 1.835), and the retraction is kept visible rather than silently overwritten.

✅ **Four new artifacts (2026-08-25), each with a backing `*_data` table so every plotted point is
checkable:**
- `method/contrast/{tables/headline_grid.md, figures/headline_grid.png}` — the four arms at their
  endpoint, both graders side by side, each anchored to **its own base**, with persona-bootstrap
  CIs. Nothing showed that in one place before; every deck and paper opens with it. Its
  primary-grader rows reproduce `arms/stats/.../main_results.md` exactly.
- `lookahead/reward/figures/k_headline_q1q2_grpo.png` — the GRPO-only companion to the four-arm
  headline. ⚠ Its contrast row plots **K=5 − K=0** (the column is named `delta_K5_minus_K0`), the
  OPPOSITE of every other K table in the tree. That is deliberate — a paper arguing for look-ahead
  should not have its headline figure point down — but it is exactly the transposition trap the
  epistemic rules exist for, so the sign is in the column name, the axis label and the caption.
- `lookahead/behaviour/figures/overpraise_judgefree.png` — the judge-free lexical marker beside
  both graders' rated rates. Its cell **asserts** that the lexical column is byte-identical across
  graders rather than claiming it in prose.
- `measurement/validity/figures/judge_saturation.png` — the agreement collapse, its SD mechanism,
  and (panel c) each rubric at `GRPOExp3_LA5_I10` ranked against **its own** 44-state spread.
⚠ Every one of these keeps the two graders on **independent y-axes** where levels are shown; only
the SD panel shares an axis, because a spread is not a level.
✅ **`judge_saturation` guards its own weak spot.** Its panel-(b) inset prints each variance ratio
**with a trend test and a re-anchored value** (`1.410× ρ=+0.44 p=0.180 → is flat`,
`re-anchored to iter 1: 1.062×`), and its suptitle is composed from that verdict rather than
hardcoding a direction. That is a direct response to the anchoring error the adversarial review
caught — see CLAUDE.md § Epistemic status rule 2b.

✅ **`METRICS_REFERENCE.md` §6a now disambiguates the two faithfulness cuts** that sound
contradictory (`matched_iters`: K=5 agrees better; `train_iter_1`, the matched-policy cut:
nothing detectable, with the sign flipping between graders). It gives an explicit rule for which
supports which claim — the mechanism claim "look-ahead helps *because* it makes the reward a better
proxy" needs `train_iter_1`, and **`train_iter_1` does not support it**.

## Headline results — the method verdict flips sign with K, on BOTH graders

**This is the result of the experiment.** At the **matched iteration-10 endpoint**, on the same 96
personas, persona-paired, Holm-corrected within judge. Owner:
`method/contrast/tables/method_paired_by_K.md` (sign: **+ = PTO higher**).

| PTO − GRPO, Q1+Q2, iteration 10 | primary (gpt-4o-mini) | held-out (claude-haiku-4-5) |
|---|---|---|
| **K = 0** | **+0.507** (dz 0.729, p_holm .000) → PTO wins | **+0.609** (dz 1.265, p_holm .000) → PTO wins |
| **K = 5** | **−0.210** (dz −0.356, p_holm .001) → GRPO wins | **−0.206** (dz −0.313, p_holm .034) → GRPO wins |

**The sign flips at the same iteration, on the same conversations, under both graders.** That was
not true a day ago: at iteration 9 the K=5 reversal was significant only on the primary
(−0.257, p_holm 1.13e-4) and not on the held-out (−0.125, p_holm 0.250). Iteration 10 makes it
unambiguous. **"PTO beats GRPO" is a K=0 statement and must never be written without naming K.**

**GRPO K=5 @10 is the best final state in the experiment on BOTH graders.** Owners:
`arms/outcomes/tables/<judge>/leaderboard_scorecard.md`.

| final @10 | Q1+Q2 primary | Q1+Q2 held-out |
|---|---|---|
| **GRPO (K=5)** | **4.517** | **2.873** |
| PTO (K=5) | 4.307 | 2.667 |
| PTO (K=0) | 4.260 | 2.866 |
| GRPO (K=0) | 3.753 | 2.257 |

On the primary it also leads every other instrument (WAI-SR 3.729, CSQ-8 3.062, MI-SAT 3.832,
MITI 4.536, PCT 0.685) and has the *lowest* MICI of any final state at **0.210**. On the held-out it
leads WAI-SR 2.957, CSQ-8 2.935, MI-SAT 3.328, MITI 2.375 and PCT 0.725.

⚠ **"Still climbing" is a primary-grader statement.** On the primary the climb is real, not a
last-point artefact: I10 − I7 = 4.517 − 4.270 = **+0.247** persona-paired (dz 0.398, Wilcoxon
p < .0001, CI [0.127, 0.371]). On the held-out judge the argmax is I7 (2.912) but I10 − I7 =
**−0.039** with dz −0.058, p = **0.646**, CI [−0.175, +0.094] — it straddles zero, and the held-out
endpoint is itself still rising (2.776 @8 → 2.858 @9 → 2.873 @10). **The correct held-out reading is
"flat since ~iteration 6, no detectable further gain", NOT "regressed from a peak."**

⚠ **Do not say best-vs-best favours PTO K=0 on the held-out judge.** The gap is
2.921 − 2.912 = **0.009**, unpaired, with **no p in any table**, and ≈ 0.1 SE
(SE ≈ 1.083 / √96 = 0.111). `method_paired_best.md`'s held-out K=5 row (I7 vs I7) is −0.177,
p_holm 0.107 — also not significant. The honest statement is that GRPO K=5 and PTO K=0 are
**indistinguishable at the top of the held-out leaderboard**, while GRPO K=5 wins the matched
endpoint on the primary (−0.210, p_holm .001).

⚠⚠ **The strongest caveat in the experiment sits on this exact state, and iteration 10 did not
clear it.** Per-conversation cross-judge agreement on Q1
(`measurement/validity/tables/validity.xlsx`, sheet `second_judge_agreement`, n = 96 per state)
collapses along GRPO K=5's last three iterations: **0.941 (I5) → 0.877 → 0.842 → 0.769 → 0.487 (I9)
→ 0.544 (I10)**. Against a Q1 column median of **0.855** across all 44 states, GRPO_LA5's **I9 and
I10 are the two lowest-agreeing states in the entire experiment** (next is PTO_LA5_I10 at 0.667).
The two graders agree on the *ranking* of this arm but disagree most about *which conversations are
good* precisely where the primary scores it highest. Do not present 4.517 without this sentence.
⚠ **It is NOT "only the rewarded rubric".** Compared against each rubric's OWN 44-state spread
(the new `judge_saturation` panel c), **MITI falls furthest** — Δ median −0.325, its **worst state
of 44** — ahead of Q1 (−0.311, 2nd worst), MICI (−0.231) and Q2 (−0.195); CSQ-8, PCT, MI-SAT and
WAI-SR move −0.02 to −0.05. Rewarded **and** behaviour-coding rubrics degrade; global-impression
ones do not. Read each rubric down its own column — their baseline agreements differ by more than
the effect, so a raw cross-rubric comparison inverts the ordering.

**The mechanism is one-sided saturation of the TRAINING grader, not homogenised outputs.** Over
GRPO_LA5 base → I10 (`lookahead/replication/tables/sd_by_iter.md`, n = 96 each): the primary's Q1 SD
collapses **monotonically** 1.336 → 0.701 (variance ratio 0.701² / 1.336² = **0.275×**; Spearman
SD-vs-iteration **−0.86, p = .001**, and the ratio holds under any anchor — 0.285 vs iteration 1,
0.544 vs the mean of iterations 1–10). The held-out judge's Q1 SD, on the same conversations,
**does not move** (ρ = **+0.44, p = .18**). Only the grader that was optimised against loses range;
had the *conversations* homogenised, both would have compressed. A correlation cannot survive one
side losing three-quarters of its variance.
⚠ **This said the held-out SD "*grows* 0.763 → 0.906 (1.410×)" until 2026-08-25.** That ratio
anchors on iteration 0, which is that series' **minimum**; re-anchored to iteration 1 it is 1.062×
and against the mean of iterations 1–10, 1.034×, with a null trend. **Flat, not growing** — and
flat is all the inference needs. A two-point ratio anchored on a series extremum is its own trap.

**Look-ahead (RQ-i) — largest at the endpoint, significant on both graders.** Sign: **+ = K=0
higher**. Owner: `lookahead/reward/tables/k_table1.md`.

| iter | PTO · primary | PTO · held-out | GRPO · primary | GRPO · held-out |
|---:|---|---|---|---|
| 6 | +0.257 (0.42)*** | +0.343 (0.51)*** | −0.263 (−0.42)*** | −0.533 (−0.55)*** |
| 8 | +0.077 (0.17) | +0.186 (0.34)** | −0.172 (−0.27)** | −0.159 (−0.22) |
| 9 | +0.041 (0.08) | +0.187 (0.29) | −0.647 (−0.93)*** | −0.856 (−0.90)*** |
| **10** | −0.047 (−0.10) | +0.199 (0.31) | **−0.765 (−0.91)\*\*\*** | **−0.616 (−1.03)\*\*\*** |

GRPO's K=5 advantage is significant on both graders at iterations 4, 6, 7, 9 and 10, and is largest
at the endpoint. **PTO never significantly favours K=5 at any iteration on either grader** — its
endpoint is null on the primary (−0.047) and non-significant on the held-out (+0.199).

**Difference-in-differences at iteration 10** (`lookahead/reward/tables/k_did.md`, Q1Q2, n = 96):
primary **0.718** (dz 0.793, CI [0.547, 0.898], p_holm .000); held-out **0.815** (dz 0.972,
CI [0.647, 0.976], p_holm .000). *"The training grader is blind to the interaction"* is emphatically
retired. ⚠ **The graders converged in RAW units only** — 0.815 / 0.718 = 1.14× against 1.52× at
iteration 9 — and that happened because the held-out raw DiD *fell* (1.044 → 0.815), not because the
primary caught up. **In effect size nothing moved**: dz 0.793 / 0.972 at iteration 10 versus
0.799 / 0.971 at iteration 9. Quote dz, not the raw ratio.

⚠ **CORRECTION to the iteration-9 retention claim.** At iteration 9 GRPO K=0's gain retention was
0.191 [−0.064, 0.382] against K=5's 0.686 [0.598, 0.788], `cis_disjoint = True`. **That was driven
by an outlier iteration**: at iteration 10 GRPO K=0 retention is **0.578** [0.457, 0.723] against
K=5's **0.668** [0.587, 0.765], and the CIs **overlap** (`cis_disjoint = False`). Only **Q1** stays
disjoint at 10 (K=0 0.295 [0.054, 0.488] vs K=5 0.676 [0.578, 0.795]). Owner:
`lookahead/transfer/tables/k_retention_summary.md`. **State retention per metric and per iteration —
do not generalise iteration 9.**

**On the compute axis, look-ahead wins for GRPO under every selection rule — but PTO still owns the
budget.** Owner: `compute/cost/tables/budget_sweep_crossjudge_verdicts.md` ("honest" = selected on
one grader, evaluated on the other).

- **GRPO_K at 51.200 GPU-h:** LA5 > LA0 on **all four** judge combinations (mean_delta 0.256–0.435,
  every p_holm .000), with primary selection now picking **iteration 10**.
- **PTO_K at 19.680 GPU-h:** LA5 < LA0 or no significant difference — **opposite sign**, so the
  interaction holds at matched budget as well as matched iteration.
- **method_K0 at 8.120 GPU-h:** PTO_LA0 ≫ GRPO_LA0 on all four (0.759–0.900, dz 1.07–1.39).
- **method_K5 at 19.680 GPU-h:** PTO_LA5 > GRPO_LA5 on three of four (n.s. on the honest
  select-primary / eval-held-out combo) — because at PTO's budget GRPO K=5 only reaches
  **iteration 3**.

**Cost ratios** (`compute/cost/tables/compute_by_arm.md`): 27.906 / 8.119 = **3.44×** (GRPO K=0 over
PTO K=0), 19.681 / 8.119 = **2.42×** (PTO K=5), 51.205 / 8.119 = **6.31×** (GRPO K=5), and
51.205 / 27.906 = **1.84×** for look-ahead within GRPO.

**The single sharpest cost sentence in the experiment:** look-ahead lets GRPO overtake its own K=0
sibling at **23.210 GPU-h** on both graders (`budget_sweep_GRPO_K_<judge>.md`; first significant
rung 35.290, +0.188, p_holm .020 primary). But at that same 23.210 GPU-h GRPO K=5 is only at I4,
scoring 4.120 / 2.784 — while **PTO K=0's entire ten-iteration run costs 8.119 GPU-h and scores
4.260 / 2.866, higher on both graders at 8.119 / 23.210 = 0.350 of the compute.**

⚠ **But "PTO always wins at matched budget" needs care.** It is true of every ladder the tables
cover (method_K0 at 8.120 GPU-h, 4/4 selection cells; method_K5 at 19.680, 3/4) — because at PTO's
budget GRPO K=5 only reaches iteration 3. It is **not** true that GRPO K=5 never overtakes PTO: at
the matched I10 endpoint it beats PTO K=5 on **both** graders (−0.210 / −0.206) and PTO K=0 on both
(4.517 − 4.260 = 0.257 primary; 2.873 − 2.866 = 0.007 held-out). **The honest summary is a
quality/cost trade, not a winner.**

**Unchanged findings.** The ICLR ordering still reproduces on the poster's own 1,440 transcripts
under the modern grader; look-ahead still *rescales* rather than sharpens the training signal
(ratio-of-ratios 1.01–1.03) and adds no faithfulness at a matched policy; 19–23% of K=5 tails end
early; the PTO/GRPO gap is still the **state distribution**, not the loss; the reward-hack is still
a compounding loop rather than a hard pull.

⚠ **"GRPO regresses into sycophancy" is wrong as stated. Over-praise capture is a K=0 phenomenon in
BOTH optimizers** — and this one is settled *without* a judge. `arms/validity/tables/<judge>/overpraise_crosscheck.md`
carries a **deterministic lexical over-praise marker rate** alongside the rated one; it is computed
from the transcripts, so it is identical under both graders. Final states:

| arm @10 | lexical marker rate | rated `MICI_OverPraiseRate` (primary) |
|---|---|---|
| GRPO K=0 | **0.671** | 0.698 |
| PTO K=0 | **0.210** | 0.299 |
| GRPO K=5 | **~0.06** | 0.051 |
| PTO K=5 | **0.045** | 0.043 |

0.671 / 0.06 ≈ **11×** within GRPO and 0.210 / 0.045 = **4.7×** within PTO. **Look-ahead suppresses
over-praise capture in both optimizers by roughly an order of magnitude**, and PTO K=0 — not
GRPO K=5 — is the second-most-sycophantic arm. Mechanism:
`arms/preference/tables/gpt-4o-mini/update_lexical_push.md` gives GRPO_LA0's per-update over-praise
push as +0.073 (SE 0.009) versus GRPO_LA5's +0.004 (SE 0.005) — under 1 SE, i.e. the K=5 gradient is
not pushing over-praise at all.

⚠ But **"GRPO K=5's MI-inconsistency returned to baseline" is a primary-grader claim.** On the
primary its `MICI_Rate` ends at 0.210 against a base of 0.209; on the held-out the same
conversations read 0.628 against a base of 0.326. Say "look-ahead slows the loop by ~10×", not
"stops it".

## Measurement validity

Oracle **ICC(2,1) 0.86–0.99**. The decoupled second judge (**Claude Haiku 4.5**, different family,
never played the patient) is on the full 44-state grid: `_selfcheck`'s `multi-judge analysis`
reports **352/352 cells complete** (44 states × 8 metrics) and **6,692 / 7,568** pairwise
arm×metric contrasts keeping their sign = **88.4%**, over 8 × C(44,2) = 8 × 946 = 7,568. That is
flat against 88.6% at 43 states and 88.5% at 40 — **the agreement rate has not moved as the grid
grew**, which is the cleanest evidence that the two graders are measuring the same construct at the
arm level.

⚠ **Dependability**: at `n_arms = 44`, read `multijudge_variance_components.md` directly. The last
corrected figures were `dependability_k1` MITI 0.622 / MICI 0.845 at 40 arms.

⚠ **Arm-level agreement does NOT license a single-state claim.** Per-conversation cross-judge
agreement varies enormously across states, and its two worst cases are GRPO K=5's last two —
Q1 r = 0.487 (I9) and 0.544 (I10) against a 44-state median of 0.855. See § Headline results.
Sign preservation is an *arm-level* statistic; quote it for orderings, never for an endpoint.

⚠ **Standing caveats** — see [eda/results/LIMITATIONS.md](Exp3_PTO_GRPO/eda/results/LIMITATIONS.md):
**no channel-level ICC at all** and **no oracle repeatability rep for any K=5 state** (therapist
decoding is unseeded, so no conversation set here is reproducible). ✅ *"No replicate draw for any
trained checkpoint" was retired 2026-08-26* — `GRPO_LA5@10` and `PTO_LA0@10` each have a second
independent draw (§ Next step 3); the other 42 states remain single draws, and a replicate bounds
*evaluation* noise only, never run-to-run *training* variance (still one training run per arm).
All 96 personas are used for both training and eval at every iteration, so every number is
in-sample w.r.t. the patient distribution.

## Cost constraint

⚠ **The OpenAI balance hit zero on 2026-08-24**, killing iteration 10 at step 70/136 and blocking
scoring until credits were added. That was the second, harder version of the 2026-08-20
organization spend cap — `insufficient_quota` / `credit_balance_exhausted` rather than a transient
throttle. **Both the run and the scoring are now unblocked and complete.**

⚠ **No dollar tally in this repo has an artifact behind it.** No billing export, invoice, or
receipts file exists anywhere; `compute.py` has zero dollar arithmetic. **Reconcile against the
vendor dashboard before quoting any figure.**

- Scoring the four new states (I7–I10) cost 4 × 8 × 96 = **3,072 calls per grader**: the Haiku side
  as two Message Batches (2,304 + 768, 0 errors), the primary live (0 errors).
- Cost is dominated by oracle scoring + (at K=5) look-ahead patient calls, both ∝ candidate count
  (`prompts×G` / `branch-points×M`) × iterations.
- Prompt caching is already maxed (~50% off the oracle's fixed prefix), so **the only lever is call
  COUNT**: cap `NUM_ITERATIONS`, drop `M`/`G` 8→4, (PTO) lower `GREEDY_TRUNK_TARGET_LEN`.
  Keep **K** (the RQ-i variable) and the **gpt-4o-mini oracle** (the measurement instrument) fixed.
- ⚠ **Price a Haiku sweep with `judge_plan.sweep_report(..., receipt=(42.0, 22272))` and read the
  `batch=False` row.** `calibrate_from_receipt` divides the receipt by its call count then
  multiplies by `(1 − batch_discount)` — i.e. it assumes list price. The $42 receipt was itself a
  Message Batches sweep, so `batch=True` halves an already-discounted rate.
- ⚠ **There is no batched primary path.** `judge_batch.py` raises "Batch path is Anthropic-only",
  so any "batched primary" estimate is unachievable — price the primary live.

## Next step

**The experiment is data-complete. Everything below is analysis and write-up, not spend on training.**

**1. ✅ DONE (2026-08-25) — the five `results/<top>/SUMMARY.md` are rewritten**, along with
`LIMITATIONS.md`, `METRICS_REFERENCE.md`, and the tree-wide censoring purge described above.
What remains here is the **P1 draft's own open items**, listed in its README.

**2. ✅ DECIDED (2026-08-27, revised later the same day): TWO submissions to ARR October 2026.**
Submission **2026-10-12**, commitment 2026-12-20; the single cycle feeds **NAACL 2027 +
COLING 2027** and the venue is chosen in December once reviews exist. Both papers are on the
**iterations-only comparison axis** (Lior's call — no GPU-hour, budget, or samples analysis; the
budget machinery stays EDA-only under `results/compute/cost/`; each paper discloses its cost
asymmetry in Limitations only).
   **P2** [`papers/2026_pto_grpo_mi/`](papers/2026_pto_grpo_mi/) (*Same Lever, Different
Optimizer* — the full 2×2): §5 (cost) + budget appendix deleted; the two-winners head-to-head
moved into §4 (`ssec:winners`); Limitations discloses matched-iterations ≠ matched-data (GRPO
302,541 vs PTO 99,622 oracle calls over ten K=0 iterations = 3.04×; 289,983 vs 121,806 = 2.38×
at K=5; sums over `compute/cost/tables/api_calls.md`). Ethics keeps only the ≈107 GPU-h total.
   **P1** [`papers/2026_grpo_lookahead_mi/`](papers/2026_grpo_lookahead_mi/) (*Scoring the
Continuation* — GRPO with look-ahead as the story): first archived when the ICLR 2027 plan was
dropped, then **revived the same day per Lior** as a second ARR submission — ported to ACL
format, PTO now cited openly as the lever's origin (`baruch2025pto`) with the contribution
framed as "we moved look-ahead to GRPO", PTO arms still nowhere as data, ICLR §5 (cost/budget)
replaced by a Limitations disclosure (oracle calls ≈matched 302,541/289,983; ≈393k K=5-only
look-ahead patient calls; median 1.92× per-step wall-clock) + a ≈79 GPU-h Ethics line. The
ICLR-formatted version stays frozen at `papers/archive/2026_grpo_lookahead_mi/`.
   ⚠ **Open risk: ARR's multiple-submission policy.** The GRPO K-lever numbers appear in both
papers (subject vs interaction cells). Claims are disjoint and neither cites the other's prose,
but two same-cycle submissions from one experiment need the supervisors' explicit sign-off.
   Candidate framings history: [`papers/BRAINSTORM_2026-08-25.md`](papers/BRAINSTORM_2026-08-25.md);
the pre-completion four-arm draft remains archived at `papers/archive/2026_lookahead_pto_grpo/`
with stale ledger keys (`iters1-5` → `iters1-10`).

**3. ✅ DONE (2026-08-26) — the second independent 96-conversation draw. EVERY HEADLINE SURVIVES.**
`GRPO_LA5@10` and `PTO_LA0@10` were re-drawn on a Colab A100 (`code/tools/replicate_colab.ipynb`,
Run All) and scored on all 8 instruments by both graders — 1,536 calls per grader, **0 errors**.
Report: [`eda/results/measurement/replicate_draw.md`](Exp3_PTO_GRPO/eda/results/measurement/replicate_draw.md)
(every headline contrast computed twice, original draw vs replicate, side by side).

**The trained-state noise floor, measured for the first time:** 2 arms × 9 metrics × 2 graders =
**36 same-policy contrasts, 0 significant after Holm, max |dz| 0.216**, largest raw Q1+Q2 gap
0.056 — the same order as the base-only floor, which until now was the only one that existed
(LIMITATIONS §5c).

**Original → replicate on Q1+Q2** (sign as named; every starred row p_holm < .05 in both):

| contrast | primary | held-out |
|---|---|---|
| K lever @10, GRPO K5−K0 | +0.765 → **+0.709** (dz .905→.919)\* | +0.616 → **+0.637** (dz 1.030→.949)\* |
| method @K0, PTO−GRPO | +0.507 → **+0.516**\* | +0.609 → **+0.659**\* |
| method @K5, PTO−GRPO | −0.210 → **−0.155**\* | −0.206 → **−0.227**\* |
| top pair, GRPO_LA5−PTO_LA0 | +0.257 → **+0.193**\* | +0.007 → **−0.022** (both n.s.) |

⚠ **The K-lever and method rows re-draw only ONE side** (the K=0 / non-replicated arm is the same
draw in both columns), so they test the contested endpoint, not the whole contrast.
⚠ **The held-out "tie" is now settled properly.** The published 0.007 was *unpaired with no p*; the
paired test gives dz +0.012 (p_holm 1.000) on the original draw and **flips sign** to −0.022
(dz −0.039, p_holm 1.000) on the replicate. GRPO_LA5 and PTO_LA0 are **indistinguishable on the
held-out judge** — now demonstrated across two independent draws, not inferred from one gap.
Endpoint levels moved 4.517 → 4.461 (primary) and 2.873 → 2.894 (held-out), i.e. the headline
"4.517" is a single draw whose replicate reads ~4.46.

Original sizing note kept below:
Every contested endpoint is a **single draw**; the only noise floor is at the base (4 independent
draws of the identical base policy: 6 pairs × 9 metrics = 54 same-policy contrasts, **0 reaching
even uncorrected p < .05**, max |dz| 0.128 primary / 0.147 held-out). The endpoints most worth
replicating are **GRPO K=5 @10** (the new best state on both graders, and the state with the worst
cross-judge agreement in the experiment) and **PTO K=0 @10** (the arm it beats by 0.007 on the
held-out). No code change needed — therapist decoding is unseeded, and
`code/tools/generate_eval_convs.py --conv-dir` keeps the replicate out of the primary partition.
At 2 adapters: 2 × 96 × 8 = **1,536 scoring calls per grader**, plus ~0.4 A100-hours (or ~1.7 free
local hours at `--batch-size 6`).

**Isolation for the replicate:** write it to `conversations/replicate/<EXP_NAME>/` via `--conv-dir`
(`discover_arms` only scans `conversations/full`), and name its lake folder with the draw marker as
an infix *before* the `_I{k}` tail — `GRPOExp3_LA5_rep1_I10`. ⚠ **Never** write it into
`conversations/full/`: `model_iter_10` matches inside `model_iter_10_rep1_TT…`, so the replicate and
the primary collide on one `conv_dirs` key and one silently wins by glob order.

**4. ✅ DONE (2026-08-26) — the adapter gate is in `compute.py`** (see "the compute mis-billing is
RESOLVED" above; verified byte-identical on the complete data).

## Write-up decisions already made

- **TWO live papers (2026-08-27), both → ARR October 2026** — P2
  [`papers/2026_pto_grpo_mi/`](papers/2026_pto_grpo_mi/) (the full 2×2) and the revived P1
  [`papers/2026_grpo_lookahead_mi/`](papers/2026_grpo_lookahead_mi/) (GRPO with look-ahead;
  PTO cited as origin, never data). Earlier drafts + the ICLR-format P1 are retired under
  `papers/archive/`; the archived `NUMBERS.md` ledgers remain the trap list for any shared
  number. ⚠ Same-cycle overlap needs supervisor sign-off (see § Next step 2).
- **The PAPER's comparison axis is iterations only (Lior, 2026-08-27)** — no GPU-hour or
  samples/budget analysis in P2; the data-per-iteration asymmetry (GRPO ≈3.0×/2.4× PTO's oracle
  calls at K=0/K=5) is a Limitations disclosure, and the budget machinery stays EDA-only. For the
  THESIS the both-axes point still stands: matched-iteration and matched-budget answer different
  questions (at matched *iteration* GRPO K=5 wins; at any budget PTO can afford, PTO wins), and
  **never state a PTO-vs-GRPO verdict without naming K** — on either axis.
- Make the look-ahead claim **about the lever, never about convergence**.
- **State the look-ahead MI-consistency result at the CHANNEL level, not as a total**, and prefer
  the *share* of MI-inconsistent acts to any per-turn or per-session figure — the arms differ in
  both turn count and turn length, in method-dependent directions.
- Report all 8 instruments flat. The "orthogonal axes" framing is retired — PCT correlates ρ≈0.79–0.94
  with the rubrics.
- **Report the head-to-head both final-vs-final AND best-vs-best.** They disagree informatively:
  GRPO K=5's *final* is its *best* on the primary but not on the held-out judge, where its peak is I7.
- **Gain-retention disjointness is metric-dependent AND iteration-dependent.** Name both. The
  iteration-9 disjointness does not survive to iteration 10 except on Q1.
- **Any claim that look-ahead reduces MI-inconsistency must name its axis and its grader** — and
  should prefer the judge-free lexical marker, which settles the over-praise question outright.
