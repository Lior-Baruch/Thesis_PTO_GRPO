# NUMBERS.md — the claims ledger

Every quantitative claim in the draft → the exact tracked artifact it came from. `results/…` paths
are relative to `Exp3_PTO_GRPO/eda/`. **Nothing enters the `.tex` that is not a row here**, and no
row is written from prose — each was read off the named table or recomputed from the score lake.

*(2026-08-27 ARR revival: this ledger is ported from the archived ICLR draft
(`papers/archive/2026_grpo_lookahead_mi/NUMBERS.md`, numbers unchanged — same grid, same tables)
with ONE structural change: the §5 Cost section is REPLACED by the Limitations cost-disclosure
block below, per the iterations-only axis decision. Sections renumber: §5 Behaviour, §6
Mechanism, §7 Measurement, §8 Discussion; Limitations and Ethics are main-text page-exempt
sections in the ACL format.)*

**Graders.** `primary` = `gpt-4o-mini` (this WAS the training reward). `held-out` =
`claude-haiku-4-5` (never touched training). ⚠ **Levels are not comparable across graders** (the
offset is 1.2–1.7 points and model-dependent) and the two are **never averaged** — only contrasts
and standardized quantities combine.

**Sign conventions — the transposition trap.** The EDA's K tables (`lookahead/reward/*`) report
**K=0 minus K=5**, so a *positive* cell there means K=0 scored higher. This paper argues for K=5,
so most body sentences carry the **opposite** sign. Every row below states the direction it is in.
`MICI` is **lower-is-better**; `k_summary`'s `*_higher` columns are raw direction and its
`*_better` columns are polarity-corrected — read `*_better` for verdicts.

**Verification status.** Rows marked ✅ were independently recomputed from the score lake
(`data.load_scores_long` + `stats.paired_arrays`, persona-paired, n=96) on 2026-08-25 and agree
with the cited table. Rows marked 📄 were read off the cited table only.

---

## Setup

| claim | value | source |
|---|---|---|
| Arms in this paper | `GRPO_LA0`, `GRPO_LA5` — 2 arms × 11 states (base + 10 iterations) = 22 model states | results/lookahead/reward/tables/k_levels.md |
| Parent experiment | 4 arms × 11 states = 44 model states, both graders — ⚠ **NOT quoted anywhere in the paper**: every statistic that used the full grid was recomputed on the 22 GRPO states (see §7). PTO appears only as cited prior work (`baruch2025pto`), never as data | results/measurement/validity/tables/multijudge_coverage.md |
| Score-lake cells per grader | 44 × 8 × 96 = 33,792 | results/measurement/validity/tables/multijudge_coverage.md |
| Personas | 2 gender × 3 cooperation × 2 problem × 2 problem duration × 2 prior attempts × 2 age = 96 | Exp3_PTO_GRPO/code/system_prompts_builder.py `generate_all_permutations` |
| Matched knobs | MCL=12, G=8, same generation temperatures, same oracle (Q1+Q2), 10 iterations each | `run_metadata.json` of both arms (CONFIG FACT) |
| Instruments | 8 — Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI, PCT, MICI | results/arms/stats/tables/gpt-4o-mini/main_results.md |
| Training reward | Q1+Q2 only (mean of the two) | CLAUDE.md § Exp3 (CONFIG FACT) |
| Bootstrap | 2,000 resamples, percentile, `BOOT_SEED` = 12345 | eda_analysis/constants.py |

## §4 Reward

| claim | direction | value | source |
|---|---|---|---|
| ✅ K contrast at iteration 10, Q1+Q2, primary | K=5 higher | **+0.765**, dz 0.905, CI [0.601, 0.943], p 6.3e-13 | results/lookahead/reward/tables/k_endpoints.md (row `GRPO_LA5_I10 − GRPO_LA0_I10`) |
| ✅ K contrast at iteration 10, Q1+Q2, held-out | K=5 higher | **+0.616**, dz 1.030, CI [0.502, 0.732], p 4.6e-13 | same row, `judge_*` columns |
| ✅ `GRPO_LA0` vs own base, Q1+Q2, primary | gain | 3.067 → 3.753 = **+0.686**, dz 0.721 | results/arms/stats/tables/gpt-4o-mini/main_results.md |
| ✅ `GRPO_LA5` vs own base, Q1+Q2, primary | gain | 2.963 → 4.517 = **+1.554**, dz 1.518 | same |
| ✅ `GRPO_LA0` vs own base, Q1+Q2, held-out | gain | 1.861 → 2.257 = **+0.396**, dz 0.658 | results/arms/stats/tables/claude-haiku-4-5/main_results.md |
| ✅ `GRPO_LA5` vs own base, Q1+Q2, held-out | gain | 1.834 → 2.873 = **+1.038**, dz 1.539 | same |
| ✅ Gain ratio, primary | K=5 / K=0 | 1.554 / 0.686 = **2.27×** | derived from the two rows above — show the arithmetic |
| ✅ Gain ratio, held-out | K=5 / K=0 | 1.038 / 0.396 = **2.62×** | derived — show the arithmetic |
| 📄 K=5 ahead on all 8 instruments at iteration 10, both graders | K=5 better | 9 metric rows (8 instruments + the Q1Q2 composite), all favouring K=5, every $p_{holm}$ .000 under both graders | results/lookahead/reward/tables/k_endpoints.md (`favours_primary` / `favours_judge` columns, `GRPO_LA5_I10 − GRPO_LA0_I10`) |
| 📄 **Endpoint table in full** — $\Delta$ (dz) per instrument, primary \| held-out. Sign flipped from the source, which reports K=0 − K=5 | K=5 higher | Q1Q2 +0.765 (0.905) \| +0.616 (1.030) · Q1 +0.858 (0.902) \| +0.865 (1.152) · Q2 +0.671 (0.864) \| +0.367 (0.609) · WAI-SR +0.291 (0.513) \| +0.288 (0.442) · CSQ-8 +0.289 (0.482) \| +0.451 (0.678) · MI-SAT +0.352 (0.531) \| +0.503 (0.829) · MITI +0.615 (0.735) \| +0.276 (0.502) · PCT +0.111 (0.516) \| +0.113 (0.563) · MICI −0.627 (−1.862) \| −0.422 (−1.567) | results/lookahead/reward/tables/k_endpoints.md, the nine `GRPO_LA5_I10 − GRPO_LA0_I10 (K lever, GRPO matched iter)` rows |
| 📄 **Appendix by-iteration table** — Q1+Q2 $\Delta$ (dz) at iterations 0–10, both graders | K=5 higher | reproduced cell-by-cell from the source's GRPO columns with **every sign negated** | results/lookahead/reward/tables/k_table1.md |
| ✅ **LEVEL columns of both appendix tables** | — | endpoint per instrument: primary K0/K5 e.g. Q1Q2 3.753/4.517, MICI 0.838/0.210; held-out Q1Q2 2.257/2.873 (all 9 instruments × 2 graders); by-iteration Q1Q2 levels e.g. primary I8 4.082/4.254, I10 3.753/4.517 | results/lookahead/reward/tables/reward.xlsx sheets `k_levels_long` (iteration-10 rows) and `k_headline_grpo_data` (mean_K0/mean_K5); level-minus-level agrees with the paired Δ to ±0.001 rounding |
| 📄 Main §4 figure is LEVELS-only | — | k_headline_q1q2_grpo: trajectories + star row (Holm), no delta strip; stars = the same tests as tab:byiter | results/lookahead/reward/figures/k_headline_q1q2_grpo.png + tables/k_headline_grpo_data.md |
| 📄 GRPO K=0's own peak and decline | — | 4.082 at iteration 8, 3.808 at 9, 3.753 at 10 (primary Q1+Q2) | results/arms/stats/tables/gpt-4o-mini/main_results.md (`target=best`, `target_iter` 8) + recomputed from the score lake |
| 📄 Iterations where K=5 is Holm-significant on Q1+Q2 | K=5 better | primary 6 of 10 (iters 4, 6, 7, 8, 9, 10); held-out 6 of 10 (iters 4, 5, 6, 7, 9, 10) | results/lookahead/reward/tables/k_summary.md, GRPO/Q1Q2 rows, `iters_sig_K5_higher` |
| 📄 Base-vs-base noise floor (iteration 0) | neither | +0.104 primary (dz 0.115, n.s.), +0.026 held-out (dz 0.043, n.s.) | results/lookahead/reward/tables/k_table1.md row 0, GRPO cols |

⚠ **Do not write "significant at every iteration."** The K advantage is null for the first three
iterations under both graders and only opens from iteration 4.

### §4 replicate draw

A second independent 96-conversation draw of `GRPO_LA5@10` (same adapter, same 96 personas, same
seed-53 shuffle, unseeded decoding), scored on all 8 instruments by both graders, 0 errors.
Source for every row: results/measurement/replicate_draw.md (written by `eda/tools/replicate_check.py`).

| claim | direction | value | source |
|---|---|---|---|
| ✅ Trained-state noise floor, `GRPO_LA5@10` draw2 − draw1 | neither | 9 metrics × 2 graders, **0 significant after Holm**, max \|dz\| **0.174** (MICI, primary); Q1+Q2 −0.056 (dz −0.121, p .160) primary, +0.021 (dz +0.031, p .963) held-out | replicate_draw.md § "GRPO_LA5 @10, draw 2 − draw 1" |
| ✅ K contrast at iteration 10, Q1+Q2, primary, **replicate** | K=5 higher | **+0.709**, dz 0.919 (original +0.765, dz 0.905) | same, § "K lever @10 — replicate" |
| ✅ K contrast at iteration 10, Q1+Q2, held-out, **replicate** | K=5 higher | **+0.637**, dz 0.949 (original +0.616, dz 1.030) | same |
| ✅ Endpoint level, primary | — | 4.517 (original) → **4.461** (replicate) | same |

⚠ **The re-draw covers the $K{=}5$ side only** — the $K{=}0$ arm is the same draw in both columns —
so it tests the contested endpoint, not the whole contrast. Say so wherever the replicate is cited.
⚠ **A replicate bounds EVALUATION noise, never TRAINING variance.** There is still one training run
per arm; do not let "it replicates" drift into "the result is run-independent."

## Limitations — the cost disclosure (replaced the retired §5 Cost)

**Axis decision (2026-08-27, Lior + supervisors): ITERATIONS ONLY.** The GPU-hour/budget analyses
(totals, the 1.835× run ratio, the budget sweep and its crossover rungs, the 4/4 crossed-grader
verdicts at 51.2 GPU-h) are out of the paper — they live in the archived ICLR draft's ledger and
in the EDA under `results/compute/cost/`. What remains in the paper is one Limitations paragraph
and the Ethics one-liner, backed by:

| claim | value | source |
|---|---|---|
| Oracle scoring calls, sum over train iters 1–10 | `GRPO_LA0` **302,541** vs `GRPO_LA5` **289,983** — "approximately matched by construction" (per-candidate calls identical; the totals differ through conversation length → slice count) | results/compute/cost/tables/api_calls.md, `oracle_calls_train` summed over the 10 iteration rows per arm — **derived by summation; re-sum if the table re-renders** |
| Patient-simulator calls inside K=5 look-ahead rollouts, sum over train iters 1–10 | `GRPO_LA5` **392,766** (≈393k); `GRPO_LA0` **0** by construction | same table, `patient_calls_tail` summed over the 10 GRPO_LA5 iteration rows — derived by summation |
| ✅ Per-step wall-clock multiplier, settled iterations 3–10 | median **1.92×**, range 1.828–2.182 | results/compute/cost/tables/step_multiplier.md, `GRPO_step_ratio_K5_over_K0` |
| ⚠ iterations 1–2 excluded from the multiplier | 2.406 and 2.119 — these ran at a smaller look-ahead sub-batch and are not comparable (stated in Appendix B) | results/compute/cost/tables/step_multiplier.md; CLAUDE.md § Gotchas |
| Ethics total (one line, no per-arm breakdown) | 27.906 + 51.205 = 79.111 ≈ **79 GPU-hours** | results/compute/cost/tables/compute_by_arm.md — show the arithmetic |

⚠ GPU-hours are **reconstructed from artifact mtimes**, with gaps outside (0, 3600 s) imputed at
the phase median; never from `iteration_metadata.json`, whose `*_time_s` fields are per-process and
undercount any resumed iteration. The paper states this in Appendix B.
⚠ `lookahead_sub_batch_size` is recorded as its **final** value only — `write_run_metadata`
overwrites in place, so the iterations-1–2 exclusion is stated explicitly, not derived from
metadata.

## §5 Behaviour

| claim | axis + grader | value | source |
|---|---|---|---|
| ✅ MICI rise, `GRPO_LA0` | rate, primary | 0.211 → 0.838 = **+0.626**, dz 1.717 | results/arms/stats/tables/gpt-4o-mini/main_results.md |
| ✅ MICI rise, `GRPO_LA5` | rate, primary | 0.209 → 0.210 = **+0.001**, dz 0.006, **n.s.** (p .711) | same |
| ✅ MICI rise, `GRPO_LA0` | rate, held-out | 0.384 → 1.050 = **+0.666**, dz 1.975 | results/arms/stats/tables/claude-haiku-4-5/main_results.md |
| ✅ MICI rise, `GRPO_LA5` | rate, held-out | 0.326 → 0.628 = **+0.301**, dz 0.845 — **NOT flat** | same |
| ✅ Held-out ratio of MICI rises | rate, held-out | 0.301 / 0.666 = **0.45** — K=5 rises at under half K=0's rate | derived — show the arithmetic |
| ✅ K contrast on MICI at iteration 10 | rate | primary −0.627 (dz −1.862); held-out −0.422 (dz −1.567), K=5 better both | results/lookahead/reward/tables/k_endpoints.md |
| ✅ Over-praise, judge-free lexical marker | **share of therapist turns containing ≥1 marker** (a bounded [0,1] incidence, NOT a per-turn count), **no grader** | `GRPO_LA0` 0.671 vs `GRPO_LA5` 0.064 at iteration 10 = **10.5×** | results/arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md, `lex_overpraise_marker_rate`; definition verified in `eda_analysis/behavior.py` (`sum(bool(RE.search(t)) for t in turns) / n`) |
| 📄 Over-praise, oracle-rated | rate, primary | 0.698 vs 0.051 at iteration 10 | same table, `MICI_OverPraiseRate` |
| 📄 Over-praise K contrast significance | rate | K=0 worse at 6 iterations (primary, iters 5–10) and 7 (held-out, iters 4–10) | results/lookahead/behaviour/tables/k_channels_summary.md, `MICI_OverPraise_rate` GRPO rows |
| 📄 Questions per therapist turn | per turn, **judge-invariant text** | K=5 higher at 7 iterations (4–10), mean dz −0.643 (sign: K0−K5) | results/lookahead/behaviour/tables/k_channels_summary.md, `q_per_turn`, `text (judge-invariant)` |

⚠ **The single most important honesty constraint in this paper.** "Look-ahead prevents the reward
hack" is a **primary-grader** statement. The held-out judge sees the same conversations and reads
K=5's MI-inconsistency as still rising. Write **"slows the loop to under half the rate"**, cite the
judge-free lexical marker as the settling evidence, and never write "stops", "eliminates", or
"returns to baseline" without "under the training oracle".
⚠ The `lex_overpraise_marker_rate` column is identical under both graders because it is computed
from the transcripts, not rated — that is exactly why it is the load-bearing evidence here. It
lives under a `<judge>/` path only because `arms/*` is a per-judge family; the value is not
grader-dependent.
⚠ **Name its axis correctly.** It is the *share of therapist turns containing at least one marker*,
so 0.671 means "67% of turns", not "0.671 markers per turn". It is a brittle keyword regex kept as
a **direction check** on the oracle's counts — cite its agreement with the rated rate, never its
absolute value as a measurement of over-praise.

## §6 Mechanism

| claim | value | source |
|---|---|---|
| 📄 Reward faithfulness, pooled over matched iterations, primary | K=0 0.873 [0.861, 0.884] vs K=5 0.909 [0.900, 0.917]; difference −0.036 [−0.051, −0.021] | results/lookahead/mechanism/tables/faithfulness_k_summary.md, GRPO `matched_iters` |
| 📄 same, held-out | K=0 0.747 vs K=5 0.800; difference −0.053 [−0.078, −0.030] | same |
| 📄 Iteration-level test | K=5 more faithful at 7 of 10 iterations; Wilcoxon over iterations p = **.084** primary, **.193** held-out | same table, `iters_K5_more_faithful` / `wilcoxon_over_iters_p` |
| 📄 Matched-policy cut (train_iter 1) | all four deltas straddle zero: primary +0.015 [−0.026, 0.059]; held-out −0.014 [−0.075, 0.048]; per-length bins 17/20 favour K0 (primary) vs 17/20 favour K5 (held-out) | results/lookahead/mechanism/tables/faithfulness_k_summary.md `train_iter_1` + METRICS_REFERENCE.md §6a |
| 📄 Dispersion — rescaling not sharpening | pooled margin ratio 1.300 [1.275, 1.326], SD ratio 1.293 [1.267, 1.317], ratio-of-ratios **1.006** [1.002, 1.010] | results/lookahead/mechanism/tables/dispersion_ratios.md, GRPO `pooled` |
| ✅ Iteration-10 inversion | margin ratio 0.679 at train_iter 10: K=0's margin jumps 0.248 → 0.339 and its SD 0.083 → 0.111, while K=5's margin falls 0.268 → 0.230 and SD 0.090 → 0.078. **Ratio-of-ratios stays 0.964**, so the margin↔SD coupling holds and the rescaling result is unaffected | results/lookahead/mechanism/tables/dispersion_ratios.md, GRPO rows 9 and 10 |
| ✅ The coinciding over-praise jump | `GRPO_LA0` judge-free marker 0.093 (iter 9) → 0.671 (iter 10) — the same iteration as the margin jump | results/arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md |
| ✅ Update direction barely changes with K | `GRPO_LA0` vs `GRPO_LA5` pooled direction cosine **0.804**, ceiling 0.945, **0.851** corrected | results/arms/preference/tables/gpt-4o-mini/update_direction_cosines.md |

⚠ The "update direction" is an **embedding-space proxy**, not the gradient:
`normalise(Σ_g w_g · emb(t_g))` with `w_g` the standardised group-relative advantage (definition in
`eda_analysis/pref.py` Part 2). Say "the advantage-weighted direction over candidate embeddings",
never "the gradient". `cosine_corrected` divides by `sqrt(r_a·r_b)`, the attenuation ceiling from
each arm's Spearman-Brown-corrected split-half reliability — quote the corrected value with its
ceiling, not alone.
⚠ **Do not write "significantly more faithful."** The pooled-pairs CI excludes zero but treats
branch pairs as independent; the iteration-level Wilcoxon (n = 10) does not clear .05 under either
grader. The defensible sentence is "more faithful at 7 of 10 iterations, with a small and
consistent pooled difference."
⚠ **The matched-policy result is a different question and does not contradict the pooled one.** See
results/METRICS_REFERENCE.md §6a for which cut supports which claim; cite it rather than
re-deriving.

## §7 Measurement

*(Every full-grid statistic below is **recomputed on the 22 GRPO states** — the `*_grpo`
artifacts. The 44-state values belong to the companion 2×2 paper and are not quoted here.)*

| claim | value | source |
|---|---|---|
| ✅ `GRPO_LA5` per-conversation cross-grader agreement on **Q1** | .941 (I5) → .877 (I6) → .842 (I7) → .769 (I8) → **.487 (I9)** → **.544 (I10)** | results/measurement/validity/tables/validity.xlsx, sheet `second_judge_agreement`, `pearson_r`; also panel-a rows of judge_saturation_grpo_data.md |
| ✅ Q1 median across the 22 GRPO states | **0.841** — ⚠ the exact value is 0.8415 (mean of the middle pair .841/.842), so the figure prints 0.841 (`:.3f`) while the data table's display rounds to 0.842; the paper quotes 0.841 to match its own figure | results/measurement/validity/tables/judge_saturation_grpo_data.md (panel-a median row); recomputed independently from validity.xlsx 2026-08-26 |
| ✅ The two lowest-agreeing states in the experiment (= the 22 GRPO states) | `GRPO_LA5_I9` .487 and `GRPO_LA5_I10` .544 (next: `GRPO_LA0_I6` .744) | same |
| ✅ `GRPO_LA0` never leaves the normal range on Q1 | its 11 states span .744–.882 | validity.xlsx, sheet `second_judge_agreement` |
| ✅ The collapse is **selective but NOT Q1-only** | at `GRPO_LA5_I10` vs each instrument's 22-state median (shortfall, rank/22): **MITI .333/.678 (−.345, 1/22 — its minimum)**, Q1 .544/.841 (−.297, 2/22), Q2 .590/.754 (−.164, 1/22), MICI .287/.399 (−.112, 4/22) — all depressed; CSQ-8 .851/.891 (−.040), PCT .928/.956 (−.028), MI-SAT .906/.931 (−.025), WAI-SR .898/.921 (−.023) — all normal | judge_saturation_grpo_data.md panel-c rows; recomputed from validity.xlsx 2026-08-26 |
| ✅ One-sided saturation of the training grader | `GRPO_LA5` on Q1: the **primary** SD falls monotonically 1.336 → 0.701, Spearman(SD, iteration) **−0.86, p = .001**; variance ratio 0.701² / 1.336² = 0.275 (robust to anchor: 0.285 vs iter 1, 0.544 vs the mean of iters 1–10). The **held-out** SD does not move: Spearman **+0.44, p = .18** | results/lookahead/replication/tables/sd_by_iter.md |
| ✅ Arm-level sign preservation, GRPO states only | **1,640 of 1,848 = 88.7%** pooled; 94.7% at \|Δ\|≥0.10; **97.0%** at \|Δ\|≥0.25; **98.9%** at \|Δ\|≥0.50; 95.4% where the judge CI excludes 0 | results/measurement/validity/tables/multijudge_sign_preservation_grpo.md |
| 📄 Contrast count arithmetic | 8 × C(22,2) = 8 × 231 = 1,848 | derived — show the arithmetic |
| 📄 Oracle self-repeatability | ICC(2,1) 0.86–0.99 across Q1 / Q2 / MICI, four K=0 anchor states only | results/measurement/validity/tables/oracle_repeatability_icc.md |
| 📄 MITI dependability | `dependability_k1` 0.624 vs 0.91–0.97 for the Likert rubrics | results/measurement/validity/tables/multijudge_variance_components.md |

⚠ **Sign preservation is an ARM-LEVEL statistic.** Quote it for orderings; it does **not** license
a per-conversation claim, and this paper's own headline state is the counter-example.
⚠⚠ **Do not write that the held-out grader's variance GREW.** That ratio anchors on iteration 0,
which is that series' minimum; re-anchored to iteration 1 it is 1.062 and the trend test is null
(ρ = +0.44, p = .18). The held-out spread is **flat**, and flat is all the inference needs: if the
*conversations* had become homogeneous both spreads would have shrunk.
⚠ **Do not write "only the rewarded rubric".** MITI hits its experiment-wide minimum at the same
cell. The defensible pattern is *rewarded + behaviour-coding rubrics depressed, global-impression
rubrics normal*.
⚠ **No K=5 state has a repeatability rep**, so cross-grader agreement on this arm cannot be
benchmarked against a measured attenuation ceiling. Said in §7's closing paragraph.

## Ethics / Setup — config facts

| claim | value | source |
|---|---|---|
| 📄 Total training compute reported (Ethics one-liner) | ≈79 GPU-hours across the two arms (27.906 + 51.205 = 79.111) | results/compute/cost/tables/compute_by_arm.md |
| 📄 Base model / precision / LoRA / lr / seed | Llama-3.2-1B, bf16, LoRA r=16 α=16, lr 1e-5, seed 42 | each arm's `run_metadata.json` (CONFIG FACT) |
| 📄 Therapist / patient temperature | 0.9 / 0.7 | same |
| 📄 GRPO knobs | G=8, KL β=0.01, temperature 1.2, batch 64 × accumulation 2, eval split 0.05, loss type `grpo` | same |
| 📄 Session / context caps | 49-utterance target, 200 tokens per response, 2,048-token therapist context, MCL=12 | same |
| ✅ **The two arms differ in exactly two substantive config fields** | `lookahead_k` (absent vs 5) and `lookahead_sub_batch_size` (absent vs 128); everything else identical except the arm's own name, adapter repo and two output paths | diff of the two `run_metadata.json` files, verified 2026-08-25 |

## Limitations (claims that must appear)

| claim | source |
|---|---|
| K ∈ {0, 5} only, by design — no dose–response | results/LIMITATIONS.md |
| One training run per arm; no training-seed replicate — every dz is across personas within one run | results/LIMITATIONS.md |
| Every endpoint is a single 96-conversation draw; therapist decoding is unseeded (endpoint replicate excepted) | results/LIMITATIONS.md § 5c |
| All 96 personas are used for both training rollouts and eval — every number is in-sample | results/LIMITATIONS.md § 5e |
| Patient simulator and training oracle are the same model; the held-out judge decouples the grader, not the generator | results/LIMITATIONS.md § 2 |
| No human MI-coder validation of any instrument | results/LIMITATIONS.md § 1 |
| Q1+Q2 is both the training reward and a reported outcome (circularity) — the held-out judge is the partial answer | results/LIMITATIONS.md § 3 |
| MITI is the least dependable instrument; treat its arm differences as provisional | results/LIMITATIONS.md § 2 |
| Matched iterations ≠ matched cost — the call/wall-clock disclosure block above | this ledger, "Limitations — the cost disclosure" |
