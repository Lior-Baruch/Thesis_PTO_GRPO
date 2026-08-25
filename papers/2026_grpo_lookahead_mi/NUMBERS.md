# NUMBERS.md — the claims ledger

Every quantitative claim in the draft → the exact tracked artifact it came from. `results/…` paths
are relative to `Exp3_PTO_GRPO/eda/`. **Nothing enters the `.tex` that is not a row here**, and no
row is written from prose — each was read off the named table or recomputed from the score lake.

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
| Parent experiment | 4 arms × 11 states = 44 model states, both graders | results/measurement/validity/tables/multijudge_coverage.md |
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
| 📄 **Table 1 in full** — $\Delta$ (dz) per instrument, primary \| held-out. Sign flipped from the source, which reports K=0 − K=5 | K=5 higher | Q1Q2 +0.765 (0.905) \| +0.616 (1.030) · Q1 +0.858 (0.902) \| +0.865 (1.152) · Q2 +0.671 (0.864) \| +0.367 (0.609) · WAI-SR +0.291 (0.513) \| +0.288 (0.442) · CSQ-8 +0.289 (0.482) \| +0.451 (0.678) · MI-SAT +0.352 (0.531) \| +0.503 (0.829) · MITI +0.615 (0.735) \| +0.276 (0.502) · PCT +0.111 (0.516) \| +0.113 (0.563) · MICI −0.627 (−1.862) \| −0.422 (−1.567) | results/lookahead/reward/tables/k_endpoints.md, the nine `GRPO_LA5_I10 − GRPO_LA0_I10 (K lever, GRPO matched iter)` rows |
| 📄 **Appendix by-iteration table** — Q1+Q2 $\Delta$ (dz) at iterations 0–10, both graders | K=5 higher | reproduced cell-by-cell from the source's GRPO columns with **every sign negated** | results/lookahead/reward/tables/k_table1.md |
| 📄 GRPO K=0's own peak and decline | — | 4.082 at iteration 8, 3.808 at 9, 3.753 at 10 (primary Q1+Q2) | results/arms/stats/tables/gpt-4o-mini/main_results.md (`target=best`, `target_iter` 8) + recomputed from the score lake |
| 📄 Iterations where K=5 is Holm-significant on Q1+Q2 | K=5 better | primary 6 of 10 (iters 4, 6, 7, 8, 9, 10); held-out 6 of 10 (iters 4, 5, 6, 7, 9, 10) | results/lookahead/reward/tables/k_summary.md, GRPO/Q1Q2 rows, `iters_sig_K5_higher` |
| 📄 Base-vs-base noise floor (iteration 0) | neither | +0.104 primary (dz 0.115, n.s.), +0.026 held-out (dz 0.043, n.s.) | results/lookahead/reward/tables/k_table1.md row 0, GRPO cols |

⚠ **Do not write "significant at every iteration."** The K advantage is null for the first three
iterations under both graders and only opens from iteration 4.

## §5 Cost

| claim | value | source |
|---|---|---|
| 📄 Total GPU-h | `GRPO_LA0` 27.906, `GRPO_LA5` 51.205 | results/compute/cost/tables/compute_by_arm.md |
| 📄 Whole-run cost ratio | 51.205 / 27.906 = **1.835×** | derived from the row above — show the arithmetic |
| ✅ Per-step multiplier, settled iterations 3–10 | median **1.92×**, range 1.828–2.182 | results/compute/cost/tables/step_multiplier.md, `GRPO_step_ratio_K5_over_K0` |
| ⚠ iterations 1–2 excluded | 2.406 and 2.119 — these ran at a smaller look-ahead sub-batch and are not comparable | results/compute/cost/tables/step_multiplier.md; CLAUDE.md § Gotchas |
| 📄 Budget rung where K=5 is still behind | 18.31 GPU-h: −0.143 (dz −0.276), p_holm .053 | results/compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md |
| 📄 Budget rung where K=5 draws level | 23.21 GPU-h: +0.038 (dz 0.074), n.s. | same |
| 📄 First Holm-significant K=5 win on budget | 35.29 GPU-h: **+0.188** (dz 0.310), p_holm .020 | same |
| 📄 K=5 win at the common budget | 51.20 GPU-h: **+0.435** (dz 0.743), p_holm .000 | same |
| 📄 All four grader select/eval combinations at 51.2 GPU-h | K=5 > K=0 in 4/4, mean_delta 0.256–0.435, every p_holm .000 | results/compute/cost/tables/budget_sweep_crossjudge_verdicts.md, `GRPO_K` rows |
| 📄 MICI at the common budget (Q1+Q2-selected) | K=5 0.210 vs K=0 0.535 = −0.325 (dz −1.129), lower better | results/compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md, `select Q1Q2 / eval MICI` row at 51.200 |

⚠ **Quote the sweep, never a single iso-compute row** — the lever's sign is a function of budget.
⚠ GPU-hours are **reconstructed from artifact mtimes**, with gaps outside (0, 3600 s) imputed at
the phase median; never from `iteration_metadata.json`, whose `*_time_s` fields are per-process and
undercount any resumed iteration.

## §6 Behaviour

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
so 0.671 means "67% of turns", not "0.671 markers per turn". The draft said "per-turn rate" until
2026-08-25; the corrected phrasing is also the more vivid one (two turns in three vs one in
sixteen). It is a brittle keyword regex kept as a **direction check** on the oracle's counts — cite
its agreement with the rated rate, never its absolute value as a measurement of over-praise.

## §7 Mechanism

| claim | value | source |
|---|---|---|
| 📄 Reward faithfulness, pooled over matched iterations, primary | K=0 0.873 [0.861, 0.884] vs K=5 0.909 [0.900, 0.917]; difference −0.036 [−0.051, −0.021] | results/lookahead/mechanism/tables/faithfulness_k_summary.md, GRPO `matched_iters` |
| 📄 same, held-out | K=0 0.747 vs K=5 0.800; difference −0.053 [−0.078, −0.030] | same |
| 📄 Iteration-level test | K=5 more faithful at 7 of 10 iterations; Wilcoxon over iterations p = **.084** primary, **.193** held-out | same table, `iters_K5_more_faithful` / `wilcoxon_over_iters_p` |
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
⚠ **The matched-policy result is a different question and does not contradict this one.** See
results/METRICS_REFERENCE.md for which cut supports which claim; cite it rather than re-deriving.

## §8 Measurement

| claim | value | source |
|---|---|---|
| ✅ `GRPO_LA5` per-conversation cross-grader agreement on **Q1** | .941 (I5) → .877 (I6) → .842 (I7) → .769 (I8) → **.487 (I9)** → **.544 (I10)** | results/measurement/validity/tables/validity.xlsx, sheet `second_judge_agreement`, `pearson_r` |
| ✅ Q1 median across all 44 states | **0.855** | same sheet |
| ✅ The two lowest-agreeing states in the experiment | `GRPO_LA5_I9` .487 and `GRPO_LA5_I10` .544 (next: `PTO_LA5_I10` .667) | same sheet |
| ✅ The collapse is **selective but NOT Q1-only** | at `GRPO_LA5_I10` vs each instrument's 44-state median: Q1 .544/.855, Q2 .590/.784, **MITI .333/.658 (its 44-state MINIMUM)**, MICI .287/.518 — all depressed; CSQ-8 .851/.903, MI-SAT .906/.930, WAI-SR .898/.922, PCT .928/.954 — all normal | same sheet |
| ✅ One-sided saturation of the training grader | `GRPO_LA5` on Q1: the **primary** SD falls monotonically 1.336 → 0.701, Spearman(SD, iteration) **−0.86, p = .001**; variance ratio 0.701² / 1.336² = 0.275 (robust to anchor: 0.285 vs iter 1, 0.544 vs the mean of iters 1–10). The **held-out** SD does not move: Spearman **+0.44, p = .18** | results/lookahead/replication/tables/sd_by_iter.md |
| 📄 Arm-level sign preservation | 6,693 of 7,568 = 88.4% pooled; 97.2% at \|Δ\|≥0.25; 99.3% at \|Δ\|≥0.50 | results/measurement/validity/tables/multijudge_sign_preservation.md |
| 📄 Contrast count arithmetic | 8 × C(44,2) = 8 × 946 = 7,568 | derived — show the arithmetic |
| 📄 Oracle self-repeatability | ICC(2,1) 0.86–0.99 across Q1 / Q2 / MICI, four K=0 anchor states only | results/measurement/validity/tables/oracle_repeatability_icc.md |

⚠ **Sign preservation is an ARM-LEVEL statistic.** Quote it for orderings; it does **not** license
a per-conversation claim, and this paper's own headline state is the counter-example.
⚠⚠ **Do not write that the held-out grader's variance GREW.** The draft said "0.906² / 0.763² =
1.410, gains ~41%" until 2026-08-25. That ratio anchors on **iteration 0, which is that series'
minimum**; re-anchored to iteration 1 it is 1.062 and against the mean of iterations 1–10 it is
1.034, and the trend test is null (ρ = +0.44, p = .18). The held-out spread is **flat**. The
argument does not need growth: if the *conversations* had become homogeneous both graders' spreads
would have shrunk, so "one shrank, the other did not" is the whole inference. A two-point ratio
anchored on a series extremum is exactly the trap this ledger exists to catch.
⚠ **Do not write "only the rewarded rubric".** The draft said that until 2026-08-25; the table
above disproves it — MITI hits its experiment-wide minimum at the same cell. The defensible
pattern is *rewarded + behaviour-coding rubrics depressed, global-impression rubrics normal*, and
MITI's weakness is independently documented (it is the least dependable instrument in the battery,
`multijudge_variance_components.md`: `dependability_k1` 0.624).
⚠ **No K=5 state has a repeatability rep**, so cross-grader agreement on this arm cannot be
benchmarked against a measured attenuation ceiling. Say so in Limitations.

## §11 Ethics / §3 Setup — config facts

| claim | value | source |
|---|---|---|
| 📄 Total training compute reported | 27.9 + 51.2 = 79.1 GPU-hours across the two arms | results/compute/cost/tables/compute_by_arm.md (27.906 + 51.205, rounded to 1 dp in prose) |
| 📄 Base model / precision / LoRA / lr / seed | Llama-3.2-1B, bf16, LoRA r=16 α=16, lr 1e-5, seed 42 | each arm's `run_metadata.json` (CONFIG FACT) |
| 📄 Therapist / patient temperature | 0.9 / 0.7 | same |
| 📄 GRPO knobs | G=8, KL β=0.01, temperature 1.2, batch 64 × accumulation 2, eval split 0.05, loss type `grpo` | same |
| 📄 Session / context caps | 49-utterance target, 200 tokens per response, 2,048-token therapist context, MCL=12 | same |
| ✅ **The two arms differ in exactly two substantive config fields** | `lookahead_k` (absent vs 5) and `lookahead_sub_batch_size` (absent vs 128); everything else identical except the arm's own name, adapter repo and two output paths | diff of the two `run_metadata.json` files, verified 2026-08-25 |

⚠ `lookahead_sub_batch_size` is recorded as its **final** value only — `write_run_metadata`
overwrites in place, so the fact that iterations 1–2 ran at a smaller sub-batch is NOT recoverable
from the metadata. That is why §5 states the exclusion explicitly rather than deriving it.

## §10 Limitations (claims that must appear)

| claim | source |
|---|---|
| K ∈ {0, 5} only, by design — no dose–response | results/LIMITATIONS.md |
| One training run per arm; no training-seed replicate — every dz is across personas within one run | results/LIMITATIONS.md |
| Every endpoint is a single 96-conversation draw; therapist decoding is unseeded | results/LIMITATIONS.md § 5c |
| All 96 personas are used for both training rollouts and eval — every number is in-sample | results/LIMITATIONS.md § 5e |
| Patient simulator and training oracle are the same model; the held-out judge decouples the grader, not the generator | results/LIMITATIONS.md § 2 |
| No human MI-coder validation of any instrument | results/LIMITATIONS.md § 1 |
| Q1+Q2 is both the training reward and a reported outcome (circularity) — the held-out judge is the partial answer | results/LIMITATIONS.md § 3 |
| MITI is the least dependable instrument; treat its arm differences as provisional | results/LIMITATIONS.md § 2 |
| Three behaviour denominators can disagree in direction; prefer the share of coded acts | results/LIMITATIONS.md § 5b |
