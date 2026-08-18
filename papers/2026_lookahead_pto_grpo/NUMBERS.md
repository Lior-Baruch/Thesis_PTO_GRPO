# NUMBERS.md — the claims ledger
Every quantitative claim in the draft -> the exact artifact it came from: the tracked EDA table under Exp3_PTO_GRPO/eda/results/<family>/tables/ (named first in each source cell), with the paper's frozen fixture table (tables/*.md|csv, its analysis/out/*.json ledger key, or the analysis/out/_findings_digest.txt 'caveats'/'digest' entry) in parentheses. Sign convention for K contrasts: tables report K=0 minus K=5 (+ => K=0 higher). Graders: gpt-4o-mini = the training oracle; claude-haiku-4-5 = held out. GRPO K=5 is right-censored at iteration 5.

**Provenance (2026-08-18).** The numbers in the text were computed by the paper-local generators
`analysis/*.py` (retired at commit b09eb6f; see `analysis/README.md`) and are kept here as a FROZEN
FIXTURE: `analysis/out/*.json` + `tables/*.md|csv`, bootstrap CIs at seed 0. Those generators were
promoted into the tracked EDA (`eda_analysis.{lookahead,transfer,compute,tails,dispersion,faithfulness,
crossgen,replication,instruments}` + the family notebooks), which reproduces every one of them under
`Exp3_PTO_GRPO/eda/results/<family>/tables/…` (`lookahead/{reward,transfer,behaviour,mechanism,replication}`,
`compute/cost`). The EDA bootstraps at `BOOT_SEED=12345`, so bootstrap CI bounds may differ in the third
decimal; every mean, dz, p and count is identical (verified 2026-08-18). Each source cell below names the
tracked successor first and the fixture in parentheses; `results/...` paths are relative to
`Exp3_PTO_GRPO/eda/`. Rows citing an `analysis/out/*.json` key add the results ledger
(`results/<family>/tables/<name>_numbers.json :: <key>`) where one exists.

## Setup

| claim | value | source |
|---|---|---|
| Base model, precision, LoRA, lr, seed | Llama-3.2-1B, bf16, LoRA r=16 alpha=16, lr 1e-5, seed 42 | run_metadata.json of the four arms (CONFIG FACTS) |
| Patient temperature / therapist temperature / completion cap / context (appendix C; setup keeps only the 49-utterance target) | 0.7 / 0.9 / 200 tokens / 2048 tokens | run_metadata.json (CONFIG FACTS) |
| Conversation target / MCL | 49 utterances / 12 | run_metadata.json (CONFIG FACTS) |
| Oracle + patient model (appendix C) | gpt-4o-mini-2024-07-18; training-time oracle: JSON schema, T=0.0, no seed; score-lake grader (Run_Eval + Exp1 re-score): T=0.1, seed 42 | training-time: code/{GRPO,PTO}_Exp3/train_*_Iterative.ipynb cell 1 EVAL_TEMPERATURE = 0.0 + code/_shared/reward.py oracle call (no seed kwarg); score lake: eda/eda_analysis/scoring/registry.py EVAL_TEMPERATURE = 0.1 + scoring/pipeline.py seed=42. run_metadata.json records neither. |
| Persona strata | Cooperative / Warms up / Resistant, 32 each | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture tables/held_out_instruments_hetero.md) (n per stratum) |
| Persona grid = 96 | 2 gender x 3 cooperation x 2 problem (smoking, obesity) x 2 problem duration x 2 prior attempts x 2 age = 96 (counselor level fixed to expert) | Exp3_PTO_GRPO/code/system_prompts_builder.py generate_all_permutations |
| PCT definition | PCT_ChangeProp = CT/(CT+ST), neutral patient utterances excluded | eda/eda_analysis/constants.py QUESTIONNAIRES['PCT']; scoring/pipeline.py denom = ct + st; results/lookahead/behaviour/tables/pct_kcontrast.md (fixture tables/held_out_instruments_pct.md) header |
| GRPO knobs | G=8, KL beta 0.01, temp 1.2, batch 64 x accum 2, loss grpo, eval split 0.05 | run_metadata.json (CONFIG FACTS) |
| PTO knobs | greedy tree, M=8, tau=0.1, branch temp 1.2, DPO beta 0.1 sigmoid, batch 2 x accum 8, trunk target 49 | run_metadata.json (CONFIG FACTS) |
| Iterations per arm | PTO K0 10, PTO K5 10, GRPO K0 10, GRPO K5 5 (censored) | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) col n_iters |
| GRPO K5 five iterations cost as much as GRPO K0 ten | 27.078 vs 27.906 GPU-h | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) col total_gpu_h |
| Number of model states | 11+11+11+6 = 39 | results/lookahead/reward/tables/k_levels.md (fixture tables/k_contrast_headline_levels.md) (rows) |
| Score-lake size per grader | 39 x 8 x 96 = 29,952 | CONFIG FACTS; Exp3 results/measurement/validity/tables/multijudge_coverage.md (formerly results/L5/tables/8_measurement) |
| Oracle self-repeatability ICC | Q1 .990 / Q2 .976 / MICI .916, four K=0 anchor states only | CONFIG FACTS (results/measurement/validity/tables/oracle_repeatability_icc.md, formerly results/L5/tables/8_measurement) |
| Repeatability band | +-0.10 | k_contrast_headline caveats (EdaConfig.oracle_noise) |
| Bootstrap resamples | 2,000, percentile, seed 0 | k_contrast_headline caveats (C.paired) |
| Holm family for by-iteration K contrasts | across iterations 0..N within (grader, method, instrument) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) header |
| Noise floor PTO base pair Q1Q2 primary | -0.003 (dz -0.003) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) row 0 |
| Noise floor GRPO base pair Q1Q2 primary | +0.104 (dz 0.1148 -> prints as 0.11, n.s.); held-out base pair +0.026 (dz 0.04) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) row 0 (+0.104 (+0.11) / +0.026 (+0.04)); results/lookahead/reward/tables/k_paired_grpo_gpt-4o-mini.md (fixture k_contrast_headline_grpo_primary.csv) row Q1Q2 iteration 0 |
| Compute reconstruction: gap cutoff, imputation (the (0, 3600 s) cutoff: appendix C; setup says "resume gaps imputed at the phase median") | gaps outside (0, 3600 s) imputed at phase median | results/compute/cost/tables/compute_by_iteration.md (fixture tables/compute_axis_by_iteration.md) header |
| Generation under-count | ~0.1 h / iteration | compute_axis caveats; results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) total_gpu_h_floor |

## Reward

| claim | value | source |
|---|---|---|
| PTO Q1Q2 primary K0 >= K5 at 8/10 trained iterations | iters 1-4,6-9 positive; 5 (-0.002), 10 (-0.047) negative | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) PTO gpt-4o-mini col |
| PTO Q1Q2 primary sig only iter 6 | +0.257 (dz 0.417), p_holm .001 | results/lookahead/reward/tables/k_paired_pto_gpt-4o-mini.md (fixture tables/k_contrast_headline_pto_primary.md) Q1Q2 iter 6; summary n_sig_K0_higher=1 |
| PTO Q1Q2 held-out K0 higher at 10/10 (the "CI excl 0 at 8/10" clause was cut) | iters 2,3,4,5,6,8,9,10 | results/lookahead/transfer/tables/k_pairs.md (fixture tables/cross_k_multijudge_pairs.md) PTO Q1Q2 (judge_ci_excl0) |
| PTO Q1Q2 held-out Holm-sig iters 5,6,8 | +0.173 (dz .330), +0.343 (.511), +0.186 (.337) | results/lookahead/reward/tables/k_table1.md + k_paired_pto_claude-haiku-4-5.md (fixture tables/k_contrast_headline_table1.md; _pto_heldout.md) |
| PTO endpoint iter 10 primary | -0.047 (dz -0.096) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) row 10 |
| PTO endpoint iter 10 held-out | +0.199 (dz 0.308), CI [0.068, 0.332], not Holm-sig | results/lookahead/transfer/tables/k_pairs.md (fixture tables/cross_k_multijudge_pairs.md) PTO Q1Q2 iter 10 |
| PTO held-out Q2 sig K0>K5 iters 5-10 | dz .427,.578,.328,.544,.569,.653 (range 0.33-0.65) | results/lookahead/reward/tables/k_table1_Q2.md (fixture tables/k_contrast_headline_table1_Q2.md) |
| PTO held-out Q1 sig only iter 6 | +0.313 (dz .379) | results/lookahead/reward/tables/k_table1_Q1.md (fixture tables/k_contrast_headline_table1_Q1.md) |
| PTO primary Q2 sig at 6 and 8 | +0.341 (dz .521); +0.145 (dz .330) | results/lookahead/reward/tables/k_table1_Q2.md (fixture tables/k_contrast_headline_table1_Q2.md) |
| PTO primary Q1 never sig | iter 6 +0.173 dz .271 p_holm .418 | results/lookahead/reward/tables/k_table1_Q1.md (fixture tables/k_contrast_headline_table1_Q1.md); digest |
| ICLR poster: K gain significant only on Q2 | (poster text, best-vs-best L0 M4 vs L5 M7) | papers/2025_iclr_pto_lookahead/submitted/paper.pdf p.9 |
| GRPO no Holm-sig contrast at iters 1-3 either grader | (table stars) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) GRPO cols |
| GRPO iter 4 primary | -0.115 (dz -0.248), p_holm .044 | results/lookahead/reward/tables/k_paired_grpo_gpt-4o-mini.md (fixture tables/k_contrast_headline_grpo_primary.md) Q1Q2 iter 4 |
| GRPO iter 5 primary | -0.070 (dz -0.135), n.s. | results/lookahead/reward/tables/k_paired_grpo_gpt-4o-mini.md (fixture tables/k_contrast_headline_grpo_primary.md) Q1Q2 iter 5 |
| GRPO iter 4/5 held-out | -0.233 (dz -0.374) / -0.311 (dz -0.429), both p_holm .006 | results/lookahead/reward/tables/k_paired_grpo_claude-haiku-4-5.md (fixture tables/k_contrast_headline_grpo_heldout.md) Q1Q2 |
| GRPO iter 5 held-out Q1 vs Q2 | Q1 -0.450 (dz -0.499, ***) vs Q2 -0.172 (n.s.) | results/lookahead/reward/tables/k_table1_Q1.md / k_table1_Q2.md (fixture tables/k_contrast_headline_table1_Q1.md/_Q2.md) |
| DiD iter 5 held-out | gap_K0 +0.265 (dz .355), gap_K5 -0.219 (dz -.377), DiD +0.484 dz .525 [0.307, 0.675], p_holm(iters) <.001, p_holm_rubrics <.001 | results/lookahead/reward/tables/k_did.md (fixture tables/cross_k_multijudge_did.md) Q1Q2 iter 5; k_method_gap.md (fixture method_gap.md) |
| DiD iter 5 primary | +0.068 dz .095 n.s. | results/lookahead/reward/tables/k_did.md (fixture tables/cross_k_multijudge_did.md) Q1Q2 iter 5 gpt-4o-mini |
| GRPO I5 own-base Q1 retention K5 vs K0 | 1.048 [0.913, 1.223] vs 0.786 [0.587, 1.003], overlap | results/lookahead/transfer/tables/k_retention_summary.md (fixture tables/cross_k_multijudge_retention_summary.md) GRPO 5 Q1 |
| GRPO Q1 retention under shared LA5 base | 1.048 vs 0.709 [0.543, 0.879], disjoint | results/lookahead/transfer/tables/k_retention.md (fixture tables/cross_k_multijudge_retention.md) method_LA5_base |
| GRPO Q1 retention under shared LA0 base | 1.155 [0.985, 1.391] vs 0.786 [0.587, 1.003], overlap | results/lookahead/transfer/tables/k_retention.md (fixture tables/cross_k_multijudge_retention.md) method_LA0_base |
| GRPO base-pair difference on Q1 (primary) | +0.110 (dz 0.120) | results/lookahead/reward/tables/k_paired_grpo_gpt-4o-mini.md (fixture tables/k_contrast_headline_grpo_primary.md) row Q1 iter 0 |
| PTO I10 Q2 retention K5 vs K0 | 0.562 [0.476, 0.652] vs 0.849 [0.746, 0.977], disjoint | results/lookahead/transfer/tables/k_retention_summary.md (fixture tables/cross_k_multijudge_retention_summary.md) PTO 10 Q2 |
| PTO I10 MITI retention | 0.268 [0.181, 0.354] vs 0.450 [0.357, 0.548], disjoint | results/lookahead/transfer/tables/k_retention_summary.md (fixture tables/cross_k_multijudge_retention_summary.md) PTO 10 MITI |
| PTO I10 Q1Q2 retention | 0.639 vs 0.823 (overlap) | results/lookahead/transfer/tables/k_retention_summary.md (fixture tables/cross_k_multijudge_retention_summary.md) |
| Sign preservation where the oracle is Holm-significant | 18/18 (over 153 cross-K contrasts = 17 iteration pairs x 9 instruments; the 37/38 at abs(D)>=0.10 and 46/49 judge-CI figures were cut with the paragraph) | results/lookahead/transfer/tables/k_sign_ladder.md (fixture tables/cross_k_multijudge_ladder.md) |

## Cost

| claim | value | source |
|---|---|---|
| GRPO median step s K0 vs K5, iters 3-5 | 79.186/79.409/78.618 vs 155.635/155.788/150.217 | results/compute/cost/tables/step_multiplier.md (fixture tables/compute_axis_step_multiplier.md) |
| GRPO step ratio K5/K0 iters 3-5 | 1.965 / 1.962 / 1.911 | results/compute/cost/tables/step_multiplier.md (fixture tables/compute_axis_step_multiplier.md) |
| GRPO step ratio iter 1 (superseded) (appendix C) | 2.406 (sub-batch 64, API tail); iters 3-5 ran at sub-batch 128 (CLAUDE.md gotcha "K=5 costs ~1.9x") | results/compute/cost/tables/step_multiplier.md (fixture tables/compute_axis_step_multiplier.md); header note |
| GRPO GPU-h per iteration ratio | 5.416 / 2.791 = 1.94 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) gpu_h_per_iter |
| GRPO totals | K5 27.078 h (5 iters) vs K0 27.906 h (10 iters) | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) total_gpu_h |
| PTO iteration ratio | 1.968 / 0.812 = 2.42 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) gpu_h_per_iter |
| PTO build ratio (totals) | 16.797 / 5.669 = 2.96 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) build_h |
| PTO build share (appendix C) | 0.853 (K5) vs 0.698 (K0) | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) build_share |
| PTO DPO step ratio iters 3-5 | 1.035 / 1.033 / 1.021 | results/compute/cost/tables/step_multiplier.md (fixture tables/compute_axis_step_multiplier.md) |
| GRPO_LA0 / PTO_LA0 total (appendix C) | 27.906 / 8.119 = 3.44 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) |
| Arm-level median step s | PTO 6.757 / 6.894; GRPO 78.902 / 155.788 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) median_step_s |
| Gen / build / train hours per arm | PTO_LA0 1.323/5.669/1.127; PTO_LA5 1.370/16.797/1.514; GRPO_LA0 1.214/0/26.692; GRPO_LA5 0.422/0/26.656 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) |
| Oracle calls K5/K0 iters 1-5 | PTO 59,177/59,991 = 0.986; GRPO 141,237/143,548 = 0.984 | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) |
| Oracle input chars K5/K0 iters 1-5 (appendix C) | PTO 1.205; GRPO 1.360 | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) |
| Tail patient calls per candidate | PTO 2.646; GRPO 2.873 | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) |
| Patient calls K5/K0 iters 1-5 | PTO 8.655; GRPO 28.673 | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) |
| Total API calls K5/K0 iters 1-5 | PTO 145,291/69,941 = 2.077; GRPO 338,476/150,427 = 2.250 | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) |
| Wall-clock GPU-bound | (statement) | CLAUDE.md Training internals; compute_axis paper-use |
| PTO K sweep primary, 4.64 h | -0.700 dz -0.805 p_holm <.001 (I2 vs I4) | results/compute/cost/tables/budget_sweep_PTO_K_gpt-4o-mini.md (fixture tables/compute_axis_budget_sweep_PTO_K_gpt-4o-mini.md) |
| PTO K sweep primary, 8.94 h (appendix A table only) | -0.372 dz -0.560 | same |
| PTO K sweep primary, 16.17 / 18.03 h (n.s. rows) | -0.116 dz -0.225 p_holm .190 (I8 vs I10); -0.063 dz -0.115 CI [-0.176, 0.046] p_holm .867 (I9 vs I10) | same |
| PTO K sweep, 12.70 h budget repeats the 10.00 h pair | 5/10 (oracle) and 5/9 (held-out), same values; collapsed to 10.00-12.70 in appendix A | results/compute/cost/tables/budget_sweep_PTO_K_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/compute_axis_budget_sweep_PTO_K_{gpt-4o-mini,claude-haiku-4-5}.md) rows 10.000 and 12.700 |
| PTO K sweep primary, 14.60 h | -0.174 dz -0.348 p_holm .012 | same |
| PTO K sweep primary, 19.68 h | +0.047 dz .096 CI [-0.054, 0.142] (I10 vs I10) | same |
| GRPO K sweep primary, 13.27 h | -0.569 dz -0.742 | results/compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md (fixture tables/compute_axis_budget_sweep_GRPO_K_gpt-4o-mini.md) |
| GRPO K sweep primary, 18.31 h | -0.143 dz -0.276 p_holm .040 | same |
| GRPO K sweep primary, 7.80 h | -0.088 dz -0.093 p_holm .814 (I1 vs I2) | same |
| GRPO K sweep primary, 23.21 h and 27.08 h (same pair) | +0.038 dz .074 CI [-0.053, 0.137] p_holm .814 (I4 vs I8 at both budgets; GRPO K5 iter 5 does not improve on 4 under the oracle) | same |
| GRPO K sweep held-out, 23.21 h | +0.147 dz .331 p_holm .012 | results/compute/cost/tables/budget_sweep_GRPO_K_claude-haiku-4-5.md (fixture tables/compute_axis_budget_sweep_GRPO_K_claude-haiku-4-5.md) |
| GRPO K sweep held-out, 27.08 h | +0.161 dz .310 CI [0.057, 0.263] p_holm .020 (I5 vs I3) | same |
| PTO K sweep held-out, 19.68 h | -0.186 dz -0.323 CI [-0.301, -0.072] p_holm .011 (I7 vs I9) | results/compute/cost/tables/budget_sweep_PTO_K_claude-haiku-4-5.md (fixture tables/compute_axis_budget_sweep_PTO_K_claude-haiku-4-5.md) |
| Cross-judge GRPO K select oracle / eval judge | +0.166 dz .266 CI [0.041, 0.291] p_holm .035 (I4 vs I8) | results/compute/cost/tables/budget_sweep_crossjudge_verdicts.md (fixture tables/compute_axis_budget_sweep_crossjudge_verdicts.md) |
| Cross-judge GRPO K select judge / eval oracle | +0.048 dz .102 n.s. | same |
| Cross-judge PTO K select judge / eval oracle | -0.153 dz -.267 p_holm .023 | same |
| Cross-judge PTO K select oracle / eval judge | -0.199 dz -.308 p_holm .065 (n.s.) | same |
| Cross-judge PTO K verdicts (all four) | judge/judge -0.186 p_holm .011; judge-selects/oracle-scores -0.153 p_holm .023; oracle-selects/judge-scores -0.199 p_holm .065 n.s.; oracle/oracle +0.047 n.s. -> deficit sign in 3 of 4, Holm-sig in 2 of 4 (NOT 'all four') | same |
| Method K0 top budget 8.12 h | +0.900 dz 1.086 (I10 vs I2) primary; +0.814 dz 1.394 (I9 vs I2) held-out; all combos p_holm <.001 | results/compute/cost/tables/budget_sweep_method_K0_*.md (fixture tables/compute_axis_budget_sweep_method_K0_*.md); budget_sweep_crossjudge_verdicts.md |
| Method K5 top budget 19.68 h | +0.445 dz .673 (I10 vs I3) primary; +0.149 dz .2948 -> prints as 0.29, p_holm .007 (I7 vs I3) held-out | results/compute/cost/tables/budget_sweep_method_K5_*.md (fixture tables/compute_axis_budget_sweep_method_K5_*.md, csv dz 0.29482) |
| Method K5 cross-judge select oracle / eval judge (sec 5 + appendix A) | +0.081 dz .132 CI [-0.040, 0.199] p_holm .075 n.s. | results/compute/cost/tables/budget_sweep_crossjudge_verdicts.md (fixture tables/compute_axis_budget_sweep_crossjudge_verdicts.md) |
| GRPO_LA5 at 19.68 h has reached iter 3 (18.31 h) (appendix A, sweepMethod caption) | cum_gpu_h I3 = 18.312 | results/compute/cost/tables/compute_by_iteration.md (fixture tables/compute_axis_by_iteration.md) |

## ICLR

| claim | value | source |
|---|---|---|
| Poster claims: K gain sig only on Q2 (best-vs-best); lowest SD for L5 M7; conv length 43.7 -> 34.4 | (poster pp. 8-9, Table 1, Fig 4) | papers/2025_iclr_pto_lookahead/submitted/paper.pdf |
| Exp1 re-score size | 1,440 conversations = 15 states x 96 | results/lookahead/replication/tables/crossgen_levels.md (fixture tables/crossgen_exp1_levels.md) |
| K5 > K0 at 7/7 iterations under both graders | Final K0-K5 negative at all 7 | results/lookahead/replication/tables/crossgen_kcontrast.md (fixture tables/crossgen_exp1_kcontrast.md) metric=Final |
| Holm-sig iterations | gpt-4o-mini: 3 (p_holm .048), 7 (.003); GPT-3.5: 5 (.007), 7 (.001) | results/lookahead/replication/tables/crossgen_kcontrast.md (fixture tables/crossgen_exp1_kcontrast.md) |
| Arm-level K0-K5 gpt-4o-mini | -0.132 dz -0.543 CI [-0.180, -0.084] | results/lookahead/replication/tables/crossgen_kcontrast_summary.md (fixture tables/crossgen_exp1_kcontrast_summary.md) 'mean over iters 1-7' |
| Arm-level K0-K5 GPT-3.5 | -0.206 dz -0.612 | same |
| ICLR best-vs-best gpt-4o-mini | -0.129 dz -0.250 Wilcoxon p .006 | results/lookahead/replication/tables/crossgen_kcontrast_summary.md (fixture tables/crossgen_exp1_kcontrast_summary.md) |
| ICLR best-vs-best GPT-3.5 | -0.206 dz -0.251 Wilcoxon p .331, paired-t p .016 | same |
| gpt-4o-mini reads Exp1 higher | 0.19-0.43 points | results/lookahead/replication/tables/crossgen_levels.md (fixture tables/crossgen_exp1_levels.md) gap column; digest |
| Gap compressed by about a third | -0.132 vs -0.206 arm-level (0.64) | results/lookahead/replication/tables/crossgen_kcontrast_summary.md (fixture tables/crossgen_exp1_kcontrast_summary.md) |
| Literal total ordering fails under both graders | 44/49 (gpt-4o-mini), 48/49 (GPT-3.5) pairs L5 higher | results/lookahead/replication/tables/crossgen_kcontrast_summary.md (fixture tables/crossgen_exp1_kcontrast_summary.md) ordering rows |
| PTO iter 10 conv_len K0 vs K5 | 20.385 vs 28.698; K0-K5 -8.312 dz -0.548 CI [-11.219, -5.260] p_holm <.001 | results/lookahead/behaviour/tables/length_kcontrast.md (fixture tables/session_shape_stability_length_kcontrast.md) |
| GRPO iter 5 conv_len | 30.677 vs 22.573; +8.104 dz 0.531 | same |
| Every arm shorter than base; PTO K0 most | PTO_LA0 -8.000, PTO_LA5 -1.792, GRPO_LA0 -3.573, GRPO_LA5 -5.719 | results/lookahead/behaviour/tables/length_endpoints.md (fixture tables/session_shape_stability_length_endpoints.md) |
| PTO K5 more dispersed on Q1Q2 primary 10/10 | n_K5_lower_sd 0/10, median SD ratio 1.174, persona-paired Pitman-Morgan Holm-sig K0 lower at 4 iterations, K5 lower at 0; unpaired Brown-Forsythe Holm-sig at 0 | results/lookahead/replication/tables/sd_tally.md (fixture tables/session_shape_stability_sd_tally.md) |
| GRPO Q1Q2 primary | median ratio 1.035, 0 sig | same |
| Lowest-SD trained state is a K=0 state (SD/mean values cut from the text) | PTO_LA0 iter 10 SD 0.601 mean 4.260 | results/lookahead/replication/tables/sd_summary.md (fixture tables/session_shape_stability_sd_summary.md) |
| Spearman(mean, SD) primary over 35 states | -0.873 (Q1Q2); the Q1 (-0.892) and Q2 (-0.906) values were cut from the text | results/lookahead/replication/tables/sd_summary.md (fixture tables/session_shape_stability_sd_summary.md) |
| Held-out never awards Q1Q2 >= 4.5 | max 4.25 | session_shape_stability caveats; results/lookahead/replication/tables/sd_by_iter.md (fixture tables/session_shape_stability_sd.md) |
| Held-out PTO Q1Q2 dispersion | median ratio 1.008, 0 sig; Spearman -0.378 | results/lookahead/replication/tables/sd_tally.md (fixture tables/session_shape_stability_sd_tally.md); sd_summary.md (fixture sd_summary.md) |
| Held-out GRPO Q1Q2 dispersion | median ratio 1.002; one PM Holm-sig row, iter 4 with K5 less dispersed (sd 0.603 vs 0.785, ratio 0.767, pm_p_holm .008, bf_p_holm .001); iter 5 0.774 vs 0.838 n.s. | results/lookahead/replication/tables/sd_tally.md (fixture tables/session_shape_stability_sd_tally.md); sd_tests.md (fixture sd_bf.md) rows claude-haiku-4-5 GRPO Q1Q2 |

## Appendix A

| claim | value | source |
|---|---|---|
| Per-instrument summary table | all cells | results/lookahead/reward/tables/k_summary.md (fixture tables/k_contrast_headline_summary.md) (csv) |
| DiD table Q1Q2 iters 0-5 both graders | all cells | results/lookahead/reward/tables/k_did.md (fixture tables/cross_k_multijudge_did.md) (csv) |
| Retention summary table | all cells | results/lookahead/transfer/tables/k_retention_summary.md (fixture tables/cross_k_multijudge_retention_summary.md) (csv) |
| Endpoint contrasts table (Q1Q2, Q1, Q2, MITI, PCT, MICI x 7 pairs) | all cells | results/lookahead/reward/tables/k_endpoints.md (fixture tables/cross_k_multijudge_endpoints.md) (csv) |
| Budget sweep K tables | all Q1Q2->Q1Q2 rows, duplicates collapsed | results/compute/cost/tables/budget_sweep_{PTO,GRPO}_K_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/compute_axis_budget_sweep_{PTO,GRPO}_K_{gpt-4o-mini,claude-haiku-4-5}.md) |
| Budget sweep method tables | all Q1Q2->Q1Q2 rows, duplicates collapsed | results/compute/cost/tables/budget_sweep_method_K{0,5}_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/compute_axis_budget_sweep_method_K{0,5}_{gpt-4o-mini,claude-haiku-4-5}.md) |

## Appendix C

| claim | value | source |
|---|---|---|
| Config facts (all) | as listed | run_metadata.json of the four arms (CONFIG FACTS) |
| Score-lake cells per grader | 29,952 | CONFIG FACTS |
| Run_Eval calls per model state per grader | 96 x 8 = 768 | results/compute/cost/tables/api_calls.md (fixture tables/tail_audit_api_calls.md) header / digest |
| ICC on four K=0 anchors only | Q1 .990 / Q2 .976 / MICI .916 | CONFIG FACTS |
| Exp1 re-score single draw T=0.1, Q1+Q2 only | (caveat) | crossgen_exp1 caveats |
| Exp1 K=3 sweep unmatched (TT 0.7, tau 0.2), not re-scored by gpt-4o-mini (original GPT-3.5 scores exist: Final 3.185/3.360/3.642/3.635) | (caveat) | results/lookahead/replication/tables/crossgen_la3_gpt35.md (fixture tables/crossgen_exp1_la3_gpt35.md) (scored_by_gpt4omini = False) |
| Retention floors | 0.15 (Likert) / 0.05 (PCT, MICI) | cross_k_multijudge caveats |
| n_imputed per arm | GRPO_LA0 3, GRPO_LA5 2, PTO_LA0 1, PTO_LA5 2 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) n_imputed |
| GRPO_LA5 iter 1 training_time_s vs true span | 14,501 s vs 7.7 h | CLAUDE.md gotcha (measured 2026-08-17); compute_axis caveats |
| PTO_LA5 iters 1-5 gen_h ~0, 0.967 h lands in iter 6 | gen_h col | results/compute/cost/tables/compute_by_iteration.md (fixture tables/compute_axis_by_iteration.md) |
| Floor-corrected totals | 9.221 / 21.083 / 28.766 / 27.415 vs 8.119 / 19.681 / 27.906 / 27.078 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) total_gpu_h_floor |
| GRPO log coverage | GRPO_LA5 iters 1-2 0.500/0.712; GRPO_LA0 iters 2/6/8 0.500/0.741/0.722 | results/compute/cost/tables/api_calls.md (fixture tables/tail_audit_api_calls.md) log_coverage |
| Oracle calls / chars K5/K0 over PTO's full ten iterations (appendix C) | calls 121,806/99,622 = 1.223; Mchars 1,849.2/987.1 = 1.873 | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) rows PTO iters 1-10 |
| GRPO_LA5 frozen at 5 completed iterations | analysis uses iterations 1-5 (27.078 GPU-h); a sixth iteration is in progress on disk (run_metadata num_iterations = 6, iteration_6/training present) and is NOT in any table | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) n_iters; data/grpo_Exp3/runs/full/GRPO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_G8/ |
| Numbers quoted outside the nine (retired) analysis scripts | ICC (results/measurement/validity/tables, formerly L5/8_measurement), over-praise share (results/lookahead/behaviour/tables/k_mici_composition.md, judge column; formerly L5/7_stats/<judge>/), update-direction cosines (results/arms/preference/tables/gpt-4o-mini, formerly L{5,0}/6_preference), config facts (run_metadata.json), 14,501 s (iteration_metadata.json) | this ledger's rows name each |
| Cross-check cell | PTO Q1Q2 iter 6 +0.257 dz 0.417 | results/lookahead/reward/tables/k_paired_pto_gpt-4o-mini.md (fixture tables/k_contrast_headline_pto_primary.md) vs results/lookahead/reward/tables/k_paired_by_method.md (formerly results/L5/tables/7_stats/gpt-4o-mini/k_paired_by_method.md) |
| ICLR Table 1 vs disk | max |diff| 0.0005 | analysis/out/crossgen_exp1.json crosscheck.iclr_table1_vs_disk_max_abs_diff (fixture) = results/lookahead/replication/tables/crossgen_numbers.json :: crosscheck.iclr_table1_vs_disk_max_abs_diff |

## Behaviour
| claim | value | source |
|---|---|---|
| PTO iter 10 over-praise per therapist turn, primary | K0 0.299 vs K5 0.043; delta +0.256, dz 1.163, p_holm <.001 | results/lookahead/behaviour/tables/k_channels_pto_gpt-4o-mini.md (fixture tables/k_contrast_headline_channels_pto_primary.md) row MICI_OverPraise_rate iter 10 |
| PTO iter 10 over-praise per turn, held-out | K0 0.448 vs K5 0.075; delta +0.373, dz 1.648 | results/lookahead/behaviour/tables/k_channels_pto_claude-haiku-4-5.md (fixture tables/k_contrast_headline_channels_pto_heldout.md) row MICI_OverPraise_rate iter 10 |
| GRPO iter 5 over-praise per turn K0>K5 | dz 0.301 (p_holm .047) primary; dz 0.687 held-out | results/lookahead/behaviour/tables/k_channels_grpo_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/k_contrast_headline_channels_grpo_{primary,heldout}.md) row MICI_OverPraise_rate iter 5 |
| Over-praise share of MI-inconsistent acts, K=0 endpoints, primary | PTO_LA0 I10 3.042/4.958 = 0.613; GRPO_LA0 I10 8.250/9.865 = 0.836 | Exp3_PTO_GRPO/eda/results/lookahead/behaviour/tables/k_mici_composition.md rows judge=gpt-4o-mini PTO_LA0/10, GRPO_LA0/10 (formerly results/L5/tables/7_stats/gpt-4o-mini/k_mici_composition.md) |
| Over-praise share, K=0 endpoints, held-out | PTO_LA0 I10 4.750/8.510 = 0.558; GRPO_LA0 I10 10.188/13.000 = 0.784 | Exp3_PTO_GRPO/eda/results/lookahead/behaviour/tables/k_mici_composition.md rows judge=claude-haiku-4-5 (formerly results/L5/tables/7_stats/claude-haiku-4-5/k_mici_composition.md) |
| Over-praise share, K=5 endpoints, primary | PTO_LA5 I10 0.625/3.344 = 0.187; GRPO_LA5 I5 0.302/3.448 = 0.088; K=0 GRPO share at the matched iteration 5 = 0.729/3.771 = 0.193 | Exp3_PTO_GRPO/eda/results/lookahead/behaviour/tables/k_mici_composition.md rows judge=gpt-4o-mini PTO_LA5/10, GRPO_LA5/5, GRPO_LA0/5 (formerly results/L5/tables/7_stats/gpt-4o-mini/k_mici_composition.md) |
| Over-praise share, K=5 endpoints, held-out | PTO_LA5 I10 1.177/7.979 = 0.147; GRPO_LA5 I5 0.438/7.135 = 0.061 | Exp3_PTO_GRPO/eda/results/lookahead/behaviour/tables/k_mici_composition.md rows judge=claude-haiku-4-5 (formerly results/L5/tables/7_stats/claude-haiku-4-5/k_mici_composition.md) |
| Advice-w/o-permission per turn higher under K=5 at PTO iter 4 | primary delta -0.067 dz -0.363 p_holm .005; held-out -0.103 dz -0.362 p_holm .005 | results/lookahead/behaviour/tables/k_channels_pto_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/k_contrast_headline_channels_pto_{primary,heldout}.md) row MICI_AdviseNoPermission_rate iter 4 |
| Over-praise per-session gap first Holm-sig | primary iter 7 (iter 6 p_holm .061); held-out iter 6 (p_holm .001) | results/lookahead/behaviour/tables/k_channels_pto_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/k_contrast_headline_channels_pto_{primary,heldout}.md) rows MICI_OverPraise |
| Directing per session, PTO iter 10 (Table 3 row only; the prose sentence was cut) | primary -0.375 dz -0.389 p_holm .003; held-out -0.677 dz -0.468 p_holm <.001 | results/lookahead/behaviour/tables/k_channels_pto_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/k_contrast_headline_channels_pto_{primary,heldout}.md) row MICI_Direct iter 10 |
| Table substitution: PTO iter 10 over-praise / session | +2.417 dz 0.887 (primary); +3.573 dz 0.999 (held-out); both p_holm <.001 | results/lookahead/behaviour/tables/k_channels_pto_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_pto_{primary,heldout}.md) row MICI_OverPraise iter 10 |
| Table substitution: PTO iter 10 advice / session | -0.312 dz -0.239 p_holm .116 (primary); -1.948 dz -0.709 p_holm <.001 (held-out) | results/lookahead/behaviour/tables/k_channels_pto_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_pto_{primary,heldout}.md) row MICI_AdviseNoPermission iter 10 |
| Table substitution: PTO iter 10 all acts / session | +1.6146 -> prints as 1.61, dz 0.446 p_holm .001 (primary); +0.531 dz 0.099 p_holm .837 (held-out); sign kept on both graders, significance only on the oracle | results/lookahead/behaviour/tables/k_channels_pto_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_pto_{primary,heldout}.csv) row MICI_BehaviorTotal iter 10 |
| Table substitution: PTO iter 10 MICI per turn | +0.228 dz 0.708 (primary); +0.2446 -> prints as 0.24, dz 0.655 (held-out); both p_holm <.001 | results/lookahead/behaviour/tables/k_channels_pto_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_pto_{primary,heldout}.csv) row MICI_Rate iter 10 |
| Table substitution: GRPO iter 5 over-praise / session | +0.427 dz 0.378 p_holm .002 (primary); +1.250 dz 0.690 p_holm <.001 (held-out) | results/lookahead/behaviour/tables/k_channels_grpo_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_grpo_{primary,heldout}.md) row MICI_OverPraise iter 5 |
| Table substitution: GRPO iter 5 advice / session | -0.031 dz -0.019 (primary); +1.042 dz 0.268 p_holm .075 (held-out) | results/lookahead/behaviour/tables/k_channels_grpo_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_grpo_{primary,heldout}.md) row MICI_AdviseNoPermission iter 5 |
| Table substitution: GRPO iter 5 all acts / session | +0.323 dz 0.127 p_holm .420 (primary); +1.885 dz 0.342 p_holm .003 (held-out) | results/lookahead/behaviour/tables/k_channels_grpo_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_grpo_{primary,heldout}.md) row MICI_BehaviorTotal iter 5 |
| Table substitution: GRPO iter 5 MICI per turn | -0.063 dz -0.243 p_holm .176 (primary); -0.018 dz -0.053 (held-out) | results/lookahead/behaviour/tables/k_channels_grpo_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_grpo_{primary,heldout}.md) row MICI_Rate iter 5 |
| Therapist turns per session, PTO iter 10 / GRPO iter 5 | PTO K0 10.229 vs K5 14.385 (delta -4.156 dz -0.548); GRPO K0 15.344 vs K5 11.312 (+4.031 dz 0.528) | results/lookahead/behaviour/tables/length_kcontrast.md (fixture tables/session_shape_stability_length_kcontrast.md) rows n_th_turns |
| Conversation length, PTO iter 10 | K0 20.385 vs K5 28.698 utterances; K5-K0 +8.312; dz -0.548; p_holm 1.5e-05 | results/lookahead/behaviour/tables/length_kcontrast.md (fixture tables/session_shape_stability_length_kcontrast.md) row PTO conv_len |
| Conversation length, GRPO iter 5 | K0 30.677 vs K5 22.573; K0-K5 +8.104; dz 0.531; p_holm 3.3e-05 | results/lookahead/behaviour/tables/length_kcontrast.md (fixture tables/session_shape_stability_length_kcontrast.md) row GRPO conv_len |
| Therapist turn length | PTO iter 10 +124.673 chars for K5 (dz -0.5548 -> prints as 0.55); GRPO iter 5 +206.556 (dz -0.8147 -> prints as 0.81) | results/lookahead/behaviour/tables/session_shape.md (fixture tables/session_shape_stability_shape.csv) rows mean_turn_len (PTO 10, GRPO 5) |
| Questions per turn, GRPO iter 5 | K0 0.324 vs K5 0.691; delta -0.367 dz -0.605 p_holm 5.4e-08 | results/lookahead/behaviour/tables/session_shape.md (fixture tables/session_shape_stability_shape.md) rows GRPO q_per_turn |
| Questions per turn, GRPO_LA0 iter 10 | 0.1509 | analysis/out/session_shape_stability.json levels.GRPO_LA0.q_per_turn_final (fixture) = results/lookahead/behaviour/tables/replication_numbers.json :: levels.GRPO_LA0 (the session_shape.md / shape.md K-contrast table stops at GRPO iter 5; length_endpoints.md carries no q_per_turn column) |
| Questions per turn, PTO iter 10 | 0.550 vs 0.616; dz -0.110; p_holm 1.000 | results/lookahead/behaviour/tables/session_shape.md (fixture tables/session_shape_stability_shape.md) row PTO q_per_turn iter 10 |
| WAI total K0-K5, PTO iter 10 | primary -0.038 dz -0.073 p_holm 1.0; held-out +0.104 dz 0.174 p_holm 1.0 | results/lookahead/behaviour/tables/wai_kcontrast.md (fixture tables/held_out_instruments_wai_kcontrast.md) rows PTO WAI_total 10 |
| bond_excess K0-K5, PTO iter 10 | primary +0.221 dz 0.430 p_holm .001; held-out +0.270 dz 0.453 p_holm .001 | results/lookahead/behaviour/tables/wai_kcontrast.md (fixture tables/held_out_instruments_wai_kcontrast.md) rows PTO bond_excess 10 |
| bond_excess K0-K5, GRPO iter 5 | primary +0.148 dz 0.277 p_holm .069; held-out +0.129 dz 0.212 p_holm .220 | results/lookahead/behaviour/tables/wai_kcontrast.md (fixture held_out_instruments_wai_kcontrast.md) rows GRPO bond_excess 5 |
| WAI total K0-K5, GRPO iter 5 (appendix B) | primary -0.255 dz -0.446 p_holm <.001; held-out -0.109 dz -0.194 p_holm .596 | results/lookahead/behaviour/tables/wai_kcontrast.md (fixture held_out_instruments_wai_kcontrast.md) rows GRPO WAI_total 5 |
| Q2 items 3 and 10, held-out, PTO iter 10 K0-K5 | item 3 +1.104 dz 1.141 p_holm <.001; item 10 +1.031 dz 1.001 p_holm <.001; mean over the other 15 items +0.269 (the generator's 'other13', which also excludes items 1-2, is +0.268) | results/lookahead/behaviour/tables/q2_items_kcontrast.md (fixture tables/held_out_instruments_q2items_kcontrast.csv) rows claude-haiku-4-5 PTO (17 items) |
| Q2 item-level K effect on the training oracle ("no item-level K effect"; the 0/17 count was cut) | 0/17 Holm-sig for either method | results/lookahead/behaviour/tables/q2_items_kcontrast.md (fixture tables/held_out_instruments_q2items_kcontrast.md) rows gpt-4o-mini |
| PCT change proportion K0-K5 | GRPO iter 5: primary -0.056 dz -0.309 p_holm .017; held-out -0.067 dz -0.373 p_holm .003. PTO iter 10: primary -0.008 dz -0.044 n.s.; held-out -0.051 dz -0.253 p_holm .030 | results/lookahead/behaviour/tables/pct_kcontrast.md (fixture tables/held_out_instruments_pct.md) rows PCT_ChangeProp |
| PCT gain sits in the Warms-up stratum (the per-stratum dz values were cut; appendix B hetero figure) | GRPO@5 primary dz -0.896, held-out -0.786; PTO@10 primary -0.252, held-out -0.459 | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture tables/held_out_instruments_hetero.md) rows PCT Warms up matched_final |
| Q1Q2 within Warms-up, GRPO@5 | primary -0.204 dz -0.557 p_holm .020; held-out -0.476 dz -0.589 p_holm .010 | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture held_out_instruments_hetero.md) rows GRPO Q1Q2 Warms up |
| Cooperative ceiling on the training oracle | Q1Q2 means 4.767-4.904 (PTO@10 4.904/4.861; GRPO@5 4.767/4.797); share>=4.5 0.875-1.000; matched-endpoint K contrasts n.s. (PTO +0.043 p_holm .461; GRPO -0.030 p_holm 1.0); the own_best GRPO row (I8 vs I4) IS Holm-sig inside the stratum (+0.168 dz 1.584), hence 'at the matched endpoints' | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture held_out_instruments_hetero.md) rows Cooperative Q1Q2 (matched_final and own_best) |
| Held-out Cooperative Q1Q2 K0-K5, PTO@10 | +0.571 dz 0.773 p_holm .002 | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture held_out_instruments_hetero.md) row claude-haiku-4-5 PTO Q1Q2 Cooperative |
| MICI per turn within strata, PTO@10 K0-K5 (appendix B, Figure hetero only) | primary Coop 0.861 / Warms 0.592 / Resist 0.782; held-out 1.652 / 0.303 / 0.507 (all p_holm <.05) | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture held_out_instruments_hetero.md) rows PTO MICI matched_final |

## Mechanism
| claim | value | source |
|---|---|---|
| Margin ratio K5/K0 at base policy (train_iter 1) | PTO 1.551 [1.455, 1.658]; GRPO 1.440 [1.324, 1.560] | results/lookahead/mechanism/tables/dispersion_ratios.md (fixture tables/dispersion_by_k_ratios.md) rows train_iter 1 |
| SD ratio K5/K0 at base policy | PTO 1.530 [1.438, 1.632]; GRPO 1.400 [1.288, 1.515] | results/lookahead/mechanism/tables/dispersion_ratios.md (fixture tables/dispersion_by_k_ratios.md) rows train_iter 1 |
| Ratio of ratios | PTO 1.013 [1.000, 1.027]; GRPO 1.028 [1.015, 1.043]; pooled 1.019 / 1.016 | results/lookahead/mechanism/tables/dispersion_ratios.md (fixture tables/dispersion_by_k_ratios.md) |
| margin/SD range vs references (appendix B) | observed 2.890-3.064; iid-normal M=8 3.153; shuffle-null 2.912-3.169 | results/lookahead/mechanism/tables/dispersion_by_iter.md (fixture tables/dispersion_by_k_by_iter.md); results/lookahead/mechanism/tables/dispersion_expectation.md (fixture dispersion_by_k_expectation.md); digest |
| Winner standardized lead K5-K0 (sec 7 + appendix B) | PTO base +0.069 [0.020, 0.120], pooled -0.002 [-0.022, 0.018]; GRPO base +0.161 [0.113, 0.206], pooled +0.101 [0.084, 0.119]; against SD ratios 1.53 / 1.40 (1.4-1.5x) | results/lookahead/mechanism/tables/dispersion_ratios.md (fixture tables/dispersion_by_k_ratios.md) columns winner_z_diff, sd_ratio |
| PTO pair yield at tau=0.1, base policy (the median-over-iterations and iters 8-10 parts: appendix B) | K0 0.824 vs K5 0.935; K0 margins x1.530 -> 0.872; share of gap closed 0.437; median 0.572 over iters 1-7; iters 8-10 K5 yield level with (iter 8: 0.737 vs 0.733) or below K0 (0.794/0.809, 0.666/0.685) | results/lookahead/mechanism/tables/dispersion_tau.md (fixture tables/dispersion_by_k_tau.md) rows tau=0.10 |
| w_len PTO_LA5 by train_iter 1..10 | 49.480, 45.147, 58.868, 29.085, 32.449, 28.301, 53.788, 20.000, 26.704, 7.016; SE 7.3-13.9; w/SE 2.28-6.51 at iters 1-9, 0.956 at iter 10 | results/lookahead/behaviour/tables/selection.md (fixture tables/session_shape_stability_selection.md) (from results/arms/preference/tables/gpt-4o-mini/update_lexical_push.md, formerly L5) |
| w_len PTO_LA0 max abs w/SE | 1.695 (train_iter 4, w=-21.730) | results/lookahead/behaviour/tables/selection.md (fixture tables/session_shape_stability_selection.md) |
| GRPO w_len at iter 1 (appendix B) | GRPO_LA0 -141.700; GRPO_LA5 -39.799; turns positive late (GRPO_LA5 iter5 +10.778; GRPO_LA0 iter10 +77.349) | results/lookahead/behaviour/tables/selection.md (fixture tables/session_shape_stability_selection.md) |
| Reward faithfulness at MCL=12, same grader | 0.865 / 0.892 / 0.860 / 0.864 (PTO_LA0/LA5, GRPO_LA0/LA5) | results/lookahead/mechanism/tables/faithfulness_curve.md (fixture tables/reward_faithfulness_curve.md) rows n_turns=12 |
| Matched-policy K0-K5 faithfulness delta (the held-out-eval pair: appendix B) | PTO +0.004 [-0.067, 0.074]; GRPO +0.015 [-0.023, 0.057] (primary eval); held-out eval +0.007 / -0.014 | results/lookahead/mechanism/tables/faithfulness_k_summary.md (fixture tables/reward_faithfulness_k_summary.md) rows cut=train_iter_1 |
| Pooled PTO K5 faithfulness lead (trajectory effect) | -0.027 [-0.047, -0.006] over matched iters | results/lookahead/mechanism/tables/faithfulness_k_summary.md (fixture reward_faithfulness_k_summary.md) row PTO matched_iters gpt-4o-mini |
| GRPO_LA0 cross-grader faithfulness collapse (appendix B) | held-out 0.798 (iter 1) -> 0.630 (iter 7), 0.699 (iter 10); same-grader 0.895 (iter 1) -> 0.848 (iter 7) -> 0.829 (iter 10) | results/lookahead/mechanism/tables/faithfulness_curve_by_iter.md (fixture tables/reward_faithfulness_curve_by_iter.md); digest reward_faithfulness finding 6 |
| Ended-early tail share | PTO_LA5 0.228 [0.225, 0.232], n=59,868; GRPO_LA5 0.192 [0.189, 0.196], n=55,552 | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) pooled rows |
| Patient-closed share; failure tails | patient_closed 0.211 / 0.161; no_tail 0.005 / 0.022; after_therapist 0.011 / 0.009; therapist_stalled 0.002 / 0.000 -> failure tails 1.8% / 3.1% (even-length tails 939/59,868 = 1.6% / 1,726/55,552 = 3.1%), quoted as 2-3% | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) pooled rows; results/lookahead/mechanism/tables/tail_score_by_realized_turns.md (fixture tail_audit_score_by_realized_turns.csv) pooled n |
| GRPO tail loop rate | 0.279 (iter 1) -> 0.002 (iter 5) | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) column tail_loop_rate |
| Within-group ended-early penalty | PTO -0.034 dz -0.2437 -> 0.24 (3,809 groups); GRPO -0.051 dz -0.2546 -> prints as 0.25 (2,878); all per-iteration p_holm <= .004 | results/lookahead/mechanism/tables/tail_within_group.md (fixture tables/tail_audit_within_group.md); analysis/out/tail_audit.json numbers.within_group.*.pooled (fixture) = results/lookahead/mechanism/tables/tails_numbers.json :: within_group.{PTO_LA5,GRPO_LA5}.pooled |
| Argmax relative risk ended-early vs full | PTO 0.770 [0.738, 0.803] (23% less likely); GRPO 0.787 [0.749, 0.823] (21%); PTO iter 1 -> 10: 0.768 -> 0.667, dz -0.241 -> -0.381 | results/lookahead/mechanism/tables/tail_within_group.md (fixture tables/tail_audit_within_group.md) |
| Penalty by realized turns (pooled) | rt0 -0.067/-0.071; rt2 -0.064/-0.071; rt4 -0.079/-0.110; rt1 -0.012/-0.009; rt3 -0.005/-0.001; rt5 +0.004/+0.004 (PTO/GRPO) | results/lookahead/mechanism/tables/tail_score_by_realized_turns.md (fixture tables/tail_audit_score_by_realized_turns.md) pooled rows |
| Tail therapist turns mirror candidate (appendix B) | ?/turn candidate 0.447 vs tail 0.424 (PTO); 0.663 vs 0.700 (GRPO) | results/lookahead/mechanism/tables/tail_cues_by_iter.md (fixture tables/tail_audit_cues_by_iter.md) pooled rows |
| Update-direction cosine PTO vs GRPO (corrected) | 0.739 under K=5; 0.317 under K=0 (each arm's iterations pooled: GRPO_LA5 1-5, the others 1-10) | Exp3_PTO_GRPO/eda/results/arms/preference/tables/gpt-4o-mini/update_direction_cosines.md (all four arms in one table; formerly results/L5 and L0 tables/6_preference/gpt-4o-mini/); eda_analysis/pref.py direction_by_arm |
| Rule swap on same data (raw cosine) | 0.961 / 0.984 (K=5); 0.908 / 0.988 (K=0) | Exp3_PTO_GRPO/eda/results/arms/preference/tables/gpt-4o-mini/weighting_decomposition.md (K column; formerly results/L{5,0}/tables/6_preference/gpt-4o-mini/) |
| Same rule, data differs (corrected cosine) | K=5 0.729 / 0.732; K=0 0.397 / 0.324 | Exp3_PTO_GRPO/eda/results/arms/preference/tables/gpt-4o-mini/weighting_decomposition.md rows K=5 / K=0 (formerly L5, L0) |
| MICI own-base retention | >1 in every defined cell from iteration 3 (1.063-4.626); CI covers 1 in four of those cells (GRPO_LA0 I10 1.063 [0.944, 1.210]; PTO_LA0 I5 1.832 [0.938, 2.956]; PTO_LA5 I3 1.800 [0.693, 3.157]; PTO_LA5 I4 1.320 [0.804, 2.080]); the one earlier defined cell PTO_LA5 I2 = 0.709 [-0.367, 1.810]; endpoints PTO_LA0 I10 1.657 [1.303, 2.151]; PTO_LA5 I10 2.442 [1.496, 3.843]; GRPO_LA5 I5 2.440 [1.759, 3.802] | results/lookahead/transfer/tables/k_retention.md (fixture tables/cross_k_multijudge_retention.md) rows metric=MICI ref_kind=own_base |

## Appendix B
| claim | value | source |
|---|---|---|
| Candidate counts audited | PTO_LA5 59,868; GRPO_LA5 55,552 | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) pooled rows |
| GRPO_LA5 log coverage iters 1-2 | 0.500 / 0.712 | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md); results/compute/cost/tables/api_calls.md (fixture tail_audit_api_calls.md) |
| Role-label / realized-turn consistency | 0 mismatches over 115k tails; SESSION ENDED marker in 0 tails | digest tail_audit finding 2 (script stdout) |
| Fingerprint validation in eval CSVs | 398/398 patient-closed rows end in whitespace vs 279/7,306 = 3.8% of ordinary patient rows | digest tail_audit finding 2 (tail_audit.py stdout) |
| Wrap-up cue rate | patient-closed 0.897 (PTO) / 0.876 (GRPO) vs full open 0.248 / 0.119 | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) columns wrapup_cue_rate_* pooled |
| Table tail_by_iter (all cells) | per-iteration ended_early / patient_closed / no_tail / after_therapist / tail_loop shares; therapist_stalled_share max 0.0043 (PTO iter 10; 0.0028 iter 8), pooled 0.0019 PTO / 0.0005 GRPO | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md); analysis/out/tail_audit.json by_iter (fixture) = results/lookahead/mechanism/tables/tails_numbers.json :: by_iter.* |
| full_closed_at_turn5 share | 0.135 PTO / 0.098 GRPO | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) pooled |
| Ended-early peak at iteration 5 | PTO 0.354, GRPO 0.252; PTO iter 10 0.182 | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) |
| Table tail_within (all cells) | delta_ee_minus_full, dz, RR, p_chosen_is_ee, base_ee_rate per iteration; RR CIs pooled [0.738, 0.803] / [0.749, 0.823] | results/lookahead/mechanism/tables/tail_within_group.md (fixture tables/tail_audit_within_group.md) |
| P(argmax) by realized turns | rt0 .061/.069, rt1 .097/.102, rt2 .095/.035, rt3 .113/.123, rt4 .044/.049, rt5 .133/.130 | results/lookahead/mechanism/tables/tail_score_by_realized_turns.md (fixture tables/tail_audit_score_by_realized_turns.md) pooled |
| Lexical cues candidate vs tail | affirm 0.068/0.058 (PTO), 0.041/0.031 (GRPO); effusive 0.030/0.040, 0.006/0.006; PTO cand len 306.7 -> 1006.3, ?/turn 0.607 -> 0.274, effusive 0.004 -> 0.078; GRPO 324.2 -> 584.9 | results/lookahead/mechanism/tables/tail_cues_by_iter.md (fixture tables/tail_audit_cues_by_iter.md) |
| Table api (all cells) | oracle calls, Mchars, patient calls, totals and K5/K0 ratios over iters 1-5 / 1-10 | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) |
| Tail patient calls per candidate | 2.646 PTO; 2.873 GRPO | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) rows all K5 iters |
| Whole-run API totals | GRPO_LA0 302,541 oracle + 15,449 patient; GRPO_LA5 141,237 + 198,320; PTO_LA0 99,622 + 18,562; PTO_LA5 121,806 + 181,008 | analysis/out/tail_audit.json keys api_totals.* (fixture) = results/lookahead/mechanism/tables/tails_numbers.json :: api_totals.* (column sums of results/compute/cost/tables/api_calls.md (fixture tables/tail_audit_api_calls.md)) |
| Run_Eval scoring calls | 96 x 8 = 768 per model state per grader | results/compute/cost/tables/api_calls.md (fixture tables/tail_audit_api_calls.md) column eval_scoring_calls_run_eval |
| PTO_LA5 iteration-5 incident | 1,654 oracle retries; 389 unscored candidates dropped; 148 zero-turn tails among 3,740 simulated candidates (95 = 2.8% among the 3,351 scored; no_tail_share 0.028 is over scored candidates); 468 branch points (iters 4/6: 613/663); 24.8% of groups with an unscored candidate; 1 group dropped | results/compute/cost/tables/api_calls.md (fixture tables/tail_audit_api_calls.md) row PTO_LA5/5 (n_candidates 3,744 incl. 4 not simulated); results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tail_audit_by_iter.md) (n_candidates 3,351, no_tail_share 0.028); direct recount of iteration_5/eda/generations.jsonl 2026-08-18; results/lookahead/mechanism/tables/dispersion_by_iter.md (fixture dispersion_by_k_by_iter.md) (frac_groups_nan_cand) |
| Other PTO iterations' retries | 0-15 | results/compute/cost/tables/api_calls.md (fixture tables/tail_audit_api_calls.md) column oracle_retries |

## Abstract
| claim | value | source |
|---|---|---|
| Design: 39 model states x 96 personas x 8 instruments, two graders | 11+11+11+6 = 39; 39 x 8 x 96 = 29,952 cells per grader | results/lookahead/reward/tables/k_levels.md (fixture tables/k_contrast_headline_levels.md) (rows); CONFIG FACTS |
| PTO: K=0 higher at 8/10 trained iterations (primary Q1Q2) | iters 1-4, 6-9 positive; 5 and 10 negative | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) PTO gpt-4o-mini col |
| PTO: Holm-sig K0>K5 at one iteration (oracle) / three (held-out) | primary iter 6 (+0.257, dz .42); held-out iters 5/6/8 (dz .330/.511/.337) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md); results/lookahead/reward/tables/k_summary.md (fixture k_contrast_headline_summary.md) (PTO, Q1Q2) |
| PTO endpoint null | primary -0.047 (dz -.10); held-out +0.199 (dz .31, not Holm-sig) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) row 10; results/lookahead/transfer/tables/k_pairs.md (fixture cross_k_multijudge_pairs.md) |
| GRPO: K5 > K0 at iters 4-5, held-out dz -.37 / -.43 | -0.233 (dz -.374), -0.311 (dz -.429), both p_holm .006 | results/lookahead/reward/tables/k_paired_grpo_claude-haiku-4-5.md (fixture tables/k_contrast_headline_grpo_heldout.md) Q1Q2 iters 4-5 |
| GRPO K5 censored at five iterations | (design) | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) n_iters |
| Cost 1.9-2.4x per step or iteration | GRPO step ratio 1.965/1.962/1.911 (iters 3-5); GRPO per-iteration 1.94; PTO per-iteration 2.42 | results/compute/cost/tables/step_multiplier.md (fixture tables/compute_axis_step_multiplier.md); results/compute/cost/tables/compute_by_arm.md (fixture compute_axis_by_arm.md) |
| API calls "about twice" | PTO 145,291/69,941 = 2.077; GRPO 338,476/150,427 = 2.250 (iters 1-5) | results/compute/cost/tables/api_ratio.md (fixture tables/tail_audit_api_ratio.md) |
| At matched GPU-h K5 never sig. beats K0 on the training oracle | PTO top +0.047 dz .096 n.s.; GRPO top +0.038 dz .074 n.s. (same I4/I8 pair from 23.21 h on); PTO negative at every intermediate budget but n.s. at 16.17/18.03 h; GRPO negative at 7.80 (n.s.)/13.27/18.31 h | results/compute/cost/tables/budget_sweep_{PTO,GRPO}_K_gpt-4o-mini.md (fixture tables/compute_axis_budget_sweep_{PTO,GRPO}_K_gpt-4o-mini.md) |
| Over-praise channel closes, per turn: PTO dz 1.2-1.6, GRPO 0.3-0.7 | PTO iter 10: dz 1.163 (primary), 1.648 (held-out); GRPO iter 5: dz 0.301 (primary, p_holm .047), 0.687 (held-out) | results/lookahead/behaviour/tables/k_channels_{pto,grpo}_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/k_contrast_headline_channels_{pto,grpo}_{primary,heldout}.md) MICI_OverPraise_rate |
| 'in PTO, MI-inconsistency relocates to unsolicited advice' | PTO advice per turn K5-higher Holm-sig at iters 4,8,9 (primary) / 4,6,7,8,9,10 (held-out); at iter 10 per session -0.312 n.s. / -1.948 p_holm <.001. GRPO iter 5: per turn -0.062 dz -0.379 p_holm .007 primary but n.s. held-out; per session -0.031 / +1.042 (K0 higher) -> relocation is a PTO result at the endpoints | results/lookahead/behaviour/tables/k_channels_summary.md (fixture tables/k_contrast_headline_channels_summary.md); k_channels_{pto,grpo}_{gpt-4o-mini,claude-haiku-4-5}.md (fixture channels_{pto,grpo}_{primary,heldout}.md) |
| 'no endpoint contrast on Q1Q2 survives correction' | PTO I10: Q1Q2 p_holm .695 / .129; but Q2 held-out, MITI held-out, MICI both, PCT held-out ARE Holm-sig at the endpoint, hence the Q1Q2 qualifier | results/lookahead/reward/tables/k_endpoints.md (fixture tables/cross_k_multijudge_endpoints.md) rows PTO_LA5_I10 - PTO_LA0_I10 |
| Session length opposite by optimizer | PTO iter 10 K5 +8.312 utterances longer; GRPO iter 5 K5 -8.104 shorter | results/lookahead/behaviour/tables/length_kcontrast.md (fixture tables/session_shape_stability_length_kcontrast.md) |
| ICLR ordering reproduced at 7/7 iterations (the arm-level dz was cut from the abstract; sec 8 keeps it) | K0-K5 negative at all 7 under both graders; gpt-4o-mini arm-level -0.132 dz -0.543 | results/lookahead/replication/tables/crossgen_kcontrast.md (fixture tables/crossgen_exp1_kcontrast.md); results/lookahead/replication/tables/crossgen_kcontrast_summary.md (fixture crossgen_exp1_kcontrast_summary.md) |
| Exp1 re-score size | 1,440 = 15 states x 96 | results/lookahead/replication/tables/crossgen_levels.md (fixture tables/crossgen_exp1_levels.md) |

## Introduction
| claim | value | source |
|---|---|---|
| Poster: K=5 higher + lower-variance scores, shorter conversations, K gain sig only on Q2 between post-hoc best iterations, one optimizer, eval = training reward | (poster text) | papers/2025_iclr_pto_lookahead/submitted/paper.pdf (Table 1, Fig 4, pp. 8-9) |
| Matched MCL 12, M=G=8 | (config) | run_metadata.json (CONFIG FACTS) |
| Cost "about 2x per step, iteration or API call" | GRPO step 1.965/1.962/1.911; GRPO per-iteration 1.94; PTO per-iteration 2.42; API 2.08/2.25 | results/compute/cost/tables/step_multiplier.md (fixture tables/compute_axis_step_multiplier.md); results/compute/cost/tables/compute_by_arm.md (fixture compute_axis_by_arm.md); results/compute/cost/tables/api_ratio.md (fixture tail_audit_api_ratio.md) |
| Poster's own transcripts keep their ordering under the new grader | Final K0-K5 negative at all 7 iterations, gpt-4o-mini and GPT-3.5 | results/lookahead/replication/tables/crossgen_kcontrast.md (fixture tables/crossgen_exp1_kcontrast.md) |

## Discussion
| claim | value | source |
|---|---|---|
| DiD iter 5 held-out dz .53; PTO ahead at K0, GRPO ahead at K5 | gap_K0 +0.265, gap_K5 -0.219, DiD +0.484 dz .525 | results/lookahead/reward/tables/k_did.md (fixture tables/cross_k_multijudge_did.md) Q1Q2 iter 5 |
| PTO K0 edge: 1 Holm-sig iteration (oracle) / 3 (held-out) | iter 6 / iters 5,6,8 | results/lookahead/reward/tables/k_summary.md (fixture tables/k_contrast_headline_summary.md) (PTO, Q1Q2) |
| GRPO K5 matched-budget lead only under held-out judge | primary top +0.038 n.s.; held-out +0.161 dz .310 p_holm .020 | results/compute/cost/tables/budget_sweep_GRPO_K_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/compute_axis_budget_sweep_GRPO_K_{gpt-4o-mini,claude-haiku-4-5}.md) |
| Both K5 arms trail until the last one or two budgets of their sweeps | PTO K5-K0 negative through 18.03 h (Holm-sig through 14.60 h), +0.047 at 19.68 h; GRPO negative through 18.31 h, +0.038 (tie) at 23.21 h under the oracle and +0.147 (dz .331 p_holm .012) at 23.21 h under the held-out judge | results/compute/cost/tables/budget_sweep_{PTO,GRPO}_K_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/compute_axis_budget_sweep_{PTO,GRPO}_K_{gpt-4o-mini,claude-haiku-4-5}.md) |
| WAI-SR total unmoved, composition shifts (PTO endpoint) | PTO iter 10 WAI_total -0.038 / +0.104 n.s.; bond_excess +0.221 / +0.270 p_holm .001. GRPO iter 5: WAI_total -0.255 dz -0.446 p_holm <.001 primary (moves), bond_excess n.s. | results/lookahead/behaviour/tables/wai_kcontrast.md (fixture tables/held_out_instruments_wai_kcontrast.md) |
| MICI per-turn falls under K5 on both graders, per-session on one (PTO endpoint) | PTO iter 10 rate dz .708 / .655; session total dz .446 (p_holm .001) / .099 (p_holm .837). GRPO iter 5: rate -0.063 dz -.243 n.s. / -0.018 n.s. (absent); session +0.323 n.s. / +1.885 dz .342 p_holm .003 (other grader) | results/lookahead/behaviour/tables/k_channels_{pto,grpo}_{gpt-4o-mini,claude-haiku-4-5}.md (fixture tables/k_contrast_headline_channels_{pto,grpo}_{primary,heldout}.md) |
| K5 gains on Warms-up personas; held-out K0 PTO edge on the Cooperative third | PCT Warms up GRPO dz -.896/-.786, PTO -.252/-.459; GRPO Q1Q2 Warms up dz -.557/-.589; held-out PTO Q1Q2 @10 Cooperative +0.571 dz .773 p_holm .002, Warms up -0.072 n.s. | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture tables/held_out_instruments_hetero.md) |
| Rescaling 1.4-1.8x, same factor | margin/SD ratios 1.44-1.56 / 1.40-1.53 base; pooled 1.50/1.48 (PTO), 1.56/1.54 (GRPO) | results/lookahead/mechanism/tables/dispersion_ratios.md (fixture tables/dispersion_by_k_ratios.md) |
| Ended-early candidates 21-23% less likely to be argmax; grows in PTO | RR 0.770 (PTO, 23%) / 0.787 (GRPO, 21%); PTO 0.768 -> 0.667 (iters 1 -> 10) | results/lookahead/mechanism/tables/tail_within_group.md (fixture tables/tail_audit_within_group.md) |
| Ended-early almost always patient closes | patient_closed 0.211 / 0.161 of 0.228 / 0.192 ended early | results/lookahead/mechanism/tables/tail_audit_by_iter.md (fixture tables/tail_audit_by_iter.md) pooled |
| PTO K5 length channel at every iteration | w_len +49.480 ... +7.016, all positive | results/lookahead/behaviour/tables/selection.md (fixture tables/session_shape_stability_selection.md) |
| Exp1 -> Exp3 changes | 7B 4-bit -> 1B bf16; GPT-3.5 -> gpt-4o-mini patient; V1 -> V3 prompts; tree from turn 0 -> seeded at MCL 12 with a prefix sliced from the current policy's eval sessions; hyperparameters matched across optimizers. NOT a change: Exp1 was also iterative and regenerated pref data with the current agent each iteration (Exp1 CLAUDE.md step 3) | Exp1_ICLR2025/CLAUDE.md; root CLAUDE.md (data lineage); CONFIG FACTS |
| Training oracle under-reports harm channel 1.1-4.6x from iteration 3 on | MICI own-base retention 1.063-4.626 over defined cells from iter 3; PTO_LA5 I2 = 0.709 | results/lookahead/transfer/tables/k_retention.md (fixture tables/cross_k_multijudge_retention.md) MICI own_base |
| Poster gain between post-hoc best iterations | L0 M4 vs L5 M7 | papers/2025_iclr_pto_lookahead/submitted/paper.pdf; results/lookahead/replication/tables/crossgen_kcontrast_summary.md (fixture tables/crossgen_exp1_kcontrast_summary.md) |

## Limitations
| claim | value | source |
|---|---|---|
| Noise floor: base-vs-base Q1Q2 primary | PTO -0.003; GRPO +0.104 (dz .1148 -> prints as 0.11) | results/lookahead/reward/tables/k_table1.md (fixture tables/k_contrast_headline_table1.md) row 0 |
| GRPO K5 stopped at iter 5 at GRPO K0's ten-iteration budget | 27.078 vs 27.906 GPU-h | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) |
| Only K in {0,5} at MCL=12 | (design) | CONFIG FACTS |
| MICI retention 1.1-4.6x | 1.063-4.626 own-base, defined cells from iteration 3 on; PTO K5 iteration 2 = 0.709 | results/lookahead/transfer/tables/k_retention.md (fixture tables/cross_k_multijudge_retention.md) |
| ICC on four K=0 anchors only | Q1 .990 / Q2 .976 / MICI .916 | CONFIG FACTS |
| Generation under-count ~0.1 h/iter; resume-gap imputations | n_imputed GRPO_LA0 3, GRPO_LA5 2, PTO_LA0 1, PTO_LA5 2 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md); compute_axis caveats |
| Cross-judge sweep weakens GRPO K5 lead without removing it (numbers in sec 5 / appendix A) | +0.166 dz .266 (select oracle / eval judge); +0.048 n.s. (select judge / eval oracle) | results/compute/cost/tables/budget_sweep_crossjudge_verdicts.md (fixture tables/compute_axis_budget_sweep_crossjudge_verdicts.md) |
| Cooperative third saturates Q1Q2 on the oracle; no K contrast at the matched endpoints (88-100% figure in sec 6 and limitations) | share_ge_4.5 0.875-1.000; matched_final K contrasts n.s.; own_best GRPO I8 vs I4 IS sig (+0.168 dz 1.584) | results/lookahead/behaviour/tables/hetero_kcontrast.md (fixture tables/held_out_instruments_hetero.md); results/lookahead/replication/tables/ceiling.md (fixture session_shape_stability_ceiling.md) |
| Ethics: 8-28 A100-hours per arm | 8.119 / 19.681 / 27.906 / 27.078 | results/compute/cost/tables/compute_by_arm.md (fixture tables/compute_axis_by_arm.md) |
