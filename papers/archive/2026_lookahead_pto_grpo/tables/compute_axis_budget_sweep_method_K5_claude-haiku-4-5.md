Budget sweep, K=5: PTO vs GRPO (PTO_LA5 = arm_a vs GRPO_LA5 = arm_b), grader = held-out judge (Claude Haiku 4.5). At each of arm_a's cumulative GPU-h budgets both arms are represented by the best checkpoint they could have reached for that money (best on select_metric under this grader; MICI selects the LOWEST), and the contrast is scored on eval_metric paired on persona_id (n personas; bootstrap 95% CI; Wilcoxon p; Holm within this table's (select_metric, eval_metric) family over the unique checkpoint pairs). Method contrast: mean_delta = PTO - GRPO (+ => PTO higher). MICI is lower-is-better. Rows select_metric=Q1Q2 -> eval_metric=MICI score the Q1Q2-selected checkpoints on MICI (does the reward-selected policy carry the hack?). Mirrors eda_analysis.compute.budget_sweep row-for-row on Q1Q2. GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h).

| judge | budget_gpu_h | select_metric | eval_metric | best_iter_a | best_iter_b | model_a | model_b | cum_gpu_h_a | cum_gpu_h_b | mean_a | mean_b | n | mean_delta | dz | ci_lo | ci_hi | p | p_holm | n_unique_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 8.940 | Q1Q2 | Q1Q2 | 4 | 1 | PTOExp3_LA5_I4 | GRPOExp3_LA5_I1 | 8.940 | 7.800 | 2.557 | 2.046 | 96 | 0.511 | 0.709 | 0.366 | 0.653 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 10.000 | Q1Q2 | Q1Q2 | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 2.579 | 2.046 | 96 | 0.533 | 0.789 | 0.396 | 0.663 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 12.700 | Q1Q2 | Q1Q2 | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 2.579 | 2.046 | 96 | 0.533 | 0.789 | 0.396 | 0.663 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 14.600 | Q1Q2 | Q1Q2 | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 2.735 | 2.142 | 96 | 0.594 | 0.821 | 0.454 | 0.737 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 16.170 | Q1Q2 | Q1Q2 | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 2.735 | 2.142 | 96 | 0.594 | 0.821 | 0.454 | 0.737 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 18.030 | Q1Q2 | Q1Q2 | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 2.735 | 2.142 | 96 | 0.594 | 0.821 | 0.454 | 0.737 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 19.680 | Q1Q2 | Q1Q2 | 7 | 3 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I3 | 14.600 | 18.310 | 2.735 | 2.586 | 96 | 0.149 | 0.295 | 0.051 | 0.248 | 0.007 | 0.007 | 4 |
| claude-haiku-4-5 | 8.940 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.397 | 0.362 | 96 | 0.035 | 0.088 | -0.045 | 0.116 | 0.320 | 0.320 | 1 |
| claude-haiku-4-5 | 10.000 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.397 | 0.362 | 96 | 0.035 | 0.088 | -0.045 | 0.116 | 0.320 | 0.320 | 1 |
| claude-haiku-4-5 | 12.700 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.397 | 0.362 | 96 | 0.035 | 0.088 | -0.045 | 0.116 | 0.320 | 0.320 | 1 |
| claude-haiku-4-5 | 14.600 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.397 | 0.362 | 96 | 0.035 | 0.088 | -0.045 | 0.116 | 0.320 | 0.320 | 1 |
| claude-haiku-4-5 | 16.170 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.397 | 0.362 | 96 | 0.035 | 0.088 | -0.045 | 0.116 | 0.320 | 0.320 | 1 |
| claude-haiku-4-5 | 18.030 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.397 | 0.362 | 96 | 0.035 | 0.088 | -0.045 | 0.116 | 0.320 | 0.320 | 1 |
| claude-haiku-4-5 | 19.680 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.397 | 0.362 | 96 | 0.035 | 0.088 | -0.045 | 0.116 | 0.320 | 0.320 | 1 |
| claude-haiku-4-5 | 8.940 | Q1Q2 | MICI | 4 | 1 | PTOExp3_LA5_I4 | GRPOExp3_LA5_I1 | 8.940 | 7.800 | 0.566 | 0.362 | 96 | 0.204 | 0.395 | 0.104 | 0.307 | 0.000 | 0.001 | 4 |
| claude-haiku-4-5 | 10.000 | Q1Q2 | MICI | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 0.621 | 0.362 | 96 | 0.260 | 0.544 | 0.169 | 0.358 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 12.700 | Q1Q2 | MICI | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 0.621 | 0.362 | 96 | 0.260 | 0.544 | 0.169 | 0.358 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 14.600 | Q1Q2 | MICI | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 0.581 | 0.370 | 96 | 0.211 | 0.642 | 0.142 | 0.275 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 16.170 | Q1Q2 | MICI | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 0.581 | 0.370 | 96 | 0.211 | 0.642 | 0.142 | 0.275 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 18.030 | Q1Q2 | MICI | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 0.581 | 0.370 | 96 | 0.211 | 0.642 | 0.142 | 0.275 | 0.000 | 0.000 | 4 |
| claude-haiku-4-5 | 19.680 | Q1Q2 | MICI | 7 | 3 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I3 | 14.600 | 18.310 | 0.581 | 0.518 | 96 | 0.063 | 0.178 | -0.006 | 0.133 | 0.068 | 0.068 | 4 |
