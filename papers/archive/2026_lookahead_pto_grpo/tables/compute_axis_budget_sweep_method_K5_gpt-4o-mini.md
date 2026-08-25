Budget sweep, K=5: PTO vs GRPO (PTO_LA5 = arm_a vs GRPO_LA5 = arm_b), grader = training oracle (gpt-4o-mini). At each of arm_a's cumulative GPU-h budgets both arms are represented by the best checkpoint they could have reached for that money (best on select_metric under this grader; MICI selects the LOWEST), and the contrast is scored on eval_metric paired on persona_id (n personas; bootstrap 95% CI; Wilcoxon p; Holm within this table's (select_metric, eval_metric) family over the unique checkpoint pairs). Method contrast: mean_delta = PTO - GRPO (+ => PTO higher). MICI is lower-is-better. Rows select_metric=Q1Q2 -> eval_metric=MICI score the Q1Q2-selected checkpoints on MICI (does the reward-selected policy carry the hack?). Mirrors eda_analysis.compute.budget_sweep row-for-row on Q1Q2. GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h).

| judge | budget_gpu_h | select_metric | eval_metric | best_iter_a | best_iter_b | model_a | model_b | cum_gpu_h_a | cum_gpu_h_b | mean_a | mean_b | n | mean_delta | dz | ci_lo | ci_hi | p | p_holm | n_unique_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 8.940 | Q1Q2 | Q1Q2 | 4 | 1 | PTOExp3_LA5_I4 | GRPOExp3_LA5_I1 | 8.940 | 7.800 | 3.888 | 3.272 | 96 | 0.616 | 0.636 | 0.432 | 0.815 | 0.000 | 0.000 | 6 |
| gpt-4o-mini | 10.000 | Q1Q2 | Q1Q2 | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 4.017 | 3.272 | 96 | 0.745 | 0.810 | 0.564 | 0.941 | 0.000 | 0.000 | 6 |
| gpt-4o-mini | 12.700 | Q1Q2 | Q1Q2 | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 4.017 | 3.272 | 96 | 0.745 | 0.810 | 0.564 | 0.941 | 0.000 | 0.000 | 6 |
| gpt-4o-mini | 14.600 | Q1Q2 | Q1Q2 | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 4.085 | 3.435 | 96 | 0.650 | 0.778 | 0.485 | 0.820 | 0.000 | 0.000 | 6 |
| gpt-4o-mini | 16.170 | Q1Q2 | Q1Q2 | 8 | 2 | PTOExp3_LA5_I8 | GRPOExp3_LA5_I2 | 16.170 | 13.270 | 4.144 | 3.435 | 96 | 0.709 | 0.887 | 0.554 | 0.880 | 0.000 | 0.000 | 6 |
| gpt-4o-mini | 18.030 | Q1Q2 | Q1Q2 | 9 | 2 | PTOExp3_LA5_I9 | GRPOExp3_LA5_I2 | 18.030 | 13.270 | 4.197 | 3.435 | 96 | 0.762 | 0.919 | 0.603 | 0.941 | 0.000 | 0.000 | 6 |
| gpt-4o-mini | 19.680 | Q1Q2 | Q1Q2 | 10 | 3 | PTOExp3_LA5_I10 | GRPOExp3_LA5_I3 | 19.680 | 18.310 | 4.307 | 3.862 | 96 | 0.445 | 0.673 | 0.313 | 0.583 | 0.000 | 0.000 | 6 |
| gpt-4o-mini | 8.940 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.222 | 0.228 | 96 | -0.006 | -0.016 | -0.071 | 0.060 | 0.986 | 1.000 | 2 |
| gpt-4o-mini | 10.000 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.222 | 0.228 | 96 | -0.006 | -0.016 | -0.071 | 0.060 | 0.986 | 1.000 | 2 |
| gpt-4o-mini | 12.700 | MICI | MICI | 1 | 1 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I1 | 2.170 | 7.800 | 0.222 | 0.228 | 96 | -0.006 | -0.016 | -0.071 | 0.060 | 0.986 | 1.000 | 2 |
| gpt-4o-mini | 14.600 | MICI | MICI | 1 | 2 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I2 | 2.170 | 13.270 | 0.222 | 0.228 | 96 | -0.005 | -0.018 | -0.058 | 0.052 | 0.894 | 1.000 | 2 |
| gpt-4o-mini | 16.170 | MICI | MICI | 1 | 2 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I2 | 2.170 | 13.270 | 0.222 | 0.228 | 96 | -0.005 | -0.018 | -0.058 | 0.052 | 0.894 | 1.000 | 2 |
| gpt-4o-mini | 18.030 | MICI | MICI | 1 | 2 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I2 | 2.170 | 13.270 | 0.222 | 0.228 | 96 | -0.005 | -0.018 | -0.058 | 0.052 | 0.894 | 1.000 | 2 |
| gpt-4o-mini | 19.680 | MICI | MICI | 1 | 2 | PTOExp3_LA5_I1 | GRPOExp3_LA5_I2 | 2.170 | 13.270 | 0.222 | 0.228 | 96 | -0.005 | -0.018 | -0.058 | 0.052 | 0.894 | 1.000 | 2 |
| gpt-4o-mini | 8.940 | Q1Q2 | MICI | 4 | 1 | PTOExp3_LA5_I4 | GRPOExp3_LA5_I1 | 8.940 | 7.800 | 0.326 | 0.228 | 96 | 0.098 | 0.301 | 0.033 | 0.162 | 0.004 | 0.021 | 6 |
| gpt-4o-mini | 10.000 | Q1Q2 | MICI | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 0.325 | 0.228 | 96 | 0.097 | 0.307 | 0.034 | 0.161 | 0.003 | 0.017 | 6 |
| gpt-4o-mini | 12.700 | Q1Q2 | MICI | 5 | 1 | PTOExp3_LA5_I5 | GRPOExp3_LA5_I1 | 10.000 | 7.800 | 0.325 | 0.228 | 96 | 0.097 | 0.307 | 0.034 | 0.161 | 0.003 | 0.017 | 6 |
| gpt-4o-mini | 14.600 | Q1Q2 | MICI | 7 | 2 | PTOExp3_LA5_I7 | GRPOExp3_LA5_I2 | 14.600 | 13.270 | 0.263 | 0.228 | 96 | 0.035 | 0.140 | -0.012 | 0.087 | 0.134 | 0.134 | 6 |
| gpt-4o-mini | 16.170 | Q1Q2 | MICI | 8 | 2 | PTOExp3_LA5_I8 | GRPOExp3_LA5_I2 | 16.170 | 13.270 | 0.290 | 0.228 | 96 | 0.063 | 0.259 | 0.017 | 0.110 | 0.007 | 0.028 | 6 |
| gpt-4o-mini | 18.030 | Q1Q2 | MICI | 9 | 2 | PTOExp3_LA5_I9 | GRPOExp3_LA5_I2 | 18.030 | 13.270 | 0.269 | 0.228 | 96 | 0.042 | 0.191 | -0.002 | 0.087 | 0.027 | 0.054 | 6 |
| gpt-4o-mini | 19.680 | Q1Q2 | MICI | 10 | 3 | PTOExp3_LA5_I10 | GRPOExp3_LA5_I3 | 19.680 | 18.310 | 0.264 | 0.308 | 96 | -0.044 | -0.160 | -0.096 | 0.013 | 0.018 | 0.054 | 6 |
