Budget sweep, K=0: PTO vs GRPO (PTO_LA0 = arm_a vs GRPO_LA0 = arm_b), grader = held-out judge (Claude Haiku 4.5). At each of arm_a's cumulative GPU-h budgets both arms are represented by the best checkpoint they could have reached for that money (best on select_metric under this grader; MICI selects the LOWEST), and the contrast is scored on eval_metric paired on persona_id (n personas; bootstrap 95% CI; Wilcoxon p; Holm within this table's (select_metric, eval_metric) family over the unique checkpoint pairs). Method contrast: mean_delta = PTO - GRPO (+ => PTO higher). MICI is lower-is-better. Rows select_metric=Q1Q2 -> eval_metric=MICI score the Q1Q2-selected checkpoints on MICI (does the reward-selected policy carry the hack?). Mirrors eda_analysis.compute.budget_sweep row-for-row on Q1Q2. GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h).

| judge | budget_gpu_h | select_metric | eval_metric | best_iter_a | best_iter_b | model_a | model_b | cum_gpu_h_a | cum_gpu_h_b | mean_a | mean_b | n | mean_delta | dz | ci_lo | ci_hi | p | p_holm | n_unique_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 2.800 | Q1Q2 | Q1Q2 | 3 | 1 | PTOExp3_LA0_I3 | GRPOExp3_LA0_I1 | 2.800 | 2.610 | 2.480 | 2.074 | 96 | 0.406 | 0.576 | 0.269 | 0.542 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 3.900 | Q1Q2 | Q1Q2 | 4 | 1 | PTOExp3_LA0_I4 | GRPOExp3_LA0_I1 | 3.900 | 2.610 | 2.680 | 2.074 | 96 | 0.606 | 0.843 | 0.464 | 0.743 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 4.660 | Q1Q2 | Q1Q2 | 5 | 1 | PTOExp3_LA0_I5 | GRPOExp3_LA0_I1 | 4.660 | 2.610 | 2.752 | 2.074 | 96 | 0.678 | 0.964 | 0.540 | 0.823 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 5.370 | Q1Q2 | Q1Q2 | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 2.850 | 2.107 | 96 | 0.743 | 1.144 | 0.617 | 0.873 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 6.070 | Q1Q2 | Q1Q2 | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 2.850 | 2.107 | 96 | 0.743 | 1.144 | 0.617 | 0.873 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 6.880 | Q1Q2 | Q1Q2 | 8 | 2 | PTOExp3_LA0_I8 | GRPOExp3_LA0_I2 | 6.880 | 5.250 | 2.895 | 2.107 | 96 | 0.789 | 1.343 | 0.671 | 0.907 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 7.490 | Q1Q2 | Q1Q2 | 9 | 2 | PTOExp3_LA0_I9 | GRPOExp3_LA0_I2 | 7.490 | 5.250 | 2.921 | 2.107 | 96 | 0.814 | 1.394 | 0.697 | 0.929 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 8.120 | Q1Q2 | Q1Q2 | 9 | 2 | PTOExp3_LA0_I9 | GRPOExp3_LA0_I2 | 7.490 | 5.250 | 2.921 | 2.107 | 96 | 0.814 | 1.394 | 0.697 | 0.929 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 2.800 | MICI | MICI | 2 | 1 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I1 | 1.930 | 2.610 | 0.346 | 0.373 | 96 | -0.027 | -0.070 | -0.106 | 0.047 | 0.450 | 0.763 | 2 |
| claude-haiku-4-5 | 3.900 | MICI | MICI | 2 | 1 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I1 | 1.930 | 2.610 | 0.346 | 0.373 | 96 | -0.027 | -0.070 | -0.106 | 0.047 | 0.450 | 0.763 | 2 |
| claude-haiku-4-5 | 4.660 | MICI | MICI | 2 | 1 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I1 | 1.930 | 2.610 | 0.346 | 0.373 | 96 | -0.027 | -0.070 | -0.106 | 0.047 | 0.450 | 0.763 | 2 |
| claude-haiku-4-5 | 5.370 | MICI | MICI | 2 | 2 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I2 | 1.930 | 5.250 | 0.346 | 0.309 | 96 | 0.037 | 0.092 | -0.041 | 0.115 | 0.382 | 0.763 | 2 |
| claude-haiku-4-5 | 6.070 | MICI | MICI | 2 | 2 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I2 | 1.930 | 5.250 | 0.346 | 0.309 | 96 | 0.037 | 0.092 | -0.041 | 0.115 | 0.382 | 0.763 | 2 |
| claude-haiku-4-5 | 6.880 | MICI | MICI | 2 | 2 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I2 | 1.930 | 5.250 | 0.346 | 0.309 | 96 | 0.037 | 0.092 | -0.041 | 0.115 | 0.382 | 0.763 | 2 |
| claude-haiku-4-5 | 7.490 | MICI | MICI | 2 | 2 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I2 | 1.930 | 5.250 | 0.346 | 0.309 | 96 | 0.037 | 0.092 | -0.041 | 0.115 | 0.382 | 0.763 | 2 |
| claude-haiku-4-5 | 8.120 | MICI | MICI | 2 | 2 | PTOExp3_LA0_I2 | GRPOExp3_LA0_I2 | 1.930 | 5.250 | 0.346 | 0.309 | 96 | 0.037 | 0.092 | -0.041 | 0.115 | 0.382 | 0.763 | 2 |
| claude-haiku-4-5 | 2.800 | Q1Q2 | MICI | 3 | 1 | PTOExp3_LA0_I3 | GRPOExp3_LA0_I1 | 2.800 | 2.610 | 0.417 | 0.373 | 96 | 0.044 | 0.108 | -0.036 | 0.125 | 0.364 | 0.728 | 6 |
| claude-haiku-4-5 | 3.900 | Q1Q2 | MICI | 4 | 1 | PTOExp3_LA0_I4 | GRPOExp3_LA0_I1 | 3.900 | 2.610 | 0.389 | 0.373 | 96 | 0.016 | 0.043 | -0.060 | 0.091 | 0.571 | 0.728 | 6 |
| claude-haiku-4-5 | 4.660 | Q1Q2 | MICI | 5 | 1 | PTOExp3_LA0_I5 | GRPOExp3_LA0_I1 | 4.660 | 2.610 | 0.503 | 0.373 | 96 | 0.130 | 0.327 | 0.051 | 0.206 | 0.003 | 0.009 | 6 |
| claude-haiku-4-5 | 5.370 | Q1Q2 | MICI | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 0.581 | 0.309 | 96 | 0.272 | 0.764 | 0.200 | 0.342 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 6.070 | Q1Q2 | MICI | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 0.581 | 0.309 | 96 | 0.272 | 0.764 | 0.200 | 0.342 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 6.880 | Q1Q2 | MICI | 8 | 2 | PTOExp3_LA0_I8 | GRPOExp3_LA0_I2 | 6.880 | 5.250 | 0.671 | 0.309 | 96 | 0.362 | 0.943 | 0.287 | 0.438 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 7.490 | Q1Q2 | MICI | 9 | 2 | PTOExp3_LA0_I9 | GRPOExp3_LA0_I2 | 7.490 | 5.250 | 0.747 | 0.309 | 96 | 0.438 | 1.172 | 0.361 | 0.511 | 0.000 | 0.000 | 6 |
| claude-haiku-4-5 | 8.120 | Q1Q2 | MICI | 9 | 2 | PTOExp3_LA0_I9 | GRPOExp3_LA0_I2 | 7.490 | 5.250 | 0.747 | 0.309 | 96 | 0.438 | 1.172 | 0.361 | 0.511 | 0.000 | 0.000 | 6 |
