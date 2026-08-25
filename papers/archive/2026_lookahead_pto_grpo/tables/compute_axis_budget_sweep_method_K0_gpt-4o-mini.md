Budget sweep, K=0: PTO vs GRPO (PTO_LA0 = arm_a vs GRPO_LA0 = arm_b), grader = training oracle (gpt-4o-mini). At each of arm_a's cumulative GPU-h budgets both arms are represented by the best checkpoint they could have reached for that money (best on select_metric under this grader; MICI selects the LOWEST), and the contrast is scored on eval_metric paired on persona_id (n personas; bootstrap 95% CI; Wilcoxon p; Holm within this table's (select_metric, eval_metric) family over the unique checkpoint pairs). Method contrast: mean_delta = PTO - GRPO (+ => PTO higher). MICI is lower-is-better. Rows select_metric=Q1Q2 -> eval_metric=MICI score the Q1Q2-selected checkpoints on MICI (does the reward-selected policy carry the hack?). Mirrors eda_analysis.compute.budget_sweep row-for-row on Q1Q2. GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h).

| judge | budget_gpu_h | select_metric | eval_metric | best_iter_a | best_iter_b | model_a | model_b | cum_gpu_h_a | cum_gpu_h_b | mean_a | mean_b | n | mean_delta | dz | ci_lo | ci_hi | p | p_holm | n_unique_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 2.800 | Q1Q2 | Q1Q2 | 3 | 1 | PTOExp3_LA0_I3 | GRPOExp3_LA0_I1 | 2.800 | 2.610 | 3.815 | 3.269 | 96 | 0.546 | 0.544 | 0.353 | 0.759 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 3.900 | Q1Q2 | Q1Q2 | 4 | 1 | PTOExp3_LA0_I4 | GRPOExp3_LA0_I1 | 3.900 | 2.610 | 4.008 | 3.269 | 96 | 0.739 | 0.810 | 0.561 | 0.925 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 4.660 | Q1Q2 | Q1Q2 | 5 | 1 | PTOExp3_LA0_I5 | GRPOExp3_LA0_I1 | 4.660 | 2.610 | 4.014 | 3.269 | 96 | 0.745 | 0.782 | 0.562 | 0.944 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 5.370 | Q1Q2 | Q1Q2 | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 4.154 | 3.359 | 96 | 0.795 | 0.909 | 0.621 | 0.972 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 6.070 | Q1Q2 | Q1Q2 | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 4.154 | 3.359 | 96 | 0.795 | 0.909 | 0.621 | 0.972 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 6.880 | Q1Q2 | Q1Q2 | 8 | 2 | PTOExp3_LA0_I8 | GRPOExp3_LA0_I2 | 6.880 | 5.250 | 4.221 | 3.359 | 96 | 0.861 | 0.993 | 0.689 | 1.044 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 7.490 | Q1Q2 | Q1Q2 | 9 | 2 | PTOExp3_LA0_I9 | GRPOExp3_LA0_I2 | 7.490 | 5.250 | 4.238 | 3.359 | 96 | 0.879 | 1.068 | 0.718 | 1.050 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 8.120 | Q1Q2 | Q1Q2 | 10 | 2 | PTOExp3_LA0_I10 | GRPOExp3_LA0_I2 | 8.120 | 5.250 | 4.260 | 3.359 | 96 | 0.900 | 1.086 | 0.744 | 1.074 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 2.800 | MICI | MICI | 1 | 1 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I1 | 0.920 | 2.610 | 0.209 | 0.228 | 96 | -0.018 | -0.067 | -0.070 | 0.035 | 0.564 | 0.607 | 2 |
| gpt-4o-mini | 3.900 | MICI | MICI | 1 | 1 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I1 | 0.920 | 2.610 | 0.209 | 0.228 | 96 | -0.018 | -0.067 | -0.070 | 0.035 | 0.564 | 0.607 | 2 |
| gpt-4o-mini | 4.660 | MICI | MICI | 1 | 1 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I1 | 0.920 | 2.610 | 0.209 | 0.228 | 96 | -0.018 | -0.067 | -0.070 | 0.035 | 0.564 | 0.607 | 2 |
| gpt-4o-mini | 5.370 | MICI | MICI | 1 | 2 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I2 | 0.920 | 5.250 | 0.209 | 0.169 | 96 | 0.040 | 0.147 | -0.014 | 0.097 | 0.303 | 0.607 | 2 |
| gpt-4o-mini | 6.070 | MICI | MICI | 1 | 2 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I2 | 0.920 | 5.250 | 0.209 | 0.169 | 96 | 0.040 | 0.147 | -0.014 | 0.097 | 0.303 | 0.607 | 2 |
| gpt-4o-mini | 6.880 | MICI | MICI | 1 | 2 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I2 | 0.920 | 5.250 | 0.209 | 0.169 | 96 | 0.040 | 0.147 | -0.014 | 0.097 | 0.303 | 0.607 | 2 |
| gpt-4o-mini | 7.490 | MICI | MICI | 1 | 2 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I2 | 0.920 | 5.250 | 0.209 | 0.169 | 96 | 0.040 | 0.147 | -0.014 | 0.097 | 0.303 | 0.607 | 2 |
| gpt-4o-mini | 8.120 | MICI | MICI | 1 | 2 | PTOExp3_LA0_I1 | GRPOExp3_LA0_I2 | 0.920 | 5.250 | 0.209 | 0.169 | 96 | 0.040 | 0.147 | -0.014 | 0.097 | 0.303 | 0.607 | 2 |
| gpt-4o-mini | 2.800 | Q1Q2 | MICI | 3 | 1 | PTOExp3_LA0_I3 | GRPOExp3_LA0_I1 | 2.800 | 2.610 | 0.252 | 0.228 | 96 | 0.025 | 0.097 | -0.025 | 0.075 | 0.159 | 0.318 | 7 |
| gpt-4o-mini | 3.900 | Q1Q2 | MICI | 4 | 1 | PTOExp3_LA0_I4 | GRPOExp3_LA0_I1 | 3.900 | 2.610 | 0.215 | 0.228 | 96 | -0.013 | -0.049 | -0.066 | 0.039 | 0.772 | 0.772 | 7 |
| gpt-4o-mini | 4.660 | Q1Q2 | MICI | 5 | 1 | PTOExp3_LA0_I5 | GRPOExp3_LA0_I1 | 4.660 | 2.610 | 0.289 | 0.228 | 96 | 0.061 | 0.211 | 0.001 | 0.114 | 0.009 | 0.028 | 7 |
| gpt-4o-mini | 5.370 | Q1Q2 | MICI | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 0.318 | 0.169 | 96 | 0.149 | 0.673 | 0.106 | 0.196 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 6.070 | Q1Q2 | MICI | 6 | 2 | PTOExp3_LA0_I6 | GRPOExp3_LA0_I2 | 5.370 | 5.250 | 0.318 | 0.169 | 96 | 0.149 | 0.673 | 0.106 | 0.196 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 6.880 | Q1Q2 | MICI | 8 | 2 | PTOExp3_LA0_I8 | GRPOExp3_LA0_I2 | 6.880 | 5.250 | 0.349 | 0.169 | 96 | 0.180 | 0.673 | 0.127 | 0.233 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 7.490 | Q1Q2 | MICI | 9 | 2 | PTOExp3_LA0_I9 | GRPOExp3_LA0_I2 | 7.490 | 5.250 | 0.471 | 0.169 | 96 | 0.302 | 1.163 | 0.249 | 0.353 | 0.000 | 0.000 | 7 |
| gpt-4o-mini | 8.120 | Q1Q2 | MICI | 10 | 2 | PTOExp3_LA0_I10 | GRPOExp3_LA0_I2 | 8.120 | 5.250 | 0.491 | 0.169 | 96 | 0.322 | 1.095 | 0.264 | 0.381 | 0.000 | 0.000 | 7 |
