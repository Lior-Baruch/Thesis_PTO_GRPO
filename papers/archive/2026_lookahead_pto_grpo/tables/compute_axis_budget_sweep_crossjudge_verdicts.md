Top-of-sweep verdicts (each contrast at arm_a's LAST cumulative budget) under every (select_judge, eval_judge) combination, from budget_sweep_crossjudge. A verdict that holds only when the same grader selects and scores is a selection artefact; the honest_selection rows are the ones to quote. + mean_delta => arm_a higher (K contrast: mean_delta = K5 - K0 (arm_a=LA5, as the tracked EDA table); delta_K0_minus_K5 = -mean_delta is the paper's convention (+ => K=0 higher). Method contrast: mean_delta = PTO - GRPO (+ => PTO higher).) Paired on persona_id; p_holm within the (contrast, select_judge, eval_judge) family. GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h).

| contrast | arm_a | arm_b | select_judge | eval_judge | honest_selection | budget_gpu_h | best_iter_a | best_iter_b | n | mean_delta | dz | ci_lo | ci_hi | p | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRPO_K | GRPO_LA5 | GRPO_LA0 | claude-haiku-4-5 | claude-haiku-4-5 | False | 27.080 | 5 | 3 | 96 | 0.161 | 0.310 | 0.057 | 0.263 | 0.007 | 0.020 | arm_a > arm_b |
| GRPO_K | GRPO_LA5 | GRPO_LA0 | claude-haiku-4-5 | gpt-4o-mini | True | 27.080 | 5 | 3 | 96 | 0.048 | 0.102 | -0.046 | 0.143 | 0.063 | 0.125 | no sig. difference |
| GRPO_K | GRPO_LA5 | GRPO_LA0 | gpt-4o-mini | claude-haiku-4-5 | True | 27.080 | 4 | 8 | 96 | 0.166 | 0.266 | 0.041 | 0.291 | 0.012 | 0.035 | arm_a > arm_b |
| GRPO_K | GRPO_LA5 | GRPO_LA0 | gpt-4o-mini | gpt-4o-mini | False | 27.080 | 4 | 8 | 96 | 0.038 | 0.074 | -0.053 | 0.137 | 0.789 | 0.814 | no sig. difference |
| PTO_K | PTO_LA5 | PTO_LA0 | claude-haiku-4-5 | claude-haiku-4-5 | False | 19.680 | 7 | 9 | 96 | -0.186 | -0.323 | -0.301 | -0.072 | 0.005 | 0.011 | arm_a < arm_b |
| PTO_K | PTO_LA5 | PTO_LA0 | claude-haiku-4-5 | gpt-4o-mini | True | 19.680 | 7 | 9 | 96 | -0.153 | -0.267 | -0.264 | -0.046 | 0.017 | 0.023 | arm_a < arm_b |
| PTO_K | PTO_LA5 | PTO_LA0 | gpt-4o-mini | claude-haiku-4-5 | True | 19.680 | 10 | 10 | 96 | -0.199 | -0.308 | -0.332 | -0.068 | 0.032 | 0.065 | no sig. difference |
| PTO_K | PTO_LA5 | PTO_LA0 | gpt-4o-mini | gpt-4o-mini | False | 19.680 | 10 | 10 | 96 | 0.047 | 0.096 | -0.054 | 0.142 | 0.087 | 0.190 | no sig. difference |
| method_K0 | PTO_LA0 | GRPO_LA0 | claude-haiku-4-5 | claude-haiku-4-5 | False | 8.120 | 9 | 2 | 96 | 0.814 | 1.394 | 0.697 | 0.929 | 0.000 | 0.000 | arm_a > arm_b |
| method_K0 | PTO_LA0 | GRPO_LA0 | claude-haiku-4-5 | gpt-4o-mini | True | 8.120 | 9 | 2 | 96 | 0.879 | 1.068 | 0.718 | 1.050 | 0.000 | 0.000 | arm_a > arm_b |
| method_K0 | PTO_LA0 | GRPO_LA0 | gpt-4o-mini | claude-haiku-4-5 | True | 8.120 | 10 | 2 | 96 | 0.759 | 1.341 | 0.649 | 0.878 | 0.000 | 0.000 | arm_a > arm_b |
| method_K0 | PTO_LA0 | GRPO_LA0 | gpt-4o-mini | gpt-4o-mini | False | 8.120 | 10 | 2 | 96 | 0.900 | 1.086 | 0.744 | 1.074 | 0.000 | 0.000 | arm_a > arm_b |
| method_K5 | PTO_LA5 | GRPO_LA5 | claude-haiku-4-5 | claude-haiku-4-5 | False | 19.680 | 7 | 3 | 96 | 0.149 | 0.295 | 0.051 | 0.248 | 0.007 | 0.007 | arm_a > arm_b |
| method_K5 | PTO_LA5 | GRPO_LA5 | claude-haiku-4-5 | gpt-4o-mini | True | 19.680 | 7 | 3 | 96 | 0.224 | 0.407 | 0.120 | 0.332 | 0.000 | 0.000 | arm_a > arm_b |
| method_K5 | PTO_LA5 | GRPO_LA5 | gpt-4o-mini | claude-haiku-4-5 | True | 19.680 | 10 | 3 | 96 | 0.081 | 0.132 | -0.040 | 0.199 | 0.075 | 0.075 | no sig. difference |
| method_K5 | PTO_LA5 | GRPO_LA5 | gpt-4o-mini | gpt-4o-mini | False | 19.680 | 10 | 3 | 96 | 0.445 | 0.673 | 0.313 | 0.583 | 0.000 | 0.000 | arm_a > arm_b |
