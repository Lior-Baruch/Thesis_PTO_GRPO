Across-iteration association between the training-proxy LEVEL and the full-conversation eval LEVEL (rows of reward_faithfulness_levels with a proxy), per arm and eval-side grader: Spearman rho and Pearson r over n_iters model states (train_iter 1..N), mean_gap = mean(proxy - eval), and the range each level spans over training. Descriptive: n <= 10 points per arm, no multiplicity correction. Proxy = training oracle by construction. GRPO_LA5 is right-censored at iteration 5 (train_iter 1..5, eval_iter 0..4).

| arm | eval_grader | n_iters | spearman_rho | spearman_p | pearson_r | pearson_p | mean_gap | proxy_range | eval_range |
|---|---|---|---|---|---|---|---|---|---|
| PTO_LA0 | gpt-4o-mini | 10 | 0.988 | 0.000 | 0.968 | 0.000 | 0.014 | 0.751 | 1.238 |
| PTO_LA0 | claude-haiku-4-5 | 10 | 0.988 | 0.000 | 0.968 | 0.000 | 1.295 | 0.751 | 1.091 |
| PTO_LA5 | gpt-4o-mini | 10 | 0.976 | 0.000 | 0.973 | 0.000 | 0.283 | 0.935 | 1.194 |
| PTO_LA5 | claude-haiku-4-5 | 10 | 0.952 | 0.000 | 0.972 | 0.000 | 1.615 | 0.935 | 0.902 |
| GRPO_LA0 | gpt-4o-mini | 10 | 0.612 | 0.060 | 0.915 | 0.000 | -0.008 | 0.884 | 1.016 |
| GRPO_LA0 | claude-haiku-4-5 | 10 | 0.321 | 0.365 | 0.620 | 0.056 | 1.424 | 0.884 | 0.776 |
| GRPO_LA5 | gpt-4o-mini | 5 | 1.000 | 0.000 | 0.981 | 0.003 | 0.099 | 1.053 | 1.157 |
| GRPO_LA5 | claude-haiku-4-5 | 5 | 1.000 | 0.000 | 0.960 | 0.009 | 1.351 | 1.053 | 0.949 |
