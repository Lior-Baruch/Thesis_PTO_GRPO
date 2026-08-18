Wilcoxon signed-rank test over n_turns BINS (paired by bin; the per-bin deltas of reward_faithfulness_matched_policy) of delta = agreement(K0) - agreement(K5). K-contrast sign: delta = K0 - K5 (+ => K=0 higher; a NEGATIVE delta means look-ahead is more faithful). Cuts: cut=train_iter_1 = MATCHED POLICY — both K arms of a method branch from the SAME base policy pi_0 (eval side = that arm's independent base draw, model_iter_0), so K=0 vs K=5 is free of policy divergence; cut=iters_1-5 pools train_iter 1..5 (GRPO_LA5's full support; policies have diverged); cut=matched_iters pools every train_iter present in BOTH K arms (PTO 1..10, GRPO 1..5). Bins are NOT independent observations (the same conversations feed neighbouring bins), so read p as descriptive; the per-bin CIs and the summary table carry the inference. n_bins = bins with >= 20 pairs in both arms.

| judge | method | cut | n_bins | bins_K5_more_faithful | bins_K0_more_faithful | mean_delta | median_delta | wilcoxon_W | wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | PTO | train_iter_1 | 19 | 10 | 9 | -0.002 | -0.001 | 86.000 | 0.738 |
| gpt-4o-mini | PTO | iters_1-5 | 19 | 19 | 0 | -0.035 | -0.028 | 0.000 | 0.000 |
| gpt-4o-mini | PTO | matched_iters | 19 | 19 | 0 | -0.046 | -0.046 | 0.000 | 0.000 |
| gpt-4o-mini | GRPO | train_iter_1 | 20 | 3 | 17 | 0.017 | 0.014 | 14.000 | 0.000 |
| gpt-4o-mini | GRPO | iters_1-5 | 20 | 3 | 17 | 0.011 | 0.010 | 6.000 | 0.000 |
| gpt-4o-mini | GRPO | matched_iters | 20 | 3 | 17 | 0.011 | 0.010 | 6.000 | 0.000 |
| claude-haiku-4-5 | PTO | train_iter_1 | 19 | 13 | 6 | -0.010 | -0.009 | 53.000 | 0.096 |
| claude-haiku-4-5 | PTO | iters_1-5 | 19 | 15 | 4 | -0.033 | -0.025 | 26.000 | 0.004 |
| claude-haiku-4-5 | PTO | matched_iters | 19 | 18 | 1 | -0.038 | -0.044 | 1.000 | 0.000 |
| claude-haiku-4-5 | GRPO | train_iter_1 | 20 | 17 | 3 | -0.039 | -0.042 | 8.000 | 0.000 |
| claude-haiku-4-5 | GRPO | iters_1-5 | 20 | 20 | 0 | -0.036 | -0.027 | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | matched_iters | 20 | 20 | 0 | -0.036 | -0.027 | 0.000 | 0.000 |
