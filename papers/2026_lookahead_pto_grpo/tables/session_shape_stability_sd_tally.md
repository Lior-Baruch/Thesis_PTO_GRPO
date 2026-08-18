**Tally of the K0-vs-K5 dispersion contrast over the trained matched iterations (1..N), per grader x method x rubric.** n_K5_lower_sd / n_K5_lower_iqr = iterations at which the K=5 arm's SD / IQR is smaller than K=0's; median_sd_ratio = median of sd_K5 / sd_K0 (< 1 => K=5 typically less dispersed); n_pm_holm_sig_K5_lower / _K0_lower = iterations where the persona-paired Pitman-Morgan test is Holm-significant (within judge x method x rubric across iterations) with K=5 resp. K=0 less dispersed; n_bf_holm_sig = Brown-Forsythe Holm-significant iterations (either direction). iter0_sd_* = the two independent base draws (noise floor for an SD difference). GRPO_LA5 is right-censored at iteration 5 (its K=0 sibling runs to 10). PTO: N=10 iterations; GRPO: N=5.

| judge | method | metric | n_iters | n_K5_lower_sd | n_K5_lower_iqr | median_sd_ratio_K5_over_K0 | n_pm_holm_sig_K5_lower | n_pm_holm_sig_K0_lower | n_bf_holm_sig | iter0_sd_K0 | iter0_sd_K5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | PTO | Q1 | 10 | 1 | 4 | 1.061 | 0 | 1 | 0 | 1.323 | 1.256 |
| gpt-4o-mini | GRPO | Q1 | 5 | 2 | 2 | 1.017 | 0 | 0 | 0 | 1.296 | 1.336 |
| gpt-4o-mini | PTO | Q2 | 10 | 0 | 4 | 1.331 | 0 | 6 | 1 | 1.178 | 1.163 |
| gpt-4o-mini | GRPO | Q2 | 5 | 2 | 3 | 1.015 | 0 | 1 | 0 | 1.157 | 1.183 |
| gpt-4o-mini | PTO | Q1Q2 | 10 | 0 | 4 | 1.174 | 0 | 4 | 0 | 1.234 | 1.191 |
| gpt-4o-mini | GRPO | Q1Q2 | 5 | 2 | 4 | 1.035 | 0 | 0 | 0 | 1.212 | 1.245 |
| claude-haiku-4-5 | PTO | Q1 | 10 | 3 | 4 | 1.029 | 0 | 0 | 0 | 0.796 | 0.752 |
| claude-haiku-4-5 | GRPO | Q1 | 5 | 3 | 3 | 0.974 | 0 | 0 | 0 | 0.741 | 0.763 |
| claude-haiku-4-5 | PTO | Q2 | 10 | 6 | 5 | 0.993 | 0 | 0 | 0 | 0.657 | 0.624 |
| claude-haiku-4-5 | GRPO | Q2 | 5 | 2 | 2 | 1.001 | 1 | 0 | 1 | 0.609 | 0.659 |
| claude-haiku-4-5 | PTO | Q1Q2 | 10 | 4 | 6 | 1.008 | 0 | 0 | 0 | 0.691 | 0.656 |
| claude-haiku-4-5 | GRPO | Q1Q2 | 5 | 2 | 2 | 1.002 | 1 | 0 | 1 | 0.648 | 0.682 |
