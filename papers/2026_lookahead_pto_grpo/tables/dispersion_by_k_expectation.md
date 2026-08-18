Simulated reference for the best-worst margin over within-group SD when the M=8 candidate scores are iid normal (pure sampling spread, no true separation between candidates): 200,000 simulated groups; ``ratio_of_means`` = E[range]/E[SD] (the estimator used for ``margin_over_sd`` in the by-iter table), ``mean_of_ratios`` = E[range/SD]. ``ddof=0`` is the population SD GRPO records as ``group_std`` and divides its advantages by; ``ddof=1`` is the sample SD; ``winner_z_mean`` = E[(max − mean)/SD], the standardized lead of the best candidate over its group. Neither depends on the grader (pure geometry of 8 draws).

| sd_estimator | n_groups | m | E_range_over_sigma | E_sd_over_sigma | ratio_of_means | mean_of_ratios | median_of_ratios | winner_z_mean |
|---|---|---|---|---|---|---|---|---|
| ddof=0 | 200000 | 8 | 2.8465 | 0.9027 | 3.1533 | 3.1533 | 3.1554 | 1.5772 |
| ddof=1 | 200000 | 8 | 2.8465 | 0.9650 | 2.9496 | 2.9496 | 2.9516 | 1.4754 |
