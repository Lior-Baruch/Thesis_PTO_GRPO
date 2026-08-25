**The ICLR 'K=5 gives shorter conversations' claim at the endpoints: persona-paired K contrast on session length at PTO iteration 10 and GRPO iteration 5 (the last matched GRPO iteration; GRPO_LA5 is right-censored there).** K5_minus_K0 is the K=5 arm's mean minus the K=0 arm's mean (positive = K=5 LONGER); mean_delta_K0_minus_K5 keeps the paper's convention (Sign: + => K=0 higher (K0 - K5).). dz / bootstrap 95% CI / Wilcoxon p on the paired deltas; p_holm within (method, metric) across iterations. Pairing unit: persona_id (the per-iteration file shuffle replayed; never file_index). Judge-free.

| contrast | method | iteration | metric | mean_K0 | mean_K5 | K5_minus_K0 | mean_delta_K0_minus_K5 | dz | ci_lo | ci_hi | p | p_holm | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PTO iter 10 | PTO | 10 | conv_len | 20.385 | 28.698 | 8.312 | -8.312 | -0.548 | -11.219 | -5.260 | 0.000 | 0.000 | 96 |
| PTO iter 10 | PTO | 10 | n_th_turns | 10.229 | 14.385 | 4.156 | -4.156 | -0.548 | -5.635 | -2.656 | 0.000 | 0.000 | 96 |
| PTO iter 10 | PTO | 10 | mean_turn_len | 686.202 | 810.875 | 124.673 | -124.673 | -0.555 | -169.253 | -79.701 | 0.000 | 0.000 | 96 |
| GRPO iter 5 | GRPO | 5 | conv_len | 30.677 | 22.573 | -8.104 | 8.104 | 0.531 | 5.052 | 11.011 | 0.000 | 0.000 | 96 |
| GRPO iter 5 | GRPO | 5 | n_th_turns | 15.344 | 11.312 | -4.031 | 4.031 | 0.528 | 2.510 | 5.479 | 0.000 | 0.000 | 96 |
| GRPO iter 5 | GRPO | 5 | mean_turn_len | 461.787 | 668.343 | 206.556 | -206.556 | -0.815 | -257.754 | -155.117 | 0.000 | 0.000 | 96 |
