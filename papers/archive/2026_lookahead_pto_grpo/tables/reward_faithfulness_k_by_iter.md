K=0 vs K=5 faithfulness per training iteration (all bins pooled), per method and eval-side grader. K-contrast sign: delta = K0 - K5 (+ => K=0 higher; a NEGATIVE delta means look-ahead is more faithful). CI = percentile of the difference of independent cluster-bootstrap replicates (the two arms are different conversation draws). Only train_iter 1 samples the SAME policy in both K arms; later rows compare diverged policies. Unit: one branch row = one training branch point (prefix of n_turns utterances, therapist+patient, ending on a patient turn) with its 8 completions sampled by that iteration's policy (PTO: the frozen iter-start policy; GRPO: the policy as it trains within the iteration over 2 epochs, plus the ~3-10% eval-split groups TRL scores at iteration end — all rows kept); proxy_score = training-oracle (gpt-4o-mini, Q1+Q2 mean) score of the CHOSEN (arg-max) completion on prefix+completion (K=0) or prefix+completion+5 simulated turns (K=5, i.e. the K-extended score). eval_score = full-conversation Q1Q2 of the eval conversation the prefix was cut from (model_iter_{train_iter-1}, same file_index / persona; GRPO prefixes are exact slices of it, PTO greedy trunks share its first MCL=12 utterances then diverge). agreement = fraction of conversation pairs within one (arm, eval_iter, n_turns) cell whose proxy ordering matches their eval ordering (ties dropped), counts pooled over eval_iters; 0.5 = chance. 95% CI = cluster bootstrap over conversations within each (arm, eval_iter) model state (B=1000). n_turns = utterances BEFORE the scored completion (MCL=12 = shortest cut). GRPO_LA5 is right-censored at iteration 5 (train_iter 1..5, eval_iter 0..4).

| judge | method | train_iter | eval_iter | agr_K0 | agr_K5 | delta_K0_minus_K5 [CI] | pairs_K0 | pairs_K5 |
|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | PTO | 1 | 0 | 0.809 | 0.805 | 0.004 [-0.067, 0.074] | 26808 | 25256 |
| gpt-4o-mini | PTO | 2 | 1 | 0.809 | 0.853 | -0.043 [-0.114, 0.026] | 19334 | 21883 |
| gpt-4o-mini | PTO | 3 | 2 | 0.808 | 0.821 | -0.014 [-0.076, 0.050] | 20935 | 22832 |
| gpt-4o-mini | PTO | 4 | 3 | 0.821 | 0.866 | -0.044 [-0.100, 0.009] | 14589 | 15443 |
| gpt-4o-mini | PTO | 5 | 4 | 0.839 | 0.892 | -0.053 [-0.109, -0.006] | 14814 | 13589 |
| gpt-4o-mini | PTO | 6 | 5 | 0.868 | 0.888 | -0.020 [-0.073, 0.026] | 13882 | 16236 |
| gpt-4o-mini | PTO | 7 | 6 | 0.877 | 0.867 | 0.011 [-0.035, 0.055] | 11541 | 20981 |
| gpt-4o-mini | PTO | 8 | 7 | 0.869 | 0.899 | -0.031 [-0.084, 0.022] | 11120 | 17762 |
| gpt-4o-mini | PTO | 9 | 8 | 0.868 | 0.879 | -0.011 [-0.076, 0.048] | 9637 | 18654 |
| gpt-4o-mini | PTO | 10 | 9 | 0.877 | 0.895 | -0.018 [-0.076, 0.041] | 8555 | 23773 |
| gpt-4o-mini | GRPO | 1 | 0 | 0.895 | 0.880 | 0.015 [-0.023, 0.057] | 24704 | 24149 |
| gpt-4o-mini | GRPO | 2 | 1 | 0.887 | 0.899 | -0.012 [-0.049, 0.027] | 19962 | 23451 |
| gpt-4o-mini | GRPO | 3 | 2 | 0.900 | 0.864 | 0.035 [-0.010, 0.085] | 30318 | 27482 |
| gpt-4o-mini | GRPO | 4 | 3 | 0.902 | 0.900 | 0.002 [-0.030, 0.031] | 24907 | 27279 |
| gpt-4o-mini | GRPO | 5 | 4 | 0.898 | 0.909 | -0.011 [-0.045, 0.021] | 27758 | 21934 |
| claude-haiku-4-5 | PTO | 1 | 0 | 0.745 | 0.738 | 0.007 [-0.074, 0.092] | 25814 | 24963 |
| claude-haiku-4-5 | PTO | 2 | 1 | 0.763 | 0.812 | -0.050 [-0.128, 0.030] | 19275 | 21520 |
| claude-haiku-4-5 | PTO | 3 | 2 | 0.799 | 0.750 | 0.049 [-0.010, 0.112] | 20990 | 22742 |
| claude-haiku-4-5 | PTO | 4 | 3 | 0.787 | 0.791 | -0.005 [-0.080, 0.069] | 14738 | 15545 |
| claude-haiku-4-5 | PTO | 5 | 4 | 0.774 | 0.809 | -0.035 [-0.099, 0.026] | 14836 | 13667 |
| claude-haiku-4-5 | PTO | 6 | 5 | 0.796 | 0.854 | -0.058 [-0.115, -0.002] | 13991 | 16178 |
| claude-haiku-4-5 | PTO | 7 | 6 | 0.823 | 0.808 | 0.014 [-0.042, 0.073] | 11695 | 21125 |
| claude-haiku-4-5 | PTO | 8 | 7 | 0.803 | 0.829 | -0.026 [-0.096, 0.040] | 11309 | 17949 |
| claude-haiku-4-5 | PTO | 9 | 8 | 0.775 | 0.807 | -0.032 [-0.110, 0.050] | 9768 | 18775 |
| claude-haiku-4-5 | PTO | 10 | 9 | 0.816 | 0.783 | 0.033 [-0.035, 0.111] | 8623 | 24300 |
| claude-haiku-4-5 | GRPO | 1 | 0 | 0.798 | 0.813 | -0.014 [-0.077, 0.050] | 24491 | 23932 |
| claude-haiku-4-5 | GRPO | 2 | 1 | 0.838 | 0.807 | 0.031 [-0.025, 0.095] | 19846 | 23313 |
| claude-haiku-4-5 | GRPO | 3 | 2 | 0.826 | 0.795 | 0.031 [-0.021, 0.085] | 30438 | 27453 |
| claude-haiku-4-5 | GRPO | 4 | 3 | 0.806 | 0.831 | -0.024 [-0.086, 0.035] | 25121 | 27426 |
| claude-haiku-4-5 | GRPO | 5 | 4 | 0.712 | 0.806 | -0.094 [-0.156, -0.018] | 28280 | 22175 |
