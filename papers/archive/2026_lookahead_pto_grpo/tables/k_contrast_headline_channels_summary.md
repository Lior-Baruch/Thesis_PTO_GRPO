**Per (grader, method, channel) summary of the behaviour-channel K contrast.** Same columns as the rubric summary; MICI channels and their counts are lower-better. Sign: + = K=0 higher (K=0 minus K=5). Paired on persona_id (96 personas). p_holm = Holm across iterations 0..N within each (judge, method, metric); iteration 0 = two independent base draws (noise floor). GRPO_LA5 is right-censored at iteration 5, so GRPO rows stop at 5.

| judge | method | metric | n_iters | n_sig_K0_higher | n_sig_K5_higher | n_sig_K0_better | n_sig_K5_better | iters_sig_K0_higher | iters_sig_K5_higher | mean_delta_iters1toN | mean_dz_iters1toN | base_delta | base_dz | max_abs_dz | max_abs_dz_iter | max_abs_dz_delta | lower_better |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | GRPO | %CR | 6 | 0 | 0 | 0 | 0 |  |  | -0.075 | -0.153 | 0.064 | 0.128 | 0.270 | 5 | -0.137 | False |
| claude-haiku-4-5 | GRPO | %MICO | 6 | 2 | 0 | 2 | 0 | 4,5 |  | 0.069 | 0.220 | -0.022 | -0.053 | 0.332 | 4 | 0.103 | False |
| claude-haiku-4-5 | GRPO | B1_GI | 6 | 2 | 0 | 2 | 0 | 4,5 |  | 0.756 | 0.191 | 0.031 | 0.010 | 0.656 | 5 | 2.552 | False |
| claude-haiku-4-5 | GRPO | B1_GI_per_turn | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.000 | 0.011 | -0.017 | -0.074 | 0.301 | 5 | 0.054 | False |
| claude-haiku-4-5 | GRPO | B2_Persuade | 6 | 0 | 1 | 0 | 1 |  | 3 | -0.215 | -0.075 | 0.010 | 0.003 | 0.337 | 3 | -0.812 | False |
| claude-haiku-4-5 | GRPO | B2_Persuade_per_turn | 6 | 0 | 4 | 0 | 4 |  | 2,3,4,5 | -0.036 | -0.271 | -0.007 | -0.028 | 0.572 | 5 | -0.059 | False |
| claude-haiku-4-5 | GRPO | B3_Q | 6 | 0 | 0 | 0 | 0 |  |  | -0.106 | -0.016 | 1.156 | 0.134 | 0.160 | 1 | -1.333 | False |
| claude-haiku-4-5 | GRPO | B3_Q_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | -0.015 | -0.056 | 0.039 | 0.109 | 0.206 | 4 | -0.048 | False |
| claude-haiku-4-5 | GRPO | B4_SR | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.135 | 0.141 | -0.479 | -0.173 | 0.282 | 5 | 0.208 | False |
| claude-haiku-4-5 | GRPO | B4_SR_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | 0.006 | 0.093 | -0.017 | -0.123 | 0.226 | 3 | 0.014 | False |
| claude-haiku-4-5 | GRPO | B5_CR | 6 | 0 | 0 | 0 | 0 |  |  | -0.008 | -0.010 | 0.042 | 0.061 | 0.110 | 2 | -0.115 | False |
| claude-haiku-4-5 | GRPO | B5_CR_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.043 | 0.001 | 0.025 | 0.142 | 5 | -0.010 | False |
| claude-haiku-4-5 | GRPO | B6_AF | 6 | 2 | 0 | 2 | 0 | 4,5 |  | 0.323 | 0.301 | 0.042 | 0.041 | 0.635 | 5 | 0.594 | False |
| claude-haiku-4-5 | GRPO | B6_AF_per_turn | 6 | 3 | 0 | 3 | 0 | 3,4,5 |  | 0.020 | 0.273 | 0.002 | 0.033 | 0.452 | 5 | 0.028 | False |
| claude-haiku-4-5 | GRPO | B7_Seek | 6 | 0 | 0 | 0 | 0 |  |  | 0.171 | 0.082 | -0.635 | -0.188 | 0.230 | 4 | 0.250 | False |
| claude-haiku-4-5 | GRPO | B7_Seek_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | 0.006 | 0.065 | -0.015 | -0.079 | 0.202 | 4 | 0.010 | False |
| claude-haiku-4-5 | GRPO | MICI_AdviseNoPermission | 6 | 0 | 1 | 1 | 0 |  | 3 | 0.088 | 0.021 | 0.406 | 0.095 | 0.310 | 3 | -0.979 | True |
| claude-haiku-4-5 | GRPO | MICI_AdviseNoPermission_rate | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.029 | -0.126 | 0.004 | 0.014 | 0.278 | 3 | -0.063 | True |
| claude-haiku-4-5 | GRPO | MICI_BehaviorTotal | 6 | 1 | 1 | 1 | 1 | 5 | 3 | -0.019 | -0.017 | 1.531 | 0.237 | 0.425 | 3 | -1.802 | True |
| claude-haiku-4-5 | GRPO | MICI_Confront | 6 | 0 | 2 | 2 | 0 |  | 3,5 | -0.100 | -0.212 | -0.062 | -0.067 | 0.400 | 3 | -0.219 | True |
| claude-haiku-4-5 | GRPO | MICI_Confront_rate | 6 | 0 | 2 | 2 | 0 |  | 3,5 | -0.009 | -0.219 | -0.005 | -0.071 | 0.399 | 3 | -0.014 | True |
| claude-haiku-4-5 | GRPO | MICI_Direct | 6 | 0 | 2 | 2 | 0 |  | 3,4 | -0.383 | -0.213 | 1.042 | 0.262 | 0.424 | 4 | -0.646 | True |
| claude-haiku-4-5 | GRPO | MICI_Direct_rate | 6 | 0 | 3 | 3 | 0 |  | 3,4,5 | -0.033 | -0.254 | 0.043 | 0.206 | 0.420 | 4 | -0.052 | True |
| claude-haiku-4-5 | GRPO | MICI_Judge | 6 | 0 | 0 | 0 | 0 |  |  | -0.027 | -0.115 | 0.042 | 0.077 | 0.196 | 3 | -0.052 | True |
| claude-haiku-4-5 | GRPO | MICI_Judge_rate | 6 | 0 | 0 | 0 | 0 |  |  | -0.002 | -0.112 | 0.004 | 0.090 | 0.207 | 3 | -0.003 | True |
| claude-haiku-4-5 | GRPO | MICI_OverPraise | 6 | 2 | 0 | 0 | 2 | 4,5 |  | 0.440 | 0.277 | 0.083 | 0.121 | 0.690 | 5 | 1.250 | True |
| claude-haiku-4-5 | GRPO | MICI_OverPraise_rate | 6 | 2 | 0 | 0 | 2 | 4,5 |  | 0.029 | 0.291 | 0.009 | 0.131 | 0.687 | 5 | 0.084 | True |
| claude-haiku-4-5 | GRPO | MICI_Rate | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.046 | -0.142 | 0.057 | 0.147 | 0.354 | 3 | -0.111 | True |
| claude-haiku-4-5 | GRPO | MICI_Severity | 6 | 1 | 1 | 1 | 1 | 5 | 3 | 0.021 | 0.040 | -0.042 | -0.038 | 0.429 | 5 | 0.333 | True |
| claude-haiku-4-5 | GRPO | MICI_Warn | 6 | 0 | 0 | 0 | 0 |  |  | -0.035 | -0.099 | 0.021 | 0.024 | 0.175 | 2 | -0.073 | True |
| claude-haiku-4-5 | GRPO | MICI_Warn_rate | 6 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.110 | 0.003 | 0.033 | 0.214 | 2 | -0.007 | True |
| claude-haiku-4-5 | GRPO | RtoQ | 6 | 0 | 0 | 0 | 0 |  |  | 0.010 | 0.019 | -0.333 | -0.207 | 0.207 | 0 | -0.333 | False |
| claude-haiku-4-5 | PTO | %CR | 11 | 0 | 0 | 0 | 0 |  |  | -0.045 | -0.100 | -0.063 | -0.187 | 0.271 | 10 | -0.128 | False |
| claude-haiku-4-5 | PTO | %MICO | 11 | 6 | 0 | 6 | 0 | 5,6,7,8,9,10 |  | 0.112 | 0.373 | -0.076 | -0.170 | 0.707 | 6 | 0.170 | False |
| claude-haiku-4-5 | PTO | B1_GI | 11 | 0 | 2 | 0 | 2 |  | 9,10 | -0.460 | -0.160 | 0.521 | 0.128 | 0.527 | 9 | -1.417 | False |
| claude-haiku-4-5 | PTO | B1_GI_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | -0.013 | -0.051 | 0.021 | 0.064 | 0.131 | 4 | -0.038 | False |
| claude-haiku-4-5 | PTO | B2_Persuade | 11 | 0 | 6 | 0 | 6 |  | 5,6,7,8,9,10 | -1.137 | -0.430 | -0.031 | -0.008 | 0.999 | 9 | -2.469 | False |
| claude-haiku-4-5 | PTO | B2_Persuade_per_turn | 11 | 0 | 7 | 0 | 7 |  | 4,5,6,7,8,9,10 | -0.061 | -0.401 | -0.007 | -0.030 | 0.731 | 8 | -0.097 | False |
| claude-haiku-4-5 | PTO | B3_Q | 11 | 0 | 1 | 0 | 1 |  | 10 | -0.060 | -0.054 | -0.729 | -0.076 | 0.317 | 10 | -0.740 | False |
| claude-haiku-4-5 | PTO | B3_Q_per_turn | 11 | 2 | 0 | 2 | 0 | 5,9 |  | 0.028 | 0.086 | 0.010 | 0.022 | 0.191 | 9 | 0.056 | False |
| claude-haiku-4-5 | PTO | B4_SR | 11 | 1 | 0 | 1 | 0 | 5 |  | 0.100 | 0.112 | -0.260 | -0.166 | 0.338 | 5 | 0.271 | False |
| claude-haiku-4-5 | PTO | B4_SR_per_turn | 11 | 3 | 0 | 3 | 0 | 5,6,8 |  | 0.014 | 0.206 | -0.011 | -0.131 | 0.403 | 6 | 0.023 | False |
| claude-haiku-4-5 | PTO | B5_CR | 11 | 0 | 1 | 0 | 1 |  | 9 | -0.066 | -0.012 | -0.115 | -0.165 | 0.369 | 9 | -0.583 | False |
| claude-haiku-4-5 | PTO | B5_CR_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | 0.004 | 0.058 | -0.005 | -0.140 | 0.259 | 6 | 0.017 | False |
| claude-haiku-4-5 | PTO | B6_AF | 11 | 5 | 0 | 5 | 0 | 5,6,8,9,10 |  | 0.317 | 0.225 | -0.177 | -0.155 | 0.551 | 8 | 0.542 | False |
| claude-haiku-4-5 | PTO | B6_AF_per_turn | 11 | 6 | 0 | 6 | 0 | 5,6,7,8,9,10 |  | 0.032 | 0.315 | -0.011 | -0.173 | 0.694 | 10 | 0.101 | False |
| claude-haiku-4-5 | PTO | B7_Seek | 11 | 0 | 1 | 0 | 1 |  | 10 | -0.030 | -0.097 | -0.188 | -0.046 | 0.307 | 10 | -0.104 | False |
| claude-haiku-4-5 | PTO | B7_Seek_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.097 | 0.004 | 0.017 | 0.307 | 10 | -0.005 | False |
| claude-haiku-4-5 | PTO | MICI_AdviseNoPermission | 11 | 0 | 6 | 6 | 0 |  | 4,6,7,8,9,10 | -1.133 | -0.357 | 0.844 | 0.162 | 0.777 | 9 | -2.521 | True |
| claude-haiku-4-5 | PTO | MICI_AdviseNoPermission_rate | 11 | 0 | 3 | 3 | 0 |  | 4,8,9 | -0.059 | -0.218 | 0.029 | 0.095 | 0.362 | 4 | -0.103 | True |
| claude-haiku-4-5 | PTO | MICI_BehaviorTotal | 11 | 0 | 1 | 1 | 0 |  | 4 | -0.799 | -0.159 | 0.188 | 0.025 | 0.435 | 4 | -1.844 | True |
| claude-haiku-4-5 | PTO | MICI_Confront | 11 | 0 | 4 | 4 | 0 |  | 6,8,9,10 | -0.126 | -0.234 | -0.010 | -0.012 | 0.345 | 8 | -0.125 | True |
| claude-haiku-4-5 | PTO | MICI_Confront_rate | 11 | 0 | 3 | 3 | 0 |  | 6,9,10 | -0.009 | -0.231 | -0.001 | -0.014 | 0.385 | 10 | -0.019 | True |
| claude-haiku-4-5 | PTO | MICI_Direct | 11 | 0 | 6 | 6 | 0 |  | 4,5,6,8,9,10 | -0.552 | -0.305 | -0.198 | -0.060 | 0.512 | 9 | -0.760 | True |
| claude-haiku-4-5 | PTO | MICI_Direct_rate | 11 | 0 | 6 | 6 | 0 |  | 4,5,6,8,9,10 | -0.043 | -0.308 | -0.009 | -0.041 | 0.454 | 9 | -0.050 | True |
| claude-haiku-4-5 | PTO | MICI_Judge | 11 | 0 | 0 | 0 | 0 |  |  | -0.049 | -0.144 | -0.073 | -0.083 | 0.263 | 6 | -0.135 | True |
| claude-haiku-4-5 | PTO | MICI_Judge_rate | 11 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.142 | -0.006 | -0.095 | 0.289 | 6 | -0.010 | True |
| claude-haiku-4-5 | PTO | MICI_OverPraise | 11 | 5 | 0 | 0 | 5 | 6,7,8,9,10 |  | 1.100 | 0.405 | -0.365 | -0.145 | 0.999 | 10 | 3.573 | True |
| claude-haiku-4-5 | PTO | MICI_OverPraise_rate | 11 | 5 | 0 | 0 | 5 | 6,7,8,9,10 |  | 0.114 | 0.583 | -0.019 | -0.173 | 1.648 | 10 | 0.373 | True |
| claude-haiku-4-5 | PTO | MICI_Rate | 11 | 2 | 1 | 1 | 2 | 9,10 | 4 | -0.004 | 0.002 | -0.006 | -0.014 | 0.655 | 10 | 0.245 | True |
| claude-haiku-4-5 | PTO | MICI_Severity | 11 | 0 | 2 | 2 | 0 |  | 4,6 | -0.124 | -0.127 | 0.104 | 0.073 | 0.418 | 4 | -0.375 | True |
| claude-haiku-4-5 | PTO | MICI_Warn | 11 | 0 | 0 | 0 | 0 |  |  | -0.039 | -0.100 | -0.010 | -0.028 | 0.233 | 4 | -0.104 | True |
| claude-haiku-4-5 | PTO | MICI_Warn_rate | 11 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.104 | -0.001 | -0.031 | 0.230 | 5 | -0.014 | True |
| claude-haiku-4-5 | PTO | RtoQ | 11 | 0 | 0 | 0 | 0 |  |  | 0.044 | 0.044 | -0.147 | -0.266 | 0.266 | 0 | -0.147 | False |
| gpt-4o-mini | GRPO | %CR | 6 | 0 | 0 | 0 | 0 |  |  | 0.008 | 0.033 | -0.002 | -0.009 | 0.098 | 5 | 0.025 | False |
| gpt-4o-mini | GRPO | %MICO | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.051 | 0.188 | -0.034 | -0.120 | 0.504 | 5 | 0.133 | False |
| gpt-4o-mini | GRPO | B1_GI | 6 | 2 | 0 | 2 | 0 | 4,5 |  | 0.321 | 0.207 | 0.229 | 0.110 | 0.598 | 5 | 0.906 | False |
| gpt-4o-mini | GRPO | B1_GI_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | 0.004 | 0.041 | 0.005 | 0.038 | 0.140 | 4 | 0.013 | False |
| gpt-4o-mini | GRPO | B2_Persuade | 6 | 0 | 1 | 0 | 1 |  | 3 | -0.125 | -0.084 | 0.125 | 0.086 | 0.376 | 3 | -0.479 | False |
| gpt-4o-mini | GRPO | B2_Persuade_per_turn | 6 | 0 | 2 | 0 | 2 |  | 3,5 | -0.020 | -0.241 | 0.011 | 0.096 | 0.531 | 5 | -0.040 | False |
| gpt-4o-mini | GRPO | B3_Q | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.379 | 0.128 | -0.167 | -0.026 | 0.421 | 5 | 1.125 | False |
| gpt-4o-mini | GRPO | B3_Q_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | -0.005 | -0.062 | -0.025 | -0.087 | 0.222 | 5 | -0.023 | False |
| gpt-4o-mini | GRPO | B4_SR | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.229 | 0.104 | -0.198 | -0.065 | 0.442 | 5 | 0.927 | False |
| gpt-4o-mini | GRPO | B4_SR_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | 0.009 | 0.078 | -0.020 | -0.137 | 0.180 | 5 | 0.017 | False |
| gpt-4o-mini | GRPO | B5_CR | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.090 | 0.076 | 0.073 | 0.037 | 0.300 | 5 | 0.385 | False |
| gpt-4o-mini | GRPO | B5_CR_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | 0.003 | 0.025 | 0.006 | 0.042 | 0.057 | 2 | 0.007 | False |
| gpt-4o-mini | GRPO | B6_AF | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.129 | 0.086 | 0.104 | 0.086 | 0.389 | 5 | 0.552 | False |
| gpt-4o-mini | GRPO | B6_AF_per_turn | 6 | 1 | 0 | 1 | 0 | 5 |  | 0.008 | 0.079 | 0.003 | 0.043 | 0.335 | 5 | 0.034 | False |
| gpt-4o-mini | GRPO | B7_Seek | 6 | 0 | 0 | 0 | 0 |  |  | 0.125 | 0.058 | 0.094 | 0.042 | 0.216 | 5 | 0.521 | False |
| gpt-4o-mini | GRPO | B7_Seek_per_turn | 6 | 0 | 0 | 0 | 0 |  |  | 0.008 | 0.068 | 0.005 | 0.039 | 0.206 | 5 | 0.027 | False |
| gpt-4o-mini | GRPO | MICI_AdviseNoPermission | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.244 | -0.135 | 0.229 | 0.099 | 0.367 | 3 | -0.656 | True |
| gpt-4o-mini | GRPO | MICI_AdviseNoPermission_rate | 6 | 0 | 3 | 3 | 0 |  | 3,4,5 | -0.041 | -0.251 | 0.010 | 0.051 | 0.379 | 5 | -0.062 | True |
| gpt-4o-mini | GRPO | MICI_BehaviorTotal | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.342 | -0.125 | 0.167 | 0.035 | 0.419 | 3 | -1.094 | True |
| gpt-4o-mini | GRPO | MICI_Confront | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.042 | -0.162 | 0.031 | 0.070 | 0.320 | 3 | -0.094 | True |
| gpt-4o-mini | GRPO | MICI_Confront_rate | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.002 | -0.125 | 0.003 | 0.068 | 0.305 | 3 | -0.006 | True |
| gpt-4o-mini | GRPO | MICI_Direct | 6 | 0 | 2 | 2 | 0 |  | 3,4 | -0.104 | -0.108 | 0.104 | 0.055 | 0.328 | 4 | -0.271 | True |
| gpt-4o-mini | GRPO | MICI_Direct_rate | 6 | 0 | 1 | 1 | 0 |  | 4 | -0.011 | -0.126 | -0.003 | -0.027 | 0.323 | 4 | -0.025 | True |
| gpt-4o-mini | GRPO | MICI_Judge | 6 | 0 | 1 | 1 | 0 |  | 2 | -0.037 | -0.140 | 0.021 | 0.031 | 0.279 | 2 | -0.146 | True |
| gpt-4o-mini | GRPO | MICI_Judge_rate | 6 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.131 | 0.001 | 0.016 | 0.261 | 3 | -0.005 | True |
| gpt-4o-mini | GRPO | MICI_OverPraise | 6 | 1 | 0 | 0 | 1 | 5 |  | 0.106 | 0.095 | -0.208 | -0.077 | 0.378 | 5 | 0.427 | True |
| gpt-4o-mini | GRPO | MICI_OverPraise_rate | 6 | 1 | 0 | 0 | 1 | 5 |  | 0.008 | 0.076 | -0.007 | -0.052 | 0.301 | 5 | 0.030 | True |
| gpt-4o-mini | GRPO | MICI_Rate | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.051 | -0.206 | 0.002 | 0.007 | 0.319 | 3 | -0.077 | True |
| gpt-4o-mini | GRPO | MICI_Severity | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.212 | -0.240 | -0.167 | -0.146 | 0.527 | 3 | -0.458 | True |
| gpt-4o-mini | GRPO | MICI_Warn | 6 | 0 | 0 | 0 | 0 |  |  | -0.021 | -0.148 | -0.010 | -0.045 | 0.179 | 3 | -0.031 | True |
| gpt-4o-mini | GRPO | MICI_Warn_rate | 6 | 0 | 0 | 0 | 0 |  |  | -0.002 | -0.146 | -0.001 | -0.055 | 0.174 | 3 | -0.002 | True |
| gpt-4o-mini | GRPO | RtoQ | 6 | 0 | 0 | 0 | 0 |  |  | 0.005 | 0.015 | -0.062 | -0.079 | 0.119 | 5 | 0.066 | False |
| gpt-4o-mini | PTO | %CR | 11 | 0 | 0 | 0 | 0 |  |  | -0.002 | -0.014 | 0.025 | 0.107 | 0.193 | 2 | -0.046 | False |
| gpt-4o-mini | PTO | %MICO | 11 | 1 | 0 | 1 | 0 | 8 |  | 0.018 | 0.069 | 0.035 | 0.105 | 0.281 | 8 | 0.071 | False |
| gpt-4o-mini | PTO | B1_GI | 11 | 0 | 0 | 0 | 0 |  |  | 0.011 | 0.001 | -0.073 | -0.030 | 0.255 | 9 | -0.312 | False |
| gpt-4o-mini | PTO | B1_GI_per_turn | 11 | 3 | 0 | 3 | 0 | 8,9,10 |  | 0.019 | 0.184 | -0.006 | -0.036 | 0.570 | 10 | 0.060 | False |
| gpt-4o-mini | PTO | B2_Persuade | 11 | 0 | 4 | 0 | 4 |  | 6,8,9,10 | -0.253 | -0.197 | 0.000 | 0.000 | 0.525 | 10 | -0.667 | False |
| gpt-4o-mini | PTO | B2_Persuade_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | -0.006 | -0.069 | -0.003 | -0.024 | 0.249 | 8 | -0.018 | False |
| gpt-4o-mini | PTO | B3_Q | 11 | 0 | 3 | 0 | 3 |  | 8,9,10 | -0.260 | -0.128 | -0.781 | -0.093 | 0.531 | 9 | -1.052 | False |
| gpt-4o-mini | PTO | B3_Q_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | 0.023 | 0.089 | 0.008 | 0.023 | 0.171 | 7 | 0.033 | False |
| gpt-4o-mini | PTO | B4_SR | 11 | 0 | 2 | 0 | 2 |  | 9,10 | -0.308 | -0.144 | -0.062 | -0.023 | 0.549 | 10 | -1.240 | False |
| gpt-4o-mini | PTO | B4_SR_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.021 | -0.000 | -0.001 | 0.261 | 10 | -0.028 | False |
| gpt-4o-mini | PTO | B5_CR | 11 | 0 | 2 | 0 | 2 |  | 9,10 | -0.221 | -0.135 | -0.010 | -0.006 | 0.448 | 10 | -0.781 | False |
| gpt-4o-mini | PTO | B5_CR_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | -0.003 | -0.031 | 0.008 | 0.073 | 0.251 | 7 | -0.023 | False |
| gpt-4o-mini | PTO | B6_AF | 11 | 4 | 0 | 4 | 0 | 7,8,9,10 |  | 0.286 | 0.166 | -0.125 | -0.116 | 0.427 | 8 | 0.635 | False |
| gpt-4o-mini | PTO | B6_AF_per_turn | 11 | 4 | 0 | 4 | 0 | 7,8,9,10 |  | 0.034 | 0.268 | -0.004 | -0.057 | 0.708 | 10 | 0.099 | False |
| gpt-4o-mini | PTO | B7_Seek | 11 | 1 | 0 | 1 | 0 | 8 |  | 0.119 | 0.080 | 0.052 | 0.013 | 0.423 | 8 | 0.552 | False |
| gpt-4o-mini | PTO | B7_Seek_per_turn | 11 | 3 | 0 | 3 | 0 | 8,9,10 |  | 0.015 | 0.150 | 0.005 | 0.027 | 0.561 | 8 | 0.055 | False |
| gpt-4o-mini | PTO | MICI_AdviseNoPermission | 11 | 0 | 3 | 3 | 0 |  | 4,8,9 | -0.299 | -0.211 | 0.250 | 0.100 | 0.498 | 8 | -0.573 | True |
| gpt-4o-mini | PTO | MICI_AdviseNoPermission_rate | 11 | 0 | 1 | 1 | 0 |  | 4 | -0.014 | -0.079 | 0.012 | 0.064 | 0.363 | 4 | -0.067 | True |
| gpt-4o-mini | PTO | MICI_BehaviorTotal | 11 | 3 | 1 | 1 | 3 | 7,9,10 | 4 | 0.099 | 0.029 | 0.562 | 0.139 | 0.446 | 10 | 1.615 | True |
| gpt-4o-mini | PTO | MICI_Confront | 11 | 0 | 0 | 0 | 0 |  |  | -0.028 | -0.128 | 0.021 | 0.051 | 0.257 | 10 | -0.062 | True |
| gpt-4o-mini | PTO | MICI_Confront_rate | 11 | 0 | 0 | 0 | 0 |  |  | -0.002 | -0.116 | 0.003 | 0.089 | 0.252 | 10 | -0.006 | True |
| gpt-4o-mini | PTO | MICI_Direct | 11 | 0 | 4 | 4 | 0 |  | 6,8,9,10 | -0.201 | -0.185 | 0.250 | 0.120 | 0.456 | 8 | -0.448 | True |
| gpt-4o-mini | PTO | MICI_Direct_rate | 11 | 0 | 1 | 1 | 0 |  | 8 | -0.015 | -0.157 | 0.015 | 0.107 | 0.389 | 8 | -0.031 | True |
| gpt-4o-mini | PTO | MICI_Judge | 11 | 0 | 0 | 0 | 0 |  |  | -0.019 | -0.088 | -0.052 | -0.078 | 0.207 | 10 | -0.042 | True |
| gpt-4o-mini | PTO | MICI_Judge_rate | 11 | 0 | 0 | 0 | 0 |  |  | -0.001 | -0.078 | -0.004 | -0.073 | 0.203 | 10 | -0.004 | True |
| gpt-4o-mini | PTO | MICI_OverPraise | 11 | 4 | 0 | 0 | 4 | 7,8,9,10 |  | 0.648 | 0.328 | 0.052 | 0.066 | 0.958 | 9 | 2.073 | True |
| gpt-4o-mini | PTO | MICI_OverPraise_rate | 11 | 5 | 0 | 0 | 5 | 6,7,8,9,10 |  | 0.072 | 0.433 | 0.005 | 0.081 | 1.425 | 9 | 0.229 | True |
| gpt-4o-mini | PTO | MICI_Rate | 11 | 3 | 1 | 1 | 3 | 7,9,10 | 4 | 0.040 | 0.144 | 0.036 | 0.123 | 0.832 | 9 | 0.202 | True |
| gpt-4o-mini | PTO | MICI_Severity | 11 | 0 | 3 | 3 | 0 |  | 4,6,8 | -0.301 | -0.304 | 0.219 | 0.161 | 0.593 | 8 | -0.479 | True |
| gpt-4o-mini | PTO | MICI_Warn | 11 | 0 | 0 | 0 | 0 |  |  | -0.002 | -0.030 | 0.042 | 0.145 | 0.179 | 1 | 0.062 | True |
| gpt-4o-mini | PTO | MICI_Warn_rate | 11 | 0 | 0 | 0 | 0 |  |  | -0.000 | -0.028 | 0.005 | 0.166 | 0.178 | 1 | 0.005 | True |
| gpt-4o-mini | PTO | RtoQ | 11 | 0 | 0 | 0 | 0 |  |  | -0.047 | -0.060 | -0.127 | -0.129 | 0.264 | 7 | -0.174 | False |
| text (judge-invariant) | GRPO | conv_len | 6 | 2 | 0 | 2 | 0 | 4,5 |  | 2.100 | 0.131 | 0.479 | 0.026 | 0.531 | 5 | 8.104 | False |
| text (judge-invariant) | GRPO | loop | 6 | 0 | 1 | 0 | 1 |  | 3 | -0.002 | -0.005 | -0.010 | -0.019 | 0.300 | 3 | -0.083 | False |
| text (judge-invariant) | GRPO | mean_turn_len | 6 | 0 | 3 | 0 | 3 |  | 3,4,5 | -86.093 | -0.388 | -12.740 | -0.070 | 0.815 | 5 | -206.556 | False |
| text (judge-invariant) | GRPO | n_th_turns | 6 | 2 | 0 | 2 | 0 | 4,5 |  | 1.046 | 0.130 | 0.229 | 0.025 | 0.528 | 5 | 4.031 | False |
| text (judge-invariant) | GRPO | q_per_turn | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.172 | -0.241 | 0.088 | 0.102 | 0.605 | 5 | -0.367 | False |
| text (judge-invariant) | PTO | conv_len | 11 | 0 | 3 | 0 | 3 |  | 8,9,10 | -2.604 | -0.185 | -2.104 | -0.099 | 0.614 | 9 | -8.292 | False |
| text (judge-invariant) | PTO | loop | 11 | 0 | 0 | 0 | 0 |  |  | -0.051 | -0.146 | 0.042 | 0.064 | 0.244 | 6 | -0.073 | False |
| text (judge-invariant) | PTO | mean_turn_len | 11 | 0 | 7 | 0 | 7 |  | 3,4,5,7,8,9,10 | -89.913 | -0.369 | 25.684 | 0.118 | 0.725 | 9 | -160.185 | False |
| text (judge-invariant) | PTO | n_th_turns | 11 | 0 | 3 | 0 | 3 |  | 8,9,10 | -1.311 | -0.186 | -0.969 | -0.092 | 0.609 | 9 | -4.125 | False |
| text (judge-invariant) | PTO | q_per_turn | 11 | 0 | 0 | 0 | 0 |  |  | 0.005 | 0.015 | 0.164 | 0.172 | 0.172 | 0 | 0.164 | False |
