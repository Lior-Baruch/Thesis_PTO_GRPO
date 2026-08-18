**Per (grader, method, rubric) summary of the K contrast across iterations.** n_sig_K0_higher / n_sig_K5_higher count iterations with Holm p<.05 and delta >0 / <0; the *_better columns flip the sign for lower-better rubrics (MICI). mean_delta_iters1toN averages the per-iteration paired deltas over trained iterations only; base_delta is the iteration-0 base-vs-base draw. Sign: + = K=0 higher (K=0 minus K=5). Paired on persona_id (96 personas). p_holm = Holm across iterations 0..N within each (judge, method, metric); iteration 0 = two independent base draws (noise floor). GRPO_LA5 is right-censored at iteration 5, so GRPO rows stop at 5.

| judge | method | metric | n_iters | n_sig_K0_higher | n_sig_K5_higher | n_sig_K0_better | n_sig_K5_better | iters_sig_K0_higher | iters_sig_K5_higher | mean_delta_iters1toN | mean_dz_iters1toN | base_delta | base_dz | max_abs_dz | max_abs_dz_iter | max_abs_dz_delta | lower_better |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | GRPO | CSQ-8 | 6 | 0 | 1 | 0 | 1 |  | 4 | -0.080 | -0.122 | -0.004 | -0.007 | 0.369 | 4 | -0.243 | False |
| claude-haiku-4-5 | GRPO | MI-SAT | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.112 | -0.200 | 0.082 | 0.123 | 0.465 | 4 | -0.255 | False |
| claude-haiku-4-5 | GRPO | MICI | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.046 | -0.142 | 0.057 | 0.147 | 0.354 | 3 | -0.111 | True |
| claude-haiku-4-5 | GRPO | MITI | 6 | 1 | 0 | 1 | 0 | 3 |  | -0.005 | -0.005 | 0.057 | 0.114 | 0.291 | 3 | 0.133 | False |
| claude-haiku-4-5 | GRPO | PCT | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.029 | -0.163 | 0.019 | 0.088 | 0.373 | 5 | -0.067 | False |
| claude-haiku-4-5 | GRPO | Q1 | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.136 | -0.152 | 0.013 | 0.018 | 0.499 | 5 | -0.450 | False |
| claude-haiku-4-5 | GRPO | Q1Q2 | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.100 | -0.140 | 0.026 | 0.043 | 0.429 | 5 | -0.311 | False |
| claude-haiku-4-5 | GRPO | Q2 | 6 | 0 | 0 | 0 | 0 |  |  | -0.064 | -0.092 | 0.040 | 0.065 | 0.257 | 5 | -0.172 | False |
| claude-haiku-4-5 | GRPO | WAI-SR | 6 | 0 | 0 | 0 | 0 |  |  | -0.004 | -0.006 | 0.057 | 0.097 | 0.194 | 5 | -0.109 | False |
| claude-haiku-4-5 | PTO | CSQ-8 | 11 | 0 | 0 | 0 | 0 |  |  | 0.023 | 0.032 | -0.057 | -0.079 | 0.139 | 1 | 0.094 | False |
| claude-haiku-4-5 | PTO | MI-SAT | 11 | 0 | 0 | 0 | 0 |  |  | -0.020 | -0.042 | -0.000 | -0.000 | 0.197 | 10 | -0.111 | False |
| claude-haiku-4-5 | PTO | MICI | 11 | 2 | 1 | 1 | 2 | 9,10 | 4 | -0.004 | 0.002 | -0.006 | -0.014 | 0.655 | 10 | 0.245 | True |
| claude-haiku-4-5 | PTO | MITI | 11 | 8 | 0 | 8 | 0 | 3,4,5,6,7,8,9,10 |  | 0.175 | 0.370 | -0.013 | -0.022 | 0.738 | 6 | 0.362 | False |
| claude-haiku-4-5 | PTO | PCT | 11 | 0 | 2 | 0 | 2 |  | 4,10 | -0.028 | -0.140 | -0.003 | -0.017 | 0.279 | 4 | -0.047 | False |
| claude-haiku-4-5 | PTO | Q1 | 11 | 1 | 0 | 1 | 0 | 6 |  | 0.100 | 0.127 | -0.010 | -0.012 | 0.379 | 6 | 0.313 | False |
| claude-haiku-4-5 | PTO | Q1Q2 | 11 | 3 | 0 | 3 | 0 | 5,6,8 |  | 0.162 | 0.260 | -0.004 | -0.006 | 0.511 | 6 | 0.343 | False |
| claude-haiku-4-5 | PTO | Q2 | 11 | 6 | 0 | 6 | 0 | 5,6,7,8,9,10 |  | 0.223 | 0.380 | 0.002 | 0.004 | 0.653 | 10 | 0.363 | False |
| claude-haiku-4-5 | PTO | WAI-SR | 11 | 0 | 0 | 0 | 0 |  |  | 0.060 | 0.101 | -0.032 | -0.046 | 0.174 | 10 | 0.104 | False |
| gpt-4o-mini | GRPO | CSQ-8 | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.074 | -0.135 | 0.007 | 0.011 | 0.361 | 5 | -0.193 | False |
| gpt-4o-mini | GRPO | MI-SAT | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.113 | -0.155 | 0.042 | 0.053 | 0.345 | 5 | -0.238 | False |
| gpt-4o-mini | GRPO | MICI | 6 | 0 | 1 | 1 | 0 |  | 3 | -0.051 | -0.206 | 0.002 | 0.007 | 0.319 | 3 | -0.077 | True |
| gpt-4o-mini | GRPO | MITI | 6 | 0 | 0 | 0 | 0 |  |  | -0.020 | -0.029 | 0.104 | 0.128 | 0.167 | 3 | 0.073 | False |
| gpt-4o-mini | GRPO | PCT | 6 | 0 | 2 | 0 | 2 |  | 4,5 | -0.032 | -0.172 | 0.016 | 0.071 | 0.321 | 4 | -0.062 | False |
| gpt-4o-mini | GRPO | Q1 | 6 | 0 | 1 | 0 | 1 |  | 4 | -0.049 | -0.077 | 0.110 | 0.120 | 0.359 | 4 | -0.202 | False |
| gpt-4o-mini | GRPO | Q1Q2 | 6 | 0 | 1 | 0 | 1 |  | 4 | -0.026 | -0.036 | 0.104 | 0.115 | 0.281 | 3 | 0.132 | False |
| gpt-4o-mini | GRPO | Q2 | 6 | 0 | 0 | 0 | 0 |  |  | -0.004 | 0.013 | 0.097 | 0.103 | 0.278 | 3 | 0.134 | False |
| gpt-4o-mini | GRPO | WAI-SR | 6 | 0 | 1 | 0 | 1 |  | 5 | -0.090 | -0.161 | 0.033 | 0.051 | 0.446 | 5 | -0.255 | False |
| gpt-4o-mini | PTO | CSQ-8 | 11 | 0 | 0 | 0 | 0 |  |  | 0.015 | 0.025 | 0.046 | 0.068 | 0.184 | 9 | 0.105 | False |
| gpt-4o-mini | PTO | MI-SAT | 11 | 0 | 0 | 0 | 0 |  |  | -0.026 | -0.034 | -0.002 | -0.002 | 0.187 | 5 | -0.127 | False |
| gpt-4o-mini | PTO | MICI | 11 | 3 | 1 | 1 | 3 | 7,9,10 | 4 | 0.040 | 0.144 | 0.036 | 0.123 | 0.832 | 9 | 0.202 | True |
| gpt-4o-mini | PTO | MITI | 11 | 1 | 0 | 1 | 0 | 6 |  | 0.092 | 0.140 | -0.016 | -0.017 | 0.345 | 6 | 0.211 | False |
| gpt-4o-mini | PTO | PCT | 11 | 0 | 0 | 0 | 0 |  |  | -0.009 | -0.041 | 0.007 | 0.038 | 0.114 | 1 | -0.027 | False |
| gpt-4o-mini | PTO | Q1 | 11 | 0 | 0 | 0 | 0 |  |  | 0.035 | 0.042 | -0.023 | -0.022 | 0.271 | 6 | 0.173 | False |
| gpt-4o-mini | PTO | Q1Q2 | 11 | 1 | 0 | 1 | 0 | 6 |  | 0.087 | 0.131 | -0.003 | -0.003 | 0.417 | 6 | 0.257 | False |
| gpt-4o-mini | PTO | Q2 | 11 | 2 | 0 | 2 | 0 | 6,8 |  | 0.139 | 0.222 | 0.017 | 0.016 | 0.521 | 6 | 0.341 | False |
| gpt-4o-mini | PTO | WAI-SR | 11 | 0 | 0 | 0 | 0 |  |  | -0.017 | -0.029 | 0.025 | 0.035 | 0.191 | 5 | -0.097 | False |
