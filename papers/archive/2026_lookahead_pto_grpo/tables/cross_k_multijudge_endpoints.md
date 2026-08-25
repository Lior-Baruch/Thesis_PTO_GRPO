**Endpoint contrasts under both graders** (`A − B` as named in `pair`; + => A higher; on MICI, lower is better, so + favours B — read `favours_*`, where A/B are the pair's left/right model). `primary_*` = training oracle gpt-4o-mini; `judge_*` = held-out Claude Haiku 4.5. Paired on persona_id (the trainer reshuffles the 96 personas every iteration; file_index is not a pairing key). CI = persona bootstrap; p = Wilcoxon; `*_p_holm` = Holm across the 9 rubrics within a pair (the tracked EDA's `compare_two_models` convention). GRPO_LA0's best iteration is chosen by mean Q1Q2 under each grader (primary I8, held-out I3). GRPO_LA5 is right-censored at iteration 5 (its full budget); PTO arms and GRPO_LA0 run to 10.

| pair | metric | primary_n | primary_delta | primary_dz | primary_ci_lo | primary_ci_hi | primary_p | primary_p_holm | judge_delta | judge_dz | judge_ci_lo | judge_ci_hi | judge_p | judge_p_holm | same_sign | judge_ci_excl0 | favours_primary | favours_judge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | Q1Q2 | 96 | 0.507 | 0.729 | 0.375 | 0.645 | 0.000 | 0.000 | 0.609 | 1.265 | 0.510 | 0.702 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | Q1 | 96 | 0.533 | 0.708 | 0.392 | 0.685 | 0.000 | 0.000 | 0.773 | 1.209 | 0.648 | 0.894 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | Q2 | 96 | 0.481 | 0.700 | 0.347 | 0.617 | 0.000 | 0.000 | 0.445 | 0.931 | 0.349 | 0.539 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | WAI-SR | 96 | 0.059 | 0.117 | -0.038 | 0.161 | 0.242 | 0.242 | 0.286 | 0.476 | 0.170 | 0.403 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | CSQ-8 | 96 | 0.172 | 0.310 | 0.066 | 0.281 | 0.004 | 0.015 | 0.324 | 0.603 | 0.223 | 0.434 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | MI-SAT | 96 | 0.174 | 0.267 | 0.052 | 0.302 | 0.004 | 0.015 | 0.248 | 0.510 | 0.158 | 0.351 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | MITI | 96 | 0.352 | 0.459 | 0.206 | 0.510 | 0.000 | 0.000 | 0.253 | 0.648 | 0.177 | 0.331 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | PCT | 96 | 0.056 | 0.287 | 0.019 | 0.094 | 0.005 | 0.015 | 0.021 | 0.122 | -0.012 | 0.056 | 0.204 | 0.204 | True | False | A | A |
| PTO_LA0_I10 − GRPO_LA0_I10 (K=0 headline) | MICI | 96 | -0.346 | -0.989 | -0.414 | -0.275 | 0.000 | 0.000 | -0.224 | -0.667 | -0.291 | -0.157 | 0.000 | 0.000 | True | True | A | A |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | Q1Q2 | 96 | 0.265 | 0.378 | 0.131 | 0.418 | 0.000 | 0.001 | -0.131 | -0.187 | -0.271 | 0.011 | 0.146 | 0.818 | False | False | A | B |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | Q1 | 96 | 0.238 | 0.290 | 0.077 | 0.415 | 0.003 | 0.015 | -0.206 | -0.227 | -0.379 | -0.017 | 0.048 | 0.384 | False | True | A | B |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | Q2 | 96 | 0.293 | 0.466 | 0.171 | 0.427 | 0.000 | 0.000 | -0.055 | -0.094 | -0.174 | 0.067 | 0.389 | 0.818 | False | False | A | B |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | WAI-SR | 96 | 0.072 | 0.116 | -0.055 | 0.197 | 0.031 | 0.094 | 0.041 | 0.064 | -0.087 | 0.171 | 0.375 | 0.818 | True | False | A | A |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | CSQ-8 | 96 | 0.046 | 0.067 | -0.085 | 0.186 | 0.454 | 0.454 | 0.072 | 0.120 | -0.043 | 0.189 | 0.136 | 0.818 | True | False | A | A |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | MI-SAT | 96 | 0.151 | 0.180 | -0.014 | 0.323 | 0.042 | 0.094 | 0.087 | 0.140 | -0.040 | 0.212 | 0.204 | 0.818 | True | False | A | A |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | MITI | 96 | 0.247 | 0.377 | 0.115 | 0.378 | 0.000 | 0.000 | -0.036 | -0.091 | -0.115 | 0.047 | 0.242 | 0.818 | False | False | A | B |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | PCT | 96 | 0.060 | 0.272 | 0.018 | 0.103 | 0.001 | 0.004 | 0.062 | 0.262 | 0.017 | 0.109 | 0.002 | 0.015 | True | True | A | A |
| PTO_LA5_I10 − GRPO_LA5_I5 (K=5 endpoints) | MICI | 96 | -0.077 | -0.288 | -0.130 | -0.022 | 0.004 | 0.015 | -0.066 | -0.174 | -0.141 | 0.009 | 0.096 | 0.672 | True | False | A | A |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | Q1Q2 | 96 | 0.047 | 0.096 | -0.054 | 0.142 | 0.087 | 0.695 | -0.199 | -0.308 | -0.332 | -0.068 | 0.032 | 0.129 | False | True | A | B |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | Q1 | 96 | 0.085 | 0.139 | -0.044 | 0.208 | 0.094 | 0.695 | -0.035 | -0.041 | -0.210 | 0.135 | 0.854 | 1.000 | False | False | A | B |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | Q2 | 96 | 0.009 | 0.020 | -0.080 | 0.093 | 0.304 | 1.000 | -0.363 | -0.653 | -0.478 | -0.251 | 0.000 | 0.000 | False | True | A | B |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | WAI-SR | 96 | 0.038 | 0.073 | -0.076 | 0.145 | 0.287 | 1.000 | -0.104 | -0.174 | -0.226 | 0.014 | 0.119 | 0.356 | False | False | A | B |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | CSQ-8 | 96 | 0.008 | 0.016 | -0.094 | 0.108 | 0.626 | 1.000 | -0.001 | -0.002 | -0.128 | 0.113 | 0.996 | 1.000 | False | False | A | B |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | MI-SAT | 96 | 0.057 | 0.085 | -0.085 | 0.189 | 0.099 | 0.695 | 0.111 | 0.197 | -0.002 | 0.226 | 0.017 | 0.085 | True | False | A | A |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | MITI | 96 | -0.016 | -0.026 | -0.143 | 0.104 | 0.945 | 1.000 | -0.203 | -0.487 | -0.286 | -0.125 | 0.000 | 0.000 | True | True | B | B |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | PCT | 96 | 0.008 | 0.044 | -0.029 | 0.043 | 0.268 | 1.000 | 0.051 | 0.253 | 0.010 | 0.092 | 0.003 | 0.017 | True | True | A | A |
| PTO_LA5_I10 − PTO_LA0_I10 (K lever, PTO endpoint) | MICI | 96 | -0.228 | -0.708 | -0.291 | -0.164 | 0.000 | 0.000 | -0.245 | -0.655 | -0.316 | -0.168 | 0.000 | 0.000 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | Q1Q2 | 96 | 0.070 | 0.135 | -0.036 | 0.177 | 0.122 | 0.365 | 0.311 | 0.429 | 0.168 | 0.455 | 0.001 | 0.006 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | Q1 | 96 | 0.119 | 0.210 | 0.004 | 0.240 | 0.041 | 0.205 | 0.450 | 0.499 | 0.267 | 0.629 | 0.000 | 0.000 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | Q2 | 96 | 0.021 | 0.039 | -0.089 | 0.124 | 0.359 | 0.365 | 0.172 | 0.257 | 0.038 | 0.308 | 0.061 | 0.183 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | WAI-SR | 96 | 0.255 | 0.446 | 0.144 | 0.378 | 0.000 | 0.001 | 0.109 | 0.194 | -0.002 | 0.220 | 0.138 | 0.275 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | CSQ-8 | 96 | 0.193 | 0.361 | 0.082 | 0.303 | 0.001 | 0.006 | 0.137 | 0.224 | 0.013 | 0.255 | 0.028 | 0.138 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | MI-SAT | 96 | 0.238 | 0.345 | 0.099 | 0.385 | 0.003 | 0.022 | 0.229 | 0.402 | 0.115 | 0.340 | 0.000 | 0.003 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | MITI | 96 | 0.070 | 0.146 | -0.029 | 0.169 | 0.164 | 0.365 | 0.099 | 0.212 | 0.003 | 0.190 | 0.043 | 0.171 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | PCT | 96 | 0.056 | 0.309 | 0.019 | 0.094 | 0.003 | 0.022 | 0.067 | 0.373 | 0.031 | 0.105 | 0.001 | 0.004 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I5 (K lever, GRPO matched iter) | MICI | 96 | 0.063 | 0.243 | 0.013 | 0.117 | 0.044 | 0.205 | 0.018 | 0.053 | -0.050 | 0.085 | 0.636 | 0.636 | True | False | B | B |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | Q1Q2 | 96 | 0.289 | 0.359 | 0.120 | 0.448 | 0.003 | 0.019 | 0.541 | 0.838 | 0.399 | 0.667 | 0.000 | 0.000 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | Q1 | 96 | 0.381 | 0.422 | 0.196 | 0.567 | 0.000 | 0.001 | 0.944 | 1.100 | 0.758 | 1.108 | 0.000 | 0.000 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | Q2 | 96 | 0.197 | 0.262 | 0.042 | 0.349 | 0.111 | 0.553 | 0.137 | 0.242 | 0.018 | 0.246 | 0.022 | 0.087 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | WAI-SR | 96 | 0.025 | 0.043 | -0.093 | 0.142 | 0.597 | 1.000 | 0.141 | 0.205 | 0.002 | 0.280 | 0.107 | 0.214 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | CSQ-8 | 96 | 0.134 | 0.216 | 0.008 | 0.260 | 0.030 | 0.182 | 0.251 | 0.345 | 0.102 | 0.397 | 0.002 | 0.010 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | MI-SAT | 96 | 0.080 | 0.097 | -0.090 | 0.247 | 0.214 | 0.855 | 0.273 | 0.385 | 0.128 | 0.417 | 0.000 | 0.001 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | MITI | 96 | 0.089 | 0.115 | -0.060 | 0.245 | 0.616 | 1.000 | 0.086 | 0.203 | -0.003 | 0.169 | 0.043 | 0.130 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | PCT | 96 | 0.004 | 0.018 | -0.042 | 0.050 | 0.991 | 1.000 | 0.010 | 0.046 | -0.033 | 0.054 | 0.806 | 0.806 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint) | MICI | 96 | -0.497 | -1.339 | -0.570 | -0.420 | 0.000 | 0.000 | -0.403 | -1.228 | -0.467 | -0.335 | 0.000 | 0.000 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | Q1Q2 | 96 | -0.041 | -0.065 | -0.163 | 0.082 | 0.346 | 1.000 | 0.181 | 0.287 | 0.054 | 0.308 | 0.018 | 0.089 | False | True | B | A |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | Q1 | 96 | 0.052 | 0.072 | -0.090 | 0.198 | 0.533 | 1.000 | 0.485 | 0.604 | 0.323 | 0.646 | 0.000 | 0.000 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | Q2 | 96 | -0.134 | -0.224 | -0.246 | -0.017 | 0.002 | 0.013 | -0.124 | -0.215 | -0.235 | -0.007 | 0.012 | 0.074 | True | True | B | B |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | WAI-SR | 96 | 0.089 | 0.152 | -0.027 | 0.210 | 0.277 | 1.000 | -0.085 | -0.142 | -0.197 | 0.040 | 0.043 | 0.171 | False | False | A | B |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | CSQ-8 | 96 | 0.159 | 0.235 | 0.022 | 0.293 | 0.012 | 0.073 | 0.055 | 0.083 | -0.069 | 0.186 | 0.751 | 0.853 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | MI-SAT | 96 | 0.118 | 0.129 | -0.068 | 0.304 | 0.077 | 0.387 | 0.181 | 0.295 | 0.059 | 0.300 | 0.004 | 0.028 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | MITI | 96 | -0.224 | -0.401 | -0.328 | -0.107 | 0.000 | 0.000 | -0.034 | -0.091 | -0.107 | 0.036 | 0.412 | 0.853 | True | False | B | B |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | PCT | 96 | 0.007 | 0.029 | -0.041 | 0.051 | 0.530 | 1.000 | 0.026 | 0.122 | -0.016 | 0.070 | 0.284 | 0.853 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I8 (K=0 best by primary Q1Q2) | MICI | 96 | -0.195 | -0.618 | -0.255 | -0.130 | 0.000 | 0.000 | -0.252 | -0.619 | -0.332 | -0.169 | 0.000 | 0.000 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | Q1Q2 | 96 | 0.048 | 0.102 | -0.046 | 0.143 | 0.063 | 0.251 | 0.161 | 0.310 | 0.057 | 0.263 | 0.007 | 0.040 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | Q1 | 96 | 0.052 | 0.099 | -0.054 | 0.160 | 0.285 | 0.290 | 0.233 | 0.343 | 0.096 | 0.371 | 0.003 | 0.020 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | Q2 | 96 | 0.044 | 0.095 | -0.047 | 0.134 | 0.097 | 0.290 | 0.089 | 0.170 | -0.017 | 0.194 | 0.196 | 0.742 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | WAI-SR | 96 | 0.162 | 0.346 | 0.072 | 0.261 | 0.003 | 0.023 | -0.048 | -0.097 | -0.143 | 0.051 | 0.185 | 0.742 | False | False | A | B |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | CSQ-8 | 96 | 0.096 | 0.193 | -0.003 | 0.202 | 0.039 | 0.235 | 0.036 | 0.065 | -0.074 | 0.145 | 0.640 | 0.742 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | MI-SAT | 96 | 0.134 | 0.210 | 0.010 | 0.267 | 0.046 | 0.235 | 0.116 | 0.222 | 0.016 | 0.226 | 0.044 | 0.218 | True | True | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | MITI | 96 | 0.073 | 0.152 | -0.023 | 0.174 | 0.139 | 0.290 | -0.198 | -0.441 | -0.286 | -0.109 | 0.000 | 0.000 | False | True | A | B |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | PCT | 96 | 0.035 | 0.203 | 0.001 | 0.069 | 0.015 | 0.105 | 0.022 | 0.129 | -0.010 | 0.058 | 0.315 | 0.742 | True | False | A | A |
| GRPO_LA5_I5 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2) | MICI | 96 | 0.110 | 0.402 | 0.058 | 0.166 | 0.000 | 0.002 | 0.239 | 0.763 | 0.177 | 0.302 | 0.000 | 0.000 | True | True | B | B |
