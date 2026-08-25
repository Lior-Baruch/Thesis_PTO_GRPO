**K0-K5 contrast WITHIN patient cooperation level** (persona trait from the patient system prompt: High -> Cooperative, StartLowAndChangesToHigh -> Warms up, Low -> Resistant; 32 personas each), on Q1Q2 (the training reward, 1-5), MICI (MI-inconsistent behaviours per therapist turn; LOWER = better, so a positive delta means K=0 is WORSE) and PCT (change-talk proportion; higher = better). K-contrast sign: + => K=0 higher (K0 - K5). Paired on persona_id (the recovered patient persona), never file_index. target=matched_final: PTO iter 10 vs 10, GRPO iter 5 vs 5 (GRPO_LA5 right-censored at 5); target=own_best: each arm at its own-oracle best iteration (selected on the training oracle's Q1Q2 mean; iter_K0/iter_K5 columns). mean_K0/mean_K5 = subgroup arm means on the paired personas; dz = mean/sd of paired deltas; 95% percentile-bootstrap CI; p = Wilcoxon signed-rank; p_holm = Holm across the three cooperation subgroups within (judge, method, metric, target) — the 'All' row (all 96 personas) is a reference, outside the family. `share_*_ge_4.5` (Q1Q2 only) = fraction of that arm's subgroup conversations scoring >= 4.5, the ceiling diagnostic for the Cooperative stratum. Graders side by side, never averaged.

| judge | method | metric | target | iter_K0 | iter_K5 | cooperation | n | mean_K0 | mean_K5 | mean_delta | dz | ci_lo | ci_hi | p | p_holm | share_K0_ge_4.5 | share_K5_ge_4.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | PTO | Q1Q2 | matched_final | 10 | 10 | Cooperative | 32 | 4.904 | 4.861 | 0.043 | 0.290 | -0.004 | 0.099 | 0.230 | 0.461 | 1.000 | 0.938 |
| gpt-4o-mini | PTO | Q1Q2 | matched_final | 10 | 10 | Warms up | 32 | 4.210 | 4.352 | -0.142 | -0.334 | -0.288 | 0.004 | 0.014 | 0.043 | 0.219 | 0.219 |
| gpt-4o-mini | PTO | Q1Q2 | matched_final | 10 | 10 | Resistant | 32 | 3.665 | 3.706 | -0.042 | -0.058 | -0.283 | 0.198 | 0.581 | 0.581 | 0.000 | 0.000 |
| gpt-4o-mini | PTO | Q1Q2 | matched_final | 10 | 10 | All | 96 | 4.260 | 4.307 | -0.047 | -0.096 | -0.142 | 0.054 | 0.087 |  | 0.406 | 0.385 |
| gpt-4o-mini | PTO | Q1Q2 | own_best | 10 | 10 | Cooperative | 32 | 4.904 | 4.861 | 0.043 | 0.290 | -0.004 | 0.099 | 0.230 | 0.461 | 1.000 | 0.938 |
| gpt-4o-mini | PTO | Q1Q2 | own_best | 10 | 10 | Warms up | 32 | 4.210 | 4.352 | -0.142 | -0.334 | -0.288 | 0.004 | 0.014 | 0.043 | 0.219 | 0.219 |
| gpt-4o-mini | PTO | Q1Q2 | own_best | 10 | 10 | Resistant | 32 | 3.665 | 3.706 | -0.042 | -0.058 | -0.283 | 0.198 | 0.581 | 0.581 | 0.000 | 0.000 |
| gpt-4o-mini | PTO | Q1Q2 | own_best | 10 | 10 | All | 96 | 4.260 | 4.307 | -0.047 | -0.096 | -0.142 | 0.054 | 0.087 |  | 0.406 | 0.385 |
| gpt-4o-mini | PTO | MICI | matched_final | 10 | 10 | Cooperative | 32 | 0.502 | 0.257 | 0.246 | 0.861 | 0.145 | 0.337 | 0.000 | 0.000 |  |  |
| gpt-4o-mini | PTO | MICI | matched_final | 10 | 10 | Warms up | 32 | 0.551 | 0.303 | 0.247 | 0.592 | 0.101 | 0.386 | 0.003 | 0.003 |  |  |
| gpt-4o-mini | PTO | MICI | matched_final | 10 | 10 | Resistant | 32 | 0.421 | 0.231 | 0.190 | 0.782 | 0.112 | 0.275 | 0.001 | 0.001 |  |  |
| gpt-4o-mini | PTO | MICI | matched_final | 10 | 10 | All | 96 | 0.491 | 0.264 | 0.228 | 0.708 | 0.164 | 0.291 | 0.000 |  |  |  |
| gpt-4o-mini | PTO | MICI | own_best | 10 | 10 | Cooperative | 32 | 0.502 | 0.257 | 0.246 | 0.861 | 0.145 | 0.337 | 0.000 | 0.000 |  |  |
| gpt-4o-mini | PTO | MICI | own_best | 10 | 10 | Warms up | 32 | 0.551 | 0.303 | 0.247 | 0.592 | 0.101 | 0.386 | 0.003 | 0.003 |  |  |
| gpt-4o-mini | PTO | MICI | own_best | 10 | 10 | Resistant | 32 | 0.421 | 0.231 | 0.190 | 0.782 | 0.112 | 0.275 | 0.001 | 0.001 |  |  |
| gpt-4o-mini | PTO | MICI | own_best | 10 | 10 | All | 96 | 0.491 | 0.264 | 0.228 | 0.708 | 0.164 | 0.291 | 0.000 |  |  |  |
| gpt-4o-mini | PTO | PCT | matched_final | 10 | 10 | Cooperative | 32 | 0.876 | 0.870 | 0.006 | 0.046 | -0.039 | 0.048 | 0.829 | 1.000 |  |  |
| gpt-4o-mini | PTO | PCT | matched_final | 10 | 10 | Warms up | 32 | 0.567 | 0.614 | -0.047 | -0.252 | -0.108 | 0.018 | 0.005 | 0.016 |  |  |
| gpt-4o-mini | PTO | PCT | matched_final | 10 | 10 | Resistant | 32 | 0.447 | 0.429 | 0.018 | 0.089 | -0.053 | 0.090 | 0.854 | 1.000 |  |  |
| gpt-4o-mini | PTO | PCT | matched_final | 10 | 10 | All | 96 | 0.630 | 0.638 | -0.008 | -0.044 | -0.043 | 0.029 | 0.268 |  |  |  |
| gpt-4o-mini | PTO | PCT | own_best | 10 | 10 | Cooperative | 32 | 0.876 | 0.870 | 0.006 | 0.046 | -0.039 | 0.048 | 0.829 | 1.000 |  |  |
| gpt-4o-mini | PTO | PCT | own_best | 10 | 10 | Warms up | 32 | 0.567 | 0.614 | -0.047 | -0.252 | -0.108 | 0.018 | 0.005 | 0.016 |  |  |
| gpt-4o-mini | PTO | PCT | own_best | 10 | 10 | Resistant | 32 | 0.447 | 0.429 | 0.018 | 0.089 | -0.053 | 0.090 | 0.854 | 1.000 |  |  |
| gpt-4o-mini | PTO | PCT | own_best | 10 | 10 | All | 96 | 0.630 | 0.638 | -0.008 | -0.044 | -0.043 | 0.029 | 0.268 |  |  |  |
| gpt-4o-mini | GRPO | Q1Q2 | matched_final | 5 | 5 | Cooperative | 32 | 4.767 | 4.797 | -0.030 | -0.129 | -0.117 | 0.041 | 0.918 | 1.000 | 0.875 | 1.000 |
| gpt-4o-mini | GRPO | Q1Q2 | matched_final | 5 | 5 | Warms up | 32 | 4.086 | 4.290 | -0.204 | -0.557 | -0.325 | -0.083 | 0.007 | 0.020 | 0.031 | 0.000 |
| gpt-4o-mini | GRPO | Q1Q2 | matched_final | 5 | 5 | Resistant | 32 | 3.063 | 3.038 | 0.025 | 0.032 | -0.246 | 0.292 | 0.784 | 1.000 | 0.000 | 0.000 |
| gpt-4o-mini | GRPO | Q1Q2 | matched_final | 5 | 5 | All | 96 | 3.972 | 4.042 | -0.070 | -0.135 | -0.177 | 0.036 | 0.122 |  | 0.302 | 0.333 |
| gpt-4o-mini | GRPO | Q1Q2 | own_best | 8 | 4 | Cooperative | 32 | 4.920 | 4.752 | 0.168 | 1.584 | 0.135 | 0.205 | 0.000 | 0.000 | 1.000 | 1.000 |
| gpt-4o-mini | GRPO | Q1Q2 | own_best | 8 | 4 | Warms up | 32 | 4.031 | 4.247 | -0.216 | -0.391 | -0.426 | -0.046 | 0.040 | 0.079 | 0.062 | 0.000 |
| gpt-4o-mini | GRPO | Q1Q2 | own_best | 8 | 4 | Resistant | 32 | 3.296 | 3.360 | -0.064 | -0.103 | -0.279 | 0.143 | 0.454 | 0.454 | 0.000 | 0.000 |
| gpt-4o-mini | GRPO | Q1Q2 | own_best | 8 | 4 | All | 96 | 4.082 | 4.120 | -0.038 | -0.074 | -0.137 | 0.053 | 0.789 |  | 0.354 | 0.333 |
| gpt-4o-mini | GRPO | MICI | matched_final | 5 | 5 | Cooperative | 32 | 0.262 | 0.334 | -0.072 | -0.273 | -0.161 | 0.020 | 0.217 | 0.651 |  |  |
| gpt-4o-mini | GRPO | MICI | matched_final | 5 | 5 | Warms up | 32 | 0.305 | 0.366 | -0.061 | -0.211 | -0.158 | 0.035 | 0.276 | 0.651 |  |  |
| gpt-4o-mini | GRPO | MICI | matched_final | 5 | 5 | Resistant | 32 | 0.265 | 0.321 | -0.056 | -0.243 | -0.136 | 0.022 | 0.318 | 0.651 |  |  |
| gpt-4o-mini | GRPO | MICI | matched_final | 5 | 5 | All | 96 | 0.277 | 0.340 | -0.063 | -0.243 | -0.117 | -0.013 | 0.044 |  |  |  |
| gpt-4o-mini | GRPO | MICI | own_best | 8 | 4 | Cooperative | 32 | 0.694 | 0.318 | 0.376 | 1.196 | 0.269 | 0.476 | 0.000 | 0.000 |  |  |
| gpt-4o-mini | GRPO | MICI | own_best | 8 | 4 | Warms up | 32 | 0.523 | 0.292 | 0.231 | 0.803 | 0.137 | 0.332 | 0.000 | 0.000 |  |  |
| gpt-4o-mini | GRPO | MICI | own_best | 8 | 4 | Resistant | 32 | 0.388 | 0.305 | 0.083 | 0.391 | 0.015 | 0.159 | 0.110 | 0.110 |  |  |
| gpt-4o-mini | GRPO | MICI | own_best | 8 | 4 | All | 96 | 0.535 | 0.305 | 0.230 | 0.774 | 0.174 | 0.285 | 0.000 |  |  |  |
| gpt-4o-mini | GRPO | PCT | matched_final | 5 | 5 | Cooperative | 32 | 0.843 | 0.830 | 0.014 | 0.119 | -0.026 | 0.051 | 0.299 | 0.342 |  |  |
| gpt-4o-mini | GRPO | PCT | matched_final | 5 | 5 | Warms up | 32 | 0.510 | 0.628 | -0.119 | -0.896 | -0.164 | -0.074 | 0.000 | 0.000 |  |  |
| gpt-4o-mini | GRPO | PCT | matched_final | 5 | 5 | Resistant | 32 | 0.214 | 0.277 | -0.063 | -0.257 | -0.149 | 0.020 | 0.171 | 0.342 |  |  |
| gpt-4o-mini | GRPO | PCT | matched_final | 5 | 5 | All | 96 | 0.522 | 0.578 | -0.056 | -0.309 | -0.094 | -0.019 | 0.003 |  |  |  |
| gpt-4o-mini | GRPO | PCT | own_best | 8 | 4 | Cooperative | 32 | 0.866 | 0.849 | 0.016 | 0.184 | -0.013 | 0.047 | 0.453 | 0.490 |  |  |
| gpt-4o-mini | GRPO | PCT | own_best | 8 | 4 | Warms up | 32 | 0.535 | 0.607 | -0.073 | -0.414 | -0.138 | -0.015 | 0.003 | 0.008 |  |  |
| gpt-4o-mini | GRPO | PCT | own_best | 8 | 4 | Resistant | 32 | 0.315 | 0.384 | -0.070 | -0.217 | -0.177 | 0.044 | 0.245 | 0.490 |  |  |
| gpt-4o-mini | GRPO | PCT | own_best | 8 | 4 | All | 96 | 0.572 | 0.614 | -0.042 | -0.192 | -0.084 | 0.002 | 0.040 |  |  |  |
| claude-haiku-4-5 | PTO | Q1Q2 | matched_final | 10 | 10 | Cooperative | 32 | 3.666 | 3.095 | 0.571 | 0.773 | 0.313 | 0.818 | 0.001 | 0.002 | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | Q1Q2 | matched_final | 10 | 10 | Warms up | 32 | 2.720 | 2.792 | -0.072 | -0.123 | -0.263 | 0.143 | 0.122 | 0.243 | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | Q1Q2 | matched_final | 10 | 10 | Resistant | 32 | 2.212 | 2.113 | 0.099 | 0.246 | -0.039 | 0.231 | 0.140 | 0.243 | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | Q1Q2 | matched_final | 10 | 10 | All | 96 | 2.866 | 2.667 | 0.199 | 0.308 | 0.068 | 0.332 | 0.032 |  | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | Q1Q2 | own_best | 10 | 10 | Cooperative | 32 | 3.666 | 3.095 | 0.571 | 0.773 | 0.313 | 0.818 | 0.001 | 0.002 | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | Q1Q2 | own_best | 10 | 10 | Warms up | 32 | 2.720 | 2.792 | -0.072 | -0.123 | -0.263 | 0.143 | 0.122 | 0.243 | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | Q1Q2 | own_best | 10 | 10 | Resistant | 32 | 2.212 | 2.113 | 0.099 | 0.246 | -0.039 | 0.231 | 0.140 | 0.243 | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | Q1Q2 | own_best | 10 | 10 | All | 96 | 2.866 | 2.667 | 0.199 | 0.308 | 0.068 | 0.332 | 0.032 |  | 0.000 | 0.000 |
| claude-haiku-4-5 | PTO | MICI | matched_final | 10 | 10 | Cooperative | 32 | 0.812 | 0.402 | 0.410 | 1.652 | 0.320 | 0.488 | 0.000 | 0.000 |  |  |
| claude-haiku-4-5 | PTO | MICI | matched_final | 10 | 10 | Warms up | 32 | 0.840 | 0.719 | 0.121 | 0.303 | -0.028 | 0.250 | 0.042 | 0.042 |  |  |
| claude-haiku-4-5 | PTO | MICI | matched_final | 10 | 10 | Resistant | 32 | 0.824 | 0.621 | 0.203 | 0.507 | 0.072 | 0.345 | 0.007 | 0.013 |  |  |
| claude-haiku-4-5 | PTO | MICI | matched_final | 10 | 10 | All | 96 | 0.825 | 0.581 | 0.245 | 0.655 | 0.168 | 0.316 | 0.000 |  |  |  |
| claude-haiku-4-5 | PTO | MICI | own_best | 10 | 10 | Cooperative | 32 | 0.812 | 0.402 | 0.410 | 1.652 | 0.320 | 0.488 | 0.000 | 0.000 |  |  |
| claude-haiku-4-5 | PTO | MICI | own_best | 10 | 10 | Warms up | 32 | 0.840 | 0.719 | 0.121 | 0.303 | -0.028 | 0.250 | 0.042 | 0.042 |  |  |
| claude-haiku-4-5 | PTO | MICI | own_best | 10 | 10 | Resistant | 32 | 0.824 | 0.621 | 0.203 | 0.507 | 0.072 | 0.345 | 0.007 | 0.013 |  |  |
| claude-haiku-4-5 | PTO | MICI | own_best | 10 | 10 | All | 96 | 0.825 | 0.581 | 0.245 | 0.655 | 0.168 | 0.316 | 0.000 |  |  |  |
| claude-haiku-4-5 | PTO | PCT | matched_final | 10 | 10 | Cooperative | 32 | 0.975 | 0.973 | 0.003 | 0.043 | -0.018 | 0.024 | 0.758 | 0.758 |  |  |
| claude-haiku-4-5 | PTO | PCT | matched_final | 10 | 10 | Warms up | 32 | 0.579 | 0.683 | -0.104 | -0.459 | -0.175 | -0.023 | 0.001 | 0.003 |  |  |
| claude-haiku-4-5 | PTO | PCT | matched_final | 10 | 10 | Resistant | 32 | 0.345 | 0.398 | -0.053 | -0.206 | -0.138 | 0.035 | 0.231 | 0.462 |  |  |
| claude-haiku-4-5 | PTO | PCT | matched_final | 10 | 10 | All | 96 | 0.633 | 0.685 | -0.051 | -0.253 | -0.092 | -0.010 | 0.003 |  |  |  |
| claude-haiku-4-5 | PTO | PCT | own_best | 10 | 10 | Cooperative | 32 | 0.975 | 0.973 | 0.003 | 0.043 | -0.018 | 0.024 | 0.758 | 0.758 |  |  |
| claude-haiku-4-5 | PTO | PCT | own_best | 10 | 10 | Warms up | 32 | 0.579 | 0.683 | -0.104 | -0.459 | -0.175 | -0.023 | 0.001 | 0.003 |  |  |
| claude-haiku-4-5 | PTO | PCT | own_best | 10 | 10 | Resistant | 32 | 0.345 | 0.398 | -0.053 | -0.206 | -0.138 | 0.035 | 0.231 | 0.462 |  |  |
| claude-haiku-4-5 | PTO | PCT | own_best | 10 | 10 | All | 96 | 0.633 | 0.685 | -0.051 | -0.253 | -0.092 | -0.010 | 0.003 |  |  |  |
| claude-haiku-4-5 | GRPO | Q1Q2 | matched_final | 5 | 5 | Cooperative | 32 | 3.180 | 3.452 | -0.273 | -0.356 | -0.549 | -0.037 | 0.246 | 0.316 | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | Q1Q2 | matched_final | 5 | 5 | Warms up | 32 | 2.574 | 3.050 | -0.476 | -0.589 | -0.749 | -0.193 | 0.003 | 0.010 | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | Q1Q2 | matched_final | 5 | 5 | Resistant | 32 | 1.706 | 1.891 | -0.185 | -0.324 | -0.385 | 0.002 | 0.158 | 0.316 | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | Q1Q2 | matched_final | 5 | 5 | All | 96 | 2.487 | 2.798 | -0.311 | -0.429 | -0.455 | -0.168 | 0.001 |  | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | Q1Q2 | own_best | 8 | 4 | Cooperative | 32 | 3.463 | 3.297 | 0.166 | 0.275 | -0.028 | 0.383 | 0.135 | 0.135 | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | Q1Q2 | own_best | 8 | 4 | Warms up | 32 | 2.487 | 2.934 | -0.447 | -0.686 | -0.672 | -0.226 | 0.001 | 0.003 | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | Q1Q2 | own_best | 8 | 4 | Resistant | 32 | 1.902 | 2.120 | -0.218 | -0.474 | -0.374 | -0.064 | 0.016 | 0.032 | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | Q1Q2 | own_best | 8 | 4 | All | 96 | 2.617 | 2.784 | -0.166 | -0.266 | -0.291 | -0.041 | 0.012 |  | 0.000 | 0.000 |
| claude-haiku-4-5 | GRPO | MICI | matched_final | 5 | 5 | Cooperative | 32 | 0.652 | 0.519 | 0.133 | 0.371 | 0.008 | 0.255 | 0.017 | 0.052 |  |  |
| claude-haiku-4-5 | GRPO | MICI | matched_final | 5 | 5 | Warms up | 32 | 0.636 | 0.710 | -0.073 | -0.259 | -0.163 | 0.021 | 0.144 | 0.144 |  |  |
| claude-haiku-4-5 | GRPO | MICI | matched_final | 5 | 5 | Resistant | 32 | 0.598 | 0.711 | -0.114 | -0.348 | -0.228 | -0.008 | 0.070 | 0.140 |  |  |
| claude-haiku-4-5 | GRPO | MICI | matched_final | 5 | 5 | All | 96 | 0.629 | 0.647 | -0.018 | -0.053 | -0.085 | 0.050 | 0.636 |  |  |  |
| claude-haiku-4-5 | GRPO | MICI | own_best | 8 | 4 | Cooperative | 32 | 1.036 | 0.453 | 0.584 | 1.463 | 0.448 | 0.722 | 0.000 | 0.000 |  |  |
| claude-haiku-4-5 | GRPO | MICI | own_best | 8 | 4 | Warms up | 32 | 0.836 | 0.615 | 0.220 | 0.816 | 0.130 | 0.315 | 0.000 | 0.000 |  |  |
| claude-haiku-4-5 | GRPO | MICI | own_best | 8 | 4 | Resistant | 32 | 0.823 | 0.618 | 0.205 | 0.638 | 0.097 | 0.321 | 0.001 | 0.001 |  |  |
| claude-haiku-4-5 | GRPO | MICI | own_best | 8 | 4 | All | 96 | 0.898 | 0.562 | 0.336 | 0.898 | 0.259 | 0.410 | 0.000 |  |  |  |
| claude-haiku-4-5 | GRPO | PCT | matched_final | 5 | 5 | Cooperative | 32 | 0.962 | 0.955 | 0.007 | 0.097 | -0.018 | 0.032 | 0.396 | 0.396 |  |  |
| claude-haiku-4-5 | GRPO | PCT | matched_final | 5 | 5 | Warms up | 32 | 0.523 | 0.662 | -0.139 | -0.786 | -0.198 | -0.079 | 0.000 | 0.001 |  |  |
| claude-haiku-4-5 | GRPO | PCT | matched_final | 5 | 5 | Resistant | 32 | 0.179 | 0.250 | -0.070 | -0.306 | -0.148 | 0.007 | 0.108 | 0.217 |  |  |
| claude-haiku-4-5 | GRPO | PCT | matched_final | 5 | 5 | All | 96 | 0.555 | 0.622 | -0.067 | -0.373 | -0.105 | -0.031 | 0.001 |  |  |  |
| claude-haiku-4-5 | GRPO | PCT | own_best | 8 | 4 | Cooperative | 32 | 0.992 | 0.964 | 0.029 | 0.466 | 0.009 | 0.049 | 0.022 | 0.045 |  |  |
| claude-haiku-4-5 | GRPO | PCT | own_best | 8 | 4 | Warms up | 32 | 0.521 | 0.638 | -0.117 | -0.518 | -0.192 | -0.043 | 0.002 | 0.005 |  |  |
| claude-haiku-4-5 | GRPO | PCT | own_best | 8 | 4 | Resistant | 32 | 0.275 | 0.351 | -0.076 | -0.262 | -0.173 | 0.026 | 0.071 | 0.071 |  |  |
| claude-haiku-4-5 | GRPO | PCT | own_best | 8 | 4 | All | 96 | 0.596 | 0.651 | -0.055 | -0.247 | -0.097 | -0.011 | 0.006 |  |  |  |
