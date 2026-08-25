GPU-hours per (arm, iteration), reconstructed from artifact mtimes by eda_analysis.compute.iteration_compute (gap_cutoff 3600 s; deltas outside (0, 3600 s) imputed at the phase median, n_imputed counts them). gen = rollout pass that produced model_iter_{k-1}; build = PTO pref-tree branching + oracle (PTO only); train = optimizer loop (GRPO: completions parquet mtimes; PTO: TensorBoard wall_time). Iteration 0 = the base policy (0 h by construction). cum_gpu_h = cost of having produced <Arm>_I{k} (headline = the tracked EDA numbers). gen_h_meta = iteration_metadata.json generation_time_s/3600 (per-PROCESS: a reloaded/resumed pass records seconds); gen_h_floor = max(gen_h, gen_h_meta) is a FLOOR on generation time, because the mtime span starts at the first conversation write and so misses the first batch of 64 (~0.1 h) and collapses to ~0 when all CSVs flush together (PTO_LA5 iters 1-5, whose time lands in iter 6). cum_gpu_h_floor re-cumulates with it. GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h).

| arm | method | K | iteration | n_steps | median_step_s | n_imputed | gen_h | build_h | train_h | gpu_h | cum_gpu_h | train_source | gen_h_meta | gen_h_floor | cum_gpu_h_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRPO_LA0 | GRPO | 0 | 0 | 0 |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 |
| GRPO_LA0 | GRPO | 0 | 1 | 108 | 74.605 | 0 | 0.061 | 0.000 | 2.554 | 2.615 | 2.615 | completions | 0.213 | 0.213 | 2.767 |
| GRPO_LA0 | GRPO | 0 | 2 | 94 | 80.020 | 1 | 0.241 | 0.000 | 2.392 | 2.632 | 5.247 | completions | 0.023 | 0.241 | 5.399 |
| GRPO_LA0 | GRPO | 0 | 3 | 118 | 79.186 | 0 | 0.077 | 0.000 | 2.888 | 2.965 | 8.212 | completions | 0.181 | 0.181 | 8.469 |
| GRPO_LA0 | GRPO | 0 | 4 | 100 | 79.409 | 0 | 0.109 | 0.000 | 2.425 | 2.534 | 10.746 | completions | 0.251 | 0.251 | 11.145 |
| GRPO_LA0 | GRPO | 0 | 5 | 118 | 78.618 | 0 | 0.107 | 0.000 | 2.860 | 2.967 | 13.713 | completions | 0.216 | 0.216 | 14.220 |
| GRPO_LA0 | GRPO | 0 | 6 | 116 | 78.049 | 1 | 0.216 | 0.000 | 2.731 | 2.947 | 16.660 | completions | 0.023 | 0.216 | 17.167 |
| GRPO_LA0 | GRPO | 0 | 7 | 128 | 79.323 | 0 | 0.112 | 0.000 | 2.974 | 3.086 | 19.746 | completions | 0.231 | 0.231 | 20.373 |
| GRPO_LA0 | GRPO | 0 | 8 | 108 | 77.678 | 1 | 0.101 | 0.000 | 2.428 | 2.529 | 22.275 | completions | 0.017 | 0.101 | 22.901 |
| GRPO_LA0 | GRPO | 0 | 9 | 80 | 79.393 | 0 | 0.096 | 0.000 | 1.853 | 1.948 | 24.223 | completions | 0.214 | 0.214 | 24.967 |
| GRPO_LA0 | GRPO | 0 | 10 | 158 | 77.955 | 0 | 0.095 | 0.000 | 3.588 | 3.683 | 27.906 | completions | 0.211 | 0.211 | 28.766 |
| GRPO_LA5 | GRPO | 5 | 0 | 0 |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 |
| GRPO_LA5 | GRPO | 5 | 1 | 108 | 179.536 | 1 | 0.060 | 0.000 | 7.742 | 7.801 | 7.801 | completions | 0.012 | 0.060 | 7.801 |
| GRPO_LA5 | GRPO | 5 | 2 | 104 | 169.586 | 1 | 0.093 | 0.000 | 5.378 | 5.471 | 13.272 | completions | 0.025 | 0.093 | 13.272 |
| GRPO_LA5 | GRPO | 5 | 3 | 112 | 155.635 | 0 | 0.080 | 0.000 | 4.960 | 5.040 | 18.312 | completions | 0.194 | 0.194 | 18.426 |
| GRPO_LA5 | GRPO | 5 | 4 | 106 | 155.788 | 0 | 0.088 | 0.000 | 4.810 | 4.898 | 23.210 | completions | 0.200 | 0.200 | 23.436 |
| GRPO_LA5 | GRPO | 5 | 5 | 88 | 150.217 | 0 | 0.101 | 0.000 | 3.767 | 3.868 | 27.078 | completions | 0.212 | 0.212 | 27.415 |
| PTO_LA0 | PTO | 0 | 0 | 0 |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 |
| PTO_LA0 | PTO | 0 | 1 | 99 | 6.459 | 0 | 0.065 | 0.682 | 0.172 | 0.919 | 0.919 | tb_wall_time | 0.023 | 0.065 | 0.919 |
| PTO_LA0 | PTO | 0 | 2 | 79 | 6.535 | 0 | 0.173 | 0.704 | 0.136 | 1.013 | 1.932 | tb_wall_time | 0.276 | 0.276 | 2.035 |
| PTO_LA0 | PTO | 0 | 3 | 77 | 6.523 | 0 | 0.151 | 0.583 | 0.136 | 0.870 | 2.801 | tb_wall_time | 0.448 | 0.448 | 3.202 |
| PTO_LA0 | PTO | 0 | 4 | 67 | 6.592 | 1 | 0.127 | 0.857 | 0.119 | 1.103 | 3.904 | tb_wall_time | 0.000 | 0.127 | 4.305 |
| PTO_LA0 | PTO | 0 | 5 | 61 | 6.686 | 0 | 0.091 | 0.556 | 0.111 | 0.758 | 4.662 | tb_wall_time | 0.202 | 0.202 | 5.174 |
| PTO_LA0 | PTO | 0 | 6 | 59 | 6.828 | 0 | 0.106 | 0.489 | 0.110 | 0.704 | 5.367 | tb_wall_time | 0.242 | 0.242 | 6.014 |
| PTO_LA0 | PTO | 0 | 7 | 53 | 6.958 | 0 | 0.152 | 0.457 | 0.097 | 0.706 | 6.072 | tb_wall_time | 0.281 | 0.281 | 6.848 |
| PTO_LA0 | PTO | 0 | 8 | 49 | 6.934 | 0 | 0.253 | 0.464 | 0.091 | 0.807 | 6.880 | tb_wall_time | 0.376 | 0.376 | 7.778 |
| PTO_LA0 | PTO | 0 | 9 | 47 | 7.016 | 0 | 0.101 | 0.419 | 0.087 | 0.607 | 7.487 | tb_wall_time | 0.202 | 0.202 | 8.486 |
| PTO_LA0 | PTO | 0 | 10 | 39 | 7.054 | 0 | 0.103 | 0.460 | 0.069 | 0.632 | 8.119 | tb_wall_time | 0.206 | 0.206 | 9.221 |
| PTO_LA5 | PTO | 5 | 0 | 0 |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 |
| PTO_LA5 | PTO | 5 | 1 | 101 | 6.479 | 0 | 0.000 | 1.989 | 0.182 | 2.172 | 2.172 | tb_wall_time | 0.297 | 0.297 | 2.468 |
| PTO_LA5 | PTO | 5 | 2 | 97 | 6.638 | 0 | 0.000 | 2.290 | 0.179 | 2.470 | 4.641 | tb_wall_time | 0.202 | 0.202 | 5.140 |
| PTO_LA5 | PTO | 5 | 3 | 95 | 6.753 | 0 | 0.000 | 2.429 | 0.180 | 2.609 | 7.250 | tb_wall_time | 0.259 | 0.259 | 8.008 |
| PTO_LA5 | PTO | 5 | 4 | 71 | 6.812 | 0 | 0.000 | 1.555 | 0.132 | 1.688 | 8.938 | tb_wall_time | 0.155 | 0.155 | 9.850 |
| PTO_LA5 | PTO | 5 | 5 | 55 | 6.826 | 0 | 0.000 | 0.964 | 0.102 | 1.066 | 10.004 | tb_wall_time | 0.132 | 0.132 | 11.049 |
| PTO_LA5 | PTO | 5 | 6 | 73 | 6.962 | 1 | 0.967 | 1.596 | 0.136 | 2.699 | 12.703 | tb_wall_time | 0.012 | 0.967 | 13.747 |
| PTO_LA5 | PTO | 5 | 7 | 87 | 7.039 | 0 | 0.092 | 1.636 | 0.167 | 1.896 | 14.599 | tb_wall_time | 0.221 | 0.221 | 15.772 |
| PTO_LA5 | PTO | 5 | 8 | 71 | 7.088 | 0 | 0.105 | 1.330 | 0.137 | 1.572 | 16.171 | tb_wall_time | 0.224 | 0.224 | 17.463 |
| PTO_LA5 | PTO | 5 | 9 | 75 | 7.063 | 1 | 0.114 | 1.596 | 0.146 | 1.855 | 18.026 | tb_wall_time | 0.011 | 0.114 | 19.318 |
| PTO_LA5 | PTO | 5 | 10 | 77 | 7.092 | 0 | 0.091 | 1.413 | 0.151 | 1.655 | 19.681 | tb_wall_time | 0.200 | 0.200 | 21.083 |
