One row per arm: iterations trained, phase GPU-hours and total, cost per iteration (eda_analysis.compute.compute_summary). build_share/train_share = phase / total. total_gpu_h_floor uses gen_h_floor = max(mtime span, recorded generation_time_s) per iteration (see by_iteration: the mtime span misses the first batch, ~0.1 h/iter, and is ~0 for PTO_LA5 iters 1-5); the headline total_gpu_h is the tracked EDA number. GRPO_LA5 is right-censored at iteration 5 (its budget stops at 27.08 GPU-h).

| arm | method | K | last_iter | n_iters | gen_h | build_h | train_h | total_gpu_h | median_step_s | n_imputed | train_source | gpu_h_per_iter | total_gpu_h_floor | build_share | train_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRPO_LA0 | GRPO | 0 | 10 | 10 | 1.214 | 0.000 | 26.692 | 27.906 | 78.902 | 3 | completions | 2.791 | 28.766 | 0.000 | 0.957 |
| GRPO_LA5 | GRPO | 5 | 5 | 5 | 0.422 | 0.000 | 26.656 | 27.078 | 155.788 | 2 | completions | 5.416 | 27.415 | 0.000 | 0.984 |
| PTO_LA0 | PTO | 0 | 10 | 10 | 1.323 | 5.669 | 1.127 | 8.119 | 6.757 | 1 | tb_wall_time | 0.812 | 9.221 | 0.698 | 0.139 |
| PTO_LA5 | PTO | 5 | 10 | 10 | 1.370 | 16.797 | 1.514 | 19.681 | 6.894 | 2 | tb_wall_time | 1.968 | 21.083 | 0.853 | 0.077 |
