The per-iteration price of look-ahead. GRPO: median optimizer-step seconds K=0 vs K=5 and their ratio (eda_analysis.compute.step_multiplier; the K=5 reward computation runs 5 extra simulated turns per candidate INSIDE the training loop). GRPO_LA5 has no rows past iteration 5 (right-censored). PTO: the DPO step carries no look-ahead (ratio ~1); PTO's look-ahead cost lands in the pref-tree BUILD phase, so its build_h ratio and whole-iteration gpu_h ratio are shown instead. Iteration 1 of GRPO_LA5 ran at LOOKAHEAD_SUB_BATCH_SIZE=64 with a fat API-latency tail (ratio 2.41), so quote the settled iterations 3-5 (~1.9x).

| iteration | GRPO_median_step_s_K0 | GRPO_median_step_s_K5 | GRPO_step_ratio_K5_over_K0 | PTO_dpo_median_step_s_K0 | PTO_dpo_median_step_s_K5 | PTO_dpo_step_ratio | PTO_build_h_K0 | PTO_build_h_K5 | PTO_build_ratio_K5_over_K0 | PTO_iter_gpu_h_K0 | PTO_iter_gpu_h_K5 | PTO_iter_ratio_K5_over_K0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.000 | 74.605 | 179.536 | 2.406 | 6.459 | 6.479 | 1.003 | 0.682 | 1.989 | 2.917 | 0.919 | 2.172 | 2.363 |
| 2.000 | 80.020 | 169.586 | 2.119 | 6.535 | 6.638 | 1.016 | 0.704 | 2.290 | 3.253 | 1.013 | 2.470 | 2.438 |
| 3.000 | 79.186 | 155.635 | 1.965 | 6.523 | 6.753 | 1.035 | 0.583 | 2.429 | 4.168 | 0.870 | 2.609 | 3.000 |
| 4.000 | 79.409 | 155.788 | 1.962 | 6.592 | 6.812 | 1.033 | 0.857 | 1.555 | 1.814 | 1.103 | 1.688 | 1.530 |
| 5.000 | 78.618 | 150.217 | 1.911 | 6.686 | 6.826 | 1.021 | 0.556 | 0.964 | 1.735 | 0.758 | 1.066 | 1.407 |
| 6.000 | 78.049 |  |  | 6.828 | 6.962 | 1.020 | 0.489 | 1.596 | 3.264 | 0.704 | 2.699 | 3.831 |
| 7.000 | 79.323 |  |  | 6.958 | 7.039 | 1.012 | 0.457 | 1.636 | 3.581 | 0.706 | 1.896 | 2.686 |
| 8.000 | 77.678 |  |  | 6.934 | 7.088 | 1.022 | 0.464 | 1.330 | 2.869 | 0.807 | 1.572 | 1.947 |
| 9.000 | 79.393 |  |  | 7.016 | 7.063 | 1.007 | 0.419 | 1.596 | 3.806 | 0.607 | 1.855 | 3.054 |
| 10.000 | 77.955 |  |  | 7.054 | 7.092 | 1.005 | 0.460 | 1.413 | 3.074 | 0.632 | 1.655 | 2.620 |
