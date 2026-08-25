| judge            | method   | cut           |   n_bins |   bins_K5_more_faithful |   bins_K0_more_faithful |   mean_delta |   median_delta |   wilcoxon_W |   wilcoxon_p |
|:-----------------|:---------|:--------------|---------:|------------------------:|------------------------:|-------------:|---------------:|-------------:|-------------:|
| gpt-4o-mini      | PTO      | train_iter_1  |       19 |                      10 |                       9 |      -0.0017 |        -0.0008 |      86.0000 |       0.7381 |
| gpt-4o-mini      | PTO      | iters_1-5     |       19 |                      19 |                       0 |      -0.0352 |        -0.0277 |       0.0000 |       0.0000 |
| gpt-4o-mini      | PTO      | matched_iters |       19 |                      19 |                       0 |      -0.0457 |        -0.0461 |       0.0000 |       0.0000 |
| gpt-4o-mini      | GRPO     | train_iter_1  |       20 |                       3 |                      17 |       0.0170 |         0.0145 |      14.0000 |       0.0002 |
| gpt-4o-mini      | GRPO     | iters_1-5     |       20 |                       3 |                      17 |       0.0107 |         0.0098 |       6.0000 |       0.0000 |
| gpt-4o-mini      | GRPO     | matched_iters |       20 |                      20 |                       0 |      -0.0465 |        -0.0483 |       0.0000 |       0.0000 |
| claude-haiku-4-5 | PTO      | train_iter_1  |       19 |                      13 |                       6 |      -0.0097 |        -0.0087 |      53.0000 |       0.0955 |
| claude-haiku-4-5 | PTO      | iters_1-5     |       19 |                      15 |                       4 |      -0.0327 |        -0.0245 |      26.0000 |       0.0039 |
| claude-haiku-4-5 | PTO      | matched_iters |       19 |                      18 |                       1 |      -0.0383 |        -0.0443 |       1.0000 |       0.0000 |
| claude-haiku-4-5 | GRPO     | train_iter_1  |       20 |                      17 |                       3 |      -0.0390 |        -0.0416 |       8.0000 |       0.0000 |
| claude-haiku-4-5 | GRPO     | iters_1-5     |       20 |                      20 |                       0 |      -0.0356 |        -0.0275 |       0.0000 |       0.0000 |
| claude-haiku-4-5 | GRPO     | matched_iters |       20 |                      20 |                       0 |      -0.0643 |        -0.0685 |       0.0000 |       0.0000 |