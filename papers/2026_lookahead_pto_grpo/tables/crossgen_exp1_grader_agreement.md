Agreement between the two graders on Exp1's conversations: Spearman rho (and Pearson r) between the gpt-4o-mini re-score and the original GPT-3.5 oracle, at the level of the 15 model-state means (Base + 7 K=0 + 7 K=5), the 14 trained-model means, and per conversation (15 x 96 pooled). Metric Final = mean(Q1,Q2).

| metric | level | spearman_rho | p | pearson_r | n |
|---|---|---|---|---|---|
| Final | 15 model means | 0.836 | 0.000 | 0.866 | 15 |
| Final | 14 trained model means | 0.798 | 0.001 | 0.843 | 14 |
| Final | per conversation (pooled) | 0.674 | 0.000 | 0.798 | 1440 |
| Q1 | 15 model means | 0.895 | 0.000 | 0.882 | 15 |
| Q1 | 14 trained model means | 0.876 | 0.000 | 0.867 | 14 |
| Q1 | per conversation (pooled) | 0.659 | 0.000 | 0.767 | 1440 |
| Q2 | 15 model means | 0.818 | 0.000 | 0.800 | 15 |
| Q2 | 14 trained model means | 0.780 | 0.001 | 0.763 | 14 |
| Q2 | per conversation (pooled) | 0.563 | 0.000 | 0.694 | 1440 |
