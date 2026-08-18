**Paired K=0 − K=5 on Q2, by iteration, both graders.** Cell = mean_delta (dz) + Holm stars. Sign: + = K=0 higher (K=0 minus K=5). Paired on persona_id (96 personas). p_holm = Holm across iterations 0..N within each (judge, method, metric); iteration 0 = two independent base draws (noise floor). GRPO_LA5 is right-censored at iteration 5, so GRPO rows stop at 5.

| iteration | PTO · gpt-4o-mini | PTO · claude-haiku-4-5 | GRPO · gpt-4o-mini | GRPO · claude-haiku-4-5 |
|---|---|---|---|---|
| 0 | +0.017 (+0.02) | +0.002 (+0.00) | +0.097 (+0.10) | +0.040 (+0.07) |
| 1 | +0.097 (+0.10) | +0.068 (+0.11) | -0.009 (-0.01) | -0.030 (-0.05) |
| 2 | +0.173 (+0.20) | +0.099 (+0.16) | -0.093 (-0.09) | -0.070 (-0.10) |
| 3 | +0.181 (+0.26) | +0.119 (+0.21) | +0.134 (+0.28) | +0.047 (+0.12) |
| 4 | +0.175 (+0.29) | +0.127 (+0.23) | -0.029 (-0.07) | -0.097 (-0.17) |
| 5 | +0.052 (+0.10) | +0.236 (+0.43)** | -0.021 (-0.04) | -0.172 (-0.26) |
| 6 | +0.341 (+0.52)*** | +0.373 (+0.58)*** | — | — |
| 7 | +0.137 (+0.26) | +0.206 (+0.33)** | — | — |
| 8 | +0.145 (+0.33)* | +0.307 (+0.54)*** | — | — |
| 9 | +0.095 (+0.18) | +0.329 (+0.57)*** | — | — |
| 10 | -0.009 (-0.02) | +0.363 (+0.65)*** | — | — |
