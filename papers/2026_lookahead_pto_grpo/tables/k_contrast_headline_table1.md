**Table 1 — paired K=0 − K=5 on the training reward Q1+Q2, by iteration, under both graders.** Cell = mean_delta (Cohen's dz) with Holm stars (* <.05, ** <.01, *** <.001; p_holm = Holm across iterations 0..N within each (judge, method, metric); iteration 0 = two independent base draws (noise floor).). Sign: + = K=0 higher (K=0 minus K=5). Paired on persona_id (96 personas). GRPO_LA5 is right-censored at iteration 5, so GRPO rows stop at 5. '—' = no matched K=5 model state.

| iteration | PTO · gpt-4o-mini | PTO · claude-haiku-4-5 | GRPO · gpt-4o-mini | GRPO · claude-haiku-4-5 |
|---|---|---|---|---|
| 0 | -0.003 (-0.00) | -0.004 (-0.01) | +0.104 (+0.11) | +0.026 (+0.04) |
| 1 | +0.083 (+0.09) | +0.060 (+0.09) | -0.003 (-0.00) | +0.028 (+0.04) |
| 2 | +0.158 (+0.18) | +0.136 (+0.20) | -0.076 (-0.08) | -0.035 (-0.05) |
| 3 | +0.141 (+0.20) | +0.130 (+0.20) | +0.132 (+0.28) | +0.051 (+0.11) |
| 4 | +0.120 (+0.20) | +0.123 (+0.21) | -0.115 (-0.25)* | -0.233 (-0.37)** |
| 5 | -0.002 (-0.00) | +0.173 (+0.33)* | -0.070 (-0.13) | -0.311 (-0.43)** |
| 6 | +0.257 (+0.42)*** | +0.343 (+0.51)*** | — | — |
| 7 | +0.044 (+0.07) | +0.078 (+0.12) | — | — |
| 8 | +0.077 (+0.17) | +0.186 (+0.34)** | — | — |
| 9 | +0.041 (+0.08) | +0.187 (+0.29) | — | — |
| 10 | -0.047 (-0.10) | +0.199 (+0.31) | — | — |
