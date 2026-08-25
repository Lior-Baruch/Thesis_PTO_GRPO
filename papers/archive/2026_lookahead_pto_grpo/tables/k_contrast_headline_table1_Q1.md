**Paired K=0 − K=5 on Q1, by iteration, both graders.** Cell = mean_delta (dz) + Holm stars. Sign: + = K=0 higher (K=0 minus K=5). Paired on persona_id (96 personas). p_holm = Holm across iterations 0..N within each (judge, method, metric); iteration 0 = two independent base draws (noise floor). GRPO_LA5 is right-censored at iteration 5, so GRPO rows stop at 5.

| iteration | PTO · gpt-4o-mini | PTO · claude-haiku-4-5 | GRPO · gpt-4o-mini | GRPO · claude-haiku-4-5 |
|---|---|---|---|---|
| 0 | -0.023 (-0.02) | -0.010 (-0.01) | +0.110 (+0.12) | +0.013 (+0.02) |
| 1 | +0.069 (+0.08) | +0.052 (+0.06) | +0.004 (+0.00) | +0.085 (+0.11) |
| 2 | +0.144 (+0.16) | +0.173 (+0.20) | -0.058 (-0.06) | -0.000 (-0.00) |
| 3 | +0.100 (+0.13) | +0.142 (+0.16) | +0.129 (+0.23) | +0.054 (+0.08) |
| 4 | +0.065 (+0.10) | +0.119 (+0.16) | -0.202 (-0.36)** | -0.369 (-0.45)*** |
| 5 | -0.056 (-0.11) | +0.110 (+0.17) | -0.119 (-0.21) | -0.450 (-0.50)*** |
| 6 | +0.173 (+0.27) | +0.313 (+0.38)** | — | — |
| 7 | -0.050 (-0.07) | -0.050 (-0.06) | — | — |
| 8 | +0.008 (+0.01) | +0.065 (+0.10) | — | — |
| 9 | -0.013 (-0.02) | +0.046 (+0.05) | — | — |
| 10 | -0.085 (-0.14) | +0.035 (+0.04) | — | — |
