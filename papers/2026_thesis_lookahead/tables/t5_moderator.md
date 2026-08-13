<!-- Look-ahead benefit vs how well the MYOPIC arm learns. la_benefit is oriented so POSITIVE = look-ahead helped. -->

| generation     | arm        | therapist            | patient                  |   n_iters |   base |   myopic_end |   myopic_gain |   myopic_slope |   la_benefit |
|:---------------|:-----------|:---------------------|:-------------------------|----------:|-------:|-------------:|--------------:|---------------:|-------------:|
| Exp1 (ICLR'25) | PTO / Q1Q2 | Llama-2-7B           | GPT-3.5 (cooperative)    |         7 |  3.865 |        3.887 |         0.023 |          0.011 |        0.132 |
| Exp2           | PTO / Q1Q2 | Llama-3.2-1B (4-bit) | gpt-4o-mini (less coop.) |         5 |  2.378 |        2.770 |         0.393 |          0.055 |        0.060 |
| Exp2           | PTO / WAI  | Llama-3.2-1B (4-bit) | gpt-4o-mini (less coop.) |         5 |  2.378 |        2.596 |         0.219 |          0.009 |       -0.020 |
| Exp2           | PTO / CSQ8 | Llama-3.2-1B (4-bit) | gpt-4o-mini (less coop.) |         5 |  2.378 |        2.629 |         0.251 |          0.072 |        0.018 |
| Exp3           | PTO / Q1Q2 | Llama-3.2-1B (bf16)  | gpt-4o-mini (less coop.) |         8 |  3.008 |        4.221 |         1.212 |          0.131 |       -0.110 |
