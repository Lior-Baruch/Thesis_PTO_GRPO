<!-- THE HEADLINE. All three generations under ONE grader (gpt-4o-mini + V5 Q1/Q2). Delta = K0-K5, so negative favours look-ahead. -->

| generation                      |   matched_contrasts | k5_ahead   |   mean_delta |   mean_dz |   holm_sig_k5 |   holm_sig_k0 | therapist          | patient     | verdict                |
|:--------------------------------|--------------------:|:-----------|-------------:|----------:|--------------:|--------------:|:-------------------|:------------|:-----------------------|
| Exp1 (ICLR'25) — Llama-2-7B     |                   7 | 7/7        |       -0.132 |    -0.180 |             2 |             0 | Llama-2-7B         | GPT-3.5     | look-ahead HELPS       |
| Exp2 — Llama-3.2-1B (4-bit)     |                  15 | 9/15       |       -0.019 |    -0.019 |             0 |             0 | Llama-3.2-1B 4-bit | gpt-4o-mini | NULL                   |
| Exp3 — Llama-3.2-1B (bf16), PTO |                   8 | 1/8        |        0.110 |     0.166 |             0 |             1 | Llama-3.2-1B bf16  | gpt-4o-mini | look-ahead NEVER LEADS |
