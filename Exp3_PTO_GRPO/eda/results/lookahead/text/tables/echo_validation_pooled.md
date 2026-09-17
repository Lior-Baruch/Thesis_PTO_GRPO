| proxy           | method   |   n_states |   rho_vs_reflections |   rho_vs_complex_reflections |   rho_vs_questions | judge            |
|:----------------|:---------|-----------:|---------------------:|-----------------------------:|-------------------:|:-----------------|
| echo            | GRPO     |         22 |               -0.046 |                        0.009 |              0.047 | claude-haiku-4-5 |
| echo            | PTO      |         22 |               -0.009 |                        0.032 |             -0.077 | claude-haiku-4-5 |
| lex_recall_prev | GRPO     |         22 |               -0.094 |                       -0.077 |             -0.202 | claude-haiku-4-5 |
| lex_recall_prev | PTO      |         22 |                0.006 |                        0.013 |             -0.284 | claude-haiku-4-5 |
| echo            | GRPO     |         22 |               -0.008 |                        0.058 |              0.046 | gpt-4o-mini      |
| echo            | PTO      |         22 |                0.018 |                        0.037 |             -0.001 | gpt-4o-mini      |
| lex_recall_prev | GRPO     |         22 |               -0.061 |                       -0.032 |             -0.012 | gpt-4o-mini      |
| lex_recall_prev | PTO      |         22 |               -0.002 |                        0.007 |             -0.077 | gpt-4o-mini      |