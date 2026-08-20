|   K | comparison                               | a                | b                |   cosine |   ceiling |   cosine_corrected | read                            |
|----:|:-----------------------------------------|:-----------------|:-----------------|---------:|----------:|-------------------:|:--------------------------------|
|   0 | as trained (rule AND data differ)        | PTO_LA0(native)  | GRPO_LA0(native) |    0.267 |     0.844 |              0.317 | cosine_corrected                |
|   0 | same data (PTO_LA0), rule swapped        | PTO_LA0(native)  | PTO_LA0(grpo)    |    0.908 |     0.793 |              1.145 | cosine (same groups — see note) |
|   0 | same data (GRPO_LA0), rule swapped       | GRPO_LA0(native) | GRPO_LA0(dpo)    |    0.988 |     0.929 |              1.064 | cosine (same groups — see note) |
|   0 | same rule (group-relative), data differs | PTO_LA0(grpo)    | GRPO_LA0(native) |    0.356 |     0.896 |              0.397 | cosine_corrected                |
|   0 | same rule (best-vs-worst), data differs  | PTO_LA0(native)  | GRPO_LA0(dpo)    |    0.266 |     0.823 |              0.324 | cosine_corrected                |
|   5 | as trained (rule AND data differ)        | PTO_LA5(native)  | GRPO_LA5(native) |    0.670 |     0.894 |              0.750 | cosine_corrected                |
|   5 | same data (PTO_LA5), rule swapped        | PTO_LA5(native)  | PTO_LA5(grpo)    |    0.961 |     0.859 |              1.119 | cosine (same groups — see note) |
|   5 | same data (GRPO_LA5), rule swapped       | GRPO_LA5(native) | GRPO_LA5(dpo)    |    0.984 |     0.941 |              1.046 | cosine (same groups — see note) |
|   5 | same rule (group-relative), data differs | PTO_LA5(grpo)    | GRPO_LA5(native) |    0.678 |     0.920 |              0.737 | cosine_corrected                |
|   5 | same rule (best-vs-worst), data differs  | PTO_LA5(native)  | GRPO_LA5(dpo)    |    0.646 |     0.878 |              0.736 | cosine_corrected                |