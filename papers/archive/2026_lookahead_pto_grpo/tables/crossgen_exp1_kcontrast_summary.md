Summary contrasts on Exp1's conversations under each grader (never averaged across graders). Best-vs-best rows: `mean_delta` = first-named model minus second (K=0 minus K=5); **+ => K=0 higher**; paired on conversation index (= persona), n = 96, dz / 95% bootstrap CI / Wilcoxon p (uncorrected: one planned contrast per row). 'ICLR best-vs-best' repeats the poster's L0_M4 vs L5_M7 comparison; 'own best-vs-best' picks each arm's best iteration under THAT grader. p_t = paired t; n_K0_higher / n_K5_higher = sign split of the 96 paired deltas (ties excluded); for the 'ordering' row they instead count, over the 49 (K=0 model, K=5 model) pairs of MEANS, how many have the K=0 / the K=5 model higher. 'mean over iters 1-7' averages each persona's score over an arm's 7 iterations first, then contrasts the arms (the arm-level 'K=5 models score higher' claim). The 'ordering' row is min(K=5 model mean) - max(K=0 model mean): + means every K=5 model outscores every K=0 model.

| grader | contrast | metric | n | mean_delta | dz | ci_lo | ci_hi | p | p_t | n_K0_higher | n_K5_higher | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | ICLR best-vs-best: L0_I4 - L5_I7 | Final | 96 | -0.129 | -0.250 | -0.228 | -0.029 | 0.006 | 0.016 | 35 | 54 | + => first-named (K=0) higher |
| gpt-4o-mini | ICLR best-vs-best: L0_I4 - L5_I7 | Q1 | 96 | -0.144 | -0.256 | -0.252 | -0.029 | 0.001 | 0.014 | 18 | 39 | + => first-named (K=0) higher |
| gpt-4o-mini | ICLR best-vs-best: L0_I4 - L5_I7 | Q2 | 96 | -0.115 | -0.205 | -0.226 | -0.002 | 0.050 | 0.047 | 35 | 48 | + => first-named (K=0) higher |
| gpt-4o-mini | own best-vs-best: LA0_I5 - LA5_I7 | Final | 96 | -0.129 | -0.191 | -0.266 | 0.002 | 0.405 | 0.065 | 45 | 48 | + => first-named (K=0) higher |
| gpt-4o-mini | own best-vs-best: LA0_I5 - LA5_I7 | Q1 | 96 | -0.163 | -0.231 | -0.308 | -0.025 | 0.058 | 0.026 | 26 | 31 | + => first-named (K=0) higher |
| gpt-4o-mini | own best-vs-best: LA0_I5 - LA5_I7 | Q2 | 96 | -0.096 | -0.131 | -0.243 | 0.048 | 0.329 | 0.202 | 42 | 46 | + => first-named (K=0) higher |
| gpt-4o-mini | mean over iters 1-7, K0 - K5 | Final | 96 | -0.132 | -0.543 | -0.180 | -0.084 | 0.000 | 0.000 | 31 | 65 | + => first-named (K=0) higher |
| gpt-4o-mini | mean over iters 1-7, K0 - K5 | Q1 | 96 | -0.150 | -0.572 | -0.202 | -0.099 | 0.000 | 0.000 | 28 | 65 | + => first-named (K=0) higher |
| gpt-4o-mini | mean over iters 1-7, K0 - K5 | Q2 | 96 | -0.114 | -0.455 | -0.163 | -0.064 | 0.000 | 0.000 | 30 | 66 | + => first-named (K=0) higher |
| gpt-4o-mini | ordering: min(L5 mean) - max(L0 mean) | Final | 7 | -0.080 |  |  |  |  |  | 5 | 44 | L5-minus-L0 of MEANS: + => every K=5 model above every K=0 model |
| gpt-3.5 | ICLR best-vs-best: L0_I4 - L5_I7 | Final | 96 | -0.206 | -0.251 | -0.379 | -0.049 | 0.331 | 0.016 | 46 | 49 | + => first-named (K=0) higher |
| gpt-3.5 | ICLR best-vs-best: L0_I4 - L5_I7 | Q1 | 96 | -0.221 | -0.199 | -0.452 | -0.008 | 0.337 | 0.054 | 38 | 46 | + => first-named (K=0) higher |
| gpt-3.5 | ICLR best-vs-best: L0_I4 - L5_I7 | Q2 | 96 | -0.191 | -0.280 | -0.329 | -0.065 | 0.023 | 0.007 | 38 | 51 | + => first-named (K=0) higher |
| gpt-3.5 | own best-vs-best: LA0_I4 - LA5_I7 | Final | 96 | -0.206 | -0.251 | -0.379 | -0.049 | 0.331 | 0.016 | 46 | 49 | + => first-named (K=0) higher |
| gpt-3.5 | own best-vs-best: LA0_I4 - LA5_I7 | Q1 | 96 | -0.221 | -0.199 | -0.452 | -0.008 | 0.337 | 0.054 | 38 | 46 | + => first-named (K=0) higher |
| gpt-3.5 | own best-vs-best: LA0_I4 - LA5_I7 | Q2 | 96 | -0.191 | -0.280 | -0.329 | -0.065 | 0.023 | 0.007 | 38 | 51 | + => first-named (K=0) higher |
| gpt-3.5 | mean over iters 1-7, K0 - K5 | Final | 96 | -0.206 | -0.612 | -0.271 | -0.140 | 0.000 | 0.000 | 28 | 68 | + => first-named (K=0) higher |
| gpt-3.5 | mean over iters 1-7, K0 - K5 | Q1 | 96 | -0.262 | -0.595 | -0.348 | -0.178 | 0.000 | 0.000 | 29 | 65 | + => first-named (K=0) higher |
| gpt-3.5 | mean over iters 1-7, K0 - K5 | Q2 | 96 | -0.151 | -0.514 | -0.208 | -0.094 | 0.000 | 0.000 | 31 | 65 | + => first-named (K=0) higher |
| gpt-3.5 | ordering: min(L5 mean) - max(L0 mean) | Final | 7 | -0.066 |  |  |  |  |  | 1 | 48 | L5-minus-L0 of MEANS: + => every K=5 model above every K=0 model |
