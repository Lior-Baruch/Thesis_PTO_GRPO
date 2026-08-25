Exp1's K=3 sweep (LookAhead_3, 4 iterations, 96 conversations each) — the only look-ahead 'dose' data on disk. NOT re-scored by gpt-4o-mini: score_crossgen.py deliberately excludes it (its manifest is the paper's K=0/K=5 pair only, and K=3 ran with different hyper-parameters — therapist temperature 0.7 and filter tau 0.2 vs 0.9 / 0.1 — so it is not a matched dose arm). Shown here are its ORIGINAL GPT-3.5 oracle means from the on-disk scores_i.csv (Q1, Q2, Final = mean). Not comparable to the K=0/K=5 rows of crossgen_exp1_levels.md without that caveat.

| model | iteration | Q1 | Q2 | Final | n_gpt35 | scored_by_gpt4omini | hyperparams |
|---|---|---|---|---|---|---|---|
| Exp1_LA3_I1 | 1 | 3.175 | 3.194 | 3.185 | 96 | False | TT0.7 / Filter(tau)0.2 / 'FullEval'  (K=0,5 sweep: TT0.9 / tau 0.1) |
| Exp1_LA3_I2 | 2 | 3.331 | 3.388 | 3.360 | 96 | False | TT0.7 / Filter(tau)0.2 / 'FullEval'  (K=0,5 sweep: TT0.9 / tau 0.1) |
| Exp1_LA3_I3 | 3 | 3.704 | 3.581 | 3.642 | 96 | False | TT0.7 / Filter(tau)0.2 / 'FullEval'  (K=0,5 sweep: TT0.9 / tau 0.1) |
| Exp1_LA3_I4 | 4 | 3.746 | 3.525 | 3.635 | 96 | False | TT0.7 / Filter(tau)0.2 / 'FullEval'  (K=0,5 sweep: TT0.9 / tau 0.1) |
