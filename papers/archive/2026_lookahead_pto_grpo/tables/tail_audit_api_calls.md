API-call accounting per arm x training iteration n (row = the cost of iteration n: the 96 eval convs generated at its start by policy pi_{n-1} = model_iter_{n-1}, PLUS the training-time calls; the last row 'final eval pass' is the post-loop generate-only pass with no training). oracle_calls_train = Q1 + Q2 calls per scored candidate (2 per candidate, + recorded retries; GRPO's TRL eval-phase groups included, n_candidates_eval_phase), read from generations.jsonl and — for GRPO — rescaled to the ground-truth step count (n_steps = training/completions/*.parquet files, 128 candidates = 16 groups x G=8 per step; log_coverage = logged / expected groups, < 1 where a crashed iteration's pre-resume records were lost: GRPO_LA5 iters 1-2, GRPO_LA0 iters 2, 6, 8); oracle_input_Mchars = the chars the oracle read (prefix + completion + tail, x2 rubrics) as a token proxy. eval_scoring_calls_run_eval = 96 x 8 instruments per model state (Run_Eval, identical for every arm; per grader). patient_calls_eval_convs = patient turns in the model_iter_{n-1} CSVs; patient_calls_trunk = PTO greedy trunk replies (<= 1 per branch point, upper bound within 96); patient_calls_tail = realized patient turns inside K=5 tails (ceil(realized_turns/2); 1 for a zero-turn tail whose first patient call was made). therapist_gens_tail = therapist turns generated inside tails (GPU, not API). GRPO_LA5 is right-censored at iteration 5.

| arm | method | K | train_iter | row_kind | eval_convs_of | n_eval_convs | eval_conv_len_mean | eval_ended_by_patient | n_steps | n_groups_logged | log_coverage | n_groups | n_candidates | n_candidates_eval_phase | oracle_calls_train | oracle_retries | oracle_input_Mchars | eval_scoring_calls_run_eval | patient_calls_eval_convs | patient_calls_trunk | patient_calls_tail | patient_calls_total | therapist_gens_tail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PTO_LA0 | PTO | 0 | 1 | iteration | model_iter_0 | 96 | 28.39 | 69 |  | 949 | 1.00 | 949 | 7592 | 0 | 15158 | 10 | 125.19 | 768 | 1349 | 949 | 0 | 2298 | 0 |
| PTO_LA0 | PTO | 0 | 2 | iteration | model_iter_1 | 96 | 25.90 | 63 |  | 772 | 1.00 | 772 | 6176 | 0 | 12348 | 12 | 100.87 | 768 | 1237 | 772 | 0 | 2009 | 0 |
| PTO_LA0 | PTO | 0 | 3 | iteration | model_iter_2 | 96 | 27.69 | 73 |  | 787 | 1.00 | 787 | 6296 | 0 | 12558 | 4 | 106.20 | 768 | 1322 | 787 | 0 | 2109 | 0 |
| PTO_LA0 | PTO | 0 | 4 | iteration | model_iter_3 | 96 | 23.93 | 73 |  | 647 | 1.00 | 647 | 5176 | 0 | 10339 | 3 | 91.82 | 768 | 1144 | 647 | 0 | 1791 | 0 |
| PTO_LA0 | PTO | 0 | 5 | iteration | model_iter_4 | 96 | 23.85 | 77 |  | 600 | 1.00 | 600 | 4800 | 0 | 9588 | 4 | 92.14 | 768 | 1143 | 600 | 0 | 1743 | 0 |
| PTO_LA0 | PTO | 0 | 6 | iteration | model_iter_5 | 96 | 23.19 | 82 |  | 591 | 1.00 | 591 | 4728 | 0 | 9437 | 1 | 98.42 | 768 | 1112 | 591 | 0 | 1703 | 0 |
| PTO_LA0 | PTO | 0 | 7 | iteration | model_iter_6 | 96 | 21.72 | 83 |  | 521 | 1.00 | 521 | 4168 | 0 | 8320 | 4 | 102.16 | 768 | 1040 | 521 | 0 | 1561 | 0 |
| PTO_LA0 | PTO | 0 | 8 | iteration | model_iter_7 | 96 | 22.72 | 82 |  | 517 | 1.00 | 517 | 4136 | 0 | 8232 | 4 | 99.01 | 768 | 1088 | 517 | 0 | 1605 | 0 |
| PTO_LA0 | PTO | 0 | 9 | iteration | model_iter_8 | 96 | 20.77 | 88 |  | 446 | 1.00 | 446 | 3568 | 0 | 7114 | 2 | 90.23 | 768 | 995 | 446 | 0 | 1441 | 0 |
| PTO_LA0 | PTO | 0 | 10 | iteration | model_iter_9 | 96 | 19.20 | 92 |  | 410 | 1.00 | 410 | 3280 | 0 | 6528 | 6 | 81.04 | 768 | 917 | 410 | 0 | 1327 | 0 |
| PTO_LA0 | PTO | 0 | 11 | final eval pass | model_iter_10 | 96 | 20.39 | 89 |  | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0.00 | 768 | 975 | 0 | 0 | 975 | 0 |
| PTO_LA5 | PTO | 5 | 1 | iteration | model_iter_0 | 96 | 30.49 | 58 |  | 890 | 1.00 | 890 | 7120 | 0 | 14199 | 5 | 136.86 | 768 | 1458 | 890 | 19041 | 21389 | 12021 |
| PTO_LA5 | PTO | 5 | 2 | iteration | model_iter_1 | 96 | 26.32 | 66 |  | 837 | 1.00 | 837 | 6696 | 0 | 13385 | 15 | 144.84 | 768 | 1256 | 837 | 17999 | 20092 | 11377 |
| PTO_LA5 | PTO | 5 | 3 | iteration | model_iter_2 | 96 | 26.61 | 76 |  | 816 | 1.00 | 816 | 6528 | 0 | 13048 | 4 | 149.35 | 768 | 1269 | 816 | 17465 | 19550 | 11009 |
| PTO_LA5 | PTO | 5 | 4 | iteration | model_iter_3 | 96 | 25.49 | 72 |  | 613 | 1.00 | 613 | 4904 | 0 | 9800 | 2 | 115.00 | 768 | 1217 | 613 | 12725 | 14555 | 7856 |
| PTO_LA5 | PTO | 5 | 5 | iteration | model_iter_4 | 96 | 22.84 | 77 |  | 468 | 1.00 | 468 | 3744 | 0 | 8745 | 1654 | 76.18 | 768 | 1093 | 468 | 8967 | 10528 | 5433 |
| PTO_LA5 | PTO | 5 | 6 | iteration | model_iter_5 | 96 | 23.72 | 82 |  | 663 | 1.00 | 663 | 5304 | 0 | 10578 | 0 | 167.48 | 768 | 1135 | 663 | 13786 | 15584 | 8526 |
| PTO_LA5 | PTO | 5 | 7 | iteration | model_iter_6 | 96 | 25.26 | 81 |  | 807 | 1.00 | 807 | 6456 | 0 | 12888 | 0 | 226.44 | 768 | 1208 | 807 | 16910 | 18925 | 10524 |
| PTO_LA5 | PTO | 5 | 8 | iteration | model_iter_7 | 96 | 23.30 | 83 |  | 767 | 1.00 | 767 | 6136 | 0 | 12234 | 0 | 250.37 | 768 | 1116 | 767 | 16391 | 18274 | 10346 |
| PTO_LA5 | PTO | 5 | 9 | iteration | model_iter_8 | 96 | 25.65 | 83 |  | 761 | 1.00 | 761 | 6088 | 0 | 12155 | 1 | 248.03 | 768 | 1229 | 761 | 16355 | 18345 | 10336 |
| PTO_LA5 | PTO | 5 | 10 | iteration | model_iter_9 | 96 | 27.49 | 77 |  | 926 | 1.00 | 926 | 7408 | 0 | 14774 | 0 | 334.60 | 768 | 1317 | 926 | 20149 | 22392 | 12822 |
| PTO_LA5 | PTO | 5 | 11 | final eval pass | model_iter_10 | 96 | 28.70 | 71 |  | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0.00 | 768 | 1374 | 0 | 0 | 1374 | 0 |
| GRPO_LA0 | GRPO | 0 | 1 | iteration | model_iter_0 | 96 | 28.77 | 65 | 108.00 | 1728 | 1.00 | 1728 | 13824 | 528 | 28720 | 16 | 243.61 | 768 | 1373 | 0 | 0 | 1373 | 0 |
| GRPO_LA0 | GRPO | 0 | 2 | iteration | model_iter_1 | 96 | 25.79 | 75 | 94.00 | 752 | 0.50 | 1504 | 12032 | 288 | 24652 | 12 | 185.20 | 768 | 1228 | 0 | 0 | 1228 | 0 |
| GRPO_LA0 | GRPO | 0 | 3 | iteration | model_iter_2 | 96 | 31.21 | 76 | 118.00 | 1888 | 1.00 | 1888 | 15104 | 928 | 32075 | 11 | 238.21 | 768 | 1492 | 0 | 0 | 1492 | 0 |
| GRPO_LA0 | GRPO | 0 | 4 | iteration | model_iter_3 | 96 | 27.60 | 85 | 100.00 | 1600 | 1.00 | 1600 | 12800 | 640 | 26888 | 8 | 204.97 | 768 | 1324 | 0 | 0 | 1324 | 0 |
| GRPO_LA0 | GRPO | 0 | 5 | iteration | model_iter_4 | 96 | 30.50 | 70 | 118.00 | 1888 | 1.00 | 1888 | 15104 | 496 | 31213 | 13 | 245.45 | 768 | 1462 | 0 | 0 | 1462 | 0 |
| GRPO_LA0 | GRPO | 0 | 6 | iteration | model_iter_5 | 96 | 30.68 | 68 | 116.00 | 1376 | 0.74 | 1856 | 14848 | 1209 | 32116 | 3 | 287.35 | 768 | 1472 | 0 | 0 | 1472 | 0 |
| GRPO_LA0 | GRPO | 0 | 7 | iteration | model_iter_6 | 96 | 32.29 | 59 | 128.00 | 2048 | 1.00 | 2048 | 16384 | 592 | 33954 | 2 | 277.95 | 768 | 1547 | 0 | 0 | 1547 | 0 |
| GRPO_LA0 | GRPO | 0 | 8 | iteration | model_iter_7 | 96 | 28.84 | 72 | 108.00 | 1248 | 0.72 | 1728 | 13824 | 642 | 28934 | 1 | 346.80 | 768 | 1381 | 0 | 0 | 1381 | 0 |
| GRPO_LA0 | GRPO | 0 | 9 | iteration | model_iter_8 | 96 | 24.11 | 87 | 80.00 | 1280 | 1.00 | 1280 | 10240 | 416 | 21396 | 116 | 319.41 | 768 | 1150 | 0 | 0 | 1150 | 0 |
| GRPO_LA0 | GRPO | 0 | 10 | iteration | model_iter_9 | 96 | 38.06 | 45 | 158.00 | 2528 | 1.00 | 2528 | 20224 | 1072 | 42593 | 1 | 317.67 | 768 | 1825 | 0 | 0 | 1825 | 0 |
| GRPO_LA0 | GRPO | 0 | 11 | final eval pass | model_iter_10 | 96 | 25.20 | 91 |  | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0.00 | 768 | 1195 | 0 | 0 | 1195 | 0 |
| GRPO_LA5 | GRPO | 5 | 1 | iteration | model_iter_0 | 96 | 28.29 | 65 | 108.00 | 864 | 0.50 | 1728 | 13824 | 496 | 28648 | 8 | 275.26 | 768 | 1349 | 0 | 38840 | 40189 | 24792 |
| GRPO_LA5 | GRPO | 5 | 2 | iteration | model_iter_1 | 96 | 28.12 | 63 | 104.00 | 1184 | 0.71 | 1664 | 13312 | 1462 | 29549 | 1 | 308.00 | 768 | 1342 | 0 | 40334 | 41676 | 25819 |
| GRPO_LA5 | GRPO | 5 | 3 | iteration | model_iter_2 | 96 | 29.74 | 74 | 112.00 | 1792 | 1.00 | 1792 | 14336 | 640 | 29952 | 0 | 309.10 | 768 | 1420 | 0 | 41008 | 42428 | 26187 |
| GRPO_LA5 | GRPO | 5 | 4 | iteration | model_iter_3 | 96 | 29.06 | 76 | 106.00 | 1696 | 1.00 | 1696 | 13568 | 944 | 29024 | 0 | 327.75 | 768 | 1392 | 0 | 39016 | 40408 | 24546 |
| GRPO_LA5 | GRPO | 5 | 5 | iteration | model_iter_4 | 96 | 25.78 | 86 | 88.00 | 1408 | 1.00 | 1408 | 11264 | 768 | 24064 | 0 | 299.70 | 768 | 1237 | 0 | 31301 | 32538 | 19293 |
| GRPO_LA5 | GRPO | 5 | 6 | final eval pass | model_iter_5 | 96 | 22.57 | 92 |  | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0.00 | 768 | 1081 | 0 | 0 | 1081 | 0 |
