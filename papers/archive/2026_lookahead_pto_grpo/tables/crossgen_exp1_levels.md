Exp1 (ICLR 2025: Llama-2-7B therapist, GPT-3.5 patient) model states, 96 conversations each (one per patient permutation), scored by TWO graders side by side (never averaged): `*_gpt4omini` = the SAME conversations re-scored by the Exp3 oracle (gpt-4o-mini-2024-07-18, T=0.1, V5 JSON-schema Q1+Q2; data/eval_scores/_crossgen); `*_gpt35` = the original GPT-3.5 oracle scores Exp1 saved beside each conversation (scores_i.csv); `*_iclr_tab1` = ICLR Table 1 as printed (reproduced by `*_gpt35` to 3 dp). Final = mean(Q1, Q2) per conversation, then averaged (= the lake's Q1Q2 composite). Iteration 0 = the untrained Llama-2-7B base (a single draw; both arms share it). Q1 = session satisfaction (5 items), Q2 = working alliance (17 items), both 1-5. Rows: Base, then K=0 iters 1-7, then K=5 iters 1-7.

| model | arm | K | iteration | n_gpt4omini | Q1_gpt4omini | Q2_gpt4omini | Final_gpt4omini | Final_sd_gpt4omini | n_gpt35 | Q1_gpt35 | Q2_gpt35 | Final_gpt35 | Final_sd_gpt35 | Q1_iclr_tab1 | Q2_iclr_tab1 | Final_iclr_tab1 | Final_gap_gpt4omini_minus_gpt35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp1_Base | Base |  | 0 | 96 | 3.925 | 3.805 | 3.865 | 0.698 | 96 | 3.521 | 3.385 | 3.453 | 0.740 | 3.521 | 3.385 | 3.453 | 0.412 |
| Exp1_LA0_I1 | L0 | 0 | 1 | 96 | 3.975 | 3.874 | 3.924 | 0.649 | 96 | 3.863 | 3.452 | 3.657 | 0.824 | 3.863 | 3.452 | 3.657 | 0.267 |
| Exp1_LA0_I2 | L0 | 0 | 2 | 96 | 3.977 | 3.801 | 3.889 | 0.648 | 96 | 3.750 | 3.435 | 3.593 | 0.878 | 3.750 | 3.435 | 3.593 | 0.297 |
| Exp1_LA0_I3 | L0 | 0 | 3 | 96 | 3.929 | 3.811 | 3.870 | 0.584 | 96 | 3.796 | 3.567 | 3.682 | 0.649 | 3.796 | 3.567 | 3.682 | 0.188 |
| Exp1_LA0_I4 | L0 | 0 | 4 | 96 | 4.108 | 3.975 | 4.042 | 0.523 | 96 | 3.969 | 3.585 | 3.777 | 0.769 | 3.969 | 3.585 | 3.777 | 0.265 |
| Exp1_LA0_I5 | L0 | 0 | 5 | 96 | 4.090 | 3.995 | 4.042 | 0.613 | 96 | 3.744 | 3.478 | 3.611 | 0.856 | 3.744 | 3.478 | 3.611 | 0.431 |
| Exp1_LA0_I6 | L0 | 0 | 6 | 96 | 4.085 | 3.942 | 4.014 | 0.643 | 96 | 3.794 | 3.494 | 3.644 | 0.834 | 3.794 | 3.494 | 3.644 | 0.370 |
| Exp1_LA0_I7 | L0 | 0 | 7 | 96 | 3.923 | 3.852 | 3.887 | 0.690 | 96 | 3.677 | 3.452 | 3.565 | 0.828 | 3.677 | 3.452 | 3.565 | 0.323 |
| Exp1_LA5_I1 | L5 | 5 | 1 | 96 | 4.075 | 3.850 | 3.963 | 0.550 | 96 | 3.898 | 3.523 | 3.710 | 0.712 | 3.898 | 3.523 | 3.710 | 0.252 |
| Exp1_LA5_I2 | L5 | 5 | 2 | 96 | 4.106 | 3.949 | 4.028 | 0.454 | 96 | 3.969 | 3.618 | 3.794 | 0.594 | 3.969 | 3.618 | 3.794 | 0.234 |
| Exp1_LA5_I3 | L5 | 5 | 3 | 96 | 4.127 | 4.023 | 4.075 | 0.511 | 96 | 4.050 | 3.683 | 3.866 | 0.611 | 4.050 | 3.683 | 3.866 | 0.209 |
| Exp1_LA5_I4 | L5 | 5 | 4 | 96 | 4.169 | 4.063 | 4.116 | 0.452 | 96 | 3.981 | 3.605 | 3.793 | 0.524 | 3.981 | 3.605 | 3.793 | 0.323 |
| Exp1_LA5_I5 | L5 | 5 | 5 | 96 | 4.215 | 4.050 | 4.132 | 0.450 | 96 | 4.225 | 3.660 | 3.942 | 0.559 | 4.225 | 3.660 | 3.942 | 0.190 |
| Exp1_LA5_I6 | L5 | 5 | 6 | 96 | 4.194 | 4.022 | 4.108 | 0.452 | 96 | 4.112 | 3.656 | 3.884 | 0.629 | 4.112 | 3.656 | 3.884 | 0.224 |
| Exp1_LA5_I7 | L5 | 5 | 7 | 96 | 4.252 | 4.091 | 4.171 | 0.408 | 96 | 4.190 | 3.775 | 3.982 | 0.414 | 4.190 | 3.775 | 3.982 | 0.189 |
