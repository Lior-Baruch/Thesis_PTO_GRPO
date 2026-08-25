**Base -> final session length per arm (arm means over 96 personas; base = the arm's own iteration-0 draw).** conv_len in utterances, n_th_turns in therapist turns, mean_turn_len in characters per therapist turn; change = final - base. GRPO_LA5 is right-censored at iteration 5 (its K=0 sibling runs to 10). The K contrast at the endpoints (PTO iter 10, GRPO iter 5) is in `session_shape_stability_length_kcontrast.md`.

| arm | final_iteration | conv_len_base | conv_len_final | conv_len_change | n_th_turns_base | n_th_turns_final | n_th_turns_change | mean_turn_len_base | mean_turn_len_final | mean_turn_len_change |
|---|---|---|---|---|---|---|---|---|---|---|
| PTO_LA0 | 10 | 28.385 | 20.385 | -8.000 | 14.333 | 10.229 | -4.104 | 300.568 | 686.202 | 385.634 |
| PTO_LA5 | 10 | 30.490 | 28.698 | -1.792 | 15.302 | 14.385 | -0.917 | 274.884 | 810.875 | 535.991 |
| GRPO_LA0 | 10 | 28.771 | 25.198 | -3.573 | 14.469 | 12.750 | -1.719 | 266.296 | 895.711 | 629.415 |
| GRPO_LA5 | 5 | 28.292 | 22.573 | -5.719 | 14.240 | 11.312 | -2.927 | 279.036 | 668.343 | 389.306 |
