**Session shape, persona-paired K0 - K5 by matched iteration.** Deterministic text metrics computed from the eval transcripts (`eda_analysis.behavior.text_metrics`; judge-invariant): conv_len = utterances per conversation (therapist + patient), n_th_turns = therapist turns, mean_turn_len = characters per therapist turn, q_per_turn = literal '?' per therapist turn, loop = share of conversations with a verbatim-repeated therapist turn (degeneracy). Sign: + => K=0 higher (K0 - K5). Pairing unit: persona_id (the per-iteration file shuffle replayed; never file_index). mean_K0/mean_K5 are the arm means over the same 96 personas; mean_delta/dz/bootstrap 95% CI/Wilcoxon p are on the paired deltas; p_holm is Holm-corrected WITHIN each (method, metric) family ACROSS iterations (the tracked 7_stats k_paired_channels corrects across channels within an iteration instead — same delta/dz/p, different p_holm scope). Iteration 0 = two INDEPENDENT base draws (same base policy) — a free noise-floor row. GRPO_LA5 is right-censored at iteration 5 (its K=0 sibling runs to 10). Length metrics are unvalenced (longer is not better).

| method | metric | iteration | mean_K0 | mean_K5 | n | mean_delta | dz | ci_lo | ci_hi | p | p_holm | metric_unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PTO | conv_len | 0 | 28.385 | 30.490 | 96 | -2.104 | -0.099 | -6.344 | 2.313 | 0.278 | 1.000 | utterances / conversation |
| PTO | conv_len | 1 | 25.896 | 26.323 | 96 | -0.427 | -0.021 | -4.397 | 3.532 | 0.968 | 1.000 | utterances / conversation |
| PTO | conv_len | 2 | 27.688 | 26.615 | 96 | 1.073 | 0.060 | -2.761 | 4.583 | 0.505 | 1.000 | utterances / conversation |
| PTO | conv_len | 3 | 23.927 | 25.490 | 96 | -1.562 | -0.082 | -5.261 | 2.375 | 0.501 | 1.000 | utterances / conversation |
| PTO | conv_len | 4 | 23.854 | 22.844 | 96 | 1.010 | 0.061 | -2.292 | 4.458 | 0.410 | 1.000 | utterances / conversation |
| PTO | conv_len | 5 | 23.188 | 23.719 | 96 | -0.531 | -0.034 | -3.855 | 2.396 | 0.799 | 1.000 | utterances / conversation |
| PTO | conv_len | 6 | 21.719 | 25.260 | 96 | -3.542 | -0.275 | -6.104 | -1.124 | 0.007 | 0.055 | utterances / conversation |
| PTO | conv_len | 7 | 22.719 | 23.302 | 96 | -0.583 | -0.040 | -3.428 | 2.292 | 0.574 | 1.000 | utterances / conversation |
| PTO | conv_len | 8 | 20.771 | 25.646 | 96 | -4.875 | -0.353 | -7.542 | -2.124 | 0.000 | 0.001 | utterances / conversation |
| PTO | conv_len | 9 | 19.198 | 27.490 | 96 | -8.292 | -0.614 | -11.148 | -5.719 | 0.000 | 0.000 | utterances / conversation |
| PTO | conv_len | 10 | 20.385 | 28.698 | 96 | -8.312 | -0.548 | -11.219 | -5.260 | 0.000 | 0.000 | utterances / conversation |
| PTO | n_th_turns | 0 | 14.333 | 15.302 | 96 | -0.969 | -0.092 | -3.073 | 1.250 | 0.312 | 1.000 | therapist turns / conversation |
| PTO | n_th_turns | 1 | 13.010 | 13.240 | 96 | -0.229 | -0.023 | -2.229 | 1.750 | 0.950 | 1.000 | therapist turns / conversation |
| PTO | n_th_turns | 2 | 13.917 | 13.396 | 96 | 0.521 | 0.058 | -1.376 | 2.261 | 0.516 | 1.000 | therapist turns / conversation |
| PTO | n_th_turns | 3 | 12.010 | 12.812 | 96 | -0.802 | -0.084 | -2.625 | 1.157 | 0.482 | 1.000 | therapist turns / conversation |
| PTO | n_th_turns | 4 | 11.948 | 11.458 | 96 | 0.490 | 0.059 | -1.146 | 2.208 | 0.421 | 1.000 | therapist turns / conversation |
| PTO | n_th_turns | 5 | 11.604 | 11.896 | 96 | -0.292 | -0.038 | -1.948 | 1.188 | 0.828 | 1.000 | therapist turns / conversation |
| PTO | n_th_turns | 6 | 10.885 | 12.677 | 96 | -1.792 | -0.280 | -3.053 | -0.583 | 0.007 | 0.053 | therapist turns / conversation |
| PTO | n_th_turns | 7 | 11.385 | 11.677 | 96 | -0.292 | -0.040 | -1.709 | 1.146 | 0.581 | 1.000 | therapist turns / conversation |
| PTO | n_th_turns | 8 | 10.406 | 12.844 | 96 | -2.438 | -0.353 | -3.792 | -1.073 | 0.000 | 0.001 | therapist turns / conversation |
| PTO | n_th_turns | 9 | 9.646 | 13.771 | 96 | -4.125 | -0.609 | -5.563 | -2.833 | 0.000 | 0.000 | therapist turns / conversation |
| PTO | n_th_turns | 10 | 10.229 | 14.385 | 96 | -4.156 | -0.548 | -5.635 | -2.656 | 0.000 | 0.000 | therapist turns / conversation |
| PTO | mean_turn_len | 0 | 300.568 | 274.884 | 96 | 25.684 | 0.118 | -17.025 | 68.641 | 0.077 | 0.154 | chars / therapist turn |
| PTO | mean_turn_len | 1 | 285.074 | 288.916 | 96 | -3.841 | -0.020 | -40.382 | 33.701 | 0.823 | 0.823 | chars / therapist turn |
| PTO | mean_turn_len | 2 | 310.382 | 366.031 | 96 | -55.649 | -0.230 | -104.174 | -9.314 | 0.040 | 0.119 | chars / therapist turn |
| PTO | mean_turn_len | 3 | 352.748 | 421.129 | 96 | -68.381 | -0.267 | -119.400 | -17.344 | 0.009 | 0.047 | chars / therapist turn |
| PTO | mean_turn_len | 4 | 394.514 | 482.746 | 96 | -88.232 | -0.339 | -138.515 | -36.520 | 0.000 | 0.002 | chars / therapist turn |
| PTO | mean_turn_len | 5 | 482.772 | 584.541 | 96 | -101.769 | -0.349 | -160.757 | -42.674 | 0.001 | 0.003 | chars / therapist turn |
| PTO | mean_turn_len | 6 | 553.494 | 618.345 | 96 | -64.850 | -0.236 | -119.978 | -9.381 | 0.022 | 0.090 | chars / therapist turn |
| PTO | mean_turn_len | 7 | 574.405 | 716.595 | 96 | -142.189 | -0.545 | -190.127 | -90.171 | 0.000 | 0.000 | chars / therapist turn |
| PTO | mean_turn_len | 8 | 641.316 | 730.672 | 96 | -89.356 | -0.428 | -130.208 | -48.281 | 0.000 | 0.002 | chars / therapist turn |
| PTO | mean_turn_len | 9 | 641.712 | 801.897 | 96 | -160.185 | -0.725 | -204.898 | -115.547 | 0.000 | 0.000 | chars / therapist turn |
| PTO | mean_turn_len | 10 | 686.202 | 810.875 | 96 | -124.673 | -0.555 | -169.253 | -79.701 | 0.000 | 0.000 | chars / therapist turn |
| PTO | q_per_turn | 0 | 0.930 | 0.766 | 96 | 0.164 | 0.172 | -0.019 | 0.362 | 0.269 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 1 | 0.844 | 0.767 | 96 | 0.077 | 0.084 | -0.092 | 0.270 | 0.809 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 2 | 0.779 | 0.779 | 96 | -0.000 | -0.000 | -0.157 | 0.156 | 0.783 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 3 | 0.856 | 0.823 | 96 | 0.034 | 0.037 | -0.155 | 0.211 | 0.534 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 4 | 0.752 | 0.766 | 96 | -0.014 | -0.013 | -0.274 | 0.178 | 0.251 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 5 | 0.804 | 0.715 | 96 | 0.088 | 0.098 | -0.102 | 0.258 | 0.050 | 0.546 | '?' / therapist turn |
| PTO | q_per_turn | 6 | 0.742 | 0.943 | 96 | -0.201 | -0.146 | -0.487 | 0.068 | 0.062 | 0.623 | '?' / therapist turn |
| PTO | q_per_turn | 7 | 0.680 | 0.595 | 96 | 0.085 | 0.101 | -0.088 | 0.239 | 0.131 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 8 | 0.675 | 0.681 | 96 | -0.006 | -0.008 | -0.145 | 0.132 | 0.847 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 9 | 0.647 | 0.589 | 96 | 0.058 | 0.103 | -0.050 | 0.173 | 0.391 | 1.000 | '?' / therapist turn |
| PTO | q_per_turn | 10 | 0.550 | 0.616 | 96 | -0.065 | -0.110 | -0.187 | 0.050 | 0.392 | 1.000 | '?' / therapist turn |
| PTO | loop | 0 | 0.490 | 0.448 | 96 | 0.042 | 0.064 | -0.083 | 0.167 | 0.527 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 1 | 0.375 | 0.427 | 96 | -0.052 | -0.084 | -0.177 | 0.073 | 0.411 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 2 | 0.271 | 0.417 | 96 | -0.146 | -0.231 | -0.271 | -0.021 | 0.027 | 0.242 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 3 | 0.104 | 0.219 | 96 | -0.115 | -0.240 | -0.208 | -0.021 | 0.022 | 0.218 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 4 | 0.042 | 0.115 | 96 | -0.073 | -0.219 | -0.135 | -0.010 | 0.035 | 0.278 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 5 | 0.010 | 0.042 | 96 | -0.031 | -0.138 | -0.083 | 0.010 | 0.180 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 6 | 0.010 | 0.083 | 96 | -0.073 | -0.244 | -0.146 | -0.021 | 0.020 | 0.216 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 7 | 0.010 | 0.010 | 96 | 0.000 | 0.000 | -0.031 | 0.031 | 1.000 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 8 | 0.010 | 0.021 | 96 | -0.010 | -0.059 | -0.042 | 0.021 | 0.564 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 9 | 0.000 | 0.000 | 96 | 0.000 |  | 0.000 | 0.000 | 1.000 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| PTO | loop | 10 | 0.000 | 0.010 | 96 | -0.010 | -0.102 | -0.031 | 0.000 | 0.317 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| GRPO | conv_len | 0 | 28.771 | 28.292 | 96 | 0.479 | 0.026 | -3.043 | 4.125 | 0.932 | 1.000 | utterances / conversation |
| GRPO | conv_len | 1 | 25.792 | 28.125 | 96 | -2.333 | -0.131 | -5.834 | 1.376 | 0.141 | 0.564 | utterances / conversation |
| GRPO | conv_len | 2 | 31.208 | 29.740 | 96 | 1.469 | 0.076 | -2.511 | 5.208 | 0.417 | 1.000 | utterances / conversation |
| GRPO | conv_len | 3 | 27.604 | 29.062 | 96 | -1.458 | -0.095 | -4.490 | 1.646 | 0.484 | 1.000 | utterances / conversation |
| GRPO | conv_len | 4 | 30.500 | 25.781 | 96 | 4.719 | 0.273 | 1.229 | 8.073 | 0.010 | 0.049 | utterances / conversation |
| GRPO | conv_len | 5 | 30.677 | 22.573 | 96 | 8.104 | 0.531 | 5.052 | 11.011 | 0.000 | 0.000 | utterances / conversation |
| GRPO | n_th_turns | 0 | 14.469 | 14.240 | 96 | 0.229 | 0.025 | -1.532 | 2.042 | 0.918 | 1.000 | therapist turns / conversation |
| GRPO | n_th_turns | 1 | 13.000 | 14.146 | 96 | -1.146 | -0.129 | -2.885 | 0.698 | 0.138 | 0.551 | therapist turns / conversation |
| GRPO | n_th_turns | 2 | 15.667 | 14.948 | 96 | 0.719 | 0.075 | -1.282 | 2.604 | 0.422 | 1.000 | therapist turns / conversation |
| GRPO | n_th_turns | 3 | 13.812 | 14.562 | 96 | -0.750 | -0.098 | -2.261 | 0.802 | 0.462 | 1.000 | therapist turns / conversation |
| GRPO | n_th_turns | 4 | 15.271 | 12.896 | 96 | 2.375 | 0.275 | 0.625 | 4.052 | 0.009 | 0.047 | therapist turns / conversation |
| GRPO | n_th_turns | 5 | 15.344 | 11.312 | 96 | 4.031 | 0.528 | 2.510 | 5.479 | 0.000 | 0.000 | therapist turns / conversation |
| GRPO | mean_turn_len | 0 | 266.296 | 279.036 | 96 | -12.740 | -0.070 | -48.006 | 23.422 | 0.546 | 1.000 | chars / therapist turn |
| GRPO | mean_turn_len | 1 | 284.241 | 292.523 | 96 | -8.282 | -0.042 | -48.324 | 29.941 | 0.628 | 1.000 | chars / therapist turn |
| GRPO | mean_turn_len | 2 | 238.834 | 282.982 | 96 | -44.148 | -0.248 | -80.100 | -7.014 | 0.021 | 0.063 | chars / therapist turn |
| GRPO | mean_turn_len | 3 | 313.278 | 394.658 | 96 | -81.380 | -0.419 | -121.262 | -44.406 | 0.000 | 0.001 | chars / therapist turn |
| GRPO | mean_turn_len | 4 | 360.412 | 450.511 | 96 | -90.099 | -0.418 | -133.572 | -48.126 | 0.000 | 0.001 | chars / therapist turn |
| GRPO | mean_turn_len | 5 | 461.787 | 668.343 | 96 | -206.556 | -0.815 | -257.754 | -155.117 | 0.000 | 0.000 | chars / therapist turn |
| GRPO | q_per_turn | 0 | 0.829 | 0.740 | 96 | 0.088 | 0.102 | -0.081 | 0.270 | 0.550 | 1.000 | '?' / therapist turn |
| GRPO | q_per_turn | 1 | 0.682 | 0.884 | 96 | -0.202 | -0.198 | -0.426 | -0.019 | 0.084 | 0.335 | '?' / therapist turn |
| GRPO | q_per_turn | 2 | 0.774 | 0.784 | 96 | -0.011 | -0.017 | -0.127 | 0.116 | 0.941 | 1.000 | '?' / therapist turn |
| GRPO | q_per_turn | 3 | 0.727 | 0.715 | 96 | 0.012 | 0.018 | -0.124 | 0.142 | 0.861 | 1.000 | '?' / therapist turn |
| GRPO | q_per_turn | 4 | 0.496 | 0.787 | 96 | -0.291 | -0.404 | -0.442 | -0.153 | 0.000 | 0.000 | '?' / therapist turn |
| GRPO | q_per_turn | 5 | 0.324 | 0.691 | 96 | -0.367 | -0.605 | -0.488 | -0.254 | 0.000 | 0.000 | '?' / therapist turn |
| GRPO | loop | 0 | 0.479 | 0.490 | 96 | -0.010 | -0.019 | -0.115 | 0.104 | 0.853 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| GRPO | loop | 1 | 0.365 | 0.354 | 96 | 0.010 | 0.017 | -0.115 | 0.135 | 0.866 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| GRPO | loop | 2 | 0.354 | 0.333 | 96 | 0.021 | 0.031 | -0.115 | 0.156 | 0.763 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| GRPO | loop | 3 | 0.000 | 0.083 | 96 | -0.083 | -0.300 | -0.146 | -0.031 | 0.005 | 0.028 | share of conversations with a verbatim-repeated therapist turn |
| GRPO | loop | 4 | 0.042 | 0.021 | 96 | 0.021 | 0.083 | -0.031 | 0.073 | 0.414 | 1.000 | share of conversations with a verbatim-repeated therapist turn |
| GRPO | loop | 5 | 0.021 | 0.000 | 96 | 0.021 | 0.145 | 0.000 | 0.052 | 0.157 | 0.786 | share of conversations with a verbatim-repeated therapist turn |
