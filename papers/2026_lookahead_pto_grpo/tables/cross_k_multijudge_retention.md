**Gain retention by look-ahead K.** `retention = Δ held-out / Δ primary` of each model state over a reference base — the train/test generalisation ratio (~1 = the gain is real to a judge that never played the patient; ~0 = it existed only in the optimised grader). `ref_kind`: `own_base` = the arm's OWN base draw; `method_LA0_base` / `method_LA5_base` = the method's K=0 / K=5 base draw as a SHARED reference for both K arms (for a PTO_LA0 row, `own_base` and `method_LA0_base` are the same reference and duplicate each other by design); `eda_view_PTO_LA{K}_base` = the tracked EDA's convention (PTO's base of the same view), given for the GRPO arms so the tracked 8_measurement/multijudge_gain_retention.md numbers can be matched. Iteration-0 rows under a shared reference are two INDEPENDENT base draws (noise floor). `retention` and its CI are suppressed (blank) where |Δ primary| < `min_primary_delta` — 0.15 on the 1–5 / 1–7 rubrics (the `reliability.gain_retention` default, whose persona-bootstrap CI this is) and 0.05 on the 0–1 rate metrics PCT/MICI (their deltas are ~3× smaller; a 0.15 floor blanks almost every MICI row). Paired on persona_id (the trainer reshuffles the 96 personas every iteration; file_index is not a pairing key). Direction-agnostic on MICI (both deltas flip together). GRPO_LA5 is right-censored at iteration 5 (its full budget); PTO arms and GRPO_LA0 run to 10. Cross-check: GRPO_LA5 Q1 iteration 5 vs eda_view_PTO_LA5_base = 1.082 [0.937, 1.274]; PTO_LA5 Q2 iteration 10 = 0.562 vs PTO_LA0 0.849 (tracked L5/L0 tables).

| arm | method | K | iteration | metric | ref_kind | reference | ref_is_own_base | n | delta_primary | delta_judge | retention | retention_ci_lo | retention_ci_hi | same_sign | min_primary_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRPO_LA0 | GRPO | 0 | 0 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.066 | 0.031 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.269 | 0.244 | 0.909 | 0.580 | 1.398 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.359 | 0.277 | 0.772 | 0.522 | 1.120 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.993 | 0.807 | 0.812 | 0.713 | 0.934 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.004 | 0.721 | 0.718 | 0.601 | 0.849 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.971 | 0.657 | 0.676 | 0.541 | 0.832 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.965 | 0.541 | 0.560 | 0.373 | 0.731 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.074 | 0.735 | 0.685 | 0.557 | 0.821 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.082 | 0.787 | 0.728 | 0.622 | 0.860 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.807 | 0.172 | 0.214 | -0.046 | 0.424 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.752 | 0.427 | 0.568 | 0.476 | 0.663 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.098 | 0.021 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.254 | 0.288 | 1.131 | 0.649 | 1.773 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.379 | 0.298 | 0.786 | 0.463 | 1.169 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.012 | 0.904 | 0.893 | 0.772 | 1.035 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.975 | 0.773 | 0.793 | 0.641 | 0.956 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.946 | 0.688 | 0.727 | 0.554 | 0.912 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.888 | 0.506 | 0.570 | 0.343 | 0.787 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.023 | 0.717 | 0.701 | 0.549 | 0.851 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.012 | 0.652 | 0.644 | 0.521 | 0.779 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.606 | 0.021 | 0.034 | -0.457 | 0.344 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.683 | 0.194 | 0.284 | 0.028 | 0.430 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.035 | 0.042 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.283 | 0.201 | 0.710 | 0.376 | 1.133 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.339 | 0.256 | 0.756 | 0.496 | 1.128 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.974 | 0.710 | 0.729 | 0.622 | 0.860 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.033 | 0.669 | 0.647 | 0.534 | 0.771 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.997 | 0.626 | 0.628 | 0.516 | 0.768 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.043 | 0.575 | 0.552 | 0.397 | 0.714 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.124 | 0.754 | 0.671 | 0.556 | 0.808 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.151 | 0.923 | 0.801 | 0.683 | 0.954 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.008 | 0.324 | 0.322 | 0.153 | 0.472 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q2 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.821 | 0.661 | 0.805 | 0.682 | 0.966 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.045 | 0.105 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.131 | 0.260 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.160 | 0.329 | 2.060 | 1.166 | 2.624 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.457 | 0.740 | 1.622 | 1.311 | 2.131 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.447 | 0.662 | 1.482 | 1.194 | 1.902 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.364 | 0.583 | 1.604 | 1.215 | 2.290 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.452 | 0.584 | 1.292 | 1.012 | 1.689 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.473 | 0.691 | 1.461 | 1.187 | 1.871 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.530 | 0.778 | 1.469 | 1.190 | 1.878 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.244 | 0.220 | 0.900 | 0.313 | 1.541 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | WAI-SR | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.594 | 0.552 | 0.930 | 0.693 | 1.182 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.022 | 0.046 |  |  |  | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.092 | 0.195 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.130 | 0.251 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.427 | 0.658 | 1.540 | 1.215 | 2.113 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.367 | 0.517 | 1.408 | 1.018 | 2.071 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.331 | 0.557 | 1.685 | 1.238 | 2.486 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.326 | 0.456 | 1.400 | 0.978 | 2.144 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.372 | 0.582 | 1.563 | 1.169 | 2.296 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.365 | 0.639 | 1.754 | 1.345 | 2.471 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.102 | 0.078 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | CSQ-8 | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.389 | 0.443 | 1.137 | 0.822 | 1.595 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.021 | 0.057 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.149 | 0.165 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.217 | 0.212 | 0.976 | 0.503 | 1.512 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.589 | 0.517 | 0.879 | 0.700 | 1.129 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.573 | 0.431 | 0.752 | 0.560 | 0.950 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.484 | 0.405 | 0.835 | 0.610 | 1.149 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.469 | 0.243 | 0.519 | 0.187 | 0.778 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.582 | 0.424 | 0.728 | 0.528 | 0.942 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.604 | 0.453 | 0.750 | 0.587 | 0.972 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.214 | -0.085 | -0.398 | -1.291 | 0.352 | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MI-SAT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.642 | 0.361 | 0.562 | 0.369 | 0.746 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.091 | 0.049 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.151 | 0.141 | 0.931 | 0.317 | 1.381 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.362 | 0.234 | 0.647 | 0.407 | 1.052 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.805 | 0.544 | 0.676 | 0.563 | 0.818 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.906 | 0.443 | 0.489 | 0.383 | 0.600 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.807 | 0.247 | 0.306 | 0.202 | 0.413 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.878 | 0.302 | 0.344 | 0.241 | 0.444 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.969 | 0.352 | 0.363 | 0.278 | 0.444 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.102 | 0.380 | 0.345 | 0.262 | 0.423 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.805 | 0.141 | 0.175 | 0.033 | 0.283 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MITI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.789 | 0.260 | 0.330 | 0.200 | 0.458 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.002 | 0.006 |  |  |  | False | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.008 | 0.019 |  |  |  | False | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.014 | 0.033 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.055 | 0.081 | 1.472 | 1.027 | 1.774 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.063 | 0.066 | 1.042 | 0.706 | 1.378 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.033 | 0.035 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.041 | 0.034 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.062 | 0.063 | 1.027 | 0.694 | 1.356 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.083 | 0.076 | 0.919 | 0.631 | 1.268 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.039 | -0.056 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | PCT | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.085 | 0.093 | 1.085 | 0.823 | 1.442 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 0 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.002 | 0.020 |  |  |  | False | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.014 | 0.009 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.044 | -0.055 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.017 | 0.043 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.038 | 0.144 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.064 | 0.265 | 4.125 | 2.372 | 5.269 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.042 | 0.248 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.057 | 0.291 | 5.122 | 2.411 | 6.085 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.322 | 0.534 | 1.659 | 1.368 | 2.040 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.137 | 0.331 | 2.421 | 1.752 | 3.792 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | MICI | eda_view_PTO_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.625 | 0.686 | 1.098 | 0.962 | 1.252 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.202 | 0.213 | 1.053 | 0.582 | 1.502 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.293 | 0.246 | 0.840 | 0.539 | 1.270 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.927 | 0.776 | 0.837 | 0.728 | 0.978 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.938 | 0.689 | 0.735 | 0.597 | 0.897 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.905 | 0.626 | 0.691 | 0.543 | 0.860 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.899 | 0.510 | 0.567 | 0.379 | 0.728 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 1.007 | 0.704 | 0.699 | 0.565 | 0.850 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 1.016 | 0.756 | 0.745 | 0.627 | 0.898 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.741 | 0.141 | 0.191 | -0.064 | 0.382 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.686 | 0.396 | 0.578 | 0.457 | 0.723 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.156 | 0.267 | 1.707 | 0.762 | 2.177 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.281 | 0.277 | 0.985 | 0.586 | 1.593 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.915 | 0.883 | 0.966 | 0.831 | 1.131 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.877 | 0.752 | 0.857 | 0.677 | 1.069 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.848 | 0.667 | 0.786 | 0.587 | 1.003 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.790 | 0.485 | 0.615 | 0.402 | 0.841 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.925 | 0.696 | 0.752 | 0.571 | 0.957 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.915 | 0.631 | 0.690 | 0.554 | 0.861 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.508 | 0.000 | 0.000 | -0.549 | 0.331 | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.585 | 0.173 | 0.295 | 0.054 | 0.488 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.248 | 0.159 | 0.642 | 0.292 | 1.087 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.304 | 0.214 | 0.706 | 0.427 | 1.164 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.939 | 0.668 | 0.711 | 0.609 | 0.833 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.998 | 0.627 | 0.628 | 0.510 | 0.762 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.962 | 0.585 | 0.608 | 0.484 | 0.766 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 1.008 | 0.534 | 0.529 | 0.388 | 0.679 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 1.089 | 0.713 | 0.654 | 0.539 | 0.795 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 1.116 | 0.881 | 0.789 | 0.666 | 0.946 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.973 | 0.282 | 0.290 | 0.146 | 0.434 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.786 | 0.619 | 0.788 | 0.658 | 0.970 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.086 | 0.155 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.115 | 0.224 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.411 | 0.635 | 1.544 | 1.232 | 2.055 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.402 | 0.557 | 1.387 | 1.061 | 1.927 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.319 | 0.478 | 1.501 | 1.114 | 2.200 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.407 | 0.479 | 1.177 | 0.858 | 1.630 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.428 | 0.586 | 1.369 | 1.109 | 1.801 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.484 | 0.673 | 1.389 | 1.118 | 1.790 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.199 | 0.115 | 0.576 | -0.044 | 1.131 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.549 | 0.447 | 0.815 | 0.610 | 1.054 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.115 | 0.150 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.152 | 0.206 | 1.350 | 0.721 | 1.804 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.449 | 0.612 | 1.362 | 1.094 | 1.778 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.389 | 0.471 | 1.211 | 0.869 | 1.656 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.353 | 0.512 | 1.450 | 1.122 | 1.980 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.348 | 0.410 | 1.180 | 0.833 | 1.703 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.395 | 0.536 | 1.360 | 1.029 | 1.812 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.387 | 0.594 | 1.535 | 1.183 | 2.089 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.124 | 0.033 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.411 | 0.397 | 0.965 | 0.678 | 1.306 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.128 | 0.108 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.196 | 0.155 | 0.788 | 0.391 | 1.153 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.568 | 0.460 | 0.810 | 0.661 | 1.004 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.552 | 0.373 | 0.676 | 0.501 | 0.870 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.464 | 0.347 | 0.749 | 0.533 | 0.986 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.448 | 0.186 | 0.415 | -0.000 | 0.680 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.561 | 0.366 | 0.653 | 0.476 | 0.834 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.583 | 0.396 | 0.679 | 0.522 | 0.846 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.193 | -0.142 | -0.739 | -1.426 | 0.176 | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.622 | 0.304 | 0.489 | 0.317 | 0.628 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.060 | 0.091 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.271 | 0.185 | 0.683 | 0.295 | 1.197 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.714 | 0.495 | 0.693 | 0.546 | 0.854 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.815 | 0.393 | 0.482 | 0.365 | 0.600 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.716 | 0.198 | 0.276 | 0.125 | 0.416 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.786 | 0.253 | 0.321 | 0.189 | 0.440 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.878 | 0.302 | 0.344 | 0.238 | 0.440 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 1.010 | 0.331 | 0.327 | 0.224 | 0.421 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.714 | 0.091 | 0.128 | -0.059 | 0.267 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MITI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.698 | 0.211 | 0.302 | 0.146 | 0.462 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | -0.007 | 0.013 |  |  |  | False | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.016 | 0.027 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.056 | 0.075 | 1.326 | 0.925 | 1.608 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.065 | 0.060 | 0.925 | 0.627 | 1.268 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.035 | 0.029 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.042 | 0.028 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.063 | 0.058 | 0.908 | 0.576 | 1.249 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.085 | 0.070 | 0.832 | 0.570 | 1.116 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | -0.037 | -0.062 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | PCT | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.087 | 0.087 | 0.997 | 0.755 | 1.299 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.016 | -0.011 |  |  |  | False | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | -0.042 | -0.075 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.019 | 0.024 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.039 | 0.124 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.066 | 0.245 | 3.713 | 2.183 | 5.057 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.043 | 0.228 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.059 | 0.271 | 4.626 | 2.278 | 5.846 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.324 | 0.515 | 1.589 | 1.297 | 2.000 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.139 | 0.312 | 2.247 | 1.557 | 3.748 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | MICI | method_LA0_base | GRPOExp3_LA0_Base | True | 96 | 0.626 | 0.666 | 1.063 | 0.944 | 1.210 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 0 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.104 | 0.026 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.306 | 0.239 | 0.783 | 0.501 | 1.258 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.396 | 0.272 | 0.687 | 0.504 | 0.925 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.030 | 0.802 | 0.779 | 0.686 | 0.900 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.041 | 0.716 | 0.688 | 0.570 | 0.803 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.009 | 0.652 | 0.647 | 0.527 | 0.775 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.002 | 0.536 | 0.535 | 0.365 | 0.675 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.111 | 0.731 | 0.658 | 0.536 | 0.782 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.119 | 0.783 | 0.699 | 0.598 | 0.828 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.844 | 0.168 | 0.199 | -0.035 | 0.387 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.789 | 0.423 | 0.535 | 0.441 | 0.645 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.110 | 0.013 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.267 | 0.279 | 1.047 | 0.604 | 1.745 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.392 | 0.290 | 0.739 | 0.504 | 1.017 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.025 | 0.896 | 0.874 | 0.752 | 1.011 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.988 | 0.765 | 0.774 | 0.628 | 0.929 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.958 | 0.679 | 0.709 | 0.543 | 0.879 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.900 | 0.498 | 0.553 | 0.359 | 0.733 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.035 | 0.708 | 0.684 | 0.529 | 0.835 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.025 | 0.644 | 0.628 | 0.504 | 0.765 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.619 | 0.013 | 0.020 | -0.416 | 0.318 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.696 | 0.185 | 0.266 | 0.075 | 0.401 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.097 | 0.040 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.345 | 0.200 | 0.579 | 0.289 | 0.897 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.401 | 0.255 | 0.636 | 0.446 | 0.911 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.036 | 0.708 | 0.684 | 0.590 | 0.796 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.095 | 0.667 | 0.609 | 0.502 | 0.716 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.059 | 0.625 | 0.590 | 0.482 | 0.707 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.105 | 0.574 | 0.520 | 0.380 | 0.658 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.186 | 0.753 | 0.635 | 0.530 | 0.764 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.213 | 0.922 | 0.760 | 0.655 | 0.895 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.070 | 0.323 | 0.302 | 0.149 | 0.439 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.883 | 0.660 | 0.747 | 0.638 | 0.897 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.033 | 0.057 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.119 | 0.212 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.148 | 0.281 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.444 | 0.693 | 1.559 | 1.283 | 2.044 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.435 | 0.615 | 1.413 | 1.121 | 1.870 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.352 | 0.536 | 1.523 | 1.210 | 2.094 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.440 | 0.536 | 1.219 | 0.930 | 1.654 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.461 | 0.643 | 1.395 | 1.136 | 1.779 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.517 | 0.730 | 1.411 | 1.160 | 1.809 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.232 | 0.172 | 0.742 | 0.177 | 1.347 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.582 | 0.504 | 0.867 | 0.681 | 1.088 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.007 | -0.004 |  |  |  | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.121 | 0.146 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.159 | 0.202 | 1.270 | 0.631 | 1.772 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.456 | 0.608 | 1.334 | 1.082 | 1.655 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.396 | 0.467 | 1.181 | 0.852 | 1.600 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.359 | 0.508 | 1.413 | 1.065 | 1.886 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.354 | 0.406 | 1.147 | 0.760 | 1.677 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.401 | 0.533 | 1.328 | 0.978 | 1.876 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.393 | 0.590 | 1.500 | 1.149 | 2.080 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.130 | 0.029 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.418 | 0.393 | 0.941 | 0.646 | 1.292 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.042 | 0.082 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.170 | 0.189 | 1.112 | 0.574 | 1.547 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.238 | 0.236 | 0.993 | 0.625 | 1.488 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.609 | 0.542 | 0.889 | 0.744 | 1.092 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.594 | 0.455 | 0.766 | 0.591 | 0.957 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.505 | 0.429 | 0.849 | 0.660 | 1.103 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.490 | 0.267 | 0.546 | 0.239 | 0.792 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.602 | 0.448 | 0.744 | 0.553 | 0.948 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.625 | 0.477 | 0.764 | 0.609 | 0.983 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.234 | -0.061 | -0.259 | -0.973 | 0.333 | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.663 | 0.385 | 0.581 | 0.430 | 0.725 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.104 | 0.057 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.164 | 0.148 | 0.905 | 0.365 | 1.351 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.375 | 0.242 | 0.646 | 0.409 | 0.959 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.818 | 0.552 | 0.675 | 0.558 | 0.800 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.919 | 0.451 | 0.490 | 0.396 | 0.585 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.820 | 0.255 | 0.311 | 0.194 | 0.423 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.891 | 0.310 | 0.348 | 0.219 | 0.459 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.982 | 0.359 | 0.366 | 0.279 | 0.447 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 1.115 | 0.388 | 0.348 | 0.268 | 0.427 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.818 | 0.148 | 0.182 | 0.038 | 0.299 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MITI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.802 | 0.268 | 0.334 | 0.206 | 0.468 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 0 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.016 | 0.019 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.010 | 0.032 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.032 | 0.046 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.073 | 0.094 | 1.290 | 0.964 | 1.678 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.081 | 0.079 | 0.974 | 0.696 | 1.269 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.052 | 0.048 | 0.937 | 0.556 | 1.273 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.059 | 0.048 | 0.808 | 0.443 | 1.157 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.080 | 0.077 | 0.962 | 0.670 | 1.297 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.101 | 0.089 | 0.887 | 0.618 | 1.178 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | -0.021 | -0.043 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | PCT | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.103 | 0.106 | 1.024 | 0.793 | 1.326 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 0 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.002 | 0.057 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.019 | 0.047 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | -0.040 | -0.017 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.021 | 0.081 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.042 | 0.181 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.068 | 0.302 | 4.427 | 2.560 | 6.058 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.046 | 0.285 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.061 | 0.328 | 5.392 | 2.782 | 6.817 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.326 | 0.572 | 1.753 | 1.458 | 2.112 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.141 | 0.369 | 2.617 | 1.939 | 3.998 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | MICI | method_LA5_base | GRPOExp3_LA5_Base | False | 96 | 0.629 | 0.723 | 1.151 | 1.040 | 1.280 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.202 | 0.213 | 1.053 | 0.582 | 1.502 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.293 | 0.246 | 0.840 | 0.539 | 1.270 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.927 | 0.776 | 0.837 | 0.728 | 0.978 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.938 | 0.689 | 0.735 | 0.597 | 0.897 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.905 | 0.626 | 0.691 | 0.543 | 0.860 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.899 | 0.510 | 0.567 | 0.379 | 0.728 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 1.007 | 0.704 | 0.699 | 0.565 | 0.850 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 1.016 | 0.756 | 0.745 | 0.627 | 0.898 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.741 | 0.141 | 0.191 | -0.064 | 0.382 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.686 | 0.396 | 0.578 | 0.457 | 0.723 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.156 | 0.267 | 1.707 | 0.762 | 2.177 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.281 | 0.277 | 0.985 | 0.586 | 1.593 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.915 | 0.883 | 0.966 | 0.831 | 1.131 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.877 | 0.752 | 0.857 | 0.677 | 1.069 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.848 | 0.667 | 0.786 | 0.587 | 1.003 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.790 | 0.485 | 0.615 | 0.402 | 0.841 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.925 | 0.696 | 0.752 | 0.571 | 0.957 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.915 | 0.631 | 0.690 | 0.554 | 0.861 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.508 | 0.000 | 0.000 | -0.549 | 0.331 | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q1 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.585 | 0.173 | 0.295 | 0.054 | 0.488 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.248 | 0.159 | 0.642 | 0.292 | 1.087 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.304 | 0.214 | 0.706 | 0.427 | 1.164 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.939 | 0.668 | 0.711 | 0.609 | 0.833 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.998 | 0.627 | 0.628 | 0.510 | 0.762 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.962 | 0.585 | 0.608 | 0.484 | 0.766 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 1.008 | 0.534 | 0.529 | 0.388 | 0.679 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 1.089 | 0.713 | 0.654 | 0.539 | 0.795 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 1.116 | 0.881 | 0.789 | 0.666 | 0.946 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.973 | 0.282 | 0.290 | 0.146 | 0.434 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | Q2 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.786 | 0.619 | 0.788 | 0.658 | 0.970 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.086 | 0.155 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.115 | 0.224 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.411 | 0.635 | 1.544 | 1.232 | 2.055 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.402 | 0.557 | 1.387 | 1.061 | 1.927 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.319 | 0.478 | 1.501 | 1.114 | 2.200 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.407 | 0.479 | 1.177 | 0.858 | 1.630 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.428 | 0.586 | 1.369 | 1.109 | 1.801 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.484 | 0.673 | 1.389 | 1.118 | 1.790 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.199 | 0.115 | 0.576 | -0.044 | 1.131 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | WAI-SR | own_base | GRPOExp3_LA0_Base | True | 96 | 0.549 | 0.447 | 0.815 | 0.610 | 1.054 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.115 | 0.150 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.152 | 0.206 | 1.350 | 0.721 | 1.804 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.449 | 0.612 | 1.362 | 1.094 | 1.778 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.389 | 0.471 | 1.211 | 0.869 | 1.656 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.353 | 0.512 | 1.450 | 1.122 | 1.980 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.348 | 0.410 | 1.180 | 0.833 | 1.703 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.395 | 0.536 | 1.360 | 1.029 | 1.812 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.387 | 0.594 | 1.535 | 1.183 | 2.089 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.124 | 0.033 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | CSQ-8 | own_base | GRPOExp3_LA0_Base | True | 96 | 0.411 | 0.397 | 0.965 | 0.678 | 1.306 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.128 | 0.108 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.196 | 0.155 | 0.788 | 0.391 | 1.153 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.568 | 0.460 | 0.810 | 0.661 | 1.004 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.552 | 0.373 | 0.676 | 0.501 | 0.870 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.464 | 0.347 | 0.749 | 0.533 | 0.986 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.448 | 0.186 | 0.415 | -0.000 | 0.680 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.561 | 0.366 | 0.653 | 0.476 | 0.834 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.583 | 0.396 | 0.679 | 0.522 | 0.846 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.193 | -0.142 | -0.739 | -1.426 | 0.176 | False | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MI-SAT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.622 | 0.304 | 0.489 | 0.317 | 0.628 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.060 | 0.091 |  |  |  | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 2 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.271 | 0.185 | 0.683 | 0.295 | 1.197 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 3 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.714 | 0.495 | 0.693 | 0.546 | 0.854 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 4 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.815 | 0.393 | 0.482 | 0.365 | 0.600 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 5 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.716 | 0.198 | 0.276 | 0.125 | 0.416 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 6 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.786 | 0.253 | 0.321 | 0.189 | 0.440 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 7 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.878 | 0.302 | 0.344 | 0.238 | 0.440 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 8 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 1.010 | 0.331 | 0.327 | 0.224 | 0.421 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 9 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.714 | 0.091 | 0.128 | -0.059 | 0.267 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 10 | MITI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.698 | 0.211 | 0.302 | 0.146 | 0.462 | True | 0.150 |
| GRPO_LA0 | GRPO | 0 | 1 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | -0.007 | 0.013 |  |  |  | False | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.016 | 0.027 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.056 | 0.075 | 1.326 | 0.925 | 1.608 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.065 | 0.060 | 0.925 | 0.627 | 1.268 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.035 | 0.029 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.042 | 0.028 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.063 | 0.058 | 0.908 | 0.576 | 1.249 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.085 | 0.070 | 0.832 | 0.570 | 1.116 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | -0.037 | -0.062 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | PCT | own_base | GRPOExp3_LA0_Base | True | 96 | 0.087 | 0.087 | 0.997 | 0.755 | 1.299 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 1 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.016 | -0.011 |  |  |  | False | 0.050 |
| GRPO_LA0 | GRPO | 0 | 2 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | -0.042 | -0.075 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 3 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.019 | 0.024 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 4 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.039 | 0.124 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 5 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.066 | 0.245 | 3.713 | 2.183 | 5.057 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 6 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.043 | 0.228 |  |  |  | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 7 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.059 | 0.271 | 4.626 | 2.278 | 5.846 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 8 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.324 | 0.515 | 1.589 | 1.297 | 2.000 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 9 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.139 | 0.312 | 2.247 | 1.557 | 3.748 | True | 0.050 |
| GRPO_LA0 | GRPO | 0 | 10 | MICI | own_base | GRPOExp3_LA0_Base | True | 96 | 0.626 | 0.666 | 1.063 | 0.944 | 1.210 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 0 | Q1Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.040 | 0.001 |  |  |  | False | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.268 | 0.213 | 0.792 | 0.434 | 1.229 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.432 | 0.308 | 0.713 | 0.498 | 1.025 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.859 | 0.752 | 0.876 | 0.757 | 1.042 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.117 | 0.950 | 0.851 | 0.751 | 0.968 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.038 | 0.964 | 0.928 | 0.803 | 1.081 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | Q1 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.035 | -0.002 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.227 | 0.192 | 0.844 | 0.334 | 1.479 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.415 | 0.288 | 0.693 | 0.403 | 1.019 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.860 | 0.840 | 0.976 | 0.832 | 1.155 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.154 | 1.131 | 0.980 | 0.860 | 1.132 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.042 | 1.127 | 1.082 | 0.936 | 1.271 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.045 | 0.004 |  |  |  | False | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.309 | 0.233 | 0.754 | 0.474 | 1.172 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.449 | 0.328 | 0.731 | 0.531 | 1.068 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.857 | 0.665 | 0.776 | 0.655 | 0.938 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.079 | 0.768 | 0.712 | 0.626 | 0.820 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q2 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.035 | 0.801 | 0.774 | 0.663 | 0.920 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | WAI-SR | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.037 | 0.016 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | WAI-SR | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.144 | 0.159 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | WAI-SR | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.225 | 0.289 | 1.286 | 0.736 | 1.968 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | WAI-SR | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.501 | 0.622 | 1.243 | 0.991 | 1.612 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | WAI-SR | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.622 | 0.702 | 1.130 | 0.919 | 1.415 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | WAI-SR | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.644 | 0.661 | 1.026 | 0.837 | 1.282 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | CSQ-8 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.017 | -0.008 |  |  |  | False | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | CSQ-8 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.057 | 0.151 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | CSQ-8 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.243 | 0.219 | 0.898 | 0.371 | 1.472 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | CSQ-8 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.492 | 0.582 | 1.183 | 0.930 | 1.510 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | CSQ-8 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.582 | 0.703 | 1.208 | 0.971 | 1.495 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | CSQ-8 | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.569 | 0.637 | 1.119 | 0.901 | 1.411 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | MI-SAT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.023 | -0.024 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MI-SAT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.108 | 0.158 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MI-SAT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.306 | 0.238 | 0.778 | 0.432 | 1.227 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MI-SAT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.651 | 0.573 | 0.880 | 0.723 | 1.085 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MI-SAT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.783 | 0.686 | 0.876 | 0.731 | 1.066 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MI-SAT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.720 | 0.634 | 0.880 | 0.739 | 1.072 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | MITI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.029 | -0.021 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MITI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.151 | 0.161 | 1.069 | 0.364 | 1.493 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MITI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.388 | 0.250 | 0.644 | 0.360 | 0.991 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MITI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.716 | 0.398 | 0.556 | 0.431 | 0.689 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MITI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.935 | 0.427 | 0.457 | 0.349 | 0.562 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MITI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.862 | 0.333 | 0.387 | 0.285 | 0.483 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | PCT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.011 | -0.017 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | PCT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.005 | -0.001 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | PCT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.033 | 0.037 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | PCT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.098 | 0.099 | 1.014 | 0.752 | 1.334 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | PCT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.132 | 0.128 | 0.964 | 0.782 | 1.170 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | PCT | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.097 | 0.099 | 1.025 | 0.755 | 1.404 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 0 | MICI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.032 | -0.044 |  |  |  | False | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | MICI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.050 | -0.009 | -0.171 | -1.196 | 1.173 | False | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | MICI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.050 | -0.000 | -0.003 | -1.007 | 1.360 | False | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | MICI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.130 | 0.148 | 1.134 | 0.518 | 1.845 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | MICI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.127 | 0.192 | 1.504 | 0.910 | 2.400 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | MICI | eda_view_PTO_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.163 | 0.276 | 1.698 | 1.224 | 2.494 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 0 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.104 | -0.026 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.205 | 0.185 | 0.905 | 0.463 | 1.346 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.368 | 0.281 | 0.762 | 0.520 | 1.128 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.795 | 0.725 | 0.912 | 0.777 | 1.083 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 1.053 | 0.923 | 0.876 | 0.768 | 1.021 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.975 | 0.937 | 0.961 | 0.835 | 1.120 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.110 | -0.013 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.152 | 0.181 | 1.192 | 0.501 | 1.681 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.340 | 0.277 | 0.816 | 0.479 | 1.302 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.785 | 0.829 | 1.056 | 0.874 | 1.276 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 1.079 | 1.121 | 1.039 | 0.898 | 1.220 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.967 | 1.117 | 1.155 | 0.985 | 1.391 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.097 | -0.040 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.257 | 0.189 | 0.736 | 0.405 | 1.171 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.397 | 0.284 | 0.716 | 0.488 | 1.090 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.805 | 0.621 | 0.772 | 0.654 | 0.932 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 1.027 | 0.724 | 0.705 | 0.613 | 0.826 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q2 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.983 | 0.757 | 0.770 | 0.664 | 0.904 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.033 | -0.057 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.074 | 0.086 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.155 | 0.216 | 1.399 | 0.597 | 1.869 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.431 | 0.549 | 1.276 | 1.018 | 1.621 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.551 | 0.629 | 1.142 | 0.942 | 1.403 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | WAI-SR | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.574 | 0.588 | 1.024 | 0.832 | 1.279 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.007 | 0.004 |  |  |  | False | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.034 | 0.163 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.220 | 0.230 | 1.047 | 0.567 | 1.591 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.469 | 0.594 | 1.267 | 1.021 | 1.652 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.559 | 0.715 | 1.280 | 1.051 | 1.592 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | CSQ-8 | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.546 | 0.648 | 1.189 | 0.972 | 1.484 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.042 | -0.082 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.089 | 0.101 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.286 | 0.181 | 0.630 | 0.250 | 1.015 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.632 | 0.516 | 0.816 | 0.662 | 1.013 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.764 | 0.628 | 0.823 | 0.705 | 0.981 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MI-SAT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.701 | 0.576 | 0.822 | 0.701 | 0.981 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | MITI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.104 | -0.057 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MITI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.076 | 0.125 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MITI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.312 | 0.214 | 0.683 | 0.332 | 1.181 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MITI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.641 | 0.362 | 0.565 | 0.422 | 0.717 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MITI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.859 | 0.391 | 0.455 | 0.334 | 0.568 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MITI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.786 | 0.297 | 0.377 | 0.259 | 0.498 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 0 | PCT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.016 | -0.019 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | PCT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.010 | -0.004 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | PCT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.028 | 0.035 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | PCT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.092 | 0.097 | 1.049 | 0.819 | 1.356 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | PCT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.127 | 0.125 | 0.987 | 0.808 | 1.234 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | PCT | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.091 | 0.097 | 1.061 | 0.823 | 1.434 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 0 | MICI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | -0.002 | -0.057 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | MICI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.017 | -0.022 |  |  |  | False | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | MICI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.016 | -0.014 |  |  |  | False | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | MICI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.097 | 0.134 | 1.392 | 0.718 | 2.471 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | MICI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.094 | 0.178 | 1.903 | 1.126 | 3.268 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | MICI | method_LA0_base | GRPOExp3_LA0_Base | False | 96 | 0.129 | 0.263 | 2.039 | 1.400 | 3.491 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.308 | 0.212 | 0.687 | 0.384 | 1.082 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.472 | 0.307 | 0.651 | 0.451 | 0.874 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.899 | 0.751 | 0.836 | 0.728 | 0.974 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 1.157 | 0.949 | 0.820 | 0.736 | 0.934 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 1.078 | 0.963 | 0.893 | 0.784 | 1.023 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.263 | 0.194 | 0.738 | 0.329 | 1.236 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.450 | 0.290 | 0.644 | 0.371 | 0.921 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.896 | 0.842 | 0.940 | 0.800 | 1.114 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 1.190 | 1.133 | 0.953 | 0.840 | 1.093 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 1.077 | 1.129 | 1.048 | 0.913 | 1.223 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.354 | 0.230 | 0.649 | 0.393 | 0.991 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.494 | 0.325 | 0.658 | 0.479 | 0.892 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.901 | 0.661 | 0.734 | 0.626 | 0.873 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 1.124 | 0.765 | 0.680 | 0.599 | 0.789 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q2 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 1.080 | 0.797 | 0.738 | 0.645 | 0.850 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.107 | 0.143 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.188 | 0.273 | 1.458 | 0.848 | 1.950 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.464 | 0.607 | 1.309 | 1.074 | 1.673 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.584 | 0.687 | 1.175 | 0.975 | 1.474 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | WAI-SR | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.607 | 0.645 | 1.063 | 0.885 | 1.295 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.040 | 0.159 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.227 | 0.227 | 1.000 | 0.493 | 1.588 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.475 | 0.590 | 1.241 | 0.992 | 1.580 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.565 | 0.711 | 1.258 | 1.028 | 1.583 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | CSQ-8 | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.552 | 0.645 | 1.167 | 0.945 | 1.464 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.130 | 0.182 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.328 | 0.262 | 0.799 | 0.506 | 1.218 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.674 | 0.597 | 0.887 | 0.718 | 1.098 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.806 | 0.710 | 0.881 | 0.740 | 1.072 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MI-SAT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.743 | 0.658 | 0.886 | 0.751 | 1.059 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MITI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.180 | 0.182 | 1.014 | 0.425 | 1.483 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MITI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.417 | 0.271 | 0.650 | 0.426 | 0.976 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MITI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.745 | 0.419 | 0.563 | 0.445 | 0.686 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MITI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.964 | 0.448 | 0.465 | 0.372 | 0.552 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MITI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.891 | 0.354 | 0.398 | 0.309 | 0.480 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | PCT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.006 | 0.015 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | PCT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.044 | 0.054 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | PCT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.109 | 0.116 | 1.067 | 0.841 | 1.367 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | PCT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.143 | 0.144 | 1.008 | 0.838 | 1.218 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | PCT | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.107 | 0.116 | 1.077 | 0.848 | 1.420 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | MICI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.019 | 0.035 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | MICI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.019 | 0.044 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | MICI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.099 | 0.192 | 1.939 | 1.349 | 3.077 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | MICI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.096 | 0.236 | 2.454 | 1.579 | 4.104 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | MICI | method_LA5_base | GRPOExp3_LA5_Base | True | 96 | 0.131 | 0.320 | 2.440 | 1.759 | 3.802 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.308 | 0.212 | 0.687 | 0.384 | 1.082 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.472 | 0.307 | 0.651 | 0.451 | 0.874 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.899 | 0.751 | 0.836 | 0.728 | 0.974 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 1.157 | 0.949 | 0.820 | 0.736 | 0.934 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 1.078 | 0.963 | 0.893 | 0.784 | 1.023 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q1 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.263 | 0.194 | 0.738 | 0.329 | 1.236 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q1 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.450 | 0.290 | 0.644 | 0.371 | 0.921 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q1 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.896 | 0.842 | 0.940 | 0.800 | 1.114 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q1 | own_base | GRPOExp3_LA5_Base | True | 96 | 1.190 | 1.133 | 0.953 | 0.840 | 1.093 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q1 | own_base | GRPOExp3_LA5_Base | True | 96 | 1.077 | 1.129 | 1.048 | 0.913 | 1.223 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.354 | 0.230 | 0.649 | 0.393 | 0.991 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.494 | 0.325 | 0.658 | 0.479 | 0.892 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.901 | 0.661 | 0.734 | 0.626 | 0.873 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 1.124 | 0.765 | 0.680 | 0.599 | 0.789 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | Q2 | own_base | GRPOExp3_LA5_Base | True | 96 | 1.080 | 0.797 | 0.738 | 0.645 | 0.850 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | WAI-SR | own_base | GRPOExp3_LA5_Base | True | 96 | 0.107 | 0.143 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | WAI-SR | own_base | GRPOExp3_LA5_Base | True | 96 | 0.188 | 0.273 | 1.458 | 0.848 | 1.950 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | WAI-SR | own_base | GRPOExp3_LA5_Base | True | 96 | 0.464 | 0.607 | 1.309 | 1.074 | 1.673 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | WAI-SR | own_base | GRPOExp3_LA5_Base | True | 96 | 0.584 | 0.687 | 1.175 | 0.975 | 1.474 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | WAI-SR | own_base | GRPOExp3_LA5_Base | True | 96 | 0.607 | 0.645 | 1.063 | 0.885 | 1.295 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | CSQ-8 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.040 | 0.159 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | CSQ-8 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.227 | 0.227 | 1.000 | 0.493 | 1.588 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | CSQ-8 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.475 | 0.590 | 1.241 | 0.992 | 1.580 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | CSQ-8 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.565 | 0.711 | 1.258 | 1.028 | 1.583 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | CSQ-8 | own_base | GRPOExp3_LA5_Base | True | 96 | 0.552 | 0.645 | 1.167 | 0.945 | 1.464 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MI-SAT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.130 | 0.182 |  |  |  | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MI-SAT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.328 | 0.262 | 0.799 | 0.506 | 1.218 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MI-SAT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.674 | 0.597 | 0.887 | 0.718 | 1.098 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MI-SAT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.806 | 0.710 | 0.881 | 0.740 | 1.072 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MI-SAT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.743 | 0.658 | 0.886 | 0.751 | 1.059 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | MITI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.180 | 0.182 | 1.014 | 0.425 | 1.483 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 2 | MITI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.417 | 0.271 | 0.650 | 0.426 | 0.976 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 3 | MITI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.745 | 0.419 | 0.563 | 0.445 | 0.686 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 4 | MITI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.964 | 0.448 | 0.465 | 0.372 | 0.552 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 5 | MITI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.891 | 0.354 | 0.398 | 0.309 | 0.480 | True | 0.150 |
| GRPO_LA5 | GRPO | 5 | 1 | PCT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.006 | 0.015 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | PCT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.044 | 0.054 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | PCT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.109 | 0.116 | 1.067 | 0.841 | 1.367 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | PCT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.143 | 0.144 | 1.008 | 0.838 | 1.218 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | PCT | own_base | GRPOExp3_LA5_Base | True | 96 | 0.107 | 0.116 | 1.077 | 0.848 | 1.420 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 1 | MICI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.019 | 0.035 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 2 | MICI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.019 | 0.044 |  |  |  | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 3 | MICI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.099 | 0.192 | 1.939 | 1.349 | 3.077 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 4 | MICI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.096 | 0.236 | 2.454 | 1.579 | 4.104 | True | 0.050 |
| GRPO_LA5 | GRPO | 5 | 5 | MICI | own_base | GRPOExp3_LA5_Base | True | 96 | 0.131 | 0.320 | 2.440 | 1.759 | 3.802 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 1 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.263 | 0.242 | 0.919 | 0.572 | 1.476 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.466 | 0.369 | 0.793 | 0.612 | 1.067 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.815 | 0.650 | 0.799 | 0.677 | 0.951 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.007 | 0.851 | 0.845 | 0.731 | 0.989 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.014 | 0.922 | 0.909 | 0.790 | 1.055 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.154 | 1.020 | 0.884 | 0.769 | 1.027 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.129 | 0.984 | 0.871 | 0.754 | 1.014 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.220 | 1.066 | 0.873 | 0.760 | 1.005 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.238 | 1.091 | 0.882 | 0.767 | 1.015 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.259 | 1.036 | 0.823 | 0.720 | 0.947 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.250 | 0.242 | 0.967 | 0.495 | 1.598 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.483 | 0.406 | 0.841 | 0.633 | 1.175 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.779 | 0.694 | 0.890 | 0.724 | 1.086 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.977 | 0.921 | 0.942 | 0.791 | 1.119 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.000 | 0.975 | 0.975 | 0.832 | 1.145 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.121 | 1.088 | 0.970 | 0.832 | 1.141 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.052 | 0.985 | 0.937 | 0.795 | 1.120 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.177 | 1.050 | 0.892 | 0.766 | 1.036 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.198 | 1.048 | 0.875 | 0.749 | 1.026 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q1 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.217 | 0.967 | 0.795 | 0.687 | 0.930 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.276 | 0.242 | 0.876 | 0.522 | 1.418 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.449 | 0.333 | 0.742 | 0.538 | 1.024 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.850 | 0.607 | 0.714 | 0.598 | 0.869 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.037 | 0.781 | 0.753 | 0.655 | 0.881 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.028 | 0.869 | 0.846 | 0.730 | 0.992 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.186 | 0.952 | 0.803 | 0.698 | 0.936 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.205 | 0.982 | 0.814 | 0.709 | 0.961 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.263 | 1.081 | 0.856 | 0.751 | 0.986 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.278 | 1.135 | 0.888 | 0.779 | 1.019 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q2 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.302 | 1.106 | 0.849 | 0.746 | 0.977 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.050 | 0.227 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.168 | 0.344 | 2.041 | 1.171 | 2.545 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.295 | 0.485 | 1.644 | 1.161 | 2.587 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.436 | 0.639 | 1.466 | 1.165 | 1.925 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.419 | 0.674 | 1.607 | 1.272 | 2.141 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.562 | 0.732 | 1.303 | 1.066 | 1.643 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.481 | 0.749 | 1.558 | 1.271 | 1.991 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.580 | 0.766 | 1.320 | 1.070 | 1.653 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.692 | 0.827 | 1.196 | 0.995 | 1.468 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.653 | 0.838 | 1.283 | 1.064 | 1.580 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.017 | 0.229 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.142 | 0.335 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.292 | 0.500 | 1.714 | 1.187 | 2.718 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.367 | 0.622 | 1.695 | 1.258 | 2.646 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.365 | 0.668 | 1.832 | 1.381 | 2.783 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.432 | 0.685 | 1.584 | 1.276 | 2.134 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.417 | 0.690 | 1.656 | 1.293 | 2.319 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.480 | 0.729 | 1.518 | 1.203 | 2.045 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.596 | 0.806 | 1.352 | 1.105 | 1.700 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.561 | 0.767 | 1.367 | 1.098 | 1.729 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.069 | 0.137 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.233 | 0.269 | 1.157 | 0.696 | 1.743 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.405 | 0.378 | 0.936 | 0.666 | 1.495 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.542 | 0.521 | 0.962 | 0.761 | 1.301 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.535 | 0.512 | 0.958 | 0.749 | 1.292 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.693 | 0.578 | 0.835 | 0.687 | 1.016 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.609 | 0.540 | 0.886 | 0.709 | 1.159 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.717 | 0.573 | 0.799 | 0.644 | 1.000 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.797 | 0.635 | 0.797 | 0.659 | 0.980 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.816 | 0.609 | 0.747 | 0.611 | 0.905 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.206 | 0.122 | 0.595 | 0.185 | 1.034 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.365 | 0.260 | 0.714 | 0.437 | 1.203 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.586 | 0.438 | 0.747 | 0.571 | 1.013 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.792 | 0.560 | 0.707 | 0.590 | 0.866 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.828 | 0.542 | 0.654 | 0.538 | 0.797 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.958 | 0.570 | 0.595 | 0.484 | 0.729 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.935 | 0.549 | 0.588 | 0.468 | 0.729 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.039 | 0.555 | 0.534 | 0.438 | 0.646 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.109 | 0.557 | 0.502 | 0.414 | 0.592 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | MITI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 1.141 | 0.513 | 0.450 | 0.357 | 0.548 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | -0.029 | -0.007 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 2 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.002 | 0.023 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 3 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.045 | 0.052 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 4 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.049 | 0.056 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 5 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.066 | 0.063 | 0.960 | 0.635 | 1.306 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 6 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.080 | 0.093 | 1.153 | 0.887 | 1.512 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 7 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.056 | 0.062 | 1.104 | 0.744 | 1.446 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 8 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.084 | 0.085 | 1.017 | 0.746 | 1.360 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 9 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.123 | 0.118 | 0.960 | 0.757 | 1.265 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 10 | PCT | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.141 | 0.113 | 0.802 | 0.642 | 0.978 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 1 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | -0.004 | -0.004 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 2 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.007 | -0.018 |  |  |  | False | 0.050 |
| PTO_LA0 | PTO | 0 | 3 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.039 | 0.053 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 4 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.001 | 0.025 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 5 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.076 | 0.139 | 1.832 | 0.938 | 2.956 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 6 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.105 | 0.217 | 2.072 | 1.278 | 3.522 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 7 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.128 | 0.254 | 1.988 | 1.379 | 3.272 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 8 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.136 | 0.307 | 2.263 | 1.581 | 3.706 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 9 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.258 | 0.382 | 1.483 | 1.197 | 1.881 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 10 | MICI | method_LA0_base | PTOExp3_LA0_Base | True | 96 | 0.278 | 0.461 | 1.657 | 1.303 | 2.151 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 0 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.003 | -0.004 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.260 | 0.238 | 0.914 | 0.564 | 1.380 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.463 | 0.366 | 0.789 | 0.573 | 1.101 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.812 | 0.647 | 0.797 | 0.674 | 0.947 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.004 | 0.847 | 0.843 | 0.737 | 0.972 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.011 | 0.918 | 0.908 | 0.794 | 1.054 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.151 | 1.016 | 0.883 | 0.777 | 1.017 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.126 | 0.980 | 0.870 | 0.748 | 1.002 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.217 | 1.062 | 0.872 | 0.767 | 1.000 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.235 | 1.087 | 0.880 | 0.771 | 1.018 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.256 | 1.032 | 0.822 | 0.723 | 0.948 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 0 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.023 | -0.010 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.227 | 0.231 | 1.018 | 0.526 | 1.608 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.460 | 0.396 | 0.860 | 0.571 | 1.280 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.756 | 0.683 | 0.904 | 0.736 | 1.116 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.954 | 0.910 | 0.954 | 0.809 | 1.123 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.977 | 0.965 | 0.987 | 0.841 | 1.170 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.098 | 1.077 | 0.981 | 0.836 | 1.152 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.029 | 0.975 | 0.947 | 0.801 | 1.119 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.154 | 1.040 | 0.901 | 0.774 | 1.043 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.175 | 1.037 | 0.883 | 0.762 | 1.035 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q1 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.194 | 0.956 | 0.801 | 0.690 | 0.941 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 0 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.017 | 0.002 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.294 | 0.244 | 0.833 | 0.524 | 1.321 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.466 | 0.335 | 0.720 | 0.512 | 1.039 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.867 | 0.610 | 0.703 | 0.588 | 0.852 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.055 | 0.783 | 0.743 | 0.651 | 0.864 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.045 | 0.872 | 0.834 | 0.720 | 0.983 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.203 | 0.955 | 0.793 | 0.698 | 0.913 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.222 | 0.984 | 0.805 | 0.697 | 0.947 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.281 | 1.084 | 0.846 | 0.744 | 0.975 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.295 | 1.137 | 0.878 | 0.768 | 1.021 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q2 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.319 | 1.108 | 0.840 | 0.737 | 0.971 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 0 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.025 | -0.032 |  |  |  | False | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.076 | 0.195 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.194 | 0.312 | 1.610 | 0.997 | 2.151 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.320 | 0.453 | 1.415 | 0.983 | 2.275 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.461 | 0.607 | 1.316 | 1.057 | 1.807 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.444 | 0.641 | 1.443 | 1.174 | 1.921 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.587 | 0.700 | 1.192 | 0.992 | 1.454 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.506 | 0.717 | 1.417 | 1.172 | 1.851 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.605 | 0.734 | 1.212 | 1.025 | 1.502 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.717 | 0.795 | 1.109 | 0.933 | 1.340 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.678 | 0.806 | 1.188 | 0.997 | 1.462 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 0 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.046 | -0.057 |  |  |  | False | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.062 | 0.172 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.188 | 0.277 | 1.479 | 0.847 | 2.093 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.337 | 0.443 | 1.313 | 0.946 | 1.949 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.413 | 0.565 | 1.369 | 1.062 | 1.948 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.410 | 0.611 | 1.489 | 1.176 | 1.986 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.478 | 0.628 | 1.313 | 1.048 | 1.684 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.462 | 0.633 | 1.369 | 1.111 | 1.778 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.526 | 0.672 | 1.277 | 1.038 | 1.667 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.642 | 0.749 | 1.166 | 0.983 | 1.403 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.607 | 0.710 | 1.170 | 0.974 | 1.439 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 0 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.002 | -0.000 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.068 | 0.137 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.231 | 0.269 | 1.165 | 0.721 | 1.731 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.403 | 0.378 | 0.940 | 0.672 | 1.454 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.540 | 0.521 | 0.965 | 0.759 | 1.306 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.533 | 0.512 | 0.961 | 0.776 | 1.260 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.691 | 0.578 | 0.837 | 0.677 | 1.028 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.608 | 0.540 | 0.889 | 0.704 | 1.159 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.715 | 0.573 | 0.801 | 0.654 | 0.993 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.795 | 0.635 | 0.799 | 0.677 | 0.949 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.814 | 0.609 | 0.748 | 0.626 | 0.894 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 0 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.016 | -0.013 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.190 | 0.109 | 0.575 | 0.047 | 1.084 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.349 | 0.247 | 0.709 | 0.441 | 1.184 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.570 | 0.424 | 0.744 | 0.562 | 1.029 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.776 | 0.547 | 0.705 | 0.563 | 0.898 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.812 | 0.529 | 0.651 | 0.535 | 0.803 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.943 | 0.557 | 0.591 | 0.485 | 0.719 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.919 | 0.536 | 0.584 | 0.462 | 0.722 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.023 | 0.542 | 0.529 | 0.430 | 0.638 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.094 | 0.544 | 0.498 | 0.404 | 0.593 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | MITI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 1.125 | 0.500 | 0.444 | 0.355 | 0.533 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 0 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.007 | -0.003 |  |  |  | False | 0.050 |
| PTO_LA0 | PTO | 0 | 1 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | -0.022 | -0.010 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 2 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.010 | 0.020 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 3 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.052 | 0.049 | 0.935 | 0.522 | 1.276 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 4 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.056 | 0.053 | 0.946 | 0.561 | 1.299 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 5 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.073 | 0.060 | 0.818 | 0.426 | 1.195 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 6 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.088 | 0.089 | 1.018 | 0.743 | 1.402 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 7 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.064 | 0.059 | 0.925 | 0.548 | 1.299 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 8 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.091 | 0.082 | 0.899 | 0.593 | 1.223 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 9 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.130 | 0.115 | 0.880 | 0.684 | 1.158 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 10 | PCT | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.149 | 0.110 | 0.740 | 0.563 | 0.904 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 0 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.036 | -0.006 |  |  |  | False | 0.050 |
| PTO_LA0 | PTO | 0 | 1 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.032 | -0.010 |  |  |  | False | 0.050 |
| PTO_LA0 | PTO | 0 | 2 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.043 | -0.024 |  |  |  | False | 0.050 |
| PTO_LA0 | PTO | 0 | 3 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.075 | 0.047 | 0.630 | -0.427 | 1.617 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 4 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.037 | 0.019 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 5 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.111 | 0.133 | 1.191 | 0.647 | 1.990 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 6 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.141 | 0.211 | 1.503 | 0.956 | 2.274 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 7 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.163 | 0.248 | 1.517 | 1.049 | 2.206 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 8 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.171 | 0.301 | 1.756 | 1.315 | 2.425 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 9 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.293 | 0.376 | 1.282 | 1.009 | 1.586 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 10 | MICI | method_LA5_base | PTOExp3_LA5_Base | False | 96 | 0.314 | 0.455 | 1.450 | 1.166 | 1.815 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 1 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 0.263 | 0.242 | 0.919 | 0.572 | 1.476 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 0.466 | 0.369 | 0.793 | 0.612 | 1.067 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 0.815 | 0.650 | 0.799 | 0.677 | 0.951 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.007 | 0.851 | 0.845 | 0.731 | 0.989 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.014 | 0.922 | 0.909 | 0.790 | 1.055 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.154 | 1.020 | 0.884 | 0.769 | 1.027 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.129 | 0.984 | 0.871 | 0.754 | 1.014 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.220 | 1.066 | 0.873 | 0.760 | 1.005 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.238 | 1.091 | 0.882 | 0.767 | 1.015 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q1Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.259 | 1.036 | 0.823 | 0.720 | 0.947 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 0.250 | 0.242 | 0.967 | 0.495 | 1.598 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 0.483 | 0.406 | 0.841 | 0.633 | 1.175 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 0.779 | 0.694 | 0.890 | 0.724 | 1.086 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 0.977 | 0.921 | 0.942 | 0.791 | 1.119 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 1.000 | 0.975 | 0.975 | 0.832 | 1.145 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 1.121 | 1.088 | 0.970 | 0.832 | 1.141 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 1.052 | 0.985 | 0.937 | 0.795 | 1.120 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 1.177 | 1.050 | 0.892 | 0.766 | 1.036 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 1.198 | 1.048 | 0.875 | 0.749 | 1.026 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q1 | own_base | PTOExp3_LA0_Base | True | 96 | 1.217 | 0.967 | 0.795 | 0.687 | 0.930 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 0.276 | 0.242 | 0.876 | 0.522 | 1.418 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 0.449 | 0.333 | 0.742 | 0.538 | 1.024 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 0.850 | 0.607 | 0.714 | 0.598 | 0.869 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.037 | 0.781 | 0.753 | 0.655 | 0.881 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.028 | 0.869 | 0.846 | 0.730 | 0.992 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.186 | 0.952 | 0.803 | 0.698 | 0.936 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.205 | 0.982 | 0.814 | 0.709 | 0.961 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.263 | 1.081 | 0.856 | 0.751 | 0.986 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.278 | 1.135 | 0.888 | 0.779 | 1.019 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | Q2 | own_base | PTOExp3_LA0_Base | True | 96 | 1.302 | 1.106 | 0.849 | 0.746 | 0.977 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.050 | 0.227 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.168 | 0.344 | 2.041 | 1.171 | 2.545 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.295 | 0.485 | 1.644 | 1.161 | 2.587 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.436 | 0.639 | 1.466 | 1.165 | 1.925 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.419 | 0.674 | 1.607 | 1.272 | 2.141 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.562 | 0.732 | 1.303 | 1.066 | 1.643 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.481 | 0.749 | 1.558 | 1.271 | 1.991 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.580 | 0.766 | 1.320 | 1.070 | 1.653 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.692 | 0.827 | 1.196 | 0.995 | 1.468 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | WAI-SR | own_base | PTOExp3_LA0_Base | True | 96 | 0.653 | 0.838 | 1.283 | 1.064 | 1.580 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.017 | 0.229 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.142 | 0.335 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.292 | 0.500 | 1.714 | 1.187 | 2.718 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.367 | 0.622 | 1.695 | 1.258 | 2.646 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.365 | 0.668 | 1.832 | 1.381 | 2.783 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.432 | 0.685 | 1.584 | 1.276 | 2.134 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.417 | 0.690 | 1.656 | 1.293 | 2.319 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.480 | 0.729 | 1.518 | 1.203 | 2.045 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.596 | 0.806 | 1.352 | 1.105 | 1.700 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | CSQ-8 | own_base | PTOExp3_LA0_Base | True | 96 | 0.561 | 0.767 | 1.367 | 1.098 | 1.729 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.069 | 0.137 |  |  |  | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.233 | 0.269 | 1.157 | 0.696 | 1.743 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.405 | 0.378 | 0.936 | 0.666 | 1.495 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.542 | 0.521 | 0.962 | 0.761 | 1.301 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.535 | 0.512 | 0.958 | 0.749 | 1.292 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.693 | 0.578 | 0.835 | 0.687 | 1.016 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.609 | 0.540 | 0.886 | 0.709 | 1.159 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.717 | 0.573 | 0.799 | 0.644 | 1.000 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.797 | 0.635 | 0.797 | 0.659 | 0.980 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | MI-SAT | own_base | PTOExp3_LA0_Base | True | 96 | 0.816 | 0.609 | 0.747 | 0.611 | 0.905 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 0.206 | 0.122 | 0.595 | 0.185 | 1.034 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 2 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 0.365 | 0.260 | 0.714 | 0.437 | 1.203 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 3 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 0.586 | 0.438 | 0.747 | 0.571 | 1.013 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 4 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 0.792 | 0.560 | 0.707 | 0.590 | 0.866 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 5 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 0.828 | 0.542 | 0.654 | 0.538 | 0.797 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 6 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 0.958 | 0.570 | 0.595 | 0.484 | 0.729 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 7 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 0.935 | 0.549 | 0.588 | 0.468 | 0.729 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 8 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 1.039 | 0.555 | 0.534 | 0.438 | 0.646 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 9 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 1.109 | 0.557 | 0.502 | 0.414 | 0.592 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 10 | MITI | own_base | PTOExp3_LA0_Base | True | 96 | 1.141 | 0.513 | 0.450 | 0.357 | 0.548 | True | 0.150 |
| PTO_LA0 | PTO | 0 | 1 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | -0.029 | -0.007 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 2 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.002 | 0.023 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 3 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.045 | 0.052 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 4 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.049 | 0.056 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 5 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.066 | 0.063 | 0.960 | 0.635 | 1.306 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 6 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.080 | 0.093 | 1.153 | 0.887 | 1.512 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 7 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.056 | 0.062 | 1.104 | 0.744 | 1.446 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 8 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.084 | 0.085 | 1.017 | 0.746 | 1.360 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 9 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.123 | 0.118 | 0.960 | 0.757 | 1.265 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 10 | PCT | own_base | PTOExp3_LA0_Base | True | 96 | 0.141 | 0.113 | 0.802 | 0.642 | 0.978 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 1 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | -0.004 | -0.004 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 2 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.007 | -0.018 |  |  |  | False | 0.050 |
| PTO_LA0 | PTO | 0 | 3 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.039 | 0.053 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 4 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.001 | 0.025 |  |  |  | True | 0.050 |
| PTO_LA0 | PTO | 0 | 5 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.076 | 0.139 | 1.832 | 0.938 | 2.956 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 6 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.105 | 0.217 | 2.072 | 1.278 | 3.522 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 7 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.128 | 0.254 | 1.988 | 1.379 | 3.272 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 8 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.136 | 0.307 | 2.263 | 1.581 | 3.706 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 9 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.258 | 0.382 | 1.483 | 1.197 | 1.881 | True | 0.050 |
| PTO_LA0 | PTO | 0 | 10 | MICI | own_base | PTOExp3_LA0_Base | True | 96 | 0.278 | 0.461 | 1.657 | 1.303 | 2.151 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 0 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.003 | 0.004 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.180 | 0.182 | 1.008 | 0.469 | 1.497 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.308 | 0.233 | 0.759 | 0.462 | 1.177 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.674 | 0.520 | 0.772 | 0.608 | 0.972 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.888 | 0.728 | 0.820 | 0.698 | 0.989 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.016 | 0.749 | 0.737 | 0.631 | 0.866 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.897 | 0.677 | 0.755 | 0.631 | 0.914 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.085 | 0.906 | 0.835 | 0.726 | 0.977 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.144 | 0.880 | 0.770 | 0.658 | 0.895 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.197 | 0.904 | 0.755 | 0.656 | 0.871 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q1Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.306 | 0.837 | 0.641 | 0.550 | 0.734 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 0 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.023 | 0.010 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.181 | 0.190 | 1.046 | 0.438 | 1.570 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.340 | 0.233 | 0.687 | 0.286 | 1.164 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.679 | 0.552 | 0.813 | 0.606 | 1.082 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.912 | 0.802 | 0.879 | 0.720 | 1.082 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.056 | 0.865 | 0.819 | 0.685 | 0.977 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.948 | 0.775 | 0.818 | 0.663 | 1.002 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.102 | 1.035 | 0.940 | 0.807 | 1.101 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.169 | 0.985 | 0.843 | 0.719 | 0.981 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.210 | 1.002 | 0.828 | 0.710 | 0.966 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q1 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.302 | 0.931 | 0.715 | 0.606 | 0.831 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 0 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.017 | -0.002 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.180 | 0.174 | 0.969 | 0.406 | 1.457 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.276 | 0.233 | 0.847 | 0.525 | 1.344 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.669 | 0.488 | 0.731 | 0.572 | 0.950 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.863 | 0.653 | 0.757 | 0.646 | 0.909 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.976 | 0.634 | 0.649 | 0.543 | 0.774 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.846 | 0.579 | 0.685 | 0.552 | 0.855 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.068 | 0.776 | 0.726 | 0.627 | 0.870 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.118 | 0.775 | 0.693 | 0.582 | 0.823 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.183 | 0.806 | 0.681 | 0.590 | 0.789 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q2 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.311 | 0.743 | 0.567 | 0.482 | 0.655 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 0 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.025 | 0.032 |  |  |  | False | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.080 | 0.128 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.183 | 0.295 | 1.611 | 0.913 | 2.248 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.330 | 0.449 | 1.361 | 0.976 | 2.068 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.454 | 0.603 | 1.329 | 1.049 | 1.786 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.516 | 0.652 | 1.262 | 1.005 | 1.623 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.525 | 0.634 | 1.207 | 0.961 | 1.561 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.522 | 0.691 | 1.324 | 1.079 | 1.702 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.555 | 0.724 | 1.305 | 1.056 | 1.682 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.654 | 0.774 | 1.185 | 1.001 | 1.433 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | WAI-SR | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.691 | 0.734 | 1.062 | 0.879 | 1.299 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 0 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.046 | 0.057 |  |  |  | False | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.027 | 0.135 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.099 | 0.268 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.234 | 0.460 | 1.961 | 1.314 | 2.829 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.341 | 0.611 | 1.790 | 1.347 | 2.737 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.438 | 0.714 | 1.631 | 1.275 | 2.222 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.397 | 0.609 | 1.534 | 1.172 | 2.189 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.451 | 0.732 | 1.624 | 1.273 | 2.223 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.467 | 0.759 | 1.624 | 1.298 | 2.263 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.491 | 0.747 | 1.523 | 1.219 | 2.015 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | CSQ-8 | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.569 | 0.766 | 1.346 | 1.102 | 1.687 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 0 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.002 | 0.000 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.101 | 0.102 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.250 | 0.174 | 0.694 | 0.243 | 1.219 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.429 | 0.378 | 0.883 | 0.634 | 1.303 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.524 | 0.538 | 1.026 | 0.813 | 1.439 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.661 | 0.547 | 0.827 | 0.648 | 1.047 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.649 | 0.516 | 0.794 | 0.626 | 1.014 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.682 | 0.630 | 0.924 | 0.762 | 1.164 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.764 | 0.658 | 0.861 | 0.712 | 1.061 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.740 | 0.691 | 0.934 | 0.766 | 1.153 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | MI-SAT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.873 | 0.720 | 0.825 | 0.697 | 0.984 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 0 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.016 | 0.013 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.078 | 0.135 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.260 | 0.138 | 0.530 | 0.200 | 0.914 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.479 | 0.276 | 0.576 | 0.385 | 0.871 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.672 | 0.398 | 0.593 | 0.452 | 0.788 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.797 | 0.336 | 0.422 | 0.309 | 0.538 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.747 | 0.208 | 0.279 | 0.144 | 0.416 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.883 | 0.375 | 0.425 | 0.318 | 0.535 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.977 | 0.391 | 0.400 | 0.299 | 0.500 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.021 | 0.349 | 0.342 | 0.251 | 0.438 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | MITI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 1.125 | 0.310 | 0.275 | 0.189 | 0.361 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 0 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.007 | 0.003 |  |  |  | False | 0.050 |
| PTO_LA5 | PTO | 5 | 1 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.002 | 0.023 |  |  |  | False | 0.050 |
| PTO_LA5 | PTO | 5 | 2 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.006 | 0.038 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 3 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.043 | 0.062 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 4 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.067 | 0.103 | 1.529 | 1.120 | 1.967 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 5 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.086 | 0.110 | 1.277 | 0.975 | 1.696 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 6 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.074 | 0.095 | 1.287 | 0.956 | 1.707 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 7 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.076 | 0.093 | 1.218 | 0.910 | 1.607 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 8 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.099 | 0.110 | 1.119 | 0.892 | 1.442 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 9 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.105 | 0.140 | 1.332 | 1.060 | 1.796 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 10 | PCT | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.149 | 0.165 | 1.105 | 0.938 | 1.332 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 0 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | -0.036 | 0.006 |  |  |  | False | 0.050 |
| PTO_LA5 | PTO | 5 | 1 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.009 | 0.033 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 2 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.017 | 0.044 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 3 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.035 | 0.133 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 4 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.113 | 0.202 | 1.793 | 1.109 | 2.920 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 5 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.112 | 0.257 | 2.298 | 1.487 | 3.783 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 6 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.107 | 0.269 | 2.508 | 1.667 | 4.097 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 7 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.050 | 0.217 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 8 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.077 | 0.236 | 3.063 | 1.990 | 4.494 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 9 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.056 | 0.247 | 4.397 | 2.307 | 5.369 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 10 | MICI | method_LA0_base | PTOExp3_LA0_Base | False | 96 | 0.051 | 0.217 | 4.288 | 2.065 | 5.096 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 1 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.178 | 0.178 | 1.002 | 0.489 | 1.403 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.305 | 0.229 | 0.753 | 0.481 | 1.178 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.671 | 0.516 | 0.769 | 0.607 | 0.953 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.885 | 0.724 | 0.818 | 0.693 | 0.973 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.013 | 0.745 | 0.735 | 0.627 | 0.863 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.894 | 0.673 | 0.753 | 0.625 | 0.920 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.082 | 0.902 | 0.833 | 0.731 | 0.960 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.141 | 0.876 | 0.768 | 0.666 | 0.897 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.194 | 0.900 | 0.754 | 0.652 | 0.866 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q1Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.303 | 0.833 | 0.639 | 0.551 | 0.738 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.158 | 0.179 | 1.132 | 0.441 | 1.587 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.317 | 0.223 | 0.704 | 0.288 | 1.209 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.656 | 0.542 | 0.825 | 0.612 | 1.072 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.890 | 0.792 | 0.890 | 0.732 | 1.082 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.033 | 0.854 | 0.827 | 0.692 | 0.981 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.925 | 0.765 | 0.827 | 0.670 | 1.012 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.079 | 1.025 | 0.950 | 0.816 | 1.114 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.146 | 0.975 | 0.851 | 0.725 | 0.996 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.188 | 0.992 | 0.835 | 0.711 | 0.978 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q1 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.279 | 0.921 | 0.720 | 0.614 | 0.837 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.197 | 0.176 | 0.897 | 0.480 | 1.346 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.293 | 0.236 | 0.805 | 0.512 | 1.271 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.686 | 0.491 | 0.716 | 0.583 | 0.878 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.880 | 0.656 | 0.745 | 0.627 | 0.890 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.993 | 0.636 | 0.640 | 0.533 | 0.771 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.863 | 0.581 | 0.674 | 0.541 | 0.831 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.085 | 0.778 | 0.717 | 0.625 | 0.839 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.135 | 0.777 | 0.684 | 0.585 | 0.804 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.200 | 0.808 | 0.673 | 0.587 | 0.785 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q2 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.328 | 0.746 | 0.562 | 0.476 | 0.652 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.105 | 0.096 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.208 | 0.263 | 1.262 | 0.736 | 1.850 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.355 | 0.417 | 1.174 | 0.850 | 1.764 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.479 | 0.571 | 1.192 | 0.942 | 1.607 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.542 | 0.620 | 1.144 | 0.932 | 1.447 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.550 | 0.602 | 1.093 | 0.890 | 1.354 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.547 | 0.659 | 1.205 | 0.980 | 1.544 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.580 | 0.692 | 1.193 | 0.988 | 1.513 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.679 | 0.742 | 1.093 | 0.922 | 1.308 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | WAI-SR | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.716 | 0.701 | 0.979 | 0.807 | 1.193 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.073 | 0.078 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.145 | 0.211 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.280 | 0.402 | 1.437 | 1.001 | 2.105 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.387 | 0.553 | 1.431 | 1.073 | 2.137 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.483 | 0.656 | 1.358 | 1.090 | 1.762 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.443 | 0.552 | 1.247 | 0.981 | 1.625 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.496 | 0.674 | 1.360 | 1.097 | 1.703 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.513 | 0.702 | 1.368 | 1.133 | 1.737 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.536 | 0.690 | 1.286 | 1.063 | 1.595 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | CSQ-8 | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.615 | 0.708 | 1.153 | 0.939 | 1.437 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.099 | 0.102 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.248 | 0.174 | 0.699 | 0.273 | 1.156 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.427 | 0.378 | 0.886 | 0.632 | 1.297 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.523 | 0.538 | 1.030 | 0.783 | 1.451 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.660 | 0.547 | 0.829 | 0.667 | 1.053 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.648 | 0.516 | 0.796 | 0.646 | 0.993 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.681 | 0.630 | 0.926 | 0.789 | 1.120 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.762 | 0.658 | 0.863 | 0.731 | 1.036 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.738 | 0.691 | 0.936 | 0.789 | 1.140 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | MI-SAT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.872 | 0.720 | 0.827 | 0.709 | 0.967 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.062 | 0.122 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.245 | 0.125 | 0.511 | 0.133 | 0.906 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.464 | 0.263 | 0.567 | 0.351 | 0.856 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.656 | 0.385 | 0.587 | 0.440 | 0.767 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.781 | 0.323 | 0.413 | 0.286 | 0.543 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.732 | 0.195 | 0.267 | 0.127 | 0.406 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.867 | 0.362 | 0.417 | 0.312 | 0.519 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.961 | 0.378 | 0.393 | 0.285 | 0.492 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.005 | 0.336 | 0.334 | 0.241 | 0.431 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | MITI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 1.109 | 0.297 | 0.268 | 0.181 | 0.354 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.005 | 0.020 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 2 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.013 | 0.035 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 3 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.050 | 0.059 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 4 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.075 | 0.100 | 1.334 | 0.921 | 1.833 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 5 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.093 | 0.107 | 1.141 | 0.863 | 1.531 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 6 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.081 | 0.092 | 1.130 | 0.791 | 1.547 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 7 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.084 | 0.090 | 1.071 | 0.721 | 1.468 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 8 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.106 | 0.107 | 1.010 | 0.773 | 1.289 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 9 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.113 | 0.137 | 1.216 | 0.923 | 1.657 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 10 | PCT | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.156 | 0.161 | 1.032 | 0.851 | 1.220 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 1 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.045 | 0.027 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 2 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.053 | 0.038 | 0.709 | -0.367 | 1.810 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 3 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.070 | 0.127 | 1.800 | 0.693 | 3.157 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 4 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.148 | 0.196 | 1.320 | 0.804 | 2.080 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 5 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.148 | 0.251 | 1.701 | 1.213 | 2.434 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 6 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.143 | 0.263 | 1.839 | 1.261 | 2.793 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 7 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.085 | 0.211 | 2.466 | 1.652 | 3.706 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 8 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.113 | 0.230 | 2.039 | 1.368 | 3.257 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 9 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.092 | 0.241 | 2.622 | 1.645 | 4.170 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 10 | MICI | method_LA5_base | PTOExp3_LA5_Base | True | 96 | 0.086 | 0.210 | 2.442 | 1.496 | 3.843 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 1 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.178 | 0.178 | 1.002 | 0.489 | 1.403 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.305 | 0.229 | 0.753 | 0.481 | 1.178 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.671 | 0.516 | 0.769 | 0.607 | 0.953 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.885 | 0.724 | 0.818 | 0.693 | 0.973 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.013 | 0.745 | 0.735 | 0.627 | 0.863 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.894 | 0.673 | 0.753 | 0.625 | 0.920 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.082 | 0.902 | 0.833 | 0.731 | 0.960 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.141 | 0.876 | 0.768 | 0.666 | 0.897 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.194 | 0.900 | 0.754 | 0.652 | 0.866 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q1Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.303 | 0.833 | 0.639 | 0.551 | 0.738 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 0.158 | 0.179 | 1.132 | 0.441 | 1.587 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 0.317 | 0.223 | 0.704 | 0.288 | 1.209 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 0.656 | 0.542 | 0.825 | 0.612 | 1.072 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 0.890 | 0.792 | 0.890 | 0.732 | 1.082 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 1.033 | 0.854 | 0.827 | 0.692 | 0.981 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 0.925 | 0.765 | 0.827 | 0.670 | 1.012 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 1.079 | 1.025 | 0.950 | 0.816 | 1.114 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 1.146 | 0.975 | 0.851 | 0.725 | 0.996 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 1.188 | 0.992 | 0.835 | 0.711 | 0.978 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q1 | own_base | PTOExp3_LA5_Base | True | 96 | 1.279 | 0.921 | 0.720 | 0.614 | 0.837 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.197 | 0.176 | 0.897 | 0.480 | 1.346 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.293 | 0.236 | 0.805 | 0.512 | 1.271 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.686 | 0.491 | 0.716 | 0.583 | 0.878 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.880 | 0.656 | 0.745 | 0.627 | 0.890 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.993 | 0.636 | 0.640 | 0.533 | 0.771 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 0.863 | 0.581 | 0.674 | 0.541 | 0.831 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.085 | 0.778 | 0.717 | 0.625 | 0.839 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.135 | 0.777 | 0.684 | 0.585 | 0.804 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.200 | 0.808 | 0.673 | 0.587 | 0.785 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | Q2 | own_base | PTOExp3_LA5_Base | True | 96 | 1.328 | 0.746 | 0.562 | 0.476 | 0.652 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.105 | 0.096 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.208 | 0.263 | 1.262 | 0.736 | 1.850 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.355 | 0.417 | 1.174 | 0.850 | 1.764 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.479 | 0.571 | 1.192 | 0.942 | 1.607 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.542 | 0.620 | 1.144 | 0.932 | 1.447 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.550 | 0.602 | 1.093 | 0.890 | 1.354 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.547 | 0.659 | 1.205 | 0.980 | 1.544 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.580 | 0.692 | 1.193 | 0.988 | 1.513 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.679 | 0.742 | 1.093 | 0.922 | 1.308 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | WAI-SR | own_base | PTOExp3_LA5_Base | True | 96 | 0.716 | 0.701 | 0.979 | 0.807 | 1.193 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.073 | 0.078 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.145 | 0.211 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.280 | 0.402 | 1.437 | 1.001 | 2.105 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.387 | 0.553 | 1.431 | 1.073 | 2.137 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.483 | 0.656 | 1.358 | 1.090 | 1.762 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.443 | 0.552 | 1.247 | 0.981 | 1.625 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.496 | 0.674 | 1.360 | 1.097 | 1.703 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.513 | 0.702 | 1.368 | 1.133 | 1.737 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.536 | 0.690 | 1.286 | 1.063 | 1.595 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | CSQ-8 | own_base | PTOExp3_LA5_Base | True | 96 | 0.615 | 0.708 | 1.153 | 0.939 | 1.437 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.099 | 0.102 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.248 | 0.174 | 0.699 | 0.273 | 1.156 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.427 | 0.378 | 0.886 | 0.632 | 1.297 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.523 | 0.538 | 1.030 | 0.783 | 1.451 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.660 | 0.547 | 0.829 | 0.667 | 1.053 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.648 | 0.516 | 0.796 | 0.646 | 0.993 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.681 | 0.630 | 0.926 | 0.789 | 1.120 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.762 | 0.658 | 0.863 | 0.731 | 1.036 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.738 | 0.691 | 0.936 | 0.789 | 1.140 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | MI-SAT | own_base | PTOExp3_LA5_Base | True | 96 | 0.872 | 0.720 | 0.827 | 0.709 | 0.967 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.062 | 0.122 |  |  |  | True | 0.150 |
| PTO_LA5 | PTO | 5 | 2 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.245 | 0.125 | 0.511 | 0.133 | 0.906 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 3 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.464 | 0.263 | 0.567 | 0.351 | 0.856 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 4 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.656 | 0.385 | 0.587 | 0.440 | 0.767 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 5 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.781 | 0.323 | 0.413 | 0.286 | 0.543 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 6 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.732 | 0.195 | 0.267 | 0.127 | 0.406 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 7 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.867 | 0.362 | 0.417 | 0.312 | 0.519 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 8 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 0.961 | 0.378 | 0.393 | 0.285 | 0.492 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 9 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 1.005 | 0.336 | 0.334 | 0.241 | 0.431 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 10 | MITI | own_base | PTOExp3_LA5_Base | True | 96 | 1.109 | 0.297 | 0.268 | 0.181 | 0.354 | True | 0.150 |
| PTO_LA5 | PTO | 5 | 1 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.005 | 0.020 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 2 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.013 | 0.035 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 3 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.050 | 0.059 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 4 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.075 | 0.100 | 1.334 | 0.921 | 1.833 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 5 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.093 | 0.107 | 1.141 | 0.863 | 1.531 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 6 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.081 | 0.092 | 1.130 | 0.791 | 1.547 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 7 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.084 | 0.090 | 1.071 | 0.721 | 1.468 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 8 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.106 | 0.107 | 1.010 | 0.773 | 1.289 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 9 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.113 | 0.137 | 1.216 | 0.923 | 1.657 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 10 | PCT | own_base | PTOExp3_LA5_Base | True | 96 | 0.156 | 0.161 | 1.032 | 0.851 | 1.220 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 1 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.045 | 0.027 |  |  |  | True | 0.050 |
| PTO_LA5 | PTO | 5 | 2 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.053 | 0.038 | 0.709 | -0.367 | 1.810 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 3 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.070 | 0.127 | 1.800 | 0.693 | 3.157 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 4 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.148 | 0.196 | 1.320 | 0.804 | 2.080 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 5 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.148 | 0.251 | 1.701 | 1.213 | 2.434 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 6 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.143 | 0.263 | 1.839 | 1.261 | 2.793 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 7 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.085 | 0.211 | 2.466 | 1.652 | 3.706 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 8 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.113 | 0.230 | 2.039 | 1.368 | 3.257 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 9 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.092 | 0.241 | 2.622 | 1.645 | 4.170 | True | 0.050 |
| PTO_LA5 | PTO | 5 | 10 | MICI | own_base | PTOExp3_LA5_Base | True | 96 | 0.086 | 0.210 | 2.442 | 1.496 | 3.843 | True | 0.050 |
