# Held-out personas: alcohol, a problem the training grid never had

*Generated 2026-10-08 by `tools/heldout_check.py` (rerunnable; a re-render never touches this file). 48 personas (2 gender x 3 cooperation x 2 duration x 2 prior attempts x 2 ages, problem = alcohol; `system_prompts_builder.generate_heldout_permutations`), simulated by four states with one patient seed (`HELDOUT_SEED`), scored by both graders on the eight instruments and the utterance coder. Persona-paired on conversation id (the list is not shuffled). Sign A - B. Holm across the nine instrument rows, and separately across the process rows, per (contrast, grader). MICI and the rows marked (lower) are lower-is-better.*

## Levels (means over conversations)

### training oracle

| measure | Base | K=0 at 8 | K=0 at 10 | K=5 at 10 |
|---|---:|---:|---:|---:|
| Q1Q2 | 2.986 | 4.003 | 3.763 | 4.361 |
| Q1 | 2.942 | 3.846 | 3.592 | 4.296 |
| Q2 | 3.029 | 4.159 | 3.935 | 4.426 |
| WAI-SR | 2.852 | 3.286 | 3.332 | 3.575 |
| CSQ-8 | 2.409 | 2.753 | 2.727 | 2.974 |
| MI-SAT | 2.913 | 3.375 | 3.368 | 3.649 |
| MITI | 3.094 | 4.214 | 3.964 | 4.396 |
| PCT | 0.565 | 0.573 | 0.587 | 0.704 |
| MICI (lower) | 0.279 | 0.606 | 0.833 | 0.311 |
| non-specific praise, share of therapist turns | 0.047 | 0.231 | 0.479 | 0.052 |
| complex reflection, share of therapist turns | 0.019 | 0.043 | 0.012 | 0.252 |
| persuasion, share of therapist turns (lower) | 0.191 | 0.173 | 0.024 | 0.271 |
| MI-consistent share of therapist turns | 0.272 | 0.331 | 0.260 | 0.475 |
| MI-inconsistent share of therapist turns (lower) | 0.249 | 0.404 | 0.502 | 0.323 |
| reflection after change talk | 0.148 | 0.032 | 0.010 | 0.332 |
| praise after sustain talk (lower) | 0.011 | 0.126 | 0.370 | 0.000 |
| persuasion after sustain talk (lower) | 0.227 | 0.285 | 0.030 | 0.438 |
| change talk after change talk (persistence) | 0.806 | 0.879 | 0.829 | 0.957 |
| change talk after sustain talk | 0.275 | 0.210 | 0.230 | 0.257 |

### held-out judge

| measure | Base | K=0 at 8 | K=0 at 10 | K=5 at 10 |
|---|---:|---:|---:|---:|
| Q1Q2 | 1.814 | 2.571 | 2.287 | 2.374 |
| Q1 | 1.604 | 2.183 | 1.892 | 2.108 |
| Q2 | 2.025 | 2.960 | 2.683 | 2.640 |
| WAI-SR | 2.215 | 2.885 | 2.670 | 2.747 |
| CSQ-8 | 2.125 | 2.659 | 2.339 | 2.688 |
| MI-SAT | 2.424 | 2.854 | 2.729 | 3.208 |
| MITI | 1.917 | 2.182 | 2.005 | 2.156 |
| PCT | 0.537 | 0.564 | 0.581 | 0.691 |
| MICI (lower) | 0.397 | 0.956 | 1.084 | 0.707 |
| non-specific praise, share of therapist turns | 0.037 | 0.557 | 0.776 | 0.114 |
| complex reflection, share of therapist turns | 0.011 | 0.016 | 0.010 | 0.291 |
| persuasion, share of therapist turns (lower) | 0.224 | 0.138 | 0.029 | 0.335 |
| MI-consistent share of therapist turns | 0.271 | 0.078 | 0.037 | 0.389 |
| MI-inconsistent share of therapist turns (lower) | 0.272 | 0.695 | 0.805 | 0.449 |
| reflection after change talk | 0.018 | 0.042 | 0.005 | 0.337 |
| praise after sustain talk (lower) | 0.045 | 0.384 | 0.747 | 0.014 |
| persuasion after sustain talk (lower) | 0.324 | 0.289 | 0.046 | 0.655 |
| change talk after change talk (persistence) | 0.731 | 0.753 | 0.695 | 0.923 |
| change talk after sustain talk | 0.125 | 0.187 | 0.221 | 0.221 |

## Instruments: persona-paired contrasts

### K=5 at 10 - K=0 at 8 (best)

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| Q1Q2 | training oracle | 48 | +0.359 | +0.628 | [+0.208, +0.518] | 8.17e-05 | 0.0006 |
| Q1 | training oracle | 48 | +0.450 | +0.681 | [+0.279, +0.637] | 5.59e-05 | 0.0004 |
| Q2 | training oracle | 48 | +0.267 | +0.527 | [+0.130, +0.404] | 4.16e-04 | 0.0025 |
| WAI-SR | training oracle | 48 | +0.288 | +0.598 | [+0.163, +0.425] | 5.95e-04 | 0.0027 |
| CSQ-8 | training oracle | 48 | +0.221 | +0.470 | [+0.104, +0.352] | 2.10e-03 | 0.0063 |
| MI-SAT | training oracle | 48 | +0.274 | +0.407 | [+0.101, +0.465] | 1.30e-02 | 0.0260 |
| MITI | training oracle | 48 | +0.182 | +0.315 | [+0.026, +0.339] | 2.51e-02 | 0.0260 |
| PCT | training oracle | 48 | +0.130 | +0.557 | [+0.066, +0.196] | 5.40e-04 | 0.0027 |
| MICI | training oracle | 48 | -0.295 | -0.942 | [-0.385, -0.213] | 2.77e-07 | 0.0000 |
| Q1Q2 | held-out judge | 48 | -0.197 | -0.336 | [-0.355, -0.032] | 2.59e-02 | 0.1295 |
| Q1 | held-out judge | 48 | -0.075 | -0.095 | [-0.292, +0.150] | 4.90e-01 | 0.9794 |
| Q2 | held-out judge | 48 | -0.320 | -0.541 | [-0.475, -0.143] | 8.43e-04 | 0.0051 |
| WAI-SR | held-out judge | 48 | -0.139 | -0.271 | [-0.280, +0.000] | 1.03e-01 | 0.4104 |
| CSQ-8 | held-out judge | 48 | +0.029 | +0.042 | [-0.151, +0.214] | 6.91e-01 | 0.9794 |
| MI-SAT | held-out judge | 48 | +0.354 | +0.656 | [+0.212, +0.510] | 1.11e-04 | 0.0009 |
| MITI | held-out judge | 48 | -0.026 | -0.056 | [-0.151, +0.109] | 3.16e-01 | 0.9485 |
| PCT | held-out judge | 48 | +0.127 | +0.559 | [+0.063, +0.191] | 4.73e-04 | 0.0033 |
| MICI | held-out judge | 48 | -0.249 | -0.752 | [-0.342, -0.154] | 1.02e-05 | 0.0001 |

### K=5 at 10 - K=0 at 10 (last)

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| Q1Q2 | training oracle | 48 | +0.598 | +0.797 | [+0.398, +0.813] | 3.60e-06 | 0.0000 |
| Q1 | training oracle | 48 | +0.704 | +0.859 | [+0.487, +0.938] | 5.34e-06 | 0.0000 |
| Q2 | training oracle | 48 | +0.491 | +0.673 | [+0.297, +0.699] | 7.82e-05 | 0.0005 |
| WAI-SR | training oracle | 48 | +0.243 | +0.471 | [+0.104, +0.389] | 2.88e-03 | 0.0115 |
| CSQ-8 | training oracle | 48 | +0.247 | +0.438 | [+0.099, +0.411] | 6.10e-03 | 0.0122 |
| MI-SAT | training oracle | 48 | +0.281 | +0.370 | [+0.076, +0.486] | 1.68e-02 | 0.0168 |
| MITI | training oracle | 48 | +0.432 | +0.645 | [+0.250, +0.599] | 1.18e-04 | 0.0006 |
| PCT | training oracle | 48 | +0.117 | +0.395 | [+0.036, +0.197] | 3.16e-03 | 0.0115 |
| MICI | training oracle | 48 | -0.522 | -1.476 | [-0.623, -0.422] | 3.47e-09 | 0.0000 |
| Q1Q2 | held-out judge | 48 | +0.087 | +0.158 | [-0.064, +0.239] | 2.95e-01 | 0.8844 |
| Q1 | held-out judge | 48 | +0.217 | +0.296 | [+0.017, +0.421] | 6.21e-02 | 0.3106 |
| Q2 | held-out judge | 48 | -0.043 | -0.080 | [-0.185, +0.110] | 4.44e-01 | 0.8844 |
| WAI-SR | held-out judge | 48 | +0.076 | +0.129 | [-0.082, +0.236] | 4.09e-01 | 0.8844 |
| CSQ-8 | held-out judge | 48 | +0.349 | +0.539 | [+0.164, +0.521] | 1.32e-03 | 0.0093 |
| MI-SAT | held-out judge | 48 | +0.479 | +0.728 | [+0.302, +0.653] | 2.52e-05 | 0.0002 |
| MITI | held-out judge | 48 | +0.151 | +0.304 | [+0.016, +0.292] | 8.25e-02 | 0.3301 |
| PCT | held-out judge | 48 | +0.111 | +0.413 | [+0.037, +0.188] | 6.42e-03 | 0.0385 |
| MICI | held-out judge | 48 | -0.377 | -1.105 | [-0.465, -0.284] | 1.49e-07 | 0.0000 |

### K=5 at 10 - Base

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| Q1Q2 | training oracle | 48 | +1.376 | +1.368 | [+1.100, +1.656] | 2.38e-09 | 0.0000 |
| Q1 | training oracle | 48 | +1.354 | +1.349 | [+1.083, +1.629] | 1.61e-08 | 0.0000 |
| Q2 | training oracle | 48 | +1.397 | +1.329 | [+1.105, +1.683] | 3.14e-09 | 0.0000 |
| WAI-SR | training oracle | 48 | +0.722 | +0.964 | [+0.509, +0.925] | 5.20e-07 | 0.0000 |
| CSQ-8 | training oracle | 48 | +0.565 | +0.778 | [+0.357, +0.760] | 1.39e-05 | 0.0000 |
| MI-SAT | training oracle | 48 | +0.736 | +0.804 | [+0.483, +0.976] | 7.25e-06 | 0.0000 |
| MITI | training oracle | 48 | +1.302 | +1.294 | [+1.016, +1.573] | 1.43e-08 | 0.0000 |
| PCT | training oracle | 48 | +0.139 | +0.513 | [+0.066, +0.214] | 9.03e-04 | 0.0018 |
| MICI | training oracle | 48 | +0.031 | +0.097 | [-0.058, +0.116] | 3.91e-01 | 0.3911 |
| Q1Q2 | held-out judge | 48 | +0.560 | +0.758 | [+0.356, +0.764] | 1.43e-05 | 0.0001 |
| Q1 | held-out judge | 48 | +0.504 | +0.587 | [+0.275, +0.742] | 1.98e-04 | 0.0006 |
| Q2 | held-out judge | 48 | +0.615 | +0.772 | [+0.397, +0.837] | 7.39e-06 | 0.0001 |
| WAI-SR | held-out judge | 48 | +0.531 | +0.824 | [+0.352, +0.712] | 8.77e-06 | 0.0001 |
| CSQ-8 | held-out judge | 48 | +0.562 | +0.726 | [+0.336, +0.771] | 6.03e-05 | 0.0002 |
| MI-SAT | held-out judge | 48 | +0.785 | +1.025 | [+0.569, +0.986] | 4.61e-07 | 0.0000 |
| MITI | held-out judge | 48 | +0.240 | +0.424 | [+0.089, +0.406] | 5.89e-03 | 0.0059 |
| PCT | held-out judge | 48 | +0.155 | +0.574 | [+0.082, +0.228] | 4.07e-04 | 0.0008 |
| MICI | held-out judge | 48 | +0.310 | +0.794 | [+0.202, +0.422] | 1.58e-05 | 0.0001 |

### K=0 at 10 - Base

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| Q1Q2 | training oracle | 48 | +0.778 | +0.888 | [+0.533, +1.041] | 1.67e-06 | 0.0000 |
| Q1 | training oracle | 48 | +0.650 | +0.713 | [+0.396, +0.912] | 5.26e-05 | 0.0003 |
| Q2 | training oracle | 48 | +0.906 | +0.991 | [+0.645, +1.181] | 2.67e-07 | 0.0000 |
| WAI-SR | training oracle | 48 | +0.479 | +0.672 | [+0.274, +0.670] | 8.32e-05 | 0.0003 |
| CSQ-8 | training oracle | 48 | +0.318 | +0.462 | [+0.120, +0.495] | 1.99e-03 | 0.0040 |
| MI-SAT | training oracle | 48 | +0.455 | +0.556 | [+0.226, +0.677] | 5.52e-04 | 0.0017 |
| MITI | training oracle | 48 | +0.870 | +0.959 | [+0.609, +1.125] | 9.62e-07 | 0.0000 |
| PCT | training oracle | 48 | +0.022 | +0.081 | [-0.052, +0.095] | 6.88e-01 | 0.6876 |
| MICI | training oracle | 48 | +0.553 | +1.148 | [+0.417, +0.680] | 6.11e-08 | 0.0000 |
| Q1Q2 | held-out judge | 48 | +0.473 | +0.792 | [+0.303, +0.637] | 9.61e-06 | 0.0001 |
| Q1 | held-out judge | 48 | +0.288 | +0.431 | [+0.100, +0.475] | 4.27e-03 | 0.0213 |
| Q2 | held-out judge | 48 | +0.658 | +1.017 | [+0.474, +0.832] | 4.10e-07 | 0.0000 |
| WAI-SR | held-out judge | 48 | +0.455 | +0.695 | [+0.269, +0.632] | 3.23e-05 | 0.0002 |
| CSQ-8 | held-out judge | 48 | +0.214 | +0.275 | [-0.008, +0.430] | 2.60e-02 | 0.0779 |
| MI-SAT | held-out judge | 48 | +0.306 | +0.476 | [+0.122, +0.486] | 5.19e-03 | 0.0213 |
| MITI | held-out judge | 48 | +0.089 | +0.183 | [-0.047, +0.219] | 2.23e-01 | 0.4467 |
| PCT | held-out judge | 48 | +0.044 | +0.160 | [-0.032, +0.122] | 2.43e-01 | 0.4467 |
| MICI | held-out judge | 48 | +0.687 | +1.694 | [+0.578, +0.799] | 2.87e-09 | 0.0000 |

### K=0 at 8 - Base

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| Q1Q2 | training oracle | 48 | +1.017 | +1.176 | [+0.777, +1.248] | 1.79e-08 | 0.0000 |
| Q1 | training oracle | 48 | +0.904 | +1.101 | [+0.679, +1.133] | 1.00e-07 | 0.0000 |
| Q2 | training oracle | 48 | +1.130 | +1.163 | [+0.846, +1.400] | 1.69e-08 | 0.0000 |
| WAI-SR | training oracle | 48 | +0.434 | +0.738 | [+0.266, +0.589] | 1.66e-05 | 0.0001 |
| CSQ-8 | training oracle | 48 | +0.344 | +0.580 | [+0.174, +0.503] | 3.55e-04 | 0.0007 |
| MI-SAT | training oracle | 48 | +0.462 | +0.635 | [+0.264, +0.656] | 6.15e-05 | 0.0002 |
| MITI | training oracle | 48 | +1.120 | +1.227 | [+0.854, +1.365] | 1.61e-08 | 0.0000 |
| PCT | training oracle | 48 | +0.009 | +0.035 | [-0.063, +0.075] | 9.67e-01 | 0.9672 |
| MICI | training oracle | 48 | +0.327 | +0.934 | [+0.227, +0.425] | 1.29e-06 | 0.0000 |
| Q1Q2 | held-out judge | 48 | +0.757 | +1.040 | [+0.558, +0.963] | 1.69e-07 | 0.0000 |
| Q1 | held-out judge | 48 | +0.579 | +0.738 | [+0.375, +0.800] | 4.53e-05 | 0.0002 |
| Q2 | held-out judge | 48 | +0.935 | +1.231 | [+0.722, +1.150] | 2.14e-08 | 0.0000 |
| WAI-SR | held-out judge | 48 | +0.670 | +0.991 | [+0.481, +0.856] | 4.17e-07 | 0.0000 |
| CSQ-8 | held-out judge | 48 | +0.534 | +0.771 | [+0.328, +0.716] | 3.69e-05 | 0.0002 |
| MI-SAT | held-out judge | 48 | +0.431 | +0.697 | [+0.260, +0.601] | 4.89e-05 | 0.0002 |
| MITI | held-out judge | 48 | +0.266 | +0.514 | [+0.125, +0.406] | 1.51e-03 | 0.0030 |
| PCT | held-out judge | 48 | +0.028 | +0.121 | [-0.036, +0.092] | 3.56e-01 | 0.3564 |
| MICI | held-out judge | 48 | +0.559 | +1.419 | [+0.448, +0.665] | 1.18e-08 | 0.0000 |

## Process (utterance coder): persona-paired contrasts

### K=5 at 10 - K=0 at 8 (best)

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| non-specific praise, share of therapist turns | training oracle | 48 | -0.179 | -0.839 | [-0.237, -0.119] | 6.66e-06 | 0.0001 |
| complex reflection, share of therapist turns | training oracle | 48 | +0.209 | +0.915 | [+0.152, +0.271] | 7.80e-08 | 0.0000 |
| persuasion, share of therapist turns | training oracle | 48 | +0.098 | +0.350 | [+0.017, +0.176] | 1.26e-02 | 0.0557 |
| MI-consistent share of therapist turns | training oracle | 48 | +0.145 | +0.421 | [+0.046, +0.240] | 1.11e-02 | 0.0557 |
| MI-inconsistent share of therapist turns | training oracle | 48 | -0.081 | -0.263 | [-0.164, +0.009] | 6.12e-02 | 0.1224 |
| reflection after change talk | training oracle | 38 | +0.294 | +0.900 | [+0.190, +0.398] | 4.64e-05 | 0.0004 |
| praise after sustain talk | training oracle | 32 | -0.126 | -0.741 | [-0.186, -0.070] | 6.37e-04 | 0.0045 |
| persuasion after sustain talk | training oracle | 32 | +0.153 | +0.383 | [+0.018, +0.290] | 3.05e-02 | 0.0916 |
| change talk after change talk (persistence) | training oracle | 38 | +0.107 | +0.422 | [+0.034, +0.192] | 5.59e-03 | 0.0335 |
| change talk after sustain talk | training oracle | 32 | +0.047 | +0.268 | [-0.010, +0.110] | 1.64e-01 | 0.1641 |
| non-specific praise, share of therapist turns | held-out judge | 48 | -0.443 | -2.000 | [-0.502, -0.383] | 1.61e-09 | 0.0000 |
| complex reflection, share of therapist turns | held-out judge | 48 | +0.275 | +0.916 | [+0.193, +0.356] | 2.27e-07 | 0.0000 |
| persuasion, share of therapist turns | held-out judge | 48 | +0.197 | +0.781 | [+0.129, +0.267] | 4.97e-06 | 0.0000 |
| MI-consistent share of therapist turns | held-out judge | 48 | +0.310 | +0.967 | [+0.220, +0.396] | 4.73e-07 | 0.0000 |
| MI-inconsistent share of therapist turns | held-out judge | 48 | -0.246 | -0.701 | [-0.340, -0.154] | 4.08e-05 | 0.0002 |
| reflection after change talk | held-out judge | 38 | +0.313 | +0.786 | [+0.184, +0.439] | 6.09e-05 | 0.0002 |
| praise after sustain talk | held-out judge | 32 | -0.371 | -1.682 | [-0.448, -0.301] | 2.51e-06 | 0.0000 |
| persuasion after sustain talk | held-out judge | 32 | +0.366 | +0.942 | [+0.230, +0.492] | 1.60e-04 | 0.0005 |
| change talk after change talk (persistence) | held-out judge | 38 | +0.142 | +0.561 | [+0.068, +0.226] | 1.22e-03 | 0.0024 |
| change talk after sustain talk | held-out judge | 32 | +0.034 | +0.150 | [-0.043, +0.108] | 4.50e-01 | 0.4504 |

### K=5 at 10 - K=0 at 10 (last)

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| non-specific praise, share of therapist turns | training oracle | 48 | -0.426 | -1.412 | [-0.511, -0.340] | 4.26e-09 | 0.0000 |
| complex reflection, share of therapist turns | training oracle | 48 | +0.240 | +1.051 | [+0.181, +0.304] | 4.05e-08 | 0.0000 |
| persuasion, share of therapist turns | training oracle | 48 | +0.247 | +1.024 | [+0.183, +0.314] | 3.47e-07 | 0.0000 |
| MI-consistent share of therapist turns | training oracle | 48 | +0.215 | +0.711 | [+0.132, +0.300] | 1.59e-05 | 0.0001 |
| MI-inconsistent share of therapist turns | training oracle | 48 | -0.179 | -0.541 | [-0.269, -0.083] | 1.11e-03 | 0.0033 |
| reflection after change talk | training oracle | 37 | +0.332 | +1.077 | [+0.234, +0.430] | 1.23e-05 | 0.0001 |
| praise after sustain talk | training oracle | 32 | -0.366 | -1.345 | [-0.460, -0.280] | 8.16e-06 | 0.0001 |
| persuasion after sustain talk | training oracle | 32 | +0.407 | +1.331 | [+0.300, +0.509] | 1.22e-05 | 0.0001 |
| change talk after change talk (persistence) | training oracle | 37 | +0.101 | +0.360 | [+0.019, +0.198] | 4.13e-02 | 0.0826 |
| change talk after sustain talk | training oracle | 32 | +0.024 | +0.106 | [-0.059, +0.093] | 1.33e-01 | 0.1331 |
| non-specific praise, share of therapist turns | held-out judge | 48 | -0.661 | -3.128 | [-0.717, -0.601] | 1.63e-09 | 0.0000 |
| complex reflection, share of therapist turns | held-out judge | 48 | +0.281 | +0.973 | [+0.202, +0.364] | 5.23e-08 | 0.0000 |
| persuasion, share of therapist turns | held-out judge | 48 | +0.306 | +1.190 | [+0.239, +0.374] | 7.59e-09 | 0.0000 |
| MI-consistent share of therapist turns | held-out judge | 48 | +0.352 | +1.178 | [+0.268, +0.433] | 1.85e-08 | 0.0000 |
| MI-inconsistent share of therapist turns | held-out judge | 48 | -0.355 | -1.130 | [-0.437, -0.265] | 9.63e-08 | 0.0000 |
| reflection after change talk | held-out judge | 38 | +0.354 | +1.102 | [+0.258, +0.454] | 1.16e-06 | 0.0000 |
| praise after sustain talk | held-out judge | 32 | -0.733 | -3.372 | [-0.805, -0.659] | 7.53e-07 | 0.0000 |
| persuasion after sustain talk | held-out judge | 32 | +0.609 | +2.249 | [+0.516, +0.702] | 1.66e-06 | 0.0000 |
| change talk after change talk (persistence) | held-out judge | 38 | +0.182 | +0.635 | [+0.099, +0.279] | 3.26e-04 | 0.0007 |
| change talk after sustain talk | held-out judge | 32 | +0.000 | +0.000 | [-0.111, +0.099] | 7.89e-01 | 0.7893 |

### K=5 at 10 - Base

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| non-specific praise, share of therapist turns | training oracle | 47 | +0.005 | +0.029 | [-0.043, +0.050] | 6.23e-01 | 0.8219 |
| complex reflection, share of therapist turns | training oracle | 47 | +0.232 | +1.008 | [+0.170, +0.299] | 6.13e-08 | 0.0000 |
| persuasion, share of therapist turns | training oracle | 47 | +0.086 | +0.271 | [-0.009, +0.174] | 2.28e-02 | 0.1139 |
| MI-consistent share of therapist turns | training oracle | 47 | +0.206 | +0.672 | [+0.124, +0.290] | 6.57e-05 | 0.0006 |
| MI-inconsistent share of therapist turns | training oracle | 47 | +0.080 | +0.231 | [-0.021, +0.174] | 5.86e-02 | 0.2345 |
| reflection after change talk | training oracle | 33 | +0.219 | +0.634 | [+0.107, +0.333] | 2.11e-03 | 0.0169 |
| praise after sustain talk | training oracle | 31 | -0.013 | -0.180 | [-0.039, +0.000] | 3.17e-01 | 0.8219 |
| persuasion after sustain talk | training oracle | 31 | +0.196 | +0.477 | [+0.054, +0.336] | 1.53e-02 | 0.0916 |
| change talk after change talk (persistence) | training oracle | 31 | +0.131 | +0.499 | [+0.050, +0.234] | 2.56e-03 | 0.0179 |
| change talk after sustain talk | training oracle | 31 | +0.058 | +0.193 | [-0.045, +0.155] | 2.74e-01 | 0.8219 |
| non-specific praise, share of therapist turns | held-out judge | 47 | +0.080 | +0.456 | [+0.035, +0.130] | 1.76e-03 | 0.0106 |
| complex reflection, share of therapist turns | held-out judge | 47 | +0.278 | +0.965 | [+0.201, +0.358] | 8.34e-08 | 0.0000 |
| persuasion, share of therapist turns | held-out judge | 47 | +0.110 | +0.270 | [-0.008, +0.223] | 3.57e-02 | 0.1047 |
| MI-consistent share of therapist turns | held-out judge | 47 | +0.114 | +0.341 | [+0.018, +0.207] | 2.62e-02 | 0.1047 |
| MI-inconsistent share of therapist turns | held-out judge | 47 | +0.179 | +0.429 | [+0.053, +0.297] | 4.25e-03 | 0.0212 |
| reflection after change talk | held-out judge | 33 | +0.314 | +0.907 | [+0.204, +0.429] | 4.16e-05 | 0.0004 |
| praise after sustain talk | held-out judge | 31 | -0.032 | -0.269 | [-0.078, +0.006] | 1.23e-01 | 0.1232 |
| persuasion after sustain talk | held-out judge | 31 | +0.325 | +0.715 | [+0.158, +0.479] | 9.54e-04 | 0.0067 |
| change talk after change talk (persistence) | held-out judge | 31 | +0.174 | +0.647 | [+0.089, +0.279] | 6.84e-05 | 0.0005 |
| change talk after sustain talk | held-out judge | 31 | +0.123 | +0.420 | [+0.026, +0.225] | 2.63e-02 | 0.1047 |

### K=0 at 10 - Base

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| non-specific praise, share of therapist turns | training oracle | 47 | +0.432 | +1.222 | [+0.333, +0.530] | 6.59e-08 | 0.0000 |
| complex reflection, share of therapist turns | training oracle | 47 | -0.007 | -0.148 | [-0.020, +0.005] | 4.21e-01 | 1.0000 |
| persuasion, share of therapist turns | training oracle | 47 | -0.167 | -0.658 | [-0.246, -0.099] | 4.36e-05 | 0.0003 |
| MI-consistent share of therapist turns | training oracle | 47 | -0.012 | -0.032 | [-0.120, +0.091] | 9.28e-01 | 1.0000 |
| MI-inconsistent share of therapist turns | training oracle | 47 | +0.255 | +0.596 | [+0.134, +0.371] | 3.83e-04 | 0.0027 |
| reflection after change talk | training oracle | 33 | -0.112 | -0.497 | [-0.195, -0.048] | 1.47e-03 | 0.0073 |
| praise after sustain talk | training oracle | 31 | +0.351 | +1.162 | [+0.248, +0.455] | 3.41e-05 | 0.0003 |
| persuasion after sustain talk | training oracle | 31 | -0.225 | -0.713 | [-0.339, -0.114] | 6.93e-04 | 0.0042 |
| change talk after change talk (persistence) | training oracle | 31 | +0.034 | +0.105 | [-0.077, +0.151] | 4.21e-01 | 1.0000 |
| change talk after sustain talk | training oracle | 31 | +0.039 | +0.120 | [-0.071, +0.149] | 2.14e-01 | 0.8560 |
| non-specific praise, share of therapist turns | held-out judge | 47 | +0.744 | +4.095 | [+0.691, +0.793] | 2.34e-09 | 0.0000 |
| complex reflection, share of therapist turns | held-out judge | 47 | -0.001 | -0.017 | [-0.014, +0.012] | 9.69e-01 | 1.0000 |
| persuasion, share of therapist turns | held-out judge | 47 | -0.194 | -0.685 | [-0.284, -0.118] | 1.59e-05 | 0.0001 |
| MI-consistent share of therapist turns | held-out judge | 47 | -0.235 | -0.752 | [-0.336, -0.147] | 3.46e-06 | 0.0000 |
| MI-inconsistent share of therapist turns | held-out judge | 47 | +0.539 | +1.530 | [+0.432, +0.632] | 1.91e-08 | 0.0000 |
| reflection after change talk | held-out judge | 33 | -0.014 | -0.141 | [-0.054, +0.014] | 4.65e-01 | 1.0000 |
| praise after sustain talk | held-out judge | 31 | +0.710 | +3.025 | [+0.629, +0.792] | 1.10e-06 | 0.0000 |
| persuasion after sustain talk | held-out judge | 31 | -0.288 | -0.828 | [-0.419, -0.175] | 1.70e-04 | 0.0008 |
| change talk after change talk (persistence) | held-out judge | 31 | -0.018 | -0.044 | [-0.166, +0.125] | 8.53e-01 | 1.0000 |
| change talk after sustain talk | held-out judge | 31 | +0.122 | +0.413 | [+0.017, +0.226] | 1.92e-02 | 0.0766 |

### K=0 at 8 - Base

| measure | grader | n | delta (A-B) | dz | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---|---:|---:|
| non-specific praise, share of therapist turns | training oracle | 47 | +0.187 | +0.897 | [+0.128, +0.245] | 3.11e-06 | 0.0000 |
| complex reflection, share of therapist turns | training oracle | 47 | +0.025 | +0.266 | [+0.001, +0.053] | 8.48e-02 | 0.5089 |
| persuasion, share of therapist turns | training oracle | 47 | -0.014 | -0.047 | [-0.103, +0.067] | 8.57e-01 | 1.0000 |
| MI-consistent share of therapist turns | training oracle | 47 | +0.061 | +0.189 | [-0.032, +0.156] | 2.00e-01 | 1.0000 |
| MI-inconsistent share of therapist turns | training oracle | 47 | +0.162 | +0.512 | [+0.068, +0.248] | 9.42e-04 | 0.0085 |
| reflection after change talk | training oracle | 33 | -0.079 | -0.358 | [-0.161, -0.013] | 1.31e-02 | 0.0919 |
| praise after sustain talk | training oracle | 31 | +0.106 | +0.613 | [+0.051, +0.171] | 2.84e-03 | 0.0227 |
| persuasion after sustain talk | training oracle | 31 | +0.038 | +0.101 | [-0.086, +0.170] | 7.37e-01 | 1.0000 |
| change talk after change talk (persistence) | training oracle | 31 | +0.049 | +0.183 | [-0.041, +0.144] | 5.56e-01 | 1.0000 |
| change talk after sustain talk | training oracle | 31 | +0.009 | +0.035 | [-0.085, +0.093] | 5.07e-01 | 1.0000 |
| non-specific praise, share of therapist turns | held-out judge | 47 | +0.526 | +2.315 | [+0.462, +0.591] | 2.29e-09 | 0.0000 |
| complex reflection, share of therapist turns | held-out judge | 47 | +0.003 | +0.060 | [-0.010, +0.018] | 9.05e-01 | 1.0000 |
| persuasion, share of therapist turns | held-out judge | 47 | -0.085 | -0.262 | [-0.184, +0.008] | 1.79e-01 | 0.8942 |
| MI-consistent share of therapist turns | held-out judge | 47 | -0.195 | -0.627 | [-0.291, -0.109] | 1.02e-04 | 0.0007 |
| MI-inconsistent share of therapist turns | held-out judge | 47 | +0.431 | +1.085 | [+0.313, +0.539] | 4.23e-07 | 0.0000 |
| reflection after change talk | held-out judge | 32 | +0.011 | +0.121 | [-0.010, +0.047] | 1.00e+00 | 1.0000 |
| praise after sustain talk | held-out judge | 31 | +0.341 | +1.194 | [+0.243, +0.435] | 1.30e-05 | 0.0001 |
| persuasion after sustain talk | held-out judge | 31 | -0.040 | -0.085 | [-0.209, +0.119] | 7.73e-01 | 1.0000 |
| change talk after change talk (persistence) | held-out judge | 30 | +0.026 | +0.078 | [-0.091, +0.141] | 3.17e-01 | 1.0000 |
| change talk after sustain talk | held-out judge | 31 | +0.091 | +0.429 | [+0.022, +0.165] | 2.61e-02 | 0.1564 |

