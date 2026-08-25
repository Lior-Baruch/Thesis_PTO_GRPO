> **Excerpt — first 60 of 396 rows.** The full table is too large to read as markdown, so it lives on sheet `k_method_gap` of the `.xlsx` workbook in this folder. Load it with `pandas.read_excel(..., sheet_name="k_method_gap")`.

| judge            |   K |   iteration | metric   | contrast                     |   n |   delta |     dz |   ci_lo |   ci_hi |     p |   p_holm |   p_holm_rubrics | favours   |
|:-----------------|----:|------------:|:---------|:-----------------------------|----:|--------:|-------:|--------:|--------:|------:|---------:|-----------------:|:----------|
| claude-haiku-4-5 |   0 |           0 | Q1Q2     | PTO_LA0_Base − GRPO_LA0_Base |  96 |  -0.031 | -0.045 |  -0.167 |   0.101 | 0.937 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           1 | Q1Q2     | PTO_LA0_I1 − GRPO_LA0_I1     |  96 |  -0.002 | -0.004 |  -0.131 |   0.117 | 0.794 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           2 | Q1Q2     | PTO_LA0_I2 − GRPO_LA0_I2     |  96 |   0.092 |  0.119 |  -0.057 |   0.246 | 0.266 |    0.840 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           3 | Q1Q2     | PTO_LA0_I3 − GRPO_LA0_I3     |  96 |  -0.156 | -0.278 |  -0.267 |  -0.053 | 0.008 |    0.038 |            0.061 | GRPO      |
| claude-haiku-4-5 |   0 |           4 | Q1Q2     | PTO_LA0_I4 − GRPO_LA0_I4     |  96 |   0.130 |  0.171 |  -0.028 |   0.283 | 0.210 |    0.840 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           5 | Q1Q2     | PTO_LA0_I5 − GRPO_LA0_I5     |  96 |   0.265 |  0.355 |   0.126 |   0.415 | 0.002 |    0.012 |            0.014 | PTO       |
| claude-haiku-4-5 |   0 |           6 | Q1Q2     | PTO_LA0_I6 − GRPO_LA0_I6     |  96 |   0.479 |  0.521 |   0.306 |   0.665 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           7 | Q1Q2     | PTO_LA0_I7 − GRPO_LA0_I7     |  96 |   0.248 |  0.326 |   0.093 |   0.401 | 0.001 |    0.004 |            0.004 | PTO       |
| claude-haiku-4-5 |   0 |           8 | Q1Q2     | PTO_LA0_I8 − GRPO_LA0_I8     |  96 |   0.278 |  0.463 |   0.156 |   0.398 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           9 | Q1Q2     | PTO_LA0_I9 − GRPO_LA0_I9     |  96 |   0.919 |  1.083 |   0.759 |   1.077 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |          10 | Q1Q2     | PTO_LA0_I10 − GRPO_LA0_I10   |  96 |   0.609 |  1.265 |   0.515 |   0.705 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           0 | Q1       | PTO_LA0_Base − GRPO_LA0_Base |  96 |  -0.021 | -0.025 |  -0.183 |   0.135 | 0.930 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           1 | Q1       | PTO_LA0_I1 − GRPO_LA0_I1     |  96 |  -0.046 | -0.062 |  -0.196 |   0.100 | 0.511 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           2 | Q1       | PTO_LA0_I2 − GRPO_LA0_I2     |  96 |   0.108 |  0.113 |  -0.085 |   0.302 | 0.240 |    0.876 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           3 | Q1       | PTO_LA0_I3 − GRPO_LA0_I3     |  96 |  -0.210 | -0.280 |  -0.356 |  -0.071 | 0.009 |    0.047 |            0.066 | GRPO      |
| claude-haiku-4-5 |   0 |           4 | Q1       | PTO_LA0_I4 − GRPO_LA0_I4     |  96 |   0.148 |  0.158 |  -0.035 |   0.344 | 0.219 |    0.876 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           5 | Q1       | PTO_LA0_I5 − GRPO_LA0_I5     |  96 |   0.288 |  0.317 |   0.112 |   0.473 | 0.005 |    0.033 |            0.027 | PTO       |
| claude-haiku-4-5 |   0 |           6 | Q1       | PTO_LA0_I6 − GRPO_LA0_I6     |  96 |   0.581 |  0.505 |   0.356 |   0.815 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           7 | Q1       | PTO_LA0_I7 − GRPO_LA0_I7     |  96 |   0.269 |  0.300 |   0.085 |   0.448 | 0.003 |    0.018 |            0.015 | PTO       |
| claude-haiku-4-5 |   0 |           8 | Q1       | PTO_LA0_I8 − GRPO_LA0_I8     |  96 |   0.398 |  0.550 |   0.248 |   0.535 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           9 | Q1       | PTO_LA0_I9 − GRPO_LA0_I9     |  96 |   1.027 |  1.025 |   0.835 |   1.211 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |          10 | Q1       | PTO_LA0_I10 − GRPO_LA0_I10   |  96 |   0.773 |  1.209 |   0.646 |   0.900 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           0 | Q2       | PTO_LA0_Base − GRPO_LA0_Base |  96 |  -0.042 | -0.063 |  -0.177 |   0.093 | 0.661 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           1 | Q2       | PTO_LA0_I1 − GRPO_LA0_I1     |  96 |   0.041 |  0.065 |  -0.088 |   0.156 | 0.609 |    1.000 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           2 | Q2       | PTO_LA0_I2 − GRPO_LA0_I2     |  96 |   0.077 |  0.109 |  -0.061 |   0.213 | 0.354 |    1.000 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           3 | Q2       | PTO_LA0_I3 − GRPO_LA0_I3     |  96 |  -0.102 | -0.198 |  -0.205 |  -0.000 | 0.099 |    0.493 |            0.296 | GRPO      |
| claude-haiku-4-5 |   0 |           4 | Q2       | PTO_LA0_I4 − GRPO_LA0_I4     |  96 |   0.112 |  0.161 |  -0.029 |   0.252 | 0.264 |    1.000 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           5 | Q2       | PTO_LA0_I5 − GRPO_LA0_I5     |  96 |   0.243 |  0.335 |   0.102 |   0.392 | 0.004 |    0.023 |            0.023 | PTO       |
| claude-haiku-4-5 |   0 |           6 | Q2       | PTO_LA0_I6 − GRPO_LA0_I6     |  96 |   0.377 |  0.474 |   0.232 |   0.532 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           7 | Q2       | PTO_LA0_I7 − GRPO_LA0_I7     |  96 |   0.227 |  0.319 |   0.085 |   0.369 | 0.001 |    0.011 |            0.010 | PTO       |
| claude-haiku-4-5 |   0 |           8 | Q2       | PTO_LA0_I8 − GRPO_LA0_I8     |  96 |   0.159 |  0.264 |   0.039 |   0.277 | 0.003 |    0.023 |            0.017 | PTO       |
| claude-haiku-4-5 |   0 |           9 | Q2       | PTO_LA0_I9 − GRPO_LA0_I9     |  96 |   0.811 |  1.027 |   0.657 |   0.961 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |          10 | Q2       | PTO_LA0_I10 − GRPO_LA0_I10   |  96 |   0.445 |  0.931 |   0.354 |   0.537 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           0 | WAI-SR   | PTO_LA0_Base − GRPO_LA0_Base |  96 |  -0.105 | -0.135 |  -0.254 |   0.043 | 0.432 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           1 | WAI-SR   | PTO_LA0_I1 − GRPO_LA0_I1     |  96 |  -0.032 | -0.048 |  -0.160 |   0.107 | 0.501 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           2 | WAI-SR   | PTO_LA0_I2 − GRPO_LA0_I2     |  96 |   0.015 |  0.019 |  -0.135 |   0.162 | 0.767 |    1.000 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           3 | WAI-SR   | PTO_LA0_I3 − GRPO_LA0_I3     |  96 |  -0.255 | -0.457 |  -0.366 |  -0.151 | 0.000 |    0.000 |            0.000 | GRPO      |
| claude-haiku-4-5 |   0 |           4 | WAI-SR   | PTO_LA0_I4 − GRPO_LA0_I4     |  96 |  -0.023 | -0.036 |  -0.154 |   0.103 | 0.712 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           5 | WAI-SR   | PTO_LA0_I5 − GRPO_LA0_I5     |  96 |   0.090 |  0.157 |  -0.025 |   0.201 | 0.108 |    0.755 |            0.233 | PTO       |
| claude-haiku-4-5 |   0 |           6 | WAI-SR   | PTO_LA0_I6 − GRPO_LA0_I6     |  96 |   0.148 |  0.216 |   0.017 |   0.280 | 0.081 |    0.648 |            0.162 | PTO       |
| claude-haiku-4-5 |   0 |           7 | WAI-SR   | PTO_LA0_I7 − GRPO_LA0_I7     |  96 |   0.058 |  0.099 |  -0.061 |   0.180 | 0.222 |    1.000 |            0.666 | PTO       |
| claude-haiku-4-5 |   0 |           8 | WAI-SR   | PTO_LA0_I8 − GRPO_LA0_I8     |  96 |  -0.012 | -0.022 |  -0.120 |   0.096 | 0.797 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           9 | WAI-SR   | PTO_LA0_I9 − GRPO_LA0_I9     |  96 |   0.608 |  0.875 |   0.473 |   0.744 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |          10 | WAI-SR   | PTO_LA0_I10 − GRPO_LA0_I10   |  96 |   0.286 |  0.476 |   0.176 |   0.412 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           0 | CSQ-8    | PTO_LA0_Base − GRPO_LA0_Base |  96 |  -0.046 | -0.067 |  -0.177 |   0.082 | 0.589 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           1 | CSQ-8    | PTO_LA0_I1 − GRPO_LA0_I1     |  96 |   0.034 |  0.048 |  -0.104 |   0.172 | 0.570 |    1.000 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           2 | CSQ-8    | PTO_LA0_I2 − GRPO_LA0_I2     |  96 |   0.083 |  0.101 |  -0.074 |   0.240 | 0.383 |    1.000 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           3 | CSQ-8    | PTO_LA0_I3 − GRPO_LA0_I3     |  96 |  -0.158 | -0.265 |  -0.276 |  -0.040 | 0.015 |    0.123 |            0.092 | GRPO      |
| claude-haiku-4-5 |   0 |           4 | CSQ-8    | PTO_LA0_I4 − GRPO_LA0_I4     |  96 |   0.105 |  0.150 |  -0.031 |   0.250 | 0.152 |    0.625 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           5 | CSQ-8    | PTO_LA0_I5 − GRPO_LA0_I5     |  96 |   0.111 |  0.185 |  -0.009 |   0.232 | 0.078 |    0.485 |            0.233 | PTO       |
| claude-haiku-4-5 |   0 |           6 | CSQ-8    | PTO_LA0_I6 − GRPO_LA0_I6     |  96 |   0.229 |  0.328 |   0.094 |   0.372 | 0.001 |    0.011 |            0.005 | PTO       |
| claude-haiku-4-5 |   0 |           7 | CSQ-8    | PTO_LA0_I7 − GRPO_LA0_I7     |  96 |   0.108 |  0.159 |  -0.029 |   0.247 | 0.069 |    0.485 |            0.347 | PTO       |
| claude-haiku-4-5 |   0 |           8 | CSQ-8    | PTO_LA0_I8 − GRPO_LA0_I8     |  96 |   0.090 |  0.163 |  -0.020 |   0.193 | 0.125 |    0.625 |            0.375 | PTO       |
| claude-haiku-4-5 |   0 |           9 | CSQ-8    | PTO_LA0_I9 − GRPO_LA0_I9     |  96 |   0.728 |  1.109 |   0.596 |   0.852 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |          10 | CSQ-8    | PTO_LA0_I10 − GRPO_LA0_I10   |  96 |   0.324 |  0.603 |   0.219 |   0.432 | 0.000 |    0.000 |            0.000 | PTO       |
| claude-haiku-4-5 |   0 |           0 | MI-SAT   | PTO_LA0_Base − GRPO_LA0_Base |  96 |  -0.057 | -0.073 |  -0.219 |   0.097 | 0.950 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           1 | MI-SAT   | PTO_LA0_I1 − GRPO_LA0_I1     |  96 |  -0.028 | -0.039 |  -0.172 |   0.115 | 0.686 |    1.000 |            1.000 | GRPO      |
| claude-haiku-4-5 |   0 |           2 | MI-SAT   | PTO_LA0_I2 − GRPO_LA0_I2     |  96 |   0.057 |  0.076 |  -0.085 |   0.208 | 0.500 |    1.000 |            1.000 | PTO       |
| claude-haiku-4-5 |   0 |           3 | MI-SAT   | PTO_LA0_I3 − GRPO_LA0_I3     |  96 |  -0.139 | -0.282 |  -0.238 |  -0.040 | 0.020 |    0.141 |            0.101 | GRPO      |
| claude-haiku-4-5 |   0 |           4 | MI-SAT   | PTO_LA0_I4 − GRPO_LA0_I4     |  96 |   0.090 |  0.144 |  -0.035 |   0.215 | 0.228 |    0.914 |            1.000 | PTO       |

_... 336 further rows in the workbook._
