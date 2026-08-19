> **Excerpt — first 60 of 1,224 rows.** The full table is too large to read as markdown, so it lives on sheet `k_paired_channels` of the `.xlsx` workbook in this folder. Load it with `pandas.read_excel(..., sheet_name="k_paired_channels")`.

| judge       | method   | family                     |   iteration | metric                       |   n |   mean_delta |      dz |       p |   p_holm |
|:------------|:---------|:---------------------------|------------:|:-----------------------------|----:|-------------:|--------:|--------:|---------:|
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_Severity                |  96 |        0.219 |   0.161 |   0.061 |    0.488 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_Rate                    |  96 |        0.036 |   0.123 |   0.302 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_Confront_rate           |  96 |        0.003 |   0.089 |   0.450 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_AdviseNoPermission_rate |  96 |        0.012 |   0.064 |   0.716 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_Warn_rate               |  96 |        0.005 |   0.166 |   0.068 |    0.488 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_Direct_rate             |  96 |        0.015 |   0.107 |   0.260 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_Judge_rate              |  96 |       -0.004 |  -0.073 |   0.635 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           0 | MICI_OverPraise_rate         |  96 |        0.005 |   0.081 |   0.499 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_Severity                |  96 |       -0.250 |  -0.193 |   0.043 |    0.344 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_Rate                    |  96 |       -0.013 |  -0.043 |   0.974 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_Confront_rate           |  96 |        0.000 |   0.015 |   1.000 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_AdviseNoPermission_rate |  96 |       -0.013 |  -0.058 |   0.692 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_Warn_rate               |  96 |        0.005 |   0.178 |   0.068 |    0.475 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_Direct_rate             |  96 |       -0.006 |  -0.048 |   0.882 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_Judge_rate              |  96 |        0.003 |   0.110 |   0.397 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           1 | MICI_OverPraise_rate         |  96 |       -0.002 |  -0.014 |   0.647 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_Severity                |  96 |       -0.188 |  -0.162 |   0.180 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_Rate                    |  96 |       -0.010 |  -0.042 |   0.793 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_Confront_rate           |  96 |       -0.000 |  -0.005 |   0.959 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_AdviseNoPermission_rate |  96 |        0.018 |   0.104 |   0.448 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_Warn_rate               |  96 |       -0.000 |  -0.020 |   0.865 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_Direct_rate             |  96 |       -0.024 |  -0.228 |   0.038 |    0.304 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_Judge_rate              |  96 |       -0.003 |  -0.083 |   0.476 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           2 | MICI_OverPraise_rate         |  96 |       -0.000 |  -0.006 |   0.968 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_Severity                |  96 |       -0.271 |  -0.231 |   0.028 |    0.222 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_Rate                    |  96 |        0.004 |   0.015 |   0.543 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_Confront_rate           |  96 |        0.000 |   0.011 |   0.854 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_AdviseNoPermission_rate |  96 |        0.004 |   0.019 |   0.386 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_Warn_rate               |  96 |       -0.001 |  -0.102 |   0.317 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_Direct_rate             |  96 |        0.005 |   0.045 |   0.850 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_Judge_rate              |  96 |       -0.002 |  -0.118 |   0.257 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           3 | MICI_OverPraise_rate         |  96 |       -0.002 |  -0.030 |   0.922 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_Severity                |  96 |       -0.448 |  -0.495 |   0.000 |    0.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_Rate                    |  96 |       -0.111 |  -0.400 |   0.000 |    0.002 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_Confront_rate           |  96 |       -0.004 |  -0.221 |   0.043 |    0.172 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_AdviseNoPermission_rate |  96 |       -0.068 |  -0.363 |   0.000 |    0.003 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_Warn_rate               |  96 |       -0.002 |  -0.144 |   0.180 |    0.359 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_Direct_rate             |  96 |       -0.024 |  -0.231 |   0.019 |    0.095 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_Judge_rate              |  96 |       -0.003 |  -0.177 |   0.109 |    0.326 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           4 | MICI_OverPraise_rate         |  96 |       -0.009 |  -0.114 |   0.257 |    0.359 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_Severity                |  96 |       -0.281 |  -0.272 |   0.008 |    0.059 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_Rate                    |  96 |       -0.036 |  -0.139 |   0.219 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_Confront_rate           |  96 |       -0.000 |  -0.004 |   0.953 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_AdviseNoPermission_rate |  96 |       -0.036 |  -0.191 |   0.022 |    0.132 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_Warn_rate               |  96 |        0.000 | nan     | nan     |  nan     |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_Direct_rate             |  96 |       -0.005 |  -0.048 |   0.854 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_Judge_rate              |  96 |        0.002 |   0.082 |   0.400 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           5 | MICI_OverPraise_rate         |  96 |        0.003 |   0.042 |   0.797 |    1.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_Severity                |  96 |       -0.479 |  -0.498 |   0.000 |    0.000 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_Rate                    |  96 |       -0.002 |  -0.009 |   0.972 |    0.972 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_Confront_rate           |  96 |       -0.003 |  -0.189 |   0.066 |    0.328 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_AdviseNoPermission_rate |  96 |       -0.017 |  -0.108 |   0.270 |    0.539 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_Warn_rate               |  96 |       -0.002 |  -0.139 |   0.180 |    0.539 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_Direct_rate             |  96 |       -0.019 |  -0.164 |   0.024 |    0.142 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_Judge_rate              |  96 |       -0.002 |  -0.156 |   0.102 |    0.410 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           6 | MICI_OverPraise_rate         |  96 |        0.041 |   0.333 |   0.002 |    0.011 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           7 | MICI_Severity                |  96 |       -0.240 |  -0.272 |   0.009 |    0.044 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           7 | MICI_Rate                    |  96 |        0.078 |   0.285 |   0.006 |    0.035 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           7 | MICI_Confront_rate           |  96 |       -0.005 |  -0.220 |   0.042 |    0.169 |
| gpt-4o-mini | PTO      | MI-inconsistent (per turn) |           7 | MICI_AdviseNoPermission_rate |  96 |       -0.013 |  -0.082 |   0.451 |    0.902 |

_... 1,164 further rows in the workbook._
