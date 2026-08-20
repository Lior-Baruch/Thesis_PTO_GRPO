| method   | window              | iters                | quantity                         |     K0_sum |     K5_sum |   K5_over_K0 | arithmetic                  |
|:---------|:--------------------|:---------------------|:---------------------------------|-----------:|-----------:|-------------:|:----------------------------|
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | oracle_calls_train               |  59991.000 |  59177.000 |        0.986 | 59,177 / 59,991             |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | oracle_input_Mchars              |    516.228 |    622.242 |        1.205 | 622.2 / 516.2               |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_total              |   9950.000 |  86114.000 |        8.655 | 86,114 / 9,950              |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_tail               |      0.000 |  76197.000 |      nan     | 76,197 / 0                  |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | n_candidates                     |  30040.000 |  28992.000 |        0.965 | 28,992 / 30,040             |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | total_api_calls                  |  69941.000 | 145291.000 |        2.077 | 145,291 / 69,941            |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | oracle_calls_train               |  99622.000 | 121806.000 |        1.223 | 121,806 / 99,622            |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | oracle_input_Mchars              |    987.099 |   1849.163 |        1.873 | 1,849.2 / 987.1             |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | patient_calls_total              |  17587.000 | 179634.000 |       10.214 | 179,634 / 17,587            |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | patient_calls_tail               |      0.000 | 159788.000 |      nan     | 159,788 / 0                 |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | n_candidates                     |  49920.000 |  60384.000 |        1.210 | 60,384 / 49,920             |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | total_api_calls                  | 117209.000 | 301440.000 |        2.572 | 301,440 / 117,209           |
| PTO      | all K5 iters        | 1,2,3,4,5,6,7,8,9,10 | patient_calls_tail_per_candidate |      0.000 |      2.646 |      nan     | 159,788 / 60,384 candidates |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | oracle_calls_train               | 143548.000 | 141237.000 |        0.984 | 141,237 / 143,548           |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | oracle_input_Mchars              |   1117.443 |   1519.808 |        1.360 | 1,519.8 / 1,117.4           |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_total              |   6879.000 | 197239.000 |       28.673 | 197,239 / 6,879             |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_tail               |      0.000 | 190499.000 |      nan     | 190,499 / 0                 |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | n_candidates                     |  68864.000 |  66304.000 |        0.963 | 66,304 / 68,864             |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | total_api_calls                  | 150427.000 | 338476.000 |        2.250 | 338,476 / 150,427           |
| GRPO     | all matched iters   | 1,2,3,4,5,6          | oracle_calls_train               | 175664.000 | 159989.000 |        0.911 | 159,989 / 175,664           |
| GRPO     | all matched iters   | 1,2,3,4,5,6          | oracle_input_Mchars              |   1404.796 |   1806.674 |        1.286 | 1,806.7 / 1,404.8           |
| GRPO     | all matched iters   | 1,2,3,4,5,6          | patient_calls_total              |   8351.000 | 221576.000 |       26.533 | 221,576 / 8,351             |
| GRPO     | all matched iters   | 1,2,3,4,5,6          | patient_calls_tail               |      0.000 | 213755.000 |      nan     | 213,755 / 0                 |
| GRPO     | all matched iters   | 1,2,3,4,5,6          | n_candidates                     |  83712.000 |  75264.000 |        0.899 | 75,264 / 83,712             |
| GRPO     | all matched iters   | 1,2,3,4,5,6          | total_api_calls                  | 184015.000 | 381565.000 |        2.074 | 381,565 / 184,015           |
| GRPO     | all K5 iters        | 1,2,3,4,5,6          | patient_calls_tail_per_candidate |      0.000 |      2.840 |      nan     | 213,755 / 75,264 candidates |