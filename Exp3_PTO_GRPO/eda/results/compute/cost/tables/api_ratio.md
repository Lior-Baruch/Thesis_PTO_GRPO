| method   | window              | iters                | quantity                         |     K0_sum |     K5_sum |   K5_over_K0 | arithmetic                   |
|:---------|:--------------------|:---------------------|:---------------------------------|-----------:|-----------:|-------------:|:-----------------------------|
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | oracle_calls_train               |  59991.000 |  59177.000 |        0.986 | 59,177 / 59,991              |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | oracle_input_Mchars              |    516.228 |    622.242 |        1.205 | 622.2 / 516.2                |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_total              |   9950.000 |  86114.000 |        8.655 | 86,114 / 9,950               |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_tail               |      0.000 |  76197.000 |      nan     | 76,197 / 0                   |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | n_candidates                     |  30040.000 |  28992.000 |        0.965 | 28,992 / 30,040              |
| PTO      | iters 1-5 (matched) | 1,2,3,4,5            | total_api_calls                  |  69941.000 | 145291.000 |        2.077 | 145,291 / 69,941             |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | oracle_calls_train               |  99622.000 | 121806.000 |        1.223 | 121,806 / 99,622             |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | oracle_input_Mchars              |    987.099 |   1849.163 |        1.873 | 1,849.2 / 987.1              |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | patient_calls_total              |  17587.000 | 179634.000 |       10.214 | 179,634 / 17,587             |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | patient_calls_tail               |      0.000 | 159788.000 |      nan     | 159,788 / 0                  |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | n_candidates                     |  49920.000 |  60384.000 |        1.210 | 60,384 / 49,920              |
| PTO      | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | total_api_calls                  | 117209.000 | 301440.000 |        2.572 | 301,440 / 117,209            |
| PTO      | all K5 iters        | 1,2,3,4,5,6,7,8,9,10 | patient_calls_tail_per_candidate |      0.000 |      2.646 |      nan     | 159,788 / 60,384 candidates  |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | oracle_calls_train               | 143548.000 | 141237.000 |        0.984 | 141,237 / 143,548            |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | oracle_input_Mchars              |   1117.443 |   1519.808 |        1.360 | 1,519.8 / 1,117.4            |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_total              |   6879.000 | 197239.000 |       28.673 | 197,239 / 6,879              |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | patient_calls_tail               |      0.000 | 190499.000 |      nan     | 190,499 / 0                  |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | n_candidates                     |  68864.000 |  66304.000 |        0.963 | 66,304 / 68,864              |
| GRPO     | iters 1-5 (matched) | 1,2,3,4,5            | total_api_calls                  | 150427.000 | 338476.000 |        2.250 | 338,476 / 150,427            |
| GRPO     | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | oracle_calls_train               | 302541.000 | 289983.000 |        0.958 | 289,983 / 302,541            |
| GRPO     | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | oracle_input_Mchars              |   2666.628 |   4669.468 |        1.751 | 4,669.5 / 2,666.6            |
| GRPO     | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | patient_calls_total              |  14254.000 | 406565.000 |       28.523 | 406,565 / 14,254             |
| GRPO     | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | patient_calls_tail               |      0.000 | 392766.000 |      nan     | 392,766 / 0                  |
| GRPO     | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | n_candidates                     | 144384.000 | 136960.000 |        0.949 | 136,960 / 144,384            |
| GRPO     | all matched iters   | 1,2,3,4,5,6,7,8,9,10 | total_api_calls                  | 316795.000 | 696548.000 |        2.199 | 696,548 / 316,795            |
| GRPO     | all K5 iters        | 1,2,3,4,5,6,7,8,9,10 | patient_calls_tail_per_candidate |      0.000 |      2.868 |      nan     | 392,766 / 136,960 candidates |