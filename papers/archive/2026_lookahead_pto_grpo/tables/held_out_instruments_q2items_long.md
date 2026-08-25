Long companion of q2items: per (grader, arm, item) the persona-paired gain over the arm's own base at its endpoint (`target_iter`), with 95% percentile-bootstrap CI, dz and Wilcoxon p; `base`/`target` = arm means on the paired personas. Paired on persona_id (the recovered patient persona), never file_index. GRPO_LA5 is right-censored at iteration 5 (PTO arms and GRPO_LA0 run to 10).

| judge | arm | item | short | group | target_iter | n | base | target | gain | gain_ci_lo | gain_ci_hi | gain_dz | gain_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | PTO_LA0 | 1 | sense of who he was | Self-disclosure | 10 | 96 | 3.073 | 3.927 | 0.854 | 0.708 | 1.010 | 1.075 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 2 | revealed his thinking | Self-disclosure | 10 | 96 | 2.635 | 4.115 | 1.479 | 1.312 | 1.656 | 1.676 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 3 | shared his feelings | Self-disclosure | 10 | 96 | 2.156 | 3.385 | 1.229 | 1.094 | 1.365 | 1.828 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 4 | knew how I was feeling | Empathy/understanding | 10 | 96 | 3.333 | 4.427 | 1.094 | 0.906 | 1.292 | 1.135 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 5 | understood me | Empathy/understanding | 10 | 96 | 3.292 | 4.427 | 1.135 | 0.938 | 1.344 | 1.100 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 6 | put himself in my shoes | Empathy/understanding | 10 | 96 | 2.781 | 4.323 | 1.542 | 1.354 | 1.740 | 1.568 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 7 | comfortable talking | Fluency/ease | 10 | 96 | 3.521 | 4.729 | 1.208 | 1.021 | 1.406 | 1.179 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 8 | relaxed and secure | Fluency/ease | 10 | 96 | 3.479 | 4.719 | 1.240 | 1.042 | 1.448 | 1.188 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 9 | took charge | Direction/control | 10 | 96 | 2.729 | 4.208 | 1.479 | 1.271 | 1.688 | 1.363 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 10 | said when happy/sad | Self-disclosure | 10 | 96 | 2.302 | 3.552 | 1.250 | 1.094 | 1.406 | 1.600 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 11 | no difficulty w/ words | Fluency/ease | 10 | 96 | 3.531 | 4.740 | 1.208 | 1.010 | 1.427 | 1.134 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 12 | expressed himself | Fluency/ease | 10 | 96 | 3.500 | 4.740 | 1.240 | 1.042 | 1.448 | 1.165 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 13 | a 'warm' partner | Warmth/closeness | 10 | 96 | 3.021 | 4.333 | 1.312 | 1.083 | 1.552 | 1.137 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 14 | did not judge me | Non-judgment/equality | 10 | 96 | 3.406 | 4.833 | 1.427 | 1.146 | 1.719 | 0.985 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 15 | treated me as equal | Non-judgment/equality | 10 | 96 | 3.344 | 4.781 | 1.438 | 1.156 | 1.729 | 1.007 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 16 | made me feel cared for | Warmth/closeness | 10 | 96 | 3.167 | 4.698 | 1.531 | 1.292 | 1.792 | 1.188 | 0.000 |
| gpt-4o-mini | PTO_LA0 | 17 | made me feel close | Warmth/closeness | 10 | 96 | 3.052 | 4.521 | 1.469 | 1.240 | 1.729 | 1.193 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 1 | sense of who he was | Self-disclosure | 10 | 96 | 3.073 | 3.969 | 0.896 | 0.740 | 1.063 | 1.100 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 2 | revealed his thinking | Self-disclosure | 10 | 96 | 2.615 | 4.073 | 1.458 | 1.292 | 1.635 | 1.632 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 3 | shared his feelings | Self-disclosure | 10 | 96 | 2.135 | 3.271 | 1.135 | 1.010 | 1.271 | 1.723 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 4 | knew how I was feeling | Empathy/understanding | 10 | 96 | 3.271 | 4.417 | 1.146 | 0.958 | 1.344 | 1.165 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 5 | understood me | Empathy/understanding | 10 | 96 | 3.146 | 4.385 | 1.240 | 1.031 | 1.458 | 1.124 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 6 | put himself in my shoes | Empathy/understanding | 10 | 96 | 2.719 | 4.240 | 1.521 | 1.333 | 1.719 | 1.599 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 7 | comfortable talking | Fluency/ease | 10 | 96 | 3.500 | 4.729 | 1.229 | 1.021 | 1.458 | 1.072 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 8 | relaxed and secure | Fluency/ease | 10 | 96 | 3.469 | 4.740 | 1.271 | 1.042 | 1.521 | 1.058 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 9 | took charge | Direction/control | 10 | 96 | 2.708 | 4.177 | 1.469 | 1.260 | 1.688 | 1.341 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 10 | said when happy/sad | Self-disclosure | 10 | 96 | 2.260 | 3.479 | 1.219 | 1.062 | 1.385 | 1.554 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 11 | no difficulty w/ words | Fluency/ease | 10 | 96 | 3.531 | 4.760 | 1.229 | 1.021 | 1.458 | 1.080 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 12 | expressed himself | Fluency/ease | 10 | 96 | 3.490 | 4.771 | 1.281 | 1.062 | 1.510 | 1.124 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 13 | a 'warm' partner | Warmth/closeness | 10 | 96 | 3.052 | 4.375 | 1.323 | 1.083 | 1.583 | 1.029 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 14 | did not judge me | Non-judgment/equality | 10 | 96 | 3.396 | 4.885 | 1.490 | 1.198 | 1.812 | 0.974 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 15 | treated me as equal | Non-judgment/equality | 10 | 96 | 3.354 | 4.854 | 1.500 | 1.219 | 1.802 | 1.004 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 16 | made me feel cared for | Warmth/closeness | 10 | 96 | 3.188 | 4.760 | 1.573 | 1.281 | 1.865 | 1.064 | 0.000 |
| gpt-4o-mini | PTO_LA5 | 17 | made me feel close | Warmth/closeness | 10 | 96 | 3.125 | 4.719 | 1.594 | 1.323 | 1.875 | 1.118 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 1 | sense of who he was | Self-disclosure | 10 | 96 | 3.094 | 3.656 | 0.562 | 0.417 | 0.719 | 0.735 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 2 | revealed his thinking | Self-disclosure | 10 | 96 | 2.635 | 3.708 | 1.073 | 0.854 | 1.281 | 1.013 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 3 | shared his feelings | Self-disclosure | 10 | 96 | 2.219 | 3.083 | 0.865 | 0.719 | 1.000 | 1.201 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 4 | knew how I was feeling | Empathy/understanding | 10 | 96 | 3.281 | 4.094 | 0.812 | 0.625 | 1.000 | 0.861 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 5 | understood me | Empathy/understanding | 10 | 96 | 3.240 | 3.979 | 0.740 | 0.531 | 0.958 | 0.692 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 6 | put himself in my shoes | Empathy/understanding | 10 | 96 | 2.792 | 3.802 | 1.010 | 0.802 | 1.208 | 0.980 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 7 | comfortable talking | Fluency/ease | 10 | 96 | 3.542 | 4.240 | 0.698 | 0.510 | 0.896 | 0.701 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 8 | relaxed and secure | Fluency/ease | 10 | 96 | 3.531 | 4.250 | 0.719 | 0.531 | 0.917 | 0.717 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 9 | took charge | Direction/control | 10 | 96 | 2.729 | 3.719 | 0.990 | 0.781 | 1.198 | 0.916 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 10 | said when happy/sad | Self-disclosure | 10 | 96 | 2.333 | 3.135 | 0.802 | 0.635 | 0.969 | 0.968 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 11 | no difficulty w/ words | Fluency/ease | 10 | 96 | 3.562 | 4.208 | 0.646 | 0.448 | 0.854 | 0.617 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 12 | expressed himself | Fluency/ease | 10 | 96 | 3.542 | 4.208 | 0.667 | 0.469 | 0.875 | 0.627 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 13 | a 'warm' partner | Warmth/closeness | 10 | 96 | 3.031 | 3.823 | 0.792 | 0.594 | 0.990 | 0.797 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 14 | did not judge me | Non-judgment/equality | 10 | 96 | 3.510 | 4.333 | 0.823 | 0.552 | 1.115 | 0.573 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 15 | treated me as equal | Non-judgment/equality | 10 | 96 | 3.396 | 4.229 | 0.833 | 0.562 | 1.135 | 0.581 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 16 | made me feel cared for | Warmth/closeness | 10 | 96 | 3.281 | 3.938 | 0.656 | 0.396 | 0.938 | 0.467 | 0.000 |
| gpt-4o-mini | GRPO_LA0 | 17 | made me feel close | Warmth/closeness | 10 | 96 | 3.198 | 3.875 | 0.677 | 0.406 | 0.969 | 0.485 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 1 | sense of who he was | Self-disclosure | 5 | 96 | 3.042 | 3.771 | 0.729 | 0.573 | 0.896 | 0.926 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 2 | revealed his thinking | Self-disclosure | 5 | 96 | 2.604 | 3.708 | 1.104 | 0.917 | 1.302 | 1.168 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 3 | shared his feelings | Self-disclosure | 5 | 96 | 2.167 | 2.896 | 0.729 | 0.594 | 0.865 | 1.084 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 4 | knew how I was feeling | Empathy/understanding | 5 | 96 | 3.156 | 4.208 | 1.052 | 0.885 | 1.219 | 1.274 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 5 | understood me | Empathy/understanding | 5 | 96 | 3.177 | 4.177 | 1.000 | 0.812 | 1.198 | 1.090 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 6 | put himself in my shoes | Empathy/understanding | 5 | 96 | 2.688 | 3.792 | 1.104 | 0.906 | 1.312 | 1.082 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 7 | comfortable talking | Fluency/ease | 5 | 96 | 3.417 | 4.500 | 1.083 | 0.844 | 1.323 | 0.928 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 8 | relaxed and secure | Fluency/ease | 5 | 96 | 3.406 | 4.531 | 1.125 | 0.896 | 1.354 | 0.991 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 9 | took charge | Direction/control | 5 | 96 | 2.677 | 3.802 | 1.125 | 0.917 | 1.344 | 1.053 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 10 | said when happy/sad | Self-disclosure | 5 | 96 | 2.302 | 3.250 | 0.948 | 0.792 | 1.104 | 1.249 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 11 | no difficulty w/ words | Fluency/ease | 5 | 96 | 3.417 | 4.531 | 1.115 | 0.885 | 1.354 | 0.977 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 12 | expressed himself | Fluency/ease | 5 | 96 | 3.406 | 4.531 | 1.125 | 0.896 | 1.354 | 0.991 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 13 | a 'warm' partner | Warmth/closeness | 5 | 96 | 2.979 | 3.958 | 0.979 | 0.760 | 1.219 | 0.850 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 14 | did not judge me | Non-judgment/equality | 5 | 96 | 3.385 | 4.646 | 1.260 | 0.958 | 1.563 | 0.851 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 15 | treated me as equal | Non-judgment/equality | 5 | 96 | 3.260 | 4.542 | 1.281 | 1.000 | 1.583 | 0.917 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 16 | made me feel cared for | Warmth/closeness | 5 | 96 | 3.135 | 4.417 | 1.281 | 0.989 | 1.562 | 0.912 | 0.000 |
| gpt-4o-mini | GRPO_LA5 | 17 | made me feel close | Warmth/closeness | 5 | 96 | 3.052 | 4.365 | 1.312 | 1.052 | 1.573 | 1.009 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 1 | sense of who he was | Self-disclosure | 10 | 96 | 2.188 | 3.094 | 0.906 | 0.739 | 1.083 | 1.055 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 2 | revealed his thinking | Self-disclosure | 10 | 96 | 1.635 | 2.354 | 0.719 | 0.604 | 0.844 | 1.177 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 3 | shared his feelings | Self-disclosure | 10 | 96 | 1.479 | 3.208 | 1.729 | 1.552 | 1.906 | 1.921 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 4 | knew how I was feeling | Empathy/understanding | 10 | 96 | 1.917 | 2.906 | 0.990 | 0.833 | 1.146 | 1.278 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 5 | understood me | Empathy/understanding | 10 | 96 | 2.000 | 2.958 | 0.958 | 0.771 | 1.156 | 0.964 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 6 | put himself in my shoes | Empathy/understanding | 10 | 96 | 1.865 | 2.906 | 1.042 | 0.896 | 1.198 | 1.359 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 7 | comfortable talking | Fluency/ease | 10 | 96 | 2.021 | 3.146 | 1.125 | 0.917 | 1.333 | 1.073 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 8 | relaxed and secure | Fluency/ease | 10 | 96 | 1.875 | 3.021 | 1.146 | 0.958 | 1.333 | 1.219 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 9 | took charge | Direction/control | 10 | 96 | 1.583 | 2.438 | 0.854 | 0.688 | 1.010 | 1.041 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 10 | said when happy/sad | Self-disclosure | 10 | 96 | 1.219 | 3.000 | 1.781 | 1.573 | 1.990 | 1.652 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 11 | no difficulty w/ words | Fluency/ease | 10 | 96 | 2.729 | 3.625 | 0.896 | 0.698 | 1.094 | 0.906 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 12 | expressed himself | Fluency/ease | 10 | 96 | 2.708 | 3.646 | 0.938 | 0.740 | 1.125 | 0.976 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 13 | a 'warm' partner | Warmth/closeness | 10 | 96 | 1.854 | 3.271 | 1.417 | 1.229 | 1.615 | 1.414 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 14 | did not judge me | Non-judgment/equality | 10 | 96 | 2.781 | 3.729 | 0.948 | 0.792 | 1.104 | 1.148 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 15 | treated me as equal | Non-judgment/equality | 10 | 96 | 2.396 | 3.240 | 0.844 | 0.677 | 1.010 | 0.978 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 16 | made me feel cared for | Warmth/closeness | 10 | 96 | 1.781 | 3.135 | 1.354 | 1.177 | 1.552 | 1.424 | 0.000 |
| claude-haiku-4-5 | PTO_LA0 | 17 | made me feel close | Warmth/closeness | 10 | 96 | 1.740 | 2.896 | 1.156 | 1.000 | 1.312 | 1.447 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 1 | sense of who he was | Self-disclosure | 10 | 96 | 2.177 | 2.719 | 0.542 | 0.365 | 0.708 | 0.606 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 2 | revealed his thinking | Self-disclosure | 10 | 96 | 1.656 | 2.188 | 0.531 | 0.417 | 0.656 | 0.864 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 3 | shared his feelings | Self-disclosure | 10 | 96 | 1.458 | 2.104 | 0.646 | 0.479 | 0.812 | 0.775 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 4 | knew how I was feeling | Empathy/understanding | 10 | 96 | 1.906 | 2.750 | 0.844 | 0.667 | 1.010 | 0.965 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 5 | understood me | Empathy/understanding | 10 | 96 | 2.062 | 2.792 | 0.729 | 0.531 | 0.938 | 0.707 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 6 | put himself in my shoes | Empathy/understanding | 10 | 96 | 1.844 | 2.740 | 0.896 | 0.719 | 1.073 | 1.022 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 7 | comfortable talking | Fluency/ease | 10 | 96 | 2.010 | 2.844 | 0.833 | 0.635 | 1.031 | 0.815 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 8 | relaxed and secure | Fluency/ease | 10 | 96 | 1.875 | 2.688 | 0.812 | 0.625 | 1.000 | 0.851 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 9 | took charge | Direction/control | 10 | 96 | 1.552 | 2.302 | 0.750 | 0.552 | 0.948 | 0.746 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 10 | said when happy/sad | Self-disclosure | 10 | 96 | 1.177 | 1.969 | 0.792 | 0.625 | 0.969 | 0.937 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 11 | no difficulty w/ words | Fluency/ease | 10 | 96 | 2.771 | 3.354 | 0.583 | 0.375 | 0.792 | 0.554 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 12 | expressed himself | Fluency/ease | 10 | 96 | 2.740 | 3.406 | 0.667 | 0.469 | 0.865 | 0.666 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 13 | a 'warm' partner | Warmth/closeness | 10 | 96 | 1.854 | 2.760 | 0.906 | 0.719 | 1.094 | 0.963 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 14 | did not judge me | Non-judgment/equality | 10 | 96 | 2.781 | 3.490 | 0.708 | 0.531 | 0.885 | 0.773 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 15 | treated me as equal | Non-judgment/equality | 10 | 96 | 2.406 | 2.896 | 0.490 | 0.312 | 0.667 | 0.527 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 16 | made me feel cared for | Warmth/closeness | 10 | 96 | 1.740 | 2.750 | 1.010 | 0.843 | 1.188 | 1.153 | 0.000 |
| claude-haiku-4-5 | PTO_LA5 | 17 | made me feel close | Warmth/closeness | 10 | 96 | 1.719 | 2.656 | 0.938 | 0.771 | 1.104 | 1.095 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 1 | sense of who he was | Self-disclosure | 10 | 96 | 2.156 | 2.500 | 0.344 | 0.219 | 0.479 | 0.519 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 2 | revealed his thinking | Self-disclosure | 10 | 96 | 1.635 | 1.802 | 0.167 | 0.042 | 0.302 | 0.253 | 0.016 |
| claude-haiku-4-5 | GRPO_LA0 | 3 | shared his feelings | Self-disclosure | 10 | 96 | 1.458 | 3.521 | 2.062 | 1.875 | 2.250 | 2.198 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 4 | knew how I was feeling | Empathy/understanding | 10 | 96 | 1.906 | 2.281 | 0.375 | 0.229 | 0.521 | 0.514 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 5 | understood me | Empathy/understanding | 10 | 96 | 2.083 | 2.292 | 0.208 | 0.021 | 0.396 | 0.222 | 0.024 |
| claude-haiku-4-5 | GRPO_LA0 | 6 | put himself in my shoes | Empathy/understanding | 10 | 96 | 1.885 | 2.302 | 0.417 | 0.281 | 0.562 | 0.578 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 7 | comfortable talking | Fluency/ease | 10 | 96 | 2.115 | 2.333 | 0.219 | 0.031 | 0.406 | 0.235 | 0.016 |
| claude-haiku-4-5 | GRPO_LA0 | 8 | relaxed and secure | Fluency/ease | 10 | 96 | 1.958 | 2.271 | 0.312 | 0.156 | 0.469 | 0.398 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 9 | took charge | Direction/control | 10 | 96 | 1.521 | 1.604 | 0.083 | -0.062 | 0.229 | 0.113 | 0.264 |
| claude-haiku-4-5 | GRPO_LA0 | 10 | said when happy/sad | Self-disclosure | 10 | 96 | 1.229 | 3.479 | 2.250 | 2.052 | 2.438 | 2.393 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 11 | no difficulty w/ words | Fluency/ease | 10 | 96 | 2.885 | 3.438 | 0.552 | 0.365 | 0.750 | 0.588 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 12 | expressed himself | Fluency/ease | 10 | 96 | 2.917 | 3.458 | 0.542 | 0.354 | 0.740 | 0.591 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 13 | a 'warm' partner | Warmth/closeness | 10 | 96 | 1.844 | 2.823 | 0.979 | 0.812 | 1.146 | 1.158 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 14 | did not judge me | Non-judgment/equality | 10 | 96 | 2.865 | 3.510 | 0.646 | 0.500 | 0.812 | 0.827 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 15 | treated me as equal | Non-judgment/equality | 10 | 96 | 2.510 | 2.833 | 0.323 | 0.156 | 0.490 | 0.397 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 16 | made me feel cared for | Warmth/closeness | 10 | 96 | 1.781 | 2.448 | 0.667 | 0.510 | 0.823 | 0.844 | 0.000 |
| claude-haiku-4-5 | GRPO_LA0 | 17 | made me feel close | Warmth/closeness | 10 | 96 | 1.729 | 2.115 | 0.385 | 0.240 | 0.521 | 0.538 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 1 | sense of who he was | Self-disclosure | 5 | 96 | 2.156 | 2.802 | 0.646 | 0.479 | 0.802 | 0.813 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 2 | revealed his thinking | Self-disclosure | 5 | 96 | 1.625 | 2.177 | 0.552 | 0.438 | 0.677 | 0.900 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 3 | shared his feelings | Self-disclosure | 5 | 96 | 1.542 | 1.844 | 0.302 | 0.146 | 0.469 | 0.367 | 0.001 |
| claude-haiku-4-5 | GRPO_LA5 | 4 | knew how I was feeling | Empathy/understanding | 5 | 96 | 1.990 | 2.750 | 0.760 | 0.615 | 0.906 | 1.033 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 5 | understood me | Empathy/understanding | 5 | 96 | 2.083 | 2.990 | 0.906 | 0.729 | 1.083 | 1.040 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 6 | put himself in my shoes | Empathy/understanding | 5 | 96 | 1.917 | 2.719 | 0.802 | 0.646 | 0.958 | 1.052 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 7 | comfortable talking | Fluency/ease | 5 | 96 | 1.948 | 3.042 | 1.094 | 0.917 | 1.281 | 1.176 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 8 | relaxed and secure | Fluency/ease | 5 | 96 | 1.854 | 2.792 | 0.938 | 0.771 | 1.104 | 1.095 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 9 | took charge | Direction/control | 5 | 96 | 1.500 | 2.365 | 0.865 | 0.708 | 1.011 | 1.095 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 10 | said when happy/sad | Self-disclosure | 5 | 96 | 1.302 | 1.688 | 0.385 | 0.250 | 0.542 | 0.517 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 11 | no difficulty w/ words | Fluency/ease | 5 | 96 | 2.677 | 3.677 | 1.000 | 0.812 | 1.208 | 1.039 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 12 | expressed himself | Fluency/ease | 5 | 96 | 2.667 | 3.667 | 1.000 | 0.802 | 1.209 | 1.016 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 13 | a 'warm' partner | Warmth/closeness | 5 | 96 | 1.812 | 2.750 | 0.938 | 0.760 | 1.104 | 1.111 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 14 | did not judge me | Non-judgment/equality | 5 | 96 | 2.844 | 3.604 | 0.760 | 0.604 | 0.917 | 0.961 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 15 | treated me as equal | Non-judgment/equality | 5 | 96 | 2.427 | 3.052 | 0.625 | 0.458 | 0.781 | 0.796 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 16 | made me feel cared for | Warmth/closeness | 5 | 96 | 1.740 | 2.719 | 0.979 | 0.812 | 1.135 | 1.233 | 0.000 |
| claude-haiku-4-5 | GRPO_LA5 | 17 | made me feel close | Warmth/closeness | 5 | 96 | 1.708 | 2.708 | 1.000 | 0.833 | 1.167 | 1.218 | 0.000 |
