# Exp1_ICLR2025 — FROZEN

Published ICLR 2025 workshop paper: [paper.pdf](paper.pdf).
Don't modify unless a re-run is explicitly requested.

## Setup
| Role | Model |
|---|---|
| Therapist | Llama-2-7B (4-bit + LoRA via DPO) |
| Patient simulator | GPT-3.5 |
| Oracle (evaluator) | GPT-3.5 |

- 96 patient permutations (gender × age × problem × duration × prior attempts × cooperation).
- "Good"-level therapist persona, single fixed name.
- PTO @ K ∈ {0, 5}, 7 iterations each (K=3 has a smaller sweep).
- Reward = mean(Q1, Q2) — Q1 (5 items, session satisfaction), Q2 (17 items, working alliance), each 1–5 Likert.
- Filter τ = 0.1 (drop pref pairs where winner − loser ≤ 0.1).

## Headline result
Every PTO model beats the baseline on Q1 (Session Satisfaction) and Q2
(Working Alliance). Deeper look-ahead (K=5) yields higher and more stable
scores; best K=5 model has lowest variance and shortest conversations.

## Layout
```
Exp1_ICLR2025/
├── paper.pdf
├── CLAUDE.md
├── code/                                V1 PTO pipeline (no JSON schema)
│   ├── system_prompts_builder.py        V1 prompts (cooperative variant)
│   ├── questionnaires.py                V1 oracle (regex-parsed, Q1+Q2 only)
│   ├── PTO_PrefData_and_Eval.ipynb      generate pref trees + eval convs
│   └── Train_model_pref_tree.ipynb      DPO training over pref data
├── data/
│   ├── conversation_trees/              pref data: LookAhead_{0,3,5,10}
│   └── conversations_eval/              end-of-iteration eval convs
│       ├── Base/                        Llama-2-7B baseline (no training)
│       ├── LookAhead_0/                 K=0 PTO models, V1..V7
│       ├── LookAhead_3/                 K=3 (smaller sweep)
│       └── LookAhead_5/                 K=5 PTO models, V1..V7
├── eda/Conv_EDA.ipynb                   paper's tables + figures
└── HF_key.txt, openai_key.txt
```

## Running the pipeline (if ever re-run)
1. Open [code/PTO_PrefData_and_Eval.ipynb](code/PTO_PrefData_and_Eval.ipynb). Set `lookAhead` and `version`, run cells. Outputs → `data/conversation_trees/LookAhead_<K>/...` and `data/conversations_eval/LookAhead_<K>/...`.
2. Open [code/Train_model_pref_tree.ipynb](code/Train_model_pref_tree.ipynb), point at the freshly generated pref-tree dir, run.
3. Iterate — after each DPO update, regenerate pref data with the new agent. The paper's K=5 sweep ran 7 such iterations.

## EDA
[eda/Conv_EDA.ipynb](eda/Conv_EDA.ipynb) loads every model variant in
`data/conversations_eval/`, computes Q1 / Q2 / mean per conversation, runs
ANOVA + Tukey HSD across models, produces the paper's bar charts. Path
strings still use legacy `LLM_DATA/Conversation_with_Eval/...` — remapped
at load time by `_resolve_data_path()`.

## Gotchas
- V1 helpers (`system_prompts_builder.py`, `questionnaires.py`) are **specific to this experiment** — don't share with other experiment dirs.
- API keys resolved by walking up from `os.getcwd()` to find `HF_key.txt`+`openai_key.txt` together → resolves to `Exp1_ICLR2025/`.
- Don't rewrite the legacy `LLM_DATA/...` strings in the EDA; the remapper handles them.
