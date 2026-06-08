# Exp2_PTO — COMPLETE (reference)

PTO sweep on Llama-3.2-1B + gpt-4o-mini against four oracles (Q1+Q2, WAI-SR,
CSQ-8, CTRL). V3 patient prompts (less cooperative), V5 oracle (JSON schema,
six questionnaires supported: Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI 4.2).
Plus a **first GRPO attempt (V1, static-data) kept as a weak baseline**.

EDA verified end-to-end: 4,512 convs / 47 models / 9 experiment groups.

## Setup
| Role | Model |
|---|---|
| Therapist | Llama-3.2-1B (4-bit NF4 + LoRA via DPO) — same base as Exp3, but Exp3 generates convs in bf16 |
| Patient simulator | `gpt-4o-mini-2024-07-18` |
| Oracle (evaluator) | `gpt-4o-mini-2024-07-18` |

- 96 patient permutations.
- PTO sweeps at K ∈ {0, 5}, iters labeled V1..V7 per oracle.
- Reward = chosen oracle's mean score. Filter τ = 0.1 on pref pairs.

## GRPO V1 (static) — why it's weak
The V1 GRPO trainer used a **fixed prompt set** (no per-iter regeneration).
Prompts never adapted to the current policy, so the training signal stayed
disconnected from where the model actually was. Kept only as a baseline
comparison point in the EDA.

## Layout
```
Exp2_PTO/
├── CLAUDE.md
├── code/
│   ├── system_prompts_builder.py            V3 prompts (less cooperative)
│   ├── questionnaires.py                    V5 (JSON schema, 6 questionnaires)
│   ├── PTO_PrefData_and_Eval.ipynb          PTO pref-data generation + eval
│   ├── Train_model_pref_tree.ipynb          DPO training over PTO pref data
│   ├── train_GRPO_Oracle_Async.ipynb        GRPO V1 trainer (static-data)
│   └── Generate_Conversations_GRPO.ipynb    eval conv generation from a GRPO checkpoint
├── data/
│   ├── conversation_trees/{CSQ-8,CTRL,Q1Q2,WAI}/LookAhead_{0,5}/   PTO pref data
│   ├── conversations_eval/                  Base + per-oracle PTO + GRPO V1 outputs:
│   │     Base/, CSQ-8/, CTRL/, Q1Q2/, WAI/, GRPO/, GRPO-Instruct/
│   └── grpo_v1_static/                      GRPO V1's static prompt set
├── eda/
│   ├── Conv_EDA.ipynb                       main analysis (aggregate across model variants)
│   ├── eval/{CSQ8,MITI,MI_SAT,Q1,Q2,WAI_SR}/   per-questionnaire result CSVs
│   └── pref_emb/preference_analysis.ipynb   pref-pair embedding analysis
└── HF_key.txt, openai_key.txt
```

## Running the pipeline
Notebooks resolve the workspace root by walking up from `os.getcwd()` for
`HF_key.txt`+`openai_key.txt` → resolves to `Exp2_PTO/`. Path strings in
notebooks use legacy `LLM_DATA/Conversation_with_Eval_V3/...` form —
**remapped at load time** to `Exp2_PTO/data/conversations_eval/...`. Don't
rewrite the literals.

### PTO sweep
1. **Generate preference trees.** [code/PTO_PrefData_and_Eval.ipynb](code/PTO_PrefData_and_Eval.ipynb). Pick oracle (Q1Q2 / WAI / CSQ-8 / CTRL) and look-ahead K.
2. **Train DPO.** [code/Train_model_pref_tree.ipynb](code/Train_model_pref_tree.ipynb) on the freshly generated trees.
3. Iterate. Outputs land under `data/conversation_trees/<oracle>/LookAhead_<K>/` and `data/conversations_eval/<oracle>/LookAhead_<K>/`.

### GRPO V1 (static — baseline only)
4. **Train GRPO.** [code/train_GRPO_Oracle_Async.ipynb](code/train_GRPO_Oracle_Async.ipynb). Uses the fixed prompt set in `data/grpo_v1_static/`.
5. **Generate eval conversations.** [code/Generate_Conversations_GRPO.ipynb](code/Generate_Conversations_GRPO.ipynb) against each saved adapter checkpoint.

### Re-run EDA
```powershell
jupyter nbconvert --to notebook --execute --inplace eda\Conv_EDA.ipynb
```
(from project root with `.venv` active). Produces per-oracle bar charts,
ANOVA + Tukey HSD across iterations, conversation-length comparisons, and
Base vs PTO vs GRPO V1 comparison panels.

## Gotchas
- **Don't extend GRPO V1.** It's archived as a baseline only.
- Helpers (`system_prompts_builder.py`, `questionnaires.py`) are versioned **per experiment** — never reach into another experiment dir.
- The legacy `LLM_DATA/...` path literals in notebooks are intentional; `_resolve_data_path()` translates them.
