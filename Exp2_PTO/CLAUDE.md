# Exp2_PTO — COMPLETE (reference)

PTO sweep on Llama-3.2-1B + gpt-4o-mini against four oracles (Q1+Q2, WAI-SR,
CSQ-8, CTRL). V3 patient prompts (less cooperative), V5 oracle (JSON schema,
six questionnaires supported: Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI 4.2).
Plus a **first GRPO attempt (V1, static-data) kept as a weak baseline**.

EDA verified end-to-end: 4,512 convs / 47 models / 9 experiment groups.

## Setup
| Role | Model |
|---|---|
| Therapist | Llama-3.2-1B (4-bit NF4 + LoRA via DPO) — quantization matters for score comparability; see "Quantization" below |
| Patient simulator | `gpt-4o-mini-2024-07-18` |
| Oracle (evaluator) | `gpt-4o-mini-2024-07-18` |

- 96 patient permutations.
- PTO sweeps at K ∈ {0, 5}, iters labeled V1..V7 per oracle.
- Reward = chosen oracle's mean score. Filter τ = 0.1 on pref pairs.

### Quantization — why Exp2 and Exp3 absolute scores are not on one axis

⚠ **Exp2 and Exp3 absolute oracle scores are NOT on the same axis — compare WITHIN Exp3 only.**
The therapist base is the *same* model in both (Llama-3.2-1B); only the generation precision
differs — **Exp2 generated its conversations in 4-bit NF4, Exp3 in bf16.**

The mechanism is degeneration, not capability. 4-bit induces phrase-loop degeneration in
**≈9.5% of therapist turns** (they run to the token cap as repeated spam) versus **≈0.3%** in
bf16 — `9.5 / 0.3 ≈ 32×` more. The oracle floors those turns, which drags the mean down:

| Arm | Q1+Q2 |
|---|---|
| Exp2 Base (4-bit, all convs) | ≈ 2.38 |
| Exp2 Base, clean (non-degenerate) subset | ≈ 2.93 |
| Exp3 Base (bf16) | ≈ 3.0 |

So the raw gap is `3.0 − 2.38 ≈ 0.62`, but removing the degenerate turns closes it to
`3.0 − 2.93 ≈ 0.07`. **That is the evidence the gap is a quantization artifact, not a real
difference in model quality.** To put Exp2 on the same axis as Exp3, its conversations would have
to be regenerated in bf16 — until then, no cross-experiment level comparison is valid (the root
[../CLAUDE.md](../CLAUDE.md) § "Data lineage" carries the same warning).

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
