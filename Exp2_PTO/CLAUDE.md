# Exp2_PTO — COMPLETE (reference) · **PTO ONLY**

PTO sweep on Llama-3.2-1B + gpt-4o-mini against four oracles (Q1+Q2, WAI-SR,
CSQ-8, CTRL). V3 patient prompts (less cooperative), V5 oracle (JSON schema,
six questionnaires supported: Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI 4.2).

> ⚠ **This experiment also contains a GRPO V1 run. It had a BUG and its results are VOID.**
> See § "GRPO V1 — void" below. **Exp2 is a PTO experiment**; treat the `GRPO/` and
> `GRPO-Instruct/` model states as absent when reading, presenting or writing up anything here.

EDA verified end-to-end: 4,512 convs / 47 models / 9 experiment groups — ⚠ **counts of everything
on disk, including the void GRPO states**. The PTO-only subset is smaller; recompute before
quoting a count in a write-up.

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

## GRPO V1 — VOID (the run had a bug)

⚠ **The Exp2 GRPO V1 run had a bug. Its scores mean nothing and must never be presented** — not as
a baseline, not as a comparison point, not as evidence about GRPO as a method. Every `GRPO_E*` /
`GRPOI_*` model state under `data/conversations_eval/` and `eda/eval/` is void output. Do not
compute, quote or plot them.

*(Established 2026-08-23. The specific defect is not recorded here — if you need it, ask rather than
reconstructing one from the code.)*

**What this file used to say, and why it was wrong.** It carried a section titled *"GRPO V1
(static) — why it's weak"* explaining the low scores as a consequence of the fixed prompt set never
adapting to the current policy. The fixed prompt set is a true description of the implementation,
but using it to *explain the scores* reads a bug's output as a finding — and it is exactly the kind
of mechanistic-sounding interpretation that survives re-reading because it is plausible. It is
retired. Anything downstream that rested on "GRPO started weak in Exp2 and got better in Exp3" is
resting on void data.

**Where the real comparison lives.** PTO vs GRPO is an **Exp3** result and only an Exp3 result:
there both methods are iterative, share `code/_shared/`, and run at matched `MCL`, `K`, candidate
budget (`M` = `G` = 8), temperature and oracle. That is what makes it controlled. Exp1 and Exp2 are
PTO-only.

## Layout
```
Exp2_PTO/
├── CLAUDE.md
├── code/
│   ├── system_prompts_builder.py            V3 prompts (less cooperative)
│   ├── questionnaires.py                    V5 (JSON schema, 6 questionnaires)
│   ├── PTO_PrefData_and_Eval.ipynb          PTO pref-data generation + eval
│   ├── Train_model_pref_tree.ipynb          DPO training over PTO pref data
│   ├── train_GRPO_Oracle_Async.ipynb        ⚠ GRPO V1 trainer — VOID run, see § GRPO V1
│   └── Generate_Conversations_GRPO.ipynb    ⚠ eval convs for that void run
├── data/
│   ├── conversation_trees/{CSQ-8,CTRL,Q1Q2,WAI}/LookAhead_{0,5}/   PTO pref data
│   ├── conversations_eval/                  Base/, CSQ-8/, CTRL/, Q1Q2/, WAI/  ← the PTO sweep
│   │     GRPO/, GRPO-Instruct/              ⚠ VOID — buggy run, never present
│   └── grpo_v1_static/                      ⚠ the void run's static prompt set
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

### GRPO V1 — ⚠ do not re-run
The two GRPO notebooks ([train_GRPO_Oracle_Async.ipynb](code/train_GRPO_Oracle_Async.ipynb),
[Generate_Conversations_GRPO.ipynb](code/Generate_Conversations_GRPO.ipynb)) produced the void run
described above. They are kept as a record of what was run, not as a pipeline step. **Running them
again would reproduce the bug, not fix it.** GRPO lives in Exp3.

### Re-run EDA
```powershell
jupyter nbconvert --to notebook --execute --inplace eda\Conv_EDA.ipynb
```
(from project root with `.venv` active). Produces per-oracle bar charts,
ANOVA + Tukey HSD across iterations, and conversation-length comparisons.

⚠ **The notebook still builds Base vs PTO vs GRPO panels** — its `GROUP_ORDER`, `GRPOI_NAME_PATTERN`
and `_NON_DPO_GROUPS` all carry the `GRPO` / `GRPO-Instruct` groups, and the committed outputs show
them. Those panels are **void** (see § GRPO V1). The notebook has NOT been changed; ignore every
GRPO series it draws, and drop those groups before reusing any frame it produces.

## Gotchas
- **GRPO V1 is void, not a baseline.** See § "GRPO V1 — VOID". Don't extend it, don't re-run it,
  don't quote it. Exp2 is PTO-only.
- Helpers (`system_prompts_builder.py`, `questionnaires.py`) are versioned **per experiment** — never reach into another experiment dir.
- The legacy `LLM_DATA/...` path literals in notebooks are intentional; `_resolve_data_path()` translates them.
