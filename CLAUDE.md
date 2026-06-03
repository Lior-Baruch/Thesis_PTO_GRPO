# Thesis — PTO vs GRPO for Motivational Interviewing

## What this is
Master's thesis (Lior Baruch, Reichman). Trains small therapist LLMs to do
Motivational Interviewing against simulated patients; reward = larger
"oracle" LLM grading validated MI questionnaires.

Three controlled comparisons, all live in Exp3:
1. **Look-ahead depth** K ∈ {0, 5} — does anticipating future turns help, and by how much?
2. **PTO vs GRPO** under matched K + MCL — does iterative GRPO compete with PTO?
3. **Oracle questionnaire** (Q1+Q2 vs WAI-SR vs CSQ-8 vs MI-SAT/MITI) — held for later.

## Experiments (chronological)
| | [Exp1_ICLR2025/](Exp1_ICLR2025/) | [Exp2_PTO/](Exp2_PTO/) | [Exp3_PTO_GRPO/](Exp3_PTO_GRPO/) |
|---|---|---|---|
| **Status** | Frozen — published | Complete — EDA verified | **Active — refactored; both trainers pending real runs** |
| **Therapist** | Llama-2-7B | Llama-3.x | Llama-3.2-1B |
| **Patient + oracle** | GPT-3.5 | gpt-4o-mini-2024-07-18 | gpt-4o-mini-2024-07-18 |
| **Patient prompts** | V1 (cooperative) | V3 (less cooperative) | V3 |
| **Oracle output** | V1 (regex; Q1+Q2 only) | V5 (JSON schema; 6 questionnaires) | V5 |
| **PTO** | K ∈ {0, 5}, 7 iters | 4 oracles × K ∈ {0, 5} | **PTO_Exp3** (refactored, iterative; rebuilt as a lean sibling of GRPO_Exp3) |
| **GRPO** | — | V1 (static prompts, weak baseline) | **GRPO_Exp3** (iterative) — both methods now share `code/_shared/` |
| **MCL filter** | — | — | **Wired in both PTO_Exp3 and GRPO_Exp3.** Encoded in `EXPERIMENT_NAME`. |
| **Training reward** | mean(Q1, Q2) | chosen oracle | Q1+Q2 only (matches Exp1) |
| **Eval reward** | Q1, Q2 | per-oracle | all 6 questionnaires |
| **EDA shape** | `Conv_EDA.ipynb` | + per-Q CSVs, `pref_emb/` | + `lib/` package, `Partial_Conv_Oracle_EDA.ipynb` |
| **Convs / models** | (paper figures) | 4,512 / 47 | 3,456 / 36 (PTO Exp2 data) + new GRPO/PTO_Exp3 runs pending |

Dirs renamed 2026-05-12 from `ICLR2025/`/`Extension/`/`NewExperiment/`.

## Data lineage
- **Exp1 → Exp2:** independent re-implementation. Stronger oracle, harder patients, JSON-schema rubric, more questionnaires. No data flow.
- **Exp2 → Exp3:** PTO `pref_trees/` and `eval_conversations/` for {Base, Q1Q2, WAI, CSQ-8, CTRL} were **copied** into `Exp3_PTO_GRPO/data/pto_Exp2/`. The Exp2 PTO results stand as a reference baseline. GRPO V1 baseline from Exp2 was **dropped** (Exp3 focuses on PTO_Exp3 vs GRPO_Exp3 only).
- **Exp3 self-loop:** GRPO_Exp3 regenerates its own training data each iter from the current policy; those same convs are the eval set (no separate generate-eval step for trained iters).

## Key methodological shift across experiments
- **Look-ahead K** stayed central throughout (the lever from the ICLR paper).
- **The hard part moved from "can PTO beat the baseline?" (Exp1, settled) to "is GRPO competitive with PTO under matched look-ahead?" (Exp3, open).**
- **Exp3 also exposed a reward-faithfulness concern** the earlier experiments never tested: the `Partial_Conv_Oracle_EDA` shows that the short-cut training reward has only ~0.66–0.73 rank agreement with the full-conv eval at `n_turns=2`. Motivates the `MIN_CONV_LENGTH` knob — now wired in both GRPO_Exp3 (slice filter) and PTO_Exp3 (branch-point filter); encoded in `EXPERIMENT_NAME` so MCL sweeps stay in disjoint folders.

## Methods (one line each)
- **PTO V1** (Exp1) = original preference-tree exploration + K look-ahead + DPO. Published.
- **GRPO V1** (Exp2) = static prompt set, weak baseline.
- **GRPO_Exp3** = current policy simulates 96 convs → per-turn prompts (MCL filter) → GRPO update with optional K-turn look-ahead. Convs double as eval.
- **PTO_Exp3** = current policy simulates 96 convs → per-turn branching (`M` candidates) with MCL filter → K-turn look-ahead + oracle → τ-filtered (chosen, rejected) pref pairs → DPO update. Lean sibling of GRPO_Exp3.

**Shared infrastructure (Exp3).** Both GRPO_Exp3 and PTO_Exp3 trainers import from
`Exp3_PTO_GRPO/code/_shared/` (5 modules: runtime, model, convs, reward, tb_plots).
Each method's trainer module (`grpo_trainer.py` / `pto_trainer.py` — named per method
so `from <method>_trainer import …` can't collide in a shared kernel) owns just the
method-specific bits (`TrainingConfig`/`PTOConfig`, iteration body, dataset shape, TRL
trainer wrapping).

**Naming:** PTO is the framework, DPO is the loss. Don't call GRPO data "pref data" — it has none.

## Layout
```
Thesis_PTO_GRPO/
├── CLAUDE.md                   (this file)
├── Exp{1,2,3}_*/CLAUDE.md      per-experiment context
├── archive/                    historical artifacts; do not extend
├── HF_key.txt, openai_key.txt  duplicated per-experiment-dir, not at root
├── requirements.txt, gen_requirements.py
└── .venv/                      Python 3.13 env
```

## Conventions
- **Each experiment dir is self-contained.** Its own `code/`, `data/`, `eda/`, local `system_prompts_builder.py`+`questionnaires.py` (versions diverge across experiments — never share a root-level module). Within Exp3, both helpers live ONCE at `code/` root; the EDA package imports the same files via a `sys.path` prepend.
- **Workspace root resolver.** Walks up from `os.getcwd()` looking for `HF_key.txt`+`openai_key.txt` together → resolves to experiment root (`Exp{1,2,3}_*/`). Used by every notebook.
- **EDA path remapping.** Legacy strings like `"LLM_DATA/Conversation_with_Eval_V3/..."` (Exp1/Exp2 EDAs) are remapped at load time by `_resolve_data_path(...)`. Don't rewrite the literals.
- **File version suffixes (`_V3`, `_V5`)** are dropped when the file lives in an experiment dir (the dir provides version context). Method-lineage subdirs in Exp3 are named after the experiment (`GRPO_Exp3/`, `PTO_Exp3/`).
- **Exp3 trainer pattern.** `code/<METHOD>_Exp3/{train_<METHOD>_Iterative.ipynb, <method>_trainer.py}` (e.g. `grpo_trainer.py`, `pto_trainer.py` — distinct module names to avoid `from trainer` collisions across notebooks in one kernel) with the per-iteration orchestration loop visible in the notebook. Shared helpers in `code/_shared/`.

## Next step
**Landed:** the *batched, safely-parallel* look-ahead rollout (the K>0 wall-clock
bottleneck) — `simulate_lookahead_batch` is now a lock-step batched rewrite, plus a
`LOOKAHEAD_SUB_BATCH_SIZE` knob and an optional equivalence harness
(`_shared/lookahead_check.py`). Implemented + logic-tested; see
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Look-ahead performance".

**Immediate (real-GPU validation):** run the optional **section 6** equivalence cell
(serial vs batched), then the K=3 look-ahead quicktest (`RUN_MODE="quicktest"`,
`LOOKAHEAD_K=3`), local **bf16** only. Entry:
[Exp3_PTO_GRPO/code/GRPO_Exp3/train_GRPO_Iterative.ipynb](Exp3_PTO_GRPO/code/GRPO_Exp3/train_GRPO_Iterative.ipynb).

**Then:** full GRPO_Exp3 sweep over K ∈ {0, 5} on Q1+Q2 at MCL = 12 (Colab). Optional:
PTO_Exp3 over the same grid via [Exp3_PTO_GRPO/code/PTO_Exp3/train_PTO_Iterative.ipynb](Exp3_PTO_GRPO/code/PTO_Exp3/train_PTO_Iterative.ipynb).

## Hardware
Local: Windows, RTX 5070 Ti (12 GB VRAM), CUDA 12.8, torch 2.11.0+cu128.
GRPO_Exp3 training is intended for Colab (GPU); EDA + Run_Eval run locally.
