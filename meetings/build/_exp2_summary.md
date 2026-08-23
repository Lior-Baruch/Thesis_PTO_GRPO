# Exp2_PTO — eval-set census and per-model-state score summary (rebuilt from disk)

> NOTE — Exp2's GRPO V1 model states (`GRPO_E*`, `GRPOI_*`) have been REMOVED from this file.
> That run had a bug and its scores are void; see `Exp2_PTO/CLAUDE.md` § "GRPO V1 — VOID".
> Exp2 is a PTO-only experiment. Do not restore them.

**Built by:** `.venv/Scripts/python.exe` over `Exp2_PTO/eda/eval/<QUESTIONNAIRE>/<model_state>/<conv_index>.csv` on 2026-08-23.
Nothing here is copied from a narrative doc; every cell is an aggregate of the score CSVs named in § 10 Provenance.
Machine-readable twin: `meetings/build/_exp2_summary.csv` (500 rows = 50 model states x 10 metrics).

## 1. Census — what is on disk

Root: `Exp2_PTO/eda/eval/`

| questionnaire dir | model_state dirs | conv CSVs | CSVs per state | header (single variant) | file-index set |
|---|---|---|---|---|---|
| `Q1/` | 50 | 4,800 | 96 (all 50 states) | `Q1_1..Q1_5, Q1_Mean, Q1_Total` | 0..95, identical in all 50 |
| `Q2/` | 50 | 4,800 | 96 (all 50 states) | `Q2_1..Q2_17, Q2_Mean, Q2_Total` | 0..95, identical in all 50 |
| `WAI_SR/` | 50 | 4,800 | 96 (all 50 states) | `WAI1..WAI12, WAI_Goal_Mean, WAI_Task_Mean, WAI_Bond_Mean, WAI_TotalMean, WAI_TotalSum` | 0..95, identical in all 50 |
| `CSQ8/` | 50 | 4,800 | 96 (all 50 states) | `CSQ1..CSQ8, CSQ8_Mean, CSQ8_Total` | 0..95, identical in all 50 |
| `MI_SAT/` | 50 | 4,800 | 96 (all 50 states) | `MI1..MI6, MI_Mean, MI_Total` | 0..95, identical in all 50 |
| `MITI/` | 50 | 4,800 | 96 (all 50 states) | `MITI1..MITI4, MITI_GlobalMean, MITI_B1..B7, MITI_BehaviorTotal` | 0..95, identical in all 50 |

- **50 model_state dirs x 96 conversations x 6 questionnaires = 28,800 one-row score CSVs.**
- Every CSV has exactly 1 data row; no NaN in any `*_Mean` column; the 50 model_state names are identical across all six questionnaire dirs.
- File index `<conv_index>.csv` = `patient_id` = the persona permutation index. Spot-checked in `Exp2_PTO/data/conversations_eval/`: index 0 = 'James, 27 / smoking', 37 = 'James, 61', 50 = 'Emma, 27', 95 = 'Emma, 61' in `Base`, `Q1Q2/LookAhead_5/..._V10`, `GRPO/GRPO_Epoch3` and `GRPO-Instruct/GRPO_Base` alike. Neither generator shuffles (`generate_all_permutations(only_expert_therapist=True)`, no `shuffle` call in either notebook), so **persona index is stable across all families and paired-by-index contrasts are valid** — unlike Exp3, which reshuffles per iteration.

## 2. Model-state naming scheme (parsed from code, not from prose)

| token | meaning | evidence |
|---|---|---|
| `Base` | Llama-3.2-1B (**non-Instruct**), 4-bit NF4, no adapter | `Exp2_PTO/code/PTO_PrefData_and_Eval.ipynb` cell 11 `therapist_model_id = "meta-llama/Llama-3.2-1B"`; cell 12 `load_in_4bit=True, nf4`; conv dir `data/conversations_eval/Base/Good_50_TT0.9_TP0.7_TE0.1` |
| `L{K}_{ORACLE}_V{N}` | PTO/DPO arm: look-ahead K, training oracle ORACLE, **N stacked DPO adapters** (= PTO iteration N) | `Exp2_PTO/eda/Conv_EDA.ipynb` cell 5 `Experiment.model_name` returns `f"L{self.lookahead}_{self.oracle}_V{self.version}"`; `PTO_PrefData_and_Eval.ipynb` cell 17 `for i in range(0, version): PeftModel.from_pretrained(...); merge_and_unload()` and cell 49 save path `..._V{version}` |

### What `_E<N>` denotes — direct evidence
`Exp2_PTO/code/Generate_Conversations_GRPO.ipynb` cell 2: `EPOCHS_TO_RUN = [3, 6, 9, 12, 15, 18, 21, 24]   # 0 = base model (no adapter); 1+ = checkpoint epochs`,
and cell 18 `build_epoch_output_dir()` writes `GRPO_Base_TT..._TP...` for epoch 0 else `GRPO_Epoch{N}_TT..._TP...`.
So **`_E<N>` = the GRPO training-epoch checkpoint number** — not an iteration of a data-regeneration loop. GRPO V1 trained once on a *static* preference-tree corpus (`DATA_SUBPATH = LLM_DATA/Conversation_Trees_GRPO_V2/LookAhead_5/...`, `train_GRPO_Oracle_Async.ipynb` cell 2), so epochs are passes over fixed data.
Corroborating: `train_GRPO_Oracle_Async.ipynb` cell 2 `NUM_TRAIN_EPOCHS = 24`, `RESTART_EVERY_N_EPOCHS = 3` (which is why the checkpoint grid is 3/6/9/...), and its stored cell-28 output lists 15 checkpoints, `checkpoint-496 ... checkpoint-7440` (496 optimizer steps per epoch x 15 epochs).

### What `GRPO_` vs `GRPOI_` denotes — what the evidence does and does not settle
**Settled by code.** `GRPOI_` is the `method="GRPO-Instruct"` family (`Conv_EDA.ipynb` cell 5) and reads conversations from `data/conversations_eval/GRPO-Instruct/`; `GRPO_` reads from `data/conversations_eval/GRPO/`. They are genuinely different generations — `conversation_0.csv` differs in text between the two dirs at the same epoch.
**Settled by the stored notebook output.** The last execution of `Generate_Conversations_GRPO.ipynb` left this in cell 2's output inside the `.ipynb`:

```
  Base model:         meta-llama/Llama-3.2-1B-Instruct
  GRPO run dir:       ../grpo_runs/GRPO_Oracle-Q1Q2_Llama32-1B-Instruct_LA5_G4_V2
  Epochs to run:      [3, 6, 9, 12, 15, 18, 21, 24]
  Output base dir:    LLM_DATA/Conversation_with_Eval_V3/GRPO-Instruct
```

So the **GRPO-Instruct (`GRPOI_`) conversations were generated with GRPO adapters mounted on `meta-llama/Llama-3.2-1B-Instruct`**, whereas every other Exp2 family runs on the plain `meta-llama/Llama-3.2-1B`. `GRPOI_Base` is that Instruct checkpoint with **no** adapter — a second, different baseline.
**Corroborated by the scores.** The two no-adapter runs (`GRPOI_Base` and `Base`) score far apart on Q1+Q2 — a gap too large for two draws of the same checkpoint, which is independent evidence that the underlying model differs. (Values withheld: they belong to the void run and are not reported anywhere in this file.)
**NOT settled — flagged rather than guessed.** The notebook's *current source* (cell 2, edited after the last run) reads `GRPO_RUN_DIR = ../grpo_runs/V1/GRPO_Oracle-Q1Q2_Llama32-1B-Instruct_LA5_G4` while the *stored output of the same cell* says `../grpo_runs/GRPO_Oracle-Q1Q2_Llama32-1B-Instruct_LA5_G4_V2`. Two GRPO runs therefore existed (a `V1/` one and a `_V2` one), the repo contains **no** `grpo_runs/` tree, no run metadata and no adapter config, and no notebook records the `GRPO/` (non-Instruct) generation. **Which run produced the `GRPO_E*` conversations, and on which base checkpoint they were generated, is unrecorded anywhere in the repo.** The CSV's `gen_base_model` field for that family says `unrecorded (see notes)`.
Also note: `train_GRPO_Oracle_Async.ipynb` cell 2 sets `BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"` (non-Instruct) while `EXPERIMENT_NAME_BASE = "GRPO_Oracle_Llama32-1B-Instruct_LA5_G4_V2"` contains the string `Instruct` — **the run name is not evidence about the base model.**
Both GRPO families share one training oracle: `QUESTIONNAIRE_IDS = [1, 2]` (Q1+Q2) with `LA5` in the run name; `NUM_GENERATIONS = 4`, `GRPO_BETA = 0.01`, `GRPO_TEMPERATURE = 1.2`, `LEARNING_RATE = 1e-5` (`train_GRPO_Oracle_Async.ipynb` cell 2).

## 3. Per-model-state means — Q1+Q2 axis (the training-reward axis)

`Q1Q2_Mean` is derived per conversation as `(Q1_Mean + Q2_Mean) / 2`, exactly as `Conv_EDA.ipynb` cell 9 `merge_q1_q2_results()` does (inner join on `Model`, `patient_id`).
`Q1_Mean` = column `Q1_Mean` of `eda/eval/Q1/<model_state>/<i>.csv`; `Q2_Mean` = column `Q2_Mean` of `eda/eval/Q2/<model_state>/<i>.csv`. n = 96 conversations for every state.
`delta / dz / p` are paired over the 96 personas against the state's own reference: `Base` for everything except the `GRPOI_*` family, which is referenced to `GRPOI_Base` (different base checkpoint — see § 2).

| model_state | group | train oracle | K | iter | n | Q1Q2_Mean | sd | 95% CI of mean | ref | delta vs ref | dz | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Base` | Base | nan | — | 0 | 96 | **2.3775** | 1.0797 | [2.162, 2.593] | Base | 0.0000 | — | — |
| `L0_Q1Q2_V1` | L0_Q1Q2 | Q1Q2 | 0 | 1 | 96 | **2.4991** | 1.2284 | [2.253, 2.745] | Base | 0.1216 | 0.108 | 0.415 |
| `L0_Q1Q2_V2` | L0_Q1Q2 | Q1Q2 | 0 | 2 | 96 | **2.6970** | 1.2057 | [2.456, 2.938] | Base | 0.3195 | 0.282 | 0.007 |
| `L0_Q1Q2_V3` | L0_Q1Q2 | Q1Q2 | 0 | 3 | 96 | **2.5441** | 1.2292 | [2.298, 2.790] | Base | 0.1666 | 0.165 | 0.104 |
| `L0_Q1Q2_V4` | L0_Q1Q2 | Q1Q2 | 0 | 4 | 96 | **2.7024** | 1.2666 | [2.449, 2.956] | Base | 0.3249 | 0.317 | 0.003 |
| `L0_Q1Q2_V5` | L0_Q1Q2 | Q1Q2 | 0 | 5 | 96 | **2.7705** | 1.1972 | [2.531, 3.010] | Base | 0.3930 | 0.402 | 5.7e-04 |
| `L5_Q1Q2_V1` | L5_Q1Q2 | Q1Q2 | 5 | 1 | 96 | **2.5148** | 1.1596 | [2.283, 2.747] | Base | 0.1373 | 0.139 | 0.342 |
| `L5_Q1Q2_V2` | L5_Q1Q2 | Q1Q2 | 5 | 2 | 96 | **2.5843** | 1.1047 | [2.363, 2.805] | Base | 0.2068 | 0.181 | 0.095 |
| `L5_Q1Q2_V3` | L5_Q1Q2 | Q1Q2 | 5 | 3 | 96 | **2.7451** | 1.1760 | [2.510, 2.980] | Base | 0.3676 | 0.338 | 3.9e-04 |
| `L5_Q1Q2_V4` | L5_Q1Q2 | Q1Q2 | 5 | 4 | 96 | **2.7667** | 1.2049 | [2.526, 3.008] | Base | 0.3892 | 0.392 | 1.7e-04 |
| `L5_Q1Q2_V5` | L5_Q1Q2 | Q1Q2 | 5 | 5 | 96 | **2.9001** | 1.1267 | [2.675, 3.126] | Base | 0.5226 | 0.591 | 1.6e-07 |
| `L5_Q1Q2_V6` | L5_Q1Q2 | Q1Q2 | 5 | 6 | 96 | **2.8136** | 1.2205 | [2.569, 3.058] | Base | 0.4361 | 0.473 | 4.2e-06 |
| `L5_Q1Q2_V7` | L5_Q1Q2 | Q1Q2 | 5 | 7 | 96 | **2.9283** | 1.1290 | [2.702, 3.154] | Base | 0.5508 | 0.515 | 3.4e-06 |
| `L5_Q1Q2_V8` | L5_Q1Q2 | Q1Q2 | 5 | 8 | 96 | **2.9631** | 1.1877 | [2.725, 3.201] | Base | 0.5855 | 0.632 | 1.7e-08 |
| `L5_Q1Q2_V9` | L5_Q1Q2 | Q1Q2 | 5 | 9 | 96 | **2.9314** | 1.2124 | [2.689, 3.174] | Base | 0.5539 | 0.525 | 1.4e-06 |
| `L5_Q1Q2_V10` | L5_Q1Q2 | Q1Q2 | 5 | 10 | 96 | **2.9676** | 1.2285 | [2.722, 3.213] | Base | 0.5901 | 0.564 | 7.7e-07 |
| `L0_WAI_V1` | L0_WAI | WAI | 0 | 1 | 96 | **2.5071** | 1.2819 | [2.251, 2.764] | Base | 0.1296 | 0.120 | 0.222 |
| `L0_WAI_V2` | L0_WAI | WAI | 0 | 2 | 96 | **2.6144** | 1.1746 | [2.379, 2.849] | Base | 0.2369 | 0.229 | 0.075 |
| `L0_WAI_V3` | L0_WAI | WAI | 0 | 3 | 96 | **2.3426** | 1.2237 | [2.098, 2.587] | Base | -0.0349 | -0.034 | 0.358 |
| `L0_WAI_V4` | L0_WAI | WAI | 0 | 4 | 96 | **2.5275** | 1.1536 | [2.297, 2.758] | Base | 0.1500 | 0.145 | 0.275 |
| `L0_WAI_V5` | L0_WAI | WAI | 0 | 5 | 96 | **2.5964** | 1.3151 | [2.333, 2.860] | Base | 0.2189 | 0.207 | 0.069 |
| `L5_WAI_V1` | L5_WAI | WAI | 5 | 1 | 96 | **2.4760** | 1.2261 | [2.231, 2.721] | Base | 0.0985 | 0.093 | 0.674 |
| `L5_WAI_V2` | L5_WAI | WAI | 5 | 2 | 96 | **2.5406** | 1.2960 | [2.281, 2.800] | Base | 0.1631 | 0.164 | 0.094 |
| `L5_WAI_V3` | L5_WAI | WAI | 5 | 3 | 96 | **2.3494** | 1.2133 | [2.107, 2.592] | Base | -0.0281 | -0.027 | 0.798 |
| `L5_WAI_V4` | L5_WAI | WAI | 5 | 4 | 96 | **2.5316** | 1.2909 | [2.273, 2.790] | Base | 0.1541 | 0.149 | 0.191 |
| `L5_WAI_V5` | L5_WAI | WAI | 5 | 5 | 96 | **2.5895** | 1.2404 | [2.341, 2.838] | Base | 0.2119 | 0.210 | 0.096 |
| `L0_CSQ8_V1` | L0_CSQ8 | CSQ8 | 0 | 1 | 96 | **2.3304** | 1.1713 | [2.096, 2.565] | Base | -0.0471 | -0.045 | 0.653 |
| `L0_CSQ8_V2` | L0_CSQ8 | CSQ8 | 0 | 2 | 96 | **2.4475** | 1.2452 | [2.198, 2.697] | Base | 0.0700 | 0.068 | 0.604 |
| `L0_CSQ8_V3` | L0_CSQ8 | CSQ8 | 0 | 3 | 96 | **2.4988** | 1.2016 | [2.258, 2.739] | Base | 0.1213 | 0.112 | 0.316 |
| `L0_CSQ8_V4` | L0_CSQ8 | CSQ8 | 0 | 4 | 96 | **2.5734** | 1.2918 | [2.315, 2.832] | Base | 0.1959 | 0.194 | 0.070 |
| `L0_CSQ8_V5` | L0_CSQ8 | CSQ8 | 0 | 5 | 96 | **2.6287** | 1.2654 | [2.376, 2.882] | Base | 0.2512 | 0.240 | 0.022 |
| `L5_CSQ8_V1` | L5_CSQ8 | CSQ8 | 5 | 1 | 96 | **2.3635** | 1.1437 | [2.135, 2.592] | Base | -0.0140 | -0.010 | 0.981 |
| `L5_CSQ8_V2` | L5_CSQ8 | CSQ8 | 5 | 2 | 96 | **2.5383** | 1.2872 | [2.281, 2.796] | Base | 0.1608 | 0.151 | 0.163 |
| `L5_CSQ8_V3` | L5_CSQ8 | CSQ8 | 5 | 3 | 96 | **2.5643** | 1.2196 | [2.320, 2.808] | Base | 0.1868 | 0.187 | 0.068 |
| `L5_CSQ8_V4` | L5_CSQ8 | CSQ8 | 5 | 4 | 96 | **2.5024** | 1.2572 | [2.251, 2.754] | Base | 0.1249 | 0.111 | 0.331 |
| `L5_CSQ8_V5` | L5_CSQ8 | CSQ8 | 5 | 5 | 96 | **2.5987** | 1.2400 | [2.351, 2.847] | Base | 0.2211 | 0.209 | 0.014 |

## 4. Per-model-state means — all six instruments

Each column is the mean over that state's 96 per-conversation CSVs of the named column in the named directory: `Q1_Mean` (`eval/Q1/`), `Q2_Mean` (`eval/Q2/`), `WAI_TotalMean` (`eval/WAI_SR/`), `CSQ8_Mean` (`eval/CSQ8/`), `MI_Mean` (`eval/MI_SAT/`), `MITI_GlobalMean` (`eval/MITI/`). The WAI subscale means (`WAI_Goal_Mean`, `WAI_Task_Mean`, `WAI_Bond_Mean`) are in the CSV twin, not shown here.

| model_state | Q1_Mean | Q2_Mean | Q1Q2_Mean | WAI_TotalMean | CSQ8_Mean | MI_Mean | MITI_GlobalMean |
|---|---|---|---|---|---|---|---|
| `Base` | 2.3604 | 2.3946 | 2.3775 | 2.7075 | 2.1823 | 2.4705 | 2.6849 |
| `L0_Q1Q2_V1` | 2.5063 | 2.4920 | 2.4991 | 2.7396 | 2.2565 | 2.6059 | 2.7135 |
| `L0_Q1Q2_V2` | 2.6771 | 2.7169 | 2.6970 | 2.7899 | 2.3164 | 2.6406 | 2.9167 |
| `L0_Q1Q2_V3` | 2.5521 | 2.5362 | 2.5441 | 2.7708 | 2.2357 | 2.5903 | 2.8411 |
| `L0_Q1Q2_V4` | 2.6750 | 2.7298 | 2.7024 | 2.7474 | 2.2969 | 2.6198 | 2.9635 |
| `L0_Q1Q2_V5` | 2.7333 | 2.8076 | 2.7705 | 2.8238 | 2.3607 | 2.7188 | 3.0547 |
| `L5_Q1Q2_V1` | 2.5167 | 2.5129 | 2.5148 | 2.7821 | 2.2708 | 2.5747 | 2.8802 |
| `L5_Q1Q2_V2` | 2.5583 | 2.6103 | 2.5843 | 2.8750 | 2.2982 | 2.6476 | 2.8828 |
| `L5_Q1Q2_V3` | 2.7083 | 2.7819 | 2.7451 | 2.9054 | 2.2982 | 2.6372 | 3.0469 |
| `L5_Q1Q2_V4` | 2.7792 | 2.7543 | 2.7667 | 2.9010 | 2.3737 | 2.7361 | 3.0651 |
| `L5_Q1Q2_V5` | 2.8438 | 2.9565 | 2.9001 | 2.9644 | 2.4466 | 2.8229 | 3.1771 |
| `L5_Q1Q2_V6` | 2.7687 | 2.8585 | 2.8136 | 2.8889 | 2.3685 | 2.7170 | 3.1016 |
| `L5_Q1Q2_V7` | 2.8646 | 2.9920 | 2.9283 | 2.9887 | 2.4180 | 2.8056 | 3.1276 |
| `L5_Q1Q2_V8` | 2.9187 | 3.0074 | 2.9631 | 2.9974 | 2.4062 | 2.8333 | 3.2109 |
| `L5_Q1Q2_V9` | 2.9062 | 2.9565 | 2.9314 | 2.9653 | 2.3906 | 2.7986 | 3.1901 |
| `L5_Q1Q2_V10` | 2.9500 | 2.9853 | 2.9676 | 2.9852 | 2.4622 | 2.8212 | 3.1823 |
| `L0_WAI_V1` | 2.5063 | 2.5080 | 2.5071 | 2.7665 | 2.2578 | 2.5573 | 2.7682 |
| `L0_WAI_V2` | 2.5958 | 2.6330 | 2.6144 | 2.7378 | 2.2604 | 2.5729 | 2.8802 |
| `L0_WAI_V3` | 2.3250 | 2.3603 | 2.3426 | 2.6502 | 2.1706 | 2.4618 | 2.5703 |
| `L0_WAI_V4` | 2.4937 | 2.5613 | 2.5275 | 2.7474 | 2.2344 | 2.5087 | 2.8594 |
| `L0_WAI_V5` | 2.6187 | 2.5741 | 2.5964 | 2.7882 | 2.3112 | 2.6667 | 2.9323 |
| `L5_WAI_V1` | 2.4667 | 2.4853 | 2.4760 | 2.7118 | 2.2253 | 2.5434 | 2.7760 |
| `L5_WAI_V2` | 2.5437 | 2.5374 | 2.5406 | 2.7682 | 2.2773 | 2.5920 | 2.8333 |
| `L5_WAI_V3` | 2.3042 | 2.3946 | 2.3494 | 2.6128 | 2.1029 | 2.3715 | 2.5729 |
| `L5_WAI_V4` | 2.5063 | 2.5570 | 2.5316 | 2.7023 | 2.1862 | 2.4514 | 2.8021 |
| `L5_WAI_V5` | 2.5938 | 2.5852 | 2.5895 | 2.7934 | 2.2630 | 2.6146 | 2.8307 |
| `L0_CSQ8_V1` | 2.3354 | 2.3254 | 2.3304 | 2.6580 | 2.1745 | 2.4826 | 2.5859 |
| `L0_CSQ8_V2` | 2.4563 | 2.4387 | 2.4475 | 2.6502 | 2.1823 | 2.4878 | 2.6354 |
| `L0_CSQ8_V3` | 2.4750 | 2.5227 | 2.4988 | 2.7622 | 2.2539 | 2.5590 | 2.7995 |
| `L0_CSQ8_V4` | 2.5396 | 2.6072 | 2.5734 | 2.7344 | 2.1888 | 2.5174 | 2.8568 |
| `L0_CSQ8_V5` | 2.6000 | 2.6575 | 2.6287 | 2.7769 | 2.2552 | 2.5382 | 2.8594 |
| `L5_CSQ8_V1` | 2.3208 | 2.4062 | 2.3635 | 2.6562 | 2.1966 | 2.4826 | 2.6615 |
| `L5_CSQ8_V2` | 2.5104 | 2.5662 | 2.5383 | 2.6788 | 2.2148 | 2.5278 | 2.7917 |
| `L5_CSQ8_V3` | 2.5729 | 2.5558 | 2.5643 | 2.7578 | 2.2318 | 2.5712 | 2.8229 |
| `L5_CSQ8_V4` | 2.4729 | 2.5319 | 2.5024 | 2.7431 | 2.2500 | 2.5451 | 2.7422 |
| `L5_CSQ8_V5` | 2.5729 | 2.6244 | 2.5987 | 2.7786 | 2.2383 | 2.6181 | 2.8464 |

## 5. (a) Base vs best PTO, per training oracle x K, on the Q1+Q2 axis

Base = **2.3775** (`eval/Q1|Q2/Base/{0..95}.csv`, n=96). 'best' = the iteration with the highest `Q1Q2_Mean` inside that (oracle, K) cell; 'final' = the highest-numbered iteration.

| training oracle | K | iterations on disk | best state | best Q1Q2_Mean | delta vs Base | dz | p | final state | final Q1Q2_Mean | delta vs Base |
|---|---|---|---|---|---|---|---|---|---|---|
| CSQ8 | 0 | 5 (V1..V5) | `L0_CSQ8_V5` | 2.6287 | +0.2512 | 0.240 | 0.022 | `L0_CSQ8_V5` | 2.6287 | +0.2512 |
| CSQ8 | 5 | 5 (V1..V5) | `L5_CSQ8_V5` | 2.5987 | +0.2211 | 0.209 | 0.014 | `L5_CSQ8_V5` | 2.5987 | +0.2211 |
| Q1Q2 | 0 | 5 (V1..V5) | `L0_Q1Q2_V5` | 2.7705 | +0.3930 | 0.402 | 5.7e-04 | `L0_Q1Q2_V5` | 2.7705 | +0.3930 |
| Q1Q2 | 5 | 10 (V1..V10) | `L5_Q1Q2_V10` | 2.9676 | +0.5901 | 0.564 | 7.7e-07 | `L5_Q1Q2_V10` | 2.9676 | +0.5901 |
| WAI | 0 | 5 (V1..V5) | `L0_WAI_V2` | 2.6144 | +0.2369 | 0.229 | 0.075 | `L0_WAI_V5` | 2.5964 | +0.2189 |
| WAI | 5 | 5 (V1..V5) | `L5_WAI_V5` | 2.5895 | +0.2119 | 0.210 | 0.096 | `L5_WAI_V5` | 2.5895 | +0.2119 |

Same question on each oracle's **own** instrument — the axis it was actually trained on:

| training oracle | metric | Base | best K=0 | delta | p | best K=5 | delta | p |
|---|---|---|---|---|---|---|---|---|
| Q1Q2 | `Q1Q2_Mean` | 2.3775 | `L0_Q1Q2_V5` 2.7705 | +0.3930 | 5.7e-04 | `L5_Q1Q2_V10` 2.9676 | +0.5901 | 7.7e-07 |
| WAI | `WAI_TotalMean` | 2.7075 | `L0_WAI_V5` 2.7882 | +0.0807 | 0.414 | `L5_WAI_V5` 2.7934 | +0.0859 | 0.117 |
| CSQ8 | `CSQ8_Mean` | 2.1823 | `L0_CSQ8_V5` 2.2552 | +0.0729 | 0.304 | `L5_CSQ8_V4` 2.2500 | +0.0677 | 0.383 |

> Only the **Q1Q2-trained** PTO arm moves its own metric by a clearly separated amount (+0.3930 at K=0, +0.5901 at K=5). The WAI- and CSQ8-trained arms move their own instrument by +0.0807 (p = 0.414) and +0.0729 (p = 0.304): **on disk, PTO trained on WAI-SR or CSQ-8 does not measurably improve WAI-SR or CSQ-8.** Their apparent gains in the Q1+Q2 table above (+0.21 to +0.25) are transfer onto a metric they were not trained on.

## 6. (b) K = 0 vs K = 5, within each training oracle

Iteration-matched, paired over the 96 personas, on `Q1Q2_Mean`:

| oracle | iter | K=5 state | K=0 state | K=5 mean | K=0 mean | delta (K5 − K0) | dz | p |
|---|---|---|---|---|---|---|---|---|
| Q1Q2 | 1 | `L5_Q1Q2_V1` | `L0_Q1Q2_V1` | 2.5148 | 2.4991 | +0.0156 | +0.016 | 0.690 |
| Q1Q2 | 2 | `L5_Q1Q2_V2` | `L0_Q1Q2_V2` | 2.5843 | 2.6970 | -0.1127 | -0.111 | 0.733 |
| Q1Q2 | 3 | `L5_Q1Q2_V3` | `L0_Q1Q2_V3` | 2.7451 | 2.5441 | +0.2010 | +0.192 | 0.038 |
| Q1Q2 | 4 | `L5_Q1Q2_V4` | `L0_Q1Q2_V4` | 2.7667 | 2.7024 | +0.0643 | +0.063 | 0.684 |
| Q1Q2 | 5 | `L5_Q1Q2_V5` | `L0_Q1Q2_V5` | 2.9001 | 2.7705 | +0.1297 | +0.137 | 0.295 |
| WAI | 1 | `L5_WAI_V1` | `L0_WAI_V1` | 2.4760 | 2.5071 | -0.0311 | -0.031 | 0.462 |
| WAI | 2 | `L5_WAI_V2` | `L0_WAI_V2` | 2.5406 | 2.6144 | -0.0738 | -0.070 | 0.479 |
| WAI | 3 | `L5_WAI_V3` | `L0_WAI_V3` | 2.3494 | 2.3426 | +0.0067 | +0.007 | 0.678 |
| WAI | 4 | `L5_WAI_V4` | `L0_WAI_V4` | 2.5316 | 2.5275 | +0.0041 | +0.004 | 0.961 |
| WAI | 5 | `L5_WAI_V5` | `L0_WAI_V5` | 2.5895 | 2.5964 | -0.0070 | -0.007 | 0.814 |
| CSQ8 | 1 | `L5_CSQ8_V1` | `L0_CSQ8_V1` | 2.3635 | 2.3304 | +0.0331 | +0.025 | 0.907 |
| CSQ8 | 2 | `L5_CSQ8_V2` | `L0_CSQ8_V2` | 2.5383 | 2.4475 | +0.0908 | +0.090 | 0.414 |
| CSQ8 | 3 | `L5_CSQ8_V3` | `L0_CSQ8_V3` | 2.5643 | 2.4988 | +0.0655 | +0.061 | 0.500 |
| CSQ8 | 4 | `L5_CSQ8_V4` | `L0_CSQ8_V4` | 2.5024 | 2.5734 | -0.0710 | -0.067 | 0.269 |
| CSQ8 | 5 | `L5_CSQ8_V5` | `L0_CSQ8_V5` | 2.5987 | 2.6287 | -0.0301 | -0.028 | 0.809 |

| oracle | mean of the 5 matched deltas (K5 − K0) | # of 5 favouring K=5 | iterations significant at 0.05 (uncorrected) |
|---|---|---|---|
| Q1Q2 | +0.0596 | 4/5 | V3 (p=0.038) |
| WAI | -0.0202 | 2/5 | none |
| CSQ8 | +0.0177 | 3/5 | none |

Best-vs-best (see the caveat under the table):

| oracle | K=5 best | K=0 best | delta | dz | p |
|---|---|---|---|---|---|
| Q1Q2 | `L5_Q1Q2_V10` 2.9676 | `L0_Q1Q2_V5` 2.7705 | +0.1972 | +0.221 | 0.089 |
| WAI | `L5_WAI_V5` 2.5895 | `L0_WAI_V2` 2.6144 | -0.0249 | -0.023 | 0.601 |
| CSQ8 | `L5_CSQ8_V5` 2.5987 | `L0_CSQ8_V5` 2.6287 | -0.0301 | -0.028 | 0.809 |

> ⚠ **The Q1Q2 best-vs-best row is confounded by iteration count.** `L5_Q1Q2` has 10 iterations on disk (V1..V10); `L0_Q1Q2` has 5 (V1..V5). Restricted to the common V1..V5 range, the best K=5 Q1Q2 state is `L5_Q1Q2_V5` = 2.9001 vs `L0_Q1Q2_V5` = 2.7705 — delta +0.1297, p = 0.295. **No K=0 run past V5 exists in this experiment, so 'K=5 wins by +0.20' is not a like-for-like comparison and must not go on a slide as one.**
> The honest one-line read of (b): **iteration-matched, K=5 vs K=0 is a wash in Exp2** — mean matched delta +0.060 (Q1Q2), −0.020 (WAI), +0.018 (CSQ8), with 1 of 15 matched pairs reaching p < 0.05 uncorrected (which is what chance produces at 15 tests).

## 7. GRPO V1 — VOID, section removed

This section previously reported where Exp2's GRPO V1 run landed against Base and against PTO.
**That run had a bug and its scores are void** — see `Exp2_PTO/CLAUDE.md` section "GRPO V1 — VOID".
The numbers are not a weak result, a baseline, or evidence about GRPO as a method; they are not a
result at all, and they have been removed from this file and from `_exp2_summary.csv`.

**Exp2 is a PTO-only experiment.** The PTO-vs-GRPO comparison belongs to Exp3, where both methods
are iterative, share `code/_shared/`, and run at matched MCL, K, candidate budget (M = G = 8),
temperature and oracle.

The naming and provenance notes in section 2 are kept only so that anyone who encounters the
leftover `GRPO_*` / `GRPOI_*` directories on disk can identify them as the void run.

## 8. Reconciliation with the '4,512 convs / 47 models / 9 experiment groups' figure

| claim | reproduces from disk? | what disk says |
|---|---|---|
| **9 experiment groups** | **YES** | `Base`, `L0_Q1Q2`, `L5_Q1Q2`, `L0_WAI`, `L5_WAI`, `L0_CSQ8`, `L5_CSQ8`, `GRPO`, `GRPO-Instruct` = 9 groups spanning the 50 model_state names |
| **47 models** | **NO — the eval tree has 50** | `ls Exp2_PTO/eda/eval/<Q>/` returns 50 dirs, for every one of the six questionnaires |
| **4,512 conversations** | **NO — the eval tree holds 4,800** | 50 states x 96 = 4,800 scored conversations per questionnaire (28,800 CSVs across all six) |

**Nearest reconstruction of 47 / 4,512 — labelled as a reconstruction, not as the doc's stated meaning.** Three scored model states have **no conversation directory left on disk**: `GRPO_E18`, `GRPO_E21`, `GRPO_E24`. `data/conversations_eval/GRPO/` holds only `GRPO_Epoch{3,6,9,12,15}`, while `eda/eval/*/GRPO_E{18,21,24}/` each hold a full 96 score CSVs. So 50 − 3 = **47 model states have both scores and conversations, and 47 x 96 = 4,512**. That arithmetic lands exactly on the claimed pair, so the figure most plausibly counted conversation directories rather than scored model states. **If a deck says '4,512 conversations were evaluated', that is wrong — 4,800 were.**

Full conversation-tree census (`Exp2_PTO/data/conversations_eval/`, files matching `conversation_*.csv`): **58 leaf dirs, 5,368 files**, of which
- 47 dirs are the ones the eval tree scores (96 files each = 4,512);
- `CTRL/LookAhead_0/..._V1..V5` (5 dirs x 96 = 480 convs) exist but are **never scored** — there is no `CTRL` model_state under `eda/eval/`;
- `Q1Q2/LookAhead_0/..._V6`, `Q1Q2/LookAhead_5/..._V6_OLD`, `WAI/LookAhead_0/..._V6` (96 each) and `Q1Q2/LookAhead_5/..._V10_OLD` (86) are unregistered leftovers;
- `CSQ-8/LookAhead_0/..._V6` and `CTRL/LookAhead_0/..._V6` are **empty directories**;
- `WAI/LookAhead_5/..._V1` and `..._V2` each carry one Drive-duplicate file (`conversation_0(1).csv`, `conversation_35(1).csv`) plus 96 legacy `scores_<i>_Q1.csv` files. The loader indexes by `conversation_{i}.csv`, so the duplicates are inert.

## 9. Numbers that do NOT exist in any Exp2 artifact

- **A 4th PTO training oracle.** The eval tree carries PTO arms for exactly **three** training oracles — `Q1Q2`, `WAI`, `CSQ8`. There is no `L0_MITI` / `L5_MITI` / `L0_MI_SAT` / `L5_MI_SAT` model_state anywhere on disk (those names appear only as unused keys in `Conv_EDA.ipynb` cell 3 `EXPERIMENT_PALETTE`). MI-SAT and MITI exist **only as eval instruments**, never as training oracles. Any '4 oracles x K' framing does not reproduce.
- **A CTRL arm result.** `CTRL/LookAhead_0/..._V1..V5` conversations exist (480 files) but were never scored into `eda/eval/`, so there is no CTRL number to quote.
- **A `GRPO_V2` / iterative-GRPO arm in Exp2.** `Conv_EDA.ipynb` cell 5 carries commented-out `GRPOV2_*` registry rows; no such conversations and no such scores exist.
- **Which base checkpoint the `GRPO_*` (non-Instruct) conversations were generated on.** No `grpo_runs/` tree, no run metadata, no adapter config is in the repo, and the two notebooks disagree with themselves (§ 2). Do not state it on a slide.
- **Per-conversation degeneration / phrase-loop rates for Exp2.** Not computed in any Exp2 artifact; only conversation length and `session_endded_by` are recoverable from the conversation CSVs.
- **Any confidence interval, effect size or p-value stored on disk.** `eda/eval/` holds raw per-conversation scores only. Every CI, dz and p in this file was computed here from those scores: n = 96 paired personas, Wilcoxon signed-rank, two-sided, **uncorrected for multiplicity**.

## 10. Provenance

- Score source: `Exp2_PTO/eda/eval/{Q1,Q2,WAI_SR,CSQ8,MI_SAT,MITI}/<model_state>/{0..95}.csv`, one row each; means taken over all 96 rows.
- Conversation source (census + length only): `Exp2_PTO/data/conversations_eval/**/conversation_*.csv`.
- Naming semantics: `Exp2_PTO/eda/Conv_EDA.ipynb` cells 3, 5, 9; `Exp2_PTO/code/Generate_Conversations_GRPO.ipynb` cells 2, 18 (+ the stored cell-2 output); `Exp2_PTO/code/train_GRPO_Oracle_Async.ipynb` cell 2 (+ stored cell-2 and cell-28 outputs); `Exp2_PTO/code/PTO_PrefData_and_Eval.ipynb` cells 11, 12, 15, 17, 43, 49.
- **No SUMMARY.md, STATUS.md, CLAUDE.md or previous deck was read while producing this file.**
