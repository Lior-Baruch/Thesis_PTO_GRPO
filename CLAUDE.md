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
| **PTO** | K ∈ {0, 5}, 7 iters | 4 oracles × K ∈ {0, 5} | **PTO_Exp3** (iterative; lean sibling of GRPO_Exp3, controlled hyperparams matched) |
| **GRPO** | — | V1 (static prompts, weak baseline) | **GRPO_Exp3** (iterative) — both methods now share `code/_shared/` |
| **MCL filter** | — | — | **Wired in both PTO_Exp3 and GRPO_Exp3.** Encoded in `EXPERIMENT_NAME`. |
| **Training reward** | mean(Q1, Q2) | chosen oracle | Q1+Q2 only (matches Exp1) |
| **Eval reward** | Q1, Q2 | per-oracle | all 6 questionnaires |
| **EDA shape** | `Conv_EDA.ipynb` | + per-Q CSVs, `pref_emb/` | + `lib/` package, `Partial_Conv_Oracle_EDA.ipynb`, per-generation `iteration_N/eda/generations.jsonl` |
| **Convs / models** | (paper figures) | 4,512 / 47 | 3,456 / 36 (PTO Exp2 data) + new GRPO/PTO_Exp3 runs pending |

Dirs renamed 2026-05-12 from `ICLR2025/`/`Extension/`/`NewExperiment/`.

## Data lineage
- **Exp1 → Exp2:** independent re-implementation. Stronger oracle, harder patients, JSON-schema rubric, more questionnaires. No data flow.
- **Exp2 → Exp3:** PTO `pref_trees/` and `eval_conversations/` for {Base, Q1Q2, WAI, CSQ-8, CTRL} were **copied** into `Exp3_PTO_GRPO/data/pto_Exp2/`. The Exp2 PTO results stand as a reference baseline. GRPO V1 baseline from Exp2 was **dropped** (Exp3 focuses on PTO_Exp3 vs GRPO_Exp3 only).
- **Exp3 self-loop:** GRPO_Exp3 regenerates its own training data each iter from the current policy; those same convs are the eval set (no separate generate-eval step for trained iters).

## Key methodological shift across experiments
- **Look-ahead K** stayed central throughout (the lever from the ICLR paper).
- **The hard part moved from "can PTO beat the baseline?" (Exp1, settled) to "is GRPO competitive with PTO under matched look-ahead?" (Exp3, open).**
- **Exp3 also exposed a reward-faithfulness concern** the earlier experiments never tested: the `Partial_Conv_Oracle_EDA` shows that the short-cut training reward has only ~0.66–0.73 rank agreement with the full-conv eval at `n_turns=2`. Motivates the `MIN_CONV_LENGTH` knob — now wired in both GRPO_Exp3 (slice filter) and PTO_Exp3 (greedy: tree-start prefix length; independent: branch-point filter); encoded in `EXPERIMENT_NAME` so MCL sweeps stay in disjoint folders.

## Methods (one line each)
- **PTO V1** (Exp1) = original preference-tree exploration + K look-ahead + DPO. Published.
- **GRPO V1** (Exp2) = static prompt set, weak baseline.
- **GRPO_Exp3** = current policy simulates 96 convs → per-turn prompts (MCL filter) → GRPO update with optional K-turn look-ahead. Convs double as eval.
- **PTO_Exp3** = per-turn branching (`M` candidates) → K-turn look-ahead + oracle → τ-filtered (chosen, rejected) pref pairs → DPO update. Lean sibling of GRPO_Exp3. **Two `PREF_TREE_MODE`s:** `greedy` (default, true PTO — start from an MCL-length prefix sliced off the step-1 conv and grow ONE trunk by appending the best-of-M completion at each therapist turn, so the choice feeds the next branch point) and `independent` (branch each patient turn of a pre-recorded conv, no feedback). Mode baked into `EXPERIMENT_NAME`.

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
**Landed (2026-06-07, runtime tuning) — Colab throughput pass on the K=5 arms.**
First full A100-80GB arms were too slow (GRPO ~7 h/iter @ 150 steps; PTO Step-2 dominated —
both K=5 look-ahead/oracle bound, not VRAM). Applied throughput knobs (conv-batch 16→64, oracle
conc 64→128, patient 48→96, look-ahead sub-batch 32→64 GRPO / 32→128 PTO), reverted PTO DPO to
16×1 + grad-ckpt off (A100-80GB; **keep 2×8 + grad-ckpt on for L4/T4**), `EPOCHS_PER_ITERATION 3→2`
(both arms, matched; K=5 + 10 iters kept), a no-op `GREEDY_TRUNK_TARGET_LEN` knob (lower it to
shorten PTO trunks — the big remaining PTO lever), and a GRPO warmup-calc fix (prints the real ~100
steps; LR horizon was always fine). Re-push `code/` + **restart** the runs to apply. See
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Runtime tuning for Colab throughput".
**Launched 2026-06-07:** GRPO LA0, GRPO LA5, PTO LA0 running on Colab (PTO LA5 pending); old
mid-flight run dirs archived with an `(Archive_V2)` suffix.

**Landed (2026-06-07, latest) — ChatML self-play / role-swap leak found in run data + fixed.**
Inspecting the quicktest output (not a crash) exposed a real data-quality bug: base Llama-3.2-1B
**self-plays `<|im_start|>` tokens** (they're not special tokens; the base model never learned the
template). Two failure modes — PTO got empty `<|im_start|>`-spam turns scored 4.5/5 → garbage
(chosen,rejected) pairs; GRPO/conv-gen got **role-swap** (one leaked `<|im_start|>user` line flips
the gpt-4o-mini patient into counselor mode → roles invert for the rest of the conv, also collapsing
GRPO `group_std`→~0.012). **Fix:** `STOP_STRINGS=["<|im_end|>","<|im_start|>"]`, a shared
`clean_completion` cut at the first marker (every decode site), end-conv-on-empty-turn, and a
`REWARD_FLOOR=0.0` for degenerate GRPO completions. **Validated locally:** PTO spam-conv dropped
(real pairs, 0 degenerate rows, roles correct); GRPO 0 leak, roles correct, `group_std` 0.013–2.04
(mean 0.28), floor reached training. (GRPO iter-2 then hit the local Blackwell save-time crash —
hardware, not the fix; full runs are on Colab.) See [Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md)
→ "ChatML self-play leak".

**Landed (2026-06-07, later) — both quicktests PASSED end-to-end LOCALLY; code review found no bugs.**
A 3-agent review of `_shared/` + both trainers + both notebooks (+ source-verified the two flagged
spots) found **no correctness bugs** — every fix below is confirmed in code. Then the FULL notebook
quicktest (not just `_local_smoke.py`) ran top-to-bottom for **both** methods via nbconvert
(`RUN_MODE="quicktest"`, `WANDB_MODE=offline`, a registered venv Jupyter kernel `thesis-venv313` —
the only pre-existing kernel pointed at system Python, which lacks torch/trl): **PTO reached
`iteration_2/adapter/` + `model_iter_2` with NO DPO step-1 OOM and NO PC reboot** (first time the
greedy iter-2 DPO step survived locally); **GRPO** `completions/mean_length` = 48.4 (cap 64) confirms
the stop-string bind held in-loop. `_local_smoke.py all` also 3× PASS. Offline W&B runs sit in each
notebook's `wandb/offline-run-*` (sync with `wandb sync` if wanted; Colab reports live).
**Next: the full K∈{0,5} sweep on Colab (4 arms).** See
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Sweep priority".

**Landed (2026-06-06/07) — first full Colab runs diagnosed + fixed; logging reverted to HF defaults.**
The first full runs (LA5/MCL12/Q1Q2) were stopped: **PTO crashed at the first DPO step** and
**GRPO ran ~11.5 h/iter while reward-hacking length**. Root causes + fixes (validated: py_compile +
import + TRL-config construct + helper unit test + **local GPU smoke** — stop-bind, DPO no-OOM with
grad-ckpt+precompute, GRPO step; `Exp3_PTO_GRPO/code/_local_smoke.py`):
- **PTO OOM** — DPO computes the LM-head logits over the FULL prompt+completion (128k vocab; no
  `logits_to_keep`, unlike GRPO), and greedy trunks are ~2.4k tokens → a ~17 GiB logits tensor
  (×copies/backward) at batch 16; plus `truncation_mode="keep_start"` sliced the *response* off
  over-long prompts. **Fix:** new `_shared/convs.py::build_truncated_training_prompt` caps the DPO
  prompt to `max_allowed_prompt_length` (drop-oldest, keeps system+recent — same as GRPO + matches
  serve-time context) at both pref builders; DPO batch `16→2` × grad-accum `1→8` (effective 16);
  `gradient_checkpointing=True` (`DPO_GRADIENT_CHECKPOINTING` knob).
- **GRPO self-play/length hack** — `<|im_end|>` is template text (not the base eos) and `GRPOConfig`
  set no stop → in-loop sampling ran to the 200-tok cap, self-playing the patient (96% clipped,
  entropy collapse). **Fix:** `GRPOConfig(generation_kwargs={"stop_strings": cfg.stop_strings})`
  (works via the existing `patch_generate` tokenizer injection) + defensive `<|im_end|>` clean in
  `make_reward_fn`.
- **Logging reverted to HF defaults** — the custom `cumulative_global_step` step-axis override fought
  HF's `WandbCallback` (which already defines `train/global_step`) and broke the charts. Now **one
  W&B run per iteration** (grouped), HF's WandbCallback owns the axis; `CumulativeStepCallback`
  removed; `TB_LIVE_LOGGING` defaults **False** (custom continuous view is opt-in); GRPO
  `LOG_COMPLETIONS` back to **True** (TRL's native completions table). `generations.jsonl` EDA still
  written. Quicktest now reports to W&B + params lowered. See
  [Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "First full-run failures + fixes".

**Landed (2026-06-05) — per-generation EDA capture + live TensorBoard.** Each iteration now
writes `data/<method>_Exp3/runs/.../iteration_N/eda/generations.jsonl` — **one row per branch**
(oracle-transcript prefix stored once; all M/G candidates nested with score +
per-questionnaire sub-scores + the K-turn look-ahead `tail`; GRPO rows carry `epoch` + group
mean/std, PTO rows carry candidate `role` + `chosen_idx`). New `_shared/eda_recorder.py`;
records emitted from the GRPO reward fn (`reward.py`) and the PTO branch builders
(`pto_trainer.py`). **Live TB:** `_shared/tb_plots.py::RunTBLogger` writes a continuous
`runs/.../tb_live/` (cumulative-step → smoothable in the TB web UI) + reward histograms +
sample completions; `plot_iteration_metrics` is now method-aware (surfaces the DPO/GRPO-specific
TRL tags the old 2×2 ignored). GRPO inline completion table silenced (`LOG_COMPLETIONS=False`
default). All flag-guarded (`SAVE_EDA_GENERATIONS`, `SAVE_LOOKAHEAD_TRANSCRIPTS`,
`TB_LIVE_LOGGING`); offline-validated, real-model quicktest pending. See
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Per-generation EDA capture + live TensorBoard".

**Landed (through 2026-06-04):** batched look-ahead rollout (`simulate_lookahead_batch`)
+ `LOOKAHEAD_SUB_BATCH_SIZE` knob, **equivalence validated on real GPU** (|Δmean| of
Q1+Q2 reward = 0.024, within oracle noise; 1.5× speedup). **torchao Colab crash fixed**
(peft 0.19.1 raises in `dispatch_torchao` on Colab's pre-baked torchao<0.16.0 → install
cell uninstalls it in both notebooks). **PTO_Exp3 brought to parity with GRPO_Exp3**
(controlled hyperparameters matched, M=8, bf16 toggle, zero-pairs/​split robustness).
Both trainer modules renamed `grpo_trainer.py` / `pto_trainer.py`. **Greedy true-PTO
mode committed (2026-06-04, `e27b9de`):** `PREF_TREE_MODE=greedy` grows ONE trunk via
best-of-M feedback (`grow_preference_trees_batch`); old slice-branch path kept as
`independent`; `_PT{greedy|indep}` baked into `EXPERIMENT_NAME`. **Greedy now slices its
MCL-prefix off the step-1 conv** (no separate prefix-gen pass; `420299b`). **Training oracle
encoded in `EXPERIMENT_NAME`** (`7cbb475`; `{Q1Q2|WAI|CSQ8|MI_SAT|MITI}` token from
`QUESTIONNAIRE_IDS`, matches EDA `oracle=<O>`) → ready for the oracle sweep. **iteration-2
local-crash fix (2026-06-04):** `precompute_ref_log_probs=True` on the PTO DPOConfig
(`DPO_PRECOMPUTE_REF_LOGPS` knob) moves the TRL `"ref"`-adapter forward out of the training
backward step — **isolated iter-2 DPO smoke test PASSED** on the local Blackwell (first time
that step survived; `_iter2_dpo_smoke.py`). GRPO quicktest block trimmed for local 12 GB.
See [Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Look-ahead performance".

**Immediate (DONE 2026-06-07):** quicktest validated end-to-end **locally** for both methods (see the
2026-06-07-later note at the top of this section) — PTO reached `iteration_2/adapter/` + `model_iter_2`
no-OOM/no-reboot; GRPO `completions/mean_length`=48.4 (cap 64). No Colab quicktest needed; go straight
to the full sweep.

**Then (NOW the immediate next):** full sweeps over K ∈ {0, 5} on Q1+Q2 at MCL = 12 (Colab) for **both**
GRPO_Exp3 and PTO_Exp3 (matched), parallel sessions — 4 arms, set `LOOKAHEAD_K` per arm in cell 1
(`EXPERIMENT_NAME` auto-encodes `LA{K}` → disjoint folders). Push `code/` to Drive first; keys from
Colab Secrets. Entries:
[GRPO](Exp3_PTO_GRPO/code/GRPO_Exp3/train_GRPO_Iterative.ipynb) ·
[PTO](Exp3_PTO_GRPO/code/PTO_Exp3/train_PTO_Iterative.ipynb).

## Hardware
Local: Windows, RTX 5070 Ti (12 GB VRAM), CUDA 12.8, torch 2.11.0+cu128.
GRPO_Exp3 training is intended for Colab (GPU); EDA + Run_Eval run locally.
