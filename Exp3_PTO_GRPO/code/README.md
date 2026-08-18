# `code/` — the Exp3 trainers

Two methods over one shared layer. **What each module does** lives here; the *algorithms* they
implement (PTO, GRPO, K-turn look-ahead, MCL, `EXPERIMENT_NAME` schemes) are specified in the root
[CLAUDE.md](../../CLAUDE.md) § "Exp3 · Algorithms", and the dated history is in
[history/CHANGELOG_TRAINER.md](../history/CHANGELOG_TRAINER.md).

```
code/
├── system_prompts_builder.py   V3 patient prompts — THE canonical copy (see "Canonical copies")
├── questionnaires.py           V5 oracle rubrics — JSON schema, 8 instruments incl. PCT + MICI
├── roles.py                    which model plays patient / oracle / judge + the arm-naming contract
├── _shared/                    cross-method modules — BOTH trainers import these
├── GRPO_Exp3/                  train_GRPO_Iterative.ipynb + grpo_trainer.py
├── PTO_Exp3/                   train_PTO_Iterative.ipynb + pto_trainer.py
└── tools/                      stand-alone utilities (moved here 2026-08-18; not imported by the trainers):
    ├── _local_smoke.py             offline smoke tests (stopgen|dpo|grpo) — no OpenAI, tiny, local GPU
    └── generate_eval_convs.{py,ipynb}  generate-only pass for ONE model state (repairs an orphaned adapter)
```

`tools/` scripts put `code/` (and, for `generate_eval_convs`, `code/PTO_Exp3/`) on `sys.path`
themselves, so they run from anywhere; nothing in `_shared/` or the method dirs imports them.

## `_shared/` — the layer both methods import

| Module | Owns |
|---|---|
| `runtime.py` | Colab/local detection, auth (`init_openai_client` / `authenticate`), path resolution, preflight |
| `model.py` | tokenizer / quantization / LoRA, checkpoint discovery, `resolve_start_state` (iteration resume) |
| `convs.py` | conversation state, async generation, per-turn prompt extraction (the **MCL filter**), `clean_completion`, the per-batch `empty_cache()` |
| `reward.py` | oracle scoring, **batched K-turn look-ahead** (`simulate_lookahead_batch`), the reward-fn factory |
| `tb_plots.py` | TensorBoard callbacks, logging lifecycle, TB parser, the post-hoc `plot_iteration_metrics` dashboard |
| `eda_recorder.py` | per-generation capture → `iteration_N/eda/generations.jsonl` (all candidates + scores + look-ahead tails) |
| `lookahead_check.py` | OPTIONAL, off the hot path — serial-vs-batched look-ahead equivalence + an OOM smoke test |
| `timing.py` | **resume-proof per-iteration timing** — appends one line per process to `iteration_N/timing_sessions.jsonl`, so cost survives a crash+resume. ⚠ The older `iteration_metadata.json` `*_time_s` fields are per-PROCESS and undercount every resumed iteration (GRPO_LA5 iter 1 logs 14,501 s for 7.7 h of work; PTO logs `pref_pair_time_s = 3.2 s` for a ~30 min build it reloaded). Read the `cumulative_*` fields when `n_timing_sessions > 1`. The EDA's `eda_analysis/compute.py` prefers this log when present and falls back to artifact-mtime reconstruction for every run that predates it. |
| `__init__.py` | public-API re-exports |

## The two method dirs

Each is `train_<METHOD>_Iterative.ipynb` (the per-iteration orchestration loop, deliberately visible
in the notebook) over `<method>_trainer.py` (`<Method>Config` + `run_one_iteration` +
`run_final_eval` + `write_run_metadata` + `build_wandb_ctx`).

⚠ **The trainer modules are named per method — `grpo_trainer.py`, `pto_trainer.py` — on purpose,**
so `from <method>_trainer import …` cannot collide when both notebooks run in one kernel.

`tools/generate_eval_convs.py` (was `PTO_Exp3/generate_eval_convs.py` until 2026-08-18) is a
generate-only pass for **one** model state; it repairs an *orphaned adapter* (trained, but its
`model_iter_N` convs were never generated) and, with `--conv-dir`, writes a replicate draw
somewhere other than the canonical `model_iter_<N>` folder. Config is rebuilt from the run's own
`run_metadata.json` and seeds are **derived** (`seed+N+1`) — `--verify-seeds` proves that against
decoy offsets before you spend anything. Run it from `code/tools/`:

```powershell
# from code/tools/
& ..\..\..\.venv\Scripts\python.exe generate_eval_convs.py --iter 5 --verify-seeds --dry-run   # free
& ..\..\..\.venv\Scripts\python.exe generate_eval_convs.py --iter 5 --batch-size 6            # the real pass (local card)
```

## `roles.py` — read this before adding any model

Exp3 runs three LLM roles besides the therapist policy, and **they are not equally safe to swap**:

| Role | Swapping it… | Comparable across the swap? |
|---|---|---|
| **patient** | changes the TASK, i.e. the environment | ❌ nothing is |
| **oracle** | changes what the policy optimizes (the training reward) | ❌ arms are not |
| **judge** | changes only the after-the-fact eval scores | ✅ safe + re-runnable |

That asymmetry is why the score lake partitions on `judge=<tag>` but bakes patient/oracle into the
**arm name**. `binding_suffix()` returns `""` when every role is on its default, so default-bound
runs keep byte-identical names and the ~50k CSVs already in the lake stay valid.

⚠ **Do not change `DEFAULT_ORACLE_MODEL` / `DEFAULT_PATIENT_MODEL` without migrating those files.**
Without the suffix, a Gemma-oracle run and a gpt-4o-mini run of the same method+K would write to the
same `eval_scores/.../<Model>/` folder — and `Run_Eval`'s skip-existing resume would report "already
scored" against the *other* model's CSVs, silently.

`roles.py` is stdlib-only and import-light by design: both the trainer and the read-only EDA import
it, and **the EDA must not pull in torch**. Provider SDKs are imported lazily inside `make_client`.

## Canonical copies

`system_prompts_builder.py` and `questionnaires.py` live **only** here. The EDA package's
`constants.py` (its import leaf) prepends `code/` to `sys.path`, so `eda/` imports these exact
files — there is no second copy to drift.

⚠ **`questionnaires.py` is layout-sensitive.** `get_prompt_eval_questionnaire` puts the fixed
instructions + rubric FIRST and the variable transcript LAST so OpenAI's prompt caching hits the
~1,084-token prefix. The margin over the 1,024-token minimum is thin — **don't trim the instructions
or move the transcript ahead of them**, or caching silently stops.

## Running things locally

```powershell
# from code/
& ..\..\.venv\Scripts\python.exe tools\_local_smoke.py all      # stopgen | dpo | grpo | all
```

Tiny, no OpenAI. Validates the stop-string bind, the DPO prompt-cap + no-OOM path, and a GRPO step
on the local GPU (~3 GB peak).

⚠ **Local sm_120 import order: `trl` must be imported BEFORE `torch`** — otherwise CUDA init
segfaults (exit 139). The trainer modules already do this; it only bites if you run something
locally that imports torch first.

⚠ **An over-budget VRAM request REBOOTS this machine** rather than raising `OutOfMemoryError`.
Batch size is a safety setting, not a throughput knob — see CLAUDE.md § "Exp3 · Gotchas" for the
arithmetic (weights 2.6 GB + ≈1.1 GB per concurrent conversation).

Local **training** stays Colab-only. Local *generation* is fine (~50 min per 96 convs at
`--batch-size 6`).
