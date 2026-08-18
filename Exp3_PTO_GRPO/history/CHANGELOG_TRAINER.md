# Exp3_PTO_GRPO — trainer / infrastructure change history

Dated "landed"/"fixed" narratives for the trainers and `code/_shared/`. Newest first within the
sections below. The EDA history is the sibling [CHANGELOG_EDA.md](CHANGELOG_EDA.md); the index is
[CHANGELOG.md](CHANGELOG.md).

The CURRENT behavior these established is summarized in the root [CLAUDE.md](../../CLAUDE.md)
§ "Exp3 · Training internals". Originally moved out of the (now merged) Exp3 CLAUDE.md on
2026-07-08; ordered as they appeared there.

---

## Where GRPO's wall-clock actually goes — measured; three standing hypotheses refuted (2026-08-17)

Prompted by "can we make the training faster?" before resuming the GRPO K=5 arm. Everything below is
measured, and it **replaced** a CLAUDE.md gotcha that had asserted the opposite from a guess.

**Read per-step times off `iteration_N/training/completions/*.parquet` mtimes, not
`iteration_metadata.json`.** ⚠ `training_time_s` is per-**PROCESS**, so a resumed iteration records
only its last session. GRPO LA5 iteration 1 logs 14,501 s but spans 108 steps containing a **40.54 h
gap at step 55**; the true cost is **7.69 h**, nearly 2× the recorded figure. I first computed the K
multiplier as 1.60× off that field and had to retract it. Never quote `training_time_s` for an
iteration that crashed and resumed.

- **K=5 costs 2.4–3.0× K=0 per step** — median 179.6 s vs 74.6 s, mean 261.2 s vs 85.9 s ⇒ look-ahead
  is **~58–67% of a K=5 iteration**. Even the floor (145 s vs 74.6 s) is ~1.94×, which is the physics
  of 5 extra simulated turns per candidate, not overhead.
- **REFUTED: `STOP_STRINGS` is the multi-× slowdown.** Benchmarked locally (bs 8/16/32, 200 tokens, 1B
  bf16): string stopping costs **1.05–1.18× vs no criteria, and the penalty SHRINKS with batch** — the
  per-step criterion is a few small tensor ops, flat in batch. The real cost is **per-CALL**:
  `get_vocab()` iteration order is unstable, so `StopStringCriteria`'s cache key never hits and it
  rebuilds a 128k-vocab table every `generate()` (0.8–1.7 s per build). Memoising the criteria object
  is bit-identical and worth ~3–6% of an iteration. ⚠ Do **not** "fix" it by switching to token-ID or
  `eos_token_id` stopping: the markers are 6-token BPE sequences (`<|im_end|>` →
  `['<','|','im','_end','|','>']`) and not special tokens, so token-ID stopping matches a strict
  subset — a science change, and K-asymmetric.
- **REFUTED: bigger batch buys throughput.** I proposed `TRAIN_BATCH_SIZE` 64→128 with `grad_accum`
  2→1 as "science-neutral"; verification against the pinned TRL source killed it. TRL already issues
  ONE `generate()` per optimizer step over the whole `generation_batch_size`
  (`grpo_config.py:909-911`: unset ⇒ `steps_per_generation = gradient_accumulation_steps`, so
  64 × 2 = 128 either way). Worse, gas 2→1 is **not** gradient-neutral: TRL divides by gas in
  `_compute_loss` and transformers divides again in `training_step` (fires because `GRPOTrainer` sets
  `model_accepts_loss_kwargs=False` and the identity collator leaves `num_items_in_batch` None), so the
  net scale is 1/gas² and halving gas **doubles** the accumulated gradient — enough to start tripping
  `max_grad_norm=1.0` on this arm's measured norms.
- **The one real lever is the look-ahead API latency tail.** ~87/135 steps sit at the floor; 9 steps run
  ~600 s longer — the `openai` default 600 s timeout on the look-ahead patient call. Capping helps
  (~1.4×) but ⚠ a short **TOTAL** budget is dangerous: exhausting the retry loop freezes a sim, the
  oracle then scores a truncated transcript, and under `scale_rewards="group"` one frozen sim shifts
  both the mean and the std of its group of 8. The neutral shape is a **short per-attempt timeout with
  MORE retries** (e.g. 60 s × ≥12). K=0 makes no patient calls during training, so anything here is
  K-asymmetric — record it in `run_metadata.json`.

⚠ **Wall-clock is a reported number in the look-ahead paper** (the cost-asymmetry argument), so any
speedup applied to one arm and not the other must be stamped with the iteration it started at.

**Still unmeasured:** the split of look-ahead's ~60% between patient-API wait and therapist-GPU
generation. `_shared/reward.py` already prints it per call (`Look-ahead: N sims × K=… in …s (… GPU
calls, sub_batch=…)`); it just has never been captured. That number decides whether
`LOOKAHEAD_SUB_BATCH_SIZE` is worth anything.

## Look-ahead knobs persisted + GRPO LA5 resumed to iteration 6 (2026-08-17)

**The invisible-setting bug.** `lookahead_k` and `lookahead_sub_batch_size` live in `LookaheadConfig`
(`_shared/reward.py`), which is **never serialised** — `write_run_metadata` snapshots only
`asdict(TrainingConfig)`. So all four arms' `run_metadata.json` read `sub_batch=None, K=None`: K was
recoverable from `EXPERIMENT_NAME` (`_LA{K}_`), but **sub-batch is in no name and left no trace**, so
changing it would silently make per-iteration wall-clock non-comparable. I had also cited "PTO already
ran sub_batch 128 on an A100" from CLAUDE.md prose and had to retract it — nothing on disk said so.
**Fix:** `lookahead_k: int = 0` and `lookahead_sub_batch_size: Optional[int] = None` added to
`TrainingConfig` (audit-only fields — the live values still come from `LookaheadConfig`), and cell 1
now passes both.

**Config for the resume** (`train_GRPO_Iterative.ipynb` cell 1): `LOOKAHEAD_K` 2 → **5** (load-bearing:
K is in `EXPERIMENT_NAME`, so at 2 it resolves to a nonexistent `_LA2_` folder and trains a brand-new
arm from scratch instead of resuming), `NUM_ITERATIONS` 10 → **6**, `LOOKAHEAD_SUB_BATCH_SIZE` 64 →
**128**. Resumes `iteration_2` from `checkpoint-30`/104.

**Two verification findings worth keeping:**
- **Checkpoint validity is only 3 files** — `HF_TRAINER_FILES` = `adapter_model.safetensors` +
  `adapter_config.json` + `trainer_state.json`. `checkpoint-30` lacks `eda_snapshot.jsonl` and
  `experiment_metadata.json`, which are *not* in that set, so it is valid and resume does **not** fall
  back to `checkpoint-20` (I claimed it would; wrong).
- **`write_run_metadata` overwrites in place**, so a resume would restamp the whole arm with the new
  `num_iterations` and `sub_batch` — including iteration 1, which ran at 64. Pre-resume copy kept as
  `run_metadata_pre_resume_iter1.json`.

**Also corrected:** a stale cell-1 comment claiming "With grad_accum=1, generation_batch_size =
TRAIN_BATCH_SIZE = 128" while the live config is 64 × 2. Its derived numbers were right, which is how
the wrong premise survived — but it invited exactly the gradient-doubling edit refuted above, and it
asserted wall-clock is API-gated.

## Generate-only eval pass for an orphaned adapter + an inter-batch VRAM leak (landed 2026-07-30)

**The gap.** `model_iter_k` is produced as Step 1 of iteration `k+1`, so a run that dies between
"iteration k's adapter saved" and "iteration k+1's Step 1 finished" leaves an adapter that can
**never be scored**. PTO LA5 was exactly there: adapters for iters 1–5, but `model_iter_5` empty
because iteration 6 was killed ~1 min in (adapter saved 02:32, `iteration_6/` created 02:33).

**New tooling** (`code/PTO_Exp3/`), runs on Colab or locally unchanged:
- **`generate_eval_convs.py`** — loads `iteration_N/adapter`, runs *only* `run_generation_only`, writes
  `model_iter_N_TT*_TP*/`. No branching / look-ahead / oracle / DPO. Config is rebuilt from the run's
  own `run_metadata.json` (= `asdict(PTOConfig)`) so it cannot drift from how iters 0..N-1 were made;
  the Colab-absolute `local_outdir`/`conv_outdir` are remapped onto the current host and the
  recomputed tail is asserted against the stored one. `PTOConfig` is frozen, so all overrides are
  applied to the dict *before* construction.
- **Seeds are derived, not typed.** `model_iter_k` ⇐ iteration `k+1`'s Step 1 ⇒ shuffle seed =
  `patient_api_seed` = `cfg.seed + k + 1`, the same formula `eda_analysis.data.persona_order` replays.
- **`--verify-seeds` proves that before spending.** Replays the shuffle for each already-generated
  `model_iter_k` and checks the patient really is the predicted persona, comparing the age the patient
  *states* to `age_value`. Result on PTO LA5: **314 correct / 0 wrong** over iters 0–4. Two subtleties
  it exposed: (a) ~1/3 of patients never state an age — those are UNRESOLVED, not mismatches (a naive
  "does the age appear anywhere" test reported 13–27 false mismatches per iteration); (b) only a
  handful of distinct ages spread over 96 personas, so a *wrong* shuffle still collides ~48% of the
  time — hence two **decoy offsets** that must fail (`seed+k+0` 163 wrong, `seed+k+2` 159 wrong). A
  decoy that passes is reported as INCONCLUSIVE. The script refuses to generate unless the gate passes.
- **`generate_eval_convs.ipynb`** — thin notebook over those helpers (visible orchestration, per the
  repo pattern). Locates `code/PTO_Exp3/` by probing for `pto_trainer.py` rather than trusting cwd (a
  notebook has no `__file__` and VS Code's cwd depends on `jupyter.notebookFileRoot`).
- **Scale-reduction knobs are guarded.** `--num-convs` / `--num-utterances` **require** an explicit
  `--conv-dir`: a partial or short set written into the real `model_iter_N/` would be treated as done
  by the per-CSV resume, so a later full pass would keep it and ship an arm mixing scales/hosts.

**The bug it surfaced — `_shared/convs.py` never freed VRAM between *successful* batches.**
`gc.collect()` + `torch.cuda.empty_cache()` existed only on the OOM/failure paths in
`_run_generation_rounds`. Consecutive batches have different max sequence lengths, so freed blocks are
the wrong size to reuse and the caching allocator keeps requesting more from the driver. Measured on
the 12 GB local card (96 convs, batch 6): **batch 1 peaked 8.0 GB, batch 2 reached 11.9/12.2 GB (97%)**.
A single-batch smoke run cannot surface this. **Fixed:** release after each successful batch too, plus
each batch line now prints `vram <N>G` (`torch.cuda.memory_reserved()`) so a regression is visible
live — it must stay flat across batches. `empty_cache` frees only unused cached blocks, so results are
bit-identical; the cost is one re-allocation per batch.

**Why it matters beyond tidiness (local hardware).** On the RTX 5070 Ti (sm_120, driver 610.62) an
over-budget VRAM request **reboots the machine** instead of raising `torch.OutOfMemoryError` — the same
signature as the earlier local crashes ("no Python traceback, hard GPU/driver fault"). Generation costs
≈1.1 GB per concurrent conversation on top of 2.6 GB of weights, so a `--batch-size 32` attempt asked
for ~38 GB and rebooted the PC. Batch 4 measured 7.1 GB (58%), batch 6 ≈8.0 GB per batch with the fix.
**Do the GB/conversation arithmetic before raising the batch on this card.**

⚠ **Module caching gotcha:** editing `_shared/convs.py` does NOT affect an already-running Jupyter
kernel — Python caches the imported module. If the batch lines lack the `vram` field, the kernel
predates the fix and is running the old code; **restart the kernel**. (Per-CSV resume makes that free.)

## Step-2 (pref-build) resume — automatic (landed 2026-06-07)

Step 2 ("Building pref pairs") is the dominant PTO phase (~41 min at K=0, hours at K=5) and
now **resumes automatically**, mirroring Step 1's per-CSV conversation resume — because
`resolve_start_state` only treats an iteration as done once `iteration_N/adapter/` exists, so
a crash *after* Step 2 but *before* the adapter (e.g. the DPO OOM) used to re-run the whole
build. Two levels, both in [pto_trainer.py](../code/PTO_Exp3/pto_trainer.py):
- **Level A — reload a completed build.** If `iteration_N/pref_pairs/pairs.csv` exists, it's
  reloaded (`_reload_pairs_csv`) and Step 2 is skipped entirely. `pairs.csv` is now both the
  audit trail AND the completion marker (written atomically). On this path the EDA recorder is
  **not** re-flushed (the existing `generations.jsonl` is preserved).
- **Level B — resume a partial build.** The greedy/independent builders own
  `iteration_N/pref_pairs/_progress.json`, an atomic per-step snapshot (greedy: after each
  depth — the lock-step boundary; independent: after each conversation) holding trunk
  `turns`/`next_speaker`/`is_active` + carried pairs + EDA records. On restart they restore
  state and continue; on success `run_one_iteration` deletes `_progress.json`.
- **Guards (`_load_pref_progress`):** a snapshot is only resumed if `mode` + `iteration` +
  config fingerprint `{MCL, M, τ, num_utterances, greedy_trunk_target_len, seed}` + the
  conversation-id set all match the current run — so a checkpoint from a different **τ** (which
  is NOT in `EXPERIMENT_NAME`) is discarded, not silently mixed. Corrupt/missing ⇒ rebuild.
- **Correctness:** resumed trees start with empty `.pairs` (old pairs live only in
  `carried_pairs`) ⇒ no double-count; resume is statistically (not bitwise) equal — post-resume
  completions are freshly sampled, already-emitted pairs are reused verbatim. Validated:
  `py_compile` + an AST-extracted helper unit test (round-trip, empty, numpy-safe, all 4 guard
  mismatches, corrupt/missing). End-to-end greedy/independent resume awaits a real GPU+oracle run.

## Sub-epoch checkpointing + resume (landed 2026-06-08)

Both trainers used to checkpoint **once per epoch** (`SAVE_STRATEGY="epoch"`, `SAVE_TOTAL_LIMIT=1`).
A GRPO epoch is ~50 optimizer steps × ~1.5–2 min/step (G=8 sampling + K=5 look-ahead + oracle), so a
mid-epoch Colab crash threw away ~an epoch. Now both notebooks checkpoint **every `SAVE_STEPS=10`
optimizer steps**.

- **Knobs (cell 1, both notebooks).** `SAVE_STRATEGY="steps"`, new `SAVE_STEPS=10`, `SAVE_TOTAL_LIMIT=2`
  (+ a `SAVE_STEPS>0` validation). A new **required** `save_steps` field on `TrainingConfig`/`PTOConfig`
  threads through `_build_grpo_args`/`_build_dpo_args` into `GRPOConfig`/`DPOConfig` (`save_steps=` is
  honored only when `save_strategy="steps"`). No HF constraint tripped: `save_strategy="steps"` +
  `eval_strategy="epoch"` is legal because neither builder sets `load_best_model_at_end` (the
  "strategies must match" rule only fires when that's True).
- **Why step checkpoints "just work" for resume.** TRL/HF names every checkpoint
  `checkpoint-{global_step}` regardless of strategy, and the existing Case-B path
  ([model.py](../code/_shared/model.py) `resolve_start_state` → `trainer.train(resume_from_checkpoint=…)`)
  reads only the dir-name step + the three required files (`adapter_model.safetensors`,
  `adapter_config.json`, `trainer_state.json`) — all present in a step checkpoint. Step accounting is
  unchanged (`step_delta = global_step − resumed_steps`; the in-progress checkpoint's steps are already
  in the startup offset → no double-count).
- **Hardened resume (walk-back).** Frequent saves raise the odds a crash lands mid-write. New
  `get_latest_valid_hf_checkpoint(training_dir)` ([model.py](../code/_shared/model.py), exported) walks
  checkpoints newest→oldest and returns the first that passes `validate_hf_checkpoint`. Case B now
  resumes from the latest **valid** checkpoint (logs a fallback if the newest is corrupt) and only
  restarts the iteration from scratch if **none** is valid; `compute_cumulative_step_offset` uses the
  same walk-back for the in-progress iteration. `SAVE_TOTAL_LIMIT=2` guarantees a good fallback is on
  disk.
- **Existing/in-flight runs continue with NO migration.** Completed iters resume from
  `iteration_N/adapter/` (Case C, strategy-agnostic); a run crashed mid-iteration under the old epoch
  config resumes from its epoch `checkpoint-N` (a valid integer-named dir), then writes step
  checkpoints going forward (`list_hf_checkpoints` sorts old+new into one monotonic sequence; the old
  epoch ckpt isn't pruned until ≥2 newer ones exist — after we've already resumed from it). To keep a
  run on per-epoch saving, set `SAVE_STRATEGY="epoch"` for that session.
- **Quicktest-safe.** With tiny step counts `SAVE_STEPS` may exceed total steps → zero
  `checkpoint-N` written, which is harmless: the completed-iteration marker is the **separate**
  `iteration_N/adapter/` save (`save_iteration_checkpoint`), which `resolve_start_state` keys off.

### EDA completeness on resume (GRPO-only, same change)

The per-generation EDA buffer ([eda_recorder.py](../code/_shared/eda_recorder.py)) is flushed once at
iteration end, and HF resume **fast-forwards skipped steps without re-invoking the reward fn** — so a
mid-iteration-resumed GRPO iter's `eda/generations.jsonl` used to drop the pre-crash candidates. Fix:
`CheckpointMetadataCallback` ([tb_plots.py](../code/_shared/tb_plots.py)) now takes an optional
`recorder` and, on each `on_save`, also writes `checkpoint-N/eda_snapshot.jsonl` (new
`EDARecorder.snapshot_to`); on a one-shot mid-iteration resume `run_one_iteration` reloads that
snapshot (`EDARecorder.load_from`) **before** training so the end-of-iter flush keeps pre-crash +
post-resume rows. Bound to the **checkpoint dir** so it stays aligned under the walk-back. The
snapshot is extra payload inside `checkpoint-N/` (invisible to `validate_hf_checkpoint` /
`resume_from_checkpoint`); a missing snapshot is a guarded no-op, so pre-feature checkpoints behave
exactly as before. **PTO needs no change** — its recorder is used only in Step-2 (already resume-aware),
and its DPO `CheckpointMetadataCallback` is constructed without a recorder. Caveat: under GRPO inner-loop
`μ>1` (quicktest=2; production=1, exactly clean) one generation batch could double-record at the
boundary — dedupe on read by `branch_id` if it ever matters.

**Validation.** py_compile (all edited files) + GRPOConfig/DPOConfig construct with the steps config +
`get_latest_valid_hf_checkpoint` walk-back unit test (skips a corrupt newest, returns it once complete,
None on empty) + snapshot/reload round-trip + callback `on_save` writes/`recorder=None` skips +
`_local_smoke.py all` (stopgen/dpo/grpo) PASS. **End-to-end crash-resume (assert the resumed iter's
`generations.jsonl` keeps pre-crash rows) awaits a GPU+oracle quicktest.** Re-push `code/` + restart to
apply.

## Look-ahead performance (K>0) — batched rollout LANDED

**Status (2026-06-02).** The K>0 wall-clock bottleneck is fixed:
`simulate_lookahead_batch` in [_shared/reward.py](../code/_shared/reward.py) is now a
**lock-step batched rollout**. All B completions advance in unison (patient →
therapist → …), so each therapist look-ahead turn is **one padded batched
`model.generate`** over the active sims instead of B serial batch-of-1 calls —
collapsing ~B·K serial generations into ~K batched ones. Semantics match the
legacy serial path (statistically equivalent, not bit-identical — sampling RNG
differs). Both GRPO (`make_reward_fn`) and PTO (`build_pref_pairs`,
[PTO_Exp3/pto_trainer.py](../code/PTO_Exp3/pto_trainer.py)) get it through the shared fn.

**How it's safe.** The batched therapist step holds `gpu_lock` per-step (never
across the patient API `await`) with the `eval()` + `use_cache=True` toggle nested
inside, restored in a `finally` (look-ahead runs *during* a GRPO step with the
policy in `train()`). OOM is handled by `_therapist_generate_chunked`: a
chunk-and-halve loop over `generate_therapist_responses_batch` that halves the
sub-batch on OOM (kept **sticky**) and freezes a sim (scores its shorter
transcript) only if even sub-batch=1 OOMs — never aborts the GRPO step. A sim is
likewise frozen on SESSION ENDED, patient-API failure, or an unparseable
transcript (the serial path let parse errors propagate; batched is deliberately
more robust). Verified by a fakes-based logic test (happy path, per-sim freezing,
OOM halving 4→2,2, sub-batch=1 OOM, parse-failure isolation, toggle restoration
after a mid-rollout exception — all pass).

**Knob.** `LOOKAHEAD_SUB_BATCH_SIZE` (notebook cell 1 → `LookaheadConfig.lookahead_sub_batch_size`;
cell 1 now sets **64 (GRPO) / 128 (PTO)** on A100-80GB — see "Runtime tuning for Colab throughput";
`None` = all active sims in one call). Halved automatically on OOM (kept sticky for the rest of the rollout).

**Telemetry.** The existing `reward_fn` line now reports the batched cost:
`Look-ahead: N sims × K=… in X.Xs (… ended early; batched, G GPU calls, sub_batch=S)`.
The legacy `simulate_lookahead_single` / `_generate_therapist_single_async` are kept
(marked LEGACY) as the equivalence-check reference, not on the hot path.

**Validation harness.** [_shared/lookahead_check.py](../code/_shared/lookahead_check.py)
(`make_quick_fixtures` + `compare_serial_vs_batched`) runs both paths on the same
fixtures and prints realized-turn + Q1+Q2 reward mean/std for each plus the batched
speedup. Wired as an **optional section 6 cell** in
[GRPO_Exp3/train_GRPO_Iterative.ipynb](../code/GRPO_Exp3/train_GRPO_Iterative.ipynb)
(guarded by `LOOKAHEAD_K > 0`). Raise `LOOKAHEAD_SUB_BATCH_SIZE` past VRAM to exercise
OOM halving.

**Validation (updated 2026-06-03).** ✅ (a) `compare_serial_vs_batched` equivalence
**passed on real GPU** (Colab, 48 fixtures, K=3): serial Q1+Q2 mean 2.577 vs batched
2.553, **|Δmean| = 0.024** (< oracle noise ~0.07–0.10); identical realized turns 2.88;
1.5× speedup (2 GPU calls, sub_batch=32). 🔄 (b) GRPO_Exp3 **K=3 bf16 quicktest** on
Colab — got through conv generation + prompt extraction, was blocked at the GRPO
training block by the torchao/peft Colab crash (now fixed; re-running). ⬜ (c) Colab
**K=5** arm after the K=3 quicktest trains through. Sequence: ✅ batched fix →
✅ equivalence → 🔄 K=3 quicktest → K=5 arm.

## Per-generation EDA capture + live TensorBoard (landed 2026-06-05)

**EDA capture.** Each iteration writes
`runs/<MODE_TAG>/<EXP_NAME>/iteration_N/eda/generations.jsonl` with **every** candidate the
policy generated (previously PTO kept only the final (chosen,rejected) pair; GRPO kept nothing
per-prompt). Owned by [_shared/eda_recorder.py](../code/_shared/eda_recorder.py) (`EDARecorder`:
in-memory buffer, one atomic flush/iteration — Drive-FUSE-friendly). **Branch-centric schema —
one JSON row per branch:**
- `prefix` (oracle-format transcript of the conv-so-far, stored ONCE), `candidates:[…]` nested
  (each: `completion`, `score`, per-questionnaire `sub_scores`, `oracle{success,retries}`,
  `lookahead{k,realized_turns,ended_early,tail}`), `chosen_idx` (= argmax score).
- `lookahead.tail` = the K simulated turns only (prefix+completion sliced off — exact, since
  look-ahead concatenates). Reconstruct a candidate's oracle-scored text =
  `prefix + "\n\n[THERAPIST]: " + completion + (tail or "")`.
- **GRPO:** one branch row per group **per epoch** (rows carry `epoch` + `group_mean/group_std`);
  recorded in the reward fn ([reward.py](../code/_shared/reward.py) `_record_grpo_generations`,
  reshapes TRL's G-consecutive completions). **PTO:** one row per branch with candidate `role`
  (chosen/rejected/neither); recorded in `_record_pto_branch` (greedy + independent).
- Base full conversations are the already-saved `model_iter_*` eval convs (greedy's base = its
  eval conv) — no separate trunk artifact. EDA load: `read_json(lines=True)` →
  `df.explode("candidates")`.
- Knobs (cell 1): `SAVE_EDA_GENERATIONS`, `SAVE_LOOKAHEAD_TRANSCRIPTS` (drops the per-candidate
  `tail` — the size lever).

**Logging = HF defaults (reverted 2026-06-07).** Training logs go through HF's own
`WandbCallback`/`TensorBoardCallback`: **one W&B run per iteration** (grouped under the experiment
via `wandb_ctx["run_id"]`), charts on the default `train/global_step` axis, TRL's native metrics +
completions table (`LOG_COMPLETIONS=True`). The earlier custom `cumulative_global_step` step-axis
override (in `init_iteration_logging`) + `CumulativeStepCallback` are **removed** — they fought HF's
own `define_metric("*", step_metric="train/global_step")` and broke the familiar charts.
**The custom continuous view is opt-in:** `TB_LIVE_LOGGING` defaults **False**; set it True to also
get [_shared/tb_plots.py](../code/_shared/tb_plots.py) `RunTBLogger`'s one continuous `tb_live/`
SummaryWriter (smoothable cross-iteration curves + reward histograms + sample completions, mirrored
to W&B) plus the EDA aggregates (`eda/*`, `pto/*`, `grpo/*`). The post-hoc matplotlib dashboard
`plot_iteration_metrics` (method-aware: DPO rewards/margins/logps; GRPO reward_std/frac_zero_std/
length) reads the per-iteration `tb_logs/` event files and works regardless. Knobs:
`TB_LIVE_LOGGING`, `TB_SAMPLE_COMPLETIONS_N`, `LOG_COMPLETIONS`.

**Status:** EDA capture validated on the first full runs (`iteration_1/eda/generations.jsonl` written
for GRPO + PTO). Logging revert validated offline (py_compile + import + TRL-config construct);
confirm clean per-iteration W&B charts on the next quicktest.

## Runtime tuning for Colab throughput (2026-06-07)

First full K=5/MCL12/Q1Q2 arms on a Colab **A100-80GB** were far too slow: GRPO
**~7 h/iteration** (150 optimizer steps — `per_device_train_batch_size=64` counts
*completions*, so with `NUM_GENERATIONS=8` that's 16 prompts/step → 803/16×3 ≈ 150),
PTO **Step-2-dominated** (greedy trunks grow 12→49 utts ≈ 18 branching depths, each a
K=5 look-ahead over ~672 candidate sims). The wall is the **K=5 look-ahead** — mostly
*sequential OpenAI API latency* + oracle scoring, which GPU batch size doesn't touch —
not VRAM (GPU sat at ~17 GB in PTO Step 2, ~67 GB in the GRPO step).

- **Throughput knobs (both notebooks cell 1; statistically equivalent, no science
  change):** `CONVERSATION_BATCH_SIZE 16→64`, `ORACLE_MAX_CONCURRENCY 64→128`,
  `PATIENT_API_CONCURRENCY 48→96`, `LOOKAHEAD_SUB_BATCH_SIZE 32→64` (GRPO; step already
  ~67 GB — auto-halves on OOM) / `32→128` (PTO; Step 2 has headroom).
- **DPO batch: kept at the proven `2×8` + grad-ckpt ON (PTO only).** I briefly tried `16×1` +
  grad-ckpt off here for A100 speed — it **OOM'd at the iter-1 DPO step (78.5/80 GB)**. DPO
  materializes logits over the full prompt+completion × 128k vocab with no `logits_to_keep`, and
  **`per_device_train_batch_size` (not the effective batch) sizes that tensor**, so 2→16 made it
  ~8× and grad-ckpt-off also retained all activations. **Reverted to `per_device=2 × grad_accum=8`
  (effective 16) + `DPO_GRADIENT_CHECKPOINTING=True`** — the config from "First full-run failures".
  Negligible cost: DPO is ~2–3 min vs Step 2's ~41 min, so per-device DPO batch is NOT a useful
  speed lever. (If DPO speed ever matters: the liger DPO loss avoids materializing full logits —
  needs `liger-kernel` installed.)
- **`EPOCHS_PER_ITERATION 3→2` (both arms, matched).** ~⅓ off GRPO training (150→~100
  steps/iter); little effect on PTO (DPO is cheap; Step 2 dominates). `NUM_ITERATIONS`
  kept at 10; K=5 kept (the science). Changes absolute scores, not the comparison
  (applied equally to both methods).
- **New PTO lever — `GREEDY_TRUNK_TARGET_LEN`** ([pto_trainer.py](../code/PTO_Exp3/pto_trainer.py)
  `PTOConfig.greedy_trunk_target_len`, wired from cell 1): caps greedy trunk growth via
  `target_len = min(NUM_UTTERANCES_FOR_DATA, GREEDY_TRUNK_TARGET_LEN)`. **Defaults to
  `NUM_UTTERANCES_FOR_DATA` = no-op.** Lower it (e.g. 30 ≈ the partial-oracle EDA's 0.9
  rank-agreement point) to grow shorter trunks → far fewer branching depths → the biggest
  remaining PTO Step-2 speedup. It's a **science change** (shallower trunks/look-ahead
  context) and is **NOT in `EXPERIMENT_NAME`**, so isolate a lowered run by clearing/renaming
  its output dir.
- **GRPO warmup-calc fix** ([_build_grpo_args](../code/GRPO_Exp3/grpo_trainer.py)): now divides
  by the real prompts/step `(train_batch_size/num_generations)*grad_accum`, so the printed
  `total_train_steps` matches the real ~100 (was 21 at 3 epochs). Only the warmup print/value
  was wrong; the cosine LR horizon was always correct (HF Trainer recomputes it from the
  dataloader length).

**To apply:** re-push `code/` to Drive and **restart** the runs (cell 1 is read only at
startup); saved `model_iter_0` conv CSVs are reused via resume, so Step-1 gen isn't repeated.
Expect GRPO ~3 h/iter, PTO ~1.5–2× faster on Step 2.

**Launched 2026-06-07 (tuned config).** Three arms running on Colab: **GRPO LA0, GRPO LA5,
PTO LA0** (PTO LA5 pending). The earlier mid-flight 3-epoch run dirs were archived (renamed
with an `(Archive_V2)` suffix) rather than deleted, so the tuned arms write fresh folders.
**PTO LA0 then OOM'd at the iter-1 DPO step** (the 16×1 + grad-ckpt-off mistake above); DPO config
reverted to `2×8` + grad-ckpt on, re-push + restart the PTO arm. PTO Step 2 took **2454 s / 782
pairs / 37 depths** before the crash (K=0 → no look-ahead; that time is branch-sampling generation
+ oracle scoring only — not yet decomposed into GPU vs API).

## First full-run failures + fixes (2026-06-06/07)

The first full Colab runs (LA5/MCL12/Q1Q2) were stopped — long + API-costly, nothing obvious in
W&B/TB. Diagnosis + fixes (validated: py_compile + import + TRL-config construct + a fake-tokenizer
unit test of the prompt cap):

- **PTO crashed at the first DPO step (OOM).** DPO's `_compute_loss` takes `outputs.logits` over the
  FULL prompt+completion (no `logits_to_keep`, unlike GRPO which restricts to the ~200 completion
  tokens — verified vs TRL 1.4.0 source). Greedy trunks are ~2.4k tokens (max ~6k), so the LM-head
  logits tensor = batch 16 × 2 (chosen+rejected) × ~2248 × 128k vocab × 2 B ≈ 17 GiB (×copies +
  backward → OOM). Latent second bug: `truncation_mode="keep_start"` slices `[:max_length]`, so for a
  prompt longer than `max_length` the *response* is dropped and `completion_mask` is all-zeros. **TB
  looked empty because only the `args`/`model_config` text summaries were written — zero training
  steps.** **Fix:** `build_truncated_training_prompt` ([convs.py](../code/_shared/convs.py)) caps the DPO
  prompt to `max_allowed_prompt_length` (drop-oldest, keeps system+recent — identical to GRPO's
  `extract_prompts_from_conversations`, and matches the serve-time context window) at both pref
  builders; DPO `per_device_train_batch_size 16→2` × `gradient_accumulation_steps 1→8` (effective 16
  unchanged — the batch is what fixes the logits OOM; grad-ckpt does NOT touch the logits tensor);
  `gradient_checkpointing=True` (`DPO_GRADIENT_CHECKPOINTING`; TRL handles the PEFT/precompute
  interplay) so it fits any Colab GPU. NOT the local Blackwell crash — `precompute_ref_log_probs` was
  already on. **(2026-06-07: a 16×1 + grad-ckpt-off attempt on A100 for speed OOM'd at the iter-1
  DPO step — this `2×8` + grad-ckpt-on config is the one that stands. `per_device` batch sizes the
  full-seq logits tensor, so keep it at 2. See "Runtime tuning for Colab throughput".)**
- **GRPO didn't crash but ran ~11.5 h/iter and reward-hacks length.** `<|im_end|>` is template text,
  not the base tokenizer's eos, and `GRPOConfig` set no stop → TRL's in-loop sampling runs to the
  200-tok cap, self-playing the patient's reply (entropy 3.97→1.92, 96% clipped), which both pollutes
  the oracle transcript and trains the ramble. **Fix:**
  `GRPOConfig(generation_kwargs={"stop_strings": cfg.stop_strings})` — `patch_generate` already
  injects the tokenizer so `stop_strings` binds (the same path look-ahead relies on during the step) —
  plus a defensive `<|im_end|>` clean in `make_reward_fn`. (The ~11.5 h/iter cost itself — in-loop K=5
  look-ahead + 3 epochs + look-ahead eval — is config/throughput, not a bug; **addressed 2026-06-07 —
  see "Runtime tuning for Colab throughput".**)

See also "Logging = HF defaults" above (the W&B charts were broken by the custom step-axis override,
now reverted to one HF run per iteration).

## ChatML self-play leak (found + fixed 2026-06-07)

Found by **reading the quicktest output** (`pref_pairs/pairs.csv` + the `model_iter_*` convs), not
from a crash. Base **Llama-3.2-1B self-plays `<|im_start|>` tokens**: they are NOT special tokens
(tokenizer vocab stays 128256; the ChatML template renders them as ordinary BPE text the base model
has never been trained on), so early in training the therapist emits `<|im_start|>` and writes the
*other* speaker's turn as literal text. Two failure modes, one cause:
- **PTO spam** — therapist turns become pure `<|im_start|>assistant/<|im_start|>patient` piles (no
  content); the oracle still scored them ~4.5/5 (it was grading the coherent *patient* turns) →
  degenerate (chosen,rejected) DPO pairs.
- **GRPO / conv-gen role-swap** — one leaked first-person `<|im_start|>user\nI've been struggling…`
  line flips the gpt-4o-mini patient into **counselor** mode → roles invert for the rest of the conv
  (patient calls the therapist "Emma"; therapist discloses problems). Coherent-looking but mislabeled;
  ~2/4 seed convs derailed; also collapsed GRPO `group_std`→~0.012 (near-zero advantages).

**Fix (in code):**
- `STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]` (both notebooks cell 1 + `_DEFAULT_STOP_STRINGS` in
  [_shared/convs.py](../code/_shared/convs.py)) — generation halts the moment a fake turn opens.
- New `_shared/convs.py::clean_completion` cuts at the FIRST marker; used at every decode site
  (`generate_therapist_responses_batch`, [reward.py](../code/_shared/reward.py) look-ahead hot+legacy,
  GRPO `reward_fn`). Empty-after-clean **ends the conversation** (`_process_session_response`);
  look-ahead sims freeze on empty.
- GRPO floors degenerate completions to `REWARD_FLOOR = 0.0` (below the oracle 1–5 range) so a
  self-played turn gets a strong negative group-relative advantage; EDA candidate `score` now records
  the floored/training reward (matches `group_mean/std`). PTO needed no extra logic (its builders
  already drop empty candidates).

**Validated locally (quicktest, 2026-06-07):** PTO spam-conv dropped (real pairs, 0 degenerate rows,
roles correct, both iters complete); GRPO 0 `<|im_start|>` leak across 56 candidates, model_iter_1
convs role-correct, `group_std` 0.013–2.04 (mean 0.28), floor reached training (1 completion → 0.0).
GRPO iter-2 then hit the local Blackwell save-time crash (hardware — training completed, save path
untouched; see Gotchas / the local-crash memory). Full K∈{0,5} sweep runs on Colab regardless.

## Sweep priority (updated 2026-06-11)

**Run status + cost (2026-06-11).** PTO LA0 = 10 iters done; **GRPO LA0 running (iter 6)** (the
fair-endpoint comparison vs PTO is in progress); **both LA5 arms PAUSED for cost** — OpenAI spend
across the Exp3 runs + quicktests hit **~$300** and is now a binding constraint, so RQ-i (K0 vs K5) is
on hold. The bill is dominated by oracle scoring + K=5 look-ahead patient calls (both ∝ candidate
count × iterations); **caching is already maxed** (~50% off the oracle's rubric-first prefix — don't
trim it), so reduce **call COUNT**: cap `NUM_ITERATIONS` ~5–6 (gains plateau by iter ~4 → ~40–50%
saving, still a matched-iter comparison), `M`/`G` 8→4, PTO `GREEDY_TRUNK_TARGET_LEN`↓; keep **K** (the
science) + the **gpt-4o-mini oracle** (comparability with already-scored data) fixed. Patient-model
swap is possible but a science change — avoid. Estimate cost/arm from cell-1 config before launching +
set an OpenAI hard usage limit. See the `project-openai-cost-constraint` memory.

0. **Quicktest (both methods) — ✅ DONE 2026-06-07, validated LOCALLY end-to-end** (not Colab; the
   full notebooks ran via nbconvert, `RUN_MODE="quicktest"`, `WANDB_MODE=offline`, venv kernel
   `thesis-venv313`). PTO OOM fix confirmed (reached `iteration_2/adapter/` + `model_iter_2`, no
   step-1 OOM, no PC reboot); GRPO stop-string fix confirmed (`completions/mean_length`=48.4 < 64
   cap). `_local_smoke.py all` also 3× PASS. Offline W&B runs in each notebook's `wandb/offline-run-*`
   (online project is empty until `wandb sync`; Colab full runs report live). See "First full-run
   failures + fixes" below and the root CLAUDE.md "Next step".

   **To run a notebook headless locally again:** register the venv as a kernel once
   (`.venv\Scripts\python.exe -m ipykernel install --user --name thesis-venv313`), then
   `WANDB_MODE=offline ... -m jupyter nbconvert --to notebook --execute
   --ExecutePreprocessor.kernel_name=thesis-venv313 <nb>` (offline avoids the W&B login hang; the
   default `python3` kernel is the system interpreter and lacks torch/trl).
1. **GRPO_Exp3 + PTO_Exp3 @ K ∈ {0, 5}, MCL = 12 (Colab) — the immediate next action.** 4 arms; set
   `LOOKAHEAD_K` per arm in cell 1 (`EXPERIMENT_NAME` auto-encodes `LA{K}` → disjoint folders); push
   `code/` to Drive first; keys from Colab Secrets. K=3 look-ahead equivalence already ✅ validated.
   **Throughput/epoch tuning applied 2026-06-07 (EPOCHS 3→2, batch + concurrency bumps) — see
   "Runtime tuning for Colab throughput".**
2. Maybe → either method @ MCL = 2.
3. Maybe → other training oracles (WAI-SR / CSQ-8 / MI-SAT / MITI).

## PTO parity + greedy mode + oracle-in-name batch (through 2026-06-04)

Alongside the batched look-ahead rollout above, the same batch landed:
- **PTO_Exp3 brought to parity with GRPO_Exp3** — controlled hyperparameters matched, M=8, bf16
  toggle, zero-pairs/split robustness. Trainer modules renamed `trainer.py` →
  `grpo_trainer.py`/`pto_trainer.py` (a `from trainer import` collision when both notebooks share one
  local kernel — sys.modules cached the first-loaded trainer).
- **Greedy true-PTO mode committed** (`e27b9de`): `PREF_TREE_MODE=greedy` grows ONE trunk via
  best-of-M feedback (`grow_preference_trees_batch`); the old slice-branch path kept as
  `independent`; `_PT{greedy|indep}` baked into `EXPERIMENT_NAME`. Greedy then made to slice its
  MCL-prefix off the step-1 conv — no separate prefix-gen pass (`420299b`).
- **Training oracle encoded in `EXPERIMENT_NAME`** (`7cbb475`): a `{Q1Q2|WAI|CSQ8|MI_SAT|MITI}` token
  derived from `QUESTIONNAIRE_IDS`, identical to the EDA `oracle=<O>` tokens → ready for the oracle
  sweep.
- **Iteration-2 local-crash fix:** `precompute_ref_log_probs=True` on the PTO DPOConfig
  (`DPO_PRECOMPUTE_REF_LOGPS` knob) moves the TRL `"ref"`-adapter forward out of the training backward
  step — the isolated iter-2 DPO smoke test PASSED on the local Blackwell for the first time
  (`_iter2_dpo_smoke.py`). GRPO quicktest block trimmed for the local 12 GB GPU.

## Dependency stack audit (2026-06-01; update 2026-06-03)

*(Moved here from CLAUDE.md 2026-07-12.)* Trainers were audited against the latest docs of the pinned
stack (`transformers==5.8.1`, `trl==1.4.0`, `peft==0.19.1`, `huggingface_hub==1.14.0`,
`wandb==0.26.1`) and **verified current** — nothing deprecated (the then-lingering "TRL v0.28"
comments in the code were cleaned up later, 2026-07-11):
- **`scale_rewards="group"`** (grpo_trainer.py) is the TRL **default** (`"group"/"batch"/"none"`), not a stale value.
- **async reward fn** (_shared/reward.py) is natively awaited by TRL 1.x (`inspect.iscoroutinefunction` → `asyncio.gather`); extra dataset columns forwarded as kwargs; per-sample `None` supported.
- `processing_class=`, `eval_strategy=` already on the new transformers-5/TRL-1 API.
- `hf_xet` is a **required transitive dep** of `huggingface_hub` 1.x — already installed, nothing to add.
- `gpt-4o-mini-2024-07-18` (patient + oracle) has **no API retirement date** per OpenAI dev docs (the only relevant shutdown is `gpt-4o-2024-05-13`, a different model).

Same-session polish: both notebooks' Colab install cell **pinned to requirements.txt** (commented;
`weave` dropped), `authenticate()` sets `WANDB_LOG_MODEL="checkpoint"` (versioned adapter artifact,
third backup), and both configs set `run_name=current_adapter_repo`.

**Update 2026-06-03.** Install cell now also (commented) `%pip uninstall -y torchao` — Colab pre-bakes
torchao<0.16.0, which peft 0.19.1 rejects by *raising* inside `get_peft_model`'s `dispatch_torchao`
(crashed both trainers at iter 1). A100 optimizer batch raised to **16 decision-points/step** (GRPO
`TRAIN_BATCH_SIZE`=128, PTO DPO 16×1 — the DPO half later reverted to 2×8, see "Runtime tuning"; LR
held). `NUM_ITERATIONS` 8→10 both.

