# Root CHANGELOG — Thesis_PTO_GRPO

Dated "Landed (date)…" change history, moved out of the root [CLAUDE.md](../CLAUDE.md) on 2026-07-07 to keep that file scannable (durable reference only). Newest first. Finer-grained per-experiment history lives in each experiment's own `history/CHANGELOG.md`.

---

**Landed (2026-07-08, latest) — GRPO LA0 FINISHED + re-scored: the fair-endpoint PTO-vs-GRPO comparison
is in hand; plus an Exp3 EDA hardening/refactor day.** GRPO LA0 reached the matched 10-iteration
endpoint and was scored on the full battery: **PTO beats GRPO at matched iter 10 (Q1+Q2 4.26 vs 3.75;
paired +0.51, dz 0.73, Holm p<0.001)** because GRPO peaks at iter 8 (4.08) then regresses into
sycophancy (affirmation drift running away: MICI 0.84 at iter 10 vs PTO 0.49), while PTO climbs stably.
Both LA5 arms remain paused for cost (~$300 OpenAI spend), so RQ-i (K0 vs K5) stays on hold. Same day:
a 20-commit Exp3 EDA hardening + package-refactor pass (`_selfcheck` guard, parquet cache, `constants.py`
leaf, plotting/plotting_style split, `oracle_scoring/` pruned to the Run_Eval scoring path, notebooks
output-clean, CLAUDE.md pruned to a lean current-state doc) — full detail in
[Exp3_PTO_GRPO/history/CHANGELOG.md](../Exp3_PTO_GRPO/history/CHANGELOG.md). *(Entry added
retroactively 2026-07-11.)*

**Landed (2026-07-07) — Exp3 EDA backlog #7 (general review) DONE + judge-prompt fix + honest
advantage signal.** Two commits (`f5e5d63`, `266ceaf`), driven by a 3-reviewer sweep + Lior's handoff
(verdict: methodology sound, remaining risk is write-up *framing*, not code; excluded: no CoT judge fields,
no Q1/Q2 edits). **MI-SAT domain bug** — items were hard-coded to "diabetes" but personas are only
smoking/obesity → reworded goal-agnostic in [questionnaires.py](Exp3_PTO_GRPO/code/questionnaires.py) and
**re-scored all 2,784 convs** (0 errors); means rose **uniformly ~+0.14** (old wording rated an intervention
that never happened), no relative conclusion changed. **Honest advantage signal** — added an unfiltered PTO
`group_range` beside GRPO's as the true like-for-like to the τ-filtered `margin`; **caught a grouping bug
mid-work** (PTO `branch_id` is the trunk *depth* and collides across conversations — must key on
`conversation_id` too; the naive key gave a spurious "PTO 8× more decisive"). Corrected: per-branch spread
modest+comparable (~0.23 PTO vs ~0.29 GRPO), and the τ-filter mildly *inflated* PTO's apparent decisiveness.
**MITI rate-normalization** (behaviour counts now per therapist turn). **Framing** (notebook markdown):
confirmatory-vs-exploratory split (PTO>GRPO on Q1+Q2 at **final AND best** iter), reward=outcome +
shared-oracle confounds named + anchored on reward-independent text metrics, **PCT loads WITH warmth**
(ρ≈0.79–0.94, not orthogonal), K0-vs-K5 descriptive-only banners, new
[eda/LIMITATIONS.md](Exp3_PTO_GRPO/eda/LIMITATIONS.md). **Hardening**: dead buggy `rank_table` deleted,
`omnibus` eps_sq→eta_sq relabel, palette-keyed colors (PTO cool/GRPO warm), `render_views` split VIEWS vs
DEFAULT_VIEWS (bare run = all/L0/L5). Re-rendered all 3 views (no failures). See the
`project-pto-branch-id-depth` + `project-orthogonal-eval-axes` memories + [Exp3 CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md).

**Landed (2026-07-03) — Exp3 EDA backlog CLEARED (all 6 concrete items #1–#6; only #7 general
review remains).** One session, six commits (`d446f31` `4246a22` `e492a25` `338efb2` + two stray-folder
cleanups); every change re-rendered across the 3 views (all/L0/L5) via `render_views.py`, no failures.
**#1 grid+subfolder everywhere** — the `1_Outcomes` combined-grid + per-metric-subfolder pattern now spans
all multi-panel families: `2_heterogeneity/<trait>_all_metrics.png` (new `plots.heterogeneity_overview_grid`,
metric×arm persona grid), `3_mechanism/behavior/<metric>.png` (new `single_behavior_trajectory`),
`3_mechanism/subscales/<parent>.png`, `4_training/reward_distribution/<arm>.png`. **#2 GRPO margin analog** —
`advantage_signal_by_iter` emits GRPO `group_range` (per-group best−worst reward), plotted beside PTO's
chosen−rejected margin on a SHARED oracle-score-gap axis. **#3 question-rate "bug" = NOT a bug** — B3_Q(count)
vs q_per_turn(rate) was count-vs-rate confusion; harmonized, the merge is conv-aligned 96/96 and the real
divergence is question *syntax* (regex `?`, collapses ~7×) vs *function* (oracle B3_Q, drops ~1.6×). Shipped an
alignment guard + disambiguated labels + fixed an overstated §4b caption. **#4 labels** — validated-instrument
acronym kept up-front (`MITI (MI Integrity)` …) + new `short_label()` for dense figures. **#5 warmth-vs-
orthogonal** — heatmap block divider + labels, loadings coloring, §3 two-family explainer. **#6 stats.py audit —
NO correctness bugs** (Holm/BH-FDR verified identical to statsmodels; tables reproduce the known headline);
documented Holm family-scope + `trajectory_test` non-independence. **Three write-up-worthy findings surfaced:**
PCT empirically loads WITH warmth (ρ≈0.79–0.94, not as orthogonal as intended); the question syntax-vs-function
divergence is itself the affirmation-drift signature; and PTO's preference margin is LARGER at K=5 than K=0
(look-ahead → more decisive oracle separation). See [Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) →
backlog (#1–#6 marked done) + the `project-exp3-new-eda` memory. Data state unchanged: full L0, partial L5, no L2.

**Landed (2026-07-02) — EDA reorg-by-topic + reward-hacking figures + readable labels.** Two
sessions on the Exp3 analysis EDA. **(A) Reward-hacking figures + labels (committed `a3b3adb`).** Added
`reward_hack_panel` (twin-axis warmth↑ + MICI↑ + PCT-flat), auto peak-marking on the Q1+Q2 curve
(GRPO's iter-8 peak-then-regress), an orthogonal effect forest with lower-is-better handling,
per-metric heterogeneity + `subgroup_endpoint_bars` (GRPO's late regression concentrates on *Resistant*
personas), a `question_rate_crosscheck`, and a central **readable-labels** layer
(`DISPLAY_NAMES`/`arm_label`, applied at draw time only — never renames data keys). **(B) Full reorg
(commits `17b16bd`+`9696cca`).** The EDA is now **topic notebooks ↔ numbered result families, 1:1**
(notebook number == `results/<view>/{figures,tables}/<N_family>/` number): `1_Outcomes`/`2_Heterogeneity`/
`3_Mechanism`/`4_Training_and_Reliability`/`5_Preference`/`6_Stats`. Dropped 4 duplicate figures ONLY;
per-metric trajectory + heterogeneity catalogs added; stats tables merged (main_results final+best;
vs-base/method/K paired) + a new `grpo_iter9_check`; labels `Q1Q2→"Q1+Q2"`; exports gained per-call
`group=` (nested) + a walk-based `build_index()` in every notebook. Validated: package smoke + full
3-view render (all/L0/L5) no failures. **A 7-item backlog for the next EDA session** (grid+subfolder
style elsewhere, GRPO preference-margin analog, resolve Questions-vs-Questions/turn, original-acronym
labels, warmth-vs-orthogonal explainer, refine stats, general review — START by asking clarifying
questions) is in [Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "NEXT EDA SESSION — backlog". Data
state: **full L0 + partial L5, no L2**. See the `project-exp3-new-eda` memory.

**Landed (2026-06-14) — orthogonal eval axes + EDA control/exports overhaul + updated results.**
Two threads. **(A) Eval made multi-dimensional.** The 6 rubrics correlated at PC1≈91% (all subjective
warmth halos), so two **orthogonal questionnaires** were added to [questionnaires.py](Exp3_PTO_GRPO/code/questionnaires.py):
**PCT** (patient change-talk vs sustain-talk + readiness, ID 8) and **MICI** (MI-inconsistent therapist
behaviors incl. over-praise/sycophancy, ID 9, lower=better), plus the *free* derived MITI-proficiency
ratios **R:Q / %CR / %MICO** promoted to first-class outcomes. Scored for all arms via `Run_Eval`.
**Result: PC1 drops 91%→≈56%** — warmth is one factor; technique + MI-inconsistency form a second. The
`text_metrics` semantic regexes were demoted to a lexical sanity-check (affirmation now = oracle MITI_B6_AF
/ MICI over-praise). **(B) EDA refactored for control + organization (2 passes).** A single flat-globals
**`EdaConfig`** (cell 1) now controls arms / metrics / selection / **focus_arms** / plot scales / exports;
figures save as **PNG**, tables as **md + xlsx** (per-group Excel workbook); artifacts route into
per-notebook `results/<figures|tables>/<group>/` + a master `INDEX.md` + provenance banners. Notebooks
**7→6**: thin **`0_Headline`** + merged **`1_Eval_and_Behavior`** + `2_Training` / `3_Reward_Reliability`
/ `4_Preference_LatentSpace` / `5_Detailed_Stats`. Repeated per-arm/K figure loops collapsed into ONE
configurable cell each (`overlay_trajectory`, `heterogeneity_grid`, `arms=`/`select_scores`); the
confusing PC1×PC2 biplot replaced by a readable **`factor_loadings_bars`**. **Updated results** below.
Validated: package smoke + all 6 notebooks via nbconvert (`thesis-venv313`). See
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Eval results so far" + the `project-orthogonal-eval-axes`
memory.

**Landed (2026-06-10, latest) — EDA readability + restructure-by-purpose (3 review rounds with Lior).**
Three iterative passes on top of the research-question refactor below, driven by Lior's feedback:
**(round 1)** fixed the four poorly-reading figures — pooled the 4 near-identical arm-bases into one
descriptive `Base` (`scores.collapse_base`), replaced the unreadable subscale grouped-bar wall with
subscale **trajectories**, added per-iteration **preference drift** (word heatmap + MI-concept lines),
saturated faint tints. **(round 2)** full-name labels (no abbreviations), Okabe-Ito **colourblind
palette** (PTO cool / GRPO warm / Base grey), base **reference line** on bars, base bar in the headline,
evergreen **concise markdown** with an explicit **`[EVAL]` vs `[TRAINING]`** tag per section (a real
source of confusion), removed the QC section, per-view selection (no global toggle). **(round 3)
reorganized the notebooks BY PURPOSE** into **7**: `0_Headline` / `1_Eval_Results` /
`2_Behavior_and_Mechanism` / `3_Training_Diagnostics` / `4_Reward_Reliability` /
`5_Preference_LatentSpace` / `6_Detailed_Stats`; **all heavy tables moved to `6`** with the "did it
work" shown as an **`effect_forest`** dot-plot; **thin arms filtered** (no NaN); **violins dropped**.
New analyses: `3` surfaces the **TensorBoard training curves** (`training.tb_curves`, self-contained
parse — no torch/trl import); `4` **rebuilds the Exp2 partial-conv reliability curve on Exp3 data** from
the per-branch `prefix` in `generations.jsonl` (no new oracle pass) — finding: **GRPO's proxy grows MORE
faithful with conversation length (0.86→0.94) while PTO's grows LESS (0.87→0.76)**, and LA5 ≥ LA0 (look-
ahead helps faithfulness slightly); `5` gains **direction-drift (2D), learned/unlearned words, K0-vs-K5**.
**Validated:** package smoke + all 7 notebooks ran top-to-bottom via nbconvert (`thesis-venv313`). See
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Restructure-by-purpose pass" + [eda/README.md](Exp3_PTO_GRPO/eda/README.md).

**Landed (2026-06-10) — EDA refactored: readable, method-symmetric, by research question.**
The Exp3 analysis EDA was reorganized on top of the `eda/eda_analysis/` package (the 2026-06-09 rebuild). The
mess was in the notebooks: byte-identical cell-1 boilerplate, the same analysis duplicated across
notebooks, hardcoded `PTO_LA0` everywhere (heterogeneity/behavior/transcripts/deep-dive ran for PTO
only), a method-gated `if GRPO…else…` advantage cell, and buried cross-method comparisons. Fixes:
(1) **hybrid plotting** — the recurring figures are now named functions in `eda_analysis/plots.py` (defined
once, called from multiple notebooks); (2) **`eda_analysis.notebook_setup()`** collapses the boilerplate to
`S = eda_analysis.notebook_setup()`; (3) **notebooks reorganized by research question** — `00_Main_Results`
(thin) / `01_Did_It_Work` / `02_PTO_vs_GRPO` (absorbs `Exp3_DeepDive`) / `03_LookAhead_K` /
`04_Mechanism_and_Behavior` / `05_Preference_LatentSpace`; (4) **full method-symmetry** — every per-arm
analysis runs for both methods + training internals shown side-by-side (only the preference probe stays
PTO-only, by construction); (5) new first-class helpers `stats.paired_method_comparison` /
`paired_k_comparison`, `training.advantage_signal_by_iter` / `reward_distribution_frame`,
`pref.pref_word_ranking`; (6) **exports trimmed to one format each** — PDF figures + Markdown tables,
idempotent `CAPTIONS.md`. **Validated:** package smoke test + all six notebooks ran top-to-bottom via
nbconvert (`thesis-venv313`) on the current disk state. See
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "EDA refactor (2026-06-10)" + [eda/README.md](Exp3_PTO_GRPO/eda/README.md).

**Landed (2026-06-09) — EDA rebuilt research-grade + first cross-method results.** Exp3's
analysis EDA was rebuilt as the `eda/eda_analysis/` package + notebooks (since reorganized — see the 2026-06-10
entry above), with true-persona recovery, both stat batteries + repeated-measures (Friedman), and a
thesis-export layer (`results/` figures + tables). Old Exp2 EDA frozen in `eda/archive_exp2/`.
**First results (this was the 0–3 GRPO snapshot; superseded by the 2026-06-14 entry above — kept as
history).** PTO LA0 3.00→4.26; GRPO LA0 reached 3.99 in 3 iters and *looked* to climb 2.4× faster
(slope 0.29 vs 0.12). **With GRPO since extended to iter 8 that fast-slope read normalized to a near-tie**
(slopes ~0.12–0.13 both; alternating tiny edges), and the "is GRPO hacking too?" question is now
**answered yes** (GRPO also affirmation-drifts late). Current numbers:
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Eval results so far" +
the `project-pto-la0-eval-results` memory.

**Landed (2026-06-08) — sub-epoch checkpointing + hardened resume (both trainers).**
Epochs are long (GRPO ~50 opt-steps/epoch × ~1.5–2 min/step with K=5), so per-epoch saves risked
losing ~an epoch on a Colab crash. Both notebooks now checkpoint **every `SAVE_STEPS=10` optimizer
steps** (`SAVE_STRATEGY="steps"`); a new required `save_steps` field on `TrainingConfig`/`PTOConfig`
threads into `GRPOConfig`/`DPOConfig`. `SAVE_TOTAL_LIMIT 1→2` + a new
[model.py](Exp3_PTO_GRPO/code/_shared/model.py) `get_latest_valid_hf_checkpoint` make Case-B resume
**walk back to the newest *complete* checkpoint** instead of discarding the iteration on a corrupt
newest write. **Existing/in-flight runs continue with no migration** (resume is format-agnostic; old
per-epoch checkpoints stay valid resume points — set `SAVE_STRATEGY="epoch"` to keep per-epoch). Also
landed an **EDA-completeness-on-resume** fix (GRPO-only): `CheckpointMetadataCallback` snapshots the
per-generation buffer into each `checkpoint-N/eda_snapshot.jsonl` (new `EDARecorder.snapshot_to/
load_from`), reloaded on resume so a crashed iteration's `generations.jsonl` keeps its pre-crash rows
(snapshot is extra payload inside the ckpt dir — invisible to HF resume; absent-snapshot = graceful
no-op). PTO unaffected (its recorder is Step-2 only). Validated: py_compile + GRPOConfig/DPOConfig
construct + walk-back unit test + snapshot/reload unit tests + `_local_smoke.py all` PASS; end-to-end
crash-resume awaits a GPU+oracle quicktest. **Re-push `code/` + restart** to apply. See
[Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) → "Sub-epoch checkpointing + resume".

**Landed (2026-06-07, latest) — automatic resume for the PTO Step-2 pref build.**
The DPO-OOM crash exposed that a failure after Step 2 but before the adapter re-ran the whole
~41-min pref build. Fixed in [pto_trainer.py](Exp3_PTO_GRPO/code/PTO_Exp3/pto_trainer.py): (A)
if `iteration_N/pref_pairs/pairs.csv` exists it's reloaded and Step 2 is skipped; (B) the
greedy/independent builders checkpoint per-step to `_progress.json` and resume mid-build. Guarded
on mode+iteration+config(incl. τ, not in EXPERIMENT_NAME)+conv-id set; atomic writes; no
double-count. Validated by py_compile + a helper unit test; end-to-end resume awaits a real
GPU+oracle run. **Restart PTO LA0 to benefit** (its crashed-run `pairs.csv` with 782 pairs is on
disk → skips straight to DPO). See [Exp3_PTO_GRPO/CLAUDE.md](Exp3_PTO_GRPO/CLAUDE.md) →
"Step-2 (pref-build) resume".

**Landed (2026-06-07, runtime tuning) — Colab throughput pass on the K=5 arms.**
First full A100-80GB arms were too slow (GRPO ~7 h/iter @ 150 steps; PTO Step-2 dominated —
both K=5 look-ahead/oracle bound, not VRAM). Applied throughput knobs (conv-batch 16→64, oracle
conc 64→128, patient 48→96, look-ahead sub-batch 32→64 GRPO / 32→128 PTO), **kept PTO DPO at 2×8 +
grad-ckpt on** (a 16×1 + grad-ckpt-off attempt OOM'd the A100 at the iter-1 DPO step — per-device
batch sizes the full-seq 128k-vocab logits tensor; DPO is ~minutes vs Step-2's ~41 min so it isn't
worth raising), `EPOCHS_PER_ITERATION 3→2`
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
