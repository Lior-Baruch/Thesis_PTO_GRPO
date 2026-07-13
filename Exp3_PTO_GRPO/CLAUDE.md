# Exp3_PTO_GRPO — ACTIVE (main thesis chapter)

Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle. Two methods compared
under matched look-ahead + oracle:
- **PTO_Exp3** (preference-tree → DPO loss). Lean sibling of GRPO_Exp3 (one notebook
  + one `pto_trainer.py`, sharing `_shared/`). **Hyperparameters matched to GRPO_Exp3:**
  NUM_ITERATIONS=10, MCL=12, K∈{0,5}, gen temps + API concurrency; M
  (`NUM_BRANCHES_PER_TURN`)=8 mirrors GRPO's `NUM_GENERATIONS`; `DPO_BETA`=0.1 (DPO loss
  temp, not GRPO's KL β). bf16 (`USE_4BIT` toggle). Output dir: `data/pto_Exp3/`.
  **Two data-gen modes via `PREF_TREE_MODE`:** `greedy` (default, true PTO — grow ONE
  trunk from an MCL prefix by appending best-of-M) and `independent` (branch each patient
  turn of a pre-recorded conv, no feedback). Baked into `EXPERIMENT_NAME`
  (`_PT{greedy|indep}`) so the arms never collide. See the algorithm section below.
- **GRPO_Exp3** (iterative). Shares `_shared/` with PTO_Exp3. **LA0 finished 10 iters;
  LA5 = base + iter 1 scored** (paused for cost — see "Run status" below).

Reward (training) = **Q1 + Q2 only**, matching the ICLR look-ahead paper.
Reward (eval) = all six MI questionnaires (Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI).

## Trainer pattern

Both trainers (`code/GRPO_Exp3/`, `code/PTO_Exp3/`) follow the same shape:

```
<METHOD>_Exp3/
├── train_<METHOD>_Iterative.ipynb   thicker — per-iteration orchestration visible
└── <method>_trainer.py              <Method>Config + run_one_iteration + run_final_eval + write_run_metadata + build_wandb_ctx
                                     (named per method — grpo_trainer.py / pto_trainer.py — so `from <m>_trainer` can't collide in a shared kernel)
```

with the per-iteration loop composed *visibly in the notebook* (no
black-box `run_iterative_training` call). Helpers shared across both methods
live in [code/_shared/](code/_shared/).

## Algorithms (PTO + look-ahead, GRPO + look-ahead)

Both methods are **iterative**: each iteration regenerates training data from
the *current* policy, performs an update, swaps the adapter, and repeats. They
share the conversation-simulation + oracle-scoring + K-turn look-ahead
machinery (in [code/_shared/](code/_shared/)) and diverge only in (a) how they
turn rollouts into training data and (b) which TRL trainer they use.

**Shared notation.**
- `π_n` — therapist policy at the start of iteration `n` (a LoRA adapter on top of
  the frozen Llama-3.2-1B base; `π_0` = base, no adapter).
- `P` — patient simulator (`gpt-4o-mini`), conditioned on a unique per-patient
  system prompt (one of 96 permutations).
- `O` — oracle scorer (`gpt-4o-mini` with JSON-schema-constrained output);
  scores a conversation on Q1+Q2 (a 22-item MI rubric) and returns the mean.
- `MCL` — `MIN_CONV_LENGTH`, minimum number of utterances in the
  conversation-so-far before a slice/branch is eligible for training.
- `K` — `LOOKAHEAD_K`, number of extra simulated turns appended after each
  candidate completion before the oracle scores it. `K=0` disables look-ahead.

### K-turn look-ahead (shared subroutine)

Given a conversation prefix `c` (a transcript ending on a patient turn) and a
candidate therapist completion `t`, look-ahead simulates `K` more alternating
turns:

```
c + t + P(c+t) + π_n(c+t+P(...)) + P(...) + ... + π_n(...)
```

i.e. the patient replies to `t`, the policy replies to that, etc., for `K` total
extra utterances. The resulting extended transcript is what the oracle scores.
The motivation, from the ICLR paper: scoring `(c + t)` alone rewards
"openings that look good in isolation" while scoring `(c + t + K future turns)`
rewards "openings that *lead somewhere good* under the current policy."

Patient turns go through the async OpenAI API (bounded concurrency); therapist
turns run on the local GPU and are serialized through an `asyncio.Lock` so they
don't trample each other. See [_shared/reward.py](code/_shared/reward.py).

### GRPO_Exp3 + K-turn look-ahead

**Per iteration `n` (loop body in [GRPO_Exp3/train_GRPO_Iterative.ipynb](code/GRPO_Exp3/train_GRPO_Iterative.ipynb), helpers in [grpo_trainer.py](code/GRPO_Exp3/grpo_trainer.py)):**

1. **Generate rollouts.** `π_n` simulates 96 conversations versus `P`, one per
   patient permutation (each iter's 96 are shuffled by `seed + n`). Saved to
   `data/grpo_Exp3/conversations/.../model_iter_{n-1}/`.
2. **Extract per-turn prompts.** Slice each conversation after every patient
   turn whose total-utterance count is `≥ MCL`. Each slice becomes a training
   sample with two fields: `prompt` (chat-template-formatted prefix, fed to
   `GRPOTrainer`) and `transcript` (plain-text version, fed to the oracle).
   Conversation-level train/eval split prevents leakage.
3. **GRPO update.** For each prompt in the train split, `GRPOTrainer`:

   a. Samples `G = NUM_GENERATIONS` completions from `π_n` at
      `GRPO_TEMPERATURE`.

   b. For each completion `t_g`, computes a reward `r_g`:
      - If `K = 0`: `r_g = O(transcript + t_g)`.
      - If `K > 0`: build the K-step extended transcript via the look-ahead
        subroutine above (using `π_n` for all therapist turns in the rollout),
        then `r_g = O(extended_transcript)`.

   c. Group-relative advantage: with `scale_rewards="group"`,
      `A_g = (r_g - mean_g(r)) / std_g(r)` over the `G` siblings for this prompt.

   d. PPO-style clipped policy gradient on the group: maximize
      `E[A_g · log π(t_g | prompt)]` minus a KL penalty `β · KL(π_n ‖ π_ref)`
      against the iteration's reference (the iter-start adapter).
4. **Train + save.** `EPOCHS_PER_ITERATION` epochs over the prompts; per-epoch
   checkpoints in `iteration_{n}/training/`, final adapter in
   `iteration_{n}/adapter/`. The same convs serve as the eval set for the
   *previous* iteration's policy (`model_iter_{n-1}`).
5. **Repeat with `π_{n+1}`.**

**After the loop**, one generate-only pass with the final adapter produces
`model_iter_{NUM_ITERATIONS}/` so the last policy has matched eval data.

**Why look-ahead helps GRPO:** the shared-subroutine motivation above, applied to
the `G` siblings — with `K > 0`, siblings that "look promising but unwind badly"
rank below siblings that "compound."

### PTO_Exp3 + K-turn look-ahead

**`PREF_TREE_MODE` selects how pref pairs are built** (default `greedy` = true PTO;
`independent` = the earlier slice-branch behavior, kept as an alternate arm). Both
share the M-branch → look-ahead → oracle-score → τ-filter → DPO machinery; the mode is
baked into `EXPERIMENT_NAME` (`_PT{greedy|indep}`) so arms never collide. The grower
runs **lock-step across all trunks** (mirrors the batched look-ahead).

**Per iteration `n` (loop body in [PTO_Exp3/train_PTO_Iterative.ipynb](code/PTO_Exp3/train_PTO_Iterative.ipynb), helpers in [pto_trainer.py](code/PTO_Exp3/pto_trainer.py)):**

1. **Eval pass.** `π_n` simulates 96 full conversations versus `P`, saved to
   `data/pto_Exp3/conversations/.../model_iter_{n-1}/` (doubles as eval, like GRPO).
2. **Build preference pairs.**
   - **`greedy` (`grow_preference_trees_batch`):** SLICE the first `MCL` utterances off
     each step-1 conv (ending on a patient turn) as the trunk seeds — no separate prefix
     pass; the seeds reuse the eval-conv openings then diverge. Then grow each trunk: at
     each therapist turn sample `M` completions from
     `π_n` at `BRANCH_SAMPLE_TEMPERATURE` → K-turn look-ahead → oracle-score → **append
     the best completion to the trunk** (so it feeds the next branch point) → `P`
     replies → repeat until the trunk reaches `NUM_UTTERANCES_FOR_DATA` utterances.
     Emit a pair `(trunk-so-far prompt, chosen, rejected)` at each branch point where
     `r_chosen − r_rejected > PREF_FILTER_TAU`; **always** append the best to advance
     the trunk (a tie just emits no pair). Freeze a trunk on SESSION ENDED / API
     failure / no valid branch score.
   - **`independent` (`build_pref_pairs_for_conversation`):** branch at every patient
     turn of the step-1 conversation whose prefix-so-far is `≥ MCL` and isn't the final
     turn — `M` completions, look-ahead, best/worst with the same τ filter — but against
     the **pre-recorded** trunk (the winner is never fed back).
3. **DPO update.** Train `DPOTrainer` on the collected pref pairs for
   `EPOCHS_PER_ITERATION` epochs. The DPO loss is
   ```
   L = -E_{(prompt, chosen, rejected)}[ log σ( β · (
       log π(chosen|prompt)   - log π_ref(chosen|prompt)
     - log π(rejected|prompt) + log π_ref(rejected|prompt)
   ))]
   ```
   where `π_ref` is the iter-start adapter, `β = DPO_BETA`. This pushes `π`
   toward `chosen` and away from `rejected` while staying close to `π_ref`.
4. **Save.** Adapter + a per-iter `pref_pairs/pairs.csv` audit trail
   (prompt + chosen + rejected + both scores per pair) for debugging "why is
   this iteration's DPO update weird?" without re-running the expensive
   branching + scoring.
5. **Repeat with `π_{n+1}`.**

Same final-eval pass + same Hub-push pattern as GRPO_Exp3.

**Why look-ahead helps PTO:** same motivation (shared subroutine above) — the
branch ranking inherits the oracle's weak signal-to-noise on short snapshots (see
the partial-conversation diagnostic below); scoring the K-step trajectory the
current policy actually takes reduces that disagreement.

### Where the two methods differ (concise)

| | GRPO_Exp3 | PTO_Exp3 |
|---|---|---|
| Per-prompt samples | `G` completions, **all kept** | `M` completions, **best+worst kept**, τ-filtered |
| Training data shape | `{prompt, transcript}` (reward computed inside trainer) | `{prompt, chosen, rejected}` (reward used only to *pick* the pair) |
| Loss | Group-relative PPO clip + KL | DPO sigmoid + implicit KL via `π_ref` |
| TRL class | `GRPOTrainer` | `DPOTrainer` |
| Output per prompt | 1 gradient step per prompt | 0 or 1 pref pair (then standard DPO loss) |
| Yields zero training rows? | No — every prompt trains | Yes, if every branch ties within τ |
| `_shared` usage | gen + reward as a reward-fn callable | gen + reward as a scorer the trainer doesn't see |

### Where the K knob plugs in (one paragraph)

Look-ahead is purely about **what context the oracle scores**, not about the
loss. In both methods, K controls the length of the post-completion rollout
appended to each candidate before the oracle is queried; everything downstream
(reward in GRPO's case, pair selection in PTO's case) is unchanged. This is why
the K∈{0, 5} comparison is meaningful on *both* methods — it isolates the
look-ahead lever from the loss family.

### Conversations double as eval data
The conversations generated at the start of iteration `n` are the output of
`model_iter_{n-1}` — so they ARE the eval set for that model state. No
separate generate-eval step for trained iters.

### Iter ↔ model-state mapping
At start of iter `n`, loaded policy = iter-(`n`−1) adapter (or base if `n=1`).

| Loop iter `n` | Generates with | Saves convs as | Produces adapter |
|---|---|---|---|
| 1 | base | `model_iter_0/` | `iteration_1/adapter/` |
| 2 | iter-1 adapter | `model_iter_1/` | `iteration_2/adapter/` |
| `N` | iter-(`N`−1) | `model_iter_{N-1}/` | `iteration_N/adapter/` |
| post-loop | iter-`N` | `model_iter_{N}/` | — |

### Vocabulary
GRPO has no preference data — only prompts. **Never** call GRPO data "pref data".
PTO is the framework; DPO is the loss it uses.

## Layout

```
Exp3_PTO_GRPO/
├── CLAUDE.md
├── code/
│   ├── system_prompts_builder.py        V3 prompts (single canonical copy; EDA also reads this one)
│   ├── questionnaires.py                V5 oracle (JSON schema, 6 questionnaires)
│   ├── _local_smoke.py                  offline smoke tests (stopgen|dpo|grpo) — no OpenAI; imports trl before torch (see Gotchas)
│   ├── _shared/                         cross-method modules (GRPO_Exp3 + PTO_Exp3 both import)
│   │   ├── __init__.py                  public-API re-exports
│   │   ├── runtime.py                   Colab/local detect, auth, paths, preflight
│   │   ├── model.py                     tokenizer/quant/LoRA + checkpoint discovery + iteration resume
│   │   ├── convs.py                     conv state + async gen + per-turn prompt extraction (MCL filter)
│   │   ├── reward.py                    oracle scoring + K-turn look-ahead (batched) + reward-fn factory
│   │   ├── tb_plots.py                  TB callbacks + logging lifecycle + TB parser + plot dashboard
│   │   ├── eda_recorder.py              per-generation EDA capture → iteration_N/eda/generations.jsonl (all candidates + scores + look-ahead tails)
│   │   └── lookahead_check.py           OPTIONAL (off hot path): serial-vs-batched look-ahead equivalence + OOM smoke
│   ├── GRPO_Exp3/
│   │   ├── train_GRPO_Iterative.ipynb   visible orchestration loop
│   │   └── grpo_trainer.py              TrainingConfig + run_one_iteration + run_final_eval + …
│   └── PTO_Exp3/
│       ├── train_PTO_Iterative.ipynb    visible orchestration loop (mirrors GRPO_Exp3)
│       └── pto_trainer.py               PTOConfig + run_one_iteration + build_pref_pairs_for_conversation + …
├── data/                               eval scores co-locate per method, labelled metric=<M>/oracle=<O>/ (M=scoring metric, O=training oracle)
│   ├── eval_coverage.csv                scoring-coverage snapshot: per model × metric done/todo counts
│   ├── grpo_Exp3/                       produced by GRPO_Exp3 runs
│   │   ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training}/
│   │   ├── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
│   │   └── eval_scores/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
│   └── pto_Exp3/                        produced by PTO_Exp3 runs (same shape as grpo_Exp3)
│       ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training, pref_pairs/}
│       ├── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
│       └── eval_scores/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
├── eda/                                 verified runnable end-to-end
│   ├── Run_Eval.ipynb                   async oracle pipeline → eval_scores/ (resume-safe; backend: eda_analysis/scoring/, registry-driven)
│   ├── 1_Outcomes.ipynb …             the 6 topic notebooks (`1_Outcomes` `2_Heterogeneity` `3_Mechanism`
│   │     … 6_Stats.ipynb              `4_Training_and_Reliability` `5_Preference` `6_Stats`) ↔ result
│   │                                  families 1:1, [EVAL]/[TRAINING]-tagged — contents table: eda/README.md
│   ├── render_views.py                         DRIVER: regenerate results/<view>/ for all 6 notebooks via nbconvert (sets EDA_VIEW; --output-dir tmp; --nb takes the notebook/family NUMBER 1..6)
│   ├── strip_notebook_outputs.py        output-clean helper (paired with the nbstrip git clean-filter)
│   ├── README.md                        EDA guide: notebook↔family table, VIEW knob, module map, roadmap
│   ├── LIMITATIONS.md                   documented measurement/inference limitations (for the thesis write-up)
│   ├── METRICS_REFERENCE.md             cheat-sheet for every EDA number (questionnaires, derived ratios, hack battery)
│   ├── eda_analysis/                    THE Exp3 EDA package (one package since the 2026-07-13 fold):
│   │                                    analysis layer (disk-discovery, read-only) = constants LEAF
│   │                                    + config / data / plotting_style / stats / behavior / training /
│   │                                    pref / exports / _selfcheck + plotting/ subpackage (topic-split
│   │                                    figures; figures+plots alias it); scoring layer = scoring/
│   │                                    subpackage (registry / conversations / pipeline / judge — the
│   │                                    Run_Eval + Judge_Reliability backend, imported explicitly, not
│   │                                    via __init__). Module map: eda/README.md § "Package".
│   ├── results/                         GENERATED thesis artifacts in 3 VIEW trees: all/ · L0/ · L5/, each with figures|tables/<N_family>/ (family number == producing-notebook number) + INDEX.md + hand-authored SUMMARY.md
│   ├── .eda_cache/                      parquet cache (gitignored; content-keyed on input CSVs)
│   └── .emb_cache/                      pref completion-embedding cache (gitignored; regenerable)
└── HF_key.txt, openai_key.txt
```

**Thesis artifacts.** `results/<view>/figures/` (`.png`) and `results/<view>/tables/` (`.md`+`.xlsx`)
are **generated** by `eda_analysis.save_fig`/`save_table` (the `formats=` kwarg can request extras for
a one-off; per-call `group=` overrides the family, incl. nested subpaths). Each notebook regenerates
its own family; `python render_views.py` regenerates everything. Reproducible from code; tracked in git.

**Change history** (the dated "pass"/"Landed" entries — both the EDA passes and the trainer /
infrastructure narratives) — moved to [history/CHANGELOG.md](history/CHANGELOG.md). The current state
is the "EDA workflow" + "Training internals" + "Run status" sections.

**Single canonical copies.** `system_prompts_builder.py` and `questionnaires.py`
live ONLY at `code/` root — `eda/eda_analysis/constants.py` (the package's leaf, imported by
everything incl. `scoring/`) prepends `code/` to `sys.path` so they import the same canonical
files. No more drift.

### EDA workflow (short version — full guide in [eda/README.md](eda/README.md))
1. **Score:** `Run_Eval.ipynb` — its `EXPERIMENTS` registry is auto-generated from
   `eda_analysis.data.discover_arms()`, so a run is scoreable as soon as its conversations land on
   disk (empty in-flight `model_iter` dirs are skipped). Writes `eval_scores/`.
2. **Analyze:** notebooks `1_Outcomes` … `6_Stats` (topic ↔ results family, 1:1); everything
   auto-discovers arms from disk — no registry edits anywhere. The **VIEW knob** (`all`/`L0`/`L5`)
   sets both the arm filter and the `results/<view>/` output root.
3. **Regenerate:** `python render_views.py` (L0+L5 default, `all` opt-in) → `results/<view>/`.
   Run **`python -m eda_analysis._selfcheck`** after any EDA change.

The VIEW system, `EdaConfig`, parquet cache, output-clean policy, and the package module map are all
documented in [eda/README.md](eda/README.md) — not here.

### Eval results (pointer — numbers are NOT maintained in this file)
Qualitative headline: **PTO wins at the matched 10-iter endpoint; GRPO peaks at iter 8 then
regresses into sycophancy (affirmation-drift reward-hack); both LA5 arms are thin/paused.** Owners:
- **Status + headline numbers + cost constraint:** root [CLAUDE.md](../CLAUDE.md) § "Current status
  & next step" (the single live copy).
- **Full narrative + numbers:** [eda/results/L0/SUMMARY.md](eda/results/L0/SUMMARY.md) (primary
  read) · [all](eda/results/all/SUMMARY.md) · [L5](eda/results/L5/SUMMARY.md); tables under
  `eda/results/<view>/tables/` (`6_stats/main_results.md`, `1_outcomes/leaderboard_scorecard.md`).
- **The dated 2026-07-08 findings write-up:** [history/CHANGELOG.md](history/CHANGELOG.md) + the
  `project-pto-la0-eval-results` memory.

## Diagnostic: partial-conversation oracle (reward-faithfulness)

Both trainers score *partial* conversations (slices as short as 2 turns) as the training reward, but
the thesis evaluates *full* conversations. The diagnostic — rebuilt on Exp3 data with no new oracle
calls in [4_Training_and_Reliability.ipynb](eda/4_Training_and_Reliability.ipynb) (from the
per-branch `prefix` in `generations.jsonl`); the original Exp2 version motivated the MCL knob —
shows pairwise rank agreement with the final-conv score is **barely above chance at `n_turns=2` and
only clears 0.8/0.9 at ~10/~30 turns**, a structural gap well above oracle reproducibility noise.
Numbers + method: [eda/METRICS_REFERENCE.md](eda/METRICS_REFERENCE.md) § 6.

**Implication.** Short training cuts can't observe whether the therapist delivered on Q1/Q2 by
session end, so the oracle scores them on "did the opening look promising?" — optimizing that proxy
biases the model toward strong-looking openings regardless of follow-through.

## MIN_CONV_LENGTH filter — wired in both trainers

Direct response to the partial-conversation reliability finding above.

- **GRPO_Exp3.** Cell 1's `MIN_CONV_LENGTH` → `TrainingConfig.min_conv_length` →
  `extract_prompts_from_conversations(min_conv_length=...)` in [_shared/convs.py](code/_shared/convs.py).
- **PTO_Exp3.** Cell 1's `MIN_CONV_LENGTH` → `PTOConfig.min_conv_length`. In `greedy`
  mode it's where the **tree starts** (prefix length, must be EVEN so the prefix ends on
  a patient turn); in `independent` mode it's the slice filter (`build_pref_pairs_for_conversation`
  skips branch points whose conv-so-far is shorter). Either way: no training context below MCL.
- **Semantics.** Drop slices/branches where the conversation-so-far has fewer than `MIN_CONV_LENGTH` total utterances (same `n_turns` unit as the partial-conv diagnostic — therapist + patient combined).
- **Default = 2** = no-op. Recommended exploratory values: `10` (EDA's 0.8 threshold), `30` (0.9 threshold).
- **Encoded in `EXPERIMENT_NAME`** as `_MCL{N}` so runs at different MCL never share an output folder.

## EXPERIMENT_NAME schemes

- GRPO_Exp3: `GRPO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_G{G}`
- PTO_Exp3:  `PTO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_M{NUM_BRANCHES_PER_TURN}_PT{greedy|indep}`

`{Oracle}` is the training-oracle token derived from `QUESTIONNAIRE_IDS` in cell 1
(`Q1Q2`|`WAI`|`CSQ8`|`MI_SAT`|`MITI`) — identical to the EDA `oracle=<O>` tokens, so a run's
folder/Hub name and its `eval_scores/.../oracle=<O>/` folder agree. An unmapped ID set raises.

Different sweep arms write to disjoint dirs — runs never collide.

## Running GRPO_Exp3

1. **Configure.** [code/GRPO_Exp3/train_GRPO_Iterative.ipynb](code/GRPO_Exp3/train_GRPO_Iterative.ipynb) cell 1 = flat globals.
2. **Train.** Run top-to-bottom. The orchestration loop is in the notebook (cells after `cfg = TrainingConfig(...)`), composed from `run_one_iteration` / `run_final_eval` in [grpo_trainer.py](code/GRPO_Exp3/grpo_trainer.py). Resumes from latest completed iter via [_shared.resolve_start_state](code/_shared/model.py). Outputs under `data/grpo_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`; per-run `run_metadata.json` at the run root.
3. **Inspect.** Last cell: `scan_scalar_tags` + `plot_iteration_metrics` + inline TensorBoard. `plot_iteration_metrics` applies per-iteration step offsets so cross-iter curves chain end-to-end (dotted vlines mark iter boundaries).
4. **Score + EDA.** Run [eda/Run_Eval.ipynb](eda/Run_Eval.ipynb) (resume-safe; its `EXPERIMENTS` registry auto-discovers the run from disk — no registry edit) → then open [eda/1_Outcomes.ipynb](eda/1_Outcomes.ipynb) (and `2`–`6`), which likewise **auto-discover** it. See "EDA workflow".

## Running PTO_Exp3

1. **Configure.** [code/PTO_Exp3/train_PTO_Iterative.ipynb](code/PTO_Exp3/train_PTO_Iterative.ipynb) cell 1 = flat globals. Key extra knobs vs GRPO: `PREF_TREE_MODE` (`greedy`|`independent`), `NUM_BRANCHES_PER_TURN`, `PREF_FILTER_TAU`, `BRANCH_SAMPLE_TEMPERATURE`, `DPO_BETA`, `DPO_LOSS_TYPE`. `greedy` mode requires an EVEN `MIN_CONV_LENGTH` (so the sliced prefix ends on a patient turn) and slices its trunk seeds from the step-1 convs (no separate prefix-generation pass).
2. **Train.** Same visible-orchestration pattern. Outputs land under `data/pto_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`. Each iteration also saves the constructed pref pairs to `iteration_N/pref_pairs/pairs.csv` (audit trail; the prompt + chosen + rejected + scores per pair).
3. **Inspect + Score + EDA.** Same as GRPO_Exp3 (the TB dashboard is shared via `_shared/tb_plots.py`).

## Training internals (current behavior)

The dated "how we got here" narratives — resume, checkpointing, batched look-ahead, per-generation
EDA capture, throughput tuning, and the first-run + ChatML-leak fixes — live in
[history/CHANGELOG.md](history/CHANGELOG.md) (Trainer / infrastructure history). Current behavior:

- **Resume.** `resolve_start_state` ([_shared/model.py](code/_shared/model.py)) treats an iteration as
  done once `iteration_N/adapter/` exists. A crashed iteration resumes from the latest **valid** sub-epoch
  checkpoint (`SAVE_STEPS=10`, `SAVE_TOTAL_LIMIT=2`; `get_latest_valid_hf_checkpoint` walks back over a
  corrupt newest). **PTO Step-2** (the ~41-min pref-build) resumes too: `iteration_N/pref_pairs/pairs.csv`
  is both the DPO audit trail AND the completion marker (reload + skip), and `pref_pairs/_progress.json`
  is a per-step snapshot for mid-build resume (guarded by a config fingerprint incl. τ, which is NOT in
  `EXPERIMENT_NAME`, so a different-τ checkpoint is discarded not mixed).
- **K-turn look-ahead is batched.** `simulate_lookahead_batch` ([_shared/reward.py](code/_shared/reward.py))
  advances all B completions in lock-step — one padded batched `model.generate` per look-ahead turn —
  ~statistically equal to the legacy serial path (validated on GPU, |Δmean|=0.024, 1.5×). Knob
  `LOOKAHEAD_SUB_BATCH_SIZE` (64 GRPO / 128 PTO on A100-80GB; auto-halves on OOM, kept sticky).
- **Per-generation EDA capture.** Each iter writes `iteration_N/eda/generations.jsonl` — one branch row
  with nested `candidates[]` (`completion`/`score`/per-questionnaire `sub_scores`/`lookahead.tail`) +
  `chosen_idx`; GRPO one row per group per epoch, PTO one row per branch. Knobs `SAVE_EDA_GENERATIONS`,
  `SAVE_LOOKAHEAD_TRANSCRIPTS`. The EDA reads these ([eda_analysis/training.py](eda/eda_analysis/training.py)).
- **Anti-degeneracy (the base 1B self-plays ChatML markers).** `STOP_STRINGS=["<|im_end|>","<|im_start|>"]`
  + `clean_completion` ([_shared/convs.py](code/_shared/convs.py)) cut generation at the first fake-turn
  marker at every decode site; empty-after-clean ends the conversation; GRPO floors degenerate completions
  to `REWARD_FLOOR=0.0`. DPO caps the prompt to the context window (`build_truncated_training_prompt`,
  drop-oldest) so the full-seq LM-head logits over the 128k vocab don't OOM (keep DPO `per_device=2`).
- **Throughput config (tuned for A100 Colab).** `EPOCHS_PER_ITERATION=2`, `CONVERSATION_BATCH_SIZE=64`,
  `ORACLE_MAX_CONCURRENCY=128`, `PATIENT_API_CONCURRENCY=96`; DPO kept at `per_device=2 × grad_accum=8`
  + grad-checkpointing (the config that fits — `per_device` sizes the full-seq logits tensor, so don't
  raise it). Optional PTO speed lever `GREEDY_TRUNK_TARGET_LEN` (shallower trunks; a science change, NOT
  in `EXPERIMENT_NAME`). Wall-clock is GPU-bound (autoregressive `model.generate`), not API-bound.
- **Logging = HF defaults.** One W&B run per iteration (grouped via `wandb_ctx["run_id"]`), TRL's native
  metrics + completions table. The continuous cross-iteration `tb_live/` view is opt-in
  (`TB_LIVE_LOGGING=False` default); the post-hoc matplotlib dashboard `plot_iteration_metrics` reads the
  per-iter `tb_logs/` regardless.

## Run status (pointer) + durable LA5-resume facts

**Run status, headline numbers, and the OpenAI cost constraint live in ONE place:** root
[CLAUDE.md](../CLAUDE.md) § "Current status & next step" (+ the `project-openai-cost-constraint`
memory). Don't restate them here.

**Durable LA5-resume facts** (what's on Drive; the dated forensics are in
[history/CHANGELOG.md](history/CHANGELOG.md), 2026-07-11 entry):
- **PTO LA5:** trained adapters for iters 1–5, but only I1–I4 scored — the iter-5 eval convs were
  **never generated** (`model_iter_5` conv dir exists but is EMPTY; `iteration_6/` stopped at
  `pref_pairs`, no adapter). Cheapest restart: one generate-only pass with the iter-5 adapter
  (96 convs; GPU + patient calls, **no training**) + `Run_Eval` scoring = a 5th PTO_LA5 point before
  any new training spend.
- **GRPO LA5:** iter-1 adapter trained AND scored; its `iteration_2/` dir is adapter-less
  (incomplete). Folder presence ≠ data.

## Dependency stack

Pins live in [../requirements.txt](../requirements.txt); both notebooks' Colab install cells are
pinned to it. The full 2026-06-01/03 audit (TRL 1.x / transformers 5 API currency, `hf_xet`,
gpt-4o-mini retirement check, batch/LR notes) moved to [history/CHANGELOG.md](history/CHANGELOG.md).
The one live install gotcha: **uninstall torchao on Colab** — peft 0.19.1 *raises* inside
`get_peft_model`'s `dispatch_torchao` on Colab's pre-baked torchao<0.16.0 (both install cells carry
the commented `%pip uninstall -y torchao`).

## Colab vs local

Realistic workflow: **training on Colab (GPU)**, **EDA + Run_Eval locally**.
EDA has no Colab branches — host-agnostic by design. Dual-host plumbing in
the trainers is only there to keep them importable + smoke-testable locally.

Experiment root resolution:
- **Local.** Walk up from `os.getcwd()` for `HF_key.txt`+`openai_key.txt` → typically `Exp3_PTO_GRPO/`.
- **Colab.** Trainer notebooks cd into `code/<METHOD>_Exp3/` after mounting Drive, then prepend `code/` to `sys.path` so `_shared` resolves as a sibling package.

### Auth (trainer only — `init_openai_client` / `authenticate` in [_shared/runtime.py](code/_shared/runtime.py))

| Secret | Colab | Local |
|---|---|---|
| OpenAI | `userdata["OPENAI_API_KEY"]` → env → file | env (`OPENAI_API_KEY`) → file |
| HF token | `userdata["huggingface"]` → env → file | env (`HF_TOKEN`/`HUGGINGFACE_TOKEN`) → file |
| W&B | `userdata["wandb"]` | env `WANDB_API_KEY` |

HF token IS used locally — Llama-3.2-1B is gated.

### Sync (Colab ↔ local)

**Results pull — Google Drive Desktop, no rclone.** `data/grpo_Exp3` and
`data/pto_Exp3` are **directory symlinks** into Drive
(`G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data\<method>`). Colab writes to mounted
Drive → Drive Desktop (kept in **streaming** mode, low disk) surfaces it locally →
files appear straight inside the repo; EDA reads through the link unchanged (all reads
go via `WORKSPACE_ROOT/data/...`). EDA only reads `conversations/` + `eval_scores/`
CSVs, so streaming downloads just those on open; big artifacts (`runs/`, adapters,
`*.safetensors`) are never read locally and also live on HF Hub + W&B.

Re-create the links (Windows **Developer Mode** on; use `mklink`, **not** PowerShell
`New-Item -ItemType SymbolicLink` — WinPS 5.1 ignores Dev Mode and still demands admin):
```powershell
$D = "G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data"
$R = "C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data"
cmd /c "mklink /D ""$R\grpo_Exp3"" ""$D\grpo_Exp3"""
cmd /c "mklink /D ""$R\pto_Exp3""  ""$D\pto_Exp3"""
```
To undo: delete the **link** (`Remove-Item "$R\grpo_Exp3"`) — Drive data untouched.

**Code push (local → Drive for Colab) is manual, `code/` only.** The whole `code/`
tree was pushed to `G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\code\` (2026-06-01, robocopy) —
that's all Colab needs; open a `train_*_Iterative.ipynb` from there in Colab. Do **not**
push `data/` (the symlink targets already live in Drive) or `eda/` (local-only). Keys come from **Colab Secrets** (`OPENAI_API_KEY`,
`huggingface`, `wandb`), not the `.txt` files. After editing code locally, push the update by **dragging the `code` folder** onto the Drive
`Exp3_PTO_GRPO\` parent — a merge that adds/overwrites but **never deletes** (Lior's default).
For an exact mirror that also **removes** files you renamed/deleted, robocopy `/MIR` — but it
is destructive on the destination, so run it **only with Lior's explicit go-ahead**:
```powershell
robocopy "C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp3_PTO_GRPO\code" `
         "G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\code" /MIR /XD __pycache__
```
Let Drive Desktop finish syncing (tray ✓) before running the Colab cell.
`rclone sync A B` mirrors (deletes extras in B); use `copy` for additive, `check` for a dry-run diff.

## EDA extension points

**Analysis layer (`eda_analysis/` top level)** needs **no registry edits** — it auto-discovers arms from
disk. Extend it by concern: a new rubric → `eda_analysis/constants.py::QUESTIONNAIRES` + `data.py` (the
scores backbone); a new arm naming scheme → `data.py::parse_experiment_name`; new stats → `stats.py`; new
figures → the topic module in `plotting/` (+ its `__init__` re-export); a new VIEW or results-layout change
→ `config.py` (the `view`/`_VIEW_KS` logic) + `exports.py`. (`figures`/`plots` are still aliased to
`plotting`; the data-module aliases `discovery`/`personas`/`scores`/`select` were retired — use
`eda_analysis.data.*` / the top-level re-exports.)

**Scoring layer (`eda_analysis/scoring/` — the Run_Eval + Judge_Reliability backend):**

- **`scoring/registry.py::ORACLE_TOKEN_ALIASES`** — add new oracle-name aliases here (CSQ vs CSQ_8 etc.). `conversations._normalize_oracle_token(strict=True)` raises on unknowns; default `strict=False` lets unknowns fall through to "Other" for backward compat.
- **`scoring/registry.py::COMPOSITE_METRICS`** — add new composites (mean across multiple source columns) here. Currently holds just `Q1Q2_Mean`; the same pattern can produce `MITI_GlobalMean` etc.
- **`scoring/registry.py::EXPERIMENTS`** — registry of trained-model data locations, **auto-generated at import** by `build_experiments_from_disk()` from `eda_analysis.data.discover_arms()` (2026-07-11). New runs are picked up automatically once their conversations land; nothing to edit. (If the Drive symlinks are offline the registry is empty and a warning prints.)
- **`scoring/judge.py`** — add second-judge providers/models here (`JudgeSpec`); outputs stay under `data/judge_check/`, never the real `eval_scores/`.

## Gotchas

- **HF model-card READMEs** inside `data/grpo_Exp3/runs/.../checkpoint-*/` are auto-generated — DO NOT delete or treat as project docs.
- **Pref-tree audit trail = resume marker.** PTO_Exp3 writes `iteration_N/pref_pairs/pairs.csv` per iter. Don't delete — it's both the DPO debug trail AND the Step-2 completion marker: its presence makes a restart **reload it and skip the ~41-min build** (see "Training internals" → Resume). The sibling `iteration_N/pref_pairs/_progress.json` is the in-build per-step checkpoint (auto-deleted on success; safe to delete manually to force a clean rebuild).
- **Per-generation EDA.** `iteration_N/eda/generations.jsonl` (one row per branch, candidates nested — see "Training internals") is separate from `pref_pairs/pairs.csv` (the PTO DPO audit trail). Off-switch: `SAVE_EDA_GENERATIONS=False`. The continuous live-TB run lives at `runs/.../tb_live/` (sibling of `iteration_N/`).
- **Local sm_120 import order: `trl` must be imported BEFORE `torch`.** On the local Blackwell GPU, `from trl import …` *after* torch is already imported **segfaults at CUDA init** (a native init-order conflict, exit 139 — not OOM, not a bug in the trainers; Colab is unaffected, which is why the full runs ran there). The trainer modules already import `trl` first; only matters if you run something locally that imports torch/`_shared` first. Verified 2026-06-07.
- **Local offline smoke:** [code/_local_smoke.py](code/_local_smoke.py) — `python _local_smoke.py {stopgen|dpo|grpo|all}`. Tiny, no OpenAI; validates the stop-string bind, the DPO prompt-cap + no-OOM (grad-ckpt+precompute), and a GRPO step on the local GPU (~3 GB peak). Imports `trl` first (see above). All three PASS as of 2026-06-07.
- **Oracle prompt caching depends on the rubric-first layout.** [questionnaires.py](code/questionnaires.py) `get_prompt_eval_questionnaire` puts the fixed instructions + questionnaire rubric FIRST and the variable transcript LAST, so OpenAI's automatic prompt caching hits the ~1,084-token fixed prefix on every oracle call (≈50 % input discount + lower latency — matters for the oracle bill, the binding cost constraint per root CLAUDE.md, even though wall-clock is GPU-bound; see next bullet). The margin over OpenAI's 1,024-token minimum is thin: **don't trim the oracle instructions/rubric or move the transcript ahead of them**, or caching silently stops (verified 2026-06-07: prefix is transcript-independent for Q1). Patient API calls auto-cache too (stable system + growing-history prefix). The therapist's local `model.generate` has **no** cross-call prefix reuse under HF — that would need vLLM (a real build here, not a flag: the look-ahead and *all* of PTO's generation use custom `model.generate`, not TRL's `use_vllm` path).
- **The run is likely GPU-bound, not API-bound (corrected 2026-06-07).** Earlier notes called the runs "API-bound" — that was inferred from GPU *memory* (17/67 GB), which does NOT measure compute. Lior observes he waits on GPU, not API. Autoregressive `model.generate` on the 1B LoRA policy (GRPO's G=8 completion sampling + K-turn look-ahead; PTO's branch sampling + look-ahead) dominates wall-clock; the `340.6 s / 8 GPU calls` look-ahead line ≈ 30–40 s per batched generate, far above the ~1–2 s of raw 1B/A100 compute → heavy per-step overhead. **Top suspect: the recently-added `STOP_STRINGS` route generation through HF `StopStringCriteria` (runs every step; known multi-× slowdown).** Before optimizing, MEASURE the split (time sampling vs look-ahead-GPU vs look-ahead-API vs backward); the K=0 arms (no look-ahead) running much faster would itself confirm generation is the cost. Faster stop than string-matching: register the two markers as single special tokens + stop on `eos_token_id`.
