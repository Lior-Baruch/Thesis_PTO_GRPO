# Exp3_PTO_GRPO — ACTIVE (main thesis chapter)

Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle. Two methods compared
under matched look-ahead + oracle:
- **PTO_Exp3** (preference-tree → DPO loss). Lean sibling of GRPO_Exp3 (one notebook
  + one `pto_trainer.py`, sharing `_shared/`). **Controlled hyperparameters matched to
  GRPO_Exp3** (2026-06-03): NUM_ITERATIONS=10, MCL=12, K∈{0,5}, gen temps + API
  concurrency; M (`NUM_BRANCHES_PER_TURN`)=8 mirrors GRPO's `NUM_GENERATIONS`;
  `DPO_BETA`=0.1 kept (DPO loss temp, not GRPO's KL β). bf16 `USE_4BIT` toggle + a
  zero-pairs actionable error + train/eval split fix also landed. Output dir:
  `data/pto_Exp3/`. The Exp2-sourced `data/pto_Exp2/` artifacts are still read by the
  EDA registry but **not regenerated here** unless you re-run PTO_Exp3.
  **Two data-gen modes via `PREF_TREE_MODE` (2026-06-03):** `greedy` (default, true PTO
  — grow ONE trunk from an MCL prefix by appending best-of-M) and `independent` (the
  earlier behavior — branch each patient turn of a pre-recorded conv, no feedback).
  Baked into `EXPERIMENT_NAME` (`_PT{greedy|indep}`) so the arms never collide. See
  the algorithm section below.
- **GRPO_Exp3** (iterative). K=3 bf16 quicktest running on Colab; full K∈{0,5} sweep not yet run — definite next step.

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

**Why look-ahead helps GRPO.** Without `K`, every sibling completion is scored
on its immediate effect on a snapshot prefix — short prefixes have weak signal
(see partial-conv EDA). With `K > 0`, each sibling is scored on the K-step
trajectory the *current policy* would actually take after it, so siblings that
"look promising but unwind badly" rank below siblings that "compound."

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

**Why look-ahead helps PTO.** Same intuition as GRPO: the branch ranking
inherits whatever signal-to-noise the oracle has on the snapshot being scored.
The partial-conv EDA shows that at short prefixes (`n_turns ≤ ~10`) the
snapshot score disagrees with the eventual full-conv score on 25-30% of
pairwise comparisons. K-turn look-ahead reduces that disagreement by scoring
the K-step trajectory the current policy actually takes.

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
│   ├── _shared/                         cross-method modules (GRPO_Exp3 + PTO_Exp3 both import)
│   │   ├── __init__.py                  public-API re-exports
│   │   ├── runtime.py                   Colab/local detect, auth, paths, preflight
│   │   ├── model.py                     tokenizer/quant/LoRA + checkpoint discovery + iteration resume
│   │   ├── convs.py                     conv state + async gen + per-turn prompt extraction (MCL filter)
│   │   ├── reward.py                    oracle scoring + K-turn look-ahead (batched) + reward-fn factory
│   │   ├── tb_plots.py                  TB callbacks + logging lifecycle + TB parser + plot dashboard
│   │   └── lookahead_check.py           OPTIONAL (off hot path): serial-vs-batched look-ahead equivalence + OOM smoke
│   ├── GRPO_Exp3/
│   │   ├── train_GRPO_Iterative.ipynb   visible orchestration loop
│   │   └── grpo_trainer.py              TrainingConfig + run_one_iteration + run_final_eval + …
│   └── PTO_Exp3/
│       ├── train_PTO_Iterative.ipynb    visible orchestration loop (mirrors GRPO_Exp3)
│       └── pto_trainer.py               PTOConfig + run_one_iteration + build_pref_pairs_for_conversation + …
├── data/                               eval scores co-locate per method, labelled metric=<M>/oracle=<O>/ (M=scoring metric, O=training oracle)
│   ├── pto_Exp2/                        Exp2-sourced PTO artifacts + their scores (NOT regenerated here)
│   │   ├── pref_trees/{CSQ-8,CTRL,Q1Q2,WAI}/
│   │   ├── eval_conversations/{Base,CSQ-8,CTRL,Q1Q2,WAI}/
│   │   └── eval_scores/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
│   │                                        ↳ partial_q1q2/<Model>/{id}_t{n_turns}.csv  (Partial_Conv_Oracle_EDA cache)
│   ├── grpo_Exp3/                       produced by GRPO_Exp3 runs
│   │   ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training}/
│   │   ├── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
│   │   └── eval_scores/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
│   └── pto_Exp3/                        produced by PTO_Exp3 runs (same shape as grpo_Exp3)
│       ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training, pref_pairs/}
│       ├── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
│       └── eval_scores/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
├── eda/                                 verified runnable end-to-end
│   ├── Run_Eval.ipynb                   async oracle pipeline → eval_scores/ (resume-safe)
│   ├── Conv_EDA.ipynb                   main analysis (~38 cells)
│   ├── Partial_Conv_Oracle_EDA.ipynb    proxy-reliability diagnostic — see below
│   ├── lib/                             5-file EDA package
│   │   ├── __init__.py                  resolves WORKSPACE_ROOT, prepends code/ to sys.path, re-exports
│   │   ├── config.py                    constants, palettes, EDAConfig, Experiment registry, ORACLE_TOKEN_ALIASES, COMPOSITE_METRICS
│   │   ├── data.py                      conv + eval loading, model metadata, ordering, composite metrics
│   │   ├── analysis.py                  stats battery + plotting
│   │   └── eval.py                      async oracle pipeline (used by Run_Eval) + metadata-driven row factory
│   └── pref_emb/preference_analysis.ipynb   standalone analysis on PTO pref_trees
└── HF_key.txt, openai_key.txt
```

**Single canonical copies.** `system_prompts_builder.py` and `questionnaires.py`
live ONLY at `code/` root — the EDA's `lib/__init__.py` prepends `code/` to
`sys.path` so EDA modules import the same canonical files. No more drift.

## Diagnostic: partial-conversation oracle (Partial_Conv_Oracle_EDA)

**Question.** PTO and GRPO_Exp3 score *partial* conversations (slices as short
as 2 turns) as their training reward, but the thesis evaluates on *final*
full conversations. Is the partial reward a faithful proxy?

**Method.** Pick `Base` + best `L5_Q1Q2_V*`. Slice each of their 96 convs at
every other patient turn, score every cut with Q1+Q2, compare against the
existing final-conv Q1+Q2 scores. All cuts cached to `data/pto_Exp2/eval_scores/partial_q1q2/`.

**Headline.** Pairwise rank agreement (sign-of-difference vs final) is
- only **0.66 (Base) / 0.73 (L5_V10)** at `n_turns=2` — barely above chance (0.5),
- clears **0.8 at n_turns ≈ 10**, **0.9 at n_turns ≈ 30**, monotonically increasing,
- oracle reproducibility noise is ~0.07–0.10 mean |Δ|, so the gap is **structural, not noise**.

**Implication.** Short training cuts can't observe whether the therapist
delivered on Q1/Q2 by session end, so the oracle scores them on "did the
opening look promising?". Optimising that proxy biases the model toward
strong-looking openings regardless of follow-through.

## MIN_CONV_LENGTH filter — wired in both trainers

Direct response to the Partial_Conv_Oracle_EDA finding.

- **GRPO_Exp3.** Cell 1's `MIN_CONV_LENGTH` → `TrainingConfig.min_conv_length` →
  `extract_prompts_from_conversations(min_conv_length=...)` in [_shared/convs.py](code/_shared/convs.py).
- **PTO_Exp3.** Cell 1's `MIN_CONV_LENGTH` → `PTOConfig.min_conv_length`. In `greedy`
  mode it's where the **tree starts** (prefix length, must be EVEN so the prefix ends on
  a patient turn); in `independent` mode it's the slice filter (`build_pref_pairs_for_conversation`
  skips branch points whose conv-so-far is shorter). Either way: no training context below MCL.
- **Semantics.** Drop slices/branches where the conversation-so-far has fewer than `MIN_CONV_LENGTH` total utterances (same `n_turns` unit as Partial_Conv_Oracle_EDA — therapist + patient combined).
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
4. **Score + EDA.** In [eda/lib/config.py](eda/lib/config.py), add a registry entry pointing to the new run's conversation folder. Then [eda/Run_Eval.ipynb](eda/Run_Eval.ipynb) (resume-safe) → [eda/Conv_EDA.ipynb](eda/Conv_EDA.ipynb).

## Running PTO_Exp3

1. **Configure.** [code/PTO_Exp3/train_PTO_Iterative.ipynb](code/PTO_Exp3/train_PTO_Iterative.ipynb) cell 1 = flat globals. Key extra knobs vs GRPO: `PREF_TREE_MODE` (`greedy`|`independent`), `NUM_BRANCHES_PER_TURN`, `PREF_FILTER_TAU`, `BRANCH_SAMPLE_TEMPERATURE`, `DPO_BETA`, `DPO_LOSS_TYPE`. `greedy` mode requires an EVEN `MIN_CONV_LENGTH` (so the sliced prefix ends on a patient turn) and slices its trunk seeds from the step-1 convs (no separate prefix-generation pass).
2. **Train.** Same visible-orchestration pattern. Outputs land under `data/pto_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`. Each iteration also saves the constructed pref pairs to `iteration_N/pref_pairs/pairs.csv` (audit trail; the prompt + chosen + rejected + scores per pair).
3. **Inspect + Score + EDA.** Same as GRPO_Exp3 (the TB dashboard is shared via `_shared/tb_plots.py`).

## Look-ahead performance (K>0) — batched rollout LANDED

**Status (2026-06-02).** The K>0 wall-clock bottleneck is fixed:
`simulate_lookahead_batch` in [_shared/reward.py](code/_shared/reward.py) is now a
**lock-step batched rollout**. All B completions advance in unison (patient →
therapist → …), so each therapist look-ahead turn is **one padded batched
`model.generate`** over the active sims instead of B serial batch-of-1 calls —
collapsing ~B·K serial generations into ~K batched ones. Semantics match the
legacy serial path (statistically equivalent, not bit-identical — sampling RNG
differs). Both GRPO (`make_reward_fn`) and PTO (`build_pref_pairs`,
[PTO_Exp3/pto_trainer.py](code/PTO_Exp3/pto_trainer.py)) get it through the shared fn.

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

**Knob.** `LOOKAHEAD_SUB_BATCH_SIZE` (notebook cell 1 → `LookaheadConfig.lookahead_sub_batch_size`,
default `32`; `None` = all active sims in one call). Halved automatically on OOM
(kept sticky for the rest of the rollout).

**Telemetry.** The existing `reward_fn` line now reports the batched cost:
`Look-ahead: N sims × K=… in X.Xs (… ended early; batched, G GPU calls, sub_batch=S)`.
The legacy `simulate_lookahead_single` / `_generate_therapist_single_async` are kept
(marked LEGACY) as the equivalence-check reference, not on the hot path.

**Validation harness.** [_shared/lookahead_check.py](code/_shared/lookahead_check.py)
(`make_quick_fixtures` + `compare_serial_vs_batched`) runs both paths on the same
fixtures and prints realized-turn + Q1+Q2 reward mean/std for each plus the batched
speedup. Wired as an **optional section 6 cell** in
[GRPO_Exp3/train_GRPO_Iterative.ipynb](code/GRPO_Exp3/train_GRPO_Iterative.ipynb)
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

## Sweep priority (updated 2026-06-04)

0. **Full local bf16 PTO_Exp3 greedy quicktest** (`RUN_MODE="quicktest"`, `USE_4BIT=False`, `PREF_TREE_MODE="greedy"`) — **immediate next action.** Shake out the mirrored config + the new greedy true-PTO mode (committed `e27b9de`) end-to-end. First real-model run of greedy; the `_greedy_smoke.py` test was local fakes only. bf16 only — 4-bit crashes on the local Blackwell GPU. **iter-2 crash mitigated:** the first attempt got through iter-1 DPO but rebooted the PC at the iter-2 DPO step (the TRL `"ref"`-adapter forward-in-backward on sm_120); `precompute_ref_log_probs=True` (`DPO_PRECOMPUTE_REF_LOGPS` knob) moves that ref forward into a no-grad pre-pass, and the **isolated `_iter2_dpo_smoke.py` test PASSED** (first time the iter-2 step survived locally). Now confirm the *full* pipeline produces `iteration_2/adapter/` + `model_iter_2`. If GRPO's quicktest (trimmed block) also reboots at iter-2, it shares the root cause but has no precompute knob → merge-each-iter or Colab.
1. **K=3 look-ahead quicktest** — ✅ equivalence validated; 🔄 GRPO end-to-end re-running on Colab post-torchao-fix.
2. GRPO_Exp3 @ K ∈ {0, 5}, **MCL = 12**.
3. **PTO_Exp3 @ K ∈ {0, 5}, MCL = 12** — config matched to GRPO; run alongside GRPO in parallel sessions.
4. Maybe → either method @ MCL = 2.
5. Maybe → other training oracles (WAI-SR / CSQ-8 / MI-SAT / MITI).

## Dependency stack — audited 2026-06-01

Trainers were audited against the latest docs of the pinned stack
(`transformers==5.8.1`, `trl==1.4.0`, `peft==0.19.1`, `huggingface_hub==1.14.0`,
`wandb==0.26.1`) and are **verified current** — despite the lingering "TRL
v0.28" comments in the code, nothing is deprecated:
- **`scale_rewards="group"`** ([GRPO_Exp3/grpo_trainer.py](code/GRPO_Exp3/grpo_trainer.py)) is the TRL **default** (`"group"/"batch"/"none"`), not a stale value.
- **async reward fn** ([_shared/reward.py](code/_shared/reward.py)) is natively awaited by TRL 1.x (`inspect.iscoroutinefunction` → `asyncio.gather`); extra dataset columns forwarded as kwargs; per-sample `None` supported.
- `processing_class=`, `eval_strategy=` already on the new transformers-5/TRL-1 API.
- `hf_xet` is a **required transitive dep** of `huggingface_hub` 1.x — already installed, nothing to add.
- `gpt-4o-mini-2024-07-18` (patient + oracle) has **no API retirement date** per OpenAI dev docs (the only relevant shutdown is `gpt-4o-2024-05-13`, a different model).

Same-session polish (now in code): both notebooks' Colab install cell is
**pinned to requirements.txt** (commented; `weave` dropped), `authenticate()`
sets `WANDB_LOG_MODEL="checkpoint"` (versioned adapter artifact, third backup),
and both configs set `run_name=current_adapter_repo`.

**Update 2026-06-03.** Install cell now also (commented) `%pip uninstall -y torchao` —
Colab pre-bakes torchao<0.16.0, which peft 0.19.1 rejects by *raising* inside
`get_peft_model`'s `dispatch_torchao` (crashed both trainers at iter 1). A100 optimizer
batch raised to **16 decision-points/step** (GRPO `TRAIN_BATCH_SIZE`=128, PTO DPO 16×1;
LR held). `NUM_ITERATIONS` 8→10 both. Trainer modules renamed `trainer.py` →
`grpo_trainer.py` / `pto_trainer.py` (avoids a `from trainer import` collision when both
notebooks share one local kernel — sys.modules cached the first-loaded trainer).

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
`data/pto_Exp2` stays a **real local dir** (2.4 GB static reference EDA reads every
run — do NOT link it).

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
push `data/` (the symlink targets already live in Drive; `pto_Exp2` is a 2.4 GB local-only
reference) or `eda/` (local-only). Keys come from **Colab Secrets** (`OPENAI_API_KEY`,
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

- **`config.ORACLE_TOKEN_ALIASES`** — add new oracle-name aliases here (CSQ vs CSQ_8 etc.). `data._normalize_oracle_token(strict=True)` raises on unknowns; default `strict=False` lets unknowns fall through to "Other" for backward compat.
- **`config.COMPOSITE_METRICS`** — add new composites (mean across multiple source columns) here. Currently holds just `Q1Q2_Mean`; the same pattern can produce `MITI_GlobalMean` etc.
- **`config.EXPERIMENTS`** — registry of trained-model data locations. Add new entries as runs land in `data/grpo_Exp3/conversations/...` or `data/pto_Exp3/conversations/...`.

## Gotchas

- **HF model-card READMEs** inside `data/grpo_Exp3/runs/.../checkpoint-*/` are auto-generated — DO NOT delete or treat as project docs.
- **`Partial_Conv_Oracle_EDA` knobs** `MIN_TURNS=2` and `SAMPLE_EVERY_N_PATIENT_TURNS=2` are part of the cache key — changing them invalidates `data/pto_Exp2/eval_scores/partial_q1q2/`.
- **Pref-tree audit trail.** PTO_Exp3 writes `iteration_N/pref_pairs/pairs.csv` per iter. Don't delete — they're how you debug "why is this iteration's DPO update weird?" without rerunning generation + branching + scoring (the expensive part).
- **An archived 23 MB K=3 PTO_Exp3 smoke-test** from the V4 era lives in `../archive/pto_v2_smoke/`. Ignore for new work.
