# Exp3_PTO_GRPO — ACTIVE (main thesis chapter)

Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle. Two methods compared
under matched look-ahead + oracle:
- **PTO_Exp3** (preference-tree → DPO loss). Refactored as a lean sibling of
  GRPO_Exp3 (one notebook + one `trainer.py`, sharing `_shared/`). Output dir:
  `data/pto_Exp3/`. The Exp2-sourced `data/pto_Exp2/` artifacts are still read by the
  EDA registry but **not regenerated here** unless you re-run PTO_Exp3.
- **GRPO_Exp3** (iterative). Sweep not yet run — definite next step.

Reward (training) = **Q1 + Q2 only**, matching the ICLR look-ahead paper.
Reward (eval) = all six MI questionnaires (Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI).

## Trainer pattern

Both trainers (`code/GRPO_Exp3/`, `code/PTO_Exp3/`) follow the same shape:

```
<METHOD>_Exp3/
├── train_<METHOD>_Iterative.ipynb   thicker — per-iteration orchestration visible
└── trainer.py                       <Method>Config + run_one_iteration + run_final_eval + write_run_metadata + build_wandb_ctx
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

**Per iteration `n` (loop body in [GRPO_Exp3/train_GRPO_Iterative.ipynb](code/GRPO_Exp3/train_GRPO_Iterative.ipynb), helpers in [trainer.py](code/GRPO_Exp3/trainer.py)):**

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

**Per iteration `n` (loop body in [PTO_Exp3/train_PTO_Iterative.ipynb](code/PTO_Exp3/train_PTO_Iterative.ipynb), helpers in [trainer.py](code/PTO_Exp3/trainer.py)):**

1. **Generate rollouts.** Same as GRPO step 1 — `π_n` simulates 96 conversations
   versus `P`. Saved to `data/pto_Exp3/conversations/.../model_iter_{n-1}/`.
2. **Build preference tree.** For each conversation, for each therapist turn
   index `t` whose prefix-so-far has `≥ MCL` utterances and that is not the
   final turn:

   a. **Branch.** Sample `M = NUM_BRANCHES_PER_TURN` candidate therapist
      completions from `π_n` at `BRANCH_SAMPLE_TEMPERATURE` (independent draws
      from the same prefix).

   b. **Score each branch.** For each candidate `t_m`, compute `r_m` exactly
      as in GRPO step 3b — including K-turn look-ahead when `K > 0`.

   c. **Best/worst with τ filter.** Let `chosen = argmax_m r_m`,
      `rejected = argmin_m r_m`. If `r_chosen - r_rejected > PREF_FILTER_TAU`,
      emit one preference pair `(prefix_prompt, chosen_text, rejected_text)`;
      otherwise skip this branch point (the policy already produces tied-quality
      siblings, no informative gradient).
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
│   ├── _shared/                         5 cross-method modules (GRPO_Exp3 + PTO_Exp3 both import)
│   │   ├── __init__.py                  public-API re-exports
│   │   ├── runtime.py                   Colab/local detect, auth, paths, preflight
│   │   ├── model.py                     tokenizer/quant/LoRA + checkpoint discovery + iteration resume
│   │   ├── convs.py                     conv state + async gen + per-turn prompt extraction (MCL filter)
│   │   ├── reward.py                    oracle scoring + K-turn look-ahead + reward-fn factory
│   │   └── tb_plots.py                  TB callbacks + logging lifecycle + TB parser + plot dashboard
│   ├── GRPO_Exp3/
│   │   ├── train_GRPO_Iterative.ipynb   visible orchestration loop
│   │   └── trainer.py                   TrainingConfig + run_one_iteration + run_final_eval + …
│   └── PTO_Exp3/
│       ├── train_PTO_Iterative.ipynb    visible orchestration loop (mirrors GRPO_Exp3)
│       └── trainer.py                   PTOConfig + run_one_iteration + build_pref_pairs_for_conversation + …
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
- **PTO_Exp3.** Cell 1's `MIN_CONV_LENGTH` → `PTOConfig.min_conv_length` →
  `build_pref_pairs_for_conversation` skips branch points whose conv-so-far is shorter.
- **Semantics.** Drop slices/branches where the conversation-so-far has fewer than `MIN_CONV_LENGTH` total utterances (same `n_turns` unit as Partial_Conv_Oracle_EDA — therapist + patient combined).
- **Default = 2** = no-op. Recommended exploratory values: `10` (EDA's 0.8 threshold), `30` (0.9 threshold).
- **Encoded in `EXPERIMENT_NAME`** as `_MCL{N}` so runs at different MCL never share an output folder.

## EXPERIMENT_NAME schemes

- GRPO_Exp3: `GRPO_Iterative_Oracle_Llama32-1B_LA{K}_MCL{MCL}_G{G}`
- PTO_Exp3:  `PTO_Iterative_Oracle_Llama32-1B_LA{K}_MCL{MCL}_M{NUM_BRANCHES_PER_TURN}`

Different sweep arms write to disjoint dirs — runs never collide.

## Running GRPO_Exp3

1. **Configure.** [code/GRPO_Exp3/train_GRPO_Iterative.ipynb](code/GRPO_Exp3/train_GRPO_Iterative.ipynb) cell 1 = flat globals.
2. **Train.** Run top-to-bottom. The orchestration loop is in the notebook (cells after `cfg = TrainingConfig(...)`), composed from `run_one_iteration` / `run_final_eval` in [trainer.py](code/GRPO_Exp3/trainer.py). Resumes from latest completed iter via [_shared.resolve_start_state](code/_shared/model.py). Outputs under `data/grpo_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`; per-run `run_metadata.json` at the run root.
3. **Inspect.** Last cell: `scan_scalar_tags` + `plot_iteration_metrics` + inline TensorBoard. `plot_iteration_metrics` applies per-iteration step offsets so cross-iter curves chain end-to-end (dotted vlines mark iter boundaries).
4. **Score + EDA.** In [eda/lib/config.py](eda/lib/config.py), add a registry entry pointing to the new run's conversation folder. Then [eda/Run_Eval.ipynb](eda/Run_Eval.ipynb) (resume-safe) → [eda/Conv_EDA.ipynb](eda/Conv_EDA.ipynb).

## Running PTO_Exp3

1. **Configure.** [code/PTO_Exp3/train_PTO_Iterative.ipynb](code/PTO_Exp3/train_PTO_Iterative.ipynb) cell 1 = flat globals. Key extra knobs vs GRPO: `NUM_BRANCHES_PER_TURN`, `PREF_FILTER_TAU`, `BRANCH_SAMPLE_TEMPERATURE`, `DPO_BETA`, `DPO_LOSS_TYPE`.
2. **Train.** Same visible-orchestration pattern. Outputs land under `data/pto_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`. Each iteration also saves the constructed pref pairs to `iteration_N/pref_pairs/pairs.csv` (audit trail; the prompt + chosen + rejected + scores per pair).
3. **Inspect + Score + EDA.** Same as GRPO_Exp3 (the TB dashboard is shared via `_shared/tb_plots.py`).

## Sweep priority (2026-05-28)

1. GRPO_Exp3 @ K ∈ {0, 5}, MCL = 10 (definite next step).
2. Maybe → PTO_Exp3 @ K ∈ {0, 5}, MCL = 10.
3. Maybe → either method @ MCL = 2.
4. Maybe → other training oracles (WAI-SR / CSQ-8 / MI-SAT / MITI).

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

### Sync (rclone)
```powershell
# pull from Drive
rclone sync gdrive:Thesis_PTO_GRPO C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO `
    --exclude "/.venv/**" --exclude "/archive/**" --progress
# push to Drive
rclone sync C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO gdrive:Thesis_PTO_GRPO `
    --exclude "/.venv/**" --exclude "/archive/**" --progress
```
`rclone sync A B` makes B mirror A — **deletes files in B not in A**. Use `rclone copy` for additive, `rclone check` for a dry-run diff.

## EDA extension points

- **`config.ORACLE_TOKEN_ALIASES`** — add new oracle-name aliases here (CSQ vs CSQ_8 etc.). `data._normalize_oracle_token(strict=True)` raises on unknowns; default `strict=False` lets unknowns fall through to "Other" for backward compat.
- **`config.COMPOSITE_METRICS`** — add new composites (mean across multiple source columns) here. Currently holds just `Q1Q2_Mean`; the same pattern can produce `MITI_GlobalMean` etc.
- **`config.EXPERIMENTS`** — registry of trained-model data locations. Add new entries as runs land in `data/grpo_Exp3/conversations/...` or `data/pto_Exp3/conversations/...`.

## Gotchas

- **HF model-card READMEs** inside `data/grpo_Exp3/runs/.../checkpoint-*/` are auto-generated — DO NOT delete or treat as project docs.
- **`Partial_Conv_Oracle_EDA` knobs** `MIN_TURNS=2` and `SAMPLE_EVERY_N_PATIENT_TURNS=2` are part of the cache key — changing them invalidates `data/pto_Exp2/eval_scores/partial_q1q2/`.
- **Pref-tree audit trail.** PTO_Exp3 writes `iteration_N/pref_pairs/pairs.csv` per iter. Don't delete — they're how you debug "why is this iteration's DPO update weird?" without rerunning generation + branching + scoring (the expensive part).
- **An archived 23 MB K=3 PTO_Exp3 smoke-test** from the V4 era lives in `../archive/pto_v2_smoke/`. Ignore for new work.
