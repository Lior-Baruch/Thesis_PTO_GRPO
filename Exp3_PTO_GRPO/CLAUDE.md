# Exp3_PTO_GRPO — ACTIVE (main thesis chapter)

Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle. Two methods compared
under matched look-ahead + oracle:
- **PTO_Exp3** (preference-tree → DPO loss). Lean sibling of GRPO_Exp3 (one notebook
  + one `pto_trainer.py`, sharing `_shared/`). **Controlled hyperparameters matched to
  GRPO_Exp3** (2026-06-03): NUM_ITERATIONS=10, MCL=12, K∈{0,5}, gen temps + API
  concurrency; M (`NUM_BRANCHES_PER_TURN`)=8 mirrors GRPO's `NUM_GENERATIONS`;
  `DPO_BETA`=0.1 kept (DPO loss temp, not GRPO's KL β). bf16 `USE_4BIT` toggle + a
  zero-pairs actionable error + train/eval split fix also landed. Output dir:
  `data/pto_Exp3/`. (The Exp2-sourced `data/pto_Exp2/` reference artifacts + the frozen
  `eda/archive_exp2/` EDA were **removed 2026-06-15** — Exp3 is the only axis now; see Data lineage.)
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
│   │                                    (the Exp2-sourced pto_Exp2/ reference was removed 2026-06-15)
│   ├── grpo_Exp3/                       produced by GRPO_Exp3 runs
│   │   ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training}/
│   │   ├── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
│   │   └── eval_scores/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
│   └── pto_Exp3/                        produced by PTO_Exp3 runs (same shape as grpo_Exp3)
│       ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training, pref_pairs/}
│       ├── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
│       └── eval_scores/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
├── eda/                                 verified runnable end-to-end
│   ├── Run_Eval.ipynb                   async oracle pipeline → eval_scores/ (resume-safe; uses oracle_scoring/, registry-driven)
│   ├── 1_Outcomes.ipynb               [EVAL] family=1_outcomes — all-metric trajectory grid + PER-METRIC learning-curve catalog (trajectories/ subfolder; peaks auto-flagged) + effect forest + per-model bars + scorecard
│   ├── 2_Heterogeneity.ipynb          [EVAL] family=2_heterogeneity — EVERY metric split by persona trait (cooperation_level/ + problem/ subfolders) + final-iteration endpoint bars
│   ├── 3_Mechanism.ipynb              [EVAL] family=3_mechanism — behaviour drift + merged behavior_by_iter table + subscales + factor structure (corr + loadings) + reward_hack_panel + question/over-praise cross-checks + session shape + transcripts
│   ├── 4_Training_and_Reliability.ipynb [TRAINING↔EVAL] family=4_training — TB curves + reward dist + advantage signal + degeneration + reward-faithfulness (reliability curve, proxy-vs-eval, PTO margin-by-depth)
│   ├── 5_Preference.ipynb             [TRAINING] family=5_preference — PTO Mass-Mean-Probe: word ranking + drift + direction-drift(2D) + learned/unlearned words + MI-concept drift + K0-vs-K5 (PTO-only)
│   ├── 6_Stats.ipynb                  [EVAL] family=6_stats — ALL heavy tables: merged main_results (target col) + Friedman + merged vs-base/method/K paired + all-metric slopes + PCA + GRPO iter-9 anomaly check
│   ├── render_views.py                         DRIVER: regenerate results/<view>/ for all 6 notebooks via nbconvert (sets EDA_VIEW; --output-dir tmp; --nb takes LIST indices 0..5)
│   ├── eda_analysis/                            Exp3 analysis package (disk-discovery, read-only). 9 modules: plumbing merged 14→9 (2026-06-18); old submodule names aliased
│   │   ├── __init__.py                  WORKSPACE_ROOT + sys.path + re-exports + QUESTIONNAIRES/WARMTH/ORTHOGONAL/LOWER_IS_BETTER + display_label + submodule aliases (figures/plots→plotting; discovery/personas/scores/select→data)
│   │   ├── config.py                    CONTROL SURFACE: EdaConfig (incl. the VIEW knob) + notebook_setup(cfg)→Setup(ARMS,SCORES,PALETTE,METRICS,ORACLE_NOISE,RESULTS_DIR,VIEW,CFG). view→ks filter + results/<view>/ root. (absorbed notebook.py)
│   │   ├── data.py                      LOAD+SHAPE: discovery (Arm/discover_arms/filter_arms) + TRUE-persona recovery + scores_long backbone (+Q1Q2/subscales/to_wide/collapse_base/add_derived_mitiprof_rows/select_scores) + all/best selection. (merged discovery+personas+scores+select)
│   │   ├── plotting.py                  FIGURE layer: style helpers (set_style/arm_palette/grid/model_order/apply_score_axis) + named plots (effect_forest/overlay_trajectory/heterogeneity_grid/factor_loadings_bars/leaderboard_scorecard/diverging rubric_correlation_heatmap…). (merged figures+plots; self-aliases `figures`)
│   │   ├── stats.py                     BOTH batteries + Friedman/Kendall-W + main_results_table + paired_method/k_comparison + rubric PCA/corr + rubric_factor_space
│   │   ├── behavior.py                  MITI behavior counts (eval) + MICI loader + over-praise cross-check + structural text metrics (semantic regex demoted to lex_* sanity-check)
│   │   ├── training.py                  generations.jsonl proxy reward + degeneracy scan + pref_pairs + advantage_signal_by_iter / reward_distribution_frame (both methods)
│   │   ├── pref.py                      PTO pref: margins + embeddings + Mass-Mean-Probe (preference_direction/word_projection/MI category_projection) + pref_word_ranking
│   │   └── exports.py                   VIEW-aware: save_fig (PNG) / save_table (md+xlsx) → results/<view>/<family>/ ; per-call group= override incl. NESTED subpaths; set_view + set_export_group + set_formats + save_provenance + walk-based build_index + reset_results (PRESERVES SUMMARY.md)
│   ├── results/                         GENERATED thesis artifacts in 3 VIEW trees: all/ · L0/ · L5/, each with figures|tables/<N_family>/ (family number == producing-notebook number) + INDEX.md + hand-authored SUMMARY.md
│   └── oracle_scoring/                  LEGACY package — kept ONLY for Run_Eval scoring (Exp3 registry; NOT the new analysis)
└── HF_key.txt, openai_key.txt
```

**Thesis artifacts.** `results/<view>/figures/` (`.png`) and `results/<view>/tables/` (`.md`+`.xlsx`)
are **generated** by `eda_analysis.save_fig`/`save_table` (the `formats=` kwarg can request extras for
a one-off; per-call `group=` overrides the family, incl. nested subpaths). Each notebook regenerates
its own family; `python render_views.py` regenerates everything. Reproducible from code; tracked in git.

**Reorg-by-topic pass (2026-07-02).** No special "main" notebook — **topic notebooks ↔ numbered result
families, 1:1** (notebook number == family number, so any artifact under `results/<view>/` traces to
its producing notebook). Per-metric catalogs added (9 trajectory curves w/ auto peak-marking under
`1_outcomes/trajectories/`; 2 traits × 9 metrics under `2_heterogeneity/<trait>/`). Dropped 4 duplicate
figures ONLY (contrast_overlay, outcomes_headline, unannotated trajectory_Q1Q2, orthogonal-only forest
— `overpraise_crosscheck` + `faithfulness_proxy_vs_eval` KEPT, re-tiered). Stats tables merged 13→~11
(main_results final+best w/ `target` col; vs_base/method/K paired tables merged with key columns; NEW
`grpo_iter9_check` probes GRPO's all-metric simultaneous iter-9 dip). Labels: `Q1Q2→"Q1+Q2"`, Q1/Q2 raw
(no "Satisfaction…"). exports.py: per-call `group=` + walk-based `build_index()` (nested folders were
silently omitted) now the final cell of EVERY notebook; `single_metric_trajectory(oracle_noise=None)`
suppresses the band; stale "PC1≈91%/6 rubrics" caveat → 9-metric PC1≈55% text.

**NEXT EDA SESSION — backlog (2026-07-02, Lior's notes; START by asking clarifying questions).**
Data state: **full L0 (PTO_LA0 + GRPO_LA0, 0–10) + partial L5 (PTO_LA5 0–4, GRPO_LA5 base)**; no L2 data yet.
1. **Propagate the `1_Outcomes` style everywhere it fits** — a MAIN combined grid (`trajectories_all_metrics`)
   + a per-metric SUBFOLDER (`trajectories/`). Apply to other families where sensible: `2_Heterogeneity`
   should get a combined "all-metrics" overview per trait alongside its per-metric subfolder; consider the
   same grid+subfolder pattern for behavior / other multi-panel figures. (CLARIFY which families.)
2. **`4_Training`: add a GRPO "preference-margin" analog.** GRPO has no chosen/rejected pairs, but a natural
   analog to PTO's chosen−rejected `margin` is the within-group reward **spread = max−min** (or plot group
   min & max) per iteration — add it beside/into `advantage_signal_sidebyside` so both methods show a
   comparable decisiveness signal. Data is in `generations.jsonl` group rows (group_mean/std already there;
   need per-group min/max). (CLARIFY: max−min range, or min+max lines, or best−worst margin.)
3. **Questions vs Questions/turn — resolve the suspected bug.** Confirm whether oracle `B3_Q` (count) and
   regex `q_per_turn` (rate) SHOULD correspond and by what relation (`B3_Q ≈ q_per_turn × n_th_turns`?), and
   whether the GRPO iter-10 divergence (B3_Q≈4.1 but q_per_turn≈0.15) is a real bug or a semantic gap
   (MITI codes question-*function* utterances; regex counts literal `?`). See `3_Mechanism` §4b
   `question_rate_crosscheck` + `behavior.py`. Add a unit/sanity check if it's a bug.
4. **Main questionnaire labels: show the ORIGINAL acronym.** `DISPLAY_NAMES` currently maps
   `MITI→"MI Integrity"`, `CSQ-8→"Client Satisfaction"`, `WAI-SR→"Working Alliance"`, `MI-SAT→"MI Satisfaction"`.
   Lior wants the original names (only or ALSO) — e.g. `"MITI"` or `"MITI (MI Integrity)"`. (CLARIFY:
   acronym-only vs acronym + descriptive.)
5. **Better articulate warmth vs orthogonal** — the 5 warmth/alliance rubrics (one PC1 factor) vs the
   orthogonal axes (PCT / MICI↓ / R:Q / %CR / %MICO). Improve the explanation/labels/grouping in the EDA
   (and possibly a one-figure or one-para explainer).
6. **Check + understand + refine `stats.py`** — review the stat batteries/tables for correctness + clarity
   (paired tests, Holm scoping, effect sizes, the new merged tables + `grpo_iter9_check`).
7. **General EDA review** — this is the active workstream; L0 is the primary read, L5 partial.
Open cosmetic: tables-only `6_Stats` still writes an empty `figures/6_stats/_provenance.md` (harmless;
INDEX ignores it) — optionally suppress provenance for tables-only notebooks.

**Single canonical copies.** `system_prompts_builder.py` and `questionnaires.py`
live ONLY at `code/` root — both `eda/oracle_scoring/__init__.py` and `eda/eda_analysis/__init__.py` prepend
`code/` to `sys.path` so they import the same canonical files. No more drift.

**EDA refactor (2026-06-10).** The analysis EDA was reorganized **by research question** and made
**method-symmetric** (the prior 2026-06-09 rebuild created the `eda_analysis/` package; this pass restructured
the notebooks on top of it). **Hybrid plotting:** the recurring figures now live as named functions in
`eda_analysis/plots.py` (defined once, called from multiple notebooks), genuinely one-off exploration stays
inline. **One-call setup** `eda_analysis.notebook_setup()` → `S.*` kills the byte-identical cell-1 boilerplate.
Notebook set, by thesis question: **`00_Main_Results`** (thin canonical artifacts + index),
**`01_Did_It_Work`** (each arm vs base — all arms), **`02_PTO_vs_GRPO`** (RQ ii; absorbs the old
`Exp3_DeepDive`; training internals shown side-by-side, never method-gated), **`03_LookAhead_K`**
(RQ i; K0-vs-K5), **`04_Mechanism_and_Behavior`** (behavior/faithfulness/heterogeneity — all arms),
**`05_Preference_LatentSpace`** (PTO Mass-Mean-Probe — PTO-only by construction) + **`Iteration_Reward_EDA`**.
Every per-arm analysis now runs for **both methods** (only the preference probe stays PTO-only — GRPO has
no chosen/rejected pairs). The buried cross-method/K comparisons became
`stats.paired_method_comparison`/`paired_k_comparison`; training internals became
`training.advantage_signal_by_iter`/`reward_distribution_frame`. **Disk-discovery-driven** (no registry),
**true-persona** recovery, **both** stat batteries. Exports trimmed to **one format each** (PDF figs /
MD tables, idempotent `CAPTIONS.md`). (The old Exp2-shaped `Conv_EDA`/`Partial_Conv_Oracle_EDA`/`pref_emb`
notebooks were **frozen in `eda/archive_exp2/`** and then **removed 2026-06-15** with the `pto_Exp2` data —
the partial-conv reliability diagnostic now lives, rebuilt on Exp3 data, in `3_Reward_Reliability.ipynb`.)
`eda/oracle_scoring/` survives ONLY for `Run_Eval.ipynb` (registry-driven
scoring). ⚠ The old `oracle_scoring` patient-characteristic join is **wrong for Exp3** (per-iter shuffle) — use
`eda_analysis/personas.py`. **Validated 2026-06-10:** all six notebooks ran top-to-bottom via nbconvert
(`thesis-venv313`) on the current disk state. See "New EDA workflow" below.

**Figure-readability pass (2026-06-10, later).** Fixed the four figures that read poorly: (1) the 4
near-identical arm-bases now pool into one descriptive `Base` via `scores.collapse_base` (cross-model
bar/rank views only — paired vs-base stats still use each arm's own base); (2) the unreadable
26-model × 3–4-subscale grouped-bar wall (`subscales_WAI_MITI.pdf`, retired) → `plots.subscale_trajectory_grid`
(subscale lines across iterations, one panel per parent×arm → `subscale_trajectories.pdf`);
(3) preference drift across iterations via `pref.pref_word_drift_heatmap` (top words × iteration) +
`pref.plot_category_drift` (MI-concept lines), beside the pooled `pref_word_ranking`; (4) polish —
saturated LA5 tints, short x-labels (`figures.short_label`), shared legends above grids, and the
PC1≈91% shared-factor caveat printed under the trajectory grid. `01` now leads with the trajectory grid
and demotes the per-model bars to an Appendix. The old `plots.subscales_by_model` was removed.
**Validated:** package smoke + `00`/`01`/`05` via nbconvert (`thesis-venv313`).

**Restructure-by-purpose pass (2026-06-10, latest).** The notebooks were **reorganized by purpose**
(was by research question) into the **7** above (`0_Headline` … `6_Detailed_Stats`), every section
tagged **`[EVAL]`** vs **`[TRAINING]`**, **markdown trimmed concise**, **all heavy tables moved to
`6_Detailed_Stats`** with the headline "did it work" shown as an **`effect_forest`** dot-plot instead,
**thin arms (<3 iters) filtered** (no NaN rows), **violins dropped**. New first-class analyses:
`3_Training_Diagnostics` surfaces the **TensorBoard training curves** (`training.tb_curves` —
self-contained TB parse, no torch/trl import so the EDA stays host-agnostic); `4_Reward_Reliability`
**rebuilds the Exp2 partial-conv reliability curve on Exp3 data** (`training.load_branch_reliability` +
`stats.rank_agreement_by_nturns`, from the per-branch `prefix` already in `generations.jsonl` — no new
oracle pass) and contrasts **LA0 vs LA5** (does look-ahead make the short reward more faithful?);
`5_Preference_LatentSpace` gains **direction-drift (2D PCA + cosine)**, **learned/unlearned words**, and
a **K0-vs-K5** preference contrast. **Validated:** package smoke + all 7 notebooks via nbconvert
(`thesis-venv313`). The 2026-06-09/-10 notes above are kept as history.

**Control + organization pass (2026-06-14, latest).** Added a single flat-globals control surface and
reorganized exports + notebooks. **(1) `EdaConfig`** (new [eda_analysis/config.py](code/../eda/eda_analysis/config.py))
bundles every knob — arm filter (`methods`/`ks`/`modes`/`arm_labels`), metric subset + `warmth_only` +
`add_derived_mitiprof`, `selection` (all/best), plot scales (`context`/`font_scale`/`dpi`/`panel`/
`ncols`/`score_ylim`/`share_y`/`palette_overrides`), and exports (`export_group`/`fig_formats`/
`table_formats`). Cell 1 is now `cfg = eda_analysis.EdaConfig(export_group=…)` → `S = notebook_setup(cfg)`
(defaults reproduce old behaviour; `notebook_setup(cfg, k=v)` overrides on the fly). `notebook_setup`
filters arms (`discovery.filter_arms`), applies scales (`figures.set_style(cfg)` + `_SCALE` defaults
read by `grid`/`apply_score_axis`), appends the derived ratios (idempotent), and writes a **provenance
banner**. **(2) Organized exports:** `save_fig`/`save_table` route into `results/<figures|tables>/
<group>/` (`set_export_group`), per-group `CAPTIONS.md`, `build_index()`→`results/INDEX.md`,
`save_provenance`, `reset_results`. The old flat dump was **wiped + regenerated** into the 6 group
subfolders. **(3) Notebooks 7→6:** merged `0_Headline`+`1_Eval_Results` → **`0_Eval_Results`** (headline
trio computed once — no duplicate forest — + full outcomes + contrasts + scorecard + appendix);
renumbered `2…6 → 1…5`. **(4) Extras:** `plots.factor_space_scatter` (PC1×PC2 — warmth clusters on PC1,
orthogonal axes load PC2; first read: PC1 59%, PC2 16% pooled), **diverging** `[-1,1]`
`rubric_correlation_heatmap`, `plots.leaderboard_scorecard` (warmth + PCT/MICI↓/R:Q/%CR/%MICO),
`display_label` (lower-is-better ↓). **Note:** PCT + MICI are now scored on disk — first read:
GRPO_LA0 is more reflective (**R:Q 1.04** vs PTO 0.75) while PTO is slightly *less* MI-inconsistent
(**MICI 0.49** vs GRPO 0.54). [pass-2 below superseded the biplot with `factor_loadings_bars`.]
**Validated:** package smoke + all 6 notebooks via nbconvert (`thesis-venv313`).

**Pass-2 polish (2026-06-14, latest) — formats + merge boundary + readable factor + per-figure control.**
Addressed Lior's notes on the pass above. **(1) Outputs:** figures default to **PNG** images
(`cfg.fig_formats=("png","pdf")` to also emit vector); tables default to **`.md` + `.xlsx`** (a per-group
Excel workbook, one sheet per table — `exports._write_xlsx_sheet`, needs `openpyxl`). `save_fig`/
`save_table` fall back to the cfg-set module defaults (`set_formats`). **(2) Merge boundary fixed:** the
intended merge was eval+behaviour, not headline+eval — split back into a thin **`0_Headline`** (3 figs +
index) and merged eval-results+behaviour into **`1_Eval_and_Behavior`**; `2…5` keep their numbers (titles
renumbered to match). **(3) Factor figure made readable:** replaced the confusing PC1×PC2 biplot with
**`plots.factor_loadings_bars`** (each metric's PC1/PC2 loading as bars — warmth rubrics ~0.44 on PC1,
orthogonal axes ~0) + a plain-language caption. **(4) Control over repetition:** new
`EdaConfig.focus_arms`/`focus_metric`; `eda_analysis.select_scores(...)`; `arms=`/`iters=` on
`single_metric_trajectory`/`trajectory_grid`; **`plots.overlay_trajectory(arms=[…])`** collapses the
per-K + per-method contrast loops into ONE configurable cell; **`plots.heterogeneity_grid`** collapses
the `char×arm` PNG explosion into one figure (panel per arm); the preference probe loops over
`focus_arms ∩ PTO`. **Validated:** package smoke (PNG + xlsx sheet + select/overlay/heterogeneity/
loadings) + all 6 notebooks via nbconvert (`thesis-venv313`); old flat `results/` wiped + regenerated.

**VIEW system + package consolidation + narrative summaries (2026-06-18, latest).** Lior's asks: cleaner
EDA, results split by look-ahead, fewer/easier-to-edit modules, and a written summary. **(1) The VIEW knob.**
Cell 1 of every notebook now leads with `VIEW = os.environ.get("EDA_VIEW", "L0")` → `EdaConfig(view=VIEW, …)`.
`view ∈ {all, L0, L5}` is ONE control that sets BOTH the arm filter (`all`=every arm, `L0`=K=0, `L5`=K=5) AND
the results root, so `results/` now holds **3 parallel trees** `all/ · L0/ · L5/`, each
`figures|tables/<group>/` + `INDEX.md` + a hand-authored `SUMMARY.md`. Wired via `EdaConfig.view` + `_VIEW_KS`
(explicit `ks=` still overrides) in [config.py](eda/eda_analysis/config.py) and a view-aware root
(`set_view`/`_results_root`/…) in [exports.py](eda/eda_analysis/exports.py); `reset_results` clears only the
active view's figures/tables and **never deletes `SUMMARY.md`** (`PRESERVE`). **(2) Plumbing merged 14→9.**
`config.py`+`notebook.py`→**config**; `discovery`+`personas`+`scores`+`select`→**data.py**;
`figures`+`plots`→**plotting.py**. Kept: `stats`/`behavior`/`training`/`pref`/`exports`. The old submodule
names are **aliased** in `__init__` (`figures=plots=plotting`, `personas=scores=discovery=select=data`, also
registered in `sys.modules` so `from eda_analysis.personas import …` resolves), so **no notebook analysis cell
changed** — only cell 1 got the VIEW knob. **(3) Driver** [render_views.py](eda/render_views.py) regenerates all
3 views × 6 notebooks via nbconvert (sets `EDA_VIEW`, `--output-dir tmp` so source notebooks aren't churned).
**(4) Narrative** `results/<view>/SUMMARY.md` (hand-authored, preserved) — L0 is the primary read.
**Validated:** import/alias + view→ks + `target="best"` smoke PASS; 0_Headline@L0 dry-run wrote
`results/L0/figures/headline/*` + `INDEX.md` with `SUMMARY.md` intact; full 3×6 matrix via nbconvert.

### New EDA workflow (replaces "add registry entry → Conv_EDA")
1. **Score** a new run: `Run_Eval.ipynb` still needs a `oracle_scoring/config.py::EXPERIMENTS` entry to know what
   to grade (this one coupling remains by design). Run it → writes `eval_scores/`.
2. **Analyze:** browse `results/<view>/` and open the notebook whose NUMBER matches the family you
   want to change — `1_Outcomes` / `2_Heterogeneity` / `3_Mechanism` / `4_Training_and_Reliability` /
   `5_Preference` / `6_Stats` (topic notebooks ↔ result families, 1:1). Every notebook's cell 1 starts
   with the **VIEW knob** `VIEW = os.environ.get("EDA_VIEW", "L0")` then
   `cfg = eda_analysis.EdaConfig(view=VIEW, export_group=…)` → `S = eda_analysis.notebook_setup(cfg)`.
   **`view` ∈ {all, L0, L5}** is the one control: it sets BOTH the arm filter (`all`=every arm,
   `L0`=K=0 arms, `L5`=K=5 arms) AND the results root, so artifacts land in
   **`eda/results/<VIEW>/<figures|tables>/<N_family>/`** (per-view `INDEX.md` + a hand-authored
   `SUMMARY.md`). All **auto-discover** every arm via `eda_analysis.discover_arms()` — no registry edit.
   Point any figure at a subset with `arms=`/`eda_analysis.select_scores(...)`.
3. **Regenerate all views at once:** `python render_views.py` (views × 6 notebooks via nbconvert,
   kernel `thesis-venv313`, writes the `results/` trees; `render_views.py L0` for one view,
   `… L5 --nb 3` for one view+notebook — `--nb` takes LIST indices 0..5, 0 = `1_Outcomes`). Re-run is
   cheap; arms not yet scored are skipped gracefully (cross-method/K cells degrade to a "not scored
   yet" banner; thin arms < 3 iters are dropped).

See [eda/README.md](eda/README.md) for the full notebook guide + an improvement roadmap.

### Eval results so far (updated 2026-06-15)
Scored: **PTO LA0** iters 0–10, **GRPO LA0** iters 0–10 (**now FINISHED**), **PTO LA5** iters 0–4,
GRPO LA5 base only. **All four arms scored on the full battery incl. the orthogonal axes** (PCT, MICI,
and the derived R:Q/%CR/%MICO). Numbers in the EDA's `Q1Q2 = mean(Q1,Q2)` convention (full tables:
`results/<view>/tables/6_stats/main_results.md` = each arm's FINAL and BEST iter in one table
(`target` column); one-glance `results/<view>/tables/1_outcomes/leaderboard_scorecard.md`).
- **Each arm vs base — large warmth gains.** PTO LA0 Q1+Q2 3.00→**4.26** (final=best, dz 1.43, Friedman
  W=0.45); GRPO LA0 3.07→**4.08 at its iter-8 peak**, falling to **3.75 by iter 10** (final dz 0.72,
  best dz 1.22, W=0.33); PTO LA5 3.00→3.89 in 4 iters (dz 0.88). All warmth rubrics **large** effect,
  Holm p≈0 everywhere.
- **PTO vs GRPO (RQ-ii) — PTO ahead at the matched 10-iter endpoint; GRPO is less stable.** The earlier
  "near-tie at iter 8" was a snapshot artifact: GRPO Q1Q2 **peaks at iter 8 (4.08) then REGRESSES (iter9
  3.81, iter10 3.75)** while PTO climbs stably (4.22→4.26). At the **matched 10-iter endpoint PTO beats
  GRPO 4.26 vs 3.75** (paired PTO−GRPO Q1Q2 +0.51, dz +0.73, Holm p<0.001; MITI/CSQ-8/MI-SAT/PCT also
  favor PTO, and PTO is less MI-inconsistent). Overall OLS slopes: GRPO **0.072**/iter (peak iter 8) vs
  PTO **0.120**/iter (peak iter 10). Earlier matched-iter reads still hold (tie 1–2, GRPO briefly ahead
  @3, PTO ahead @8). ⇒ **Revised core answer: GRPO is competitive *up to its peak* but overshoots into
  reward-hacking and degrades; PTO sustains gains across 10 iters.** With GRPO, peak-iter selection /
  early stopping matters — its best (4.08 @ iter 8) is still below PTO's best (4.26 @ iter 10).
- **Conversation-level mechanism (iter-10, same resistant persona for both methods).** GRPO iter-10
  collapses into nonstop empty praise and never gives the practical advice a resistant patient demands
  6+ times; PTO iter-10 also drifts toward affirmation but **converges to concrete steps** and the
  patient softens. Across all 96 iter-10 convs: **GRPO 0.13 q/turn, 3.61 praise-words/turn vs PTO 0.50
  q/turn, 1.02 praise/turn** — GRPO emits ~3.5× more praise and ~4× fewer questions. The iter-10 eval
  regression IS the over-praise reward-hack the full-conv oracle penalizes; GRPO falls into it harder.
- **Reward-hacking / multi-skill — the orthogonal axes pay off.** As warmth rises, **MI-INCONSISTENT
  behavior rises ~2.3–2.5×** (MICI base 0.21 → 0.49 PTO / 0.54 GRPO; dz 0.78/0.89) — the warmth gains
  come *with* more over-praise/advice, in BOTH methods. **Affirmation drift is confirmed in GRPO too,
  and at iter 10 it is the WORSE offender** (B6_AF 0.52→**1.98**, questions B3_Q 6.4→**4.1**, q/turn
  0.83→**0.15**, R:Q→**1.44** by iter 10) — i.e. GRPO's late regression is exactly this drift running
  away. Mid-run (≤iter 8) GRPO looked *more* reflective (R:Q 1.04 > PTO 0.75); by iter 10 it has dropped
  questions almost entirely. **PTO's drift is milder and plateaus** (iter-10 B6_AF 1.64, q/turn 0.55).
  Patient **change-talk rises modestly**, more for PTO (PCT 0.49→0.63 medium vs GRPO 0.49→0.57
  small). Both kill degeneration loops (loop% 0.49→0). **Adding the orthogonal axes drops PC1 from ≈91%
  → ≈56%** (PC2 ≈16%): warmth is one factor; technique (R:Q/%CR/%MICO) + MICI form a second — so "all
  rubrics up" is genuinely *not* multi-skill. (PCT partly loads on PC1 ≈0.39 — change-talk co-moves with
  warmth.)
- **PTO preference probe is real:** wins_correct 0.65→0.71 over iters (>0.5 = the chosen−rejected
  direction separates the pairs), strengthening late.
- **K0 vs K5 (RQ-i):** still preliminary — PTO LA5 only 4 iters, GRPO LA5 base only; no significant
  K0-vs-K5 difference yet. **Both LA5 arms paused for cost.** See the `project-pto-la0-eval-results`
  memory.

## Diagnostic: partial-conversation oracle (reward-faithfulness)

> The original Exp2 version (`Partial_Conv_Oracle_EDA.ipynb` + its `pto_Exp2` cache) was **removed
> 2026-06-15**. The diagnostic now lives, **rebuilt on Exp3 data with no new oracle calls**, in
> [3_Reward_Reliability.ipynb](eda/3_Reward_Reliability.ipynb) (from the per-branch `prefix` in
> `generations.jsonl`). The Exp2 finding below is kept as the original motivation for the MCL knob.

**Question.** PTO and GRPO_Exp3 score *partial* conversations (slices as short
as 2 turns) as their training reward, but the thesis evaluates on *final*
full conversations. Is the partial reward a faithful proxy?

**Method (Exp2, historical).** Pick `Base` + best `L5_Q1Q2_V*`. Slice each of their 96 convs at
every other patient turn, score every cut with Q1+Q2, compare against the
existing final-conv Q1+Q2 scores. (The Exp2 cut cache under `data/pto_Exp2/` is gone; the Exp3
rebuild reuses cuts already in `generations.jsonl`.)

**Headline.** Pairwise rank agreement (sign-of-difference vs final) is
- only **0.66 (Base) / 0.73 (L5_V10)** at `n_turns=2` — barely above chance (0.5),
- clears **0.8 at n_turns ≈ 10**, **0.9 at n_turns ≈ 30**, monotonically increasing,
- oracle reproducibility noise is ~0.07–0.10 mean |Δ|, so the gap is **structural, not noise**.

**Implication.** Short training cuts can't observe whether the therapist
delivered on Q1/Q2 by session end, so the oracle scores them on "did the
opening look promising?". Optimising that proxy biases the model toward
strong-looking openings regardless of follow-through.

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
4. **Score + EDA.** Add a `oracle_scoring/config.py::EXPERIMENTS` entry for the run (Run_Eval scoring only), run [eda/Run_Eval.ipynb](eda/Run_Eval.ipynb) (resume-safe) → then open [eda/1_Outcomes.ipynb](eda/1_Outcomes.ipynb) (and `2`–`6`), which **auto-discover** the run (no further registry edits). See "New EDA workflow".

## Running PTO_Exp3

1. **Configure.** [code/PTO_Exp3/train_PTO_Iterative.ipynb](code/PTO_Exp3/train_PTO_Iterative.ipynb) cell 1 = flat globals. Key extra knobs vs GRPO: `PREF_TREE_MODE` (`greedy`|`independent`), `NUM_BRANCHES_PER_TURN`, `PREF_FILTER_TAU`, `BRANCH_SAMPLE_TEMPERATURE`, `DPO_BETA`, `DPO_LOSS_TYPE`. `greedy` mode requires an EVEN `MIN_CONV_LENGTH` (so the sliced prefix ends on a patient turn) and slices its trunk seeds from the step-1 convs (no separate prefix-generation pass).
2. **Train.** Same visible-orchestration pattern. Outputs land under `data/pto_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`. Each iteration also saves the constructed pref pairs to `iteration_N/pref_pairs/pairs.csv` (audit trail; the prompt + chosen + rejected + scores per pair).
3. **Inspect + Score + EDA.** Same as GRPO_Exp3 (the TB dashboard is shared via `_shared/tb_plots.py`).

## Step-2 (pref-build) resume — automatic (landed 2026-06-07)

Step 2 ("Building pref pairs") is the dominant PTO phase (~41 min at K=0, hours at K=5) and
now **resumes automatically**, mirroring Step 1's per-CSV conversation resume — because
`resolve_start_state` only treats an iteration as done once `iteration_N/adapter/` exists, so
a crash *after* Step 2 but *before* the adapter (e.g. the DPO OOM) used to re-run the whole
build. Two levels, both in [pto_trainer.py](code/PTO_Exp3/pto_trainer.py):
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
  ([model.py](code/_shared/model.py) `resolve_start_state` → `trainer.train(resume_from_checkpoint=…)`)
  reads only the dir-name step + the three required files (`adapter_model.safetensors`,
  `adapter_config.json`, `trainer_state.json`) — all present in a step checkpoint. Step accounting is
  unchanged (`step_delta = global_step − resumed_steps`; the in-progress checkpoint's steps are already
  in the startup offset → no double-count).
- **Hardened resume (walk-back).** Frequent saves raise the odds a crash lands mid-write. New
  `get_latest_valid_hf_checkpoint(training_dir)` ([model.py](code/_shared/model.py), exported) walks
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

The per-generation EDA buffer ([eda_recorder.py](code/_shared/eda_recorder.py)) is flushed once at
iteration end, and HF resume **fast-forwards skipped steps without re-invoking the reward fn** — so a
mid-iteration-resumed GRPO iter's `eda/generations.jsonl` used to drop the pre-crash candidates. Fix:
`CheckpointMetadataCallback` ([tb_plots.py](code/_shared/tb_plots.py)) now takes an optional
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

**Knob.** `LOOKAHEAD_SUB_BATCH_SIZE` (notebook cell 1 → `LookaheadConfig.lookahead_sub_batch_size`;
cell 1 now sets **64 (GRPO) / 128 (PTO)** on A100-80GB — see "Runtime tuning for Colab throughput";
`None` = all active sims in one call). Halved automatically on OOM (kept sticky for the rest of the rollout).

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

## Per-generation EDA capture + live TensorBoard (landed 2026-06-05)

**EDA capture.** Each iteration writes
`runs/<MODE_TAG>/<EXP_NAME>/iteration_N/eda/generations.jsonl` with **every** candidate the
policy generated (previously PTO kept only the final (chosen,rejected) pair; GRPO kept nothing
per-prompt). Owned by [_shared/eda_recorder.py](code/_shared/eda_recorder.py) (`EDARecorder`:
in-memory buffer, one atomic flush/iteration — Drive-FUSE-friendly). **Branch-centric schema —
one JSON row per branch:**
- `prefix` (oracle-format transcript of the conv-so-far, stored ONCE), `candidates:[…]` nested
  (each: `completion`, `score`, per-questionnaire `sub_scores`, `oracle{success,retries}`,
  `lookahead{k,realized_turns,ended_early,tail}`), `chosen_idx` (= argmax score).
- `lookahead.tail` = the K simulated turns only (prefix+completion sliced off — exact, since
  look-ahead concatenates). Reconstruct a candidate's oracle-scored text =
  `prefix + "\n\n[THERAPIST]: " + completion + (tail or "")`.
- **GRPO:** one branch row per group **per epoch** (rows carry `epoch` + `group_mean/group_std`);
  recorded in the reward fn ([reward.py](code/_shared/reward.py) `_record_grpo_generations`,
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
get [_shared/tb_plots.py](code/_shared/tb_plots.py) `RunTBLogger`'s one continuous `tb_live/`
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
- **New PTO lever — `GREEDY_TRUNK_TARGET_LEN`** ([pto_trainer.py](code/PTO_Exp3/pto_trainer.py)
  `PTOConfig.greedy_trunk_target_len`, wired from cell 1): caps greedy trunk growth via
  `target_len = min(NUM_UTTERANCES_FOR_DATA, GREEDY_TRUNK_TARGET_LEN)`. **Defaults to
  `NUM_UTTERANCES_FOR_DATA` = no-op.** Lower it (e.g. 30 ≈ the partial-oracle EDA's 0.9
  rank-agreement point) to grow shorter trunks → far fewer branching depths → the biggest
  remaining PTO Step-2 speedup. It's a **science change** (shallower trunks/look-ahead
  context) and is **NOT in `EXPERIMENT_NAME`**, so isolate a lowered run by clearing/renaming
  its output dir.
- **GRPO warmup-calc fix** ([_build_grpo_args](code/GRPO_Exp3/grpo_trainer.py)): now divides
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
  steps.** **Fix:** `build_truncated_training_prompt` ([convs.py](code/_shared/convs.py)) caps the DPO
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
  [_shared/convs.py](code/_shared/convs.py)) — generation halts the moment a fake turn opens.
- New `_shared/convs.py::clean_completion` cuts at the FIRST marker; used at every decode site
  (`generate_therapist_responses_batch`, [reward.py](code/_shared/reward.py) look-ahead hot+legacy,
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
(The old `data/pto_Exp2` real local dir was **removed 2026-06-15** — Exp2 reference dropped.)

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

**New analysis EDA (`eda_analysis/`)** needs **no registry edits** — it auto-discovers arms from disk. Extend
it in the 9 modules: a new rubric → `eda_analysis/__init__.py::QUESTIONNAIRES` + `data.py` (the scores
backbone); a new arm naming scheme → `data.py::parse_experiment_name`; new stats → `stats.py`; new figures →
`plotting.py`; a new VIEW or results-layout change → `config.py` (the `view`/`_VIEW_KS` logic) + `exports.py`.
(The old submodule names `discovery`/`personas`/`scores`/`select`/`figures`/`plots` are aliased to
`data`/`plotting`, so existing references still resolve.) The bullets below apply to the **old
`oracle_scoring/` package**, which now only powers `Run_Eval.ipynb` (scoring):

- **`config.ORACLE_TOKEN_ALIASES`** — add new oracle-name aliases here (CSQ vs CSQ_8 etc.). `data._normalize_oracle_token(strict=True)` raises on unknowns; default `strict=False` lets unknowns fall through to "Other" for backward compat.
- **`config.COMPOSITE_METRICS`** — add new composites (mean across multiple source columns) here. Currently holds just `Q1Q2_Mean`; the same pattern can produce `MITI_GlobalMean` etc.
- **`config.EXPERIMENTS`** — registry of trained-model data locations. Add new entries as runs land in `data/grpo_Exp3/conversations/...` or `data/pto_Exp3/conversations/...`.

## Gotchas

- **HF model-card READMEs** inside `data/grpo_Exp3/runs/.../checkpoint-*/` are auto-generated — DO NOT delete or treat as project docs.
- **Pref-tree audit trail = resume marker.** PTO_Exp3 writes `iteration_N/pref_pairs/pairs.csv` per iter. Don't delete — it's both the DPO debug trail AND the Step-2 completion marker: its presence makes a restart **reload it and skip the ~41-min build** (see "Step-2 (pref-build) resume"). The sibling `iteration_N/pref_pairs/_progress.json` is the in-build per-step checkpoint (auto-deleted on success; safe to delete manually to force a clean rebuild).
- **Per-generation EDA.** `iteration_N/eda/generations.jsonl` (one row per branch, candidates nested — see "Per-generation EDA capture") is separate from `pref_pairs/pairs.csv` (the PTO DPO audit trail). Off-switch: `SAVE_EDA_GENERATIONS=False`. The continuous live-TB run lives at `runs/.../tb_live/` (sibling of `iteration_N/`).
- **An archived 23 MB K=3 PTO_Exp3 smoke-test** from the V4 era lives in `../archive/pto_v2_smoke/`. Ignore for new work.
- **Local sm_120 import order: `trl` must be imported BEFORE `torch`.** On the local Blackwell GPU, `from trl import …` *after* torch is already imported **segfaults at CUDA init** (a native init-order conflict, exit 139 — not OOM, not a bug in the trainers; Colab is unaffected, which is why the full runs ran there). The trainer modules already import `trl` first; only matters if you run something locally that imports torch/`_shared` first. Verified 2026-06-07.
- **Local offline smoke:** [code/_local_smoke.py](code/_local_smoke.py) — `python _local_smoke.py {stopgen|dpo|grpo|all}`. Tiny, no OpenAI; validates the stop-string bind, the DPO prompt-cap + no-OOM (grad-ckpt+precompute), and a GRPO step on the local GPU (~3 GB peak). Imports `trl` first (see above). All three PASS as of 2026-06-07.
- **Oracle prompt caching depends on the rubric-first layout.** [questionnaires.py](code/questionnaires.py) `get_prompt_eval_questionnaire` puts the fixed instructions + questionnaire rubric FIRST and the variable transcript LAST, so OpenAI's automatic prompt caching hits the ~1,084-token fixed prefix on every oracle call (≈50 % input discount + lower latency — the run is **API-bound**, so this matters). The margin over OpenAI's 1,024-token minimum is thin: **don't trim the oracle instructions/rubric or move the transcript ahead of them**, or caching silently stops (verified 2026-06-07: prefix is transcript-independent for Q1). Patient API calls auto-cache too (stable system + growing-history prefix). The therapist's local `model.generate` has **no** cross-call prefix reuse under HF — that would need vLLM (a real build here, not a flag: the look-ahead and *all* of PTO's generation use custom `model.generate`, not TRL's `use_vllm` path).
- **The run is likely GPU-bound, not API-bound (corrected 2026-06-07).** Earlier notes called the runs "API-bound" — that was inferred from GPU *memory* (17/67 GB), which does NOT measure compute. Lior observes he waits on GPU, not API. Autoregressive `model.generate` on the 1B LoRA policy (GRPO's G=8 completion sampling + K-turn look-ahead; PTO's branch sampling + look-ahead) dominates wall-clock; the `340.6 s / 8 GPU calls` look-ahead line ≈ 30–40 s per batched generate, far above the ~1–2 s of raw 1B/A100 compute → heavy per-step overhead. **Top suspect: the recently-added `STOP_STRINGS` route generation through HF `StopStringCriteria` (runs every step; known multi-× slowdown).** Before optimizing, MEASURE the split (time sampling vs look-ahead-GPU vs look-ahead-API vs backward); the K=0 arms (no look-ahead) running much faster would itself confirm generation is the cost. Faster stop than string-matching: register the two markers as single special tokens + stop on `eos_token_id`.
