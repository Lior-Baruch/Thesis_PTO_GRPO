# Thesis — Looking Ahead in Goal-Oriented Dialogue: Comparing Preference-Tree and Group-Relative Optimization of Small Language Models for Motivational Interviewing

## What this is
Master's thesis (Lior Baruch, Reichman). Trains small therapist LLMs to do
Motivational Interviewing against simulated patients; reward = larger
"oracle" LLM grading validated MI questionnaires.

Three controlled comparisons, all live in Exp3:
1. **Look-ahead depth** K ∈ {0, 5} — does anticipating future turns help, and by how much?
2. **PTO vs GRPO** under matched K + MCL — does iterative GRPO compete with PTO?
3. **Oracle questionnaire** (Q1+Q2 vs WAI-SR vs CSQ-8 vs MI-SAT/MITI) — held for later.

> **This file is also the Exp3 context file.** As of 2026-07-29 there is no
> `Exp3_PTO_GRPO/CLAUDE.md` — its content lives in "Exp3_PTO_GRPO — the active experiment"
> below, so the active experiment is always in context and there is exactly one file to update.
> Exp1 and Exp2 keep their own `CLAUDE.md` because they are frozen/complete.

## Experiments (chronological)
| | [Exp1_ICLR2025/](Exp1_ICLR2025/) | [Exp2_PTO/](Exp2_PTO/) | [Exp3_PTO_GRPO/](Exp3_PTO_GRPO/) |
|---|---|---|---|
| **Status** | Frozen — published | Complete — EDA verified | **Active — main thesis chapter; see "Current status & next step" below** |
| **Therapist** | Llama-2-7B | Llama-3.2-1B (4-bit NF4) | Llama-3.2-1B (bf16) |
| **Patient + oracle** | GPT-3.5 | gpt-4o-mini-2024-07-18 | gpt-4o-mini-2024-07-18 |
| **Patient prompts** | V1 (cooperative) | V3 (less cooperative) | V3 |
| **Oracle output** | V1 (regex; Q1+Q2 only) | V5 (JSON schema; 6 questionnaires) | V5 + PCT/MITI-style coders → **8 instruments** |
| **PTO** | K ∈ {0, 5}, 7 iters | 4 oracles × K ∈ {0, 5} | **PTO_Exp3** (iterative; lean sibling of GRPO_Exp3, controlled hyperparams matched) |
| **GRPO** | — | V1 (static prompts, weak baseline) | **GRPO_Exp3** (iterative) — both methods share `code/_shared/` |
| **MCL filter** | — | — | **Wired in both PTO_Exp3 and GRPO_Exp3.** Encoded in `EXPERIMENT_NAME`. |
| **Training reward** | mean(Q1, Q2) | chosen oracle | Q1+Q2 only (matches Exp1) |
| **Eval reward** | Q1, Q2 | per-oracle | **all 8 rubrics** — the 6 questionnaires + `PCT` + `MICI` (added 2026-06-14) |
| **EDA shape** | `Conv_EDA.ipynb` | + per-Q CSVs, `pref_emb/` | `eda_analysis/` package (analysis top level + `scoring/` subpackage backing `Run_Eval`) + tier-based notebooks `1_Outcomes`–`7_Stats` (+ `0_headline/` family; final-vs-best endpoint pairs); per-generation `iteration_N/eda/generations.jsonl` |
| **Convs / models** | (paper figures) | 4,512 / 47 | 2,880 / 30 scored on both graders (PTO+GRPO LA0 to iter 10 + partial LA5: **PTO I1–5**, GRPO I1) |

Dirs renamed 2026-05-12 from `ICLR2025/`/`Extension/`/`NewExperiment/`.

## Data lineage
- **Exp1 → Exp2:** independent re-implementation. Stronger oracle, harder patients, JSON-schema rubric, more questionnaires. No data flow.
- **Exp2 → Exp3:** independent re-implementation — **Exp3 is a complete, fresh experiment that shares no data with Exp2** (both PTO_Exp3 and GRPO_Exp3 generate all their own convs from scratch each iteration; see the Exp3 self-loop below).
  - ⚠ **Exp2 and Exp3 absolute oracle scores are NOT on the same axis.** Same therapist base (Llama-3.2-1B), but Exp2 generated its convs in **4-bit NF4** and Exp3 in **bf16**. 4-bit induces ~30× more phrase-loop degeneration (≈9.5% vs 0.3% of therapist turns run to the token cap as repeated spam), which the oracle floors — so Exp2 Base ≈ 2.38 Q1+Q2 vs Exp3 Base ≈ 3.0, *even though it's the same model*. The clean (non-degenerate) Exp2 subset scores ≈ 2.93 ≈ Exp3. **Compare within Exp3 only**; to put Exp2 on the same axis, regenerate its convs in bf16.
- **Exp3 self-loop:** GRPO_Exp3 regenerates its own training data each iter from the current policy; those same convs are the eval set (no separate generate-eval step for trained iters).

## Key methodological shift across experiments
- **Look-ahead K** stayed central throughout (the lever from the ICLR paper).
- **The hard part moved from "can PTO beat the baseline?" (Exp1, settled) to "is GRPO competitive with PTO under matched look-ahead?" (Exp3, open).**
- **Exp3 also exposed a reward-faithfulness concern** the earlier experiments never tested: the partial-conversation oracle diagnostic (originally `Partial_Conv_Oracle_EDA` on Exp2 data; now rebuilt on Exp3 data in [Exp3_PTO_GRPO/eda/notebooks/analysis/5_Training.ipynb](Exp3_PTO_GRPO/eda/notebooks/analysis/5_Training.ipynb)) shows that the short-cut training reward has only ~0.66–0.73 rank agreement with the full-conv eval at `n_turns=2`. Motivates the `MIN_CONV_LENGTH` knob — now wired in both GRPO_Exp3 (slice filter) and PTO_Exp3 (greedy: tree-start prefix length; independent: branch-point filter); encoded in `EXPERIMENT_NAME` so MCL sweeps stay in disjoint folders.

## Methods (one line each)
- **PTO V1** (Exp1) = original preference-tree exploration + K look-ahead + DPO. Published.
- **GRPO V1** (Exp2) = static prompt set, weak baseline.
- **GRPO_Exp3** = current policy simulates 96 convs → per-turn prompts (MCL filter) → GRPO update with optional K-turn look-ahead. Convs double as eval.
- **PTO_Exp3** = per-turn branching (`M` candidates) → K-turn look-ahead + oracle → τ-filtered (chosen, rejected) pref pairs → DPO update. Lean sibling of GRPO_Exp3. **Two `PREF_TREE_MODE`s:** `greedy` (default, true PTO — start from an MCL-length prefix sliced off the step-1 conv and grow ONE trunk by appending the best-of-M completion at each therapist turn, so the choice feeds the next branch point) and `independent` (branch each patient turn of a pre-recorded conv, no feedback). Mode baked into `EXPERIMENT_NAME`.

**Naming:** PTO is the framework, DPO is the loss. Don't call GRPO data "pref data" — it has none.

## Current status & next step
**THE single live copy of run status + headline numbers + cost constraint** (all other docs point
here — see "Doc map"). Updated 2026-07-29.

- **Run status:** PTO LA0 = 10 iters scored; **GRPO LA0 = 10 iters (FINISHED, re-scored)**. **Both
  LA5 arms PAUSED/thin** (PTO LA5: I1–I4 scored + an unscored iter-5 adapter whose eval convs were
  never generated; GRPO LA5: I1 trained AND fully scored).
- **Headline:** **PTO wins at the matched 10-iter endpoint (Q1+Q2 4.26 vs 3.75; paired +0.51,
  dz 0.73)** because GRPO peaks at iter 8 (4.08) then regresses into sycophancy (MICI endpoint 0.84
  vs PTO 0.49); PTO climbs stably. Full narrative + tables:
  [Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md) (L0 = primary read).
- **Judge validity (2026-07-26, extended 07-27/28):** the measurement instrument is now measured,
  not assumed — oracle **ICC(2,1) 0.86–0.99** (mean |Δ| 0.04–0.09, confirming the "≈0.10 noise"
  folklore; Q1/Q2 hold 0.96–0.99 and only MICI dips below 0.90 — floor is MICI PTO@10 at 0.864),
  and a decoupled second judge (**Claude Haiku 4.5**, different family, never played the patient)
  reproduces every endpoint contrast with the same sign (**18/18** after the full enumeration; it
  *widens* the PTO−GRPO Q1 gap to +0.77 vs the primary's +0.53). Q1/Q2 cross-judge r 0.80–0.88 vs a
  measured 0.96–0.98 ceiling; MICI agrees weakly (r 0.20–0.55) so the sycophancy claim holds at the
  contrast level, not as a precise rate. Buys down LIMITATIONS §1–§2. Cost ~$5.30.
  See `eda/notebooks/analysis/8_Measurement_Validity` §1.
- **Cost constraint:** OpenAI spend hit **~$300** and is binding — RQ-i (K0 vs K5) on hold. Cost is
  dominated by oracle scoring + (at K=5) look-ahead patient calls, both ∝ candidate count
  (`prompts×G` / `branch-points×M`) × iterations; prompt caching is already maxed (~50% off the
  oracle's fixed prefix), so the only lever is call **COUNT**: cap `NUM_ITERATIONS` ~5–6 (curves
  plateau by iter ~4), drop `M`/`G` 8→4, (PTO) lower `GREEDY_TRUNK_TARGET_LEN` — keep **K** (the
  RQ-i variable) and the **gpt-4o-mini oracle** (the measurement instrument) fixed. See the
  `project-openai-cost-constraint` memory.
- **RQ-i first matched point — DONE 2026-07-30.** `PTOExp3_LA5_I5` is generated, scored on BOTH
  graders (23,040 Haiku cells now; parity kept), folded, and rendered. **PTO LA5 = iters 0–5.**
  - **The read: look-ahead buys no Q1+Q2 advantage at equal iteration count.** K=5 trails K=0 by
    0.08–0.16 through iters 1–4, then ties at iter 5 (**4.017 vs 4.014**) — both at the ~4.01
    plateau K=0 passes on its way to 4.26 by iter 10. Since K=5 costs materially more per iteration,
    that is a substantive negative result. Unpaired means, no matched endpoint, one crossover point
    — directional only. Numbers + caveats:
    [L5/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L5/SUMMARY.md) §3.
  - ⚠ **WITHIN PTO only** — GRPO LA5 still has just iter 1, so this is not a K×method comparison.
  - ⚠ **RQ-i still has no tracked artifact.** The contrast needs both K arms in one frame, which
    neither `L0` nor `L5` provides; it lives only in the retired `all` view. Promote `all` back or
    move `k_paired_by_method` into `L5`.
  - Tooling = [`code/PTO_Exp3/generate_eval_convs.{py,ipynb}`](Exp3_PTO_GRPO/code/PTO_Exp3/generate_eval_convs.py)
    (repairs any orphaned adapter). Mechanics + the VRAM leak it exposed:
    [CHANGELOG_TRAINER.md](Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md) (2026-07-30 entry).
  - **Next:** resume PTO LA5 → iter 10 and GRPO LA5 from iter 1 when budget allows. Cost: $1.33
    (Haiku batched) for the iter-5 sweep; `judge_plan.estimate_cost` prices the rest.
  - **Durable LA5-resume facts** (what's actually on Drive; dated forensics in
    [CHANGELOG_EDA.md](Exp3_PTO_GRPO/history/CHANGELOG_EDA.md), 2026-07-11 entry):
    **PTO LA5** has trained adapters for iters 1–5 but only I1–I4 scored — the iter-5 eval convs
    were never generated by the run itself (`iteration_6/` died ~1 min in: adapter saved 02:32, iter-6
    dirs created 02:33). **GRPO LA5**: iter-1 adapter trained AND scored; its `iteration_2/` is
    adapter-less. **Folder presence ≠ data.**
  - **Cleared 2026-07-30:** `iteration_6/pref_pairs/pairs.csv` (1 byte) + `eda/generations.jsonl`
    (0 bytes) were deleted. `pairs.csv` is the Step-2 **completion marker**, so an empty one would
    have made a resumed iter 6 reload 0 pairs, skip the ~41-min build, and run a silent **no-op DPO
    update**. Check for empty markers before resuming any arm.
- **SECOND JUDGE IS NOW CO-PRIMARY — full sweep COMPLETE 2026-07-27.** Claude Haiku 4.5 has scored
  **22,272 / 22,272** cells (29 model states × 8 rubrics × 96 convs; 232/232 cells at full n=96),
  matching the primary oracle's grid exactly. Cost **$42** via Message Batches (50% off; measured
  3,621 input + 71 output tokens/call). The whole EDA now runs under either grader:
  `python tools/render_views.py --judge anthropic_claude-haiku-4-5` → artifacts nest at
  `results/<view>/figures/<family>/<judge>/` — **every** grader nests under its short label
  (`gpt-4o-mini/`, `claude-haiku-4-5/`) since 2026-07-28; the primary is no longer flat, so a figure
  path always names the grader that produced it. Notebooks 5+6 **refuse** a
  second judge — they read the training side, which cannot be re-graded after the fact.
- **Multi-judge EDA — BUILT 2026-07-27** (was queued 2026-07-26). Lands as
  `8_Measurement_Validity` (free, inside `tools/render_views.py`; family `8_measurement/`,
  no `<judge>/` level because every artifact contains both graders) + `Judge_Reliability.ipynb`
  **§3** (the paid full sweep). The four results that carry weight — all
  on the tracked `L0` view (22 arms), **numbers owned by
  [L0/SUMMARY.md](Exp3_PTO_GRPO/eda/results/L0/SUMMARY.md) §7, caveats by
  [LIMITATIONS.md](Exp3_PTO_GRPO/eda/docs/LIMITATIONS.md) §1–§3:**
  - **Sign preservation.** 18/18 on the thesis-critical anchor contrasts; **88.3%** across all
    1,848 arm×metric contrasts, rising to **98.9%** at |Δ|≥0.50 — the judges disagree only about
    differences too small to claim.
  - **Variance decomposition.** Only **1.2–6.9%** of arm-mean variance is arm×judge: they disagree
    about *level*, not about arm *ordering*.
  - ⚠ **MITI is the exception** — dependability 0.65 off one judge, and the weakest sign
    preservation (77.5%). **Treat MITI arm differences as provisional.** This is a thesis
    limitation, not a footnote.
  - **Gain retention is the reward-hacking test.** Q1 retention PTO@10 **0.80** vs GRPO@10
    **0.28**, non-overlapping, and per-iteration it is an *onset curve* — GRPO decays from ~0.89
    (I3) to 0.28 (I10) while PTO holds 0.80–0.98. Stronger evidence for sycophancy than the MICI
    rate, and it buys down LIMITATIONS §3 (circularity).
  - **Cost, measured**: 3,621 input + 71 output tokens/call (`judge_batch.probe_usage`); full sweep
    **$42 batched / $84 direct**; the free char-based estimator lands within 12%. Parity gate 8/8.
    Deliberately **1 rep, not 3** — oracle noise adds ≈0.01 to a 96-conv arm mean vs ≈0.09 from
    persona sampling, so breadth beats depth at equal cost.
    ⚠ **Haiku 4.5 caches nothing on this prompt** — confirmed empirically
    (`cached_input_tokens = 0`): its cacheable-prefix minimum is 4,096 and only Q1/Q2 come close.
- **Second-judge ICC — MEASURED 2026-07-28**, closing the last named validity gap (was the
  "cheapest remaining validity buy"). 2 further Haiku reps on the anchor subset, 2,304 calls,
  0 errors. Haiku's own ICC: **Q1 0.951–0.978, Q2 0.938–0.963, MICI 0.525–0.929** — near-parity on
  Q1/Q2, but its MICI repeatability *falls as the MI-inconsistency rate rises* (GRPO@10 0.525), so
  it is least reliable exactly where the sycophancy claim lives. Against the corrected ceiling,
  agreement recovers Q1 86–91% / Q2 83–88% but MICI only **29–59%**: partly the judge's noise,
  mostly construct disagreement. **No headline result moves** — the MICI caveat stands and gain
  retention remains the load-bearing evidence. Tables + the full argument: LIMITATIONS §1–§2.
  - `reliability.agreement` computes `sqrt(ICC_primary × ICC_judge)` from measured values and records
    `ceiling_basis`; it falls back to the assumption only where a judge has <2 reps.
  - **Cost $9.16** (batched would have been $4.58) against "~$1–2" previously documented here — that
    was an unchecked estimate. Price judge spend with `judge_plan.estimate_cost`.
- **ONE SCORE LAKE — 2026-07-28.** Every grader's scores now live in a single judge-partitioned tree,
  `data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<id>.csv`, replacing four
  stores under two schemes (the primary's reported draw was split per method with no `judge=`/`rep=`
  level; every other grader sat in a separate local-only tree that had both). `judge` is an ordinary
  partition key now, `rep=0` is each judge's full-grid draw, and there is one resolver instead of a
  primary-vs-other branch. 50,320 files copied, hash-verified, then removed at source; **no headline
  number moves** (45/45 endpoint cells and 25,056 rows identical). Two consequences worth knowing:
  - **The lake is a Drive symlink**, so the second judge's $42 sweep and the $9.16 ICC reps are
    backed up for the first time — previously they existed only on one laptop, gitignored.
  - **The primary's ICC now spans 4 draws** (the reported one included, as the second judge's
    already did), which is why the range above reads 0.86–0.99 rather than the older 0.90–0.99.
    Only MICI moves; Q1/Q2 shift ≤0.007.
- **Reproducible figures + the fold as a read path — 2026-07-28**, both found while proving the
  score-lake migration changed nothing.
  - **Seaborn's bootstrap CIs were unseeded**, so re-rendering rewrote 90 PNGs on unchanged data
    (three consecutive renders each differed by ~6% of pixels). `BOOT_SEED = 12345` was promoted
    from a private `stats._BOOT_SEED` to `constants` and passed at all seven
    `errorbar=("ci", 95)` callsites — the figure side and the table side now share one seed, and a
    thesis figure is reproducible rather than merely stable-looking.
  - **The parquet fold is now a read path, not archival-only.** `eda_analysis/score_archive.py`
    owns the layout, the staleness guard and the read; `tools/consolidate_scores.py` is a thin CLI
    over it. `iter_conv_rows` serves from the fold when the per-partition content signature in
    `_parquet/_manifest.json` still matches disk and falls back to the CSVs otherwise —
    **4.3–6.1×** faster (`scores_long` 86 s → 16 s), with all seven per-conversation loaders proven
    identical under `assert_frame_equal(rtol=0, atol=0)` via either path. `_selfcheck` is now
    **14 checks** and asserts both halves (fold-equals-CSV, and that a tampered signature is
    refused rather than served).

## Doc map (one owner per fact)
| Fact | Lives ONLY in |
|---|---|
| Run status + headline numbers + cost constraint | this file → "Current status & next step" |
| Method mechanics, trainer internals, gotchas | this file → "Exp3_PTO_GRPO — the active experiment" |
| Detailed eval narrative + numbers | `Exp3_PTO_GRPO/eda/results/<view>/SUMMARY.md` |
| EDA how-to (VIEW + JUDGE knobs, `EdaConfig`, package module map) | `Exp3_PTO_GRPO/eda/README.md` |
| Metric definitions (no current values) | `Exp3_PTO_GRPO/eda/docs/METRICS_REFERENCE.md` |
| Measurement / inference limitations (for the write-up) | `Exp3_PTO_GRPO/eda/docs/LIMITATIONS.md` |
| Dated history | `Exp3_PTO_GRPO/history/` — [CHANGELOG_EDA.md](Exp3_PTO_GRPO/history/CHANGELOG_EDA.md) + [CHANGELOG_TRAINER.md](Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md) behind a stable [index](Exp3_PTO_GRPO/history/CHANGELOG.md). There is no root changelog. |
| Supervisor decks + emails | `Exp3_PTO_GRPO/meetings/README.md` |
| Data/artifact policy (what's gitignored, how it regenerates) | `README.md` § "Data & large artifacts" |

Update a fact in its owner file only; everywhere else keep a pointer.

## Layout
```
Thesis_PTO_GRPO/
├── CLAUDE.md                   (this file — cross-experiment map + the full Exp3 context)
├── README.md, LICENSE          README also owns the data/artifact policy (no DATA_README)
├── Exp{1,2}_*/CLAUDE.md        per-experiment context for the FROZEN experiments only
├── Exp3_PTO_GRPO/history/      the only dated history: CHANGELOG_{EDA,TRAINER}.md + an index
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

---

# Exp3_PTO_GRPO — the active experiment

Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle. Two methods compared under matched
look-ahead + oracle. **Hyperparameters matched across the two:** `NUM_ITERATIONS=10`, `MCL=12`,
K ∈ {0,5}, gen temps + API concurrency; PTO's `M` (`NUM_BRANCHES_PER_TURN`)=8 mirrors GRPO's
`NUM_GENERATIONS`; `DPO_BETA`=0.1 is the DPO loss temperature, **not** GRPO's KL β. bf16
(`USE_4BIT` toggle). Output dirs `data/pto_Exp3/` and `data/grpo_Exp3/`.

Reward (training) = **Q1 + Q2 only**, matching the ICLR look-ahead paper.
Reward (eval) = **all 8 instruments** — the 6 MI questionnaires (Q1, Q2, WAI-SR, CSQ-8, MI-SAT,
MITI) plus `PCT` (patient change-talk) and `MICI` (MI-inconsistent behaviour, lower = better).

**Shared infrastructure.** Both trainers import from
[Exp3_PTO_GRPO/code/_shared/](Exp3_PTO_GRPO/code/_shared/) (runtime, model, convs, reward,
tb_plots, eda_recorder; + optional lookahead_check). Each method's trainer module
(`grpo_trainer.py` / `pto_trainer.py` — named per method so `from <method>_trainer import …` can't
collide in a shared kernel) owns just the method-specific bits (`TrainingConfig`/`PTOConfig`,
iteration body, dataset shape, TRL trainer wrapping).

**Change history** (the dated "pass"/"Landed" entries — both the EDA passes and the trainer /
infrastructure narratives) lives in
[Exp3_PTO_GRPO/history/CHANGELOG.md](Exp3_PTO_GRPO/history/CHANGELOG.md). The current state is the
sections below.

**Single canonical copies.** `system_prompts_builder.py` and `questionnaires.py` live ONLY at
`code/` root — `eda/eda_analysis/constants.py` (the package's leaf, imported by everything incl.
`scoring/`) prepends `code/` to `sys.path` so they import the same canonical files. No drift.

## Exp3 · Trainer pattern

Both trainers (`code/GRPO_Exp3/`, `code/PTO_Exp3/`) follow the same shape:

```
<METHOD>_Exp3/
├── train_<METHOD>_Iterative.ipynb   thicker — per-iteration orchestration visible
└── <method>_trainer.py              <Method>Config + run_one_iteration + run_final_eval + write_run_metadata + build_wandb_ctx
                                     (named per method — grpo_trainer.py / pto_trainer.py — so `from <m>_trainer` can't collide in a shared kernel)
```

with the per-iteration loop composed *visibly in the notebook* (no black-box
`run_iterative_training` call). Helpers shared across both methods live in
[Exp3_PTO_GRPO/code/_shared/](Exp3_PTO_GRPO/code/_shared/).

## Exp3 · Algorithms (PTO + look-ahead, GRPO + look-ahead)

Both methods are **iterative**: each iteration regenerates training data from the *current* policy,
performs an update, swaps the adapter, and repeats. They share the conversation-simulation +
oracle-scoring + K-turn look-ahead machinery (in `code/_shared/`) and diverge only in (a) how they
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
don't trample each other. See [_shared/reward.py](Exp3_PTO_GRPO/code/_shared/reward.py).

### GRPO_Exp3 + K-turn look-ahead

**Per iteration `n`** (loop body in
[train_GRPO_Iterative.ipynb](Exp3_PTO_GRPO/code/GRPO_Exp3/train_GRPO_Iterative.ipynb), helpers in
[grpo_trainer.py](Exp3_PTO_GRPO/code/GRPO_Exp3/grpo_trainer.py)):

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

**Per iteration `n`** (loop body in
[train_PTO_Iterative.ipynb](Exp3_PTO_GRPO/code/PTO_Exp3/train_PTO_Iterative.ipynb), helpers in
[pto_trainer.py](Exp3_PTO_GRPO/code/PTO_Exp3/pto_trainer.py)):

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

## Exp3 · Layout

```
Exp3_PTO_GRPO/
├── code/
│   ├── system_prompts_builder.py        V3 prompts (single canonical copy; EDA also reads this one)
│   ├── questionnaires.py                V5 oracle (JSON schema, 8 instruments incl. PCT + MICI)
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
│       ├── pto_trainer.py               PTOConfig + run_one_iteration + build_pref_pairs_for_conversation + …
│       ├── generate_eval_convs.py       GENERATE-ONLY pass for ONE model state — repairs an "orphaned
│       │                                adapter" (trained but its model_iter_N convs never generated).
│       │                                Config rebuilt from the run's OWN run_metadata.json; seeds
│       │                                DERIVED (seed+N+1); --verify-seeds proves that vs decoy offsets
│       │                                before spending; --num-convs/--num-utterances require --conv-dir.
│       └── generate_eval_convs.ipynb    thin notebook over it (Colab or local; cwd-robust bootstrap)
├── data/                               ALL THREE subdirs are Google Drive symlinks (backed up + reachable from Colab)
│   ├── eval_coverage.csv                scoring-coverage snapshot: per model × metric done/todo counts
│   ├── eval_scores/                     THE SCORE LAKE — every grader's scores, one shape (2026-07-28)
│   │   ├── judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
│   │   │                                M=scoring metric, O=training oracle. rep=0 is each judge's
│   │   │                                FULL-GRID draw (the reported one); rep>=1 are repeatability
│   │   │                                draws on the anchor subset. No method level — <Model>
│   │   │                                already carries it (GRPOExp3_* / PTOExp3_*).
│   │   ├── _parquet/judge=<tag>/rep=<r>/metric=<M>.parquet   derived fold of the CSVs (50,305 → 31)
│   │   │   + _manifest.json             built by tools/consolidate_scores.py; READ by
│   │   │                                iter_conv_rows (4.3–6.1× faster) but ONLY while the
│   │   │                                manifest's per-partition content signature still matches
│   │   │                                disk — any mismatch silently falls back to the CSVs.
│   │   ├── _batches/<tag>/rep=<r>/*.json   Message Batches manifests (submit → collect state)
│   │   └── summary/                     convenience CSV snapshots from Judge_Reliability
│   ├── grpo_Exp3/                       produced by GRPO_Exp3 runs
│   │   ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training}/
│   │   └── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
│   └── pto_Exp3/                        produced by PTO_Exp3 runs (same shape as grpo_Exp3)
│       ├── runs/<MODE_TAG>/<EXP_NAME>/   run_metadata.json + iteration_N/{adapter, training, pref_pairs/}
│       └── conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
├── eda/                                 verified runnable end-to-end (2026-07-27 reorg: notebooks/ docs/ tools/)
│   ├── README.md                        EDA guide: notebook↔family table, VIEW + JUDGE knobs, module map
│   ├── notebooks/
│   │   ├── analysis/                    FREE + reproducible — the 7 tier-based topic notebooks
│   │   │                                (`1_Outcomes` `2_Questionnaire_Detail` `3_Validity_and_Hacking`
│   │   │                                `4_Heterogeneity` `5_Training` `6_Preference` `7_Stats`
│   │   │                                `8_Measurement_Validity`) ↔ result families 1:1 (+ a `0_headline/` family of
│   │   │                                re-saved presentation figures), [EVAL]/[TRAINING]-tagged,
│   │   │                                endpoint artifacts as final+best pairs. Driven by tools/render_views.py
│   │   └── scoring/                     **$$ PAID + manual** (RUN_* switches; never in render_views):
│   │                                    `Run_Eval.ipynb` (async oracle pipeline → eval_scores/, resume-safe)
│   │                                    `Judge_Reliability.ipynb` (ICC + second judge + §3 full sweep)
│   ├── docs/
│   │   ├── LIMITATIONS.md               documented measurement/inference limitations (for the thesis write-up)
│   │   └── METRICS_REFERENCE.md         cheat-sheet for every EDA number (questionnaires, derived ratios, hack battery)
│   ├── tools/
│   │   ├── render_views.py              DRIVER: regenerate results/<view>/ via nbconvert (sets EDA_VIEW/EDA_JUDGE;
│   │   │                                --output-dir tmp; --nb takes the notebook/family NUMBER 1..7; --judge <tag>)
│   │   ├── consolidate_scores.py        {build|verify|report} the score lake's parquet fold
│   │   │                                (CLI over eda_analysis/score_archive.py). Run `build`
│   │   │                                after any new scoring — stale is safe, just slow.
│   │   └── strip_notebook_outputs.py    output-clean helper (paired with the nbstrip git clean-filter)
│   ├── eda_analysis/                    THE Exp3 EDA package (one package since the 2026-07-13 fold):
│   │                                    analysis layer (disk-discovery, read-only) = constants LEAF
│   │                                    + config / data / score_archive / plotting_style / stats /
│   │                                    behavior / training / reliability /
│   │                                    pref / exports / _selfcheck + plotting/ subpackage (topic-split
│   │                                    figures; figures+plots alias it); scoring layer = scoring/
│   │                                    subpackage (registry / conversations / pipeline / judge — the
│   │                                    Run_Eval + Judge_Reliability backend, imported explicitly, not
│   │                                    via __init__). Module map: eda/README.md § "Package".
│   ├── results/                         GENERATED thesis artifacts in 2 tracked VIEW trees: L0/ · L5/, each with figures|tables/<N_family>/<judge>/ (family number == producing-notebook number) + INDEX.md + hand-authored SUMMARY.md. EVERY grader nests under its own short label (gpt-4o-mini/ · claude-haiku-4-5/) since 2026-07-28 — the primary is no longer flat, so a path always names the grader. (The pooled all/ view was retired 2026-07-27 — renderable, gitignored, not a deliverable.)
│   ├── .eda_cache/                      parquet cache (gitignored; content-keyed on input CSVs)
│   └── .emb_cache/                      pref completion-embedding cache (gitignored; regenerable)
├── figures/                             hand-authored METHOD schematics — NOT data-derived, so they
│                                        live outside eda/results/ (no view, no <judge>/ level, no
│                                        producing notebook). build_method_figures.py draws the PTO +
│                                        GRPO framework diagrams (ICLR Fig 1 redrawn + its GRPO twin)
│                                        and the two generation diagrams (ICLR Fig 2 redrawn for greedy
│                                        mode + the GRPO group). No in-figure titles — captions live in
│                                        CAPTIONS.md, the slide title, or the LaTeX caption.
├── meetings/                            supervisor-facing output ONLY — never imported by code/ or eda/
│   ├── README.md                        which deck is which + rebuild/export commands
│   ├── build/                           build_supervisor_deck.py (full) · build_results_snapshot.py (lean)
│   │                                    · export_pdf.ps1 (pptx→pdf via PowerPoint COM). Paths resolve
│   │                                    off __file__; each script's OUT names its dated folder.
│   └── <YYYY-MM-DD>/                    one folder per meeting: deck (.pptx gitignored, .pdf tracked) + email draft
└── HF_key.txt, openai_key.txt
```

**Thesis artifacts.** `results/<view>/figures/` (`.png`) and `results/<view>/tables/` (`.md`+`.xlsx`)
are **generated** by `eda_analysis.save_fig`/`save_table` (the `formats=` kwarg can request extras for
a one-off; per-call `group=` overrides the family, incl. nested subpaths). Each notebook regenerates
its own family; `python tools/render_views.py` regenerates everything. Reproducible from code
(seeded — see `BOOT_SEED`); tracked in git.

## Exp3 · EDA workflow (short version — full guide in [eda/README.md](Exp3_PTO_GRPO/eda/README.md))
1. **Score:** `Run_Eval.ipynb` — its `EXPERIMENTS` registry is auto-generated from
   `eda_analysis.data.discover_arms()`, so a run is scoreable as soon as its conversations land on
   disk (empty in-flight `model_iter` dirs are skipped). Writes
   `data/eval_scores/judge=<tag>/rep=<r>/`.
2. **Analyze:** notebooks `1_Outcomes` … `7_Stats` (topic ↔ results family, 1:1; tier-based
   drill-down: global scores → per-questionnaire detail → validity/heterogeneity/training/stats);
   everything auto-discovers arms from disk — no registry edits anywhere. The **VIEW knob**
   (`all`/`L0`/`L5`) sets both the arm filter and the `results/<view>/` output root; the orthogonal
   **JUDGE knob** selects which grader's scores are read and which `<judge>/` subfolder is written.
3. **Regenerate:** `python tools/render_views.py` (renders the two tracked views, L0+L5) → `results/<view>/`.
   The pooled `all` view was RETIRED 2026-07-27 — still renderable, but gitignored scratch, not a deliverable.
   Run **`python -m eda_analysis._selfcheck`** after any EDA change (14 checks).

The VIEW/JUDGE systems, `EdaConfig`, parquet cache, output-clean policy, and the package module map
are all documented in [eda/README.md](Exp3_PTO_GRPO/eda/README.md) — not here. Eval **numbers** are
not maintained here either: see the Doc map.

## Exp3 · Diagnostic: partial-conversation oracle (reward-faithfulness)

Both trainers score *partial* conversations (slices as short as 2 turns) as the training reward, but
the thesis evaluates *full* conversations. The diagnostic — rebuilt on Exp3 data with no new oracle
calls in [5_Training.ipynb](Exp3_PTO_GRPO/eda/notebooks/analysis/5_Training.ipynb)
(from the per-branch `prefix` in `generations.jsonl`); the original Exp2 version motivated the MCL
knob — shows pairwise rank agreement with the final-conv score is **barely above chance at
`n_turns=2` and only clears 0.8/0.9 at ~10/~30 turns**, a structural gap well above oracle
reproducibility noise. Numbers + method:
[eda/docs/METRICS_REFERENCE.md](Exp3_PTO_GRPO/eda/docs/METRICS_REFERENCE.md) § 6.

**Implication.** Short training cuts can't observe whether the therapist delivered on Q1/Q2 by
session end, so the oracle scores them on "did the opening look promising?" — optimizing that proxy
biases the model toward strong-looking openings regardless of follow-through.

## Exp3 · MIN_CONV_LENGTH filter — wired in both trainers

Direct response to the partial-conversation reliability finding above.

- **GRPO_Exp3.** Cell 1's `MIN_CONV_LENGTH` → `TrainingConfig.min_conv_length` →
  `extract_prompts_from_conversations(min_conv_length=...)` in
  [_shared/convs.py](Exp3_PTO_GRPO/code/_shared/convs.py).
- **PTO_Exp3.** Cell 1's `MIN_CONV_LENGTH` → `PTOConfig.min_conv_length`. In `greedy`
  mode it's where the **tree starts** (prefix length, must be EVEN so the prefix ends on
  a patient turn); in `independent` mode it's the slice filter (`build_pref_pairs_for_conversation`
  skips branch points whose conv-so-far is shorter). Either way: no training context below MCL.
- **Semantics.** Drop slices/branches where the conversation-so-far has fewer than `MIN_CONV_LENGTH` total utterances (same `n_turns` unit as the partial-conv diagnostic — therapist + patient combined).
- **Default = 2** = no-op. Recommended exploratory values: `10` (EDA's 0.8 threshold), `30` (0.9 threshold).
- **Encoded in `EXPERIMENT_NAME`** as `_MCL{N}` so runs at different MCL never share an output folder.

## Exp3 · EXPERIMENT_NAME schemes

- GRPO_Exp3: `GRPO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_G{G}`
- PTO_Exp3:  `PTO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_M{NUM_BRANCHES_PER_TURN}_PT{greedy|indep}`

`{Oracle}` is the training-oracle token derived from `QUESTIONNAIRE_IDS` in cell 1
(`Q1Q2`|`WAI`|`CSQ8`|`MI_SAT`|`MITI`) — identical to the EDA `oracle=<O>` tokens, so a run's
folder/Hub name and its `eval_scores/.../oracle=<O>/` folder agree. An unmapped ID set raises.

Different sweep arms write to disjoint dirs — runs never collide.

## Exp3 · Running the trainers

**GRPO_Exp3**
1. **Configure.** [train_GRPO_Iterative.ipynb](Exp3_PTO_GRPO/code/GRPO_Exp3/train_GRPO_Iterative.ipynb) cell 1 = flat globals.
2. **Train.** Run top-to-bottom. The orchestration loop is in the notebook (cells after `cfg = TrainingConfig(...)`), composed from `run_one_iteration` / `run_final_eval` in [grpo_trainer.py](Exp3_PTO_GRPO/code/GRPO_Exp3/grpo_trainer.py). Resumes from latest completed iter via `_shared.resolve_start_state`. Outputs under `data/grpo_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`; per-run `run_metadata.json` at the run root.
3. **Inspect.** Last cell: `scan_scalar_tags` + `plot_iteration_metrics` + inline TensorBoard. `plot_iteration_metrics` applies per-iteration step offsets so cross-iter curves chain end-to-end (dotted vlines mark iter boundaries).
4. **Score + EDA.** Run [Run_Eval.ipynb](Exp3_PTO_GRPO/eda/notebooks/scoring/Run_Eval.ipynb) (resume-safe; its `EXPERIMENTS` registry auto-discovers the run from disk — no registry edit) → then open [1_Outcomes.ipynb](Exp3_PTO_GRPO/eda/notebooks/analysis/1_Outcomes.ipynb) (and `2`–`7`), which likewise **auto-discover** it. See "EDA workflow".

**PTO_Exp3**
1. **Configure.** [train_PTO_Iterative.ipynb](Exp3_PTO_GRPO/code/PTO_Exp3/train_PTO_Iterative.ipynb) cell 1 = flat globals. Key extra knobs vs GRPO: `PREF_TREE_MODE` (`greedy`|`independent`), `NUM_BRANCHES_PER_TURN`, `PREF_FILTER_TAU`, `BRANCH_SAMPLE_TEMPERATURE`, `DPO_BETA`, `DPO_LOSS_TYPE`. `greedy` mode requires an EVEN `MIN_CONV_LENGTH` (so the sliced prefix ends on a patient turn) and slices its trunk seeds from the step-1 convs (no separate prefix-generation pass).
2. **Train.** Same visible-orchestration pattern. Outputs land under `data/pto_Exp3/runs/<MODE_TAG>/<EXPERIMENT_NAME>/`. Each iteration also saves the constructed pref pairs to `iteration_N/pref_pairs/pairs.csv` (audit trail; the prompt + chosen + rejected + scores per pair).
3. **Inspect + Score + EDA.** Same as GRPO_Exp3 (the TB dashboard is shared via `_shared/tb_plots.py`).

## Exp3 · Training internals (current behavior)

The dated "how we got here" narratives — resume, checkpointing, batched look-ahead, per-generation
EDA capture, throughput tuning, and the first-run + ChatML-leak fixes — live in
[history/CHANGELOG_TRAINER.md](Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md). Current behavior:

- **Resume.** `resolve_start_state` ([_shared/model.py](Exp3_PTO_GRPO/code/_shared/model.py)) treats an iteration as
  done once `iteration_N/adapter/` exists. A crashed iteration resumes from the latest **valid** sub-epoch
  checkpoint (`SAVE_STEPS=10`, `SAVE_TOTAL_LIMIT=2`; `get_latest_valid_hf_checkpoint` walks back over a
  corrupt newest). **PTO Step-2** (the ~41-min pref-build) resumes too: `iteration_N/pref_pairs/pairs.csv`
  is both the DPO audit trail AND the completion marker (reload + skip), and `pref_pairs/_progress.json`
  is a per-step snapshot for mid-build resume (guarded by a config fingerprint incl. τ, which is NOT in
  `EXPERIMENT_NAME`, so a different-τ checkpoint is discarded not mixed).
- **K-turn look-ahead is batched.** `simulate_lookahead_batch` ([_shared/reward.py](Exp3_PTO_GRPO/code/_shared/reward.py))
  advances all B completions in lock-step — one padded batched `model.generate` per look-ahead turn —
  ~statistically equal to the legacy serial path (validated on GPU, |Δmean|=0.024, 1.5×). Knob
  `LOOKAHEAD_SUB_BATCH_SIZE` (64 GRPO / 128 PTO on A100-80GB; auto-halves on OOM, kept sticky).
- **Per-generation EDA capture.** Each iter writes `iteration_N/eda/generations.jsonl` — one branch row
  with nested `candidates[]` (`completion`/`score`/per-questionnaire `sub_scores`/`lookahead.tail`) +
  `chosen_idx`; GRPO one row per group per epoch, PTO one row per branch. Knobs `SAVE_EDA_GENERATIONS`,
  `SAVE_LOOKAHEAD_TRANSCRIPTS`. The EDA reads these ([eda_analysis/training.py](Exp3_PTO_GRPO/eda/eda_analysis/training.py)).
- **Anti-degeneracy (the base 1B self-plays ChatML markers).** `STOP_STRINGS=["<|im_end|>","<|im_start|>"]`
  + `clean_completion` ([_shared/convs.py](Exp3_PTO_GRPO/code/_shared/convs.py)) cut generation at the first fake-turn
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

## Exp3 · Dependency stack

Pins live in [requirements.txt](requirements.txt); both notebooks' Colab install cells are
pinned to it. The full 2026-06-01/03 audit (TRL 1.x / transformers 5 API currency, `hf_xet`,
gpt-4o-mini retirement check, batch/LR notes) is in
[history/CHANGELOG_TRAINER.md](Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md).
The one live install gotcha: **uninstall torchao on Colab** — peft 0.19.1 *raises* inside
`get_peft_model`'s `dispatch_torchao` on Colab's pre-baked torchao<0.16.0 (both install cells carry
the commented `%pip uninstall -y torchao`).

## Exp3 · Colab vs local

Realistic workflow: **training on Colab (GPU)**, **EDA + Run_Eval locally**.
EDA has no Colab branches — host-agnostic by design. Dual-host plumbing in
the trainers is only there to keep them importable + smoke-testable locally.

**Local GPU generation is viable — training is not.** `generate_eval_convs.{py,ipynb}` runs a
96-conv generate-only pass on the 12 GB local card in ~50 min at `--batch-size 6` (~16 batches),
and is API-bound there (mean GPU util 28% at batch 4 — patient calls dominate, which is why big
batches on an A100 win: they amortize the API wait across all 96 conversations). Respect the VRAM
ceiling in § Gotchas. Local *training* remains Colab-only for unrelated reasons (see the
`project-local-training-blackwell-crash` memory).

Experiment root resolution:
- **Local.** Walk up from `os.getcwd()` for `HF_key.txt`+`openai_key.txt` → typically `Exp3_PTO_GRPO/`.
- **Colab.** Trainer notebooks cd into `code/<METHOD>_Exp3/` after mounting Drive, then prepend `code/` to `sys.path` so `_shared` resolves as a sibling package.

### Auth (trainer only — `init_openai_client` / `authenticate` in [_shared/runtime.py](Exp3_PTO_GRPO/code/_shared/runtime.py))

| Secret | Colab | Local |
|---|---|---|
| OpenAI | `userdata["OPENAI_API_KEY"]` → env → file | env (`OPENAI_API_KEY`) → file |
| HF token | `userdata["huggingface"]` → env → file | env (`HF_TOKEN`/`HUGGINGFACE_TOKEN`) → file |
| W&B | `userdata["wandb"]` | env `WANDB_API_KEY` |

HF token IS used locally — Llama-3.2-1B is gated.

### Sync (Colab ↔ local)

**Results pull — Google Drive Desktop, no rclone.** `data/eval_scores`, `data/grpo_Exp3` and
`data/pto_Exp3` are all **directory symlinks** into Drive
(`G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data\<name>`). Colab writes to mounted
Drive → Drive Desktop (kept in **streaming** mode, low disk) surfaces it locally →
files appear straight inside the repo; EDA reads through the link unchanged (all reads
go via `WORKSPACE_ROOT/data/...`). EDA only reads `conversations/` + the score lake's
CSVs/parquet, so streaming downloads just those on open; big artifacts (`runs/`, adapters,
`*.safetensors`) are never read locally and also live on HF Hub + W&B.

Re-create the links (Windows **Developer Mode** on; use `mklink`, **not** PowerShell
`New-Item -ItemType SymbolicLink` — WinPS 5.1 ignores Dev Mode and still demands admin):
```powershell
$D = "G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data"
$R = "C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data"
cmd /c "mklink /D ""$R\eval_scores"" ""$D\eval_scores"""
cmd /c "mklink /D ""$R\grpo_Exp3""   ""$D\grpo_Exp3"""
cmd /c "mklink /D ""$R\pto_Exp3""    ""$D\pto_Exp3"""
```
To undo: delete the **link** (`Remove-Item "$R\grpo_Exp3"`) — Drive data untouched.

⚠ **The score lake is the only copy of ~$350 of oracle + judge calls.** Re-scoring it is not
affordable (see the cost constraint above), so it lives on Drive rather than local-only, and
`_parquet/` gives a 31-file form that is cheap to copy somewhere else again.

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

## Exp3 · EDA extension points

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
- **`scoring/judge.py`** — add second-judge providers/models here (`JudgeSpec`); outputs land in `data/eval_scores/judge=<tag>/rep=<r>/`, never in another grader's partition. **Claude judges:** `json_schema` rejects `minimum`/`maximum`/`minItems`/`maxItems` (folded into `description` instead — do NOT just drop them, or the array-shaped rubrics lose their one-score-per-item guarantee), and Sonnet 5 / Opus 4.8+ need `thinking={"type":"disabled"}` or adaptive thinking eats `max_tokens`.
- **`scoring/judge_plan.py`** (FREE pre-flight, no API) — `check_rubric_parity()` is **the gate before any second-judge spend**: it verifies every constraint stripped for Claude was restated in `description` and the encodings are otherwise structurally identical. Runs automatically in `_selfcheck`. Also `prefix_report()` (which rubrics actually prompt-cache), `plan_sweep()` (coverage-aware call count, skips existing CSVs), `estimate_cost`/`sweep_report`. **Pricing lives in `JUDGE_PRICING` — verify against the billing dashboard before quoting a number.**
- **`scoring/judge_batch.py`** (PAID) — the full-sweep path via **Anthropic Message Batches (50% off)**: `submit_sweep` → `poll_batches` → `collect_batches`, three separate phases with manifests persisted under `data/eval_scores/_batches/` so collection works from a fresh kernel. `custom_id` is an opaque index into that manifest, never an encoded path (model+metric+oracle overflows the 64-char limit and a truncation collision would write a score to the wrong model's folder). Anthropic-only by design — the primary judge already has a full rep, and extra reps are cheap enough for the live path.
- **`reliability.py`** (analysis layer, disk-only) — the FREE read side of `data/eval_scores/`: ICC/agreement/contrast tables for `8_Measurement_Validity` §1, plus the **multi-judge** layer for its §2 (`variance_components_arm` → arm vs judge-level vs arm×judge + `dependability_k1/k2`, `gain_retention`, `all_pairs_contrasts`, `sign_preservation`, `concordance_by_effect_size`). Figures in `plotting/reliability.py`. Keep the paid scoring in `scoring/judge*.py` and the presentation here, so judge results render inside `tools/render_views.py`.
  - ⚠ **Never average raw scores across judges.** The primary oracle WAS the training reward and the second judge is held out — that is train-vs-test, not two raters. The level offset is 1.2–1.7 points *and model-dependent*, so averaging applies a silent model-dependent shrinkage to every effect. Combine only contrasts or standardized quantities.
  - ⚠ **Pair on `persona_id`, not `file_index`** (`attach_persona`). The 96 personas are reshuffled each iteration, so a `file_index` join across unmatched iterations pairs unrelated conversations. Means survive it; `dz` and CIs do not.
- **Prompt caching is narrower than the gotcha below implies** (measured 2026-07-27 by `prefix_report`): only **Q1 and Q2** clear OpenAI's 1,024-token minimum. WAI-SR/CSQ-8/MI-SAT are rubric-first but too short (403–507 tok); **MITI/PCT/MICI interpolate a per-conversation utterance count into the instructions ahead of the rubric**, truncating their prefix to 138–206 tok. Documented, NOT fixed — those counts are the rate metrics' denominators, and editing the prompt would break comparability with all 22,272 conversations already scored.

## Exp3 · Gotchas

- **HF model-card READMEs** inside `data/grpo_Exp3/runs/.../checkpoint-*/` are auto-generated — DO NOT delete or treat as project docs.
- **Pref-tree audit trail = resume marker.** PTO_Exp3 writes `iteration_N/pref_pairs/pairs.csv` per iter. Don't delete — it's both the DPO debug trail AND the Step-2 completion marker: its presence makes a restart **reload it and skip the ~41-min build** (see "Training internals" → Resume). The sibling `iteration_N/pref_pairs/_progress.json` is the in-build per-step checkpoint (auto-deleted on success; safe to delete manually to force a clean rebuild).
- **Per-generation EDA.** `iteration_N/eda/generations.jsonl` (one row per branch, candidates nested — see "Training internals") is separate from `pref_pairs/pairs.csv` (the PTO DPO audit trail). Off-switch: `SAVE_EDA_GENERATIONS=False`. The continuous live-TB run lives at `runs/.../tb_live/` (sibling of `iteration_N/`).
- **PTO `branch_id` is trunk DEPTH, not a unique id.** Unlike GRPO's, it repeats across conversations, so any per-branch aggregation must key on `(conversation_id, branch_id)` — pooling on `branch_id` alone mixes unrelated conversations.
- **Local GPU: an over-budget VRAM request REBOOTS the PC — it does not raise `OutOfMemoryError`.**
  On the RTX 5070 Ti (12,227 MiB, sm_120, driver 610.62) exceeding VRAM is a hard GPU/driver fault
  that takes the OS down with no Python traceback — so you cannot catch it, and `--batch-size` is a
  safety setting, not just a throughput knob. **Measured 2026-07-30** for conv generation: weights
  2.6 GB + **≈1.1 GB per concurrent conversation** ⇒ batch 4 = 7.1 GB (58%), batch 6 ≈ 8.0 GB/batch,
  **batch 32 ≈ 38 GB → rebooted the machine**. Do that arithmetic before raising the batch. Full
  batch (96) means Colab. **Do NOT reason "inference-only, so it's safe"** — the crash is about the
  memory request, not the backward pass. Watch the `vram <N>G` field on each batch line
  (`torch.cuda.memory_reserved()`): flat across batches = healthy, climbing = the inter-batch
  `empty_cache()` in [_shared/convs.py](Exp3_PTO_GRPO/code/_shared/convs.py) has regressed. A
  **single-batch** smoke test cannot detect that class of leak — it needs ≥2 batches.
- **Editing `_shared/` does NOT affect an already-running Jupyter kernel.** Python caches imported
  modules, so a live kernel keeps the old code. Symptom of exactly this: the batch lines lack the
  `vram` field that ships with the current `convs.py`. **Restart the kernel** — per-CSV conversation
  resume makes it free.
- **Local sm_120 import order: `trl` must be imported BEFORE `torch`.** On the local Blackwell GPU, `from trl import …` *after* torch is already imported **segfaults at CUDA init** (a native init-order conflict, exit 139 — not OOM, not a bug in the trainers; Colab is unaffected, which is why the full runs ran there). The trainer modules already import `trl` first; only matters if you run something locally that imports torch/`_shared` first. Verified 2026-06-07.
- **Local offline smoke:** [code/_local_smoke.py](Exp3_PTO_GRPO/code/_local_smoke.py) — `python _local_smoke.py {stopgen|dpo|grpo|all}`. Tiny, no OpenAI; validates the stop-string bind, the DPO prompt-cap + no-OOM (grad-ckpt+precompute), and a GRPO step on the local GPU (~3 GB peak). Imports `trl` first (see above). All three PASS as of 2026-06-07.
- **Oracle prompt caching depends on the rubric-first layout.** [questionnaires.py](Exp3_PTO_GRPO/code/questionnaires.py) `get_prompt_eval_questionnaire` puts the fixed instructions + questionnaire rubric FIRST and the variable transcript LAST, so OpenAI's automatic prompt caching hits the ~1,084-token fixed prefix on every oracle call (≈50 % input discount + lower latency — matters for the oracle bill, the binding cost constraint above, even though wall-clock is GPU-bound; see next bullet). The margin over OpenAI's 1,024-token minimum is thin: **don't trim the oracle instructions/rubric or move the transcript ahead of them**, or caching silently stops (verified 2026-06-07: prefix is transcript-independent for Q1). Patient API calls auto-cache too (stable system + growing-history prefix). The therapist's local `model.generate` has **no** cross-call prefix reuse under HF — that would need vLLM (a real build here, not a flag: the look-ahead and *all* of PTO's generation use custom `model.generate`, not TRL's `use_vllm` path).
- **The run is likely GPU-bound, not API-bound (corrected 2026-06-07).** Earlier notes called the runs "API-bound" — that was inferred from GPU *memory* (17/67 GB), which does NOT measure compute. Lior observes he waits on GPU, not API. Autoregressive `model.generate` on the 1B LoRA policy (GRPO's G=8 completion sampling + K-turn look-ahead; PTO's branch sampling + look-ahead) dominates wall-clock; the `340.6 s / 8 GPU calls` look-ahead line ≈ 30–40 s per batched generate, far above the ~1–2 s of raw 1B/A100 compute → heavy per-step overhead. **Top suspect: the recently-added `STOP_STRINGS` route generation through HF `StopStringCriteria` (runs every step; known multi-× slowdown).** Before optimizing, MEASURE the split (time sampling vs look-ahead-GPU vs look-ahead-API vs backward); the K=0 arms (no look-ahead) running much faster would itself confirm generation is the cost. Faster stop than string-matching: register the two markers as single special tokens + stop on `eos_token_id`.

## Hardware
Local: Windows, RTX 5070 Ti (12 GB VRAM), CUDA 12.8, torch 2.11.0+cu128.
Training (both methods) is intended for Colab (GPU); EDA + Run_Eval run locally.
