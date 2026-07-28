# Thesis — Looking Ahead in Goal-Oriented Dialogue: Comparing Preference-Tree and Group-Relative Optimization of Small Language Models for Motivational Interviewing

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
| **Status** | Frozen — published | Complete — EDA verified | **Active — main thesis chapter; see "Current status & next step" below** |
| **Therapist** | Llama-2-7B | Llama-3.2-1B (4-bit NF4) | Llama-3.2-1B (bf16) |
| **Patient + oracle** | GPT-3.5 | gpt-4o-mini-2024-07-18 | gpt-4o-mini-2024-07-18 |
| **Patient prompts** | V1 (cooperative) | V3 (less cooperative) | V3 |
| **Oracle output** | V1 (regex; Q1+Q2 only) | V5 (JSON schema; 6 questionnaires) | V5 |
| **PTO** | K ∈ {0, 5}, 7 iters | 4 oracles × K ∈ {0, 5} | **PTO_Exp3** (iterative; lean sibling of GRPO_Exp3, controlled hyperparams matched) |
| **GRPO** | — | V1 (static prompts, weak baseline) | **GRPO_Exp3** (iterative) — both methods now share `code/_shared/` |
| **MCL filter** | — | — | **Wired in both PTO_Exp3 and GRPO_Exp3.** Encoded in `EXPERIMENT_NAME`. |
| **Training reward** | mean(Q1, Q2) | chosen oracle | Q1+Q2 only (matches Exp1) |
| **Eval reward** | Q1, Q2 | per-oracle | all 6 questionnaires |
| **EDA shape** | `Conv_EDA.ipynb` | + per-Q CSVs, `pref_emb/` | `eda_analysis/` package (analysis top level + `scoring/` subpackage backing `Run_Eval`) + tier-based notebooks `1_Outcomes`–`7_Stats` (+ `0_headline/` family; final-vs-best endpoint pairs); per-generation `iteration_N/eda/generations.jsonl` |
| **Convs / models** | (paper figures) | 4,512 / 47 | 2,784 / 29 (PTO+GRPO LA0 to iter 10 + partial LA5: PTO I1–4, GRPO I1) |

Dirs renamed 2026-05-12 from `ICLR2025/`/`Extension/`/`NewExperiment/`.

## Data lineage
- **Exp1 → Exp2:** independent re-implementation. Stronger oracle, harder patients, JSON-schema rubric, more questionnaires. No data flow.
- **Exp2 → Exp3:** independent re-implementation — **Exp3 is a complete, fresh experiment that shares no data with Exp2** (both PTO_Exp3 and GRPO_Exp3 generate all their own convs from scratch each iteration; see the Exp3 self-loop below).
  - ⚠ **Exp2 and Exp3 absolute oracle scores are NOT on the same axis.** Same therapist base (Llama-3.2-1B), but Exp2 generated its convs in **4-bit NF4** and Exp3 in **bf16**. 4-bit induces ~30× more phrase-loop degeneration (≈9.5% vs 0.3% of therapist turns run to the token cap as repeated spam), which the oracle floors — so Exp2 Base ≈ 2.38 Q1+Q2 vs Exp3 Base ≈ 3.0, *even though it's the same model*. The clean (non-degenerate) Exp2 subset scores ≈ 2.93 ≈ Exp3. **Compare within Exp3 only**; to put Exp2 on the same axis, regenerate its convs in bf16.
- **Exp3 self-loop:** GRPO_Exp3 regenerates its own training data each iter from the current policy; those same convs are the eval set (no separate generate-eval step for trained iters).

## Key methodological shift across experiments
- **Look-ahead K** stayed central throughout (the lever from the ICLR paper).
- **The hard part moved from "can PTO beat the baseline?" (Exp1, settled) to "is GRPO competitive with PTO under matched look-ahead?" (Exp3, open).**
- **Exp3 also exposed a reward-faithfulness concern** the earlier experiments never tested: the partial-conversation oracle diagnostic (originally `Partial_Conv_Oracle_EDA` on Exp2 data; now rebuilt on Exp3 data in `eda/notebooks/analysis/5_Training_and_Reliability.ipynb`) shows that the short-cut training reward has only ~0.66–0.73 rank agreement with the full-conv eval at `n_turns=2`. Motivates the `MIN_CONV_LENGTH` knob — now wired in both GRPO_Exp3 (slice filter) and PTO_Exp3 (greedy: tree-start prefix length; independent: branch-point filter); encoded in `EXPERIMENT_NAME` so MCL sweeps stay in disjoint folders.

## Methods (one line each)
- **PTO V1** (Exp1) = original preference-tree exploration + K look-ahead + DPO. Published.
- **GRPO V1** (Exp2) = static prompt set, weak baseline.
- **GRPO_Exp3** = current policy simulates 96 convs → per-turn prompts (MCL filter) → GRPO update with optional K-turn look-ahead. Convs double as eval.
- **PTO_Exp3** = per-turn branching (`M` candidates) → K-turn look-ahead + oracle → τ-filtered (chosen, rejected) pref pairs → DPO update. Lean sibling of GRPO_Exp3. **Two `PREF_TREE_MODE`s:** `greedy` (default, true PTO — start from an MCL-length prefix sliced off the step-1 conv and grow ONE trunk by appending the best-of-M completion at each therapist turn, so the choice feeds the next branch point) and `independent` (branch each patient turn of a pre-recorded conv, no feedback). Mode baked into `EXPERIMENT_NAME`.

**Shared infrastructure (Exp3).** Both GRPO_Exp3 and PTO_Exp3 trainers import from
`Exp3_PTO_GRPO/code/_shared/` (runtime, model, convs, reward, tb_plots, eda_recorder; + optional lookahead_check).
Each method's trainer module (`grpo_trainer.py` / `pto_trainer.py` — named per method
so `from <method>_trainer import …` can't collide in a shared kernel) owns just the
method-specific bits (`TrainingConfig`/`PTOConfig`, iteration body, dataset shape, TRL
trainer wrapping).

**Naming:** PTO is the framework, DPO is the loss. Don't call GRPO data "pref data" — it has none.

## Layout
```
Thesis_PTO_GRPO/
├── CLAUDE.md                   (this file)
├── README.md, DATA_README.md, LICENSE
├── Exp{1,2,3}_*/CLAUDE.md      per-experiment context
├── history/CHANGELOG.md        thin dated cross-experiment index (detail lives per-experiment)
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

## Current status & next step
**THE single live copy of run status + headline numbers + cost constraint** (all other docs point
here — see "Doc map"). Updated 2026-07-08.

- **Run status:** PTO LA0 = 10 iters scored; **GRPO LA0 = 10 iters (FINISHED, re-scored)**. **Both
  LA5 arms PAUSED/thin** (PTO LA5: I1–I4 scored + an unscored iter-5 adapter whose eval convs were
  never generated; GRPO LA5: I1 trained AND fully scored).
- **Headline:** **PTO wins at the matched 10-iter endpoint (Q1+Q2 4.26 vs 3.75; paired +0.51,
  dz 0.73)** because GRPO peaks at iter 8 (4.08) then regresses into sycophancy (MICI endpoint 0.84
  vs PTO 0.49); PTO climbs stably. Full narrative + tables:
  `Exp3_PTO_GRPO/eda/results/<view>/SUMMARY.md` (L0 = primary read).
- **Judge validity (NEW 2026-07-26):** the measurement instrument is now measured, not assumed —
  oracle **ICC(2,1) 0.86–0.99** (mean |Δ| 0.04–0.09, confirming the "≈0.10 noise" folklore; Q1/Q2
  hold 0.96–0.99 and only MICI dips below 0.90 — floor is MICI PTO@10 at 0.864), and a
  decoupled second judge (**Claude Haiku 4.5**, different family, never played the patient)
  reproduces **6/6 endpoint contrasts with the same sign** (it *widens* the PTO−GRPO Q1 gap to
  +0.77 vs the primary's +0.53). Q1/Q2 cross-judge r 0.80–0.88 vs a ~0.98 ceiling; MICI agrees
  weakly (r 0.20–0.55) so the sycophancy claim holds at the contrast level, not as a precise rate.
  Buys down LIMITATIONS §1–§2. Cost ~$5.30. See `eda/notebooks/analysis/5_Training_and_Reliability` §7.
- **Cost constraint:** OpenAI spend hit **~$300** and is binding — RQ-i (K0 vs K5) on hold. Cost is
  dominated by oracle scoring + (at K=5) look-ahead patient calls, both ∝ candidate count
  (`prompts×G` / `branch-points×M`) × iterations; prompt caching is already maxed (~50% off the
  oracle's fixed prefix), so the only lever is call **COUNT**: cap `NUM_ITERATIONS` ~5–6 (curves
  plateau by iter ~4), drop `M`/`G` 8→4, (PTO) lower `GREEDY_TRUNK_TARGET_LEN` — keep **K** (the
  RQ-i variable) and the **gpt-4o-mini oracle** (the measurement instrument) fixed. See the
  `project-openai-cost-constraint` memory.
- **Next step:** cheapest RQ-i point = one generate-only pass with the existing PTO LA5 iter-5
  adapter (96 convs, no training) + `Run_Eval` scoring; then resume an LA5 arm when budget allows.
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
  `5_Training_and_Reliability` **§8** (free, in `tools/render_views.py`) + `Judge_Reliability.ipynb` **§3**
  (the paid full sweep). Headline results, now on the FULL 29-arm grid:
  ⚠ **Numbers below are the `L0` view (22 arms), the tracked deliverable.** Pre-2026-07-28 versions
  of this block quoted the pooled 29-arm `all` view, which is now gitignored scratch and therefore
  not reproducible from a tracked artifact. Same story, ±1–2 percentage points.
  - **18/18 pairwise contrasts keep their sign** on the six the thesis leans on hardest (was 6/6 —
    only two hand-picked pairs were ever checked); the two newly covered are the best-vs-best
    steelman (PTO@10 − GRPO@8) and the regression claim (GRPO@8 − GRPO@10). Bootstrap CIs exclude
    zero on all but MICI GRPO@8 − PTO@10.
    **On the FULL enumeration — all 1,848 arm×metric contrasts in L0 — 88.3% keep their sign,
    rising to 94.1% at |Δ|≥0.10, 97.0% at ≥0.25, 98.9% at ≥0.50** (94.7% among the 1,299 whose
    judge-side CI excludes zero). The judges disagree only about differences too small to claim. Per
    rubric, MITI is worst (77.5%) and Q1 is 86.6%, vs PCT 93.5% / MICI 92.2%. New tracked tables
    `multijudge_sign_preservation{,_by_metric}.md` (`reliability.sign_preservation`, added
    2026-07-28 — the ladder was previously computed nowhere).
  - **Variance decomposition** (22 arms × 2 judges): arm-mean variance is 3.6–72% arm, 22–95% judge
    *level*, and only **1.2–6.9% arm×judge** — the judges disagree about level, not about arm
    ordering. Dependability of an arm mean off ONE judge = 0.88–0.95 on seven rubrics.
    ⚠ **MITI is the exception: 3.6% arm / 94.5% judge / dependability 0.65.** Almost all of MITI's
    arm-mean spread is grader level, so a single-judge MITI ranking is materially less trustworthy
    than the others — averaging both judges lifts it to 0.79. Treat MITI arm differences as
    provisional unless both judges agree. Corroborated independently by its 77.5% sign preservation
    above. (Q1 10.9% arm, dep 0.90; Q2 13.2%, 0.91.)
  - **Gain retention (the reward-hacking test)**: Q1 retention PTO@10 **0.80** [0.68, 0.93] vs
    GRPO@10 **0.28** [0.06, 0.43] — non-overlapping — while every Q2 interval overlaps (0.80–0.85).
    Confirmed unchanged on the full grid (was measured on 4 anchors).
    Under a held-out judge GRPO's net 10-iter Q1 gain is ≈0.19, not the primary's ≈0.68. Stronger
    evidence for sycophancy than the MICI rate. Buys down LIMITATIONS §3 (circularity).
    **NEW 2026-07-28 — it is an *onset* curve, not an endpoint fact.** Per-iteration Q1 retention:
    PTO holds 0.80–0.98 across all 10 iters; GRPO decays monotonically in trend
    (I3 0.89 → I6 0.57 → I9 0.03 → I10 0.28), the two arms being indistinguishable for the first
    three. The held-out grader stops crediting GRPO's gains *progressively*.
    Figure: `multijudge_retention_trajectory.png` (existed, was never written up).
  - **Concordance vs effect size** replaces a scalar r/ρ (level bias dominates Pearson; ranks
    discard magnitude). Per *conversation pair*, so it is NOT a confidence in an arm-level claim.
  - **Sweep history (for the record).** First submission (9 batches, 21,120 requests) landed 43% —
    12,090 failed on `invalid_request_error: "Your credit balance is too low"`; those were never
    billed. After a top-up the remainder was resubmitted (5 batches) and completed, plus 13
    stragglers filled via the live path (`Grammar compilation timed out`, transient). Coverage is
    now 100%, so `reliability.filter_complete_cells` drops nothing and `multijudge_coverage.md`
    reports 232/232.
  - **Cost, measured**: 3,621 input + 71 output tokens/call (`judge_batch.probe_usage`); full sweep
    **$42 batched / $84 direct**; the free char-based estimator lands within 12%. Parity gate 8/8.
    Deliberately **1 rep, not 3** — oracle noise adds ≈0.01 to a 96-conv arm mean vs ≈0.09 from
    persona sampling, so breadth beats depth at equal cost.
    ⚠ **Haiku 4.5 caches nothing on this prompt** — confirmed empirically
    (`cached_input_tokens = 0`): its cacheable-prefix minimum is 4,096 and only Q1/Q2 come close.
- **Second-judge ICC — MEASURED 2026-07-28**, closing the last named validity gap (was the
  "cheapest remaining validity buy"). 2 further Haiku reps on the anchor subset, 2,304 calls,
  0 errors. Haiku's own ICC: **Q1 0.951–0.978, Q2 0.938–0.963, MICI 0.525–0.929.**
  - The prior assumption (`ICC_judge == ICC_primary`) held on Q1/Q2 but not on MICI: Haiku's
    repeatability falls as the MI-inconsistency rate rises (PTO Base 0.929 → PTO@10 0.815 →
    GRPO@8 0.749 → **GRPO@10 0.525**), i.e. it is least reliable on the arms the sycophancy claim
    concerns, where the achievable ceiling is 0.70, not 0.93.
  - **Attribution: partly the judge's noise, mostly construct disagreement.** Against the corrected
    ceiling, agreement recovers Q1 86–91%, Q2 83–88%, MICI only **29–59%**. The MICI caveat stands
    and gain retention remains the load-bearing sycophancy evidence — **no headline result moves.**
  - ⚠ **Constrains §8:** the multi-judge analysis reads Haiku **rep 0 only**, and single-rep Haiku
    MICI on GRPO@10 is ICC 0.525. Treat one-rep MICI on high-MICI arms as indicative; the 3 anchor
    reps now on disk would resolve it for those four model states.
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
  number moves** (45/45 endpoint cells and 25,056 rows identical, `_selfcheck` 13/13). Two
  consequences worth knowing:
  - **The lake is a Drive symlink**, so the second judge's $42 sweep and the $9.16 ICC reps are
    backed up for the first time — previously they existed only on one laptop, gitignored.
  - **The primary's ICC now spans 4 draws** (the reported one included, as the second judge's
    already did), which is why the range above reads 0.86–0.99 rather than the older 0.90–0.99.
    Only MICI moves; Q1/Q2 shift ≤0.007.
  - Archival fold: `eda/tools/consolidate_scores.py` collapses the 50,305 CSVs to 31 parquet files
    (verified lossless). Not a read path — deliberately unimported, so it cannot feed a stale figure.

## Doc map (one owner per fact)
| Fact | Lives ONLY in |
|---|---|
| Run status + headline numbers + cost constraint | this file → "Current status & next step" |
| Detailed eval narrative + numbers | `Exp3_PTO_GRPO/eda/results/<view>/SUMMARY.md` |
| EDA how-to (VIEW knob, `EdaConfig`, package module map) | `Exp3_PTO_GRPO/eda/README.md` |
| Metric definitions (no current values) | `Exp3_PTO_GRPO/eda/docs/METRICS_REFERENCE.md` |
| Dated history | `Exp3_PTO_GRPO/history/CHANGELOG.md` (detail); root [history/CHANGELOG.md](history/CHANGELOG.md) = thin index |
| Method mechanics, trainer internals, gotchas | `Exp3_PTO_GRPO/CLAUDE.md` |

Update a fact in its owner file only; everywhere else keep a pointer.

## Hardware
Local: Windows, RTX 5070 Ti (12 GB VRAM), CUDA 12.8, torch 2.11.0+cu128.
GRPO_Exp3 training is intended for Colab (GPU); EDA + Run_Eval run locally.
