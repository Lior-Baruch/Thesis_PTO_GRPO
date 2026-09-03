# Thesis — Looking Ahead in Goal-Oriented Dialogue: Comparing Preference-Tree and Group-Relative Optimization of Small Language Models for Motivational Interviewing

## What this is
Master's thesis (Lior Baruch, Reichman). Trains small therapist LLMs to do
Motivational Interviewing against simulated patients; reward = larger
"oracle" LLM grading validated MI questionnaires.

Three controlled comparisons, all live in Exp3:
1. **Look-ahead depth** K ∈ {0, 5} — does anticipating future turns help, and by how much?
2. **PTO vs GRPO** under matched K + MCL — does iterative GRPO compete with PTO?
3. **Oracle questionnaire** (Q1+Q2 vs WAI-SR vs CSQ-8 vs MI-SAT/MITI) — held for later.

> **This file is also the Exp3 context file.** There is no `Exp3_PTO_GRPO/CLAUDE.md` — its content
> lives in "Exp3_PTO_GRPO — the active experiment" below, so the active experiment is always in
> context and there is exactly one spec file to update. Exp1 and Exp2 keep their own `CLAUDE.md`
> because they are frozen/complete.
>
> **What is NOT here:** anything that changes weekly → [STATUS.md](STATUS.md); anything dated →
> [history/](Exp3_PTO_GRPO/history/CHANGELOG.md). See the Doc map below.

## Experiments (chronological)
| | [Exp1_ICLR2025/](Exp1_ICLR2025/) | [Exp2_PTO/](Exp2_PTO/) | [Exp3_PTO_GRPO/](Exp3_PTO_GRPO/) |
|---|---|---|---|
| **Status** | Frozen — published | Complete — EDA verified | **Active — main thesis chapter; live run status in [STATUS.md](STATUS.md)** |
| **Therapist** | Llama-2-7B | Llama-3.2-1B (4-bit NF4) | Llama-3.2-1B (bf16) |
| **Patient + oracle** | GPT-3.5 | gpt-4o-mini-2024-07-18 | gpt-4o-mini-2024-07-18 |
| **Patient prompts** | V1 (cooperative) | V3 (less cooperative) | V3 |
| **Oracle output** | V1 (regex; Q1+Q2 only) | V5 (JSON schema; 6 questionnaires) | V5 + PCT/MITI-style coders → **8 instruments** |
| **PTO** | K ∈ {0, 5}, 7 iters | 4 oracles × K ∈ {0, 5} | **PTO_Exp3** (iterative; lean sibling of GRPO_Exp3, controlled hyperparams matched) |
| **GRPO** | — | V1 — ⚠ **buggy; results VOID, not a baseline** | **GRPO_Exp3** (iterative) — both methods share `code/_shared/` |
| **MCL filter** | — | — | **Wired in both PTO_Exp3 and GRPO_Exp3.** Encoded in `EXPERIMENT_NAME`. |
| **Training reward** | mean(Q1, Q2) | chosen oracle | Q1+Q2 only (matches Exp1) |
| **Eval reward** | Q1, Q2 | per-oracle | **all 8 rubrics** — the 6 questionnaires + `PCT` + `MICI` (added 2026-06-14) |
| **EDA shape** | `Conv_EDA.ipynb` | + per-Q CSVs, `pref_emb/` | `eda_analysis/` package (analysis top level + `scoring/` subpackage backing `Run_Eval`) + one notebook per results **family** `notebooks/<top>/<sub>.ipynb` — `arms/*` (per-arm descriptives, all four arms, per judge), `lookahead/*` (K=0 vs K=5), `method/contrast`, `compute/cost`, `measurement/validity` (final-vs-best endpoint pairs); per-generation `iteration_N/eda/generations.jsonl` |
| **Convs / models** | (paper figures) | 4,512 / 47 — ⚠ **includes the void GRPO states; the PTO-only subset is smaller** | scored on both graders — **live counts in [STATUS.md](STATUS.md)** |

Dirs renamed 2026-05-12 from `ICLR2025/`/`Extension/`/`NewExperiment/`.

**Side project — [Exp4_OpenStack/](Exp4_OpenStack/).** The same PTO-vs-GRPO + look-ahead comparison
on a **fully open model stack**: oracle and patient are a Gemma-4 model (default
`google/gemma-4-E4B-it`, selectable) behind a local vLLM OpenAI-compatible server, so an arm costs
**$0 in API** — the constraint that stopped GRPO_LA5 in Exp3 — and runs on a **Colab A100 80 GB**
(GPU-hours are its only cost; the 40 GB card is a fallback). The therapist is also selectable
per arm (Llama-3.2-1B **Instruct** — the default — or the base variant), encoded in the arm name. Not a thesis chapter unless the results earn it. It is **self-contained and additive**: its
own spec, contract and status live in [Exp4_OpenStack/CLAUDE.md](Exp4_OpenStack/CLAUDE.md), nothing
in this file describes it, and no Exp3 file was modified for it.
⚠ **Exp3 and Exp4 scores are not on the same axis** — different grader. Compare within Exp4 only.

## Data lineage
- **Exp1 → Exp2:** independent re-implementation. Stronger oracle, harder patients, JSON-schema rubric, more questionnaires. No data flow.
- **Exp2 → Exp3:** independent re-implementation — **Exp3 is a complete, fresh experiment that shares no data with Exp2** (both PTO_Exp3 and GRPO_Exp3 generate all their own convs from scratch each iteration; see the Exp3 self-loop below).
  - ⚠ **Exp2 and Exp3 absolute oracle scores are NOT on the same axis.** Same therapist base (Llama-3.2-1B), but Exp2 generated its convs in **4-bit NF4** and Exp3 in **bf16**. 4-bit induces ~30× more phrase-loop degeneration (≈9.5% vs 0.3% of therapist turns run to the token cap as repeated spam), which the oracle floors — so Exp2 Base ≈ 2.38 Q1+Q2 vs Exp3 Base ≈ 3.0, *even though it's the same model*. The clean (non-degenerate) Exp2 subset scores ≈ 2.93 ≈ Exp3. **Compare within Exp3 only**; to put Exp2 on the same axis, regenerate its convs in bf16.
- **Exp3 self-loop:** GRPO_Exp3 regenerates its own training data each iter from the current policy; those same convs are the eval set (no separate generate-eval step for trained iters).

## Key methodological shift across experiments
- **Look-ahead K** stayed central throughout (the lever from the ICLR paper).
- **The hard part moved from "can PTO beat the baseline?" (Exp1, settled) to "is GRPO competitive with PTO under matched look-ahead?" (Exp3, open).**
- **Exp3 also exposed a reward-faithfulness concern** the earlier experiments never tested: the partial-conversation oracle diagnostic (originally `Partial_Conv_Oracle_EDA` on Exp2 data; now rebuilt on Exp3 data in [Exp3_PTO_GRPO/eda/notebooks/arms/training.ipynb](Exp3_PTO_GRPO/eda/notebooks/arms/training.ipynb), the K=0-vs-K=5 contrast at a matched policy in [lookahead/mechanism.ipynb](Exp3_PTO_GRPO/eda/notebooks/lookahead/mechanism.ipynb)) shows that the short-cut training reward has only ~0.66–0.73 rank agreement with the full-conv eval at `n_turns=2`. Motivates the `MIN_CONV_LENGTH` knob — now wired in both GRPO_Exp3 (slice filter) and PTO_Exp3 (greedy: tree-start prefix length; independent: branch-point filter); encoded in `EXPERIMENT_NAME` so MCL sweeps stay in disjoint folders.

## Methods (one line each)
- **PTO V1** (Exp1) = original preference-tree exploration + K look-ahead + DPO. Published.
- **GRPO V1** (Exp2) = static prompt set. ⚠ **The run had a BUG — its scores are void.** Not a weak
  baseline, not a comparison point, not evidence about GRPO: nothing. Never quote, plot or present
  them, and never build a cross-experiment "GRPO started weak and improved" arc on them. **The
  PTO-vs-GRPO comparison exists only in Exp3**, where both methods are iterative, share
  `code/_shared/`, and have matched hyperparameters — which is what makes it controlled and Exp2's
  not. Exp1 and Exp2 are **PTO-only** in every write-up and deck.
- **GRPO_Exp3** = current policy simulates 96 convs → per-turn prompts (MCL filter) → GRPO update with optional K-turn look-ahead. Convs double as eval.
- **PTO_Exp3** = per-turn branching (`M` candidates) → K-turn look-ahead + oracle → τ-filtered (chosen, rejected) pref pairs → DPO update. Lean sibling of GRPO_Exp3. **Two `PREF_TREE_MODE`s:** `greedy` (default, true PTO — start from an MCL-length prefix sliced off the step-1 conv and grow ONE trunk by appending the best-of-M completion at each therapist turn, so the choice feeds the next branch point) and `independent` (branch each patient turn of a pre-recorded conv, no feedback). Mode baked into `EXPERIMENT_NAME`.

**Naming:** PTO is the framework, DPO is the loss. Don't call GRPO data "pref data" — it has none.

## Current status & next step
**Lives in [STATUS.md](STATUS.md)** — run status, headline numbers, the cost constraint, the next
step, and the write-up decisions already made. It is deliberately short and rewritten in place
rather than appended to; the dated narrative it replaces retired to
[history/CHANGELOG_STATUS.md](Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md).

**Read STATUS.md before planning any spend or making any claim about a number.**

## Doc map (one owner per fact)

Docs are split by **rate of change**, not by topic. Update a fact in its owner file only;
everywhere else keep a pointer.

| Fact | Lives ONLY in | Changes |
|---|---|---|
| Run status, headline numbers, cost constraint, next step | [STATUS.md](STATUS.md) | weekly |
| Method mechanics, algorithms, trainer internals, gotchas, conventions | this file | when the code does |
| What each `code/` module does | `Exp3_PTO_GRPO/code/README.md` | when the code does |
| EDA how-to (FAMILY + JUDGE knobs, `EdaConfig`, exports API, `render_results.py`, package module map) | `Exp3_PTO_GRPO/eda/README.md` (the 2026-08-18 old→new migration table retired to `history/CHANGELOG_EDA.md` on 2026-08-26) | when the EDA does |
| **Results navigation** — each research question → its headline artifacts + reading rules | `Exp3_PTO_GRPO/eda/results/README.md` (hand-authored, in `exports.PRESERVE`) | when the artifacts do |
| Detailed eval narrative + numbers, per research question | `Exp3_PTO_GRPO/eda/results/<top>/SUMMARY.md` (`arms` · `lookahead` · `method` · `compute` · `measurement`) | per render |
| Metric definitions (no current values) | `Exp3_PTO_GRPO/eda/results/METRICS_REFERENCE.md` | rarely |
| Measurement / inference limitations (for the write-up) | `Exp3_PTO_GRPO/eda/results/LIMITATIONS.md` | rarely |
| **Paper drafts + the claim→artifact ledger** | [`papers/README.md`](papers/README.md), then each paper's `README.md` + **`NUMBERS.md`**. **Two live drafts, both → ARR October 2026, both iterations-only** (2026-08-27): [`papers/2026_pto_grpo_mi/`](papers/2026_pto_grpo_mi/) (*Same Lever, Different Optimizer* — the full 2×2) and [`papers/2026_grpo_lookahead_mi/`](papers/2026_grpo_lookahead_mi/) (*GRPO with Look-Ahead in Motivational Interviewing* — GRPO with look-ahead; PTO cited as origin, never data; ⚠ same-cycle overlap needs supervisor sign-off). The ICLR-format P1 and every earlier Exp3 draft are retired to `papers/archive/` (tracked). The EDA's `lookahead/` and `compute/` families own the cross-K numbers; the retired *Same Lever* draft's `analysis/out/` + `tables/` stay the EDA self-check's frozen fixture | per draft |
| Supervisor decks + emails | [`meetings/README.md`](meetings/README.md) | per meeting |
| Data/artifact policy (what's gitignored, how it regenerates) | `README.md` § "Data & large artifacts" | rarely |
| **Everything about the Exp4 side project** (spec, module contract, naming grammar, data layout, phase gates) | [`Exp4_OpenStack/CLAUDE.md`](Exp4_OpenStack/CLAUDE.md) | its own cadence |
| Dated history | `Exp3_PTO_GRPO/history/` — [CHANGELOG_STATUS.md](Exp3_PTO_GRPO/history/CHANGELOG_STATUS.md) (status + findings) · [CHANGELOG_EDA.md](Exp3_PTO_GRPO/history/CHANGELOG_EDA.md) · [CHANGELOG_TRAINER.md](Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md), behind a stable [index](Exp3_PTO_GRPO/history/CHANGELOG.md). There is no root changelog. | append-only |

⚠ **Nothing dated belongs in this file or in STATUS.md.** A dated entry is history — it goes to
`history/`. STATUS.md is rewritten in place; CLAUDE.md describes how things *are*.

## ⚠ Epistemic status of these docs — read before brainstorming

Docs here split by rate of change. They do **not** split by *how the claim was established*, and
that distinction matters more when generating ideas than when checking facts:

| Tier | Files | What it is |
|---|---|---|
| **MEASUREMENT** | `eda/results/<top>/<sub>/{tables,figures}/**` | auto-generated from the score lake. Numbers only, no interpretation. **The evidence.** |
| **INTERPRETATION** | `results/<top>/SUMMARY.md`, `STATUS.md` headline numbers, `LIMITATIONS.md`, this file's results claims, the auto-loaded memories | hand-authored readings *of* the measurements — written in earlier sessions, largely by Claude. **Not independent evidence.** |

**The failure mode this exists to prevent** is a closed loop: an interpretation is written into a
doc, loads as a prior next session, gets extended rather than re-derived, and after a few rounds
reads like an established finding while never having been checked against a table. Four such errors
shipped into a draft before being caught (the "exploration" framing, a retention interval claimed
disjoint that overlaps, a cell count nobody multiplied out, and a rate quoted from the wrong table).
**Every one came from prose about tables. The tables were never wrong.**

### Rules that follow

1. **Brainstorming framings, angles or papers: read the TABLES first, cold.** Do not open
   `SUMMARY.md` / `STATUS.md` / prior drafts until your own candidate framings are written down.
   Then read the narratives and *diff* — where the two disagree is the anchoring, made visible.
   Skipping this produces options that are the section headers of `SUMMARY.md` re-indexed, which
   is retrieval, not brainstorming.
2. **Any composite number must show its arithmetic** wherever it is quoted
   (`8 × 40 × 96 = 30,720`, not `30,720`). Atomic-looking numbers do not get audited.
2b. **A two-point ratio is only as stable as its anchor — quote a TREND beside it, or don't quote
   it.** Added 2026-08-25, after "the held-out grader's variance *grows* 1.410×" survived into
   STATUS.md, a family SUMMARY, a rendered figure and a paper draft. Iteration 0 happened to be
   that series' **minimum**: re-anchored to iteration 1 the ratio is 1.062× and the trend test is
   null (ρ = +0.44, p = .18). The arithmetic was correct and the inference was backwards. Before
   quoting `end / start`, check whether either endpoint is a series extremum, and report the
   trend (or the re-anchored value) in the same breath. The sibling claim in the same sentence —
   the primary's 0.275× collapse — was fine precisely because it is monotone (ρ = −0.86, p = .001)
   and survives re-anchoring.
3. **A number in prose is a claim about a table, not a number.** Before reusing one, open the table
   it cites — and check it cites the *right* table. (`SUMMARY.md` §4 has quoted the regex question
   rate while pointing at the oracle-coded one.)
4. **Interpretive vocabulary needs a mechanism before it ships.** "Exploration", "sharper
   discrimination", "more decisive" are compressions of something specific; state the specific
   thing and verify it against config or artifact.
5. **The docs record what WAS analysed, so what was never analysed is invisible.** A cold table read
   cannot fix this — tables only exist for questions someone already asked. When brainstorming,
   explicitly list what is *not* covered by any artifact.

## Conventions
- **Each experiment dir is self-contained.** Its own `code/`, `data/`, `eda/`, local `system_prompts_builder.py`+`questionnaires.py` (versions diverge across experiments — never share a root-level module). Within Exp3, both helpers live ONCE at `code/` root; the EDA package imports the same files via a `sys.path` prepend.
- **Workspace root resolver.** Walks up from `os.getcwd()` looking for `HF_key.txt`+`openai_key.txt` together → resolves to experiment root (`Exp{1,2,3}_*/`). Used by every notebook.
- **EDA path remapping.** Legacy strings like `"LLM_DATA/Conversation_with_Eval_V3/..."` (Exp1/Exp2 EDAs) are remapped at load time by `_resolve_data_path(...)`. Don't rewrite the literals.
- **File version suffixes (`_V3`, `_V5`)** are dropped when the file lives in an experiment dir (the dir provides version context). Method-lineage subdirs in Exp3 are named after the experiment (`GRPO_Exp3/`, `PTO_Exp3/`).
- **Exp3 trainer pattern.** `code/<METHOD>_Exp3/{train_<METHOD>_Iterative.ipynb, <method>_trainer.py}` (e.g. `grpo_trainer.py`, `pto_trainer.py` — distinct module names to avoid `from trainer` collisions across notebooks in one kernel) with the per-iteration orchestration loop visible in the notebook. Shared helpers in `code/_shared/`.

---

# Exp3_PTO_GRPO — the active experiment

Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle. Two methods compared under matched
look-ahead + oracle. **Hyperparameters matched across the two:** `MCL=12`,
K ∈ {0,5}, gen temps + API concurrency; PTO's `M` (`NUM_BRANCHES_PER_TURN`)=8 mirrors GRPO's
`NUM_GENERATIONS`; `DPO_BETA`=0.1 is the DPO loss temperature, **not** GRPO's KL β. ⚠ `NUM_ITERATIONS`
is **not** currently matched (live cell 1: GRPO 6, PTO 8) — it is volatile, so read cell 1 or
[STATUS.md](STATUS.md), never a number here. bf16
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

Directory **purposes** below. The per-file map is NOT duplicated here — it drifts. For the current
file list use `git ls-files Exp3_PTO_GRPO`, and for what each module does see
[code/README.md](Exp3_PTO_GRPO/code/README.md) and [eda/README.md](Exp3_PTO_GRPO/eda/README.md)
§ "Package".

```
Exp3_PTO_GRPO/
├── README.md      map of this folder only (code/ eda/ history/ + how they connect); the spec stays HERE
├── code/          the trainers. Two method dirs (GRPO_Exp3/, PTO_Exp3/) over one _shared/ layer;
│                  system_prompts_builder.py + questionnaires.py live here ONCE (canonical copies —
│                  the EDA package sys.path-prepends code/ to import the same files); tools/ holds the
│                  stand-alone utilities (_local_smoke.py, generate_eval_convs.{py,ipynb}). → code/README.md
├── data/          ALL THREE subdirs are Google Drive symlinks (backed up + reachable from Colab).
│                  GITIGNORED, so the schemas below are the only record of their shape:
│   ├── eval_scores/   THE SCORE LAKE — every grader's scores, one shape:
│   │                    judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
│   │                  M=scoring metric, O=training oracle. rep=0 is each judge's FULL-GRID draw
│   │                  (the reported one); rep>=1 are repeatability draws on the anchor subset. No
│   │                  method level — <Model> already carries it (GRPOExp3_* / PTOExp3_*).
│   │                  _parquet/judge=<tag>/rep=<r>/metric=<M>.parquet + _manifest.json — derived
│   │                  fold (50,305 -> 31 files), READ by iter_conv_rows but only while the
│   │                  manifest's per-partition content signature still matches disk.
│   │                  _batches/<tag>/rep=<r>/*.json — Message Batches manifests (submit -> collect).
│   └── {grpo,pto}_Exp3/   runs/<MODE_TAG>/<EXP_NAME>/ (run_metadata.json + iteration_N/{adapter,
│                          training}/, PTO also pref_pairs/) and
│                          conversations/<MODE_TAG>/<EXP_NAME>/model_iter_<N>_TT*_TP*/
├── eda/           the analysis. eda_analysis/ package + notebooks/<top>/<sub>.ipynb (one per results
│                  FAMILY: arms/{outcomes,questionnaires,validity,heterogeneity,training,preference,stats},
│                  lookahead/{reward,transfer,behaviour,mechanism,replication}, method/contrast,
│                  compute/cost, measurement/validity) + notebooks/scoring/ (the PAID side) + tools/
│                  (render_results.py, consolidate_scores.py, score_crossgen.py, strip_notebook_outputs.py)
│                  + results/. Artifacts nest results/<top>/<sub>/{figures,tables}/[<judge>/][<group>/]
│                  — the <judge>/ leaf (short label) ONLY under arms/*, whose artifacts one grader
│                  produced; every other family is judge-invariant (both graders inside, no judge
│                  level). NAVIGATION starts at the hand-authored results/README.md (question →
│                  headline artifacts + reading rules). Each <top>/ carries a hand-authored
│                  SUMMARY.md + auto INDEX.md;
│                  results/{METRICS_REFERENCE,LIMITATIONS}.md and results/schematics/ (the
│                  hand-authored METHOD diagrams — build_method_figures.py + CAPTIONS.md, no
│                  notebook, no judge level, in exports.PRESERVE) sit at the results root.
│                  → eda/README.md
└── history/       the only dated history: CHANGELOG_{STATUS,EDA,TRAINER}.md behind a stable index.
```

(`figures/` — the schematics' old home — and `eda/docs/` moved into `eda/results/` on 2026-08-18;
the last pre-reorg state is commit `abe5cb3`; its *code and docs* only — the pre-reorg
`results/L0|L5` renders live solely in the archival bundle `G:\My Drive\Thesis_PTO_GRPO\_git_archive\Thesis_PTO_GRPO_prerewrite_2026-08-19_b7b44ac.bundle`.)

`meetings/` moved OUT of here to the repo root — decks span experiments and now also present the
`papers/` drafts, so they sit beside `papers/` rather than inside one experiment. The deck builders
still read this experiment's `eda/results/`; they resolve it as `REPO/Exp3_PTO_GRPO/`.

## Exp3 · EDA workflow (short version — full guide in [eda/README.md](Exp3_PTO_GRPO/eda/README.md))
1. **Score:** `Run_Eval.ipynb` — its `EXPERIMENTS` registry is auto-generated from
   `eda_analysis.data.discover_arms()`, so a run is scoreable as soon as its conversations land on
   disk (empty in-flight `model_iter` dirs are skipped). Writes
   `data/eval_scores/judge=<tag>/rep=<r>/`.
2. **Analyze:** one notebook per results **family**, `notebooks/<top>/<sub>.ipynb` ↔
   `results/<top>/<sub>/` 1:1 — `arms/*` (per-arm descriptives, all four arms on one axis),
   `lookahead/*` (RQ-i, K=0 vs K=5 within each optimizer), `method/contrast` (RQ-ii — incl.
   **`headline_grid`**, THE four-arm endpoint grid with both graders side by side, each arm anchored
   to its own base), `compute/cost`
   (GPU-h + API axis, budget sweeps), `measurement/validity` (judge validity, multi-judge, and
   **`judge_saturation`** — the per-conversation agreement collapse and its SD mechanism); everything
   auto-discovers arms from disk — no registry edits anywhere. Cell 1 is always
   `EdaConfig(family="<top>/<sub>", judge=os.environ.get("EDA_JUDGE", ""))` → `notebook_setup`. The
   **FAMILY knob** sets the output root (the default arm filter is every arm; there is no VIEW); the
   orthogonal **JUDGE knob** selects which grader's scores are read and, for `arms/*` only, which
   `<judge>/` leaf is written — every other family is judge-invariant (loads both graders via
   `scores_by_judge`, exports with no judge level, ignores `EDA_JUDGE`).
3. **Regenerate:** `python tools/render_results.py` renders **everything** — `arms/*` once per grader
   in the score lake + the four judge-invariant tops once (units = (top, judge), parallel; a bare
   run can no longer leave a held-out judge's leaf stale). Subsets: `--top arms lookahead`,
   `--family lookahead/reward`, `--judge <tag>` (`--judge ""` = primary only), `--list`. → `results/<top>/<sub>/`.
   Hand-authored files (`results/<top>/SUMMARY.md`, `METRICS_REFERENCE.md`, `LIMITATIONS.md`,
   `schematics/`) are never touched.
   Run **`python -m eda_analysis._selfcheck`** after any EDA change (26 checks; `--fast` = the 12
   structural ones).

The FAMILY/JUDGE systems, `EdaConfig`, the exports API (`save_fig`/`save_table`/`save_numbers`/
`build_index`/`reset_results`/`PRESERVE`), parquet cache, output-clean policy, the package module map
and the 2026-08-18 old→new migration table are all documented in
[eda/README.md](Exp3_PTO_GRPO/eda/README.md) — not here. Eval **numbers** are not maintained here
either: see the Doc map.

## Exp3 · Diagnostic: partial-conversation oracle (reward-faithfulness)

Both trainers score *partial* conversations (slices as short as 2 turns) as the training reward, but
the thesis evaluates *full* conversations. The diagnostic — rebuilt on Exp3 data with no new oracle
calls in [notebooks/arms/training.ipynb](Exp3_PTO_GRPO/eda/notebooks/arms/training.ipynb) (per arm;
the K=0-vs-K=5 faithfulness contrast at a matched policy is in
[notebooks/lookahead/mechanism.ipynb](Exp3_PTO_GRPO/eda/notebooks/lookahead/mechanism.ipynb))
(from the per-branch `prefix` in `generations.jsonl`); the original Exp2 version motivated the MCL
knob — shows pairwise rank agreement with the final-conv score is **barely above chance at
`n_turns=2` and only clears 0.8/0.9 at ~10/~30 turns**, a structural gap well above oracle
reproducibility noise. Numbers + method:
[eda/results/METRICS_REFERENCE.md](Exp3_PTO_GRPO/eda/results/METRICS_REFERENCE.md) § 6.

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
4. **Score + EDA.** Run [Run_Eval.ipynb](Exp3_PTO_GRPO/eda/notebooks/scoring/Run_Eval.ipynb) (resume-safe; its `EXPERIMENTS` registry auto-discovers the run from disk — no registry edit) → then open the family notebooks [notebooks/arms/outcomes.ipynb](Exp3_PTO_GRPO/eda/notebooks/arms/outcomes.ipynb) (and the rest of `arms/*`, `lookahead/*`, `method/contrast`, `compute/cost`, `measurement/validity`), which likewise **auto-discover** it — or regenerate everything with `python tools/render_results.py` from `Exp3_PTO_GRPO/eda/`. See "EDA workflow".

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
  **Checkpoint validity is exactly 3 files** (`HF_TRAINER_FILES` = `adapter_model.safetensors`,
  `adapter_config.json`, `trainer_state.json`) — the project's own `eda_snapshot.jsonl` /
  `experiment_metadata.json` are NOT required, so a checkpoint missing only those still resumes.
  ⚠ `write_run_metadata` **overwrites `run_metadata.json` in place**, so a resume under changed knobs
  restamps the whole arm, earlier iterations included; copy it aside first if the old values matter.
- **K-turn look-ahead is batched.** `simulate_lookahead_batch` ([_shared/reward.py](Exp3_PTO_GRPO/code/_shared/reward.py))
  advances all B completions in lock-step — one padded batched `model.generate` per look-ahead turn —
  ~statistically equal to the legacy serial path (validated on GPU, |Δmean|=0.024, 1.5×). Knob
  `LOOKAHEAD_SUB_BATCH_SIZE` (auto-halves on OOM, kept sticky). GRPO sends
  `TRAIN_BATCH_SIZE × grad_accum` = 128 completions per optimizer step, so 128 = one chunk per
  simulated turn. ⚠ **This knob and `LOOKAHEAD_K` live in `LookaheadConfig`, which is not serialised** —
  they are mirrored onto `TrainingConfig` as audit-only fields so `run_metadata.json` records them.
  Sub-batch is in no `EXPERIMENT_NAME`, so without that mirror a change leaves no trace and silently
  makes per-iteration wall-clock non-comparable. Keep the mirror in sync when editing cell 1.
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

**Local GPU generation is viable — training is not.** [code/tools/generate_eval_convs.{py,ipynb}](Exp3_PTO_GRPO/code/tools/generate_eval_convs.py) (moved from `code/PTO_Exp3/` on 2026-08-18; run it from `code/tools/`) runs a
96-conv generate-only pass on the 12 GB local card in ~50 min at `--batch-size 6` (~16 batches),
and is API-bound there (mean GPU util 28% at batch 4 — patient calls dominate, which is why big
batches on an A100 win: they amortize the API wait across all 96 conversations). Respect the VRAM
ceiling in § Gotchas. Local *training* remains Colab-only for unrelated reasons (see the
`project-local-training-blackwell-crash` memory).

Experiment root resolution:
- **Local.** Walk up from `os.getcwd()` for `HF_key.txt`+`openai_key.txt` → typically `Exp3_PTO_GRPO/`.
- **Colab.** Trainer notebooks cd into `code/<METHOD>_Exp3/` after mounting Drive, then prepend `code/` to `sys.path` so `_shared` resolves as a sibling package.

### Auth (trainer only — `init_openai_client` / `authenticate` in [_shared/runtime.py](Exp3_PTO_GRPO/code/_shared/runtime.py))

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
figures → the topic module in `plotting/` (+ its `__init__` re-export); **a new family (a new question)** →
one entry in `config.py::FAMILIES` (+ `PER_JUDGE_TOPS` if one grader produces its artifacts) + one notebook
`notebooks/<top>/<sub>.ipynb` on the cell-1 contract (`_selfcheck`'s `family map` keeps the two 1:1;
`render_results.py` and `build_index` pick it up; a new *top* also gets a hand-authored `results/<top>/SUMMARY.md`);
a results-layout change → `exports.py` (leaf composition) + `config.py` (`FAMILIES`/`PER_JUDGE_TOPS`);
anything about **what a run COST** → `compute.py` (see below). The eight modules promoted from the paper
generators on 2026-08-18 — `lookahead`, `transfer`, `tails`, `dispersion`, `faithfulness`, `crossgen`,
`replication`, `instruments` (+ their `plotting/` twins) — take frames and return tidy DataFrames/figs,
never write to disk, seed with `constants.BOOT_SEED`, and must keep reproducing the paper's frozen
`analysis/out/*.json` fixture (the `paper fixture anchors` self-check). (`figures`/`plots` are still aliased to
`plotting`; the data-module aliases `discovery`/`personas`/`scores`/`select` were retired — use
`eda_analysis.data.*` / the top-level re-exports.)

**The COMPUTE axis (`eda_analysis/compute.py`).** Every other contrast in the EDA is
indexed by **iteration**, which is not a fixed unit of spend — a K=5 step costs ~1.9× a K=0 step and a
whole PTO iteration costs a fraction of a GRPO one. `compute.py` reconstructs GPU-hours per (arm,
iteration) from **artifact mtimes** and exposes `iso_compute_contrast` / `budget_sweep` so a lever can be
read at matched *budget*. Rendered by `notebooks/compute/cost.ipynb` into `results/compute/cost/tables/{compute_by_arm,
compute_by_iteration,iso_compute_contrast,step_multiplier,budget_sweep_<contrast>_<judge>,
budget_sweep_crossjudge{,_verdicts},iso_channels{,_selected},api_calls,api_ratio}` + figures
`{compute_trajectory,cost_breakdown,budget_sweep,api_calls}` — no `<judge>/` level: both graders are inside.
  - ⚠ **Never time a run from `iteration_metadata.json`.** `training_time_s` / `generation_time_s` /
    `pref_pair_time_s` are per-PROCESS, so a resumed iteration records only its last session
    (GRPO_LA5 iter 1 logs 14,501 s for 7.7 h of steps; PTO logs `pref_pair_time_s = 3.2 s` for a
    ~30 min build it reloaded from `pairs.csv`).
  - An iteration is billed `generate + build + train`. **`build` is PTO-only and is its DOMINANT
    phase** (5.7 of PTO_LA0's 8.1 h) — GRPO has no build because its reward computation happens
    inside the training loop, which is why per-step timings alone cannot compare the two methods.
  - `train` is timed from `training/completions/*.parquet` for GRPO (one per optimizer step) and
    from TensorBoard `wall_time` for PTO (DPOTrainer writes no per-step artifact).
  - Any mtime delta outside `(0, 3600 s)` is a resume gap or a re-synced Drive mtime and is
    **imputed at the phase median**, so the step counts once rather than being dropped or billing
    days of idle time. `n_imputed` reports how often that fired.
  - ⚠ **Iso-compute pairs DIFFERENT iterations across arms**, so `file_index` pairing is invalid
    there (personas reshuffle `seed + k + 1`); everything in the module pairs on `persona_id`.
    Pinned by the `compute axis (GPU-hours)` self-check.
  - ⚠ **Quote `budget_sweep`, not a single iso-compute row** — the lever's sign is a function of
    budget (GRPO K=5 is clearly worse at ≤18 GPU-h and only draws level at ~23–27).

**Scoring layer (`eda_analysis/scoring/` — the Run_Eval + Judge_Reliability backend):**

- **`scoring/registry.py::ORACLE_TOKEN_ALIASES`** — add new oracle-name aliases here (CSQ vs CSQ_8 etc.). `conversations._normalize_oracle_token(strict=True)` raises on unknowns; default `strict=False` lets unknowns fall through to "Other" for backward compat.
- **`scoring/registry.py::COMPOSITE_METRICS`** — add new composites (mean across multiple source columns) here. Currently holds just `Q1Q2_Mean`; the same pattern can produce `MITI_GlobalMean` etc.
- **`scoring/registry.py::EXPERIMENTS`** — registry of trained-model data locations, **auto-generated at import** by `build_experiments_from_disk()` from `eda_analysis.data.discover_arms()` (2026-07-11). New runs are picked up automatically once their conversations land; nothing to edit. (If the Drive symlinks are offline the registry is empty and a warning prints.)
- **`scoring/judge.py`** — add second-judge providers/models here (`JudgeSpec`); outputs land in `data/eval_scores/judge=<tag>/rep=<r>/`, never in another grader's partition. **Claude judges:** `json_schema` rejects `minimum`/`maximum`/`minItems`/`maxItems` (folded into `description` instead — do NOT just drop them, or the array-shaped rubrics lose their one-score-per-item guarantee), and Sonnet 5 / Opus 4.8+ need `thinking={"type":"disabled"}` or adaptive thinking eats `max_tokens`.
- **`scoring/judge_plan.py`** (FREE pre-flight, no API) — `check_rubric_parity()` is **the gate before any second-judge spend**: it verifies every constraint stripped for Claude was restated in `description` and the encodings are otherwise structurally identical. Runs automatically in `_selfcheck`. Also `prefix_report()` (which rubrics actually prompt-cache), `plan_sweep()` (coverage-aware call count, skips existing CSVs), `estimate_cost`/`sweep_report`. **Pricing lives in `JUDGE_PRICING` — verify against the billing dashboard before quoting a number.**
- **`scoring/judge_batch.py`** (PAID) — the full-sweep path via **Anthropic Message Batches (50% off)**: `submit_sweep` → `poll_batches` → `collect_batches`, three separate phases with manifests persisted under `data/eval_scores/_batches/` so collection works from a fresh kernel. `custom_id` is an opaque index into that manifest, never an encoded path (model+metric+oracle overflows the 64-char limit and a truncation collision would write a score to the wrong model's folder). Anthropic-only by design — the primary judge already has a full rep, and extra reps are cheap enough for the live path.
- **`reliability.py`** (analysis layer, disk-only) — the FREE read side of `data/eval_scores/`: ICC/agreement/contrast tables for `measurement/validity.ipynb` §1, plus the **multi-judge** layer for its §2 (`variance_components_arm` → arm vs judge-level vs arm×judge + `dependability_k1/k2`, `gain_retention`, `all_pairs_contrasts`, `sign_preservation`, `concordance_by_effect_size`). Figures in `plotting/reliability.py`. Keep the paid scoring in `scoring/judge*.py` and the presentation here, so judge results render inside `tools/render_results.py`.
  - ⚠ **Never average raw scores across judges.** The primary oracle WAS the training reward and the second judge is held out — that is train-vs-test, not two raters. The level offset is 1.2–1.7 points *and model-dependent*, so averaging applies a silent model-dependent shrinkage to every effect. Combine only contrasts or standardized quantities.
  - ⚠ **Pair on `persona_id`, not `file_index`** (`attach_persona`). The 96 personas are reshuffled each iteration, so a `file_index` join across unmatched iterations pairs unrelated conversations. Means survive it; `dz` and CIs do not.
- **Prompt caching is narrower than the gotcha below implies** (measured 2026-07-27 by `prefix_report`): only **Q1 and Q2** clear OpenAI's 1,024-token minimum. WAI-SR/CSQ-8/MI-SAT are rubric-first but too short (403–507 tok); **MITI/PCT/MICI interpolate a per-conversation utterance count into the instructions ahead of the rubric**, truncating their prefix to 138–206 tok. Documented, NOT fixed — those counts are the rate metrics' denominators, and editing the prompt would break comparability with every conversation already scored (`8 × 44 × 96 = 33,792` cells per grader as of 2026-08-25; read the live count off `results/measurement/validity/tables/multijudge_coverage.md`).

## Exp3 · Gotchas

- **HF model-card READMEs** inside `data/grpo_Exp3/runs/.../checkpoint-*/` are auto-generated — DO NOT delete or treat as project docs.
- **Pref-tree audit trail = resume marker.** PTO_Exp3 writes `iteration_N/pref_pairs/pairs.csv` per iter. Don't delete — it's both the DPO debug trail AND the Step-2 completion marker: its presence makes a restart **reload it and skip the ~41-min build** (see "Training internals" → Resume). The sibling `iteration_N/pref_pairs/_progress.json` is the in-build per-step checkpoint (auto-deleted on success; safe to delete manually to force a clean rebuild).
  ⚠ **An EMPTY marker FAILS LOUDLY — it is not a silent no-op.** A 1-byte `pairs.csv` makes a resumed iteration reload 0 pairs and skip the build, but `if not pref_pairs: raise ValueError("… produced 0 pref pairs …")` sits at function-body indentation *after* the reload/build branch ([pto_trainer.py:1788](Exp3_PTO_GRPO/code/PTO_Exp3/pto_trainer.py#L1788)), so both paths hit it before any adapter is written. Delete the file to force a clean rebuild. ⚠ The raise's own guidance ("lower `PREF_FILTER_TAU`") is a **science change mid-arm** — τ is not in `EXPERIMENT_NAME`, so acting on it silently mixes arms; delete the empty marker instead. *(This bullet claimed a "silent no-op DPO update … it looks like success" until 2026-08-17. That was never true of this code — the guard landed 2026-06-03, ~8 weeks before the `iteration_6/` event, and `history/CHANGELOG_STATUS.md:139` correctly says an empty marker "**would have**" caused it. The counterfactual lost its "would have" when it was promoted here.)*
- **"The conv dir exists" ≠ "the convs exist", AND "the dir reads as empty" ≠ "the run died".** `data/` is a Google Drive Desktop symlink, and the mount can wedge on a single folder — a populated `model_iter_N/` read as 0 files with an intermittent `WinError 1450` while all 96 convs were present in Drive the whole time; a Drive restart fixed it. **Before concluding an arm is unfinished, check the cloud** (the Drive MCP connector lists the folder directly). The alternative was a needless ~50-min regeneration.
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
- **Local offline smoke:** [code/tools/_local_smoke.py](Exp3_PTO_GRPO/code/tools/_local_smoke.py) (moved from `code/` on 2026-08-18) — from `code/`: `python tools\_local_smoke.py {stopgen|dpo|grpo|all}`. Tiny, no OpenAI; validates the stop-string bind, the DPO prompt-cap + no-OOM (grad-ckpt+precompute), and a GRPO step on the local GPU (~3 GB peak). Imports `trl` first (see above). All three PASS as of 2026-06-07.
- **Oracle prompt caching depends on the rubric-first layout.** [questionnaires.py](Exp3_PTO_GRPO/code/questionnaires.py) `get_prompt_eval_questionnaire` puts the fixed instructions + questionnaire rubric FIRST and the variable transcript LAST, so OpenAI's automatic prompt caching hits the ~1,084-token fixed prefix on every oracle call (≈50 % input discount + lower latency — matters for the oracle bill, the binding cost constraint above, even though wall-clock is GPU-bound; see next bullet). The margin over OpenAI's 1,024-token minimum is thin: **don't trim the oracle instructions/rubric or move the transcript ahead of them**, or caching silently stops (verified 2026-06-07: prefix is transcript-independent for Q1). Patient API calls auto-cache too (stable system + growing-history prefix). The therapist's local `model.generate` has **no** cross-call prefix reuse under HF — that would need vLLM (a real build here, not a flag: the look-ahead and *all* of PTO's generation use custom `model.generate`, not TRL's `use_vllm` path).
- **Where GRPO's wall-clock actually goes (MEASURED 2026-08-17 — supersedes the earlier guesses).**
  Read per-step times off the mtimes of `iteration_N/training/completions/*.parquet`, not off
  `iteration_metadata.json`: ⚠ **`training_time_s` is per-PROCESS, so a resumed iteration records
  only its last session.** LA5 iter 1 logs 14,501 s but spans 108 steps with a 40.54 h gap at step
  55 — the true cost is **7.69 h**, nearly 2× the recorded figure. Never quote `training_time_s`
  for an iteration that crashed and resumed.
  - **K=5 costs ~1.9× K=0 per step** — median ratios **1.96 / 1.96 / 1.91** at iterations 3 / 4 / 5,
    once `LOOKAHEAD_SUB_BATCH_SIZE` is 128 and the API tail has drained. ⚠ The earlier "2.4–3.0×"
    figure was **iteration 1 only**, which ran at sub-batch 64 AND carried 12 steps > 500 s; it is
    superseded. ~1.9× is the physics of 5 extra simulated turns per candidate. Per-step cost is
    now a rendered artifact (`k_step_multiplier`), so quote it from there, not from prose.
  - **`STOP_STRINGS` is NOT the multi-× slowdown** the earlier note claimed. Benchmarked locally
    (bs 8/16/32, 200 tokens, 1B bf16): string stopping costs **1.05–1.18× vs no criteria, and the
    penalty SHRINKS with batch** — the per-step criterion is a handful of small tensor ops, flat in
    batch. The real stop-string cost is **per-CALL**, not per-step: `get_vocab()` iteration order is
    unstable, so `StopStringCriteria`'s internal cache key never hits and it rebuilds a 128k-vocab
    table on every `generate()` (0.8–1.7 s per build, measured). Memoising the criteria object is
    bit-identical (same table, same per-step decisions, zero RNG consumed) and worth ~3–6% of an
    iteration. **Do NOT "fix" it by swapping to token-ID or `eos_token_id` stopping**: the markers
    are 6-token BPE sequences (`<|im_end|>` → `['<','|','im','_end','|','>']`) and not special
    tokens, so token-ID stopping matches a strict subset — a science change, and K-asymmetric.
  - **Bigger batch buys nothing.** TRL already issues ONE `generate()` per optimizer step over the
    whole `generation_batch_size` (= `per_device × steps_per_generation`), so 64×2 and 128×1 emit
    the same single call. Worse, `grad_accum` 2→1 is **not** gradient-neutral: TRL divides by it in
    `_compute_loss` and transformers divides again in `training_step` (fires because GRPOTrainer
    sets `model_accepts_loss_kwargs=False` and the identity collator leaves `num_items_in_batch`
    None), so the net scale is 1/gas² and halving gas **doubles** the accumulated gradient — enough
    to start tripping `max_grad_norm=1.0` on this arm's measured grad norms.
  - **The one real lever is the look-ahead API latency tail.** Excess-over-floor per step:
    ~87/135 steps sit at the floor, with 9 steps ~600 s longer — the `openai` default 600 s timeout
    on the look-ahead patient call. Capping helps (~1.4×) but ⚠ **a short TOTAL budget is
    dangerous**: exhausting the retry loop freezes a sim, the oracle then scores a truncated
    transcript, and under `scale_rewards="group"` one frozen sim shifts the mean AND std of its
    group of 8. The neutral shape is a **short per-attempt timeout with MORE retries** (e.g. 60 s ×
    ≥12), which lowers the freeze probability below the status quo. K=0 makes no patient calls
    during training, so anything here is K-asymmetric — record it in `run_metadata.json`.
  - ⚠ **Wall-clock is a reported number in the look-ahead paper** (the cost-asymmetry argument), so
    any speedup applied to one arm and not the other must be stamped with the iteration it started
    at, or the cost multiplier stops being comparable.

## Hardware
Local: Windows, RTX 5070 Ti (12 GB VRAM), CUDA 12.8, torch 2.11.0+cu128.
Training (both methods) is intended for Colab (GPU); EDA + Run_Eval run locally.
