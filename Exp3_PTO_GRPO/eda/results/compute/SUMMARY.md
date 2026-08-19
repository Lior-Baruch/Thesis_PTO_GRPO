# Exp3 EDA Summary — `compute/` (the cost axis: GPU-hours + API calls, budget sweeps)

*Ported from `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` on 2026-08-18 (reorg by research
question); numbers unchanged, paths rewritten.*

> ⚠ **This file is INTERPRETATION, not evidence** — a hand-authored reading of the tables under
> [`cost/tables/`](cost/tables/), written in past sessions, largely by Claude. Brainstorm from the
> tables cold, not from here; quote numbers from the tables, not from here. See
> [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs".

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`cost/{figures,tables}/…` — no `<judge>/` level: the sweep tables carry BOTH graders).
The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

## What this top covers

`compute/` owns the **cost axis** that every other contrast in the EDA is missing. Every other
family is indexed by *iteration*, which is not a fixed unit of spend: a K=5 optimizer step costs
~1.9× a K=0 step and a whole PTO iteration costs a fraction of a GRPO one, so matched-iteration
rows silently compare unequal budgets. `cost/` reconstructs GPU-hours per (arm, iteration) from
artifact mtimes ([`cost/tables/compute_by_arm.md`](cost/tables/compute_by_arm.md),
[`cost/tables/compute_by_iteration.md`](cost/tables/compute_by_iteration.md),
[`cost/tables/step_multiplier.md`](cost/tables/step_multiplier.md)), counts the API side
([`cost/tables/api_calls.md`](cost/tables/api_calls.md) / [`cost/tables/api_ratio.md`](cost/tables/api_ratio.md)), and re-reads each lever at matched *budget*
([`cost/tables/iso_compute_contrast.md`](cost/tables/iso_compute_contrast.md), the budget sweeps —
`cost/tables/budget_sweep_<contrast>_<judge>.md` for `PTO_K`, `GRPO_K`, `method_K0`, `method_K5`, each on
both graders — plus [`budget_sweep_crossjudge.md`](cost/tables/budget_sweep_crossjudge.md) /
[`_verdicts`](cost/tables/budget_sweep_crossjudge_verdicts.md) and the iso-compute channel tables
`iso_channels{,_selected}.md`); figures
[`cost/figures/compute_trajectory.png`](cost/figures/compute_trajectory.png),
[`cost/figures/cost_breakdown.png`](cost/figures/cost_breakdown.png),
[`cost/figures/budget_sweep.png`](cost/figures/budget_sweep.png). Backing module
`eda_analysis/compute.py` (+ `tails.api_calls`). In the retired tree this was `L5/SUMMARY.md` §2
(owned by the `L5` view) and the method paragraph `L0/SUMMARY.md` §2b; the reading below is the
one every section of [`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) and
[`../method/SUMMARY.md`](../method/SUMMARY.md) has to be read against.

⚠ Never time a run from `iteration_metadata.json` — its `*_time_s` fields are per-PROCESS and
undercount every resumed iteration. Iso-compute reads a *different* iteration from each arm, so it
pairs on `persona_id`, never `file_index`. **Quote a budget sweep, never one iso-compute row** — the
lever's sign is a function of budget.

---

## 2. ⚠ Read everything on TWO axes — iteration and compute

An iteration is not a fixed unit of spend. A K=5 optimizer step costs **~1.9×** a K=0 step
([`cost/tables/step_multiplier.md`](cost/tables/step_multiplier.md): 1.96 / 1.96 / 1.91 at
iterations 3 / 4 / 5), so:

- **The two GRPO arms cost the same money.** 27.08 vs 27.91 GPU-h — 27.08 / 27.91 = **0.970**,
  within 3% — even though one ran twice the iterations. **"GRPO_LA5 only reached iteration 5" is a
  statement about iteration count, not about spend.** Every matched-*iteration* row in
  [`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) §3–§4 therefore hands the K=5 arm roughly
  double the compute per cell.
- **At matched budget the K=5 arm wins on GRPO, and by more than the iteration axis shows.**
  `GRPO_LA5 @5` (27.08 h) vs `GRPO_LA0 @10` (27.91 h; budget ratio 1.031, so K=0 gets 3% *more*):

  | metric | primary Δ / dz / p_holm | held-out Δ / dz / p_holm |
  |---|---|---|
  | Q1+Q2 | **+0.289 / 0.359 / .018** | **+0.540 / 0.838 / <.001** |
  | Q1 | **+0.381 / 0.422 / .001** | **+0.944 / 1.100 / <.001** |
  | MICI *(per therapist turn, lower better)* | **−0.497 / −1.339 / <.001** | **−0.403 / −1.228 / <.001** |
  | MI-SAT | +0.080 / 0.097 / ns | **+0.273 / 0.385 / .001** |
  | MITI | +0.088 / 0.115 / ns | +0.086 / 0.203 / ns |
  | PCT | +0.004 / 0.018 / ns | +0.010 / 0.046 / ns |

  (Holm across the 9 rubrics at that budget — the family the rendered table uses.)

  ⚠ **`MICI` is `MICI_Rate` — acts per THERAPIST TURN, and the denominators differ** (11.31 vs
  12.75 turns). The per-SESSION counts point the same way and are ~13× larger:
  `MICI_BehaviorTotal` **3.45 vs 9.87** acts (primary), **7.14 vs 13.00** (held-out).

  ⚠ **The `MICI` sign here is the OPPOSITE of the matched-iteration reading in
  [`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) §3.** At equal spend the look-ahead arm is
  far *less* MI-inconsistent, because its K=0 comparator has by then run four more iterations down
  the reward-hacking curve.

  ⚠⚠ **This row is the FIXED-ENDPOINT framing, and on the primary oracle it disagrees with the
  best-checkpoint framing below.** Freezing `GRPO_LA0` at iteration 10 puts it *after* its
  4.082 → 3.753 regression (−0.33, larger than the whole +0.289 delta). Let both arms pick their
  best checkpoint within the same budget and the primary-oracle effect vanishes (dz 0.07, p .79);
  only the held-out judge survives both framings. Read the two together, or say which you mean.
- ⚠ **The lever's sign is a function of budget, not a constant.** From the GRPO K sweep
  ([`cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md`](cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md) /
  [`budget_sweep_GRPO_K_claude-haiku-4-5.md`](cost/tables/budget_sweep_GRPO_K_claude-haiku-4-5.md);
  the retired single `budget_sweep.md` carried both graders' columns), each arm at its best checkpoint
  reachable within the budget:

  | budget (GPU-h) | primary dz (K5 − K0) | held-out dz |
  |---|---|---|
  | 7.8 | −0.09 ns | −0.08 ns |
  | 13.3 | **−0.74** (K=0 far ahead) | **−0.78** |
  | 18.3 | −0.28 | −0.11 ns |
  | 23.2 | +0.07 ns | **+0.33** |
  | 27.1 | +0.07 ns | **+0.31** |

  **Quote the curve, never one row** — and note it is not monotone in significance: K=5 is
  *clearly* worse only around **13 GPU-h**, null at 7.8 and (held-out) 18.3, and at the top budget
  it is **null on the primary oracle** (dz 0.07, p .79) while **ahead on the held-out judge**
  (dz 0.31, p .007). The single defensible summary is: look-ahead never pays below ~18 GPU-h, and
  above it only the held-out grader scores it a win.
- **PTO is the cheap method, by a wide margin.** `PTO_LA0` reaches iteration 10 for **8.1** GPU-h
  against `GRPO_LA0`'s **27.9** for the same ten — 27.91 / 8.12 = **3.4× cheaper** — and scores
  higher. The cost split ([`cost/figures/cost_breakdown.png`](cost/figures/cost_breakdown.png))
  explains it: PTO's preference-tree **build** is 5.7 of its 8.1 h and happens **once per
  iteration**, whereas GRPO recomputes its reward *inside* the training loop on every step.

## 2b. PTO vs GRPO on the compute axis — 3.4× cheaper, and the split picture at matched budget
Source: [`cost/tables/compute_by_arm.md`](cost/tables/compute_by_arm.md) and
[`cost/tables/iso_compute_contrast.md`](cost/tables/iso_compute_contrast.md).

| arm | iterations | GPU-h | h / iteration |
|---|---|---|---|
| `PTO_LA0`  | 10 | **8.1** | 0.81 |
| `GRPO_LA0` | 10 | **27.9** | 2.79 |

**PTO reaches the same iteration 10 for 27.91 / 8.12 = 3.4× less compute, and scores higher.** On
the compute axis PTO dominates GRPO outright — a strictly stronger claim than the matched-iteration
one, and one that does not depend on the grader. The reason is structural rather than incidental:
PTO's preference-tree **build** (5.7 of its 8.1 h) runs *once per iteration*, whereas GRPO
recomputes its reward *inside* the training loop on every optimizer step.

⚠ **At matched BUDGET the picture is split, and worth stating honestly.** At 8.1 GPU-h — PTO's
entire run — GRPO has only reached iteration 3 (8.21 h, budget ratio 1.011). `PTO_LA0 @10` vs
`GRPO_LA0 @3`, persona-paired:

| metric | primary Δ / dz / p_holm | held-out Δ / dz / p_holm |
|---|---|---|
| Q1+Q2 | **+0.266 / 0.529 / <.001** | **+0.230 / 0.456 / .0002** |
| MICI (lower better) | **+0.261 / 0.904 / <.001** | **+0.418 / 1.280 / <.001** |
| MITI | +0.336 / 0.602 / <.001 | −0.031 / ns |

PTO wins the reward at equal spend but is **markedly worse on MI-inconsistency** there — because at
equal spend it has trained ten iterations to GRPO's three, and MI-inconsistency accumulates with
training depth in both methods ([`../arms/SUMMARY.md`](../arms/SUMMARY.md) §3). **The honest
summary is: PTO buys more reward per GPU-hour, and more reward-hacking per GPU-hour with it.**
Neither half should be quoted without the other. The matched-iteration method contrast is
[`../method/SUMMARY.md`](../method/SUMMARY.md).

## 3. The API side

The look-ahead's cost is not only GPU time: every K=5 candidate spends patient calls and an oracle
call on its K-step tail. [`cost/tables/api_calls.md`](cost/tables/api_calls.md) counts oracle + patient calls per
arm × train iteration (+ the final eval pass; figure [`cost/figures/api_calls.png`](cost/figures/api_calls.png)) and
[`cost/tables/api_ratio.md`](cost/tables/api_ratio.md) the K5/K0 ratios per method and window; what those tails actually contained (19–23% end early, almost always
because the simulated patient closes) is the tail audit in
[`../lookahead/mechanism/`](../lookahead/mechanism/). API spend is the binding constraint on the
project — the cost constraint itself lives in `STATUS.md`.

## 4. Caveats

- **Every claim needs its axis named.** Iteration and compute disagree on `MICI`'s sign for the
  GRPO K contrast, and both are correct answers to different questions.
- `PTO_LA5`'s `n_imputed` is 2 and `GRPO_LA5`'s 2 in `compute_by_arm` — two mtime intervals per arm
  were resume gaps or Drive re-syncs and were back-filled at the phase median. The totals are
  robust to that; a single iteration's row is less so.
- GPU-hours are reconstructed from artifact mtimes (`training/completions/*.parquet` for GRPO,
  TensorBoard `wall_time` for PTO); any mtime delta outside `(0, 3600 s)` is imputed at the phase
  median. Wall-clock is a reported number in the look-ahead paper (the cost-asymmetry argument), so
  any speedup applied to one arm and not the other must be stamped with the iteration it started at.
- ⚠ The 27.9 vs 27.1 GPU-h GRPO budgets are the *realised* budgets — `GRPO_LA5` was stopped ~2 min
  into iteration 6 (one step, no adapter); its "iteration 5" endpoint is its full budget, not an
  early stop.
- **Every endpoint is a single 96-conversation draw**; therapist decoding is unseeded. Iso-compute
  cells pair different iterations of the two arms on `persona_id`.
