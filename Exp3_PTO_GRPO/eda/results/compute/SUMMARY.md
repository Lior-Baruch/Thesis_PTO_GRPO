# Exp3 EDA Summary — `compute/` (the cost axis: GPU-hours + API calls, budget sweeps)

> ⚠ **This file is INTERPRETATION, not evidence** — a hand-authored reading of the tables under
> [`cost/tables/`](cost/tables/), written in a working session, largely by Claude. Brainstorm from
> the tables cold, not from here; quote numbers from the tables, not from here. See
> [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs".

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`cost/{figures,tables}/…` — no `<judge>/` level: every table in this top carries BOTH
graders in a `judge` column). The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

*Ported from `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` in the 2026-08-18 reorg, then
**rewritten from the tables on 2026-08-25**, after `GRPO_LA5` resumed and finished at iteration 10.
Every number below was re-read off the artifact it cites. Claims that were not merely out of date
but **wrong** are marked "(Corrected 2026-08-25: …)" where they stood, because several of them
propagated into other documents.*

---

## What this top covers

`compute/` owns the **cost axis** that every other contrast in the EDA is missing. Every other
family is indexed by *iteration*, which is not a fixed unit of spend: a K=5 GRPO optimizer step
costs ~1.9× a K=0 step, and a whole PTO iteration costs a fraction of a GRPO one — so a
matched-iteration row silently compares unequal budgets. `cost/` reconstructs GPU-hours per (arm,
iteration) from artifact mtimes ([`compute_by_arm`](cost/tables/compute_by_arm.md),
[`compute_by_iteration`](cost/tables/compute_by_iteration.md),
[`step_multiplier`](cost/tables/step_multiplier.md)), counts the API side
([`api_calls`](cost/tables/api_calls.md), [`api_ratio`](cost/tables/api_ratio.md)), and re-reads
every lever at matched *budget* ([`iso_compute_contrast`](cost/tables/iso_compute_contrast.md), the
four `budget_sweep_<contrast>_<judge>.md` pairs for `PTO_K` / `GRPO_K` / `method_K0` / `method_K5`,
the cross-judge selection tables
[`budget_sweep_crossjudge`](cost/tables/budget_sweep_crossjudge.md) /
[`_verdicts`](cost/tables/budget_sweep_crossjudge_verdicts.md), and the behaviour-channel views
[`iso_channels`](cost/tables/iso_channels.md) /
[`iso_channels_selected`](cost/tables/iso_channels_selected.md)). Figures:
[`compute_trajectory`](cost/figures/compute_trajectory.png) (+ a column variant
[`compute_trajectory_col`](cost/figures/compute_trajectory_col.png)),
[`cost_breakdown`](cost/figures/cost_breakdown.png),
[`budget_sweep`](cost/figures/budget_sweep.png), [`api_calls`](cost/figures/api_calls.png).
Backing module `eda_analysis/compute.py` (+ `tails.api_calls`); the number ledger, with every
ratio's arithmetic and the table it was read from, is
[`compute_numbers.json`](cost/tables/compute_numbers.json).

Three standing rules, all of them earned the hard way:

- ⚠ **Never time a run from `iteration_metadata.json`.** Its `*_time_s` fields are per-PROCESS, so a
  resumed iteration records only its last session. Everything here is mtime-reconstructed.
- ⚠ **Iso-compute reads a *different* iteration from each arm**, so it pairs on `persona_id`, never
  `file_index` — the 96 personas are reshuffled (`seed + k + 1`) every iteration.
- ⚠ **Quote a budget sweep, never one iso-compute row.** The sign of the look-ahead lever is a
  function of budget, and on GRPO it changes sign across the ladder.

---

## 1. The correction that matters most: the two GRPO arms do **not** cost the same

> **(Corrected 2026-08-25.)** This file previously said: *"The two GRPO arms cost the same money.
> 27.08 vs 27.91 GPU-h — 27.08 / 27.91 = 0.970, within 3% — even though one ran twice the
> iterations."* That was read when `GRPO_LA5` had stopped at iteration 5. **It is now flatly
> false.** `GRPO_LA5` resumed and finished at iteration 10 for **51.205 GPU-h**, against
> `GRPO_LA0`'s **27.906** — i.e. **51.205 / 27.906 = 1.835**, or 51.205 − 27.906 = 23.299 extra
> GPU-hours. The same passage's caveat that *"`GRPO_LA5` was stopped ~2 min into iteration 6 (one
> step, no adapter); its 'iteration 5' endpoint is its full budget, not an early stop"* is false for
> the same reason. Anything downstream that inherited "the two GRPO arms cost about the same" needs
> re-checking against [`compute_by_arm`](cost/tables/compute_by_arm.md).

**The iso-compute *rows* that framing sat on were fine; the framing was not.** The pair
`GRPO_LA5 @5` (27.08 h) vs `GRPO_LA0 @10` (27.91 h) still reads exactly as before in
[`iso_compute_contrast`](cost/tables/iso_compute_contrast.md). What changed is that 27.08 h is no
longer the K=5 arm's whole run — it is iteration 5 of 10, a *midpoint*, and the ladder continues
above it (§ 4b).

---

## 2. What the four arms cost, and why the phases differ

From [`compute_by_arm`](cost/tables/compute_by_arm.md) — all four arms, 10 trained iterations each:

| arm | GPU-h total | h / iteration | generate | build (PTO only) | train |
|---|---|---|---|---|---|
| `PTO_LA0`  | **8.119**  | 0.812 | 1.323 | **5.669** | 1.127 |
| `PTO_LA5`  | **19.681** | 1.968 | 1.370 | **16.797** | 1.514 |
| `GRPO_LA0` | **27.906** | 2.791 | 1.214 | 0.000 | **26.692** |
| `GRPO_LA5` | **51.205** | 5.120 | 0.915 | 0.000 | **50.290** |

The four cost ratios worth carrying, each with its arithmetic:

- **Look-ahead on GRPO:** 51.205 / 27.906 = 1.835 (per iteration, 5.120 / 2.791 = 1.83).
- **Look-ahead on PTO:** 19.681 / 8.119 = 2.424 (per iteration, 1.968 / 0.812 = 2.42).
- **Method at K=0:** 27.906 / 8.119 = 3.437 — GRPO costs 3.4× PTO for the same ten iterations.
- **Method at K=5:** 51.205 / 19.681 = 2.602.

**Why the phases differ — this is a mechanism, not an accident.** PTO's dominant phase is the
**preference-tree build**: 5.669 / 8.119 = 0.698 of `PTO_LA0`'s hours and
16.797 / 19.681 = 0.853 of `PTO_LA5`'s. The build runs **once per iteration** — branch, look ahead,
oracle-score, pick the pair — and the DPO update that follows is nearly free (1.127 h across all
ten iterations for `PTO_LA0`). **GRPO has no build phase at all**, because its reward is computed
*inside* the training loop, on every optimizer step, for every one of the `G` siblings:
26.692 / 27.906 = 0.96 of `GRPO_LA0`'s hours and 50.290 / 51.205 = 0.982 of `GRPO_LA5`'s sit in
`train`. **This is why per-step timings alone cannot compare the two methods** — a PTO step and a
GRPO step are not the same unit of work, and PTO's real cost is not in its steps.
[`cost_breakdown`](cost/figures/cost_breakdown.png) shows the split per arm; the numeric columns
are `build_share` / `train_share`.

**Floors.** The same table carries `total_gpu_h_floor` — 28.766 / 51.648 / 9.221 / 21.083 for
`GRPO_LA0` / `GRPO_LA5` / `PTO_LA0` / `PTO_LA5` — which substitutes the recorded
`generation_time_s` wherever the mtime span under-reads generation. The headline ratios above use
`total_gpu_h`; the floor variants move them only slightly (51.648 / 28.766 = 1.80 for GRPO's K
lever, 28.766 / 9.221 = 3.12 for the K=0 method gap, 51.648 / 21.083 = 2.45 at K=5). Read the floor
totals off the columns themselves, never from prose.

---

## 3. The per-step price of look-ahead

[`step_multiplier`](cost/tables/step_multiplier.md), GRPO median optimizer-step seconds per
iteration:

| iteration | K=0 (s) | K=5 (s) | ratio |
|---|---|---|---|
| 1 | 74.605 | 179.536 | 2.406 |
| 2 | 80.020 | 169.586 | 2.119 |
| 3 | 79.186 | 155.635 | 1.965 |
| 4 | 79.409 | 155.788 | 1.962 |
| 5 | 78.618 | 150.217 | 1.911 |
| 6 | 78.049 | 170.320 | 2.182 |
| 7 | 79.323 | 153.057 | 1.930 |
| 8 | 77.678 | 143.556 | 1.848 |
| 9 | 79.393 | 145.136 | 1.828 |
| 10 | 77.955 | 146.091 | 1.874 |

**Report ~1.9×, and say which iterations you excluded.** **Iterations 1–2 are not comparable and
must be dropped:** `GRPO_LA5` ran them at `LOOKAHEAD_SUB_BATCH_SIZE = 64` (half the settled value)
with a fat API-latency tail, which is what produces 2.406 and 2.119. Across the settled iterations
3–10 the ratio stays between 1.828 (iteration 9) and 2.182 (iteration 6); the ledger's canonical
window is iterations 3–5 (1.965 / 1.962 / 1.911). Iteration 6 is the one settled outlier, and is
also `GRPO_LA5`'s shortest iteration (70 steps in
[`compute_by_iteration`](cost/tables/compute_by_iteration.md)) — a resumed one, so its median rests
on fewer steps. ~1.9× is simply the physics of five extra simulated turns per candidate.

**PTO's K cost lands somewhere else entirely.** Its DPO step ratio is ~1.0 at every iteration
(1.003–1.035), because the DPO update never sees a look-ahead. All of PTO's K cost is in the
**build**: the build ratio runs 1.735–4.168 across the ten iterations and the whole-iteration ratio
1.407–3.831. That is § 2's phase story seen per iteration — for PTO, "the price of K" is a question
about the pref-tree build, not about the optimizer.

---

## 4. Matched **iteration** and matched **budget** are different questions — and here they disagree

Name the cost axis on every verdict. The two axes give opposite answers about which optimizer wins
at K=5, and both are correct answers to different questions.

### 4a. The endpoint (matched-iteration) picture, for reference

At iteration 10 on Q1+Q2, primary grader, from
[`../arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md`](../arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md):
`GRPO_LA5` **4.517**, `PTO_LA5` **4.307**, `PTO_LA0` **4.260**, `GRPO_LA0` **3.753**. The
PTO-vs-GRPO verdict is therefore **an interaction with K** and must never be stated without naming
K ([`../method/contrast/tables/method_paired_by_K.md`](../method/contrast/tables/method_paired_by_K.md),
persona-paired, n = 96, sign + = PTO higher):

| K | primary Δ / dz / p_holm | held-out Δ / dz / p_holm | verdict |
|---|---|---|---|
| 0 | **+0.507 / 0.729 / <.001** | **+0.609 / 1.265 / <.001** | PTO wins |
| 5 | **−0.210 / −0.356 / .001** | **−0.206 / −0.313 / .034** | GRPO wins |

Those deltas are the endpoint levels differenced: 4.260 − 3.753 = 0.507 at K=0 and
4.307 − 4.517 = −0.210 at K=5. **The price of that flip is what this whole top is about:** the arm
that wins at K=5 is the one that spent 51.205 GPU-h — 51.205 / 8.119 = 6.31× the cheapest arm's
bill — for a level gain of 4.517 − 4.260 = 0.257 over `PTO_LA0`, i.e. 51.205 − 8.119 = 43.086 extra
GPU-hours.

### 4b. GRPO's K lever at matched budget: loses small, **wins big** — report the crossover

This is the key qualitative change since the file was last written. Each arm is represented by the
best checkpoint it could have reached for the money, selected on Q1+Q2 under the scoring grader,
paired on `persona_id`, n = 96
([`budget_sweep_GRPO_K_gpt-4o-mini`](cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md) /
[`budget_sweep_GRPO_K_claude-haiku-4-5`](cost/tables/budget_sweep_GRPO_K_claude-haiku-4-5.md);
sign + = K=5 higher):

| budget (GPU-h) | primary Δ / dz / p_holm | held-out Δ / dz / p_holm |
|---|---|---|
| 7.80  | −0.088 / −0.093 / .814 | −0.060 / −0.082 / .690 |
| 13.27 | **−0.569 / −0.742 / <.001** | **−0.495 / −0.780 / <.001** |
| 18.31 | −0.143 / −0.276 / .053 | −0.051 / −0.108 / .690 |
| 23.21 | +0.038 / 0.074 / .814 | **+0.147 / 0.331 / .012** |
| 27.08 | +0.038 / 0.074 / .814 | **+0.161 / 0.310 / .020** |
| 30.53 | +0.147 / 0.241 / .063 | **+0.267 / 0.509 / <.001** |
| 35.29 | **+0.188 / 0.310 / .020** | **+0.275 / 0.489 / <.001** |
| 39.85 | **+0.188 / 0.310 / .020** | **+0.275 / 0.489 / <.001** |
| 45.43 | **+0.372 / 0.680 / <.001** | **+0.275 / 0.489 / <.001** |
| 51.20 | **+0.435 / 0.743 / <.001** | **+0.275 / 0.489 / <.001** |

**The shape — which is the artifact to quote, not any one rung — is lose, cross, win.** Look-ahead
is *clearly* worse around 13 GPU-h (it has bought fewer iterations for the money), null at 7.8 and
18.3, crosses zero between 18.3 and 23.2, becomes Holm-significant on the **held-out** judge from
23.2 and on the **primary** oracle from 35.3, and is a large effect on both by the top rung. Every
rung at or below 27.08 is **unchanged** from the earlier reading; everything from 30.53 up is new.
The curve is [`budget_sweep`](cost/figures/budget_sweep.png).

> **(Corrected 2026-08-25.)** This file previously concluded: *"look-ahead never pays below ~18
> GPU-h, and above it only the held-out grader scores it a win."* The first half survives. The
> second does not — the primary oracle scores K=5 a significant win from 35.29 GPU-h
> (Δ +0.188, dz 0.310, p_holm .020) rising to Δ +0.435 / dz 0.743 at 51.20.

⚠ **The top rungs are budget *ceilings*, not equal spend.** From 23.21 GPU-h onward the K=0
comparator stops moving: the primary grader's best `GRPO_LA0` checkpoint is I8 (22.28 h) and the
held-out grader's is I3 (8.21 h), and the whole `GRPO_LA0` run cost only 27.906 h, so there is
nothing further for it to buy. At the 51.20 rung the two sides differ by 51.200 / 22.280 = 2.30 in
realised spend. That is a legitimate "what can each configuration reach for a given budget?"
question — it is not an iso-compute one.

**The honest equal-spend rung** is the last one where
[`iso_compute_contrast`](cost/tables/iso_compute_contrast.md) reports a `budget_ratio` inside
0.9–1.1: `GRPO_LA5 @6` (30.53 h) vs `GRPO_LA0 @10` (27.91 h), ratio 27.91 / 30.53 = 0.914. There
look-ahead wins broadly on both graders — Q1+Q2 **+0.476 / dz 0.644** primary and
**+0.646 / dz 1.131** held-out, with Q1, MITI, MI-SAT, CSQ-8 and PCT all positive and
Holm-significant under both. The earlier equal-spend rung `@5` vs `@10` (27.08 vs 27.91, ratio
27.91 / 27.08 = 1.031) reads +0.289 / dz 0.359 / p_holm .019 primary and +0.540 / dz 0.838 /
p_holm <.001 held-out. The effect therefore strengthens as the ladder climbs *within* the
equal-spend region, so it is not an artefact of the ceiling.

### 4c. PTO's K lever at matched budget: it never pays on its own ladder

[`budget_sweep_PTO_K_gpt-4o-mini`](cost/tables/budget_sweep_PTO_K_gpt-4o-mini.md) /
[`budget_sweep_PTO_K_claude-haiku-4-5`](cost/tables/budget_sweep_PTO_K_claude-haiku-4-5.md), same
construction, sign + = K=5 higher:

| budget (GPU-h) | primary Δ / dz / p_holm | held-out Δ / dz / p_holm |
|---|---|---|
| 2.17  | **−0.286 / −0.299 / .046** | **−0.188 / −0.259 / .015** |
| 4.64  | **−0.700 / −0.805 / <.001** | **−0.617 / −0.943 / <.001** |
| 7.25  | **−0.546 / −0.732 / <.001** | **−0.546 / −0.759 / <.001** |
| 8.94  | **−0.372 / −0.560 / <.001** | **−0.364 / −0.614 / <.001** |
| 10.00 | **−0.243 / −0.428 / <.001** | **−0.342 / −0.738 / <.001** |
| 12.70 | **−0.243 / −0.428 / <.001** | **−0.342 / −0.738 / <.001** |
| 14.60 | **−0.174 / −0.348 / .012** | **−0.186 / −0.323 / .011** |
| 16.17 | −0.116 / −0.225 / .190 | **−0.186 / −0.323 / .011** |
| 18.03 | −0.063 / −0.115 / .867 | **−0.186 / −0.323 / .011** |
| 19.68 | +0.047 / 0.096 / .190 | **−0.186 / −0.323 / .011** |

The curve climbs monotonically toward zero and only *reaches* it at the top of `PTO_LA5`'s own
budget, where the primary oracle is null (+0.047, p_holm .190) and the held-out judge still has
K=0 ahead (−0.186, dz −0.323, p_holm .011). **On PTO, K=5 has not paid for itself at any budget the
tables cover.** Note the asymmetry with § 4b: PTO's entire K=5 run costs 19.681 GPU-h, which is
*below* the 23-GPU-h region where GRPO's lever first crosses over. So this is not evidence that
look-ahead cannot pay for PTO — only that PTO's ladder ends before the region where GRPO's does.
Nothing on disk tests the other side of that.

### 4d. PTO vs GRPO at matched budget: PTO wins the **reward** on every ladder the tables cover — and, at K=0, pays for it in MI-inconsistency

`method_K0` tops out at `PTO_LA0`'s whole run, 8.12 GPU-h, where `GRPO_LA0` has reached I2 (5.25 h)
under either grader's selection
([`budget_sweep_method_K0_gpt-4o-mini`](cost/tables/budget_sweep_method_K0_gpt-4o-mini.md) /
[`_claude-haiku-4-5`](cost/tables/budget_sweep_method_K0_claude-haiku-4-5.md); sign + = PTO higher):
**+0.900 / dz 1.086 / p_holm <.001** primary and **+0.814 / dz 1.394 / p_holm <.001** held-out.
Every rung below it is also a significant PTO win on both graders.

> ⚠ **That reward half must never be quoted without the MI-inconsistency half.** *(Restored
> 2026-08-25: the rewrite of this file dropped this counterweight and left only the reward win —
> a live caveat lost, which is worse than an out-of-date number. Re-verified below against the
> current tables.)* At the **same 8.12 GPU-h rung**, selecting on Q1+Q2 and then scoring
> **`MICI_Rate` — MI-inconsistent acts per THERAPIST TURN, oracle-coded and therefore
> grader-dependent, so both graders are given** — PTO is **worse** on each (MICI is lower-better;
> sign + = PTO higher):
> **+0.322 / dz 1.095 / p_holm <.001** primary (`PTO_LA0 I10` 0.491 vs `GRPO_LA0 I2` 0.169) and
> **+0.438 / dz 1.172 / p_holm <.001** held-out (`PTO_LA0 I9` 0.747 vs `GRPO_LA0 I2` 0.309) — same
> two tables as above, `select_metric = Q1Q2`, `eval_metric = MICI`. The true equal-spend row in
> [`iso_compute_contrast`](cost/tables/iso_compute_contrast.md) agrees (`PTO_LA0 @10` vs
> `GRPO_LA0 @3`, 8.12 vs 8.21 GPU-h, ratio 8.21 / 8.12 = 1.011): MICI **+0.261 / dz 0.904 /
> p_holm <.001** primary and **+0.418 / dz 1.280 / p_holm <.001** held-out.
> **The honest summary is: PTO buys more reward per GPU-hour, and more reward-hacking per GPU-hour
> with it. Neither half should be quoted without the other.**
>
> **This is not "GRPO is safer" — it is that 8.12 GPU-h buys GRPO only two iterations.** On the
> primary grader `GRPO_LA0`'s own `MICI_Rate` climbs from 0.169 at I2 to **0.838** by I10
> ([`../arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md`](../arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md)),
> i.e. past PTO's 0.491; the depth story is [`../arms/SUMMARY.md`](../arms/SUMMARY.md) § 3.
> ⚠ The counterweight is a **K=0** result and does not carry to K=5: at `method_K5`'s 19.68 rung,
> Q1Q2-selected then MICI-scored, both graders are null (primary −0.044 / dz −0.160 / p_holm .054;
> held-out +0.063 / dz 0.178 / p_holm .068).

`method_K5` tops out at `PTO_LA5`'s 19.68 GPU-h, where `GRPO_LA5` has reached I3 (18.31 h)
([`budget_sweep_method_K5_gpt-4o-mini`](cost/tables/budget_sweep_method_K5_gpt-4o-mini.md) /
[`_claude-haiku-4-5`](cost/tables/budget_sweep_method_K5_claude-haiku-4-5.md)):
**+0.445 / dz 0.673 / p_holm <.001** primary, **+0.149 / dz 0.295 / p_holm .007** held-out — but the
two graders are **not buying the same PTO checkpoint** at that rung: the primary's sweep takes
`PTO_LA5 I10` (19.68 h), the held-out judge's takes `PTO_LA5 I7` (14.60 h), because within its own
scores it ranks I7 above the endpoint. The matching row in
[`iso_compute_contrast`](cost/tables/iso_compute_contrast.md)
(`PTO_LA5 @10` vs `GRPO_LA5 @3`, 19.68 vs 18.31, ratio 18.31 / 19.68 = 0.930) forces the endpoint on
both. It therefore agrees on the primary (+0.445 / dz 0.673) but is **not significant on the
held-out judge** (+0.081, dz 0.132, p_holm .299).

> **(Corrected 2026-08-25.)** This file explained that gap as *"that judge rates `GRPO_LA5 @3` more
> highly than the primary does"* — **retracted twice over.** It is a level comparison **across**
> graders, which this project never makes (the offset is 1.2–1.7 points and model-dependent), and
> it is also false in direction: that judge's `GRPOExp3_LA5_I3` mean is 2.586 against the primary's
> 3.862. **The divergence is entirely on the PTO side**, and stays inside one grader: both rows use
> the same `GRPO_LA5 @3` comparator, and on the held-out judge `PTO_LA5 I7` scores 2.735
> ([`budget_sweep_method_K5_claude-haiku-4-5`](cost/tables/budget_sweep_method_K5_claude-haiku-4-5.md))
> against `PTO_LA5 I10`'s 2.667
> ([`../arms/outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md`](../arms/outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md)).
> Forcing the iso row onto I10 hands back exactly that difference: 2.735 − 2.667 = 0.068, and
> 0.149 − 0.068 = 0.081.

⚠ **Do not inflate this into "PTO always wins".** These ladders stop where PTO's money stops. Give
GRPO the 51.205 GPU-h it actually spent at K=5 and it ends at 4.517 on the primary, above both PTO
arms (§ 4a). **The defensible statement is a quality/cost trade, and all three halves have to travel
together:** *at matched BUDGET, PTO reaches a strong policy far more cheaply and beats GRPO on
reward at every budget either PTO arm can afford — while at K=0 being markedly more MI-inconsistent
per therapist turn at that same spend, on both graders; and at matched ITERATION, GRPO with
look-ahead reaches the single best endpoint in the experiment, paying roughly 6× `PTO_LA0`'s bill to
get there.*

### 4e. The honest-selection check

[`budget_sweep_crossjudge`](cost/tables/budget_sweep_crossjudge.md) re-runs every ladder under all
four (select-grader, eval-grader) combinations; `honest_selection = True` marks the two where the
grader that picked the checkpoint is not the grader that scores it. Those are the ones to quote —
same-judge selection flatters the selecting grader. Top-of-sweep verdicts
([`budget_sweep_crossjudge_verdicts`](cost/tables/budget_sweep_crossjudge_verdicts.md)):

| contrast | budget | verdict across the four select/eval combinations |
|---|---|---|
| `GRPO_K` (K5 vs K0) | 51.20 | **K=5 wins all four** (Δ +0.256 … +0.435, dz 0.384 … 0.743, all p_holm <.001) |
| `method_K0` (PTO vs GRPO) | 8.12 | **PTO wins all four** (Δ +0.759 … +0.900, dz 1.068 … 1.394, all p_holm <.001) |
| `method_K5` (PTO vs GRPO) | 19.68 | PTO wins 3 of 4; the select-primary / eval-held-out cell is null (+0.081, p_holm .075) |
| `PTO_K` (K5 vs K0) | 19.68 | K=0 wins **both cells the held-out judge *selects*** (Δ −0.186 p_holm .011 eval-held-out, −0.153 p_holm .023 eval-primary); **both cells the primary selects are null** (−0.199 p_holm .065, +0.047 p_holm .190) |

**On `PTO_K` the discriminating variable is which grader *picks* the checkpoint, not which grader
*scores* it.** The held-out judge's pick is `PTO_LA5 I7` vs `PTO_LA0 I9`, and that pair reads K=0
ahead under **either** scorer; the primary's pick is I10 vs I10, and that pair is null under either
scorer. So the finding to carry is *"K=0 wins whenever the held-out judge selects the checkpoint"*.
Of the two `honest_selection = True` cells, the one where that judge does the picking (select
held-out / eval primary) is the significant one, Δ −0.153 / p_holm .023; the other (select primary /
eval held-out) is the null at p_holm .065 — so the split runs along the select axis there too.

> **(Corrected 2026-08-25.)** This row previously read *"K=0 wins the two cells the held-out judge
> **scores**; the other two are null."* That inverts select-judge and eval-judge. In
> [`budget_sweep_crossjudge_verdicts`](cost/tables/budget_sweep_crossjudge_verdicts.md) the two
> `arm_a < arm_b` cells are (select `claude-haiku-4-5`, eval `claude-haiku-4-5`) and (select
> `claude-haiku-4-5`, eval `gpt-4o-mini`) — the **select** column is constant across them, the
> **eval** column is not. The cell the old sentence would have counted as a win, (select
> `gpt-4o-mini`, eval `claude-haiku-4-5`), is `no sig. difference` at p_holm .065.

**`GRPO_K` and `method_K0` are the two verdicts that survive every selection route.** That is what
makes "GRPO's look-ahead wins at its own top budget" worth asserting — provided the budget-ceiling
caveat from § 4b travels with it. Note that `method_K0`'s four-way win is a **reward** verdict only;
its MI-inconsistency counterweight (§ 4d) points the other way and holds under each grader's own
selection. (The crossjudge tables cover Q1+Q2 only, so the two cross-selected MICI cells are not on
disk.)

---

## 5. MI-consistency on the compute axis — name the axis and the grader

Tables: [`iso_channels`](cost/tables/iso_channels.md) (matched compute) and
[`iso_channels_selected`](cost/tables/iso_channels_selected.md) (the checkpoints an operator would
actually deploy). `MICI` in the sweep and iso tables is **`MICI_Rate` — MI-inconsistent acts per
THERAPIST TURN**; `MICI_BehaviorTotal` is the same construct **per session**. Both are oracle-coded
and therefore grader-dependent. `conv_len` and `mean_turn_len` are deterministic text measures and
are the only grader-independent channels here.

**Under the primary oracle, at its own selected checkpoints, GRPO's look-ahead looks like a
reduction in the hack.** At the 51.20 rung the primary picks `GRPO_LA5 @10` vs `GRPO_LA0 @8`:
over-praise **per session** 0.719 vs 3.875 (Δ −3.156, dz −1.279); MI-inconsistent acts **per
session** 2.906 vs 5.646 (Δ −2.740, dz −0.859); over-praise **per therapist turn** 0.051 vs 0.369
(Δ −0.318, dz −1.674) — all Holm-significant. But MITI-coded **affirmations per therapist turn**
fall too (0.029 vs 0.095, Δ −0.066, dz −0.518): the K=5 policy emits fewer coded acts of *both*
valences, and it talks longer (`conv_len` 31.896 vs 24.115 utterances, Δ +7.781, dz 0.468 —
grader-independent).

⚠ **Under the held-out judge the sign reverses, and the reason is the comparator, not the
therapist.** That judge's best-within-budget `GRPO_LA0` checkpoint is I3, not I8 — so it compares
`GRPO_LA5 @7` against a much earlier, much less hacked K=0 policy, and reports over-praise per
session **1.625 vs 0.531** (Δ +1.094, dz 0.523) and MI-inconsistent acts per session
**8.208 vs 5.448** (Δ +2.760, dz 0.625) — i.e. K=5 **worse**. Both readings are correct about their
own comparison. **A reward-hacking claim on this axis is meaningless without naming the comparator
checkpoint as well as the unit and the grader.**

⚠ **Matched-iteration and matched-budget MICI can disagree, and did.** At matched iteration 5 the
K=5 GRPO arm was *more* MI-inconsistent (0.340 vs 0.277 per therapist turn, primary,
[`../lookahead/reward/tables/k_paired_grpo_gpt-4o-mini.md`](../lookahead/reward/tables/k_paired_grpo_gpt-4o-mini.md)),
while the equal-spend pair at 27.08 vs 27.91 GPU-h has it far *less* (Δ −0.497 per therapist turn,
dz −1.339, primary) — because at equal spend its K=0 comparator has run five more iterations down
the reward-hacking curve. At matched iteration **10** the two axes now agree: 0.210 vs 0.838 per
therapist turn on the primary. Say which axis you mean.

**PTO's K lever reduces MI-inconsistency on both graders** at its own top budget, unlike its null
reward effect: at 19.68 GPU-h, Q1Q2-selected then MICI-scored, Δ −0.228 (dz −0.708) primary and
Δ −0.166 (dz −0.411) held-out, per therapist turn — and per session 3.344 vs 4.958 acts
(Δ −1.615, dz −0.446, primary). The judge-free lexical over-praise cross-check that validates the
*direction* of these oracle codes lives in the `arms/validity` artifacts behind
[`../arms/SUMMARY.md`](../arms/SUMMARY.md), not here.

---

## 6. The API side

Look-ahead's cost is not only GPU time: every K=5 candidate spends patient calls on its K-step tail
before the oracle ever sees it. [`api_calls`](cost/tables/api_calls.md) counts oracle + patient
calls per arm × training iteration (plus the final generate-only eval pass; figure
[`api_calls`](cost/figures/api_calls.png)); [`api_ratio`](cost/tables/api_ratio.md) gives the K5/K0
ratios per method and window, each with its own `arithmetic` column. Summed over all ten matched
iterations:

| quantity | GRPO K5/K0 | PTO K5/K0 |
|---|---|---|
| patient calls, total | 406,565 / 14,254 = **28.5** | 179,634 / 17,587 = **10.2** |
| oracle calls (training reward) | 289,983 / 302,541 = **0.96** | 121,806 / 99,622 = **1.22** |
| oracle input, Mchars | 4,669.5 / 2,666.6 = **1.75** | 1,849.2 / 987.1 = **1.87** |
| total API calls | 696,548 / 316,795 = **2.20** | 301,440 / 117,209 = **2.57** |

Three things to read off that:

- **The look-ahead tax is paid in patient calls, not oracle calls.** Oracle call counts are roughly
  flat — K=0 even makes slightly *more* of them on GRPO, because the number of candidates per
  iteration differs between arms (the prompt count follows the current policy's eval-conv length).
  It is the tail turns that explode: 392,766 / 136,960 = **2.87** patient tail calls per K=5
  candidate for GRPO and 159,788 / 60,384 = **2.65** for PTO, against a ceiling of 3 patient calls
  per full K=5 tail. The shortfall below 3.0 is tails that end early — 19–23% of them, almost
  always because the simulated patient closes the session; that audit lives in
  [`../lookahead/mechanism/`](../lookahead/mechanism/).
- **The oracle *reads* far more even where it is called no more often** (1.75× / 1.87× the input
  characters), because every scored transcript now carries a K-step tail. That is the token bill,
  and it is the half that scales with K even where the call count does not.
- **GRPO's API bill dwarfs PTO's in absolute terms** — 406,565 vs 179,634 patient calls at K=5 —
  so the API axis compounds the GPU gap rather than offsetting it. API spend is the binding
  constraint on the project; the constraint itself lives in
  [`STATUS.md`](../../../../STATUS.md).

---

## 7. Caveats and traps

- **Every claim needs its axis named** — both the cost axis (matched-iteration vs matched-budget)
  and the metric axis (per therapist turn vs per session). § 4 and § 5 each contain a pair of
  individually correct statements that point opposite ways.
- **GPU-hours are reconstructed from artifact mtimes**: `training/completions/*.parquet` per
  optimizer step for GRPO, TensorBoard `wall_time` for PTO (DPOTrainer writes no per-step
  artifact). Any mtime delta outside `(0, 3600 s)` is a resume gap or a re-synced Drive mtime and is
  imputed at the phase median, so the step counts once rather than being dropped or billing days of
  idle time. `n_imputed` per arm: `GRPO_LA5` **6**, `GRPO_LA0` **3**, `PTO_LA5` **2**, `PTO_LA0`
  **1**. Totals are robust to that; a single iteration's row is less so.
  *(Corrected 2026-08-25: this file said `PTO_LA5`'s `n_imputed` is 2 and `GRPO_LA5`'s 2 — the
  latter is now 6, five further iterations having been added to that arm.)*
- ⚠ **Never quote `iteration_metadata.json` timings.** `training_time_s` / `generation_time_s` /
  `pref_pair_time_s` are per-PROCESS. `GRPO_LA5` iteration 1 logs 14,501 s against 7.69 h of actual
  steps; PTO logs `pref_pair_time_s = 3.2 s` for a ~30 min build it reloaded from `pairs.csv`.
- ⚠ **`PTO_LA5`'s per-iteration `gen_h` split is wrong for iterations 1–5** — their conversation CSV
  mtimes were batch-flushed and the time lands in iteration 6 (0.967 h). Cumulative totals are
  right; the per-iteration generate column for that arm is not.
- ⚠ **Iso-compute pairs a different iteration from each arm**, so it pairs on `persona_id`, never
  `file_index`. Means survive a `file_index` join; `dz` and CIs do not.
- ⚠ **A `budget_ratio` outside 0.9–1.1 is not an iso-compute comparison.** Many rows in
  [`iso_compute_contrast`](cost/tables/iso_compute_contrast.md) — including every `PTO_K` row past
  iteration 4 and every `GRPO_K` row past iteration 6 — fall outside it, and
  [`iso_channels`](cost/tables/iso_channels.md) flags them `iso_ok = False`. Quote the sweeps there
  instead.
- ⚠ **The auto-generated `cost/tables/CAPTIONS.md` still carries the boilerplate "GRPO_LA5 is
  right-censored".** With all four arms at iteration 10, the arm that now runs out of ladder is
  **`GRPO_LA0`**, whose entire run costs 27.906 GPU-h — which is exactly why the top half of the
  GRPO K sweep is a budget ceiling rather than equal spend (§ 4b). Read that caption's caveat as
  "the two GRPO arms have very different budget ceilings", in the direction the tables show.
- **Every endpoint is a single 96-conversation draw** and therapist decoding is unseeded, so an
  arm's level at one iteration carries sampling noise that no bootstrap over personas removes.
- **Wall-clock is a reported number in the look-ahead paper** (the cost-asymmetry argument), so any
  speedup applied to one arm and not the other must be stamped with the iteration it started at, or
  the cost multiplier stops being comparable across arms.

The matched-iteration readings this top exists to be read against are
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md) (the K lever) and
[`../method/SUMMARY.md`](../method/SUMMARY.md) (PTO vs GRPO); grader-validity questions belong to
[`../measurement/SUMMARY.md`](../measurement/SUMMARY.md). Metric definitions are in
[`../METRICS_REFERENCE.md`](../METRICS_REFERENCE.md); inference limits in
[`../LIMITATIONS.md`](../LIMITATIONS.md).

---

**Artifact note (2026-08-26).** Two GRPO-only companions for the GRPO-scoped paper:
`cost/figures/compute_trajectory_grpo` (Q1+Q2 levels vs cumulative GPU-h, the two GRPO arms) and
`cost/figures/api_calls_grpo`. The four-arm figures and every budget_sweep table are unchanged.
