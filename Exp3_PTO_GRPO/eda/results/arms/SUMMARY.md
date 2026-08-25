# Exp3 EDA Summary — `arms/` (per-arm descriptives, all four arms on one axis)

*Rewritten 2026-08-25 against the complete four-arm grid. Every number below was re-read off the
table it cites; the pre-2026-08-25 reading (written while `GRPO_LA5` was right-censored) is
superseded, and the places where it was **wrong** rather than merely out of date are flagged inline.*

> ⚠ **This file is INTERPRETATION, not evidence.** It is a hand-authored reading of the tables under
> [`*/tables/`](INDEX.md) — written in past sessions, largely by Claude. The tables are the evidence;
> this is a claim *about* them. Two consequences:
>
> - **Do not brainstorm framings from this file.** Read the tables cold first, write your own
>   candidates down, and only then read this and diff. Otherwise the section headers below become
>   the option space. See [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs".
> - **Do not quote a number from here into a paper.** Open the table it cites, and check it cites
>   the right one — §4 has quoted the regex question rate while pointing at the oracle-coded table.

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`<sub>/{figures,tables}/<judge>/…`); numbers are full-conversation eval (the held-out
outcome), persona-paired over the 96 shared personas, on the **primary** grader unless a held-out
column is named. The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

---

## What this top covers

`arms/` holds the **per-arm descriptives**: what each of the four arms (`PTO_LA0`, `PTO_LA5`,
`GRPO_LA0`, `GRPO_LA5`) scores, how it moves across iterations, what its therapist actually does,
and whether its training signal was usable and faithful. The *contrasts* live elsewhere: PTO vs GRPO
in [`../method/SUMMARY.md`](../method/SUMMARY.md), K=0 vs K=5 in
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md), spend in
[`../compute/SUMMARY.md`](../compute/SUMMARY.md), whether the ruler is trustworthy in
[`../measurement/SUMMARY.md`](../measurement/SUMMARY.md).

**This is the one PER-JUDGE top.** Every artifact here is rendered once per grader into a `<judge>/`
leaf, so there are two of most tables and you must name which one you are quoting:

| leaf | grader | what it is |
|---|---|---|
| `gpt-4o-mini/` | **primary** | `gpt-4o-mini` — **this grader WAS the training reward.** Not a neutral referee. |
| `claude-haiku-4-5/` | **held-out** | `claude-haiku-4-5` — never touched training. |

⚠ **Never average the two, and never compare their levels** — the held-out judge's offset is
1.2–1.7 points and model-dependent. Compare only contrasts (deltas, dz) or base-relative ratios.

### The grid, now complete

| Arm | K | Iters scored | GPU-h | Notes |
|---|---|---|---|---|
| `PTO_LA0`  | 0 | 0–10 | **8.119** | pref-tree → DPO, greedy trunks, MCL=12, training oracle = Q1+Q2 |
| `PTO_LA5`  | 5 | 0–10 | **19.681** | |
| `GRPO_LA0` | 0 | 0–10 | **27.906** | group-relative, MCL=12, training oracle = Q1+Q2 |
| `GRPO_LA5` | 5 | 0–10 | **51.205** | finished 2026-08-25; the most expensive arm by a wide margin |

Eleven matched model states per arm (base + iterations 1–10), 4 × 11 = 44 states, each scored by
both graders on 8 instruments over 96 personas ⇒ 8 × 44 × 96 = 33,792 cells per grader
([`../measurement/validity/tables/multijudge_coverage.md`](../measurement/validity/tables/multijudge_coverage.md),
every row 96/96 complete). `k_iters = 11` for all four arms in
[`stats/tables/gpt-4o-mini/friedman_omnibus.md`](stats/tables/gpt-4o-mini/friedman_omnibus.md), and
[`validity/tables/gpt-4o-mini/session_end_reasons.md`](validity/tables/gpt-4o-mini/session_end_reasons.md)
counts 11 × 96 = 1,056 conversations for every arm (`GRPO_LA0` 260 + 793 + 3 = 1,056; `GRPO_LA5`
239 + 814 + 3 = 1,056; `PTO_LA0` 178 + 871 + 7 = 1,056; `PTO_LA5` 226 + 826 + 4 = 1,056). GPU-h from
[`../compute/cost/tables/compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md).

> **(Corrected 2026-08-25.)** This table used to read `GRPO_LA5` as `0–5`, **27.1** GPU-h, "stopped
> ~2 min into iteration 6", and a later caveat claimed the arm was "budget-matched to `GRPO_LA0`
> within 3%". All of that is retracted: the arm ran to iteration 10 and cost 51.205 GPU-h, which is
> 51.205 / 27.906 = 1.835× `GRPO_LA0`, not a match. Every four-arm figure in this top now draws all
> four arms to iteration 10. One artifact has not caught up: the auto-caption for
> `session_end_reasons` in
> [`validity/tables/gpt-4o-mini/CAPTIONS.md`](validity/tables/gpt-4o-mini/CAPTIONS.md) still says
> "GRPO K=5 contributes fewer conversations because it is right-censored" — the table it describes
> refutes it (1,056 rows for every arm). Trust the table.

**Metrics.** Five **global-evaluation rubrics** (Q1Q2, WAI-SR, CSQ-8, MI-SAT, MITI) — an *empirical*
halo cluster, not one official construct; provenance incl. the CLPsych-2024 Q1/Q2 source in
[`METRICS_REFERENCE.md`](../METRICS_REFERENCE.md) §1 — **plus** `PCT` (patient change-talk), `MICI`
(MI-inconsistent therapist behaviour, **lower = better**, unit = **acts per therapist turn**) and the
free MITI-proficiency ratios `R:Q` / `%CR` / `%MICO`. Reported flat, not as "orthogonal" families
(§5).

---

## 1. The endpoint: four arms, and a method verdict that depends on K

See [`outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md`](outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md),
its twin [`outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md`](outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md),
[`outcomes/figures/gpt-4o-mini/outcomes_by_model_final.png`](outcomes/figures/gpt-4o-mini/outcomes_by_model_final.png),
[`outcomes/figures/gpt-4o-mini/effect_vs_base_forest_final.png`](outcomes/figures/gpt-4o-mini/effect_vs_base_forest_final.png)
and [`stats/tables/gpt-4o-mini/main_results.md`](stats/tables/gpt-4o-mini/main_results.md).

**All four arms beat their own base on every rubric *except* `MICI`, most of them by a large
margin.** `MICI` is a row of the same table and it is **lower = better**, so its positive deltas are
losses, not gains: at iteration 10 every arm moves the *wrong* way on it — primary +0.626
(`GRPO_LA0`) / +0.278 (`PTO_LA0`) / +0.086 (`PTO_LA5`) / +0.001 (`GRPO_LA5`), held-out +0.666 /
+0.461 / +0.210 / +0.301. That is the reward-hack story, read in §3; the "every rubric went up"
sentence covers the higher-is-better rows only. Paired
vs-base Q1+Q2 deltas at iteration 10 under the primary grader
([`stats/tables/gpt-4o-mini/main_results.md`](stats/tables/gpt-4o-mini/main_results.md), `target=final`):
`GRPO_LA5` +1.554 (dz 1.518, large), `PTO_LA5` +1.303 (dz 1.353), `PTO_LA0` +1.259 (dz 1.429),
`GRPO_LA0` +0.686 (dz 0.721, *medium* — the only non-large Q1+Q2 gain). Holm p ≈ 0 everywhere.
Friedman across the 11 states is significant for every arm × rubric, with Q1+Q2 Kendall's W
0.611 (`GRPO_LA5`) / 0.451 (`PTO_LA0` and `PTO_LA5`) / 0.326 (`GRPO_LA0`)
([`stats/tables/gpt-4o-mini/friedman_omnibus.md`](stats/tables/gpt-4o-mini/friedman_omnibus.md)).

**Endpoint levels, primary grader** (leaderboard, `target=final`, iteration 10 for all four):

| arm | Q1+Q2 | MITI | WAI-SR | CSQ-8 | MI-SAT | PCT | MICI ↓ |
|---|---|---|---|---|---|---|---|
| `GRPO_LA5` | **4.517** | 4.536 | 3.729 | 3.062 | 3.832 | 0.685 | **0.210** |
| `PTO_LA5`  | 4.307 | 4.258 | 3.536 | 2.953 | 3.710 | 0.638 | 0.264 |
| `PTO_LA0`  | 4.260 | 4.273 | 3.497 | 2.945 | 3.653 | 0.630 | 0.491 |
| `GRPO_LA0` | 3.753 | 3.922 | 3.438 | 2.773 | 3.479 | 0.574 | **0.838** |

**The PTO-vs-GRPO verdict is an interaction with K — the sign flips.** At the matched iteration-10
endpoint on Q1+Q2 (sign + = PTO higher,
[`../method/contrast/tables/method_paired_by_K.md`](../method/contrast/tables/method_paired_by_K.md)):

| K | primary | held-out |
|---|---|---|
| 0 | **+0.507** (dz 0.729, Holm p ≈ 0) | **+0.609** (dz 1.265, Holm p ≈ 0) |
| 5 | **−0.210** (dz −0.356, Holm p 0.001) | **−0.206** (dz −0.313, Holm p 0.034) |

Both graders agree on both signs. The leaderboard levels reconstruct the contrast exactly, which is
a useful cross-check that the two tops are reading the same lake: primary 4.260 - 3.753 = 0.507 and
4.517 - 4.307 = 0.210; held-out 2.866 - 2.257 = 0.609 and 2.873 - 2.667 = 0.206.

⚠ **Never state a PTO-vs-GRPO verdict without naming K,** and never without naming the **cost axis**
— the table above is matched on *iteration*, and at matched *budget* the K=5 half reverses. At
19.680 GPU-h (`PTO_LA5`'s whole run) `GRPO_LA5` has only reached iteration 3, and PTO is ahead by
+0.445 (dz 0.673) on the primary and +0.149 / +0.224 on the held-out selections
([`../compute/cost/tables/budget_sweep_crossjudge_verdicts.md`](../compute/cost/tables/budget_sweep_crossjudge_verdicts.md)).
Quote the **sweep**, not one row — the lever's sign is a function of budget, and the full sweep lives
in [`../compute/SUMMARY.md`](../compute/SUMMARY.md).

> **(Corrected 2026-08-25: this section used to headline "PTO is stronger *and* more stable", read
> off the K=0 pair while `GRPO_LA5` was censored, and quoted the paired contrast as a bare
> "PTO−GRPO +0.51". That is now a K=0 claim only. At K=5 GRPO wins on Q1+Q2 at matched iteration, on
> both graders.)**
>
> **(Corrected 2026-08-25, second pass: the opening sentence read "All four arms beat their own base
> on **every** rubric". Retracted — that was false for `MICI`, which is a row of the very table cited
> (`main_results.md`, `target=final`) and is lower-is-better: all four arms move the wrong way on it
> under **both** graders — the primary reads `GRPO_LA5` as flat (+0.001, p 0.711) where the held-out
> judge reads +0.301 *large*, which is §3's grader disagreement, not an exemption. §3 always said
> so; §1 contradicted it. The exception is now explicit.)**

## 2. `GRPO_LA0` peaks at iteration 8 and does not come back

See [`outcomes/figures/gpt-4o-mini/trajectories/trajectory_Q1Q2.png`](outcomes/figures/gpt-4o-mini/trajectories/trajectory_Q1Q2.png),
[`stats/tables/gpt-4o-mini/slope_by_arm.md`](stats/tables/gpt-4o-mini/slope_by_arm.md),
[`validity/tables/gpt-4o-mini/grpo_iter9_check.md`](validity/tables/gpt-4o-mini/grpo_iter9_check.md)
and the per-iteration levels in
[`../lookahead/reward/tables/k_means_by_iter.md`](../lookahead/reward/tables/k_means_by_iter.md).

- **The regression is real and it is terminal.** Primary Q1+Q2 by iteration for `GRPO_LA0`:
  3.067 → 3.269 → 3.359 → 3.993 → 4.004 → 3.972 → 3.966 → 4.074 → **4.082 (peak)** → 3.808 → 3.753.
  Peak-to-endpoint 4.082 - 3.753 = 0.329 as arm means; persona-paired, `it10-it8` = **−0.330
  (dz −0.426, Holm p ≈ 0)**.
- **`GRPO_LA0` is the only arm whose `best` ≠ `final` under the primary grader** — best = iteration 8
  (Q1+Q2 4.082, dz 1.220 vs base), final = iteration 10 (3.753, dz 0.721). Every other arm's best is
  its final. With GRPO at K=0, checkpoint selection is worth 4.082 - 3.753 = 0.329 of Q1+Q2, and even
  its best sits below `PTO_LA0`'s 4.260.
- **Climb rate.** OLS Q1+Q2 slope per iteration: `GRPO_LA5` 0.144, `PTO_LA5` 0.127, `PTO_LA0` 0.120,
  `GRPO_LA0` 0.072 (`slope_by_arm`, which also carries `peak_iter`: 8 for `GRPO_LA0` Q1+Q2, 10 for
  the other three).
- **Per-metric learning curves** (every metric, peaks auto-flagged) are in
  [`outcomes/figures/gpt-4o-mini/trajectories/`](outcomes/figures/gpt-4o-mini/trajectories/), with
  the held-out twin at
  [`outcomes/figures/claude-haiku-4-5/trajectories/`](outcomes/figures/claude-haiku-4-5/trajectories/).

> **(Corrected 2026-08-25: the old §2 said `GRPO_LA0` "dips at iter 9 across most metrics
> simultaneously then partially recovers at 10". On Q1+Q2 there is no recovery — `grpo_iter9_check`
> gives `it10-it9` = −0.055, dz −0.071, Holm p 1.000, and `it10-it8` = −0.330, Holm p ≈ 0. Only
> WAI-SR recovers (`it10-it9` = +0.350, dz 0.656, Holm p ≈ 0). Say "a one-iteration dip **on
> WAI-SR** on top of a monotonic Q1+Q2 decline", not "partially recovers".)**

**The held-out grader tells a harsher version of the same story, and picks a different checkpoint.**
Under `claude-haiku-4-5`, `GRPO_LA0`'s Q1+Q2 peaks at **iteration 3** (2.637) and ends at 2.257 —
2.637 - 2.257 = 0.380 lost — and its iteration-9 state (2.002) is barely above its base (1.861).
Held-out `best` iterations are **3** (`GRPO_LA0`), **7** (`GRPO_LA5` and `PTO_LA5`) and **9**
(`PTO_LA0`); primary `best` iterations are 8, 10, 10, 10
([`stats/tables/claude-haiku-4-5/main_results.md`](stats/tables/claude-haiku-4-5/main_results.md) vs
its primary twin). ⚠ **"Best iteration" is a grader-dependent quantity.** Any model-selection claim
must say which grader selected, and selecting and evaluating on the same grader is optimistic — the
`honest_selection` column in the budget-sweep verdicts table exists for exactly this.

## 3. The reward hack — lead with the marker no grader can move

See [`validity/figures/gpt-4o-mini/reward_hack_panel.png`](validity/figures/gpt-4o-mini/reward_hack_panel.png),
[`validity/figures/gpt-4o-mini/overpraise_crosscheck.png`](validity/figures/gpt-4o-mini/overpraise_crosscheck.png),
[`validity/tables/gpt-4o-mini/overpraise_crosscheck.md`](validity/tables/gpt-4o-mini/overpraise_crosscheck.md)
and its twin [`validity/tables/claude-haiku-4-5/overpraise_crosscheck.md`](validity/tables/claude-haiku-4-5/overpraise_crosscheck.md).

**Start here, not with a rubric.** `overpraise_crosscheck` carries two columns side by side:
`lex_overpraise_marker_rate`, a **regex over the transcripts**, and `MICI_OverPraiseRate`, the
**oracle-coded** rate. The lexical column is computed from text, so it is **identical in both
`<judge>/` leaves** — open the two files and the column matches row for row. That makes it the one
reward-hacking measurement that cannot be an artifact of who was grading.

⚠ **Its axis is an INCIDENCE, not a per-turn count.**
[`eda_analysis/behavior.py`](../../eda_analysis/behavior.py) computes it as
`sum(bool(RE.search(t)) for t in therapist_turns) / n_therapist_turns` — the **share of therapist
turns carrying at least one over-praise marker**, bounded in [0, 1]. So 0.671 reads "**67% of
`GRPO_LA0`'s therapist turns contain an over-praise marker**", *not* "0.671 markers per turn", and
every ratio drawn off it below is a ratio of incidences.

⚠ **And it is a direction check, not a primary metric.** `behavior.py`'s own docstring keeps the
`lex_*` regexes "ONLY as a sanity-check that *validates the direction of*" the oracle-coded
`MICI_OverPraise` counts, and deliberately excludes them from `_BEHAVIOR_METRICS`. What is
load-bearing here is therefore the **agreement** of the two columns, not the absolute lexical value:
over the same rows the oracle-coded `MICI_OverPraiseRate` runs 0.019 → 0.698 (`GRPO_LA0`),
0.013 → 0.299 (`PTO_LA0`), 0.026 → 0.051 (`GRPO_LA5`), 0.008 → 0.043 (`PTO_LA5`) — the same arm
ordering and the same K split as the lexical column, one axis rated by the grader and one not. Cite
the judge-free column as corroboration of the rated rate, never as a replacement for it, and quote
its numbers as incidences.

**Judge-free over-praise marker incidence — share of therapist turns carrying ≥1 marker — base
(iter 0) → endpoint (iter 10):**

| arm | base | iter 10 | change |
|---|---|---|---|
| `GRPO_LA0` | 0.003 | **0.671** | 0.671 - 0.003 = 0.668 |
| `PTO_LA0`  | 0.003 | 0.210 | 0.210 - 0.003 = 0.207 |
| `GRPO_LA5` | 0.000 | 0.064 | 0.064 - 0.000 = 0.064 |
| `PTO_LA5`  | 0.000 | 0.045 | 0.045 - 0.000 = 0.045 |

So, with no grader in the loop: **the drift is a K=0 phenomenon, and worst under GRPO.** `GRPO_LA0`
ends at 0.671 / 0.210 = 3.195× `PTO_LA0`'s incidence and 0.671 / 0.064 = 10.484× its own K=5
sibling's (ratios of **incidences** — of how often a turn over-praises at all, not of how much it
over-praises per turn). The `GRPO_LA0` series is also non-monotone — 0.275 at iteration 8, 0.093 at
9, 0.671 at 10 — so the endpoint is a jump, not the top of a ramp.

**Now the graders, and they do not agree about `GRPO_LA5`.** MICI (acts per therapist turn), paired
vs-base delta at iteration 10, from the two `main_results.md` leaves:

| arm | primary Δ (dz) | held-out Δ (dz) |
|---|---|---|
| `GRPO_LA0` | +0.626 (1.717, large) | +0.666 (1.975, large) |
| `PTO_LA0`  | +0.278 (0.780, medium) | +0.461 (0.992, large) |
| `PTO_LA5`  | +0.086 (0.312, small) | +0.210 (0.480, small) |
| `GRPO_LA5` | **+0.001 (0.006, negligible, p 0.711)** | **+0.301 (0.845, large, Holm p ≈ 0)** |

⚠ **This is the single biggest grader disagreement in the top.** The primary oracle — the grader
that *was* the reward — reads `GRPO_LA5` as having no MI-inconsistency cost at all (0.210 - 0.209 =
0.001 on levels 0.209 → 0.210). The held-out judge reads the same conversations as 0.326 → 0.628, a
0.628 - 0.326 = 0.302 rise it calls *large*. The judge-free marker corroborates the *direction* of
the held-out reading rather than the primary's flat one: over-praise-marker incidence does go
0.000 → 0.064 in that arm — as does the primary's own oracle-coded over-praise component, 0.026 →
0.051, even while its MICI *total* stays flat — a real rise, but one an order of magnitude below
`GRPO_LA0`'s. (Direction only: the marker is on a third axis and arbitrates no
level.) **The honest statement is: look-ahead greatly reduces the over-praise
drift but does not abolish it, and the primary oracle understates what is left.** Do not repeat
"K=5 is flat on MICI" without the held-out column beside it.

**What drives MICI, per arm** ([`questionnaires/tables/gpt-4o-mini/mici_behavior_by_iter.md`](questionnaires/tables/gpt-4o-mini/mici_behavior_by_iter.md),
all rates per therapist turn): for `GRPO_LA0` the total moves 0.838 - 0.211 = 0.627 while the
over-praise component alone moves 0.698 - 0.019 = 0.679 — i.e. over-praise more than accounts for the
whole rise, with the other components net *falling*. Advise-without-permission, the largest component
at base, is flat: 0.136 - 0.131 = 0.005. `PTO_LA0` is the same shape, smaller (over-praise
0.013 → 0.299). Under the held-out grader the same decomposition is steeper everywhere
(`GRPO_LA0` over-praise 0.016 → 0.826;
[`questionnaires/tables/claude-haiku-4-5/mici_behavior_by_iter.md`](questionnaires/tables/claude-haiku-4-5/mici_behavior_by_iter.md)).
The per-behaviour zoom figures are under
[`questionnaires/figures/gpt-4o-mini/mici_detail_grid.png`](questionnaires/figures/gpt-4o-mini/mici_detail_grid.png).

> **(Corrected 2026-08-25: the old §3 said MI-inconsistency "rises ~2.3× (PTO) / ~4× (GRPO)" without
> naming K or the grader. Those were the K=0 arms on the primary: `PTO_LA0` 0.491 / 0.213 = 2.305×,
> `GRPO_LA0` 0.838 / 0.211 = 3.972×. The K=5 arms are 0.264 / 0.178 = 1.483× (`PTO_LA5`) and
> 0.210 / 0.209 = 1.005× (`GRPO_LA5`) on the same grader — and 0.825 / 0.364 = 2.266×,
> 1.050 / 0.384 = 2.734×, 0.581 / 0.370 = 1.570× and 0.628 / 0.326 = 1.926× on the held-out one.
> A bare "PTO vs GRPO" MICI ratio is not a defined quantity. The old §4 line "a lexical praise-word
> count puts GRPO at ~3.5× PTO's praise rate" is now 0.671 / 0.210 = 3.195× and is `GRPO_LA0` vs
> `PTO_LA0`.)**

> **(Corrected 2026-08-25, second pass: this section named the judge-free marker's axis as "markers
> per therapist turn" — wrong, on the one metric it calls load-bearing. `behavior.py` divides a
> **boolean** per turn by the turn count, so it is the share of turns containing ≥1 marker, bounded
> in [0, 1]; "a lexical praise-word **count**" in the box above inherited the same error. The
> numbers are unchanged and the ratios still evaluate, but they are ratios of incidences. The
> section also leant on the lexical column's absolute value as primary evidence; `behavior.py` keeps
> it as a direction check on the oracle-coded rate only, so the load-bearing claim has been moved to
> the two columns' agreement.)**

## 4. Mechanism — what the therapist actually does

See [`questionnaires/figures/gpt-4o-mini/miti_detail_grid.png`](questionnaires/figures/gpt-4o-mini/miti_detail_grid.png),
[`questionnaires/tables/gpt-4o-mini/miti_detail_by_iter.md`](questionnaires/tables/gpt-4o-mini/miti_detail_by_iter.md),
[`validity/tables/gpt-4o-mini/session_shape_by_iter.md`](validity/tables/gpt-4o-mini/session_shape_by_iter.md)
and [`validity/figures/gpt-4o-mini/question_decomposition.png`](validity/figures/gpt-4o-mini/question_decomposition.png).

**Questions collapse only at K=0, and only badly under GRPO.** Deterministic `"?"` marks per
therapist turn (judge-invariant — identical in both leaves), base → iteration 10:

| arm | base | iter 10 | retained |
|---|---|---|---|
| `GRPO_LA0` | 0.829 | 0.151 | 0.151 / 0.829 = 0.182 |
| `PTO_LA0`  | 0.930 | 0.550 | 0.550 / 0.930 = 0.591 |
| `PTO_LA5`  | 0.766 | 0.616 | 0.616 / 0.766 = 0.804 |
| `GRPO_LA5` | 0.740 | 0.719 | 0.719 / 0.740 = 0.972 |

`GRPO_LA5` keeps asking questions; what changes is that it packs more of them into the turns that
ask, `q_per_q_turn` 1.282 → 1.710 (1.710 / 1.282 = 1.334).

⚠ **Quote the axis.** Those are `"?"` **marks per turn** from `session_shape_by_iter` /
[`validity/tables/gpt-4o-mini/question_rate_crosscheck.md`](validity/tables/gpt-4o-mini/question_rate_crosscheck.md),
**not** the oracle-coded MITI `B3_Q` **acts per turn**. The two disagree, and *how* they disagree is
itself a finding:

| source | `GRPO_LA0` base | iter 10 | retained |
|---|---|---|---|
| deterministic `"?"` | 0.829 | 0.151 | 0.151 / 0.829 = 0.182 |
| held-out `claude-haiku-4-5` B3_Q | 0.430 | 0.083 | 0.083 / 0.430 = 0.193 |
| primary `gpt-4o-mini` B3_Q | 0.446 | 0.319 | 0.319 / 0.446 = 0.715 |

(Base-relative ratios only — the three rows are on different units and their *levels* are not
comparable.) The held-out judge tracks the deterministic collapse almost exactly; **the primary
oracle, the one that was paying the reward, codes the arm as still asking 71.5% of its baseline
questions when the transcripts retain 18.2% of the question marks.** That is a reward-model blind
spot, measured, and it is the mechanism behind §3.

**Length inflation is universal and largest where the hack is largest.** `GRPO_LA0`'s mean therapist
turn goes 266.296 → 895.711 chars (895.711 / 266.296 = 3.364×) and sentences per turn
4.184 → 10.548 (10.548 / 4.184 = 2.521×). All four arms end in the 686–896 char band.

**Affirmations, per therapist turn** (`miti_detail_by_iter`, `B6_AF_per_turn`, base → iter 10):
`GRPO_LA0` 0.029 → 0.154, `PTO_LA0` 0.025 → 0.142, `PTO_LA5` 0.029 → 0.043, `GRPO_LA5` 0.026 → 0.029.
Same K split as everything else in §3. Endpoint `R:Q` is 1.435 / 1.114 / 0.750 / 0.951
(`GRPO_LA0` / `GRPO_LA5` / `PTO_LA0` / `PTO_LA5`).

> **(Corrected 2026-08-25: the old §4 wrote "GRPO B6_AF 0.52 → 1.98, questions B3_Q 6.4 → 4.1" with
> no axis. Those were per-**conversation** counts; the rendered table is per **therapist turn**, and
> the arm was `GRPO_LA0`. It also read "GRPO collapses to 0.15 questions/turn vs PTO's 0.55" as a
> PTO-vs-GRPO statement — it is a K=0 statement; at K=5 the same comparison is 0.719 vs 0.616, i.e.
> GRPO asks *more*.)**

**Absolute anchor — official MITI 4.2.1 competency thresholds.** Under the primary grader
([`questionnaires/figures/gpt-4o-mini/miti_proficiency_thresholds.png`](questionnaires/figures/gpt-4o-mini/miti_proficiency_thresholds.png),
[`questionnaires/tables/gpt-4o-mini/miti_threshold_verdicts.md`](questionnaires/tables/gpt-4o-mini/miti_threshold_verdicts.md)),
training moves every arm from below basic competence to fair-or-good on the two global ratings —
Relational crosses "good" in all four (`GRPO_LA5` 4.79, `PTO_LA0` 4.61, `PTO_LA5` 4.58,
`GRPO_LA0` 4.20) and `GRPO_LA5` is the only arm to cross "good" on Technical (4.29). Neither PTO arm
reaches even "fair" on `R:Q` (0.75 / 0.95) or `%CR` (0.36 / 0.37), while `GRPO_LA0`'s `R:Q` 1.43
"fair" is reached the *pathological* way — the question collapse above shrank the denominator.
⚠ **The verdicts are grader-dependent and do not survive the swap:** on the held-out leaf
([`questionnaires/tables/claude-haiku-4-5/miti_threshold_verdicts.md`](questionnaires/tables/claude-haiku-4-5/miti_threshold_verdicts.md))
**no arm reaches even "fair" on either global**, though three of four cross "good" on `%CR`.
Thresholds are the manual's expert opinion, defined for 20-minute human sessions — an anchor for one
grader's scale, not a certification.

**Which reward components the optimizer exploits**
([`questionnaires/figures/gpt-4o-mini/q2_item_deltas_final.png`](questionnaires/figures/gpt-4o-mini/q2_item_deltas_final.png),
[`questionnaires/tables/gpt-4o-mini/q2_item_deltas.md`](questionnaires/tables/gpt-4o-mini/q2_item_deltas.md),
`target=final`; Δ = difference of arm means vs that arm's own base, **not** persona-paired). The top
endpoint-Δ Q2 item is **method-linked, not K-linked**: both GRPO arms top out on *"revealed his
thinking"* (self-disclosure — `GRPO_LA0` 1.073, `GRPO_LA5` 1.813), while `PTO_LA0` tops on *"put
himself in my shoes"* (1.542, with *"made me feel cared for"* 1.531 ahead of self-disclosure) and
`PTO_LA5` on *"made me feel close"* (1.594). Q2 items 1/2/3 reward therapist self-disclosure, which
MI does not prescribe — so part of the drift traces to the reward's own composition, not only to the
optimizer.

> **(Corrected 2026-08-25: the old §4 said "self-disclosure tops only the GRPO arm", reading one
> GRPO arm and one PTO arm. With all four arms rendered it tops **both** GRPO arms and neither PTO
> arm — the split is by method, not by K.)**

**Degeneration health gate is clean.** ChatML-marker leaks are 0.000% in every (arm, iteration) row
of [`training/tables/gpt-4o-mini/degeneration_scan.md`](training/tables/gpt-4o-mini/degeneration_scan.md);
empty-after-clean peaks at 0.580% (`PTO_LA0` iter 10); the phrase-loop fraction falls from
0.448–0.490 at base to 0.000–0.010 at iteration 10 in `session_shape_by_iter`. The 2026-06-07
stop-string fixes held.

## 5. One factor, not many — and PCT is inside it

See [`validity/figures/gpt-4o-mini/factor_loadings.png`](validity/figures/gpt-4o-mini/factor_loadings.png),
[`validity/figures/gpt-4o-mini/rubric_correlation.png`](validity/figures/gpt-4o-mini/rubric_correlation.png),
[`validity/tables/gpt-4o-mini/rubric_pca_expanded.md`](validity/tables/gpt-4o-mini/rubric_pca_expanded.md)
and [`stats/tables/gpt-4o-mini/rubric_pca_pc1.md`](stats/tables/gpt-4o-mini/rubric_pca_pc1.md).

- The five halo rubrics alone give **PC1 = 91.1%** (`POOLED_halo_only`). Adding PCT, MICI and the
  three MITI ratios drops it to **54.8%** pooled — 91.1 - 54.8 = 36.3 percentage points — and
  54.8–55.7% per arm. A second dimension exists. (Part of that drop is mechanical: more, less
  correlated columns. Read it as "a second dimension exists", not as an effect size.)
- **The second dimension is not PCT.** Change-talk loads **0.402** on PC1, indistinguishable from
  the five halo rubrics (0.386–0.418). It co-moves with the halo and does not isolate MI technique.
  What sits off PC1 is MICI (0.033) and the ratios (−0.082 to 0.049).
- Consequently: **"every rubric went up" is not evidence of multi-skill improvement.** Report all
  eight instruments flat; do not describe them as orthogonal families.
- PCT does rise, most under `GRPO_LA5` (+0.214, dz 0.859, large), then `PTO_LA5` (+0.156, dz 0.664),
  `PTO_LA0` (+0.141, dz 0.620) and `GRPO_LA0` (+0.087, dz 0.363, small). The patient-side detail —
  Importance / Confidence / Readiness and the change/sustain/neutral proportions — is in
  [`questionnaires/tables/gpt-4o-mini/pct_patient_by_iter.md`](questionnaires/tables/gpt-4o-mini/pct_patient_by_iter.md),
  where `GRPO_LA5` also has the largest fall in sustain-talk share (0.446 → 0.276).

## 6. Who benefits — the heterogeneity split

See [`heterogeneity/figures/gpt-4o-mini/subgroup_endpoint_cooperation_level_final.png`](heterogeneity/figures/gpt-4o-mini/subgroup_endpoint_cooperation_level_final.png)
and [`heterogeneity/tables/gpt-4o-mini/subgroup_endpoint_means_cooperation_level.md`](heterogeneity/tables/gpt-4o-mini/subgroup_endpoint_means_cooperation_level.md)
(n = 32 personas per category; disjoint subsets, **not** persona-paired).

Endpoint Q1+Q2 by the persona's cooperation level, primary grader:

| cooperation | `GRPO_LA5` | `PTO_LA5` | `PTO_LA0` | `GRPO_LA0` |
|---|---|---|---|---|
| High | 4.939 | 4.861 | 4.904 | 4.737 |
| StartLowAndChangesToHigh | 4.637 | 4.352 | 4.210 | 3.676 |
| Low | 3.975 | 3.706 | 3.665 | 2.845 |

**The arms are nearly indistinguishable on cooperative patients and separate on difficult ones.**
`GRPO_LA5` minus `GRPO_LA0` is 4.939 - 4.737 = 0.202 on High but 3.975 - 2.845 = 1.130 on Low — i.e.
look-ahead buys GRPO roughly five times as much where the patient resists. The High row is near
ceiling for every arm (`GRPO_LA0`'s own base is already 4.372 there), so vs-base gains on High are
compressed by construction. MICI runs the other way: `GRPO_LA0`'s endpoint MI-inconsistency is
*worst* with cooperative patients (1.088 High vs 0.597 Low) — over-praise is easiest when the patient
is agreeing with you.

> **(Corrected 2026-08-25: the old §2 said "GRPO's endpoint collapse concentrates on the *Resistant*
> personas". There is no `Resistant` category — the split is `cooperation_level` ∈ {High, Low,
> StartLowAndChangesToHigh}. The direction of the old claim survives on the Low category; the label
> did not.)**

## 7. Is the training reward faithful?

See [`training/figures/gpt-4o-mini/reward_reliability_curve.png`](training/figures/gpt-4o-mini/reward_reliability_curve.png),
[`training/tables/gpt-4o-mini/reward_reliability_by_nturns.md`](training/tables/gpt-4o-mini/reward_reliability_by_nturns.md)
and [`training/tables/gpt-4o-mini/faithfulness_proxy_vs_eval.md`](training/tables/gpt-4o-mini/faithfulness_proxy_vs_eval.md).

Rank agreement between the short training-proxy reward and the full-conversation eval (0.5 = chance),
at the `MCL=12` floor and at the longest scored cut, **eval graded by the primary**:

| arm | n_turns 12 | longest cut | direction |
|---|---|---|---|
| `PTO_LA5`  | 0.892 | 0.826 (n=48) | falls |
| `PTO_LA0`  | 0.865 | 0.756 (n=48) | falls, 0.865 - 0.756 = 0.109 |
| `GRPO_LA5` | 0.886 | 0.936 (n=50) | rises |
| `GRPO_LA0` | 0.860 | 0.897 (n=50) | rises, 0.897 - 0.860 = 0.037 |

`MCL=12` keeps every arm well clear of the unreliable short-cut regime that motivated the knob (Exp2
saw agreement as low as 0.66 at `n_turns=2`). The methods differ in *shape*: GRPO's proxy gets
*more* faithful on longer conversations, PTO's less. **Re-grading the eval side with the held-out
judge lowers every curve and compresses the spread**
([`training/tables/claude-haiku-4-5/reward_reliability_by_nturns.md`](training/tables/claude-haiku-4-5/reward_reliability_by_nturns.md)):
at `n_turns=12` the four arms sit at 0.760 / 0.799 / 0.804 / 0.817 and end at 0.703–0.797 — so part
of the primary-grader faithfulness is the proxy and the eval sharing a grader.

`faithfulness_proxy_vs_eval` adds the level story (its rows are iterations 0–9 — the training proxy
has no iteration-10 row, so "of 10" means those ten states): the proxy **under**-rates `GRPO_LA0` in
**8 of 10** rows — `proxy_minus_eval` is negative at every iteration from 1 through 8 and positive
only at the base state (+0.008) and at iteration 9 (+0.103) — and **over**-rates `PTO_LA5` in
**7 of 10**, negative only at iterations 4, 5 and 8. The under-rating is steadiest in `GRPO_LA5`,
negative in 9 of 10 (positive only at iteration 8, +0.002).
The K=0-vs-K=5 faithfulness contrast *at a matched policy* is not here — it lives in
[`../lookahead/mechanism/`](../lookahead/mechanism/), read in
[`../lookahead/SUMMARY.md`](../lookahead/SUMMARY.md).

> **(Corrected 2026-08-25: the old §5 flagged a "≈0.94" figure as `GRPO_LA5`'s and warned the figure
> might be drawn on a subset of arms. The `_provenance.md` beside the figure now lists all four arms
> for both leaves, and the value at the longest cut is 0.936. The warning can be retired.)**

> **(Corrected 2026-08-25, second pass: the level paragraph said the proxy under-rates GRPO
> "throughout (negative in 9 of 10 `GRPO_LA0` rows)" and over-rates `PTO_LA5` "in 6 of 10". Both
> counts were wrong against `faithfulness_proxy_vs_eval`: `GRPO_LA0` is **8** of 10 (iterations 0 and
> 9 are positive) and `PTO_LA5` is **7** of 10. 9 of 10 is `GRPO_LA5`'s count, which is the likely
> source of the slip, and "throughout" overstated an 8/10 pattern with the endpoint iteration among
> the exceptions.)**

## 8. What the update actually pushed for — and why there is no held-out twin here

See [`preference/figures/gpt-4o-mini/update_lexical_push.png`](preference/figures/gpt-4o-mini/update_lexical_push.png),
[`preference/figures/gpt-4o-mini/generation_vs_selection.png`](preference/figures/gpt-4o-mini/generation_vs_selection.png)
and the tables under [`preference/tables/gpt-4o-mini/`](preference/tables/gpt-4o-mini/). Both methods
weight the candidates of a group and step along the weighted sum (DPO ±1 on chosen/rejected, GRPO the
standardized advantage), so rescaling to a common per-group size puts them on one probe.

⚠ **Read this first: `preference/` renders under the PRIMARY leaf only, and `training/` renders only
partly under the held-out leaf.** That is by design, not a gap. These sections measure the
*training side* — the candidate rewards the optimizer actually saw — and those rewards were produced
by the training oracle (`gpt-4o-mini`). There is no second-judge version of a number the optimizer
never had. Concretely: `preference/{tables,figures}/` has a `gpt-4o-mini/` leaf and no
`claude-haiku-4-5/` one at all; `training/` has both, but the held-out leaf carries only the two
artifacts whose **eval side** is swappable (`reward_reliability_by_nturns*` and
`faithfulness_proxy_vs_eval`) — the training-side tables (`update_*`, `advantage_signal_by_iter`,
`degeneration_scan`, `reward_distribution_by_iter`, `pto_margin_by_depth`) exist once. If you are
looking for a missing twin here, it is not missing.

- **The affirmation push is a K=0 phenomenon on the training side too**
  ([`preference/tables/gpt-4o-mini/update_lexical_push.md`](preference/tables/gpt-4o-mini/update_lexical_push.md);
  Σ(w × feature) per group ± SE, every gradient group, no embedding, no sampling). `w_affirm` by
  training iteration 1 → 10: `GRPO_LA0` −0.006 → **+0.086 ± 0.008**; `PTO_LA0` +0.008 → **+0.103 ±
  0.029 at iteration 8**, then 0.072 and 0.039 ± 0.038. The K=5 arms never develop one: `GRPO_LA5`
  stays within [−0.003, +0.008] and ends at 0.004 ± 0.007; `PTO_LA5` ends at −0.023 ± 0.016. §3's
  outcome-side hack now has an independent training-side measurement, and it carries the same K
  split.
- **`GRPO_LA0`'s push flips negative at training iteration 9** (`w_affirm` −0.015, `w_len` −112.370)
  — the same iteration the outcome grid dips, from a completely independent source.
- **The hack is a compounding loop, not a hard pull**
  ([`preference/tables/gpt-4o-mini/generation_pool_means.md`](preference/tables/gpt-4o-mini/generation_pool_means.md)).
  ⚠ **Indexing trap:** `train_iter n` samples from the *iter-start* policy, so a `train_iter 10` row
  describes the state the eval set calls `model_iter_9`, not the endpoint. On that axis, what
  `GRPO_LA0` *generates* goes `pool_affirm` 0.021 → 0.538 (0.538 / 0.021 = 25.619×) and
  `pool_overpraise` 0.003 → 0.741 (0.741 - 0.003 = 0.738), while `pool_question` collapses
  0.710 → 0.063 (0.063 / 0.710 = 0.089). Per-iteration *selection* pressure never exceeds ≈0.10.
  Small, persistent, same-signed pressure applied each iteration to an already-more-effusive policy
  — which is why the selection contrast understates the drift. `GRPO_LA5`'s pool moves far less
  (`pool_affirm` 0.042 → 0.161 = 3.833×).
- **The two losses want different things at K=0 and similar things at K=5.** Attenuation-corrected
  cosine between the two methods' pooled update directions
  ([`preference/tables/gpt-4o-mini/weighting_decomposition.md`](preference/tables/gpt-4o-mini/weighting_decomposition.md),
  the `as trained` rows): **0.317 at K=0** but **0.756 at K=5**. Swapping the *weighting rule* on the
  same groups barely moves anything (raw cosine 0.908–0.988 across all four rule-swap rows), while
  holding the rule fixed across each method's own groups leaves K=0 far apart (0.397 / 0.324
  corrected) and K=5 close (0.740 / 0.730). **So the PTO-vs-GRPO difference is about the state
  distribution the two methods train on, not about DPO vs group-relative weighting — and look-ahead
  substantially closes that distributional gap.**
  [`preference/tables/gpt-4o-mini/update_direction_cosines.md`](preference/tables/gpt-4o-mini/update_direction_cosines.md)
  makes the same point from the other side: K barely changes GRPO's direction (`GRPO_LA0` vs
  `GRPO_LA5` = 0.851 corrected) but transforms PTO's (`PTO_LA0` vs `PTO_LA5` = 0.198).

  > **(Corrected 2026-08-25: the old §6 reported 0.317 as *the* PTO-vs-GRPO cosine, "at matched K
  > and a shared oracle", without naming K — it was the K=0 row, and the K=5 row is 0.756. The old
  > note that STATUS.md had re-worded "exploration" as "state distribution" is folded into the
  > wording above.)**
- **PTO's training signal starves; GRPO's degrades late**
  ([`preference/tables/gpt-4o-mini/training_signal_yield.md`](preference/tables/gpt-4o-mini/training_signal_yield.md)).
  `PTO_LA0` branch points built fall 949 → 410 (410 / 949 = 0.432) and the τ yield 0.824 → 0.685, so
  groups that actually trained fall 782 → 281 (281 / 782 = 0.359), with the best−worst margin
  decaying 0.274 → 0.196. `PTO_LA5` builds as many branch points at the end as at the start
  (890 → 926) but its yield falls 0.935 → 0.666, so trained groups fall 832 → 617
  (617 / 832 = 0.742). A flattening PTO curve may partly be a data-starvation curve — and the two
  PTO arms starve for different reasons.

  > **(Corrected 2026-08-25: the old §6 said "GRPO trains on 94–98% of its groups throughout". True
  > of `GRPO_LA0` (0.938–0.984 across all ten iterations) but **not** of `GRPO_LA5`, whose yield
  > falls to 0.812 at iteration 9 and 0.842 at iteration 10 as its margins compress
  > (`mean_margin` 0.546 → 0.230).)**
- **The push predicts the MICI move in `GRPO_LA0`, and the *opposite* in `GRPO_LA5`**
  ([`preference/tables/gpt-4o-mini/pref_outcome_correlations.md`](preference/tables/gpt-4o-mini/pref_outcome_correlations.md)
  is an excerpt; the full 648 rows are on sheet `pref_outcome_correlations` of
  [`preference/tables/gpt-4o-mini/preference.xlsx`](preference/tables/gpt-4o-mini/preference.xlsx)).
  Read `rho_partial_iter` (train_iter partialled out of both sides; the raw ρ is confounded with
  iteration by construction). For metric MICI: `GRPO_LA0` tracks its affirmation push ρ 0.647
  (p 0.043), its length push 0.706 (p 0.023) and its over-praise push 0.617 (p 0.057). `GRPO_LA5`
  runs the other way — affirmation −0.161 (ns), over-praise **−0.635 (p 0.049)**. `PTO_LA0` shows
  nothing (−0.492, ns) while `PTO_LA5` does track over-praise (+0.700, p 0.024). n ≤ 10 iterations
  per arm, uncorrected, unit = one training iteration: descriptive, not causal.

  > **(Corrected 2026-08-25: the old §6 attributed these to "GRPO" and "PTO" as methods. They are
  > arm-level rows, and the K=5 arms do not follow their K=0 siblings — on over-praise `GRPO_LA5`'s
  > partial ρ has the opposite sign and is nominally significant. A method-level version of this
  > claim does not exist in the table.)**

> ⚠️ **Standing correction (unchanged, still load-bearing).** This section once reported
> `wins_correct` 0.65 → 0.71 as evidence that "the DPO signal is real and its latent target drifts
> toward affirmation". That number is **in-sample** — the direction was scored on the very pairs it
> was fitted on. Pooled and held out
> ([`preference/tables/gpt-4o-mini/update_direction_quality_pooled.md`](preference/tables/gpt-4o-mini/update_direction_quality_pooled.md)),
> `wins_holdout` is 0.597 / 0.566 / 0.560 / 0.550 (`GRPO_LA0` / `PTO_LA0` / `GRPO_LA5` / `PTO_LA5`)
> and the split-half cosine is 0.911 / 0.597 / 0.880 / 0.717. Per-*iteration* latent-drift artifacts
> are therefore mostly estimation noise. What survives is the pooled direction and the exact lexical
> contrasts above, which need no embedding at all.

## 9. Does the arm-level picture survive the second grader?

Mostly yes on **direction**, unreliably on **checkpoint choice**, and not at all on **absolute
verdicts**:

- **Direction survives.** All four arms beat their own base on Q1+Q2 under both graders; the
  K=0/K=5 sign flip in the method contrast reproduces on both (§1); the reward-hack ordering
  `GRPO_LA0` > `PTO_LA0` > `PTO_LA5` ≈ `GRPO_LA5` reproduces on both, and on the judge-free lexical
  marker as well (§3).
- **Magnitude does not.** Held-out MICI deltas are larger than primary in every arm, dramatically so
  for `GRPO_LA5` (+0.301 vs +0.001).
- **Checkpoint choice does not.** `best` iteration under the primary is 8 / 10 / 10 / 10; under the
  held-out judge it is 3 / 7 / 9 / 7 (§2).
- **Absolute verdicts do not.** The MITI competency thresholds are crossed by all four arms on the
  primary and by none on the held-out globals (§4).

The formal judge-validity evidence — variance decomposition, sign preservation over the pairwise
contrasts, gain retention, the MITI exception — is
[`../measurement/SUMMARY.md`](../measurement/SUMMARY.md), not here. Every arm-level reading above can
be re-checked on the held-out leaf directly: the same artifact exists at
`<sub>/{figures,tables}/claude-haiku-4-5/…`, except in `preference/` and the training-side half of
`training/` (§8). **Never average the two graders**, only compare contrasts.

## 10. Caveats and traps

- **Oracle reproducibility is measured, not assumed.** Across the four repeatability draws
  ([`../measurement/validity/tables/oracle_repeatability_icc.md`](../measurement/validity/tables/oracle_repeatability_icc.md),
  n = 96 convs × 4 reps on the anchor subset) ICC(2,1) runs **0.864–0.994** and mean |Δ|
  **0.037–0.089**; Q1/Q2 sit at 0.955–0.994, and only MICI falls below 0.90 (floor 0.864 at
  `PTOExp3_LA0_I10`). The project's informal "≈0.10 noise" band is a conservative upper bound, and
  it shrinks by ~√96 at the arm-mean level this summary reports.
- **A cheap noise sanity check.** The four arms' iteration-0 states are four independent draws from
  the *same* base policy. Their Q1+Q2 base means span 3.067 - 2.963 = 0.104 on the primary and
  1.861 - 1.830 = 0.031 on the held-out grader
  ([`stats/tables/gpt-4o-mini/main_results.md`](stats/tables/gpt-4o-mini/main_results.md) and its
  [held-out twin](stats/tables/claude-haiku-4-5/main_results.md)). Any arm-level difference of that
  size is indistinguishable from draw noise.
- **Every endpoint is a single 96-conversation draw.** Therapist decoding is unseeded, so no
  conversation set is reproducible.
- **All 96 personas are used for both training and eval**, so everything here is in-sample with
  respect to the patient distribution.
- **Absolute scores are Exp3-internal only** — not comparable to Exp2 (4-bit vs bf16 generation) and
  **never comparable across graders** (the held-out judge's level offset is 1.2–1.7 points and
  model-dependent).
- **Pair on `persona_id`, never `file_index`.** The 96 personas reshuffle every iteration; means
  survive a `file_index` join, dz and CIs do not. Everything in this top is persona-paired except
  where a caption says otherwise — the item-delta tables (`*_item_deltas`) and the heterogeneity
  subgroup tables are **differences of arm means**, not paired contrasts.
- **`arms/` is indexed by iteration, which is not a unit of spend.** A K=5 step costs ~1.9× a K=0
  step and a whole PTO iteration costs a fraction of a GRPO one — `PTO_LA0` ran ten iterations for
  8.119 GPU-h while `GRPO_LA5` needed 51.205 (51.205 / 8.119 = 6.307×). Every cross-arm reading here
  is matched-*iteration*; for matched-*budget* go to
  [`../compute/SUMMARY.md`](../compute/SUMMARY.md) and quote a **sweep**, never a single
  iso-compute row.
- **MITI arm differences are provisional** — see the MITI warning in
  [`../measurement/SUMMARY.md`](../measurement/SUMMARY.md).
- Measurement and inference limitations for the write-up: [`../LIMITATIONS.md`](../LIMITATIONS.md).
