# Exp3_PTO_GRPO — CHANGELOG (status & findings)

Dated entries for **run status and headline findings** — the narrative that used to accrete inside
the root `CLAUDE.md` § "Current status & next step". Newest first.

The **current** state these established lives in [STATUS.md](../../STATUS.md); the detailed eval
narrative lives in `eda/results/<view>/SUMMARY.md`. This file is provenance only — read it to
answer *"when did we learn this, and what did we believe before?"*, never to answer *"what is
true now?"*.

Sibling changelogs: [CHANGELOG_EDA.md](CHANGELOG_EDA.md) (the EDA package, notebooks, score lake)
and [CHANGELOG_TRAINER.md](CHANGELOG_TRAINER.md) (trainers + `code/_shared/`).

---

## 2026-08-27 (later) — P1 revived as a SECOND ARR October submission: "GRPO with look-ahead", PTO cited but never data

**Same day, second decision.** Hours after choosing "one submission" (the entry below), Lior asked
for a new paper — also for ARR October — telling the GRPO-with-look-ahead story with the PTO arms
removed: PTO may be discussed and the ICLR 2025 paper cited as the lever's origin, but the
contribution is framed as *moving* look-ahead to GRPO ("the new algo"). The archived ICLR draft
was **revived at its original path** `papers/2026_grpo_lookahead_mi/` (the archive keeps the
frozen ICLR-format snapshot): ported to ACL/ARR format, an explicit lineage paragraph and an
expanded PTO related-work paragraph added, the ICLR §5 (cost/budget — GPU-hour totals, budget
sweep, crossover rungs) deleted per the same iterations-only axis decision, replaced by a
Limitations disclosure (oracle calls ≈matched 302,541 vs 289,983; 392,766 K=5-only look-ahead
patient calls, sums over `api_calls.md`; median 1.92× per-step wall-clock) and a ≈79 GPU-h Ethics
line. Sections renumbered (no gap), body ends within the 8-page ACL limit, clean build,
`sync_figures --check` 0/0. **Open risk logged everywhere it matters: ARR's multiple-submission
policy** — the GRPO K-lever numbers appear in both same-cycle papers (subject vs interaction
cells); disjoint claims, no cross-citation of prose, but it needs the supervisors' sign-off.

## 2026-08-27 — supervisor meeting: ONE submission (P2 → ARR October), P1/ICLR dropped, iterations-only axis

**The decision.** After the 2026-08-27 meeting, Lior + supervisors chose a single Exp3 paper for
the **ARR October 2026 cycle** (submission 2026-10-12, commitment 2026-12-20; the one cycle feeds
NAACL 2027 + COLING 2027, venue chosen in December from reviews). The two ideas on the table were
the two live drafts' framings — a GRPO-look-ahead story with PTO as appendix, vs the full 2×2 —
and the **2×2 (P2, `papers/2026_pto_grpo_mi/`) won**, framed as the interaction rather than
"GRPO wins" (the held-out grader reads the two winners as tied, replicated across draws).

**P1 (`Scoring the Continuation`, GRPO-only) was archived** to
`papers/archive/2026_grpo_lookahead_mi/` and its ICLR 2027 plan (abstract 2026-09-18) dropped —
a complete, building draft retired by scope decision, not defect. Until this day STATUS carried
both as "two live papers with disjoint scopes" (P1 retargeted to ICLR on 2026-08-26; P2 drafted
the same day with a GPU-hour cost section).

**Axis decision (Lior): the paper compares on ITERATIONS ONLY** — he rejected both the GPU-hour
axis and a samples/data-count axis. P2's §5 (cost) and its budget appendix (16-cell verdict
table, budget sweep, cost breakdown, API-calls figure) were deleted; the two-winners
head-to-head was relocated into §4 as `ssec:winners` (iteration-matched, grader-conditional);
Limitations gained the matched-iterations ≠ matched-data disclosure — over ten iterations GRPO
consumed 302,541 vs PTO's 99,622 training-oracle calls at K=0 (3.04×) and 289,983 vs 121,806 at
K=5 (2.38×), sums over `compute/cost/tables/api_calls.md` — and the Ethics statement kept only
the ≈107 GPU-h total. All budget machinery remains EDA-owned under `results/compute/cost/`;
nothing was deleted from the EDA.

## 2026-08-25 (later) — iteration 10 lands; the 2x2 closes and the K x method interaction goes significant on both graders

**Same day, second event.** The entry below was written when GRPO LA5 stopped at iteration 9 with
iteration 10 at 70/136 steps. Credits were topped up, the run resumed, and iteration 10 completed:
**136/136** optimizer steps, adapter written 05:14, 136 completions parquets, `generations.jsonl`
with 2,266 rows, and the post-loop generate-only pass produced `model_iter_10` (96 conversations).
`GRPOExp3_LA5_I10` was scored on both graders (768 calls each, 0 errors). The grid is **44 states**
= 44 x 8 x 96 = **33,792 cells** per grader, and **all four arms reach iteration 10** for the first
time.

**What the matched endpoint settled.**

- **The method verdict flips sign with K, on BOTH graders.** `method_paired_by_K.md` at iteration
  10, PTO - GRPO on Q1Q2, n = 96 persona-paired: K=0 gives **+0.507** (dz 0.729, p_holm .000)
  primary and **+0.609** (dz 1.265, p_holm .000) held-out; K=5 gives **-0.210** (dz -0.356,
  p_holm .001) primary and **-0.206** (dz -0.313, p_holm .034) held-out. At iteration 9 the K=5
  reversal was significant only on the primary. Iteration 10 makes it unambiguous.
- **GRPO K=5 @10 is the best final state on BOTH graders** — Q1+Q2 **4.517** primary (vs PTO K=5
  4.307, PTO K=0 4.260, GRPO K=0 3.753) and **2.873** held-out (vs PTO K=0 2.866). It leads every
  other instrument on the primary and five of them on the held-out, and its primary best equals its
  final, i.e. it was still climbing: 4.229 (I6) -> 4.270 -> 4.254 -> 4.454 -> 4.517.
- **But best-vs-best on the held-out still favours PTO K=0**: GRPO K=5's held-out peak is I7 (2.912)
  vs PTO K=0's I9 (2.921), and `method_paired_best.md`'s held-out K=5 row is -0.177, p_holm 0.107,
  n.s. Matched-endpoint and best-vs-best now answer differently; both must be reported.
- **The K contrast is largest at the endpoint**: GRPO **-0.765** (dz -0.91) primary and **-0.616**
  (dz -1.03) held-out, both p_holm .000. PTO never significantly favours K=5 at any iteration on
  either grader.
- **The DiD converged across graders**: 0.718 (dz 0.793) primary and 0.815 (dz 0.972) held-out, ratio
  0.815 / 0.718 = 1.14x, against 1.52x at iteration 9 and 1.68x at iteration 6.
- **Cost**: GRPO LA5 finished at **51.205 GPU-h** = 51.205 / 27.906 = 1.84x GRPO K=0,
  51.205 / 8.119 = 6.31x PTO K=0, 51.205 / 19.681 = 2.60x PTO K=5. At PTO's own budget (19.68 GPU-h)
  GRPO K=5 only reaches iteration 3, so the budget sweep still favours PTO at every affordable rung.
  The honest framing is a quality/cost trade, not a winner.

**Two corrections to the entry below, both caught by re-reading the tables at 10.**

1. **The gain-retention disjointness was an iteration-9 artefact.** At 9, GRPO K=0 retention was
   0.191 [-0.064, 0.382] vs K=5's 0.686 [0.598, 0.788], `cis_disjoint = True`. At 10 it is 0.578
   [0.457, 0.723] vs 0.668 [0.587, 0.765] and the CIs **overlap**; only Q1 stays disjoint. GRPO_LA0's
   held-out score at iteration 9 was the outlier. Retention claims must name metric AND iteration.
2. **The cross-judge collapse is NOT an iteration-9 fluke and did not clear.** Per-conversation Q1
   agreement for GRPO_LA5 runs 0.941 (I5) -> 0.877 -> 0.842 -> 0.769 -> 0.487 (I9) -> **0.544 (I10)**.
   Against a 44-state median of 0.855, I9 and I10 are the two lowest-agreeing states in the whole
   experiment. The graders agree on this arm's *ranking* but disagree most about *which conversations
   are good*, precisely where the primary scores it highest.

**The censoring fix proved itself.** With no arm censored any more, the derived `SERIES` in
`faithfulness.py` emits four series all labelled `(iters 1-10)` and drops the like-for-like subset
row entirely. The hardcoded version it replaced would still be printing `iters 1-5`. The compute
mis-billing also resolved itself now that iteration 10 has an adapter — and it was never an
"adapter gate": compute.py derives last_iter/n_iters from billed parquet rows with the sole
exclusion len(steps) < 3, so any arm with >=3 timed steps and no adapter will inflate n_iters again.

**Render**: 6 units / 21 notebook executions, **1,086 s, no failures**, on a rebuilt parquet fold
(31 files, 73,344 rows). `_selfcheck`: 25 passed, 1 warn, 0 failed; `multi-judge analysis` reports
352/352 cells and 6,692 / 7,568 = 88.4% sign-preserving contrasts — flat against 88.6% at 43 states
and 88.5% at 40.

---

## 2026-08-25 — GRPO LA5 reaches iteration 9; the method verdict becomes an interaction with K

**The 2026-08-21 entry below is superseded in every particular.** GRPO LA5 did not stop at
iteration 6. It trained iterations 7, 8 and 9 to completion (106/106, 110/110, 130/130 optimizer
steps; adapters + `eda/generations.jsonl` for all three, written 2026-08-23/24) and reached
**70 of 136** steps of iteration 10 before dying at 2026-08-24 15:25 UTC.

**Two things changed about the failure story.** Failure B (Drive accepting appends but not new
files) did not recur after iteration 7 — iterations 8 and 10 lost nothing, iteration 9 lost 6 steps.
And the OpenAI failure escalated from the transient `organization_spend_limit_exceeded` to
`insufficient_quota` / `credit_balance_exhausted`, i.e. the balance actually reached zero; three
sessions died on it. Iteration 7 in fact took **seven** Colab sessions, not the four this file
recorded. The resume changed no knob: `run_metadata.json` differs from
`run_metadata_pre_resume_iter1.json` only in `started_at` and the two audit-mirror fields.

**Scoring.** `GRPOExp3_LA5_I7/I8/I9` were scored on both graders on 2026-08-25 — 3 states x 8
rubrics x 96 personas x 2 graders = 4,608 cells, the held-out side as one Message Batch of 2,304
(0 errors), the primary live (0 errors). The grid went 40 -> **43 states**, 33,024 cells per grader.

**What the three states did to the results.**

- **The method verdict flipped sign with K.** `method_paired_best.md`: at K=0, PTO - GRPO = +0.177
  (dz 0.296, p_holm .010) primary and +0.284 (dz 0.568, p_holm .000) held-out — PTO wins. At K=5 it
  is **-0.148** (dz -0.271, p_holm .016) primary and -0.177 (dz -0.270, p_holm .107 n.s.) held-out
  — GRPO wins or draws. The matched-iteration-9 version agrees: -0.257 (dz -0.426, p_holm 1.13e-4)
  primary. "PTO beats GRPO" had been stated unconditionally since the first eval; it is a **K=0**
  statement.
- **GRPO K=5 @9 became the best state in the experiment on the primary grader**, on every
  instrument: Q1+Q2 4.454 vs PTO K=5's 4.307 and PTO K=0's 4.260, and MICI 0.212, the lowest of any
  final state.
- **The held-out judge refused to confirm it**, and the disagreement is itself the finding. I6 -> I9
  moved +0.225 on the primary and -0.045 on the held-out, and per-conversation cross-judge Q1
  agreement fell 0.941 (I5) -> 0.877 -> 0.842 -> 0.769 -> **0.487** (I9) — the minimum of that
  column over all 43 states.
- **The K contrast got much larger, not smaller.** `k_table1.md` iteration 9: GRPO -0.647 (dz -0.93)
  primary and -0.856 (dz -0.90) held-out, both p_holm .000, against -0.263 / -0.533 at iteration 6.
  The DiD at 9 is 0.688 (dz 0.799) primary and 1.044 (dz 0.971) held-out.
- **Gain retention separated with disjoint CIs.** At GRPO iteration 9, K=0 retains 0.191
  [-0.064, 0.382] of its primary gain under the held-out judge while K=5 retains 0.686
  [0.598, 0.788] — `cis_disjoint = True` on Q1Q2, Q1 and Q2.
- **The compute picture worsened for GRPO K=5 and did not change the ordering.** The arm cost
  45.432 GPU-h through iteration 9 = 45.432 / 27.906 = 1.63x GRPO K=0 and 45.432 / 8.119 = 5.60x
  PTO K=0. The "budget-matched to within 3%" framing, already dead on 2026-08-21, is now off by 63%.

**Both 2026-08-21 EDA defects were fixed before this render.** The hardcoded `CENSOR_NOTE` is gone
from 16 package modules and 15 notebooks, replaced by `constants.support_note()` /
`constants.last_iterations()` derived per frame and per grader; `faithfulness.py`'s asymmetric
`SERIES` now derives its matched pair, so the like-for-like GRPO rows carry 36,632 vs 36,117 pairs
at `n_turns=12` instead of 128,176 vs 141,487. Two rendered ledger keys were renamed as a
consequence (`curve.*.iters1-5.*` -> `iters1-9.*`; `grpo_la5_censored_at_train_iter` ->
`last_train_iter_by_arm`).

---

## 2026-08-21 — GRPO LA5 stopped, not stalled; I6 lands and moves two headline claims

**The arm is not training.** GRPO LA5 has written nothing since 2026-08-20 12:09 UTC. Yesterday's
entry recorded it as "extended, in flight toward iteration 10"; that reading is retired. Iteration 7
needs `max_steps = 106` and has **40** persisted steps (37.7%), no `adapter/`.

Four Colab sessions attempted iteration 7, and the failure is **two distinct failures**, reconstructed
from the four TB event files in `iteration_7/training/tb_logs/` (one per VM) against what landed, plus
the W&B logs at `code/GRPO_Exp3/wandb/run-*_iter7/files/output.log`:

| # | started (UTC) | host | trained | persisted |
|---|---|---|---|---|
| 1 | 08-19 14:22 | `252f11bc05b1` | steps 1–**103** | 1–30 |
| 2 | 08-20 08:38 | `38c8e9447bf8` | 0 | 0 |
| 3 | 08-20 10:39 | `7393ee7109e2` | 0 | 0 |
| 4 | 08-20 11:42 | `60b484945ba8` | steps 31–**99** | 31–40 |

- **Failure A — the OpenAI organization spend cap actually bound.** Sessions 2 and 3 died at their
  first optimizer step: **384 of 395 log lines** are
  `Error code: 429 … 'code': 'organization_spend_limit_exceeded'`, on both patient and oracle calls,
  ending `Oracle batch: 0/128 succeeded (0%), 128 rewards → None`. Cleared by ~11:42. The "cost
  constraint (binding)" section had been a projection since it was written; on 2026-08-20 it stopped
  being one.
- **Failure B — Drive stopped accepting NEW file creations while appends kept working.** Sessions 1
  and 4 both logged `Oracle batch: 128/128 succeeded (100%)` — no API problem — and their TB streams
  kept flushing scalars for 3 h 28 m and 2 h 47 m past the last parquet. Lost: 103 − 30 = 73 and
  99 − 40 = 59, i.e. **132 optimizer steps ≈ 6.25 h of K=5 GRPO training computed and discarded**.
  Session 1 reached step 103 of 106 — three steps short — and banked none of it. **No artifact names
  the cause.** Persistence yield per session: 30 / 0 / 0 / 10.

**`GRPOExp3_LA5_I6` was scored on both graders on 2026-08-20** (8 × 96 = 768 cells per grader,
0 errors), taking the lake from 39 to 40 states: 40 × 8 × 96 = **30,720** cells per grader.
`_selfcheck` now reports `40/40 states complete`.

**A full re-render ran 2026-08-21** (6 units / 21 notebook executions, no failures, 2,460 s), so the
"one scoring pass behind" warning is retired. Two things it did **not** fix:

- **33 rendered files still assert "GRPO_LA5 is right-censored at iteration 5 (27.08 GPU-h)"** — a
  hardcoded `CENSOR_NOTE` in eight modules (`compute.py:636,1190`, `faithfulness.py:130`,
  `instruments.py:122`, `lookahead.py:84`, `replication.py:108`, `transfer.py:75`,
  `plotting/tails.py:54`). `compute_by_arm.md` now prints `n_iters 7 / 32.424` beside a caption
  saying 27.08. The data is right; the auto-generated prose next to it is not.
- **`faithfulness.py:110`'s `SERIES` is asymmetric** — `GRPO_LA0` "1-5" is pinned to
  `frozenset({0,1,2,3,4})` while `GRPO_LA5` "1-5" is `None` (full support). With
  `iteration_6/eda/generations.jsonl` present (1,172 rows), the column *labelled* 1-5 pools 1–6 for
  GRPO_LA5 only: 141,487 pairs vs GRPO_LA0's 128,176. A wrong value, not a caption.

**Two headline claims moved.**

1. **"The primary oracle cannot see the K×method interaction" is RETIRED.** It was an iteration-≤5
   statement. At iteration 6 the primary DiD on Q1Q2 is 0.188 − (−0.332) = **+0.520, dz 0.605,
   p_holm .000**; the held-out is +0.876, dz 0.754. The held-out is still 0.876/0.520 = **1.68×**
   larger, so "the second judge sees it more sharply" survives — "the training grader is blind to
   it" does not.
2. **Iteration 6 makes the K lever unambiguous and opposite by optimizer**, significant on *both*
   graders in *both* methods (sign + = K=0 higher): PTO +0.257 (dz 0.42) primary / +0.343 (dz 0.51)
   held-out; GRPO −0.263 (dz −0.42) / −0.533 (dz −0.55). And **GRPO K=5 @6 is the top held-out
   "final" row of all four arms** (2.903 vs PTO K=0's 2.866) and the best GRPO state on the primary
   (4.229 vs GRPO K=0's best 4.082).

**What STATUS.md said before today, and why it was wrong.**

1. **"GRPO LA5 … 30.5 and climbing", "+7 in flight".** It is not climbing; it stopped. Also
   "costs **31.98** — ~15% MORE than its K=0 sibling" was stale *and* self-contradictory with the
   same file's own 32.42 four lines earlier. Live: 30.528 through the last adapter
   (30.528/27.906 = **+9.4%**) and 32.424 billed (**+16.2%**).
2. **"`compute_by_arm.md` should report the iteration count that actually has adapters."** It does
   not, and that is coded behaviour, not a render bug: `compute.py` has **no adapter gate** (it
   excludes only at `< 3` timed steps), so it reports `last_iter 7, n_iters 7` against six adapters
   and understates `gpu_h_per_iter` by (5.088 − 4.632)/5.088 = **9.0%**.
3. **Retention "1.08 [0.94, 1.27] vs 0.73 [0.57, 0.92], disjoint".** Matched no artifact. The cited
   `k_retention_summary.md` says **1.048 [0.913, 1.223] vs 0.786 [0.587, 1.003]**, `cis_disjoint =
   False`; the 1.08 comes from `transfer.xlsx` under a *different* reference kind, and disjointness
   holds under only 2 of 5 conventions. The parenthetical "(own-base + shared-reference kinds)" was
   also false — that table is own-base only.
4. **"MITI dependability 0.55 and MICI 0.63."** Neither appears in any table. At n_arms = 40,
   `multijudge_variance_components.md` gives `dependability_k1` **MITI 0.622 / MICI 0.845** — MICI is
   much better than the standing caveat claimed.
5. **"88.4% of all 5,928 arm×metric contrasts."** 5,928 = 8 × C(39,2). At 40 states it is
   8 × C(40,2) = 8 × 780 = **6,240**, and 88.5% (ladder 88.5 / 94.5 / 97.4 / 99.5).
6. **"~$4.50 to score four states."** The double-discounted row. On the file's own $2.08/state basis
   it is 4 × $2.08 = **$8.32**. Relatedly, **"~$1.2 primary, batched"** in the replicate estimate is
   unachievable — `judge_batch.py` raises "Batch path is Anthropic-only".
7. **"Drive Desktop rewrites mtimes on sync, so the new CSVs do not read as recent."** False — the
   I6 CSVs were the newest files on disk. The render-freshness check's blindness has exactly one
   cause, the `os.walk` early `break`, not two.
8. **"~$317 spent."** No billing artifact exists anywhere in the repo, and the same commit's meeting
   deck says ≈$351. A re-derivation off `api_calls.md` puts the training-oracle line alone near $400.
   Recorded as unverifiable rather than corrected.

Also noted: **`_selfcheck` clobbers two provenance banners.** `_setup_quiet` (`_selfcheck.py:366`,
called at `:497/:524/:595`) guards against *creating* a phantom `_provenance.md` for an unrendered
family but not against *overwriting* a real one, so after any selfcheck run
`results/lookahead/reward/figures/_provenance.md` describes the check's narrow `ks=[5]`, 2-arm,
20,414-row probe instead of the render's 4-arm frame — and shows up as an unexplained git diff.
`_provenance.md` is a marker for neither mtime nor content.

And: **CLAUDE.md:412 and eda/README.md:618 still say "23 checks"**; the suite is **26**
(12 structural + 14 data + 1 opt-in probe), per `_selfcheck.py:17` and a live run (25 passed,
1 skipped).

## 2026-08-20 — drift becomes machine-checkable; three STATUS.md claims corrected; GRPO LA5 extended

**Two guards added to `_selfcheck` (23 → 26 checks; the docstring's own count was already off by
one, claiming "12 structural + 11 data" against 12 actual data checks).**

- **`score coverage (disk vs lake)`** — walks `discover_arms()` against the score lake by directory
  listing and names any state with conversations but no scores. WARNs, never FAILs: an unscored
  state is the normal condition between a training run and a scoring run. It independently
  re-found `GRPOExp3_LA5_I6` (39/40 states complete), which until now had been noticed by hand.
- **`doc drift (prose vs tables)`** — audits the 15 numeric-claim docs: every `a × b × c = d`
  evaluates and every cited artifact path resolves (both FAIL); a cited table newer than its citing
  doc WARNs. `history/` and `papers/archive/` are exempt from liveness, as are citations after a
  provenance marker ("formerly", "ported from") — those cite the past deliberately. Validated
  against 26 adversarial cases before wiring, because a check only ever seen passing is untested.
  ⚠ It does **not** catch a claim that is arithmetically fine and cites a live path but is wrong
  about the world; two of the three corrections below are exactly that class.

**What STATUS.md said before today, and why it was wrong.**

1. **"GRPO LA5 trained 1–7, 32.0 GPU-h."** Adapters exist for iterations 1–6; `iteration_7/` holds
   training artifacts but no `adapter/`, which is how `resolve_start_state` defines "not done". The
   32.0 (31.98) figure counted that *partial* iteration — `compute.py` bills any iteration with
   training artifacts, and iteration 7 had 40 steps against 70–112 for a completed one. Cumulative
   through the last completed adapter is **30.53**; including the partial it is 32.42, and both
   move while the run advances.
2. **"Scoring `I6` costs ~$1.1 (~$0.7 Haiku)."** Inconsistent with the same file's replicate
   estimate of $8.7 Haiku for exactly 5× the cells. On the receipt basis the file itself mandates,
   `$42 / 22,272 = $0.001886` per cell, so 768 cells is `768 × 0.001886 = $1.45` and the true
   all-in figure is **~$1.85**. The $0.7 was the wrong one. Neither figure was written as an
   equation, which is why the new arithmetic check could not have caught it.
3. **"Not recommended: extending GRPO LA5 to iteration 10."** The run is doing exactly that, on
   Colab, toward its configured `num_iterations = 10`. The advice was sound and is now moot; the
   ~50-vs-28 GPU-h asymmetry it warned about is a cost to absorb rather than a decision to make,
   and it means the compute axis needs **re-deriving**, not merely re-rendering.

Also settled: the replicate draw's isolation scheme (`conversations/replicate/` via `--conv-dir`,
lake folder `GRPOExp3_LA0_rep1_I10` with the draw marker as an infix *before* the `_I{k}` tail).
Writing a replicate into `conversations/full/` would collide with the primary, because
`model_iter_10` matches inside `model_iter_10_rep1_TT…` and one silently wins by glob order.

## 2026-08-18 — the look-ahead paper: two drafts retired, one live draft built, six cross-K findings

The two 2026 drafts (`2026_clpsych_mi_reward_hacking`, K=0 only; `2026_lookahead_hack_substitution`,
PTO only) were **retired to `papers/archive/`** (tracked; `.gitignore` un-ignores it like
`meetings/archive/`) and replaced by one live paper, **`papers/2026_lookahead_pto_grpo/`** — *Same
Lever, Different Optimizer: Does K-Turn Look-Ahead Help a Small MI Therapist?* — that reads all four
arms under both graders on both cost axes. It was built cold: a six-reader table sweep with the
narrative docs closed, four framing lenses, a diff against STATUS/SUMMARY/drafts, 17 load-bearing
numbers re-opened, then nine paper-local generators (`analysis/*.py`, each importing `eda_analysis`
and reproducing a tracked cell before writing), three writers, an editor pass, a 15-section
adversarial number audit and a reviewer read. Every number sits in `NUMBERS.md` → `tables/*.md` →
`analysis/out/*.json`.

Findings that did not exist as artifacts before today (all now in `papers/2026_lookahead_pto_grpo/tables/`):

- **The four-arm, persona-paired K contrast under both graders in one frame** (`k_contrast_headline_*`).
  PTO: K=0 ≥ K=5 at 8/10 iterations, Holm-sig in K=0's favour at 6 (primary) and 5/6/8 (held-out),
  carried by Q2 — the ICLR Q2-only K finding, reversed. GRPO: K=5 > K=0 at 4–5, on Q1.
- **The ICLR ordering reproduces on the poster's own 1,440 transcripts under gpt-4o-mini**
  (`crossgen_exp1_*`; the `_crossgen/` re-score of 08-12 had never been analysed): K=5 above K=0 at
  7/7 iterations under both graders, dz −0.54 vs −0.61, Spearman 0.84 across the 15 model means.
  The Exp3 null is a regime change, not a judge change.
- **Look-ahead rescales the training signal rather than sharpening it** (`dispersion_by_k_*`): margin
  and SD rise by the same factor (ratio-of-ratios 1.01–1.03), margin/SD at the 8-draw expectation
  everywhere; ~half of PTO K=5's τ-yield edge at the base policy is rescaling. **Matched-policy
  faithfulness nil** (`reward_faithfulness_*`): K0−K5 +0.004 [−0.067, 0.074] PTO, +0.015 GRPO.
- **The tail audit** (`tail_audit_*`): 19–23% of K=5 tails end early, almost always the patient
  closing (the `SESSION ENDED` marker is stripped by `handle_session_end`; a trailing-whitespace
  fingerprint survives at ~90% precision); ended-early siblings score lower within group (dz
  −0.24/−0.26) and are ~23% less likely to be the argmax; API calls ×2.1–2.3 (patient calls are the
  multiplier; oracle calls per candidate are matched).
- **Session shape reverses by optimizer** (`session_shape_stability_*`): PTO K=5 +8.3 utterances,
  GRPO K=5 −8.1; both write longer turns; the ICLR "lowest SD" claim fails (PTO K=5 more dispersed at
  10/10 iterations; SD is a ceiling artefact, Spearman(mean, SD) −0.87).
- **Held-out instruments** (`held_out_instruments_*`): WAI-SR Bond→Goal/Task composition shift (both
  graders, dz ≈ 0.44); held-out Q2 items 3/10 carry K=0's late gain (+1.1 over K=5); PCT change-talk
  rises under K=5 in Warms-up personas; Cooperative third at ceiling on the primary.
- **Cross-judge sweep** (`compute_axis_budget_sweep_crossjudge`): GRPO K=5's held-out lead at the top
  budget survives honest selection (+0.166 dz 0.27 when the primary selects); PTO K=5's deficit
  survives in sign under every select/score combination.

Prose-about-tables errors caught in the retired drafts and STATUS by the cold read (fixed here or
noted in the paper's README traps): "identical for eight iterations" (K=5 sig *higher* at iter 4),
"reward indifferent throughout" (K=0 sig ahead at 5–8), the S6 off-by-one on selection-weight
iterations, LIMITATIONS §5b "2.4 more therapist turns" (real: +4.16; 2.4 is the over-praise Δ),
"only the held-out grader can see the flip" (iteration-5-specific; the primary sees GRPO > PTO at
iter 4), "no extra branch points" (7,548 vs 6,240 over the run). `PTO_LA5 gen_h = 0.000` for iters
1–5 in `compute_by_iteration` is a batch-flushed-mtime artefact (time lands in iter 6; totals
intact).

## 2026-08-18 — GRPO LA5 lands; the COMPUTE axis reframes RQ-i and RQ-ii

**Run status.** GRPO LA5 trained iterations 1–5 and was **stopped ~2 minutes into iteration 6**
(`iteration_6/` holds one optimizer step and tb_logs, no adapter, no checkpoint). Its four new model
states were scored on **both** graders — 4 states × 8 rubrics × 96 personas = **3,072 calls per
grader, 0 errors**, ~$4.50, taking spend ~$312 → **~$317**. That completes the grid: **39 scored
model states**, 39 × 8 × 96 = **29,952 cells per grader**, no thin arm anywhere. RQ-i became a
K×method comparison for the first time.

**What we believed before this entry, and what changed.**

1. *"GRPO LA5 only reached iteration 5, so the K×method comparison is truncated."* **Superseded by
   the compute axis.** Reconstructing GPU-hours from artifact mtimes (`eda_analysis/compute.py`, new)
   shows the two GRPO arms cost **27.08 vs 27.91 GPU-h — 0.970, within 3%**. Iteration 5 is not an
   early stop for GRPO LA5; it is the arm's *full budget*. Conversely every matched-iteration table
   hands K=5 ~2× the compute per cell.
2. *"K=5 costs 2.4–3.0× K=0 per step."* **Wrong — it is ~1.9×** (median ratios 1.96 / 1.96 / 1.91 at
   iterations 3 / 4 / 5). The old figure came from iteration 1 alone, which ran at
   `LOOKAHEAD_SUB_BATCH_SIZE=64` and carried 12 API-tail steps > 500 s. Corrected in `CLAUDE.md`;
   now a rendered artifact (`k_step_multiplier`).
3. *"Look-ahead never significantly leads."* **True of PTO, false of GRPO.** GRPO K=5 leads on Q1+Q2
   at iteration 4 on both graders (dz 0.248 primary / 0.374 held-out) and at iteration 5 on the
   held-out judge (dz 0.429). At matched *budget* it leads by more (dz 0.359 / 0.838) and the `MICI`
   contrast **reverses sign** relative to the iteration axis (dz −1.339 / −1.228, K=5 far better).
4. *"PTO's lead over GRPO does not appear until iteration ≥6."* **Primary-oracle only.** On the
   held-out judge it clears Holm at **iteration 5** (+0.265, dz 0.355, p_holm .014) — the exact
   point GRPO LA5 stops, which is what made the interaction below measurable rather than
   extrapolated.

**The new results.**

- **Look-ahead reverses which method wins, and only the held-out grader sees it.** At iteration 5,
  K=0 → PTO leads (+0.265, dz 0.355, p_holm .014); K=5 → GRPO leads (−0.219, dz 0.377, p_holm .005).
  Difference-in-differences on the same 96 personas: Q1Q2 dz **0.525**, p_holm **.0001** (Q1 0.473,
  Q2 0.474, MITI 0.441). **On the primary oracle the same interaction is null** — largest dz 0.211,
  nothing survives Holm. The grader that *was* the training reward cannot see it.
- **PTO is 3.4× cheaper.** `PTO_LA0` reaches iteration 10 for **8.1** GPU-h vs `GRPO_LA0`'s **27.9**
  (27.91 / 8.12 = 3.44) *and* scores higher — PTO dominates on the compute axis, not just on the
  iteration axis. ⚠ But at matched *budget* (8.1 h, GRPO only at iteration 3) PTO wins the reward
  (+0.266 / +0.230) while being **worse on MICI** (+0.261 / +0.418), because it is ten iterations
  deep to GRPO's three.
- **The GRPO reward gain is partly verbosity.** Coded MI acts per 1,000 therapist characters roughly
  halve under K=5 at iteration 5 in *both* valences (all coded acts 4.08 → 2.22 primary, dz 0.717;
  4.97 → 2.59 held-out, dz 0.650) — half as behaviourally dense per word. Verbosity is a
  training-DEPTH channel, not a look-ahead one: chars/turn `GRPO_LA0` 394 @5 → 905 @10 vs
  `GRPO_LA5` 678 @5.
- **Substitution replicates on a second method, denominator-free.** Over-praise as a *share* of
  MI-inconsistent acts 0.178 → 0.086 (primary, p_holm .0045) / 0.182 → 0.063 (held-out, p_holm
  <.0001), while the overall MI-inconsistent share is flat or slightly worse. ⚠ The per-turn rate,
  per-session count and per-1k-character measures **disagree in direction** on GRPO — turns −26%,
  chars/turn +72% — so only the share is denominator-free.

**Process failures worth remembering.** (a) A bare `render_views.py` renders the **primary oracle
only**; the held-out judge's subtrees stayed stale for a full render cycle before anyone noticed,
and a stale judge subtree is *silent*. `--all-judges` is now documented in `eda/README.md`.
(b) `iteration_6/` read as EMPTY at 09:05 while holding a step written at 08:44 — the documented
Drive-symlink lag, hit again. Re-check the cloud before concluding a run died.

## 2026-08-11 — PTO LA5 reaches its endpoint; eight matched iterations say the same thing

**Run status.** PTO LA0 = 10 iters scored; GRPO LA0 = 10 iters (FINISHED, re-scored); **PTO LA5 =
iters 0–8 trained AND scored on BOTH graders — the arm has reached its configured
`NUM_ITERATIONS=8` endpoint**; GRPO LA5 still thin (I1 trained AND fully scored). **33 scored model
states.**

**The read (K=5 NEVER LEADS, at any of 8 matched iterations, under EITHER grader).** Primary oracle
Δ(K0−K5) on Q1+Q2: +0.08…+0.16 at iters 1–4, −0.002 at 5, **+0.257 at iter 6 (dz 0.42,
p_holm 0.0004 — the first Holm-significant Q1Q2 result in the whole K comparison)**, +0.044 at 7,
**+0.077 at 8** (dz 0.17, p_holm 0.21 — n.s., though Q2 alone is +0.145, dz 0.33, p_holm 0.010).
Under the held-out judge **K=0 leads at every iteration 1–8**, widening iter 6 to +0.343 and making
**iter 8 Holm-significant (+0.186, dz 0.34, p_holm 0.0019; Q2 +0.307, MITI +0.164)**. Levels at the
endpoint: primary 4.221 (K=0) vs 4.144 (K=5); judge 2.895 vs 2.710.

⚠ **The old "arms tie at iter 5" claim is SUPERSEDED** — it was a primary-oracle artifact that did
not survive iter 6, and the held-out judge never saw it as a tie. Make the claim about the lever,
never about convergence.

⚠ **The MICI tilt REVERSES at iters 7–8.** Δ(K0−K5) turns *positive*, i.e. K=5 is the MI-consistent
arm — +0.078/+0.059 primary (iter 7 p_holm 0.029) and +0.071 judge at iter 8 (p_holm 0.043). A
claim that flips sign across the run is a claim about *when*, not about the lever — **drop
"look-ahead costs MI-consistency" from the write-up** rather than restate it with more iterations.

**Cost.** ~**$1.25** for iter 8 (~$0.53 primary live, **$0.72 Haiku batched**, 1,536 cells,
0 errors). ⚠ Price a Haiku sweep off `judge_plan.sweep_report(..., receipt=(42.0, 22272))` — the
receipt-calibrated basis put iter 8 at $0.72 where the char estimator said $1.33 and a pro-rata
guess off the *live-priced* 08-10 run said $1.87.

**Operational lesson (learned twice the same day, over the same folder).** PTO LA5's `model_iter_8/`
first looked populated, then read as 0 files with an intermittent `WinError 1450` — but all 96 convs
were present in Drive the whole time; the local Drive Desktop mount had wedged on that one folder
and a Drive restart fixed it. **Before concluding an arm is unfinished, check the cloud** (the Drive
MCP connector lists the folder directly) — the alternative was a needless ~50-min regeneration.
Promoted to a standing gotcha in `CLAUDE.md`.

## 2026-08-10 — more training data did not help; data starvation is dead as an explanation

LA5 carried **1.2×** LA0's pref pairs at iter 6 (568 vs 475) and **1.7×** at iter 7 (689 vs 400),
and scored *worse* at 6 and *tied* at 7.

Three independent measurements now agree that look-ahead changes the **scale** of the reward signal,
not its **information content**:

1. it creates no extra branch points;
2. it multiplies the within-group spread ~1.55× with **margin and SD rising by the identical
   factor** — so it does not separate the winner from the pack; margin/SD sits at the pure-noise
   expectation for 8 draws in every arm;
3. at **matched policy** (train_iter 1, both arms on the base model) look-ahead adds **zero** reward
   faithfulness (11/19 depth bins, weighted −0.005, p=0.59). The pooled faithfulness advantage is
   confounded with the policy difference.

Cost: ~$5 (iters 6+7: $1.06 primary live, $3.74 Haiku batched, 3,072 cells, 0 errors). Full
evidence: the `project-lookahead-negative-result` memory.

## 2026-08-02 — the training signal becomes measurable for BOTH methods

`6_Preference` was PTO-only because GRPO has no preference pairs; but preference was never the
essential thing — both methods weight a group's candidates and step along the weighted sum (DPO ±1
on the logged chosen/rejected, GRPO the standardized advantage), so rescaling each group to a common
size puts them on **one probe**. Results, all in
[L0/SUMMARY.md](../eda/results/L0/SUMMARY.md) §6:

- **The affirmation push grows over training in BOTH methods** (exact, embedding-free, every group):
  GRPO −0.006 → **+0.086 ± 0.008**, PTO 0.008 → **0.103 ± 0.029** (iter 8). The reward-hack now has
  a *training-side* measurement, not only the outcome-side inference. GRPO's series dips negative at
  **iter 9** — the same iteration the outcome grid dips, from an independent source.
- **The two losses do not want the same thing:** pooled update-direction cosine 0.267 raw,
  **0.317 attenuation-corrected** (ceiling 0.844) — under a third of achievable agreement, at
  matched K and a shared oracle.
- **The push predicts the MICI move in GRPO only:** with `train_iter` partialled out (mandatory —
  the raw ρ is confounded with iteration by construction), GRPO's ΔMICI tracks its affirmation push
  **ρ 0.647 (p .043)**, length 0.706, over-praise 0.617; PTO's does not. Same direction as the
  endpoint MICI gap, reached from the training data. n ≤ 10/arm, uncorrected — mechanism, not cause.
- **THE GAP IS THE DATA, NOT THE LOSS** (second pass, same day). Swapping the weighting rule on the
  *same* groups barely moves the update direction (**0.908** on PTO's groups, **0.988** on GRPO's);
  holding the rule fixed across the two methods' *own* groups leaves them as far apart as ever
  (0.397 / 0.324 corrected, vs 0.317 as trained). At matched K and a shared oracle the two losses
  extract nearly the same direction from the same eight completions — **"PTO vs GRPO" is a statement
  about exploration, not about DPO vs group-relative weighting.**
- **The reward-hack is a compounding loop, not a hard pull.** Per-iteration *selection* pressure on
  affirmation is ≈0.01 → 0.10, while what the policy *generates* moves **0.02 → 0.54** (GRPO) /
  0.04 → 0.57 (PTO); over-praise reaches **0.74** of GRPO's candidates and questions collapse
  **0.71 → 0.06**. Small persistent pressure applied each iteration to an already-more-effusive
  policy — by the end the update is choosing between two effusive completions.
- **PTO's training signal shrinks by two-thirds:** branch points built 949 → 410, τ yield
  0.82 → 0.69, groups trained **782 → 281**, margin 0.274 → 0.196. GRPO trains on 94–98% of its
  groups throughout. A flattening PTO curve may partly be data starvation. *(Superseded 2026-08-10:
  more pairs did not help, so starvation is not the explanation for the K result.)*
- ⚠ **It also audited the old probe and the old probe lost.** §1's `wins_correct` was IN-sample;
  held out, a per-iteration PTO direction wins **0.47–0.59** with split-half reliability
  **0.15–0.32**. The per-iteration latent-drift artifacts (word drift, learn/unlearn, MI-concept
  curves) are **mostly estimation noise** — the L0 SUMMARY §6 claim built on them was corrected in
  place. Pooled directions (0.597 PTO / 0.911 GRPO) and the exact lexical contrasts are what survive.

## 2026-08-02 — RQ-i becomes a tracked artifact

The K read got stronger because it is persona-**paired** rather than a hand-computed difference of
means. `L5` owns it (`config.RQ_I_VIEW`); `7_Stats` §4c builds it from
`eda_analysis.cross_k_scores(S)`, which rebuilds the score frame with **only** the K filter dropped
and leaves export routing alone, so the pooled `all` view stays retired. Three artifacts under
`results/L5/{tables,figures}/7_stats/<judge>/`: `k_means_by_iter` (levels), `k_paired_by_method`
(Δ/dz/Holm p), `k_trajectory_Q1Q2` (all four arms in one frame). Guarded by the `cross-K frame
(RQ-i)` self-check.

Paired result as of this date: K=5 trails K=0 by 0.08–0.16 at iters 1–4 with *dz* ≤ 0.20 and never
significant, then ties at iter 5. ⚠ **The iter-5 tie was the primary oracle's picture only** — the
same tables under the held-out judge put K=0 ahead at **every** iteration 1–5, at iter 5 by
**+0.173 Q1+Q2 (dz 0.33, p_holm 0.017)**, plus MITI +0.206 and Q2 +0.236, all Holm-significant. Sign
agreement 68.5% overall but 92.9% at |Δ|≥0.10 and 100% at |Δ|≥0.15. Both graders also flagged **PTO
iter-4 MICI with K=5 worse** (−0.111 primary / −0.177 judge, both Holm-significant), and the judge
saw that tilt at iters 2–5. *(Both the tie and the MICI tilt were superseded on 2026-08-11.)*

## 2026-07-30 — RQ-i gets its first matched point; an orphaned-adapter repair path

`PTOExp3_LA5_I5` generated, scored on BOTH graders (23,040 Haiku cells; parity kept), folded, and
rendered. **PTO LA5 = iters 0–5.**

Tooling = [`code/PTO_Exp3/generate_eval_convs.{py,ipynb}`](../code/PTO_Exp3/generate_eval_convs.py),
which repairs any orphaned adapter (trained, but its `model_iter_N` convs were never generated).
Mechanics + the VRAM leak it exposed: [CHANGELOG_TRAINER.md](CHANGELOG_TRAINER.md), 2026-07-30 entry.

**Cleared the same day:** `iteration_6/pref_pairs/pairs.csv` (1 byte) + `eda/generations.jsonl`
(0 bytes) were deleted. `pairs.csv` is the Step-2 **completion marker**, so an empty one would have
made a resumed iter 6 reload 0 pairs, skip the ~41-min build, and run a silent **no-op DPO update**.
**Check for empty markers before resuming any arm.** Promoted to a standing gotcha in `CLAUDE.md`.

**Durable LA5-resume forensics** (dated detail in [CHANGELOG_EDA.md](CHANGELOG_EDA.md), 2026-07-11
entry): at that point **PTO LA5** had trained adapters for iters 1–5 but only I1–I4 scored — the
iter-5 eval convs were never generated by the run itself (`iteration_6/` died ~1 min in: adapter
saved 02:32, iter-6 dirs created 02:33). **GRPO LA5**: iter-1 adapter trained AND scored; its
`iteration_2/` is adapter-less. **Folder presence ≠ data.**

## 2026-07-28 — one score lake; second-judge ICC; reproducible figures

**ONE SCORE LAKE.** Every grader's scores now live in a single judge-partitioned tree,
`data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<id>.csv`, replacing four stores
under two schemes (the primary's reported draw was split per method with no `judge=`/`rep=` level;
every other grader sat in a separate local-only tree that had both). `judge` is an ordinary partition
key now, `rep=0` is each judge's full-grid draw, and there is one resolver instead of a
primary-vs-other branch. 50,320 files copied, hash-verified, then removed at source; **no headline
number moved** (45/45 endpoint cells and 25,056 rows identical). Two consequences:

- **The lake is a Drive symlink**, so the second judge's $42 sweep and the $9.16 ICC reps are backed
  up for the first time — previously they existed only on one laptop, gitignored.
- **The primary's ICC now spans 4 draws** (the reported one included, as the second judge's already
  did), which is why the range reads 0.86–0.99 rather than the older 0.90–0.99. Only MICI moves;
  Q1/Q2 shift ≤0.007.

**Second-judge ICC — MEASURED**, closing the last named validity gap (was the "cheapest remaining
validity buy"). 2 further Haiku reps on the anchor subset, 2,304 calls, 0 errors. Haiku's own ICC:
**Q1 0.951–0.978, Q2 0.938–0.963, MICI 0.525–0.929** — near-parity on Q1/Q2, but its MICI
repeatability *falls as the MI-inconsistency rate rises* (GRPO@10 0.525), so it is least reliable
exactly where the sycophancy claim lives. Against the corrected ceiling, agreement recovers
Q1 86–91% / Q2 83–88% but MICI only **29–59%**: partly the judge's noise, mostly construct
disagreement. **No headline result moved** — the MICI caveat stands and gain retention remains the
load-bearing evidence. `reliability.agreement` computes `sqrt(ICC_primary × ICC_judge)` from measured
values and records `ceiling_basis`; it falls back to the assumption only where a judge has <2 reps.
**Cost $9.16** (batched would have been $4.58) against "~$1–2" previously documented — that was an
unchecked estimate. Price judge spend with `judge_plan.estimate_cost`.

**Reproducible figures + the fold as a read path**, both found while proving the score-lake
migration changed nothing:

- **Seaborn's bootstrap CIs were unseeded**, so re-rendering rewrote 90 PNGs on unchanged data
  (three consecutive renders each differed by ~6% of pixels). `BOOT_SEED = 12345` was promoted from a
  private `stats._BOOT_SEED` to `constants` and passed at all seven `errorbar=("ci", 95)` callsites —
  the figure side and the table side now share one seed, and a thesis figure is reproducible rather
  than merely stable-looking.
- **The parquet fold is now a read path, not archival-only.** `eda_analysis/score_archive.py` owns
  the layout, the staleness guard and the read; `tools/consolidate_scores.py` is a thin CLI over it.
  `iter_conv_rows` serves from the fold when the per-partition content signature in
  `_parquet/_manifest.json` still matches disk and falls back to the CSVs otherwise — **4.3–6.1×**
  faster (`scores_long` 86 s → 16 s), with all seven per-conversation loaders proven identical under
  `assert_frame_equal(rtol=0, atol=0)` via either path. `_selfcheck` asserts both halves
  (fold-equals-CSV, and that a tampered signature is refused rather than served).

Also this date: **every** grader nests under its own short label in the results tree
(`gpt-4o-mini/`, `claude-haiku-4-5/`); the primary is no longer flat, so a figure path always names
the grader that produced it.

## 2026-07-27 — the second judge becomes co-primary; multi-judge EDA built

**Full sweep COMPLETE.** Claude Haiku 4.5 scored **22,272 / 22,272** cells (29 model states ×
8 rubrics × 96 convs; 232/232 cells at full n=96), matching the primary oracle's grid exactly. Cost
**$42** via Message Batches (50% off; measured 3,621 input + 71 output tokens/call). The whole EDA
now runs under either grader: `python tools/render_views.py --judge anthropic_claude-haiku-4-5`.
Notebooks 5+6 **refuse** a second judge — they read the training side, which cannot be re-graded
after the fact.

**Multi-judge EDA BUILT** (queued 2026-07-26). Lands as `8_Measurement_Validity` (free, inside
`tools/render_views.py`; family `8_measurement/`, no `<judge>/` level because every artifact contains
both graders) + `Judge_Reliability.ipynb` §3 (the paid full sweep). The four results that carry
weight, all on the tracked `L0` view (22 arms):

- **Sign preservation.** 18/18 on the thesis-critical anchor contrasts; **88.3%** across all 1,848
  arm×metric contrasts, rising to **98.9%** at |Δ|≥0.50 — the judges disagree only about differences
  too small to claim.
- **Variance decomposition.** Only **1.2–6.9%** of arm-mean variance is arm×judge: they disagree
  about *level*, not about arm *ordering*.
- ⚠ **MITI is the exception** — dependability 0.65 off one judge, and the weakest sign preservation
  (77.5%). **Treat MITI arm differences as provisional.** A thesis limitation, not a footnote.
- **Gain retention is the reward-hacking test.** Q1 retention PTO@10 **0.80** vs GRPO@10 **0.28**,
  non-overlapping, and per-iteration it is an *onset curve* — GRPO decays from ~0.89 (I3) to 0.28
  (I10) while PTO holds 0.80–0.98. Stronger evidence for sycophancy than the MICI rate, and it buys
  down LIMITATIONS §3 (circularity).
- **Cost, measured**: full sweep **$42 batched / $84 direct**; the free char-based estimator lands
  within 12%. Parity gate 8/8. Deliberately **1 rep, not 3** — oracle noise adds ≈0.01 to a 96-conv
  arm mean vs ≈0.09 from persona sampling, so breadth beats depth at equal cost.
  ⚠ **Haiku 4.5 caches nothing on this prompt** — confirmed empirically (`cached_input_tokens = 0`):
  its cacheable-prefix minimum is 4,096 and only Q1/Q2 come close.

## 2026-07-26 — judge validity: the instrument gets measured

The measurement instrument is now measured, not assumed — oracle **ICC(2,1) 0.86–0.99** (mean |Δ|
0.04–0.09, confirming the "≈0.10 noise" folklore; Q1/Q2 hold 0.96–0.99 and only MICI dips below
0.90 — floor is MICI PTO@10 at 0.864), and a decoupled second judge (**Claude Haiku 4.5**, different
family, never played the patient) reproduces every endpoint contrast with the same sign (**18/18**
after the full enumeration; it *widens* the PTO−GRPO Q1 gap to +0.77 vs the primary's +0.53).
Q1/Q2 cross-judge r 0.80–0.88 vs a measured 0.96–0.98 ceiling; MICI agrees weakly (r 0.20–0.55) so
the sycophancy claim holds at the contrast level, not as a precise rate. Buys down LIMITATIONS
§1–§2. Cost ~$5.30. See `eda/notebooks/analysis/8_Measurement_Validity` §1.
