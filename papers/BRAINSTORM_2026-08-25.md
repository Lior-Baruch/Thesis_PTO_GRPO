# Paper brainstorm — 2026-08-25 (post-completion of the full 4-arm grid)

**Context.** GRPO_LA5 finished at iteration 10; all 44 model states are scored under both graders.
Every prior Exp3 draft was retired to `papers/archive/` today because their grids were censored.
This document is the cold-read brainstorm for what comes next.

**Method note (CLAUDE.md § Epistemic status, rule 1).** §1–§3 below were written from the tables
alone — before opening `STATUS.md`, any `SUMMARY.md`, or the archived drafts' argument sections.
§5 is the diff against those narratives, added afterwards. One disclosed contamination: the
archive task required reading the retired draft's `README.md` header (its five "moves" with
numbers) before this cold read, so the section-structure of the retired paper was in mind; every
number and claim below was nonetheless taken directly from a named table, not from that README.

---

## 1. What the tables say (cold read, with paths)

All contrasts persona-paired, n=96, Holm-corrected unless noted. "Primary" = gpt-4o-mini (the
training oracle), "held-out" = claude-haiku-4-5.

**The K lever flips sign by optimizer** (`lookahead/reward/tables/k_summary.md`, `k_endpoints.md`):

- **GRPO: K=5 wins, big, everywhere.** At matched iteration 10, K=5 − K=0 on Q1Q2 =
  **+0.765 (dz 0.91)** primary and **+0.616 (dz 1.03)** held-out; significant on **all 8
  instruments under both judges**, incl. MICI (lower=better): −0.627 (dz −1.86) primary /
  −0.422 (dz −1.57) held-out. K=5 is significantly ahead on Q1Q2 at 6 of 10 iterations
  (primary: 4,6,7,8,9,10). Even against K=0's *best* iteration under either judge's honest
  selection, K=5's endpoint still wins broadly (`k_endpoints.md` rows "K=0 best by …").
- **PTO: K=5 is null-to-negative on outcomes.** Endpoint Q1Q2: +0.047 (p_holm 0.70) primary,
  −0.199 (dz −0.31) held-out; Q2 held-out −0.363 (dz −0.65, p<.001); MITI held-out −0.203
  (dz −0.49). The one clear PTO K=5 win is **MICI: −0.228 (dz −0.71) primary / −0.245 (dz −0.66)
  held-out** — look-ahead reduces MI-inconsistent behaviour even where it doesn't raise scores.
- **The interaction is a crossover, not an attenuation**
  (`lookahead/reward/tables/k_did.md`, `method/contrast/tables/method_paired_by_K.md`): at K=0
  PTO beats GRPO at iteration 10 (Q1Q2 +0.507, dz 0.73 primary; +0.609, dz 1.27 held-out); at
  K=5 GRPO beats PTO (−0.210, dz −0.36 primary; −0.206, dz −0.31 held-out). The
  difference-in-differences is significant from iteration 4 on and reaches **dz ≈ 0.79–0.97 at
  iteration 10 on both judges**.

**Endpoint levels** (`arms/stats/tables/gpt-4o-mini/main_results.md`, base ≈ 3.0 Q1Q2):
GRPO_LA5 ends highest (2.963 + 1.554 = **4.52**), then PTO_LA5 (3.003 + 1.303 = 4.31) and
PTO_LA0 (3.000 + 1.259 = 4.26), then GRPO_LA0 (3.067 + 0.686 = **3.75**). Held-out own-base
gains at iteration 10 (`lookahead/transfer/tables/k_retention_summary.md`): GRPO_LA5 **+1.038**
and PTO_LA0 **+1.036** — the two winning arms are indistinguishable on held-out gain, at very
different prices (below).

**Compute** (`compute/cost/tables/compute_by_arm.md`, `step_multiplier.md`,
`budget_sweep_crossjudge_verdicts.md`):

- Totals: PTO_LA0 **8.1** GPU-h, PTO_LA5 19.7, GRPO_LA0 27.9, GRPO_LA5 **51.2**. GRPO K=5 steps
  cost 1.83–1.97× K=0 (settled iterations); PTO's K=5 iteration costs ~2.4× (the tree build is
  70–85% of PTO's bill).
- **The GRPO K verdict survives iso-compute now.** At the 51.2 GPU-h common budget, GRPO K=5 >
  K=0 under **all four** select/eval judge combinations (dz 0.38–0.74), including honest
  cross-judge selection. On the trajectory (`iso_compute_contrast.md`): K=5 loses below ~18
  GPU-h, draws at ~23–27, wins from ~27–30 GPU-h on.
- **But the method verdict at matched budget still favours PTO.** At 19.7 GPU-h, PTO_LA5 ≥
  GRPO_LA5 in 3 of 4 combos (dz 0.30–0.67; the held-out-eval-with-primary-selection cell is
  null). And at 8.1 GPU-h, PTO_LA0 crushes budget-matched GRPO_LA0 (dz 1.07–1.39). PTO_LA0
  reaches +1.26 Q1Q2 (primary) in 8 GPU-h; GRPO needs 51 GPU-h and the K=5 lever to beat that.

**Behaviour** (`lookahead/behaviour/tables/k_channels_summary.md`; MICI channels lower=better):

- **K=0 arms reward-hack; K=5 arms don't — on the primary grader.** Vs own base, endpoint MICI:
  GRPO_LA0 0.211 → 0.837, PTO_LA0 0.213 → 0.491, while GRPO_LA5 stays flat (0.209 → 0.210) and
  PTO_LA5 near-flat (0.178 → 0.264) (`main_results.md` MICI rows; delta = raw rise).
  ⚠ *Correction from the narrative diff (§5):* "flat" is a **primary-grader claim** — the held-out
  judge reads GRPO_LA5's same conversations 0.326 → 0.628. The judge-free statement is the lexical
  over-praise marker (`arms/validity/tables/<judge>/overpraise_crosscheck.md`): 0.671 vs ~0.06
  within GRPO, 0.210 vs 0.045 within PTO — "look-ahead slows the loop by ~10×", never "stops it".
- The dominant K=0 channel is **over-praise**: significantly higher under K=0 in 5–7 of the
  last 7 iterations for both optimizers, max dz 2.62 (GRPO rate, held-out).
- **PTO shows the substitution**: under K=5, advice-without-permission is significantly *higher*
  in 6 of the last 7 iterations (held-out), while over-praise closes — so PTO's aggregate MICI
  win is late (iters 9–10). GRPO shows only traces of this (3 iterations, count metric).
- GRPO K=5 also asks **more questions per turn** (sig. higher at 7 iterations, mean dz 0.64,
  judge-invariant text metric) and gives fewer affirmations per turn (both graders).

**Mechanism** (`lookahead/mechanism/tables/faithfulness_k_summary.md`): pooled over matched
iterations, the K=5 training cut agrees with the full-conversation eval slightly *better* than
the K=0 cut (GRPO primary 0.909 vs 0.873, delta CI excludes 0; PTO 0.863 vs 0.836) — small, but
the sign is consistent on both judges and both methods.

**The regime dependence is measured, not asserted**
(`lookahead/replication/tables/crossgen_kcontrast_summary.md`): the ICLR-era Exp1 conversations,
re-scored with the modern oracle, still show K=5 > K=0 for PTO (mean over iterations 1–7:
dz −0.54 modern grader, −0.61 original) — so PTO's Exp3 null is a property of the Exp3 regime
(bf16, harder patients, iterative loop), not of the judge, and not a refutation of the poster.

**Measurement** (`measurement/validity/tables/multijudge_sign_preservation.md`): the held-out
judge preserves the sign of 88.4% of all 7,568 cross-arm contrasts, 97.2% of those with
|Δ primary| ≥ 0.25, 99.3% at ≥ 0.50.

## 2. What is NOT covered by any artifact (rule 5 — the invisible part)

- **One training run per arm.** No seed replicates of training; every dz is across personas
  within a single run. (Judge repeatability has reps; training does not.)
- **No human MI-expert ratings anywhere.** All eight instruments are LLM-administered.
- **No intermediate K** (dose–response between 0 and 5 is unmeasured), no K > 5.
- **Single base model (Llama-3.2-1B), single patient simulator, single domain (MI).**
- The API-cost axis exists (`api_calls.md`) but the budget sweeps are GPU-h only.
- **Why GRPO benefits and PTO doesn't is not directly measured** — the dispersion/faithfulness
  tables are consistent with more than one mechanism story (all-8-rewards vs best-worst-pair
  selection; on-policy group baseline vs off-policy DPO reference). Any mechanism section must
  be stated as consistent-with, not shown.
- Nothing measures whether the K=5 GRPO policy is *actually better MI* in a clinician's
  judgment vs. better at satisfying LLM-administered instruments.

## 3. Candidate papers (cold, ranked)

### P1 — "Trajectory reward for group-relative RL in goal-oriented dialogue" (GRPO arms only) ★ recommended first
Lior's suggestion, and the tables carry it comfortably. The ICLR-poster question — does scoring
a turn on its K-turn continuation help — answered for GRPO in a multi-turn MI setting:

1. **Result:** K=5 more than doubles the endpoint gain (+1.554 vs +0.686 own-base Q1Q2 primary;
   +1.04 vs +0.40 held-out), significant on all 8 instruments, both judges, from iteration 4 on.
2. **Cost honesty:** 1.9× per step, 1.84× total; loses below ~18 GPU-h, wins at the full budget
   under honest cross-judge selection — quote `budget_sweep`, never one iso-compute row.
3. **Safety/behaviour:** K=0 GRPO reward-hacks (over-praise; MICI 0.21 → 0.84); K=5 holds MICI
   flat and redirects toward questions. Look-ahead as *hack prophylaxis* is the memorable claim.
4. **Mechanism (consistent-with):** the K=5 training cut is a more faithful proxy of the
   final-conversation score (0.909 vs 0.873 rank agreement).
5. **Scope honesty:** cite the thesis/companion for "this does not transfer to PTO" in one
   paragraph — do not import the 2×2.

**Two caveats P1 must carry in its own results section** (from STATUS.md, both table-owned):
(a) the hero state is also the experiment's **worst per-conversation cross-judge agreement** —
GRPO_LA5 Q1 agreement falls 0.941 (I5) → 0.487 (I9) / 0.544 (I10) against a 44-state median of
0.855, driven by one-sided saturation of the training grader (its Q1 SD collapses to 0.275× of
base while the held-out's *grows* 1.41×). Never print 4.517 without this. (b) on the held-out
judge the endpoint is "flat since ~iteration 6", not still climbing, and MICI-flat is
primary-only (see §1). These aren't fatal — the held-out endpoint still wins the K contrast at
dz 1.03 — but they are the difference between an honest paper and a rebuttal magnet.

Shape: short paper / workshop (CLPsych, or an RLHF/alignment workshop). Positive, large,
cross-judge-robust, cost-transparent. Fastest path to a submission; no new spend.

### P2 — "Same lever, different optimizer" v2 (the full 2×2, uncensored) — the main-venue paper
The retired draft's question now has a *decisive* answer instead of a censored shrug: the
identical trajectory-level reward, at matched MCL/M=G/temperatures/personas, **helps GRPO with
dz ≈ 1 and does nothing-to-harm for PTO**; the DiD is dz ≈ 0.8–1.0 on both judges; and the
optimizer ranking itself flips with K (PTO wins K=0, GRPO wins K=5, matched iteration). Add the
budget caveat (at matched GPU-h PTO ≥ GRPO even at K=5) and the ICLR-replication section (the
lever's PTO-gain was real in its regime — regime-dependence, not judge artifact). This is the
thesis's central chapter and the strongest full-length paper. It should be a **fresh draft**
(new NUMBERS.md from the current tables), reusing the archived draft's style files only.

### P3 — "Look-ahead as reward-hacking prophylaxis" (behaviour-first, both optimizers)
Lead with §1's behaviour block: turn-level reward teaches flattery in both optimizers; K=5
closes it in both; in PTO the inconsistency partially *relocates* to unsolicited advice (the
substitution result, now on the full grid), in GRPO it doesn't. Aggregate-vs-channel and
grader-dependence traps from the retired ledgers still bind. Venue: CLPsych / safety workshop.
Risk: overlaps P1 §3 and P2 §6 — write it only if P1/P2 reviews ask for depth here, or as the
thesis's behaviour chapter excerpted.

### P4 — Measurement note: "Does a held-out judge preserve your conclusions?"
Sign preservation by effect size (88% → 99%), gain retention 0.27–0.85 by arm×metric, judge
level offsets, never-average-judges, repeatability ICC. Best as a NeurIPS/ACL eval-workshop
short or as P2's measurement section. Not a standalone priority.

### P5 — Compute note: "Iteration is the wrong denominator"
Per-iteration conclusions reverse at matched GPU-h (three concrete reversals in §1). Best folded
into P1/P2 as the cost section; standalone only for an efficiency workshop.

**Recommendation:** P1 now (small, self-contained, all-positive, no new compute), P2 as the
main paper right after (it subsumes P3–P5 as sections). P3–P5 stay sections unless a deadline
calls for them.

## 4. EDA review — recommendations (from the cold read)

1. **Audit one suspicious coincidence:** `arms/stats/tables/gpt-4o-mini/main_results.md` rows
   GRPO_LA5 final: Q1, Q2, and Q1Q2 deltas are all exactly **1.554** to 3 decimals (bases and
   dz differ; target means are internally consistent at 4.464/4.570/4.517). Probably a real
   coincidence, but cheap to verify the generator isn't reusing a column. *(Verified 2026-08-25
   against the score lake via `load_scores_long`: the deltas differ at the 4th decimal —
   ΔQ1 1.55417, ΔQ2 1.55392, ΔQ1Q2 1.55404. Coincidence at 3-dp rounding; no bug.)*
2. **The completed-grid headline artifact is missing.** No single table/figure shows the four
   endpoint levels ± CI under both judges side by side (the thing every deck and paper opens
   with). Candidate: `method/contrast` gains a `headline_grid.md` + one figure (4 arms × Q1Q2 ×
   2 judges, base-anchored).
3. **Faithfulness pooled-vs-matched-policy tension.** `faithfulness_k_summary.md` (pooled: K=5
   more faithful) vs the matched-policy table the retired draft quoted ("adds no faithfulness").
   Both can be true (selection vs pooling); one CAPTIONS/METRICS_REFERENCE sentence should say
   which cut supports which claim, or a new draft will misquote one of them.
4. **Budget sweeps lack an API-cost axis.** `api_calls.md` exists; a `budget_sweep` variant on
   $-cost (patient+oracle calls priced) would close the "GPU-h is not the whole bill" hole that
   a reviewer of P1 §2 will poke. Cheap: compute.py already has the call counts.
5. **Dose–response gap is invisible in the artifacts** — no table says "K ∈ {0,5} only, by
   design". One LIMITATIONS.md line prevents a reader from assuming K was swept.
6. **`k_summary.md` mixes count-columns and lower-better logic** (`n_sig_K0_higher` vs
   `n_sig_K5_better`) — correct but easy to misread; a CAPTIONS example row ("for MICI,
   K0_higher means K5 better") would prevent the exact class of prose error the epistemic rules
   exist for.
7. Process: seed the next paper's `NUMBERS.md` at EDA-render time (a script that emits
   claim → table-path stubs for the headline numbers), so the ledger starts complete instead of
   being reverse-engineered from prose.

## 5. Diff vs the narrative docs (added after the cold read)

Read after §1–§4 were written: `STATUS.md` (current, rewritten 2026-08-25 on the complete grid).
The five `results/<top>/SUMMARY.md` were **deliberately not read** — STATUS.md records that they
are pre-completion prose under staleness banners, so diffing against them would only measure
their staleness, not my anchoring.

- **No contradictions.** Every number in §1 that STATUS.md also quotes agrees (method flip
  +0.507/−0.210, DiD 0.718/0.815 raw, cost ratios, budget-sweep verdicts, endpoint levels).
  The cold read and the narrative were derived independently and converge — the epistemic-rules
  outcome you want.
- **Three things the cold read missed** (now folded into §1/§3 with ⚠ marks):
  1. The **per-conversation cross-judge agreement collapse** on GRPO_LA5's last iterations
     (0.487/0.544 vs median 0.855) + its mechanism (one-sided primary-grader saturation:
     Q1 SD ratio 0.275× vs 1.41×). I read the arm-level sign-preservation table and stopped one
     table short of the per-state one. Changes P1's obligations, not its viability.
  2. The **judge-free lexical over-praise cross-check** (11× / 4.7× suppression) — *strengthens*
     P1 §3 and P3 beyond what I had.
  3. **MICI-flat is primary-only** (held-out 0.326 → 0.628 on the same conversations).
- **One planning delta:** STATUS.md's next-step #2 ("re-cut the paper") is superseded by today's
  decision to archive all drafts and start fresh; STATUS.md was updated accordingly. Its
  next-step #3 (a second independent 96-conversation draw for GRPO_LA5@10 and PTO_LA0@10) is
  adopted below as the one spend-gated item, because it directly de-risks P1's single-draw
  endpoint — and the cross-judge collapse at that exact state makes a replicate more valuable,
  not less.

## 6. Implementation plan (for the next session — Opus)

Ordering matters; A and B are pure writing/code with no spend, C is Lior-gated.

**A. Rewrite the five `results/<top>/SUMMARY.md`** (STATUS.md's declared largest outstanding
task). One file per top (`arms`, `lookahead`, `method`, `compute`, `measurement`), from the
*current* tables only, staleness banners removed. Rules: every composite number shows its
arithmetic; every number names its table path; interpretive vocabulary gets a mechanism or gets
cut; the eight traps in STATUS.md §"Write-up decisions" bind (axis + grader named on every
MI-consistency claim; never a method verdict without naming K; dz over raw ratios for the DiD;
channel level, not totals). `compute/SUMMARY.md`'s "two GRPO arms cost the same within 3%" is
the known worst offender (actual 51.205 / 27.906 = 1.84×).

**B. EDA touch-ups before any paper figures** (§4 items, small):
   1. `method/contrast`: add the completed-grid headline artifact (4 arms × endpoint ± CI ×
      both judges, base-anchored) — table + one figure, wired through the family notebook so
      `render_results.py` owns it.
   2. One clarifying sentence in `METRICS_REFERENCE.md` (or the mechanism CAPTIONS) separating
      the pooled faithfulness result (K=5 slightly more faithful) from the matched-policy null —
      which cut supports which claim.
   3. `LIMITATIONS.md`: one line each for "K ∈ {0,5} by design (no dose–response)" and "one
      training run per arm".
   4. `k_summary.md` CAPTIONS example row for the lower-better column logic.
   5. Optional: API-dollar variant of the budget sweep from `api_calls.md` counts (compute.py
      already holds the pieces; no dollar figure quoted without a dashboard reconcile —
      STATUS.md's rule).
   After edits: `python -m eda_analysis._selfcheck` (all 26), then `render_results.py --top
   method` (or the touched families) and prove the refactor by rendering twice, not by diffing
   against the committed tree.

**C. (Lior's call — small spend + local GPU time) The replicate draw**, exactly as specified in
STATUS.md next-step #3: `generate_eval_convs.py --conv-dir conversations/replicate/...` for
GRPO_LA5@10 and PTO_LA0@10, lake folders `*_rep1_I10`, 2 × 96 × 8 = 1,536 scoring calls per
grader. Do NOT write into `conversations/full/` (glob-collision trap documented there).

**D. Scaffold P1** (`papers/2026_grpo_lookahead_mi/` or similar):
   - Copy vendored `acl.sty` / `acl_natbib.bst` / `refs.bib` from
     `papers/archive/2026_lookahead_pto_grpo/` (reuse-by-copy, per convention).
   - `README.md` = P1's argument (§3 above, with both caveats); `NUMBERS.md` seeded
     claim-by-claim from the table paths cited in §1 **after** re-checking each against the
     rendered tree — this brainstorm is INTERPRETATION tier, not a ledger.
   - Adapt `sync_figures.py`; figures come only from `results/`, filenames keep the grader tag.
   - Draft order: results (§4-style: the K contrast → cost → behaviour/hack → mechanism →
     measurement caveat) before intro; Limitations gets §2 of this doc almost verbatim.
   - The archived ledgers' ⚠ traps carry over wherever a number is shared.

**E. P2 planning happens after P1 drafts**, not in parallel — same tables, different scope; a
fresh cold pass at that point is cheap and keeps the two papers from cross-contaminating.

