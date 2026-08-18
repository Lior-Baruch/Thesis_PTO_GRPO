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
