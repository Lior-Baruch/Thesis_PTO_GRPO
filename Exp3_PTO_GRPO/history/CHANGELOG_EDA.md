# Exp3_PTO_GRPO — EDA change history

Dated "pass"/"Landed" entries for the **EDA** (the `eda_analysis` package, the notebooks, the score
lake, the results tree). Newest first. The trainer / `_shared/` infrastructure history is the
sibling [CHANGELOG_TRAINER.md](CHANGELOG_TRAINER.md); the index is [CHANGELOG.md](CHANGELOG.md).

These are superseded by the current-state sections in the root
[CLAUDE.md](../../CLAUDE.md) — kept here for provenance.

---

**Landed (2026-08-02, second pass) — three questions the aggregate curves could not answer.**
With both methods on one probe, three follow-ups became askable from data already on disk. All
three landed in `6_Preference` §5.

- **Is "PTO vs GRPO" the LOSS or the DATA?** The as-trained direction cosine confounds two
  differences: the methods see different candidate *pools* and apply different weighting *rules*.
  Every group logs all its candidates' scores, so `reweight` holds the groups fixed and swaps only
  the rule. The answer is unambiguous: **swapping the rule on the same groups barely moves the
  direction (0.908 on PTO's groups, 0.988 on GRPO's), while holding the rule fixed across the two
  methods' own groups leaves them as far apart as ever (0.356 / 0.266 raw; 0.397 / 0.324
  corrected, against 0.317 as trained).** So the divergence is **entirely about which candidates
  each method generates**, not about DPO vs group-relative weighting — at matched K and a shared
  oracle the two losses extract nearly the same direction from the same eight completions. That
  reframes the thesis comparison as one about *exploration*, and is the strongest single result of
  this pass.
  - ⚠ The attenuation correction assumes independent estimation error. True across arms; false for
    the same-groups rows, where both directions share their noise — the correction over-corrects
    and can exceed 1.0. The table carries a `read` column naming which cosine to quote per row.
  - The score-only `dpo` rule reproduces PTO's recorded roles: it picks a maximum-scoring candidate
    **100%** of the time (and a minimum 100%). Exact row identity is only 82%, purely because **40%
    of PTO groups have TIED maxima** where any tie-break is arbitrary — which is why the check
    asserts "picks a maximum", not "picks the same row".
- **Does the update PULL the drift or FOLLOW it?** §3 measures what the update selects for *within*
  a group; `pool_mean_by_iter` measures what the policy *generates* at all. The two are on
  completely different scales: the affirmation-marker **selection** contrast is ≈0.01 → 0.10, while
  the **generated** rate goes **0.02 → 0.54 (GRPO) and 0.04 → 0.57 (PTO)**; over-praise
  0.003 → **0.74** (GRPO); questions collapse **0.71 → 0.06** (GRPO), 0.67 → 0.27 (PTO); mean
  completion length roughly triples. The reward-hack is therefore **not one hard pull** — it is a
  small, persistent, same-signed selection pressure compounding through an on-policy loop, each
  iteration branching from an already-more-effusive policy. By the last iterations the update is
  choosing between two effusive completions, which is also why the selection contrast *understates*
  how far the pool has travelled. Indexing is off by one on purpose and documented: `train_iter n`'s
  pool describes the policy the eval set calls `model_iter_{n-1}`.
- **How much usable signal is there?** GRPO trains on **94–98%** of the groups it builds, flat
  across training. PTO's τ filter and its shrinking trunks compound: branch points built fall
  **949 → 410**, yield falls **0.82 → 0.69**, so groups actually trained fall **782 → 281 (−64%)**
  while the mean best−worst margin decays 0.274 → 0.196. PTO's late iterations train on a third of
  the data its early ones did — a candidate explanation for a flattening curve that no outcome
  figure can see.
- **Text exhibits.** `pref_examples` prints the most decisive up- vs down-weighted completions per
  iteration. At iter 1 PTO is choosing between an agenda-setting turn and a question; by iter 10 it
  is choosing between *"You're brave, open, and strong…"* and *"You're courageous and courageous,
  and your resilience is inspiring…"*. Deliberately the most decisive groups, not a random sample,
  and labelled as such.
- Self-check (still 17, extended): both counterfactual rules must preserve `Σw = 0` / `Σ|w| = 2`,
  and the `dpo` rule must select a maximum-scoring candidate in every PTO group.

---

**Landed (2026-08-02) — the preference EDA stops being PTO-only, and starts checking itself.**
`6_Preference` could describe what PTO's update wanted but had no way to ask the same of GRPO ("no
preference pairs"), and no way to ask whether wanting it did anything. Both gaps close, and the
second one turned up a problem in the notebook's existing half.

- **"Preference" was never the essential thing.** Both methods weight the candidates of a group and
  step along the weighted sum; they differ only in the weights — DPO puts ±1 on the *recorded*
  `chosen`/`rejected` roles, GRPO uses the standardized advantage `(r−mean)/std` that actually
  scales each completion's gradient. `pref.load_weighted_candidates` reads both out of
  `generations.jsonl` and rescales each group to `Σ|w| = 2` (DPO's natural size), which is what
  makes them comparable at all: without it GRPO's weights carry the group's reward spread and every
  cross-method number is a scale artifact. Every existing probe takes a plain direction dict, so
  word ranking / MI concepts / drift now work for GRPO unchanged.
- **A bug this found in itself.** The first cut scaled *any* group with non-zero weight, including
  the ~16% of PTO branch points that logged a `chosen` but no `rejected` — the τ filter emitted no
  pair there and DPO never saw them. Rescaling that lone `chosen` to a full +2 one-sided push
  produced a spectacular fake result (affirmation contrast 0.036 → 0.476 over training) that was
  really just "what chosen completions look like". Groups now need both signs to survive; the real
  effect is ~10× smaller. The self-check pins it (PTO groups must hold exactly 2 candidates).
- **What the updates actually push toward** (`update_lexical_push`, exact, every group, with SEs
  because these are small per-pair numbers): the **affirmation-marker push grows over training in
  BOTH methods** — GRPO −0.006 → **+0.086 ± 0.008**, PTO 0.008 → **0.103 ± 0.029** at iter 8. The
  reward-hacking story now has a training-side measurement, not only an outcome-side inference.
  GRPO's series also dips negative at **iter 9** — the same iteration the outcome grid dips across
  almost every metric, from an independent data source.
- **The two losses do not want the same thing.** Pooled direction cosine PTO vs GRPO is 0.267 raw,
  but a raw cosine is capped by how well each direction is estimated, so `pooled_direction_cosines`
  reports the **attenuation ceiling** (0.844) and the corrected value: **0.317**. Under a third of
  the achievable agreement, at matched K and a shared oracle.
- **Does the signal predict the move?** `link_to_outcomes` joins each iteration's features to the
  persona-paired eval delta *that update* produced (train_iter n → `model_iter_n` vs
  `model_iter_{n-1}`; the self-check verifies the join against raw iteration means, because an
  off-by-one here would silently credit every update to its neighbour). The raw correlations are
  confounded — nearly every feature trends with iteration and so do the deltas — so
  `outcome_correlations` reports **`rho_partial_iter`** with `train_iter` partialled out of both
  sides. After that control: **GRPO's ΔMICI tracks its affirmation push (ρ 0.647, p .043), its
  length push (0.706, p .023) and its over-praise push (0.617, p .057); PTO's does not** (−0.492,
  ns). Which is the direction the outcome side already points (GRPO MICI endpoint 0.84 vs PTO 0.49)
  — reached from the training data instead. n ≤ 10 per arm, uncorrected: a mechanism consistent
  with the curves, not a cause.
- **⚠ The probe audit is the uncomfortable part.** `direction_quality` adds two numbers the original
  probe never had, and they do not flatter it: `wins_holdout` (each half scored by the *other*
  half's direction) and `split_half_cos`. A PTO direction estimated from ONE iteration's pairs
  scores **split-half 0.15–0.32** and **held-out wins 0.47–0.59** — at several iterations, chance —
  against the **0.66–0.73** in-sample `wins_correct` §1 reports. So §1–§2's per-iteration artifacts
  (word drift, learn/unlearn, MI-concept curves, direction drift) are **largely estimation noise**,
  and the notebook now says so in its header, its §1 banner and its "how to read". Pooled over
  iterations the same direction reaches 0.597 (GRPO: 0.911, having 8 candidates a group instead of
  2), which is why every cross-arm claim uses `direction_by_arm`.
- **Sanity gate.** `direction_agreement_with_pairs` cross-checks the new candidate-derived PTO
  direction against the one built from `pairs.csv` — two independent logs of the same DPO update.
  Cosine is **exactly 1.000 at every iteration small enough to escape the 400-group sampling cap**
  and 0.82–0.96 where sampling bites, i.e. the only deviation is the cap. The GRPO side is read the
  same way, so that agreement is what licenses it.
- Sampling is seeded **per (arm, iteration)**, not from one shared stream, so a cell's draw does not
  change when another arm joins the frame — verified identical between the `L0` and `all` renders.
  The 400 cap itself was chosen by measuring reliability at 50/100/200/400, and `sample_groups`
  prints what it dropped.
- `RE_EFFUSIVE` moved to `constants` beside `RE_AFFIRM`: the conversation side (`behavior`) and the
  training side (`pref`) now test over-praise with one regex, or the two sides could drift apart.
- Self-check is **17** (`update probe (both methods)`): one weight scale in every group of both
  methods, τ-filtered PTO groups excluded, and the iteration join landing on the right eval step.

---

**Landed (2026-08-02) — RQ-i stops being a number computed in the margin.** The look-ahead contrast
had no tracked artifact. It needs both K arms in one frame, and neither tracked view has that: `L0`
holds the K=0 arms, `L5` the K=5 arms, so `stats.paired_k_comparison` returned an empty frame in
both and `7_Stats` printed *"K0-vs-K5 not comparable yet"*. The contrast lived only in the pooled
`all` view, retired to gitignored scratch on 2026-07-27 — so the K0-vs-K5 table in
[`results/L5/SUMMARY.md`](../eda/results/L5/SUMMARY.md) §3 was hand-computed from
`load_scores_long()` and said so. A thesis research question was resting on a number no deliverable
contained.

- **The fix is a wider READ, not a wider view.** `config.cross_k_scores(S)` rebuilds `scores_long`
  with **only** the K filter dropped — methods/modes/labels, judge + rep, persona attachment and
  derived MITI-proficiency rows all still come from the active `EdaConfig` — and touches **nothing**
  about routing. The tempting alternative (relax the view, or bring `all` back) would have
  re-pointed every *other* artifact in the notebook at a pooled frame to fix one table.
- **One owner.** `config.RQ_I_VIEW = "L5"` names the view that saves the RQ-i artifacts (the
  look-ahead view, whose SUMMARY already narrates the question); other views print a pointer instead
  of a duplicate. Asserted K-specific in the self-check, so pointing it at `all` fails loudly.
- **Three artifacts**, `7_Stats` §4c → `results/L5/{tables,figures}/7_stats/<judge>/`:
  `k_means_by_iter` (new `stats.k_means_by_iter` — both arms' means per method × rubric × iteration,
  keeping one-sided rows so "K=0 kept climbing after K=5 stopped" stays visible),
  `k_paired_by_method` (the paired Δ/*dz*/Holm *p*, which the view previously could not produce at
  all), and `k_trajectory_Q1Q2` — the first figure family `7_stats/` has ever held, and the "both K
  arms in one frame" the old ⚠ asked for.
- **The read got stronger, and slightly worse for look-ahead.** Persona-paired now, not a difference
  of means: K=5 trails K=0 by 0.08–0.16 at iters 1–4 with *dz* ≤ 0.20, **never significant**, then
  ties at iter 5 (4.014 vs 4.016). Indistinguishable at matched iteration count — for a lever that
  costs materially more per iteration, that is a negative result rather than a null one.
- **And rendering it under the second judge immediately paid for itself.** Haiku 4.5 puts K=0 ahead
  at **every** iteration 1–5, with iter 5 at **+0.173 Q1+Q2 (*dz* 0.33, p_holm 0.017)** plus MITI
  +0.206 and Q2 +0.236 — i.e. **the iter-5 tie is the primary oracle's picture, not a fact about the
  lever**. The conclusion "K=5 never wins" survives both graders; "the arms converge at iter 5" does
  not, and SUMMARY §3 + the root CLAUDE.md now say so. Sign agreement runs the familiar ladder:
  68.5% over all 54 contrasts, 92.9% at |Δ|≥0.10, 100% at |Δ|≥0.15. The one contrast that clears
  Holm under the primary — PTO iter-4 **MICI**, K=5 worse by 0.111 (*dz* −0.40) — is **flagged by
  both judges** (−0.177 held-out), and the held-out judge sees the same tilt at iters 2–5, so it
  reads as a suggestive cost in MI-consistency rather than the 1-in-54 fluke it looked like off one
  grader.
- **Sign convention unified.** Both tables use Δ = K0 − K5 (`+` ⇒ K=0 higher, matching
  `paired_k_comparison`); SUMMARY §3 previously printed K5 − K0. The self-check pins them together —
  on every complete 96/96 cell the unpaired delta must equal the paired `mean_delta` to 1e-9, which
  is exactly the invariant a future refactor of either function would break silently.
- **Guard:** new `cross-K frame (RQ-i)` check (self-check is now **16**, and the docs' "14" was
  already stale by one). It asserts both halves — the returned frame widens K while the view's own
  scores hold exactly one, **and** `exports._results_root()`/`_fig_dir()` are byte-identical across
  the call.

---

**Landed (2026-07-29) — the judge evidence gets its own family, and a cross-judge figure stops
being filed as one grader's output.** Notebook 5 was `[TRAINING ↔ EVAL]`: §1–§6 read the training
side, §7–§8 read `data/eval_scores/`. That split was visible in the artifacts — **every one of
family 5's 14 tables came from §7–§8**, as did 8 of its 13 figures — but the family was called
`5_training/`.

The real defect was in the layout, not the name. `figures/5_training/gpt-4o-mini/multijudge_variance_decomposition.png`
plots **both** graders, yet sat under `gpt-4o-mini/`, whose stated meaning since 2026-07-28 is *"the
grader that produced this file"*. And because notebook 5 was in `TRAINING_SIDE_NOTEBOOKS`, a
`--judge` render skipped it — so these artifacts *could only ever* be written under the primary.
The one figure proving the two judges agree was filed as the primary oracle's own output.

- **Split.** `5_Training_and_Reliability.ipynb` → `5_Training.ipynb` (§1–§6, unchanged, still
  `[TRAINING]` and still judge-refusing) + new **`8_Measurement_Validity.ipynb`** (§7–§8, renumbered
  §1–§2) → new family `8_measurement/`. Named *Measurement_Validity*, not *Judge_Reliability*, so it
  cannot be confused with the **paid** `notebooks/scoring/Judge_Reliability.ipynb` that writes the
  scores this notebook reads.
- **`JUDGE_INVARIANT_GROUPS` is now a first-class concept** in `exports.py`, applied at `_leaf` —
  the single place group and judge compose — so family 8 exports with **no `<judge>/` segment**.
  `reset_results` gains the matching branch (with no judge level, the family folder itself is the
  active scope), and `_write_xlsx_sheet` too (it named the workbook one level up from the leaf,
  assuming the leaf was a judge; without the fix family 8's workbook came out as `tables.xlsx`).
- **Rendered once, not once per grader.** `reliability.py` loads every judge from the lake
  explicitly and ignores `EDA_JUDGE`, so a per-judge render would rewrite the same bytes.
  `render_views.py` now skips notebook 8 on `--judge` runs — the same mechanism as notebooks 5/6 but
  for the opposite reason, and it says which reason on stdout.
- **Appended as 8, not renumbered in.** Slotting measurement-validity next to `3_validity` would
  have churned ~15 doc references, both `SUMMARY.md` trees and the deck builders for no analytical
  gain. 1–7 then 8 as the instrument appendix also reads honestly: it is about the ruler, not the
  result.
- **Guarded** by a new `_selfcheck` check (**15 total**): no `multijudge_*` / `second_judge_*` /
  `oracle_repeatability_*` / `judge_*` artifact may sit anywhere but directly inside a
  judge-invariant family. Asserted as a **path shape** rather than against a list of grader names,
  so it keeps working when a third judge lands. It failed loudly on the pre-render tree and passes
  after — which is the evidence that the move actually happened.
- **Also fixed:** a stray `results/L0/figures/gpt-4o-mini/_provenance.md` — a phantom family at
  family depth, left by an interactive `notebook_setup(EdaConfig())` with no export group.
  `save_provenance` now returns `""` instead of writing when there is no family, since a banner that
  records the config for *nothing* documents nothing.
- Both deck builders learned the rule (`JUDGE_INVARIANT_FAMILIES` in `_jp`); they currently cite only
  family 5's training-side figures, so nothing in a deck moved.
- **Verification:** full re-render of L0+L5 × 8 notebooks, no failures; `_selfcheck` 15/15;
  0 broken relative links across the 15 hand-authored docs + both `INDEX.md`; `--judge` dry run
  prints both skip reasons and exits 0.

---

**Landed (2026-07-29) — docs consolidated: one context file, one changelog, one owner per number.**
A full audit of the 15 hand-authored markdown files (the 213 generated ones under `eda/results/`
are untouched — `render_views.py` owns those). Four structural changes and a staleness sweep.

- **`Exp3_PTO_GRPO/CLAUDE.md` merged into the root `CLAUDE.md` and deleted.** Exp3 is the active
  experiment, so every landing had to be written into *both* files and the two drifted. The root
  file now carries the cross-experiment map **plus** the full Exp3 context (algorithms, trainer
  pattern, layout, EDA workflow, MCL, `EXPERIMENT_NAME`, training internals, Colab/local + sync,
  EDA extension points, gotchas) under an `Exp3_PTO_GRPO — the active experiment` heading; all
  paths rewritten repo-root-relative. **Exp1 and Exp2 keep their own** — they are frozen, so a
  per-directory file costs nothing to maintain. Inbound pointers fixed in `data.py`,
  `judge_plan.py`, `_local_smoke.py`, `README.md`.
- **Root `history/CHANGELOG.md` deleted.** It was a thin index whose every entry pointed here, i.e.
  a second place to write each landing for no reader benefit — the same duplication the CLAUDE.md
  merge removed. `history/` is now empty at root.
- **This file split.** The single changelog had passed 1,000 lines; it is now
  `CHANGELOG_EDA.md` (this file) + [`CHANGELOG_TRAINER.md`](CHANGELOG_TRAINER.md), along the
  EDA/trainer boundary that already existed inside it, with [`CHANGELOG.md`](CHANGELOG.md) kept as
  a stable 2-entry index so the ~15 inbound links keep resolving.
- **`DATA_README.md` folded into `README.md`** as a "Data & large artifacts" section — 27 lines
  that README already linked to twice.
- **The multi-judge numbers had four copies** (root `CLAUDE.md` §status, `LIMITATIONS.md` §1–§3,
  `L0/SUMMARY.md` §7, `METRICS_REFERENCE.md` §7b) — the doc map's own one-owner rule, broken, and
  the reason the ICC table in `LIMITATIONS` §1 still carried the pre-4-draw primary column while
  the table directly above it carried the corrected one. Split by *kind*, not by file size:
  **SUMMARY §7 owns the findings** (sign-preservation ladder, variance components, gain
  retention), **LIMITATIONS owns the measurement-quality evidence** (both judges' ICCs, agreement
  vs the attenuation ceiling, coverage + sweep provenance) and the caveats, **METRICS_REFERENCE
  returns to definitions-only** (the contract it states in §5 and broke in §7b), **root CLAUDE.md
  keeps the headline**. Each cites the others instead of restating.
- **Corrections found by the audit** (each verified against a tracked artifact, not against another
  doc): `LIMITATIONS` §1's primary-ICC column was 3-rep (Q1 0.981–0.994 / Q2 0.962–0.992 /
  MICI 0.895–0.958) where the 4-draw values are 0.982–0.994 / 0.955–0.992 / **0.864–0.943**; the
  MICI attenuation ceiling read 0.70–0.94 vs a measured max of 0.931; root `CLAUDE.md` still said
  `_selfcheck` 13/13 (**14**, confirmed by a full run: 14 passed, 0 failed, known means reproduce
  at PTO@10 4.26 / GRPO@10 3.75), still described the parquet fold as "not a read path"
  (it has been one since 2026-07-28), still said "6/6 contrasts" (**18/18** — six anchor pairs ×
  {Q1, Q2, MICI}, verified row-by-row against `multijudge_all_pairs_contrasts.md`), and still
  described the eval battery as "6 questionnaires" (**8 instruments** since PCT + MICI landed
  2026-06-14). Broken relative links in `METRICS_REFERENCE` (4) and `LIMITATIONS` (1), left over
  from the 2026-07-27 move into `eda/docs/`, now resolve. `L5/SUMMARY.md` pointed at
  `render_views.py` without its `tools/` prefix.
- **Newly documented, previously nowhere:** the `BOOT_SEED` rule in `eda/README.md` (every new
  `errorbar=` callsite must pass `seed=BOOT_SEED` or figures stop being reproducible), the
  `<judge>/` path level in `meetings/README.md`, and what the 14 self-checks actually cover.

---

**Landed (2026-07-28) — the tracked figures are reproducible; the fold becomes a read path.**
Two follow-ups to the score-lake migration below, both found while proving that migration changed
nothing.

**(a) Seaborn's bootstrap CIs were unseeded.** Re-rendering rewrote 90 PNGs on unchanged data. Cause:
seven `sns.lineplot`/`barplot` callsites pass `errorbar=("ci", 95)`, and seaborn 0.13.2 defaults to
`seed=None` with `n_boot=1000` — so every confidence band was a *fresh* bootstrap. Measured: three
consecutive renders of one notebook on identical data each differed by ~6% of pixels. The project's
own `stats.bootstrap_ci` was properly seeded, which is why the numeric tables never moved and this
went unnoticed; only the seaborn-drawn bands wobbled.
- Fix: `BOOT_SEED = 12345` promoted from a private `stats._BOOT_SEED` to `constants`, and passed at
  all seven callsites, so the figure side and the table side now share one seed.
- Consequence: `results/` PNGs stop churning in git on every render, and a thesis figure is
  reproducible rather than merely stable-looking.

**(b) `iter_conv_rows` now reads through the parquet fold.** The fold shipped as archival-only,
explicitly *not* a read path, on the argument that a second read path can drift from its source and
fail silently. That risk is real but it is the guard's job, not a reason to forgo a 5× speedup.
- New `eda_analysis/score_archive.py` owns the layout, the guard and the read path; the
  `tools/consolidate_scores.py` CLI is now a thin driver over it.
- **The guard:** `build` records a per-partition content signature in `_parquet/_manifest.json`;
  `rows_for` recomputes it and refuses to serve on any mismatch, falling back to the CSVs. Same
  (name, size, mtime) mechanism `load_cached` already trusts — not a new assumption — and the fold
  is only ever written by an explicit `build`, never by the scorers. Reads are therefore always
  correct and merely fast when current.
- **Equivalence gate before wiring it in:** all seven per-conversation loaders (`scores_long`,
  `load_items` ×2, `subscales`, MITI/MICI/PCT behaviour) produce frames identical under
  `assert_frame_equal(rtol=0, atol=0)` via either path. Speedups **4.3–6.1×** (`scores_long`
  86 s → 16 s). The residual is the per-row `Series` interface, not I/O.
- `iter_conv_rows` now yields in `file_index` order on BOTH paths; the CSV branch previously
  followed `os.listdir` order, which no caller depended on (they all group or aggregate).
- New `_selfcheck` check (14 total) asserts both halves: fold-equals-CSV row for row, and that a
  deliberately corrupted signature is refused rather than served.

---

**Landed (2026-07-28) — one score lake: `judge=` becomes an ordinary partition key.**
Scores lived in four stores under two different schemes. The primary oracle's reported draw sat
co-located per method (`data/{grpo,pto}_Exp3/eval_scores/`, no `judge=` or `rep=` level) while its
three ICC reps and the whole Haiku sweep sat in a separate local-only `data/eval_scores_by_judge/`
tree that *did* have both levels. So "where are gpt-4o-mini's Q1 scores for PTO@10" had two answers,
the primary was partitioned by a dimension (method) no other grader was, and `Arm.eval_dir()`
carried a primary-vs-other branch that `_selfcheck` pinned in place. Everything now lives in
`data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<id>.csv`.
- **Migration: 50,320 files, copy → hash-verify → delete.** Zero target collisions, all pairs
  byte-identical before any source was removed. The census came out symmetric — both judges hold
  22,272 files at `rep=0`, which independently confirms the two grids match cell-for-cell.
- **`rep=0` is now each judge's full-grid draw.** The primary's three anchor reps shifted to 1–3 to
  free it. That makes `judge_rep=0` mean the same thing for every grader, and it is why
  `EdaConfig.judge=""` now resolves to `PRIMARY_JUDGE_TAG` rather than to a special case.
- **The primary's ICC now spans four draws, not three** — the reported one included, matching how
  the second judge's was already computed. Free, and more honest: the question is how reproducible
  the *reported* number is. Only MICI moves (floor 0.895 → **0.864** at PTO@10); Q1/Q2 shift ≤0.007.
  The documented range is corrected to **0.86–0.99** in root `CLAUDE.md`, `LIMITATIONS.md` §1,
  `METRICS_REFERENCE.md` §7 and both `SUMMARY.md`s. **No headline result moves** — all 45 endpoint
  cells and all 25,056 score rows reproduce exactly, `_selfcheck` 13/13.
- **The lake is on Drive.** It was the one thing that mattered more than the layout: the second
  judge's $42 sweep and the $9.16 ICC reps existed only on one laptop, gitignored and backed up
  nowhere. `data/eval_scores` joins `grpo_Exp3`/`pto_Exp3` as a Drive symlink.
- **Parquet fold (`tools/consolidate_scores.py {build|verify|report}`).** 50,305 CSVs averaging
  ~190 bytes → 31 parquet files, 0.6 MB: 1,623× fewer files, 16× smaller, `verify` re-reads every
  CSV to prove it lossless. One-file-per-conversation is a *write-time* shape (a file is one
  completed unit, so an interrupted run resumes by skipping what exists) and a bad archival one.
  Deliberately **not** a read path — nothing in `eda_analysis` imports it, so a stale fold can never
  silently feed a figure; the EDA keeps reading the CSVs through its content-keyed cache.
- **Code.** `constants.EVAL_SCORES` replaces `EVAL_SCORES_BY_JUDGE`; `judge_partition_dir("")`
  resolves to the primary; `Arm.eval_dir` loses its branch and `Arm.eval_root` is deleted;
  `registry.eval_scores_root(judge_tag, rep)` replaces `eval_scores_root_for_method(method)`;
  `reliability.judge_reps()` is new and `available()` no longer returns vacuously True now that the
  primary is itself a `judge=` folder. `_selfcheck`'s judge-routing assertions invert: both graders
  must be siblings under the lake, and the primary must *not* be method-scoped.
- ⚠ **Batch manifests store `out_path` relative to the old root.** Harmless — `collect_batches`
  rebuilds paths from the live layout and treats the stored value as forensics only (all 14 batches
  are collected anyway), but do not resurrect it as a path source.

---

**Landed (2026-07-28) — the second judge's own ICC is measured; the MICI attribution is resolved.**
The last named validity gap. `reliability.agreement` computed the attenuation ceiling as
`sqrt(ICC_primary × ICC_judge)` with `ICC_judge` *assumed* equal to the primary's, collapsing it to
`ICC_primary`. That left MICI's weak cross-judge agreement (r 0.20–0.55) unattributable between
"Haiku is noisy" and "the judges disagree about the construct". Two further Haiku reps on the anchor
subset (4 model states × {Q1, Q2, MICI} × 96 convs, 2,304 calls, 0 errors) close it.
- **Measured: Q1 0.951–0.978, Q2 0.938–0.963, MICI 0.525–0.929.** Near-parity with the primary on
  Q1/Q2, so the assumption was sound there. Not on MICI: Haiku's repeatability falls as the
  MI-inconsistency rate rises (PTO Base 0.929 → PTO@10 0.815 → GRPO@8 0.749 → **GRPO@10 0.525**),
  i.e. it is least reliable on the arms the sycophancy claim concerns, where the achievable ceiling
  is **0.70** rather than the assumed 0.93.
- **Attribution: partly the judge's noise, mostly construct disagreement.** Against the corrected
  ceiling, agreement recovers Q1 86–91%, Q2 83–88%, MICI only **29–59%**. Haiku's own noise is a
  genuine contributor but accounts for a minority of the MICI gap, so the `LIMITATIONS.md` §2 MICI
  caveat stands and gain retention remains the load-bearing sycophancy evidence. No headline result
  changes; a stated limitation moves from assumed to measured.
- **A design check preceded the spend ($0.16, 40 calls).** The primary's reps differ by an explicit
  per-rep API `seed` at temperature 0.1; the Claude path passes neither (Anthropic has no `seed`
  parameter), so Haiku reps differ only by inherent API nondeterminism. Had it replayed
  deterministically, an ICC over those reps would have measured API determinism rather than judge
  reliability. Probed on MICI: 9/20 conversations differed, mean |Δ| 0.095 — informative, so the
  full run was justified. Worth repeating before any same-prompt rep purchase on a seedless API.
- **Cost: $9.16 actual, against "~$1–2" documented.** The old figure was an unchecked estimate;
  `judge_plan.estimate_cost` gives $9.16 direct / $4.58 batched (2,304 calls × 3,621 input + 71
  output tokens at $1/$5 per MTok). Run live rather than batched — under $5 of difference against
  minutes versus up to 24h. Corrected at the source with a note to price judge spend with the
  estimator.
- **Code.** `reliability.agreement` derives the second judge's ICC from disk where it has ≥2 full
  reps (new `_second_judge_icc`, which drops cells with lopsided rep coverage), computes the real
  `sqrt(ICC_p × ICC_j)`, and records `ceiling_basis`; new `icc_judge` column. The same `pearson_r`
  reads differently under the two bases, so the basis travels with the number. Cells with no ICC on
  either side are labelled as such rather than as an assumption that is not being applied.
- ⚠ **Consequence for §8:** the multi-judge analysis reads Haiku **rep 0 only**, and single-rep
  Haiku MICI on GRPO@10 is ICC 0.525. One-rep MICI on the high-MICI arms is indicative only;
  averaging the three anchor reps would resolve it for those four model states.

---

**Landed (2026-07-28) — the judge level is now SYMMETRIC: every grader gets a folder, the primary
included.** Lior's call. The layout had the second judge nested at `<family>/<judge>/` while the
primary rendered *flat* at `<family>/` — deliberate at the time (adding a grader moved no path the
thesis cited), but once Haiku had scored the same full 22,272-cell grid the asymmetry made the
**layout** assert something the project no longer believes: that one grader is the default and the
other an annex. A figure's path now always names the grader that produced it.
- **Naming: short model labels, both sides.** `gpt-4o-mini/` and `claude-haiku-4-5/` — so the
  existing `anthropic_claude-haiku-4-5/` folders were renamed too. NEW
  `constants.judge_dirname()` drops the provider prefix and any trailing ISO release date
  (`openai_gpt-4o-mini-2024-07-18 -> gpt-4o-mini`), generalizing to future judges; `judge_label` is
  kept as an alias. The **score** tree deliberately keeps the full `judge=<tag>` partition — a
  stable key there, a human-readable path here. NEW `constants.PRIMARY_JUDGE_TAG`.
- **One-line mechanism.** `exports._judge_sub()` returned `""` for the primary; it now returns
  `judge_dirname()`, which is never empty. Everything else composes through `_leaf()` already.
  `reset_results` lost its two primary-vs-judge branches (every judge has a leaf now), and `PRESERVE`
  became structural rather than an active name filter.
- **Bug found and fixed in passing:** `save_provenance` built its path from `_figures_root()`
  directly instead of `_fig_dir()`, so it never nested — meaning a `--judge` render **silently
  overwrote the primary's `_provenance.md` with the second judge's config**, in a file whose entire
  job is recording which config produced the figures. Now routed through `_fig_dir()`.
- **Second bug, caused BY the move and caught by verification:** `_write_xlsx_sheet` named the
  per-family Excel workbook after `os.path.basename(dir_path)` — which used to be the family and is
  now the JUDGE, so the first re-render emitted `gpt-4o-mini.xlsx` beside the stale `1_outcomes.xlsx`
  in every family. Now takes `parts[-2]` (the judge is always the leaf by construction of `_leaf`),
  which also keeps nested subgroups right: `2_questionnaires/mici/<judge>/` → `mici.xlsx`, as before.
  Cleanup: 28 mis-named primary workbooks deleted and rebuilt by a re-render; the second judge's 10
  were renamed off the old `anthropic_` tag rather than regenerated (content was already correct).
  Found by diffing the tracked file list at HEAD against the new one, mapping each old path to where
  the move should have put it. Final state: **671 → 675 files, 0 unaccounted for**, the only four
  genuinely new ones being the sign-preservation tables added the same day. Worth keeping as the
  pattern — git's own rename detection reported the duplicate workbooks as ordinary churn, so a bulk
  move is only verified when every old path is shown to have a new home.
- **Migration:** 32 judge-dir renames + 382 files moved into `gpt-4o-mini/`, across both views'
  figures and tables. Then 73 links rewritten in `L0/SUMMARY.md` (61), `L5/SUMMARY.md` (11) and
  `eda/README.md`; the deck builders gained a `_jp()` helper injecting the judge segment (one place
  each, not ~40 call sites) with `JUDGE = "gpt-4o-mini"` at the top to retarget a whole deck.
  Verified: 35/35 deck paths and 42/42 SUMMARY links resolve on disk.
- **Also caught:** `L0/SUMMARY.md` §1 pointed at `../../METRICS_REFERENCE.md`, stale since the
  2026-07-27 `eda/docs/` reorg — the link checker written for this migration found it.
- `_selfcheck`'s judge-routing case was inverted accordingly: it used to assert the primary stays
  flat, and now asserts the two judges are **siblings** under one family dir with distinct labels.

---

**Landed (2026-07-28) — the judge evidence reaches the narrative docs; two numbers surfaced that
were computed but never written up; provenance unified on the tracked `L0` view.** No new figures,
no new API spend — a pass over the hand-authored layer, which had fallen behind the 2026-07-27 sweep.
- **`results/L0/SUMMARY.md` — the primary read — had ZERO mention of the second judge** (no "judge",
  "Haiku", "retention", or "dependability" anywhere), even though root CLAUDE.md designates it as the
  full narrative and the judge work is the strongest validity evidence in the thesis. New **§7 "Does
  the result survive a different judge?"** carries sign preservation, the variance decomposition, the
  MITI warning, gain retention, and the retention trajectory; old §7 Caveats → §8, with the folklore
  "oracle noise ≈ 0.10" line replaced by the measured ICC.
- **Two numbers existed in the artifacts but in no document.**
  (a) **Full-grid sign preservation.** The docs quoted "18/18", which is `contrasts()` over six
  hand-picked pairs × 3 metrics. `all_pairs_contrasts` had meanwhile enumerated **1,848** arm×metric
  contrasts in `L0`: **88.3%** keep their sign, **94.1%** at |Δ|≥0.10, **97.0%** at ≥0.25, **98.9%**
  at ≥0.50 (94.7% among the 1,299 with a judge-side CI excluding zero). The judges disagree *only*
  about differences too small to claim. Per rubric MITI is worst at **77.5%** — an independent
  confirmation of its 0.65 dependability, from a different statistic. L5: 160/168 = 95.2%, and
  102/102 at |Δ|≥0.10.
  **Made reproducible, not just quoted:** NEW `reliability.sign_preservation(pairs, thresholds, by)`
  + two tracked tables (`multijudge_sign_preservation{,_by_metric}.md`) saved from notebook 5 §8b.
  Writing a doc number that no artifact produced is the same provenance bug fixed below, so the
  ladder was turned into code in the same pass rather than left in a scratch script.
  The per-rubric ladder then **sharpened the MITI limitation**: every other rubric reaches 95.5–100%
  sign preservation once |Δ|≥0.25, MITI only 88.2% (needing ≥0.50 for 97.6%). MITI is therefore not
  merely noisier — it is the one rubric where a *claimable-size* difference can still flip sign under
  a different grader. ⚠ Thresholds are absolute, so the ladder is readable down a rubric but not
  across them (PCT/MICI are 0–1, the rest 1–5 or 1–7); caveat is in the table caption and both docs.
  (b) **Gain retention is an ONSET curve.** Only the endpoint was ever reported. Per iteration, PTO's
  Q1 retention holds 0.80–0.98 for the whole run while GRPO decays monotonically in trend
  (I3 0.89 → I6 0.57 → I9 0.03 → I10 0.28), the two arms indistinguishable for three iterations.
  `multijudge_retention_trajectory.png` already existed. This is a stronger statement than the
  endpoint contrast: the held-out grader withdraws credit *progressively*, as drift onto
  grader-specific features predicts.
- **Provenance unified — THREE different scopes were in circulation.** `LIMITATIONS.md` §2's variance
  table quoted 29-arm figures from the pooled `all` view — **retired to gitignored scratch
  2026-07-27**, so no longer reproducible from any tracked artifact — while §3's retention table was
  already `L0`-sourced, and `METRICS_REFERENCE.md` §7b quoted a *third* set (0.5–4.6% arm×judge,
  dep 0.91–0.98) from the 4-anchor measurement, spliced onto full-grid retention values in the same
  paragraph. All three now cite `L0` (22 arms, every cell n=96) and say so: CSQ 70.6 not 72.6, MITI
  **3.6% arm / 94.5% judge / dep 0.65** not 4.7/93.0/0.68, Q1 10.9% dep 0.90, arm×judge 1.2–6.9%.
  Same story throughout, ±1–2 points. Root CLAUDE.md's block re-cited to match and labelled.
  `METRICS_REFERENCE.md` §7b also gained a `pct_same_sign` row for the new ladder.
- **Stale text fixed.** `LIMITATIONS.md` §1(b) still described the sweep as "landed 43%" with
  partials "quarantined" — coverage has been 232/232 since 2026-07-27. Rewritten; the 43% forensics
  kept as a blockquote because it explains why `filter_complete_cells` remains in the pipeline.
- **Broken links fixed + a gap named.** `results/L5/SUMMARY.md` linked to `../all/` three times.
  Removing them exposed that **RQ-i (K0 vs K5) has no tracked artifact at all** — `k_paired_by_method`
  only exists in the retired `all` view. §3 now says so explicitly and records the decision to make
  when the K=5 arms resume (promote `all` back, or move the table into `L5`). Also added an L5 §5
  second-judge check and refreshed both views' caveats.

---

**Landed (2026-07-26) — judge reliability MEASURED: oracle ICC + a decoupled Claude second judge;
LIMITATIONS §1–§2 bought down.** `Judge_Reliability.ipynb` had been ready-but-never-run since
2026-07-16 (blocked on a missing Anthropic key). Ran it end-to-end on the anchor subset (4 models ×
{Q1, Q2, MICI} × 96 convs = 4,608 calls, ~$5.30, 0 errors).
- **Results.** Oracle ICC(2,1) **0.90–0.99** (mean |Δ| 0.03–0.09 — confirms the "≈0.10 oracle
  noise" folklore as a conservative bound). **Claude Haiku 4.5** as the decoupled second judge:
  **6/6 endpoint contrasts keep their sign**, and it *widens* the headline PTO−GRPO Q1 gap (+0.77
  vs the primary's +0.53). Q1/Q2 cross-judge r 0.80–0.88 against a ~0.98 attenuation ceiling.
  Haiku is systematically harsher (Q1 −1.25…−1.74) — a LEVEL shift that cancels in contrasts.
  **MICI is the weak spot**: r 0.20–0.55 across families despite the primary oracle's own
  ICC 0.90–0.96, so the sycophancy claim is now stated at the contrast level, not as a rate.
- **Two real bugs found before spending.** (a) `_strip_unsupported_constraints` dropped
  `minItems`/`maxItems` for Claude's `json_schema`, which for the ARRAY-shaped rubrics (Q1/Q2/WAI/
  CSQ8/MI-SAT) removed the only guarantee of one score per item — a wrong-length array fails
  `parse_json_response`, gets swallowed, and silently drops the conversation. Now folded into
  `description` instead of deleted (verified: 24/24 smoke, then 1,152/1,152 clean). (b) `JudgeSpec`
  gained a `thinking` passthrough: Sonnet 5 / Opus 4.8+ run ADAPTIVE thinking when `thinking` is
  omitted, billing against the same `max_tokens=1024` → truncated JSON. Haiku 4.5 needs no config;
  the notebook now carries the warning inline.
- **New split: paid scorer vs free presenter.** `Judge_Reliability.ipynb` only *scores*;
  the new **`eda_analysis/reliability.py`** (+ **`plotting/reliability.py`**: `oracle_repeatability_bars`,
  `judge_agreement_scatter`, `judge_contrast_bars`) *reads* `data/judge_check/` with no API calls,
  and **`5_Training_and_Reliability` §7** renders it — so this lands inside `render_views.py` and
  `results/<view>/` like everything else, instead of only in gitignored `data/judge_check/summary/`.
  Mirrors the `Run_Eval` → notebooks 1–7 split. Section self-skips in the L5 view (subset is all K=0).
- **Also.** `anthropic==0.116.0` + `tabulate` added (`gen_requirements.py`); `anthropic_key.txt`
  added to `.gitignore` (it matched NO existing pattern — `*.key` doesn't cover `*_key.txt`);
  `LIMITATIONS.md` §1/§2 rewritten with measured numbers + what's still uncovered (no human coder;
  sampling-noise-only ICC; 3 of 8 rubrics; generator still coupled); `METRICS_REFERENCE.md` §7 new.
  `_selfcheck` 10/10; notebook 5 re-rendered for L0 + L5.

---

**Landed (2026-07-26) — the "orthogonal axes" framing is retired; PCT/MICI/ratios are just more
evaluation metrics.** Lior's call: the group was never orthogonal — `PCT` correlates with the five
global-eval rubrics at ρ≈0.79–0.94 (the EDA had already recorded this in the §1 captions of
`3_Validity_and_Hacking`, but the *label* stayed), and `MICI` is lower-is-better rather than
independent. **Decision: no collective name at all** — report all eight instruments flat, with MICI
flagged ↓. Parts:
- **Code.** `constants.ORTHOGONAL_METRICS` → **`EXTRA_METRICS`** (same 5 members: PCT, MICI, R:Q,
  %CR, %MICO) + a docstring stating it is a *membership list only* — plot order and the factor
  space — asserting nothing about independence. Renamed at all call sites (`stats.py` ×2,
  `plotting/outcomes.py`, `__init__.py` export + `__all__`, `3_Validity_and_Hacking` cell).
- **Figure text.** `rubric_correlation_heatmap`'s right-hand block label "Orthogonal axes" →
  **"PCT · MICI · MITI ratios"** (name the block by content, not by a claim); same for the
  `factor_loadings_bars` legend patch. Docstrings in `structure.py` / `trajectories.py` /
  `outcomes.py` / `data.py` / `scoring/{pipeline,registry}.py` reworded.
- **Prose.** Notebooks 1/3/7 markdown + `save_fig`/`save_table` captions, `eda/README.md`,
  `METRICS_REFERENCE.md` §1 (the `EXTRA_METRICS` bullet now states the PCT finding inline),
  `LIMITATIONS.md` §4 heading, root `README.md`, and all three hand-authored
  `results/<view>/SUMMARY.md` (L0 §1/§3, L5 §2, all §1/§3) — each now says the second factor is
  **MICI + the MITI ratios, not PCT**. The only surviving occurrences of the word are the two
  "do NOT call these orthogonal" guard comments.
- **Re-rendered** notebooks 1/3/7 × views all/L0/L5 (`render_views.py`); `_selfcheck` 10/10 incl.
  the headline-mean pins (PTO@10 4.26 / GRPO@10 3.75). Dated changelog entries left untouched —
  they record what was believed at the time.
- **Also this day (not EDA) — new `meetings/` tree.** The supervisor-facing files had accumulated
  loose in the Exp3 root (2 builder scripts + 4 decks + an email draft); they now live under
  `meetings/` = `build/` (the generators + NEW `export_pdf.ps1`, pptx→pdf via PowerPoint COM,
  `-Png` for layout checks) + one **dated folder per meeting** holding that meeting's deck and
  anything sent with it, described in NEW `meetings/README.md`. Both builders now resolve `ROOT`
  off `__file__` instead of a hard-coded absolute path, and each `OUT` names its dated folder.
  `build_supervisor_deck.py` was path-updated but **not re-run** (re-running would overwrite the
  historical 2026-07-16 deck with today's re-rendered figures).
  The new deck itself: NEW `build_results_snapshot.py` — a lean 11-slide results-only deck for the
  publication-meeting email (results only, minimum interpretation; opens with an ICLR reminder
  slide + an Exp1/Exp2/Exp3 lineage slide carrying the 4-bit-vs-bf16 score-axis caveat), plus
  `email_draft_2026-07-26.md`. Layout tree in [../CLAUDE.md](../../CLAUDE.md) updated.

---

**Landed (2026-07-16) — EDA reorg: 7 tier-based families + `0_headline/` + generic questionnaire
item detail + final-vs-best everywhere.** Complete reorganization of the notebook/results layer
around a drill-down hierarchy (Level 1 global scores → Level 2 inside-each-questionnaire → Level 3
cross-cutting), with every endpoint artifact reported as a **final + best pair** (best = own-oracle
peak via `best_per_experiment`; GRPO_LA0→I8). Package architecture untouched. Parts:
- **(A) Families/notebooks renumbered 1..7** (`git mv`, 1:1 convention kept): `1_Outcomes` (main
  grid + forest/bars/scorecard final+best) · **NEW `2_Questionnaire_Detail`** · `3_Validity_and_Hacking`
  (ex-`3_Mechanism` minus the per-questionnaire sections; session shape now exported —
  `session_shape.png` + `session_end_reasons`) · `4_Heterogeneity` (endpoint bars final+best) ·
  `5_Training_and_Reliability` · `6_Preference` · `7_Stats` (+ NEW `method_paired_best`: PTO@best
  vs GRPO@best persona-paired across different iterations — PTO wins even vs GRPO's iter-8 peak,
  Q1Q2 +0.18 dz 0.30 p_holm .01).
- **(B) Notebook 2 — one uniform detail section per rubric**, all in the `trajectories_all_metrics`
  small-multiples style: per-item grids for Q1(5)/Q2(17)/WAI-SR(12+3 subscales)/CSQ-8(8)/MI-SAT(6)
  via NEW generic `data.load_items` over `constants.ITEM_QUESTIONNAIRES` (per-item columns were
  already on disk — no oracle re-run; `load_q2_items` now wraps it), MITI detail grid (4 globals +
  7 per-turn rates + R:Q/%CR/%MICO via NEW `behavior.miti_detail_by_iter`; old `behavior_drift` is
  its subset) + 4.2.1 thresholds moved here, PCT + MICI detail grids moved here. Every rubric also
  gets "which items drive the change" delta bars at final AND best (NEW generic
  `stats.item_endpoint_deltas` + `plots.item_trajectory_grid`/`item_delta_bars` in NEW
  `plotting/questionnaires.py`; the Q2 figures delegate to them).
- **(C) `0_headline/` family** — the ~7 presentation artifacts re-saved by notebooks 1–3 via
  per-call `group="0_headline"` (main grid, forest final+best, MITI+MICI detail grids, reward-hack
  panel, scorecard) — always in sync, no extra notebook.
- **(D) Clean rename** — stale `results/<view>/` trees purged + re-rendered (L0/L5/all); refs
  updated in SUMMARY.md ×3, eda/README, both CLAUDE.md, METRICS_REFERENCE, LIMITATIONS,
  `build_supervisor_deck.py` (paths only, not re-run; now at `meetings/build/`). New cache names only (`items_*`,
  `miti_detail_by_iter`, `session_shape_by_iter`) — existing cached frames untouched; `%MICO` added
  per-conv in `load_miti_behavior` (uncached path). `_selfcheck` 10/10 incl. headline-mean pins.

---

**Landed (2026-07-13) — EDA structural refactor: `oracle_scoring/` folded into
`eda_analysis/scoring/` + `plotting.py` split into a topic subpackage.** Pre-writing polish pass
(no behavior change; `_selfcheck` extended + 10/10 incl. the known headline means). Three parts:
- **(A) One package.** The legacy `oracle_scoring/` package (the Run_Eval + Judge_Reliability
  backend) moved into `eda_analysis/scoring/` with purpose-named modules — `config.py`→`registry.py`
  (kills the duplicate `config.py`/`data.py` names across two packages), `data.py`→`conversations.py`,
  `eval.py`→`pipeline.py` (stops shadowing the `eval` builtin), `judge_check.py`→`judge.py`. Renames
  with the fold: `EDAConfig`→`ScoringConfig` (ended the near-collision with `EdaConfig`); the
  registry's `discover_arms` import is now a clean intra-package relative import (the old
  cross-package `sys.path` hack is gone); `DATA_DIR`/workspace-root resolution deduplicated onto the
  `constants` leaf. `scoring/` is deliberately NOT imported by `eda_analysis/__init__` (its registry
  scans disk; analysis notebooks never need it). Both scoring notebooks' import cells updated;
  Run_Eval's stale header (Conv_EDA / pto_Exp2 references) fixed.
- **(B) `plotting/` subpackage.** The 935-line, 27-figure `plotting.py` split by topic —
  `outcomes` / `trajectories` / `heterogeneity` / `structure` / `behavior` / `training` (+ a tiny
  `_shared` leaf for `_metrics` + the qualitative palette) — behind a re-exporting `__init__`, so
  the public surface (incl. the `figures`/`plots` aliases and the re-exported `plotting_style`
  helpers) is byte-for-byte compatible; the `figures = sys.modules[__name__]` self-alias hack is
  gone (submodules import the style helpers directly).
- **(C) Polish.** `render_views.py --nb` now takes the notebook/family NUMBER (1..6, `--nb 3` =
  `3_Mechanism`) instead of list indices 0..5; `_selfcheck` gained a scoring-surface check (31
  public names + everything the two scoring notebooks reference); README/CLAUDE.md maps updated
  (roadmap's last open item — the fold — closed).

 Triggered by the question "is the
'warmth' split an official thing?" — answer: no, it's an empirical halo/redundancy set, and Q1/Q2's
provenance is the lab's own **CLPsych 2024** paper (Yosef, Zisquit, Cohen, Brunstein Klomek, Bar &
Friedman, *Assessing Motivational Interviewing Sessions with AI-Generated Patient Simulations*, ACL
Anthology 2024.clpsych-1.1 — verified; validates the prompts AS LLM evaluators). Four deliverables:
- **(A) Relabel** "warmth" → "global-evaluation (halo) cluster" everywhere thesis-facing: SUMMARY.md
  (L0+all), METRICS_REFERENCE (new per-instrument provenance block w/ the CLPsych citation),
  LIMITATIONS §3–4, figure-embedded text in `plotting.py` (heatmap block label, PC1 titles,
  reward-hack panel axes/suptitle, `_SHARED_FACTOR_CAVEAT`), notebook headers + captions
  (`1_Outcomes` §1/§5, `3_Mechanism` §3/§4a, `6_Stats` §0). `WARMTH_RUBRICS` kept as historical code
  name (documented as such).
- **(B) Official MITI 4.2.1 competency thresholds** (verified against the CASAA manual PDF §H–I:
  R:Q 1:1/2:1, %CR 40%/50%, Technical 3/4, Relational 3.5/4 — expert opinion, 20-min-session
  domain caveat): `constants.MITI_THRESHOLDS`, official Technical/Relational summary scores (the
  2-global splits, not `MITI_GlobalMean`; MITI2 now loaded), `behavior.miti_proficiency_by_iter`
  (cached), `plots.miti_threshold_panel` + `miti_threshold_table`, new `3_Mechanism` §2b. First
  numbers: both L0 arms go below-competence → fair-to-good on globals (Relational crosses "good"),
  **neither reaches "good" on the technique ratios**; GRPO's iter-10 R:Q 1.43 "fair" is the
  pathological fewer-questions route.
- **(C) Q2 item-level reward composition** (free — per-item `Q2_1..17` already stored):
  `data.load_q2_items` (cached), `stats.q2_item_endpoint_deltas`, `plots.q2_item_delta_bars` +
  `q2_item_group_trajectory`, `constants.Q2_ITEM_SHORT`/`Q2_ITEM_GROUPS` (face-content grouping,
  explicitly NOT a validated subscale), new `3_Mechanism` §4f. First numbers: **"revealed his
  thinking" (self-disclosure) tops BOTH arms' Δ ranking** — the Q1+Q2 reward composition itself
  incentivizes the emotive drift (items 1/2/3/10 reward self-disclosure MI doesn't prescribe).
- **(D-ready) Judge-reliability pipeline** (LIMITATIONS §1–2, ready to run, no spend yet):
  `oracle_scoring/judge_check.py` + `Judge_Reliability.ipynb` — Part 1 oracle repeatability (3 reps,
  per-rep seeds — the pipeline's pinned seed=42 would fake perfect ICC) → ICC(2,1) + mean|Δ|; Part 2
  pluggable second judge (Claude via `anthropic` SDK [installed, 0.116.0; structured output via
  `output_config.format`, bounds stripped] or OpenAI) → agreement + the PTO−GRPO
  **contrast-preservation** check. Gated behind RUN_* flags; cost preview in-notebook; outputs to
  `data/judge_check/` (never eval_scores). Model choice = Lior's (default knob claude-haiku-4-5).
  Validated: `_selfcheck` 9/9 (68 notebook refs), offline smokes for B/C on real data + judge-check
  synthetic ICC/agreement; L0+L5 `1_Outcomes`+`3_Mechanism` re-rendered.

**Landed (2026-07-12) — docs refactor: one owner per fact.** The hand-maintained markdown set was
deduplicated around a single-source-of-truth rule (run status + headline numbers + cost constraint →
root CLAUDE.md "Current status & next step"; detailed eval narrative → `eda/results/<view>/SUMMARY.md`;
EDA how-to + module map → `eda/README.md`; metric definitions → `eda/METRICS_REFERENCE.md`; dated
history → this file). Exp3 CLAUDE.md ~600→~420 lines ("Eval results so far" → pointer block, "Run
status" → durable LA5-resume facts + pointer, the 2026-06-01/03 dependency audit moved HERE, the EDA
workflow + module map deduped vs eda/README, look-ahead intuition stated once). The root
`history/CHANGELOG.md` became a thin dated index into this file; its root-only details were merged in
below first (the 2026-07-08 results entry, the 2026-06-14 orthogonal-axes thread, the 2026-06-09
first-results snapshot, the 2026-06-04 trainer batch, the 2026-06-01/03 dependency audit).
eda/README + METRICS_REFERENCE inline result numbers → pointers; root CLAUDE.md gained a "Doc map"
ownership table. No content deleted — only moved or replaced by a pointer to its owner.

**Landed (2026-07-11) — roadmap #7 DONE: Run_Eval's `EXPERIMENTS` registry auto-generated from
`discover_arms()`; EDA backlog now clear.** `oracle_scoring/config.py` builds the registry at import
via `build_experiments_from_disk()` — one entry per `model_iter_N` conv dir per discovered arm, paths
experiment-root-relative, warning if discovery finds nothing (Drive symlinks offline). The
hand-maintained list (and the pre-staged commented LA2 block) is gone; a new run is scoreable as soon
as its conversations land. `discover_arms()` now also **skips empty `model_iter` dirs** (no
`conversation_*.csv`) — in-flight/paused generation leftovers are not data points. Verified: the
auto-registry reproduces the exact 29 known model states (and correctly EXCLUDES `PTOExp3_LA5_I5` —
`model_iter_5` is an empty paused-mid-generation dir, which the old hand-list's "in flight" comment
knew but 2026-07-11's doc audit initially mis-read as scoreable data); `_selfcheck` 9/9 (known means
reproduce). Same session: docs corrected on the true LA5 pause state (PTO adapters 1–5 trained /
I1–I4 scored / iter-5 eval convs never generated / iteration_6 stopped at pref_pairs; GRPO iteration_2
adapter-less) — folder presence ≠ data.
**Morning batch (hardening):** new `eda_analysis._selfcheck` regression guard (invariants + known
headline means + cache round-trip; run after any EDA change); committed notebooks made output-clean
(`strip_notebook_outputs.py` + `nbstrip` git clean-filter); dead data-module submodule aliases retired
(`discovery`/`personas`/`scores`/`select` — `figures`/`plots` kept); `plotting.py` split into named
figures + a `plotting_style` helper sibling (re-imported, public surface unchanged); **parquet cache**
for `scores_long` + `behavior_by_iter` (roadmap #5 — content-keyed on input CSVs → `eda/.eda_cache/`,
on by default, `EDA_NO_CACHE`/`EdaConfig(cache=False)` bypass); `meetings/` folder + stale references
removed; Exp3 CLAUDE.md pruned to a lean current-state doc (851→579 lines, dated narratives moved
HERE) + a currency pass on root + Exp3 CLAUDE.md. **Afternoon batch (the "Exp3 EDA (1/8)…(8/8)"
series + neighbors):** dead-code sweep across `eda_analysis` + `oracle_scoring` (incl. stale TRL
comment fix); **`oracle_scoring/` pruned to ONLY the Run_Eval scoring path** (config/data/eval;
1279→904 lines — analysis lives in `eda_analysis/`); half-wired L2 view removed (re-add is one line
in `_VIEW_KS`); `.emb_cache` relocated out of the package source to `eda/.emb_cache/`; **`constants.py`
leaf extracted** — broke the `__init__`↔submodule import cycle (submodules now import the leaf
top-level; ~20 deferred in-function imports gone); the 5× duplicated per-conversation CSV loader
unified (`data.iter_conv_rows`, also used by `behavior.py`); `RE_AFFIRM` shared via the leaf + unused
notebook imports trimmed; docstring-currency pass; eda README refactored to a pure guide (DRY vs
CHANGELOG/SUMMARY); L0 + L5 result views fully re-rendered. Data state unchanged (full L0, partial L5).

**Landed (2026-07-08) — GRPO LA0 FINISHED (10 iters) + re-scored: the fair-endpoint PTO-vs-GRPO
comparison is in hand.** *(Detailed eval narrative moved here from CLAUDE.md's "Eval results so far"
2026-07-12; the living numbers are in `eda/results/<view>/SUMMARY.md`.)* Scored on the full battery
incl. the orthogonal axes (PCT, MICI, derived R:Q/%CR/%MICO): PTO LA0 iters 0–10, GRPO LA0 0–10,
PTO LA5 0–4, GRPO LA5 0–1. (MI-SAT re-scored 2026-07-07 under corrected goal-agnostic wording; means
rose uniformly ~+0.14, no headline changed.)
- **Each arm vs base — large warmth gains.** PTO LA0 Q1+Q2 3.00→**4.26** (final=best, dz 1.43,
  Friedman W=0.45); GRPO LA0 3.07→**4.08 at its iter-8 peak**, falling to **3.75 by iter 10** (final
  dz 0.72, best dz 1.22, W=0.33); PTO LA5 3.00→3.89 in 4 iters (dz 0.88). All warmth rubrics large,
  Holm p≈0 everywhere.
- **PTO vs GRPO (RQ-ii).** The earlier "near-tie at iter 8" was a snapshot artifact: GRPO Q1Q2 peaks
  at iter 8 (4.08) then REGRESSES (iter9 3.81, iter10 3.75) while PTO climbs stably (4.22→4.26). At
  the matched 10-iter endpoint **PTO beats GRPO 4.26 vs 3.75** (paired +0.51, dz +0.73, Holm p<0.001;
  MITI/CSQ-8/MI-SAT/PCT also favor PTO, and PTO is less MI-inconsistent). Overall OLS slopes GRPO
  0.072/iter (peak iter 8) vs PTO 0.120/iter (peak iter 10); earlier matched-iter reads still hold
  (tie 1–2, GRPO briefly ahead @3, PTO ahead @8). ⇒ Revised core answer: GRPO is competitive *up to
  its peak* but overshoots into reward-hacking and degrades; PTO sustains gains across 10 iters. With
  GRPO, peak-iter selection / early stopping matters — its best (4.08 @8) is still below PTO's best
  (4.26 @10).
- **Conversation-level mechanism (iter-10, same resistant persona).** GRPO iter-10 collapses into
  nonstop empty praise and never gives the practical advice the patient demands 6+ times; PTO iter-10
  also drifts toward affirmation but converges to concrete steps and the patient softens. Across all
  96 iter-10 convs: GRPO 0.13 q/turn, 3.61 praise-words/turn vs PTO 0.50 q/turn, 1.02 praise/turn —
  the iter-10 eval regression IS the over-praise reward-hack the full-conv oracle penalizes.
- **Reward-hacking / multi-skill.** MICI rises with warmth: base 0.21 → 0.49 PTO (~2.3×, dz 0.78) /
  **0.84** GRPO (~4×, dz 1.72) at the endpoint (GRPO's iter-8 peak was still 0.54, dz 0.89, before
  the late regression blew it up). Affirmation drift in BOTH methods, worse in late GRPO (B6_AF
  0.52→**1.98**, B3_Q 6.4→**4.1**, q/turn 0.83→**0.15**, R:Q→**1.44** by iter 10; mid-run GRPO looked
  *more* reflective, R:Q 1.04 > PTO 0.75); PTO's drift is milder and plateaus (B6_AF 1.64, q/turn
  0.55). PCT rises modestly, more for PTO (0.49→0.63 medium vs GRPO →0.57 small). Both kill
  degeneration loops (loop% 0.49→0). Adding the orthogonal axes drops PC1 ≈91%→≈55% (PC2 ≈16%; PCT
  loads ~0.39 on PC1 — change-talk co-moves with warmth).
- **PTO preference probe is real:** wins_correct 0.65→0.71 over iters, strengthening late. **K0 vs K5
  (RQ-i): still preliminary** (PTO LA5 4 scored iters, GRPO LA5 1); both LA5 arms paused for cost.
Same day: the 20-commit EDA hardening/refactor pass — see the 2026-07-11 entry above.

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

**Landed (2026-07-07) — backlog #7 (general review) DONE + judge-prompt fixes + honest advantage signal.**
Two commits (`f5e5d63` framing/EDA batch; MI-SAT re-score results follow). Driven by a 3-reviewer sweep +
Lior's handoff (verdict: methodology sound, remaining risk is *write-up framing*, not code; excluded:
no CoT judge fields, no Q1/Q2 edits). **(A) Judge/oracle.** MI-SAT items were hard-coded to "diabetes"
(personas are smoking/obesity) → reworded goal-agnostic in [questionnaires.py](../code/questionnaires.py) and
**re-scored all 2,784 convs** (0 errors); means rose **uniformly ~+0.14** (old diabetes wording rated an
intervention that never happened) — every relative conclusion preserved (PTO_LA0 still leads). **(B) Honest
advantage signal.** Added an UNFILTERED PTO `group_range` (best−worst over a branch's M candidates) beside
GRPO's as the true like-for-like analog to the τ-filtered `margin`. **Caught a grouping bug mid-work**
(PTO `branch_id` is the trunk *depth* and collides across conversations — verified in
[pto_trainer.py](../code/PTO_Exp3/pto_trainer.py); the naive key pooled cross-conversation spread) → fixed by
keying on `conversation_id` too. Corrected finding: per-branch spread is modest+comparable (~0.23 PTO vs
~0.29 GRPO); the τ-filter mildly *inflated* PTO's apparent decisiveness (so the old "comparable ~0.3" read
was margin-vs-range, not like-for-like). K=5>K=0 holds on both. **(C) Rate-normalization.** MITI behavior
counts now shown per therapist turn (`B*_per_turn` in `behavior.py`; drift figure switched) so length
doesn't inflate them. **(D) Framing (notebook markdown, no data change):** confirmatory-vs-exploratory split
(`6_Stats` §0; confirmatory = PTO>GRPO on Q1+Q2 at **final AND best** iter + vs-base + reward-hack orthogonal
contrasts); reward=outcome + shared-oracle confounds named (`3_Mechanism` §4, anchored on reward-independent
deterministic text metrics); **PCT loads WITH warmth** (ρ≈0.79–0.94, NOT orthogonal — fixed contradicting
loadings captions, `3_Mechanism` §3); K0-vs-K5 descriptive-only banners (`4_Training` §4, `5_Preference` §2);
PCA-mechanical + bootstrap-seed caveats (`6_Stats` §5); new [eda/LIMITATIONS.md](../eda/docs/LIMITATIONS.md).
**(E) Hardening/cosmetic:** deleted dead buggy `rank_table`; `omnibus` eps_sq→eta_sq (η²_H mislabel);
`behavior_by_iter` orphan-row warn; advantage/reward-distribution colors keyed to arm palette (PTO cool/GRPO
warm) + `arm_label` titles; `render_views` split VIEWS (allowed) vs DEFAULT_VIEWS (bare run = all/L0/L5,
explicit L2 still valid). Re-rendered all 3 views twice (no failures); 15 stale raw-count behavior figures
removed. Data state unchanged (full L0, partial L5, no L2).

**Landed (2026-07-03) — grid+subfolder pattern extended to all multi-panel families (backlog #1 DONE).**
Applied the `1_Outcomes` combined-grid + per-metric-subfolder pattern across `2_Heterogeneity` /
`3_Mechanism` / `4_Training`, adding whichever half each family lacked. **2_Heterogeneity:** new combined
**all-metrics overview per trait** — `2_heterogeneity/<trait>_all_metrics.png` (metric×arm trajectory grid,
each cell split by persona category, shared legend; new `plots.heterogeneity_overview_grid`) alongside the
existing per-metric `<trait>/<metric>.png` subfolder; became §1 (overview→detail), later sections renumbered.
**3_Mechanism:** per-metric behavior **subfolder** `3_mechanism/behavior/<metric>.png` (new
`plots.single_behavior_trajectory`) beside the combined `behavior_drift`; per-parent subscale **subfolder**
`3_mechanism/subscales/<parent>.png` (reuse `subscale_trajectory_grid(parents=(p,))`). **4_Training:**
per-arm reward-distribution **subfolder** `4_training/reward_distribution/<arm>.png` (reuse
`reward_distribution` on a one-arm slice) beside the combined `reward_distribution_by_arm`. Thin arms
auto-dropped (GRPO_LA5, base-only, correctly absent from the L5 overview → single-column PTO grid).
Validated: 3 views × 3 edited notebooks via `render_views.py` (`thesis-venv313`), no failures; all new PNGs
present in all/L0/L5.

**NEXT EDA SESSION — backlog (2026-07-02, Lior's notes; START by asking clarifying questions).**
Data state: **full L0 (PTO_LA0 + GRPO_LA0, 0–10) + partial L5 (PTO_LA5 0–4, GRPO_LA5 base)**; no L2 data yet.
1. ✅ **DONE (2026-07-03)** — see the Landed note above. Combined all-metrics overview per trait added to
   `2_Heterogeneity` (metric×arm trajectory grid); per-metric/-parent/-arm subfolders added to `3_Mechanism`
   (behavior, subscales) and `4_Training` (reward distribution). Scope chosen: **all** multi-panel families.
2. ✅ **DONE (2026-07-03) — best−worst range.** `training.advantage_signal_by_iter` now also emits GRPO
   `group_range` = per-group **best − worst** candidate reward (computed from the group's own candidate
   scores in `generations.jsonl`), the direct analog to PTO's chosen−rejected `margin` — both are 0–5
   oracle-score gaps, so `advantage_signal_sidebyside` now plots them on ONE **shared y-axis** (GRPO range
   solid + `group_std` faint; PTO margin solid + median faint). Findings: PTO K=0 margin declines steadily
   (~0.32→0.27); GRPO K=0 range dips then **rebounds late** (0.38→0.23→0.34, echoing the iter-8 hack);
   **PTO K=5 margin (~0.41–0.47) > K=0** — look-ahead makes the oracle discriminate candidates more decisively.
3. ✅ **DONE (2026-07-03) — NOT a bug; semantic gap.** The "4.1 vs 0.15" was count-vs-rate confusion (B3_Q is
   a per-conv COUNT, q_per_turn a RATE). Harmonized (both /turn, SAME denominator), the merge is conv-aligned
   96/96, so no computation error. The real divergence: regex literal-`?` collapses ~7× for GRPO (12.4→1.7/conv)
   while oracle B3_Q drops only ~1.6× (6.4→4.1) — question **syntax** vs **function** (late affirmation/advice
   turns carry question-function without a `?`). Shipped: (a) alignment guard in `behavior.question_rate_crosscheck`
   (warns if the inner-join drops >10% of convs — catches a future persona-shuffle mis-join); (b) disambiguated
   labels (`B3_Q`→"Questions / conv (MITI)", `q_per_turn`→"Questions / turn (regex ?)"); (c) fixed the §4b
   caption/markdown that OVERSTATED agreement (they diverge — the widening gap IS the drift signature).
4. ✅ **DONE (2026-07-03) — acronym + descriptive.** `DISPLAY_NAMES` now keeps the validated-instrument
   acronym up-front with the gloss in parens: `MITI (MI Integrity)`, `CSQ-8 (Client Satisfaction)`,
   `WAI-SR (Working Alliance)`, `MI-SAT (MI Satisfaction)`, `PCT (Patient Change-Talk)`,
   `MICI (MI-Inconsistency)`; `Q1+Q2` unchanged (Lior); R:Q/%CR/%MICO keep their descriptive (keys already are
   acronyms). New `short_label()` (acronym-only, ↓-flagged) for DENSE figures where the gloss overflows.
5. ✅ **DONE (2026-07-03) — grouping + labels + paragraph.** The two families are now explicit: the
   `rubric_correlation` heatmap uses `short_label` ticks + a heavy divider at the warmth/orthogonal boundary
   + blue/orange block labels; `factor_loadings_bars` keeps the blue(warmth)/orange(orthogonal) coding (now via
   the `WARMTH_RUBRICS` constant); `3_Mechanism` §3 markdown rewritten as a two-family explainer (Warmth = one
   PC1≈91% factor; Orthogonal = PCT/MICI↓/R:Q/%CR/%MICO define PC2). Surfaced finding: **PCT empirically loads
   WITH warmth** (ρ≈0.79–0.94; high PC1 loading) despite being nominally orthogonal — now visible in both figures.
6. ✅ **DONE (2026-07-03) — audited, NO correctness bugs.** Holm + BH-FDR verified identical to `statsmodels`;
   dz / Cliff's δ / Friedman+Kendall-W / epsilon² all standard; tables reproduce the known headline (PTO 4.26 vs
   GRPO 3.75 @ iter10; PTO−GRPO Q1+Q2 +0.51 dz0.73; MICI −0.35 favouring PTO); merge alignment sound. Fixes were
   REPORTING clarity: documented the Holm **family scope** (the cross-arm `method_paired_by_K`/`k_paired_by_method`
   `p_holm` is corrected across rubrics WITHIN each matched (K/method, iteration) — NOT pooled across iterations) in
   the captions + §4 markdown + `stats._paired_arm_comparison` docstring; noted `trajectory_test` p-values are
   descriptive (non-independent rows → use Friedman for RM inference) in the docstring + §5 markdown.
7. ✅ **DONE (2026-07-07)** — see the Landed note above. General review surfaced + fixed: the MI-SAT
   "diabetes" domain bug (re-scored), an honest PTO range-vs-range advantage signal (+ a grouping-bug catch),
   MITI rate-normalization, and a batch of write-up-framing edits (confirmatory/exploratory split, reward=
   outcome + shared-oracle confounds, PCT-loads-with-warmth, K-descriptive banners, LIMITATIONS.md).
All backlog items (#1–#7) are now done.
Open cosmetic: tables-only `6_Stats` still writes an empty `figures/6_stats/_provenance.md` (harmless;
INDEX ignores it) — optionally suppress provenance for tables-only notebooks.


---

**EDA rebuilt research-grade + first cross-method results (2026-06-09).** Exp3's analysis EDA was
rebuilt as the `eda/eda_analysis/` package + notebooks (reorganized the next day — see the 2026-06-10
passes below), with true-persona recovery, both stat batteries + repeated-measures (Friedman), and a
thesis-export layer (`results/` figures + tables). Old Exp2 EDA frozen in `eda/archive_exp2/` (then
removed 2026-06-15 with the `pto_Exp2` data). **First-results snapshot (0–3 GRPO iters; superseded by
2026-06-14 → 2026-07-08):** PTO LA0 3.00→4.26; GRPO LA0 reached 3.99 in 3 iters and *looked* to climb
2.4× faster (slope 0.29 vs 0.12) — with GRPO extended to iter 8 that fast-slope read normalized to a
near-tie, and by iter 10 GRPO regressed outright (see the 2026-07-08 entry at the top).

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
reorganized exports + notebooks. **(1) `EdaConfig`** (new [eda_analysis/config.py](../eda/eda_analysis/config.py))
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

**Orthogonal eval axes (2026-06-14, same day as the two passes above).** The 6 rubrics correlated at
PC1≈91% (a subjective warmth halo), so two **orthogonal questionnaires** were added to
[questionnaires.py](../code/questionnaires.py): **PCT** (patient change-talk vs sustain-talk +
readiness, ID 8) and **MICI** (MI-inconsistent therapist behaviors incl. over-praise/sycophancy, ID 9,
lower=better), plus the *free* derived MITI-proficiency ratios **R:Q / %CR / %MICO** promoted to
first-class outcomes. Scored for all arms via `Run_Eval`. **Result: PC1 drops 91%→≈56%** — warmth is
one factor; technique + MI-inconsistency form a second. The `text_metrics` semantic regexes were
demoted to a lexical sanity-check (affirmation now = oracle MITI_B6_AF / MICI over-praise).

**VIEW system + package consolidation + narrative summaries (2026-06-18, latest).** Lior's asks: cleaner
EDA, results split by look-ahead, fewer/easier-to-edit modules, and a written summary. **(1) The VIEW knob.**
Cell 1 of every notebook now leads with `VIEW = os.environ.get("EDA_VIEW", "L0")` → `EdaConfig(view=VIEW, …)`.
`view ∈ {all, L0, L5}` is ONE control that sets BOTH the arm filter (`all`=every arm, `L0`=K=0, `L5`=K=5) AND
the results root, so `results/` now holds **3 parallel trees** `all/ · L0/ · L5/`, each
`figures|tables/<group>/` + `INDEX.md` + a hand-authored `SUMMARY.md`. Wired via `EdaConfig.view` + `_VIEW_KS`
(explicit `ks=` still overrides) in [config.py](../eda/eda_analysis/config.py) and a view-aware root
(`set_view`/`_results_root`/…) in [exports.py](../eda/eda_analysis/exports.py); `reset_results` clears only the
active view's figures/tables and **never deletes `SUMMARY.md`** (`PRESERVE`). **(2) Plumbing merged 14→9.**
`config.py`+`notebook.py`→**config**; `discovery`+`personas`+`scores`+`select`→**data.py**;
`figures`+`plots`→**plotting.py**. Kept: `stats`/`behavior`/`training`/`pref`/`exports`. The old submodule
names are **aliased** in `__init__` (`figures=plots=plotting`, `personas=scores=discovery=select=data`, also
registered in `sys.modules` so `from eda_analysis.personas import …` resolves), so **no notebook analysis cell
changed** — only cell 1 got the VIEW knob. **(3) Driver** [render_views.py](../eda/tools/render_views.py) regenerates all
3 views × 6 notebooks via nbconvert (sets `EDA_VIEW`, `--output-dir tmp` so source notebooks aren't churned).
**(4) Narrative** `results/<view>/SUMMARY.md` (hand-authored, preserved) — L0 is the primary read.
**Validated:** import/alias + view→ks + `target="best"` smoke PASS; 0_Headline@L0 dry-run wrote
`results/L0/figures/headline/*` + `INDEX.md` with `SUMMARY.md` intact; full 3×6 matrix via nbconvert.
