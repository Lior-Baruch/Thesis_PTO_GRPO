# Exp3_PTO_GRPO — CHANGELOG

Dated history moved out of [../CLAUDE.md](../CLAUDE.md) to keep the active reference scannable — superseded by the current-state sections in CLAUDE.md, kept here for provenance. **Two parts: the EDA passes (immediately below) and the trainer / infrastructure history (at the bottom).** Newest first within each.

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
(personas are smoking/obesity) → reworded goal-agnostic in [questionnaires.py](code/questionnaires.py) and
**re-scored all 2,784 convs** (0 errors); means rose **uniformly ~+0.14** (old diabetes wording rated an
intervention that never happened) — every relative conclusion preserved (PTO_LA0 still leads). **(B) Honest
advantage signal.** Added an UNFILTERED PTO `group_range` (best−worst over a branch's M candidates) beside
GRPO's as the true like-for-like analog to the τ-filtered `margin`. **Caught a grouping bug mid-work**
(PTO `branch_id` is the trunk *depth* and collides across conversations — verified in
[pto_trainer.py](code/PTO_Exp3/pto_trainer.py); the naive key pooled cross-conversation spread) → fixed by
keying on `conversation_id` too. Corrected finding: per-branch spread is modest+comparable (~0.23 PTO vs
~0.29 GRPO); the τ-filter mildly *inflated* PTO's apparent decisiveness (so the old "comparable ~0.3" read
was margin-vs-range, not like-for-like). K=5>K=0 holds on both. **(C) Rate-normalization.** MITI behavior
counts now shown per therapist turn (`B*_per_turn` in `behavior.py`; drift figure switched) so length
doesn't inflate them. **(D) Framing (notebook markdown, no data change):** confirmatory-vs-exploratory split
(`6_Stats` §0; confirmatory = PTO>GRPO on Q1+Q2 at **final AND best** iter + vs-base + reward-hack orthogonal
contrasts); reward=outcome + shared-oracle confounds named (`3_Mechanism` §4, anchored on reward-independent
deterministic text metrics); **PCT loads WITH warmth** (ρ≈0.79–0.94, NOT orthogonal — fixed contradicting
loadings captions, `3_Mechanism` §3); K0-vs-K5 descriptive-only banners (`4_Training` §4, `5_Preference` §2);
PCA-mechanical + bootstrap-seed caveats (`6_Stats` §5); new [eda/LIMITATIONS.md](eda/LIMITATIONS.md).
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
reorganized exports + notebooks. **(1) `EdaConfig`** (new [eda_analysis/config.py](code/../eda/eda_analysis/config.py))
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
(explicit `ks=` still overrides) in [config.py](eda/eda_analysis/config.py) and a view-aware root
(`set_view`/`_results_root`/…) in [exports.py](eda/eda_analysis/exports.py); `reset_results` clears only the
active view's figures/tables and **never deletes `SUMMARY.md`** (`PRESERVE`). **(2) Plumbing merged 14→9.**
`config.py`+`notebook.py`→**config**; `discovery`+`personas`+`scores`+`select`→**data.py**;
`figures`+`plots`→**plotting.py**. Kept: `stats`/`behavior`/`training`/`pref`/`exports`. The old submodule
names are **aliased** in `__init__` (`figures=plots=plotting`, `personas=scores=discovery=select=data`, also
registered in `sys.modules` so `from eda_analysis.personas import …` resolves), so **no notebook analysis cell
changed** — only cell 1 got the VIEW knob. **(3) Driver** [render_views.py](eda/render_views.py) regenerates all
3 views × 6 notebooks via nbconvert (sets `EDA_VIEW`, `--output-dir tmp` so source notebooks aren't churned).
**(4) Narrative** `results/<view>/SUMMARY.md` (hand-authored, preserved) — L0 is the primary read.
**Validated:** import/alias + view→ks + `target="best"` smoke PASS; 0_Headline@L0 dry-run wrote
`results/L0/figures/headline/*` + `INDEX.md` with `SUMMARY.md` intact; full 3×6 matrix via nbconvert.


---

# Trainer / infrastructure history

> Moved from [../CLAUDE.md](../CLAUDE.md) on 2026-07-08 when it was pruned to a lean current-state doc. These are the dated "landed"/"fixed" narratives for the trainer + `_shared/` infrastructure; the CURRENT behavior they established is summarized in CLAUDE.md's "Training internals" section. Ordered as they appeared in the old CLAUDE.md.

## Step-2 (pref-build) resume — automatic (landed 2026-06-07)

Step 2 ("Building pref pairs") is the dominant PTO phase (~41 min at K=0, hours at K=5) and
now **resumes automatically**, mirroring Step 1's per-CSV conversation resume — because
`resolve_start_state` only treats an iteration as done once `iteration_N/adapter/` exists, so
a crash *after* Step 2 but *before* the adapter (e.g. the DPO OOM) used to re-run the whole
build. Two levels, both in [pto_trainer.py](code/PTO_Exp3/pto_trainer.py):
- **Level A — reload a completed build.** If `iteration_N/pref_pairs/pairs.csv` exists, it's
  reloaded (`_reload_pairs_csv`) and Step 2 is skipped entirely. `pairs.csv` is now both the
  audit trail AND the completion marker (written atomically). On this path the EDA recorder is
  **not** re-flushed (the existing `generations.jsonl` is preserved).
- **Level B — resume a partial build.** The greedy/independent builders own
  `iteration_N/pref_pairs/_progress.json`, an atomic per-step snapshot (greedy: after each
  depth — the lock-step boundary; independent: after each conversation) holding trunk
  `turns`/`next_speaker`/`is_active` + carried pairs + EDA records. On restart they restore
  state and continue; on success `run_one_iteration` deletes `_progress.json`.
- **Guards (`_load_pref_progress`):** a snapshot is only resumed if `mode` + `iteration` +
  config fingerprint `{MCL, M, τ, num_utterances, greedy_trunk_target_len, seed}` + the
  conversation-id set all match the current run — so a checkpoint from a different **τ** (which
  is NOT in `EXPERIMENT_NAME`) is discarded, not silently mixed. Corrupt/missing ⇒ rebuild.
- **Correctness:** resumed trees start with empty `.pairs` (old pairs live only in
  `carried_pairs`) ⇒ no double-count; resume is statistically (not bitwise) equal — post-resume
  completions are freshly sampled, already-emitted pairs are reused verbatim. Validated:
  `py_compile` + an AST-extracted helper unit test (round-trip, empty, numpy-safe, all 4 guard
  mismatches, corrupt/missing). End-to-end greedy/independent resume awaits a real GPU+oracle run.

## Sub-epoch checkpointing + resume (landed 2026-06-08)

Both trainers used to checkpoint **once per epoch** (`SAVE_STRATEGY="epoch"`, `SAVE_TOTAL_LIMIT=1`).
A GRPO epoch is ~50 optimizer steps × ~1.5–2 min/step (G=8 sampling + K=5 look-ahead + oracle), so a
mid-epoch Colab crash threw away ~an epoch. Now both notebooks checkpoint **every `SAVE_STEPS=10`
optimizer steps**.

- **Knobs (cell 1, both notebooks).** `SAVE_STRATEGY="steps"`, new `SAVE_STEPS=10`, `SAVE_TOTAL_LIMIT=2`
  (+ a `SAVE_STEPS>0` validation). A new **required** `save_steps` field on `TrainingConfig`/`PTOConfig`
  threads through `_build_grpo_args`/`_build_dpo_args` into `GRPOConfig`/`DPOConfig` (`save_steps=` is
  honored only when `save_strategy="steps"`). No HF constraint tripped: `save_strategy="steps"` +
  `eval_strategy="epoch"` is legal because neither builder sets `load_best_model_at_end` (the
  "strategies must match" rule only fires when that's True).
- **Why step checkpoints "just work" for resume.** TRL/HF names every checkpoint
  `checkpoint-{global_step}` regardless of strategy, and the existing Case-B path
  ([model.py](code/_shared/model.py) `resolve_start_state` → `trainer.train(resume_from_checkpoint=…)`)
  reads only the dir-name step + the three required files (`adapter_model.safetensors`,
  `adapter_config.json`, `trainer_state.json`) — all present in a step checkpoint. Step accounting is
  unchanged (`step_delta = global_step − resumed_steps`; the in-progress checkpoint's steps are already
  in the startup offset → no double-count).
- **Hardened resume (walk-back).** Frequent saves raise the odds a crash lands mid-write. New
  `get_latest_valid_hf_checkpoint(training_dir)` ([model.py](code/_shared/model.py), exported) walks
  checkpoints newest→oldest and returns the first that passes `validate_hf_checkpoint`. Case B now
  resumes from the latest **valid** checkpoint (logs a fallback if the newest is corrupt) and only
  restarts the iteration from scratch if **none** is valid; `compute_cumulative_step_offset` uses the
  same walk-back for the in-progress iteration. `SAVE_TOTAL_LIMIT=2` guarantees a good fallback is on
  disk.
- **Existing/in-flight runs continue with NO migration.** Completed iters resume from
  `iteration_N/adapter/` (Case C, strategy-agnostic); a run crashed mid-iteration under the old epoch
  config resumes from its epoch `checkpoint-N` (a valid integer-named dir), then writes step
  checkpoints going forward (`list_hf_checkpoints` sorts old+new into one monotonic sequence; the old
  epoch ckpt isn't pruned until ≥2 newer ones exist — after we've already resumed from it). To keep a
  run on per-epoch saving, set `SAVE_STRATEGY="epoch"` for that session.
- **Quicktest-safe.** With tiny step counts `SAVE_STEPS` may exceed total steps → zero
  `checkpoint-N` written, which is harmless: the completed-iteration marker is the **separate**
  `iteration_N/adapter/` save (`save_iteration_checkpoint`), which `resolve_start_state` keys off.

### EDA completeness on resume (GRPO-only, same change)

The per-generation EDA buffer ([eda_recorder.py](code/_shared/eda_recorder.py)) is flushed once at
iteration end, and HF resume **fast-forwards skipped steps without re-invoking the reward fn** — so a
mid-iteration-resumed GRPO iter's `eda/generations.jsonl` used to drop the pre-crash candidates. Fix:
`CheckpointMetadataCallback` ([tb_plots.py](code/_shared/tb_plots.py)) now takes an optional
`recorder` and, on each `on_save`, also writes `checkpoint-N/eda_snapshot.jsonl` (new
`EDARecorder.snapshot_to`); on a one-shot mid-iteration resume `run_one_iteration` reloads that
snapshot (`EDARecorder.load_from`) **before** training so the end-of-iter flush keeps pre-crash +
post-resume rows. Bound to the **checkpoint dir** so it stays aligned under the walk-back. The
snapshot is extra payload inside `checkpoint-N/` (invisible to `validate_hf_checkpoint` /
`resume_from_checkpoint`); a missing snapshot is a guarded no-op, so pre-feature checkpoints behave
exactly as before. **PTO needs no change** — its recorder is used only in Step-2 (already resume-aware),
and its DPO `CheckpointMetadataCallback` is constructed without a recorder. Caveat: under GRPO inner-loop
`μ>1` (quicktest=2; production=1, exactly clean) one generation batch could double-record at the
boundary — dedupe on read by `branch_id` if it ever matters.

**Validation.** py_compile (all edited files) + GRPOConfig/DPOConfig construct with the steps config +
`get_latest_valid_hf_checkpoint` walk-back unit test (skips a corrupt newest, returns it once complete,
None on empty) + snapshot/reload round-trip + callback `on_save` writes/`recorder=None` skips +
`_local_smoke.py all` (stopgen/dpo/grpo) PASS. **End-to-end crash-resume (assert the resumed iter's
`generations.jsonl` keeps pre-crash rows) awaits a GPU+oracle quicktest.** Re-push `code/` + restart to
apply.

## Look-ahead performance (K>0) — batched rollout LANDED

**Status (2026-06-02).** The K>0 wall-clock bottleneck is fixed:
`simulate_lookahead_batch` in [_shared/reward.py](code/_shared/reward.py) is now a
**lock-step batched rollout**. All B completions advance in unison (patient →
therapist → …), so each therapist look-ahead turn is **one padded batched
`model.generate`** over the active sims instead of B serial batch-of-1 calls —
collapsing ~B·K serial generations into ~K batched ones. Semantics match the
legacy serial path (statistically equivalent, not bit-identical — sampling RNG
differs). Both GRPO (`make_reward_fn`) and PTO (`build_pref_pairs`,
[PTO_Exp3/pto_trainer.py](code/PTO_Exp3/pto_trainer.py)) get it through the shared fn.

**How it's safe.** The batched therapist step holds `gpu_lock` per-step (never
across the patient API `await`) with the `eval()` + `use_cache=True` toggle nested
inside, restored in a `finally` (look-ahead runs *during* a GRPO step with the
policy in `train()`). OOM is handled by `_therapist_generate_chunked`: a
chunk-and-halve loop over `generate_therapist_responses_batch` that halves the
sub-batch on OOM (kept **sticky**) and freezes a sim (scores its shorter
transcript) only if even sub-batch=1 OOMs — never aborts the GRPO step. A sim is
likewise frozen on SESSION ENDED, patient-API failure, or an unparseable
transcript (the serial path let parse errors propagate; batched is deliberately
more robust). Verified by a fakes-based logic test (happy path, per-sim freezing,
OOM halving 4→2,2, sub-batch=1 OOM, parse-failure isolation, toggle restoration
after a mid-rollout exception — all pass).

**Knob.** `LOOKAHEAD_SUB_BATCH_SIZE` (notebook cell 1 → `LookaheadConfig.lookahead_sub_batch_size`;
cell 1 now sets **64 (GRPO) / 128 (PTO)** on A100-80GB — see "Runtime tuning for Colab throughput";
`None` = all active sims in one call). Halved automatically on OOM (kept sticky for the rest of the rollout).

**Telemetry.** The existing `reward_fn` line now reports the batched cost:
`Look-ahead: N sims × K=… in X.Xs (… ended early; batched, G GPU calls, sub_batch=S)`.
The legacy `simulate_lookahead_single` / `_generate_therapist_single_async` are kept
(marked LEGACY) as the equivalence-check reference, not on the hot path.

**Validation harness.** [_shared/lookahead_check.py](code/_shared/lookahead_check.py)
(`make_quick_fixtures` + `compare_serial_vs_batched`) runs both paths on the same
fixtures and prints realized-turn + Q1+Q2 reward mean/std for each plus the batched
speedup. Wired as an **optional section 6 cell** in
[GRPO_Exp3/train_GRPO_Iterative.ipynb](code/GRPO_Exp3/train_GRPO_Iterative.ipynb)
(guarded by `LOOKAHEAD_K > 0`). Raise `LOOKAHEAD_SUB_BATCH_SIZE` past VRAM to exercise
OOM halving.

**Validation (updated 2026-06-03).** ✅ (a) `compare_serial_vs_batched` equivalence
**passed on real GPU** (Colab, 48 fixtures, K=3): serial Q1+Q2 mean 2.577 vs batched
2.553, **|Δmean| = 0.024** (< oracle noise ~0.07–0.10); identical realized turns 2.88;
1.5× speedup (2 GPU calls, sub_batch=32). 🔄 (b) GRPO_Exp3 **K=3 bf16 quicktest** on
Colab — got through conv generation + prompt extraction, was blocked at the GRPO
training block by the torchao/peft Colab crash (now fixed; re-running). ⬜ (c) Colab
**K=5** arm after the K=3 quicktest trains through. Sequence: ✅ batched fix →
✅ equivalence → 🔄 K=3 quicktest → K=5 arm.

## Per-generation EDA capture + live TensorBoard (landed 2026-06-05)

**EDA capture.** Each iteration writes
`runs/<MODE_TAG>/<EXP_NAME>/iteration_N/eda/generations.jsonl` with **every** candidate the
policy generated (previously PTO kept only the final (chosen,rejected) pair; GRPO kept nothing
per-prompt). Owned by [_shared/eda_recorder.py](code/_shared/eda_recorder.py) (`EDARecorder`:
in-memory buffer, one atomic flush/iteration — Drive-FUSE-friendly). **Branch-centric schema —
one JSON row per branch:**
- `prefix` (oracle-format transcript of the conv-so-far, stored ONCE), `candidates:[…]` nested
  (each: `completion`, `score`, per-questionnaire `sub_scores`, `oracle{success,retries}`,
  `lookahead{k,realized_turns,ended_early,tail}`), `chosen_idx` (= argmax score).
- `lookahead.tail` = the K simulated turns only (prefix+completion sliced off — exact, since
  look-ahead concatenates). Reconstruct a candidate's oracle-scored text =
  `prefix + "\n\n[THERAPIST]: " + completion + (tail or "")`.
- **GRPO:** one branch row per group **per epoch** (rows carry `epoch` + `group_mean/group_std`);
  recorded in the reward fn ([reward.py](code/_shared/reward.py) `_record_grpo_generations`,
  reshapes TRL's G-consecutive completions). **PTO:** one row per branch with candidate `role`
  (chosen/rejected/neither); recorded in `_record_pto_branch` (greedy + independent).
- Base full conversations are the already-saved `model_iter_*` eval convs (greedy's base = its
  eval conv) — no separate trunk artifact. EDA load: `read_json(lines=True)` →
  `df.explode("candidates")`.
- Knobs (cell 1): `SAVE_EDA_GENERATIONS`, `SAVE_LOOKAHEAD_TRANSCRIPTS` (drops the per-candidate
  `tail` — the size lever).

**Logging = HF defaults (reverted 2026-06-07).** Training logs go through HF's own
`WandbCallback`/`TensorBoardCallback`: **one W&B run per iteration** (grouped under the experiment
via `wandb_ctx["run_id"]`), charts on the default `train/global_step` axis, TRL's native metrics +
completions table (`LOG_COMPLETIONS=True`). The earlier custom `cumulative_global_step` step-axis
override (in `init_iteration_logging`) + `CumulativeStepCallback` are **removed** — they fought HF's
own `define_metric("*", step_metric="train/global_step")` and broke the familiar charts.
**The custom continuous view is opt-in:** `TB_LIVE_LOGGING` defaults **False**; set it True to also
get [_shared/tb_plots.py](code/_shared/tb_plots.py) `RunTBLogger`'s one continuous `tb_live/`
SummaryWriter (smoothable cross-iteration curves + reward histograms + sample completions, mirrored
to W&B) plus the EDA aggregates (`eda/*`, `pto/*`, `grpo/*`). The post-hoc matplotlib dashboard
`plot_iteration_metrics` (method-aware: DPO rewards/margins/logps; GRPO reward_std/frac_zero_std/
length) reads the per-iteration `tb_logs/` event files and works regardless. Knobs:
`TB_LIVE_LOGGING`, `TB_SAMPLE_COMPLETIONS_N`, `LOG_COMPLETIONS`.

**Status:** EDA capture validated on the first full runs (`iteration_1/eda/generations.jsonl` written
for GRPO + PTO). Logging revert validated offline (py_compile + import + TRL-config construct);
confirm clean per-iteration W&B charts on the next quicktest.

## Runtime tuning for Colab throughput (2026-06-07)

First full K=5/MCL12/Q1Q2 arms on a Colab **A100-80GB** were far too slow: GRPO
**~7 h/iteration** (150 optimizer steps — `per_device_train_batch_size=64` counts
*completions*, so with `NUM_GENERATIONS=8` that's 16 prompts/step → 803/16×3 ≈ 150),
PTO **Step-2-dominated** (greedy trunks grow 12→49 utts ≈ 18 branching depths, each a
K=5 look-ahead over ~672 candidate sims). The wall is the **K=5 look-ahead** — mostly
*sequential OpenAI API latency* + oracle scoring, which GPU batch size doesn't touch —
not VRAM (GPU sat at ~17 GB in PTO Step 2, ~67 GB in the GRPO step).

- **Throughput knobs (both notebooks cell 1; statistically equivalent, no science
  change):** `CONVERSATION_BATCH_SIZE 16→64`, `ORACLE_MAX_CONCURRENCY 64→128`,
  `PATIENT_API_CONCURRENCY 48→96`, `LOOKAHEAD_SUB_BATCH_SIZE 32→64` (GRPO; step already
  ~67 GB — auto-halves on OOM) / `32→128` (PTO; Step 2 has headroom).
- **DPO batch: kept at the proven `2×8` + grad-ckpt ON (PTO only).** I briefly tried `16×1` +
  grad-ckpt off here for A100 speed — it **OOM'd at the iter-1 DPO step (78.5/80 GB)**. DPO
  materializes logits over the full prompt+completion × 128k vocab with no `logits_to_keep`, and
  **`per_device_train_batch_size` (not the effective batch) sizes that tensor**, so 2→16 made it
  ~8× and grad-ckpt-off also retained all activations. **Reverted to `per_device=2 × grad_accum=8`
  (effective 16) + `DPO_GRADIENT_CHECKPOINTING=True`** — the config from "First full-run failures".
  Negligible cost: DPO is ~2–3 min vs Step 2's ~41 min, so per-device DPO batch is NOT a useful
  speed lever. (If DPO speed ever matters: the liger DPO loss avoids materializing full logits —
  needs `liger-kernel` installed.)
- **`EPOCHS_PER_ITERATION 3→2` (both arms, matched).** ~⅓ off GRPO training (150→~100
  steps/iter); little effect on PTO (DPO is cheap; Step 2 dominates). `NUM_ITERATIONS`
  kept at 10; K=5 kept (the science). Changes absolute scores, not the comparison
  (applied equally to both methods).
- **New PTO lever — `GREEDY_TRUNK_TARGET_LEN`** ([pto_trainer.py](code/PTO_Exp3/pto_trainer.py)
  `PTOConfig.greedy_trunk_target_len`, wired from cell 1): caps greedy trunk growth via
  `target_len = min(NUM_UTTERANCES_FOR_DATA, GREEDY_TRUNK_TARGET_LEN)`. **Defaults to
  `NUM_UTTERANCES_FOR_DATA` = no-op.** Lower it (e.g. 30 ≈ the partial-oracle EDA's 0.9
  rank-agreement point) to grow shorter trunks → far fewer branching depths → the biggest
  remaining PTO Step-2 speedup. It's a **science change** (shallower trunks/look-ahead
  context) and is **NOT in `EXPERIMENT_NAME`**, so isolate a lowered run by clearing/renaming
  its output dir.
- **GRPO warmup-calc fix** ([_build_grpo_args](code/GRPO_Exp3/grpo_trainer.py)): now divides
  by the real prompts/step `(train_batch_size/num_generations)*grad_accum`, so the printed
  `total_train_steps` matches the real ~100 (was 21 at 3 epochs). Only the warmup print/value
  was wrong; the cosine LR horizon was always correct (HF Trainer recomputes it from the
  dataloader length).

**To apply:** re-push `code/` to Drive and **restart** the runs (cell 1 is read only at
startup); saved `model_iter_0` conv CSVs are reused via resume, so Step-1 gen isn't repeated.
Expect GRPO ~3 h/iter, PTO ~1.5–2× faster on Step 2.

**Launched 2026-06-07 (tuned config).** Three arms running on Colab: **GRPO LA0, GRPO LA5,
PTO LA0** (PTO LA5 pending). The earlier mid-flight 3-epoch run dirs were archived (renamed
with an `(Archive_V2)` suffix) rather than deleted, so the tuned arms write fresh folders.
**PTO LA0 then OOM'd at the iter-1 DPO step** (the 16×1 + grad-ckpt-off mistake above); DPO config
reverted to `2×8` + grad-ckpt on, re-push + restart the PTO arm. PTO Step 2 took **2454 s / 782
pairs / 37 depths** before the crash (K=0 → no look-ahead; that time is branch-sampling generation
+ oracle scoring only — not yet decomposed into GPU vs API).

## First full-run failures + fixes (2026-06-06/07)

The first full Colab runs (LA5/MCL12/Q1Q2) were stopped — long + API-costly, nothing obvious in
W&B/TB. Diagnosis + fixes (validated: py_compile + import + TRL-config construct + a fake-tokenizer
unit test of the prompt cap):

- **PTO crashed at the first DPO step (OOM).** DPO's `_compute_loss` takes `outputs.logits` over the
  FULL prompt+completion (no `logits_to_keep`, unlike GRPO which restricts to the ~200 completion
  tokens — verified vs TRL 1.4.0 source). Greedy trunks are ~2.4k tokens (max ~6k), so the LM-head
  logits tensor = batch 16 × 2 (chosen+rejected) × ~2248 × 128k vocab × 2 B ≈ 17 GiB (×copies +
  backward → OOM). Latent second bug: `truncation_mode="keep_start"` slices `[:max_length]`, so for a
  prompt longer than `max_length` the *response* is dropped and `completion_mask` is all-zeros. **TB
  looked empty because only the `args`/`model_config` text summaries were written — zero training
  steps.** **Fix:** `build_truncated_training_prompt` ([convs.py](code/_shared/convs.py)) caps the DPO
  prompt to `max_allowed_prompt_length` (drop-oldest, keeps system+recent — identical to GRPO's
  `extract_prompts_from_conversations`, and matches the serve-time context window) at both pref
  builders; DPO `per_device_train_batch_size 16→2` × `gradient_accumulation_steps 1→8` (effective 16
  unchanged — the batch is what fixes the logits OOM; grad-ckpt does NOT touch the logits tensor);
  `gradient_checkpointing=True` (`DPO_GRADIENT_CHECKPOINTING`; TRL handles the PEFT/precompute
  interplay) so it fits any Colab GPU. NOT the local Blackwell crash — `precompute_ref_log_probs` was
  already on. **(2026-06-07: a 16×1 + grad-ckpt-off attempt on A100 for speed OOM'd at the iter-1
  DPO step — this `2×8` + grad-ckpt-on config is the one that stands. `per_device` batch sizes the
  full-seq logits tensor, so keep it at 2. See "Runtime tuning for Colab throughput".)**
- **GRPO didn't crash but ran ~11.5 h/iter and reward-hacks length.** `<|im_end|>` is template text,
  not the base tokenizer's eos, and `GRPOConfig` set no stop → TRL's in-loop sampling runs to the
  200-tok cap, self-playing the patient's reply (entropy 3.97→1.92, 96% clipped), which both pollutes
  the oracle transcript and trains the ramble. **Fix:**
  `GRPOConfig(generation_kwargs={"stop_strings": cfg.stop_strings})` — `patch_generate` already
  injects the tokenizer so `stop_strings` binds (the same path look-ahead relies on during the step) —
  plus a defensive `<|im_end|>` clean in `make_reward_fn`. (The ~11.5 h/iter cost itself — in-loop K=5
  look-ahead + 3 epochs + look-ahead eval — is config/throughput, not a bug; **addressed 2026-06-07 —
  see "Runtime tuning for Colab throughput".**)

See also "Logging = HF defaults" above (the W&B charts were broken by the custom step-axis override,
now reverted to one HF run per iteration).

## ChatML self-play leak (found + fixed 2026-06-07)

Found by **reading the quicktest output** (`pref_pairs/pairs.csv` + the `model_iter_*` convs), not
from a crash. Base **Llama-3.2-1B self-plays `<|im_start|>` tokens**: they are NOT special tokens
(tokenizer vocab stays 128256; the ChatML template renders them as ordinary BPE text the base model
has never been trained on), so early in training the therapist emits `<|im_start|>` and writes the
*other* speaker's turn as literal text. Two failure modes, one cause:
- **PTO spam** — therapist turns become pure `<|im_start|>assistant/<|im_start|>patient` piles (no
  content); the oracle still scored them ~4.5/5 (it was grading the coherent *patient* turns) →
  degenerate (chosen,rejected) DPO pairs.
- **GRPO / conv-gen role-swap** — one leaked first-person `<|im_start|>user\nI've been struggling…`
  line flips the gpt-4o-mini patient into **counselor** mode → roles invert for the rest of the conv
  (patient calls the therapist "Emma"; therapist discloses problems). Coherent-looking but mislabeled;
  ~2/4 seed convs derailed; also collapsed GRPO `group_std`→~0.012 (near-zero advantages).

**Fix (in code):**
- `STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]` (both notebooks cell 1 + `_DEFAULT_STOP_STRINGS` in
  [_shared/convs.py](code/_shared/convs.py)) — generation halts the moment a fake turn opens.
- New `_shared/convs.py::clean_completion` cuts at the FIRST marker; used at every decode site
  (`generate_therapist_responses_batch`, [reward.py](code/_shared/reward.py) look-ahead hot+legacy,
  GRPO `reward_fn`). Empty-after-clean **ends the conversation** (`_process_session_response`);
  look-ahead sims freeze on empty.
- GRPO floors degenerate completions to `REWARD_FLOOR = 0.0` (below the oracle 1–5 range) so a
  self-played turn gets a strong negative group-relative advantage; EDA candidate `score` now records
  the floored/training reward (matches `group_mean/std`). PTO needed no extra logic (its builders
  already drop empty candidates).

**Validated locally (quicktest, 2026-06-07):** PTO spam-conv dropped (real pairs, 0 degenerate rows,
roles correct, both iters complete); GRPO 0 `<|im_start|>` leak across 56 candidates, model_iter_1
convs role-correct, `group_std` 0.013–2.04 (mean 0.28), floor reached training (1 completion → 0.0).
GRPO iter-2 then hit the local Blackwell save-time crash (hardware — training completed, save path
untouched; see Gotchas / the local-crash memory). Full K∈{0,5} sweep runs on Colab regardless.

## Sweep priority (updated 2026-06-11)

**Run status + cost (2026-06-11).** PTO LA0 = 10 iters done; **GRPO LA0 running (iter 6)** (the
fair-endpoint comparison vs PTO is in progress); **both LA5 arms PAUSED for cost** — OpenAI spend
across the Exp3 runs + quicktests hit **~$300** and is now a binding constraint, so RQ-i (K0 vs K5) is
on hold. The bill is dominated by oracle scoring + K=5 look-ahead patient calls (both ∝ candidate
count × iterations); **caching is already maxed** (~50% off the oracle's rubric-first prefix — don't
trim it), so reduce **call COUNT**: cap `NUM_ITERATIONS` ~5–6 (gains plateau by iter ~4 → ~40–50%
saving, still a matched-iter comparison), `M`/`G` 8→4, PTO `GREEDY_TRUNK_TARGET_LEN`↓; keep **K** (the
science) + the **gpt-4o-mini oracle** (comparability with already-scored data) fixed. Patient-model
swap is possible but a science change — avoid. Estimate cost/arm from cell-1 config before launching +
set an OpenAI hard usage limit. See the `project-openai-cost-constraint` memory.

0. **Quicktest (both methods) — ✅ DONE 2026-06-07, validated LOCALLY end-to-end** (not Colab; the
   full notebooks ran via nbconvert, `RUN_MODE="quicktest"`, `WANDB_MODE=offline`, venv kernel
   `thesis-venv313`). PTO OOM fix confirmed (reached `iteration_2/adapter/` + `model_iter_2`, no
   step-1 OOM, no PC reboot); GRPO stop-string fix confirmed (`completions/mean_length`=48.4 < 64
   cap). `_local_smoke.py all` also 3× PASS. Offline W&B runs in each notebook's `wandb/offline-run-*`
   (online project is empty until `wandb sync`; Colab full runs report live). See "First full-run
   failures + fixes" below and the root CLAUDE.md "Next step".

   **To run a notebook headless locally again:** register the venv as a kernel once
   (`.venv\Scripts\python.exe -m ipykernel install --user --name thesis-venv313`), then
   `WANDB_MODE=offline ... -m jupyter nbconvert --to notebook --execute
   --ExecutePreprocessor.kernel_name=thesis-venv313 <nb>` (offline avoids the W&B login hang; the
   default `python3` kernel is the system interpreter and lacks torch/trl).
1. **GRPO_Exp3 + PTO_Exp3 @ K ∈ {0, 5}, MCL = 12 (Colab) — the immediate next action.** 4 arms; set
   `LOOKAHEAD_K` per arm in cell 1 (`EXPERIMENT_NAME` auto-encodes `LA{K}` → disjoint folders); push
   `code/` to Drive first; keys from Colab Secrets. K=3 look-ahead equivalence already ✅ validated.
   **Throughput/epoch tuning applied 2026-06-07 (EPOCHS 3→2, batch + concurrency bumps) — see
   "Runtime tuning for Colab throughput".**
2. Maybe → either method @ MCL = 2.
3. Maybe → other training oracles (WAI-SR / CSQ-8 / MI-SAT / MITI).

## PTO parity + greedy mode + oracle-in-name batch (through 2026-06-04)

Alongside the batched look-ahead rollout above, the same batch landed:
- **PTO_Exp3 brought to parity with GRPO_Exp3** — controlled hyperparameters matched, M=8, bf16
  toggle, zero-pairs/split robustness. Trainer modules renamed `trainer.py` →
  `grpo_trainer.py`/`pto_trainer.py` (a `from trainer import` collision when both notebooks share one
  local kernel — sys.modules cached the first-loaded trainer).
- **Greedy true-PTO mode committed** (`e27b9de`): `PREF_TREE_MODE=greedy` grows ONE trunk via
  best-of-M feedback (`grow_preference_trees_batch`); the old slice-branch path kept as
  `independent`; `_PT{greedy|indep}` baked into `EXPERIMENT_NAME`. Greedy then made to slice its
  MCL-prefix off the step-1 conv — no separate prefix-gen pass (`420299b`).
- **Training oracle encoded in `EXPERIMENT_NAME`** (`7cbb475`): a `{Q1Q2|WAI|CSQ8|MI_SAT|MITI}` token
  derived from `QUESTIONNAIRE_IDS`, identical to the EDA `oracle=<O>` tokens → ready for the oracle
  sweep.
- **Iteration-2 local-crash fix:** `precompute_ref_log_probs=True` on the PTO DPOConfig
  (`DPO_PRECOMPUTE_REF_LOGPS` knob) moves the TRL `"ref"`-adapter forward out of the training backward
  step — the isolated iter-2 DPO smoke test PASSED on the local Blackwell for the first time
  (`_iter2_dpo_smoke.py`). GRPO quicktest block trimmed for the local 12 GB GPU.

## Dependency stack audit (2026-06-01; update 2026-06-03)

*(Moved here from CLAUDE.md 2026-07-12.)* Trainers were audited against the latest docs of the pinned
stack (`transformers==5.8.1`, `trl==1.4.0`, `peft==0.19.1`, `huggingface_hub==1.14.0`,
`wandb==0.26.1`) and **verified current** — nothing deprecated (the then-lingering "TRL v0.28"
comments in the code were cleaned up later, 2026-07-11):
- **`scale_rewards="group"`** (grpo_trainer.py) is the TRL **default** (`"group"/"batch"/"none"`), not a stale value.
- **async reward fn** (_shared/reward.py) is natively awaited by TRL 1.x (`inspect.iscoroutinefunction` → `asyncio.gather`); extra dataset columns forwarded as kwargs; per-sample `None` supported.
- `processing_class=`, `eval_strategy=` already on the new transformers-5/TRL-1 API.
- `hf_xet` is a **required transitive dep** of `huggingface_hub` 1.x — already installed, nothing to add.
- `gpt-4o-mini-2024-07-18` (patient + oracle) has **no API retirement date** per OpenAI dev docs (the only relevant shutdown is `gpt-4o-2024-05-13`, a different model).

Same-session polish: both notebooks' Colab install cell **pinned to requirements.txt** (commented;
`weave` dropped), `authenticate()` sets `WANDB_LOG_MODEL="checkpoint"` (versioned adapter artifact,
third backup), and both configs set `run_name=current_adapter_repo`.

**Update 2026-06-03.** Install cell now also (commented) `%pip uninstall -y torchao` — Colab pre-bakes
torchao<0.16.0, which peft 0.19.1 rejects by *raising* inside `get_peft_model`'s `dispatch_torchao`
(crashed both trainers at iter 1). A100 optimizer batch raised to **16 decision-points/step** (GRPO
`TRAIN_BATCH_SIZE`=128, PTO DPO 16×1 — the DPO half later reverted to 2×8, see "Runtime tuning"; LR
held). `NUM_ITERATIONS` 8→10 both.

