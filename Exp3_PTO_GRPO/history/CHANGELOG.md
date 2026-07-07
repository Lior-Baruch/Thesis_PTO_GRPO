# Exp3_PTO_GRPO — EDA CHANGELOG

Dated EDA-pass history, moved out of [../CLAUDE.md](../CLAUDE.md) on 2026-07-07 to keep the active reference scannable. These passes are fully superseded by the current "New EDA workflow" + "Eval results so far" sections in CLAUDE.md; kept here for provenance. Newest first.

---

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

