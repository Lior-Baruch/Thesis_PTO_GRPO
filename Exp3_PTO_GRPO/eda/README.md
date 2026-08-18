# Exp3 EDA — guide

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), four
arms (PTO/GRPO × K=0/K=5) across training iterations under matched MCL, scored on two graders.
Everything lives in ONE package, `eda_analysis/` (since the 2026-07-13 fold of the legacy
`oracle_scoring/`): the analysis layer at the top level + the oracle-scoring layer in the `scoring/`
subpackage. The recurring figures are named functions in the `eda_analysis/plotting/` subpackage
(called once from multiple notebooks), and genuinely one-off exploration stays inline (the **hybrid**
plotting split). Thesis figures/tables are exported per **FAMILY** into
`results/<top>/<sub>/{figures,tables}/[<judge>/]` — figures `.png`, tables `.md` + `.xlsx`, number
ledgers `.json`.

**Organization = by research question, notebooks ↔ result families 1:1 (2026-08-18 reorg).**
`results/` used to be organised by *arm subset* (a VIEW knob: `L0/` = K=0 arms, `L5/` = K=5 arms).
The thesis' questions are *contrasts* — look-ahead, method, cost, measurement — and none fits a
K-filtered view (the cross-K artifacts ended up gated to one view, the paper had to build its own
four-arm generators outside `eda/`, and every reader hand-joined `L0` and `L5`). Now every notebook
owns exactly one **family** `"<top>/<sub>"`, and any artifact under `results/` traces straight back
to `notebooks/<top>/<sub>.ipynb`. Per-arm descriptives show **all four arms on one axis**.
Endpoint artifacts still come as a **final + best pair** (best = each arm's peak iteration on its
own training oracle via `best_per_experiment`): figures `<name>_final.png` + `<name>_best.png`,
tables merged with a `target` column.

## The results tree

```
results/
├── INDEX.md                                  auto: one line per family, with artifact counts
├── METRICS_REFERENCE.md  LIMITATIONS.md      hand-authored reference docs (moved here from eda/docs/)
├── schematics/                               hand-authored METHOD diagrams — build_method_figures.py,
│                                             CAPTIONS.md, 4 PNGs. No notebook, no judge level, no
│                                             figures/tables split (moved from Exp3_PTO_GRPO/figures/)
├── arms/                                     per-arm descriptives, ALL FOUR ARMS on one axis — PER JUDGE
│   ├── SUMMARY.md  INDEX.md                  hand-authored narrative · auto artifact map (all subs, all judges)
│   └── <sub>/{figures,tables}/<judge>/[<group>/]…    sub ∈ outcomes questionnaires validity
│                                                      heterogeneity training preference stats
├── lookahead/                                RQ-i: K=0 vs K=5 within optimizer, BOTH graders inside
│   ├── SUMMARY.md  INDEX.md
│   └── <sub>/{figures,tables}/[<group>/]…    sub ∈ reward transfer behaviour mechanism replication
├── method/contrast/{figures,tables}/…        RQ-ii: PTO vs GRPO at each K          (+ SUMMARY.md, INDEX.md)
├── compute/cost/{figures,tables}/…           GPU-h + API axis, budget sweeps       (+ SUMMARY.md, INDEX.md)
└── measurement/validity/{figures,tables}/…   judge validity, multi-judge           (+ SUMMARY.md, INDEX.md)
```

- **`arms/*` is per-judge** — its artifacts are *produced by* one grader, so the grader is named in
  the path (`<judge>/` leaf, short label from `constants.judge_dirname`) and the family is rendered
  once per grader on disk. **Everything else is judge-invariant** — those notebooks load both graders
  explicitly (`scores_by_judge`) and put them side by side in one table/figure (columns
  `oracle`/`held-out`, or two panels), so there is no `<judge>/` segment; a path naming one grader
  would be false.
- The `figures/` vs `tables/` split sits BELOW the family. Nested `group=`s work inside a family:
  `save_fig(fig, name, group="trajectories")` → `arms/outcomes/figures/gpt-4o-mini/trajectories/`.
- Every leaf carries `CAPTIONS.md` (auto-accumulated captions) and `_provenance.md` (which config
  produced it); a family may carry `numbers.json`-style ledgers (`tables/<name>.json`, via
  `save_numbers`) so papers can cite `results/<family>/tables/<name>.json :: <key>`.
- Each `<top>/` has a hand-authored **`SUMMARY.md`** (the interpretation — see the epistemic-status
  box in each) beside the auto **`INDEX.md`** (the evidence map).

## Notebook ↔ family

Every section is tagged **`[EVAL]`** (full-conversation oracle scores — the held-out outcome) or
**`[TRAINING]`** (partial-branch rewards / preference pairs — what the policy is updated on). Every
notebook ends with `build_index()` so `results/<top>/INDEX.md` + `results/INDEX.md` are complete
whatever runs last.

| Notebook | Answers | Main artifacts (`results/<family>/…`) |
|---|---|---|
| `arms/outcomes.ipynb` | **Level 1 — global scores**, all four arms | `trajectories_all_metrics` (THE main figure) · per-metric learning-curve catalog `trajectories/trajectory_<metric>` (peaks auto-flagged) · `outcomes_by_model_{final,best}` · `effect_vs_base_forest_{final,best}` · `leaderboard_scorecard` (+ `.json` ledger) · the presentation re-saves under `group="headline"` |
| `arms/questionnaires.ipynb` | **Level 2 — inside each rubric** | `<slug>_detail_grid` + `<slug>_item_deltas_{final,best}` for Q1/Q2/WAI-SR/CSQ-8/MI-SAT · `q2_item_group_trajectories` · `wai_subscales` · `miti_detail_grid` + `miti_detail_by_iter` + **official MITI 4.2.1 thresholds** (`miti_proficiency_thresholds`, `miti_threshold_verdicts`) · zoom groups `miti/` `pct/` `mici/` |
| `arms/validity.ipynb` | **Level 3 — is it real skill?** | `rubric_correlation` + `factor_loadings` + `rubric_pca_expanded` · `reward_hack_panel` · `question_rate_crosscheck` · `overpraise_crosscheck` · `session_shape` / `session_shape_by_iter` · `session_end_reasons` · `grpo_iter9_check` |
| `arms/heterogeneity.ipynb` | every metric split by persona trait | groups `cooperation_level/`, `problem/` (one figure per metric) · `<trait>_all_metrics` · `subgroup_endpoint_<trait>_{final,best}` + `subgroup_endpoint_means_<trait>` |
| `arms/training.ipynb` | **`[TRAINING]` — did the optimiser get a usable, faithful signal?** per arm | `tb_curves/<arm>` · `reward_distribution/<arm>` + `reward_distribution_by_iter` · `advantage_signal_sidebyside` / `_by_iter` · `degeneration_scan` · reward-faithfulness: `reward_reliability_curve` + `reward_reliability_by_nturns` · `faithfulness_proxy_vs_eval` · `pto_margin_by_depth` · `training_numbers.json`. Training-side sections save under the primary leaf only (see "Judge dimension") |
| `arms/preference.ipynb` | **what the training signal pushes toward**, both methods on one probe | PTO Mass-Mean-Probe over `pairs.csv` (word ranking/drift, direction drift, learn/unlearn, `<arm>_pref_MI_concepts`, `<arm>_pref_probe_quality`) · `update_lexical_push` · `update_direction_quality{,_pooled}` · `update_direction_cosines` · `pref_outcome_link/_correlations` · `weighting_decomposition` · `generation_vs_selection` + `generation_pool_means` · `training_signal_yield` · `<arm>_examples` (the K-mechanism panel moved to `lookahead/mechanism`) |
| `arms/stats.ipynb` | the heavy per-arm tables | `main_results` (`target` col) · `friedman_omnibus` · `vs_base_paired` · `slope_by_arm` · `rubric_pca_pc1` |
| `lookahead/reward.ipynb` | **RQ-i on the reward** — K=0 vs K=5 within each optimizer, persona-paired, both graders | `k_table1{,_Q1,_Q2,_MICI,_PCT}` · `k_paired_{pto,grpo}_<judge>` + `k_paired_long` + `k_paired_by_method` · `k_means_by_iter` · `k_levels{,_long}` · `k_summary` · `k_did` + `k_method_gap` + `k_endpoints` (difference-in-differences, method gap by K, endpoint contrasts) · figs `k_headline_q1q2`, `k_delta_grid_<judge>`, `k_contrast_both_judges`, `k_did`, `k_trajectory_Q1Q2` · `k_numbers.json` |
| `lookahead/transfer.ipynb` | does the K contrast **transfer** to the held-out judge? | `k_pairs` (primary vs held-out contrast pairs) · `k_sign_ladder` · `k_retention` + `k_retention_summary` (gain retention by K, several reference kinds) · fig `k_retention` · `transfer_numbers.json` |
| `lookahead/behaviour.ipynb` | what look-ahead **does to behaviour** — channels, substitution, session shape, held-out instruments | `k_paired_channels` · `k_means_channels` · `k_channels_{pto,grpo}_<judge>` + `k_channels_text_*` + `k_channels_summary` · `k_mici_composition` · `session_shape` · `length_endpoints` / `length_kcontrast` · `selection` · held-out instruments `wai_subscales` / `wai_kcontrast` / `wai_fig_data`, `pct_kcontrast`, `q2_items{,_long,_kcontrast}`, `hetero_kcontrast` · figs `k_channels_grid`, `k_channel_forest_<judge>`, `k_cost_benefit_<judge>`, `k_mici_composition_grid_<judge>`, `k_overpraise_trajectory_<judge>`, `session_shape`, `wai`, `hetero` · `k_channels_numbers.json`, `instruments_numbers.json`, `replication_numbers.json` |
| `lookahead/mechanism.ipynb` | **why** — the K-mechanism chain, dispersion, reward faithfulness at matched policy, tail audit | `k_mechanism_overpraise_chain` (+ fig `k_mechanism_overpraise`) · `dispersion_{by_iter,ratios,tau,expectation}` (+ figs `dispersion`, `dispersion_tau`) · `faithfulness_{curve,curve_heldout,curve_by_iter,k_by_iter,matched_policy,matched_policy_tests,k_summary,by_coop,levels,levels_rho}` (+ figs `faithfulness`, `faithfulness_heldout`) · the tail-audit tables (by iter, cues, within group, score by realized turns) + fig `tail_audit` |
| `lookahead/replication.ipynb` | does the ICLR result **replicate**? Exp1 transcripts re-scored + the SD/stability claim | `crossgen_levels` / `_kcontrast` / `_kcontrast_summary` / `_grader_agreement` / `_vsbase` / `_la3_gpt35` (+ figs `crossgen`, `crossgen_col`) · `sd_by_iter` · `sd_tests` · `sd_tally` · `sd_summary` · `ceiling` (+ fig `sd`) · `crossgen_numbers.json`, `replication_numbers.json` |
| `method/contrast.ipynb` | **RQ-ii — PTO vs GRPO at each K** | `method_paired_by_K` · `method_paired_best` (best-vs-best steelman) · fig `method_gap` · `method_contrast.json`; budget-matched method contrasts are in `compute/cost` |
| `compute/cost.ipynb` | **the COMPUTE axis** — GPU-h + API calls per (arm, iteration); every lever at matched budget | `compute_by_arm` · `compute_by_iteration` · `step_multiplier` · `iso_compute_contrast` · `budget_sweep_<contrast>_<judge>` (`PTO_K`, `GRPO_K`, `method_K0`, `method_K5` × both graders) + `budget_sweep_crossjudge{,_verdicts}` · `iso_channels{,_selected}` · `api_calls` / `api_ratio` · figs `compute_trajectory{,_col}`, `cost_breakdown`, `budget_sweep`, `api_calls` · `compute_numbers.json` |
| `measurement/validity.ipynb` | **is the ruler trustworthy?** §1 judge reliability (oracle ICC + second-judge agreement + contrast preservation) · §2 multi-judge (variance decomposition, gain retention, all-pairs contrasts, sign-preservation ladder, concordance-vs-effect-size) | `multijudge_*` tables + figures (`multijudge_variance_decomposition`, `multijudge_gain_retention`, `multijudge_retention_trajectory`, `multijudge_all_pairs_contrasts`, `multijudge_sign_preservation{,_by_metric}`, `multijudge_variance_components`, `multijudge_coverage`) · oracle repeatability + judge agreement |
| *(scoring — PAID, never part of a render)* | `scoring/Run_Eval.ipynb` · `scoring/Judge_Reliability.ipynb` · `scoring/Local_Judge_Validation.ipynb` | write the score lake `data/eval_scores/judge=<tag>/rep=<r>/` — see "Run order" |

Where a family is not on disk yet, `render_results.py` skips its notebook with a note and
`results/INDEX.md` marks it *(not rendered yet)*.

## The cell-1 contract (the one control)

Cell 1 of every notebook starts with:
```python
import os, eda_analysis
cfg = eda_analysis.EdaConfig(family="arms/outcomes", judge=os.environ.get("EDA_JUDGE", ""))
S   = eda_analysis.notebook_setup(cfg)      # Setup(ARMS, SCORES, PALETTE, METRICS, ORACLE_NOISE, RESULTS_DIR, FAMILY, JUDGE, CFG)
```
`family` is a `"<top>/<sub>"` path validated against `config.FAMILIES` (a typo raises with the full
list rather than creating a phantom folder). It sets **the results root**, `results/<family>/`
(`S.RESULTS_DIR`); it does *not* filter arms — the default arm filter is every arm (`ks=None`), and
`ks` / `methods` / `modes` / `arm_labels` remain explicit filters. `judge` selects which grader's
scores are read (`S.SCORES`); for a per-judge family it also routes exports to
`results/<family>/{figures,tables}/<judge>/`. **Judge-invariant families ignore `EDA_JUDGE`** — a
note is printed, `S.JUDGE == ""`, `S.SCORES` is the primary oracle's frame — and load every grader
themselves:

```python
SC   = eda_analysis.scores_by_judge(S)     # {'gpt-4o-mini': scores_long, 'claude-haiku-4-5': scores_long}, primary FIRST
arms = eda_analysis.cross_k_arms(S)        # == S.ARMS under the all-arms default
```
`scores_by_judge` applies the same arm/metric filters under each grader on disk (default: primary +
every second judge in the lake), switching the active judge per load and **restoring it
afterwards**, so calling it mid-notebook never re-points later loads. `cross_k_scores(S)` /
`cross_k_arms(S)` rebuild the frame with only an explicit `ks` filter dropped (identical to
`S.SCORES`/`S.ARMS` under the default config; kept for configs that narrow K). Neither touches
export routing — pinned by the `cross-K frame` self-check.

Behaviour channels per grader have no `scores_by_judge` twin: loop
`constants.set_active_judge(tag, 0)` → `behavior.channel_scores_long(arms)` → restore `""` (recipe in
`lookahead.channel_k_frames`' docstring); `behavior.text_metrics(arms, attach_persona=True)` is
judge-invariant and passed once.

### Regenerate the results tree
```
python tools/render_results.py                          # everything: arms × every judge on disk + the 4 invariant tops
python tools/render_results.py --top arms               # one top (arms → every judge on disk)
python tools/render_results.py --top lookahead compute  # several tops
python tools/render_results.py --family lookahead/reward            # one notebook
python tools/render_results.py --top arms --judge anthropic_claude-haiku-4-5   # one grader's arms/* leaves
python tools/render_results.py --judge ""               # arms/* for the primary oracle only (+ invariant tops)
python tools/render_results.py --jobs 1                 # sequential (low memory); --isolate = one kernel per notebook (debug only)
python tools/render_results.py --list                   # print the family/judge/unit plan and exit
```
The unit of work is **(top, judge)**: `arms` is rendered once per grader in the score lake (a bare
run covers EVERY judge, so a held-out judge's leaf can no longer go stale silently — the old
`render_views.py` bare run was primary-only); each judge-invariant top is rendered exactly once with
`EDA_JUDGE` unset. Each unit starts one kernel and feeds it its notebooks sequentially in `FAMILIES`
order with `%reset -f` between them (required: the notebooks of one top share
`results/<top>/INDEX.md` and their leaves' `CAPTIONS.md`); units run in parallel (`--jobs`, default
= #units capped at 4). Executed copies go to a throwaway output dir — only `results/` is the
deliverable — and the driver never touches the hand-authored files (`SUMMARY.md`,
`METRICS_REFERENCE.md`, `LIMITATIONS.md`, `schematics/`).

**Committed notebooks are kept output-clean** by `strip_notebook_outputs.py` (zero-dependency): run
it in place (`python tools/strip_notebook_outputs.py`), as a regression guard (`--check`), or wire it
as a git clean filter (see the `.gitattributes` note) so `git add` strips outputs automatically while
the working tree keeps them for viewing. Needs the venv kernel `thesis-venv313` (register once:
`.venv\Scripts\python.exe -m ipykernel install --user --name thesis-venv313`).

> **Renders are deterministic — keep them that way.** Seaborn's `errorbar=("ci", 95)` defaults to
> `seed=None`, so every confidence band used to be a *fresh* 1,000-draw bootstrap: three renders of
> one notebook on identical data differed by ~6% of pixels and re-rendering churned 90 PNGs in git
> for no reason (found 2026-07-28). `constants.BOOT_SEED = 12345` is now passed at **every** seaborn
> callsite that draws a CI, and `stats.bootstrap_ci` / `stats.paired_arrays` and every promoted
> module's bootstrap read the same constant — the figure side and the table side share one seed.
> **Any new `errorbar=` callsite must pass `seed=BOOT_SEED`** (the `seeded bootstrap` self-check
> scans for it), or a thesis figure stops being reproducible. ⚠ The paper generators these modules
> were promoted from used other seeds, so CI *bounds* differ from the frozen paper tables at
> Monte-Carlo scale (≤ 0.02–0.04 on the rubric scale, more on count-scale channels) — point
> estimates, dz, p, n reproduce to ≤ 1e-9. Compare CIs only at a scale-relative tolerance.

## Configuring a notebook (`EdaConfig`)
`EdaConfig` is the single flat-globals control surface (`eda_analysis/config.py`). `EdaConfig()` =
every arm / all present metrics / **no family** (interactive exploration only — every `save_*`
raises `NoFamilyError` until a family is set; there is deliberately no bare-root fallback). Knobs
beyond `family` + `judge`:
- **Arms:** `methods` (`["PTO"]`), `ks` (`[0]` | `[0, 5]`; `None` = both), `modes`, `arm_labels`,
  `include_archived`.
- **Judge:** `judge` (tag; `""` = primary), `judge_rep` (0 = the full-grid draw every judge reports;
  ≥1 = repeatability draws on the anchor subset only, so a non-zero rep yields a mostly-empty frame
  outside those cells).
- **Metrics:** `metrics` (explicit ordered subset), `add_derived_mitiprof` (free R:Q/%CR/%MICO),
  `warmth_only`.
- **Selection / focus:** `selection="all"|"best"`; **`focus_arms`** (default arm subset for
  overlay/trajectory figures) + `focus_metric`.
- **Plot scales:** `context`, `font_scale`, `dpi`, `savefig_dpi`, `panel`, `ncols`, `score_ylim`,
  `share_y`, `palette_overrides` (all default = inherit the publication style).
- **Exports:** `fig_formats` (**default `("png",)`**; `("png","pdf")` to also emit vector),
  `table_formats` (**default `("md","xlsx")`** — readable Markdown + one Excel workbook per leaf,
  one sheet per table). A per-call `group=` on `save_fig`/`save_table`/`save_numbers` nests a
  subpath *inside* the family (`group="trajectories"`, `group="miti"`, `group="headline"`).
- **Cache:** `cache` (**default `True`**) parquet-memoizes the slow disk reads — `scores_long`
  (~60 s cold → ~0.3 s) and the `behavior_by_iter` family (~30 s → ~0.3 s) — to `eda/.eda_cache/`
  (gitignored). Content-keyed on the input CSVs' `(name, size, mtime)`, so a re-score / re-gen
  auto-invalidates; it can never serve stale numbers. Bypass with `EdaConfig(cache=False)`, the
  `EDA_NO_CACHE=1` env var, or `eda_analysis.reset_cache()`. Different arm subsets / judges cache
  independently, so `render_results.py` builds each frame once then reads it across notebooks.
- **Misc:** `oracle_noise` (the 0.10 reproducibility band), `attach_persona`, `verbose`, `note`
  (free text, recorded in the provenance banner). `cfg.with_(**overrides)` returns a patched copy;
  `notebook_setup(cfg, selection="best")` patches on the fly.

**Per-figure control.** Trajectory plots take `arms=`/`iters=`/`metric=`; slice `S.SCORES` with
plain pandas (e.g. `S.SCORES[S.SCORES.arm.isin([...])]`) to point any figure at a subset.
`plots.single_metric_trajectory(..., mark_peaks=True)` auto-flags peak-then-regression arms
(`oracle_noise=None` suppresses the Q1Q2-only noise band); `plots.heterogeneity_grid(S.SCORES, char,
arms=[...])` is one figure (panel per arm).

`notebook_setup(cfg)` resolves the family (→ results root + whether a `<judge>/` level applies) and
the judge (validated against the lake), applies the style + scales, **discovers + filters** the
arms, builds `scores_long` (with the derived ratios) + palette + present metrics, routes exports
via `exports.set_family`, warns loudly if a second judge has NOT scored every conversation of every
arm, and writes a **provenance banner** (`results/<family>/figures/[<judge>/]_provenance.md`).
Prints the family, the judge and the arm list.

### Exports API (`eda_analysis/exports.py`)
| Call | Writes |
|---|---|
| `save_fig(fig, name, *, group=None, formats=None, dpi=200, caption=None)` | `<family>/figures/[<judge>/][<group>/]<name>.png` (+ `.pdf` if asked); caption → the leaf's `CAPTIONS.md` |
| `save_table(df, name, *, group=None, formats=None, float_format, index, caption)` | `<family>/tables/[<judge>/][<group>/]<name>.md` + one sheet in the leaf workbook (`<sub>.xlsx` / `<group>.xlsx`). A **0-row frame** writes an explicit `> **EMPTY TABLE.**` marker and warns, never a silent 0-byte file (an empty artifact masqueraded as rendered once — the L5 `multijudge_gain_retention`, caught 2026-08-18). A `.md` over 64 KB is written as a head excerpt + pointer to the workbook |
| `save_numbers(name, mapping, *, group=None, caption=None)` | a **number ledger** `<family>/tables/[<judge>/]<name>.json` — `{dotted.key: {"value","source","note"}}` like the paper ledgers; existing keys are REPLACED and other keys kept, so several cells feed one ledger. Every promoted module ships a `<topic>_numbers(...)` that returns exactly this mapping |
| `save_provenance(cfg, scores)` | `_provenance.md` (called by `notebook_setup`; re-call after `reset_results`) |
| `build_index()` | `results/<top>/INDEX.md` (every subfamily + judge of that top: figures with captions, tables, ledgers) AND refreshes `results/INDEX.md`; prunes orphan captions first; atomic writes |
| `reset_results(groups=None)` | clears the ACTIVE family's generated `figures/` + `tables/` — **judge-scoped** (a per-judge family clears only the active judge's leaf, never another grader's copy) — or just the named nested `groups`. Never the family/top root, so `SUMMARY.md` survives structurally |
| `set_family(family)` · `active_family()` · `is_judge_invariant()` · `family_root()` · `set_formats()` | routing state (`notebook_setup` does `set_family` from `EdaConfig.family`) |

`PRESERVE = {SUMMARY.md, METRICS_REFERENCE.md, LIMITATIONS.md, schematics}` — never deleted by
`reset_results` / rewritten by `prune_orphan_captions`; enforced structurally (the helpers only
descend into a family's `figures/` + `tables/`) and by a `_guard_path` assertion.
`JUDGE_INVARIANT_GROUPS` is derived from `config.PER_JUDGE_TOPS` (everything except `arms`), never
listed by hand.

## Run order
1. **`scoring/Run_Eval.ipynb`** — async oracle scoring → the score lake,
   `data/eval_scores/judge=<tag>/rep=<r>/` (see "Where the scores live"). The
   `eda_analysis/scoring/registry.py::EXPERIMENTS` registry is **auto-generated from
   `discover_arms()`** (2026-07-11) — a new run is scoreable as soon as its conversations land; no
   registry edit, and empty in-flight `model_iter_*` dirs (no `conversation_*.csv`) are skipped.
   Resume-safe. Score **PCT** + **MICI** with `QUESTIONNAIRE_FILTER=["PCT","MICI"]`.
2. **The family notebooks** in any order (the table above says what lives where), or
   `python tools/render_results.py`. Every notebook auto-discovers arms from disk via
   `eda_analysis.discover_arms()` (no path literals) and ends with `build_index()`. Notebooks run with
   the venv kernel `thesis-venv313`, cwd = `eda/` (cell 1 walks up to find `eda_analysis/`).
3. *(optional, costs API budget)* **`scoring/Judge_Reliability.ipynb`** — measurement-validity
   re-scoring on a subset: oracle repeatability (ICC, per-rep seeds) + a pluggable **second judge**
   (Claude via the `anthropic` SDK, or another OpenAI model) with the PTO−GRPO contrast-preservation
   check. Gated behind explicit `RUN_*` flags; writes to `data/eval_scores/judge=<tag>/rep=<r>/`;
   NOT part of `render_results.py`. Backing module: `eda_analysis/scoring/judge.py`. Addresses
   `results/LIMITATIONS.md` §1–§2 (measured 2026-07-26 with Claude Haiku 4.5 as the second judge).
   Its **§3 promotes the second judge to a full sweep** — all 39 model states × all 8 rubrics —
   through `scoring/judge_plan.py` (free pre-flight: rubric-parity gate, coverage plan, cost model)
   and `scoring/judge_batch.py` (Anthropic Message Batches, 50% off, submit/poll/collect).
   **Presentation is split off deliberately:** this notebook only *scores*;
   `measurement/validity.ipynb` *reads* the same tree via `eda_analysis/reliability.py` (no API
   calls) and exports the tracked tables + figures — same paid-pipeline/free-notebook split as
   `Run_Eval` → the family notebooks, which keeps `render_results.py` fully reproducible without
   spending.

   > **Run the parity gate before any second-judge spend.** Claude's `json_schema` rejects
   > `minimum`/`maxItems`/…, so those are folded into `description`; `check_rubric_parity()`
   > verifies each dropped constraint was restated and that the encodings are otherwise identical.
   > It runs automatically in `python -m eda_analysis._selfcheck`.
   >
   > **Prompt caching, measured not assumed** (2026-07-27, by `prefix_report`): caching does **not**
   > hit on every oracle call — only **Q1 (~1.1k tok) and Q2 (~2.2k tok)** clear OpenAI's
   > 1,024-token cacheable-prefix minimum. WAI-SR/CSQ-8/MI-SAT are rubric-first but too short
   > (403–507 tok); MITI/PCT/MICI interpolate a *per-conversation* utterance count into the
   > instructions **ahead of** the rubric, truncating their prefix to 138–206 tok. Haiku 4.5's
   > minimum is 4,096, so a Claude judge never caches — confirmed empirically
   > (`cached_input_tokens = 0` on every probe call). ⚠ **This is documented, not fixed — do NOT
   > restructure MITI/PCT/MICI to chase the discount:** those interpolated counts are the rate
   > metrics' denominators, and any prompt edit breaks comparability with every conversation
   > already scored (**39 × 8 × 96 = 29,952** cells per grader), for a discount that still would
   > not materialize.
   >
   > **Two quantities that are easy to confuse when costing a sweep** (`prefix_report` returns
   > both): `prefix_tokens_approx` is the *cacheable* prefix and drives the **discount**;
   > `fixed_prompt_tokens_approx` is the whole instruction+rubric block and drives the **input
   > cost**. They diverge sharply on MITI/PCT/MICI (138–206 vs ~1,000 tok) because of the
   > utterance-count invalidator above, so costing off the cacheable prefix underestimates input by
   > ~25%. Likewise `judge_batch.probe_usage` samples at quantile **midpoints**: spreading
   > endpoint-to-endpoint puts the shortest *and* longest transcript in the sample, which at
   > `n=2` is `(min + max) / 2` — 2.1× the true mean on this right-skewed data.
4. *(one-off, costs API budget)* **`tools/score_crossgen.py`** re-scores the ICLR poster's Exp1
   transcripts with the modern grader into `data/eval_scores/_crossgen/`; `lookahead/replication`
   reads it (`crossgen.load_crossgen`).

## Where the scores live

```
data/eval_scores/                                        (a Google Drive symlink)
├── judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<patient_id>.csv
├── _parquet/judge=<tag>/rep=<r>/metric=<M>.parquet      derived fold — the fast READ path
│                                                        while its signature matches (below)
├── _batches/<tag>/rep=<r>/*.json                        Message Batches manifests
├── _crossgen/                                           the re-scored Exp1 transcripts (replication)
└── summary/                                             CSV snapshots from Judge_Reliability
```

Every grader is an equal `judge=` partition — see the box under "Judge dimension" below for what
that buys and what it replaced. `rep=0` is each judge's full-grid draw and the one the thesis
reports; `rep>=1` are repeatability draws on the anchor subset only, so setting `judge_rep` to a
non-zero value yields a mostly-empty frame outside those four model states.

**The parquet fold.** One CSV per conversation is a *write-time* shape: a file is one completed unit
of work, so an interrupted scoring run resumes by skipping what exists. That is right for writing
and wrong for everything else — 50,305 files averaging ~190 bytes are slow to sync and to stat, and
a cold `load_scores_long` spent ~86 s opening them one at a time.
`python tools/consolidate_scores.py {build|verify|report}` folds them to 31 parquet files (0.6 MB,
1,623× fewer files); `verify` re-reads every CSV to prove the fold lossless.

`iter_conv_rows` — the shared inner loop of every per-conversation reader — serves from the fold
when one is present, which is a **4.3–6.1× speedup** on every builder (`scores_long` 86 s → 16 s;
the remainder is the per-row `Series` interface, not I/O). Logic lives in
`eda_analysis/score_archive.py`.

> **The staleness guard is the design.** A second read path that drifts from its source fails
> *silently* — a figure rendered off scores no longer on disk. So `build` records a content
> signature per partition in `_parquet/_manifest.json`, and `rows_for` recomputes it before serving
> anything; any mismatch (re-score, new model, deleted CSV) falls back to the CSVs automatically.
> That is the same (name, size, mtime) mechanism `load_cached` already trusts, not a new
> assumption, and the fold is only ever written by an explicit `build` — never by the scorers.
> Reading is therefore always *correct* and merely *fast when current*. `_selfcheck` asserts both
> halves: fold-equals-CSV, and that a tampered signature is refused rather than served.
>
> Rebuild after new scoring, or delete `_parquet/` — a stale fold costs speed, never accuracy.

## Judge dimension — running the EDA under a second grader

`FAMILY` and `JUDGE` are **orthogonal knobs**: `FAMILY` says which *question* (results folder) a
notebook owns, `JUDGE` selects which *grader's scores* are read.

| `JUDGE` | Reads | Writes (per-judge families, `arms/*`) |
|---|---|---|
| `""` (default) | `data/eval_scores/judge=openai_gpt-4o-mini-2024-07-18/rep=0/` — the primary oracle, the numbers the thesis reports | `results/arms/<sub>/{figures,tables}/gpt-4o-mini/` |
| `anthropic_claude-haiku-4-5` | `data/eval_scores/judge=<tag>/rep=<r>/` | `results/arms/<sub>/{figures,tables}/claude-haiku-4-5/` |

The write path uses the **short model label** (`constants.judge_dirname`: provider prefix and
release date dropped), while the *score* tree keeps the full `judge=<tag>` partition — a path a
human reads vs. a stable partition key.

> **One lake, no privileged grader.** Every score any judge ever produced lives under
> `data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<id>.csv`, so `judge` is an
> ordinary partition key alongside `metric`/`oracle`/`rep` and `Arm.eval_dir()` is a single
> resolver rather than a primary-vs-other branch. `rep=0` is each judge's **full-grid** draw (the
> reported one); `rep>=1` are repeatability draws on the anchor subset. There is no method level —
> `<Model>` already carries it. Before the 2026-07-28 migration the primary lived co-located per
> method while other judges lived in a separate local-only tree, which split the primary across two
> roots under two partition schemes and left the second judge's scores backed up nowhere.

```
results/arms/outcomes/figures/
├── gpt-4o-mini/                             ← primary oracle (the training reward)
│   ├── outcomes_by_model_final.png
│   └── effect_vs_base_forest_final.png
└── claude-haiku-4-5/                        ← same family, held-out judge
    ├── outcomes_by_model_final.png
    └── effect_vs_base_forest_final.png
```

The **judge is the deepest level** of a per-arm family, so its output from every grader sits side
by side and compares in one glance. **Every judge nests, including the primary** (since 2026-07-28):
a figure's path always names the grader that produced it.

**Three kinds of family, three behaviours under `EDA_JUDGE`:**

- **Per-judge `[EVAL]` families (`arms/outcomes questionnaires validity heterogeneity stats`).**
  They read `scores_long` / the oracle behaviour counts and re-grade cleanly — rendered once per
  grader on disk, each into its own `<judge>/` leaf.
- **Per-judge families with `[TRAINING]` sections (`arms/training`, `arms/preference`).**
  Candidate rewards in `generations.jsonl`, PTO preference pairs, TensorBoard curves were produced by
  the *training* oracle during the run and cannot be re-graded after the fact. Under a held-out
  judge those sections print a pointer instead of emitting byte-identical copies into that judge's
  folder (which would imply a measurement that never happened); only the sections that join the
  training reward to the **eval** side (`S.SCORES`, grader-dependent) render per judge.
- **Judge-invariant families (`lookahead/* method/* compute/* measurement/*`).** They contain
  *every* grader — `scores_by_judge` / `reliability.py` load each judge from the lake explicitly and
  ignore `EDA_JUDGE`. Rendered exactly **once**, exported with **no `<judge>/` level**
  (`exports.JUDGE_INVARIANT_GROUPS`) — a path naming one grader would assert that grader produced a
  cross-judge figure. Their tables carry both graders in one frame (columns `_primary`/`_heldout`,
  a `judge` column, or `<judge>`-suffixed file names such as `k_paired_pto_gpt-4o-mini.md`); their
  figures use two panels or the primary column first.

This works at all because `scoring/judge*.py` writes its CSVs in the *identical*
`metric=/oracle=/<Model>/<file_index>.csv` layout with identical column names, so `Arm.eval_dir()`
only has to swap a root.

**Coverage is checked, not assumed.** `notebook_setup` warns loudly when a judge has not scored
every conversation of every arm — a partially-landed sweep otherwise yields arm means that look
like the primary's but rest on smaller, unequal samples, and persona-paired contrasts between two
such arms overlap on only a fraction of personas.

### Adding a third grader — the open-weights path (`$0` in API spend)

API spend is the binding constraint on the project, so there is a track for auditioning a **locally
served open-weights grader**: if one reproduces the contrasts, every future experiment's measurement
is free; if it doesn't, that was learned for free rather than inside a training run.

| Piece | Role |
|---|---|
| [`scoring/local_server.py`](eda_analysis/scoring/local_server.py) | starts vLLM (or attaches to a server you started), waits for ready, and wraps it as an ordinary `JudgeSpec` |
| [`notebooks/scoring/Local_Judge_Validation.ipynb`](notebooks/scoring/Local_Judge_Validation.ipynb) | the audition — two gates, then the sweep |

**It adds no scoring code.** vLLM / llama.cpp / Ollama speak the OpenAI protocol *including*
`response_format={"type":"json_schema"}` via constrained decoding, so a local Gemma is reachable
through the same `JudgeSpec` path — same prompts, same parsing, same validation, same
resume-by-skipping-CSVs, same score-lake partition (`judge=local_<shorttag>`).

⚠ **`check_rubric_parity` is NOT the gate here.** It asks a *static* question — were the constraints
stripped for Claude restated in prose. A local server strips nothing, so parity is trivially clean
and tells you nothing. The real risks are empirical, and there are two gates for them:

1. **Schema** — does the backend honour each rubric's `json_schema`? `run_judge_scoring` swallows
   per-call errors and skips the conversation, so a rubric the model can't satisfy shows up as
   **biased missingness**, not as an error.
2. **Discrimination** — can it separate two arms the primary oracle puts far apart (Base vs PTO@I10
   is +1.26 on Q1+Q2, the largest contrast in the experiment)? A small model can honour the schema
   perfectly and answer from a template — every item a 4, near-zero per-conversation SD. That
   parses, writes valid CSVs, and produces a judge that cannot tell any two arms apart. **Nothing
   downstream would flag it**; the agreement tables would come back ≈0 and look like a finding.

`probe_rubrics` and `probe_discrimination` catch both in a handful of calls, before committing to a
22k-cell sweep. Run it on Colab, where the GPU is otherwise idle during scoring.

## Package (`eda_analysis/`) — analysis modules on a `constants` leaf + `scoring/` and `plotting/` subpackages
Plumbing was consolidated (2026-06-18) from 14 modules to 9; the analysis/topic files stay separate.
`figures`/`plots` still resolve as aliases of `plotting`; the data-module aliases were retired
(2026-07-08). The Layer-0 core was extracted (2026-07-08) into a **`constants` leaf**, breaking the
old `__init__`↔submodule import cycle — submodule imports are plain top-level
`from .constants import ...` (only genuinely cross-module imports remain deferred). On 2026-07-13
the legacy `oracle_scoring/` package was **folded in** as the `scoring/` subpackage and `plotting.py`
was **split** into the `plotting/` subpackage's topic modules behind an unchanged public surface. On
2026-08-18 the paper's eight analysis generators (`papers/2026_lookahead_pto_grpo/analysis/*.py`)
were **promoted** into eight new modules + their plotting twins (below); the paper's
`analysis/out/*.json` + `tables/*.csv` are kept as the frozen fixture the self-check compares
against.

- **`constants`** — the LEAF (imports nothing from the package): workspace-root resolution +
  `sys.path` bootstrap, `QUESTIONNAIRES`/`QUESTIONNAIRE_ORDER`/`WARMTH_RUBRICS` (the global-eval
  halo cluster — historical code name)/`EXTRA_METRICS`/`LOWER_IS_BETTER`,
  `MITI_THRESHOLDS` (official 4.2.1 fair/good), `Q1_ITEM_SHORT`/`Q2_ITEM_SHORT`/`Q2_ITEM_GROUPS`
  (item labels + face-content groups), `ITEM_QUESTIONNAIRES` (per-item column layout of every
  Likert-item rubric; item text source of truth = `code/questionnaires.py`),
  `DISPLAY_NAMES`/`ARM_LABELS`, `display_label`/`short_label`/`arm_label`/`item_short_label`,
  the shared `RE_AFFIRM` cue, `BOOT_SEED`, the judge tag helpers (`set_active_judge`,
  `active_judge`, `judge_dirname`, `PRIMARY_JUDGE_TAG`).
- **`config`** — `FAMILIES` (top → subs) + `PER_JUDGE_TOPS`, `split_family`/`is_per_judge`/
  `all_families`, `EdaConfig` (the single control surface, incl. `family` + `judge` + PNG/xlsx
  defaults) + `notebook_setup(cfg)` → `Setup` (`S.FAMILY`, `S.JUDGE`, `S.RESULTS_DIR`, `S.CFG`, …),
  `scores_by_judge(S)` (the judge-invariant read path), `cross_k_scores(S)` / `cross_k_arms(S)`.
- **`data`** — the load+shape layer: arm **discovery** (`discover_arms`/`filter_arms`/`Arm`), TRUE-
  **persona** recovery (`attach_personas`/`canonical_personas` — replays the per-iter shuffle), the
  **`scores_long`** backbone (`load_scores_long`/`load_subscales`/`load_items` [generic per-item
  loader over `ITEM_QUESTIONNAIRES`; `load_q2_items` wraps it]/`to_wide`/`collapse_base`/
  `add_derived_mitiprof_rows`), and **selection** (`all_models`/`best_per_experiment`/
  `final_per_experiment`/`best_iteration_by_arm` — the final-vs-best machinery).
- **`score_archive`** — the score lake's parquet fold: `build`-side helpers, the signature-guarded
  read path (`rows_for`, used by `data.iter_conv_rows`), and `fold_status()`. Imports only
  `constants`, so `data` can depend on it without a cycle. See "Where the scores live" above.
- **`compute`** — the **COMPUTE axis** (`compute/cost`): GPU-hours per (arm, iteration)
  reconstructed from artifact mtimes (`iteration_compute`, `compute_summary`, `step_multiplier`,
  `iso_compute_pairs`, `iso_compute_contrast`, `budget_sweep`, `score_by_compute`) — extended
  2026-08-18 with the floor columns (`compute_by_iteration_with_floor`, `compute_by_arm_with_floor`),
  `cost_ratios`, `step_multiplier_table`, `budget_sweep_ci` / `all_budget_sweeps` /
  `budget_sweep_top` / `budget_sweep_crossjudge` / `crossjudge_verdicts`, `iso_channels` /
  `iso_channels_selected`, `compute_numbers`. Every *other* contrast is indexed by iteration, which
  is not a fixed unit of spend — a K=5 step costs ~1.9x a K=0 step and a whole PTO iteration costs
  a fraction of a GRPO one. ⚠ Never time a run from `iteration_metadata.json` (per-PROCESS
  `*_time_s`); ⚠ iso-compute pairs on `persona_id`, never `file_index`; ⚠ quote a budget *sweep*,
  not one iso-compute row — the lever's sign is a function of budget.
- **`lookahead`** (→ `lookahead/reward`; channels/text → `lookahead/behaviour`) — the persona-paired
  K contrast per grader: `paired_k_frames`, `k_levels`, `k_table1`, `k_summary`
  (**qualify: `lookahead.k_summary`** — `faithfulness` and `crossgen` define different tables of the
  same name), `channel_k_frames`, `did_by_iter`, `method_gap_by_iter`, `endpoint_contrasts`,
  `lookahead_numbers`; helpers `model_name`, `wide_by_persona`, `holm_within`, `best_iteration`.
- **`transfer`** (→ `lookahead/transfer`) — does the K contrast survive the held-out judge:
  `cross_k_pairs`, `sign_ladder`, `retention_by_k` (several reference kinds; its CIs use
  `gain_retention`'s own seed), `transfer_numbers`, `to_reliability_long`.
- **`tails`** (→ `lookahead/mechanism` for the audit, `compute/cost` for the API-call tables) —
  what the K-step reward actually saw: `tail_audit_frames` (the ONE expensive pass, ~1–2 min per
  LA5 arm, memoized) → `tail_audit_by_iter` / `tail_cues_by_iter` / `tail_within_group` /
  `score_by_realized_turns`; `api_calls` / `api_ratio`; `tails_numbers`.
- **`dispersion`** (→ `lookahead/mechanism`) — does look-ahead sharpen or rescale the training
  signal: `iid_expectation`, `dispersion_by_iter`, `dispersion_ratios`, `tau_sensitivity`,
  `dispersion_numbers` (pass `gens=` between calls to avoid reloads).
- **`faithfulness`** (→ `lookahead/mechanism`) — reward faithfulness by K at a matched policy:
  `faithfulness_data` (~65 s after `load_branch_reliability` ~90 s) → `faithfulness_curve`,
  `faithfulness_by_iter`, `k_faithfulness_by_iter`, `matched_policy`, `k_summary` (**qualify**),
  `by_cooperation`, `proxy_levels`, `check_against_rank_agreement`, `faithfulness_numbers`.
- **`crossgen`** (→ `lookahead/replication`) — the ICLR poster's Exp1 transcripts under the modern
  grader: `ICLR_TABLE1`, `load_crossgen`, `load_exp1_gpt35`, `persona_alignment_check`,
  `table1_crosscheck`, `levels` / `k_contrast` / `k_summary` (**qualify**) / `ordering_claims` /
  `grader_agreement` / `vs_base` / `la3_gpt35` / `la3_cost_estimate`, `crossgen_all`,
  `crossgen_numbers`.
- **`replication`** (→ `lookahead/behaviour` for shape/length/selection; `lookahead/replication`
  for SD/ceiling) — session shape + the "lowest SD = more stable" claim: `shape_text_metrics`,
  `session_shape_levels` / `session_shape_paired`, `length_endpoints` / `length_kcontrast`,
  `sd_by_iter`, `sd_tests` (Brown-Forsythe + Pitman-Morgan), `sd_tally` / `sd_summary` / `ceiling`,
  `selection_table`, `replication_numbers`.
- **`instruments`** (→ `lookahead/behaviour`) — the held-out instruments under K:
  `instrument_frames_by_judge` (the one disk reader), `endpoints` / `matched_endpoints`,
  `wai_subscales` / `wai_kcontrast` / `wai_fig_data` / `wai_subscale_parity`, `pct_kcontrast`,
  `q2_items`, `hetero_kcontrast` / `hetero_ceiling`, `instruments_numbers`.
  ⚠ Names defined by more than one of the promoted modules (`ARMS`, `METHODS`, `GROUP_KEYS`,
  `TEXT_CHANNELS`, `COOP_LABEL`, `COOP_ORDER`, `SIGN_NOTE`, `CENSOR_NOTE`, `CAPTIONS`, `k_of`,
  `method_of`, `k_summary`) are NOT re-exported at package top level — qualify them
  (`eda_analysis.lookahead.k_summary`).
- **`plotting_style`** — the style/scaffold helpers (Okabe-Ito palette [PTO cool / GRPO warm / Base
  grey], `grid`, `set_style(cfg)`, `clean_label`, `apply_score_axis`, `model_order`, `relabel_*`,
  `add_base_line`, `figure_legend_from`). Re-imported into `plotting`, so `figures.set_style(...)`
  etc. still resolve. (No `K_STYLE` here — each promoted plotting module defines its own; use
  `plotting.<module>.K_STYLE`.)
- **`plotting/`** (subpackage) — the named figures, split by topic behind a re-exporting `__init__`
  (the public surface is the flat module's): `outcomes` (per-model bars, `effect_forest`,
  `leaderboard_scorecard` — endpoint figures take `title=`/`selection=` for the final-vs-best
  pairs) · `trajectories` (`trajectory_grid`, `single_metric_trajectory`, subscales,
  `reward_hack_panel`) · `compute` (`compute_trajectory`, `budget_sweep_plot`, `cost_breakdown` —
  + the promoted `budget_sweep_grid`, `trajectory_by_compute` [`layout='wide'|'col'`],
  `cost_breakdown_by_iteration`) · `heterogeneity` (persona splits; `subgroup_endpoint_bars(iter_by_arm=)`
  for best-iteration bars) · `structure` (`reliability_curve`, proxy-vs-eval, diverging
  `rubric_correlation_heatmap`, `factor_loadings_bars`) · `behavior` (the generic wide-frame
  detail grid reused by MITI/MICI/PCT + session shape, MITI thresholds, cross-checks) ·
  `questionnaires` (`item_trajectory_grid` + `item_delta_bars` — the uniform per-rubric item
  figures — + the Q2 specializations) · `training` (reward distribution, advantage side-by-side) ·
  `reliability` (`oracle_repeatability_bars`, `judge_agreement_scatter`, `judge_contrast_bars`) ·
  the promoted twins **`lookahead`** (`k_headline_fourarm`, `k_delta_grid`, `k_channels_grid`,
  `k_contrast_both_judges`, `k_retention`, `k_did`; + the pre-existing `k_channel_trajectory{,_grid}`,
  `k_mechanism_panel`, `k_channel_forest`, `k_cost_benefit`) · **`tails`** (`tail_audit_fig`,
  `api_calls_fig`) · **`dispersion`** (`dispersion_fig`, `tau_fig`) · **`faithfulness`**
  (`faithfulness_fig`, once per eval grader) · **`crossgen`** (`crossgen_fig`, `layout='wide'|'col'`) ·
  **`replication`** (`shape_fig`, `sd_fig`) · **`instruments`** (`wai_fig`, `hetero_fig`).
  *(aliased back as `eda_analysis.figures`/`plots`.)*
- **`stats`** — persona-paired Wilcoxon/dz/bootstrap + Friedman/Kendall-W + `main_results_table` +
  `paired_method_comparison` (PTO vs GRPO) + `paired_best_method_comparison` (best-vs-best model
  selection) + `paired_k_comparison` (K0 vs K5) + **`paired_arrays(a, b, *, n_boot=2000,
  seed=BOOT_SEED)`** → `{n, mean_delta, dz, ci_lo, ci_hi, p}` (the one paired primitive every
  promoted module uses) + `holm` + `item_endpoint_deltas` (generic "which items drive the change";
  `q2_item_endpoint_deltas` wraps it) + `rank_agreement_by_nturns` (reward reliability) +
  `rubric_pca`/`rubric_factor_space` + `filter_thin_arms`.
- **`behavior`** — MITI counts (+ per-conv `%MICO`) + over-praise cross-check + structural text
  metrics + `miti_detail_by_iter` (the MITI drill-down frame behind `miti_detail_grid`) +
  `session_shape_by_iter` (exported text metrics) + `miti_proficiency_by_iter` (the
  official-threshold summary scores) + `channel_scores_long` / `text_metrics` (the channel frames
  the look-ahead families contrast).
- **`training`** — `generations.jsonl` proxy reward + degeneracy scan + pref pairs +
  `advantage_signal_by_iter`/`reward_distribution_frame` + `load_branch_reliability` +
  `tb_curves`/`parse_run_tb` (self-contained TensorBoard parse, no torch/trl); `load_generations`
  is memoized (~60 s cold).
- **`pref`** — what the training signal pushes toward, in three layers (`arms/preference`).
  - *PTO pairs (original):* Mass-Mean-Probe over `pairs.csv` — word ranking/drift,
    `preference_direction_drift`, `learn_unlearn_words`, MI-concept projection.
  - *Both methods (2026-08-02):* `load_weighted_candidates` reads `generations.jsonl` and gives
    every candidate the weight its method's update applies (DPO's recorded ±1 chosen/rejected;
    GRPO's standardized group-relative advantage), rescaled per group to `Σ|w| = 2` so the two are
    one scale. Then `weighted_lexical_contrast` (exact, every group, with SEs),
    `direction_by_iter`/`direction_by_arm` (`normalize(Σ w·emb)` — §1's probe generalized to any
    weighting), and `pooled_direction_cosines` with an **attenuation ceiling**.
    `direction_quality` audits the probe: `wins_holdout` (each half judged by the other half's
    direction) and `split_half_cos`. That audit is why §1's per-iteration PTO readouts now carry a
    caveat — measured split-half ≈ 0.19 per iteration, vs 0.60 pooled.
  - *Signal → outcome:* `preference_features_by_iter` → `link_to_outcomes` (persona-paired eval
    delta of the update's own iteration) → `outcome_correlations` (**read `rho_partial_iter`** —
    the raw ρ is confounded with iteration index by construction).
  - *Loss vs data:* `reweight` swaps one method's weighting rule onto the other's groups, and
    `weighting_decomposition` turns that into as-trained / same-data-other-rule /
    same-rule-other-data cosines — the test of whether "PTO vs GRPO" is about the loss or about
    the candidates each generates. `rule_reconstruction_check` guards the reconstruction.
  - *Generation vs selection:* `pool_mean_by_iter` (what the policy produces) against
    `weighted_lexical_contrast` (what the update selects for). `pair_yield_by_iter` counts how many
    groups actually trained; `pref_examples` pulls the decisive pairs as text.
  - Figures: `plot_lexical_push`, `plot_pref_outcome`, `plot_category_compare`,
    `plot_selection_vs_generation`, `plot_pair_yield`.
- **`reliability`** — MEASUREMENT-validity tables from the `data/eval_scores/` lake (all judges, all reps).
  Disk-only — the paid scoring lives in `scoring/judge*.py`; this is the free read side. It backs
  **`measurement/validity`**.
  - *§1 (single-judge validity):* `repeatability` (ICC(2,1) + mean |Δ|), `agreement` (second judge
    vs primary + attenuation ceiling), `contrasts` (does each endpoint contrast keep its sign?),
    `arm_means_by_judge`, `summary_line`.
  - *§2 (multi-judge):* `variance_components_conversation` / `variance_components_arm` (two-way
    random-effects decomposition → arm vs judge-level vs **arm×judge**, plus `dependability_k1/k2`),
    `gain_retention` (the reward-hacking transfer test, persona-bootstrap CI), `all_pairs_contrasts`
    (every model pair, paired on the recovered `persona_id` — see `attach_persona`, since
    `file_index` is reshuffled per iteration), `sign_preservation` (the *rate* over that table,
    laddered by effect size and optionally `by=["metric"]` — a pooled rate is uninterpretable because
    it counts contrasts too small to claim), `concordance_by_effect_size`,
    `multi_judge_summary_line`. **Never averages raw scores across judges** — the primary judge was
    the training reward and the second is held out, so this is train-vs-test, not two raters.
  - Also `judge_tags()` / `second_judge_tags()` — the lake's grader list that `scores_by_judge`
    and `render_results.py` iterate.
- **`exports`** — see "Exports API" above: `save_fig` / `save_table` / `save_numbers` /
  `save_provenance` / `build_index` / `reset_results` / `set_family` / `PRESERVE`.
- **`scoring/`** (subpackage; NOT imported by `__init__` — its registry scans disk, which the
  analysis notebooks never need; the scoring notebooks import it explicitly) — the
  oracle-scoring layer, folded in from the legacy `oracle_scoring/` package (2026-07-13):
  `registry` (eval settings + the `EXPERIMENTS` registry — auto-generated from
  `eda_analysis.data.discover_arms()` since 2026-07-11 — + the `eval_scores/` layout helpers +
  `ScoringConfig`, formerly `EDAConfig`) · `conversations` (scoring-side conversation loading +
  model-name metadata) · `pipeline` (the async oracle pipeline behind `Run_Eval.ipynb`; formerly
  `eval.py`) · `judge` (the `Judge_Reliability.ipynb` backend — pluggable OpenAI/Anthropic judges,
  ICC(2,1), agreement + contrast-preservation stats) · `judge_plan` (**free pre-flight** for a full
  sweep: `check_rubric_parity` — the gate that both judges receive the same rubric —
  `prefix_report` (which rubrics actually prompt-cache), `plan_sweep` (coverage-aware call count),
  `estimate_cost` / `sweep_report`) · `judge_batch` (the **Anthropic Message Batches** path, 50% off:
  `submit_sweep` → `poll_batches` → `collect_batches`, three phases with disk-persisted manifests so
  a fresh kernel can collect; plus `probe_usage` for a measured token profile) · `local_server`
  (vLLM as a `JudgeSpec` — see "Adding a third grader").
- **`_selfcheck`** — the guard, **23 checks** (12 structural + 11 data + 1 opt-in probe): package
  invariants + `__all__`, the **family map** (every `FAMILIES` entry ↔ `notebooks/<top>/<sub>.ipynb`;
  `PER_JUDGE_TOPS ⊂ FAMILIES`), `EdaConfig` round-trip, the scoring surface, notebook symbol refs,
  the cache round-trip, the second-judge rubric-parity gate, the cross-judge artifact layout, exports
  routing (`save_*` refuse without a family; `PRESERVE` guards; frozen xlsx timestamps), seeded
  bootstraps, role bindings; then arm discovery, arm-identity collisions, the known headline means,
  the cross-K frame, the compute axis, **the paper fixture anchors** (the frozen
  `papers/2026_lookahead_pto_grpo/analysis/out/*.json` numbers reproduce from the promoted modules —
  hard checks), render freshness per judge, the update probe, the persona permutation, judge
  routing, the parquet fold (equals the CSVs *and* refuses a tampered signature), and the
  multi-judge layer. Run `python -m eda_analysis._selfcheck` after any EDA change (`--fast` runs the
  12 structural ones; `--probe` adds the heavy pref probe + the module-side tail-audit anchor).
- **`__init__`** — thin re-export hub: re-exports the `constants` leaf + every analysis submodule's
  public names (the eight promoted modules are also attributes: `eda_analysis.lookahead`, …), and
  the `figures`/`plots` → `plotting` aliases. No definitions of its own.

## Extension points — where each kind of change goes

**Analysis layer (`eda_analysis/` top level) needs NO registry edits** — it auto-discovers arms from
disk. Extend it by concern:

| Adding | Goes in |
|---|---|
| **a new family** (a new question) | one entry in `config.py::FAMILIES` (+ `PER_JUDGE_TOPS` if its artifacts are produced by one grader) + one notebook `notebooks/<top>/<sub>.ipynb` following the cell-1 contract; `_selfcheck`'s `family map` asserts the two stay 1:1, `render_results.py` iterates the dict, `build_index` picks the folder up. A new **top** also gets a hand-authored `results/<top>/SUMMARY.md` |
| a new rubric | `constants.py::QUESTIONNAIRES` + `data.py` (the `scores_long` backbone) |
| a new judge | `scoring/judge.py` (`JudgeSpec`) — its scores land in `data/eval_scores/judge=<tag>/rep=<r>/`; `arms/*` picks it up on the next bare render, `scores_by_judge` returns it to every judge-invariant notebook (columns are suffixed `_<label>` once there is more than one held-out grader) |
| a new arm naming scheme | `data.py::parse_experiment_name` |
| new stats | `stats.py` (paired primitives → `paired_arrays`) |
| new figures | the topic module in `plotting/` (+ its `__init__` re-export); a promoted family's figures go in its plotting twin |
| a results-layout change | `exports.py` (leaf composition) + `config.py` (`FAMILIES` / `PER_JUDGE_TOPS`) — nowhere else knows the tree |
| anything about what a run COST | `compute.py` (+ `tails.api_calls` for the API side) |

Contract for module authors (from the promotion): functions take frames (`scores_long` per judge,
arms) and return tidy `pd.DataFrame`s / `matplotlib` figs; **no disk writes** (notebooks call
`exports.*`); use `stats.paired_arrays`, `stats.holm`, `constants.BOOT_SEED`, `plotting_style`;
keep every caveat as a docstring note; give the module an `__all__` and re-export it from
`eda_analysis/__init__.py` (the `__all__` self-check must still resolve).

(`figures`/`plots` are still aliased to `plotting`; the data-module aliases
`discovery`/`personas`/`scores`/`select` were retired — use `eda_analysis.data.*` or the top-level
re-exports.)

> ⚠ **An arm whose folder name `parse_experiment_name` cannot match is INVISIBLE, silently.**
> `discover_arms` parses every folder and `continue`s on `None` with no warning — so a trained,
> paid arm never reaches `Run_Eval`'s auto-generated registry or any notebook, and the EDA looks
> complete. `_selfcheck` does **not** catch this: its naming check only asserts that the legacy
> `PTO_Iterative_…_PTgreedy` / `GRPO_Iterative_…_G8` forms (± role-binding suffixes) parse — it never
> asks whether every folder on disk did. After touching `EXPERIMENT_NAME` or `data.py::_EXP_RE`,
> compare `len(discover_arms())` against the folder count under
> `data/{pto,grpo}_Exp3/conversations/full/`.

**Scoring layer ([`eda_analysis/scoring/`](eda_analysis/scoring/) — the `Run_Eval` +
`Judge_Reliability` backend):**

- **[`scoring/registry.py`](eda_analysis/scoring/registry.py)`::ORACLE_TOKEN_ALIASES`** — new
  oracle-name aliases go here (`CSQ` vs `CSQ8` vs `CSQ_8`), **not** in `conversations.py`. Lookup is
  on the uppercased, `-`→`_` form, so keys must be canonical uppercase.
  `conversations._normalize_oracle_token(strict=True)` raises on an unknown token; the default
  `strict=False` lets it fall through to the `"Other"` group for backward compat.
- **`scoring/registry.py::COMPOSITE_METRICS`** — new composites (a mean across several source
  columns). Currently only `Q1Q2_Mean` (`sources=["Q1_Mean","Q2_Mean"]`, `aggregator="mean"`); the
  same pattern would produce `MITI_GlobalMean` from the 4 MITI globals.
- **`scoring/registry.py::EXPERIMENTS`** — the registry of trained-model data locations,
  **auto-generated at import** by `build_experiments_from_disk()` from
  `eda_analysis.data.discover_arms()` (2026-07-11). Nothing to edit; new runs are picked up once
  their conversations land. (If the Drive symlinks are offline the registry is empty and a warning
  prints.)
- **[`scoring/judge.py`](eda_analysis/scoring/judge.py)** — add second-judge providers/models here
  (`JudgeSpec`); output lands in `data/eval_scores/judge=<tag>/rep=<r>/`, never in another grader's
  partition. **Claude judges:** `json_schema` rejects `minimum`/`maximum`/`minItems`/`maxItems`, so
  those are folded into `description` — ⚠ **do NOT just drop them**, or the array-shaped rubrics lose
  their one-score-per-item guarantee. Sonnet 5 / Opus 4.8+ additionally need
  `thinking={"type":"disabled"}`, or adaptive thinking eats `max_tokens`.
- **[`scoring/judge_plan.py`](eda_analysis/scoring/judge_plan.py)** (FREE pre-flight, no API) —
  `check_rubric_parity()` is **the gate before any second-judge spend**: it verifies that every
  constraint stripped for Claude was restated in `description` and that the two encodings are
  otherwise structurally identical. It runs automatically in `_selfcheck`. Also `prefix_report()`
  (which rubrics actually prompt-cache — only Q1/Q2 clear OpenAI's 1,024-token minimum; measured
  scope + the do-not-fix prohibition under "Run order" §3), `plan_sweep()` (coverage-aware call
  count, skips conversations whose CSVs already exist), `estimate_cost` / `sweep_report`.
  ⚠ **Pricing lives in `JUDGE_PRICING` — verify against the billing dashboard before quoting a
  number.** Its `claude-sonnet-5` row is flagged *intro pricing thru 2026-08-31*, and a model with no
  row raises rather than guessing.
- **[`scoring/judge_batch.py`](eda_analysis/scoring/judge_batch.py)** (PAID) — the full-sweep path
  via **Anthropic Message Batches (50% off)**: `submit_sweep` → `poll_batches` → `collect_batches`,
  three separate phases with manifests persisted under `data/eval_scores/_batches/` so collection
  works from a fresh kernel. ⚠ **`custom_id` is an opaque index into that manifest, NEVER an encoded
  path** — model+metric+oracle overflows the 64-char limit, and a truncation collision would write a
  score into the wrong model's folder. Anthropic-only by design: the primary judge already has a full
  rep, and extra reps are cheap enough for the live path.
- **[`reliability.py`](eda_analysis/reliability.py)** (analysis layer, disk-only) — the FREE read
  side of `data/eval_scores/`: the ICC / agreement / contrast tables for `measurement/validity` §1
  plus the multi-judge layer for its §2 (`variance_components_arm` → arm vs judge-level vs
  arm×judge + `dependability_k1/k2`, `gain_retention`, `all_pairs_contrasts`, `sign_preservation`,
  `concordance_by_effect_size`). Figures in `plotting/reliability.py`. Keep the paid scoring in
  `scoring/judge*.py` and the presentation here, so judge results render inside
  `tools/render_results.py` without spending.

### ⚠ Two things never to do to these numbers

- ⚠ **Never average raw scores across judges.** The primary oracle WAS the training reward and the
  second judge is held out — that is train-vs-test, not two raters. The level offset is **1.2–1.7
  points *and* model-dependent**, so averaging applies a silent model-dependent shrinkage to every
  effect. Combine only **contrasts or standardized quantities**. The judge-invariant families put
  the two graders *side by side* for exactly this reason — two columns, never one mean.
- ⚠ **Pair on `persona_id`, not `file_index`.** The 96 personas are reshuffled every iteration, so a
  `file_index` join across unmatched iterations pairs unrelated conversations. Means survive it;
  `dz` and CIs do not. Recover the true persona with `reliability.attach_persona` (the multi-judge
  read path) or `data.attach_personas` (the frame-level shuffle replay) — **two different real
  functions, not a typo; do not "unify" them.**

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → `Run_Eval` (the registry auto-discovers
the run) → the notebooks pick it up automatically (re-run `python tools/render_results.py`).

## Results
Not duplicated here (so they can't drift). The written analysis per research question lives in
**`results/<top>/SUMMARY.md`** (`arms/`, `lookahead/`, `method/`, `compute/`, `measurement/`) beside
each top's auto `INDEX.md`; the metric definitions in [`results/METRICS_REFERENCE.md`](results/METRICS_REFERENCE.md)
and the measurement/inference limitations in [`results/LIMITATIONS.md`](results/LIMITATIONS.md)
(both moved from `eda/docs/` on 2026-08-18); the hand-authored method schematics in
[`results/schematics/`](results/schematics/) (their captions in `CAPTIONS.md` there). The live status
+ headline is [STATUS.md](../../STATUS.md) at the repo root.

## Migration (2026-08-18) — old → new

The results tree was reorganised by research question (the design note lived at
`eda/REORG_2026-08-18.md` while the work was in flight). **Commit `b09eb6f` is the last pre-reorg
state** — check it out to re-run anything that read the retired `L0`/`L5` tree (the deck builders
written before that date, the paper's own `analysis/*.py` generators). `L0/`, `L5/`, the `VIEW`
knob, `RQ_I_VIEW`, `render_views.py` (`EDA_VIEW`) and `EdaConfig(view=, export_group=)` are gone;
`lookahead/` owns cross-K.

| Old | New |
|---|---|
| `results/L0/figures/1_outcomes/<judge>/x.png` | `results/arms/outcomes/figures/<judge>/x.png` (now all four arms) |
| `results/L5/tables/7_stats/<judge>/k_paired_by_method.md` | `results/lookahead/reward/tables/k_paired_by_method.md` (both graders inside) |
| `results/L5/tables/7_stats/<judge>/{compute_*, budget_sweep, iso_compute_contrast, k_step_multiplier}.md` | `results/compute/cost/tables/…` |
| `results/L5/tables/7_stats/<judge>/{method_paired_by_K, method_paired_best}.md` | `results/method/contrast/tables/…` |
| `results/L5/tables/7_stats/<judge>/{main_results, friedman_omnibus, vs_base_paired, slope_by_arm, rubric_pca_pc1}.md` | `results/arms/stats/tables/<judge>/…` |
| `results/L5/tables/6_preference/gpt-4o-mini/k_mechanism_overpraise_chain.md` | `results/lookahead/mechanism/tables/…` |
| `results/*/tables/8_measurement/…` | `results/measurement/validity/tables/…` |
| `papers/2026_lookahead_pto_grpo/tables/k_contrast_headline_*.md` | `results/lookahead/reward/tables/…` (rubrics) / `lookahead/behaviour/tables/…` (channels, text) |
| `papers/…/tables/cross_k_multijudge_{pairs,ladder,retention*}.md` | `results/lookahead/transfer/tables/…` |
| `papers/…/tables/cross_k_multijudge_{did,method_gap,endpoints}.md` | `results/lookahead/reward/tables/…` |
| `papers/…/tables/compute_axis_*.md`, `tail_audit_api_*.md` | `results/compute/cost/tables/…` |
| `papers/…/tables/tail_audit_{by_iter,within_group,score_by_realized_turns,cues_by_iter}.md`, `dispersion_by_k_*.md`, `reward_faithfulness_*.md` | `results/lookahead/mechanism/tables/…` |
| `papers/…/tables/session_shape_stability_{shape,length_*,selection}.md`, `held_out_instruments_*.md` | `results/lookahead/behaviour/tables/…` |
| `papers/…/tables/session_shape_stability_{sd,sd_bf,sd_tally,sd_summary,ceiling}.md`, `crossgen_exp1_*.md` | `results/lookahead/replication/tables/…` |
| `Exp3_PTO_GRPO/figures/*` | `results/schematics/*` |
| `eda/docs/{METRICS_REFERENCE,LIMITATIONS}.md` | `results/{METRICS_REFERENCE,LIMITATIONS}.md` |
| `code/_local_smoke.py`, `code/PTO_Exp3/generate_eval_convs.{py,ipynb}` | `code/tools/…` |
| `tools/render_views.py` (`EDA_VIEW`) | `tools/render_results.py` (`EDA_JUDGE`) |
| `EdaConfig(view=, export_group=)`, `RQ_I_VIEW` | `EdaConfig(family=)`; no view owner — `lookahead/` owns cross-K |

Old numbered notebooks (`notebooks/analysis/1_Outcomes` … `8_Measurement_Validity`) map onto the
family notebooks as: `1_Outcomes` → `arms/outcomes` (the `0_headline` re-saves become
`group="headline"`); `2_Questionnaire_Detail` → `arms/questionnaires`; `3_Validity_and_Hacking` (+
`7_Stats` `grpo_iter9_check`) → `arms/validity`; `4_Heterogeneity` → `arms/heterogeneity`;
`5_Training` → `arms/training` (per-arm) with the K-faithfulness contrast in `lookahead/mechanism`;
`6_Preference` §1–§5 → `arms/preference`, its §5d K-mechanism panel → `lookahead/mechanism`;
`7_Stats` → `arms/stats` (main tables) + `lookahead/reward` (§4c) + `lookahead/behaviour` (§4d) +
`method/contrast` (§4a/§4b) + `compute/cost` (§4e); `8_Measurement_Validity` →
`measurement/validity` (unchanged content). The hand-authored `L0`/`L5` `SUMMARY.md`s were
redistributed by section into `results/<top>/SUMMARY.md` (numbers unchanged, paths rewritten).

## Roadmap
Dated pass history (2026-06-09 → today) is in [history/CHANGELOG_EDA.md](../history/CHANGELOG_EDA.md).
