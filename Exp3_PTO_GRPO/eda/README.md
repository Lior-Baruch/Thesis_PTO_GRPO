# Exp3 EDA — guide + improvement roadmap

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), across
training iterations, under matched look-ahead K and MCL. All data/compute/stats lives in the
`eda_analysis/` package; the recurring figures are named functions in `eda_analysis/plotting.py`
(called once from multiple notebooks), and genuinely one-off exploration stays inline (the **hybrid**
plotting split). Thesis figures/tables are exported per **VIEW** into
`results/<view>/figures|tables/<family>/` — figures `.png`, tables `.md` + `.xlsx`.

**Organization = topic notebooks ↔ numbered result families, 1:1 (2026-07-02 reorg).** Every notebook
is a topic; its NUMBER equals its results-family number, so any artifact under `results/<view>/`
traces straight back to the notebook that produces it (browse the results, open the matching notebook
to edit / dig deeper):

| Notebook | Family (figures + tables) | Contents |
|---|---|---|
| `1_Outcomes.ipynb` | `1_outcomes/` | all-metric trajectory grid · per-metric learning-curve catalog (`trajectories/`, peaks auto-flagged) · effect forest · per-model bars · scorecard |
| `2_Heterogeneity.ipynb` | `2_heterogeneity/` | every metric split by persona trait (`cooperation_level/`, `problem/` subfolders) + final-iteration endpoint bars |
| `3_Mechanism.ipynb` | `3_mechanism/` | behaviour drift (all 7 MITI behaviours) + merged behaviour table · subscales · factor structure · reward-hack panel · question/over-praise cross-checks · **MICI per-behaviour detail (§4d)** · **PCT patient-detail (§4e)** · session shape · transcripts |
| `4_Training_and_Reliability.ipynb` | `4_training/` | TB curves · candidate reward + advantage · degeneration · reward-faithfulness (reliability curve, proxy-vs-eval, PTO margin-by-depth) |
| `5_Preference.ipynb` | `5_preference/` | PTO Mass-Mean-Probe (word ranking/drift, direction drift, learn/unlearn, MI concepts, K0-vs-K5) |
| `6_Stats.ipynb` | `6_stats/` | all heavy tables: merged main_results (`target` col) · Friedman · merged vs-base/method/K paired · all-metric slopes · PCA · GRPO iter-9 anomaly check |

Every section is tagged **`[EVAL]`** (full-conversation oracle scores — the held-out outcome) or
**`[TRAINING]`** (partial-branch rewards / preference pairs — what the policy is updated on). Every
notebook ends with `build_index()` so the per-view `INDEX.md` is complete whatever runs last.

## The VIEW knob (the one control)
Cell 1 of every notebook starts with:
```python
VIEW = os.environ.get("EDA_VIEW", "L0")        # "all" | "L0" | "L5"
cfg  = eda_analysis.EdaConfig(view=VIEW, export_group="...")
S    = eda_analysis.notebook_setup(cfg)
```
`view` sets **both** the arm filter **and** the results root:

| `view` | arms kept | writes to |
|---|---|---|
| `all` | every arm (PTO/GRPO × LA0/LA5) | `results/all/…` |
| `L0`  | K=0 arms (`PTO_LA0`, `GRPO_LA0`) | `results/L0/…` |
| `L5`  | K=5 arms (`PTO_LA5`, `GRPO_LA5`, thin) | `results/L5/…` |

So `results/` holds three parallel trees. Edit the `VIEW` default for interactive use, or set the
`EDA_VIEW` env var. An explicit `ks=[...]` overrides the view's arm filter (the view is a convenience
default). Each view root also has a hand-authored **`SUMMARY.md`** (the written analysis) and an
auto-generated **`INDEX.md`** (the artifact map).

### Regenerate every view
```
python render_views.py            # every view × 6 notebooks via nbconvert
python render_views.py L0         # just the L0 view
python render_views.py L5 --nb 3  # one view, one notebook (--nb takes LIST indices 0..5: 0 = 1_Outcomes)
```
`render_views.py` sets `EDA_VIEW` per run and executes each notebook to a throwaway `--output-dir`
(so the committed notebooks' outputs aren't churned — only the `results/` tree is the deliverable).
**Committed notebooks are kept output-clean** by `strip_notebook_outputs.py` (zero-dependency): run
it in place (`python strip_notebook_outputs.py`), as a regression guard (`--check`), or wire it as a
git clean filter (see the `.gitattributes` note) so `git add` strips outputs automatically while the
working tree keeps them for viewing.
Needs the venv kernel `thesis-venv313` (register once:
`.venv\Scripts\python.exe -m ipykernel install --user --name thesis-venv313`). The hand-authored
`SUMMARY.md` files are never touched.

## Configuring a notebook (`EdaConfig`)
`EdaConfig` is the single flat-globals control surface (`eda_analysis/config.py`). `EdaConfig()` =
the `all` view / all present metrics. Knobs beyond `view`:
- **Arms:** `methods` (`["PTO"]`), `ks` (overrides the view's K filter), `modes`, `arm_labels`,
  `include_archived`.
- **Metrics:** `metrics` (explicit ordered subset), `add_derived_mitiprof` (free R:Q/%CR/%MICO),
  `warmth_only`.
- **Selection / focus:** `selection="all"|"best"`; **`focus_arms`** (default arm subset for
  overlay/trajectory figures) + `focus_metric`.
- **Plot scales:** `context`, `font_scale`, `dpi`, `savefig_dpi`, `panel`, `ncols`, `score_ylim`,
  `share_y`, `palette_overrides` (all default = inherit the publication style).
- **Exports:** `export_group` (→ `results/<view>/<figures|tables>/<family>/`; set it to the
  notebook's family, e.g. `"1_outcomes"`), `fig_formats` (**default `("png",)`**; `("png","pdf")` to
  also emit vector), `table_formats` (**default `("md","xlsx")`** — readable Markdown + a per-family
  Excel workbook, one sheet per table). A per-call `group=` on `save_fig`/`save_table` overrides the
  family for one save and supports **nested subpaths** (`group="1_outcomes/trajectories"`,
  `group="2_heterogeneity/problem"`).
- **Cache:** `cache` (**default `True`**) parquet-memoizes the slow disk reads — `scores_long`
  (~60 s cold → ~0.3 s) and the `behavior_by_iter` family (~30 s → ~0.3 s) — to `eda/.eda_cache/`
  (gitignored). Content-keyed on the input CSVs' `(name, size, mtime)`, so a re-score / re-gen
  auto-invalidates; it can never serve stale numbers. Bypass with `EdaConfig(cache=False)`, the
  `EDA_NO_CACHE=1` env var, or `eda_analysis.reset_cache()`. Different arm-subsets (L0 vs L5) cache
  independently, so `render_views.py` builds each frame once then reads it across notebooks.

**Per-figure control.** Trajectory plots take `arms=`/`iters=`/`metric=`; use
`eda_analysis.select_scores(S.SCORES, arms=[...], iters=[...], metrics=[...])` to slice any figure.
`plots.single_metric_trajectory(..., mark_peaks=True)` auto-flags peak-then-regression arms
(`oracle_noise=None` suppresses the Q1Q2-only noise band); `plots.heterogeneity_grid(S.SCORES, char,
arms=[...])` is one figure (panel per arm); `plots.overlay_trajectory` remains as an interactive
utility (no longer exported).

`notebook_setup(cfg)` resolves the view (→ arm filter + results root), applies the style + scales,
**filters + discovers** the arms, builds `scores_long` (with the derived ratios) + palette + present
metrics, sets the export group, and writes a **provenance banner**
(`results/<view>/figures/<group>/_provenance.md`). `S.CFG` carries the config; `S.VIEW` is the
resolved view; `S.RESULTS_DIR` is the view dir; `S.ARMS / S.SCORES / S.PALETTE / S.METRICS /
S.ORACLE_NOISE` as before. Override on the fly: `notebook_setup(cfg, selection="best")`.

## Run order
1. **`Run_Eval.ipynb`** — async oracle scoring → `data/<method>/eval_scores/`. Registry-driven: add a
   `oracle_scoring/config.py::EXPERIMENTS` entry per new run. Resume-safe. Score **PCT** + **MICI** with
   `QUESTIONNAIRE_FILTER=["PCT","MICI"]`.
2. **`1_Outcomes.ipynb`** → **`6_Stats.ipynb`** in any order (the notebook↔family table above says
   what lives where). Every notebook auto-discovers arms from disk via `eda_analysis.discover_arms()`
   (no path literals) and ends with `build_index()` → `results/<view>/INDEX.md`. Notebooks run with
   the venv kernel `thesis-venv313`, cwd = `eda/`.

## Package (`eda_analysis/`) — 9 analysis modules (+ `plotting_style` helpers, `_selfcheck` guard)
Plumbing was consolidated (2026-06-18) from 14 modules to 9; the analysis/topic files stay separate.
`figures`/`plots` still resolve as aliases of `plotting`; the data-module aliases were retired
(2026-07-08). `plotting` was split (2026-07-08) into the named figures + a `plotting_style` helper
sibling (re-imported into `plotting`, so the public surface is unchanged).

- **`config`** — `EdaConfig` (the single control surface, incl. `view` + PNG/xlsx defaults) +
  `notebook_setup(cfg)` → `Setup` (incl. `S.VIEW`, `S.CFG`). *(absorbed the old `notebook.py`.)*
- **`data`** — the load+shape layer: arm **discovery** (`discover_arms`/`filter_arms`/`Arm`), TRUE-
  **persona** recovery (`attach_personas`/`canonical_personas` — replays the per-iter shuffle), the
  **`scores_long`** backbone (`load_scores_long`/`load_subscales`/`to_wide`/`collapse_base`/
  `add_derived_mitiprof_rows`/`select_scores`), and **selection** (`all_models`/`best_per_experiment`).
  *(merged `discovery`+`personas`+`scores`+`select` into one module; the old submodule aliases have
  been retired — use the canonical `eda_analysis.data.*` / top-level re-exports.)*
- **`plotting_style`** — the style/scaffold helpers (Okabe-Ito palette [PTO cool / GRPO warm / Base
  grey], `grid`, `set_style(cfg)`, `clean_label`, `apply_score_axis`, `model_order`, `relabel_*`,
  `add_base_line`, `figure_legend_from`). Re-imported into `plotting`, so `figures.set_style(...)`
  etc. still resolve.
- **`plotting`** — the named figures (`effect_forest`, `reliability_curve`, `subscale_trajectory_grid`,
  `overlay_trajectory`, `heterogeneity_grid`, `factor_loadings_bars`, `leaderboard_scorecard`,
  diverging `rubric_correlation_heatmap`, …), calling the `plotting_style` helpers. *(aliased back as
  `eda_analysis.figures`/`plots`.)*
- **`stats`** — persona-paired Wilcoxon/dz/bootstrap + Friedman/Kendall-W + `main_results_table` +
  `paired_method_comparison` (PTO vs GRPO) + `paired_k_comparison` (K0 vs K5) +
  `rank_agreement_by_nturns` (reward reliability) + `rubric_pca`/`rubric_factor_space` +
  `filter_thin_arms`.
- **`behavior`** — MITI counts + over-praise cross-check + structural text metrics.
- **`training`** — `generations.jsonl` proxy reward + degeneracy scan + pref pairs +
  `advantage_signal_by_iter`/`reward_distribution_frame` + `load_branch_reliability` +
  `tb_curves`/`parse_run_tb` (self-contained TensorBoard parse, no torch/trl).
- **`pref`** — PTO Mass-Mean-Probe (word ranking/drift, `preference_direction_drift`,
  `learn_unlearn_words`, MI-concept projection).
- **`exports`** — `save_fig` (PNG) / `save_table` (MD+XLSX) → `results/<view>/<group>/`;
  `set_view` / `set_export_group` / `set_formats` / `save_provenance` / `build_index` /
  `reset_results` (clears the active view's figures/tables; **preserves `SUMMARY.md`**).
- **`__init__`** — workspace-root resolution, `QUESTIONNAIRES`/`WARMTH_RUBRICS`/`ORTHOGONAL_METRICS`/
  `display_label`, public re-exports, and the backward-compat submodule aliases.

Two packages, by purpose: **`eda_analysis/`** = the analysis layer (notebooks `1`–`6`, disk-discovery,
no registry) and **`oracle_scoring/`** = the legacy package kept ONLY to power `Run_Eval.ipynb`'s
scoring (its `EXPERIMENTS` registry is Exp3-only). ⚠ the old `oracle_scoring` persona join is wrong for
Exp3 (per-iter shuffle) — use `eda_analysis` (`data.attach_personas`).

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → add an `EXPERIMENTS` entry → `Run_Eval` →
the notebooks pick it up automatically (re-run `python render_views.py`).

## Latest results (snapshot 2026-06-18; MI-SAT re-scored 2026-07-07)
> MI-SAT was re-scored 2026-07-07 under corrected goal-agnostic wording (was hard-coded to "diabetes"); its
> means rose uniformly ~+0.14 but no headline below changes (it is a redundant warmth rubric).

Scored: **PTO LA0** 0–10, **GRPO LA0** 0–10 (finished), **PTO LA5** 0–4, GRPO LA5 base — all on the full
battery incl. the orthogonal axes (PCT, MICI, R:Q/%CR/%MICO). Headlines: large warmth gains vs base
(PTO LA0 Q1+Q2 4.26; GRPO LA0 peaks 4.08 @ iter 8 then **regresses to 3.75 @ iter 10**); **PTO is ahead
at the matched 10-iter endpoint** (4.26 vs 3.75, dz +0.73) — GRPO is competitive only up to its peak,
then overshoots into sycophancy. The orthogonal axes show the warmth gains come *with* a ~2.3× rise in
**MI-inconsistent** behaviour and **affirmation drift in both methods** (PC1 drops 91%→≈55% once the new
axes are included). Per-view narratives: `results/<view>/SUMMARY.md` (L0 is the primary read). Full
numbers in `6_Stats` and the `project-pto-la0-eval-results` memory.

---

## Improvement roadmap — making the EDA better & more readable
Prioritized; none are blocking. Ordered by value-for-effort.

**Landed (2026-06-09 → 2026-06-18).** The `eda_analysis/` package + disk-discovery + true-persona
recovery + both stat batteries; hybrid plotting + `notebook_setup()`; method-symmetry; the by-purpose
notebooks; concise evergreen markdown with the `[EVAL]`/`[TRAINING]` tag; pooled descriptive **Base**;
subscale **trajectories** + **`effect_forest`** + reliability curve + TB curves + richer preference
latent space; Okabe-Ito **colourblind palette**, base lines, **no violins**; heavy tables in `5`,
thin arms filtered; the orthogonal eval axes (PCT/MICI/R:Q/%CR/%MICO); then (2026-06-18) the
**VIEW system** (`all`/`L0`/`L5` result trees + `render_views.py`), the **9-module** package
consolidation, and per-view **`SUMMARY.md`** narratives; then (2026-07-02) the **reorg-by-topic pass**
— topic notebooks ↔ numbered result families 1:1, per-metric trajectory + heterogeneity catalogs,
dedup of 4 duplicate figures, merged stats tables, readable labels, per-call `group=` exports, the
GRPO iter-9 anomaly check, and the walk-based `build_index()` in every notebook; then (2026-07-07) the
**#7 general-review batch** — MITI behaviour counts **per therapist turn** (`B*_per_turn`, drift figure);
an honest **unfiltered PTO `group_range`** beside GRPO's in the advantage signal (keyed on
`(conversation_id, branch_id)` — PTO's `branch_id` is trunk depth and collides across conversations);
confirmatory-vs-exploratory split (`6_Stats` §0); reward=outcome + shared-oracle confounds + PCT-loads-
WITH-warmth reframes (`3_Mechanism` §3/§4); K-descriptive banners; `LIMITATIONS.md`; palette-keyed colors;
dead `rank_table` removed; `render_views` `DEFAULT_VIEWS`; then (2026-07-08) the **package-refinement
batch** — **parquet caching** (item 5: `data.load_cached`, content-keyed on the input CSVs, on by
default → `scores_long`/`behavior_by_iter` ~60/30 s → ~0.3 s, 176×/76×); the **`_selfcheck` regression
guard** (item 8: `python -m eda_analysis._selfcheck`); `plotting` split into `plotting` + a
`plotting_style` helper sibling; the data-module submodule aliases retired; and all committed
notebooks stripped output-clean (`strip_notebook_outputs.py` + `filter=nbstrip`). **Remaining roadmap:**

**Reproducibility / speed:**
7. **Discovery should skip empty `model_iter` dirs**; `Run_Eval`'s registry could be auto-generated
   from `discover_arms()` to remove the last hand-maintained list.

**Recommended next:** 7 (auto-generate `Run_Eval`'s registry from `discover_arms()` — removes the last
hand-maintained list, the main reason `oracle_scoring/` stays a separate package).
