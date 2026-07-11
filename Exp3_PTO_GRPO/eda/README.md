# Exp3 EDA — guide

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

**Per-figure control.** Trajectory plots take `arms=`/`iters=`/`metric=`; slice `S.SCORES` with
plain pandas (e.g. `S.SCORES[S.SCORES.arm.isin([...])]`) to point any figure at a subset.
`plots.single_metric_trajectory(..., mark_peaks=True)` auto-flags peak-then-regression arms
(`oracle_noise=None` suppresses the Q1Q2-only noise band); `plots.heterogeneity_grid(S.SCORES, char,
arms=[...])` is one figure (panel per arm).

`notebook_setup(cfg)` resolves the view (→ arm filter + results root), applies the style + scales,
**filters + discovers** the arms, builds `scores_long` (with the derived ratios) + palette + present
metrics, sets the export group, and writes a **provenance banner**
(`results/<view>/figures/<group>/_provenance.md`). `S.CFG` carries the config; `S.VIEW` is the
resolved view; `S.RESULTS_DIR` is the view dir; `S.ARMS / S.SCORES / S.PALETTE / S.METRICS /
S.ORACLE_NOISE` as before. Override on the fly: `notebook_setup(cfg, selection="best")`.

## Run order
1. **`Run_Eval.ipynb`** — async oracle scoring → `data/<method>/eval_scores/`. The
   `oracle_scoring/config.py::EXPERIMENTS` registry is **auto-generated from `discover_arms()`**
   (2026-07-11, roadmap #7) — a new run is scoreable as soon as its conversations land; no registry
   edit. Resume-safe. Score **PCT** + **MICI** with `QUESTIONNAIRE_FILTER=["PCT","MICI"]`.
2. **`1_Outcomes.ipynb`** → **`6_Stats.ipynb`** in any order (the notebook↔family table above says
   what lives where). Every notebook auto-discovers arms from disk via `eda_analysis.discover_arms()`
   (no path literals) and ends with `build_index()` → `results/<view>/INDEX.md`. Notebooks run with
   the venv kernel `thesis-venv313`, cwd = `eda/`.

## Package (`eda_analysis/`) — analysis modules on a `constants` leaf (+ `plotting_style` helpers, `_selfcheck` guard)
Plumbing was consolidated (2026-06-18) from 14 modules to 9; the analysis/topic files stay separate.
`figures`/`plots` still resolve as aliases of `plotting`; the data-module aliases were retired
(2026-07-08). `plotting` was split (2026-07-08) into the named figures + a `plotting_style` helper
sibling (re-imported into `plotting`, so the public surface is unchanged). The Layer-0 core was
extracted (2026-07-08) into a **`constants` leaf**, breaking the old `__init__`↔submodule import
cycle — submodule imports are now plain top-level `from .constants import ...` (the ~20 deferred
in-function imports are gone; only genuinely cross-module ones remain deferred).

- **`constants`** — the LEAF (imports nothing from the package): workspace-root resolution +
  `sys.path` bootstrap, `QUESTIONNAIRES`/`QUESTIONNAIRE_ORDER`/`WARMTH_RUBRICS`/
  `ORTHOGONAL_METRICS`/`LOWER_IS_BETTER`, `DISPLAY_NAMES`/`ARM_LABELS`,
  `display_label`/`short_label`/`arm_label`, the shared `RE_AFFIRM` cue.
- **`config`** — `EdaConfig` (the single control surface, incl. `view` + PNG/xlsx defaults) +
  `notebook_setup(cfg)` → `Setup` (incl. `S.VIEW`, `S.CFG`). *(absorbed the old `notebook.py`.)*
- **`data`** — the load+shape layer: arm **discovery** (`discover_arms`/`filter_arms`/`Arm`), TRUE-
  **persona** recovery (`attach_personas`/`canonical_personas` — replays the per-iter shuffle), the
  **`scores_long`** backbone (`load_scores_long`/`load_subscales`/`to_wide`/`collapse_base`/
  `add_derived_mitiprof_rows`), and **selection** (`all_models`/`best_per_experiment`).
  *(merged `discovery`+`personas`+`scores`+`select` into one module; the old submodule aliases have
  been retired — use the canonical `eda_analysis.data.*` / top-level re-exports.)*
- **`plotting_style`** — the style/scaffold helpers (Okabe-Ito palette [PTO cool / GRPO warm / Base
  grey], `grid`, `set_style(cfg)`, `clean_label`, `apply_score_axis`, `model_order`, `relabel_*`,
  `add_base_line`, `figure_legend_from`). Re-imported into `plotting`, so `figures.set_style(...)`
  etc. still resolve.
- **`plotting`** — the named figures (`effect_forest`, `reliability_curve`, `subscale_trajectory_grid`,
  `trajectory_grid`, `heterogeneity_grid`, `factor_loadings_bars`, `leaderboard_scorecard`,
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
- **`_selfcheck`** — the guard: package invariants + known headline means + cache round-trip.
  Run `python -m eda_analysis._selfcheck` after any EDA change.
- **`__init__`** — thin re-export hub: re-exports the `constants` leaf + every submodule's public
  names, and the `figures`/`plots` → `plotting` aliases. No definitions of its own.

Two packages, by purpose: **`eda_analysis/`** = the analysis layer (notebooks `1`–`6`, disk-discovery,
no registry) and **`oracle_scoring/`** = the legacy package, **pruned (2026-07-08) to ONLY the
`Run_Eval.ipynb` scoring path** (config `EXPERIMENTS` registry — since 2026-07-11 auto-generated from
`eda_analysis.data.discover_arms()` — + eval settings, conversation loading, the async oracle
pipeline). Its old analysis/persona-join code was removed — persona recovery lives in
`eda_analysis` (`data.attach_personas`, which replays the per-iter shuffle correctly).

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → `Run_Eval` (the registry auto-discovers
the run) → the notebooks pick it up automatically (re-run `python render_views.py`).

## Results
Headline: **PTO wins at the matched 10-iter endpoint (Q1+Q2 4.26 vs GRPO 3.75)** — GRPO peaks @ iter 8
(4.08) then regresses into sycophancy; the orthogonal axes (PCT/MICI/R:Q/%CR/%MICO) show the warmth
gains come *with* a rise in MI-inconsistency in both methods, ~2.3× PTO / ~4× GRPO at the endpoint
(PC1 drops ≈91%→≈55%). The full
narrative + numbers live in **`results/<view>/SUMMARY.md`** (L0 is the primary read), the Exp3
[CLAUDE.md](../CLAUDE.md) "Eval results so far" section, and the `project-pto-la0-eval-results` memory —
not duplicated here so they can't drift.

## Roadmap
Dated pass history (2026-06-09 → 2026-07-11) is in [history/CHANGELOG.md](../history/CHANGELOG.md).
The backlog is clear — the last item (**auto-generate `Run_Eval`'s `EXPERIMENTS` registry from
`discover_arms()`**, incl. skipping empty `model_iter` dirs) landed 2026-07-11. Optional future step:
fold scoring into `eda_analysis/` entirely (the registry was the main reason `oracle_scoring/` stayed
a separate package).
