# Exp3 EDA — guide

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), across
training iterations, under matched look-ahead K and MCL. Everything lives in ONE package,
`eda_analysis/` (since the 2026-07-13 fold of the legacy `oracle_scoring/`): the analysis layer at
the top level + the oracle-scoring layer in the `scoring/` subpackage. The recurring figures are
named functions in the `eda_analysis/plotting/` subpackage (called once from multiple notebooks),
and genuinely one-off exploration stays inline (the **hybrid** plotting split). Thesis
figures/tables are exported per **VIEW** into `results/<view>/figures|tables/<family>/` — figures
`.png`, tables `.md` + `.xlsx`.

**Organization = tier-based drill-down, notebooks ↔ numbered result families 1:1 (2026-07-16 reorg).**
Level 1 = global scores → Level 2 = inside each questionnaire → Level 3+ = cross-cutting analyses.
Every notebook is a topic; its NUMBER equals its results-family number, so any artifact under
`results/<view>/` traces straight back to the notebook that produces it (browse the results, open the
matching notebook to edit / dig deeper). Endpoint artifacts always come as a **final + best pair**
(best = each arm's peak iteration on its own training oracle via `best_per_experiment`; GRPO_LA0→I8):
figures as `<name>_final.png` + `<name>_best.png`, tables merged with a `target` column.

| Notebook | Family (figures + tables) | Contents |
|---|---|---|
| *(re-saves)* | `0_headline/` | the ~7 presentation artifacts, re-saved by notebooks 1–3 via a per-call `group="0_headline"` (main grid, forest final+best, MITI + MICI detail grids, reward-hack panel, scorecard) |
| `1_Outcomes.ipynb` | `1_outcomes/` | **Level 1 — global scores:** all-metric trajectory grid (THE main figure) · per-metric learning-curve catalog (`trajectories/`, peaks auto-flagged) · effect forest final+best · endpoint bars final+best · scorecard final+best |
| `2_Questionnaire_Detail.ipynb` | `2_questionnaires/` | **Level 2 — one uniform detail section per rubric:** Q1/Q2/WAI-SR/CSQ-8/MI-SAT item grids (`<slug>_detail_grid`) + item-delta bars final+best (`<slug>_item_deltas_*`) · Q2 face-content groups · WAI subscales · MITI detail grid (globals + 7 behaviour rates + ratios; zooms in `miti/`) + **official MITI 4.2.1 thresholds** · PCT detail (`pct/`) · MICI detail (`mici/`) |
| `3_Validity_and_Hacking.ipynb` | `3_validity/` | **Level 3 — is it real skill?** rubric factor structure (correlation + PCA loadings) · reward-hack panel · question-rate/over-praise cross-checks · session shape (deterministic text metrics, exported) · transcripts |
| `4_Heterogeneity.ipynb` | `4_heterogeneity/` | every metric split by persona trait (`cooperation_level/`, `problem/` subfolders) + endpoint bars final+best |
| `5_Training_and_Reliability.ipynb` | `5_training/` | TB curves · candidate reward + advantage · degeneration · reward-faithfulness (reliability curve, proxy-vs-eval, PTO margin-by-depth) |
| `6_Preference.ipynb` | `6_preference/` | PTO Mass-Mean-Probe (word ranking/drift, direction drift, learn/unlearn, MI concepts, K0-vs-K5) |
| `7_Stats.ipynb` | `7_stats/` | all heavy tables: merged main_results (`target` col) · Friedman · merged vs-base/method/K paired · **best-vs-best method contrast (`method_paired_best`)** · all-metric slopes · PCA · GRPO iter-9 anomaly check |

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
python render_views.py            # every view × 7 notebooks via nbconvert
python render_views.py L0         # just the L0 view
python render_views.py L5 --nb 3  # one view, one notebook (--nb takes the notebook/family NUMBER: 3 = 3_Validity_and_Hacking)
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
  `group="4_heterogeneity/problem"`).
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
   `eda_analysis/scoring/registry.py::EXPERIMENTS` registry is **auto-generated from
   `discover_arms()`** (2026-07-11, roadmap #7) — a new run is scoreable as soon as its
   conversations land; no registry edit. Resume-safe. Score **PCT** + **MICI** with
   `QUESTIONNAIRE_FILTER=["PCT","MICI"]`.
2. **`1_Outcomes.ipynb`** → **`7_Stats.ipynb`** in any order (the notebook↔family table above says
   what lives where). Every notebook auto-discovers arms from disk via `eda_analysis.discover_arms()`
   (no path literals) and ends with `build_index()` → `results/<view>/INDEX.md`. Notebooks run with
   the venv kernel `thesis-venv313`, cwd = `eda/`.
3. *(optional, costs API budget)* **`Judge_Reliability.ipynb`** — measurement-validity re-scoring on
   a subset: oracle repeatability (ICC, per-rep seeds) + a pluggable **second judge** (Claude via the
   `anthropic` SDK, or another OpenAI model) with the PTO−GRPO contrast-preservation check. Gated
   behind explicit `RUN_*` flags; writes to `data/judge_check/` (never the real `eval_scores/`);
   NOT part of `render_views.py`. Backing module: `eda_analysis/scoring/judge.py`. Addresses
   `LIMITATIONS.md` §1–§2.

## Package (`eda_analysis/`) — analysis modules on a `constants` leaf + `scoring/` and `plotting/` subpackages
Plumbing was consolidated (2026-06-18) from 14 modules to 9; the analysis/topic files stay separate.
`figures`/`plots` still resolve as aliases of `plotting`; the data-module aliases were retired
(2026-07-08). The Layer-0 core was extracted (2026-07-08) into a **`constants` leaf**, breaking the
old `__init__`↔submodule import cycle — submodule imports are plain top-level
`from .constants import ...` (only genuinely cross-module imports remain deferred). On 2026-07-13
the legacy `oracle_scoring/` package was **folded in** as the `scoring/` subpackage (one package,
purpose-named modules — no more duplicate `config.py`/`data.py` names across two packages) and
`plotting.py` (935 lines, 27 figures) was **split** into the `plotting/` subpackage's topic modules
behind an unchanged public surface.

- **`constants`** — the LEAF (imports nothing from the package): workspace-root resolution +
  `sys.path` bootstrap, `QUESTIONNAIRES`/`QUESTIONNAIRE_ORDER`/`WARMTH_RUBRICS` (the global-eval
  halo cluster — historical code name)/`ORTHOGONAL_METRICS`/`LOWER_IS_BETTER`,
  `MITI_THRESHOLDS` (official 4.2.1 fair/good), `Q1_ITEM_SHORT`/`Q2_ITEM_SHORT`/`Q2_ITEM_GROUPS`
  (item labels + face-content groups), `ITEM_QUESTIONNAIRES` (per-item column layout of every
  Likert-item rubric; item text source of truth = `code/questionnaires.py`),
  `DISPLAY_NAMES`/`ARM_LABELS`, `display_label`/`short_label`/`arm_label`/`item_short_label`,
  the shared `RE_AFFIRM` cue.
- **`config`** — `EdaConfig` (the single control surface, incl. `view` + PNG/xlsx defaults) +
  `notebook_setup(cfg)` → `Setup` (incl. `S.VIEW`, `S.CFG`). *(absorbed the old `notebook.py`.)*
- **`data`** — the load+shape layer: arm **discovery** (`discover_arms`/`filter_arms`/`Arm`), TRUE-
  **persona** recovery (`attach_personas`/`canonical_personas` — replays the per-iter shuffle), the
  **`scores_long`** backbone (`load_scores_long`/`load_subscales`/`load_items` [generic per-item
  loader over `ITEM_QUESTIONNAIRES`; `load_q2_items` wraps it]/`to_wide`/`collapse_base`/
  `add_derived_mitiprof_rows`), and **selection** (`all_models`/`best_per_experiment`/
  `final_per_experiment`/`best_iteration_by_arm` — the final-vs-best machinery).
  *(merged `discovery`+`personas`+`scores`+`select` into one module; the old submodule aliases have
  been retired — use the canonical `eda_analysis.data.*` / top-level re-exports.)*
- **`plotting_style`** — the style/scaffold helpers (Okabe-Ito palette [PTO cool / GRPO warm / Base
  grey], `grid`, `set_style(cfg)`, `clean_label`, `apply_score_axis`, `model_order`, `relabel_*`,
  `add_base_line`, `figure_legend_from`). Re-imported into `plotting`, so `figures.set_style(...)`
  etc. still resolve.
- **`plotting/`** (subpackage) — the named figures, split by topic behind a re-exporting `__init__`
  (the public surface is the flat module's): `outcomes` (per-model bars, `effect_forest`,
  `leaderboard_scorecard` — endpoint figures take `title=`/`selection=` for the final-vs-best
  pairs) · `trajectories` (`trajectory_grid`, `single_metric_trajectory`, subscales,
  `reward_hack_panel`) · `heterogeneity` (persona splits; `subgroup_endpoint_bars(iter_by_arm=)`
  for best-iteration bars) · `structure` (`reliability_curve`, proxy-vs-eval, diverging
  `rubric_correlation_heatmap`, `factor_loadings_bars`) · `behavior` (the generic wide-frame
  detail grid reused by MITI/MICI/PCT + session shape, MITI thresholds, cross-checks) ·
  `questionnaires` (`item_trajectory_grid` + `item_delta_bars` — the uniform per-rubric item
  figures — + the Q2 specializations) · `training` (reward distribution, advantage side-by-side).
  *(aliased back as `eda_analysis.figures`/`plots`.)*
- **`stats`** — persona-paired Wilcoxon/dz/bootstrap + Friedman/Kendall-W + `main_results_table` +
  `paired_method_comparison` (PTO vs GRPO) + `paired_best_method_comparison` (best-vs-best model
  selection) + `paired_k_comparison` (K0 vs K5) + `item_endpoint_deltas` (generic "which items
  drive the change"; `q2_item_endpoint_deltas` wraps it) + `rank_agreement_by_nturns` (reward
  reliability) + `rubric_pca`/`rubric_factor_space` + `filter_thin_arms`.
- **`behavior`** — MITI counts (+ per-conv `%MICO`) + over-praise cross-check + structural text
  metrics + `miti_detail_by_iter` (the MITI drill-down frame behind `miti_detail_grid`) +
  `session_shape_by_iter` (exported text metrics) + `miti_proficiency_by_iter` (the
  official-threshold summary scores).
- **`training`** — `generations.jsonl` proxy reward + degeneracy scan + pref pairs +
  `advantage_signal_by_iter`/`reward_distribution_frame` + `load_branch_reliability` +
  `tb_curves`/`parse_run_tb` (self-contained TensorBoard parse, no torch/trl).
- **`pref`** — PTO Mass-Mean-Probe (word ranking/drift, `preference_direction_drift`,
  `learn_unlearn_words`, MI-concept projection).
- **`exports`** — `save_fig` (PNG) / `save_table` (MD+XLSX) → `results/<view>/<group>/`;
  `set_view` / `set_export_group` / `set_formats` / `save_provenance` / `build_index` /
  `reset_results` (clears the active view's figures/tables; **preserves `SUMMARY.md`**).
- **`scoring/`** (subpackage; NOT imported by `__init__` — its registry scans disk, which the
  analysis notebooks never need; the two scoring notebooks import it explicitly) — the
  oracle-scoring layer, folded in from the legacy `oracle_scoring/` package (2026-07-13):
  `registry` (eval settings + the `EXPERIMENTS` registry — auto-generated from
  `eda_analysis.data.discover_arms()` since 2026-07-11 — + the `eval_scores/` layout helpers +
  `ScoringConfig`, formerly `EDAConfig`) · `conversations` (scoring-side conversation loading +
  model-name metadata) · `pipeline` (the async oracle pipeline behind `Run_Eval.ipynb`; formerly
  `eval.py`) · `judge` (the `Judge_Reliability.ipynb` backend — pluggable OpenAI/Anthropic judges,
  ICC(2,1), agreement + contrast-preservation stats).
- **`_selfcheck`** — the guard: package invariants + the scoring surface + known headline means +
  cache round-trip. Run `python -m eda_analysis._selfcheck` after any EDA change.
- **`__init__`** — thin re-export hub: re-exports the `constants` leaf + every analysis submodule's
  public names, and the `figures`/`plots` → `plotting` aliases. No definitions of its own.

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → `Run_Eval` (the registry auto-discovers
the run) → the notebooks pick it up automatically (re-run `python render_views.py`).

## Results
Not duplicated here (so they can't drift). The full narrative + numbers live in
**`results/<view>/SUMMARY.md`** (L0 is the primary read); the live status + headline is the root
[CLAUDE.md](../../CLAUDE.md) § "Current status & next step".

## Roadmap
Dated pass history (2026-06-09 → 2026-07-16) is in [history/CHANGELOG.md](../history/CHANGELOG.md);
the backlog is clear (last item — the tier-based 7-family reorg + `0_headline/` + generic
questionnaire item detail + final-vs-best everywhere — landed 2026-07-16).
