# Exp3 EDA — guide + improvement roadmap

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), across
training iterations, under matched look-ahead K and MCL. All data/compute/stats lives in the
`eda_analysis/` package; the recurring figures are named functions in `eda_analysis/plotting.py`
(called once from multiple notebooks), and genuinely one-off exploration stays inline (the **hybrid**
plotting split). Thesis figures/tables are exported per **VIEW** into
`results/<view>/figures|tables/<group>/` — figures `.png`, tables `.md` + `.xlsx`.

The notebooks are **organized by purpose**: one notebook each for eval outcomes + behaviour, the
training signal, reward reliability, the preference latent space, and the detailed stats. Every
section is tagged **`[EVAL]`** (full-conversation oracle scores — the held-out outcome) or
**`[TRAINING]`** (partial-branch rewards / preference pairs — what the policy is updated on).

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
python render_views.py            # 3 views × 6 notebooks (18 runs) via nbconvert
python render_views.py L0         # just the L0 view
python render_views.py L5 --nb 4  # one view, one notebook (e.g. re-render L5 when its data lands)
```
`render_views.py` sets `EDA_VIEW` per run and executes each notebook to a throwaway `--output-dir`
(so the committed notebooks' outputs aren't churned — only the `results/` tree is the deliverable).
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
- **Exports:** `export_group` (→ `results/<view>/<figures|tables>/<group>/`), `fig_formats`
  (**default `("png",)`**; `("png","pdf")` to also emit vector), `table_formats` (**default
  `("md","xlsx")`** — readable Markdown + a per-group Excel workbook, one sheet per table).

**Per-figure control.** Trajectory/headline/contrast plots take `arms=`/`iters=`/`metric=`; use
`eda_analysis.select_scores(S.SCORES, arms=[...], iters=[...], metrics=[...])` to slice any figure.
`plots.overlay_trajectory(S.SCORES, metric, arms=[...])` is the one configurable contrast;
`plots.heterogeneity_grid(S.SCORES, char, arms=[...])` is one figure (panel per arm).

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
2. **`0_Headline.ipynb`** `[EVAL]` (group `headline`) — thin: the 3 canonical thesis figures (best-vs-
   base bars, vs-base **effect forest**, Q1+Q2 curve) + the per-view artifact index.
3. **`1_Eval_and_Behavior.ipynb`** `[EVAL]` (group `eval`) — **eval-results + behaviour.** *Part A:*
   all-rubric + subscale trajectories, the **one configurable contrast** cell (`overlay_trajectory`),
   the **scorecard** (warmth beside PCT / MICI↓ / R:Q / %CR / %MICO), appendix bars. *Part B:* MITI
   behaviour drift, rubric factor structure (diverging corr + **factor-loadings bars**), over-praise
   cross-check, one `heterogeneity_grid` per trait, session-end/length, transcripts.
4. **`2_Training_Diagnostics.ipynb`** `[TRAINING]` (group `training`) — TensorBoard training curves per
   arm (`training.tb_curves`), per-candidate reward distribution, method-native advantage signal,
   degeneration check.
5. **`3_Reward_Reliability.ipynb`** `[TRAINING↔EVAL]` (group `reliability`) — is the partial-conv
   training reward faithful to the full-conv eval? rank-agreement-vs-`n_turns` curve (LA0 vs LA5),
   proxy-vs-eval scatter, PTO margin-by-branch-depth.
6. **`4_Preference_LatentSpace.ipynb`** `[TRAINING]` (group `preference`, PTO only) — Mass-Mean-Probe:
   word ranking + drift, direction drift in 2D, learned/unlearned words, MI-concept drift, K0-vs-K5.
7. **`5_Detailed_Stats.ipynb`** `[EVAL]` (group `stats`) — **all heavy tables** (main results,
   Friedman, paired method/K, per-arm vs-base, slopes, rankings, PCA), thin arms filtered.

Everything **auto-discovers** arms from disk via `eda_analysis.discover_arms()` (no path literals).
`eda_analysis.build_index()` writes the per-view `results/<view>/INDEX.md`. Notebooks run with the
venv kernel `thesis-venv313`, cwd = `eda/`.

## Package (`eda_analysis/`) — 9 modules
Plumbing was consolidated (2026-06-18) from 14 modules to 9; the analysis/topic files stay separate.
The old submodule names still resolve via aliases, so notebook code is unchanged.

- **`config`** — `EdaConfig` (the single control surface, incl. `view` + PNG/xlsx defaults) +
  `notebook_setup(cfg)` → `Setup` (incl. `S.VIEW`, `S.CFG`). *(absorbed the old `notebook.py`.)*
- **`data`** — the load+shape layer: arm **discovery** (`discover_arms`/`filter_arms`/`Arm`), TRUE-
  **persona** recovery (`attach_personas`/`canonical_personas` — replays the per-iter shuffle), the
  **`scores_long`** backbone (`load_scores_long`/`load_subscales`/`to_wide`/`collapse_base`/
  `add_derived_mitiprof_rows`/`select_scores`), and **selection** (`all_models`/`best_per_experiment`).
  *(merged `discovery`+`personas`+`scores`+`select`; aliased back as `eda_analysis.discovery/personas/
  scores/select`.)*
- **`plotting`** — style helpers (Okabe-Ito palette [PTO cool / GRPO warm / Base grey], `grid`,
  `set_style(cfg)`, `clean_label`, `apply_score_axis`, `model_order`) **+** the named figures
  (`effect_forest`, `reliability_curve`, `subscale_trajectory_grid`, `overlay_trajectory`,
  `heterogeneity_grid`, `factor_loadings_bars`, `leaderboard_scorecard`, diverging
  `rubric_correlation_heatmap`, …). *(merged `figures`+`plots`; aliased back as `eda_analysis.figures/
  plots`.)*
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

Two packages, by purpose: **`eda_analysis/`** = the analysis layer (notebooks `0`–`5`, disk-discovery,
no registry) and **`oracle_scoring/`** = the legacy package kept ONLY to power `Run_Eval.ipynb`'s
scoring (its `EXPERIMENTS` registry is Exp3-only). ⚠ the old `oracle_scoring` persona join is wrong for
Exp3 (per-iter shuffle) — use `eda_analysis` (`data.attach_personas`).

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → add an `EXPERIMENTS` entry → `Run_Eval` →
the notebooks pick it up automatically (re-run `python render_views.py`).

## Latest results (snapshot 2026-06-18)
Scored: **PTO LA0** 0–10, **GRPO LA0** 0–10 (finished), **PTO LA5** 0–4, GRPO LA5 base — all on the full
battery incl. the orthogonal axes (PCT, MICI, R:Q/%CR/%MICO). Headlines: large warmth gains vs base
(PTO LA0 Q1+Q2 4.26; GRPO LA0 peaks 4.08 @ iter 8 then **regresses to 3.75 @ iter 10**); **PTO is ahead
at the matched 10-iter endpoint** (4.26 vs 3.75, dz +0.73) — GRPO is competitive only up to its peak,
then overshoots into sycophancy. The orthogonal axes show the warmth gains come *with* a ~2.3× rise in
**MI-inconsistent** behaviour and **affirmation drift in both methods** (PC1 drops 91%→≈55% once the new
axes are included). Per-view narratives: `results/<view>/SUMMARY.md` (L0 is the primary read). Full
numbers in `5_Detailed_Stats` and the `project-pto-la0-eval-results` memory.

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
consolidation, and per-view **`SUMMARY.md`** narratives. **Remaining roadmap:**

**Reproducibility / speed:**
5. **Cache `scores_long` + `behavior_by_iter` to parquet.** `behavior`/`text_metrics` re-read ~2k
   conversation CSVs in every notebook (slow) — and now ×3 views. A `data.load_cached()` writing
   `results/cache/*.parquet` (keyed by arm+iter set) would make notebooks near-instant.
   `notebook_setup()` is the natural home for the toggle. **Biggest speed win now that views multiply
   the reads.**
7. **Discovery should skip empty `model_iter` dirs**; `Run_Eval`'s registry could be auto-generated
   from `discover_arms()` to remove the last hand-maintained list.

**Rigor / correctness polish:**
8. **Self-check script.** Commit the ad-hoc validation as `eda_analysis/_selfcheck.py` (persona recovery
   100%, known means reproduce, probe `wins_correct`>0.5, the view→ks + alias surface) — a fast
   regression test after any change.

**Recommended next:** 5 (parquet caching) then 8 (commit the self-check as a regression guard).
