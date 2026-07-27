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
| `5_Training_and_Reliability.ipynb` | `5_training/` | TB curves · candidate reward + advantage · degeneration · reward-faithfulness (reliability curve, proxy-vs-eval, PTO margin-by-depth) · **judge reliability §7** (oracle ICC + second-judge agreement + contrast preservation) · **multi-judge §8** (variance decomposition, gain retention, all-pairs contrasts, concordance-vs-effect-size) — both read from `data/eval_scores_by_judge/` |
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
| `all` | every arm (PTO/GRPO × LA0/LA5) | `results/all/…` — **RETIRED as a deliverable (2026-07-27)**: gitignored scratch, still renderable |
| `L0`  | K=0 arms (`PTO_LA0`, `GRPO_LA0`) | `results/L0/…` |
| `L5`  | K=5 arms (`PTO_LA5`, `GRPO_LA5`, thin) | `results/L5/…` |

So `results/` holds **two** tracked trees, `L0` and `L5`.

> **`all` was retired on 2026-07-27.** It existed to compare L0 against L5, but with the K=5 arms
> thin and paused that comparison isn't live, and a future look-ahead comparison will get its own
> dedicated view rather than a pooled one. `all` still *renders*
> (`python tools/render_views.py all`) as ad-hoc scratch, but `results/all/` is gitignored and is
> no longer a tracked deliverable. Its hand-authored narrative is recoverable with
> `git show HEAD:Exp3_PTO_GRPO/eda/results/all/SUMMARY.md`.

Edit the `VIEW` default for interactive use, or set the `EDA_VIEW` env var. An explicit `ks=[...]` overrides the view's arm filter (the view is a convenience
default). Each view root also has a hand-authored **`SUMMARY.md`** (the written analysis) and an
auto-generated **`INDEX.md`** (the artifact map).

### Regenerate every view
```
python tools/render_views.py            # the tracked views (L0 + L5) × 7 notebooks
python tools/render_views.py L0         # just the L0 view
python tools/render_views.py L5 --nb 3  # one view, one notebook (--nb takes the notebook/family NUMBER: 3 = 3_Validity_and_Hacking)
```
`tools/render_views.py` sets `EDA_VIEW` per run and executes each notebook to a throwaway `--output-dir`
(so the committed notebooks' outputs aren't churned — only the `results/` tree is the deliverable).
**Committed notebooks are kept output-clean** by `strip_notebook_outputs.py` (zero-dependency): run
it in place (`python tools/strip_notebook_outputs.py`), as a regression guard (`--check`), or wire it as a
git clean filter (see the `.gitattributes` note) so `git add` strips outputs automatically while the
working tree keeps them for viewing.
Needs the venv kernel `thesis-venv313` (register once:
`.venv\Scripts\python.exe -m ipykernel install --user --name thesis-venv313`). The hand-authored
`SUMMARY.md` files are never touched.

## Configuring a notebook (`EdaConfig`)
`EdaConfig` is the single flat-globals control surface (`eda_analysis/config.py`). `EdaConfig()` =
the `all` arm filter (every arm) / all present metrics; notebooks set `view=VIEW` explicitly. Knobs beyond `view`:
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
  independently, so `tools/render_views.py` builds each frame once then reads it across notebooks.

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
   NOT part of `tools/render_views.py`. Backing module: `eda_analysis/scoring/judge.py`. Addresses
   `LIMITATIONS.md` §1–§2 (measured 2026-07-26 with Claude Haiku 4.5 as the second judge).
   Its **§3 promotes the second judge to a full sweep** — all 29 model states × all 8 rubrics —
   through `scoring/judge_plan.py` (free pre-flight: rubric-parity gate, coverage plan, cost model)
   and `scoring/judge_batch.py` (Anthropic Message Batches, 50% off, submit/poll/collect).
   **Presentation is split off deliberately:** this notebook only *scores*;
   `5_Training_and_Reliability` §7–§8 *read* the same tree via `eda_analysis/reliability.py` (no API
   calls) and export the tracked tables + figures — same paid-pipeline/free-notebook split as
   `Run_Eval` → notebooks 1–7, which keeps `tools/render_views.py` fully reproducible without spending.

   > **Run the parity gate before any second-judge spend.** Claude's `json_schema` rejects
   > `minimum`/`maxItems`/…, so those are folded into `description`; `check_rubric_parity()`
   > verifies each dropped constraint was restated and that the encodings are otherwise identical.
   > It runs automatically in `python -m eda_analysis._selfcheck`.
   >
   > **Prompt caching, measured not assumed:** only **Q1 (~1.1k tok) and Q2 (~2.2k tok)** clear
   > OpenAI's 1,024-token cacheable-prefix minimum. WAI-SR/CSQ-8/MI-SAT are rubric-first but too
   > short (403–507 tok); MITI/PCT/MICI interpolate a *per-conversation* utterance count into the
   > instructions **ahead of** the rubric, truncating their prefix to 138–206 tok. Haiku 4.5's
   > minimum is 4,096, so a Claude judge never caches — confirmed empirically
   > (`cached_input_tokens = 0` on every probe call). **This is documented, not fixed** — those
   > counts are the rate metrics' denominators, and changing the prompt would break comparability
   > with every conversation already scored, for a discount that still would not materialize.
   >
   > **Two quantities that are easy to confuse when costing a sweep** (`prefix_report` returns
   > both): `prefix_tokens_approx` is the *cacheable* prefix and drives the **discount**;
   > `fixed_prompt_tokens_approx` is the whole instruction+rubric block and drives the **input
   > cost**. They diverge sharply on MITI/PCT/MICI (138–206 vs ~1,000 tok) because of the
   > utterance-count invalidator above, so costing off the cacheable prefix underestimates input by
   > ~25%. Likewise `judge_batch.probe_usage` samples at quantile **midpoints**: spreading
   > endpoint-to-endpoint puts the shortest *and* longest transcript in the sample, which at
   > `n=2` is `(min + max) / 2` — 2.1× the true mean on this right-skewed data.

## Judge dimension — running the EDA under a second grader

`VIEW` and `JUDGE` are **orthogonal knobs**: `VIEW` filters which *arms* are analysed, `JUDGE`
selects which *grader's scores* are read.

| `JUDGE` | Reads | Writes |
|---|---|---|
| `""` (default) | `data/<method>_Exp3/eval_scores/` — the primary oracle, the numbers the thesis reports | `results/<view>/figures/<family>/` (**unchanged**) |
| `anthropic_claude-haiku-4-5` | `data/eval_scores_by_judge/judge=<tag>/rep=<r>/` | `results/<view>/figures/<family>/<tag>/` |

> **Why the two score trees differ in location.** The primary oracle's scores sit *inside each
> method's run directory* (`data/{grpo,pto}_Exp3/eval_scores/`) because those are **Google Drive
> symlinks** — backed up and reachable from Colab. Other judges score into a **local**
> `data/eval_scores_by_judge/` tree. The split is about **storage**, not status: both use the same
> `metric=/oracle=/<Model>/<id>.csv` layout, and `Arm.eval_dir()` hides the difference so no
> analysis ever needs to know. (Renamed from `judge_check/` on 2026-07-27 — that name framed a
> second grader as a validation aside, but a judge here has now scored the same full grid as the
> primary.)

```
results/L0/figures/1_outcomes/
├── outcomes_by_model.png                    ← primary oracle (flat)
├── effect_forest.png
└── anthropic_claude-haiku-4-5/              ← same family, second judge
    ├── outcomes_by_model.png
    └── effect_forest.png
```

The **judge is the deepest level**, so a family's output from every grader sits side by side and
compares in one glance. The primary stays **flat** so every existing artifact, `SUMMARY.md` link
and cross-reference keeps working — adding a grader must not move a path the thesis already cites.
This works at all because `scoring/judge*.py` writes its CSVs in the *identical*
`metric=/oracle=/<Model>/<file_index>.csv` layout with identical column names, so `Arm.eval_dir()`
only has to swap a root.

`reset_results()` is **judge-scoped**: with a judge active it clears only `<family>/<judge>/`, never
the family folder itself. Otherwise a routine `--judge` regenerate would delete the primary tree.

```bash
python tools/render_views.py                                       # primary oracle (unchanged)
python tools/render_views.py --judge anthropic_claude-haiku-4-5    # the same EDA, Claude's scores
python tools/render_views.py --judge anthropic_claude-haiku-4-5 --nb 1 7   # just outcomes + stats
```

> ⚠ **Only eval-score-derived notebooks are judge-swappable.** Notebooks **1, 2, 3, 4, 7** read
> `scores_long` / the oracle behaviour counts and re-grade cleanly. Notebooks **5 and 6** are
> **training-side** — candidate rewards in `generations.jsonl`, PTO preference pairs, TensorBoard
> curves — all produced by the *training* oracle during the run and impossible to re-grade after
> the fact. Re-rendering them under a second judge would emit byte-identical figures into that
> judge's folder, implying a measurement that never happened, so both the notebook (a `SystemExit`
> guard in cell 1) and `render_views.py --judge` (skips them, with a printed reason) refuse.
> The genuinely multi-judge work lives in `5_Training_and_Reliability` §7–§8, which reads
> `data/judge_check/` directly and puts **both** graders in the same figure.

**Coverage is checked, not assumed.** `notebook_setup` warns loudly when a judge has not scored
every conversation of every arm — a partially-landed sweep otherwise yields arm means that look
like the primary's but rest on smaller, unequal samples, and persona-paired contrasts between two
such arms overlap on only a fraction of personas.

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
  halo cluster — historical code name)/`EXTRA_METRICS`/`LOWER_IS_BETTER`,
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
  figures — + the Q2 specializations) · `training` (reward distribution, advantage side-by-side) ·
  `reliability` (`oracle_repeatability_bars`, `judge_agreement_scatter`, `judge_contrast_bars`).
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
- **`reliability`** — MEASUREMENT-validity tables from the `data/judge_check/` re-scoring tree.
  Disk-only — the paid scoring lives in `scoring/judge*.py`; this is the free read side.
  - *§7 (single-judge validity):* `repeatability` (ICC(2,1) + mean |Δ|), `agreement` (second judge
    vs primary + attenuation ceiling), `contrasts` (does each endpoint contrast keep its sign?),
    `arm_means_by_judge`, `summary_line`.
  - *§8 (multi-judge):* `variance_components_conversation` / `variance_components_arm` (two-way
    random-effects decomposition → arm vs judge-level vs **arm×judge**, plus `dependability_k1/k2`),
    `gain_retention` (the reward-hacking transfer test, persona-bootstrap CI), `all_pairs_contrasts`
    (every model pair, paired on the recovered `persona_id` — see `attach_persona`, since
    `file_index` is reshuffled per iteration), `concordance_by_effect_size`,
    `multi_judge_summary_line`. **Never averages raw scores across judges** — the primary judge was
    the training reward and the second is held out, so this is train-vs-test, not two raters.
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
  ICC(2,1), agreement + contrast-preservation stats) · `judge_plan` (**free pre-flight** for a full
  sweep: `check_rubric_parity` — the gate that both judges receive the same rubric —
  `prefix_report` (which rubrics actually prompt-cache), `plan_sweep` (coverage-aware call count),
  `estimate_cost` / `sweep_report`) · `judge_batch` (the **Anthropic Message Batches** path, 50% off:
  `submit_sweep` → `poll_batches` → `collect_batches`, three phases with disk-persisted manifests so
  a fresh kernel can collect; plus `probe_usage` for a measured token profile).
- **`_selfcheck`** — the guard: package invariants + the scoring surface + known headline means +
  cache round-trip. Run `python -m eda_analysis._selfcheck` after any EDA change.
- **`__init__`** — thin re-export hub: re-exports the `constants` leaf + every analysis submodule's
  public names, and the `figures`/`plots` → `plotting` aliases. No definitions of its own.

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → `Run_Eval` (the registry auto-discovers
the run) → the notebooks pick it up automatically (re-run `python tools/render_views.py`).

## Results
Not duplicated here (so they can't drift). The full narrative + numbers live in
**`results/<view>/SUMMARY.md`** (L0 is the primary read); the live status + headline is the root
[CLAUDE.md](../../CLAUDE.md) § "Current status & next step".

## Roadmap
Dated pass history (2026-06-09 → 2026-07-16) is in [history/CHANGELOG.md](../history/CHANGELOG.md);
the backlog is clear (last item — the tier-based 7-family reorg + `0_headline/` + generic
questionnaire item detail + final-vs-best everywhere — landed 2026-07-16).
