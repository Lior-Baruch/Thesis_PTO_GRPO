# Exp3 EDA — guide + improvement roadmap

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), across
training iterations, under matched look-ahead K and MCL. All data/compute/stats lives in the `exp3/`
package; the recurring figures are named functions in `exp3/plots.py` (called once from multiple
notebooks), and genuinely one-off exploration stays inline (the **hybrid** plotting split). Thesis
figures/tables are exported to `results/` — **one format each**: figures `.pdf`, tables `.md`.

The notebooks are **organized by purpose** (not by research question): one notebook each for eval
outcomes, behaviour, the training signal, reward reliability, the preference latent space, and the
detailed stats. Every section is tagged **`[EVAL]`** (full-conversation oracle scores — the held-out
outcome) or **`[TRAINING]`** (partial-branch rewards / preference pairs — what the policy is updated
on). Markdown is kept concise (≤3-line sections).

## Configuring a notebook (`EdaConfig`)
Cell 1 of every notebook builds one **`exp3.EdaConfig`** from flat globals and passes it to
`notebook_setup` — the single place to control a run (mirrors the trainer notebooks' cell-1 pattern;
reproducible + git-diffable). Defaults reproduce "all arms / all present metrics", so
`exp3.EdaConfig()` is a safe no-op. Knobs:
- **Arms:** `methods` (`["PTO"]`), `ks` (`[0,5]`), `modes`, `arm_labels`, `include_archived`.
- **Metrics:** `metrics` (explicit ordered subset), `add_derived_mitiprof` (free R:Q/%CR/%MICO),
  `warmth_only`.
- **Selection / focus:** `selection="all"|"best"`; **`focus_arms`** (default arm subset for
  overlay/trajectory figures — the lever for "show only these arms instead of looping per arm") +
  `focus_metric`.
- **Plot scales:** `context`, `font_scale`, `dpi`, `savefig_dpi`, `panel`, `ncols`, `score_ylim`,
  `share_y`, `palette_overrides` (all `None`/default = inherit the publication style).
- **Exports:** `export_group` (→ `results/<figures|tables>/<group>/`), `fig_formats` (**default
  `("png",)`** images; `("png","pdf")` to also emit vector), `table_formats` (**default
  `("md","xlsx")`** — readable Markdown + a per-group Excel workbook, one sheet per table).

**Per-figure control.** Trajectory/headline/contrast plots take `arms=`/`iters=`/`metric=`; use
`exp3.select_scores(S.SCORES, arms=[...], iters=[...], metrics=[...])` to slice for any figure.
`plots.overlay_trajectory(S.SCORES, metric, arms=[...])` is the one configurable contrast (replaces
the per-K/per-method loops); `plots.heterogeneity_grid(S.SCORES, char, arms=[...])` is one figure
(panel per arm) instead of the old `char × arm` PNG explosion.

`notebook_setup(cfg)` applies the style + scales, **filters + discovers** the arms, builds
`scores_long` (with the derived ratios), the palette + present-metric list, sets the export group, and
writes a **provenance banner** (`results/<group>/_provenance.md`) so every figure set is traceable.
`S.CFG` carries the config; `S.ARMS / S.SCORES / S.PALETTE / S.METRICS / S.ORACLE_NOISE / S.RESULTS_DIR`
are unchanged. Override on the fly: `notebook_setup(cfg, selection="best")`.

## Run order
1. **`Run_Eval.ipynb`** — async oracle scoring → `data/<method>/eval_scores/`. Registry-driven: add a
   `lib/config.py::EXPERIMENTS` entry per new run (the only place you hand-edit). Resume-safe.
   Score the new **PCT** + **MICI** questionnaires with `QUESTIONNAIRE_FILTER=["PCT","MICI"]`.
2. **`0_Headline.ipynb`** `[EVAL]` (group `headline`) — thin: the 3 canonical thesis figures (best-vs-
   base bars, vs-base **effect forest**, Q1+Q2 curve) + the master artifact index.
3. **`1_Eval_and_Behavior.ipynb`** `[EVAL]` (group `eval`) — **merges eval-results + behaviour.**
   *Part A:* all-rubric + subscale trajectories, the **one configurable contrast** cell
   (`overlay_trajectory`, edit the arm list — replaces the per-K/per-method loops), the **scorecard**
   (warmth beside PCT / MICI↓ / R:Q / %CR / %MICO), appendix bars. *Part B:* MITI behaviour drift +
   text metrics, rubric factor structure (**diverging** corr heatmap + **factor-loadings bars** —
   readable replacement for the old biplot), over-praise cross-check, **one** `heterogeneity_grid`
   figure per trait, session-end/length, transcripts.
4. **`2_Training_Diagnostics.ipynb`** `[TRAINING]` (group `training`) — TensorBoard training curves per
   arm (`training.tb_curves`), per-candidate reward distribution, method-native advantage signal
   (group_std / margin), degeneration check.
5. **`3_Reward_Reliability.ipynb`** `[TRAINING↔EVAL]` (group `reliability`) — is the partial-conv
   training reward faithful to the full-conv eval? rank-agreement-vs-`n_turns` curve (LA0 vs LA5),
   proxy-vs-eval scatter, PTO margin-by-branch-depth.
6. **`4_Preference_LatentSpace.ipynb`** `[TRAINING]` (group `preference`, PTO only) — Mass-Mean-Probe:
   word ranking + drift, **direction drift in 2D**, **learned/unlearned words**, MI-concept drift,
   **K0-vs-K5** contrast. The per-arm probe loops over `focus_arms ∩ PTO`.
7. **`5_Detailed_Stats.ipynb`** `[EVAL]` (group `stats`) — **all the heavy tables** (main results,
   Friedman, paired method/K, per-arm vs-base, slopes, rankings, PCA), thin arms filtered.

Future: an oracle-comparison notebook (research question iii) once non-Q1Q2 oracles are run.

Everything **auto-discovers** arms from disk via `exp3.discover_arms()` (no path literals).
Artifacts land in **per-notebook subfolders** `results/figures/<group>/` + `results/tables/<group>/`
with a per-group `CAPTIONS.md`; `exp3.build_index()` writes the master `results/INDEX.md`. Notebooks
run with the venv kernel `thesis-venv313`, cwd = `eda/`.

## Package (`exp3/`)
`discovery` (arms manifest) · `personas` (TRUE-persona recovery — replays the per-iter shuffle; the old
`lib` join is wrong for Exp3) · `scores` (`scores_long` backbone + `load_subscales` + `to_wide`) ·
`select` (all vs best-per-experiment) · `stats` (omnibus/Mann-Whitney+FDR + persona-paired Wilcoxon/dz/
bootstrap + **Friedman/Kendall-W** + `main_results_table` + `paired_method_comparison` (PTO vs GRPO) +
`paired_k_comparison` (K0 vs K5) + **`rank_agreement_by_nturns`** (reward reliability) +
**`filter_thin_arms`** (drop <3-iter arms → no NaN rows)) · `behavior` (MITI counts + regex text
metrics) · `training` (generations.jsonl proxy reward + degeneracy + pref pairs +
`advantage_signal_by_iter` / `reward_distribution_frame` + **`load_branch_reliability`** (per-branch
n_turns + proxy, from the stored `prefix`) + **`tb_curves`**/`parse_run_tb` (self-contained TensorBoard
curve parse — no torch/trl import)) · `pref` (Mass-Mean-Probe: pooled `pref_word_ranking` +
`pref_word_drift_heatmap` + `plot_category_drift` + `top_words_by_iter` + **`preference_direction_drift`**
+ **`learn_unlearn_words`**) · `scores` (`scores_long` backbone + `load_subscales` + `to_wide` +
**`collapse_base`** — pool the 4 arm-bases into one descriptive Base) · `plots` (**named figure
functions**; **`effect_forest`** (the table-replacing forest), **`reliability_curve`**,
**`subscale_trajectory_grid`**; bar figures draw a dotted **base line**; no violins) · `figures`
(Okabe-Ito colourblind palette [PTO = cool, GRPO = warm, Base = grey] + grid + **`clean_label`** /
`relabel_xticks` / `add_base_line` / `figure_legend_from` + **`apply_score_axis`** + cfg-aware
`set_style(cfg)`) · `config` (**`EdaConfig`** — the single control surface, incl. `focus_arms` +
PNG/xlsx format defaults) · `notebook` (`notebook_setup(cfg)` → `Setup` incl. `S.CFG`) · `exports`
(`save_fig` PNG / `save_table` MD+XLSX → `results/<group>/`; **`set_export_group`** / **`set_formats`** /
**`save_provenance`** / **`build_index`** / **`reset_results`**). Selection + collapse helpers:
**`select_scores`**, `plots.overlay_trajectory` (configurable contrast), `plots.heterogeneity_grid`
(one fig, panel per arm), `arms=`/`iters=` on `single_metric_trajectory`/`trajectory_grid`. New
analysis: **`plots.factor_loadings_bars`** (readable PC1/PC2 loadings — replaces the biplot),
`plots.leaderboard_scorecard` (warmth + orthogonal axes), diverging `rubric_correlation_heatmap`,
`stats.rubric_factor_space`, `display_label` (lower-is-better ↓).

`lib/` is the OLD Exp2 package, kept only for `Run_Eval` scoring. `archive_exp2/` is the frozen Exp2 EDA.

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → add an `EXPERIMENTS` entry → `Run_Eval` →
the notebooks pick it up automatically. (Only register `model_iter` dirs that actually contain convs.)

## Latest results (snapshot 2026-06-14)
Scored: **PTO LA0** 0–10, **GRPO LA0** 0–8, **PTO LA5** 0–4, GRPO LA5 base — all on the full battery
incl. the orthogonal axes (PCT, MICI, R:Q/%CR/%MICO). Headlines: large warmth gains vs base (PTO LA0
Q1+Q2 4.26, GRPO LA0 4.08); **PTO vs GRPO is a near-tie at matched budget** (slopes ~0.12–0.13/iter);
the orthogonal axes show the warmth gains come *with* a ~2.3× rise in **MI-inconsistent** behavior and
**affirmation drift in both methods** (so "all rubrics up" is not multi-skill — PC1 drops 91%→≈56% once
the new axes are included). Regenerate with `0_Headline.ipynb` + `1_Eval_and_Behavior.ipynb`; full
numbers in `5_Detailed_Stats` and the `project-pto-la0-eval-results` memory.

---

## Improvement roadmap — making the EDA better & more readable
Prioritized; none are blocking. Ordered by value-for-effort.

**Landed (2026-06-09 → 2026-06-10).** The `exp3/` package + disk-discovery + true-persona recovery +
both stat batteries (2026-06-09 rebuild), then four passes of readability/restructure: hybrid plotting
(recurring figures as `plots.py` functions) + `notebook_setup()`; **method-symmetry** (every per-arm
view runs for both methods); the **7 by-purpose notebooks** above; concise evergreen markdown with the
**`[EVAL]`/`[TRAINING]`** tag; pooled descriptive **Base** (`scores.collapse_base`); subscale
**trajectories** (was a bar wall) + **`effect_forest`** (was the wide table) + reliability curve +
TB curves + richer preference latent space; Okabe-Ito **colourblind palette**, full labels, base lines,
**no violins**; heavy tables only in `6`, **thin arms filtered**. Full blow-by-blow lives in the Exp3
`CLAUDE.md` log. **Remaining roadmap:**

**Reproducibility / speed:**
5. **Cache `scores_long` + `behavior_by_iter` to parquet.** `behavior`/`text_metrics` re-read ~2k
   conversation CSVs in every notebook (slow). A `scores.load_cached()` that writes/reads
   `results/cache/*.parquet` (keyed by arm+iter set) would make notebooks near-instant and consistent.
   `notebook_setup()` is the natural home for the toggle.
7. **Discovery should skip empty `model_iter` dirs** (a partial arm's empty iter dir produces blank
   rows) so partial arms never pollute the views; and `Run_Eval`'s registry could be auto-generated
   from `discover_arms()` to remove the last hand-maintained list.

**Rigor / correctness polish:**
8. **Self-check script.** Commit the ad-hoc validation as `exp3/_selfcheck.py` (persona recovery 100%,
   known means reproduce, probe `wins_correct`>0.5) — a 10-second regression test after any change.
9. **Unify styling.** `lib.set_plot_style` and `figures.set_style` both exist; the live EDA should use
   only `figures.set_style` (publication rcParams).

**Recommended next:** 5 (parquet caching — biggest speed win now that the structure is settled) then 8
(commit the self-check as a regression guard).
