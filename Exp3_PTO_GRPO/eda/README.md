# Exp3 EDA — guide + improvement roadmap

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), across
training iterations, under matched look-ahead K and MCL. All data/compute/stats lives in the `exp3/`
package; the recurring figures are named functions in `exp3/plots.py` (called once from multiple
notebooks), and genuinely one-off exploration stays inline (the **hybrid** plotting split). Thesis
figures/tables are exported to `results/` — **one format each**: figures `.pdf`, tables `.md`.

The notebooks are **organized by purpose** (not by research question): a thin headline entry, then one
notebook each for eval outcomes, behaviour, the training signal, reward reliability, the preference
latent space, and the detailed stats. Every section is tagged **`[EVAL]`** (full-conversation oracle
scores — the held-out outcome) or **`[TRAINING]`** (partial-branch rewards / preference pairs — what
the policy is updated on). Markdown is kept concise (≤3-line sections).

## Run order
1. **`Run_Eval.ipynb`** — async oracle scoring → `data/<method>/eval_scores/`. Registry-driven: add a
   `lib/config.py::EXPERIMENTS` entry per new run (the only place you hand-edit). Resume-safe.
2. **`0_Headline.ipynb`** `[EVAL]` — thin: the 3 canonical thesis figures (headline outcomes vs pooled
   Base, vs-base **effect forest**, Q1+Q2 learning curve) + artifact index.
3. **`1_Eval_Results.ipynb`** `[EVAL]` — full-conversation outcomes: all-rubric trajectories, subscale
   trajectories, effect forest, **PTO-vs-GRPO** + **K0-vs-K5** contrast figures, leaderboard, appendix
   per-model bars. Verdicts in prose; heavy tables live in `6`.
4. **`2_Behavior_and_Mechanism.ipynb`** `[EVAL]` — MITI behaviour drift + text metrics, rubric factor
   structure (PC1 + corr heatmap), heterogeneity by true persona, session-end/length, persona-matched
   transcripts. All arms. (No violins.)
5. **`3_Training_Diagnostics.ipynb`** `[TRAINING]` — the TensorBoard training curves per arm
   (`training.tb_curves`), per-candidate reward distribution, method-native advantage signal
   (group_std / margin), degeneration check.
6. **`4_Reward_Reliability.ipynb`** `[TRAINING↔EVAL]` — is the partial-conv training reward faithful to
   the full-conv eval? rank-agreement-vs-`n_turns` curve (LA0 vs LA5), proxy-vs-eval scatter, PTO
   margin-by-branch-depth.
7. **`5_Preference_LatentSpace.ipynb`** `[TRAINING]` (PTO only) — Mass-Mean-Probe: word ranking + drift,
   **direction drift in 2D**, **learned/unlearned words**, MI-concept drift, **K0-vs-K5** contrast.
8. **`6_Detailed_Stats.ipynb`** `[EVAL]` — **all the heavy tables** (main results, Friedman, paired
   method/K, per-arm vs-base, slopes, rankings, PCA), thin arms filtered, exported to `results/tables/`.
9. **`Iteration_Reward_EDA.ipynb`** — live in-flight training health check (uses the old `lib`).

Future: an oracle-comparison notebook (research question iii) once non-Q1Q2 oracles are run.

Everything **auto-discovers** arms from disk via `exp3.discover_arms()` (no path literals). Every
notebook's cell 1 is `S = exp3.notebook_setup()` → `S.ARMS / S.SCORES / S.PALETTE / S.METRICS /
S.ORACLE_NOISE / S.RESULTS_DIR`. Notebooks run with the venv kernel `thesis-venv313`, cwd = `eda/`.

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
`relabel_xticks` / `add_base_line` / `figure_legend_from`) · `notebook` (`notebook_setup`) · `exports`
(`save_fig` PDF / `save_table` MD → `results/`).

`lib/` is the OLD Exp2 package, kept only for `Run_Eval` scoring. `archive_exp2/` is the frozen Exp2 EDA.

## Adding a new run
Train → it writes `conversations/full/<EXP>/model_iter_*` → add an `EXPERIMENTS` entry → `Run_Eval` →
the notebooks pick it up automatically. (Only register `model_iter` dirs that actually contain convs.)

## Latest results
Not hardcoded here (arms are still training). Run `0_Headline.ipynb`; the current snapshot +
interpretation lives in the `project-pto-la0-eval-results` memory.

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
