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
Not hardcoded here (arms are still training). Run `00_Main_Results.ipynb`; the current snapshot +
interpretation lives in the `project-pto-la0-eval-results` memory.

---

## Improvement roadmap — making the EDA better & more readable
Prioritized; none are blocking. Ordered by value-for-effort.

**Landed in the 2026-06-10 (restructure) pass** (Lior's round-3 notes): notebooks **reorganized by
purpose** into the 7 above (was 6 by research question); **concise evergreen markdown** with the
`[EVAL]`/`[TRAINING]` tag per section; **all heavy tables moved to `6_Detailed_Stats`** and the headline
"did it work" shown as an **`effect_forest`** dot-plot instead; **thin arms filtered** (no NaN rows);
**violins dropped**. New analyses: `3_Training_Diagnostics` surfaces the **TensorBoard training curves**
(`training.tb_curves`, self-contained parse); `4_Reward_Reliability` **rebuilds the Exp2 partial-conv
reliability curve on Exp3 data** from the per-branch `prefix` in `generations.jsonl` (no new oracle pass)
and contrasts LA0 vs LA5; `5_Preference_LatentSpace` gains **direction-drift (2D)**, **learned/unlearned
words**, and a **K0-vs-K5** contrast. Remaining:

**Landed in the 2026-06-10 (later) figure-readability pass** (roadmap items 1/3/4/10 + the four
figures Lior flagged): (a) **pooled Base** — `scores.collapse_base` merges the 4 near-identical
arm-bases into one descriptive `Base` for the cross-model bar/rank views (paired vs-base stats keep
each arm's own base); (b) **subscales** — the unreadable 26-model × 3–4-subscale grouped-bar wall
(`subscales_WAI_MITI.pdf`, retired) is replaced by `plots.subscale_trajectory_grid` (subscale lines
across iterations, one panel per parent×arm → `subscale_trajectories.pdf`); (c) **preference drift** —
`pref.pref_word_drift_heatmap` (top words × iteration) + `pref.plot_category_drift` (MI-concept lines)
now show how the preference shifts over training, alongside the pooled `pref_word_ranking` snapshot;
(d) **polish** — Okabe-Ito colourblind palette (PTO = cool, GRPO = warm, Base = grey), full
no-abbreviation labels (`figures.clean_label`), single shared legends above grids, a dotted base line on
bar figures, and the PC1≈91% **shared-factor caveat** printed under the trajectory grid; (e)
**restructure** — `01` leads with the trajectory grid (headline) and demotes the per-model bars to an
Appendix. Validated: package smoke + `00`/`01`/`05` ran top-to-bottom via nbconvert (`thesis-venv313`).

**Landed in the 2026-06-10 (notebook-narrative) pass** (Lior's second round of notes): every notebook's
markdown is now **evergreen + audience-framed** (what / for whom / what the output is — no numbers that
go stale) with an explicit **[EVAL]** (full-conversation oracle scores) vs **[TRAINING]** (partial-conv
branch rewards / preference pairs) tag per section, because the two were easy to confuse; the global
all-vs-best **selection toggle was removed** in favour of a per-view choice (learning curves = all iters,
leaderboard = best + Base, appendix = all); the redundant **QC section was dropped** from `01`; `05` now
prints a **per-iteration top-words table + first→last MI-concept read-out**. Remaining:

**Landed in the 2026-06-10 refactor** (readability + method-symmetry + research-question reorg):
reorganized the notebooks by research question; moved the recurring figures into `exp3/plots.py`
(hybrid plotting); added `notebook_setup()` to kill the boilerplate; made every per-arm analysis run
for **both methods** (only the preference probe stays PTO-only by construction); lifted the buried
cross-method/K comparisons into `stats.paired_method_comparison` / `paired_k_comparison`; added a
symmetric `training.advantage_signal_by_iter`; trimmed exports to **one format each** (PDF figures, MD
tables) with idempotent `CAPTIONS.md`; filled the takeaway cells; added the §6 artifact index in `00`
(roadmap items 2, partial 1+3+4). Remaining:

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
10. **Annotate the oracle-noise band consistently** (~0.10) on trajectories so readers see which
    differences are above the reproducibility floor (already in `00`; extend to `01`).

**Recommended first pass:** 1 + 2 + 5 (narrative + caching) — biggest readability and speed gains for
the least churn.
