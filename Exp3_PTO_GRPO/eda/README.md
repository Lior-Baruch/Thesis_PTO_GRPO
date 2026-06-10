# Exp3 EDA — guide + improvement roadmap

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), across
training iterations, under matched look-ahead K and MCL. All data/compute/stats lives in the `exp3/`
package; the recurring figures are named functions in `exp3/plots.py` (called once from multiple
notebooks), and genuinely one-off exploration stays inline (the **hybrid** plotting split). Thesis
figures/tables are exported to `results/` — **one format each**: figures `.pdf`, tables `.md`.

The notebooks are **organized by the thesis's research questions**, not by analysis type.

## Run order
1. **`Run_Eval.ipynb`** — async oracle scoring → `data/<method>/eval_scores/`. Registry-driven: add a
   `lib/config.py::EXPERIMENTS` entry per new run (the only place you hand-edit). Resume-safe.
2. **`00_Main_Results.ipynb`** — regenerates the canonical thesis figures + tables into `results/`
   (thin: only calls shared `stats.*`/`plots.*` helpers; ends with an artifact index).
3. **`01_Did_It_Work.ipynb`** — foundational: did **each arm** beat its own base? QC, selection toggle,
   outcomes/ranks/subscales/trajectories, the per-arm vs-base battery (familiar + persona-paired +
   Friedman), selection sensitivity. Symmetric over all arms.
4. **`02_PTO_vs_GRPO.ipynb`** — research question (ii): PTO vs GRPO at matched K. Matched-iteration
   paired contrast + stats, **training internals side-by-side** (PTO margin vs GRPO group_std, ungated),
   per-iteration climb rate. Absorbs the old `Exp3_DeepDive`.
5. **`03_LookAhead_K.ipynb`** — research question (i): K=0 vs K=5 within each method (paired K0−K5,
   margin under K). Preliminary while the LA5 arms are thin.
6. **`04_Mechanism_and_Behavior.ipynb`** — behavior drift (MITI counts + text), reward faithfulness,
   rubric PCA, heterogeneity by true persona, session-end, persona-matched transcripts — **all arms**.
7. **`05_Preference_LatentSpace.ipynb`** (PTO only) — Mass-Mean-Probe: which words / MI-concepts the
   policy prefers + drift across iterations (asymmetric by design — GRPO has no pairs).
8. **`Iteration_Reward_EDA.ipynb`** — live in-flight training health check (uses the old `lib`).

Future: an oracle-comparison notebook (research question iii) once non-Q1Q2 oracles are run.

Everything **auto-discovers** arms from disk via `exp3.discover_arms()` (no path literals). Every
notebook's cell 1 is `S = exp3.notebook_setup()` → `S.ARMS / S.SCORES / S.PALETTE / S.METRICS /
S.ORACLE_NOISE / S.RESULTS_DIR`. Notebooks run with the venv kernel `thesis-venv313`, cwd = `eda/`.

## Package (`exp3/`)
`discovery` (arms manifest) · `personas` (TRUE-persona recovery — replays the per-iter shuffle; the old
`lib` join is wrong for Exp3) · `scores` (`scores_long` backbone + `load_subscales` + `to_wide`) ·
`select` (all vs best-per-experiment) · `stats` (omnibus/Mann-Whitney+FDR + persona-paired Wilcoxon/dz/
bootstrap + **Friedman/Kendall-W** + `main_results_table` + **`paired_method_comparison`** (PTO vs GRPO)
+ **`paired_k_comparison`** (K0 vs K5)) · `behavior` (MITI counts + regex text metrics) · `training`
(generations.jsonl proxy reward + degeneracy + pref pairs + **`advantage_signal_by_iter`** /
**`reward_distribution_frame`** — both methods) · `pref` (preference embeddings + Mass-Mean-Probe +
`pref_word_ranking`) · `plots` (**named figure functions** — the hybrid core) · `figures` (style/palette/
grid helpers only) · `notebook` (`notebook_setup`) · `exports` (`save_fig` PDF / `save_table` MD →
`results/`).

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
