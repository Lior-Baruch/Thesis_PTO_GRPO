# Exp3 EDA — guide + improvement roadmap

Analysis for **PTO_Exp3 vs GRPO_Exp3** (Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle), across
training iterations, under matched look-ahead K and MCL. All analysis lives in the `exp3/` package
(data + compute + stats) with **plotting inline in the notebooks** (so figures are editable); thesis
figures/tables are exported to `results/`.

## Run order
1. **`Run_Eval.ipynb`** — async oracle scoring → `data/<method>/eval_scores/`. Registry-driven: add a
   `lib/config.py::EXPERIMENTS` entry per new run (the only place you hand-edit). Resume-safe.
2. **`00_Main_Results.ipynb`** — regenerates the canonical thesis figures + tables into `results/`.
3. **`01_Outcomes_and_Stats.ipynb`** — outcomes, rankings, subscales, trajectories, stats (familiar +
   persona-paired + Friedman), PTO-vs-GRPO / K0-vs-K5, selection-sensitivity.
4. **`02_Mechanism_and_Exploration.ipynb`** — behavior drift (MITI counts + text), reward faithfulness,
   rubric PCA, heterogeneity by true persona, transcript sampler + persona-matched evolution.
5. **`03_Preference_Analysis.ipynb`** (PTO) — Latent-space Mass-Mean-Probe: which words / MI-concepts
   the policy prefers, and drift across iterations.
6. **`Exp3_DeepDive.ipynb`** / **`Iteration_Reward_EDA.ipynb`** — per-arm training internals.

Everything **auto-discovers** arms from disk via `exp3.discover_arms()` (no path literals). Notebooks
run with the venv kernel `thesis-venv313`, cwd = `eda/`.

## Package (`exp3/`)
`discovery` (arms manifest) · `personas` (TRUE-persona recovery — replays the per-iter shuffle; the old
`lib` join is wrong for Exp3) · `scores` (`scores_long` backbone + `load_subscales` + `to_wide`) ·
`select` (all vs best-per-experiment) · `stats` (omnibus/Mann-Whitney+FDR + persona-paired Wilcoxon/dz/
bootstrap + **Friedman/Kendall-W** + `main_results_table`) · `behavior` (MITI counts + regex text
metrics) · `training` (generations.jsonl proxy reward + degeneracy + pref pairs) · `pref` (preference
embeddings + Mass-Mean-Probe) · `figures` (style/palette/grid helpers only) · `exports` (`save_fig`/
`save_table` → `results/`).

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

**Readability (highest value):**
1. **Per-figure "takeaway" lines.** After each figure/table, a one-sentence *what this shows* + *why it
   matters* markdown cell. Currently the takeaways sections are placeholders — filling them turns the
   notebooks into a narrative a reader (or thesis committee) can skim. Pair with the `CAPTIONS.md`
   already generated.
2. **Research-question headers.** Open each notebook with the explicit thesis question it answers
   (look-ahead K? PTO vs GRPO? reward faithfulness?) and which figure/table is the answer.
3. **Trim dense tables to the columns that matter.** The stat tables show everything; for the thesis,
   surface a 4–5 column view (Δ, dz+label, Holm p, CI) and keep the full table in `results/` only.
4. **A figure gallery.** A tiny `results/INDEX.md` (or a cell in `00`) that embeds every
   `results/figures/*.png` under its caption — one page to review all deliverables.

**Reproducibility / speed:**
5. **Cache `scores_long` + `behavior_by_iter` to parquet.** `behavior`/`text_metrics` re-read ~2k
   conversation CSVs in every notebook (slow). A `scores.load_cached()` that writes/reads
   `results/cache/*.parquet` (keyed by arm+iter set) would make notebooks near-instant and consistent.
6. **One config block.** A small `exp3/config.py` (or top-of-notebook `CONFIG`) for the focus arm,
   default `SELECTION`, metric order, and `ORACLE_NOISE` — instead of `ARM="PTO_LA0"` hardcoded in
   several places.
7. **Discovery should skip empty `model_iter` dirs** (e.g. GRPO LA5 iter1 currently has 0 convs) so
   partial arms never produce blank rows; and `Run_Eval`'s registry could be auto-generated from
   `discover_arms()` to remove the last hand-maintained list.

**Rigor / correctness polish:**
8. **Self-check script.** Commit the ad-hoc validation as `exp3/_selfcheck.py` (persona recovery 100%,
   known means reproduce, probe `wins_correct`>0.5) — a 10-second regression test after any change.
9. **Unify styling.** `lib.set_plot_style` and `figures.set_style` both exist; the live EDA should use
   only `figures.set_style` (publication rcParams).
10. **Annotate the oracle-noise band consistently** (~0.10) on trajectories so readers see which
    differences are above the reproducibility floor (already in `00`; extend to `01`).

**Recommended first pass:** 1 + 2 + 5 (narrative + caching) — biggest readability and speed gains for
the least churn.
