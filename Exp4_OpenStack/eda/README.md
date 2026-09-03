# Exp4 EDA — how the analysis layer works

The **spec** (what Exp4 is, the arm grammar, the data layout, the VRAM budget) lives in
[../CLAUDE.md](../CLAUDE.md). This file owns the **mechanics**: how a notebook is wired, where its
output goes, how to regenerate everything, and where to put a change. Nothing here is dated and
nothing here is a result — narrative and numbers live in `results/<top>/SUMMARY.md`.

## The model, in one paragraph

`results/` is organised by **research question**, not by arm subset. One question = one **family**
`"<top>/<sub>"` = one notebook = one output folder, and the mapping is 1:1 and total:
`notebooks/<top>/<sub>.ipynb` writes `results/<top>/<sub>/{figures,tables}/`, and the set of legal
families is [`eda_analysis/config.py`](eda_analysis/config.py)`::FAMILIES`. Nothing else selects a
family: `render_results.py` iterates that dict, so an entry with no notebook renders nothing and a
notebook with no entry is never rendered. **Arms are discovered from disk** — a run becomes
analysable the moment its conversations land, and there is no registry to edit anywhere.

## The four families

| Family | Notebook | Question |
|---|---|---|
| `arms/outcomes` | `notebooks/arms/outcomes.ipynb` | Per-arm descriptives: every arm on one axis, across model states. |
| `lookahead/reward` | `notebooks/lookahead/reward.ipynb` | **RQ-i** — K=0 vs K=5 *within* each optimizer. |
| `method/contrast` | `notebooks/method/contrast.ipynb` | **RQ-ii** — PTO vs GRPO at matched K. |
| `compute/cost` | `notebooks/compute/cost.ipynb` | The spend axis: GPU-hours per (arm, iteration) and API calls. |

Four is the whole of v1, and `arms` renders first — its descriptive tables are what a reader checks
a contrast against.

## The results tree

```
results/
├── INDEX.md                     auto — one line per family, with artifact counts
├── METRICS_REFERENCE.md  LIMITATIONS.md  schematics/     hand-authored, never regenerated
└── <top>/                       arms | lookahead | method | compute
    ├── SUMMARY.md               HAND-AUTHORED narrative (the interpretation)
    ├── INDEX.md                 auto — artifact map for this top, with captions
    └── <sub>/
        ├── figures/[<group>/]<name>.png    + CAPTIONS.md, + _provenance.md at the leaf root
        └── tables/[<group>/]<name>.md      + <sub>.xlsx (one workbook per leaf, one sheet per
                                              table) + <name>.json number ledgers
```

The four names in `exports.PRESERVE` (`SUMMARY.md`, `METRICS_REFERENCE.md`, `LIMITATIONS.md`,
`schematics`) are never written or deleted by any automated path — structurally, because every walk
descends only into a family's `figures/` + `tables/`, and explicitly, because `_guard_path` raises
if it ever reaches one. **There is no `<judge>/` level**; see "The judge dimension" below.

## The cell-1 contract

Cell 1 of every family notebook is exactly this, and nothing else configures a render:

```python
import os, eda_analysis
cfg = eda_analysis.EdaConfig(family="arms/outcomes")
S   = eda_analysis.notebook_setup(cfg)
```

`notebook_setup` returns a frozen `Setup` with eight fields:

| | |
|---|---|
| `S.ARMS` | discovered arms after the config's filters; each carries `.label`, `.iters`, `.info`, and its paths |
| `S.SCORES` | the long score frame for `S.JUDGE` — **may be empty**; an arm that has trained but not been scored is a normal state, so every family must render something rather than raise |
| `S.PALETTE` | `{arm label: colour}` covering every arm in `S.SCORES` |
| `S.METRICS` | metric keys actually present in the data, in `METRIC_ORDER` |
| `S.RESULTS_DIR` | `results/<top>/<sub>/` for this family |
| `S.FAMILY` | the validated `"<top>/<sub>"` |
| `S.JUDGE` | the **resolved** judge tag — never `""`, so a caption can always name the grader |
| `S.CFG` | the config that actually ran, keyword overrides already applied. Read this, not the notebook's own `cfg` |

The order inside `notebook_setup` is deliberate: family validated **before** any disk work, export
routing configured **before** anything is computed (a notebook cannot compute a figure it has
nowhere to put), provenance written **last**, from the frame that was actually loaded.

`EdaConfig` is frozen; derive a variant with `cfg.with_(ks=[5], note="K=5 only")` so the change is
visible at the call site. Arm filters (`methods`, `ks`, `modes`, `arm_labels`) default to `None` =
no filter — every arm shares one axis in every family, so narrowing is the exception.

### Regenerate everything

```powershell
cd Exp4_OpenStack\eda
..\..\.venv\Scripts\python.exe tools\render_results.py            # every family
..\..\.venv\Scripts\python.exe tools\render_results.py --help     # subset flags
```

⚠ Use the repo venv explicitly. A bare `python` on this machine is not it, and the EDA's
dependencies (pandas, pyarrow, matplotlib, seaborn, openpyxl, nbformat) are installed there.

## The judge dimension

The score lake partitions on the grader:

```
data/eval_scores/judge=<tag>/rep=<r>/metric=<M>/<EXP_NAME>/model_iter_<N>.parquet
```

A judge grades **after the fact**, so a judge swap is re-runnable — which is why it is a partition
key, while the oracle and the patient, which are not re-runnable, are baked into the arm name.
`rep=0` is the full draw every family reports; `rep>=1` are repeatability re-draws over a subset.
`EdaConfig.judge` selects which grader a notebook loads **by default**; a family that wants
several calls `data.scores_by_judge(...)`, which returns one frame with a `judge` column. The judge
travels as an argument, never as module-global state — `render_results.py` runs families in
parallel, and a process-global "active grader" would be a race.

**There is no `<judge>/` results leaf.** Exp3 nested one under its per-arm families and rendered
them once per grader; Exp4 does not, because the interesting table puts the graders side by side —
the default judge shares a model with the training oracle and a second judge is genuinely held out —
and a directory named after one grader would assert something false about its own contents. A
genuinely single-grader artifact gets a judge-qualified **name** (`outcomes_gemma4E4B`), never a
path level.

⚠ **Never average raw scores across judges.** One grader was the training oracle — the thing the
policy was optimized against — and any other is held out. That is train-vs-test, not two raters of
one construct, and they do not share a scale (Exp3 measured a 1.2–1.7 point level offset that was
itself model-dependent, so a mean over judges applies a silent, model-dependent shrinkage to every
effect it touches). Combine only **contrasts** (a difference between two model states under *one*
judge) or standardized quantities. A finding worth reporting survives in each judge's own column.

⚠ Only **eval-side** numbers are judge-swappable. Candidate rewards in `generations.jsonl`, PTO's
`pairs.csv` and the TensorBoard curves were produced by the *training* oracle while the run
happened; re-rendering them under a second judge's label would imply a measurement that never took
place.

## Exports — the one door to disk

[`eda_analysis/exports.py`](eda_analysis/exports.py) owns path composition, captions, byte
determinism and the delete. Figure builders in `plotting.py` **return** figures and never save;
only this module writes.

| | |
|---|---|
| `save_fig(fig, name, *, group=None, formats=None, dpi=200, caption=None)` | → `figures/[<group>/]<name>.<fmt>`. Does not close the figure. |
| `save_table(df, name, *, group=None, formats=None, float_format="%.3f", index=False, caption=None)` | → `tables/[<group>/]<name>.md` + a sheet in the leaf workbook. |
| `save_numbers(name, values, *, group=None, caption=None)` | → `tables/[<group>/]<name>.json`, a **merged** ledger of `{key: {value, source, note}}` so a write-up can cite `…/<name>.json :: <key>`. |
| `save_provenance(cfg, scores)` | → `figures/_provenance.md`. Called by `notebook_setup`. |
| `build_index()` | writes `results/<top>/INDEX.md` and refreshes `results/INDEX.md`. **End every notebook with it.** |
| `reset_results(groups=None)` | clears the ACTIVE family's `figures/` + `tables/` before a clean regenerate. |
| `PRESERVE` | the four hand-authored names no automated path may touch. |

- **A family is required.** Every `save_*` raises `NoFamilyError` until `set_family` has run
  (`notebook_setup` does it). There is deliberately no bare-`results/` fallback — that is how an
  artifact ends up somewhere no index points at. `name` must be a bare stem; use `group=` for a
  subfolder.
- **An empty frame writes an explicit EMPTY-TABLE marker**, not a 0-byte file, which would read as
  "rendered" to every check that only asks whether the file exists. A table over `MD_MAX_BYTES`
  (64 KB) becomes a head excerpt plus a pointer to the workbook sheet holding every row.
- **Captions are the artifact's only description.** Write one at the save call or it never gets
  written; `build_index` is what surfaces it.

### Determinism — why a re-render must be byte-identical

An unchanged number must produce an unchanged file, or a diff of thirty artifacts reads as "the
results moved" when only a clock did, and the one artifact that *did* change is invisible.

- **`BOOT_SEED` everywhere.** Every resampler in `stats.py` defaults to it, and every seaborn call
  that draws a bootstrap error bar passes `seed=BOOT_SEED`. Seaborn's `errorbar=("ci", 95)` defaults
  to `seed=None`, i.e. a fresh 1,000-sample bootstrap per call; left unset, three renders of the
  same notebook on identical data differ by a few percent of pixels.
- **Frozen `.xlsx` timestamps.** openpyxl stamps the current clock into `docProps/core.xml` *and*
  into every zip entry's mtime, so `_normalize_xlsx` rewrites the archive afterwards with
  `EXPORT_EPOCH` pinned, member content copied through untouched. Sheets are re-sorted
  alphabetically for the same reason (openpyxl appends a replaced sheet at the end).
- **No timestamps in ledgers or `_provenance.md`**, and **`CAPTIONS.md` is kept sorted and
  de-duplicated** so a leaf written by several cells does not churn.

## Scoring — the paid side

`notebooks/scoring/Run_Eval.ipynb` writes the score lake. It is **never part of a render**:
`eda_analysis.scoring` is deliberately absent from the package's lazy-attribute map, so it can only
be reached by an explicit `from eda_analysis import scoring` — and that import is the point at which
someone is choosing to spend.

- **It runs on the GPU host, not locally.** The judge is served by vLLM on the card that trains
  (Colab A100 80 GB, or a GPU server); the local 12 GB card cannot hold the E4B judge (14.89 GiB
  of weights alone), so locally the notebook is a smoke test against a fake or remote endpoint.
  Its setup cell carries the same Drive-mount preamble as the trainer notebooks (`COLAB_EDA_DIR`),
  which is why **`eda/` is pushed to Drive beside `code/`** (additively; never `data/`) — the
  scoring module and the canonical `code/` modules it imports both have to be on the mount. The
  analysis families stay local: they read parquet and JSONL through the Drive symlinks.
- **Its last cell is the prompt-length gate.** `scoring.gather_transcripts` →
  `scoring.prompt_length_gate` → `scoring.check_prompt_length_gate` measures every real Exp4 oracle
  prompt (through `tools.oracle_sanity.prompt_length_report`, in the served model's tokenizer)
  against the server's `max_model_len` — the cap the server itself reports on `/tokenize`, never
  the notebook's `SERVE_MAX_MODEL_LEN` literal (an adopted server keeps its launch-time cap; the
  literal is forwarded only when no server is used, as the offline fallback, and a mismatch is
  printed as a note); a prompt over the cap is a conversation that cannot be graded, and the ones
  that overflow are the longest — arm- and K-dependent missingness. The gate refuses to pass when
  the report function is absent or the report is undecidable, and an empty lake is a *vacuous*
  pass that renders as one ("nothing measured"), not as a failed verdict.
- **Arms auto-discover.** A run is scoreable as soon as its conversations land; nothing to register.
- **Resume is by whole parquet.** The unit is one `(judge, rep, metric, arm, model state)` file of
  96 rows. A state whose parquet already exists is skipped entirely, so a second pass over a
  finished lake issues zero grader calls and a partial pass costs only what is missing.
- **A re-score is free** — the default judge is the same local Gemma the trainer serves, so
  regenerating the lake costs GPU time and nothing else. That is the single biggest practical
  difference from Exp3, where the lake was several hundred dollars of irreplaceable API calls and
  "delete the partition and re-run" was not an available move. Here it is: drop
  `judge=<tag>/rep=<r>/metric=<M>/` and re-run. ⚠ The freedom ends the moment a binding points at a
  vendor API — then the lake is expensive again and Exp3's rules apply.

## Fields the trainers write that the readers may or may not surface yet

The pre-run review (see [`../history/CHANGELOG.md`](../history/CHANGELOG.md)) added artifacts on
the write side. What the EDA does with each:

| Artifact | Written by | Read by |
|---|---|---|
| `timing_sessions.jsonl` lines with `"partial": true` (`note: "training partial: checkpoint-N"`) | `core.timing.log_training_progress` from the trainers' `on_save`; the closing line from `finalize_training` | `data.load_timing` → `core.timing.cumulative_seconds`, unchanged: partial lines carry the same per-process token and **sum like any other line**, so `training_s` for a preempted-then-resumed iteration is the true total and `n_sessions_production` still counts processes, not lines. `compute/cost` needs no change. The flag is audit only. |
| `iteration_metadata.json` → `peak_reserved_gib_<phase>` / `peak_allocated_gib_<phase>` — ONE flat shape for BOTH trainers since the 2026-09-03 repair round (phase ∈ `generate`, `build` (PTO only), `train`, `eval_generate`; `generate` is absent on a mid-training resume; `torch.cuda.max_memory_reserved` / `max_memory_allocated` per phase, stamped by the trainers) and `run_metadata.json` → a `runtime` block for BOTH methods (`gpu_total_gib` + its source, `vllm_version`, `package_versions` from `core.runtime.describe_environment`; PTO gained it in the same round) | trainers / `core.runtime.describe_environment` | No loader reads `iteration_metadata.json` today (its per-process timing fields are the Exp3 undercount, on purpose). Read the peak-memory keys by hand after the `QUICK_TEST` rehearsal — the same key names on both arms, so a GRPO-vs-PTO peak comparison is a flat join, not a reshape; `data.load_run_metadata` returns the whole dict, `runtime` block included. |
| `generations.jsonl` candidate keys `not_graded_reason` (only when `score` is null: `oracle_failed` / `patient_error` / `gpu_error` / `prompt_overflow` / `parse_error`), `ended_by_candidate` (every candidate), `lookahead.stop_reason` | `core.reward.CandidateScore.to_record`, `core.lookahead.LookaheadResult.to_record`; on PTO rows `gpu_error` / `prompt_overflow` can also come from the branch sampler (`oracle_attempts == 0`, no `lookahead` dict) | **Columns of `data.load_generations`** since the 2026-09-02 gate pass: `not_graded_reason` (None on every graded row — so a NaN `score` no longer mixes a grader failure with a simulator failure), `ended_by_candidate` (bool) and `lookahead_stop_reason` (`""` ran to K / `session_ended` / `degenerate` / a not-graded reason). ⚠ For `ended_by_candidate` rows the reconstruction rule `prefix + "\n\n[THERAPIST]: " + completion + tail` no longer reproduces the graded text — split `completion` at `SESSION ENDED` first (`core.lookahead.split_session_end`). |

## Self-check

```powershell
..\..\.venv\Scripts\python.exe -m eda_analysis._selfcheck            # everything
..\..\.venv\Scripts\python.exe -m eda_analysis._selfcheck --fast     # structural subset only
```

`--fast` runs the checks that need nothing but the package: the `FAMILIES` ↔ notebook map is 1:1,
the metric registry agrees with the canonical `questionnaires.py`, judge and metric partition tokens
are legal directory names, `constants` resolved the *canonical* `code/` modules rather than another
experiment's copy, every reader's empty-frame column contract holds, the export layer refuses a
bare-root save, and `PRESERVE` survives a `reset_results` (exercised against a temp `RESULTS_DIR`,
never the real tree). A full run additionally exercises the loaders against whatever is on disk.
**Run `--fast` after any change to the package, and the full check before a render you intend to
commit.**

## Package map

| Module | What it owns |
|---|---|
| [`eda_analysis/constants.py`](eda_analysis/constants.py) | The **leaf**: workspace paths, the metric registry, judge tags, `BOOT_SEED`, label/colour keys, persona vocabulary. Imports nothing from the package, and performs the `sys.path` insert that makes `naming` / `roles` / `questionnaires` resolve to the single canonical copies under `../code/`. |
| [`eda_analysis/config.py`](eda_analysis/config.py) | The control surface: `FAMILIES`, `EdaConfig`, `Setup`, `notebook_setup`, and the sibling contract those depend on. |
| [`eda_analysis/data.py`](eda_analysis/data.py) | Every number's origin: `discover_arms` / `filter_arms` / `Arm`, the five readers (`load_scores_long`, `load_conversations`, `load_generations`, `load_timing`, `load_pref_pairs`), `scores_by_judge`, `load_run_metadata`, and the frame cache. |
| [`eda_analysis/exports.py`](eda_analysis/exports.py) | Path composition, captions, indices, byte determinism, and the only two functions that delete. |
| [`eda_analysis/stats.py`](eda_analysis/stats.py) | `paired_arrays`, `paired_contrast`, `bootstrap_ci`, `holm`, `spearman`, `cohens_dz`, `orient_contrast`, `summarize_contrasts`. Repeated-measures by default; no scipy (the t tail and Spearman are ~40 lines of stdlib maths). |
| [`eda_analysis/plotting.py`](eda_analysis/plotting.py) | One style, one deterministic arm palette, four reusable figure builders (`score_trajectory`, `arm_distribution`, `contrast_forest`, `cost_benefit`). Returns figures; never saves. |
| `eda_analysis/_selfcheck.py` | The check suite above. |
| [`eda_analysis/scoring.py`](eda_analysis/scoring.py) | The **paid** side: builds a grader client, writes the score lake, and carries the prompt-length gate (`gather_transcripts` / `prompt_length_gate` / `check_prompt_length_gate`). Never imported implicitly; runs on the GPU host. |
| `tools/render_results.py` | Executes every family notebook headlessly and rebuilds the indices. |

`import eda_analysis` loads only `constants` and `config`; pandas, matplotlib, seaborn and pyarrow
arrive with the first attribute that needs them.

**`stats.py` verification note** (referenced from that module's docstring). scipy is not a
dependency, so the two distribution functions it reimplements were checked against it once, in a
throwaway session: over 1,000 paired samples at n ∈ {5, 12, 30, 96, 200},
`stats.paired_contrast(...)["p"]` matched `scipy.stats.ttest_rel` to a worst-case 2.8e-12, and
`stats.spearman` matched `scipy.stats.spearmanr` to 3.3e-16. Re-run that comparison if the t tail
or the rank code is ever touched.

⚠ **The EDA must never import torch, `core.policy` or `core.lookahead`.** It reads finished
artifacts; anything here that wants a GPU is in the wrong package. Importing `core.config`,
`core.timing`, `core.recorder` and `core.conversations` is fine and deliberate — they are
stdlib-only, and borrowing the *writer's* path builder is what stops a layout change from
half-landing.

### Frame cache

`data.load_cached` memoises built frames to `eda/.eda_cache/*.parquet`, keyed on the frame name, its
parameters, the arms, the judge and rep, **and a content signature over every input file's
`(name, size, mtime_ns)`**. It self-invalidates: a rewritten input is a miss, and so is an input
that only just appeared. Disable per call (`cache=False`), per session (`data.set_cache(False)`), or
per shell (`EDA_NO_CACHE=1`); `data.reset_cache()` clears it. `EDA_VERBOSE=1` narrates what
discovery skipped.

## Extension points

| Change | Where it goes |
|---|---|
| **A new question** | One entry in `config.FAMILIES` + one notebook at `notebooks/<top>/<sub>.ipynb` on the cell-1 contract. A new *top* also gets a hand-authored `results/<top>/SUMMARY.md`. |
| **A new instrument** | One entry in `constants.METRICS` (or `COMPOSITE_METRICS`) + its place in `METRIC_ORDER`. Item labels, counts and scale derive from `../code/questionnaires.py` — never retype them. |
| **A new grader** | A `RoleBinding` in [`../code/roles.py`](../code/roles.py). Its tag comes from `model_tag`, the lake grows a `judge=<tag>/` partition, and the EDA needs no change. |
| **A new arm** | Nothing. It is discovered from `data/conversations/`. |
| **A new arm-name field** | [`../code/naming.py`](../code/naming.py) only — it is the one parser, for the trainers and the EDA both. |
| **A results-layout change** | `exports._leaf` and nothing else. A second place that composes a path is a second layout. |
| **A new trainer phase** | `PHASE_KEYS` in [`../code/core/timing.py`](../code/core/timing.py); `data.TIMING_COLUMNS` follows automatically. |

⚠ **Families are self-contained.** Each recomputes what it needs from the score lake. Exp3 had one
family read another's *rendered* tables, which made render order load-bearing and raced the moment
the driver parallelised. The fix for "I need that number" is to compute it, not to read a sibling's
Markdown.

## Two things never to do to these numbers

1. **Pair on `persona_id`, never on file order.** The same 96 personas face every arm and every
   iteration, so every contrast here is repeated-measures — subtract *within* persona and analyse
   the deltas. Pairing by position (`a["score"].values - b["score"].values`) subtracts unrelated
   conversations, and the *mean* survives it (a permutation of the same 96 values has the same mean
   difference), which is exactly what makes it dangerous: nothing looks wrong, while `dz`, the
   bootstrap CI and the p-value are all garbage. Use `stats.paired_arrays`.
2. **MICI is lower-is-better.** It counts MI-*inconsistent* therapist behaviour, so a `mean_delta`
   of `-0.3` is a **gain**. Multiply by `constants.sign_of(metric)` before any `argmax`, sort,
   "best checkpoint" pick or diverging colour scale, and run raw contrasts through
   `stats.orient_contrast` wherever a table or figure says "better".

## Coming from Exp3?

Four things you would look for are **deliberately absent** — each worked around a defect that Exp4
fixes upstream:

- **No persona-shuffle replay.** Exp3 named conversations by a per-iteration *shuffled* processing
  index, so `conversation_3.csv` was a different person each iteration and every module re-derived
  `Random(seed + k + 1)` before it could pair anything. Exp4 names files by the **stable persona id**
  and stores `persona_id` as a column. The join key is simply there.
- **No parquet fold cache, manifest or content-signature machinery for the lake** — it is already
  one 96-row parquet per model state, not ~50k single-row CSVs. (`.eda_cache/` above is a different
  thing: a memo of *built frames*, not a re-fold of the lake.)
- **No mtime forensics for compute.** The trainer appends per-phase wall-clock to
  `runs/<ARM>/iteration_<N>/timing_sessions.jsonl` — and, for the training phase, a partial line at
  every checkpoint save — so `compute/cost` reads a log instead of reconstructing GPU-hours from
  artifact mtimes, and a preemption mid-training loses at most `save_steps` of wall-clock rather
  than the whole phase. `n_sessions_production > 1` means the iteration was
  resumed — exactly the case Exp3's per-process fields got wrong. ⚠ **Not `n_sessions > 1`:** the
  post-loop final-eval pass appends an eval-gen-only session to each arm's LAST iteration (and every
  `tools/generate_convs.py` repair appends another), so the raw session count reports every healthy
  arm as resumed — a caveat attached to precisely the endpoint every budget sweep is read at.
- **No `oracle=` path level.** Role tags are always encoded in `EXPERIMENT_NAME`, so the training
  oracle is already inside the arm name.

Also gone on purpose: `FAMILY_READS` (see "Families are self-contained"), per-judge render units,
and a "primary-only" family list.

One Exp3 convention is unchanged and still bites: `model_iter_<N>` names the policy that
**generated** conversations, `iteration_<N>` the training pass that **consumed** them, and the two
are off by one. **Join a training-side frame to a score frame on `state_index`, never on
`iteration`** — the full rule is in `data.py`'s module docstring.
