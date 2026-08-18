# Exp3_PTO_GRPO — the active experiment (map of this folder)

Llama-3.2-1B therapist vs gpt-4o-mini patient/oracle; PTO vs GRPO under matched look-ahead K ∈ {0, 5}
and MCL=12, four arms, scored on two graders. **This file is only a map.** The spec, the numbers and
the history each have exactly one owner:

| You want | Read |
|---|---|
| **The spec** — methods, algorithms, trainer internals, `EXPERIMENT_NAME` schemes, data schemas, gotchas | the root [`CLAUDE.md`](../CLAUDE.md) (its "Exp3_PTO_GRPO — the active experiment" half; there is deliberately no `Exp3_PTO_GRPO/CLAUDE.md`) |
| **The numbers** — run status, headline results, cost constraint, next step | the root [`STATUS.md`](../STATUS.md) |
| What each `code/` module does, how to run the trainers/tools locally | [`code/README.md`](code/README.md) |
| How the EDA works — families, `EdaConfig`, exports, render tool, judge dimension, module map | [`eda/README.md`](eda/README.md) |
| The written analysis per research question | `eda/results/<top>/SUMMARY.md` (hand-authored) beside the auto `INDEX.md` |
| Metric definitions · measurement limitations | [`eda/results/METRICS_REFERENCE.md`](eda/results/METRICS_REFERENCE.md) · [`eda/results/LIMITATIONS.md`](eda/results/LIMITATIONS.md) |
| Dated history (status, EDA passes, trainer changes) | [`history/CHANGELOG.md`](history/CHANGELOG.md) (index) |

## Layout

```
Exp3_PTO_GRPO/
├── README.md              this map
├── code/                  the trainers — see code/README.md
│   ├── GRPO_Exp3/             train_GRPO_Iterative.ipynb + grpo_trainer.py
│   ├── PTO_Exp3/              train_PTO_Iterative.ipynb + pto_trainer.py
│   ├── _shared/               runtime, model, convs, reward (batched K-turn look-ahead), tb_plots,
│   │                          eda_recorder, timing, lookahead_check — BOTH trainers import these
│   ├── tools/                 _local_smoke.py (offline smoke) · generate_eval_convs.{py,ipynb}
│   │                          (generate-only pass for one model state / a replicate draw)
│   ├── system_prompts_builder.py · questionnaires.py     the CANONICAL copies (the EDA imports them)
│   └── roles.py               patient / oracle / judge bindings + the arm-naming contract
├── data/                  ALL Google Drive symlinks, gitignored — schemas live in CLAUDE.md
│   ├── eval_scores/           THE SCORE LAKE: judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<id>.csv
│   │                          (+ _parquet/ fold, _batches/ manifests, _crossgen/ re-scored Exp1)
│   ├── grpo_Exp3/  pto_Exp3/  runs/<MODE>/<EXP>/iteration_N/{adapter,training[,pref_pairs],eda}/
│                              + conversations/<MODE>/<EXP>/model_iter_<N>_TT*_TP*/
├── eda/                   the analysis — see eda/README.md
│   ├── eda_analysis/          the package: analysis modules on a `constants` leaf + plotting/ + scoring/
│   ├── notebooks/             ONE notebook per results family, notebooks/<top>/<sub>.ipynb:
│   │   ├── arms/                  outcomes questionnaires validity heterogeneity training preference stats
│   │   ├── lookahead/             reward transfer behaviour mechanism replication      (RQ-i, K=0 vs K=5)
│   │   ├── method/                contrast                                             (RQ-ii, PTO vs GRPO)
│   │   ├── compute/               cost                                                 (GPU-h + API axis)
│   │   ├── measurement/           validity                                             (judge validity)
│   │   └── scoring/               Run_Eval · Judge_Reliability · Local_Judge_Validation (the PAID side)
│   ├── tools/                 render_results.py · consolidate_scores.py · score_crossgen.py ·
│   │                          strip_notebook_outputs.py
│   └── results/               the deliverable tree, mirrored on the notebooks:
│       ├── INDEX.md               auto — one line per family
│       ├── METRICS_REFERENCE.md · LIMITATIONS.md    hand-authored reference docs
│       ├── schematics/            hand-authored METHOD diagrams (build_method_figures.py + CAPTIONS.md)
│       ├── arms/<sub>/{figures,tables}/<judge>/…      per-arm descriptives, all four arms, ONE LEAF PER GRADER
│       ├── lookahead/<sub>/{figures,tables}/…         both graders inside each artifact (no judge level)
│       ├── method/contrast/ · compute/cost/ · measurement/validity/   likewise judge-invariant
│       └── <top>/{SUMMARY.md, INDEX.md}               hand-authored narrative + auto artifact map per top
└── history/               CHANGELOG_STATUS.md · CHANGELOG_EDA.md · CHANGELOG_TRAINER.md behind CHANGELOG.md
```

## How the pieces connect

1. **Train** on Colab: `code/<METHOD>_Exp3/train_<METHOD>_Iterative.ipynb` (cell 1 = flat globals;
   the per-iteration loop is visible in the notebook). Each iteration regenerates 96 conversations
   from the current policy — those convs ARE the eval set of the previous model state — then updates
   the adapter. Outputs land in `data/{grpo,pto}_Exp3/` on Drive.
2. **Score** locally: `eda/notebooks/scoring/Run_Eval.ipynb` auto-discovers every arm on disk and
   writes the primary oracle's scores into the score lake; `Judge_Reliability.ipynb` adds a held-out
   grader (Claude Haiku 4.5) as another `judge=<tag>` partition. Paid; never part of a render.
3. **Analyse**: `python tools/render_results.py` (from `eda/`) executes every family notebook —
   `arms/*` once per grader on disk, the four judge-invariant tops once — and regenerates
   `eda/results/`. Free; fully reproducible from the lake. `python -m eda_analysis._selfcheck` guards
   the package after any EDA change.
4. **Write**: `results/<top>/SUMMARY.md` is the interpretation, the tables under it are the evidence;
   `STATUS.md` carries the headline; `papers/` and `meetings/` (repo root) read `eda/results/` and
   never the other way round.

Both `code/` and `eda/` are self-contained: `eda_analysis/constants.py` prepends `code/` to
`sys.path` so the rubrics and patient prompts are imported from their single canonical copies, and
nothing in `code/` imports the EDA. Absolute scores are Exp3-internal (bf16) — do not compare them
with Exp2's 4-bit numbers; see CLAUDE.md § "Data lineage".
