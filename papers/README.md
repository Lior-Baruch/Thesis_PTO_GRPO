# papers/ — paper drafts

One subfolder per paper. Papers span experiments, so they live at the repo root rather than
inside an `Exp*/` directory. Nothing here is imported by `code/` or `eda/` — a paper folder only
*reads* generated artifacts out of `Exp3_PTO_GRPO/eda/results/` and the hand-authored method
schematics in `Exp3_PTO_GRPO/figures/`.

| Folder | Paper | Domain | Status |
|---|---|---|---|
| [`2026_clpsych_mi_pto_grpo/`](2026_clpsych_mi_pto_grpo/) | *Trained on a judge, tested by another* — PTO vs GRPO for MI at K=0 | Exp3, `L0` view | **drafting** |
| [`2026_thesis_lookahead/`](2026_thesis_lookahead/) | *Does Looking Ahead Help?* — the look-ahead (K) chapter | **all three generations** (Exp1 + Exp2 + Exp3 `L5`) | **drafting** |

**The two papers split the levers cleanly and share no claims.** The CLPsych paper owns **PTO vs
GRPO at matched `K=0`** (the optimizer varies, look-ahead does not); the thesis chapter is **PTO
only** across three generations (look-ahead varies, the optimizer does not). If you change a shared
artifact, check both `NUMBERS.md` ledgers.

## Conventions

- **Numbers come from tracked artifacts, never from memory.** Each paper carries a `NUMBERS.md`
  ledger mapping every quantitative claim in the draft to the exact
  `Exp3_PTO_GRPO/eda/results/<view>/{tables,figures}/...` path it came from. If the EDA is
  re-rendered and a number moves, the ledger is how you find every sentence that has to change.
- **Figures are copied or generated, never symlinked** — either way the draft compiles standalone
  and a submitted PDF is frozen against later EDA reruns. The two papers get there differently:

  | Paper | How its `figures/` + `tables/` are produced | Re-run after |
  |---|---|---|
  | `2026_clpsych_mi_pto_grpo/` | `sync_figures.py` **copies** PNGs out of `eda/results/` | `render_views.py` |
  | `2026_thesis_lookahead/` | `analysis/crossgen.py` **generates** every table and figure (nothing hand-computed) — it is the cross-generation analysis, not a copier | any re-scoring |

- **`.pdf` and other LaTeX build output is gitignored; `.tex`, `.bib`, `NUMBERS.md`, the analysis
  script, and the produced `figures/` + `tables/` are tracked.** A PDF that was actually submitted
  or circulated is the exception — force-add it (`git add -f submitted/<name>.pdf`) so there is a
  record of what people saw.
- Which grader produced a figure is part of its path in the EDA (`<family>/<judge>/`). Keep that
  in the copied filename (`..._gpt-4o-mini.png`) so a figure in a draft always names its grader.
