# papers/ — paper drafts

One subfolder per paper. Papers span experiments, so they live at the repo root rather than
inside an `Exp*/` directory. Nothing here is imported by `code/` or `eda/` — a paper folder
only *reads* generated artifacts out of `Exp3_PTO_GRPO/eda/results/` (via each paper's
`sync_figures.py`) and the hand-authored method schematics in `Exp3_PTO_GRPO/figures/`.

| Folder | Paper | Domain | Status |
|---|---|---|---|
| [`2026_clpsych_mi_pto_grpo/`](2026_clpsych_mi_pto_grpo/) | *Trained on a judge, tested by another* — PTO vs GRPO for MI at K=0 | Exp3, `L0` view | **drafting** |
| `2026_lookahead/` (not yet created) | The look-ahead (K) paper | Exp3, `L5` view | planned |

## Conventions

- **Numbers come from tracked artifacts, never from memory.** Each paper carries a `NUMBERS.md`
  ledger mapping every quantitative claim in the draft to the exact
  `Exp3_PTO_GRPO/eda/results/<view>/{tables,figures}/...` path it came from. If the EDA is
  re-rendered and a number moves, the ledger is how you find every sentence that has to change.
- **Figures are copied, not symlinked.** `sync_figures.py` copies the PNGs a paper uses into its
  own `figures/` so the draft compiles standalone and a submitted PDF is frozen against later EDA
  reruns. Re-run it after `render_views.py`.
- **`.pdf` build output is gitignored; `.tex`, `.bib`, `NUMBERS.md` and the copied `figures/` are
  tracked.**
- Which grader produced a figure is part of its path in the EDA (`<family>/<judge>/`). Keep that
  in the copied filename (`..._gpt-4o-mini.png`) so a figure in a draft always names its grader.
