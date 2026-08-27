# Same Lever, Different Optimizer — the four-arm (2×2) paper

**THE submission — the single live paper** (decided with the supervisors on 2026-08-27; the
GRPO-only companion P1 was archived the same day, its ICLR 2027 plan dropped).

**Target: ARR October 2026 cycle** (submission 2026-10-12, commitment 2026-12-20; one review
cycle feeding both **NAACL 2027** and **COLING 2027** — the venue is chosen in December once
reviews exist). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory
unnumbered Limitations section (excluded from the page limit), optional Ethics Statement
(excluded). `acl.sty` builds in `[review]` mode (line numbers, anonymized); switch to `[final]`
for camera-ready.

**Axis decision (2026-08-27): ITERATIONS ONLY.** Per Lior + supervisors, the paper does not
argue on a GPU-hour or samples/budget axis — the old §5 (cost) is gone, the budget appendix
material with it (all of it remains in the EDA under `results/compute/cost/`). What replaced it:
the two-winners head-to-head lives in §4 (`ssec:winners`, iteration-matched), and Limitations
carries the matched-iterations≠matched-data disclosure (GRPO consumes ~3.0×/2.4× PTO's oracle
scoring calls per ten iterations at K=0/K=5 — arithmetic in `NUMBERS.md`). The Ethics statement
keeps a one-line ≈107 GPU-h total (standard resource disclosure), with no per-arm breakdown.

**The question.** The K-turn look-ahead lever and the optimizer family are usually chosen
independently. This paper shows they interact: the optimizer ranking flips with the reward
horizon (PTO wins at K=0, GRPO wins at K=5, both graders, DiD dz 0.79–0.97), the top of the grid
is grader-conditional (the training oracle scores GRPO K=5 best; the held-out judge reads it as
tied with PTO K=0, a null that replicates), the horizon — not the optimizer — decides whether
the policy learns to flatter, and the earlier ICLR-era regime shows the same lever helping PTO,
so the interaction is regime-scoped.

This is **P2** of `papers/BRAINSTORM_2026-08-25.md`. It is a **fresh draft**: the retired
`archive/2026_lookahead_pto_grpo` draft contributed its vendored `acl.sty` +
`acl_natbib.bst` and nothing else; every number was read off the current results tree into
`NUMBERS.md` on 2026-08-26.

## Contents

- `main.tex` + `sections/*.tex` — the draft. Body sections 00–09 (**no 05**: the cost section
  was deleted with the 2026-08-27 iterations-only decision; the numbering gap is deliberate),
  then `10_limitations` (mandatory for ARR) and `11_ethics`, then appendices A (supplementary
  results) and B (repro).
- `NUMBERS.md` — the claim → artifact ledger. Nothing enters the `.tex` that is not a row there.
- `sync_figures.py` — copies every referenced figure from `Exp3_PTO_GRPO/eda/results/` into
  `figures/` (never symlinks). **Figure policy: four-arm everywhere** (this paper's subject is
  the 2×2), and **levels over deltas** wherever a level artifact exists.
- `refs.bib` — superset of the P1 bibliography + the DPO-vs-PPO comparison.

## Build (MiKTeX on Windows — see ../README.md)

```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```

## Relation to P1 (`../2026_grpo_lookahead_mi`)

P1 — the GRPO-only look-ahead paper — was archived on 2026-08-27 when its ICLR plan was dropped,
then **revived the same day (per Lior) as a second submission to the same ARR October cycle**,
telling the GRPO-with-look-ahead story with PTO cited as the lever's origin but never as data.
The GRPO K-lever numbers therefore appear in both papers — there as the subject (its §§5–7 go
deeper: the reward hack, the mechanism analysis, the full saturation analysis on the 22 GRPO
states), here as cells of the interaction; this paper uses the full 44-state grid for measurement. The papers never cite each
other's prose. ⚠ **Two same-cycle ARR submissions from one experiment need supervisor sign-off
against ARR's multiple-submission policy** — see P1's README for the overlap inventory.
