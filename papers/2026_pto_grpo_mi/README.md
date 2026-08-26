# Same Lever, Different Optimizer — the four-arm (2×2) paper

**Target: ARR October 2026 cycle** (one review cycle feeding both **NAACL 2027** and
**COLING 2027**). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory
unnumbered Limitations section (excluded from the page limit), optional Ethics Statement
(excluded). `acl.sty` builds in `[review]` mode (line numbers, anonymized); switch to `[final]`
for camera-ready.

**The question.** The K-turn look-ahead lever and the optimizer family are usually chosen
independently. This paper shows they interact: the optimizer ranking flips with the reward
horizon (PTO wins at K=0, GRPO wins at K=5, both graders, DiD dz 0.79–0.97), budget reverses the
practical rankings again (PTO K=0 matches GRPO K=5's held-out gain at 6.3× less compute), the
horizon — not the optimizer — decides whether the policy learns to flatter, and the earlier
ICLR-era regime shows the same lever helping PTO, so the interaction is regime-scoped.

This is **P2** of `papers/BRAINSTORM_2026-08-25.md` — the full 2×2 the GRPO-scoped ICLR draft
(`papers/2026_grpo_lookahead_mi`, P1) deliberately does not import. It is a **fresh draft**: the
retired `archive/2026_lookahead_pto_grpo` draft contributed its vendored `acl.sty` +
`acl_natbib.bst` and nothing else; every number was read off the current results tree into
`NUMBERS.md` on 2026-08-26.

## Contents

- `main.tex` + `sections/*.tex` — the draft. Body sections 00–09, then `10_limitations`
  (mandatory for ARR) and `11_ethics`, then appendices A (supplementary results) and B (repro).
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

## Division of labour vs. P1 (`2026_grpo_lookahead_mi`)

| | P1 (ICLR 2027) | this paper (ARR Oct → NAACL/COLING 2027) |
|---|---|---|
| Arms | the two GRPO arms only | all four |
| Headline | look-ahead more than doubles GRPO's gain | the optimizer ranking flips with the horizon |
| Measurement stats | recomputed on the 22 GRPO states | the full 44-state grid |
| Behaviour | GRPO reward hack + questions substitution | both optimizers; the PTO advice relocation |
| Regime | not imported | the ICLR-era re-scoring section |

Shared numbers (the GRPO K lever, the saturation mechanism, compute totals) are cited from the
same EDA artifacts in both ledgers; the papers never cite each other's prose.
