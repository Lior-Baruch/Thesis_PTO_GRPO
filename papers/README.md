# papers/ — paper drafts

One subfolder per paper. Papers span experiments, so they live at the repo root rather than
inside an `Exp*/` directory. Nothing here is imported by `code/` or `eda/` — a paper folder only
*reads* generated artifacts out of `Exp3_PTO_GRPO/eda/results/` and the hand-authored method
schematics in `Exp3_PTO_GRPO/figures/`.

| Folder | Paper | Domain | Status |
|---|---|---|---|
| [`2025_iclr_pto_lookahead/`](2025_iclr_pto_lookahead/) | *Preference Tree Optimization: Enhancing Goal-Oriented Dialogue with Look-Ahead Simulations* | Exp1 | **published — frozen** (ICLR 2025 SSI-FM workshop, poster) |
| [`2026_clpsych_mi_reward_hacking/`](2026_clpsych_mi_reward_hacking/) | *Affirmation Without Inquiry: Reward Hacking When an LLM Judge Trains a Motivational Interviewing Therapist* | Exp3, `L0` view | **drafting** (CLPsych / ACL style, 8pp) |
| [`2026_lookahead_hack_substitution/`](2026_lookahead_hack_substitution/) | *The Hack Moves: Trajectory-Level Reward Redirects Rather Than Reduces Reward Hacking in a Motivational Interviewing Therapist* | Exp3, `L5` view | **drafting** (CLPsych / ACL style, 8pp) |

## Scope of the two current drafts — they must share no claims

They are split on the **K axis**, and the split is what keeps their ledgers disjoint:

- **`2026_clpsych_mi_reward_hacking/` is K=0 only.** The optimizer (PTO vs GRPO) is the one thing
  that varies. It establishes *that* the LLM-judge reward is hacked, and through which channel.
- **`2026_lookahead_hack_substitution/` is PTO only, K ∈ {0,5}.** The optimizer is held fixed and
  K is the one thing that varies. It tests look-ahead as a *mitigation* and finds it relocates the
  hack rather than reducing it. ⚠ State the aggregate carefully: at the matched endpoint the
  MI-inconsistent per-session total is unchanged **under the held-out judge** (dz 0.10, ns) while
  the training oracle scores it reduced (dz 0.45) — the paper claims the substitution, never the
  reduction, and the judge-dependence of the total IS its second finding.

Neither is a K×method result — but that is now a **scope choice, not a data limitation**. GRPO LA5
trained iterations 1–5 and is scored 0–5 on both graders (39 scored model states in all, per
[`STATUS.md`](../STATUS.md)), so a K×method contrast exists over iterations 0–5 and is already
rendered in `Exp3_PTO_GRPO/eda/results/L5/tables/7_stats/*/k_paired_by_method.md`. Both drafts
state their claims at iteration 10, outside that window, and both say so in Limitations. **Before touching a shared artifact, check both
`NUMBERS.md` ledgers**; the over-praise numbers in particular appear in both, at different
iterations and under different framings, and are easy to cross-contaminate.

⚠ **They read different views.** The K=0 draft reads `results/L0/`; the look-ahead draft reads
`results/L5/`, which is `eda_analysis.RQ_I_VIEW` — the only view whose K-contrast notebook
sections execute at all. Each `sync_figures.py` points at its own view; do not "fix" one to match
the other.

## Conventions

- **Numbers come from tracked artifacts, never from memory.** Each paper carries a `NUMBERS.md`
  ledger mapping every quantitative claim to the exact
  `Exp3_PTO_GRPO/eda/results/<view>/{tables,figures}/...` path it came from. If the EDA is
  re-rendered and a number moves, the ledger is how you find every sentence that has to change.
  It also records the traps — claims that are easy to get subtly wrong — as ⚠ callouts.
- **Figures are copied, never symlinked**, by each paper's `sync_figures.py`, so the draft
  compiles standalone and a submitted PDF is frozen against later EDA reruns. Re-run it after
  every `eda/tools/render_views.py` pass; it exits non-zero if a *source* figure vanished, which
  is the failure worth catching.
- Which grader produced a figure is part of its path in the EDA (`<family>/<judge>/`). Keep that
  in the copied filename (`..._gpt-4o-mini.png`) so a figure in a draft always names its grader.
- **`.pdf` and other LaTeX build output is gitignored; `.tex`, `.bib`, `NUMBERS.md`, and the
  produced `figures/` + `tables/` are tracked.** A PDF that was actually submitted or circulated
  is the exception — force-add it (`git add -f submitted/<name>.pdf`) so there is a record of
  what people saw.
- **Cite the ICLR paper as the SSI-FM *workshop* poster, not the main conference.** Its PDF
  header says "Published as a conference paper at ICLR 2025" — that is stock template
  boilerplate. The canonical BibTeX is in
  [`2025_iclr_pto_lookahead/README.md`](2025_iclr_pto_lookahead/README.md).

## Building (MiKTeX on Windows)

Both the vendored `acl.sty` and `acl_natbib.bst` live inside each paper folder, so a draft builds
with no network round-trip. Four passes, no Perl, **no `latexmk`** (MiKTeX ships no Perl, and the
`perl` bundled with Git for Windows is only on the PATH inside a Git Bash session — so a build
that works in a terminal can still fail in VS Code):

```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```

Three `pdflatex` passes, not two: one to write `.aux`, one to absorb the `.bbl` and place floats,
one to settle the resulting page and reference numbers.

## History

Two drafts (`2026_clpsych_mi_pto_grpo/`, `2026_thesis_lookahead/`) were deleted in `5545be5`
because the writing wasn't what it needed to be. They remain recoverable from that commit's
parent, and the current draft reuses their vendored style files and `refs.bib`. Every number they
cited came from tracked artifacts under `Exp3_PTO_GRPO/eda/results/`, which were untouched.
