# papers/ — paper drafts

One subfolder per paper. Papers span experiments, so they live at the repo root rather than
inside an `Exp*/` directory. Nothing here is imported by `code/` or `eda/` — a paper folder only
*reads* generated artifacts out of `Exp3_PTO_GRPO/eda/results/` (including the hand-authored
method schematics under `Exp3_PTO_GRPO/eda/results/schematics/`).

| Folder | Paper | Domain | Status |
|---|---|---|---|
| [`2025_iclr_pto_lookahead/`](2025_iclr_pto_lookahead/) | *Preference Tree Optimization: Enhancing Goal-Oriented Dialogue with Look-Ahead Simulations* | Exp1 | **published — frozen** (ICLR 2025 SSI-FM workshop, poster) |
| [`2026_grpo_lookahead_mi/`](2026_grpo_lookahead_mi/) | *Scoring the Continuation: $K$-Turn Look-Ahead Rewards for Group-Relative Policy Optimization in Motivational Interviewing* | Exp3, the **two GRPO arms only**, both graders, both cost axes | **drafting** — the live draft |
| [`archive/2026_lookahead_pto_grpo/`](archive/2026_lookahead_pto_grpo/) | *Same Lever, Different Optimizer: Does $K$-Turn Look-Ahead Help a Small Motivational-Interviewing Therapist?* | Exp3, all four arms (both K, both optimizers, both graders); reads `results/{lookahead,compute}` | **retired 2026-08-25** — drafted while GRPO K=5 was right-censored at iteration 5; that arm has since finished at 10 (all states scored), so its endpoint/retention/iso-compute claims read a stale grid. Its `analysis/out/` remains the EDA's frozen fixture |
| [`archive/2026_clpsych_mi_reward_hacking/`](archive/2026_clpsych_mi_reward_hacking/) | *Affirmation Without Inquiry: Reward Hacking When an LLM Judge Trains a Motivational Interviewing Therapist* | Exp3, `L0` view | **retired 2026-08-18** — its K=0 reward-hacking result was absorbed by *Same Lever*'s §6 |
| [`archive/2026_lookahead_hack_substitution/`](archive/2026_lookahead_hack_substitution/) | *The Hack Moves: Trajectory-Level Reward Redirects Rather Than Reduces Reward Hacking in a Motivational Interviewing Therapist* | Exp3, `L5` view, PTO only | **retired 2026-08-18** — its substitution result was absorbed by *Same Lever*'s §6; its `NUMBERS.md` traps still apply |

**`archive/` holds retired drafts, tracked** (un-ignored in `.gitignore` exactly like
`meetings/archive/`). They are records of what was argued and how far the writing got, not live
work: do not edit them in place. Their ledgers (`NUMBERS.md`) remain the fastest way to find the
trap list for a number that appears in the live draft too.

## Scope of the live draft

[`2026_grpo_lookahead_mi/`](2026_grpo_lookahead_mi/) is the **one** live paper. It takes the
narrowest question the completed grid answers decisively — does scoring a candidate turn by its
$K$-turn continuation help a **group-relative** optimizer — and answers it on the two GRPO arms
across both graders and both cost axes. Planned in
[`BRAINSTORM_2026-08-25.md`](BRAINSTORM_2026-08-25.md), which reads the re-rendered tables cold
and ranks five candidate papers; this is P1, chosen to go first because it is self-contained,
all-positive, and needs no new spend. The full $2\times2$ (the K $\times$ optimizer interaction,
where the lever's sign flips) is P2 and is deliberately **not** folded in — it is a different,
larger paper and mixing them would weaken both.

Every earlier Exp3 draft was retired to `archive/` because it predates the completed grid
(GRPO K=5 finished at iteration 10 on 2026-08-25). A new draft starts from a fresh cold read of
`Exp3_PTO_GRPO/eda/results/` (CLAUDE.md § "Epistemic status" rule 1), never by patching an
archived one; the archived ledgers remain the trap list for any shared number.

**The cross-K analysis machinery is EDA-owned, not paper-owned.** The four-arm persona-paired K
contrast under both graders, the cross-K judge test, the compute sweeps on the channels, the
look-ahead tail audit, the ICLR conversations re-scored under the modern oracle, … are rendered
by the tracked EDA's `lookahead/` and `compute/` families
(`Exp3_PTO_GRPO/eda/results/lookahead/{reward,transfer,behaviour,mechanism,replication}/`,
`results/compute/cost/`; notebooks under `eda/notebooks/{lookahead,compute}/`, modules
`eda_analysis.{lookahead,transfer,compute,tails,dispersion,faithfulness,crossgen,replication,
instruments}`). *Same Lever*'s `analysis/*.py` generators, which first produced those numbers,
were retired on 2026-08-18 (git `abe5cb3`) once promoted; the archived paper still carries their
output as a **frozen fixture** (`archive/2026_lookahead_pto_grpo/analysis/out/*.json` ledgers +
`tables/*.md|csv`, seed-0 bootstrap CIs) that the EDA self-check (`paper fixture anchors`)
asserts against — the fixture pins the promoted modules' behaviour, not current results.
Each paper's `sync_figures.py` copies every figure its `.tex` references from the results tree.

## Conventions

- **Numbers come from tracked artifacts, never from memory.** Each paper carries a `NUMBERS.md`
  ledger mapping every quantitative claim to the exact
  `Exp3_PTO_GRPO/eda/results/<family>/{tables,figures}/...` path it came from. If the EDA is
  re-rendered and a number moves, the ledger is how you find every sentence that has to change.
  It also records the traps — claims that are easy to get subtly wrong — as ⚠ callouts.
- **Figures are copied, never symlinked**, by each paper's `sync_figures.py`, so the draft
  compiles standalone and a submitted PDF is frozen against later EDA reruns. Re-run it after
  every `eda/tools/render_results.py` pass; it exits non-zero if a *source* figure vanished, which
  is the failure worth catching.
- Which grader produced a per-judge figure is part of its path or name in the EDA
  (`arms/<sub>/figures/<judge>/`, or a `_<judge>` suffix inside the judge-invariant families).
  Keep that in the copied filename (`..._gpt-4o-mini.png`) so a figure in a draft always names
  its grader; judge-invariant figures (both graders inside) carry no judge segment.
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
