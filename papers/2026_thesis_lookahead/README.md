# Does Looking Ahead Help?

**Target:** thesis chapter (no page limit). **Domain:** all three experiment generations —
`Exp1_ICLR2025`, `Exp2_PTO`, `Exp3_PTO_GRPO` (the `L5` view).

Companion to [`../2026_clpsych_mi_pto_grpo/`](../2026_clpsych_mi_pto_grpo/), which owns the
**PTO vs GRPO** comparison at matched `K=0`. That paper excludes the look-ahead lever; this
chapter is that excluded half. The split is clean in both directions: **this chapter is PTO
only** (`crossgen.py::EXP3_METHODS`), so the optimizer never varies here and look-ahead never
varies there. The two share no claims — check both if you change a shared artifact.

## The argument in one line

Look-ahead simulation clearly helped in the published ICLR 2025 experiment, did nothing across
105 paired contrasts in the second generation, and never leads across eight matched iterations
and two independent graders in the third — and neither a weak statistical test nor the change
of grader explains the reversal, because a stricter test *strengthens* the original result and
the modern oracle *reproduces* it on the original conversations.

## What is new here (not reused from the experiments' own EDA)

1. **Exp1 and Exp2 had never been analysed with a persona-paired, matched-iteration test.**
   Both turn out to be pairable — `conversation_N.csv` is the same persona across arms and
   iterations in both (verified; Exp3 is the odd one out, reshuffling each iteration).
2. **Exp1 was re-graded onto the Exp3 measurement axis.** 1,344 conversations, 2,880 oracle
   calls, 0 errors, ≈$1.5. This is the control that separates "the grader changed" from "the
   experiment changed", and it is what makes a three-generation comparison legitimate.
3. **Exp2 and Exp3 were shown to already share an axis** — byte-identical Q1/Q2 prompts,
   schemas and labels, same grader model — so only Exp1 needed re-scoring.

## Files

```
main.tex               preamble + \input order. \drafttrue enables \todo{}.
sections/              one file per section, numbered in reading order
  00_abstract   01_intro       02_background  03_generations  04_method
  05_gen1       06_gen2        07_gen3        08_mechanism
  09_discussion 10_limitations (+ conclusion)
analysis/crossgen.py   THE analysis. Produces every table and figure. Nothing is hand-computed.
tables/  figures/      GENERATED — do not edit by hand; re-run crossgen.py
NUMBERS.md             THE CLAIMS LEDGER: every number -> its artifact, plus open TODOs
refs.bib               copied from the companion paper (two entries carry TODO notes)
```

## Regenerating

```powershell
# tables + figures (free, no API calls)
& ..\..\.venv\Scripts\python.exe analysis\crossgen.py

# the paid step — only needed once; resume-safe, skips existing CSVs
& ..\..\.venv\Scripts\python.exe ..\..\Exp3_PTO_GRPO\eda\tools\score_crossgen.py --gen exp1 --dry-run
& ..\..\.venv\Scripts\python.exe ..\..\Exp3_PTO_GRPO\eda\tools\score_crossgen.py --gen exp1
```

Re-scored Exp1 scores land in a **separate `_crossgen` partition** of the Exp3 score lake:

```
Exp3_PTO_GRPO/data/eval_scores/_crossgen/judge=<tag>/rep=0/metric=<M>/oracle=<O>/<Model>/<id>.csv
```

The `_crossgen` prefix is deliberate: nothing in the Exp3 analysis layer globs it (arm discovery
reads conversations under `data/{grpo,pto}_Exp3/`, and judge partitions resolve by explicit tag),
so **no tracked Exp3 result can move because this chapter exists**. It still sits inside the
Drive-backed `eval_scores` symlink, so the spend is backed up — same convention as the existing
`_parquet/` and `_batches/` directories.

## Building

Same toolchain as the companion paper — MiKTeX, no Perl, no `latexmk`:

```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```

Currently builds at **20 pages, 0 overfull boxes, 0 undefined references**. `\usepackage{times}`
is required, not cosmetic — without it `microtype`'s font expansion fails fatally on MiKTeX's
bitmap Computer Modern.

Unlike the companion paper this uses the `report` class with `\chapter`, not `acl.sty`. To fold
it into the thesis, drop the preamble and `\input` the section files under the thesis's own
`\chapter` — the section files all start at `\section`, so they nest without edits.

## Keeping it honest

`NUMBERS.md` distinguishes two classes of number: **class A** (outcome contrasts, computed by
`crossgen.py`) and **class B** (§8's training-signal results, which come from Exp3's own EDA and
are cited rather than recomputed). §8 states this in the text as well. Three claims are easy to
get subtly wrong and are flagged there explicitly:

- The **0/8** figure is the **held-out judge only**; the primary oracle is 7/8 by sign, with the
  eighth cell a −0.002 dead heat. Never write "0 of 8 under either grader".
- The moderator correlation is **not significant** (n = 5 arms, p = 0.17) and **one arm
  contradicts it** (Exp2/WAI-SR). It is labelled an interpretation everywhere it appears.
- **Absolute levels are not comparable across generations** (4-bit vs bf16 depresses Exp2 by
  ≈0.6). Every claim is a within-generation paired contrast, and the moderator uses gains and
  slopes relative to each generation's own base for this reason.

`NUMBERS.md` carries the remaining open TODOs. The two that concerned the chapter's own
integrity are closed: cross-grader agreement is now a tracked artifact
(`tables/t11_exp1_grader_agreement.md`), and the byte-identical-prompt premise behind §4.1 is
asserted at run time by `crossgen.py::verify_shared_axis`, which raises if either
`questionnaires.py` drifts. What remains is housekeeping in the *Exp3* tree rather than here:
`results/L5/SUMMARY.md` still narrates the superseded "arms tie at iteration 5" reading, and
the L5 view's tracked tables stop at iteration 7 while this chapter uses iteration 8 — run
`python tools/render_views.py L5` so the two agree.
