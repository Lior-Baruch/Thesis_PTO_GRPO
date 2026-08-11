# Trained on a Judge, Tested by Another

**Target:** CLPsych / NLP-for-psychology workshop (ACL style). **Domain:** Exp3, `L0` view
(PTO vs GRPO at matched look-ahead $K=0$). The look-ahead ($K$) comparison is deliberately
excluded — it is a separate paper on the `L5` view.

## The argument in one line

Both methods raise the LLM-judge reward a lot; PTO leads at the matched endpoint because
GRPO peaks and regresses; the gains in both arms come with a drift toward affirmation and
away from inquiry; a held-out grader credits 0.80 of PTO's Q1 gain and 0.28 of GRPO's; and
reading the training signal directly shows the method gap is about **exploration**, not
about DPO vs group-relative weighting.

## Files

```
main.tex               preamble + \input order. \drafttrue enables \todo{} / \note{}.
sections/              one file per section, numbered in reading order
  00_abstract  01_intro  02_related  03_method  04_setup  05_results
  06_hacking   07_validity  08_mechanism  09_discussion(+conclusion)
  10_limitations  11_ethics
  A_appendix     reliability tables, sign ladder, cost, reproducibility
  B_mechanism    the training-signal probe in full (body §8 is its summary)
refs.bib               bibliography (two entries flagged TODO — see NUMBERS.md)
figures/               PNGs copied from the EDA by sync_figures.py — do not edit by hand
NUMBERS.md             THE CLAIMS LEDGER: every number → the artifact it came from
sync_figures.py        re-copy figures after an eda/tools/render_views.py pass
```

## Length — measured, not estimated

**The body runs to 8 pages + ~47% of page 9, i.e.\ about half a page over the 8-page
limit.** (Limitations, Ethics, References and both appendices are unlimited and don't
count; the whole PDF is 15 pages.) Half a page is roughly one table or ~450 words. Word
counts are a bad proxy here — an earlier 6,840-word count implied ~900 words over, nearly
double the real gap — so **always measure from the compiled PDF**:

```powershell
& ..\..\.venv\Scripts\python.exe -c "import fitz; d=fitz.open('main.pdf'); t=' '.join(d[8].get_text().split()); i=t.find('Limitations'); print(f'body = 8 pages + {i/len(t):.0%} of p.9')"
```

Candidates for the cut, in order: §7 Validity is the longest section; §3's look-ahead
formalism is appendix material; §4's prose largely duplicates its own two tables.

## Building

Everything needed is here — `acl.sty` and `acl_natbib.bst` are vendored from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), and `acl.sty` sets
`\bibliographystyle{acl_natbib}` itself (which is why `main.tex` only sets a style in its
fallback branch).

**In VS Code:** LaTeX Workshop, recipe configured in the repo's `.vscode/settings.json`.
Build with `Ctrl+Alt+B` or on save. The engine is MiKTeX
(`winget install --id MiKTeX.MiKTeX -e`), installed 2026-08-10 with on-demand package
fetching enabled (`initexmf --set-config-value="[MPM]AutoInstall=1"`). MiKTeX puts itself on
the *user* PATH, so **VS Code must be restarted once** after installing it.

⚠ **Do not switch the recipe to `latexmk`.** MiKTeX's `latexmk` is a Perl script and MiKTeX
ships no Perl, so it fails with *"MiKTeX could not find the script engine 'perl'"*. The trap
is that it can look like it works: Git for Windows bundles a `perl` under
`C:\Program Files\Git\usr\bin`, but only `C:\Program Files\Git\cmd` is on the real PATH — so
that perl exists **only inside a Git Bash session**, and a build that succeeds in a terminal
will still fail in VS Code. The configured recipe calls `pdflatex`/`bibtex` directly and
needs no Perl. (If you want latexmk's build-until-converged behaviour:
`winget install --id StrawberryPerl.StrawberryPerl -e`, then pick the second recipe.)

**From a shell** — the same four passes, Perl-free:
```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
```
Three `pdflatex` passes are needed, not two: one to write `.aux`, one to absorb the `.bbl`
and place floats, one to settle the resulting page/reference numbers. Confirm convergence by
checking the last pass has no `undefined` in `main.log`.

Two preamble notes worth knowing before you touch it:
- `\usepackage{times}` is **required**, not cosmetic. Without it `microtype`'s font
  expansion fails on MiKTeX's bitmap Computer Modern with a fatal
  *"auto expansion is only possible with scalable fonts"* and no PDF is produced.
- `main.tex` still carries an `\IfFileExists{acl.sty}` fallback to single-column `article`,
  so the draft compiles even if the style files go missing. If output suddenly looks
  single-column, that's the fallback firing — check `acl.sty` is present.

Switch `\drafttrue` → `\draftfalse` in `main.tex` to hide all `\todo{}` markers, and
`\usepackage[review]{acl}` → `\usepackage{acl}` for the camera-ready (this removes the line
numbers and will change the page count).

## House rules for edits

Keep the log clean — the draft currently builds with **0 overfull boxes and 0 undefined
references**. Two-column ACL columns are narrow; if you widen a table, check:

```bash
& ..\..\.venv\Scripts\python.exe -c "import re;print([l for l in open('main.log',errors='replace') if 'Overfull' in l])"
```

## Keeping it honest

Every quantitative claim is in `NUMBERS.md` with the tracked artifact path it came from.
After any `render_views.py` rerun:

```powershell
& ..\..\.venv\Scripts\python.exe sync_figures.py --check   # did any figure move?
& ..\..\.venv\Scripts\python.exe sync_figures.py           # re-copy
```

then walk `NUMBERS.md` for anything that moved. Three claims are easy to get subtly wrong
and are called out there explicitly: the PTO affirmation-push peak is at **iteration 8**
(not 10), the Q2 top-item claim is "the same three items," not "the same top item," and the
≈\$300 OpenAI figure is **project-wide** and covers the K=5 arms this paper does not use.
