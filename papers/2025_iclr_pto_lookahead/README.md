# 2025_iclr_pto_lookahead — *Preference Tree Optimization: Enhancing Goal-Oriented Dialogue with Look-Ahead Simulations*

Lior Baruch, Moshe Butman, Kfir Bar, Doron Friedman · ICLR 2025.

**PUBLISHED — FROZEN.** Nothing in this folder gets edited. It exists so the published paper sits
in `papers/` alongside the chapters and drafts still to be written, rather than only inside the
experiment directory that produced it.

## What's here

| Path | What |
|---|---|
| [`submitted/paper.pdf`](submitted/paper.pdf) | the published PDF, copied verbatim from [`Exp1_ICLR2025/paper.pdf`](../../Exp1_ICLR2025/paper.pdf) (both copies kept — Exp1 stays self-contained) |

**No `.tex` sources.** The LaTeX for this paper was never in this repo, so unlike a draft there is
nothing here to build — `submitted/paper.pdf` IS the artifact. `*.pdf` is gitignored by
[`../.gitignore`](../.gitignore), so this one is force-added (`git add -f`) under that file's
"a PDF that was actually submitted or circulated" exception.

## The experiment behind it

Llama-2-7B therapist, GPT-3.5 patient + oracle, PTO at K ∈ {0, 5} over 7 iterations,
reward = mean(Q1, Q2). Setup, layout, and re-run instructions live in
[`Exp1_ICLR2025/CLAUDE.md`](../../Exp1_ICLR2025/CLAUDE.md) — **not** duplicated here.

⚠ **Exp1 scores are not on the same axis as Exp2/Exp3** (different therapist base, patient prompts,
and oracle). See § "Data lineage" in the root [`CLAUDE.md`](../../CLAUDE.md).

## Citing it

This is where PTO and the look-ahead lever were introduced, so anything written in `papers/` cites
it — as prior work, and as generation 1 of the three.

🔲 **Open, and worth settling before the next draft's bibliography:** main conference vs. workshop
track. The PDF's header line reads "Published as a conference paper at ICLR 2025", but that string
is ICLR template boilerplate, and the project notes elsewhere describe this as the workshop paper.
Resolve against the acceptance email / OpenReview page.
