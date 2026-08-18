# Affirmation Without Inquiry

*Reward Hacking When an LLM Judge Trains a Motivational Interviewing Therapist*

**Target:** CLPsych / NLP-for-psychology workshop (ACL style, 8-page body).
**Domain:** Exp3, `L0` view only — PTO vs GRPO at matched look-ahead $K=0$.
The look-ahead comparison is deliberately excluded (see [`../README.md`](../README.md)).

## The argument in one line

An LLM-judge reward takes a 1B therapist from below basic competence into the fair-to-good MITI
band, but what it teaches is affirmation without inquiry — and a held-out judge from another
model family credits 0.80 of one optimizer's Q1 gain and only 0.28 of the other's.

## The three moves

The paper is **one claim with three moves**, not five parallel claims. PTO-vs-GRPO is not a
section — it is the contrast variable running through all three.

1. **§4 The loop works on its own terms.** Large effects on all five global rubrics; MITI
   relational crosses "good" in both arms.
2. **§5 What it actually learned.** Turns 2.3–3.4× longer, affirmations 5–6× up, MICI 2.3×
   (PTO) / 4.0× (GRPO), questions collapse. The load-bearing detail is the **regex-vs-oracle
   question-rate divergence**: the two measures agree at base and *invert* in GRPO by the
   endpoint, so part of the apparent competence is a property of the instrument.
3. **§6 How much is real.** Gain retention against a decoupled second judge. Non-overlapping on
   Q1 (0.80 vs 0.28), overlapping on Q2 (the control), and an **onset curve** that separates the
   arms four iterations before the reward curve does.

## Files

```
main.tex               preamble + \input order. \drafttrue enables \todo{} / \note{}.
sections/              one file per section, in reading order
  00_abstract  01_intro  02_related  03_setup  04_gains  05_learned
  06_heldout   07_discussion  08_limitations  09_ethics
  A_measurement  B_probe  C_repro
tables/                hand-built .tex tables (instruments, main_results, thresholds,
                       behaviour, retention) — every cell sourced in NUMBERS.md
figures/               PNGs copied from the EDA by sync_figures.py — do not edit by hand
NUMBERS.md             THE CLAIMS LEDGER: every number -> the artifact it came from
sync_figures.py        re-copy figures after an eda/tools/render_views.py pass
refs.bib               bibliography (2 entries unverified — see NUMBERS.md TODOs)
acl.sty acl_natbib.bst vendored from acl-org/acl-style-files
```

## Length — measured, not estimated

**The body fills exactly 8 pages** — the Limitations heading opens at the very top of p.9
(measured 2026-08-18, after the Discussion gained the compute-axis paragraph, Figure 3 became
the Q1-only panel with the 7-panel grid moved to Appendix A, and matching cuts were taken:
Discussion ¶1 deduplicated against Appendix B, the compounding-loop paragraph folded into the
mechanism reading, §5/§6 prose tightened, body figures at `0.9\columnwidth`, float separations
reduced in the preamble) — with **0 overfull boxes and 0 undefined references**; the whole PDF
is 14 pages. Limitations, Ethics, References and the three appendices are unlimited and don't
count. Word counts are a bad proxy for two-column ACL — always measure from the compiled PDF:

```powershell
& ..\..\.venv\Scripts\python.exe -c "import fitz; d=fitz.open('main.pdf'); [print(f'p.{i+1} {j/len(t):.0%}') for i,p in enumerate(d) for t in [' '.join(p.get_text().split())] for j in [t.find('Limitations')] if j>=0]"
```

⚠ That search also matches §5's cross-reference to the Limitations section on p.6 — the
**last** hit is the real heading.

It is at the limit with no slack, so **any addition needs a matching cut.** Where the slack was
already taken: §4's threshold prose was folded into Table 3's caption, §6's measurement detail
into Appendix A, and Discussion's "what we did not vary" into Limitations (which is unlimited —
moving body content there is the cheapest legitimate win).

Check overfull boxes after any table edit:

```bash
& ..\..\.venv\Scripts\python.exe -c "print([l for l in open('main.log',errors='replace') if 'Overfull' in l])"
```

## Building

See [`../README.md`](../README.md) § Building — four passes, MiKTeX, no `latexmk`. Two preamble
notes worth knowing before touching it:

- `\usepackage{times}` is **required, not cosmetic.** Without it `microtype`'s font expansion
  fails fatally on MiKTeX's bitmap Computer Modern and no PDF is produced.
- `main.tex` carries an `\IfFileExists{acl.sty}` fallback to single-column `article`. If the
  output is suddenly single-column, that fallback fired — check `acl.sty` is present.

Switch `\drafttrue` → `\draftfalse` to hide `\todo{}` markers, and `\usepackage[review]{acl}` →
`\usepackage{acl}` for camera-ready (removes line numbers, changes the page count). In review
mode the author block renders as "Anonymous ACL submission" regardless of `\author`.

## Keeping it honest

Every quantitative claim is in [`NUMBERS.md`](NUMBERS.md) with its artifact path. After any
`render_views.py` rerun:

```powershell
& ..\..\.venv\Scripts\python.exe sync_figures.py --check   # did any figure move?
& ..\..\.venv\Scripts\python.exe sync_figures.py           # re-copy
```

then walk the ledger. Seven claims are easy to get subtly wrong and are flagged there explicitly:

- **Retention intervals are non-overlapping at iterations 9–10 ONLY** — not at best-vs-best.
  An earlier revision claimed otherwise. The robust statement is the *ordering* (PTO above GRPO
  at every iteration from 4 on), not a peak-vs-peak significant contrast.
- **The two question rates are different measures.** `L0/SUMMARY.md` §4 quotes the regex rate
  (0.83→0.15) while citing the table that holds the *oracle* rate (0.446→0.319). Always say which.
- **The top Q2 items are NOT "the same three in both arms"** — an earlier revision of §5 said so.
  Self-disclosure tops only GRPO; PTO's top two are warmth items (*shoes* +1.54, *cared for*
  +1.53). What survives: self-disclosure + direction items sit in the top four of both.
- **The PTO affirmation push peaks at iteration 8**, not 10 (iter 10 falls back to 0.039).
- **Never average the two graders' raw scores** — train-vs-test, not two raters.
- **Don't resurrect `wins_correct` 0.65→0.71** — it was in-sample.
- **Never quote judge cost pro-rata**; price it off the receipt-calibrated basis.

## One methodological point worth not losing

At $K=0$ the two arms' *candidate* distributions are matched by construction — verified against
both runs' `run_metadata.json`: `branch_sample_temperature = grpo_temperature = 1.2`,
$M = G = 8$, `min_conv_length = 12`, `num_utterances_for_data = 49`. So the difference is **not**
that one samples more widely.

What differs is the **state distribution**: GRPO's prompts are slices of an unmodified on-policy
rollout, while PTO (confirmed `PTgreedy` on disk) grows its trunk by appending the oracle-argmax
of 8 at each therapist turn, a selection that compounds down the trunk. PTO therefore trains on
states from a best-of-$M$ reranked policy — closer to expert iteration than to exploration.
**Do not call this "exploration"** in the draft; an earlier revision did, and it is imprecise in
a way a reviewer would catch. The empirical decomposition in Appendix B is unaffected either way.
