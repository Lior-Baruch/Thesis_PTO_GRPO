# meetings/ — supervisor-facing decks and emails

Everything that leaves the repo for a human: meeting decks, their generators, and the email drafts
that accompanied them.

> ⚠ **Each builder is a SNAPSHOT of what was presented on its date, not a live view.** The numbers
> are hard-coded, so re-running a builder reproduces the deck *as delivered* — which is the point,
> and the reason none of them is updated when the data moves. **All pre-2026-08-18 meetings now
> live under [`archive/`](archive/)** (moved 2026-08-18) and several are stale against
> [`../STATUS.md`](../STATUS.md): they describe the GRPO K=5 arm as having **one** iteration (it
> has six, scored 0–6 on both graders), quote the held-out grid as **22,272 cells / 29 model
> states** (now 40 × 8 × 96 = **30,720**), and predate the compute axis entirely. The current deck
> is [`2026-08-24/`](2026-08-24/) (`build_figure_atlas_2026-08-24.py`) — the **figure atlas**, one
> graph per slide with descriptive captions and no interpretation, for reading together in the
> meeting. The deck before it, [`2026-08-23/`](2026-08-23/)
> (`build_thesis_arc_deck_2026-08-23.py`), is the argued version of the same material: the first to
> cover all three experiments rather than Exp3 alone, with numbers gathered by a cold read of the
> owning tables plus direct recomputation from Exp1's and Exp2's raw files, then independently
> re-verified against those sources. **Both are current** — they serve different meetings.
>
> ⚠ **Every builder before 2026-08-23 presents Exp2's GRPO V1 as a "weak baseline". That run had a
> bug and its results are VOID** (`Exp2_PTO/CLAUDE.md` § "GRPO V1 — VOID"). Exp1 and Exp2 are
> **PTO-only**. Do not carry a GRPO-in-Exp2 slide forward from any archived deck.
> **Before re-presenting any slide from an
> archived builder, re-check its numbers against `STATUS.md` and the tables** — or copy the
> builder to a new dated one and update there. Do not edit a past deck in place; it is a record of
> what the supervisors were actually shown.
>
> ⚠ **The 2026-08-18 deck is now itself stale in two load-bearing ways**: it asserts the two GRPO
> arms are “budget-matched within 3%” (true at iteration 5 only — at iteration 6 the K=5 arm costs
> 9.4% *more*), and it predates the iteration-6 scoring that made the K × optimizer interaction
> visible to the primary oracle. It also cannot regenerate itself, since its builder reads the
> retired `results/{L0,L5}/` tree.

Nothing here is imported by `code/` or `eda/` — it only *reads* the generated EDA artifacts under
[`../Exp3_PTO_GRPO/eda/results/`](../Exp3_PTO_GRPO/eda/results/).

> **2026-08-18 — the EDA results tree was reorganised by research question.** Every builder
> written before that date (`build_supervisor_deck.py`, `build_results_snapshot.py`,
> `build_meeting_deck.py`, `build_paper_deck.py`, `build_status_deck.py`,
> `build_status_deck_2026-08-18.py`) reads the **retired** `Exp3_PTO_GRPO/eda/results/L0|L5/…`
> tree (`<view>/figures|tables/<N_family>/<judge>/`) and the method schematics at
> `Exp3_PTO_GRPO/figures/` (now `eda/results/schematics/`). Both are recoverable at commit
> **`abe5cb3`** (the last pre-reorg state — code and docs; the pre-reorg `results/` renders are in
> the archival bundle, not in git) — check it out to re-run one of those builders; **do
> not edit past builders** to chase the new paths (they are records of what was shown). New builders
> read `Exp3_PTO_GRPO/eda/results/<top>/<sub>/{figures,tables}/[<judge>/]…` — per-arm figures under
> `arms/*` (a `<judge>/` leaf), cross-K / method / compute / measurement artifacts under
> `lookahead/*`, `method/contrast`, `compute/cost`, `measurement/validity` (no judge level — both
> graders are inside), and schematics under `results/schematics/`. Regenerate with
> `python tools/render_results.py` from `Exp3_PTO_GRPO/eda/` (`render_views.py` is gone). Old→new
> path map: [`../Exp3_PTO_GRPO/eda/README.md`](../Exp3_PTO_GRPO/eda/README.md) § "Migration (2026-08-18)".

**This directory lives at the repo root**, beside [`../papers/`](../papers/), because decks span
experiments and now also present the paper drafts. It used to sit inside `Exp3_PTO_GRPO/`. The
builders resolve the experiment explicitly:

```python
REPO = os.path.dirname(os.path.dirname(HERE))   # repo root — decks are written under REPO/meetings/
ROOT = os.path.join(REPO, "Exp3_PTO_GRPO")      # the experiment the artifacts come from
```

If a future deck reads a different experiment's artifacts, change `ROOT` in that one script.

```
meetings/
├── build/                          the generators (run from anywhere; paths resolve off __file__)
│   ├── build_supervisor_deck.py    FULL deck — progress + results deep-dive + stats appendix
│   ├── build_results_snapshot.py   LEAN deck — results only, minimum interpretation
│   ├── build_meeting_deck.py       MEETING deck — snapshot results + measurement validity + decisions
│   ├── build_paper_deck.py         PAPER deck — what a draft argues, what it won't claim, what to decide
│   ├── build_status_deck.py        STATUS deck — the whole project: all arms, both graders, the
│   │                               look-ahead result, and the scope/budget decisions
│   ├── _deck_kit.py                shared house visual system (palette, bands, tables, Deck,
│   │                               figpage/factstrip) — imported by builders from 2026-08-23 on
│   ├── build_figure_atlas_2026-08-24.py
│   │                               FIGURE ATLAS — one graph per slide, descriptive captions only,
│   │                               no interpretation; all three experiments
│   ├── make_exp1_exp2_figs.py      renders Exp1 + Exp2 figures into _figs/ (their EDAs write none)
│   ├── build_thesis_arc_deck_2026-08-23.py
│   │                               THESIS ARC deck — three acts, one per experiment, each closing
│   │                               on SETTLED / HANDED FORWARD; Exp1 + Exp2 are PTO-only
│   ├── _exp2_summary.{md,csv}      Exp2 per-model-state scores rebuilt from Exp2_PTO/eda/eval/**,
│   │                               so Act II's numbers cite a file the way Exp3's do (void GRPO
│   │                               states stripped)
│   ├── build_rq_deck_2026-08-21.py RQ deck — organised by the three research questions, with a
│   │                               VERDICT / evidence / NOT THIS band on every answer slide
│   └── export_pdf.ps1              .pptx -> .pdf (PowerPoint COM); -Png also dumps slide images
├── archive/<YYYY-MM-DD>/           past meetings (moved here 2026-08-18) — records, never edited
└── <YYYY-MM-DD>/                   the current meeting: the deck + anything sent with it
```

| Meeting | Contents | Built by |
|---|---|---|
| [archive/2026-07-09](archive/2026-07-09/) | supervisor progress deck | `build_supervisor_deck.py` (earlier revision) |
| [archive/2026-07-13](archive/2026-07-13/) | supervisor progress deck | `build_supervisor_deck.py` (earlier revision) |
| [archive/2026-07-16](archive/2026-07-16/) | supervisor progress deck — steelman best-vs-best headline | `build_supervisor_deck.py` |
| [archive/2026-07-26](archive/2026-07-26/) | results snapshot (pptx + **pdf**) + `email_draft_2026-07-26.md` | `build_results_snapshot.py` |
| [archive/2026-08-03](archive/2026-08-03/) | supervision meeting deck (a room, not one reader — no "since the email" framing) — measurement validity, the first matched-budget K comparison, the training-signal mechanism block, and the framing/budget decisions | `build_meeting_deck.py` |
| [archive/2026-08-16](archive/2026-08-16/) | **two decks.** `paper_…` — the CLPsych draft *Affirmation Without Inquiry*: its three moves, the evidence behind each, an explicit "what the paper does not claim" slide, and the submission decisions. `status_…` — the whole project: all four arms, the main result, measurement validity, the **new look-ahead substitution result**, both drafts, and the scope/budget decisions | `build_paper_deck.py` · `build_status_deck.py` |
| [**2026-08-18**](2026-08-18/) | **current status deck** (pptx + pdf, 21 slides): all four arms complete + budget-matched GRPO pair, the **compute axis**, the **K × method answer** (look-ahead pays on GRPO, not PTO — mostly visible only to the held-out grader), the **retention-by-K result** (new that day, from the un-broken L5 table), the table-first-audit slide, both drafts as revised 2026-08-18, and the updated decisions (replicate draw = recommended buy; GRPO-at-K=5 done and off the list) | `build_status_deck_2026-08-18.py` |
| [**2026-08-24**](2026-08-24/) | **current deck — the FIGURE ATLAS** (pptx + pdf + png, 84 slides). Built for a brainstorming session: one figure per slide with a strip stating the x axis, the y axis, the series and how the numbers were produced, and **no interpretation anywhere**. Covers all three experiments — Exp1 and Exp2 figures were generated for this deck by `make_exp1_exp2_figs.py` (neither experiment's EDA writes figures to disk), Exp3 figures come from the rendered results tree. Sections: method schematics · Exp1 · Exp2 · Exp3 outcomes / look-ahead / method / behaviour / mechanism / training signal / compute / measurement / rubrics | `build_figure_atlas_2026-08-24.py` |
| [2026-08-23](2026-08-23/) | thesis-arc deck (pptx + pdf + png, 36 slides) — the first deck covering **all three experiments**. Act I re-audits the ICLR paper: all 15 published rows reproduce from the shipped data, but the omnibus ANOVA was run on 3 groups of 15, the K contrast is significant on 1 of 4 metrics (Q2, p = .0315), and re-scoring the 1,440 transcripts under gpt-4o-mini keeps K=5 ahead 7/7 in direction while erasing the K=0 arm's only significant gain. The "lowest SD = most stable" claim is a scoring ceiling. Act II reports the Exp2 PTO sweep and its negative result — training on WAI-SR or CSQ-8 moved neither (p = .414 / .304) — plus the finding that the "clean Exp2 subset ≈ 2.93 ≈ Exp3" claim does **not** reproduce (measured 2.648; degeneration closes only 41% of the gap). Act III is the Exp3 RQ block. Closes on a cross-experiment scorecard and three decisions | `build_thesis_arc_deck_2026-08-23.py` |
| [2026-08-21](2026-08-21/) | RQ deck (pptx + pdf, 22 slides) — the first deck organised by the **three research questions** rather than by project status. RQ-i: look-ahead helps GRPO and hurts PTO, both significant at iteration 6 on both graders, and the K × optimizer interaction is now visible to the **primary** oracle too (DiD +0.520, dz 0.605). RQ-ii: PTO wins decisively at K=0 and **GRPO wins at K=5** — so neither question has an answer independent of the other. RQ-iii: held. Plus the compute axis, the measurement thread, one honest slide on the run that stopped, and three decisions | `build_rq_deck_2026-08-21.py` |

## Which deck to build

- **`build_supervisor_deck.py`** — the weekly/progress deck. Full rigor: status, results deep-dive,
  measurement-validity thread, threats, next-steps decision, plus an appendix of native stats
  tables. For people already in the loop.
- **`build_results_snapshot.py`** — the lean one. Results only, captions state what is plotted and
  the value and draw no conclusions; opens with an ICLR reminder + an Exp1/Exp2/Exp3 lineage slide.
  Written for a reader coming in cold, and for cases where the interpretation should happen live
  rather than on the slide.
- **`build_meeting_deck.py`** — the snapshot's visual language, but meant to be *talked through*:
  the same background + results block condensed, then the measurement-validity evidence (oracle
  ICC, second-judge sweep, sign preservation, gain retention, and where the graders disagree), the
  look-ahead comparison at its first matched-budget point, a mechanism block reading the training
  signal directly (generation vs selection; the loss-vs-exploration decomposition), and finally the
  framing and budget decisions the snapshot deliberately left off the slide. 24 slides.
- **`build_paper_deck.py`** — about a *draft*, not about run status. The paper's argument and the
  evidence carrying each move, then an explicit **"what the paper does not claim"** slide, then the
  decisions needed to submit (co-author list, venue, date, human-coder validation). 14 slides.
  The refusals slide is the point of this deck: claims that were weakened during drafting are named
  on a slide, so nobody in the room leaves repeating a stronger version than the paper defends.
  Numbers here are owned by the paper's `NUMBERS.md` ledger, not by `eda/results/<top>/SUMMARY.md`.

- **`build_figure_atlas_2026-08-24.py`** — the **current** deck (84 slides). Not an argument: a
  **figure catalogue** for reading together. One figure per slide plus a labelled strip giving the
  axes, the series, the n and the computation — and **deliberately no interpretation at all**, so
  the reading happens in the room. All three experiments. If you edit it, the rule is in its
  docstring: *a caption may say what was measured and how; it may not say what it shows.*
  Its Exp1/Exp2 figures come from **`make_exp1_exp2_figs.py`** (run that first).
- **`make_exp1_exp2_figs.py`** — renders the Exp1 and Exp2 figures to `_figs/{exp1,exp2}/`.
  Needed because **neither experiment's EDA writes any figure to disk** — their `Conv_EDA.ipynb`
  notebooks draw inline only, so every Exp1/Exp2 figure ever shown was recomputed ad hoc. Exp1
  comes from the raw per-conversation CSVs; Exp2 from `_exp2_summary.csv`. Exp1's 15 published
  models are identified by exact numeric match against the paper's Table 1 — the paper never
  discloses which of its six baseline directories it used.
- **`build_thesis_arc_deck_2026-08-23.py`** — the previous deck (36 slides), and the first to
  present **all three experiments** instead of Exp3 with a lineage reminder. Spine = three acts,
  one per experiment, each closing on a two-column *SETTLED / HANDED FORWARD* slide, then a
  cross-experiment scorecard of which claims survived. Act I re-audits the published ICLR paper
  against its own shipped data and re-scores its transcripts under the modern grader; Act II
  reports Exp2's PTO sweep including the negative result that two of three training instruments
  produced no gain; Act III is the Exp3 RQ block, carrying the **VERDICT / NOT THIS** bands from
  the 2026-08-21 deck. **Exp1 and Exp2 are PTO-only** — see the GRPO V1 warning above.
  Build this one for the next meeting, or copy it forward.
- **`_deck_kit.py`** — not a deck. The house visual system (palette, bands, tables, dividers,
  the `Deck` slide counter) factored out of `build_rq_deck_2026-08-21.py` on 2026-08-23 so a new
  builder starts with slides rather than 250 lines of transcribed primitives. Imported only by
  builders written on or after that date; **add primitives, never change an existing one's
  behaviour**, or a rebuild of a deck that imports it would render differently from what was shown.
- **`build_rq_deck_2026-08-21.py`** — the previous deck (22 slides), and the first that was NOT a
  fork of an earlier builder. Spine = the thesis's three research questions; every answer slide carries
  the same three bands — **VERDICT** (the claim), the table or figure that carries it with its
  `results/…` path printed on the slide, and **NOT THIS** (the nearest stronger claim the evidence does
  not support). Exp3 only.
- **`build_status_deck_2026-08-18.py`** — the previous whole-project deck (21 slides): the
  2026-08-16 status deck's structure, re-verified numbers, plus three slides that did not exist
  then — the compute axis, the K × method answer, and retention-by-K — and updated decision
  slides (the GRPO-at-K=5 purchase is done; the live ask is the replicate draw). Build this one
  for the next meeting, or copy it forward.
- **`build_status_deck.py`** — the 2026-08-16 **snapshot** of the whole project, for a room where
  not everyone has read the drafts. Background and setup, the four arms and what is scored, the main PTO-vs-GRPO
  result, what the model actually learned, measurement validity, then a three-slide look-ahead
  block (the channel closes · the aggregate does not move · where the intervention acts), a slide
  naming **two readings that were corrected** and why, both drafts, and the scope/budget decisions.
  18 slides. It overlaps `build_meeting_deck.py` on the background/results block by design — the
  difference is that this one carries the look-ahead result and the two-paper scope question, which
  did not exist on 2026-08-03.

Each script's `OUT` names the file it writes, so a dated folder can hold **more than one** deck
(2026-08-16 holds two: the paper deck and the status deck). To build a deck for a **new** meeting,
copy the closest script, change `OUT` (and the date on the title slide), and create the dated
folder. Never point a new script at an existing deck's filename.

## Rebuilding

```powershell
# from meetings/build/
& ..\..\.venv\Scripts\python.exe build_meeting_deck.py
.\export_pdf.ps1 ..\2026-08-03\meeting_2026-08-03.pptx             # add -Png to eyeball layout
```

`-Png` writes a scratch `<name>_png\` folder next to the deck. It is **not** gitignored — delete it
once you've checked the layout.

Use the repo `.venv` python — the system python has neither `python-pptx` nor `pillow`.

The scripts read PNGs and markdown tables straight out of `../Exp3_PTO_GRPO/eda/results/` — the
pre-2026-08-18 builders from the retired `<view>/…` tree (see the note at the top; check out
`abe5cb3` to re-run them), new builders from `<top>/<sub>/…` — so a deck is only as current as the
last render (`render_results.py` now; `render_views.py` before). If a figure moved or was renamed by
an EDA refactor, the script fails loudly at `add_picture` and writes nothing — the file on disk is
never half-updated.

`build_meeting_deck.py` also reads the method schematics (the PTO and GRPO framework diagrams and
the two generation diagrams) — at `../Exp3_PTO_GRPO/figures/` when it was written, now
[`../Exp3_PTO_GRPO/eda/results/schematics/`](../Exp3_PTO_GRPO/eda/results/schematics/). Those are
hand-authored, not data-derived — regenerate them with `build_method_figures.py` in that directory,
never with the render tool.

⚠ **Artifact paths carry a `<judge>/` level** (`<family>/gpt-4o-mini/<name>.png`) since 2026-07-28 —
every grader nests, including the primary. The builders' path helpers were updated then; a deck
built off a hand-written flat path will fail the `add_picture` check above.

## What's tracked

`.pptx` is gitignored repo-wide (regenerable from `build/`). The **PDF is tracked**: it is the
artifact that was actually emailed, and since the pptx isn't versioned it's the only record of what
the supervisors saw on that date. Email drafts are tracked for the same reason.
