# meetings/ — supervisor-facing decks and emails

Everything that leaves the repo for a human: meeting decks, their generators, and the email drafts
that accompanied them.

> ⚠ **Each builder is a SNAPSHOT of what was presented on its date, not a live view.** The numbers
> are hard-coded, so re-running a builder reproduces the deck *as delivered* — which is the point,
> and the reason none of them is updated when the data moves. **All pre-2026-08-18 meetings now
> live under [`archive/`](archive/)** (moved 2026-08-18) and several are stale against
> [`../STATUS.md`](../STATUS.md): they describe the GRPO K=5 arm as having **one** iteration (it
> has five, scored 0–5 on both graders), quote the held-out grid as **22,272 cells / 29 model
> states** (now 39 × 8 × 96 = **29,952**), and predate the compute axis entirely. The current deck
> is [`2026-08-18/`](2026-08-18/) (`build_status_deck_2026-08-18.py`), whose numbers were
> re-verified against the rendered tables on its date. **Before re-presenting any slide from an
> archived builder, re-check its numbers against `STATUS.md` and the tables** — or copy the
> builder to a new dated one and update there. Do not edit a past deck in place; it is a record of
> what the supervisors were actually shown.

Nothing here is imported by `code/` or `eda/` — it only *reads* the generated EDA artifacts under
[`../Exp3_PTO_GRPO/eda/results/`](../Exp3_PTO_GRPO/eda/results/).

> **2026-08-18 — the EDA results tree was reorganised by research question.** Every builder
> written before that date (`build_supervisor_deck.py`, `build_results_snapshot.py`,
> `build_meeting_deck.py`, `build_paper_deck.py`, `build_status_deck.py`,
> `build_status_deck_2026-08-18.py`) reads the **retired** `Exp3_PTO_GRPO/eda/results/L0|L5/…`
> tree (`<view>/figures|tables/<N_family>/<judge>/`) and the method schematics at
> `Exp3_PTO_GRPO/figures/` (now `eda/results/schematics/`). Both are recoverable at commit
> **`b09eb6f`** (the last pre-reorg state) — check it out to re-run one of those builders; **do
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

- **`build_status_deck_2026-08-18.py`** — the **current** whole-project deck (21 slides): the
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
`b09eb6f` to re-run them), new builders from `<top>/<sub>/…` — so a deck is only as current as the
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
