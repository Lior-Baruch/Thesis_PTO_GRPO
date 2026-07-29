# meetings/ — supervisor-facing decks and emails

Everything that leaves the repo for a human: meeting decks, their generators, and the email drafts
that accompanied them. Nothing here is imported by `code/` or `eda/` — it only *reads* the generated
EDA artifacts under [`../eda/results/`](../eda/results/).

```
meetings/
├── build/                          the generators (run from anywhere; paths resolve off __file__)
│   ├── build_supervisor_deck.py    FULL deck — progress + results deep-dive + stats appendix
│   ├── build_results_snapshot.py   LEAN deck — results only, minimum interpretation
│   ├── build_meeting_deck.py       MEETING deck — snapshot results + measurement validity + decisions
│   └── export_pdf.ps1              .pptx -> .pdf (PowerPoint COM); -Png also dumps slide images
└── <YYYY-MM-DD>/                   one folder per meeting: the deck + anything sent with it
```

| Meeting | Contents | Built by |
|---|---|---|
| [2026-07-09](2026-07-09/) | supervisor progress deck | `build_supervisor_deck.py` (earlier revision) |
| [2026-07-13](2026-07-13/) | supervisor progress deck | `build_supervisor_deck.py` (earlier revision) |
| [2026-07-16](2026-07-16/) | supervisor progress deck — steelman best-vs-best headline | `build_supervisor_deck.py` |
| [2026-07-26](2026-07-26/) | results snapshot (pptx + **pdf**) + `email_draft_2026-07-26.md` | `build_results_snapshot.py` |
| [2026-08-03](2026-08-03/) | meeting deck for Kfir Bar — adds the measurement-validity block and the framing/budget decisions | `build_meeting_deck.py` |

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
  ICC, second-judge sweep, sign preservation, gain retention, and where the graders disagree), then
  the framing and budget decisions the snapshot deliberately left off the slide. 18 slides.

Only one script targets each meeting folder — a script's `OUT` names the dated folder it writes
into. To build a deck for a **new** meeting, copy the closest script, change `OUT` (and the date on
the title slide), and create the dated folder.

## Rebuilding

```powershell
# from meetings/build/
& ..\..\..\.venv\Scripts\python.exe build_meeting_deck.py
.\export_pdf.ps1 ..\2026-08-03\meeting_kfir_2026-08-03.pptx        # add -Png to eyeball layout
```

`-Png` writes a scratch `<name>_png\` folder next to the deck. It is **not** gitignored — delete it
once you've checked the layout.

Use the repo `.venv` python — the system python has neither `python-pptx` nor `pillow`.

The scripts read PNGs and markdown tables straight out of `../eda/results/<view>/`, so a deck is
only as current as the last `render_views.py` run. If a figure moved or was renamed by an EDA
refactor, the script fails loudly at `add_picture` and writes nothing — the file on disk is never
half-updated.

`build_meeting_deck.py` also reads the method schematics in [`../figures/`](../figures/) (the PTO
and GRPO framework diagrams and the two generation diagrams). Those are hand-authored, not
data-derived — regenerate them with `build_method_figures.py` in that directory, not with
`render_views.py`.

⚠ **Artifact paths carry a `<judge>/` level** (`<family>/gpt-4o-mini/<name>.png`) since 2026-07-28 —
every grader nests, including the primary. The builders' path helpers were updated then; a deck
built off a hand-written flat path will fail the `add_picture` check above.

## What's tracked

`.pptx` is gitignored repo-wide (regenerable from `build/`). The **PDF is tracked**: it is the
artifact that was actually emailed, and since the pptx isn't versioned it's the only record of what
the supervisors saw on that date. Email drafts are tracked for the same reason.
