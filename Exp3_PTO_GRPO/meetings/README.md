# meetings/ — supervisor-facing decks and emails

Everything that leaves the repo for a human: meeting decks, their generators, and the email drafts
that accompanied them. Nothing here is imported by `code/` or `eda/` — it only *reads* the generated
EDA artifacts under [`../eda/results/`](../eda/results/).

```
meetings/
├── build/                          the generators (run from anywhere; paths resolve off __file__)
│   ├── build_supervisor_deck.py    FULL deck — progress + results deep-dive + stats appendix
│   ├── build_results_snapshot.py   LEAN deck — results only, minimum interpretation
│   └── export_pdf.ps1              .pptx -> .pdf (PowerPoint COM); -Png also dumps slide images
└── <YYYY-MM-DD>/                   one folder per meeting: the deck + anything sent with it
```

| Meeting | Contents | Built by |
|---|---|---|
| [2026-07-09](2026-07-09/) | supervisor progress deck | `build_supervisor_deck.py` (earlier revision) |
| [2026-07-13](2026-07-13/) | supervisor progress deck | `build_supervisor_deck.py` (earlier revision) |
| [2026-07-16](2026-07-16/) | supervisor progress deck — steelman best-vs-best headline | `build_supervisor_deck.py` |
| [2026-07-26](2026-07-26/) | results snapshot (pptx + **pdf**) + `email_draft_2026-07-26.md` | `build_results_snapshot.py` |

## Which deck to build

- **`build_supervisor_deck.py`** — the weekly/progress deck. Full rigor: status, results deep-dive,
  measurement-validity thread, threats, next-steps decision, plus an appendix of native stats
  tables. For people already in the loop.
- **`build_results_snapshot.py`** — the lean one. Results only, captions state what is plotted and
  the value and draw no conclusions; opens with an ICLR reminder + an Exp1/Exp2/Exp3 lineage slide.
  Written for a reader coming in cold, and for cases where the interpretation should happen live
  rather than on the slide.

Only one script targets each meeting folder — a script's `OUT` names the dated folder it writes
into. To build a deck for a **new** meeting, copy the closest script, change `OUT` (and the date on
the title slide), and create the dated folder.

## Rebuilding

```powershell
# from meetings/build/
& ..\..\..\.venv\Scripts\python.exe build_results_snapshot.py
.\export_pdf.ps1 ..\2026-07-26\results_snapshot_2026-07-26.pptx
```

Use the repo `.venv` python — the system python has neither `python-pptx` nor `pillow`.

Both scripts read PNGs and markdown tables straight out of `../eda/results/<view>/`, so a deck is
only as current as the last `render_views.py` run. If a figure moved or was renamed by an EDA
refactor, the script fails loudly at `add_picture` and writes nothing — the file on disk is never
half-updated.

## What's tracked

`.pptx` is gitignored repo-wide (regenerable from `build/`). The **PDF is tracked**: it is the
artifact that was actually emailed, and since the pptx isn't versioned it's the only record of what
the supervisors saw on that date. Email drafts are tracked for the same reason.
