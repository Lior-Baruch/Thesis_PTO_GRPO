# GRPO with Look-Ahead in Motivational Interviewing

*Rewarding a Therapist Turn by Where It Leads*

**THE submission — the single live paper** (Lior, 2026-09-04: "Archive P2, we are going with P1").
**Target: ARR October 2026 cycle** (submission **2026-10-12**, commitment 2026-12-20; the single
cycle feeds **NAACL 2027** and **COLING 2027**, and the venue is chosen in December once reviews
exist). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory unnumbered
Limitations (page-exempt), optional Ethics Statement (page-exempt). `acl.sty` builds in `[review]`
mode (line numbers, anonymized); switch to `[final]` for camera-ready. **The body ends exactly at
the bottom of page 8** (Limitations opens page 9); 22 pages in all.

**Provenance.** Revived 2026-08-27 on Lior's instruction, ported from the archived ICLR-format
draft at [`../archive/2026_grpo_lookahead_mi/`](../archive/2026_grpo_lookahead_mi/). **Rewritten
in full on 2026-09-02** (new title, was *Scoring the Continuation*; dedicated method section with
the group schematic as Figure 1; the rollout audit, the best-checkpoint steelman, the
MI-inconsistency composition, the directive residue). **Refined on 2026-09-04**, the day the 2×2
companion draft (`../archive/2026_pto_grpo_mi/`) was retired and this became the one submission:

- **Endpoint table moved into the body** (Table 1, §5): the paper had no results table in its
  eight pages.
- **A matched-persona transcript excerpt** (Table 2, §6) with utterances 1–9 of both arms'
  iteration-10 conversations verbatim in a new **Appendix D**. The persona is chosen by rule
  ([`select_example_persona.py`](select_example_persona.py): the one of the 96 whose K contrast
  ranks closest to the median under **both** graders → persona 93), and the K=5 turn's own flaws
  are stated in the caption. Every paragraph was diffed against the stored CSV text.
- **Figures 3 and 4 redrawn at page proportions** by [`render_paper_figures.py`](render_paper_figures.py)
  from the tracked tables behind the EDA renders (which were notebook-proportioned and illegible
  at ACL width). The schematic's PTO-referencing side note is cropped away.
- **Rollout audit moved out of the method section** (it is a result) into the Limitations
  paragraph on the continuation pressure; §7 (mechanism) compressed to one paragraph with the
  full analysis in Appendix B.
- **Related work extended**: multi-turn GRPO with turn-level credit and with simulated users
  (`wei2025multiturn`, `qian2025userrl`), BOLT's LLM-therapist behavioural coding
  (`chiu2024bolt`, which the directive residue echoes), AnnoMI (`wu2022annomi`), reward-model
  ensembles (`coste2024ensembles`); the "not a documented failure mode" claim reframed as the
  variance side of reward-model over-optimisation.
- **Every number re-audited against its table** (437 cells, an independent pass). Five prose
  errors fixed: the replicate's "within 0.08" (0.081), the Likert dependability range (0.91–0.96,
  not 0.97), the judge level offset (1.1–1.8 on Q1+Q2 over the 22 states, recomputed), "falls
  monotonically" (it rises at iterations 5 and 8; now "steadily"), and PTO's origin regime
  ("non-iterative" was false — Exp1 ran 7 iterations). All logged as **AUDIT-FIX** in `NUMBERS.md`.
- A Limitations paragraph on the 200-token response cap (both arms grew into it), and a
  camera-ready TODO on the author block in `main.tex`.

**Revised 2026-09-14** (a full review pass before the supervisors' read; decisions Lior's):

- **Headline reframed (robust anchors).** "More than doubles" was anchored on the K=0 arm's
  post-decline last checkpoint; the paper now leads with "leads from iteration 4 on, beats K=0's
  best checkpoint as well as its last" and quotes the gain ratio at BOTH anchors, 2.27× / 2.62×
  (endpoint) and 1.53× / 1.34× (vs. K=0's best), i.e. "1.3 to 2.6×". §5 retitled. The 67%
  over-praise figure is now given with its trajectory (27% → 9% → 67% over iterations 8–10) and
  the separation-from-iteration-5 statement carries the claim.
- **§8 reframed as ceiling-driven.** The training oracle's spread tracks its LEVEL along both arms
  (K=0 compresses to 0.92–1.01 at a mean near 3.9 and re-expands when the mean falls); K=5 sits
  at the ceiling (58% of conversations ≥ 4.5 on Q1, 40% at the maximum); the held-out judge has
  headroom (0% ≥ 4.5). Figure 4 is now three panels with all four grader × arm series.
  Contribution (iii), the related-work sentence and the discussion paragraph reworded to match.
- **Novelty sentence** in the intro: a single rollout is a one-sample Monte Carlo estimate; PTO's
  pair selection filters that noise, GRPO standardises within the group and trains on all eight.
- **Figure 1 redrawn** as a landscape `figure*` by [`render_schematic.py`](render_schematic.py)
  (the EDA's portrait schematic printed at ~4 pt in one column).
- **Appendix C gained** an instruments table (sources, item counts, scales, what is reported), the
  patient and therapist prompt templates, and the ten patterns of the lexical over-praise marker.
  §4 now cites Yosef et al. (2024) for Q1/Q2, names MI-SAT as adapted and PCT/MICI as ours, says
  CSQ-8 is 1–4, and states that sessions end when the patient closes them (base mean 28
  utterances) and why MCL=12.
- **Related work** adds VinePPO, REFUEL, SWEET-RL and PATIENT-Ψ (all verified against source).
- **Prose thinned** to keep the body on 8 pages: numbers removed where a table holds them,
  comma chains untangled, meta-commentary cut; abstract ~215 words.
- Appendix floats: Appendix A starts on its own page, B follows on the same page, the tail-audit
  figure is `[t]`.
- **After Lior's read (same day):** Table 2 replaced by a **clear case** (persona 84, the first
  therapist reply to a byte-identical patient opening: K=0 praises, K=5 asks), chosen from the
  ranked list produced by [`select_example_illustrative.py`](select_example_illustrative.py); the
  caption says it is an illustration, and the median-rule persona 93 stays in Appendix D.2 as the
  typical case. Figure 1 and Figure 4 text overlaps fixed (boxes widened, in-panel labels moved
  to the caption / legend). A numbers audit and a citations audit were run by two independent
  agents; their findings are logged below the NEW-0914 block in `NUMBERS.md`.
- **After Lior's second read:** Figure 4a's legend re-placed; Figures 7 and 8 redrawn from their
  tables with plain labels (Figure 8's EDA jargon was unreadable; Figure 7 now uses the paper's
  sign convention); the better arm's level is bold in Tables 1 and 3.

  Every new number is a NEW-0914 row in `NUMBERS.md`. The Figure 4 source block sat at the top
  of `sections/07_mechanism.tex` on purpose (a `figure*` met in a right-hand column is deferred two
  pages) until 2026-09-16, when it moved to `08_measurement.tex` — see the float rule below.

**Restructured 2026-09-16** (Claude, on Lior's instruction to implement the whole of
[`REVIEW_2026-09-16.md`](REVIEW_2026-09-16.md); numbers unchanged, every move logged in
`NUMBERS.md` § "2026-09-16"):

- **Focus rebalanced toward the method and the domain.** §3 gained **Algorithm 1** (the iterative
  loop with the look-ahead reward), a "Why the transfer is not trivial" paragraph (the one-sample
  Monte Carlo argument, moved out of the intro), a "Minimum context length" paragraph (the pilot
  rationale, quoted without Exp2 numbers, plus the Exp3 faithfulness at 12 utterances), and a
  "Cost" paragraph. §4 gained "Why MI" (change talk / sustain talk; MITI's trajectory-level
  globals) and a "Terminology" note (training oracle · held-out judge · instruments · coders).
- **Saturation demoted.** Contributions are now two; the saturation finding is "we also document".
  The old §8 is now **§7**, halved (sign preservation, the per-conversation collapse, the ceiling
  mechanism, what it undermines; the instrument-by-instrument paragraph is one sentence pointing at
  Table 4). The old §7 (mechanism) is one paragraph of the discussion (now **§8**), "What we could
  not isolate", with Appendix B unchanged; `sections/07_mechanism.tex` was deleted.
- **MI vocabulary.** "Flattery" → over-praise / unearned affirmation throughout; §6 grounds the
  over-praise code in the MITI 4.2.1 Affirm definition (which lists "I am really proud of you" as
  not coded — almost the Table 2 turn), names the directive residue as MI's *righting reflex*, and
  quotes the PCT contrast (K=5 elicits more change talk) that the section's mechanism sentence had
  been asserting without its number.
- **Register pass.** Neutral section titles ("Results: reward and evaluation instruments",
  "Behavioural analysis: over-praise under turn-level reward", "Saturation of the training oracle
  at the winning checkpoint"); the aphorisms and the lab-notes sentences in Appendices B–C
  rewritten; the abstract ends on the result with one caveat sentence; "on the rewarded rubric"
  added to the best-checkpoint claim in the abstract and §1 (the ledger's steelman warning).
- **Limitations consolidated** from eleven paragraphs to seven; the rollout-audit numbers moved to
  Appendix A's text (Figure 8's caption already had them). Table 5 lost "arm A / arm B" and glosses
  its trainer-internal rows.
- **Fitting to 8 pages.** The first pass narrowed Figures 2–4 to 0.64–0.68 `\textwidth`, which
  only shrank their type (a `figure*` carries a fixed text height whatever its width, so the
  labels printed at ~4.8 pt). Fixed the same day: `render_paper_figures.py` now draws each figure
  at the exact width the `.tex` includes it at (parsed from `sections/*.tex`), so its point sizes
  are **true page points** (7 pt labels, 6.2 pt ticks, nothing below 5.8), and the page-space
  knob is the drawn **aspect** — Figures 2–3 are back at 0.94 `\textwidth` at close to their
  original proportions (aspects 0.34 / 0.29; the body ends on page 8 with roughly half a column
  to spare), legends moved inside an empty axes region, explanatory legend entries (base line,
  star) moved to the captions. Figure 1 stays at 0.82. The saturation figure (then Figure 4)
  was dropped from §7 at Lior's request the same day: its three panels only repeated numbers the
  section's text states, and it was the float whose placement rule (left column of page 7) made
  the layout fragile. Table 2 is `[tb]`. After any width or aspect change: re-run the script,
  rebuild, and confirm the body still ends by page 8.
- **Line numbers were printing on the text** (Lior, page 5; in fact 98 numbers on seven pages).
  Cause and fix under "Build" below: the four-step chain is one pass short for `lineno`'s pagewise
  mode; `build.py` now builds to convergence and scans the PDF.
- Done since the review: the author block (as on the ICLR 2025 PTO paper). Still open: a
  human-coded sample; the E-questions in the review for the cover note to the supervisors.

**Framing.** PTO is discussed openly as the lever's origin — `baruch2025pto` is cited in the
intro, related work, and discussion as the predecessor that introduced $K$-turn look-ahead with
preference trees + DPO — and this paper's contribution is **moving the lever to GRPO**. The PTO
*arms* of Exp3 appear **nowhere as data**; every full-grid statistic is the 22-GRPO-state
recomputation (`*_grpo` artifacts). The discussion's optimiser×horizon pointer now says "outside
this paper's scope" (it used to point at the companion draft).

**Axis: iterations only** (the 2026-08-27 decision, unchanged). No GPU-hour or budget analysis;
the honest-cost content is one Limitations paragraph (oracle calls ≈matched 302,541 vs 289,983;
≈393k $K{=}5$-only patient calls; median 1.92× per-step wall-clock) and a one-line ≈79 GPU-h
Ethics total.

**Domain:** Exp3, the **two GRPO arms** — `GRPO_LA0` and `GRPO_LA5`, matched MCL=12, G=8, 96
personas, 8 instruments, 10 iterations each, scored by two graders (gpt-4o-mini = the training
oracle; Claude Haiku 4.5 = held out). 2 arms × 11 states = 22 model states.

## The argument in one line

Scoring a candidate therapist turn by the $K$-turn continuation it leads to, rather than by the
turn itself, raises what group-relative RL extracts from the same oracle to 1.3–2.6× the
turn-level gain (depending on grader and on whether the turn-level arm is read at its last or its
best checkpoint), and it decides whether the policy learns motivational interviewing or learns to
over-praise the patient.

## Section map (files under `sections/`)

| file | section | content |
|---|---|---|
| 00_abstract | Abstract | |
| 01_intro | §1 | the turn-only default; MI as the setting; GRPO with look-ahead and its PTO lineage; the controlled pair; results + the two caveats; three contributions |
| 02_related | §2 | GRPO; multi-turn RL for dialogue (incl. multi-turn GRPO); look-ahead/search in preference learning + PTO; reward hacking & LLM judges (+ over-optimisation, ensembles); MI (+ AnnoMI, BOLT) |
| 03_method | §3 | **GRPO with look-ahead** — notation; Figure 1 = the group schematic; **Algorithm 1** = the iterative loop; the look-ahead reward (the $\tau_K$ equation); why the transfer is not trivial; minimum context length; cost |
| 04_setup | §4 | task/simulator/oracle; **why MI**; instruments; terminology; the two arms; evaluation & statistics |
| 05_reward | §5 | *Results: reward and evaluation instruments* — Figure 2 + **Table 1 (endpoint, every instrument, both graders)**; endpoint; gain vs. the turn-level arm (both anchors); onset + the K=0 decline + the best-checkpoint steelman; the replicate draw |
| 06_behaviour | §6 | *Behavioural analysis: over-praise under turn-level reward* — **Table 2 (the clear-case excerpt)**; over-praise + composition + the MITI Affirm definition + the PCT contrast; the judge-free marker (Figure 3); what look-ahead does instead (righting reflex); under the held-out judge |
| 08_measurement | §7 | *Saturation of the training oracle at the winning checkpoint* — no figure (dropped 2026-09-16; the text carries its numbers); arm-level sign preservation; the per-conversation collapse (not Q1-only, one sentence); the ceiling mechanism; what it undermines |
| 09_discussion | §8 | the horizon selects the hack; what we could not isolate (→ Appendix B); scope (one optimizer, one regime; evidence that would overturn it); conclusion with the monitoring recommendation |
| 10_limitations | Limitations (page-exempt) | one run per arm; evaluation draws; matched iterations ≠ matched cost; K∈{0,5}; the continuation pressure (summary; numbers in Appendix A); simulation only / in-sample / same-model patient / no human validation / the response cap; instruments (reward is an outcome, MITI reliability, no per-channel reliability) |
| 11_ethics | Ethics (page-exempt) | |
| A_tables | Appendix A | by-iteration table, per-instrument agreement table, level grids ×2, channel forest, tail audit figure (+ the rollout-audit numbers in the intro text) |
| B_mechanism | Appendix B | the mechanism analysis in full |
| C_repro | Appendix C | configuration, anti-degeneracy, statistics, cost accounting, artifacts (incl. the five script-drawn figures + the excerpt's provenance) |
| D_example | Appendix D | utterances 1–9 of both iteration-10 conversations with persona 93, verbatim; selection rule and scores |

## Scripts

- [`sync_figures.py`](sync_figures.py) — copies (and crops) every EDA-rendered figure the .tex
  references; `--check` reports drift. Does **not** cover Figures 1, 2, 3, 6 and 7, which the
  two render scripts below draw.
- [`render_schematic.py`](render_schematic.py) — draws Figure 1 (the GRPO-group schematic) at
  page width; reads no data.
- [`render_paper_figures.py`](render_paper_figures.py) — draws Figures 2, 3, 6 and 7 from the
  tracked tables (`reward.xlsx::k_headline_grpo_data`, `behaviour.xlsx::overpraise_judgefree_data`,
  `behaviour.xlsx::k_channels_grpo_gpt-4o-mini` + `k_channels_text_grpo`, `mechanism.xlsx::tail_*`),
  each at the exact width the `.tex` includes it at. Figure 6 is drawn in the paper's sign
  (K=5 − K=0). Its `saturation()` (the saturation figure dropped 2026-09-16) is not called by
  `main()`; run it by hand for the Spearman / variance-ratio printout that checks §7's numbers
  (`validity.xlsx::judge_saturation_grpo_data`, `replication.xlsx::sd_by_iter`). Re-run after any
  EDA render pass, then `sync_figures.py` (which now copies only the two level grids, Figures 4–5).
- [`select_example_illustrative.py`](select_example_illustrative.py) — ranks every (persona,
  therapist-turn) pair at iteration 10 by lexical features of the contrast and dumps a persona's
  transcripts; the source of Table 2 / Appendix D.1 (the clear case, persona 84).
- [`select_example_persona.py`](select_example_persona.py) — the median-contrast rule behind
  Appendix D.2 (the typical case, persona 93), plus the transcript dump (`--dump out.json`). Both
  scripts need the Drive-backed conversation data on disk.
- [`make_overleaf_zip.py`](make_overleaf_zip.py) — the Overleaf bundle, for the FIRST upload (see below).
- [`overleaf.py`](overleaf.py) — two-way sync with the Overleaf project after that (see below).

## Conventions

Same as the repo standard (see [`../README.md`](../README.md)): every number in
[`NUMBERS.md`](NUMBERS.md) with its exact `Exp3_PTO_GRPO/eda/results/...` source; figures copied
(never symlinked) by `sync_figures.py`, or drawn from tracked tables by `render_paper_figures.py`;
sign conventions stated at every table (the EDA's K tables report K=0−K=5 — this paper flips
them); grader named on every number, levels never compared across graders; behaviour claims name
their denominator. Cite the ICLR 2025 paper as the SSI-FM *workshop* poster (canonical BibTeX in
[`../2025_iclr_pto_lookahead/README.md`](../2025_iclr_pto_lookahead/README.md)).

## Overleaf

**First upload.** `overleaf.zip` (gitignored; regenerate with `make_overleaf_zip.py`) holds
exactly what Overleaf needs and nothing else: `main.tex`, `sections/*.tex`, `figures/*.png`,
`refs.bib`, `acl.sty`, `acl_natbib.bst`. Upload it as a new project (New Project → Upload
Project), set the compiler to **pdfLaTeX** and the main document to `main.tex`; Overleaf runs
BibTeX itself. The draft is in `[review]` mode (line numbers, anonymous byline), which is the
right mode for supervisor comments; `\usepackage[final]{acl}` in `main.tex` restores the author
block and drops the line numbers.

**Afterwards, sync — do not re-upload.** A second zip upload makes a NEW project with a new URL
and strands the comments on the old one. Use [`overleaf.py`](overleaf.py) (needs Overleaf's Git
access, a premium/site-licence feature) against the project's git URL, from Menu → Sync → Git:

```powershell
& ..\..\.venv\Scripts\python.exe overleaf.py init https://git.overleaf.com/<project-id>   # once
& ..\..\.venv\Scripts\python.exe overleaf.py status   # what differs, both directions
& ..\..\.venv\Scripts\python.exe overleaf.py pull     # Overleaf edits -> this folder
& ..\..\.venv\Scripts\python.exe overleaf.py push     # this folder -> Overleaf
```

It keeps a throwaway clone of the Overleaf repo outside this tree
(`%LOCALAPPDATA%\overleaf-mirrors\`) and copies across the same file list
`make_overleaf_zip.py` defines, so the Overleaf project stays exactly the compilable paper and
this repo's history stays linear and Overleaf-free — `NUMBERS.md`, the READMEs, the review notes
and these scripts are never pushed. `push` refuses when the project has changed since the last
sync (pull first), and `pull` refuses when this folder has uncommitted changes. Overleaf
authentication is a Git token from Account Settings → Git integration, given as the *password*
at the git prompt. Overleaf compiles with `latexmk`, which iterates to convergence on its own, so
the line-number problem `build.py` exists to prevent is local-only.
Figures are already cropped/drawn, so nothing in the zip depends on the repo.

## Build (MiKTeX on Windows — see ../README.md)

```powershell
& ..\..\.venv\Scripts\python.exe build.py            # build until converged, then check
& ..\..\.venv\Scripts\python.exe build.py --check    # audit the existing main.pdf only
```

[`build.py`](build.py) runs `pdflatex`, `bibtex`, then `pdflatex` **until `main.aux`/`main.out`
stop changing**, and then scans the PDF for line numbers printed inside a text column and for
unresolved references (exit status non-zero on any of these, on LaTeX errors, or on
non-convergence). ⚠ **Do not hand-run the usual `pdflatex · bibtex · pdflatex · pdflatex`.**
`acl.sty`'s `[review]` mode loads `lineno` with `switch`, which is *pagewise* mode: each line's
column is read back from the previous pass's `.aux`, so the line numbers are placed correctly only
once two consecutive passes have the same layout, and after `bibtex` moves the back matter the
four-step chain is one pass short. On 2026-09-16 that put 98 line numbers on top of the text
across seven pages with a clean log (cold build: 477 → 6 → 114 → 98 → 0 misplaced by pass).
Nothing in the log reports it; only the scan does.

To eyeball the layout, the repo `.venv` has PyMuPDF: `fitz.open("main.pdf")[p].get_pixmap(dpi=100).save(...)`.

## Before submission (open items)

- Supervisors' read of the revised draft (they signed off on the 2×2 on 2026-08-27; this draft
  supersedes it as the submission and has not been through them since the 2026-09-02 rewrite).
- **Responsible NLP checklist:** answer the generative-AI question truthfully — this draft was
  written and its code built with substantial AI assistance (ACL policy allows it with disclosure;
  a misleading checklist is desk-reject grounds). Camera-ready: the same disclosure in the
  Acknowledgements (see the TODO comment in `main.tex`).
- Camera-ready only: complete the author block in `main.tex` and switch `acl` to `[final]`.
- Optional, if a co-author wants it: a human MI coder on a sample of the endpoint conversations
  would close the paper's most-cited limitation.
