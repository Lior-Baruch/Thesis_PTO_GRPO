# GRPO with Look-Ahead in Motivational Interviewing

*Rewarding a Therapist Turn by Where It Leads*

**THE submission — the single live paper** (Lior, 2026-09-04: "Archive P2, we are going with P1").
**Target: ARR October 2026 cycle** (submission **2026-10-12**, commitment 2026-12-20; the single
cycle feeds **NAACL 2027** and **COLING 2027**, and the venue is chosen in December once reviews
exist). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory unnumbered
Limitations (page-exempt), optional Ethics Statement (page-exempt). `acl.sty` builds in `[review]`
mode (line numbers, anonymized); switch to `[final]` for camera-ready. **The body ends exactly at
the bottom of page 8** (Limitations opens page 9); 20 pages in all.

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
turn itself, more than doubles what group-relative RL extracts from the same oracle, and it is
the difference between a policy that learns motivational interviewing and one that learns to
flatter the judge.

## Section map (files under `sections/`)

| file | section | content |
|---|---|---|
| 00_abstract | Abstract | |
| 01_intro | §1 | the turn-only default; MI as the setting; GRPO with look-ahead and its PTO lineage; the controlled pair; results + the two caveats; three contributions |
| 02_related | §2 | GRPO; multi-turn RL for dialogue (incl. multi-turn GRPO); look-ahead/search in preference learning + PTO; reward hacking & LLM judges (+ over-optimisation, ensembles); MI (+ AnnoMI, BOLT) |
| 03_method | §3 | **GRPO with look-ahead** — the iterative loop, the group, the look-ahead reward (the $\tau_K$ equation), what the lever costs; Figure 1 = the group schematic |
| 04_setup | §4 | task/simulator/oracle, instruments, the two arms, evaluation & statistics |
| 05_reward | §5 | Figure 2 + **Table 1 (endpoint, every instrument, both graders)**; headline; onset + the best-checkpoint steelman; the replicate draw |
| 06_behaviour | §6 | **Table 2 (the matched-persona excerpt)**; over-praise + composition; the judge-free marker (Figure 3); what look-ahead does instead; the honest version |
| 07_mechanism | §7 | one paragraph: three candidate mechanisms, none confirmed (→ Appendix B) |
| 08_measurement | §8 | arm-level sign preservation; the per-conversation collapse; not Q1-only; one-sided saturation (Figure 4); what it undermines |
| 09_discussion | §9 | the horizon selects the hack; scope (one optimizer, one regime; what would change our minds); saturation as a named failure mode; conclusion |
| 10_limitations | Limitations (page-exempt) | one run per arm; evaluation draws; matched iterations ≠ matched cost; K∈{0,5}; the continuation pressure (with the rollout audit numbers); the response cap; in-sample personas; generator not decoupled; no human validation; reward is an outcome; instrument reliability |
| 11_ethics | Ethics (page-exempt) | |
| A_tables | Appendix A | by-iteration table, per-instrument agreement table, level grids ×2, channel forest, tail audit figure |
| B_mechanism | Appendix B | the mechanism analysis in full |
| C_repro | Appendix C | configuration, anti-degeneracy, statistics, cost accounting, artifacts (incl. the two redrawn figures + the excerpt's provenance) |
| D_example | Appendix D | utterances 1–9 of both iteration-10 conversations with persona 93, verbatim; selection rule and scores |

## Scripts

- [`sync_figures.py`](sync_figures.py) — copies (and crops) every EDA-rendered figure the .tex
  references; `--check` reports drift. Does **not** cover Figures 3–4.
- [`render_paper_figures.py`](render_paper_figures.py) — draws Figures 3–4 from the tracked
  tables (`behaviour.xlsx::overpraise_judgefree_data`, `validity.xlsx::judge_saturation_grpo_data`).
  Re-run after any EDA render pass, then `sync_figures.py`.
- [`select_example_persona.py`](select_example_persona.py) — the persona-selection rule behind
  Table 2 / Appendix D, plus the transcript dump (`--dump out.json`). Needs the Drive-backed
  conversation data on disk.
- [`make_overleaf_zip.py`](make_overleaf_zip.py) — the Overleaf bundle (see below).

## Conventions

Same as the repo standard (see [`../README.md`](../README.md)): every number in
[`NUMBERS.md`](NUMBERS.md) with its exact `Exp3_PTO_GRPO/eda/results/...` source; figures copied
(never symlinked) by `sync_figures.py`, or drawn from tracked tables by `render_paper_figures.py`;
sign conventions stated at every table (the EDA's K tables report K=0−K=5 — this paper flips
them); grader named on every number, levels never compared across graders; behaviour claims name
their denominator. Cite the ICLR 2025 paper as the SSI-FM *workshop* poster (canonical BibTeX in
[`../2025_iclr_pto_lookahead/README.md`](../2025_iclr_pto_lookahead/README.md)).

## Overleaf

`overleaf.zip` (gitignored; regenerate with `make_overleaf_zip.py`) holds exactly what Overleaf
needs and nothing else: `main.tex`, `sections/*.tex`, `figures/*.png`, `refs.bib`, `acl.sty`,
`acl_natbib.bst`. Upload it as a new project (New Project → Upload Project), set the compiler to
**pdfLaTeX** and the main document to `main.tex`; Overleaf runs BibTeX itself. The draft is in
`[review]` mode (line numbers, anonymous byline), which is the right mode for supervisor comments;
`\usepackage[final]{acl}` in `main.tex` restores the author block and drops the line numbers.
Figures are already cropped/drawn, so nothing in the zip depends on the repo.

## Build (MiKTeX on Windows — see ../README.md)

```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```

To eyeball the layout, the repo `.venv` has PyMuPDF: `fitz.open("main.pdf")[p].get_pixmap(dpi=100).save(...)`.

## Before submission (open items)

- Supervisors' read of the refined draft (they signed off on the 2×2 on 2026-08-27; this draft
  supersedes it as the submission and has not been through them since the 2026-09-02 rewrite).
- Camera-ready only: complete the author block in `main.tex` and switch `acl` to `[final]`.
- Optional, if a co-author wants it: a human MI coder on a sample of the endpoint conversations
  would close the paper's most-cited limitation.
