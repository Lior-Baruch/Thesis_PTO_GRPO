# GRPO with Look-Ahead in Motivational Interviewing

*Rewarding a Therapist Turn by Where It Leads*

**Target: ARR October 2026 cycle** (submission **2026-10-12**, commitment 2026-12-20; the single
cycle feeds **NAACL 2027** and **COLING 2027**, and the venue is chosen in December once reviews
exist). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory unnumbered
Limitations (page-exempt), optional Ethics Statement (page-exempt). `acl.sty` builds in `[review]`
mode (line numbers, anonymized); switch to `[final]` for camera-ready.

**Provenance.** Revived 2026-08-27 on Lior's instruction as a second ARR submission beside the
2×2 paper (`../2026_pto_grpo_mi/`), ported from the archived ICLR-format draft at
[`../archive/2026_grpo_lookahead_mi/`](../archive/2026_grpo_lookahead_mi/). **Rewritten in full on
2026-09-02** (Fable 5.1): new title (was *Scoring the Continuation*), a dedicated method section
with the group schematic promoted to Figure 1, tighter prose throughout, and four additions that
the earlier draft did not carry: the look-ahead rollout audit (how often the $K{=}5$ window ran
to length, and the mild keep-the-session-open pressure it implies), the best-checkpoint steelman
(the $K{=}5$ endpoint beats the $K{=}0$ arm's *best* iteration under either grader's selection),
the MI-inconsistency composition at the endpoint (84% of the $K{=}0$ policy's coded acts are
over-praise; the $K{=}5$ policy's residue is the base's, in kind and amount), and the directive
residue on the channel forest (direct/order and persuasion rise modestly under $K{=}5$). Related
work gained a multi-turn-RL-for-dialogue paragraph (ArCHer, imagined conversations, SOTOPIA-π,
multi-turn RLHF) and a DeepSeek-R1 citation for GRPO's current standing.

**Framing.** PTO is discussed openly as the lever's origin — `baruch2025pto` is cited in the
intro, related work, and discussion as the predecessor that introduced $K$-turn look-ahead with
preference trees + DPO — and this paper's contribution is **moving the lever to GRPO**. The PTO
*arms* of Exp3 appear **nowhere as data**; every full-grid statistic is the 22-GRPO-state
recomputation (`*_grpo` artifacts).

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
| 01_intro | §1 | the turn-only default; MI as the setting; GRPO with look-ahead and its PTO lineage; the controlled pair; results; the two caveats; three contributions |
| 02_related | §2 | GRPO; multi-turn RL for dialogue (new ¶); look-ahead/search in preference learning + PTO; reward hacking & LLM judges; MI |
| 03_method | §3 | **GRPO with look-ahead** — the iterative loop, the group, the look-ahead reward (equation), what the lever costs, the rollout audit; Figure 1 = the group schematic |
| 04_setup | §4 | task/simulator/oracle, instruments, the two arms, evaluation & statistics |
| 05_reward | §5 | headline, every instrument, onset + the best-checkpoint steelman, the replicate draw; Figure 2 |
| 06_behaviour | §6 | over-praise + composition, the judge-free marker, what look-ahead does instead (questions up, affirmations down, a directive residue, longer sessions), the honest version; Figure 3 |
| 07_mechanism | §7 | three candidate mechanisms, none confirmed (faithfulness, dispersion, direction) |
| 08_measurement | §8 | arm-level sign preservation; the per-conversation collapse; not Q1-only; one-sided saturation; what it undermines; Figure 4 |
| 09_discussion | §9 | the horizon selects the hack; scope (one optimizer, one regime; what would change our minds); saturation as a named failure mode |
| 10_limitations | Limitations (page-exempt) | one run per arm; evaluation draws; matched iterations ≠ matched cost; K∈{0,5}; the continuation pressure (new); in-sample personas; generator not decoupled; no human validation; reward is an outcome; instrument reliability |
| 11_ethics | Ethics (page-exempt) | |
| A_tables | Appendix A | endpoint table, by-iteration table, level grids ×2, channel forest, per-instrument agreement table, tail audit figure |
| B_mechanism | Appendix B | the mechanism analysis in full |
| C_repro | Appendix C | configuration, anti-degeneracy, statistics, cost accounting, artifacts |

⚠ **Dual-submission overlap with the 2×2 paper** (`../2026_pto_grpo_mi/`, same ARR cycle): the
GRPO K-lever numbers (+0.765/+0.616, the gain ratios, the overpraise marker, the saturation
mechanism) appear in both papers — here as the subject, there as cells of the interaction. The
papers never cite each other's prose and their claims are disjoint (single-lever deep-dive vs the
optimizer×horizon interaction), but ARR's multiple-submission policy is a judgment call the
authors must clear with the supervisors before 2026-10-12. As of 2026-09-02 the supervisors have
signed off on the 2×2 only.

## Conventions

Same as the repo standard (see [`../README.md`](../README.md)): every number in
[`NUMBERS.md`](NUMBERS.md) with its exact `Exp3_PTO_GRPO/eda/results/...` source; figures copied
(never symlinked) by [`sync_figures.py`](sync_figures.py); sign conventions stated at every
table (the EDA's K tables report K=0−K=5 — this paper flips them); grader named on every number,
levels never compared across graders; behaviour claims name their denominator. Cite the ICLR 2025
paper as the SSI-FM *workshop* poster (canonical BibTeX in
[`../2025_iclr_pto_lookahead/README.md`](../2025_iclr_pto_lookahead/README.md)).

## Overleaf

`overleaf.zip` (gitignored; regenerate with `make_overleaf_zip.py`) holds exactly what Overleaf
needs and nothing else: `main.tex`, `sections/*.tex`, `figures/*.png`, `refs.bib`, `acl.sty`,
`acl_natbib.bst`. Upload it as a new project (New Project → Upload Project), set the compiler to
**pdfLaTeX** and the main document to `main.tex`; Overleaf runs BibTeX itself. The draft is in
`[review]` mode (line numbers, anonymous byline), which is the right mode for supervisor comments;
`\usepackage[final]{acl}` in `main.tex` restores the author block and drops the line numbers.
Figures are already cropped by `sync_figures.py`, so nothing in the zip depends on the repo.

## Build (MiKTeX on Windows — see ../README.md)

```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```

To eyeball the layout, the repo `.venv` has PyMuPDF: `fitz.open("main.pdf")[p].get_pixmap(dpi=100).save(...)`.
