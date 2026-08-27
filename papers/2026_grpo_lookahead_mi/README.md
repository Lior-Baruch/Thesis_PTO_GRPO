# Scoring the Continuation

*$K$-Turn Look-Ahead Rewards for Group-Relative Policy Optimization in Motivational Interviewing*

**Target: ARR October 2026 cycle** (submission **2026-10-12**, commitment 2026-12-20; the single
cycle feeds **NAACL 2027** and **COLING 2027**, and the venue is chosen in December once reviews
exist). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory unnumbered
Limitations (page-exempt), optional Ethics Statement (page-exempt). `acl.sty` builds in `[review]`
mode (line numbers, anonymized); switch to `[final]` for camera-ready.

**Provenance.** Revived 2026-08-27 on Lior's instruction as a second ARR submission beside the
2×2 paper. The content is ported from the archived ICLR-formatted draft at
[`../archive/2026_grpo_lookahead_mi/`](../archive/2026_grpo_lookahead_mi/) (retired earlier the
same day when the ICLR plan was dropped) with three changes:

1. **Format:** ACL/ARR instead of ICLR (preamble mirrors the 2×2 paper's `main.tex`).
2. **Framing:** PTO is now discussed openly as the lever's origin — `baruch2025pto` is cited in
   the intro, related work, and discussion as the predecessor that introduced $K$-turn look-ahead
   with preference trees + DPO, and this paper's contribution is **moving the lever to GRPO**
   ("GRPO with $K$-turn look-ahead"). The PTO *arms* of Exp3 still appear **nowhere as data**;
   every full-grid statistic is the 22-GRPO-state recomputation (`*_grpo` artifacts).
3. **Axis: iterations only** (the same 2026-08-27 decision as the 2×2 paper). The ICLR draft's §5
   (cost/budget: GPU-hour totals, the budget sweep, crossover rungs, crossed-grader budget
   verdicts) is **deleted**, not ported; the honest-cost content survives as one Limitations
   paragraph (oracle calls ≈matched: 302,541 vs 289,983; ≈393k K=5-only patient calls; median
   1.92× per-step wall-clock) and a one-line ≈79 GPU-h Ethics total. `compute_trajectory_grpo`
   and `api_calls_grpo` figures are dropped; the budget machinery stays EDA-only.

**Domain:** Exp3, the **two GRPO arms** — `GRPO_LA0` and `GRPO_LA5`, matched MCL=12, G=8, 96
personas, 8 instruments, 10 iterations each, scored by two graders (gpt-4o-mini = the training
oracle; Claude Haiku 4.5 = held out). 2 arms × 11 states = 22 model states.

## The argument in one line

Scoring a candidate therapist turn by the $K$-turn continuation it leads to, rather than by the
turn itself, more than doubles what group-relative RL extracts from the same oracle — and it is
the difference between a policy that learns motivational interviewing and one that learns to
flatter the judge.

## Section map (files under `sections/`; no 05 gap — renumbered at the revival)

| file | section | keeps from the ICLR draft |
|---|---|---|
| 00_abstract | Abstract | cost sentence removed; lineage sentence added |
| 01_intro | §1 | new lineage ¶ (PTO → GRPO); "three things omitted" → two; contribution (i) recast as "GRPO with K-turn look-ahead" |
| 02_related | §2 | PTO ¶ expanded (what PTO is; what we keep vs change) |
| 03_setup | §3 | + matched-iterations sentence pointing at Limitations |
| 04_reward | §4 | unchanged |
| 05_behaviour | §5 | unchanged (was 06) |
| 06_mechanism | §6 | unchanged (was 07) |
| 07_measurement | §7 | unchanged (was 08) |
| 08_discussion | §8 | "wrong denominator" ¶ removed; scope ¶ now names PTO as the lever's origin |
| 09_limitations | Limitations (page-exempt) | was appendix C; + "matched iterations ≠ matched cost" ¶ replacing "reconstructed compute" |
| 10_ethics | Ethics (page-exempt) | compute line reduced to the ≈79 GPU-h total |
| A_tables, B_repro | appendices | api_calls figure dropped; B gains "Cost accounting behind the disclosures" |

⚠ **Dual-submission overlap with the 2×2 paper** (`../2026_pto_grpo_mi/`, same ARR cycle): the
GRPO K-lever numbers (+0.765/+0.616, the gain ratios, the overpraise marker, the saturation
mechanism) appear in both papers — here as the subject, there as cells of the interaction. The
papers never cite each other's prose and their claims are disjoint (single-lever deep-dive vs the
optimizer×horizon interaction), but ARR's multiple-submission policy is a judgment call the
authors must clear with the supervisors before 2026-10-12.

## Conventions

Same as the repo standard (see [`../README.md`](../README.md)): every number in
[`NUMBERS.md`](NUMBERS.md) with its exact `Exp3_PTO_GRPO/eda/results/...` source; figures copied
(never symlinked) by [`sync_figures.py`](sync_figures.py); sign conventions stated at every
table (the EDA's K tables report K=0−K=5 — this paper flips them); grader named on every number,
levels never compared across graders; behaviour claims name their denominator. Cite the ICLR 2025
paper as the SSI-FM *workshop* poster (canonical BibTeX in
[`../2025_iclr_pto_lookahead/README.md`](../2025_iclr_pto_lookahead/README.md)).

## Build (MiKTeX on Windows — see ../README.md)

```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```
