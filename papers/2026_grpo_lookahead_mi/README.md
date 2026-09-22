# GRPO with Look-Ahead in Motivational Interviewing

*Rewarding a Therapist Turn by Where It Leads*

**THE submission — the single live paper** (Lior, 2026-09-04: "Archive P2, we are going with P1").
**Target: ARR October 2026 cycle** (submission **2026-10-12**, commitment 2026-12-20; the single
cycle feeds **NAACL 2027** and **COLING 2027**, and the venue is chosen in December once reviews
exist). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory unnumbered
Limitations (page-exempt), optional Ethics Statement (page-exempt). `acl.sty` builds in `[review]`
mode (line numbers, anonymized); switch to `[final]` for camera-ready. **The body ends exactly at
the bottom of page 8** (Limitations opens page 9); 25 pages in all.

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

**Cleaned up 2026-09-17** (Claude, on Lior's "clean it up for me and my supervisors"; no
number changed; logged in `NUMBERS.md` § "2026-09-17"):

- **`main.tex` reduced to what compiles the paper.** Gone: the `acl.sty`-missing fallback branch,
  the `\todo`/`\note` draft macros and their `\ifdraft` switch (nothing used them), the unused
  `\mici`/`\GRPO` shorthands, and the unused packages (`multirow`, `amssymb`, `xcolor`, `array`,
  `subcaption`). The comments left are a three-line header, the float-packing note, the
  author-block note and the camera-ready Acknowledgements reminder.
- **Section files renumbered contiguously** (the gap the retired `07_mechanism` left):
  `08_measurement` → `07_measurement`, `09_discussion` → `08_discussion`, `10_limitations` →
  `09_limitations`, `11_ethics` → `10_ethics`. Section numbers in the paper are unchanged.
  `overleaf.py push` removes the old names on Overleaf.
- **`refs.bib` regrouped by topic** (seven groups); the dated "added/verified on …" comments are
  gone, one line keeps the warning that `baruch2025pto` is the SSI-FM *workshop* paper. Entries
  verbatim; the six uncited entries stay (they do not render).
- **§3 states the objective.** The reward equation is numbered (Eq. 1) and Algorithm 1 cites it;
  a new "The update" paragraph gives the GRPO objective (Eq. 2, the DeepSeekMath form that TRL's
  `loss_type="grpo"` implements) and says that with one policy update per sampled batch (new
  Table 5 row, from `grpo_inner_iterations: 1` in both arms' `run_metadata.json`) the ratio is one
  and the clipping is inactive, so the step is the advantage-weighted policy gradient with the KL
  penalty. "PPO-clipped step" is retired. Algorithm 1 gained `π ← π_n`, §3 defines π (the policy
  being updated) beside π_n (the iteration-start policy), and the rollout is stated to use π,
  which is what the trainer does (the reward function rolls out with the live `trainer.model`).
  **Figure 1 redrawn** to match: rollout nodes labelled π, the update box an objective
  (maximise Σ A_g log π − β KL) rather than a gradient plus a penalty.
- §6: the "(the held-out judge differs; see below)" parenthetical folded into its sentence. C.8
  shortened to one paragraph, which also removed a near-empty page (four lines of Appendix C had
  spilled onto their own page before the float page). 21 pages; the body still ends at the bottom
  of page 8; `build.py` CHECK OK.

**Submission-readiness pass, 2026-09-17** (Claude, on Lior's "make the paper more ready for
submission"; logged in `NUMBERS.md` § "2026-09-17 (b)"):

- **Mechanical audit against the ARR CFP** (fetched that day): PDF metadata carries no author or
  title; all 22 embedded fonts are Type 1, none Type 3; no identifying string in the sections (the
  self-citation is third person); Limitations present and page-exempt; body ends on page 8.
- **A false claim in §3 fixed.** "prompts and number of gradient steps are identical across $K$"
  was never true: the prompts are sliced from each policy's own conversations, so the counts
  differ (**1,128 optimizer steps for K=0 vs 1,070 for K=5** over ten iterations; per iteration
  80–158 vs 70–136, `compute/cost/tables/compute_by_iteration.md`). §3 now says the loss, the
  advantage normalisation, the KL penalty and the prompt-construction rule are identical; the step
  counts are a Table 5 row and are quoted beside the oracle-call counts in the Limitations
  ("approximately matched by construction" — K=5 in fact took *fewer* steps).
- **The iso-compute reading is now disclosed** (one passage in the Limitations "Matched
  iterations are not matched cost" paragraph, quoted from `budget_sweep_GRPO_K_*`): 27.9 vs 51.2
  GPU-hours for the two runs; at ~13 GPU-h the turn-level arm leads (dz −0.74 / −0.78), at ~23
  GPU-h level under the training oracle (dz 0.07, n.s.) and look-ahead ahead under the held-out
  judge (dz 0.33, p_holm .012), beyond K=0's total budget the best-checkpoint comparison of §5.
  ⚠ This touches the 2026-08-27 "iterations only" decision: the axis is unchanged (no
  GPU-hour figure or table, no budget analysis in the body), but the Limitations no longer say
  "nothing here compares the arms at matched cost". Lior can revert the passage if he wants the
  stricter line.
- **Reproducibility rows added to Table 5**: software versions (TRL 1.4.0, transformers 5.8.1,
  PEFT 0.19.1, from `requirements.txt`), hardware (one A100, Google Colab), optimizer steps.
  Appendix C.7 now also says where the step counts come from and carries `\label{app:repro-cost}`.
- **[`CHECKLIST_ARR.md`](CHECKLIST_ARR.md)** (local only, never pushed): draft answers to every
  Responsible NLP checklist question with the backing section, the generative-AI disclosure
  wording, and the pre-submission list (supervisors' read, anonymised code archive as a .zip —
  the CFP rejects cloud-drive links — the preprint option, the two explicit checklist statements).
- References: a currency pass (arXiv preprints since published; canonical DOIs/URLs) was run by a
  web-verifying agent; its verified changes are logged in `NUMBERS.md` under the same heading.

**Refactored 2026-09-17 (c)** (Claude, on Lior's "the story is GRPO with look-ahead in MI and a
deep analysis"; every new number in `NUMBERS.md` § "2026-09-17 (c)"):

- **Two new EDA families feed the paper**: `lookahead/process` (the `MIPROC` utterance-level MI
  process coder — one MITI/MISC-style code per therapist utterance, one valence per patient
  utterance, both graders, all 2 × 11 × 96 = 2,112 GRPO evaluation conversations) and
  `lookahead/text` (sentence-embedding repertoire / drift / diversity, judge-free).
- **The body is now the process analysis.** §6 *What look-ahead teaches the therapist* (code
  mix, the judge-free marker, where the graders disagree, embedding space) and a new §7 *What the
  therapist's turns do to the patient* (yields, responsiveness, the within-session change-talk
  trajectory), with **Figure 3** (`process_grpo.png`: code mix × 2, yields, trajectory) and
  **Table 3** (the process endpoint under both graders). §5 is shorter; the saturation section is
  **Appendix E** whole (Table 7); the marker figure is single-panel in Appendix A (Figure 4);
  Appendix A also gains the held-out process figure (5), the responsiveness trajectories (6) and
  the embedding-space figure (7); Appendix C gains C.3 *The utterance-level process coder*
  (codebook, numbering, parity with MITI/PCT). Contributions are (i) the method and (ii) the
  utterance-level account; the abstract and §1 lead with praise-after-sustain-talk vs
  reflection-after-change-talk. The mechanism paragraph and the scope paragraph are compressed
  (scope now a Limitations paragraph). Body ends at the bottom of page 8; 25 pages.
- `render_paper_figures.py` now draws Figures 2–7, 10 and 11 (`process()`, `process_heldout()`,
  `responsiveness()`, `textspace()` read `process.xlsx` / `text.xlsx`); `sync_figures.py` still
  copies only the two level grids.
- Section files: `06_behaviour.tex` → `06_therapist.tex`, new `07_patient.tex`,
  `07_measurement.tex` → `E_saturation.tex`. `overleaf.py push` removes the old names.

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
best checkpoint), and it decides what kind of therapist the policy becomes: coded utterance by
utterance, the turn-level policy learns non-specific praise delivered after the patient's sustain
talk, and the look-ahead policy learns complex reflections delivered after the patient's change
talk, which are followed by change talk 85–89% of the time.

## Section map (files under `sections/`)

| file | section | content |
|---|---|---|
| 00_abstract | Abstract | |
| 01_intro | §1 | the turn-only default; MI as the setting; GRPO with look-ahead and its PTO lineage; the controlled pair; results + the two caveats; three contributions |
| 02_related | §2 | GRPO; multi-turn RL for dialogue (incl. multi-turn GRPO); look-ahead/search in preference learning + PTO; reward hacking & LLM judges (+ over-optimisation, ensembles); MI (+ AnnoMI, BOLT) |
| 03_method | §3 | **Method**, one section in three subsections (Doron's 2026-09-21 note: unify 3 and 4, setup before algorithm). **§3.1 Task, simulator and oracle** — task/simulator/oracle; **why MI**. **§3.2 GRPO with look-ahead** — notation; Figure 1 = the group schematic; **Algorithm 1** = the iterative loop; the look-ahead reward (the $\tau_K$ equation); why the transfer is not trivial; minimum context length; cost. **§3.3 Evaluation design** (`sec:setup`) — instruments; the process coder; terminology; the two arms; evaluation & statistics. The former `04_setup.tex` is retired; results sections are now §4–§6 |
| 05_reward | §5 | *Results: reward and evaluation instruments* — Figure 2 + **Table 1 (endpoint, every instrument, both graders)**; endpoint + gain ratios (both anchors); onset + the K=0 decline + the best-checkpoint steelman; the replicate draw |
| 06_therapist | §6 | *What look-ahead teaches the therapist* — **Table 2 (the clear-case excerpt)**; **Figure 3 (code mix × 2, yields, within-session change talk)**; the two policies learn different behaviours (praise vs complex reflections, the MI-adherent share, open-question turns vanish in both, the persuasion residue); the judge-free marker (Appendix Figure 4) + the MICI composition; where the graders disagree; embedding space (Appendix Figure 7) |
| 07_patient | §7 | *What the therapist's turns do to the patient* — **Table 3 (the process endpoint, both graders)**; yields per code; responsiveness (reflects change talk / praises sustain talk; Appendix Figure 6); the session as a whole (change-talk trajectory, patient turn length, the disengagement cue) |
| 08_discussion | §8 | the horizon selects which behaviour pays (+ the compressed mechanism paragraph → Appendix B); conclusion with the process-coding recommendation |
| 09_limitations | Limitations (page-exempt) | one run per arm; evaluation draws; matched iterations ≠ matched cost; K∈{0,5}; one optimiser, one regime (moved from §8); the continuation pressure (summary; numbers in Appendix A); simulation only / in-sample / same-model patient / no human validation / the response cap; instruments and the process coder (reward is an outcome, MITI reliability, the coder's one-code-per-turn construct and its partial parity with MITI) |
| 10_ethics | Ethics (page-exempt) | |
| A_tables | Appendix A | by-iteration table, the judge-free marker figure, the held-out process figure, the responsiveness trajectories, the embedding-space figure, level grids ×2, channel forest, tail audit figure (+ the rollout-audit numbers in the intro text) |
| B_mechanism | Appendix B | the mechanism analysis in full; **Figure B.1 = faithfulness by prefix length, both graders** (`faithfulness_grpo.png`, added 2026-09-22 on Doron's note) in B.1 (`app:faithfulness`) |
| C_repro | Appendix C | configuration, instruments, **C.3 the utterance-level process coder** (codebook, numbering, parity), prompts, the marker, anti-degeneracy, statistics, cost accounting, artifacts |
| D_example | Appendix D | utterances 1–9 of both iteration-10 conversations with persona 93, verbatim; selection rule and scores |
| E_saturation | Appendix E | *Saturation of the training oracle at the winning checkpoint* (the former §7, moved whole 2026-09-17; Table 7 = the per-instrument agreement table) |

## Scripts

- [`sync_figures.py`](sync_figures.py) — copies (and crops) every EDA-rendered figure the .tex
  references; `--check` reports drift. Does **not** cover Figures 1, 2, 3, 6 and 7, which the
  two render scripts below draw.
- [`render_schematic.py`](render_schematic.py) — draws Figure 1 (the GRPO-group schematic) at
  page width; reads no data.
- [`render_paper_figures.py`](render_paper_figures.py) — draws Figures 2, 3, 6, 7 and the Appendix B faithfulness figure from the
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
& ..\..\.venv\Scripts\python.exe overleaf.py push     # this folder -> Overleaf, one commit per changed file (--single: one commit)
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
- The full pre-submission list, with the checklist answers, is in [`CHECKLIST_ARR.md`](CHECKLIST_ARR.md).
