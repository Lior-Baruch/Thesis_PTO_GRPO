# GRPO with Look-Ahead in Motivational Interviewing

*Rewarding a Therapist Turn by Where It Leads*

**THE submission — the single live paper** (Lior, 2026-09-04: "Archive P2, we are going with P1").
**Target: ARR October 2026 cycle** (submission **2026-10-12**, commitment 2026-12-20; the single
cycle feeds **NAACL 2027** and **COLING 2027**, and the venue is chosen in December once reviews
exist). ACL long-paper format: 8-page body, unlimited references/appendix, mandatory unnumbered
Limitations (page-exempt), optional Ethics Statement (page-exempt). `acl.sty` builds in `[review]`
mode (line numbers, anonymized); switch to `[final]` for camera-ready. ⚠ **The body runs about
two and a half pages over:** the Conclusion ends about 42 lines into page 11 (build of 2026-10-05
after step 11a, 32 pages in all; the limit puts its end on page 8; the visible review-note markers take
some of those lines and are hidden in step 13). It grew with Doron's §1–2
rewrite, his passes 2–3, Figure 4's return to the body, the §2 judges paragraph, on
2026-10-01 the figures redrawn at print width (+36 pt) and the worth-a-look citations, and on
2026-10-05 step 11a (about ten lines). Accepted as is: the length pass is
deliberately **last** (Lior, 2026-09-22; re-confirmed 2026-10-01 and 2026-10-05: after the
contributions/abstract pass that goes to Doron and Lior's own read-through). Do not trim ahead of that.

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
  patient and therapist prompt templates, and the ten patterns of the keyword marker (then called the lexical over-praise marker).
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

**Supervisor pass 1, 2026-09-21/22** (Doron's Overleaf edit of 2026-09-21, pulled as `944f26a`;
Lior: keep ALL his edits, answer his notes; nothing pushed until he said so):

- **§1 rewritten by Doron** — opens two levels up (RL from model-generated feedback; verifiable
  vs non-verifiable domains; multi-turn = temporal credit assignment; counselling; MI), the
  results-preview paragraphs are commented out (his call: no "executive summary"), and a new
  two-part contributions paragraph he marked *"most important — we will get back to it at the
  end"* (**open**). **§2 restructured** by him into four paragraphs (GRPO; delayed credit; reward
  models and judges; MI) with a new hinge — in dialogue the future is jointly produced with the
  interlocutor. Five of his citation keys were renamed onto existing entries; "from our group"
  anonymised (`[review]` mode). His American spelling is left for the final copyedit.
- **§3 + §4 merged into one §3 Method** on his note (*"unify 3 and 4; setup before algorithm"*):
  3.1 Task, simulator and oracle (+ Why MI) · 3.2 GRPO with look-ahead · 3.3 Evaluation design
  (`sec:setup`). `04_setup.tex` retired; results are §4–§6. His §3 notes answered: the arms differ
  in the $K$-turn rollout and hence the rewarded transcript (K=0 scores the candidate alone); the
  cost clause is out of §3.2 (Limitations/Appendix C keep it); "Why the transfer is not trivial"
  deleted (no PTO arms here — one PTO-free sentence on the one-sample estimate kept); the
  process-coder sentence rewritten; the 96 personas cite Yosef et al. 2024 (introduced) + the PTO
  paper; and a **new Appendix B.1 figure** (`faithfulness_grpo.png`, `render_paper_figures.py::
  faithfulness`, from `mechanism.xlsx::faithfulness_curve_long`, GRPO arms, both graders).
- **Open from his pass:** the Q1+Q2 training-reward justification (his note stays in §3.1; "here
  briefly, also in intro and/or discussion — the excuse is following Yosef et al."), the
  contributions paragraph, `rafailov2023dpo` now uncited (DPO is named in §1), the abstract still
  opens with the pre-rewrite pitch, and the length (see the top). More notes are coming; the
  length pass waits for them.
- A pre-existing `\S<CR>ef` corruption in `E_saturation.tex` (ours, not his) repaired.

**Supervisor passes 2–3 and the plan, 2026-09-24** (Doron's Overleaf edits of 22 Sep 23:26 and
23 Sep 18:25, pulled as `bca1d37`: 18 bracketed notes on §4–§5, 17 on §6, all open; 37 notes open
with pass 1's two). Lior's decisions before any text changes, chosen from options:

- **Vocabulary.** `K=0` / `K=5` only in the results (the words turn-level / look-ahead only in §3);
  "the K=5 policy" (what it says) vs "the K=5 run" (training); iteration wording, **iteration 0 =
  Base**; "utterance coder", and §3.3 says each judge does two jobs; "significant" defined once in
  §3.3 (no "clears Holm"); MI terms defined once in §3.1, no metaphors in the results; where both
  judges appear, name both and keep numbers few.
- **Judges.** gpt-4o-mini (still "training oracle") is the main judge for everything in the body;
  the held-out judge stays in Table 1 plus one sentence per results section; the rest moves to a
  new appendix. Appendix E and the Limitations must be reframed to match.
- **Structure.** The all-instrument grid (old Figure 8) replaces Figure 2 in §4, redrawn at page
  width; **one shared Base** everywhere (the two base draws pooled; the paper no longer compares
  them); a **complete score table** per judge (21 rows × 9 instruments, mean + a star on the better
  run where the K contrast is significant) in the appendices; Table 4 retires.

The plan, one step at a time, each section ending in a build, Lior's read, a commit and a push:
0 pull (done) · 1 data and figures (done 2026-09-24, below) · 2 §3 (done) · 3 §4 (done) ·
4 §5 (done 2026-09-24, below) · 5 §6 (done 2026-09-24, below) rewritten
around responsiveness → change-talk persistence → the session (heads-up to Doron first) ·
6 appendices (done 2026-09-24, below: the Claude appendix, the complete tables, Appendix E and
the Limitations reframed) ·
7 carry through (done 2026-09-24: abstract claims, §7 mechanism paragraph rebuilt on measurements,
the Limitations paragraph on patient replies inside the K=5 reward, Ethics wording) · 8 Doron's
contributions paragraph, then the
abstract · 9 layout fixes · 10 length pass, checklist, code zip, submit ≈ 9 Oct.

**Step 1 (2026-09-24).** New EDA family `lookahead/shared_base` recomputes every Base-dependent
number on the shared Base under both graders, plus the complete score tables and the change-talk
persistence analysis (P(patient change talk | the patient's previous code), which replaces the
unconditioned "yield" as the patient-side claim); `lookahead/mechanism` gained the GRPO praise
premium (does the training reward pay for praise, net of prevalence). `render_paper_figures.py`
draws the new grids (`levels_grid_grpo_<judge>.png`, not yet included) and now takes figure
names on the command line. Every number that moves, old → new, is in `NUMBERS.md` § "2026-09-24".

**Step 4 (2026-09-24), §5.** Training oracle only in the body, plus one held-out sentence; Doron's
15 notes answered in the text, his two edits kept. New: an honest timing sentence (K=5's lower
MI-inconsistency is K=0's late praise, not an MI gain of K=5: K=0 is significantly lower at 2, 6,
9 and K=5 only at 10), the "keyword marker" named and its three-way agreement spelled out, and a
column-width Figure 4 (`text_grpo_body.png`, panels a–b of Appendix Figure `text_grpo.png`) with
the encoder cited (`reimers2019sbert`). The figures that read Base data now read
`lookahead/shared_base`.

**Step 5 (2026-09-24), §6.** Rebuilt as responsiveness → change-talk persistence → the session;
Doron's 17 notes answered by the rewrite. Table 3 is training-oracle only, single-column, with a
Base column and two new reply rows (`pers_after_st`, `refl_after_st`, added to the EDA for this):
the honest addition is that K=5 answers sustain talk with persuasion (0.39, Base 0.24) as K=0
answers it with praise. The per-code yield is retired (placement-confounded); Figure 3c is now
persistence. His "??" is answered by the measured praise premium (new appendix subsection + figure,
C.4 since step 6). The held-out Table 3 is Appendix Table 5; the responsiveness figure shows both
judges × four replies.

**Step 6 (2026-09-24), appendices.** A = training-oracle extras, now led by the complete score table
(Table 4; the old `tab:byiter` retires); **B = the held-out judge** (`sections/A2_heldout.tex`: its
grid, complete score table, process figure and table, and one paragraph of where it agrees and
differs); C mechanism (the praise premium is C.4); D reproducibility (+ the Base-pooling sentence);
E examples; F saturation, reframed as a limit of the main judge on 21 states. Each appendix starts
on a new page. The Limitations drop the base-vs-base sentence and call the held-out judge and the
six outside instruments "the checks", not "the load-bearing evidence".

**Supervisor passes 4–5 and the plan, 2026-09-28** (Doron's Overleaf edits of 26 Sep 10:20 UTC,
notes on §7, the Limitations and the Ethics, and of 27 Sep 11:30 UTC, a new related-work block at
the top of §2; pulled as `0f2868e` and `a3c299a`). Lior went through every note on a note-cards
page (private claude.ai artifact https://claude.ai/artifact/NSVdJcCvmmpGVFmbptBfeY; his pick per
note is stored there, db collection `picks`, doc id = card id) and chose keep / rephrase / cut /
later for each. This plan follows those picks and supersedes steps 8–10 of the 2026-09-24 plan.

- **Doron's notes stay in the text.** `main.tex` defines `\dnotedone{how}{note}` (green,
  "[handled: … | Doron: …]") and `\dnoteopen{why}{note}` (red); `\dnotesfalse` hides all of them
  (flip it before submission). Once a section is checked, its handled notes are commented out
  (`% \dnotedone{…}`: still in the source, gone from the PDF), so the PDF shows the section under
  review plus every open note. The 44 handled notes of passes 1–3 are back in §1 and §3–§6 as such
  comments, beside the text that answered them. `build.py` counts the literal `??` in his notes as
  unresolved references until the notes are hidden; check `main.log` for "undefined" instead.
- **MITI, option A.** The paper drops only MITI's seven behaviour counts: the behaviour-channel
  figure (`fig:forest`, Appendix A), the MITI half of Appendix D's coder check (the PCT half stays)
  and the MITI clauses of the Limitations. MITI stays one of the eight instruments through its four
  global ratings. ⚠ Found while checking: Appendix D's "the MITI coder counts every function a long
  turn performs" is false. MITI's behaviour total equals the number of therapist turns in 79% of
  conversations under gpt-4o-mini and 96% under Claude; the gap to the utterance coder comes from
  MITI's seven-code list having no praise and no "other" code. Option A removes the sentence.
- **The intermediate-K reminder.** Doron's "indeed this is a major limitation - in case you have
  time for more runs" stays in the Limitations as an open note; whether to run an intermediate K
  on the lab server time he mentions is Lior's call.

The plan. Each step: pull, edit, build, Lior's read, commit, push to Overleaf on his go; then the
step's handled notes are commented out. Rewritten 2026-09-30 after Lior's own read: his notes are
steps 7–8, and the former steps 7–10 are now 9–12. **Re-ordered 2026-10-01 (Lior):** a simple
contributions rewrite now (step 10), then his read-through of the whole paper as a note loop
(step 11: he sends notes, Claude proposes changes, he approves, repeat), then the contributions,
abstract and Q1+Q2 with Doron (step 12), and the length pass last (step 13). The former step 11
(layout) folds into the read-through. **Re-ordered again 2026-10-05 (Lior):** step 12 runs
FIRST, drafted with Lior and pushed to Overleaf so Doron can read it and send notes while Lior
does his read-through (step 11); his notes on it fold into the loop. Step numbers are kept (the
ledger cites them), so the table below lists 12 before 11.

| Step | What | Status |
|---|---|---|
| 1 | §7 Discussion: a plain summary first; the slogan cut; praise and change talk in plain words; the reward-hacking story one idea at a time; the conclusion's first sentence and the recommendations cut | done (`70218dd`), on Overleaf since 2026-09-28 with the note marks; approved 2026-09-29, handled markers commented out |
| 2 | Limitations. **Cut:** evaluation draws, one optimiser/one regime, reward for continuing, patient replies inside the reward, the circularity sentence. **Rephrase:** matched cost (plain words; keep the GPU-hours and the equal-compute result in one sentence), K ∈ {0,5} ("more runs are needed", his reminder open), simulation only (credit the extra measurements, then name the one gap: the patient), "without a model in the loop". Fix the stray `''`. Knock-ons: Appendix A's text and the rollout-audit caption point at the cut "continuing" paragraph; §3.3's "(Limitations)" points at the cut circularity sentence | done and approved 2026-09-29 (handled markers commented out; Lior asked for a source comment on evaluating with newly authored personas, "easy and cheap", beside the in-sample sentence). The call counts, the 1.92× median and the equal-GPU-hour d_z values moved to Appendix D.8 (`app:repro-cost`), which now states them itself; the saturation pointer survives as a plain sentence; the 31.9 vs 25.2 session lengths left the paper with their paragraph. Ledger block "2026-09-29" |
| 3 | Ethics: tone down the judges paragraph; cut the compute paragraph, move its licence sentence to Appendix D (Appendix D.8 no longer names the Limitations or the Ethics as the home of the GPU-hour totals, done in step 2); keep the section order (ARR's rule); re-point the ARR checklist answers C1 (its Ethics part) and B2 | done and approved 2026-09-29 (judges paragraph: Lior picked the neutral rewrite; the licence sentence closes Appendix D "Artifacts"); handled markers commented out |
| 4 | Count the therapist turns that hit the 200-token cap, per model state, in the EDA (local; no GPU, no API) → `NUMBERS.md` → the number replaces "many" in the Limitations | counted 2026-09-29 (EDA table `lookahead/shared_base/tables/cap_hits.md`: 97% of K=0 / 81% of K=5 therapist turns at iteration 10, 3% at the Base). After reading the sentence Lior found it too negative and CUT the cap from the Limitations (and §5's pointer to it); §3.1, §5 (Table 2's caption and the turn-length sentence), Table 9 and Appendix E still mention the cap. Approved 2026-09-29, handled marker commented out. Ledger block "step 4" |
| 5 | MITI option A (above); cut the Limitations' "least dependable" sentence; shorten the utterance-coder paragraph | done and approved 2026-09-29 (after his read also: D.2's "exactly one of seven behaviour codes" sentence removed; markers commented out): channel forest dropped, MITI row = "4 globals", Appendix D.3's coder check is PCT-only, Limitations coder paragraph short with the human-check sentence (Lior's picks). 30 pages |
| 6 | Consistency: §6 and Appendix C.4's title still say "paid for" / "siblings", the words Doron flagged in §7; Appendix E's subsection titles still say "Turn-level reward" / "Look-ahead reward" | done and approved 2026-09-29 (after his read also: "premium" removed from the visible paper, Figure 12 relabelled): §6 "paid for … above their siblings" → "gave higher scores to … above the candidates that did not praise", "voices" → "expresses" (Doron's word); App C.4 retitled "How much the training reward favours praise", "pays" → "favours" / "premium"; App E titles "The K=0 / K=5 policy at iteration 10". No number changed; no flagged word left outside the abstract and Doron's intro |
| 7 | **Lior's read, 2026-09-30.** Table 1: no bold, two decimals. The embedding figure: first moved to the appendix, then (his review) kept in the body with clearer panels. Figure 8 (rollout audit) dropped, its paragraph shortened | done 2026-09-30 and REVISED the same day after Lior's review of the first pass ("do the fixes and also honour Doron's request"; `40f810a`..): Table 1 no bold AND two decimals (§4's restating prose follows); **Figure 4 stays in the body** with panels (a) variation by patient and (b) same-turn similarity — (b) shows both runs' turns growing alike, K=5 the higher at 7 of 10 iterations — and no appendix twin; Figure 8 dropped, the rollout check in two sentences without the "favours continuing" clause. Doron's "show diagram, explain" note visible, re-marked. Keyword-figure y-label unclipped. Body ends on page 11 again (the figure is back), 29 pages. Approved 2026-09-30 ("continue"), handled marker commented out, pushed to GitHub and Overleaf. ⚠ Found on the way: `overleaf.py pull` run while local commits are unpushed copies Overleaf's OLDER files over them (it compares files, not history); they were restored from git. Pull only when local == Overleaf, i.e. before editing. Ledger block "step 7" + its revision |
| 8 | Complete process tables (Lior: "do we have a full table of all iterations?" — no, only figures). EDA first: Table 3's measures at every iteration, the shared Base, stars from the per-iteration tests → `NUMBERS.md` → one table in Appendix A (training oracle), one in Appendix B (held-out) | done and approved 2026-09-30 ("Continue"), pushed to GitHub and Overleaf: no EDA change was needed (every value was already in `shared_base.xlsx`); `render_process_tables.py` prints both tables and checks iteration 10 against the two iteration-10 process tables. Lior's mid-step ask "bold each column with best score" applied to both new tables and to the complete score tables (Tables 4 and 7). Appendix tables renumbered (the new ones are Tables 5 and 8). 30 pages. Ledger block "step 8" |
| 9 | Doron's related-work block: merge into §2 as one bold-headed paragraph at about half its length; `\citet` for Yuan et al. and Wu et al.; the four missing references (`wu2025metarewarding`, `wang2026serpo`, `wang2026dynamicrubric`, `chu2026jzero`) from Doron or found and verified | done and approved 2026-09-30, pushed to GitHub and Overleaf (handled markers commented out; Doron's original block stays in the .tex as a comment, Lior: "dont delete his original"): one paragraph "Judges that change during training" after the judges paragraph (Lior's placement pick), one shared contrast, Doron's closing sentence kept, his original commented out beside it; the four references found and checked against their pages. Body now ends ~5 lines above the foot of page 10 (this row said "~5 lines into page 11" until 2026-10-01; the step-9 PDF shows page 10). Ledger block "step 9" |
| 10 | A simple contributions rewrite before Lior's read-through (Lior, 2026-10-01): Doron's "most important" note stays OPEN and his original paragraph stays in the .tex as a `%` comment, so the paragraph is revisited with him in step 12 | done and approved 2026-10-01 (his one change: the closing persuasion sentence cut), pushed to GitHub and Overleaf; the note stays open for step 12: still "twofold" and still "Second, and more importantly" (Doron's structure); (1) look-ahead moved to GRPO, the controlled pair, ahead on all eight instruments under both judges (§4); (2) utterance coding: turn-level reward teaches non-specific praise incl. in reply to sustain talk, look-ahead teaches reflecting change talk and the patient keeps expressing it (§5–6). No new number. The "behavior s" typo now lives only in Doron's commented original. The body still ends on page 10 (its last line) |
| 10b | The float atlas (private artifact https://claude.ai/artifact/CQGDKtm5zJFPxZRNj3M6zt: every figure, table and the algorithm, where it prints, every citing sentence) and its 62 checked "worth a look" items, fixed before the read-through (Lior: "fix the worth a look before my notes"). His picks: figures redrawn at print width; Figure 4(b)'s "seven of the ten iterations" dropped; the held-out K=0 MI-adherent lead at iterations 2–3 not added; the bold rule kept | done 2026-10-01 and pushed to GitHub and Overleaf at Lior's request before his notes (his read of it folds into step 11): 45 items fixed in the text, captions and the two figure scripts, 12 left by decision (informational, float placement, his picks), the 5 that a verification pass found only partly fixed then finished (Figure 8 caption, C.1's pooled-interval sentence, the 5% validation split is by conversation, 86–89% is the training oracle's, Table 9's sub-batch footnote). Every figure now prints at its include width; Figures 1, 3, 4 grew +7/+14/+15 pt, and the body now ends 15 lines into page 11. The atlas itself predates these fixes (its page numbers and some explanations are now stale). Ledger block "2026-10-01 — the float atlas's worth-a-look items" |
| 12 | **Runs before 11 (Lior, 2026-10-05).** The contributions paragraph (Doron's "most important" note), then the abstract; the Q1+Q2 justification in the intro and Discussion (his pass-1 note, open; §3.1 has it). Drafted with Lior, then pushed to Overleaf for Doron's read and notes. Lior's picks: Doron's two sentences as the frame + one result sentence each; abstract rewritten to the intro's framing, ≤200 words (ACL); Q1+Q2 in the intro AND the Discussion | done, approved and pushed to GitHub and Overleaf 2026-10-05 (`a8f2714`..): abstract 289 → 194 words; contributions as "We make two contributions:" + two bullets (Lior disliked "Our contributions are twofold"; `enumitem` in `main.tex`); no new number; body ends ~32 lines into page 11. **Step 12's two green handled markers (§1 contributions, §3.1 Q1+Q2) and the Discussion's are LEFT VISIBLE on purpose (Lior) for Doron's read; comment them out after he has read it.** Ledger block "step 12" |
| 11a | **Pre-read fixes (planned with Lior 2026-10-05 by question round, from a recap of the paper and the direction check in [`../../meetings/2026-10-05_doron_direction_check/`](../../meetings/2026-10-05_doron_direction_check/RESULTS.md)); done before his read-through, reviewed by him as ONE batch, pushed to Overleaf after his approval (pull first; merge any Doron edits).** (1) **EDA first:** the per-iteration K0-vs-K5 reward-direction cosines (noise-corrected) and K=0's iteration-9 reversal; a count of the malformed chat-marker leak (training candidates, evaluation turns, its within-round reward correlation). (2) **Claim fixes:** Ethics "its score went up" → its training reward favoured praising replies; Discussion "changed which shortcut the reward favoured" → "which MI-inconsistent habit the policy learned"; patient section "For the same reason" → "We therefore"; Conclusion "whatever the patient said" reworded. (3) **Mechanism appendix, "The update-direction proxy"** rewritten by iteration from the EDA port (same at 1–3, partly different from 4, when K=5 first leads; unrelated at 9–10). (4) **One clause in the therapist section:** the reward that produced K=0's iteration 9 favoured agreement instead of praise (fits the 0.22 → 0.08 → 0.41 dip). (5) **Leak:** one neutral sentence in the reproducibility appendix's "Anti-degeneracy" paragraph; nothing in the Limitations. (6) **Doron's text, edited as the author's call, his originals kept as `%` comments:** the Introduction's "converges toward non-specific praise regardless of the patient's preceding response" reworded; the hypothesis paragraph ("This makes MI particularly suitable…") moved up behind the MI paragraph; the WHOLE paper to **American** spelling (the verbatim prompts keep theirs). (7) **Additions:** Sotopia-RL (arXiv 2508.03905) in Related work; "1B" in the Abstract (≤ 200 words). (8) **Small fixes:** double parentheses + en dash in "The final policies", two decimals in the Results prose, whether the held-out dz 0.386 is significant, the ICC's "two K=0 states" named (iterations 8 and 10), Ethics "saturated design" reworded, Llama-3.2-1B named as the base model. **Declined:** the persistence-vs-share sentence, the simulator-and-persuasion sentence, Doron's direction analysis as an appendix paragraph. **Deferred to step 13:** MI motivated four times (first cut on the length list) | done 2026-10-05, awaiting Lior's one-batch review; local commits only (EDA `cde8a0a`, then the paper per section), not yet on Overleaf. Two departures from the plan, both in the ledger block "step 11a": the direction numbers come from the paper's OWN estimator (MiniLM, advantage-weighted, every gradient group), not from the meeting's gte run (they agree); and the iteration-9 clause does NOT say "agreement" (a lexical premium check found none: 0.12 SD, z 1.87), it says the praise premium dipped and the direction reversed. Abstract at exactly 200 words. Body ~42 lines into page 11, 32 pages |
| 11 | **Lior's read-through of the whole paper**, after step 12, as a loop: he sends notes, Claude proposes the changes, he approves, repeat until he is through; Doron's notes on step 12 fold in. The five items the 2026-10-01 README audit had parked for it were settled with him on 2026-10-05, before his notes: four fixed (the abstract names the training oracle for its 97% / 80%; §3.3's cross-judge rule scoped to questionnaire scores, so the 0.05 / 0.20 praise comparison no longer contradicts it; §1 expands and cites DPO, `rafailov2023dpo`; Appendix E.1 matches the script, which now ranks every therapist reply: Lior asked why only the first five, and widening the window changed nothing at the top), float crowding moved to step 13 | the four fixes done, approved and pushed to GitHub and Overleaf 2026-10-05 (`ff7660d`..`7631493`; ledger block "2026-10-05"); his notes not yet started |
| 13 | Hide the notes (`\dnotesfalse`, `main.tex`); comment out step 12's green markers once Doron has read it; length pass (the body ends ~32 lines into page 11 on 2026-10-05 and must end on page 8: a little over two pages, less the note lines); float crowding, after the cut (moved from step 11 on 2026-10-05); ARR checklist (its section, appendix and table pointers are stale since the 2026-09-24 restructure, e.g. "Table 5" for the configuration table that is now Table 9, "C.x" for what is now Appendix D, 22 model states for 21; re-point them AFTER the length pass, which will move them again); code zip; submit ≈ 9 Oct (deadline 12 Oct) | |
| later | Appendix E.2's typical case has two $K{=}5$ turns cut at the cap (utterances 6 and 8). A re-pick by the same median rule among the 39 of 96 personas whose first three $K{=}5$ turns end below the cap is possible (E.1's $K{=}5$ turns already do: 73, 66, 137 tokens). Lior: "maybe later" | parked |

**Framing.** PTO is discussed openly as the origin of $K$-turn look-ahead: `baruch2025pto` is
cited in §1, in §2 and twice in §3.1 (the personas; Q1+Q2 trained the predecessor PTO policy);
the abstract names preference-tree optimisation without a citation, and the Discussion does not
mention PTO. The contributions (§1) are **"We make two contributions:" + two bullets** (Lior removed "Our
contributions are twofold" on 2026-10-05): (1) look-ahead reward moved from PTO to GRPO, and,
"more importantly", (2) what each reward horizon teaches the policy, read from the utterance
coding. Each bullet is Doron's sentence plus one result sentence. The PTO *arms* of Exp3 appear **nowhere as data**.

**Axis: iterations** (the 2026-08-27 decision): every headline contrast is at matched iteration.
Cost is a disclosure only: Limitations ¶2 "Matched iterations are not matched cost" in plain words
(about the same oracle calls and training steps, about twice the time per step, 51.2 vs 27.9
GPU-hours) plus the equal-GPU-hour reading in one sentence; the exact figures are in Appendix D.8
(302,541 vs 289,983 oracle calls; 392,766 patient calls that only the K=5 run makes; a median
1.92× per step over iterations 3–10; at about 13 GPU-hours K=0 is ahead, d_z −0.74 training
oracle / −0.78 held out; at about 23 the runs are level under the training oracle, 0.07 n.s., and
K=5 leads under the held-out judge, 0.33, significant) and the optimizer-step counts in Table 9. The Ethics
Statement has no compute paragraph since 2026-09-29; the 51.2 + 27.9 = 79.1 GPU-hour sum appears
only in `CHECKLIST_ARR.md` (C1).

**Domain:** Exp3, the **two GRPO runs**, `GRPO_LA0` (K=0) and `GRPO_LA5` (K=5): matched MCL=12,
G=8, 96 personas, 10 iterations each. **21 model states** = one shared Base + 2 × 10 iterations:
the two runs' Base draws are pooled into one Base of 192 conversations (since 2026-09-24), so
2 × 11 × 96 = 2,112 conversations per judge, plus a second 96-conversation draw of the final K=5
policy. **Two judges, each doing two jobs** (the 8 instruments + one utterance-coder label per
utterance): gpt-4o-mini, the training oracle, is the main judge and the body's default; Claude
Haiku 4.5, the held-out judge, is named wherever it appears: in the body, Table 1 and short
agreement sentences in §4–§6; its full results in Appendix B; and beside the training oracle in
Figure 6, Appendix C, D.3, D.8, Appendix F and the Limitations. Two checks involve no LLM judge:
the keyword marker (Appendix D.5, Figure 5) and sentence embeddings (Figure 4). Most numbers come
from `lookahead/shared_base` (Appendix F included); Table 1 and the best-checkpoint d_z values
from `reward.xlsx::k_endpoints`; Appendices A and C from `mechanism.xlsx`; D.8 and the
Limitations' cost sentence from the compute/cost tables. `NUMBERS.md` names the source of each.

## The argument in one line

Scoring a candidate therapist turn by the $K$-turn continuation it leads to, rather than by the
turn alone, raises what the same GRPO optimiser extracts from the same oracle, and it decides
what kind of therapist the policy becomes. In a controlled pair of ten-iteration runs that differ
only in the horizon, K=5 ends ahead of K=0 on all eight MI instruments under both judges; its
Q1+Q2 gain over the Base is 1.3 to 2.5 times K=0's (training oracle 2.04 / 1.41, held out
2.50 / 1.30, against K=0's last / best checkpoint). Coded utterance by utterance (training
oracle), K=0 learns non-specific praise, late and unevenly (0.22, 0.08, 0.41 of its turns at
iterations 8, 9, 10), including in reply to a third of the patient's sustain talk; K=5 learns
complex reflections (0.02 of the Base's turns, 0.23 of its own) and reflects 0.26 of the
patient's change talk (K=0: 0.003), and under it a patient's change talk is followed by change
talk again in 97% of cases (K=0 80%, Base 75%). The caveats the paper carries with it: K=5 meets
sustain talk with persuasion (0.39 of its replies, Base 0.24), the two judges disagree on how
much K=5 still praises (0.05 of its turns under the training oracle, 0.20 held out), and each
horizon is a single training run.

## Section map (files under `sections/`)

Numbers are from `main.aux` (build of 2026-10-01, 31 pages). ACL numbers figures and tables
globally, so an appendix float is "Figure 9", never "Figure C.1". Body = §1–§7 (pp. 1–11; about
two pages over the limit). Limitations and Ethics are unnumbered and page-exempt (pp. 11–12),
references from p. 12, Appendices A–F pp. 16–31, each on a new page.

| file | section | content |
|---|---|---|
| 00_abstract | Abstract | 194 words (ACL limit 200; rewritten in step 12 to the intro's framing): RL from model feedback works on verifiable tasks; the judge sees nothing after the turn; MI; LA-GRPO extends look-ahead from preference-tree optimisation to GRPO; the controlled pair, K=5 ahead on all 8 instruments under both judges; praise vs complex reflections; persistence 97% vs 80% under the training oracle; persuasion in reply to sustain talk; one run per horizon |
| 01_intro | §1 Introduction | RL from model-generated feedback in non-verifiable domains; multi-turn dialogue as temporal credit assignment; counselling and MI; why MI suits the reward horizon (the hypothesis; moved up here in step 11a); the question + a short result preview (Doron's "converges … regardless" sentence reworded in step 11a, his original a `%` comment); LA-GRPO and its PTO lineage; why Q1+Q2 (PTO's reward, Yosef et al.'s validated questionnaires; one sentence, step 12); Doron's "most important" note marked handled (visible until he reads it); **contributions** = "We make two contributions:" + two bullets (`itemize` via `enumitem`), Doron's original paragraph kept as a `%` comment |
| 02_related | §2 Related work | bold-headed paragraphs: Group-relative policy optimisation · Delayed credit in multi-turn dialogue (Sotopia-RL added in step 11a; + an unheaded paragraph on simulated continuations: MCTS, VinePPO, PTO as a preliminary workshop paper) · Reward models and LLM judges (over-optimisation, sycophancy, self-preference, length) · **Judges that change during training** (Self-Rewarding, Meta-Rewarding, SERPO, DynamicRubric, J-Zero; Doron's original block kept as a `%` comment) · Motivational interviewing (+ AnnoMI, BOLT, RL-trained counsellors) |
| 03_method | §3 Method | **§3.1 Task, simulator and oracle** (`sec:task`): personas, session cap, the Q1+Q2 justification (Doron's note marked handled in step 12, visible until he reads it; the intro and Discussion now carry it too), *Why MI*. **§3.2 GRPO with look-ahead** (`sec:lagrpo`): notation; **Figure 1** = the group schematic (`fig:schematic`); **Algorithm 1** (`alg:loop`); the iterative loop; the look-ahead reward (Eq. 1, $\tau_K$; a one-sample Monte Carlo estimate); the update (Eq. 2); minimum context length (MCL=12; 86–89% prefix agreement → Appendix C.1, Figure 9). **§3.3 Evaluation design** (`sec:setup`): instruments (MITI = mean of its 4 global ratings); the utterance coder; judges (training oracle = main judge, held out → Appendix B); the two runs (K=0 / K=5; "policy" vs "run"); evaluation and statistics |
| 05_reward | §4 Results: Reward and evaluation instruments | **Figure 2** = every instrument by iteration, training oracle (`fig:headline`); **Table 1** = iteration 10, every instrument, both judges (`tab:endpoint`; two decimals, no bold); *The final policies* (gain ratios 2.04 / 1.41, held out 2.50 / 1.30); *Onset, and the K=0 decline* (the best-checkpoint comparison); *A second draw of the final policy* |
| 06_therapist | §5 What look-ahead teaches the therapist | **Table 2** = the clear-case excerpt (`tab:excerpt`); **Figure 3** = code mix ×2, change-talk persistence, change-talk share by session position (`fig:process`); *The two policies learn different behaviours* (praise under K=0 vs complex reflections under K=5, the MI-adherent share, open questions vanish in both, turns triple in length, the persuasion residue, one held-out praise sentence); *Corroboration without a judge* (keyword marker → Figure 5; MICI composition, 84% over-praise); *In embedding space* (**Figure 4**, `fig:textspace-body`: (a) variation by patient, (b) same-turn similarity) |
| 07_patient | §6 What the therapist's turns do to the patient | **Table 3** = the process at iteration 10, training oracle (`tab:process`; every iteration: Table 5; held out: Table 6); *How the therapist replies* (reflects change talk; praises / persuades / reflects sustain talk; Figure 6; how much both rewards favour praise → Appendix C.4); *Whether change talk continues* (persistence 0.97 vs 0.80, Base 0.75; why the change talk after a code is not read as that code's effect); *The session as a whole* (change-talk share and trajectory, patient utterance length, the disengagement cue); one held-out paragraph |
| 08_discussion | §7 Discussion and conclusion | a no-numbers opening; the praise paragraph (both rewards favour praise mid-training, only K=0's late: 0.16 vs 0.04 at iteration 9 → Appendix C.4; persistence 97% vs 80%); look-ahead changes which shortcut the reward favours + the compressed mechanism paragraph → Appendix C (the `sec:mechanism` label sits here); why the reward is Q1+Q2 and that another instrument as reward is untested (step 12); *Conclusion* |
| 09_limitations | Limitations (unnumbered, page-exempt) | five paragraphs: one training run per K (the principal limitation); matched iterations are not matched cost (51.2 vs 27.9 GPU-hours; equal-GPU-hour reading → Appendix D.8); K ∈ {0, 5} only (Doron's intermediate-K note kept open); simulation only, in sample, without human validation; instruments and the utterance coder (saturation → Appendix F; one code per utterance; the PCT check; praise vs affirmation is where the judges differ; no human coder) |
| 10_ethics | Ethics Statement (unnumbered, page-exempt) | four paragraphs: no human subjects and no clinical claim; the failure mode we document is safety-relevant; simulated patients encode a narrow population; judges inherit their models' biases (no compute paragraph since 2026-09-29) |
| A_tables | Appendix A Supplementary results | intro; *The look-ahead rollouts* (the rollout check in two sentences: 121,088 logged K=5 candidates, 82% ran all five turns); **Table 4** = every instrument at every iteration (`tab:scores`); **Table 5** = the process at every iteration (`tab:process-all`); **Figure 5** = the keyword marker (`fig:overpraise`); **Figure 6** = the therapist's replies by iteration, both judges (`fig:responsiveness`). Complete tables bold each column's best value |
| A2_heldout | Appendix B The held-out judge | *Where it agrees and where it differs* (21-state sign agreement, 7 of 10 iterations, ratios 2.50 / 1.30); **Table 6** = held-out Table 3 (`tab:process-heldout`); **Figure 7** = held-out levels grid (`fig:grid-heldout`); **Table 7** = held-out Table 4 (`tab:scores-heldout`); **Table 8** = held-out Table 5 (`tab:process-all-heldout`); **Figure 8** = held-out Figure 3 (`fig:process-heldout`) |
| B_mechanism | Appendix C The mechanism analysis in full | C.1 the faithfulness statistic (`app:faithfulness`; **Figure 9**, both judges); C.2 at a matched policy, the effect disappears; C.3 dispersion, and the iteration-10 inversion; C.4 how much the training reward favours praise (`app:premium`; **Figure 10**); C.5 the update-direction proxy (`app:direction`; since step 11a per training iteration: the same at 1–3, apart from 4, K=5 unmeasurable at 9–10, K=0 reverses at 9 and 10) |
| C_repro | Appendix D Reproducibility details | D.1 configuration (**Table 9**, `tab:config`); D.2 instruments (**Table 10**, `tab:instruments`); D.3 the utterance coder (`app:coder`: codebook, role numbering, process quantities, the check against the PCT counts); D.4 the simulated patient and the therapist prompt; D.5 the keyword marker; D.6 anti-degeneracy (+ the malformed-marker leak since step 11a); D.7 evaluation and statistics; D.8 cost accounting (`app:repro-cost`); D.9 artifacts (closes with the licence sentence) |
| D_example | Appendix E Two matched-persona examples in full | E.1 the clear case (Table 2's persona 84, chosen by lexical ranking; utterances 1–7); E.2 the typical case (both-judges median rule, persona 93; utterances 1–9); verbatim, with scores |
| E_saturation | Appendix F Saturation of the training oracle at the winning checkpoint | 21-state sign agreement, 1,484 of 8 × C(21,2) = 1,680 pairs (88.3%); **Table 11** = per-instrument agreement (`tab:agreement`); per-conversation agreement collapses at the winning checkpoint; the mechanism is a ceiling; what this does and does not undermine |

## Scripts

- [`build.py`](build.py) — the only way to build the PDF; see § Build.
- [`render_schematic.py`](render_schematic.py) — draws Figure 1 (the GRPO-group schematic,
  `method_grpo_group.png`); reads no data. It draws at the include width it reads from
  `03_method.tex` (`0.82\textwidth`), every text at ≥ 6 pt, and refuses to save if any text
  overflows its box, overlaps another or misses the width (since 2026-10-01).
- [`render_paper_figures.py`](render_paper_figures.py) — `main()` draws Figures 2–10, each saved by
  `save_at_width` at exactly the width the `.tex` includes it at (since 2026-10-01; before, the
  `bbox_inches="tight"` crop made the PNGs wider and LaTeX shrank their type to ~0.9), no text
  below 5.8 pt: the two level grids (Figures 2 and 7), `process` and
  `process_heldout` (Figures 3 and 8), `textspace_body` (Figure 4), `overpraise` (Figure 5),
  `responsiveness` (Figure 6), `faithfulness` (Figure 9) and `praise_premium` (Figure 10). They read
  `lookahead/shared_base/tables/shared_base.xlsx` (sheets `levels_long`, `k_contrast`,
  `marker_and_length`, `process_levels_<judge>`, `persist_levels_<judge>`, `ct_trajectory_<judge>`,
  `text_diversity`) and `mechanism.xlsx` (`praise_premium_grpo`, `faithfulness_curve_long`). Kept
  but not called: `headline()`, `forest()` (the channel forest, out since 2026-09-29),
  `textspace()`, `tail_audit()` (the rollout-audit figure, out since 2026-09-30) and `saturation()`.
  Run `saturation()` by hand for the Spearman ρ, p and end/start variance-ratio printout behind
  Appendix F (ρ = −0.864 and +0.436; ratio 0.275 → the paper's 0.28); its iteration 0 is each
  run's own Base draw (`replication.xlsx::sd_by_iter`), so it does not reproduce Appendix F's
  shared-Base starting SD of 1.314. ⚠ It also writes `figures/judge_saturation_grpo.png`, a
  figure the paper no longer includes: delete it afterwards, or `overleaf.py pull` refuses (an
  untracked file makes the folder dirty) and `push` uploads it (`figures/*.png` is managed).
  Re-run `main()` after any EDA render pass. (Its docstring still
  describes an older figure set.)
- [`render_process_tables.py`](render_process_tables.py) — prints the two complete process tables
  (Table 5, Appendix A; Table 8, Appendix B) from `shared_base.xlsx`, after checking their
  iteration-10 rows against the iteration-10 process tables (`tab:process`, `tab:process-heldout`)
  and bolding each column's best value; paste its output between the tables' header and
  `\bottomrule`.
- [`select_example_illustrative.py`](select_example_illustrative.py) — at iteration 10, ranks every
  (persona, therapist turn) pair by lexical features of the contrast, earlier replies preferred
  (until 2026-10-05 it ranked only utterances 2–10; widening it left the top 34 and the pick's
  7th place unchanged), and dumps a persona's transcripts; the
  source of Table 2 / Appendix E.1 (the clear case, persona 84).
- [`select_example_persona.py`](select_example_persona.py) — the median-contrast rule behind
  Appendix E.2 (the typical case, persona 93), plus the transcript dump (`--dump out.json`). Both
  selection scripts need the Drive-backed conversation CSVs AND the score lake on disk.
- [`sync_figures.py`](sync_figures.py) — copies nothing since 2026-09-24 (its `FIGURES` list holds
  only comments, `CROP` is empty): every figure is drawn by the two render scripts. Kept as the
  home for a future copied figure; `--check` reports "0 drifted, 0 missing".
- [`make_overleaf_zip.py`](make_overleaf_zip.py) — defines the file list Overleaf gets; built the
  first-upload bundle (see § Overleaf).
- [`overleaf.py`](overleaf.py) — two-way sync with the Overleaf project (see § Overleaf).

## Conventions

Same as the repo standard (see [`../README.md`](../README.md)): every number in
[`NUMBERS.md`](NUMBERS.md) with its exact `Exp3_PTO_GRPO/eda/results/...` source; every figure
drawn from tracked tables by `render_paper_figures.py` (Figure 1 by `render_schematic.py`), none
copied or symlinked. **Signs:** the paper reports K=5 − K=0 everywhere. Some EDA sources store
K=0 − K=5 and are flipped (`k_table1*`, `k_paired*`, `shared_base` `k_process_paired` behind
Tables 3 and 6); others are already K=5 − K=0 and are copied as is (`reward.xlsx::k_endpoints`
behind Table 1; the `delta_K5_minus_K0` / `dz_K5_minus_K0` columns of `shared_base` `k_contrast`).
**Judges:** body numbers are the training oracle's by default (§3.3 says so); every held-out
number names its judge. Instrument score levels are not compared across judges as evidence about
the policies (each judge has its own units): the paper states the offset (the held-out judge
scores 1.2–1.8 points lower on Q1+Q2, §3.3) and, in Appendix F's ceiling argument, sets the two
judges' Q1 levels side by side, but every contrast is read within a judge. Utterance-coder rates
are compared across judges once, as a finding: K=5's praise share, 0.05 vs 0.20. Behaviour claims
name their denominator. Cite the ICLR 2025 paper as the SSI-FM *workshop* poster (canonical BibTeX
in [`../2025_iclr_pto_lookahead/README.md`](../2025_iclr_pto_lookahead/README.md)).

## Overleaf

**The project exists; never upload a zip again.** It was created on 2026-09-16 from
`overleaf.zip` (pdfLaTeX, main document `main.tex`; Overleaf runs BibTeX itself) and has carried
supervisor edits since 2026-09-21. A second zip upload would make a NEW project with a new URL and
strand the comments on the old one. ⚠ The `overleaf.zip` on disk (gitignored) is the stale
2026-09-16 bundle (26 entries, including retired sections and figures): do not upload it.
`make_overleaf_zip.py` would rebuild a current one (30 files), but only a brand-new project would
need it.

**Sync with [`overleaf.py`](overleaf.py)** (Overleaf Git access, a premium/site-licence feature;
the mirror is already initialised):

```powershell
& ..\..\.venv\Scripts\python.exe overleaf.py status   # what differs, both directions; lists incoming Overleaf commits with author
& ..\..\.venv\Scripts\python.exe overleaf.py pull     # Overleaf edits -> this folder
& ..\..\.venv\Scripts\python.exe overleaf.py push     # this folder -> Overleaf, one commit per changed file (--single: one commit)
```

**The rule: `pull` before editing any paper file, `push` after the change is built, committed
and approved by Lior.** It keeps a throwaway clone of the Overleaf repo outside this tree
(`%LOCALAPPDATA%\overleaf-mirrors\`) and copies across the file list `make_overleaf_zip.py`
defines, so the Overleaf project stays exactly the compilable paper and this repo's history stays
linear and Overleaf-free: `NUMBERS.md`, the READMEs, the review-notes file, `CHECKLIST_ARR.md`
and the scripts are never pushed. Both directions **overwrite whole files and never merge**.

- `push` refuses when the Overleaf project has moved since the last sync (pull first); `push
  --force` deliberately discards the Overleaf side. `push` also deletes managed Overleaf files
  that no longer exist locally.
- `pull` refuses only when this folder has **uncommitted** changes. ⚠ It does not see committed
  but **unpushed** local commits: it compares files, not history, so it copies Overleaf's older
  version over them (and brings back files deleted locally). This happened on 2026-09-30; the
  files were restored with `git checkout`. Pull only when local == Overleaf (`status` says "in
  sync" or lists only incoming Overleaf commits), i.e. at the start of a session.
- Overleaf review-panel **comments do not sync**; only edits do.

Authentication is a Git token from Overleaf → Account Settings → Git integration, given as the
*password* at the git prompt. Overleaf compiles with `latexmk`, which iterates to convergence on
its own, so the line-number problem `build.py` exists to prevent is local-only.

**`[review]` vs `[final]`.** The draft is in `[review]` mode (line numbers, anonymous byline),
the right mode for supervisor comments. `\usepackage[final]{acl}` in `main.tex` shows the author
block (already filled in, `main.tex`) and drops the line and page numbers. It does NOT hide
Doron's notes: that is `\dnotesfalse` (`main.tex`, the `\newif\ifdnotes` line), a separate step
before submission.

## Build (MiKTeX on Windows — see ../README.md)

```powershell
& ..\..\.venv\Scripts\python.exe build.py            # build until converged, then check
& ..\..\.venv\Scripts\python.exe build.py --check    # audit the existing main.pdf only
```

[`build.py`](build.py) runs `pdflatex`, `bibtex`, then `pdflatex` **until `main.aux`/`main.out`
stop changing**, and then scans every page of the PDF for line numbers printed inside a text
column and for unresolved references (exit status non-zero on any of these, on LaTeX errors, on
non-convergence, or when PyMuPDF is missing). ⚠ Until 2026-10-01 the scan matched only
three-digit line numbers, so every page from line 1000 on (the late references and all the
appendices) went unchecked; a re-scan of the 2026-09-30 PDF found none misplaced. ⚠ **Do not
hand-run the usual `pdflatex · bibtex · pdflatex · pdflatex`.** `acl.sty`'s `[review]` mode loads
`lineno` with `switch`, which is *pagewise* mode: each line's column is read back from the
previous pass's `.aux`, so the line numbers are placed correctly only once two consecutive passes
have the same layout, and after `bibtex` moves the back matter the four-step chain is one pass
short. On 2026-09-16 that put 98 line numbers on top of the text across seven pages with a clean
log (cold build: 477 → 6 → 114 → 98 → 0 misplaced by pass). Nothing in the log reports it; only
the scan does. Every note of Doron's that contains a literal `??` is commented out now, so a
`??` the scan reports is a real unresolved reference. (A visible note with `??` would be counted
too; then check `main.log` for "undefined".)

To eyeball the layout, the repo `.venv` has PyMuPDF: `fitz.open("main.pdf")[p].get_pixmap(dpi=100).save(...)`.

## Before submission (open items)

The ordered work is plan steps 12, 11, 13 above (the contributions and abstract, pushed for
Doron's notes; Lior's read-through; then notes hidden + length pass + checklist + code zip).
Beyond those:

- **Supervisors.** Doron has read the draft in five Overleaf passes (21–27 Sep), all answered
  through plan step 12; one of his notes is open on purpose (the intermediate-K reminder in the
  Limitations). Step 12 (contributions, abstract, Q1+Q2) went to Overleaf on 2026-10-05 for his
  read, its handled markers visible. No read by the other
  co-authors is recorded here.
- **Responsible NLP checklist:** answer the generative-AI question truthfully: this draft was
  written and its code built with substantial AI assistance (ACL policy allows it with disclosure;
  a misleading checklist is desk-reject grounds). Camera-ready: the same disclosure in the
  Acknowledgements (the comment near the end of `main.tex` marks the spot).
- **`CHECKLIST_ARR.md` pointers are stale** (section, appendix and table numbers from before the
  2026-09-24 restructure; the Ethics compute paragraph it cites is gone; 22 model states for 21).
  Re-point them after the length pass.
- Camera-ready only: switch `acl` to `[final]` (the author block is already filled in) and add the
  Acknowledgements disclosure.
- Optional, if a co-author wants it: a human MI coder on a sample of the endpoint conversations
  would close the Limitations' "no human coder has checked the codes".
- The full pre-submission list, with the checklist answers, is in [`CHECKLIST_ARR.md`](CHECKLIST_ARR.md).
