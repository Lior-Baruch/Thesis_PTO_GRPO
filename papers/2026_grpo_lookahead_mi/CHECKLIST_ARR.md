# ARR submission: Responsible NLP checklist answers + pre-submission list

Drafted 2026-09-17 against the checklist as published at
https://aclrollingreview.org/responsibleNLPresearch/ and the CFP at https://aclrollingreview.org/cfp
(both fetched that day). The checklist is a **form in the submission system**, not part of the PDF;
"incorrect, incomplete or misleading information in the checklist can result in desk rejection".
Answers below are drafts to paste, each with the section of the paper that backs it. Every pointer
was re-pointed against the final build on 2026-10-11 (below); re-check only if the PDF is rebuilt
with structural changes. This file is local only: `overleaf.py` never pushes it.

**Re-pointed 2026-10-12 after Kfir's notes** (headings title-cased; two new body tables: Table 1 = the worked look-ahead example, Table 2 =
endpoint (was 1), Table 3 = the persona-87 exchange, and every appendix table +2; Appendix E is
Example Conversations with E.1-E.3 and no selection procedure (Lior, same day); Eq. 2 moved to D.1;
section numbers unchanged; 38 pages; table numbers below read off `main.aux`).

**Numbering of the final build** (2026-10-11, README row 18: read off `main.aux`, confirmed in
`main.pdf`, then 37 pages): body §1 Introduction, §2 Related Work, §3 Method (3.1 Task, Simulator
and Oracle; 3.2 GRPO with Look-Ahead; 3.3 Evaluation Design), §4 Results: Instrument Scores, §5 What
Look-Ahead Teaches the Therapist, §6 What the Patient Says Next, §7 Discussion and Conclusion; then
the unnumbered Limitations (five paragraphs: "One training run per K", "Matched iterations are not
matched cost", "K ∈ {0, 5} only", "Simulation only, without human validation", "Instruments and the
utterance coder") and the unnumbered Ethics Statement (four paragraphs: "No human subjects, and no
clinical claim", "The failure mode we document is safety-relevant", "Simulated patients encode a
narrow population", "Judges inherit their models' biases"). Appendices: A Supplementary results
(A.1 The look-ahead rollouts, A.2 Against the Base, A.3 Where the lead comes from, A.4 A test over
the whole run, A.5 A second draw of the final policy, A.6 The two Base draws, A.7 Change talk after
each reply, A.8 The patient's side, A.9 Patients with an unseen problem), B The held-out judge,
C The mechanism analysis in full (C.1–C.7; C.6 The update-direction proxy, C.7 How much the
therapist's turns depend on the patient), D Reproducibility details (D.1 The training loop,
D.2 Configuration, D.3 Instruments, D.4 The utterance coder, D.5 The simulated patient and the
therapist prompt, D.6 The keyword marker, D.7 Anti-degeneracy, D.8 Evaluation and statistics,
D.9 Cost accounting behind the disclosures, D.10 Artifacts), E Example Conversations (E.1 the example of Table 3, E.2 a second example, E.3 a look-ahead group from training),
F Saturation of the training oracle at the winning checkpoint. **Table 13 = trainer logs, Table 14
= configuration (in D.2), Table 15 = instruments (in D.3); Algorithm 1 and Eq. 2 are in D.1.**

## A. For every submission

| # | Question | Answer | Where |
|---|---|---|---|
| A1 | Limitations? | **Yes.** | Unnumbered "Limitations" section after §7 Discussion and conclusion (five paragraphs since 2026-09-29): "One training run per K"; "Matched iterations are not matched cost" (plain words; the iso-compute numbers in Appendix D.9, Cost accounting behind the disclosures); "K ∈ {0, 5} only"; "Simulation only, without human validation" (evaluated on new conversations but with the same 96 personas that supply the training prompts; a 48-persona new-problem check in Appendix A.9, Patients with an unseen problem; same-model patient); "Instruments and the utterance coder" (incl. the judges' low per-utterance agreement, κ 0.08–0.26, since 2026-10-06, and no human coder). (The 200-token cap was cut from the Limitations 2026-09-29; §3.1 and Table 14 state it, the opening of Appendix E explains the mid-sentence endings.) |
| A2 | Potential risks? | **Yes.** | Unnumbered "Ethics Statement": "No human subjects, and no clinical claim" (no clinical claim, none of the 96 personas discloses a crisis, so crisis handling is untested, released adapters get a research-only model card); "The failure mode we document is safety-relevant" (the K=0 policy's over-praise, and the K=5 policy's persuasion when the patient argues against change); "Simulated patients encode a narrow population"; "Judges inherit their models' biases". (The compute paragraph was cut; compute is in the Limitations, "Matched iterations are not matched cost", and Appendix D.9.) |

## B. Scientific artifacts used or created

| # | Question | Answer | Where |
|---|---|---|---|
| B1 | Cited the creators of artifacts used? | **Yes.** | GRPO (Shao et al., 2024) §1–3; PTO origin (Baruch et al., 2025) §1–3; the persona grid credited as adapted from Yosef et al. (2024), as used by Baruch et al., with the two uncooperative clauses rewritten: §3.1 (Task, simulator and oracle), the Ethics Statement ("Simulated patients encode a narrow population") and Appendix D.5 (The simulated patient and the therapist prompt); instruments Q1/Q2 (Yosef et al., 2024), WAI-SR (Hatcher & Gillaspy, 2006), CSQ-8 (Larsen et al., 1979), MITI 4.2.1 (Moyers et al., 2015, 2016) + MISC 2.5 (Houck et al., 2010) for MITI, PCT, MICI and the utterance coder, in §3.3 and Table 15 (Appendix D.3, Instruments); the five encoders of Appendix C.6 (The update-direction proxy; all-MiniLM-L6-v2 cited to Reimers & Gurevych, 2019, in C.7); Llama 3 (Grattafiori et al., 2024), LoRA (Hu et al., 2022), TRL (von Werra et al., 2020), Transformers (Wolf et al., 2020), PEFT (Mangrulkar et al., 2022) in Table 14 (Appendix D.2, Configuration; added 2026-10-06), Llama 3.2 and LoRA also cited in §3.1 (2026-10-12). The two API models are identified by snapshot ID (`gpt-4o-mini-2024-07-18`, `claude-haiku-4-5-20251001`), not cited. |
| B2 | License / terms discussed? | **Yes** — Appendix D.10 (Artifacts) names the therapist model's license (Llama 3.2 Community License; added 2026-10-11) and says the patient simulator and judges are commercial APIs used within their terms of service (moved out of the Ethics Statement 2026-09-29). Name the license in the checklist box too; the created artifacts (personas, coder prompts, code) will be released upon publication (D.10, decided 2026-10-11: no archive with the submission); state their license then. |
| B3 | Use consistent with intended use? | **Yes.** | Ethics Statement, "No human subjects, and no clinical claim": research artifact only, no clinical use, and any released adapter carries a model card saying it is a research artifact not for use with people; Appendix D.10 (Artifacts): the base model is openly licensed and the APIs were used within their terms of service; released adapters are research-only (Ethics Statement, "No human subjects, and no clinical claim"). |
| B4 | Checks for PII / offensive content? | **N/A for data** (no human data: every conversation is between two language models, Ethics Statement, "No human subjects, and no clinical claim"). The persona prompts are synthetic (Appendix D.5, The simulated patient and the therapist prompt; the 48 alcohol personas in Appendix A.9, Patients with an unseen problem) and none of the 96 discloses a crisis (same Ethics paragraph). |
| B5 | Documentation of artifacts? | **Yes.** | Appendix D.3 (Instruments: the instruments and their sources, Table 15), D.4 (The utterance coder), D.5 (The simulated patient and the therapist prompt: persona and therapist prompts, verbatim), D.6 (The keyword marker); the 48 alcohol personas in Appendix A.9 (Patients with an unseen problem). Language: English only (§3.1: "an English text conversation") — say so explicitly in the checklist. |
| B6 | Relevant statistics? | **Yes.** | 96 personas per model state; 21 model states (the pooled Base of 192 conversations, two draws, + 10 iterations × 2 runs = 1 + 2 × 10 = 21), so 2 × 11 × 96 = 2,112 evaluation conversations (Appendix D.4); 8 instruments (PCT computed from the utterance coder's patient codes since 2026-10-06; undefined for 4 held-out conversations, Appendix D.3) + the utterance coder; two judges; the trainer's 0.05 validation split (Table 14; Algorithm 1 in Appendix D.1); conversation lengths (§3.1: session cap 50 utterances, the Base averages 28.5; §5: therapist turn length in characters; Appendix A.8, The patient's side: patient utterance length; Table 14: session cap); plus a second draw of 96 conversations of the final K=5 policy (Appendix A.5) and 4 × 48 = 192 conversations on the alcohol personas (Appendix A.9). |

## C. Computational experiments

| # | Question | Answer | Where |
|---|---|---|---|
| C1 | Parameters, compute budget, infrastructure? | **Yes.** | 1B-parameter policy with LoRA r=16 on all attention and MLP projections (Table 14, Appendix D.2); one A100 (Table 14); 27.9 (K=0) + 51.2 (K=5) = 79.1 GPU-hours (Limitations, "Matched iterations are not matched cost"; Appendix D.9, Cost accounting behind the disclosures — not in the table; the Ethics compute paragraph was cut 2026-09-29); API call counts (Appendix D.9; moved out of the Limitations 2026-09-29). |
| C2 | Experimental setup and hyperparameter search? | **Yes, with a caveat.** | Table 14 (Appendix D.2, Configuration) lists the resolved configuration (since 2026-10-06 incl. optimizer, schedule, warm-up, clipping, LoRA targets, KL form, advantage formula, clip ε, sampling and judge decoding); both arms share one configuration and differ in two tracked fields (§3.3 "The two runs", Appendix D.2). **No hyperparameter search was run** — the GRPO settings were fixed a priori and matched across arms, and the paper says so in Appendix D.2 ("No hyperparameter search was run: the configuration, with MCL set beforehand in a pilot (§3.2), is shared by both runs"); MCL = 12 was set beforehand in a pilot on an earlier configuration (§3.2, "Minimum conversation length"). Say so in the checklist ("single configuration, no search; MCL from a pilot"). |
| C3 | Descriptive statistics (error bars, single run vs mean)? | **Yes.** | Mean ± SE over the 96 personas or over conversations (Figures 2, 3, 5, 6, 7, 8); ± 1.96 SE over groups in Figure 10; conversation-level bootstrap bands in Figure 9 (cluster bootstrap) and Figure 12a; persona-paired Wilcoxon, Cohen's d_z, fixed-seed percentile bootstrap intervals (2,000 resamples for Appendix C's dispersion ratios and Appendix A.2's persona-level interval; 1,000 for the faithfulness intervals and Figure 9; 500 for the category intervals of Appendix C.6; 300 for Figure 12a's band), Holm correction within each family (§3.3, "Evaluation and statistics"; Appendix D.8, Evaluation and statistics); **single training run per arm**, stated in the Limitations ("One training run per K"); the evaluation re-draw of the final policy (§4, paragraph "Onset, and the K=0 decline"; Appendix A.5, A second draw of the final policy). |
| C4 | Packages, versions, settings? | **Yes.** | TRL 1.4.0, transformers 5.8.1, PEFT 0.19.1 (Table 14); TRL `loss_type="grpo"` with clip ε 0.2 and one inner update per batch, and the group-standardized advantage (r − mean)/(std + 10⁻⁴) — i.e. `scale_rewards="group"` — in Table 14, §3.2 (paragraph "The update"; Eq. 2 in Appendix D.1) and Appendix D.1 (The training loop, Algorithm 1). |

## D. Human annotators / human participants

**No** (D1–D5 N/A). No human subjects, no annotators, no human-coded sample (Ethics Statement, "No
human subjects, and no clinical claim"; Limitations, "Simulation only, without human validation" and
"Instruments and the utterance coder", which says no human coder has checked the codes).

## E. AI assistants

| # | Question | Answer |
|---|---|---|
| E1 | AI-assistant use disclosed? | **Yes.** Draft wording for the checklist box: *"Generative AI assistants (Anthropic Claude, via Claude Code) were used for drafting and revising the manuscript text, for writing and reviewing the experiment and analysis code, and for auditing the reported numbers against the analysis tables. The research questions, experimental design, decisions on what to report and all claims are the authors'. The disclosure will also appear in the Acknowledgements of the camera-ready version."* (The CFP: use "for writing or coding, as well as its scope, must be disclosed in the Responsible NLP Checklist. Details should be included in the Acknowledgements section.") |

## Pre-submission list (things only the authors can do)

1. **Supervisors' read** of the current draft (they signed off on the retired 2×2 on 2026-08-27; this
   draft has not been through them). Cover-note questions: see `REVIEW_2026-09-16.md` § E.
2. **Anonymised code archive.** The CFP wants software as "a single .tgz or .zip archive" uploaded
   with the submission, or an anonymised repository (Anonymous GitHub); "links to cloud services
   like Google Drive, Dropbox etc. are not acceptable". **DECIDED 2026-10-11 (Lior): no archive with the submission.** Appendix D.10 (Artifacts), C.6 and
   E.1/E.2 now say the code and scripts "will be released upon publication" (PDF grep: no
   "supplementary archive" left). For the release later — decide what goes in the archive (the two trainer
   notebooks + `_shared/`, the EDA package, the results tables, `NUMBERS.md`, the paper's scripts)
   and strip names/paths/keys from it.
   **Revised by review round 5 (2026-10-08, M1 + A3): the archive is GRPO-only.** Shipping the PTO
   trainer, or tables with PTO rows, would amount to an unreported PTO-vs-GRPO comparison that ties
   the paper to Baruch et al., and the Hub handle identifies the authors.
   - **Contents:** `code/GRPO_Exp3/`, `code/_shared/` (PTO/DPO code paths and docstrings cut,
     e.g. `tb_plots.py` and `eda_recorder.py`), `code/system_prompts_builder.py`,
     `code/questionnaires.py`, and the paper's scripts: `render_process_tables.py`,
     `render_paper_figures.py`, `render_schematic.py`, `select_example_*.py` (not `build.py`,
     `overleaf.py`, `make_overleaf_zip.py`, `sync_figures.py`), with their
     `HERE.parent.parent / "Exp3_PTO_GRPO"` roots and `_shared/runtime.py`'s default
     `experiment_name` replaced by a neutral relative root. **Not** `code/PTO_Exp3/`. The cited
     EDA tables filtered by script to rows whose arm is Base, GRPO_LA0 or GRPO_LA5 (model Base or
     `GRPOExp3_*`), dropping every `PTO_LA*`/`PTOExp3_*` row and every `PTO_*` column (most tables
     carry PTO rows or columns, e.g. `compute/cost/tables/step_multiplier.md`); `NUMBERS.md` only
     after its lookups are re-checked against the filtered copy.
   - **In the archived copies:** `HUB_ENTITY = "<HF_USER>"` (it is `"LBK95"` in
     `train_GRPO_Iterative.ipynb`, the handle that also hosts the cited PTO paper's adapters); a
     relative root instead of `MyDrive/Thesis_PTO_GRPO` paths; **all saved notebook outputs
     cleared**; no `run_metadata.json`.
   - **Archived notebook cell 1:** `NUM_ITERATIONS = 10` (the tracked notebook says 6) with
     `# Both reported runs used 10 iterations` and `# LOOKAHEAD_K: 0 for the K=0 run, 5 for the K=5
     run`; replace the comments that argue for 6 and quote the superseded 2.4–3.0× cost with the
     measured median 1.92× per step (`compute/cost/tables/step_multiplier.md`,
     `GRPO_step_ratio_K5_over_K0`, iterations 3–10: (1.911 + 1.930) / 2 = 1.92).
   - **`questionnaires.py`:** replace `CSQ8_ITEMS` (fee-licensed instrument) with placeholders, and
     `MI_SAT_ITEMS` (unpublished) too unless its author agrees. Check whether the lab holds a CSQ-8
     permission before wording anything about it.
   - **Strip-grep, case-insensitive, over the built zip as well as the PDF:**
     `lior|doron|kfir|butman|moshe|almog|baruc|reichman|runi\.ac|C:/Users|My Drive|MyDrive|Thesis_PTO|LBK95|HUB_ENTITY|current_adapter_repo|PTO_|PTOExp3`,
     plus `\bPTO\b|\bDPO` over the zip only (the PDF cites PTO)
     (round 4's B9 list, extended in round 5; `butman|moshe|runi\.ac` added for the fourth author
     and the e-mail domain; `PTO_` also catches `Exp3_PTO_GRPO`, `PTO_LA*` and `PTO_*` columns).
     Inspect every hit rather than expecting zero: `HUB_ENTITY`/`current_adapter_repo` must carry
     placeholders; in the PDF expect only the Baruch et al. in-text citations and the reference-list author
     names of Baruch et al. (2025), Yosef et al. (2024) and the Lior Shani reference (checked 2026-10-11: clean).
   - If the adapter repos `GRPO_Iterative_Q1Q2_Llama32-1B_LA{0,5}_MCL12_G8_full` are public,
     consider making them private until the decision. Fallback if the archive is not ready:
     change Appendix D.10 (Artifacts), which now says the archive "accompanies the paper", and the three
     "is in the supplementary archive" sentences (Appendix C.6 'What the direction points to', E.1, E.2) to
     "will be released upon publication", upload no archive, and grep the PDF for "supplementary archive".
3. **Preprint option.** Decide whether to select the binding "no non-anonymous preprint" option
   in the form; there is otherwise no anonymity period.
4. **Fill the checklist** in the form from the tables above; state "English only" (B5), the
   Llama 3.2 Community License (B2) and "no hyperparameter search; MCL from a pilot" (C2) explicitly.
5. **Author block and `[final]`** are camera-ready only; keep `[review]` for the submission.
6. **Optional but strong:** a human MI coder on a sample of endpoint conversations (the paper's
   most-repeated caveat); a second training seed per arm is the principal limitation but is out of
   budget before 2026-10-12 (≈79 GPU-h plus the oracle bill).
7. **Designated service contributor — MUST, form due by Oct 14** (added by review round 5, A1;
   rules checked 2026-10-08 against https://aclrollingreview.org/cfp and the ACL sustainable
   reviewing policy 2026). A submission is guaranteed review only if it names one qualified
   designated service contributor, who must complete a registration form within 48 hours of the
   submission deadline (Oct 12 → by Oct 14). Without one the paper goes into a lottery, and a paper
   not drawn is desk-rejected. A contributor serves at most two submissions per cycle (five reviews,
   or eight meta-reviews, each) and must confirm the paper is ready for a top-tier venue.
   Qualification is by publication record: a PhD holder or senior academic (postdoc/faculty) with at
   least 2 publications in major ACL events, Findings of the ACL or major ML venues; a doctoral
   student, or an industry researcher with at least a Master's, needs those 2 plus at least 3
   publications in total. A non-author can be nominated but must vouch for the work. Action: name a
   co-author who qualifies (Kfir Bar or Doron Friedman — confirm their record against the criteria)
   and is not already the contributor for two other October submissions; make sure they file the
   form from their console by Oct 14.
8. **Complete OpenReview profiles for all four authors — MUST, by Oct 19** (one week after the
   deadline; review round 5, A1): ORCID, affiliation history, e-mails, conflicts of interest and,
   for those with publications, DBLP / ACL Anthology links. The designated contributor's profile
   must show the record that qualifies them. Re-read https://aclrollingreview.org/cfp on submission
   day in case the rules changed.
