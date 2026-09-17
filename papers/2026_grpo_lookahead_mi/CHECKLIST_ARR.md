# ARR submission: Responsible NLP checklist answers + pre-submission list

Drafted 2026-09-17 against the checklist as published at
https://aclrollingreview.org/responsibleNLPresearch/ and the CFP at https://aclrollingreview.org/cfp
(both fetched that day). The checklist is a **form in the submission system**, not part of the PDF;
"incorrect, incomplete or misleading information in the checklist can result in desk rejection".
Answers below are drafts to paste, each with the section of the paper that backs it. Re-check
section numbers against the final PDF before pasting. This file is local only: `overleaf.py` never
pushes it.

## A. For every submission

| # | Question | Answer | Where |
|---|---|---|---|
| A1 | Limitations? | **Yes.** | Unnumbered "Limitations" section after §8: one training run per arm; evaluation draws; matched iterations vs matched cost (incl. the iso-compute reading); K ∈ {0, 5} only; the continuation pressure; simulation only / in-sample / same-model patient / no human validation / the 200-token cap; instruments. |
| A2 | Potential risks? | **Yes.** | "Ethics Statement": no clinical claim; the over-praise failure mode is safety-relevant; the simulated population is narrow; judges inherit their models' biases; compute. |

## B. Scientific artifacts used or created

| # | Question | Answer | Where |
|---|---|---|---|
| B1 | Cited the creators of artifacts used? | **Yes.** | Llama-3.2-1B, gpt-4o-mini, Claude Haiku 4.5 named in §4 and Table 5; GRPO (Shao et al., 2024) §2–3; DPO/PTO origin §1–2; instruments Q1/Q2 (Yosef et al., 2024), WAI-SR (Hatcher & Gillaspy, 2006), CSQ-8 (Larsen et al., 1979), MITI 4.2.1 (Moyers et al., 2015) in §4 and Table 6; TRL / transformers / PEFT versions in Table 5. |
| B2 | License / terms discussed? | **Partly** — Ethics Statement says the base model is openly licensed and the APIs were used within their terms of service. Add the exact license name (Llama 3.2 Community License) to the Ethics paragraph if a reviewer asks; the created artifacts (personas, coder prompts, code) will be released under a permissive license — state which one when the archive is prepared. |
| B3 | Use consistent with intended use? | **Yes.** | Ethics Statement: research artifact only, no clinical use; the base model is used within its license; the created artifacts are for research. |
| B4 | Checks for PII / offensive content? | **N/A for data** (no human data: every conversation is between two language models, Ethics Statement ¶1). The persona prompts are synthetic (Appendix C.3). |
| B5 | Documentation of artifacts? | **Yes.** | Appendix C.2 (instruments and prompts), C.3 (persona and therapist prompts, verbatim), C.4 (the lexical marker), Table 6. Language: English only — say so explicitly in the checklist. |
| B6 | Relevant statistics? | **Yes.** | 96 personas per model state; 2 arms × 11 states = 22 states; 8 instruments; two graders; the trainer's 0.05 eval split (Table 5); conversation lengths (§4, §6, Appendix D). |

## C. Computational experiments

| # | Question | Answer | Where |
|---|---|---|---|
| C1 | Parameters, compute budget, infrastructure? | **Yes.** | 1B-parameter policy with LoRA r=16 (Table 5); ≈79 GPU-hours total, 27.9 (K=0) + 51.2 (K=5), one A100 (Ethics; Limitations; Table 5; Appendix C.7); API call counts (Limitations; C.7). |
| C2 | Experimental setup and hyperparameter search? | **Yes, with a caveat.** | Every hyperparameter in Table 5; both arms share one configuration and differ in two tracked fields (§4, C.1). **No hyperparameter search was run** — the GRPO settings were fixed a priori and matched across arms; say so in the checklist ("single configuration, no search"). |
| C3 | Descriptive statistics (error bars, single run vs mean)? | **Yes.** | Mean ± SE over 96 personas (Figures 2, 4, 5); persona-paired Wilcoxon, Cohen's d_z, 2,000-resample bootstrap CIs, Holm correction (§4, C.6); **single training run per arm**, stated in the Limitations and the abstract; the evaluation re-draw (§5). |
| C4 | Packages, versions, settings? | **Yes.** | TRL 1.4.0, transformers 5.8.1, PEFT 0.19.1 (Table 5); TRL `loss_type="grpo"`, `scale_rewards="group"`, one inner update per batch (Table 5, §3). |

## D. Human annotators / human participants

**No** (D1–D5 N/A). No human subjects, no annotators, no human-coded sample (Ethics Statement ¶1;
Limitations "Simulation only, in sample, without human validation").

## E. AI assistants

| # | Question | Answer |
|---|---|---|
| E1 | AI-assistant use disclosed? | **Yes.** Draft wording for the checklist box: *"Generative AI assistants (Anthropic Claude, via Claude Code) were used for drafting and revising the manuscript text, for writing and reviewing the experiment and analysis code, and for auditing the reported numbers against the analysis tables. The research questions, experimental design, decisions on what to report and all claims are the authors'. The disclosure will also appear in the Acknowledgements of the camera-ready version."* (The CFP: use "for writing or coding, as well as its scope, must be disclosed in the Responsible NLP Checklist. Details should be included in the Acknowledgements section.") |

## Pre-submission list (things only the authors can do)

1. **Supervisors' read** of the current draft (they signed off on the retired 2×2 on 2026-08-27; this
   draft has not been through them). Cover-note questions: see `REVIEW_2026-09-16.md` § E.
2. **Anonymised code archive.** The CFP wants software as "a single .tgz or .zip archive" uploaded
   with the submission, or an anonymised repository (Anonymous GitHub); "links to cloud services
   like Google Drive, Dropbox etc. are not acceptable". Appendix C.8 promises the analysis tables,
   the claims ledger and the selection scripts — decide what goes in the archive (the two trainer
   notebooks + `_shared/`, the EDA package, the results tables, `NUMBERS.md`, the paper's scripts)
   and strip names/paths/keys from it.
3. **Preprint option.** Decide whether to select the binding "no non-anonymous preprint" option
   in the form; there is otherwise no anonymity period.
4. **Fill the checklist** in the form from the tables above; state "English only" (B5) and "no
   hyperparameter search" (C2) explicitly.
5. **Author block and `[final]`** are camera-ready only; keep `[review]` for the submission.
6. **Optional but strong:** a human MI coder on a sample of endpoint conversations (the paper's
   most-repeated caveat); a second training seed per arm is the principal limitation but is out of
   budget before 2026-10-12 (≈79 GPU-h plus the oracle bill).
