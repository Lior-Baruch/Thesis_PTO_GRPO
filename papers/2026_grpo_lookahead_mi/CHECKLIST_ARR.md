# ARR submission: Responsible NLP checklist answers + pre-submission list

Drafted 2026-09-17 against the checklist as published at
https://aclrollingreview.org/responsibleNLPresearch/ and the CFP at https://aclrollingreview.org/cfp
(both fetched that day). The checklist is a **form in the submission system**, not part of the PDF;
"incorrect, incomplete or misleading information in the checklist can result in desk rejection".
Answers below are drafts to paste, each with the section of the paper that backs it. Re-check
section numbers against the final PDF before pasting. This file is local only: `overleaf.py` never
pushes it.

**Numbering as of 2026-10-06** (re-pointed against `main.aux` after step 11's reviewer batch B):
body §1 Introduction, §2 Related work, §3 Method (3.1 task, simulator and oracle; 3.2 GRPO with
look-ahead; 3.3 evaluation design), §4 Results (reward and instruments), §5 the therapist, §6 the
patient, §7 Discussion. Appendices: A supplementary results, B held-out judge, C mechanism (C.1–C.6),
D Reproducibility (D.1 configuration, D.2 instruments, D.3 utterance coder, D.4 patient and therapist
prompts, D.5 keyword marker, D.6 anti-degeneracy, D.7 evaluation and statistics, D.8 cost accounting,
D.9 artifacts), E examples, F saturation. **Table 12 = configuration, Table 13 = instruments.**

## A. For every submission

| # | Question | Answer | Where |
|---|---|---|---|
| A1 | Limitations? | **Yes.** | Unnumbered "Limitations" section after the Discussion (five paragraphs since 2026-09-29): one training run per K; matched iterations vs matched cost (plain words; the iso-compute numbers in D.8); K ∈ {0, 5} only; simulation only / in-sample / same-model patient / no human validation; instruments and the utterance coder (incl. the judges' low per-utterance agreement, κ 0.08–0.26, since 2026-10-06). (The 200-token cap was cut from the Limitations 2026-09-29; §3.1 and Table 12 state it, Appendix E explains the mid-sentence endings.) |
| A2 | Potential risks? | **Yes.** | "Ethics Statement": no clinical claim, no persona discloses a crisis (so crisis handling is untested), released adapters get a research-only model card; the over-praise failure mode is safety-relevant; the simulated population is narrow; judges inherit their models' biases. (The compute paragraph was cut; compute is in D.8.) |

## B. Scientific artifacts used or created

| # | Question | Answer | Where |
|---|---|---|---|
| B1 | Cited the creators of artifacts used? | **Yes.** | GRPO (Shao et al., 2024) §2–3; PTO origin §1–2; instruments Q1/Q2 (Yosef et al., 2024), WAI-SR (Hatcher & Gillaspy, 2006), CSQ-8 (Larsen et al., 1979), MITI 4.2.1 (Moyers et al., 2016) + MISC 2.5 (Houck et al., 2010) for MITI, PCT, MICI and the utterance coder, in §3.3 and Table 13; the five encoders of Appendix C.6; Llama 3 (Grattafiori et al., 2024), LoRA (Hu et al., 2022), TRL (von Werra et al., 2020), Transformers (Wolf et al., 2020), PEFT (Mangrulkar et al., 2022) in Table 12 (added 2026-10-06, appendix only so the body does not grow). The two API models are identified by snapshot ID (`gpt-4o-mini-2024-07-18`, `claude-haiku-4-5-20251001`), not cited. |
| B2 | License / terms discussed? | **Partly** — Appendix D "Artifacts" says the base model is openly licensed and the APIs were used within their terms of service (moved out of the Ethics Statement 2026-09-29). Add the exact license name (Llama 3.2 Community License) there if a reviewer asks; the created artifacts (personas, coder prompts, code) will be released under a permissive license — state which one when the archive is prepared. |
| B3 | Use consistent with intended use? | **Yes.** | Ethics Statement: research artifact only, no clinical use; the base model is used within its license; the created artifacts are for research. |
| B4 | Checks for PII / offensive content? | **N/A for data** (no human data: every conversation is between two language models, Ethics Statement ¶1). The persona prompts are synthetic (Appendix D.4) and none discloses a crisis (Ethics Statement ¶1). |
| B5 | Documentation of artifacts? | **Yes.** | Appendix D.2 (instruments and their sources, Table 13), D.3 (the utterance coder), D.4 (persona and therapist prompts, verbatim), D.5 (the keyword marker). Language: English only — say so explicitly in the checklist. |
| B6 | Relevant statistics? | **Yes.** | 96 personas per model state; 21 model states (the pooled Base of 192 conversations, two draws, + 10 iterations × 2 runs = 1 + 2 × 10 = 21); 8 instruments (PCT computed from the utterance coder's patient codes since 2026-10-06; undefined for 4 held-out conversations) + the utterance coder; two judges; the trainer's 0.05 validation split (§3.2, Table 12); conversation lengths (§4, §6, Appendix D). |

## C. Computational experiments

| # | Question | Answer | Where |
|---|---|---|---|
| C1 | Parameters, compute budget, infrastructure? | **Yes.** | 1B-parameter policy with LoRA r=16 on all attention and MLP projections (Table 12); one A100 (Table 12); 27.9 (K=0) + 51.2 (K=5) = 79.1 GPU-hours (Limitations; Appendix D.8 — not in the table; the Ethics compute paragraph was cut 2026-09-29); API call counts (D.8; moved out of the Limitations 2026-09-29). |
| C2 | Experimental setup and hyperparameter search? | **Yes, with a caveat.** | Table 12 lists the resolved configuration (since 2026-10-06 incl. optimizer, schedule, warm-up, clipping, LoRA targets, KL form, advantage formula, clip ε, sampling and judge decoding); both arms share one configuration and differ in two tracked fields (§3.3 "The two runs", D.1). **No hyperparameter search was run** — the GRPO settings were fixed a priori and matched across arms; say so in the checklist ("single configuration, no search"). |
| C3 | Descriptive statistics (error bars, single run vs mean)? | **Yes.** | Mean ± SE over the 96 personas or over conversations (Figures 2, 3c, 5, 6, 7, 8c); a conversation-level bootstrap band in Figure 4a; persona-paired Wilcoxon, Cohen's d_z, percentile bootstrap intervals (2,000 resamples for the dispersion ratios of Appendix C, 1,000 elsewhere), Holm correction within each family (§3.3, D.7); **single training run per arm**, stated in the Limitations and the abstract; the evaluation re-draw of the final policy (§4, "A second draw of the final policy"). |
| C4 | Packages, versions, settings? | **Yes.** | TRL 1.4.0, transformers 5.8.1, PEFT 0.19.1 (Table 12); TRL `loss_type="grpo"` with clip ε 0.2 and one inner update per batch, and the group-standardized advantage (r − mean)/(std + 10⁻⁴) — i.e. `scale_rewards="group"` — in Table 12 and §3.2 (Algorithm 1). |

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
   like Google Drive, Dropbox etc. are not acceptable". Appendix D.9 promises the analysis tables,
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
