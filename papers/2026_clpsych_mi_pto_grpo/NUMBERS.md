# Claims ledger — every number in the draft → the artifact it came from

All paths are relative to `Exp3_PTO_GRPO/eda/results/L0/` unless marked otherwise.
View = `L0` (K=0 only), grader = `gpt-4o-mini` (primary) unless the row says otherwise.
Regenerate the sources with `eda/tools/render_views.py`; re-copy the figures with
`sync_figures.py`. **If a number here changes, grep the draft for it before trusting the prose.**

## §4 Setup

| Claim | Source |
|---|---|
| Llama-3.2-1B bf16, LoRA r=16 α=16 dropout 0.05, all attn+MLP projections | `code/PTO_Exp3/train_PTO_Iterative.ipynb` cell 1; `data/*/runs/full/*/run_metadata.json` |
| patient + oracle = `gpt-4o-mini-2024-07-18` | both `run_metadata.json` |
| 96 personas = 2 gender × 3 cooperation × 2 problem × 2 duration × 2 attempts × 2 age (27/61) | `code/system_prompts_builder.py::generate_all_permutations` |
| 10 iters × 2 epochs, 96 convs/iter, MCL=12, M=G=8, temp 1.2, lr 1e-5 | `run_metadata.json` (both arms) |
| PTO τ=0.1, DPO β=0.1 sigmoid; GRPO KL β=0.01 | `run_metadata.json` |
| eval scoring temperature 0.1, seed 42 | `eda_analysis/scoring/registry.py::EVAL_TEMPERATURE`, `pipeline.py` |
| 22 model states × 8 rubrics × 96 convs | `tables/8_measurement/multijudge_coverage.md` |
| held-out judge full coverage 22,272/22,272 | `tables/8_measurement/multijudge_coverage.md` |
| judge level offset 1.2–1.7 pts, model-dependent | `docs/LIMITATIONS.md` §2 finding 3 |

## §5 Results

| Claim | Value | Source |
|---|---|---|
| PTO Q1+Q2 base → @10 | 3.000 → 4.260, Δ 1.259, dz 1.429 | `tables/7_stats/gpt-4o-mini/main_results.md` |
| GRPO Q1+Q2 base → @10 | 3.067 → 3.753, Δ 0.686, dz 0.721 | same |
| GRPO Q1+Q2 peak (iter 8) | 4.082, Δ 1.016, dz 1.220 | same (`target=best`) |
| Table `tab:endpoint` levels (all 7 metrics, both arms) | — | `tables/1_outcomes/gpt-4o-mini/leaderboard_scorecard.md` |
| Table `tab:endpoint` Δ/dz vs base | — | `tables/7_stats/gpt-4o-mini/main_results.md` |
| Table `tab:endpoint` paired PTO−GRPO @10 + p_holm | Q1Q2 +0.507 dz 0.729 p 0.000; WAI +0.059 p 0.242; CSQ +0.172 p 0.015; MI-SAT +0.174 p 0.015; MITI +0.352 p 0.000; PCT +0.056 p 0.015; MICI −0.346 p 0.000 | `tables/7_stats/gpt-4o-mini/method_paired_by_K.md` (iteration=10) |
| gap at iteration 8 = +0.138 | dz 0.306, p_holm 0.028 | same (iteration=8) |
| steelman PTO@10 − GRPO@8 = +0.177 | dz 0.296, p_holm 0.010 | `tables/7_stats/gpt-4o-mini/method_paired_best.md` |
| OLS slopes 0.120 (PTO) / 0.072 (GRPO); peaks 10 / 8 | — | `tables/7_stats/gpt-4o-mini/slope_by_arm.md` |
| iter-9 dip | — | `tables/7_stats/gpt-4o-mini/grpo_iter9_check.md` |
| loop% base 0.479 (GRPO) / 0.490 (PTO) → 0.000 | — | `tables/3_validity/gpt-4o-mini/session_shape_by_iter.md` |
| mean turn length 266→896 (GRPO), 301→686 (PTO) chars | — | same |
| PC1 91% → 55.0 / 55.7% | — | `tables/7_stats/gpt-4o-mini/rubric_pca_pc1.md`; the 91% figure is the pre-`EXTRA_METRICS` value, owned by `results/L0/SUMMARY.md` §3 |
| PCT ρ 0.79–0.94 with the halo rubrics | — | `SUMMARY.md` §3 / `docs/LIMITATIONS.md` §4 |

## §6 Reward hacking

| Claim | Value | Source |
|---|---|---|
| MICI base → @10 | 0.213→0.491 (PTO, ×2.3); 0.211→0.838 (GRPO, ×4.0, dz 1.717) | `main_results.md` + `leaderboard_scorecard.md` |
| PCT base → @10 | 0.489→0.630 (PTO); 0.487→0.574 (GRPO) | same |
| B6_AF per turn | PTO 0.025→0.142; GRPO 0.029→0.154 | `tables/2_questionnaires/gpt-4o-mini/miti_detail_by_iter.md` |
| literal q_per_turn | PTO 0.930→0.550; GRPO 0.829→0.151 | `tables/3_validity/gpt-4o-mini/session_shape_by_iter.md` |
| oracle B3_Q per turn | PTO 0.485→0.405; GRPO 0.446→0.319 | `miti_detail_by_iter.md` |
| question syntax vs function divergence | — | `docs/METRICS_REFERENCE.md` §4 (audited, not a bug) |
| lexical praise: GRPO ≈3.5× PTO @10 | — | `SUMMARY.md` §4 (`overpraise_crosscheck`) |
| Table `tab:miti` threshold verdicts | R:Q 0.61→0.75 / 0.70→1.43; %CR 0.31→0.36 / 0.30→0.41; Tech 2.92→3.93 / 3.05→3.64; Rel 3.34→4.61 / 3.40→4.20 | `tables/2_questionnaires/gpt-4o-mini/miti_threshold_verdicts.md` |
| MITI thresholds are expert opinion, 20-min audio sessions | — | `docs/METRICS_REFERENCE.md` §2b |
| top Q2 item deltas @10 | PTO: "put himself in my shoes" +1.542, "revealed his thinking" +1.479, "took charge" +1.479. GRPO: +1.073 / +1.010 / +0.990 (same three, different order) | `tables/2_questionnaires/gpt-4o-mini/q2_item_deltas.md` (target=final) |

⚠ Draft rounds PTO's three to +1.54 / +1.48 / +1.48 and GRPO's to +1.07 / +1.01 / +0.99.
The prose says "the three largest per-item gains are the same three in both arms" — true;
it does **not** claim a single shared top item, because PTO's top is item 6 and GRPO's is item 2.

## §7 Judge validity

| Claim | Value | Source |
|---|---|---|
| ICC primary / held-out | Q1 .982–.994 / .951–.978; Q2 .955–.992 / .938–.963; MICI .864–.943 / .525–.929 | `docs/LIMITATIONS.md` §1; `tables/8_measurement/oracle_repeatability_icc.md`, `second_judge_agreement.md` |
| sign preservation 88.3 / 94.1 / 97.0 / 98.9% over 1,848 | — | `tables/8_measurement/multijudge_sign_preservation.md` |
| 18/18 anchor contrasts preserved | — | `tables/8_measurement/second_judge_contrasts.md`; `SUMMARY.md` §7 |
| held-out judge widens Q1 gap +0.77 vs +0.53 | — | `docs/LIMITATIONS.md` §2 finding 1 |
| arm×judge = 1.2–6.9% of arm-mean variance; dependability 0.88–0.95 | — | `tables/8_measurement/multijudge_variance_components.md` |
| MITI: 3.6% between-arm signal, dependability 0.65, 77.5% signs, 88.2% at \|Δ\|≥0.25 | — | `multijudge_variance_components.md` + `multijudge_sign_preservation_by_metric.md` |
| cross-judge r vs measured ceiling (Q1 86–91%, Q2 83–88%, MICI 29–59%) | — | `docs/LIMITATIONS.md` §1 |
| Q1 gain retention PTO@10 0.795 [0.677,0.934]; GRPO@10 0.284 [0.057,0.427]; GRPO@8 0.644 [0.524,0.782] | — | `tables/8_measurement/multijudge_gain_retention.md` |
| Q2 retention flat 0.80–0.85, overlapping | PTO@10 0.849; GRPO@10 0.805 | same |
| GRPO net Q1 gain ≈0.19 (judge) vs ≈0.68 (primary) | delta_judge 0.194 / delta_primary 0.683 | same |
| Table `tab:retention` per-iteration Q1 retention | PTO .97 .84 .89 .94 .98 .97 .94 .89 .88 .80 / GRPO 1.13 .79 .89 .79 .73 .57 .70 .64 .03 .28 | same |

## §8 Mechanism (body summary) + Appendix B (full)

Split 2026-08-10: the body keeps the affirmation-push headline, the loss-vs-data table and
the PTO-yield caution; `sections/B_mechanism.tex` carries the probe construction, the probe
audit, both figures, and the per-iteration series. Every row below applies to whichever of
the two states the number.


| Claim | Value | Source |
|---|---|---|
| weight rescaling to Σ\|w\|=2; DPO ±1 vs GRPO standardized advantage | — | `docs/METRICS_REFERENCE.md` §6b |
| per-iteration directions unusable (split-half 0.15–0.32) | — | `tables/6_preference/gpt-4o-mini/update_direction_quality.md`; `SUMMARY.md` §6 correction box |
| affirmation push GRPO −0.006 → +0.086 ±0.008 (iter 10) | — | `tables/6_preference/gpt-4o-mini/update_lexical_push.md` |
| affirmation push PTO 0.008 → 0.103 ±0.029 **(iter 8, its max)** | PTO@10 is 0.039 ±0.038 — noisier, fewer groups | same |
| GRPO push dips negative at iter 9 | −0.015 | same |
| pool affirm 0.021→0.538 (GRPO), 0.044→0.571 (PTO) | — | `tables/6_preference/gpt-4o-mini/generation_pool_means.md` |
| pool over-praise → 0.741 (GRPO) | PTO 0.318 | same |
| pool question 0.710→0.063 (GRPO) | PTO 0.668→0.272 | same |
| pooled update-direction cosine 0.267 raw / 0.317 corrected (ceiling 0.844) | — | `tables/6_preference/gpt-4o-mini/update_direction_cosines.md` |
| weighting decomposition 0.908 / 0.988 (rule swapped) vs 0.397 / 0.324 (data swapped) | — | `tables/6_preference/gpt-4o-mini/weighting_decomposition.md` |
| PTO yield: built 949→410, trained 782→281, yield 0.824→0.685, margin 0.274→0.196 | — | `tables/6_preference/gpt-4o-mini/training_signal_yield.md` |
| GRPO yield 0.938–0.984 | draft says 94–98% | same |
| ΔMICI ~ push (partial ρ, iter partialled out): GRPO affirm 0.647 p.043, len 0.706 p.023, overpraise 0.617 p.057; PTO −0.492 ns | — | `tables/6_preference/gpt-4o-mini/pref_outcome_correlations.md` |

⚠ **Read the partial ρ, never the raw** — both sides trend with iteration by construction
(`METRICS_REFERENCE.md` §6b).

## Appendix

| Claim | Source |
|---|---|
| held-out sweep \$42 batched; 3,621 in + 71 out tokens/call | `SUMMARY.md` §7 / root `STATUS.md` |
| ICC reps \$9.16 | root `history/CHANGELOG_STATUS.md`, 2026-07-28 |
| 1 rep not 3: oracle noise ≈0.01 vs persona ≈0.09 on an arm mean | `docs/LIMITATIONS.md` §1 "Why breadth was bought before depth" |
| **OpenAI total ≈\$300 — covers ALL arms incl. K=5** | root `STATUS.md`; ⚠ must be split before quoting in this paper |

## Open TODOs blocking submission

1. **Author list** and whether the ICLR 2025 paper is cited as main conference or workshop.
2. **Cost split** — the ≈\$300 OpenAI figure is project-wide; this paper needs the K=0 share.
3. **GPU hours** per arm — pull from `run_metadata.json` / W&B.
4. **Related work** needs 1–2 verified recent MI+LLM-simulation citations (`refs.bib` has a
   flagged placeholder — do not cite it unverified).
5. **Release plan** — conversations + score lake + code?
6. `tab:signladder` per-row `n` values.
