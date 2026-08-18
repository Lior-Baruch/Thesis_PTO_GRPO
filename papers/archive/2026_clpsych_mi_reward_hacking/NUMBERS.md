# NUMBERS.md — the claims ledger

Every quantitative claim in the draft, mapped to the tracked artifact it came from. **No number
in this paper may be written from memory.** If `eda/tools/render_views.py` is re-run and a value
moves, this file is how you find every sentence that has to change.

Paths are relative to `Exp3_PTO_GRPO/eda/results/L0/`. The grader is `gpt-4o-mini` (the primary
oracle, which *was* the training reward) unless the row says `claude-haiku-4-5` or the artifact
lives under `8_measurement/` (which holds both graders in one file, so it has no `<judge>/` level).

**View is `L0` throughout** — this paper is K=0 only. Nothing here reads the `L5` view.

---

## How to re-verify after an EDA re-render

```powershell
& ..\..\.venv\Scripts\python.exe sync_figures.py --check   # did any figure move?
& ..\..\.venv\Scripts\python.exe sync_figures.py           # re-copy
```

then walk the tables below. `sync_figures.py` exits non-zero if a *source* figure vanished
(an EDA refactor renamed it), which is the failure worth catching automatically.

---

## §3 Setup

| Claim | Value | Source |
|---|---|---|
| PC1 share, 5 global rubrics only | ≈91% | `docs/METRICS_REFERENCE.md` §1 |
| PC1 share, all 8 instruments | 55.0 GRPO / 55.7 PTO | `tables/7_stats/gpt-4o-mini/rubric_pca_pc1.md` |
| PCT correlates with halo rubrics | ρ ≈ 0.79–0.94 | `L0/SUMMARY.md` §3 |
| Bootstrap resamples | 2,000 | `eda_analysis/stats.py::bootstrap_ci` default |
| Model states evaluated | 22 (2 arms × 11) | `tables/8_measurement/multijudge_variance_components.md` (`n_arms`) |

### Matched-hyperparameter claims (§3 "What is held fixed")

Verified against **both runs' `run_metadata.json`** on disk, not from `CLAUDE.md`:
`data/pto_Exp3/runs/full/PTO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_M8_PTgreedy/` and
`data/grpo_Exp3/runs/full/GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8/`.

| Parameter | PTO | GRPO | Matched? |
|---|---|---|---|
| candidate sampling temp | `branch_sample_temperature` 1.2 | `grpo_temperature` 1.2 | ✅ |
| candidates per branch point | `num_branches_per_turn` 8 | `num_generations` 8 | ✅ |
| `min_conv_length` | 12 | 12 | ✅ |
| `num_utterances_for_data` | 49 | 49 | ✅ |
| `temperature_therapist_gen` | 0.9 | 0.9 | ✅ |
| `temperature_patient` | 0.7 | 0.7 | ✅ |
| `epochs_per_iteration` / `num_iterations` | 2 / 10 | 2 / 10 | ✅ |
| KL/loss temperature | `dpo_beta` 0.1 | `grpo_beta` 0.01 | ✗ — different objectives, not comparable |
| `pref_filter_tau` | 0.1 | — | PTO-only by construction |

⚠ **The candidate distributions are matched; the STATE distributions are not.** PTO_LA0 ran in
`PTgreedy` mode (in the run name, and `pto_trainer.py::grow_preference_trees_batch` appends the
best completion to advance the trunk), so after a 12-utterance on-policy seed its branch points
come from a best-of-*M* reranked policy. GRPO's prompts are slices of an unmodified on-policy
rollout. **Do not describe this as "exploration"** — an earlier revision did, and it is imprecise;
it is closer to expert iteration. See the paper README § "One methodological point".

## §4 The gains (Table 2 = `tables/main_results.tex`)

Source for **every** cell of Table 2 and all of §4's effect sizes:
`tables/7_stats/gpt-4o-mini/main_results.md` (rows `target=final`, plus `target=best` for the
GRPO iter-8 block).

| Claim | Value | Source |
|---|---|---|
| PTO Q1+Q2 base → iter 10 | 3.000 → 4.259 (Δ 1.259, dz 1.429) | `main_results.md` |
| GRPO Q1+Q2 base → iter 10 | 3.067 → 3.753 (Δ 0.686, dz 0.721) | `main_results.md` |
| GRPO Q1+Q2 at iter-8 peak | 4.082 (Δ 1.016, dz 1.220) | `main_results.md` (`target=best`) |
| PTO MITI base → final | 3.133 → 4.273 (dz 1.347) | `main_results.md` |
| GRPO MICI dz | 1.717 (*deterioration*) | `main_results.md` |
| **Paired FINAL vs FINAL** (10 v 10), Q1+Q2 | +0.507, dz 0.729, p_holm 0.000 | `tables/7_stats/gpt-4o-mini/method_paired_by_K.md` (K=0, iteration=10) |
| …MITI / MI-SAT / CSQ-8 / PCT | +0.352 / +0.174 / +0.172 / +0.056 | same |
| …MICI (PTO better) | −0.346, dz −0.989 | same |
| Arms level or GRPO-favouring ≤ iter 5 | iter 3 Δ = −0.179 (GRPO ahead) | same |
| **Paired BEST vs BEST** (PTO@10 v GRPO@8), Q1+Q2 | +0.177, dz 0.296, p_holm 0.010 | `tables/7_stats/gpt-4o-mini/method_paired_best.md` |
| …Q1 / MI-SAT / CSQ-8 / WAI-SR (all survive Holm) | +0.204 / +0.212 / +0.197 / +0.123 | same |
| …**MITI n.s.** | +0.039, dz 0.064, p_holm **0.566** | same |
| …**MICI n.s.** | −0.044, dz −0.134, p_holm **0.516** | same |
| "Best" under the HELD-OUT judge | PTO@**9** vs GRPO@**3** (not 10 v 8) | `tables/7_stats/claude-haiku-4-5/method_paired_best.md` |
| OLS slopes | PTO 0.120, GRPO 0.072 | `tables/7_stats/gpt-4o-mini/slope_by_arm.md` |
| Peak iterations | PTO 10, GRPO 8 | same (`peak_iter`) |
| Endpoint leaderboard levels | PTO 4.260 / GRPO 3.753 / GRPO@8 4.082 | `tables/1_outcomes/gpt-4o-mini/leaderboard_scorecard.md` |

### Table 3 (thresholds) — `tables/thresholds.tex`

Every cell from `tables/2_questionnaires/gpt-4o-mini/miti_threshold_verdicts.md`, verbatim.

⚠ **Do not restate the technique-ratio verdicts as a finding.** MITI dependability is 0.65 and
its sign preservation is the worst of the eight instruments (§6 ledger below); the draft states
these descriptively and leans no contrast on them. Keep it that way.

## §5 What the therapist learned (Table 4 = `tables/behaviour.tex`)

| Claim | Value | Source |
|---|---|---|
| Turn length PTO / GRPO | 300.6 → 686.2 / 266.3 → 895.7 chars | `tables/3_validity/gpt-4o-mini/session_shape_by_iter.md` (`mean_turn_len`) |
| **Question rate, regex `?`** PTO / GRPO | 0.930 → 0.550 / 0.829 → 0.151 | same (`q_per_turn`) |
| **Question rate, oracle MITI B3** PTO / GRPO | 0.485 → 0.405 / 0.446 → 0.319 | `tables/2_questionnaires/gpt-4o-mini/miti_detail_by_iter.md` (`B3_Q_per_turn`) |
| GRPO curves cross | between iters 4 and 5 | `figures/3_validity/gpt-4o-mini/question_rate_crosscheck.png` (read off the figure) |
| PTO never inverts | regex stays above coded all 11 states | both tables above, compared per iteration |
| Affirmations/turn (B6) | PTO 0.025 → 0.142; GRPO 0.029 → 0.154 | `miti_detail_by_iter.md` |
| Complex reflections/turn | PTO 0.086 → 0.105; GRPO 0.093 → 0.167 | `miti_detail_by_iter.md` (`B5_CR_per_turn`) |
| MICI rate | PTO 0.213 → 0.491; GRPO 0.211 → 0.838 | `tables/2_questionnaires/gpt-4o-mini/mici_behavior_by_iter.md` (`MICI_Rate`) |
| Over-praise rate | PTO 0.013 → 0.299; GRPO 0.019 → 0.698 | same (`MICI_OverPraise_rate`) |
| Advice-without-permission flat | PTO 0.121 → 0.163; GRPO 0.131 → 0.136 | same |
| Confrontation → 0 | PTO 0.007 → 0.000; GRPO 0.008 → 0.000 | same |
| Conversation length | PTO 28.4 → 20.4; GRPO 28.8 → 25.2 utt. | `session_shape_by_iter.md` (`conv_len`) |
| Phrase-loop rate | 0.490 / 0.479 → 0.000 | same (`loop`) |
| Top Q2 item deltas (GRPO, final) | revealed thinking +1.073; shoes +1.010; took charge +0.990 | `tables/2_questionnaires/gpt-4o-mini/q2_item_deltas.md` |
| Top Q2 item deltas (PTO, final) | shoes +1.542; cared for +1.531; revealed thinking +1.479; took charge +1.479 (tie) | same |

⚠ **Three easy mistakes here, all previously made.**
1. **`L0/SUMMARY.md` §4 quotes the regex rate (0.83 → 0.15) while citing `miti_detail_by_iter.md`,
   which holds the *oracle* rate (0.446 → 0.319).** They are different measures of the same
   construct, and the divergence is this paper's §5 diagnostic. Always name which one you mean.
2. The summary's "B6_AF 0.52 → 1.98" and "B3_Q 6.4 → 4.1" are **raw per-conversation counts**,
   not rates (`per_turn × n_th_turns`). The draft uses rates throughout.
3. **The top Q2 items are NOT "the same three in both arms"** — an earlier revision said so.
   Self-disclosure (*revealed thinking*) tops only GRPO; PTO's top two are *shoes* and *cared
   for* (warmth), with *revealed thinking*/*took charge* tied just behind at +1.479. The claim
   that survives is that self-disclosure and direction items sit in the **top four of both**.

## §6 Held-out judge (Table 5 = `tables/retention.tex`)

| Claim | Value | Source |
|---|---|---|
| Grid size, **this paper** | 8 × 22 states × 96 = **16,896** cells | `tables/8_measurement/multijudge_coverage.md` |
| Grid size, **full project sweep** | 8 × 39 states × 96 = **29,952** cells | `results/*/tables/8_measurement/multijudge_coverage.md` (312/312 model×metric cells) |
| Sign preservation ladder | 88.3 / 94.1 / 97.0 / 98.9 %; n = 1848 | `tables/8_measurement/multijudge_sign_preservation.md` |
| arm×judge variance share | 1.2–6.9 % | `tables/8_measurement/multijudge_variance_components.md` |
| Dependability, 7 instruments | 0.879–0.953 | same (`dependability_k1`, excluding MITI) |
| **MITI exception** | 3.6 % arm, 94.5 % judge, dep. 0.652 | same |
| MITI sign preservation | 77.5 % all; 88.2 % at \|Δ\|≥0.25 | `tables/8_measurement/multijudge_sign_preservation_by_metric.md` |
| Judge level offset | 1.2–1.7 points | `L0/SUMMARY.md` §7 |
| **Q1 retention PTO@10** | 0.795 [0.677, 0.934] | `tables/8_measurement/multijudge_gain_retention.md` |
| **Q1 retention GRPO@10** | 0.284 [0.057, 0.427] | same |
| Q1 retention PTO@8 / GRPO@8 | 0.892 [0.756, 1.044] / 0.644 [0.524, 0.782] | same |
| Q2 retention, all four | 0.801–0.856, all overlapping | same |
| GRPO@10 Q1 Δ primary vs held-out | 0.683 vs 0.194 | same (`delta_primary`, `delta_judge`) |
| Q1 retention trajectory, GRPO | 1.13, 0.79, 0.89, 0.79, 0.73, 0.57, 0.70, 0.64, 0.03, 0.28 | same, iters 1–10 |
| Q1 retention trajectory, PTO | 0.97, 0.84, 0.89, 0.94, 0.98, 0.97, 0.94, 0.89, 0.88, 0.80 | same |
| MICI cross-judge agreement | r 0.20–0.55 | `L0/SUMMARY.md` §7 |

⚠ **29,952 is the FULL project sweep (39 model states), not this paper's grid.** This paper's
`L0` view has 22 states, so its grid is 8 × 22 × 96 = **16,896**. An earlier revision wrote
"22,272 cells — 8 instruments × 22 model states × 96", which does not multiply out. The **$42
sweep cost** of $42 was for the 22,272 cells that existed when it was paid; the grid has since
grown to 29,952 (the four GRPO K=5 states added ~$2.90 batched). Only the per-paper cell count is
16,896, and that is unchanged — this paper's 22 model states did not move.

⚠ **The retention intervals are non-overlapping at iterations 9–10 ONLY.** An earlier revision
claimed they were "still disjoint at GRPO's iteration-8 peak" — **false**, and caught by asking
for a best-vs-best comparison. Check the arithmetic before restating it:

| pair | PTO CI | GRPO CI | overlap? |
|---|---|---|---|
| matched @10 | [0.677, 0.934] | [0.057, 0.427] | **no** — the claim |
| matched @8 | [0.756, 1.044] | [0.524, 0.782] | yes (0.756 < 0.782) |
| best v best (PTO@10, GRPO@8) | [0.677, 0.934] | [0.524, 0.782] | yes |

What IS robust: **PTO's Q1 retention exceeds GRPO's at every iteration from 4 onward**
(.94/.98/.97/.94/.89/.88/.80 vs .79/.73/.57/.70/.64/.03/.28). State it as a consistent ordering
whose endpoint separates, never as a peak-vs-peak significant contrast.

⚠ **Never average the two graders' raw scores.** The primary oracle was the training reward and
the second judge is held out — that is train-vs-test, not two raters. The level offset is
1.2–1.7 points *and model-dependent*, so averaging applies a silent model-dependent shrinkage to
every effect. Only contrasts and standardized quantities may be combined.

## §7 Discussion — the compute-axis paragraph

⚠ **These artifacts live under `results/L5/`, the one deliberate exception to "view is `L0`
throughout".** The compute axis is *owned* by the L5 results tree (`eda_analysis/compute.py`
renders there), but these rows describe the two **L0 arms** this paper compares — no claim
overlaps the sibling draft, which is PTO-only and never quotes GRPO compute.

| Claim | Value | Source |
|---|---|---|
| GPU-hours, 10 iterations | PTO_LA0 **8.119** vs GRPO_LA0 **27.906**; 27.906/8.119 = **3.4×** | `../L5/tables/7_stats/gpt-4o-mini/compute_by_arm.md` |
| Why: build vs in-loop reward | PTO build 5.669 of 8.119 h, once per iteration | same (`build_h`) |
| Matched-budget contrast (PTO@10 vs GRPO@3, ratio 1.011) | Q1+Q2 +0.266, dz 0.529, p_holm <.001 | `../L5/tables/7_stats/gpt-4o-mini/iso_compute_contrast.md` |
| …MICI there (PTO **worse**) | +0.261, dz 0.904, p_holm <.001 | same |
| …both replicate held-out | Q1+Q2 +0.230/dz 0.456/p .0002; MICI +0.418/dz 1.280/p <.001 | `../L5/tables/7_stats/claude-haiku-4-5/iso_compute_contrast.md` |

⚠ Never quote the matched-budget reward win without the MICI deterioration beside it — at equal
spend PTO has trained ten iterations to GRPO's three, and the hack tracks optimization depth.

## §7 Discussion / Appendix B (the probe)

| Claim | Value | Source |
|---|---|---|
| Update cosine as trained | 0.267 raw / 0.844 ceiling / **0.317 corrected** | `tables/6_preference/gpt-4o-mini/weighting_decomposition.md` |
| Same data, rule swapped | 0.908 (PTO groups) / 0.988 (GRPO groups) | same |
| Same rule, data differ | 0.397 (group-rel.) / 0.324 (best-worst) | same |
| Affirmation push GRPO | −0.006 → +0.086 ± 0.008 (iter 10) | `tables/6_preference/gpt-4o-mini/update_lexical_push.md` |
| Affirmation push PTO | 0.008 → 0.103 ± 0.029 **at iter 8** (iter 10 = 0.039 ± 0.038) | same |
| PTO groups built → trained | 949 → 410 built; 782 → 281 trained | `tables/6_preference/gpt-4o-mini/training_signal_yield.md` |
| PTO τ yield / margin decay | 0.824 → 0.685; margin 0.274 → 0.196 | same |
| GRPO yield rate | 0.938–0.984 ("94–98 %") | same |
| Pooled split-half reliability | PTO 0.597 / GRPO 0.911 | `tables/6_preference/gpt-4o-mini/update_direction_quality_pooled.md` |
| Pooled held-out wins | PTO 0.566 / GRPO 0.597 | same |
| Generation-side drift | affirm 0.02 → 0.54 GRPO / 0.04 → 0.57 PTO; over-praise 0.74; questions 0.71 → 0.06 | `L0/SUMMARY.md` §6 |
| ΔMICI ~ push partial ρ (GRPO) | affirm 0.647 (p .043); length 0.706 (.023); over-praise 0.617 (.057) | `tables/6_preference/gpt-4o-mini/pref_outcome_correlations.md` |
| PTO equivalent | −0.492, n.s. | same |

⚠ **The PTO affirmation push peaks at iteration 8, not 10.** Writing "0.103 at the endpoint" is
wrong; iteration 10 falls back to 0.039 on PTO's thinnest group count (281 groups).

⚠ **Do not resurrect the `wins_correct` 0.65 → 0.71 claim.** It was in-sample — the direction was
scored on the pairs it was fitted on. Held out, per-iteration PTO directions win 0.47–0.59 with
split-half reliability 0.15–0.32. Only the *pooled* direction and the exact lexical contrasts
survive, and Appendix B uses pooled only.

## Appendix A / C (measurement + reproducibility)

| Claim | Value | Source |
|---|---|---|
| Oracle ICC(2,1) | Q1 .982–.994; Q2 .955–.992; MICI .864–.943 | `docs/LIMITATIONS.md` §1 |
| Mean \|Δ\| between reps | .047–.070 / .076–.089 / .037–.069 | same |
| Oracle noise vs persona sampling | ≈0.01 vs ≈0.09 on an arm mean | `docs/LIMITATIONS.md` §1 |
| Haiku full sweep cost | ≈$42 for the 22,272 cells then extant; grid now 29,952 | `project-multijudge-eda-next` memory + `docs/LIMITATIONS.md` §1 |
| Project API spend | ≈$317 | `STATUS.md` § Cost constraint |

⚠ **Never quote a judge sweep cost from memory or pro-rata.** Price it off
`judge_plan.sweep_report(..., receipt=(42.0, 22272))` — the receipt-calibrated basis put one
iteration at $0.72 where the char estimator said $1.33 and a pro-rata guess said $1.87.

---

## Open TODOs

- [x] **Figure 3 legibility — resolved 2026-08-18.** The EDA now renders a Q1-only variant
      (`8_measurement/multijudge_retention_trajectory_Q1.png`); the body embeds it as Figure 3
      and the full 7-panel grid moved to Appendix A (`fig:retention_grid`), keeping the
      instrument-specificity argument with a figure to point at.
- [ ] **Co-author list and order** — currently commented out in `main.tex`; confirm before submission.
- [ ] **Artifact release** — decide what ships publicly (transcripts + scores are synthetic and
      releasable; adapter release needs co-author sign-off). Placeholder `\todo` in `C_repro.tex`.
- [ ] `moyers2016miti` has key year 2016 but `year = {2015}` (manual revision June 2015). The
      rendered citation says 2015, which is correct; the key name is merely misleading. Left as-is
      because renaming the key touches every citation.
- [x] **Both inherited `refs.bib` entries verified 2026-08-18.** `steenstra2024scaffolding` was a
      raw placeholder → replaced with the real entry (Steenstra, Nouraei & Bickmore, **CHI 2025**,
      arXiv:2502.18673), key renamed `steenstra2025scaffolding`. `chen2025broaden` (ICLR 2025) is
      real but is a conversation-*planning* paper — its citation moved from the MI paragraph to
      the tree-search/look-ahead paragraph, and the MI paragraph now cites
      `perezrosas2019goodcounselor` (ACL 2019) instead. Five further verified entries added
      (`amodei2016concrete`, `pan2022effects`, `singhal2024long`, `lightman2024letsverify`,
      `perezrosas2019goodcounselor`), shared with the sibling draft's bib.
