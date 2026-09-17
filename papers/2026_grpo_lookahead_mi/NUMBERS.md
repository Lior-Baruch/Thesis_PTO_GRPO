# NUMBERS.md — the claims ledger

Every quantitative claim in the draft → the exact tracked artifact it came from. `results/…` paths
are relative to `Exp3_PTO_GRPO/eda/`. **Nothing enters the `.tex` that is not a row here**, and no
row is written from prose — each was read off the named table or recomputed from the score lake.

*(2026-09-02 rewrite: sections renumbered — §3 is now the method, §4 the setup, §5 reward, §6
behaviour, §7 mechanism, §8 measurement, §9 discussion; appendices A tables · B mechanism · C repro.
New rows are marked **NEW**. Numbers unchanged from the 2026-08-27 ledger keep their marks.)*

*(2026-09-16 restructuring — sections renumbered again: §7 is now the saturation section (was §8)
and §8 the discussion (was §9); the one-paragraph mechanism section was folded into §8 as the
paragraph "What we could not isolate" (`\label{sec:mechanism}` still resolves there). §3 gained
Algorithm 1. Numbers unchanged; the rows moved or added by this pass are in the **2026-09-16**
block at the end.)*

*(2026-09-04 refinement — the 2×2 companion draft was retired, so this is the ONE submission.
Structure: the endpoint table is now **Table 1 in the body** (§5), the matched-persona excerpt is
**Table 2** (§6) with the full utterances in a new **Appendix D**, the rollout audit moved from §3
to the Limitations, §7 is a single paragraph, and Figures 3–4 are redrawn from their tracked
tables by `render_paper_figures.py`. An independent audit of every number in the .tex against its
table (437 cells) found five prose discrepancies, all fixed and marked **AUDIT-FIX** below. Rows
marked **NEW-0904** were added for the excerpt, the redrawn figures and the new citations.)*

**Graders.** `primary` = `gpt-4o-mini` (this WAS the training reward). `held-out` =
`claude-haiku-4-5` (never touched training). ⚠ **Levels are not comparable across graders** and
the two are **never averaged** — only contrasts and standardized quantities combine. **AUDIT-FIX:**
the offset the paper quotes is **1.1–1.8 points on Q1+Q2 across the 22 GRPO states** (recomputed
2026-09-04 from `results/lookahead/reward/tables/reward.xlsx` sheet `k_levels_long`, primary −
held-out per state: 1.13 at `GRPO_LA5` base … 1.81 at `GRPO_LA0` I9; endpoints 1.50 / 1.64). The
old "1.2–1.7" was a four-arm figure with no table behind it in this ledger.

**Sign conventions — the transposition trap.** The EDA's K tables (`lookahead/reward/*`,
`lookahead/behaviour/*`) report **K=0 minus K=5**, so a *positive* cell there means K=0 scored
higher / did more. This paper argues for K=5, so most body sentences carry the **opposite** sign.
Every row below states the direction it is in. `MICI` is **lower-is-better**; `k_summary`'s
`*_higher` columns are raw direction and its `*_better` columns are polarity-corrected.

**Verification status.** Rows marked ✅ were independently recomputed from the score lake
(`data.load_scores_long` + `stats.paired_arrays`, persona-paired, n=96) on 2026-08-25 and agree
with the cited table. Rows marked 📄 were read off the cited table only.

---

## §3 Method + §4 Setup (config facts)

| claim | value | source |
|---|---|---|
| Arms in this paper | `GRPO_LA0`, `GRPO_LA5` — 2 arms × 11 states (base + 10 iterations) = 22 model states | results/lookahead/reward/tables/k_levels.md |
| Parent experiment | 4 arms × 11 states = 44 model states, both graders — ⚠ **NOT quoted anywhere in the paper**: every full-grid statistic was recomputed on the 22 GRPO states (§8). PTO appears only as cited prior work (`baruch2025pto`), never as data | results/measurement/validity/tables/multijudge_coverage.md |
| Personas | 2 gender × 3 cooperation × 2 problem × 2 duration × 2 prior attempts × 2 age = 96 | Exp3_PTO_GRPO/code/system_prompts_builder.py `generate_all_permutations` |
| Matched knobs | MCL=12, G=8, KL β=0.01, temp 1.2, batch 64×2, 2 epochs/iter, 10 iterations, same oracle (Q1+Q2), same temps | `run_metadata.json` of both arms (CONFIG FACT) |
| ✅ **The two arms differ in exactly two substantive config fields** | `lookahead_k` (absent vs 5) and `lookahead_sub_batch_size` (absent vs 128); everything else identical except the arm's own name, adapter repo and two output paths | diff of the two `run_metadata.json` files, verified 2026-08-25 |
| Instruments | 8 — Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI, PCT, MICI | results/arms/stats/tables/gpt-4o-mini/main_results.md |
| Training reward | Q1+Q2 only (mean of the two) | CLAUDE.md § Exp3 (CONFIG FACT) |
| Bootstrap | 2,000 resamples, percentile, `BOOT_SEED` = 12345 | eda_analysis/constants.py |
| **NEW** 📄 K=5 reward's extra calls per candidate | 3 patient-simulator calls + 2 policy generations (5 further turns: P, π, P, π, P) beyond the shared single oracle call | CLAUDE.md § "K-turn look-ahead" (CONFIG FACT: K counts utterances, alternating, patient first) |
| **NEW** 📄 Rollout audit — full-tail share, pooled | `full_share` **0.818** (82%); `realized_turns_mean` **4.401** (4.4); `ended_early_rate` **0.182**, range **0.122** (iter 10) to **0.304** (iter 6); `patient_closed_share` **0.156** (16%); `n_candidates` **121,088** | results/lookahead/mechanism/tables/tail_audit_by_iter.md, `GRPO_LA5` rows (pooled + iters 6, 10) |
| **NEW** 📄 Rollout audit — early-ending candidates score at or below the group mean | `dev_mean` by realised turns, GRPO_LA5 pooled: 0 turns −0.050, 1 (patient closed) −0.012, 2 (therapist end) −0.083, 3 (patient closed) +0.001, 4 (therapist end) −0.086, 5 (full) +0.003 → "at or below, by up to 0.09 depending on how the rollout ended" | results/lookahead/mechanism/tables/tail_score_by_realized_turns.md, `GRPO_LA5 / pooled` rows |
| **NEW** 📄 Rollout audit — early-ending candidates are the group argmax less often than chance | `p_chosen_given_ee` **0.101** vs `p_chosen_given_full` **0.130**; chance 1/8 = 0.125; relative risk 0.773 [0.749, 0.797] | results/lookahead/mechanism/tables/tail_within_group.md, `GRPO_LA5 / pooled` |

⚠ **"0.05–0.09 below" is only true of the no-tail and therapist-ended rollouts.** Patient-closed
rollouts (1 or 3 realised turns) sit within ±0.012 of the group mean. The paper says "at or below
… by up to 0.09 points, depending on how the rollout ended" — do not simplify it back.

## §5 Reward

| claim | direction | value | source |
|---|---|---|---|
| ✅ K contrast at iteration 10, Q1+Q2, primary | K=5 higher | **+0.765**, dz 0.905, CI [0.601, 0.943], p 6.3e-13 | results/lookahead/reward/tables/k_endpoints.md (row `GRPO_LA5_I10 − GRPO_LA0_I10`) |
| ✅ K contrast at iteration 10, Q1+Q2, held-out | K=5 higher | **+0.616**, dz 1.030, CI [0.502, 0.732], p 4.6e-13 | same row, `judge_*` columns |
| ✅ `GRPO_LA0` vs own base, Q1+Q2, primary | gain | 3.067 → 3.753 = **+0.686**, dz 0.721 | results/arms/stats/tables/gpt-4o-mini/main_results.md |
| ✅ `GRPO_LA5` vs own base, Q1+Q2, primary | gain | 2.963 → 4.517 = **+1.554**, dz 1.518 | same |
| ✅ `GRPO_LA0` vs own base, Q1+Q2, held-out | gain | 1.861 → 2.257 = **+0.396**, dz 0.658 | results/arms/stats/tables/claude-haiku-4-5/main_results.md |
| ✅ `GRPO_LA5` vs own base, Q1+Q2, held-out | gain | 1.834 → 2.873 = **+1.038**, dz 1.539 | same |
| ✅ Gain ratio, primary | K=5 / K=0 | 1.554 / 0.686 = **2.27×** | derived — show the arithmetic |
| ✅ Gain ratio, held-out | K=5 / K=0 | 1.038 / 0.396 = **2.62×** | derived |
| 📄 K=5 ahead on all 8 instruments at iteration 10, both graders | K=5 better | 9 metric rows (8 instruments + the Q1Q2 composite), all favouring K=5, every $p_{holm}$ .000 under both graders | results/lookahead/reward/tables/k_endpoints.md (`favours_primary` / `favours_judge`, `GRPO_LA5_I10 − GRPO_LA0_I10`) |
| 📄 **Endpoint table in full** (Table 1, now in the BODY, §5) — $\Delta$ (dz) per instrument, primary \| held-out. **AUDIT note:** the cited `k_endpoints.md` row is already K5 − K0 and is copied unflipped; only `k_table1.md` (Table 3) needed the sign flip | K=5 higher | Q1Q2 +0.765 (0.905) \| +0.616 (1.030) · Q1 +0.858 (0.902) \| +0.865 (1.152) · Q2 +0.671 (0.864) \| +0.367 (0.609) · WAI-SR +0.291 (0.513) \| +0.288 (0.442) · CSQ-8 +0.289 (0.482) \| +0.451 (0.678) · MI-SAT +0.352 (0.531) \| +0.503 (0.829) · MITI +0.615 (0.735) \| +0.276 (0.502) · PCT +0.111 (0.516) \| +0.113 (0.563) · MICI −0.627 (−1.862) \| −0.422 (−1.567) | results/lookahead/reward/tables/k_endpoints.md, the nine `GRPO_LA5_I10 − GRPO_LA0_I10 (K lever, GRPO matched iter)` rows |
| 📄 **By-iteration table** (Table 3 since 2026-09-04; Appendix A) — Q1+Q2 $\Delta$ (dz) at iterations 0–10, both graders | K=5 higher | reproduced cell-by-cell from the source's GRPO columns with **every sign negated**; the 22 star decisions were re-checked 2026-09-04 against the exact `p_holm` in `reward.xlsx` sheet `k_headline_grpo_data` (primary I4 .044 *, I6 9.4e-4 ***, I7/I8 3.6e-3 **, I9/I10 ***; I3 .070 and I5 .487 unstarred; held-out I4/I5 6.96e-3 **, I6 7.4e-5 ***, I7 6.1e-4 ***, I8 .0505 unstarred, I9/I10 ***) | results/lookahead/reward/tables/k_table1.md |
| ✅ **LEVEL columns of both appendix tables** | — | endpoint per instrument: primary K0/K5 e.g. Q1Q2 3.753/4.517, MICI 0.838/0.210; held-out Q1Q2 2.257/2.873; by-iteration Q1Q2 levels e.g. primary I8 4.082/4.254 | results/lookahead/reward/tables/reward.xlsx sheets `k_levels_long` and `k_headline_grpo_data` |
| 📄 Figure 2 is LEVELS-only | — | k_headline_q1q2_grpo: trajectories + Holm star row, no delta strip | results/lookahead/reward/figures/k_headline_q1q2_grpo.png + tables/k_headline_grpo_data.md |
| 📄 GRPO K=0's own peak and decline (primary) | — | 4.082 at iteration 8, 3.807 at 9, 3.753 at 10 | results/arms/stats/tables/gpt-4o-mini/main_results.md (`target=best`, `target_iter` 8) + k_table1 |
| **NEW** 📄 GRPO K=0's held-out best state | — | iteration **3** (2.637; `target=best`, `target_iter` 3) | results/arms/stats/tables/claude-haiku-4-5/main_results.md |
| 📄 Iterations where K=5 is Holm-significant on Q1+Q2 | K=5 better | primary 6 of 10 (iters 4, 6, 7, 8, 9, 10); held-out 6 of 10 (iters 4, 5, 6, 7, 9, 10) | results/lookahead/reward/tables/k_summary.md, GRPO/Q1Q2 rows, `iters_sig_K5_higher` |
| 📄 Base-vs-base noise floor (iteration 0) | neither | +0.104 primary (dz 0.115, n.s.), +0.026 held-out (dz 0.043, n.s.) | results/lookahead/reward/tables/k_table1.md row 0, GRPO cols |
| **NEW** 📄 **Best-checkpoint steelman**, Q1Q2 — K=5 endpoint vs K=0's best by primary (I8) | K=5 higher | **+0.435** (dz 0.743, p_holm .000) primary; **+0.256** (dz 0.384, p_holm .000) held-out | results/lookahead/reward/tables/k_endpoints.md, rows `GRPO_LA5_I10 − GRPO_LA0_I8 (K=0 best by primary Q1Q2)`, Q1Q2 |
| **NEW** 📄 Best-checkpoint steelman, Q1Q2 — K=5 endpoint vs K=0's best by held-out (I3) | K=5 higher | **+0.524** (dz 0.977, p_holm .000) primary; **+0.236** (dz 0.386, p_holm .001) held-out | same table, rows `GRPO_LA5_I10 − GRPO_LA0_I3 (K=0 best by held-out Q1Q2)`, Q1Q2 |

⚠ **Do not write "significant at every iteration."** The K advantage is null for the first three
iterations under both graders and only opens from iteration 4.
⚠ **The steelman is a Q1+Q2 statement.** Against `GRPO_LA0_I3` under the held-out judge, MITI is
−0.008 (n.s.) and MICI is +0.220 (K=5 *worse*); against `GRPO_LA0_I8` held-out, Q2 (p_holm .172)
and WAI-SR (.578) are n.s. The paper says "leads on Q1+Q2 under both graders" — keep it there.

### §5 replicate draw

A second independent 96-conversation draw of `GRPO_LA5@10` (same adapter, same 96 personas, same
seed-53 shuffle, unseeded decoding), scored on all 8 instruments by both graders, 0 errors.
Source for every row: results/measurement/replicate_draw.md (written by `eda/tools/replicate_check.py`).

| claim | direction | value | source |
|---|---|---|---|
| ✅ Trained-state noise floor, `GRPO_LA5@10` draw2 − draw1 | neither | 9 metrics × 2 graders, **0 significant after Holm**, max \|dz\| **0.174** (MICI, primary); Q1+Q2 −0.056 (dz −0.121, p .160) primary, +0.021 (dz +0.031, p .963) held-out | replicate_draw.md § "GRPO_LA5 @10, draw 2 − draw 1" |
| ✅ K contrast at iteration 10, Q1+Q2, primary, **replicate** | K=5 higher | **+0.709**, dz 0.919 (original +0.765, dz 0.905) | same, § "K lever @10 — replicate" |
| ✅ K contrast at iteration 10, Q1+Q2, held-out, **replicate** | K=5 higher | **+0.637**, dz 0.949 (original +0.616, dz 1.030) | same |
| ✅ Endpoint level, primary | — | 4.517 (original) → **4.461** (replicate; not printed in the report — derived as 4.517 − 0.056) | same |
| **AUDIT-FIX** effect sizes "within 0.09" | — | held-out dz 1.030 → 0.949 = **0.081**, primary 0.905 → 0.919 = 0.014. The 2026-09-02 text said "within 0.08", which 0.081 violates | same |

⚠ **The re-draw covers the $K{=}5$ side only** — the $K{=}0$ arm is the same draw in both columns.
⚠ **A replicate bounds EVALUATION noise, never TRAINING variance.** One training run per arm.

## Limitations — the cost disclosure

| claim | value | source |
|---|---|---|
| Oracle scoring calls, sum over train iters 1–10 | `GRPO_LA0` **302,541** vs `GRPO_LA5` **289,983** — "approximately matched by construction" | results/compute/cost/tables/api_calls.md, `oracle_calls_train` summed over the 10 iteration rows per arm — **derived by summation; re-sum if the table re-renders** |
| Patient-simulator calls inside K=5 look-ahead rollouts, sum over train iters 1–10 | `GRPO_LA5` **392,766** (≈393k); `GRPO_LA0` **0** by construction | same table, `patient_calls_tail` summed over the 10 GRPO_LA5 rows |
| ✅ Per-step wall-clock multiplier, settled iterations 3–10 | median **1.92×**, range 1.828–2.182 | results/compute/cost/tables/step_multiplier.md, `GRPO_step_ratio_K5_over_K0` |
| ⚠ iterations 1–2 excluded from the multiplier | 2.406 and 2.119 — smaller look-ahead sub-batch, not comparable (stated in Appendix C) | same; CLAUDE.md § Gotchas |
| Ethics total (one line, no per-arm breakdown) | 27.906 + 51.205 = 79.111 ≈ **79 GPU-hours** | results/compute/cost/tables/compute_by_arm.md — show the arithmetic |

⚠ GPU-hours are **reconstructed from artifact mtimes**, gaps outside (0, 3600 s) imputed at the
phase median; never from `iteration_metadata.json` (per-process, undercounts resumed iterations).

## §6 Behaviour

| claim | axis + grader | value | source |
|---|---|---|---|
| ✅ MICI rise, `GRPO_LA0` | rate, primary | 0.211 → 0.838 = **+0.626**, dz 1.717 | results/arms/stats/tables/gpt-4o-mini/main_results.md |
| ✅ MICI rise, `GRPO_LA5` | rate, primary | 0.209 → 0.210 = **+0.001**, dz 0.006, **n.s.** (p .711) | same |
| ✅ MICI rise, `GRPO_LA0` | rate, held-out | 0.384 → 1.050 = **+0.666**, dz 1.975 | results/arms/stats/tables/claude-haiku-4-5/main_results.md |
| ✅ MICI rise, `GRPO_LA5` | rate, held-out | 0.326 → 0.628 = **+0.301**, dz 0.845 — **NOT flat** | same |
| ✅ Held-out ratio of MICI rises | rate, held-out | 0.301 / 0.666 = **0.45** — under half | derived |
| ✅ K contrast on MICI at iteration 10 | rate | primary −0.627 (dz −1.862); held-out −0.422 (dz −1.567), K=5 better both | results/lookahead/reward/tables/k_endpoints.md |
| **NEW** 📄 **MI-inconsistency composition at the endpoint** (per session, primary coder) | count/session | `GRPO_LA0` I10: `MICI_BehaviorTotal` **9.865** (paper: 9.9), `MICI_OverPraise` **8.250** (8.3), `MICI_OverPraise_share` **0.836** (84%); base I0: total **2.844**. `GRPO_LA5` I10: total **2.906** (2.9), over-praise 0.719 (share 0.247), advise-without-permission 1.583 (share 0.545), direct 0.594 (0.204); base I0: total **2.677** (2.7), advise share 0.533 | results/lookahead/behaviour/tables/k_mici_composition.md, `gpt-4o-mini` rows for `GRPO_LA0` / `GRPO_LA5` at iterations 0 and 10 |
| ✅ Over-praise, judge-free lexical marker | **share of therapist turns containing ≥1 marker**, no grader | `GRPO_LA0` 0.671 vs `GRPO_LA5` 0.064 at iteration 10 = **10.5×** | results/arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md, `lex_overpraise_marker_rate`; definition in `eda_analysis/behavior.py` |
| 📄 Over-praise, oracle-rated | rate, primary | 0.698 vs 0.051 at iteration 10 | same table, `MICI_OverPraiseRate` |
| 📄 Over-praise K contrast significance | rate | K=0 worse at 6 iterations (primary, iters 5–10) and 7 (held-out, iters 4–10) | results/lookahead/behaviour/tables/k_channels_summary.md, `MICI_OverPraise_rate` GRPO rows |
| 📄 Questions per therapist turn (text) | per turn, **judge-invariant** | K=5 higher at 7 iterations (4–10), mean dz −0.643 (sign K0−K5; paper quotes 0.643 as K=5 higher) | results/lookahead/behaviour/tables/k_channels_summary.md, `q_per_turn`, `text (judge-invariant)` |
| **NEW** 📄 Endpoint channel effect sizes on the forest (primary coder; sign K0−K5 in the source, quoted as magnitudes in the paper) | per turn, dz | over-praise/turn **+2.29**; affirmations/turn **+0.75**; `'?' marks/turn` **−1.30**; direct/order per turn **−0.60** (`MICI_Direct_rate` −0.598); persuasion per turn **−0.52** (`B2_Persuade_per_turn` −0.522); all Holm-sig | results/lookahead/behaviour/figures/k_channel_forest_grpo_gpt-4o-mini.png (bar labels) + results/lookahead/behaviour/tables/behaviour.xlsx sheet `k_channels_grpo_gpt-4o-mini`, iteration-10 rows |
| **NEW** 📄 Persuasion per turn higher under K=5 at six iterations, held-out coder | per turn | `B2_Persuade_per_turn`, claude-haiku-4-5, `iters_sig_K5_higher` = 3, 4, 5, 7, 9, 10 | results/lookahead/behaviour/tables/k_channels_summary.md |
| **NEW** 📄 Session length at the endpoint | utterances/conversation, judge-invariant | K=0 **25.198** vs K=5 **31.896** (paper 25.2 / 31.9), dz −0.431 (K0−K5; paper quotes 0.43), p_holm .002; therapist turns 12.75 vs 15.97 | results/lookahead/behaviour/tables/length_kcontrast.md, `GRPO iter 10` rows |
| **NEW** 📄 Turn length grew ~3× in both arms | chars/therapist turn | K=0 266.3 → 895.7; K=5 279.0 → 849.3 (base → iter 10) | results/lookahead/behaviour/tables/length_endpoints.md |

⚠ **"Look-ahead prevents the reward hack" is a primary-grader statement.** Write "slows the loop
to under half the rate", cite the judge-free marker, and never write "stops"/"eliminates" without
"under the training oracle".
⚠ The `lex_overpraise_marker_rate` column is identical under both graders (computed from text);
it lives under a `<judge>/` path only because `arms/*` is a per-judge family.
⚠ **Name its axis.** 0.671 = "67% of turns contain ≥1 marker", not "0.671 markers per turn". A
brittle regex kept as a direction check — cite its agreement with the rated rate, never its
absolute value as a measurement.
⚠ **The directive residue is real but small next to over-praise** (dz 0.5–0.6 vs 2.3). The paper
says "a smaller residue" and "part of what it selects is a therapist who pushes" — do not let it
grow into "look-ahead trades flattery for coercion".
⚠ `%MICO` (MITI's MI-consistent share) is **higher under K=0** (dz +0.77 at the endpoint) because
MITI counts affirmations as MI-adherent regardless of whether they are earned. Not quoted in the
paper; if it ever is, say why it points the "wrong" way.

### §6 + Appendix D — the matched-persona excerpt (NEW-0904)

Source: the stored conversation CSVs (`data/grpo_Exp3/conversations/full/<arm>/model_iter_10_TT0.9_TP0.7/`)
and the score lake, via [`select_example_persona.py`](select_example_persona.py) (run 2026-09-04;
`--dump` writes the pick + transcripts as JSON). Nothing in the excerpt is paraphrased: a
normalising diff of every appendix paragraph against the stored text passed on 2026-09-04
(curly quotes → LaTeX quotes, em-dashes → `---` are the only edits).

| claim | value | source |
|---|---|---|
| Selection rule | of the 96 personas, the one minimising \|rank_primary − 48.5\| + \|rank_held-out − 48.5\| of its persona-paired K5 − K0 Q1+Q2 contrast at iteration 10 → **persona 93** (score 7.0; next 90 at 13.0). ⚠ The single-grader rule (closest to the primary median alone) picks persona 47, whose held-out contrast is −0.09 (rank 84/96) — that is why the rule uses BOTH graders | `select_example_persona.py`; `data.canonical_personas()` |
| Persona 93 | Female, 61, Obesity, ManyYears, tried Never, cooperation StartLowAndChangesToHigh (the paper: "61-year-old woman with long-standing obesity … never tried to change … uncooperative at first") | `system_prompts_builder.get_patient_permutation_characteristics(93)` |
| Q1+Q2 at iteration 10, primary | K=0 **3.847** (paper 3.85) vs K=5 **4.376** (4.38); Δ +0.529 (paper +0.53) vs the 96-persona median **+0.532** (+0.53); rank 49/96 | score lake, `Q1Q2` composite, `GRPOExp3_LA{0,5}_I10`, persona-paired |
| Q1+Q2 at iteration 10, held-out | K=0 **1.976** (1.98) vs K=5 **2.700** (2.70); Δ +0.724 (+0.72) vs median **+0.635** (+0.64); rank 42/96 | same, judge `anthropic_claude-haiku-4-5` |
| Files + lengths | both arms: `conversation_87.csv` of `model_iter_10_TT0.9_TP0.7` (file index 87 ↔ persona 93 under the iteration-10 shuffle, `persona_order(42, 10)`); **16 utterances / 8 therapist turns in both** | conversation dirs |
| Table 2 excerpt | utterances 3 (patient, elided with […]) and 4 (therapist) of each; the K=0 therapist turn is cut after "taking the first step." (the remainder proposes SMART goals and hits the 200-token cap); the K=5 therapist turn is complete | Appendix D has 1–9 in full |
| Therapist turns ending mid-sentence | hit the 200-token response cap (`MAX_NEW_TOKENS` 200) — e.g. K=0 utt. 4 ends "based on your", K=5 utt. 8 ends "let's say ``I'm" | config fact; Limitations ¶ "Both policies grew into the response cap" |

⚠ **Superseded 2026-09-14 (Lior, after his read): Table 2 is now a CLEAR case, chosen on
purpose, and its caption says so.** The median-rule persona 93 above stays in the paper as the
TYPICAL case, in Appendix D.2 with both conversations in full. Never present the clear case as
typical, and never drop the typical case.

### Table 2 + Appendix D.1 — the clear case (NEW-0914b)

Source: `select_example_illustrative.py` (this folder) ranks every (persona, therapist-turn) pair
at iteration 10 by transparent lexical features (praise words in the K=0 turn, questions and no
praise in the K=5 turn, resistance in the preceding patient turn, no first-person slip, not cut by
the cap); the pick was made by eye from its top 10. `--persona 84 --turn 2 --dump` writes the
transcripts; the appendix paragraphs were generated from that dump with LaTeX escaping and only
typographic changes (curly quotes, `…` → `\ldots{}`).

| claim | value | source |
|---|---|---|
| Persona 84 | Female, 27, Smoking, ManyYears, tried Never, StartLowAndChangesToHigh ("a 27-year-old woman who has smoked for years, has never tried to quit, and was sent to therapy") | `data.canonical_personas()` |
| Files | both arms `conversation_79.csv` of `model_iter_10_TT0.9_TP0.7` (file index 79 ↔ persona 84 under the iteration-10 shuffle) | conversation dirs |
| Lengths | K=0 **25 utterances / 13 therapist turns**; K=5 **50 utterances = the session cap (49 after the scripted opener) / 25 therapist turns** | same |
| Utterance 1 identical across arms | byte-identical patient opening (`PATIENT1 IDENTICAL: True` from the dump check) — the table shows it once | same |
| Q1+Q2 at iteration 10, primary | K=0 **2.306** (paper 2.31) vs K=5 **4.812** (4.81) | score lake, `Q1Q2`, `GRPOExp3_LA{0,5}_I10`, persona-paired |
| Q1+Q2 at iteration 10, held-out | K=0 **1.676** (1.68) vs K=5 **2.306** (2.31) | same, judge `anthropic_claude-haiku-4-5` |
| Table 2 excerpt | utterance 1 (patient, in full) + utterance 2 of each arm; the K=0 turn is elided twice with […] and ends at the 200-token cap ("I'm ready to support"); the K=5 turn is complete (332 chars) | Appendix D.1 has utterances 1–7 of both |
| Ranking position | the pair scored 21.5 = 7th of all pairs; the pairs above it were later turns (more context needed) or had a K=5 flaw ("user" artifact, a first-person slip, a truncated turn) | `select_example_illustrative.py --top 10` |

## §7 Mechanism

| claim | value | source |
|---|---|---|
| 📄 Reward faithfulness, pooled over matched iterations, primary | K=0 0.873 [0.861, 0.884] vs K=5 0.909 [0.900, 0.917]; difference −0.036 [−0.051, −0.021] | results/lookahead/mechanism/tables/faithfulness_k_summary.md, GRPO `matched_iters` |
| 📄 same, held-out | K=0 0.747 vs K=5 0.800; difference −0.053 [−0.078, −0.030] | same |
| 📄 Iteration-level test | K=5 more faithful at 7 of 10 iterations; Wilcoxon over iterations p = **.084** primary, **.193** held-out | same, `iters_K5_more_faithful` / `wilcoxon_over_iters_p` |
| 📄 Matched-policy cut (train_iter 1) | all deltas straddle zero: primary +0.015 [−0.026, 0.059]; held-out −0.014 [−0.075, 0.048]; per-length bins 17/20 favour K0 (primary) vs 17/20 favour K5 (held-out) | same table `train_iter_1` + METRICS_REFERENCE.md §6a |
| 📄 Dispersion — rescaling not sharpening | pooled margin ratio 1.300 [1.275, 1.326], SD ratio 1.293 [1.267, 1.317], ratio-of-ratios **1.006** [1.002, 1.010] | results/lookahead/mechanism/tables/dispersion_ratios.md, GRPO `pooled` |
| ✅ Iteration-10 inversion | margin ratio 0.679 at train_iter 10: K=0's margin 0.248 → 0.339, K=5's 0.268 → 0.230; ratio-of-ratios 0.964 | same, GRPO rows 9 and 10 |
| ✅ The coinciding over-praise jump | `GRPO_LA0` judge-free marker 0.093 (iter 9) → 0.671 (iter 10) | results/arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md |
| ✅ Update direction barely changes with K | pooled direction cosine **0.804**, ceiling 0.945, **0.851** corrected | results/arms/preference/tables/gpt-4o-mini/update_direction_cosines.md |

⚠ The "update direction" is an **embedding-space proxy**, not the gradient. Quote the corrected
cosine with its ceiling. ⚠ **Do not write "significantly more faithful."** ⚠ **The matched-policy
result is a different question and does not contradict the pooled one** (METRICS_REFERENCE §6a).

## §8 Measurement

*(Every full-grid statistic below is **recomputed on the 22 GRPO states** — the `*_grpo` artifacts.)*

| claim | value | source |
|---|---|---|
| ✅ `GRPO_LA5` per-conversation cross-grader agreement on **Q1** | .941 (I5) → .877 → .842 → .769 → **.487 (I9)** → **.544 (I10)** | results/measurement/validity/tables/validity.xlsx, sheet `second_judge_agreement`; panel-a rows of judge_saturation_grpo_data.md |
| ✅ Q1 median across the 22 GRPO states | **0.842** (exact 0.8415, an even-count median tie; the cited table rounds half-even to 0.842 and since 2026-09-14 the paper matches the table — Figure 4a no longer prints the value) | results/measurement/validity/tables/judge_saturation_grpo_data.md |
| ✅ The two lowest-agreeing states among the 22 | `GRPO_LA5_I9` .487 and `GRPO_LA5_I10` .544 (next: `GRPO_LA0_I6` .744) | same |
| ✅ `GRPO_LA0` never leaves the normal range on Q1 | .744–.882 across its 11 states | validity.xlsx, `second_judge_agreement` |
| ✅ The collapse is selective but NOT Q1-only (Table 3) | at `GRPO_LA5_I10` vs each instrument's 22-state median: **MITI .333/.678 (−.345, 1/22)**, Q1 .544/.841 (−.297, 2/22), Q2 .590/.754 (−.164, 1/22), MICI .287/.399 (−.112, 4/22); CSQ-8 −.040, PCT −.028, MI-SAT −.025, WAI-SR −.023 | judge_saturation_grpo_data.md panel-c rows |
| ✅ One-sided saturation of the training grader | primary Q1 SD 1.336 → 0.701, Spearman(SD, iteration) **−0.86, p = .001**; variance ratio 0.701²/1.336² = 0.275 (0.285 vs iter 1). Held-out SD: Spearman **+0.44, p = .18**. **AUDIT-FIX:** the series is NOT monotone (1.336, 1.312, 1.189, 1.013, 0.823, **1.029**, 0.887, 0.882, **0.962**, 0.702, 0.701 — up at iterations 5 and 8); the paper says "falls steadily", never "monotonically". ρ/p are printed by no table: recomputed from the 11 SD rows (−0.864, p .0006; +0.436, p .180) by `render_paper_figures.py`, which prints them into the Figure 4 legend | results/lookahead/replication/tables/sd_by_iter.md; `judge_saturation_grpo_data.md` panel-b rows |
| ✅ Arm-level sign preservation, GRPO states only | **1,640 of 1,848 = 88.7%**; 97.0% at \|Δ\|≥0.25; 98.9% at \|Δ\|≥0.50 | results/measurement/validity/tables/multijudge_sign_preservation_grpo.md |
| 📄 Contrast count arithmetic | 8 × C(22,2) = 8 × 231 = 1,848 | derived |
| 📄 Oracle self-repeatability | ICC(2,1) 0.86–0.99 across Q1 / Q2 / MICI, four K=0 anchor states only | results/measurement/validity/tables/oracle_repeatability_icc.md |
| 📄 MITI dependability | `dependability_k1` 0.624 vs **0.91–0.96** for the Likert rubrics (WAI .948, CSQ .945, MI-SAT .955, Q1 .928, Q2 .914). **AUDIT-FIX:** the 2026-09-02 text said 0.91–0.97; 0.97 appears only in `dependability_k2` (two graders) and for PCT (.974, a rate, not a Likert rubric) | results/measurement/validity/tables/multijudge_variance_components.md |

⚠ **Sign preservation is ARM-LEVEL.** ⚠⚠ **Do not write that the held-out grader's variance
GREW** (two-point ratio anchored on the series minimum; trend null). ⚠ **Do not write "only the
rewarded rubric"** — MITI hits its minimum at the same cell. ⚠ **No K=5 state has a repeatability
rep**, so agreement on this arm is raw, not attenuation-corrected.

## Figures 3 and 4 — drawn from tables, not copied (NEW-0904)

| figure | drawn from | what the script recomputes |
|---|---|---|
| Figure 3 `overpraise_judgefree_grpo.png` | results/lookahead/behaviour/tables/behaviour.xlsx sheet `overpraise_judgefree_data`, GRPO rows: `lex_overpraise_marker_rate`, `MICI_OverPraiseRate_gpt-4o-mini`, `MICI_OverPraiseRate_claude-haiku-4-5` by iteration | nothing — plotted as read |
| former Figure 4 `judge_saturation_grpo.png` — **dropped from the paper 2026-09-16**; the same rows now back §7's text only | results/measurement/validity/tables/validity.xlsx sheet `judge_saturation_grpo_data`: panel-a rows (`cross_judge_pearson_r` per state + the 22-state median), panel-b rows (`sd_of_per_conversation_score` per grader) | the Spearman ρ/p (from the 11 SD rows; must equal the §7 text — `render_paper_figures.py::saturation()` prints them) |

Both by [`render_paper_figures.py`](render_paper_figures.py); `sync_figures.py` no longer lists
them. Re-render the EDA → re-run that script → the pictures move with the tables.

## §9 Discussion — regime facts (AUDIT-FIX)

| claim | value | source |
|---|---|---|
| PTO's origin regime | Llama-2-7B therapist; GPT-3.5 as patient AND oracle; V1 (cooperative) patient prompts; **7 iterations** of PTO — so the paper says "a 7B policy, more cooperative simulated patients, a weaker model as patient and judge" and **no longer says "non-iterative"** (the 2026-09-02 text did; Exp1 was iterative) | Exp1_ICLR2025/CLAUDE.md §§ "Setup", "Method" |

## Config facts (Appendix C)

| claim | value | source |
|---|---|---|
| 📄 Base model / precision / LoRA / lr / seed | Llama-3.2-1B, bf16, LoRA r=16 α=16, lr 1e-5, seed 42 | each arm's `run_metadata.json` |
| 📄 Therapist / patient temperature | 0.9 / 0.7 | same |
| 📄 GRPO knobs | G=8, KL β=0.01, temperature 1.2, batch 64 × accumulation 2, eval split 0.05, loss type `grpo` | same |
| 📄 Session / context caps | 49-utterance target, 200 tokens per response, 2,048-token therapist context, MCL=12 | same |

## Limitations (claims that must appear)

| claim | source |
|---|---|
| K ∈ {0, 5} only, by design — no dose–response | results/LIMITATIONS.md |
| One training run per arm; no training-seed replicate | results/LIMITATIONS.md |
| Every endpoint is a single 96-conversation draw; therapist decoding is unseeded (endpoint replicate excepted) | results/LIMITATIONS.md § 5c |
| **NEW** The look-ahead reward is also a reward for continuing (the rollout audit above) | this ledger, §3 rows |
| All 96 personas are used for both training rollouts and eval — every number is in-sample | results/LIMITATIONS.md § 5e |
| Patient simulator and training oracle are the same model; the held-out judge decouples the grader, not the generator | results/LIMITATIONS.md § 2 |
| No human MI-coder validation of any instrument | results/LIMITATIONS.md § 1 |
| Q1+Q2 is both the training reward and a reported outcome | results/LIMITATIONS.md § 3 |
| MITI is the least dependable instrument | results/LIMITATIONS.md § 2 |
| Matched iterations ≠ matched cost — the call/wall-clock disclosure block above | this ledger |

## 2026-09-14 review pass — new and retired numbers (NEW-0914)

**Framing change.** The endpoint-anchored gain ratio ("more than doubles", 2.27× / 2.62×) is no
longer the headline: it is anchored on the K=0 arm's post-decline LAST checkpoint (CLAUDE.md
epistemic rule 2b). The paper now quotes both anchors and the range **1.3 to 2.6×**. §8's
"one-sided saturation" is reframed as **ceiling-driven**: the training oracle's spread tracks its
LEVEL along both arms, and the held-out judge keeps its spread because it has headroom.

| claim | value | source |
|---|---|---|
| ✅ K=0 best-checkpoint gain, Q1+Q2, primary | 3.067 → 4.082 at I8 = **+1.016** (`target=best`, `target_iter` 8) | results/arms/stats/tables/gpt-4o-mini/main_results.md |
| ✅ K=0 best-checkpoint gain, Q1+Q2, held-out | 1.861 → 2.637 at I3 = **+0.776** (`target=best`, `target_iter` 3) | results/arms/stats/tables/claude-haiku-4-5/main_results.md |
| ✅ Gain ratio vs K=0's BEST checkpoint | primary 1.554 / 1.016 = **1.53×**; held-out 1.038 / 0.776 = **1.34×**. Paper range "1.3 to 2.6×" spans these and the endpoint ratios 2.27× / 2.62× | derived — show the arithmetic |
| ✅ Training-oracle Q1 SD along K=0 (the new Figure 4b series) | iters 0–10: 1.296, 1.271, 1.278, 0.974, 0.943, 1.011, 0.921, 0.932, 0.977, 0.982, 1.142; means 3.021, 3.177, 3.302, 3.935, 3.898, 3.869, 3.810, 3.946, 3.935, 3.529, 3.606. Paper: "compresses to 0.92–1.01 while the mean sits near 3.9 over iterations 3–8 and re-expands to 1.14 when the mean falls at iteration 10". Spearman(SD, iter) −0.44, p .18 (printed by `render_paper_figures.py`, not quoted in the text) | results/lookahead/replication/tables/sd_by_iter.md, `gpt-4o-mini / Q1 / GRPO_LA0` rows (= `replication.xlsx` sheet `sd_by_iter`) |
| ✅ Ceiling shares, training oracle, Q1, K=5 (Figure 4c + §8) | `share_ge45` 0.135 (base) → **0.583** (I10) = "14% → 58%"; `share_eq5` **0.396** at I10 = "40% receive the maximum score" | same table, `gpt-4o-mini / Q1 / GRPO_LA5` rows |
| ✅ Held-out judge is nowhere near its ceiling | `share_ge45` = 0.000 at every GRPO state under claude-haiku-4-5 (Figure 4c note); K=5 I10 Q1 mean **2.731** = "near 2.7" | same table, `claude-haiku-4-5 / Q1` rows |
| 📄 Base session length | 28.771 (K=0) / 28.292 (K=5) utterances → "base-policy sessions average 28 utterances" (§4) | results/lookahead/behaviour/tables/length_endpoints.md, iteration-0 columns |
| 📄 Lexical over-praise marker = 10 patterns (Appendix C.4) | `RE_EFFUSIVE`: i'?m so proud · proud of you · inspiration to me · you got this · beautiful · beacon · shining · warrior · hero of your · you are a (light or beacon), case-insensitive | Exp3_PTO_GRPO/eda/eda_analysis/constants.py |
| 📄 Instrument facts (Appendix C.2, Table 6) | Q1 5 items, Q2 17 (`yosef2024assessing`, CLPsych 2024), WAI-SR 12, CSQ-8 8 on a **1–4** scale, MI-SAT 6, MITI 4 globals + 7 counts, PCT 3 globals + 3 counts, MICI 1 global + 6 counts. Reported columns: `Q1_Mean`, `Q2_Mean`, `WAI_TotalMean`, `CSQ8_Mean`, `MI_Mean`, `MITI_GlobalMean`, `PCT_ChangeProp` = CT/(CT+ST), `MICI_Rate` = acts per therapist turn | Exp3_PTO_GRPO/code/questionnaires.py; eda_analysis/constants.py `QUESTIONNAIRES` |
| 📄 Prompts (Appendix C.3) | patient template + the three cooperation clauses, the smoking/obesity clauses, names James/Emma, ages 27/61 (quoted verbatim incl. spelling); therapist = the `Good` counsellor prompt (`only_expert_therapist=True`), name David, applied through the chat template | Exp3_PTO_GRPO/code/system_prompts_builder.py; `_shared/convs.py` |
| 📄 Figure 1 | drawn by `render_schematic.py` (landscape figure*, 6–7 pt labels); the EDA's portrait schematic is no longer copied by `sync_figures.py` | this folder |

**Retired wording (do not reintroduce):** "more than doubles" as a headline or section title
(endpoint-anchored); "which to our knowledge has not been isolated for LLM judges" (replaced by
the ceiling framing); "the direct predecessor of this work" (now "the closest prior work");
"ends more than twice as far from base" in §7 (now "further from base by a margin that survives
the K=0 arm's best checkpoint").

**References added 2026-09-14** (verified against arXiv / ACL Anthology / OpenReview that day):
`kazemnejad2024vineppo` (arXiv 2410.01679) · `gao2024refuel` (ICLR 2025; arXiv 2410.04612) ·
`wang2024patientpsi` (EMNLP 2024, `2024.emnlp-main.711`) · `zhou2025sweetrl` (arXiv 2503.15478).

## 2026-09-14, third pass (Lior's second read) — figures and tables

- **Figures 7 and 8 are now drawn from tables** by `render_paper_figures.py` (`forest()` from
  `behaviour.xlsx` sheets `k_channels_grpo_gpt-4o-mini` + `k_channels_text_grpo`, iteration-10
  rows; `tail_audit()` from `mechanism.xlsx` sheets `tail_audit_by_iter`,
  `tail_score_by_realized_turns`, `tail_within_group`, `GRPO_LA5` rows). ⚠ **Figure 7 is in the
  PAPER's sign (K=5 − K=0): every dz is the sheet's value negated**, so the transposition trap the
  old EDA copy carried is gone. Channels zero in both arms (confront, warn) and the per-session
  duplicates of the per-turn rates are omitted. The script prints the pooled audit numbers
  (ended early 0.182, patient closed 0.156, n 121,088) that the Figure 8 caption quotes.
- **Bold in tables** = the better arm's level per row and grader (Table 1: K=5 in every row,
  MICI being lower-is-better; Table 3: the higher level per row — K=0 at iterations 0 and 3 under
  the training oracle and 0, 1, 3 under the held-out judge, exactly the rows the caption names as
  "nominally ahead"). Tables 4–6 carry no scores to bold.
- Figure 4a's legend moved to the empty lower-left with short labels (it had sat on the K=5 peak
  at iteration 5 when placed at the top).
- **Figure 2 is now drawn from `reward.xlsx` sheet `k_headline_grpo_data`** (`headline()`), the
  same sheet the ledger already cites for its levels and star decisions; the EDA render's legend
  printed at ~5 pt at column width. Same content: mean ± SE, each arm's base dotted, Holm stars,
  endpoint means (3.75 / 4.52 primary, 2.26 / 2.87 held-out).

## 2026-09-14 audits (two independent agents, after Lior's read) — findings and what changed

**Numbers audit** (~530 cells checked against their tables, every derived ratio recomputed): no
wrong cell, no transposed sign, no arithmetic error among the values that carry the argument.
Thirteen framing/provenance findings, all applied:

| finding | fix in the paper |
|---|---|
| "all 121,088 scored K=5 candidates" — 121,088 is the LOGGED subset (`log_coverage` 0.884 pooled); the run scored 136,960 = 17,120 groups × 8 (`compute/cost/tables/api_calls.md`) | Limitations + Figure 8 caption now say "the 121,088 logged candidates (88% of those scored)" |
| "largest at the endpoint" — primary dz peaks at I9 (+0.932 vs +0.905) and held-out Δ peaks at I9 (+0.856 vs +0.616) | §5 now says "at its widest over iterations 9–10" (true in all four columns of Table 3) |
| Appendix B matched-policy deltas were copied in the table's K0−K5 orientation | flipped to the paper's K5−K0 convention and labelled: training oracle −0.015 [−0.059, 0.026], held-out +0.014 [−0.048, 0.075]; the 17/20-bins sentence is consistent with these signs |
| §8 dependability "0.624 vs 0.91–0.96" is a 44-state (four-arm) statistic (`multijudge_variance_components.md`, `n_arms = 44`); no 22-state version exists | numbers dropped from §8; the qualitative claim points at the Limitations. ⚠ Do not quote the dependability figures in this paper unless recomputed on the 22 GRPO states |
| Limitations ICC "0.86–0.99 on the measured K=0 subset" — the 0.86 floor is `PTO_LA0_I10` MICI; GRPO-only anchors (`GRPO_LA0_I8`, `GRPO_LA0_I10`) span 0.924–0.994 | now "ICC 0.92–0.99 on the two K=0 states with a repeat draw" (`oracle_repeatability_icc.md`, GRPO rows) |
| Limitations "four independent draws … 54 same-policy contrasts, none p<.05" — two of the four draws are PTO base states; no ledger row; no primary p-values in any table | replaced by the GRPO base pair from Table 3: dz 0.115 (primary) / 0.043 (held-out), neither significant (ledger row "Base-vs-base noise floor") |
| Table 4 / §8 medians: paper rounded ties half-up (0.841 / 0.891 / 0.921), the cited table half-even (0.842 / 0.890 / 0.920); exact 0.8415 / 0.8905 / 0.9205 | paper now matches the cited table: Q1 0.842 (Δ −0.298), CSQ-8 0.890 (Δ −0.039), WAI-SR 0.920 (Δ −0.022); §8 text 0.842. (Figure 4a no longer prints the value.) |
| "98.9% at |Δ| ≥ 0.50" is not in `multijudge_sign_preservation_grpo.md` | value OK — recomputed from `validity.xlsx` sheet `multijudge_all_pairs_contrasts` restricted to the 22 GRPO states: 450/455 = 98.9% (832/858 = 97.0% at ≥ 0.25). Source recorded here |
| "in the worst case here by nearly 2×" (Appendix C.7) had no row | GRPO_LA5 `iteration_1/iteration_metadata.json` `training_time_s` 14,501 s = 4.03 h vs reconstructed 7.742 h (`compute_by_iteration.md`) = 1.92× |
| K=0 spread "re-expands to 1.14 when the mean falls at iteration 10" — the Q1 mean falls at I9 (3.935 → 3.529, SD 0.982) and the SD jumps at I10 (mean 3.606) | now "re-expands to 1.14 once the mean has fallen to 3.6 over iterations 9–10" |
| over-praise "higher under K=0 from the fifth/fourth iteration" are the Holm-significant iterations; nominally higher from iteration 3 | "significantly higher" |
| base session length: 28.771 / 28.292, cross-arm mean 28.53 | "28–29 utterances" |
| Appendix C.3 misdescribed the low-then-high cooperation clause as "the low clause followed by …" | the clause is now quoted verbatim (`system_prompts_builder.py:93`) |
| run_metadata diff also differs in `started_at` | added "and start timestamp" |
| Figure 4 caption "does not move (0.76 → 0.91)" invites the anchor query (rule 2b) | "shows no trend (0.76 → 0.91, ρ = +0.44, p = .18)" |

**Citations audit** (every key checked against arXiv / ACL Anthology / NeurIPS-ICLR-PMLR proceedings /
Crossref; OpenReview and dblp were bot-blocked and replaced by proceedings pages): no wrong
citation; nine bib corrections and six prose corrections, all applied:

- Bib: full author lists for `ouyang2022instructgpt` (20, now NeurIPS 2022), `shao2024deepseekmath`
  (11), `zheng2023judging` (13), `sharma2024sycophancy` (19, per the camera-ready PDF); the
  `{DeepSeek-AI}` collective author added to `guo2025deepseekr1` (arXiv lists it first; the
  Nature 2025 version, 645:633–638, lists individuals); `panickssery2024selfpreference` → NeurIPS
  2024; `dubois2024lengthcontrolled` → COLM 2024; `yu2023promptmcts` → EMNLP 2023, pp. 7101–7125;
  `chen2025broaden` title casing (`SCOPE`, `Multi-turn`); `pace2024westofn` title → the current
  arXiv v2 ("Synthetic Preferences for Self-Improving Reward Models"); `perezrosas2019goodcounselor`
  pages 926–935; `steenstra2025scaffolding` lost its arXiv `url` (the DOI stays). New keys:
  `levin2000stochastic` (IEEE TSAP 8(1):11–23), `singh2002optimizing` (JAIR 16:105–133),
  `miller2013mi` (3rd edition). `perez2023discovering` keeps "and others" (63 authors).
  `skalse2022defining` keeps the PDF title "Reward Hacking" (the NeurIPS record says "Gaming").
- Prose: "the founding premise of RL for dialogue (Li 2016)" → "as old as RL-based dialogue
  management (Levin 2000; Singh 2002); brought to neural dialogue generation by Li 2016";
  Hong 2023 is offline RL on LLM-imagined conversations (no simulated interlocutor in training);
  SOTOPIA-π is "self-reinforcement", not self-play; "LLM-as-judge adds sycophancy" → judges and
  preference-trained reward models "can reward sycophantic answers" (Sharma / Perez document
  assistant sycophancy and preference-model bias, not judge sycophancy); GDP-Zero (Yu 2023) is
  inference-time planning and "Let's Verify" (Lightman 2024) has no simulated continuations —
  both re-slotted; SCOPE (Chen 2025) is not about counselling — moved to the planning clause;
  "change talk" is 2013-edition vocabulary — the intro now cites `miller2013mi`; "standard
  optimiser" → "most widely used"; the intro no longer says PTO keeps "the best and worst of
  eight candidates" (the PTO paper never fixes its branching factor).
- Unverifiable from the web and left as is: the §9 description of PTO's regime ("a 7B policy,
  more cooperative patients, a weaker judge") — a fact from Exp1's own records, not from the
  cited paper's abstract.

## References added 2026-09-02 (verified against the venue pages)

`guo2025deepseekr1` (arXiv 2501.12948; also Nature 2025) · `zhou2024archer` (ICML 2024, PMLR 235) ·
`shani2024multiturn` (NeurIPS 2024) · `wang2024sotopiapi` (ACL 2024 long, pp. 12912–12940) ·
`hong2023imagined` (arXiv 2311.05584).

## References added 2026-09-04 (verified against the arXiv abstract pages / dblp)

`wei2025multiturn` (arXiv 2505.11821 — multi-turn GRPO/PPO with turn-level credit assignment; ⚠
first author is Quan **Wei**, not Zeng) · `qian2025userrl` (arXiv 2509.19736 — GRPO rollouts
against LLM-simulated users) · `chiu2024bolt` (arXiv 2401.00820 — BOLT, LLM therapists coded
against MI categories; the "advice where a counsellor would reflect" finding) ·
`coste2024ensembles` (ICLR 2024, dblp `conf/iclr/CosteAK024`) · `wu2022annomi` (ICASSP 2022,
pp. 6177–6181). **Considered and NOT added:** Wen et al. 2024 "Language Models Learn to Mislead
Humans via RLHF" — an ICLR 2026 poster disputes its evidence, so it is left out.

## 2026-09-16 restructuring pass (Claude, on Lior's "implement everything") — moves, additions, retirements

**What changed structurally.** Abstract rewritten to end on the result (one caveat sentence);
contributions cut to two, saturation demoted to "we also document"; §3 gained **Algorithm 1**, a
"Why the transfer is not trivial" paragraph (the one-sample Monte Carlo argument, moved out of the
intro), a "Minimum context length" paragraph and a "Cost" paragraph; §4 gained "Why MI" and a
"Terminology" note; §5–§7 got neutral titles; the mechanism section became one paragraph of §8;
§7 (saturation) was halved; the Limitations were consolidated from eleven paragraphs to seven; the
appendices lost their lab-notes sentences. Body ends at the bottom of page 8; 22 pages in all
(the extra page is Appendix D reflowing after the figure pass below).
**No number that carries the argument changed.** Figure 1 was narrowed 0.93→0.82 `\textwidth`.
Figures 2–4 were first narrowed to 0.64–0.68 (which only shrank their type) and then, the same
day, restored to 0.94 with `render_paper_figures.py` made width-aware: each figure is drawn at
its included width, so its point sizes are true page points, and the page budget is met through
the drawn aspect (Figures 2 and 3 at 0.34 / 0.29, close to their original proportions, once the
saturation figure was dropped — see below; the body ends on page 8 with about half a column to
spare) and legends placed inside the axes. Figure 2's legend now lists only the two arms; the
dotted base line and the Holm star are defined in its caption.
The saturation figure (then Figure 4) was then dropped at Lior's request: its three panels
repeated numbers that §7's text states, so nothing left the paper; `saturation()` stays in the
script, un-called by `main()`, for the Spearman / variance-ratio printout that checks those
numbers. Figures renumber: level grids 4–5, channel forest 6, tail audit 7. No value moved.

| claim (new or moved) | value | source |
|---|---|---|
| **NEW** 📄 MCL rationale, §3 "Minimum context length" | quoted **without numbers**: "a pilot on an earlier configuration of this task" in which short-prefix rankings "agreed poorly … and recovered from roughly ten utterances". The Exp2 figures (~0.66–0.73 at `n_turns=2`, 0.8 at ~10) are deliberately not printed — METRICS_REFERENCE §6 forbids citing them as Exp3 outputs | Exp3_PTO_GRPO/eda/results/METRICS_REFERENCE.md § 6 (the ⚠ paragraph) |
| **NEW** 📄 Faithfulness at the shortest admitted prefix (§3 "86–89%", Appendix B.1) | `n_turns=12`: GRPO_LA0 **0.860** [0.848, 0.872], GRPO_LA5 **0.886** [0.875, 0.895] (primary grader, iters 1–10 pooled); `n_turns=50`: **0.897** [0.869, 0.922] / **0.936** [0.904, 0.960]. Paper: "orders candidates as the full-session score does in 86–89% of pairs … agreement does not fall with prefix length". ⚠ This is a pooled-iterations, primary-grader statistic; the held-out columns are not quoted | results/lookahead/mechanism/tables/faithfulness_curve.md, rows 12 and 50, GRPO columns (= `arms/training/tables/gpt-4o-mini/reward_reliability_by_nturns.md`) |
| ✅ PCT contrast now quoted in §6 (was Table 1 only) | K=5 higher: primary **+0.111** (dz 0.516), held-out **+0.113** (dz 0.563), both p_holm .000 | results/lookahead/reward/tables/k_endpoints.md, PCT row of `GRPO_LA5_I10 − GRPO_LA0_I10` (same row Table 1 prints) |
| **NEW** citation fact, §6 — MITI 4.2.1 Affirm definition | manual: "Affirm should not be coded automatically for the clinician's agreeing with, approval of, cheerleading for, or non-specific praising of the client"; example list: "I am really proud of you. (Not coded; not specific)"; "You did great! (Not coded)". Paper: "the MITI manual codes an affirmation only when it is specific to a client strength or effort, excludes 'cheerleading' and non-specific praise, and lists 'I am really proud of you' as not coded" | Moyers et al. 2015, MITI 4.2.1 manual (casaa.unm.edu/assets/docs/miti4_2.pdf), Affirm (AF) section — verified 2026-09-16 |
| **NEW** citation fact, §6 + §8 — "righting reflex" | the counsellor's urge to correct/persuade; Miller & Rollnick's term (the MITI 4.2.1 manual also uses it under the Partnership global) | `miller2013mi`; MITI 4.2.1 Partnership section — verified 2026-09-16 |
| **NEW** wording, §4 "Why MI" | MITI globals "cultivating change talk" and "softening sustain talk" are named as trajectory-level ratings; change talk / sustain talk defined | `moyers2016miti`; the four globals are listed in Appendix C.2 (unchanged) |

**Numbers that left the BODY but remain in an appendix** (nothing left the paper):
held-out faithfulness 0.800 vs 0.747 (Appendix B.1); update-direction cosine 0.804 / 0.851 /
ceiling 0.945 (Appendix B.4); the rollout audit detail — 121,088 logged (88%), early-ending range
12% (iter 10) to 30% (iter 6), 16% patient closed, argmax 0.10 vs 0.125 (Appendix A intro text +
the tail-audit figure's caption; the Limitations keep 82%, "up to 0.09", "less often than
chance"); the SD endpoints (1.34→0.70, 0.76→0.91) and the Spearman ρ/p are in §7's text (the
saturation figure that also showed them was dropped later the same day); §6 no longer says turn
length tripled (the Limitations do).

**Retired wording (do not reintroduce):** "flattery"/"flatter" (→ over-praise / unearned
affirmation; the word survives only in Appendix D transcripts); "Turn-level reward teaches
flattery", "Why it works, as far as we can tell", "Where the training grader stops discriminating",
"Look-ahead leads on every instrument and keeps the lead" (section titles → neutral); "The move is
not obviously safe"; "checkable rather than asserted"; "the defensible sentence"; "what changed is
the ruler"; "did not make the policy honest … dishonest strategy"; "What would change our minds";
"the limitation a reader should weight most heavily"; Appendix B "the shape of no effect" / "a
result we can omit while quoting the pooled row"; Appendix C "the single easiest way to get these
numbers wrong". Also retired: "so the gain is not an artifact of the optimisation target" and
"separates a genuine gain from a target-specific one" (→ "consistent with a gain that is not
specific to the optimisation target" — a larger held-out dz supports, it does not prove).

**Qualifier added (ledger warning "the steelman is a Q1+Q2 statement"):** abstract and §1 now say
"beats the K=0 policy's best checkpoint **on the rewarded rubric**". The turn = utterance
definition now appears in §1 ("K=5 appends patient, therapist, patient, therapist, patient").

**Label rename:** `sec:behaviour-honest` → `sec:behaviour-heldout` (referenced from §7 and Ethics).
`sections/07_mechanism.tex` was retired (no longer `\input`) and deleted the same day.

## 2026-09-17 clean-up pass (Claude, on Lior's "clean it up for me and my supervisors") — no number changed

**Structure.** Section files renumbered contiguously (`07_measurement`, `08_discussion`,
`09_limitations`, `10_ethics`); the paper's section NUMBERS are unchanged (§7 saturation, §8
discussion), so every row above still resolves. `main.tex` lost its fallback branch, draft macros
and unused packages; `refs.bib` was regrouped by topic with its entries verbatim. Body still ends
at the bottom of page 8; 21 pages.

| claim (new) | value | source |
|---|---|---|
| **NEW** ✅ CONFIG FACT — policy updates per sampled batch (Table 5 row; §3 "The update") | `grpo_inner_iterations: 1` in BOTH arms (= TRL `GRPOConfig.num_iterations=1`) | `run_metadata.json` of `GRPO_Iterative_Q1Q2_Llama32-1B_LA{0,5}_MCL12_G8`, key `grpo_inner_iterations`; wired in `code/GRPO_Exp3/grpo_trainer.py` (`num_iterations=cfg.grpo_inner_iterations`) — verified 2026-09-17 |
| **NEW** code fact — "ρ = 1 at the update, the clipping is inactive" (§3) | with `num_iterations == 1` and `steps_per_generation <= gradient_accumulation_steps` (both arms: grad accum 2, `steps_per_generation` at its default = grad accum) TRL sets `old_per_token_logps = per_token_logps.detach()`, so the importance ratio is identically 1 and the min/clip is a no-op; the gradient is the token-averaged Σ_g A_g ∇log π(t_g|c) minus β ∇KL | trl 1.4.0 `trainer/grpo_trainer.py` (the comment + code just above `coef_1 = torch.exp(log_importance_weights)`), `requirements.txt` pin — verified 2026-09-17 |
| **NEW** citation fact — Eq. 2 is the DeepSeekMath objective with TRL's `grpo` normalisation | per-sequence mean over tokens, then mean over the batch; KL estimated per token to the reference, which is π_n here (the iteration-start adapter) | Shao et al. 2024 eq. (3); trl 1.4.0 `grpo_trainer.py`: `if self.loss_type in ["grpo", "sapo"]: loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)…).mean()` |
| **NEW** code fact — the look-ahead rollout uses the LIVE policy π, not the frozen π_n (§3 notation + "The look-ahead reward", Algorithm 1 line 4, Figure 1 labels) | the reward functions are built with `therapist_model=policy`, the same object `GRPOTrainer(model=policy)` trains in place (already a PEFT model, so no wrapping copy); the abstract and §1 already said "the current policy" | `grpo_trainer.py` (`therapist_model=policy` in the reward-function factory; `GRPOTrainer(model=policy, …)`; `patch_generate(trainer.model, …)`) — verified 2026-09-17 |

**Retired wording (do not reintroduce):** "PPO-clipped step" (→ "one step on Eq. 2" in
Algorithm 1 and "one step on the group-standardised scores" in the loop paragraph; the clip is
formally in Eq. 2 and stated inactive). Figure 1's update box no longer reads "Σ A_g ∇log π + β KL"
(a gradient plus a penalty) but "maximising Σ_g A_g log π(t_g | c) − β KL(π ‖ π_n)"; its rollout
nodes read π, not π_n.

## 2026-09-17 (b) submission-readiness pass — new rows

| claim | value | source |
|---|---|---|
| **NEW** 📄 Optimizer steps per arm over the ten training iterations (Table 5 last row; Limitations "matched by construction"; §3 no longer claims equal step counts) | GRPO_LA0 **1,128** (per iteration 108, 94, 118, 100, 118, 116, 128, 108, 80, 158; range 80–158); GRPO_LA5 **1,070** (108, 104, 112, 106, 88, 70, 106, 110, 130, 136; range 70–136). Ratio 1,070/1,128 = 0.949, consistent with the oracle-call ratio 289,983/302,541 = 0.958 | results/compute/cost/tables/compute_by_iteration.md, `n_steps` (= per-step `training/completions/*.parquet` count), GRPO rows, iterations 1–10; summed 2026-09-17 |
| **NEW** 📄 GPU-hours per run (Limitations; Ethics already had the ≈79 total) | GRPO_LA0 **27.906** → "27.9"; GRPO_LA5 **51.205** → "51.2"; sum 79.111 ≈ 79 | results/compute/cost/tables/compute_by_arm.md, `total_gpu_h` |
| **NEW** 📄 Iso-compute reading, ~13 GPU-h (Limitations) | budget 13.27 GPU-h, select+eval on Q1Q2: primary LA5_I2 vs LA0_I4, Δ −0.569, **dz −0.742**, p_holm .000; held-out LA5_I2 vs LA0_I3, Δ −0.495, **dz −0.780**, p_holm .000. Paper: "dz −0.74 under the training oracle, −0.78 held out" (K=5 minus K=0; the table's `dz` column is already in that direction, `dz_K0_minus_K5` is the flipped one) | results/compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md and …_claude-haiku-4-5.md, `budget_gpu_h = 13.270`, `select_metric = eval_metric = Q1Q2` |
| **NEW** 📄 Iso-compute reading, ~23 GPU-h (Limitations) | budget 23.21 GPU-h: primary LA5_I4 (23.21 h) vs LA0_I8 (22.28 h), Δ +0.038, **dz 0.074**, p .789 → "level, n.s."; held-out LA5_I4 vs LA0_I3 (8.21 h; K=0's held-out best), Δ +0.147, **dz 0.331**, **p_holm .012** | same tables, `budget_gpu_h = 23.210` |
| **NEW** 📄 "beyond the turn-level arm's total budget the comparison is the best-checkpoint one of §5" | K=0's total is 27.9 GPU-h, so at every budget ≥ 27.9 its best-within-budget state is its overall best under the selecting grader (I8 primary / I3 held-out) — exactly the §5 steelman rows (at 51.2 GPU-h: LA5_I10 vs LA0_I8, +0.435, dz 0.743, matching §5's "+0.435 / dz 0.743") | same tables, rows `budget_gpu_h ≥ 30.53`; §5 steelman row in this ledger |
| **NEW** CONFIG FACT — software versions (Table 5) | TRL 1.4.0, transformers 5.8.1, PEFT 0.19.1 (also accelerate 1.13.0, datasets 4.8.5, openai 2.36.0, anthropic 0.116.0 — not printed) | `requirements.txt` (repo root), the pin both Colab install cells use |
| **NEW** CONFIG FACT — hardware (Table 5) | one NVIDIA A100 on Google Colab (memory size deliberately not printed: the Exp3 runs were tuned for "A100 Colab" and the card size is not recorded per run) | CLAUDE.md § Exp3 "Throughput config (tuned for A100 Colab)"; Ethics Statement already said "a single cloud A100-class GPU" |

**Retired wording:** "prompts and number of gradient steps are identical across $K$" (false; → "the
loss, the advantage normalisation, the KL penalty and the prompt-construction rule are identical
across $K$"); "nothing here compares the arms at matched cost" (→ the iso-compute passage above).

## References added 2026-09-17 (verified against the arXiv abstract pages / ACL Anthology that day)

Added after two web searches for 2025–2026 work a reviewer would expect (multi-turn GRPO with
simulated users and next-turn / future-turn credit; RL-trained MI and clinical dialogue agents;
faithfulness of LLM-simulated MI patients). Each entry's title, authors, date and venue were read
off the page named in the source column; the one-line characterisation in §2 / Limitations is
taken from the abstract only.

| key | what the paper says it does (from its abstract) | where cited | source |
|---|---|---|---|
| `zhao2026faca` | FACA: in interactive GRPO, the next user turn is "noisy, temporally local evidence about the preceding" assistant segment; a locally normalised "reaction advantage" is added to the terminal-outcome advantage "without an extra critic or rollout" | §2 multi-turn ("treat the next user turn of an interactive GRPO rollout as local credit evidence for the preceding assistant segment") | arxiv.org/abs/2608.17499 (submitted 2026-08-18; no venue) |
| `peng2026atgrpo` | AT-GRPO: dialogue trajectories as trees; "each node … aggregate[s] rewards from a stage-aware range" of future turns; two-agent game with a user agent | §2 multi-turn ("aggregate rewards over a stage-dependent window of future turns in a tree-structured GRPO against a user agent") | arxiv.org/abs/2602.08533 (v2 2026-02-10) |
| `li2026patr` | PATR: process-scorer-guided adaptive tree rollout for multi-turn agent RL; "uses task-appropriate process feedback to score partial trajectories, selectively branches from promising states" (FrozenLake, SWE-Bench) | §2 look-ahead/search ("let process scores decide where a multi-turn rollout tree branches") | arxiv.org/abs/2607.15610 (submitted 2026-07-17; preprint) |
| `yang2026mithinker` | MIThinker: "two-stage training combining supervised fine-tuning and reinforcement learning" of a thinker for MI counselling agents | §2 MI ("supervised and RL stages for MI") | aclanthology.org/2026.findings-acl.163 — Findings of ACL 2026, pp. 3292–3328, DOI 10.18653/v1/2026.findings-acl.163 |
| `lievin2026residencyrl` | ResidencyRL: multi-turn RL "through simulated multi-turn clinical encounters (up to 60 dialogue turns …)" against "LLM simulators capable of complex, adversarial behaviors" with a structured reward; 35 authors → first eight + "and others" | §2 MI ("multi-turn RL for clinical encounters") | arxiv.org/abs/2608.07418 (submitted 2026-08-07) |
| `hoang2026standardised` | LLM-simulated MI patients vs human patients "given identical profiles": "semantically similar content … their modes of expression differ substantially"; "human patients exhibit a mix of positive and negative responses, LLM patients skew toward uniformly [positive] ones" | Limitations "Simulation only" ("express themselves more uniformly positively than human patients do … bears directly on a finding about praise") | aclanthology.org/2026.clpsych-1.21 — CLPsych 2026, pp. 258–270, DOI 10.18653/v1/2026.clpsych-1.21 |

§2 was rewritten around these (the multi-turn paragraph now ends on the two nearest works and a
one-sentence placement of look-ahead; the reward-hacking paragraph was tightened by two lines to
pay for it). No number changed.

## Reference currency pass, 2026-09-17 (web-verifying agent + spot checks by hand)

Scope: every arXiv-cited entry checked for a since-published version; canonical DOI/URL sought for
the rest. **Applied only what was read off the venue's own page** (Nature, PMLR, proceedings.iclr.cc,
ACL Anthology, JAIR) or the publisher's Crossref deposit (IEEE, ACM). OpenReview and dblp were
behind bot walls all day, so nothing rests on them.

| entry | change | evidence |
|---|---|---|
| `guo2025deepseekr1` | arXiv → **Nature 645(8081):633–638, 2025**, DOI 10.1038/s41586-025-09422-z; printed title "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning"; Nature lists 194 individual authors and no "DeepSeek-AI" collective, so the author line is now "Guo, Daya … Bi, Xiao and others". In-text citation becomes (Guo et al., 2025) | nature.com/articles/s41586-025-09422-z + Crossref |
| `kazemnejad2024vineppo` | arXiv → **ICML 2025, PMLR 267:29557–29590**; retitled on the venue page "VinePPO: Refining Credit Assignment in RL Training of LLMs". Key kept (in-text now 2025) | proceedings.mlr.press/v267/kazemnejad25a.html |
| `yuan2024selfrewarding` (uncited) | arXiv → **ICML 2024, PMLR 235:57905–57923**; the bib had omitted the 4th author, Xian Li | proceedings.mlr.press/v235/yuan24d.html |
| `yuan2024eurus` (uncited) | arXiv → **ICLR 2025**; the bib's author list had skipped Boji Shan and Zeyuan Liu before Jia Deng | proceedings.iclr.cc 2025 hash 3e2c12c1… |
| `yosef2024assessing` | + pages 1–11, address, DOI 10.18653/v1/2024.clpsych-1.1 (author order kept as "Brunstein Klomek", her actual name; the Anthology record inverts it) | aclanthology.org/2024.clpsych-1.1 |
| `perezrosas2019goodcounselor` | + publisher, address, DOI 10.18653/v1/P19-1088, URL | aclanthology.org/P19-1088 |
| `wang2024patientpsi` | + pages 12772–12797, publisher, address, DOI 10.18653/v1/2024.emnlp-main.711, URL | aclanthology.org/2024.emnlp-main.711 |
| `li2016deeprl` | + pages 1192–1202, publisher, address, DOI 10.18653/v1/D16-1127, URL | aclanthology.org/D16-1127 |
| `wu2022annomi` | + publisher IEEE, DOI 10.1109/ICASSP43922.2022.9746035 (booktitle wording left as is: Xplore itself was blocked) | IEEE Crossref deposit |
| `steenstra2025scaffolding` | + pages 1–22, address | ACM Crossref deposit |
| `levin2000stochastic` | + DOI 10.1109/89.817450 | IEEE Crossref deposit |
| `singh2002optimizing` | + DOI 10.1613/jair.859, URL | jair.org |
| `moyers2016miti` | + URL casaa.unm.edu/assets/docs/miti4_21.pdf — the PDF was downloaded and its first page read: "Motivational Interviewing Treatment Integrity Coding Manual 4.2.1 … Revised June 2015", Moyers, Manuel & Ernst. (The 2026-09-16 ledger row named `miti4_2.pdf`; `miti4_21.pdf` is the file that resolves today.) Year stays 2015 | the PDF itself |
| `yang2026mithinker`, `hoang2026standardised` | + DOIs (from the Anthology records read earlier today) | aclanthology.org |
| `baruch2025pto` | venue string **confirmed** (SSI-FM workshop; listed without an Oral tag on the workshop's accepted-papers page). No URL added: the OpenReview forum returned a bot challenge, so the id could not be opened | sites.google.com/berkeley.edu/selfimprovingfoundationmodels/accepted-papers; iclr.cc/virtual/2025/workshop/23971 |
| `chiu2024bolt`, `hong2023imagined`, `zhou2025sweetrl`, `wei2025multiturn`, `qian2025userrl`, `pace2024westofn`, `xie2024mctsdpo`, `shao2024deepseekmath`, `schulman2017ppo`, `guo2024oaif` | **no change** — still arXiv-only as far as the arXiv record (no journal-ref / acceptance comment), the iclr.cc / neurips.cc virtual sites, ML Anthology and the ACL 2026 / EMNLP 2026 accepted lists show. `wei2025multiturn` matches arXiv v3 (2026-08-21) exactly; its ICLR 2026 and `qian2025userrl`'s ICLR 2026 submissions could not be read on OpenReview | arXiv abstract pages |
| 24 entries **not re-verified** (zhou2024archer, wang2024sotopiapi, shani2024multiturn, gao2024refuel, christiano2017deeprl, ouyang2022instructgpt, rafailov2023dpo, lightman2024letsverify, yu2023promptmcts, chen2025broaden, amodei2016concrete, skalse2022defining, pan2022effects, gao2023scaling, coste2024ensembles, zheng2023judging, sharma2024sycophancy, perez2023discovering, panickssery2024selfpreference, singhal2024long, dubois2024lengthcontrolled, hatcher2006waisr, larsen1979csq, koo2016icc, shrout1979icc) | none — the agent's parallel checks for these had not returned. Each was verified against its venue page when added (2026-08-18 / 09-04 / 09-14 blocks above); none is an arXiv preprint, so no currency change is expected | — |
