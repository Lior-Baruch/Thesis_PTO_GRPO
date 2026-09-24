# NUMBERS.md — the claims ledger

Every quantitative claim in the draft → the exact tracked artifact it came from. `results/…` paths
are relative to `Exp3_PTO_GRPO/eda/`. **Nothing enters the `.tex` that is not a row here**, and no
row is written from prose — each was read off the named table or recomputed from the score lake.

*(2026-09-22: Method and Setup merged into one §3 on Doron's note — §3.1 task/simulator/oracle,
§3.2 GRPO with look-ahead, §3.3 evaluation design; reward is now §4, therapist §5, patient §6,
discussion §7. Rows below still say "§4" for the setup and "§5"–"§8" for what follows; read them
one lower. Appendices unchanged.)*

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

### Addendum, same day: the agent's final report covered all 51 entries

The "24 entries not re-verified" row above is superseded: the parallel checks returned and every
entry is now verified on its venue's page or the publisher's Crossref deposit. Applied:

| entry | change | evidence |
|---|---|---|
| `wang2024sotopiapi` | **author order corrected** to … Sap, **Bisk, Neubig**, Zhu (the bib had Neubig before Bisk); + DOI 10.18653/v1/2024.acl-long.698, URL | aclanthology.org/2024.acl-long.698 (.bib) |
| `rafailov2023dpo` | **author order corrected** to … Mitchell, **Manning, Ermon**, Finn (the bib had Ermon before Manning); + pages 53728–53741, DOI 10.52202/075280-2338, URL | proceedings.neurips.cc 2023 bibtex endpoint |
| `perez2023discovering` | booktitle → "Findings of the Association for Computational Linguistics: ACL 2023" (year was missing); + pages 13387–13434, DOI, URL; author list stays "and others" (63 authors) | aclanthology.org/2023.findings-acl.847 |
| `gao2023scaling` | booktitle → "Proceedings of the 40th ICML", PMLR 202:10835–10866, URL | proceedings.mlr.press/v202/gao23h.html |
| `zhou2024archer` | + PMLR 235:62178–62209, URL | proceedings.mlr.press/v235/zhou24t.html |
| `shani2024multiturn` | + pages 118953–118993, DOI 10.52202/079017-3779, URL. Title kept as "from Preference Human Feedback": the camera-ready PDF and arXiv print "from"; only the proceedings metadata prints "with" | proceedings.neurips.cc 2024 + camera-ready PDF |
| `ouyang2022instructgpt`, `skalse2022defining`, `zheng2023judging`, `panickssery2024selfpreference` | + pages, DOI (10.52202/…), proceedings URL; `christiano2017deeprl` + URL only (the official NIPS 2017 bib has no pages) | proceedings.neurips.cc bibtex endpoints |
| `lightman2024letsverify`, `gao2024refuel`, `chen2025broaden`, `sharma2024sycophancy` | + proceedings.iclr.cc pages and the OpenReview URL whose id the iclr.cc poster page carries; `coste2024ensembles` + pages only (its OpenReview id came from a third-party mirror and was not added); `pan2022effects` + OpenReview URL (iclr.cc poster page). Lightman's author spellings kept as arXiv/bib (Yura Burda, Harri Edwards); ICLR prints the OpenReview profile forms | iclr.cc/virtual poster pages; proceedings.iclr.cc bibtex |
| `singhal2024long`, `dubois2024lengthcontrolled` | + OpenReview URL from colmweb.org's accepted-papers list; Dubois title/author line kept (colmweb lists 3 authors and a different subtitle; arXiv v2 and the author's page give the bib's form — unresolved, flagged) | colmweb.org/2024/AcceptedPapers.html |
| `yu2023promptmcts` | + DOI 10.18653/v1/2023.emnlp-main.439, URL | aclanthology.org |
| `hatcher2006waisr`, `larsen1979csq`, `koo2016icc`, `shrout1979icc` | + DOIs (10.1080/10503300500352500; 10.1016/0149-7189(79)90094-6; 10.1016/j.jcm.2016.02.012; 10.1037/0033-2909.86.2.420) | publishers' Crossref deposits |
| every arXiv-only entry | + `url = https://arxiv.org/abs/<id>` derived from the journal field, so all entries but the two books, the manual and the PTO workshop paper now carry a link | — |
| `baruch2025pto` | OpenReview id fTVhWlzCuk reported (via the workshop's ML Anthology mirror) but **not added**: OpenReview could not be opened. Lior can paste the forum URL from his own account | — |
| `miller1991mi`, `miller2013mi`, `chiu2024bolt`, `hong2023imagined` (NeurIPS 2023 FMDM workshop only), `zhou2025sweetrl`, `wei2025multiturn`, `qian2025userrl`, `guo2024oaif`, `pace2024westofn` (ICLR 2024 DPFM workshop only), `xie2024mctsdpo` (NeurIPS 2024 System-2 workshop only), `shao2024deepseekmath`, `schulman2017ppo`, `amodei2016concrete` | no venue change | LOC records; arXiv abstract pages; iclr.cc / neurips.cc workshop pages; ACL 2026 + EMNLP 2026 lists |

Not applied by choice: `editor` lists (acl_natbib prints them and they would double the length of
a dozen entries); `volume={2024}`-style ICLR volumes; the "(ACL)/(EMNLP)" suffix removal the
Anthology's own strings would imply (a style choice, kept as is).

## 2026-09-17 (c) — the MI-process refactor (Claude, on Lior's "the story is GRPO with look-ahead in MI and a deep analysis")

**Structure.** §6 *What look-ahead teaches the therapist* (was *Behavioural analysis*) + a new
§7 *What the therapist's turns do to the patient*; the saturation section moved whole to
**Appendix E** (`E_saturation.tex`, `\label{app:saturation}`; its Table 4 is now Table 7 there;
no number changed); §8 keeps the horizon paragraph with the mechanism paragraph folded in, the
"Scope" paragraph moved to the Limitations as *One optimiser, one regime*; the method's *Cost*
paragraph folded into *The look-ahead reward*. New **Figure 3** (`process_grpo.png`, body) and
**Table 3** (process endpoint, body); the judge-free marker figure is now single-panel and in
Appendix A (**Figure 4**); new appendix figures **5** (`process_grpo_heldout.png`), **6**
(`responsiveness_grpo.png`), **7** (`text_grpo.png`). Body ends at the bottom of page 8; 25 pages.
Every table below is under `Exp3_PTO_GRPO/eda/results/lookahead/{process,text}/tables/` (family
notebooks `lookahead/process.ipynb`, `lookahead/text.ipynb`; rendered 2026-09-17 on the four-arm
grid, GRPO rows quoted). ⚠ `k_process_paired` / `k_text_paired` store **K=0 − K=5**; every
contrast below is in the PAPER's sign (K=5 − K=0); levels are levels.

**The coder (config facts).** `questionnaires.py` id 10; therapist codes OQ CQ SR CR AF PRA GI
PERS SEEK CONF OTH, patient codes CT ST NEU; transcript numbered `[THERAPIST #k]`/`[PATIENT #k]`;
arrays pinned to the utterance counts; opener pinned to OQ and excluded from every rate; both
graders; 2 × 11 × 96 = **2,112** conversations per grader, complete (three held-out batch rows
came back one code short and were re-scored on the live path). Derived quantities are recomputed
in `eda_analysis/process.py::conversation_metrics` from the stored code strings.

| claim | direction | value | source |
|---|---|---|---|
| 📄 Code shares at base (iteration 0), primary / held-out | levels | GRPO_LA0 base: OQ 0.096 / 0.182, CQ 0.050 / 0.273, SR 0.127 / 0.005, CR 0.016 / 0.042, AF 0.033 / 0.023, PRA 0.031 / 0.036, GI 0.394 / 0.191, PERS 0.198 / 0.151; GRPO_LA5 base: OQ 0.078 / 0.169, CR 0.020 / 0.020, PRA 0.051 / 0.041, PERS 0.208 / 0.180 | `process.xlsx::process_levels_<judge>`, `th_<CODE>_rate`, iteration 0 |
| 📄 **K=0 learns non-specific praise** — `PRA` share at iteration 10 | level | **0.407** primary / **0.762** held-out (base 0.031 / 0.036) | `process_levels_<judge>`, GRPO_LA0 iteration 10 |
| 📄 **K=5 learns complex reflections** — `CR` share at iteration 10 | level | **0.230** / **0.242** (base 0.020 / 0.020); K=0 endpoint 0.020 / 0.016 | same, GRPO_LA5 / GRPO_LA0 |
| 📄 Open-question turns at the endpoints | level | K=0 0.000 / 0.000; K=5 0.011 / 0.032 ("0.00–0.03") | same |
| 📄 Persuasion under K=5 | level | 0.208 → 0.279 primary (paper: 0.21 → 0.28); held-out 0.180 → 0.268 | same |
| 📄 MI-adherent share (OQ+SR+CR+AF+SEEK) at iteration 10 | level | K=5 0.438 / 0.360 vs K=0 0.290 / 0.037 | same, `mi_adherent_rate` |
| 📄 Held-out judge's praise share for K=5 at iteration 10 | level | PRA 0.203 (paper "0.20"; primary 0.053, "0.05") | `process_levels_claude-haiku-4-5`, GRPO_LA5 |
| 📄 **Table 3, therapist block** (iteration 10, persona-paired dz, K5−K0) primary \| held-out | K=5 − K=0 | PRA 0.407→0.053 dz −1.19 \| 0.762→0.203 dz −2.19 · CR 0.020→0.230 +0.90 \| 0.016→0.242 +0.94 · PERS 0.055→0.279 +0.79 \| 0.033→0.268 +0.84 · MI-adherent 0.290→0.438 +0.40 (p_holm .0044) \| 0.037→0.360 +1.35 · MI-inconsistent 0.463→0.332 −0.37 (p_holm .020) \| 0.795→0.471 −1.08; all others p_holm < 1e-6 | `process.xlsx::k_process_paired`, `method = GRPO`, `iteration = 10`, columns `mean_K0, mean_K5, dz, p_holm` — **dz and delta negated** |
| 📄 Table 3, responsiveness block | K=5 − K=0 | refl_after_ct 0.003→0.264 dz +0.99 (n 76) \| 0.006→0.258 +0.99 (n 80) · pra_after_st 0.315→0.015 dz −1.10 (n 65) \| 0.723→0.011 −3.10 (n 64) | same rows; `n` = conversations where the condition occurred |
| 📄 Table 3, patient row + the "any change talk" sentence | K=5 − K=0 | ct_prop 0.528→0.683 dz +0.57 \| 0.545→0.721 +0.70; reached_ct 0.865→0.917 dz +0.13 (p_holm 1.0) \| 0.885→0.948 +0.18 (p_holm .75) — quoted in §7 as "0.92 vs 0.87, n.s." | same |
| 📄 Not in Table 3 (space); in the text / figures only | K=5 − K=0 | AF 0.196→0.147 dz −0.18 (n.s.) \| 0.009→0.067 +0.56; OQ 0.000→0.011 +0.33 (p_holm .027) \| 0.000→0.032 +0.57 | same |
| 📄 **Trend rows** (which of the ten matched iterations clear Holm) primary \| held-out | direction as named | PRA higher under K=0 at 8, 10 \| 5, 6, 8, 9, 10 · CR higher under K=5 at 4, 5, 7–10 \| 6–10 (paper: "six / five") · PERS higher under K=5 at 2, 6–10 \| 3–6, 8–10 ("six / seven") · refl_after_ct higher under K=5 at 1, 6–10 \| 6–10 ("six / five") · pra_after_st higher under K=0 at 8–10 \| 5–10 · ct_prop higher under K=5 at 6–10 \| 4–10 ("sixth / fourth on") · mi_incons_rate: primary K=0 higher at 10, K=5 higher at 2, 6, 9; **held-out K=0 higher at 8, 10 and K=5 higher at 3, 5** (the "along the run … mixed" sentence) | `process.xlsx::k_process_summary`, `method = GRPO`, `iters_sig_K0_higher` / `iters_sig_K5_higher` |
| 📄 **Yields at iteration 10** (Figure 3c / 5c, §7) | P(CT next) | CR: K=5 **0.854** (n 389) / **0.889** (n 431); K=0 0.095 (n 21) / 0.087 (n 23) · PRA: K=0 0.582 (n 467) / 0.486 (n 813); K=5 0.758 (n 66) / 0.973 (n 299) · GI: K=0 0.539 / 0.709 ("no better than its plain information-giving turns": 0.582 vs 0.539 primary, 0.486 vs 0.709 held-out) · AF: K=0 0.729 (n 170) / [n 12, not quoted], K=5 0.982 / 0.949 · PERS: K=0 0.346 / 0.406, K=5 0.528 / 0.372 | `process.xlsx::yield_<judge>`, GRPO rows, `iteration = 10`, `p_ct`, `n`; Wilson bounds `p_ct_lo/hi` |
| 📄 Praise yield at base | P(CT next) | GRPO_LA0 base PRA 0.971 (n 34) primary / 0.722 (n 36) held-out ("97% / 72%") | `yield_<judge>`, iteration 0 |
| 📄 Responsiveness at base | level | refl_after_ct: GRPO_LA0 0.122 / 0.036, GRPO_LA5 0.153 / 0.023 ("4–15%"); pra_after_st ≤ 0.032 at every base ("at most 3%") | `process_levels_<judge>`, iteration 0 |
| 📄 **Within-session change talk** (Figure 3d / 5d) | CT share by patient-turn bin | iteration 10, primary: K=5 0.177 / 0.538 / 0.796 / **0.846**; K=0 0.167 / 0.472 / **0.680** / **0.496**. Held-out: K=5 0.323 / 0.583 / 0.790 / **0.876**; K=0 0.328 / 0.479 / **0.623** / **0.499**. Bins 1-2 / 3-5 / 6-9 / 10+; the 10+ bin is reached by 77% (K=5) / 65% (K=0) of conversations | `process.xlsx::ct_trajectory_<judge>`, GRPO rows, `ct_prop`, `share_convs_reaching` |
| 📄 Base 10+ bin (the "below the base" claim) | CT share | primary 0.650 (LA0 base) / 0.516 (LA5 base); held-out 0.655 / 0.503 — K=0's endpoint 0.496 / 0.499 is below both | same, iteration 0 |
| 📄 **Parity of the coder with the same grader's instruments** (Appendix C.3, Limitations) | pooled within-state Spearman ρ, GRPO states only | primary: PCT CT **0.881**, ST **0.898**; MITI B1_GI 0.426, B2_Persuade 0.315, B3_Q 0.193, B4_SR 0.281, B5_CR 0.216, B6_AF 0.205, B7_Seek 0.019 ("0.02–0.43"). Held-out: CT **0.915**, ST **0.942**; B3_Q 0.731, B1_GI 0.677, B2_Persuade 0.550, B4_SR 0.196, B5_CR 0.303, B6_AF 0.215, B7_Seek 0.101 ("0.10–0.30"). MITI mean questions 5.469 vs coder 1.723 per conversation, primary ("5.5 vs 1.7") | derived: Fisher-z, n-weighted pool of the 22 GRPO rows of `process.xlsx::parity_<judge>` (the rendered `parity_pooled_<judge>` pools all 44 states — do not quote that one here) |
| 📄 Embedding drift (§6, Figure 7a) | cosine between the two GRPO arms' displacement vectors | iteration 10 **0.438**; iterations 3–8: 0.714, 0.642, 0.578, 0.328, 0.469, 0.585 ("0.33–0.71"); iteration 9 −0.053 | `text.xlsx::drift_cosines`, `cos_K0_K5_GRPO` |
| 📄 Persona variance share (§6, Figure 7b) | level | GRPO_LA0 0.374 → 0.192; GRPO_LA5 0.410 → 0.302 ("0.37 → 0.19", "0.41 → 0.30") | `text.xlsx::diversity_by_state`, `persona_var_share`, iterations 0 and 10 |
| 📄 Template similarity (Figure 7c) | level | GRPO_LA0 0.265 → 0.585; GRPO_LA5 0.262 → 0.494 | same, `template_sim` |
| 📄 Patient-side text contrasts at iteration 10 (§7) | K=5 − K=0 | pt_turn_len 439.7 → 600.3 chars, dz +1.24 ("600 vs 440"); pt_disengage_rate 0.213 → 0.276, dz +0.31 (p_holm .026) | `text.xlsx::k_text_paired`, `method = GRPO`, `iteration = 10` — **negated** |
| 📄 Echo proxies are NOT reflection proxies (why §6 does not call them that) | pooled ρ vs MITI reflections | −0.09 to +0.02 under both graders | `text.xlsx::echo_validation_pooled` |
| 📄 Abstract's "2,112 evaluation conversations" | count | 2 arms × 11 states × 96 = 2,112 per grader | coverage (`tools/score_miproc.py plan`) |

**Retired sentences.** "the look-ahead policy asks questions instead" (abstract / §1 / old §6):
the dominant-function coder shows open-question *turns* vanishing in both arms; the `?`-count
claim ("more questions per therapist turn, 7 of 10 iterations, mean dz 0.643") survives only as
the unit note in Appendix C.3. "Look-ahead prevents the over-praise drift under the training
oracle" → "removes the praise habit and roughly halves the MI-inconsistency the held-out judge
sees" (the held-out coder reads 0.20 of K=5's endpoint turns as praise). The "1.3–2.6×" gain
sentence is unchanged.

## 2026-09-24 — ONE shared Base, the complete score table, change-talk persistence, the praise premium (step 1 of the supervisor-notes plan; numbers only, no paper text changed yet)

**Decisions (Lior, 2026-09-24).** One Base: the two GRPO base draws pooled (192 conversations; per
persona the mean of its two; both runs share the iteration-0 persona order); the paper no longer
compares the two draws. gpt-4o-mini is the main judge in the body; the held-out judge stays in
Table 1 plus one sentence per results section, the rest moves to an appendix. The all-instrument
grid replaces Figure 2. A complete score table per judge goes to the appendices; Table 4 retires.

**Where the numbers live.** New EDA family `lookahead/shared_base` (notebook
`notebooks/lookahead/shared_base.ipynb`; module `eda_analysis/shared_base.py`, plus
`process.persistence_by_state` / `persistence_metrics` / `conditioned_yield`); tables under
`Exp3_PTO_GRPO/eda/results/lookahead/shared_base/tables/` (`shared_base.xlsx`, ledger
`shared_base_numbers.json`). The praise premium is in `lookahead/mechanism`
(`praise_premium_grpo`, `k_mechanism_overpraise_chain_grpo`; `pref.feature_premium`). ⚠ K contrasts
now start at iteration 1, so the Holm family is iterations 1..10 (was 0..10) and a few
"significant at N iterations" counts move. New figures: `render_paper_figures.py
levels_grid_primary levels_grid_heldout` → `figures/levels_grid_grpo_{gpt-4o-mini,claude-haiku-4-5}.png`
(not yet included by any section). Pairs below are training oracle / held-out judge.

| claim (where) | old | new (shared Base) | source (`lookahead/shared_base/tables/` unless named) |
|---|---|---|---|
| Base-policy session length (§3.1) | 28.771 / 28.292 ("28–29") | **28.531** utterances | `marker_and_length`, iteration 0 |
| Judge level offset on Q1+Q2 (§3.3) | 1.1–1.8 | **1.167–1.805** over 21 states ("1.2–1.8") | `judge_offset` |
| Base Q1+Q2 (§4) | 3.067 & 2.963 / 1.861 & 1.834 | **3.015 / 1.848** | `levels_long`, iteration 0 |
| Gains over the Base at iteration 10, Q1+Q2 (§4) | K=0 +0.686, K=5 +1.554 / +0.396, +1.038 | K=0 **+0.738** (dz 0.885), K=5 **+1.502** (dz 1.675) / **+0.409** (dz 0.802), **+1.025** (dz 1.705) | `gains`, anchor `last` |
| Gain ratio K=5 / K=0 (§4) | 2.27× / 2.62× (last); 1.53× / 1.34× (K=0 best) | 1.502 / 0.738 = **2.04×**; 1.025 / 0.409 = **2.50×**; vs K=0's best (it 8 / it 3): 1.502 / 1.067 = **1.41×**, 1.025 / 0.789 = **1.30×** → "1.3–2.5×" | `gains`, `ratio_K5_over_K0` |
| Q1+Q2 significant iterations (§4) | 4, 6–10 / 4–7, 9, 10 ("6 of 10 under each") | 4, 6–10 (**6 of 10**) / 4–10 (**7 of 10**) | `significant_iterations` |
| Base code shares (§5) | PRA 0.031/0.036 (K=0 draw), CR 0.020/0.020 (K=5 draw), PERS 0.21 (K=5 draw), OQ 0.10/0.18 | PRA **0.041/0.038**, CR **0.018/0.031**, PERS **0.203/0.166**, OQ **0.087/0.175**; MI-adherent 0.306/0.283, MI-inconsistent 0.248/0.214 | `process_levels_<judge>`, iteration 0 |
| CR share higher under K=5, significant iterations (§5) | 6 / 5 | 4–10 (**7**) / 6–10 (**5**) | `k_process_paired`, `th_CR_rate` |
| Persuasion higher under K=5 (§5) | 6 / 7 | 2, 6–10 (**6**) / 2–6, 8–10 (**8**) | same, `th_PERS_rate` |
| Praise drift, significant iterations (§5) | 8, 10 / 5, 6, 8–10 | unchanged | same, `th_PRA_rate` |
| Judge-free marker (§5, Figure 4) | K=0 0.275 / 0.093 / 0.671 at it 8 / 9 / 10; K=5 ≤ 0.064 | K=0 0.275 / **0.094** / 0.671; K=5 ≤ 0.064; Base 0.002 (every conversation of a state; the old figure read the MICI-joined subset) | `marker_and_length` |
| Cosine between the runs' displacements (§5) | 0.44 at 10; 0.33–0.71 at 3–8 (−0.053 at 9) | **0.446** at 10; **0.351–0.731** at 3–8; −0.028 at 9 (from the shared Base centroid, GRPO runs only) | `text_drift_cosines` |
| Between-conversation variance share (§5) | 0.37 → 0.19 (K=0), 0.41 → 0.30 (K=5) | Base **0.393** → 0.192 (K=0), 0.302 (K=5) | `text_diversity` |
| Template similarity (Figure 7c) | 0.265 / 0.262 → 0.585 / 0.494 | Base **0.264** → 0.585 / 0.494 | same |
| Responsiveness at the Base (§6) | reflects CT 4–15%; praises ST ≤ 3% | reflects CT **13.7% / 3.0%**; praises ST **1.9% / 1.1%** ("3–14%", "at most 2%") | `process_levels_<judge>`, iteration 0 |
| Praise yield at the Base (§6) | 97% (n 34) / 72% (n 36) | **85%** (n 100) / **67%** (n 82); K=5 it 10 76% / 97% | `yield_<judge>` |
| Change-talk share, 10+ bin (§6) | Base 0.650 & 0.516 (per draw) | Base **0.584 / 0.580**; K=0 it 10 0.496 / 0.499 (below the Base); K=5 0.846 / 0.876 | `ct_trajectory_<judge>` |
| **NEW** Change-talk persistence, pooled over turns | — | P(CT \| previous CT): Base **0.890 / 0.860**; it 10 K=0 **0.897 / 0.857**, K=5 **0.987 / 0.970**. P(CT \| previous ST): Base 0.134 / 0.099; it 10 K=0 0.168 / 0.160, K=5 0.171 / 0.165 | `persistence_<judge>` |
| **NEW** Persistence per conversation, persona-paired | — | `ct_persist` at it 10: K=0 0.796 vs K=5 0.968 (dz −0.62) / 0.734 vs 0.935 (dz −0.65); K=5 significantly higher at 5–10 / 3–10. `ct_relapse`: significant only at 10 (both). `st_to_ct`: 5, 6, 7, 9 / 9 only | `k_persistence`, `persist_levels_<judge>` |
| **NEW** Yield conditioned on the previous patient code, it 10 | — | oracle, after CT: K=5 all turns 0.987 (n 972), CR 0.994 (322), PERS 0.985 (200); K=0 all 0.897 (526), PRA 0.871 (241). After ST: K=5 all 0.171 (380), CR 0.159 (63); K=0 all 0.168 (487), CR 0.050 (20), PRA 0.201 (174). Held-out, after CT: K=5 all 0.970, CR 0.992; K=0 all 0.857, PRA 0.835 | `cond_yield_<judge>` |
| **NEW** Praise premium in the training reward (answers Doron's "??" in §6) | — | within groups holding both kinds, praise-marker replies score above their siblings by (within-group SD) K=0 +0.33, +0.22, +0.30, +0.31, +0.07, +0.17 at policy iterations 4–9 (z 1.7–7.7); K=5 +0.16, +0.21, +0.20, +0.18, +0.01, +0.04 (z 0.3–3.5); neither positive before iteration 3 | `lookahead/mechanism/tables/praise_premium_grpo` |
| Mean therapist turn length, Base → it 10 (Limitations) | 266.3 / 279.0 → 895.7 / 849.3 | Base **272.7** → 895.7 (3.28×) / 849.3 (3.11×) — "roughly tripled" holds | `marker_and_length` |
| Parity, pooled ρ (App C.3), 21 states | CT 0.88/0.92, ST 0.90/0.94; oracle 0.02–0.43; held-out Q 0.73, GI 0.68, PERS 0.55, others 0.10–0.30 | CT 0.881/0.915, ST 0.898/0.942; oracle 0.019–0.426; held-out Q 0.731, GI 0.677, PERS 0.549, others 0.101–0.304 — unchanged at two decimals | `parity_pooled_<judge>` |
| App E: Q1 ceiling shares, K=5, oracle | 14% → 58%; 40% at the maximum | Base **12.5%** → 58.3%; 39.6% at the maximum | `sd_by_iter` |
| App E: Q1 per-conversation SD, K=5, oracle | 1.336 → 0.701; ρ −0.86, p .001; variance ratio 0.275 (0.285 vs it 1) | Base **1.314** → 0.701; ρ −0.864, p .001; variance ratio **0.284** (0.285 vs it 1) | `sd_trend` |
| App E: held-out SD trend, K=5 | ρ +0.44, p .18 | ρ +0.436, p .180 | same |
| App E: Q1 agreement — median / next-lowest / K=0 range | 0.842 over 22 states / 0.744 / 0.744–0.882 | 0.842 over **21** states (Base 0.858) / 0.744 / 0.744–0.882 | `agreement_by_state` |
| App E Table 7 (median, Δ median, rank) | e.g. MITI 0.678 / −0.345 / 1 of 22; MICI 0.399 / −0.112 / 4 of 22 | Q1 0.842 / −0.298 / 2 of 21; Q2 0.752 / −0.162 / 1; WAI-SR 0.920 / −0.023 / 3; CSQ-8 0.888 / −0.037 / 4; MI-SAT 0.930 / −0.025 / 4; MITI 0.666 / −0.334 / 1; PCT 0.956 / −0.027 / 3; MICI 0.411 / −0.123 / 4 | `agreement_summary` |
| App E: sign preservation | 1,640 of 8 × C(22,2) = 1,848 (88.7%); 98.9% at \|Δ\| ≥ 0.50 | 1,484 of 8 × C(21,2) = 8 × 210 = **1,680 (88.3%)**; 98.9% (373 of 377) at \|Δ\| ≥ 0.50 | `sign_preservation` |

**§4 rewritten on the shared Base (2026-09-24, step 3 of the plan).** Every number in the new §4
is in the table above, plus: the first iteration at which each instrument's K contrast is
significant in K=5's favour (training oracle) — Q1 4, CSQ-8 4, MI-SAT 4, PCT 4, WAI-SR 5, MITI 6,
MICI 8, Q2 9 ("six of the eight separate by iteration 6, Q2 and MICI at iterations 9 and 8"); the
one significant difference in K=0's favour is MICI at iteration 3 (held-out: MICI and MITI at 3)
— all `significant_iterations`; and the best-checkpoint contrasts, unchanged by the Base (K=5 at
10 vs K=0 at 8: dz 0.743 training oracle; vs K=0 at 3: dz 0.386 held-out; the §5 rows above).
Retired from §4: the base-vs-base sentence (2.963 vs 3.067, dz 0.115) and Doron-marked "MI-
inconsistent behaviour shows the largest standardised effect of all". Figure 2 is now
`levels_grid_grpo_gpt-4o-mini.png` (its held-out twin replaces the copied grid in Appendix A);
the Q1+Q2-only headline and the two copied EDA grids left `figures/`.

**§5 rewritten on the shared Base (2026-09-24, step 4 of the plan).** The body now reports the
training oracle only, plus one held-out sentence. Numbers the new §5 adds to the rows above, all
from `lookahead/shared_base/tables/` unless named:

| Claim (§5) | Value | Source |
|---|---|---|
| K=0 praise share, "late and unevenly" | 0.041 (Base) → 0.222 / 0.080 / **0.407** at it 8 / 9 / 10 | `process_levels_gpt-4o-mini`, `th_PRA_rate` |
| K=5 complex-reflection share | 0.018 (Base) → **0.230** at it 10; higher than K=0 at 4–10 ("every iteration from 4 on") | same + `k_process_paired`, `th_CR_rate` |
| MI-adherent share at it 10 | K=5 **0.438** vs K=0 0.290 (Base 0.306); K=5 higher at 6, 8, 9, 10 | same, `mi_adherent_rate` |
| Open questions lost | 0.087 (Base) → 0.000 (K=0) / 0.010 (K=5) at it 10 | same, `th_OQ_rate` |
| Turn length "roughly triple" | 272.7 chars (Base) → 895.7 (K=0, 3.28×) / 849.3 (K=5, 3.11×) → "273 … 850–900" | `marker_and_length`, `mean_turn_len` |
| Persuasion residue | 0.203 (Base) → **0.279** (K=5) at it 10; higher than K=0 at 2, 6–10 (6 of 10) | `process_levels_gpt-4o-mini` + `k_process_paired`, `th_PERS_rate` |
| **NEW** MI-inconsistent timing ("the gap is K=0's late praise, not an MI gain of K=5") | it 10: K=5 **0.332** vs K=0 0.463 (Base 0.248); significant in K=0's favour at **2, 6, 9**, in K=5's only at **10**. K=5's own share: 0.28–0.38 over it 1–10. Held-out: K=0 lower at 3, 5; K=5 lower at 8, 10 | `k_process_paired`, `mi_incons_rate` |
| Held-out sentence | K=5 praise share at it 10: **0.203** held-out vs 0.053 training oracle | `process_levels_<judge>` |
| Keyword marker vs the coder's praise share ("same rise, dip and rise") | marker 0.275 / 0.094 / 0.671 vs coder 0.222 / 0.080 / 0.407 at it 8 / 9 / 10 | `marker_and_length`, `process_levels_gpt-4o-mini` |
| MICI composition, K=0 at it 10 (not Base-dependent) | total **9.865** ("9.9"), over-praise **8.250** ("8.3"), share **0.836** ("84%") | `lookahead/behaviour/tables/k_mici_composition.md` |
| Cosine between the runs' moves | max **0.731** (it 3); **0.446** at it 10; **−0.028** at it 9, where K=0's turns shorten (mean 338 chars vs 822 at it 8 and 896 at it 10) and its praise share dips (0.080) | `text_drift_cosines`; `marker_and_length`; `process_levels_gpt-4o-mini` |
| Between-conversation share | Base **0.393** → K=5 0.302, K=0 0.192 at it 10 | `text_diversity` |

Retired from §5: the "Where the graders disagree" paragraph (its dz −0.37 / −1.08 and "roughly
halves the MI-inconsistency the held-out judge sees"), replaced by the timing sentence and the one
held-out sentence. New body figure `text_grpo_body.png` (Figure 4) = panels (a) and (b) of
Appendix Figure `text_grpo.png`, drawn by `render_paper_figures.py::textspace_body`. The judge-free
marker, the process figures, the responsiveness figure and the embedding figure now read the
shared-Base workbook; their data at iterations 1–10 is unchanged. The lexical marker is renamed
"the keyword marker" in §5, Appendix A and Appendix C.5; the Limitations still say "a deterministic
lexical marker" until step 7.

**Reference added 2026-09-24** (verified against the ACL Anthology page that day):
`reimers2019sbert` — Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks*, EMNLP-IJCNLP 2019, pp. 3982–3992, doi 10.18653/v1/D19-1410 — cited in §5 for the
`all-MiniLM-L6-v2` sentence encoder (Doron's "[which one]").

**§6 rewritten on the shared Base (2026-09-24, step 5 of the plan).** Responsiveness → persistence
→ the session; training oracle in the body, one held-out sentence. Two per-conversation measures
were added to `eda_analysis/process.py` for it — `pers_after_st` and `refl_after_st`, the other two
answers to sustain talk — and `lookahead/process` + `lookahead/shared_base` were re-rendered; every
pre-existing value in both workbooks was checked identical after the render (only the new columns
and metric rows were added). Table 3's rows are printed from the workbook by a script, not typed.

| Claim (§6 / Table 3) | Value (training oracle; Base / K=0 / K=5 at it 10) | Source |
|---|---|---|
| Table 3, therapist block | PRA 0.041 / 0.407 / 0.053 (dz −1.19); CR 0.018 / 0.020 / 0.230 (+0.90); PERS 0.203 / 0.055 / 0.279 (+0.79); MI-adherent 0.306 / 0.290 / 0.438 (+0.40); MI-inconsistent 0.248 / 0.463 / 0.332 (**−0.36**, was printed −0.37: dz 0.3650 rounds down) | `process_levels_gpt-4o-mini`, `k_process_paired` (dz negated to K5 − K0) |
| Reflects change talk | 0.137 / 0.003 / 0.264 (+0.99, n 76); significant at **1 and 6–10** | same, `refl_after_ct` |
| Praises sustain talk | 0.019 / 0.315 / 0.015 (−1.10, n 65) | same, `pra_after_st` |
| **NEW** Persuades after sustain talk | 0.242 / **0.073** / **0.393** (+0.90, n 65); K=5 higher at 8–10. Held-out 0.253 / 0.056 / **0.580** (+1.61; 3–6, 8–10) | same, `pers_after_st` |
| **NEW** Reflects sustain talk | 0.217 / 0.256 / 0.368 (+0.27, p_holm .30; significant only at 9). Held-out 0.081 / 0.050 / 0.171 (+0.49, **; 8–10) | same, `refl_after_st` |
| Change talk after change talk (per conversation) | **0.750 / 0.796 / 0.968** (+0.62, n 76); K=5 higher at **5–10**. Held-out 0.715 / 0.734 / 0.935 (+0.65; 3–10) | `persist_levels_<judge>`, `k_persistence`, `ct_persist` |
| Change talk after sustain talk (per conversation) | 0.281 / 0.276 / 0.326 (+0.18, n.s. at 10); K=5 higher at **5–7, 9**. Held-out +0.04 (only 9) | same, `st_to_ct` |
| "four most common replies to change talk … at least 97%" (pooled over turns, it 10, K=5) | CR 0.994 (n 322), AF 0.995 (205), PERS 0.985 (200), GI 0.972 (176); all replies 0.987 (972) | `cond_yield_gpt-4o-mini` |
| "83% of K=5's complex reflections come right after change talk" | responsiveness shares × n: after CT 975 × 0.331 = 322.7; after ST 381 × 0.165 = 62.9; after NEU 81 × 0.049 = 4.0 → 322.7 / 389.6 = **0.828** | `responsiveness_gpt-4o-mini` |
| "reflections do about as well as the other replies" | after CT: CR 0.994 vs all 0.987 ("0.99 against 0.99"); after ST: CR 0.159 (n 63) vs all 0.171 (380) | `cond_yield_gpt-4o-mini` |
| Change-talk share of patient utterances | 0.448 / 0.528 / 0.683 (+0.56); K=5 higher at **6–10**; reached any CT 0.724 / 0.865 / 0.917, n.s. at 10 | `process_levels_gpt-4o-mini`, `ct_prop`, `reached_ct` |
| Late session (10+ patient turns) | Base 0.584 (133 of 192 sessions), K=0 0.496 (62 of 96), K=5 0.846 (74 of 96); turns 6–9: 0.510 / 0.680 / 0.796 | `ct_trajectory_gpt-4o-mini` |
| Patient utterance length, disengagement cue (not Base-dependent) | 439.7 → 600.3 chars (dz 1.24); disengage 0.213 → 0.276 (dz 0.31, p_holm .026) | `lookahead/text` `k_text_paired`, it 10 |
| Praise premium (answers Doron's "??") | K=0 policy it 4–7: 0.331 / 0.220 / 0.302 / 0.309 ("0.22–0.33"; z 4.7, 2.8, 5.5, 7.7); K=5: 0.165 / 0.208 / 0.197 / 0.178 ("0.16–0.21"; z 1.6, 1.9, 2.6, 3.5); it 8: 0.070 / 0.013; it 9: **0.165 (z 5.7) / 0.039 (z 1.1)**; early: K=0 −0.375, −0.625 at it 0–1, K=5 −0.501 at it 2; mixed-group share ≈ 2% at it 0–2, K=0 0.52 / 0.51 / 0.57 at it 7–9 vs K=5 0.28 / 0.33 / 0.44 | `lookahead/mechanism/tables/praise_premium_grpo` (iteration = train_iter − 1) |
| Held-out Table 5 (appendix twin of Table 3) | persuasion dz **+0.83** (was printed +0.84: 0.8345 rounds down); every other printed value unchanged | `process_levels_claude-haiku-4-5`, `k_process_paired` |

Retired from §6: the per-code yield paragraph and Figure 3c's yield bars (placement-confounded — see
the two rows above), the base-praise-yield sentence (97% / 72%, then 85% / 67% on the shared Base),
and "the praise spiral … at scale". New: Figure 3c = change-talk persistence by iteration
(`persist_levels_<judge>`), Appendix B.4 + `praise_premium_grpo.png`, Appendix Table 5 (held-out
Table 3), the responsiveness figure redrawn 2 judges × 4 replies. Figure legends name the runs "K=0"
/ "K=5" only (the "(turn-level)" / "(look-ahead)" suffixes dropped, per the vocabulary decision).
*(Letters moved in step 6: the praise premium is now Appendix C.4, the held-out Table 3 is Table 5
in the new Appendix B.)*

**Appendices restructured (2026-09-24, step 6 of the plan; Lior chose "new Appendix B, saturation
stays").** A = training-oracle extras; **B = the held-out judge** (new, `sections/A2_heldout.tex`:
its grid, complete score table, process figure and process table, plus one paragraph of where it
agrees and differs); C = mechanism (+ the praise premium, C.4); D = reproducibility; E = the two
examples; F = saturation, reframed as a limit of the main judge. Each appendix starts on a new page
so its floats stay with it. Table 4 (`tab:byiter`, Q1+Q2 at every iteration with the two Base
draws) is **retired**; its iteration-0 row and "noise floor" wording go with it.

| Claim (appendices, Limitations) | Value | Source |
|---|---|---|
| Complete score tables (Tables 4 and 6) | printed from the workbook by a script: 21 rows × 9 instruments, mean over 96 personas (Base over 192), a star on the better run where p_holm < .05 (Holm across iterations 1–10 within instrument); K=0 cells starred: MICI at it 3 (oracle); MITI and MICI at it 3 (held-out) | `shared_base` `score_table_<judge>` |
| App B: judge offset on Q1+Q2 | 1.167–1.805 → "1.2–1.8" | `judge_offset` |
| App B: held-out summary | Q1+Q2 significant at 4–10 (7 of 10) vs oracle 6; K=0's best held-out checkpoint it 3 (2.637 vs 2.617 at it 8), dz 0.386 vs the final K=5; gain ratios 2.50 / 1.30; praise share K=5 0.203 vs oracle 0.053; MI-inconsistent 0.471 / 0.795 / Base 0.214, K=0 lower at 3, 5 and K=5 lower at 8, 10; persuades after ST 0.580 (Base 0.253); reflects ST 0.171 vs 0.050; persistence significant from it 3 | the §4–§6 rows above |
| App F: sign preservation | 1,484 of 8 × C(21,2) = 8 × 210 = 1,680 (88.3%); 373 of 377 (98.9%) at \|Δ\| ≥ 0.50 | `sign_preservation` |
| App F: Q1 agreement | median 0.842 over 21 states (Base 0.858); next-lowest 0.744 (K=0 it 6); K=0 range 0.744–0.882; K=5 it 5→10 0.941, 0.877, 0.842, 0.769, 0.487, 0.544 | `agreement_by_state` |
| App F Table 9 | printed by script: MITI 0.333 / 0.666 / −0.334 / 1 of 21; Q1 0.544 / 0.842 / −0.298 / 2; Q2 0.590 / 0.752 / −0.162 / 1; MICI 0.287 / 0.411 / −0.123 / 4; CSQ-8 0.851 / 0.888 / −0.037 / 4; PCT 0.928 / 0.956 / −0.027 / 3; MI-SAT 0.905 / 0.930 / −0.025 / 4; WAI-SR 0.898 / 0.920 / −0.023 / 3 | `agreement_summary` |
| App F: the ceiling | Q1 ≥ 4.5: Base 12.5% → 58.3% at it 10, 39.6% at the maximum score; SD 1.314 → 0.701, ρ −0.864, **p = .0006 → "p < .001"** (was "p = .001"); variance ratio 0.2845 vs Base, 0.2854 vs it 1 → "0.28 either way" (the anchors agree, and the trend is monotone — rule 2b holds) | `sd_by_iter`, `sd_trend` |
| App F: K=0 spread tracks level | SD 0.922–1.011 over it 3–8 (means 3.81–3.95); 0.982 / 1.142 at it 9 / 10 (means 3.53 / 3.61) | `sd_by_iter` |
| App F: held-out K=5 | share ≥ 4.5 is 0 at every state ("gives no conversation 4.5 or more"; replaces "the upper half of its scale unused", which was loose — up to 9.4% of its K=5 conversations score ≥ 4); ρ +0.436, p .180 | `sd_by_iter`, `sd_trend` |
| Limitations + App D: parity | pooled ρ CT 0.881 / 0.915, ST 0.898 / 0.942 over 21 states → "0.88–0.94" (was "0.88–0.95 across the 22 states"; the 0.95 was a slip) | `parity_pooled_<judge>` |

Also in step 6: Limitations reframed (the base-vs-base sentence and its Table 4 citation gone; "the
held-out judge and the instruments outside the reward are the load-bearing evidence" → they are
"the checks on it"; arm → run, grader → judge, process coder → utterance coder, "deterministic
lexical marker" → "the keyword marker"); Appendix D gains the Base-pooling sentence and "21 model
states"; the config table's "Base model" row is now "Therapist model" (Base means iteration 0).
Still open for step 7: the Discussion and Ethics wording ("arm", "graders") and the Limitations
paragraph on patient replies inside the K=5 reward.
