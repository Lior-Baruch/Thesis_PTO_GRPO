# NUMBERS.md — the claims ledger

Every quantitative claim in the draft → the exact tracked artifact it came from. `results/…` paths
are relative to `Exp3_PTO_GRPO/eda/`. **Nothing enters the `.tex` that is not a row here**, and no
row is written from prose — each was read off the named table or recomputed from the score lake.

*(2026-09-02 rewrite: sections renumbered — §3 is now the method, §4 the setup, §5 reward, §6
behaviour, §7 mechanism, §8 measurement, §9 discussion; appendices A tables · B mechanism · C repro.
New rows are marked **NEW**. Numbers unchanged from the 2026-08-27 ledger keep their marks.)*

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

⚠ **Never swap in a "better" K=5 turn or a "worse" K=0 turn.** The persona and the utterance
index are fixed by rule; the K=5 turn's flaws (agreeing with the pessimism, the first-person slip)
are stated in the caption on purpose.

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
| ✅ Q1 median across the 22 GRPO states | **0.841** (exact 0.8415; the data table's display rounds to 0.842, the figure prints 0.841 — the paper matches its figure) | results/measurement/validity/tables/judge_saturation_grpo_data.md |
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
| Figure 4 `judge_saturation_grpo.png` | results/measurement/validity/tables/validity.xlsx sheet `judge_saturation_grpo_data`: panel-a rows (`cross_judge_pearson_r` per state + the 22-state median), panel-b rows (`sd_of_per_conversation_score` per grader) | the Spearman ρ/p in the legend (from the 11 SD rows; must equal the §8 text) |

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
