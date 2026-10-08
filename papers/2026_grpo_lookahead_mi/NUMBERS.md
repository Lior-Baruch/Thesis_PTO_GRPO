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
| Bootstrap | percentile, `BOOT_SEED` = 12345; 2,000 resamples for the dispersion ratios (the `stats.py` default), 1,000 for the faithfulness intervals and Figure 9's bands (App D.7 says so since 2026-10-05; it used to claim 2,000 for every interval) | eda_analysis/constants.py (seed only); eda_analysis/stats.py (2,000); results/lookahead/mechanism/tables/CAPTIONS.md (dispersion_ratios: 2,000; faithfulness_*: B=1000) |
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
| ~~Ethics total (one line, no per-arm breakdown)~~ — **retired 2026-09-29** with the Ethics compute paragraph; the per-run totals stay (Limitations, Appendix D.8) | 27.906 + 51.205 = 79.111 ≈ **79 GPU-hours** | results/compute/cost/tables/compute_by_arm.md — show the arithmetic |

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
| ~~Every endpoint is a single 96-conversation draw~~ — **cut from the Limitations 2026-09-29** (Doron pass 4); §4's last paragraph carries the re-draw | results/LIMITATIONS.md § 5c |
| ~~The look-ahead reward is also a reward for continuing~~ — **cut from the Limitations 2026-09-29**; the rollout audit stays in Appendix A (text + Figure `fig:tails` caption, which now states the pressure in plain words) | this ledger, §3 rows |
| All 96 personas are used for both training rollouts and eval — every number is in-sample | results/LIMITATIONS.md § 5e |
| Patient simulator and training oracle are the same model; the held-out judge decouples the grader, not the generator | results/LIMITATIONS.md § 2 |
| No human MI-coder validation of any instrument | results/LIMITATIONS.md § 1 |
| Q1+Q2 is both the training reward and a reported outcome — **§3.3 only since 2026-09-29** (the Limitations sentence was cut as a repeat) | results/LIMITATIONS.md § 3 |
| MITI is the least dependable instrument | results/LIMITATIONS.md § 2 |
| Matched iterations ≠ matched cost — **since 2026-09-29 plain words in the Limitations (GPU-hours 51.2 / 27.9, ≈2× per step, the equal-GPU-hour verdict without numbers); the call counts, the 1.92× median and the iso-compute d_z values live in Appendix D (`app:repro-cost`)** | this ledger |

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

**Carry-through (2026-09-24, step 7 of the plan).** No new number; the claims that §5–§6 changed
are carried into the abstract, the Discussion, the Limitations and the Ethics statement.
- **Abstract:** the retired "a look-ahead reflection is followed by change talk in 85–89% of
  cases" (placement-confounded) → persistence, "97% of cases, against 80%" (per conversation,
  `persist_levels_gpt-4o-mini`); new sentence that look-ahead meets sustain talk with persuasion;
  arm → horizon, graders → judges. The full abstract rewrite is step 8.
- **Discussion:** the mechanism paragraph is rebuilt on measurements instead of the old story
  ("a patient met with reassurance … produces sustain talk", "reflecting … which the patient then
  elaborates" — neither was measured): the praise premium (K=0 still 0.16 SD at it 9, K=5 0.04;
  `praise_premium_grpo`), K=0's praise share 0.41 and praise after a third of sustain talk
  (0.315), its late-session change talk below the Base (0.50 vs 0.58), persistence 97% vs 80%,
  and K=5's persuasion after sustain talk. Conclusion: "reflects the patient's own case for change"
  (a metaphor) → "reflects the patient's change talk and hears more of it, though it still
  persuades when the patient resists".
- **Limitations:** new paragraph "The patient's replies are inside the look-ahead reward" (the
  K=5 reward is given after three patient replies, the K=0 reward before any, so §6's patient-side
  results partly measure what the K=5 reward optimised, on the reward model's own simulator);
  "endpoint" → iteration wording; session lengths 31.9 vs 25.2 (`marker_and_length`, it 10).
- **Ethics:** graders → judges, arms → runs, "base model" → "therapist model"; the residual drift
  sentence now names what look-ahead learned instead (persuasion when the patient resists).

## 2026-09-29 — step 2 of the pass-4 plan: the Limitations (Doron's pass-4 notes, Lior's picks)

No number changed; numbers moved. Re-checked against the tables that day:
`compute/cost/tables/budget_sweep_GRPO_K_{gpt-4o-mini,claude-haiku-4-5}.md` (13.270 h: d_z −0.742 /
−0.780; 23.210 h: 0.074 p .789 / 0.331 p_holm .012) and `compute_by_arm.md` (27.906 / 51.205).

| What | Where it was | Where it is now |
|---|---|---|
| Oracle calls 302,541 / 289,983; 392,766 K=5-only patient calls; median 1.92× per step | Limitations "Matched iterations" | Appendix D `app:repro-cost` ¶1 (exact 392,766 instead of "≈393k") |
| GPU-hours 27.9 / 51.2 | Limitations + Ethics | Limitations (plain sentence) + Appendix D ¶1; Ethics until step 3 |
| Iso-compute d_z −0.74 / −0.78 (≈13 GPU-h), 0.07 n.s. / 0.33 (≈23 GPU-h) | Limitations | Appendix D `app:repro-cost` ¶2 ("significant" instead of `p_holm = .012`); the Limitations keep the verdict only |
| Re-draw \|d_z\| ≤ 0.174 | Limitations "Evaluation draws" + §4 | §4 only |
| Rollout audit 82%, "up to 0.09", "less often than chance" | Limitations + Appendix A | Appendix A only |
| Session length 31.9 vs 25.2 utterances (it 10) | Limitations | **gone from the paper** (the paragraph was cut) |

Cut paragraphs (Lior's picks l2, l5, l6, l7, l13): Evaluation draws; One optimiser, one regime;
the reward for continuing; the patient's replies inside the reward; the Q1+Q2 circularity sentence
(its Appendix F pointer kept as a plain sentence). Knock-ons: §3.3 lost "(Limitations)"; Appendix
A's text and the `fig:tails` caption no longer point at the Limitations ("lever" → "the
look-ahead"); Appendix D no longer says the GPU-hour totals are "in the Limitations and the Ethics
Statement".


## 2026-09-29 — step 4: the 200-token cap, counted (replaces "many" in the Limitations)

New EDA artifact: `results/lookahead/shared_base/tables/cap_hits.md` (`behavior.cap_hits` →
`shared_base.cap_hits_by_state`; ledger keys `cap.*` in `shared_base_numbers.json`). A turn is
capped when its stored text re-tokenizes (the therapist's own Llama-3.2-1B tokenizer) to ≥ 199
tokens; the counts spike at 199–201, so the threshold is not a judgement call. Opener excluded.
Unit = share of therapist turns, pooled over the state's conversations (Lior's picks: token count,
iteration 10 + Base, share of turns).

| Claim (Limitations) | Value | Source |
|---|---|---|
| Base | 79 / 2,564 = 0.031 → "3%" | `cap_hits.md`, iteration 0 (shared Base, 192 conversations) |
| K=0 at iteration 10 | 1,091 / 1,128 = 0.967 → "97%" | same, GRPO_LA0 row 10 |
| K=5 at iteration 10 | 1,162 / 1,437 = 0.809 → "81%" | same, GRPO_LA5 row 10 |
| "the K=5 share climbs from iteration 5 on" | 0.331, 0.423, 0.610, 0.766, 0.805, 0.809 (it 5–10; non-decreasing, it 9→10 flat) | same, GRPO_LA5 rows 5–10 |
| "the K=0 share swings, 9% at iteration 9" | 0.156, 0.101, 0.333, 0.719, **0.091**, 0.967 (it 5–10) | same, GRPO_LA0 rows 5–10 |
| "binds the K=0 policy somewhat more often" at iteration 10 | 0.967 vs 0.809 | same |

Not in the paper (scratch check, 2026-09-29): capped turns that also carry a malformed ChatML
marker (`<\|?im_`) are ≤ 3.6% of turns in every late state checked (K=0 it 7–10, K=5 it 8–10), so
the marker leak does not drive the cap hits. The old clause "so it does not bias the contrast" was
replaced (Lior's pick) — the cap binds the two runs differently at iteration 10.

**Same day, after Lior's read ("too negative"): the cap sentences are CUT from the Limitations
(his pick among four options), and §5's "near the 200-token cap (see Limitations)" lost its pointer.
None of the step-4 numbers above appears in the paper any more.** The cap is still stated in §3
("responses capped at 200 tokens", Table 7) and Appendix E's note still explains the mid-sentence
endings. The count stays in the EDA (`cap_hits.md`, EDA `LIMITATIONS.md` §5h).

## 2026-09-29 — step 5: MITI option A (the paper drops MITI's seven behaviour counts)

Lior's picks: drop the channel forest (Appendix A, `fig:forest`), drop "7 behaviour counts" from
the MITI row of the instruments table, and the short utterance-coder paragraph in the Limitations
(human-check sentence kept). MITI stays one of the eight instruments through its four global
ratings. **Retired from the paper** (the tables still hold them): the forest's channel effect sizes
(row "Endpoint channel effect sizes on the forest" above); the therapist-side parity numbers of
Appendix D.3 — training oracle 0.02–0.43 over the seven MITI codes, held-out 0.73 / 0.68 / 0.55 /
0.10–0.30 — and "5.5 questions per conversation vs 1.7 question-turns"; the Limitations' "pooled
ρ 0.88–0.94" (Appendix D.3 keeps the four patient-side values 0.88 / 0.92 / 0.90 / 0.94); the
Limitations' "MITI-style coder is the least dependable instrument". Also removed as FALSE (found
2026-09-28): Appendix D.3's "the MITI coder counts every function a long turn performs".
`render_paper_figures.py::forest()` is kept but no longer called; `figures/k_channel_forest_grpo_gpt-4o-mini.png`
is deleted (the Overleaf push deletes it there too).

**Steps 5–6, after Lior's joint read (2026-09-29).** No number changed. "Premium" is gone from the
visible paper (Lior dislikes the word): App C.4's text and Figure 12's caption, y-label and panel
title now say "how much higher the reward scores praising replies" / "difference" / "praising −
other" (the labels `app:premium`, `fig:premium` and the file `praise_premium_grpo.png` keep their
internal names). A missed MITI remnant was fixed: Appendix D.2 said the MITI coder "assigns exactly
one of seven behaviour codes to each therapist utterance" — the behaviour codes are no longer part of
the paper, and "exactly one" was also contradicted by the 79% / 96% measurement; the sentence now
lists only the four global ratings. A sweep found no other MITI behaviour-count remnant (the MICI
row's "6 behaviour counts" is MICI's and stays).

## 2026-09-30 — step 7: Lior's read (Table 1, the embedding figures, the rollout audit)

No number changed; three displays changed.
- **Table 1** (`tab:endpoint`) lost its bold: every one of the 18 bold cells was a K=5 value, so it
  marked nothing. The caption now says K=5 is better on every instrument under both judges (MICI:
  lower), every row p < .001 (Holm, nine rows per judge) — the same values as before.
- **The body's embedding figure** (`fig:textspace-body`, `text_grpo_body.png`) is gone; §5 keeps two
  sentences with the between-conversation share at iteration 10: Base **0.393**, K=5 **0.302**,
  K=0 **0.192** (`shared_base.xlsx::text_diversity`, `persona_var_share`, re-read 2026-09-30).
  Appendix Figure `fig:textspace` (`text_grpo.png`) is now that panel alone, column width, drawn by
  `render_paper_figures.py::textspace()`. **Retired from the paper**: the same-direction cosine
  (at most 0.73 at iteration 3, 0.45 at iteration 10, near zero at 9;
  `shared_base.xlsx::text_drift_cosines`) and the template-similarity panel. The three-panel and
  two-panel drawings are kept in the script, not called.
- **The rollout audit figure** (`fig:tails`, `tail_audit_grpo.png`) is gone; Appendix A keeps its
  paragraph under a run-in heading, with the caption's reading folded in ("the look-ahead reward
  slightly favours turns after which the session goes on"). Its numbers are the rows "Rollout
  audit" above (82% full, 12–30% early, 16% patient-closed, up to 0.09 below the group mean,
  0.10 vs 0.125 best-of-group).
- Figure 4's y-label (the keyword marker) was clipped at column width; now on two lines.

**Step 7 revised the same day (Lior: "do the fixes and also honour Doron's request").** Final state:
- **Table 1 at two decimals**, rounded from the full-precision sheets (`reward.xlsx::k_endpoints`,
  pair `GRPO_LA5_I10 − GRPO_LA0_I10 (K=5 endpoint vs K=0 endpoint)`, and `::k_levels_long`,
  iteration 10), every cell checked to agree with the old 3-dp cell. ⚠ Two cells differ from
  naive rounding of the 3-dp string: Q1+Q2 Δ is 0.7646 → **+0.76** (not 0.77), held-out CSQ-8 K=5
  is 2.9349 → **2.93**. The caption adds "Δ is computed before rounding" (4.52 − 3.75 = 0.77).
  §4's prose that restates Table 1 follows it: +0.76 (dz 0.91), held-out +0.62 (dz 1.03), the
  replicate |dz| 0.17 and dz 0.92 vs 0.91 (held out 0.95 vs 1.03); the other §4 numbers keep three
  decimals (they come from Table 4).
- **Figure 4 stays in the BODY** (Doron's pass-3 request), redrawn: panel (a) the
  between-conversation share (unchanged), panel (b) the same-turn similarity
  (`shared_base.xlsx::text_diversity`, `template_sim`): Base **0.264** → K=5 **0.494**, K=0
  **0.585** at iteration 10; K=5 the higher at **7 of 10** iterations (1, 2, 4, 5, 6, 7, 9). No CI
  exists for this measure, so the text makes no significance claim. The same-direction cosine
  stays retired. The Appendix A embedding figure is gone (it would duplicate Figure 4;
  `text_grpo.png` deleted, `textspace()` kept, not called).
- **The rollout paragraph** is two sentences: 82% full, 16% patient-closed, at most 0.09 below the
  group mean. Dropped: the 12–30% range, the 0.10 vs 0.125 best-of-group rate and the "favours
  turns after which the session goes on" clause (the point Lior cut from the Limitations).

## 2026-09-30 — step 8: the complete process tables (Appendix A and B)

Lior: "Table 3: do we have a full table of all iterations in the appendix?" (there was none, only
figures); his picks: Table 3's twelve measures only, one table per judge. New Tables
`tab:process-all` (Appendix A, training oracle) and `tab:process-all-heldout` (Appendix B):
Base + K=0 iterations 1–10 + K=5 iterations 1–10, printed by the new tracked script
`render_process_tables.py` from `shared_base.xlsx` — levels from `process_levels_<judge>` and
`persist_levels_<judge>` (`ct_persist_mean`, `st_to_ct_mean`), stars from `k_process_paired` and
`k_persistence` (star on the `better` run where `p_holm` < .05; Holm across iterations 1–10 within
measure). The script asserts, before printing: the two arms' Base rows are identical (shared Base);
each measure's `lower_better` flag matches the table's arrow; every iteration-10 cell equals the
printed Table 3 / held-out process-table cell (`tab:process`, `tab:process-heldout`) to three decimals; and the starred iteration-10 cells are exactly the
cells those two tables bold. Star counts: training oracle **51** (K=0 12, K=5 39), held-out
**67** (K=0 19, K=5 48); K=0's stars are persuasion, MI-inconsistent share, persuades-after-sustain
talk and (held-out) MI-adherent share at iterations 2–3. The first three are the measures §5–§6 say
favour K=0; the held-out MI-adherent lead at iterations 2–3 is NOT stated anywhere in the text
(corrected 2026-10-01; Lior chose to leave it to Table 8).
No number is new to the paper's claims: the tables only lay out values the figures already drew.

**Bold, mid-step (Lior: "in the tables bold each column with best score").** In both new tables
AND in the complete score tables (`tab:scores`, `tab:scores-heldout`: Tables 4 and 7 after this step, same layout) the best printed value of each
column over the 21 model states is bold (lowest where lower is better: MICI and the ↓ process
columns; ties at the printed precision all bold). No value changed. Tables 1, 3 and 5 keep their own
rules (Table 1 no bold; the iteration-10 process tables bold the better run where significant). Best per column —
Table 4: Q1+Q2 4.517, Q1 4.465, Q2 4.570, WAI-SR 3.729 (all K=5 it 10), CSQ-8 3.078, MI-SAT 3.847,
PCT 0.699 (K=5 it 9), MITI 4.536 (K=5 it 10), MICI **0.169 (K=0 it 2)**. Table 7 (held-out):
Q1+Q2 2.912 (K=5 it 7), Q1 2.898 (it 6), Q2 3.015, CSQ-8 2.935 (it 10), WAI-SR 3.014, MI-SAT 3.394,
MITI 2.398, PCT 0.754 (it 9), MICI **0.309 (K=0 it 2)**.

## 2026-09-30 — step 9: Doron's related-work block merged (§2)

No numbers. Lior's picks: the block becomes one bold-headed paragraph, "Judges that change during
training", placed after "Reward models and LLM judges", with one shared closing contrast (Doron's
own closing sentence, kept); his original three paragraphs stay in the .tex as a comment until the
merge is approved. The four missing references were found and checked that day against their
pages: `wu2025metarewarding` (EMNLP 2025, pp. 11537–11554, doi 10.18653/v1/2025.emnlp-main.583),
`wang2026serpo` (arXiv 2607.26873), `wang2026dynamicrubric` (arXiv 2607.20083), `chu2026jzero`
(arXiv 2608.26582, v2 title "... Self-Evolution from Zero Data", four authors). Claims checked
against each paper: SERPO evolves query-specific rubrics with the policy at test time, on
HealthBench, ResearchQA and four OOD benchmarks, one response per prompt; DynamicRubric generates
weighted binary rubric items per candidate set (AlpacaEval2, ArenaHard v2.0, WildBench,
WritingBench, MATH-500, MMLU-Pro, CodeScope); J-Zero co-trains Challenger, Solver and Judge
(AlpacaEval 2.0, Arena-Hard v2.0, EQ-Bench Creative Writing v3 on the unverifiable side); none of
the three evaluates an interaction the policy conducts over several turns, nor counselling
role-play. Dropped with the halving: Doron's per-paper remarks and "continued gains across ten
iterations" (J-Zero reports "at least ten iterations", which is true but not needed).

## 2026-10-01 — the float atlas's "worth a look" items fixed (before Lior's read-through)

An atlas of every float (private artifact https://claude.ai/artifact/CQGDKtm5zJFPxZRNj3M6zt) listed
62 checked items; Lior: fix them first. His picks: figures redrawn at print width; Figure 4(b)'s
"K=5 the higher at seven of the ten iterations" dropped (a count of point values with no interval;
`text_diversity.template_sim` K=5 > K=0 at 1, 2, 4–7, 9, e.g. 0.466 vs 0.461 at 7, K=0 higher at
3, 8, 10); the held-out K=0 MI-adherent lead at iterations 2–3 NOT added to the text; the bold rule
kept. New or changed numbers in the paper, each checked against its source:

| Where | Number | Source |
|---|---|---|
| §5, the Base's code mix | information giving 0.378 + persuasion 0.203; then simple reflections 0.137, open questions 0.087; "no other code above 0.05" (praise 0.041, closed questions 0.039, seeking collaboration 0.035, other 0.029, affirmation 0.028, complex reflections 0.018, confront 0.004) | `shared_base.xlsx::process_levels_gpt-4o-mini`, iteration 0 |
| Figure 4 caption | conversations with ≥1 non-empty therapist turn after the opener: 184 of the Base's 192; 88–96 per run state | `shared_base.xlsx::text_diversity.n_convs` (`text.py::diversity_by_state`, `content_mask`) |
| §6, held-out paragraph | the held-out paired effect is the larger on 10 of the 12 measures (\|d_z\| Table 6 vs Table 3: larger on all but "change talk: reflects it", 0.99 = 0.99, and "after sustain talk", 0.04 < 0.18) | Tables 3 and 6 |
| Figure 8 caption | affirmation at iteration 10: held-out K=5 0.067 vs K=0 0.009 (d_z −0.56, p_holm < .001); training oracle K=0 0.196 vs K=5 0.147 (p_holm .98); at the Base held-out open questions 0.1752, closed questions 0.2452, simple reflections 0.0198, information giving 0.2048 (printed 0.18, 0.25, 0.02, 0.20) vs training oracle 0.0873, 0.0394, 0.1373, 0.3779 (0.09, 0.04, 0.14, 0.38) | `shared_base.xlsx::k_process_paired`, `process_levels_<judge>` |
| Appendix D.1, D.8, Table 9 | the K=5 run's look-ahead sub-batch was 64 in iteration 1 and the first 30 of iteration 2's 104 steps, then 128 (the run metadata, rewritten on resume, records 128) | `Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md` (resume of `iteration_2` from `checkpoint-30`/104) |

Text-only fixes (no number): §3.2 and Algorithm 1 say the prompts of 5% of the conversations are
the trainer's validation split (`grpo_trainer.py::build_iteration_datasets` splits by conversation;
Table 9 now "0.05 of conversations"); §3.2's 86–89% and "rises slightly with prefix length" are
the training oracle's (held out at 12 utterances: 0.760 / 0.799), and Figure 9's caption ties the
86–89% to panel (a); C.1's "branch pairs" became conversations, and its pooled interval is described
as what it is (conversations resampled within iteration, ignoring between-iteration variation); §3.3 cites Table 10
and calls the three coders "coders" (MITI's counts left the paper on 2026-09-29); §4 cites Tables 4
and 7 and Figure 7 for its per-iteration and held-out claims; Figure 2/7 captions: the Base's SE
is over its 192 conversations; §5 cites Figure 3a and Table 5 (not Table 3) for iterations 8–9 and
"the two runs' final policies" replaces "the two methods"; MICI is no longer "the session-level
count of Table 1" (Table 1 gives it per therapist turn; Table 10's caption and D.2 now say §5 uses
its over-praise count); §6 points to Table 5 for its per-iteration tests and to Figure 10; the
Discussion's "iteration 9" is the model the candidates were sampled from; Figure 3(c) caption
"next patient utterance"; Figure 6 caption "second row block" and "Iteration 0 is the Base"; Table
8 caption defines praise/CT/ST; Appendix B cites Table 8; Figure 9 caption and C.1: conversation
pairs, prefix length, training iterations 1–10 = conversations of π_0–π_9, training reward always
the training oracle's, chance below the axis; D.5 renamed "The keyword marker"; Table 9 gains the
persona and sub-batch rows; Appendix F names CSQ-8, MI-SAT, WAI-SR and PCT and reads Table 11
along its rows. Figures: every PNG now prints at its include width (no text below 5.8 pt;
`save_at_width` in `render_paper_figures.py`; `render_schematic.py` sized to 0.82 textwidth);
Figure 1 says "within the group" and its update box matches Eq. 2; the levels-grid key star
matches the panels; Figure 3/8 legend "confront", panel (c) label clear of the titles; Figure 9
y from 0.65; Figure 10 (a) y to 0.95. The redraw made Figures 1, 3 and 4 taller (+7, +14, +15 pt);
with the text fixes the body now ends about 16 lines into page 11 (it ended on the last line of
page 10).

## 2026-10-05 — the four waiting items fixed (before Lior's read-through)
Text-only fixes (no number changed), each Lior's pick: the abstract's "97% of cases, against 80%"
now says "under the training oracle" (`ct_persist` at iteration 10, per conversation: K=5 0.968 vs
K=0 0.796, `lookahead/shared_base/tables/persist_levels_gpt-4o-mini.md`; held out 0.935 vs 0.734,
`persist_levels_claude-haiku-4-5.md`; both significant, `k_persistence.md`); §3.3's cross-judge
rule is scoped to questionnaire scores ("not on a common scale … compared only through contrasts
and never averaged"), so the praise-share comparison (K=5 0.05 vs 0.20 held out) no longer
contradicts it; §1 expands and cites DPO at its first mention (`rafailov2023dpo`, now in the
references); Appendix E.1 and `select_example_illustrative.py` now agree. The script ranked only
utterances 2–10 (a `range(…, 12)` cap) while E.1 said "every pair"; Lior asked why only the first
five, so the cap was removed: ranked over every reply, the top 34 pairs are identical, the first
reply past utterance 10 enters at 35th (persona 7, utterance 14, 17.64), and persona 84's
utterance 2 is still 7th at 21.5 (row "Ranking position" above stays true). E.1 now says every
reply was ranked, paired by position, earlier replies preferred (the score's −0.4 per later
reply). Float crowding moved to step 13 (the length pass will move every
float). The body still ends about 15 lines into page 11 (31 pages).

## 2026-10-05 — step 12: contributions, abstract, Q1+Q2 (drafted for Doron's read)
Run before Lior's read-through at his request, so Doron can read it first. No new number: every
value is one the paper already carried and this ledger already sources. **Contributions (§1):**
Doron's two original sentences restored verbatim as the frame ("behavior s" → "behaviors"), each
followed by one result sentence: all eight instruments under both judges (§4; Table 1), and the
utterance-coding account (§5–6), worded as in the approved 2026-10-01 rewrite. His "most
important" note marked handled; his original stays as a `%` comment. **Abstract:** rewritten to
the intro's framing (verifiable vs non-verifiable, the judge sees nothing after the turn) and cut
from 289 to 194 words (the ACL formatting guide: "The abstract should be no longer than 200
words"; whitespace count after stripping LaTeX). Kept: all eight instruments under both judges;
97% vs 80% persistence under the training oracle (`ct_persist`, block above); persuasion in reply
to sustain talk; single run. Dropped: "leads from the fourth iteration", the evaluation re-draw,
"2,112 conversations", the judges' disagreement on praise. **Q1+Q2:** §1 gains one sentence (PTO's
reward, Yosef et al.'s validated questionnaires; six instruments outside the reward + the
held-out judge check it — 8 − 2 = 6); the Discussion gains a short paragraph before the
Conclusion (why Q1+Q2; they rate the session from the patient's side, §3.3's description; another
instrument as reward is untested). §3.1's note marked handled with both locations. The body now
ends 30 lines into page 11 (was ~15; the visible note markers account for part of it).
Same day, Lior's read: "Our contributions are twofold" removed (his pick of three options): "We
make two contributions:" + two bullets (Doron's sentences unchanged otherwise; "First" dropped,
"Second, and more importantly" → "More importantly"); `enumitem` added to `main.tex` for a compact
list. Body ends ~32 lines into page 11.

## 2026-10-05 — step 11a: pre-read fixes (the recap + the direction check)
Planned with Lior by question round before his read-through; EDA first (commit `cde8a0a`). Every
new paper number below is read off a tracked EDA table.

| Claim (where) | Value in the paper | Source |
|---|---|---|
| K0 vs K5 update direction per training iteration, noise-corrected (App C.5) | 0.92–0.96 at 1–3 (0.9188 / 0.9256 / 0.9562); 0.80 at 4–5 (0.7979 / 0.8000); 0.57–0.74 at 6–8 (0.5672 / 0.7439 / 0.6982) | results/lookahead/mechanism/tables/direction_k_by_iter_grpo.md, `corrected` |
| K=5 direction's own split-half agreement (App C.5) | 0.32, 0.15, 0.20 at 8–10 (0.3211 / 0.1495 / 0.2016); 0.55–0.89 at 1–7 (0.5518 … 0.8946) | same table, `rel_b` |
| K=0 direction reverses (App C.5) | −0.66 at training iteration 9 vs 8 (−0.6569); −0.35 at 10 vs 9 (−0.3547); both cells' own agreement 0.77–0.93 | results/lookahead/mechanism/tables/direction_stability_grpo.md, GRPO_LA0 rows 8→9, 9→10, `corrected`, `rel_a`/`rel_b` |
| Pooled cosine kept (App C.5) | 0.804 / ceiling 0.945 / 0.851 | unchanged: results/arms/preference/tables/gpt-4o-mini/update_direction_cosines.md |
| "the one late round of training in which the K=0 reward barely favored praise" (§5 clause) | premium 0.070 (z 1.66) for candidates from the iteration-8 model, against 0.22–0.33 from the four models before and 0.165 after | results/lookahead/mechanism/tables/praise_premium_grpo.md (unchanged; App C.4 already quotes it) + the reversal row above |
| Malformed markers, training candidates (App D.6) | up to 58% of K=0's (0.5813, training iteration 7); K=5 below 2.5% (max 0.0238) | results/lookahead/mechanism/tables/marker_leak_training_grpo.md, `share` |
| Malformed markers, evaluation turns (App D.6) | K=0 56% at iteration 7 (721 / 1,292 = 0.558), 4% at 10 (43 / 1,128 = 0.038); K=5 below 1.5% (max 15 / 1,289 = 0.012, iteration 6) | results/lookahead/shared_base/tables/marker_leaks.md, `share_of_turns` |
| "neither favored nor penalized" (App D.6) | within-group correlation +0.05 (K=0 pooled 0.0521; K=5 −0.0062) | marker_leak_training_grpo.md, `all` rows, `r_within` |
| Held-out best-checkpoint lead "significantly" (§4) | dz 0.39 (0.386, p_holm .001) | row 93 above (k_endpoints, `GRPO_LA5_I10 − GRPO_LA0_I3`) |
| §4 prose to two decimals | +1.50 / +0.74 (1.502 / 0.738; the ratio 2.04 is computed before rounding); 4.08 / 3.75; dz 0.74 | the rows above, unchanged values |
| Limitations ICC "iterations 8 and 10" | ICC 0.92–0.99 | row 353 above (`GRPO_LA0_I8`, `GRPO_LA0_I10`) |

Notes. (1) The direction check (`meetings/2026-10-05_doron_direction_check/`) used gte-base and
Doron's best-minus-worst estimator; the EDA port uses the paper's own estimator (all-MiniLM-L6-v2,
advantage-weighted, every gradient group — all 237,221 weighted candidates were already in the
embedding cache) and 50 conversation split-halves. The two agree: gte noise-corrected 0.95 / 0.96 /
0.90 / 0.76 / 0.79 / 0.48 / 0.68 / 0.75 at 1–8 against MiniLM 0.92 / 0.93 / 0.96 / 0.80 / 0.80 /
0.57 / 0.74 / 0.70; K=0's 8→9 reversal −0.32 (gte) vs −0.66 (MiniLM). (2) The planned clause said
the iteration-9 reward "favored agreement". That did not survive a lexical check: over the groups
holding both kinds, replies with an agreement phrase ("you're absolutely right", "I couldn't agree
more", …) scored 0.12 within-group SD above the rest at training iteration 9 (z 1.87, not
significant) and an "I understand where you're coming from" family −0.05; the agreement reading
came from projecting sentences onto the direction, not from the reward. The clause therefore says
what is measured: the praise premium dipped and the direction reversed. (Scratch check, not an EDA
table.) (3) American spelling throughout the visible text (67 lines; comments, Doron's note texts,
verbatim prompts and transcripts untouched). (4) Abstract: "of a 1B-parameter therapist" added;
200 words by the step-12 count (the limit). (5) Build: 32 pages; the body ends ~42 lines into
page 11 (was ~32).

## 2026-10-05 — step 11, notes round 1: the Base, the best checkpoint, praise, every code
Lior's four questions after a recap ("do we show vs Base? tables for the utterance coding? only
iteration 10? the praise graphs with and without a model?") and his "implement 1-4", then "use
best iteration only on the training oracle (8 and 10)". EDA first (commit `232ef51`:
`shared_base.gains` now picks K=0's best checkpoint on the training oracle and scores that same
iteration under both judges, and gains a `p_holm` column, Holm across the nine rubrics within
judge × anchor × run). Every new paper number is read off a tracked table.

| Claim (where) | Value in the paper | Source |
|---|---|---|
| Table 1 Base columns, training oracle | 3.01 / 2.97 / 3.06 / 2.87 / 2.36 / 2.84 / 3.17 / 0.48 / 0.21 (Q1+Q2 3.01498, so 3.01; the 3.015 of `levels_long` is itself rounded) | results/lookahead/shared_base/tables/levels_long.md, `GRPO_LA0` iteration 0 (the shared Base, n = 192), unrounded value recomputed with `shared_base.levels` |
| Table 1 Base columns, held out | 1.85 / 1.69 / 2.01 / 2.19 / 2.09 / 2.48 / 1.86 / 0.52 / 0.36 (MICI 0.3551) | same table, `claude-haiku-4-5` |
| "Against the Base, each run's change is significant on every row … except the K=5 policy's on MICI under the training oracle" (Table 1 caption, §4) | every `last` row p_holm < .005 except gpt-4o-mini MICI GRPO_LA5: gain +0.0003, dz 0.002, p_holm 0.908 | results/lookahead/shared_base/tables/gains.md, anchor `last`, column `p_holm` |
| MICI vs the Base (§4) | K=5 0.21 against 0.21; K=0 0.84; held out K=5 0.63 (gain +0.273, dz 0.89), K=0 1.05 (+0.695), Base 0.36 — all "above it" rows significant | gains.md, metric MICI, anchor `last` |
| Held-out gain ratio vs K=0's best (§4, App B, README) | **1.33** (1.025 / 0.769 = 1.333; K=0 at iteration 8 held out 2.617 − 1.848); was 1.30 at the held-out judge's own pick (iteration 3). Range "1.3 to 2.5" unchanged | gains.md, `claude-haiku-4-5`, Q1Q2, anchor `best_K0`, `ratio_K5_over_K0` |
| Best checkpoint: K=5 at 10 vs K=0 at 8, training oracle, all nine rows (§4) | all significant (every primary_p_holm < .001); Q1+Q2 dz 0.74 (0.743); \|dz\| 0.52–1.13 (MITI 0.517 … MICI −1.129, favours K=5) | results/lookahead/reward/tables/k_endpoints.md, pair `GRPO_LA5_I10 − GRPO_LA0_I8 (K=0 best by primary Q1Q2)`, `primary_*` |
| Same pair, held out (§4, App B) | seven of nine significant; Q1+Q2 dz 0.38 (0.384, p_holm < .001); not Q2 (dz 0.182, p_holm .172) or WAI-SR (dz 0.097, p_holm .578); MITI is the weakest that clears (dz 0.289, p_holm .048) | same pair, `judge_*` |
| App B: "on its own Q1+Q2 the K=0 run peaks at iteration 3 (2.64, against 2.62 at iteration 8)" | 2.6366 / 2.6172 | levels_long.md, `claude-haiku-4-5`, Q1Q2, GRPO_LA0 |
| Table 6 (new, App A): every code at the Base and iteration 10, both judges, dz K5 − K0 | the printed cells, by `render_process_tables.py --codes` (which asserts the praise, complex-reflection and persuasion rows equal Tables 3 and 7, levels and dz); held-out simple reflection dz undefined (both runs 0.000 in every conversation) | results/lookahead/shared_base/tables/shared_base.xlsx, sheets `process_levels_<judge>` (`th_<CODE>_rate`) and `k_process_paired` (iteration 10, `dz` negated, stars from `p_holm`) |
| Table 6 caption: the judges code differently | Base simple reflection 0.137 vs 0.020, closed question 0.039 vs 0.245; K=0 at iteration 10 affirmation 0.196 / praise 0.407 (training oracle) vs 0.009 / 0.762 (held out) | the table's own cells |
| Figure 5 (redrawn, App A): praise three ways | coder praise share K=0 0.222 / 0.080 / 0.407 (training oracle) and 0.500 / 0.238 / 0.762 (held out) at iterations 8 / 9 / 10; keyword marker 0.275 / 0.094 / 0.671; K=5 at or below 0.07 on the training oracle (max 0.070, iteration 6) and the marker (max 0.064), held out rising to 0.203 | `process_levels_<judge>` `th_PRA_rate`(+`_se`); `marker_and_length` `lex_overpraise_marker_rate`(+`_se`) |

Notes. (1) The best-checkpoint rule (Lior): K=0's best is chosen once, on the training oracle's
Q1+Q2 (iteration 8), and scored under both judges; picking it on the held-out judge would select
on the evaluation itself. This replaced the held-out judge's own pick (iteration 3) in §4
("dz 0.39" → "dz 0.38", the same size by coincidence) and in Appendix B (ratio 1.30 → 1.33).
Appendix D.8's equal-GPU-hour comparison still chooses within budget "by the same judge"; not
changed (a different selection, within a budget), raised with Lior. (2) The `k_endpoints` table
still carries the iteration-3 pair; the paper no longer cites it. (3) Table numbers from Table 6 on
moved by one (held-out process 6 → 7, held-out scores 7 → 8, held-out process at every iteration
8 → 9, configuration 9 → 10, instruments 10 → 11, agreement 11 → 12); earlier ledger blocks keep
the numbers of their day. (4) Build: 33 pages; the body ends ~57 lines into page 11 (was ~42).

## 2026-10-05 — step 11, notes round 2: every float made readable (the clarity audit)
Lior: "go over the tables and figs and make sure they are understandable (for example in table 3
you bold the best of k=0 and k=5 and ignore base, and the dz in not understood if it is k=5 vs k=0
or with the base also)". A multi-agent audit (per float group: a source-aware auditor and a cold
reader who saw only the PDF; every proposed fix checked by a reader lens and a fact-and-style lens;
one synthesis for consistent conventions) produced 61 edits, applied with the over-long appendix
captions trimmed by hand. The conventions they implement: every K=5 − K=0 column sits under a
"K=5 − K=0" header (Tables 1, 3, 6, 7) with the judge's name over Base / K=0 / K=5 only, so the
Base is visibly outside the contrast; signed d_z everywhere; every caption says what bold means
(best of a column over all 21 states, Base included, in Tables 4, 5, 8, 9; the better run where
d_z is starred in Tables 3 and 7, "even if the Base is better"), that a star never involves the
Base, and that the appendix figures mark no test (pointing to the tables that do). Numbers new to
the paper:

| Claim (where) | Value in the paper | Source |
|---|---|---|
| Base conversations with a therapist turn after the scripted opener (Tables 5, 6, 7 captions) | 184 of 192 | results/lookahead/shared_base/tables/text_diversity.md, iteration 0, `n_convs` (already Figure 4's 184; the coder's per-turn shares are NaN on the same 8 conversations) |
| Table 3 caption: paired personas for the reply / next-utterance rows | 76 (change talk), 65 (sustain talk) — the old range "65–76" | shared_base.xlsx `k_process_paired` / `k_persistence`, gpt-4o-mini, GRPO, iteration 10, column `n` |
| Table 7 caption: the same, held out | 80, 64 — the old range "64–80" | same sheets, claude-haiku-4-5 |
| Figure 5 caption: iterations at which the praise contrast is significant | training oracle 8 and 10; held out 5, 6, 8–10; K=5 lower at each | `k_process_paired`, metric `th_PRA_rate`, `p_holm` < .05, `better` = K5 (the starred praise cells of Tables 5 and 9) |
| App D.7: resample counts | dispersion ratios 2,000; faithfulness intervals and Figure 9's bands 1,000 | results/lookahead/mechanism/tables/CAPTIONS.md (`dispersion_ratios`; `faithfulness_*`, B=1000) |
| Table 10: look-ahead temperatures 0.9 / 0.7 (therapist / patient) | 0.9 / 0.7, the same as the conversations' | `LOOKAHEAD_TEMP_THERAPIST` / `LOOKAHEAD_TEMP_PATIENT` in cell 1 of code/GRPO_Exp3/train_GRPO_Iterative.ipynb; not in `run_metadata.json` (checked: it records `temperature_therapist_gen`, `temperature_patient`, `max_tokens_per_response`, `grpo_temperature`, `lookahead_k`, `lookahead_sub_batch_size`) |
| Table 10: "64 × 2 = 128 completions (16 prompts)" | 128 / G = 8 → 16 | Table 10's own row and §3.2 |

Also: Table 6's caption no longer claims a turn-level mapping (that the held-out judge labels "as
praise much of what the training oracle labels affirmation"); it says what the table shows, more of
K=0's turns as praise and fewer as affirmation (decision D-table6-turn-claim (a); a raw cross-tab
found giving information is the larger source, untracked so not quoted). Figure 3/8's simple
reflection colour #9ecae1 → #6baed6 (it printed as the same grey as closed question). The bootstrap
row of the conventions table above was corrected (it claimed 2,000 resamples everywhere). Build: 33
pages; the body ends ~61 lines into page 11 (Table 3's new header row and Figure 3's panel-(d)
note).

### Lior's picks on the audit's decisions (same day, by question round)

- **Bold in Tables 3 and 7 = each row's best of the Base, K=0 and K=5 (ties at three decimals all
  bold)**, the rule of the complete tables; significance stays on the d_z stars. Newly bold: the
  Base's praise and MI-inconsistent share in both tables (0.041 / 0.248; 0.038 / 0.214), K=5's
  "sustain talk: reflects it" 0.368 and "after sustain talk" 0.326 (Table 3) and 0.238 (Table 7),
  and the held-out Base's 0.011 on "sustain talk: praises" (tied with K=5's 0.011: 0.0111 vs
  0.0107). `render_process_tables.py` now asserts this rule and the iteration-10 d_z cells.
- **Praise stated within each judge** (no cross-judge level in §5, App B, Figure 8 or the
  Discussion): under the held-out judge K=5's praise share rises from the Base's 0.04 (0.038) to
  0.20 (0.203) at iteration 10 — `process_levels_claude-haiku-4-5`, `th_PRA_rate`. App B's opening
  rule scoped to the instruments; Figure 8's Base code-mix numbers replaced by a pointer to Table 6.
- **Figure 4's praise link**: "from iteration 3 on, long before its praise peaks" — K=0's
  between-conversation variance share 0.322 → 0.200 at iteration 3 (`text_diversity`,
  `persona_var_share`, GRPO_LA0), while all three praise measures peak at iteration 10 (Figure 5).
- **Table 12 / App F**: "The disagreement is real" → no test attached, the evidence is rank and
  distance below median. K=0 at iteration 10: MICI r **0.204**, lower than K=5's 0.287; the other
  seven near their medians (MITI 0.645 vs 0.666, Q1 0.856 vs 0.842, Q2 0.806 vs 0.752, CSQ-8 0.877
  vs 0.888, PCT 0.949 vs 0.956, MI-SAT 0.947 vs 0.930, WAI-SR 0.919 vs 0.920) —
  results/lookahead/shared_base/tables/agreement_by_state.md, rows `GRPO_LA0 it 10`, `pearson_r`;
  medians from Table 12. App F no longer lists MICI among the K=5-specific falls.
- **Table 3 defines "reflects"** (a simple or complex reflection; `process.py` REFLECT = {SR, CR});
  one body line.
- **Table 2 and Appendix E.1 re-picked (supersedes the NEW-0914b persona-84 rows above).**
  `select_example_illustrative.py --coder` keeps only the (persona, therapist turn) pairs at
  iteration 10 where BOTH judges' coders label the K=0 reply non-specific praise (PRA) and the K=5
  reply a complex reflection (CR): **28 pairs**. Its lexical ranking (unchanged) puts **persona 87,
  utterance 2** first (score 17.5; next 15.58); it is also the only one of the 28 whose K=5 reply
  answers sustain talk (both coders code K=5's utterance 1 ST; K=0's is NEU under the training
  oracle and ST held out — hence §6's "the kind of reply Table 2 shows", no longer "the reply of
  Table 2").

| claim | value | source |
|---|---|---|
| Persona 87 | Female, 61, Smoking, ManyYears, tried ManyTimes, StartLowAndChangesToHigh ("a 61-year-old woman who has smoked for many years, has tried to quit many times and was sent to therapy") | `select_example_illustrative.py --persona 87` (`data.canonical_personas()`) |
| Files | both arms `conversation_20.csv` at iteration 10 | same dump |
| Lengths | K=0 **32 utterances / 16 therapist turns**; K=5 **18 / 9** (counts include the opener) | same |
| Utterance 1 differs across arms | sampled anew; Table 2 shows both, each elided with […] | same |
| Q1+Q2 at iteration 10 | training oracle K=0 **4.306** (4.31) vs K=5 **4.376** (4.38); held out **2.235** (2.24) vs **3.753** (3.75) | score lake via the script, `Q1Q2`, persona 87 |
| Codes | K=0 reply PRA / PRA; K=5 reply CR / CR (training oracle / held out) | MIPROC, `th_codes` position 1 |
| Cap | the K=0 reply ends mid-phrase ("every step of"): the 200-token cap; the K=5 reply is complete (614 chars) | the dump |
| Appendix E.1 | utterances 1–7 of both conversations, generated from the dump (curly apostrophes → ', em dash → ---, straight and curly double quotes typeset, paragraph breaks inside an utterance → `\newline`) | scratch `make_e1.py`, not tracked |

Build: 34 pages (Appendix E.1 is longer); the body ends ~70 lines into page 11.

## 2026-10-06 — the two questions left open by round 2

**Appendix D.8: equal-GPU-hour checkpoints chosen by the training oracle only** (Lior: "yes, also
D.8 choose best checkpoint using training oracle only"). No EDA change: the cross-judge sweep
already selects on one judge and scores on the other. Source:
results/compute/cost/tables/budget_sweep_crossjudge.md, contrast `GRPO_K`, `select_judge` =
gpt-4o-mini, `eval_judge` = each judge, `select_metric` = `eval_metric` = Q1Q2 (paper sign K=5 −
K=0 = the table's `mean_delta`/`dz`, arm_a = LA5).

| Claim (App D.8) | Value | Row |
|---|---|---|
| ~13 GPU-h, training oracle | dz **−0.74** (−0.742), p_holm < .001; K=5 iteration 2 (13.27 GPU-h) vs K=0 iteration 4 (10.75) | `budget_gpu_h` 13.27, eval gpt-4o-mini (unchanged) |
| ~13 GPU-h, held out | dz **−0.49** (−0.489), p_holm < .001; same checkpoints — was −0.78 when the held-out judge picked K=0's iteration 3 | eval claude-haiku-4-5 |
| ~23 GPU-h, training oracle | dz **0.07** (0.074), p_holm .814, not significant; K=5 iteration 4 (23.21) vs K=0 iteration 8 (22.28) | eval gpt-4o-mini (unchanged) |
| ~23 GPU-h, held out | dz **0.27** (0.266), p_holm .035, significant; same checkpoints — was 0.33 with K=0's held-out pick (iteration 3) | eval claude-haiku-4-5 |
| "beyond 27.9 GPU-h the comparison is §4's best-checkpoint one" | K=0's best within any budget ≥ 27.9 is iteration 8 for both judges now, as in §4 | same table, `best_iter_b` = 8 |

The Limitations sentence keeps its verdicts (K=0 ahead at ~13 GPU-h; level under the training
oracle and K=5 ahead held out at ~23); the superseded held-out rows are NEW lines 498–499 above.

**MI-SAT's provenance** (Lior: "momi sent it to my email with some details (Aug 2025)"). Gmail is
not reachable from here; the details are on Drive as `Satisfaction Survey.docx` (2025-08-25, a
second copy 2025-09-24): "Satisfaction was evaluated with a 6-item survey on how helpful,
enjoyable, interesting, easy to use, worth the time spent and likelihood of changing behavior
(Appendix G). Items were rated on a 5-point Likert-type scale … (1 = not at all; 5 = very much).
The survey was created by the PI for this study and therefore has not undergone psychometric
analysis." It does not name the study. Appendix D.2 now says that much and lists the six items
(from `MI_SAT_ITEMS` in Exp3_PTO_GRPO/code/questionnaires.py); Table 11's source cell stayed
"adapted, this work" until the study's reference was known. Settled the same day from Lior's
copy of the email (Momi Zisquit, 2025-08-18, to Lior and Doron): she attached "the two
questionnaires I found in my files regarding the study we did with Stav. We didn't end up using
them in Stav's studies." So the survey is unpublished and there is nothing to cite: Table 11 says
"unpublished survey, adapted", Appendix D.2 "shared with us by a colleague" (anonymous for
review), and the camera-ready Acknowledgements thank her (README, Before submission).

## 2026-10-06 — step 11, notes round 3 (first batch: coder sources + two must-fixes)

Lior's picks on the round-3 page (https://claude.ai/artifact/F8t9EzUh5GA6exTRgDkEMz): apply the coder
sources and the two must-fixes first; Figure 9 re-score, the persona-21 example and the embedding
appendix follow. No number changed in this batch.

| Edit | Where | Source (checked against the primary document) |
|---|---|---|
| PCT = MISC 2.5 client categories; CT/(CT+ST) = MISC "Percentage Client Change Talk (%CT)" | Table 11 Source cell; App D.2; §3.3 one sentence | Houck et al. MISC 2.5 PDF, p. 38 (three mutually exclusive client categories) and p. 48 (%CT); casaa.unm.edu/assets/docs/misc25.pdf. Undated PDF; 2010 is the conventional year |
| %CT relates to outcomes | App D.2 | Magill et al. 2018, JCCP 86(2):140-157, doi 10.1037/ccp0000250 (abstract: "higher proportion change talk was related to reductions in risk behavior at follow up") |
| DARN-CAT incl. Activation; importance/confidence/readiness rulers | App D.2 | Miller & Rollnick 2013 (already cited), via the Guilford MI-3 glossary |
| MICI: confront, advise w/o permission, warn, direct = MISC MI-inconsistent codes; raise concern w/o permission unused; cheerleading coded as confront | App D.2 | MISC 2.5 pp. 19-20 (cheerleading as Confront), pp. 47-48 (MIIN) |
| Our confront also names warning, so one act can count twice | App D.2 | questionnaires.py `MICI_BEHAVIOR_ITEMS["MICI_Confront"]` |
| Coders never get the target behavior | App D.2 | `change_goal` is passed nowhere in eda_analysis/scoring, eda/tools or code/_shared (grep) |
| Utterance coder = MITI 4.2.1 + MISC 2.5 (open/closed split, direct, sequential coding of both speakers; client codes); one code per turn as AnnoMI | §3.3; Table 11; App D.3 | MITI 4.2.1 E.4.d ("Closed and open questions are not differentiated"), E.4.f.2 (non-specific praise not coded), E.4.a.1 (structure not GI), E.4.g.2 (confront incl. warning, moralizing); AnnoMI ICASSP 2022 ("the annotator is required to choose the main behaviour") |
| MITI 4 peer-reviewed paper beside the manual | §3.3; Table 11 | Moyers, Rowell, Manuel, Ernst & Houck 2016, J Subst Abuse Treat 65:36-42, doi 10.1016/j.jsat.2016.01.001 (the 4.2.1 manual's pages read "Draft: Do not cite without permission") |
| Must-fix: "At K=0 the oracle scores the candidate alone" contradicted Eq. 1 | §3.2 | now "the conversation so far, ending on the candidate, with nothing after it" |
| Must-fix: CollabLLM and RLHS cited | §2 (Delayed credit) | CollabLLM: PMLR 267:67260-67283 (proceedings.mlr.press/v267/wu25i.html); multiturn-aware reward forward-samples w = 1-3 turns with a user simulator; PPO and offline/online DPO. RLHS: arXiv 2501.08617 (v3 June 2025, no venue) |

## 2026-10-06 — step 11, notes round 3 (second batch: Appendix E.2 = a persona from a seeded draw)

Lior's pick: Table 2 / E.1 stay persona 87; E.2 becomes persona 21, utterances 1-4, chosen from a
seeded random draw (replacing the median-rule persona 93). Source: `select_example_random.py`
(`numpy.random.default_rng(20261006).choice(96, size=8, replace=False)` = 93, 34, 51, 89, 73, 65, 29, 21).
The pick: three independent judge lenses (MI coder, NLP reader, skeptical co-author) all ranked persona
21 first among the eight (round-3 decision page).

| Claim (E.2) | Value | Source |
|---|---|---|
| persona 21 | Male, 61, smoking many years, never tried, cooperation High | `canonical_personas()` |
| conversation lengths | K=0 20 utterances (10 therapist turns), K=5 50 (session cap) | conversation_84.csv in each run's model_iter_10 |
| Q1+Q2, training oracle | K=0 4.91, K=5 5.00 (diff +0.09, rank 66 of 96) | score lake, iteration 10 |
| Q1+Q2, held-out judge | K=0 2.98, K=5 3.42 (diff +0.44, rank 59 of 96) | same |
| "below the median" | medians of the 96 differences: +0.53 training oracle, +0.64 held out | same |
| codes, utterances 2 and 4 | K=0: PRA/PRA, AF/PRA; K=5: PERS/CQ, CR/CR (training oracle / held out) | MIPROC th_codes, position = utterance // 2 |
| "persuasion is K=5's most frequent code at iteration 10 under both judges" | 0.279 vs complex reflection 0.230 and giving information 0.227 (training oracle); 0.268 vs 0.242 (held out) | Table 6 (`tab:codes`) |
| first patient utterance identical in both conversations | yes, verbatim | the two CSVs |

## 2026-10-06 — step 11, notes round 3 (third batch: the update direction in five encoders, App C.5 + §5 clause)

Lior's pick: "Appendix + one body clause". Source: the EDA's `lookahead/mechanism` family, section 1d
(`eda_analysis/encoders.py`, caches by `eda/tools/embed_encoders.py`), tables under
`Exp3_PTO_GRPO/eda/results/lookahead/mechanism/tables/` (all also in `mechanism.xlsx`). Estimator = the
paper's (advantage-weighted, sum |w| = 2 per group, all train candidates); the MiniLM rows equal
`direction_k_by_iter_grpo.md` / `direction_stability_grpo.md` exactly (max |diff| 4.7e-16).

| Claim | Value | Table |
|---|---|---|
| K0 vs K5 corrected, iterations 1-3, five encoders | 0.797 (qwen3, it. 1) to 0.971 (gte, it. 2) → "0.80--0.97" | `direction_encoders_k_by_iter_grpo` |
| furthest apart before 9: iteration 6 | 0.428 (qwen3) to 0.766 (llama8) → "0.43--0.77"; iterations 4-8 otherwise ≥ 0.62 | same |
| K0 reverses at 9 (vs 8) | −0.728 (qwen3) to −0.369 (gte) → "−0.73 to −0.37" | `direction_encoders_stability_grpo` |
| K5 split-half 8-10 | llama8 0.352-0.690; the other four 0.055-0.399 → "0.35--0.69 … 0.06--0.40" | `rel_b` of the k_by_iter table |
| pool | 4,045 sentences (4,178 seen ≥ 3 times, minus marker sentences incl. `\|im` fragments) | notebook 1d print |
| K0 praise, iterations 4-8 and 10 | z_min 0.57, 1.05, 1.05, 0.87, 1.15, 1.11; all five intervals above 0; iteration 9: mean −0.27, none above 0 | `direction_encoders_categories_summary_grpo` |
| K5 advice through 5, questions at 6 | advice intervals above 0 in 5,5,5,5,4 encoders (it. 1-5); question 5 at it. 6 | same |
| K5 praise | mean −0.26 to 0.40 (it. 1-6), 0.59 to 0.88 (7-10); max 3 of 5 intervals above 0 | same |
| sentences in ≥ 3 encoders' top 5 (K0−K5) | it. 3: "I'm thrilled to support you on your weight loss journey." / "What could you focus on instead?"; it. 5: "Fantastic, I'm so proud of you!" / "What if we focus on the bigger picture?"; it. 10: thanks for honesty/vulnerability/trust in minilm, gte, mxbai, qwen3, "You are amazing, and I'm so proud of you" in llama8 | `direction_encoders_top_sentences_grpo` |
| words (clean: no leaked marker, no degenerate text; 22.8% of K0's, 4.9% of K5's candidates dropped) | K0 4-10: you 9.0, i 8.7, so 7.0, proud 6.6, every 6.6, i'm 6.2, step 5.8; K0 it. 10: me 5.3, my 4.0, myself 4.0, admire 3.8, unwavering 3.7, courage 3.6; K5 1-3: yourself 5.5, cravings 4.6, small 4.2, behavior 4.1, losers i −5.3, me −3.6; K5 4-10: healthy 3.3, quit 3.2, exercise 3.1, schedule 2.9 | `direction_lexical_logodds_clean_grpo` |
| §5 clause "in every other round from the fourth on … five sentence encoders" | training iterations 4-8 and 10 (the dip round is 9) | categories summary |

## 2026-10-06 — step 11, notes round 3 (fourth batch: Figure 9 = the Base prefix scored alone from 2 utterances)

Lior's pick: re-score the Base's prefixes (he gave the go; 5,444 gpt-4o-mini calls, `eda/tools/score_partial.py
--run`, all landed, 0 errors). Source: `lookahead/mechanism/tables/faithfulness_prefix_alone_grpo.md` (pooled
over the two Base draws) and the `train_iter_1` GRPO rows of `faithfulness_matched_policy_long.md`.

| Claim | Value | Table |
|---|---|---|
| prefix alone, training oracle: 2 / 10 / 12 / 50 utterances | 0.735 [0.690, 0.774] / 0.803 / 0.846 / 0.976 → §3.2 "73% … 85% … 98%" (was 74%, rounded twice from 0.7346; corrected in the content pass) | prefix_alone, judge gpt-4o-mini, pooled |
| prefix alone, held-out judge: 2 / 12 / 50 | 0.688 / 0.782 / 0.838 | judge claude-haiku-4-5 |
| 2,722 prefixes, 192 conversations | 1,373 (LA0 Base) + 1,349 (LA5 Base); n_convs 192 at n_turns 2 | score_partial dry run; table |
| iteration-1 training reward at 12 / 50, training oracle | K0 0.846 / 0.977, K5 0.831 / 0.963 | matched_policy_long, cut train_iter_1 |
| pooled 1-10 at 12 / 50 (unchanged, now prose only) | K0 0.860 / 0.897, K5 0.886 / 0.936 | faithfulness_curve_long |

## 2026-10-06 — step 11, notes round 3 (batch A text: cooperation, WAI subscales, trajectory test, Base draws, trainer logs, coder kappa)

Lior's picks: cooperation + WAI-SR, trainer logs, trajectory test + noise floor → appendix + short body
mentions; inter-judge kappa → App D.3 + one Limitations clause; length-adjusted contrasts → NOT in the paper
(EDA only: `lookahead/behaviour/tables/length_adjusted_k*.md`). Analyses verified by an independent agent
(scratch `batchA/verify/`); EDA commit `1d71a8f`. ⚠ The .md tables print 4-dp-rounded values at 3 dp and
can be off by one in the 3rd decimal (e.g. 0.21751 → "0.217"); quote from the xlsx/ledgers.

| Claim | Value | Source |
|---|---|---|
| Table 7 (`tab:coop`) cells | every Δ (dz) and Holm star | `shared_base/tables/coop_strata` (xlsx), anchors `last` (10 v 10) and `best_K0` (K5 10 v K0 8); generated by scratch `make_coop_table.py` |
| cooperative Q1+Q2 vs 8, training oracle: dz shown as † | both runs 100% of conversations ≥ 4.5 (share_K0_ge = share_K5_ge = 1.000) | same |
| "91% and 100% at 4.5 or above" (10 v 10 cooperative) | 0.906 / 1.000 | same |
| over-praise with cooperative / resistant at it. 10, K0 | 0.937 / 0.484 acts per therapist turn; K5 max 0.075 | `coop_strata_overpraise` |
| WAI-SR gains over the Base, training oracle | K5 Task +0.958, Goal +0.849, Bond +0.760; K0 Task +0.573, Goal +0.490, Bond +0.633 | `wai_subscale_gains` |
| WAI-SR K contrast at 10 (dz) | Task 0.509 / 0.681, Goal 0.573 / 0.531, Bond 0.264 / −0.057 (training oracle / held out) | `wai_subscale_k` |
| trajectory test 1-10 | Q1+Q2 dz 0.921 / 1.128; |dz| 0.661-1.044 (oracle), 0.543-1.256 (held out); all p_holm < .001 | `shared_base/tables/k_trajectory` window 1-10 |
| MICI over 1-7 favors K0, training oracle | dz +0.378, p_holm .002 | same, window 1-7 |
| Base draw 1 vs 2 | max |dz| 0.128 (oracle) / 0.147 (held out) on 9 instruments; 0.180 / 0.240 on 12 process measures; no raw p < .05 | `base_draws_summary` |
| trainer logs (Table 11, `tab:trainer`) | KL ×10³, grad norm, entropy per iteration; medians KL 0.0160 / 0.0059, grad 0.387 / 0.196; K0 KL and grad norm higher 10/10; K5 entropy higher 8/10 | `mechanism/tables/trainer_diagnostics_table_grpo`, `_k_contrast` |
| therapist kappa between judges | 21 states 0.078-0.260, median 0.201; Base 0.188; K0 it10 0.078; K5 it10 0.152 | `process/tables/judge_agreement_overall` (positions = policy turns, opener excluded) |
| per-code kappa, all states | PRA 0.236, CR 0.270, PERS 0.253, AF+PRA merged 0.407 | `judge_agreement_codes` scope all states |
| patient kappa | 21 states 0.411-0.740, median 0.603; CT 0.711, ST 0.696 | same, all patient turns |

## 2026-10-06 — step 11, notes round 3 (batch B: reproducibility)

Lior's pick: "Yes, all six". Values read (read-only, no API calls) by an independent agent from the
runs' checkpoints, the code and the generation logs; scratch dumps in the session scratchpad
`repro/` (`training_args_dump.json`, `gen_audit.json`), not tracked. Table 12 (`tab:config`) became a
two-block `table*` — as a one-column list it was 88 pt taller than a page.

| Claim | Value | Source |
|---|---|---|
| Table 12 optimizer / schedule / clipping rows | AdamW fused, β (0.9, 0.999), ε 1e-8, weight decay 0; LR 1e-5 cosine; warm-up 1–2 steps (`warmup_steps` = ⌈1% of the iteration's steps⌉: 2, or 1 in K=0 iterations 2 and 9 and K=5 iterations 5 and 6); `max_grad_norm` 1.0 | `training_args.bin` of all 38 saved checkpoints under `data/grpo_Exp3/runs/full/GRPO_Iterative_Q1Q2_Llama32-1B_LA{0,5}_MCL12_G8/iteration_*/training/checkpoint-*/`; identical except `warmup_steps` |
| Table 12 GRPO rows | `loss_type` grpo, `epsilon` 0.2, `num_iterations` 1 (clip inactive), `beta` 0.01, `scale_rewards` group, `importance_sampling_level` token, `temperature` 1.2, `top_p` 1.0 / `top_k` 0 (none), `mask_truncated_completions` False (kept in the loss) | same |
| advantage `(r − mean)/(std + 1e-4)` | TRL 1.4.0 GRPOTrainer with `scale_rewards="group"` | pinned TRL source |
| LoRA dropout 0.05, all attention + MLP projections | `adapter_config.json` of each checkpoint | same checkpoints |
| therapist top-p 0.9 / top-k 50 | top-p 0.9 from the base model's `generation_config.json`, top-k 50 the transformers default; neither is overridden at the conversation / look-ahead decode sites | `_shared/convs.py`, `_shared/reward.py` decode calls |
| training oracle T 0, max 256 tokens, strict schema; evaluation primary T 0.1 seed 42; held-out default temperature, no extended thinking; both max 1,024 tokens | as stated (training: no seed, 60 s × 3 attempts) | `_shared/reward.py` (training); `eda/eda_analysis/scoring/registry.py` (T 0.1), `pipeline.py` (seed, attempts), `judge.py` (held-out: no temperature or thinking sent, max 1,024) |
| empty candidate → reward 0 (oracle range 1–5) | `REWARD_FLOOR = 0.0`; 29 of 130,688 logged K=0 candidates, 28 of 121,088 K=5 | `grpo_trainer.py`; per-iteration `generations.jsonl` (count of floored-empty candidates) |
| six iterations with partial generation logs | K=0 iterations 2, 6, 8; K=5 iterations 1, 2, 7 (log coverage < 1) | same; `tail_audit_by_iter` `log_coverage` for K=5 |
| KL-reference exception | K=5 iteration 1 crashed at step 54, resumed from a checkpoint without the saved reference; steps 55–108 regularized toward the step-54 policy | the iteration's checkpoints + the trainer's resume path |
| rollouts ending neither full nor patient-closed (A_tables) | 2.54% = 3,081 of 121,088 (2,019 no tail + 1,010 ending on a therapist turn + 52 empty therapist turn); per iteration 5.6, 4.9, 3.4, 1.6, 1.7, 3.6, 2.1, 2.3, 1.9, 1.1% | `results/lookahead/mechanism/tables/tail_audit_by_iter.md`, `tails_numbers.json` (GRPO_LA5) |
| look-ahead patient call attempted three times | back-off 1 s then 2 s, on top of the SDK's own retries; after the last failure the rollout freezes and the oracle scores the transcript so far | `_shared/convs.py` `generate_patient_response_async`; freeze in `_shared/reward.py` |
| Q1 anchors quoted in D.2 | motivation item: "suggests practical steps to achieve the patient's goal, provides uplifting messages, or encourages perseverance"; relevance item: "provides advice or information that directly relates to challenges or tasks that the patient faces regularly" | `code/questionnaires.py` `get_questionnaire_1` (lines 284, 292) |
| D.8 call counts | read from the generation logs, scaled to the optimizer-step count in the six partial-log iterations (the earlier "exact counts" wording was wrong) | `compute/cost` tables |

## 2026-10-06/07 — PCT rebuilt from the utterance coder (Lior: "rebuild pct from the coder")

The PCT call and the utterance coder's patient side were the same judge sorting the same patient
turns into change / sustain / neutral; per-conversation Spearman of the two CT/(CT+ST) ratios 0.92 /
0.96 (training oracle / held-out). **PCT is now CT/(CT+ST) over the coder's patient codes** in the EDA
(`constants.pct_from_coder`; EDA commit of the same date; `history/CHANGELOG_EDA.md`) and the paper.
The PCT call stays only as the coder's check (App D.3). The coder was run on the two replicate draws
for this (384 calls, ~$1, 2026-10-06; one held-out call re-scored). **Every PCT number above this
block that predates it is the PCT call's and is superseded** (e.g. the "PCT contrast now quoted in §6"
row: +0.111 / dz 0.516 → now +0.150 / 0.515; the 1185–1187 K=0 agreement list: PCT 0.949 → 0.888).
Values below are unrounded EDA reads (scratch `pct/paper_pct_rows.py`, `pct/agreement_unrounded.py`);
the `.md` tables round 4-dp values to 3 dp and can be off by one.

| Claim | Value | Source |
|---|---|---|
| Table 1 PCT row | training oracle Base 0.491 / K=0 0.583 / K=5 0.732, Δ +0.1495, dz +0.515, p 2.05e-6; held-out 0.500 / 0.580 / 0.738, Δ +0.1572, dz +0.636, p 4.5e-8; every Table 1 row still p_holm < .001 (Holm across nine rows) | `lookahead/shared_base` `levels_long`, `k_contrast` (iteration 10) |
| vs Base, PCT | K=0 +0.092 (p_holm .0002) / held-out +0.080 (.0005); K=5 +0.241 / +0.237 (< 1e-11) — "each run improves on every row but MICI" holds | `gains` |
| Tables 4 / 9 PCT column | 21 cells each (3 dp, unrounded source); stars: training oracle iterations **6–10** (was 4–10; it 4 p .85, it 5 p_holm .47), held-out 4–10 (unchanged); bold K=5 it 9 (0.736 / 0.740) | `levels_long`, `k_contrast`, `significant_iterations` |
| "six of the eight separate by iteration 6" (§4) | holds: PCT's first Holm-significant iteration is now 6 | `significant_iterations` |
| vs K=0 at 8 (§4, App B) | PCT dz 0.558 (oracle) / 0.546 (held-out), inside "0.52–1.13"; held-out "all but Q2 and WAI-SR" holds | `coop_strata` All rows, anchor `best_K0` |
| whole-run test (App A) | PCT dz 0.786 / 0.954, inside 0.66–1.04 / 0.54–1.26 | `k_trajectory` window 1-10 |
| Base draws, max \|dz\| | training oracle 0.128 (MITI, unchanged); held-out **0.169 (PCT)**, was 0.147; min raw p .06 → "no row p < .05" holds | `base_draws_summary` |
| Table 7 PCT block | cooperative Δ +0.03 / 0.00 / −0.01 / −0.01 with dz not shown (‡: ≥ 90% of both runs' cooperative conversations at PCT = 1: shares 0.938/0.969, 0.969/0.969, 1.000/0.906, 1.000/0.906); warms up +0.23*** (1.07), +0.28*** (1.16), +0.21*** (1.29), +0.22*** (1.01); resistant +0.19* (0.46), +0.18* (0.49), +0.27*** (0.78; p_holm 0.00098), +0.20* (0.58) | `coop_strata` (unrounded; `share_K*_ge` now covers PCT at 1.0) |
| App F sign agreement | **1,477 of 8 × C(21,2) = 1,680 (87.9%)**; 373 of 377 at \|Δ\| ≥ 0.50 unchanged | `sign_preservation` |
| Table 14 PCT row | r 0.930, median 0.950, −0.020, rank 5/21 (rows re-sorted: PCT last); MI-SAT r corrected 0.905 → **0.906** (0.90554, a double-rounding slip) | `agreement_by_state` (unrounded) |
| Table 14 caption, K=0 it 10 | MICI r 0.204; **PCT r 0.888, its lowest of the 21**; the other six near their medians | same |
| Second draw (§4) | max \|dz\| still 0.174 (MICI, training oracle); PCT −0.107 / −0.038; none significant; Q1+Q2 dz 0.92 vs 0.91 (held out 0.95 vs 1.03) unchanged | `results/measurement/replicate_draw.md` (rebuilt by `tools/replicate_check.py`) |
| PCT undefined (App D.2, captions of Fig 7 / Table 9) | 4 of 2,112 GRPO conversations under the held-out judge (K=0 it 1 ×2, it 3; K=5 it 1; one patient utterance each), none under the training oracle; none in the K=5 second draw | `levels_long` n per state |
| Coder check (App D.3) | CT ρ 0.881 / 0.915, ST 0.898 / 0.942 (unchanged); **neutral 0.171 / 0.512** (now stated; PCT does not use it) | `parity_pooled_<judge>` |
| Figure 11 PNG re-drawn | `direction_categories_grpo.png` was rendered before its 0.94\textwidth include existed (1889 px → 1776 px wide); same data, same printed size, fonts now at their intended size | `render_paper_figures.py` |

Audit of this block (2026-10-07, workflow `pct-rebuild-audit`: every changed paper number recomputed
from the raw lake CSVs with independent scripts — no wrong number found). Two caption placements in
Appendix B were tightened (PCT's 94/95-conversation cells named, outside the Base's parenthetical).
⚠ The same EDA re-render also caught up MIPROC (added to `QUESTIONNAIRE_ORDER` on 2026-09-17) in
families the paper does not read (`arms/*`, `lookahead/reward`, `method/contrast`, `compute/cost`,
`measurement/validity`): their non-PCT `p_holm` values moved because MIPROC joined their Holm
families — **reverted the same day (Lior: MIPROC out of every outcome correction, `OUTCOME_ORDER`)**.
The 2026-09 rows above that cite `multijudge_sign_preservation_grpo` (1,640 of 1,848) are superseded
there: now 1,633 of 8 × C(22,2) = 8 × 231 = 1,848 (88.4%) with PCT from the coder; not quoted in the paper.


## 2026-10-07 — review round 4, steps 14–15 (EDA anchors; floats + text batch)

Decision record: [`REVIEW_2026-10-07.md`](REVIEW_2026-10-07.md); plan rows 14–18 of the README.
EDA `9b625c6` (`shared_base.anchor_contrasts` → `lookahead/shared_base/tables/anchor_contrasts.md`)
and `ee33a38` (`levels_long` and `anchor_contrasts` saved at six decimals, so a two-decimal level
such as the Base's Q1+Q2 3.01498 rounds from the table itself). Every cell of the new Tables 1–2 and
Table 8 is printed by `render_process_tables.py` (`--endpoint`, `--anchors`), which asserts the
anchor-'last' d_z against `k_contrast` / `k_process_paired` / `k_persistence` at iteration 10 and the
anchor means against the level sheets.

| Claim | Value | Source |
|---|---|---|
| Table 1 (two anchors) | levels: training oracle, Base / K=0 at 8 / K=0 at 10 / K=5 at 10, two decimals; d_z of K=5 at 10 against K=0 at 8 and at 10, both judges; stars = Holm across the nine rows per (judge, anchor); last column = `significant_iterations` (training oracle; MICI "8–10; 3 (K=0)") | `anchor_contrasts` (family instruments), `levels_long`, `significant_iterations` |
| §4 "+0.43 (dz=0.74) against the best" | Q1+Q2 4.5172 − 4.0823 = 0.4349; dz 0.7432 | `anchor_contrasts`, best_K0, gpt-4o-mini |
| §4 held-out vs K=0 at 8 | significant on 7 of 9 rows; Q2 (dz 0.182, p_holm .172) and WAI-SR (0.097, .578) not; MITI 0.289 at p_holm .048 | `anchor_contrasts`, best_K0, claude-haiku-4-5 |
| §4 "A second draw ... |dz| ≤ 0.17" (moved to App A) | max \|dz\| 0.174 (MICI, training oracle); unchanged | `results/measurement/replicate_draw.md` |
| Table 2 (process, two anchors) | three-decimal levels incl. K=0 at 8 (praise 0.222, CR 0.025, persuasion 0.191, MI-consistent 0.250, MI-inconsistent 0.412, reflects CT 0.009, praises ST 0.130, persuades ST 0.295, reflects ST 0.230, CT→CT 0.880, ST→CT 0.224, CT share 0.518); d_z vs 8 / vs 10 with stars Holm across the twelve rows; last column from the per-iteration tests (Holm across iterations) | `anchor_contrasts` (family process), `process_levels_gpt-4o-mini`, `persist_levels_gpt-4o-mini`, `k_process_paired`, `k_persistence` |
| ⚠ Table 2 iteration-10 stars changed family | MI-inconsistent share −0.36 was * (iterations family), now ** (rows family); every d_z value unchanged | same |
| §6 "reflects sustain talk ... 37%, against 26% under K=0 and 22% at the Base, significantly more only against K=0's best checkpoint" | 0.368 / 0.256 / 0.217; vs 8 dz +0.380 p_holm .009; vs 10 +0.265 p_holm .067 | `anchor_contrasts` |
| §6 "88% at its best checkpoint" (CT → CT) | 0.880 (K=0 at 8) | `persist_levels_gpt-4o-mini` |
| §6 "after sustain talk ... 33% ... 28% ... significant against K=0's best checkpoint (22%) but not against its last" | 0.326 / 0.276 / Base 0.281; K=0 at 8 0.224; vs 8 dz +0.431 p_holm .003; vs 10 +0.176 p_holm .133 | `anchor_contrasts`, `persist_levels` |
| §6 "52% at its best checkpoint" (change-talk share) | 0.518 | `process_levels_gpt-4o-mini` |
| Table 8 (held-out process, two anchors) | as Table 2 under the held-out judge; K=0 at 8 still the training oracle's choice | `anchor_contrasts`, `process_levels_claude-haiku-4-5` |
| §5 turn length | 896 (K=0) / 849 (K=5) characters at iteration 10, Base 273 (was "850–900") | `length_by_state::th_chars_mean` |
| §5 embedding sentence (Figure 4 moved to App C.7) | between-conversation share 0.39 (Base) → 0.19 (K=0) / 0.30 (K=5) at iteration 10 | `text_diversity::persona_var_share` |
| §5 update direction | K=0 points to praise at training iterations 4–8 and 10 in all five encoders; K=5 at no iteration in more than three | App C.6 (existing), `lookahead/mechanism/tables/direction_encoders_categories_*` |
| App B "levels off after iteration 6" | held-out K=5 Q1+Q2 2.903 (6), 2.912, 2.776, 2.858, 2.873 (10); training oracle 4.229 (6) → 4.517 (10) | `levels_long` |
| Discussion / App C.4 gradient ratio | K=0/K=5 gradient norm 0.21/0.17, 0.21/0.19, 0.27/0.22, 0.30/0.22 at iterations 1–4 → "1.1–1.3 times" (unrounded 0.207/0.172 = 1.21, 1.11, 1.26, 1.34; audit fix); median 0.39 / 0.20 | Table 11 (`trainer_diagnostics_by_iter`) |
| App C.4 "K=0 complex reflection ≤ 0.03 at every iteration; K=5 0.10 at 7, 0.23 at 10" | K=0 max 0.027; K=5 0.098 / 0.230 | Table 4 (`process_levels_gpt-4o-mini`) |
| Limitations ICC | ICC(2,1) 0.924–0.994 on Q1, Q2, MICI for GRPO K=0 iterations 8 and 10, 4 scorings each (rep 0 + 3) | `measurement/validity/tables/oracle_repeatability_icc.md` |
| Abstract | 198 words (source count, same counter as the 200 of 2026-10-05) | — |
| Removed from the paper | the four gain ratios (2.04 / 1.41 / 2.50 / 1.33) and "1.3 to 2.5 times" (Lior's pick); the cooperative-persona sentence of §4 (false for MICI; App A keeps the breakdown); the late-session rise/fall with old Figure 3d; Method's 86–89% sentence (App C.1 keeps the values) | — |

## 2026-10-07 — review round 4, step 16 (length pass)

No number changed and none was added. Moved or removed from the body only: the process table
(`tab:process`) is now Appendix Table 3 (same cells, same script); the praise-reward numbers
(0.22–0.33 / 0.16–0.21 within-group SD at iterations 4–7; 0.16 vs 0.04 in the last round) are
stated once, in the Discussion (they had been in §6 and the Discussion); the held-out "reflects
sustain talk 17% vs 5%" left §6 (it stays in Appendix B and Table 8); the Discussion's mechanism
sentence no longer gives the gradient ratio (Appendix C.4 keeps "1.1–1.3 at iterations 1–4, median
0.39 vs 0.20"). Related work and the Introduction lost repeated sentences only; every citation key
of the section survives (checked by script).

## 2026-10-07 — content pass after step 16 (Lior: "errors + clarity", the rest on my judgement)

A content review of the 8-page draft (six lenses; 33 suggestions, kept in the session record) and
the edits applied from it. Numbers that are new to the body, or that moved into a new sentence:

| Claim | Value | Source |
|---|---|---|
| §4 "the K=5 policy scores 4.52, against 3.75 at K=0's last checkpoint (dz=0.91) and 4.08 at its best (dz=0.74)" | Q1+Q2 4.5172 / 3.7548 / 4.0823; dz 0.910 / 0.743 (replaces "+0.76 / +0.43", which came from unrounded means and did not match Table 1's levels) | `levels_long`, `anchor_contrasts` (gpt-4o-mini) |
| §4 "significantly on every row under the training oracle" | max p_holm over the nine rows: 7e-6 (vs 8), 2.5e-5 (vs 10) | `anchor_contrasts` (family instruments, gpt-4o-mini) |
| §4 onset "K=5 leads significantly on Q1+Q2 at iteration 4 and from 6 on ... before that only MICI differs, at iteration 3, in K=0's favor" | Q1+Q2 significant at 4, 6–10 (5 is not; was "leads from iteration 4"); no instrument significant at 1–2; at 3 only MICI (K=0) | `significant_iterations` (Table 1's last column) |
| §4 held-out "on all nine rows against K=0's last checkpoint and on seven against its best" | vs 10: max p_holm 1.3e-4; vs 8: Q2 and WAI-SR not significant (p_holm .172, .578) | `anchor_contrasts` (claude-haiku-4-5) |
| §5 held-out praise "0.76 of the K=0 policy's turns ... against 0.20 of the K=5 policy's, itself above the Base's 0.04" | 0.7617 / 0.2028 at iteration 10; Base 0.04 | `k_process_paired` (th_PRA_rate, claude-haiku-4-5), `process_levels_claude-haiku-4-5` |
| §5 "MICI agrees: 8.3 of the 9.9 MI-inconsistent acts per session ... (84%) fall under its over-praise code" | K=0 at 10: OverPraise 8.250 of BehaviorTotal 9.865 per conversation; share 0.836 | `lookahead/behaviour/tables/k_mici_composition.md` |
| Figure 3 panel (b), the keyword marker | `lex_overpraise_marker_rate` ± SE by iteration, no tests | `marker_and_length` |
| Figure 3 / Figure 8 captions, star direction | persuasion stars favor K=0 under both judges; praise, complex reflection, persistence stars favor K=5 | `k_process_paired`, `k_persistence` |
| §6 held-out "persuades after 58% of the patient's sustain talk (Base 25%) ... change talk is followed by change talk in 94% of cases under K=5, against 73% under K=0 (Base 71%)" | per the held-out process/persistence levels at iteration 10 | `process_levels_claude-haiku-4-5`, `persist_levels_claude-haiku-4-5` |
| App A "the share of sessions that reach any change talk at all does not differ (92% vs 86%)" | moved from §6 unchanged | `process_levels_gpt-4o-mini` (reached_ct) |
| Method "A session ends ... at 50 utterances" | the scripted opener + 49 generated utterances (`_shared/convs.py`, `num_utterances=49`); was "after 49", which conflicted with "at fifty" two paragraphs later | Table 11 (unchanged) |
| Method MICI "counts six MI-inconsistent codes: four of the MISC 2.5, judging or labeling, and an over-praise code of our own" | Confront, AdviseNoPermission, Warn, Direct + Judge + OverPraise | `code/questionnaires.py` (MICI schema) |
| Discussion "under K=5 clearly above zero only at 6–7" | z above 2 only at iterations 6–7 | App C.5 (existing text) |
| Discussion "the two runs' update directions diverge from training iteration 4 on" | noise-corrected cosine 0.92–0.96 at 1–3, 0.80 at 4–5, 0.57–0.74 at 6–8 (replaces "the two rewards prefer different candidates within a group": the rewards never scored the same candidates) | App C.6 (existing text) |
| Abstract | 197 words (same source counter) | — |
| Removed from the body | "+0.76 / +0.43"; "We did not test whether ... MITI ratings" (now one clause of the Limitations' "K in {0,5} only"); the Discussion's third mention of the held-out K=5 praise; "differ only in the horizon" / "differ in nothing else" (now "otherwise matched": the Method discloses the sub-batch field and the one-resume KL reference) | — |
| Method "73% of them at two utterances" (audit fix) | 0.7346 → 73% (was 74%, rounded via 0.735) | `lookahead/mechanism/tables/faithfulness_prefix_alone_grpo.md` (gpt-4o-mini, pooled, n_turns 2) |
| §4 onset, the other instruments (audit fix: "follow" implied a later onset) | first significant K=5 lead: CSQ-8, MI-SAT, Q1 at 4; WAI-SR 5; MITI, PCT 6; MICI 8; Q2 9 | `significant_iterations` (gpt-4o-mini) |

Audit: two independent auditors checked 100 claims (54 in the Results sections, whole files; 46 in
the changed lines elsewhere, cited papers against their abstracts); 3 flagged, each re-checked by a
refuter: the 74% and the "follow" wording confirmed and fixed; the embedding sentence refuted, but
reworded to the appendix's "relative to their variation within a session" anyway.

## 2026-10-08 — review round 5 (Lior: clear fixes + judgement items on my judgement; 'make the paper strong, not weaker')

A 17-lens review (items M#, U#, A#, P#; kept in the session record); M4 declined, so the "ahead on all
eight ... under both judges" wording of the Abstract, contribution 1 and the Discussion opening is
unchanged. `results/…` paths are relative to `Exp3_PTO_GRPO/eda/` as above. One row per number that
is new, changed, re-worded or re-pointed; ⚠ marks a row whose source is not a tracked EDA table.

| Claim | Value | Source | Where |
|---|---|---|---|
| Abstract "In the average conversation under look-ahead, ... change talk is followed by change talk again in 97% of cases (training oracle)" (number unchanged; now worded as a per-conversation mean) | 0.968 (K=5, iteration 10, n 87 conversations) | `results/lookahead/shared_base/tables/persist_levels_gpt-4o-mini.md`, GRPO_LA5 / 10, `ct_persist_mean` | 00_abstract, last sentence |
| Abstract "against 80% without look-ahead" (unchanged, per-conversation mean) | 0.796 (K=0, iteration 10, n 82) | same table, GRPO_LA0 / 10, `ct_persist_mean` | 00_abstract, last sentence |
| Intro contribution 2 "under it, change talk continued more often than without look-ahead" (new comparator, no number printed) | 0.968 vs 0.796, dz −0.624, p_holm < .001 (training oracle); 0.935 vs 0.734, dz −0.650, p_holm < .001 (held-out) | `results/lookahead/shared_base/tables/k_persistence.md`, gpt-4o-mini and claude-haiku-4-5 / GRPO / ct_persist / 10 | 01_intro, contribution 2 |
| Abstract word count | 197 → 200 (same source counter: LaTeX stripped, `\LAGRPO` = 3 words) | — | 00_abstract |
| Method: Q1/Q2 "shown to separate three prompted levels of skill" | three (poor, average, expert therapists, each prompt-engineered) | Yosef et al. 2024, abstract + Sec. 3; corroborated by `results/METRICS_REFERENCE.md` l.49 (a citation fact, not an EDA number) | 03_method, Instruments |
| Method "98% at the session cap" (was "at fifty"; number unchanged) | 0.9760 at n_turns 50 (CI 0.9475–0.9942; 541 pairs; 48 conversations); 73% (n_turns 2, 0.7346) and 85% (12, 0.8456) unchanged | `results/lookahead/mechanism/tables/faithfulness_prefix_alone_grpo.md`, gpt-4o-mini / pooled | 03_method, Minimum conversation length |
| §4 held-out MICI "both policies end above the Base of 0.36 (K=0 1.05, K=5 0.63)" | Base 0.355 (n 192); GRPO_LA0 10 1.050; GRPO_LA5 10 0.628 | `results/lookahead/shared_base/tables/levels_long.md`, claude-haiku-4-5 / MICI / 0, 10 | 05_reward, The final policies |
| §4 "K=0's Q1+Q2 peaks at 8 and then falls" (M7: the peak is a Q1+Q2 fact) | 4.082 at 8 (highest of 1–10; 4.074 at 7), 3.807 at 9, 3.753 at 10 | `levels_long.md`, gpt-4o-mini / GRPO_LA0 / Q1Q2 | 05_reward, Onset |
| Table 1 caption "which favors K=0 on Q1+Q2" (no number printed) | on other rows K=0 is higher at 10 than at 8: WAI-SR 3.44 vs 3.37, CSQ-8 2.77 vs 2.75, MI-SAT 3.48 vs 3.44, PCT 0.583 vs 0.577 | Table 1 levels; PCT from `levels_long.md` gpt-4o-mini / GRPO_LA0 | 05_reward, Table 1 caption |
| Table 1 caption "K=5's best is its last" (A13) | GRPO_LA5 Q1Q2 4.517 at 10 (highest of 1–10), 4.454 at 9; held-out peaks at 7 (2.912 vs 2.873 at 10), so the clause rides on "chosen on these Q1+Q2 scores" | `levels_long.md`, GRPO_LA5 / Q1Q2, both judges | 05_reward, Table 1 caption |
| §6 "PCT is 0.73 under K=5, against 0.58 under K=0 at either checkpoint (Base 0.49; Table 1)" (U5; replaces the `ct_prop` sentence 68% / 53% / 52% / 45%, which counted neutral utterances) | GRPO_LA5 10 0.732; GRPO_LA0 8 0.577, 10 0.583; Base 0.491 (n 192) | `levels_long.md`, gpt-4o-mini / PCT (= Table 1's PCT row) | 07_patient, The session as a whole |
| §5 "the K=0 policy's replies vary less from patient to patient than the K=5 policy's ... from iteration 3 on" (M5) | `persona_var_share` K=0 vs K=5: 0.200/0.272 (3), 0.187/0.280 (4), 0.210/0.301 (5), 0.234/0.280 (6), 0.248/0.282 (7), 0.235/0.310 (8), 0.203/0.306 (9), 0.192/0.302 (10); 0.337/0.378 (1), 0.322/0.329 (2) | `results/lookahead/shared_base/tables/text_diversity.md`, GRPO_LA0 / GRPO_LA5, I1–I10 | 06_therapist, Corroboration |
| §5 "long before its praise peaks" (M5) | K=0 praise share 0.046 at 3, 0.222 at 8, 0.407 at 10 | `results/lookahead/shared_base/tables/process_levels_gpt-4o-mini.md`, GRPO_LA0, `th_PRA_rate` | 06_therapist, Corroboration |
| §5 K=0 direction "toward praise sentences at training iterations 4–8 and 10 in all five encoders" (unchanged) | `n_ci_above_0` = 5 at train_iter 4, 5, 6, 7, 8, 10; 0 at 9 | `results/lookahead/mechanism/tables/direction_encoders_categories_summary_grpo.md`, GRPO_LA0 / praise | 06_therapist, Corroboration |
| §5 + App C.6 "further than the K=5 direction in at least four of the five at 3–6" / "in at most two at 7–10" (M5; replaces "the K=5 direction at no iteration in more than three") | K0−K5 praise `n_ci_above_0` = 4, 4, 4, 5 at train_iter 3–6; 2, 2, 0, 1 at 7–10 (z_min 0.107, 0.321, 0.529, 0.740 at 3–6) | same table, arm K0-K5 / praise | 06_therapist, Corroboration; B_mechanism C.6 |
| Discussion "only the K=0 reward ... clearly favored them again" (M6: "0.16 against 0.04" removed again) | iteration-9 model: K=0 premium 0.165, z 5.69; K=5 0.039, z 1.06; K=0 at the iteration-8 model 0.070, z 1.66 (hence "again") | `results/lookahead/mechanism/tables/praise_premium_grpo.md`, overpraise_marker, iterations 8–9 | 08_discussion, paragraph 2 |
| Conclusion "reflects change talk and praises far less" (M8) | praise share at 10: training oracle K=5 0.053 vs K=0 0.407 (0.222 at 8); held out 0.203 vs 0.762 (0.500 at 8) | `process_levels_gpt-4o-mini.md` and `process_levels_claude-haiku-4-5.md`, `th_PRA_rate` | 08_discussion, Conclusion |
| §5 "the K=5 policy's own share rises above the Base's with its persuasion, though its MICI under the training oracle does not" (U13; "under the training oracle" added in the fix pass, because the held-out judge's MICI for K=5 does rise) | `mi_incons_rate` K=5 0.332 vs Base 0.248; MICI K=5 0.210 vs Base 0.210 (training oracle); held-out MICI K=5 0.628 vs Base 0.355 (significant, App A.2) | `process_levels_gpt-4o-mini.md`; `levels_long.md` gpt-4o-mini / MICI and claude-haiku-4-5 / MICI, iterations 0 and 10 | 06_therapist |
| §5 "The held-out judge agrees on the direction of each coded change at iteration 10" ("coded" added in the fix pass; MICI is not a coded change) | held-out, Base → iteration 10: persuasion 0.166 → K=5 0.268; MI-inconsistent share 0.214 → K=5 0.471, K=0 0.795; complex reflections 0.031 → K=5 0.242; praise 0.038 → K=0 0.762, K=5 0.203; open questions 0.175 → K=0 0.000, K=5 0.032; MI-consistent share K=5 0.360 vs K=0 0.037 (training oracle 0.438 vs 0.290) — every direction as under the training oracle | `results/lookahead/shared_base/tables/process_levels_claude-haiku-4-5.md` and `process_levels_gpt-4o-mini.md`, GRPO_LA0 / GRPO_LA5, iterations 0 and 10 (`th_PERS_rate`, `mi_incons_rate`, `th_CR_rate`, `th_PRA_rate`, `th_OQ_rate`, `mi_adherent_rate`) | 06_therapist |
| §5 "open-question turns" (U3; numbers unchanged) | `th_OQ_rate` Base 0.087, K=0 10 0.000, K=5 10 0.011 | `process_levels_gpt-4o-mini.md` | 06_therapist |
| §6 "answers with a reflection in 26% of cases, against 0.3%" (U16; numbers unchanged) | `refl_after_ct` K=5 0.264, K=0 0.003, Base 0.137 | `process_levels_gpt-4o-mini.md` | 07_patient, How the therapist replies |
| §4 K=0 MICI above the Base "(0.84; Appendix A.2)" (A9 pointer; number unchanged) | GRPO_LA0 MICI 10 0.838 | `levels_long.md`, gpt-4o-mini / MICI / 10 | 05_reward, The final policies |
| Limitations: K=0 "ahead at about 13 GPU-hours under both judges" (verdict widened, no number added) | 13.27 GPU-h, training oracle selects (K=5 2 vs K=0 4): dz −0.742 (training oracle), −0.489 (held-out), both p_holm < .001; the held-out judge selecting its own: −0.780 | `results/compute/cost/tables/budget_sweep_crossjudge.md`, GRPO_K / 13.270 / select gpt-4o-mini | 09_limitations, Matched iterations are not matched cost |
| Limitations: the judges agree "little beyond chance" on individual therapist codes (was "rarely"; kappa range unchanged) | therapist positions, opener excluded, 21 GRPO states: p_observed 0.278–0.539; kappa 0.078–0.261 (printed 0.08–0.26) | `results/lookahead/process/tables/judge_agreement_overall.md`, GRPO / policy turns / therapist | 09_limitations, Instruments and the utterance coder |
| Limitations: "decline on Q1+Q2 over its last two iterations" (M7) | K=0 Q1+Q2 4.082 / 3.807 / 3.753 at 8–10; WAI-SR, CSQ-8, MI-SAT, PCT higher at 10 than at 8 | Table 2 (A_tables, per-iteration), i.e. `levels_long.md` | 09_limitations, One training run per K |
| Limitations ICC removed; verdict "highly self-repeatable on Q1, Q2 and MICI (Appendix D.8)" kept — the 2026-10-07 "Limitations ICC" row now belongs to D.8 | ICC(2,1) Q1 0.992 / 0.994, Q2 0.975 / 0.983, MICI 0.943 / 0.924 (K=0 8 / 10, 4 scorings) | `results/measurement/validity/tables/oracle_repeatability_icc.md`, GRPOExp3_LA0_I8 / _I10, `icc_2_1` | 09_limitations; C_repro D.8 |
| Limitations "the two depths that Baruch et al. (2025) compared for PTO" | look-ahead depths 0 and 5 | PTO paper (`papers/2025_iclr_pto_lookahead/submitted/PTO_paper.pdf`), §4.2 "Experimental variables" (a citation fact) | 09_limitations, K in {0, 5} only |
| Table 3 caption: the Base's therapist-turn rows are over 184 conversations | 184 of 192 | `text_diversity.md`, GRPO_LA0 / 0, `n_convs` | A_tables, tab:process caption |
| Table 2 caption: the Base's unrounded Q1+Q2 | 3.01498 (3.014982; the .md prints 3.015) | `results/lookahead/shared_base/tables/shared_base.xlsx`, sheet levels_long, gpt-4o-mini / GRPO_LA0 / Q1Q2 / 0 | A_tables, tab:scores caption |
| App A.2 (new, Against the Base): every row improves significantly for both runs under both judges except MICI; K=5's MICI unchanged under the training oracle | K=5 MICI last: +0.000 (dz 0.002), 95% interval −0.040 to 0.039, p_holm 0.908; every other "last" row p_holm ≤ .004 | `results/lookahead/shared_base/tables/gains.md`, anchor last (Holm across the nine rubrics within judge × anchor × run) | A_tables, app:vsbase (the 2026-10-05 gains row now cites this subsection) |
| App A.2: MICI rises significantly for K=0 under both judges and for K=5 under the held-out judge | K=0 +0.627 (training oracle), +0.695 (held-out); K=5 held-out +0.273 (dz 0.888); all p_holm < .001 | `gains.md`, MICI / last | A_tables, app:vsbase |
| App A.2: K=0 at 8 improves on every row except MICI, which rises significantly | every best_K0 GRPO_LA0 row p_holm ≤ .001 under both judges; MICI +0.325 / +0.543 | `gains.md`, anchor best_K0 | A_tables, app:vsbase |
| App A.3: the held-out judge finds a significant gap in all three cooperation thirds and scores no conversation 4.5 or above | +0.69 (0.686), +0.81 (0.807), +0.35 (0.355); p_holm < .001 each; share ≥ 4.5 = 0.000 for both runs | `results/lookahead/shared_base/tables/coop_strata.md`, claude-haiku-4-5 / Q1Q2 / last | A_tables, app:cooperation |
| App A.3: Kruskal–Wallis across the thirds' persona differences | p < .001 (het_H 38.811) | `coop_strata.md`, gpt-4o-mini / Q1Q2 / last / All, `het_H`, `het_p` | A_tables, app:cooperation |
| App A.3: K=0 over-praise by third, iteration 10, training oracle | cooperative 0.94 (0.937), warms up 0.67 (0.671), resistant 0.48 (0.484); K=5 at most 0.08 (0.075) | `results/lookahead/shared_base/tables/coop_strata_overpraise.md`, gpt-4o-mini / last | A_tables, app:cooperation |
| App A.4: Q1+Q2 gap averaged over iterations 1–10 | 0.22 (0.217) training oracle; 0.30 (0.301) held out | `results/lookahead/shared_base/tables/k_trajectory.md`, window 1–10 / Q1Q2, `delta_K5_minus_K0` | A_tables, app:wholerun |
| App A.4: the iteration-10 gap it is compared with | 0.76 (0.7646) training oracle; 0.62 (0.6158) held out | `results/lookahead/shared_base/tables/k_contrast.md` (and shared_base.xlsx), GRPO / Q1Q2 / 10 | A_tables, app:wholerun |
| App A.4: iterations 1–7, MICI favors K=0 under the training oracle | 0.24 (0.240) against 0.28 (0.277); dz 0.38 (0.378), p_holm .003 | `k_trajectory.md`, gpt-4o-mini / window 1–7 / MICI | A_tables, app:wholerun |
| App A.7: the K=5 policy's per-conversation rates (next utterance change talk, after change / sustain talk) | 0.97 (0.968), 0.33 (0.326) | `persist_levels_gpt-4o-mini.md`, GRPO_LA5 / 10 (= Table 3) | A_tables, app:reply |
| App A.8: share of sessions reaching any change talk, K=5 vs K=0 at 10, not significant | 92% (0.9167) against 86% (0.8646); dz +0.13 (EDA sign −0.132), p 0.197, p_holm 1.0 | `shared_base.xlsx`, sheet k_process_paired, gpt-4o-mini / reached_ct / 10 | A_tables, app:patientside |
| Table 6 PCT, cooperative, training oracle, vs K=0 at 8 (P1 sign) | +0.00 (delta 0.002) | `coop_strata.md`, gpt-4o-mini / PCT / best_K0 / Cooperative | A_tables, tab:coop |
| Table 4 caption: K=5's lone iteration-1 star (reflecting change talk) is not significant under the held-out judge | training oracle p_holm .0037 (dz −0.484, EDA sign); held out p_holm .768 (dz −0.020) | `shared_base.xlsx`, sheet k_process_paired, refl_after_ct / 1, both judges | A_tables, tab:process-all caption |
| ⚠ App A.5 / A.6: the two Base draws share seeds; the second iteration-10 draw reuses the original's seed (no number printed) | seed 42 in both runs; Base draws seed + 1 = 43; iteration-10 draws seed + 10 + 1 = 53 | not an EDA table: both runs' `run_metadata.json` (`seed`: 42); `code/GRPO_Exp3/grpo_trainer.py` (seed + iteration; seed + num_iterations + 1); `code/tools/generate_eval_convs.py` `seeds_for` | A_tables, app:basedraws, app:seconddraw |
| App B: the held-out judge separates the runs on Q1+Q2 at iterations 4–10, the training oracle at 4 and 6–10 (replaces "7 of the 10 ... where the training oracle does at 6") | held-out 4–10; training oracle 4, 6, 7, 8, 9, 10 | `results/lookahead/shared_base/tables/significant_iterations.md`, Q1Q2, `iters_K5_better` | A2_heldout, Where it agrees and where it differs |
| App B: K=5 under the held-out judge "2.90 at 6, 2.87 at 10, lowest 2.78 at 8" (replaces "between 2.78 and 2.91") | 2.903 (6), 2.912 (7), 2.776 (8), 2.858 (9), 2.873 (10) | `levels_long.md`, claude-haiku-4-5 / GRPO_LA5 / Q1Q2 | A2_heldout |
| App B: training oracle 4.23 → 4.52 (number unchanged; now cites Table 2) | 4.229 (6) → 4.517 (10) | `levels_long.md`, gpt-4o-mini / GRPO_LA5 / Q1Q2 | A2_heldout |
| App B: the held-out judge favors K=0 significantly only on MITI and MICI, both at iteration 3 | `n_sig_K0_better` = 1 (iteration 3) for MITI and MICI; 0 on the other six | `significant_iterations.md`, claude-haiku-4-5 | A2_heldout |
| App B: dz vs K=0's best checkpoint, held out (format only, `\dz{=}x`) | Q1Q2 0.384; Q2 0.182 (p_holm .172); WAI-SR 0.097 (.578) | `results/lookahead/shared_base/tables/anchor_contrasts.md`, claude-haiku-4-5 / best_K0 | A2_heldout |
| App C opener: the generation logs are partial in six training iterations, 50–74% of gradient groups logged | K=0 train_iter 2 0.500, 6 0.741, 8 0.722; K=5 1 0.500, 2 0.712, 7 0.623; every other 1.000 | `results/compute/cost/tables/api_calls.md`, `log_coverage` (K=5 rows = `lookahead/mechanism/tables/tail_audit_by_iter.md`) | B_mechanism, opener |
| App C.2: the K=5 run's log of training iteration 1 "holds only its second epoch, steps 55–108, after the resume" (fix pass: ⚠ dropped, tracked sources found) | 864 of 1,728 groups logged (0.500) = 54 steps × 16, of n_steps 108 | `results/compute/cost/tables/api_calls.md`, GRPO_LA5 / train_iter 1 (`n_steps` 108, `n_groups_logged` 864, `log_coverage` 0.500); `results/lookahead/mechanism/tables/CAPTIONS.md`, `tail_audit_by_iter` ("iterations that crashed and resumed logged only their post-resume steps (GRPO_LA5 iters 1-2 and 7 ~0.50/0.71/0.62)") and `trainer_diagnostics_by_iter` ("GRPO_LA5 iteration 1 resumed at step 54 ... its steps 55-108"); corroborated by the raw `iteration_1/eda/generations.jsonl` (every logged row has `epoch` ≥ 1.0) | B_mechanism C.2 |
| App C.6 "Other encoders": the two directions are furthest apart "at iteration 6 or 7 (0.43–0.77)" | per-encoder minimum of the noise-corrected cosine over train_iter 1–8: minilm 0.567 (6), gte 0.567 (6), mxbai 0.669 (7; 0.675 at 6), qwen3 0.427 (6), llama8 0.766 (6) | `results/lookahead/mechanism/tables/direction_encoders_k_by_iter_grpo.md`, `corrected` | B_mechanism C.6 |
| App C.5: early mixed groups "16–44 per iteration"; z −2.1 and −2.7 (K=0, models 0–1), −3.0 (K=5, model 2) | n_mixed K=0 41, 16, 42; K=5 21, 24, 44 at models 0–2; z −2.091, −2.662 (K=0), −2.970 (K=5); mixed_share 0.021–0.025 | `praise_premium_grpo.md`, `n_mixed`, `z`, `mixed_share` (iteration = train_iter − 1) | B_mechanism C.5 |
| App C.1 + Figure 9: "at 50 utterances, the session cap, the prefix is the whole session" (was "can be") | n_convs 48, agreement 0.9760 at n_turns 50 | `faithfulness_prefix_alone_grpo.md`, gpt-4o-mini / pooled / 50 | B_mechanism C.1, Figure 9 caption |
| App D.9: K=5 not significant under the training oracle at about 30.5 GPU-hours (replaces the "beyond 27.9 GPU-hours the comparison is the best-checkpoint comparison" rows of 2026-09-17 (b) and 2026-10-06) | dz 0.241, p_holm .063 (K=5 6 vs K=0 8) | `budget_sweep_crossjudge.md`, GRPO_K / select = eval = gpt-4o-mini / Q1Q2 / 30.530 | C_repro D.9 |
| App D.9: K=5 leads significantly under the training oracle from about 35 GPU-hours, dz 0.31 to 0.74 | 35.29 and 39.85: dz 0.310, p_holm .020 (K=5 7); 45.43: 0.680 (9); 51.20: 0.743 (10); p_holm < .001 for the last two; K=0 at 8 throughout | same table, budgets 35.290 / 39.850 / 45.430 / 51.200 | C_repro D.9 |
| App D.9: K=5 leads significantly under the held-out judge at every budget beyond 27.9 GPU-hours, dz 0.34 to 0.49 | 30.53: 0.489 (6); 35.29 / 39.85: 0.418, p_holm .001 (7); 45.43: 0.339, .001 (9); 51.20: 0.384 (10) | same table, select gpt-4o-mini / eval claude-haiku-4-5 | C_repro D.9 |
| App D.9: K=0 stays at 8 while K=5's best checkpoint moves from 6 to 10; the best-checkpoint comparison is reached only at 51.2 GPU-hours | `best_iter_b` = 8 at every budget ≥ 22.28; `best_iter_a` = 6, 7, 7, 9, 10 at 30.53 … 51.20 | same table, `best_iter_a` / `best_iter_b` | C_repro D.9 |
| App D.7: malformed markers in 32% of K=0 training candidates in the last training iteration | 0.3160 | `results/lookahead/mechanism/tables/marker_leak_training_grpo.md`, GRPO_LA0 / 10, `share` | C_repro D.7 |
| App D.7: "up to 58% ... (training iteration 7)" (number unchanged; iteration now stated) | 0.5813 | same, GRPO_LA0 / 7 | C_repro D.7 |
| App D.7: within-group correlation with the reward positive in nine of the ten K=0 training iterations, +0.11 in the last, +0.05 pooled (M17: "neither favored nor penalized" → "barely favored") | `r_within` positive at 1, 3–10, −0.0190 at 2; 0.1098 at 10; pooled 0.0521 | same, GRPO_LA0 rows 1–10 and "all" | C_repro D.7 |
| App D.7: under K=5 below 2.5% of candidates (unchanged) | max `share` 0.0238 (train_iter 6) | same, GRPO_LA5 | C_repro D.7 |
| App D.8: training-oracle repeatability ICC(2,1) 0.92–0.99 on Q1, Q2 and MICI (moved here from the Limitations) | Q1 0.992 / 0.994, Q2 0.975 / 0.983, MICI 0.943 / 0.924 (K=0 8 / 10; rep 0 + reps 1–3) | `oracle_repeatability_icc.md`, GRPOExp3_LA0_I8 / _I10, `icc_2_1` (its PTO rows are not quoted) | C_repro D.8 (app:stats) |
| App D.8: held-out judge ICC 0.96–0.98 on Q1, 0.94–0.96 on Q2, 0.75 and 0.52 on MICI (iterations 8 and 10) (fix pass: was 0.53, a double rounding of the stored 0.525) | Q1 0.978 / 0.964; Q2 0.957 / 0.938; MICI 0.749 / 0.525 (unrounded 0.7489 / 0.5249, recomputed with `eda_analysis.scoring.judge.icc_2_1` on the Haiku reps 0–2, MICI_Rate, 96 × 3, for GRPOExp3_LA0_I8 / _I10; Q1 0.9781 / 0.9645, Q2 0.9569 / 0.9382) | `results/measurement/validity/tables/validity.xlsx`, sheet second_judge_agreement, GRPOExp3_LA0_I8 / _I10, `icc_judge` (stored rounded to 3 dp) | C_repro D.8 |
| ⚠ App D.8: replicates 1–3 (training oracle) and 1–2 (held-out judge) re-score Q1, Q2 and MICI only | partitions `judge=*/rep=1..3` (gpt-4o-mini), `rep=1..2` (Haiku), metric Q1 / Q2 / MICI only | the score-lake listing `Exp3_PTO_GRPO/data/eval_scores/_parquet/` (2026-10-08), not an EDA table; consistent with `oracle_repeatability_icc.md` n_reps 4 and `results/LIMITATIONS.md` "two further Haiku reps" | C_repro D.8 |
| App D.6: the keyword marker's ten phrases now quoted in full, ending "you are a light" | `RE_EFFUSIVE`: i'?m so proud / proud of you / inspiration to me / you got this / beautiful / beacon / shining / warrior / hero of your / you are a (light\|beacon) | `Exp3_PTO_GRPO/eda/eda_analysis/constants.py` l.515–517 | C_repro D.6 |
| App D.6: the disengagement cue (new definition, pointed to from A.8) | `RE_DISENGAGE`: i don'?t know / not sure / i guess / whatever / i suppose / don'?t (really )?(want\|care\|see the point) / told to (be\|come) / leave me / waste of time / not interested / doesn'?t matter / i'?m fine | `Exp3_PTO_GRPO/eda/eda_analysis/text.py` l.56–58 | C_repro D.6 |
| App D.5: the scripted opening line, quoted verbatim (source apostrophe is a backtick) | "Hello, welcome to your first motivational session with me. My name is David and I`m a professional motivational counselor. ..." | `Exp3_PTO_GRPO/code/system_prompts_builder.py` l.140–142 (Good level, `_shared/model.py` l.444–450); stored `conversation_0.csv`, row 1 | C_repro D.5 |
| Table 11: therapist context — training prompts drop their oldest turns and keep the system prompt; conversations and rollouts keep the last 2,048 tokens (M24, minimal) | `max_allowed_prompt_length` = `therapist_max_input_tokens` = 2048 (both runs); `truncation_side='left'`; drop_oldest | both runs' `run_metadata.json`; `_shared/model.py` l.64; `_shared/convs.py` l.435–442; `_shared/reward.py` l.167 | C_repro Table 11 + the therapist-prompt sentence |
| Table 11: the validation split is "logged at each epoch end, never used to choose a checkpoint" | `eval_strategy='epoch'`; no `load_best_model_at_end` / `metric_for_best_model`; the saved adapter is the end-of-training policy | `code/GRPO_Exp3/grpo_trainer.py` l.609–647, l.541; `_shared/runtime.py` l.256 | C_repro Table 11 |
| App D.3: the judge sees only the speaker-labeled transcript | one user message = rubric + [THERAPIST]/[PATIENT] transcript | `code/questionnaires.py` l.1111–1197; `_shared/convs.py` `format_conversation_for_oracle`; `eda_analysis/scoring/pipeline.py` l.234 | C_repro D.3 |
| App D.3: the MITI score is the mean of the technical and relational summary globals | MITI_GlobalMean = mean of the 4 globals; Technical = (CCT + SST)/2, Relational = (Partnership + Empathy)/2 | `eda_analysis/constants.py` l.55; `behavior.py` l.88–94 | C_repro D.3 |
| App D.3: Q2's "seventeen relational items, none of them a WAI-SR item" | 0 of 17 shared | `code/questionnaires.py` l.305–390 vs l.406–419 | C_repro D.3 |
| App D.5: the cooperation clauses are rewritten relative to Yosef et al.'s prompts (M28; was Baruch et al.) | Yosef et al. 2024, Figure 1 prints the original low-then-high clause | Yosef et al., CLPsych 2024, p.4; `Exp1_ICLR2025/code/system_prompts_builder.py` l.85–89 | C_repro D.5 |
| App F: on Q1+Q2 the held-out judge ranks K=5 at 10 above K=0's last checkpoint (dz 1.03; training oracle 0.91) and its best by the training oracle, iteration 8 (0.38; training oracle 0.74) (M16; "by the training oracle" added in the fix pass — on its own Q1+Q2 the held-out judge's K=0 peak is iteration 3) | judge_dz 1.030 / primary_dz 0.905 (vs I10); judge_dz 0.384 / primary_dz 0.743 (vs I8); all p_holm < .001 | `results/lookahead/reward/tables/k_endpoints.md`, rows GRPO_LA5_I10 − GRPO_LA0_I10 and − GRPO_LA0_I8, Q1Q2 (= Table 1) | E_saturation, What this does and does not undermine |
| App E.2: the K=5 policy's first reply is coded persuasion by the training oracle's coder and a closed question by the held-out judge's; the second a complex reflection under both (A5; labels verified) | ThCodes gpt-4o-mini OQ\|PERS\|CR\|…; Haiku OQ\|CQ\|CR\|… (index 0 = scripted opener) | the score lake, `rep=0/metric=MIPROC/oracle=Q1Q2/GRPOExp3_LA5_I10/84.csv`, both judges, `MIPROC_ThCodes` | D_example E.2 |
| Table 1 row label "CSQ-8 (1–4)" (A18) | the 1–4 scale | Table 12 (tab:instruments, "8 items, 1–4"); `code/questionnaires.py` CSQ8_ITEMS anchors | 05_reward, Table 1 |
| Tables 1, 3 and 8 re-printed with `\phantom` star padding (P5) | every level, dz, star, bold and iteration cell identical to the previous rows once `\phantom{}` and the CSQ-8 label are stripped (checked by script at integration: 0 differing cells in 9 + 19 + 19 rows) | `render_process_tables.py --endpoint / --anchors` from `shared_base.xlsx` (asserts pass) | 05_reward tab:endpoint; A_tables tab:process; A2_heldout tab:process-heldout |
| CHECKLIST_ARR.md (not paper text): the archived notebook comment gives K=5 a median 1.92× K=0 per step | 1.92 = median of iterations 3–10 = (1.911 + 1.930)/2 | `results/compute/cost/tables/step_multiplier.md`, `GRPO_step_ratio_K5_over_K0` | CHECKLIST_ARR.md, pre-submission item 2 |

Removed from the paper this round: the Discussion's "0.16 against 0.04" (M6; it had returned in the
length pass); §5's "(84%)" after "8.3 of the 9.9" (U16; the two counts stay); §6's 68% / 53% / 52% /
45% change-talk-share sentence (U5; Table 3 keeps the row); the Limitations' inline ICC(2,1) range
(now in D.8); Appendix D.9's "beyond 27.9 GPU-hours the comparison is the best-checkpoint comparison"
(M2; false between 27.9 and 51.2); `\label{sec:mechanism}` (unreferenced; the header note at the top
of this file is stale on that point).

Numberless changes, by group (each in its file, Doron's replaced sentences kept as `%` comments):
- **Front matter.** "the extended conversation" (U1); Q1/Q2 "adapted and validated" (M27, Intro and
  Method); "each look-ahead provides a one-sample estimate" (U15, Doron's sentence); FACA credits "the
  next user turn" (M14); Wei et al. "for search agents" + bib v3 (M12); RLHS sentence (U15); "challenge
  as well as support" (U15); "letting the judge or its criteria evolve" + cite order (M13, P3); bib
  venue/publisher/url/edition/brace fixes (P3); `\hyphenation{CollabLLM AlpacaEval DynamicRubric}` (P4).
- **Method.** "either speaker closes" + pointer to A.1 (M11); MI terms and the PCT clause name change
  talk as MI's in-session target (M15); "Minimum conversation length" (renamed from "context", also in
  Algorithm 1 and Table 11); U4, U7, U14 wording; A14 (why the training oracle is the main judge); MITI
  cite order (P3); persona product in text mode (P4); "(I like my routine; both from our simulated patients)" (A19; the full label, which fits without a new line).
- **Results.** Clearest-case example in two sentences with the K=5 quote that makes it a complex
  reflection (U18); "a pair picked from eight personas drawn at random" + E.2 heading (M3); pointers to
  the new A.1–A.8 subsections (U11); `\boldmath` in run-in headings with math (P4); Figure 3c pointer.
- **Appendices.** Appendix A's seven run-ins became numbered subsections A.1–A.8 with A.2 new (U11, A9);
  the Appendix C opener lists its contents (U8) and its log coverage (A11); C.2 heading "At a matched
  policy, nothing is detectable" (M20); "praise" for "over-praise" where no over-praise code is meant
  (M25); "supplementary archive" for "released with the paper" (A3); the `\clearpage` before Appendices
  D, E and F removed and Table 13 set to `[b]` (P2); Figure 8 caption letters (a, c, d, e) with the
  figure re-rendered, and Figures 2 and 7 panel titles (P6); Figure 8 and Table 13 captions shortened (P7).
- **Integration cuts (to keep the Conclusion on page 8; no number changed).** Table 1 caption "Last
  column: the iterations at which K=5 is significantly better under the training oracle" → "Last
  column: significantly better under the training oracle" (the column header already says "K=5 better
  at iterations"); Introduction "with six further instruments outside the reward and with a held-out
  judge" → "with six instruments outside the reward and a held-out judge"; Method, Judges "since we
  ask" → "as we ask" and "briefly in each results section and in Appendix B" → "in each results section
  and Appendix B"; §6 "answers it with a reflection" → "answers with a reflection", the first "but not
  against its last" → "but not its last", and "PCT is 0.73 under K=5 at iteration 10" → "PCT is 0.73
  under K=5" (the section opener already fixes iteration 10). Built: 36 pages, the Conclusion ends on
  page 8 and the Limitations open page 9; one line of slack on page 8 (tested by padding).
- **Fix pass after the audit (no number changed except D.8's 0.53 → 0.52).** Table 1 caption "Last
  column: $\Kf$ significantly better under the training oracle" (subject restored); Method "a
  judge/label code, and an over-praise code of our own" (comma restored: only over-praise is ours,
  as Appendix D says), "Tested policy contrasts are persona-paired" (was "All tested contrasts": C.1's
  faithfulness difference uses cluster bootstraps and C.5's praise z values are uncorrected), A19's
  full label; §5 "though its MICI under the training oracle does not" and "each coded change" (U13),
  "long before the $\Kz$ praise peaks" (M5 pronoun); Appendix A opener "and then give further checks"
  (A.3 and A.6 are not cited from the body); Appendix B "so on the instruments they are compared only
  with each other" and "the held-out judge favors $\Kz$ significantly only on MITI and MICI" (M19);
  Appendix C opener "iterations 2, 6 and 8 of the $\Kz$ run and 1, 2 and 7 of the $\Kf$ run" and C.1
  "in the first iteration" for "at the Base" (A11); Appendix D Q2 sentence reordered (A7), Artifacts
  "Every reported result ... ; Table 11 gives the sources of the configuration" (A3); Appendix F "its
  best by the training oracle" (M16). U13 adds one line on page 7 (both U13 options were tested and
  each adds one); page 8 absorbs it: the Conclusion still ends on page 8 and the Limitations open
  page 9, and a padding test (two one-line paragraphs appended to §6) still kept the Conclusion on
  page 8, by squeezing the column glue around the §6/§7 headings rather than by empty lines.
