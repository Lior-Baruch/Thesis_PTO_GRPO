# NUMBERS.md — the claims ledger

Every quantitative claim in the draft → the exact tracked artifact it came from. `results/…`
paths are relative to `Exp3_PTO_GRPO/eda/`. **Nothing enters the `.tex` that is not a row here**,
and no row is written from prose — each was read off the named table on 2026-08-26.

**Graders.** `primary` = `gpt-4o-mini` (this WAS the training reward). `held-out` =
`claude-haiku-4-5` (never touched training). ⚠ Levels are not comparable across graders (offset
1.2–1.7 points, model-dependent) and the two are **never averaged** — only contrasts and
standardized quantities combine.

**Sign conventions.** `method_paired_by_K` reports **PTO − GRPO** (+ = PTO higher). The EDA's K
tables (`k_paired_*`, `k_table1`) report **K=0 − K=5**; this paper's prose flips them to K5−K0
where it argues about look-ahead — every row below states its direction. `MICI` is
lower-is-better.

**Axis decision (2026-08-27, Lior + supervisors): ITERATIONS ONLY.** The GPU-hour/budget
analyses are out of the paper entirely (they remain EDA artifacts under `results/compute/cost/`);
the only cost-flavoured content left is the matched-iterations≠matched-data disclosure in
Limitations and the one-line compute total in the Ethics statement.

---

## §3 Setup (config facts)

| claim | value | source |
|---|---|---|
| Grid | 4 arms × 11 states (base + 10 iters) = 44 model states, 8 instruments, 96 personas, 2 graders | results/measurement/validity/tables/multijudge_coverage.md |
| Matched knobs | M = G = 8, MCL=12, same oracle (Q1+Q2), same temps, 10 iterations, personas reshuffled seed+k+1 | both arms' `run_metadata.json` (CONFIG FACT) |
| Within-optimizer arm diff | exactly 2 substantive fields (lookahead_k + sub-batch mirror) | diff of `run_metadata.json` pairs, verified 2026-08-25 |
| Optimizer-specific knobs | GRPO: KL β=0.01, temp 1.2, batch 64×2; PTO: DPO β=0.1, τ filter, greedy trunk | `run_metadata.json` (CONFIG FACT) |
| Bootstrap | 2,000 resamples, percentile, BOOT_SEED | eda_analysis/constants.py |

## §4 The interaction

| claim | direction | value | source |
|---|---|---|---|
| Own-base gains @10, Q1Q2, primary | gain | PTO_LA0 +1.259 (dz 1.429), PTO_LA5 +1.304 (1.353), GRPO_LA0 +0.686 (0.721), GRPO_LA5 +1.554 (1.518) | results/method/contrast/tables/headline_grid.md |
| Own-base gains @10, Q1Q2, held-out | gain | +1.036 (1.653), +0.833 (1.124), +0.396 (0.658), +1.038 (1.539) | same |
| K lever @10, GRPO | K5 higher | +0.765 (dz 0.905) primary; +0.616 (1.030) held-out; sig on all 8 instruments both graders | results/lookahead/reward/tables/k_endpoints.md (signs flipped from K0−K5) |
| K lever @10, PTO | null / K0 higher | Q1Q2 +0.047 (p_holm .695) primary; −0.199 (dz −0.308, p .032, **p_holm .227**) held-out. Held-out K0-higher sig: Q2 −0.363 (dz −0.653, ***), MITI −0.203 (−0.487, ***). K5-better sig: MICI −0.228/−0.245 (dz −0.708/−0.655, *** both graders), PCT held-out +0.051 (dz 0.253, *) | results/lookahead/reward/tables/k_paired_pto_{gpt-4o-mini,claude-haiku-4-5}.md iter-10 rows (signs flipped) |
| Method @K0 @10 (PTO−GRPO) | PTO higher | +0.507 (dz 0.729, <.001) primary; +0.609 (1.265, <.001) held-out | results/method/contrast/tables/method_paired_by_K.md |
| Method @K5 @10 (PTO−GRPO) | GRPO higher | −0.210 (−0.356, .001) primary; −0.206 (−0.313, .034) held-out | same |
| DiD @10 (gap_K0 − gap_K5) | interaction | 0.718 (dz 0.793, <.001) primary; 0.815 (0.972, <.001) held-out; sig from iteration 4 on | results/lookahead/reward/tables/k_did.md |
| Replicate: noise floor | neither | 36 same-policy contrasts (9 × 2 graders × 2 states), 0 sig, max \|dz\| **0.216** (PTO_LA0 held-out MICI, p_holm .212) | results/measurement/replicate_draw.md |
| Replicate: method @K0 | PTO higher | +0.516 (0.736) / +0.659 (1.389), both <.001 | same |
| Replicate: method @K5 | GRPO higher | −0.155 (−0.293, .007) / −0.227 (−0.342, .013) | same |
| Replicate: GRPO K lever | K5 higher | +0.709 (0.919) / +0.637 (0.949), both <.001 | same |
| Two winners @10 (GRPO_LA5 − PTO_LA0) | GRPO higher on primary only | primary +0.257 (dz 0.492, <.001); held-out **+0.007 (dz 0.012, p_holm 1.000)** — replicate +0.193 (0.436) / **−0.022 (p_holm 1.000)** | results/measurement/replicate_draw.md § top pair |
| Two winners: held-out gain tie | tie | PTO_LA0 +1.036 vs GRPO_LA5 +1.038 | headline_grid.md held-out Q1Q2 rows |

⚠ **The re-draw covers the two winning endpoints only** (GRPO_LA5@10, PTO_LA0@10); the other 40
states are single draws. A replicate bounds EVALUATION noise, never TRAINING variance.
⚠ **PTO's held-out K contrast is NOT Holm-significant on Q1Q2** (p_holm .227) — write
"null-to-negative", never "significantly worse" for the composite; the significantly-worse
claims are Q2 and MITI held-out only.

## Limitations — matched iterations ≠ matched data (the disclosure that replaced §5 Cost)

| claim | value | source |
|---|---|---|
| Oracle scoring calls, sum over train iters 1–10, K=0 | GRPO_LA0 **302,541** vs PTO_LA0 **99,622** → 302,541 / 99,622 = **3.04×** (paper says "roughly 302k vs 100k, 3.0×") | results/compute/cost/tables/api_calls.md, `oracle_calls_train` column summed over the 10 `iteration` rows per arm |
| Oracle scoring calls, sum over train iters 1–10, K=5 | GRPO_LA5 **289,983** vs PTO_LA5 **121,806** → 289,983 / 121,806 = **2.38×** (paper says "290k vs 122k, 2.4×") | same |
| Why the asymmetry | GRPO slices every eligible prompt (1,120–2,528 groups/iter across both arms) vs PTO's per-trunk branch points (410–949/iter), both × 8 candidates | same table, `n_groups` column |

⚠ These sums are **derived** (10-row additions over `api_calls.md`) — if that table re-renders,
re-sum; do not trust these totals as atomic numbers.
⚠ The retired budget/GPU-hour rows (totals 8.119/19.681/27.906/51.205 GPU-h, the 6.31× spread,
the 16 budget-sweep verdict cells) are in git history and in the EDA
(`results/compute/cost/`), deliberately NOT in this paper.

## §6 Behaviour

| claim | axis | value | source |
|---|---|---|---|
| Judge-free lexical marker, base → @10 | share of therapist turns, no grader | GRPO_LA0 0.003→**0.671**; PTO_LA0 0.003→**0.210**; GRPO_LA5 0.000→**0.064**; PTO_LA5 0.000→**0.045** | results/arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md `lex_overpraise_marker_rate` (judge-invariant; lives under a judge path only because arms/* is per-judge) |
| Oracle-coded over-praise rate @10 | per-turn rate, primary | 0.698 / 0.299 / 0.051 / 0.043 (same arm order) | same table, `MICI_OverPraiseRate` |
| Over-praise K contrast sig | rate | K0 worse: GRPO 6 iters primary (5–10), 7 held-out (4–10); PTO 5 both (6–10) | results/lookahead/behaviour/tables/k_channels_summary.md `MICI_OverPraise_rate` |
| MICI totals vs own base @10 | count/session | primary: GRPO_LA5 0.209→0.210 (+0.002, n.s. p .711), PTO_LA5 0.177→0.264 (+0.086, p .002); held-out: GRPO_LA5 0.326→0.628, PTO_LA5 0.370→0.581 vs K0 siblings 1.050 / 0.825 | headline_grid.md MICI rows |
| PTO substitution | count/session, held-out | advise-without-permission sig HIGHER under K5 at 6 of last 7 iters (4,6,7,8,9,10); GRPO only 3 (3,8,10) | k_channels_summary.md `MICI_AdviseNoPermission` |
| Questions per turn | judge-invariant text | GRPO K5 higher at 7 iters (4–10), mean dz 0.643 (sign-flipped from K0−K5 −0.643); PTO 0 iters | k_channels_summary.md `q_per_turn` |

⚠ "Flat MICI" is a **training-oracle** statement for GRPO_LA5 (and near-flat for PTO_LA5); the
held-out judge reads both K5 arms as rising. Write "slows the drift to a fraction / removes the
dominant channel", never "stops"/"eliminates".
⚠ The lexical marker is the *share of therapist turns containing ≥1 marker* (0.671 = 67% of
turns), a brittle keyword regex kept as a direction check — cite its agreement with the coded
rate, never its absolute value as a measurement.

## §7 Regime (crossgen re-scoring of the ICLR-era conversations)

| claim | value | source |
|---|---|---|
| Mean over iters 1–7, modern grader | K5 higher by 0.132 (dz 0.543, p<.001; sign-flipped from K0−K5 −0.132) | results/lookahead/replication/tables/crossgen_kcontrast_summary.md |
| Mean over iters 1–7, gpt-3.5-matched grader | K5 higher by 0.206 (dz 0.612, p<.001) | same |
| Best-vs-best (ICLR pick), modern grader | K5 higher by 0.129 (dz 0.250, p .006, p_t .016) | same |

⚠ The earlier regime differs in several ingredients at once (7B policy, cooperative personas,
non-iterative loop) — the artifact shows the sign varies with regime, not which ingredient did it.
⚠ Exp1 and Exp3 score axes are not comparable; the crossgen re-scoring exists precisely so no
cross-experiment level is ever quoted.

## §8 Measurement (full-grid — this paper's scope IS the four arms)

| claim | value | source |
|---|---|---|
| Sign preservation | 6,693 of 8 × C(44,2) = 7,568 (88.4%); 97.2% at \|Δ\|≥0.25; 99.3% at \|Δ\|≥0.50 | results/measurement/validity/tables/multijudge_sign_preservation.md |
| Q1 agreement, 44-state median | 0.855 | results/measurement/validity/tables/validity.xlsx `second_judge_agreement` |
| The two lowest states | GRPO_LA5_I9 .487, GRPO_LA5_I10 .544; next cluster PTO_LA5_I10 .667 | same |
| Both K5 arms fall late, neither K0 does | Q1 agreement, last two iterations | same sheet + judge_saturation figure |
| One-sided saturation (GRPO_LA5, Q1) | primary SD 1.336→0.701 (ρ −0.86, p .001; variance ratio 0.275, anchor-robust); held-out SD flat (ρ +0.44, p .18) | results/lookahead/replication/tables/sd_by_iter.md |
| Oracle self-repeatability | ICC(2,1) 0.86–0.99, K=0 anchor states only | results/measurement/validity/tables/oracle_repeatability_icc.md |

⚠ Sign preservation is ARM-level; it licenses no per-conversation claim — the winning checkpoint
is the counter-example. ⚠ Never write that the held-out grader's variance GREW (two-point ratio
anchored on a series minimum; the trend is null — "flat").

## Ethics / totals

| claim | value | source |
|---|---|---|
| Total compute (Ethics statement only — no per-arm breakdown in the paper) | 8.119 + 19.681 + 27.906 + 51.205 ≈ 107 GPU-h | compute_by_arm.md — the paper quotes only the ≈107 total |
