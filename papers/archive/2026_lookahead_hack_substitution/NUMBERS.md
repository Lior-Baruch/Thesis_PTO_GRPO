# NUMBERS.md — the claims ledger

Every quantitative claim in the draft → the exact artifact it came from. If the EDA is re-rendered
and a number moves, this is how you find every sentence that has to change.

**Scored through iteration 10 on both graders (2026-08-16).** PTO LA5 was extended from 8 to 10
training iterations; 1,536 new cells per grader, 0 errors on both. The matched K comparison now
runs over **11 points (iterations 0–10)**.

**View is `L5`, not `L0`.** `L5` is `eda_analysis.RQ_I_VIEW` — the only view whose `7_Stats` §4c/§4d
and `6_Preference` §5d execute, because the K contrast is gated to one owner. Paths below are
relative to `Exp3_PTO_GRPO/eda/results/L5/`.

**Judges.** `gpt-4o-mini` = the primary oracle, which **was the training reward**. `claude-haiku-4-5`
= held out, different family, never played the client. Treat agreement between them as a
train/test generalisation check, never as inter-rater reliability, and **never average their raw
scores** — the level offset is model-dependent.

---

## ⚠ Traps — the ways to get this paper wrong

1. **The AGGREGATE reduction does NOT replicate.** Primary: +1.615 acts/session, dz .45, p .0003.
   Held-out: +0.531, **dz .099, p .167, ns**. The paper claims **substitution, not reduction**.
   Any sentence saying look-ahead reduces MI-inconsistency must carry "under the primary oracle
   only". This is the single easiest way to overclaim.
2. **Counts, not rates, are primary.** The per-turn MICI rate shows a large effect under BOTH
   judges (dz .71 / .66) while the per-session count does not replicate. That is a denominator
   effect: at iteration 10 K=5 takes 14.385 therapist turns to K=0's 10.229. §5 says this
   explicitly; do not quietly promote the rate because it is the cleaner number.
3. **Prevention-vs-delay is RESOLVED (prevention).** The old Limitations paragraph claiming it was
   open has been removed. Do not reinstate it. What remains open is whether K=5 would eventually
   drift past ten iterations — a different, weaker claim, and it is in Limitations.
4. **"K=5 never leads" is false in its literal form.** At iteration 10 under the primary oracle the
   K=5 Q1+Q2 mean is *higher* (4.307 vs 4.260), not significantly (dz −0.096, p_holm .695). Under the
   held-out judge K=0 is nominally ahead (dz .308) but that is **not Holm-significant either**
   (raw p .032, p_holm .130). At iteration 10 NEITHER judge separates the arms on reward.
5. **Iteration 8 numbers are still correct** — they are one point on a longer curve, not superseded.
   §5 uses iteration 8 for the "identical totals" claim and iteration 10 for the endpoint.
6. **Composite numbers show their arithmetic** wherever quoted (`3.042/4.958 = 61.3%`).

---

## §3 Setup

| Claim | Value | Source |
|---|---|---|
| Both PTO arms run to 10 iterations | 11 matched points (0–10) | `tables/7_stats/*/k_means_by_iter.md` |
| Preference groups, K=5, all 10 iters | built **7,548**, trained **6,416** | `tables/6_preference/gpt-4o-mini/training_signal_yield.md` |
| Preference groups, K=0, all 10 iters | built **6,240**, trained **4,935** | same |
| Matched hyperparameters | temp 1.2, M=8, MCL=12, τ, DPO β | both runs' `run_metadata.json` (not an EDA artifact) |

## §4 The channel

All from `tables/7_stats/<judge>/k_paired_channels.md`, family `MI-inconsistent (per session)`.

| iteration | Δ over-praise | dz | p_holm |
|---|---|---|---|
| 5 | −0.02 | −0.02 | 1.00 |
| 6 | +0.33 | +0.25 | .052 |
| 7 | +1.01 | +0.67 | 4.7e-8 |
| 8 | +1.03 | +0.79 | 5.6e-9 |
| 9 | +2.07 | +0.96 | <1e-4 |
| 10 | +2.42 | +0.89 | <1e-4 |

| Claim | Value | Source |
|---|---|---|
| Over-praise K=0, iters 0→10 (per session) | 0.167 → 3.042 | `k_means_channels.md`, `MICI_OverPraise` |
| Over-praise K=5, iters 0→10 | 0.115 → 0.625 | same |
| K=5 creeps 0.47 → 0.63 over iters 8–10 | 0.469, 0.615, 0.625 | same |
| Held-out judge, iter 10 | +3.573, dz 0.999, p<1e-4 | `tables/7_stats/claude-haiku-4-5/k_paired_channels.md` |
| Therapist turns, iter 10 | 10.229 (K=0) vs 14.385 (K=5) | `tables/3_validity/*/session_shape_by_iter.md` |
| Q1+Q2 iter 10, primary | 4.260 vs **4.307**, dz −0.096, raw p .087, **p_holm .695** | `k_means_by_iter.md` + `k_paired_by_method.md` |
| Q1+Q2 iter 10, held-out | 2.866 vs 2.667, dz 0.308, raw p .032, **p_holm .130** | `tables/7_stats/claude-haiku-4-5/` same two |
| **Gain retention, K=5 vs K=0 (§4 "nor more real")** | Q1: 0.720 [0.608, 0.839] (K=5) vs 0.795 [0.677, 0.934] (K=0, overlapping); Q2: 0.562 [0.472, 0.662] vs 0.849 [0.743, 0.979] (**disjoint, K=5 worse**) | K=5 rows: `tables/8_measurement/multijudge_gain_retention.md` (this view; reference `PTOExp3_LA5_Base`; table was EMPTY until the 2026-08-18 fix). K=0 rows: `../L0/tables/8_measurement/multijudge_gain_retention.md` (reference `PTOExp3_LA0_Base`). ⚠ cross-view comparison against different (identical-policy) base draws — the measured base noise floor (max \|dz\| 0.15) bounds the draw effect; the Q2 gap is far beyond it |

## §5 The aggregate — the paper's table

Iteration 10, PTO, persona-paired n=96, per-session counts, from `k_paired_channels.md` under each
judge (family `MI-inconsistent (per session)`).

| channel | Δ primary | dz | Δ held-out | dz |
|---|---|---|---|---|
| Over-praise | +2.42 | +0.89* | +3.57 | +1.00* |
| Advice w/o permission | −0.31 | −0.24 | −1.95 | −0.71* |
| Directing | −0.38 | −0.39* | −0.68 | −0.47* |
| **All acts** | **+1.61** | **+0.45*** | **+0.53** | **+0.10** |

`*` = survives Holm within the seven-channel family.

| Claim | Value | Source |
|---|---|---|
| Iteration-8 totals identical | 3.448 vs 3.458, dz 0.004, p .97 | `k_mici_composition.md` + `k_paired_channels.md` |
| Iteration-10 totals, primary | 4.958 vs 3.344 | `k_mici_composition.md` |
| Iteration-10 totals, held-out | 8.510 vs 7.979 | same, haiku |
| Over-praise runaway, K=0 | 1.500 → 2.688 → 3.042 (it 8→9→10) | `k_means_channels.md` |
| Share, K=0 iter 10 | 3.042/4.958 = **61.3%** over-praise; 1.594/4.958 = **32.1%** advice | `k_mici_composition.md` (`*_share`) |
| Share, K=5 iter 10 | 0.625/3.344 = **18.7%**; 1.906/3.344 = **57.0%** | same |
| Per-turn MICI rate, iter 10 | dz 0.708 (primary), 0.655 (held-out), both p<1e-4 | `k_paired_channels.md`, family `(per turn)` |
| Power argument | components detected at dz 0.46–0.79 in the same comparison | `k_paired_channels.md` |

## §6 Mechanism

| Claim | Value | Source |
|---|---|---|
| K=0 `w_overpraise` peak | +0.06…+0.08 at train_iter 7–9 | `tables/6_preference/gpt-4o-mini/k_mechanism_overpraise_chain.md` |
| K=5 `w_overpraise` max | +0.025 | same |
| Max selection pressure anywhere | 0.086 | `.../update_lexical_push.md` |
| Pool over-praise, K=0, train_iter 1→10 | 0.002 → 0.318 | `.../generation_pool_means.md` |
| Pool over-praise, K=5, train_iter 1→10 | 0.0035 → **0.0649** | same |
| Indexing | train_iter n samples the iter-start policy = eval's `model_iter_{n-1}` | chain table applies the −1 shift |

## §8 Limitations

| Claim | Value | Source |
|---|---|---|
| Severity, iter 8 | primary dz −0.59 (p<1e-3); held-out dz −0.11 (ns) | `k_paired_channels.md`, family `(per turn)` |
| Severity does not persist to endpoint | primary dz −0.156, held-out ns at iter 10 | same |
| GRPO K=5 trained 1-5, scored 0-5 on both graders | 6 model states (1 base + 5 trained) | `STATUS.md` run table; `results/L5/tables/7_stats/*/compute_by_arm.md` (`last_iter` 5, 27.078 GPU-h) |

## §A Appendix (Table 2 — every coded channel at iteration 10)

Every numeric cell of `tables/A` comes from `tables/7_stats/<judge>/7_stats.xlsx`, sheet
`k_paired_channels`, rows `method=PTO, iteration=10` — the md excerpt truncates at 60 rows, so
the xlsx sheet is the citable source. Families as printed there; Δ = K0 − K5.

| Claim | Value | Source |
|---|---|---|
| MICI channels + total, iter 10 | (same as §5 table, plus Confront −0.06/−0.26, Judge −0.04/−0.21, Warn −0.01/−0.10 primary; −0.29/−0.31*, −0.07/−0.24, −0.05/−0.20 held-out) | `k_paired_channels` xlsx, family `MI-inconsistent (per session)` |
| **Persuade (MITI B2) rises under K=5** | Δ −0.67, dz −0.52, p_holm <.001 (primary); Δ −2.63, dz −0.87, p_holm <.001 (held-out) | same, family `MI-consistent (per session)`, metric `B2_Persuade` — quoted in §5 body too |
| Affirmations (B6) lower under K=5, both measures | per session +0.79/dz 0.41* (+1.00/0.48* held-out); per turn +0.099/dz 0.71* (+0.101/0.69* held-out) | same, families `MI-consistent (per session)` / `(per turn)` |
| Questions/reflections per session higher under K=5, per turn null | B3 −0.92/dz −0.43* but B3_per_turn +0.029 ns (primary) | same |
| Ratios, iter 10 | R:Q −0.20/−0.26 (p .05); %CR +0.00/0.01; %MICO +0.03/0.10 primary, +0.20/0.56* held-out | same, family `MI-consistent (per turn)` |
| Session shape deltas, iter 10 | chars/turn −124.7, conv len −8.31, therapist turns −4.16, all dz −0.55* | same, family `session shape` (judge-invariant) |
| Session shape levels, iter 10 | turn len 686.2 vs 810.9; therapist turns 10.2 vs 14.4; conv len 20.4 vs 28.7 | `tables/3_validity/gpt-4o-mini/session_shape_by_iter.md` |

## §B Reproducibility (values verified against `run_metadata.json` on disk, 2026-08-18)

| Claim | Value | Source |
|---|---|---|
| Base model / LoRA / LR / DPO | `meta-llama/Llama-3.2-1B`, r=16 α=16, lr 1e-5, β=0.1 sigmoid, τ=0.1 | `data/pto_Exp3/runs/full/PTO_Iterative_Q1Q2_Llama32-1B_LA{0,5}_MCL12_M8_PTgreedy/run_metadata.json` |
| Temps / caps | therapist 0.9, patient 0.7, branch 1.2, 200-token cap, MCL 12, target 49 utt | same |
| Oracle/patient snapshot | `gpt-4o-mini-2024-07-18` (both roles) | same (`oracle_model_id`, `patient_model_id`) |
| Adapters on HF Hub | `LBK95/PTO_Iterative_Q1Q2_Llama32-1B_LA{0,5}_MCL12_M8_PTgreedy_full` | same (`current_adapter_repo`) |
| Noise floor quoted in §B | 54 base-vs-base contrasts, 0 significant, max \|dz\| 0.147 | `STATUS.md` § Next step (base replicate draws) |

⚠ `run_metadata.json` is overwritten in place on resume — the LA5 file's `started_at`
(2026-08-16) is the extension restart, not the run's origin. Do not quote it as a run date.

## Scoring provenance for this revision

| | |
|---|---|
| New cells | 2 models × 96 convs × 8 metrics = **1,536** per grader |
| Primary oracle | 1,536 completed, 0 errors, ≈ $0.74 |
| Held-out judge | 1 Message Batch, 1,536 succeeded, 0 errored, ≈ $1.45 (receipt basis $42.00 / 22,272 calls) |
| Rubric-parity gate | passed 8/8 before submission |
| Running project total | ≈ $312 |

---

## Open TODOs before submission

- [x] `sections/02_related.tex` — written 2026-08-18; every entry verified against arXiv/ACL
      Anthology/venue pages (Steenstra et al. is CHI **2025**, key `steenstra2025scaffolding`).
- [x] `sections/A_channels.tex` — full channel table pasted from the `k_paired_channels` xlsx
      (see §A ledger above); surfaced the MITI-B2 persuade corroboration now also quoted in §5.
- [x] `sections/B_repro.tex` — filled from both runs' `run_metadata.json` (see §B ledger above).
- [ ] Co-author list/order, venue, submission date.
- [ ] Decide whether the GRPO arms stay in the figures as context (they currently do) or are cut
      to keep the paper strictly within-optimiser.
- [ ] Consider a third grader to break the aggregate tie (§5). This is the one purchase that would
      turn the paper's central hedge into a finding either way.
