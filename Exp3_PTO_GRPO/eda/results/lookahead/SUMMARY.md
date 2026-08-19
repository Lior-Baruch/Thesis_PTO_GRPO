# Exp3 EDA Summary — `lookahead/` (RQ-i: K=0 vs K=5 within each optimizer, both graders)

*Ported from `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` on 2026-08-18 (reorg by research
question); numbers unchanged, paths rewritten.*

> ⚠ **This file is INTERPRETATION, not evidence** — a hand-authored reading of the tables under
> [`*/tables/`](INDEX.md), written in past sessions, largely by Claude. Brainstorm from the tables
> cold, not from here; quote numbers from the tables, not from here. See
> [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs". The `L5` summary this
> was ported from spent ~2 months narrating iteration 5 and an "arms converge" reading while its own
> tables ran to iteration 8 and said otherwise; it then spent weeks titled *"look-ahead never
> significantly leads"*, which was true of PTO and false of GRPO.

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`<sub>/{figures,tables}/…` — no `<judge>/` level: every table here carries BOTH graders).
The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

## What this top covers

`lookahead/` owns the **look-ahead contrast** — K=0 vs K=5 within PTO and within GRPO, persona-paired
on the 96 shared personas, under the primary oracle *and* the held-out judge side by side (columns,
never a mean). Before the reorg this contrast had no natural home: `L0` held the K=0 arms, `L5` the
K=5 arms, and the cross-K tables were gated to one view (`RQ_I_VIEW`); the paper had to build its own
four-arm generators. Those generators are now the modules behind the five families here:
[`reward/`](reward/) (the K contrast on the rubrics, difference-in-differences, method gap by K),
[`transfer/`](transfer/) (does the contrast survive the held-out judge — sign ladder, gain retention
by K), [`behaviour/`](behaviour/) (channels, substitution, session shape, held-out instruments),
[`mechanism/`](mechanism/) (the over-praise chain, signal dispersion, reward faithfulness at a matched
policy, the K-step tail audit) and [`replication/`](replication/) (the ICLR transcripts under the
modern grader; the SD/stability claim). Read every contrast against
[`../compute/SUMMARY.md`](../compute/SUMMARY.md): an iteration is not a fixed unit of spend, and the
K lever's sign is a function of budget. All four arms are trained and fully scored on both graders —
eleven matched points for PTO (0–10), six for GRPO (0–5); RQ-i is a genuine **K×method** comparison.

---

## 3. RQ-i — the answer is METHOD-dependent

Built from a cross-K score frame by [`notebooks/lookahead/reward.ipynb`](../../notebooks/lookahead/reward.ipynb) into
[`reward/tables/k_means_by_iter.md`](reward/tables/k_means_by_iter.md) (levels) and
[`reward/tables/k_paired_by_method.md`](reward/tables/k_paired_by_method.md) (Δ / *dz* / Holm *p*;
per-grader long forms `reward/tables/k_paired_{pto,grpo}_<judge>.md`, the paper's Table 1 shape in
[`reward/tables/k_table1.md`](reward/tables/k_table1.md)).
**Δ = K0 − K5**, so a *positive* Δ means look-ahead **cost** score.

**On PTO, look-ahead never leads on the reward.** Over eleven matched iterations, under either
grader, K=5 never leads Holm-significantly on Q1+Q2. At iteration 10 the primary oracle has it
nominally ahead for the first time (**4.307 vs 4.260**, dz −0.096, p_holm .695) while the held-out
judge has K=0 ahead (**2.866 vs 2.667**, dz 0.308, p_holm .130) — neither separates the arms.
K=0 *leads* significantly at iteration 6 on the primary (+0.257, dz 0.42) and at 5/6/8 under the
held-out judge (dz 0.33–0.51), the edge carried by **Q2** (the ICLR poster's own Q2-only K finding,
reversed) — [`reward/tables/k_table1.md`](reward/tables/k_table1.md).

**On GRPO, look-ahead does lead**, on both graders:

| iteration | metric | primary Δ / dz / p_holm | held-out Δ / dz / p_holm |
|---|---|---|---|
| 4 | Q1+Q2 | **−0.115 / −0.248 / .037** | **−0.233 / −0.374 / .005** |
| 5 | Q1+Q2 | −0.070 / −0.135 / ns | **−0.311 / −0.429 / .006** |
| 5 | MI-SAT | **−0.238 / −0.345 / .022** | **−0.229 / −0.402 / .0031** |
| 5 | PCT | **−0.056 / −0.309 / .022** | **−0.067 / −0.373 / .0039** |

**Nothing happens for the first two updates.** Across iterations 0–2 not one of
3 × 9 × 2 = 54 paired tests clears p_holm < .05 on either grader — and the null survives with *no*
correction at all (0/54 at raw p < .05). Iteration 0 is a clean null control: the same untrained
policy in both arms, two independent 96-conversation draws, largest |dz| anywhere 0.147.

**The first thing look-ahead buys GRPO is negative.** At iteration 3 the only Holm-significant
cells run *against* K=5 — MICI dz −0.319 (primary, p_holm .013) and −0.354 (held-out, .0034), and
that one survives the turn-count control (turns matched, 13.81 vs 14.56) on the per-session count
too (2.95 vs 4.04 acts, dz −0.419). It does **not** survive normalising on therapist *language*
(per 1k chars: 0.765 vs 0.810 ns primary, 1.348 vs 1.385 ns held-out) — see §5.

**At matched budget the GRPO gain is larger and MICI reverses sign** — that row (Q1+Q2 +0.289 dz
0.359 p_holm .018 primary / +0.540 dz 0.838 p_holm <.001 held-out; MICI −0.497 dz −1.339 /
−0.403 dz −1.228) is the compute axis's and is quoted with its denominators and both framings in
[`../compute/SUMMARY.md`](../compute/SUMMARY.md) §2.

## 4. Look-ahead REVERSES which method wins — and at iteration 5 only the held-out grader sees it

*(Qualifier added 2026-08-18: the primary oracle does see GRPO > PTO under K=5 one iteration earlier —
iter 4 Q1Q2 −0.232, dz −0.351, p_holm .024; MITI dz −0.411 — in
[`../method/contrast/tables/method_paired_by_K.md`](../method/contrast/tables/method_paired_by_K.md).
The difference-in-differences null below is an iteration-5 statement.)*

At iteration 5, with both K arms of both methods scored on the same 96 personas:

| at iteration 5 | PTO − GRPO, held-out judge |
|---|---|
| K = 0 | **+0.265**, dz 0.355, p_holm **.014** — PTO wins |
| K = 5 | **−0.219**, dz 0.377, p_holm **.005** — GRPO wins |

Difference-in-differences on the same personas — Q1+Q2 dz **0.525** (p_holm **.0001**), Q1 0.473,
Q2 0.474, MITI 0.441, all p_holm ≤ .0005. Tables: [`reward/tables/k_did.md`](reward/tables/k_did.md)
(the DiD by iteration) and [`reward/tables/k_method_gap.md`](reward/tables/k_method_gap.md) (the
method gap at each K); figure [`reward/figures/k_did.png`](reward/figures/k_did.png).

⚠ **On the primary oracle the same interaction is null** — largest dz 0.211 (WAI-SR), nothing
survives Holm. The grader that *was* the training reward cannot see an effect the held-out grader
measures at dz 0.5. That is the sharpest single argument in the thesis for why the second judge
exists, and it is consistent with the circularity limitation
([`LIMITATIONS.md`](../LIMITATIONS.md) §3).

⚠ **Scope.** GRPO_LA5 stops at iteration 5, and at K=0 PTO's lead keeps *growing* after that
(+0.609, dz 1.27 by iteration 10 on the held-out judge). The reversal at 5 is measured; whether it
survives to 10 is not known. On the compute axis, though, iteration 5 is not an early stopping
point for GRPO_LA5 — it is the arm's full budget ([`../compute/SUMMARY.md`](../compute/SUMMARY.md)).

## 5. The behaviour channels — substitution, and a verbosity confound

Sources: [`behaviour/tables/k_paired_channels.md`](behaviour/tables/k_paired_channels.md),
[`behaviour/tables/k_mici_composition.md`](behaviour/tables/k_mici_composition.md),
[`behaviour/figures/k_mici_composition_grid_gpt-4o-mini.png`](behaviour/figures/k_mici_composition_grid_gpt-4o-mini.png)
(+ `_claude-haiku-4-5.png`); the per-grader long forms `behaviour/tables/k_channels_{pto,grpo}_<judge>.md`
+ `k_channels_text_*.md` + `k_channels_summary.md`.

**Substitution replicates on a second method and on a denominator-free measure.** Over-praise as a
*share of MI-inconsistent acts*, GRPO at iteration 5: **0.178 → 0.086** (primary, dz 0.344,
p_holm .0045) and **0.182 → 0.063** (held-out, dz 0.722, p_holm <.0001) — while the overall
MI-inconsistent *share* is flat or slightly worse (primary 0.195 → 0.230, dz −0.244, p_holm .037;
held-out ns). One channel is suppressed; the total is not.

The cleanest one-for-one swap is at **iteration 4** on the held-out judge, where the channel deltas
sum exactly to an unmoved aggregate: +0.688 (over-praise) + 0.448 (advise) − 0.104 (confront)
− 0.646 (direct) − 0.021 (judge) − 0.021 (warn) = **+0.344** = the measured `MICI_BehaviorTotal`
delta, dz 0.085, ns.

⚠ **Three denominators disagree in direction here, and two of them are moving.** GRPO's K=5 arm
takes **26% fewer** therapist turns at iteration 5 (11.31 vs 15.34) while writing **1.7× longer**
turns (678 vs 394 chars). So:

| normalisation | GRPO K=5 vs K=0 at iteration 5 |
|---|---|
| per therapist TURN (`MICI_Rate`) | K=5 **worse** (dz −0.243, ns) |
| per SESSION (count) | K=5 better on held-out (dz 0.342, p_holm .005), ns primary |
| per 1,000 therapist CHARS | K=5 better on both (dz 0.339 / 0.467) |
| **share of all coded acts** | **total unchanged, over-praise specifically down** |

Only the share has no moving denominator. **Prefer it.**

⚠ **The per-1k-character improvement is dilution, not skill.** Coded MI acts per 1,000 therapist
characters roughly **halve** under K=5 in *both* valences: MI-consistent 3.32 → 1.68 (primary,
dz 0.764) / 3.12 → 1.58 (held-out, dz 0.720); all coded acts 4.08 → 2.22 (dz 0.717) / 4.97 → 2.59
(dz 0.650). K=5's text is about half as behaviourally dense per word. Over-praise still falls
*faster* than the general dilution (82% vs 48% on the held-out judge), which is why the share-based
substitution claim survives — but the reward gain in §3 is partly a verbosity effect.

⚠ **Verbosity is a training-DEPTH channel, not a look-ahead one.** Chars per therapist turn:
`GRPO_LA0` **394 @5 → 905 @10**, versus `GRPO_LA5` **678 @5**. At matched iteration K=5 is the
verbose arm; at matched compute K=0 is, by a wider margin (11,542 vs 7,671 chars/session). Both
arms inflate; look-ahead reaches a given inflation sooner per iteration and later per GPU-hour.

## 6. Second-judge check

Every number above is reported on both graders because the primary oracle **was** the training
reward. The held-out judge (Claude Haiku 4.5, different family, never played the patient) has the
full grid for every arm here. Where the two disagree it is stated inline, and the disagreements are
not decorative: the K×method interaction (§4) is significant on one grader and null on the other,
and the aggregate MI-inconsistency claim flips with it. Whether the primary-oracle K contrasts keep
their sign under the held-out judge, contrast by contrast, is
[`transfer/tables/k_pairs.md`](transfer/tables/k_pairs.md) laddered by effect size in
[`transfer/tables/k_sign_ladder.md`](transfer/tables/k_sign_ladder.md).

Never average the two — that is train-vs-test, not two raters. Combine contrasts only. The judge's
own validity (ICC, variance decomposition, all-pairs sign preservation across every arm pair) is
[`../measurement/SUMMARY.md`](../measurement/SUMMARY.md).

### 6b. Gain retention for the K=5 arms — method-dependent again

Source: [`transfer/tables/k_retention.md`](transfer/tables/k_retention.md) +
[`transfer/tables/k_retention_summary.md`](transfer/tables/k_retention_summary.md), figure
[`transfer/figures/k_retention.png`](transfer/figures/k_retention.png) (reference kind `own_base`
= each arm's own base draw; the table also carries the `method_LA0_base` / `method_LA5_base` /
`eda_view_PTO_LA{K}_base` reference kinds so the two K arms are compared against a named base).
*(Porting note: in the retired tree the K=5 rows came from `L5/tables/8_measurement/multijudge_gain_retention.md`
(reference `PTOExp3_LA5_Base`; ⚠ an **empty 0-byte file until 2026-08-18** — the notebook hardcoded
the L0 base as reference, which the L5 K-filter excluded; `save_table` now writes an explicit
empty-table marker instead of a silent 0-byte artifact) and the K=0 comparators from the `L0` view's
retention table (reference `PTOExp3_LA0_Base`) — a cross-view comparison against different draws of
the *identical* base policy; the measured base noise floor (max |dz| 0.15) bounds the draw effect.
The all-arm multijudge table is now
[`../measurement/validity/tables/multijudge_gain_retention.md`](../measurement/validity/tables/multijudge_gain_retention.md).)*

- **GRPO: look-ahead makes the gains REAL to the held-out judge.** Q1 retention at the matched
  iteration 5: **1.08 [0.94, 1.27] (K=5) vs 0.73 [0.57, 0.92] (K=0) — disjoint intervals**; at
  iteration 4, 0.98 [0.86, 1.13] vs 0.79. Under K=5 the held-out judge credits GRPO's full Q1
  gain; under K=0 it was already withdrawing a quarter of it by iteration 5 (and 72% by 10).
  This is the retention-space counterpart of §4's interaction and coheres with it.
- **PTO: look-ahead retains the same or LESS.** Q1 at the endpoint: 0.72 [0.61, 0.84] (K=5) vs
  0.80 [0.68, 0.93] (K=0), overlapping; **Q2: 0.56 [0.47, 0.66] vs 0.85 [0.74, 0.98] — disjoint,
  K=5 worse.** Closing the flattery channel did not make PTO's claimed gains more transferable —
  quoted in the substitution draft's §4 ("nor does it make the reward gains more real").
- MITI retention stays low everywhere (0.27–0.59), consistent with its judge-dependence; MICI
  "retention" for GRPO_LA5@5 is 1.70 (the held-out judge sees a *larger* inconsistency rise).

## 6c. Cross-K findings from the paper generators (2026-08-18)

Carried from `STATUS.md` § "Cross-K findings from the paper generators"; the generators
(`papers/2026_lookahead_pto_grpo/analysis/`) were promoted into `eda_analysis` and now render here.
Each bullet names its family; the paper's frozen `analysis/out/*.json` is the fixture the
self-check compares those modules against.

- **The ICLR ordering reproduces on its own transcripts under the modern grader** →
  [`replication/`](replication/) (`crossgen_*`). Re-scoring the poster's 1,440 Exp1 conversations
  (`eval_scores/_crossgen/`) with gpt-4o-mini keeps K=5 above K=0 at 7/7 iterations (arm-level dz
  −0.54 vs −0.61 under GPT-3.5; Spearman 0.84 between the graders' 15 model means) — the Exp3 null
  is a property of the **regime** (1B therapist, V3 patients, MCL=12, iterative regeneration,
  bf16), not of the judge.
- **Look-ahead rescales the training signal, it does not sharpen it** →
  [`mechanism/`](mechanism/) (`dispersion_{by_iter,ratios,tau,expectation}.md`, `faithfulness_{curve,k_by_iter,matched_policy,k_summary,by_coop,levels}.md`;
  figures `dispersion.png`, `dispersion_tau.png`, `faithfulness.png`, `faithfulness_heldout.png`). Best–worst margin and
  within-group SD rise by the same ~1.4–1.8× (ratio-of-ratios 1.01–1.03); margin/SD sits at the
  8-draw expectation in every arm; ~half of PTO K=5's higher τ-yield at the base policy is that
  rescaling. At a **matched policy** (train_iter 1) look-ahead adds **no** reward faithfulness
  (K0−K5 +0.004 [−0.067, 0.074] PTO; +0.015 [−0.023, 0.057] GRPO).
- **What the K-step reward sees** → [`mechanism/`](mechanism/) (the tail-audit tables + figure
  `tail_audit.png`; the API-call side is `api_calls.md` / `api_ratio.md` in [`../compute/cost/tables/`](../compute/cost/tables/)). 19–23% of K=5 tails end early, almost always
  because the simulated patient closes; ended-early siblings score lower within group (dz
  −0.24/−0.26) and are ~23% less likely to be the argmax (RR 0.77/0.79), a pressure that grows over
  PTO training.
- **Session shape reverses by optimizer** → [`behaviour/`](behaviour/) ([`session_shape.md`](behaviour/tables/session_shape.md),
  `length_endpoints.md`, `length_kcontrast.md`, `selection.md`; figure `session_shape.png`). PTO K=5 sessions +8.3 utterances *longer* at iter 10 (dz 0.55), GRPO K=5 −8.1 *shorter*
  at iter 5 (dz 0.53); both K=5 arms write longer turns; PTO_LA5 is the only arm whose update pushes
  for length at every iteration (`w_len` +49.5 … +7.0).
- **The ICLR 'lowest SD = more stable' claim fails** → [`replication/`](replication/)
  (`sd_by_iter`, `sd_tests`, `sd_tally`, `sd_summary`, `ceiling`). PTO K=5 is *more* dispersed than
  K=0 at 10/10 iterations on the primary (Pitman–Morgan sig at 4); SD is a ceiling artefact
  (Spearman(mean, SD) −0.87 over 35 states, the cooperative third saturating ≥ 4.5) and absent
  under the held-out judge.
- **WAI-SR composition** shifts from Bond (K=0) to Goal/Task (K=5) on both graders (bond-excess
  K0−K5 +0.22/+0.27, dz ≈ 0.44); the held-out judge puts K=0's late Q2 gain on the two emotional
  self-disclosure items (+1.1 over K=5, dz > 1); PCT change-talk rises under K=5 in Warms-up
  personas → [`behaviour/`](behaviour/) (held-out instruments: `wai_subscales.md`, `wai_kcontrast.md`,
  `q2_items{,_long,_kcontrast}.md`, `pct_kcontrast.md`, `hetero_kcontrast.md`; figures `wai.png`, `hetero.png`).
- The reward-side headline (K=0 leads PTO at iteration 6 on the primary and at 5/6/8 held-out,
  edge carried by Q2; GRPO K=5 leads at 4–5) → [`reward/`](reward/) (§3 above); retention by K →
  [`transfer/`](transfer/) (§6b above); the verbosity/substitution reading → [`behaviour/`](behaviour/)
  (§5 above).

## 7. Caveats

- **Every claim needs its axis named.** Iteration and compute disagree on `MICI`'s sign for the
  GRPO K contrast, and both are correct answers to different questions
  ([`../compute/SUMMARY.md`](../compute/SUMMARY.md)).
- **`MITI` dependability is 0.553 and `MICI` 0.628**, and those two instruments carry §5. There is
  **no channel-level ICC at all**, and **no repeatability rep for any K=5 state**.
- **Every endpoint is a single 96-conversation draw.** The only measured noise floor is at the base
  (54 same-policy contrasts, 0 significant, max |dz| 0.128 / 0.147). Therapist decoding is
  unseeded, so no conversation set is reproducible.
- **All 96 personas are used for both training and eval**, so everything is in-sample with respect
  to the patient distribution.
- ⚠ `GRPO_LA5` runs to iteration **5**, not 10, so every matched-*iteration* row here hands the K=5
  arm ~2× the compute per cell; on the compute axis it is budget-matched to `GRPO_LA0` within 3%.
- The promoted modules seed their bootstraps with `constants.BOOT_SEED`; the paper generators used
  other seeds, so CI *bounds* in the rendered tables differ from the paper's frozen tables at
  Monte-Carlo scale (≤ ~0.02 on the rubric scale; one or two `judge_ci_excl0` flags flip in
  `transfer/`). Point estimates, dz, p, n reproduce exactly.
