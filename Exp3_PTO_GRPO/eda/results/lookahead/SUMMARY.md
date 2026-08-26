# Exp3 EDA Summary — `lookahead/` (RQ-i: K=0 vs K=5 within each optimizer, both graders)

*Ported from the retired `results/L0/SUMMARY.md` + `results/L5/SUMMARY.md` on 2026-08-18 (reorg by
research question); rewritten in full on 2026-08-25 against the completed four-arm grid.*

> ⚠ **This file is INTERPRETATION, not evidence** — a hand-authored reading of the tables under
> [`*/tables/`](INDEX.md), written in past sessions, largely by Claude. Brainstorm from the tables
> cold, not from here; quote numbers from the tables, not from here. See
> [`CLAUDE.md`](../../../../CLAUDE.md) § "Epistemic status of these docs". This file has been wrong
> before, in exactly the way that section predicts. The `L5` summary it descends from spent ~2 months
> narrating iteration 5 and an "arms converge" reading while its own tables ran to iteration 8 and
> said otherwise; it then spent weeks titled *"look-ahead never significantly leads"*, which was true
> of PTO and false of GRPO; and the version this replaces declared the K×method interaction
> **invisible to the training grader**, which was an iteration-5 observation stated as a general one —
> the primary oracle sees it plainly from iteration 6 on (§3). Every one of those errors was prose
> about tables. The tables were never wrong.

*Preserved across reruns / `reset_results`. Artifacts are referenced by relative path from this
folder (`<sub>/{figures,tables}/…` — no `<judge>/` level: every table here carries BOTH graders).
The auto-generated artifact map is [`INDEX.md`](INDEX.md).*

## What this top covers

`lookahead/` owns the **look-ahead contrast** — K=0 vs K=5 within PTO and within GRPO, persona-paired
on the 96 shared personas, under the primary oracle *and* the held-out judge side by side (columns,
never a mean). Five families:
[`reward/`](reward/) (the K contrast on the rubrics, the difference-in-differences, the method gap at
each K), [`transfer/`](transfer/) (does the contrast survive the held-out judge — sign ladder, gain
retention), [`behaviour/`](behaviour/) (channels, substitution, session shape, held-out instruments),
[`mechanism/`](mechanism/) (the over-praise chain, signal dispersion, reward faithfulness at a matched
policy, the K-step tail audit) and [`replication/`](replication/) (the ICLR transcripts re-scored under
the modern grader; the SD/stability claim).

**All four arms now run to iteration 10 and are fully scored by both graders** —
4 arms × 11 model states = 44 states, and the held-out grid is complete at 44 × 8 × 96 = 33,792 cells
with zero partial cells
([`../measurement/validity/tables/multijudge_coverage.md`](../measurement/validity/tables/multijudge_coverage.md)).
RQ-i is therefore a genuine, fully-matched **K×method** comparison over eleven matched points for
*both* methods. *(Corrected 2026-08-25: this section said "eleven matched points for PTO (0–10), six
for GRPO (0–5)". `GRPO_LA5` was right-censored for months; it finished. Every claim below that used
to be an iteration-5 statement has been re-read at iteration 10, and several changed sign.)*

⚠ **K has exactly two levels, {0, 5}.** Nothing here interpolates or extrapolates — "look-ahead helps
GRPO" means "K=5 beat K=0 for GRPO, at this budget, on this grader"
([`../LIMITATIONS.md`](../LIMITATIONS.md) §5a).

⚠ **Read every contrast in this top against [`../compute/SUMMARY.md`](../compute/SUMMARY.md).** An
iteration is not a fixed unit of spend, matched-iteration and matched-budget are different questions,
and here they **disagree** (§8).

---

## 1. The one-paragraph answer

**Look-ahead is a GRPO lever, not a PTO lever, and at the iteration-10 endpoint the GRPO effect is
large and unanimous.** At matched iteration 10 on Q1+Q2, GRPO's K=5 arm beats its K=0 arm by
**+0.765 (dz 0.905)** under the primary oracle and **+0.616 (dz 1.030)** under the held-out judge,
Holm p < .001 on both, and K=5 wins on **all nine rubric rows** — the eight instruments plus the
Q1+Q2 composite — under **both** graders
([`reward/tables/k_endpoints.md`](reward/tables/k_endpoints.md), rows
`GRPO_LA5_I10 − GRPO_LA0_I10`). PTO's K contrast at the same point is null-to-negative: **−0.047
(dz −0.096, p_holm .695)** on the primary, **+0.199 (dz 0.308, p_holm .227)** *against* K=5 on the
held-out judge ([`reward/tables/k_table1.md`](reward/tables/k_table1.md) and its long form
[`k_paired_long.md`](reward/tables/k_paired_long.md), sign K0 − K5, Holm across iterations). That
divergence is the whole result, and it is what makes the thesis's PTO-vs-GRPO verdict an
**interaction with K** rather than a ranking (§3).

**On the compute axis the sign is budget-dependent, and PTO's look-ahead never pays** (§8). Quote the
budget sweep, never a single iso-compute row.

Endpoint Q1+Q2 levels ([`reward/tables/k_levels.md`](reward/tables/k_levels.md)), primary oracle:
`GRPO_LA5` 4.517 · `PTO_LA5` 4.307 · `PTO_LA0` 4.260 · `GRPO_LA0` 3.753. Held-out judge:
`GRPO_LA5` 2.873 · `PTO_LA0` 2.866 · `PTO_LA5` 2.667 · `GRPO_LA0` 2.257. ⚠ Never compare a level
across the two columns — the held-out judge sits 1.2–1.7 points lower and the offset is
model-dependent.

## 2. `reward/` — the K contrast, iteration by iteration

Built by [`../../notebooks/lookahead/reward.ipynb`](../../notebooks/lookahead/reward.ipynb) into
[`reward/tables/k_means_by_iter.md`](reward/tables/k_means_by_iter.md) (levels),
[`reward/tables/k_paired_by_method.md`](reward/tables/k_paired_by_method.md) (Δ / *dz* / Holm *p*),
the paper's Table 1 shape in [`reward/tables/k_table1.md`](reward/tables/k_table1.md) (+ per-rubric
twins `k_table1_{Q1,Q2,PCT,MICI}.md`), the across-iteration tallies in
[`reward/tables/k_summary.md`](reward/tables/k_summary.md), and the endpoint pairs in
[`reward/tables/k_endpoints.md`](reward/tables/k_endpoints.md). Figures:
[`reward/figures/k_headline_q1q2.png`](reward/figures/k_headline_q1q2.png),
[`reward/figures/k_trajectory_Q1Q2.png`](reward/figures/k_trajectory_Q1Q2.png),
[`reward/figures/k_delta_grid_gpt-4o-mini.png`](reward/figures/k_delta_grid_gpt-4o-mini.png) (+ its
held-out twin), [`reward/figures/k_contrast_both_judges.png`](reward/figures/k_contrast_both_judges.png).

⚠ **Sign convention.** `k_table1`, `k_summary`, `k_paired_*` and `k_pairs` all report **Δ = K0 − K5**,
so a *negative* cell means **K=5 won**. `k_endpoints` reports the pair named in its `pair` column
(`GRPO_LA5_I10 − GRPO_LA0_I10` is K5 − K0, so *positive* means K=5 won there). State which one you
are quoting, every time.

**Nothing happens for the first two updates.** Over iterations 0–2,
2 methods × 3 iterations × 9 rubrics × 2 graders = 108 paired cells produce **zero**
Holm-significant results, and only 2 clear even an uncorrected p < .05 (both PTO under the held-out
judge at iteration 2). Iteration 0 is a clean null control — the same untrained policy in both arms,
two independent 96-conversation draws: 2 methods × 9 rubrics × 2 graders = 36 same-policy contrasts,
largest |dz| 0.147, none at raw p < .05
([`reward/tables/k_paired_long.md`](reward/tables/k_paired_long.md)).

**GRPO: look-ahead leads from iteration 4 and the lead widens to the end.** On Q1+Q2, K=5 is
Holm-significantly ahead at **6 of 11 iterations under each grader** — 4 and 6 through 10 on the
primary, 4 through 7 plus 9 and 10 on the held-out judge — with a mean paired delta over the trained
iterations of −0.218 (mean dz −0.304) primary and −0.301 (−0.386) held-out
([`reward/tables/k_summary.md`](reward/tables/k_summary.md)). The endpoint is the largest gap of the
run. Instrument by instrument at iteration 10 (K5 − K0, primary / held-out, every p_holm < .001):
Q1 +0.858 / +0.865, Q2 +0.671 / +0.367, WAI-SR +0.291 / +0.288, CSQ-8 +0.289 / +0.451,
MI-SAT +0.352 / +0.503, MITI +0.615 / +0.276, PCT +0.111 / +0.113, and MICI (lower is better)
−0.627 (dz −1.862) / −0.422 (dz −1.567). *(Corrected 2026-08-25: this section previously reported the
GRPO K effect as "leads at 4–5" with deltas around −0.1 to −0.3 — those were the only iterations that
existed. The effect is roughly three times larger at the endpoint. The same paragraph's claim that
"the first thing look-ahead buys GRPO is negative" rested on an iteration-3 MICI cell and on a
per-1,000-character normalisation that no artifact in `results/` computes; see §5.)*

**PTO: look-ahead never leads on the training reward, and on the held-out judge it costs.** Over
eleven matched iterations K=5 never leads Holm-significantly on Q1+Q2 under either grader. K=0 leads
significantly at iteration 6 on the primary (+0.257, dz 0.417) and at iterations 5, 6 and 8 under the
held-out judge (dz 0.33–0.51). The endpoint is a wash on the primary (−0.047, ns) and nominally
against K=5 on the held-out judge (+0.199, ns after Holm). The held-out edge is carried by **Q2** —
K=0 higher at 6 of 11 iterations (5 through 10; mean delta +0.223, dz 0.380) — and by **MITI** —
K=0 higher at 8 of 11 (3 through 10; +0.175, dz 0.370). That is the ICLR poster's own Q2-only K
finding, reversed, in this regime.

**The one thing look-ahead reliably buys PTO is less MI-inconsistency.** At iteration 10 the MICI
rate (MI-inconsistent behaviours **per therapist turn**, lower better) is K5 − K0 = −0.228
(dz −0.708) primary and −0.245 (dz −0.655) held-out, both p_holm < .001
([`reward/tables/k_endpoints.md`](reward/tables/k_endpoints.md)). §5 is where that goes.

⚠ **Iteration 9 is an outlier state for `GRPO_LA0`** on both graders (Q1+Q2 3.807 primary / 2.002
held-out, against 4.082 / 2.617 at iteration 8 and 3.753 / 2.257 at 10; mean therapist turn length
338 chars against 822 at 8 and 896 at 10 —
[`behaviour/tables/session_shape.md`](behaviour/tables/session_shape.md)). Every K and DiD row at
iteration 9 is inflated by it. Quote iteration 10, and treat 9 as a single draw with the shape of a
bad one ([`../LIMITATIONS.md`](../LIMITATIONS.md) §5c).

## 3. The K×method interaction — and the retirement of "the training grader is blind to it"

The PTO-vs-GRPO verdict is **not** a ranking; it is an interaction with K, and at the endpoint it is
significant on both graders. At iteration 10, on the same 96 personas
([`../method/contrast/tables/method_paired_by_K.md`](../method/contrast/tables/method_paired_by_K.md),
sign + = PTO higher):

| at iteration 10, Q1+Q2 | primary oracle | held-out judge |
|---|---|---|
| K = 0 · PTO − GRPO | **+0.507**, dz 0.729, p_holm <.001 — PTO wins | **+0.609**, dz 1.265, p_holm <.001 — PTO wins |
| K = 5 · PTO − GRPO | **−0.210**, dz −0.356, p_holm .0005 — GRPO wins | **−0.206**, dz −0.313, p_holm .034 — GRPO wins |

Computed persona by persona, the difference-in-differences
([`reward/tables/k_did.md`](reward/tables/k_did.md); the method gap at each K is
[`reward/tables/k_method_gap.md`](reward/tables/k_method_gap.md); figure
[`reward/figures/k_did.png`](reward/figures/k_did.png)) is **did 0.718, dz 0.793, CI [0.547, 0.898]**
on the primary and **did 0.815, dz 0.972, CI [0.647, 0.976]** held-out, both p_holm < .001. Positive
means PTO's lead over GRPO is larger at K=0 than at K=5 — equivalently, look-ahead helps GRPO much
more than it helps PTO. Every rubric agrees at the endpoint: on the held-out judge the DiD runs
Q1 0.900 (dz 0.784), Q2 0.730 (dz 0.964), WAI-SR 0.392, CSQ-8 0.452, MI-SAT 0.392, MITI 0.479.

⚠ **The claim that the primary oracle cannot see this interaction is RETIRED.** *(Corrected
2026-08-25: this file said "On the primary oracle the same interaction is null — largest dz 0.211,
nothing survives Holm. The grader that WAS the training reward cannot see an effect the held-out
grader measures at dz 0.5. That is the sharpest single argument in the thesis for why the second
judge exists." That was read at iteration 5 and is true only there — primary DiD 0.068, dz 0.095,
p_holm 1.000, against held-out 0.484, dz 0.525, p_holm < .001. From iteration 6 the primary sees it
too: dz 0.605, 0.277, 0.354, 0.799, 0.793 at iterations 6 through 10, every one p_holm ≤ .006. The
held-out judge does detect it **two iterations earlier** — first Holm-significant at iteration 4
(0.356, dz 0.401) against iteration 6 on the primary — and that earlier detection is a defensible,
much weaker version of the argument. The strong version must not be repeated.)*

The second judge still earns its place here, but on a different footing: it agrees, at a **larger**
effect size, about a contrast the training oracle also sees. That is corroboration rather than
correction; the circularity limitation ([`../LIMITATIONS.md`](../LIMITATIONS.md) §3) is bounded by
it, not confirmed by it. §4 measures transfer properly.

## 4. `transfer/` — does the held-out judge credit the gains?

Every number in this top is reported on both graders because the primary oracle **was** the training
reward; the held-out judge (Claude Haiku 4.5, a different model family, never played the patient,
never scored a training branch) has the full grid for every arm here.

**Sign preservation first.** Of the 2 methods × 11 iterations × 9 rubrics = 198 cross-K contrasts,
158 (79.8%) keep their sign under the held-out judge; that rises to 24/24 (100%) at
|Δ primary| ≥ 0.25 and to 49/49 (100%) where both graders call the contrast Holm-significant
([`transfer/tables/k_sign_ladder.md`](transfer/tables/k_sign_ladder.md); the contrast-by-contrast
table is [`transfer/tables/k_pairs.md`](transfer/tables/k_pairs.md)). ⚠ The pooled rate is
meaningless on its own — it counts contrasts too small to claim. The method split is itself a result:
GRPO 88/99 (88.9%) against PTO 70/99 (70.7%), i.e. PTO's K contrasts are both smaller and less
reproducible across graders.

**Gain retention** (`retention` = Δ held-out ÷ Δ primary over the arm's own base; ~1 = the gain is
real to a judge that never played the patient, ~0 = it existed only in the optimised grader) is
[`transfer/tables/k_retention_summary.md`](transfer/tables/k_retention_summary.md) (own-base
reference; the full table with four reference kinds is
[`transfer/tables/k_retention.md`](transfer/tables/k_retention.md), figure
[`transfer/figures/k_retention.png`](transfer/figures/k_retention.png)).

⚠ **Retention must be quoted per metric AND per iteration — the pattern is not uniform, and the
previous generalisation of it was wrong.** At the iteration-10 endpoint, own-base reference:

| iteration 10, own base | K=0 retention [CI] | K=5 retention [CI] | CIs disjoint? |
|---|---|---|---|
| GRPO · Q1 | 0.295 [0.054, 0.488] | 0.676 [0.578, 0.795] | **yes** — K=5 retains far more |
| GRPO · Q1+Q2 | 0.578 [0.457, 0.723] | 0.668 [0.587, 0.765] | no |
| GRPO · Q2 | 0.788 [0.658, 0.970] | 0.661 [0.581, 0.755] | no (K=0 nominally higher) |
| GRPO · MITI | 0.302 [0.146, 0.462] | 0.384 [0.305, 0.471] | no |
| PTO · Q1+Q2 | 0.823 [0.720, 0.947] | 0.639 [0.551, 0.738] | no (just barely) |
| PTO · Q2 | 0.849 [0.746, 0.977] | 0.562 [0.476, 0.652] | **yes** — K=5 retains *less* |
| PTO · MITI | 0.450 [0.357, 0.548] | 0.268 [0.181, 0.354] | **yes** — K=5 retains *less* |

*(Corrected 2026-08-25: this file said "GRPO: look-ahead makes the gains REAL to the held-out judge —
Q1 retention at the matched iteration 5: 1.08 [0.94, 1.27] (K=5) vs 0.73 [0.57, 0.92] (K=0) —
disjoint intervals." Two things were wrong. Those two intervals came from **different reference
kinds** in a cross-view join, not from the like-for-like own-base comparison; under own base at
iteration 5 the GRPO Q1 intervals are 0.786 [0.587, 1.003] (K=0) against 1.048 [0.901, 1.221] (K=5)
and they **overlap**. And the general reading does not survive to the endpoint: at iteration 10 only
**Q1** separates, and on Q2 the ordering is nominally the other way.)*

**PTO's half of the old reading does hold.** Look-ahead did not make PTO's claimed gains more
transferable — on Q2 and MITI it made them **less** so, with disjoint intervals at the endpoint.
Closing the flattery channel (§5) and making a gain real to a held-out judge are different things.

⚠ **Retention is a ratio, so read its denominator.** `GRPO_LA0`'s own primary gain at iteration 10 is
only 3.753 − 3.067 = 0.686 against `GRPO_LA5`'s 4.517 − 2.963 = 1.554
([`reward/tables/k_levels.md`](reward/tables/k_levels.md)). A small, wobbling denominator is exactly
why GRPO's K=0 Q1 retention interval is as wide as [0.054, 0.488]. Retention below the
`min_primary_delta` floor is blanked rather than reported — `GRPO_LA5`'s MICI retention at iteration
10 is blank for that reason (its primary MICI delta is 0.001).

Never average the two graders — that is train-vs-test, not two raters — and never compare a level
across them. Combine contrasts or standardized quantities only. The judge's own validity (ICC,
variance decomposition, all-pairs sign preservation) is
[`../measurement/SUMMARY.md`](../measurement/SUMMARY.md); the all-arm retention table is
[`../measurement/validity/tables/multijudge_gain_retention.md`](../measurement/validity/tables/multijudge_gain_retention.md).

## 5. `behaviour/` — over-praise closes under K=5 in both optimizers, and PTO relocates it

Sources: [`behaviour/tables/k_mici_composition.md`](behaviour/tables/k_mici_composition.md)
(per-session counts **and** denominator-free shares),
[`behaviour/tables/k_paired_channels.md`](behaviour/tables/k_paired_channels.md) (paired tests),
[`behaviour/tables/k_channels_summary.md`](behaviour/tables/k_channels_summary.md) (across-iteration
tallies); the per-grader long forms are `behaviour/tables/k_channels_{pto,grpo}_<judge>.md`. Figures:
[`behaviour/figures/k_mici_composition_grid_gpt-4o-mini.png`](behaviour/figures/k_mici_composition_grid_gpt-4o-mini.png)
(+ `_claude-haiku-4-5.png`),
[`behaviour/figures/k_overpraise_trajectory_gpt-4o-mini.png`](behaviour/figures/k_overpraise_trajectory_gpt-4o-mini.png),
[`behaviour/figures/k_channel_forest_gpt-4o-mini.png`](behaviour/figures/k_channel_forest_gpt-4o-mini.png).

**Over-praise closes under K=5 in both optimizers, on both graders.** Per session at iteration 10 —
counts, then that channel's share of all coded MI-inconsistent acts:

| iteration 10, per session | over-praise K=0 → K=5 | MICI total K=0 → K=5 | over-praise share K=0 → K=5 |
|---|---|---|---|
| GRPO · primary | 8.250 → 0.719 | 9.865 → 2.906 | 0.836 → 0.247 |
| GRPO · held-out | 10.188 → 3.458 | 13.000 → 9.781 | 0.784 → 0.354 |
| PTO · primary | 3.042 → 0.625 | 4.958 → 3.344 | 0.613 → 0.187 |
| PTO · held-out | 4.750 → 1.177 | 8.510 → 7.979 | 0.558 → 0.148 |

Paired, the over-praise contrast is dz 2.053 (primary) / 1.306 (held-out) for GRPO and 0.887 / 0.999
for PTO, every one p_holm < .001.

**The substitution result is that the over-praise drop does not fully reach the total — and it
reaches it much less under PTO.** Take the fraction of the per-session over-praise reduction that
survives into `MICI_BehaviorTotal`: under the primary, PTO keeps 1.615 / 2.417 = 0.668 of it while
GRPO keeps 6.958 / 7.531 = 0.924; under the held-out judge, PTO keeps 0.531 / 3.573 = 0.149 while
GRPO keeps 3.219 / 6.729 = 0.478. On both graders GRPO retains the larger share. The displaced volume
goes to **advice without permission**, whose share climbs to 0.545 (GRPO) and 0.570 (PTO) on the
primary and to 0.588 / 0.701 on the held-out judge; paired, that channel is Holm-significantly higher
under K=5 at 6 of 11 iterations for PTO against 3 for GRPO under the held-out judge (3 against 1 on
the primary — [`behaviour/tables/k_channels_summary.md`](behaviour/tables/k_channels_summary.md)).
Under PTO on the held-out judge the aggregate does not move at all (Δ +0.531, dz 0.099, ns): one
channel is suppressed, the total is not. **Substitution is a PTO-dominant pattern at the endpoint,
not a symmetric one.**

⚠ **Name the axis on every one of these.** The three denominators are per therapist TURN
(`MICI_Rate`), per SESSION (counts), and share of coded acts — and only the **share** has no moving
denominator ([`../LIMITATIONS.md`](../LIMITATIONS.md) §5b). Prefer it. At the completed endpoint all
three happen to agree in direction for both methods — the per-turn rate at iteration 10 is
K0 − K5 = +0.228 (dz 0.708) / +0.245 (dz 0.655) for PTO and +0.627 (dz 1.862) / +0.422 (dz 1.567) for
GRPO, i.e. K=5 better everywhere — but that agreement is a property of iteration 10, not of the
contrast; at iteration 5 the three disagreed.

⚠ **Do not quote a per-1,000-character MI-act rate.** *(Corrected 2026-08-25: this file carried a
whole sub-argument — "the per-1k-character improvement is dilution, not skill", with figures like
"MI-consistent 3.32 → 1.68" and a normalisation table row "per 1,000 therapist CHARS". **No artifact
in `results/` computes a per-1,000-character rate.** Those numbers came from a retired generator and
cannot be checked against any table here. The underlying worry is real and is now stated as an
argument from the length tables in [`../LIMITATIONS.md`](../LIMITATIONS.md) §5b, not as a measured
result.)*

### 5b. Session shape — the ICLR "K=5 gives shorter conversations" claim, at the endpoint

Judge-free, computed from the transcripts:
[`behaviour/tables/session_shape.md`](behaviour/tables/session_shape.md),
[`behaviour/tables/length_endpoints.md`](behaviour/tables/length_endpoints.md),
[`behaviour/tables/length_kcontrast.md`](behaviour/tables/length_kcontrast.md); figure
[`behaviour/figures/session_shape.png`](behaviour/figures/session_shape.png).

At iteration 10, **K=5 takes MORE therapist turns in both optimizers**: PTO 10.229 → 14.385
(Δ −4.156, dz −0.548) and GRPO 12.750 → 15.969 (Δ −3.219, dz −0.415), both p_holm < .005; sessions
run 8.312 and 6.698 utterances longer respectively. The method-dependence has moved into turn
**length**: K=5 writes longer turns under PTO (686.202 → 810.875 chars, dz −0.555) and slightly
*shorter* ones under GRPO (895.711 → 849.274, dz +0.519).

*(Corrected 2026-08-25, and corrected again the same day on the character figures. This file said
"GRPO's K=5 arm takes **26% fewer** therapist turns at iteration 5 (11.31 vs 15.34) while writing
**1.7× longer** turns (678 vs 394 chars)"; its now-retired cross-K bullet list said "GRPO K=5 −8.1
utterances *shorter* at iter 5 (dz 0.53)"; and a retired verbosity bullet gave `GRPO_LA0` chars per
therapist turn as "394 @5 → 905 @10". Split the two channels:*
- *The **turn** figures hold. [`behaviour/tables/session_shape.md`](behaviour/tables/session_shape.md),
  GRPO at iteration 5: `n_th_turns` 15.344 (K=0) against 11.312 (K=5), and 11.312 / 15.344 = 0.737,
  so "26% fewer" is right; `conv_len` 30.677 against 22.573, i.e. K=5 shorter by a paired 8.104
  utterances, dz 0.531.*
- *The **character** figures do not. `mean_turn_len` at iteration 5 is 461.787 (K=0) against 668.343
  (K=5): 668.343 / 461.787 = 1.447, a **1.45×** ratio, not 1.7×, and neither 678 nor 394 is a GRPO
  iteration-5 cell of that table. `GRPO_LA0` at iteration 10 is **895.711**, not 905. All three
  figures match no cell of the rows they were attributed to; they came from a retired generator and
  must not be requoted.*

*What survives is the structure, and it is what the paragraph above states from the table: at
iteration 5 GRPO's K=5 arm wrote fewer, longer turns, and **both signs reverse by iteration 10**
(12.750 → 15.969 therapist turns, 895.711 → 849.274 chars). Any prose that reads the ICLR shortening
claim off GRPO must name the iteration.)*

### 5c. The question channel — a deterministic, judge-free signature of the K=0 collapse

`q_per_turn` counts literal `?` per therapist turn straight from the transcript; no grader is
involved. At iteration 10 `GRPO_LA0` has all but stopped asking questions — **0.151** per therapist
turn against `GRPO_LA5`'s **0.719** (Δ −0.568, dz −1.303, p_holm < .001) — and K=5 is
Holm-significantly higher at **7 of 11 iterations** (4 through 10). PTO shows nothing on this channel
at any iteration ([`behaviour/tables/k_channels_text_grpo.md`](behaviour/tables/k_channels_text_grpo.md),
[`behaviour/tables/k_channels_summary.md`](behaviour/tables/k_channels_summary.md)).

The generation side agrees, from a completely different artifact: the mean question rate over **all**
candidates the policy generated falls 0.710 → 0.063 across `GRPO_LA0`'s ten training iterations while
`GRPO_LA5` holds 0.665 → 0.572 ([`behaviour/tables/selection.md`](behaviour/tables/selection.md),
`pool_question`). This is the cleanest single piece of evidence in the top that K=0 and K=5 put GRPO
on genuinely *different* policies rather than on one policy at two strengths — and it needs no judge
at all, which is why it is worth more than its effect size suggests.

Same table, the update-selection side: **`PTO_LA5` is the only arm whose update pushes for length at
every training iteration** (`w_len` +49.480 at train_iter 1 falling to +7.016 at 10, positive
throughout); `PTO_LA0`, `GRPO_LA0` and `GRPO_LA5` all change sign at least once.

### 5d. Who the gain lands on, and the held-out instruments

[`behaviour/tables/hetero_kcontrast.md`](behaviour/tables/hetero_kcontrast.md) splits the K contrast
by the patient's cooperation trait (32 personas each); figure
[`behaviour/figures/hetero.png`](behaviour/figures/hetero.png).

**GRPO's look-ahead gain is concentrated on the hard patients.** At iteration 10 on the primary,
Δ = K0 − K5 on Q1+Q2 is −0.203 (dz −0.310, ns after Holm) for Cooperative personas, **−0.961
(dz −1.281)** for Warms-up and **−1.130 (dz −1.358)** for Resistant. The Cooperative stratum is
simply at ceiling: 0.906 of `GRPO_LA0`'s and **1.000** of `GRPO_LA5`'s cooperative conversations
score ≥ 4.5. Under the held-out judge, which has no ceiling problem here, all three strata favour
K=5 (−0.686 / −0.807 / −0.355, all Holm-significant).

**PTO's held-out K=0 edge is entirely a Cooperative-stratum effect.** Same table, held-out judge,
iteration 10: Cooperative **+0.571 (dz 0.773, p_holm .0017)** for K=0, Warms-up −0.072 (ns),
Resistant +0.099 (ns). On the primary all three strata are flat (largest |dz| 0.334, at Warms-up).

Three instrument-level reads, all at iteration 10, all on both graders:

- **Change talk rises under K=5 for GRPO, everywhere but the ceiling**
  ([`behaviour/tables/pct_kcontrast.md`](behaviour/tables/pct_kcontrast.md)): Δ = −0.111 (dz −0.516)
  primary and −0.113 (dz −0.563) held-out overall, driven by Warms-up (−0.185, dz −1.058 primary) and
  Resistant (−0.146, dz −0.506); Cooperative is flat at ~0.88 either way. PTO shows a much smaller
  version (−0.051, dz −0.253, p_holm .030, held-out only).
- **K=0's alliance gain is bond-weighted; K=5's shifts to goal/task**
  ([`behaviour/tables/wai_kcontrast.md`](behaviour/tables/wai_kcontrast.md), `bond_excess` =
  Bond − mean(Goal, Task)): PTO +0.221 (dz 0.430) primary / +0.270 (dz 0.453) held-out; GRPO +0.245
  (dz 0.500) / +0.503 (dz 0.699), all p_holm ≤ .0015. This now holds for **both** methods; the older
  reading quoted only PTO's numbers and generalised them.
- **The two emotional self-disclosure items are a K=0 signature in both optimizers**, even where K=5
  wins Q2 overall ([`behaviour/tables/q2_items_kcontrast.md`](behaviour/tables/q2_items_kcontrast.md),
  held-out judge): item 3 "shared his feelings" +1.104 (dz 1.141) PTO and +1.323 (dz 1.271) GRPO;
  item 10 "said when happy/sad" +1.031 (dz 1.001) and +1.490 (dz 1.452). GRPO's K=5 arm meanwhile
  leads by ~0.8–1.0 on the empathy, warmth and fluency items. Look-ahead trades therapist
  self-disclosure for perceived understanding.

## 6. `mechanism/` — what look-ahead does to the training signal

**Look-ahead RESCALES the signal; it does not sharpen it.**
[`mechanism/tables/dispersion_ratios.md`](mechanism/tables/dispersion_ratios.md) (figure
[`mechanism/figures/dispersion.png`](mechanism/figures/dispersion.png)): pooled over training
iterations, K=5 widens both the best–worst margin and the within-group SD by almost exactly the same
factor — PTO margin ×1.504 and SD ×1.475 (ratio-of-ratios **1.019 [1.014, 1.025]**), GRPO margin
×1.300 and SD ×1.293 (**1.006 [1.002, 1.010]**). Margin-over-SD sits at 2.946–3.003 in every arm, at
or just under the **3.154** iid-normal expectation for 8 draws
([`mechanism/tables/dispersion_expectation.md`](mechanism/tables/dispersion_expectation.md)); the
"does the winner stand out?" statistic `winner_z` moves by −0.002 [−0.021, 0.017] for PTO and +0.029
[0.015, 0.041] for GRPO, against an expectation of 1.576. A wider spread is not better
discrimination.
⚠ One exception worth knowing: at GRPO's **train_iter 10** the ratios invert (margin ×0.679,
SD ×0.705) — the K=5 arm's own spread collapsed as its policy converged.

**The τ-yield companion is only true with its conditions attached**
([`mechanism/tables/dispersion_tau.md`](mechanism/tables/dispersion_tau.md) — **PTO only**; that
table has no GRPO rows). At the trainer's own **τ = 0.10**, multiplying every K=0 margin by
r1 = 1.530 — the K=5/K=0 within-group SD ratio at **train_iter 1**, the one cut where both arms are
still the same policy — lifts K=0's pair yield from 0.824 to 0.872 against K=5's raw 0.935, i.e. it
closes 0.437 of the 0.935 − 0.824 = 0.111 gap (`share_gap_closed_r1`). The median over the seven
training iterations whose raw gap clears 0.05 (1 through 7) is 0.572; **pooled over all ten the
rescale overshoots** (1.127 — more than the whole gap); and at train_iters 8–10 the raw gap is ≈0 or
negative, because K=5's own spread shrank as its policy diverged, so the share is undefined there and
the gap must be quoted in yield points instead. *(Corrected 2026-08-25: this said "About half of
PTO's higher τ-yield under K=5 is reproduced by rescaling K=0's margins alone", with none of those
conditions. "About half" is the **train_iter-1, τ = 0.10** value; at the same τ the pooled figure is
more than all of it, and at other τ or at the last three iterations the share is different or
undefined. A bare "about half" is not recoverable from the table.)*

**Reward faithfulness — read [`../METRICS_REFERENCE.md`](../METRICS_REFERENCE.md) §6a before writing
anything here.** It disambiguates two cuts that sound contradictory, and its rule is binding:

- *"Over the run, the K=5 arms' training signal ranked conversations more like the full-session eval
  did"* → the **`matched_iters`** rows of
  [`mechanism/tables/faithfulness_k_summary.md`](mechanism/tables/faithfulness_k_summary.md)
  (train_iters 1 through 10, all ten realized). Primary: GRPO 0.873 [0.861, 0.884] against 0.909
  [0.900, 0.917], delta −0.036 [−0.051, −0.021]; PTO 0.836 against 0.863, delta −0.027
  [−0.048, −0.007]. Held-out: GRPO delta −0.053 [−0.078, −0.030]; PTO −0.012 [−0.036, 0.013], CI
  covering 0. **Name the grader** (significant for GRPO on both, for PTO on the primary only) and say
  that the two arms are different policies scored on different conversation sets.
- *"Look-ahead helps **because** it makes the training reward a better proxy"* → this is a mechanism
  claim, so it needs the matched-policy **`train_iter_1`** cut, and that cut does **not** support it.
  All four deltas straddle zero: primary PTO +0.004 [−0.069, 0.073] and GRPO +0.015 [−0.026, 0.059];
  held-out PTO +0.007 [−0.076, 0.092] and GRPO −0.014 [−0.075, 0.048]. Per bin, the sign even flips
  with the grader for GRPO
  ([`mechanism/tables/faithfulness_matched_policy_tests.md`](mechanism/tables/faithfulness_matched_policy_tests.md)),
  which is the shape of no effect. **Report the mechanism as unsupported, not as established.**

⚠ Quoting either cut as a refutation of the other is the error §6a exists to prevent; they condition
differently and both are correct. And faithfulness is agreement between two *rankings* — a higher
`agr_K5` says nothing about which arm *scores* higher, which is §2's question.
*(Corrected 2026-08-25: the retired bullet quoted only the `train_iter_1` numbers, under the heading
"look-ahead rescales the training signal", with no mention that the `matched_iters` cut points the
other way. Both belong, each with its condition named.)*

**The over-praise chain** ([`mechanism/tables/k_mechanism_overpraise_chain.md`](mechanism/tables/k_mechanism_overpraise_chain.md),
figure [`mechanism/figures/k_mechanism_overpraise.png`](mechanism/figures/k_mechanism_overpraise.png))
follows the channel through all three levels it must pass to become policy, for the two PTO arms.
What the update *selects for* (`w_overpraise`) is within a standard error of zero almost everywhere in
both arms; what the policy *generates* (`pool_overpraise`) drifts 0.0016 → 0.3178 for `PTO_LA0` but
only 0.0035 → 0.0649 for `PTO_LA5`; and the eval's coded rate follows, 0.0129 → 0.2990 against
0.0081 → 0.0432. The reading is a **compounding on-policy loop**, not one hard pull by the update.

**What the K-step reward actually sees** ([`mechanism/tables/tail_audit_by_iter.md`](mechanism/tables/tail_audit_by_iter.md),
[`mechanism/tables/tail_within_group.md`](mechanism/tables/tail_within_group.md), figure
[`mechanism/figures/tail_audit.png`](mechanism/figures/tail_audit.png)): pooled over training,
**22.8% [22.5, 23.2] of `PTO_LA5`'s tails and 18.2% [17.9, 18.4] of `GRPO_LA5`'s end before the fifth
simulated turn**, almost always because the simulated patient closed the session (0.211 and 0.156 of
all candidates). Within a group, ended-early siblings score *lower* than full-tail ones (−0.034,
dz −0.244 for PTO; −0.041, dz −0.219 for GRPO) and are about 23% less likely to be the arg-max (risk
ratio 0.770 [0.738, 0.804] and 0.773 [0.749, 0.797]). Look-ahead therefore carries a mild, systematic
penalty on candidates that lead to an early close. The API-call side of the same rollouts is
`api_calls.md` / `api_ratio.md` in [`../compute/cost/tables/`](../compute/cost/tables/).

## 7. `replication/` — the ICLR ordering reproduces; the ICLR stability claim does not

**Re-scoring the poster's own Exp1 transcripts under the modern grader keeps K=5 above K=0.**
15 model states × 96 = 1,440 conversations from Exp1, re-scored with gpt-4o-mini
([`replication/tables/crossgen_kcontrast.md`](replication/tables/crossgen_kcontrast.md), figure
[`replication/figures/crossgen.png`](replication/figures/crossgen.png)): the K0 − K5 delta on the
paper's `Final` metric is negative at **all 7 iterations**, Holm-significant at 3 (−0.205, dz −0.289)
and 7 (−0.284, dz −0.371). Averaged over iterations 1–7 the arm-level effect is −0.132 (dz −0.543)
under the modern grader against −0.206 (dz −0.612) under the original GPT-3.5
([`replication/tables/crossgen_kcontrast_summary.md`](replication/tables/crossgen_kcontrast_summary.md)),
and the two graders correlate at Spearman 0.836 over the 15 model means
([`replication/tables/crossgen_grader_agreement.md`](replication/tables/crossgen_grader_agreement.md)).

**So PTO's null in Exp3 is a property of the REGIME, not of the judge** — 1B therapist, V3 patients,
MCL=12, iterative regeneration, bf16 — and the same grader that reports the null reproduces the
original ordering on the original transcripts. This is the strongest available defence of the null
and should be stated whenever the null is. ⚠ The K=3 rows in
[`replication/tables/crossgen_la3_gpt35.md`](replication/tables/crossgen_la3_gpt35.md) are **Exp1's**
sweep and were never re-scored by the Exp3 grader; they are not a dose point
([`../LIMITATIONS.md`](../LIMITATIONS.md) §5a).

**The ICLR "lowest SD = more stable" claim fails here, as a ceiling artefact**
([`replication/tables/sd_tally.md`](replication/tables/sd_tally.md),
[`replication/tables/sd_summary.md`](replication/tables/sd_summary.md),
[`replication/tables/ceiling.md`](replication/tables/ceiling.md), figure
[`replication/figures/sd.png`](replication/figures/sd.png)). On Q1+Q2 under the primary, PTO's K=5 arm
has the *lower* SD at **0 of 10** iterations (median SD ratio K5/K0 1.174; Pitman–Morgan
Holm-significant with K=0 lower at 4). Across 40 trained model states, Spearman(mean, SD) is
**−0.844** (p < .001) — SD falls because the mean approaches the top of the scale. The tell: the
lowest-SD state in the whole grid, `GRPO_LA5` at iteration 10 (SD 0.559), is also the
**highest-mean** state (4.517), and 1.000 of its cooperative conversations sit ≥ 4.5 with 0.344 at a
flat 5. Under the held-out judge, which has no ceiling here, the association is −0.289 (p 0.071), the
median PTO SD ratio is 1.008, and no Pitman–Morgan test survives Holm in either direction. **SD is
not a stability measure on a bounded, saturating scale.** *(Corrected 2026-08-25: the retired bullet
gave "Spearman(mean, SD) −0.87 over 35 states"; the completed grid has 40 trained states and the
value is −0.844.)*

## 8. The cost axis — matched iteration and matched budget disagree

Whole-arm GPU-hours
([`../compute/cost/tables/compute_by_arm.md`](../compute/cost/tables/compute_by_arm.md)):
`PTO_LA0` 8.119 · `PTO_LA5` 19.681 · `GRPO_LA0` 27.906 · `GRPO_LA5` 51.205. So look-ahead cost
51.205 / 27.906 = 1.835× within GRPO and 19.681 / 8.119 = 2.424× within PTO, and the GRPO per-step
ratio at the endpoint is 1.874
([`../compute/cost/tables/step_multiplier.md`](../compute/cost/tables/step_multiplier.md)). **Every
matched-iteration row in this top hands the K=5 arm roughly twice the compute per cell.**
*(Corrected 2026-08-25: this file asserted in two places that GRPO's two arms were budget-matched —
"on the compute axis, iteration 5 is not an early stopping point for GRPO_LA5, it is the arm's full
budget" and "budget-matched to `GRPO_LA0` within 3%". That was an artifact of the censoring. The two
GRPO arms are nowhere near iso-compute, and any argument built on "look-ahead was free here" is
void.)*

⚠ **Quote the sweep, never a single iso-compute row — the lever's sign is a function of budget.**
On Q1+Q2, Δ = K0 − K5
([`../compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md`](../compute/cost/tables/budget_sweep_GRPO_K_gpt-4o-mini.md)
and its [held-out twin](../compute/cost/tables/budget_sweep_GRPO_K_claude-haiku-4-5.md);
[PTO primary](../compute/cost/tables/budget_sweep_PTO_K_gpt-4o-mini.md) and
[PTO held-out](../compute/cost/tables/budget_sweep_PTO_K_claude-haiku-4-5.md)):

- **GRPO, primary:** K=5 is **worse** at every rung up to 18.31 GPU-h (+0.569, dz 0.742,
  p_holm < .001 at 13.27), **level** at 23.21–27.08, and **better** from 30.53 on — first
  Holm-significant at 35.29 (−0.188, dz −0.310, p_holm .020), reaching −0.435 (dz −0.743) at 51.20.
- **GRPO, held-out:** at 18.31 K=0 is still **nominally ahead** (+0.051, dz 0.108, ns). The sign
  flips at the **same rung as on the primary** (23.21), but there it is Holm-significant
  *immediately* (−0.147, dz −0.331, p_holm .012) instead of only at 35.29 — i.e. 35.29 − 23.21 =
  12.08 GPU-h earlier — and it holds at −0.275 (dz −0.489) to the top rung. What the held-out judge
  brings forward is the **significance** of K=5's lead, not the crossover itself.
  *(Corrected 2026-08-25: this bullet read "the crossover is earlier — level at 18.31 (−0.051, ns)".
  The cell is **+0.051**, and under this table's Δ = K0 − K5 convention a positive delta means K=0 is
  ahead — so 18.31 is not a crossover point on either grader, and the "earlier crossover" conclusion
  drawn from it was wrong. Sign, rung and conclusion are all replaced above.)*
- **PTO, both graders:** K=0 is ahead at **every** rung. The best K=5 ever manages is a nominal
  −0.047 (dz −0.096, ns) at its full 19.68 GPU-h on the primary, against +0.186 (dz 0.323,
  p_holm .011) still favouring K=0 on the held-out judge.

At the top rung the GRPO verdict is unanimous across all four select-judge × eval-judge combinations
([`../compute/cost/tables/budget_sweep_crossjudge_verdicts.md`](../compute/cost/tables/budget_sweep_crossjudge_verdicts.md)).
So **"look-ahead does not pay for GRPO" is a low-budget statement, not a general one — and for PTO it
is true at every budget measured.**

⚠ The MI-consistency reading at matched budget is selection- *and* grader-dependent and must never be
quoted bare; the three-way flip is worked out in [`../LIMITATIONS.md`](../LIMITATIONS.md) §5.

## 9. Caveats

- **Every claim needs its axis named** — iteration against budget (§8), and for behaviour, per turn
  against per session against share of coded acts (§5). They disagree, and each is a correct answer
  to a different question ([`../compute/SUMMARY.md`](../compute/SUMMARY.md)).
- **The two graders are never averaged, and levels are never compared across them.** The held-out
  judge's offset is 1.2–1.7 points and model-dependent. Contrasts and standardized quantities only.
- **The instruments carrying §5 are the least dependable ones.** `dependability_k1` is **0.624 for
  MITI** and **0.812 for MICI** over `n_arms = 44`
  ([`../measurement/validity/tables/multijudge_variance_components.md`](../measurement/validity/tables/multijudge_variance_components.md)),
  the two weakest in the battery (the next weakest is Q2 at 0.914). There is **no channel-level
  reliability estimate at all** — the MICI ICC is computed on `MICI_Rate` only, never on
  `MICI_OverPraise` or `MICI_AdviseNoPermission`, which is the granularity every substitution claim
  lives at. *(Corrected 2026-08-25: this read "MITI dependability is 0.553 and MICI 0.628", values
  from an earlier and smaller grid.)*
- **Every endpoint is a single 96-conversation draw, and no K=5 state has a repeatability rep.**
  Reps exist for 3 metrics × 4 model states = 12 rows, all K=0, so cross-judge agreement on either
  look-ahead arm cannot be benchmarked against a ceiling. The only measured noise floor is at the base
  (§2), where sessions are short and homogeneous; therapist decoding has no per-call seed, so no
  conversation set is reproducible ([`../LIMITATIONS.md`](../LIMITATIONS.md) §5c).
- **All 96 personas are used for both training and eval**, so everything is in-sample with respect to
  the patient distribution, and the cooperative stratum saturates the primary oracle's scale (§7).
- **One training run per arm.** Run-to-run training variance is entirely unmeasured, which matters
  most for `GRPO_LA0`, whose trajectory is the least monotone of the four
  ([`../LIMITATIONS.md`](../LIMITATIONS.md) §5g).
- ✅ *(Resolved 2026-08-25 — kept so the warning's history is visible.)* This file warned that the
  rendered `CAPTIONS.md` in this top still carried right-censoring boilerplate for `GRPO_LA5`
  ("(PTO 10, GRPO 5 …)", "no matched K=5 model state (GRPO after 5)"). The tree-wide caption purge
  later the same day (see STATUS.md) removed every such string — support is now DERIVED per render
  (`constants.support_note`), so a caption can no longer hard-code where an arm stops. Verified by
  grep on 2026-08-26: no `CAPTIONS.md` under this top asserts censoring.
- **Bootstrap seeds.** The promoted modules seed with `constants.BOOT_SEED`; the paper generators used
  other seeds, so CI *bounds* in the rendered tables differ from the paper's frozen tables at
  Monte-Carlo scale (≤ ~0.02 on the rubric scale; one or two `judge_ci_excl0` flags flip in
  `transfer/`). Point estimates, dz, p and n reproduce exactly.

---

**Artifact note (2026-08-26).** The GRPO-scoped paper's figure rescope added GRPO-only,
levels-style companions in this top: `reward/figures/k_headline_q1q2_grpo` (redesigned to levels +
Holm-star row; its data table keeps the full delta columns), `reward/figures/k_levels_grid_grpo_<judge>`
(the 9-rubric battery in levels), `mechanism/figures/tail_audit_grpo`, and
`behaviour/figures/k_channel_forest_grpo_<judge>` (⚠ the un-suffixed `k_channel_forest_<judge>` is
the **PTO** forest). The four-arm and delta artifacts are unchanged and stay canonical.
