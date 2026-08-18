# Brainstorm — "Does look-ahead help MI sessions?" (2026-08-18)

Working note, not a draft. Built the way CLAUDE.md § "Epistemic status" demands: six readers
cold-read every table in `eda/results/{L5,L0}/tables/**` (narrative docs forbidden), four framing
lenses proposed papers from those facts alone, then two agents diffed against STATUS / SUMMARY /
LIMITATIONS / both drafts / the 08-18 deck, 17 load-bearing numbers were adversarially re-opened
(9 CONFIRMED, 8 PARTLY — wording, never the number), and one scout checked what new EDA the
data on disk can actually support. Every number below is one the verifier re-opened.

---

## 0. Bottom line

1. **There is room for exactly one more look-ahead paper, and it is not the one the four lenses
   wrote first.** All four framings put over-praise suppression / MICI composition / "judge-dependent
   total" at their centre — that is the *entire* content of `2026_lookahead_hack_substitution`
   ("The Hack Moves"). `papers/README.md` forbids shared claims across drafts. What is genuinely
   **unclaimed** anywhere (only STATUS.md + deck slides 10/15/16 carry it):
   - (a) the **reward-level K answer per optimizer** — PTO: K=0 ≥ K=5, significant at scattered
     iterations, on Q2 (a direct non-replication of the ICLR paper's own Q2-only K effect);
     GRPO: K=5 > K=0 at iters 4–5 on Q1, both graders, right-censored at 5;
   - (b) the **K × method interaction** and the held-out-judge method-ranking reversal;
   - (c) **gain retention by K** (GRPO 1.08 vs 0.73; PTO Q2 0.56 vs 0.85);
   - (d) the **compute axis** as a first-class result (1.9×/step, budget sweeps on both methods
     and both judges, PTO 3.4× cheaper than GRPO);
   - (e) **session shape + the selection-level length push** (PTO K=5 sessions +8.3 utterances,
     GRPO K=5 −8.1; `w_len` positive at every PTO_LA5 iteration);
   - (f) the **claim-by-claim ICLR replication** — "shorter = focused" reverses for PTO,
     "lowest SD = stable" has no Exp3 artifact at all.
2. **Recommended shape: Framing 1 with Framing 2 folded in as its cost section** — title in the
   family of *"Same lever, different optimizer: K-turn look-ahead helps GRPO but not PTO in
   simulated Motivational Interviewing"*. Framing 3 ("helps according to whom") is the
   substitution draft re-indexed with GRPO grafted on; Framing 4 (training-signal mechanism)
   has thin novel residue and its lead claim ("wider margins = sharper signal") is the reading
   the 08-10 memory already killed (scale, not information).
3. **This competes with the thesis chapter for the same material.** Deck Decision 1 recorded
   Lior's read as (a): submit the K=0 paper first, make look-ahead + K × method the *thesis core*.
   Whether this brainstorm becomes a third short paper (deck option c) or the thesis chapter's
   outline is the live decision — the outline below serves either.
4. **The single biggest hole is structural, not scientific: no artifact puts LA0 and LA5 on one
   axis.** L5 = the two K=5 arms only; L0 = the two K=0 arms only. Every reader hand-subtracted
   across views. The only cross-K artifacts are `7_stats/<judge>/k_*` (paired, per judge folder)
   and `6_preference/.../k_mechanism_overpraise_chain` (PTO only). 8_measurement has *never*
   judge-tested a cross-K contrast. Figure 1 / Table 1 of any look-ahead paper does not exist yet
   (§5, rank 1).

---

## 1. What the ICLR paper actually claimed (read off `papers/2025_iclr_pto_lookahead/submitted/paper.pdf`)

Setup: Llama-2-7B, GPT-3.5 patient + oracle, 96 profiles, K ∈ {0,5}, 7 iterations, τ = 0.1,
reward = mean(Q1, Q2), **eval = the training reward**. Table 4 (Tukey): L0-M4 vs L5-M7 —
Final +0.206 (p = .081, n.s.), Q1 +0.221 (p = .21, n.s.), **Q2 +0.191 (p = .032, sig.)**,
length −3.94 (p = .099). So the published K effect is **Q2-only, and only at each arm's post-hoc
best iteration**. Secondary claims: L5-M7 has the lowest SD ("more stable"), shortest
conversations (43.7 → 34.4 turns, read as "focused"). Untested there: any second optimizer,
any held-out instrument or judge, cost, mechanism, seeds.

Exp3 is positioned to **replicate** (K=5 > K=0 for PTO on Q1/Q2), **extend** (GRPO, 8 instruments,
held-out judge, GPU-hours, mechanism), and **overturn** ("stable", "shorter", "generalizable").

---

## 2. The verified spine (numbers the verifier re-opened; both judges unless stated)

**R1 — Reward-level K effect is method-conditional.**
- PTO, `k_paired_by_method` (K0−K5 on Q1+Q2), primary iters 1–10:
  +0.083, +0.158, +0.141, +0.120, −0.002, **+0.257 (dz .42, p_holm .000)**, +0.044, +0.077,
  +0.041, −0.047 → mean 0.872/10 = +0.087. Held-out: +0.060 … +0.199, positive 10/10,
  Holm-sig at iters 5/6/8 (dz .33/.51/.34); mean 1.615/10 = +0.16. **K=5 never *significantly*
  beats K=0** (numerically ahead at iters 5 and 10, n.s.). The K=0 edge is *larger on Q2*
  (primary iter 8: Q1 +0.008 vs Q2 +0.145 dz .33; held-out iter 10: Q1 +0.035 vs Q2 +0.363
  dz .65) — the ICLR Q2 finding, reversed. Endpoint: 4.307 vs 4.260 (dz −.10, p_holm .70) /
  2.667 vs 2.866 (dz .31, p_holm .13).
- GRPO: K=5 > K=0 at iter 4 on both judges (primary −0.115 dz −.25 p_holm .037; held-out
  −0.233 dz −.37 p_holm .005) and iter 5 held-out only (−0.311 dz −.43 p_holm .006); the K=5
  edge sits on **Q1** (iter 4 primary Q1 dz −.36 vs Q2 −.07). Right-censored: GRPO_LA5 has
  6 states (0–5), GRPO_LA0 collapses at iters 9–10 (it9−it8 −0.275 dz −.41 / −0.615 dz −.74).
- Noise floor: independent base draws differ by up to 0.104 on Q1+Q2 (GRPO, primary) —
  the size of the primary-judge PTO mean effect, but ~6× smaller than the held-out one.

**R2 — Cost.** `k_step_multiplier` ratio_median iters 1–5: 2.41, 2.12, **1.97, 1.96, 1.91**
(quote 3–5). Per iteration: GRPO 5.416/2.791 = 1.94×; PTO 1.968/0.812 = 2.42× (build
16.797/5.669 = 2.96×). Totals: GRPO_LA0 27.906 h (10 it), GRPO_LA5 27.078 h (5 it —
27.078/27.906 = 0.97, budget-matched), PTO_LA0 8.119, PTO_LA5 19.681 (= 2.42×).
`budget_sweep` (quote the curve): PTO K5−K0 significantly negative at every budget < 18 h on
both judges (dz −0.30 … −0.81), primary levels at 19.68 h (+0.047, dz .10, p .087), held-out
stays behind (−0.186, dz −.32, p .005). GRPO: −0.569 (dz −.74) at 13.3 h both judges;
primary level at ≥ 23 h (+0.038 n.s.), held-out **ahead** (+0.147/+0.161, dz .33/.31) — but
only because GRPO_LA0's held-out best is **iteration 3** (2.637, 8.2 h): the K=0 arm never
improved under the held-out judge past 8 GPU-h. Say that.

**R3 — Retention (held-out judge).** GRPO_LA5 Q1 retention I3–I5 = 0.98/0.98/1.08 vs GRPO_LA0
I9/I10 = 0.03/0.28. PTO_LA5 vs PTO_LA0 at I10: Q1 0.72 vs 0.80, **Q2 0.56 vs 0.85**, WAI 0.98
vs 1.28, CSQ 1.15 vs 1.37, MITI 0.27 vs 0.45 (smaller on 5/8). Under the primary PTO_LA5 ≥ PTO_LA0
on 7/8 instruments; under the held-out on 3/8. The K=5 endpoint method claim
"PTO_LA5_I10 > GRPO_LA5_I5" **reverses on Q1** under haiku (primary +0.238 → judge −0.206,
CI [−0.390, −0.025]), ties on Q2/MITI. ⚠ cross-K retention uses different base references
(LA0_Base vs LA5_Base) — differences < ~0.1 unresolved.

**R4 — Session shape (deterministic text metrics).** PTO iter 10: K=5 sessions **+8.31
utterances longer** (dz .55), turns +125 chars (dz .56); GRPO iter 5: K=5 sessions **−8.10
shorter** (dz .53), turns +207 chars (dz .82). "K=5 gives shorter, focused conversations" does
not replicate for PTO. Selection level: PTO_LA5 `w_len` = +49.5, +45.1, +58.9, +29.1, +32.4,
+28.3, +53.8, +20.0, +26.7, +7.0 (SE 7–14) — the only arm positive at every iteration; PTO_LA0
never > 1.7 SE from 0. Loops cleared *slower* under K=5 early (iter 3: PTO 0.219 vs 0.104).

**R5 — The behavioural K effect (cite the substitution draft; do not re-claim).** Over-praise/turn
K0−K5 at PTO iters 9–10 dz 1.16–1.65 both judges; K=0 endpoint over-praise share 61–84 %
(primary; 56–78 % held-out) vs 9–19 % under K=5. Total MICI is **unit-specific, not only
judge-specific**: per-turn MICI_Rate at PTO iter 10 is lower under K=5 on *both* judges
(dz .71/.66); the per-session BehaviorTotal is the held-out tie (dz .10) because K=5 sessions
are 8 utterances longer. Advice-without-permission excess under K=5 is Holm-sig from **iter 4**
(both judges) — three iterations *before* the over-praise gap opens (iter 7), which the draft's
"pressure has to go somewhere" ordering does not support (see §4).

**R6 — Training signal (primary oracle only, no held-out judge on candidates).** Best–worst
margin K5/K0 = 1.44/1.68/1.51 (GRPO it1/3/5), 1.55/1.61/1.19 (PTO it1/5/10); PTO τ-yield 0.93 vs
0.82 for iters 1–5, 1.27× more DPO steps (802 vs 630). ⚠ **Scale, not information** — the
08-10 finding (margin and SD rise by the same 1.55×; margin/SD at the noise expectation for 8
draws) lives only in a memory + notebook; `training_signal_yield.md` has no SD column, so
"decisive"/"sharper" cannot be read from any tracked table. Cross-method update-direction
cosine 0.74 (K=5) vs 0.32 (K=0), rule-swap 0.91–0.99: K acts through the data.

**R7 — Held-out-instrument tilts (both judges, both methods, but unpaired arm means).**
PCT own-base gain larger under K=5 in 8/8 method × judge × endpoint cells — the *paired* table
says dz 0.01–0.11 primary (none Holm-sig), 0.02–0.28 held-out (sig at iters 4, 10): a consistent
tilt, not a win. WAI-SR Bond − mean(Goal,Task) positive for every K=0 arm, negative for every
K=5 arm (8/8). Cooperative personas at ceiling from base; obesity/smoking a null axis; K effect
on Resistant ≈ 0 at matched iteration (PTO it10 3.65 vs 3.65).

---

## 3. The four framings, one line each, and why F1(+F2) survives

| | Title family | Distinct from "The Hack Moves"? | Kill condition |
|---|---|---|---|
| **F1 direct / replication** | *Same lever, different optimizer* | **Yes** — spine is R1–R4 + ICLR claim-by-claim | a second seed flipping the PTO K sign, or GRPO_LA5 6–10 drifting into praise |
| **F2 compute-normalized** | *Look-ahead is not free* | **Yes** — but reads as STATUS "compute axis" re-indexed; novel residue = PTO sweep rows + early-stopping frontier | best-checkpoint selection on the scoring judge collapses the top-of-sweep verdicts |
| F3 helps-according-to-whom | *Judged by the oracle, judged by a stranger* | **No** — S4–S6 + a measurement taxonomy | same as the draft's |
| F4 training-signal mechanism | *What does the look-ahead reward reward?* | Partly — novel = length push, GRPO anti-MI-concept alignment, tails; lead claim contradicts the recorded margin/SD nil | same-candidate K=0-vs-K=5 re-scoring showing rankings unchanged |

**Proposed outline (F1 + F2 §):** 1 Intro (the ICLR claim; four things it could not test)
· 2 Setup (state GRPO_LA5 = 5 iterations up front) · 3 Does K help the reward? (R1, four-arm
figure, both judges) · 4 What does K cost? (R2, both sweeps) · 5 What does K change? (R4 +
one paragraph citing the substitution paper for R5) · 6 Why (R6 at the level of margins /
alignment / length — not the over-praise chain, which is the sibling's) · 7 ICLR revisited
(Q2, stability, shorter) · 8 Limitations (single seed; censoring; K ∈ {0,5}; two LLM graders;
no K=5 repeatability; API cost not in the axis; best-of-N selection bias in the sweep).

Venue: CLPsych / "Insights from Negative Results" / NLP4Health short. Cite ICLR as the SSI-FM
workshop poster.

---

## 4. Prose-about-tables errors the cold read caught (fix at the source, before any reuse)

| Where | Says | Table says |
|---|---|---|
| substitution draft abstract/S5 | "statistically identical for **eight iterations**" | identical *at* iter 8 (dz .004); K=5 is Holm-sig **higher** at iter 4 (both judges: dz −.36/−.44) and iter 6 (held-out) — a sign-changing curve, not a plateau |
| substitution draft abstract/S4 | "the reward … indifferent **throughout**" | null at the endpoint; **favours K=0** at iters 5–8, both graders at iter 6 (dz .42/.51). This is the ICLR non-replication — not incidental |
| substitution draft S6 | "zero for five iterations … 5–8 … never exceeds 0.086" | six null iterations, lift at policy 6/7/8 (0.062/0.034/0.083), K=0 weight goes **negative** at policy 9 (−0.039); 0.086 is GRPO_LA0's *affirmation* weight. NUMBERS.md and deck slide 14 are right; S6 is off by one |
| substitution draft S6 mechanism | advice rises *because* praise is blocked | advice excess sig. from iter 4, praise gap from iter 7 — order reversed; the untested alternative is the K=5 length/directiveness push (`w_len`) |
| LIMITATIONS §5b | "at PTO's endpoint K=5 takes ~2.4 MORE therapist turns" | 14.39 vs 10.23 = **+4.16** (dz .55); "2.4" is the over-praise per-session Δ (2.417) transposed onto the turn row |
| STATUS / deck | "only the held-out grader can see" the K × method flip | iteration-5-specific: the primary sees GRPO_LA5 > PTO_LA5 at **iter 4** (Q1+Q2 dz −.35 p_holm .024; MITI dz −.41) |
| STATUS budget table | held-out "GRPO K=5 ahead at top budget" | true, and driven by GRPO_LA0's held-out best being iter 3 — say the driver |
| L0/SUMMARY §6 | still says "exploration" | STATUS retired it |
| memory 08-10 | "look-ahead creates no extra branch points" | true through iter 5 (3,624 vs 3,755); 7,548 vs 6,240 over the run — LA5's longer trunks make more branch points late |
| STATUS §5 | verbosity is a training-depth channel, not a K one | true for GRPO at matched compute; for PTO `w_len` is a **selection-level K effect** (both arms at their 10-iteration endpoint) |

Artifact hazards (render bugs, not data): `L5/tables/8_measurement/second_judge_contrasts.md`
(primary_n = 0, spurious same_sign = False ×16); `trajectory_MICI` "peak → regresses" flag is
direction-blind for lower-is-better; `L5/figures/2_questionnaires/CAPTIONS.md` describes K=0
phenomena in a K=5-only view; `faithfulness_proxy_vs_eval` x-label says "K=0 branch" on the LA5
view (data is the K-extended score); `compute_by_iteration` gen_h = 0.000 for PTO_LA5 iters 1–5
(batch-flushed conv mtimes ⇒ span 1 s; the time lands in iter 6's 0.967 h — totals intact,
cum_gpu_h understated by 0.1–0.5 h; don't quote to two decimals).

---

## 5. New EDA — what the paper needs, ranked (FREE unless flagged; hosting module named)

| # | Artifact | Why | Data / module | Effort |
|---|---|---|---|---|
| 1 | **Four-arm K-contrast headline**: persona-paired LA0−LA5 Δ/dz/p_holm at every matched iteration (0–10; 0–5 GRPO), 8 rubrics + MICI/MITI channels + session shape, **both judges in one frame**, iteration-0 row as the noise-floor line; + the four-arm trajectory figure with a paired-delta strip | Fig 1 / Table 1 does not exist; every reader hand-joined L0×L5 | score lake; `stats.paired_k_comparison` + `config.cross_k_scores` exist; extend `plotting/lookahead.k_channel_trajectory`; 7_Stats §4 under `RQ_I_VIEW` | 3–5 h |
| 2 | **Cross-K multi-judge**: `all_pairs_contrasts` / sign-preservation ladder / `gain_retention` for LA0_In vs LA5_In per method, with a shared reference | the K contrast has never been judge-tested; retention now uses two bases | `reliability.*` accept arbitrary model lists → feed `config.cross_k_arms`; verify seed 42 shared so `attach_persona` pairs; 8_Measurement §2 | 3–4 h |
| 3 | **K at matched compute on the channels**: `iso_compute_contrast` + `budget_sweep` with metric ∈ {over-praise, advise, total, B6_AF, conv_len, turn_len}, both judges; 4-panel sweep figure (PTO K, GRPO K, method@K0, method@K5) | behavioural claim is iteration-indexed while the cost claim is budget-indexed | `compute.*` already take a metric list + `behavior.channel_scores_long`; 7_Stats §4e | 3–4 h (+2 fig) |
| 4 | **Look-ahead tail audit** (LA5 arms): realized_turns, ended_early rate, tail length, who ended it, loops in tails, praise/advice/question cues tail vs candidate; within-group: does ended_early/tail length predict score and chosen status | **27.4 % of GRPO_LA5 iter-5 tails ended early** (657/2400 in the first 300 groups; realized_turns {5:1743, 3:233, 1:374, 0:45}) — "the patient ended the session" is an untested reward channel; also the API-cost story | `iteration_N/eda/generations.jsonl` (LA5 rows carry full `lookahead.tail`); `training.load_generations` currently drops the tail → add `training.tail_audit`; 5_Training | 4–6 h |
| 5 | **Reward-faithfulness as a table** with bootstrap CIs and stated unit; the matched-policy cut (train_iter 1, both arms on base — the 08-10 nil lives only in a memory); by cooperation level; length-stratified; fix the LA5 x-label | the K-lifts-faithfulness claim is a visual read | `stats.rank_agreement_by_nturns` (+CI); 5_Training | 3 h |
| 6 | **Within-group dispersion by K**: SD, margin, margin/SD vs the n=8 order-statistic expectation, τ-sensitivity of yield, floored-group share | makes the "scale not information" finding a tracked table; guards F4's lead claim | `training.advantage_signal_by_iter`, `pref.pair_yield_by_iter`; 6_Preference | 3–4 h |
| 7 | **Length-controlled selection probe** (`w_overpraise`/`w_affirm`/`w_question` with completion length partialled out) | is K=0 praise selection / K=5 advice a length confound? | `pref.weighted_lexical_contrast`; 6_Preference | 4–6 h |
| 8 | **Denominator table**: per turn / per session / per 1k chars / share, at matched iteration and compute, + per-conversation cross-judge r/ICC for over-praise, advise, B6_AF | which unit flips the K answer | `behavior.channels_per_conv`, `reliability.agreement`; 7_Stats/8_Meas | 3 h |
| 9 | **`_crossgen` analysis** — Exp1 (ICLR) Base + LA0_I1–7 + LA5_I1–7, already re-scored under the Exp3 oracle on 2026-08-12 (`eval_scores/_crossgen/…/metric={Q1,Q2}`), analysed **nowhere**: does gpt-4o-mini still see K=5 > K=0 on the ICLR conversations? Separates "grader changed" from "model/task changed" | the replication frame's missing link; also **Exp1 `LookAhead_3/` is on disk and unscored** — the only K ∈ {0,3,5} dose data in the repo (~768 calls, ~$1–2) | small loader (analysis layer deliberately does not glob `_crossgen`) + `stats.paired_compare`; standalone notebook | 3–4 h (+ LA3: PAID, tiny) |
| 10 | **ICLR "stability" check**: per-arm/iteration SD of Q1/Q2/Q1+Q2, Brown-Forsythe K0 vs K5, ceiling share by cooperation | the ICLR secondary claim has no Exp3 artifact; cooperative ceiling may explain low SD | `stats.py`; 7_Stats §4 | 2 h |
| 11 | **API-call cost axis** per arm × iteration (oracle + patient calls, token estimates) beside GPU-h | K=5 multiplies API calls; GPU-hours miss it | `compute.api_calls_by_iteration` (new); 7_Stats §4e | 3–4 h |
| 12 | **Cross-judge-selected budget sweep** (select on primary, score on held-out, and vice versa) + (Q1Q2, over-praise) Pareto frontier per budget with an over-praise-monitored stopping rule | do the top-of-sweep verdicts survive honest selection; does an early-stopped K=0 dominate K=5? | `compute.budget_sweep(select_on, score_on)`, `compute.pareto_by_budget` (new) | 4–5 h |
| 13 | Subgroup K contrast (cooperation) paired, both judges; session_end_reasons by iteration × K × subgroup | which patients K helps; when sessions end | `stats.paired_k_comparison` + filter; 4_Het / 3_Val | 3 h |

Ranks 1–5 (≈ 18 h) close the load-bearing gaps of every framing.

**PAID but small** (cost constraint is binding — price with `judge_plan` first):
- **Second independent 96-conv draw from the contested adapters** — STATUS's own recommended
  spend (~$11 oracle; local GPU viable via `generate_eval_convs.py`, ~50 min/adapter at
  `--batch-size 6`). Every framing names single-draw fragility as its biggest risk and *none*
  listed this. Needs a `draw` axis in `data.discover_arms`. **Highest value per dollar.**
- Same-candidate K=0 vs K=5 re-scoring (+ depth decomposition prefix+cand / +first reply /
  +full tail): 200 groups × 8 × {Q1,Q2} = 3,200 calls per depth, prompt-cached — F4's kill test
  and the mechanism the ICLR paper deferred. Write to a `_candidates/` partition, not the lake.
- Held-out judge on the training signal (haiku on ~100 groups × 8 × 4 arms × 2 rubrics = 6,400
  calls via Message Batches).
- Oracle repeatability on two K=5 anchors (PTO_LA5_I10, GRPO_LA5_I5) × {Q1,Q2,MICI} × 3 reps
  ≈ 1,728 calls/judge — the ~0.10 noise band is currently quoted for arms it was never measured on.
- Exp1 `LookAhead_3` into `_crossgen` (~768 calls).

**Infeasible without new runs:** K ∈ {1,2,3,10} on Exp3; a second seed per arm; GRPO_LA5 iters
6–10 (STATUS: not recommended, ~$118 + 23–34 A100-h, and it un-matches the GRPO budget pair;
`iteration_6/` holds one optimizer step from the 08-18 stop); MCL sweep; human MITI/MICI coding;
a third grader (README documents a $0 open-weights path — a scoring project, not EDA).

---

## 6. Write-up rules already on record that bind this paper (from STATUS / papers README / NUMBERS)

Both axes (iteration AND GPU-h) and name the axis · quote `budget_sweep` as a curve, never a
row · the K claim is about the *lever*, never convergence ("never *significantly* leads") ·
MI-consistency at the channel level, counts before rates, name axis + grader · never average the
two judges · every number via NUMBERS.md with arithmetic · figures copied not symlinked, grader
in the filename · drafts share no claims (check both ledgers before touching over-praise numbers)
· prevention-vs-delay is resolved (prevention within 10 iterations; beyond 10 open) · cite ICLR
as the SSI-FM workshop poster.
