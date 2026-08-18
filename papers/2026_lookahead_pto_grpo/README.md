# Same Lever, Different Optimizer

*Does K-Turn Look-Ahead Help a Small Motivational-Interviewing Therapist?*

**Target:** ACL-family venue (ACL two-column style) — long-paper length. **Current length: §1–§9 fill
10 pages in `[review]` mode; Limitations + Ethics + References on p.11; appendix pp.12–23.** For an
8-page venue the pre-planned cuts (in order, ~2 pages) are: (1) demote §8's re-score figure to the
appendix and keep §8 as one paragraph; (2) fold Table 3 (substitution) into two sentences and point
at the appendix composition grids; (3) drop the schematics figure (cite the ICLR poster's Fig. 2);
(4) shorten §7 to the rescaling + tail results and move faithfulness/data-not-loss to Appendix B.
Do not shrink the headline figure or Table 1 — legibility of the four-arm K contrast is the point.
**How it was built:** see [`BRAINSTORM_2026-08-18.md`](BRAINSTORM_2026-08-18.md) (the cold table read,
the four framings and the diff against the retired drafts that fixed this paper's scope).
**Domain:** Exp3, all four arms — PTO and GRPO at K ∈ {0, 5}, matched MCL=12, M=G=8, 96 personas,
8 instruments, two graders (gpt-4o-mini = training oracle; Claude Haiku 4.5 = held out),
persona-paired, on both the iteration axis and a reconstructed GPU-hour axis. GRPO K=5 ran
5 iterations (right-censored); the other three ran 10. Plus a re-score of the ICLR SSI-FM
poster's own 1,440 Exp1 conversations with the Exp3 oracle.
**Predecessors:** [`../archive/2026_lookahead_hack_substitution/`](../archive/2026_lookahead_hack_substitution/)
(PTO K contrast only, iteration 10) and
[`../archive/2026_clpsych_mi_reward_hacking/`](../archive/2026_clpsych_mi_reward_hacking/) (K=0 only).
This draft supersedes both on the K question; their ledgers are still worth reading for traps.

## The argument in one line

"Does look-ahead help" has no optimizer-, grader- or budget-independent answer: it is a ~2×-cost
lever that changes *what the policy learns to do* (and which reward hack it commits) more than
*how well the oracle scores it*, and the ICLR-era gain was real for its regime and does not transfer.

## The moves (one per results section)

1. **§4 Reward** — Look-ahead does not raise PTO's training reward: K=0 ≥ K=5 at 8/10 iterations,
   Holm-significant in K=0's favour at iteration 6 on the oracle (+0.257, dz .42) and at 5/6/8 on
   the held-out judge (dz .33–.51), the edge carried by Q2 (the poster's own Q2 finding, reversed);
   endpoint null on both graders. GRPO flips: K=5 > K=0 at iterations 4–5 (held-out dz −.37/−.43),
   on Q1. K × method DiD at iteration 5: dz .52 held-out, null on the oracle. Retention: GRPO K=5
   keeps its Q1 gain (1.05 own-base), PTO K=5 keeps less (Q2 .56 vs .85, MITI .27 vs .45).
2. **§5 Cost** — K=5 costs 1.9–2.0× per GRPO step (settled iters 3–5), 1.94× per GRPO iteration,
   2.42× per PTO iteration (build 3×); API calls ~2.1–2.3× (patient-simulator calls are the
   multiplier; oracle calls per candidate are matched). At matched GPU-hours K=5 never
   significantly beats K=0 on the training oracle for either method; under the held-out judge
   GRPO K=5 ends ahead (+0.161, dz .31) and PTO K=5 behind (−0.186, dz −.32). PTO beats GRPO at
   matched budget under both graders and both K.
3. **§6 Behaviour** — Over-praise closes on both optimizers (per turn: PTO iter 10 dz 1.16/1.65;
   GRPO iter 5 dz 0.30/0.69) but in PTO MI-inconsistency relocates to advice-without-permission
   (from iteration 4), so the total is unit- and grader-specific; session shape reverses by optimizer (PTO K=5 +8.3 utterances, GRPO K=5 −8.1);
   WAI-SR gain shifts Bond → Goal/Task; the held-out judge localises K=0's late Q2 gain on
   self-disclosure items 3 and 10; PCT change talk rises under K=5 in Warms-up personas.
4. **§7 Mechanism** — Look-ahead *rescales* the training signal (margin and SD ×1.4–1.8 by the
   same factor; margin/SD at the 8-draw expectation), adds no reward faithfulness at a matched
   policy, and adds a session-closing pressure (ended-early tails 21–23% less likely to be argmax);
   the two optimizers' update directions align far more under K=5 (cosine .74 vs .32) — K acts
   through the data, not the loss.
5. **§8 ICLR revisited** — Re-scoring the poster's 1,440 transcripts with gpt-4o-mini reproduces
   K=5 > K=0 at 7/7 iterations under both graders (arm-level dz −.54 vs −.61), so the Exp3 null is
   the regime, not the judge. "Shorter" reverses for PTO; "lowest SD" is a ceiling artefact.

## What this draft deliberately does NOT claim

- **Not** that look-ahead is useless, and **not** that it helps. The sign is optimizer-, grader-
  and budget-conditional; §9 says which.
- **Not** that look-ahead "reduces MI-inconsistency". Per turn under K=5 it is lower on both
  graders (PTO iter 10); per session only under the training oracle. Always unit + grader.
- **Not** that GRPO K=5's reversal at iterations 4–5 persists — the arm is censored at 5.
- **Not** which of the Exp1 → Exp3 changes (1B vs 7B, bf16 vs 4-bit, V3 patients, MCL=12,
  matched hyperparameters, iterative reseeding) carries the non-transfer. We list them and stop.
- **Not** that the crossgen levels are on the Exp3 axis. gpt-4o-mini reads Exp1 0.19–0.43 higher
  than GPT-3.5 and Exp1 Base (3.87) is not Exp3 Base (~3.0); only the within-Exp1 K contrast transfers.
- **Not** an average over graders, anywhere.

## Files

```
main.tex               preamble + \input order (macros \QQ \dz \primaryjudge \heldoutjudge \Kz \Kf \gpuh \todo \note)
sections/              00_abstract 01_intro 02_related 03_setup 04_reward 05_cost 06_behaviour
                       07_mechanism 08_iclr 09_discussion 10_limitations 11_ethics
                       A_tables B_tails C_repro
analysis/              the paper's FROZEN FIXTURE (see analysis/README.md): analysis/out/*.json = one
                       ledger per retired generator (every quotable number, with its source table),
                       _findings_digest.txt (findings, caveats, paper-use), _appendix_rows.tex. The nine
                       generators (analysis/*.py) were retired 2026-08-18 — they live in git at b09eb6f and
                       were promoted into eda_analysis.{lookahead,transfer,compute,tails,dispersion,
                       faithfulness,crossgen,replication,instruments} + the family notebooks under
                       Exp3_PTO_GRPO/eda/notebooks/{lookahead,compute}/. Nothing under analysis/ regenerates;
                       the EDA self-check's 'paper fixture anchors' reads analysis/out/.
tables/                176 .md/.csv frozen with the fixture — never edit by hand; the live successors are
                       Exp3_PTO_GRPO/eda/results/{lookahead/*,compute/cost}/tables/ (map in NUMBERS.md)
figures/               every figure the .tex references, copied from the tracked results tree by
                       sync_figures.py (Exp3_PTO_GRPO/eda/results/<family>/figures/ + the hand-drawn
                       schematics under results/schematics/) under the paper's historical filenames.
                       Body figures (6): the schematics pair, k_contrast_headline_fig_q1q2 (a figure* at
                       0.85\textwidth; source lookahead/reward/k_headline_q1q2), compute_axis_fig_trajectory_col
                       (compute/cost/compute_trajectory_col), k_contrast_headline_fig_channels
                       (lookahead/behaviour/k_channels_grid), tail_audit_fig (lookahead/mechanism/tail_audit),
                       crossgen_exp1_fig_col (lookahead/replication/crossgen_col); everything else is
                       appendix. The `_col` files are the single-column STACKED variants (same panels, same
                       data, narrow-width fonts). Other PNG/PDF files still on disk here are stale
                       generator output no longer referenced by any .tex and no longer synced.
NUMBERS.md             THE CLAIMS LEDGER: every number in the text -> the tracked results table (+ the
                       fixture table / ledger key in parentheses)
BRAINSTORM_2026-08-18.md  how the framing was chosen (cold table read → framings → diff vs the retired drafts)
sync_figures.py        copy every referenced figure from the tracked results tree into figures/ (--check = drift report)
refs.bib               33 entries; acl.sty / acl_natbib.bst vendored
```

## Regenerating

There is no paper-local generator loop any more. The cross-K artifacts are EDA-owned: re-render
the two families that carry them, re-sync the figures, then rebuild. From the repo root, with the
repo venv (the shell's `python` is not it):

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py Exp3_PTO_GRPO\eda\tools\render_results.py --top lookahead compute   # results/{lookahead/*,compute/cost}
& $py papers\2026_lookahead_pto_grpo\sync_figures.py                       # copy figures into ./figures/
& $py papers\2026_lookahead_pto_grpo\sync_figures.py --check               # must print "0 drifted, 0 missing"
```

(The schematics are hand-drawn under `Exp3_PTO_GRPO/eda/results/schematics/` and are not part of
any render.) The re-rendered tables land under `Exp3_PTO_GRPO/eda/results/<family>/tables/`; walk
`NUMBERS.md` against them before touching a `.tex`. The EDA bootstraps at `BOOT_SEED=12345` where
the frozen fixture used seed 0, so bootstrap CI bounds may differ from the fixture in the third
decimal — every mean, dz, p and count is identical. `tables/` and `analysis/out/` are the frozen
fixture and are NOT rewritten by anything.

**Build:** four passes with MiKTeX, no `latexmk` — see [`../README.md`](../README.md) § Building
(`pdflatex` → `bibtex` → `pdflatex` → `pdflatex`; `\usepackage{times}` is load-bearing).

## Traps (distilled from NUMBERS.md and the archived ledgers)

- **Sign convention.** Tables report K=0 minus K=5 (+ ⇒ K=0 higher). The budget-sweep tables
  and figure carry K=5 minus K=0 beside it. Say which arm is higher in words.
- **Never "reduces MI-inconsistency" without unit + grader.** PTO iter 10: per turn lower under
  K=5 on both graders (dz .71/.66); per session lower only under the training oracle (dz .45 vs
  .10), because the held-out judge counts the substituted advice at 1.95 acts to the oracle's 0.31.
- **Counts before rates.** The arms differ in therapist-turn count (PTO K=5 14.4 vs K=0 10.2 at
  iteration 10; GRPO K=5 11.3 vs K=0 15.3 at iteration 5), so every rate has a moving denominator,
  and it moves in opposite directions for the two optimizers.
- **"K=5 never leads" is literally false.** On the primary, PTO K=5 is nominally above K=0 at
  iterations 5 (−0.002) and 10 (−0.047), and GRPO K=5 leads at 4–5. The true statement is "K=5
  is never *significantly* above K=0 for PTO" and "at matched budget K=5 never significantly
  beats K=0 on the training oracle".
- **Quote the sweep, not a row.** Both K=5 arms trail until the last one or two budgets of their
  sweeps (PTO K=5 to 19.7 h, Holm-significantly through 14.6 h; GRPO K=5 to 23.2 h, where it
  already ties under the oracle and leads under the held-out judge); a single iso-compute row can
  carry either sign.
- **GRPO K=5 is right-censored at iteration 5.** Every GRPO K contrast, DiD, retention and
  "endpoint" is at iteration ≤ 5; comparisons against a K=0 iteration > 5 must name it (the
  endpoint table carries I5-vs-I10, I5-vs-I8 = primary best, I5-vs-I3 = held-out best).
- **Never average the two graders.** The oracle was the reward; the held-out judge is a
  train/test split, not a second rater. Level offset is 1.2–1.7 points and model-dependent.
- **Retention needs its base reference.** GRPO K=5's Q1 retention "disjointness" at iteration 5
  holds only under a shared LA5 (or PTO LA5) base; with own bases the intervals overlap, and the
  two GRPO base draws differ by 0.104 on Q1Q2 (primary). Report own-base with the shared range.
- **The crossgen levels are not on the Exp3 axis.** Exp1 Base under gpt-4o-mini is 3.87 (Exp3
  Base ≈ 3.0): different therapist, patients, length. Only the within-Exp1 K contrast transfers.
- **Noise floor.** Iteration 0 is two independent base draws: PTO pair −0.003, GRPO pair +0.104
  (dz .11, n.s.) on Q1Q2 primary. A single-iteration GRPO contrast of that size is not a claim.
- **Iteration ≠ spend.** A K=5 GRPO step costs ~1.9× a K=0 step; a PTO iteration costs a fraction
  of a GRPO one (3.44× over ten iterations at K=0). Never time a run from `iteration_metadata.json`.
- **Holm families differ by table** (across iterations for the by-iteration K tables; across
  instruments for endpoints; across unique checkpoint pairs for sweeps). Quote the family with the p.
- **The training oracle under-reports the harm channel.** MICI own-base retention is 1.06–4.63
  on every arm — the held-out judge is where MI-inconsistency should be read.
- **Cite the ICLR paper as the SSI-FM workshop poster**, not the main conference (its PDF header
  is stock template boilerplate).
