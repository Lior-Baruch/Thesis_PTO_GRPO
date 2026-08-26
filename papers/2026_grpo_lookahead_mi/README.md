# Scoring the Continuation

*$K$-Turn Look-Ahead Rewards for Group-Relative Policy Optimization in Motivational Interviewing*

**Target:** **ICLR 2027** (abstract deadline **2026-09-18**, full paper **2026-09-25**, both
23:59 UTC-12; conference April 2027). Single-column `iclr2027_conference` style, vendored here
from the official `iclr-2027-style-files.zip`. **Status: drafting** — the one live draft.
Retargeted from the earlier ACL/CLPsych two-column draft on 2026-08-26 (the ACL version is in git
history; `acl.sty`/`acl_natbib.bst` were removed with it).
**Current length (clean 4-pass build, draft notes visible): 23 pages — main text §1–§9 ends on
p. 9** (ICLR submission limit: 9; camera-ready: 10). Ethics Statement + Reproducibility Statement
(exempt from the limit) follow, then references and appendices A (supplementary results, incl. the
full endpoint table and the method schematic), B (mechanism in full), C (repro details),
D (limitations in full).
**Double-blind:** `\iclrfinalcopy` stays commented out until camera-ready — the submission renders
as "Anonymous authors". The `\author` block in `main.tex` is ignored while it is commented. The
ICLR poster is cited in third person throughout; keep it that way.
**Do not cut §8 (measurement) or the per-grader qualifier in §6** — those are the paper's honesty,
and cutting them would change what it claims. §7 (mechanism) is deliberately compressed with the
full analysis in Appendix B; the endpoint-per-instrument table is Appendix Table 1 by design (the
9-page budget), with its verdict stated in §4 prose.
**Domain:** Exp3, the **two GRPO arms only** — `GRPO_LA0` and `GRPO_LA5`, matched MCL=12, G=8,
96 personas, 8 instruments, 10 iterations each, scored by two graders (gpt-4o-mini = the training
oracle; Claude Haiku 4.5 = held out). Both arms complete; 2 arms × 11 states = 22 model states.
**How it was planned:** [`../BRAINSTORM_2026-08-25.md`](../BRAINSTORM_2026-08-25.md) (the cold
table read on the completed grid, five candidate papers, and why this one goes first).
**Predecessors:** the ICLR 2025 SSI-FM poster
([`../2025_iclr_pto_lookahead/`](../2025_iclr_pto_lookahead/)) introduced PTO and the look-ahead
lever; three retired Exp3 drafts under [`../archive/`](../archive/) argued earlier, censored-grid
versions of neighbouring questions. Their `NUMBERS.md` traps still bind on any shared number.

## The argument in one line

Scoring a candidate therapist turn by the $K$-turn continuation it leads to, rather than by the
turn itself, more than doubles what group-relative RL extracts from the same oracle — and it is
the difference between a policy that learns motivational interviewing and one that learns to
flatter the judge.

## The moves (one per results section)

1. **§4 Reward — the lever works, and it is not small.** At the matched iteration-10 endpoint,
   $K{=}5$ beats $K{=}0$ on Q1+Q2 by **+0.765 ($d_z$ 0.905)** under the training oracle and
   **+0.616 ($d_z$ 1.030)** under the held-out judge, persona-paired over 96 personas, and is
   significantly ahead on **all 8 instruments under both graders**. Against its own base the gain
   is **+1.554 vs +0.686** (primary) and **+1.038 vs +0.396** (held out) — a ratio of
   1.554 / 0.686 = 2.27× and 1.038 / 0.396 = 2.62×. The advantage is Holm-significant on Q1+Q2 at
   6 of 10 iterations and largest at the endpoint.
2. **§5 Cost — honest, and it still wins.** $K{=}5$ costs a median 1.92× per optimizer step over
   the settled iterations 3–10 (range 1.83–2.18; iterations 1–2 ran at a smaller look-ahead
   sub-batch and are excluded) and 51.205 / 27.906 = 1.835× over the whole run. On a reconstructed
   GPU-hour axis the lever *loses* below ~18 GPU-h and only draws level at 23.2 (+0.038, n.s.), so
   a per-iteration claim would flatter it; its first Holm-significant win is at 35.29 GPU-h
   (+0.188, $d_z$ 0.310) and at the common 51.2 GPU-h budget it wins under **all four** grader
   select/evaluate combinations, including honest cross-grader selection.
3. **§6 Behaviour — what the two rewards actually teach.** Turn-level reward produces the textbook
   hack: `GRPO_LA0`'s MI-inconsistency rises 0.211 → 0.838 (primary), carried by **over-praise**,
   which a *judge-free deterministic lexical marker* finds in **67.1% of therapist turns** at the
   endpoint against **6.4%** for $K{=}5$ — a ratio of 0.671 / 0.064 = 10.5×. ⚠ That column is the
   **share of turns containing ≥1 marker**, not a per-turn count; name the axis that way.
   Trajectory-level reward does not do this. ⚠ **State it per grader**: the
   primary reads $K{=}5$'s MICI as flat (0.209 → 0.210, n.s.), the held-out judge reads the same
   conversations as still rising (0.326 → 0.628, $d_z$ 0.845) but at 0.301 / 0.666 = 0.45 of
   $K{=}0$'s rise. The defensible claim is **"slows and roughly halves the loop"**, never "stops"
   it. $K{=}5$ also asks more questions per turn (judge-free text metric).
4. **§7 Mechanism — consistent-with, not shown.** The $K{=}5$ training cut is a better rank proxy
   for the final-conversation score than the $K{=}0$ cut (0.909 vs 0.873 pooled agreement over
   matched iterations, primary; 0.800 vs 0.747 held out). ⚠ **Report both tests**: the CI on the
   pooled-pairs difference excludes 0 (−0.036 [−0.051, −0.021]) but the iteration-level Wilcoxon
   does *not* clear .05 (p = .084 primary, .193 held out, $n$ = 10 iterations) — the pooled
   interval treats branch pairs as independent, so it is the weaker of the two. Say
   "K=5 more faithful at 7 of 10 iterations, pooled difference small but consistent", not
   "significantly more faithful". Look-ahead also *rescales* the group's reward spread rather than
   sharpening it (pooled margin ratio 1.300 and SD ratio 1.293 move together; ratio-of-ratios
   1.006 [1.002, 1.010]) — though iteration 10 inverts (margin ratio 0.679), so quote the pooled
   row and show the by-iteration table. ⚠ At a **matched policy** look-ahead adds no faithfulness
   — that result conditions differently and `METRICS_REFERENCE.md` owns the distinction. Present
   this section as mechanism *evidence consistent with* the effect, never as its demonstration.
5. **§8 Measurement — the caveat that must travel with the headline.** The winning state is also
   the experiment's **worst per-conversation cross-grader agreement on Q1**: `GRPO_LA5`'s Q1
   agreement falls 0.941 (I5) → 0.769 (I8) → 0.487 (I9) → 0.544 (I10) against a 44-state Q1 median
   of 0.855 — I9 and I10 are the two lowest of all 44 states. ⚠ **It is NOT "only the rewarded
   rubric"** (the draft said that until 2026-08-25 and the table disproves it): at the same state
   Q2 (0.590 vs 0.784), **MITI (0.333 vs 0.658 — its 44-state minimum)** and MICI (0.287 vs 0.518)
   are all depressed, while CSQ-8, MI-SAT, WAI-SR and PCT are normal. The pattern is *rewarded +
   behaviour-coding rubrics degrade, global-impression rubrics don't* — two causes stacking, and
   the reason §6's argument rests on the judge-free marker rather than on the MITI/MICI counts.
   The Q1 collapse itself is because the
   training grader saturates one-sidedly: its Q1 variance falls to 0.275× of base, monotonically
   (Spearman −0.86, p = .001), while the held-out judge's spread over the same conversations
   **does not move** (ρ = +0.44, p = .18). ⚠ **Do not say the held-out variance "grows 1.41×"** —
   that ratio anchors on iteration 0, which is that series' minimum (1.06× against iteration 1).
   Flat is all the argument needs: homogenised conversations would have shrunk *both*. The
   arm-level conclusion survives — the held-out judge independently ranks $K{=}5$ ahead at
   $d_z$ 1.03 — but the per-conversation ruler does not.

## What this draft deliberately does NOT claim

- **Not** that look-ahead helps optimizers in general. It is a single-optimizer result; the same
  lever applied to PTO in the same experiment is null-to-negative on outcomes. One paragraph in
  §9 says so and points at the companion paper rather than importing its 2×2.
- **Not** that $K{=}5$ eliminates reward hacking — see the per-grader split in move 3.
- **Not** anything about $K$ as a continuous knob. $K \in \{0, 5\}$ by design; there is no
  dose–response evidence and the Limitations section says so.
- **Not** a clinical claim. Every instrument is LLM-administered; no human MI coder has rated any
  conversation in this experiment.
- **Not** a generalisation claim about patients. All 96 personas are used for both training
  rollouts and evaluation, so every number is in-sample with respect to the patient distribution.

## Conventions this draft inherits

- **Numbers come from tracked artifacts, never from memory or from prose.** Every quantitative
  claim is in [`NUMBERS.md`](NUMBERS.md) with the exact
  `Exp3_PTO_GRPO/eda/results/<family>/tables/…` path it came from. If the EDA is re-rendered and a
  number moves, that ledger is how you find every sentence that has to change.
- **Sign conventions are stated at every table.** The EDA's K tables report $K{=}0$ minus $K{=}5$
  (+ ⇒ $K{=}0$ higher); this paper argues for $K{=}5$, so most body sentences flip the sign —
  which is exactly the kind of transposition that produces a wrong number. Say which direction
  every quoted delta is in.
- **Name the grader on every number, and never compare levels across graders** (the offset is
  1.2–1.7 points and model-dependent). Never average the two graders' raw scores: the primary was
  the training reward and the held-out judge was not, so that is train-vs-test, not two raters.
- **Name the axis on every behaviour claim** (per therapist turn / per session / share of coded
  acts). The two arms differ in both turn count and turn length, so the three denominators can
  disagree in direction; prefer the share, and prefer the judge-free lexical marker where it
  exists.
- **Figures are copied, never symlinked**, by [`sync_figures.py`](sync_figures.py), so the draft
  compiles standalone and a submitted PDF is frozen against later EDA re-renders. Re-run it after
  every `eda/tools/render_results.py` pass.
- **Cite the ICLR paper as the SSI-FM *workshop* poster, not the main conference** — its PDF
  header is stock template boilerplate. Canonical BibTeX:
  [`../2025_iclr_pto_lookahead/README.md`](../2025_iclr_pto_lookahead/README.md).

## Building (MiKTeX on Windows)

`iclr2027_conference.{sty,bst}`, `natbib.sty` and `fancyhdr.sty` are vendored here (from the
official ICLR 2027 style zip), so the draft builds with no network round-trip. Four passes, no
Perl, **no `latexmk`**:

```bash
export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```

Three `pdflatex` passes, not two: one to write `.aux`, one to absorb the `.bbl` and place floats,
one to settle the resulting page and reference numbers.
