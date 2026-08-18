# The Hack Moves

*Trajectory-Level Reward Redirects Rather Than Reduces Reward Hacking in a Motivational
Interviewing Therapist*

> ⚠ **RETIRED 2026-08-18 — archived, not live.** Superseded by [`../../2026_lookahead_pto_grpo/`](../../2026_lookahead_pto_grpo/) (*Same Lever, Different Optimizer*), which absorbs this paper's substitution result as its behaviour section and extends it to both optimizers, both graders and both cost axes. Do not edit this draft; its `NUMBERS.md` **traps still bind** on the shared numbers. Artifact paths below point at the pre-reorg `results/L0|L5/` tree (recoverable at commit `b09eb6f`; map them via `Exp3_PTO_GRPO/eda/README.md` § Migration).

**Target:** CLPsych / NLP-for-psychology workshop (ACL style, 8-page body).
**Domain:** Exp3, **`L5` view** — PTO at matched $K \in \{0,5\}$, **iterations 0–10** (both arms complete).
**Sibling draft:** [`../2026_clpsych_mi_reward_hacking/`](../2026_clpsych_mi_reward_hacking/) is
`L0` only and excludes K entirely. The two share **no claims** — check both ledgers before
touching a shared artifact.

## The argument in one line

Look-ahead closes the flattery channel exactly as intended — verifiably, at the reward, in the
policy's output, under a held-out judge, and for the whole run — and what it buys on total
MI-inconsistency is a number the two graders cannot agree on, because unsolicited advice rises by
close to what flattery gives up.

## The three moves

1. **§4 The channel closes, and stays closed.** Over-praise is $2.42$ acts/session lower under
   $K{=}5$ at iteration 10 ($d_z{=}0.89$; $1.00$ held-out), null for six iterations then divergent
   and widening. Not a denominator artifact (raw per-session counts) and not data starvation
   ($K{=}5$ trained on *more* pairs: 6,416 vs 4,935). **Prevention, not delay.**
2. **§5 The aggregate does not follow.** Identical for eight iterations ($3.448$ vs $3.458$,
   $d_z{=}0.004$). At the endpoint the primary oracle scores the totals apart ($d_z{=}0.45$,
   $p{=}.0003$) and the held-out judge scores them null ($d_z{=}0.10$, $p{=}.17$) — while agreeing
   on the sign and significance of **every component**. **This is the paper.**
3. **§6 Where it acts.** The intervention is visible in what the *reward* selects for, not only in
   the outcome — so the closure is mechanistic, and so is the reason the pressure relocates.

## What this draft deliberately does NOT claim

- **Not** that look-ahead *reduces* MI-inconsistency. That holds only under the grader that was the
  training reward ($d_z{=}0.45$); the held-out judge sees no aggregate change ($d_z{=}0.10$,
  $n.s.$). §5 leads with the non-replication rather than burying it.
- **Not** that look-ahead is useless, or that reward hacking is conserved as a law. One task, one
  reward, one intervention.
- **Not** that $K{=}5$'s remaining violations are more severe. The training oracle says so at
  iteration 8 ($d_z{=}-0.59$); the held-out judge does not ($d_z{=}-0.11$, $n.s.$), and neither
  effect persists to the endpoint.
- **Not** a $K \times$ optimiser result --- by **scope**, not by data. The GRPO $K{=}5$ arm ran
  five iterations (scored 0--5 on both graders) before being stopped, so a cross-optimiser $K$
  contrast does exist over iterations 0--5; this draft states its claims at iteration~10, outside
  that window.
- **Not** that $K{=}5$ would never drift given more iterations — its over-praise is still creeping
  ($0.47 \to 0.63$ over iterations 8–10) while $K{=}0$'s roughly doubles over the same stretch.
  Prevention-vs-delay *within ten iterations* is settled; beyond that is not.

## Files

```
main.tex               preamble + \input order. \drafttrue enables \todo{} / \note{}.
sections/              one file per section, in reading order
  00_abstract  01_intro  02_related  03_setup  04_channel  05_aggregate
  06_mechanism  07_discussion  08_limitations  09_ethics
  A_channels  B_repro
figures/               PNGs copied from the EDA by sync_figures.py — do not edit by hand
NUMBERS.md             THE CLAIMS LEDGER: every number -> the artifact it came from
sync_figures.py        re-copy figures after an eda/tools/render_views.py pass
refs.bib               shared with the sibling draft
acl.sty acl_natbib.bst vendored from acl-org/acl-style-files
```

## ⚠ This paper reads the `L5` view

`L5` is `eda_analysis.RQ_I_VIEW`. `7_Stats` §4c/§4d and `6_Preference` §5d are **gated** to it, so
the K contrast has exactly one owner; under `L0` those sections print a skip message and produce
nothing. Rendering the sources this draft needs:

```powershell
cd ..\..\Exp3_PTO_GRPO\eda
& ..\..\.venv\Scripts\python.exe tools\render_views.py L5 --nb 7
& ..\..\.venv\Scripts\python.exe tools\render_views.py L5 --nb 7 --judge anthropic_claude-haiku-4-5
& ..\..\.venv\Scripts\python.exe tools\render_views.py L5 --nb 6
```

`--nb` takes one notebook per invocation — passing it twice keeps only the last. Notebook 6 is
training-side and **refuses** a `--judge`; its mechanism panel's top two rows are judge-invariant
by construction (they record the training oracle's own selection during the run).

## Length — measured, not estimated

Currently **10 pages total**, body ending at ~68% of p.7, **0 overfull boxes, 0 undefined refs**
(measured 2026-08-18, after the related-work section was written, the full channel table landed
in Appendix A, and §4 gained the gain-retention paragraph). ~1.3 columns of body slack remain
before the 8-page limit.

```powershell
& ..\..\.venv\Scripts\python.exe -c "import fitz; d=fitz.open('main.pdf'); [print(f'p.{i+1} {j/len(t):.0%}') for i,p in enumerate(d) for t in [' '.join(p.get_text().split())] for j in [t.find('Limitations')] if j>=0]"
& ..\..\.venv\Scripts\python.exe -c "print([l for l in open('main.log',errors='replace') if 'Overfull' in l])"
```

⚠ The `Limitations` search also matches §7's cross-reference — the **last** hit is the heading.

## Building

See [`../README.md`](../README.md) § Building — four passes, MiKTeX, no `latexmk`. `\usepackage{times}`
is required, not cosmetic: without it `microtype` fails fatally on MiKTeX's bitmap Computer Modern.

## Keeping it honest

After any `render_views.py` rerun:

```powershell
& ..\..\.venv\Scripts\python.exe sync_figures.py --check   # did any figure move?
& ..\..\.venv\Scripts\python.exe sync_figures.py           # re-copy
```

then walk [`NUMBERS.md`](NUMBERS.md), whose top section lists the six traps specific to this paper.
The first two are the ones that would actually change the conclusion:

- **Counts before rates.** The arms differ in therapist-turn count, so every `*_rate` has a moving
  denominator. All primary contrasts are per-session counts.
- **Never quote the channel effect without the aggregate beside it.** That omission is the exact
  error this paper was written to correct — an earlier pass of this analysis concluded
  "look-ahead reduces the reward hack ~4×" and it was wrong about the therapist's behaviour.
