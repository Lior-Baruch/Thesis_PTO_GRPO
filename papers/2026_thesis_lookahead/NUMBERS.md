# NUMBERS.md — the claims ledger

Every quantitative claim in the chapter, with the artifact it came from. After any rerun of
`analysis/crossgen.py`, walk this file for anything that moved.

Regenerate everything:

```powershell
& ..\..\.venv\Scripts\python.exe analysis\crossgen.py
```

**Sign convention throughout: `Δ = K0 − K5`. Negative Δ means look-ahead HELPED.**

---

## Provenance: two classes of number

| Class | Where it comes from | How to re-verify |
|---|---|---|
| **A — outcome contrasts** | `analysis/crossgen.py`, from the three generations' score files | re-run the script |
| **B — generation-3 training-signal results** | Exp3's own EDA (`history/CHANGELOG_EDA.md` 2026-08-10 entry; `eda/results/L*/tables/6_preference/`) | cited, not recomputed here — §8 says so explicitly |

Class B is confined to §8 (Mechanism). Everything else is class A.

**Scope: PTO only.** All three generations are PTO. Exp3's GRPO arms are excluded (`crossgen.py::EXP3_METHODS`) and belong to the companion paper. Removing them changed no number in this chapter — the GRPO K=5 arm had one matched iteration and was already below the moderator's 3-iteration threshold.

---

## §Summary / §7.4 — the headline

| Claim | Value | Artifact |
|---|---|---|
| Gen 1, K=5 ahead | 7/7 matched iterations | `tables/t1_generations.md` |
| Gen 1, mean Δ (re-graded) | −0.132 | `tables/t1_generations.md` |
| Gen 2, contrasts significant | 0 of 105 | `tables/t9_exp2_all_rubrics.md` |
| Gen 2, K=5 ahead | 61/105 = 58.1% | `tables/t9_exp2_all_rubrics.md` |
| Gen 2, mean Δ (Q1+Q2) | −0.019 over 15 contrasts | `tables/t1_generations.md` |
| Gen 3, K=5 ahead (primary) | 1/8 — that cell is the −0.002 dead heat at iter 5 | `tables/t4_exp3_k.md` |
| Gen 3, K=5 ahead (held-out) | **0/8** | `tables/t10_exp3_k_heldout.md` |
| Gen 3, mean Δ | +0.110 primary / +0.154 held-out | `t4_exp3_k.md`, `t10_exp3_k_heldout.md` |

⚠ **The 0/8 figure belongs to the held-out judge only.** The primary oracle is 7/8 by sign.
Do not write "0 of 8 under either grader".

## §5 — Generation 1

| Claim | Value | Artifact |
|---|---|---|
| Published best-vs-best, Final | +0.206, Welch p=0.023 | `main.pdf` of the ICLR paper; reproduced in the scratch re-analysis |
| Published best-vs-best, Q2 | +0.191, p=0.011 | ditto |
| Published best-vs-best, Q1 | +0.221, p=0.063 (n.s.) | ditto |
| Paired, GPT-3.5, mean Δ | −0.206 | `tables/t2_exp1_two_graders.md` |
| Paired, GPT-3.5, cells favouring K=5 | 21/21, 7 Holm-sig | `tables/t2_exp1_two_graders.md` |
| Paired, re-graded, mean Δ | −0.132 | `tables/t2_exp1_two_graders.md` |
| Paired, re-graded, cells favouring K=5 | 20/21, 6 Holm-sig | `tables/t2_exp1_two_graders.md` |
| Per-iteration Δ, both graders (Table 1.2) | see table | `tables/t2_exp1_two_graders.md` |
| Cross-grader r | mean 0.774, range [0.542, 0.867] | `tables/t11_exp1_grader_agreement.md` |
| Grader level offset | +0.269 | `tables/t11_exp1_grader_agreement.md` |
| Gen 1 base (re-graded) | 3.865 | `tables/t6_exp1_levels.md` |
| Gen 1 K=0 endpoint | 3.887 (gain +0.023) | `tables/t6_exp1_levels.md` |
| Gen 1 K=5 endpoint | 4.171 | `tables/t6_exp1_levels.md` |
| Gen 1 slopes | K=0 +0.011, K=5 +0.030 per iter | `tables/t5_moderator.md` (K=0 col) |
| Re-grading cost | 2,880 calls, 0 errors, ≈\$1.5 | `eda/tools/score_crossgen.py --dry-run` |

## §6 — Generation 2

| Claim | Value | Artifact |
|---|---|---|
| Per-oracle K=5-ahead counts | Q1+Q2 4/5, WAI-SR 2/5, CSQ-8 3/5 | `tables/t3_exp2_k.md` |
| Per-oracle mean Δ | −0.060 / +0.020 / −0.018 | `tables/t3_exp2_k.md` |
| All-rubric sweep | 105 contrasts, 0 Holm-sig, 61 favour K=5 | `tables/t9_exp2_all_rubrics.md` |
| Per-metric mean Δ range | −0.033 … −0.008 | `tables/t9_exp2_all_rubrics.md` |
| Gen 2 base | 2.378 | `tables/t5_moderator.md` |
| Gen 2 myopic endpoints | 2.770 / 2.596 / 2.629 | `tables/t7_exp2_levels.md` |

## §7 — Generation 3

| Claim | Value | Artifact |
|---|---|---|
| Per-iteration Δ, primary | +0.083 … +0.077 (see Table 1.3) | `tables/t4_exp3_k.md` |
| Iter 6 (the significant cell) | Δ +0.257, dz 0.42, Holm p<0.001 | `tables/t4_exp3_k.md` |
| Iter 5 dead heat | Δ −0.002, dz −0.004 | `tables/t4_exp3_k.md` |
| Endpoint (iter 8) levels | 4.221 vs 4.144 | `tables/t4_exp3_k.md` |
| Held-out judge, per iteration | 0.060 … 0.186 | `tables/t10_exp3_k_heldout.md` |
| Held-out Holm-sig cells | iters 5, 6, 8 (p 0.005 / 0.000 / 0.001) | `tables/t10_exp3_k_heldout.md` |
| MICI sign reversal at iters 7–8 | +0.078 / +0.059 primary, +0.071 held-out | Exp3 `results/L5/tables/7_stats/*/k_paired_by_method.md` (**class B**) |

## §8 — Mechanism (class B — Exp3's own analysis)

| Claim | Value | Artifact |
|---|---|---|
| Pref pairs, LA5 vs LA0, iter 6 | 568 vs 475 (1.2×) | `Exp3.../results/{L5,L0}/tables/6_preference/gpt-4o-mini/training_signal_yield.md` ✔ re-verified |
| Pref pairs, iter 7 | 689 vs 400 (1.7×) | same ✔ re-verified |
| Spread multiplier | ~1.55× | `history/CHANGELOG_EDA.md` 2026-08-10 |
| margin/SD | ≈2.85 in every arm (pure-noise expectation for best of 8) | same |
| τ rescaling closes yield gap | 44–87% | same |
| Matched-policy faithfulness | 11/19 depth bins, weighted −0.005, Wilcoxon p=0.59 | same |
| Pooled (confounded) faithfulness | +0.052 | same |
| Matched-policy base check | Q1+Q2 3.000 vs 3.003, p=0.987 | same |

## §9 — Moderator

| Claim | Value | Artifact |
|---|---|---|
| r (myopic slope vs LA benefit) | −0.716, p=0.173 | `tables/t5_moderator.md` (attrs) |
| r (myopic gain vs LA benefit) | −0.839, p=0.076 | same |
| Spearman | −0.400 / −0.600 | same |
| n | 5 arms (Exp1; Exp2 x3 oracles; Exp3) | `tables/t5_moderator.md` |
| Contradicting arm | Exp2/WAI-SR: slope +0.009, benefit −0.020 | `tables/t5_moderator.md` |
| Gen 3 myopic gain | +1.212 | `tables/t5_moderator.md` |

## §4 / §10 — Method and limitations

| Claim | Value | Source |
|---|---|---|
| Exp2/Exp3 Q1+Q2 prompts byte-identical | asserted at run time (Q1 4,390 / Q2 8,477 chars) | `crossgen.py::verify_shared_axis` |
| Oracle ICC(2,1) | 0.86–0.99 overall; 0.96–0.99 Q1/Q2 | Exp3 `eda/docs/LIMITATIONS.md` §1 |
| Oracle mean abs. test–retest | 0.04–0.09 | same |
| 4-bit vs bf16 degeneration | ≈9.5% vs ≈0.3% of therapist turns | root `CLAUDE.md` § "Data lineage" |
| Gen 2 level depression | ≈0.6 points | same |

---

## TODOs before this is submission-clean

Items 1-2 are closed; 3-5 are open.

1. ~~Fold the cross-grader agreement numbers into `crossgen.py`.~~ **DONE** —
   `tables/t11_exp1_grader_agreement.md`.
2. ~~Make the byte-identical-prompt check an assertion.~~ **DONE** —
   `crossgen.py::verify_shared_axis` runs first and raises if either `questionnaires.py`
   drifts, so the comparison cannot be silently invalidated.
3. **`results/L5/SUMMARY.md` is stale** — it still narrates the superseded "arms tie at
   iteration 5" reading and stops at iteration 5. The chapter supersedes it; that file
   should be rewritten or explicitly marked stale so the two do not contradict.
4. **Re-render the Exp3 L5 view.** Its tracked `k_paired_by_method.md` stops at iteration 7;
   iteration 8 exists in the score lake and is used here. Run
   `python tools/render_views.py L5` so the Exp3 artifacts and this chapter agree.
5. **Two `refs.bib` entries carry TODO notes** inherited from the companion paper
   (`steenstra2024scaffolding` needs verifying or removing; `baruch2025pto` needs the
   main-conference-vs-workshop decision).
