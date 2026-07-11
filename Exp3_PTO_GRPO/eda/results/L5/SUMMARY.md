# Exp3 EDA Summary — view `L5` (look-ahead, K=5) — PRELIMINARY

*Hand-authored narrative. Preserved across reruns / `reset_results`. The auto-generated artifact
map is [`INDEX.md`](INDEX.md).*

> ⚠ **This view is thin and incomplete — read it as preliminary.** The K=5 arms were **paused for
> OpenAI-API budget** (cost ∝ candidate count × iterations, and K=5 look-ahead adds patient calls).
> Many cross-method / per-arm cells degrade to "not enough data" banners here. The settled results
> are in the [`L0` view](../L0/SUMMARY.md); the overview is in the [`all` view](../all/SUMMARY.md).

---

## 1. What this view covers
| Arm | K | Iters scored | Usable? |
|---|---|---|---|
| `PTO_LA5`  | 5 | 0–4 | yes (4 iters — short trajectory) |
| `GRPO_LA5` | 5 | 0–1 | **no** — dropped from per-arm batteries (<3 iters) |

So in practice this view shows the **PTO_LA5** trajectory and very little for GRPO_LA5. PTO-vs-GRPO
at K=5 is **not yet comparable** (GRPO has only base + iter 1).

## 2. What we can say so far
See [`figures/1_outcomes/`](figures/1_outcomes/),
[`tables/6_stats/main_results.md`](tables/6_stats/main_results.md), and
[`tables/6_stats/slope_by_arm.md`](tables/6_stats/slope_by_arm.md).

- **PTO_LA5 improves over base:** Q1+Q2 **3.00 → 3.89 in 4 iterations** (dz 0.88, *large*, Holm
  p≈0). Per-iteration slope is steep (0.226/iter) but over a short horizon — not directly
  comparable to the 10-iter arms' endpoints.
- **Same reward-hack signature, milder so far:** MICI 0.18 → 0.33 over the 4 iters (smaller than
  PTO_LA0's eventual 0.49, but PTO_LA5 has only run 4 iters).
- The orthogonal-axis structure holds (per-arm PC1 ≈57%).

## 3. The look-ahead question (RQ-i) is on hold
The K0-vs-K5 contrast needs both K arms, so it lives in the `all` view
([`../all/tables/6_stats/k_paired_by_method.md`](../all/tables/6_stats/k_paired_by_method.md)): it shows
**no significant K0-vs-K5 difference at matched early iterations** for PTO; GRPO can't be compared
(LA5 has only 1 scored iter). The one suggestive signal is on **reward faithfulness** — look-ahead
appears to make the short training reward slightly more faithful (LA5 ≥ LA0) in the reliability
curve ([`figures/4_training/`](figures/4_training/)) — but this needs the full K=5 runs to
confirm.

## 4. To complete this view
The arms paused mid-run (2026-06-09/10): PTO_LA5 has a trained-but-unscored **iter-5 adapter** whose
eval conversations were never generated (`model_iter_5` is empty; iteration_6 stopped at pref_pairs);
GRPO_LA5's iteration_2 is incomplete (no adapter). Cheapest first step: a generate-only pass with the
existing PTO iter-5 adapter (96 convs, no training) + `Run_Eval.ipynb` scoring — a 5th PTO_LA5 point.
Then resume both K=5 arms (`PTO_LA5` to iter 10, `GRPO_LA5` from iter 1) when budget allows, re-score,
and regenerate with `python render_views.py L5`. The structure here is ready to fill — every
figure/table will populate once the arms are scored.

## 5. Caveats
- Everything above is over ≤4 iterations for one arm — treat as directional, not conclusive.
- Oracle noise ≈ 0.10 mean |Δ|; absolute scores are Exp3-internal (bf16) only.
