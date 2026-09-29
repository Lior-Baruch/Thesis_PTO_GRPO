# What separates a winning GRPO reply from a losing one

Two estimates of the direction in reply-embedding space that GRPO's rounds reward: one from each
round's best and worst candidate, one from all 8. Both GRPO arms, training iterations 1–10.

- **Data.** The `train` rounds whose 8 rewards are not all equal: 29,861 rounds (K=0 15,726 + K=5
  14,135), 238,888 candidates.
- **Embedding.** Every reply is embedded with `gte-base-en-v1.5` (mean-pooled, unit length).
- **Within-round only.** A round's candidates share one conversation so far, so every quantity here
  compares candidates *within* a round.
- **Code.** [`analyze_grpo_round_directions_2026-09-29.py`](../../build/analyze_grpo_round_directions_2026-09-29.py)
  reads only the shared folder.

## The two analyses

**1. Best vs worst.** Each round gives one pair: embedding(best) − embedding(worst), called Δ. The
direction is a Bradley–Terry probe: P(best beats worst) = σ(w·Δ), fitted as an L2-regularised logistic
regression. The plain mean of Δ is shown beside it as the no-model baseline.

**2. All 8.** Every candidate counts, through the value GRPO actually trains on: its advantage,
(reward − round mean) / round std. The direction is a ridge regression of the advantage on the
round-centred embedding. The advantage-weighted mean embedding is the no-model baseline; it is GRPO's
own weighting.

**Scoring.** Directions are fitted on 4/5 of the patients and scored on the other 1/5, in 5 folds. The
folds are the same for every arm and iteration, so no patient is ever on both sides. Every round gets
three scores:
- **best-vs-worst accuracy:** does the best candidate project above the worst;
- **all-pairs accuracy:** the same question over every pair of candidates with different rewards;
- **within-round Spearman:** projection vs reward over all the round's candidates.

Every score is also computed for reply length alone. 95% intervals resample patients.

**Regularisation.** Chosen by the mean held-out score over all 20 arm × iteration cells: C = 1 for
Bradley–Terry (grid 0.1–1,000) and α = 10 for ridge (grid 1–10,000). Choosing it this way is slightly
optimistic.

**Degenerate text.** Candidates that are empty, more than 3% non-ASCII, or mostly outside the corpus
vocabulary are flagged: 3.4% of candidates. 78.7% of rounds have none, and every score is repeated on
those clean rounds.

## Findings

**1. Best vs worst: a consistent direction for K=0; K=5's fades to chance** ([figure](figures/analysis1_best_worst.png)).
- **K=0.** Held-out accuracy is 0.640–0.745 (mean 0.693), and all 10 intervals are above 0.5 (lowest
  bound 0.608).
- **K=5.** It rises from 0.595 to 0.660 (iterations 4–5), then falls to 0.516 at iteration 10, with an
  interval of 0.495–0.537 there (trend ρ = −0.68, p = .029).
- **Not length.** Reply length alone scores 0.35–0.60 (means 0.481 and 0.512).
- **Not degenerate text.** On clean rounds the mean drops only to 0.670 (K=0) and 0.584 (K=5).
- **It tracks the size of the reward difference.** For K=0, accuracy rises from 0.636 in the rounds
  whose best and worst are closest to 0.785 in the rounds where they are furthest apart. For K=5 it
  rises from 0.556, then levels off at about 0.62.

**2. All 8: the same picture, ranked across the whole group** ([figure](figures/analysis2_all8.png)).
- **K=0.** Within-round Spearman is 0.13–0.28 (mean 0.205).
- **K=5.** It rises to 0.176 at iteration 3, then falls to 0.006 at iteration 10 (interval −0.010 to
  0.023; trend ρ = −0.83, p = .003).
- **Ridge beats GRPO's own weighting.** It beats the advantage-weighted mean in 19 of 20 cells (mean
  0.205 vs 0.171 for K=0, 0.111 vs 0.101 for K=5).
- **Clean rounds.** The means are 0.181 (K=0) and 0.090 (K=5).

**3. Using all 8 gives a slightly better, more stable version of the same direction** ([figure](figures/compare.png)).
- **Same direction.** The two directions have cosine 0.85–0.93 in every cell.
- **All 8 scores higher.** It beats best-vs-worst on best-vs-worst accuracy in 14 of 20 cells, on
  Spearman in 17, and on split-half reliability in 18.
- **The gain is small** (K=0 means):
  - best-vs-worst accuracy: 0.701 vs 0.693;
  - Spearman: 0.205 vs 0.198;
  - reliability: 0.480 vs 0.414.
- **Reliability is modest everywhere.** K=0's all-8 direction is at 0.35–0.73. K=5's falls from 0.37
  (iterations 4–5) to 0.07 (iteration 10).

**4. The arms share a direction early; late K=5 has none** ([figure](figures/transfer.png), all-8 numbers).
- **Early transfer works both ways.** A direction fitted on K=0 picks K=5's winners at 0.62–0.65 in
  iterations 1–5. One fitted on K=5 picks K=0's winners at 0.61–0.73.
- **Nothing predicts late K=5.** No direction fitted on another arm or iteration picks K=5 iteration 9
  or 10 winners better than 0.55, and their own held-out directions reach only 0.56 and 0.52.
- **K=0 iteration 9 runs the other way.** K=5's directions from iterations 1–8 pick its winners *below*
  chance, at 0.43–0.48.

**5. What the direction is** ([figure](figures/features.png), `tables/examples.md`).
- **Coherent, finished replies win.** The direction's strongest correlates are degenerate text (−0.45
  to −0.51) and a reply cut off mid-sentence (−0.21 to −0.29). In both cases it amplifies what the
  reward itself only mildly prefers: the reward's correlations are −0.09 to −0.13 and −0.06 to −0.07.
- **The content differs by arm.**
  - K=0's direction also favours advice (+0.14), praise phrases (+0.12) and "you are…" affirmations
    (+0.08 to +0.09).
  - K=5's favours advice (+0.10) but not praise (+0.005).
- **The leaked chat marker is not what the direction encodes** (−0.01 for K=0), although the reward
  slightly favours it (+0.05).
- **Example replies.**
  - K=0's highest-scoring candidates are praise and commitment talk: "I'm so proud of you for
    committing to your plan…".
  - K=5's affirm the patient's own insight: "I really commend you for visualizing this whole process…".
  - In both arms, most of the lowest-scoring are replies that turn garbled partway through, which the
    degenerate-text flag misses.

## Caveats

- **The encoder is external.** gte is a sentence encoder, not the therapist model's own representations.
- **The direction is weak per round.** Even at its best, a held-out direction picks the winner in
  3 rounds out of 4, and ranks the 8 at Spearman ≈ 0.2–0.3. Most of what decides a round is not a
  single direction.
- **Degenerate text is under-detected.** The flag misses replies that start fine and turn to salad, so
  the clean-round scores are an upper bound on how much survives without degenerate text.
- **Rounds lost to crashes.** Iterations that crashed and resumed are missing epoch-1 rounds (K=0: 2, 6,
  8; K=5: 1, 2, 7). Tied rounds are excluded.

## Files

- `figures/`: `analysis1_best_worst.png`, `analysis2_all8.png`, `compare.png`, `transfer.png`,
  `features.png`.
- `tables/`:
  - `scores_by_iteration.csv`: every score and interval, for both analyses and both baselines, on all
    and on clean rounds, plus reliability and the cosine between the two directions;
  - `by_reward_difference.csv`, `transfer_best_worst.csv`, `transfer_all8.csv`, `features.csv`;
  - `examples.md`;
  - `hyperparameters.json`.
