# The win-minus-lose embedding direction, per GRPO round

The Exp2 preference-direction analysis, redone on the GRPO rounds in `grpo_groups/`, per round and
using the whole group of 8.

- **Data.** The `train` rounds of both GRPO arms, iterations 1–10, whose 8 rewards are not all equal:
  29,861 rounds (K=0 15,726 + K=5 14,135), 238,888 candidates.
- **Embedding.** Same as Exp2: `Alibaba-NLP/gte-base-en-v1.5`, mean-pooled, normalised. It reproduces
  Exp2's cached word embeddings exactly (cosine 1.00000 on a 2,000-word sample).
- **Code.** [`analyze_grpo_group_directions_2026-09-29.py`](../../build/analyze_grpo_group_directions_2026-09-29.py)
  reads only the shared folder, so it reruns on Doron's copy.

## What was computed

- **Direction.** For each round, Δ = embedding(best) − embedding(worst). For each arm and iteration,
  the direction is the normalised mean Δ, which is Exp2's probe.
- **Held-out scores.** The direction is fitted on 4/5 of the patients and scored on the rest, in 5
  folds. Two scores:
  - **accuracy**: the share of rounds where the best candidate projects above the worst;
  - **within-round ρ**: the Spearman correlation between projection and reward over all the round's
    candidates.

  95% CIs resample patients.
- **Length controls.** The median reply grows from 216 characters at iteration 1 to 982 at iteration
  10 (K=5: 226 to 1,023). Every score therefore has two companions:
  - a length-only predictor (the longer candidate wins);
  - the direction with its linear length component removed.
- **GRPO's own weighting.** A second direction weights every candidate by its advantage. It is almost
  the same direction (cosine 0.81–0.99 with the best-minus-worst one), and its accuracy is within 0.02
  of it at every iteration.

## Findings

**1. K=0 has a consistent direction throughout; K=5's fades to nothing** ([probe.png](figures/probe.png)).
- **K=0.** Accuracy is 0.59–0.71 (mean 0.648), and all 10 CIs are above 0.5 (lowest bound 0.551).
  Within-round ρ is 0.10–0.24.
- **K=5.** Accuracy rises from 0.564 to 0.641 at iteration 4, then falls every iteration to 0.509 at
  iteration 10 (trend over iterations ρ = −0.66, p = .038). The CIs include 0.5 at iterations 9 and 10.
  Within-round ρ falls to 0.010.
- **Split-half reliability** tells the same story ([direction_similarity.png](figures/direction_similarity.png), b):
  - K=0 stays at 0.60–0.92 (trend ρ = −0.32, p = .37).
  - K=5 falls from 0.84 at iteration 4 to 0.07 at iteration 10 (ρ = −0.83, p = .003).
- **Reading.** By the end of training, what separates a K=5 winner from a loser is no longer one
  direction in the text of the reply.

**2. It is not length.**
- Removing the length component barely moves accuracy: mean 0.648 → 0.632 for K=0 and 0.581 → 0.579
  for K=5.
- Length alone is at or below chance: mean 0.478 for K=0 and 0.510 for K=5.
- Early in K=0 the **shorter** reply tends to win: the length-only accuracy is 0.356 at iteration 1.

**3. Per round, most of the difference is specific to that round** ([per_round_alignment.png](figures/per_round_alignment.png)).
- The median cosine between one round's Δ and the held-out direction is 0.03–0.10 for K=0 and
  0.00–0.07 for K=5.
- The direction is a small shared component on top of mostly round-specific differences.

**4. K=0 and K=5 prefer the same thing early, then part** ([direction_similarity.png](figures/direction_similarity.png)).
- **Iterations 1–5.** The two arms' directions have cosine 0.70–0.86 at the same iteration.
  Consecutive iterations have cosine 0.82–0.91 within each arm.
- **Iterations 6–8.** The cross-arm cosine drops to 0.37–0.50. Part of that drop is noise, since K=5's
  reliability is down to 0.30–0.46.
- **K=0 iteration 9 is an outlier.** Its cosine with every other direction is between −0.28 and 0.26,
  although it is reliable within itself (split-half 0.70). It is also where length flips: the shorter
  reply wins, with length-only accuracy 0.39, against 0.56 at iterations 8 and 10.

**5. The models move along their own direction** ([policy_shift.png](figures/policy_shift.png)).
- **Direction of the move.** The shift of the model's own replies from `iter_{N-1}` to `iter_N`
  (therapist turns 13 and later, in `conversations/`) is closer to its own arm's iteration-N direction
  than to the other arm's in 18 of 20 iterations. The two exceptions are K=0 iteration 1 and K=5
  iteration 10, whose direction is not reliable.
- **K=5 over time.** Its replies move steadily along its pooled direction up to `iter_05` (0 → 0.176),
  then stop (0.158–0.177 through `iter_10`). This is where its direction fades.
- **K=0 over time.** Its replies move in jumps: 0.03–0.06 through `iter_07`, then 0.152, −0.030 and
  0.186.

**6. What the direction is** ([words.png](figures/words.png), `tables/examples_pooled.csv`).
- **Words.** In both arms the most winner-like words are *encouragement, motivation, empowerment,
  inspiration*. The two arms' word scores correlate at Spearman 0.82 over 47,361 words.
- **K=5 compared with K=0.**
  - K=5's direction leans toward *habit, mindset, procrastination, planner, budgeting, prioritise*.
  - K=0's leans toward *embraced, applauded, proud, cheered, affectionate*.
- **Example replies.**
  - K=0's most winner-like candidates are praise: "I'm so proud of your progress…", from iterations
    8–10.
  - K=5's are small-goal reframes: "setting a more achievable goal … rather than … losing 2 pounds in
    a week".
  - In both arms the most loser-like candidates are garbled multilingual text from the early
    iterations.
- **Exp2's MI word lists** (change talk, sustain talk, therapist actions) all sit above the average
  word, at z = 0.8–1.9. They do not separate the categories: sustain talk is above average too.

## Caveats

- **The embedding is external.** gte is a sentence encoder, not the therapist model's own
  representations, and single-word projections are a coarse reading of a sentence-level direction.
- **Garbled replies.** A rough heuristic (more than 3% non-ASCII characters) finds garbled text in the
  worst candidate of 0.8–9.5% of rounds and in the best of 0.1–3.8%. So it accounts for a few points of the
  early accuracy at most.
- **Leaked chat marker.** In K=0 iterations 7–10, 24–58% of candidates contain a malformed chat-marker
  fragment, mostly `<|im_end>`. The trainer's stop strings catch only the exact `<|im_end|>` and
  `<|im_start|>`. The fragment is more common in the best than in the worst candidate: 41.6% vs 28.5% at
  iteration 10. The late K=0 direction may partly encode it. At K=5 it appears in at most 1.8% of
  candidates.
- **Rounds lost to crashes.** Iterations that crashed and resumed have fewer epoch-1 rounds: K=0
  iterations 2, 6, 8 and K=5 iterations 1, 2, 7.
- **Ties dropped.** Tied rounds are dropped: 16–19% of K=5 train rounds in iterations 9–10, and at most
  6.3% elsewhere.

## Files

`figures/`: `probe.png`, `direction_similarity.png`, `per_round_alignment.png`, `policy_shift.png`,
`words.png`.

`tables/`: `probe_by_iteration.csv` (every score + CI per arm and iteration), `direction_cosines.csv`,
`per_round_alignment.csv`, `policy_shift.csv`, `policy_trajectory.csv`, `word_projection_pooled.csv`,
`mi_categories.csv`, `examples_pooled.csv`.
