# Checking Doron's "win/lose direction" deck (gte, 2026-10-05)

Doron's October deck reads, for each GRPO arm and training iteration, the mean of
embedding(best reply) − embedding(worst reply) over all rounds. He compares these directions across
iterations and between the arms, and reads them by projecting a pool of frequent real sentences onto
them. This re-runs his quantities on our data with one encoder, gte-base, and adds three checks: remove
the leaked-chat-marker and degenerate candidates, correct the cosines for noise, and test his K0-vs-K5
reading against text features that were fixed before his deck existed.

- **Data.** The 29,861 `train` rounds from the 2026-09-29 analysis (K=0 15,726, K=5 14,135; rounds with
  all 8 rewards tied excluded), 238,888 candidates. Embeddings come from that analysis's cache.
- **Ties.** When several candidates share the best (or worst) reward, their embeddings are averaged;
  Doron takes a single best and a single worst.
- **Sentence pool.** Doron's recipe: sentences of 4–10 words that occur at least 3 times verbatim, capped
  at 4,488. On our rounds only 4,178 sentences reach 3 occurrences (out of 225,240 distinct), and 3.1% of
  them contain a chat marker.
- **Variants.** `all`; `no_leak` (candidates containing `<|im_…` removed); `clean` (leaked and degenerate
  candidates removed). In the filtered variants, best and worst are re-chosen among the candidates that
  remain.
- **Noise-corrected cosine.** Patients are split in half 50 times. The corrected cosine is
  cos(half A of one direction, half B of the other), divided by √(reliability₁ × reliability₂), where
  reliability is a direction's own split-half cosine. A value near 1 means "the same direction once
  noise is removed".
- **Code.** [`check_doron_directions_2026-10-05.py`](../build/check_doron_directions_2026-10-05.py).

## Findings

**1. Leaked chat markers do not lose. The markers in the "lose" sentence lists are a readout artefact**
(`tables/artefact_rates.csv`, `tables/marker_share_top_sentences.csv`).
- **K=0 leaks a lot.** 19.9% of K=0 candidates contain a leaked marker (58.1% at iteration 7, 50.1% at
  iteration 8); K=5 1.7%.
- **The leak is not on the losing side.** Among K=0's best candidates 21.3% leak, among its worst 19.3%.
  The within-round correlation with reward is +0.05.
- **Removing leaked candidates barely moves the directions.** Cosine with the all-candidates direction:
  K=0 mean 0.96 (lowest 0.83), K=5 0.99 or higher.
- **Yet markers crowd the "lose" end of the sentence pool.** 45–48% of each arm's 10 lowest-projecting
  sentences contain a marker, against 3.1% of the pool. With leaked candidates removed from the direction
  it is still 44–47%. With degenerate candidates removed as well it drops to 12–19%.
- **Reading.** The direction pushes away from degenerate text: degenerate replies are several times more
  common among the worst candidates than among the best (K=0 6.2% vs 1.6%, K=5 5.2% vs 1.9%). In gte
  space, sentences that carry a marker sit near garbled text, so they surface at the "lose" end even though
  leaked replies themselves win and lose at the base rate.
- **So the deck's "lose = templated list openers with chat tokens" describes the encoder, not the
  reward.** This corrects my caveat of 2026-10-04.

**2. The "win" side replicates** (`tables/top_sentences.md`).
- **K=0.** From iteration 3 on, K=0's top win sentences are support and praise in every variant ("I'm
  thrilled to support you on your weight loss journey", "I'm so proud of you and your commitment to
  change").
- **K=5.** Its top win sentences are encouragement about the process ("Remember, quitting smoking is a
  journey, not a destination") and open questions ("How do you want to approach weight loss?").

**3. Most of the late divergence and instability is noise** (`tables/cosines.csv`).

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K0 vs K5, raw | 0.83 | 0.87 | 0.84 | 0.72 | 0.68 | 0.36 | 0.49 | 0.53 | −0.03 | 0.12 |
| K0 vs K5, noise-corrected | 0.95 | 0.96 | 0.90 | 0.76 | 0.79 | 0.48 | 0.68 | 0.75 | −0.05 | 0.37 |
| K=0 reliability | 0.93 | 0.83 | 0.90 | 0.89 | 0.69 | 0.59 | 0.61 | 0.74 | 0.77 | 0.92 |
| K=5 reliability | 0.64 | 0.81 | 0.84 | 0.87 | 0.80 | 0.58 | 0.50 | 0.38 | 0.17 | 0.07 |

- **The raw numbers match Doron's.** K0 and K5 agree early, and the raw cosine falls late.
- **K=5 stops having a direction.** Its reliability drops from 0.87 (iteration 4) to 0.17 and 0.07 at
  iterations 9–10. There is nothing left to agree with, which matches the 2026-09-29 finding that late
  K=5 rounds have no direction. Its top sentences at iterations 9–10 are noise.
- **Real but partial divergence at iterations 4–8.** The noise-corrected cosine is 0.48–0.79 there.
- **Stability within each arm holds once noise is removed.** The raw cosine between consecutive iterations
  falls (K=0 from 0.92 to 0.70 by iteration 7; K=5 from 0.92 to 0.18 by iteration 10). Noise-corrected it
  is 0.91–0.98 for K=0 through iteration 8, and 0.77–1.03 for K=5. K=5's iteration 9–10 values rest on
  reliabilities of 0.17 and 0.07, so they say little.

**4. K=0's iteration 9 is a real reversal, not the grader failure.**
- **It is reliable.** K=0's iteration-9 direction has split-half reliability 0.77 over 1,200 rounds, and
  its cosine with iteration 8 is −0.30 (noise-corrected −0.32).
- **The grader failure cannot explain it.** Only 32 candidates lost their reward, and they are excluded
  from every round.
- **Agreement wins that iteration.** Its winning sentences are agreement replies ("Absolutely, you're
  absolutely right", "I understand exactly where you're coming from").
- **Iteration 10 follows neither.** Its direction is unrelated to iteration 9's (cosine −0.08).

**5. K0 − K5: the praise side replicates; the "reframing" side only partly** (`tables/k0_minus_k5_sentences.md`,
`tables/k0_minus_k5_features.csv`, `tables/doron_keyword_counts.csv`).
- **The K=0 side is praise and support** in most iterations, with or without the filter: "I couldn't be
  more proud of you", "You have my full support and unconditional acceptance".
- **The K=5 side is mixed.** It has reframes and open questions ("What other ideas come to mind?",
  "So, instead of focusing on what you need to do,"; iterations 3 and 5). It also has tips ("One tip is to
  keep a food journal", "Remember, consistency is key") and smoking-cessation topic sentences
  (iterations 4 and 9).
- **Doron's own keyword lists, applied to the 10 top sentences per side per iteration:**
  - his praise list hits 49 of 100 on the K=0 side (51 clean) and 0 on the K=5 side;
  - his reframing list hits 10 of 100 on the K=5 side and 0 on the K=0 side.
- **Text features fixed on 2026-09-29.** Each is correlated with the projection on K0 − K5, within each
  arm's own candidates (clean variant):
  - *praise phrase:* lines up with the K=0 side from iteration 5 on (+0.13 to +0.32, except iterations 7
    and 9) and not before (−0.02 to +0.07);
  - *reflection opener:* about zero throughout (−0.09 to +0.03);
  - *length:* the strongest correlate. The K=5 side holds the longer replies in 8 of 10 iterations
    (correlation as low as −0.52); in iterations 7 and 10 the sign flips (+0.27, +0.18).
- **Reading.** "K=0's reward pulls toward validation and praise" holds. "K=5's reward pulls toward
  reframing" is too strong. What the data supports is that K=5's reward lacks K=0's pull toward praise,
  and its side of the difference mixes questions, reframes, tips and topic content.

## Caveats

- **One encoder.** Doron's claim that the result is the same in every embedder is not tested here. gte is
  an external sentence encoder, like his Qwen3 and mxbai.
- **The directions are weak per round.** On 2026-09-29, a held-out direction picked K=0's winner in about
  69% of rounds and ranked a round's 8 candidates at Spearman ≈ 0.2. The mean direction summarises a
  tendency; it does not decide most rounds.
- **Top-sentence lists are read by eye.** Only the keyword counts and the feature correlations are
  quantitative, and the keyword lists were chosen by Doron after he saw his sentences.
