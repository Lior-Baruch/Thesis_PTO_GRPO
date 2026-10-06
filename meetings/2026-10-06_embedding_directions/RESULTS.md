# Doron's win/lose direction in five embedders, read through sentences and words (2026-10-06)

Lior, 2026-10-06: "lets run the embedding eda like doron did (also words and also sentences with multiple
embedders) and think how it can fit the paper". This runs the quantity of Doron's October deck (*Win/Lose
Direction: Does the Embedding Model Matter?*) in five embedding spaces, with the checks of the 2026-10-05
gte re-run built in. Numbers here live in `meetings/`, not the EDA: port them before any enters the paper.

- **Data.** The GRPO rounds of the Doron share (`train` split, rounds whose scored candidates hold at least
  two rewards): 29,861 rounds, 238,888 candidates (11.3% contain a leaked chat marker, 3.4% degenerate text).
- **Directions.** Per run x training iteration: `winlose` (Doron's: mean of emb(best) - emb(worst), ties
  averaged) and `advw` (the paper's App C.5 estimator: reward deviation rescaled to sum |w| = 2 per round).
  Candidate sets `all` and `clean` (no leaked marker, no degenerate text; best/worst re-chosen).
- **Embedders.** all-MiniLM-L6-v2 (the paper's), gte-base-en-v1.5, mxbai-embed-large-v1 and
  Qwen3-Embedding-0.6B (Doron's two), Llama-3.2-1B layer 8 mean-pooled without BOS (the therapist's base).
- **Readouts.** Noise-corrected cosines (50 patient split-halves); Doron's sentence pool (4,047 sentences
  without a chat marker); seven sentence categories fixed before the run, scored as (mean projection of the
  category - mean of the rest) / pool SD with 500 patient bootstraps; single words (wordfreq top 20k and
  the 4,766-word corpus vocabulary) embedded alone; and an embedding-free log-odds of words in the best vs
  the worst reply of each round (Monroe, Colaresi and Quinn 2008).
- **Code.** [`embedding_directions_2026-10-06.py`](../build/embedding_directions_2026-10-06.py) (about 1 h on
  the local GPU; caches in `.emb_cache/`, gitignored) and
  [`plot_embedding_directions_2026-10-06.py`](../build/plot_embedding_directions_2026-10-06.py).

## Findings

**1. The App C.5 direction findings hold in all five embedders** (`tables/cosines.csv`,
`figures/cosines_advw.png`, `figures/cosines_winlose.png`). All candidates, the paper's estimator:

| quantity | MiniLM | range over 5 embedders |
|---|---|---|
| K0 vs K5 corrected, iterations 1-3 | 0.92-0.96 | 0.80-0.96 |
| K0 vs K5 corrected, iteration 6 | 0.57 | 0.44-0.78 |
| K0 vs K5 corrected, iteration 9 | -0.32 | -0.65 to -0.07 |
| K0 iteration 9 vs 8 (the reversal) | -0.65 | -0.72 to -0.37 |
| K5 split-half reliability, iterations 8-10 | 0.15-0.32 | 0.05-0.67 |

- MiniLM reproduces `lookahead/mechanism/tables/direction_k_by_iter_grpo.md` within 0.05 (pipeline check).
- Llama layer 8 measures late K5 best (reliability 0.35-0.67).
- Without leaked/degenerate candidates the runs agree less (MiniLM iteration 6: 0.57 -> 0.27); part of the
  shared early direction is "away from broken text".

**2. What the directions point to** (`tables/categories.csv`, `tables/categories_summary.csv`,
`figures/categories_summary.png`; clean, winlose):

- K0 toward praise sentences at iterations 4-8 in every embedder (lowest embedder 0.56-1.27 pool SD), down at
  9 (mean -0.27), back at 10 (1.42).
- K5 toward questions and advice through iteration 5-6; toward praise only at 7-8 (means 0.69, 0.89).
- Reflection openers: flat for both (a small category, 55 sentences).
- Embedders rank the pool alike only moderately (Spearman 0.50-0.70 between pairs,
  `tables/embedder_agreement.csv`).
- Top sentences (`tables/top_sentences_<embedder>.md`): K0 side "I'm thrilled to support you on your weight
  loss journey" (it. 3, 4 of 5 embedders), "Fantastic, I'm so proud of you!" (5), "Thank you for being so
  honest and vulnerable" (10); K5 side "What could you focus on instead?", "What other ideas come to mind?",
  "Instead of focusing on ..." (3-8), tips and lists at 10.

**3. Words** (`tables/top_words_<embedder>.md`, `tables/lexical_logodds.md`).

- Words embedded alone give topic words and noise (Llama mostly noise), as Doron found.
- The embedding-free log-odds is clear. K0 winners over iterations 4-10: you 9.0, i 8.7, so 7.0, proud 6.6,
  every 6.6, i'm 6.2; at 10: me, i, my, myself, admire, unwavering, courage.
- K5 winners: yourself, cravings, small, behavior (1-3); healthy, quit, exercise (4-10). First person loses
  under K5 (i -5.3, me -3.6 at 1-3).

## Caveats

- Per-round directions are weak (2026-09-29: about 69% held-out best-vs-worst); these are cell averages.
- The categories are regex groups written for this run; "advice" and "reflection opener" are small (95, 55).
- From iteration 2 on the two runs sample from different policies, so a direction mixes what each reward
  prefers with what each policy proposes.
- Decision page for Lior (private): https://claude.ai/artifact/F8t9EzUh5GA6exTRgDkEMz
