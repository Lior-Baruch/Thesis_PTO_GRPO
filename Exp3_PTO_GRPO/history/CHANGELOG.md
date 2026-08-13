# Exp3_PTO_GRPO — CHANGELOG (index)

The dated history, split by subject because a single file had grown past 1,000 lines. Newest first
within each; this index is stable so inbound links keep resolving.

| File | Covers |
|---|---|
| [CHANGELOG_STATUS.md](CHANGELOG_STATUS.md) | **run status + findings** — what we learned when, and what we believed before (2026-07-26 → today) |
| [CHANGELOG_EDA.md](CHANGELOG_EDA.md) | the **EDA** — the `eda_analysis` package, the notebooks, the score lake, the results tree (2026-06-09 → today) |
| [CHANGELOG_TRAINER.md](CHANGELOG_TRAINER.md) | the **trainers + `code/_shared/`** — resume, checkpointing, batched look-ahead, EDA capture, throughput, the first-run and ChatML-leak fixes, the dependency audit |

All three are provenance only — read them for *"when did we learn this?"*, never for *"what is true
now?"*. The **current** state they established lives in:

- [STATUS.md](../../STATUS.md) — run status, headline numbers, cost constraint, next step
- [CLAUDE.md](../../CLAUDE.md) § "Exp3 · Training internals" (trainer behaviour) and
  § "Exp3 · EDA workflow" (the EDA)

There is no longer a root-level changelog: it was a thin index whose every entry pointed here, so
it was removed on 2026-07-29 and these two files became the only dated history in the repo.
(Exp1 is frozen and Exp2 complete — neither will gain entries.)
