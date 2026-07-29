# figures/ — method schematics

Hand-authored diagrams explaining how the two methods work. Regenerate with:

```powershell
& ..\..\.venv\Scripts\python.exe build_method_figures.py
```

Unlike everything under [`../eda/results/`](../eda/results/), these read no data: they have no
VIEW, no `<judge>/` level, and no producing notebook. They are here rather than there so
`tools/render_views.py` never tries to own them.

They carry **no in-figure title** — the title belongs to the slide or the LaTeX caption, and
repeating it inside the PNG duplicates it everywhere the figure is used. Each one does carry its
colour legend at the bottom, so it still reads standalone. Use the captions below verbatim in the
thesis, or as the starting point for one.

| File | Caption |
|---|---|
| `pto_framework.png` | **Preference Tree Optimization (PTO) — one training iteration.** Figure 1 of the ICLR 2025 paper redrawn for Exp3. The current policy π*ₙ*, the simulated patient and the oracle jointly produce preference-tree data; a DPO update on that data yields π*ₙ₊₁*, which becomes the next iteration's policy. Two things differ from the paper: a pair is kept only where the two candidates differ by more than τ, and the conversations generated at the start of the iteration double as the evaluation set for π*ₙ*. |
| `grpo_framework.png` | **Group-Relative Policy Optimization (GRPO) — one training iteration.** The same loop with the same models. The structural difference is where the oracle sits: in PTO it scores candidates *before* the update in order to select a preference pair, whereas in GRPO it is called *inside* the update as the reward for every one of the G completions. The prompt list is sliced from the rollout once per iteration and then fixed. |
| `pto_preference_tree.png` | **PTO — inside one branch point (greedy mode).** Figure 2 of the ICLR 2025 paper redrawn for Exp3. At each therapist turn the policy samples M candidates; each is rolled forward K further turns before the oracle scores the resulting trajectory. Exactly two candidates survive — best and worst — and only where their scores differ by more than τ, so a branch point can produce no training data at all. The winner is appended to the trunk, which is what makes the search sequential: each choice conditions the next branch point. |
| `grpo_group_rollout.png` | **GRPO — inside one group.** Drawn on the same row grid as `pto_preference_tree.png` so the two can be read against each other. Everything above the oracle is identical — same policy, same G samples, same K-turn look-ahead, same full-trajectory scoring. The methods diverge only in what happens to the scores: all G completions carry a gradient weighted by the group-relative advantage, there is no threshold, and there is no trunk to advance. |

**Parameters shown.** M = G = 8, K = 5, MCL = 12 — the Exp3 settings. The look-ahead rows are drawn
as two explicit turns plus an ellipsis rather than all K.

**Colour code** (shared by all four): green = data the iteration produces · purple = an API model
we call (patient or trajectory) · blue = the policy being trained · orange = the oracle and the
update step · grey = intermediate scalars.
