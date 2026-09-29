# GRPO rounds: all 8 candidates with their scores

In each GRPO round the model is given a conversation so far and writes 8 candidate therapist replies.
gpt-4o-mini scores each one, and the update pushes the model toward the candidates that scored above
the round's average and away from those below it. These files keep every round of both GRPO arms, with
all 8 candidates.

```
grpo_groups/<arm>/iter_XX.jsonl.gz    one line per round (gzipped JSON lines)
```

**`iter_XX` is training iteration XX**: the model moving from `iter_{XX-1}` to `iter_XX` (the weights
in `adapters/`). Its conversations so far are cut from `conversations/<arm>/iter_{XX-1}`. The model is
updated during the iteration, so later rounds in a file come from a slightly more trained model.

## One round

```json
{"arm": "GRPO_K5", "iteration": 3, "round": 0, "epoch": 1, "split": "train",
 "patient_id": 83, "turn": 13,
 "conversation_so_far": "[THERAPIST]: Hello, welcome ...\n\n[PATIENT]: ...",
 "candidates": [{"text": "Well, there's no easy solution ...", "reward": 3.11, "Q1": 2.8, "Q2": 3.41,
                 "lookahead": "\n\n[PATIENT]: I've thought about it, but ..."},
                ... 7 more ...]}
```

| field | meaning |
|---|---|
| `round` | number of the round within the file, in training order |
| `epoch` | each iteration runs 2 epochs over the same conversation-so-far points and samples 8 new candidates each time |
| `split` | `train` rounds were used for the update; `eval` rounds are held-out points that were only scored |
| `patient_id` | the same ids as `patients.csv` and `conversations/` |
| `turn` | the candidates are this turn of the conversation (13–51) |
| `reward` | the training reward: the mean of Q1 and Q2 |
| `Q1`, `Q2` | the grader's two scores, 1–5 |
| `lookahead` | `GRPO_K5` only: the (up to) 5 turns the patient and the model continued after this candidate. The grader scored the conversation so far + the candidate + these turns |

## How the update uses a round

Each candidate's weight is its **advantage**: (reward − the mean of the 8) / the standard deviation of
the 8. There is no single winner: all 8 candidates push, in proportion to their advantage.

None of the 8 candidates continues the conversation. The conversation so far comes from the model's own
conversation in `conversations/<arm>/iter_{XX-1}`.

## Things to know

- **Ties.** A round where all 8 rewards are equal has no advantage, so it gives no update signal. That
  is 1–6% of rounds, except 16–18% in `GRPO_K5` iterations 9–10.
- **Missing epoch-1 rounds.** Some iterations crashed and resumed, and the rounds before the crash were
  not recorded. `GRPO_K0` iteration 2 and `GRPO_K5` iteration 1 have no epoch-1 rounds; `GRPO_K0`
  iterations 6, 8 and `GRPO_K5` iterations 2, 7 have part of epoch 1. Epoch 2 is complete everywhere.
- **Empty replies** get a reward of 0 (a handful per iteration at most). In `GRPO_K0` iteration 1 they
  were not sent to the grader, so their `Q1` and `Q2` are empty.
- **`reward` is empty** for 32 candidates, all in `GRPO_K0` iteration 9: the grader call failed.
- **Empty `lookahead`** for 1–4% of `GRPO_K5` candidates: the continuation stopped before its first turn,
  so the grader scored just the conversation so far + the candidate.

## Reading it

```python
import gzip, json
import pandas as pd

rounds = [json.loads(line) for line in gzip.open("grpo_groups/GRPO_K5/iter_03.jsonl.gz", "rt", encoding="utf-8")]
candidates = pd.json_normalize(rounds, "candidates", ["iteration", "round", "epoch", "split", "patient_id"])
```
