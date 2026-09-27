# Exp3 data: conversations, scores, adapters

A small therapist model, Llama-3.2-1B (the base model, not Instruct), was trained to do Motivational
Interviewing with a simulated patient (gpt-4o-mini). There are 4 training conditions ("arms"), each
trained for 10 iterations.

```
patients.csv                                  the 96 simulated patients
conversations/<arm>/iter_XX/patient_YY.csv    one conversation: turn, speaker, text, code_gpt-4o-mini, code_claude-haiku-4-5
scores_gpt-4o-mini.csv                        one row per conversation (arm, iteration, patient_id, scores)
scores_claude-haiku-4-5.csv                   the same conversations, second grader
adapters/<arm>/iter_XX/                       the trained LoRA weights
```

## Arms

| arm | training method | look-ahead |
|---|---|---|
| `GRPO_K0` | GRPO | none |
| `GRPO_K5` | GRPO | 5 turns |
| `PTO_K0` | PTO (preference trees + DPO) | none |
| `PTO_K5` | PTO (preference trees + DPO) | 5 turns |

**GRPO_K0 vs GRPO_K5 are the two conditions in the paper.**

**Training reward.** During training, gpt-4o-mini scores each candidate therapist reply as the mean of
Q1 and Q2 (below).

**Look-ahead.** The patient and the current model first continue the conversation 5 more turns past
the reply, and the grader scores that longer conversation.

Otherwise the arms are identical: same base model, patients, reward and training length.

## Conversations

- **Iterations.** `iter_00` is the untrained model. `iter_N` is the model after N training iterations,
  and its weights are in `adapters/<arm>/iter_N`. Every arm has its own `iter_00`: a separate sample
  from the same untrained model.
- **Patients.** Each iteration has one conversation with each of the 96 patients. `patient_17` is the
  same patient everywhere. Patients differ in gender, age, problem (smoking or obesity), how long they
  have had it, whether they tried to change before, and how cooperative they are (`patients.csv`).
- **Opening and ending.** Turn 1 is always the same fixed therapist greeting; the model does not
  generate it. A conversation ends in one of three ways:
  - either side writes "SESSION ENDED" (the marker is removed from the text);
  - it reaches 50 turns;
  - in a few cases (192 of 4,224), the therapist model produces an empty reply.

## Scores

Two LLM graders read each full conversation. **gpt-4o-mini** is the same model that gave the training
reward; **claude-haiku-4-5** was never used in training. Compare within a grader, not across the two.

| column | what it measures | range |
|---|---|---|
| `Q1Q2` | mean of Q1 and Q2; what training optimized | 1–5 |
| `Q1` | session satisfaction | 1–5 |
| `Q2` | working alliance / relational communication | 1–5 |
| `WAI_SR` | working alliance (WAI-SR) | 1–5 |
| `CSQ8` | client satisfaction (CSQ-8) | 1–4 |
| `MI_SAT` | satisfaction with the MI session | 1–5 |
| `MITI` | MI treatment integrity: mean of the 4 MITI global ratings | 1–5 |
| `PCT` | patient change talk: change / (change + sustain) | 0–1 |
| `MICI` | MI-inconsistent therapist behaviours per therapist turn (**lower is better**) | ≥ 0 |

## Utterance codes

Each grader also gave every utterance one MITI/MISC-style code; these are the last two columns of
each conversation file.

**Therapist codes**

| code | meaning |
|---|---|
| `OQ` | open question |
| `CQ` | closed question |
| `SR` | simple reflection |
| `CR` | complex reflection |
| `AF` | affirmation of a specific strength or effort |
| `PRA` | non-specific praise or cheerleading |
| `GI` | giving information |
| `PERS` | persuading or advising |
| `SEEK` | seeking collaboration, emphasizing autonomy |
| `CONF` | confronting or judging |
| `OTH` | other |

**Patient codes:** `CT` change talk · `ST` sustain talk · `NEU` neither.

One conversation has no claude-haiku codes: `PTO_K5/iter_03/patient_15`.

## Adapters

- **Format.** Standard PEFT LoRA adapters for `meta-llama/Llama-3.2-1B`, which is gated on Hugging
  Face. They use r = 16 and α = 16 on the q, k, v, o, gate, up and down projections of all 16 layers.
- **Weight change.** The change to each weight matrix is ΔW = (α/r)·B·A. Compare these products, not
  A and B separately.
- **Cumulative.** Each iteration continues training the previous adapter, so `iter_10` is the whole
  change from the base model.
