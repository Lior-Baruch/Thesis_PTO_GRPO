# Exp3 data: conversations, scores, adapters

A small therapist model (Llama-3.2-1B) was trained to do Motivational Interviewing against a
simulated patient (gpt-4o-mini). It was trained under 4 conditions, for 10 iterations each.

```
patients.csv                                  the 96 simulated patients
conversations/<arm>/iter_XX/patient_YY.csv    one conversation: turn, speaker, text
scores_gpt-4o-mini.csv                        one row per conversation
scores_claude-haiku-4-5.csv                   the same conversations, second grader
adapters/<arm>/iter_XX/                       the trained LoRA weights
```

**Arms**

| arm | training method | look-ahead |
|---|---|---|
| `GRPO_K0` | GRPO | none |
| `GRPO_K5` | GRPO | 5 turns |
| `PTO_K0` | PTO (preference trees + DPO) | none |
| `PTO_K5` | PTO (preference trees + DPO) | 5 turns |

**GRPO_K0 vs GRPO_K5 are the two conditions in the paper.**

During training, every candidate therapist reply is scored by gpt-4o-mini, and the score is the mean
of Q1 and Q2 below. With look-ahead, the conversation is first simulated 5 turns past the reply, and
the grader scores that longer conversation. Apart from the method and the look-ahead, the four arms
are identical: same base model, patients, reward and training length.

**Iterations.** `iter_00` is the untrained base model. `iter_N` is the model after N training
iterations; its weights are in `adapters/<arm>/iter_N`. Each iteration has one conversation with each
of the 96 patients.

**Patients.** `patient_17` is the same simulated patient in every arm and iteration. The patients
differ in gender, age, problem (smoking or obesity), how long they have had it, whether they have
tried to change, and how cooperative they are. `patients.csv` lists all of these. The therapist
always opens with the same fixed line, so `turn 1` is not model-generated.

**Scores.** Two LLM graders read each full conversation and fill in the questionnaires. Each column
holds one number per conversation (defined in the table below).
- **gpt-4o-mini** is the same model that gave the training reward.
- **claude-haiku-4-5** was not used in training.

Compare within a grader, not across them.

| column | what it measures | range |
|---|---|---|
| `Q1Q2` | mean of Q1 and Q2; the quantity training optimized | 1–5 |
| `Q1` | session satisfaction | 1–5 |
| `Q2` | working alliance / relational communication | 1–5 |
| `WAI_SR` | working alliance (WAI-SR) | 1–5 |
| `CSQ8` | client satisfaction (CSQ-8) | 1–4 |
| `MI_SAT` | satisfaction with the MI session | 1–5 |
| `MITI` | MI treatment integrity, mean of the 4 MITI global ratings | 1–5 |
| `PCT` | share of patient change talk: change / (change + sustain) | 0–1 |
| `MICI` | MI-inconsistent therapist behaviours per therapist turn (**lower is better**) | ≥ 0 |

**Adapters.**
- **Format.** LoRA with r = 16 and α = 16 on the q, k, v, o, gate, up and down projections of all 16
  layers of `meta-llama/Llama-3.2-1B` (that model is gated on Hugging Face).
- **Weight change.** The change to each weight matrix is ΔW = (α/r)·B·A. Compare these products, not
  A and B on their own.
- **Cumulative.** Each iteration continues training the previous adapter, so `iter_10` holds the
  whole change from the base model.
