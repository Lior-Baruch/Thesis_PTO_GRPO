# Exp3: conversations, scores and LoRA adapters

Lior Baruch, thesis Experiment 3. A Llama-3.2-1B "therapist" trained to do Motivational Interviewing
against a simulated patient (gpt-4o-mini), under four training conditions. Built 2026-09-27.

## Start here

- **Reading the two conditions.** The paper's two conditions are **GRPO_K0** (no look-ahead) and
  **GRPO_K5** (5-turn look-ahead). Open `transcripts/GRPO_K0/GRPO_K0_iter10.txt` and
  `transcripts/GRPO_K5/GRPO_K5_iter10.txt` side by side. `iter00` in each folder is the untrained
  starting point. `patient_17` in any file is the same simulated patient as `patient_17` in every
  other file.
- **Analysis.** Everything is in `data/`: flat CSVs keyed by `(arm, iteration, patient_id)`.
- **Weight changes.** The adapters are in `adapters/<arm>/iter_NN/`. See [LoRA adapters](#lora-adapters)
  below.

## Folder layout

```
README.md
arms.csv                         the 4 arms and their training hyperparameters
personas.csv                     the 96 simulated patients: traits + the patient model's system prompt
prompts/                         the therapist's system prompt, fixed opening line and chat template
transcripts/<arm>/<arm>_iterNN.txt   readable; 96 conversations per file, ordered by patient_id
data/
  conversations.csv              one row per conversation
  utterances.csv                 one row per utterance, with each grader's MIPROC code
  scores_main_gpt-4o-mini.csv    one row per conversation, every instrument (main grader)
  scores_heldout_claude-haiku-4-5.csv   the same, held-out grader
  score_means_by_state.csv       per arm x iteration means of the headline columns (orientation only)
adapters/<arm>/
  run_config.json                full training config of that arm
  iter_01 ... iter_10/           adapter_model.safetensors + adapter_config.json
```

Size: 4 arms × 11 model states × 96 patients = **4,224 conversations** (113,941 utterances), plus
4 × 10 = **40 adapters** (45 MB each, ≈1.8 GB in total).

## The four arms

| arm | optimizer | look-ahead K | training run |
|---|---|---|---|
| `GRPO_K0` | GRPO | 0 | `GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8` |
| `GRPO_K5` | GRPO | 5 | `GRPO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_G8` |
| `PTO_K0` | PTO (DPO loss) | 0 | `PTO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_M8_PTgreedy` |
| `PTO_K5` | PTO (DPO loss) | 5 | `PTO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_M8_PTgreedy` |

What all four arms share:
- the same base model (`meta-llama/Llama-3.2-1B`, the base model, not Instruct)
- the same 96 patients
- the same training reward: the mean of Q1 and Q2, scored by gpt-4o-mini
- the same schedule and optimizer settings: 10 iterations × 2 epochs, LoRA r=16, learning rate 1e-5, seed 42

`arms.csv` lists the hyperparameters and `adapters/<arm>/run_config.json` holds each full config.

**One iteration.** The current policy talks to all 96 patients; those are the conversations in this
folder. It is then trained on data built from them, which produces the next adapter. Only turns after
the first 12 utterances of a conversation are used for training (`min_conv_length` = 12).
- **GRPO:** samples 8 replies per therapist turn and scores each. The update uses the group-relative
  advantage, with a KL penalty (β=0.01) to the policy the iteration started from.
- **PTO:** branches 8 replies per therapist turn. Best vs. worst becomes a DPO pair when their scores
  differ by more than τ=0.1, and the best reply extends the conversation (DPO β=0.1).

**Look-ahead K** only changes how a candidate reply is scored during training. With K=5, the reply is
extended by 5 more simulated turns (patient plus the current policy) and the grader scores that
extended conversation. With K=0, the reply is scored as it is. The conversations in this folder were
generated the same way for every arm.

## Iterations and model states

- **iteration 0** is the untrained base model (no adapter).
- **iteration N** (1–10) is the policy after N training iterations, i.e. `adapters/<arm>/iter_NN`.

Each arm has its own iteration-0 set. The four sets are independent samples of the same untrained
model talking to the same 96 patients.

## Patients (and the unshuffling)

There are 96 personas, one per combination of six traits:

- gender (2)
- age (27 / 61)
- problem (Smoking / Obesity)
- how long they have had it (FewMonths / ManyYears)
- whether they tried to solve it (Never / ManyTimes)
- cooperation (Low / High / StartLowAndChangesToHigh)

That gives 2 × 2 × 2 × 2 × 2 × 3 = 96. The name follows the gender (James / Emma). `personas.csv`
has each persona's traits and the full system prompt the patient model received.

**Why patient_id needed recovering.** At every iteration the trainer ran the 96 patients in a
different seeded random order and saved each file by position. So the raw `conversation_i.csv` is a
different patient in every iteration. Here that is undone: `patient_id` (0–95) is the canonical
persona everywhere, recovered by replaying the trainer's shuffle.

**Check.** In all 4,224 conversations, the patient's first utterance never contradicts the recovered
persona: no wrong name, wrong age or wrong problem. As a control, the raw file order contradicts it
in 963 of 1,152 checked conversations (4 arms × iterations 1, 5, 10 × 96). `data/conversations.csv`
keeps the original file in `source_file`.

## How a conversation is generated

- **Therapist.** Llama-3.2-1B plus the adapter. The base model has no chat tuning, so it is prompted
  in ChatML (`prompts/therapist_chat_template.jinja`) with
  `prompts/therapist_system_prompt.txt`, the same prompt for every patient. The first therapist
  utterance (T1) is always the fixed line in `prompts/therapist_opening_utterance.txt`; the model
  does not generate it.
- **Patient.** gpt-4o-mini-2024-07-18, with the persona prompt from `personas.csv`.
- **Sampling.** Therapist temperature 0.9, patient 0.7, at most 200 tokens per reply.
- **Ending.** Either side can end the session by writing "SESSION ENDED" and a condition number;
  `session_ended_by` and `session_ended_explanation` keep what was parsed, raw. Otherwise the
  conversation runs to the 50-utterance cap. If `session_ended_by` is blank and the conversation has
  fewer than 50 utterances, generation stopped early. One example: a therapist reply that was empty
  once the fake chat-turn markers were cut.
- **Sample size.** One conversation per patient per model state, so n = 96 per state.

## Scores

Two LLM graders read each full conversation and fill in every instrument. They are in separate files.
Compare within one grader; never average the two.

- **main: gpt-4o-mini-2024-07-18.** ⚠ This is the same model that produced the training reward, so
  under this grader Q1 and Q2 are the quantity the policy was trained to raise.
- **held-out: Claude Haiku 4.5** (`claude-haiku-4-5`). It was never used in training.

| instrument | item columns | headline column | range | direction |
|---|---|---|---|---|
| Q1: session satisfaction | `Q1_1`–`Q1_5` | `Q1_Mean` | 1–5 | higher better |
| Q2: working alliance / relational communication | `Q2_1`–`Q2_17` | `Q2_Mean` | 1–5 | higher better |
| Q1Q2: the training reward's axis | — | `Q1Q2_Mean` = mean(`Q1_Mean`, `Q2_Mean`) | 1–5 | higher better |
| WAI-SR: working alliance | `WAI1_…`–`WAI12_…` | `WAI_TotalMean` (+ Goal / Task / Bond means) | 1–5 | higher better |
| CSQ-8: client satisfaction | `CSQ1_…`–`CSQ8_…` | `CSQ8_Mean` | 1–4 | higher better |
| MI-SAT: satisfaction with the MI session | `MI1_…`–`MI6_…` | `MI_Mean` | 1–5 | higher better |
| MITI: MI treatment integrity | 4 globals + 7 behaviour counts | `MITI_GlobalMean` | 1–5 | higher better |
| PCT: patient change talk | 3 globals + change / sustain / neutral counts | `PCT_ChangeProp` = CT / (CT + ST) | 0–1 | higher better |
| MICI: MI-inconsistent therapist behaviour | severity + 6 behaviour counts | `MICI_Rate` = inconsistent behaviours per therapist turn | ≥ 0 | **lower better** |
| MIPROC: utterance-level process codes | one code per utterance (below) | `MIPROC_PctCR` = CR / (SR + CR), plus per-code counts and rates | 0–1 | higher better |

Instrument sources:
- **Q1 and Q2:** the LLM-evaluator questionnaires from Yosef et al., CLPsych 2024.
- **WAI-SR and CSQ-8:** standard self-report scales, filled in by the grader in the patient's voice.
- **MI-SAT:** an adapted MI satisfaction survey.
- **MITI 4.2:** the standard MI fidelity coding system.
- **PCT, MICI and MIPROC:** MITI-style coders written for this experiment.

Each transcript header shows the headline column of every instrument under both graders; `%CR` there
is `MIPROC_PctCR`.

## MIPROC utterance codes

Each grader gives one code per utterance. The codes are in two columns of `data/utterances.csv`, and
the transcripts show them as `[main/held-out]`.

**Therapist codes**

| code | meaning |
|---|---|
| `OQ` | open question |
| `CQ` | closed question |
| `SR` | simple reflection |
| `CR` | complex reflection (adds meaning: an inferred feeling, a double-sided reflection, continuing the paragraph) |
| `AF` | affirmation of a specific strength or effort (MITI definition) |
| `PRA` | non-specific praise or cheerleading ("I'm so proud of you"), kept separate from `AF` on purpose |
| `GI` | giving information / structuring |
| `PERS` | persuasion: advising, lecturing, warning |
| `SEEK` | seeking collaboration / emphasizing autonomy |
| `CONF` | confronting, directing or judging |
| `OTH` | other: greeting, filler, incoherent or degenerate text |

**Patient codes**

| code | meaning |
|---|---|
| `CT` | change talk |
| `ST` | sustain talk |
| `NEU` | neither |

## LoRA adapters

- **Files.** `adapters/<arm>/iter_NN/adapter_model.safetensors` plus `adapter_config.json`, for
  4 arms × 10 iterations = 40 adapters, stored in fp32.
- **LoRA settings.** r=16, α=16 (scale α/r = 1), dropout 0.05, on `q/k/v/o/gate/up/down_proj` in all
  16 layers of `meta-llama/Llama-3.2-1B`. That model is gated on Hugging Face, so loading the base
  weights needs access.
- **Tensor names and ΔW.** The keys are
  `base_model.model.model.layers.{L}.{self_attn|mlp}.{q,k,v,o,gate,up,down}_proj.lora_{A,B}.weight`.
  The change added to each base matrix is **ΔW = (α/r)·B·A**.
- **Each adapter is cumulative.** Iteration N loads adapter N−1 and keeps training it (no merge). So
  `iter_NN` is the whole change from the base model to that state, and the change made during
  iteration N alone is ΔW(iter N) − ΔW(iter N−1). Both GRPO's KL term and DPO's reference use the
  policy the iteration started from, not the base model. Nothing pulls the adapter back toward the
  base across iterations.
- **Compare B·A products, not A or B on their own.** The factorisation is not unique, so
  differences between the factors themselves mean nothing.

Per-layer change for the two GRPO conditions at iteration 10 (tested on these files; takes seconds
on CPU):

```python
import re
import pandas as pd
from safetensors.torch import load_file

SCALE = 16 / 16          # lora_alpha / r  (see adapter_config.json)

def delta_weights(path):
    """{(layer, module): dW} with dW = (alpha/r) * B @ A  -- the change added to the base weight."""
    sd = load_file(path)
    out = {}
    for k, A in sd.items():
        if k.endswith("lora_A.weight"):
            B = sd[k.replace("lora_A", "lora_B")]
            layer, module = re.search(r"layers\.(\d+)\.\w+\.(\w+)\.lora_A", k).groups()
            out[(int(layer), module)] = SCALE * (B.float() @ A.float())
    return out

k0 = delta_weights("adapters/GRPO_K0/iter_10/adapter_model.safetensors")
k5 = delta_weights("adapters/GRPO_K5/iter_10/adapter_model.safetensors")
rows = [{"layer": l, "module": m,
         "norm_dW_K0": k0[(l, m)].norm().item(),
         "norm_dW_K5": k5[(l, m)].norm().item(),
         "norm_K5_minus_K0": (k5[(l, m)] - k0[(l, m)]).norm().item()}
        for (l, m) in sorted(k0)]
df = pd.DataFrame(rows)
print(df.groupby("layer")[["norm_dW_K0", "norm_dW_K5", "norm_K5_minus_K0"]].sum())
```

To express a change relative to the size of the original weight (‖ΔW‖ / ‖W‖), load the base model's
weights too. To generate text with an adapter:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base, "adapters/GRPO_K5/iter_10")
```

## `data/` columns

- **`conversations.csv`** (4,224 rows): `arm, method, K, iteration, patient_id`, then:
  - `n_utterances, n_therapist, n_patient`
  - `session_ended_by, session_ended_explanation`
  - `source_file, source_file_index` (the original file and its position)
  - the persona traits
- **`utterances.csv`** (113,941 rows): the same keys, then:
  - `utt_index`: 0-based position in the conversation
  - `speaker`
  - `speaker_turn`: 1-based count within that speaker, the `T#` / `P#` of the transcripts
  - `text`
  - `miproc_code_main_gpt-4o-mini`, `miproc_code_heldout_claude-haiku-4-5`
- **`scores_*.csv`** (4,224 rows each): the same keys, then every item and summary column of every
  instrument. That includes the MIPROC per-code counts (`MIPROC_TH_*`, `MIPROC_PT_*`) and rates.
  The code sequences themselves are in `utterances.csv`.
- **`score_means_by_state.csv`**: the per-(arm, iteration) mean of each headline column, per grader.
  No confidence intervals; it is only for choosing what to read.

All CSVs are UTF-8 with a BOM, so Excel shows the quotation marks correctly.

## Gaps and what's not here

- **One missing coding.** The held-out grader has no MIPROC codes for one conversation (`PTO_K5`,
  iteration 3, `patient_id` 15); those cells are blank. Every other cell is filled.
- **Not included (available on request):**
  - the training-time data: GRPO's sampled replies with their rewards, and PTO's preference pairs
  - the repeatability re-scorings
  - two replicate generations of iteration 10 (`GRPO_K5`, `PTO_K0`)
