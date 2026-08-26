# Exp4_OpenStack — PTO vs GRPO on a fully open model stack

## What this is

A **side project** off the thesis (Lior Baruch, Reichman). Same science as
[Exp3](../Exp3_PTO_GRPO/) — iterative PTO vs iterative GRPO on a Llama-3.2-1B therapist doing
Motivational Interviewing, with a K-turn look-ahead lever — but **every LLM role is selectable**
and the defaults are **open models served locally**, so a full run costs **$0 in API**.

Exp3's binding constraint was API cost, and on 2026-08-20 it stopped being theoretical: the OpenAI
organization spend cap killed two Colab sessions outright and left GRPO_LA5 stalled mid-iteration-7.
Exp4 removes that constraint entirely. If the results are interesting it may become a thesis
chapter or a paper; until then it is exploratory and **shares no data and no score axis with Exp3**.

⚠ **Exp3 and Exp4 scores are NOT comparable.** Different grader ⇒ different axis. Compare within
Exp4 only. (The same warning Exp2↔Exp3 carries for a different reason.)

| | Exp3_PTO_GRPO | **Exp4_OpenStack** |
|---|---|---|
| Therapist | Llama-3.2-1B bf16 + LoRA | **same** (deliberately — the policy is not the variable) |
| Patient | `gpt-4o-mini-2024-07-18` | **`google/gemma-4-E4B-it`** (selectable; E2B = fallback) |
| Training oracle | `gpt-4o-mini-2024-07-18` | **`google/gemma-4-E4B-it`** (selectable) |
| Eval judge | gpt-4o-mini + Claude Haiku 4.5 | **`google/gemma-4-E4B-it`** (selectable; `judge=` partitions from day 1) |
| Serving | vendor APIs | **one local vLLM OpenAI-compatible server** (vLLM ≥ 0.19.1 for Gemma 4) |
| Training questionnaire | Q1+Q2 (fixed) | **selectable** (default Q1+Q2) |
| Logging | W&B + TensorBoard | **TensorBoard only** |
| Cost per arm | ~$25–120 API | **$0 API** (Colab GPU-h only) |

**The grader models, measured (HF API, 2026-08-26; both ungated, Apache 2.0 — no license
click-through needed):**

| | params (bf16) | checkpoint | vLLM `gpu_memory_utilization` on A100 40 GB |
|---|---|---|---|
| `google/gemma-4-E4B-it` (default) | 7.996B | **14.89 GiB** | **0.50** (20 GiB = weights + ~4–5 GiB KV) |
| `google/gemma-4-E2B-it` (fallback) | 5.123B | **9.54 GiB** | 0.35 (14 GiB = weights + ~3.5–4 GiB KV) |

Both are Gemma 4 (released 2026-07): per-layer embeddings, 128K context, configurable thinking
modes. ⚠ **vLLM loads the raw checkpoint — never budget on the "effective" parameter count.** The
"~3 GB" figure earlier revisions of this file carried was wrong by 3–5×; the numbers above are the
checkpoint bytes. E4B is the default because **the grader IS the instrument** — pick between them
by running the FULL `oracle_sanity` against both (Spearman + spread), not by size. Thinking is off
by default in Gemma 4 AND explicitly disabled per request (`enable_thinking: false` via
`chat_template_kwargs` — the key matches the official vLLM Gemma 4 recipe and the model card;
`smoke.py roles` still verifies no thinking tokens reach the wire).

## Relationship to Exp3

**Fresh rewrite — lessons, not code.** Exp3 is untouched; Exp4 adds files only. Three categories:

| | Files | Why |
|---|---|---|
| **Verbatim copies** | `code/questionnaires.py`, `code/system_prompts_builder.py` | The rubrics and the 96 V3 personas ARE the instrument. Copying them keeps the task identical so the model-stack swap is the only variable. Do not edit. |
| **Ported + adapted** | `code/roles.py` (from Exp3's), `code/core/timing.py` (from `_shared/timing.py`), `code/tools/vllm_serve.py` (from `eda/eda_analysis/scoring/local_server.py`), the ChatML template + `patch_generate` + checkpoint-walk in `code/core/policy.py` | Proven, small, and load-bearing. |
| **Rewritten lean** | everything else | Exp3's trainer is ~10.3k LOC and its EDA ~16k; Exp4 targets ~6k total. |

### The five Exp3 defects Exp4 fixes by construction

1. **Patient API calls had no timeout** → every `RoleBinding` carries `request_timeout`, and the
   shape is *short per-attempt timeout × many retries* (a long total budget freezes a sim, and one
   frozen sim shifts the mean AND std of its GRPO group of 8).
2. **Conversation files were named by the shuffled processing index**, so `conversation_3.csv` is a
   different persona each iteration → Exp3's EDA had to replay `Random(seed+k+1)` in every module.
   Exp4 names files by the **stable persona id** (`pers07.csv` is persona 7 in every iteration,
   forever) and stores `persona_id` as a CSV column.
3. **Per-process timing fields undercounted resumed iterations** (one iteration logged 14,501 s for
   7.7 h of work) → spawned 1,336 LOC of artifact-mtime forensics. Exp4 writes an **append-only
   `timing_sessions.jsonl`** per iteration from day 1.
4. **~50k single-row score CSVs** needed a parquet fold cache + manifest. Exp4 writes **one parquet
   per (judge, rep, metric, arm, model state)** — 96 rows each. No fold, no cache, no manifest.
5. **`run_metadata.json` was overwritten in place on resume**, restamping earlier iterations under
   changed knobs → Exp4 keeps `run_metadata.json` (current) **plus** append-only
   `run_metadata_history.jsonl` (one line per process).

## Layout

```
Exp4_OpenStack/
├── CLAUDE.md          this file — the spec AND the implementation contract
├── README.md          folder map + data-symlink recreation
├── code/
│   ├── questionnaires.py          VERBATIM from Exp3 — do not edit
│   ├── system_prompts_builder.py  VERBATIM from Exp3 — do not edit
│   ├── roles.py                   role→provider bindings + vLLM serve planning
│   ├── naming.py                  THE arm-name grammar (one regex, shared by trainers + EDA)
│   ├── core/                      the shared trainer layer
│   │   ├── concurrency.py         async primitives (loop-keyed semaphores, gpu_lock, _run_async)
│   │   ├── config.py              frozen config dataclasses
│   │   ├── runtime.py             host detect, auth, workspace root, import-order guard
│   │   ├── policy.py              tokenizer/ChatML, LoRA, patch_generate, checkpoints, resume
│   │   ├── conversations.py       ConversationState, patient calls, conv loop, transcripts, MCL
│   │   ├── lookahead.py           K-turn lock-step simulation
│   │   ├── oracle.py              schema-constrained scoring + aggregation
│   │   ├── reward.py              make_reward_fn (oracle ∘ lookahead) + GRPO group recording
│   │   ├── recorder.py            generations.jsonl capture
│   │   ├── timing.py              append-only per-phase session log
│   │   └── tb.py                  TensorBoard writers + post-hoc dashboard
│   ├── grpo/{grpo_trainer.py, train_grpo.ipynb}
│   ├── pto/{pto_trainer.py, train_pto.ipynb}
│   └── tools/{vllm_serve.py, oracle_sanity.py, smoke.py, generate_convs.py, fixtures/sanity/}
├── data/              GITIGNORED — three Google Drive symlinks (see README)
└── eda/               lean analysis package + notebooks + render driver
```

## Naming — the arm identity grammar

**One regex, one module** ([code/naming.py](code/naming.py)), imported by both the trainers (write
side) and the EDA (read side). `EXPERIMENT_NAME` is **computed, never typed** — which is why Exp3's
`assert_name_matches_roles` guard has no Exp4 equivalent: the failure it prevented cannot occur.

```
{GRPO|PTO}4_{QTAG}_LA{K}_MCL{N}_{G{G} | M{M}_PT{greedy|indep}}_O{otag}_Pat{ptag}

GRPO4_Q1Q2_LA5_MCL12_G8_Ogemma4E4B_Patgemma4E4B
PTO4_Q1Q2_LA0_MCL12_M8_PTgreedy_Ogemma4E4B_Patgemma4E4B
GRPO4_WAI_LA0_MCL12_G8_Ogpt4m_Patgemma4E4B          # oracle flipped to the OpenAI API
```

⚠ **The grammar can spell a SPLIT stack; the v1 trainers cannot run one.** Both trainers thread a
single async client through `run_one_iteration` and use it for the oracle *and* every patient call,
so `validate_config` (`core.config._roles_errors`) refuses a bundle whose oracle and patient differ
in `(provider, base_url)` — otherwise one of the two roles is silently sent to a server that does
not host its model, 404s per turn, and only after the base model has loaded. The third example
above is therefore a legal *name* that needs its patient moved to the same endpoint before it is a
runnable *arm*. An all-vendor stack (oracle **and** patient on the OpenAI API) is fine; so is the
all-open default. The judge is exempt — no trainer calls it; the EDA builds its own client.

- **Role tags are ALWAYS present** (unlike Exp3, where a suffix appeared only for non-default
  bindings — a subtlety that existed to protect 50k already-written CSVs. Exp4 has no legacy lake).
- `QTAG` ∈ `Q1Q2 | Q1 | Q2 | WAI | CSQ8 | MISAT | MITI` — derived from `QUESTIONNAIRE_IDS`.
  ⚠ **No underscores inside a token** (`MI_SAT` → `MISAT`), or the field structure the regex depends
  on breaks.
- Model tags come from `roles.model_tag` and are `[A-Za-z0-9]` only → every name is a legal Windows
  path segment and a legal TensorBoard logdir.
- The **therapist** is not encoded (fixed across all Exp4 arms; recorded in `run_metadata.json`).
  Changing it is a grammar-version bump, not a new token.

## Data layout

All three live under `Exp4_OpenStack/data/` as Google Drive directory symlinks (gitignored; the
schema here is the only record of their shape — same arrangement as Exp3).

```
data/
├── runs/<EXP_NAME>/
│   ├── run_metadata.json              current config (overwritten)
│   ├── run_metadata_history.jsonl     append-only, one line per process  ← Exp3 fix #5
│   └── iteration_<N>/
│       ├── adapter/                   presence ⟺ "iteration done"
│       ├── training/                  HF Trainer output_dir (checkpoint-*, tb_logs/)
│       ├── eda/generations.jsonl      per-branch capture
│       ├── pref_pairs/{pairs.csv,pairs_fingerprint.json,_progress.json}   PTO only — the
│       │                              sidecar records the config the pairs were BUILT under;
│       │                              the reload path compares it and warns on mismatch
│       ├── iteration_metadata.json
│       └── timing_sessions.jsonl      append-only per-phase timing  ← Exp3 fix #3
├── conversations/<EXP_NAME>/model_iter_<N>/pers<PID>.csv    PID = stable persona id 00–95  ← fix #2
└── eval_scores/judge=<tag>/rep=<r>/metric=<M>/<EXP_NAME>/model_iter_<N>.parquet   ← fix #4
```

`model_iter_<N>` is labelled by the **generating** policy: iteration `n` generates `model_iter_{n-1}`
and produces `iteration_n/adapter/`. A post-loop generate-only pass produces `model_iter_{N}`. So
`N` iterations yield `N+1` conversation folders. (Identical to Exp3 — see its
[iter↔model-state table](../CLAUDE.md).)

There is **no `oracle=<O>` path level** (Exp3 had one): the training oracle is inside `<EXP_NAME>`
because role tags are always encoded.

## Module contract

Signatures below are the contract between modules. Implementations may add helpers; they may not
change these shapes without updating this file.

### `code/roles.py`

```python
DEFAULT_ORACLE_MODEL  = "google/gemma-4-E4B-it"
DEFAULT_PATIENT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_JUDGE_MODEL   = "google/gemma-4-E4B-it"

@dataclass(frozen=True)
class RoleBinding:
    model: str
    provider: str = "openai_compat"        # openai_compat | openai | anthropic
    base_url: Optional[str] = None         # filled in by serve_roles() for open roles
    api_key_env: Optional[str] = None
    temperature: Optional[float] = None
    request_timeout: float = 90.0          # PER ATTEMPT  ← Exp3 fix #1
    max_retries: int = 8
    extra_body_json: Optional[str] = None  # JSON string (keeps the dataclass hashable/frozen)
    @property
    def tag(self) -> str: ...              # model_tag(self.model)
    @property
    def is_local(self) -> bool: ...        # provider == "openai_compat"
    @property
    def extra_body(self) -> Optional[dict]: ...   # parsed extra_body_json, {} -> None

@dataclass(frozen=True)
class ServeSpec:
    model: str
    port: int = 8000
    gpu_memory_utilization: float = 0.25   # default for tests/planning; notebooks pass the
                                           # model-derived value (E4B 0.50 / E2B 0.35 — § VRAM budget)
    max_model_len: int = 16384             # NOT an escape hatch — see § VRAM budget
    dtype: str = "bfloat16"
    extra_args: Tuple[str, ...] = ()

def model_tag(model_id: str) -> str            # google/gemma-4-E4B-it -> gemma4E4B ; [A-Za-z0-9] only
def thinking_off_extra_body() -> str           # '{"chat_template_kwargs": {"enable_thinking": false}}'
def make_binding(provider, model, *, base_url=None, disable_thinking=True, **kw) -> RoleBinding
def plan_servers(bindings: Dict[str, RoleBinding], *, base_port=8000, **spec_kw) -> List[ServeSpec]
def make_client(binding: RoleBinding, *, api_key: Optional[str] = None)
        # AsyncOpenAI/AsyncAnthropic, cached PER RUNNING EVENT LOOP (None outside a loop) —
        # pooled keep-alive connections cannot cross loops, and run_async spawns a fresh loop
        # per call, so a same-binding client from another loop is evicted on sight. The async
        # entry points (generation pass, look-ahead, oracle scoring, PTO build) re-resolve their
        # client via make_client INSIDE the running coroutine rather than reusing a handle built
        # elsewhere.
```

`plan_servers` **dedupes by model id**: patient + oracle + judge all on the same Gemma ⇒ exactly
**one** `ServeSpec`, because one vLLM server serves every role (the roles differ only in per-request
sampling params). Bindings whose provider is not `openai_compat` produce no spec.

⚠ `extra_body` is for `openai_compat` only — the OpenAI API 400s on unknown body keys.

### `code/naming.py`

```python
@dataclass(frozen=True)
class ArmInfo:
    method: str          # "GRPO" | "PTO"
    qtag: str            # "Q1Q2" | ...
    k: int
    mcl: int
    g: Optional[int]     # GRPO
    m: Optional[int]     # PTO
    mode: Optional[str]  # "greedy" | "indep" (PTO)
    oracle_tag: str
    patient_tag: str
    @property
    def label(self) -> str      # short display label, e.g. "GRPO_LA5"
    @property
    def experiment_name(self) -> str

QTAG_BY_IDS: Dict[FrozenSet[int], str]                 # {1,2}->Q1Q2, {3}->WAI, {4}->CSQ8, {6}->MISAT, {7}->MITI, {1}->Q1, {2}->Q2
def qtag_for(questionnaire_ids: Sequence[int]) -> str   # raises on an unmapped set
def build_experiment_name(method, questionnaire_ids, k, mcl, *, g=None, m=None, mode=None,
                          oracle_model: str, patient_model: str) -> str
def parse_experiment_name(name: str) -> ArmInfo         # raises ValueError on non-match
def model_state_label(n: int) -> str                    # "model_iter_0" ...
```

### `code/core/concurrency.py`

```python
class AsyncPrimitives:
    """Semaphores + GPU lock, created lazily and keyed by id(running loop).

    TRL may run the reward coroutine on a different event loop than the notebook's, and
    Python >= 3.10 raises when an asyncio primitive crosses loops. Stale loops are evicted.
    """
    def __init__(self, *, oracle_concurrency: int, patient_concurrency: int): ...
    def oracle_sem(self) -> asyncio.Semaphore: ...
    def patient_sem(self) -> asyncio.Semaphore: ...
    def gpu_lock(self) -> asyncio.Lock: ...

def run_async(coro):
    """Run *coro* to completion from sync code, including inside a live Jupyter loop.

    When a loop is already running, spawn a daemon thread with its own fresh loop and join.
    Deliberately NOT nest_asyncio (broken on py>=3.13). CUDA is per-process, so torch calls
    from that thread are fine.
    """
```

⚠ **`gpu_lock` is held only across therapist `generate`, never across a patient `await`.** That
invariant is what lets the look-ahead API calls overlap with nothing blocking the GPU.

### `code/core/policy.py`

```python
CHATML_TEMPLATE: str                 # verbatim from Exp3 — the base Llama has no chat template
STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]
ADAPTER_FILES    = ("adapter_model.safetensors", "adapter_config.json")
HF_TRAINER_FILES = ADAPTER_FILES + ("trainer_state.json",)

def setup_tokenizer(tokenizer_id: str, padding_side: str = "left")
def setup_base_model(base_model_id: str, *, use_4bit: bool = False)
def attach_lora(model, *, r, alpha, dropout, target_modules)
def patch_generate(model, tokenizer)          # idempotent; injects tokenizer= for stop_strings
def clean_completion(text: Optional[str]) -> str          # cut at first ChatML marker; "" == degenerate
def generate_therapist_batch(model, tokenizer, batch_messages, *, max_tokens, temperature,
                             max_input_tokens, stop_strings) -> Tuple[Optional[List[str]], Optional[str]]
        # (responses, None) | (None, "oom") | (None, "runtime_error") — never raises on OOM

def list_iteration_checkpoints(run_dir) -> List[Tuple[int, str]]
        # "iteration done" ⟺ adapter/ holds BOTH ADAPTER_FILES — dir presence alone is a torn
        # save and is treated as INCOMPLETE (warns; resolve_start_state then resumes case B)
def get_latest_iteration(run_dir) -> int
def validate_iteration_checkpoint(iteration_dir) -> bool
def get_latest_valid_hf_checkpoint(training_dir) -> Optional[str]
def resolve_start_state(run_dir, base_policy, tokenizer) -> Tuple[int, Any, Optional[str]]
        # case B (mid-training crash) returns the ITERATION-START policy (prev adapter, or the
        # bare base for an iteration-1 resume) + the valid checkpoint path: both TRL trainers
        # snapshot the handed-in policy as their frozen reference (DPO copies default->"ref" and
        # precomputes ref logps INSIDE __init__; GRPO snapshots the same way) BEFORE
        # train(resume_from_checkpoint=...) restores the mid-training weights — loading the
        # checkpoint here would silently re-anchor the KL/DPO reference to the crash point
def compute_cumulative_step_offset(run_dir) -> int
```

⚠ **`patch_generate` must be re-applied after every re-wrap** — `PeftModel.from_pretrained` and
both TRL trainers hand back a fresh `generate`. Call it at base load, in `resolve_start_state`, and
on both sides of `trainer.train()`.

⚠ **The therapist is a BASE model with a hand-written ChatML template.** `<|im_start|>` /
`<|im_end|>` are ordinary BPE pieces, not special tokens, so the model happily writes both speakers.
The whole anti-degeneracy stack is load-bearing: `STOP_STRINGS` at every decode site +
`clean_completion` + `patch_generate` + GRPO's `generation_kwargs` + `REWARD_FLOOR`.

### `code/core/conversations.py`

```python
@dataclass
class ConversationState:
    persona_id: int                 # 0..95, STABLE — index into generate_all_permutations() order
    turns: List[Dict]               # [{"role": "therapist"|"patient", "content": str}, ...]
    messages_therapist: List[Dict]
    messages_patient: List[Dict]
    session_ended_by: str = ""
    session_ended_explanation: str = ""
    failed: bool = False

SESSION_END_KEYWORD = "SESSION ENDED"     # both system prompts instruct the model to emit it

def save_conversation_csv(state, save_dir) -> str       # pers{persona_id:02d}.csv
def load_conversation_csv(path) -> ConversationState
def format_conversation_for_oracle(messages_or_turns) -> str
        # "[THERAPIST]: ...\n\n[PATIENT]: ..."   — system dropped
def parse_transcript_to_messages(transcript, sp_therapist, sp_patient) -> Tuple[List[Dict], List[Dict]]
        # inverse of the above; UNLABELLED segments are CONTINUATIONS of the previous labelled turn
        # (turn content can itself contain "\n\n")
def turns_to_messages(turns, system_prompt) -> List[Dict]
def turns_to_patient_messages(turns, system_prompt) -> List[Dict]

async def generate_patient_response(client, binding: RoleBinding, messages, sem, *, max_tokens,
                                    temperature, seed=None) -> str
        # asyncio.wait_for(..., binding.request_timeout) PER ATTEMPT; binding.max_retries attempts
        # with exponential backoff slept OUTSIDE the semaphore; binding.extra_body passed through
async def generate_patient_batch(client, binding, batch_messages, sem, **kw) -> List
        # asyncio.gather(return_exceptions=True) — per-conversation failures come back as exceptions

async def conversation_loop_batch(...) -> Tuple[List[ConversationState], Optional[str], List[int]]
def generate_all_conversations(..., allow_partial=False) -> List[ConversationState]
        # resumes from per-persona CSVs already on disk (written ATOMICALLY: temp + os.replace,
        # so a preemption mid-write can't leave a shorter-but-parseable truncation that the
        # resume then treats as a complete conversation); bounded no-progress retries; an OOM
        # batch HALVES the batch size stickily and re-slices (mirrors the look-ahead);
        # RAISES RuntimeError when personas are still missing after the retry bound unless
        # allow_partial=True — the failures correlate with length/difficulty, so a partial set
        # feeding training or eval is biased missingness on the headline metric;
        # gc.collect()+empty_cache() BETWEEN batches (not cosmetic — the allocator high-water
        # mark grows otherwise) and prints a `vram <N>G` field per batch line

def build_truncated_training_prompt(turns, system_prompt, tokenizer, max_prompt_tokens,
                                    truncation_mode="drop_oldest") -> Optional[str]
        # None when even one most-recent turn exceeds budget -> caller SKIPS the pair
def extract_prompts_from_conversations(states, system_prompt, tokenizer, *, min_conv_length,
                                       max_prompt_tokens, permutations) -> List[Dict]
        # one sample after each patient turn whose conv-so-far has >= min_conv_length utterances
        # keys: prompt, transcript, conversation_id, persona_id, patient_system_prompt
```

⚠ **The transcript format is load-bearing.** Look-ahead reconstructs message lists from the
transcript string and recovers its tail by **exact string slicing**
(`seed = f"{transcript}\n\n[THERAPIST]: {completion}"`; `tail = ext[len(seed):]`). Changing the
labels or the `"\n\n"` joiner breaks look-ahead silently.

### `code/core/lookahead.py`

```python
@dataclass(frozen=True)
class LookaheadConfig:
    k: int = 0                              # 0 disables look-ahead entirely
    temperature_therapist: float = 0.9
    temperature_patient: float = 0.7
    max_tokens: int = 200
    max_input_tokens: int = 2048
    patient_binding: RoleBinding = ...
    stop_strings: Tuple[str, ...] = ...
    sub_batch_size: Optional[int] = None    # None = one padded generate over all active sims

async def simulate_lookahead_batch(model, tokenizer, client, cfg: LookaheadConfig,
                                   primitives, transcripts, completions,
                                   sp_therapist, sp_patient_list) -> List[LookaheadResult]
```

Advances all B sims **in lock-step**: one padded batched `model.generate` per simulated therapist
turn, then one batched patient round. OOM halves the sub-batch and **the halving is sticky** across
steps; at sub-batch 1 an OOM freezes that single sim. A non-OOM runtime error freezes the chunk and
advances (deliberately unlike the conversation loop, which aborts). Toggles `model.eval()` +
`use_cache=True` and restores both in `finally` — this runs while the policy is in `train()`
mid-optimizer-step.

`LookaheadResult` carries at least `extended_transcript`, `tail`, `realized_turns`, `ended_early`.

### `code/core/oracle.py`

```python
REWARD_FLOOR = 0.0

@dataclass(frozen=True)
class OracleConfig:
    binding: RoleBinding                    # MANDATORY (Exp3's None-fallback is gone)
    questionnaire_ids: Tuple[int, ...] = (1, 2)
    eval_temperature: float = 0.0
    max_tokens: int = 256                   # was hardcoded in Exp3
    max_retries: int = 3
    request_timeout: float = 120.0
    max_concurrency: int = 64
    min_success_ratio: float = 0.5

def response_format_for(binding, schema, name) -> dict
        # THE provider-quirk shim. OpenAI: {"type":"json_schema", "json_schema":{..., "strict":True}}
        # openai_compat: same shape, `strict` stripped if the pinned vLLM rejects it.
async def get_evaluation_json(client, cfg, primitives, conversation_text, questionnaire_id)
        -> Tuple[Optional[dict], int, int]      # (data|None, n_questions, attempts)
async def score_conversation(client, cfg, primitives, conversation_text) -> dict
        # {"score": float|None, "sub_scores": {qid: mean}, "success": bool, "attempts": int}
```

The validation ladder (kept from Exp3, unchanged): the returned `questionnaire_id` must echo the
request, `len(scores)` must equal the item count, every score must be an `int` inside
`[scale_min, scale_max]`. Then `mean_score = mean(scores)` per questionnaire, and the reward is the
**unweighted mean across questionnaires** — so Q1 (5 items) and Q2 (17 items) carry equal weight.
⚠ **If any single questionnaire fails, that candidate's `score` is `None`** — "not graded", never
"graded badly". ⚠ **`None` must never reach TRL.** The pinned `trl==1.4.0` maps it to NaN
(`grpo_trainer.py:1259`) and then reduces with `nansum` (`:2145`), which turns NaN into **0.0** —
so an ungraded candidate would be optimised as the worst possible completion and would re-scale the
advantages of all G−1 siblings. `core.reward.rewards_for_trl` therefore substitutes the candidate's
**group mean** (advantage ≈ 0, group mean unchanged) before the vector is returned, and records the
substitution as `reward_used` on the EDA candidate. PTO keeps the raw `None` and excludes such a
candidate from the τ comparison.

⚠ **The rubric-first prompt layout in `questionnaires.py` is load-bearing.** Fixed instructions +
rubric FIRST, transcript LAST — that is what vLLM's prefix caching (and OpenAI's, on an API arm)
reuses across every call. Never move the transcript ahead of the rubric.

### `code/core/reward.py`

```python
def make_reward_fn(model, tokenizer, client, oracle_cfg, la_cfg, primitives, *,
                   recorder=None, sp_therapist=None) -> Callable
def rewards_for_trl(candidates, num_generations) -> List[Optional[float]]
        # per-group repair of an ungraded candidate (see the oracle section's None warning)
```

The TRL reward callable: cleans completions, floors degenerate ones to `REWARD_FLOOR`, runs
look-ahead when `la_cfg.k > 0`, scores with the oracle, records the group to the recorder.
⚠ **Raises `RuntimeError` when the batch success rate falls below `oracle_cfg.min_success_ratio`** —
training on a biased subset is worse than stopping. TRL passes `transcript` / `persona_id` /
`patient_system_prompt` through as `**kwargs` (needs `remove_unused_columns=False`).

⚠ TRL hands completions back as **G-consecutive blocks per prompt**; reshape `(-1, G)` to recover
groups, and skip the record with a warning when `n % G != 0` rather than mis-grouping.

### `code/core/recorder.py`

One JSONL row per branch — prefix stored **once**, candidates nested:

```json
{"phase": "group|tree|independent", "iteration": 3, "conversation_id": "...", "persona_id": 7,
 "branch_id": 0, "eval_pass": false, "prefix": "...",
 "candidates": [{"completion": "...", "score": 3.4, "sub_scores": {"1": 3.0, "2": 3.8},
                 "lookahead": {"tail": "...", "realized_turns": 5, "ended_early": false}}],
 "chosen_idx": 2}
```

`score` is the RAW grader result (`null` = the oracle failed). A candidate also carries
`reward_used` **only when the number GRPO optimised differs from it** — i.e. the group-mean
substitution above. `group_mean` / `group_std` are TRL's own reduction of that vector (surviving
`null` at 0.0, **sample** SD, ddof=1), so `sign(reward_used − group_mean)` reconstructs the sign of
the advantage.

Reconstruct a scored text as `prefix + "\n\n[THERAPIST]: " + completion + (tail or "")`.
`snapshot_to(path)` / `load_from(path)` support checkpoint-resume (HF fast-forwards skipped batches
**without re-invoking the reward fn**, so the recorder must be restored from the checkpoint).

⚠ **`branch_id` is trunk DEPTH for PTO, not a unique id** — it repeats across conversations. Any
per-branch aggregation must key on `(conversation_id, branch_id)`.

⚠ **`eval_pass` is written on EVERY row** (never omitted). With a GRPO eval split, TRL calls the
reward function during `evaluate()` too — held-out prompts, policy in eval mode, no gradient — and
those groups land in the same `generations.jsonl`. `EDARecorder.aggregate()` reports the
gradient-bearing rows under the existing keys and the held-out half under `eda/eval_*`; the EDA's
`load_generations` returns the flag as a column. Anything that pools the two answers a different
question at a blend ratio nobody chose.

### `code/core/timing.py`

Ported from Exp3 `_shared/timing.py`, one change: `PHASE_KEYS` gains an eval-generation phase.

```python
PHASE_KEYS            = ("generation_s", "pref_pair_s", "training_s", "eval_gen_s")
PRODUCTION_PHASE_KEYS = ("generation_s", "pref_pair_s", "training_s")   # the COST axis
def log_session(iter_dir, *, generation_s=0.0, training_s=0.0, pref_pair_s=0.0,
                eval_gen_s=0.0, started_at=None, note="") -> dict
def cumulative_seconds(iter_dir) -> Dict[str, float]
        # + total_s, production_s, n_sessions, n_sessions_production
def metadata_fields(iter_dir) -> Dict[str, float]         # cumulative_* to splat into metadata
```

**Phases log themselves AS THEY COMPLETE** — one line after generation, one after the preference
build (PTO), one after training — so a Colab preemption during training still leaves the finished
generation phase on the cost record (the Exp3 undercount this module exists to fix). Every line
carries a per-process token (`host:pid:start`), and `n_sessions` / `n_sessions_production` count
distinct PROCESSES, not lines, so per-phase logging does not inflate the counters.

`n_sessions_production > 1` ⟺ the iteration was resumed ⟺ any per-process number for it is wrong.
⚠ **Not `n_sessions > 1`.** The post-loop final-eval pass logs an `eval_gen_s`-only session against
`iteration_{N}` (and every `tools/generate_convs.py` repair appends another), so the raw session
count reports the last iteration of *every healthy arm* as resumed.

⚠ **Bill compute on `production_s`, never `total_s`.** State `j`'s conversations are generated at
the start of iteration `j+1`, so its measurement falls outside the `1..j` cumulative sum — for
every state but the last, whose eval pass has no next iteration and is logged against
`iteration_N`. Summing `total_s` prices exactly one point per arm — the endpoint every budget sweep
is read at — under a different rule, shifted right by a whole generation pass.

### `code/tools/vllm_serve.py`

```python
@dataclass
class ServerHandle:
    model: str; base_url: str; process: Optional[subprocess.Popen]; log_path: Optional[str]; spec: ServeSpec
    def stop(self, timeout: float = 30.0) -> None
    def tail_log(self, n: int = 40) -> str
    def is_alive(self) -> bool

def wait_until_ready(base_url, *, timeout=900.0, process=None, poll_seconds=3.0) -> None
def start_server(spec: ServeSpec, *, log_dir=None, timeout=900.0, executable="vllm") -> ServerHandle
def adopt_if_running(spec: ServeSpec) -> Optional[ServerHandle]     # idempotence
def serve_roles(bindings: Dict[str, RoleBinding], *, base_port=8000, **spec_kw)
        -> Tuple[Dict[str, RoleBinding], Dict[str, ServerHandle]]
        # plan -> adopt-or-start each -> return bindings with base_url filled in
def ensure_alive(handle: ServerHandle, *, max_restarts: int = 3) -> ServerHandle
def report_weights_gib(handle) -> Optional[float]      # parsed from the vLLM startup log
```

`serve_roles` is **idempotent**: a healthy server already on the port serving the right model is
adopted, not duplicated. `ensure_alive` is called at every phase boundary and from the retry path
on a burst of connection errors.

⚠ `gpu_memory_utilization` is a **pre-allocation, not a growing ceiling**. Sharing the card with a
live trainer wants it LOW (0.25) and the server started **FIRST**, because training memory is the
spiky side.

### Notebook cell-order contract (both trainers)

1. flat globals (cell 1) → 2. runtime detect + auth → 3. **`serve_roles()` — before any torch
import** → 4. `import trl` **then** torch/model build → 5. visible orchestration loop.

⚠ Step 4's order is not stylistic, and there are **two** native-init conflicts on the local
Blackwell card (sm_120), both with the same signature — exit 139, a Windows access violation, no
Python traceback, nothing to catch:

| order | result |
|---|---|
| `import torch, trl` | segfault (CUDA init) |
| `import torch, datasets` | segfault (inside `pyarrow.dataset`) |
| `import trl, torch, datasets` | segfault (pyarrow) |
| **`import trl, datasets, torch`** | **OK** |

So the safe order is **trl → datasets → torch**. Both are measured, not inferred.

⚠ **The trainers were only accidentally safe.** They pull pandas — and therefore pyarrow — in
through `core.*` before their own `import torch`, so the pyarrow conflict never fired. That is an
accident of import order that survives exactly until someone reorders those lines;
`tools/smoke.py` had no such accident and segfaulted before reaching its first check. Both are now
explicit, and `core.runtime.assert_import_order()` asserts **both** pairs (it inspects
`sys.modules` and uses `find_spec`, so it never imports anything itself). Colab is unaffected by
either.

## VRAM budget

**Colab A100 40 GB** (the primary target). Derived from the MEASURED checkpoint sizes (HF API,
2026-08-26): E4B **14.89 GiB** bf16, E2B **9.54 GiB** — the "~3 GB" figure earlier revisions
carried was wrong by 3–5×, and at the old `util 0.25` (10 GiB) the E4B server could not even hold
its weights:

| | budget | note |
|---|---|---|
| vLLM server (started first) | **E4B: ~20 GB** (`--gpu-memory-utilization 0.50`) · E2B: ~14 GB (0.35) | weights (14.89 / 9.54 GiB) + ~4–5 GiB KV pool + vLLM overhead. **`--max-model-len 16384`** (see below). Prefix caching on. Verify the measured weights line at the Phase 1 gate. |

⚠ **`max_model_len` must be 16384, not 8192.** Measured against the 192 real Exp3 PTO_LA0
transcripts (o200k tokenizer, rubric + transcript, the actual oracle prompt): the Q2 rubric alone
is 1,508 tokens, and full Q2 prompts run median 3,794 / p95 6,524 / **max 10,042**. At 8192,
**2.1% of Q2 prompts and 1.0% of Q1 prompts would not fit**; at 16384 none do, with ~60% headroom.

That 2% is not a rounding error, it is a **biased-missingness** hazard on the headline metric: the
prompts that overflow are the LONGEST conversations, and Exp3 measured session length varying *by
arm and by K* (PTO K=5 longer at iteration 10, GRPO K=5 shorter at 5). An arm-dependent dropout
rate on Q1+Q2 would bias the very contrast the experiment exists to measure — and it would do it
silently, because an unscoreable conversation is simply absent, not an error.

The memory cost is near zero: the KV pool is sized by `gpu_memory_utilization`, and `max_model_len`
caps one sequence rather than multiplying the pool. It lowers the theoretical concurrency ceiling,
but real concurrency is governed by actual prompt lengths (median ~3.8k), not the cap.

Two independent tokenizers agree on the figures above (o200k: Q2 max 10,042, 4/192 over 8192;
Llama-3.2: Q2 max 10,279, 5/192 over 8192; **neither has a single prompt over 16,384**).
⚠ **But the transcripts measured were written by Exp3's `gpt-4o-mini` patient.** Exp4's Gemma
patient may write longer turns, which would push more prompts up. 16384 absorbs a ~60% increase
over the observed maximum; **re-run this measurement on real Exp4 conversations at the Phase 2
gate** and raise the cap if the margin has eaten in. `tools/oracle_sanity.py` is the natural place
to report the realised prompt-length distribution.
| Therapist training | **E4B: ~19 GB** · E2B: ~25 GB | 1B bf16 + LoRA/optimizer + GRPO 128-completion generate + look-ahead at `sub_batch=64` + DPO's full-sequence 128k-vocab logits spike. The ~19 GB envelope under E4B is UNMEASURED on A100 — the auto sub-batch halving and `CONVERSATION_BATCH_SIZE` are the levers if it does not fit. |
| Headroom | ~1 GB | two CUDA contexts |

Escape hatches in order: switch grader E4B→E2B (frees ~6 GB; a science change — new arm name),
look-ahead sub-batch halving (automatic and sticky), `CONVERSATION_BATCH_SIZE` down.
⚠ **`max_model_len` is NOT an escape hatch** — lowering it below 16384 reintroduces the
biased-missingness hazard above. ⚠ Nor is `gpu_memory_utilization` below the weights + a usable
KV pool: the server either fails to start (no KV memory) or serves with near-zero concurrency.
⚠ Verify the real weights figure from the vLLM startup log at the Phase 1 gate rather than
trusting the arithmetic.

**Local RTX 5070 Ti (12 GB)** is for smoke tests and generate-only passes ONLY. ⚠ **An over-budget
VRAM request REBOOTS the machine — it does not raise `OutOfMemoryError`.** `tools/smoke.py` does the
arithmetic (weights + ~1.1 GB per concurrent conversation + the vLLM pre-allocation) and **refuses
before any CUDA allocation**. Conversation batch ≤ 2 when a server is also resident.

## The oracle-sanity gate

An open-weights grader fails in two ways, and only one of them is loud:

1. It ignores the schema or returns the wrong number of item scores → caught by validation, surfaces
   as retries and then as *biased missingness*.
2. **It honours the schema perfectly and returns degenerate scores** — every item a 4, near-zero
   variance across conversations. That parses, writes valid parquet, and produces a grader that
   cannot tell any two arms apart. Nothing downstream flags it; the contrast tables just come back
   ~0 and look like a finding.

[tools/oracle_sanity.py](code/tools/oracle_sanity.py) gates both against a committed fixture of
Exp3 transcripts spanning the quality range, with frozen `gpt-4o-mini` reference scores.

- **Hard gates** (block the run): 100% schema-valid on Q1+Q2; pooled per-conversation SD not
  degenerate.
- **Soft** (reported): Spearman rank agreement vs the reference, mean level offset.

The trainer notebooks run the **full 12-transcript gate** before iteration 1 whenever the oracle
is local (`--quick` only when a paid vendor API is the grader — a local call costs nothing, and
only the full gate enforces the degenerate-SD check at n ≥ 5), and archive the report next to
`run_metadata.json`. Run the full report once per candidate grader (E4B and E2B) before choosing
the arm's oracle.

## Hyperparameters (matched across methods, as in Exp3)

`MCL=12`, K ∈ {0, 5}, `NUM_CONVERSATIONS_PER_ITER=96`, PTO's `M` = GRPO's `G` = 8, matched
generation temperatures. `DPO_BETA=0.1` is the DPO loss temperature, **not** GRPO's KL β.

**`EPOCHS_PER_ITERATION=1`, matched — and 1 is the only value at which the match is real.** A GRPO
"epoch" re-SAMPLES G fresh completions per prompt and re-grades them all (2 epochs = twice the
reward-side work, the second pass on partially-updated weights), while a DPO epoch re-treads the
SAME fixed pairs. Exp3 ran both at 2, which quietly gave GRPO double the data generation per
iteration; Exp4 sets 1 so "one pass over data produced by this iteration's policy" holds for both
methods. Raise `NUM_ITERATIONS`, never epochs, for more updates.

⚠ **GRPO's `gradient_accumulation_steps=2` exists for the prompts/step match, not gradient
scale.** On the pinned trl 1.4.0, `gas` changes are gradient-scale-neutral: trl bypasses
transformers' `training_step` scaling (non-None `compute_loss_func` sentinel, installed trl
`grpo_trainer.py` ~:652–657) and divides the loss exactly once by
`current_gradient_accumulation_steps` (~:2351–2352). The earlier "net scale is 1/gas², halving gas
doubles the gradient" claim was measured on Exp3's stack and is FALSE here — re-verify those two
line references on any trl bump. `per_device_train_batch_size` counts **completions**, not
prompts: unique prompts/step = `(64/8) × 2 = 16`, matched to PTO's 16 pairs.

⚠ **PTO must pre-cap its DPO prompt.** TRL 1.4.0's `DPOConfig` dropped `max_prompt_length` and caps
prompt+completion with one `max_length` under `truncation_mode='keep_start'` — which slices the
**response** off. Hence `build_truncated_training_prompt` + `max_length = max_prompt +
max_completion + DPO_FRAMING_HEADROOM_TOKENS`. The headroom is load-bearing, not slack: TRL frames
both halves after the pre-cap has measured them (BOS prepended by its `processing_class(text=...)`
call, EOS appended to `chosen`/`rejected` by `add_eos`), so the two configured numbers alone are
two tokens short of the worst case and `keep_start` would still bite the longest pairs.

## Vocabulary

PTO is the framework; DPO is the loss. GRPO has no preference data — only prompts. **Never** call
GRPO data "pref data".

## Status

**Code complete and locally gated. Nothing has been trained; no `data/` exists yet.**

| Phase | Gate | State |
|---|---|---|
| 0 Scaffold | `smoke.py naming` — arm names round-trip through the parser | ✅ 24 checks |
| — | `smoke.py config` · `convs` · `vram` | ✅ 26 · 27 · 7 checks |
| — | `smoke.py stopgen` · `dpo` · `grpo` — real TRL steps on the local 12 GB card | ✅ 3 · 4 · 3 checks |
| 5 EDA | `_selfcheck --fast`; every family renders on an empty lake | ✅ 14 passed, 0 failed |
| 2 Oracle path | request carries a real `json_schema`; validation ladder accepts a good answer, **rejects a short array and prose**; aggregation is the unweighted mean across rubrics | ✅ vs `tools/fake_oracle_server.py` |
| 2 Sanity gate | passes a healthy grader, **fails a degenerate one** (exit 1) | ✅ both directions |
| 2 Generation | base policy + patient endpoint → `pers<PID>.csv` → reader round-trip → oracle score | ✅ full loop, local GPU |
| 1 Serving | `smoke.py roles` on Colab — chat + json_schema per binding, **no thinking tokens**, kill→restart, real weights GiB | ⬜ needs Colab + vLLM |
| 2 Real grader | 96-conv base pass vs the real Gemma patient; full `oracle_sanity` against the real Gemma oracle | ⬜ |
| 3 GRPO | mini-arm trains, resumes mid-iteration, `generations.jsonl` valid, prompts/step = 16 | ⬜ |
| 4 PTO | mini-arm trains; `pairs.csv` / `_progress.json` resume semantics verified | ⬜ |
| 6 First real arm | full GRPO K=0 arm on Colab, $0 API | ⬜ |

Everything runnable without Colab is green: **94 smoke checks** (GPU parts included) plus the EDA
self-check. `dpo` runs a real `DPOTrainer` and `grpo` a real `GRPOTrainer` step whose completion
lengths come back well under the cap, which is the anti-degeneracy stack working rather than merely
wired up.

**What [tools/fake_oracle_server.py](code/tools/fake_oracle_server.py) buys.** It is a test double
for the *endpoint*, so the plumbing between Exp4 and the wire is verifiable with no vLLM, no Colab
and no GPU for the oracle half. Proven against it: the request really does carry a `json_schema`
with the right item count and bounds; the thinking-off `extra_body` reaches the wire; the
validation ladder rejects a short `scores` array and rejects prose instead of coercing either into
a number; and the reward is the unweighted mean across rubrics. Then, with the same double standing
in as the patient and the real Llama on the local card, a generate-only pass produced
`pers<PID>.csv` files that round-trip through the reader into the oracle transcript format and
score — the whole loop, end to end.

⚠ **This proves the plumbing, not the grader.** A healthy-shaped double says nothing about whether
Gemma-4-E2B can actually measure MI quality. That is what the real `oracle_sanity` run on Colab is
for, and it remains the gate that must pass before any arm.

The reason the double is worth keeping: a *real* grader that happens to be healthy cannot tell you
whether the degeneracy gate would have caught a bad one. Pointing `--policy degenerate` at it is
the only way to test the gate itself, and that check now runs in about a second.

### The 2026-08-26 audit round (the re-run the previous session asked for)

Six read-only auditors + one independent skeptic per finding, run to completion this time:
**9 confirmed findings (0 refuted), all applied**, plus 15 desk-reviewed findings applied or
documented. The headline classes: per-phase timing logging (a preempted process now leaves its
finished phases on the cost record); the mid-training-resume reference bug (both TRL trainers
snapshot the handed-in policy as their frozen KL/DPO reference *in* `__init__`, so
`resolve_start_state` case B now returns iteration-start weights); reload-only generation on
mid-training resume (HF fast-forwards batches positionally, so the dataset must match the crashed
process's exactly); sticky OOM halving + a completeness raise in the conversation loop (no more
silent biased subsets); the loop-keyed client cache (pooled keep-alive connections cannot cross
event loops — measured: every parked connection poisons exactly one call on the next loop); EDA
display-label disambiguation (a quicktest arm can no longer merge into the real arm's figures);
the adapter↔iteration guard in the repair tool; file-validated "iteration done"; atomic
conversation CSVs; the `pairs.csv` fingerprint sidecar; and the corrected VRAM budget above (the
"~3 GB Gemma weights" premise was wrong by 3–5×). The "1/gas²" gradient-scale claim was verified
FALSE on the pinned trl 1.4.0 and rewritten at all its sites.

### Next session — start here

1. **Colab, in this order** — do not jump to a full arm:
   `smoke.py roles` → `oracle_sanity` (full) against **both** E4B and E2B (choose the grader by
   Spearman + spread) → a 2-iteration mini-arm (`QUICK_TEST=True`, lands in a disjoint folder) →
   a real arm (GRPO K=0 first).
2. **Before any of that**: push `code/` to Drive (additively), add the `huggingface` Colab secret
   (Llama-3.2-1B is gated; **Gemma 4 is NOT** — Apache 2.0, no click-through), and uncomment the
   install cell — **vLLM first** (Gemma 4 needs ≥ 0.19.1), then the pinned training stack, then
   restart the kernel.
3. At the Phase 1 gate, check the **measured weights line** against 14.89 GiB (E4B) / 9.54 GiB
   (E2B), and at Phase 2 re-run the oracle-prompt-length measurement on real Exp4 conversations
   (the 16384 cap was measured on Exp3's gpt-4o-mini patient).

⚠ **What is still unverified because nothing has run on Colab:** the vLLM build's behaviour
(`smoke.py roles` gates serving, thinking-off on the wire, and `json_schema` handling — the
`enable_thinking` key now matches the official recipe and thinking is off by default, so this is
belt-and-braces rather than a coin flip); the E4B trainer envelope (~19 GB alongside `util 0.50`
is arithmetic, not a measurement); and whether either Gemma actually measures MI quality — a
grader can honour the schema perfectly and return near-constant scores, which is exactly what
`oracle_sanity`'s degeneracy gate exists for.
