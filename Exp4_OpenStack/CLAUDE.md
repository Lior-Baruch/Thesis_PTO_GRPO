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
| Therapist | Llama-3.2-1B bf16 + LoRA | **selectable variant, tagged in the arm name** — `Llama-3.2-1B-Instruct` (default, `_ThL1Bi`) or the base `Llama-3.2-1B` (`_ThL1B`); same 1B family + LoRA either way |
| Patient | `gpt-4o-mini-2024-07-18` | **`google/gemma-4-E4B-it`** (selectable; E2B = fallback) |
| Training oracle | `gpt-4o-mini-2024-07-18` | **`google/gemma-4-E4B-it`** (selectable) |
| Eval judge | gpt-4o-mini + Claude Haiku 4.5 | **`google/gemma-4-E4B-it`** (selectable; `judge=` partitions from day 1) |
| Serving | vendor APIs | **one local vLLM OpenAI-compatible server** (vLLM ≥ 0.19.1 for Gemma 4) |
| Training questionnaire | Q1+Q2 (fixed) | **selectable** (default Q1+Q2) |
| Logging | W&B + TensorBoard | **TensorBoard only** |
| Cost per arm | ~$25–120 API | **$0 API** — GPU-hours only. Target card: **Colab A100 80 GB**; 40 GB is the fallback (§ VRAM budget) |

**The grader models, measured (HF API, 2026-08-26; both ungated, Apache 2.0 — no license
click-through needed):**

| | params (bf16) | checkpoint | vLLM `gpu_memory_utilization` (`roles.default_serve_util`) → pre-allocation on A100 **80 GB** (target) · 40 GB (fallback) |
|---|---|---|---|
| `google/gemma-4-E4B-it` (default) | 7.996B | **14.89 GiB** | **0.50** → `0.50 × 80 = 40 GiB` = weights + **~22 GiB KV pool** · `0.50 × 40 = 20 GiB` = weights + ~5 GiB KV |
| `google/gemma-4-E2B-it` (fallback) | 5.123B | **9.54 GiB** | 0.35 → `0.35 × 80 = 28 GiB` = weights + ~16 GiB KV · `0.35 × 40 = 14 GiB` = weights + ~3.5–4 GiB KV |

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
├── eda/               lean analysis package + notebooks + render driver; notebooks/scoring/Run_Eval.ipynb
│                      runs on the GPU host (Colab / server) — eda/ is pushed to Drive beside code/
└── history/           CHANGELOG.md — the only DATED narrative (decision rounds, the pre-run review)
```

## Naming — the arm identity grammar

**One regex, one module** ([code/naming.py](code/naming.py)), imported by both the trainers (write
side) and the EDA (read side). `EXPERIMENT_NAME` is **computed, never typed** — which is why Exp3's
`assert_name_matches_roles` guard has no Exp4 equivalent: the failure it prevented cannot occur.

```
{GRPO|PTO}4_{QTAG}_LA{K}_MCL{N}_{G{G} | M{M}_PT{greedy|indep}}_O{otag}_Pat{ptag}_Th{ttag}

GRPO4_Q1Q2_LA5_MCL12_G8_Ogemma4E4B_Patgemma4E4B_ThL1Bi
PTO4_Q1Q2_LA0_MCL12_M8_PTgreedy_Ogemma4E4B_Patgemma4E4B_ThL1Bi
GRPO4_WAI_LA0_MCL12_G8_Ogpt4m_Patgemma4E4B_ThL1Bi       # oracle flipped to the OpenAI API
GRPO4_Q1Q2_LA0_MCL12_G8_Ogemma4E4B_Patgemma4E4B_ThL1B   # therapist flipped to the BASE model
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
- The **therapist IS encoded** (`_Th{ttag}`, added 2026-08-27 while the grammar had produced zero
  folders — the only moment a mandatory field can be added without a version bump). Two variants:
  `L1Bi` = `meta-llama/Llama-3.2-1B-Instruct` (default) and `L1B` = the template-less base. The tag
  names a *family*; `run_metadata.json` records the exact snapshot. ⚠ `roles._slugify` no longer
  strips `-Instruct` (it used to — which would have slugged base and Instruct to the SAME tag and
  collapsed two different-policy arms into one folder; the smoke gate now pins the distinctness).

## Data layout

All three live under `Exp4_OpenStack/data/` as Google Drive directory symlinks (gitignored; the
schema here is the only record of their shape — same arrangement as Exp3).

```
data/
├── runs/<EXP_NAME>/
│   ├── run_metadata.json              current config (overwritten) + a `runtime` block for BOTH
│   │                                  methods (core.runtime.describe_environment: gpu_total_gib +
│   │                                  source, vllm_version, package_versions) and disable_dropout
│   ├── run_metadata_history.jsonl     append-only, one line per process  ← Exp3 fix #5
│   └── iteration_<N>/
│       ├── adapter/                   presence ⟺ "iteration done"
│       ├── training/                  HF Trainer output_dir (checkpoint-*, tb_logs/)
│       ├── eda/generations.jsonl      per-branch capture (GRPO: one atomic flush at the end;
│       │                              PTO: appended per trunk depth, normalised on resume)
│       ├── pref_pairs/{pairs.csv,pairs_fingerprint.json,_progress.json}   PTO only — the
│       │                              sidecar records the config the pairs were BUILT under;
│       │                              the reload path compares it and warns on mismatch.
│       │                              _progress.json = trunks + pairs + counters + n_eda_flushed
│       │                              (no EDA rows); pairs.csv is written only after the EDA rows
│       │                              are on disk, so it is also the "build complete" marker
│       ├── iteration_metadata.json    per-process phase seconds (+ cumulative_* from timing),
│       │                              per-phase CUDA peaks in ONE flat shape for both trainers:
│       │                              peak_reserved_gib_<phase> / peak_allocated_gib_<phase>, phase ∈
│       │                              {generate, build (PTO only), train, eval_generate}; `generate`
│       │                              is omitted on a mid-training resume (that attempt never ran
│       │                              it); truncation_* (TRUNCATION_COUNTER deltas per phase),
│       │                              lookahead_{sub_batch,gpu_calls,oom_events,runtime_errors,
│       │                              prompt_overflows}, prompt_bos_rule, disable_dropout (both
│       │                              methods), PTO branch_* counters
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
DEFAULT_ORACLE_MODEL    = "google/gemma-4-E4B-it"
DEFAULT_PATIENT_MODEL   = "google/gemma-4-E4B-it"
DEFAULT_JUDGE_MODEL     = "google/gemma-4-E4B-it"
DEFAULT_THERAPIST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"   # the policy; _ThL1Bi in every name

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
    gpu_memory_utilization: float = 0.25   # default for tests/planning; every entry point passes
                                           # default_serve_util(model) (E4B 0.50 / E2B 0.35 — § VRAM budget)
    max_model_len: int = 16384             # NOT an escape hatch — see § VRAM budget
    dtype: str = "bfloat16"
    extra_args: Tuple[str, ...] = ()

def model_tag(model_id: str) -> str            # google/gemma-4-E4B-it -> gemma4E4B ; [A-Za-z0-9] only;
                                               # raises ValueError for an UNCURATED id that slugs onto a
                                               # curated tag (base Gemma vs -it would share a folder)
DEFAULT_SERVE_UTIL: Dict[str, float]           # {model id: gpu_memory_utilization} — E4B 0.50, E2B 0.35
def default_serve_util(model_id: str, *, fallback: Optional[float] = None) -> float
        # THE one table of the fraction: both trainer notebooks assert their cell-1 literal against
        # it in their serve cells, smoke.py vram / roles read it (its private copy is retired).
        # ValueError for an unsized model unless fallback= is given — a 0.25 nobody sized is how
        # E4B failed to load. Run_Eval deliberately does NOT read it: scoring on an idle GPU uses
        # its own SERVE_GPU_MEMORY_UTILIZATION = 0.85.
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
    therapist_tag: str   # "L1Bi" (Instruct, default) | "L1B" (base)
    @property
    def label(self) -> str      # short display label, e.g. "GRPO_LA5"; non-default therapist
                                # appends its tag ("GRPO_LA0_ThL1B")
    @property
    def experiment_name(self) -> str

QTAG_BY_IDS: Dict[FrozenSet[int], str]                 # {1,2}->Q1Q2, {3}->WAI, {4}->CSQ8, {6}->MISAT, {7}->MITI, {1}->Q1, {2}->Q2
def qtag_for(questionnaire_ids: Sequence[int]) -> str   # raises on an unmapped set
def build_experiment_name(method, questionnaire_ids, k, mcl, *, g=None, m=None, mode=None,
                          oracle_model: str, patient_model: str,
                          therapist_model: str = DEFAULT_THERAPIST_MODEL) -> str
        # therapist_model is the ONE role with a keyword default: unlike Exp3's optional
        # suffixes this cannot collide (the tag is encoded either way); the config builders
        # always pass BASE_MODEL_ID explicitly.
def parse_experiment_name(name: str) -> ArmInfo         # raises ValueError on non-match
def model_state_label(n: int) -> str                    # "model_iter_0" ...
```

### `code/core/concurrency.py`

```python
class AsyncPrimitives:
    """Semaphores + GPU lock, created lazily, keyed by id(running loop) AND verified by identity.

    TRL may run the reward coroutine on a different event loop than the notebook's, and
    Python >= 3.10 raises when an asyncio primitive crosses loops. The cache stores the loop
    object beside the primitive: a hit requires the stored loop to BE the running loop (an id
    can be recycled once a loop is garbage-collected); anything else is evicted, never returned.
    """
    def __init__(self, *, oracle_concurrency: int, patient_concurrency: int): ...
    def oracle_sem(self) -> asyncio.Semaphore: ...
    def patient_sem(self) -> asyncio.Semaphore: ...
    def gpu_lock(self) -> asyncio.Lock: ...

def run_async(coro):
    """Run *coro* to completion from sync code, including inside a live Jupyter loop.

    When a loop is already running, spawn a daemon thread with its own fresh loop and join;
    that loop runs shutdown_asyncgens() + shutdown_default_executor() before it is closed, so
    an un-exhausted async generator or a run_in_executor job cannot leak past the call.
    Deliberately NOT nest_asyncio (broken on py>=3.13). CUDA is per-process, so torch calls
    from that thread are fine.
    """
```

⚠ **`gpu_lock` is held only across therapist `generate`, never across a patient `await`.** That
invariant is what lets the look-ahead API calls overlap with nothing blocking the GPU.

### `code/core/policy.py`

```python
CHATML_TEMPLATE: str                 # verbatim from Exp3 — installed ONLY when the checkpoint
                                     # ships no template (the base therapist); Instruct keeps
                                     # its native Llama-3 template
STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]             # BASE-therapist string stops
LLAMA3_END_MARKERS = ("<|eot_id|>", "<|eom_id|>", "<|start_header_id|>")
CHAT_TEMPLATE_DATE = "26 Jul 2024"   # pinned date_string on EVERY render — the Llama-3.2
                                     # template interpolates TODAY's date otherwise, so an arm
                                     # resumed on another day would see shifted prompts
ADAPTER_FILES    = ("adapter_model.safetensors", "adapter_config.json")
HF_TRAINER_FILES = ADAPTER_FILES + ("trainer_state.json",)

def setup_tokenizer(tokenizer_id: str, padding_side: str = "left")
        # keeps a shipped chat template (Instruct); installs CHATML_TEMPLATE when none (base)
def setup_base_model(base_model_id: str, *, use_4bit: bool = False)
def attach_lora(model, *, r, alpha, dropout, target_modules)
def patch_generate(model, tokenizer)          # idempotent; injects tokenizer= for stop_strings
def therapist_stop_token_ids(tokenizer) -> List[int]
        # eos + the LLAMA3_END_MARKERS present in vocab, deduped — [eot, eom, start_header] on
        # Instruct (its eos IS eot); inert extras on base. Passed as eos_token_id by every
        # decode path and installed on generation_config by sync_pad_token.
def clean_completion(text: Optional[str]) -> str          # cut at first ChatML marker; "" == degenerate

# --- THE PROMPT RULE (see the warning below the block) ------------------------------------
def tokenizer_adds_bos(tokenizer) -> bool      # probed once (tokenizer("x")[0] == bos_token_id), cached
                                               # per tokenizer; False without a bos_token_id (smoke stub)
def strip_leading_bos(text: str, tokenizer) -> str        # removes ONE leading bos_token; no-op without one
def render_prompt(messages, tokenizer) -> str  # apply_chat_template(add_generation_prompt=True,
                                               # date_string=CHAT_TEMPLATE_DATE) then strip_leading_bos —
                                               # THE render every prompt goes through
def prompt_token_ids(text: str, tokenizer) -> List[int]   # exactly one BOS then the text (add_special_tokens
                                               # iff tokenizer_adds_bos); idempotent; == trl GRPOTrainer's
                                               # processing_class(text=...) ids
def count_prompt_tokens(text: str, tokenizer) -> int      # len(prompt_token_ids) — what every budget compares to
def message_overheads(tokenizer) -> Dict[str, int]        # {"user": n, "assistant": n} wrapper cost, measured live
def estimate_message_costs(messages, tokenizer) -> List[int]   # per NON-system message, order-aligned
def system_overhead(system_prompt: str, tokenizer) -> int      # exact BOS + system message + generation suffix
def truncate_messages_drop_oldest(messages, tokenizer, max_tokens, *, message_token_costs=None,
                                  system_overhead=None) -> Tuple[Optional[List[Dict[str, str]]], int]
        # (kept_messages, n_dropped): leading system message kept, oldest non-system messages
        # dropped WHOLE until count_prompt_tokens(render) <= max_tokens; None when even the newest
        # turn alone (with the system message) exceeds the budget; CANONICAL (longest suffix that
        # fits) whatever estimates are passed
def build_prompt(messages, tokenizer, max_tokens, *, message_token_costs=None,
                 system_overhead=None) -> Tuple[Optional[str], int]
        # truncate_messages_drop_oldest + render_prompt: (BOS-free text | None, n_dropped). The
        # decode path AND both training-prompt builders use THIS function
@dataclass
class TruncationCounter:                       # prompts / truncated / dropped_turns / overflow
    def snapshot(self) -> Dict[str, int]; def delta_since(self, snapshot) -> Dict[str, int]; def reset(self)
TRUNCATION_COUNTER: TruncationCounter          # process-wide; generate_therapist_batch increments per
                                               # call; trainers snapshot at phase boundaries, log the delta

def generate_therapist_batch(model, tokenizer, batch_messages, *, max_tokens, temperature,
                             max_input_tokens=2048, stop_strings=None)
        -> Tuple[Optional[List[Optional[str]]], Optional[str]]
        # (responses, None) | (None, "oom") | (None, "runtime_error") — never raises on OOM
        # responses[i]: cleaned str | "" (degenerate) | None (NO PROMPT COULD BE BUILT: the newest
        # turn alone > max_input_tokens; that item was not generated, the others were)
        # max_input_tokens is BOS-INCLUSIVE; the call NEVER token-truncates (build_prompt drops
        # oldest turns whole instead). stop_strings: None = the ChatML defaults; EMPTY = no string
        # criteria (Instruct arms stop on the eos-id list alone — no per-call StopStringCriteria build)

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
        # checkpoint here would silently re-anchor the KL/DPO reference to the crash point.
        # The TRAINER then restores the crashed process's trained "default" adapter ITSELF,
        # between trainer construction and train(): transformers' _load_from_checkpoint loads
        # ONLY adapter subdirs when any exist (trl's ref/), so the root default would otherwise
        # stay at iteration start — grpo_trainer.restore_default_adapter /
        # pto_trainer._restore_default_adapter_from_checkpoint, both via
        # load_adapter(ckpt, "default", is_trainable=True); the iteration adapter is saved with
        # selected_adapters=["default"] so no ref/ ships. smoke.py resume pins all of this.
def compute_cumulative_step_offset(run_dir) -> int
```

⚠ **`patch_generate` must be re-applied after every re-wrap** — `PeftModel.from_pretrained` and
both TRL trainers hand back a fresh `generate`. Call it at base load, in `resolve_start_state`, and
on both sides of `trainer.train()`.

⚠ **THE PROMPT RULE — rendered text never carries a BOS; every tokenization adds exactly one.**
`add_special_tokens=True` on both Llama-3.2 tokenizers, which is what trl `GRPOTrainer`'s
`processing_class(text=prompts)` does, so the decode path and TRL see byte-identical ids (the
previous code double-BOS'd the Instruct template and gave the base template no BOS at all).
`max_input_tokens` / `max_prompt_tokens` are **BOS-inclusive** (the length of `prompt_token_ids`).
The decode path **never token-truncates**: an over-budget conversation drops its OLDEST turns whole
and keeps the system message (`core.policy.build_prompt`), and that is the SAME function
`build_truncated_training_prompt` / `extract_prompts_from_conversations` use — so a PTO branch
candidate is sampled from byte-identical text to the DPO prompt it trains on, and a GRPO prompt is
what the policy generated from. Science change vs Exp3, whose serve-time prompts were
left-truncated at the token level (past ~utterance 24 every therapist turn was generated from a
prompt starting mid-utterance with no system prompt); applies to both methods and both K arms. For
the base therapist it also ADDS a BOS at serving that the old code lacked — deliberate (Llama-3.2
base was pretrained with BOS; no Exp4 data existed). Keep `THERAPIST_MAX_INPUT_TOKENS ==
MAX_ALLOWED_PROMPT_LENGTH` (2048) or the two prompts diverge. `tokenizer.truncation_side="left"` is
still set in `setup_tokenizer` as belt-and-braces for any HF/TRL path that truncates on its own; no
Exp4 path relies on it. `TRUNCATION_COUNTER` makes the truncation rate a logged number — a
truncated prompt is a policy that no longer sees the session start. The DPO tokenization path is
verified: trl 1.4.0 tokenizes a string `prompt` through `_prepare_dataset → _tokenize →
processing_class(text=prompt)` (installed `dpo_trainer.py:875`, default `add_special_tokens=True`
→ exactly one BOS on the BOS-free text; the `add_special_tokens=False` sites at ~:332–346 belong to
`DataCollatorForVisionPreference` — the VISION-dataset collator, which Exp4's string-prompt dataset
never goes through; there is no "conversational branch" on the string path), measured
`== prompt_token_ids(prompt)` on the Instruct tokenizer, and `pto_trainer.build_dpo_dataset` asserts
exactly one leading BOS per prompt.

⚠ **The anti-degeneracy stack is a BASE-therapist (`_ThL1B`) concern.** There the policy is a base
model on the hand-written ChatML template: `<|im_start|>` / `<|im_end|>` are ordinary BPE pieces,
not special tokens, so the model happily writes both speakers, and the whole stack is load-bearing —
`STOP_STRINGS` at every decode site + `clean_completion` + `patch_generate` + GRPO's
`generation_kwargs` + `REWARD_FLOOR`. On the INSTRUCT therapist (`_ThL1Bi`, the default) the failure
class does not exist: the native template ends every turn on the single special token `<|eot_id|>`,
`STOP_STRINGS="auto"` resolves to `()`, and stopping is the `therapist_stop_token_ids` eos list
(GRPO passes it via `generation_kwargs={"eos_token_id": ...}`). `clean_completion` + `REWARD_FLOOR`
stay wired on both variants (an empty completion is degenerate either way).

⚠ **Known BASE-arm asymmetry (documented, deliberately NOT changed).** On the base ChatML therapist
the two methods train on different turn terminators: the policy *emits* the 6-piece BPE string
`<|im_end|>`, which `clean_completion` strips before anything is scored or stored; PTO's DPO path
then trains on `chosen` / `rejected` texts to which trl's `add_eos` appends the tokenizer's real EOS
`<|end_of_text|>` — a terminator the policy never generated — while GRPO trains on the raw emitted
ids, `<|im_end|>` pieces included. So a base-arm PTO policy is pushed toward `<|end_of_text|>` and a
base-arm GRPO policy toward the ChatML string; the Instruct arms (`_ThL1Bi`, the default) are
consistent on `<|eot_id|>` in both methods, because there the emitted terminator IS the tokenizer's
EOS. This is a property of the base template + trl, it affects only `_ThL1B` arms, and it is left as
is because the base variant is the non-default alternate and any "fix" (custom EOS, stripping
`add_eos`) is a science change to one method. Anyone reading a base-arm PTO-vs-GRPO contrast must
know the terminator differs.

⚠ `core.policy.build_prompt` (via `_fit_messages`) accepts a message list with **no system
message** — the look-ahead and the branch sampler occasionally hand it one — and budgets it
correctly (no system overhead is charged, the drop-oldest loop keeps the newest turn); a
system-less list is not an error and must not be padded with an empty system message.

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
def has_session_end(text: str) -> bool    # THE ONE presence test for the keyword — the same regex
                                          # handle_session_end / split_session_end cut on; the conv
                                          # loop, the look-ahead, the reward fn and the PTO grower all
                                          # ask THIS (a presence test that diverges from the cutter is
                                          # a turn that ends in one place and is graded from another)

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
        # with exponential backoff slept OUTSIDE the semaphore; binding.extra_body passed through.
        # ""/whitespace-only content is retried like None; HTTP 4xx other than 408/429 raise
        # IMMEDIATELY (no retry — a 404/400/422/401 cannot change on resend); timeouts, 408, 429,
        # 5xx and connection errors keep retrying
async def generate_patient_batch(client, binding, batch_messages, sem, **kw) -> List
        # asyncio.gather(return_exceptions=True) — per-conversation failures come back as exceptions

async def conversation_loop_batch(...) -> Tuple[List[ConversationState], Optional[str], List[int]]
def generate_all_conversations(..., allow_partial=False) -> List[ConversationState]
        # num_utterances counts ADDITIONAL utterances after the scripted therapist opener:
        # total <= num_utterances + 1 = 50 at the default 49 (NUM_UTTERANCES_FOR_DATA);
        # resumes from per-persona CSVs already on disk (written ATOMICALLY: temp + os.replace,
        # so a preemption mid-write can't leave a shorter-but-parseable truncation that the
        # resume then treats as a complete conversation); bounded no-progress retries; an OOM
        # batch HALVES the batch size stickily and re-slices (mirrors the look-ahead);
        # RAISES RuntimeError when personas are still missing after the retry bound unless
        # allow_partial=True — the failures correlate with length/difficulty, so a partial set
        # feeding training or eval is biased missingness on the headline metric;
        # gc.collect()+empty_cache() BETWEEN batches (not cosmetic — the allocator high-water
        # mark grows otherwise) and prints per batch line a `vram <N>G` field and a
        # `trunc <n>/<B>` field (therapist prompts that lost >= 1 oldest turn / prompts built,
        # from core.policy.TRUNCATION_COUNTER deltas; plus `overflow <k>` when a prompt could not
        # be built at all — that conversation is marked FAILED: a budget misconfiguration, not
        # policy behaviour)

def build_truncated_training_prompt(turns, system_prompt, tokenizer, max_prompt_tokens,
                                    truncation_mode="drop_oldest") -> Optional[str]
        # == core.policy.build_prompt(turns_to_messages(...)); the returned text is BOS-FREE (TRL
        # adds the one BOS) and byte-identical to what generate_therapist_batch generated from for
        # the same turns + budget; max_prompt_tokens is BOS-inclusive.
        # None when even one most-recent turn exceeds budget -> caller SKIPS the pair
def extract_prompts_from_conversations(states, system_prompt, tokenizer, *, min_conv_length,
                                       max_prompt_tokens, permutations) -> List[Dict]
        # one sample after each patient turn whose conv-so-far has >= min_conv_length utterances
        # keys: prompt (BOS-FREE), transcript, conversation_id, persona_id, patient_system_prompt
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
    max_input_tokens: int = 2048            # therapist prompt budget, BOS included; an over-budget
                                            # rollout drops its OLDEST turns whole and keeps the
                                            # system message (core.policy.build_prompt) — never
                                            # token-truncates
    patient_binding: RoleBinding = ...
    stop_strings: Tuple[str, ...] = ...
    sub_batch_size: Optional[int] = None    # None = one padded generate over all active sims

NOT_GRADED_STOP_REASONS = frozenset({"patient_error", "gpu_error", "prompt_overflow", "parse_error"})
def split_session_end(text: str) -> Tuple[str, bool]
        # (content_before_keyword.strip(), keyword_present) — delegates to
        # core.conversations.handle_session_end: ONE definition of where a closing turn ends

@dataclass(frozen=True)
class LookaheadResult:
    extended_transcript: str; tail: str; realized_turns: int; ended_early: bool
    stop_reason: str = ""                   # "" ran to k | "session_ended" | "degenerate" (complete,
                                            # graded) | "patient_error" | "gpu_error" | "prompt_overflow"
                                            # | "parse_error" (NOT_GRADED_STOP_REASONS — the simulator failed)
    k: int = 0
    @property
    def graded(self) -> bool                # stop_reason not in NOT_GRADED_STOP_REASONS
    def to_record(self) -> dict             # {k, tail, realized_turns, ended_early, stop_reason}

async def simulate_lookahead_batch(model, tokenizer, client, cfg: LookaheadConfig,
                                   primitives, transcripts, completions,
                                   sp_therapist, sp_patient_list) -> List[LookaheadResult]
```

Advances all B sims **in lock-step**: one padded batched `model.generate` per simulated therapist
turn, then one batched patient round. OOM halves the sub-batch and **the halving is sticky** across
steps; at sub-batch 1 an OOM freezes that single sim. A non-OOM runtime error is handled the same
way, but **locally**: the failing chunk is halved and retried down to size 1, and only the sim that
still fails alone freezes as `gpu_error` — the other sims of the chunk advance (deliberately unlike
the conversation loop, which aborts). Toggles `model.eval()` +
`use_cache=True` and restores both in `finally` — this runs while the policy is in `train()`
mid-optimizer-step.

A sim the simulator could not finish (`stop_reason` in `NOT_GRADED_STOP_REASONS`) is **not
graded**: `core.reward` gives it `score=None`, `not_graded_reason=<stop_reason>`, keeps the
transcript it reached in `scored_text`, counts it as a FAILURE in the `min_success_ratio` gate, and
GRPO's `rewards_for_trl` substitutes the group mean. A frozen sim must never be scored as if the
conversation had ended there — a short transcript reads as "the therapist stopped", which is a
policy judgement about an infrastructure fault. A per-item `None` from `generate_therapist_batch`
(no prompt could be built: the newest turn alone exceeds `max_input_tokens`) freezes the sim as
**`prompt_overflow`** — a budget fault, never a GPU one — counted in
`LookaheadState.prompt_overflows` and stamped into `iteration_metadata.json` as
`lookahead_prompt_overflows` beside `lookahead_runtime_errors` (the `gpu_error` count) by both
trainers.

### `code/core/oracle.py`

```python
REWARD_FLOOR = 0.0
RETRYABLE_4XX = frozenset({408, 429})
def is_non_retryable_http_error(exc: BaseException) -> bool
        # openai.APIStatusError with 400 <= status < 500 and not in RETRYABLE_4XX: raised after ONE
        # call; timeouts, connection errors, 5xx and validation ValueErrors still retry

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

**Retry ladder.** A validation failure (schema echo, item count, type/range) retries up to
`max_retries` — the grader may do better next time. An HTTP 4xx other than 408/429 short-circuits
after ONE call (`is_non_retryable_http_error`): a resent 404/400/422/401 cannot change, and at
`G=8 × 16 prompts × 2 rubrics` the old behaviour was a long silent stall before
`min_success_ratio` finally fired. The patient path applies the same rule (`core.conversations`).

⚠ **The rubric-first prompt layout in `questionnaires.py` is load-bearing.** Fixed instructions +
rubric FIRST, transcript LAST — that is what vLLM's prefix caching (and OpenAI's, on an API arm)
reuses across every call. Never move the transcript ahead of the rubric.

### `code/core/reward.py`

```python
ORACLE_FAILED_REASON = "oracle_failed"

@dataclass
class CandidateScore:
    completion: str                 # CLEANED; KEEPS the SESSION ENDED keyword when ended_by_candidate
    score: Optional[float]          # raw oracle mean | REWARD_FLOOR (degenerate) | None (NOT graded)
    sub_scores: Optional[Dict[str, float]]
    success: bool; attempts: int; degenerate: bool
    scored_text: str                # exactly what the oracle read (a closing candidate: text BEFORE the keyword)
    lookahead: Optional[Dict[str, Any]]   # {k, tail, realized_turns, ended_early, stop_reason} | None
    not_graded_reason: Optional[str] = None   # ORACLE_FAILED_REASON | a NOT_GRADED_STOP_REASONS value | None
    ended_by_candidate: bool = False          # contained SESSION ENDED: graded on its seed, no rollout
    def to_record(self, idx, *, role=None, reward_used=None) -> dict
        # always carries "ended_by_candidate"; "not_graded_reason" ONLY when score is None

def make_reward_fn(model, tokenizer, client, oracle_cfg, la_cfg, primitives, *,
                   recorder=None, sp_therapist=None) -> Callable
def rewards_for_trl(candidates, num_generations) -> List[Optional[float]]
        # per-group repair of an ungraded candidate (see the oracle section's None warning);
        # now also covers a look-ahead simulator failure (score None)
```

The TRL reward callable: cleans completions, floors degenerate ones to `REWARD_FLOOR`, **splits
`SESSION ENDED` candidates** (the seed is graded, the explanation dropped as in a saved
conversation, no rollout runs — at K>0 recorded as a complete zero-turn rollout with
`stop_reason="session_ended"`; a keyword-only candidate is degenerate), runs look-ahead when
`la_cfg.k > 0`, **leaves simulator-failed look-aheads ungraded** (`score=None`, gate-counted), scores
the rest with the oracle, records the group to the recorder.
⚠ **Raises `RuntimeError` when the GATE rate falls below `oracle_cfg.min_success_ratio`** — the gate
rate is `graded / (sent to the oracle + look-ahead failures)`; the grader-only rate is
`oracle_success_rate` (TensorBoard `oracle/success_rate`), and `reward/graded_frac` +
`lookahead/not_graded_frac` split the two failure classes. Training on a biased subset is worse than
stopping. TRL passes `transcript` / `persona_id` / `patient_system_prompt` through as `**kwargs`
(needs `remove_unused_columns=False`).

⚠ TRL hands completions back as **G-consecutive blocks per prompt**; reshape `(-1, G)` to recover
groups, and skip the record with a warning when `n % G != 0` rather than mis-grouping.

### `code/core/recorder.py`

One JSONL row per branch — prefix stored **once**, candidates nested:

```json
{"phase": "group|tree|independent", "iteration": 3, "conversation_id": "...", "persona_id": 7,
 "branch_id": 0, "eval_pass": false, "prefix": "...",
 "candidates": [{"completion": "...", "score": 3.4, "sub_scores": {"1": 3.0, "2": 3.8},
                 "ended_by_candidate": false,
                 "lookahead": {"k": 5, "tail": "...", "realized_turns": 5, "ended_early": false,
                               "stop_reason": ""}},
                {"completion": "...", "score": null, "not_graded_reason": "patient_error",
                 "ended_by_candidate": false,
                 "lookahead": {"k": 5, "tail": "...", "realized_turns": 2, "ended_early": true,
                               "stop_reason": "patient_error"}}],
 "chosen_idx": 0}
```

`score` is the RAW grader result (`null` = NOT graded — `not_graded_reason` says why:
`"oracle_failed"` | `"patient_error"` | `"gpu_error"` | `"prompt_overflow"` | `"parse_error"`,
present only on `null` rows; on PTO rows `gpu_error` / `prompt_overflow` can also come from the
BRANCH SAMPLER — `pto_trainer._apply_sampling_failures`, written with `oracle {success:false,
attempts:0}`, `degenerate false` and `lookahead null`, which is how they are told apart from a
look-ahead failure). `ended_by_candidate` is on EVERY candidate: `true` means the completion contained
`SESSION ENDED`, the oracle graded only the text before the keyword, no rollout ran, and
`completion` still holds the keyword. `lookahead.stop_reason` ∈ `"" | session_ended | degenerate |
patient_error | gpu_error | prompt_overflow | parse_error`. A candidate also carries
`reward_used` **only when the number GRPO optimised differs from it** — i.e. the group-mean
substitution above. `group_mean` / `group_std` are TRL's own reduction of that vector (surviving
`null` at 0.0, **sample** SD, ddof=1), so `sign(reward_used − group_mean)` reconstructs the sign of
the advantage.

Reconstruct a scored text as `prefix + "\n\n[THERAPIST]: " + completion + (tail or "")`. ⚠ For
`ended_by_candidate` rows split `completion` at the keyword first
(`core.lookahead.split_session_end`) — the rule no longer reproduces the graded text there.
(`recorder.py`'s schema docstring carries the same keys; `EDARecorder.aggregate()` emits
`eda/ended_by_candidate_frac` and `eda/lookahead_not_graded_frac` from them.)
`snapshot_to(path)` / `load_from(path)` support checkpoint-resume (HF fast-forwards skipped batches
**without re-invoking the reward fn**, so the recorder must be restored from the checkpoint).
**Two write paths, one line format** (`EDARecorder.jsonl_line`): GRPO buffers the iteration and
`flush()`es once, atomically, at the end; PTO calls `append_to_disk(n_already)` after every trunk
depth (an O_APPEND of the new rows, counted only after the write returned) and `rewrite(rows)` on
resume to drop a torn trailing line — so a preempted hour-long build keeps the rows of the depths
that finished.

⚠ **`branch_id` is trunk DEPTH for PTO, not a unique id** — it repeats across conversations. Any
per-branch aggregation must key on `(conversation_id, branch_id)`.

⚠ **`eval_pass` is written on EVERY row** (never omitted). With a GRPO eval split, TRL calls the
reward function during `evaluate()` too — held-out prompts, policy in eval mode, no gradient — and
those groups land in the same `generations.jsonl`. `EDARecorder.aggregate()` reports the
gradient-bearing rows under the existing keys and the held-out half under `eda/eval_*`; the EDA's
`load_generations` returns the flag as a column. Anything that pools the two answers a different
question at a blend ratio nobody chose.

### `code/core/timing.py`

Ported from Exp3 `_shared/timing.py`, two changes: `PHASE_KEYS` gains an eval-generation phase, and
the training phase logs **partial** lines at every checkpoint save.

```python
PHASE_KEYS            = ("generation_s", "pref_pair_s", "training_s", "eval_gen_s")
PRODUCTION_PHASE_KEYS = ("generation_s", "pref_pair_s", "training_s")   # the COST axis
def log_session(iter_dir, *, generation_s=0.0, training_s=0.0, pref_pair_s=0.0,
                eval_gen_s=0.0, started_at=None, note="") -> dict     # record gains "partial": false
def begin_training_phase(iter_dir) -> None
        # call at the start of EVERY training attempt, right before the phase clock starts: resets
        # this process's per-attempt partial-time ledger for iter_dir, so the first on_save of the
        # new attempt logs its increment from 0 rather than from the previous attempt's last partial.
        # An in-kernel re-run of a crashed iteration is a NEW attempt (same process, fresh clock) —
        # without the reset its first partial line would be negative/clamped and the phase undercounted
def log_training_progress(iter_dir, *, elapsed_s: float, note="") -> dict
        # from TrainerCallback.on_save, elapsed_s = time.time() - train_started_at (the SAME clock
        # the phase's training_s is read off); appends only the INCREMENT since this process's
        # previous partial line for iter_dir ("partial": true) in the CURRENT attempt (opened by
        # begin_training_phase); {} when nothing accrued
def finalize_training(iter_dir, total_s: float, *, started_at=None, note="") -> dict
        # right after trainer.train() returns: logs only the remaining delta (a zero line for the
        # completion note); clamps a total below the partials to 0 with a warning; idempotent
def cumulative_seconds(iter_dir) -> Dict[str, float]
        # + total_s, production_s, n_sessions, n_sessions_production — partial lines sum like any
        # other (the flag is audit only); every line carries the per-process token
def metadata_fields(iter_dir) -> Dict[str, float]         # cumulative_* to splat into metadata
```

**Phases log themselves AS THEY COMPLETE** — one line after generation, one after the preference
build (PTO), and for training one partial line per `save_steps` checkpoint plus the closing
`finalize_training` line — so a Colab preemption mid-training still leaves the finished generation
phase AND the training time up to the last checkpoint on the cost record (the Exp3 undercount this
module exists to fix). Every line carries a per-process token (`host:pid:start`), and `n_sessions` /
`n_sessions_production` count distinct PROCESSES, not lines, so per-phase and partial logging do not
inflate the counters. ⚠ **Once `on_save` is wired, never `log_session(training_s=...)` at the end of
the phase — `finalize_training` replaces it, or the phase is double-counted.** Both trainers make
exactly that swap: both trainers open every training attempt with `begin_training_phase(iter_dir)`
and then start the clock, `grpo_trainer.CheckpointMetadataCallback.on_save` and PTO's callback in
`run_training_phase` call `log_training_progress` on the phase's own clock, `run_one_iteration`
closes with `finalize_training`, and the only remaining `log_session(...)` calls are the
generation / pref-build / eval-gen phases. The ledger is **per attempt, not per process**: a
crashed iteration re-run in the same kernel is a new attempt with a fresh ledger, so its partial
lines do not inherit the previous attempt's offset.

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
DEFAULT_READY_TIMEOUT = 1800.0            # E4B's cold start on Colab compiles CUDA graphs; the old 900 s cut it off
SPEC_SOURCES = ("launched", "cmdline", "requested")
DEFAULT_LOG_DIRNAME = "_vllm_logs"        # default log dir = ./_vllm_logs (the notebooks pass /content/vllm_logs)

@dataclass
class ServerHandle:
    model: str; base_url: str; process: Optional[subprocess.Popen]; log_path: Optional[str]; spec: ServeSpec
    restarts: int = 0
    spec_source: str = "launched"         # "launched" (this process) | "cmdline" (recovered via pgrep from a
                                          # running `vllm serve` argv) | "requested" (adopted over HTTP only —
                                          # the spec is what WE asked for, not necessarily what it runs)
    pid: Optional[int] = None
    executable: str = "vllm"
    @property
    def owns_process(self) -> bool
    def stop(self, timeout: float = 30.0) -> None      # no-op unless owns_process
    def tail_log(self, n: int = 40) -> str
    def is_alive(self) -> bool

def wait_until_ready(base_url, *, timeout=1800.0, process=None, poll_seconds=3.0, log_path=None, pid=None) -> None
        # the deadline EXTENDS while the startup log keeps growing (hard cap 4x); fast-fails if the process/pid dies
def start_server(spec, *, log_dir=None, timeout=1800.0, executable="vllm") -> ServerHandle
        # adopts a server already LOADING for the spec (find_loading_server) instead of launching a second one
def launched_servers() -> Dict[int, ServerHandle]      # the process-wide launch registry, by port
def spec_from_cmdline(argv, *, fallback: ServeSpec) -> ServeSpec
def find_loading_server(spec, *, log_dir=None) -> Optional[ServerHandle]
def served_max_model_len(base_url, *, model=None) -> Optional[int]
def adopt_if_running(spec, *, log_dir=None) -> Optional[ServerHandle]
        # the owning handle for this process's own server; else the spec recovered via pgrep ("cmdline");
        # else an HTTP-only adoptee carrying the REQUESTED spec — with its max_model_len READ BACK from
        # the server (served_max_model_len), so a "requested" handle never advertises a cap the server
        # was not launched with
def serve_roles(bindings, *, base_port=8000, log_dir=None, timeout=1800.0, executable="vllm", **spec_kw)
        -> Tuple[Dict[str, RoleBinding], Dict[str, ServerHandle]]
        # plan -> adopt-or-start each -> return bindings with base_url filled in; the summary line
        # prints the weights GiB, the KV-cache-tokens line and the max-concurrency line
def ensure_alive(handle, *, max_restarts=3, grace_seconds=30.0, timeout=1800.0) -> ServerHandle
        # a health probe at a PHASE BOUNDARY (never inside a retry path): a handle whose pid is alive
        # but whose endpoint no longer answers is treated as WEDGED — killed and relaunched, not waited
        # on; relaunches a "launched" handle's spec, or a "cmdline" one under the recovered spec; REFUSES
        # (RuntimeError) to relaunch a "requested" adoptee — Run_Eval's util 0.85 server vs the
        # trainer's 0.50 would otherwise be silently replaced by the wrong pre-allocation
def report_weights_gib(handle) -> Optional[float]       # parsed from the vLLM startup log
def report_kv_cache_tokens(handle) -> Optional[int]     # the "GPU KV cache size: N tokens" line
def report_max_concurrency(handle) -> Optional[float]   # the "Maximum concurrency ... Xx" line
```

`serve_roles` is **idempotent**: a healthy server already on the port serving the right model is
adopted, not duplicated, and the process-wide **launch registry** refuses a second launch of the
same spec (a re-run cell cannot double-start a server and hold two pre-allocations). `ensure_alive`
is called from exactly these sites and no others: **GRPO** at the generate → train boundary and
the train → next-iteration boundary (`grpo_trainer.run_one_iteration`); **PTO** at the loop top,
before the preference build and before the DPO step (`pto_trainer.run_one_iteration(...,
server_handles=, client_factory=)` / `run_final_eval(...)` take the handles and a client factory
so a relaunch can rebuild the async client on the new endpoint). **Never from a core retry path**
— `core.conversations` / `core.oracle` / `core.lookahead` retry their own calls and know nothing
about the server process; a probe inside a retry loop would race the very requests it is retrying.
Its `RuntimeError` on a `requested` adoptee propagates out of the trainer unchanged (there is no
"restart the serve cell" wrapper). Read BOTH the weights line and the **KV cache tokens** line at
the Phase 1 gate. ⚠ vLLM's
`/tokenize` shape and the two log lines were written from the OpenAI-compatible protocol as
documented, not verified against a pinned build (none is installed locally) — each degrades to a
labelled fallback / `None` rather than a wrong number; confirm at the Phase 1 gate.

### `code/tools/oracle_sanity.py` and `code/tools/smoke.py`

```python
async def run_sanity(binding, *, questionnaire_ids=(1, 2), quick=False, concurrency=8,
                     max_tokens=256, max_retries=3, request_timeout=120.0, ...) -> SanityReport
def check_gates(report) -> Tuple[bool, List[str]]
def format_report(report) -> str ; def write_report(report, path) -> str
def prompt_length_report(source: str | Sequence[str], questionnaire_ids=(1, 2), *, base_url=None,
                         tokenizer=None, model=None, max_model_len=None, progress=False) -> PromptLengthReport
        # THE PHASE 2 MEASUREMENT. `source` is a model_iter conversations folder (the CLI's
        # --conv-dir; tools/generate_convs.py runs it after every pass and exits 1 on overflow) OR
        # a list of oracle-formatted transcripts in memory (Run_Eval § 8 over every arm on disk).
        # Builds the REAL oracle prompt (questionnaires.get_prompt_eval_questionnaire) around each
        # transcript and counts it through the server's /tokenize when base_url is given (THE
        # measurement), else a local HF tokenizer for `model`, else a labelled chars/3.5 estimate
        # (`.method` says which). `.to_dict()` carries per_rubric {label: {n, median, p95, max,
        # n_over, frac_over, headroom (fraction of the cap unused)}} AND the flat gate keys
        # n_transcripts, per_questionnaire {qid: ...}, n_over, max_tokens (longest prompt),
        # headroom (= max_model_len / max_tokens). format_prompt_length_report(report | dict) -> str;
        # check_prompt_lengths(report) -> (passed, reasons). eda_analysis.scoring.prompt_length_gate
        # (Run_Eval § 8) passes base_url=JUDGE.base_url, converts the dataclass to its dict, and
        # check_prompt_length_gate refuses to pass when the function or the n_over key is absent.
```

`smoke.py` subcommands (check counts as of the 2026-09-03 review-repair round): `naming` 32 ·
`config` 29 · `convs` 29 · `vram` 23 + 1 WARNING (sized for 80 GB AND 40 GB; **WARNS** on the 40 GB card, where
the 19.7 GiB envelope leaves ≈ 0 headroom beside the 20 GiB server) · `resume` 13 (mid-training
resume keeps the iteration-start reference, `restore_default_adapter` reloads the trained `default`,
BOS rule — exercised through the trainers' REAL restore helpers, not a re-implementation) ·
`prompts` 30 (THE PROMPT RULE + drop-oldest truncation, system-led and system-less, on both therapist tokenizers) · `serve` ·
`roles` 24 against an ADOPTED server / 28 when it LAUNCHES the server (+ the kill → restart checks;
Colab; the thinking gate — SKIP without vLLM on PATH) · `stopgen` 3 · `dpo` 7 · `grpo` 6 (GPU,
~3 GB peak). `all` = 172 checks locally with `serve`/`roles` skipped. ⚠ Never raise a batch
size, sequence length or model size inside the GPU parts — an over-budget request reboots the
local machine.

⚠ `gpu_memory_utilization` is a **pre-allocation, not a growing ceiling**. Sharing the card with a
live trainer wants the sanctioned per-model fraction (`roles.default_serve_util`: E4B 0.50, E2B
0.35 — never lower: below the weights + a usable KV pool the server fails to start or serves at
near-zero concurrency, § VRAM budget) and the server started **FIRST**, because training memory is
the spiky side.

### Notebook cell-order contract (both trainers)

0. install cell (guarded; Colab only) → 1. flat globals (cell 1) → 2. Drive mount + runtime detect
+ auth → 3. **`serve_roles()` — before any torch import** → 4. `import trl` **then** torch/model
build → 5. visible orchestration loop.

The install cell pip-checks the pinned set, probes the vLLM build, and after installing anything
**RAISES to stop the kernel** (the two install cells are byte-identical, gate-checked): a
"run all" must not continue on a half-installed kernel holding pre-install modules. After the
restart the kernel's cwd is back at `/content`, so **re-run the mount cell** before anything that
touches `code/`. `Run_Eval.ipynb` carries the same mount preamble (its cell 2).

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

**Target: Colab A100 80 GB** (80 GiB as `nvidia-smi` reports it); **A100 40 GB is the fallback**,
and every number below is shown for both. Derived from the MEASURED checkpoint sizes (HF API,
2026-08-26): E4B **14.89 GiB** bf16, E2B **9.54 GiB** — the "~3 GB" figure earlier revisions carried
was wrong by 3–5×, and at the old `util 0.25` (10 GiB on a 40 GB card) the E4B server could not even
hold its weights. `run_metadata.json` records which card an arm ran on
(`core.runtime.describe_environment()` → `gpu_total_gib`, `gpu_total_gib_source`, `vllm_version`,
`package_versions`), so an 80 GB arm is distinguishable from a 40 GB one after the fact.

### The server (started first)

| card | `gpu_memory_utilization` | pre-allocation | inside it |
|---|---|---|---|
| **80 GB** | **0.50** (E4B) | `0.50 × 80 = 40 GiB` | 14.89 GiB weights + **~22 GiB KV pool** (vLLM keeps ~2–3 GiB for activations / CUDA graphs) |
| 40 GB | 0.50 (E4B) | `0.50 × 40 = 20 GiB` | 14.89 GiB weights + **~5 GiB KV** — serves, at a fraction of the concurrency |
| 80 GB | 0.35 (E2B) | `0.35 × 80 = 28 GiB` | 9.54 GiB weights + ~16 GiB KV |
| 40 GB | 0.35 (E2B) | `0.35 × 40 = 14 GiB` | 9.54 GiB weights + ~3.5–4 GiB KV |

The fraction is a function of the MODEL, not the card — `roles.default_serve_util(model)`, the one
table the trainer notebooks (their serve cells assert the cell-1 literal against it) and
`smoke.py vram` / `roles` read, so the trainer-side entry points cannot disagree. `Run_Eval` scores
on an otherwise idle GPU and deliberately uses its own `SERVE_GPU_MEMORY_UTILIZATION = 0.85`
(`0.85 × 80 = 68 GiB` = weights + a ~50 GiB KV pool).
The 80 GB card buys a ~4× KV pool at the same fraction, i.e. concurrency: that is what makes
`ORACLE_MAX_CONCURRENCY=64` + `PATIENT_CONCURRENCY=96` real requests rather than a queue. **Read the
`KV cache tokens` line** the startup log prints (reported beside the weights line by
`tools/vllm_serve.py`) at the Phase 1 gate: pool ÷ ~4k-token median prompt = how many oracle calls
the server holds at once. `--max-model-len 16384` (next). Prefix caching on.

### `max_model_len`

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
gate** and raise the cap if the margin has eaten in. That measurement now exists in the scoring
path: `Run_Eval.ipynb` § 8 → `eda_analysis.scoring.prompt_length_gate` →
`tools.oracle_sanity.prompt_length_report` (counts in the served model's own tokenizer; hard-fails
on any prompt over the cap; notes headroom under 1.25×).

### The trainer envelope (arithmetic — the QUICK_TEST rehearsal is what MEASURES it)

Llama-3.2-1B: 16 layers, 8 KV heads × 64 head-dim, bf16 ⇒ KV cost **32 KiB per token per sequence**
(`16 × 2 × 8 × 64 × 2 B`). A therapist sequence is at most `2048 prompt + 200 completion = 2,248`
tokens.

| phase | GRPO (`per_device 16 × gas 8`, checkpointing ON) | PTO / DPO (`per_device 2 × gas 8`, checkpointing ON) |
|---|---|---|
| weights + LoRA + optimizer | **~2.5 GiB** (1B bf16 ≈ 2.3 GiB; LoRA r=16 + Adam states ≈ 0.1 GiB) | same |
| generation | ONE `generate` per optimizer step of `per_device × steps_per_generation = 16 × 8 = 128` completions (`steps_per_generation` defaults to `gas`, installed trl `grpo_config.py:909–911`): `128 × 2,248 × 32 KiB ≈ 8.8 GiB` KV — **≤ 8.8 GiB** | branch sampling batched by `CONVERSATION_BATCH_SIZE=64`: `64 × 2,248 × 32 KiB ≈ 4.4 GiB` |
| look-ahead (K=5) | `LOOKAHEAD_SUB_BATCH_SIZE=64`: `64 × 2,248 × 32 KiB ≈ 4.4 GiB` — **≤ 4.4 GiB** (auto-halves on OOM, sticky) | same 4.4 GiB, during the build phase |
| loss forward | `per_device=16` sequences, logits over the completion only (`logits_to_keep`), fp32: `16 × 200 × 128,256 × 4 B ≈ 1.5 GiB`, ×2 for old/ref log-probs, checkpointed activations small — **~2–4 GiB** | full-sequence logits × 128k vocab (DPO keeps them all): `2 × 2,448 × 128,256 × 4 B ≈ 2.3 GiB` per forward, chosen + rejected; **~17 GiB measured** for the whole DPO step in Exp3 (`2 × 8`, ckpt on) |
| **conservative envelope (sum)** | **`2.5 + 8.8 + 4.4 + 4.0 = 19.7 GiB`** (a PLAN, unmeasured — `smoke.TRAINER_ENVELOPE_GIB` carries the same four terms, so every entry point prints this one arithmetic) | **≈ 17 GiB** (Exp3 measurement) |

The GRPO phases are sequential (the 128-completion KV is freed before the look-ahead runs, and both
before the loss forward), so the true peak is nearer `2.5 + 8.8 ≈ 11–13 GiB`; the sum is the
envelope to budget against until `peak_reserved_gib_*` says otherwise.

| | 80 GB (target) | 40 GB (fallback) |
|---|---|---|
| vLLM E4B pre-allocation | `0.50 × 80 = 40 GiB` | `0.50 × 40 = 20 GiB` |
| room left for the trainer (card − server − ~2 GiB for two CUDA contexts) | `80 − 40 − 2 ≈ 38 GiB` | `40 − 20 − 1 ≈ 19 GiB` |
| trainer envelope (GRPO, conservative) | 19.7 GiB | 19.7 GiB |
| **headroom** (room − envelope) | **`38 − 19.7 ≈ 18 GiB`** | **`19 − 19.7 ≈ 0` → `smoke.py vram` WARNS; the fallback needs the escape hatches below from the start** |

**Why the old config was retired.** Exp3 measured `per_device 64 × gas 2` **without** checkpointing
at **~67 GB** for the GRPO step on an A100-80GB with no vLLM beside it, and a DPO `16 × 1` without
checkpointing **OOM'd at 78.5 / 80 GB**
([Exp3 CHANGELOG_TRAINER.md:327–345](../Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md)). Exp4 shipped
that same `64 × 2` (ckpt off) beside a 20–40 GiB server pre-allocation, which fits on neither card.
`16 × 8` + checkpointing keeps the 128-completion generation batch and the 16 prompts/step match
(§ Hyperparameters) while cutting the loss-side tensors 4× and the activations to checkpoints.

**Escape hatches, in order** — none is in the arm name, so stamp `run_metadata.json` with the
iteration each started at (a one-arm speedup makes the cost multiplier non-comparable):
1. `LOOKAHEAD_SUB_BATCH_SIZE` 64→32 (−2.2 GiB; the loop does this itself on OOM, stickily);
2. `CONVERSATION_BATCH_SIZE` 64→32 (−2.2 GiB on the generation pass and PTO's branch sampling);
3. `TRAIN_BATCH_SIZE` 16→8 with `gas` 8→16 — same 128-completion generation batch, same 16
   prompts/step, loss forward halves (−1–2 GiB);
4. grader E4B→E2B (frees 12 GiB on 80 GB / 6 GiB on 40 GB — a science change: a new arm name).

⚠ **`max_model_len` is NOT an escape hatch** — lowering it below 16384 reintroduces the
biased-missingness hazard above. ⚠ Nor is `gpu_memory_utilization` below the weights + a usable
KV pool: the server either fails to start (no KV memory) or serves with near-zero concurrency.
⚠ Verify the real weights figure from the vLLM startup log at the Phase 1 gate rather than
trusting the arithmetic.

**`QUICK_TEST` IS the VRAM rehearsal — it keeps every per-forward shape it can.** It shrinks only
counts (G=4 / M=3, 8 conversations, 2 iterations) and leaves `TRAIN_BATCH_SIZE`, `gas`,
`MAX_COMPLETION_LENGTH`, `THERAPIST_MAX_INPUT_TOKENS` and `LOOKAHEAD_SUB_BATCH_SIZE` at the real
values, so the loss forward, GRPO's 128-completion generate and the K=5 look-ahead are measured at
the real arm's shape. The one term it does NOT rehearse is the conversation pass:
`CONVERSATION_BATCH_SIZE` drops to 8 because only 8 conversations exist (cell 1 of both notebooks),
so the `64 × 2,248 × 32 KiB ≈ 4.4 GiB` generation / PTO branch-sampling KV term stays arithmetic
(scale the rehearsal's generate-phase peak ×8). **Read the per-phase peaks from
`iteration_metadata.json`** — the same flat keys in both trainers, `peak_reserved_gib_<phase>` (+
`peak_allocated_gib_<phase>`): GRPO `generate` / `train` (the look-ahead runs inside `train`) /
`eval_generate` on the post-loop pass; PTO `generate` / `build` (the look-ahead runs inside
`build`) / `train` (the DPO step) / `eval_generate`; `generate` is absent on a mid-training resume
— after the rehearsal, and only then commit 10 iterations × 4 arms. A rehearsal that shrank the
per-forward shapes would measure nothing.

**Local RTX 5070 Ti (12 GB)** is for smoke tests and generate-only passes ONLY. It cannot host the
E4B judge (14.89 GiB of weights > `0.85 × 12 = 10.2 GiB`), so scoring runs on the GPU host too. ⚠ **An over-budget
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

**The Phase 2 gate — prompt length on real Exp4 conversations.**
`tools.oracle_sanity.prompt_length_report` builds the real oracle prompt around each transcript,
counts it in the served model's tokenizer and reports the distribution against `max_model_len`;
`Run_Eval.ipynb` § 8 runs it over every conversation on disk after scoring
(`eda_analysis.scoring.gather_transcripts` → `prompt_length_gate` → `check_prompt_length_gate`:
hard-fails on any prompt over the cap, notes headroom under 1.25×, and REFUSES to pass when the
report function is absent — an undecidable gate must not read as a green one). Contract in
§ Module contract, `code/tools/oracle_sanity.py`.

## Hyperparameters (matched across methods, as in Exp3)

`MCL=12`, K ∈ {0, 5}, `NUM_CONVERSATIONS_PER_ITER=96`, PTO's `M` = GRPO's `G` = 8, matched
generation temperatures. `DPO_BETA=0.1` is the DPO loss temperature, **not** GRPO's KL β.

**`NUM_ITERATIONS=10`, matched** (2026-08-27; Exp3 ran GRPO 6 / PTO 8, unmatched). Not in the arm
name, so extending an arm later is free: resume continues from iteration 11, and its generation
pass finds the final-eval `model_iter_10` CSVs already on disk and reuses them.

**Patient call policy matched: 90 s per attempt × 8 retries in BOTH notebooks** (2026-08-27; the
GRPO notebook briefly said 60 × 12 — a per-method timeout asymmetry is a method-confounded freeze
probability). 90 s carries headroom because the local vLLM server QUEUES requests and queue wait
counts against the attempt; if the mini-arm shows timeout storms under saturation, raise it in
both notebooks together.

**`EPOCHS_PER_ITERATION=1`, matched — and 1 is the only value at which the match is real.** A GRPO
"epoch" re-SAMPLES G fresh completions per prompt and re-grades them all (2 epochs = twice the
reward-side work, the second pass on partially-updated weights), while a DPO epoch re-treads the
SAME fixed pairs. Exp3 ran both at 2, which quietly gave GRPO double the data generation per
iteration; Exp4 sets 1 so "one pass over data produced by this iteration's policy" holds for both
methods. Raise `NUM_ITERATIONS`, never epochs, for more updates.

⚠ **GRPO's `TRAIN_BATCH_SIZE=16 × GRADIENT_ACCUMULATION_STEPS=8` (was `64 × 2`) exists for the
prompts/step match AND the VRAM budget, never for gradient scale.** `per_device_train_batch_size`
counts **completions**, not prompts: unique prompts/step = `(16 / 8) × 8 = 16`, matched to PTO's
`2 × 8 = 16` pairs. TRL still generates ONE batch of `16 × 8 = 128` completions per optimizer step
(`steps_per_generation` defaults to `gas`, installed trl `grpo_config.py` ~:909–911), so the
reward-side work per step is unchanged and only the loss-side tensors shrink 4×. On the pinned
trl 1.4.0, `gas` changes are gradient-scale-neutral: trl bypasses transformers' `training_step`
scaling (non-None `compute_loss_func` sentinel, installed trl `grpo_trainer.py` ~:652–657) and
divides the loss exactly once by `current_gradient_accumulation_steps` (`_compute_loss`,
~:2568–2570; the same division at ~:2351 is `compute_liger_loss`, the liger path). The earlier
"net scale is 1/gas², halving gas doubles the gradient" claim was measured on Exp3's stack and is
FALSE here — re-verify those two line references on any trl bump. `EVAL_BATCH_SIZE=16` to match.

**Gradient checkpointing ON in both trainers** (GRPO had it off). The `64 × 2` + ckpt-off shape
measured **~67 GB** for the GRPO step on Exp3's A100-80GB with no vLLM beside it
([Exp3 CHANGELOG_TRAINER.md:327–345](../Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md)); Exp4 puts a
40 GiB server on the same card. Arithmetic in § VRAM budget.

**Dropout OFF in both methods: `LORA_DROPOUT=0.0` and `disable_dropout=True` on both TRL configs.**
A correctness knob, not a memory one. Both losses compare the policy's log-probs against log-probs
from another forward — GRPO's clipped ratio against `old_per_token_logps` and its KL against the
reference; DPO's implicit reward `log π − log π_ref` — and with dropout active the two forwards see
different masks, so the ratio / margin carries dropout noise unrelated to the update. TRL's
`DPOConfig` already defaults `disable_dropout=True` for that reason; `GRPOConfig` ships the same flag
defaulting to False (installed trl `dpo_config.py:42`, `grpo_config.py:42`), so it is set
explicitly. The knob lives ONCE, on `core.config.TrainingConfigBase.disable_dropout`, fed by the
`DISABLE_DROPOUT` cell-1 global of BOTH notebooks and recorded in both methods' `run_metadata.json`
/ `iteration_metadata.json` — so the two arms cannot silently diverge on it. Exp3 ran LoRA dropout
0.05 on both; Exp4 turns it off on both so the change is method-symmetric. A pre-data science
change ([history/CHANGELOG.md](history/CHANGELOG.md)).

**`QUICK_TEST` is the VRAM and resume rehearsal, so it keeps every per-forward SHAPE.** It shrinks
only counts — `NUM_GENERATIONS` 8→4 (GRPO) / `NUM_BRANCHES_PER_TURN` 8→3 (PTO), 8 conversations,
2 iterations, `SAVE_STEPS` 10→2 **in BOTH notebooks** (so the rehearsal's few optimizer steps
still produce a mid-iteration `checkpoint-*` to kill and resume on) — and leaves
`TRAIN_BATCH_SIZE`, `gas`, `MAX_COMPLETION_LENGTH`, `THERAPIST_MAX_INPUT_TOKENS` and
`LOOKAHEAD_SUB_BATCH_SIZE` at the real values, so its training / look-ahead peaks are the real
arm's (`CONVERSATION_BATCH_SIZE` necessarily drops to 8 — see § VRAM budget for the one term that
stays arithmetic). Steps/epoch floor rather than round, so 8 conversations still yield whole
optimizer steps.

⚠ **PTO must pre-cap its DPO prompt.** TRL 1.4.0's `DPOConfig` dropped `max_prompt_length` and caps
prompt+completion with one `max_length` under `truncation_mode='keep_start'` — which slices the
**response** off. Hence `build_truncated_training_prompt` + `max_length = max_prompt +
max_completion + DPO_FRAMING_HEADROOM_TOKENS`. The headroom is load-bearing, not slack: the prompt
budget is BOS-INCLUSIVE (`count_prompt_tokens`), so trl's one BOS is already inside
`max_prompt_tokens`; what the two configured numbers do NOT cover is the EOS `add_eos` appends to
`chosen` / `rejected` (+1, measured) and prompt/completion retokenisation drift at the boundary —
`DPO_FRAMING_HEADROOM_TOKENS` (8) absorbs both, or `keep_start` would still bite the longest pairs.
trl 1.4.0's live DPO tokenisation for a string prompt is `_prepare_dataset → _tokenize →
processing_class(text=prompt)` (one BOS on the BOS-free prompt, measured on the Instruct
tokenizer); `build_dpo_dataset` asserts exactly one leading BOS per prompt, and `validate_config`
REFUSES a PTO config whose `THERAPIST_MAX_INPUT_TOKENS != MAX_ALLOWED_PROMPT_LENGTH` (the branch
candidates are sampled and the pair trained on ONE `build_prompt` budget; `pto_trainer` asserts the
parity again per pair at run time).

## Running off Colab (a GPU server over SSH)

Nothing in Exp4 requires Colab; the Colab-only branches (Drive mount, Colab Secrets,
`COLAB_CODE_DIR`) all fall through cleanly (`core.runtime`'s module docstring carries the full
list). The differences:

- **Install.** `pip install -U vllm` FIRST (≥ 0.19.1 for Gemma 4; it brings its own torch), then
  `pip install -r requirements.txt` (the repo-root pins) on top, then `pip uninstall torchao` —
  the same order the notebooks' install cell uses. That cell refuses to run outside Colab, so do
  this by hand once per environment.
- **Credentials.** Export `HF_TOKEN` (Llama-3.2-1B is gated; `HUGGING_FACE_HUB_TOKEN` /
  `HUGGINGFACE_TOKEN` also work) and, only for a vendor-bound role, `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY`. `core.runtime.get_secret` reads the environment FIRST; no key files needed.
- **Workspace.** `export EXP4_WORKSPACE_ROOT=/abs/path/Exp4_OpenStack` for a job whose cwd is
  elsewhere (a scheduler); otherwise the walk-up from the cwd resolves it. `COLAB_CODE_DIR` in
  the notebooks is ignored off Colab; open them from `code/<method>/`.
- **`data/` lives on the server's own disk** — real directories, not Drive symlinks: create
  `data/{runs,conversations,eval_scores}` before iteration 1. Results reach the repo by
  `rsync` / `rclone` into `G:\My Drive\Thesis_PTO_GRPO\Exp4_OpenStack\data\` (Drive Desktop then
  surfaces them through the local symlinks), or by repointing the local symlinks. `runs/` is the
  large part (adapters, checkpoints); the EDA reads only `conversations/`, `eval_scores/` and the
  per-iteration JSON / JSONL, so sync those first.
- **Run** the notebooks headless (`jupyter nbconvert --execute`) or via a remote kernel. The
  trl-before-torch order is a local-Blackwell concern; the guard is a no-op unless that pair is
  present (`EXP4_SKIP_IMPORT_ORDER_CHECK=1` bypasses it on a card that does not reproduce it).
- **VRAM.** The budget above is per card; `describe_environment()['gpu_total_gib']` stamps which
  one ran. On anything other than 80 / 40 GiB redo the arithmetic before `serve_roles`.

## Vocabulary

PTO is the framework; DPO is the loss. GRPO has no preference data — only prompts. **Never** call
GRPO data "pref data".

## Status

**Code complete, locally gated, and through the 2026-09-02 pre-run review (below). Nothing has been
trained; no `data/` exists yet.**

| Phase | Gate | State |
|---|---|---|
| 0 Scaffold | `smoke.py naming` — arm names round-trip through the parser (incl. the therapist field + base/Instruct tag distinctness + sanctioned shared tags) | ✅ 32 checks |
| — | `smoke.py config` · `convs` · `vram` | ✅ 29 · 29 · 23 checks (+ 1 vram WARNING on the 40 GB fallback card; final gate, 2026-09-03) |
| — | `smoke.py stopgen` · `dpo` · `grpo` — real TRL steps on the local 12 GB card | ✅ 3 · 7 · 6 checks |
| — | `smoke.py resume` · `prompts` — mid-training resume keeps the iteration-start reference and reloads the trained `default` through the trainers' real restore helpers; THE PROMPT RULE + drop-oldest truncation (system-led and system-less) on both therapist tokenizers | ✅ 13 · 30 checks |
| 5 EDA | `_selfcheck` (full); every family renders on an empty lake (`render_results.py`: 4 rendered, 0 failed) | ✅ 14 passed, 0 failed, 4 skipped (no arms on disk) |
| 2 Oracle path | request carries a real `json_schema`; validation ladder accepts a good answer, **rejects a short array and prose**; aggregation is the unweighted mean across rubrics | ✅ vs `tools/fake_oracle_server.py` |
| 2 Sanity gate | passes a healthy grader, **fails a degenerate one** (exit 1) | ✅ both directions |
| 2 Generation | base policy + patient endpoint → `pers<PID>.csv` → reader round-trip → oracle score | ✅ full loop, local GPU |
| 1 Serving | `smoke.py roles` on Colab — chat + json_schema per binding, **no thinking tokens**, kill→restart, real weights GiB, the KV-cache-tokens line | ⬜ needs Colab + vLLM |
| 2 Real grader | 96-conv base pass vs the real Gemma patient; full `oracle_sanity` against the real Gemma oracle; `Run_Eval` § 8 prompt-length gate | ⬜ |
| 3 GRPO | `QUICK_TEST` rehearsal trains, is killed mid-training on purpose and resumes; `generations.jsonl` valid; prompts/step = 16; `peak_reserved_gib_*` read | ⬜ |
| 4 PTO | `QUICK_TEST` rehearsal trains; `pairs.csv` / `_progress.json` resume semantics verified; peak memory read | ⬜ |
| 6 First real arm | one real-config iteration read for memory / latency / wall-clock, THEN the full GRPO K=0 arm on Colab, $0 API | ⬜ |

Everything runnable without Colab is green: **172 smoke checks** (`32 + 29 + 29 + 23 + 13 + 30 + 3 +
7 + 6`, GPU parts included, plus the one deliberate `vram` WARNING for the 40 GB fallback card;
`serve` / `roles` skip without vLLM on PATH) plus the EDA self-check. `dpo` runs a real `DPOTrainer` and `grpo` a real `GRPOTrainer` step whose completion
lengths come back well under the cap, which is the anti-degeneracy stack working rather than merely
wired up.

### The 2026-09-02 pre-run review (blockers + should-fixes, applied while no data exists)

Four read-only reviewers + owner spot-checks over the whole tree, then one agent per layer
applying the fixes concurrently. One line each; the narrative, the 80 GB decision and the
science-change rationale are in [history/CHANGELOG.md](history/CHANGELOG.md). Everything is
pre-data, so nothing had to be re-run — which is exactly why it was done now.

- **Target card is the Colab A100 80 GB** (40 GB fallback). § VRAM budget rewritten with the
  arithmetic; `describe_environment` records `gpu_total_gib` + `vllm_version` + package versions.
- **GRPO `64 × 2` (ckpt off) → `16 × 8` + gradient checkpointing; `EVAL_BATCH_SIZE=16`.** The old
  shape measured ~67 GB in Exp3 with no vLLM beside it.
- **Dropout off in both methods** (`LORA_DROPOUT=0.0`, `disable_dropout=True`) — method-symmetric.
- **THE PROMPT RULE** (`core.policy`): rendered text never carries BOS, every tokenization adds
  exactly one — fixed a double BOS on Instruct / no BOS on base.
- **Drop-oldest generation truncation** (`core.policy.build_prompt`) replaces token-level left
  truncation at serve time: the system prompt survives past utterance ~24, and training prompts
  are byte-identical to what the policy generated from. Science change, both methods, both K.
- **Look-ahead simulator failures are NOT graded** (`NOT_GRADED_STOP_REASONS`; `score=None`,
  `not_graded_reason`), gate-counted; `SESSION ENDED` candidates graded on the seed only.
- **Patient + oracle paths:** an empty patient reply is retried; HTTP 4xx (except 408/429) raise
  immediately on both paths instead of burning the whole retry budget.
- **Timing:** `log_training_progress` at every checkpoint save + `finalize_training` — a preempted
  training phase now leaves its partial wall-clock on the cost record.
- **Resume:** both trainers reload the adapter via `load_adapter("default")` (the previous path
  re-anchored the reference); `smoke.py resume` pins it.
- **`QUICK_TEST` is a real rehearsal** (shapes kept, counts shrunk: G=4 / M=3, 8 conversations,
  2 iterations); per-phase peak memory → `iteration_metadata.json`; steps/epoch floor.
- **PTO:** samples branches from the exact DPO training prompt (asserted); BOS rule on the DPO
  tokenization path; incremental EDA flush + slimmer `_progress.json`; chunk halve-and-retry;
  metadata written before the adapter.
- **Tools:** `roles.DEFAULT_SERVE_UTIL` + `default_serve_util()` (one table for the trainer
  notebooks — asserted in their serve cells — and smoke; `Run_Eval` keeps its own idle-GPU 0.85);
  `smoke.py vram` for 80 + 40 GB; the thinking gate; vLLM double-launch registry, 1800 s
  startup timeout, KV-cache-tokens report; `fake_oracle_server` serves E4B;
  `oracle_sanity.prompt_length_report`; smoke `resume` + BOS checks.
- **Install cell** pip-checks, probes vLLM and RAISES to stop after installing (a run-all cannot
  continue on a half-installed kernel).
- **Scoring can run:** `Run_Eval.ipynb` gained the Colab mount preamble, E4B comments, an
  80 GB-accurate `SERVE_GPU_MEMORY_UTILIZATION` note and the prompt-length gate (§ 8);
  `scoring.py` gained `gather_transcripts` / `prompt_length_gate` / `check_prompt_length_gate`;
  `eda/` is pushed to Drive with `code/`.
- **Docs:** this spec's module contract carries the new helpers; `history/CHANGELOG.md` created;
  the root `CLAUDE.md` Exp4 paragraph names the 80 GB target.

Every claim above was reconciled against the modules by the gate pass that closed the review
(2026-09-02, [history/CHANGELOG.md](history/CHANGELOG.md) § the gate pass): the cross-file
requests the fix agents could not apply themselves were applied there (`prompt_overflow` landed;
`EDARecorder.append_to_disk` / `rewrite`; the `prompt_length_report` signature the scoring gate
calls; a PTO config with unequal prompt budgets is refused; the install cells are byte-identical
again; the serve cells assert the fraction against `roles.DEFAULT_SERVE_UTIL`), and the module
contract now describes what the modules do.

### The 2026-09-03 review-repair round

Four adversarial reviewers over the 2026-09-02 batch, then one repair agent per layer. Still
pre-data. The full list is in [history/CHANGELOG.md](history/CHANGELOG.md) § the review-repair
round; what changed in this spec: `core.timing.begin_training_phase` (the training ledger is per
ATTEMPT), `core.conversations.has_session_end` (one keyword matcher), look-ahead runtime errors
halve-and-retry locally, PTO's `pairs.csv` marker is written only after every EDA row is on disk,
PTO's peak-memory keys took GRPO's flat shape and its `run_metadata.json` gained the `runtime`
block, `disable_dropout` moved to `TrainingConfigBase` (`DISABLE_DROPOUT` in both notebooks),
`QUICK_TEST` sets `SAVE_STEPS=2` in both notebooks, the trainer envelope is ONE arithmetic
(`2.5 + 8.8 + 4.4 + 4.0 = 19.7 GiB`, 40 GB headroom ≈ 0 and `smoke.py vram` warns), `ensure_alive`'s
real call sites, `smoke.py roles` 24 / 28, the scoring gate's served-cap precedence, and the
base-arm terminator asymmetry recorded above as deliberately unchanged.

### The 2026-08-27 decision round (pre-Colab review with Lior)

Decisions: grader = **E4B only** for now (a future grader swap is a NEW arm by construction — role
tags are always encoded — so nothing needs re-running); **therapist became selectable** (base +
Instruct, `_Th{tag}` appended to the grammar while zero folders existed); Instruct arms use the
**native Llama-3 template + `<|eot_id|>` token-id stopping** (base arms keep ChatML + string
stops); `NUM_ITERATIONS=10` matched; patient timeout matched at 90 s × 8; the GRPO notebook gained
the QUICK_TEST block (G→4 → disjoint `_G4_` folder); `LOOKAHEAD_SUB_BATCH_SIZE=64` and
TensorBoard-only logging stand. Fallout landed with the change: `roles._slugify` no longer strips
`-Instruct` (base/Instruct would have shared a tag), every chat-template render pins
`date_string=CHAT_TEMPLATE_DATE` (the Llama-3.2 template otherwise interpolates TODAY's date —
prompts would drift across resume days), and `core.conversations`' token-budget estimators now
MEASURE per-role wrapper overheads on the live template instead of hardcoding the ChatML wrapper
(which would have over-billed every Instruct turn ~2×). The Instruct decode path is verified on the
local GPU end-to-end (native template kept, eos list `[eot, eom, start_header]`, clean batched
generations with empty stop strings, truncation budget respected).

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
Gemma-4-E4B can actually measure MI quality. That is what the real `oracle_sanity` run on Colab is
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

**The gate ladder, on Colab (A100 80 GB), in this order.** Each rung is cheap next to the one after
it; do not jump to a full arm.

0. **Before Colab:** push `code/` AND `eda/` to Drive (additively; never `data/`) and add the
   `huggingface` Colab secret (Llama-3.2-1B is gated; **Gemma 4 is NOT** — Apache 2.0, no
   click-through). Run the install cell: on a fresh runtime it installs vLLM first (≥ 0.19.1),
   layers the pinned stack on top, drops Colab's torchao, and **raises to stop** — restart, re-run
   the mount cell, continue; on a warm runtime it prints one line and skips. It refuses to install
   anywhere but Colab, so opening a notebook locally cannot write into the repo `.venv`.
1. **`smoke.py roles`** — chat + `json_schema` per binding, **no thinking tokens** on the wire,
   kill→restart. Read the **measured weights line** (14.89 GiB E4B / 9.54 GiB E2B) and the
   **KV cache tokens** line (pool ÷ ~4k median prompt = concurrent oracle calls the server holds).
2. **Full `oracle_sanity` on E4B** (12 transcripts, both hard gates, `--quick` is for vendor APIs
   only). E2B only if E4B fails the gate.
3. **`QUICK_TEST=True` rehearsal, both notebooks** (G=4 / M=3, 8 conversations, 2 iterations, real
   per-forward shapes; lands in a disjoint `_G4_` / `_M3_` folder). **Kill the kernel on purpose
   during iteration 1's training phase, after a `checkpoint-*` save, and resume:** `smoke.py
   resume` pins the reference-anchoring semantics offline, the rehearsal proves them on the real
   loop — expect `n_sessions_production == 2`, the partial `training_s` lines summing to the phase,
   `resume_from_checkpoint` picking the valid checkpoint, and the iteration-start adapter as the
   reference. Read **`peak_reserved_gib_*` per phase** from `iteration_metadata.json`.
4. **One real-config iteration** (`QUICK_TEST=False`; `NUM_ITERATIONS` is not in the arm name, so
   the arm simply resumes into the full run later). Read: peak memory again (96 conversations now),
   the server's KV-cache-tokens line against the realised concurrency, oracle **p95 latency and the
   timeout count** (`oracle/success_rate`, `lookahead/not_graded_frac`, `reward/graded_frac`), the
   `trunc <n>/<B>` field on the batch lines, and the phase wall-clocks from `timing_sessions.jsonl`.
   **Extrapolate before committing 10 iterations × 4 arms.** Per iteration, both methods, K=0:
   `96 convs × 19 branch points × 8 candidates × 2 rubrics ≈ 29k oracle calls`
   (`(50 − 12) / 2 = 19` patient turns at or past MCL=12 in a 50-utterance conversation). K=5 adds
   `96 × 19 × 8 × 3 patient turns ≈ 44k patient calls` per iteration (5 extra utterances = 3
   patient + 2 therapist turns per candidate) on top of the oracle calls.
5. **Then** the first real arm: GRPO K=0, Instruct therapist. Score it with `Run_Eval.ipynb` on the
   same GPU; its § 8 prompt-length gate is the Phase 2 measurement (the 16384 cap stands on Exp3's
   gpt-4o-mini patient until then).

⚠ **What is still unverified because nothing has run on Colab:** the vLLM build's behaviour
(`smoke.py roles` gates serving, thinking-off on the wire, and `json_schema` handling — the
`enable_thinking` key matches the official recipe and thinking is off by default, so this is
belt-and-braces rather than a coin flip); the trainer envelope (`2.5 + 8.8 + 4.4 + 4.0 = 19.7 GiB`
conservative beside the 40 GiB E4B server on 80 GB is arithmetic, not a measurement — the
rehearsal measures it); and
whether either Gemma actually measures MI quality — a grader can honour the schema perfectly and
return near-constant scores, which is exactly what `oracle_sanity`'s degeneracy gate exists for.
