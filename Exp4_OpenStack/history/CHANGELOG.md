# Exp4_OpenStack — dated history

The only place in Exp4 where a date belongs. [CLAUDE.md](../CLAUDE.md) describes how things *are*
(plus its Status / decision-round sections, which point here); [README.md](../README.md) maps the
folder; [eda/README.md](../eda/README.md) explains the analysis mechanics. Entries are append-only,
newest first. Earlier rounds (2026-08-26 audit, 2026-08-27 decision round) are summarised in
CLAUDE.md § Status and were not moved — this file starts with the pre-run review.

---

## 2026-09-03 — the review-repair round

Four adversarial reviewers, each over the whole 2026-09-02 batch (the pre-run review + its gate
pass), looking for anything a reviewer could break or a doc could mislead on; then one repair
agent per layer (core · PTO + config · GRPO + tools · scoring + docs), concurrent, and a gate agent
reconciling afterwards. Still pre-data: nothing has trained, so every fix is free.

**What the reviewers confirmed** (no change needed): the mid-training-resume reference anchoring
(`resolve_start_state` case B + `load_adapter("default")`); THE PROMPT RULE and the drop-oldest
truncation on both tokenizers; `rewards_for_trl`'s group-mean substitution for an ungraded
candidate; the `prompt_overflow` stop reason and its counters; the loop-keyed client cache; the
`/tokenize` counting path of the Phase 2 gate; the arm-name grammar and the therapist tag.

**Issues found and fixed** (one line each; the owning module in brackets)
- **Training ledger per attempt** [`core.timing`]: `begin_training_phase(iter_dir)` resets the
  partial-time ledger at every training attempt start — an in-kernel re-run of a crashed iteration
  is a new attempt, and without the reset its first `on_save` increment inherited the previous
  attempt's offset (a clamped-to-zero partial line and an undercounted phase).
- **PTO marker after the EDA rows** [`pto_trainer`]: `pairs.csv` is written only after every
  `generations.jsonl` row of the build is on disk, and the trainer RAISES if a row is missing —
  the marker can no longer claim a build whose EDA capture is short.
- **Look-ahead runtime-error halving** [`core.lookahead`]: a non-OOM runtime error now halves the
  chunk and retries locally down to size 1; only the sim that fails alone freezes as `gpu_error`
  (the old path froze the whole chunk on one bad sim).
- **Session-end matcher unification** [`core.conversations`]: `has_session_end` is the one
  presence test, the same regex `handle_session_end` / `split_session_end` cut on — the loop, the
  look-ahead, the reward fn and the PTO grower had three ways of asking.
- **System-less prompt fit** [`core.policy`]: `_fit_messages` budgets a message list with no
  system message correctly instead of charging a phantom system overhead.
- **PTO metadata: environment + dropout + peak keys** [`pto_trainer`, `core.config`]: PTO's
  `run_metadata.json` gained the `runtime` block (card GiB + source, vLLM version) GRPO already
  wrote; `disable_dropout` moved to `TrainingConfigBase` behind a `DISABLE_DROPOUT` cell-1 global
  recorded for both methods; PTO's nested `peak_gpu_gib → {generation, pref_build, dpo}` became
  GRPO's flat `peak_reserved_gib_<phase>` / `peak_allocated_gib_<phase>` with phases
  `generate` / `build` / `train` / `eval_generate` (`generate` omitted on a mid-training resume).
- **PTO `SAVE_STEPS=2` under `QUICK_TEST`** [`train_pto.ipynb`]: matches GRPO, so the PTO rehearsal
  also produces a checkpoint to kill on. PTO also probes `ensure_alive` before the build and before
  DPO (`run_one_iteration` / `run_final_eval` take `server_handles=` / `client_factory=`).
- **Envelope arithmetic unified at 19.7 GiB** [`smoke.py`, `CLAUDE.md`]: the trainer envelope is
  ONE arithmetic everywhere, `2.5 + 8.8 + 4.4 + 4.0 = 19.7 GiB` (a plan, unmeasured) → ~38 GiB of
  room beside the 40 GiB server on 80 GB, ≈ 0 headroom on 40 GB, where `smoke.py vram` now WARNS;
  the docs had "≈ 20 GiB", "~19 GiB headroom" and an unstated sum.
- **Smoke `resume` tautology** [`smoke.py`]: the resume checks re-implemented the restore step they
  claimed to test; they now call the trainers' real restore helpers.
- **`vram 0.25` leftover** [`smoke.py`, `CLAUDE.md`]: the last "share the card at 0.25" wording —
  the fraction at which E4B cannot load — replaced by `roles.default_serve_util` (E4B 0.50, E2B
  0.35).
- **Roles sanctioned shared tags** [`roles.py`]: `_SANCTIONED_SHARED_TAGS` names the model ids
  allowed to share a tag on purpose, so the collision guard cannot be silenced by accident.
- **`vllm_serve` wedged / served-cap fixes** [`tools/vllm_serve.py`]: `ensure_alive` treats a live
  pid whose endpoint stopped answering as wedged (kill + relaunch) instead of waiting on it, and is
  called only at the trainers' phase boundaries (GRPO generate + train; PTO loop-top / build / DPO),
  never from a core retry path; `adopt_if_running`'s `requested` branch reads the served
  `max_model_len` back instead of asserting the requested one.
- **Scoring gate: served-cap precedence + vacuous-pass rendering** [`eda_analysis.scoring`,
  `Run_Eval.ipynb`]: `prompt_length_gate`'s `max_model_len` became optional and is forwarded only
  when NO server is used — with a `base_url` the cap the server reports on `/tokenize` wins, and a
  differing literal is printed as a note (before, `SERVE_MAX_MODEL_LEN` silently overrode an adopted
  server's real cap, which is exactly the case the gate exists for); the zero-transcript dict now
  carries `gate` / `questionnaire_ids` / `method: "none"` / `measured: False`, so an empty lake
  renders as "nothing measured (vacuous pass)" instead of `VERDICT: FAIL` followed by
  `GATE PASSED`; `Run_Eval` § 8 passes the literal only without a server and quotes the served cap.
  `oracle_sanity.prompt_length_report` uses the SMALLER of an explicit cap and the served cap, with
  a note, and its formatter renders `n_transcripts == 0` as the vacuous pass.
- **Doc drift** [`CLAUDE.md`, `eda/README.md`, `Run_Eval.ipynb`]: `ensure_alive` "at every phase
  boundary and from the retry path" → the real sites; "restart the serve cell" surfacing that
  never existed; `roles` 24 → 24 adopted / 28 launched; the trl `dpo_trainer.py:332–346` sites are
  `DataCollatorForVisionPreference`, not a "conversational branch"; `Run_Eval`'s comment named a
  `VLLM_GPU_MEM_UTIL` knob that exists in neither notebook; the flat peak keys + `runtime` block
  documented for both methods.

**Deliberately NOT changed** (recorded so nobody "fixes" them by accident)
- **Base-arm terminator asymmetry** (`_ThL1B` only): the ChatML policy emits the 6-BPE
  `<|im_end|>` string, `clean_completion` strips it, DPO's `add_eos` then appends the tokenizer's
  `<|end_of_text|>` to `chosen` / `rejected` while GRPO trains on the raw emitted ids. Instruct arms
  are consistent on `<|eot_id|>`. Documented in CLAUDE.md § `core/policy.py`; any fix is a
  one-method science change on the non-default variant.
- **`_progress.json` pair prompts**: the mid-build checkpoint still stores each pair's prompt text
  in full (the slimmer-progress change of 2026-09-02 dropped only the EDA rows). Left as is this
  round unless the PTO agent took it; the cost is disk, not correctness, and the reload path
  compares the fingerprint sidecar either way.

**Verification (scoring + docs agent):** `py_compile scoring.py`; `nbformat.validate` on
`Run_Eval.ipynb`; a probe of the gate on `[]` (vacuous: `gate.passed`, formatter renders "nothing
measured (vacuous pass)"), on two transcripts offline (`argument` / `assumed` cap sources) and
against `tools/fake_oracle_server.py` serving `max_model_len=4096` on `/tokenize` (the served
4096 wins over an explicit 16384, with the note; no note when no literal is passed);
`eda_analysis._selfcheck --fast`; leftover greps of the owned docs for `0.25` / `retry path` /
`24 checks` / `18.2` / `20 GiB` (each survivor is the `ServeSpec` dataclass default, the
40 GB-card server pre-allocation, or historical wording, on purpose).

**Final gate (owner, after all five repair agents landed):** `py_compile` 37 files / 0 failures;
`nbformat.validate` on the three notebooks; the two trainers' install cells byte-identical
(7,713 chars); no torch / trl / peft / transformers / datasets import before the serve cell and
the model cell imports `trl → datasets → torch` in both trainers; `smoke.py all` exit 0 =
**172 checks** (naming 32, config 29, convs 29, vram 23 + 1 WARNING for the 40 GB fallback,
resume 13, prompts 30, stopgen 3, dpo 7, grpo 6; `serve` / `roles` skip without vLLM);
`_selfcheck` (full) 14 passed / 4 skipped (no arms on disk) / 0 failed; `render_results.py`
4 rendered / 0 failed on the empty lake with no `gemma4E2B`-named artifact left; the leftover
greps (`per_device=64`, `64 x 2`, `peak_gpu_gib`, `LOW (0.25)`) hit only historical wording.

---

## 2026-09-02 — the gate pass (closing the pre-run review)

One agent over the whole tree after the six fix agents landed, with two jobs: apply the cross-file
requests each agent had filed against files it could not touch, and run every gate. Nothing was
redesigned; every change below either reconciles two agents' work or fixes a gate.

**Reconciliations applied**
- `core.lookahead`: a per-item `None` from `generate_therapist_batch` (no prompt could be built)
  now freezes the sim as **`prompt_overflow`** — added to `NOT_GRADED_STOP_REASONS` in
  `lookahead.py` and its stdlib mirror in `recorder.py`, counted in
  `LookaheadState.prompt_overflows`, stamped as `lookahead_prompt_overflows` in both trainers'
  `iteration_metadata.json`, and a `n_la_prompt_overflow` telemetry key in `core.reward`. The
  stale "Truncation is LEFT" docstring in `LookaheadConfig` went with it.
- `core.recorder`: `EDARecorder.jsonl_line` / `append_to_disk(n_already)` / `rewrite(rows)` — the
  per-depth write path PTO needs, owned by the recorder; `pto_trainer._flush_eda_rows` /
  `_resume_eda_rows` / `_reset_eda_rows` are now thin wrappers (their private line-format copy is
  gone). Schema docstring carries `prompt_overflow` and the PTO sampler-row convention.
- `tools.oracle_sanity.prompt_length_report` accepts a **list of transcripts** as well as a
  `conv_dir` (the docs agent's `scoring.prompt_length_gate` was calling it with a list against a
  folder-only signature, and expecting a dict from a dataclass — the Phase 2 gate could not have
  passed on real data); `to_dict()` now carries the flat gate keys (`n_transcripts`,
  `per_questionnaire`, `n_over`, `max_tokens`, `headroom`), `format_prompt_length_report` renders
  the dict too, `scoring.prompt_length_gate` converts the dataclass and forwards `base_url` so
  `Run_Eval` § 8 counts through the server's `/tokenize`.
- `core.config.validate_config` REFUSES a PTO config with
  `therapist_max_input_tokens != max_prompt_tokens` (GRPO keeps the warning).
- `eda_analysis.data.load_generations` gained the `not_graded_reason`, `ended_by_candidate` and
  `lookahead_stop_reason` columns (and `GENERATION_COLUMNS` / dtypes); `eda/README.md` no longer
  calls them raw-JSONL-only.
- Notebooks: the GRPO install cell is byte-identical to PTO's again; PTO's
  `VLLM_STARTUP_TIMEOUT` 900 → 1800 (the `tools.vllm_serve` default); both serve cells assert every
  planned `ServeSpec.gpu_memory_utilization == roles.default_serve_util(model)`; `Run_Eval` § 8
  passes `base_url=JUDGE.base_url`.
- `CLAUDE.md`: every "reconcile" marker replaced by the verified state — the `vllm_serve` contract
  (`spec_source`, 1800 s, the KV-cache / concurrency reports, the `requested`-adoptee refusal),
  the `prompt_length_report` contract, the DPO tokenisation path (installed `dpo_trainer.py:875`)
  and the corrected headroom rationale, the trl loss-division reference (`_compute_loss`
  ~:2568–2570; ~:2351 is the liger path), the real peak-memory key names per trainer, the
  resume note (`restore_default_adapter`), the recorder write paths, the smoke counts, and the
  honest `QUICK_TEST` statement: `CONVERSATION_BATCH_SIZE` drops to 8 in the rehearsal, so the
  conversation-pass KV term is the one that stays arithmetic.

**Gates** — `py_compile` 37 files; `nbformat` on the three notebooks + install-cell identity;
`smoke.py all` 156 checks (naming 29, config 29, convs 27, vram 22, resume 7, prompts 26, stopgen
3, dpo 7, grpo 6; `serve` / `roles` skip locally); `_selfcheck` 14 passed / 4 skipped;
`render_results.py` 4 rendered / 0 failed (the E2B-named artifacts were replaced by E4B-named
ones); the static import-order check (no torch/transformers/peft/trl before the serve cell; model
cell imports trl → datasets → torch); leftover greps (every `64 × 2` / `40 GB` / `E2B` hit is
historical or fallback wording; no `add_special_tokens=False` on a generate path; no
`truncation=True` in `core/policy.py`).

**Not done (deliberately, listed for the next session)** — the optional `smoke.py reward` / `pto`
subcommands (the probes stay in the session scratchpad); `score_pref_candidates(sampler_failed=)`
(PTO's post-hoc `_apply_sampling_failures` is correct, so the reward log line's `n_degenerate`
over-counts sampler failures by design until then); everything that needs Colab + vLLM
(`smoke.py roles`, the QUICK_TEST rehearsal and its `peak_*` numbers, the pgrep recovery path,
vLLM's `/tokenize` shape).

---

## 2026-09-02 — the pre-run review

**Why now.** Nothing has been trained and no `data/` exists, so every science-affecting change is
still free: no arm has to be re-run, no score axis moves. The review was run *before* the first
Colab session for exactly that reason. Four read-only reviewers went over the whole tree
(core layer, trainers, tools, docs/EDA) with owner spot-checks on top; the findings were then
applied concurrently by one agent per layer — core (two agents: `policy` + `conversations`, and
`reward` + `lookahead` + `oracle` + `timing` + `runtime` + `concurrency`), GRPO, PTO, tools, and
docs/EDA. Each agent's structured report (files, finding → fix → where, verification, cross-file
requests) is the source for the lines below; where this file describes another agent's code it
records what that agent was *instructed* to do and reported doing, and the review phase reconciles
that against the modules.

### The 80 GB decision

Exp4 targets the **Colab A100 80 GB**; the 40 GB card is the fallback. The spec had been written
for 40 GB. Two measurements from Exp3's own history forced the change
([Exp3 CHANGELOG_TRAINER.md:327–345](../../Exp3_PTO_GRPO/history/CHANGELOG_TRAINER.md)): the GRPO
step at `per_device 64 × gas 2` without gradient checkpointing sat at **~67 GB** on an A100-80GB
with no vLLM beside it, and a DPO `16 × 1` without checkpointing **OOM'd at 78.5 / 80 GB**. Exp4
had shipped that same GRPO shape next to a 20–40 GiB vLLM pre-allocation, which fits on neither
card. The fix has two halves: the GRPO config moved to `16 × 8` + checkpointing (same
128-completion generation batch, same 16 prompts/step, loss-side tensors 4× smaller), and the
budget was re-derived with the arithmetic shown (CLAUDE.md § VRAM budget: server `0.50 × 80 =
40 GiB` = 14.89 GiB weights + ~22 GiB KV; trainer envelope `2.5 + 8.8 + 4.4 + 4 ≈ 20 GiB`
conservative; ~19 GiB headroom on 80 GB, ~0 on 40 GB). `core.runtime.describe_environment` now
records `gpu_total_gib` and `vllm_version` so the card an arm ran on is in `run_metadata.json`.
The EDA is expected to read a runtime's `peak_reserved_gib_<phase>` from `iteration_metadata.json`
(GRPO agent) rather than trust this arithmetic.

### Science changes (deliberate, pre-data)

These change what the policy sees or what the optimizer optimises. None is in the arm name; all
apply to both methods and both K arms, so the 2×2 stays controlled. Recorded so a future reader
does not mistake them for drift.

1. **Drop-oldest generation truncation keeps the system prompt.** The decode path used
   `truncation=True` under `truncation_side="left"`, so past ~utterance 24 every conversation-loop
   and look-ahead therapist turn was generated from a prompt that started mid-utterance with no
   system message — while the *training* prompt for the same turn dropped whole oldest turns and
   kept it. `core.policy.build_prompt` (message-level, canonical longest-suffix-that-fits) is now
   the ONE function the decode path, `build_truncated_training_prompt` and
   `extract_prompts_from_conversations` all use, so a PTO branch is sampled from byte-identical
   text to the DPO prompt it trains on and a GRPO prompt is what the policy generated from.
   `TRUNCATION_COUNTER` makes the truncation rate a logged number (`trunc <n>/<B>` per batch line;
   trainers log per-phase deltas). Per-item overflow (newest turn alone over budget) returns
   `None` and fails that conversation as a misconfiguration.
2. **THE PROMPT RULE — rendered text never carries a BOS; every tokenization adds exactly one.**
   The Instruct template's text started with `<|begin_of_text|>` and trl's
   `processing_class(text=prompts)` added another (double BOS); the base ChatML template rendered
   none and the serving path tokenized with `add_special_tokens=False` (no BOS at all). The base
   therapist now sees a BOS at serving that it did not before — deliberate: Llama-3.2 base was
   pretrained with BOS, and no Exp4 data existed. The PTO agent was asked to make the DPO
   tokenization path produce `prompt_token_ids(prompt)` (trl's DPO path has both an
   `add_special_tokens=True` and an `=False` branch).
3. **Dropout off in both methods** — `LORA_DROPOUT 0.05 → 0.0`, `disable_dropout=True` on both TRL
   configs. Both losses compare log-probs across two forwards (GRPO: policy vs `old`/reference; DPO:
   policy vs `π_ref`); with dropout active the two see different masks and the ratio carries
   dropout noise unrelated to the update. `DPOConfig` defaults the flag on, `GRPOConfig` off
   (trl 1.4.0), so Exp3 had it asymmetric by default. Exp4 sets it explicitly on both.
4. **Look-ahead simulator failures are not graded.** A frozen sim (`patient_error`, `gpu_error`,
   `parse_error`) used to be sent to the oracle as if the conversation had ended there — a short
   transcript that reads as "the therapist stopped", i.e. a policy judgement about an
   infrastructure fault. Such candidates now get `score=None` + `not_graded_reason`, count as
   FAILURES in the `min_success_ratio` gate, and GRPO substitutes the group mean. Candidates that
   themselves emit `SESSION ENDED` are graded on the seed only (no rollout) and recorded as a
   complete zero-turn rollout at K>0.
5. **GRPO batch shape `64 × 2` → `16 × 8`, checkpointing on, `EVAL_BATCH_SIZE 64 → 16`.** Not a
   gradient-scale change (trl 1.4.0 divides the loss exactly once by `gas`); a memory one. See the
   80 GB decision.

### Findings → fixes, by layer

**Core — policy + conversations** (`core/policy.py`, `core/conversations.py`)
- A1 double / missing BOS → the prompt rule (`tokenizer_adds_bos`, `strip_leading_bos`,
  `render_prompt`, `prompt_token_ids`, `count_prompt_tokens`); budgets are BOS-inclusive.
- A2 token-level generation truncation → `truncate_messages_drop_oldest` / `build_prompt` /
  `TruncationCounter`; `generate_therapist_batch` never token-truncates and can return a per-item
  `None`.
- A3 an empty patient reply was saved as a "reached cap" conversation → retried like `None`, then
  the conversation fails and is regenerated.
- A4 every patient-path exception retried `max_retries × backoff` → HTTP 4xx other than 408/429
  raise immediately with the status and a body excerpt.
- A5 `handle_session_end` indexed the original string by `.upper()` offsets → one
  case-insensitive regex, offsets on the original text; `load_conversation_csv` reads `dtype=str`.
- A6 `num_utterances` counts ADDITIONAL utterances after the scripted opener (total ≤ 50 at 49) —
  documented, not changed.

**Core — reward, look-ahead, oracle, timing, runtime, concurrency**
- B1 freezes graded as complete rollouts → `NOT_GRADED_STOP_REASONS`, `LookaheadResult.graded`,
  `stop_reason` on the record, gate rate = graded / gradable, new TB metrics
  `reward/graded_frac` + `lookahead/not_graded_frac`.
- B2 `SESSION ENDED` candidates graded verbatim and rolled out → `split_session_end` (public;
  delegates to `handle_session_end`), seed-only grading, `ended_by_candidate`.
- B3 training wall-clock logged only when `train()` returned → `log_training_progress` (from
  `on_save`) + `finalize_training`; the trainers were asked to replace their end-of-phase
  `log_session(training_s=...)`.
- B4 non-retryable 4xx retried on the oracle path → `RETRYABLE_4XX`, `is_non_retryable_http_error`.
- B5 no machine-readable reason on `score=None` → `CandidateScore.not_graded_reason` /
  `ended_by_candidate`, written to the EDA candidate dict.
- B6 environment record could not tell 80 GB from 40 GB → `gpu_total_gib` (+ source),
  `vllm_version`, `package_versions`; off-Colab guidance in the module docstring.
- B7 `run_async` thread-loop teardown and `id(loop)` keying → identity-verified cache, asyncgen +
  executor shutdown before close.

**GRPO trainer + notebook** (instructed; reconcile against the module): `TRAIN_BATCH_SIZE 64→16`,
`gas 2→8`, gradient checkpointing on, `EVAL_BATCH_SIZE 16`, `QUICK_TEST` as a real rehearsal
(G=4, 8 conversations, 2 iterations, real per-forward shapes), resume via `load_adapter("default")`
(the previous path re-anchored the reference), `disable_dropout=True` + `LORA_DROPOUT 0.0`,
`num_completions_to_print=4`, steps/epoch floored, install cell pip-check + vLLM probe +
raise-to-stop, per-phase peak-memory metadata, `finalize_training` + `on_save` partial timing,
`TRUNCATION_COUNTER` deltas logged.

**PTO trainer + notebook** (instructed): the same resume fix, an assertion that a branch is sampled
from the exact DPO training prompt, the BOS rule on the DPO tokenization path,
`disable_dropout=True`, incremental EDA flush + a slimmer `_progress.json`, look-ahead chunk
halve-and-retry, `iteration_metadata.json` written before the adapter, `QUICK_TEST` M=3 /
8 conversations, `finalize_training` + `on_save` partial timing.

**Tools** (instructed): `roles.DEFAULT_SERVE_UTIL` + `default_serve_util()` (one table for the
notebooks, `smoke.py roles` and `Run_Eval`); `smoke.py vram` sized for 80 GB and 40 GB; the
thinking gate; a vLLM double-launch registry, 1800 s startup timeout and the KV-cache-tokens
report; `fake_oracle_server` serves E4B; `tools.oracle_sanity.prompt_length_report(...)` (the
Phase 2 measurement); smoke `resume` + BOS checks.

**Docs + EDA** (this round's docs agent — the files it owns)
- D1 scoring could not run anywhere: `Run_Eval.ipynb` cell 2 walked up from `/content` on Colab
  and died on an `ImportError`; `README.md` said `eda/` was local-only; the local card cannot
  host the E4B judge (14.89 GiB > `0.85 × 12 = 10.2 GiB`). → Colab mount + `chdir` preamble
  (`COLAB_EDA_DIR`), a clear `RuntimeError` when no `eda_analysis/` is found, cell-1/setup notes
  that scoring runs on the GPU host, the push rule rewritten (push `code/` AND `eda/`, never
  `data/`), the E2B leftovers in the notebook and in `scoring.py`'s docstrings corrected to
  E4B / `DEFAULT_JUDGE_MODEL`, and the `SERVE_GPU_MEMORY_UTILIZATION` comment made 80 GB-accurate.
- D2 the Phase 2 prompt-length measurement had no home in the scoring path. →
  `scoring.gather_transcripts` / `prompt_length_gate` / `check_prompt_length_gate` +
  `Run_Eval` § 8; the wrapper imports `tools.oracle_sanity.prompt_length_report` lazily,
  keyword-filters against its real signature, and REFUSES to pass when the function is absent or
  the report is undecidable. Signature contract in CLAUDE.md § Module contract.
- D3 CLAUDE.md: VRAM budget rewritten for 80 GB (40 GB fallback) with the arithmetic; the
  Hyperparameters section (16×8, checkpointing, dropout, QUICK_TEST shapes); the module contract
  carries every new core helper; the notebook cell-order contract (install raises; re-run the
  mount cell); the Status table + this review's row set; the gate ladder; a "Running off Colab"
  section.
- D4 this file. D5 `eda/README.md` (scoring on Colab, push rule, partial-timing and peak-memory
  fields, the `not_graded_reason` / `ended_by_candidate` / `stop_reason` keys). D6 the root
  `CLAUDE.md` Exp4 paragraph names the 80 GB target.

### Verification the docs agent ran
`py_compile scoring.py`; `nbformat.validate(Run_Eval.ipynb)`; an offline probe of the gate
(12 checks: absent function → `ImportError`, empty input → vacuous pass, documented shape →
pass, `n_over > 0` → fail with the raise-the-cap message, headroom note, unsupported keyword
dropped with a warning, undecidable report → fail, per-prompt token-list fallback, non-mapping →
`TypeError`, empty-frame column contract); `eda_analysis._selfcheck --fast`; a grep of the owned
docs for `40 GB` / `E2B` / `40 GiB` leftovers (each remaining mention is the fallback card or the
fallback grader, on purpose).
