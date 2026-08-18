"""Generate-only eval pass for ONE PTO_Exp3 model state — no training, no oracle.

Why this exists
---------------
``model_iter_k`` conversations are normally produced as Step 1 of iteration
``k+1``. When a run dies between "iteration k's adapter saved" and "iteration
k+1's Step 1 finished", the adapter exists but has NO eval data — it can never be
scored. That is exactly the PTO LA5 state: adapters for iterations 1..5, but
``model_iter_5_TT0.9_TP0.7/`` is empty because iteration 6 was killed ~1 minute
in (see history/CHANGELOG_EDA.md, 2026-07-11).

This script runs *just* that missing Step 1: load ``iteration_N/adapter``,
simulate ``num_conversations_per_iter`` conversations against the patient
simulator, write them to ``model_iter_N_TT*_TP*/``. No branching, no look-ahead,
no oracle calls, no DPO — so it costs patient API calls + GPU generation only.

Fidelity
--------
Everything comes from the run's own ``run_metadata.json`` (which is
``asdict(PTOConfig)``), so the pass cannot drift from how iterations 0..N-1 were
generated. The two seeds are derived, not guessed:

    model_iter_k  ⇐  iteration k+1's Step 1  ⇒  shuffle seed = patient seed = cfg.seed + k + 1

which is the same ``seed + k + 1`` formula ``eda_analysis.data.persona_order``
replays to recover ``persona_id``. ``--verify-seeds`` PROVES it on this run
before spending anything, by replaying the shuffle for the already-generated
iterations and checking each conversation's patient really is the persona the
formula predicts.

Usage
-----
    python generate_eval_convs.py --iter 5 --verify-seeds --dry-run   # free, no model load
    python generate_eval_convs.py --iter 5                            # the real pass
    python generate_eval_convs.py --iter 5 --batch-size 4             # 12 GB local card

⚠ ``--batch-size`` is a SAFETY setting on the local GPU, not a throughput knob: an over-budget
VRAM request REBOOTS the machine instead of raising ``OutOfMemoryError`` (no traceback, nothing to
catch). Budget ≈ 2.6 GB weights + ≈1.1 GB per concurrent conversation, so batch 4 ≈ 7.1 GB of the
12 GB card and batch 6 ≈ 8.0 GB, while **batch 32 ≈ 38 GB has already rebooted this machine**.
The default is the run's stored ``conversation_batch_size`` (64 ≈ 73 GB — an A100 value), so
ALWAYS pass ``--batch-size`` explicitly when running locally. See CLAUDE.md § Gotchas.

Runs on Colab (mounts Drive, uses Colab Secrets) or locally (walks up for the
key files) with no edits. Generation is resume-safe per conversation CSV, so an
interrupted pass can simply be re-run.
"""

# ─── sm_120 import-order gotcha: trl BEFORE torch, or CUDA init segfaults (exit
#     139) on the local Blackwell GPU. Harmless on Colab. See CLAUDE.md § Gotchas.
import trl  # noqa: F401  (import for side effect: must precede torch)

import argparse
import json
import os
import random
import re
import sys
from dataclasses import fields as dataclass_fields


DEFAULT_EXPERIMENT = "PTO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_M8_PTgreedy"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     BOOTSTRAP (Colab / local, same as the notebook)        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def bootstrap_paths() -> str:
    """Put ``code/`` + ``code/PTO_Exp3/`` on sys.path. Returns the code dir.

    Mirrors the notebook's cell-5 bootstrap so imports resolve identically on
    either host.
    """
    if "google.colab" in sys.modules:  # pragma: no cover - Colab only
        from google.colab import drive
        drive.mount("/content/drive")
        os.chdir("/content/drive/MyDrive/Thesis_PTO_GRPO/Exp3_PTO_GRPO/code/PTO_Exp3")

    here = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(here, ".."))
    for p in (code_dir, here):
        if p not in sys.path:
            sys.path.insert(0, p)
    return code_dir


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     CONFIG — rebuilt from the run's own metadata           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_cfg(experiment_root: str, experiment: str, mode_tag: str, batch_size=None,
             overrides: dict = None):
    """Rebuild the run's ``PTOConfig`` from its ``run_metadata.json``.

    ``run_metadata.json``'s ``config`` block is ``asdict(cfg)``, so it round-trips
    exactly — this is the run's real config, not a re-derivation from notebook
    globals that may have been edited since.

    The stored ``local_outdir`` / ``conv_outdir`` are Colab-absolute
    (``/content/drive/...``); they are recomputed against *this* host's
    experiment root, and the recomputed tail is asserted against the stored one
    so a renamed experiment can never silently misroute output.
    """
    from pto_trainer import PTOConfig

    runs_root = os.path.join(experiment_root, "data", "pto_Exp3", "runs", mode_tag, experiment)
    meta_path = os.path.join(runs_root, "run_metadata.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"No run_metadata.json at {meta_path}\n"
                         f"  (is --experiment / --mode-tag right?)")

    with open(meta_path) as f:
        stored = json.load(f)["config"]

    conv_root = os.path.join(
        experiment_root, "data", "pto_Exp3", "conversations", mode_tag, experiment
    )
    for key, rebuilt in (("local_outdir", runs_root), ("conv_outdir", conv_root)):
        want, got = os.path.basename(stored[key]), os.path.basename(rebuilt)
        if want != got:
            raise SystemExit(f"{key} mismatch: metadata ends in {want!r}, rebuilt ends in {got!r}")
    stored["local_outdir"], stored["conv_outdir"] = runs_root, conv_root

    # PTOConfig is frozen, so every override goes in BEFORE construction.
    if batch_size is not None:
        print(f"  conversation_batch_size: {stored['conversation_batch_size']} -> {batch_size} "
              f"(throughput only — batching does not change sampled outputs)")
        stored["conversation_batch_size"] = batch_size
    for key, val in (overrides or {}).items():
        if val is None:
            continue
        print(f"  ! SCALE OVERRIDE {key}: {stored.get(key)} -> {val}")
        stored[key] = val

    known = {f.name for f in dataclass_fields(PTOConfig)}
    unknown = set(stored) - known
    if unknown:
        print(f"  ! ignoring {len(unknown)} unknown metadata field(s): {sorted(unknown)}")
    return PTOConfig(**{k: v for k, v in stored.items() if k in known})


def seeds_for(cfg, model_iter: int) -> int:
    """The shuffle + patient-API seed for ``model_iter_{model_iter}``.

    ``model_iter_k`` is Step 1 of iteration ``k+1``, which uses
    ``random.Random(cfg.seed + iteration)`` for the persona shuffle and the same
    value as ``patient_api_seed`` — i.e. ``cfg.seed + k + 1``. Matches
    ``eda_analysis.data.persona_order``, so ``persona_id`` recovery keeps working.
    """
    return cfg.seed + model_iter + 1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          SEED VERIFICATION (free — proves the convention on this run)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# The patient states its age verbatim in the opening turn — but only ~2/3 of the
# time, so a conversation that states none is UNRESOLVED, not a mismatch. (Testing
# "does the expected age appear anywhere in the text" instead is far too weak: only
# a handful of distinct ages are spread over the 96 personas, so even a wrong
# shuffle collides ~40% of the time.)
_AGE_RE = re.compile(
    r"(\d{1,3})\s*(?:-|\s)?year[s]?[\s-]*old"      # "61 years old", "61-year-old"
    r"|\bam\s+(\d{1,3})\b"                          # "I am 61"
    r"|\bI'?m\s+(\d{1,3})\b"                        # "I'm 61"
    r"|\b(\d{1,3})\s+years\s+of\s+age"
)


def _stated_age(text: str):
    """The age the patient states in *text*, or None if it states none."""
    m = _AGE_RE.search(text)
    return int(next(g for g in m.groups() if g)) if m else None


def _score_offset(cfg, conv_dirs, n_personas: int, offset: int):
    """(correct, wrong, unresolved) for shuffle seed ``cfg.seed + k + offset``."""
    import pandas as pd
    from system_prompts_builder import get_patient_permutation_characteristics

    correct = wrong = unresolved = 0
    for k, conv_dir, files in conv_dirs:
        order = list(range(n_personas))
        random.Random(cfg.seed + k + offset).shuffle(order)  # order[file_index] = persona_id
        for file_index, fname in files:
            if file_index >= len(order):
                continue
            expected = (get_patient_permutation_characteristics(order[file_index]) or {}).get("age_value")
            patient = pd.read_csv(os.path.join(conv_dir, fname)).query("role == 'patient'")
            said = _stated_age(str(patient.iloc[0]["conversation"])) if not patient.empty else None
            if expected is None or said is None:
                unresolved += 1
            elif int(said) == int(expected):
                correct += 1
            else:
                wrong += 1
    return correct, wrong, unresolved


def verify_seeds(cfg, n_personas: int, max_iters: int = 10) -> bool:
    """Prove ``seed + k + 1`` on THIS run before spending anything.

    Replays the shuffle for every already-generated ``model_iter_k`` and checks
    that each conversation's patient really is the persona the formula predicts,
    by comparing the age the patient states to the canonical persona's
    ``age_value``. Also scores two decoy offsets: if they passed too the test
    would be vacuous, so a decoy that does NOT fail is itself a failure.

    If this holds for iterations 0..N-1, applying the same formula to N is safe.
    """
    print("\n── Verifying the seed convention against already-generated iterations ──")
    suffix = f"_TT{cfg.temperature_therapist_gen}_TP{cfg.temperature_patient}"

    conv_dirs = []
    for k in range(max_iters + 1):
        conv_dir = os.path.join(cfg.conv_outdir, f"model_iter_{k}{suffix}")
        if not os.path.isdir(conv_dir):
            continue
        files = sorted((int(m.group(1)), m.group(0))
                       for m in (re.match(r"conversation_(\d+)\.csv$", f)
                                 for f in os.listdir(conv_dir)) if m)
        if files:
            conv_dirs.append((k, conv_dir, files))

    if not conv_dirs:
        print("  ! no generated iterations found — nothing to verify against")
        return False

    for k, _, files in conv_dirs:
        ok, bad, none = _score_offset(cfg, [(k, _, files)], n_personas, 1)
        print(f"  [{'OK ' if bad == 0 else 'FAIL'}] model_iter_{k}: seed {seeds_for(cfg, k)} — "
              f"{ok} correct, {bad} wrong, {none} state no age  ({len(files)} convs)")

    ok, bad, none = _score_offset(cfg, conv_dirs, n_personas, 1)
    passed = bad == 0 and ok > 0
    print(f"  seed+k+1 : {ok} correct / {bad} wrong  ({none} unresolved)")

    # Decoys: a neighbouring offset must FAIL, or the test proves nothing.
    discriminates = True
    for decoy in (0, 2):
        d_ok, d_bad, _ = _score_offset(cfg, conv_dirs, n_personas, decoy)
        print(f"  seed+k+{decoy} : {d_ok} correct / {d_bad} wrong   (decoy — must fail)")
        discriminates &= d_bad > 0

    if passed and not discriminates:
        print("  => INCONCLUSIVE: a decoy offset also passed, so this test discriminates nothing")
        return False
    print(f"  => seed convention {'CONFIRMED' if passed else 'REJECTED'} for this run")
    return passed


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                  MAIN                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iter", type=int, required=True,
                    help="MODEL STATE to evaluate: loads iteration_N/adapter, writes model_iter_N/")
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="EXPERIMENT_NAME")
    ap.add_argument("--mode-tag", default="full", choices=["full", "quicktest"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override conversation_batch_size (VRAM lever; does not change outputs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit BEFORE loading the model or calling any API")
    ap.add_argument("--verify-seeds", action="store_true",
                    help="prove the seed convention against already-generated iterations (free)")
    # ── Scale-reduction knobs: for a SMOKE TEST of the generation path, not for real eval data.
    #    Any of them makes the output a scaled-down subset, so --conv-dir becomes mandatory
    #    (see the guard below) — a short/partial run must never land in the real eval tree.
    ap.add_argument("--num-convs", type=int, default=None,
                    help="SMOKE: generate only this many conversations (requires --conv-dir)")
    ap.add_argument("--num-utterances", type=int, default=None,
                    help="SMOKE: cap conversation length in utterances (requires --conv-dir)")
    ap.add_argument("--conv-dir", default=None,
                    help="write conversations HERE instead of the canonical model_iter_<N> dir")
    args = ap.parse_args()

    scaled = {"num_conversations_per_iter": args.num_convs,
              "num_utterances_for_data": args.num_utterances}
    if any(v is not None for v in scaled.values()) and not args.conv_dir:
        raise SystemExit(
            "--num-convs / --num-utterances produce a SCALED-DOWN subset, so they require an "
            "explicit --conv-dir.\nWriting a partial or short-conversation set into the real "
            "model_iter_<N>/ would silently corrupt the eval arm: the per-CSV resume would treat "
            "those files as done, so a later full pass would keep them and mix scales/hosts."
        )

    bootstrap_paths()

    from _shared import (
        detect_runtime, init_openai_client, authenticate,
        setup_tokenizer, load_base_model, sync_pad_token, patch_generate,
        setup_permutations,
    )
    from pto_trainer import run_generation_only

    rt = detect_runtime(run_env="auto", experiment_name="Exp3_PTO_GRPO")
    cfg = load_cfg(rt.experiment_root, args.experiment, args.mode_tag, args.batch_size, scaled)

    adapter_dir = os.path.join(cfg.local_outdir, f"iteration_{args.iter}", "adapter")
    conv_dir = args.conv_dir or os.path.join(
        cfg.conv_outdir,
        f"model_iter_{args.iter}_TT{cfg.temperature_therapist_gen}_TP{cfg.temperature_patient}",
    )
    seed = seeds_for(cfg, args.iter)

    if not os.path.isdir(adapter_dir):
        raise SystemExit(f"No adapter at {adapter_dir} — iteration {args.iter} never finished training.")

    existing = (sorted(f for f in os.listdir(conv_dir) if f.startswith("conversation_"))
                if os.path.isdir(conv_dir) else [])

    is_smoke = bool(args.conv_dir)
    print("=" * 70)
    print(f"GENERATE-ONLY {'SMOKE TEST' if is_smoke else 'EVAL PASS'} — model_iter_{args.iter}")
    if is_smoke:
        print("  ** SMOKE: output goes to --conv-dir, NOT the eval tree. Not scoreable data. **")
    print("=" * 70)
    print(f"  Experiment:   {cfg.experiment_name}  [{cfg.mode_tag}]")
    print(f"  Adapter:      {adapter_dir}")
    print(f"  Output:       {conv_dir}")
    print(f"  Seeds:        shuffle = patient_api = {cfg.seed} + {args.iter} + 1 = {seed}")
    print(f"  Convs:        {cfg.num_conversations_per_iter} x {cfg.num_utterances_for_data} utts "
          f"(TT {cfg.temperature_therapist_gen} / TP {cfg.temperature_patient}, "
          f"max {cfg.max_tokens_per_response} tok)")
    print(f"  Patient:      {cfg.patient_model_id}  (concurrency {cfg.patient_api_concurrency})")
    print(f"  Batch size:   {cfg.conversation_batch_size}")
    print(f"  Already on disk: {len(existing)} conversation CSV(s)"
          + ("  → those are SKIPPED (per-CSV resume)" if existing else ""))
    print("  No oracle calls, no branching, no look-ahead, no training.")

    random.seed(cfg.seed)  # therapist persona is drawn from the global RNG — seed as the notebook does
    all_permutations, therapist_system_prompt, therapist_init_utterance = setup_permutations(
        only_expert_therapist=True,
    )

    if args.verify_seeds and not verify_seeds(cfg, len(all_permutations)):
        raise SystemExit("Seed verification did not pass — refusing to generate. "
                         "Investigate before spending; a wrong shuffle silently breaks persona pairing.")

    shuffled = list(all_permutations)
    random.Random(seed).shuffle(shuffled)
    active_permutations = shuffled[: cfg.num_conversations_per_iter]

    if args.dry_run:
        print("\n  [dry-run] stopping before model load — nothing generated, nothing spent.")
        return 0

    if len(existing) >= cfg.num_conversations_per_iter:
        print(f"\n  ✓ Already complete ({len(existing)} convs) — nothing to do.")
        return 0

    client = init_openai_client(rt)
    authenticate(rt, hf=True, wandb_enabled=False)  # HF token: Llama-3.2-1B is gated

    tokenizer = setup_tokenizer(cfg.base_model_id)
    # for_training=True matches the notebook's load; run_generation_only flips
    # use_cache back on and calls .eval(), so the generation path is identical.
    base_policy = load_base_model(cfg.base_model_id, None, for_training=True)
    sync_pad_token(base_policy, tokenizer)

    from peft import PeftModel
    policy = PeftModel.from_pretrained(base_policy, adapter_dir, is_trainable=False)
    patch_generate(policy, tokenizer)  # re-patch: PeftModel wrapping drops the stop_strings binding
    print(f"\n✓ Loaded adapter iteration_{args.iter} onto {cfg.base_model_id} (bf16)")

    _states, gen_time, avg_len = run_generation_only(
        policy=policy, tokenizer=tokenizer, client=client,
        active_permutations=active_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=conv_dir, cfg=cfg,
        patient_api_seed=seed,
    )

    final = sorted(f for f in os.listdir(conv_dir) if f.startswith("conversation_"))
    print("\n" + "=" * 70)
    print(f"  ✓ Done in {gen_time / 60:.1f} min — {len(final)} conversations, avg len {avg_len:.1f}")
    print(f"    {conv_dir}")
    if is_smoke:
        print("    SMOKE run — throwaway output, NOT eval data. The generation path works;")
        print("    run the real pass (no --num-convs/--num-utterances/--conv-dir) to produce it.")
    else:
        print("    Next: score with eda/notebooks/scoring/Run_Eval.ipynb (auto-discovers this arm).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
