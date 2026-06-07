"""Local offline smoke test for the PTO/GRPO fixes — tiny hyperparameters, no OpenAI.

Validates, on the local GPU with the cached base model, that:
  stopgen : stop_strings binds through patch_generate (the GRPO stop mechanism)
  dpo     : long trunk -> build_truncated_training_prompt cap -> DPO trains with
            gradient_checkpointing + precompute_ref_log_probs and does NOT OOM
            (the PTO fix for the first-DPO-step crash)
  grpo    : a real GRPO step with GRPOConfig(generation_kwargs={"stop_strings": ...})
            completes (tokenizer injected through TRL's unwrap path) and the stop fires

Run ONE part per process (each frees VRAM on exit; a crash in one won't lose the rest):
    python _local_smoke.py stopgen
    python _local_smoke.py dpo
    python _local_smoke.py grpo
    python _local_smoke.py all     # runs all three as subprocesses

IMPORTANT (local only): `trl` is imported BEFORE `torch`. On the local Blackwell (sm_120)
importing trl *after* torch segfaults at CUDA init — an environment/init-order quirk, not a
bug in the trainers (Colab is unaffected; the notebooks train there). See CLAUDE.md gotchas.
Training is meant for Colab; this script is only a local sanity check of the fixes.
"""
import os, sys, subprocess
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

# trl FIRST (before torch) — see module docstring.
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer  # noqa: E402
from datasets import Dataset                                    # noqa: E402
import torch                                                    # noqa: E402
from peft import LoraConfig                                     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `_shared` resolves
from _shared import (setup_tokenizer, load_base_model, sync_pad_token,  # noqa: E402
                     patch_generate, build_truncated_training_prompt, turns_to_messages)

MODEL = "meta-llama/Llama-3.2-1B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
SYS = ("You are a motivational interviewing counselor named David. You are empathetic and "
       "help the patient explore ambivalence about change.")
MAXPROMPT, MAXRESP = 128, 32   # dropped hard for a local smoke


def _peak():
    return f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB peak" if torch.cuda.is_available() else "cpu"


def _load():
    tok = setup_tokenizer(MODEL)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = load_base_model(MODEL, None, for_training=True)  # bf16
    sync_pad_token(model, tok)
    patch_generate(model, tok)
    return tok, model, lora


def part_stopgen():
    from transformers import GenerationConfig
    tok, model, _ = _load()
    model.eval()
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": "Hi, I want to talk about my smoking."}]
    ids = tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
              return_tensors="pt").to(model.device)

    def gen(stop):
        gc = GenerationConfig(max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id,
                              eos_token_id=tok.eos_token_id, **({"stop_strings": stop} if stop else {}))
        with torch.no_grad():
            out = model.generate(**ids, generation_config=gc)  # patch injects tokenizer
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)

    base, stopped = gen(None), gen(["."])
    nb, ns = len(tok(base).input_ids), len(tok(stopped).input_ids)
    ok = ns < nb and stopped.rstrip().endswith(".")
    print(f"  no-stop tokens={nb} | stop=['.'] tokens={ns} text={stopped[:60]!r}")
    print(f"  [{'PASS' if ok else 'FAIL'}] stop_strings binds via patch_generate | {_peak()}")
    assert ok


def part_dpo():
    tok, model, lora = _load()
    turns = [{"role": "therapist" if i % 2 == 0 else "patient",
              "content": f"Turn {i:02d}: lorem ipsum dolor sit amet consectetur."} for i in range(30)]
    full = tok.apply_chat_template(turns_to_messages(turns, SYS), add_generation_prompt=True, tokenize=False)
    capped = build_truncated_training_prompt(turns, SYS, tok, max_prompt_tokens=MAXPROMPT)
    n_full = len(tok(full, add_special_tokens=False).input_ids)
    n_cap = len(tok(capped, add_special_tokens=False).input_ids)
    assert n_cap <= MAXPROMPT and "counselor named David" in capped and turns[-1]["content"][:8] in capped
    print(f"  prompt cap: full={n_full} -> capped={n_cap} tok (budget {MAXPROMPT}); system+recent kept")

    ds = Dataset.from_list([{"prompt": capped,
                             "chosen": f"What makes change matter to you now? ({i})",
                             "rejected": f"You should just quit. ({i})"} for i in range(8)])
    args = DPOConfig(output_dir=os.path.join(os.environ.get("TEMP", "/tmp"), "smoke_dpo"),
                     per_device_train_batch_size=1, gradient_accumulation_steps=2, num_train_epochs=1,
                     learning_rate=1e-4, beta=0.1, max_length=MAXPROMPT + MAXRESP, bf16=True,
                     gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                     precompute_ref_log_probs=True, logging_steps=1, save_strategy="no",
                     report_to=[], remove_unused_columns=False, seed=42)
    trainer = DPOTrainer(model=model, args=args, processing_class=tok, train_dataset=ds, peft_config=lora)
    trainer.train()
    ok = trainer.state.global_step > 0
    print(f"  DPO global_step={trainer.state.global_step} | [{'PASS' if ok else 'FAIL'}] "
          f"trains w/ grad-ckpt+precompute, no OOM | {_peak()}")
    assert ok


def part_grpo():
    tok, model, lora = _load()
    rows = [{"prompt": tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        add_generation_prompt=True, tokenize=False)}
        for q in ["I keep putting off quitting.", "I'm not sure I can change.",
                  "My doctor told me to cut down.", "I feel stuck about my habit."]]
    ds = Dataset.from_list(rows)

    def reward(prompts, completions, **kw):
        return [float(len(c)) / 50.0 for c in completions]

    args = GRPOConfig(output_dir=os.path.join(os.environ.get("TEMP", "/tmp"), "smoke_grpo"),
                      per_device_train_batch_size=2, gradient_accumulation_steps=1, num_generations=2,
                      num_train_epochs=1, learning_rate=1e-4, beta=0.0, max_completion_length=MAXRESP,
                      temperature=1.2, bf16=True, generation_kwargs={"stop_strings": ["<|im_end|>"]},
                      logging_steps=1, save_strategy="no", report_to=[], remove_unused_columns=False, seed=42)
    trainer = GRPOTrainer(model=model, args=args, processing_class=tok, reward_funcs=reward,
                          train_dataset=ds, peft_config=lora)
    patch_generate(trainer.model, tok)
    trainer.train()
    lens = [h["completions/mean_terminated_length"] for h in trainer.state.log_history
            if "completions/mean_terminated_length" in h]
    ok = trainer.state.global_step > 0
    stop_fired = bool(lens) and min(lens) < MAXRESP
    print(f"  GRPO global_step={trainer.state.global_step} terminated_len={lens} (cap {MAXRESP}) "
          f"stop_fired={stop_fired}")
    print(f"  [{'PASS' if ok else 'FAIL'}] GRPO step w/ generation_kwargs stop_strings completed | {_peak()}")
    assert ok


PARTS = {"stopgen": part_stopgen, "dpo": part_dpo, "grpo": part_grpo}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    if name == "all":
        for part in ("stopgen", "dpo", "grpo"):
            print(f"\n=== running {part} (subprocess) ===", flush=True)
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), part])
            print(f"=== {part} exit {r.returncode} ===")
    else:
        print(f"=== SMOKE PART: {name} ===")
        PARTS[name]()
        print(f"=== {name} DONE ===")
