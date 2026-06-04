"""Isolated reproduction of the iteration-2 PTO DPO step — the one that rebooted the
local Blackwell (sm_120) PC — reusing the *already-saved* iter-1 adapter + iter-2 pref
pairs. NO conversation generation, so a fault costs seconds, and you can A/B-test the
mitigation safely instead of re-running the whole quicktest.

What it reproduces: the policy is `base + iteration_1 adapter` (an already-PEFT model), so
`DPOTrainer` adds the frozen "ref" reference adapter and runs the reference forward — exactly
the iteration-2 path that iteration 1 never hits. With PRECOMPUTE=True that reference forward
is moved into a no-grad pre-pass (out of the training backward step).

Run:
    .venv\\Scripts\\python.exe Exp3_PTO_GRPO\\code\\PTO_Exp3\\_iter2_dpo_smoke.py

Toggle the knobs below. If it PASSES with PRECOMPUTE=True → run the full local quicktest.
If it still reboots the PC even with PRECOMPUTE=True → the fault is in the "ref"-adapter
forward itself; use a fallback (merge-adapter-each-iter, or train on Colab).
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_code = os.path.dirname(_here)                 # .../Exp3_PTO_GRPO/code
_root = os.path.dirname(_code)                 # .../Exp3_PTO_GRPO
for _p in (_code, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from datasets import Dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from _shared import setup_tokenizer, load_base_model, sync_pad_token, patch_generate

# ── knobs ────────────────────────────────────────────────────────────────────
PRECOMPUTE = True          # the fix under test (DPOConfig.precompute_ref_log_probs)
BATCH = 2                  # matches the PTO quicktest train batch
EXPERIMENT = "PTO_Iterative_Q1Q2_Llama32-1B_LA3_MCL12_M3_PTgreedy"
BASE_MODEL = "meta-llama/Llama-3.2-1B"
MAX_LENGTH = 2048 + 64     # MAX_ALLOWED_PROMPT_LENGTH + MAX_COMPLETION_LENGTH (quicktest)
# ─────────────────────────────────────────────────────────────────────────────

run_dir = os.path.join(_root, "data", "pto_Exp3", "runs", "quicktest", EXPERIMENT)
adapter_dir = os.path.join(run_dir, "iteration_1", "adapter")
pairs_csv = os.path.join(run_dir, "iteration_2", "pref_pairs", "pairs.csv")
for _p in (adapter_dir, pairs_csv):
    if not os.path.exists(_p):
        raise FileNotFoundError(f"missing saved artifact: {_p}")

tokenizer = setup_tokenizer(BASE_MODEL)
base = load_base_model(BASE_MODEL, None, for_training=True)   # bf16, no quant (Blackwell-safe)
sync_pad_token(base, tokenizer)
policy = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)  # iter-1 adapter = "default"
patch_generate(policy, tokenizer)

df = pd.read_csv(pairs_csv)
dataset = Dataset.from_dict({k: df[k].tolist() for k in ("prompt", "chosen", "rejected")})
print(f"Loaded {len(dataset)} iteration-2 pref pairs; policy = base + iteration_1 adapter.")

args = DPOConfig(
    output_dir=os.path.join(run_dir, "iteration_2", "_dpo_smoke"),
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=1,
    max_length=MAX_LENGTH,
    beta=0.1,
    loss_type="sigmoid",
    precompute_ref_log_probs=PRECOMPUTE,   # ← the operation change being validated
    remove_unused_columns=False,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
)
# Already-PEFT model + no explicit ref_model → TRL adds the frozen "ref" adapter (the iter-2 path).
trainer = DPOTrainer(model=policy, args=args, processing_class=tokenizer, train_dataset=dataset)
patch_generate(trainer.model, tokenizer)

print(f"Running iteration-2 DPO smoke (precompute_ref_log_probs={PRECOMPUTE}, batch={BATCH}) ...")
trainer.train()
print("\nSMOKE PASSED — the iteration-2 DPO step completed without crashing.")
