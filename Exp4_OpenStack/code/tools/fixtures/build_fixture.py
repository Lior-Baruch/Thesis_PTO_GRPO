"""Build the Exp4 oracle-sanity fixture from Exp3 artifacts.

Picks conversations spanning the quality range as the PRIMARY oracle scored them, and freezes
their gpt-4o-mini Q1/Q2 scores as the reference an Exp4 open grader is checked against. Costs
nothing: every number already exists in the Exp3 score lake.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

EXP3 = r"c:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp3_PTO_GRPO"
LAKE = os.path.join(EXP3, "data", "eval_scores",
                    "judge=openai_gpt-4o-mini-2024-07-18", "rep=0")
OUT = os.path.join(r"c:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp4_OpenStack",
                   "code", "tools", "fixtures", "sanity", "transcripts.json")

_PTO_LA0 = os.path.join(EXP3, "data", "pto_Exp3", "conversations", "full",
                        "PTO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_M8_PTgreedy")
_GRPO_LA0 = os.path.join(EXP3, "data", "grpo_Exp3", "conversations", "full",
                         "GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8")

# (score-lake model name, conversation root, conversation dir, oracle partition, n to pick)
# Three sources on purpose: the untrained base spans the bottom of the range, late PTO the top,
# and late GRPO contributes text that regressed into sycophancy - a different FAILURE SHAPE, which
# is what a discrimination check needs rather than three samples of the same distribution.
SOURCES = [
    ("PTOExp3_LA0_Base", _PTO_LA0, "model_iter_0_TT0.9_TP0.7", "none", 5),
    ("PTOExp3_LA0_I10", _PTO_LA0, "model_iter_10_TT0.9_TP0.7", "Q1Q2", 5),
    ("GRPOExp3_LA0_I10", _GRPO_LA0, "model_iter_10_TT0.9_TP0.7", "Q1Q2", 4),
]

MAX_CHARS = 18000         # median Exp3 transcript is ~10k chars; this keeps ~85% of them


def read_metric(model: str, oracle: str, metric: str, idx: int):
    fp = os.path.join(LAKE, f"metric={metric}", f"oracle={oracle}", model, f"{idx}.csv")
    if not os.path.isfile(fp):
        return None
    df = pd.read_csv(fp)
    col = f"{metric}_Mean"
    if col not in df.columns:
        return None
    return float(df[col].iloc[0])


def read_transcript(conv_root: str, conv_dir: str, idx: int):
    fp = os.path.join(conv_root, conv_dir, f"conversation_{idx}.csv")
    if not os.path.isfile(fp):
        return None, 0
    df = pd.read_csv(fp, keep_default_na=False)
    if "role" not in df.columns or "conversation" not in df.columns:
        return None, 0
    parts = []
    for _, r in df.iterrows():
        label = "[THERAPIST]" if str(r["role"]).lower().startswith("t") else "[PATIENT]"
        parts.append(f"{label}: {str(r['conversation']).strip()}")
    return "\n\n".join(parts), len(df)


def main():
    items = []
    for model, conv_root, conv_dir, oracle, n_pick in SOURCES:
        scored = []
        for idx in range(96):
            q1 = read_metric(model, oracle, "Q1", idx)
            q2 = read_metric(model, oracle, "Q2", idx)
            if q1 is None or q2 is None:
                continue
            scored.append((idx, q1, q2, (q1 + q2) / 2.0))
        if not scored:
            print(f"  !! no scores found for {model}")
            continue
        scored.sort(key=lambda t: t[3])
        print(f"{model}: n={len(scored)} Q1Q2 range {scored[0][3]:.2f}..{scored[-1][3]:.2f} "
              f"mean {sum(s[3] for s in scored)/len(scored):.3f}")

        # Spread the picks evenly across the ranked range so the fixture carries real variance,
        # which is what a degenerate-grader check needs.
        picks, step = [], max(1, len(scored) // n_pick)
        for j in range(0, len(scored), step):
            if len(picks) >= n_pick:
                break
            idx, q1, q2, mean = scored[j]
            text, n_utt = read_transcript(conv_root, conv_dir, idx)
            if not text or len(text) > MAX_CHARS:
                continue
            picks.append({
                "id": f"{model}_c{idx}",
                "source_model_state": model,
                "source_file_index": idx,
                "n_utterances": n_utt,
                "n_chars": len(text),
                "transcript": text,
                "reference": {"Q1": round(q1, 3), "Q2": round(q2, 3),
                              "Q1Q2": round(mean, 3)},
            })
        items.extend(picks)
        print(f"  picked {len(picks)}")

    payload = {
        "schema_version": 1,
        "built_at": time.strftime("%Y-%m-%d"),
        "reference_judge": "gpt-4o-mini-2024-07-18",
        "reference_source": ("Exp3_PTO_GRPO/data/eval_scores/"
                             "judge=openai_gpt-4o-mini-2024-07-18/rep=0 - the scores already in "
                             "the Exp3 lake, not a fresh scoring run"),
        "note": ("Transcripts are Exp3 PTO_LA0 rollouts (Llama-3.2-1B therapist vs a gpt-4o-mini "
                 "V3 patient), chosen to span the quality range the primary oracle assigned. They "
                 "exist so an Exp4 open-weights grader can be checked for (a) schema compliance "
                 "and (b) non-degenerate variance, and reported for rank agreement, BEFORE any "
                 "training spend. Exp4 and Exp3 absolute levels are NOT on the same axis - only "
                 "the ORDERING and the SPREAD are meaningful here."),
        "items": items,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)

    means = [it["reference"]["Q1Q2"] for it in items]
    print(f"\nwrote {OUT}")
    print(f"  {len(items)} items, Q1Q2 {min(means):.2f}..{max(means):.2f}, "
          f"size {os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
