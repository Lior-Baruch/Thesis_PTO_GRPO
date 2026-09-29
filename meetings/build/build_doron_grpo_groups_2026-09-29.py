"""
build_doron_grpo_groups_2026-09-29.py — every GRPO round (all 8 candidates + scores) for Doron.

Doron asked (2026-09-29) whether the full group of 8 GRPO candidates is saved for both GRPO arms, to
redo the win-minus-lose embedding-direction analysis per round. It is: each training iteration writes
``iteration_N/eda/generations.jsonl`` (one line per group). This repackages those files into the
share's vocabulary — arm names, canonical ``patient_id`` — with the trainer-internal fields dropped:

    <out>/grpo_groups/README.md                 <- copied from meetings/2026-09-29_doron_grpo_groups/README.md
    <out>/grpo_groups/<ARM>/iter_NN.jsonl.gz    one line per round, the 8 candidates nested

The one non-trivial transform is ``patient_id``: a round's ``conversation_id`` is the file position in
that iteration's shuffled ``model_iter_{N-1}`` conversations, so it is unshuffled with
``persona_order(seed, N-1)`` (the same replay the 2026-09-27 share uses). Every round's conversation-so-far
is then checked utterance-by-utterance against the conversation it was cut from, and the build aborts
on any mismatch — so a wrong mapping cannot ship.

Usage (from the repo root, with the repo .venv):
    .venv/Scripts/python.exe meetings/build/build_doron_grpo_groups_2026-09-29.py --out <dir>

Reads only; the run folders are never written to.
"""

import argparse
import glob
import gzip
import json
import os
import re
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(REPO, "Exp3_PTO_GRPO")
DATA = os.path.join(ROOT, "data")
README_SRC = os.path.join(REPO, "meetings", "2026-09-29_doron_grpo_groups", "README.md")

os.chdir(ROOT)                                   # eda_analysis resolves the workspace from cwd
sys.path.insert(0, os.path.join(ROOT, "eda"))
from eda_analysis.data import persona_order      # noqa: E402

ARMS = {
    "GRPO_K0": "GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8",
    "GRPO_K5": "GRPO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_G8",
}
ITERS = range(1, 11)                             # training iterations; N moves iter_{N-1} -> iter_N
N = 96
G = 8
_SPLIT = re.compile(r"\n\n(?=\[(?:THERAPIST|PATIENT)\]: )")
_TAG = re.compile(r"^\[(THERAPIST|PATIENT)\]: ?", re.S)


def utterances(transcript: str) -> list:
    """``[(speaker, text), ...]`` from the oracle-format transcript the trainer stored."""
    out = []
    for chunk in _SPLIT.split(transcript):
        m = _TAG.match(chunk)
        assert m, chunk[:80]
        out.append((m.group(1).lower(), chunk[m.end():]))
    return out


def conv_dir(exp: str, k: int) -> str:
    hits = glob.glob(os.path.join(DATA, "grpo_Exp3", "conversations", "full", exp, f"model_iter_{k}_TT*_TP*"))
    assert len(hits) == 1, (exp, k, hits)
    return hits[0]


def build_arm(arm: str, exp: str, out: str) -> dict:
    run = os.path.join(DATA, "grpo_Exp3", "runs", "full", exp)
    seed = json.load(open(os.path.join(run, "run_metadata.json")))["config"]["seed"]
    stats = {}
    for it in ITERS:
        order = persona_order(seed, it - 1, N)   # the rounds are cut from model_iter_{it-1}
        cdir = conv_dir(exp, it - 1)
        convs = {}
        rows, mismatches = [], []
        for line in open(os.path.join(run, f"iteration_{it}", "eda", "generations.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            cid = int(r["conversation_id"])
            if cid not in convs:
                df = pd.read_csv(os.path.join(cdir, f"conversation_{cid}.csv"))
                convs[cid] = list(zip(df["role"], df["conversation"].astype(str)))
            prefix = utterances(r["prefix"])
            source = convs[cid][:len(prefix)]
            if len(source) != len(prefix) or any(
                    s[0] != p[0] or s[1].strip() != p[1].strip() for s, p in zip(source, prefix)):
                mismatches.append((cid, r["branch_id"]))
            assert prefix[-1][0] == "patient", (arm, it, r["branch_id"])
            cands = r["candidates"]
            assert len(cands) == G, (arm, it, r["branch_id"], len(cands))
            rows.append({
                "arm": arm,
                "iteration": it,
                "round": int(r["branch_id"]),
                "epoch": int(r["epoch"]) + 1 if r["phase"] == "train" else int(round(r["epoch"])),
                "split": r["phase"],
                "patient_id": order[cid],
                "turn": len(prefix) + 1,
                "conversation_so_far": r["prefix"],
                "candidates": [{
                    "text": c["completion"],
                    "reward": c["score"],
                    "Q1": (c.get("sub_scores") or {}).get("1"),
                    "Q2": (c.get("sub_scores") or {}).get("2"),
                    "lookahead": ((c.get("lookahead") or {}).get("tail") or None) if arm.endswith("K5") else None,
                } for c in sorted(cands, key=lambda c: c["idx"])],
            })
        assert not mismatches, f"{arm} iter {it}: {len(mismatches)} rounds do not match their conversation, e.g. {mismatches[:5]}"
        ddir = os.path.join(out, "grpo_groups", arm)
        os.makedirs(ddir, exist_ok=True)
        with gzip.open(os.path.join(ddir, f"iter_{it:02d}.jsonl.gz"), "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_train = sum(r["split"] == "train" for r in rows)
        stats[it] = (len(rows), n_train, len({r["patient_id"] for r in rows}))
        print(f"  {arm} iter_{it:02d}: {len(rows)} rounds ({n_train} train), {stats[it][2]} patients, all prefixes match", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for arm, exp in ARMS.items():
        build_arm(arm, exp, out)
    shutil.copy2(README_SRC, os.path.join(out, "grpo_groups", "README.md"))
    print("done:", out)


if __name__ == "__main__":
    main()
