"""
build_doron_share_2026-09-27.py — the Exp3 data package shared with Doron (Drive folder).

Doron asked (2026-09-27) for the texts from both conditions and the saved runs. Deliberately
simple: the data, organised, with a short README — he analyses it however he wants.

    <out>/README.md                              <- copied from meetings/2026-09-27_doron_share/README.md
    <out>/patients.csv                           the 96 simulated patients (id + traits)
    <out>/conversations/<ARM>/iter_NN/patient_PP.csv   turn, speaker, text
    <out>/scores_gpt-4o-mini.csv                 one row per conversation, one column per instrument
    <out>/scores_claude-haiku-4-5.csv            same, held-out grader
    <out>/adapters/<ARM>/iter_NN/{adapter_model.safetensors, adapter_config.json}   (--adapters)

The one non-trivial transform is UNSHUFFLING: every iteration the trainer simulates the same 96
personas in a seeded shuffled order and saves conversation_{position}.csv, so file index != persona.
``eda_analysis.data.persona_order`` replays that shuffle; every conversation is then checked against
the patient's own self-introduction and the build aborts on any contradiction.

Usage (from the repo root, with the repo .venv):
    .venv/Scripts/python.exe meetings/build/build_doron_share_2026-09-27.py --out <dir> [--adapters]

Reads only; the score lake and the run folders are never written to. (The first, much larger
version of this package — transcripts, per-utterance MIPROC codes, item-level scores — is commit
f848ffa; Lior judged it overkill.)
"""

import argparse
import glob
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
README_SRC = os.path.join(REPO, "meetings", "2026-09-27_doron_share", "README.md")

os.chdir(ROOT)                                   # eda_analysis resolves the workspace from cwd
sys.path.insert(0, os.path.join(ROOT, "eda"))
from eda_analysis.data import iter_conv_rows, persona_order          # noqa: E402
from eda_analysis.constants import PERSONA_COLS                      # noqa: E402
import system_prompts_builder as spb                                 # noqa: E402  (code/ on path)

# Arm label (what Doron sees) -> (method dir, EXPERIMENT_NAME, score-lake model prefix).
ARMS = {
    "GRPO_K0": ("grpo_Exp3", "GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8", "GRPOExp3_LA0"),
    "GRPO_K5": ("grpo_Exp3", "GRPO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_G8", "GRPOExp3_LA5"),
    "PTO_K0":  ("pto_Exp3", "PTO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_M8_PTgreedy", "PTOExp3_LA0"),
    "PTO_K5":  ("pto_Exp3", "PTO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_M8_PTgreedy", "PTOExp3_LA5"),
}
ITERS = range(0, 11)                             # 0 = Base (no adapter), 1..10 = trained states
N = 96

GRADERS = {"openai_gpt-4o-mini-2024-07-18": "scores_gpt-4o-mini.csv",
           "anthropic_claude-haiku-4-5": "scores_claude-haiku-4-5.csv"}
# Output column -> (score-lake metric, the per-conversation headline column the EDA uses).
INSTRUMENTS = {
    "Q1":     ("Q1", "Q1_Mean"),
    "Q2":     ("Q2", "Q2_Mean"),
    "WAI_SR": ("WAI_SR", "WAI_TotalMean"),
    "CSQ8":   ("CSQ8", "CSQ8_Mean"),
    "MI_SAT": ("MI_SAT", "MI_Mean"),
    "MITI":   ("MITI", "MITI_GlobalMean"),
    "PCT":    ("PCT", "PCT_ChangeProp"),
    "MICI":   ("MICI", "MICI_Rate"),
}
ADAPTER_FILES = ["adapter_model.safetensors", "adapter_config.json"]


def seed_of(arm: str) -> int:
    mdir, exp, _ = ARMS[arm]
    return json.load(open(os.path.join(DATA, mdir, "runs", "full", exp, "run_metadata.json")))["config"]["seed"]


# ── patients ──────────────────────────────────────────────────────────────────────────────────
def build_patients() -> pd.DataFrame:
    perms = spb.generate_all_permutations(only_expert_therapist=True)
    assert len(perms) == N, len(perms)
    rows = []
    for pid, p in enumerate(perms):
        ch = spb.get_patient_permutation_characteristics(pid)
        row = {"patient_id": pid, "name": re.search(r"Your name is (\w+)", p["patient_system_prompt"]).group(1)}
        row.update({c: ch[c] for c in PERSONA_COLS})
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"age_value": "age"}).set_index("patient_id")


OTHER_NAME = {"James": "Emma", "Emma": "James"}
OTHER_AGE = {27: 61, 61: 27}


def contradicts(patient: pd.Series, intro: str) -> list:
    """Ways the patient's first utterance contradicts the recovered persona ([] = consistent).

    A low-cooperation patient often omits its name or age, so absence proves nothing; a wrong
    unshuffle shows up as the OTHER name, the OTHER age, or the other problem. (Control: the raw file
    order contradicts in 963/1,152 conversations checked; the replayed order in 0/4,224.)
    """
    bad, text = [], str(intro)
    low = text.lower()
    if re.search(rf"\b{OTHER_NAME[patient['name']]}\b", text):
        bad.append("name")
    if re.search(rf"\b{OTHER_AGE[int(patient['age'])]}\b", text):
        bad.append("age")
    smoke = "smok" in low or "cigar" in low
    weight = any(w in low for w in ("weight", "obes", "overweight", "diet"))
    if (patient["problem"] == "Smoking" and weight and not smoke) or \
       (patient["problem"] == "Obesity" and smoke and not weight):
        bad.append("problem")
    return bad


# ── conversations ─────────────────────────────────────────────────────────────────────────────
def write_conversations(out: str, patients: pd.DataFrame) -> int:
    problems, n_convs = [], 0
    for arm, (mdir, exp, _) in ARMS.items():
        seed = seed_of(arm)
        for k in ITERS:
            hits = glob.glob(os.path.join(DATA, mdir, "conversations", "full", exp, f"model_iter_{k}_TT*_TP*"))
            assert len(hits) == 1, (arm, k, hits)
            order = persona_order(seed, k, N)
            ddir = os.path.join(out, "conversations", arm, f"iter_{k:02d}")
            os.makedirs(ddir, exist_ok=True)
            for fi in range(N):
                df = pd.read_csv(os.path.join(hits[0], f"conversation_{fi}.csv"))
                pid = order[fi]
                assert all(r == ("therapist" if i % 2 == 0 else "patient") for i, r in enumerate(df["role"])), (arm, k, fi)
                if len(df) > 1:
                    bad = contradicts(patients.loc[pid], df["conversation"].iloc[1])
                    if bad:
                        problems.append((arm, k, fi, pid, bad))
                pd.DataFrame({"turn": range(1, len(df) + 1), "speaker": df["role"], "text": df["conversation"]}) \
                    .to_csv(os.path.join(ddir, f"patient_{pid:02d}.csv"), index=False, encoding="utf-8-sig")
                n_convs += 1
        print(f"  conversations {arm} ok", flush=True)
    assert not problems, f"persona recovery contradicted by {len(problems)} intros, e.g. {problems[:5]}"
    return n_convs


# ── scores ────────────────────────────────────────────────────────────────────────────────────
def build_scores(grader: str) -> pd.DataFrame:
    rows = {}
    for arm, (_, _, prefix) in ARMS.items():
        seed = seed_of(arm)
        for k in ITERS:
            model = f"{prefix}_Base" if k == 0 else f"{prefix}_I{k}"
            oracle = "none" if k == 0 else "Q1Q2"
            order = persona_order(seed, k, N)
            for col, (metric, headline) in INSTRUMENTS.items():
                ddir = os.path.join(DATA, "eval_scores", f"judge={grader}", "rep=0",
                                    f"metric={metric}", f"oracle={oracle}", model)
                n = 0
                for fi, row in iter_conv_rows(ddir):
                    rows.setdefault((arm, k, order[fi]), {"arm": arm, "iteration": k, "patient_id": order[fi]})[col] = row[headline]
                    n += 1
                assert n == N, (grader, arm, k, metric, n)
    df = pd.DataFrame(list(rows.values())).sort_values(["arm", "iteration", "patient_id"])
    df.insert(3, "Q1Q2", (df["Q1"] + df["Q2"]) / 2)                  # the training reward's axis
    return df.round(4)


# ── adapters ──────────────────────────────────────────────────────────────────────────────────
def copy_adapters(out: str):
    for arm, (mdir, exp, _) in ARMS.items():
        for k in range(1, 11):
            src = os.path.join(DATA, mdir, "runs", "full", exp, f"iteration_{k}", "adapter")
            dst = os.path.join(out, "adapters", arm, f"iter_{k:02d}")
            os.makedirs(dst, exist_ok=True)
            for fn in ADAPTER_FILES:
                s, t = os.path.join(src, fn), os.path.join(dst, fn)
                if not (os.path.exists(t) and os.path.getsize(t) == os.path.getsize(s)):
                    shutil.copy2(s, t)
        print(f"  adapters {arm} ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", action="store_true", help="also copy the 40 LoRA adapters (~1.8 GB)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    patients = build_patients()
    patients.reset_index().to_csv(os.path.join(out, "patients.csv"), index=False, encoding="utf-8-sig")
    print("conversations ...", flush=True)
    print(f"  {write_conversations(out, patients)} conversations written, 0 intro contradictions")
    for grader, fn in GRADERS.items():
        s = build_scores(grader)
        s.to_csv(os.path.join(out, fn), index=False, encoding="utf-8-sig")
        print(f"  {fn}: {len(s)} rows, {int(s.isna().sum().sum())} empty cells")
    if a.adapters:
        copy_adapters(out)
    shutil.copy2(README_SRC, os.path.join(out, "README.md"))
    print("done:", out)


if __name__ == "__main__":
    main()
