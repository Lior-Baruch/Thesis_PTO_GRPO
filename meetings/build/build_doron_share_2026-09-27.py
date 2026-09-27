"""
build_doron_share_2026-09-27.py — the Exp3 data package shared with Doron (Drive folder).

Doron asked (2026-09-27) for the texts from both conditions, and for the saved runs so the
parameter changes can be compared layer by layer. This builds one self-describing folder:

    <out>/README.md                  <- copied from meetings/2026-09-27_doron_share/README.md
    <out>/personas.csv               96 canonical patients (id, name, traits, system prompt)
    <out>/arms.csv                   the 4 arms + their training hyperparameters
    <out>/prompts/                   therapist system prompt + fixed opening utterance + chat template
    <out>/transcripts/<ARM>/<ARM>_iterNN.txt   readable, 96 conversations each, by patient_id
    <out>/data/utterances.csv        one row per utterance (+ MIPROC code from each grader)
    <out>/data/conversations.csv     one row per conversation (persona + ending + source file)
    <out>/data/scores_main_gpt-4o-mini.csv          one row per conversation, all 8 instruments
    <out>/data/scores_heldout_claude-haiku-4-5.csv  same, held-out grader
    <out>/data/score_means_by_state.csv             per arm x iteration means (orientation only)
    <out>/adapters/<ARM>/iter_NN/{adapter_model.safetensors, adapter_config.json} (--adapters)
    <out>/adapters/<ARM>/run_config.json

The one non-trivial transform is UNSHUFFLING: every iteration the trainer simulates the same 96
personas in a seeded shuffled order and saves conversation_{position}.csv, so file index != persona.
``eda_analysis.data.persona_order`` replays that shuffle; every conversation is then checked against
the patient's own self-introduction (name -> gender, age, problem) and the build aborts on a mismatch.

Usage (from the repo root, with the repo .venv):
    .venv/Scripts/python.exe meetings/build/build_doron_share_2026-09-27.py --out <dir> [--adapters]

Reads only; the score lake and the run folders are never written to.
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

# Arm label (what Doron sees) -> (method dir, EXPERIMENT_NAME, score-lake model prefix, method, K).
# GRPO first: the GRPO K=0 vs K=5 pair is the paper's two conditions.
ARMS = {
    "GRPO_K0": ("grpo_Exp3", "GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8", "GRPOExp3_LA0", "GRPO", 0),
    "GRPO_K5": ("grpo_Exp3", "GRPO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_G8", "GRPOExp3_LA5", "GRPO", 5),
    "PTO_K0":  ("pto_Exp3", "PTO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_M8_PTgreedy", "PTOExp3_LA0", "PTO", 0),
    "PTO_K5":  ("pto_Exp3", "PTO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_M8_PTgreedy", "PTOExp3_LA5", "PTO", 5),
}
ITERS = range(0, 11)                             # 0 = Base (no adapter), 1..10 = trained states
N = 96

JUDGES = {
    "main": ("openai_gpt-4o-mini-2024-07-18", "scores_main_gpt-4o-mini.csv"),
    "heldout": ("anthropic_claude-haiku-4-5", "scores_heldout_claude-haiku-4-5.csv"),
}
METRICS = ["Q1", "Q2", "WAI_SR", "CSQ8", "MI_SAT", "MITI", "PCT", "MICI", "MIPROC"]
# (display name, column, higher-is-better) — the per-instrument headline the EDA reports.
HEADLINE = [
    ("Q1Q2", "Q1Q2_Mean", True), ("WAI-SR", "WAI_TotalMean", True), ("CSQ-8", "CSQ8_Mean", True),
    ("MI-SAT", "MI_Mean", True), ("MITI", "MITI_GlobalMean", True), ("PCT", "PCT_ChangeProp", True),
    ("MICI", "MICI_Rate", False), ("%CR", "MIPROC_PctCR", True),
]
CODE_COLS = ["MIPROC_ThCodes", "MIPROC_PtCodes"]
ADAPTER_FILES = ["adapter_model.safetensors", "adapter_config.json"]


# ── personas ──────────────────────────────────────────────────────────────────────────────────
def build_personas() -> pd.DataFrame:
    perms = spb.generate_all_permutations(only_expert_therapist=True)
    assert len(perms) == N, len(perms)
    rows = []
    for pid, p in enumerate(perms):
        ch = spb.get_patient_permutation_characteristics(pid)
        name = re.search(r"Your name is (\w+)", p["patient_system_prompt"]).group(1)
        row = {"patient_id": pid, "name": name}
        row.update({c: ch[c] for c in PERSONA_COLS})
        row["patient_system_prompt"] = p["patient_system_prompt"]
        rows.append(row)
    df = pd.DataFrame(rows).rename(columns={"age_value": "age"})
    # The therapist side is identical for every persona — assert it so the README can say so.
    assert len({p["counselor_system_prompt"] for p in perms}) == 1
    assert len({p["counselor_init_utterance"] for p in perms}) == 1
    return df, perms[0]["counselor_system_prompt"], perms[0]["counselor_init_utterance"]


OTHER_NAME = {"James": "Emma", "Emma": "James"}
OTHER_AGE = {27: 61, 61: 27}


def check_intro(persona: pd.Series, intro: str):
    """Match the patient's first utterance against the recovered persona.

    Returns ``(contradictions, confirmed)``. A low-cooperation patient often omits its name or age,
    so absence proves nothing; what would expose a wrong unshuffle is the OTHER persona's name, the
    OTHER age, or the other problem. ``confirmed`` = the intro states both the right name and age.
    """
    bad = []
    text = str(intro)
    low = text.lower()
    if re.search(rf"\b{OTHER_NAME[persona['name']]}\b", text):
        bad.append(f"says {OTHER_NAME[persona['name']]}, persona is {persona['name']}")
    if re.search(rf"\b{OTHER_AGE[int(persona['age'])]}\b", text):
        bad.append(f"says {OTHER_AGE[int(persona['age'])]}, persona is {persona['age']}")
    smoke = "smok" in low or "cigar" in low
    weight = any(w in low for w in ("weight", "obes", "overweight", "diet"))
    if persona["problem"] == "Smoking" and weight and not smoke:
        bad.append("talks about weight, persona is Smoking")
    if persona["problem"] == "Obesity" and smoke and not weight:
        bad.append("talks about smoking, persona is Obesity")
    confirmed = (re.search(rf"\b{persona['name']}\b", text) is not None
                 and re.search(rf"\b{int(persona['age'])}\b", text) is not None)
    return bad, confirmed


# ── conversations ─────────────────────────────────────────────────────────────────────────────
def conv_dir(arm: str, k: int) -> str:
    mdir, exp, *_ = ARMS[arm]
    hits = glob.glob(os.path.join(DATA, mdir, "conversations", "full", exp, f"model_iter_{k}_TT*_TP*"))
    assert len(hits) == 1, (arm, k, hits)
    return hits[0]


def load_conversations(personas: pd.DataFrame):
    convs, utts, problems = [], [], []
    confirmed = 0
    for arm, (mdir, exp, prefix, method, K) in ARMS.items():
        seed = json.load(open(os.path.join(DATA, mdir, "runs", "full", exp, "run_metadata.json")))["config"]["seed"]
        for k in ITERS:
            d = conv_dir(arm, k)
            order = persona_order(seed, k, N)
            for fi in range(N):
                path = os.path.join(d, f"conversation_{fi}.csv")
                df = pd.read_csv(path).rename(columns={
                    "session_endded_by": "session_ended_by",
                    "session_endded_explanation": "session_ended_explanation"})
                pid = order[fi]
                persona = personas.loc[pid]
                roles = df["role"].tolist()
                # The scorer labels utterances by position (even = therapist); make sure the saved
                # role column agrees, or the MIPROC alignment below would be off by one.
                assert all(r == ("therapist" if i % 2 == 0 else "patient") for i, r in enumerate(roles)), path
                if len(df) > 1:
                    bad, ok = check_intro(persona, df["conversation"].iloc[1])
                    confirmed += ok
                    if bad:
                        problems.append((arm, k, fi, pid, bad))
                rel = os.path.relpath(path, DATA).replace(os.sep, "/")
                convs.append({
                    "arm": arm, "method": method, "K": K, "iteration": k, "patient_id": pid,
                    "n_utterances": len(df),
                    "n_therapist": roles.count("therapist"), "n_patient": roles.count("patient"),
                    "session_ended_by": df["session_ended_by"].iloc[0],
                    "session_ended_explanation": df["session_ended_explanation"].iloc[0],
                    "source_file": rel, "source_file_index": fi,
                })
                t = p = 0
                for i, (role, text) in enumerate(zip(roles, df["conversation"])):
                    if role == "therapist":
                        t += 1; turn = t
                    else:
                        p += 1; turn = p
                    utts.append({"arm": arm, "method": method, "K": K, "iteration": k,
                                 "patient_id": pid, "utt_index": i, "speaker": role,
                                 "speaker_turn": turn, "text": text})
    return pd.DataFrame(convs), pd.DataFrame(utts), problems, confirmed


# ── scores ────────────────────────────────────────────────────────────────────────────────────
def load_scores(judge_tag: str):
    """One wide row per conversation. Returns ``(df, gaps)``; a gap = a (state, metric) with < 96
    scored conversations — its cells stay NaN rather than stopping the build."""
    rows, gaps = {}, []
    for arm, (mdir, exp, prefix, method, K) in ARMS.items():
        seed = json.load(open(os.path.join(DATA, mdir, "runs", "full", exp, "run_metadata.json")))["config"]["seed"]
        for k in ITERS:
            model = f"{prefix}_Base" if k == 0 else f"{prefix}_I{k}"
            oracle = "none" if k == 0 else "Q1Q2"
            order = persona_order(seed, k, N)
            for metric in METRICS:
                ddir = os.path.join(DATA, "eval_scores", f"judge={judge_tag}", "rep=0",
                                    f"metric={metric}", f"oracle={oracle}", model)
                n = 0
                for fi, row in iter_conv_rows(ddir):
                    key = (arm, k, order[fi])
                    rec = rows.setdefault(key, {"arm": arm, "method": method, "K": K,
                                                "iteration": k, "patient_id": order[fi]})
                    rec.update({c: row[c] for c in row.index})
                    n += 1
                if n != N:
                    got = {order[fi] for fi, _ in iter_conv_rows(ddir)}
                    gaps.append((arm, k, metric, n, sorted(set(range(N)) - got)))
    df = pd.DataFrame(list(rows.values()))
    assert len(df) == len(ARMS) * len(ITERS) * N, len(df)
    df.insert(5, "Q1Q2_Mean", (df["Q1_Mean"] + df["Q2_Mean"]) / 2)   # the training reward's axis
    return df, gaps


def attach_codes(utts: pd.DataFrame, scores: pd.DataFrame, col: str):
    """Put each grader's MIPROC code next to the utterance it codes (k-th code <-> k-th turn)."""
    codes = {}
    mismatched = 0
    counts = utts.groupby(["arm", "iteration", "patient_id", "speaker"]).size()
    for _, r in scores.iterrows():
        key = (r["arm"], r["iteration"], r["patient_id"])
        if pd.isna(r["MIPROC_ThCodes"]) and pd.isna(r["MIPROC_PtCodes"]):
            continue                                         # not coded (a reported gap)
        th = str(r["MIPROC_ThCodes"]).split("|") if pd.notna(r["MIPROC_ThCodes"]) else []
        pt = str(r["MIPROC_PtCodes"]).split("|") if pd.notna(r["MIPROC_PtCodes"]) else []
        n_th = counts.get(key + ("therapist",), 0)
        n_pt = counts.get(key + ("patient",), 0)
        if len(th) != n_th or len(pt) != n_pt:
            mismatched += 1
            continue                                         # leave blank rather than misalign
        for i, c in enumerate(th, 1):
            codes[key + ("therapist", i)] = c
        for i, c in enumerate(pt, 1):
            codes[key + ("patient", i)] = c
    keys = list(zip(utts["arm"], utts["iteration"], utts["patient_id"], utts["speaker"], utts["speaker_turn"]))
    utts[col] = [codes.get(k, "") for k in keys]
    return mismatched


# ── readable transcripts ──────────────────────────────────────────────────────────────────────
def fmt_scores(r) -> str:
    return " | ".join(f"{name} {r[col]:.2f}" if pd.notna(r[col]) else f"{name} n/a"
                      for name, col, _ in HEADLINE)


def write_transcripts(out, personas, convs, utts, scores):
    main = scores["main"].set_index(["arm", "iteration", "patient_id"])
    held = scores["heldout"].set_index(["arm", "iteration", "patient_id"])
    by_conv = {k: g for k, g in utts.groupby(["arm", "iteration", "patient_id"])}
    cinfo = convs.set_index(["arm", "iteration", "patient_id"])
    for arm in ARMS:
        adir = os.path.join(out, "transcripts", arm)
        os.makedirs(adir, exist_ok=True)
        for k in ITERS:
            state = "Base model (no training)" if k == 0 else f"policy after {k} training iteration{'s' * (k > 1)}"
            lines = [
                f"{arm} | iteration {k:02d} | {state}",
                f"96 conversations, ordered by patient_id. The same patient_id is the same simulated patient in every file.",
                f"Scores: main = gpt-4o-mini-2024-07-18 (also the training oracle for Q1/Q2); held-out = claude-haiku-4-5.",
                f"Codes in [main/held-out] brackets are the MIPROC utterance codes from each grader (legend in README.md).",
                "",
            ]
            for pid in range(N):
                key = (arm, k, pid)
                ps, ci = personas.loc[pid], cinfo.loc[key]
                ended = ci["session_ended_by"]
                lines += [
                    "#" * 100,
                    f"patient_{pid:02d} | {ps['name']}, {ps['gender'].lower()}, {ps['age']} | problem={ps['problem']} "
                    f"({ps['problem_time']}) | tried_to_solve={ps['tried_to_solve']} | cooperation={ps['cooperation_level']}",
                    f"{ci['n_utterances']} utterances | session_ended_by={ended} | explanation={str(ci['session_ended_explanation']).strip()}",
                    f"main:     {fmt_scores(main.loc[key])}",
                    f"held-out: {fmt_scores(held.loc[key])}",
                    "-" * 100,
                ]
                for _, u in by_conv[key].iterrows():
                    tag = ("T" if u["speaker"] == "therapist" else "P") + str(u["speaker_turn"])
                    codes = f"[{u['miproc_main'] or '-'}/{u['miproc_heldout'] or '-'}]"
                    body = str(u["text"]).replace("\r\n", "\n").replace("\n", "\n" + " " * 8)
                    lines += [f"{tag:<4}{codes:<12}{u['speaker'].upper()}: {body}", ""]
            with open(os.path.join(adir, f"{arm}_iter{k:02d}.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))


# ── arms + adapters ───────────────────────────────────────────────────────────────────────────
ARM_KEYS = ["seed", "num_iterations", "num_conversations_per_iter", "epochs_per_iteration",
            "learning_rate", "lora_r", "lora_alpha", "min_conv_length", "num_utterances_for_data",
            "max_tokens_per_response", "temperature_therapist_gen", "temperature_patient",
            "num_generations", "grpo_beta", "grpo_temperature", "grpo_loss_type",
            "train_batch_size", "gradient_accumulation_steps",
            "pref_tree_mode", "num_branches_per_turn", "pref_filter_tau",
            "branch_sample_temperature", "dpo_beta", "dpo_loss_type"]
DROP_FROM_CONFIG = {"local_outdir", "conv_outdir", "current_adapter_repo", "report_to",
                    "push_to_hub", "gen_verbose", "gen_verbose_detailed"}


def build_arms(out, copy_adapters: bool):
    rows = []
    for arm, (mdir, exp, prefix, method, K) in ARMS.items():
        run = os.path.join(DATA, mdir, "runs", "full", exp)
        cfg = json.load(open(os.path.join(run, "run_metadata.json")))["config"]
        row = {"arm": arm, "method": method, "K_lookahead": K, "run_name": exp,
               "loss": "GRPO (group-relative policy gradient + KL)" if method == "GRPO" else "DPO on preference-tree pairs"}
        row.update({c: cfg.get(c, "") for c in ARM_KEYS})
        rows.append(row)
        adir = os.path.join(out, "adapters", arm)
        os.makedirs(adir, exist_ok=True)
        clean = {k: v for k, v in cfg.items() if k not in DROP_FROM_CONFIG}
        clean["lookahead_k"] = K                              # only the LA5 arms carry the audit mirror
        with open(os.path.join(adir, "run_config.json"), "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2)
        if copy_adapters:
            for k in range(1, 11):
                src = os.path.join(run, f"iteration_{k}", "adapter")
                dst = os.path.join(adir, f"iter_{k:02d}")
                os.makedirs(dst, exist_ok=True)
                for fn in ADAPTER_FILES:
                    s, t = os.path.join(src, fn), os.path.join(dst, fn)
                    if os.path.exists(t) and os.path.getsize(t) == os.path.getsize(s):
                        continue
                    shutil.copy2(s, t)
                print(f"  adapters {arm} iter_{k:02d} ok", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", action="store_true", help="also copy the 40 LoRA adapters (~1.8 GB)")
    ap.add_argument("--adapters-only", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    if a.adapters_only:
        build_arms(out, copy_adapters=True)
        return

    personas, th_prompt, th_open = build_personas()
    print("loading conversations ...", flush=True)
    convs, utts, problems, confirmed = load_conversations(personas)
    print(f"  {len(convs)} conversations, {len(utts)} utterances; intro contradictions: {len(problems)}; "
          f"intros stating the recovered name AND age: {confirmed}/{len(convs)}")
    for p in problems[:20]:
        print("   ", p)
    assert not problems, "persona recovery failed a self-introduction check — not writing"

    scores = {}
    for jk, (tag, _) in JUDGES.items():
        print(f"loading scores: {tag} ...", flush=True)
        scores[jk], gaps = load_scores(tag)
        for g in gaps:
            print(f"  GAP (left blank): arm={g[0]} iteration={g[1]} metric={g[2]} scored={g[3]}/96 missing patient_id={g[4]}")
        mism = attach_codes(utts, scores[jk], f"miproc_{jk}")
        print(f"  {len(scores[jk])} conversation rows; MIPROC code/utterance count mismatches left blank: {mism}")

    # ── write ──
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    os.makedirs(os.path.join(out, "prompts"), exist_ok=True)
    personas.to_csv(os.path.join(out, "personas.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out, "prompts", "therapist_system_prompt.txt"), "w", encoding="utf-8") as fh:
        fh.write(th_prompt + "\n")
    with open(os.path.join(out, "prompts", "therapist_opening_utterance.txt"), "w", encoding="utf-8") as fh:
        fh.write(th_open + "\n")
    # The ChatML template every arm's therapist was run with (identical across arms — asserted).
    tmpls = {open(os.path.join(DATA, m, "runs", "full", e, "iteration_10", "adapter", "chat_template.jinja"),
                  encoding="utf-8").read() for m, e, *_ in ARMS.values()}
    assert len(tmpls) == 1, "chat templates differ across arms"
    with open(os.path.join(out, "prompts", "therapist_chat_template.jinja"), "w", encoding="utf-8") as fh:
        fh.write(tmpls.pop())

    pcols = ["patient_id", "name", "gender", "age", "problem", "problem_time", "tried_to_solve", "cooperation_level"]
    key = ["arm", "iteration", "patient_id"]
    convs = convs.merge(personas[pcols], on="patient_id").sort_values(key)
    convs.to_csv(os.path.join(out, "data", "conversations.csv"), index=False, encoding="utf-8-sig")
    utts.sort_values(key + ["utt_index"]).rename(columns={
        "miproc_main": "miproc_code_main_gpt-4o-mini",
        "miproc_heldout": "miproc_code_heldout_claude-haiku-4-5",
    }).to_csv(os.path.join(out, "data", "utterances.csv"), index=False, encoding="utf-8-sig")

    means = []
    for jk, (tag, fn) in JUDGES.items():
        s = scores[jk].drop(columns=CODE_COLS).sort_values(key)
        s.to_csv(os.path.join(out, "data", fn), index=False, encoding="utf-8-sig")
        m = s.groupby(["arm", "iteration"])[[c for _, c, _ in HEADLINE]].mean().reset_index()
        m.insert(0, "grader", {"main": "main_gpt-4o-mini", "heldout": "heldout_claude-haiku-4-5"}[jk])
        means.append(m)
    pd.concat(means).round(4).to_csv(os.path.join(out, "data", "score_means_by_state.csv"), index=False, encoding="utf-8-sig")

    print("writing transcripts ...", flush=True)
    write_transcripts(out, personas.set_index("patient_id"), convs, utts, scores)

    build_arms(out, copy_adapters=a.adapters).to_csv(os.path.join(out, "arms.csv"), index=False, encoding="utf-8-sig")
    shutil.copy2(README_SRC, os.path.join(out, "README.md"))
    print("done:", out)


if __name__ == "__main__":
    main()
