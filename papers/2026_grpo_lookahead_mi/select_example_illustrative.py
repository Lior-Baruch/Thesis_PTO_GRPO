"""Rank (persona, therapist-turn index) pairs at iteration 10 for the body's Table 2 excerpt.

Lior's brief (2026-09-14): show a CLEAR case of the behavioural contrast -- the K=0 turn flatters a
patient who has just expressed resistance or said nothing praise-worthy, while the K=5 turn at the
same point does not flatter and answers better (a question or a reflection, no first-person slip,
not cut by the 200-token cap). This script scores every candidate pair with transparent lexical
features and prints the top ones with their texts so the pick is made by eye from a ranked list,
not from an unranked browse. The median-rule pick (``select_example_persona.py``) stays in the
appendix as the typical case.

    & ..\\..\\.venv\\Scripts\\python.exe select_example_illustrative.py            # top 12
    & ..\\..\\.venv\\Scripts\\python.exe select_example_illustrative.py --top 20 --dump out.json
    & ..\\..\\.venv\\Scripts\\python.exe select_example_illustrative.py --persona 47 --turn 4 --dump p47.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDA = HERE.parent.parent / "Exp3_PTO_GRPO" / "eda"
HELDOUT_TAG = "anthropic_claude-haiku-4-5"

# The paper's own over-praise marker (constants.RE_EFFUSIVE) plus generic praise openers.
RE_EFFUSIVE = re.compile(
    r"\bi'?m so proud|proud of you|inspiration to me|you got this|beautiful|beacon|"
    r"shining|warrior|hero of your|you are a (light|beacon)", re.I)
RE_GENERIC_PRAISE = re.compile(
    r"\b(amazing|incredible|wonderful|fantastic|brave|courageous|inspir\w+|"
    r"great (job|to hear|that)|that'?s (great|awesome|amazing|fantastic|wonderful)|"
    r"i'?m so (glad|happy|thrilled|excited)|you'?re (amazing|incredible|so strong|strong|brave))\b", re.I)
RE_PRAISE_OPEN = re.compile(r"^\W*(that'?s|that is|what a|wow|i love|i'?m so (glad|happy|thrilled)|great)", re.I)
RE_RESIST = re.compile(
    r"\b(don'?t|do not|not|never|no point|pointless|waste|forced|told to|made me|why should|"
    r"can'?t|won'?t|doubt|sceptical|skeptical|suspicious|bother|fine on my own|leave me)\b", re.I)
RE_SLIP = re.compile(
    r"\b(i feel|i'?ve been|i am (also|just)|i have (also|been)|that'?s (exactly )?how i feel|"
    r"i know (exactly )?how (that|it) feels|my (own )?(pain|weight|smoking))\b", re.I)
RE_REFLECT = re.compile(
    r"\b(sounds like|it seems|it feels like you|you feel|you'?re feeling|you mentioned|"
    r"what i'?m hearing|you'?re saying|i hear)\b", re.I)


def ends_clean(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", '"', "'", ")"))


def feats(patient: str, turn: str) -> dict:
    return {
        "effusive": len(RE_EFFUSIVE.findall(turn)) + len(RE_GENERIC_PRAISE.findall(turn)),
        "opens_praise": bool(RE_PRAISE_OPEN.match(turn)),
        "questions": turn.count("?"),
        "reflect": len(RE_REFLECT.findall(turn)),
        "slip": bool(RE_SLIP.search(turn)),
        "clean_end": ends_clean(turn),
        "chars": len(turn),
        "patient_resist": len(RE_RESIST.findall(patient)),
    }


def score(k0: dict, k5: dict, t: int) -> float:
    s = 0.0
    s += 3.0 * min(k0["effusive"], 4) + (2.0 if k0["opens_praise"] else 0.0)
    s += 1.0 * min(k0["patient_resist"], 3)        # K=0 praised a resistant patient
    s -= 5.0 * k5["effusive"] - (0.0 if not k5["opens_praise"] else 3.0)
    s += 1.5 * min(k5["questions"], 3) + 1.5 * min(k5["reflect"], 2)
    s -= 4.0 if k5["slip"] else 0.0
    s -= 2.0 if not k5["clean_end"] else 0.0
    s -= 0.4 * max(0, (t - 2) // 2)                 # earlier turns need less context
    s -= 0.002 * max(0, k5["chars"] - 700)          # very long K=5 turns cost table space
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--dump", help="write the ranked candidates (or the --persona pick) as JSON")
    ap.add_argument("--persona", type=int, help="dump this persona's full turns instead of ranking")
    ap.add_argument("--turn", type=int, default=None, help="with --persona: the therapist turn index shown")
    ap.add_argument("--n-turns", type=int, default=16)
    a = ap.parse_args()

    os.chdir(EDA)
    sys.path.insert(0, str(EDA))
    import pandas as pd  # noqa: E402
    import eda_analysis as E  # noqa: E402
    from eda_analysis import constants as C  # noqa: E402

    arms = E.filter_arms(E.discover_arms(), methods=["GRPO"])
    scores = {}
    s10p = None
    for tag, label in [("", "primary"), (HELDOUT_TAG, "heldout")]:
        C.set_active_judge(tag)
        S = E.load_scores_long(arms)
        s10 = S[(S.iteration == 10) & (S.questionnaire == "Q1Q2")]
        w = s10.pivot_table(index="persona_id", columns="K", values="score")
        w["delta"] = w[5] - w[0]
        scores[label] = w
        if label == "primary":
            s10p = s10
        C.set_active_judge("")
    personas = E.data.canonical_personas()

    def conv(arm, pid):
        r = s10p[(s10p.K == arm.K) & (s10p.persona_id == pid)].iloc[0]
        fi = int(r.file_index)
        df = pd.read_csv(os.path.join(arm.conv_dir(10), f"conversation_{fi}.csv"))
        return fi, [(str(row.role), str(row.conversation)) for _, row in df.iterrows()]

    arm0 = next(x for x in arms if x.K == 0)
    arm5 = next(x for x in arms if x.K == 5)

    if a.persona is not None:
        out = {"persona_id": a.persona, "persona": personas.loc[a.persona].to_dict(),
               "primary": {k: float(scores["primary"].loc[a.persona, c]) for k, c in (("K0", 0), ("K5", 5), ("delta", "delta"))},
               "heldout": {k: float(scores["heldout"].loc[a.persona, c]) for k, c in (("K0", 0), ("K5", 5), ("delta", "delta"))},
               "turn": a.turn, "conversations": {}}
        for arm in (arm0, arm5):
            fi, turns = conv(arm, a.persona)
            out["conversations"][arm.label] = {
                "file": f"conversation_{fi}.csv", "n_utterances": len(turns),
                "n_therapist_turns": sum(1 for r, _ in turns if r == "therapist"),
                "turns": [{"i": i, "role": r, "text": t} for i, (r, t) in enumerate(turns) if i < a.n_turns]}
        print(json.dumps({k: v for k, v in out.items() if k != "conversations"}, indent=1, default=str))
        for lab, c in out["conversations"].items():
            print(f"{lab}: {c['file']}, {c['n_utterances']} utterances, {c['n_therapist_turns']} therapist turns")
        if a.dump:
            Path(a.dump).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            print("wrote", a.dump)
        return 0

    cands = []
    for pid in personas.index:
        fi0, t0 = conv(arm0, pid)
        fi5, t5 = conv(arm5, pid)
        for t in range(2, min(len(t0), len(t5), 12), 2):
            if t0[t][0] != "therapist" or t5[t][0] != "therapist":
                continue
            f0 = feats(t0[t - 1][1], t0[t][1])
            f5 = feats(t5[t - 1][1], t5[t][1])
            cands.append({
                "persona_id": int(pid), "turn": t, "score": round(score(f0, f5, t), 2),
                "k0": f0, "k5": f5,
                "primary": {"K0": float(scores["primary"].loc[pid, 0]), "K5": float(scores["primary"].loc[pid, 5])},
                "heldout": {"K0": float(scores["heldout"].loc[pid, 0]), "K5": float(scores["heldout"].loc[pid, 5])},
                "persona": {k: str(v) for k, v in personas.loc[pid].to_dict().items()},
                "files": {"K0": fi0, "K5": fi5},
                "text": {"K0_patient": t0[t - 1][1], "K0_therapist": t0[t][1],
                         "K5_patient": t5[t - 1][1], "K5_therapist": t5[t][1]},
            })
    cands.sort(key=lambda c: -c["score"])
    for c in cands[: a.top]:
        pr, hd = c["primary"], c["heldout"]
        print("=" * 100)
        print(f"persona {c['persona_id']}  turn {c['turn']}  score {c['score']}  "
              f"Q1Q2 primary K0 {pr['K0']:.2f} K5 {pr['K5']:.2f} | held-out K0 {hd['K0']:.2f} K5 {hd['K5']:.2f}")
        print("  persona:", c["persona"])
        print("  K0 feats:", c["k0"], "\n  K5 feats:", c["k5"])
        for key in ("K0_patient", "K0_therapist", "K5_patient", "K5_therapist"):
            txt = c["text"][key]
            print(f"  [{key}] {txt[:650]}{' [...]' if len(txt) > 650 else ''}")
    if a.dump:
        Path(a.dump).write_text(json.dumps(cands[: max(a.top, 40)], ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", a.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
