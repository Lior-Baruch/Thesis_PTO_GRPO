"""Draw the personas for Appendix E.2's example and print what each shows (Lior, 2026-10-06).

Appendix E has two matched-persona examples. E.1 (and Table 2) is hand-chosen from the top of a ranking
(``select_example_illustrative.py --coder``: persona 87). E.2 is "half hand-chosen": eight of the 96
personas are drawn with a fixed seed, and the clearest of those eight is picked by eye. This script
makes the draw reproducible and prints, for every drawn persona, utterances 1-8 of both runs'
iteration-10 conversations with each judge's utterance code, and Q1+Q2 under both judges with its
rank among the 96 K=5 - K=0 differences. The pick (persona 21, utterances 1-4) and the reasons are in
NUMBERS.md (2026-10-06); E.2 reproduces those utterances verbatim with typographic changes only
(curly quotes -> LaTeX quotes, a paragraph break -> a space). It replaced the median-rule persona 93
(``select_example_persona.py``), which happens to be in this draw too.

    & ..\\..\\.venv\\Scripts\\python.exe select_example_random.py               # the draw, printed
    & ..\\..\\.venv\\Scripts\\python.exe select_example_random.py --dump out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EDA = HERE.parent.parent / "Exp3_PTO_GRPO" / "eda"
HELDOUT_TAG = "anthropic_claude-haiku-4-5"
SEED, N_DRAW, IT, N_SHOWN = 20261006, 8, 10, 9          # utterances 0-8 (0 = the scripted opener)


def draw_personas() -> list[int]:
    """The draw itself: eight of the canonical persona ids 0-95, without replacement."""
    return [int(x) for x in np.random.default_rng(SEED).choice(96, size=N_DRAW, replace=False)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="write the draw with transcripts, codes and scores as JSON")
    a = ap.parse_args()

    os.chdir(EDA)                                       # the EDA resolves the experiment root from the cwd
    sys.path.insert(0, str(EDA))
    import pandas as pd  # noqa: E402
    import eda_analysis as E  # noqa: E402
    from eda_analysis import constants as C  # noqa: E402
    from eda_analysis import process as Pr  # noqa: E402

    arms = E.filter_arms(E.discover_arms(), methods=["GRPO"])
    arm = {x.K: x for x in arms}
    personas = E.data.canonical_personas()
    assert list(personas.index) == list(range(96)), "canonical persona ids are expected to be 0-95"

    q12, files, codes = {}, {}, {}
    for tag, lab in [("", "primary"), (HELDOUT_TAG, "heldout")]:
        C.set_active_judge(tag)
        s = E.load_scores_long(arms)
        s = s[(s.iteration == IT) & (s.questionnaire == "Q1Q2")]
        w = s.pivot_table(index="persona_id", columns="K", values="score")
        w["delta"] = w[5] - w[0]
        w["rank"] = w["delta"].rank(ascending=False, method="min").astype(int)   # 1 = largest K=5 lead
        q12[lab] = w
        files[lab] = {(int(r.K), int(r.persona_id)): int(r.file_index) for r in s.itertuples()}
        mp = Pr.load_miproc(arms)
        mp = mp[mp.iteration == IT]
        codes[lab] = {(int(r.K), int(r.file_index)): (r.th_codes.split("|") if r.th_codes else [],
                                                      r.pt_codes.split("|") if r.pt_codes else [])
                      for r in mp.itertuples()}
        C.set_active_judge("")
    assert files["primary"] == files["heldout"], "persona -> file mapping differs between judges"

    def code(lab, k, fi, i, role):
        th, pt = codes[lab].get((k, fi), ([], []))
        p = i // 2 if role == "therapist" else (i - 1) // 2
        seq = th if role == "therapist" else pt
        return seq[p] if p < len(seq) else None

    out = {"call": f"numpy.random.default_rng({SEED}).choice(96, size={N_DRAW}, replace=False)",
           "draw": draw_personas(),
           "median_delta": {lab: float(q12[lab]["delta"].median()) for lab in q12}, "personas": []}
    print(out["call"], "->", out["draw"])
    print("median K=5 - K=0 on Q1+Q2 over the 96: "
          + ", ".join(f"{lab} {v:+.2f}" for lab, v in out["median_delta"].items()))
    for pid in out["draw"]:
        item = {"persona_id": pid, "persona": {k: str(v) for k, v in personas.loc[pid].to_dict().items()},
                "q1q2": {lab: {c: (float(q12[lab].loc[pid, c]) if c != "rank" else int(q12[lab].loc[pid, c]))
                               for c in (0, 5, "delta", "rank")} for lab in q12},
                "conversations": {}}
        print("\n" + "=" * 100)
        print(f"persona {pid}: {item['persona']}")
        for lab, v in item["q1q2"].items():
            print(f"  Q1+Q2 {lab}: K=0 {v[0]:.2f}  K=5 {v[5]:.2f}  diff {v['delta']:+.2f} (rank {v['rank']} of 96)")
        for k in (0, 5):
            fi = files["primary"][(k, pid)]
            df = pd.read_csv(os.path.join(arm[k].conv_dir(IT), f"conversation_{fi}.csv"))
            turns = [(str(r.role), str(r.conversation)) for r in df.itertuples()]
            utts = [{"i": i, "role": role, "text": text,
                     "codes": f"{code('primary', k, fi, i, role)}/{code('heldout', k, fi, i, role)}"}
                    for i, (role, text) in enumerate(turns[:N_SHOWN])]
            item["conversations"][f"K{k}"] = {"file": f"conversation_{fi}.csv", "n_utterances": len(turns),
                                              "utterances": utts}
            print(f"  -- K={k}: {fi=}, {len(turns)} utterances")
            for u in utts[1:]:
                print(f"    [{u['i']} {u['role']} {u['codes']}] {u['text'][:400]}{' [...]' if len(u['text']) > 400 else ''}")
        out["personas"].append(item)
    if a.dump:
        Path(a.dump).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", a.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
