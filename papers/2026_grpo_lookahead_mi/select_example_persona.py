"""Select the persona shown in Table 2 / Appendix D and dump both arms' iteration-10 transcripts.

Selection rule (stated in the paper): of the 96 personas, the one whose persona-paired K=5 - K=0
contrast on Q1+Q2 at iteration 10 ranks closest to the median under BOTH graders, i.e. the persona
minimising |rank_primary - 48.5| + |rank_heldout - 48.5|. A typical case, chosen by rule rather
than by eye. The transcripts are read verbatim from the stored conversation CSVs; the paper's
sections/D_example.tex reproduces utterances 1-9 of each with only typographic changes
(curly quotes -> LaTeX quotes, em-dashes -> ---).

Runs against the repo's EDA package and the conversation data on disk:

    & ..\\..\\.venv\\Scripts\\python.exe select_example_persona.py            # print the pick
    & ..\\..\\.venv\\Scripts\\python.exe select_example_persona.py --dump out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDA = HERE.parent.parent / "Exp3_PTO_GRPO" / "eda"
HELDOUT_TAG = "anthropic_claude-haiku-4-5"
N_TURNS_DUMPED = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="write the pick + transcripts as JSON to this path")
    a = ap.parse_args()

    # The EDA resolves the experiment root by walking up from the cwd.
    os.chdir(EDA)
    sys.path.insert(0, str(EDA))
    import pandas as pd  # noqa: E402
    import eda_analysis as E  # noqa: E402
    from eda_analysis import constants as C  # noqa: E402

    arms = E.filter_arms(E.discover_arms(), methods=["GRPO"])
    frames = {}
    for tag, label in [("", "primary"), (HELDOUT_TAG, "heldout")]:
        C.set_active_judge(tag)
        S = E.load_scores_long(arms)
        s10 = S[(S.iteration == 10) & (S.questionnaire == "Q1Q2")]
        w = s10.pivot_table(index="persona_id", columns="K", values="score")
        w["delta"] = w[5] - w[0]
        w["rank"] = w.delta.rank(ascending=False)
        frames[label] = (w, s10)
        C.set_active_judge("")
    wp, s10p = frames["primary"]
    wh, _ = frames["heldout"]
    mid = (len(wp) + 1) / 2
    score = (wp["rank"] - mid).abs() + (wh["rank"] - mid).abs()
    pid = int(score.idxmin())
    persona = E.data.canonical_personas().loc[pid].to_dict()
    out = {
        "persona_id": pid,
        "persona": persona,
        "rule": "min |rank_primary - median| + |rank_heldout - median| of the K5-K0 Q1+Q2 contrast at iteration 10",
        "primary": {"K0": float(wp.loc[pid, 0]), "K5": float(wp.loc[pid, 5]), "delta": float(wp.loc[pid, "delta"]),
                    "median_delta": float(wp.delta.median()), "rank": float(wp.loc[pid, "rank"])},
        "heldout": {"K0": float(wh.loc[pid, 0]), "K5": float(wh.loc[pid, 5]), "delta": float(wh.loc[pid, "delta"]),
                    "median_delta": float(wh.delta.median()), "rank": float(wh.loc[pid, "rank"])},
        "conversations": {},
    }
    for arm in arms:
        r = s10p[(s10p.K == arm.K) & (s10p.persona_id == pid)].iloc[0]
        fi = int(r.file_index)
        df = pd.read_csv(os.path.join(arm.conv_dir(10), f"conversation_{fi}.csv"))
        out["conversations"][arm.label] = {
            "file": f"conversation_{fi}.csv", "n_utterances": int(len(df)),
            "n_therapist_turns": int((df.role == "therapist").sum()),
            "turns": [{"i": int(i), "role": row.role, "text": str(row.conversation)}
                      for i, row in df.iterrows() if i < N_TURNS_DUMPED],
        }
    print(json.dumps({k: v for k, v in out.items() if k != "conversations"}, indent=1))
    for lab, c in out["conversations"].items():
        print(f"{lab}: {c['file']}, {c['n_utterances']} utterances, {c['n_therapist_turns']} therapist turns")
    if a.dump:
        Path(a.dump).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", a.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
