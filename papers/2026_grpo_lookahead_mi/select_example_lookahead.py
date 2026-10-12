"""Print the worked look-ahead example (Table 1 and Appendix E.3) from the K=5 run's training log.

THE PAPER'S EXAMPLE (since 2026-10-12, Lior's pick from candidates a search of iterations 1-10 of the
K=5 logs listed): iteration 5, log line 480 (0-based), conversation 51 -- Reply A is candidate idx 5
(Q1+Q2 3.87, the group's second-highest), Reply B is idx 6 (2.98, the lowest). Print it with

    & ..\\..\\.venv\\Scripts\\python.exe select_example_lookahead.py --pick

Everything below documents the rule-based pick from the iteration-1 log that the paper showed until
then (log line 69); it is kept for provenance and still runs without --pick.

Kfir (2026-10-12): the paper never shows the method in action. The K=5 run's per-generation log
(``iteration_N/eda/generations.jsonl``) stores, for every GRPO group, the prompt, the eight sampled
candidates, each candidate's five-turn look-ahead rollout and the oracle's Q1+Q2 score of the extended
conversation. This script picks two groups from the log of the first training iteration by stated
rules and prints them verbatim:

* readable: the prompt's last patient turn, the highest- and the lowest-scoring candidate, and all
  five rollout turns of both are at most 450 characters, end in punctuation, contain only English-text
  characters and no word longer than 16 letters (candidates are sampled at temperature 1.2, and many
  are garbled); both rollouts ran all five turns;
* the gap of a group is its highest minus its lowest candidate score; the median is over every group
  in the log;
* the CLEAR case (body Table 1, Appendix E.3) is the first readable group in log order whose gap is at
  least twice the median gap (the factor 2 was set after looking at a few groups);
* the TYPICAL case is the readable group whose gap is closest to the median, ties broken by log
  order. It is printed for reference only: Appendix E.4, which showed it, was removed on 2026-10-12
  (Lior: the paper presents its examples as examples, without a selection procedure or a typical
  case), so the paper shows the clear case only.

    & ..\\..\\.venv\\Scripts\\python.exe select_example_lookahead.py                 # print both
    & ..\\..\\.venv\\Scripts\\python.exe select_example_lookahead.py --dump out.json

The log holds no turn-level (K=0) score for these candidates, so the example cannot show look-ahead
changing a ranking; Appendix E.3 says so. The iteration-1 log starts at epoch 1.0 (the run crashed and
resumed in its first iteration, Appendix D), which does not affect the reward.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = (HERE.parent.parent / "Exp3_PTO_GRPO" / "data" / "grpo_Exp3" / "runs" / "full"
       / "GRPO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_G8" / "iteration_1" / "eda" / "generations.jsonl")
MAXLEN, MAXWORD, FACTOR = 450, 16, 2.0

SPLIT = re.compile(r"\n*\[(PATIENT|THERAPIST)\]:\s*")
BAD = re.compile(r"[^A-Za-z0-9\s.,;:!?'’‘\"“”\-—–()…/%$&]")
ROLES = ["PATIENT", "THERAPIST", "PATIENT", "THERAPIST", "PATIENT"]


def turns(text: str | None) -> list[tuple[str, str]]:
    """``[ROLE]: text`` blocks -> [(role, text)]."""
    parts = SPLIT.split(text or "")
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def readable(t: str) -> bool:
    words = t.replace("’", "'").split()
    return bool(words) and len(t) <= MAXLEN and not BAD.search(t) \
        and all(len(w.strip(".,;:!?\"'()")) <= MAXWORD for w in words) \
        and t.rstrip().endswith((".", "?", "!", '"', "'", ")", "”"))


def candidate_ok(c: dict) -> bool:
    la = c.get("lookahead") or {}
    tail = turns(la.get("tail"))
    comp = c["completion"].strip()
    return (c.get("score") is not None and la.get("realized_turns") == 5 and not la.get("ended_early")
            and 40 <= len(comp) and readable(comp)
            and [r for r, _ in tail] == ROLES and all(readable(t) for _, t in tail))


def scan() -> tuple[float, list[dict]]:
    gaps, eligible = [], []
    with open(LOG, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            r = json.loads(line)
            scored = [c for c in r["candidates"] if c.get("score") is not None]
            if not scored:
                continue
            ranked = sorted(scored, key=lambda c: -c["score"])          # stable: ties keep idx order
            hi, lo = ranked[0], ranked[-1]
            gap = hi["score"] - lo["score"]
            gaps.append(gap)
            pt = turns(r["prefix"])
            if not pt or pt[-1][0] != "PATIENT" or not readable(pt[-1][1]):
                continue
            if candidate_ok(hi) and candidate_ok(lo):
                eligible.append({"line": ln, "gap": gap, "row": r, "hi": hi, "lo": lo})
    return st.median(gaps), eligible


def show(tag: str, g: dict) -> dict:
    r, pt = g["row"], turns(g["row"]["prefix"])
    print(f"\n######## {tag}: log line {g['line']} (0-based), conversation {r['conversation_id']}, "
          f"branch {r['branch_id']}, epoch {r['epoch']:.3f}, prompt of {len(pt)} utterances, gap {g['gap']:.3f}")
    print("  eight scores (idx order):", [round(c["score"], 3) for c in r["candidates"]])
    print(f"  [{pt[-2][0]}] {pt[-2][1]}")
    print(f"  [{pt[-1][0]}] {pt[-1][1]}")
    out = {"line": g["line"], "conversation_id": r["conversation_id"], "branch_id": r["branch_id"],
           "epoch": r["epoch"], "prompt_utterances": len(pt), "gap": g["gap"],
           "scores": [c["score"] for c in r["candidates"]], "prompt_last_two": pt[-2:]}
    for name, c in (("highest", g["hi"]), ("lowest", g["lo"])):
        print(f"  --- {name}: idx {c['idx']}, Q1+Q2 {c['score']:.3f} (Q1 {c['sub_scores']['1']}, "
              f"Q2 {c['sub_scores']['2']:.3f})")
        print(f"  [THERAPIST] {c['completion'].strip()}")
        for role, t in turns(c["lookahead"]["tail"]):
            print(f"  [{role}] {t}")
        out[name] = {"idx": c["idx"], "score": c["score"], "sub_scores": c["sub_scores"],
                     "completion": c["completion"].strip(), "rollout": turns(c["lookahead"]["tail"])}
    return out


PICK_ITER, PICK_LINE, PICK_IDX = 5, 480, (5, 6)          # the paper's Table 1: Reply A, Reply B


def show_pick() -> int:
    log = LOG.parents[2] / f"iteration_{PICK_ITER}" / "eda" / "generations.jsonl"
    with open(log, encoding="utf-8") as f:
        r = next(json.loads(line) for ln, line in enumerate(f) if ln == PICK_LINE)
    pt = turns(r["prefix"])
    print(f"log: {log}\nline {PICK_LINE}, conversation {r['conversation_id']}, epoch {r['epoch']:.3f}, "
          f"prompt of {len(pt)} utterances")
    print("  eight scores (idx order):", [round(c["score"], 3) for c in r["candidates"]])
    for role, t in pt[-2:]:
        print(f"  [{role}] {t}")
    for name, idx in zip(("Reply A", "Reply B"), PICK_IDX):
        c = next(c for c in r["candidates"] if c["idx"] == idx)
        print(f"  --- {name}: idx {idx}, Q1+Q2 {c['score']:.3f}")
        print(f"  [THERAPIST] {c['completion'].strip()}")
        for role, t in turns(c["lookahead"]["tail"]):
            print(f"  [{role}] {t}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="write both groups as JSON")
    ap.add_argument("--pick", action="store_true", help="print the paper's Table 1 group")
    a = ap.parse_args()
    if a.pick:
        return show_pick()
    med, eligible = scan()
    print(f"log: {LOG}\nmedian gap over all groups: {med:.3f}; readable groups: {len(eligible)}")
    clear = next(g for g in eligible if g["gap"] >= FACTOR * med)
    typical = min(eligible, key=lambda g: (abs(g["gap"] - med), g["line"]))
    res = {"median_gap": med, "n_readable": len(eligible),
           "clear": show("CLEAR (first readable group with gap >= 2 x median)", clear),
           "typical": show("TYPICAL (readable group with gap closest to the median)", typical)}
    if a.dump:
        Path(a.dump).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
