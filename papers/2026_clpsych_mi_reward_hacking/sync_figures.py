"""Copy the figures this paper uses out of the EDA results tree into ./figures/.

Run after any `eda/tools/render_views.py` pass. Copies (never symlinks) so the draft
compiles standalone and a submitted PDF is frozen against later EDA reruns.

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

Every source path names its grader (`<family>/<judge>/`); the destination keeps that in the
filename, so a figure in the paper always says which judge produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EDA = REPO / "Exp3_PTO_GRPO" / "eda" / "results" / "L0" / "figures"
SCHEMATICS = REPO / "Exp3_PTO_GRPO" / "figures"
DEST = HERE / "figures"

# (source path, destination filename). Destination names are what the .tex references —
# renaming one here means editing the \includegraphics call too.
FIGURES: list[tuple[Path, str]] = [
    # sec:gains -- the loop works on its own terms.
    (EDA / "1_outcomes" / "trajectories" / "gpt-4o-mini" / "trajectory_Q1Q2.png",
     "trajectory_Q1Q2_gpt-4o-mini.png"),
    (EDA / "2_questionnaires" / "gpt-4o-mini" / "miti_proficiency_thresholds.png",
     "miti_proficiency_thresholds_gpt-4o-mini.png"),
    # sec:learned -- what the therapist actually learned. question_rate_crosscheck is the
    # paper's central diagnostic (regex `?` rate vs the oracle's own MITI B3 question code).
    (EDA / "3_validity" / "gpt-4o-mini" / "question_rate_crosscheck.png",
     "question_rate_crosscheck_gpt-4o-mini.png"),
    (EDA / "3_validity" / "gpt-4o-mini" / "reward_hack_panel.png",
     "reward_hack_panel_gpt-4o-mini.png"),
    # sec:heldout -- gain retention. Both graders in one artifact, so no <judge>/ level.
    (EDA / "8_measurement" / "multijudge_retention_trajectory.png",
     "multijudge_retention_trajectory.png"),
    (EDA / "8_measurement" / "multijudge_gain_retention.png",
     "multijudge_gain_retention.png"),
    # app:probe -- training-signal side, primary grader only by construction.
    (EDA / "6_preference" / "gpt-4o-mini" / "generation_vs_selection.png",
     "generation_vs_selection_gpt-4o-mini.png"),
    # Method schematics (hand-authored: no view, no judge, no producing notebook).
    (SCHEMATICS / "pto_preference_tree.png", "method_pto_branch.png"),
    (SCHEMATICS / "grpo_group_rollout.png", "method_grpo_group.png"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report which copies are missing or stale; copy nothing")
    args = ap.parse_args()

    DEST.mkdir(exist_ok=True)
    missing, stale, ok = [], [], 0

    for src, name in FIGURES:
        dst = DEST / name
        if not src.exists():
            missing.append(f"SOURCE GONE  {src.relative_to(REPO)}")
            continue
        if dst.exists() and digest(src) == digest(dst):
            ok += 1
            continue
        if args.check:
            stale.append(f"{'stale' if dst.exists() else 'absent'}  {name}")
            continue
        shutil.copy2(src, dst)
        print(f"copied  {name}  <-  {src.relative_to(REPO)}")

    for line in missing + stale:
        print(line, file=sys.stderr)
    print(f"\n{ok} up to date, {len(stale)} to copy, {len(missing)} missing at source")
    # A missing source is a real failure (an EDA refactor moved or renamed a figure).
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
