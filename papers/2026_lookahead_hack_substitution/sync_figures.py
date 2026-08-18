"""Copy the figures this paper uses out of the EDA results tree into ./figures/.

Run after any `eda/tools/render_views.py` pass. Copies (never symlinks) so the draft
compiles standalone and a submitted PDF is frozen against later EDA reruns.

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

⚠ This paper reads the **L5** view, not L0. L5 is the RQ-i owner: it is the only view whose
`7_Stats` §4c/§4d and `6_Preference` §5d actually execute, because those sections are gated to
`eda_analysis.RQ_I_VIEW` so the K contrast has exactly one owner. Pointing this at L0 yields
nothing — the sections there print a skip message.

Every source path names its grader (`<family>/<judge>/`); the destination keeps that in the
filename, so a figure in the paper always says which judge produced it. The mechanism panel's
top two rows are judge-invariant by construction (they record the training oracle's own
selection), which is why it exists only under the primary grader.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EDA = REPO / "Exp3_PTO_GRPO" / "eda" / "results" / "L5" / "figures"
SCHEMATICS = REPO / "Exp3_PTO_GRPO" / "figures"
DEST = HERE / "figures"

# (source path, destination filename). Destination names are what the .tex references —
# renaming one here means editing the \includegraphics call too.
FIGURES: list[tuple[Path, str]] = [
    # sec:channel -- look-ahead closes the channel it targets.
    (EDA / "7_stats" / "gpt-4o-mini" / "k_overpraise_trajectory.png",
     "k_overpraise_trajectory_gpt-4o-mini.png"),
    (EDA / "7_stats" / "claude-haiku-4-5" / "k_overpraise_trajectory.png",
     "k_overpraise_trajectory_claude-haiku-4-5.png"),
    # sec:aggregate -- ...and the aggregate does not move. THE paper figure.
    (EDA / "7_stats" / "gpt-4o-mini" / "k_mici_composition_grid.png",
     "k_mici_composition_grid_gpt-4o-mini.png"),
    (EDA / "7_stats" / "claude-haiku-4-5" / "k_mici_composition_grid.png",
     "k_mici_composition_grid_claude-haiku-4-5.png"),
    # sec:aggregate / app:channels -- the trade-off across every channel at the endpoint.
    (EDA / "7_stats" / "gpt-4o-mini" / "k_channel_forest.png",
     "k_channel_forest_gpt-4o-mini.png"),
    # sec:channel -- reward vs channel vs aggregate, all as dz, one frame.
    (EDA / "7_stats" / "gpt-4o-mini" / "k_cost_benefit.png",
     "k_cost_benefit_gpt-4o-mini.png"),
    # sec:mechanism -- select -> generate -> evaluate. Training-side: primary grader only.
    (EDA / "6_preference" / "gpt-4o-mini" / "k_mechanism_overpraise.png",
     "k_mechanism_overpraise_gpt-4o-mini.png"),
    # Method schematic (hand-authored: no view, no judge, no producing notebook).
    (SCHEMATICS / "pto_preference_tree.png", "method_pto_branch.png"),
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
