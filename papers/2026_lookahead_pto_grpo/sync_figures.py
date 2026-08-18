"""Copy the EDA-owned figures this paper uses into ./figures/ (never symlink).

This paper has TWO figure sources:

1. **EDA-owned** figures under ``Exp3_PTO_GRPO/eda/results/L5/figures/`` and the hand-authored
   method schematics under ``Exp3_PTO_GRPO/figures/`` — copied here by this script, exactly like the
   sibling drafts. Re-run after any ``eda/tools/render_views.py`` pass. The list below is exactly
   the set the .tex files reference (schematics in sec:setup, composition grids in app:tails).
2. **Paper-owned** figures produced by the generators in ``./analysis/*.py`` (they write straight
   into ``./figures/`` and ``./tables/`` and log every quotable number to ``./analysis/out/*.json``).
   Those are NOT touched here; regenerate them by running the scripts.

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

Every EDA source path names its grader (``<family>/<judge>/``); the destination keeps that in the
filename so a figure in the paper always says which judge produced it.
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

# (source path, destination filename). Destination names are what the .tex references.
FIGURES: list[tuple[Path, str]] = [
    # Method schematics (hand-authored: no view, no judge, no producing notebook).
    (SCHEMATICS / "pto_preference_tree.png", "method_pto_branch.png"),
    (SCHEMATICS / "grpo_group_rollout.png", "method_grpo_group.png"),
    # app:tails (fig:composition) -- MICI composition grid (over-praise vs advice), both graders.
    (EDA / "7_stats" / "gpt-4o-mini" / "k_mici_composition_grid.png",
     "k_mici_composition_grid_gpt-4o-mini.png"),
    (EDA / "7_stats" / "claude-haiku-4-5" / "k_mici_composition_grid.png",
     "k_mici_composition_grid_claude-haiku-4-5.png"),
    # NOTE: k_overpraise_trajectory_* and k_mechanism_overpraise_* were dropped from the paper
    # (the channels figure and Table 3 carry the over-praise story); the stale copies remain in
    # ./figures/ but are no longer synced.
]


def _sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift, copy nothing")
    a = ap.parse_args()
    DEST.mkdir(exist_ok=True)
    missing, drift, copied = [], [], []
    for src, name in FIGURES:
        dst = DEST / name
        if not src.exists():
            missing.append(str(src))
            continue
        same = dst.exists() and _sha(src) == _sha(dst)
        if a.check:
            if not same:
                drift.append(name)
        elif not same:
            shutil.copy2(src, dst)
            copied.append(name)
    for m in missing:
        print("MISSING source:", m)
    if a.check:
        for d in drift:
            print("DRIFT:", d)
        print(f"{len(drift)} drifted, {len(missing)} missing")
    else:
        for c in copied:
            print("copied:", c)
        print(f"{len(copied)} copied, {len(FIGURES) - len(copied) - len(missing)} unchanged, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
