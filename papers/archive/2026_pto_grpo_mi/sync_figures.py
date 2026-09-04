"""Copy every figure this paper's .tex files reference from the tracked EDA results tree into
./figures/ (never symlink).

Figures are EDA-owned. Nothing here generates a figure: each entry points at an artifact rendered
by ``Exp3_PTO_GRPO/eda/tools/render_results.py``, or at a hand-authored method schematic under
``eda/results/schematics/``. Re-run after every render pass:

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

**Figure scope policy.** This is the FOUR-ARM paper — the full 2x2 (optimizer x reward horizon)
is its subject, so every results figure deliberately carries all four arms. Levels are
preferred over deltas wherever a level artifact exists (k_trajectory_Q1Q2, headline_grid); the
delta-style artifacts stay in the EDA. The comparison axis is ITERATIONS ONLY (decided
2026-08-27): no compute/budget figures — the compute/cost family stays EDA-only.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "Exp3_PTO_GRPO" / "eda" / "results"
SCHEMATICS = RESULTS / "schematics"                 # hand-authored (no notebook, no judge level)
REWARD = RESULTS / "lookahead" / "reward" / "figures"
BEHAVIOUR = RESULTS / "lookahead" / "behaviour" / "figures"
REPLICATION = RESULTS / "lookahead" / "replication" / "figures"
MEASUREMENT = RESULTS / "measurement" / "validity" / "figures"
METHOD = RESULTS / "method" / "contrast" / "figures"
DEST = HERE / "figures"

# (source path, destination filename). Destination names are what the .tex references.
FIGURES: list[tuple[Path, str]] = [
    # --- sec:setup — the two optimizers' schematics (hand-authored) ------------------------------
    (SCHEMATICS / "pto_preference_tree.png", "method_pto_tree.png"),
    (SCHEMATICS / "grpo_group_rollout.png", "method_grpo_group.png"),
    # --- sec:interaction -------------------------------------------------------------------------
    # All four arms' Q1+Q2 level by iteration, one panel per grader (LEVELS, not deltas).
    (REWARD / "k_trajectory_Q1Q2.png", "k_trajectory_Q1Q2.png"),
    # The endpoint grid: four arms x both graders, each anchored to its own base.
    (METHOD / "headline_grid.png", "headline_grid.png"),
    # --- sec:behaviour ---------------------------------------------------------------------------
    # The judge-free lexical over-praise marker beside both graders' rated rates, four arms.
    (BEHAVIOUR / "overpraise_judgefree.png", "overpraise_judgefree.png"),
    # --- sec:regime ------------------------------------------------------------------------------
    (REPLICATION / "crossgen.png", "crossgen.png"),
    # --- sec:measurement -------------------------------------------------------------------------
    (MEASUREMENT / "judge_saturation.png", "judge_saturation.png"),
    # --- appendix --------------------------------------------------------------------------------
    (BEHAVIOUR / "k_channel_forest_gpt-4o-mini.png", "k_channel_forest_pto_gpt-4o-mini.png"),
    (BEHAVIOUR / "k_channel_forest_grpo_gpt-4o-mini.png", "k_channel_forest_grpo_gpt-4o-mini.png"),
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
