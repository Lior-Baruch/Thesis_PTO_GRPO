"""Copy every figure this paper's .tex files reference from the tracked EDA results tree into
./figures/ (never symlink).

Figures are EDA-owned. Nothing here generates a figure: each entry points at an artifact rendered
by ``Exp3_PTO_GRPO/eda/tools/render_results.py``, or at a hand-authored method schematic under
``eda/results/schematics/``. Re-run after every render pass:

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

**Figure scope policy.** This is the GRPO-ONLY paper — its subjects are the two GRPO arms, so
every results figure is a ``*_grpo`` artifact (recomputed/cropped to the 22 GRPO states); the
four-arm artifacts belong to the companion 2x2 paper (papers/2026_pto_grpo_mi). Figures read
SCORES (levels incl. each arm's base), not K5-K0 deltas. The comparison axis is ITERATIONS ONLY
(decided 2026-08-27): no compute/budget or API-call figures — the compute/cost family stays
EDA-only, and the cost disclosure in Limitations cites tables, not figures.
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
MECHANISM = RESULTS / "lookahead" / "mechanism" / "figures"
MEASUREMENT = RESULTS / "measurement" / "validity" / "figures"
DEST = HERE / "figures"

# (source path, destination filename). Destination names are what the .tex references.
FIGURES: list[tuple[Path, str]] = [
    # --- sec:method — Figure 1, the GRPO group schematic (hand-authored) --------------------------
    (SCHEMATICS / "grpo_group_rollout.png", "method_grpo_group.png"),
    # --- sec:reward — the two arms' Q1+Q2 levels by iteration, one panel per grader --------------
    (REWARD / "k_headline_q1q2_grpo.png", "k_headline_q1q2_grpo.png"),
    # --- sec:behaviour — judge-free lexical marker beside both graders' rated rates --------------
    (BEHAVIOUR / "overpraise_judgefree_grpo.png", "overpraise_judgefree_grpo.png"),
    # --- sec:measurement -------------------------------------------------------------------------
    (MEASUREMENT / "judge_saturation_grpo.png", "judge_saturation_grpo.png"),
    # --- appendix --------------------------------------------------------------------------------
    (REWARD / "k_levels_grid_grpo_gpt-4o-mini.png", "k_levels_grid_grpo_gpt-4o-mini.png"),
    (REWARD / "k_levels_grid_grpo_claude-haiku-4-5.png", "k_levels_grid_grpo_claude-haiku-4-5.png"),
    (BEHAVIOUR / "k_channel_forest_grpo_gpt-4o-mini.png", "k_channel_forest_grpo_gpt-4o-mini.png"),
    (MECHANISM / "tail_audit_grpo.png", "tail_audit_grpo.png"),
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
