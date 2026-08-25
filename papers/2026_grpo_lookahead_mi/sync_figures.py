"""Copy every figure this paper's .tex files reference from the tracked EDA results tree into
./figures/ (never symlink).

Figures are EDA-owned. Nothing here generates a figure: each entry below points at an artifact
rendered by ``Exp3_PTO_GRPO/eda/tools/render_results.py`` into
``Exp3_PTO_GRPO/eda/results/<family>/figures/``, or at a hand-authored method schematic under
``eda/results/schematics/``. Re-run this after every render pass:

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

⚠ **This paper analyses the two GRPO arms only, but several source figures are FOUR-ARM** (they
carry the PTO arms of the parent experiment too). That is a deliberate choice, not an oversight:
the draft shows the full experiment and says in the caption which series it analyses, rather than
cropping the companion arms out of view. Any caption for a four-arm figure MUST name the two GRPO
series as this paper's subject and point at the companion draft for PTO. The entries below are
marked ``FOUR-ARM`` where this applies.

Where a source figure is produced per grader (``..._<judge>.png``), the destination keeps the
grader in the filename so a figure in the paper always says which judge produced it; the
judge-invariant families (both graders inside one figure) carry no judge segment anywhere.
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
COST = RESULTS / "compute" / "cost" / "figures"
MEASUREMENT = RESULTS / "measurement" / "validity" / "figures"
METHOD = RESULTS / "method" / "contrast" / "figures"
DEST = HERE / "figures"

# (source path, destination filename). Destination names are what the .tex references.
FIGURES: list[tuple[Path, str]] = [
    # --- sec:setup — method schematic (hand-authored: no notebook, no judge) ---------------------
    (SCHEMATICS / "grpo_group_rollout.png", "method_grpo_group.png"),
    # --- sec:reward -----------------------------------------------------------------------------
    # GRPO-only headline (added to the EDA 2026-08-25 for this paper): the two arms this draft
    # analyses, without the two PTO arms competing for the eye. Its bottom row plots K=5 - K=0,
    # the OPPOSITE sign to the source tables - see NUMBERS.md.
    (REWARD / "k_headline_q1q2_grpo.png", "k_headline_q1q2_grpo.png"),
    # --- sec:cost -------------------------------------------------------------------------------
    (COST / "budget_sweep.png", "budget_sweep.png"),
    # --- sec:behaviour --------------------------------------------------------------------------
    # The judge-free lexical marker beside both graders' rated rates (added 2026-08-25). This is
    # the section's load-bearing evidence, so it gets the figure.
    (BEHAVIOUR / "overpraise_judgefree.png", "overpraise_judgefree.png"),
    (BEHAVIOUR / "k_channel_forest_gpt-4o-mini.png", "k_channel_forest_gpt-4o-mini.png"),
    # --- sec:measurement ------------------------------------------------------------------------
    # The grader-saturation figure (added 2026-08-25): the agreement collapse and its mechanism.
    (MEASUREMENT / "judge_saturation.png", "judge_saturation.png"),
    # --- appendix A -----------------------------------------------------------------------------
    (REWARD / "k_headline_q1q2.png", "k_headline_q1q2.png"),                # FOUR-ARM (context)
    (BEHAVIOUR / "k_overpraise_trajectory_gpt-4o-mini.png", "k_overpraise_trajectory_gpt-4o-mini.png"),
    (REWARD / "k_delta_grid_gpt-4o-mini.png", "k_delta_grid_gpt-4o-mini.png"),
    (REWARD / "k_delta_grid_claude-haiku-4-5.png", "k_delta_grid_claude-haiku-4-5.png"),
    (METHOD / "headline_grid.png", "headline_grid.png"),                    # FOUR-ARM
    (MECHANISM / "tail_audit.png", "tail_audit.png"),
    (COST / "api_calls.png", "api_calls.png"),                              # FOUR-ARM
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
