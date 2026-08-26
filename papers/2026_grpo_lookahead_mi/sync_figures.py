"""Copy every figure this paper's .tex files reference from the tracked EDA results tree into
./figures/ (never symlink).

Figures are EDA-owned. Nothing here generates a figure: each entry below points at an artifact
rendered by ``Exp3_PTO_GRPO/eda/tools/render_results.py`` into
``Exp3_PTO_GRPO/eda/results/<family>/figures/``, or at a hand-authored method schematic under
``eda/results/schematics/``. Re-run this after every render pass:

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

⚠ **Figure scope policy (revised 2026-08-26).** MAIN-TEXT figures show the two GRPO arms only —
the EDA renders ``*_grpo`` variants of the shared figures for exactly this use — with ONE
deliberate exception: ``judge_saturation``, whose four arms are load-bearing (the claim is "the
two lowest-agreeing of 44 states", and §8 uses PTO K=5's parallel decline). APPENDIX figures stay
FOUR-ARM on purpose: their declared job is to show the full experiment rather than crop the
companion arms out of view, and each caption names the two GRPO series as this paper's subject.
Entries below are marked ``FOUR-ARM`` where this applies.

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
    # GRPO-only variant (added 2026-08-26): the 2x2 original carries a PTO column sec:cost never
    # discusses (and which previews the companion paper's negative result).
    (COST / "budget_sweep_grpo.png", "budget_sweep_grpo.png"),
    # --- sec:behaviour --------------------------------------------------------------------------
    # The judge-free lexical marker beside both graders' rated rates — the section's load-bearing
    # evidence. GRPO-only variant (added 2026-08-26); the four-arm original stays in the EDA.
    (BEHAVIOUR / "overpraise_judgefree_grpo.png", "overpraise_judgefree_grpo.png"),
    (BEHAVIOUR / "k_channel_forest_gpt-4o-mini.png", "k_channel_forest_gpt-4o-mini.png"),
    # --- sec:measurement ------------------------------------------------------------------------
    # The grader-saturation figure: the agreement collapse and its mechanism. FOUR-ARM by design —
    # the "two lowest-agreeing of 44 states" claim and §8's PTO-K=5 sentence need the other arms.
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
