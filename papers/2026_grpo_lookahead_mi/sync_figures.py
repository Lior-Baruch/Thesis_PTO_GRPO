"""Copy every figure this paper's .tex files reference from the tracked EDA results tree into
./figures/ (never symlink).

Figures are EDA-owned. Nothing here generates a figure: each entry below points at an artifact
rendered by ``Exp3_PTO_GRPO/eda/tools/render_results.py`` into
``Exp3_PTO_GRPO/eda/results/<family>/figures/``, or at a hand-authored method schematic under
``eda/results/schematics/``. Re-run this after every render pass:

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

⚠ **Figure scope policy (revised again 2026-08-26, superseding the same-day four-arm policy).**
EVERY figure in this paper — main text AND appendix — shows the two GRPO arms only. The paper is
scoped to GRPO + look-ahead; the companion PTO arms of the parent experiment appear nowhere in it,
and every statistic that used to be quoted over the full four-arm grid (the sign-preservation
ladder, the agreement medians/ranks) was RECOMPUTED over the 22 GRPO states in the EDA
(``multijudge_sign_preservation_grpo``, ``judge_saturation_grpo_data``) rather than re-scoped in
prose. The EDA renders ``*_grpo`` variants of every shared figure for exactly this use; the
four-arm originals stay canonical in the results tree for the thesis and the method-contrast
paper. A second same-day revision: the paper reads SCORES, not deltas — the headline and budget
figures are level-trajectory variants (delta strips live on in the EDA tables/figures).

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
    # GRPO-only headline in LEVELS (redesigned 2026-08-26): the two arms' Q1+Q2 trajectories from
    # their own bases, stars = Holm-cleared paired contrasts; no delta strip. Numbers behind every
    # mark: the EDA's k_headline_grpo_data table.
    (REWARD / "k_headline_q1q2_grpo.png", "k_headline_q1q2_grpo.png"),
    # --- sec:cost -------------------------------------------------------------------------------
    # Scores vs cumulative GPU-hours, GRPO only (added 2026-08-26; replaces the delta-style
    # budget_sweep_grpo in the paper). The Holm-tested best-within-budget contrast stays quoted
    # from the budget_sweep_GRPO_K_<judge> tables.
    (COST / "compute_trajectory_grpo.png", "compute_trajectory_grpo.png"),
    # --- sec:behaviour --------------------------------------------------------------------------
    # The judge-free lexical marker beside both graders' rated rates — the section's load-bearing
    # evidence. GRPO-only variant; the four-arm original stays in the EDA.
    (BEHAVIOUR / "overpraise_judgefree_grpo.png", "overpraise_judgefree_grpo.png"),
    # --- sec:measurement ------------------------------------------------------------------------
    # The grader-saturation figure, GRPO-only companion (added 2026-08-26): agreement vs the
    # 22-GRPO-state median + the SD mechanism. The four-arm original stays canonical in the EDA.
    (MEASUREMENT / "judge_saturation_grpo.png", "judge_saturation_grpo.png"),
    # --- appendix A (all GRPO-only) -------------------------------------------------------------
    (REWARD / "k_levels_grid_grpo_gpt-4o-mini.png", "k_levels_grid_grpo_gpt-4o-mini.png"),
    (REWARD / "k_levels_grid_grpo_claude-haiku-4-5.png", "k_levels_grid_grpo_claude-haiku-4-5.png"),
    (BEHAVIOUR / "k_channel_forest_grpo_gpt-4o-mini.png", "k_channel_forest_grpo_gpt-4o-mini.png"),
    (MECHANISM / "tail_audit_grpo.png", "tail_audit_grpo.png"),
    (COST / "api_calls_grpo.png", "api_calls_grpo.png"),
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
