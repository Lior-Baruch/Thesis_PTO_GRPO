"""Copy every figure this paper's .tex files reference from the tracked EDA results tree into
./figures/ (never symlink).

**All figures now sync from the tracked tree.** The nine paper-local generators that used to live
in ``./analysis/*.py`` were PROMOTED into ``eda_analysis`` modules + the family notebooks under
``Exp3_PTO_GRPO/eda/notebooks/{lookahead,compute}/`` on 2026-08-18, so their figures are rendered
by the EDA under ``Exp3_PTO_GRPO/eda/results/<family>/figures/`` (see ``eda/README.md`` §
"Migration (2026-08-18)"). The method schematics are hand-authored under
``eda/results/schematics/`` (no notebook, no judge level). ``./analysis/out/`` + ``./tables/`` stay
behind as the paper's FROZEN FIXTURE (see ``analysis/README.md``); nothing here regenerates them.

The FIGURES list below is exactly the set the .tex files reference (``grep includegraphics
sections/*.tex``). Destination names are the paper's historical filenames, so no .tex edit is
needed when the EDA re-renders; the source names are the notebooks' ``save_fig`` names.

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

Re-run after ``python Exp3_PTO_GRPO/eda/tools/render_results.py --top lookahead compute``.
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
SCHEMATICS = RESULTS / "schematics"                 # hand-authored (moved from Exp3_PTO_GRPO/figures/ 2026-08-18)
REWARD = RESULTS / "lookahead" / "reward" / "figures"
BEHAVIOUR = RESULTS / "lookahead" / "behaviour" / "figures"
MECHANISM = RESULTS / "lookahead" / "mechanism" / "figures"
REPLICATION = RESULTS / "lookahead" / "replication" / "figures"
COST = RESULTS / "compute" / "cost" / "figures"
DEST = HERE / "figures"

# (source path, destination filename). Destination names are what the .tex references.
FIGURES: list[tuple[Path, str]] = [
    # --- sec:setup — method schematics (hand-authored: no notebook, no judge) --------------------
    (SCHEMATICS / "pto_preference_tree.png", "method_pto_branch.png"),
    (SCHEMATICS / "grpo_group_rollout.png", "method_grpo_group.png"),
    # --- body figures ---------------------------------------------------------------------------
    (REWARD / "k_headline_q1q2.png", "k_contrast_headline_fig_q1q2.png"),          # sec:reward
    (COST / "compute_trajectory_col.png", "compute_axis_fig_trajectory_col.png"),  # sec:cost
    (BEHAVIOUR / "k_channels_grid.png", "k_contrast_headline_fig_channels.png"),   # sec:behaviour
    (MECHANISM / "tail_audit.png", "tail_audit_fig.png"),                          # sec:mechanism
    (REPLICATION / "crossgen_col.png", "crossgen_exp1_fig_col.png"),               # sec:iclr
    # --- appendix A (tables + K grids + DiD + sweep) ---------------------------------------------
    (REWARD / "k_delta_grid_gpt-4o-mini.png", "k_contrast_headline_fig_grid_primary.png"),
    (REWARD / "k_delta_grid_claude-haiku-4-5.png", "k_contrast_headline_fig_grid_heldout.png"),
    (REWARD / "k_did.png", "cross_k_multijudge_fig_did.png"),
    (COST / "budget_sweep.png", "compute_axis_fig_budget_sweep.png"),
    # --- appendix B (tails, behaviour detail, mechanism) -----------------------------------------
    (BEHAVIOUR / "session_shape.png", "session_shape_stability_fig_shape.png"),
    (BEHAVIOUR / "wai.png", "held_out_instruments_fig_wai.png"),
    (BEHAVIOUR / "hetero.png", "held_out_instruments_fig_hetero.png"),
    (BEHAVIOUR / "k_mici_composition_grid_gpt-4o-mini.png", "k_mici_composition_grid_gpt-4o-mini.png"),
    (BEHAVIOUR / "k_mici_composition_grid_claude-haiku-4-5.png", "k_mici_composition_grid_claude-haiku-4-5.png"),
    (MECHANISM / "dispersion.png", "dispersion_by_k_fig.png"),
    (MECHANISM / "faithfulness.png", "reward_faithfulness_fig.png"),
    (COST / "api_calls.png", "tail_audit_fig_api.png"),
    # NOTE: the other PNG/PDF files in ./figures/ (the *_fig.pdf twins, side-by-side variants,
    # k_overpraise_trajectory_*, k_mechanism_overpraise_*, session_shape_stability_fig_sd, ...) are
    # stale generator output no longer referenced by any .tex and no longer synced.
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
