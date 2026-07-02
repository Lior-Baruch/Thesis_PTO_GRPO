#!/usr/bin/env python
"""
render_views.py — regenerate ``results/{all,L0,L2,L5}/`` for the 6 Exp3 analysis notebooks.

Each notebook's cell 1 reads ``VIEW = os.environ.get("EDA_VIEW", "L0")``, so this driver simply
sets ``EDA_VIEW`` and executes the notebook via ``nbconvert`` (no notebook-JSON mutation, no
papermill). The executed copies are written to a throwaway ``--output-dir`` so the committed
notebooks' outputs are NOT churned — the deliverable is the ``results/`` tree the notebooks write
as a side effect (figures, tables, INDEX.md, _provenance.md).

Usage (run from the ``eda/`` directory, or anywhere — it cd's itself)::

    python render_views.py                 # all 4 views x all 6 notebooks (24 runs)
    python render_views.py L0              # just the L0 view
    python render_views.py L0 all          # L0 then all
    python render_views.py L5 --nb 0 1     # L5 view, only notebooks 0_ and 1_
    python render_views.py --list          # print the view + notebook lists and exit

Needs the ``thesis-venv313`` Jupyter kernel (the venv with torch/trl/pandas). Register it once:
    .venv\\Scripts\\python.exe -m ipykernel install --user --name thesis-venv313

The hand-authored ``results/<view>/SUMMARY.md`` is never touched by this driver.
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = ["all", "L0", "L2", "L5"]
NOTEBOOKS = [
    "0_Headline.ipynb",
    "1_Eval_and_Behavior.ipynb",
    "2_Training_Diagnostics.ipynb",
    "3_Reward_Reliability.ipynb",
    "4_Preference_LatentSpace.ipynb",
    "5_Detailed_Stats.ipynb",
]
KERNEL = "thesis-venv313"
TIMEOUT = 1800  # seconds per notebook (the preference embedding cell is the slow one)


def run_one(view: str, nb: str, outdir: str) -> bool:
    """Execute one notebook under EDA_VIEW=<view>; return True on success."""
    env = {**os.environ, "EDA_VIEW": view, "WANDB_MODE": "offline"}
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        "--output-dir", outdir,
        os.path.join(HERE, nb),
    ]
    print(f"[render] view={view:<3} nb={nb}")
    res = subprocess.run(cmd, env=env, cwd=HERE)
    if res.returncode != 0:
        print(f"[render] FAILED view={view} nb={nb} (exit {res.returncode})")
    return res.returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate results/{all,L0,L5}/ for the Exp3 EDA notebooks.")
    ap.add_argument("views", nargs="*", default=None,
                    help="views to render (subset of all/L0/L5); default = all three")
    ap.add_argument("--nb", nargs="*", type=int, default=None,
                    help="notebook indices to render (0..5); default = all six")
    ap.add_argument("--list", action="store_true", help="print the view/notebook lists and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("views:", VIEWS)
        print("notebooks:", {i: nb for i, nb in enumerate(NOTEBOOKS)})
        return 0

    views = args.views or VIEWS
    bad = [v for v in views if v not in VIEWS]
    if bad:
        ap.error(f"unknown view(s) {bad}; choose from {VIEWS}")
    notebooks = [NOTEBOOKS[i] for i in args.nb] if args.nb is not None else NOTEBOOKS

    failures = []
    with tempfile.TemporaryDirectory(prefix="eda_render_") as tmp:
        for view in views:
            for nb in notebooks:
                if not run_one(view, nb, tmp):
                    failures.append((view, nb))

    print("\n" + "=" * 60)
    if failures:
        print(f"DONE with {len(failures)} failure(s):")
        for v, nb in failures:
            print(f"  - view={v} nb={nb}")
        return 1
    print(f"DONE — rendered {len(views)} view(s) x {len(notebooks)} notebook(s), no failures.")
    print("results trees:", [os.path.join("results", v) for v in views])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
