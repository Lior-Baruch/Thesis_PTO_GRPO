#!/usr/bin/env python
"""
render_views.py — regenerate ``results/{L0,L5,all}/`` for the 6 Exp3 analysis notebooks.

Each notebook's cell 1 reads ``VIEW = os.environ.get("EDA_VIEW", "L0")``, so this driver simply
sets ``EDA_VIEW`` and executes the notebook via ``nbconvert`` (no notebook-JSON mutation, no
papermill). The executed copies are written to a throwaway ``--output-dir`` so the committed
notebooks' outputs are NOT churned — the deliverable is the ``results/`` tree the notebooks write
as a side effect (figures, tables, INDEX.md, _provenance.md).

**Speed.** Views are rendered **in parallel** — one worker per view (``--jobs`` to tune) — and a
bare run renders only **L0 + L5**, the two views that hold distinct data. ``all`` is a merged
SUPERSET of L0+L5 that rarely earns its render cost, so it is now **opt-in** (``render_views.py
all``). Within a view the 6 notebooks run **sequentially**: they share that view's ``INDEX.md`` +
per-family ``CAPTIONS.md`` (``build_index`` rewrites them), so parallelism is ACROSS views, never
within one. For a one-figure tweak, render just the affected notebook of the view you need
(``render_views.py L0 --nb 2``) — far cheaper than a full sweep.

Usage (run from the ``eda/`` directory, or anywhere — it cd's itself)::

    python render_views.py                 # default = L0 + L5, in parallel (fast)
    python render_views.py L0              # just the L0 view (the meeting view)
    python render_views.py all             # the merged superset view (opt-in)
    python render_views.py all L0 L5       # all three, in parallel
    python render_views.py L0 --nb 3       # L0 view, only 3_Mechanism (one-figure tweak)
    python render_views.py --jobs 1        # force sequential (low memory)
    python render_views.py --list          # print the view + notebook lists and exit

Notebook numbering == results family numbering (1_Outcomes → figures/1_outcomes/, …), and ``--nb``
takes exactly those numbers (``--nb 3`` = ``3_Mechanism.ipynb``).

Needs the ``thesis-venv313`` Jupyter kernel (the venv with torch/trl/pandas). Register it once:
    .venv\\Scripts\\python.exe -m ipykernel install --user --name thesis-venv313

The hand-authored ``results/<view>/SUMMARY.md`` is never touched by this driver.
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# VIEWS = the views that MAY be requested. DEFAULT_VIEWS = what a bare run renders. `all` is a
# merged SUPERSET of L0+L5 that rarely earns its render cost, so it is opt-in (request it
# explicitly). A new K view (e.g. L2) is added here + in config._VIEW_KS once its data lands.
VIEWS = ["all", "L0", "L5"]
DEFAULT_VIEWS = ["L0", "L5"]
# Topic notebooks — notebook number == results family number (figures|tables/N_<family>/).
NOTEBOOKS = [
    "1_Outcomes.ipynb",
    "2_Heterogeneity.ipynb",
    "3_Mechanism.ipynb",
    "4_Training_and_Reliability.ipynb",
    "5_Preference.ipynb",
    "6_Stats.ipynb",
]
NB_BY_NUMBER = {int(nb.split("_")[0]): nb for nb in NOTEBOOKS}
KERNEL = "thesis-venv313"
TIMEOUT = 1800  # seconds per notebook (the preference embedding cell is the slow one)
MAX_PARALLEL_VIEWS = 4  # cap default parallelism — each concurrent view is one live nbconvert kernel


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
    print(f"[render] view={view:<3} nb={nb}", flush=True)
    res = subprocess.run(cmd, env=env, cwd=HERE)
    if res.returncode != 0:
        print(f"[render] FAILED view={view} nb={nb} (exit {res.returncode})", flush=True)
    return res.returncode == 0


def run_view(view: str, notebooks, tmp_root: str):
    """Render every notebook of ONE view, sequentially; return the list of (view, nb) failures.

    Sequential within a view is REQUIRED: the notebooks share that view's ``INDEX.md`` + per-family
    ``CAPTIONS.md`` (each notebook's last cell calls ``build_index`` → ``prune_orphan_captions``),
    so running them concurrently would race those shared files. Parallelism happens across views
    (disjoint ``results/<view>/`` trees). Each view also gets its own throwaway output dir so two
    concurrent views can't collide on the executed-notebook filename (both write e.g.
    ``3_Mechanism.ipynb``).
    """
    outdir = os.path.join(tmp_root, view)
    os.makedirs(outdir, exist_ok=True)
    return [(view, nb) for nb in notebooks if not run_one(view, nb, outdir)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate results/<view>/ for the Exp3 EDA notebooks (views rendered in parallel).")
    ap.add_argument("views", nargs="*", default=None,
                    help="views to render (subset of all/L0/L5); default = L0 L5 (all is opt-in)")
    ap.add_argument("--nb", nargs="*", type=int, default=None,
                    help="notebook NUMBERS to render (1..6 — the filename/family number, "
                         "e.g. 3 = 3_Mechanism); default = all six")
    ap.add_argument("--jobs", "-j", type=int, default=None,
                    help=f"parallel views (default = #views, capped at {MAX_PARALLEL_VIEWS}); 1 = sequential")
    ap.add_argument("--list", action="store_true", help="print the view/notebook lists and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("views:", VIEWS, "  default:", DEFAULT_VIEWS)
        print("notebooks (--nb number: file):", NB_BY_NUMBER)
        return 0

    views = args.views or DEFAULT_VIEWS
    bad = [v for v in views if v not in VIEWS]
    if bad:
        ap.error(f"unknown view(s) {bad}; choose from {VIEWS}")
    if args.nb is not None:
        bad_nb = [n for n in args.nb if n not in NB_BY_NUMBER]
        if bad_nb:
            ap.error(f"unknown notebook number(s) {bad_nb}; choose from {sorted(NB_BY_NUMBER)} "
                     f"(the filename/family number, e.g. 3 = {NB_BY_NUMBER[3]})")
        notebooks = [NB_BY_NUMBER[n] for n in args.nb]
    else:
        notebooks = NOTEBOOKS
    jobs = args.jobs if args.jobs is not None else min(len(views), MAX_PARALLEL_VIEWS)
    jobs = max(1, min(jobs, len(views)))

    failures = []
    with tempfile.TemporaryDirectory(prefix="eda_render_") as tmp:
        if jobs == 1:
            for view in views:
                failures += run_view(view, notebooks, tmp)
        else:
            print(f"[render] {len(views)} view(s) x {len(notebooks)} notebook(s), "
                  f"{jobs} views in parallel", flush=True)
            with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
                for fails in ex.map(lambda v: run_view(v, notebooks, tmp), views):
                    failures += fails

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
