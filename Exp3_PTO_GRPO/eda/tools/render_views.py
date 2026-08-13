#!/usr/bin/env python
"""
render_views.py — regenerate ``results/{L0,L5,all}/`` for the 8 Exp3 analysis notebooks.

Each notebook's cell 1 reads ``VIEW = os.environ.get("EDA_VIEW", "L0")``, so this driver simply
sets ``EDA_VIEW`` and executes the notebook via ``nbconvert`` (no notebook-JSON mutation, no
papermill). The executed copies are written to a throwaway ``--output-dir`` so the committed
notebooks' outputs are NOT churned — the deliverable is the ``results/`` tree the notebooks write
as a side effect (figures, tables, INDEX.md, _provenance.md).

**Speed — the unit is a (view, judge) pair, and each unit shares ONE kernel.**

A bare run renders **L0 + L5** on the primary oracle; ``all`` is a merged SUPERSET of L0+L5 that
rarely earns its render cost, so it is opt-in (``render_views.py all``).

Two things make it fast, both measured on 2026-08-13:

1. **One kernel per unit, not per notebook.** Every notebook used to get its own ``nbconvert``
   subprocess, paying ~3.6 s of kernel spawn + ~1.4 s of ``import eda_analysis`` + ~2.7 s of
   ``notebook_setup`` before doing any work — ~7.7 s × 26 executions ≈ 200 s of a full refresh
   spent on nothing. A unit now starts one kernel and feeds it every notebook in order, with
   ``%reset -f`` between them so the user namespace is still empty at each notebook's first cell
   (a notebook that leans on a predecessor's globals fails here exactly as it would in Jupyter).
   ``--isolate`` restores the old one-kernel-per-notebook path for diagnosing a suspected leak.
2. **Judge passes are parallel units, not a second command.** ``--all-judges`` renders the primary
   AND every judge on disk as concurrent units. L0-primary / L5-primary / L0-judge / L5-judge write
   disjoint trees, so all four run at once.

Within a unit the notebooks stay **sequential** — they share that view's ``INDEX.md`` + per-family
``CAPTIONS.md`` (``build_index`` rewrites them), so concurrency there would race those files.

For a one-figure tweak, render just the affected notebook (``render_views.py L0 --nb 2``).

Usage (run from the ``eda/`` directory, or anywhere — it cd's itself)::

    python render_views.py                 # L0 + L5, primary oracle (the common case)
    python render_views.py --all-judges    # FULL REFRESH: every view x every judge, in parallel
    python render_views.py L0              # just the L0 view (the meeting view)
    python render_views.py all             # the merged superset view (opt-in)
    python render_views.py L0 --nb 3       # L0 view, only 3_Validity_and_Hacking (one-figure tweak)
    python render_views.py --judge anthropic_claude-haiku-4-5    # one named grader
    python render_views.py --jobs 1        # force sequential (low memory)
    python render_views.py --isolate       # one kernel per notebook (slow; debugging only)
    python render_views.py --list          # print the view + notebook lists and exit

Notebook numbering == results family numbering (1_Outcomes → figures/1_outcomes/, …), and ``--nb``
takes exactly those numbers (``--nb 3`` = ``3_Validity_and_Hacking.ipynb``).

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
import time

HERE = os.path.dirname(os.path.abspath(__file__))          # .../eda/tools
EDA_DIR = os.path.dirname(HERE)                            # .../eda
NB_DIR = os.path.join(EDA_DIR, "notebooks", "analysis")    # the seven free analysis notebooks

# This script lives in eda/tools/, so Python puts TOOLS on sys.path — not eda/. Anything here
# that imports the package (the --judge validation) needs eda/ added explicitly; without it the
# import fails only on that one branch, which is exactly the kind of gap a smoke test misses.
if EDA_DIR not in sys.path:
    sys.path.insert(0, EDA_DIR)
# VIEWS = the views that MAY be requested. DEFAULT_VIEWS = what a bare run renders. `all` is a
# merged SUPERSET of L0+L5 that rarely earns its render cost, so it is opt-in (request it
# explicitly). A new K view (e.g. L2) is added here + in config._VIEW_KS once its data lands.
VIEWS = ["all", "L0", "L5"]
DEFAULT_VIEWS = ["L0", "L5"]
# Topic notebooks — notebook number == results family number (figures|tables/N_<family>/).
NOTEBOOKS = [
    "1_Outcomes.ipynb",
    "2_Questionnaire_Detail.ipynb",
    "3_Validity_and_Hacking.ipynb",
    "4_Heterogeneity.ipynb",
    "5_Training.ipynb",
    "6_Preference.ipynb",
    "7_Stats.ipynb",
    "8_Measurement_Validity.ipynb",
]
NB_BY_NUMBER = {int(nb.split("_")[0]): nb for nb in NOTEBOOKS}
KERNEL = "thesis-venv313"
TIMEOUT = 1800  # seconds per notebook (the preference embedding cell is the slow one)
MAX_PARALLEL_VIEWS = 4  # cap default parallelism — each concurrent view is one live nbconvert kernel


# ── Notebooks a --judge render skips, for two OPPOSITE reasons ─────────────────
# TRAINING-SIDE: they read candidate rewards, preference pairs and TB curves, produced by the
# training oracle during the run and impossible to re-grade. Rendering them under another grader
# would emit byte-identical figures under that grader's name — a measurement that never happened.
TRAINING_SIDE_NOTEBOOKS = {"5_Training.ipynb", "6_Preference.ipynb"}
# JUDGE-INVARIANT: notebook 8 already contains EVERY grader. reliability.py loads each judge from
# the score lake explicitly and ignores EDA_JUDGE, and its family (`8_measurement`) is exported with
# no <judge>/ level (exports.JUDGE_INVARIANT_GROUPS), so a per-judge render would rewrite the same
# files with the same bytes. It is rendered exactly once per view, on the primary pass.
JUDGE_INVARIANT_NOTEBOOKS = {"8_Measurement_Validity.ipynb"}


def _unit_env(view: str, judge: str) -> dict:
    env = {**os.environ, "EDA_VIEW": view, "WANDB_MODE": "offline", "MPLBACKEND": "Agg"}
    if judge:
        env["EDA_JUDGE"] = judge
    else:
        env.pop("EDA_JUDGE", None)
    return env


def run_one(view: str, nb: str, outdir: str, judge: str = "") -> bool:
    """Execute one notebook in its OWN nbconvert subprocess; True on success.

    The isolated path (``--isolate``). Correct but slow: every notebook pays a cold kernel spawn
    (~3.6 s) plus ``import eda_analysis`` (~1.4 s) before doing any work. :func:`run_unit` is the
    default and shares one kernel across a unit's notebooks; keep this one as the escape hatch for
    diagnosing a suspected cross-notebook state leak.
    """
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        "--output-dir", outdir,
        os.path.join(NB_DIR, nb),
    ]
    print(f"[render] view={view:<3} judge={judge or 'primary':<28} nb={nb}", flush=True)
    res = subprocess.run(cmd, env=_unit_env(view, judge), cwd=NB_DIR)
    if res.returncode != 0:
        print(f"[render] FAILED view={view} nb={nb} (exit {res.returncode})", flush=True)
    return res.returncode == 0


# Cleared between notebooks in a shared kernel. `%reset -f` drops the USER namespace but leaves
# sys.modules intact — which is the whole point: the next notebook's `import eda_analysis` is a
# dict lookup instead of a 1.4 s import, while its variables still start empty, so a notebook that
# accidentally depended on a predecessor's globals still fails here exactly as it would in Jupyter.
_RESET_SRC = "%reset -f\nimport matplotlib.pyplot as _plt; _plt.close('all'); del _plt\n"


def run_unit(view: str, notebooks, judge: str = ""):
    """Render every notebook of ONE (view, judge) unit in ONE shared kernel.

    Sequential within a unit is REQUIRED: the notebooks share that view's ``INDEX.md`` + per-family
    ``CAPTIONS.md`` (each notebook's last cell calls ``build_index`` → ``prune_orphan_captions``),
    so running them concurrently would race those shared files. Parallelism happens ACROSS units,
    which write disjoint ``results/<view>/.../<judge>/`` trees.

    Sharing the kernel is what makes this fast: a unit pays the kernel spawn and the package import
    ONCE instead of once per notebook. Namespace isolation is preserved via :data:`_RESET_SRC`.
    Falls back to the per-notebook subprocess path if the kernel cannot be started.
    """
    import nbformat
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient

    tag = f"view={view:<3} judge={judge or 'primary':<28}"
    km = KernelManager(kernel_name=KERNEL)
    try:
        km.start_kernel(cwd=NB_DIR, env=_unit_env(view, judge))
    except Exception as ex:                       # no kernelspec / kernel refuses to boot
        print(f"[render] {tag} shared kernel unavailable ({type(ex).__name__}: {ex}); "
              f"falling back to one subprocess per notebook", flush=True)
        with tempfile.TemporaryDirectory(prefix="eda_iso_") as d:
            return [(view, nb) for nb in notebooks if not run_one(view, nb, d, judge)]

    failures = []
    try:
        for nb_name in notebooks:
            t0 = time.time()
            print(f"[render] {tag} nb={nb_name}", flush=True)
            nb = nbformat.read(os.path.join(NB_DIR, nb_name), as_version=4)
            client = NotebookClient(nb, km=km, timeout=TIMEOUT, kernel_name=KERNEL,
                                    allow_errors=False,
                                    resources={"metadata": {"path": NB_DIR}})
            try:
                client.execute()
                print(f"[render] {tag} nb={nb_name} OK {time.time() - t0:6.1f}s", flush=True)
            except Exception as ex:
                print(f"[render] FAILED {tag} nb={nb_name} "
                      f"({type(ex).__name__}: {str(ex)[:200]})", flush=True)
                failures.append((view, nb_name))
            finally:
                # Always reset, even after a failure — otherwise a half-executed notebook's globals
                # leak into the next one and turn one failure into a confusing cascade.
                try:
                    reset = nbformat.v4.new_notebook(
                        cells=[nbformat.v4.new_code_cell(_RESET_SRC)])
                    NotebookClient(reset, km=km, timeout=120, kernel_name=KERNEL,
                                   allow_errors=True,
                                   resources={"metadata": {"path": NB_DIR}}).execute()
                except Exception:
                    pass
    finally:
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate results/<view>/ for the Exp3 EDA notebooks (views rendered in parallel).")
    ap.add_argument("views", nargs="*", default=None,
                    help="views to render (subset of all/L0/L5); default = L0 L5 (all is opt-in)")
    ap.add_argument("--nb", nargs="*", type=int, default=None,
                    help="notebook NUMBERS to render (1..8 — the filename/family number, "
                         "e.g. 3 = 3_Validity_and_Hacking); default = all eight")
    ap.add_argument("--jobs", "-j", type=int, default=None,
                    help=f"parallel UNITS, a unit being one (view, judge) pair (default = #units, "
                         f"capped at {MAX_PARALLEL_VIEWS}); 1 = sequential")
    ap.add_argument("--judge", nargs="*", default=None,
                    help="score source(s): omit for the primary oracle, or pass one or more judge "
                         "tags such as anthropic_claude-haiku-4-5 -> figures/<family>/<judge>/. "
                         "Training-side notebooks (5, 6) and the judge-invariant one (8) are "
                         "skipped for a non-primary judge.")
    ap.add_argument("--all-judges", action="store_true",
                    help="render the primary oracle AND every judge scored on disk, as parallel "
                         "units — the full refresh, in one command.")
    ap.add_argument("--isolate", action="store_true",
                    help="one kernel per notebook (the old, slow path). Use only to diagnose a "
                         "suspected cross-notebook state leak.")
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
    judges = list(args.judge or [])
    if args.all_judges:
        from eda_analysis import reliability as _rel
        judges = [""] + [j for j in _rel.judge_tags() if j not in judges]
    if not judges:
        judges = [""]                                    # primary oracle only
    if any(judges):
        from eda_analysis import reliability as _rel
        known = _rel.judge_tags()
        bad_j = [j for j in judges if j and j not in known]
        if bad_j:
            ap.error(f"unknown judge(s) {bad_j}; scored judges on disk: {known or '(none)'}")

    # ── Collapse judges that share an OUTPUT DIRECTORY ────────────────────────
    # A unit is safe to run concurrently only because units own disjoint trees, and the tree is
    # keyed by `judge_dirname(tag)`, NOT by the tag. The primary oracle has BOTH spellings —
    # "" and its own tag `openai_gpt-4o-mini-2024-07-18` — and both resolve to `gpt-4o-mini/`.
    # `--all-judges` therefore used to emit the primary twice and run the two copies in parallel
    # against the same files, with `reset_results()` (judge-scoped) able to delete one writer's
    # output mid-run. Dedupe on the directory, keeping the first spelling seen so "" wins.
    from eda_analysis.constants import judge_dirname
    seen_dirs, deduped = {}, []
    for j in judges:
        d = judge_dirname(j)
        if d in seen_dirs:
            print(f"[render] judge {j or '(primary)'!r} writes to the same tree as "
                  f"{seen_dirs[d] or '(primary)'!r} ({d}/) — rendering it once.", flush=True)
            continue
        seen_dirs[d] = j
        deduped.append(j)
    judges = deduped

    # A UNIT is one (view, judge) pair — the smallest thing that owns a disjoint output tree, so
    # units are what we parallelise over. Previously only views were parallel and each judge pass
    # was a separate sequential command, which serialised the two halves of a full refresh.
    def notebooks_for(judge: str):
        if not judge:
            return notebooks
        return [n for n in notebooks
                if n not in TRAINING_SIDE_NOTEBOOKS and n not in JUDGE_INVARIANT_NOTEBOOKS]

    for judge in judges:
        if not judge:
            continue
        skipped = [n for n in notebooks if n in TRAINING_SIDE_NOTEBOOKS]
        invariant = [n for n in notebooks if n in JUDGE_INVARIANT_NOTEBOOKS]
        if skipped:
            print(f"[render] judge={judge}: skipping training-side notebook(s) "
                  f"{', '.join(skipped)} — their scores come from the training oracle and "
                  f"cannot be re-graded.", flush=True)
        if invariant:
            print(f"[render] judge={judge}: skipping judge-invariant notebook(s) "
                  f"{', '.join(invariant)} — they already read every grader and export with no "
                  f"<judge>/ level; render them on the primary pass.", flush=True)

    units = [(v, j) for j in judges for v in views if notebooks_for(j)]
    if not units:
        print("[render] nothing to render.")
        return 0
    jobs = args.jobs if args.jobs is not None else min(len(units), MAX_PARALLEL_VIEWS)
    jobs = max(1, min(jobs, len(units)))

    n_nb = sum(len(notebooks_for(j)) for _, j in units)
    mode = "one kernel per notebook (--isolate)" if args.isolate else "one shared kernel per unit"
    print(f"[render] {len(units)} unit(s) / {n_nb} notebook execution(s), "
          f"{jobs} in parallel, {mode}", flush=True)
    t_start = time.time()

    def render_unit(unit):
        view, judge = unit
        nbs = notebooks_for(judge)
        if args.isolate:
            with tempfile.TemporaryDirectory(prefix="eda_iso_") as d:
                return [(view, nb) for nb in nbs if not run_one(view, nb, d, judge)]
        return run_unit(view, nbs, judge)

    failures = []
    if jobs == 1:
        for unit in units:
            failures += render_unit(unit)
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            for fails in ex.map(render_unit, units):
                failures += fails

    print("\n" + "=" * 60)
    if failures:
        print(f"DONE with {len(failures)} failure(s) in {time.time() - t_start:.0f}s:")
        for v, nb in failures:
            print(f"  - view={v} nb={nb}")
        return 1
    print(f"DONE — {len(units)} unit(s), {n_nb} notebook execution(s), no failures, "
          f"{time.time() - t_start:.0f}s.")
    print("results trees:", [os.path.join("results", v) for v in views])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
