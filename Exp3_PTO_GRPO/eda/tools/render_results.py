#!/usr/bin/env python
"""
render_results.py — regenerate ``results/<top>/<sub>/`` for the Exp3 analysis notebooks (2026-08-18).

Replaces ``render_views.py`` (the retired VIEW = ``L0``/``L5`` driver). The results tree is now
organised by **family** — ``config.FAMILIES`` maps each top (``arms``, ``lookahead``, ``method``,
``compute``, ``measurement``) to its subfamilies, and every family is ONE notebook,
``notebooks/<top>/<sub>.ipynb``, whose cell 1 reads::

    cfg = eda_analysis.EdaConfig(family="<top>/<sub>", judge=os.environ.get("EDA_JUDGE", ""))

so this driver only sets ``EDA_JUDGE`` and executes the notebook (no notebook-JSON mutation, no
papermill). Executed copies go to a throwaway output dir; the deliverable is the ``results/`` tree
the notebooks write as a side effect (figures, tables, ledgers, INDEX.md, _provenance.md).

**Units.** The unit of work is ``(top, judge)``:

* ``arms`` (``config.PER_JUDGE_TOPS``) is rendered **once per grader on disk** — its artifacts are
  produced by one judge and land under ``arms/<sub>/{figures,tables}/<judge>/``. A bare run
  renders it for EVERY judge in the score lake, so a held-out judge's leaf can no longer go stale
  silently (the old ``render_views.py`` bare run was primary-only).
* every other top is **judge-invariant** (its notebooks load both graders via
  ``scores_by_judge`` and export with no ``<judge>/`` level) and is rendered exactly once, with
  ``EDA_JUDGE`` unset.

Each unit starts ONE kernel and feeds it its notebooks sequentially in ``FAMILIES`` order, with
``%reset -f`` between them (mechanism inherited from ``render_views.py``, measured 2026-08-13:
one kernel per unit saves ~7.7 s of spawn+import+setup per notebook). Sequential within a unit is
REQUIRED — the notebooks of one top share ``results/<top>/INDEX.md`` and their leaves'
``CAPTIONS.md`` (each notebook's last cell calls ``build_index``), so concurrency there would race
those files. Units run in parallel (``--jobs``); two ``arms`` units for different judges write
disjoint ``<judge>/`` leaves (they both rewrite ``arms/INDEX.md``, atomically — last writer wins,
and the last one to finish sees the complete tree).

A notebook that does not exist yet is SKIPPED with a note, not an error — the reorg lands the
notebooks in a later phase than this driver.

Usage (run from anywhere — it cd's itself)::

    python render_results.py                          # everything: arms x every judge + the 4 invariant tops
    python render_results.py --top arms               # one top (arms -> every judge on disk)
    python render_results.py --top lookahead compute  # several tops
    python render_results.py --family lookahead/reward         # one notebook
    python render_results.py --top arms --judge anthropic_claude-haiku-4-5   # one grader's arms/* leaves
    python render_results.py --judge ""               # arms/* for the primary oracle only (+ invariant tops)
    python render_results.py --jobs 1                 # force sequential (low memory)
    python render_results.py --isolate                # one kernel per notebook (slow; debugging only)
    python render_results.py --list                   # print the family/judge/unit plan and exit

Needs the ``thesis-venv313`` Jupyter kernel (the venv with torch/trl/pandas). Register it once:
    .venv\\Scripts\\python.exe -m ipykernel install --user --name thesis-venv313

Hand-authored files (``results/<top>/SUMMARY.md``, ``METRICS_REFERENCE.md``, ``LIMITATIONS.md``,
``schematics/``) are never touched by this driver.
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
NB_ROOT = os.path.join(EDA_DIR, "notebooks")               # notebooks/<top>/<sub>.ipynb

# This script lives in eda/tools/, so Python puts TOOLS on sys.path — not eda/. The package (the
# family map, the judge list) needs eda/ added explicitly.
if EDA_DIR not in sys.path:
    sys.path.insert(0, EDA_DIR)

from eda_analysis.config import FAMILIES, PER_JUDGE_TOPS, all_families  # noqa: E402

KERNEL = "thesis-venv313"
TIMEOUT = 1800  # seconds per notebook (the preference embedding cell is the slow one)
MAX_PARALLEL_UNITS = 4  # cap default parallelism — each concurrent unit is one live kernel


def notebook_path(family: str) -> str:
    top, sub = family.split("/")
    return os.path.join(NB_ROOT, top, f"{sub}.ipynb")


def _unit_env(judge: str) -> dict:
    env = {**os.environ, "WANDB_MODE": "offline", "MPLBACKEND": "Agg"}
    env.pop("EDA_VIEW", None)                        # the retired knob must never leak in
    if judge:
        env["EDA_JUDGE"] = judge
    else:
        env.pop("EDA_JUDGE", None)
    return env


def _tag(top: str, judge: str) -> str:
    return f"top={top:<11} judge={judge or 'primary':<28}"


def run_one(top: str, family: str, outdir: str, judge: str = "") -> bool:
    """Execute one notebook in its OWN nbconvert subprocess; True on success.

    The isolated path (``--isolate``). Correct but slow: every notebook pays a cold kernel spawn
    plus ``import eda_analysis`` before doing any work. :func:`run_unit` is the default and shares
    one kernel across a unit's notebooks; keep this one as the escape hatch for diagnosing a
    suspected cross-notebook state leak.
    """
    nb = notebook_path(family)
    nb_dir = os.path.dirname(nb)
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        "--output-dir", outdir,
        nb,
    ]
    print(f"[render] {_tag(top, judge)} nb={family}", flush=True)
    res = subprocess.run(cmd, env=_unit_env(judge), cwd=nb_dir)
    if res.returncode != 0:
        print(f"[render] FAILED {_tag(top, judge)} nb={family} (exit {res.returncode})", flush=True)
    return res.returncode == 0


# Cleared between notebooks in a shared kernel. `%reset -f` drops the USER namespace but leaves
# sys.modules intact — which is the whole point: the next notebook's `import eda_analysis` is a
# dict lookup instead of a 1.4 s import, while its variables still start empty, so a notebook that
# accidentally depended on a predecessor's globals still fails here exactly as it would in Jupyter.
_RESET_SRC = "%reset -f\nimport matplotlib.pyplot as _plt; _plt.close('all'); del _plt\n"


def run_unit(top: str, families, judge: str = ""):
    """Render every notebook of ONE (top, judge) unit in ONE shared kernel.

    Sequential within a unit is REQUIRED: the notebooks share ``results/<top>/INDEX.md`` + their
    leaves' ``CAPTIONS.md`` (each notebook's last cell calls ``build_index`` →
    ``prune_orphan_captions``), so running them concurrently would race those shared files.
    Parallelism happens ACROSS units.

    Falls back to the per-notebook subprocess path if the kernel cannot be started. Returns the
    list of ``(top, family)`` failures.
    """
    import nbformat
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient

    tag = _tag(top, judge)
    nb_dir = os.path.join(NB_ROOT, top)
    km = KernelManager(kernel_name=KERNEL)
    try:
        km.start_kernel(cwd=nb_dir, env=_unit_env(judge))
    except Exception as ex:                       # no kernelspec / kernel refuses to boot
        print(f"[render] {tag} shared kernel unavailable ({type(ex).__name__}: {ex}); "
              f"falling back to one subprocess per notebook", flush=True)
        with tempfile.TemporaryDirectory(prefix="eda_iso_") as d:
            return [(top, fam) for fam in families if not run_one(top, fam, d, judge)]

    failures = []
    try:
        for fam in families:
            t0 = time.time()
            print(f"[render] {tag} nb={fam}", flush=True)
            nb = nbformat.read(notebook_path(fam), as_version=4)
            client = NotebookClient(nb, km=km, timeout=TIMEOUT, kernel_name=KERNEL,
                                    allow_errors=False,
                                    resources={"metadata": {"path": nb_dir}})
            try:
                client.execute()
                print(f"[render] {tag} nb={fam} OK {time.time() - t0:6.1f}s", flush=True)
            except Exception as ex:
                print(f"[render] FAILED {tag} nb={fam} "
                      f"({type(ex).__name__}: {str(ex)[:200]})", flush=True)
                failures.append((top, fam))
            finally:
                # Always reset, even after a failure — otherwise a half-executed notebook's globals
                # leak into the next one and turn one failure into a confusing cascade.
                try:
                    reset = nbformat.v4.new_notebook(
                        cells=[nbformat.v4.new_code_cell(_RESET_SRC)])
                    NotebookClient(reset, km=km, timeout=120, kernel_name=KERNEL,
                                   allow_errors=True,
                                   resources={"metadata": {"path": nb_dir}}).execute()
                except Exception:
                    pass
    finally:
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass
    return failures


def plan_units(tops, families, judges):
    """``[(top, judge, [family, ...]), ...]`` — per-judge tops × judges, invariant tops once.

    ``judges`` is the list of judge tags for the per-judge tops (``""`` = primary). Families whose
    notebook does not exist are dropped here with a note (the reorg lands notebooks after this
    driver), so a missing notebook is never a failure.
    """
    units = []
    for top in tops:
        fams = [f for f in families if f.split("/")[0] == top]
        present, missing = [], []
        for f in fams:
            (present if os.path.isfile(notebook_path(f)) else missing).append(f)
        if missing:
            print(f"[render] top={top}: skipping {len(missing)} family notebook(s) not on disk yet: "
                  f"{missing}", flush=True)
        if not present:
            continue
        if top in PER_JUDGE_TOPS:
            for j in judges:
                units.append((top, j, present))
        else:
            units.append((top, "", present))
    return units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate results/<top>/<sub>/ for the Exp3 EDA notebooks "
                    "(units = (top, judge); arms x every judge on disk, invariant tops once).")
    ap.add_argument("--top", nargs="*", default=None,
                    help=f"tops to render (subset of {list(FAMILIES)}); default = all")
    ap.add_argument("--family", nargs="*", default=None,
                    help="specific families '<top>/<sub>' to render (e.g. lookahead/reward); "
                         "default = every family of the selected tops")
    ap.add_argument("--judge", nargs="*", default=None,
                    help="judge tag(s) for the per-judge tops (arms/*): omit for EVERY grader on "
                         "disk; pass '' for the primary oracle only, or one or more tags such as "
                         "anthropic_claude-haiku-4-5. Judge-invariant tops ignore this.")
    ap.add_argument("--jobs", "-j", type=int, default=None,
                    help=f"parallel UNITS (default = #units, capped at {MAX_PARALLEL_UNITS}); "
                         f"1 = sequential")
    ap.add_argument("--isolate", action="store_true",
                    help="one kernel per notebook (the old, slow path). Use only to diagnose a "
                         "suspected cross-notebook state leak.")
    ap.add_argument("--list", action="store_true", help="print the family/judge/unit plan and exit")
    args = ap.parse_args(argv)

    tops = list(FAMILIES) if not args.top else list(dict.fromkeys(args.top))
    bad = [t for t in tops if t not in FAMILIES]
    if bad:
        ap.error(f"unknown top(s) {bad}; choose from {list(FAMILIES)}")
    families = all_families()
    if args.family:
        bad_f = [f for f in args.family if f not in families]
        if bad_f:
            ap.error(f"unknown family(ies) {bad_f}; choose from {families}")
        families = [f for f in families if f in set(args.family)]     # keep FAMILIES order
        tops = [t for t in tops if any(f.split("/")[0] == t for f in families)]

    # Judges for the per-judge tops: every grader on disk by default (a bare run must refresh the
    # held-out judge's leaves too — that silent staleness is what the old primary-only bare run
    # produced). Collapse tags that share an OUTPUT DIRECTORY: the primary has BOTH spellings ("" and
    # its own tag) and both resolve to `gpt-4o-mini/`; rendering both would race one tree.
    from eda_analysis import reliability as _rel
    from eda_analysis.constants import judge_dirname
    known = _rel.judge_tags()
    if args.judge is None:
        judges = [""] + list(known)
    else:
        judges = list(args.judge)
        bad_j = [j for j in judges if j and j not in known]
        if bad_j:
            ap.error(f"unknown judge(s) {bad_j}; scored judges on disk: {known or '(none)'}")
    seen_dirs, deduped = {}, []
    for j in judges:
        d = judge_dirname(j)
        if d in seen_dirs:
            if args.judge is not None:
                print(f"[render] judge {j or '(primary)'!r} writes to the same tree as "
                      f"{seen_dirs[d] or '(primary)'!r} ({d}/) — rendering it once.", flush=True)
            continue
        seen_dirs[d] = j
        deduped.append(j)
    judges = deduped or [""]

    units = plan_units(tops, families, judges)

    if args.list:
        print("families (config.FAMILIES order):")
        for f in all_families():
            mark = "present" if os.path.isfile(notebook_path(f)) else "MISSING notebook"
            pj = "per judge" if f.split("/")[0] in PER_JUDGE_TOPS else "invariant"
            print(f"  {f:<24} {pj:<10} {mark}")
        print("judges on disk:", known or "(none)", "| per-judge tops rendered for:",
              [j or "(primary)" for j in judges])
        print("units:")
        for top, j, fams in units:
            print(f"  ({top}, {j or 'primary'}): {fams}")
        return 0

    if not units:
        print("[render] nothing to render (no family notebooks on disk for the selection).")
        return 0
    jobs = args.jobs if args.jobs is not None else min(len(units), MAX_PARALLEL_UNITS)
    jobs = max(1, min(jobs, len(units)))

    n_nb = sum(len(fams) for _, _, fams in units)
    mode = "one kernel per notebook (--isolate)" if args.isolate else "one shared kernel per unit"
    print(f"[render] {len(units)} unit(s) / {n_nb} notebook execution(s), "
          f"{jobs} in parallel, {mode}", flush=True)
    t_start = time.time()

    def render_unit(unit):
        top, judge, fams = unit
        if args.isolate:
            with tempfile.TemporaryDirectory(prefix="eda_iso_") as d:
                return [(top, f) for f in fams if not run_one(top, f, d, judge)]
        return run_unit(top, fams, judge)

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
        for top, fam in failures:
            print(f"  - {fam}")
        return 1
    print(f"DONE — {len(units)} unit(s), {n_nb} notebook execution(s), no failures, "
          f"{time.time() - t_start:.0f}s.")
    print("results trees:", sorted({os.path.join("results", top) for top, _, _ in units}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
