#!/usr/bin/env python
"""render_results.py -- regenerate ``eda/results/`` by executing every family notebook.

One family is one research question, one notebook and one output folder::

    notebooks/<top>/<sub>.ipynb   -->   results/<top>/<sub>/{figures,tables}/

The notebooks are the source of truth for what a family reports, and ``results/`` is the
deliverable. This driver is what makes the second statement true: it re-runs every notebook
non-interactively so the rendered tree is reproducible from git rather than from whichever cells
someone last happened to run.

What it does NOT do is mutate the notebooks. Executed copies go to a throwaway temp directory and
the repo's ``.ipynb`` files are never written back, so they stay output-free and a render produces
no notebook diff at all -- only ``results/``.

The unit of work is ONE TOP
---------------------------
Exp3's unit was ``(top, judge)``, because its per-arm families wrote a ``<judge>/`` leaf and had to
be rendered once per grader. Exp4 has no judge level: every family loads whichever graders it wants
and puts them side by side inside one table. So a top is rendered exactly once, this driver sets no
``EDA_JUDGE`` (the judge is a config field, never ambient state), and the race Exp3 documented --
two units of the same top rewriting one ``INDEX.md`` -- cannot occur.

Within a top the notebooks run SEQUENTIALLY in one shared kernel. Sequential is required, not an
optimisation: the notebooks of a top share ``results/<top>/INDEX.md`` and their leaves'
``CAPTIONS.md``, and each one's closing ``build_index()`` rewrites both. Sharing a kernel saves the
cold-import cost per notebook; a ``%reset -f`` between them clears the user namespace (but not
``sys.modules``), so a notebook that accidentally depended on its predecessor's globals fails here
exactly as it would in Jupyter. Tops run in PARALLEL -- each writes a disjoint subtree.

The kernel
----------
No named kernelspec is required. By default the notebooks run in **the interpreter that runs this
script**, via a throwaway kernelspec written into the temp directory: running the driver with the
repo venv renders with the repo venv, and nothing has to be registered first. Exp3 hardcoded a
``thesis-venv313`` kernel name and failed with ``NoSuchKernel`` on any machine where it had not been
installed. ``--kernel NAME`` opts back into a registered kernelspec.

Usage (run from ``eda/``; it locates itself either way)::

    python tools/render_results.py                       # every family
    python tools/render_results.py --top arms lookahead  # some tops
    python tools/render_results.py --family method/contrast
    python tools/render_results.py --jobs 1              # sequential (low memory)
    python tools/render_results.py --list                # inventory + plan, then exit
    python tools/render_results.py --dry-run             # resolved plan, execute nothing
    python tools/render_results.py --kernel thesis-venv313

A family whose notebook is not on disk is SKIPPED with a note, not failed -- families and notebooks
land in different phases of a build-out, and a missing notebook is ``_selfcheck``'s "family map"
check to report, not this driver's job to die on.

Hand-authored files (``results/<top>/SUMMARY.md``, ``METRICS_REFERENCE.md``, ``LIMITATIONS.md``,
``schematics/``) are never touched: the notebooks write through ``eda_analysis.exports``, which
refuses to modify anything in its ``PRESERVE`` set.

Exit status is 0 only if every executed notebook succeeded.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))          # .../eda/tools
EDA_DIR = os.path.dirname(HERE)                            # .../eda
NB_ROOT = os.path.join(EDA_DIR, "notebooks")               # notebooks/<top>/<sub>.ipynb
RESULTS_ROOT = os.path.join(EDA_DIR, "results")

# This script lives in eda/tools/, so Python puts TOOLS on sys.path -- not eda/. The family map
# lives in the package, so eda/ has to be added explicitly.
if EDA_DIR not in sys.path:
    sys.path.insert(0, EDA_DIR)

from eda_analysis.config import FAMILIES, all_families  # noqa: E402

__all__ = ["main", "notebook_path", "plan_units", "run_unit"]


#: Seconds one notebook may run before it is killed. Generous: a family that reads every
#: conversation of every arm is minutes, and a timeout kill loses the whole unit's kernel.
DEFAULT_TIMEOUT = 1800

#: Default parallelism cap. Each concurrent unit is one live kernel holding a pandas working set,
#: so the ceiling is memory, not cores.
MAX_PARALLEL_UNITS = 4

#: Kernelspec name for the throwaway "run in this interpreter" kernel (see module docstring).
LOCAL_KERNEL_NAME = "exp4-render-local"

#: Cleared between notebooks in a shared kernel. ``%reset -f`` drops the USER namespace and leaves
#: ``sys.modules`` alone -- which is the point: the next notebook's ``import eda_analysis`` is a
#: dict lookup rather than a fresh import, while its variables still start empty.
_RESET_SOURCE = (
    "%reset -f\n"
    "import matplotlib.pyplot as _plt; _plt.close('all'); del _plt\n"
)


# ==============================================================================
#  Planning
# ==============================================================================


@dataclass
class Unit:
    """One top: the notebooks to execute, in ``FAMILIES`` order, plus the ones that are absent."""

    top: str
    families: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


@dataclass
class Outcome:
    """What happened to one family. ``status`` is ``ok`` | ``failed`` | ``skipped``."""

    family: str
    status: str
    seconds: float = 0.0
    detail: str = ""


def notebook_path(family: str) -> str:
    """``notebooks/<top>/<sub>.ipynb`` for a ``"<top>/<sub>"`` family."""
    top, sub = family.split("/")
    return os.path.join(NB_ROOT, top, f"{sub}.ipynb")


def plan_units(tops: Sequence[str], families: Sequence[str]) -> List[Unit]:
    """Group the selected families into one :class:`Unit` per top, in ``FAMILIES`` order.

    A family whose notebook does not exist is recorded in ``Unit.missing`` rather than dropped, so
    the summary can say it was skipped instead of silently reporting a shorter run. A top with no
    notebook at all yields no unit (there is nothing to start a kernel for).
    """
    units: List[Unit] = []
    for top in tops:
        unit = Unit(top=top)
        for family in families:
            if family.split("/")[0] != top:
                continue
            (unit.families if os.path.isfile(notebook_path(family)) else unit.missing).append(
                family)
        if unit.families or unit.missing:
            units.append(unit)
    return units


# ==============================================================================
#  Kernel
# ==============================================================================


@dataclass(frozen=True)
class Kernel:
    """How to start a kernel: a registered name, or a throwaway spec in ``search_dirs``."""

    name: str
    search_dirs: Tuple[str, ...] = ()
    description: str = ""

    def manager(self, *, cwd: str, env: Dict[str, str]):
        """Start and return a live ``KernelManager``.

        A fresh ``KernelSpecManager`` per unit: the driver runs units in a thread pool, and one
        shared traitlets object doing filesystem lookups from several threads is a needless risk
        for an object this cheap to build.
        """
        from jupyter_client.kernelspec import KernelSpecManager
        from jupyter_client.manager import KernelManager

        ksm = KernelSpecManager(kernel_dirs=list(self.search_dirs)) if self.search_dirs \
            else KernelSpecManager()
        km = KernelManager(kernel_name=self.name, kernel_spec_manager=ksm)
        km.start_kernel(cwd=cwd, env=env)
        return km


def resolve_kernel(explicit: Optional[str], workdir: str) -> Kernel:
    """The kernel to render with: *explicit* if given, otherwise this interpreter.

    Args:
        explicit: A registered kernelspec name (``--kernel``). Validated here so an unknown name
            fails before any notebook is copied, with the available names in the message.
        workdir: Where to write the throwaway kernelspec for the default path.

    Returns:
        A :class:`Kernel`.

    Notes:
        The default deliberately does NOT require a registered kernel. Running
        ``.venv/Scripts/python.exe tools/render_results.py`` renders with that venv, full stop --
        which is the same interpreter ``_selfcheck`` and the notebooks' imports were verified
        against. A registered kernelspec is an extra installation step whose only effect is the
        chance of pointing at a different environment than the one the caller chose.
    """
    if explicit:
        from jupyter_client.kernelspec import KernelSpecManager

        ksm = KernelSpecManager()
        available = sorted(ksm.find_kernel_specs())
        if explicit not in available:
            raise SystemExit(
                f"[render] unknown kernel {explicit!r}; registered kernelspecs: {available}. "
                f"Omit --kernel to render with this interpreter ({sys.executable}).")
        return Kernel(name=explicit, description=f"kernelspec {explicit!r}")

    spec_root = os.path.join(workdir, "_kernelspec")
    spec_dir = os.path.join(spec_root, LOCAL_KERNEL_NAME)
    os.makedirs(spec_dir, exist_ok=True)
    with open(os.path.join(spec_dir, "kernel.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Exp4 render (this interpreter)",
            "language": "python",
        }, handle)
    return Kernel(name=LOCAL_KERNEL_NAME, search_dirs=(spec_root,),
                  description=f"this interpreter ({sys.executable})")


def kernel_env() -> Dict[str, str]:
    """The environment every render kernel gets.

    * ``MPLBACKEND=Agg`` -- headless. A render happens over SSH and in scheduled shells, where an
      interactive backend either fails at import or opens a window nothing closes.
    * ``PYTHONPATH`` gains ``eda/`` so ``import eda_analysis`` resolves regardless of which
      directory the kernel starts in. The kernel's cwd is the notebook's own folder, matching what
      Jupyter does interactively, and that folder is NOT the package's parent.
    * ``EDA_JUDGE`` is REMOVED. Exp3 passed the grader to its notebooks through this variable;
      Exp4's judge is a field on ``EdaConfig``, and a stray value in the ambient environment must
      not be able to change what a rendered artifact means.
    * ``EDA_NO_CACHE`` is left alone: the parquet memo keys on input content, so it cannot serve a
      stale frame, and a from-clean render of four families is where it earns its keep.
    """
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        [EDA_DIR] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("EDA_JUDGE", None)
    return env


# ==============================================================================
#  Execution
# ==============================================================================


def _tag(top: str) -> str:
    return f"top={top:<10}"


#: IPython colours its tracebacks, so an nbclient error message arrives full of SGR escapes. They
#: are invisible in a terminal and pure noise in a redirected log, which is where a failed render
#: is usually read.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _first_error_line(exc: BaseException) -> str:
    """One readable line out of an nbclient failure, whose ``str`` is a whole cell traceback."""
    text = _ANSI_RE.sub("", str(exc)).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail = lines[-1] if lines else text
    return f"{type(exc).__name__}: {tail[:220]}"


def run_unit(unit: Unit, *, workdir: str, kernel: Kernel, timeout: int) -> List[Outcome]:
    """Execute one top's notebooks sequentially in one shared kernel.

    Args:
        unit: The top and its families.
        workdir: Temp root. Each notebook is copied to ``<workdir>/<top>/<sub>.ipynb`` and the
            EXECUTED copy is written back there -- never over the repo's notebook, which must stay
            output-free.
        kernel: From :func:`resolve_kernel`.
        timeout: Seconds per notebook.

    Returns:
        One :class:`Outcome` per family, including a ``skipped`` one per missing notebook.

    Notes:
        The kernel's cwd is the notebook's own directory, which is what Jupyter does interactively
        -- so a path a notebook opens relative to itself behaves the same in both. Package imports
        do not depend on it (see :func:`kernel_env`).

        A notebook that fails does NOT abort the unit: the remaining notebooks still run, after a
        ``%reset -f``. Their results are independent, and a partial render with a named failure is
        more useful than an unexplained gap.
    """
    import nbformat
    from nbclient import NotebookClient

    outcomes = [Outcome(f, "skipped", 0.0, f"no notebook at {_rel(notebook_path(f))}")
                for f in unit.missing]
    if not unit.families:
        return outcomes

    nb_dir = os.path.join(NB_ROOT, unit.top)
    out_dir = os.path.join(workdir, unit.top)
    os.makedirs(out_dir, exist_ok=True)

    try:
        km = kernel.manager(cwd=nb_dir, env=kernel_env())
    except Exception as exc:                                # noqa: BLE001 -- report, do not raise
        detail = (f"kernel would not start ({type(exc).__name__}: {exc}); "
                  f"try --kernel <name> or check that ipykernel is installed")
        print(f"[render] {_tag(unit.top)} FAILED -- {detail}", flush=True)
        return outcomes + [Outcome(f, "failed", 0.0, detail) for f in unit.families]

    try:
        for family in unit.families:
            source = notebook_path(family)
            copy = os.path.join(out_dir, os.path.basename(source))
            shutil.copy2(source, copy)
            notebook = nbformat.read(copy, as_version=4)

            print(f"[render] {_tag(unit.top)} nb={family}", flush=True)
            started = time.time()
            try:
                NotebookClient(notebook, km=km, timeout=timeout, allow_errors=False,
                               resources={"metadata": {"path": nb_dir}}).execute()
                elapsed = time.time() - started
                outcomes.append(Outcome(family, "ok", elapsed))
                print(f"[render] {_tag(unit.top)} nb={family} OK {elapsed:6.1f}s", flush=True)
            except Exception as exc:                       # noqa: BLE001 -- one family, not the run
                elapsed = time.time() - started
                detail = _first_error_line(exc)
                outcomes.append(Outcome(family, "failed", elapsed, detail))
                print(f"[render] {_tag(unit.top)} nb={family} FAILED {elapsed:6.1f}s -- {detail}",
                      flush=True)
            finally:
                # The executed copy is the debugging artifact: it carries the traceback in the cell
                # that raised. Written even on success so a passing render can be inspected too.
                try:
                    nbformat.write(notebook, copy)
                except Exception:                           # noqa: BLE001 -- never mask the result
                    pass
                _reset_kernel(km, nb_dir)
    finally:
        try:
            km.shutdown_kernel(now=True)
        except Exception:                                   # noqa: BLE001
            pass
    return outcomes


def _reset_kernel(km, nb_dir: str) -> None:
    """Run ``%reset -f`` in the shared kernel between notebooks.

    Always run, including after a failure: otherwise a half-executed notebook's globals leak into
    the next one and turn one failure into a cascade whose first real cause is buried.
    """
    try:
        import nbformat
        from nbclient import NotebookClient

        reset = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(_RESET_SOURCE)])
        NotebookClient(reset, km=km, timeout=120, allow_errors=True,
                       resources={"metadata": {"path": nb_dir}}).execute()
    except Exception:                                       # noqa: BLE001 -- best effort
        pass


def _rel(path: str) -> str:
    try:
        out = os.path.relpath(path, EDA_DIR).replace(os.sep, "/")
    except ValueError:
        return path
    return path if out.startswith("..") else out


# ==============================================================================
#  Reporting
# ==============================================================================

_STATUS_MARK = {"ok": "OK", "failed": "FAILED", "skipped": "SKIPPED"}


def print_summary(outcomes: Sequence[Outcome], elapsed: float) -> None:
    """One row per family: status, wall-clock, and the reason for anything that is not ``ok``."""
    ordered = {f: None for f in all_families()}
    rows = sorted(outcomes, key=lambda o: list(ordered).index(o.family)
                  if o.family in ordered else len(ordered))
    width = max([len(o.family) for o in rows] + [len("family")])

    print("\n" + "=" * (width + 46))
    print(f"  {'family'.ljust(width)}  {'status':<8} {'seconds':>8}  detail")
    print("  " + "-" * (width + 44))
    for outcome in rows:
        seconds = f"{outcome.seconds:8.1f}" if outcome.status != "skipped" else f"{'-':>8}"
        print(f"  {outcome.family.ljust(width)}  {_STATUS_MARK[outcome.status]:<8} {seconds}  "
              f"{outcome.detail}")
    print("  " + "-" * (width + 44))

    counts = {s: sum(1 for o in rows if o.status == s) for s in _STATUS_MARK}
    print(f"  {counts['ok']} rendered, {counts['failed']} failed, {counts['skipped']} skipped "
          f"in {elapsed:.0f}s")
    if counts["ok"]:
        tops = sorted({o.family.split("/")[0] for o in rows if o.status == "ok"})
        print(f"  results: {[_rel(os.path.join(RESULTS_ROOT, t)) for t in tops]}")


def print_inventory(units: Sequence[Unit], kernel_desc: str, jobs: int) -> None:
    """The ``--list`` / ``--dry-run`` view: every family, its notebook, and the execution plan."""
    print("families (config.FAMILIES order):")
    for family in all_families():
        path = notebook_path(family)
        mark = "present" if os.path.isfile(path) else "MISSING notebook"
        print(f"  {family:<22} {mark:<17} {_rel(path)}")
    print(f"\nunits (one per top, {jobs} in parallel, notebooks sequential within a unit):")
    if not units:
        print("  (none -- no family notebook on disk for this selection)")
    for unit in units:
        print(f"  top={unit.top:<10} render={unit.families or '[]'}"
              + (f"  skip={unit.missing}" if unit.missing else ""))
    print(f"\nkernel : {kernel_desc}")
    print(f"results: {_rel(RESULTS_ROOT)}/<top>/<sub>/  (no <judge>/ level; EDA_JUDGE unset)")


# ==============================================================================
#  Entry point
# ==============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, plan the units, render them, and report. Returns 1 if anything failed."""
    parser = argparse.ArgumentParser(
        description="Regenerate eda/results/<top>/<sub>/ by executing the family notebooks "
                    "(one unit per top; tops in parallel, notebooks sequential within a top).")
    parser.add_argument("--top", nargs="*", default=None,
                        help=f"tops to render (subset of {list(FAMILIES)}); default = all")
    parser.add_argument("--family", nargs="*", default=None,
                        help="specific families '<top>/<sub>' (e.g. method/contrast); "
                             "default = every family of the selected tops")
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help=f"parallel units (default = #units capped at {MAX_PARALLEL_UNITS}); "
                             f"1 = sequential")
    parser.add_argument("--kernel", default=None,
                        help="registered kernelspec name; default = this interpreter")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"seconds per notebook (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--list", action="store_true",
                        help="print the family inventory and the unit plan, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve everything and print what would run, but execute nothing")
    args = parser.parse_args(argv)

    tops = list(FAMILIES) if not args.top else list(dict.fromkeys(args.top))
    unknown_tops = [t for t in tops if t not in FAMILIES]
    if unknown_tops:
        parser.error(f"unknown top(s) {unknown_tops}; choose from {list(FAMILIES)}")

    families = all_families()
    if args.family:
        unknown = [f for f in args.family if f not in families]
        if unknown:
            parser.error(f"unknown family(ies) {unknown}; choose from {families}")
        wanted = set(args.family)
        families = [f for f in families if f in wanted]        # keep FAMILIES order
        tops = [t for t in tops if any(f.split("/")[0] == t for f in families)]

    units = plan_units(tops, families)
    n_notebooks = sum(len(u.families) for u in units)
    jobs = args.jobs if args.jobs is not None else min(max(len(units), 1), MAX_PARALLEL_UNITS)
    jobs = max(1, min(jobs, max(len(units), 1)))

    if args.list:
        print_inventory(units, "this interpreter" if not args.kernel
                        else f"kernelspec {args.kernel!r}", jobs)
        return 0

    workdir = tempfile.mkdtemp(prefix="exp4_render_")
    kernel = resolve_kernel(args.kernel, workdir)

    if args.dry_run:
        print_inventory(units, kernel.description, jobs)
        print(f"\n[render] dry run -- nothing executed. A real run copies each notebook into a "
              f"fresh directory under {tempfile.gettempdir()} and executes the copy; only "
              f"results/ is written back.")
        shutil.rmtree(workdir, ignore_errors=True)
        return 0

    for unit in units:
        if unit.missing:
            print(f"[render] {_tag(unit.top)} skipping {len(unit.missing)} family/families with "
                  f"no notebook: {unit.missing}", flush=True)
    if not n_notebooks:
        print("[render] nothing to render (no family notebook on disk for this selection).")
        shutil.rmtree(workdir, ignore_errors=True)
        return 0

    print(f"[render] {len(units)} unit(s) / {n_notebooks} notebook(s), {jobs} in parallel, "
          f"kernel = {kernel.description}", flush=True)
    started = time.time()

    def render(unit: Unit) -> List[Outcome]:
        return run_unit(unit, workdir=workdir, kernel=kernel, timeout=args.timeout)

    outcomes: List[Outcome] = []
    if jobs == 1:
        for unit in units:
            outcomes += render(unit)
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(render, units):
                outcomes += result

    print_summary(outcomes, time.time() - started)
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        # Keep the executed copies: they carry the traceback in the cell that raised, which is the
        # only place the real error lives once the kernel is gone.
        print(f"  executed notebooks kept for debugging: {workdir}")
        return 1
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
