#!/usr/bin/env python
"""
strip_notebook_outputs.py — keep committed notebooks output-clean (zero external deps).

The ``results/<view>/`` tree is the deliverable; the notebooks are throwaway drivers
(``render_views.py`` executes them into a *tmp* ``--output-dir``, never touching the committed
copies). So committed notebooks should carry **no** cell outputs — otherwise rendered PNGs bloat
git (nb 4 + nb 5 alone were ~2 MB before this).

Three modes:

* **in-place** (default) — strip every ``eda/*.ipynb`` (or the paths given) and rewrite::

      python strip_notebook_outputs.py               # all eda/*.ipynb
      python strip_notebook_outputs.py 4_Training_and_Reliability.ipynb

* **--check** — exit non-zero (and list offenders) if any tracked notebook still has outputs.
  Cheap regression guard; wire into CI / the self-check / a pre-push hook::

      python strip_notebook_outputs.py --check

* **--filter** — act as a git *clean* filter: read one notebook on stdin, write the stripped
  notebook to stdout. Install once (local, solo-repo) so ``git add`` strips automatically while
  the working tree keeps outputs for interactive viewing::

      git config filter.nbstrip.clean "<venv-python> Exp3_PTO_GRPO/eda/strip_notebook_outputs.py --filter"
      # .gitattributes already routes Exp3_PTO_GRPO/eda/*.ipynb through filter=nbstrip

  (A missing filter config is a harmless no-op on another machine — re-run the one-liner.)

Stripping clears every code cell's ``outputs`` and ``execution_count`` and drops per-cell
execution metadata; markdown cells, sources, and notebook-level metadata are left untouched.
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob
from typing import List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
# Per-cell metadata keys that only record a specific execution — safe to drop.
_EXEC_META_KEYS = ("execution", "collapsed", "scrolled")


def strip_nb(nb: dict) -> Tuple[dict, bool]:
    """Return ``(nb, changed)`` with all code-cell outputs/exec-counts cleared."""
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        meta = cell.get("metadata", {})
        for k in _EXEC_META_KEYS:
            if k in meta:
                del meta[k]
                changed = True
    return nb, changed


def _load(path_or_text: str, *, is_text: bool) -> dict:
    return json.loads(path_or_text) if is_text else json.load(open(path_or_text, encoding="utf-8"))


def _dump(nb: dict) -> str:
    # Match Jupyter's on-disk style: 1-space indent, trailing newline, non-ASCII preserved.
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def _targets(args: List[str]) -> List[str]:
    paths = [a for a in args if not a.startswith("-")]
    if paths:
        return [p if os.path.isabs(p) else os.path.join(_HERE, p) for p in paths]
    return sorted(glob(os.path.join(_HERE, "*.ipynb")))


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    if "--filter" in argv:                       # git clean filter: stdin -> stdout (LF, exact bytes)
        nb, _ = strip_nb(_load(sys.stdin.buffer.read().decode("utf-8"), is_text=True))
        sys.stdout.buffer.write(_dump(nb).encode("utf-8"))
        return 0

    check = "--check" in argv
    targets = _targets(argv)
    dirty: List[str] = []
    for path in targets:
        nb, changed = strip_nb(_load(path, is_text=False))
        rel = os.path.relpath(path, _HERE)
        if not changed:
            continue
        if check:
            dirty.append(rel)
        else:
            open(path, "w", encoding="utf-8", newline="\n").write(_dump(nb))
            print(f"  stripped {rel}")

    if check:
        if dirty:
            print("NOT output-clean (run strip_notebook_outputs.py): " + ", ".join(dirty))
            return 1
        print(f"all {len(targets)} notebook(s) output-clean")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
