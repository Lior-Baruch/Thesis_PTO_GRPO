"""Build overleaf.zip: exactly the files an Overleaf project needs to compile this paper.

    & ..\\..\\.venv\\Scripts\\python.exe make_overleaf_zip.py

Contents: main.tex, sections/*.tex, figures/*.png, refs.bib, acl.sty, acl_natbib.bst — nothing
else (no ledger, no README, no build output, no sync scripts). Upload the zip as a new Overleaf
project, compiler pdfLaTeX, main document main.tex. The zip is gitignored; rebuild it after any
edit you want the Overleaf copy to see.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "overleaf.zip"

FILES = [
    HERE / "main.tex",
    HERE / "refs.bib",
    HERE / "acl.sty",
    HERE / "acl_natbib.bst",
    *sorted((HERE / "sections").glob("*.tex")),
    *sorted((HERE / "figures").glob("*.png")),
]


def main() -> int:
    missing = [p for p in FILES if not p.exists()]
    for m in missing:
        print("MISSING:", m)
    if missing:
        return 1
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in FILES:
            z.write(p, p.relative_to(HERE).as_posix())
    print(f"wrote {OUT.name}: {len(FILES)} files, {OUT.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
