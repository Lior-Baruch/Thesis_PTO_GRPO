"""Build main.pdf and keep running pdflatex until the layout has CONVERGED, then check it.

    & ..\\..\\.venv\\Scripts\\python.exe build.py            # build + check
    & ..\\..\\.venv\\Scripts\\python.exe build.py --check    # check the existing main.pdf only

Why a loop and not the usual four steps. acl.sty's [review] mode loads lineno with the ``switch``
option, which puts lineno in *pagewise* mode: which column (and page) each line belongs to is
read back from the PREVIOUS pass's .aux, and the margin side and offset of every line number are
computed from that. So the numbers are right only once two consecutive passes produce the same
layout. ``pdflatex, bibtex, pdflatex, pdflatex`` stops one pass short whenever bibtex moved the
back matter: on 2026-09-16 that left 98 line numbers printed ON the text across seven pages
(cold 477 -> 6 -> 114 after bibtex -> 98 -> 0 on the fifth pass). This script runs pdflatex until
main.aux and main.out stop changing (at most MAX_PASSES), then scans the PDF for any line number
that sits inside a text column rather than in a margin, and for unresolved references.

Requires MiKTeX's pdflatex/bibtex on PATH; the scan needs PyMuPDF (``fitz``) from the repo venv
and is skipped with a warning if it is missing. Exit status is non-zero on LaTeX errors, on
non-convergence, on misplaced line numbers, or on unresolved references.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = "main"
MAX_PASSES = 8

# The line-number font acl.sty sets (\font\aclhv = phvb at 8pt, \color{lightgray}); a number is
# misplaced when its left edge is inside the text area instead of in a margin.
LN_FONT, LN_SIZE, LN_COLOR = "NimbusSanL-Bold", 7.97, 0xBFBFBF
MARGIN_LEFT, MARGIN_RIGHT = 0.11, 0.89          # fractions of the page width


def _run(*cmd: str) -> int:
    return subprocess.run(cmd, cwd=HERE, capture_output=True).returncode


def _digest(*names: str) -> str:
    h = hashlib.md5()
    for n in names:
        p = HERE / n
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def build() -> int:
    """pdflatex, bibtex, then pdflatex until .aux/.out are stable. Returns the pass count."""
    _run("pdflatex", "-interaction=nonstopmode", JOB)
    _run("bibtex", JOB)
    passes = 1
    before = _digest(f"{JOB}.aux", f"{JOB}.out")
    for _ in range(MAX_PASSES):
        _run("pdflatex", "-interaction=nonstopmode", JOB)
        passes += 1
        after = _digest(f"{JOB}.aux", f"{JOB}.out")
        if after == before:
            return passes
        before = after
    raise RuntimeError(f"layout did not converge in {MAX_PASSES} passes after bibtex")


def latex_errors() -> list[str]:
    log = (HERE / f"{JOB}.log").read_text(encoding="utf-8", errors="replace")
    return [l for l in log.splitlines() if l.startswith("!")]


def check_pdf() -> list[str]:
    """Problems found in main.pdf: misplaced line numbers per page, unresolved references."""
    problems: list[str] = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ["PyMuPDF not installed: line-number placement NOT checked"]
    doc = fitz.open(HERE / f"{JOB}.pdf")
    text = ""
    for pno, page in enumerate(doc, 1):
        width = page.rect.width
        bad = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if (LN_FONT in span["font"] and abs(span["size"] - LN_SIZE) < 0.2
                            and span["color"] == LN_COLOR and re.fullmatch(r"\d{3}", span["text"].strip())):
                        x = span["bbox"][0] / width
                        if MARGIN_LEFT < x < MARGIN_RIGHT:
                            bad.append(int(span["text"]))
        if bad:
            problems.append(f"page {pno}: {len(bad)} line numbers printed on the text ({bad[0]}..{bad[-1]})")
        text += page.get_text()
    if "??" in text:
        problems.append(f"{text.count('??')} unresolved reference(s)")
    problems.append(f"info: {len(doc)} pages")
    return problems


def main(argv: list[str]) -> int:
    if "--check" not in argv:
        passes = build()
        errs = latex_errors()
        print(f"built {JOB}.pdf in {passes} pdflatex passes (+ bibtex); LaTeX errors: {len(errs)}")
        for e in errs:
            print("  ", e)
        if errs:
            return 1
    problems = check_pdf()
    bad = [p for p in problems if not p.startswith("info:")]
    for p in problems:
        print("  ", p)
    print("CHECK", "FAILED" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
