"""
exports.py — save publication figures + result tables for the thesis (one format each).

One artifact, one file — no duplicate formats cluttering ``results/``:
- figures → ``results/figures/<name>.pdf`` (vector, for LaTeX/Overleaf)
- tables  → ``results/tables/<name>.md`` (paste-able / readable)

The ``formats=`` kwarg still lets a one-off call request extra formats explicitly (e.g.
``save_fig(fig, name, formats=("pdf", "png"))``), but the defaults are a single format apiece.

Notebooks keep showing plots inline AND call :func:`save_fig` / :func:`save_table` on their key
artifacts with stable, thesis-ready names, so re-running a notebook regenerates its deliverables.
Captions accumulate in ``results/figures/CAPTIONS.md`` / ``results/tables/CAPTIONS.md``.
"""

import os
from typing import Optional, Sequence

import pandas as pd

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


def _append_caption(dir_path: str, name: str, caption: Optional[str]):
    """Record (or refresh) the caption line for *name* in CAPTIONS.md — idempotent.

    Re-running a notebook overwrites the existing line for that artifact instead of appending
    a duplicate, so CAPTIONS.md stays one-line-per-artifact across reruns.
    """
    if not caption:
        return
    path = os.path.join(dir_path, "CAPTIONS.md")
    line = f"- **{name}** — {caption}\n"
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if not l.startswith(f"- **{name}** —")]
    lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def save_fig(fig, name: str, *, formats: Sequence[str] = ("pdf",),
             dpi: int = 200, caption: Optional[str] = None) -> str:
    """Save *fig* to ``results/figures/<name>.<fmt>`` for each format; log the caption.

    Defaults to vector ``pdf`` only (LaTeX/Overleaf). Pass ``formats=("pdf","png")`` for a
    one-off extra. Returns the figures dir. Call right before/after ``plt.show()`` — the
    inline display is unaffected.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{fmt}"), dpi=dpi, bbox_inches="tight")
    _append_caption(FIGURES_DIR, name, caption)
    return FIGURES_DIR


def save_table(df: pd.DataFrame, name: str, *, formats: Sequence[str] = ("md",),
               float_format: str = "%.3f", index: bool = False,
               caption: Optional[str] = None) -> str:
    """Save *df* to ``results/tables/<name>.<fmt>``; log the caption. Returns the tables dir.

    Defaults to ``.md`` only (paste-able / readable); pass ``formats=("md","csv","tex")`` for a
    one-off extra. ``.tex`` is booktabs-ready (``\\usepackage{booktabs}``); ``.md`` falls back to a
    manual writer if ``tabulate`` isn't installed.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    base = os.path.join(TABLES_DIR, name)
    if "csv" in formats:
        df.to_csv(f"{base}.csv", index=index)
    if "tex" in formats:
        try:
            tex = df.to_latex(index=index, float_format=lambda x: (float_format % x),
                              escape=True, bold_rows=False)
        except Exception:
            tex = df.to_latex(index=index)
        with open(f"{base}.tex", "w", encoding="utf-8") as f:
            f.write(tex)
    if "md" in formats:
        with open(f"{base}.md", "w", encoding="utf-8") as f:
            f.write(_to_markdown(df, index=index, float_format=float_format))
    _append_caption(TABLES_DIR, name, caption)
    return TABLES_DIR


def _to_markdown(df: pd.DataFrame, *, index: bool, float_format: str) -> str:
    """Markdown table via pandas/tabulate, with a dependency-free fallback."""
    try:
        return df.to_markdown(index=index, floatfmt=float_format.replace("%", "").replace("f", "f"))
    except Exception:
        d = df.copy()
        if index:
            d = d.reset_index()
        def fmt(v):
            try:
                return float_format % float(v)
            except (TypeError, ValueError):
                return str(v)
        cols = list(d.columns)
        head = "| " + " | ".join(map(str, cols)) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
        return "\n".join([head, sep, *rows]) + "\n"
