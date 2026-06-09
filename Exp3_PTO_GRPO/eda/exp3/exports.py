"""
exports.py — save publication figures + result tables for the thesis (portable formats).

The thesis platform is undecided, so everything is written in formats that work for either
LaTeX/Overleaf or Word:
- figures → ``results/figures/<name>.pdf`` (vector, for LaTeX) **and** ``.png`` (200 dpi, for Word/preview)
- tables  → ``results/tables/<name>.csv`` + ``.tex`` (booktabs-ready) + ``.md`` (paste-able)

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
    if not caption:
        return
    with open(os.path.join(dir_path, "CAPTIONS.md"), "a", encoding="utf-8") as f:
        f.write(f"- **{name}** — {caption}\n")


def save_fig(fig, name: str, *, formats: Sequence[str] = ("pdf", "png"),
             dpi: int = 200, caption: Optional[str] = None) -> str:
    """Save *fig* to ``results/figures/<name>.<fmt>`` for each format; log the caption.

    Vector ``pdf`` for LaTeX, 200-dpi ``png`` for Word/preview. Returns the figures dir.
    Call right before/after ``plt.show()`` — the inline display is unaffected.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{fmt}"), dpi=dpi, bbox_inches="tight")
    _append_caption(FIGURES_DIR, name, caption)
    return FIGURES_DIR


def save_table(df: pd.DataFrame, name: str, *, formats: Sequence[str] = ("csv", "tex", "md"),
               float_format: str = "%.3f", index: bool = False,
               caption: Optional[str] = None) -> str:
    """Save *df* to ``results/tables/<name>.{csv,tex,md}``; log the caption. Returns the tables dir.

    ``.tex`` is booktabs-ready (``\\usepackage{booktabs}``); ``.md`` falls back to a manual writer if
    ``tabulate`` isn't installed.
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
