"""
exports.py — save publication figures + result tables for the thesis (one format each).

One artifact, one file — no duplicate formats cluttering ``results/``:
- figures → ``results/figures/<group>/<name>.pdf`` (vector, for LaTeX/Overleaf)
- tables  → ``results/tables/<group>/<name>.md`` (paste-able / readable)

``<group>`` is the per-notebook export group (``"eval"``, ``"behavior"``, …) set once via
:func:`set_export_group` (``notebook_setup`` does this from ``EdaConfig.export_group``). With no
group set, artifacts fall back to the flat ``results/figures/`` / ``results/tables/`` roots.

The ``formats=`` kwarg still lets a one-off call request extra formats explicitly (e.g.
``save_fig(fig, name, formats=("pdf", "png"))``), but the defaults are a single format apiece.

Notebooks keep showing plots inline AND call :func:`save_fig` / :func:`save_table` on their key
artifacts with stable, thesis-ready names, so re-running a notebook regenerates its deliverables.
Captions accumulate in each group's ``CAPTIONS.md``; :func:`build_index` writes a top-level
``results/INDEX.md`` listing every artifact across groups.
"""

import os
import re
import shutil
from typing import Optional, Sequence

import pandas as pd

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# The active export group (per-notebook subfolder). Empty = flat roots (legacy behaviour).
_GROUP = ""
# Default formats used when a save_* call doesn't pass `formats=` explicitly. Set by
# notebook_setup() from EdaConfig (figures -> PNG images, tables -> readable .md + Excel .xlsx).
_FIG_FORMATS = ("png",)
_TABLE_FORMATS = ("md", "xlsx")


def set_export_group(group: str = "") -> None:
    """Set the per-notebook export subfolder for subsequent ``save_fig``/``save_table`` calls.

    ``notebook_setup`` calls this from ``EdaConfig.export_group``. Pass ``""`` for the flat roots.
    """
    global _GROUP
    _GROUP = (group or "").strip().strip("/\\")


def set_formats(fig_formats=None, table_formats=None) -> None:
    """Set the default save formats (``notebook_setup`` calls this from ``EdaConfig``)."""
    global _FIG_FORMATS, _TABLE_FORMATS
    if fig_formats:
        _FIG_FORMATS = tuple(fig_formats)
    if table_formats:
        _TABLE_FORMATS = tuple(table_formats)


def _fig_dir() -> str:
    return os.path.join(FIGURES_DIR, _GROUP) if _GROUP else FIGURES_DIR


def _tab_dir() -> str:
    return os.path.join(TABLES_DIR, _GROUP) if _GROUP else TABLES_DIR


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


def save_fig(fig, name: str, *, formats: Optional[Sequence[str]] = None,
             dpi: int = 200, caption: Optional[str] = None) -> str:
    """Save *fig* to ``results/figures/<group>/<name>.<fmt>`` for each format; log the caption.

    ``formats=None`` uses the notebook default (``EdaConfig.fig_formats`` → PNG images by default;
    set ``cfg.fig_formats=("png","pdf")`` to also emit vector PDF). Returns the (group) figures dir.
    Call right before/after ``plt.show()`` — the inline display is unaffected.
    """
    d = _fig_dir()
    os.makedirs(d, exist_ok=True)
    for fmt in (formats or _FIG_FORMATS):
        fig.savefig(os.path.join(d, f"{name}.{fmt}"), dpi=dpi, bbox_inches="tight")
    _append_caption(d, name, caption)
    return d


def save_table(df: pd.DataFrame, name: str, *, formats: Optional[Sequence[str]] = None,
               float_format: str = "%.3f", index: bool = False,
               caption: Optional[str] = None) -> str:
    """Save *df* to ``results/tables/<group>/<name>.<fmt>``; log the caption. Returns the tables dir.

    ``formats=None`` uses the notebook default (``EdaConfig.table_formats`` → ``.md`` + ``.xlsx``).
    ``.xlsx`` collects every table of the group into one workbook ``<group>.xlsx`` (one sheet per
    table — sortable/filterable in Excel). ``.md`` is paste-able/readable; ``.csv``/``.tex`` available
    on request. ``.md`` falls back to a manual writer if ``tabulate`` isn't installed.
    """
    formats = formats or _TABLE_FORMATS
    d = _tab_dir()
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, name)
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
    if "xlsx" in formats:
        _write_xlsx_sheet(d, name, df, index=index)
    _append_caption(d, name, caption)
    return d


def _write_xlsx_sheet(dir_path: str, name: str, df: pd.DataFrame, *, index: bool = False) -> None:
    """Write/replace ``df`` as a sheet in the group workbook ``<group_or_tables>.xlsx``.

    One workbook per tables subfolder, one sheet per table name (Excel caps sheet names at 31 chars).
    Re-running a notebook overwrites that sheet (idempotent). Requires ``openpyxl``.
    """
    group = os.path.basename(dir_path.rstrip("/\\")) or "tables"
    xpath = os.path.join(dir_path, f"{group}.xlsx")
    sheet = re.sub(r"[\[\]:*?/\\]", "_", name)[:31]
    try:
        if os.path.exists(xpath):
            with pd.ExcelWriter(xpath, engine="openpyxl", mode="a",
                                if_sheet_exists="replace") as xw:
                df.to_excel(xw, sheet_name=sheet, index=index)
        else:
            with pd.ExcelWriter(xpath, engine="openpyxl", mode="w") as xw:
                df.to_excel(xw, sheet_name=sheet, index=index)
    except Exception as e:   # missing engine / locked file — don't break the .md export
        print(f"  [exports] xlsx skipped for {name}: {e}")


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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   PROVENANCE · INDEX · RESET (organization)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def save_provenance(cfg, scores=None, *, group: Optional[str] = None) -> str:
    """Write a per-run provenance banner to ``results/<figures>/<group>/_provenance.md``.

    Records the active ``EdaConfig`` + the arms/metrics actually present in ``scores`` so every
    regenerated figure set is traceable to the config that produced it. Returns the file path.
    """
    g = (group if group is not None else _GROUP) or ""
    d = os.path.join(FIGURES_DIR, g) if g else FIGURES_DIR
    os.makedirs(d, exist_ok=True)
    cfgd = cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg)
    lines = [f"# Provenance — group `{g or '(flat)'}`\n"]
    if scores is not None and not getattr(scores, "empty", True):
        arms = sorted(scores["arm"].unique()) if "arm" in scores.columns else []
        mets = sorted(scores["questionnaire"].unique()) if "questionnaire" in scores.columns else []
        lines.append(f"- **arms scored:** {arms}")
        lines.append(f"- **metrics present:** {mets}")
        lines.append(f"- **rows:** {len(scores)}")
    lines.append("\n## EdaConfig")
    for k, v in cfgd.items():
        lines.append(f"- `{k}` = {v}")
    path = os.path.join(d, "_provenance.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build_index() -> str:
    """Write ``results/INDEX.md`` listing every figure + table across all group subfolders.

    A master artifact map so the reader sees, in one place, which notebook (group) produced what.
    Returns the index path.
    """
    lines = ["# Exp3 EDA artifact index\n",
             "_Generated by `eda_analysis.build_index()` — figures (`.pdf`) + tables (`.md`) by group._\n"]
    for kind, root in (("Figures", FIGURES_DIR), ("Tables", TABLES_DIR)):
        lines.append(f"\n## {kind}")
        if not os.path.isdir(root):
            lines.append("_(none)_")
            continue
        # group subfolders first, then any flat artifacts at the root
        groups = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        for g in groups + [""]:
            gdir = os.path.join(root, g) if g else root
            if not os.path.isdir(gdir):
                continue
            arts = sorted(f for f in os.listdir(gdir)
                          if f.lower().endswith((".pdf", ".md")) and not f.startswith(("CAPTIONS", "_prov")))
            if not arts:
                continue
            lines.append(f"\n### {g or '(flat)'}")
            lines += [f"- `{a}`" for a in arts]
    path = os.path.join(RESULTS_DIR, "INDEX.md")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def reset_results(groups: Optional[Sequence[str]] = None, *, flat: bool = False) -> None:
    """Clear generated artifacts before a clean regenerate.

    - ``groups`` given → remove just those ``results/{figures,tables}/<group>/`` subfolders.
    - ``groups=None`` → remove ALL group subfolders under both roots.
    - ``flat=True`` → also delete loose ``*.pdf`` / ``*.md`` sitting at the flat roots (the
      legacy dump). Subfolders are recreated lazily on the next save.
    """
    for root, ext in ((FIGURES_DIR, ".pdf"), (TABLES_DIR, ".md")):
        if not os.path.isdir(root):
            continue
        subs = ([os.path.join(root, g) for g in groups] if groups is not None
                else [os.path.join(root, d) for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))])
        for s in subs:
            if os.path.isdir(s):
                shutil.rmtree(s)
        if flat:
            for f in os.listdir(root):
                fp = os.path.join(root, f)
                if os.path.isfile(fp) and (f.lower().endswith(ext) or f == "CAPTIONS.md"):
                    os.remove(fp)
