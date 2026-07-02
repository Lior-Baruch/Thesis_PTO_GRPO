"""
exports.py — save publication figures + result tables for the thesis, organized by VIEW + group.

Two-level layout::

    results/<view>/figures/<group>/<name>.png      # view = all | L0 | L5
    results/<view>/tables/<group>/<name>.md (+ .xlsx)
    results/<view>/INDEX.md                        # per-view artifact map
    results/<view>/SUMMARY.md                      # HAND-AUTHORED narrative (never auto-deleted)

- ``<view>`` is set once per notebook via :func:`set_view` (``notebook_setup`` does this from
  ``EdaConfig.view``). It splits the artifacts into the parallel look-ahead trees the user
  asked for. With no view set (``""``) artifacts fall back to the legacy bare ``results/`` root.
- ``<group>`` is the notebook's topic family (``"1_outcomes"``, ``"2_heterogeneity"``,
  ``"3_mechanism"``, ``"4_training"``, ``"5_preference"``, ``"6_stats"``) set via
  :func:`set_export_group` — the family NUMBER matches the producing notebook's number, so any
  artifact traces straight back to its notebook. A per-call ``group=`` on ``save_fig``/``save_table``
  overrides it for one save and may be a NESTED subpath within the family
  (``"1_outcomes/trajectories"``, ``"2_heterogeneity/problem"``). With no group set, artifacts fall
  back to the view's flat roots.

The ``formats=`` kwarg lets a one-off call request extra formats (e.g.
``save_fig(fig, name, formats=("pdf", "png"))``); the defaults are PNG figures + ``.md``/``.xlsx`` tables.

Notebooks keep showing plots inline AND call :func:`save_fig` / :func:`save_table` on their key
artifacts with stable, thesis-ready names. Captions accumulate in each group's ``CAPTIONS.md``;
:func:`build_index` writes the per-view ``results/<view>/INDEX.md``. :func:`reset_results` clears the
generated figure/table subfolders of the active view but PRESERVES the hand-authored ``SUMMARY.md``.
"""

import os
import re
import shutil
from typing import Optional, Sequence

import pandas as pd

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")   # legacy bare roots (when no view is set)
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Files at a view root that must NEVER be deleted by reset_results (hand-authored, not regenerated).
PRESERVE = {"SUMMARY.md"}

# Figure/table file extensions recognized by build_index / reset_results, per root.
_FIG_EXTS = (".png", ".pdf", ".svg")
_TAB_EXTS = (".md",)

# The active view subfolder. Empty = legacy bare roots (results/figures/...).
_VIEW = ""
# The active export group (per-notebook subfolder). Empty = the view's flat roots.
_GROUP = ""
# Default formats used when a save_* call doesn't pass `formats=` explicitly. Set by
# notebook_setup() from EdaConfig (figures -> PNG images, tables -> readable .md + Excel .xlsx).
_FIG_FORMATS = ("png",)
_TABLE_FORMATS = ("md", "xlsx")


def set_view(view: str = "") -> None:
    """Set the active VIEW subfolder for subsequent saves (``results/<view>/...``).

    ``notebook_setup`` calls this from ``EdaConfig.view`` (``all``/``L0``/``L5``). Pass ``""`` for
    the legacy bare ``results/`` root.
    """
    global _VIEW
    _VIEW = (view or "").strip().strip("/\\")


def _norm_group(group) -> str:
    """Normalize a group (sub)path: trim whitespace + leading/trailing slashes; interior kept."""
    return (group or "").strip().strip("/\\")


def set_export_group(group: str = "") -> None:
    """Set the per-notebook export subfolder for subsequent ``save_fig``/``save_table`` calls.

    ``notebook_setup`` calls this from ``EdaConfig.export_group``. Pass ``""`` for the flat roots.
    """
    global _GROUP
    _GROUP = _norm_group(group)


def set_formats(fig_formats=None, table_formats=None) -> None:
    """Set the default save formats (``notebook_setup`` calls this from ``EdaConfig``)."""
    global _FIG_FORMATS, _TABLE_FORMATS
    if fig_formats:
        _FIG_FORMATS = tuple(fig_formats)
    if table_formats:
        _TABLE_FORMATS = tuple(table_formats)


# ── View-aware path helpers (everything downstream routes through these) ───────
def _results_root() -> str:
    return os.path.join(RESULTS_DIR, _VIEW) if _VIEW else RESULTS_DIR


def _figures_root() -> str:
    return os.path.join(_results_root(), "figures")


def _tables_root() -> str:
    return os.path.join(_results_root(), "tables")


def _fig_dir(group: Optional[str] = None) -> str:
    """Figures dir for *group* (per-call override, may be nested) or the module default."""
    g = _norm_group(group) if group is not None else _GROUP
    return os.path.join(_figures_root(), g) if g else _figures_root()


def _tab_dir(group: Optional[str] = None) -> str:
    """Tables dir for *group* (per-call override, may be nested) or the module default."""
    g = _norm_group(group) if group is not None else _GROUP
    return os.path.join(_tables_root(), g) if g else _tables_root()


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


def save_fig(fig, name: str, *, group: Optional[str] = None,
             formats: Optional[Sequence[str]] = None,
             dpi: int = 200, caption: Optional[str] = None) -> str:
    """Save *fig* to ``results/<view>/figures/<group>/<name>.<fmt>`` for each format; log the caption.

    ``group=None`` uses the notebook's family (``set_export_group``); pass a value to override for
    this one save — including NESTED subpaths within the family (``group="1_outcomes/trajectories"``).
    ``formats=None`` uses the notebook default (``EdaConfig.fig_formats`` → PNG images by default;
    set ``cfg.fig_formats=("png","pdf")`` to also emit vector PDF). Returns the (group) figures dir.
    Call right before/after ``plt.show()`` — the inline display is unaffected.
    """
    d = _fig_dir(group)
    os.makedirs(d, exist_ok=True)
    for fmt in (formats or _FIG_FORMATS):
        fig.savefig(os.path.join(d, f"{name}.{fmt}"), dpi=dpi, bbox_inches="tight")
    _append_caption(d, name, caption)
    return d


def save_table(df: pd.DataFrame, name: str, *, group: Optional[str] = None,
               formats: Optional[Sequence[str]] = None,
               float_format: str = "%.3f", index: bool = False,
               caption: Optional[str] = None) -> str:
    """Save *df* to ``results/<view>/tables/<group>/<name>.<fmt>``; log the caption. Returns the dir.

    ``group=None`` uses the notebook's family (``set_export_group``); pass a value to override for
    this one save (nested subpaths supported). ``formats=None`` uses the notebook default
    (``EdaConfig.table_formats`` → ``.md`` + ``.xlsx``). ``.xlsx`` collects every table of the group
    into one workbook ``<group>.xlsx`` (one sheet per table — sortable/filterable in Excel). ``.md``
    is paste-able/readable; ``.csv``/``.tex`` available on request. ``.md`` falls back to a manual
    writer if ``tabulate`` isn't installed.
    """
    formats = formats or _TABLE_FORMATS
    d = _tab_dir(group)
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
    """Write a per-run provenance banner to ``results/<view>/figures/<group>/_provenance.md``.

    Records the active ``EdaConfig`` (incl. the view) + the arms/metrics actually present in
    ``scores`` so every regenerated figure set is traceable to the config that produced it.
    Returns the file path.
    """
    g = (group if group is not None else _GROUP) or ""
    d = os.path.join(_figures_root(), g) if g else _figures_root()
    os.makedirs(d, exist_ok=True)
    cfgd = cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg)
    lines = [f"# Provenance — view `{_VIEW or '(flat)'}` · group `{g or '(flat)'}`\n"]
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
    """Write ``results/<view>/INDEX.md`` listing every figure + table of the active view.

    A per-view artifact map so the reader sees, in one place, which notebook (group) produced what.
    Returns the index path. (The hand-authored ``SUMMARY.md`` is the narrative companion to this map.)
    """
    view = _VIEW or "(flat)"
    lines = [f"# Exp3 EDA artifact index — view `{view}`\n",
             "_Generated by `eda_analysis.build_index()`. See `SUMMARY.md` for the written analysis._\n",
             "_Family number = producing notebook number (e.g. `1_outcomes` ← `1_Outcomes.ipynb`)._\n"]
    for kind, root, exts in (("Figures", _figures_root(), _FIG_EXTS),
                             ("Tables", _tables_root(), _TAB_EXTS)):
        lines.append(f"\n## {kind}")
        if not os.path.isdir(root):
            lines.append("_(none)_")
            continue
        # Recursive walk so NESTED family subfolders (1_outcomes/trajectories,
        # 2_heterogeneity/<trait>, …) are listed too; dirnames sorted for numeric-ish family order.
        any_listed = False
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            rel = os.path.relpath(dirpath, root)
            g = "" if rel == "." else rel.replace(os.sep, "/")
            arts = sorted(f for f in filenames
                          if f.lower().endswith(exts) and not f.startswith(("CAPTIONS", "_prov")))
            if not arts:
                continue
            any_listed = True
            lines.append(f"\n### {g or '(flat)'}")
            lines += [f"- `{a}`" for a in arts]
        if not any_listed:
            lines.append("_(none)_")
    out_dir = _results_root()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "INDEX.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def reset_results(groups: Optional[Sequence[str]] = None, *, flat: bool = False) -> None:
    """Clear generated artifacts of the ACTIVE VIEW before a clean regenerate.

    Operates only on ``results/<view>/{figures,tables}/`` — never the view root, so the
    hand-authored ``SUMMARY.md`` (and anything else in :data:`PRESERVE`) is always kept.

    - ``groups`` given (e.g. ``["1_outcomes", "3_mechanism"]``) → remove just those
      ``figures/<group>/`` + ``tables/<group>/`` subfolders (nested content included).
    - ``groups=None`` → remove ALL group subfolders under both roots.
    - ``flat=True`` → also delete loose figure/table files sitting at the (view's) flat roots.
      Subfolders are recreated lazily on the next save.
    """
    for root, exts in ((_figures_root(), _FIG_EXTS), (_tables_root(), _TAB_EXTS)):
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
                if f in PRESERVE:
                    continue
                fp = os.path.join(root, f)
                if os.path.isfile(fp) and (f.lower().endswith(exts) or f == "CAPTIONS.md"):
                    os.remove(fp)
