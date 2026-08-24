"""exports.py -- the one door every artifact goes through on its way to disk.

Analysis notebooks produce three kinds of output that outlive the kernel: figures, tables and
number ledgers. If each notebook wrote them itself, four things would go wrong immediately and
silently, and all four did in Exp3 before this module existed:

1. **Paths drift.** Two notebooks spell the same leaf differently, and a table that was supposed to
   replace last render's table lands beside it instead. Nobody notices until a paper cites the
   stale one.
2. **Artifacts lose their captions.** A ``.png`` with no sentence saying what it plots is
   unreviewable six weeks later, and the sentence is only ever written at the moment of saving.
3. **Renders churn git.** An unchanged table rewritten with a fresh clock stamp shows up as a
   modified file, so a diff of 30 workbooks reads as "the numbers moved" when only the clock did --
   and the one workbook that DID change is invisible in the noise.
4. **A regenerate deletes something hand-written.** ``reset_results`` is a recursive delete living
   in the same folder as the hand-authored ``SUMMARY.md``.

So: **one module owns path composition, caption bookkeeping, byte determinism and the delete.**

Layout::

    results/<top>/<sub>/figures/[<group>/]<name>.png     figures (PNG; PDF/SVG opt-in)
    results/<top>/<sub>/tables/[<group>/]<name>.md       tables  (+ ONE .xlsx workbook per leaf)
    results/<top>/<sub>/tables/[<group>/]<name>.json     number ledgers (save_numbers)
    results/<top>/<sub>/figures/_provenance.md           the EdaConfig that produced the family
    results/<top>/INDEX.md                               auto: every subfamily of <top>, with captions
    results/INDEX.md                                     auto: one line per family, with counts
    results/<top>/SUMMARY.md                             HAND-AUTHORED narrative (never auto-touched)
    results/{METRICS_REFERENCE,LIMITATIONS}.md           hand-authored (never auto-touched)
    results/schematics/                                  hand-authored method diagrams (never touched)

``<top>/<sub>`` is the FAMILY -- one research question, one notebook, one output folder -- set once
per notebook by :func:`set_family` and validated against ``config.FAMILIES``. A family is REQUIRED:
every ``save_*`` raises :class:`NoFamilyError` until one is set, because a bare-root fallback is how
an artifact ends up somewhere no index points at. ``<group>`` is an optional nested subpath for one
call (``save_fig(fig, name, group="trajectories")``).

**There is no ``<judge>/`` level.** Exp3 nested one under the per-arm families because those
artifacts were *produced by* a single grader. Exp4 has no such family: every family loads both
graders and puts them side by side inside the same table or figure, so a path segment naming one
grader would be a false claim about the file underneath it. If a future family really is
single-grader, give the artifact a judge-qualified NAME (``outcomes_gemma4E2B``); do not reintroduce
a path level, which would silently re-partition every existing leaf.

Import contract: this module needs exactly two names from :mod:`.config` -- ``FAMILIES`` (the
top -> subfamilies map, walked by :func:`build_index`) and ``split_family`` (the validator). It
imports nothing else from the package, so it can never be the reason an import cycle forms.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import numbers as _numbers
import os
import re
import shutil
import uuid
import zipfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import FAMILIES, split_family

__all__ = [
    "RESULTS_DIR",
    "PRESERVE",
    "EXPORT_EPOCH",
    "MD_MAX_BYTES",
    "MD_EXCERPT_ROWS",
    "FIG_FORMATS",
    "TABLE_FORMATS",
    "NoFamilyError",
    "set_family",
    "active_family",
    "family_root",
    "top_root",
    "set_formats",
    "active_formats",
    "save_fig",
    "save_table",
    "save_numbers",
    "save_provenance",
    "build_index",
    "reset_results",
    "prune_orphan_captions",
]


# ==============================================================================
#  Constants
# ==============================================================================

#: Root of the rendered results tree: ``eda/results/``. A module-level global rather than a
#: computed constant on purpose -- ``_selfcheck`` swaps it for a temp directory to exercise the
#: savers and the delete without touching the real tree, so every helper below must read it at
#: CALL time (never capture it at import).
RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))

#: Names this module must never delete or rewrite: hand-authored prose and diagrams that no
#: notebook regenerates. Enforced twice over -- structurally, because every walk here descends only
#: into a family's ``figures/`` + ``tables/`` (which is not where any of these live), and explicitly
#: by :func:`_guard_path`, so a future refactor that widened a walk raises instead of deleting.
PRESERVE = frozenset({"SUMMARY.md", "METRICS_REFERENCE.md", "LIMITATIONS.md", "schematics"})

#: Extensions :func:`build_index` and :func:`prune_orphan_captions` recognise, by artifact kind.
#: A figure format outside ``_FIG_EXTS`` would be written but never indexed, which is why
#: :func:`set_formats` refuses one.
_FIG_EXTS = (".png", ".pdf", ".svg")
_TAB_EXTS = (".md",)
_NUM_EXTS = (".json",)

#: Formats :func:`save_fig` / :func:`save_table` accept.
FIG_FORMATS = ("png", "pdf", "svg")
TABLE_FORMATS = ("md", "xlsx", "csv", "tex")

#: Byte ceiling for a rendered ``.md`` table. Past this the markdown stops being a document and
#: becomes a dataset in markdown clothing: unreadable in a diff, unreviewable in a PR, slow to
#: render. Over the limit we write a HEAD EXCERPT plus a pointer to the leaf workbook, which holds
#: every row on a sortable sheet. Only applies when ``xlsx`` is also being written -- with no
#: complete copy elsewhere, truncating would lose data.
MD_MAX_BYTES = 64 * 1024
MD_EXCERPT_ROWS = 60

#: Fixed timestamp stamped into every ``.xlsx`` (see :func:`_normalize_xlsx`). Any constant works;
#: what matters is that it never changes, so an unchanged table produces an unchanged file.
EXPORT_EPOCH = datetime.datetime(2026, 1, 1, 0, 0, 0)

#: Excel's hard cap on a sheet name.
_SHEET_NAME_MAX = 31
_SHEET_BAD_CHARS = re.compile(r"[\[\]:*?/\\]")

# Active routing + default formats. Set by ``notebook_setup`` from the EdaConfig; module-level
# because every notebook is one family and passing it through every call site would be noise.
_FAMILY = ""
_FIG_FORMATS: Sequence[str] = ("png",)
_TABLE_FORMATS: Sequence[str] = ("md", "xlsx")


class NoFamilyError(RuntimeError):
    """Raised when a ``save_*`` / index / reset helper runs before :func:`set_family`."""


# ==============================================================================
#  Routing state
# ==============================================================================


def set_family(family: str = "") -> None:
    """Point subsequent saves at ``results/<top>/<sub>/``.

    Args:
        family: A ``"<top>/<sub>"`` family from ``config.FAMILIES``. Pass ``""`` to clear the
            routing, after which every ``save_*`` raises :class:`NoFamilyError`.

    Raises:
        ValueError: from ``config.split_family`` if the family is unknown. A typo would otherwise
            create a phantom results folder that no index, summary or paper ledger points at --
            invisible precisely because nothing references it.

    Notes:
        ``notebook_setup`` calls this from ``EdaConfig.family``; a notebook should not call it
        directly. Clearing is what ``_selfcheck`` uses to assert there is no bare-root fallback.
    """
    global _FAMILY
    fam = _norm_rel(family)
    if fam:
        split_family(fam)                       # raises on an unknown family
    _FAMILY = fam


def active_family() -> str:
    """The active ``"<top>/<sub>"``, or ``""`` when routing is cleared."""
    return _FAMILY


def set_formats(*, fig_formats: Optional[Sequence[str]] = None,
                table_formats: Optional[Sequence[str]] = None) -> None:
    """Set the default save formats for this session (``notebook_setup`` calls it from EdaConfig).

    Args:
        fig_formats: any of :data:`FIG_FORMATS`; ``None`` leaves the current default.
        table_formats: any of :data:`TABLE_FORMATS`; ``None`` leaves the current default.

    Raises:
        ValueError: on an unknown format. Exp3 ignored unknown values, so a typo (``"xslx"``)
            silently produced no workbook and the loss only surfaced when someone went looking for
            a sheet.

    Warning:
        Keyword-only, unlike Exp3's positional version. A positional call is a ``TypeError`` here
        rather than a silent swap of the two lists.
    """
    global _FIG_FORMATS, _TABLE_FORMATS
    if fig_formats:
        _FIG_FORMATS = _check_formats(fig_formats, FIG_FORMATS, "figure")
    if table_formats:
        _TABLE_FORMATS = _check_formats(table_formats, TABLE_FORMATS, "table")


def active_formats() -> Dict[str, Sequence[str]]:
    """The current default formats, for the provenance banner and ``_selfcheck``."""
    return {"fig_formats": tuple(_FIG_FORMATS), "table_formats": tuple(_TABLE_FORMATS)}


def _check_formats(formats: Sequence[str], valid: Sequence[str], kind: str) -> Sequence[str]:
    out = tuple(str(f).strip().lstrip(".").lower() for f in formats)
    bad = [f for f in out if f not in valid]
    if bad:
        raise ValueError(f"unknown {kind} format(s) {bad}; valid: {list(valid)}")
    return out


# ==============================================================================
#  Path composition -- ONE function owns the layout
# ==============================================================================


def _norm_rel(path: Optional[str]) -> str:
    """Trim whitespace and leading/trailing separators; normalise ``\\`` to ``/``. Interior kept."""
    return (path or "").strip().replace("\\", "/").strip("/")


def _require_family() -> str:
    if not _FAMILY:
        raise NoFamilyError(
            "no results family is set -- call eda_analysis.notebook_setup(EdaConfig(family="
            "'<top>/<sub>')) (or exports.set_family(...)) before saving. There is deliberately no "
            "bare results/ fallback; valid families are listed in eda_analysis.config.FAMILIES.")
    return _FAMILY


def family_root() -> str:
    """``results/<top>/<sub>/`` for the active family. Raises :class:`NoFamilyError` if unset."""
    return os.path.join(RESULTS_DIR, *_require_family().split("/"))


def top_root() -> str:
    """``results/<top>/`` for the active family. Raises :class:`NoFamilyError` if unset."""
    return os.path.join(RESULTS_DIR, _require_family().split("/")[0])


def _leaf(kind: str, group: Optional[str] = None) -> str:
    """``results/<top>/<sub>/<kind>/[<group>/]`` -- the ONLY place the layout is composed.

    Args:
        kind: ``"figures"`` or ``"tables"``.
        group: optional nested subpath inside the family.

    Warning:
        Every path this module writes, indexes or deletes comes from here. Changing the layout means
        changing this function and nothing else; a second place that composes a path is a second
        layout, and the two will disagree the first time one of them is edited.
    """
    if kind not in ("figures", "tables"):
        raise ValueError(f"_leaf: kind must be 'figures' or 'tables', got {kind!r}")
    parts = [family_root(), kind]
    grp = _norm_rel(group)
    if grp:
        parts.extend(grp.split("/"))
    return os.path.join(*parts)


def _fig_dir(group: Optional[str] = None) -> str:
    return _leaf("figures", group)


def _tab_dir(group: Optional[str] = None) -> str:
    return _leaf("tables", group)


def _workbook_stem(group: Optional[str] = None) -> str:
    """Workbook name for a tables leaf: the family's ``<sub>`` (``outcomes.xlsx``), or the innermost
    group for a nested leaf (``trajectories.xlsx``). One workbook per leaf, one sheet per table."""
    grp = _norm_rel(group)
    return grp.split("/")[-1] if grp else _require_family().split("/")[1]


def _check_name(name: str) -> str:
    """Validate an artifact ``name``: a bare stem, no path separators, no ``..``.

    A name carrying a separator would quietly create a subfolder outside the leaf that ``group=``
    describes -- so the file would exist, and the index (which walks the leaf) would still not list
    it. Refusing is cheaper than explaining that later.
    """
    nm = str(name or "").strip()
    if not nm:
        raise ValueError("artifact name is empty")
    if "/" in nm or "\\" in nm or nm in (".", "..") or os.path.isabs(nm):
        raise ValueError(f"artifact name {name!r} must be a bare stem -- use group= for subfolders")
    return nm


def _guard_path(path: str) -> None:
    """Refuse to touch anything in :data:`PRESERVE` or outside ``results/`` (defence in depth).

    Called by every writer and by both deletes. The structural guarantee is that no walk here ever
    reaches a preserved name; this is the assertion that fires if that ever stops being true.
    """
    abs_path = os.path.abspath(path)
    root = os.path.abspath(RESULTS_DIR)
    if not (abs_path == root or abs_path.startswith(root + os.sep)):
        raise RuntimeError(f"refusing to modify a path outside results/: {path}")
    hits = [p for p in os.path.relpath(abs_path, root).replace("\\", "/").split("/")
            if p in PRESERVE]
    if hits:
        raise RuntimeError(f"refusing to modify a PRESERVED artifact ({hits[0]}): {path}")


def _prepare(kind: str, group: Optional[str], name: str) -> tuple:
    """``(leaf_dir, checked_name)`` -- create the leaf, guard it, validate the name."""
    nm = _check_name(name)
    leaf = _leaf(kind, group)
    _guard_path(leaf)
    os.makedirs(leaf, exist_ok=True)
    return leaf, nm


# ==============================================================================
#  Captions
# ==============================================================================


def _append_caption(dir_path: str, name: str, caption: Optional[str]) -> None:
    """Record (or refresh) the caption line for *name* in the leaf's ``CAPTIONS.md`` -- idempotent.

    Re-running a notebook REPLACES that artifact's line rather than appending a duplicate, and the
    file is kept SORTED rather than in save order, so a leaf written by more than one cell (or
    re-rendered in a different order) never churns in git without its content changing.
    """
    if not caption:
        return
    path = os.path.join(dir_path, "CAPTIONS.md")
    prefix = f"- **{name}** -- "
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh if not ln.startswith(prefix)]
    lines.append(prefix + " ".join(str(caption).split()) + "\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(sorted(lines))


def _read_captions(dir_path: str) -> Dict[str, str]:
    """``{name: caption}`` parsed from a leaf's ``CAPTIONS.md`` (empty when there is none)."""
    path = os.path.join(dir_path, "CAPTIONS.md")
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("- **") and "** -- " in line:
                out[line[4:line.index("** -- ")]] = line[line.index("** -- ") + 6:].rstrip("\n")
    return out


# ==============================================================================
#  Savers
# ==============================================================================


def save_fig(fig, name: str, *, group: Optional[str] = None,
             formats: Optional[Sequence[str]] = None, dpi: int = 200,
             caption: Optional[str] = None) -> List[str]:
    """Save *fig* to ``results/<family>/figures/[<group>/]<name>.<fmt>`` and log its caption.

    Args:
        fig: anything with ``.savefig`` (a Matplotlib ``Figure``, a seaborn ``FacetGrid``).
        name: bare stem -- no separators; use ``group`` for a subfolder.
        group: optional nested subpath inside the family.
        formats: any of :data:`FIG_FORMATS`; ``None`` uses the session default (PNG).
        dpi: raster resolution.
        caption: one sentence saying what the figure shows. Write it here or it never gets
            written -- it is what makes the artifact readable in the index months later.

    Returns:
        Absolute paths of the image files written, in the order the formats were given.

    Notes:
        Safe to call right before or after ``plt.show()``; the inline display is unaffected. The
        figure is NOT closed -- that stays the notebook's decision.
    """
    leaf, nm = _prepare("figures", group, name)
    fmts = _check_formats(formats, FIG_FORMATS, "figure") if formats else _FIG_FORMATS
    written = []
    for fmt in fmts:
        path = os.path.join(leaf, f"{nm}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    _append_caption(leaf, nm, caption)
    return written


def save_table(df: pd.DataFrame, name: str, *, group: Optional[str] = None,
               formats: Optional[Sequence[str]] = None, float_format: str = "%.3f",
               index: bool = False, caption: Optional[str] = None) -> List[str]:
    """Save *df* to ``results/<family>/tables/[<group>/]<name>.<fmt>`` and log its caption.

    Args:
        df: the table. **An empty frame is not silently fine** -- see below.
        name: bare stem; also the workbook SHEET name (see :func:`_sheet_name`).
        group: optional nested subpath inside the family.
        formats: any of :data:`TABLE_FORMATS`; ``None`` uses the session default
            (``.md`` + ``.xlsx``). ``.csv`` / ``.tex`` on request.
        float_format: printf-style format for floats in ``.md`` / ``.tex``.
        index: write the frame index as a column.
        caption: one sentence saying what the table reports.

    Returns:
        Absolute paths written (the workbook counts once, however many sheets it holds).

    Warning:
        **An EMPTY frame writes an explicit EMPTY-TABLE marker, not an empty file.** Exp3 shipped a
        0-byte ``.md`` for weeks: an upstream filter had dropped every row, and a 0-byte artifact
        reads as "rendered" to every check that asks whether the file exists. The marker says out
        loud that the producer returned nothing, so the absence is a finding rather than a gap.

    Notes:
        A table over :data:`MD_MAX_BYTES` is written to ``.md`` as a head excerpt plus a pointer to
        the workbook sheet that holds every row. Nothing is lost; the markdown stops pretending to
        be the complete artifact.
    """
    leaf, nm = _prepare("tables", group, name)
    fmts = _check_formats(formats, TABLE_FORMATS, "table") if formats else _TABLE_FORMATS
    base = os.path.join(leaf, nm)

    if df is None or len(df) == 0:
        path = f"{base}.md"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"> **EMPTY TABLE.** The producing notebook saved `{nm}` with 0 rows -- "
                     f"either the analysis found nothing to report for these arms, or an upstream "
                     f"filter dropped every row. Check the producer's inputs before reading this "
                     f"as an absence of effect.\n")
        print(f"[save_table] WARNING: {nm!r} is EMPTY -- wrote an explicit empty-table marker")
        _append_caption(leaf, nm, caption)
        return [path]

    written = []
    if "csv" in fmts:
        df.to_csv(f"{base}.csv", index=index)
        written.append(f"{base}.csv")
    if "tex" in fmts:
        try:
            tex = df.to_latex(index=index, float_format=lambda x: (float_format % x), escape=True)
        except Exception:
            tex = df.to_latex(index=index)
        with open(f"{base}.tex", "w", encoding="utf-8") as fh:
            fh.write(tex)
        written.append(f"{base}.tex")
    if "md" in fmts:
        md = _to_markdown(df, index=index, float_format=float_format)
        if len(md.encode("utf-8")) > MD_MAX_BYTES and "xlsx" in fmts:
            md = _md_excerpt(df, nm, index=index, float_format=float_format)
        with open(f"{base}.md", "w", encoding="utf-8") as fh:
            fh.write(md)
        written.append(f"{base}.md")
    if "xlsx" in fmts:
        xpath = _write_xlsx_sheet(leaf, nm, df, index=index, workbook=_workbook_stem(group))
        if xpath:
            written.append(xpath)
    _append_caption(leaf, nm, caption)
    return written


def save_numbers(name: str, values: Dict[str, Any], *, group: Optional[str] = None,
                 caption: Optional[str] = None) -> str:
    """Write/merge a NUMBER LEDGER at ``results/<family>/tables/[<group>/]<name>.json``.

    Args:
        name: ledger stem. One ledger per topic; several cells may contribute to one.
        values: ``{dotted.key: value}``. Each value is either a bare number/dict/list or already a
            ``{"value", "source", "note"}`` record; both are normalised to that record shape, so a
            write-up can cite ``results/<family>/tables/<name>.json :: <key>`` and get the number
            together with where it came from.
        group: optional nested subpath.
        caption: one sentence describing the ledger.

    Returns:
        The ledger path.

    Notes:
        **Merged, not overwritten.** Keys present in ``values`` REPLACE their records; every other
        key already in the ledger survives, so re-running one cell refreshes only its own numbers.
        Written atomically (tmp + ``os.replace``).

        Numpy/pandas scalars are coerced to plain Python; NaN and +/-inf become ``null``. The
        coercion is done BEFORE encoding, not in a ``default=`` hook: ``json.dump`` serialises a
        bare ``float('nan')`` natively as the literal ``NaN``, which no ``default=`` hook ever sees
        and which is not valid JSON -- Python reads it back, other parsers reject the file.
        ``allow_nan=False`` here turns any non-finite that survives coercion into a loud error
        rather than an unparseable ledger.

        The document carries no timestamp on purpose: a rendered-at field would make every ledger
        differ on every render and drown the ones that actually changed.
    """
    leaf, nm = _prepare("tables", group, name)
    path = os.path.join(leaf, f"{nm}.json")

    numbers: Dict[str, dict] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                numbers = dict(json.load(fh).get("numbers", {}))
        except (OSError, ValueError):
            numbers = {}                       # a corrupt ledger is replaced, not propagated

    for key, val in (values or {}).items():
        if isinstance(val, dict) and "value" in val and set(val) <= {"value", "source", "note"}:
            rec = {"value": _coerce(val["value"]),
                   "source": str(val.get("source", "")), "note": str(val.get("note", ""))}
        else:
            rec = {"value": _coerce(val), "source": "", "note": ""}
        numbers[str(key)] = rec

    doc = {"_family": _require_family(), "_name": nm, "numbers": numbers}
    _atomic_write(path, json.dumps(doc, indent=1, ensure_ascii=True, allow_nan=False,
                                   default=_json_default) + "\n")
    _append_caption(leaf, nm, caption)
    return path


# ------------------------------------------------------------------ coercion --


def _coerce(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and containers to JSON-native Python.

    Non-finite floats (NaN, +/-inf) become ``None`` -- JSON has no spelling for them, and a ledger
    that cannot be parsed by anything but Python is not a ledger.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if value is pd.NaT or value is getattr(pd, "NA", object()):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, (str, bool)) or value is None:
            return value
    if isinstance(value, _numbers.Integral):
        return int(value)
    if isinstance(value, _numbers.Real):
        num = float(value)
        return None if (math.isnan(num) or math.isinf(num)) else num
    if isinstance(value, np.ndarray):
        return [_coerce(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return [_coerce(rec) for rec in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_coerce(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


def _json_default(obj):
    """Last-resort encoder for a type :func:`_coerce` did not recognise: stringify it."""
    if hasattr(obj, "item"):
        try:
            return _coerce(obj.item())
        except Exception:
            pass
    return str(obj)


# -------------------------------------------------------------------- markdown --


def _to_markdown(df: pd.DataFrame, *, index: bool, float_format: str) -> str:
    """Markdown table via pandas/tabulate, with a dependency-free fallback."""
    try:
        return df.to_markdown(index=index, floatfmt=float_format.replace("%", ""))
    except Exception:
        frame = df.reset_index() if index else df

        def fmt(val):
            try:
                return float_format % float(val)
            except (TypeError, ValueError):
                return str(val)

        cols = list(frame.columns)
        head = "| " + " | ".join(map(str, cols)) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
                for row in frame.itertuples(index=False)]
        return "\n".join([head, sep, *rows]) + "\n"


def _md_excerpt(df: pd.DataFrame, name: str, *, index: bool, float_format: str) -> str:
    """Render an over-:data:`MD_MAX_BYTES` table as a head excerpt + a pointer to the workbook."""
    head = df.head(MD_EXCERPT_ROWS)
    sheet = _sheet_name(name)
    return (
        f"> **Excerpt -- first {len(head):,} of {len(df):,} rows.** The full table is too large to "
        f"read as markdown, so it lives on sheet `{sheet}` of the `.xlsx` workbook in this folder. "
        f"Load it with `pandas.read_excel(..., sheet_name=\"{sheet}\")`.\n\n"
        + _to_markdown(head, index=index, float_format=float_format)
        + f"\n\n_... {len(df) - len(head):,} further rows in the workbook._\n"
    )


# ----------------------------------------------------------------- workbooks --


def _sheet_name(name: str) -> str:
    """Excel sheet name for a table: illegal characters replaced, length capped at 31.

    A name longer than the cap keeps its first 24 characters plus a short digest of the FULL name.
    Plain truncation is not safe: two tables whose names share a 31-character prefix would land on
    one sheet and the second would silently overwrite the first -- inside a workbook that still
    looks complete. The digest is content-free and deterministic, so the sheet a ``.md`` excerpt
    points at is the sheet that exists.
    """
    clean = _SHEET_BAD_CHARS.sub("_", str(name))
    if len(clean) <= _SHEET_NAME_MAX:
        return clean
    digest = hashlib.md5(str(name).encode("utf-8")).hexdigest()[:6]
    return f"{clean[:_SHEET_NAME_MAX - 7]}_{digest}"


def _write_xlsx_sheet(dir_path: str, name: str, df: pd.DataFrame, *, index: bool,
                      workbook: str) -> Optional[str]:
    """Write/replace *df* as a sheet in the leaf workbook ``<workbook>.xlsx``.

    One workbook per tables leaf, one sheet per table, so a reader opens one file and gets every
    table of that leaf sortable and filterable. Re-running a notebook replaces its own sheet.

    Sheets are re-sorted alphabetically after each write: openpyxl appends a replaced sheet at the
    END, so without this a partial re-render would reorder the workbook and change its bytes while
    every number stayed identical.

    Returns the workbook path, or ``None`` if the write failed (a missing engine or a workbook
    open in Excel must not take the ``.md`` export down with it).
    """
    xpath = os.path.join(dir_path, f"{workbook}.xlsx")
    try:
        mode = "a" if os.path.exists(xpath) else "w"
        kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
        with pd.ExcelWriter(xpath, engine="openpyxl", mode=mode, **kwargs) as writer:
            df.to_excel(writer, sheet_name=_sheet_name(name), index=index)
            _sort_sheets(writer.book)
        _normalize_xlsx(xpath)
        return xpath
    except Exception as exc:
        print(f"  [exports] xlsx skipped for {name}: {exc}")
        return None


def _sort_sheets(book) -> None:
    """Best-effort alphabetical sheet order. Never fatal -- a workbook in the wrong order is a
    cosmetic diff, while raising here would lose the table."""
    try:
        book._sheets.sort(key=lambda ws: ws.title)
    except Exception:
        pass


_CORE_TS_RE = re.compile(
    rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)")


def _normalize_xlsx(path: str) -> None:
    """Rewrite *path* with fixed timestamps so an unchanged table stays BYTE-IDENTICAL.

    openpyxl stamps the current clock in two places: ``docProps/core.xml`` (and it rewrites
    ``modified`` during save, so setting the property beforehand does not stick) and every zip
    entry's mtime. The consequence is that a re-render rewrites EVERY workbook even when no number
    moved -- and a diff of 30 changed workbooks reads as "the numbers moved" when only the clock
    did, which is exactly the noise that hides the one workbook that did change.

    Rewriting the archive afterwards is the reliable fix: member CONTENT is copied through
    untouched, only the timestamps are pinned. The rewrite is atomic (tmp + ``os.replace``) so a
    crash mid-normalise cannot leave a half-written workbook where a valid one was.
    """
    with zipfile.ZipFile(path) as src:
        items = [(info.filename, src.read(info.filename)) for info in src.infolist()]
    stamp = (EXPORT_EPOCH.year, EXPORT_EPOCH.month, EXPORT_EPOCH.day,
             EXPORT_EPOCH.hour, EXPORT_EPOCH.minute, EXPORT_EPOCH.second)
    iso = EXPORT_EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    tmp = _tmp_path(path)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for fname, data in items:
            if fname == "docProps/core.xml":
                data = _CORE_TS_RE.sub(rb"\g<1>" + iso + rb"\g<2>", data)
            info = zipfile.ZipInfo(fname, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            out.writestr(info, data)
    os.replace(tmp, path)


# ==============================================================================
#  Provenance
# ==============================================================================


def save_provenance(cfg, scores=None) -> str:
    """Write ``results/<family>/figures/_provenance.md``: the config that produced this family.

    Args:
        cfg: the active ``EdaConfig`` (anything with ``as_dict()``, or a mapping).
        scores: the scores frame the notebook is working from, if any. Its arms, metrics, graders
            and row count are recorded.

    Returns:
        The banner path.

    Notes:
        This is what makes a rendered artifact traceable: a figure with no record of the arm filter,
        the metric list or the row count behind it cannot be reproduced or contradicted -- it can
        only be believed. The banner carries no timestamp, for the same reason the ledgers do not:
        it would change on every render and hide the renders that changed something.

        Column names are probed rather than assumed, so a change in the score frame's schema
        degrades to a shorter banner instead of an exception in the middle of a render.
    """
    leaf = _fig_dir(None)
    _guard_path(leaf)
    os.makedirs(leaf, exist_ok=True)

    lines = [f"# Provenance -- family `{_require_family()}`\n",
             "_Written by `exports.save_provenance`. Every artifact in this family was produced by "
             "the configuration below._\n"]

    if scores is not None and not getattr(scores, "empty", True):
        cols = list(getattr(scores, "columns", []))
        lines.append("## Data")
        lines.append(f"- **rows:** {len(scores):,}")
        for label, candidates in (("arms", ("arm", "arm_label", "experiment_name")),
                                  ("metrics", ("metric", "questionnaire")),
                                  ("model states", ("model_state", "model_iter", "model")),
                                  ("graders", ("judge", "judge_tag"))):
            col = next((c for c in candidates if c in cols), None)
            if col is not None:
                vals = sorted(str(v) for v in pd.unique(scores[col]))
                lines.append(f"- **{label}** (`{col}`): {vals}")
        lines.append(f"- **columns:** {cols}")
        lines.append("")

    cfgd = cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg)
    lines.append("## EdaConfig")
    for key, val in cfgd.items():
        lines.append(f"- `{key}` = {val}")
    lines.append("")
    lines.append("## Export formats")
    for key, val in active_formats().items():
        lines.append(f"- `{key}` = {list(val)}")

    path = os.path.join(leaf, "_provenance.md")
    _atomic_write(path, "\n".join(lines) + "\n")
    return path


# ==============================================================================
#  Index
# ==============================================================================


def _tmp_path(path: str) -> str:
    """Unique sibling temp path. The uuid matters: ``render_results.py`` renders families in a
    THREAD pool, so two writers of ``results/INDEX.md`` share a pid and would collide on a
    pid-only name -- one would delete the other's temp file mid-write."""
    return f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"


def _atomic_write(path: str, text: str) -> None:
    """Write *text* to *path* via tmp + ``os.replace``, so a reader never sees a partial file."""
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _list_artifacts(filenames: Sequence[str], exts: Sequence[str]) -> List[str]:
    """Indexable artifacts in one directory listing: matching extension, not private/bookkeeping."""
    return sorted(f for f in filenames
                  if f.lower().endswith(tuple(exts)) and not f.startswith(("CAPTIONS", "_")))


def _count_family(fam_dir: str) -> Dict[str, int]:
    """``{figures, tables, numbers}`` counts under one ``results/<top>/<sub>/``."""
    counts = {"figures": 0, "tables": 0, "numbers": 0}
    for kind, exts, key in (("figures", _FIG_EXTS, "figures"),
                            ("tables", _TAB_EXTS, "tables"),
                            ("tables", _NUM_EXTS, "numbers")):
        root = os.path.join(fam_dir, kind)
        if not os.path.isdir(root):
            continue
        for _dirpath, _dirnames, filenames in os.walk(root):
            counts[key] += len(_list_artifacts(filenames, exts))
    return counts


def build_index() -> List[str]:
    """Write ``results/<top>/INDEX.md`` for the active family's TOP and refresh ``results/INDEX.md``.

    The top index lists every subfamily of that top -- figures with their captions, then tables,
    then number ledgers -- so one file answers "what did this research question produce?". The root
    index lists every family in ``config.FAMILIES`` order with artifact counts, and marks the ones
    that have never been rendered.

    Returns:
        ``[<top>/INDEX.md, INDEX.md]`` (absolute paths).

    Notes:
        Stale caption lines are pruned first (:func:`prune_orphan_captions`), so an artifact that
        stopped being generated stops being described. Both files are written atomically because
        the renderer runs families concurrently and two units of the same top can finish together.

        The index is a MAP, not a reading: ``SUMMARY.md`` beside it is the hand-authored narrative
        and is never touched here.
    """
    prune_orphan_captions()
    top = _require_family().split("/")[0]
    troot = top_root()
    _guard_path(troot)
    os.makedirs(troot, exist_ok=True)

    lines = [f"# Exp4 EDA artifact index -- `{top}/`\n",
             "_Generated by `eda_analysis.build_index()`. See `SUMMARY.md` in this folder for the "
             "written analysis, and `../INDEX.md` for the map of every family._\n",
             f"_Each subfamily below is written by `notebooks/{top}/<sub>.ipynb`. Exp4 artifacts "
             "carry every grader INSIDE the table or figure, so there is no `<judge>/` level._\n"]

    for sub in FAMILIES.get(top, []):
        fam_dir = os.path.join(troot, sub)
        lines.append(f"\n## {top}/{sub}")
        if not os.path.isdir(fam_dir):
            lines.append("_(not rendered yet)_")
            continue
        listed = False
        for kind, exts, label in (("figures", _FIG_EXTS, "Figures"),
                                  ("tables", _TAB_EXTS, "Tables"),
                                  ("tables", _NUM_EXTS, "Number ledgers")):
            root = os.path.join(fam_dir, kind)
            if not os.path.isdir(root):
                continue
            block = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort()
                arts = _list_artifacts(filenames, exts)
                if not arts:
                    continue
                rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
                caps = _read_captions(dirpath)
                block.append(f"\n**{label}** -- `{kind}/" + ("" if rel == "." else rel + "/") + "`")
                for art in arts:
                    cap = caps.get(os.path.splitext(art)[0])
                    block.append(f"- `{art}`" + (f" -- {cap}" if cap else ""))
            if block:
                listed = True
                lines += block
        if not listed:
            lines.append("_(empty)_")

    top_index = os.path.join(troot, "INDEX.md")
    _atomic_write(top_index, "\n".join(lines) + "\n")
    return [top_index, _refresh_root_index()]


def _refresh_root_index() -> str:
    """Rewrite ``results/INDEX.md``: one line per family with figure/table/ledger counts."""
    lines = ["# Exp4 EDA results -- family map\n",
             "_Generated by `eda_analysis.build_index()`; one line per family "
             "(`results/<top>/<sub>/`). Each `<top>/` folder carries its own `INDEX.md` (artifact "
             "list with captions) and a hand-authored `SUMMARY.md`. `METRICS_REFERENCE.md`, "
             "`LIMITATIONS.md` and `schematics/` are hand-authored and never regenerated._\n",
             "| family | figures | tables | ledgers | notebook |",
             "|---|---:|---:|---:|---|"]
    for top, subs in FAMILIES.items():
        for sub in subs:
            fam_dir = os.path.join(RESULTS_DIR, top, sub)
            notebook = f"`notebooks/{top}/{sub}.ipynb`"
            if not os.path.isdir(fam_dir):
                lines.append(f"| `{top}/{sub}` | - | - | - | {notebook} _(not rendered yet)_ |")
                continue
            n = _count_family(fam_dir)
            lines.append(f"| `{top}/{sub}` | {n['figures']} | {n['tables']} | {n['numbers']} "
                         f"| {notebook} |")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "INDEX.md")
    _atomic_write(path, "\n".join(lines) + "\n")
    return path


# ==============================================================================
#  Housekeeping (the two functions that delete)
# ==============================================================================


def prune_orphan_captions() -> int:
    """Drop ``CAPTIONS.md`` lines whose artifact is no longer in that folder. Returns lines removed.

    :func:`_append_caption` refreshes a caption by name but never removes the line for a figure or
    table that stopped being generated, so orphans accumulate across renders and the index starts
    describing files that do not exist. This walks the ACTIVE FAMILY's ``figures/`` + ``tables/``
    and keeps only ``- **<name>** -- ...`` lines with a matching ``<name>.<ext>`` beside them.

    A caption's name always equals its file stem (both come from the one ``save_*`` ``name=``
    argument), so only genuine orphans are dropped; lines that are not captions are left alone.
    """
    removed = 0
    exts = _FIG_EXTS + _TAB_EXTS + _NUM_EXTS
    for kind in ("figures", "tables"):
        root = os.path.join(family_root(), kind)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "CAPTIONS.md" not in filenames:
                continue
            _guard_path(dirpath)
            stems = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(exts)}
            cap_path = os.path.join(dirpath, "CAPTIONS.md")
            with open(cap_path, encoding="utf-8") as fh:
                lines = fh.readlines()
            kept = []
            for line in lines:
                if (line.startswith("- **") and "** -- " in line
                        and line[4:line.index("** -- ")] not in stems):
                    removed += 1
                    continue
                kept.append(line)
            if len(kept) != len(lines):
                with open(cap_path, "w", encoding="utf-8") as fh:
                    fh.writelines(kept)
    return removed


def reset_results(groups: Optional[Sequence[str]] = None) -> int:
    """Clear the ACTIVE FAMILY's generated artifacts before a clean regenerate. Returns files removed.

    Args:
        groups: ``None`` clears the whole family scope; a list of group subpaths (e.g.
            ``["trajectories"]``) removes just those nested subfolders. Subfolders are recreated
            lazily on the next save.

    Warning:
        **This function deletes recursively.** It is confined two ways, and both must hold: it
        descends ONLY into ``results/<family>/{figures,tables}/`` -- never the family root, never
        the top root, so the hand-authored ``SUMMARY.md`` and everything else in :data:`PRESERVE`
        is out of reach structurally -- and every target is passed through :func:`_guard_path`
        first, which refuses a preserved name or any path outside ``results/``. If you ever widen
        the walk, the guard is what stops the widening from being a data-loss bug.

        It also clears only the ACTIVE family. A render of one family can never delete another's
        output, which is what lets the renderer run families concurrently.
    """
    removed = 0
    for kind in ("figures", "tables"):
        scope = _leaf(kind, None)
        if not os.path.isdir(scope):
            continue
        if groups is None:
            targets = [scope]
        else:
            targets = [os.path.join(scope, *_norm_rel(g).split("/"))
                       for g in groups if _norm_rel(g)]
        for target in targets:
            if not os.path.isdir(target):
                continue
            _guard_path(target)
            for _dirpath, _dirnames, filenames in os.walk(target):
                removed += len(filenames)
            shutil.rmtree(target)
    return removed
