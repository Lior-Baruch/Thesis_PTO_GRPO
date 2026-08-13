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
- ``<group>`` is the notebook's topic family (``"1_outcomes"``, ``"2_questionnaires"``,
  ``"3_validity"``, ``"4_heterogeneity"``, ``"5_training"``, ``"6_preference"``, ``"7_stats"``,
  ``"8_measurement"``) set via
  :func:`set_export_group` — the family NUMBER matches the producing notebook's number, so any
  artifact traces straight back to its notebook. A per-call ``group=`` on ``save_fig``/``save_table``
  overrides it for one save and may be a NESTED subpath within the family
  (``"1_outcomes/trajectories"``, ``"4_heterogeneity/problem"``, ``"0_headline"``). With no group set, artifacts fall
  back to the view's flat roots.

The ``formats=`` kwarg lets a one-off call request extra formats (e.g.
``save_fig(fig, name, formats=("pdf", "png"))``); the defaults are PNG figures + ``.md``/``.xlsx`` tables.

Notebooks keep showing plots inline AND call :func:`save_fig` / :func:`save_table` on their key
artifacts with stable, thesis-ready names. Captions accumulate in each group's ``CAPTIONS.md``;
:func:`build_index` writes the per-view ``results/<view>/INDEX.md``. :func:`reset_results` clears the
generated figure/table subfolders of the active view but PRESERVES the hand-authored ``SUMMARY.md``.
"""

import datetime
import os
import re
import shutil
import zipfile
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

# ── Families that are ABOUT the judges rather than PRODUCED BY one ──────────────
# The <judge>/ level means "this grader produced this artifact". For a cross-judge artifact that
# claim is false: `multijudge_variance_decomposition.png` plots BOTH graders, and filing it under
# `gpt-4o-mini/` says the primary made it alone — the one figure proving the judges agree, attributed
# to a single judge. Such families skip the segment entirely (2026-07-29).
#
# They are also judge-INVARIANT: `reliability.py` loads every judge from the score lake explicitly
# and ignores the active EDA_JUDGE, so re-rendering under another grader would reproduce byte-
# identical output. render_views.py therefore renders them exactly once.
JUDGE_INVARIANT_GROUPS = frozenset({"8_measurement"})


def _is_judge_invariant(group: str) -> bool:
    """True if *group* (or its top-level family, for nested subpaths) skips the ``<judge>/`` level."""
    return bool(group) and group.split("/", 1)[0] in JUDGE_INVARIANT_GROUPS


# The active view subfolder. Empty = legacy bare roots (results/figures/...).
_VIEW = ""
# The active export group (per-notebook subfolder). Empty = the view's flat roots.
_GROUP = ""
# Default formats used when a save_* call doesn't pass `formats=` explicitly. Set by
# notebook_setup() from EdaConfig (figures -> PNG images, tables -> readable .md + Excel .xlsx).
_FIG_FORMATS = ("png",)
_TABLE_FORMATS = ("md", "xlsx")

# Byte ceiling for a rendered ``.md`` table. Past this the markdown stops being a document and
# becomes a dataset in markdown clothing — unreadable in a diff, unreviewable in a PR, and slow to
# render on GitHub (``multijudge_all_pairs_contrasts`` reached 407 KB / 1,849 rows). Over the limit
# we write a HEAD EXCERPT plus a pointer to the group workbook, which already holds every row on a
# sortable sheet. Only applies when ``.xlsx`` is also being written — with no complete copy
# elsewhere, truncating would lose data.
MD_MAX_BYTES = 64 * 1024
MD_EXCERPT_ROWS = 60

# Fixed timestamp stamped into every .xlsx (see _freeze_workbook_timestamps). Any constant works;
# what matters is that it never changes, so an unchanged table produces an unchanged file.
EXPORT_EPOCH = datetime.datetime(2026, 1, 1, 0, 0, 0)


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


# ── View/judge-aware path helpers (everything downstream routes through these) ─
#
# Layout:  results/<view>/{figures,tables}/<group>/<judge>/<name>.<ext>
#
# The JUDGE is the DEEPEST level, so a family's outputs from every grader sit side by side and are
# trivially comparable (`1_outcomes/gpt-4o-mini/` next to `1_outcomes/claude-haiku-4-5/`).
#
# EVERY judge gets a folder, including the primary (2026-07-28). Until then the primary rendered
# FLAT at `<group>/` so that adding a second grader churned no existing path — but that made the
# layout assert something the project no longer believes: that one grader is the default and the
# other an annex. A figure's path now always names the grader that produced it, which is the whole
# point of having measured them against each other.
def _results_root() -> str:
    return os.path.join(RESULTS_DIR, _VIEW) if _VIEW else RESULTS_DIR


def _judge_sub() -> str:
    """Deepest path segment for the active judge — the short model label, never empty."""
    from .constants import judge_dirname
    return judge_dirname()


def _figures_root() -> str:
    return os.path.join(_results_root(), "figures")


def _tables_root() -> str:
    return os.path.join(_results_root(), "tables")


def _leaf(root: str, group: Optional[str]) -> str:
    """``<root>/<group>/[<judge>]`` — the single place group + judge are composed.

    A :data:`JUDGE_INVARIANT_GROUPS` family gets no ``<judge>`` segment: its artifacts are about
    the graders, not the output of one.
    """
    g = _norm_group(group) if group is not None else _GROUP
    parts = [root]
    if g:
        parts.append(g)
    j = "" if _is_judge_invariant(g) else _judge_sub()
    if j:
        parts.append(j)
    return os.path.join(*parts)


def _fig_dir(group: Optional[str] = None) -> str:
    """Figures dir for *group* (per-call override, may be nested), under the active judge."""
    return _leaf(_figures_root(), group)


def _tab_dir(group: Optional[str] = None) -> str:
    """Tables dir for *group* (per-call override, may be nested), under the active judge."""
    return _leaf(_tables_root(), group)


def _append_caption(dir_path: str, name: str, caption: Optional[str]):
    """Record (or refresh) the caption line for *name* in CAPTIONS.md — idempotent.

    Re-running a notebook overwrites the existing line for that artifact instead of appending
    a duplicate, so CAPTIONS.md stays one-line-per-artifact across reruns.

    Lines are kept SORTED rather than in save order. ``0_headline/`` is written by three different
    notebooks, so append order depended on which finished first — the file churned in git on every
    render without its content ever changing (2026-07-28, alongside the seaborn seed fix).
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
        f.writelines(sorted(lines))


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
        md = _to_markdown(df, index=index, float_format=float_format)
        if len(md.encode("utf-8")) > MD_MAX_BYTES and "xlsx" in formats:
            md = _md_excerpt(df, name, index=index, float_format=float_format)
        with open(f"{base}.md", "w", encoding="utf-8") as f:
            f.write(md)
    if "xlsx" in formats:
        _write_xlsx_sheet(d, name, df, index=index)
    _append_caption(d, name, caption)
    return d


def _md_excerpt(df: pd.DataFrame, name: str, *, index: bool, float_format: str) -> str:
    """Render an over-:data:`MD_MAX_BYTES` table as a head excerpt + a pointer to the workbook.

    The full frame is always on the ``<family>.xlsx`` sheet of the same name, so nothing is lost —
    the ``.md`` stops pretending to be the complete artifact and says where the complete one is.
    Called only when ``xlsx`` is among the requested formats (see :func:`save_table`).
    """
    head = df.head(MD_EXCERPT_ROWS)
    sheet = re.sub(r"[\[\]:*?/\\]", "_", name)[:31]
    return (
        f"> **Excerpt — first {len(head):,} of {len(df):,} rows.** The full table is too large to "
        f"read as markdown, so it lives on sheet `{sheet}` of the `.xlsx` workbook in this folder. "
        f"Load it with `pandas.read_excel(..., sheet_name=\"{sheet}\")`.\n\n"
        + _to_markdown(head, index=index, float_format=float_format)
        + f"\n\n_... {len(df) - len(head):,} further rows in the workbook._\n"
    )


def _write_xlsx_sheet(dir_path: str, name: str, df: pd.DataFrame, *, index: bool = False) -> None:
    """Write/replace ``df`` as a sheet in the group workbook ``<group_or_tables>.xlsx``.

    One workbook per tables subfolder, one sheet per table name (Excel caps sheet names at 31 chars).
    Re-running a notebook overwrites that sheet (idempotent). Requires ``openpyxl``.

    Named for the FAMILY, not the leaf directory. Since 2026-07-28 the leaf is normally the judge
    (see :func:`_leaf`), and ``gpt-4o-mini.xlsx`` would say nothing about what is inside — so skip
    one level up. The judge is already unambiguous from the containing path. Works for nested
    subgroups too: ``tables/2_questionnaires/mici/<judge>/`` -> ``mici.xlsx``, as before the move.

    A :data:`JUDGE_INVARIANT_GROUPS` family has NO judge level, so there the leaf already *is* the
    family and skipping a level up would reach the ``tables/`` root — yielding ``tables.xlsx``.
    """
    parts = [p for p in dir_path.rstrip("/\\").replace("\\", "/").split("/") if p]
    if any(p in JUDGE_INVARIANT_GROUPS for p in parts):
        group = parts[-1]                                   # no judge level: leaf IS the family
    else:
        group = parts[-2] if len(parts) >= 2 else "tables"  # [-1] is the judge
    xpath = os.path.join(dir_path, f"{group}.xlsx")
    sheet = re.sub(r"[\[\]:*?/\\]", "_", name)[:31]
    try:
        mode = "a" if os.path.exists(xpath) else "w"
        kw = {"if_sheet_exists": "replace"} if mode == "a" else {}
        with pd.ExcelWriter(xpath, engine="openpyxl", mode=mode, **kw) as xw:
            df.to_excel(xw, sheet_name=sheet, index=index)
        _normalize_xlsx(xpath)
    except Exception as e:   # missing engine / locked file — don't break the .md export
        print(f"  [exports] xlsx skipped for {name}: {e}")


_CORE_TS_RE = re.compile(
    rb"(<dcterms:(created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)")


def _normalize_xlsx(path: str) -> None:
    """Rewrite ``path`` with fixed timestamps so an unchanged table stays byte-identical.

    openpyxl stamps the current time in TWO places — ``docProps/core.xml`` (and it overwrites
    ``modified`` during save, so setting the property beforehand does not stick) and every zip
    entry's mtime. The result was that re-rendering rewrote EVERY ``.xlsx`` on unchanged data: the
    same non-determinism :data:`~eda_analysis.constants.BOOT_SEED` fixed on the figure side, and
    just as misleading — a diff showing 30 changed workbooks reads as "the numbers moved" when only
    the clock did.

    Rewriting the archive afterwards is the reliable fix: member CONTENT is copied through
    untouched (verified below by the ``_selfcheck`` round-trip), only the timestamps are pinned.
    """
    with zipfile.ZipFile(path) as src:
        items = [(i.filename, src.read(i.filename)) for i in src.infolist()]
    stamp = (EXPORT_EPOCH.year, EXPORT_EPOCH.month, EXPORT_EPOCH.day,
             EXPORT_EPOCH.hour, EXPORT_EPOCH.minute, EXPORT_EPOCH.second)
    iso = EXPORT_EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    tmp = f"{path}.{os.getpid()}.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for fname, data in items:
            if fname == "docProps/core.xml":
                data = _CORE_TS_RE.sub(rb"\g<1>" + iso + rb"\g<3>", data)
            zi = zipfile.ZipInfo(fname, date_time=stamp)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            out.writestr(zi, data)
    os.replace(tmp, path)


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
    """Write a per-run provenance banner to ``results/<view>/figures/<group>/<judge>/_provenance.md``.

    Records the active ``EdaConfig`` (incl. the view) + the arms/metrics actually present in
    ``scores`` so every regenerated figure set is traceable to the config that produced it.
    Returns the file path.

    Routed through :func:`_fig_dir` so it lands in the JUDGE leaf like every other artifact — the
    config it records includes ``judge``, so one file per judge is the only correct arity. (Before
    2026-07-28 it wrote to the group root, where a ``--judge`` render silently overwrote the
    primary's record with the second judge's config.)

    **No group, no banner** (2026-07-29). A provenance file records which config produced a
    *family* of artifacts; with no family it documents nothing and lands at ``figures/<judge>/``,
    i.e. a judge folder sitting at family depth, which reads like a phantom family in the results
    tree. An interactive ``notebook_setup(EdaConfig())`` left exactly one such file behind. Returns
    ``""`` in that case instead of writing.
    """
    g = (group if group is not None else _GROUP) or ""
    if not g:
        return ""
    d = _fig_dir(g)
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


def prune_orphan_captions() -> int:
    """Drop stale ``CAPTIONS.md`` lines whose artifact no longer exists in that folder.

    :func:`_append_caption` refreshes a caption by name but never removes the line for a figure or
    table that stopped being generated, so 'orphan' captions accumulate across renders (e.g. a
    per-metric zoom that a later pass dropped). This walks the active view's ``figures/`` +
    ``tables/`` roots and rewrites each ``CAPTIONS.md`` to keep only ``- **<name>** — …`` lines for
    which a ``<name>.<ext>`` artifact sits alongside it. A valid caption's name always equals its
    file stem (both come from the single ``save_fig``/``save_table`` ``name=`` arg), so only genuine
    orphans are dropped; non-caption lines are left untouched. Returns the number of lines removed.
    """
    removed = 0
    for root, exts in ((_figures_root(), _FIG_EXTS), (_tables_root(), _TAB_EXTS)):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, filenames in os.walk(root):
            if "CAPTIONS.md" not in filenames:
                continue
            stems = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(exts)}
            cap = os.path.join(dirpath, "CAPTIONS.md")
            with open(cap, encoding="utf-8") as f:
                lines = f.readlines()
            kept = []
            for ln in lines:
                if ln.startswith("- **") and "** — " in ln and ln[4:ln.index("** — ")] not in stems:
                    removed += 1
                    continue
                kept.append(ln)
            if len(kept) != len(lines):
                with open(cap, "w", encoding="utf-8") as f:
                    f.writelines(kept)
    return removed


def build_index() -> str:
    """Write ``results/<view>/INDEX.md`` listing every figure + table of the active view.

    A per-view artifact map so the reader sees, in one place, which notebook (group) produced what.
    Returns the index path. (The hand-authored ``SUMMARY.md`` is the narrative companion to this map.)
    """
    prune_orphan_captions()  # self-heal: drop caption lines for figures/tables no longer generated
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
        # 4_heterogeneity/<trait>, …) are listed too; dirnames sorted for numeric-ish family order.
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
    hand-authored ``SUMMARY.md`` (and anything else in :data:`PRESERVE`) is kept structurally,
    by never descending where it lives, rather than by name-filtering during deletion.

    - ``groups`` given (e.g. ``["1_outcomes", "2_questionnaires"]``) → remove just those
      ``figures/<group>/`` + ``tables/<group>/`` subfolders (nested content included).
    - ``groups=None`` → remove ALL group subfolders under both roots.
    - ``flat=True`` → also delete loose figure/table files sitting at the (view's) flat roots.
      Subfolders are recreated lazily on the next save.

    **Judge-scoped.** Clears only the ACTIVE judge's leaf (``<group>/<judge>/``) and never the
    group folder itself, which holds every other grader's copy of the same family. Without this
    scoping a routine ``--judge`` regenerate would delete another judge's tree as a side effect.
    Since 2026-07-28 this includes the primary, which is a judge like any other here.

    A :data:`JUDGE_INVARIANT_GROUPS` family has no ``<judge>`` level, so there the whole family
    folder IS the active scope and is cleared directly — it belongs to no single grader.
    """
    judge = _judge_sub()
    for root in (_figures_root(), _tables_root()):
        if not os.path.isdir(root):
            continue
        group_dirs = ([os.path.join(root, g) for g in groups] if groups is not None
                      else [os.path.join(root, d) for d in os.listdir(root)
                            if os.path.isdir(os.path.join(root, d)) and d != judge])
        # The judge is the deepest level: drop <group>/<judge>, keep <group> for other graders.
        # Judge-invariant families have no such level — clear the family folder itself.
        for d in group_dirs:
            s = d if _is_judge_invariant(os.path.basename(d)) else os.path.join(d, judge)
            if os.path.isdir(s):
                shutil.rmtree(s)
        if flat:
            # Group-less saves land at <root>/<judge>/ rather than loose in <root>/.
            flat_root = os.path.join(root, judge)
            if os.path.isdir(flat_root):
                shutil.rmtree(flat_root)
