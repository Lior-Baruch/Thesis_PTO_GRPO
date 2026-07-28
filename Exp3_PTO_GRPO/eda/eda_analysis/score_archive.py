"""score_archive.py — the score lake's parquet fold: build it, verify it, and read through it.

One CSV per conversation is a *write-time* shape: a file is one completed unit of work, so an
interrupted scoring run resumes by skipping what exists. It is a poor shape for everything else —
the lake is ~50,300 files averaging ~190 bytes, and a cold ``load_scores_long`` spent ~90s opening
them one at a time (measured 2026-07-28; ~64s of it raw per-file overhead through the Drive
symlink). This module folds each ``(judge, rep, metric)`` partition into one parquet file: 31 files,
0.6 MB, and a cold rebuild drops to ~1s.

**The staleness guard is the whole design.** A second read path that drifts from its source is
worse than no second read path, because it fails silently — a figure would render off scores that
are no longer on disk. So ``build()`` records a content signature per partition in
``_parquet/_manifest.json``, and :func:`rows_for` recomputes that signature before serving anything.
Any mismatch (a re-score, a new model, a deleted CSV) degrades to the CSVs automatically. This is
the same (name, size, mtime) mechanism ``data.load_cached`` already trusts for its own cache, not a
new assumption — and the fold is never written by the scorers, only by an explicit ``build``.

Reading is therefore always *correct*; it is only *fast* when the fold happens to be current.
Rebuild with ``python tools/consolidate_scores.py build`` after new scoring, or delete
``_parquet/`` — a stale fold costs speed, never accuracy.
"""
import hashlib
import json
import os
from typing import Dict, Iterator, Optional, Tuple

import pandas as pd

from .constants import EVAL_SCORES, JUDGE_PARTITION

PARQUET_ROOT = os.path.join(EVAL_SCORES, "_parquet")
MANIFEST_PATH = os.path.join(PARQUET_ROOT, "_manifest.json")

# Columns this module adds to a partition frame; stripped before rows go back to callers so a
# parquet-served row is indistinguishable from one read straight out of its CSV.
_INDEX_COLS = ("oracle", "model", "file_index")

_manifest_cache: Optional[Dict[str, str]] = None
_frame_cache: Dict[str, Optional[pd.DataFrame]] = {}
_checked: Dict[str, bool] = {}


# ── layout ────────────────────────────────────────────────────────────────────
def partition_key(tag: str, rep: int, metric: str) -> str:
    return f"{JUDGE_PARTITION}{tag}/rep={rep}/metric={metric}"


def parquet_path(tag: str, rep: int, metric: str) -> str:
    return os.path.join(PARQUET_ROOT, f"{JUDGE_PARTITION}{tag}", f"rep={rep}",
                        f"metric={metric}.parquet")


def metric_dir(tag: str, rep: int, metric: str) -> str:
    return os.path.join(EVAL_SCORES, f"{JUDGE_PARTITION}{tag}", f"rep={rep}", f"metric={metric}")


def partition_signature(mdir: str) -> str:
    """blake2b over every ``*.csv`` under one ``metric=<M>/`` tree as (relpath, size, mtime_ns).

    Recursive, unlike ``data._content_signature`` (which takes explicit leaf dirs) — a partition's
    membership is the whole subtree, so a model folder appearing or vanishing must change the
    digest too, not just an edit to a file already known.
    """
    h = hashlib.blake2b(digest_size=16)
    if not os.path.isdir(mdir):
        return "missing"
    for dp, dns, fns in os.walk(mdir):
        dns.sort()
        rel = os.path.relpath(dp, mdir).replace("\\", "/")
        for fn in sorted(fns):
            if not fn.endswith(".csv"):
                continue
            try:
                st = os.stat(os.path.join(dp, fn))
            except OSError:
                continue
            h.update(f"{rel}/{fn}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def parse_eval_dir(ddir: str) -> Optional[Tuple[str, int, str, str, str]]:
    """``…/judge=<tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>`` -> its five parts, or None.

    None whenever the path is not inside the lake or does not have exactly this shape — the read
    path treats that as "not foldable" and falls back, rather than guessing.
    """
    try:
        rel = os.path.relpath(os.path.abspath(ddir), EVAL_SCORES)
    except ValueError:                                   # different drive
        return None
    parts = rel.replace("\\", "/").split("/")
    if len(parts) != 5 or parts[0].startswith(".."):
        return None
    jd, rd, md, od, model = parts
    if not (jd.startswith(JUDGE_PARTITION) and rd.startswith("rep=")
            and md.startswith("metric=") and od.startswith("oracle=")):
        return None
    try:
        rep = int(rd[len("rep="):])
    except ValueError:
        return None
    return jd[len(JUDGE_PARTITION):], rep, md[len("metric="):], od[len("oracle="):], model


# ── read path ─────────────────────────────────────────────────────────────────
def _manifest() -> Dict[str, str]:
    global _manifest_cache
    if _manifest_cache is None:
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as fh:
                _manifest_cache = dict(json.load(fh).get("partitions", {}))
        except Exception:
            _manifest_cache = {}
    return _manifest_cache


def _partition_frame(tag: str, rep: int, metric: str) -> Optional[pd.DataFrame]:
    """The partition's frame if its fold is provably current, else None. Verified once per process."""
    key = partition_key(tag, rep, metric)
    if key in _frame_cache:
        return _frame_cache[key]
    recorded = _manifest().get(key)
    frame = None
    if recorded and recorded == partition_signature(metric_dir(tag, rep, metric)):
        try:
            frame = pd.read_parquet(parquet_path(tag, rep, metric))
        except Exception:                                # missing/corrupt -> fall back to CSVs
            frame = None
    _frame_cache[key] = frame
    return frame


def rows_for(ddir: str) -> Optional[Iterator[Tuple[int, pd.Series]]]:
    """``(file_index, row)`` pairs for one eval dir, served from the fold — or None to use CSVs.

    Rows come back in ``file_index`` order with the bookkeeping columns stripped, so they match what
    reading the directory's CSVs yields. Returns None (never raises, never a partial answer) if the
    path is not in the lake, the fold is absent, or its signature no longer matches disk.
    """
    parsed = parse_eval_dir(ddir)
    if parsed is None:
        return None
    tag, rep, metric, oracle, model = parsed
    frame = _partition_frame(tag, rep, metric)
    if frame is None:
        return None
    sub = frame[(frame["oracle"] == oracle) & (frame["model"] == model)]
    if sub.empty:
        # An empty slice is ambiguous — genuinely no scores, or a model this fold predates. The
        # signature already proved the fold current, so trust it and report the empty result.
        return iter(())
    sub = sub.sort_values("file_index")
    idx = sub["file_index"].to_numpy()
    payload = sub.drop(columns=[c for c in _INDEX_COLS if c in sub.columns])
    return ((int(i), row) for i, (_, row) in zip(idx, payload.iterrows()))


def fold_status() -> str:
    """One-line human summary: how many partitions exist and how many are current."""
    man = _manifest()
    if not man:
        return "score fold: absent (reads go straight to CSVs)"
    fresh = sum(1 for k, sig in man.items()
                if sig == partition_signature(os.path.join(EVAL_SCORES, k)))
    return (f"score fold: {fresh}/{len(man)} partitions current"
            + ("" if fresh == len(man) else " — stale ones fall back to CSVs"))


def reset_cache() -> None:
    """Drop the memoized manifest/frames (tests, or after a rebuild inside one process)."""
    global _manifest_cache
    _manifest_cache = None
    _frame_cache.clear()
    _checked.clear()
