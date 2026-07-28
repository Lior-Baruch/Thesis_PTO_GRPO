"""consolidate_scores.py — CLI for the score lake's parquet fold.

    python tools/consolidate_scores.py build     # write data/eval_scores/_parquet/ + its manifest
    python tools/consolidate_scores.py verify    # re-read every CSV and diff against the fold
    python tools/consolidate_scores.py report    # sizes, file counts, how many partitions are current

Rationale, the staleness guard, and the read path live in ``eda_analysis/score_archive.py``; this
file only drives them. Run ``build`` after any new scoring — a stale fold is detected and bypassed,
so it costs speed rather than correctness, but there is no reason to leave it stale.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.dirname(HERE)
sys.path.insert(0, EDA)

from eda_analysis.constants import EVAL_SCORES, JUDGE_PARTITION      # noqa: E402
from eda_analysis import score_archive as A                          # noqa: E402


def _partitions():
    """Yield ``(judge_tag, rep, metric, metric_dir)`` for every partition on disk."""
    if not os.path.isdir(EVAL_SCORES):
        return
    for jd in sorted(os.listdir(EVAL_SCORES)):
        if not jd.startswith(JUDGE_PARTITION):
            continue                                   # _batches/, summary/, _parquet/
        tag = jd[len(JUDGE_PARTITION):]
        jroot = os.path.join(EVAL_SCORES, jd)
        for rd in sorted(os.listdir(jroot)):
            if not rd.startswith("rep="):
                continue
            rroot = os.path.join(jroot, rd)
            for md in sorted(os.listdir(rroot)):
                if md.startswith("metric="):
                    yield tag, int(rd[4:]), md[len("metric="):], os.path.join(rroot, md)


def _read_partition(mdir: str) -> pd.DataFrame:
    """Every CSV under one ``metric=<M>/`` folder as one frame, + oracle/model/file_index."""
    frames = []
    for od in sorted(os.listdir(mdir)):
        if not od.startswith("oracle="):
            continue
        oracle = od[len("oracle="):]
        oroot = os.path.join(mdir, od)
        for model in sorted(os.listdir(oroot)):
            mroot = os.path.join(oroot, model)
            if not os.path.isdir(mroot):
                continue
            for fn in sorted(os.listdir(mroot)):
                stem, ext = os.path.splitext(fn)
                if ext != ".csv" or not stem.isdigit():
                    continue
                df = pd.read_csv(os.path.join(mroot, fn))
                if df.empty:
                    continue
                df = df.head(1).copy()                 # one row per conversation, by construction
                df.insert(0, "file_index", int(stem))
                df.insert(0, "model", model)
                df.insert(0, "oracle", oracle)
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["oracle", "model", "file_index"]).reset_index(drop=True)


def build() -> int:
    n_files = n_rows = 0
    manifest = {}
    for tag, rep, metric, mdir in _partitions():
        df = _read_partition(mdir)
        if df.empty:
            print(f"  (empty) judge={tag} rep={rep} metric={metric}")
            continue
        dst = A.parquet_path(tag, rep, metric)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        df.to_parquet(dst, index=False)
        manifest[A.partition_key(tag, rep, metric)] = A.partition_signature(mdir)
        n_files += 1
        n_rows += len(df)
        print(f"  judge={tag:38s} rep={rep} metric={metric:7s} rows={len(df):>6,}")
    # Manifest LAST and atomically: it is what the read path trusts, so it must never name a
    # partition whose parquet failed to write. A missing entry just degrades to reading CSVs.
    os.makedirs(A.PARQUET_ROOT, exist_ok=True)
    tmp = A.MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"partitions": manifest}, fh, indent=1, sort_keys=True)
    os.replace(tmp, A.MANIFEST_PATH)
    A.reset_cache()
    print(f"\nwrote {n_files} parquet files, {n_rows:,} rows -> {A.PARQUET_ROOT}")
    print(f"manifest ({len(manifest)} partitions) -> {A.MANIFEST_PATH}")
    return 0


def verify() -> int:
    """Re-read every CSV and diff against the fold, then check the read path agrees too."""
    bad, total = [], 0
    for tag, rep, metric, mdir in _partitions():
        dst = A.parquet_path(tag, rep, metric)
        if not os.path.exists(dst):
            bad.append(f"missing parquet: judge={tag} rep={rep} metric={metric}")
            continue
        want, got = _read_partition(mdir), pd.read_parquet(dst)
        total += len(want)
        if list(want.columns) != list(got.columns):
            bad.append(f"columns differ: judge={tag} rep={rep} metric={metric}")
            continue
        try:
            pd.testing.assert_frame_equal(want, got, check_dtype=False, rtol=0, atol=0)
        except AssertionError as e:
            bad.append(f"values differ: judge={tag} rep={rep} metric={metric}: "
                       f"{str(e).splitlines()[0]}")
        sig = A.partition_signature(mdir)
        if A._manifest().get(A.partition_key(tag, rep, metric)) != sig:
            bad.append(f"manifest signature stale: judge={tag} rep={rep} metric={metric}")
    print(f"compared {total:,} rows across the lake")
    for b in bad:
        print("  FAIL:", b)
    print("VERIFY:", "PASS" if not bad else f"FAIL ({len(bad)} partitions)")
    return 0 if not bad else 1


def report() -> int:
    def tree_stats(root, suffix):
        n = size = 0
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith(suffix):
                    n += 1
                    size += os.path.getsize(os.path.join(dp, fn))
        return n, size

    csv_n, csv_b = tree_stats(EVAL_SCORES, ".csv")
    pq_n, pq_b = tree_stats(A.PARQUET_ROOT, ".parquet") if os.path.isdir(A.PARQUET_ROOT) else (0, 0)
    print(f"  CSVs    : {csv_n:>7,} files  {csv_b / 1e6:>8.1f} MB")
    print(f"  parquet : {pq_n:>7,} files  {pq_b / 1e6:>8.1f} MB")
    if pq_n:
        print(f"  ratio   : {csv_n / pq_n:>7.0f}x fewer files, {csv_b / pq_b:>5.1f}x smaller")
    print(f"  {A.fold_status()}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    fn = {"build": build, "verify": verify, "report": report}.get(cmd)
    if fn is None:
        sys.exit(f"usage: {os.path.basename(__file__)} {{build|verify|report}}")
    sys.exit(fn())
