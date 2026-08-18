"""
_selfcheck.py — a fast regression guard for the ``eda_analysis`` package.

Run it after ANY refactor of the EDA package (module splits, alias changes, plotting moves,
caching) to confirm the invariants the notebooks rely on still hold::

    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck          # full (structural + data)
    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck --fast   # structural only (no disk reads)
    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck --probe  # + the heavy PTO preference probe

It is deliberately dependency-light and self-contained: no notebook execution, no torch/trl, no
OpenAI. Data checks are SKIPPED (not failed) when the Exp3 eval data isn't readable locally, so the
structural half still guards a machine without the Drive mount.

Checks
------
Structural (always, no disk):
  * package imports; ``__all__`` names all resolve.
  * the VIEW->ks map + case-insensitive aliases are consistent; ``EdaConfig`` round-trips.
  * every submodule-qualified call in the notebooks (``plots.x`` / ``figures.x`` / ``stats.x`` / …)
    resolves to a real attribute — this is the guard that catches a plotting/module split that
    drops or renames a public name.
Data (skipped if data absent):
  * ``discover_arms`` finds the LA0 arms; ``load_scores_long`` is non-empty with Q1Q2 present.
  * known Q1Q2 endpoints reproduce (PTO_LA0 final ~= 4.26, GRPO_LA0 final ~= 3.75).
  * persona recovery is an exact 0..n-1 permutation for every iter of every arm.
  * the compute axis costs every trained iteration, its phases sum, and its iso-compute contrast
    pairs on persona (not file_index) with the ``stats.py`` sign convention.
  * every rendered ``<judge>/`` subtree is newer than that judge's newest score — the guard against
    a bare ``render_views.py`` (primary-only) silently leaving a held-out judge's tables behind.
Probe (opt-in, heavy — needs sentence-transformers + pref pairs):
  * the PTO Mass-Mean-Probe ``wins_correct`` > 0.5 (the chosen-rejected direction separates pairs).
"""

from __future__ import annotations

import json
import io
import os
import re
import sys
import traceback
from glob import glob
from typing import Callable, List, Tuple

import numpy as np

# Import the package the same way the notebooks do (cwd = eda/, package on the path).
import eda_analysis as E  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_EDA_DIR = os.path.dirname(_HERE)                       # .../eda

# Known-good endpoints (EDA's Q1Q2 = mean(Q1,Q2) convention; see project memory / SUMMARY.md).
_KNOWN_Q1Q2_FINAL = {"PTO_LA0": 4.26, "GRPO_LA0": 3.75}
_KNOWN_TOL = 0.02

# Submodule names a notebook may qualify a call with (live modules + the figures/plots aliases).
_SUBMODULES = ("plotting", "plots", "figures", "data",
               "stats", "behavior", "training", "pref", "exports", "compute")


# ── check harness ─────────────────────────────────────────────────────────────
class _Skip(Exception):
    """Raised by a check to mark itself SKIPPED (e.g. data absent) rather than FAILED."""


_Results = List[Tuple[str, str, str]]   # (name, status, detail)


def _run(name: str, fn: Callable[[], str], results: _Results) -> None:
    try:
        detail = fn() or ""
        results.append((name, "PASS", detail))
    except _Skip as s:
        results.append((name, "SKIP", str(s)))
    except Exception as e:                                          # noqa: BLE001
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        if os.environ.get("SELFCHECK_TRACE"):
            traceback.print_exc()


# ── structural checks ─────────────────────────────────────────────────────────
def _c_all_resolves() -> str:
    missing = [n for n in E.__all__ if not hasattr(E, n)]
    assert not missing, f"__all__ names not resolvable on package: {missing}"
    return f"{len(E.__all__)} __all__ names resolve"


def _c_view_map() -> str:
    from eda_analysis import config as C
    assert set(C._VIEW_KS) == {"all", "L0", "L5"}, C._VIEW_KS
    # Every alias target is a real view; case-insensitive.
    for k, v in C._VIEW_ALIASES.items():
        assert v in C._VIEW_KS, f"alias {k!r} -> unknown view {v!r}"
    assert C._VIEW_ALIASES["l0"] == "L0" and C._VIEW_ALIASES["all"] == "all"
    # The view that owns the RQ-i artifacts must be a real, K-SPECIFIC view: `all` would put the K
    # contrast back in the retired pooled tree, and an unknown name would silently save nowhere.
    assert C.RQ_I_VIEW in C._VIEW_KS, f"RQ_I_VIEW {C.RQ_I_VIEW!r} is not a view"
    assert C._VIEW_KS[C.RQ_I_VIEW], f"RQ_I_VIEW {C.RQ_I_VIEW!r} must be a K-specific view"
    return f"view->ks {C._VIEW_KS} | RQ-i owner {C.RQ_I_VIEW}"


def _c_config_roundtrip() -> str:
    cfg = E.EdaConfig(view="L0", export_group="1_outcomes", selection="best")
    d = cfg.as_dict()
    assert d["view"] == "L0" and d["selection"] == "best"
    cfg2 = cfg.with_(selection="all")
    assert cfg2.selection == "all" and cfg.selection == "best", "with_ must not mutate original"
    return "EdaConfig.as_dict/with_ OK"


def _c_live_aliases() -> str:
    # These two aliases are heavily used in notebooks and MUST keep resolving to plotting.
    assert E.figures is E.plotting and E.plots is E.plotting, "figures/plots must alias plotting"
    return "figures/plots -> plotting"


def _c_scoring_surface() -> str:
    """The scoring subpackage (Run_Eval + Judge_Reliability backend) keeps its public surface.

    Imports ``eda_analysis.scoring`` (NOT imported by the package __init__ — it scans disk for the
    registry) and asserts every name the two scoring notebooks reference still resolves. Works with
    an empty registry (Drive offline): the check is structural, not data.
    """
    from eda_analysis import scoring
    from eda_analysis.scoring import registry, conversations, pipeline, judge
    missing = [n for n in scoring.__all__ if not hasattr(scoring, n)]
    assert not missing, f"scoring.__all__ names not resolvable: {missing}"
    for mod, attrs in {
        registry: ("ScoringConfig", "EXPERIMENTS", "resolve_paths", "get_model_names",
                   "get_model_eval_layout", "eval_csv_dir", "EVAL_QUESTIONNAIRE_DIRS"),
        conversations: ("load_data", "combine_data", "reconstruct_conversation_text",
                        "add_model_metadata_columns"),
        pipeline: ("EVAL_CODE_AVAILABLE",),
        judge: ("JudgeSpec", "PRIMARY_JUDGE", "EVAL_SCORES_ROOT", "JUDGE_METRIC_COLS",
                "run_judge_scoring", "load_judge_scores", "repeatability_table",
                "agreement_table", "contrast_preservation", "icc_2_1"),
    }.items():
        bad = [a for a in attrs if not hasattr(mod, a)]
        assert not bad, f"{mod.__name__} missing: {bad}"
    if pipeline.EVAL_CODE_AVAILABLE:
        for a in ("evaluate_conversation", "build_default_eval_configs", "run_all_evaluations_async"):
            assert hasattr(pipeline, a), f"pipeline missing {a}"
    return (f"scoring surface OK ({len(scoring.__all__)} names; registry has "
            f"{len(registry.EXPERIMENTS)} experiments)")


def _notebook_symbol_refs() -> dict:
    """Scan committed notebooks for ``<submodule>.<attr>(`` calls -> {submodule: {attr, ...}}."""
    pat = re.compile(r"\b(" + "|".join(_SUBMODULES) + r")\.([A-Za-z_][A-Za-z0-9_]*)")
    refs: dict = {m: set() for m in _SUBMODULES}
    for nb in glob(os.path.join(_EDA_DIR, "notebooks", "**", "*.ipynb"), recursive=True):
        d = json.load(open(nb, encoding="utf-8"))
        for cell in d.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            for mod, attr in pat.findall(src):
                refs[mod].add(attr)
    return refs


def _c_notebook_refs_resolve() -> str:
    refs = _notebook_symbol_refs()
    bad = []
    total = 0
    for mod, attrs in refs.items():
        submod = getattr(E, mod, None)
        for attr in attrs:
            total += 1
            if submod is None or not hasattr(submod, attr):
                bad.append(f"{mod}.{attr}")
    assert not bad, f"notebook-referenced symbols not resolvable: {sorted(bad)}"
    used = {m: len(a) for m, a in refs.items() if a}
    return f"{total} notebook symbol refs resolve across {used}"


def _c_cache_mechanism() -> str:
    """Data-independent: memoize a dummy frame, assert miss->build / hit->read / bypass / invalidate.

    Uses a temp input file so it needs no Drive data; cleans up its own cache entries and never
    touches real caches (unique probe name). Guards :func:`~eda_analysis.data.load_cached`.
    """
    import glob as _glob
    import tempfile
    from eda_analysis import data

    class _FakeArm:                       # minimal shape load_cached's arm-signature needs
        exp_name = "_selfcheck_probe"
        iters = [0]

    saved = os.environ.pop("EDA_NO_CACHE", None)     # allow the cache ON for this check
    tmp = tempfile.mkdtemp()
    sig = os.path.join(tmp, "sig.csv")
    probe = "_selfcheck_probe"
    calls = {"n": 0}

    def builder():
        import pandas as pd
        calls["n"] += 1
        return pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    def _clean():
        for fp in _glob.glob(os.path.join(data._CACHE_DIR, f"{probe}__*.parquet")):
            try:
                os.remove(fp)
            except OSError:
                pass

    try:
        open(sig, "w").write("a\n1\n")
        data.set_cache(True)
        _clean()
        f1 = data.load_cached(probe, [_FakeArm()], builder, input_roots=[tmp])   # miss -> build+write
        f2 = data.load_cached(probe, [_FakeArm()], builder, input_roots=[tmp])   # hit  -> read parquet
        assert calls["n"] == 1, f"miss+hit should build once, built {calls['n']}"
        assert f1.equals(f2), "cached frame != freshly built (parquet round-trip mismatch)"
        # bypass path rebuilds
        data.set_cache(False)
        data.load_cached(probe, [_FakeArm()], builder, input_roots=[tmp])
        assert calls["n"] == 2, "set_cache(False) should bypass the cache (rebuild)"
        # content invalidation: rewrite the input (new size) -> new signature -> rebuild
        data.set_cache(True)
        open(sig, "w").write("a\n1\n2\n3\n")
        data.load_cached(probe, [_FakeArm()], builder, input_roots=[tmp])
        assert calls["n"] == 3, "changed input file should invalidate the cache (rebuild)"
    finally:
        _clean()
        for f in (sig,):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass
        data.set_cache(None)
        if saved is not None:
            os.environ["EDA_NO_CACHE"] = saved
    return "miss->build, hit->read (parquet equal), bypass rebuilds, content-change invalidates"


# ── data checks ───────────────────────────────────────────────────────────────
def _discover_or_skip():
    arms = E.discover_arms()
    if not arms:
        raise _Skip("no arms on disk (Drive data not mounted?)")
    return arms


def _c_discover() -> str:
    arms = _discover_or_skip()
    labels = {a.label for a in arms}
    for need in ("PTO_LA0", "GRPO_LA0"):
        assert need in labels, f"expected arm {need} missing; found {sorted(labels)}"
    return f"{len(arms)} arms: {sorted(labels)}"


def _c_update_probe() -> str:
    """The cross-method preference probe: one weight scale, real groups only, right iteration join.

    Three things the probe would otherwise get silently wrong:
    1. **Scale.** DPO's ±1 pair and GRPO's standardized advantages are only comparable after the
       per-group rescale; if it regresses, every cross-method contrast becomes a scale artifact
       rather than a finding. Asserted as Σw = 0 and Σ|w| = 2 in EVERY group of BOTH methods.
    2. **Which groups count.** A PTO branch point that logged a ``chosen`` but no ``rejected`` was
       τ-filtered out and never trained on — it must not survive as a one-sided +2 push. Asserted
       as "PTO groups hold exactly 2 candidates, and every group has both signs".
    3. **The iteration join.** ``link_to_outcomes`` credits train_iter *n* with
       ``eval(model_iter_n) − eval(model_iter_{n-1})``; an off-by-one would attribute every
       update's effect to its neighbour and no test would notice. Asserted against the raw
       iteration means (identical when both cells are complete over the same 96 personas).
    """
    arms = _discover_or_skip()
    from eda_analysis import pref
    cands = pref.load_weighted_candidates(arms)
    if cands.empty:
        raise _Skip("no generations.jsonl on disk")
    per_group = (cands.assign(_abs=cands["weight"].abs())
                 .groupby(pref._GROUP_KEYS)
                 .agg(wsum=("weight", "sum"), wabs=("_abs", "sum"),
                      wmax=("weight", "max"), wmin=("weight", "min")))
    worst_sum = float(per_group.wsum.abs().max())
    assert worst_sum < 1e-9, f"group weights do not cancel (max |Σw| = {worst_sum:.2e})"
    assert bool(np.allclose(per_group.wabs, 2.0)), (
        f"group weights not on the shared Σ|w|=2 scale "
        f"(range {per_group.wabs.min():.4f}–{per_group.wabs.max():.4f})")
    assert (per_group.wmax > 0).all() and (per_group.wmin < 0).all(), (
        "a group survived without both an up- and a down-weighted side")
    pto = cands[cands.method == "PTO"]
    if not pto.empty:
        sizes = pto.groupby(pref._GROUP_KEYS).size().unique()
        assert set(sizes) == {2}, f"PTO groups must be exactly (chosen, rejected); sizes seen: {sizes}"

    # 3 — the iteration join, checked against raw means on a complete arm.
    scores = E.load_scores_long(arms)
    feats = pref.weighted_lexical_contrast(cands)
    link = pref.link_to_outcomes(feats, scores, metrics=["Q1Q2"])
    assert not link.empty, "link_to_outcomes produced no rows"
    checked = 0
    for arm, d in link.groupby("arm"):
        q = scores[(scores.arm == arm) & (scores.questionnaire == "Q1Q2")]
        means = q.groupby("iteration")["score"].mean()
        for r in d.itertuples(index=False):
            it = int(r.train_iter)
            if it in means.index and (it - 1) in means.index:
                want = float(means[it] - means[it - 1])
                assert abs(r.delta_mean - want) < 1e-6, (
                    f"{arm} train_iter {it}: link delta {r.delta_mean:.4f} != "
                    f"eval(iter {it}) - eval(iter {it-1}) = {want:.4f} — iteration join is off")
                checked += 1
    assert checked, "no complete step to verify the iteration join against"

    # 4 — the counterfactual re-weighting keeps the group set and the scale, and its score-only
    # `dpo` rule must select a MAXIMUM-scoring candidate (exact row identity is the wrong test:
    # ~40% of PTO groups have tied maxima, where any tie-break is arbitrary).
    all_c = pref.load_weighted_candidates(arms, drop_zero_weight=False)
    for rule in ("dpo", "grpo"):
        rw = pref.reweight(all_c, rule)
        pg = (rw.assign(_abs=rw["weight"].abs()).groupby(pref._GROUP_KEYS)
              .agg(wsum=("weight", "sum"), wabs=("_abs", "sum")))
        assert float(pg.wsum.abs().max()) < 1e-9, f"reweight({rule!r}) weights do not cancel"
        assert bool(np.allclose(pg.wabs, 2.0)), f"reweight({rule!r}) left the shared scale"
    pto_arm = next((a.label for a in arms if a.method == "PTO"), None)
    if pto_arm:
        chk = pref.rule_reconstruction_check(all_c, pto_arm)
        assert chk and chk["chosen_picks_a_maximum"] == 1.0 and chk["rejected_picks_a_minimum"] == 1.0, (
            f"the score-only dpo rule disagrees with {pto_arm}'s recorded roles: {chk}")
    return (f"{len(cands)} candidates, {cands.groupby(pref._GROUP_KEYS).ngroups} groups on one "
            f"weight scale; {checked} (arm, iter) joins land on the right eval step; "
            f"both counterfactual rules preserve scale + pick the extremes")


def _c_cross_k() -> str:
    """RQ-i's escape hatch: ``cross_k_scores`` must widen the READ without moving the WRITE.

    The K0-vs-K5 contrast is the one comparison a K-specific view cannot serve, and the tempting
    wrong fix is to relax the view itself — which would silently re-point every other artifact in
    the notebook at a pooled frame. So assert both halves: the returned frame carries both K arms
    while the view's own scores carry exactly one, AND the export root is byte-identical before and
    after the call. Also pins the sign convention shared by ``k_means_by_iter`` and
    ``paired_k_comparison`` (+ => K=0 higher), which the tables' captions promise.
    """
    _discover_or_skip()
    from eda_analysis import exports
    S = E.notebook_setup(E.EdaConfig(view=E.RQ_I_VIEW, export_group="7_stats", verbose=False))
    if S.SCORES.empty:
        raise _Skip(f"no scores for view {E.RQ_I_VIEW}")
    root_before = exports._results_root(), exports._fig_dir(None)
    ks_view = set(S.SCORES.K.unique())
    cross = E.cross_k_scores(S)
    root_after = exports._results_root(), exports._fig_dir(None)
    assert root_before == root_after, f"cross_k_scores moved the export root: {root_before} -> {root_after}"
    assert len(ks_view) == 1, f"view {E.RQ_I_VIEW} should hold ONE K, holds {sorted(ks_view)}"
    ks_cross = set(cross.K.unique())
    assert ks_view < ks_cross, f"cross-K frame did not widen K: view {sorted(ks_view)} vs {sorted(ks_cross)}"

    means = E.k_means_by_iter(cross, "PTO")
    paired = E.paired_k_comparison(cross, "PTO")
    assert not means.empty and not paired.empty, "PTO K contrast empty on the cross-K frame"
    # Same sign convention on both sides: where a cell is complete (96/96), the unpaired difference
    # of arm means IS the paired mean_delta — pairing buys precision, not a different centre.
    m = means[(means.metric == "Q1Q2") & (means.n_K0 == 96) & (means.n_K5 == 96)]
    p = paired[paired.metric == "Q1Q2"].set_index("iteration")["mean_delta"]
    checked = 0
    for r in m.itertuples():
        if r.iteration in p.index:
            assert abs(r.delta - float(p.loc[r.iteration])) < 1e-9, (
                f"iter {r.iteration}: k_means delta {r.delta:.4f} != paired mean_delta "
                f"{float(p.loc[r.iteration]):.4f} — sign/ordering convention diverged")
            checked += 1
    assert checked, "no complete matched iteration to cross-check the K delta against"
    return (f"view {E.RQ_I_VIEW} K={sorted(ks_view)} -> cross-K K={sorted(ks_cross)}, "
            f"exports unmoved, {checked} matched iters agree on delta")


def _c_compute_axis() -> str:
    """The COMPUTE axis: costs must be positive, monotone, and pair on persona.

    This is the newest and least battle-tested frame in the package, and it is the one that can
    silently invert a headline: an iso-compute contrast reads a DIFFERENT iteration from each arm,
    so a ``file_index`` join there would pair unrelated conversations (personas are reshuffled
    ``seed + k + 1`` every iteration). Asserts, in order:

      * every arm on disk gets a cost row, ``cum_gpu_h`` is non-decreasing, and iteration 0 is free;
      * the phase decomposition adds up (``gen + build + train == gpu_h``);
      * PTO's cost is recovered at all — its DPO trainer writes no per-step artifact, so its rows
        come from the TB ``wall_time`` fallback, and a silent regression there would zero PTO out
        rather than raise;
      * the iso-compute contrast pairs on ``persona_id`` (checked by construction: the paired n
        equals the persona count even though iter_a != iter_b);
      * the sign convention matches ``stats.py`` (+ => arm_a higher).
    """
    _discover_or_skip()
    arms = E.discover_arms()
    comp = E.iteration_compute(arms)
    if comp.empty:
        raise _Skip("no run artifacts readable (Drive symlinks offline?)")

    trained = comp[comp.iteration > 0]
    assert not trained.empty, "no trained iterations timed"
    assert (trained.gpu_h > 0).all(), "a trained iteration billed <= 0 GPU-h"
    assert (comp[comp.iteration == 0].cum_gpu_h == 0).all(), "base state is not free"
    phase_sum = trained[["gen_h", "build_h", "train_h"]].sum(axis=1)
    assert np.allclose(phase_sum, trained.gpu_h), "phase decomposition does not sum to gpu_h"
    for arm, g in comp.groupby("arm"):
        c = g.sort_values("iteration").cum_gpu_h.to_numpy()
        assert (np.diff(c) >= -1e-9).all(), f"{arm}: cum_gpu_h is not monotone"

    methods = set(trained.method.unique())
    if "PTO" in methods:
        pto = trained[trained.method == "PTO"]
        assert (pto.build_h > 0).any(), (
            "PTO rows carry no build_h — the pref-tree phase is its DOMINANT cost, so a zero here "
            "means the pairs.csv/conversation mtime probe regressed, not that it was free")
        assert (pto.train_h > 0).all(), "PTO train_h missing — the TB wall_time fallback regressed"

    S = E.notebook_setup(E.EdaConfig(view=E.RQ_I_VIEW, export_group="7_stats", verbose=False))
    if S.SCORES.empty:
        return f"{len(trained)} costed iterations; scores absent for the contrast half"
    KS = E.cross_k_scores(S)
    pairs_checked = contrast_rows = 0
    for a, b in (("GRPO_LA5", "GRPO_LA0"), ("PTO_LA5", "PTO_LA0")):
        if not {a, b} <= set(KS.arm.unique()):
            continue
        P = E.iso_compute_pairs(comp, a, b)
        if P.empty:
            continue
        pairs_checked += len(P)
        T = E.iso_compute_contrast(KS, comp, a, b, metrics=["Q1Q2"])
        if T.empty:
            continue
        contrast_rows += len(T)
        n_personas = KS[KS.arm == a].persona_id.nunique()
        off = T[T.iter_a != T.iter_b]
        assert not off.empty, "no off-diagonal budget match to test persona pairing on"
        assert (off.n == n_personas).all(), (
            f"{a} vs {b}: iso-compute paired n={sorted(off.n.unique())} != {n_personas} personas — "
            "this is what a file_index join across unmatched iterations looks like")
        # sign convention: + mean_delta must mean arm_a scored higher
        r = T.iloc[0]
        ma = KS[(KS.model == r.model_a) & (KS.questionnaire == "Q1Q2")].score.mean()
        mb = KS[(KS.model == r.model_b) & (KS.questionnaire == "Q1Q2")].score.mean()
        assert np.sign(r.mean_delta) == np.sign(ma - mb) or abs(ma - mb) < 1e-9, (
            f"iso-compute sign convention inverted: mean_delta {r.mean_delta:+.4f} vs "
            f"arm means {ma:.4f} - {mb:.4f}")
    return (f"{len(trained)} costed iterations across {trained.arm.nunique()} arms; "
            f"phases sum, cum monotone; {pairs_checked} budget matches, {contrast_rows} "
            f"contrast rows persona-paired with the sign convention pinned")


def _c_render_freshness() -> str:
    """Every judge's rendered subtree must be newer than that judge's newest score.

    The silent failure this exists for: ``render_views.py`` with no arguments renders the PRIMARY
    ORACLE ONLY. After a second judge's scores land, its ``<judge>/`` folders keep rendering fine —
    they just carry the *previous* grid, and nothing says so. That happened on 2026-08-18: the
    held-out judge's `k_paired_by_method.md` still held 1 iteration of GRPO_LA5 while the primary's
    held 5, and the only visible symptom was a row count nobody had reason to check.

    Compares, per (view, judge), the newest ``tables/**/<judge>/*.md`` mtime against the newest
    score CSV for that judge. SKIPs rather than fails when a tree has not been rendered at all
    (a fresh clone, or a view nobody renders) — the check is for *staleness*, not for absence.
    """
    _discover_or_skip()
    from eda_analysis.constants import EVAL_SCORES, JUDGE_PARTITION, PRIMARY_JUDGE_TAG, judge_label
    results_root = os.path.join(_EDA_DIR, "results")
    if not os.path.isdir(results_root):
        raise _Skip("no results/ tree")
    if not os.path.isdir(EVAL_SCORES):
        raise _Skip("score lake not readable")

    # newest score per judge tag (sample the parquet fold when present — walking every CSV is slow
    # over the Drive mount and the fold is rebuilt from them anyway)
    newest_score = {}
    for jd in os.listdir(EVAL_SCORES):
        if not jd.startswith(JUDGE_PARTITION):
            continue
        tag = jd[len(JUDGE_PARTITION):]
        root = os.path.join(EVAL_SCORES, jd)
        best = 0.0
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".csv"):
                    try:
                        best = max(best, os.path.getmtime(os.path.join(dirpath, fn)))
                    except OSError:
                        pass
            if best:
                break                      # one populated level is enough to date the partition
        if best:
            newest_score[tag] = best
    if not newest_score:
        raise _Skip("no judge partitions on disk")

    stale, checked = [], 0
    for view in sorted(os.listdir(results_root)):
        tdir = os.path.join(results_root, view, "tables")
        if not os.path.isdir(tdir):
            continue
        for tag, score_t in newest_score.items():
            label = judge_label(tag) if tag != PRIMARY_JUDGE_TAG else judge_label("")
            newest_art = 0.0
            for dirpath, _d, filenames in os.walk(tdir):
                if os.path.basename(dirpath) != label:
                    continue
                for fn in filenames:
                    if fn.endswith(".md"):
                        try:
                            newest_art = max(newest_art, os.path.getmtime(os.path.join(dirpath, fn)))
                        except OSError:
                            pass
            if not newest_art:
                continue                   # this judge has never been rendered into this view
            checked += 1
            if newest_art < score_t:
                stale.append(f"{view}/{label} (rendered {int((score_t - newest_art) / 3600)}h "
                             f"before its newest score)")
    if not checked:
        raise _Skip("no rendered judge subtrees to date")
    assert not stale, (
        "STALE judge subtree(s): " + "; ".join(stale) +
        " — a bare `render_views.py` renders the primary oracle ONLY. Re-render with "
        "`python tools/render_views.py --all-judges`.")
    return f"{checked} (view, judge) subtrees all newer than their scores"


def _c_scores_and_means() -> str:
    arms = _discover_or_skip()
    s = E.load_scores_long(arms)
    assert not s.empty, "load_scores_long empty"
    assert "Q1Q2" in set(s.questionnaire.unique()), "Q1Q2 composite missing"
    q = s[s.questionnaire == "Q1Q2"]
    checked = []
    for arm, expected in _KNOWN_Q1Q2_FINAL.items():
        a = q[q.arm == arm]
        if a.empty:
            continue
        fin = int(a.iteration.max())
        got = float(a[a.iteration == fin].score.mean())
        assert abs(got - expected) <= _KNOWN_TOL, (
            f"{arm} final(iter {fin}) Q1Q2={got:.3f} != {expected}±{_KNOWN_TOL}")
        checked.append(f"{arm}@{fin}={got:.2f}")
    assert checked, "no known-mean arm present to verify"
    return "known means reproduce: " + ", ".join(checked)


def _c_persona_permutation() -> str:
    arms = _discover_or_skip()
    from eda_analysis.data import persona_order
    n_ok = 0
    for a in arms:
        n = a.n_personas or 96
        for k in a.iters:
            order = persona_order(a.seed, k, n)
            assert sorted(order) == list(range(n)), (
                f"{a.label} model_iter_{k}: persona recovery not a 0..{n-1} permutation")
            n_ok += 1
    return f"persona order is an exact permutation for {n_ok} (arm,iter) pairs"


def _c_rubric_parity() -> str:
    """The gate that must hold before ANY second-judge spend: the Claude-encoded schema must ask
    the same rubric as the OpenAI one. Structural + free, so it runs on every self-check rather
    than only when someone remembers to look."""
    from eda_analysis.scoring import judge_plan as jp
    tab = jp.check_rubric_parity()
    bad = tab[~tab.parity_ok]
    assert bad.empty, ("rubric parity FAILED — a second-judge sweep would measure a different "
                       "rubric: " + "; ".join(f"{r.metric}: {r.problems}" for r in bad.itertuples()))
    return (f"{len(tab)} rubrics parity-clean "
            f"({int(tab.arrays_pinned.sum())} pinned arrays, "
            f"{int(tab.numeric_bounded.sum())} bounded fields restated in prose)")


def _c_judge_dimension() -> str:
    """The JUDGE axis routes BOTH reads and writes, and always restores the primary.

    Guards the two ways this can go wrong silently: a score read that still points at the primary
    ``eval_scores/`` tree (so a 'Claude' figure is really gpt's numbers), and an export that lands
    in the primary results root (overwriting the thesis artifacts with another grader's).
    """
    from eda_analysis import constants as K, exports as E
    from eda_analysis.data import discover_arms
    from eda_analysis import reliability as rel
    tags = rel.judge_tags()
    second = [t for t in tags if t != "openai_gpt-4o-mini-2024-07-18"]
    if not second:
        raise _Skip("no second judge on disk")
    arms = _discover_or_skip()
    arm = arms[0]
    k = sorted(arm.iters)[-1]
    try:
        assert K.active_judge() == "", "a previous test leaked an active judge"
        E.set_view("L0")
        E.set_export_group("1_outcomes")
        primary_dir = arm.eval_dir(k, "Q1")
        primary_fig = E._fig_dir()

        K.set_active_judge(second[0], 0)
        judge_dir = arm.eval_dir(k, "Q1")
        judge_fig = E._fig_dir()

        from eda_analysis.constants import (EVAL_SCORES, judge_partition_dir, judge_dirname)
        # Score side: ONE lake, every grader an equal `judge=` partition of it (2026-07-28).
        # Before that the primary lived in a per-method tree and only second judges were
        # partitioned, so the resolver carried a primary-vs-other branch and the primary's
        # scores were split across two roots at once.
        assert judge_dir.startswith(judge_partition_dir(second[0])), \
            f"judge score dir did not route to the judge= partition: {judge_dir}"
        assert primary_dir.startswith(judge_partition_dir("")), \
            f"primary score dir did not route to its own judge= partition: {primary_dir}"
        assert os.path.dirname(judge_partition_dir("")) == EVAL_SCORES, \
            "judge partitions must sit directly under the lake root"
        assert "rep=" in primary_dir and "rep=" in judge_dir, \
            f"score dirs must carry a rep= partition: {primary_dir} | {judge_dir}"
        assert "grpo_Exp3" not in primary_dir and "pto_Exp3" not in primary_dir, \
            f"primary score dir must not be method-scoped any more: {primary_dir}"
        # layout: results/<view>/figures/<group>/<judge>/ — the judge is the DEEPEST level and
        # EVERY grader gets one, primary included (2026-07-28), so a figure path always names the
        # grader that produced it. The two must be SIBLINGS, never nested one inside the other.
        p_label, j_label = judge_dirname(""), judge_dirname(second[0])
        assert os.path.basename(primary_fig) == p_label, \
            f"primary figures must nest under its own judge label {p_label!r}: {primary_fig}"
        assert os.path.basename(judge_fig) == j_label, \
            f"judge figures must nest under {j_label!r}: {judge_fig}"
        assert os.path.dirname(primary_fig) == os.path.dirname(judge_fig), \
            f"judges must be siblings under one group dir: {primary_fig} vs {judge_fig}"
        assert p_label != j_label, "two judges collapsed to the same folder label"
    finally:
        K.set_active_judge("")
        E.set_export_group("")
    assert K.active_judge() == "", "active judge not restored"
    return (f"reads -> eval_scores/judge={{{p_label},{j_label}}}/rep=N; "
            f"writes -> <group>/{{{p_label},{j_label}}}/ (every judge nests, none is flat)")


def _c_score_fold() -> str:
    """The parquet fold serves the SAME rows as the CSVs, and a stale signature is refused.

    The fold is a second read path over the score lake, so the risk it carries is silent drift: a
    figure rendered off scores that are no longer on disk. That risk is entirely borne by the
    staleness guard, which is what this asserts — equivalence AND that tampering with the recorded
    signature makes the read path fall back instead of serving anything.
    """
    from eda_analysis import score_archive as A
    from eda_analysis.data import iter_conv_rows
    arms = _discover_or_skip()
    arm = arms[0]
    k = sorted(arm.iters)[-1]
    ddir = arm.eval_dir(k, "Q1")
    parsed = A.parse_eval_dir(ddir)
    assert parsed is not None, f"lake path not parseable: {ddir}"
    assert A.parse_eval_dir(os.path.join(os.sep, "not", "the", "lake")) is None, \
        "a path outside the lake must not parse as a partition"

    A.reset_cache()
    served = A.rows_for(ddir)
    if served is None:
        raise _Skip("no parquet fold on disk (run tools/consolidate_scores.py build)")
    fold = {fi: r for fi, r in served}

    # Force the CSV path and compare.
    real, A.rows_for = A.rows_for, lambda d: None
    try:
        csv = {fi: r for fi, r in iter_conv_rows(ddir)}
    finally:
        A.rows_for = real
    assert set(fold) == set(csv), \
        f"fold/CSV disagree on which conversations exist ({len(fold)} vs {len(csv)})"
    for fi in sorted(fold):
        a, b = fold[fi], csv[fi]
        assert list(a.index) == list(b.index), f"column mismatch at {fi}: {list(a.index)} vs {list(b.index)}"
        for c in a.index:
            assert float(a[c]) == float(b[c]), f"value mismatch at {fi}.{c}: {a[c]} vs {b[c]}"

    # Tamper with the recorded signature -> the guard must refuse to serve.
    A.reset_cache()
    key = A.partition_key(parsed[0], parsed[1], parsed[2])
    A._manifest()[key] = "deliberately-wrong-signature"
    A._frame_cache.clear()
    assert A.rows_for(ddir) is None, "a stale signature was served instead of falling back to CSVs"
    A.reset_cache()
    return f"fold == CSVs on {len(fold)} convs; stale signature correctly refused"


def _c_multi_judge() -> str:
    """Multi-judge analysis runs end-to-end on whatever is on disk, and the variance components
    stay in-range. Guards the arithmetic in reliability.variance_components_* / gain_retention."""
    from eda_analysis import reliability as rel
    tags = rel.second_judge_tags()
    if not tags:
        raise _Skip("no second-judge scores on disk")
    jl = rel.load_judge_long(tags[0])
    if jl.empty:
        raise _Skip("second-judge tree is empty")
    metrics = sorted(jl.metric.unique())
    models = sorted(jl.model.unique())
    pl = rel.load_primary_long(models, metrics)
    if pl.empty:
        raise _Skip("no matching primary scores")
    # Match the notebook: analyse only fully-scored cells, so a partially-landed sweep can never
    # make this check pass on a grid the published tables would refuse.
    n_cells = len(rel.coverage_table(jl))
    jl, pl = rel.filter_complete_cells(jl, pl, verbose=False)
    if jl.empty:
        raise _Skip("no fully-scored second-judge cells")
    metrics = sorted(jl.metric.unique())
    cc = rel.variance_components_conversation(jl, pl)
    va = rel.variance_components_arm(jl, pl, conv_components=cc)
    assert not va.empty, "variance_components_arm returned nothing"
    shares = va[["pct_arm", "pct_judge", "pct_arm_x_judge"]].sum(axis=1)
    assert ((shares - 100).abs() < 0.5).all(), f"variance shares do not sum to 100%: {list(shares)}"
    for col in ("dependability_k1", "dependability_k2"):
        assert va[col].between(0, 1).all(), f"{col} outside [0,1]: {list(va[col])}"
    assert (va.dependability_k2 >= va.dependability_k1 - 1e-9).all(), \
        "averaging two judges cannot LOWER dependability — check the G-coefficient formula"
    pairs = rel.all_pairs_contrasts(jl, pl, n_boot=200)
    n_complete = len(rel.coverage_table(jl))
    return (f"{n_complete}/{n_cells} cells complete; {len(metrics)} metrics; variance shares sum "
            f"to 100%; {int(pairs.same_sign.sum())}/{len(pairs)} pairwise contrasts keep their sign")


def _c_cross_judge_layout() -> str:
    """A CROSS-JUDGE artifact must never sit under a single judge's folder.

    The ``<judge>/`` path segment asserts "this grader produced this file". For a multi-judge
    artifact that is false — ``multijudge_variance_decomposition.png`` plots BOTH graders, and
    filing it under ``gpt-4o-mini/`` credits the primary with the very figure that proves the two
    agree. That is exactly where these lived until 2026-07-29 (inside the training-side family 5,
    which refuses a second judge, so they could only ever be written under the primary).

    Structural, not data-dependent: it walks the committed results trees.
    """
    import re as _re
    from eda_analysis import exports as X

    # Asserted as a PATH SHAPE, not against a list of judge names: a cross-judge artifact must sit
    # directly in a judge-invariant family, with no segment of any kind between family and file.
    # That catches a <judge>/ level without needing to know what the graders are called.
    cross = _re.compile(r"^(multijudge_|second_judge_|oracle_repeatability_|judge_)")
    offenders, seen = [], 0
    for view in ("L0", "L5"):
        for kind in ("figures", "tables"):
            root = os.path.join(X.RESULTS_DIR, view, kind)
            if not os.path.isdir(root):
                continue
            for dirpath, _dirs, files in os.walk(root):
                rel = os.path.relpath(dirpath, root)
                parts = [] if rel == "." else rel.split(os.sep)
                for f in files:
                    if not cross.match(f):
                        continue
                    seen += 1
                    if len(parts) != 1 or parts[0] not in X.JUDGE_INVARIANT_GROUPS:
                        offenders.append(os.path.join(view, kind, rel, f))
    assert not offenders, ("cross-judge artifacts must live directly in a JUDGE_INVARIANT_GROUPS "
                           f"family, never under a <judge>/ level: {sorted(offenders)[:6]}")
    assert "8_measurement" in X.JUDGE_INVARIANT_GROUPS, "family 8 must be judge-invariant"
    assert X._leaf("root", "8_measurement") == os.path.join("root", "8_measurement"), \
        "_leaf still appends a <judge> segment to a judge-invariant family"
    return f"{seen} cross-judge artifacts, none under a <judge>/ dir; family 8 judge-invariant"


# ── probe (opt-in, heavy) ─────────────────────────────────────────────────────
def _c_probe() -> str:
    arms = _discover_or_skip()
    pto = [a for a in arms if a.method == "PTO"]
    if not pto:
        raise _Skip("no PTO arm for the preference probe")
    try:
        from eda_analysis import training, pref
    except Exception as e:                                          # noqa: BLE001
        raise _Skip(f"probe deps unavailable: {e}")
    arm = pto[0]
    try:
        pairs = training.load_pref_pairs([arm])
        if pairs is None or pairs.empty:
            raise _Skip("no pref pairs on disk for the PTO arm")
        emb = pref.embed_pairs(pairs)
        directions = pref.preference_direction_by_iter(emb)
        pq = pref.probe_quality_by_iter(emb, directions)
    except _Skip:
        raise
    except Exception as e:                                          # noqa: BLE001
        raise _Skip(f"probe could not run ({type(e).__name__}: {e})")
    assert not pq.empty, "probe produced no rows"
    overall = float(pq["wins_correct"].mean())
    assert overall > 0.5, f"probe wins_correct={overall:.3f} not > 0.5 (direction doesn't separate)"
    return f"probe wins_correct mean={overall:.3f} (>0.5) over {len(pq)} iters"


# ── driver ────────────────────────────────────────────────────────────────────
def _c_seeded_bootstrap() -> str:
    """Every seaborn call that bootstraps a CI must pass seed=BOOT_SEED.

    lineplot/barplot/pointplot default to errorbar=("ci", 95) — a 1000-resample
    bootstrap — so a callsite that never NAMES errorbar still draws a randomised CI. The
    2026-07-28 reproducibility pass seeded the eight callsites that spelled errorbar out and
    could not see the ones relying on the default, which is why 4_heterogeneity still rewrote
    20 PNGs per view on unchanged data (found 2026-08-13 while proving a render race was benign:
    two consecutive clean renders of the same notebook disagreed).

    Source-level on purpose. The data-level symptom is "some PNGs differ between two identical
    renders", which is slow to notice, easy to blame on the data, and exactly what a thesis figure
    must not do.
    """
    import re as _re
    pkg = os.path.dirname(os.path.abspath(__file__))
    call = _re.compile(r"\bsns\.(lineplot|barplot|pointplot)\s*\(")
    offenders, seen = [], 0
    for root, _dirs, files in os.walk(pkg):
        if "__pycache__" in root:
            continue
        for fn in sorted(f for f in files if f.endswith(".py")):
            fp = os.path.join(root, fn)
            lines = io.open(fp, encoding="utf-8").read().splitlines()
            for i, line in enumerate(lines):
                if not call.search(line):
                    continue
                if line.lstrip().startswith(("#", '"', "'")) or "..." in line:
                    continue                      # docstring example, not a callsite
                block = "\n".join(lines[i:i + 6])
                seen += 1
                if any(a in block for a in ('errorbar=None', 'errorbar="se"', "errorbar='se'")):
                    continue                      # analytic: no bootstrap, no seed needed
                if "seed=" not in block:
                    offenders.append(f"{os.path.relpath(fp, pkg)}:{i + 1}")
    if offenders:
        raise AssertionError(
            f"{len(offenders)} seaborn callsite(s) bootstrap a CI without seed=BOOT_SEED "
            f"(non-reproducible figures): {', '.join(offenders[:6])}")
    return f"{seen} seaborn CI callsites, all seeded or analytic"


def _c_role_bindings() -> str:
    """Role-binding names round-trip, and default-bound runs keep their historical names.

    ``roles.binding_suffix`` widens an arm's identity when the patient/oracle model is not the
    Exp3 default. Three properties have to hold or the score lake corrupts quietly:

    1. **Default ⇒ empty suffix.** Every run to date used gpt-4o-mini for both roles; if the
       suffix stopped being empty for them, all ~50k CSVs would be orphaned under new names.
    2. **Round-trip.** A name built with a suffix must parse back to the same tags, so the
       reader (``Arm.model_name``) reconstructs the writer's folder exactly.
    3. **No ``_PT`` collision.** PTO names end in ``_PT{greedy|indep}``; the patient prefix is
       ``_Pat`` precisely so ``_PTgreedy`` is not read as patient tag "Tgreedy".
    """
    from roles import (DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL, binding_suffix,
                       suffix_from_tags, assert_name_matches_roles, model_tag)
    from .data import parse_experiment_name

    assert binding_suffix(DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL) == "", \
        "default role models must produce an EMPTY suffix (else existing arms are renamed)"
    assert binding_suffix(None, None) == ""

    gem = "google/gemma-3n-E4B-it"
    tag = model_tag(gem)
    assert tag.isalnum(), f"model tag {tag!r} must be alphanumeric for the arm-name regex"
    assert binding_suffix(gem) == f"_O{tag}"
    assert binding_suffix(None, gem) == f"_Pat{tag}"

    legacy = "PTO_Iterative_Q1Q2_Llama32-1B_LA5_MCL12_M8_PTgreedy"
    p = parse_experiment_name(legacy)
    assert p and p["oracle_tag"] is None and p["patient_tag"] is None, \
        f"legacy PTO name must parse with NO bindings, got {p}"
    assert p["mode"] == "greedy", "the _PTgreedy mode token must survive the binding groups"

    for suffix, exp_o, exp_p in ((f"_O{tag}", tag, None),
                                 (f"_Pat{tag}", None, tag),
                                 (f"_O{tag}_Pat{tag}", tag, tag)):
        q = parse_experiment_name(legacy + suffix)
        assert q, f"name with suffix {suffix!r} failed to parse"
        assert (q["oracle_tag"], q["patient_tag"]) == (exp_o, exp_p), \
            f"{suffix!r} round-trip gave {(q['oracle_tag'], q['patient_tag'])}"
        assert suffix_from_tags(q["oracle_tag"], q["patient_tag"]) == suffix

    grpo = "GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8"
    assert parse_experiment_name(grpo)["oracle_tag"] is None
    assert parse_experiment_name(grpo + f"_O{tag}")["oracle_tag"] == tag

    assert_name_matches_roles(legacy, DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL)
    assert_name_matches_roles(legacy + f"_O{tag}", gem, DEFAULT_PATIENT_MODEL)
    try:
        assert_name_matches_roles(legacy, gem, DEFAULT_PATIENT_MODEL)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-default oracle with an unsuffixed name must RAISE")
    return f"default⇒'', round-trip ok, _PT collision guarded (tag {tag!r})"


def _c_arm_identity_unique() -> str:
    """No two discovered arms may share a score-lake folder, and writer must equal reader.

    ``Arm.model_name`` (read side) names where the EDA looks for scores; the scoring
    registry's ``Experiment.model_name`` (write side) names where Run_Eval puts them. They
    are now derived once and carried, but this asserts it — a divergence would make
    Run_Eval's skip-existing resume compare against another arm's CSVs.
    """
    arms = _discover_or_skip()
    seen = {}
    for a in arms:
        for k in a.iters:
            name = a.model_name(k)
            if name in seen:
                raise AssertionError(
                    f"model_name collision: {name!r} claimed by both {seen[name]!r} and "
                    f"{a.exp_name!r} — their scores would share one eval_scores folder")
            seen[name] = a.exp_name

    # The module-level registry, NOT a fresh build_experiments_from_disk(): re-running discovery
    # costs a second full walk of the Drive-streamed data dirs (minutes when Drive is cold), and
    # EXPERIMENTS is what Run_Eval actually writes from, so checking it is the more faithful test.
    from .scoring.registry import EXPERIMENTS
    by_path = {e.path: e.model_name for e in EXPERIMENTS}
    import os as _os
    from .constants import WORKSPACE_ROOT
    n = 0
    for a in arms:
        for k in a.iters:
            rel = _os.path.relpath(a.conv_dirs[k], WORKSPACE_ROOT)
            if rel in by_path:
                assert by_path[rel] == a.model_name(k), (
                    f"write/read model_name disagree for {rel}: registry says "
                    f"{by_path[rel]!r}, Arm says {a.model_name(k)!r}")
                n += 1
    return f"{len(seen)} unique model names, {n} write/read pairs agree"


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    fast = "--fast" in argv
    probe = "--probe" in argv

    # Validate GROUND TRUTH: the data checks below bypass the parquet cache so a stale cache can
    # never mask a real data regression (the cache mechanism check manages this env var itself).
    os.environ["EDA_NO_CACHE"] = "1"

    results: _Results = []
    # Structural — always.
    _run("import + __all__ resolve", _c_all_resolves, results)
    _run("view->ks map", _c_view_map, results)
    _run("EdaConfig round-trip", _c_config_roundtrip, results)
    _run("live aliases (figures/plots)", _c_live_aliases, results)
    _run("scoring subpackage surface", _c_scoring_surface, results)
    _run("notebook symbol refs resolve", _c_notebook_refs_resolve, results)
    _run("cache mechanism + invalidation", _c_cache_mechanism, results)
    _run("rubric parity (2nd judge gate)", _c_rubric_parity, results)
    _run("cross-judge artifact layout", _c_cross_judge_layout, results)
    _run("seeded bootstrap (repro figures)", _c_seeded_bootstrap, results)
    _run("role bindings + name suffix", _c_role_bindings, results)
    # Data — unless --fast.
    if not fast:
        _run("discover_arms", _c_discover, results)
        _run("arm identity is collision-free", _c_arm_identity_unique, results)
        _run("scores_long + known means", _c_scores_and_means, results)
        _run("cross-K frame (RQ-i)", _c_cross_k, results)
        _run("compute axis (GPU-hours)", _c_compute_axis, results)
        _run("render freshness (per judge)", _c_render_freshness, results)
        _run("update probe (both methods)", _c_update_probe, results)
        _run("persona permutation", _c_persona_permutation, results)
        _run("judge dimension routing", _c_judge_dimension, results)
        _run("score fold (parquet read path)", _c_score_fold, results)
        _run("multi-judge analysis", _c_multi_judge, results)
    if probe:
        _run("PTO preference probe", _c_probe, results)

    width = max(len(n) for n, _, _ in results)
    print("\n eda_analysis self-check")
    print(" " + "-" * (width + 30))
    for name, status, detail in results:
        mark = {"PASS": "OK  ", "SKIP": "skip", "FAIL": "FAIL"}[status]
        print(f"  [{mark}] {name.ljust(width)}  {detail}")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(" " + "-" * (width + 30))
    print(f"  {n_pass} passed, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
