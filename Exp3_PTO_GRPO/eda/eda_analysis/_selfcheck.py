"""
_selfcheck.py — a fast regression guard for the ``eda_analysis`` package.

Run it after ANY refactor of the EDA package (module splits, alias changes, plotting moves,
caching) to confirm the invariants the notebooks rely on still hold::

    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck          # full (structural + data)
    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck --fast   # structural only (no disk reads)
    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck --probe  # + the heavy probes (pref probe, tail audit anchor)

It is deliberately dependency-light and self-contained: no notebook execution, no torch/trl, no
OpenAI. Data checks are SKIPPED (not failed) when the Exp3 eval data isn't readable locally, so the
structural half still guards a machine without the Drive mount. A check may also end in WARN — a
known, expected gap (e.g. a notebook a later phase of the 2026-08-18 reorg has not landed yet, or
an anchor whose module is not importable) — which is reported but does not fail the run.

Checks (23; 12 structural + 11 data, + 1 opt-in probe)
------------------------------------------------------
Structural (always, no disk):
  * ``import + __all__ resolve`` — package imports; every ``__all__`` name resolves.
  * ``family map`` — every ``config.FAMILIES`` entry ↔ ``notebooks/<top>/<sub>.ipynb`` (missing
    notebooks WARN until Phase C lands); ``PER_JUDGE_TOPS ⊂ FAMILIES``; ``split_family`` rejects junk.
  * ``EdaConfig round-trip`` — ``as_dict`` / ``with_`` / ``family`` validation.
  * ``live aliases (figures/plots)`` — the ``figures``/``plots`` → ``plotting`` aliases.
  * ``scoring subpackage surface`` — the Run_Eval + Judge_Reliability backend keeps its names.
  * ``notebook symbol refs resolve`` — every ``<submodule>.<attr>`` a notebook calls exists.
  * ``cache mechanism + invalidation`` — miss→build, hit→read, bypass, content-change invalidates.
  * ``rubric parity (2nd judge gate)`` — the Claude-encoded rubric asks the same as the OpenAI one.
  * ``cross-judge artifact layout`` — ``arms/*`` nests a ``<judge>/`` leaf; every other family is
    judge-invariant and no cross-judge artifact sits under a ``<judge>/`` folder.
  * ``exports routing (family root)`` — ``save_*`` refuse without a family; leaves compose
    ``<family>/{figures,tables}/[<judge>/][<group>/]``; ``PRESERVE`` guards; xlsx timestamps frozen.
  * ``seeded bootstrap (repro figures)`` — every seaborn CI callsite passes ``seed=BOOT_SEED``.
  * ``role bindings + name suffix`` — patient/oracle role-binding suffixes round-trip.
Data (skipped if data absent):
  * ``discover_arms`` — the LA0 arms are found.
  * ``arm identity is collision-free`` — no two arms share a score-lake folder; writer == reader.
  * ``scores_long + known means`` — Q1Q2 present; PTO_LA0 final ≈ 4.26, GRPO_LA0 final ≈ 3.75.
  * ``cross-K frame`` — ``cross_k_scores`` widens-or-equals the K set and never moves the export root.
  * ``compute axis (GPU-hours)`` — every trained iteration costed; phases sum; iso-compute pairs on
    persona with the ``stats.py`` sign convention.
  * ``paper fixture anchors`` — the paper's frozen numbers
    (``papers/2026_lookahead_pto_grpo/analysis/out/*.json``, kept as a fixture) match the promoted
    modules: PTO Q1Q2 iter 6 K contrast +0.257 / dz 0.417 via ``stats.paired_arrays``; GRPO
    per-step K5/K0 ratio at iters 3–5 = 1.965/1.962/1.911 (``compute.step_multiplier``); PTO
    iter-1 ``margin_mean`` PTO_LA5 0.424 + PTO_LA0 0.274 (``dispersion.dispersion_by_iter``);
    ended_early GRPO_LA5 iter 5 first 300 groups 657/2400 + its realized-turn histogram
    (``tails.SCOUT_EXPECTED``; read cheaply from ONE ``generations.jsonl`` in the default pass,
    ``tails.tail_audit_frames`` itself under ``--probe``). Hard checks — the modules have landed.
  * ``render freshness (per judge)`` — every rendered ``arms/*/tables/<judge>/`` leaf is newer than
    that judge's newest score; invariant families newer than the newest score of ANY judge. An
    unrendered results tree WARNs.
  * ``update probe (both methods)`` — the cross-method preference probe: one weight scale, real
    groups only, the right iteration join, counterfactual re-weighting keeps scale.
  * ``persona permutation`` — persona recovery is an exact 0..n-1 permutation per (arm, iter).
  * ``judge dimension routing`` — the JUDGE axis routes reads AND writes and restores the primary.
  * ``score fold (parquet read path)`` — the fold equals the CSVs and refuses a tampered signature.
  * ``multi-judge analysis`` — variance components in range; sign preservation runs end-to-end.
Probe (opt-in, heavy — needs sentence-transformers + pref pairs):
  * ``PTO preference probe`` — Mass-Mean-Probe ``wins_correct`` > 0.5.
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

# Submodule names a notebook may qualify a call with (live modules + the figures/plots aliases +
# the modules the 2026-08-18 reorg promoted from the paper generators).
_SUBMODULES = ("plotting", "plots", "figures", "data",
               "stats", "behavior", "training", "pref", "exports", "compute", "reliability",
               "lookahead", "transfer", "tails", "dispersion", "faithfulness", "crossgen",
               "replication", "instruments")


# ── check harness ─────────────────────────────────────────────────────────────
class _Skip(Exception):
    """Raised by a check to mark itself SKIPPED (e.g. data absent) rather than FAILED."""


class _Warn(Exception):
    """Raised by a check to mark itself WARN — a known, expected gap that is reported but does not
    fail the run (a notebook a later reorg phase has not landed yet; an anchor whose module is not
    importable). Distinct from SKIP (nothing to check) and FAIL (an invariant broke)."""


_Results = List[Tuple[str, str, str]]   # (name, status, detail)


def _run(name: str, fn: Callable[[], str], results: _Results) -> None:
    try:
        detail = fn() or ""
        results.append((name, "PASS", detail))
    except _Skip as s:
        results.append((name, "SKIP", str(s)))
    except _Warn as w:
        results.append((name, "WARN", str(w)))
    except Exception as e:                                          # noqa: BLE001
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        if os.environ.get("SELFCHECK_TRACE"):
            traceback.print_exc()


# ── structural checks ─────────────────────────────────────────────────────────
def _c_all_resolves() -> str:
    missing = [n for n in E.__all__ if not hasattr(E, n)]
    assert not missing, f"__all__ names not resolvable on package: {missing}"
    return f"{len(E.__all__)} __all__ names resolve"


def _c_family_map() -> str:
    """``config.FAMILIES`` <-> ``notebooks/<top>/<sub>.ipynb`` are 1:1, and the per-judge set is sane.

    The results tree is organised by family (2026-08-18 reorg), and ``tools/render_results.py``
    executes ``notebooks/<top>/<sub>.ipynb`` for every entry — a family with no notebook renders
    nothing and a notebook with no family writes nowhere the index knows. Missing notebooks are a
    WARN (expected until Phase C of the reorg lands the new notebooks); a notebook under a
    ``<top>/`` folder that names NO family, or a ``PER_JUDGE_TOPS`` entry that is not a top, FAILS.
    ``scoring/`` (Run_Eval etc.) is not a family.
    """
    from eda_analysis import config as C
    assert C.FAMILIES and all(subs for subs in C.FAMILIES.values()), "FAMILIES must be non-empty"
    assert set(C.PER_JUDGE_TOPS) <= set(C.FAMILIES),         f"PER_JUDGE_TOPS {sorted(C.PER_JUDGE_TOPS)} not a subset of FAMILIES tops"
    assert "arms" in C.PER_JUDGE_TOPS, "arms/* must be per-judge (one leaf per grader)"
    assert "measurement" not in C.PER_JUDGE_TOPS, "measurement/* must be judge-invariant"
    fams = C.all_families()
    assert len(fams) == len(set(fams)), "duplicate family"
    for f in fams:
        top, sub = C.split_family(f)
        assert f"{top}/{sub}" == f
        assert C.is_per_judge(f) == (top in C.PER_JUDGE_TOPS)
    for junk in ("", "arms", "arms/", "arms/nope", "nope/outcomes", "a/b/c"):
        try:
            C.split_family(junk)
        except ValueError:
            pass
        else:
            raise AssertionError(f"split_family accepted junk family {junk!r}")

    nb_root = os.path.join(_EDA_DIR, "notebooks")
    missing = [f for f in fams if not os.path.isfile(os.path.join(nb_root, *f.split("/")) + ".ipynb")]
    orphans = []
    for top in sorted(os.listdir(nb_root)) if os.path.isdir(nb_root) else []:
        if top in ("scoring", "analysis") or not os.path.isdir(os.path.join(nb_root, top)):
            continue
        for fn in sorted(os.listdir(os.path.join(nb_root, top))):
            if fn.endswith(".ipynb") and f"{top}/{fn[:-6]}" not in fams:
                orphans.append(f"{top}/{fn}")
    assert not orphans, f"notebooks under notebooks/<top>/ that name no family: {orphans}"
    if missing:
        raise _Warn(f"{len(fams)} families, {len(fams) - len(missing)} notebooks present; "
                    f"MISSING (expected until the reorg's Phase C notebooks land): {missing}")
    return f"{len(fams)} families <-> notebooks 1:1; per-judge tops {sorted(C.PER_JUDGE_TOPS)}"


def _c_config_roundtrip() -> str:
    cfg = E.EdaConfig(family="arms/outcomes", selection="best")
    d = cfg.as_dict()
    assert d["family"] == "arms/outcomes" and d["selection"] == "best"
    assert "view" not in d and "export_group" not in d, "retired knobs leaked into as_dict"
    cfg2 = cfg.with_(selection="all")
    assert cfg2.selection == "all" and cfg.selection == "best", "with_ must not mutate original"
    assert E.EdaConfig().family == "" and E.EdaConfig().ks is None,         "default config must be every arm with no family (exports disabled)"
    for bad in ("L0", "1_outcomes", "arms/nope"):
        try:
            E.notebook_setup(E.EdaConfig(family=bad, verbose=False))
        except ValueError:
            pass
        else:
            raise AssertionError(f"notebook_setup accepted unknown family {bad!r}")
    return "EdaConfig.as_dict/with_ OK; unknown family rejected"


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
    """Scan committed notebooks for ``<submodule>.<attr>(`` calls -> {submodule: {attr, ...}}.

    Scans every live notebook: ``notebooks/<top>/`` for each ``config.FAMILIES`` top, plus
    ``notebooks/scoring/``. (The pre-reorg ``notebooks/analysis/`` folder and its legacy scan were
    removed with the 2026-08-18 reorg; they live in the 2026-08-19 archival bundle.)
    """
    from eda_analysis.config import FAMILIES
    # ``(?<![\w.])`` — the submodule name must not itself be an attribute: ``plotting.lookahead.k_did``
    # is a ``plotting`` ref (attr ``lookahead``), NOT a ``lookahead`` ref (the analysis module has no
    # ``k_did``). Same-named analysis + plotting modules exist for every promoted topic.
    pat = re.compile(r"(?<![\w.])(" + "|".join(_SUBMODULES) + r")\.([A-Za-z_][A-Za-z0-9_]*)")
    refs: dict = {m: set() for m in _SUBMODULES}
    tops = list(FAMILIES) + ["scoring"]
    for top in tops:
        for nb in glob(os.path.join(_EDA_DIR, "notebooks", top, "*.ipynb")):
            d = json.load(open(nb, encoding="utf-8"))
            for cell in d.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                # code only — a ``# see reliability.py`` comment is prose, not a symbol ref
                src = "\n".join(re.sub(r"#.*$", "", ln) for ln in "".join(cell.get("source", [])).splitlines())
                # ...and string literals are data, not symbol refs: save_fig(fig, "crossgen.png") or a
                # ledger source "transfer.xlsx" would otherwise read as crossgen.png / transfer.xlsx.
                src = re.sub(r"(\"[^\"\n]*\"|'[^'\n]*')", "''", src)
                for mod, attr in pat.findall(src):
                    refs[mod].add(attr)
    return refs


def _c_notebook_refs_resolve() -> str:
    def _resolve(refs):
        bad, total = [], 0
        for mod, attrs in refs.items():
            submod = getattr(E, mod, None)
            for attr in attrs:
                total += 1
                if submod is None or not hasattr(submod, attr):
                    bad.append(f"{mod}.{attr}")
        return sorted(bad), total
    refs = _notebook_symbol_refs()
    bad, total = _resolve(refs)
    assert not bad, f"notebook-referenced symbols not resolvable: {bad}"
    used = {m: len(a) for m, a in refs.items() if a}
    return f"{total} live notebook symbol refs resolve across {used}"


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
def _setup_quiet(cfg):
    """``notebook_setup`` for a check, without leaving a phantom ``_provenance.md`` behind.

    ``notebook_setup`` writes the family's provenance banner as a side effect; for a family
    nobody has rendered yet that would create ``results/<family>/figures/_provenance.md`` alone
    in an otherwise empty tree. Remove it (and the empty dirs up to the top) when this call
    created it; leave a pre-existing banner (a rendered family) untouched.
    """
    from eda_analysis import exports as X
    S = E.notebook_setup(cfg)
    fam = X.active_family()
    if not fam:
        return S
    prov = os.path.join(X._fig_dir(None), "_provenance.md")
    figs = X._fig_dir(None)
    only_banner = (os.path.isdir(figs) and os.listdir(figs) == ["_provenance.md"])
    if only_banner and not os.path.isdir(os.path.join(X.family_root(), "tables")):
        os.remove(prov)
        d = figs
        top = os.path.dirname(X.family_root())
        while d != top and os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
            d = os.path.dirname(d)
        if os.path.isdir(top) and not os.listdir(top):
            os.rmdir(top)
    return S


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
    """``cross_k_scores`` must widen-or-equal the READ without moving the WRITE.

    Since the 2026-08-18 reorg the default arm filter is every arm, so on a default config the
    cross-K frame EQUALS ``S.SCORES`` (asserted); on a config that narrowed ``ks`` it must WIDEN
    back to both K arms (asserted with ``ks=[5]``). In neither case may it touch the export root —
    the tempting wrong fix for a K contrast is to relax the arm filter globally, which would
    silently re-point every other artifact in the notebook. Also pins the sign convention shared by
    ``k_means_by_iter`` and ``paired_k_comparison`` (+ => K=0 higher), which the tables' captions
    promise.
    """
    _discover_or_skip()
    from eda_analysis import exports
    S = _setup_quiet(E.EdaConfig(family="lookahead/reward", verbose=False))
    if S.SCORES.empty:
        raise _Skip("no scores on disk")
    try:
        root_before = exports.family_root(), exports._fig_dir(None)
        ks_S = set(S.SCORES.K.unique())
        cross = E.cross_k_scores(S)
        root_after = exports.family_root(), exports._fig_dir(None)
        assert root_before == root_after, f"cross_k_scores moved the export root: {root_before} -> {root_after}"
        ks_cross = set(cross.K.unique())
        assert ks_S <= ks_cross, f"cross-K frame narrowed K: {sorted(ks_S)} -> {sorted(ks_cross)}"
        assert cross.shape == S.SCORES.shape, (
            f"default config (every arm): cross_k_scores should equal S.SCORES, got "
            f"{cross.shape} vs {S.SCORES.shape}")
        assert exports.active_family() == "lookahead/reward", "export family drifted"

        # A narrowed config must widen back to both K arms, still without moving the root.
        S5 = _setup_quiet(E.EdaConfig(family="lookahead/reward", ks=[5], verbose=False))
        ks_view = set(S5.SCORES.K.unique())
        cross5 = E.cross_k_scores(S5)
        assert ks_view == {5}, f"ks=[5] config should hold ONE K, holds {sorted(ks_view)}"
        assert ks_view < set(cross5.K.unique()), "cross-K frame did not widen a ks=[5] config"
        assert (exports.family_root(), exports._fig_dir(None)) == root_before, \
            "a narrowed config moved the export root"

        means = E.k_means_by_iter(cross, "PTO")
        paired = E.paired_k_comparison(cross, "PTO")
        assert not means.empty and not paired.empty, "PTO K contrast empty on the cross-K frame"
        # Same sign convention on both sides: where a cell is complete (96/96), the unpaired
        # difference of arm means IS the paired mean_delta — pairing buys precision, not a
        # different centre.
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
    finally:
        exports.set_family("")
    return (f"default K={sorted(map(int, ks_S))} == cross-K (same frame); ks=[5] -> cross-K "
            f"K={sorted(map(int, cross5.K.unique()))}; exports unmoved; {checked} matched iters agree on delta")


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

    S = _setup_quiet(E.EdaConfig(family="compute/cost", verbose=False))
    E.exports.set_family("")
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
    """Every rendered results leaf must be newer than the scores it was rendered from.

    The silent failure this exists for: a per-judge render that refreshes one grader's leaf and not
    the other's. After a second judge's scores land, its ``<judge>/`` folders keep rendering fine —
    they just carry the *previous* grid, and nothing says so. That happened on 2026-08-18: the
    held-out judge's `k_paired_by_method.md` still held 1 iteration of GRPO_LA5 while the primary's
    held 5, and the only visible symptom was a row count nobody had reason to check.

    New tree (2026-08-18 reorg): per-judge families ``results/<top>/<sub>/tables/<judge>/`` are
    compared against that judge's newest score; judge-invariant families
    ``results/<top>/<sub>/tables/`` (both graders inside) against the newest score of ANY judge.
    An empty / not-yet-rendered results tree WARNs (a fresh clone, or before the reorg's first
    render) — the check is for *staleness*, not for absence.
    """
    _discover_or_skip()
    from eda_analysis.constants import EVAL_SCORES, JUDGE_PARTITION, PRIMARY_JUDGE_TAG, judge_label
    from eda_analysis.config import FAMILIES, is_per_judge
    results_root = os.path.join(_EDA_DIR, "results")
    if not os.path.isdir(EVAL_SCORES):
        raise _Skip("score lake not readable")

    # newest score per judge tag (one populated level is enough to date the partition — walking
    # every CSV is slow over the Drive mount)
    newest_score = {}
    for jd in sorted(os.listdir(EVAL_SCORES)):
        if not jd.startswith(JUDGE_PARTITION):
            continue
        tag = jd[len(JUDGE_PARTITION):]
        best = 0.0
        for dirpath, _dirnames, filenames in os.walk(os.path.join(EVAL_SCORES, jd)):
            for fn in filenames:
                if fn.endswith(".csv"):
                    try:
                        best = max(best, os.path.getmtime(os.path.join(dirpath, fn)))
                    except OSError:
                        pass
            if best:
                break
        if best:
            newest_score[tag] = best
    if not newest_score:
        raise _Skip("no judge partitions on disk")
    newest_any = max(newest_score.values())

    def _newest_md(root: str) -> float:
        best = 0.0
        if not os.path.isdir(root):
            return best
        for dirpath, _d, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".md") and fn != "CAPTIONS.md":
                    try:
                        best = max(best, os.path.getmtime(os.path.join(dirpath, fn)))
                    except OSError:
                        pass
        return best

    stale, checked = [], 0
    for top, subs in FAMILIES.items():
        for sub in subs:
            tdir = os.path.join(results_root, top, sub, "tables")
            if not os.path.isdir(tdir):
                continue
            if is_per_judge(top):
                for tag, score_t in newest_score.items():
                    label = judge_label(tag) if tag != PRIMARY_JUDGE_TAG else judge_label("")
                    art_t = _newest_md(os.path.join(tdir, label))
                    if not art_t:
                        continue                   # this judge has never been rendered here
                    checked += 1
                    if art_t < score_t:
                        stale.append(f"{top}/{sub}/{label} (rendered "
                                     f"{int((score_t - art_t) / 3600)}h before its newest score)")
            else:
                art_t = _newest_md(tdir)
                if not art_t:
                    continue
                checked += 1
                if art_t < newest_any:
                    stale.append(f"{top}/{sub} (rendered {int((newest_any - art_t) / 3600)}h "
                                 f"before the newest score of any judge)")
    if not checked:
        raise _Warn("no rendered family leaves under results/<top>/<sub>/tables/ yet — expected "
                    "until the first `python tools/render_results.py` after the reorg")
    assert not stale, (
        "STALE results leaf/leaves: " + "; ".join(stale) +
        " — re-render with `python tools/render_results.py` (a bare run renders arms/* for EVERY "
        "judge on disk plus the judge-invariant tops).")
    return f"{checked} rendered leaves all newer than their scores"


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
    in the primary results leaf (overwriting the thesis artifacts with another grader's).
    """
    from eda_analysis import constants as K, exports as E
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
        E.set_family("arms/outcomes")
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
        # layout: results/arms/<sub>/figures/<judge>/ — for a per-judge family the judge is the
        # leaf directly under figures/, EVERY grader gets one, primary included (2026-07-28), so a
        # figure path always names the grader that produced it. The two must be SIBLINGS.
        p_label, j_label = judge_dirname(""), judge_dirname(second[0])
        assert os.path.basename(primary_fig) == p_label, \
            f"primary figures must nest under its own judge label {p_label!r}: {primary_fig}"
        assert os.path.basename(judge_fig) == j_label, \
            f"judge figures must nest under {j_label!r}: {judge_fig}"
        assert os.path.dirname(primary_fig) == os.path.dirname(judge_fig) == \
            os.path.join(E.family_root(), "figures"), \
            f"judges must be siblings directly under <family>/figures/: {primary_fig} vs {judge_fig}"
        assert p_label != j_label, "two judges collapsed to the same folder label"
        # A judge-invariant family ignores the active judge on the WRITE side.
        E.set_family("lookahead/reward")
        inv_fig = E._fig_dir()
        assert inv_fig == os.path.join(E.family_root(), "figures"), \
            f"judge-invariant family must not nest a <judge>/ level: {inv_fig}"
    finally:
        K.set_active_judge("")
        E.set_family("")
    assert K.active_judge() == "", "active judge not restored"
    return (f"reads -> eval_scores/judge={{{p_label},{j_label}}}/rep=N; "
            f"writes -> arms/<sub>/figures/{{{p_label},{j_label}}}/ (siblings); invariant families flat")


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
    """A CROSS-JUDGE artifact must never sit under a single judge's folder; ``arms/*`` always does.

    The ``<judge>/`` path segment asserts "this grader produced this file". For a multi-judge
    artifact that is false — ``multijudge_variance_decomposition.png`` plots BOTH graders, and
    filing it under ``gpt-4o-mini/`` credits the primary with the very figure that proves the two
    agree. Since the 2026-08-18 reorg the rule is by TOP: ``arms/*`` (``PER_JUDGE_TOPS``) nests a
    judge leaf and is rendered once per grader; every other top (``lookahead``, ``method``,
    ``compute``, ``measurement``) is judge-invariant with no such level.

    Structural: asserts the router (``exports._leaf``) and walks whatever of the new results tree
    exists (an absent tree passes the structural half only).
    """
    import re as _re
    from eda_analysis import exports as X
    from eda_analysis.config import FAMILIES, PER_JUDGE_TOPS, is_per_judge
    from eda_analysis.constants import judge_dirname

    # Router: per-judge family -> <family>/figures/<judge>/ ; invariant -> <family>/figures/
    try:
        X.set_family("arms/outcomes")
        assert X._leaf("figures") == os.path.join(X.RESULTS_DIR, "arms", "outcomes", "figures",
                                                 judge_dirname("")), X._leaf("figures")
        assert X._leaf("tables", "mici") == os.path.join(X.RESULTS_DIR, "arms", "outcomes", "tables",
                                                        judge_dirname(""), "mici")
        for fam in ("measurement/validity", "lookahead/reward", "method/contrast", "compute/cost"):
            X.set_family(fam)
            top, sub = fam.split("/")
            assert not is_per_judge(fam) and X.is_judge_invariant()
            assert X._leaf("figures") == os.path.join(X.RESULTS_DIR, top, sub, "figures"), \
                f"{fam}: _leaf still appends a <judge> segment to a judge-invariant family"
    finally:
        X.set_family("")
    assert "measurement" not in PER_JUDGE_TOPS, "measurement/* must be judge-invariant"

    # Tree walk (whatever exists): cross-judge artifacts never below a <judge>/ folder of a
    # per-judge family; per-judge families never hold artifacts directly under figures|tables/.
    cross = _re.compile(r"^(multijudge_|second_judge_|oracle_repeatability_|judge_)")
    art_ext = (".png", ".pdf", ".svg", ".md", ".json")
    offenders, flat, seen = [], [], 0
    for top, subs in FAMILIES.items():
        for sub in subs:
            for kind in ("figures", "tables"):
                root = os.path.join(X.RESULTS_DIR, top, sub, kind)
                if not os.path.isdir(root):
                    continue
                for dirpath, _dirs, files in os.walk(root):
                    rel = os.path.relpath(dirpath, root)
                    parts = [] if rel == "." else rel.split(os.sep)
                    for f in files:
                        if f in ("CAPTIONS.md",) or f.startswith("_prov"):
                            continue
                        if is_per_judge(top) and not parts and f.lower().endswith(art_ext):
                            flat.append(f"{top}/{sub}/{kind}/{f}")
                        if cross.match(f):
                            seen += 1
                            if is_per_judge(top):
                                offenders.append(f"{top}/{sub}/{kind}/{rel}/{f}")
    assert not offenders, ("cross-judge artifacts must live in a judge-invariant family, never "
                           f"under arms/*/<judge>/: {sorted(offenders)[:6]}")
    assert not flat, ("per-judge family artifacts must sit under a <judge>/ leaf, not directly "
                      f"under figures|tables/: {sorted(flat)[:6]}")
    return (f"router: arms/* -> <judge>/ leaf, {len(FAMILIES) - len(PER_JUDGE_TOPS)} invariant "
            f"tops flat; tree walk: {seen} cross-judge artifacts, none under a <judge>/ dir")


def _c_exports_routing() -> str:
    """The exports router: no family -> refuse; family -> the documented leaf; PRESERVE guarded.

    Exercised against a TEMP results root (the module's ``RESULTS_DIR`` is swapped for the
    duration and restored), so nothing under the real ``results/`` is touched. Asserts:
    ``save_*`` raise :class:`~eda_analysis.exports.NoFamilyError` with no family; ``save_table`` /
    ``save_fig`` / ``save_numbers`` land at ``<family>/{tables,figures}/[<judge>/][<group>/]``;
    the workbook is named for the family sub / innermost group; ``build_index`` writes
    ``<top>/INDEX.md`` + ``INDEX.md`` with captions; ``reset_results`` clears only the active leaf;
    ``_guard_path`` refuses ``PRESERVE`` names; ``.xlsx`` bytes are identical across two saves of
    the same frame (the frozen-timestamp guarantee).
    """
    import shutil
    import tempfile
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eda_analysis import exports as X
    from eda_analysis.constants import judge_dirname

    real_root = X.RESULTS_DIR
    tmp = tempfile.mkdtemp(prefix="eda_exports_probe_")
    X.RESULTS_DIR = tmp
    try:
        X.set_family("")
        for fn, args in ((X.save_table, (pd.DataFrame({"a": [1]}), "t")),
                         (X.save_numbers, ("n", {"k": 1})),
                         (X.build_index, ()), (X.reset_results, ())):
            try:
                fn(*args)
            except X.NoFamilyError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} ran with NO family set (bare-root fallback?)")

        X.set_family("arms/outcomes")
        j = judge_dirname("")
        d = X.save_table(pd.DataFrame({"a": [1, 2]}), "t1", caption="cap t1")
        assert d == os.path.join(tmp, "arms", "outcomes", "tables", j), d
        assert os.path.isfile(os.path.join(d, "t1.md")) and os.path.isfile(os.path.join(d, "outcomes.xlsx"))
        d2 = X.save_table(pd.DataFrame({"a": [1]}), "t2", group="mici")
        assert d2 == os.path.join(d, "mici") and os.path.isfile(os.path.join(d2, "mici.xlsx"))
        b1 = open(os.path.join(d, "outcomes.xlsx"), "rb").read()
        X.save_table(pd.DataFrame({"a": [1, 2]}), "t1", caption="cap t1")
        b2 = open(os.path.join(d, "outcomes.xlsx"), "rb").read()
        assert b1 == b2, "re-saving an unchanged table changed the workbook bytes (timestamps leak)"
        fig, ax = plt.subplots(); ax.plot([0, 1])
        fd = X.save_fig(fig, "f1", caption="cap f1"); plt.close(fig)
        assert fd == os.path.join(tmp, "arms", "outcomes", "figures", j) and \
            os.path.isfile(os.path.join(fd, "f1.png"))
        np_ = X.save_numbers("nums", {"a.b": 1.5, "c": {"value": 2, "source": "s"}})
        doc = json.load(open(np_, encoding="utf-8"))
        assert doc["numbers"]["a.b"] == {"value": 1.5, "source": "", "note": ""} and \
            doc["numbers"]["c"]["source"] == "s", doc
        X.save_numbers("nums", {"a.b": 9})
        doc = json.load(open(np_, encoding="utf-8"))
        assert doc["numbers"]["a.b"]["value"] == 9 and "c" in doc["numbers"], "ledger merge broke"
        idx = X.build_index()
        assert idx == os.path.join(tmp, "arms", "INDEX.md")
        txt = open(idx, encoding="utf-8").read()
        assert "f1.png" in txt and "cap f1" in txt and "t1.md" in txt and "nums.json" in txt, txt[:400]
        root_idx = open(os.path.join(tmp, "INDEX.md"), encoding="utf-8").read()
        assert "`arms/outcomes`" in root_idx and "`lookahead/reward`" in root_idx
        # PRESERVE guard
        os.makedirs(os.path.join(tmp, "arms"), exist_ok=True)
        open(os.path.join(tmp, "arms", "SUMMARY.md"), "w").write("hand-authored\n")
        for bad in (os.path.join(tmp, "arms", "SUMMARY.md"), os.path.join(tmp, "schematics"),
                    os.path.join(tmp, "..", "outside")):
            try:
                X._guard_path(bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"_guard_path allowed {bad}")
        # judge-scoped reset: another judge's leaf must survive
        other = os.path.join(tmp, "arms", "outcomes", "tables", "other-judge")
        os.makedirs(other); open(os.path.join(other, "x.md"), "w").write("x")
        X.reset_results()
        assert not os.path.isdir(d) and not os.path.isdir(fd), "reset_results left the active leaf"
        assert os.path.isfile(os.path.join(other, "x.md")), "reset_results deleted ANOTHER judge's leaf"
        assert os.path.isfile(os.path.join(tmp, "arms", "SUMMARY.md")), "reset touched SUMMARY.md"
        # invariant family: flat leaf, reset clears figures/ + tables/ themselves
        X.set_family("measurement/validity")
        d3 = X.save_table(pd.DataFrame({"a": [1]}), "t3")
        assert d3 == os.path.join(tmp, "measurement", "validity", "tables"), d3
        assert os.path.isfile(os.path.join(d3, "validity.xlsx"))
        X.reset_results()
        assert not os.path.isdir(d3)
    finally:
        X.set_family("")
        X.RESULTS_DIR = real_root
        shutil.rmtree(tmp, ignore_errors=True)
    return ("no-family refused; leaves compose <family>/{figures,tables}/[<judge>/][<group>/]; "
            "ledger merge; index+captions; PRESERVE guarded; judge-scoped reset; xlsx bytes stable")


_FIXTURE_DIR = os.path.normpath(os.path.join(_EDA_DIR, "..", "..", "papers",
                                             "2026_lookahead_pto_grpo", "analysis", "out"))


def _fixture(name: str) -> dict:
    fp = os.path.join(_FIXTURE_DIR, f"{name}.json")
    if not os.path.isfile(fp):
        raise _Skip(f"paper fixture missing: {fp}")
    return json.load(open(fp, encoding="utf-8"))["numbers"]


def _scout_first_groups(arm, *, n_groups: int | None = None) -> dict:
    """The tails SCOUT anchor read the cheap way: stream ONE file — the arm's
    ``iteration_<train_iter>/eda/generations.jsonl`` — and keep only the first ``n_groups`` logged
    groups (the paper's 300 = 2,400 candidates), instead of ``training.load_generations`` over the
    whole arm with tails attached (the ~2 min path :func:`tails.tail_audit_frames` takes).

    Row semantics are exactly ``tails._audit_one_arm``'s: drop ``phase == "eval"`` records, drop
    candidates that were not simulated (``lookahead.realized_turns`` null) or not scored
    (``score`` null), take the first ``n_groups`` distinct ``tails.GROUP_KEYS`` in file order and
    every kept candidate of those groups (``drop_duplicates(...).head(n)`` + ``merge``, so a group
    that reappears later in the file is still counted, as in the module). Returns the same dict
    shape as ``TailAudit.scout_check`` (``n``, ``realized_turns`` histogram, ``ended_early``, rate).
    """
    import pandas as pd
    from eda_analysis import tails as _tails
    exp = _tails.SCOUT_EXPECTED
    n_groups = int(n_groups or exp["n_groups"])
    ti = int(exp["train_iter"])
    keys = list(_tails.GROUP_KEYS)                    # train_iter, conversation_id, branch_id, epoch
    fp = os.path.join(arm.runs_dir, f"iteration_{ti}", "eda", "generations.jsonl")
    if not os.path.isfile(fp):
        raise _Skip(f"{arm.label} iteration_{ti}/eda/generations.jsonl not on disk")
    rows = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                                   # noqa: BLE001
                continue
            if rec.get("iteration") != ti or rec.get("phase") == "eval":
                continue
            ep = rec.get("epoch")
            base = (ti, rec.get("conversation_id"), rec.get("branch_id"), -1.0 if ep is None else ep)
            for c in rec.get("candidates", []):
                la = c.get("lookahead") or {}
                rt, sc = la.get("realized_turns"), c.get("score")
                if rt is None or sc is None or (isinstance(sc, float) and sc != sc):
                    continue
                rows.append((*base, int(rt), bool(la.get("ended_early"))))
    df = pd.DataFrame(rows, columns=keys + ["realized_turns", "ended_early"])
    if df.empty:
        raise _Skip(f"{arm.label} iteration {ti}: no scored + simulated candidates in generations.jsonl")
    first = df.drop_duplicates(keys)[keys].head(n_groups)
    sub = df.merge(first, on=keys)
    return {"n": int(len(sub)),
            "realized_turns": {int(k): int(v) for k, v in sub["realized_turns"].value_counts().items()},
            "ended_early": int(sub["ended_early"].sum()),
            "ended_early_rate": float(sub["ended_early"].mean()),
            "n_groups": int(len(first))}


def _c_paper_fixture(probe: bool = False) -> str:
    """The paper's frozen numbers reproduce from the promoted modules (the anchor cells).

    ``papers/2026_lookahead_pto_grpo/analysis/out/*.json`` is KEPT as a fixture after the paper's
    generators were promoted into the package (2026-08-18 reorg): if a promoted module drifts, the
    paper's quoted numbers stop matching what the EDA renders, and nothing else would notice.
    Anchors (means / dz / ratios / counts must match to the fixture's precision; the fixture's
    bootstrap CIs are seed-0 while the package seeds with ``BOOT_SEED``, so CIs are NOT compared —
    every module report showed CIs exact under seed 0 and within Monte-Carlo noise otherwise):

      * PTO Q1Q2 iter 6, K=0 - K=5, primary judge: +0.257 / dz 0.417 — ``stats.paired_arrays`` on
        the ``persona_id x model`` pivot of ``PTOExp3_LA0_I6`` vs ``PTOExp3_LA5_I6``.
      * GRPO per-step K5/K0 median ratio, iters 3/4/5 = 1.965 / 1.962 / 1.911 —
        ``compute.step_multiplier`` (tol 5e-4, the fixture's rounding).
      * PTO iter-1 ``margin_mean``: PTO_LA5 0.424 and PTO_LA0 0.274 —
        ``dispersion.dispersion_by_iter`` (both PTO arms, ``n_perm=2``: the anchor is a mean, not
        the shuffle-null columns).
      * GRPO_LA5 iter 5, first 300 groups: ended_early 657 / 2400 (+ the realized-turn histogram
        in ``tails.SCOUT_EXPECTED``) — read the CHEAP way by :func:`_scout_first_groups` (one
        file, first 300 groups) so it runs in the default pass; under ``--probe`` the module's own
        ``tails.tail_audit_frames(...).scout_check`` (the full ~2 min pass) must agree too.

    Data-guarded like every other data check (SKIP when the arms / files are not on disk). A
    missing anchor module is a FAIL now that Phase C1 has landed every promoted module.
    """
    arms = _discover_or_skip()
    notes = []

    # 1 — the headline K contrast via stats.paired_arrays (always).
    fx = _fixture("k_contrast_headline")["xcheck.pto_q1q2_iter6_primary"]["value"]
    scores = E.load_scores_long(arms)
    q = scores[(scores.questionnaire == "Q1Q2") & (scores.model.isin(["PTOExp3_LA0_I6", "PTOExp3_LA5_I6"]))]
    if q.empty:
        raise _Skip("PTOExp3_LA0_I6 / PTOExp3_LA5_I6 not scored")
    piv = q.pivot_table(index="persona_id", columns="model", values="score", aggfunc="mean")
    from eda_analysis.stats import paired_arrays
    r = paired_arrays(piv["PTOExp3_LA0_I6"].to_numpy(), piv["PTOExp3_LA5_I6"].to_numpy())
    assert r["n"] == 96, f"paired n {r['n']} != 96"
    assert abs(r["mean_delta"] - fx["mean_delta"]) < 1e-9, \
        f"PTO Q1Q2 iter6 mean_delta {r['mean_delta']:.6f} != fixture {fx['mean_delta']:.6f}"
    assert abs(r["dz"] - fx["dz"]) < 1e-9, f"PTO Q1Q2 iter6 dz {r['dz']:.6f} != fixture {fx['dz']:.6f}"
    assert abs(r["p"] - fx["p"]) < 1e-9, f"PTO Q1Q2 iter6 p {r['p']:.3e} != fixture {fx['p']:.3e}"
    notes.append(f"K contrast PTO@6 {r['mean_delta']:+.3f}/dz {r['dz']:.3f}")

    # 2 — GRPO per-step K5/K0 ratio (compute.step_multiplier).
    from eda_analysis import compute as _cmp
    fxc = _fixture("compute_axis")
    if not {"GRPO_LA0", "GRPO_LA5"} <= {a.label for a in arms}:
        raise _Skip("GRPO_LA0 + GRPO_LA5 needed for the step-ratio anchor")
    comp = _cmp.iteration_compute(arms)
    sm = _cmp.step_multiplier(comp).set_index("iteration")
    got = []
    for it in (3, 4, 5):
        want = fxc[f"step_multiplier.I{it}"]["value"]["GRPO_step_ratio"]
        have = float(sm.loc[it, "ratio_median"])
        assert abs(have - want) < 5e-4, f"GRPO step ratio iter {it}: {have:.4f} != fixture {want}"
        got.append(f"{have:.3f}")
    notes.append("GRPO step ratio I3-5 " + "/".join(got))

    # 3 — PTO iter-1 mean margin, both K arms (dispersion.dispersion_by_iter).
    from eda_analysis import dispersion as _disp
    fxd = _fixture("dispersion_by_k")["ratios.PTO.iter1"]["value"]
    pto = [a for a in arms if a.label in ("PTO_LA0", "PTO_LA5")]
    if len(pto) < 2:
        raise _Skip("PTO_LA0 + PTO_LA5 needed for the margin anchor")
    by = _disp.dispersion_by_iter(pto, arm_labels=["PTO_LA0", "PTO_LA5"], n_perm=2)
    got = []
    for label, key in (("PTO_LA5", "margin_K5"), ("PTO_LA0", "margin_K0")):
        want = float(fxd[key])
        cell = by[(by.arm == label) & (by.train_iter == 1)]["margin_mean"]
        assert len(cell) == 1, f"dispersion_by_iter: {label} train_iter 1 row missing"
        have = float(cell.iloc[0])
        assert abs(have - want) < 1e-6, f"{label} iter1 margin_mean {have:.6f} != fixture {want:.6f}"
        got.append(f"{label}@1 margin {have:.3f}")
    notes.append(", ".join(got))

    # 4 — GRPO_LA5 iter 5 first-300-groups ended_early (tails scout anchor; cheap path always).
    from eda_analysis import tails as _tails
    want = _fixture("tail_audit")["scout_check.GRPO_LA5.iter5.first300groups"]["value"]
    exp = _tails.SCOUT_EXPECTED
    g5 = [a for a in arms if a.label == exp["arm"]]
    if not g5:
        raise _Skip(f"{exp['arm']} not on disk")
    sc = _scout_first_groups(g5[0])
    assert sc["n_groups"] == exp["n_groups"], f"scout read {sc['n_groups']} groups, expected {exp['n_groups']}"
    assert int(sc["n"]) == int(want["n"]) == int(exp["n"]), \
        f"scout n {sc['n']} != fixture {want['n']} / SCOUT_EXPECTED {exp['n']}"
    assert int(sc["ended_early"]) == int(want["ended_early"]) == int(exp["ended_early"]), \
        f"scout ended_early {sc['ended_early']} != fixture {want['ended_early']}"
    want_rt = {int(k): int(v) for k, v in want["realized_turns"].items()}
    assert sc["realized_turns"] == want_rt == dict(exp["realized_turns"]), \
        f"scout realized-turn histogram {sc['realized_turns']} != fixture {want_rt}"
    notes.append(f"GRPO_LA5@5 ended_early {sc['ended_early']}/{sc['n']} (first {sc['n_groups']} groups)")
    if probe:                                                   # the module's own heavy pass
        full = _tails.tail_audit_frames(g5, verbose=False).scout_check
        assert full and int(full["n"]) == sc["n"] and int(full["ended_early"]) == sc["ended_early"] \
            and {int(k): int(v) for k, v in full["realized_turns"].items()} == sc["realized_turns"], \
            f"tails.tail_audit_frames scout_check {full} != cheap read {sc}"
        notes.append("tails.tail_audit_frames agrees")

    return "; ".join(notes)


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
    _run("family map", _c_family_map, results)
    _run("EdaConfig round-trip", _c_config_roundtrip, results)
    _run("live aliases (figures/plots)", _c_live_aliases, results)
    _run("scoring subpackage surface", _c_scoring_surface, results)
    _run("notebook symbol refs resolve", _c_notebook_refs_resolve, results)
    _run("cache mechanism + invalidation", _c_cache_mechanism, results)
    _run("rubric parity (2nd judge gate)", _c_rubric_parity, results)
    _run("cross-judge artifact layout", _c_cross_judge_layout, results)
    _run("exports routing (family root)", _c_exports_routing, results)
    _run("seeded bootstrap (repro figures)", _c_seeded_bootstrap, results)
    _run("role bindings + name suffix", _c_role_bindings, results)
    # Data — unless --fast.
    if not fast:
        _run("discover_arms", _c_discover, results)
        _run("arm identity is collision-free", _c_arm_identity_unique, results)
        _run("scores_long + known means", _c_scores_and_means, results)
        _run("cross-K frame", _c_cross_k, results)
        _run("compute axis (GPU-hours)", _c_compute_axis, results)
        _run("paper fixture anchors", lambda: _c_paper_fixture(probe=probe), results)
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
        mark = {"PASS": "OK  ", "SKIP": "skip", "WARN": "WARN", "FAIL": "FAIL"}[status]
        print(f"  [{mark}] {name.ljust(width)}  {detail}")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    print(" " + "-" * (width + 30))
    print(f"  {n_pass} passed, {n_warn} warned, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
