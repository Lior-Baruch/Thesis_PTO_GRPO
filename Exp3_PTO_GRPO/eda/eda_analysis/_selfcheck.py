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
Probe (opt-in, heavy — needs sentence-transformers + pref pairs):
  * the PTO Mass-Mean-Probe ``wins_correct`` > 0.5 (the chosen-rejected direction separates pairs).
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from glob import glob
from typing import Callable, List, Tuple

# Import the package the same way the notebooks do (cwd = eda/, package on the path).
import eda_analysis as E  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_EDA_DIR = os.path.dirname(_HERE)                       # .../eda

# Known-good endpoints (EDA's Q1Q2 = mean(Q1,Q2) convention; see project memory / SUMMARY.md).
_KNOWN_Q1Q2_FINAL = {"PTO_LA0": 4.26, "GRPO_LA0": 3.75}
_KNOWN_TOL = 0.02

# Submodule names a notebook may qualify a call with (live modules + back-compat aliases).
_SUBMODULES = ("plotting", "plots", "figures", "data", "personas", "scores", "discovery",
               "select", "stats", "behavior", "training", "pref", "exports")


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
    assert set(C._VIEW_KS) == {"all", "L0", "L2", "L5"}, C._VIEW_KS
    # Every alias target is a real view; case-insensitive.
    for k, v in C._VIEW_ALIASES.items():
        assert v in C._VIEW_KS, f"alias {k!r} -> unknown view {v!r}"
    assert C._VIEW_ALIASES["l0"] == "L0" and C._VIEW_ALIASES["all"] == "all"
    return f"view->ks {C._VIEW_KS}"


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


def _notebook_symbol_refs() -> dict:
    """Scan committed notebooks for ``<submodule>.<attr>(`` calls -> {submodule: {attr, ...}}."""
    pat = re.compile(r"\b(" + "|".join(_SUBMODULES) + r")\.([A-Za-z_][A-Za-z0-9_]*)")
    refs: dict = {m: set() for m in _SUBMODULES}
    for nb in glob(os.path.join(_EDA_DIR, "*.ipynb")):
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
def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    fast = "--fast" in argv
    probe = "--probe" in argv

    results: _Results = []
    # Structural — always.
    _run("import + __all__ resolve", _c_all_resolves, results)
    _run("view->ks map", _c_view_map, results)
    _run("EdaConfig round-trip", _c_config_roundtrip, results)
    _run("live aliases (figures/plots)", _c_live_aliases, results)
    _run("notebook symbol refs resolve", _c_notebook_refs_resolve, results)
    # Data — unless --fast.
    if not fast:
        _run("discover_arms", _c_discover, results)
        _run("scores_long + known means", _c_scores_and_means, results)
        _run("persona permutation", _c_persona_permutation, results)
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
