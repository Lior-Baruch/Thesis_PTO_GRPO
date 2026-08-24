"""_selfcheck.py -- the regression guard for ``eda_analysis``: run it after ANY EDA change.

The EDA is a package plus four notebooks that only ever run end-to-end. That combination hides a
specific class of breakage: a renamed function, a family with no notebook, an export that lands
somewhere no index points at, a figure whose error bars are re-randomised on every render. None of
those raise when they happen -- they surface as an empty table, a stale leaf, or a PNG that churns
in git -- and every one of them costs a full render (or a paper draft) to discover.

So this module asserts the invariants CHEAPLY, in one process, with no notebook execution, no
network and no GPU::

    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck          # structural + data
    ../../.venv/Scripts/python.exe -m eda_analysis._selfcheck --fast   # structural only, no disk
    SELFCHECK_TRACE=1 ... -m eda_analysis._selfcheck                   # tracebacks on FAIL

Four statuses, and the difference between them matters:

======  =========================================================================================
PASS    the invariant holds.
WARN    a known, expected gap that is reported but does NOT fail the run -- an arm that has
        trained but not been scored yet, an iteration that was resumed, a results tree that has
        never been rendered. These are normal states of a live experiment.
SKIP    there was nothing to check (no ``data/`` mount, no arms on disk, no notebooks yet).
FAIL    an invariant broke. The run exits non-zero.
======  =========================================================================================

Checks (18: 14 structural + 4 data)
-----------------------------------
Structural (always; ``--fast`` runs only these, and they touch no experiment data):

* ``imports + __all__``    -- every package module imports and every ``__all__`` name resolves,
  and the paid ``scoring`` module stays reachable only by an explicit import.
* ``family map``           -- ``FAMILIES`` <-> ``notebooks/<top>/<sub>.ipynb``, 1:1 both ways.
* ``EdaConfig round-trip`` -- ``as_dict`` / ``with_`` / family validation / JSON-serialisable.
* ``metric registry``      -- the eight instruments are still DERIVED from ``questionnaires.py``,
  and from the canonical copy under ``code/`` rather than another experiment's.
* ``palette parity``       -- ``constants.ARM_COLORS`` and ``plotting.ARM_COLORS`` still agree.
* ``partition tokens``     -- every judge tag and metric partition is a legal directory name,
  validated by the WRITER's own path builder.
* ``empty-frame contracts`` -- every reader returns a typed empty frame on an empty tree.
* ``notebook cell-1``      -- every notebook's first code cell builds an ``EdaConfig`` whose family
  matches its own path and calls ``notebook_setup``.
* ``notebook symbol refs`` -- every ``<submodule>.<attr>(`` a notebook calls actually exists.
* ``seeded bootstrap``     -- every resampling callsite is seeded with ``BOOT_SEED``.
* ``exports routing``      -- ``save_*`` land under ``results/<top>/<sub>/`` and NEVER under a
  ``judge=`` or ``<judge>/`` segment; PRESERVE guarded; workbook bytes stable.
* ``arm identity``         -- a grid of built names round-trips through ``naming`` and no two
  distinct configurations collide on one name.
* ``no torch``             -- importing the package pulls in no torch/transformers/peft/trl.
* ``MICI orientation``     -- MICI is registered lower-is-better and nothing ranks it ascending.

Data (skipped cleanly when ``data/`` is absent or empty):

* ``score coverage``       -- on-disk model states vs states present in the lake, per judge.
* ``persona coverage``     -- every scored parquet holds 96 unique ``persona_id`` values.
* ``timing logs``          -- every COMPLETED iteration has a ``timing_sessions.jsonl``.
* ``render freshness``     -- rendered artifacts are newer than the inputs they were built from.

What this deliberately does NOT do
----------------------------------
Exp3's version grew to 1,719 lines and 26 checks, several of which were archaeology for that
experiment's own history: a frozen paper fixture, a rubric-parity gate for a second commercial
judge, a cross-generation probe, and a prose-arithmetic parser that audited the repo's Markdown.
None are ported. Four more Exp3 checks have no Exp4 analogue because the defect they guarded is
fixed upstream: there is no persona-shuffle replay to verify, no parquet fold signature, no
mtime-reconstructed compute axis, and no ``oracle=`` path level.

Nothing here executes a notebook -- that is ``tools/render_results.py``'s job. This module answers
"would a render be correct?"; that one answers "does a render still work?".
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import json
import os
import re
import sys
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# Imported the way a notebook imports it: cwd = eda/, package on the path.
import eda_analysis as E

__all__ = ["main"]


_HERE = os.path.dirname(os.path.abspath(__file__))
_EDA_DIR = os.path.dirname(_HERE)                       # .../eda
_PKG_NAME = "eda_analysis"

#: The ANALYSIS modules, in dependency order (leaf first). ``_selfcheck`` itself is not one.
_PACKAGE_MODULES: Tuple[str, ...] = ("constants", "config", "data", "exports", "stats", "plotting")

#: The PAID side. ``scoring`` is the only module that talks to a model, and it is deliberately
#: absent from the package's lazy attribute map so ``import eda_analysis`` cannot reach it by
#: accident -- ``Run_Eval.ipynb`` imports it explicitly, and that import is the moment someone
#: chooses to spend. It is checked separately here for exactly that reason.
_PAID_MODULES: Tuple[str, ...] = ("scoring",)

#: Submodule names a notebook may qualify a call with. Used by the symbol-reference scan.
_REF_MODULES: Tuple[str, ...] = _PACKAGE_MODULES

#: Import roots the EDA must never pull in. The EDA reads finished artifacts; anything here means
#: an analysis module imported trainer code, which makes ``import eda_analysis`` cost a CUDA init
#: (and, on the local Blackwell card, makes import ORDER load-bearing -- see CLAUDE.md gotchas).
_FORBIDDEN_ROOTS: Tuple[str, ...] = (
    "torch", "torchvision", "torchaudio", "transformers", "peft", "trl",
    "accelerate", "bitsandbytes", "vllm",
)

#: Trainer modules the EDA must not import even though they are torch-free at the top: they are
#: the GPU side of the experiment and importing them here means a layering mistake.
_FORBIDDEN_CORE: Tuple[str, ...] = ("core.policy", "core.lookahead")

#: Notebook folders under ``notebooks/`` that are NOT families. ``scoring/`` is the PAID side
#: (Run_Eval), which writes the score lake rather than ``results/``.
_NON_FAMILY_NOTEBOOK_DIRS: frozenset = frozenset({"scoring"})

#: The full persona grid every conversation folder and every score partition should cover.
_N_PERSONAS = int(getattr(E, "N_PERSONAS", 96))

#: Sentinel for "this keyword was not passed at all", distinct from a keyword passed as ``None``.
_MISSING_KW = object()


# ==============================================================================
#  Harness
# ==============================================================================


class _Skip(Exception):
    """Raised by a check to mark itself SKIPPED: there was nothing to check."""


class _Warn(Exception):
    """Raised by a check to mark itself WARN: a known, expected gap, not a broken invariant.

    Distinct from SKIP (nothing to check) and FAIL (an invariant broke). A WARN never changes the
    exit code, so it is the right status for a normal state of a live experiment -- an unscored
    arm, a resumed iteration, an unrendered results tree.
    """


_Results = List[Tuple[str, str, str]]        # (name, status, detail)


def _run(name: str, fn: Callable[[], str], results: _Results) -> None:
    """Run one check, recording PASS / WARN / SKIP / FAIL and never propagating.

    A check that raises anything other than :class:`_Skip` / :class:`_Warn` is a FAIL: the point is
    to report every invariant in one pass, so one broken check must not hide the nine behind it.
    Set ``SELFCHECK_TRACE=1`` to print the traceback of a failure.
    """
    try:
        detail = fn() or ""
        results.append((name, "PASS", detail))
    except _Skip as skip:
        results.append((name, "SKIP", str(skip)))
    except _Warn as warn:
        results.append((name, "WARN", str(warn)))
    except Exception as exc:                                   # noqa: BLE001 -- report, never raise
        results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        if os.environ.get("SELFCHECK_TRACE"):
            traceback.print_exc()


def _rel(path: str) -> str:
    """A path relative to the experiment root, for readable messages.

    Falls back to the absolute path when *path* is outside the workspace (a temp probe tree, a
    Drive symlink target), because ``../../../AppData/Local/Temp/...`` is less readable than the
    real thing, not more.
    """
    try:
        out = os.path.relpath(path, E.WORKSPACE_ROOT).replace(os.sep, "/")
    except ValueError:
        return path
    return path if out.startswith("..") else out


def _notebook_path(family: str) -> str:
    top, sub = family.split("/")
    return os.path.join(E.constants.NOTEBOOKS_DIR, top, f"{sub}.ipynb")


def _read_notebook(path: str) -> dict:
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _code_cells(doc: dict) -> List[str]:
    """Source of every code cell, joined. Raw strings -- no comment or literal stripping."""
    out: List[str] = []
    for cell in doc.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            out.append(source if isinstance(source, str) else "".join(source))
    return out


_COMMENT_RE = re.compile(r"#.*$", re.M)
_LITERAL_RE = re.compile(r"(\"\"\".*?\"\"\"|'''.*?'''|\"[^\"\n]*\"|'[^'\n]*')", re.S)


def _strip_prose(source: str) -> str:
    """Blank out comments and string literals so a scan sees CODE only.

    Without this, ``save_fig(fig, "stats.png")`` reads as a ``stats.png`` symbol reference and a
    ``# see plotting.contrast_forest`` note reads as a call. Both are prose about the code, not
    code, and both produced false positives in Exp3 before it did the same thing.
    """
    return _COMMENT_RE.sub("", _LITERAL_RE.sub("''", source))


def _existing_family_notebooks() -> List[Tuple[str, str]]:
    """``[(family, path), ...]`` for the family notebooks that are actually on disk."""
    return [(fam, _notebook_path(fam)) for fam in E.all_families()
            if os.path.isfile(_notebook_path(fam))]


# ==============================================================================
#  Structural checks
# ==============================================================================


def _c_imports() -> str:
    """Every package module imports, and every name it advertises actually resolves.

    ``__all__`` is the package's public contract -- ``from eda_analysis.stats import *`` and the
    lazy ``eda_analysis.__getattr__`` both read it. A name listed there but never defined is an
    ImportError that only fires the first time a notebook happens to use it, which is typically
    twenty minutes into a render.
    """
    modules = _PACKAGE_MODULES + _PAID_MODULES
    checked = 0
    for name in modules:
        module = importlib.import_module(f".{name}", _PKG_NAME)
        exported = list(getattr(module, "__all__", ()))
        assert exported, f"{_PKG_NAME}.{name} declares no __all__"
        missing = [n for n in exported if not hasattr(module, n)]
        assert not missing, f"{_PKG_NAME}.{name}.__all__ names that do not resolve: {missing}"
        checked += len(exported)

    missing_pkg = [n for n in E.__all__ if not hasattr(E, n)]
    assert not missing_pkg, (
        f"{_PKG_NAME}.__all__ names that do not resolve: {missing_pkg}. These go through the lazy "
        f"__getattr__, so the usual cause is an entry-point map that names a module which no "
        f"longer defines the symbol."
    )

    # The paid module must stay unreachable by accident: not a lazily-searched submodule and not
    # the source of any re-exported entry point. Otherwise a bare attribute lookup on the package
    # can import the one module that builds a client.
    for paid in _PAID_MODULES:
        assert paid not in getattr(E, "_SUBMODULES", ()), (
            f"{_PKG_NAME}.{paid} is in the lazy _SUBMODULES search list -- a plain attribute "
            f"lookup would import the module that talks to a grader.")
        assert paid not in set(getattr(E, "_ENTRY_POINTS", {}).values()), (
            f"{_PKG_NAME}.{paid} backs a re-exported entry point; it must be imported explicitly.")
    return (f"{len(modules)} modules, {checked} module-level + {len(E.__all__)} package-level "
            f"names resolve; {list(_PAID_MODULES)} reachable only explicitly")


def _c_family_map() -> str:
    """``config.FAMILIES`` and ``notebooks/<top>/<sub>.ipynb`` are 1:1 in BOTH directions.

    ``tools/render_results.py`` iterates ``FAMILIES``, so the two failure modes are silent and
    opposite: a family with no notebook renders nothing (the results folder simply never appears,
    and ``results/INDEX.md`` marks it "not rendered yet" forever), while a notebook with no family
    entry is never executed by the driver at all and its artifacts go stale the moment someone
    renders instead of running it by hand.

    Notes:
        ``notebooks/scoring/`` is exempt -- it is the PAID side (Run_Eval writes the score lake,
        not ``results/``) and deliberately has no family.
    """
    from eda_analysis import config as C

    assert C.FAMILIES and all(subs for subs in C.FAMILIES.values()), \
        "config.FAMILIES must be non-empty and every top must own at least one subfamily"
    fams = C.all_families()
    assert len(fams) == len(set(fams)), f"duplicate family in FAMILIES: {fams}"
    for fam in fams:
        top, sub = C.split_family(fam)
        assert f"{top}/{sub}" == fam, f"split_family({fam!r}) round-trip gave {top}/{sub}"
    for junk in ("", "arms", "arms/", "arms/nope", "nope/outcomes", "a/b/c"):
        try:
            C.split_family(junk)
        except ValueError:
            pass
        else:
            raise AssertionError(f"split_family accepted junk family {junk!r}")

    nb_root = E.constants.NOTEBOOKS_DIR
    assert os.path.isdir(nb_root), (
        f"notebooks root {_rel(nb_root)} does not exist -- every family needs "
        f"notebooks/<top>/<sub>.ipynb")

    missing = [f for f in fams if not os.path.isfile(_notebook_path(f))]

    # Reverse direction: a notebook under a FAMILIES top that names no family, or a whole top
    # folder holding notebooks while not being a family top at all.
    orphans: List[str] = []
    for entry in sorted(os.listdir(nb_root)):
        top_dir = os.path.join(nb_root, entry)
        if not os.path.isdir(top_dir) or entry in _NON_FAMILY_NOTEBOOK_DIRS:
            continue
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(top_dir)
                       if f.endswith(".ipynb") and not f.startswith("."))
        if entry not in C.FAMILIES:
            orphans += [f"{entry}/{s}.ipynb (top {entry!r} is not in FAMILIES)" for s in stems]
            continue
        orphans += [f"{entry}/{s}.ipynb (no FAMILIES entry {entry}/{s})"
                    for s in stems if s not in C.FAMILIES[entry]]

    problems = []
    if missing:
        problems.append(
            f"{len(missing)} family/families have NO notebook: {missing} -- expected at "
            f"notebooks/<top>/<sub>.ipynb; render_results.py skips them, so they render nothing")
    if orphans:
        problems.append(
            f"{len(orphans)} notebook(s) name no family: {orphans} -- add a config.FAMILIES entry "
            f"or the driver will never execute them")
    assert not problems, "; ".join(problems)
    return f"{len(fams)} families <-> {len(fams)} notebooks, 1:1"


def _c_config_roundtrip() -> str:
    """``EdaConfig`` survives ``as_dict`` / ``with_``, and an unknown family is refused.

    ``as_dict`` is what lands in ``figures/_provenance.md``: it is the only record of which arms,
    metrics and grader produced a rendered figure, so it has to be JSON-clean and complete.
    ``with_`` must not mutate its source, because the original config is what provenance records.
    """
    cfg = E.EdaConfig(family="arms/outcomes", ks=[0], note="selfcheck")
    payload = cfg.as_dict()
    json.dumps(payload)                                      # provenance must be JSON-clean
    assert payload["family"] == "arms/outcomes", payload
    assert payload["ks"] == [0], payload
    assert payload["boot_seed"] == E.BOOT_SEED, payload
    for retired in ("view", "selection", "export_group"):
        assert retired not in payload, f"retired knob {retired!r} leaked into as_dict"

    wider = cfg.with_(ks=[0, 5])
    assert list(wider.ks) == [0, 5] and list(cfg.ks) == [0], "with_ mutated the original config"
    assert wider.family == cfg.family, "with_ dropped a field it was not asked to change"

    default = E.EdaConfig()
    assert default.family == "" and default.ks is None and default.judge == "", \
        "the default config must be every arm, every K, no family (exports disabled)"

    # Family validation happens BEFORE any disk work in notebook_setup, so this stays --fast-safe.
    for bad in ("arms", "nope/outcomes", "arms/nope"):
        try:
            E.notebook_setup(E.EdaConfig(family=bad, verbose=False))
        except ValueError:
            pass
        else:
            raise AssertionError(f"notebook_setup accepted unknown family {bad!r}")
    return "as_dict JSON-clean, with_ non-mutating, unknown family refused"


def _c_metric_registry() -> str:
    """The metric registry is DERIVED from the canonical rubrics, and from the RIGHT copy of them.

    ``constants`` reads ``questionnaires.py`` at import rather than transcribing its 17 Q2 item
    labels, its item counts and its rating scales -- a second copy is a copy that drifts. This
    asserts the derivation still holds, item for item.

    It also asserts WHICH file the derivation read. ``sys.path`` is process-global: if an Exp3
    notebook in the same kernel already imported its own ``questionnaires``, that module object is
    in ``sys.modules`` and every later import returns Exp3's copy no matter what ``constants``
    prepends. The two are byte-identical by contract -- but if that ever stops being true, every
    number downstream was computed against a different instrument, and nothing else would say so.
    """
    import naming as N
    import questionnaires as Q
    import roles as R
    from questionnaires import get_questionnaire

    for module in (Q, N, R):
        source = os.path.dirname(os.path.abspath(module.__file__))
        assert source == E.CODE_DIR, (
            f"{module.__name__!r} was imported from {source!r}, not this experiment's "
            f"{E.CODE_DIR!r} -- another experiment's copy is in sys.modules for this kernel. "
            f"Restart it and import eda_analysis first.")

    for key, spec in E.METRICS.items():
        rubric = get_questionnaire(spec.questionnaire_id, conversation_text="")
        assert spec.item_columns == tuple(rubric.labels), \
            f"{key}: item columns drifted from questionnaires.py"
        assert len(spec.item_columns) == rubric.questions_count, \
            f"{key}: {len(spec.item_columns)} labels but questions_count={rubric.questions_count}"
        assert spec.scale == (rubric.scale_min, rubric.scale_max), \
            f"{key}: scale {spec.scale} != rubric ({rubric.scale_min}, {rubric.scale_max})"
        assert spec.partition and spec.score_column, f"{key}: a stored metric needs both"

    for key, components in E.COMPOSITES.items():
        assert components, f"composite {key!r} lists no components"
        unknown = [c for c in components if c not in E.METRICS]
        assert not unknown, f"composite {key!r} is built from unregistered metrics {unknown}"
        assert E.ALL_METRICS[key].partition is None, \
            f"composite {key!r} claims a lake partition; composites are computed, never stored"

    assert set(E.METRIC_ORDER) == set(E.ALL_METRICS), (
        f"METRIC_ORDER and ALL_METRICS disagree: "
        f"{sorted(set(E.METRIC_ORDER) ^ set(E.ALL_METRICS))} -- a metric missing from the order is "
        f"silently dropped from every figure that iterates it")
    for key, (token, column) in E.QUESTIONNAIRES.items():
        spec = E.ALL_METRICS[key]
        assert (token, column) == (spec.partition, spec.score_column), \
            f"the QUESTIONNAIRES projection disagrees with the registry for {key!r}"
    assert E.TRAINING_REWARD_METRIC in E.ALL_METRICS, "the training-reward metric is not registered"
    return (f"{len(E.METRICS)} stored + {len(E.COMPOSITE_METRICS)} composite metrics derived from "
            f"the canonical rubrics in {_rel(E.CODE_DIR)}/")


def _c_palette_parity() -> str:
    """``constants.ARM_COLORS`` and ``plotting.ARM_COLORS`` are the same map.

    They are currently two literals for one fact, and ``constants.py`` says so in a warning: the
    proper fix is for ``plotting`` to import the map rather than restate it. Until it does, this is
    what keeps them equal -- and the failure they would otherwise produce is the kind nobody reads
    off a plot, because a figure and its legend each look internally consistent while disagreeing
    about which hue is PTO.

    ``arm_color`` is checked through as well: it is the function every figure actually calls, so a
    pinned entry that the resolver does not return is the same bug one layer down.
    """
    from eda_analysis import plotting as P

    mismatch = {k: (E.ARM_COLORS.get(k), P.ARM_COLORS.get(k))
                for k in set(E.ARM_COLORS) | set(P.ARM_COLORS)
                if E.ARM_COLORS.get(k) != P.ARM_COLORS.get(k)}
    assert not mismatch, (
        f"constants.ARM_COLORS and plotting.ARM_COLORS disagree on {mismatch} -- they are two "
        f"copies of one fact, and a figure whose legend uses the other copy is wrong in a way that "
        f"cannot be seen by looking at it.")
    for label, colour in E.ARM_COLORS.items():
        assert P.arm_color(label) == colour, (
            f"plotting.arm_color({label!r}) returned {P.arm_color(label)!r}, not the pinned "
            f"{colour!r}")
    palette = P.arm_palette(list(E.ARM_COLORS))
    subset = P.arm_palette(list(E.ARM_COLORS)[:2])
    assert all(subset[k] == palette[k] for k in subset), \
        "arm_palette gave an arm a different colour when fewer arms were passed"
    return f"{len(E.ARM_COLORS)} pinned arm colours agree across constants/plotting and arm_color"


def _c_partition_tokens() -> str:
    """Every judge tag and metric partition is a legal directory name.

    A tag and a partition token are not strings, they are PATH SEGMENTS in the score lake
    (``judge=<tag>/rep=<r>/metric=<M>/``). One with a separator, a colon or a leading dot does not
    raise where it is defined -- it raises (or worse, silently writes somewhere else) inside
    ``os.makedirs`` much later, on the machine that was doing the scoring.

    The check runs the tokens through the WRITER's own validator (``core.config.RunPaths``), so the
    reader cannot approve a token the writer would refuse.
    """
    from core.config import RunPaths
    from naming import build_experiment_name
    from roles import DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL

    name = build_experiment_name("GRPO", [1, 2], 0, 12, g=8,
                                 oracle_model=DEFAULT_ORACLE_MODEL,
                                 patient_model=DEFAULT_PATIENT_MODEL)
    paths = RunPaths(data_root=E.DATA_DIR, experiment_name=name)

    tags = [E.DEFAULT_JUDGE_TAG] + [E.judge_tag(m) for m in
                                    (DEFAULT_ORACLE_MODEL, "gpt-4o-mini-2024-07-18",
                                     "claude-haiku-4-5")]
    partitions = [E.metric_partition(k) for k in _stored_metrics()]
    for tag in tags:
        for partition in partitions:
            leaf = paths.score_partition_dir(tag, 0, partition)      # raises on an illegal token
            assert os.path.basename(leaf) == name, leaf
    for key in E.COMPOSITES:
        try:
            E.metric_partition(key)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"metric_partition({key!r}) returned a path for a COMPOSITE -- it is computed "
                f"after loading and no such directory will ever exist")
    return (f"{len(set(tags))} judge tag(s) x {len(partitions)} metric partition(s) are legal "
            f"path segments")


def _c_empty_frame_contracts() -> str:
    """Every reader returns a correctly-TYPED empty frame when nothing is on disk.

    A family notebook run before any arm has trained must render empty artifacts, not die. A bare
    ``pd.DataFrame()`` makes it die on the first ``df["arm_label"]`` instead, which reads like a
    broken notebook rather than like "no data yet".

    Dtypes matter as much as names: an ``object``-typed empty ``score`` column turns the first
    ``.mean()`` into a TypeError rather than NaN, and an ``object`` ``iteration`` breaks a merge
    against a populated frame -- both of which surface only once real data exists, which is the
    worst possible time to discover them.
    """
    import shutil
    import tempfile

    from eda_analysis import data as D
    from naming import build_experiment_name, parse_experiment_name
    from roles import DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL

    scratch = tempfile.mkdtemp(prefix="exp4_empty_probe_")
    name = build_experiment_name("PTO", [1, 2], 5, 12, m=8, mode="greedy",
                                 oracle_model=DEFAULT_ORACLE_MODEL,
                                 patient_model=DEFAULT_PATIENT_MODEL)
    arm = D.Arm(experiment_name=name, info=parse_experiment_name(name), iters=(),
                data_root=scratch)

    try:
        cases = {
            "load_scores_long": (D.load_scores_long([], cache=False), D.SCORE_COLUMNS),
            "load_timing": (D.load_timing([]), D.TIMING_COLUMNS),
            "load_conversations": (D.load_conversations(arm, 0), D.CONVERSATION_COLUMNS),
            "load_generations": (D.load_generations(arm, 1), D.GENERATION_COLUMNS),
            "load_pref_pairs": (D.load_pref_pairs(arm, 1), D.PREF_PAIR_COLUMNS),
            "scores_by_judge": (D.scores_by_judge([], judges=[], cache=False), D.SCORE_COLUMNS),
        }
        metadata = D.load_run_metadata(arm)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    problems: List[str] = []
    for reader, (frame, contract) in cases.items():
        if len(frame) != 0:
            problems.append(f"{reader} returned {len(frame)} rows from an empty tree")
            continue
        if list(frame.columns) != list(contract):
            problems.append(f"{reader} columns {list(frame.columns)} != contract {list(contract)}")
            continue
        for column in contract:
            expected = D._dtype_for(column)
            actual = str(frame[column].dtype)
            if actual != expected:
                problems.append(f"{reader}.{column} is {actual}, contract says {expected}")
    assert not problems, "; ".join(problems[:6])
    assert metadata == {}, \
        "load_run_metadata must return {} for an arm with no metadata, not raise"
    return f"{len(cases)} readers return typed empty frames on an empty tree"


_CELL1_CONFIG_RE = re.compile(r"EdaConfig\s*\(")
_CELL1_FAMILY_RE = re.compile(r"family\s*=\s*[\"']([^\"']+)[\"']")
_CELL1_SETUP_RE = re.compile(r"notebook_setup\s*\(")


def _c_notebook_cell1() -> str:
    """Every notebook's FIRST code cell builds its own family's config and calls ``notebook_setup``.

    The cell-1 contract is what makes a render reproducible from a file instead of from a sequence
    of hand-edited cells: ``notebook_setup`` is the only place style, export routing, arm discovery
    and provenance are wired together, so a notebook that skips it writes artifacts with no
    provenance banner and possibly into the wrong family's folder.

    The family string is checked against the notebook's own PATH. A copy-pasted cell 1 that still
    names the family it was copied from is the specific accident this catches -- and it is silent,
    because the notebook runs fine and simply overwrites a sibling family's artifacts.
    """
    present = _existing_family_notebooks()
    if not present:
        raise _Skip("no family notebooks on disk yet")

    problems: List[str] = []
    for family, path in present:
        cells = _code_cells(_read_notebook(path))
        if not cells:
            problems.append(f"{family}: notebook has no code cell")
            continue
        first = _strip_prose(cells[0])
        # The family literal is a string, so read it BEFORE prose stripping blanks it out.
        raw_first = cells[0]
        if not _CELL1_CONFIG_RE.search(first):
            problems.append(f"{family}: cell 1 does not construct an EdaConfig")
        if not _CELL1_SETUP_RE.search(first):
            problems.append(f"{family}: cell 1 does not call notebook_setup")
        declared = _CELL1_FAMILY_RE.search(raw_first)
        if declared is None:
            problems.append(f"{family}: cell 1 sets no family=\"<top>/<sub>\"")
        elif declared.group(1).strip().strip("/") != family:
            problems.append(
                f"{family}: cell 1 declares family={declared.group(1)!r} but the notebook lives at "
                f"{_rel(path)} -- it would write into another family's results folder")
    assert not problems, "; ".join(problems)
    return f"{len(present)} notebook(s) satisfy the cell-1 contract"


def _notebook_symbol_refs() -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Scan the family notebooks for ``<submodule>.<attr>(`` calls and package-level calls.

    Returns:
        ``({submodule: {attr, ...}}, {package_level_attr, ...})``.

    Notes:
        The negative lookbehind ``(?<![\\w.])`` keeps ``plotting.stats.foo`` from being read as a
        ``stats`` reference. A dotted ``eda_analysis.<mod>.<attr>`` form is matched separately, as
        is whatever alias the notebook imported the package under (``import eda_analysis as E``).
    """
    refs: Dict[str, Set[str]] = {m: set() for m in _REF_MODULES}
    package_refs: Set[str] = set()
    mods = "|".join(_REF_MODULES)
    bare = re.compile(r"(?<![\w.])(" + mods + r")\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")

    for _family, path in _existing_family_notebooks():
        doc = _read_notebook(path)
        source = "\n".join(_code_cells(doc))
        aliases = {"eda_analysis"} | set(
            re.findall(r"import\s+eda_analysis\s+as\s+([A-Za-z_]\w*)", source))
        code = _strip_prose(source)
        for mod, attr in bare.findall(code):
            refs[mod].add(attr)
        for alias in aliases:
            qualified = re.compile(
                r"\b" + re.escape(alias) + r"\.(" + mods + r")\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
            for mod, attr in qualified.findall(code):
                refs[mod].add(attr)
            top_level = re.compile(
                r"\b" + re.escape(alias) + r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
            for attr in top_level.findall(code):
                if attr not in _REF_MODULES:
                    package_refs.add(attr)
    return refs, package_refs


def _c_notebook_symbol_refs() -> str:
    """Every symbol a notebook calls exists NOW -- the cheapest possible rename guard.

    A renamed or deleted function is invisible until the notebook that calls it runs, and a render
    is minutes per family. This is a source-level scan, so it costs milliseconds and it fails with
    the exact ``module.attr`` that moved.

    Warning:
        It sees only ``module.attr(`` call syntax. A symbol reached through a local alias
        (``fn = plotting.score_trajectory``) or a bare attribute access is not covered -- this
        narrows the surface, it does not close it.
    """
    if not _existing_family_notebooks():
        raise _Skip("no family notebooks on disk yet")

    refs, package_refs = _notebook_symbol_refs()
    bad: List[str] = []
    total = 0
    for mod, attrs in refs.items():
        module = importlib.import_module(f".{mod}", _PKG_NAME)
        for attr in sorted(attrs):
            total += 1
            if not hasattr(module, attr):
                bad.append(f"{mod}.{attr}")
    for attr in sorted(package_refs):
        total += 1
        if not hasattr(E, attr):
            bad.append(f"{_PKG_NAME}.{attr}")
    assert not bad, (
        f"{len(bad)} notebook-referenced symbol(s) do not exist: {sorted(bad)} -- a render would "
        f"die on these. Either restore the name or update the notebook.")
    used = {m: len(a) for m, a in refs.items() if a}
    return f"{total} notebook symbol refs resolve across {used or '{}'}" + (
        f" + {len(package_refs)} package-level" if package_refs else "")


#: Seaborn plotters whose ``errorbar`` DEFAULTS to ``("ci", 95)`` -- a 1,000-resample bootstrap.
#: A callsite that never names ``errorbar`` still draws a randomised CI, which is exactly the case
#: a search for the word "errorbar" cannot see.
_SNS_BOOTSTRAP_FUNCS = frozenset({"lineplot", "barplot", "pointplot", "catplot", "relplot"})

#: ``errorbar`` values that are computed analytically and consume no randomness.
_ANALYTIC_ERRORBAR = frozenset({None, "se", "sd"})


def _dotted_name(node) -> str:
    """``sns.lineplot`` / ``np.random.default_rng`` for a call's ``func``; ``""`` if dynamic."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _c_seeded_bootstrap() -> str:
    """Every resampling callsite is seeded, so two renders of unchanged data are byte-identical.

    Seaborn's ``lineplot`` / ``barplot`` / ``pointplot`` default to ``errorbar=("ci", 95)``, a
    1,000-sample bootstrap with ``seed=None``. A callsite that never NAMES ``errorbar`` still draws
    a randomised CI, which is why Exp3's first reproducibility pass -- which seeded only the
    callsites that spelled ``errorbar`` out -- left a notebook rewriting 20 PNGs per render on
    identical data. Every tracked figure then churns in git, and the one figure that actually
    changed is invisible in the diff.

    Three things are asserted:

    1. every seaborn callsite that can bootstrap passes ``seed=`` (or opts out with an analytic
       ``errorbar=None|"se"|"sd"``);
    2. ``numpy.random.default_rng()`` is never called without a seed;
    3. ``stats.bootstrap_ci`` and ``stats.paired_contrast`` still DEFAULT to ``BOOT_SEED``, since
       every caller in the package relies on that default rather than passing a seed.

    Notes:
        The scan is an AST walk, not a text search. A text search cannot tell a CALL from a
        function SIGNATURE, and ``plotting.score_trajectory`` legitimately exposes
        ``errorbar=("ci", 95)`` as a parameter default that its own seeded ``sns.lineplot`` call
        then forwards -- a line-based check reports that as an offender and there is no way to
        silence it that does not also silence a real one.

        Source-level on purpose: the data-level symptom is "some PNGs differ between two identical
        renders", which is slow to notice and easy to blame on the data.
    """
    offenders: List[str] = []
    seen = rng_calls = 0
    for root, _dirs, files in os.walk(_HERE):
        if "__pycache__" in root:
            continue
        for name in sorted(f for f in files if f.endswith(".py")):
            if name == os.path.basename(__file__):
                continue                        # this file's own literals are not callsites
            path = os.path.join(root, name)
            tree = ast.parse(io.open(path, encoding="utf-8").read(), filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                dotted = _dotted_name(node.func)
                keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                where = f"{_rel(path)}:{node.lineno}"

                bootstraps = dotted.split(".")[-1] in _SNS_BOOTSTRAP_FUNCS and \
                    dotted.split(".")[0] in ("sns", "seaborn")
                if bootstraps or "errorbar" in keywords:
                    seen += 1
                    errorbar = keywords.get("errorbar")
                    analytic = (isinstance(errorbar, ast.Constant)
                                and errorbar.value in _ANALYTIC_ERRORBAR)
                    if analytic:
                        continue
                    seed = keywords.get("seed", _MISSING_KW)
                    if seed is _MISSING_KW:
                        offenders.append(f"{where} ({dotted or 'call'} bootstraps a CI with no "
                                         f"seed=)")
                    elif isinstance(seed, ast.Constant) and seed.value is None:
                        offenders.append(f"{where} ({dotted or 'call'} passes seed=None)")

                if dotted.endswith("default_rng"):
                    rng_calls += 1
                    if not node.args and "seed" not in keywords:
                        offenders.append(f"{where} (default_rng() with no seed)")

    from eda_analysis import stats as S
    for fn in (S.bootstrap_ci, S.paired_contrast):
        default = inspect.signature(fn).parameters["seed"].default
        if default != E.BOOT_SEED:
            offenders.append(f"stats.{fn.__name__} defaults to seed={default!r}, not BOOT_SEED")

    assert not offenders, (
        f"{len(offenders)} unseeded resampling site(s) -- renders would not be reproducible: "
        + "; ".join(offenders[:6]))
    return (f"{seen} bootstrap-capable plot callsite(s) seeded or analytic, {rng_calls} seeded "
            f"rng, stats defaults = BOOT_SEED ({E.BOOT_SEED})")


def _c_exports_routing() -> str:
    """Artifacts land under ``results/<top>/<sub>/`` and never under a judge segment.

    Run against a TEMP results root (``exports.RESULTS_DIR`` is swapped for the duration and
    restored), so the real tree is untouched. Asserts:

    * every ``save_*`` refuses without a family -- there is deliberately no bare-``results/``
      fallback, because an artifact written there is one no index points at;
    * leaves compose ``<top>/<sub>/{figures,tables}/[<group>/]`` and **no path component is a
      judge**. Exp3 nested ``<judge>/`` under its per-arm families; Exp4 puts both graders inside
      one table, so a directory named after one grader would assert something false about its own
      contents;
    * the number ledger MERGES rather than overwrites, so one re-run cell refreshes its own keys;
    * ``.xlsx`` bytes are identical across two saves of the same frame (the frozen-timestamp
      guarantee that keeps a render from churning 30 workbooks);
    * ``reset_results`` clears only the ACTIVE family and ``_guard_path`` refuses everything in
      ``PRESERVE`` -- that function is a recursive delete living beside hand-authored prose.
    """
    import shutil
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from eda_analysis import exports as X
    from eda_analysis.constants import DEFAULT_JUDGE_TAG, judge_dirname

    judge_segments = {f"judge={DEFAULT_JUDGE_TAG}", DEFAULT_JUDGE_TAG, judge_dirname("")}
    real_root = X.RESULTS_DIR
    tmp = tempfile.mkdtemp(prefix="exp4_exports_probe_")
    X.RESULTS_DIR = tmp
    written: List[str] = []
    try:
        X.set_family("")
        for fn, args in ((X.save_table, (pd.DataFrame({"a": [1]}), "t")),
                         (X.save_numbers, ("n", {"k": 1})),
                         (X.save_fig, (plt.figure(), "f")),
                         (X.build_index, ()),
                         (X.reset_results, ())):
            try:
                fn(*args)
            except X.NoFamilyError:
                pass
            else:
                raise AssertionError(
                    f"exports.{fn.__name__} ran with NO family set -- a bare results/ fallback "
                    f"writes artifacts nowhere the index points at")
        plt.close("all")

        for family in E.all_families():
            top, sub = family.split("/")
            X.set_family(family)
            tables = X.save_table(pd.DataFrame({"a": [1, 2]}), "probe_table",
                                  caption="selfcheck probe table")
            written += tables
            assert os.path.dirname(tables[0]) == os.path.join(tmp, top, sub, "tables"), tables
            fig, ax = plt.subplots()
            ax.plot([0, 1])
            figs = X.save_fig(fig, "probe_fig", caption="selfcheck probe figure")
            plt.close(fig)
            written += figs
            assert os.path.dirname(figs[0]) == os.path.join(tmp, top, sub, "figures"), figs
            written.append(X.save_numbers("probe_nums",
                                          {"a.b": 1.5, "c": {"value": 2, "source": "s"}}))

        X.set_family("arms/outcomes")
        leaf = os.path.join(tmp, "arms", "outcomes")
        grouped = X.save_table(pd.DataFrame({"a": [1]}), "grouped", group="per_metric")
        written += grouped
        assert os.path.dirname(grouped[0]) == os.path.join(leaf, "tables", "per_metric"), grouped

        for path in written:
            parts = os.path.relpath(path, tmp).replace("\\", "/").split("/")
            hits = [p for p in parts if p in judge_segments or p.startswith("judge=")]
            assert not hits, (
                f"artifact {_rel(path)} contains a judge path segment {hits} -- Exp4 has no "
                f"<judge>/ results level; put both graders inside the table instead")

        # Ledger merge: a second write refreshes its own keys and leaves the rest alone.
        ledger = os.path.join(leaf, "tables", "probe_nums.json")
        X.save_numbers("probe_nums", {"a.b": 9})
        doc = json.load(io.open(ledger, encoding="utf-8"))
        assert doc["numbers"]["a.b"]["value"] == 9 and "c" in doc["numbers"], \
            f"save_numbers overwrote the ledger instead of merging: {doc}"

        # Frozen workbook timestamps: re-saving an unchanged table must not change one byte.
        workbook = os.path.join(leaf, "tables", "outcomes.xlsx")
        before = io.open(workbook, "rb").read()
        X.save_table(pd.DataFrame({"a": [1, 2]}), "probe_table", caption="selfcheck probe table")
        after = io.open(workbook, "rb").read()
        assert before == after, \
            "re-saving an unchanged table changed the workbook bytes (a timestamp is leaking)"

        indexes = X.build_index()
        text = io.open(indexes[0], encoding="utf-8").read()
        assert "probe_fig.png" in text and "selfcheck probe figure" in text, text[:400]
        root_index = io.open(os.path.join(tmp, "INDEX.md"), encoding="utf-8").read()
        for family in E.all_families():
            assert f"`{family}`" in root_index, f"{family} missing from results/INDEX.md"

        os.makedirs(os.path.join(tmp, "arms"), exist_ok=True)
        summary = os.path.join(tmp, "arms", "SUMMARY.md")
        io.open(summary, "w", encoding="utf-8").write("hand-authored\n")
        for bad in (summary, os.path.join(tmp, "schematics"),
                    os.path.join(tmp, "..", "outside")):
            try:
                X._guard_path(bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"_guard_path allowed {bad!r}")

        sibling = os.path.join(tmp, "method", "contrast", "tables", "probe_table.md")
        X.reset_results()
        assert not os.path.isdir(os.path.join(leaf, "tables")), \
            "reset_results left the active family's tables leaf in place"
        assert os.path.isfile(sibling), \
            "reset_results deleted ANOTHER family's artifacts -- families must be independent"
        assert os.path.isfile(summary), "reset_results touched a PRESERVEd SUMMARY.md"
    finally:
        X.set_family("")
        X.RESULTS_DIR = real_root
        plt.close("all")
        shutil.rmtree(tmp, ignore_errors=True)
    return (f"{len(written)} probe artifacts under <top>/<sub>/{{figures,tables}}, no judge "
            f"segment; ledger merges; xlsx bytes stable; PRESERVE guarded; reset scoped")


def _c_arm_identity() -> str:
    """A grid of arm names round-trips through ``naming``, and no two configurations collide.

    The arm name is the ONLY channel between the trainer and the EDA: it is the conversations
    folder, the runs folder and the score-lake partition. Two distinct configurations that render
    to one name share all three, and the damage is silent -- a resume-by-skipping-existing scorer
    reports "already scored" against the other arm's numbers, and the contrast table that results
    is a blend of two policies with nothing to indicate it.

    The round trip is checked in both directions: ``parse(build(cfg))`` reproduces every field, and
    ``ArmInfo.experiment_name`` re-renders the same string.
    """
    from naming import ArmInfo, build_experiment_name, parse_experiment_name
    from roles import DEFAULT_ORACLE_MODEL, DEFAULT_PATIENT_MODEL

    models = (DEFAULT_ORACLE_MODEL, "gpt-4o-mini-2024-07-18", "claude-haiku-4-5")
    qtags = ("Q1Q2", "Q1", "WAI", "MISAT", "MITI")
    ids_by_qtag = {"Q1Q2": [1, 2], "Q1": [1], "WAI": [3], "MISAT": [6], "MITI": [7]}

    seen: Dict[str, tuple] = {}
    checked = 0
    for qtag in qtags:
        for k in (0, 5):
            for mcl in (2, 12, 30):
                for oracle in models:
                    for patient in (DEFAULT_PATIENT_MODEL, models[1]):
                        variants = [("GRPO", {"g": 8}), ("GRPO", {"g": 4})]
                        variants += [("PTO", {"m": 8, "mode": mode})
                                     for mode in ("greedy", "independent")]
                        for method, extra in variants:
                            name = build_experiment_name(
                                method, ids_by_qtag[qtag], k, mcl,
                                oracle_model=oracle, patient_model=patient, **extra)
                            info = parse_experiment_name(name)
                            checked += 1

                            assert info.experiment_name == name, \
                                f"{name!r} re-renders as {info.experiment_name!r}"
                            assert (info.method, info.qtag, info.k, info.mcl) == \
                                (method, qtag, k, mcl), f"{name!r} decoded as {info}"
                            if method == "GRPO":
                                assert info.g == extra["g"] and info.m is None \
                                    and info.mode is None, f"{name!r} decoded as {info}"
                            else:
                                assert info.m == extra["m"] and info.g is None, \
                                    f"{name!r} decoded as {info}"
                                assert info.mode == ("greedy" if extra["mode"] == "greedy"
                                                     else "indep"), f"{name!r} decoded as {info}"

                            identity = (method, qtag, k, mcl, extra.get("g"), extra.get("m"),
                                        info.mode, info.oracle_tag, info.patient_tag)
                            if name in seen and seen[name] != identity:
                                raise AssertionError(
                                    f"arm-name COLLISION: {seen[name]} and {identity} both render "
                                    f"as {name!r} -- they would share one conversations folder and "
                                    f"one score partition")
                            seen[name] = identity

    assert len(seen) == checked, f"{checked} configurations produced only {len(seen)} names"

    for junk in ("", "GRPO_Q1Q2_LA5_MCL12_G8", "GRPO4_NOPE_LA5_MCL12_G8_Ogemma4E2B_Patgemma4E2B",
                 "GRPO4_Q1Q2_LA5_MCL12_G8_Ogemma4E2B", "model_iter_0"):
        try:
            parse_experiment_name(junk)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_experiment_name accepted a non-arm name {junk!r}")

    # Cross-field consistency is enforced on EVERY construction path, not only on parse.
    for bad_kwargs in ({"method": "GRPO", "m": 8, "mode": "greedy", "g": None},
                       {"method": "PTO", "g": 8, "m": None, "mode": None}):
        try:
            ArmInfo(qtag="Q1Q2", k=0, mcl=12, oracle_tag="x", patient_tag="y", **bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ArmInfo accepted an inconsistent arm: {bad_kwargs}")
    return f"{checked} arm names round-trip, all distinct; malformed names rejected"


def _c_no_torch() -> str:
    """Importing the EDA must not pull in torch, transformers, peft or trl.

    The EDA reads finished artifacts. If something here needs a GPU it is in the wrong package --
    and the cost is not only import time: on the local Blackwell card ``import trl`` after torch
    segfaults at CUDA init, so an EDA that drags in trainer code makes import ORDER load-bearing in
    a process that has no reason to care.

    EVERY module is imported first -- the analysis ones and the paid ``scoring`` -- so the check is
    independent of which other checks happened to run before it.
    """
    modules = _PACKAGE_MODULES + _PAID_MODULES
    for name in modules:
        importlib.import_module(f".{name}", _PKG_NAME)

    loaded = {m.split(".")[0] for m in sys.modules}
    forbidden = sorted(r for r in _FORBIDDEN_ROOTS if r in loaded)
    assert not forbidden, (
        f"importing {_PKG_NAME} loaded {forbidden} -- the EDA must never import trainer/GPU code. "
        f"Move whatever needs it into code/, or import it lazily inside the function that does.")
    core_hits = sorted(m for m in _FORBIDDEN_CORE if m in sys.modules)
    assert not core_hits, (
        f"importing {_PKG_NAME} loaded {core_hits} -- these are the GPU side of the trainer; the "
        f"EDA may import core.config / core.conversations / core.oracle / core.recorder / "
        f"core.timing only.")
    return f"{len(modules)} modules imported, none of {list(_FORBIDDEN_ROOTS)} loaded"


_RANKING_RE = re.compile(r"\.(idxmax|idxmin|argmax|argmin|nlargest|nsmallest)\s*\(")
_ORIENTATION_TOKENS = ("sign_of", "metric_sign", "higher_is_better", "is_lower_better",
                       "LOWER_IS_BETTER", "sign")


def _c_mici_orientation() -> str:
    """MICI is registered lower-is-better, and no ranking in the package ignores that.

    MICI counts MI-INCONSISTENT therapist behaviour, so an improvement is a DECREASE. It is the one
    instrument of nine whose sign is inverted, which makes it exactly the one a call site forgets:
    an ``idxmax`` over a score column picks the WORST checkpoint on MICI and nothing raises, the
    number is real, the table renders, and the conclusion is backwards.

    Two halves:

    1. the registry says so (``higher_is_better`` False, ``sign_of`` -1, ``stats`` agrees), and
       ``orient_contrast`` flips the delta AND swaps the CI ends -- negating ``ci_lo``/``ci_hi`` in
       place is the obvious one-liner and it produces ``lo > hi``, which plots as a backwards
       whisker;
    2. no ranking call in the package (``idxmax``/``argmin``/``nlargest``/...) sits in a block that
       never mentions orientation. Today there are none, so this starts guarding the moment
       somebody writes the first "best iteration" selection.
    """
    from eda_analysis import stats as S

    mici = E.metric("MICI")
    assert mici.higher_is_better is False, "constants.METRICS['MICI'] must be lower-is-better"
    assert mici.sign == -1 and E.sign_of("MICI") == -1, "sign_of('MICI') must be -1"
    assert E.is_lower_better("MICI"), "is_lower_better('MICI') must be True"
    assert S.higher_is_better("MICI") is False, "stats.higher_is_better('MICI') must be False"
    assert S.metric_sign("MICI") == -1, "stats.metric_sign('MICI') must be -1"
    for key in ("Q1Q2", "Q1", "Q2", "WAI_SR", "CSQ8", "MI_SAT", "MITI", "PCT"):
        assert S.metric_sign(key) == 1, f"{key} must be higher-is-better"

    contrast = {"mean_delta": -0.30, "dz": -0.5, "ci_lo": -0.50, "ci_hi": -0.10}
    oriented = S.orient_contrast(contrast, "MICI")
    assert oriented["gain"] == 0.30 and oriented["improved"] is True, oriented
    assert oriented["gain_ci_lo"] == 0.10 and oriented["gain_ci_hi"] == 0.50, (
        f"orient_contrast did not SWAP the interval ends on a lower-is-better metric: {oriented}")
    assert oriented["mean_delta"] == -0.30, "orient_contrast must leave the raw delta untouched"

    offenders: List[str] = []
    ranking = 0
    for root, _dirs, files in os.walk(_HERE):
        if "__pycache__" in root:
            continue
        for name in sorted(f for f in files if f.endswith(".py")):
            if name == os.path.basename(__file__):
                continue
            path = os.path.join(root, name)
            lines = io.open(path, encoding="utf-8").read().splitlines()
            for i, line in enumerate(lines):
                if line.lstrip().startswith("#") or not _RANKING_RE.search(line):
                    continue
                ranking += 1
                block = "\n".join(lines[max(0, i - 8):i + 8])
                if not any(tok in block for tok in _ORIENTATION_TOKENS):
                    offenders.append(f"{_rel(path)}:{i + 1}")
    assert not offenders, (
        f"{len(offenders)} ranking callsite(s) with no orientation in scope: {offenders} -- "
        f"multiply by constants.sign_of(metric) before an argmax, or MICI ranks backwards.")
    return (f"MICI lower-is-better across registry+stats, CI ends swap on orientation, "
            f"{ranking} ranking callsite(s) all orientation-aware")


# ==============================================================================
#  Data checks
# ==============================================================================


def _stored_metrics() -> List[str]:
    """Metric keys that actually have a ``metric=<M>`` partition (composites excluded)."""
    return [k for k in E.METRIC_ORDER if E.ALL_METRICS[k].partition]


def _arms_or_skip():
    """Discovered arms, or SKIP with a reason -- ``data/`` is a Drive symlink and may be absent."""
    if not os.path.isdir(E.DATA_DIR):
        raise _Skip(f"no data/ directory at {E.DATA_DIR} (Drive symlinks not created)")
    arms = E.discover_arms()
    if not arms:
        raise _Skip("no arms with conversations on disk yet")
    return arms


def _score_partitions(arms,
                      judges: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Enumerate the score lake against what is on disk.

    Returns:
        ``([(path, descriptor), ...] for partitions that exist, [descriptor, ...] for those that
        do not)``, where a descriptor is the compact ``judge/arm/model_iter_N/metric`` form used in
        every message here -- a full lake path is ~150 characters and unreadable in a summary line.
    """
    existing: List[Tuple[str, str]] = []
    missing: List[str] = []
    for judge in judges:
        for arm in arms:
            for state in arm.iters:
                for metric in _stored_metrics():
                    path = arm.score_path(state, metric, judge=judge, rep=0)
                    desc = f"{judge}/{arm.label}/model_iter_{state}/{metric}"
                    if os.path.isfile(path):
                        existing.append((path, desc))
                    else:
                        missing.append(desc)
    return existing, missing


def _c_score_coverage() -> str:
    """Every model state on disk is scored, on every metric, by every grader in the lake.

    This is the one gap no other check can see: the conversations are there, the compute axis
    reports the full trained range, every table renders -- and a contrast is silently truncated to
    the states that happen to have scores. In Exp3 exactly that shipped (an arm scored to iteration
    6 while its conversations ran to 7) and the only symptom was a row count nobody had a reason to
    check.

    WARNS rather than fails: an unscored state is the normal condition between a training run and a
    scoring run. The point is that it is stated out loud, next to the numbers.
    """
    arms = _arms_or_skip()
    judges = E.data.judge_tags()
    if not judges:
        raise _Skip("score lake holds no judge= partition yet (nothing has been scored)")

    existing, missing = _score_partitions(arms, judges)
    total = len(existing) + len(missing)
    head = (f"{len(existing)}/{total} partitions present "
            f"({len(arms)} arms x {len(_stored_metrics())} metrics x {len(judges)} judge(s))")
    if missing:
        shown = "; ".join(missing[:4])
        more = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
        raise _Warn(f"{head} -- UNSCORED: {shown}{more}")
    return head


def _c_persona_coverage() -> str:
    """Every scored parquet carries the full grid of 96 distinct ``persona_id`` values.

    One parquet per model state, one row per persona, and ``persona_id`` is the join key for every
    paired contrast in the experiment. Two things break it, in opposite ways:

    * a SHORT partition means some conversations were never scored, so a paired contrast quietly
      drops those personas from both sides. That is normal mid-scoring, so it WARNS;
    * a DUPLICATE persona makes the join ambiguous -- a merge multiplies rows and the "pairs" stop
      being pairs. That is corruption, so it FAILS.

    Only the ``persona_id`` column is read, which keeps this cheap over the Drive mount.
    """
    import pandas as pd

    arms = _arms_or_skip()
    judges = E.data.judge_tags()
    if not judges:
        raise _Skip("score lake holds no judge= partition yet (nothing has been scored)")
    existing, _missing = _score_partitions(arms, judges)
    if not existing:
        raise _Skip("no score partitions on disk yet")

    short: List[str] = []
    dupes: List[str] = []
    unreadable: List[str] = []
    for path, desc in existing:
        try:
            frame = pd.read_parquet(path, columns=["persona_id"])
        except Exception:                                   # noqa: BLE001 -- engine or column gap
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:                        # noqa: BLE001
                unreadable.append(f"{desc} ({type(exc).__name__})")
                continue
        if "persona_id" not in frame.columns:
            unreadable.append(f"{desc} (no persona_id column)")
            continue
        ids = frame["persona_id"]
        if ids.duplicated().any():
            dupes.append(f"{desc} repeats "
                         f"{sorted(ids[ids.duplicated()].unique().tolist())[:4]}")
        elif ids.nunique() != _N_PERSONAS:
            short.append(f"{desc} has {ids.nunique()}/{_N_PERSONAS}")

    assert not dupes, (
        f"{len(dupes)} score partition(s) repeat a persona_id: {dupes[:3]} -- every pairing "
        f"downstream joins on persona_id, so these partitions are ambiguous.")
    assert not unreadable, (
        f"{len(unreadable)} score partition(s) could not be read for persona coverage: "
        f"{unreadable[:3]} -- a silently skipped partition is how biased missingness reaches a "
        f"headline number.")
    head = f"{len(existing)} partitions, all {_N_PERSONAS} personas, no duplicates"
    if short:
        raise _Warn(f"{len(existing) - len(short)}/{len(existing)} partitions complete -- "
                    f"PARTIAL: {'; '.join(short[:4])}")
    return head


def _c_timing_logs() -> str:
    """Every COMPLETED iteration has an append-only timing log, and resumes are surfaced.

    ``iteration_N/timing_sessions.jsonl`` is the ONLY timing record in Exp4 -- there is no mtime
    reconstruction and none is planned, so an iteration that never called ``log_session`` has a
    cost that cannot be recovered afterwards. A completed iteration (``adapter/`` present) with no
    log therefore FAILS: the compute family would report it as free.

    ``n_sessions_production > 1`` means the iteration was resumed. That WARNS -- it is a normal
    event, but it is also the flag any cost table must carry, because for such an iteration every
    per-PROCESS number in ``iteration_metadata.json`` is an undercount. ⚠ It is the PRODUCTION
    session count, not ``n_sessions``: the post-loop final-eval pass appends an ``eval_gen_s``-only
    session to the last training iteration of every healthy arm, so the raw count would warn on
    every completed arm and the warning would stop meaning anything.
    """
    from core.timing import cumulative_seconds

    arms = _arms_or_skip()
    completed = 0
    unlogged: List[str] = []
    resumed: List[str] = []
    for arm in arms:
        for iteration in arm.iterations_on_disk():
            if not os.path.isdir(arm.paths.adapter_dir(iteration)):
                continue                                   # in flight: nothing to account for yet
            completed += 1
            totals = cumulative_seconds(arm.iteration_dir(iteration))
            sessions = int(totals.get("n_sessions", 0) or 0)
            production = int(totals.get("n_sessions_production", 0) or 0)
            if sessions == 0:
                unlogged.append(f"{arm.label}/iteration_{iteration}")
            elif production > 1:
                resumed.append(f"{arm.label}/iteration_{iteration} ({production} production "
                               f"session(s) of {sessions}, "
                               f"{totals['production_s'] / 3600.0:.1f} GPU-h)")
    if not completed:
        raise _Skip("no completed iteration (none has an adapter/ yet)")
    assert not unlogged, (
        f"{len(unlogged)} completed iteration(s) have no timing_sessions.jsonl: {unlogged[:4]} -- "
        f"their cost is unrecoverable, and the compute family will report them as free.")
    head = f"{completed} completed iteration(s) all logged"
    if resumed:
        raise _Warn(f"{head}; {len(resumed)} RESUMED: {'; '.join(resumed[:3])} -- any per-process "
                    f"timing field for these is an undercount; quote the cumulative log")
    return head


def _c_render_freshness() -> str:
    """Rendered artifacts are newer than the data they were built from.

    The silent failure: scores land, nobody re-renders, and every table keeps rendering fine -- it
    just carries the previous grid. Nothing says so; the only symptom is a row count that has to be
    noticed by hand.

    Inputs are the enumerated score partitions plus each arm's per-iteration training artifacts
    (the timing log and ``generations.jsonl``, which the compute and training views read). A family
    that has never been rendered WARNS (a fresh clone, or before the first render); a family
    rendered BEFORE its newest input FAILS.
    """
    arms = _arms_or_skip()
    judges = E.data.judge_tags()
    existing, _missing = _score_partitions(arms, judges) if judges else ([], [])

    newest_input = 0.0
    for path, _desc in existing:
        try:
            newest_input = max(newest_input, os.path.getmtime(path))
        except OSError:
            pass
    for arm in arms:
        for iteration in arm.iterations_on_disk():
            for path in (arm.paths.timing_sessions_path(iteration),
                         arm.paths.generations_path(iteration)):
                try:
                    newest_input = max(newest_input, os.path.getmtime(path))
                except OSError:
                    pass
    if not newest_input:
        raise _Skip("no scores or training artifacts on disk to date a render against")

    def _newest_artifact(root: str) -> float:
        best = 0.0
        for kind in ("tables", "figures"):
            scope = os.path.join(root, kind)
            if not os.path.isdir(scope):
                continue
            for dirpath, _dirnames, filenames in os.walk(scope):
                for name in filenames:
                    if name == "CAPTIONS.md" or not name.lower().endswith(
                            (".md", ".json", ".xlsx", ".csv", ".png", ".pdf", ".svg")):
                        continue
                    try:
                        best = max(best, os.path.getmtime(os.path.join(dirpath, name)))
                    except OSError:
                        pass
        return best

    stale: List[str] = []
    checked = 0
    for family in E.all_families():
        top, sub = family.split("/")
        rendered = _newest_artifact(os.path.join(E.RESULTS_DIR, top, sub))
        if not rendered:
            continue
        checked += 1
        if rendered < newest_input:
            stale.append(f"{family} (rendered {int((newest_input - rendered) / 3600)}h before its "
                         f"newest input)")
    if not checked:
        raise _Warn("no family has been rendered yet -- run `python tools/render_results.py`")
    assert not stale, (
        "STALE results family/families: " + "; ".join(stale) +
        " -- re-render with `python tools/render_results.py`.")
    return f"{checked} rendered family/families newer than their inputs"


# ==============================================================================
#  Entry point
# ==============================================================================

#: ``(display name, callable, needs_data)`` in run order.
_CHECKS: Tuple[Tuple[str, Callable[[], str], bool], ...] = (
    ("imports + __all__ resolve", _c_imports, False),
    ("family map", _c_family_map, False),
    ("EdaConfig round-trip", _c_config_roundtrip, False),
    ("metric registry vs questionnaires", _c_metric_registry, False),
    ("palette parity (constants/plotting)", _c_palette_parity, False),
    ("partition tokens are legal paths", _c_partition_tokens, False),
    ("empty-frame contracts", _c_empty_frame_contracts, False),
    ("notebook cell-1 contract", _c_notebook_cell1, False),
    ("notebook symbol refs resolve", _c_notebook_symbol_refs, False),
    ("seeded bootstrap (repro figures)", _c_seeded_bootstrap, False),
    ("exports routing (no judge level)", _c_exports_routing, False),
    ("arm identity round-trip", _c_arm_identity, False),
    ("no torch in the EDA", _c_no_torch, False),
    ("MICI orientation", _c_mici_orientation, False),
    ("score coverage (disk vs lake)", _c_score_coverage, True),
    ("persona coverage (96 per parquet)", _c_persona_coverage, True),
    ("timing logs (completed iterations)", _c_timing_logs, True),
    ("render freshness", _c_render_freshness, True),
)

_MARKS = {"PASS": "OK  ", "SKIP": "skip", "WARN": "WARN", "FAIL": "FAIL"}


def main(argv: Optional[List[str]] = None) -> int:
    """Run the checks and print one line each. Returns 1 if anything FAILED, else 0.

    Args:
        argv: ``["--fast"]`` runs only the structural checks (no disk reads at all), which is what
            a pre-commit or a machine with no Drive mount wants. ``--trace`` prints tracebacks.

    Notes:
        The parquet memo is disabled for the whole run (``EDA_NO_CACHE``): the data checks validate
        GROUND TRUTH, and a cached frame could mask exactly the regression they exist to catch.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    fast = "--fast" in argv
    if "--trace" in argv:
        os.environ["SELFCHECK_TRACE"] = "1"
    unknown = [a for a in argv if a not in ("--fast", "--trace")]
    if unknown:
        print(f"usage: python -m {_PKG_NAME}._selfcheck [--fast] [--trace]  "
              f"(unknown argument(s): {unknown})")
        return 2

    os.environ["EDA_NO_CACHE"] = "1"

    results: _Results = []
    for name, fn, needs_data in _CHECKS:
        if fast and needs_data:
            continue
        _run(name, fn, results)

    width = max(len(name) for name, _, _ in results)
    print(f"\n {_PKG_NAME} self-check" + ("  [--fast: structural only]" if fast else ""))
    print(" " + "-" * (width + 34))
    for name, status, detail in results:
        print(f"  [{_MARKS[status]}] {name.ljust(width)}  {detail}")
    counts = {s: sum(1 for _, st, _ in results if st == s) for s in _MARKS}
    print(" " + "-" * (width + 34))
    print(f"  {counts['PASS']} passed, {counts['WARN']} warned, {counts['SKIP']} skipped, "
          f"{counts['FAIL']} failed")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
