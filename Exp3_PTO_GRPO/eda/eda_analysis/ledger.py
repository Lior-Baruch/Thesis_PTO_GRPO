"""ledger.py — the quotable-numbers primitives every ``*_numbers.json`` writer needs.

A **ledger** is the `{dotted.key: {value, source, note}}` document that
:func:`eda_analysis.exports.save_numbers` writes and each paper's ``NUMBERS.md`` cites
claim-by-claim. Nine analysis modules build one, and each needs the same three things: make a
value JSON-safe, wrap it in the record shape, and round it for quoting.

Why this module exists
----------------------
These arrived as **private copies in eight modules** when the paper generators were promoted into
the package (the deleted ``papers/…/analysis/_common.py`` had one of each). By then they had
already diverged on ``np.bool_``: ``tails``/``dispersion``/``faithfulness`` converted it with
``bool()``, ``crossgen``/``replication`` folded it into the ``.item()`` branch, and ``instruments``
handled it not at all — so the same value could serialize three ways depending on which ledger
wrote it, and one ledger could leak a raw ``np.bool_`` into JSON. :func:`json_scalar` is the union
of all three, so every previously-correct copy is bit-identical and the gap is closed.

It is a **leaf**: numpy + stdlib only, no package imports, so any module can import it at top level
without circular-import risk. It is deliberately NOT part of ``constants``, whose docstring
promises stdlib-only imports.

⚠ These set the precision and encoding of numbers the papers quote. Changing
:func:`round3`'s default, or what :func:`json_scalar` does with a NaN, changes published values —
the ``paper fixture anchors`` self-check is what will catch you.
"""

from __future__ import annotations

import numpy as np

__all__ = ["json_scalar", "ledger_entry", "round3"]


def json_scalar(v):
    """Recursively make a value JSON-safe: numpy scalars → Python, NaN → ``None``, keys → ``str``.

    ``json.dumps`` stringifies non-string dict keys anyway, so forcing ``str(k)`` here is
    behaviour-preserving for the copies that omitted it, and makes the in-memory ledger match the
    file it becomes.
    """
    if isinstance(v, dict):
        return {str(k): json_scalar(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_scalar(x) for x in v]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def ledger_entry(value, source: str = "", note: str = "") -> dict:
    """One ledger record: ``{"value": …, "source": …, "note": …}``.

    ``source`` is the artifact the number can be checked against — the field that makes a ledger
    auditable rather than a list of assertions. Keep it a real path or table name.
    """
    return {"value": json_scalar(value), "source": source, "note": note}


def round3(x, nd: int = 3):
    """Round to *nd* dp; ``None``/NaN → ``None``; anything non-numeric passes through unchanged.

    The pass-through is the safe half: the three copies this replaces disagreed on it, and one
    raised on a string where the others returned it.
    """
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return None if np.isnan(xf) else round(xf, nd)
