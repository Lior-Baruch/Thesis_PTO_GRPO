"""
personas.py — recover the TRUE patient persona behind each saved conversation.

The trainer simulates the same 96 personas every iteration but in a *seeded
shuffled order*, and saves each conversation under its shuffled position
(``conversation_{position}.csv``). So ``conversation_{i}.csv`` is a different
persona each iteration. The shuffle is deterministic:

    iter_rng = random.Random(cfg.seed + iteration)   # in-loop, saves model_iter_{iteration-1}
    final    = random.Random(cfg.seed + num_iterations + 1)  # saves model_iter_{N}

which collapses to a single rule: **model_iter_k uses shuffle seed
``seed + k + 1``**. Replaying ``Random(seed+k+1).shuffle(list(range(96)))``
reproduces ``order`` where ``order[file_index] = canonical_persona_id`` (the
canonical id = the index into ``generate_all_permutations()``, the same order
``get_patient_permutation_characteristics`` uses).

Validated against conversation content (turn-1 age/gender) — exact.
"""

import os
import random
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

from . import PERSONA_COLS


# ── Canonical persona table (the unshuffled 96) ──────────────────────────────
@lru_cache(maxsize=1)
def canonical_personas(n: int = 96) -> pd.DataFrame:
    """The 96 canonical personas + characteristics, indexed by ``persona_id`` 0..n-1.

    Columns: :data:`PERSONA_COLS` (gender, age_value, problem, problem_time,
    tried_to_solve, cooperation_level).
    """
    from system_prompts_builder import get_patient_permutation_characteristics
    rows = []
    for pid in range(n):
        ch = get_patient_permutation_characteristics(pid) or {}
        row = {"persona_id": pid}
        row.update({c: ch.get(c) for c in PERSONA_COLS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("persona_id")


# ── The shuffle replay ───────────────────────────────────────────────────────
def persona_order(seed: int, model_iter: int, n: int = 96) -> List[int]:
    """``order`` where ``order[file_index] == canonical_persona_id`` for ``model_iter``.

    Reproduces the trainer's ``shuffled = list(all_permutations); rng.shuffle(shuffled)``
    by replaying the identical seeded shuffle on ``list(range(n))`` (the swap
    sequence is content-independent, so this is exact).
    """
    order = list(range(n))
    random.Random(seed + model_iter + 1).shuffle(order)
    return order


def file_to_persona(seed: int, model_iter: int, n: int = 96) -> Dict[int, int]:
    """``{file_index: canonical_persona_id}`` for ``model_iter``."""
    return {i: pid for i, pid in enumerate(persona_order(seed, model_iter, n))}


def attach_personas(
    df: pd.DataFrame,
    seed: int,
    *,
    iter_col: str = "iteration",
    file_col: str = "file_index",
    n: int = 96,
) -> pd.DataFrame:
    """Add ``persona_id`` + :data:`PERSONA_COLS` to *df* by replaying the per-iter shuffle.

    *df* must carry an iteration column (``model_iter`` k) and a file-index column
    (the saved ``conversation_{i}`` / ``{patient_id}.csv`` index). One ``seed`` per
    call — split by arm before calling if arms differ in seed (they don't today).
    """
    out = df.copy()
    cano = canonical_personas(n)
    # Build the (iter, file_index) -> persona_id map only for the iters present.
    pid = []
    cache: Dict[int, List[int]] = {}
    for it, fi in zip(out[iter_col].astype(int), out[file_col].astype(int)):
        order = cache.get(it)
        if order is None:
            order = persona_order(seed, it, n)
            cache[it] = order
        pid.append(order[fi] if 0 <= fi < n else -1)
    out["persona_id"] = pid
    chars = cano.reindex(out["persona_id"].values).reset_index(drop=True)
    for c in PERSONA_COLS:
        out[c] = chars[c].values
    return out


# ── Gating validation ────────────────────────────────────────────────────────
def validate_recovery(
    conv_dir_for_iter,
    seed: int,
    iters: List[int],
    *,
    n: int = 96,
    sample_every: int = 8,
    verbose: bool = True,
) -> dict:
    """Assert the replay is sound and (optionally) matches conversation content.

    ``conv_dir_for_iter(k)`` -> absolute path to ``model_iter_k``'s conversation
    folder (so this stays IO-agnostic). Checks per iter that recovered ids form a
    full 0..n-1 permutation, and that the age stated in the patient's first turn
    matches the recovered persona on a sampled subset. Returns a small report;
    raises ``AssertionError`` if the permutation check fails.
    """
    import re
    cano = canonical_personas(n)
    age_ok = age_tot = 0
    for k in iters:
        order = persona_order(seed, k, n)
        assert sorted(order) == list(range(n)), (
            f"persona recovery for model_iter_{k} is not a permutation of 0..{n-1} "
            f"(seed={seed}); the seed/order assumption is wrong."
        )
        cdir = conv_dir_for_iter(k)
        if not cdir or not os.path.isdir(cdir):
            continue
        for fi in range(0, n, sample_every):
            fp = os.path.join(cdir, f"conversation_{fi}.csv")
            if not os.path.exists(fp):
                continue
            try:
                cdf = pd.read_csv(fp)
            except Exception:
                continue
            pt = cdf[cdf["role"] == "patient"]["conversation"]
            if not len(pt):
                continue
            m = re.search(r"\b(\d{2})\b", str(pt.iloc[0])[:120])
            if not m:
                continue
            age_tot += 1
            if str(cano.loc[order[fi], "age_value"]) == m.group(1):
                age_ok += 1
    rep = {
        "iters_checked": list(iters),
        "permutation_ok": True,
        "age_match": (age_ok, age_tot),
        "age_match_rate": (age_ok / age_tot) if age_tot else None,
    }
    if verbose:
        rate = rep["age_match_rate"]
        print(f"persona recovery: permutation OK for iters {list(iters)}; "
              f"age-in-intro match {age_ok}/{age_tot}"
              + (f" ({rate:.0%})" if rate is not None else ""))
    return rep
