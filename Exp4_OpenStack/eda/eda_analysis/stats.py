"""stats.py -- the small set of statistics every Exp4 family shares.

Exp4's design hands the analysis one big gift: **the same 96 personas face every arm, every
iteration, forever.** Persona 7 is ``pers07.csv`` in ``model_iter_0`` of the GRPO K=0 arm and in
``model_iter_5`` of the PTO K=5 arm -- the same patient, the same opening problem, the same
cooperation level. So every contrast in this experiment is a *repeated-measures* contrast, and the
right way to compute one is to subtract within persona and analyse the deltas.

Why that matters more than it sounds
------------------------------------
Persona variance dominates. The 96 personas span cooperative to actively resistant clients, and the
spread of scores *across personas* is far larger than the spread *between two policies on the same
persona*. Pairing removes that variance from the standard error; not pairing buries the effect in
it. Concretely: a between-groups test on the same data can leave a real effect non-significant, and
an effect size computed unpaired answers a different question than ``dz``.

**Pair on ``persona_id``. Never on row order, file position, or a row index.**
Exp3 could not: its conversation files were named by a per-iteration *shuffled* processing index, so
``conversation_3.csv`` was a different person in every iteration and every module had to replay
``Random(seed + k + 1)`` to recover the mapping. Exp4 stores ``persona_id`` in the file name and in
the CSV, so the join key is simply there. If you nonetheless pair by position -- ``df_a["score"].values
- df_b["score"].values`` -- you subtract unrelated conversations. The *mean* survives that mistake
(subtracting a permutation of the same 96 values gives the same mean difference), which is exactly
what makes it dangerous: nothing looks wrong. ``dz``, the bootstrap CI and the p-value are all
garbage, and they are the numbers a claim rests on. :func:`paired_arrays` exists so no caller has to
hand-roll the join.

Reproducibility
---------------
Every resampling routine here takes ``seed=BOOT_SEED`` from :mod:`.constants` by default, so two
renders of the same notebook on the same data produce byte-identical tables. (Exp3 learned this the
hard way on the *figure* side: seaborn's ``errorbar=("ci", 95)`` defaults to ``seed=None`` and every
tracked PNG churned in git on each render.)

Orientation
-----------
Not every metric is higher-is-better. ``MICI`` counts MI-*inconsistent* therapist behaviour, so an
improvement is a **decrease** and a raw ``mean_delta`` of ``-0.3`` is a *gain*. Rather than trusting
each family to remember that, :func:`orient_contrast` turns a raw contrast into a signed *gain* by
consulting the metric registry in :mod:`.constants`. Use it wherever a table or a figure says
"better".

No scipy
--------
The distribution functions this module needs (Student-t tail, Spearman via ranks) are ~40 lines of
stdlib maths, and keeping them here means the EDA's dependency surface is numpy + pandas +
matplotlib. The t tail is the regularized incomplete beta by continued fraction; it agrees with
``scipy.stats.ttest_rel`` to ~1e-12 (see the module's verification notes in the EDA README).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import BOOT_SEED

__all__ = [
    "paired_arrays",
    "paired_contrast",
    "bootstrap_ci",
    "holm",
    "spearman",
    "cohens_dz",
    "effect_label",
    "summarize_contrasts",
    "higher_is_better",
    "metric_sign",
    "orient_contrast",
    "stars",
]


# ==============================================================================
#  Pairing
# ==============================================================================


def paired_arrays(df_a: pd.DataFrame,
                  df_b: pd.DataFrame,
                  *,
                  on: str = "persona_id",
                  value: str = "score") -> Tuple[np.ndarray, np.ndarray]:
    """Align two score frames on *on* and return the two matched value vectors.

    Args:
        df_a: Left frame -- one row per key. Typically one (arm, model state, metric) cell.
        df_b: Right frame, same shape requirement.
        on: The join key. **Leave this at ``persona_id``** unless you have a specific reason;
            see the module docstring for why positional pairing silently corrupts every
            dispersion-based statistic while leaving the mean intact.
        value: Column holding the number to compare.

    Returns:
        ``(a, b)`` -- two ``float`` arrays of equal length, ordered by the sorted join key, with
        any pair missing a value on either side dropped. The order is deterministic, so a
        downstream bootstrap is reproducible.

    Raises:
        KeyError: if either frame lacks *on* or *value*. Named explicitly rather than surfacing
            as a pandas merge error three frames later.
        ValueError: if either frame has more than one row per key. That means the caller has not
            filtered down to a single cell -- most often the metric filter was forgotten, so each
            persona appears once per rubric and the "pairs" would be a cross product. Silently
            aggregating would hide the mistake and change what the contrast measures.

    Notes:
        An inner join: personas present on only one side are dropped, and the returned length is
        the honest ``n``. That is the correct behaviour for a partially-scored arm (an iteration
        still being graded), and it is why every routine here reports ``n`` alongside the effect.
    """
    for name, frame in (("df_a", df_a), ("df_b", df_b)):
        if frame is None:
            raise KeyError(f"paired_arrays: {name} is None")
        missing = [c for c in (on, value) if c not in frame.columns]
        if missing:
            raise KeyError(
                f"paired_arrays: {name} has no column(s) {missing}; it has {list(frame.columns)}"
            )

    left = df_a[[on, value]].dropna(subset=[on])
    right = df_b[[on, value]].dropna(subset=[on])
    for name, frame in (("df_a", left), ("df_b", right)):
        dupes = frame[on].duplicated().sum()
        if dupes:
            raise ValueError(
                f"paired_arrays: {name} has {dupes} duplicate {on!r} value(s); expected one row "
                f"per key. Filter to a single (arm, model state, metric) cell first -- pairing a "
                f"frame that still holds several metrics would form a cross product, not pairs."
            )

    merged = (left.merge(right, on=on, how="inner", suffixes=("_a", "_b"))
                  .sort_values(on, kind="mergesort"))
    a = merged[f"{value}_a"].to_numpy(dtype=float)
    b = merged[f"{value}_b"].to_numpy(dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return a[ok], b[ok]


# ==============================================================================
#  Contrasts
# ==============================================================================


def cohens_dz(a, b) -> float:
    """Paired effect size: mean of the deltas over their SD (``ddof=1``).

    ``dz`` is the repeated-measures effect size and it is NOT comparable to a between-groups
    ``d``: its denominator is the SD of the *differences*, which shrinks as the pairing gets more
    informative. A large ``dz`` therefore means "the direction is consistent across personas", not
    "the two distributions barely overlap".

    Returns ``nan`` for fewer than two pairs or a zero delta SD (every persona moved by the exact
    same amount -- real only in degenerate data, so ``nan`` is safer than ``inf``).
    """
    d = _deltas(a, b)
    if d.size < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    return float(d.mean() / sd) if sd > 0 else float("nan")


def paired_contrast(a,
                    b,
                    *,
                    n_boot: int = 2000,
                    seed: int = BOOT_SEED,
                    alpha: float = 0.05) -> dict:
    """Full paired contrast of two aligned vectors. Sign convention: **``a - b``, so positive
    means *a* scored higher** (higher, not better -- see :func:`orient_contrast`).

    Args:
        a: Left vector, already aligned by :func:`paired_arrays`.
        b: Right vector.
        n_boot: Bootstrap resamples for the CI of the mean delta.
        seed: Resampling seed. Defaults to :data:`constants.BOOT_SEED` so a re-render reproduces
            the table exactly; pass something else only in a deliberate seed-sensitivity check.
        alpha: Two-sided CI level (``0.05`` -> 95%).

    Returns:
        ``{"n", "mean_delta", "sd_delta", "dz", "ci_lo", "ci_hi", "t", "p"}``.

        * ``t``/``p`` are the paired t-test (equivalently, a one-sample t-test on the deltas),
          two-sided, ``df = n - 1``.
        * ``ci_lo``/``ci_hi`` are a percentile bootstrap of the mean delta -- deliberately not the
          t interval, because rubric scores are bounded 1--5 and skewed near the ceiling, where the
          normal-theory interval overshoots.

        With fewer than 3 pairs every statistic is ``nan`` except ``n``: reporting an effect size
        off two personas is worse than reporting nothing. When every delta is exactly zero, ``p``
        is 1.0 and ``dz`` is ``nan`` (no variation to standardize by).

    Notes:
        The bootstrap resamples PAIRS (the deltas), not the two vectors independently -- that is
        what preserves the pairing inside the interval.
    """
    d = _deltas(a, b)
    n = int(d.size)
    out = {"n": n, "mean_delta": float("nan"), "sd_delta": float("nan"), "dz": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan"), "t": float("nan"), "p": float("nan")}
    if n < 3:
        if n:
            out["mean_delta"] = float(d.mean())
        return out

    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    out["mean_delta"] = mean
    out["sd_delta"] = sd

    if sd > 0:
        out["dz"] = mean / sd
        t = mean / (sd / math.sqrt(n))
        out["t"] = float(t)
        out["p"] = _t_two_sided_p(t, n - 1)
    else:
        # Every delta identical. Zero deltas => no effect (p = 1); a constant non-zero shift is a
        # deterministic difference, which a t-test cannot express -- report p = 0 rather than nan.
        out["t"] = 0.0 if mean == 0 else float("inf") * (1.0 if mean > 0 else -1.0)
        out["p"] = 1.0 if mean == 0 else 0.0

    lo, hi = bootstrap_ci(d, np.mean, n_boot=n_boot, seed=seed, alpha=alpha)
    out["ci_lo"], out["ci_hi"] = lo, hi
    return out


def bootstrap_ci(x,
                 stat: Callable[..., Any] = np.mean,
                 *,
                 n_boot: int = 2000,
                 seed: int = BOOT_SEED,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for *stat* of a one-dimensional sample.

    Args:
        x: The sample (NaNs dropped). For a paired contrast this is the vector of DELTAS -- pass
           ``a - b``, never the two vectors separately, or the interval loses the pairing.
        stat: Statistic to resample. Anything numpy-shaped works; ``np.mean`` and ``np.median``
            take the vectorized fast path.
        n_boot: Number of resamples.
        seed: Defaults to :data:`constants.BOOT_SEED`. Two calls with the same seed, sample and
            ``n_boot`` return identical bounds -- that is the contract that keeps rendered tables
            stable across runs.
        alpha: Two-sided level; ``0.05`` -> the 2.5th and 97.5th percentiles.

    Returns:
        ``(lo, hi)``, or ``(nan, nan)`` for an empty sample.

    Notes:
        Percentile, not BCa: with n = 96 and a near-symmetric delta distribution the bias
        correction moves the bounds less than the third decimal, and BCa needs a jackknife pass
        per cell, which is the whole cost of the compute family's budget sweep.
    """
    d = np.asarray(x, dtype=float).ravel()
    d = d[~np.isnan(d)]
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    resamples = d[idx]
    try:
        boots = np.asarray(stat(resamples, axis=1), dtype=float)
    except TypeError:                                   # a statistic with no axis= support
        boots = np.asarray([float(stat(row)) for row in resamples], dtype=float)
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return (float(lo), float(hi))


# ==============================================================================
#  Multiplicity
# ==============================================================================


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Args:
        pvalues: Raw p-values. ``nan`` entries are passed through as ``nan`` and are NOT counted
            in the family size -- a cell that could not be tested must not inflate the correction
            applied to the cells that could.

    Returns:
        Adjusted p-values as a float array, enforced non-decreasing along the sorted order (the
        step-down monotonicity requirement) and clipped at 1.0.

    Warning:
        **The family is whatever you pass in.** Holm over the rubrics of one arm is a different
        correction than Holm over arm x rubric, and concatenating two per-arm corrected tables does
        NOT produce a jointly corrected one. State the family wherever the column is shown.
    """
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx], kind="mergesort")]
    m = int(idx.size)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * float(p[i]))
        running = max(running, adj)
        out[i] = running
    return out


def stars(p: float) -> str:
    """Significance marker for a p-value: ``***`` < .001, ``**`` < .01, ``*`` < .05, else ``""``.

    A reading aid for a rendered table, never a decision rule -- the effect size and its CI are
    what a claim should quote.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    p = float(p)
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def effect_label(d: float) -> str:
    """Cohen-style magnitude label for ``|dz|``: negligible / small / medium / large.

    Thresholds 0.2 / 0.5 / 0.8. Conventional, not calibrated to this task -- a "small" effect on a
    1--5 rubric graded by an LLM may still be the largest thing the instrument can resolve.
    """
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return ""
    a = abs(float(d))
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def summarize_contrasts(rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
                        *,
                        p_col: str = "p") -> pd.DataFrame:
    """Collect contrast dicts into one frame and add ``p_holm`` + ``stars``.

    Args:
        rows: An iterable of :func:`paired_contrast`-shaped dicts (usually with extra descriptive
            keys like ``arm``/``metric``/``iteration`` merged in), or an existing DataFrame.
        p_col: Which column holds the raw p-values.

    Returns:
        A copy with ``p_holm`` (Holm-adjusted over the rows PASSED IN) and ``stars`` (read off
        ``p_holm``, not off the raw p). Returns an empty frame unchanged; a frame without *p_col*
        comes back with neither column added rather than raising, so a caller assembling a
        descriptive table can reuse this.

    Warning:
        The correction family is exactly the row set handed over. Build the frame you intend to
        correct as a family -- do not correct per arm and then concatenate, and do not correct a
        frame that mixes a primary contrast with its diagnostics.
    """
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty or p_col not in df.columns:
        return df
    df["p_holm"] = holm(df[p_col].to_numpy(dtype=float))
    df["stars"] = [stars(p) for p in df["p_holm"]]
    return df


# ==============================================================================
#  Correlation
# ==============================================================================


def spearman(x, y) -> float:
    """Spearman rank correlation -- ranks (ties averaged) then Pearson. No scipy.

    Pairs with a NaN on either side are dropped. Returns ``nan`` for fewer than three complete
    pairs or when either ranked vector is constant (no variance to correlate).

    Notes:
        Rank-then-Pearson is the definition, and it handles ties correctly *because* the ranks are
        averaged -- the shortcut ``1 - 6*sum(d^2)/(n^3-n)`` is only valid without ties, and score
        columns on a 1--5 rubric are full of them.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"spearman: length mismatch ({a.size} vs {b.size})")
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


# ==============================================================================
#  Orientation -- which direction is "better"
# ==============================================================================

_ORIENTATION_CACHE: dict = {}


def higher_is_better(metric: str, *, default: Optional[bool] = None) -> bool:
    """Does a LARGER value of *metric* mean a better therapist?

    True for every rubric on the 1--5 satisfaction/alliance scale and for ``PCT`` (patient
    change-talk share). **False for ``MICI``**, which counts MI-inconsistent therapist behaviour:
    an improvement there is a decrease, so a "gain" is a negative ``mean_delta``.

    Args:
        metric: Metric key as it appears in the score lake (``metric=<M>`` path level).
        default: What to return for a metric the registry does not know. ``None`` (the default)
            raises instead -- an unrecognised key is usually a typo, and quietly assuming
            higher-is-better is precisely the silent sign error this function exists to prevent.
            Pass ``default=True`` for a derived quantity you own the orientation of.

    Raises:
        KeyError: unknown metric and no *default*.
        RuntimeError: :mod:`.constants` exposes no metric registry at all.

    Notes:
        The registry lives in :mod:`.constants` (the package leaf). This function is the ONE place
        that reads it, so the orientation of a metric is defined once and every family inherits it.
        Do not re-derive it from a hard-coded set at a call site.
    """
    key = str(metric)
    if key in _ORIENTATION_CACHE:
        known = _ORIENTATION_CACHE[key]
    else:
        known = _lookup_orientation(key)
        if known is not None:
            _ORIENTATION_CACHE[key] = known
    if known is None:
        if default is None:
            raise KeyError(
                f"higher_is_better: metric {metric!r} is not in the constants metric registry. "
                f"Add it there (one definition, every family inherits it), or pass an explicit "
                f"default= if you own this quantity's orientation."
            )
        return bool(default)
    return bool(known)


def metric_sign(metric: str, *, default: Optional[bool] = None) -> int:
    """``+1`` when a larger value of *metric* is better, ``-1`` when a smaller one is.

    Multiply a raw delta by this to turn it into a GAIN. See :func:`higher_is_better` for the
    *default* argument.
    """
    return 1 if higher_is_better(metric, default=default) else -1


def orient_contrast(contrast: Mapping[str, Any],
                    metric: str,
                    *,
                    default: Optional[bool] = None) -> dict:
    """Re-express a :func:`paired_contrast` result as a signed **gain** for *metric*.

    Args:
        contrast: A ``paired_contrast``-shaped mapping (needs at least ``mean_delta``; ``dz``,
            ``ci_lo`` and ``ci_hi`` are oriented too when present).
        metric: The metric the contrast was computed on.
        default: Passed to :func:`higher_is_better` for unregistered metrics.

    Returns:
        A copy of *contrast* plus ``gain``, ``gain_dz``, ``gain_ci_lo``, ``gain_ci_hi``,
        ``sign`` and ``improved``. A positive ``gain`` always means "better", on every metric.
        The raw ``mean_delta`` / ``dz`` / CI are left untouched so a table can still show the
        measured quantity next to its interpretation.

    Warning:
        Flipping an interval **swaps its ends**: the negated upper bound is the new lower bound.
        Negating ``ci_lo`` and ``ci_hi`` in place -- the obvious one-liner -- produces an interval
        with ``lo > hi`` that silently plots as a backwards whisker. This function does the swap.

    Notes:
        ``t`` and ``p`` are deliberately NOT oriented: the two-sided p-value is direction-free, and
        a sign-flipped ``t`` would invite a one-sided reading nobody planned for.
    """
    sign = metric_sign(metric, default=default)
    out = dict(contrast)
    out["sign"] = sign
    out["gain"] = _signed(contrast.get("mean_delta"), sign)
    out["gain_dz"] = _signed(contrast.get("dz"), sign)
    lo, hi = contrast.get("ci_lo"), contrast.get("ci_hi")
    if sign > 0:
        out["gain_ci_lo"], out["gain_ci_hi"] = _as_float(lo), _as_float(hi)
    else:
        out["gain_ci_lo"], out["gain_ci_hi"] = _signed(hi, sign), _signed(lo, sign)
    gain = out["gain"]
    out["improved"] = bool(gain > 0) if gain == gain else False   # nan-safe
    return out


# ==============================================================================
#  Internals
# ==============================================================================


def _deltas(a, b) -> np.ndarray:
    """``a - b`` over two aligned vectors, NaN pairs dropped. Raises on a length mismatch.

    The length check is the guard against positional pairing: two arms with different numbers of
    scored personas would otherwise broadcast or truncate, which is the failure mode described in
    the module docstring.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"paired statistics need aligned vectors, got {x.size} and {y.size}. Build them with "
            f"paired_arrays(df_a, df_b, on='persona_id') rather than slicing two frames."
        )
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok] - y[ok]


def _as_float(v) -> float:
    return float("nan") if v is None else float(v)


def _signed(v, sign: int) -> float:
    f = _as_float(v)
    return f if f != f else sign * f          # leave nan alone (nan != nan)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank (the 'average' method)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i
        while j + 1 < sorted_x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0      # mean of 1-based ranks i+1..j+1
        i = j + 1
    return ranks


def _lookup_orientation(metric: str) -> Optional[bool]:
    """Read *metric*'s orientation out of :mod:`.constants`; ``None`` if it is not registered.

    Accepts the shapes a metric registry plausibly takes -- a callable, a mapping of specs
    (dataclass or dict entries) carrying ``higher_is_better``, or a lower-is-better container --
    so the registry's exact spelling is a decision :mod:`.constants` owns rather than a coupling
    this module hard-codes.

    Raises:
        RuntimeError: when constants exposes none of them. That is a package-assembly error, not a
            data condition, so it fails loudly instead of defaulting to higher-is-better and
            silently flipping every MICI sign.
    """
    from . import constants                                   # leaf module; no cycle

    fn = getattr(constants, "higher_is_better", None)
    if callable(fn):
        return bool(fn(metric))

    found_registry = False
    for name in ("METRICS", "METRIC_REGISTRY", "QUESTIONNAIRES"):
        registry = getattr(constants, name, None)
        if not isinstance(registry, Mapping):
            continue
        found_registry = True
        spec = registry.get(metric)
        if spec is None:
            continue
        value = (spec.get("higher_is_better") if isinstance(spec, Mapping)
                 else getattr(spec, "higher_is_better", None))
        if value is not None:
            return bool(value)

    lower = getattr(constants, "LOWER_IS_BETTER", None)
    if lower is not None:
        return metric not in lower

    if not found_registry:
        raise RuntimeError(
            "eda_analysis.stats: constants exposes no metric registry, so metric orientation "
            "cannot be resolved. constants must provide one of: a higher_is_better(metric) "
            "callable, a METRICS/METRIC_REGISTRY mapping whose entries carry higher_is_better, "
            "or a LOWER_IS_BETTER container."
        )
    return None


# ---- Student-t two-sided tail, via the regularized incomplete beta -------------


def _t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p for a t statistic: ``I_{df/(df+t^2)}(df/2, 1/2)``.

    Exact in the same sense scipy's is -- the identity is the standard relation between the
    Student-t CDF and the regularized incomplete beta; the only approximation is the continued
    fraction in :func:`_betacf`, which converges to machine precision here.
    """
    if df <= 0 or not np.isfinite(t):
        return 0.0 if np.isinf(t) else float("nan")
    x = df / (df + float(t) * float(t))
    return float(_betainc(df / 2.0, 0.5, x))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` (Numerical Recipes 6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    # Use whichever tail converges fastest; the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) makes the
    # SAME `front` correct for both branches (a*log x + b*log(1-x) is invariant under the swap).
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, *, max_iter: int = 300, eps: float = 1e-15) -> float:
    """Continued fraction for the incomplete beta, evaluated by the modified Lentz method."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # even step
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # odd step
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h
