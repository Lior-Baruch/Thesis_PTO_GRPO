"""reliability.py — MEASUREMENT-VALIDITY data layer: reads the re-scoring tree written by
``Judge_Reliability.ipynb`` (``data/eval_scores_by_judge/``) and builds the tables the EDA displays.

Analysis-layer counterpart to :mod:`eda_analysis.scoring.judge`, which OWNS the paid API path that
*writes* that tree. **Nothing here calls an API** — this module is disk-only and free to re-run,
like every other ``eda_analysis`` module, so ``5_Training_and_Reliability`` can render these
tables/figures inside ``render_views.py`` while the money stays behind the ``RUN_*`` switches in
``Judge_Reliability.ipynb``. Same split as ``Run_Eval`` (paid, manual) → notebooks 1–7 (free, auto).

Two questions, from the re-scoring subset (anchor models × Q1/Q2/MICI × 96 convs):

1. **Repeatability** — the primary oracle re-scoring the SAME conversations N times (seeds differ,
   nothing else) → ICC(2,1) + mean |Δ|. This is the measurement error of the instrument.
2. **Second-judge agreement** — a different-family judge (Claude Haiku 4.5) scoring the same cells
   once → r / ρ / bias vs the primary oracle, and the defense-critical check: does the PTO−GRPO
   endpoint contrast keep its SIGN under a judge that never played the patient?

Read agreement against the **attenuation ceiling**, never against 1.0: two noisy raters cannot
correlate above ``sqrt(ICC_a × ICC_b)``. The subset scores the second judge once, so ``ICC_judge``
is unmeasured; :func:`agreement` therefore reports the ceiling under the assumption that the second
judge is as self-consistent as the primary (which makes the ceiling simply ``ICC_primary``) and
flags it as an upper bound — if Haiku is noisier, the true ceiling is lower and the observed
agreement is correspondingly better than it looks.

Figures: :mod:`eda_analysis.plotting.reliability`. Metric definitions: ``METRICS_REFERENCE.md`` §7.
"""

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .scoring import judge as _judge
from .scoring import registry as _registry

JUDGE_CHECK_ROOT = _judge.JUDGE_CHECK_ROOT
PRIMARY_TAG = _judge.PRIMARY_JUDGE.tag

# The endpoint contrasts worth checking for judge-independence. (a − b); for MICI lower = better,
# so a NEGATIVE delta on the first pair is the "PTO is less MI-inconsistent" direction.
DEFAULT_CONTRAST_PAIRS: List[Tuple[str, str]] = [
    ("PTOExp3_LA0_I10", "GRPOExp3_LA0_I10"),   # THE headline: matched 10-iter endpoint
    ("PTOExp3_LA0_I10", "PTOExp3_LA0_Base"),   # did PTO improve on base at all?
]


# ── discovery / guards ────────────────────────────────────────────────────────
def judge_tags() -> List[str]:
    """Judge folders present under ``data/eval_scores_by_judge/``.

    Excludes ``summary/`` and any ``_``-prefixed directory. The underscore rule is load-bearing:
    :mod:`eda_analysis.scoring.judge_batch` keeps its batch manifests in ``eval_scores_by_judge/_batches/``,
    and without this filter that directory is discovered as a judge — sorting FIRST, so
    ``second_judge_tags()[0]`` silently resolves to an empty tag and every multi-judge table comes
    back blank with no error. Keep bookkeeping directories underscore-prefixed.
    """
    if not os.path.isdir(JUDGE_CHECK_ROOT):
        return []
    from .constants import JUDGE_PARTITION
    return sorted(d[len(JUDGE_PARTITION):] for d in os.listdir(JUDGE_CHECK_ROOT)
                  if d.startswith(JUDGE_PARTITION)
                  and os.path.isdir(os.path.join(JUDGE_CHECK_ROOT, d)))


def second_judge_tags() -> List[str]:
    """Judge tags other than the primary oracle (i.e. the decoupled second judges)."""
    return [t for t in judge_tags() if t != PRIMARY_TAG]


def available() -> bool:
    """True when any re-scoring data exists — notebooks guard their section on this."""
    return bool(judge_tags())


# Readable names for figure titles / captions (keys are the MODEL half of a judge tag).
_JUDGE_DISPLAY = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-4-8": "Claude Opus 4.8",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
}


def judge_display(tag: str) -> str:
    """``anthropic_claude-haiku-4-5`` -> ``Claude Haiku 4.5`` (unknown models pass through)."""
    model = str(tag).split("_", 1)[-1]
    return _JUDGE_DISPLAY.get(model, model)


# ── loading ───────────────────────────────────────────────────────────────────
def load_judge_long(tag: str, *, reps: Optional[List[int]] = None) -> pd.DataFrame:
    """Tidy ``(judge, rep, metric, oracle, model, file_index, value)`` for one judge tag."""
    return _judge.load_judge_scores(tag, reps=reps)


def load_primary_long(models: Sequence[str], metrics: Sequence[str],
                      layout: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """The PRODUCTION scores for the same cells, read from the real ``eval_scores/`` tree.

    This is the comparison baseline for the second judge — the numbers the thesis actually
    reports — not a re-score. ``layout`` defaults to the auto-discovered registry layout.
    """
    if layout is None:
        exps = [e for e in _registry.EXPERIMENTS if e.model_name in set(models)]
        layout = _registry.get_model_eval_layout(exps)
    rows = []
    for model in models:
        entry = layout.get(model)
        if entry is None:
            continue
        for name in metrics:
            subdir, col = _judge.JUDGE_METRIC_COLS[name]
            ddir = _registry.eval_csv_dir(entry["root"], entry["oracle"], subdir, model)
            if not os.path.isdir(ddir):
                continue
            for fn in os.listdir(ddir):
                stem, ext = os.path.splitext(fn)
                if ext != ".csv" or not stem.isdigit():
                    continue
                try:
                    df = pd.read_csv(os.path.join(ddir, fn))
                except Exception:
                    continue
                if len(df) and col in df.columns:
                    rows.append({"metric": name, "model": model, "file_index": int(stem),
                                 "value": float(df[col].iloc[0])})
    return pd.DataFrame(rows)


# ── tables ────────────────────────────────────────────────────────────────────
def repeatability(judge_long: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per (metric, model): ICC(2,1) across reps, mean |Δ| between rep pairs, n.

    Koo & Li (2016) reading: ICC ≥ 0.75 good, ≥ 0.90 excellent. ``mean_abs_diff`` is the typical
    per-conversation swing between two scorings — the number to quote as "oracle noise".
    Returns an empty frame when fewer than 2 reps exist.
    """
    if judge_long is None:
        judge_long = load_judge_long(PRIMARY_TAG)
    if judge_long.empty or judge_long.rep.nunique() < 2:
        return pd.DataFrame()
    return _judge.repeatability_table(judge_long)


def repeatability_by_metric(rep_tab: pd.DataFrame) -> pd.DataFrame:
    """Collapse :func:`repeatability` to one row per metric (mean over models) — the citable line."""
    if rep_tab is None or rep_tab.empty:
        return pd.DataFrame()
    return (rep_tab.groupby("metric")
            .agg(n_models=("model", "nunique"), n_convs=("n_convs", "sum"),
                 icc_2_1=("icc_2_1", "mean"), mean_abs_diff=("mean_abs_diff", "mean"))
            .round(3).reset_index())


def agreement(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
              rep_tab: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Second judge vs primary oracle per (metric, model): r, ρ, bias, + the attenuation ceiling.

    ``bias_judge_minus_primary`` is the LEVEL offset (a harsher judge scores everything lower);
    it is irrelevant to the thesis claims, which are all *contrasts* — see :func:`contrasts`.
    ``ceiling`` is the max r two raters of this reliability could reach — the classical attenuation
    bound ``sqrt(ICC_primary * ICC_judge)``; ``r_pct_of_ceiling`` is the observed agreement as a
    share of it.

    **The second judge's own ICC is used when it has been measured** (≥2 full reps on disk for that
    judge, 2026-07-28). Before that it was *assumed* equal to the primary's, collapsing the ceiling
    to ``ICC_primary`` — which flattered MICI badly: Haiku's measured MICI ICC runs 0.53–0.93, so
    the true ceiling there is as low as 0.70 rather than the assumed 0.93. ``ceiling_basis`` records
    which applied, because the two give materially different readings of the same ``pearson_r``.
    """
    tab = _judge.agreement_table(judge_long, primary_long)
    if tab.empty:
        return tab
    if rep_tab is None:
        rep_tab = repeatability()
    if rep_tab is not None and not rep_tab.empty:
        icc = rep_tab.set_index(["metric", "model"])["icc_2_1"]
        tab["icc_primary"] = [icc.get((m, mo), np.nan) for m, mo in zip(tab.metric, tab.model)]
        tab["icc_judge"] = np.nan
        # Repeatability is only ever measured on the anchor subset, so most cells have NO ICC on
        # either side. Say that, rather than labelling them with an assumption that isn't being
        # made — the ceiling there is NaN, not a value derived from ICC_primary.
        tab["ceiling_basis"] = np.where(tab["icc_primary"].isna(),
                                        "no ICC measured for this cell",
                                        "assumed ICC_judge == ICC_primary")
        j_icc = _second_judge_icc(judge_long)
        if j_icc is not None:
            measured = np.array([j_icc.get((m, mo), np.nan)
                                 for m, mo in zip(tab.metric, tab.model)], dtype=float)
            have = ~np.isnan(measured) & tab["icc_primary"].notna().to_numpy()
            tab.loc[have, "icc_judge"] = measured[have].round(3)
            tab.loc[have, "ceiling_basis"] = "measured both judges"
        # Fall back to the primary's ICC wherever the judge's is unmeasured.
        icc_j = tab["icc_judge"].fillna(tab["icc_primary"])
        tab["ceiling"] = np.sqrt(tab["icc_primary"].clip(lower=0) * icc_j.clip(lower=0)).round(3)
        tab["r_pct_of_ceiling"] = (100 * tab["pearson_r"] / tab["ceiling"]).round(1)
    return tab


def _second_judge_icc(judge_long: pd.DataFrame) -> Optional[pd.Series]:
    """``(metric, model) -> ICC(2,1)`` for the second judge, or ``None`` if it has <2 reps on disk.

    ``judge_long`` is typically a single rep, so the reps are re-read from disk by tag rather than
    taken from it. Cells whose rep coverage is partial are dropped — an ICC over a lopsided rep set
    would understate reliability for reasons that have nothing to do with the judge.
    """
    if judge_long is None or judge_long.empty or "judge" not in judge_long.columns:
        return None
    tags = [t for t in judge_long["judge"].dropna().unique()]
    if len(tags) != 1:
        return None
    full = load_judge_long(str(tags[0]))
    if full.empty or full.rep.nunique() < 2:
        return None
    per_cell = full.groupby(["metric", "model", "rep"]).size().rename("n").reset_index()
    counts = per_cell.groupby(["metric", "model"]).rep.nunique()
    complete = counts[counts >= 2].index
    full = full[pd.MultiIndex.from_arrays([full.metric, full.model]).isin(complete)]
    if full.empty:
        return None
    return repeatability(full).set_index(["metric", "model"])["icc_2_1"]


def contrasts(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
              metrics: Sequence[str], pairs: Sequence[Tuple[str, str]] = None) -> pd.DataFrame:
    """THE defense check: does each endpoint contrast keep its SIGN under the second judge?

    One row per (pair, metric) with both judges' paired deltas and ``same_sign``. Pairing is by
    ``file_index``, which is persona-valid only at matched iterations (same ``seed+k+1`` shuffle) —
    :data:`DEFAULT_CONTRAST_PAIRS` respects that.
    """
    pairs = DEFAULT_CONTRAST_PAIRS if pairs is None else pairs
    rows = [_judge.contrast_preservation(judge_long, primary_long, a, b, m)
            for a, b in pairs for m in metrics]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["contrast"] = df.model_a.str.replace("Exp3", "", regex=False) + " − " \
            + df.model_b.str.replace("Exp3", "", regex=False)
    return df


def arm_means_by_judge(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                       judge_name: str = "second judge") -> pd.DataFrame:
    """Wide ``model × (metric, judge)`` means — the level-shift view behind the bias column."""
    if judge_long.empty:
        return pd.DataFrame()
    both = pd.concat([
        judge_long.assign(judge=judge_name)[["judge", "metric", "model", "value"]],
        primary_long.assign(judge="primary")[["judge", "metric", "model", "value"]],
    ])
    return both.pivot_table(index="model", columns=["metric", "judge"], values="value").round(3)


def summary_line(rep_tab: pd.DataFrame, agr_tab: pd.DataFrame, con_tab: pd.DataFrame) -> str:
    """One-sentence verdict for the notebook/SUMMARY — safe on empty inputs."""
    bits = []
    if rep_tab is not None and not rep_tab.empty:
        bits.append(f"oracle ICC(2,1) {rep_tab.icc_2_1.min():.2f}–{rep_tab.icc_2_1.max():.2f} "
                    f"(mean |Δ| {rep_tab.mean_abs_diff.mean():.2f})")
    if agr_tab is not None and not agr_tab.empty:
        q = agr_tab[agr_tab.metric.isin(["Q1", "Q2"])]
        if not q.empty:
            bits.append(f"Q1/Q2 cross-judge r {q.pearson_r.min():.2f}–{q.pearson_r.max():.2f}")
    if con_tab is not None and not con_tab.empty:
        bits.append(f"{int(con_tab.same_sign.sum())}/{len(con_tab)} contrasts keep their sign")
    return "; ".join(bits) if bits else "no re-scoring data on disk"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   MULTI-JUDGE ANALYSIS — variance decomposition, transfer, concordance       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Everything above answers "is the instrument reliable?". Everything below answers the harder
# question the thesis actually needs: **when two judges disagree, which part of the disagreement
# threatens a claim?** Three views, in increasing order of what they buy:
#
#   1. variance_components_*  — how much of the measured variance is signal (arm), how much is
#      judge level bias (harmless), and how much is arm x judge INTERACTION (the dangerous part:
#      a score that depends on who is grading).
#   2. gain_retention         — what fraction of each arm's improvement over Base survives a judge
#      swap. The standard reward-hacking test: a policy that overfits its grader does not transfer.
#   3. concordance_by_effect_size — "when the primary judge says A beats B by >= x, how often does
#      the second judge agree?" Read the agreement rate at the effect size you are claiming.
#
# NOTE ON THE TWO JUDGES' STATUS. They are not interchangeable measurements of one quantity: the
# PRIMARY judge (gpt-4o-mini Q1+Q2) *was the training reward*, and the second judge never touched
# training. That makes this an optimization-target vs held-out-test comparison, not a
# two-rater average — which is why nothing here ever averages raw scores across judges. Level is
# judge-specific (bias runs 1.2-1.7 points and is MODEL-DEPENDENT, comparable to the headline
# effect); only contrasts and standardized quantities are comparable across judges.

_MODEL_ITER_RE = re.compile(r"^(PTO|GRPO)Exp3_LA(\d+)_(Base|I\d+)$")
_DEFAULT_SEED = 42


def model_iteration(model: str) -> int:
    """``PTOExp3_LA0_I10`` -> 10, ``..._Base`` -> 0 (the ``model_iter`` index on disk)."""
    m = _MODEL_ITER_RE.match(str(model))
    if not m:
        return -1
    tail = m.group(3)
    return 0 if tail == "Base" else int(tail[1:])


def attach_persona(long_df: pd.DataFrame, *, seed: int = _DEFAULT_SEED,
                   n_personas: int = 96) -> pd.DataFrame:
    """Add ``persona_id`` to a judge/primary long frame by replaying the per-iteration shuffle.

    **Why this is required for anything paired.** The trainer re-shuffles the same 96 personas
    every iteration (``model_iter_k`` uses shuffle seed ``seed + k + 1``), so ``file_index`` 7 is a
    *different patient* in ``model_iter_0`` than in ``model_iter_10``. Pairing on ``file_index``
    across unmatched iterations therefore pairs unrelated conversations.

    A difference of MEANS is unaffected by this (both arms cover all 96 personas either way) —
    which is why :func:`contrasts` reports valid deltas today. But a paired SD, a Cohen's dz, or a
    paired CI computed on ``file_index`` pairs is wrong, and those are exactly what a thesis table
    reports. Everything below pairs on ``persona_id``.
    """
    from .data import persona_order
    out = long_df.copy()
    iters = out["model"].map(model_iteration)
    cache: Dict[int, List[int]] = {}
    pids = []
    for it, fi in zip(iters, out["file_index"].astype(int)):
        if it < 0:
            pids.append(-1)
            continue
        order = cache.get(it)
        if order is None:
            order = persona_order(seed, it, n_personas)
            cache[it] = order
        pids.append(order[fi] if 0 <= fi < n_personas else -1)
    out["persona_id"] = pids
    return out


# ── 0. coverage — never analyse a partially-scored cell alongside a complete one ──

def coverage_table(judge_long: pd.DataFrame, n_expected: int = 96) -> pd.DataFrame:
    """Conversations scored per (metric, model), with a ``complete`` flag.

    A second-judge sweep can land partially — a batch can be cut short by rate limits, an expired
    batch, or an exhausted credit balance. The failures are random with respect to persona, so a
    partial cell's mean is unbiased, but it is NOT comparable to a complete one: precision differs
    (SE scales as sqrt(96/n)) and, worse, persona-PAIRED statistics collapse, because two arms each
    covering a random ~43% overlap on only ~0.43^2 ~ 18% of personas. Mixing the two silently is
    the failure mode this guards against.
    """
    if judge_long is None or judge_long.empty:
        return pd.DataFrame()
    cov = (judge_long.groupby(["metric", "model"])["file_index"].nunique()
           .rename("n_scored").reset_index())
    cov["n_expected"] = n_expected
    cov["pct"] = (100 * cov.n_scored / n_expected).round(1)
    cov["complete"] = cov.n_scored >= n_expected
    return cov.sort_values(["metric", "model"]).reset_index(drop=True)


def filter_complete_cells(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                          *, n_required: int = 96, verbose: bool = True
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only (metric, model) cells the second judge scored ``n_required`` times.

    Returns ``(judge_long, primary_long)`` restricted to the same cells, so every downstream table
    is computed on one consistent, fully-scored grid. **Always reports what it dropped** — a
    silently truncated grid reads as full coverage, which is how a partial sweep turns into a
    published number.
    """
    if judge_long is None or judge_long.empty:
        return judge_long, primary_long
    cov = coverage_table(judge_long, n_expected=n_required)
    keep = {(r.metric, r.model) for r in cov.itertuples() if r.complete}
    dropped = [(r.metric, r.model, r.n_scored) for r in cov.itertuples() if not r.complete]
    if verbose and dropped:
        by_metric: Dict[str, int] = {}
        for m, _mo, _n in dropped:
            by_metric[m] = by_metric.get(m, 0) + 1
        print(f"[coverage] keeping {len(keep)}/{len(cov)} (metric, model) cells with the full "
              f"{n_required} conversations; dropped {len(dropped)} partial cells "
              f"(n {min(d[2] for d in dropped)}–{max(d[2] for d in dropped)}): "
              + ", ".join(f"{k} x{v}" for k, v in sorted(by_metric.items())))
    jl = judge_long[[(m, mo) in keep for m, mo in zip(judge_long.metric, judge_long.model)]]
    pl = primary_long
    if primary_long is not None and not primary_long.empty:
        pl = primary_long[[(m, mo) in keep
                           for m, mo in zip(primary_long.metric, primary_long.model)]]
    return jl.copy(), (pl.copy() if pl is not None else pl)


# ── 1. variance decomposition (generalizability theory) ───────────────────────

def _two_way_components(m: np.ndarray) -> dict:
    """Two-way random-effects ANOVA variance components for an ``n_targets x k_raters`` matrix
    with ONE observation per cell (Shrout & Fleiss / Brennan expected-mean-square estimators).

    Returns ``var_target`` (between-target "true score" variance), ``var_rater`` (systematic level
    difference between raters) and ``var_resid`` (target x rater interaction, inseparable from
    error at one observation per cell). Negative component estimates are clamped to 0, the standard
    convention — a negative estimate means the component is indistinguishable from zero.

    ``ICC(2,1) = var_target / (var_target + var_rater + var_resid)``, so :func:`icc_2_1` is the
    ratio of these same quantities; this function exposes the parts.
    """
    m = np.asarray(m, float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape if m.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return {"n": n, "k": k, "var_target": np.nan, "var_rater": np.nan, "var_resid": np.nan}
    mean_t, mean_r, grand = m.mean(axis=1), m.mean(axis=0), m.mean()
    msr = (k * ((mean_t - grand) ** 2).sum()) / (n - 1)
    msc = (n * ((mean_r - grand) ** 2).sum()) / (k - 1)
    mse = ((m - mean_t[:, None] - mean_r[None, :] + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return {"n": n, "k": k,
            "var_target": max((msr - mse) / k, 0.0),
            "var_rater": max((msc - mse) / n, 0.0),
            "var_resid": max(mse, 0.0)}


def _judge_primary_matrix(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                          metric: str, model: str) -> pd.DataFrame:
    """Conversations x {primary, judge} value matrix for one (metric, model)."""
    j = (judge_long[(judge_long.metric == metric) & (judge_long.model == model)]
         .groupby("file_index")["value"].mean().rename("judge"))
    p = (primary_long[(primary_long.metric == metric) & (primary_long.model == model)]
         .groupby("file_index")["value"].mean().rename("primary"))
    return pd.concat([p, j], axis=1).dropna()


def variance_components_conversation(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                                     metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per (metric, model): decompose per-CONVERSATION score variance across the two judges.

    Targets = the 96 conversations, raters = the two judges. Answers "within one arm, do the
    judges rank individual conversations the same way?" — ``var_resid`` here is per-conversation
    disagreement, which is what attenuates cross-judge correlations.
    """
    metrics = list(metrics or sorted(set(judge_long.metric) & set(primary_long.metric)))
    rows = []
    for metric in metrics:
        models = sorted(set(judge_long[judge_long.metric == metric].model)
                        & set(primary_long[primary_long.metric == metric].model))
        for model in models:
            wide = _judge_primary_matrix(judge_long, primary_long, metric, model)
            if len(wide) < 3:
                continue
            c = _two_way_components(wide[["primary", "judge"]].to_numpy())
            tot = c["var_target"] + c["var_rater"] + c["var_resid"]
            rows.append({"metric": metric, "model": model, "n_convs": c["n"],
                         "var_conversation": round(c["var_target"], 4),
                         "var_judge": round(c["var_rater"], 4),
                         "var_resid": round(c["var_resid"], 4),
                         "pct_conversation": round(100 * c["var_target"] / tot, 1) if tot else np.nan,
                         "pct_judge": round(100 * c["var_rater"] / tot, 1) if tot else np.nan,
                         "pct_resid": round(100 * c["var_resid"] / tot, 1) if tot else np.nan})
    return pd.DataFrame(rows)


def variance_components_arm(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                            metrics: Optional[Sequence[str]] = None,
                            conv_components: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per metric: decompose ARM-MEAN variance across models x judges. **The decision-relevant one.**

    Targets = model states (arms), raters = the two judges, cells = the 96-conversation arm mean —
    i.e. exactly the numbers the thesis reports. Three components:

    - ``var_arm``        — real between-arm differences. The signal.
    - ``var_judge``      — the two judges' overall level offset. Large, and harmless: it cancels in
      every contrast.
    - ``var_arm_x_judge``— **the dangerous term.** Arm ordering that depends on who is grading. A
      policy that overfits its grader shows up here and nowhere else.

    At one observation per cell, interaction and sampling error are confounded. Sampling error IS
    separately estimable from the conversation level (``var_resid / n_convs``), so when
    ``conv_components`` is supplied this subtracts it to give ``var_arm_x_judge_adj`` — the
    interaction net of what 96-conversation sampling alone would produce.

    ``dependability_k1`` / ``dependability_k2`` are the generalizability coefficients for an arm
    mean measured with one judge vs. the average of two:
    ``G(k) = var_arm / (var_arm + var_arm_x_judge/k + var_resid/(n_convs*k))``.
    The k1 -> k2 gain is the honest answer to "does adding a second judge make my arm ranking more
    dependable?".
    """
    metrics = list(metrics or sorted(set(judge_long.metric) & set(primary_long.metric)))
    if conv_components is None:
        conv_components = variance_components_conversation(judge_long, primary_long, metrics)
    rows = []
    for metric in metrics:
        models = sorted(set(judge_long[judge_long.metric == metric].model)
                        & set(primary_long[primary_long.metric == metric].model))
        if len(models) < 2:
            continue
        cells, n_convs = [], []
        for model in models:
            wide = _judge_primary_matrix(judge_long, primary_long, metric, model)
            if wide.empty:
                continue
            cells.append([wide["primary"].mean(), wide["judge"].mean()])
            n_convs.append(len(wide))
        if len(cells) < 2:
            continue
        c = _two_way_components(np.asarray(cells))
        n_bar = float(np.mean(n_convs))

        cc = conv_components[conv_components.metric == metric]
        var_resid_conv = float(cc["var_resid"].mean()) if not cc.empty else np.nan
        sampling = var_resid_conv / n_bar if np.isfinite(var_resid_conv) and n_bar else np.nan
        inter_adj = (max(c["var_resid"] - sampling, 0.0)
                     if np.isfinite(sampling) else np.nan)

        def dependability(k: int) -> float:
            inter = inter_adj if np.isfinite(inter_adj) else c["var_resid"]
            noise = inter / k + ((var_resid_conv / (n_bar * k)) if np.isfinite(var_resid_conv) else 0.0)
            den = c["var_target"] + noise
            return float(c["var_target"] / den) if den else np.nan

        tot = c["var_target"] + c["var_rater"] + c["var_resid"]
        rows.append({
            "metric": metric, "n_arms": c["n"], "n_convs_mean": round(n_bar, 1),
            "var_arm": round(c["var_target"], 4),
            "var_judge": round(c["var_rater"], 4),
            "var_arm_x_judge": round(c["var_resid"], 4),
            "var_arm_x_judge_adj": (round(inter_adj, 4) if np.isfinite(inter_adj) else np.nan),
            "pct_arm": round(100 * c["var_target"] / tot, 1) if tot else np.nan,
            "pct_judge": round(100 * c["var_rater"] / tot, 1) if tot else np.nan,
            "pct_arm_x_judge": round(100 * c["var_resid"] / tot, 1) if tot else np.nan,
            "dependability_k1": round(dependability(1), 3),
            "dependability_k2": round(dependability(2), 3),
        })
    return pd.DataFrame(rows)


# ── 2. does the improvement transfer to a held-out judge? ─────────────────────

def gain_retention(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                   reference_model: str, metrics: Optional[Sequence[str]] = None,
                   min_primary_delta: float = 0.15, *, n_boot: int = 2000,
                   seed: int = _DEFAULT_SEED) -> pd.DataFrame:
    """**The reward-hacking test.** What fraction of each arm's gain over ``reference_model``
    survives a judge swap?

    ``retention = delta_judge / delta_primary``. Because the primary judge WAS the training reward
    and the second judge is held out, this is a train/test generalization ratio, not a
    reliability statistic:

    - retention ~1.0  — the gain is a real behaviour change both judges see.
    - retention ~0    — the gain existed only in the grader that was optimized. Textbook
      reward hacking.

    A *uniform* retention across arms (e.g. every arm at ~0.8) is scale compression, not hacking —
    the finding is only interesting when retention DIFFERS by arm on some metric while staying flat
    on others. ``retention`` is suppressed (NaN) when ``|delta_primary| < min_primary_delta``,
    where the ratio is dominated by noise in the denominator. Direction-agnostic: both deltas flip
    together for lower-is-better metrics such as MICI.

    Retention is a RATIO OF TWO ESTIMATED DELTAS and is therefore noisier than either — a point
    estimate alone would overstate what it can support. ``retention_ci_lo/hi`` is a percentile
    bootstrap over PERSONAS (resampling the 96 personas jointly for both arms and both judges, so
    the numerator and denominator stay coupled the way they are in the data). Compare arms by
    whether their intervals overlap, not by their point estimates.
    """
    metrics = list(metrics or sorted(set(judge_long.metric) & set(primary_long.metric)))
    jl, pl = attach_persona(judge_long, seed=seed), attach_persona(primary_long, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        models = sorted(set(jl[jl.metric == metric].model) & set(pl[pl.metric == metric].model))
        if reference_model not in models:
            continue

        def series(src: pd.DataFrame, model: str) -> pd.Series:
            g = src[(src.metric == metric) & (src.model == model)]
            return g.groupby("persona_id")["value"].mean()

        ref_j, ref_p = series(jl, reference_model), series(pl, reference_model)
        for model in models:
            if model == reference_model:
                continue
            mod_j, mod_p = series(jl, model), series(pl, model)
            common = (ref_j.index.intersection(ref_p.index)
                      .intersection(mod_j.index).intersection(mod_p.index))
            if len(common) < 3:
                continue
            aj, ap = mod_j.loc[common].to_numpy(), mod_p.loc[common].to_numpy()
            bj, bp = ref_j.loc[common].to_numpy(), ref_p.loc[common].to_numpy()
            dj, dp = float((aj - bj).mean()), float((ap - bp).mean())
            keep = abs(dp) >= min_primary_delta

            ci_lo = ci_hi = np.nan
            if keep and dp and n_boot:
                idx = rng.integers(0, len(common), size=(n_boot, len(common)))
                bdj = (aj[idx] - bj[idx]).mean(axis=1)
                bdp = (ap[idx] - bp[idx]).mean(axis=1)
                ok = np.abs(bdp) >= min_primary_delta
                if ok.sum() >= 20:
                    ratios = bdj[ok] / bdp[ok]
                    ci_lo = round(float(np.percentile(ratios, 2.5)), 3)
                    ci_hi = round(float(np.percentile(ratios, 97.5)), 3)

            rows.append({"metric": metric, "model": model, "reference": reference_model,
                         "n": int(len(common)),
                         "delta_primary": round(dp, 3), "delta_judge": round(dj, 3),
                         "retention": round(dj / dp, 3) if keep and dp else np.nan,
                         "retention_pct": round(100 * dj / dp, 1) if keep and dp else np.nan,
                         "retention_ci_lo": ci_lo, "retention_ci_hi": ci_hi,
                         "same_sign": bool(np.sign(dj) == np.sign(dp))})
    return pd.DataFrame(rows)


def all_pairs_contrasts(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                        metrics: Optional[Sequence[str]] = None,
                        models: Optional[Sequence[str]] = None,
                        *, seed: int = _DEFAULT_SEED, n_boot: int = 2000) -> pd.DataFrame:
    """Every ordered model pair x metric, under both judges, with persona-correct pairing.

    :data:`DEFAULT_CONTRAST_PAIRS` checks two hand-picked contrasts. With four anchor models there
    are six, and the two that were never checked are the two the thesis leans on hardest: the
    best-vs-best steelman (PTO@10 vs GRPO@8) and the regression claim (GRPO@8 vs GRPO@10). This
    enumerates all of them.

    Pairing is on ``persona_id`` (see :func:`attach_persona`), so ``dz`` and the bootstrap CI are
    valid across unmatched iterations — unlike a ``file_index`` join. ``same_sign`` is the
    headline: a contrast that keeps its sign under a judge that never saw training is a contrast
    the coupling critique cannot touch.
    """
    metrics = list(metrics or sorted(set(judge_long.metric) & set(primary_long.metric)))
    jl, pl = attach_persona(judge_long, seed=seed), attach_persona(primary_long, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        avail = sorted(set(jl[jl.metric == metric].model) & set(pl[pl.metric == metric].model))
        use = [m for m in (models or avail) if m in avail]
        for i, a in enumerate(use):
            for b in use[i + 1:]:
                rec = {"metric": metric, "model_a": a, "model_b": b}
                for src_name, src in (("judge", jl), ("primary", pl)):
                    g = src[src.metric == metric]
                    va = g[g.model == a].groupby("persona_id")["value"].mean()
                    vb = g[g.model == b].groupby("persona_id")["value"].mean()
                    common = va.index.intersection(vb.index)
                    d = (va.loc[common] - vb.loc[common]).astype(float).to_numpy()
                    rec[f"{src_name}_n"] = int(len(d))
                    if len(d) < 3:
                        rec[f"{src_name}_delta"] = np.nan
                        rec[f"{src_name}_dz"] = np.nan
                        continue
                    rec[f"{src_name}_delta"] = round(float(d.mean()), 3)
                    sd = float(d.std(ddof=1))
                    rec[f"{src_name}_dz"] = round(float(d.mean() / sd), 3) if sd else np.nan
                    if src_name == "judge" and n_boot:
                        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                        bs = d[idx].mean(axis=1)
                        rec["judge_ci_lo"] = round(float(np.percentile(bs, 2.5)), 3)
                        rec["judge_ci_hi"] = round(float(np.percentile(bs, 97.5)), 3)
                rec["same_sign"] = bool(np.sign(rec.get("judge_delta", np.nan))
                                        == np.sign(rec.get("primary_delta", np.nan)))
                rec["contrast"] = (a.replace("Exp3", "") + " − " + b.replace("Exp3", ""))
                rows.append(rec)
    return pd.DataFrame(rows)


def sign_preservation(pairs: pd.DataFrame,
                      *, thresholds: Sequence[float] = (0.10, 0.25, 0.50),
                      by: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """How often the held-out judge agrees on DIRECTION, as a function of the gap being claimed.

    :func:`all_pairs_contrasts` returns one row per contrast; the headline the thesis actually
    quotes is the *rate* over that table, and it is only meaningful against an effect size. A
    pooled "88% of contrasts agree" reads as a poor result until you see that the disagreements
    sit entirely in gaps too small to claim — hence the ladder rather than a single number.

    ``by=["metric"]`` gives the same ladder per rubric, which is how MITI's weakness shows up
    independently of the variance decomposition in :func:`variance_components_arm`.

    ⚠ ``thresholds`` are **absolute**, so a per-rubric ladder is comparable DOWN its own rubric and
    never ACROSS rubrics: ``PCT``/``MICI`` live on a 0-1 scale and never reach 0.25 at all, while
    Q1/Q2/WAI-SR/MITI are 1-5 or 1-7. The ``all contrasts`` row is the cross-rubric comparison.

    The last row of each ladder restricts to contrasts whose judge-side bootstrap CI excludes
    zero — i.e. the ones the second judge itself calls non-null. Omitted when ``pairs`` carries no
    CI columns (``n_boot=0``).
    """
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=["subset", "n_contrasts", "n_same_sign", "pct_same_sign"])

    df = pairs.dropna(subset=["judge_delta", "primary_delta"]).copy()
    has_ci = {"judge_ci_lo", "judge_ci_hi"}.issubset(df.columns)

    subsets: List[tuple] = [("all contrasts", df)]
    for t in thresholds:
        subsets.append((f"|Δ primary| ≥ {t:.2f}", df[df.primary_delta.abs() >= t]))
    if has_ci:
        ci = df[(df.judge_ci_lo > 0) | (df.judge_ci_hi < 0)]
        subsets.append(("judge CI excludes 0", ci))

    groups = [(None, df)] if not by else list(df.groupby(list(by), sort=False))
    rows = []
    for key, _ in groups:
        keys = dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {}
        for label, sub in subsets:
            if by:  # re-slice this subset down to the current group
                mask = pd.Series(True, index=sub.index)
                for col, val in keys.items():
                    mask &= sub[col] == val
                sub = sub[mask]
            n = len(sub)
            rows.append({**keys, "subset": label, "n_contrasts": n,
                         "n_same_sign": int(sub.same_sign.sum()) if n else 0,
                         "pct_same_sign": round(100 * sub.same_sign.mean(), 1) if n else np.nan})
    return pd.DataFrame(rows)


# ── 3. concordance as a function of effect size ───────────────────────────────

def concordance_by_effect_size(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                               metric: str, *, bins: Sequence[float] = (0, .1, .25, .5, 1., 2., 99.),
                               max_pairs: int = 400_000, seed: int = _DEFAULT_SEED,
                               scope: str = "cross_model") -> pd.DataFrame:
    """"When the primary judge says A beats B by at least x, how often does the second judge agree?"

    Reported as a CURVE over ``|delta_primary|`` rather than a single correlation, because the
    decision the thesis needs is not "do the judges correlate" but "is a gap of THIS size
    trustworthy". Read the agreement rate at the effect size you are claiming.

    A scalar r or rho is the wrong summary here anyway: cross-judge level bias is ~1.2-1.7 points,
    so Pearson is dominated by a shift that cancels in every contrast, while a rank statistic
    throws away the magnitude that decides whether a gap matters.

    ⚠ **DO NOT read a bin's concordance as the confidence in an arm-level claim.** Every pair here
    is TWO SINGLE CONVERSATIONS. The thesis reports differences between 96-conversation MEANS,
    which are ~10x better resolved — the headline Q1 endpoint gap of 0.53 lands in the [0.5, 1)
    bin at ~0.69 per-conversation concordance, and that is emphatically *not* "69% confidence in
    the headline". The arm-level answers are :func:`all_pairs_contrasts` (sign preservation with
    bootstrap CIs) and ``dependability_k1`` in :func:`variance_components_arm`. What this curve is
    for: showing how much per-conversation resolving power a given gap carries, i.e. WHY 96
    conversations per arm are needed and how small a gap would have to get before the design stops
    supporting it.

    ``scope``:
      - ``"cross_model"`` (default) — pairs drawn from DIFFERENT model states. This is the
        contrast question the thesis actually asks.
      - ``"within_model"`` — pairs from the same model state: fine-grained discrimination between
        two conversations of one policy. Much harder; expect lower agreement. Useful as a
        contrast to show that the cross-model number is not just an easy-task artifact.
      - ``"all"`` — both.

    Sampling is seeded and capped at ``max_pairs`` (full enumeration is O(N^2)); the cap is
    reported in ``n_pairs`` so a truncated sweep is never mistaken for full coverage.
    """
    j = (judge_long[judge_long.metric == metric]
         .groupby(["model", "file_index"])["value"].mean().rename("judge"))
    p = (primary_long[primary_long.metric == metric]
         .groupby(["model", "file_index"])["value"].mean().rename("primary"))
    d = pd.concat([p, j], axis=1).dropna().reset_index()
    if len(d) < 3:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    n = len(d)
    model_code = pd.factorize(d["model"])[0]
    pv, jv = d["primary"].to_numpy(), d["judge"].to_numpy()

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        ia, ib = np.triu_indices(n, k=1)
    else:
        ia = rng.integers(0, n, max_pairs)
        ib = rng.integers(0, n, max_pairs)
        keep = ia != ib
        ia, ib = ia[keep], ib[keep]

    same_model = model_code[ia] == model_code[ib]
    if scope == "cross_model":
        sel = ~same_model
    elif scope == "within_model":
        sel = same_model
    else:
        sel = np.ones_like(same_model, dtype=bool)
    ia, ib = ia[sel], ib[sel]
    if not len(ia):
        return pd.DataFrame()

    dp = pv[ia] - pv[ib]
    dj = jv[ia] - jv[ib]

    # EXACT TIES IN THE PRIMARY JUDGE ARE EXCLUDED, not counted as disagreement. These rubrics are
    # means of a handful of integers, so exact ties are common (~8% of Q1 pairs) — and a tie states
    # no ordering for the second judge to reproduce. Scoring them as failures drives the smallest
    # bin far BELOW chance (sign(0) matches almost nothing) and misreads "the primary judge could
    # not separate these" as "the judges disagree". The tie rate is reported separately instead.
    untied = dp != 0
    n_ties = int((~untied).sum())
    ia, ib, dp, dj = ia[untied], ib[untied], dp[untied], dj[untied]
    if not len(dp):
        return pd.DataFrame()
    absdp = np.abs(dp)
    # A tie in the SECOND judge still counts as disagreement: it did fail to reproduce a stated
    # ordering. That is the conservative reading for a defence.
    agree = np.sign(dp) == np.sign(dj)

    rows = []
    edges = list(bins)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (absdp >= lo) & (absdp < hi)
        if not m.any():
            continue
        rows.append({"metric": metric, "scope": scope,
                     "delta_lo": lo, "delta_hi": hi,
                     "bin": f"[{lo:g}, {hi:g})" if hi < 90 else f">= {lo:g}",
                     "n_pairs": int(m.sum()),
                     "concordance": round(float(agree[m].mean()), 3),
                     "mean_abs_delta_primary": round(float(absdp[m].mean()), 3)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs["truncated"] = total_pairs > max_pairs
        out.attrs["total_possible_pairs"] = total_pairs
        out.attrs["n_primary_ties_excluded"] = n_ties
        out.attrs["pct_primary_ties"] = round(100 * n_ties / (n_ties + len(dp)), 1)
    return out


def multi_judge_summary_line(var_arm: pd.DataFrame, ret: pd.DataFrame,
                             pairs: pd.DataFrame) -> str:
    """One-sentence verdict over the multi-judge tables — safe on empty inputs."""
    bits = []
    if var_arm is not None and not var_arm.empty:
        bits.append(f"arm-mean variance {var_arm.pct_arm.mean():.0f}% arm / "
                    f"{var_arm.pct_judge.mean():.0f}% judge-level / "
                    f"{var_arm.pct_arm_x_judge.mean():.0f}% arm×judge")
    if ret is not None and not ret.empty:
        r = ret.dropna(subset=["retention_pct"])
        if not r.empty:
            bits.append(f"gain retention {r.retention_pct.min():.0f}–{r.retention_pct.max():.0f}%")
    if pairs is not None and not pairs.empty:
        bits.append(f"{int(pairs.same_sign.sum())}/{len(pairs)} of all pairwise contrasts "
                    f"keep their sign")
    return "; ".join(bits) if bits else "no multi-judge data on disk"
