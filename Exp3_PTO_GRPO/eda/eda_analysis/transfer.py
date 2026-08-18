"""transfer.py — does the look-ahead (K) contrast itself survive the held-out judge?

The tracked measurement family judge-tests contrasts WITHIN one K (its ``all_pairs_contrasts``
enumerates the K=5 model states, or the K=0 ones — never a K=0 vs K=5 pair), and its gain-retention
table uses one shared PTO base per K. This module asks the cross-K transfer questions:

1. :func:`cross_k_pairs` — for each method, ``LA0_In − LA5_In`` at every matched iteration
   (iteration 0 = the two INDEPENDENT base draws, a free noise-floor row) on every rubric, with the
   primary (training oracle) and held-out (Claude Haiku 4.5) delta / dz / bootstrap CI / Wilcoxon p /
   Holm p side by side, ``same_sign`` and ``judge_ci_excl0``.
2. :func:`sign_ladder` — sign preservation of those contrasts under the held-out judge, as a ladder
   over the gap the primary reports (mirrors :func:`eda_analysis.reliability.sign_preservation`)
   plus Holm-significance rungs and an iteration ≥ 1 rung.
3. :func:`retention_by_k` — ``retention = Δ held-out / Δ primary`` of every model state over a
   reference base — vs the arm's OWN base and vs SHARED references (the method's LA0 base, its LA5
   base, and — for the GRPO arms — the tracked measurement family's PTO_LA{K}_Base) so the
   base-reference caveat is visible; plus the compact K=0-vs-K=5 endpoint summary.

**Provenance.** Promoted on 2026-08-18 from
``papers/2026_lookahead_pto_grpo/analysis/cross_k_multijudge.py`` §1-2 (its §3 DiD / method gap and
§4 endpoints live in :mod:`eda_analysis.lookahead`). The paper's frozen
``tables/cross_k_multijudge_{pairs,ladder,retention,retention_summary}.csv`` are the fixture.

Conventions:

* **Sign: cross-K contrast ``+ => K=0 higher``.** MICI is lower-is-better, so on MICI a positive
  K contrast favours K=5 — every table carries explicit ``favours_*`` columns.
* **Pairing unit: ``persona_id``** (never ``file_index`` — the trainer reshuffles the 96 personas
  every iteration). Retention pairs through :func:`eda_analysis.reliability.attach_persona`
  (replays the shuffle from ``file_index``); the pairs frame uses the ``persona_id`` already on the
  ``scores_long`` frames.
* **Censoring:** GRPO_LA5 is right-censored at iteration 5 (its full budget); PTO arms and
  GRPO_LA0 run to 10. Last iterations are read off the data.
* **Holm** in ``cross_k_pairs`` = across ITERATIONS within (grader, method, metric).
* **Retention floors:** ``retention`` and its persona-bootstrap CI are suppressed (NaN) where
  ``|Δ primary| < min_primary_delta`` — 0.15 on the 1-5 / 1-7 rubrics (the
  :func:`~eda_analysis.reliability.gain_retention` default) and 0.05 on the 0-1 rate metrics
  PCT / MICI, whose deltas are ~3x smaller (a 0.15 floor blanks almost every MICI row).
  Direction-agnostic on MICI (both deltas flip together).
* **The two graders' raw scores are never averaged** — the primary WAS the training reward and the
  second judge is held out (train vs test, not two raters); only contrasts / ratios are combined.
* **Seeds.** :func:`cross_k_pairs` CIs use :func:`eda_analysis.stats.paired_arrays`
  (:data:`~eda_analysis.constants.BOOT_SEED`; the paper seeded with 0, so CI bounds may differ in
  the third decimal — n / means / dz / p / Holm p are exact). :func:`retention_by_k` calls
  :func:`~eda_analysis.reliability.gain_retention` exactly as the paper did (its own seed 42, one
  fresh generator per (arm, reference, metric-group) call), so retention CIs reproduce exactly.

Contract: frames in, tidy ``pd.DataFrame`` s out (or a dict keyed by the paper's table-name
suffix); no disk I/O. :func:`transfer_numbers` returns the quotable ledger.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .lookahead import (RUBRICS, RATE_METRICS, METHODS, favours, holm_within, model_name,
                        wide_by_persona, k_of, method_of, _nan_none)
from .stats import paired_arrays

__all__ = [
    "ARMS", "DEFAULT_REFERENCE_KINDS", "SCALE_FLOOR", "RATE_FLOOR",
    "to_reliability_long", "cross_k_pairs", "sign_ladder", "retention_by_k", "transfer_numbers",
]

ARMS = ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5")
#: Reference kinds for :func:`retention_by_k`, as templates over ``{method}`` / ``{K}``. The
#: ``eda_view_PTO_LA{K}_base`` kind is emitted for the GRPO arms only (for a PTO arm it duplicates
#: ``method_LA{K}_base``): it is the tracked measurement family's convention (PTO's base of the same
#: K), so its numbers can be matched against ``multijudge_gain_retention``.
DEFAULT_REFERENCE_KINDS = ("own_base", "method_LA0_base", "method_LA5_base", "eda_view_PTO_LA{K}_base")
#: |Δ primary| floors below which retention is suppressed (see the module note).
SCALE_FLOOR, RATE_FLOOR = 0.15, 0.05

CENSOR = "GRPO_LA5 is right-censored at iteration 5 (its full budget); PTO arms and GRPO_LA0 run to 10."
PAIRING = ("Paired on persona_id (the trainer reshuffles the 96 personas every iteration; "
           "file_index is not a pairing key).")


# ── shapes ────────────────────────────────────────────────────────────────────

def _is_scores_long(df: pd.DataFrame) -> bool:
    return "questionnaire" in df.columns and "score" in df.columns


def to_reliability_long(df: pd.DataFrame) -> pd.DataFrame:
    """``scores_long`` → the ``(metric, model, file_index, value)`` shape ``reliability.*`` expects.

    A frame already in that shape passes through unchanged (columns re-ordered).
    """
    if _is_scores_long(df):
        df = df.rename(columns={"questionnaire": "metric", "score": "value"})
    return df[["metric", "model", "file_index", "value"]].reset_index(drop=True)


def _as_scores_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reliability-shaped → scores_long shape (with ``persona_id`` and ``arm``/``iteration``)."""
    if _is_scores_long(df):
        out = df
    else:
        out = df.rename(columns={"metric": "questionnaire", "value": "score"})
    if "persona_id" not in out.columns:
        from . import reliability as R
        tmp = R.attach_persona(out.rename(columns={"questionnaire": "metric", "score": "value"}))
        out = out.assign(persona_id=tmp["persona_id"].to_numpy())
    if "arm" not in out.columns or "iteration" not in out.columns:
        from . import reliability as R
        mm = out["model"].astype(str)
        out = out.assign(arm=[f"{m.split('Exp3')[0]}_LA{m.split('_LA')[1].split('_')[0]}" for m in mm],
                         iteration=mm.map(R.model_iteration))
    return out


def _last_iters(long: pd.DataFrame) -> Dict[str, int]:
    return {arm: int(long.loc[long["arm"] == arm, "iteration"].max())
            for arm in ARMS if (long["arm"] == arm).any()}


def _present(long: pd.DataFrame, metrics: Optional[Sequence[str]]) -> list:
    have = set(long["questionnaire"].unique())
    return [m for m in (RUBRICS if metrics is None else metrics) if m in have]


# ── 1. cross-K pairs ──────────────────────────────────────────────────────────

def cross_k_pairs(judge_long: pd.DataFrame, primary_long: pd.DataFrame, *,
                  methods: Sequence[str] = METHODS, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """``LA0_In − LA5_In`` per method at every matched iteration, under both graders.

    ``judge_long`` = the held-out judge's frame, ``primary_long`` = the training oracle's — either
    ``scores_long`` (with ``persona_id``) or the reliability ``(metric, model, file_index, value)``
    shape (persona re-attached by replaying the shuffle). Columns mirror
    :func:`~eda_analysis.reliability.all_pairs_contrasts`: ``primary_*`` / ``judge_*`` (n, delta,
    dz, ci_lo, ci_hi, p, p_holm — Holm across ITERATIONS within (grader, method, metric)),
    ``same_sign`` (the graders agree on direction), ``judge_ci_excl0`` (the held-out CI excludes 0),
    ``favours_primary`` / ``favours_judge`` (``K0`` / ``K5``, flipped for MICI). Sorted method →
    :data:`~eda_analysis.lookahead.RUBRICS` order → iteration. Reproduces
    ``cross_k_multijudge_pairs.csv`` (17 iteration pairs × 9 rubrics = 153 rows: PTO 11×9, GRPO 6×9).
    Cross-check: PTO Q1Q2 iteration 6 primary = +0.257, dz 0.417.
    """
    JL, PL = _as_scores_long(judge_long), _as_scores_long(primary_long)
    mets = _present(PL, metrics)
    last = _last_iters(PL)
    W = {"primary": {m: wide_by_persona(PL, m) for m in mets},
         "judge": {m: wide_by_persona(JL, m) for m in mets}}
    rows = []
    for method in methods:
        a_arm, b_arm = f"{method}_LA0", f"{method}_LA5"
        if a_arm not in last or b_arm not in last:
            continue
        for it in range(0, min(last[a_arm], last[b_arm]) + 1):
            a, b = model_name(method, 0, it), model_name(method, 5, it)
            for m in mets:
                rec = {"metric": m, "model_a": a, "model_b": b}
                for pref in ("primary", "judge"):
                    w = W[pref][m]
                    if a not in w.columns or b not in w.columns:
                        r = dict(n=0, mean_delta=np.nan, dz=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
                    else:
                        r = paired_arrays(w[a].to_numpy(), w[b].to_numpy())
                    rec.update({f"{pref}_n": r["n"], f"{pref}_delta": r["mean_delta"], f"{pref}_dz": r["dz"],
                                f"{pref}_ci_lo": r["ci_lo"], f"{pref}_ci_hi": r["ci_hi"], f"{pref}_p": r["p"]})
                rec["same_sign"] = bool(np.sign(rec["judge_delta"]) == np.sign(rec["primary_delta"]))
                rec["judge_ci_excl0"] = bool(rec["judge_ci_lo"] > 0 or rec["judge_ci_hi"] < 0)
                rec["primary_ci_excl0"] = bool(rec["primary_ci_lo"] > 0 or rec["primary_ci_hi"] < 0)
                tag = "Base" if it == 0 else f"I{it}"
                rec.update(method=method, iteration=it, contrast=f"{method}_LA0_{tag} − {method}_LA5_{tag}")
                rows.append(rec)
    cols = ["method", "iteration", "metric", "contrast", "primary_n",
            "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi", "primary_p", "primary_p_holm",
            "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p", "judge_p_holm",
            "same_sign", "judge_ci_excl0", "favours_primary", "favours_judge"]
    if not rows:
        return pd.DataFrame(columns=cols)
    pairs = pd.DataFrame(rows)
    pairs = holm_within(pairs, ["method", "metric"], "primary_p", "primary_p_holm")
    pairs = holm_within(pairs, ["method", "metric"], "judge_p", "judge_p_holm")
    pairs["favours_primary"] = [favours(m, d, "K0", "K5") for m, d in zip(pairs["metric"], pairs["primary_delta"])]
    pairs["favours_judge"] = [favours(m, d, "K0", "K5") for m, d in zip(pairs["metric"], pairs["judge_delta"])]
    order = list(RUBRICS) + sorted(set(pairs["metric"]) - set(RUBRICS))
    pairs["metric"] = pd.Categorical(pairs["metric"], order, ordered=True)
    pairs = pairs.sort_values(["method", "metric", "iteration"]).reset_index(drop=True)
    pairs["metric"] = pairs["metric"].astype(str)
    return pairs[cols]


# ── 2. sign ladder ────────────────────────────────────────────────────────────

def _extra_rungs(df: pd.DataFrame) -> pd.DataFrame:
    subs = [("primary p_holm < 0.05", df[df["primary_p_holm"] < 0.05]),
            ("judge p_holm < 0.05", df[df["judge_p_holm"] < 0.05]),
            ("both graders p_holm < 0.05", df[(df["primary_p_holm"] < 0.05) & (df["judge_p_holm"] < 0.05)]),
            ("iteration >= 1 (base row excluded)", df[df["iteration"] >= 1])]
    out = []
    for label, sub in subs:
        n = len(sub)
        out.append({"subset": label, "n_contrasts": n,
                    "n_same_sign": int(sub["same_sign"].sum()) if n else 0,
                    "pct_same_sign": round(100 * sub["same_sign"].mean(), 1) if n else np.nan})
    return pd.DataFrame(out)


def sign_ladder(pairs: pd.DataFrame, *, methods: Sequence[str] = METHODS,
                metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Sign preservation of the cross-K contrast under the held-out judge, laddered by |Δ primary|.

    Rows = :func:`~eda_analysis.reliability.sign_preservation`'s rungs (all contrasts, |Δ primary| ≥
    0.10 / 0.25 / 0.50, judge CI excludes 0) plus Holm rungs (primary / judge / both graders
    ``p_holm < .05``) and ``iteration >= 1`` (base-vs-base rows dropped) — for ``group`` = all
    contrasts, each method, each metric. ⚠ Thresholds are ABSOLUTE, so per-metric rows compare
    DOWN a rubric, never across (PCT / MICI live on 0-1). Reproduces ``cross_k_multijudge_ladder.csv``.
    """
    from . import reliability as R
    if metrics is None:
        metrics = [m for m in RUBRICS if m in set(pairs["metric"])]
    groups = ([("all cross-K contrasts", pairs)]
              + [(f"method={m}", pairs[pairs["method"] == m]) for m in methods]
              + [(f"metric={m}", pairs[pairs["metric"] == m]) for m in metrics])
    ladder = pd.concat([pd.concat([R.sign_preservation(sub), _extra_rungs(sub)], ignore_index=True).assign(group=label)
                        for label, sub in groups], ignore_index=True)
    return ladder[["group", "subset", "n_contrasts", "n_same_sign", "pct_same_sign"]]


# ── 3. gain retention by K ────────────────────────────────────────────────────

def retention_by_k(judge_long: pd.DataFrame, primary_long: pd.DataFrame, *,
                   reference_kinds: Sequence[str] = DEFAULT_REFERENCE_KINDS,
                   metrics: Optional[Sequence[str]] = None, arms: Sequence[str] = ARMS,
                   scale_floor: float = SCALE_FLOOR, rate_floor: float = RATE_FLOOR,
                   summary_metrics: Sequence[str] = ("Q1Q2", "Q1", "Q2", "MITI", "MICI"),
                   summary_iters: Optional[Sequence[int]] = None) -> Dict[str, pd.DataFrame]:
    """``retention = Δ held-out / Δ primary`` of every model state over a reference base, by K.

    Wraps :func:`~eda_analysis.reliability.gain_retention` per (arm, reference kind), once for the
    point-scale rubrics with ``min_primary_delta=scale_floor`` (0.15) and once for the 0-1 rate
    metrics PCT / MICI with ``rate_floor`` (0.05). ``reference_kinds`` are templates over
    ``{method}`` / ``{K}``: ``own_base`` (the arm's own base draw), ``method_LA0_base`` /
    ``method_LA5_base`` (the method's K=0 / K=5 base as a SHARED reference for both K arms — for a
    PTO_LA0 row ``own_base`` and ``method_LA0_base`` are the same reference and duplicate by design),
    ``eda_view_PTO_LA{K}_base`` (GRPO arms only: the tracked measurement family's convention).
    Iteration-0 rows under a shared reference are two INDEPENDENT base draws (noise floor).

    Returns ``{"retention": ret, "retention_summary": ret_sum}``. ``ret`` columns: ``arm, method, K,
    iteration, metric, ref_kind, reference, ref_is_own_base, n, delta_primary, delta_judge,
    retention, retention_ci_lo, retention_ci_hi, same_sign, min_primary_delta``. ``ret_sum`` puts
    K=0 and K=5 side by side (own-base reference) for ``summary_metrics`` at ``summary_iters``
    (default: 5 — the last iteration all four arms share — and each K=5 arm's endpoint) with
    ``cis_disjoint``. Reproduces ``cross_k_multijudge_retention{,_summary}.csv``. Cross-checks:
    GRPO_LA5 Q1 iteration 5 vs ``eda_view_PTO_LA5_base`` = 1.082 [0.936, 1.271] (the tracked
    measurement table gives the same point estimate with CI [0.937, 1.274] — a different model
    subset per call consumes the bootstrap generator differently); PTO_LA5 Q2 iteration 10 own-base
    = 0.562 vs PTO_LA0 0.849.
    """
    from . import reliability as R
    JL, PL = to_reliability_long(judge_long), to_reliability_long(primary_long)
    PLs = _as_scores_long(primary_long)
    mets = _present(PLs, metrics)
    last = _last_iters(PLs)
    scale_metrics = [m for m in mets if m not in RATE_METRICS]
    rate_metrics = [m for m in mets if m in RATE_METRICS]

    def _sub(long, models):
        return long[long["model"].isin(set(models))]

    ret_rows = []
    for arm in arms:
        if arm not in last:
            continue
        method, K = method_of(arm), k_of(arm)
        arm_models = [model_name(method, K, i) for i in range(0, last[arm] + 1)]
        refs = []
        for kind in reference_kinds:
            if kind == "own_base":
                refs.append((kind, model_name(method, K, 0)))
            elif kind.startswith("method_LA"):
                refs.append((kind, model_name(method, int(kind[len("method_LA")]), 0)))
            elif kind.startswith("eda_view_PTO_LA"):
                if method == "GRPO":
                    refs.append((kind.format(K=K), model_name("PTO", K, 0)))
            else:
                raise ValueError(f"unknown reference kind {kind!r}")
        for kind, ref in refs:
            models = sorted(set(arm_models) | {ref})
            parts = []
            for group, floor in ((scale_metrics, scale_floor), (rate_metrics, rate_floor)):
                if not group:
                    continue
                rt = R.gain_retention(_sub(JL, models), _sub(PL, models), reference_model=ref,
                                      metrics=group, min_primary_delta=floor)
                rt["min_primary_delta"] = floor
                parts.append(rt)
            if not parts:
                continue
            rt = pd.concat(parts, ignore_index=True)
            rt = rt[rt["model"].isin(arm_models)].copy()
            rt["arm"], rt["K"], rt["method"] = arm, K, method
            rt["iteration"] = rt["model"].map(R.model_iteration)
            rt["ref_kind"] = kind
            rt["ref_is_own_base"] = (ref == model_name(method, K, 0))
            ret_rows.append(rt)
    cols = ["arm", "method", "K", "iteration", "metric", "ref_kind", "reference", "ref_is_own_base", "n",
            "delta_primary", "delta_judge", "retention", "retention_ci_lo", "retention_ci_hi", "same_sign",
            "min_primary_delta"]
    if not ret_rows:
        return {"retention": pd.DataFrame(columns=cols), "retention_summary": pd.DataFrame()}
    ret = pd.concat(ret_rows, ignore_index=True)[cols]
    order = list(RUBRICS) + sorted(set(ret["metric"]) - set(RUBRICS))
    ret["metric"] = pd.Categorical(ret["metric"], order, ordered=True)
    ret = ret.sort_values(["method", "K", "ref_kind", "metric", "iteration"]).reset_index(drop=True)
    ret["metric"] = ret["metric"].astype(str)

    def ret_at(arm, it, m, kind="own_base"):
        r = ret[(ret["arm"] == arm) & (ret["iteration"] == it) & (ret["metric"] == m) & (ret["ref_kind"] == kind)]
        return r.iloc[0] if len(r) else None

    sum_rows = []
    for method in METHODS:
        if f"{method}_LA5" not in last or f"{method}_LA0" not in last:
            continue
        iters = summary_iters if summary_iters is not None else sorted({5, last[f"{method}_LA5"]})
        for it in iters:
            for m in summary_metrics:
                r0, r5 = ret_at(f"{method}_LA0", it, m), ret_at(f"{method}_LA5", it, m)
                if r0 is None or r5 is None:
                    continue
                disjoint = (not np.isnan(r0["retention_ci_lo"]) and not np.isnan(r5["retention_ci_lo"])
                            and (r0["retention_ci_hi"] < r5["retention_ci_lo"] or r5["retention_ci_hi"] < r0["retention_ci_lo"]))
                sum_rows.append({"method": method, "iteration": it, "metric": m,
                                 "K0_delta_primary": r0["delta_primary"], "K0_delta_judge": r0["delta_judge"],
                                 "K0_retention": r0["retention"], "K0_ci_lo": r0["retention_ci_lo"], "K0_ci_hi": r0["retention_ci_hi"],
                                 "K5_delta_primary": r5["delta_primary"], "K5_delta_judge": r5["delta_judge"],
                                 "K5_retention": r5["retention"], "K5_ci_lo": r5["retention_ci_lo"], "K5_ci_hi": r5["retention_ci_hi"],
                                 "cis_disjoint": bool(disjoint)})
    return {"retention": ret, "retention_summary": pd.DataFrame(sum_rows)}


# ── 4. the ledger ─────────────────────────────────────────────────────────────

def transfer_numbers(pairs: Optional[pd.DataFrame] = None, ladder: Optional[pd.DataFrame] = None,
                     retention: Optional[pd.DataFrame] = None,
                     retention_summary: Optional[pd.DataFrame] = None) -> dict:
    """Every number the write-up may quote, as ``{dotted.key: {"value", "source", "note"}}``.

    Key families match the paper's frozen ``cross_k_multijudge.json``: ``kcontrast.<method>.<metric>.iter<n>``,
    ``ladder.<group>.<subset>``, ``retention.<arm>.<metric>.iter<n>.<ref_kind>``,
    ``retention_summary.<method>.<metric>.iter<n>``. Only frames passed are emitted; feed the result
    to ``exports.save_numbers``.
    """
    L: Dict[str, dict] = {}

    def put(key, value, *, source="", note=""):
        L[key] = {"value": value, "source": source, "note": note}

    def row(r, cols):
        return {c: _nan_none(r[c]) for c in cols}

    if pairs is not None and len(pairs):
        for _, r in pairs.iterrows():
            put(f"kcontrast.{r['method']}.{r['metric']}.iter{int(r['iteration'])}",
                row(r, ["primary_n", "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi", "primary_p",
                        "primary_p_holm", "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p",
                        "judge_p_holm", "same_sign", "judge_ci_excl0", "favours_primary", "favours_judge"]),
                source=f"cross_k_pairs row method={r['method']} metric={r['metric']} iteration={int(r['iteration'])}",
                note="+ => K=0 higher; MICI lower-better")
    if ladder is not None and len(ladder):
        for _, r in ladder.iterrows():
            put(f"ladder.{r['group']}.{r['subset']}", row(r, ["n_contrasts", "n_same_sign", "pct_same_sign"]),
                source=f"sign_ladder row group={r['group']} subset={r['subset']}")
    if retention is not None and len(retention):
        for _, r in retention.iterrows():
            put(f"retention.{r['arm']}.{r['metric']}.iter{int(r['iteration'])}.{r['ref_kind']}",
                row(r, ["reference", "n", "delta_primary", "delta_judge", "retention", "retention_ci_lo",
                        "retention_ci_hi", "same_sign"]),
                source=f"retention_by_k['retention'] row arm={r['arm']} metric={r['metric']} iteration={int(r['iteration'])} ref_kind={r['ref_kind']}")
    if retention_summary is not None and len(retention_summary):
        for _, r in retention_summary.iterrows():
            put(f"retention_summary.{r['method']}.{r['metric']}.iter{int(r['iteration'])}",
                row(r, [c for c in retention_summary.columns if c not in ("method", "metric", "iteration")]),
                source=f"retention_by_k['retention_summary'] row method={r['method']} metric={r['metric']} iteration={int(r['iteration'])}")
    return L
