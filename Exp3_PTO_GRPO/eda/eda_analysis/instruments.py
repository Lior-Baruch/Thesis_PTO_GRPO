"""
instruments.py — K-turn look-ahead read on instruments OUTSIDE the training reward.

Promoted (2026-08-18) from the look-ahead paper's generator
``papers/2026_lookahead_pto_grpo/analysis/held_out_instruments.py`` (paper-local script; its outputs
were frozen at commit abe5cb3, 2026-08-18). Its frozen outputs —
``papers/2026_lookahead_pto_grpo/tables/held_out_instruments_*.csv`` and
``analysis/out/held_out_instruments.json`` — are the FIXTURE these functions reproduce (means /
dz / p exactly; bootstrap CI bounds within ~0.02, because the paper seeded its bootstrap with 0
while the package uses :data:`constants.BOOT_SEED`; counts exactly). Rendered by
``notebooks/lookahead/behaviour.ipynb`` into ``results/lookahead/behaviour/`` (judge-invariant:
both graders live inside every frame, in a ``judge`` column — never averaged).

Four questions, each read under BOTH graders (the training oracle gpt-4o-mini and the held-out
Claude Haiku 4.5), **paired on ``persona_id`` (never ``file_index``)**, K-contrast sign
**``+ => K=0 higher`` (K0 − K5)**:

1. **WAI-SR subscale composition** — Task (items 1,2,10,12) / Goal (4,6,8,11) / Bond (3,5,7,9),
   the WAI-SR standard map (Hatcher & Gillaspy 2006), which is *identical* to the score lake's
   ``WAI_{Task,Goal,Bond}_Mean`` columns (``code/questionnaires.py``): :func:`wai_subscale_parity`
   asserts it (max |diff| < 1e-9 on every conversation, both graders). Per arm × iteration levels +
   gain over own base (:func:`wai_subscales`); persona-paired K0−K5 contrast on the *bond excess*
   = Bond − mean(Goal, Task) at every matched iteration (:func:`wai_kcontrast`); the endpoint
   gain-by-subscale figure data (:func:`wai_fig_data`).
2. **PCT (patient change talk)** — the lake's ``PCT`` metric is ``PCT_ChangeProp`` = CT/(CT+ST);
   the components (three 1-5 globals + utterance counts) come from
   :func:`behavior.load_pct_behavior`. Paired K0−K5 by matched iteration (:func:`pct_kcontrast`).
3. **Q2 item profile** — per-item endpoint gain over own base for every arm + the per-item K0−K5
   contrast at the matched endpoint (:func:`q2_items`). Items 1/2/3/10 = the "self-disclosure"
   face-content group of ``constants.Q2_ITEM_GROUPS`` (analytical, not a validated subscale);
   3 and 10 = emotional self-disclosure ("shared his feelings", "said when happy/sad").
4. **Heterogeneity** — the K0−K5 contrast on Q1Q2 / MICI / PCT WITHIN patient cooperation level
   (``High → Cooperative``, ``StartLowAndChangesToHigh → Warms up``, ``Low → Resistant``; 32
   personas each) at the matched endpoint and at each arm's own-oracle best iteration
   (:func:`hetero_kcontrast`).

Conventions every table shares (state them in the notebook captions):

* ``mean_K0`` / ``mean_K5`` = arm means on the paired personas; ``mean_delta`` = mean of the
  paired deltas; ``dz`` = mean / SD (ddof=1) of the paired deltas; ``ci_lo``/``ci_hi`` = 95%
  percentile bootstrap over the paired deltas (:func:`stats.paired_arrays`, 2000 draws,
  ``BOOT_SEED``); ``p`` = Wilcoxon signed-rank; ``p_holm`` = Holm within the family named in
  each function's docstring.
* **Iteration 0 = two independent base draws** of the same base model (one per arm), so the
  iteration-0 K contrast is the noise floor, not a treatment effect.
* **Censoring:** GRPO_LA5 is right-censored at iteration 5 (PTO arms and GRPO_LA0 run to 10),
  so the matched endpoint is PTO iter 10 vs 10 and GRPO iter 5 vs 5, and every "endpoint" table
  says which iteration it read (``target_iter`` / ``iter_K0`` / ``iter_K5`` / ``@N`` columns).
* **Own-oracle best iteration** (:func:`data.best_iteration_by_arm`) is selected on the TRAINING
  oracle's Q1Q2 mean (the primary grader) and reused for the held-out grader, so both graders
  judge the same checkpoints.
* Graders are side by side (a ``judge`` column, short label from :func:`constants.judge_dirname`),
  **never averaged** — the primary grader was the training reward and the second is held out.

Contract (as for every promoted module): functions take frames and return tidy
:class:`pandas.DataFrame`s / dicts — NO disk writes; the notebook owns ``exports.*``. The only
disk reader is :func:`instrument_frames_by_judge`, the per-judge item-level loader (it swaps the
active judge and restores it). :func:`instruments_numbers` returns the quotable-numbers ledger
(``{dotted.key: {"value","source","note"}}``) for ``exports.save_numbers``.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import (Q2_ITEM_SHORT, Q2_ITEM_GROUP_OF, active_judge, active_judge_rep,
                        judge_dirname, set_active_judge)

__all__ = [
    # constants / notes
    "ARM_ORDER", "METHODS", "WAI_SUBSCALES", "WAI_MEASURES", "COOP_LABEL", "COOP_ORDER",
    "PCT_METRICS", "PCT_LABEL", "Q2_SELF_DISCLOSURE", "Q2_EMOTIONAL", "HETERO_METRICS",
    "SIGN_NOTE", "PAIR_NOTE", "CENSOR_NOTE",
    # loading + endpoints
    "instrument_frames_by_judge", "endpoints", "matched_endpoints",
    # WAI-SR
    "wai_conversation_frame", "wai_subscale_parity", "wai_subscales", "wai_kcontrast",
    "wai_fig_data",
    # PCT / Q2 / heterogeneity
    "pct_kcontrast", "q2_items", "hetero_kcontrast", "hetero_ceiling",
    # ledger
    "instruments_numbers",
]

# ── constants ─────────────────────────────────────────────────────────────────
ARM_ORDER = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]
METHODS = ["PTO", "GRPO"]

# WAI-SR standard subscale map (Hatcher & Gillaspy 2006) — identical to code/questionnaires.py
# ``WAI_Goal / WAI_Task / WAI_Bond`` (the columns the package's wai_subscales figure reads).
WAI_SUBSCALES = {"Task": [1, 2, 10, 12], "Goal": [4, 6, 8, 11], "Bond": [3, 5, 7, 9]}
# Per-conversation WAI measures the tables carry (in this order). ``WAI_total`` = mean of the
# 12 items (the item-derived total, not the lake's ``WAI_SR_Mean`` column — they agree, but the
# parity assertion covers only the three subscales).
WAI_MEASURES = ["Task", "Goal", "Bond", "bond_excess", "WAI_total"]
_WAI_TOTAL_ITEMS = "WAI_total_items"          # internal name of the item-derived total
_WAI_INTERNAL = ["Task", "Goal", "Bond", "bond_excess", _WAI_TOTAL_ITEMS]

from .constants import COOP_LABEL, COOP_ORDER  # noqa: E402,F401
from .constants import k_of as _k_of_canonical, method_of as _method_of_canonical  # noqa: E402
from .ledger import json_scalar, ledger_entry, round3  # noqa: E402,F401

PCT_METRICS = ["PCT_ChangeProp", "PCT_GlobalMean", "PCT_Importance", "PCT_Confidence",
               "PCT_Readiness", "PCT_ChangeTalk", "PCT_SustainTalk", "PCT_Neutral",
               "PCT_BehaviorTotal"]
PCT_LABEL = {"PCT_ChangeProp": "ChangeProp = CT/(CT+ST) [= lake 'PCT']",
             "PCT_GlobalMean": "GlobalMean (Importance/Confidence/Readiness, 1-5)",
             "PCT_Importance": "Importance (1-5)", "PCT_Confidence": "Confidence (1-5)",
             "PCT_Readiness": "Readiness (1-5)", "PCT_ChangeTalk": "change-talk utterances (count)",
             "PCT_SustainTalk": "sustain-talk utterances (count)",
             "PCT_Neutral": "neutral utterances (count)",
             "PCT_BehaviorTotal": "patient utterances (count)"}

Q2_SELF_DISCLOSURE = [1, 2, 3, 10]           # the face-content "self-disclosure" group
Q2_EMOTIONAL = [3, 10]                       # emotional self-disclosure within it
Q2_ITEMS = list(range(1, 18))
HETERO_METRICS = ["Q1Q2", "MICI", "PCT"]

SIGN_NOTE = "K-contrast sign: + => K=0 higher (K0 - K5)."
PAIR_NOTE = "Paired on persona_id (the recovered patient persona), never file_index."
CENSOR_NOTE = "GRPO_LA5 is right-censored at iteration 5 (PTO arms and GRPO_LA0 run to 10)."


# ── small helpers ─────────────────────────────────────────────────────────────
def _k_of(arm: str) -> int:
    """Re-export of :func:`eda_analysis.constants.k_of`."""
    return _k_of_canonical(arm)


def _method_of(arm: str) -> str:
    return arm.split("_")[0]


def _model(method: str, K: int, it: int) -> str:
    """Score-lake model name for ``<method>_LA<K>`` at iteration ``it`` (0 = Base)."""
    return f"{method}Exp3_LA{K}_{'Base' if it == 0 else f'I{it}'}"


def _arms_present(df: pd.DataFrame, arms: Optional[Sequence[str]] = None) -> List[str]:
    have = set(df["arm"].unique())
    if arms is None:
        arms = ARM_ORDER + sorted(a for a in have if a not in ARM_ORDER)
    return [a for a in arms if a in have]


def _pair(df: pd.DataFrame, value: str, model_a: str, model_b: str, *, key: str = "persona_id") -> dict:
    """persona-aligned paired contrast ``model_a - model_b`` on ``value``.

    Wraps :func:`stats.paired_arrays` (mean_delta / dz / bootstrap CI / Wilcoxon p / n) and adds
    ``mean_a`` / ``mean_b`` = the two arm means on the paired personas only.
    """
    from .stats import paired_arrays
    a = df[df["model"] == model_a][[key, value]].dropna().groupby(key)[value].mean()
    b = df[df["model"] == model_b][[key, value]].dropna().groupby(key)[value].mean()
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    out = paired_arrays(m["a"].to_numpy(), m["b"].to_numpy())
    out["mean_a"] = float(m["a"].mean()) if len(m) else np.nan
    out["mean_b"] = float(m["b"].mean()) if len(m) else np.nan
    return out


def _iters_both(df: pd.DataFrame, method: str) -> List[int]:
    i0 = set(df.loc[df["arm"] == f"{method}_LA0", "iteration"])
    i5 = set(df.loc[df["arm"] == f"{method}_LA5", "iteration"])
    return sorted(int(i) for i in (i0 & i5))


def _k_contrast_by_iter(df: pd.DataFrame, value: str, method: str) -> pd.DataFrame:
    """K0 − K5 at every matched iteration (iteration 0 = two independent base draws).

    ``p_holm`` = Holm across the iterations of this (judge, method, value) family.
    """
    from .stats import holm
    rows = []
    for it in _iters_both(df, method):
        r = _pair(df, value, _model(method, 0, it), _model(method, 5, it))
        rows.append({"method": method, "iteration": it, "mean_K0": r["mean_a"],
                     "mean_K5": r["mean_b"], "n": r["n"], "mean_delta": r["mean_delta"],
                     "dz": r["dz"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"].to_numpy())
    return out


def _by_judge(by_judge: Dict[str, object], key: str) -> Dict[str, pd.DataFrame]:
    """Accept ``{judge: frame}`` OR the nested ``{judge: {key: frame, ...}}`` the loader returns."""
    out = {}
    for j, v in by_judge.items():
        out[j] = v[key] if isinstance(v, dict) else v
    return out


_fmt3 = round3


_clean = json_scalar          # one definition — see eda_analysis/ledger.py
#   ⚠ this copy had NO np.bool_ branch, so a bool could reach json.dumps raw.


# ── endpoints ─────────────────────────────────────────────────────────────────
def endpoints(df: pd.DataFrame, arms: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """``{arm: max scored iteration}`` — each arm's endpoint (GRPO_LA5 is censored at 5)."""
    return {arm: int(df.loc[df["arm"] == arm, "iteration"].max()) for arm in _arms_present(df, arms)}


def matched_endpoints(end: Dict[str, int], methods: Sequence[str] = METHODS) -> Dict[str, int]:
    """``{method: min(END[<m>_LA0], END[<m>_LA5])}`` — the K-matched endpoint per method
    (PTO 10 vs 10, GRPO 5 vs 5 today)."""
    out = {}
    for m in methods:
        a, b = f"{m}_LA0", f"{m}_LA5"
        if a in end and b in end:
            out[m] = int(min(end[a], end[b]))
    return out


# ── loading (the one disk reader here) ────────────────────────────────────────
def instrument_frames_by_judge(arms, judges: Optional[Sequence[str]] = None, *,
                               check_parity: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Item-level frames for every grader — ``{judge_label: {"wai_items", "wai_subscales",
    "wai_conv", "q2_items", "pct"}}``.

    Reads, under each judge tag in turn (the active judge is swapped and RESTORED afterwards):
    ``data.load_items("WAI-SR")``, ``data.load_subscales()`` (WAI-SR rows only),
    ``data.load_items("Q2")`` and ``behavior.load_pct_behavior(attach_persona=True)`` — the same
    loaders the paper generator used — with ``persona_id`` + characteristics attached per arm
    (each arm's own seed). ``judges=None`` = the primary oracle first, then every second judge on
    disk (:func:`reliability.second_judge_tags`). Keys are the short labels
    (:func:`constants.judge_dirname`: ``gpt-4o-mini``, ``claude-haiku-4-5``), the same strings the
    tables' ``judge`` column carries.

    ``wai_conv`` is :func:`wai_conversation_frame` (per-conversation Task/Goal/Bond/bond_excess/
    WAI_total_items); with ``check_parity`` the item-derived subscales are asserted against the
    lake's ``WAI_{Task,Goal,Bond}_Mean`` columns (:func:`wai_subscale_parity`).

    ⚠ Do not interleave with other loaders while this runs: the active judge is module-level state.
    """
    from . import data as D, behavior as B
    if judges is None:
        from .reliability import second_judge_tags
        judges = [""] + list(second_judge_tags())
    prev = (active_judge(), active_judge_rep())
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    try:
        for tag in judges:
            set_active_judge(tag, 0)
            label = judge_dirname(tag)
            wai = _attach(D.load_items("WAI-SR", arms), arms)
            sub = _attach(D.load_subscales(arms), arms)
            sub = sub[sub["parent"] == "WAI-SR"].copy()
            q2 = _attach(D.load_items("Q2", arms), arms)
            pct = B.load_pct_behavior(arms, attach_persona=True)
            conv = wai_conversation_frame(wai, sub if check_parity else None)
            out[label] = {"wai_items": wai, "wai_subscales": sub, "wai_conv": conv,
                          "q2_items": q2, "pct": pct}
    finally:
        set_active_judge(*prev)
    return out


def _attach(df: pd.DataFrame, arms) -> pd.DataFrame:
    """persona_id + characteristics per arm (each arm's own seed; all 42 today)."""
    from .data import attach_personas
    if df.empty:
        return df
    seed_by_arm = {a.label: a.seed for a in arms}
    parts = [attach_personas(g, seed_by_arm.get(lab, 42))
             for lab, g in df.groupby("arm", sort=False)]
    return pd.concat(parts, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WAI-SR SUBSCALES
# ═══════════════════════════════════════════════════════════════════════════════
def wai_conversation_frame(wai_items: pd.DataFrame,
                           wai_subscales_lake: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per conversation: the 12 WAI-SR items as columns 1..12 + ``Task``/``Goal``/``Bond``
    (standard map means), ``bond_excess`` = Bond − mean(Goal, Task), ``WAI_total_items`` = mean of
    the 12 items. Keys ``arm, model, iteration, file_index, persona_id``.

    Idempotent: a frame that already carries the ``Task`` column (i.e. this function's output, or
    the loader's ``wai_conv``) is returned unchanged. When ``wai_subscales_lake`` (the
    ``data.load_subscales`` WAI-SR rows) is given, :func:`wai_subscale_parity` is ASSERTED first
    (the WAI-SR standard map must reproduce the lake's ``WAI_{Task,Goal,Bond}_Mean`` to 1e-9).
    """
    if "Task" in wai_items.columns and "bond_excess" in wai_items.columns:
        piv = wai_items
    else:
        keys = ["arm", "model", "iteration", "file_index", "persona_id"]
        piv = wai_items.pivot_table(index=keys, columns="item", values="score").reset_index()
        for name, ids in WAI_SUBSCALES.items():
            piv[name] = piv[[i for i in ids]].mean(axis=1)
        piv["bond_excess"] = piv["Bond"] - (piv["Goal"] + piv["Task"]) / 2
        piv[_WAI_TOTAL_ITEMS] = piv[list(range(1, 13))].mean(axis=1)
    if wai_subscales_lake is not None:
        chk = wai_subscale_parity(piv, wai_subscales_lake)
        assert chk["max_abs_diff_items_vs_lake"] < 1e-9, (
            "WAI-SR standard map (Task 1,2,10,12 / Goal 4,6,8,11 / Bond 3,5,7,9) disagrees with the "
            f"score lake's WAI_{{Task,Goal,Bond}}_Mean columns: max |diff| = {chk['max_abs_diff_items_vs_lake']:.3g}")
    return piv


def wai_subscale_parity(wai_conv: pd.DataFrame, wai_subscales_lake: pd.DataFrame) -> dict:
    """``{"n_convs", "max_abs_diff_items_vs_lake"}`` — item-derived Task/Goal/Bond vs the lake's
    ``WAI_{Task,Goal,Bond}_Mean`` (``data.load_subscales`` WAI-SR rows), joined on
    ``(arm, model, file_index)``. The paper reported 3,744 convs and 0.0 under both graders."""
    piv = wai_conversation_frame(wai_conv)
    lake = wai_subscales_lake.pivot_table(index=["arm", "model", "file_index"],
                                          columns="subscale", values="score").reset_index()
    m = piv.merge(lake, on=["arm", "model", "file_index"], suffixes=("", "_lake"))
    if m.empty:
        return {"n_convs": 0, "max_abs_diff_items_vs_lake": float("nan")}
    maxdiff = max(float((m[s] - m[f"{s}_lake"]).abs().max()) for s in WAI_SUBSCALES)
    return {"n_convs": int(len(m)), "max_abs_diff_items_vs_lake": maxdiff}


def wai_subscales(items_by_judge: Dict[str, object],
                  subscales_by_judge: Optional[Dict[str, pd.DataFrame]] = None, *,
                  arms: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """WAI-SR subscale levels + gain over own base, per grader × arm × iteration
    (fixture ``held_out_instruments_wai``).

    ``items_by_judge`` = ``{judge_label: WAI-SR items long}`` (or the loader's nested dict);
    ``subscales_by_judge`` (optional) = the lake subscale frames → parity asserted per grader.
    Columns: ``judge, arm, iteration, n, Task, Task_gain, Goal, Goal_gain, Bond, Bond_gain,
    bond_excess, bond_excess_gain, WAI_total, WAI_total_gain``. Levels are plain conversation
    means; ``*_gain`` = mean persona-paired difference vs the arm's OWN iteration-0 base (1-5
    Likert points; iteration-0 rows have gain 0 by construction). ``n`` = conversations.
    """
    convs = _by_judge(items_by_judge, "wai_conv") if _nested(items_by_judge) else items_by_judge
    subs = _by_judge(subscales_by_judge, "wai_subscales") if subscales_by_judge and _nested(subscales_by_judge) else (subscales_by_judge or {})
    rows = []
    for j, it_frame in convs.items():
        piv = wai_conversation_frame(it_frame, subs.get(j))
        for arm in _arms_present(piv, arms):
            g = piv[piv["arm"] == arm]
            base = g[g["iteration"] == 0].set_index("persona_id")
            for it in sorted(g["iteration"].unique()):
                gi = g[g["iteration"] == it]
                row = {"judge": j, "arm": arm, "iteration": int(it), "n": int(len(gi))}
                for s in _WAI_INTERNAL:
                    row[s] = float(gi[s].mean())
                    mm = gi.set_index("persona_id")[s].to_frame("a").join(base[s].rename("b"), how="inner")
                    row[f"{s}_gain"] = float((mm["a"] - mm["b"]).mean())
                rows.append(row)
    out = pd.DataFrame(rows)
    return out.rename(columns={_WAI_TOTAL_ITEMS: "WAI_total", f"{_WAI_TOTAL_ITEMS}_gain": "WAI_total_gain"})


def wai_kcontrast(items_by_judge: Dict[str, object],
                  subscales_by_judge: Optional[Dict[str, pd.DataFrame]] = None, *,
                  methods: Sequence[str] = METHODS,
                  measures: Sequence[str] = ("bond_excess", "Bond", "Goal", "Task", "WAI_total")
                  ) -> pd.DataFrame:
    """Persona-paired K0−K5 contrast on the WAI-SR subscales, per grader × method × measure ×
    matched iteration (fixture ``held_out_instruments_wai_kcontrast``).

    Columns: ``judge, method, measure, iteration, mean_K0, mean_K5, n, mean_delta, dz, ci_lo,
    ci_hi, p, p_holm``. ``bond_excess`` = Bond − mean(Goal, Task): a POSITIVE delta means the K=0
    arm's alliance gain is MORE bond-weighted (relational) relative to its task/goal component than
    the K=5 arm's. ``p_holm`` = Holm within (judge, method, measure) across iterations. Iteration 0
    = two independent base draws (noise floor). Sign ``+ => K=0 higher``.
    """
    convs = _by_judge(items_by_judge, "wai_conv") if _nested(items_by_judge) else items_by_judge
    subs = _by_judge(subscales_by_judge, "wai_subscales") if subscales_by_judge and _nested(subscales_by_judge) else (subscales_by_judge or {})
    parts = []
    for j, it_frame in convs.items():
        piv = wai_conversation_frame(it_frame, subs.get(j))
        for method in methods:
            for val in measures:
                col = _WAI_TOTAL_ITEMS if val == "WAI_total" else val
                t = _k_contrast_by_iter(piv, col, method)
                if t.empty:
                    continue
                t.insert(0, "judge", j); t.insert(2, "measure", val)
                parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["judge", "method", "measure", "iteration", "mean_K0", "mean_K5", "n",
                                     "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"])
    return pd.concat(parts, ignore_index=True)


def wai_fig_data(items_by_judge: Dict[str, object], *,
                 end: Optional[Dict[str, int]] = None,
                 matched_end: Optional[Dict[str, int]] = None,
                 arms: Optional[Sequence[str]] = None,
                 subscales: Sequence[str] = ("Task", "Goal", "Bond")) -> pd.DataFrame:
    """Data behind :func:`plotting.instruments.wai_fig` (fixture ``held_out_instruments_fig_wai_data``):
    persona-paired gain over own base per WAI-SR subscale at each arm's endpoint, PLUS the
    K=0 arm of any censored method at the matched iteration (GRPO_LA0 @ 5 beside GRPO_LA5 @ 5),
    both graders. Columns ``judge, arm, iteration, subscale, gain, ci_lo, ci_hi, n``
    (95% percentile-bootstrap CI over the paired deltas).

    ``end`` / ``matched_end`` default to :func:`endpoints` / :func:`matched_endpoints` of the
    frames themselves. Series order = arm order, endpoint before the matched extra iteration.
    """
    convs = _by_judge(items_by_judge, "wai_conv") if _nested(items_by_judge) else items_by_judge
    first = next(iter(convs.values()))
    end = dict(end) if end is not None else endpoints(first, arms)
    matched_end = dict(matched_end) if matched_end is not None else matched_endpoints(end)
    series = []
    for arm in _arms_present(first, arms):
        if arm not in end:
            continue
        series.append((arm, int(end[arm])))
        me = matched_end.get(_method_of(arm))
        if _k_of(arm) == 0 and me is not None and me != end[arm]:
            series.append((arm, int(me)))
    rows = []
    for j, it_frame in convs.items():
        piv = wai_conversation_frame(it_frame)
        for arm, it in series:
            g = piv[piv["arm"] == arm]
            meth, K = _method_of(arm), _k_of(arm)
            for s in subscales:
                r = _pair(g, s, _model(meth, K, it), _model(meth, K, 0))
                rows.append({"judge": j, "arm": arm, "iteration": it, "subscale": s,
                             "gain": r["mean_delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"]})
    return pd.DataFrame(rows)


def _nested(by_judge: Dict[str, object]) -> bool:
    return any(isinstance(v, dict) for v in by_judge.values())


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PCT — paired K0−K5 by iteration on the lake metric + components
# ═══════════════════════════════════════════════════════════════════════════════
def pct_kcontrast(pct_by_judge: Dict[str, object], *, methods: Sequence[str] = METHODS,
                  metrics: Sequence[str] = PCT_METRICS) -> pd.DataFrame:
    """Persona-paired K0−K5 contrast on PCT (patient change talk) and its components, per grader ×
    method × metric × matched iteration (fixture ``held_out_instruments_pct``).

    ``pct_by_judge`` = ``{judge_label: behavior.load_pct_behavior frame}`` (persona attached; or
    the loader's nested dict). Columns: ``judge, method, metric, iteration, mean_K0, mean_K5, n,
    mean_delta, dz, ci_lo, ci_hi, p, p_holm``. ``PCT_ChangeProp`` = CT/(CT+ST) is the score lake's
    ``PCT`` metric (higher = more change talk); ``PCT_GlobalMean`` = mean of the three 1-5 patient
    globals; the three utterance counts sum to ``PCT_BehaviorTotal``. Metrics absent / all-NaN in a
    frame are skipped. ``p_holm`` = Holm within (judge, method, metric) across iterations.
    Sign ``+ => K=0 higher``. Iteration 0 = two independent base draws.
    """
    frames = _by_judge(pct_by_judge, "pct") if _nested(pct_by_judge) else pct_by_judge
    parts = []
    for j, pct in frames.items():
        for method in methods:
            for m in metrics:
                if m not in pct.columns or pct[m].notna().sum() == 0:
                    continue
                t = _k_contrast_by_iter(pct, m, method)
                if t.empty:
                    continue
                t.insert(0, "judge", j); t.insert(2, "metric", m)
                parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["judge", "method", "metric", "iteration", "mean_K0", "mean_K5", "n",
                                     "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"])
    return pd.concat(parts, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Q2 ITEM PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
def q2_items(q2_by_judge: Dict[str, object], *, end: Optional[Dict[str, int]] = None,
             matched_end: Optional[Dict[str, int]] = None, arms: Optional[Sequence[str]] = None,
             methods: Sequence[str] = METHODS) -> Dict[str, pd.DataFrame]:
    """The Q2 item profile — three frames keyed like the fixture tables:

    * ``q2items`` (wide, paper-facing; one row per (judge, item)): ``judge, item, short, group,
      base_mean(4 arms), <arm>_gain@N ..., <method>_K0-K5@N, <method>_dz, <method>_p_holm ...``.
      ``<arm>_gain@N`` = persona-paired mean gain over the arm's OWN base at its endpoint N;
      ``base_mean(4 arms)`` = mean of the arms' independent base draws (descriptive);
      ``<method>_K0-K5@N`` = the K contrast at the matched endpoint (sign ``+ => K=0 higher``),
      ``p_holm`` = Holm across the 17 items within (judge, method).
    * ``q2items_long``: per (judge, arm, item) the endpoint gain with 95% bootstrap CI, dz and
      Wilcoxon p; ``base``/``target`` = arm means on the paired personas; ``target_iter``.
    * ``q2items_kcontrast``: per (judge, method, item) the K0−K5 contrast at the matched endpoint
      (``k_delta, k_dz, k_ci_lo, k_ci_hi, k_p, k_p_holm``, ``mean_K0``/``mean_K5``).

    ``q2_by_judge`` = ``{judge_label: data.load_items("Q2") long, persona attached}`` (or the
    loader's nested dict). Groups are the face-content reading of ``constants.Q2_ITEM_GROUPS``
    (analytical, not a validated subscale); items 1,2,3,10 = self-disclosure, 3 and 10 = emotional
    self-disclosure. GRPO_LA5 is right-censored at 5.
    """
    frames = _by_judge(q2_by_judge, "q2_items") if _nested(q2_by_judge) else q2_by_judge
    first = next(iter(frames.values()))
    arm_list = _arms_present(first, arms)
    end = dict(end) if end is not None else endpoints(first, arm_list)
    matched_end = dict(matched_end) if matched_end is not None else matched_endpoints(end, methods)
    from .stats import holm

    long_rows = []
    for j, q2 in frames.items():
        for arm in arm_list:
            g = q2[q2["arm"] == arm]
            meth, K = _method_of(arm), _k_of(arm)
            for item in Q2_ITEMS:
                gi = g[g["item"] == item]
                r = _pair(gi, "score", _model(meth, K, end[arm]), _model(meth, K, 0))
                long_rows.append({"judge": j, "arm": arm, "item": item, "short": Q2_ITEM_SHORT[item],
                                  "group": Q2_ITEM_GROUP_OF[item], "target_iter": int(end[arm]), "n": r["n"],
                                  "base": r["mean_b"], "target": r["mean_a"], "gain": r["mean_delta"],
                                  "gain_ci_lo": r["ci_lo"], "gain_ci_hi": r["ci_hi"], "gain_dz": r["dz"],
                                  "gain_p": r["p"]})
    Q2L = pd.DataFrame(long_rows)

    k_parts = []
    for j, q2 in frames.items():
        for method in methods:
            if method not in matched_end:
                continue
            it = matched_end[method]
            rows = []
            for item in Q2_ITEMS:
                gi = q2[q2["item"] == item]
                r = _pair(gi, "score", _model(method, 0, it), _model(method, 5, it))
                rows.append({"judge": j, "method": method, "iteration": it, "item": item,
                             "short": Q2_ITEM_SHORT[item], "group": Q2_ITEM_GROUP_OF[item], "n": r["n"],
                             "mean_K0": r["mean_a"], "mean_K5": r["mean_b"], "k_delta": r["mean_delta"],
                             "k_dz": r["dz"], "k_ci_lo": r["ci_lo"], "k_ci_hi": r["ci_hi"], "k_p": r["p"]})
            t = pd.DataFrame(rows); t["k_p_holm"] = holm(t["k_p"].to_numpy())
            k_parts.append(t)
    Q2K = pd.concat(k_parts, ignore_index=True) if k_parts else pd.DataFrame(
        columns=["judge", "method", "iteration", "item", "short", "group", "n", "mean_K0", "mean_K5",
                 "k_delta", "k_dz", "k_ci_lo", "k_ci_hi", "k_p", "k_p_holm"])

    wide_rows = []
    for j in frames:
        for item in Q2_ITEMS:
            row = {"judge": j, "item": item, "short": Q2_ITEM_SHORT[item], "group": Q2_ITEM_GROUP_OF[item]}
            bases = []
            for arm in arm_list:
                r = Q2L[(Q2L.judge == j) & (Q2L.arm == arm) & (Q2L.item == item)].iloc[0]
                bases.append(r["base"]); row[f"{arm}_gain@{end[arm]}"] = r["gain"]
            row[f"base_mean({len(arm_list)} arms)"] = float(np.mean(bases))
            for method in methods:
                if method not in matched_end:
                    continue
                r = Q2K[(Q2K.judge == j) & (Q2K.method == method) & (Q2K.item == item)].iloc[0]
                row[f"{method}_K0-K5@{matched_end[method]}"] = r["k_delta"]
                row[f"{method}_dz"] = r["k_dz"]; row[f"{method}_p_holm"] = r["k_p_holm"]
            wide_rows.append(row)
    Q2W = pd.DataFrame(wide_rows)
    cols = (["judge", "item", "short", "group", f"base_mean({len(arm_list)} arms)"]
            + [f"{a}_gain@{end[a]}" for a in arm_list]
            + [c for m in methods if m in matched_end
               for c in (f"{m}_K0-K5@{matched_end[m]}", f"{m}_dz", f"{m}_p_holm")])
    Q2W = Q2W[cols]
    return {"q2items": Q2W, "q2items_long": Q2L, "q2items_kcontrast": Q2K}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HETEROGENEITY by cooperation level
# ═══════════════════════════════════════════════════════════════════════════════
def _primary_label(by_judge: Dict[str, object], primary: Optional[str]) -> str:
    if primary is not None:
        return primary
    p = judge_dirname("")
    return p if p in by_judge else next(iter(by_judge))


def hetero_kcontrast(scores_by_judge: Dict[str, pd.DataFrame], *,
                     best_by_arm: Optional[Dict[str, int]] = None,
                     end: Optional[Dict[str, int]] = None,
                     matched_end: Optional[Dict[str, int]] = None,
                     metrics: Sequence[str] = HETERO_METRICS,
                     methods: Sequence[str] = METHODS,
                     primary: Optional[str] = None) -> pd.DataFrame:
    """K0−K5 contrast WITHIN patient cooperation level (fixture ``held_out_instruments_hetero``).

    ``scores_by_judge`` = ``{judge_label: scores_long}`` (``cross_k_scores`` / ``scores_by_judge``
    output — persona characteristics attached, ``cooperation_level`` present). Strata: ``High →
    Cooperative``, ``StartLowAndChangesToHigh → Warms up``, ``Low → Resistant`` (32 personas each)
    + an ``All`` reference row (96 personas). Two targets per method: ``matched_final`` (PTO iter
    10 vs 10, GRPO iter 5 vs 5 — GRPO_LA5 is right-censored at 5) and ``own_best`` (each arm at its
    own-oracle best iteration; ``best_by_arm`` defaults to :func:`data.best_iteration_by_arm` on
    the PRIMARY grader's frame — selection on the training oracle's Q1Q2, reused for the held-out
    grader so both graders judge the same checkpoints).

    Columns: ``judge, method, metric, target, iter_K0, iter_K5, cooperation, n, mean_K0, mean_K5,
    mean_delta, dz, ci_lo, ci_hi, p, p_holm, share_K0_ge_4.5, share_K5_ge_4.5``. Metrics: Q1Q2
    (training reward, 1-5), MICI (MI-inconsistent behaviours per therapist turn; LOWER = better, so
    a positive delta means K=0 is WORSE), PCT (change-talk proportion; higher = better). ``p_holm``
    = Holm across the three cooperation strata within (judge, method, metric, target) — the ``All``
    row is a reference outside the family (NaN). ``share_*_ge_4.5`` (Q1Q2 only) = fraction of that
    arm's stratum conversations scoring ≥ 4.5 — the ceiling diagnostic for the Cooperative stratum.
    Sign ``+ => K=0 higher``; paired on ``persona_id``.
    """
    from .stats import holm
    from .data import best_iteration_by_arm
    plabel = _primary_label(scores_by_judge, primary)
    first = scores_by_judge[plabel]
    end = dict(end) if end is not None else endpoints(first)
    matched_end = dict(matched_end) if matched_end is not None else matched_endpoints(end, methods)
    best = dict(best_by_arm) if best_by_arm is not None else best_iteration_by_arm(first)

    parts = []
    for j, sc in scores_by_judge.items():
        sc = sc.copy()
        sc["coop"] = sc["cooperation_level"].map(COOP_LABEL)
        for method in methods:
            if method not in matched_end:
                continue
            targets = [("matched_final", matched_end[method], matched_end[method])]
            if f"{method}_LA0" in best and f"{method}_LA5" in best:
                targets.append(("own_best", int(best[f"{method}_LA0"]), int(best[f"{method}_LA5"])))
            for metric in metrics:
                d = sc[sc["questionnaire"] == metric]
                if d.empty:
                    continue
                for tname, it0, it5 in targets:
                    rows = []
                    for coop in COOP_ORDER + ["All"]:
                        dd = d if coop == "All" else d[d["coop"] == coop]
                        r = _pair(dd, "score", _model(method, 0, it0), _model(method, 5, it5))
                        k0 = dd[dd["model"] == _model(method, 0, it0)]["score"]
                        k5 = dd[dd["model"] == _model(method, 5, it5)]["score"]
                        rows.append({"judge": j, "method": method, "metric": metric, "target": tname,
                                     "iter_K0": it0, "iter_K5": it5, "cooperation": coop, "n": r["n"],
                                     "mean_K0": r["mean_a"], "mean_K5": r["mean_b"], "mean_delta": r["mean_delta"],
                                     "dz": r["dz"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"],
                                     "share_K0_ge_4.5": float((k0 >= 4.5).mean()) if metric == "Q1Q2" and len(k0) else np.nan,
                                     "share_K5_ge_4.5": float((k5 >= 4.5).mean()) if metric == "Q1Q2" and len(k5) else np.nan})
                    t = pd.DataFrame(rows)
                    mask = t["cooperation"] != "All"
                    ph = np.full(len(t), np.nan); ph[mask.to_numpy()] = holm(t.loc[mask, "p"].to_numpy())
                    t["p_holm"] = ph
                    parts.append(t)
    cols = ["judge", "method", "metric", "target", "iter_K0", "iter_K5", "cooperation", "n", "mean_K0",
            "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "share_K0_ge_4.5", "share_K5_ge_4.5"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


def hetero_ceiling(scores_by_judge: Dict[str, pd.DataFrame], *, end: Optional[Dict[str, int]] = None,
                   arms: Optional[Sequence[str]] = None, metric: str = "Q1Q2") -> pd.DataFrame:
    """Ceiling note behind the Cooperative stratum: per grader × arm × {base, endpoint} × stratum
    the ``mean``, ``share_ge_4.5`` and ``n`` of ``metric`` (default Q1Q2). Tidy columns
    ``judge, arm, iteration, cooperation, mean, share_ge_4.5, n``."""
    rows = []
    for j, sc in scores_by_judge.items():
        d = sc[sc["questionnaire"] == metric].copy()
        d["coop"] = d["cooperation_level"].map(COOP_LABEL)
        e = dict(end) if end is not None else endpoints(d, arms)
        for arm in _arms_present(d, arms):
            for it in sorted({0, int(e[arm])}):
                g = d[(d["arm"] == arm) & (d["iteration"] == it)]
                for c in COOP_ORDER:
                    gc = g[g["coop"] == c]["score"]
                    rows.append({"judge": j, "arm": arm, "iteration": it, "cooperation": c,
                                 "mean": float(gc.mean()) if len(gc) else np.nan,
                                 "share_ge_4.5": float((gc >= 4.5).mean()) if len(gc) else np.nan,
                                 "n": int(len(gc))})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# LEDGER — the quotable numbers (mirrors the paper's out/held_out_instruments.json keys)
# ═══════════════════════════════════════════════════════════════════════════════
def _judge_keys(labels: Iterable[str], primary: str) -> Dict[str, str]:
    """Ledger key segment per grader — ``primary`` for the training oracle, ``heldout`` when there
    is exactly one other grader (the paper's convention), else the grader's own label."""
    labels = list(labels)
    others = [l for l in labels if l != primary]
    return {l: ("primary" if l == primary else ("heldout" if len(others) == 1 else l)) for l in labels}


def instruments_numbers(*, wai: pd.DataFrame, wai_k: pd.DataFrame, fig_wai: pd.DataFrame,
                        pct: pd.DataFrame, q2: Dict[str, pd.DataFrame], hetero: pd.DataFrame,
                        scores_by_judge: Optional[Dict[str, pd.DataFrame]] = None,
                        end: Optional[Dict[str, int]] = None,
                        matched_end: Optional[Dict[str, int]] = None,
                        best_by_arm: Optional[Dict[str, int]] = None,
                        parity: Optional[Dict[str, dict]] = None,
                        primary: Optional[str] = None,
                        table_prefix: str = "held_out_instruments") -> Dict[str, dict]:
    """The quotable-numbers ledger — ``{dotted.key: {"value", "source", "note"}}`` — built from
    the frames the other functions return, key-for-key the paper's
    ``analysis/out/held_out_instruments.json`` (endpoints, best iterations, matched endpoints,
    ``wai.endpoint.*``, ``wai.kcontrast.*``, ``wai.fig_gain.*``, ``pct.kcontrast.*``,
    ``q2.item{1,2,3,10}.*``, ``q2.groups.*``, ``q2.kcontrast_summary.*``, ``hetero.*``,
    ``hetero.ceiling.*``, ``wai.subscale_map_check.*``, plus the two ``crosscheck.*`` anchors).
    Values are rounded to 3 decimals (``_fmt3``) like the paper; pass the dict to
    ``exports.save_numbers``. ``source`` strings name the table each number can be re-read from
    (``tables/<table_prefix>_<name>.md``).

    ``scores_by_judge`` (persona-attached ``scores_long`` per grader) enables the ``hetero.ceiling``
    block and the Q1Q2 crosscheck; ``parity`` = ``{judge_label: wai_subscale_parity(...)}``.
    ``end`` / ``matched_end`` / ``best_by_arm`` default from the frames (``best_by_arm`` from
    ``hetero``'s ``own_best`` rows).
    """
    L: Dict[str, dict] = {}

    def put(key, value, *, source="", note=""):
        L[key] = {"value": _clean(value), "source": source, "note": note}

    judges = list(dict.fromkeys(wai["judge"]))
    plabel = _primary_label({j: None for j in judges}, primary)
    JK = _judge_keys(judges, plabel)
    arms = list(dict.fromkeys(wai["arm"]))
    end = dict(end) if end is not None else endpoints(wai, arms)
    matched_end = dict(matched_end) if matched_end is not None else matched_endpoints(end)
    if best_by_arm is None and not hetero.empty:
        ob = hetero[hetero["target"] == "own_best"]
        best_by_arm = {}
        for _, r in ob.drop_duplicates(["method"]).iterrows():
            best_by_arm[f"{r.method}_LA0"] = int(r.iter_K0); best_by_arm[f"{r.method}_LA5"] = int(r.iter_K5)
    put("endpoints", end, source="data: max scored iteration per arm (both graders agree)")
    if best_by_arm:
        put("best_iteration_by_arm", best_by_arm, source="eda_analysis.data.best_iteration_by_arm on primary Q1Q2")
    put("matched_endpoint", matched_end, source="min(END[LA0], END[LA5]) per method")
    for j, chk in (parity or {}).items():
        put(f"wai.subscale_map_check.{JK.get(j, j)}", chk,
            source="in-module check: WAI-SR standard map (Task 1,2,10,12 / Goal 4,6,8,11 / Bond 3,5,7,9) "
                   "vs eval_scores WAI_{Task,Goal,Bond}_Mean (code/questionnaires.py map)")

    # ── WAI
    for j in judges:
        for arm in arms:
            sel = wai[(wai["judge"] == j) & (wai["arm"] == arm) & (wai["iteration"] == end[arm])]
            if sel.empty:
                continue
            r = sel.iloc[0]
            put(f"wai.endpoint.{JK[j]}.{arm}", {c: _fmt3(r[c]) for c in
                ["Task", "Goal", "Bond", "bond_excess", "WAI_total", "Task_gain", "Goal_gain", "Bond_gain",
                 "bond_excess_gain", "WAI_total_gain"]} | {"iteration": int(r["iteration"]), "n": int(r["n"])},
                source=f"tables/{table_prefix}_wai.md row judge={j} arm={arm} iteration={end[arm]}")
        for method in list(dict.fromkeys(wai_k["method"])):
            for val in list(dict.fromkeys(wai_k["measure"])):
                t = wai_k[(wai_k["judge"] == j) & (wai_k["method"] == method) & (wai_k["measure"] == val)]
                if t.empty or method not in matched_end:
                    continue
                me = matched_end[method]
                sel = t[t["iteration"] == me]
                if not sel.empty:
                    r = sel.iloc[0]
                    put(f"wai.kcontrast.{JK[j]}.{method}.{val}.iter{me}",
                        {c: _fmt3(r[c]) for c in ["mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]}
                        | {"n": int(r["n"])},
                        source=f"tables/{table_prefix}_wai_kcontrast.md row judge={j} method={method} "
                               f"measure={val} iteration={me}")
                sig = t[(t["iteration"] > 0) & (t["p_holm"] < 0.05)]
                put(f"wai.kcontrast.{JK[j]}.{method}.{val}.n_iters_holm_sig",
                    {"n_sig": int(len(sig)), "n_iters": int((t["iteration"] > 0).sum()),
                     "iters_sig": [int(i) for i in sig["iteration"]],
                     "sign_of_sig": [("K0>K5" if d > 0 else "K5>K0") for d in sig["mean_delta"]]},
                    source=f"tables/{table_prefix}_wai_kcontrast.md (judge={j}, method={method}, measure={val})")
    for _, r in fig_wai.iterrows():
        put(f"wai.fig_gain.{r.judge}.{r.arm}.iter{int(r.iteration)}.{r.subscale}",
            {"gain": _fmt3(r.gain), "ci_lo": _fmt3(r.ci_lo), "ci_hi": _fmt3(r.ci_hi), "n": int(r.n)},
            source=f"tables/{table_prefix}_fig_wai_data.md / figures/{table_prefix}_fig_wai.png")

    # ── PCT
    for j in judges:
        for method in list(dict.fromkeys(pct["method"])):
            for m in list(dict.fromkeys(pct["metric"])):
                t = pct[(pct["judge"] == j) & (pct["method"] == method) & (pct["metric"] == m)]
                if t.empty or method not in matched_end:
                    continue
                me = matched_end[method]
                sel = t[t["iteration"] == me]
                if not sel.empty:
                    r = sel.iloc[0]
                    put(f"pct.kcontrast.{JK[j]}.{method}.{m}.iter{me}",
                        {c: _fmt3(r[c]) for c in ["mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm"]}
                        | {"n": int(r["n"])},
                        source=f"tables/{table_prefix}_pct.md row judge={j} method={method} metric={m} iteration={me}")
                sig = t[(t["iteration"] > 0) & (t["p_holm"] < 0.05)]
                put(f"pct.kcontrast.{JK[j]}.{method}.{m}.holm_sig_iters",
                    {"iters_sig": [int(i) for i in sig["iteration"]],
                     "sign_of_sig": [("K0>K5" if d > 0 else "K5>K0") for d in sig["mean_delta"]],
                     "n_iters": int((t["iteration"] > 0).sum())},
                    source=f"tables/{table_prefix}_pct.md (judge={j}, method={method}, metric={m})")
                tt = t[t["iteration"] > 0]
                if not tt.empty:
                    rr = tt.loc[tt["dz"].abs().idxmax()]
                    put(f"pct.kcontrast.{JK[j]}.{method}.{m}.max_abs_dz",
                        {"iteration": int(rr["iteration"]), "mean_delta": _fmt3(rr["mean_delta"]),
                         "dz": _fmt3(rr["dz"]), "p": _fmt3(rr["p"]), "p_holm": _fmt3(rr["p_holm"])},
                        source=f"tables/{table_prefix}_pct.md (judge={j}, method={method}, metric={m})")

    # ── crosscheck vs the tracked EDA (pre-reorg results/L5/tables/7_stats/gpt-4o-mini/k_paired_by_method.md,
    #    commit abe5cb3): PCT (=ChangeProp) PTO iter 6 +0.006/dz .037; GRPO iter 4 -0.062/-0.321;
    #    Q1Q2 PTO iter 6 +0.257/dz .417.
    cc = {}
    p6 = pct[(pct.judge == plabel) & (pct.method == "PTO") & (pct.metric == "PCT_ChangeProp") & (pct.iteration == 6)]
    g4 = pct[(pct.judge == plabel) & (pct.method == "GRPO") & (pct.metric == "PCT_ChangeProp") & (pct.iteration == 4)]
    if not p6.empty:
        cc["pct_pto_iter6"] = {"mine": {"mean_delta": _fmt3(p6.iloc[0].mean_delta), "dz": _fmt3(p6.iloc[0].dz)},
                               "tracked": {"mean_delta": 0.006, "dz": 0.037}}
    if not g4.empty:
        cc["pct_grpo_iter4"] = {"mine": {"mean_delta": _fmt3(g4.iloc[0].mean_delta), "dz": _fmt3(g4.iloc[0].dz)},
                                "tracked": {"mean_delta": -0.062, "dz": -0.321}}
    if scores_by_judge is not None and plabel in scores_by_judge:
        q = scores_by_judge[plabel]
        r = _pair(q[q["questionnaire"] == "Q1Q2"], "score", _model("PTO", 0, 6), _model("PTO", 5, 6))
        cc["q1q2_pto_iter6"] = {"mine": {"mean_delta": _fmt3(r["mean_delta"]), "dz": _fmt3(r["dz"])},
                                "tracked": {"mean_delta": 0.257, "dz": 0.417}}
    if cc:
        put("crosscheck.tracked_k_paired_by_method", cc,
            source="pre-reorg Exp3_PTO_GRPO/eda/results/L5/tables/7_stats/gpt-4o-mini/k_paired_by_method.md "
                   "(commit abe5cb3) — now results/lookahead/reward/tables/k_paired_by_method.md")

    # ── Q2
    Q2L, Q2K = q2["q2items_long"], q2["q2items_kcontrast"]
    for j in judges:
        for item in Q2_SELF_DISCLOSURE:
            for arm in arms:
                sel = Q2L[(Q2L.judge == j) & (Q2L.arm == arm) & (Q2L.item == item)]
                if sel.empty:
                    continue
                r = sel.iloc[0]
                put(f"q2.item{item}.gain.{JK[j]}.{arm}", {"base": _fmt3(r.base), "target": _fmt3(r.target),
                    "gain": _fmt3(r.gain), "ci_lo": _fmt3(r.gain_ci_lo), "ci_hi": _fmt3(r.gain_ci_hi),
                    "dz": _fmt3(r.gain_dz), "target_iter": int(r.target_iter), "n": int(r.n)},
                    source=f"tables/{table_prefix}_q2items_long.md row judge={j} arm={arm} item={item}")
            for method in list(dict.fromkeys(Q2K["method"])):
                sel = Q2K[(Q2K.judge == j) & (Q2K.method == method) & (Q2K.item == item)]
                if sel.empty:
                    continue
                r = sel.iloc[0]
                put(f"q2.item{item}.kcontrast.{JK[j]}.{method}.iter{int(r.iteration)}",
                    {"mean_K0": _fmt3(r.mean_K0), "mean_K5": _fmt3(r.mean_K5), "k_delta": _fmt3(r.k_delta),
                     "dz": _fmt3(r.k_dz), "ci_lo": _fmt3(r.k_ci_lo), "ci_hi": _fmt3(r.k_ci_hi), "p": _fmt3(r.k_p),
                     "p_holm": _fmt3(r.k_p_holm), "n": int(r.n)},
                    source=f"tables/{table_prefix}_q2items.md row judge={j} item={item} ({method} columns)")
        for arm in arms:
            t = Q2L[(Q2L.judge == j) & (Q2L.arm == arm)]
            if t.empty:
                continue
            sd = t[t["item"].isin(Q2_SELF_DISCLOSURE)]["gain"].mean()
            rest = t[~t["item"].isin(Q2_SELF_DISCLOSURE)]["gain"].mean()
            emo = t[t["item"].isin(Q2_EMOTIONAL)]["gain"].mean()
            rank = t.sort_values("gain", ascending=False)["item"].tolist()
            put(f"q2.groups.{JK[j]}.{arm}", {"self_disclosure_1_2_3_10_mean_gain": _fmt3(sd),
                "emotional_3_10_mean_gain": _fmt3(emo), "other_13_items_mean_gain": _fmt3(rest),
                "all_17_mean_gain": _fmt3(t["gain"].mean()),
                "rank_of_item3": int(rank.index(3) + 1), "rank_of_item10": int(rank.index(10) + 1),
                "top3_items": [int(i) for i in rank[:3]], "bottom3_items": [int(i) for i in rank[-3:]]},
                source=f"tables/{table_prefix}_q2items.md (judge={j}, column {arm}_gain@{end[arm]})")
        for method in list(dict.fromkeys(Q2K["method"])):
            t = Q2K[(Q2K.judge == j) & (Q2K.method == method)]
            if t.empty:
                continue
            sig = t[t["k_p_holm"] < 0.05]
            put(f"q2.kcontrast_summary.{JK[j]}.{method}", {
                "iteration": int(t["iteration"].iloc[0]), "n_items_holm_sig": int(len(sig)),
                "items_sig": [int(i) for i in sig["item"]],
                "sign_of_sig": [("K0>K5" if d > 0 else "K5>K0") for d in sig["k_delta"]],
                "mean_k_delta_all17": _fmt3(t["k_delta"].mean()),
                "mean_k_delta_selfdisc": _fmt3(t[t["item"].isin(Q2_SELF_DISCLOSURE)]["k_delta"].mean()),
                "mean_k_delta_emotional_3_10": _fmt3(t[t["item"].isin(Q2_EMOTIONAL)]["k_delta"].mean()),
                "mean_k_delta_other13": _fmt3(t[~t["item"].isin(Q2_SELF_DISCLOSURE)]["k_delta"].mean())},
                source=f"tables/{table_prefix}_q2items.md (judge={j}, {method} columns)")
    # crosscheck vs tracked q2_item_deltas (unpaired base/target means -> identical gains when both cells are complete)
    chk = Q2L[(Q2L.judge == plabel) & (Q2L.arm == "PTO_LA5") & (Q2L.item == 3)]
    if not chk.empty:
        c = chk.iloc[0]
        put("crosscheck.tracked_q2_item_deltas.PTO_LA5_item3_final_primary",
            {"mine": {"base": _fmt3(c.base), "gain": _fmt3(c.gain)}, "tracked": {"base": 2.135, "delta": 1.135}},
            source="pre-reorg Exp3_PTO_GRPO/eda/results/L5/tables/2_questionnaires/gpt-4o-mini/q2_item_deltas.md "
                   "(commit abe5cb3) — now results/arms/questionnaires/tables/gpt-4o-mini/q2_item_deltas.md")

    # ── heterogeneity
    for _, r in hetero.iterrows():
        put(f"hetero.{JK.get(r.judge, r.judge)}.{r.method}.{r.metric}.{r.target}.{str(r.cooperation).replace(' ', '_')}",
            {"iter_K0": int(r.iter_K0), "iter_K5": int(r.iter_K5), "n": int(r.n),
             "mean_K0": _fmt3(r.mean_K0), "mean_K5": _fmt3(r.mean_K5), "mean_delta": _fmt3(r.mean_delta),
             "dz": _fmt3(r.dz), "ci_lo": _fmt3(r.ci_lo), "ci_hi": _fmt3(r.ci_hi), "p": _fmt3(r.p),
             "p_holm": _fmt3(r.p_holm), "share_K0_ge_4.5": _fmt3(r["share_K0_ge_4.5"]),
             "share_K5_ge_4.5": _fmt3(r["share_K5_ge_4.5"])},
            source=f"tables/{table_prefix}_hetero.md row judge={r.judge} method={r.method} metric={r.metric} "
                   f"target={r.target} cooperation={r.cooperation}")
    if scores_by_judge is not None:
        ceil = hetero_ceiling(scores_by_judge, end=end, arms=arms)
        for j in list(dict.fromkeys(ceil["judge"])):
            out = {}
            cj = ceil[ceil["judge"] == j]
            for (arm, it), g in cj.groupby(["arm", "iteration"], sort=False):
                out[f"{arm}.iter{int(it)}"] = {r.cooperation: {"mean": _fmt3(r["mean"]),
                                                               "share_ge_4.5": _fmt3(r["share_ge_4.5"]),
                                                               "n": int(r.n)} for _, r in g.iterrows()}
            put(f"hetero.ceiling.{JK.get(j, j)}", out,
                source="from scores_long (Q1Q2 by cooperation stratum); "
                       f"see tables/{table_prefix}_hetero.md share_K0_ge_4.5 columns")
    return L
