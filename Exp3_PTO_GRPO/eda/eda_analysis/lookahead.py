"""lookahead.py — RQ-i, the look-ahead contrast: K=0 vs K=5 within one optimizer, both graders.

The four-arm, persona-paired K contrast (``<METHOD>_LA0`` minus ``<METHOD>_LA5``) at every matched
iteration, on every rubric, under the training oracle (gpt-4o-mini) and the held-out judge
(Claude Haiku 4.5) side by side — plus the oracle-coded behaviour channels (MICI / MITI, per turn
and per session), the deterministic text channels, the K × method difference-in-differences, the
method gap at each K, and the endpoint contrasts the write-up quotes.

**Provenance.** Promoted on 2026-08-18 from the look-ahead paper's generators
``papers/2026_lookahead_pto_grpo/analysis/k_contrast_headline.py`` (rubric + channel contrasts,
levels, Table 1, summaries) and ``…/cross_k_multijudge.py`` (§3 DiD + method gap, §4 endpoints).
The paper's frozen ``tables/*.csv`` + ``analysis/out/*.json`` are the fixture these functions
reproduce cell-for-cell (means / dz / p exact; bootstrap CIs to the seed — see the note below).
The cross-K *transfer* half of ``cross_k_multijudge.py`` (pairs, sign ladder, gain retention) is
:mod:`eda_analysis.transfer`.

Conventions (restate them in every caption):

* **Sign: ``+ => K=0 higher`` (K=0 minus K=5)** — mirrors :func:`eda_analysis.stats.paired_k_comparison`.
  MICI and every ``MICI_*`` channel are lower-is-better, so there ``+`` means K=0 is WORSE.
* **Pairing unit: ``persona_id``** (the 96 patient personas recur in every model state; the trainer
  reshuffles them each iteration, so ``file_index`` is not a pairing key). Frames must carry
  ``persona_id`` — :func:`~eda_analysis.config.scores_by_judge` attaches it.
* **Iteration 0 = two INDEPENDENT base draws** (the K=0 arm's base vs the K=5 arm's base): a free
  noise-floor row, computed with the same machinery. Never drop it silently — it is the reference
  the trained rows are read against.
* **Support:** an arm has no state past its last iteration on disk, and how far it is SCORED can
  be grader-dependent, so the DiD is estimable only over the overlap of the arms in the frame.
  Matched iterations are derived from the data (the
  intersection of both arms' iterations), never hard-coded; neither is any endpoint a caption
  quotes (:func:`eda_analysis.constants.support_note` reads it off the frame in hand).
* **Holm family = ITERATIONS 0..N within (judge, method, metric)** (``holm_family="iterations"``,
  the paper's default). This is a DIFFERENT family from the tracked ``k_paired_by_method`` table
  (Holm across rubrics within one iteration; ``holm_family="rubrics"``), so ``p`` agrees with that
  table cell-for-cell while ``p_holm`` need not.
* **The two graders' raw scores are never averaged.** Every frame keeps a ``judge`` column (or
  ``primary_*`` / ``judge_*`` column pairs) — combine contrasts, never levels.
* **Bootstrap CIs** come from :func:`eda_analysis.stats.paired_arrays`, seeded with
  :data:`eda_analysis.constants.BOOT_SEED` (12345). The paper's generators seeded with 0, so
  ``ci_lo``/``ci_hi`` may differ from the frozen fixture in the third decimal; ``n``, means, ``dz``,
  ``p`` and ``p_holm`` are exact.

Contract: functions take frames and return tidy ``pd.DataFrame``s (or a dict of them keyed by the
paper's table-name suffix); no disk I/O — the notebook owns ``exports.save_*``.
:func:`lookahead_numbers` returns the quotable ledger for ``exports.save_numbers``.
"""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import LOWER_IS_BETTER as _LIB
from .constants import support_note
from .stats import holm, paired_arrays
from .constants import k_of as _k_of_canonical, method_of as _method_of_canonical  # noqa: E402
from .ledger import json_scalar, ledger_entry, round3  # noqa: E402,F401

__all__ = [
    "RUBRICS", "FIVE_POINT", "RATE_METRICS", "TEXT_CHANNELS", "FIG_CHANNELS", "METHODS",
    "LOWER_BETTER", "TEXT_JUDGE_LABEL", "SIGN_NOTE", "CENSOR_NOTE", "HOLM_NOTE",
    "model_name", "k_of", "method_of", "stars", "favours", "wide_by_persona", "holm_within",
    "paired_k_frames", "k_levels", "k_table1", "k_summary", "channel_k_frames",
    "did_by_iter", "method_gap_by_iter", "endpoint_contrasts", "best_iteration",
    "lookahead_numbers",
]

# ── vocabulary ────────────────────────────────────────────────────────────────

#: The 9 eval rubrics in reporting order (the 6 questionnaires + PCT + MICI, Q1Q2 first).
RUBRICS = ["Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI"]
#: Rubrics on the 1-5 / 1-7 point scale that the ±0.10 oracle-repeatability band refers to.
FIVE_POINT = ["Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI"]
#: 0-1 rate metrics (deltas ~3x smaller than the point-scale rubrics; no ±0.10 band).
RATE_METRICS = ["PCT", "MICI"]
#: Deterministic text channels (judge-invariant). ``loop`` is a 0/1 flag per conversation.
TEXT_CHANNELS = ["conv_len", "n_th_turns", "mean_turn_len", "q_per_turn", "loop"]
#: The three oracle-coded channels the channels figure shows.
FIG_CHANNELS = ["MICI_OverPraise_rate", "MICI_AdviseNoPermission_rate", "B6_AF_per_turn"]
METHODS = ("PTO", "GRPO")
#: Lower-is-better metrics/channels: the package registry plus MICI itself.
LOWER_BETTER = set(_LIB) | {"MICI"}
#: ``judge`` value carried by the text-channel contrasts (they are computed once, off transcripts).
TEXT_JUDGE_LABEL = "text (judge-invariant)"

SIGN_NOTE = "Sign: + = K=0 higher (K=0 minus K=5). Paired on persona_id (96 personas)."
# NOT an assertion that any arm IS censored — a legend for how to read support. It is the fallback
# used only when a frame is not available to derive from; `constants.support_note` is the real
# mechanism and returns "" when every arm reaches the same iteration. This string asserted
# "GRPO_LA5 is right-censored" until 2026-08-25, which shipped into ~20 captions after that arm
# finished at iteration 10 — a hardcoded claim outliving the condition it described.
CENSOR_NOTE = ("Each arm's rows run to its own last scored iteration, which can differ by grader "
               "- read each arm's endpoint off the table's own iteration column "
               "(constants.support_note derives the sentence when an arm is genuinely short).")
HOLM_NOTE = ("p_holm = Holm across iterations 0..N within each (judge, method, metric); "
             "iteration 0 = two independent base draws (noise floor).")


def model_name(method: str, K: int, it: int) -> str:
    """``("PTO", 5, 0) -> "PTOExp3_LA5_Base"``, ``("GRPO", 0, 3) -> "GRPOExp3_LA0_I3"``."""
    return f"{method}Exp3_LA{K}_{'Base' if it == 0 else f'I{it}'}"


def k_of(arm: str) -> int:
    """``"PTO_LA5" -> 5``. Re-export of :func:`eda_analysis.constants.k_of` (THE canonical parse).

    Kept importable here because ``transfer``/``crossgen`` already import it from this module.
    """
    return _k_of_canonical(arm)


def method_of(arm: str) -> str:
    """``"GRPO_LA0" -> "GRPO"``. Re-export of :func:`eda_analysis.constants.method_of`."""
    return _method_of_canonical(arm)


def stars(p) -> str:
    """Holm significance marker: ``***`` <.001, ``**`` <.01, ``*`` <.05, else ``""`` (NaN → ``""``)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def favours(metric: str, delta, plus_label: str, minus_label: str) -> str:
    """Readable direction label for a signed contrast; flips for lower-is-better metrics (MICI).

    ``favours("Q1Q2", +0.2, "K0", "K5") -> "K0"``; ``favours("MICI", +0.02, "K0", "K5") -> "K5"``.
    Empty for NaN / exactly zero.
    """
    if delta is None or (isinstance(delta, float) and np.isnan(delta)) or delta == 0:
        return ""
    hi = plus_label if delta > 0 else minus_label
    if metric in LOWER_BETTER:
        hi = minus_label if delta > 0 else plus_label
    return hi


# ── frame helpers ─────────────────────────────────────────────────────────────

def wide_by_persona(scores_long: pd.DataFrame, metric: str) -> pd.DataFrame:
    """``persona_id x model`` matrix of one metric (mean over duplicates; NaN where unscored).

    The pairing unit is the persona, so every contrast below is a column difference of this
    pivot. ``scores_long`` is the ``load_scores_long`` shape (``questionnaire``/``score``/
    ``model``/``persona_id``); a channel frame from :func:`~eda_analysis.behavior.channel_scores_long`
    or a melted text-metric frame qualifies as long as it carries those four columns.
    """
    d = scores_long[scores_long["questionnaire"] == metric]
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index="persona_id", columns="model", values="score", aggfunc="mean")


def holm_within(df: pd.DataFrame, by: Sequence[str], pcol: str, out: str) -> pd.DataFrame:
    """Holm-adjust ``pcol`` within each ``by`` group into a new column ``out`` (NaNs preserved)."""
    df = df.copy()
    df[out] = np.nan
    if df.empty:
        return df
    for _, g in df.groupby(list(by), sort=False):
        df.loc[g.index, out] = holm(g[pcol].to_numpy())
    return df


def _judge_items(scores_by_judge: Mapping[str, pd.DataFrame]):
    """(label, frame) pairs in the mapping's order (primary first by construction)."""
    return list(scores_by_judge.items())


def _present_metrics(long: pd.DataFrame, metrics: Optional[Sequence[str]]) -> list:
    have = set(long["questionnaire"].unique()) if not long.empty else set()
    if metrics is None:
        return [m for m in RUBRICS if m in have]
    return [m for m in metrics if m in have]


def _matched_iterations(long: pd.DataFrame, method: str) -> list:
    """Iterations present in BOTH ``<method>_LA0`` and ``<method>_LA5`` (censoring falls out)."""
    a = {int(i) for i in long.loc[long["arm"] == f"{method}_LA0", "iteration"]}
    b = {int(i) for i in long.loc[long["arm"] == f"{method}_LA5", "iteration"]}
    return sorted(a & b)


def _last_iter(long: pd.DataFrame, arm: str) -> int:
    d = long.loc[long["arm"] == arm, "iteration"]
    return int(d.max()) if len(d) else -1


# ── 1. the paired K contrast (rubrics or channels) ────────────────────────────

def _k_contrast_one_judge(long: pd.DataFrame, metrics: Sequence[str], *, judge: str,
                          methods: Sequence[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for m in metrics:
            W = wide_by_persona(long, m)
            if W.empty:
                continue
            for it in _matched_iterations(long, method):
                a, b = model_name(method, 0, it), model_name(method, 5, it)
                if a not in W.columns or b not in W.columns:
                    continue
                r = paired_arrays(W[a].to_numpy(), W[b].to_numpy())
                rows.append({"judge": judge, "method": method, "metric": m, "iteration": it,
                             "mean_K0": float(W[a].mean()), "mean_K5": float(W[b].mean()),
                             "se_K0": float(W[a].std(ddof=1) / np.sqrt(W[a].notna().sum())),
                             "se_K5": float(W[b].std(ddof=1) / np.sqrt(W[b].notna().sum())),
                             **r})
    return pd.DataFrame(rows)


_K_COLS = ["judge", "method", "metric", "iteration", "n", "mean_K0", "mean_K5", "se_K0", "se_K5",
           "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "sig", "lower_better"]


def _finish_k_frame(df: pd.DataFrame, holm_family: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_K_COLS)
    if holm_family == "iterations":
        df = holm_within(df, ["judge", "method", "metric"], "p", "p_holm")
    elif holm_family == "rubrics":
        df = holm_within(df, ["judge", "method", "iteration"], "p", "p_holm")
    else:
        raise ValueError(f"holm_family must be 'iterations' or 'rubrics', got {holm_family!r}")
    df["sig"] = df["p_holm"].map(stars)
    df["lower_better"] = df["metric"].isin(LOWER_BETTER)
    return df[_K_COLS].reset_index(drop=True)


def paired_k_frames(scores_by_judge: Mapping[str, pd.DataFrame], *,
                    methods: Sequence[str] = METHODS, metrics: Optional[Sequence[str]] = None,
                    holm_family: str = "iterations") -> pd.DataFrame:
    """The paired K contrast, long form: one row per (judge, method, metric, iteration).

    ``scores_by_judge`` — ``{judge_label: scores_long}`` (:func:`~eda_analysis.config.scores_by_judge`
    or any mapping of ``load_scores_long``-shaped frames with ``persona_id``); the ``judge`` column
    takes the mapping's keys, in its order (primary first). ``metrics`` default = the :data:`RUBRICS`
    present in each frame (channel frames: pass the channel list explicitly).

    Per row: ``n`` paired personas, the two arm levels ``mean_K0``/``mean_K5`` with per-arm SEs,
    ``mean_delta`` (K=0 − K=5), Cohen's ``dz`` (mean/SD of the persona deltas), the bootstrap 95 %
    CI, Wilcoxon ``p``, ``p_holm`` (family per ``holm_family``: ``"iterations"`` = across
    iterations 0..N within (judge, method, metric) — the paper's convention; ``"rubrics"`` = across
    metrics within (judge, method, iteration) — the tracked ``k_paired_by_method`` convention),
    ``sig`` (Holm stars) and ``lower_better``. Iteration 0 = the two arms' independent base draws.

    Reproduces ``k_contrast_headline_{pto,grpo}_{primary,heldout}.csv`` (drop the ``judge`` /
    ``method`` columns) and ``k_contrast_headline_all_long.csv``.
    """
    parts = [_k_contrast_one_judge(long, _present_metrics(long, metrics), judge=lab, methods=methods)
             for lab, long in _judge_items(scores_by_judge)]
    parts = [p for p in parts if not p.empty]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _finish_k_frame(df, holm_family)


# ── 2. levels ─────────────────────────────────────────────────────────────────

def k_levels(scores_by_judge: Mapping[str, pd.DataFrame], *, metrics: Optional[Sequence[str]] = None,
             wide_metric: str = "Q1Q2", arms: Sequence[str] = ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"),
             iterations: Optional[Sequence[int]] = None) -> Dict[str, pd.DataFrame]:
    """Arm × rubric × iteration levels under each grader — NOT paired (read dz/p off the contrast).

    Returns ``{"levels": wide, "levels_long": long}``. ``levels_long`` has one row per (judge, arm,
    metric, iteration) with ``n``, ``mean``, ``sd`` (ddof=1), ``se``. ``levels`` is the compact
    ``wide_metric`` layout — rows = iterations (default 0..max present), one column per
    ``"<arm> · <judge>"`` (judges outer, arms inner) holding the arm mean; NaN where the arm has no
    such iteration (GRPO_LA5 after 5). Reproduces ``k_contrast_headline_levels{,_long}.csv``.
    """
    parts = []
    for lab, long in _judge_items(scores_by_judge):
        mets = _present_metrics(long, metrics)
        d = long[long["questionnaire"].isin(mets)]
        if d.empty:
            continue
        g = (d.groupby(["arm", "questionnaire", "iteration"])["score"]
             .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="count").reset_index())
        g["se"] = g["sd"] / np.sqrt(g["n"])
        g["judge"] = lab
        g = g.rename(columns={"questionnaire": "metric"})
        parts.append(g[["judge", "arm", "metric", "iteration", "n", "mean", "sd", "se"]])
    cols = ["judge", "arm", "metric", "iteration", "n", "mean", "sd", "se"]
    LV = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    lvq = LV[LV["metric"] == wide_metric]
    if iterations is None:
        iterations = list(range(0, int(lvq["iteration"].max()) + 1)) if len(lvq) else []
    wide = pd.DataFrame({"iteration": list(iterations)})
    for lab, _ in _judge_items(scores_by_judge):
        for arm in arms:
            d = lvq[(lvq["judge"] == lab) & (lvq["arm"] == arm)].set_index("iteration")
            wide[f"{arm} · {lab}"] = [d["mean"].get(i, np.nan) for i in wide["iteration"]]
    return {"levels": wide, "levels_long": LV}


# ── 3. Table 1 + summary ──────────────────────────────────────────────────────

def k_table1(frames: pd.DataFrame, metric: str = "Q1Q2", *, methods: Sequence[str] = METHODS,
             judges: Optional[Sequence[str]] = None,
             iterations: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """The compact paper Table 1 for one metric: rows = iteration, columns = ``"<method> · <judge>"``.

    Cell = ``"{mean_delta:+.3f} ({dz:+.2f}){sig}"`` (Holm stars); ``"—"`` where the method has no
    matched K=5 model state (GRPO after 5). ``iterations`` default 0..max in ``frames``; ``judges``
    default = the frame's ``judge`` order of first appearance. Reproduces
    ``k_contrast_headline_table1{,_Q1,_Q2,_MICI,_PCT}.csv``.
    """
    d = frames[frames["metric"] == metric]
    if judges is None:
        judges = list(dict.fromkeys(frames["judge"].tolist()))
    if iterations is None:
        iterations = list(range(0, int(frames["iteration"].max()) + 1)) if len(frames) else []
    out = pd.DataFrame({"iteration": list(iterations)})
    for method in methods:
        for js in judges:
            cell = {}
            for _, r in d[(d["method"] == method) & (d["judge"] == js)].iterrows():
                cell[int(r["iteration"])] = f"{r['mean_delta']:+.3f} ({r['dz']:+.2f}){r['sig']}"
            out[f"{method} · {js}"] = [cell.get(i, "—") for i in out["iteration"]]
    return out


def k_summary(frames: pd.DataFrame, *, group_cols: Sequence[str] = ("judge", "method", "metric"),
              alpha: float = 0.05) -> pd.DataFrame:
    """Per (judge, method, metric): how many iterations reach Holm ``p < alpha`` in each direction.

    ``n_sig_K0_higher`` / ``n_sig_K5_higher`` count significant iterations with delta > 0 / < 0;
    the ``*_better`` twins flip the sign for lower-better metrics (MICI). ``mean_delta_iters1toN``
    / ``mean_dz_iters1toN`` average the per-iteration paired deltas over TRAINED iterations only;
    ``base_delta`` / ``base_dz`` are the iteration-0 base-vs-base draw; ``max_abs_dz`` (+ its
    iteration and delta) is the largest |dz| over all iterations incl. 0. Reproduces
    ``k_contrast_headline_summary.csv`` (rubrics) and ``…_channels_summary.csv`` (channels).
    """
    rows = []
    group_cols = list(group_cols)
    for key, g in frames.groupby(group_cols, sort=True):
        g = g.sort_values("iteration")
        trained = g[g["iteration"] >= 1]
        sig = g[g["p_holm"] < alpha]
        lb = bool(g["lower_better"].iloc[0])
        pos, neg = sig[sig["mean_delta"] > 0], sig[sig["mean_delta"] < 0]
        k0_better, k5_better = (neg, pos) if lb else (pos, neg)
        imax = g.loc[g["dz"].abs().idxmax()] if g["dz"].notna().any() else None
        base = g[g["iteration"] == 0]
        rows.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                     "n_iters": int(len(g)),
                     "n_sig_K0_higher": int(len(pos)), "n_sig_K5_higher": int(len(neg)),
                     "n_sig_K0_better": int(len(k0_better)), "n_sig_K5_better": int(len(k5_better)),
                     "iters_sig_K0_higher": ",".join(str(int(i)) for i in pos["iteration"]),
                     "iters_sig_K5_higher": ",".join(str(int(i)) for i in neg["iteration"]),
                     "mean_delta_iters1toN": float(trained["mean_delta"].mean()) if len(trained) else np.nan,
                     "mean_dz_iters1toN": float(trained["dz"].mean()) if len(trained) else np.nan,
                     "base_delta": float(base["mean_delta"].iloc[0]) if len(base) else np.nan,
                     "base_dz": float(base["dz"].iloc[0]) if len(base) else np.nan,
                     "max_abs_dz": float(abs(imax["dz"])) if imax is not None else np.nan,
                     "max_abs_dz_iter": int(imax["iteration"]) if imax is not None else -1,
                     "max_abs_dz_delta": float(imax["mean_delta"]) if imax is not None else np.nan,
                     "lower_better": lb})
    return pd.DataFrame(rows)


# ── 4. behaviour channels ─────────────────────────────────────────────────────

def _text_long(text_metrics: pd.DataFrame, channels: Sequence[str]) -> pd.DataFrame:
    """Melt :func:`~eda_analysis.behavior.text_metrics` (wide, per conversation) to the long shape."""
    id_cols = [c for c in ["arm", "method", "K", "model", "iteration", "is_base", "file_index", "persona_id"]
               if c in text_metrics.columns]
    if "persona_id" not in id_cols:
        raise ValueError("text_metrics must carry persona_id (call behavior.text_metrics(..., attach_persona=True))")
    tm = text_metrics.copy()
    use = [c for c in channels if c in tm.columns]
    for c in use:
        tm[c] = tm[c].astype(float)          # `loop` is a bool flag; the contrast wants a rate
    return (tm.melt(id_vars=id_cols, value_vars=use, var_name="questionnaire", value_name="score")
              .dropna(subset=["score"]))


def channel_k_frames(channels_by_judge: Mapping[str, pd.DataFrame], text_metrics: Optional[pd.DataFrame] = None, *,
                     channels: Optional[Sequence[str]] = None, text_channels: Sequence[str] = TEXT_CHANNELS,
                     methods: Sequence[str] = METHODS, holm_family: str = "iterations") -> Dict[str, pd.DataFrame]:
    """The paired K contrast on the behaviour channels: oracle-coded per judge + text once.

    ``channels_by_judge`` — ``{judge_label: channel_scores_long}`` (one
    :func:`~eda_analysis.behavior.channel_scores_long` per grader, read from that judge's MITI /
    MICI partition; text channels inside it are ignored here). The oracle-coded channels are
    judge-DEPENDENT, so build the mapping by switching the active judge per load (there is no
    ``scores_by_judge`` twin for channels)::

        arms = eda_analysis.cross_k_arms(cfg)
        CH = {}
        for lab, tag in (("gpt-4o-mini", ""), ("claude-haiku-4-5", HELDOUT_TAG)):
            set_active_judge(tag, 0); CH[lab] = behavior.channel_scores_long(arms)
        set_active_judge("", 0)                       # leave the process on the primary grader

    ``channels`` default = every :data:`~eda_analysis.behavior.BEHAVIOR_CHANNELS` minus its
    ``TEXT_CHANNELS``.
    ``text_metrics`` — :func:`~eda_analysis.behavior.text_metrics` (wide, ``attach_persona=True``);
    the deterministic :data:`TEXT_CHANNELS` (incl. ``loop`` as a 0/1 rate) are contrasted once and
    labelled ``judge = "text (judge-invariant)"``.

    Returns ``{"channels": KC, "channels_text": KT, "channels_summary": KCS}`` — the same columns as
    :func:`paired_k_frames`; the summary is :func:`k_summary` over ``KC ∪ KT``. Reproduces
    ``k_contrast_headline_channels_{pto,grpo}_{primary,heldout}.csv``, ``…_channels_text_{pto,grpo}.csv``
    and ``…_channels_summary.csv``. MICI channels (``*_rate``, per-session counts) are lower-better.
    """
    from . import behavior
    if channels is None:
        channels = [c for c in behavior.BEHAVIOR_CHANNELS if c not in behavior.TEXT_CHANNELS]
    KC = paired_k_frames({lab: ch[~ch["questionnaire"].isin(behavior.TEXT_CHANNELS)]
                          for lab, ch in channels_by_judge.items()},
                         methods=methods, metrics=list(channels), holm_family=holm_family)
    if text_metrics is not None and not text_metrics.empty:
        TXT = _text_long(text_metrics, text_channels)
        KT = paired_k_frames({TEXT_JUDGE_LABEL: TXT}, methods=methods, metrics=list(text_channels),
                             holm_family=holm_family)
    else:
        KT = pd.DataFrame(columns=_K_COLS)
    both = pd.concat([KC, KT], ignore_index=True)
    KCS = k_summary(both) if not both.empty else pd.DataFrame()
    return {"channels": KC, "channels_text": KT, "channels_summary": KCS}


# ── 5. K x method: DiD, method gap, endpoints (cross_k_multijudge §3-4) ──────

def _judge_wide(scores_by_judge: Mapping[str, pd.DataFrame], metrics: Sequence[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    return {lab: {m: wide_by_persona(long, m) for m in metrics} for lab, long in _judge_items(scores_by_judge)}


def did_by_iter(scores_by_judge: Mapping[str, pd.DataFrame], *, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """K × method interaction — persona-level difference-in-differences, per grader.

    ``did = (PTO_LA0 − GRPO_LA0) − (PTO_LA5 − GRPO_LA5)`` computed persona by persona at every
    iteration ALL FOUR arms share (derived as ``min`` over the four arms' last iterations, so a
    censored arm caps it and nothing past that point is estimable; iteration
    0 = four independent base draws, the noise floor). **Sign: ``+`` => PTO's lead over GRPO is
    LARGER at K=0 than at K=5** (equivalently, look-ahead helps GRPO more than PTO); on MICI the sign
    reads the other way round (lower is better).

    Columns: ``judge, iteration, metric, n, gap_K0, gap_K5`` (mean method gaps at each K, ``+`` =>
    PTO higher), ``did_mean, did_dz, did_ci_lo, did_ci_hi, p`` (Wilcoxon), ``p_holm`` (Holm across
    iterations within (judge, metric)), ``p_holm_rubrics`` (Holm across the rubrics within (judge,
    iteration)). Sorted judge → :data:`RUBRICS` order → iteration. Reproduces
    ``cross_k_multijudge_did.csv``. Cross-check: held-out Q1Q2 iteration 5 dz ≈ 0.525.
    """
    rows = []
    for lab, long in _judge_items(scores_by_judge):
        mets = _present_metrics(long, metrics)
        n_last = min(_last_iter(long, a) for a in ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"))
        for m in mets:
            w = wide_by_persona(long, m)
            for it in range(0, n_last + 1):
                need = [model_name("PTO", 0, it), model_name("GRPO", 0, it), model_name("PTO", 5, it), model_name("GRPO", 5, it)]
                if any(c not in w.columns for c in need):
                    continue
                g0 = w[need[0]] - w[need[1]]
                g5 = w[need[2]] - w[need[3]]
                r = paired_arrays(g0.to_numpy(), g5.to_numpy())
                rows.append({"judge": lab, "iteration": it, "metric": m,
                             "gap_K0": float(np.nanmean(g0)), "gap_K5": float(np.nanmean(g5)),
                             "did_mean": r["mean_delta"], "did_dz": r["dz"], "did_ci_lo": r["ci_lo"],
                             "did_ci_hi": r["ci_hi"], "p": r["p"], "n": r["n"]})
    cols = ["judge", "iteration", "metric", "n", "gap_K0", "gap_K5", "did_mean", "did_dz", "did_ci_lo",
            "did_ci_hi", "p", "p_holm", "p_holm_rubrics"]
    if not rows:
        return pd.DataFrame(columns=cols)
    did = pd.DataFrame(rows)
    did = holm_within(did, ["judge", "metric"], "p", "p_holm")
    did = holm_within(did, ["judge", "iteration"], "p", "p_holm_rubrics")
    return _sort_metric(did, ["judge", "metric", "iteration"])[cols]


def method_gap_by_iter(scores_by_judge: Mapping[str, pd.DataFrame], *, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The method gap ``PTO_LA{K}_In − GRPO_LA{K}_In`` at each K, every matched iteration, per grader.

    **Sign: ``+`` => PTO higher**; on MICI (lower-is-better) ``+`` favours GRPO — read ``favours``.
    Iteration 0 = two independent base draws. ``p_holm`` = Holm across ITERATIONS within (judge, K,
    metric); ``p_holm_rubrics`` = Holm across the rubrics within (judge, K, iteration) — the tracked
    EDA's ``method_paired_by_K`` convention. Reproduces ``cross_k_multijudge_method_gap.csv``.
    """
    rows = []
    for lab, long in _judge_items(scores_by_judge):
        mets = _present_metrics(long, metrics)
        for m in mets:
            w = wide_by_persona(long, m)
            for K in (0, 5):
                n_last = min(_last_iter(long, f"PTO_LA{K}"), _last_iter(long, f"GRPO_LA{K}"))
                for it in range(0, n_last + 1):
                    a, b = model_name("PTO", K, it), model_name("GRPO", K, it)
                    if a not in w.columns or b not in w.columns:
                        continue
                    r = paired_arrays(w[a].to_numpy(), w[b].to_numpy())
                    rows.append({"judge": lab, "K": K, "iteration": it, "metric": m,
                                 "contrast": f"{a.replace('Exp3', '')} − {b.replace('Exp3', '')}", **r,
                                 "favours": favours(m, r["mean_delta"], "PTO", "GRPO")})
    cols = ["judge", "K", "iteration", "metric", "contrast", "n", "delta", "dz", "ci_lo", "ci_hi", "p",
            "p_holm", "p_holm_rubrics", "favours"]
    if not rows:
        return pd.DataFrame(columns=cols)
    gap = pd.DataFrame(rows).rename(columns={"mean_delta": "delta"})
    gap = holm_within(gap, ["judge", "K", "metric"], "p", "p_holm")
    gap = holm_within(gap, ["judge", "K", "iteration"], "p", "p_holm_rubrics")
    return _sort_metric(gap, ["judge", "K", "metric", "iteration"])[cols]


def _sort_metric(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Sort with ``metric`` in :data:`RUBRICS` order (unknown metrics after, alphabetically)."""
    order = list(RUBRICS) + sorted(set(df["metric"]) - set(RUBRICS))
    df = df.copy()
    df["metric"] = pd.Categorical(df["metric"], order, ordered=True)
    df = df.sort_values(list(keys)).reset_index(drop=True)
    df["metric"] = df["metric"].astype(str)
    return df


def best_iteration(long: pd.DataFrame, arm: str, metric: str = "Q1Q2") -> int:
    """The arm's best TRAINED iteration by mean ``metric`` in this frame (ties → the earliest)."""
    d = long[(long["arm"] == arm) & (long["questionnaire"] == metric) & (long["iteration"] >= 1)]
    if d.empty:
        return -1
    means = d.groupby("iteration")["score"].mean()
    return int(means.idxmax())


def _both_graders(W: Mapping[str, Dict[str, pd.DataFrame]], primary: str, heldout: str,
                  metric: str, a: str, b: str) -> dict:
    """One row with ``primary_*`` and ``judge_*`` columns (names mirror ``reliability.all_pairs_contrasts``)."""
    rec = {"metric": metric, "model_a": a, "model_b": b}
    for lab, pref in ((primary, "primary"), (heldout, "judge")):
        w = W[lab][metric]
        r = (paired_arrays(w[a].to_numpy(), w[b].to_numpy()) if (a in w.columns and b in w.columns)
             else dict(n=0, mean_delta=np.nan, dz=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan))
        rec.update({f"{pref}_n": r["n"], f"{pref}_delta": r["mean_delta"], f"{pref}_dz": r["dz"],
                    f"{pref}_ci_lo": r["ci_lo"], f"{pref}_ci_hi": r["ci_hi"], f"{pref}_p": r["p"]})
    rec["same_sign"] = bool(np.sign(rec["judge_delta"]) == np.sign(rec["primary_delta"]))
    rec["judge_ci_excl0"] = bool(rec["judge_ci_lo"] > 0 or rec["judge_ci_hi"] < 0)
    rec["primary_ci_excl0"] = bool(rec["primary_ci_lo"] > 0 or rec["primary_ci_hi"] < 0)
    return rec


def _primary_heldout(scores_by_judge: Mapping[str, pd.DataFrame], primary, heldout):
    labs = [lab for lab, _ in _judge_items(scores_by_judge)]
    if len(labs) < 2:
        raise ValueError("need the primary AND a held-out judge in scores_by_judge")
    return (primary if primary is not None else labs[0]), (heldout if heldout is not None else labs[1])


def endpoint_contrasts(scores_by_judge: Mapping[str, pd.DataFrame], *, pairs=None,
                       metrics: Optional[Sequence[str]] = None, primary: Optional[str] = None,
                       heldout: Optional[str] = None, best_metric: str = "Q1Q2") -> pd.DataFrame:
    """The endpoint contrasts the write-up quotes, under both graders (``primary_*`` / ``judge_*``).

    ``pairs`` = ``[(label, model_a, model_b), ...]``; ``+`` => ``model_a`` higher (on MICI, lower is
    better, so ``+`` favours B — read ``favours_*``, where A/B are the pair's left/right model).
    Default pairs (the paper's): the K=0 headline ``PTO_LA0_I10 − GRPO_LA0_I10``; the K=5 endpoints
    ``PTO_LA5_I<last> − GRPO_LA5_I<last>`` (each arm's own last iteration, GRPO's earlier because
    it is censored); the K lever at each method's endpoint / matched iteration; GRPO's
    K=5 endpoint vs GRPO_LA0's endpoint and vs GRPO_LA0's BEST iteration by mean ``best_metric``
    under each grader (:func:`best_iteration`; one extra pair when the two graders disagree). The
    default endpoints are read off the data (each arm's last iteration), so they follow the arms.

    ``*_p_holm`` = Holm across the metrics WITHIN a pair (the tracked ``compare_two_models``
    convention). ``primary``/``heldout`` default to the mapping's first / second key. Reproduces
    ``cross_k_multijudge_endpoints.csv``.
    """
    primary, heldout = _primary_heldout(scores_by_judge, primary, heldout)
    P = scores_by_judge[primary]
    mets = _present_metrics(P, metrics)
    W = _judge_wide({primary: P, heldout: scores_by_judge[heldout]}, mets)
    if pairs is None:
        L = {arm: _last_iter(P, arm) for arm in ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5")}
        best = {lab: best_iteration(scores_by_judge[lab], "GRPO_LA0", best_metric) for lab in (primary, heldout)}
        pairs = [
            (f"PTO_LA0_I{L['PTO_LA0']} − GRPO_LA0_I{L['GRPO_LA0']} (K=0 headline)",
             model_name("PTO", 0, L["PTO_LA0"]), model_name("GRPO", 0, L["GRPO_LA0"])),
            (f"PTO_LA5_I{L['PTO_LA5']} − GRPO_LA5_I{L['GRPO_LA5']} (K=5 endpoints)",
             model_name("PTO", 5, L["PTO_LA5"]), model_name("GRPO", 5, L["GRPO_LA5"])),
            (f"PTO_LA5_I{L['PTO_LA5']} − PTO_LA0_I{L['PTO_LA0']} (K lever, PTO endpoint)",
             model_name("PTO", 5, L["PTO_LA5"]), model_name("PTO", 0, L["PTO_LA0"])),
            (f"GRPO_LA5_I{L['GRPO_LA5']} − GRPO_LA0_I{L['GRPO_LA5']} (K lever, GRPO matched iter)",
             model_name("GRPO", 5, L["GRPO_LA5"]), model_name("GRPO", 0, L["GRPO_LA5"])),
            (f"GRPO_LA5_I{L['GRPO_LA5']} − GRPO_LA0_I{L['GRPO_LA0']} (K=5 endpoint vs K=0 endpoint)",
             model_name("GRPO", 5, L["GRPO_LA5"]), model_name("GRPO", 0, L["GRPO_LA0"])),
            (f"GRPO_LA5_I{L['GRPO_LA5']} − GRPO_LA0_I{best[primary]} (K=0 best by primary {best_metric})",
             model_name("GRPO", 5, L["GRPO_LA5"]), model_name("GRPO", 0, best[primary])),
        ]
        if best[heldout] != best[primary]:
            pairs.append((f"GRPO_LA5_I{L['GRPO_LA5']} − GRPO_LA0_I{best[heldout]} (K=0 best by held-out {best_metric})",
                          model_name("GRPO", 5, L["GRPO_LA5"]), model_name("GRPO", 0, best[heldout])))
    rows = []
    for label, a, b in pairs:
        for m in mets:
            rec = _both_graders(W, primary, heldout, m, a, b)
            rec["pair"] = label
            rows.append(rec)
    cols = ["pair", "metric", "primary_n", "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi",
            "primary_p", "primary_p_holm", "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p",
            "judge_p_holm", "same_sign", "judge_ci_excl0", "favours_primary", "favours_judge"]
    if not rows:
        return pd.DataFrame(columns=cols)
    end = pd.DataFrame(rows)
    end = holm_within(end, ["pair"], "primary_p", "primary_p_holm")
    end = holm_within(end, ["pair"], "judge_p", "judge_p_holm")
    end["favours_primary"] = [favours(m, d, "A", "B") for m, d in zip(end["metric"], end["primary_delta"])]
    end["favours_judge"] = [favours(m, d, "A", "B") for m, d in zip(end["metric"], end["judge_delta"])]
    return end[cols].reset_index(drop=True)


# ── 6. the ledger ─────────────────────────────────────────────────────────────

_nan_none = json_scalar       # one definition — see eda_analysis/ledger.py


def _row(r, cols):
    return {c: _nan_none(r[c]) for c in cols}


def lookahead_numbers(frames: pd.DataFrame, *, summary: Optional[pd.DataFrame] = None,
                      levels_long: Optional[pd.DataFrame] = None,
                      channels: Optional[pd.DataFrame] = None, channels_text: Optional[pd.DataFrame] = None,
                      channels_summary: Optional[pd.DataFrame] = None,
                      did: Optional[pd.DataFrame] = None, method_gap: Optional[pd.DataFrame] = None,
                      endpoints: Optional[pd.DataFrame] = None,
                      ledger_metrics: Sequence[str] = ("Q1Q2", "Q1", "Q2", "MICI", "PCT"),
                      fig_channels: Sequence[str] = FIG_CHANNELS, oracle_noise: float = 0.10) -> dict:
    """Every number the write-up may quote, as ``{dotted.key: {"value", "source", "note"}}``.

    Feed it to ``exports.save_numbers``. Key families (identical to the paper's frozen
    ``k_contrast_headline.json`` + the did / method_gap / endpoint families of
    ``cross_k_multijudge.json``, so the fixture can be diffed key-for-key):
    ``k.<method>.<metric>.iter<n>.<judge>``, ``base_vs_base.<method>.<metric>.<judge>``,
    ``summary.<method>.<metric>.<judge>``, ``pto_q1_vs_q2_split.<judge>``,
    ``grpo_q1_vs_q2_split_iter4_5.<judge>``, ``level.<wide_metric>.<arm>.iter<n>.<judge>``,
    ``channel.<method>.<channel>.iter<n>.<judge>``, ``channel_text.<method>.<channel>.iter<n>``,
    ``channel_summary.<method>.<channel>.<judge>``, ``did.<metric>.iter<n>.<judge>``,
    ``method_gap.K<K>.<metric>.iter<n>.<judge>``, ``endpoint.<pair>.<metric>``, ``conventions``.
    Only frames passed are emitted. ``source`` names the producing frame + row, not a file path —
    the notebook decides the table names.
    """
    L: Dict[str, dict] = {}

    def put(key, value, *, source="", note=""):
        L[key] = {"value": value, "source": source, "note": note}

    kcols = ["n", "mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "sig"]
    judges = list(dict.fromkeys(frames["judge"].tolist())) if len(frames) else []
    for m in ledger_metrics:
        for _, r in frames[frames["metric"] == m].iterrows():
            put(f"k.{r['method'].lower()}.{m}.iter{int(r['iteration'])}.{r['judge']}", _row(r, kcols),
                source=f"paired_k_frames row judge={r['judge']} method={r['method']} metric={m} iteration={int(r['iteration'])}")
    for _, r in frames[frames["iteration"] == 0].iterrows():
        put(f"base_vs_base.{r['method'].lower()}.{r['metric']}.{r['judge']}", _row(r, kcols),
            source=f"paired_k_frames row judge={r['judge']} method={r['method']} metric={r['metric']} iteration=0",
            note="two independent base draws of the same 96 personas — the noise floor of the paired contrast")
    if summary is not None:
        scols = ["n_iters", "n_sig_K0_higher", "n_sig_K5_higher", "n_sig_K0_better", "n_sig_K5_better",
                 "iters_sig_K0_higher", "iters_sig_K5_higher", "mean_delta_iters1toN", "mean_dz_iters1toN",
                 "base_delta", "max_abs_dz", "max_abs_dz_iter", "max_abs_dz_delta"]
        for _, r in summary.iterrows():
            put(f"summary.{r['method'].lower()}.{r['metric']}.{r['judge']}", _row(r, scols),
                source=f"k_summary row judge={r['judge']} method={r['method']} metric={r['metric']}")
    # PTO Q2-vs-Q1 split: iterations where Q2 carries the K=0 edge (Q2 delta > Q1 delta and Q2 sig)
    for js in judges:
        d = frames[(frames["judge"] == js) & (frames["method"] == "PTO")]
        q1 = d[d["metric"] == "Q1"].set_index("iteration"); q2 = d[d["metric"] == "Q2"].set_index("iteration")
        if len(q1) and len(q2):
            split = {}
            for it in sorted(set(q1.index) & set(q2.index)):
                split[f"iter{int(it)}"] = {
                    "Q1_delta": _nan_none(q1.loc[it, "mean_delta"]), "Q1_dz": _nan_none(q1.loc[it, "dz"]),
                    "Q1_p_holm": _nan_none(q1.loc[it, "p_holm"]),
                    "Q2_delta": _nan_none(q2.loc[it, "mean_delta"]), "Q2_dz": _nan_none(q2.loc[it, "dz"]),
                    "Q2_p_holm": _nan_none(q2.loc[it, "p_holm"]),
                    "Q2_carries_edge": bool(q2.loc[it, "mean_delta"] > q1.loc[it, "mean_delta"]
                                            and q2.loc[it, "p_holm"] < 0.05)}
            put(f"pto_q1_vs_q2_split.{js}", split, source=f"paired_k_frames (judge={js}, PTO, metrics Q1, Q2)",
                note="Q2_carries_edge = Q2 delta exceeds Q1 delta AND Q2 Holm p<.05 (K=0 higher on Q2)")
        d = frames[(frames["judge"] == js) & (frames["method"] == "GRPO")]
        q1 = d[d["metric"] == "Q1"].set_index("iteration"); q2 = d[d["metric"] == "Q2"].set_index("iteration")
        if len(q1) and len(q2):
            put(f"grpo_q1_vs_q2_split_iter4_5.{js}",
                {f"iter{it}": {"Q1_delta": _nan_none(q1.loc[it, "mean_delta"]), "Q1_dz": _nan_none(q1.loc[it, "dz"]),
                               "Q1_p_holm": _nan_none(q1.loc[it, "p_holm"]),
                               "Q2_delta": _nan_none(q2.loc[it, "mean_delta"]), "Q2_dz": _nan_none(q2.loc[it, "dz"]),
                               "Q2_p_holm": _nan_none(q2.loc[it, "p_holm"])}
                 for it in (4, 5) if it in q1.index and it in q2.index},
                source=f"paired_k_frames (judge={js}, GRPO, metrics Q1, Q2, iterations 4-5)")
    if levels_long is not None and len(levels_long):
        for _, r in levels_long[levels_long["metric"] == "Q1Q2"].iterrows():
            put(f"level.Q1Q2.{r['arm']}.iter{int(r['iteration'])}.{r['judge']}",
                {"mean": _nan_none(r["mean"]), "se": _nan_none(r["se"]), "n": int(r["n"])},
                source=f"k_levels['levels_long'] row judge={r['judge']} arm={r['arm']} metric=Q1Q2 iteration={int(r['iteration'])}")
    if channels is not None and len(channels):
        for _, r in channels[channels["metric"].isin(list(fig_channels))].iterrows():
            put(f"channel.{r['method'].lower()}.{r['metric']}.iter{int(r['iteration'])}.{r['judge']}", _row(r, kcols),
                source=f"channel_k_frames['channels'] row judge={r['judge']} method={r['method']} metric={r['metric']} iteration={int(r['iteration'])}")
    if channels_text is not None and len(channels_text):
        for _, r in channels_text.iterrows():
            put(f"channel_text.{r['method'].lower()}.{r['metric']}.iter{int(r['iteration'])}", _row(r, kcols),
                source=f"channel_k_frames['channels_text'] row method={r['method']} metric={r['metric']} iteration={int(r['iteration'])}")
    if channels_summary is not None and len(channels_summary):
        cscols = ["n_iters", "n_sig_K0_higher", "n_sig_K5_higher", "n_sig_K0_better", "n_sig_K5_better",
                  "iters_sig_K0_higher", "iters_sig_K5_higher", "mean_delta_iters1toN", "max_abs_dz", "max_abs_dz_iter"]
        for _, r in channels_summary.iterrows():
            put(f"channel_summary.{r['method'].lower()}.{r['metric']}.{r['judge']}", _row(r, cscols),
                source=f"channel_k_frames['channels_summary'] row judge={r['judge']} method={r['method']} metric={r['metric']}")
    if did is not None and len(did):
        for _, r in did.iterrows():
            put(f"did.{r['metric']}.iter{int(r['iteration'])}.{r['judge']}",
                _row(r, ["n", "gap_K0", "gap_K5", "did_mean", "did_dz", "did_ci_lo", "did_ci_hi", "p", "p_holm", "p_holm_rubrics"]),
                source=f"did_by_iter row judge={r['judge']} metric={r['metric']} iteration={int(r['iteration'])}",
                note="did = (PTO_LA0-GRPO_LA0)-(PTO_LA5-GRPO_LA5) per persona; + => PTO lead larger at K=0")
    if method_gap is not None and len(method_gap):
        for _, r in method_gap.iterrows():
            put(f"method_gap.K{int(r['K'])}.{r['metric']}.iter{int(r['iteration'])}.{r['judge']}",
                _row(r, ["n", "delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "p_holm_rubrics", "favours"]),
                source=f"method_gap_by_iter row judge={r['judge']} K={int(r['K'])} metric={r['metric']} iteration={int(r['iteration'])}",
                note="+ => PTO higher")
    if endpoints is not None and len(endpoints):
        for _, r in endpoints.iterrows():
            put(f"endpoint.{r['pair'].split(' (')[0]}.{r['metric']}",
                _row(r, ["primary_n", "primary_delta", "primary_dz", "primary_ci_lo", "primary_ci_hi", "primary_p",
                         "primary_p_holm", "judge_delta", "judge_dz", "judge_ci_lo", "judge_ci_hi", "judge_p",
                         "judge_p_holm", "same_sign", "judge_ci_excl0"]),
                source=f"endpoint_contrasts row pair='{r['pair']}' metric={r['metric']}", note="+ => A (left model) higher")
    put("conventions", {"sign": "+ => K=0 higher (K=0 minus K=5)", "pairing": "persona_id (n=96)",
                        "holm_family": "iterations 0..N within (judge, method, metric)",
                        "oracle_repeatability_band": oracle_noise,
                        "censoring": (support_note(levels_long, base_col="")
                                      if levels_long is not None else "") or CENSOR_NOTE,
                        "iteration0": "two independent base draws (K=0-arm base vs K=5-arm base)"},
        source="eda_analysis.lookahead")
    return L
