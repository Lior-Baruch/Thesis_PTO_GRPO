"""
shared_base.py — the GRPO look-ahead contrast against ONE shared Base (family ``lookahead/shared_base``).

Both GRPO runs start from the same untrained Llama-3.2-1B, and each drew its own 96 iteration-0
conversations. Every other ``lookahead`` family keeps those two draws apart: iteration 0 is read as
a free noise floor (two independent draws of one policy). The GRPO paper reports ONE Base instead
(Lior, 2026-09-24): the two draws pooled, 192 conversations, two per persona. Both runs use the same
iteration-0 persona order, so ``file_index`` *i* is the same persona in both draws. This module is
that view and nothing else:

* :func:`share_base` — the one transform. ``mode="per_arm"`` gives EVERY arm of a method the pooled
  Base as its iteration 0, so any per-arm function (levels, vs-Base gains, trajectories, pooled-turn
  process statistics) reads the same Base on both sides. ``mode="single"`` attaches the pooled Base
  to the method's first arm only — the 21-state view (Base + 10 + 10) for statistics taken ACROSS
  model states, which must not count the Base twice. The second draw's ``file_index`` is offset by
  :data:`DRAW_OFFSET` so ``(arm, iteration, file_index)`` stays unique; ``persona_id`` is untouched,
  so persona-paired code averages the Base's two conversations per persona
  (:func:`~eda_analysis.lookahead.wide_by_persona` pivots with ``mean``).
* Every Base-dependent table the paper quotes, recomputed on shared-Base frames by calling the
  owning module's own function: levels, the K contrast, gains over the Base, the complete score
  table, the process tables (+ the change-talk persistence split of
  :func:`~eda_analysis.process.persistence_by_state`), the embedding-space tables, the judge-free
  over-praise marker, session length, and the saturation statistics of the paper's Appendix E.

**K contrasts start at iteration 1.** Iteration 0 is one policy with one Base, so there is nothing
to contrast there; the Holm family is ITERATIONS 1..N within (judge, method, metric) — one test
fewer than the per-arm-base families, whose family includes the base-vs-base row.

Contract (as :mod:`eda_analysis.lookahead`): functions take frames and return tidy frames; no disk
I/O; :func:`shared_base_numbers` returns the ledger for ``exports.save_numbers``.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import BOOT_SEED, DISPLAY_NAMES, k_of, method_of
from .ledger import ledger_entry, round3
from .lookahead import (LOWER_BETTER, RUBRICS, best_iteration, model_name, paired_k_frames,
                        wide_by_persona)
from .stats import paired_arrays
from . import process as _process

__all__ = [
    "DRAW_OFFSET", "INSTRUMENTS", "share_base", "share_base_frames", "levels", "k_contrast",
    "significant_iterations", "score_table", "gains", "process_tables", "text_tables",
    "marker_and_length", "sd_tables", "sd_trend", "agreement_by_state", "agreement_summary",
    "state_pair_contrasts", "judge_offset", "shared_base_numbers",
]

#: Added to the second base draw's ``file_index`` so (arm, iteration, file_index) stays unique.
DRAW_OFFSET = 1000
#: The eight instruments of the cross-state statistics (the Q1+Q2 composite is not a ninth rater).
INSTRUMENTS = ["Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI"]


# ── the transform ───────────────────────────────────────────────────────────────

def share_base(df: pd.DataFrame, *, mode: str = "per_arm", arm_col: str = "arm",
               iter_col: str = "iteration", file_col: str = "file_index") -> pd.DataFrame:
    """Replace each arm's iteration-0 rows with the union of its method's base draws.

    ``per_arm``: every arm of the method gets the union (labels ``arm``/``K``/``model`` rewritten to
    that arm's). ``single``: only the method's first arm (sorted: ``*_LA0``) gets it and the other
    arms lose their iteration 0. Rows gain ``base_draw`` (the arm whose draw they came from; NaN
    off the Base). Works on any per-conversation, per-utterance or long score frame that carries
    ``arm`` and ``iteration``.
    """
    if mode not in ("per_arm", "single"):
        raise ValueError(f"mode must be 'per_arm' or 'single', got {mode!r}")
    if df is None or df.empty or iter_col not in df.columns:
        return df
    is0 = df[iter_col].astype(int) == 0
    parts = [df[~is0]]
    arms_all = sorted(df[arm_col].dropna().unique())
    for m in sorted({method_of(a) for a in arms_all}):
        arms_m = [a for a in arms_all if method_of(a) == m]
        draws = []
        for r, a in enumerate(arms_m):
            b = df[is0 & (df[arm_col] == a)].copy()
            if b.empty:
                continue
            b["base_draw"] = a
            if file_col in b.columns:
                b[file_col] = b[file_col].astype(int) + r * DRAW_OFFSET
            draws.append(b)
        if not draws:
            continue
        union = pd.concat(draws)
        for a in (arms_m if mode == "per_arm" else arms_m[:1]):
            u = union.copy()
            u[arm_col] = a
            if "K" in u.columns:
                u["K"] = k_of(a)
            if "model" in u.columns:
                u["model"] = model_name(m, k_of(a), 0)
            parts.append(u)
    return pd.concat(parts, ignore_index=True)


def share_base_frames(frames: Mapping[str, pd.DataFrame], *, mode: str = "per_arm") -> Dict[str, pd.DataFrame]:
    """:func:`share_base` over a ``{judge: frame}`` mapping, order kept (primary first)."""
    return {j: share_base(f, mode=mode) for j, f in frames.items()}


def _judges(frames: Mapping[str, pd.DataFrame]):
    keys = list(frames)
    return keys[0], (keys[1] if len(keys) > 1 else None)


def _better(metric: str, mean_k0: float, mean_k5: float) -> str:
    if np.isnan(mean_k0) or np.isnan(mean_k5) or mean_k0 == mean_k5:
        return ""
    k5_higher = mean_k5 > mean_k0
    return "K5" if (k5_higher != (metric in LOWER_BETTER)) else "K0"


# ── levels, the K contrast, the complete score table, gains ─────────────────────

def levels(sc_sb: Mapping[str, pd.DataFrame], metrics: Sequence[str] = RUBRICS) -> pd.DataFrame:
    """Per (judge, arm, metric, iteration): n, mean, sd, se over the state's conversations.

    On ``per_arm`` frames the Base rows are identical for both arms (192 conversations each)."""
    rows = []
    for j, f in sc_sb.items():
        d = f[f["questionnaire"].isin(metrics)]
        g = (d.groupby(["arm", "questionnaire", "iteration"])["score"]
             .agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1)).reset_index())
        g["se"] = g["sd"] / np.sqrt(g["n"])
        rows.append(g.rename(columns={"questionnaire": "metric"}).assign(judge=j))
    out = pd.concat(rows, ignore_index=True)
    order = {m: i for i, m in enumerate(metrics)}
    return (out.assign(_o=out["metric"].map(order))
            .sort_values(["judge", "_o", "arm", "iteration"], key=lambda s: s if s.name != "judge"
                         else s.map({j: i for i, j in enumerate(sc_sb)}))
            .drop(columns="_o")[["judge", "arm", "metric", "iteration", "n", "mean", "sd", "se"]]
            .reset_index(drop=True))


def k_contrast(sc_sb: Mapping[str, pd.DataFrame], metrics: Sequence[str] = RUBRICS, *,
               method: str = "GRPO") -> pd.DataFrame:
    """The persona-paired K contrast at iterations 1..N (Holm across those iterations within
    (judge, metric)), in the EDA's sign (``mean_delta`` = K=0 − K=5) PLUS the paper's sign
    (``delta_K5_minus_K0``, ``dz_K5_minus_K0``, CI swapped) and ``better`` (``K5``/``K0``, read
    through lower-is-better)."""
    trained = {j: f[f["iteration"] > 0] for j, f in sc_sb.items()}
    fr = paired_k_frames(trained, methods=(method,), metrics=list(metrics), holm_family="iterations")
    if fr.empty:
        return fr
    fr["delta_K5_minus_K0"] = -fr["mean_delta"]
    fr["dz_K5_minus_K0"] = -fr["dz"]
    fr["ci_lo_K5_minus_K0"] = -fr["ci_hi"]
    fr["ci_hi_K5_minus_K0"] = -fr["ci_lo"]
    fr["better"] = [_better(m, a, b) for m, a, b in zip(fr["metric"], fr["mean_K0"], fr["mean_K5"])]
    return fr


def significant_iterations(kc: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per (judge, metric): the iterations where the K contrast clears Holm, split by which arm is
    better, with counts — the "significant at 6 of 10 iterations" sentences, read off a table."""
    rows = []
    for (j, m), g in kc.groupby(["judge", "metric"], sort=False):
        sig = g[g["p_holm"] < alpha]
        k5 = sorted(int(i) for i in sig.loc[sig["better"] == "K5", "iteration"])
        k0 = sorted(int(i) for i in sig.loc[sig["better"] == "K0", "iteration"])
        rows.append({"judge": j, "metric": m, "n_iterations": int(g["iteration"].nunique()),
                     "n_sig_K5_better": len(k5), "iters_K5_better": ", ".join(map(str, k5)),
                     "n_sig_K0_better": len(k0), "iters_K0_better": ", ".join(map(str, k0)),
                     "first_K5_lead": int(g.loc[g["better"] == "K5", "iteration"].min())
                     if (g["better"] == "K5").any() else np.nan})
    return pd.DataFrame(rows)


def score_table(lv: pd.DataFrame, kc: pd.DataFrame, judge: str, *,
                metrics: Sequence[str] = RUBRICS, alpha: float = 0.05, decimals: int = 3) -> pd.DataFrame:
    """The complete score table for one judge: rows = Base, K=0 iterations 1..N, K=5 iterations
    1..N; one column per instrument (display names); cell = the mean, with ``*`` on the BETTER
    arm's cell at each iteration where the K contrast clears Holm (read through lower-is-better)."""
    d = lv[(lv["judge"] == judge) & lv["metric"].isin(metrics)]
    k = kc[(kc["judge"] == judge) & kc["metric"].isin(metrics)].set_index(["metric", "iteration"])
    iters = sorted(int(i) for i in d.loc[d["iteration"] > 0, "iteration"].unique())
    rows = []

    def cell(arm, it, m):
        v = d[(d["arm"] == arm) & (d["iteration"] == it) & (d["metric"] == m)]["mean"]
        if v.empty:
            return ""
        s = f"{float(v.iloc[0]):.{decimals}f}"
        if it > 0 and (m, it) in k.index:
            r = k.loc[(m, it)]
            if float(r["p_holm"]) < alpha and r["better"] == ("K0" if arm.endswith("LA0") else "K5"):
                s += "*"
        return s

    base_arm = sorted(d["arm"].unique())[0]
    rows.append({"row": "Base", "arm": "Base", "iteration": 0,
                 **{DISPLAY_NAMES.get(m, m): cell(base_arm, 0, m) for m in metrics}})
    for arm, lab in (("GRPO_LA0", "K=0"), ("GRPO_LA5", "K=5")):
        for it in iters:
            rows.append({"row": f"{lab}, iteration {it}", "arm": arm, "iteration": it,
                         **{DISPLAY_NAMES.get(m, m): cell(arm, it, m) for m in metrics}})
    return pd.DataFrame(rows)


def gains(sc_sb: Mapping[str, pd.DataFrame], metrics: Sequence[str] = ("Q1Q2",), *,
          method: str = "GRPO", anchor_metric: str = "Q1Q2") -> pd.DataFrame:
    """Each arm's gain over the shared Base, persona-paired (Base = the persona's two-draw mean).

    Anchors: ``last`` (both arms at their last iteration) and ``best_K0`` (the K=0 arm at its best
    trained iteration on ``anchor_metric`` under THAT judge, the K=5 arm still at its last).
    ``ratio_K5_over_K0`` = K=5's gain / K=0's gain at that anchor (on the K=5 rows)."""
    rows = []
    a0, a5 = f"{method}_LA0", f"{method}_LA5"
    for j, f in sc_sb.items():
        last0 = int(f.loc[f["arm"] == a0, "iteration"].max())
        last5 = int(f.loc[f["arm"] == a5, "iteration"].max())
        best0 = best_iteration(f, a0, anchor_metric)
        base_col = model_name(method, 0, 0)
        for m in metrics:
            W = wide_by_persona(f, m)
            if W.empty or base_col not in W.columns:
                continue
            base = W[base_col]
            for anchor, it0 in (("last", last0), ("best_K0", best0)):
                g = {}
                for arm, it in ((a0, it0), (a5, last5)):
                    col = model_name(method, k_of(arm), it)
                    if col not in W.columns:
                        continue
                    r = paired_arrays(W[col].to_numpy(), base.to_numpy())
                    g[arm] = r["mean_delta"]
                    rows.append({"judge": j, "metric": m, "anchor": anchor, "arm": arm, "iteration": it,
                                 "base_mean": float(base.mean()), "mean": float(W[col].mean()),
                                 "gain": r["mean_delta"], "dz": r["dz"], "ci_lo": r["ci_lo"],
                                 "ci_hi": r["ci_hi"], "p": r["p"], "n": r["n"]})
                if a0 in g and a5 in g and g[a0]:
                    rows[-1]["ratio_K5_over_K0"] = g[a5] / g[a0]
    out = pd.DataFrame(rows)
    if "ratio_K5_over_K0" not in out.columns:
        out["ratio_K5_over_K0"] = np.nan
    return out


# ── the process coder (MIPROC) ──────────────────────────────────────────────────

def process_tables(conv: pd.DataFrame, miti: Optional[pd.DataFrame], pct: Optional[pd.DataFrame],
                   arms, *, judge: str, method: str = "GRPO") -> Dict[str, pd.DataFrame]:
    """Every process table of the paper on a shared Base, for ONE judge's MIPROC frame.

    Keys: ``levels`` (per-state means of the per-conversation metrics), ``k_paired`` (K contrast,
    iterations 1..N), ``yield``, ``responsiveness``, ``ct_trajectory``, ``persistence`` (pooled
    turns), ``k_persistence`` (K contrast on the per-conversation persistence metrics),
    ``conditioned_yield``, ``parity`` + ``parity_pooled`` (21 states: ``single`` mode)."""
    sb = share_base(conv, mode="per_arm")
    one = share_base(conv, mode="single")
    out = {"levels": _process.state_table(sb),
           "yield": _process.transition_yield(sb),
           "responsiveness": _process.responsiveness(sb),
           "ct_trajectory": _process.ct_trajectory(sb),
           "persistence": _process.persistence_by_state(sb),
           "conditioned_yield": _process.conditioned_yield(sb)}
    trained = sb[sb["iteration"] > 0]
    sl = _process.to_scores_long(trained, arms)
    kp = paired_k_frames({judge: sl}, methods=(method,), metrics=_process.PROCESS_K_METRICS,
                         holm_family="iterations")
    if not kp.empty:
        kp["lower_better"] = kp["metric"].isin(_process.LOWER_BETTER)
        kp["better"] = [("" if np.isnan(a) or np.isnan(b) or a == b else
                         ("K5" if ((b > a) != (m in _process.LOWER_BETTER)) else "K0"))
                        for m, a, b in zip(kp["metric"], kp["mean_K0"], kp["mean_K5"])]
    out["k_paired"] = kp
    pm = _process.persistence_metrics(sb)
    pm_long = pm[pm["iteration"] > 0].melt(
        id_vars=[c for c in pm.columns if c not in _process.PERSISTENCE_METRICS],
        value_vars=_process.PERSISTENCE_METRICS, var_name="questionnaire", value_name="score"
    ).dropna(subset=["score"])
    kpe = paired_k_frames({judge: pm_long}, methods=(method,), metrics=_process.PERSISTENCE_METRICS,
                          holm_family="iterations")
    if not kpe.empty:
        lb = {"ct_relapse"}
        kpe["lower_better"] = kpe["metric"].isin(lb)
        kpe["better"] = [("" if np.isnan(a) or np.isnan(b) or a == b else
                          ("K5" if ((b > a) != (m in lb)) else "K0"))
                         for m, a, b in zip(kpe["metric"], kpe["mean_K0"], kpe["mean_K5"])]
    out["k_persistence"] = kpe
    out["persistence_levels"] = (pm.groupby(["arm", "iteration"])[_process.PERSISTENCE_METRICS]
                                 .agg(["mean", "sem", "count"]))
    out["persistence_levels"].columns = [f"{a}_{b}" for a, b in out["persistence_levels"].columns]
    out["persistence_levels"] = out["persistence_levels"].reset_index()
    if miti is not None and pct is not None and not (miti.empty and pct.empty):
        par = _process.parity(one, share_base(miti, mode="single"), share_base(pct, mode="single"))
        out["parity"] = par
        out["parity_pooled"] = _process.parity_pooled(par) if not par.empty else pd.DataFrame()
    return out


# ── the text (sentence-embedding) tables ────────────────────────────────────────

def text_tables(utt: pd.DataFrame, E: np.ndarray, *, n_boot: int = 300,
                seed: int = BOOT_SEED) -> Dict[str, pd.DataFrame]:
    """Drift from the shared Base centroid, the cosine between the two arms' displacements, and the
    diversity scalars (template similarity, between-conversation variance share), on a shared Base.

    ``utt`` must be the frame ``E`` is aligned to (row i ↔ ``E[i]``)."""
    from . import text as _text
    u = utt.reset_index(drop=True).assign(_row=np.arange(len(utt)))
    sb = share_base(u, mode="per_arm")
    Esb = E[sb["_row"].to_numpy()]
    sb = sb.drop(columns="_row")
    cents = _text.state_centroids(sb, Esb)
    drift, pooled = _text.drift_by_state(cents)
    cos = _text.drift_cosines(cents, pooled)
    cos = cos[[c for c in cos.columns if c == "iteration" or c.endswith("_GRPO")]]
    div = _text.diversity_by_state(sb, Esb, n_boot=n_boot, seed=seed)
    return {"drift": drift, "drift_cosines": cos, "diversity": div}


# ── the judge-free marker, session length, turn length ─────────────────────────

def marker_and_length(tm: pd.DataFrame, *, marker_col: str = "lex_overpraise_marker_rate") -> pd.DataFrame:
    """Per (arm, iteration) on a shared Base: the judge-free over-praise marker (share of therapist
    turns with at least one of the ten patterns, mean over conversations), conversation length
    (utterances) and mean therapist turn length (characters)."""
    sb = share_base(tm, mode="per_arm")
    cols = [c for c in (marker_col, "conv_len", "mean_turn_len", "n_th_turns") if c in sb.columns]
    g = sb.groupby(["arm", "iteration"])
    out = g[cols].mean().join(g[cols].sem().add_suffix("_se")).join(g.size().rename("n_conv"))
    return out.reset_index()


# ── Appendix E: saturation, agreement, sign preservation ───────────────────────

def sd_tables(sc_sb: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (judge, metric, arm, iteration): n, mean, median, sd, iqr and the ceiling shares
    (``share_ge4`` / ``share_ge45`` / ``share_eq5``) — :func:`replication.sd_by_iter` on a shared
    Base (the Base rows use all 192 conversations)."""
    from .replication import sd_by_iter
    return sd_by_iter(dict(sc_sb))


def sd_trend(sd: pd.DataFrame, metrics: Sequence[str] = ("Q1", "Q1Q2")) -> pd.DataFrame:
    """Per (judge, metric, arm): Spearman ρ of the per-conversation SD against iteration (0..N and
    1..N), and the variance ratio of the last iteration against the Base and against iteration 1."""
    from scipy.stats import spearmanr
    rows = []
    for (j, m, a), g in sd[sd["metric"].isin(metrics)].groupby(["judge", "metric", "arm"], sort=False):
        g = g.sort_values("iteration")
        s = g.set_index("iteration")["sd"]
        last = int(s.index.max())
        r_all = spearmanr(s.index, s.values)
        r_tr = spearmanr(s.index[s.index > 0], s.values[s.index > 0])
        rows.append({"judge": j, "metric": m, "arm": a, "sd_base": float(s.get(0, np.nan)),
                     "sd_iter1": float(s.get(1, np.nan)), "sd_last": float(s.loc[last]), "last": last,
                     "rho_0_to_last": float(r_all.statistic), "p_0_to_last": float(r_all.pvalue),
                     "rho_1_to_last": float(r_tr.statistic), "p_1_to_last": float(r_tr.pvalue),
                     "var_ratio_last_over_base": float(s.loc[last] ** 2 / s.get(0, np.nan) ** 2),
                     "var_ratio_last_over_iter1": float(s.loc[last] ** 2 / s.get(1, np.nan) ** 2)})
    return pd.DataFrame(rows)


def agreement_by_state(sc_one: Mapping[str, pd.DataFrame], metrics: Sequence[str] = INSTRUMENTS) -> pd.DataFrame:
    """Per (state, metric): per-conversation Pearson r between the two judges, on ``single`` frames
    (21 states; the Base state pairs all 192 conversations)."""
    pj, hj = _judges(sc_one)
    keys = ["arm", "iteration", "file_index", "questionnaire"]
    P = sc_one[pj][keys + ["score"]]
    H = sc_one[hj][keys + ["score"]]
    M = P.merge(H, on=keys, suffixes=("_primary", "_heldout"))
    M = M[M["questionnaire"].isin(metrics)]
    rows = []
    for (a, it, m), g in M.groupby(["arm", "iteration", "questionnaire"]):
        x, y = g["score_primary"].astype(float), g["score_heldout"].astype(float)
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan
        rows.append({"state": "Base" if it == 0 else f"{a} it {it}", "arm": a if it else "Base",
                     "iteration": int(it), "metric": m, "n": len(g), "pearson_r": r})
    return pd.DataFrame(rows)


def agreement_summary(agr: pd.DataFrame, *, focus_arm: str = "GRPO_LA5", focus_iter: int = 10) -> pd.DataFrame:
    """Per metric: r at the focus state, the median over all states, the difference, and the focus
    state's rank (1 = the worst-agreeing state of that metric), out of the number of states."""
    rows = []
    for m, g in agr.groupby("metric", sort=False):
        g = g.dropna(subset=["pearson_r"]).sort_values("pearson_r").reset_index(drop=True)
        f = g[(g["arm"] == focus_arm) & (g["iteration"] == focus_iter)]
        if f.empty:
            continue
        r = float(f["pearson_r"].iloc[0])
        rows.append({"metric": m, "r_focus": r, "median": float(g["pearson_r"].median()),
                     "delta_median": r - float(g["pearson_r"].median()),
                     "rank": int(f.index[0]) + 1, "n_states": len(g)})
    order = {m: i for i, m in enumerate(INSTRUMENTS)}
    return pd.DataFrame(rows).sort_values("metric", key=lambda s: s.map(order)).reset_index(drop=True)


def state_pair_contrasts(sc_one: Mapping[str, pd.DataFrame], metrics: Sequence[str] = INSTRUMENTS, *,
                         seed: int = BOOT_SEED, n_boot: int = 2000) -> pd.DataFrame:
    """Every pair of the 21 states × metric under both judges, persona-paired (the Base's persona
    value is its two-draw mean): deltas, dz, the held-out side's bootstrap CI and ``same_sign`` —
    the columns :func:`reliability.sign_preservation` reads."""
    pj, hj = _judges(sc_one)
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        Wp, Wh = wide_by_persona(sc_one[pj], m), wide_by_persona(sc_one[hj], m)
        models = sorted(set(Wp.columns) & set(Wh.columns))
        for a, b in combinations(models, 2):
            rec = {"metric": m, "model_a": a, "model_b": b}
            for src, W in (("primary", Wp), ("judge", Wh)):
                d = (W[a] - W[b]).dropna().to_numpy(float)
                rec[f"{src}_n"] = len(d)
                rec[f"{src}_delta"] = float(d.mean()) if len(d) else np.nan
                sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
                rec[f"{src}_dz"] = float(d.mean() / sd) if sd else np.nan
                if src == "judge" and n_boot and len(d) >= 3:
                    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
                    rec["judge_ci_lo"], rec["judge_ci_hi"] = (float(np.percentile(bs, 2.5)),
                                                             float(np.percentile(bs, 97.5)))
            rec["same_sign"] = bool(np.sign(rec["judge_delta"]) == np.sign(rec["primary_delta"]))
            rows.append(rec)
    return pd.DataFrame(rows)


def judge_offset(lv_one: pd.DataFrame, metric: str = "Q1Q2") -> pd.DataFrame:
    """Per state: the primary's mean minus the held-out judge's on ``metric`` (21 states)."""
    judges = list(dict.fromkeys(lv_one["judge"]))
    pj, hj = judges[0], judges[1]
    d = lv_one[lv_one["metric"] == metric].pivot_table(index=["arm", "iteration"], columns="judge",
                                                       values="mean").reset_index()
    d["offset"] = d[pj] - d[hj]
    return d.rename(columns={pj: "mean_primary", hj: "mean_heldout"})


# ── the ledger ──────────────────────────────────────────────────────────────────

def shared_base_numbers(*, lv: pd.DataFrame, gains_t: pd.DataFrame, sig: pd.DataFrame,
                        proc: Mapping[str, Dict[str, pd.DataFrame]], text_t: Dict[str, pd.DataFrame],
                        marker: pd.DataFrame, sd: pd.DataFrame, trend: pd.DataFrame,
                        agr: pd.DataFrame, agr_sum: pd.DataFrame, sign: pd.DataFrame,
                        offset: pd.DataFrame) -> Dict[str, dict]:
    """The quotable scalars of the family, each citing the table it is read from."""
    out: Dict[str, dict] = {}

    def put(key, value, source, note=""):
        out[key] = ledger_entry(value, source, note)

    for r in lv[lv["iteration"] == 0].drop_duplicates(["judge", "metric"]).itertuples():
        put(f"levels.{r.judge}.{r.metric}.base", round3(r.mean), "levels_long",
            f"shared Base, n = {int(r.n)} conversations")
    for r in lv[lv["iteration"] == lv.groupby(["judge", "arm"])["iteration"].transform("max")].itertuples():
        put(f"levels.{r.judge}.{r.metric}.{r.arm}.final", round3(r.mean), "levels_long", f"iteration {r.iteration}")
    for r in gains_t.itertuples():
        put(f"gains.{r.judge}.{r.metric}.{r.anchor}.{r.arm}", round3(r.gain), "gains",
            f"iteration {r.iteration} vs Base; dz {r.dz:.3f}")
        if not pd.isna(r.ratio_K5_over_K0):
            put(f"gains.{r.judge}.{r.metric}.{r.anchor}.ratio", round(float(r.ratio_K5_over_K0), 3), "gains",
                "K=5 gain / K=0 gain")
    for r in sig.itertuples():
        put(f"sig.{r.judge}.{r.metric}.K5_better", r.iters_K5_better, "significant_iterations",
            f"{r.n_sig_K5_better} of {r.n_iterations} iterations (Holm across iterations 1..N)")
        if r.n_sig_K0_better:
            put(f"sig.{r.judge}.{r.metric}.K0_better", r.iters_K0_better, "significant_iterations",
                f"{r.n_sig_K0_better} of {r.n_iterations}")
    for j, t in proc.items():
        lvp = t["levels"]
        base = lvp[lvp["iteration"] == 0].iloc[0]
        for mcol in ("th_PRA_rate", "th_CR_rate", "th_PERS_rate", "th_OQ_rate", "mi_adherent_rate",
                     "mi_incons_rate", "refl_after_ct", "pra_after_st", "pers_after_st",
                     "refl_after_st", "ct_prop"):
            if mcol in lvp.columns:
                put(f"process.{j}.{mcol}.base", round3(base[mcol]), f"process_levels_{j}", "shared Base")
        y = t["yield"]
        by = y[(y["iteration"] == 0) & (y["arm"] == y["arm"].min())]
        for rr in by.itertuples():
            put(f"process.{j}.yield.base.{rr.th_code}", round3(rr.p_ct), f"yield_{j}", f"n = {rr.n} turns")
        pe = t["persistence"]
        for rr in pe[pe["prev_code"].isin(["CT", "ST"])].itertuples():
            if rr.iteration == 0 and rr.arm != pe["arm"].min():
                continue
            lab = "base" if rr.iteration == 0 else f"{rr.arm}.it{rr.iteration}"
            put(f"persistence.{j}.{lab}.after_{rr.prev_code}.p_ct", round3(rr.p_ct), f"persistence_{j}",
                f"n = {rr.n} pairs")
    dv = text_t["diversity"]
    for rr in dv.itertuples():
        if rr.iteration == 0 and rr.arm != dv["arm"].min():
            continue
        lab = "base" if rr.iteration == 0 else f"{rr.arm}.it{rr.iteration}"
        put(f"text.{lab}.persona_var_share", round3(rr.persona_var_share), "text_diversity")
        put(f"text.{lab}.template_sim", round3(rr.template_sim), "text_diversity")
    for rr in text_t["drift_cosines"].itertuples():
        put(f"text.cos_K0_K5.it{rr.iteration}", round3(rr.cos_K0_K5_GRPO), "text_drift_cosines")
    for rr in marker.itertuples():
        if rr.iteration == 0 and rr.arm != marker["arm"].min():
            continue
        lab = "base" if rr.iteration == 0 else f"{rr.arm}.it{rr.iteration}"
        put(f"marker.{lab}", round3(getattr(rr, "lex_overpraise_marker_rate")), "marker_and_length")
        if lab == "base":
            put("length.base.conv_len", round3(rr.conv_len), "marker_and_length", "utterances, shared Base")
            put("length.base.mean_turn_len", round3(rr.mean_turn_len), "marker_and_length", "chars per therapist turn")
    q1 = sd[(sd["metric"] == "Q1")]
    for rr in q1[(q1["iteration"].isin([0, q1["iteration"].max()]))].itertuples():
        if rr.iteration == 0 and rr.arm != q1["arm"].min():
            continue
        lab = "base" if rr.iteration == 0 else f"{rr.arm}.it{rr.iteration}"
        put(f"sat.{rr.judge}.Q1.{lab}.share_ge45", round3(rr.share_ge45), "sd_by_iter")
        put(f"sat.{rr.judge}.Q1.{lab}.share_eq5", round3(rr.share_eq5), "sd_by_iter")
        put(f"sat.{rr.judge}.Q1.{lab}.sd", round3(rr.sd), "sd_by_iter")
    for rr in trend.itertuples():
        put(f"sat.{rr.judge}.{rr.metric}.{rr.arm}.rho_0_to_last", round3(rr.rho_0_to_last), "sd_trend",
            f"p = {rr.p_0_to_last:.3f}")
        put(f"sat.{rr.judge}.{rr.metric}.{rr.arm}.var_ratio_last_over_base", round3(rr.var_ratio_last_over_base),
            "sd_trend", f"vs iteration 1: {rr.var_ratio_last_over_iter1:.3f}")
    for rr in agr_sum.itertuples():
        put(f"agreement.{rr.metric}.GRPO_LA5_it10", round3(rr.r_focus), "agreement_summary",
            f"median over {rr.n_states} states {rr.median:.3f}; rank {rr.rank}/{rr.n_states}")
    q1a = agr[(agr["metric"] == "Q1")].sort_values("pearson_r")
    if len(q1a):
        put("agreement.Q1.median", round3(q1a["pearson_r"].median()), "agreement_by_state",
            f"{len(q1a)} states")
    ss = sign.set_index("subset") if "subset" in sign.columns else pd.DataFrame()
    if len(ss):
        a = ss.loc["all contrasts"]
        put("sign.all", f"{int(a.n_same_sign)} of {int(a.n_contrasts)}", "sign_preservation",
            f"{a.pct_same_sign:.1f}% same sign")
    put("offset.Q1Q2.min", round3(offset["offset"].min()), "judge_offset", "primary − held-out, over states")
    put("offset.Q1Q2.max", round3(offset["offset"].max()), "judge_offset", "primary − held-out, over states")
    return out
