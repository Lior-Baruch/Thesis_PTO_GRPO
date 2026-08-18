"""dispersion.py — does K-turn look-ahead WIDEN the training signal, or merely RESCALE it?

Promoted 2026-08-18 from the look-ahead paper's generator
``papers/2026_lookahead_pto_grpo/analysis/dispersion_by_k.py`` (its ``tables/dispersion_by_k_*.csv``
and ``analysis/out/dispersion_by_k.json`` are the frozen fixture these functions reproduce; the
self-check anchors on ``PTO_LA5 margin_mean at train_iter 1 = 0.424``). Renders in
``lookahead/mechanism``.

What it tests
-------------
Guards against the "look-ahead sharpens the signal" misreading. If look-ahead only multiplied
the within-group score spread by a constant, then (a) the best-worst margin and the within-group
SD rise by the SAME factor, (b) margin/SD stays at the value expected for M=8 iid draws (pure
sampling spread), and (c) part of K=5's higher PTO pair yield is an artefact of the ABSOLUTE
τ = 0.1 filter applied to margins that were rescaled up. All three are tested here:

* :func:`dispersion_by_iter`  — per arm × train_iter: within-group SD, best−worst margin, the
  scale-free margin/SD, the winner's standardized lead, a shuffle null, τ-yield, reward quantiles.
* :func:`dispersion_ratios`   — the "same factor" test: K5/K0 ratios of margin and SD (+ the
  ratio-of-ratios) with bootstrap CIs over groups, per method × iteration + pooled.
* :func:`tau_sensitivity`     — PTO pair yield vs τ, raw and after rescaling K=0's margins by the
  iteration-1 SD ratio (how much of K=5's yield advantage is pure rescaling).
* :func:`iid_expectation`     — the simulated iid-normal reference values (pure geometry of 8 draws).
* :func:`dispersion_numbers`  — the quotable-numbers ledger (dotted keys → value/source/note) for
  ``exports.save_numbers``.

Data + unit
-----------
Every candidate the trainers scored (``iteration_N/eda/generations.jsonl`` via
:func:`eda_analysis.training.load_generations`), all four arms, TRAINING phase only (GRPO's TRL
``eval``-loop generations are dropped, mirroring :func:`eda_analysis.pref.pair_yield_by_iter`).
A GROUP = the 8 candidates the policy sampled at one branch point (PTO: one trunk depth of one
conversation; GRPO: one prompt group in one epoch). Group key = :data:`GROUP_KEYS` =
(arm, train_iter, conversation_id, branch_id, epoch) == ``pref._GROUP_KEYS`` — ``conversation_id``
is REQUIRED for PTO because its ``branch_id`` is the trunk depth and repeats across conversations.
The grader is the training oracle (gpt-4o-mini, Q1+Q2 mean) by construction — these numbers are
NOT judge-swappable (candidate rewards cannot be re-graded after the fact).

Iteration axis
--------------
``train_iter`` n = the branching done by policy π_n. π_1 = the BASE model in every arm, so
train_iter 1 is the clean cut where the two K arms of a method differ ONLY in the look-ahead
measurement (same policy, same 96 personas) — that is where the SD ratio used for the rescaling
test is taken. ``eval_iter = train_iter − 1`` (the model_iter whose convs the policy also
produced). **GRPO_LA5 is right-censored at train_iter 5.**

Sign / direction convention
---------------------------
Ratios are **K5 / K0** (> 1 = K=5 wider); ``winner_z_diff`` = K5 − K0 (+ = K=5's winner stands
out more). This is the paper's convention for THIS table family and deliberately differs from the
rubric-contrast convention (``+ => K=0 higher``) used elsewhere in ``lookahead/``. Higher SD /
margin = wider spread, NOT better discrimination.

Estimators
----------
Within-group SD is the POPULATION SD (ddof=0) — exactly what GRPO records as ``group_std``
(checked: identical to 1e-9; :func:`dispersion_by_iter` re-asserts it) and what
``scale_rewards="group"`` divides by; the same estimator is applied to PTO. Margin = max − min of
the group's scores (τ-free, over ALL 8 candidates, i.e. ``pair_yield_by_iter``'s ``mean_margin``).
The iid-normal expectation for margin/SD (and for the winner's standardized lead
(max − mean)/SD) is SIMULATED (200k groups of 8) with the same ddof=0 estimator; a
distribution-free shuffle null (scores permuted across groups within an arm-iteration) is reported
alongside because the oracle score is bounded and discrete. NaN-scored candidates are dropped WITHIN
the group (mirrors ``pair_yield_by_iter``'s ``score.notna()``); groups with < 2 valid scores are
excluded from every statistic (``n_groups`` vs ``n_groups_all``).

Seeds
-----
Every random draw here (ratio bootstrap, shuffle null, iid simulation) is seeded with
:data:`eda_analysis.constants.BOOT_SEED` by default. The paper's frozen fixture used its own seeds
(bootstrap 0, shuffle null 7, iid simulation 123) — pass ``seed=0`` / ``null_seed=7`` /
``seed=123`` to reproduce it bit-for-bit; under the package seed the point estimates are identical
and only the CI bounds / simulated references move in the third decimal.

Contract: functions take frames / arms and return tidy ``pd.DataFrame``s (NO disk writes — the
notebook calls ``exports.*``); figures live in :mod:`eda_analysis.plotting.dispersion`.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import BOOT_SEED

__all__ = [
    "GROUP_KEYS", "TAU_TRAINER", "TAUS", "M_BRANCHES", "ARMS", "GRADER",
    "group_frame", "load_group_frame", "shuffle_null", "iid_expectation",
    "dispersion_by_iter", "dispersion_ratios", "tau_sensitivity", "dispersion_numbers",
]

GROUP_KEYS = ["arm", "train_iter", "conversation_id", "branch_id", "epoch"]   # == pref._GROUP_KEYS
TAU_TRAINER = 0.10                      # PREF_FILTER_TAU in train_PTO_Iterative.ipynb cell 1
TAUS = (0.05, 0.10, 0.15, 0.20, 0.30)
M_BRANCHES = 8                          # NUM_BRANCHES_PER_TURN == NUM_GENERATIONS
ARMS = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]
GRADER = "training oracle (gpt-4o-mini)"
_N_BOOT = 2000


def _reward_floor() -> float:
    from .training import REWARD_FLOOR
    return REWARD_FLOOR


# ── 0. references ────────────────────────────────────────────────────────────────────────────
def iid_expectation(M: int = M_BRANCHES, n_sim: int = 200_000, seed: int = BOOT_SEED) -> pd.DataFrame:
    """E[range]/E[sd] and E[range/sd] for ``M`` iid N(0,1) draws, for both SD estimators.

    One row per ``sd_estimator`` (``ddof=0`` = the population SD GRPO records as ``group_std`` and
    divides its advantages by; ``ddof=1`` = the sample SD). Columns: ``n_groups, m,
    E_range_over_sigma, E_sd_over_sigma, ratio_of_means`` (= E[range]/E[SD], the estimator used for
    ``margin_over_sd`` in the by-iter table), ``mean_of_ratios`` (= E[range/SD]),
    ``median_of_ratios``, ``winner_z_mean`` (= E[(max − mean)/SD], the standardized lead of the
    best candidate over its group). Neither depends on the grader (pure geometry of 8 draws).
    The paper's fixture used ``seed=123``.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_sim, M))
    rngs = x.max(axis=1) - x.min(axis=1)
    wz = x.max(axis=1) - x.mean(axis=1)
    rows = []
    for ddof in (0, 1):
        sd = x.std(axis=1, ddof=ddof)
        rows.append({"sd_estimator": f"ddof={ddof}", "n_groups": n_sim, "m": M,
                     "E_range_over_sigma": float(rngs.mean()),
                     "E_sd_over_sigma": float(sd.mean()),
                     "ratio_of_means": float(rngs.mean() / sd.mean()),
                     "mean_of_ratios": float((rngs / sd).mean()),
                     "median_of_ratios": float(np.median(rngs / sd)),
                     "winner_z_mean": float((wz / sd).mean())})
    return pd.DataFrame(rows)


def shuffle_null(scores, m: int = M_BRANCHES, n_perm: int = 100, seed: int = BOOT_SEED) -> dict:
    """Distribution-free reference: permute an (arm, iter)'s candidate scores across groups
    (destroying within-group structure but keeping the arm's actual bounded/discrete score
    distribution), regroup into groups of ``m``, and return the mean margin/SD (ratio of means) and
    mean winner z over permutations. NB ``null_winner_z`` is sensitive to the skew of the pooled
    score distribution, which differs across arms/iterations. The paper's fixture used ``seed=7``.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(scores, float)
    n = (len(x) // m) * m
    if n < 2 * m:
        return {"null_margin_over_sd": np.nan, "null_winner_z": np.nan}
    ms, wzs = [], []
    for _ in range(n_perm):
        y = rng.permutation(x)[:n].reshape(-1, m)
        sd = y.std(axis=1, ddof=0)
        rg = y.max(axis=1) - y.min(axis=1)
        ms.append(rg.mean() / sd.mean())
        ok = sd > 0
        wzs.append(((y.max(axis=1) - y.mean(axis=1))[ok] / sd[ok]).mean())
    return {"null_margin_over_sd": float(np.mean(ms)), "null_winner_z": float(np.mean(wzs))}


# ── 1. per-group frame ────────────────────────────────────────────────────────────────────────
def group_frame(gens: pd.DataFrame) -> pd.DataFrame:
    """One row per group (branch point / prompt group) from a ``load_generations`` frame.

    Columns: the :data:`GROUP_KEYS` + ``method, K, n_cand, n_nan, n_floored, rec_group_std``
    (GRPO's logged ``group_std``), ``n_valid, sd0, sd1, smin, smax, smean, margin, winner_z,
    eval_iter``. Drops TRL eval-phase rows (no gradient). NaN-scored candidates are dropped WITHIN
    the group (mirrors ``pair_yield_by_iter``'s ``score.notna()``); the count is kept as ``n_nan``.
    ``winner_z`` = (max − mean)/sd0, NaN where sd0 == 0.
    """
    d = gens[gens["phase"] != "eval"].copy()
    d["epoch"] = d["epoch"].fillna(-1.0)          # PTO has no epoch; groupby drops NaN keys
    d["score_na"] = d["score"].isna()
    d["floored_"] = d["score"].notna() & (d["score"] <= _reward_floor())
    meta = d.groupby(GROUP_KEYS, observed=True).agg(
        method=("method", "first"), K=("K", "first"),
        n_cand=("score", "size"), n_nan=("score_na", "sum"), n_floored=("floored_", "sum"),
        rec_group_std=("group_std", "first"),
    )
    v = d.dropna(subset=["score"])
    gb = v.groupby(GROUP_KEYS, observed=True)["score"]
    st = pd.DataFrame({
        "n_valid": gb.size(),
        "sd0": gb.std(ddof=0),
        "sd1": gb.std(ddof=1),
        "smin": gb.min(), "smax": gb.max(), "smean": gb.mean(),
    })
    out = meta.join(st, how="left").reset_index()
    out["margin"] = out["smax"] - out["smin"]
    out["winner_z"] = (out["smax"] - out["smean"]) / out["sd0"].where(out["sd0"] > 0)
    out["eval_iter"] = out["train_iter"] - 1
    return out


def load_group_frame(arms=None, *, gens: Optional[pd.DataFrame] = None,
                     arm_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """``group_frame(load_generations(arms))`` restricted to ``arm_labels`` (default: every arm
    passed). ``gens`` short-circuits the load (e.g. a frame the notebook already holds)."""
    from .training import load_generations
    if gens is None:
        gens = load_generations(arms, keep_tail=False)
    G = group_frame(gens)
    if arm_labels is not None:
        G = G[G["arm"].isin(list(arm_labels))].copy()
    return G


def _as_group_frame(source, *, gens=None) -> pd.DataFrame:
    """Accept a group frame (has ``sd0``) or an arms list and return the group frame."""
    if isinstance(source, pd.DataFrame):
        if "sd0" in source.columns:
            return source
        if "score" in source.columns and "cand_idx" in source.columns:   # a raw generations frame
            return group_frame(source)
        raise ValueError("expected a group frame (from load_group_frame) or a load_generations frame")
    return load_group_frame(source, gens=gens)


def _check_grpo_group_std(G: pd.DataFrame) -> float:
    """Recorded GRPO ``group_std`` == our ddof=0 SD (max |diff|); raises if not."""
    chk = G[(G["method"] == "GRPO") & G["rec_group_std"].notna() & G["sd0"].notna()]
    max_abs = float((chk["rec_group_std"] - chk["sd0"]).abs().max()) if len(chk) else float("nan")
    if max_abs > 1e-6:
        raise AssertionError(f"GRPO group_std is not ddof=0 (max |diff|={max_abs})")
    return max_abs


# ── 2. per arm × train_iter summary ──────────────────────────────────────────────────────────
def dispersion_by_iter(arms=None, *, gens: Optional[pd.DataFrame] = None,
                       arm_labels: Optional[Sequence[str]] = None, tau: float = TAU_TRAINER,
                       n_perm: int = 100, null_seed: int = BOOT_SEED) -> pd.DataFrame:
    """Within-group dispersion of the TRAINING reward per arm × training iteration
    (paper table ``dispersion_by_k_by_iter``).

    One row per (arm, train_iter): ``n_groups`` (groups with ≥ 2 valid scores) / ``n_groups_all``
    (every group logged); ``sd_mean/sd_median`` (population SD, ddof=0, of the group's scores);
    ``margin_mean/margin_median`` (best − worst over ALL candidates, τ-free); ``margin_over_sd`` =
    mean margin / mean SD (compare with the iid-normal ``ratio_of_means``); ``mean_group_ratio`` =
    mean of per-group margin/SD over groups with SD>0; ``winner_z`` = mean (best − group mean)/SD
    over groups with SD>0 (the "does the winner stand out?" statistic); ``null_margin_over_sd`` /
    ``null_winner_z`` = the same two after shuffling that arm-iteration's candidate scores across
    groups (``n_perm`` permutations — compare ``winner_z`` across K arms on the RAW value, the null
    is skew-sensitive); ``frac_sd_zero`` (all 8 scores identical); ``frac_groups_floored`` (a
    candidate at REWARD_FLOOR); ``frac_groups_nan_cand`` (an unscored candidate);
    ``yield_tau0.10`` (share of groups with margin > τ; PTO's τ, informative-only for GRPO);
    ``reward_median/iqr/q25/q75`` per candidate. Sorted by (method, K, train_iter).

    ``arms`` = discovered ``Arm`` objects (or ``gens`` = an already-loaded generations frame);
    ``arm_labels`` restricts the arms kept (default: all present). Re-asserts that GRPO's recorded
    ``group_std`` is the ddof=0 SD. Attaches ``.attrs["grpo_group_std_max_abs_diff"]``.
    """
    from .training import load_generations
    if gens is None:
        gens = load_generations(arms, keep_tail=False)
    G = group_frame(gens)
    if arm_labels is not None:
        G = G[G["arm"].isin(list(arm_labels))].copy()
    max_abs = _check_grpo_group_std(G)
    valid = G[G["n_valid"] >= 2].copy()
    rows = []
    for (arm, ti), g in valid.groupby(["arm", "train_iter"], observed=True):
        raw = gens[(gens["arm"] == arm) & (gens["train_iter"] == ti) & (gens["phase"] != "eval")]["score"].dropna()
        q25, q50, q75 = np.percentile(raw, [25, 50, 75])
        nz = g[g["sd0"] > 0]
        null = shuffle_null(raw.to_numpy(float), n_perm=n_perm, seed=null_seed)
        rows.append({
            "arm": arm, "method": g["method"].iloc[0], "K": int(g["K"].iloc[0]), "train_iter": int(ti),
            "eval_iter": int(ti) - 1,
            "n_groups": int(len(g)),
            "n_groups_all": int(((G["arm"] == arm) & (G["train_iter"] == ti)).sum()),
            "sd_mean": float(g["sd0"].mean()), "sd_median": float(g["sd0"].median()),
            "margin_mean": float(g["margin"].mean()), "margin_median": float(g["margin"].median()),
            "margin_over_sd": float(g["margin"].mean() / g["sd0"].mean()),
            "mean_group_ratio": float((nz["margin"] / nz["sd0"]).mean()) if len(nz) else np.nan,
            "winner_z": float(nz["winner_z"].mean()) if len(nz) else np.nan,
            "null_margin_over_sd": null["null_margin_over_sd"], "null_winner_z": null["null_winner_z"],
            "frac_sd_zero": float((g["sd0"] <= 1e-9).mean()),
            "frac_groups_floored": float((g["n_floored"] > 0).mean()),
            "frac_groups_nan_cand": float((g["n_nan"] > 0).mean()),
            f"yield_tau{tau:.2f}": float((g["margin"] > tau).mean()),
            "reward_median": float(q50), "reward_iqr": float(q75 - q25),
            "reward_q25": float(q25), "reward_q75": float(q75),
        })
    by_iter = pd.DataFrame(rows).sort_values(["method", "K", "train_iter"]).reset_index(drop=True)
    by_iter.attrs["grpo_group_std_max_abs_diff"] = max_abs
    by_iter.attrs["n_candidates_loaded"] = int(len(gens))
    by_iter.attrs["n_groups"] = int(len(G))
    return by_iter


# ── 3. K5/K0 ratios per method × iteration with bootstrap CIs over groups ────────────────────
def _ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def dispersion_ratios(source, *, gens: Optional[pd.DataFrame] = None,
                      methods: Sequence[str] = ("PTO", "GRPO"), n_boot: int = _N_BOOT,
                      seed: int = BOOT_SEED) -> pd.DataFrame:
    """The "same factor" test (paper table ``dispersion_by_k_ratios``): K=5 / K=0 ratio of the mean
    best-worst margin and of the mean within-group SD (ddof=0), per method × training iteration
    plus a ``pooled`` row (all iterations both arms have — GRPO 1–5 right-censored, PTO 1–10).

    ``source`` = a group frame (:func:`load_group_frame`) or an arms list. Columns:
    ``n_groups_K0/K5, margin_K0/K5, margin_ratio [+_lo/_hi], sd_K0/K5, sd_ratio [+_lo/_hi],
    ratio_of_ratios`` (= margin_ratio / sd_ratio; 1.0 = look-ahead scales margin and SD by the same
    factor, i.e. it rescales the spread without pulling the winner away from the pack) with
    ``ror_lo/ror_hi/ror_ci_covers_1``, ``margin_over_sd_K0/K5`` (compare to the iid-normal
    ``ratio_of_means``), ``winner_z_K0/K5`` (mean (best − group mean)/SD, groups with SD>0) and
    ``winner_z_diff`` = K5 − K0 with CI (+ = K=5's winner stands out more, in within-group SD
    units — the direct "sharper signal" statistic). 95% CIs = percentile bootstraps over groups
    (``n_boot`` resamples, groups resampled independently within each arm, the SAME resample
    driving margin and SD so the ratio-of-ratios CI is valid). Ratio > 1 = K=5 wider. train_iter 1
    is the same-policy (base) row and the only rescaling factor free of policy divergence. The
    paper's fixture used ``seed=0``.
    """
    G = _as_group_frame(source, gens=gens)
    valid = G[G["n_valid"] >= 2]
    rng = np.random.default_rng(seed)
    rrows = []
    for method in methods:
        a0, a5 = f"{method}_LA0", f"{method}_LA5"
        iters = sorted(set(valid.loc[valid.arm == a0, "train_iter"]) & set(valid.loc[valid.arm == a5, "train_iter"]))
        if not iters:
            continue
        blocks = [(int(t), valid[(valid.arm == a0) & (valid.train_iter == t)],
                   valid[(valid.arm == a5) & (valid.train_iter == t)]) for t in iters]
        blocks.append(("pooled", valid[(valid.arm == a0) & valid.train_iter.isin(iters)],
                       valid[(valid.arm == a5) & valid.train_iter.isin(iters)]))
        for t, g0, g5 in blocks:
            m0, m5 = g0["margin"].to_numpy(float), g5["margin"].to_numpy(float)
            s0, s5 = g0["sd0"].to_numpy(float), g5["sd0"].to_numpy(float)
            # joint resample: the same group indices drive margin and SD so the ratio-of-ratios CI is valid
            i0 = rng.integers(0, len(g0), size=(n_boot, len(g0)))
            i5 = rng.integers(0, len(g5), size=(n_boot, len(g5)))
            b_m = m5[i5].mean(axis=1) / m0[i0].mean(axis=1)
            b_s = s5[i5].mean(axis=1) / s0[i0].mean(axis=1)
            b_rr = b_m / b_s
            r_m, r_s = m5.mean() / m0.mean(), s5.mean() / s0.mean()
            cm, cs, crr = _ci(b_m), _ci(b_s), _ci(b_rr)
            # winner's standardized lead (best − mean)/SD: K5 − K0 difference (groups with SD>0)
            w0 = g0["winner_z"].to_numpy(float); w5 = g5["winner_z"].to_numpy(float)
            w0 = w0[~np.isnan(w0)]; w5 = w5[~np.isnan(w5)]
            b_w5 = w5[rng.integers(0, len(w5), size=(n_boot, len(w5)))].mean(axis=1)   # K5 drawn first
            b_w0 = w0[rng.integers(0, len(w0), size=(n_boot, len(w0)))].mean(axis=1)
            cw = _ci(b_w5 - b_w0)
            rrows.append({
                "method": method, "train_iter": t,
                "n_groups_K0": int(len(g0)), "n_groups_K5": int(len(g5)),
                "margin_K0": float(m0.mean()), "margin_K5": float(m5.mean()),
                "margin_ratio": float(r_m), "margin_ratio_lo": cm[0], "margin_ratio_hi": cm[1],
                "sd_K0": float(s0.mean()), "sd_K5": float(s5.mean()),
                "sd_ratio": float(r_s), "sd_ratio_lo": cs[0], "sd_ratio_hi": cs[1],
                "ratio_of_ratios": float(r_m / r_s), "ror_lo": crr[0], "ror_hi": crr[1],
                "ror_ci_covers_1": bool(crr[0] <= 1.0 <= crr[1]),
                "margin_over_sd_K0": float(m0.mean() / s0.mean()),
                "margin_over_sd_K5": float(m5.mean() / s5.mean()),
                "winner_z_K0": float(w0.mean()), "winner_z_K5": float(w5.mean()),
                "winner_z_diff": float(w5.mean() - w0.mean()), "winner_z_diff_lo": cw[0], "winner_z_diff_hi": cw[1],
            })
    return pd.DataFrame(rrows)


# ── 4. τ-sensitivity (PTO): yield(τ) raw vs rescaled by the iteration-1 SD ratio ────────────
def tau_sensitivity(source, *, gens: Optional[pd.DataFrame] = None, taus: Sequence[float] = TAUS,
                    ratios: Optional[pd.DataFrame] = None, method: str = "PTO",
                    n_boot: int = _N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """PTO τ-sensitivity (paper table ``dispersion_by_k_tau``): pair yield = share of branch points
    (groups) whose best-worst margin exceeds τ (strict >, as in the trainer; the run used
    τ = :data:`TAU_TRAINER`), per training iteration and ``pooled`` over every iteration of each
    arm (PTO: 1–10 in both).

    ``source`` = a group frame or an arms list; ``ratios`` = :func:`dispersion_ratios` (computed
    here if omitted — the SD ratios come from it). Columns: ``yield_K*_raw`` on the recorded
    margins; ``yield_K0_x_r1`` after multiplying every K=0 margin by r1 = the K=5/K=0 within-group
    SD ratio at train_iter 1 (where both arms are the base policy); ``yield_K5_div_r1`` after
    dividing K=5 margins by the same factor; ``yield_K0_x_r_iter`` uses each iteration's own SD
    ratio. ``share_gap_closed_*`` = (rescaled K=0 yield − raw K=0 yield) / (raw K=5 yield − raw K=0
    yield): the fraction of K=5's yield advantage reproduced by pure rescaling of K=0's spread
    (1.0 = all of it; can exceed 1 when rescaling overshoots; NaN when the raw gap is ~0 —
    iterations 8–10 have a raw gap ≈ 0 or negative because K=5's own spread shrank as its policy
    diverged, so quote the gap in yield points there). Groups with < 2 valid candidate scores
    excluded. NOT a comparison of policies at iterations ≥ 2 (the arms have diverged) — only
    train_iter 1 is same-policy. The rescale factor is attached as ``.attrs["r_iter1"]``.
    """
    G = _as_group_frame(source, gens=gens)
    valid = G[G["n_valid"] >= 2]
    if ratios is None:
        ratios = dispersion_ratios(G, methods=(method,), n_boot=n_boot, seed=seed)
    rm = ratios[ratios["method"] == method]
    r_iter1 = float(rm[rm.train_iter == 1]["sd_ratio"].iloc[0])
    r_by_iter = {int(r.train_iter): float(r.sd_ratio) for r in rm[rm.train_iter != "pooled"].itertuples()}
    a0, a5 = f"{method}_LA0", f"{method}_LA5"
    p0_all = valid[valid.arm == a0]
    p5_all = valid[valid.arm == a5]
    iters = sorted(set(p0_all.train_iter) & set(p5_all.train_iter))
    blocks = [(int(t), p0_all[p0_all.train_iter == t], p5_all[p5_all.train_iter == t]) for t in iters]
    blocks.append(("pooled", p0_all, p5_all))
    trows = []
    for t, g0, g5 in blocks:
        m0 = g0["margin"].to_numpy(float); m5 = g5["margin"].to_numpy(float)
        f_it = g0["train_iter"].map(r_by_iter).to_numpy(float)     # per-iteration factor
        for tau in taus:
            y0 = float((m0 > tau).mean()); y5 = float((m5 > tau).mean())
            y0_r1 = float((m0 * r_iter1 > tau).mean())          # K=0 margins scaled UP by iter-1 factor
            y5_r1 = float((m5 / r_iter1 > tau).mean())          # K=5 margins scaled DOWN by iter-1 factor
            y0_rit = float((m0 * f_it > tau).mean())
            gap = y5 - y0
            trows.append({
                "train_iter": t, "tau": tau,
                "n_groups_K0": int(len(m0)), "n_groups_K5": int(len(m5)),
                "yield_K0_raw": y0, "yield_K5_raw": y5, "gap_K5_minus_K0": gap,
                "yield_K0_x_r1": y0_r1, "gap_after_rescale_K0": y5 - y0_r1,
                "share_gap_closed_r1": (y0_r1 - y0) / gap if abs(gap) > 1e-12 else np.nan,
                "yield_K5_div_r1": y5_r1, "share_gap_closed_K5down": (y5 - y5_r1) / gap if abs(gap) > 1e-12 else np.nan,
                "yield_K0_x_r_iter": y0_rit,
                "share_gap_closed_r_iter": (y0_rit - y0) / gap if abs(gap) > 1e-12 else np.nan,
            })
    tau_df = pd.DataFrame(trows)
    tau_df.attrs["r_iter1"] = r_iter1
    tau_df.attrs["method"] = method
    tau_df.attrs["tau_trainer"] = TAU_TRAINER
    return tau_df


# ── 5. ledger ────────────────────────────────────────────────────────────────────────────────
def _num(v):
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _put(d: dict, key: str, value, *, source: str = "", note: str = "") -> None:
    d[key] = {"value": _num(value), "source": source, "note": note}


def dispersion_numbers(by_iter: pd.DataFrame, ratios: pd.DataFrame, tau: pd.DataFrame,
                       expectation: pd.DataFrame, *, tau_trainer: float = TAU_TRAINER,
                       methods: Sequence[str] = ("PTO", "GRPO"), extra_meta: Optional[dict] = None) -> Dict[str, dict]:
    """The quotable-numbers ledger (the paper's ``analysis/out/dispersion_by_k.json`` ``numbers``
    block): ``{dotted.key: {"value", "source", "note"}}`` for ``exports.save_numbers``.

    Keys: ``expectation.<sd_estimator>``, ``check.grpo_recorded_group_std_vs_ddof0_max_abs_diff``,
    ``by_iter.<arm>.iter<n>``, ``crosscheck.training_signal_yield.mean_margin_iter1``,
    ``ratios.<method>.iter<n|pooled>``, ``tau.rescale_factor_iter1_sd_ratio``,
    ``tau.<method>.iter<n|pooled>.tau<τ>``, ``tau.<method>.headline_tau<τ>``,
    ``headline.<method>``, ``meta``. Sources name the table + row so a reader can re-derive.
    """
    L: Dict[str, dict] = {}
    E0 = expectation[expectation["sd_estimator"] == "ddof=0"].iloc[0]
    exp_rom, exp_wz = float(E0["ratio_of_means"]), float(E0["winner_z_mean"])
    for _, r in expectation.iterrows():
        _put(L, f"expectation.{r['sd_estimator']}",
             {k: float(r[k]) for k in ("E_range_over_sigma", "E_sd_over_sigma", "ratio_of_means",
                                        "mean_of_ratios", "median_of_ratios", "winner_z_mean")},
             source="tables/dispersion_expectation.md",
             note=f"iid N(0,1), M={int(r['m'])}, {int(r['n_groups']):,} simulated groups")
    _put(L, "check.grpo_recorded_group_std_vs_ddof0_max_abs_diff",
         by_iter.attrs.get("grpo_group_std_max_abs_diff"),
         source="internal check (no table)",
         note="GRPO's logged group_std is the ddof=0 SD of the 8 candidate scores; PTO uses the same estimator here")
    ycol = f"yield_tau{tau_trainer:.2f}"
    for _, r in by_iter.iterrows():
        _put(L, f"by_iter.{r['arm']}.iter{int(r['train_iter'])}",
             {k: r[k] for k in ("n_groups", "sd_mean", "sd_median", "margin_mean", "margin_median",
                                "margin_over_sd", "mean_group_ratio", "winner_z",
                                "null_margin_over_sd", "null_winner_z", "frac_sd_zero",
                                "frac_groups_floored", "frac_groups_nan_cand", ycol,
                                "reward_median", "reward_iqr")},
             source=f"tables/dispersion_by_iter.md row arm={r['arm']} train_iter={int(r['train_iter'])}")

    def _mm(arm, ti):
        s = by_iter[(by_iter.arm == arm) & (by_iter.train_iter == ti)]["margin_mean"]
        return float(s.iloc[0]) if len(s) else None
    _put(L, "crosscheck.training_signal_yield.mean_margin_iter1",
         {a: _mm(a, 1) for a in sorted(by_iter["arm"].unique())},
         source="tables/dispersion_by_iter.md vs arms/preference/tables/<judge>/training_signal_yield.md (mean_margin)",
         note="tracked EDA at promotion time: PTO_LA5 0.424, PTO_LA0 0.274, GRPO_LA5 0.546, GRPO_LA0 0.379")
    for _, r in ratios.iterrows():
        _put(L, f"ratios.{r['method']}.iter{r['train_iter']}",
             {k: r[k] for k in ("n_groups_K0", "n_groups_K5", "margin_K0", "margin_K5", "margin_ratio",
                                "margin_ratio_lo", "margin_ratio_hi", "sd_K0", "sd_K5", "sd_ratio",
                                "sd_ratio_lo", "sd_ratio_hi", "ratio_of_ratios", "ror_lo", "ror_hi",
                                "margin_over_sd_K0", "margin_over_sd_K5", "winner_z_K0", "winner_z_K5",
                                "winner_z_diff", "winner_z_diff_lo", "winner_z_diff_hi")},
             source=f"tables/dispersion_ratios.md row method={r['method']} train_iter={r['train_iter']}",
             note=(f"margin_ratio = {r['margin_K5']:.3f}/{r['margin_K0']:.3f} = {r['margin_ratio']:.3f}; "
                   f"sd_ratio = {r['sd_K5']:.3f}/{r['sd_K0']:.3f} = {r['sd_ratio']:.3f}; "
                   f"ratio_of_ratios = {r['margin_ratio']:.3f}/{r['sd_ratio']:.3f} = {r['ratio_of_ratios']:.3f}"))
    # τ block
    tmethod = tau.attrs.get("method", "PTO")
    r_iter1 = tau.attrs.get("r_iter1")
    if r_iter1 is None:
        rm = ratios[(ratios.method == tmethod) & (ratios.train_iter == 1)]
        r_iter1 = float(rm["sd_ratio"].iloc[0]) if len(rm) else None
    _put(L, "tau.rescale_factor_iter1_sd_ratio", r_iter1,
         source=f"tables/dispersion_ratios.md row method={tmethod} train_iter=1 (sd_ratio)",
         note="K=5 mean SD / K=0 mean SD at train_iter 1 (both arms = base policy); the factor applied to K=0 margins")
    for _, r in tau.iterrows():
        _put(L, f"tau.{tmethod}.iter{r['train_iter']}.tau{r['tau']:.2f}",
             {k: r[k] for k in ("n_groups_K0", "n_groups_K5", "yield_K0_raw", "yield_K5_raw", "gap_K5_minus_K0",
                                "yield_K0_x_r1", "share_gap_closed_r1", "yield_K5_div_r1",
                                "share_gap_closed_K5down", "yield_K0_x_r_iter", "share_gap_closed_r_iter")},
             source=f"tables/dispersion_tau.md row train_iter={r['train_iter']} tau={r['tau']:.2f}",
             note=(f"share_gap_closed_r1 = ({r['yield_K0_x_r1']:.3f} − {r['yield_K0_raw']:.3f}) / "
                   f"({r['yield_K5_raw']:.3f} − {r['yield_K0_raw']:.3f})"))
    at_tau = tau[(tau.tau == tau_trainer) & (tau.train_iter != "pooled")].copy()
    if len(at_tau):
        stable = at_tau[at_tau["gap_K5_minus_K0"] > 0.05]
        pooled_row = tau[(tau.tau == tau_trainer) & (tau.train_iter == "pooled")].iloc[0]
        it1 = at_tau[at_tau.train_iter == 1]
        _put(L, f"tau.{tmethod}.headline_tau{tau_trainer:.2f}",
             {"iters_with_raw_gap_gt_0.05": [int(t) for t in stable.train_iter],
              "share_gap_closed_r1_over_those_iters_min": float(stable.share_gap_closed_r1.min()) if len(stable) else None,
              "share_gap_closed_r1_over_those_iters_median": float(stable.share_gap_closed_r1.median()) if len(stable) else None,
              "share_gap_closed_r1_over_those_iters_max": float(stable.share_gap_closed_r1.max()) if len(stable) else None,
              "iter1_gap_raw": float(it1.gap_K5_minus_K0.iloc[0]) if len(it1) else None,
              "iter1_gap_after_rescale_K0": float(it1.gap_after_rescale_K0.iloc[0]) if len(it1) else None,
              "iter1_share_gap_closed_r1": float(it1.share_gap_closed_r1.iloc[0]) if len(it1) else None,
              "pooled_yield_K0_raw": float(pooled_row.yield_K0_raw),
              "pooled_yield_K5_raw": float(pooled_row.yield_K5_raw),
              "pooled_yield_K0_x_r1": float(pooled_row.yield_K0_x_r1),
              "pooled_gap_raw": float(pooled_row.gap_K5_minus_K0),
              "pooled_gap_after_rescale_K0": float(pooled_row.gap_after_rescale_K0),
              "pooled_share_gap_closed_r1": float(pooled_row.share_gap_closed_r1)},
             source=f"tables/dispersion_tau.md rows tau={tau_trainer:.2f}",
             note=("share = (yield_K0_x_r1 − yield_K0_raw)/(yield_K5_raw − yield_K0_raw); iterations 8–10 have a "
                   "raw gap ≈ 0 or negative (K=5's own spread shrank as its policy diverged), so the share is "
                   "undefined there — quote the gap in yield points instead"))
    # headline per method
    for method in methods:
        rr = ratios[ratios.method == method]
        if rr.empty:
            continue
        r1 = rr[rr.train_iter == 1].iloc[0]
        rp = rr[rr.train_iter == "pooled"].iloc[0]
        per = rr[rr.train_iter != "pooled"]
        _put(L, f"headline.{method}",
             {"iter1_margin_ratio": r1.margin_ratio, "iter1_sd_ratio": r1.sd_ratio,
              "iter1_ratio_of_ratios": r1.ratio_of_ratios, "iter1_ror_ci": [r1.ror_lo, r1.ror_hi],
              "pooled_margin_ratio": rp.margin_ratio, "pooled_sd_ratio": rp.sd_ratio,
              "pooled_ratio_of_ratios": rp.ratio_of_ratios, "pooled_ror_ci": [rp.ror_lo, rp.ror_hi],
              "n_iters_ror_ci_covers_1": int(per["ror_ci_covers_1"].sum()), "n_iters": int(len(per)),
              "margin_over_sd_range_K0": [float(per.margin_over_sd_K0.min()), float(per.margin_over_sd_K0.max())],
              "margin_over_sd_range_K5": [float(per.margin_over_sd_K5.min()), float(per.margin_over_sd_K5.max())],
              "iid_expectation": exp_rom, "iid_winner_z": exp_wz},
             source=f"tables/dispersion_ratios.md rows method={method}",
             note=(f"iter1: margin {r1.margin_K5:.3f}/{r1.margin_K0:.3f}={r1.margin_ratio:.3f}, "
                   f"SD {r1.sd_K5:.3f}/{r1.sd_K0:.3f}={r1.sd_ratio:.3f}"))
    meta = {"group_keys": GROUP_KEYS, "sd_estimator": "ddof=0 (population)", "M": M_BRANCHES,
            "tau_trainer": tau_trainer, "taus": sorted(set(float(t) for t in tau["tau"].unique())),
            "arms": sorted(by_iter["arm"].unique()), "grader": GRADER,
            "grpo_la5_censored_at_train_iter": 5,
            "n_candidates_loaded": by_iter.attrs.get("n_candidates_loaded"),
            "n_groups": by_iter.attrs.get("n_groups"),
            "seed_default": BOOT_SEED,
            "promoted_from": "papers/2026_lookahead_pto_grpo/analysis/dispersion_by_k.py (2026-08-18)"}
    if extra_meta:
        meta.update(extra_meta)
    _put(L, "meta", meta, source="module constants")
    return L
