"""faithfulness.py — is the partial-conversation TRAINING reward a faithful proxy for the
full-conversation EVAL, and does K-turn look-ahead make it more faithful?

Promoted 2026-08-18 from the look-ahead paper's generator
``papers/2026_lookahead_pto_grpo/analysis/reward_faithfulness.py`` (its
``tables/reward_faithfulness_*.csv`` + ``analysis/out/reward_faithfulness.json`` are the frozen
fixture these functions reproduce). Renders in ``lookahead/mechanism``. The per-arm reliability
CURVE the tracked EDA already had (``arms/training``, ``reliability_curve``) becomes here TABLES
with cluster-bootstrap CIs, a stated unit, a matched-policy cut, a persona-cooperation cut, and the
proxy-vs-eval level table — under BOTH graders side by side.

UNIT (restated in every table caption)
--------------------------------------
* A *branch row* = one branch point of the training run: a conversation-so-far ("prefix",
  ``n_turns`` utterances therapist+patient combined, ending on a patient turn) plus M=8 (PTO) /
  G=8 (GRPO) therapist completions sampled from the iter-start policy π_{train_iter−1}. Each
  completion was scored AT TRAINING TIME by the training oracle (gpt-4o-mini, mean of Q1+Q2) on
  ``prefix + completion`` (K=0) or ``prefix + completion + 5 simulated turns`` (K=5).
  ``proxy_score`` = the score of the CHOSEN candidate = the arg-max candidate (PTO: the one
  appended to the greedy trunk; GRPO: the recorder marks the arg-max). Read from
  ``generations.jsonl`` — NO new oracle calls
  (:func:`eda_analysis.training.load_branch_reliability` ``which="chosen"``).
* ``n_turns`` = utterances in the prefix BEFORE the scored completion (MCL=12 is the shortest
  cut). The text the oracle actually saw has n_turns+1 (+K) utterances.
* ``eval_score`` = full-conversation Q1Q2 of the eval conversation the prefix was cut from
  (``model_iter_{train_iter−1}``, joined on ``conversation_id == file_index``; the shuffle seed is
  shared, so this is also the same persona). For GRPO the prefix IS a slice of that eval
  conversation at every n_turns; for PTO (greedy trunk) it shares the first MCL=12 utterances and
  afterwards follows the greedy best-of-M continuation, so the two diverge as n_turns grows.
* ``agreement`` (per arm, per n_turns) = the fraction of conversation PAIRS, formed within one
  (arm, eval_iter, n_turns) cell, whose proxy-score ordering matches their eval-score ordering
  (ties dropped), pooled (summed counts) over eval_iters. 0.5 = chance. Point estimates reproduce
  :func:`eda_analysis.stats.rank_agreement_by_nturns` EXACTLY — the paper generator asserted
  bin-for-bin equality (max |dev| = 0.0 on every (arm, n_turns) bin, ``n_pairs`` identical) and
  :func:`check_against_rank_agreement` re-runs that assertion on demand.
* CIs = 95% percentile CLUSTER bootstrap: conversations are resampled with replacement within
  each (arm, eval_iter) model state (the same resample applied to every n_turns bin of that state,
  so pooled-over-bins numbers are cluster-correct too); B=1000. Pairs within a cell are NOT
  independent (each conversation enters n−1 pairs) — CIs come from the conversation-level cluster
  bootstrap, never from ``n_pairs``.
* **K-contrast sign: ``+ => K=0 higher``** (delta = K0 − K5), the paper's convention. NOTE that for
  faithfulness a NEGATIVE delta means look-ahead helped.
* Graders: the proxy is ALWAYS the training oracle's score (it cannot be re-graded). The eval side
  is computed under EVERY grader passed in ``scores_by_judge`` — primary = gpt-4o-mini (the same
  oracle → same-grader faithfulness), held-out = Claude Haiku 4.5 (cross-grader faithfulness).
  **Never averaged across graders.**
* **Support is derived, never assumed**: each arm's branch rows stop at its own last training
  iteration on disk and its eval side at the last state THAT GRADER scored, so support can be
  grader-dependent and is derived per grader (:func:`_series`). GRPO_LA0 is shown both on its full support and
  restricted to GRPO_LA5's — like-for-like by construction, never by a frozen range; the ``iters``
  label on every row states the range actually pooled.

Further caveats (kept from the paper generator)
-----------------------------------------------
* PTO greedy trunks share only the first MCL=12 utterances with the eval conversation; beyond
  n_turns=12 the PTO prefix follows the best-of-M trunk, so PTO's curve mixes cut length with
  trunk divergence (GRPO prefixes are exact slices).
* GRPO branch rows include the policy drifting within an iteration (2 epochs) and ~3–10%
  eval-split groups TRL scores at iteration end (all rows kept); PTO rows come from the frozen
  iter-start policy.
* Wilcoxon over n_turns bins treats correlated bins as observations — descriptive only.
* GRPO_LA5 iteration 1 captured only its second epoch (the run resumed after a crash and the
  recorder flushes once per iteration).

Seeds: the cluster bootstrap is seeded with :data:`eda_analysis.constants.BOOT_SEED` by default;
the paper's fixture used ``seed=0`` (and ``seed + 1 + stratum_index`` per cooperation stratum) —
pass ``seed=0`` to reproduce its CI bounds bit-for-bit. Point estimates do not depend on the seed.

API
---
Build the shared data once — :func:`faithfulness_data(arms, scores_by_judge)` → a
:class:`FaithfulnessData` — then hand it to :func:`faithfulness_curve` (long; :func:`curve_wide`
for the per-grader wide table), :func:`faithfulness_by_iter`, :func:`k_faithfulness_by_iter`,
:func:`matched_policy` (→ per-bin frame + Wilcoxon-over-bins tests), :func:`k_summary`,
:func:`by_cooperation`, :func:`proxy_levels` (→ levels + Spearman/Pearson frame) and
:func:`faithfulness_numbers` (the ledger). Every one of those also accepts
``(arms, scores_by_judge)`` directly and builds the data itself. ``*_display`` helpers produce the
"agreement [lo, hi]" presentational tables the paper saved as ``.md``. NO disk writes; figures live
in :mod:`eda_analysis.plotting.faithfulness`.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from .constants import BOOT_SEED, PRIMARY_JUDGE_TAG, judge_dirname

__all__ = [
    "METRIC", "METHODS", "ARMS", "COARSE", "COOP_LABEL", "COOP_ORDER", "CUTS", "SERIES",
    "PRIMARY_LABEL", "UNIT_NOTE", "SIGN_NOTE", "CENSOR_NOTE", "GRADER_NOTE", "CUT_NOTE", "CAVEATS",
    "judge_display", "fmt_ci", "eval_frame", "AgreementBoot", "delta_ci",
    "FaithfulnessData", "faithfulness_data", "check_against_rank_agreement",
    "faithfulness_curve", "curve_wide", "faithfulness_by_iter", "by_iter_display",
    "k_faithfulness_by_iter", "k_by_iter_display",
    "matched_policy", "matched_policy_display", "k_summary", "k_summary_display",
    "by_cooperation", "by_cooperation_display", "proxy_levels", "faithfulness_numbers",
]

METRIC = "Q1Q2"
METHODS = ["PTO", "GRPO"]
ARMS = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]
COARSE = [("12-20", 12, 20), ("22-34", 22, 34), ("36-50", 36, 50)]
from .constants import COOP_LABEL, COOP_ORDER  # noqa: E402,F401
from .constants import k_of as _k_of_canonical, method_of as _method_of_canonical  # noqa: E402
from .ledger import json_scalar, ledger_entry, round3  # noqa: E402,F401
CUTS = ["train_iter_1", "iters_1-5", "matched_iters"]
#: Curve series SPEC: ``(arm, matched_to)``. Every arm is drawn on its OWN support; a PAIRED arm is
#: drawn a SECOND time restricted to the support the pair SHARES (the like-for-like row). The pair
#: is symmetric — naming it on one arm restricts BOTH — so which of the two is written down here
#: carries no meaning beyond column order. The iteration-range LABEL and the eval_iters subset are
#: BOTH derived per grader by :func:`_series` — they are never written down here. PTO's pair is
#: dormant while its two arms cover the same iterations (no second row is emitted); it exists so
#: the guarantee does not depend on which arm happens to be the censored one.
#:
#: ⚠ This list used to carry the ranges as literals (``("GRPO_LA5", "1-5", None)`` and a
#: ``frozenset({0,1,2,3,4})`` for GRPO_LA0). Two things were wrong with that, and both were silent:
#: the ``None`` on GRPO_LA5 applied NO filter at all, so the row pooled every eval_iter that
#: existed while the label still said ``1-5`` (a false label over correct numbers); and the frozen
#: GRPO_LA0 subset stopped matching GRPO_LA5's real support the moment that arm advanced, so the
#: "like-for-like" row silently compared different iteration sets. An arm's support is a property
#: of the per-judge join (``fd.AB[judge].cells``) — the score lake covers an arm further under one
#: grader than another — so it can only be read off the frame, per grader, every time.
SERIES = [("PTO_LA0", "PTO_LA5"), ("PTO_LA5", None), ("GRPO_LA0", "GRPO_LA5"), ("GRPO_LA5", None)]
PRIMARY_LABEL = judge_dirname(PRIMARY_JUDGE_TAG)          # "gpt-4o-mini"
_N_BOOT = 1000
_MIN_PAIRS = 20

UNIT_NOTE = (
    "Unit: one branch row = one training branch point (prefix of n_turns utterances, therapist+patient, "
    "ending on a patient turn) with its 8 completions sampled by that iteration's policy (PTO: the frozen iter-start "
    "policy; GRPO: the policy as it trains within the iteration over 2 epochs, plus the ~3-10% eval-split groups TRL "
    "scores at iteration end — all rows kept); proxy_score = training-oracle (gpt-4o-mini, "
    "Q1+Q2 mean) score of the CHOSEN (arg-max) completion on prefix+completion (K=0) or "
    "prefix+completion+5 simulated turns (K=5, i.e. the K-extended score). eval_score = full-conversation "
    "Q1Q2 of the eval conversation the prefix was cut from (model_iter_{train_iter-1}, same file_index / "
    "persona; GRPO prefixes are exact slices of it, PTO greedy trunks share its first MCL=12 utterances then "
    "diverge). agreement = fraction of conversation pairs within one (arm, eval_iter, n_turns) cell whose "
    "proxy ordering matches their eval ordering (ties dropped), counts pooled over eval_iters; 0.5 = chance. "
    "95% CI = cluster bootstrap over conversations within each (arm, eval_iter) model state (B=1000). "
    "n_turns = utterances BEFORE the scored completion (MCL=12 = shortest cut)."
)
SIGN_NOTE = "K-contrast sign: delta = K0 - K5 (+ => K=0 higher; a NEGATIVE delta means look-ahead is more faithful)."
# A LEGEND, not an assertion (see the note on compute.CENSOR_NOTE — this said "GRPO_LA5 is
# right-censored" long after that arm finished at iteration 10).
CENSOR_NOTE = ("Each arm's branch rows stop at its own last training iteration on disk and its eval side at "
               "the last state that grader scored, so support can be grader-dependent - the `iters` "
               "label on each row states the train_iter range actually pooled.")
GRADER_NOTE = ("Proxy = the training oracle (gpt-4o-mini) by construction; eval side under the grader named in "
               "the table (primary = gpt-4o-mini, held-out = Claude Haiku 4.5). Never averaged across graders.")
CUT_NOTE = ("Cuts: cut=train_iter_1 = MATCHED POLICY — both K arms of a method branch from the SAME base policy pi_0 "
            "(eval side = that arm's independent base draw, model_iter_0), so K=0 vs K=5 is free of policy divergence; "
            "cut=iters_1-5 pools train_iter 1..5 - a FIXED early window (it was GRPO_LA5's full support when the "
            "cut was defined; that arm has since advanced, so read it as an early-iterations cut, not a censoring "
            "boundary), policies have diverged; cut=matched_iters pools every train_iter present in BOTH K arms of "
            "the method, derived from the data (the train_iters column of each row says which).")
CAVEATS = [
    "proxy_score is the training oracle's score by construction; it cannot be re-graded, so the held-out-judge tables change only the EVAL side.",
    "PTO greedy trunks share only the first MCL=12 utterances with the eval conversation; beyond n_turns=12 the PTO prefix follows the best-of-M trunk, so PTO's curve mixes cut length with trunk divergence (GRPO prefixes are exact slices).",
    "GRPO branch rows include the policy drifting within an iteration (2 epochs) and ~3-10% eval-split groups scored at iteration end; PTO rows come from the frozen iter-start policy.",
    "K reads use each arm's own-support series / cuts, and where the two K arms' supports differ the longer arm is "
    "drawn a second time restricted to exactly the shorter one's support (derived per grader, not a frozen range).",
    "Pairs within a cell are not independent (each conversation enters n-1 pairs) — CIs come from the conversation-level cluster bootstrap, not from n_pairs.",
    "Wilcoxon over n_turns bins treats correlated bins as observations — descriptive only.",
    "GRPO_LA5 iteration 1 captured only its second epoch (the run resumed after a crash and the recorder flushes once per iteration).",
]


def judge_display(label: str) -> str:
    """Long grader label for titles: the primary → ``training oracle (gpt-4o-mini)``, anything
    else → ``held-out judge (<label>)`` (``claude-haiku-4-5`` spelled out as Claude Haiku 4.5)."""
    if label in ("", PRIMARY_LABEL, PRIMARY_JUDGE_TAG):
        return f"training oracle ({PRIMARY_LABEL})"
    pretty = {"claude-haiku-4-5": "Claude Haiku 4.5"}.get(label, label)
    return f"held-out judge ({pretty})"


def fmt_ci(a, lo, hi, nd=3) -> str:
    """``"0.865 [0.851, 0.877]"``; empty string for NaN."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return ""
    return f"{a:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _method_of(arm: str) -> str:
    return arm.split("_")[0]


def _k_of(arm: str) -> int:
    """Re-export of :func:`eda_analysis.constants.k_of`."""
    return _k_of_canonical(arm)


# ── data ─────────────────────────────────────────────────────────────────────

def eval_frame(scores_long: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """(arm, eval_iter, conversation_id) → ``eval_score`` (+ ``model, persona_id, coop``) from a
    ``scores_long`` frame. Mirrors the join in :func:`stats.rank_agreement_by_nturns`
    (``iteration`` → ``eval_iter``, ``file_index`` → ``conversation_id``)."""
    d = scores_long[scores_long["questionnaire"] == metric]
    d = d.rename(columns={"iteration": "eval_iter", "file_index": "conversation_id"})
    agg = dict(eval_score=("score", "mean"), model=("model", "first"))
    if "persona_id" in d.columns:
        agg["persona_id"] = ("persona_id", "first")
    if "cooperation_level" in d.columns:
        agg["coop_raw"] = ("cooperation_level", "first")
    ev = d.groupby(["arm", "eval_iter", "conversation_id"], as_index=False).agg(**agg)
    if "persona_id" not in ev.columns:
        ev["persona_id"] = np.nan
    ev["coop"] = ev["coop_raw"].map(COOP_LABEL) if "coop_raw" in ev.columns else np.nan
    return ev


class AgreementBoot:
    """Pairwise sign-agreement between proxy and eval with a conversation-level cluster bootstrap.

    ``df`` columns: arm, eval_iter, conversation_id, n_turns, proxy_score, eval_score.
    Cells are keyed (arm, eval_iter, n_turns); each holds the point counts (conc, tot) over pairs
    and the B bootstrap replicate counts. Any aggregate = sum of counts over a set of cells, so
    per-bin, per-iteration and overall numbers all come from the same replicates.
    """

    def __init__(self, df: pd.DataFrame, *, B: int = _N_BOOT, seed: int = BOOT_SEED):
        self.B = B
        self.cells = {}
        rng = np.random.default_rng(seed)
        for (arm, ei), g in df.groupby(["arm", "eval_iter"], sort=True):
            convs = np.array(sorted(g["conversation_id"].unique()))
            n = len(convs)
            if n < 2:
                continue
            pos = {c: i for i, c in enumerate(convs)}
            idx = rng.integers(0, n, size=(B, n))
            iu0, iu1 = np.triu_indices(n, 1)
            ev = g.groupby("conversation_id")["eval_score"].first()
            e_full = np.full(n, np.nan)
            e_full[[pos[c] for c in ev.index]] = ev.values
            for nt, gg in g.groupby("n_turns", sort=True):
                pm = gg.groupby("conversation_id")["proxy_score"].mean()
                p = np.full(n, np.nan)
                p[[pos[c] for c in pm.index]] = pm.values
                e = e_full.copy()
                e[np.isnan(p)] = np.nan
                M = np.sign(p[:, None] - p[None, :]) * np.sign(e[:, None] - e[None, :])
                M = np.nan_to_num(M, nan=0.0).astype(np.int8)
                pairs = M[iu0, iu1]
                conc, tot = int((pairs > 0).sum()), int((pairs != 0).sum())
                Mb = M[idx[:, iu0], idx[:, iu1]]              # (B, n_pairs) resampled pairs
                self.cells[(arm, int(ei), int(nt))] = dict(
                    conc=conc, tot=tot, n_convs=int((~np.isnan(p)).sum()),
                    bconc=(Mb > 0).sum(1).astype(np.int64), btot=(Mb != 0).sum(1).astype(np.int64))

    def keys(self, arm=None, eval_iters=None, nt_lo=None, nt_hi=None):
        out = []
        for (a, ei, nt) in self.cells:
            if arm is not None and a != arm:
                continue
            if eval_iters is not None and ei not in eval_iters:
                continue
            if nt_lo is not None and nt < nt_lo:
                continue
            if nt_hi is not None and nt > nt_hi:
                continue
            out.append((a, ei, nt))
        return out

    def agg(self, keys) -> dict:
        keys = list(keys)
        if not keys:
            return dict(agreement=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_pairs=0, n_cells=0,
                        n_convs_mean=np.nan, boots=np.full(self.B, np.nan))
        conc = sum(self.cells[k]["conc"] for k in keys)
        tot = sum(self.cells[k]["tot"] for k in keys)
        bconc = np.sum([self.cells[k]["bconc"] for k in keys], axis=0)
        btot = np.sum([self.cells[k]["btot"] for k in keys], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = np.where(btot > 0, bconc / np.maximum(btot, 1), np.nan)
        agr = conc / tot if tot > 0 else np.nan
        lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan))
        return dict(agreement=agr, ci_lo=float(lo), ci_hi=float(hi), n_pairs=int(tot), n_cells=len(keys),
                    n_convs_mean=float(np.mean([self.cells[k]["n_convs"] for k in keys])), boots=boots)


def delta_ci(bootsA, bootsB, a, b) -> dict:
    """delta = a − b with a percentile CI from independent replicate arrays."""
    d = bootsA - bootsB
    if not np.isfinite(d).any():
        return dict(delta=np.nan, d_lo=np.nan, d_hi=np.nan)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return dict(delta=a - b, d_lo=float(lo), d_hi=float(hi))


class FaithfulnessData:
    """The shared inputs every table reads: branch rows (``BR``), the per-grader eval frames
    (``EV``), the joined per-grader frames (``DF``) and the per-grader :class:`AgreementBoot`
    (``AB``). ``judges`` lists the grader labels (primary first); ``primary`` names the primary
    label. Built by :func:`faithfulness_data`; the K-contrast block is memoised on it."""

    def __init__(self, BR: pd.DataFrame, scores_by_judge: Dict[str, pd.DataFrame], *,
                 metric: str = METRIC, B: int = _N_BOOT, seed: int = BOOT_SEED,
                 arm_labels: Optional[Sequence[str]] = None):
        self.metric, self.B, self.seed = metric, B, seed
        labels = list(scores_by_judge)
        prim = [j for j in labels if j in ("", PRIMARY_LABEL, PRIMARY_JUDGE_TAG)]
        if not prim:
            raise ValueError(f"scores_by_judge must contain the primary grader {PRIMARY_LABEL!r}; got {labels}")
        self.primary = prim[0]
        self.judges = [self.primary] + [j for j in labels if j != self.primary]
        self.arm_labels = list(arm_labels) if arm_labels is not None else [a for a in ARMS if a in set(BR["arm"])]
        self.BR = BR[BR["arm"].isin(self.arm_labels)].reset_index(drop=True)
        self.EV = {j: eval_frame(scores_by_judge[j], metric) for j in self.judges}
        self.DF = {j: self.BR.merge(self.EV[j][["arm", "eval_iter", "conversation_id", "eval_score", "persona_id", "coop"]],
                                    on=["arm", "eval_iter", "conversation_id"], how="inner") for j in self.judges}
        self.AB = {j: AgreementBoot(self.DF[j][["arm", "eval_iter", "conversation_id", "n_turns", "proxy_score", "eval_score"]],
                                    B=B, seed=seed) for j in self.judges}
        self.all_bins = sorted(int(b) for b in self.BR["n_turns"].unique())
        self.train_iters = {a: sorted(int(t) for t in self.BR.loc[self.BR["arm"] == a, "train_iter"].unique())
                            for a in self.arm_labels}
        self._k_block = None

    # ── helpers ──
    @property
    def heldout(self) -> List[str]:
        return [j for j in self.judges if j != self.primary]

    def suffix(self, judge: str) -> str:
        """Column suffix for a grader in wide-by-grader frames: primary → ``""``; the only held-out
        grader → ``"_heldout"`` (the paper's shape); several held-out graders → ``"_<label>"``."""
        if judge == self.primary:
            return ""
        return "_heldout" if len(self.heldout) == 1 else f"_{judge}"

    def cut_iters(self, method: str, cut: str) -> set:
        """eval_iters used by a cut: ``train_iter_1`` → ``{0}``; ``iters_1-5`` → the fixed early
        window ``{0..4}``; ``matched_iters`` → every train_iter present in BOTH K arms of the
        method, derived from the branch rows (never a written-down range).

        ⚠ ``iters_1-5`` is a FIXED window, kept as-is so its numbers stay comparable with the
        frozen paper fixture. It was GRPO_LA5's full support when it was named; that arm has since
        advanced, so it no longer marks a censoring boundary — it is just an early-iterations cut.
        """
        both = set(self.train_iters.get(f"{method}_LA0", [])) & set(self.train_iters.get(f"{method}_LA5", []))
        if cut == "train_iter_1":
            ti = {1}
        elif cut == "iters_1-5":
            ti = {t for t in both if t <= 5}
        else:
            ti = both
        return {int(t) - 1 for t in ti}

    def methods(self) -> List[str]:
        return [m for m in METHODS if f"{m}_LA0" in self.arm_labels and f"{m}_LA5" in self.arm_labels]


def faithfulness_data(arms=None, scores_by_judge: Optional[Dict[str, pd.DataFrame]] = None, *,
                      BR: Optional[pd.DataFrame] = None, metric: str = METRIC, B: int = _N_BOOT,
                      seed: int = BOOT_SEED, arm_labels: Optional[Sequence[str]] = None) -> FaithfulnessData:
    """Load the branch rows (``training.load_branch_reliability(arms, which="chosen")`` — ~3 min,
    re-reads every ``generations.jsonl``; pass ``BR`` to reuse a frame) and join them to each
    grader's eval Q1Q2 in ``scores_by_judge`` (``{judge_label: scores_long}``, primary included —
    e.g. from ``eda_analysis.scores_by_judge``). Returns the :class:`FaithfulnessData` every
    table function consumes."""
    if isinstance(arms, FaithfulnessData):
        return arms
    if scores_by_judge is None:
        raise ValueError("scores_by_judge ({judge_label: scores_long}) is required")
    if BR is None:
        from .training import load_branch_reliability
        BR = load_branch_reliability(arms, which="chosen")
    return FaithfulnessData(BR, scores_by_judge, metric=metric, B=B, seed=seed, arm_labels=arm_labels)


def _as_data(data, scores_by_judge=None, **kw) -> FaithfulnessData:
    return data if isinstance(data, FaithfulnessData) else faithfulness_data(data, scores_by_judge, **kw)


def check_against_rank_agreement(fd: FaithfulnessData, scores_long_primary: pd.DataFrame,
                                 *, min_pairs: int = _MIN_PAIRS) -> float:
    """Assert the point estimates reproduce :func:`stats.rank_agreement_by_nturns` bin-for-bin
    (agreement to 1e-9, ``n_pairs`` identical) on the primary grader; returns the max |dev|."""
    from . import stats as est
    ref = est.rank_agreement_by_nturns(fd.BR, scores_long_primary, metric=fd.metric, min_pairs=min_pairs)
    ab = fd.AB[fd.primary]
    max_dev = 0.0
    for _, r in ref.iterrows():
        mine = ab.agg(ab.keys(arm=r["arm"], nt_lo=int(r["n_turns"]), nt_hi=int(r["n_turns"])))
        max_dev = max(max_dev, abs(mine["agreement"] - r["agreement"]))
        if mine["n_pairs"] != r["n_pairs"]:
            raise AssertionError((r["arm"], r["n_turns"], mine["n_pairs"], r["n_pairs"]))
    if not max_dev < 1e-9:
        raise AssertionError(f"point estimates deviate from stats.rank_agreement_by_nturns: {max_dev}")
    return max_dev


# ── 1) curve ─────────────────────────────────────────────────────────────────

def _arm_eval_iters(fd: FaithfulnessData, judge: str, arm: str) -> frozenset:
    """The eval_iters ``arm`` actually HAS under ``judge`` — read off that grader's bootstrap cells
    (i.e. off the branch-row-to-eval join), never assumed. Grader-dependent by construction: the
    score lake can cover an arm further under one judge than another."""
    return frozenset(int(ei) for (a, ei, _nt) in fd.AB[judge].cells if a == arm)


def _iters_label(eval_iters) -> str:
    """``{0..5}`` → ``"1-6"`` (train_iter = eval_iter + 1); a gap → ``"1,2,5"``; empty → ``""``."""
    ts = sorted(int(e) + 1 for e in eval_iters)
    if not ts:
        return ""
    return f"{ts[0]}-{ts[-1]}" if ts == list(range(ts[0], ts[-1] + 1)) else ",".join(str(t) for t in ts)


def _matched_partners(arm: str) -> List[str]:
    """The arms ``arm`` must be shown against on a SHARED support: the partner it names in
    :data:`SERIES` **and** every arm that names it. Symmetric on purpose — see :func:`_series`."""
    return list(dict.fromkeys([m for a, m in SERIES if a == arm and m]
                              + [a for a, m in SERIES if m == arm]))


def _series(fd: FaithfulnessData, judge: str):
    """``[(arm, iters_label, eval_iters|None)]`` for ONE grader, derived from that grader's join.

    Each arm appears on its own support (no filter). For a matched pair (A, B), BOTH arms appear
    again restricted to ``A & B`` whenever that is narrower than the arm's own support — so the
    pair's two rows cover exactly ``A & B`` on every frame, whichever way the supports lie.

    ⚠ The restriction has to be applied to both sides. Restricting only the arm that NAMES the
    partner is equivalent **only while the partner's support is a subset of it** — the shape the
    live data happens to have. A hole anywhere in A (a missing ``generations.jsonl``, an
    ``AgreementBoot`` cell dropped for n < 2 conversations, one model state not yet in that
    grader's lake), or B simply running ahead of A, and the one-sided version silently pools
    different iteration sets under a row labelled "like-for-like".
    """
    out, seen = [], set()
    own = {}
    for arm, _m in SERIES:
        if arm in fd.arm_labels:
            s = _arm_eval_iters(fd, judge, arm)
            if s:
                own[arm] = s
    for arm, _m in SERIES:
        if arm not in own:
            continue
        out.append((arm, _iters_label(own[arm]), None))
        for partner in _matched_partners(arm):
            if partner not in own:
                continue
            sub = own[arm] & own[partner]
            if sub and sub != own[arm] and (arm, sub) not in seen:
                seen.add((arm, sub))
                out.append((arm, _iters_label(sub), frozenset(sub)))
    return out


def faithfulness_curve(data, scores_by_judge=None, *, min_pairs: int = _MIN_PAIRS, **kw) -> pd.DataFrame:
    """Long form of the reward-faithfulness curve (paper ``reward_faithfulness_curve_long``):
    sign-agreement between the training proxy and the full-conversation eval Q1Q2 per (eval-side
    ``judge``, ``arm``, ``iters`` range, ``n_turns`` bin), pooled over training iterations; rows
    ``n_turns='all'`` pool every bin, ``'12-20'/'22-34'/'36-50'`` pool the coarse ranges. Columns:
    ``judge, arm, iters, method, K, n_turns (str), agreement, ci_lo, ci_hi, n_pairs, n_iters,
    n_convs_mean``. Bins with < ``min_pairs`` pairs are dropped. GRPO_LA0 appears on its full
    support AND restricted to GRPO_LA5's support (like-for-like with the censored arm) — both the
    ``iters`` label and the restriction are derived PER GRADER off that grader's join, so the two
    matched rows always cover the same iterations and no label can go stale.
    Use :func:`curve_wide` for the per-grader wide table (``reward_faithfulness_curve`` /
    ``_curve_heldout``)."""
    fd = _as_data(data, scores_by_judge, **kw)
    rows = []
    for j in fd.judges:
        ab = fd.AB[j]
        for arm, iters_lab, eis in _series(fd, j):
            for nt in fd.all_bins + ["all"] + [c[0] for c in COARSE]:
                if nt == "all":
                    ks = ab.keys(arm=arm, eval_iters=eis)
                elif isinstance(nt, str):
                    lo, hi = [c for c in COARSE if c[0] == nt][0][1:]
                    ks = ab.keys(arm=arm, eval_iters=eis, nt_lo=lo, nt_hi=hi)
                else:
                    ks = ab.keys(arm=arm, eval_iters=eis, nt_lo=nt, nt_hi=nt)
                if not ks:
                    continue
                r = ab.agg(ks)
                if r["n_pairs"] < min_pairs:
                    continue
                rows.append(dict(judge=j, arm=arm, iters=iters_lab, method=_method_of(arm), K=_k_of(arm),
                                 n_turns=str(nt), agreement=r["agreement"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                 n_pairs=r["n_pairs"], n_iters=len({k[1] for k in ks}),
                                 n_convs_mean=r["n_convs_mean"]))
    return pd.DataFrame(rows)


def _label_rank(lab: str) -> int:
    """Sort key putting the WIDER iteration range first (``"1-10"`` before ``"1-6"``). String sort
    would not: ``"1-10" < "1-6"`` happens to be right and ``"1-9" < "1-10"`` is wrong."""
    ns = [int(x) for x in re.findall(r"\d+", str(lab))]
    return -(max(ns) - min(ns)) if ns else 0


def curve_wide(curve: pd.DataFrame, judge: str, *, bins: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """One grader's curve as the paper's wide table: rows = n_turns bins (+ coarse ranges + all),
    columns ``"<arm> (iters <range>)"`` = ``agreement [lo, hi]`` and ``"... pairs"``."""
    d = curve[curve["judge"] == judge]
    if bins is None:
        bins = sorted({int(b) for b in d["n_turns"] if str(b).isdigit()})
    # Derived off the frame, not off SERIES: the iteration ranges are per-grader labels that
    # faithfulness_curve computed. SERIES supplies only the arm ORDER; within an arm the wider
    # support comes first (its own support, then the like-for-like restriction).
    order = {a: i for i, (a, _m) in enumerate(SERIES)}
    series = sorted({(str(a), str(l)) for a, l in zip(d["arm"], d["iters"])},
                    key=lambda t: (order.get(t[0], len(order)), _label_rank(t[1]), t[1]))
    rows = []
    for nt in [str(b) for b in bins] + [c[0] for c in COARSE] + ["all"]:
        row = {"n_turns": nt}
        for arm, iters_lab in series:
            col = f"{arm} (iters {iters_lab})"
            s = d[(d["arm"] == arm) & (d["iters"] == iters_lab) & (d["n_turns"] == nt)]
            if s.empty:
                row[col] = ""; row[f"{col} pairs"] = ""
            else:
                s = s.iloc[0]
                row[col] = fmt_ci(s["agreement"], s["ci_lo"], s["ci_hi"])
                row[f"{col} pairs"] = int(s["n_pairs"])
        rows.append(row)
    return pd.DataFrame(rows)


# ── 1b) per train_iter ───────────────────────────────────────────────────────

def faithfulness_by_iter(data, scores_by_judge=None, **kw) -> pd.DataFrame:
    """Reward faithfulness per (arm, train_iter), all n_turns bins pooled + the three coarse ranges
    (paper ``reward_faithfulness_curve_by_iter_long``). Primary-grader columns ``agreement, ci_lo,
    ci_hi, n_pairs, n_convs, n_bins, agr_<range>/lo_/hi_/pairs_<range>``; each held-out grader adds
    ``agreement<sfx>, ci_lo<sfx>, ci_hi<sfx>, n_pairs<sfx>`` (``sfx`` = ``_heldout`` for a single
    held-out grader). train_iter n branches from policy π_{n−1}, whose eval conversations are
    ``model_iter_{n−1}`` = ``eval_iter``; train_iter 1 = the BASE policy for every arm (a
    matched-policy row)."""
    fd = _as_data(data, scores_by_judge, **kw)
    rows = []
    for arm in fd.arm_labels:
        for ti in fd.train_iters[arm]:
            ei = int(ti) - 1
            row = dict(arm=arm, method=_method_of(arm), K=_k_of(arm), train_iter=int(ti), eval_iter=ei)
            for j in fd.judges:
                ab = fd.AB[j]
                r = ab.agg(ab.keys(arm=arm, eval_iters={ei}))
                tag = fd.suffix(j)
                row[f"agreement{tag}"] = r["agreement"]; row[f"ci_lo{tag}"] = r["ci_lo"]; row[f"ci_hi{tag}"] = r["ci_hi"]
                row[f"n_pairs{tag}"] = r["n_pairs"]
                if j == fd.primary:
                    d = fd.DF[j]
                    row["n_convs"] = int(d[(d["arm"] == arm) & (d["eval_iter"] == ei)]["conversation_id"].nunique())
                    row["n_bins"] = r["n_cells"]
                    for lab, lo, hi in COARSE:
                        rc = ab.agg(ab.keys(arm=arm, eval_iters={ei}, nt_lo=lo, nt_hi=hi))
                        row[f"agr_{lab}"] = rc["agreement"]; row[f"lo_{lab}"] = rc["ci_lo"]; row[f"hi_{lab}"] = rc["ci_hi"]
                        row[f"pairs_{lab}"] = rc["n_pairs"]
            rows.append(row)
    return pd.DataFrame(rows)


def by_iter_display(by_iter: pd.DataFrame, *, min_pairs: int = _MIN_PAIRS) -> pd.DataFrame:
    """The paper's ``reward_faithfulness_curve_by_iter`` presentation: ``agreement [CI]`` per grader
    + coarse-range cells (blank when < ``min_pairs`` pairs)."""
    md = by_iter.copy()
    out_cols = ["arm", "train_iter", "eval_iter", "n_convs", "n_pairs"]
    md["agreement [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(md["agreement"], md["ci_lo"], md["ci_hi"])]
    out_cols.append("agreement [CI]")
    for lab, _, _ in COARSE:
        md[f"agr {lab} [CI]"] = [fmt_ci(a, l, h) if n >= min_pairs else "" for a, l, h, n in
                                 zip(md[f"agr_{lab}"], md[f"lo_{lab}"], md[f"hi_{lab}"], md[f"pairs_{lab}"])]
        out_cols.append(f"agr {lab} [CI]")
    for c in [c for c in md.columns if c.startswith("agreement_")]:
        sfx = c[len("agreement"):]
        md[f"agreement{sfx} [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(md[c], md[f"ci_lo{sfx}"], md[f"ci_hi{sfx}"])]
        out_cols.append(f"agreement{sfx} [CI]")
    return md[out_cols]


def k_faithfulness_by_iter(data, scores_by_judge=None, **kw) -> pd.DataFrame:
    """K=0 vs K=5 faithfulness per training iteration (all bins pooled), per method and eval-side
    grader (paper ``reward_faithfulness_k_by_iter``). Columns ``judge, method, train_iter,
    eval_iter, agr_K0, agr_K5, delta_K0_minus_K5, d_lo, d_hi, pairs_K0, pairs_K5``. Sign: + ⇒ K=0
    higher (a NEGATIVE delta = look-ahead more faithful). CI = percentile of the difference of
    independent cluster-bootstrap replicates (the two arms are different conversation draws). Only
    train_iter 1 samples the SAME policy in both K arms; later rows compare diverged policies."""
    fd = _as_data(data, scores_by_judge, **kw)
    rows = []
    for j in fd.judges:
        ab = fd.AB[j]
        for m in fd.methods():
            for ti in fd.train_iters[f"{m}_LA5"]:
                ei = int(ti) - 1
                r0, r5 = ab.agg(ab.keys(arm=f"{m}_LA0", eval_iters={ei})), ab.agg(ab.keys(arm=f"{m}_LA5", eval_iters={ei}))
                d = delta_ci(r0["boots"], r5["boots"], r0["agreement"], r5["agreement"])
                rows.append(dict(judge=j, method=m, train_iter=int(ti), eval_iter=ei,
                                 agr_K0=r0["agreement"], agr_K5=r5["agreement"], delta_K0_minus_K5=d["delta"],
                                 d_lo=d["d_lo"], d_hi=d["d_hi"], pairs_K0=r0["n_pairs"], pairs_K5=r5["n_pairs"]))
    return pd.DataFrame(rows)


def k_by_iter_display(dk: pd.DataFrame) -> pd.DataFrame:
    """``reward_faithfulness_k_by_iter`` presentation (delta with ``[CI]``)."""
    d = dk.copy()
    d["delta_K0_minus_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(d["delta_K0_minus_K5"], d["d_lo"], d["d_hi"])]
    return d[["judge", "method", "train_iter", "eval_iter", "agr_K0", "agr_K5", "delta_K0_minus_K5 [CI]", "pairs_K0", "pairs_K5"]]


# ── 2) K contrast: matched-policy cut + pooled cuts ──────────────────────────

def _k_block(fd: FaithfulnessData, min_pairs: int = _MIN_PAIRS):
    """The per-bin K contrast under three iteration cuts, the Wilcoxon-over-bins tests and the
    pooled summary — one pass, memoised on ``fd``."""
    if fd._k_block is not None:
        return fd._k_block
    mp_rows, test_rows, sum_rows = [], [], []
    for j in fd.judges:
        ab = fd.AB[j]
        for m in fd.methods():
            a0, a5 = f"{m}_LA0", f"{m}_LA5"
            for cut in CUTS:
                eis = fd.cut_iters(m, cut)
                deltas = []
                for nt in fd.all_bins + [c[0] for c in COARSE] + ["all"]:
                    if nt == "all":
                        lo, hi = None, None
                    elif isinstance(nt, str):
                        lo, hi = [c for c in COARSE if c[0] == nt][0][1:]
                    else:
                        lo, hi = nt, nt
                    k0 = ab.keys(arm=a0, eval_iters=eis, nt_lo=lo, nt_hi=hi)
                    k5 = ab.keys(arm=a5, eval_iters=eis, nt_lo=lo, nt_hi=hi)
                    if not k0 or not k5:
                        continue
                    r0, r5 = ab.agg(k0), ab.agg(k5)
                    if r0["n_pairs"] < min_pairs or r5["n_pairs"] < min_pairs:
                        continue
                    d = delta_ci(r0["boots"], r5["boots"], r0["agreement"], r5["agreement"])
                    mp_rows.append(dict(judge=j, method=m, cut=cut, n_turns=str(nt),
                                        agr_K0=r0["agreement"], K0_lo=r0["ci_lo"], K0_hi=r0["ci_hi"], pairs_K0=r0["n_pairs"],
                                        agr_K5=r5["agreement"], K5_lo=r5["ci_lo"], K5_hi=r5["ci_hi"], pairs_K5=r5["n_pairs"],
                                        delta_K0_minus_K5=d["delta"], d_lo=d["d_lo"], d_hi=d["d_hi"]))
                    if not isinstance(nt, str):
                        deltas.append(d["delta"])
                deltas = np.asarray(deltas, float)
                if len(deltas) >= 3 and np.any(deltas != 0):
                    w = sps.wilcoxon(deltas, zero_method="wilcox")
                    wstat, wp = float(w.statistic), float(w.pvalue)
                else:
                    wstat, wp = np.nan, np.nan
                test_rows.append(dict(judge=j, method=m, cut=cut, n_bins=len(deltas),
                                      bins_K5_more_faithful=int((deltas < 0).sum()), bins_K0_more_faithful=int((deltas > 0).sum()),
                                      mean_delta=float(deltas.mean()) if len(deltas) else np.nan,
                                      median_delta=float(np.median(deltas)) if len(deltas) else np.nan,
                                      wilcoxon_W=wstat, wilcoxon_p=wp))
                # summary row: pooled-pairs delta (all bins) + equal-weight-per-iteration delta
                r0, r5 = ab.agg(ab.keys(arm=a0, eval_iters=eis)), ab.agg(ab.keys(arm=a5, eval_iters=eis))
                dp = delta_ci(r0["boots"], r5["boots"], r0["agreement"], r5["agreement"])
                per_it, per_it_boots = [], []
                for ei in sorted(eis):
                    q0, q5 = ab.agg(ab.keys(arm=a0, eval_iters={ei})), ab.agg(ab.keys(arm=a5, eval_iters={ei}))
                    if q0["n_pairs"] < min_pairs or q5["n_pairs"] < min_pairs:
                        continue
                    per_it.append(q0["agreement"] - q5["agreement"])
                    per_it_boots.append(q0["boots"] - q5["boots"])
                per_it = np.asarray(per_it, float)
                if len(per_it):
                    mb = np.nanmean(np.vstack(per_it_boots), axis=0)
                    it_lo, it_hi = np.nanpercentile(mb, [2.5, 97.5])
                else:
                    it_lo = it_hi = np.nan
                if len(per_it) >= 5 and np.any(per_it != 0):
                    wi_p = float(sps.wilcoxon(per_it, zero_method="wilcox").pvalue)
                else:
                    wi_p = np.nan
                sum_rows.append(dict(judge=j, method=m, cut=cut, n_iters=len(per_it),
                                     train_iters=",".join(str(e + 1) for e in sorted(eis)),
                                     agr_K0=r0["agreement"], K0_lo=r0["ci_lo"], K0_hi=r0["ci_hi"], pairs_K0=r0["n_pairs"],
                                     agr_K5=r5["agreement"], K5_lo=r5["ci_lo"], K5_hi=r5["ci_hi"], pairs_K5=r5["n_pairs"],
                                     delta_pooled=dp["delta"], dp_lo=dp["d_lo"], dp_hi=dp["d_hi"],
                                     delta_iter_mean=float(per_it.mean()) if len(per_it) else np.nan, di_lo=float(it_lo), di_hi=float(it_hi),
                                     iters_K5_more_faithful=int((per_it < 0).sum()), iters_K0_more_faithful=int((per_it > 0).sum()),
                                     wilcoxon_over_iters_p=wi_p))
    fd._k_block = (pd.DataFrame(mp_rows), pd.DataFrame(test_rows), pd.DataFrame(sum_rows))
    return fd._k_block


def matched_policy(data, scores_by_judge=None, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """K=0 vs K=5 reward faithfulness per n_turns bin under three iteration cuts
    (paper ``reward_faithfulness_matched_policy_long`` + ``_matched_policy_tests``).

    Returns ``(per_bin, tests)``. ``per_bin`` columns: ``judge, method, cut, n_turns (str; digits,
    the coarse ranges and 'all'), agr_K0, K0_lo, K0_hi, pairs_K0, agr_K5, K5_lo, K5_hi, pairs_K5,
    delta_K0_minus_K5, d_lo, d_hi``. Cuts: ``train_iter_1`` = MATCHED POLICY (both K arms branch from
    the SAME base policy π_0; eval side = that arm's independent base draw), ``iters_1-5`` (a FIXED
    early window, NOT a censoring boundary - see :meth:`FaithfulnessData.cut_iters`; diverged
    policies), ``matched_iters`` (every train_iter in BOTH K arms). Sign:
    + ⇒ K=0 higher. Per-bin CI = percentile of the difference of independent cluster-bootstrap
    replicates. Bins with < 20 pairs in either arm are dropped.
    ``tests`` = Wilcoxon signed-rank over the numeric n_turns BINS (paired by bin) of
    delta = agreement(K0) − agreement(K5): ``n_bins, bins_K5_more_faithful, bins_K0_more_faithful,
    mean_delta, median_delta, wilcoxon_W, wilcoxon_p``. Bins are NOT independent observations (the
    same conversations feed neighbouring bins), so read p as descriptive; the per-bin CIs and
    :func:`k_summary` carry the inference."""
    fd = _as_data(data, scores_by_judge, **kw)
    mp, tests, _ = _k_block(fd)
    return mp, tests


def matched_policy_display(mp: pd.DataFrame) -> pd.DataFrame:
    """``reward_faithfulness_matched_policy`` presentation (``[CI]`` strings)."""
    m = mp.copy()
    m["agr_K0 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(m["agr_K0"], m["K0_lo"], m["K0_hi"])]
    m["agr_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(m["agr_K5"], m["K5_lo"], m["K5_hi"])]
    m["delta_K0_minus_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(m["delta_K0_minus_K5"], m["d_lo"], m["d_hi"])]
    return m[["judge", "method", "cut", "n_turns", "agr_K0 [CI]", "pairs_K0", "agr_K5 [CI]", "pairs_K5", "delta_K0_minus_K5 [CI]"]]


def k_summary(data, scores_by_judge=None, **kw) -> pd.DataFrame:
    """SUMMARY of the K contrast in reward faithfulness (all n_turns bins pooled) per eval-side
    grader, method and iteration cut (paper ``reward_faithfulness_k_summary_long``). Columns:
    ``judge, method, cut, n_iters, train_iters, agr_K0 [K0_lo, K0_hi, pairs_K0], agr_K5 [...],
    delta_pooled/dp_lo/dp_hi`` ('delta pooled pairs' — every conversation pair weighted equally, so
    arms with more branch points at some iterations weigh those iterations more),
    ``delta_iter_mean/di_lo/di_hi`` (mean of the per-iteration K0−K5 deltas, equal weight per
    training iteration; CI from the same replicates), ``iters_K5_more_faithful,
    iters_K0_more_faithful, wilcoxon_over_iters_p`` (n_iters ≥ 5 only). Sign: + ⇒ K=0 higher."""
    fd = _as_data(data, scores_by_judge, **kw)
    return _k_block(fd)[2]


def k_summary_display(ks: pd.DataFrame) -> pd.DataFrame:
    """``reward_faithfulness_k_summary`` presentation."""
    k = ks.copy()
    k["agr_K0 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(k["agr_K0"], k["K0_lo"], k["K0_hi"])]
    k["agr_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(k["agr_K5"], k["K5_lo"], k["K5_hi"])]
    k["delta pooled pairs [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(k["delta_pooled"], k["dp_lo"], k["dp_hi"])]
    k["delta iter-mean [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(k["delta_iter_mean"], k["di_lo"], k["di_hi"])]
    return k[["judge", "method", "cut", "train_iters", "agr_K0 [CI]", "pairs_K0", "agr_K5 [CI]", "pairs_K5",
              "delta pooled pairs [CI]", "delta iter-mean [CI]", "iters_K5_more_faithful", "iters_K0_more_faithful",
              "wilcoxon_over_iters_p"]]


# ── 3) by cooperation level ──────────────────────────────────────────────────

def by_cooperation(data, scores_by_judge=None, *, judge: Optional[str] = None,
                   coop_order: Sequence[str] = COOP_ORDER, **kw) -> pd.DataFrame:
    """Reward faithfulness by patient cooperation level (paper ``reward_faithfulness_by_coop_long``;
    32 personas each: Cooperative = 'High', Warms up = 'StartLowAndChangesToHigh', Resistant =
    'Low'; persona attached via the eval conversation's file_index → persona_id). Pairs are formed
    WITHIN a cooperation stratum, so the statistic asks whether the training proxy ranks
    same-cooperation conversations like the full-conversation eval does (the easy between-strata
    ordering is removed). Eval side = ``judge`` (default the primary). Columns per (coop, arm):
    ``n_convs_mean, agreement, ci_lo, ci_hi, n_pairs`` (all n_turns + iterations pooled),
    ``agr_/lo_/hi_/pairs_<range>`` for the coarse ranges, ``agr_iter1/lo_iter1/hi_iter1/pairs_iter1``
    (the matched-policy train_iter 1 cut). The within-stratum K deltas (all iters + iter 1) are
    attached as ``.attrs["k_deltas"]`` (``{(coop, method): {all_iters, train_iter_1}}``). Each
    stratum's bootstrap is seeded ``seed + 1 + stratum_index`` (the paper's scheme)."""
    fd = _as_data(data, scores_by_judge, **kw)
    j = judge or fd.primary
    d0 = fd.DF[j]
    rows, kd = [], {}
    for ci, coop in enumerate(coop_order):
        sub = d0[d0["coop"] == coop][["arm", "eval_iter", "conversation_id", "n_turns", "proxy_score", "eval_score"]]
        abc = AgreementBoot(sub, B=fd.B, seed=fd.seed + 1 + ci)
        for arm in fd.arm_labels:
            r = abc.agg(abc.keys(arm=arm))
            row = dict(coop=coop, arm=arm, method=_method_of(arm), K=_k_of(arm),
                       n_convs_mean=r["n_convs_mean"], agreement=r["agreement"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n_pairs=r["n_pairs"])
            for lab, lo, hi in COARSE:
                rc = abc.agg(abc.keys(arm=arm, nt_lo=lo, nt_hi=hi))
                row[f"agr_{lab}"] = rc["agreement"]; row[f"lo_{lab}"] = rc["ci_lo"]; row[f"hi_{lab}"] = rc["ci_hi"]; row[f"pairs_{lab}"] = rc["n_pairs"]
            r1 = abc.agg(abc.keys(arm=arm, eval_iters={0}))
            row["agr_iter1"] = r1["agreement"]; row["lo_iter1"] = r1["ci_lo"]; row["hi_iter1"] = r1["ci_hi"]; row["pairs_iter1"] = r1["n_pairs"]
            rows.append(row)
        for m in fd.methods():
            r0, r5 = abc.agg(abc.keys(arm=f"{m}_LA0")), abc.agg(abc.keys(arm=f"{m}_LA5"))
            d = delta_ci(r0["boots"], r5["boots"], r0["agreement"], r5["agreement"])
            r0i, r5i = abc.agg(abc.keys(arm=f"{m}_LA0", eval_iters={0})), abc.agg(abc.keys(arm=f"{m}_LA5", eval_iters={0}))
            di = delta_ci(r0i["boots"], r5i["boots"], r0i["agreement"], r5i["agreement"])
            kd[(coop, m)] = dict(all_iters=dict(delta=d["delta"], lo=d["d_lo"], hi=d["d_hi"]),
                                 train_iter_1=dict(delta=di["delta"], lo=di["d_lo"], hi=di["d_hi"]))
    out = pd.DataFrame(rows)
    out.attrs["k_deltas"] = kd
    out.attrs["judge"] = j
    return out


def by_cooperation_display(coop: pd.DataFrame) -> pd.DataFrame:
    """``reward_faithfulness_by_coop`` presentation."""
    cm = coop.copy()
    cm["agreement [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(cm["agreement"], cm["ci_lo"], cm["ci_hi"])]
    for lab, _, _ in COARSE:
        cm[f"agr {lab} [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(cm[f"agr_{lab}"], cm[f"lo_{lab}"], cm[f"hi_{lab}"])]
    cm["agr train_iter_1 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(cm["agr_iter1"], cm["lo_iter1"], cm["hi_iter1"])]
    cm["n_convs_mean"] = cm["n_convs_mean"].round(1)
    return cm[["coop", "arm", "n_convs_mean", "n_pairs", "agreement [CI]"] + [f"agr {lab} [CI]" for lab, _, _ in COARSE]
              + ["agr train_iter_1 [CI]", "pairs_iter1"]]


# ── 4) proxy-vs-eval LEVELS ──────────────────────────────────────────────────

def proxy_levels(data, scores_by_judge=None, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Proxy-vs-eval LEVELS per arm × model state (paper ``reward_faithfulness_levels`` +
    ``_levels_rho``). Returns ``(levels, rho)``.

    ``levels``: ``proxy_mean`` = mean over that iteration's branch points of the CHOSEN (arg-max)
    candidate's training-oracle score (K=0: prefix+completion; K=5: the K-EXTENDED score) — the
    reward the update actually optimised, indexed by the policy that produced it (train_iter =
    eval_iter + 1; each arm's final model state — its last eval_iter, earlier for the
    censored arm — was evaluated but never trained on, hence NaN proxy). ``eval_mean<sfx>`` = mean full-conversation Q1Q2 of the same
    model state under each grader (primary unsuffixed, held-out ``_heldout``) — never averaged.
    ``gap_proxy_minus_eval<sfx>`` = proxy_mean − eval_mean (+ ⇒ the training reward reads higher
    than the full-conversation eval). ``mean_n_turns`` = mean prefix length of the branch points.
    Iteration 0 = two independent base draws per method (one per K arm).
    ``rho``: across-iteration association between the proxy LEVEL and each grader's eval LEVEL per
    arm — Spearman ρ, Pearson r over ``n_iters`` model states (train_iter 1..N), ``mean_gap``, and
    the range each level spans. Descriptive: n ≤ 10 points per arm, no multiplicity correction."""
    fd = _as_data(data, scores_by_judge, **kw)
    BR = fd.BR
    rows = []
    for arm in fd.arm_labels:
        b = BR[BR["arm"] == arm]
        evs = {j: fd.EV[j][fd.EV[j]["arm"] == arm].groupby("eval_iter").agg(
            mean=("eval_score", "mean"), std=("eval_score", "std"), count=("eval_score", "count"),
            model=("model", "first")) for j in fd.judges}
        eval_iters = sorted(set().union(*[set(e.index) for e in evs.values()]))
        for ei in eval_iters:
            bb = b[b["eval_iter"] == ei]
            model = next((str(evs[j].loc[ei, "model"]) for j in fd.judges if ei in evs[j].index), None)
            row = dict(arm=arm, method=_method_of(arm), K=_k_of(arm), eval_iter=int(ei), model=model,
                       train_iter=int(ei) + 1 if len(bb) else np.nan,
                       n_branch_points=int(len(bb)), n_convs_branched=int(bb["conversation_id"].nunique()) if len(bb) else 0,
                       mean_n_turns=float(bb["n_turns"].mean()) if len(bb) else np.nan,
                       proxy_mean=float(bb["proxy_score"].mean()) if len(bb) else np.nan,
                       proxy_sd=float(bb["proxy_score"].std(ddof=1)) if len(bb) > 1 else np.nan)
            for j in fd.judges:
                sfx = fd.suffix(j) or "_primary"
                e = evs[j]
                row[f"eval_mean{sfx}"] = float(e.loc[ei, "mean"]) if ei in e.index else np.nan
                if j == fd.primary:
                    row[f"eval_sd{sfx}"] = float(e.loc[ei, "std"]) if ei in e.index else np.nan
                row[f"n_eval{sfx}"] = int(e.loc[ei, "count"]) if ei in e.index else 0
            for j in fd.judges:
                sfx = fd.suffix(j) or "_primary"
                row[f"gap_proxy_minus_eval{sfx}"] = row["proxy_mean"] - row[f"eval_mean{sfx}"]
            rows.append(row)
    LV = pd.DataFrame(rows)
    rho_rows = []
    for arm in fd.arm_labels:
        d = LV[(LV["arm"] == arm) & LV["proxy_mean"].notna()]
        for j in fd.judges:
            col = f"eval_mean{fd.suffix(j) or '_primary'}"
            dd = d[[col, "proxy_mean"]].dropna()
            rho, p = (sps.spearmanr(dd["proxy_mean"], dd[col]) if len(dd) >= 3 else (np.nan, np.nan))
            pr, pp = (sps.pearsonr(dd["proxy_mean"], dd[col]) if len(dd) >= 3 else (np.nan, np.nan))
            rho_rows.append(dict(arm=arm, eval_grader=j, n_iters=len(dd), spearman_rho=float(rho), spearman_p=float(p),
                                 pearson_r=float(pr), pearson_p=float(pp),
                                 mean_gap=float((dd["proxy_mean"] - dd[col]).mean()) if len(dd) else np.nan,
                                 proxy_range=float(dd["proxy_mean"].max() - dd["proxy_mean"].min()) if len(dd) else np.nan,
                                 eval_range=float(dd[col].max() - dd[col].min()) if len(dd) else np.nan))
    return LV, pd.DataFrame(rho_rows)


# ── 5) ledger ────────────────────────────────────────────────────────────────

_num = json_scalar            # one definition — see eda_analysis/ledger.py


def _put(d: dict, key: str, value, *, source: str = "", note: str = "") -> None:
    d[key] = ledger_entry(value, source, note)


_r3 = round3


def faithfulness_numbers(fd: FaithfulnessData, *, curve: pd.DataFrame, by_iter: pd.DataFrame,
                         k_by_iter: pd.DataFrame, matched: pd.DataFrame, tests: pd.DataFrame,
                         summary: pd.DataFrame, coop: pd.DataFrame, levels: pd.DataFrame,
                         rho: pd.DataFrame, selfcheck_max_dev: Optional[float] = None) -> Dict[str, dict]:
    """The quotable-numbers ledger (the paper's ``analysis/out/reward_faithfulness.json``
    ``numbers`` block): ``{dotted.key: {"value", "source", "note"}}`` for ``exports.save_numbers``.
    Grader keys are ``primary`` / ``heldout`` (a single held-out grader) or the grader's label.
    Keys: ``curve.<grader>.<arm>.iters<range>.nt<bin>`` (headline bins), ``k_by_iter.…``,
    ``by_iter.<arm>.train_iter<n>``, ``k_summary.<grader>.<method>.<cut>``,
    ``matched_policy_test.…``, ``matched_policy.<grader>.<method>.<cut>.nt<bin>``,
    ``by_coop.primary.<coop>.<method>.delta_K0_minus_K5``, ``by_coop.primary.<coop>.<arm>``,
    ``levels.<arm>.eval_iter<n>``, ``levels_rho.<arm>.<grader>``, ``bootstrap``, ``caveats``,
    ``n_branch_rows``, ``n_branch_rows_joined_<grader>``, ``selfcheck.max_abs_dev_vs_eda_function``."""
    L: Dict[str, dict] = {}
    gk = {j: ("primary" if j == fd.primary else ("heldout" if len(fd.heldout) == 1 else j)) for j in fd.judges}
    if selfcheck_max_dev is not None:
        _put(L, "selfcheck.max_abs_dev_vs_eda_function", selfcheck_max_dev,
             source="stats.rank_agreement_by_nturns(BR, primary scores) vs faithfulness_curve point estimates")
    for _, s in curve.iterrows():
        if s["n_turns"] in ("12", "20", "30", "40", "12-20", "22-34", "36-50", "all"):
            _put(L, f"curve.{gk[s['judge']]}.{s['arm']}.iters{s['iters']}.nt{s['n_turns']}",
                 dict(agreement=_r3(s["agreement"]), ci_lo=_r3(s["ci_lo"]), ci_hi=_r3(s["ci_hi"]), n_pairs=int(s["n_pairs"])),
                 source=f"tables/faithfulness_curve_long.md row judge={s['judge']}, arm={s['arm']}, iters={s['iters']}, n_turns={s['n_turns']}")
    for _, r in k_by_iter.iterrows():
        _put(L, f"k_by_iter.{gk[r['judge']]}.{r['method']}.train_iter{int(r['train_iter'])}",
             dict(agr_K0=_r3(r["agr_K0"]), agr_K5=_r3(r["agr_K5"]), delta_K0_minus_K5=_r3(r["delta_K0_minus_K5"]),
                  d_lo=_r3(r["d_lo"]), d_hi=_r3(r["d_hi"])),
             source=f"tables/faithfulness_k_by_iter.md row judge={r['judge']}, method={r['method']}, train_iter={int(r['train_iter'])}")
    for _, r in by_iter.iterrows():
        v = dict(agreement=_r3(r["agreement"]), ci_lo=_r3(r["ci_lo"]), ci_hi=_r3(r["ci_hi"]),
                 n_pairs=int(r["n_pairs"]), n_convs=int(r["n_convs"]))
        for j in fd.heldout:
            v[f"agreement{fd.suffix(j)}"] = _r3(r.get(f"agreement{fd.suffix(j)}"))
        _put(L, f"by_iter.{r['arm']}.train_iter{int(r['train_iter'])}", v,
             source=f"tables/faithfulness_curve_by_iter.md row arm={r['arm']}, train_iter={int(r['train_iter'])}")
    for _, r in summary.iterrows():
        _put(L, f"k_summary.{gk[r['judge']]}.{r['method']}.{r['cut']}",
             dict(train_iters=r["train_iters"], agr_K0=_r3(r["agr_K0"]), agr_K5=_r3(r["agr_K5"]),
                  delta_pooled=_r3(r["delta_pooled"]), dp_lo=_r3(r["dp_lo"]), dp_hi=_r3(r["dp_hi"]),
                  delta_iter_mean=_r3(r["delta_iter_mean"]), di_lo=_r3(r["di_lo"]), di_hi=_r3(r["di_hi"]),
                  n_iters=int(r["n_iters"]), iters_K5_more_faithful=int(r["iters_K5_more_faithful"]),
                  wilcoxon_over_iters_p=_r3(r["wilcoxon_over_iters_p"], 4),
                  pairs_K0=int(r["pairs_K0"]), pairs_K5=int(r["pairs_K5"])),
             source=f"tables/faithfulness_k_summary.md row judge={r['judge']}, method={r['method']}, cut={r['cut']}")
    for _, r in tests.iterrows():
        _put(L, f"matched_policy_test.{gk[r['judge']]}.{r['method']}.{r['cut']}",
             dict(n_bins=int(r["n_bins"]), bins_K5_more_faithful=int(r["bins_K5_more_faithful"]),
                  mean_delta=_r3(r["mean_delta"]), median_delta=_r3(r["median_delta"]), wilcoxon_p=_r3(r["wilcoxon_p"], 4)),
             source=f"tables/faithfulness_matched_policy_tests.md row judge={r['judge']}, method={r['method']}, cut={r['cut']}")
    for _, r in matched.iterrows():
        if r["n_turns"] in ("all", "12", "12-20", "22-34", "36-50"):
            _put(L, f"matched_policy.{gk[r['judge']]}.{r['method']}.{r['cut']}.nt{r['n_turns']}",
                 dict(agr_K0=_r3(r["agr_K0"]), agr_K5=_r3(r["agr_K5"]), delta_K0_minus_K5=_r3(r["delta_K0_minus_K5"]),
                      d_lo=_r3(r["d_lo"]), d_hi=_r3(r["d_hi"]), pairs_K0=int(r["pairs_K0"]), pairs_K5=int(r["pairs_K5"])),
                 source=f"tables/faithfulness_matched_policy.md row judge={r['judge']}, method={r['method']}, cut={r['cut']}, n_turns={r['n_turns']}")
    cj = gk.get(coop.attrs.get("judge", fd.primary), "primary")
    for (cp, m), d in coop.attrs.get("k_deltas", {}).items():
        _put(L, f"by_coop.{cj}.{cp}.{m}.delta_K0_minus_K5",
             dict(all_iters=dict(delta=_r3(d["all_iters"]["delta"]), lo=_r3(d["all_iters"]["lo"]), hi=_r3(d["all_iters"]["hi"])),
                  train_iter_1=dict(delta=_r3(d["train_iter_1"]["delta"]), lo=_r3(d["train_iter_1"]["lo"]), hi=_r3(d["train_iter_1"]["hi"]))),
             source=f"tables/faithfulness_by_coop.md rows coop={cp}, arms {m}_LA0/{m}_LA5 (delta computed from the same replicates)")
    for _, r in coop.iterrows():
        _put(L, f"by_coop.{cj}.{r['coop']}.{r['arm']}",
             dict(agreement=_r3(r["agreement"]), ci_lo=_r3(r["ci_lo"]), ci_hi=_r3(r["ci_hi"]),
                  n_pairs=int(r["n_pairs"]), agr_train_iter_1=_r3(r["agr_iter1"])),
             source=f"tables/faithfulness_by_coop.md row coop={r['coop']}, arm={r['arm']}")
    for _, r in levels.iterrows():
        if pd.isna(r["proxy_mean"]):
            continue
        v = dict(proxy_mean=_r3(r["proxy_mean"]), eval_mean_primary=_r3(r["eval_mean_primary"]),
                 gap_primary=_r3(r["gap_proxy_minus_eval_primary"]), n_branch_points=int(r["n_branch_points"]),
                 mean_n_turns=_r3(r["mean_n_turns"], 2))
        for j in fd.heldout:
            v[f"eval_mean{fd.suffix(j)}"] = _r3(r.get(f"eval_mean{fd.suffix(j)}"))
        _put(L, f"levels.{r['arm']}.eval_iter{int(r['eval_iter'])}", v,
             source=f"tables/faithfulness_levels.md row arm={r['arm']}, eval_iter={int(r['eval_iter'])}")
    for _, r in rho.iterrows():
        _put(L, f"levels_rho.{r['arm']}.{gk[r['eval_grader']]}",
             dict(spearman_rho=_r3(r["spearman_rho"]), spearman_p=_r3(r["spearman_p"], 4),
                  pearson_r=_r3(r["pearson_r"]), n_iters=int(r["n_iters"]), mean_gap=_r3(r["mean_gap"])),
             source=f"tables/faithfulness_levels_rho.md row arm={r['arm']}, eval_grader={r['eval_grader']}")
    _put(L, "bootstrap", dict(B=fd.B, seed=fd.seed, cluster="conversation within (arm, eval_iter)"), source="module constants")
    _put(L, "caveats", list(CAVEATS), source="module docstring")
    _put(L, "n_branch_rows", int(len(fd.BR)), source="training.load_branch_reliability(which='chosen') row count")
    for j in fd.judges:
        _put(L, f"n_branch_rows_joined_{gk[j]}", int(len(fd.DF[j])),
             source=f"inner join to {j} eval Q1Q2 on (arm, eval_iter, conversation_id)")
    _put(L, "meta", dict(metric=fd.metric, arms=list(fd.arm_labels), judges=list(fd.judges), primary=fd.primary,
                         promoted_from="papers/2026_lookahead_pto_grpo/analysis/reward_faithfulness.py (2026-08-18)"),
         source="module constants")
    return L
